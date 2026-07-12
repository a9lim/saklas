"""CPU-only contract tests for the first-class live SAE runtime."""
from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from saklas.core.sae import MockSaeBackend, sae_device_str, select_runtime_layer
from saklas.core.session import SaklasSession
from saklas.core.steering_composer import SteeringComposer
from saklas.core.steering_expr import parse_expr
from saklas.core.steering_expr import format_expr


class _Capture:
    def __init__(self, row: torch.Tensor, layer: int = 1) -> None:
        self.layer = layer
        self.row = row

    def per_layer_buckets(self) -> dict[int, list[torch.Tensor]]:
        return {self.layer: [self.row]}

    def latest_per_layer(self) -> dict[int, torch.Tensor]:
        return {self.layer: self.row}

    def tail_slice_at(self, _index: int) -> dict[int, torch.Tensor]:
        return {self.layer: self.row}

    def stacked(self) -> dict[int, torch.Tensor]:
        return {self.layer: self.row.unsqueeze(0)}


def _session() -> SaklasSession:
    session: Any = SaklasSession.__new__(SaklasSession)
    session._layers = [object(), object(), object(), object()]
    session._device = torch.device("cpu")
    session._dtype = torch.float32
    session._sae_backend = MockSaeBackend(
        layers=frozenset({1}), d_model=4, d_feature=4,
    )
    session._sae_layer = 1
    session._sae_width = 4
    session._sae_feature_meta = {
        "2": {"label": "feature two", "max_act": 10.0, "checked": True},
    }
    session._live_sae = None
    session._sae_probes = {}
    session._sae_step_stash = None
    session._last_sae_step_readings = None
    session._profiles = {}
    session._probe_hash_cache = {}
    session._analytics_cpu_cache = {}
    session._readout_long_tensor_cache = {}
    session._capture = _Capture(torch.tensor([0.2, 3.0, 5.0, 1.0]))
    session._live_sae_active_for_generation = True
    session._invalidate_prefix_cache = lambda: None  # type: ignore[method-assign]
    session._invalidate_analytics_cache = lambda: None  # type: ignore[method-assign]
    return cast(SaklasSession, session)


def test_sae_device_str_strips_mps_index() -> None:
    # safetensors rejects the indexed MPS form a live model reports
    # ("device mps:0 is invalid"); CUDA indices must survive.
    assert sae_device_str(torch.device("mps", 0)) == "mps"
    assert sae_device_str("mps:0") == "mps"
    assert sae_device_str("mps") == "mps"
    assert sae_device_str("cuda:1") == "cuda:1"
    assert sae_device_str("cpu") == "cpu"


def test_select_runtime_layer_prefers_workspace_near_65_percent() -> None:
    assert select_runtime_layer({1, 8, 14, 18}, 24) == 14
    assert select_runtime_layer({1, 8, 14, 18}, 24, requested=8) == 8
    with pytest.raises(ValueError, match="does not cover"):
        select_runtime_layer({1, 8}, 24, requested=9)


def test_sae_encoder_readout_is_detached_from_autograd() -> None:
    session = _session()
    acts = session._encode_sae_hidden(torch.ones(4, requires_grad=True))
    assert not acts.requires_grad


def test_sae_feature_validation_and_decoder_row_registration() -> None:
    session = _session()
    assert session.validate_sae_feature(2) == {
        "id": 2, "label": "feature two", "layer": 1, "max_act": 10.0,
    }
    name = session.register_sae_direction(2)
    assert name == "sae/2"
    assert torch.equal(session._profiles[name][1], torch.tensor([0.0, 0.0, 1.0, 0.0]))
    with pytest.raises(ValueError, match="out of range"):
        session.validate_sae_feature(9)


def test_live_sae_readout_and_probe_share_one_encoder_result() -> None:
    session = _session()
    session._sae_probes["sae/2"] = {
        "feature_id": 2, "layer": 1, "label": "feature two", "max_act": 10.0,
    }
    assert session.enable_live_sae(top_k=3) == {"layer": 1, "top_k": 3}
    readout = session._live_sae_readout_step()
    assert readout == [
        (2, 5.0, "feature two", 10.0),
        (1, 3.0, None, None),
        (3, 1.0, None, None),
    ]
    # The probe channel is normalized strength — activation / maxActApprox —
    # while the readout row keeps the raw activation beside the unit.
    reading = session._last_sae_step_readings["sae/2"]  # type: ignore[index]
    assert reading.coords == (0.5,)
    assert reading.coords_per_layer == {1: (0.5,)}


def test_live_sae_readout_seeds_topk_raw_values_for_pinned_probes() -> None:
    session = _session()
    session._sae_probes["sae/2"] = {
        "feature_id": 2, "layer": 1, "label": "feature two", "max_act": 10.0,
    }
    session.enable_live_sae(top_k=3)
    seen_raw: dict[int, float] = {}
    original = session._sae_probe_values

    def spy_probe_values(
        activations: torch.Tensor,
        *,
        only: set[str] | None = None,
        raw_by_fid: dict[int, float] | None = None,
    ) -> list[tuple[str, int, float, float]]:
        seen_raw.update(raw_by_fid or {})
        return original(
            activations, only=only, raw_by_fid=raw_by_fid,
        )

    session._sae_probe_values = spy_probe_values  # type: ignore[method-assign]

    session._live_sae_readout_step()

    assert seen_raw[2] == pytest.approx(5.0)


def test_sae_probe_without_metadata_reads_raw_activation() -> None:
    session = _session()
    session._sae_probes["sae/1"] = {
        "feature_id": 1, "layer": 1, "label": None, "max_act": None,
    }
    session.enable_live_sae(top_k=1)
    session._live_sae_readout_step()
    reading = session._last_sae_step_readings["sae/1"]  # type: ignore[index]
    assert reading.coords == (3.0,)


def test_sae_gate_scalar_stashes_activations_for_live_step() -> None:
    session = _session()
    session._sae_probes["sae/1"] = {
        "feature_id": 1, "layer": 1, "label": None, "max_act": None,
    }
    scalars = session._score_sae_gate_scalars()
    assert scalars["sae/1"] == pytest.approx(3.0)
    assert scalars["sae/1[0]"] == pytest.approx(3.0)
    assert scalars["sae/1:fraction"] == pytest.approx(0.0)
    assert scalars["sae/1:membership"] == pytest.approx(1.0)
    assert session._sae_step_stash is not None
    assert session._sae_step_stash["fresh"] is True


def test_sae_gate_scalar_and_live_step_share_one_encoder_result() -> None:
    session = _session()
    session._sae_probes["sae/2"] = {
        "feature_id": 2, "layer": 1, "label": "feature two", "max_act": 10.0,
    }
    session.enable_live_sae(top_k=3)
    calls = 0
    original = session._encode_sae_hidden

    def counting_encode(hidden: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return original(hidden)

    session._encode_sae_hidden = counting_encode  # type: ignore[method-assign]

    scalars = session._score_sae_gate_scalars({"sae/2"})
    readout = session._live_sae_readout_step()

    assert calls == 1
    assert scalars["sae/2"] == pytest.approx(0.5)
    assert readout is not None
    assert session._last_sae_step_readings is not None
    assert session._last_sae_step_readings["sae/2"].coords == pytest.approx((0.5,))


def test_sae_gate_raw_values_seed_live_probe_reads_outside_topk() -> None:
    session = _session()
    session._sae_probes["sae/0"] = {
        "feature_id": 0, "layer": 1, "label": None, "max_act": None,
    }
    session.enable_live_sae(top_k=1)  # feature 2 wins; gated feature 0 is outside top-k.
    seen_raw: dict[int, float] = {}
    original = session._sae_probe_values

    def spy_probe_values(
        activations: torch.Tensor,
        *,
        only: set[str] | None = None,
        raw_by_fid: dict[int, float] | None = None,
    ) -> list[tuple[str, int, float, float]]:
        del only
        seen_raw.update(raw_by_fid or {})
        return original(activations, raw_by_fid=raw_by_fid)

    session._sae_probe_values = spy_probe_values  # type: ignore[method-assign]

    scalars = session._score_sae_gate_scalars({"sae/0"})
    readout = session._live_sae_readout_step()

    assert scalars["sae/0"] == pytest.approx(0.2)
    assert readout is not None
    assert seen_raw[0] == pytest.approx(0.2)
    assert session._last_sae_step_readings is not None
    assert session._last_sae_step_readings["sae/0"].coords == pytest.approx((0.2,))


def test_sae_probe_values_reuse_feature_selector_tensor() -> None:
    session = _session()
    session._sae_probes["sae/1"] = {
        "feature_id": 1, "layer": 1, "label": None, "max_act": None,
    }
    session._sae_probes["sae/2"] = {
        "feature_id": 2, "layer": 1, "label": "feature two", "max_act": 10.0,
    }
    latest = session._capture.latest_per_layer()
    acts = session._encode_sae_hidden(latest[1])

    first = session._score_sae_probes(activations=acts)
    first_ids = {
        key: id(value)
        for key, value in session._readout_long_tensor_cache.items()
    }
    second = session._score_sae_probes(activations=acts)

    assert first["sae/1"].coords == second["sae/1"].coords
    assert first["sae/2"].coords == second["sae/2"].coords
    assert first_ids
    assert {
        key: id(value)
        for key, value in session._readout_long_tensor_cache.items()
    } == first_ids


def test_sae_gate_scalar_is_normalized_when_metadata_known() -> None:
    session = _session()
    session._sae_probes["sae/2"] = {
        "feature_id": 2, "layer": 1, "label": "feature two", "max_act": 10.0,
    }
    scalars = session._score_sae_gate_scalars()
    assert scalars["sae/2"] == pytest.approx(0.5)


def test_sae_gate_scalar_scores_only_referenced_probe() -> None:
    session = _session()
    session._sae_probes["sae/1"] = {
        "feature_id": 1, "layer": 1, "label": None, "max_act": None,
    }
    session._sae_probes["sae/2"] = {
        "feature_id": 2, "layer": 1, "label": "feature two", "max_act": 10.0,
    }

    scalars = session._score_sae_gate_scalars({"sae/1"})

    assert scalars["sae/1"] == pytest.approx(3.0)
    assert scalars["sae/1[0]"] == pytest.approx(3.0)
    assert "sae/2" not in scalars


def test_composer_detects_attached_sae_gate() -> None:
    stub = SimpleNamespace(
        _sae_probes={"sae/2": {"feature_id": 2}},
        _lens_probes={},
        _monitor=SimpleNamespace(probe_names=()),
    )
    composer = SteeringComposer(stub)  # type: ignore[arg-type]
    steering = parse_expr("0.3 sae/2@when:sae/2>3")
    composer._stack.append(dict(steering.alphas))  # type: ignore[arg-type]
    assert composer.gated_sae_probe_keys() == {"sae/2"}


@pytest.mark.parametrize(
    "expr",
    [
        "0.3 sae/9143",
        "!sae/9143",
        "0.3 sae/9143@when:sae/9143>3",
    ],
)
def test_sae_integer_atom_grammar_roundtrips(expr: str) -> None:
    parsed = parse_expr(expr)
    assert format_expr(parsed) == expr


def test_sae_runtime_metadata_roundtrip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    from saklas.io.sae import load_sae_metadata, save_sae_metadata

    path = save_sae_metadata("org/model", "release/name", {
        "layer": 14, "width": 16_384, "revision": "main",
        "fingerprint": "abc", "sae_id": "layer_14", "repo_id": "org/sae",
        "neuronpedia_id": None,
    })
    assert path.exists()
    assert load_sae_metadata("org/model", "release/name") == {
        "format_version": 3,
        "model_id": "org/model",
        "release": "release/name",
        "layer": 14,
        "width": 16_384,
        "revision": "main",
        "fingerprint": "abc",
        "sae_id": "layer_14",
        "repo_id": "org/sae",
        "neuronpedia_id": None,
    }


def test_sae_feature_meta_roundtrip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    from saklas.io.sae import (
        load_sae_feature_meta,
        save_sae_feature_meta,
    )

    assert load_sae_feature_meta("org/model", "rel") == {}
    save_sae_feature_meta("org/model", "rel", cast("dict[str, dict[str, Any]]", {
        "7": {"label": "days of the week", "max_act": 121.11, "checked": True},
        "9": {"label": None, "max_act": None},
    }))
    meta = load_sae_feature_meta("org/model", "rel")
    assert meta["7"] == {"label": "days of the week", "max_act": 121.11}
    assert meta["9"] == {"label": None, "max_act": None}


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.update(format_version=1),
        lambda payload: payload.update(model_id="other/model"),
        lambda payload: payload.update(release="other-release"),
        lambda payload: payload.update(extra=True),
        lambda payload: payload["features"]["7"].update(checked=True),
        lambda payload: payload["features"].update({"8": {"label": None}}),
    ],
)
def test_sae_feature_meta_rejects_non_current_shapes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutation: Any,
) -> None:
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    import json

    from saklas.io.sae import (
        SAE_RUNTIME_FORMAT_VERSION,
        load_sae_feature_meta,
        sae_features_path,
    )

    path = sae_features_path("org/model", "rel")
    path.parent.mkdir(parents=True)
    payload = {
        "format_version": SAE_RUNTIME_FORMAT_VERSION,
        "model_id": "org/model",
        "release": "rel",
        "features": {"7": {"label": "weekdays", "max_act": 3.0}},
    }
    mutation(payload)
    path.write_text(json.dumps(payload))
    assert load_sae_feature_meta("org/model", "rel") == {}


def test_stream_aggregate_keeps_lens_and_sae_probe_readings() -> None:
    # ``probe_reading_aggregate`` filters result readings to attached probes;
    # lens/SAE probes live on their own session registries (readout channels,
    # not the Monitor), so the filter must union all three rosters — it used
    # to drop their end-of-gen aggregates from every streaming done frame.
    from saklas.core.results import ProbeReading
    from saklas.server.streaming import probe_reading_aggregate

    readings = {
        "confident.uncertain": ProbeReading(0.1, [], coords=(0.3,)),
        "jlens/fake": ProbeReading(0.0, [], coords=(0.02,)),
        "sae/548": ProbeReading(0.0, [], coords=(0.84,)),
        "sae/999": ProbeReading(0.0, [], coords=(0.5,)),  # detached — dropped
    }
    session = SimpleNamespace(
        monitor=SimpleNamespace(probe_names=("confident.uncertain",)),
        lens_probe_names=["jlens/fake"],
        sae_probe_names=["sae/548"],
    )
    result = SimpleNamespace(probe_readings=readings)
    out = probe_reading_aggregate(session, result)  # type: ignore[arg-type]
    assert set(out) == {"confident.uncertain", "jlens/fake", "sae/548"}
    assert out["sae/548"]["coords"] == [0.84]


def test_fetch_sae_feature_meta_batch_caches_and_updates_probes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    session = _session()
    session._model_info = {"model_id": "org/model"}
    session._sae_probes["sae/1"] = {
        "feature_id": 1, "layer": 1, "label": None, "max_act": None,
    }
    session._probe_hash_cache["sae/1"] = "stale"
    fetched: list[int] = []

    def fake_fetch(idx: int) -> dict[str, object] | None:
        fetched.append(idx)
        if idx == 3:
            return None  # network miss — retryable, not cached
        return {"label": f"feature {idx}", "max_act": 4.0, "checked": True}

    session._fetch_neuronpedia_feature = fake_fetch  # type: ignore[method-assign]
    out = session.fetch_sae_feature_meta([1, 1, 2, 3, 99])
    # id 2 was already cached (max_act present) — no refetch; 99 is out of
    # range; 3 failed and stays absent so a later call retries it.
    assert sorted(fetched) == [1, 3]
    assert out["1"] == {"label": "feature 1", "max_act": 4.0}
    assert out["2"] == {"label": "feature two", "max_act": 10.0}
    assert "3" not in out and "99" not in out
    # The attached probe's spec + hash cache reflect the new unit.
    assert session._sae_probes["sae/1"]["max_act"] == 4.0
    assert "sae/1" not in session._probe_hash_cache
    # Persisted — a fresh load sees the merged cache.
    from saklas.io.sae import load_sae_feature_meta

    on_disk = load_sae_feature_meta("org/model", "mock-release")
    assert on_disk["1"]["max_act"] == 4.0


@pytest.mark.parametrize("max_act", [float("nan"), float("inf")])
def test_sae_feature_meta_rejects_nonfinite_scale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, max_act: float,
) -> None:
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    from saklas.io.sae import save_sae_feature_meta

    with pytest.raises(ValueError, match="invalid SAE feature metadata"):
        save_sae_feature_meta("org/model", "release", {
            "1": {"label": "feature", "max_act": max_act},
        })


@pytest.mark.parametrize("feature_id", ["", "01", "-1", "feature"])
def test_sae_feature_meta_rejects_noncanonical_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, feature_id: str,
) -> None:
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    from saklas.io.sae import save_sae_feature_meta

    with pytest.raises(ValueError, match="feature id"):
        save_sae_feature_meta("org/model", "release", {
            feature_id: {"label": "feature", "max_act": 1.0},
        })
