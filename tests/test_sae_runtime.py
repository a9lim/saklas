"""CPU-only contract tests for the first-class live SAE runtime."""
from __future__ import annotations

from contextlib import nullcontext
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


def test_failed_provider_binding_does_not_half_adopt_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from saklas.core import sae as sae_module
    from saklas.io import sae as sae_io

    session: Any = SaklasSession.__new__(SaklasSession)
    previous = MockSaeBackend(
        layers=frozenset({0}), d_model=4, release="previous",
    )
    replacement = MockSaeBackend(
        layers=frozenset({1}), d_model=4, release="replacement",
        revision="commit", repo_id="org/sae",
    )
    session._device = torch.device("cpu")
    session._dtype = torch.float32
    session._layers = [object(), object()]
    session._model_info = {"model_id": "org/model", "hidden_dim": 4}
    session._sae_backend = previous
    session._sae_layer = 0
    session._sae_width = 4
    session._sae_feature_meta = {"1": {"label": "old", "max_act": 1.0}}
    session._model_exclusive = lambda *_args, **_kwargs: nullcontext()

    monkeypatch.setattr(sae_module, "load_sae_backend", lambda *_a, **_kw: replacement)
    monkeypatch.setattr(sae_io, "load_sae_feature_meta", lambda *_a: {})
    monkeypatch.setattr(
        sae_io, "save_sae_metadata",
        lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("disk full")),
    )

    with pytest.raises(RuntimeError, match="disk full"):
        session.load_sae("replacement", layer=1)

    assert session._sae_backend is previous
    assert session._sae_layer == 0
    assert session._sae_width == 4
    assert session._sae_feature_meta == {
        "1": {"label": "old", "max_act": 1.0},
    }


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
    original = session._sae_instrument.probe_values

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

    session._sae_instrument.probe_values = spy_probe_values  # type: ignore[method-assign]

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
    scalars = session._score_sae_gate_scalars(step_id=6)
    assert scalars["sae/1"] == pytest.approx(3.0)
    assert scalars["sae/1[0]"] == pytest.approx(3.0)
    # The fake geometry constants are gone (5.x): an SAE probe emits only
    # its real strength channel; unsupported channels are a preflight error,
    # never a silently-constant comparison.
    assert "sae/1:fraction" not in scalars
    assert "sae/1:membership" not in scalars
    assert session._sae_step_stash is not None
    assert session._sae_step_stash["step"] == 6


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

    scalars = session._score_sae_gate_scalars({"sae/2"}, step_id=2)
    readout = session._live_sae_readout_step(step_id=2)

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
    original = session._sae_instrument.probe_values

    def spy_probe_values(
        activations: torch.Tensor,
        *,
        only: set[str] | None = None,
        raw_by_fid: dict[int, float] | None = None,
    ) -> list[tuple[str, int, float, float]]:
        del only
        seen_raw.update(raw_by_fid or {})
        return original(activations, raw_by_fid=raw_by_fid)

    session._sae_instrument.probe_values = spy_probe_values  # type: ignore[method-assign]

    scalars = session._score_sae_gate_scalars({"sae/0"}, step_id=9)
    readout = session._live_sae_readout_step(step_id=9)

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


def test_unsupported_sae_gate_channel_raises_at_preflight() -> None:
    """A gate on a channel the SAE family can never produce is a preflight
    error (the 5.x replacement for the silently-constant fake channels)."""
    from saklas.core.errors import UnsupportedProbeChannelError
    from saklas.core.instruments.sae import SaeInstrument

    stub = SimpleNamespace(
        _lens_probes={},
        _monitor=SimpleNamespace(probe_names=()),
    )
    instrument = SaeInstrument(stub)  # type: ignore[arg-type]
    instrument.probes["sae/2"] = {
        "feature_id": 2, "layer": 1, "label": None, "max_act": None,
    }
    stub._sae_instrument = instrument
    stub._sae_probes = instrument.probes
    composer = SteeringComposer(stub)  # type: ignore[arg-type]
    steering = parse_expr("0.3 sae/2@when:sae/2:membership>0.5")
    composer._stack.append(dict(steering.alphas))  # type: ignore[arg-type]
    with pytest.raises(UnsupportedProbeChannelError) as exc_info:
        composer.gated_sae_probe_keys()
    status, text = exc_info.value.user_message()
    assert status == 400
    assert "sae/2:membership" in text
    # A supported-channel gate on the same probe still composes.
    composer._stack.clear()
    ok = parse_expr("0.3 sae/2@when:sae/2>3")
    composer._stack.append(dict(ok.alphas))  # type: ignore[arg-type]
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


def test_sae_registry_lock_snapshots_idle_reads() -> None:
    """round-6 P2: the idle-passthrough spec source is a per-call
    coherent snapshot under the SAE registry lock — handing out the live
    dict let one idle read RuntimeError mid-iteration under the
    un-locked detach, and the metadata backfill could rewrite a unit
    mid-read.  The session's ``remove_probe`` SAE branch routes through
    the atomic ``try_detach`` and blocks while the lock is held."""
    import threading

    session = _session()
    inst = session._sae_instrument
    session._sae_probes["sae/1"] = {
        "feature_id": 1, "layer": 1, "label": None, "max_act": None,
    }
    session._probe_hash_cache = {}

    # Idle _measurement_specs is a per-call copy, not the live dict.
    snapshot = inst._measurement_specs()
    assert snapshot is not inst.probes
    assert inst.try_detach("sae/1") is True
    assert "sae/1" in snapshot  # the copy is immune to the detach
    assert inst.try_detach("sae/1") is False

    # And the removal path serializes on the registry lock.
    session._sae_probes["sae/1"] = {
        "feature_id": 1, "layer": 1, "label": None, "max_act": None,
    }
    entered = threading.Event()

    def _detach() -> None:
        entered.set()
        SaklasSession.remove_probe(cast(Any, session), "sae/1")

    with inst.state_lock:
        detacher = threading.Thread(target=_detach)
        detacher.start()
        assert entered.wait(timeout=5.0)
        detacher.join(timeout=0.2)
        assert detacher.is_alive()  # blocked on the registry lock
        assert "sae/1" in inst.probes
    detacher.join(timeout=5.0)
    assert not detacher.is_alive()
    assert "sae/1" not in inst.probes


def test_bound_run_freezes_sae_unit_against_metadata_backfill() -> None:
    """The InstrumentBinding snapshot: a Neuronpedia backfill landing
    mid-generation (it mutates attached specs + the meta cache WITHOUT the
    generation lock) must not change a running generation's strength unit.
    Between generations (idle run) the refresh applies immediately, as
    before."""
    from saklas.core.instruments.types import ReadRequest

    session = _session()
    inst = session._sae_instrument
    # Attached before any metadata exists: the unit is raw at bind time.
    session._sae_probes["sae/1"] = {
        "feature_id": 1, "layer": 1, "label": None, "max_act": None,
    }
    acts = session._encode_sae_hidden(
        session._capture.latest_per_layer()[1]
    )
    prep = inst.prepare(ReadRequest(final_aggregate=True))
    inst.bind(inst.plan(prep), prep)
    before = {n: v for n, _f, _r, v in inst.probe_values(acts)}
    assert before["sae/1"] == pytest.approx(3.0)  # raw

    # The backfill lands mid-generation.
    session._sae_feature_meta["1"] = {
        "label": "one", "max_act": 4.0, "checked": True,
    }
    session._refresh_sae_probe_meta({"1": {"label": "one", "max_act": 4.0}})
    during = {n: v for n, _f, _r, v in inst.probe_values(acts)}
    assert during["sae/1"] == pytest.approx(3.0)  # unit frozen at bind

    inst.close_run()  # generation boundary
    after = {n: v for n, _f, _r, v in inst.probe_values(acts)}
    assert after["sae/1"] == pytest.approx(3.0 / 4.0)  # refresh now applies


def test_sae_bind_resolves_unit_from_meta_cache() -> None:
    """Bind-time resolution: a spec whose ``max_act`` is unset but whose
    unit exists in the metadata cache freezes the RESOLVED unit — the
    live-cache fallback never runs under a bound run, so a mid-generation
    cache mutation cannot flip the unit either."""
    from saklas.core.instruments.types import ReadRequest

    session = _session()
    inst = session._sae_instrument
    # Feature 2's unit (10.0) is already in the fixture's meta cache; the
    # spec itself carries no max_act (validate-time fetch never ran).
    session._sae_probes["sae/2"] = {
        "feature_id": 2, "layer": 1, "label": None, "max_act": None,
    }
    acts = session._encode_sae_hidden(
        session._capture.latest_per_layer()[1]
    )
    prep = inst.prepare(ReadRequest(final_aggregate=True))
    inst.bind(inst.plan(prep), prep)
    assert inst.current_run.binding.specs["sae/2"]["max_act"] == 10.0
    values = {n: v for n, _f, _r, v in inst.probe_values(acts)}
    assert values["sae/2"] == pytest.approx(5.0 / 10.0)

    # A mid-generation cache mutation is invisible to the bound run.
    session._sae_feature_meta["2"]["max_act"] = 2.0
    values = {n: v for n, _f, _r, v in inst.probe_values(acts)}
    assert values["sae/2"] == pytest.approx(5.0 / 10.0)


def test_sae_gate_keys_none_vs_empty_contract() -> None:
    """``gate_keys=None`` scores the full roster; an explicit ``set()``
    scores nothing.  The SAE member of the three-family contract pin."""
    session = _session()
    session._sae_probes["sae/2"] = {
        "feature_id": 2, "layer": 1, "label": "feature two", "max_act": 10.0,
    }
    inst = session._sae_instrument
    full = inst.gate_scalars(None, step_id=0)
    assert "sae/2" in full
    assert inst.gate_scalars(set(), step_id=0) == {}


def test_sae_negative_step_observe_never_caches() -> None:
    """``step_id < 0`` never populates the SAE run's observe memo —
    repeated negative observations rescore (the family-parameterized pin
    of the shared negative-step fix)."""
    from saklas.core.instruments.types import ReadRequest

    session = _session()
    inst = session._sae_instrument
    session._sae_probes["sae/2"] = {
        "feature_id": 2, "layer": 1, "label": None, "max_act": 10.0,
    }
    prep = inst.prepare(ReadRequest(final_aggregate=True))
    run = inst.bind(inst.plan(prep), prep)

    calls: list[int] = []

    def _fresh_probes(*_a: Any, **_kw: Any) -> dict[str, Any]:
        calls.append(1)
        return {"sae/2": object()}

    inst.score_probes = _fresh_probes  # type: ignore[method-assign]
    first = run.observe(-1, {})
    second = run.observe(-1, {})
    assert len(calls) == 2 and first is not second
    assert run._memo_step is None
    inst.close_run()


def test_detach_during_bound_generation_keeps_aggregate_roster() -> None:
    """Mutations apply next generation: a probe detached mid-generation
    (e.g. the synchronous DELETE route, which takes no generation lock)
    stays in the bound generation's aggregate roster; the next bind sees
    the removal."""
    from saklas.core.instruments.types import ReadRequest

    session = _session()
    inst = session._sae_instrument
    session._sae_probes["sae/1"] = {
        "feature_id": 1, "layer": 1, "label": None, "max_act": None,
    }
    prep = inst.prepare(ReadRequest(final_aggregate=True))
    inst.bind(inst.plan(prep), prep)
    pooled = session._capture.latest_per_layer()

    del inst.probes["sae/1"]  # the un-locked detach lands mid-generation

    readings = inst.score_aggregate([1], pooled=pooled)
    assert "sae/1" in readings  # frozen roster still measures it

    inst.close_run()
    readings = inst.score_aggregate([1], pooled=pooled)
    assert readings == {}  # the removal applies at the next boundary


def test_idle_observe_never_memoizes_stale_readings() -> None:
    """The idle passthrough run persists indefinitely, so ``observe`` must
    not key a memo on ``step_id`` alone — a repeated step with different
    hidden states returns fresh readings."""
    session = _session()
    inst = session._sae_instrument
    session._sae_probes["sae/1"] = {
        "feature_id": 1, "layer": 1, "label": None, "max_act": None,
    }
    run = inst.current_run
    assert run.bound is False
    first = run.observe(0, {1: torch.tensor([0.2, 3.0, 5.0, 1.0])})
    second = run.observe(0, {1: torch.tensor([0.2, 8.0, 5.0, 1.0])})
    assert first["sae/1"].coords == (3.0,)
    assert second["sae/1"].coords == (8.0,)


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
