"""CPU-only contract tests for the first-class live SAE runtime."""
from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from saklas.core.sae import MockSaeBackend, select_runtime_layer
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
    session._sae_labels = {"2": "feature two"}
    session._live_sae = None
    session._sae_probes = {}
    session._sae_step_stash = None
    session._last_sae_step_readings = None
    session._profiles = {}
    session._probe_hash_cache = {}
    session._analytics_cpu_cache = {}
    session._capture = _Capture(torch.tensor([0.2, 3.0, 5.0, 1.0]))
    session._live_sae_active_for_generation = True
    session._invalidate_prefix_cache = lambda: None  # type: ignore[method-assign]
    session._invalidate_analytics_cache = lambda: None  # type: ignore[method-assign]
    return cast(SaklasSession, session)


def test_select_runtime_layer_prefers_workspace_near_65_percent() -> None:
    assert select_runtime_layer({1, 8, 14, 18}, 24) == 14
    assert select_runtime_layer({1, 8, 14, 18}, 24, requested=8) == 8
    with pytest.raises(ValueError, match="does not cover"):
        select_runtime_layer({1, 8}, 24, requested=9)


def test_sae_feature_validation_and_decoder_row_registration() -> None:
    session = _session()
    assert session.validate_sae_feature(2) == {
        "id": 2, "label": "feature two", "layer": 1,
    }
    name = session.register_sae_direction(2)
    assert name == "sae/2"
    assert torch.equal(session._profiles[name][1], torch.tensor([0.0, 0.0, 1.0, 0.0]))
    with pytest.raises(ValueError, match="out of range"):
        session.validate_sae_feature(9)


def test_live_sae_readout_and_probe_share_one_encoder_result() -> None:
    session = _session()
    session._sae_probes["sae/2"] = {
        "feature_id": 2, "layer": 1, "label": "feature two",
    }
    assert session.enable_live_sae(top_k=3) == {"layer": 1, "top_k": 3}
    readout = session._live_sae_readout_step()
    assert readout == [
        (2, 5.0, "feature two"),
        (1, 3.0, None),
        (3, 1.0, None),
    ]
    reading = session._last_sae_step_readings["sae/2"]  # type: ignore[index]
    assert reading.coords == (5.0,)
    assert reading.coords_per_layer == {1: (5.0,)}


def test_sae_gate_scalar_stashes_activations_for_live_step() -> None:
    session = _session()
    session._sae_probes["sae/1"] = {
        "feature_id": 1, "layer": 1, "label": None,
    }
    scalars = session._score_sae_gate_scalars()
    assert scalars["sae/1"] == pytest.approx(3.0)
    assert scalars["sae/1[0]"] == pytest.approx(3.0)
    assert session._sae_step_stash is not None
    assert session._sae_step_stash["fresh"] is True


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
        "layer": 14, "width": 16_384,
    })
    assert path.exists()
    assert load_sae_metadata("org/model", "release/name") == {
        "format_version": 1,
        "model_id": "org/model",
        "release": "release/name",
        "layer": 14,
        "width": 16_384,
    }
