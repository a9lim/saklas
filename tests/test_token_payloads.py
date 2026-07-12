"""Token-probe payload helper tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from saklas.core.results import ProbeReading
from saklas.core.token_payloads import build_token_probe_payload


def test_build_token_probe_payload_scores_once_and_shapes_channels() -> None:
    reading = ProbeReading(
        fraction=0.5,
        nearest=[("happy", 0.1)],
        coords=(0.25,),
        coords_per_layer={3: (0.25,)},
    )
    monitor = SimpleNamespace(
        score_single_token=lambda hidden: {"toy": reading},
    )
    capture = SimpleNamespace(
        per_layer_buckets=lambda: {3: [torch.ones(2)]},
    )
    capture_state = SimpleNamespace(incremental=False, lean=False)

    payload = build_token_probe_payload(
        monitor=monitor,
        capture=capture,
        capture_state=capture_state,
        incremental_readings=[],
        needs_scores=True,
        persists_layer_scores=True,
        assistant_node_id="node-1",
    )

    assert payload.scores == {"toy": pytest.approx(0.25)}
    assert payload.probe_readings == {"toy": reading}
    assert payload.per_layer_scores == {"3": {"toy": pytest.approx(0.25)}}
    assert payload.to_token_payload(lens={"3": [("tok", 1.0)]})["lens"] == {
        "3": [("tok", 1.0)]
    }


def test_build_token_probe_payload_reuses_incremental_reading() -> None:
    reading = ProbeReading(
        fraction=0.4,
        nearest=[("calm", 0.2)],
        coords=(0.75,),
    )

    def _should_not_score(_hidden: Any) -> dict[str, ProbeReading]:
        raise AssertionError("incremental reading should avoid rescoring")

    payload = build_token_probe_payload(
        monitor=SimpleNamespace(score_single_token=_should_not_score),
        capture=SimpleNamespace(per_layer_buckets=lambda: {3: [torch.ones(2)]}),
        capture_state=SimpleNamespace(incremental=True, lean=False),
        incremental_readings=[{"toy": reading}],
        needs_scores=True,
        persists_layer_scores=False,
        assistant_node_id="node-1",
    )

    assert payload.scores == {"toy": pytest.approx(0.75)}
    assert payload.probe_readings == {"toy": reading}
