"""Token-probe payload helper tests."""

from __future__ import annotations

import math
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


def test_token_payload_builds_endpoint_shaped_captured_record() -> None:
    reading = ProbeReading(
        fraction=0.5,
        nearest=[("happy", 0.1)],
        coords=(0.25,),
        coords_per_layer={3: (0.25,)},
    )
    payload = build_token_probe_payload(
        monitor=SimpleNamespace(
            score_single_token=lambda _hidden: {"toy": reading},
        ),
        capture=SimpleNamespace(
            per_layer_buckets=lambda: {3: [torch.ones(2)]},
        ),
        capture_state=SimpleNamespace(incremental=False, lean=False),
        incremental_readings=[],
        needs_scores=True,
        persists_layer_scores=True,
        assistant_node_id="node-1",
    ).to_token_payload(
        lens={3: [(" tok", 0.25)]},
        lens_aggregate=[(" tok", 0.2, 0.6, 0.1)],
        lens_token_ids={3: [42]},
        lens_source="local:default",
        sae=[(7, 3.5, "feature seven", 5.0)],
        sae_source="saelens:release",
        sae_layer=3,
        steering="0.3 calm",
    )

    captured = payload["captured"]
    assert captured["probes"]["provenance"] == "captured"
    assert captured["probes"]["readings"]["toy"] == reading.to_dict()
    assert captured["lens"] == {
        "provenance": "captured",
        "source": "local:default",
        "steering": "0.3 calm",
        "layers": [{
            "layer": 3,
            "tokens": [{
                "token": " tok",
                "id": 42,
                "logprob": pytest.approx(math.log(0.25)),
            }],
        }],
        "aggregate": [{
            "token": " tok",
            "strength": 0.2,
            "com": 0.6,
            "spread": 0.1,
        }],
    }
    assert captured["sae"] == {
        "provenance": "captured",
        "source": "saelens:release",
        "steering": "0.3 calm",
        "layer": 3,
        "features": [{
            "id": 7,
            "activation": 3.5,
            "label": "feature seven",
            "max_act": 5.0,
        }],
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
