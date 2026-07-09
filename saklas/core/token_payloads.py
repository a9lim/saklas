"""Token-level probe payload shaping for generation callbacks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from saklas.core.results import ProbeReading


@dataclass(slots=True)
class TokenProbePayload:
    """Probe-derived payloads for one generated token."""

    scores: dict[str, float] | None = None
    readings: dict[str, ProbeReading] | None = None
    per_layer_scores: dict[str, dict[str, float]] | None = None
    probe_readings: dict[str, ProbeReading] | None = None

    def to_token_payload(
        self, *, lens: Any = None, lens_aggregate: Any = None,
    ) -> dict[str, Any]:
        return {
            "scores": self.scores,
            "readings": self.readings,
            "per_layer_scores": self.per_layer_scores,
            "probe_readings": self.probe_readings,
            "lens": lens,
            "lens_aggregate": lens_aggregate,
        }


def _axis0_scores(
    readings: dict[str, ProbeReading],
) -> dict[str, float]:
    return {
        name: (reading.coords[0] if reading.coords else 0.0)
        for name, reading in readings.items()
    }


def _per_layer_axis0(
    readings: dict[str, ProbeReading],
) -> dict[str, dict[str, float]] | None:
    by_layer: dict[str, dict[str, float]] = {}
    for probe_name, reading in readings.items():
        for layer, coord in reading.coords_per_layer.items():
            by_layer.setdefault(str(layer), {})[probe_name] = round(
                float(coord[0] if coord else 0.0), 6,
            )
    return by_layer or None


def build_token_probe_payload(
    *,
    monitor: Any,
    capture: Any,
    capture_state: Any,
    incremental_readings: list[dict[str, ProbeReading]],
    needs_scores: bool,
    wants_live_token_scores: bool,
    persists_layer_scores: bool,
    assistant_node_id: str | None,
) -> TokenProbePayload:
    """Score and shape probe payloads for one generated token.

    This is the token-probe slice of ``SaklasSession._token_tap`` extracted into
    a typed helper. It performs at most one monitor geometry pass: either reuse
    the most recent incremental reading, or score the latest captured hidden
    states once and derive all scalar/per-layer/live payloads from that result.
    """
    if not needs_scores:
        return TokenProbePayload()

    has_incremental_reading = bool(
        (capture_state.incremental or capture_state.lean)
        and incremental_readings
    )
    latest_hidden_for_token: dict[int, torch.Tensor] | None = None
    if not has_incremental_reading:
        latest_hidden_for_token = {
            layer_idx: bucket[-1]
            for layer_idx, bucket in capture.per_layer_buckets().items()
            if bucket
        }

    readings: dict[str, ProbeReading] | None = None
    if has_incremental_reading:
        readings = incremental_readings[-1] or None
    elif latest_hidden_for_token:
        readings = monitor.score_single_token(latest_hidden_for_token) or None

    if not readings:
        return TokenProbePayload()

    return TokenProbePayload(
        scores=_axis0_scores(readings),
        readings=readings,
        per_layer_scores=(
            _per_layer_axis0(readings)
            if assistant_node_id is not None and persists_layer_scores
            else None
        ),
        probe_readings=readings if wants_live_token_scores else None,
    )
