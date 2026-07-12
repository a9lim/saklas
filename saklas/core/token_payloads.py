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
    per_layer_scores: dict[str, dict[str, float]] | None = None
    probe_readings: dict[str, ProbeReading] | None = None

    def to_token_payload(
        self, *, lens: Any = None, lens_aggregate: Any = None, sae: Any = None,
    ) -> dict[str, Any]:
        return {
            "scores": self.scores,
            "per_layer_scores": self.per_layer_scores,
            "probe_readings": self.probe_readings,
            "lens": lens,
            "lens_aggregate": lens_aggregate,
            "sae": sae,
        }

    def merge_readings(
        self,
        extra: dict[str, ProbeReading],
        *,
        per_layer: bool = False,
    ) -> None:
        """Merge additional per-probe readings (e.g. J-lens token probes,
        which score on the lens path rather than through the Monitor) into
        the canonical rich channel and its derived scalar views."""
        if not extra:
            return
        self.scores = {**(self.scores or {}), **_axis0_scores(extra)}
        self.probe_readings = {**(self.probe_readings or {}), **extra}
        if per_layer:
            merged = dict(self.per_layer_scores or {})
            for layer, row in (_per_layer_axis0(extra) or {}).items():
                merged[layer] = {**merged.get(layer, {}), **row}
            self.per_layer_scores = merged or None


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
        per_layer_scores=(
            _per_layer_axis0(readings)
            if assistant_node_id is not None and persists_layer_scores
            else None
        ),
        probe_readings=readings,
    )
