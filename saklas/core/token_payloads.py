"""Token-level probe payload shaping for generation callbacks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast

import torch

from saklas.core.results import ProbeReading


@dataclass(slots=True)
class TokenProbePayload:
    """Probe-derived payloads for one generated token."""

    scores: dict[str, float] | None = None
    per_layer_scores: dict[str, dict[str, float]] | None = None
    probe_readings: dict[str, ProbeReading] | None = None

    def to_token_payload(
        self,
        *,
        lens: Any = None,
        lens_aggregate: Any = None,
        lens_token_ids: Any = None,
        lens_source: str | None = None,
        sae: Any = None,
        sae_source: str | None = None,
        sae_layer: int | None = None,
        steering: str | None = None,
    ) -> dict[str, Any]:
        payload = {
            "scores": self.scores,
            "per_layer_scores": self.per_layer_scores,
            "probe_readings": self.probe_readings,
            "lens": lens,
            "lens_aggregate": lens_aggregate,
            "sae": sae,
        }
        captured = serialize_captured_token_channels(
            payload,
            lens_token_ids=lens_token_ids,
            lens_source=lens_source,
            sae_source=sae_source,
            sae_layer=sae_layer,
            steering=steering,
        )
        if captured:
            payload["captured"] = captured
        return payload

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


def serialize_captured_token_channels(
    payload: dict[str, Any],
    *,
    lens_token_ids: dict[int, list[int]] | None = None,
    lens_source: str | None = None,
    sae_source: str | None = None,
    sae_layer: int | None = None,
    steering: str | None = None,
) -> dict[str, Any]:
    """Return the JSON-safe, loom-owned measurement record for one token.

    The decode tap calls this once, then places the same ``captured`` object
    on both the loom token row and the native WebSocket frame.  Legacy scalar
    aliases remain outside this record for existing consumers, but this is the
    authoritative rich shape used by historical hover and drilldown surfaces.
    """
    captured: dict[str, Any] = {}

    scores = payload.get("scores")
    per_layer_scores = payload.get("per_layer_scores")
    readings = payload.get("probe_readings")
    if scores or per_layer_scores or readings:
        probes: dict[str, Any] = {"provenance": "captured"}
        if scores:
            probes["scores"] = {
                str(name): round(float(value), 6)
                for name, value in scores.items()
            }
        if per_layer_scores:
            probes["per_layer_scores"] = per_layer_scores
        if readings:
            probes["readings"] = {
                str(name): reading.to_dict()
                for name, reading in readings.items()
            }
        captured["probes"] = probes

    lens = cast(
        dict[int, list[tuple[str, float]]] | None,
        payload.get("lens"),
    )
    aggregate = cast(
        list[tuple[str, float, float, float]] | None,
        payload.get("lens_aggregate"),
    )
    if lens or aggregate:
        id_rows = lens_token_ids or {}
        layers: list[dict[str, Any]] = []
        for layer, row in sorted(lens.items() if lens else ()):
            token_ids = id_rows.get(int(layer), [])
            tokens: list[dict[str, Any]] = []
            for index, pair in enumerate(row):
                token, probability = pair
                token_id = token_ids[index] if index < len(token_ids) else -1
                # The live display already calibrated this row once.  Convert
                # that probability into the endpoint's logprob unit without a
                # second softmax; the floor keeps strict JSON finite.
                logprob = math.log(max(float(probability), 1e-45))
                tokens.append({
                    "token": str(token),
                    "id": int(token_id),
                    "logprob": float(logprob),
                })
            layers.append({"layer": int(layer), "tokens": tokens})
        captured["lens"] = {
            "provenance": "captured",
            "source": lens_source,
            "steering": steering,
            "layers": layers,
            "aggregate": [
                {
                    "token": str(token),
                    "strength": float(strength),
                    "com": float(com),
                    "spread": float(spread),
                }
                for token, strength, com, spread in (aggregate or ())
            ],
        }

    sae = payload.get("sae")
    if sae:
        captured["sae"] = {
            "provenance": "captured",
            "source": sae_source,
            "steering": steering,
            "layer": int(sae_layer) if sae_layer is not None else None,
            "features": [
                {
                    "id": int(row[0]),
                    "activation": float(row[1]),
                    "label": row[2],
                    "max_act": (
                        float(row[3])
                        if len(row) > 3 and row[3] is not None
                        else None
                    ),
                }
                for row in sae
            ],
        }

    return captured


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
