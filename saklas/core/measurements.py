"""The measurement envelope — the one wire record for read-side data.

Every surface that ships per-token, aggregate, or replayed instrument
data — the native WS ``token``/``done`` frames, loom token rows, the
token-replay endpoints — carries this single versioned shape (5.x; it
replaces the ``captured`` record and the six legacy top-level aliases:
``scores`` / ``per_layer_scores`` / ``probe_readings`` / ``lens_readout``
/ ``lens_aggregate`` / ``sae_readout``).

Shape::

    {
      "version": 1,
      "scope": "token" | "aggregate" | "replay",
      "provenance": "captured" | "replayed",
      "scores": {probe: axis0, ...},              # flat cross-family view
      "per_layer_scores": {layer: {probe: v}},    # optional heatmap view
      "instruments": {
        "geometry": {"readings": {name: ProbeReading.to_dict()}},
        "lens": {
          "binding": {"source": ..., "steering": ...},
          "readings": {name: ...},                # attached jlens/<word> probes
          "readout": {"layers": [...], "aggregate": [...]}   # native top-k
        },
        "sae": {
          "binding": {"source": ..., "steering": ..., "layer": N},
          "readings": {name: ...},                # attached sae/<id> probes
          "readout": {"features": [...]}          # native top-k discovery
        }
      }
    }

The two axes sol's review named are explicit: a family's ``readings``
are its *attached probes'* values; its ``readout`` is the *native
discovery* surface (per-layer top-k matrix / feature list).  ``scores``
and ``per_layer_scores`` stay envelope-level because their consumers
(transcript tinting, the loom heatmap) key probes across families by
name — they are flat views derived from the same readings, never extra
data.  ``binding`` records what the family was measuring (source
identity + recipe steering) so historical rows stay interpretable after
a source switch.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from saklas.core.results import ProbeReading

MEASUREMENTS_VERSION = 1


def _axis0(readings: Mapping[str, ProbeReading]) -> dict[str, float]:
    return {
        str(name): round(
            float(reading.coords[0] if reading.coords else 0.0), 6,
        )
        for name, reading in readings.items()
    }


def _readings_dict(
    readings: Mapping[str, ProbeReading] | None,
) -> dict[str, Any] | None:
    if not readings:
        return None
    return {str(name): reading.to_dict() for name, reading in readings.items()}


def build_measurements(
    *,
    scope: str,
    provenance: str = "captured",
    geometry_readings: Mapping[str, ProbeReading] | None = None,
    lens_readings: Mapping[str, ProbeReading] | None = None,
    sae_readings: Mapping[str, ProbeReading] | None = None,
    per_layer_scores: Mapping[str, Mapping[str, float]] | None = None,
    lens_readout: Mapping[int, list[tuple[str, float]]] | None = None,
    lens_aggregate: list[tuple[str, float, float, float]] | None = None,
    lens_token_ids: Mapping[int, list[int]] | None = None,
    lens_source: str | None = None,
    sae_features: list[tuple[int, float, str | None, float | None]] | None = None,
    sae_source: str | None = None,
    sae_layer: int | None = None,
    steering: str | None = None,
) -> dict[str, Any] | None:
    """Build one measurement envelope; ``None`` when nothing measured.

    JSON-safe throughout (floats rounded/floored exactly as the historical
    ``captured`` serializers did, so persisted loom rows stay comparable
    across the 5.x boundary).
    """
    instruments: dict[str, Any] = {}
    scores: dict[str, float] = {}

    if geometry_readings:
        scores.update(_axis0(geometry_readings))
        instruments["geometry"] = {
            "readings": _readings_dict(geometry_readings),
        }

    lens_channel: dict[str, Any] = {}
    if lens_readings:
        scores.update(_axis0(lens_readings))
        lens_channel["readings"] = _readings_dict(lens_readings)
    if lens_readout or lens_aggregate:
        id_rows = lens_token_ids or {}
        layers: list[dict[str, Any]] = []
        for layer, row in sorted(lens_readout.items() if lens_readout else ()):
            token_ids = id_rows.get(int(layer), [])
            tokens: list[dict[str, Any]] = []
            for index, (token, probability) in enumerate(row):
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
        lens_channel["readout"] = {
            "layers": layers,
            "aggregate": [
                {
                    "token": str(token),
                    "strength": float(strength),
                    "com": float(com),
                    "spread": float(spread),
                }
                for token, strength, com, spread in (lens_aggregate or ())
            ],
        }
    if lens_channel:
        lens_channel["binding"] = {
            "source": lens_source,
            "steering": steering,
        }
        instruments["lens"] = lens_channel

    sae_channel: dict[str, Any] = {}
    if sae_readings:
        scores.update(_axis0(sae_readings))
        sae_channel["readings"] = _readings_dict(sae_readings)
    if sae_features:
        sae_channel["readout"] = {
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
                for row in sae_features
            ],
        }
    if sae_channel:
        sae_channel["binding"] = {
            "source": sae_source,
            "steering": steering,
            "layer": int(sae_layer) if sae_layer is not None else None,
        }
        instruments["sae"] = sae_channel

    if not instruments:
        return None

    envelope: dict[str, Any] = {
        "version": MEASUREMENTS_VERSION,
        "scope": scope,
        "provenance": provenance,
        "instruments": instruments,
    }
    if scores:
        envelope["scores"] = scores
    if per_layer_scores:
        envelope["per_layer_scores"] = {
            str(layer): dict(row) for layer, row in per_layer_scores.items()
        }
    return envelope


__all__ = ["MEASUREMENTS_VERSION", "build_measurements"]
