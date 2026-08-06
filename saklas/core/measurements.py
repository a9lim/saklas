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
from typing import Any, Literal, Mapping, NotRequired, TypedDict, cast

from saklas.core.instruments.types import Reading, ScalarReading, reading_axis0
from saklas.core.results import ProbeReading

MEASUREMENTS_VERSION = 1

MeasurementScope = Literal["token", "aggregate", "replay"]
MeasurementProvenance = Literal["captured", "replayed"]


# --------------------------------------------------------------------------
# The envelope, declared
# --------------------------------------------------------------------------
# The wire contract used to live only in hand-mirrored TypeScript, which is
# how `probe_measurements_aggregate`'s output went unread and how a unit no
# producer held ended up on the wire.  These TypedDicts are the Python-side
# declaration; ``tests/test_measurements_envelope.py`` pins the exact key
# sets of a token frame, a done aggregate, and all three replay envelopes,
# so wire drift is a test failure rather than a dashboard bug.

class MeasurementBinding(TypedDict):
    """What a family was measuring: source identity + recipe steering."""

    source: str | None
    steering: str | None
    layer: NotRequired[int | None]


class DepthSummaryDict(TypedDict):
    """Wire form of ``instruments.types.DepthSummary``.

    ``basis`` travels with the numbers because ``center`` means three
    mathematically unrelated things across the families.
    """

    center: list[float]
    spread: list[float]
    basis: str


class ScalarReadingDict(TypedDict):
    """Wire form of ``ScalarReading.to_dict()`` — the lens/SAE reading.

    One value with an explicit ``unit``, its per-layer trace, and a depth
    summary.  No geometry fields: a readout channel has no subspace behind
    it.  ``meta`` is emitted only when the producer attached one.
    """

    value: float
    unit: str
    per_layer: dict[str, float]
    depth: DepthSummaryDict | None
    meta: NotRequired[dict[str, Any]]


class ProbeReadingDict(TypedDict):
    """Wire form of ``ProbeReading.to_dict()`` — the geometry reading.

    The complete key set that serializer emits, in its order.  Declaring it
    here (rather than as an anonymous ``dict[str, Any]`` on the channel)
    is what lets the generated dashboard types name the geometry reading
    instead of falling back to an opaque record.
    """

    fraction: float
    nearest: list[tuple[str, float]]
    coords: list[float]
    residual: float
    fraction_per_layer: dict[str, float]
    coords_per_layer: dict[str, list[float]]
    residual_per_layer: dict[str, float]
    # The live serializer always emits these five, but a *persisted* loom
    # row written before they existed does not, and the same wire type
    # covers both — so they are optional on read.
    assignment: NotRequired[list[tuple[str, float]]]
    membership: NotRequired[float]
    depth_com: NotRequired[list[float]]
    depth_spread: NotRequired[list[float]]
    subspace_coords_per_layer: NotRequired[dict[str, list[float]]]


class LensReadoutToken(TypedDict):
    token: str
    id: int
    logprob: float


class LensReadoutLayer(TypedDict):
    layer: int
    tokens: list[LensReadoutToken]


class LensAggregateToken(TypedDict):
    token: str
    strength: float
    com: float
    spread: float


class LensReadout(TypedDict):
    layers: list[LensReadoutLayer]
    aggregate: list[LensAggregateToken]


class SaeFeature(TypedDict):
    id: int
    activation: float
    label: str | None
    max_act: float | None


class SaeReadout(TypedDict):
    features: list[SaeFeature]


class GeometryChannel(TypedDict):
    # Never null: ``build_measurements`` creates the channel only when there
    # are geometry readings to put in it, so the historical ``| None`` was
    # defensive over-typing that every consumer then had to defend against.
    readings: dict[str, ProbeReadingDict]
    binding: NotRequired[MeasurementBinding]


class LensChannel(TypedDict):
    binding: MeasurementBinding
    readings: NotRequired[dict[str, ScalarReadingDict]]
    readout: NotRequired[LensReadout]


class SaeChannel(TypedDict):
    binding: MeasurementBinding
    readings: NotRequired[dict[str, ScalarReadingDict]]
    readout: NotRequired[SaeReadout]


class Instruments(TypedDict):
    geometry: NotRequired[GeometryChannel]
    lens: NotRequired[LensChannel]
    sae: NotRequired[SaeChannel]


class Measurements(TypedDict):
    """The one versioned read-side wire record."""

    version: int
    scope: MeasurementScope
    provenance: MeasurementProvenance
    instruments: Instruments
    scores: NotRequired[dict[str, float]]
    per_layer_scores: NotRequired[dict[str, dict[str, float]]]


class MeasurementsEnvelope(TypedDict):
    """The replay endpoints' response body."""

    measurements: Measurements | None


def _axis0(readings: Mapping[str, Reading]) -> dict[str, float]:
    return {
        str(name): round(reading_axis0(reading), 6)
        for name, reading in readings.items()
    }


def _readings_dict(
    readings: Mapping[str, Reading] | None,
) -> dict[str, Any] | None:
    """Serialize a family's attached-probe readings in ITS OWN shape.

    Geometry emits the full ``ProbeReading``; the single-axis families emit
    the native ``ScalarReading`` (``{value, unit, per_layer, depth}``).  The
    eight constant geometry fields a synthesized ``ProbeReading`` used to
    carry for lens/SAE probes — ``fraction`` 0, ``residual`` 0, empty
    ``nearest``/``assignment``, ``membership`` 1.0 and three empty per-layer
    maps — are off the wire: they were constants masquerading as
    measurements, and the client had to infer the unit they omitted.
    """
    if not readings:
        return None
    return {str(name): reading.to_dict() for name, reading in readings.items()}


def build_measurements(
    *,
    scope: str,
    provenance: str = "captured",
    geometry_readings: Mapping[str, ProbeReading] | None = None,
    geometry_binding: Mapping[str, Any] | None = None,
    lens_readings: Mapping[str, ScalarReading] | None = None,
    sae_readings: Mapping[str, ScalarReading] | None = None,
    per_layer_scores: Mapping[str, Mapping[str, float]] | None = None,
    lens_readout: Mapping[int, list[tuple[str, float]]] | None = None,
    lens_aggregate: list[tuple[str, float, float, float]] | None = None,
    lens_token_ids: Mapping[int, list[int]] | None = None,
    lens_source: str | None = None,
    sae_features: list[tuple[int, float, str | None, float | None]] | None = None,
    sae_source: str | None = None,
    sae_layer: int | None = None,
    steering: str | None = None,
) -> Measurements | None:
    """Build one measurement envelope; ``None`` when nothing measured.

    JSON-safe throughout (floats rounded/floored exactly as the historical
    ``captured`` serializers did, so persisted loom rows stay comparable
    across the 5.x boundary).
    """
    instruments: dict[str, Any] = {}
    scores: dict[str, float] = {}

    if geometry_readings:
        scores.update(_axis0(geometry_readings))
        geometry_channel: dict[str, Any] = {
            "readings": _readings_dict(geometry_readings),
        }
        # Opt-in only (the replay envelope records its applied steering,
        # mirroring lens/sae): live token/aggregate envelopes stay
        # binding-less on the geometry channel — the recipe already lives
        # on the loom node.
        if geometry_binding is not None:
            geometry_channel["binding"] = dict(geometry_binding)
        instruments["geometry"] = geometry_channel

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
                # Every producer holds the per-layer probability ``p_l`` (the
                # readout's one unit) and every consumer wants it back; this
                # is the ONE conversion into the wire's logprob key, and the
                # floor keeps strict JSON finite.
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

    envelope: Measurements = {
        "version": MEASUREMENTS_VERSION,
        "scope": cast(MeasurementScope, scope),
        "provenance": cast(MeasurementProvenance, provenance),
        "instruments": cast(Instruments, instruments),
    }
    if scores:
        envelope["scores"] = scores
    if per_layer_scores:
        envelope["per_layer_scores"] = {
            str(layer): dict(row) for layer, row in per_layer_scores.items()
        }
    return envelope


__all__ = [
    "DepthSummaryDict",
    "GeometryChannel",
    "Instruments",
    "LensAggregateToken",
    "LensChannel",
    "LensReadout",
    "LensReadoutLayer",
    "LensReadoutToken",
    "MEASUREMENTS_VERSION",
    "MeasurementBinding",
    "MeasurementProvenance",
    "MeasurementScope",
    "Measurements",
    "MeasurementsEnvelope",
    "ProbeReadingDict",
    "SaeChannel",
    "SaeFeature",
    "SaeReadout",
    "ScalarReadingDict",
    "build_measurements",
]
