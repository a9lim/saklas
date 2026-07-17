"""Token-level probe payload shaping for generation callbacks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from saklas.core.measurements import build_measurements
from saklas.core.results import ProbeReading

# Per-family reading slot names, keyed by the ``family`` argument to
# :meth:`TokenProbePayload.merge_readings`.
_FAMILY_SLOTS = {
    "geometry": "geometry_readings",
    "lens": "lens_readings",
    "sae": "sae_readings",
}


@dataclass(slots=True)
class TokenProbePayload:
    """Probe-derived payloads for one generated token.

    Readings are kept in per-family slots (``geometry_readings`` /
    ``lens_readings`` / ``sae_readings``) so the 5.x measurement envelope can
    split them into ``instruments.geometry`` / ``.lens`` / ``.sae`` while the
    flat ``scores`` / ``per_layer_scores`` cross-family views stay derived from
    all three.
    """

    scores: dict[str, float] | None = None
    per_layer_scores: dict[str, dict[str, float]] | None = None
    geometry_readings: dict[str, ProbeReading] | None = None
    lens_readings: dict[str, ProbeReading] | None = None
    sae_readings: dict[str, ProbeReading] | None = None

    @property
    def all_readings(self) -> dict[str, ProbeReading]:
        """The three family slots merged into one per-probe dict.

        Some internal consumers (the library ``TokenEvent.probe_readings``
        field, the aggregate split) want the cross-family union rather than a
        single family's slot.
        """
        merged: dict[str, ProbeReading] = {}
        for slot in (self.geometry_readings, self.lens_readings, self.sae_readings):
            if slot:
                merged.update(slot)
        return merged

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
        """Shape this token's payload as ``{"measurements": <envelope>}``.

        The single 5.x wire record — the envelope replaces the former
        ``captured`` record and the six top-level scalar aliases.  ``lens`` is
        the native per-layer readout (``lens_readout`` in the envelope) and
        ``sae`` the native feature discovery (``sae_features``); the attached
        per-family readings come from the payload's own slots.
        """
        measurements = build_measurements(
            scope="token",
            geometry_readings=self.geometry_readings,
            lens_readings=self.lens_readings,
            sae_readings=self.sae_readings,
            per_layer_scores=self.per_layer_scores,
            lens_readout=lens,
            lens_aggregate=lens_aggregate,
            lens_token_ids=lens_token_ids,
            lens_source=lens_source,
            sae_features=sae,
            sae_source=sae_source,
            sae_layer=sae_layer,
            steering=steering,
        )
        return {"measurements": measurements}

    def merge_readings(
        self,
        extra: dict[str, ProbeReading],
        *,
        family: str,
        per_layer: bool = False,
    ) -> None:
        """Merge additional per-probe readings into a named family's slot.

        ``family`` is one of ``"geometry"`` / ``"lens"`` / ``"sae"`` — J-lens
        token probes and SAE feature probes score on their own paths rather
        than through the Monitor, so their readings land in their own slot.
        The flat ``scores`` (and, when ``per_layer``, ``per_layer_scores``)
        cross-family views are updated the same way regardless of family.
        """
        if not extra:
            return
        try:
            slot_name = _FAMILY_SLOTS[family]
        except KeyError:
            raise ValueError(f"unknown reading family {family!r}") from None
        self.scores = {**(self.scores or {}), **_axis0_scores(extra)}
        current: dict[str, ProbeReading] | None = getattr(self, slot_name)
        setattr(self, slot_name, {**(current or {}), **extra})
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
    geometry_run: Any = None,
    step_id: int = -1,
) -> TokenProbePayload:
    """Score and shape probe payloads for one generated token.

    This is the token-probe slice of ``SaklasSession._token_tap`` extracted into
    a typed helper. It performs at most one monitor geometry pass: either reuse
    the most recent incremental reading, or score the latest captured hidden
    states through the geometry run's step-keyed ``observe`` (FULL-retention
    mode — a gate callback that already full-roster-scored this forward primed
    the memo, so the tap hits it instead of rescoring; ``monitor`` remains the
    fallback for callers without a run).
    The monitor readings land in the ``geometry`` family slot.
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
        if capture_state.lean or geometry_run is None:
            # LEAN rows are deliberately partial (``coords_only``) and are
            # never primed into the observe memo — read the sink's row
            # directly (routing through ``observe`` would either rescore
            # the full roster per token or serve a partial as full).
            readings = incremental_readings[-1] or None
        else:
            # FULL-incremental: the sink primed the run's memo for this
            # forward, so ``observe`` is a hit returning the sink's exact
            # row.  A miss falls back to the sink's appended row (aligned
            # 1:1 with forwards — the miss cases are an empty-capture row,
            # which never primes, and a narrow caller's idle run).
            readings = (
                geometry_run.observe(step_id, {})
                or incremental_readings[-1]
                or None
            )
    elif latest_hidden_for_token:
        readings = (
            geometry_run.observe(step_id, latest_hidden_for_token)
            if geometry_run is not None
            else monitor.score_single_token(latest_hidden_for_token)
        ) or None

    if not readings:
        return TokenProbePayload()

    return TokenProbePayload(
        scores=_axis0_scores(readings),
        per_layer_scores=(
            _per_layer_axis0(readings)
            if assistant_node_id is not None and persists_layer_scores
            else None
        ),
        geometry_readings=readings,
    )
