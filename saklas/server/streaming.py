"""Shared streaming-finalization plumbing for the three streaming protocols.

The SSE (OpenAI), NDJSON (Ollama), and WebSocket co-stream paths each close a
generation by deriving the same three things — the finish reason, the token
``usage`` rollup, and the per-attached-probe aggregate reading — and then format
them into protocol-specific wire frames.  This module owns the *derivation* so
the three sites can't drift; each protocol keeps its own framing (raw vs mapped
finish reason, usage chunk vs Ollama duration stats, the ``x-saklas-probe-readings``
extension vs the native ``probe_readings`` block).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from saklas.core.results import GenerationResult
    from saklas.core.session import SaklasSession


def probe_reading_aggregate(
    session: "SaklasSession", result: "GenerationResult | None",
) -> dict[str, Any]:
    """Per-attached-probe ``ProbeReading.to_dict()`` from ``result``.

    Returns ``{}`` when no result is recorded or no manifold probes are
    attached.  Surfaced under the ``x-saklas-probe-readings`` extension on the
    OpenAI and Ollama responses so vector-probe clients keep working unchanged
    and manifold-aware clients pick up the geometric channel; the WS path reads
    the same dict under its native ``probe_readings`` key.  Result-parameterized
    (rather than reading ``session.last_result``) so the WS per-sibling done
    frames score each sibling's own result.
    """
    if result is None:
        return {}
    readings = result.probe_readings or {}
    if not readings:
        return {}
    attached = set(session.monitor.probe_names)
    # Pinned J-lens token probes and SAE feature probes live on their own
    # session registries (readout channels, not Monitor probes) but land in
    # ``result.probe_readings`` all the same — without this union the
    # attached-filter silently dropped their end-of-gen aggregates from every
    # streaming done frame.
    attached.update(session.lens_probe_names)
    attached.update(session.sae_probe_names)
    return {
        name: reading.to_dict()
        for name, reading in readings.items()
        if name in attached
    }


def probe_measurements_aggregate(
    session: "SaklasSession", result: "GenerationResult | None",
) -> dict[str, Any] | None:
    """Aggregate-scope measurement envelope for the native WS ``done`` frame.

    Splits ``result.probe_readings`` by family — geometry (Monitor probes),
    lens (``session.lens_probe_names``), SAE (``session.sae_probe_names``) — and
    builds one ``scope="aggregate"`` envelope
    (:func:`saklas.core.measurements.build_measurements`).  ``None`` when no
    probe is attached / no result recorded.  The compat ``probe_readings``
    aggregate still rides the done frame separately (see
    :func:`probe_reading_aggregate`); this is the additional 5.x envelope.

    Source / layer binding fields come from the live lens / SAE configs when a
    family actually contributed readings; otherwise ``None`` (so a historical
    row stays interpretable after a source switch).
    """
    from saklas.core.measurements import build_measurements

    if result is None:
        return None
    readings = result.probe_readings or {}
    if not readings:
        return None

    geometry_names = set(session.monitor.probe_names)
    lens_names = set(getattr(session, "lens_probe_names", []) or [])
    sae_names = set(getattr(session, "sae_probe_names", []) or [])
    geometry = {n: r for n, r in readings.items() if n in geometry_names} or None
    lens = {n: r for n, r in readings.items() if n in lens_names} or None
    sae = {n: r for n, r in readings.items() if n in sae_names} or None

    live_lens = getattr(session, "_live_lens", None)
    live_sae = getattr(session, "_live_sae", None)
    lens_source = (
        live_lens.get("source") if lens and isinstance(live_lens, dict) else None
    )
    sae_source = (
        live_sae.get("source") if sae and isinstance(live_sae, dict) else None
    )
    sae_layer = (
        live_sae.get("layer") if sae and isinstance(live_sae, dict) else None
    )

    return build_measurements(
        scope="aggregate",
        geometry_readings=geometry,
        lens_readings=lens,
        sae_readings=sae,
        lens_source=lens_source,
        sae_source=sae_source,
        sae_layer=sae_layer,
        steering=getattr(result, "applied_steering", None),
    )


def _usage_dict(result: "GenerationResult") -> dict[str, int]:
    pt = result.prompt_tokens
    ct = result.token_count
    return {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}


def stream_finalizer(
    session: "SaklasSession", result: "GenerationResult | None",
) -> tuple[str | None, dict[str, int] | None, dict[str, Any]]:
    """Derive the shared end-of-stream triple ``(finish_reason, usage, probe_agg)``.

    ``finish_reason`` comes off ``result`` (the engine stamps it there from
    ``GenerationState.finish_reason`` at result-build time, so it equals the
    live gen-state value the non-streaming sites read), falling back to the live
    gen state when no result is recorded.  ``usage`` is the OpenAI-shaped token
    rollup (``None`` when there is no result; Ollama derives its duration stats
    separately).  ``probe_agg`` is :func:`probe_reading_aggregate`.  Each
    protocol formats these into its own wire shape.
    """
    if result is not None:
        finish_reason = result.finish_reason
        usage: dict[str, int] | None = _usage_dict(result)
    else:
        finish_reason = session.generation_state.finish_reason
        usage = None
    probe_agg = probe_reading_aggregate(session, result)
    return finish_reason, usage, probe_agg
