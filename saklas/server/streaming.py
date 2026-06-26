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

from contextlib import suppress
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from saklas.core.results import GenerationResult, RunSet
    from saklas.core.session import SaklasSession


def probe_reading_aggregate(
    session: "SaklasSession", result: "GenerationResult | RunSet | None",
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
    readings = getattr(result, "probe_readings", None) or {}
    if not readings:
        return {}
    try:
        attached = set(session.monitor.probe_names)
    except Exception:
        attached = set(readings.keys())
    out: dict[str, Any] = {}
    for name, agg in readings.items():
        if name not in attached:
            continue
        with suppress(Exception):
            out[name] = agg.to_dict()
    return out


def _usage_dict(result: "GenerationResult") -> dict[str, int]:
    pt = result.prompt_tokens
    ct = result.token_count
    return {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}


def stream_finalizer(
    session: "SaklasSession", result: "GenerationResult | RunSet | None",
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
        finish_reason = getattr(result, "finish_reason", None)
        usage: dict[str, int] | None = _usage_dict(result)  # pyright: ignore[reportArgumentType]
    else:
        finish_reason = session.generation_state.finish_reason
        usage = None
    probe_agg = probe_reading_aggregate(session, result)
    return finish_reason, usage, probe_agg
