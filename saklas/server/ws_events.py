"""Native WebSocket event-shaping helpers."""

from __future__ import annotations

from typing import Any

from saklas.core.results import TokenAlt
from saklas.core.session import SaklasSession


def build_token_event(
    session: SaklasSession,
    node_holder: list[str | None],
    *,
    text: str,
    is_thinking: bool,
    tid: int | None,
    lp: float | None,
    top_alts: list[TokenAlt] | None,
    perplexity: float | None = None,
) -> dict[str, Any]:
    """Build one native WS ``token`` frame from the current engine state."""
    node_id = node_holder[0]
    if node_id is None:
        node_id = session.tree.active_node_id
        node_holder[0] = node_id

    event: dict[str, Any] = {
        "type": "token",
        "text": text,
        "thinking": bool(is_thinking),
        "token_id": int(tid) if tid is not None else None,
        "node_id": node_id,
    }
    if lp is not None:
        event["logprob"] = float(lp)
    if perplexity is not None:
        event["perplexity"] = float(perplexity)
    if top_alts:
        event["top_alts"] = [
            {"id": int(a.id), "text": a.text, "logprob": float(a.logprob)}
            for a in top_alts
        ]

    emap = session.generation_state.emit_map
    if emap:
        event["raw_index"] = int(emap[-1][0])

    # The token tap owns this payload.  Do not reconstruct it from persisted
    # loom rows: doing so creates a second wire authority and masks tap bugs.
    payload = session.token_probe_payload

    # The 5.x measurement envelope is the single read-side wire record: geometry
    # / lens / SAE ``instruments`` plus the flat ``scores`` / ``per_layer_scores``
    # views (:func:`saklas.core.measurements.build_measurements`).  The token tap
    # builds it once and stores it on the loom row; the WebSocket forwards it
    # verbatim so live and rehydrated tokens share one rich-channel authority.
    # It replaces the former top-level ``scores`` / ``per_layer_scores`` /
    # ``probe_readings`` / ``captured`` / ``lens_readout`` / ``lens_aggregate`` /
    # ``sae_readout`` aliases, which are gone from the token frame.
    measurements = payload.get("measurements")
    if measurements:
        event["measurements"] = measurements

    return event
