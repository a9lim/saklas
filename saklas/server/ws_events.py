"""Native WebSocket event-shaping helpers."""

from __future__ import annotations

from contextlib import suppress
from typing import Any


def build_token_event(
    session: Any,
    node_holder: list[str | None],
    *,
    text: str,
    is_thinking: bool,
    tid: int | None,
    lp: float | None,
    top_alts: list[Any] | None,
) -> dict[str, Any]:
    """Build one native WS ``token`` frame from the current engine state."""
    node_id = node_holder[0]
    if node_id is None:
        with suppress(Exception):
            candidate = session.tree.active_node_id
            if isinstance(candidate, str):
                node_id = candidate
                node_holder[0] = candidate

    event: dict[str, Any] = {
        "type": "token",
        "text": text,
        "thinking": bool(is_thinking),
        "token_id": int(tid) if tid is not None else None,
        "node_id": node_id,
    }
    if lp is not None:
        event["logprob"] = float(lp)
    if top_alts:
        event["top_alts"] = [
            {"id": int(a.id), "text": a.text, "logprob": float(a.logprob)}
            for a in top_alts
        ]

    with suppress(Exception):
        emap = session.generation_state.emit_map
        if emap:
            event["raw_index"] = int(emap[-1][0])

    with suppress(Exception):
        node = session.tree.nodes.get(node_id) if node_id else None
        rows = (
            node.thinking_tokens if is_thinking else node.tokens
        ) if node is not None else None
        last = rows[-1] if rows else None
        if last is not None:
            # ``probes`` is the per-probe coordinate axis-0 float the session
            # already collapsed from each reading; ``per_layer_scores`` is the
            # per-layer coordinate map.  Both keep their existing wire shapes
            # ({name: float} / {layer: {name: float}}) for the un-updated webui.
            probes_blob = last.get("probes")
            if probes_blob:
                event["scores"] = probes_blob
            per_layer_blob = last.get("per_layer_scores")
            if per_layer_blob:
                event["per_layer_scores"] = per_layer_blob

    # Rich channel: the full per-probe reading (coords + fraction + nearest)
    # for the latest token.  ``readings`` and ``probe_readings`` in the payload
    # are the *same* unified per-probe dict — the token tap is the single owner
    # of live probe scoring, so event shaping never reaches into private capture
    # buffers or re-scores the token on the WebSocket path.
    with suppress(Exception):
        payload = getattr(session, "_last_token_probe_payload", None)
        readings = None
        if isinstance(payload, dict):
            readings = payload.get("readings") or payload.get("probe_readings")
        if readings:
            event["probe_readings"] = {
                name: r.to_dict() for name, r in readings.items()
            }

    # Live J-lens workspace readout: the step's top-k lens tokens per selected
    # layer (``enable_live_lens``), stashed by the token tap alongside the
    # probe readings.  String layer keys to match ``per_layer_scores``' wire
    # shape; ``[token, score]`` pairs serialize as 2-arrays.
    with suppress(Exception):
        payload = getattr(session, "_last_token_probe_payload", None)
        lens = payload.get("lens") if isinstance(payload, dict) else None
        if lens:
            event["lens_readout"] = {
                str(layer): [[tok, float(score)] for tok, score in pairs]
                for layer, pairs in lens.items()
            }

    return event
