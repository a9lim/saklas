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
        emap = session._gen_state.emit_map
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

    # Additive rich channel: the full per-probe vector coordinate reading
    # (coords + fraction + nearest) for the latest token, lifted from the
    # session's per-token probe payload.  New key — the old webui ignores it,
    # while a coordinate-aware native client can read coords without the
    # ``scores`` axis-0 shape changing.
    with suppress(Exception):
        payload = getattr(session, "_last_token_probe_payload", None)
        vector_readings = (
            payload.get("readings") if isinstance(payload, dict) else None
        )
        if vector_readings:
            event["probe_readings"] = {
                name: r.to_dict() for name, r in vector_readings.items()
            }

    with suppress(Exception):
        payload = getattr(session, "_last_token_probe_payload", None)
        readings = (
            payload.get("manifold_readings")
            if isinstance(payload, dict)
            else None
        )
        if readings is None:
            mf_monitor = getattr(session, "_manifold_monitor", None)
            capture = getattr(session, "_capture", None)
            per_layer = (
                getattr(capture, "_per_layer", None)
                if capture is not None
                else None
            )
            if (
                mf_monitor is not None
                and mf_monitor.probe_names
                and per_layer
            ):
                latest_hidden = {
                    layer_idx: bucket[-1]
                    for layer_idx, bucket in per_layer.items()
                    if bucket
                }
                if latest_hidden:
                    readings = mf_monitor.score_single_token(latest_hidden)
        if readings:
            event["manifold_readings"] = {
                name: r.to_dict() for name, r in readings.items()
            }

    return event
