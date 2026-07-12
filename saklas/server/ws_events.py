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

    payload = None
    with suppress(Exception):
        raw_payload = getattr(session, "_last_token_probe_payload", None)
        if isinstance(raw_payload, dict):
            payload = raw_payload

    scores_blob = payload.get("scores") if payload is not None else None
    if scores_blob:
        # Match the loom-row fallback's wire unit: the session rounds the
        # appended ``probes`` row to six decimals.  Prefer this already-shaped
        # token payload so the WS path does not re-read the just-appended tree
        # row every token.
        event["scores"] = {
            name: round(float(value), 6)
            for name, value in scores_blob.items()
        }
    per_layer_blob = (
        payload.get("per_layer_scores") if payload is not None else None
    )
    if per_layer_blob:
        event["per_layer_scores"] = per_layer_blob

    with suppress(Exception):
        if "scores" not in event or "per_layer_scores" not in event:
            node = session.tree.nodes.get(node_id) if node_id else None
            rows = (
                node.thinking_tokens if is_thinking else node.tokens
            ) if node is not None else None
            last = rows[-1] if rows else None
        else:
            last = None
        if last is not None:
            # ``probes`` is the per-probe coordinate axis-0 float the session
            # already collapsed from each reading; ``per_layer_scores`` is the
            # per-layer coordinate map.  Both keep their existing wire shapes
            # ({name: float} / {layer: {name: float}}) for the un-updated webui.
            probes_blob = last.get("probes") if "scores" not in event else None
            if probes_blob:
                event["scores"] = probes_blob
            fallback_per_layer_blob = (
                last.get("per_layer_scores")
                if "per_layer_scores" not in event
                else None
            )
            if fallback_per_layer_blob:
                event["per_layer_scores"] = fallback_per_layer_blob

    # Rich channel: the full per-probe reading (coords + fraction + nearest)
    # for the latest token. ``probe_readings`` is the unified per-probe dict;
    # the token tap is the single owner
    # of live probe scoring, so event shaping never reaches into private capture
    # buffers or re-scores the token on the WebSocket path.
    with suppress(Exception):
        readings = None
        if payload is not None:
            readings = payload.get("probe_readings")
        if readings:
            event["probe_readings"] = {
                name: r.to_dict() for name, r in readings.items()
            }

    # Live J-lens workspace readout: the step's top-k lens tokens per selected
    # layer (``enable_live_lens``), stashed by the token tap alongside the
    # probe readings.  String layer keys to match ``per_layer_scores``' wire
    # shape; ``[token, score]`` pairs serialize as 2-arrays.
    with suppress(Exception):
        lens = payload.get("lens") if payload is not None else None
        if lens:
            event["lens_readout"] = {
                str(layer): [[tok, float(score)] for tok, score in pairs]
                for layer, pairs in lens.items()
            }
        # The layer-aggregated chip list riding the same step: ``[token,
        # strength, com, spread]`` 4-arrays, strength-descending (mean
        # band probability + probability-mass-weighted depth center of mass).
        agg = (
            payload.get("lens_aggregate") if payload is not None else None
        )
        if agg:
            event["lens_aggregate"] = [
                [tok, float(s), float(com), float(spread)]
                for tok, s, com, spread in agg
            ]

        sae = payload.get("sae") if payload is not None else None
        if sae:
            # ``max_act`` is the cached Neuronpedia maxActApprox (the strength
            # unit — clients render ``activation / max_act`` as the normalized
            # 0..1 strength); ``None`` until the metadata backfill lands.
            event["sae_readout"] = [
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
            ]

    return event
