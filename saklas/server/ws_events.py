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

    scores_blob = payload.get("scores")
    if scores_blob:
        event["scores"] = {
            name: round(float(value), 6)
            for name, value in scores_blob.items()
        }
    per_layer_blob = (
        payload.get("per_layer_scores")
    )
    if per_layer_blob:
        event["per_layer_scores"] = per_layer_blob

    # Rich channel: the full per-probe reading (coords + fraction + nearest)
    # for the latest token. ``probe_readings`` is the unified per-probe dict;
    # the token tap is the single owner
    # of live probe scoring, so event shaping never reaches into private capture
    # buffers or re-scores the token on the WebSocket path.
    readings = payload.get("probe_readings")
    if readings:
        event["probe_readings"] = {
            name: r.to_dict() for name, r in readings.items()
        }

    # Live J-lens workspace readout: the step's top-k lens tokens per selected
    # layer (``enable_live_lens``), stashed by the token tap alongside the
    # probe readings.  String layer keys to match ``per_layer_scores``' wire
    # shape; ``[token, score]`` pairs serialize as 2-arrays.
    lens = payload.get("lens")
    if lens:
        event["lens_readout"] = {
            str(layer): [[tok, float(score)] for tok, score in pairs]
            for layer, pairs in lens.items()
        }
        # The layer-aggregated chip list riding the same step: ``[token,
        # strength, com, spread]`` 4-arrays, strength-descending (mean
        # band probability + probability-mass-weighted depth center of mass).
    agg = payload.get("lens_aggregate")
    if agg:
        event["lens_aggregate"] = [
            [tok, float(s), float(com), float(spread)]
            for tok, s, com, spread in agg
        ]

    sae = payload.get("sae")
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
