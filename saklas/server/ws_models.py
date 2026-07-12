"""WebSocket request schemas and serializers for native saklas streams."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

from saklas.core.results import GenerationResult, RunSet
from saklas.core.sampling import SamplingConfig
from saklas.core.session import SaklasSession


class WSSamplingParams(BaseModel):
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    seed: int | None = None
    stop: list[str] | None = None
    logit_bias: dict[int, float] | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    return_top_k: int = 0
    return_probe_readings: bool = True
    persist_per_layer_scores: bool = False
    persist_subspace_coords: bool = False
    user_role: str | None = None
    assistant_role: str | None = None


class WSGenerateMessage(BaseModel):
    type: str
    input: Any = None
    steering: str | None = None
    sampling: WSSamplingParams | None = None
    thinking: bool | None = None
    stateless: bool = True
    raw: bool = False
    parent_node_id: str | None = None
    n: int = 1
    recipe_override: Any = None
    fork_node_id: str | None = None
    fork_raw_index: int | None = None
    fork_alt_token_id: int | None = None
    prefill_node_id: str | None = None
    prefill_text: str | None = None
    commit_role: Literal["user", "assistant"] | None = None
    commit_text: str | None = None
    # Optional committed thinking block riding a commit (any seat) —
    # rendered through the family think delimiters by the scene
    # stitcher; rejected with 400 when the family can't carry it.
    commit_thinking: str | None = None
    # Cast model: which seat the generated turn occupies.  ``"user"``
    # renders the generation prompt as a user-seat header and lands the
    # node with ``role="user"`` + a stamped recipe (generated is
    # provenance, not a seat).  ``None`` = assistant (the classic flow).
    # Pair with ``input: null`` for a continue — no committed turn, the
    # model speaks next from the current leaf (a/a and u/u sequences).
    generate_seat: Literal["user", "assistant"] | None = None


def build_sampling(body: WSSamplingParams | None) -> SamplingConfig | None:
    if body is None:
        return None
    stop = tuple(body.stop) if body.stop else None
    return SamplingConfig(
        temperature=body.temperature,
        top_p=body.top_p,
        top_k=body.top_k,
        max_tokens=body.max_tokens,
        seed=body.seed,
        stop=stop,
        logit_bias=body.logit_bias,
        presence_penalty=body.presence_penalty or 0.0,
        frequency_penalty=body.frequency_penalty or 0.0,
        return_top_k=body.return_top_k,
        return_probe_readings=bool(body.return_probe_readings),
        user_role=(body.user_role or None),
        assistant_role=(body.assistant_role or None),
        persist_per_layer_scores=bool(body.persist_per_layer_scores),
        persist_subspace_coords=bool(body.persist_subspace_coords),
    )


def result_to_json(result: GenerationResult | RunSet | None) -> dict[str, Any]:
    if result is None:
        return {}
    prompt_tokens = getattr(result, "prompt_tokens", 0) or 0
    completion = getattr(result, "token_count", 0) or 0
    return {
        "text": getattr(result, "text", ""),
        "tokens": completion,
        "finish_reason": getattr(result, "finish_reason", "stop"),
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion,
            "total_tokens": prompt_tokens + completion,
        },
    }


def per_token_probes(
    session: SaklasSession, n_tokens: int,
) -> list[dict[str, Any]]:
    scores = session.last_per_token_scores
    if not scores:
        return []
    return [
        {
            "token_idx": i,
            "probes": {
                name: float(vals[i])
                for name, vals in scores.items()
                if i < len(vals)
            },
        }
        for i in range(n_tokens)
    ]
