"""Cross-branch joint logprobs (Phase 5 of docs/plans/logit-pass.md).

Given two assistant LoomNodes A and B that share a parent, replay each
branch token-by-token under the recipe stamped on that node and report,
for each aligned assistant-token position pair:

* ``lp_a_in_a`` / ``lp_b_in_b`` — chosen-token logprob under each
  branch's own distribution.  Mirrors what the engine already captures
  at generation time, recomputed here so the cross-evaluation and the
  self-evaluation use bit-identical math.
* ``lp_a_in_b`` / ``lp_b_in_a`` — chosen-token logprob under the *other*
  branch's distribution at the byte-aligned position.  Answers "what
  would B have given the token A picked here?" and vice versa.
* ``rank_changed`` — true iff the argmax token differs between the two
  distributions at this aligned position.  This is the canonical
  "steering shifted the head of the distribution, not just the
  argmax" signal.
* ``approx_kl`` — top-K-truncated KL(P_A || P_B), summed over the union
  of each side's top-K tokens.  The tail is unobserved, so this is
  documented as approximate signal not measurement (per Decision 5).

The route is fired lazily on NodeCompareDrawer open per Decision 9;
results cache on the session keyed by sorted ``(a_id, b_id)`` for the
session lifetime.  Tree mutations that rename / delete the involved
nodes invalidate the entries (see ``SaklasSession`` cache wiring).
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import torch

from saklas.core.generation import (
    GenerationConfig,
    _PenaltyState,
    _advance_no_cache_input,
    _effective_topk,
    _sampler_logprob_vector,
    build_chat_input,
    supports_thinking,
)
from saklas.core.loom_diff import per_token_diff
from saklas.core.sampling import SamplingConfig
from saklas.core.steering import Steering

if TYPE_CHECKING:  # avoid a hard import cycle at module load
    from saklas.core.session import SaklasSession


# Truncation budget for the approximate KL.  ~32 covers the practical
# mass at typical sampler temperatures; we don't try to estimate the
# tail because the engine isn't shipping it.
_KL_TOP_K = 32


@dataclass(frozen=True)
class JointLogprobRow:
    """One aligned-position record in a :class:`JointLogprobs` result.

    Indices are positions in each branch's *assistant-only* token list
    (i.e. relative to the divergence point), so they line up with the
    drawer's per-token row rendering.  Text fields carry the decoded
    token strings, ready for display without re-tokenization.
    """

    a_index: int
    b_index: int
    a_text: str
    b_text: str
    aligned: bool
    lp_a_in_a: float | None
    lp_b_in_b: float | None
    lp_a_in_b: float | None
    lp_b_in_a: float | None
    rank_changed: bool
    approx_kl: float | None


@dataclass(frozen=True)
class JointLogprobs:
    a_id: str
    b_id: str
    parent_id: str | None
    rows: tuple[JointLogprobRow, ...]
    n_rank1_changed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "a_id": self.a_id,
            "b_id": self.b_id,
            "parent_id": self.parent_id,
            "n_rank1_changed": self.n_rank1_changed,
            "rows": [
                {
                    "a_index": r.a_index,
                    "b_index": r.b_index,
                    "a_text": r.a_text,
                    "b_text": r.b_text,
                    "aligned": r.aligned,
                    "lp_a_in_a": r.lp_a_in_a,
                    "lp_b_in_b": r.lp_b_in_b,
                    "lp_a_in_b": r.lp_a_in_b,
                    "lp_b_in_a": r.lp_b_in_a,
                    "rank_changed": r.rank_changed,
                    "approx_kl": r.approx_kl,
                }
                for r in self.rows
            ],
        }


# ---------------------------------------------------------------------------
# Pure-math core (tested in isolation; see tests/test_joint_logprobs.py)
# ---------------------------------------------------------------------------


def _approx_kl_topk(
    logp_a: torch.Tensor,  # [V] fp32 sampler logprobs
    logp_b: torch.Tensor,  # [V] fp32 sampler logprobs
    top_k: int,
) -> float:
    """Truncated KL(P_A || P_B) over the union of each side's top-K.

    Within the union, contributions to ``Σ P_A(t) (log P_A(t) − log P_B(t))``
    are summed exactly; the tail (tokens neither side ranked in its top-K)
    is dropped — that's the "approx" in ``approx_kl``.  For typical
    post-sampler distributions K=32 captures the bulk; the residual is
    well under 0.1 nats in practice.
    """
    vocab = logp_a.shape[-1]
    k = min(top_k, vocab)
    top_a = torch.topk(logp_a, k).indices
    top_b = torch.topk(logp_b, k).indices
    union = torch.unique(torch.cat([top_a, top_b]))
    la = logp_a.index_select(0, union)
    lb = logp_b.index_select(0, union)
    support = torch.isfinite(la)
    if not bool(support.any().item()):
        return 0.0
    if bool((~torch.isfinite(lb[support])).any().item()):
        return float("inf")
    pa = la[support].exp()
    diff = la[support] - lb[support]
    return float((pa * diff).sum().item())


def _finite_float(value: torch.Tensor | float) -> float | None:
    """Convert finite tensor/float values to JSON-safe floats."""
    out = float(value.item()) if isinstance(value, torch.Tensor) else float(value)
    return out if math.isfinite(out) else None


_LogprobRows = torch.Tensor | dict[int, torch.Tensor]


def _logprob_row(rows: _LogprobRows, row_idx: int) -> torch.Tensor | None:
    if isinstance(rows, dict):
        return rows.get(row_idx)
    if 0 <= row_idx < rows.shape[0]:
        return rows[row_idx]
    return None


def _logprob_value(rows: _LogprobRows, row_idx: int, token_id: int) -> float | None:
    row = _logprob_row(rows, row_idx)
    if row is None or token_id < 0 or token_id >= row.shape[-1]:
        return None
    return _finite_float(row[token_id])


def _compute_rows(
    logp_a: _LogprobRows,        # sampler-renormalized logprobs by predictor row
    logp_b: _LogprobRows,
    token_ids_a: list[int],      # full sequence (prefix + assistant) for A
    token_ids_b: list[int],
    token_strs_a: list[str],     # decoded text per id (display) for A's full seq
    token_strs_b: list[str],
    prefix_len: int,             # shared prefix length (in tokens)
    *,
    kl_top_k: int = _KL_TOP_K,
) -> list[JointLogprobRow]:
    """Pure-tensor inner loop — no session / tokenizer / IO.

    Aligns A's assistant tail and B's assistant tail via the shared
    :func:`per_token_diff` byte-offset walker, then looks up logprobs
    from the precomputed sampler-logprob tables.  Position ``prefix_len + i``
    in the full sequence is *predicted* by the logits at position
    ``prefix_len + i - 1`` — that's the index we read from on each row.
    """
    assistant_ids_a = token_ids_a[prefix_len:]
    assistant_ids_b = token_ids_b[prefix_len:]
    assistant_strs_a = token_strs_a[prefix_len:]
    assistant_strs_b = token_strs_b[prefix_len:]

    # ``per_token_diff`` walks byte-offset alignment over the per-token
    # display strings; we feed it the assistant tail so ``a_index`` /
    # ``b_index`` come back in the same space we'll surface to the UI.
    spans = per_token_diff(assistant_strs_a, assistant_strs_b)

    rows: list[JointLogprobRow] = []
    for sp in spans:
        a_idx = sp.a_index
        b_idx = sp.b_index
        # Logits *at* full-sequence position k predict the token at k+1,
        # so to score the token at full-position prefix_len+i we read
        # logp[prefix_len + i - 1].  ``max(0, …)`` guards the (degenerate)
        # case where prefix_len is 0 and i is 0 — fall back to position
        # 0's logits, which are conditioned on nothing and will produce
        # the unigram-like prior.
        pa_pos = max(0, prefix_len + a_idx - 1)
        pb_pos = max(0, prefix_len + b_idx - 1)

        # Self-evaluation: chosen logprob under own distribution.
        lp_a_in_a: float | None = None
        lp_b_in_b: float | None = None
        if 0 <= a_idx < len(assistant_ids_a):
            lp_a_in_a = _logprob_value(logp_a, pa_pos, assistant_ids_a[a_idx])
        if 0 <= b_idx < len(assistant_ids_b):
            lp_b_in_b = _logprob_value(logp_b, pb_pos, assistant_ids_b[b_idx])

        # Cross-evaluation: only meaningful when the positions actually
        # align (byte-equal context up to here).  On divergent rows the
        # cross-prob is ambiguous (which prior position do we score
        # against?) so we leave it null.
        lp_a_in_b: float | None = None
        lp_b_in_a: float | None = None
        rank_changed = False
        approx_kl: float | None = None
        if sp.aligned and 0 <= a_idx < len(assistant_ids_a) and 0 <= b_idx < len(assistant_ids_b):
            row_a = _logprob_row(logp_a, pa_pos)
            row_b = _logprob_row(logp_b, pb_pos)
            if row_a is not None and row_b is not None:
                lp_a_in_b = _logprob_value(
                    logp_b, pb_pos, assistant_ids_a[a_idx],
                )
                lp_b_in_a = _logprob_value(
                    logp_a, pa_pos, assistant_ids_b[b_idx],
                )
                # Rank-1 change: does the argmax differ at this aligned
                # position?  Cheap signal — one ``argmax`` per side.
                argmax_a = int(row_a.argmax().item())
                argmax_b = int(row_b.argmax().item())
                rank_changed = argmax_a != argmax_b
                approx_kl = _finite_float(_approx_kl_topk(
                    row_a, row_b, kl_top_k,
                ))

        rows.append(JointLogprobRow(
            a_index=a_idx,
            b_index=b_idx,
            a_text=sp.a_text,
            b_text=sp.b_text,
            aligned=sp.aligned,
            lp_a_in_a=lp_a_in_a,
            lp_b_in_b=lp_b_in_b,
            lp_a_in_b=lp_a_in_b,
            lp_b_in_a=lp_b_in_a,
            rank_changed=rank_changed,
            approx_kl=approx_kl,
        ))
    return rows


# ---------------------------------------------------------------------------
# IO wrapper — talks to the session, tokenizer, model
# ---------------------------------------------------------------------------


def _shared_prefix_len(ids_a: list[int], ids_b: list[int]) -> int:
    """Longest common prefix length between two token-id lists.

    Both branches share their parent's chat-template prefix verbatim,
    so this lands at the divergence point — exactly where the assistant
    content starts to differ.  Used instead of re-tokenizing the parent
    prefix separately to dodge template-boundary surprises.
    """
    n = min(len(ids_a), len(ids_b))
    i = 0
    while i < n and ids_a[i] == ids_b[i]:
        i += 1
    return i


@dataclass(frozen=True)
class _ReplayBranch:
    prompt_ids: list[int]
    response_ids: list[int]
    token_ids: list[int]
    token_strs: list[str]
    thinking_ids: list[int]
    sampling: SamplingConfig
    steering: Steering | None


def _decode_each(tokenizer: Any, ids: list[int]) -> list[str]:
    # Batch-decode-per-id rather than ``decode(ids)`` to keep the
    # per-position list aligned 1:1 with the id list.  Coerce to ``str``
    # because some tokenizer signatures are typed loosely.
    return [str(tokenizer.decode([tid])) for tid in ids]


def _row_token_ids(rows: list[dict[str, Any]] | None) -> tuple[list[int], list[str]]:
    ids: list[int] = []
    texts: list[str] = []
    for row in rows or []:
        try:
            tid = int(row.get("token_id"))  # pyright: ignore[reportArgumentType]  # None raises TypeError, caught below
        except (TypeError, ValueError):
            continue
        if tid < 0:
            # Buffered partial UTF-8 rows do not correspond to a single
            # model token and cannot be forced through replay.
            continue
        ids.append(tid)
        text = row.get("text")
        texts.append(str(text) if text is not None else "")
    return ids, texts


def _sampling_from_recipe(recipe: Any) -> SamplingConfig:
    sampling = getattr(recipe, "sampling", None)
    if not isinstance(sampling, SamplingConfig):
        sampling = SamplingConfig()
    seed = getattr(recipe, "seed", None)
    if seed is not None and sampling.seed is None:
        sampling = replace(sampling, seed=int(seed))
    return sampling


def _supports_thinking_safe(tokenizer: Any) -> bool:
    try:
        return bool(supports_thinking(tokenizer))
    except Exception:
        return False


def _compose_replay_config(
    session: "SaklasSession",
    sampling: SamplingConfig,
) -> GenerationConfig:
    compose = getattr(session, "_compose_gen_config", None)
    if compose is not None:
        return compose(sampling)

    base = getattr(session, "config", None)
    if isinstance(base, GenerationConfig):
        cfg = base
    else:
        cfg = GenerationConfig(
            max_new_tokens=int(getattr(base, "max_new_tokens", 1024)),
            temperature=float(getattr(base, "temperature", 1.0)),
            top_p=float(getattr(base, "top_p", 0.9)),
            top_k=getattr(base, "top_k", None),
            system_prompt=getattr(base, "system_prompt", None),
        )
    overrides: dict[str, Any] = {}
    if sampling.temperature is not None:
        overrides["temperature"] = sampling.temperature
    if sampling.top_p is not None:
        overrides["top_p"] = sampling.top_p
    if sampling.top_k is not None:
        overrides["top_k"] = sampling.top_k
    if sampling.max_tokens is not None:
        overrides["max_new_tokens"] = sampling.max_tokens
    return replace(cfg, **overrides) if overrides else cfg


def _branch_inputs(session: "SaklasSession", node_id: str) -> _ReplayBranch:
    tree = session.tree
    tokenizer = session.tokenizer
    node = tree.nodes[node_id]
    recipe = getattr(node, "recipe", None)
    sampling = _sampling_from_recipe(recipe)
    steering = Steering.from_value(
        getattr(recipe, "steering", None),
        profile_names=set(getattr(session, "_profiles", {})),
    )

    stamped_thinking = getattr(recipe, "thinking", None)
    if stamped_thinking is None:
        if steering is not None and steering.thinking is not None:
            thinking = bool(steering.thinking)
        else:
            thinking = _supports_thinking_safe(tokenizer)
    else:
        thinking = bool(stamped_thinking)

    parent_id = getattr(node, "parent_id", None)
    parent = tree.nodes.get(parent_id) if parent_id is not None else None
    if parent is not None and parent.role == "user":
        prompt_messages = tree.messages_for(parent.id)
    else:
        prompt_messages = tree.messages_for(node_id)
    system_prompt = getattr(getattr(session, "config", None), "system_prompt", None) or None
    prompt_input = build_chat_input(
        tokenizer,
        prompt_messages,
        system_prompt=system_prompt,
        thinking=thinking,
        add_generation_prompt=True,
    )
    prompt_ids = [int(t) for t in prompt_input[0].tolist()]

    response_ids, response_texts = _row_token_ids(getattr(node, "tokens", None))
    thinking_ids, _thinking_texts = _row_token_ids(
        getattr(node, "thinking_tokens", None)
    )

    if not response_ids:
        full_input = build_chat_input(
            tokenizer,
            tree.messages_for(node_id),
            system_prompt=system_prompt,
            thinking=thinking,
            add_generation_prompt=False,
        )
        full_ids = [int(t) for t in full_input[0].tolist()]
        cut = _shared_prefix_len(prompt_ids, full_ids)
        prompt_ids = full_ids[:cut]
        response_ids = full_ids[cut:]
        response_texts = _decode_each(tokenizer, response_ids)
    elif len(response_texts) != len(response_ids) or any(t == "" for t in response_texts):
        response_texts = _decode_each(tokenizer, response_ids)

    prompt_strs = _decode_each(tokenizer, prompt_ids)
    return _ReplayBranch(
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        token_ids=prompt_ids + response_ids,
        token_strs=prompt_strs + response_texts,
        thinking_ids=thinking_ids,
        sampling=sampling,
        steering=steering,
    )


def _call_model(model: Any, **kwargs: Any) -> Any:
    try:
        return model(**kwargs)
    except TypeError as e:
        msg = str(e)
        if (
            "attention_mask" not in msg
            and "past_key_values" not in msg
            and "cache_position" not in msg
        ):
            raise
        return model(input_ids=kwargs["input_ids"], use_cache=kwargs.get("use_cache", False))


def _replay_branch_logprobs(
    session: "SaklasSession",
    branch: _ReplayBranch,
) -> dict[int, torch.Tensor]:
    """Force-replay one branch and return visible response-row logprobs.

    The result is keyed by predictor row in the full branch sequence.  Only
    response-token rows are retained; prompt rows and thinking-only rows are
    intentionally absent so long branches do not allocate dense
    ``[n_rows, vocab]`` tensors filled mostly with ``-inf``.
    """
    model = session.model
    try:
        device = next(model.parameters()).device
    except StopIteration:  # pragma: no cover - defensive for odd test doubles
        device = torch.device("cpu")

    forced_ids = branch.thinking_ids + branch.response_ids
    if not forced_ids:
        return {}
    if not branch.prompt_ids:
        raise ValueError("joint-logprob replay requires a non-empty prompt")

    config = _compose_replay_config(session, branch.sampling)
    logit_bias = branch.sampling.logit_bias
    presence_penalty = branch.sampling.presence_penalty
    frequency_penalty = branch.sampling.frequency_penalty
    use_penalties = presence_penalty != 0.0 or frequency_penalty != 0.0
    penalty_state = _PenaltyState(
        max(len(forced_ids), 1), device, torch.float32,
    ) if use_penalties else None

    bias_idx: torch.Tensor | None = None
    bias_val: torch.Tensor | None = None
    if logit_bias:
        bias_idx = torch.tensor(list(logit_bias.keys()), dtype=torch.long, device=device)
        bias_val = torch.tensor(list(logit_bias.values()), dtype=torch.float32, device=device)

    ctx = getattr(getattr(session, "_steering", None), "ctx", None)
    steering_cm = contextlib.nullcontext()
    if branch.steering is not None and branch.steering.alphas and hasattr(session, "steering"):
        steering_cm = session.steering(branch.steering)

    row_logps: dict[int, torch.Tensor] = {}
    vocab_size: int | None = None

    with steering_cm:
        if ctx is not None:
            ctx.reset()
        begin_capture = getattr(session, "_begin_capture", None)
        end_capture = getattr(session, "_end_capture", None)
        monitor = getattr(session, "_monitor", None)
        if begin_capture is not None:
            begin_capture(widen=False)
        if monitor is not None and hasattr(monitor, "begin_live"):
            monitor.begin_live()
        try:
            needs_gating = (
                bool(session._steering_needs_probe_gating())
                if hasattr(session, "_steering_needs_probe_gating")
                else False
            )
            gating_callback = (
                session._build_gating_score_callback()
                if needs_gating and hasattr(session, "_build_gating_score_callback")
                else None
            )

            current_input = torch.tensor(
                [branch.prompt_ids],
                dtype=torch.long,
                device=device,
            )
            forced_tensor = torch.tensor(
                [forced_ids],
                dtype=torch.long,
                device=device,
            )
            attn_mask_buf = torch.ones(
                (1, current_input.shape[1] + max(len(forced_ids), 1)),
                dtype=torch.long,
                device=device,
            )
            past_key_values = None
            no_cache_mode = False
            no_cache_buf: torch.Tensor | None = None
            no_cache_len = int(current_input.shape[1])
            prefill = True

            def _advance_current_input(next_token: torch.Tensor) -> None:
                nonlocal current_input, no_cache_buf, no_cache_len
                current_input, no_cache_buf, no_cache_len = _advance_no_cache_input(
                    next_token,
                    current_input=current_input,
                    no_cache_buf=no_cache_buf,
                    no_cache_len=no_cache_len,
                    no_cache_mode=no_cache_mode,
                    max_extra=len(forced_ids),
                )

            with torch.inference_mode():
                for forced_idx, token_id in enumerate(forced_ids):
                    if ctx is not None:
                        ctx.is_prefill = prefill
                        ctx.thinking = forced_idx < len(branch.thinking_ids)
                        ctx.gen_step = forced_idx

                    kwargs: dict[str, Any] = {
                        "input_ids": current_input,
                        "use_cache": True,
                    }
                    if past_key_values is not None and not no_cache_mode:
                        kwargs["past_key_values"] = past_key_values
                    if prefill or no_cache_mode:
                        kwargs["attention_mask"] = attn_mask_buf[
                            :, :current_input.shape[1]
                        ]

                    outputs = _call_model(model, **kwargs)
                    prefill = False

                    if gating_callback is not None and ctx is not None:
                        ctx.probe_scores = gating_callback()

                    if not no_cache_mode:
                        past_key_values = getattr(outputs, "past_key_values", None)
                        if past_key_values is None and current_input.shape[1] > 1:
                            no_cache_mode = True

                    logits = outputs.logits[:, -1, :]
                    logits.nan_to_num_(nan=0.0, posinf=100.0, neginf=-100.0)
                    logits.clamp_(-100.0, 100.0)

                    if penalty_state is not None:
                        penalty_state.apply(
                            logits,
                            presence_penalty=presence_penalty,
                            frequency_penalty=frequency_penalty,
                        )
                    if bias_idx is not None:
                        # bias_val is always set when bias_idx is set (both come from logit_bias)
                        assert bias_val is not None
                        logits[0, bias_idx] += bias_val.to(logits.dtype)

                    vocab_size = int(logits.shape[-1])
                    topk_k = _effective_topk(config, vocab_size)
                    logp = _sampler_logprob_vector(logits, config, topk_k)

                    if forced_idx >= len(branch.thinking_ids):
                        response_idx = forced_idx - len(branch.thinking_ids)
                        visible_pos = len(branch.prompt_ids) + response_idx - 1
                        row_logps[visible_pos] = logp.detach().to("cpu")

                    if penalty_state is not None:
                        penalty_state.add(token_id)

                    next_token = forced_tensor[:, forced_idx:forced_idx + 1]
                    _advance_current_input(next_token)
        finally:
            if end_capture is not None:
                end_capture()
            if monitor is not None and hasattr(monitor, "end_live"):
                monitor.end_live()

    return row_logps if vocab_size is not None else {}


def compute_joint_logprobs(
    session: "SaklasSession",
    a_id: str,
    b_id: str,
) -> JointLogprobs:
    """Run cross-evaluation between two assistant sibling nodes.

    Builds each branch's prompt through the chat template, force-replays
    its stored response tokens under that node's recipe, and assembles
    per-aligned-position records.  Caller is responsible for holding
    ``session.lock`` — model forwards must serialize against any
    concurrent generation on the same session.

    Raises ``KeyError`` when either node id is unknown to the tree.
    Returns an empty-rows :class:`JointLogprobs` when the branches share
    no divergent assistant tokens (e.g. one node is empty), which is the
    least-surprising shape for the drawer to render.
    """
    tree = session.tree
    a_node = tree.nodes[a_id]
    b_node = tree.nodes[b_id]
    parent_id = a_node.parent_id if a_node.parent_id == b_node.parent_id else None
    branch_a = _branch_inputs(session, a_id)
    branch_b = _branch_inputs(session, b_id)
    ids_a = branch_a.token_ids
    ids_b = branch_b.token_ids
    prefix_len = _shared_prefix_len(ids_a, ids_b)

    # If neither side has any assistant tokens past the prefix, return
    # an empty result — nothing to align.
    if prefix_len >= len(ids_a) and prefix_len >= len(ids_b):
        return JointLogprobs(
            a_id=a_id, b_id=b_id, parent_id=parent_id,
            rows=(), n_rank1_changed=0,
        )

    logp_a = _replay_branch_logprobs(session, branch_a)
    logp_b = _replay_branch_logprobs(session, branch_b)

    rows = _compute_rows(
        logp_a, logp_b, ids_a, ids_b,
        branch_a.token_strs, branch_b.token_strs, prefix_len,
    )
    n_changed = sum(1 for r in rows if r.aligned and r.rank_changed)
    return JointLogprobs(
        a_id=a_id, b_id=b_id, parent_id=parent_id,
        rows=tuple(rows), n_rank1_changed=n_changed,
    )


# ---------------------------------------------------------------------------
# Cache helpers — symmetric key so (a, b) and (b, a) dedupe
# ---------------------------------------------------------------------------


def _cache_key(a_id: str, b_id: str) -> tuple[str, str]:
    """Symmetric (a, b) ↔ (b, a) cache key.

    Sorted so the drawer hits the same entry regardless of which node
    the user right-clicked first.  Result layout still respects the
    caller's ``(a, b)`` orientation — see :func:`reorient_for_request`.
    """
    return (a_id, b_id) if a_id <= b_id else (b_id, a_id)


def reorient_for_request(
    result: JointLogprobs, requested_a_id: str, requested_b_id: str,
) -> JointLogprobs:
    """Flip the result's a/b labelling to match the caller's request.

    Cache stores under the sorted key, so a request for ``(B, A)`` after
    ``(A, B)`` was already computed needs the columns swapped before
    the drawer renders them.  Pure metadata work — no recomputation.
    """
    if (result.a_id, result.b_id) == (requested_a_id, requested_b_id):
        return result
    swapped_rows = tuple(
        JointLogprobRow(
            a_index=r.b_index,
            b_index=r.a_index,
            a_text=r.b_text,
            b_text=r.a_text,
            aligned=r.aligned,
            lp_a_in_a=r.lp_b_in_b,
            lp_b_in_b=r.lp_a_in_a,
            lp_a_in_b=r.lp_b_in_a,
            lp_b_in_a=r.lp_a_in_b,
            rank_changed=r.rank_changed,
            approx_kl=r.approx_kl,
        )
        for r in result.rows
    )
    return JointLogprobs(
        a_id=requested_a_id,
        b_id=requested_b_id,
        parent_id=result.parent_id,
        rows=swapped_rows,
        n_rank1_changed=result.n_rank1_changed,
    )


__all__ = [
    "JointLogprobRow",
    "JointLogprobs",
    "compute_joint_logprobs",
    "_compute_rows",
    "_approx_kl_topk",
    "_shared_prefix_len",
    "_cache_key",
    "reorient_for_request",
    "_KL_TOP_K",
]
