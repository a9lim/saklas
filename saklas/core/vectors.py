"""Extraction, saving, and loading of activation steering/probe vectors."""

from __future__ import annotations

import functools
import json
import logging
import warnings
from collections.abc import Sequence
from contextlib import suppress
from importlib import resources as _resources
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

log = logging.getLogger(__name__)


# Default chunk size for the batched capture path (:func:`_encode_and_capture_all_batch`
# and its callers ``compute_node_centroid`` / ``compute_neutral_activations``).  One
# forward over up to this many (prompt, response) pairs replaces that many sequential
# batch-1 forwards — the dominant extraction-capture cost.  Sized conservatively so a
# chunk's attention working set (``B · heads · T²``) stays comfortable on a single
# 24 GB GPU at the A2 response cap; raise it on a roomier device for fewer forwards.
_CAPTURE_BATCH = 16


def _capture_all_hidden_states(
    model: torch.nn.Module,
    layers: torch.nn.ModuleList,
    input_ids: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    pool_index: "int | torch.Tensor | None" = None,
):
    """Run one forward pass capturing hidden states at ALL layers.

    Uses ``use_cache=False`` to avoid polluting any persistent KV cache.
    ``attention_mask`` (when supplied) is forwarded so right-padded batches
    don't attend to pad tokens.

    ``pool_index`` selects what is retained per layer:

    - ``None`` — the full ``(B, seq, dim)`` hidden state (the default).
    - ``int`` — the single-sequence (``B == 1``) last-content pool: a ``(dim,)``
      fp32 vector at that position.
    - ``Tensor`` ``(B,)`` — the **per-row** batched pool: a ``(B, dim)`` fp32
      stack gathered at each row's position *inside the hook*, so only ``(B, D)``
      is retained per layer rather than the full ``(B, T, D)``.

    Returns a dict mapping layer index to the retained tensor.
    """
    captured_hidden: dict[int, torch.Tensor] = {}
    per_row = isinstance(pool_index, torch.Tensor)
    # Precompute the gather index tensors once (not per layer/hook fire); both
    # are set iff ``per_row`` (the assert in the hook narrows them back).
    _rows = (
        torch.arange(input_ids.shape[0], device=input_ids.device)
        if per_row else None
    )
    _pos = pool_index.to(input_ids.device) if isinstance(pool_index, torch.Tensor) else None

    def _make_hook(idx: int):
        def _hook(module: torch.nn.Module, input: Any, output: Any):
            h = output if isinstance(output, torch.Tensor) else output[0]
            if per_row:
                assert _pos is not None and _rows is not None  # set iff per_row
                pos = _pos.clamp(max=h.shape[1] - 1)
                # Advanced indexing returns a fresh tensor (a copy), so the
                # gathered (B, D) doesn't alias h — h is free to be released as
                # the forward proceeds, keeping peak memory at one layer's
                # (B, T, D) rather than every layer's.
                pooled = h[_rows, pos, :].detach()
                captured_hidden[idx] = (
                    pooled.to(torch.float32)
                    if pooled.dtype != torch.float32
                    else pooled
                )
                return
            if pool_index is not None:
                pos = min(max(int(pool_index), 0), h.shape[1] - 1)
                pooled = h[0, pos, :].detach()
                captured_hidden[idx] = (
                    pooled.to(torch.float32)
                    if pooled.dtype != torch.float32
                    else pooled.clone()
                )
                return
            # No .clone() — with use_cache=False and inference_mode() the
            # residual-stream tensors are fresh allocations at each layer
            # boundary (residual add produces a new tensor).  Detach severs
            # the autograd graph reference so the rest of the forward pass
            # can't invalidate the data.
            captured_hidden[idx] = h.detach()
        return _hook

    handles = [layers[idx].register_forward_hook(_make_hook(idx)) for idx in range(len(layers))]
    try:
        with torch.inference_mode():
            if attention_mask is not None:
                model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
            else:
                model(input_ids=input_ids, use_cache=False)
        # Single sync after the full forward pass — lazy backends (MPS)
        # may not have materialised tensor data yet.
        if input_ids.device.type == "mps":
            torch.mps.synchronize()
    finally:
        for h in handles:
            h.remove()

    return captured_hidden


def special_token_ids(tokenizer: Any) -> set[int]:
    """Token ids that are structural rather than content.

    The canonical "non-content" set every last-content-token pooling site
    shares: the tokenizer's ``all_special_ids`` *plus* everything in
    ``added_tokens_encoder``.  The second arm matters for tokenizers that
    don't promote chat-boundary markers to ``all_special_ids`` (talkie's
    ``<|user|>``/``<|end|>``/``<|assistant|>`` are added tokens but not
    "special") — without it, pooling lands on a structural turn marker
    whose hidden state is disconnected from the content.
    """
    skip = set(getattr(tokenizer, "all_special_ids", []) or [])
    added = getattr(tokenizer, "added_tokens_encoder", None) or {}
    skip.update(int(v) for v in added.values())
    return skip


def last_content_index(token_ids: Sequence[int], tokenizer: Any) -> int:
    """Index of the last non-special token in ``token_ids``.

    Walks backward from the final position past every id in
    :func:`special_token_ids` (``all_special_ids`` + ``added_tokens_encoder``),
    flooring at 0 so a sequence that is entirely special tokens still
    yields a valid index.  This is the single definition of "last content
    token" shared by extraction (:func:`_encode_and_capture_all`), the
    aggregate vector probe (:meth:`Monitor.score_per_token`), the
    incremental scoring path (``session._score_incremental``), and the
    manifold aggregate (:meth:`Monitor.score_aggregate`) — every
    reported single-state value pools from here so the discipline can't
    drift per-site.
    """
    idx = len(token_ids) - 1
    if idx < 0:
        return 0
    skip = special_token_ids(tokenizer)
    if skip:
        while idx > 0 and int(token_ids[idx]) in skip:
            idx -= 1
    return idx


# Shared one-paragraph length directive — a system prompt applied to BOTH
# generation (``session.generate_responses`` / ``generate_neutral_responses``)
# AND capture (every ``_encode_and_capture_all`` site), for node and neutral
# corpora alike.  Identical and symmetric across all four sites, so it is
# common-mode: it cancels in ``node − neutral`` (and ``concept − ν``) instead of
# leaking a "be brief" offset into the extracted directions, and it matches the
# capture framing to the generation framing so the pooled responses are not
# out-of-distribution.  Without it the model rambles past the token cap and
# truncates mid-thought.  (For a node, generation also prepends the persona to
# this system prompt; capture keeps only the directive — the persona stays
# generation-only, its signal carried in the response text, pooled in
# standard-assistant space.)
_LENGTH_DIRECTIVE = "Answer in one short paragraph."


def _encode_and_capture_all(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    response: str,
    layers: torch.nn.ModuleList,
    device: torch.device,
    *,
    role: str | None = None,
    model_type: str | None = None,
    system_msg: str = _LENGTH_DIRECTIVE,
):
    """Capture the last-content-token hidden state per layer for a turn pair, fp32.

    Conversational (4.0 / A2) capture: the corpus item is an assistant
    *response* to a fixed baseline *prompt*, so extraction pools the model in
    its real generation regime — ``[system: directive, user: prompt, assistant:
    response]`` with the standard assistant label.  ``system_msg`` defaults to
    the shared :data:`_LENGTH_DIRECTIVE` so capture matches the framing the
    corpus was generated under (and cancels as common-mode against the neutral
    baseline); pass ``""`` to drop the system turn.  ``role`` overrides the
    assistant label only when an explicit per-node role is set (the
    persona-baselined fit); ``role=None`` is the swap-back default.

    Pools from the response's last non-special token — chat templates append
    trailing markers (Llama's <|eot_id|>, Gemma's <end_of_turn>, Qwen's
    <|im_end|>) whose hidden states are disconnected from content; the last
    content token is the attention-weighted summary the model uses for
    next-token prediction.

    Returns:
        dict mapping layer_idx -> pooled vector (dim,) in fp32.
    """
    ids, content_end = _render_and_tokenize_for_capture(
        tokenizer, prompt, response, device,
        role=role, model_type=model_type, system_msg=system_msg,
    )
    hidden_per_layer = _capture_all_hidden_states(
        model, layers, ids, pool_index=content_end,
    )
    return {
        idx: h if h.ndim == 1 else h[0, min(content_end, h.shape[1] - 1)].float()
        for idx, h in hidden_per_layer.items()
    }


def _encode_and_capture_all_batch(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: Sequence[str],
    responses: Sequence[str],
    layers: torch.nn.ModuleList,
    device: torch.device,
    *,
    role: str | None = None,
    model_type: str | None = None,
    system_msg: str = _LENGTH_DIRECTIVE,
) -> dict[int, torch.Tensor]:
    """Batched conversational capture — one forward over a chunk of pairs.

    The batched sibling of :func:`_encode_and_capture_all`: each
    ``[system, user: prompt, assistant: response]`` turn is rendered + tokenized
    through the same :func:`_render_and_tokenize_for_capture` front half, then
    the rows are **right-padded** to a common length and run through ONE forward
    pass.  Right-padding is the correct choice for capture (not generation):
    causal attention at a row's last-content position only sees positions to its
    *left*, all real tokens, so the pooled hidden state is identical to the
    unpadded single-sequence forward — the per-row ``content_end`` indices need
    no adjustment, and an ``attention_mask`` keeps the trailing pads out of every
    real token's context.  Pooling happens **inside the capture hook** (per-row
    gather at ``content_end``), so only ``(B, D)`` is retained per layer, never
    the full ``(B, T, D)``.

    The attention mask is built only when the chunk actually has ragged lengths;
    a uniform-length (or ``B == 1``) chunk runs full attention, bit-identical to
    the single-pair path.

    Returns ``{layer_idx: (B, D)}`` in fp32 (on the model device).
    """
    if len(prompts) != len(responses):
        raise ValueError(
            "batched capture needs len(prompts) == len(responses) "
            f"({len(prompts)} != {len(responses)})"
        )
    rendered = [
        _render_and_tokenize_for_capture(
            tokenizer, prompt, response, device,
            role=role, model_type=model_type, system_msg=system_msg,
        )
        for prompt, response in zip(prompts, responses, strict=True)
    ]
    seqs = [ids[0] for ids, _ in rendered]              # each (L_i,) on device
    ends = [content_end for _, content_end in rendered]
    lengths = [int(s.shape[0]) for s in seqs]
    batch = len(seqs)
    max_len = max(lengths)

    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None) or 0
    input_ids = torch.full(
        (batch, max_len), int(pad_id), dtype=seqs[0].dtype, device=device,
    )
    for i, seq in enumerate(seqs):
        input_ids[i, : lengths[i]] = seq                # right-pad
    # An attention mask is only needed when padding is actually present; a
    # ragged chunk masks the pads, a uniform/B=1 chunk runs full attention
    # (matching the single-pair path exactly).
    if min(lengths) != max_len:
        attn = torch.zeros((batch, max_len), dtype=torch.long, device=device)
        for i, length in enumerate(lengths):
            attn[i, :length] = 1
    else:
        attn = None
    pool_index = torch.tensor(ends, dtype=torch.long, device=device)
    return _capture_all_hidden_states(
        model, layers, input_ids,
        attention_mask=attn, pool_index=pool_index,
    )


def _render_and_tokenize_for_capture(
    tokenizer: Any,
    prompt: str,
    response: str,
    device: torch.device,
    *,
    role: str | None = None,
    model_type: str | None = None,
    system_msg: str = _LENGTH_DIRECTIVE,
) -> tuple[torch.Tensor, int]:
    """Render a ``[system, user, assistant]`` turn + tokenize, locating the last content token.

    The shared front half of :func:`_encode_and_capture_all`.  Conversational
    (4.0 / A2) capture: ``response`` is an assistant turn answering ``prompt``,
    rendered under ``system_msg`` (the shared :data:`_LENGTH_DIRECTIVE` by
    default, matching generation; ``""`` drops the system turn) with the standard
    assistant label.  ``role`` (when set) substitutes a custom assistant-role
    label via :func:`saklas.core.role_templates.apply_with_role` for the
    persona-baselined fit; ``role=None`` is the swap-back default.  Returns
    ``(ids [1, T] on device, content_end)`` where ``content_end`` is the
    response's last non-special token — the canonical pooling position.
    """
    if getattr(tokenizer, "chat_template", None) is not None:
        # Disable thinking/reasoning mode for models that support it
        # (Qwen 3.5, QwQ, etc.) — thinking tokens would contaminate pooling.
        kwargs: dict[str, Any] = {}
        if "enable_thinking" in (getattr(tokenizer, "chat_template", "") or ""):
            kwargs["enable_thinking"] = False

        messages = []
        if system_msg:
            messages.append({"role": "system", "content": system_msg})
        messages.extend([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ])
        if role is None:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False, **kwargs,
            )
        else:
            if model_type is None:
                raise ValueError(
                    "_render_and_tokenize_for_capture: role= requires model_type= "
                    "so the family's role-header registry entry can be looked up"
                )
            from saklas.core.role_templates import apply_with_role
            text = apply_with_role(
                tokenizer, messages,
                role=role, model_type=model_type,
                add_generation_prompt=False, tokenize=False, **kwargs,
            )
    else:
        # Base model (no chat template) — there are no turn roles to render and
        # A2 role-swap cannot apply; capture the prompt+response continuation as
        # raw text, with the length directive prepended (when set) so the framing
        # still matches generation.
        text = f"{system_msg}\n{prompt}\n{response}" if system_msg else f"{prompt}\n{response}"
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    ids = enc["input_ids"]
    if ids.numel() == 0:
        bos_id = tokenizer.bos_token_id
        if bos_id is None:
            bos_id = tokenizer.eos_token_id or 0
        ids = torch.tensor([[bos_id]])
    ids = ids.to(device)

    # Last non-special token — chat templates append trailing markers (Llama's
    # <|eot_id|>, Gemma's <end_of_turn>, Qwen's <|im_end|>) whose hidden states
    # are disconnected from content.  ``last_content_index`` is the canonical
    # walkback (skips ``all_special_ids`` + ``added_tokens_encoder``), shared by
    # every single-state readout so the pooling position is defined once.
    content_end = last_content_index(ids[0].tolist(), tokenizer)
    return ids, content_end


@functools.cache
def _load_neutral_prompts() -> list[str]:
    """Load neutral prompts, preferring a user override at ~/.saklas/neutral_statements.json."""
    from saklas.io.paths import neutral_statements_path
    user_path = neutral_statements_path()
    if user_path.exists():
        with open(user_path) as f:
            return json.load(f)
    with _resources.files("saklas.data").joinpath("neutral_statements.json").open() as f:
        return json.load(f)


@functools.cache
def _load_baseline_prompts() -> list[str]:
    """Load the shared A2 baseline user prompts.

    Prefers a user override at ``~/.saklas/baseline_prompts.json``, else the
    bundled ``saklas/data/baseline_prompts.json`` (48 affect-neutral prompts).
    Conversational capture pairs each node response with its prompt by
    ``response[i] -> prompt[i % len(prompts)]``.
    """
    from saklas.io.paths import baseline_prompts_path
    user_path = baseline_prompts_path()
    if user_path.exists():
        with open(user_path) as f:
            return json.load(f)
    with _resources.files("saklas.data").joinpath("baseline_prompts.json").open() as f:
        return json.load(f)


def _neutral_pairs() -> list[tuple[str, str]]:
    """The neutral baseline as ``(prompt, response)`` pairs for conversational capture.

    The neutral corpus (``neutral_statements.json``) is the model's organic,
    no-persona / no-role responses to the shared baseline prompts (generated and
    captured under the shared length directive only — see :data:`_LENGTH_DIRECTIVE`);
    pairing is positional, ``response[i] -> baseline[i % k]`` — the same alignment
    a node corpus uses.  Raises if the corpus length is not a multiple of the
    prompt set (regenerate neutrals against the shared baseline).
    """
    responses = _load_neutral_prompts()
    baseline = _load_baseline_prompts()
    k = len(baseline)
    if k == 0:
        raise ValueError("no baseline prompts available for neutral capture")
    if len(responses) % k != 0:
        raise ValueError(
            f"neutral corpus ({len(responses)}) must be a multiple of the "
            f"baseline prompt set ({k}); regenerate neutrals against it"
        )
    return [(baseline[i % k], r) for i, r in enumerate(responses)]


def compute_neutral_activations(
    model: torch.nn.Module,
    tokenizer: Any,
    layers: torch.nn.ModuleList,
    device: torch.device | None = None,
) -> dict[int, torch.Tensor]:
    """Per-layer ``[N, D]`` stack across the neutral corpus.

    Last non-special-token pooling, fp32.  The ``(prompt, response)`` pairs are
    captured in **batched chunks** (:func:`_encode_and_capture_all_batch`, one
    forward per ``_CAPTURE_BATCH`` pairs) rather than one forward per pair, and
    the MPS allocator flush is amortized to once per chunk.  Returns one stacked
    tensor per layer (rows = pairs, original order preserved).  This is the
    single per-model neutral artifact: the Mahalanobis whitener's covariance is
    built from the stack and the probe-centering means are its per-layer
    ``X.mean(0)`` (:func:`saklas.io.probes_bootstrap.bootstrap_layer_means`).
    Used by cross-model alignment (:func:`saklas.io.alignment.fit_alignment`)
    which needs paired observations to fit Procrustes; the means alone (N=1)
    are degenerate for that fit.

    Storage cost: ~90 · n_layers · hidden_dim · 4B in fp32 (≈ 56MB on
    a 4096-dim, 80-layer model).  Callers persist this through
    :func:`saklas.io.alignment.load_or_compute_neutral_activations`.
    """
    if device is None:
        device = next(model.parameters()).device
    assert device is not None  # device is always set by this point

    n_layers = len(layers)
    chunks_by_layer: dict[int, list[torch.Tensor]] = {
        idx: [] for idx in range(n_layers)
    }
    _mps = device.type == "mps"

    pairs = _neutral_pairs()
    prompts = [prompt for prompt, _ in pairs]
    responses = [response for _, response in pairs]
    n = len(pairs)

    for start in range(0, n, _CAPTURE_BATCH):
        end = min(start + _CAPTURE_BATCH, n)
        per_layer = _encode_and_capture_all_batch(
            model, tokenizer,
            prompts[start:end], responses[start:end],
            layers, device,
        )
        # Move each chunk's (B, D) to CPU before discarding the GPU-side dict —
        # same MPS discipline as before, now per chunk instead of per pair.
        for idx in range(n_layers):
            chunks_by_layer[idx].append(per_layer[idx].detach().to("cpu"))
        del per_layer
        if _mps:
            torch.mps.empty_cache()

    return {
        idx: torch.cat(chunks, dim=0)  # (N, D), fp32 on cpu
        for idx, chunks in chunks_by_layer.items()
    }


def compute_dls_axes(
    node_centroids: dict[int, torch.Tensor],
    bases: dict[int, torch.Tensor],
    layer_means: dict[int, torch.Tensor] | None,
) -> dict[int, set[int]]:
    """Per-axis Discriminative-Layer-Selection over **N node centroids**.

    The N-node generalization of the bipolar opposite-sign test (Dang & Ngo
    2026, Eq. 9) — the form a flat *subspace* of any node count uses.  Each
    layer carries a basis ``[R, D]`` (its ``LayerSubspace.basis`` rows) and
    ``node_centroids[L]`` is the ``[K, D]`` stack of that layer's per-node
    mean activations.  Every row ``d̂_r`` is kept iff the node projections
    relative to the neutral baseline, ``{(c_i − ν)·d̂_r : i}``, **straddle
    zero** (both a negative and a positive present) — the axis then
    discriminates node *position* across the baseline rather than encoding a
    common offset / intensity every node shares.

    **N=2 parity (the spine invariant).**  With
    ``node_centroids[L] = stack([μ_pos, μ_neg])`` the straddle
    ``min < 0 < max`` is exactly the bipolar opposite-sign product test, bit
    for bit, so a 2-node subspace reproduces a difference-of-means steering
    vector's DLS keep set exactly (the ``R = 1`` layer set), and an N-node
    subspace (e.g. ``personas``) prunes per-axis the same way.

    Returns ``{layer: {kept axis indices}}``; a layer absent / mapped to an
    empty set is fully dropped.  ``layer_means`` ``None``/empty disables DLS
    (every axis of every layer kept).  A layer whose baseline ``ν`` is missing
    keeps every checkable axis conservatively (over-include rather than drop a
    real discriminative axis for missing data).  Degenerate (zero-norm) rows
    are dropped and excluded from the all-fail fallback; when no axis anywhere
    passes, falls back to the full checkable set with a one-time warning.

    Args:
        node_centroids: per-layer ``[K, D]`` stack of node mean activations.
        bases: per-layer ``[R, D]`` basis (rows are the candidate axes).
        layer_means: per-layer neutral baseline; ``None``/empty disables.
    """
    if layer_means is None or not layer_means:
        out: dict[int, set[int]] = {}
        for L in node_centroids:
            B = bases.get(L)
            r = int(B.shape[0]) if B is not None else 1
            out[L] = set(range(r))
        return out
    keep: dict[int, set[int]] = {}
    # ``checkable`` collects the (layer, axis) pairs we *attempted* the test
    # on — rows with a valid (non-degenerate) direction.  Rows skipped for
    # zero norm are excluded from the all-failed fallback.
    checkable: dict[int, set[int]] = {}
    any_pass = False
    for L in node_centroids:
        B = bases.get(L)
        if B is None:
            continue
        B32 = B.to(dtype=torch.float32, device="cpu").reshape(int(B.shape[0]), -1)
        mu_n = layer_means.get(L)
        mu_n_cpu = (
            mu_n.to(dtype=torch.float32, device="cpu").reshape(-1)
            if mu_n is not None
            else None
        )
        C = node_centroids[L].to(dtype=torch.float32, device="cpu")
        if C.ndim == 1:
            C = C.reshape(1, -1)
        C = C.reshape(int(C.shape[0]), -1)              # (K, D)
        layer_checkable: set[int] = set()
        layer_keep: set[int] = set()
        for r in range(int(B32.shape[0])):
            row = B32[r]
            row_norm = float(row.norm())
            if row_norm < 1e-12:
                continue  # degenerate axis — drop, exclude from fallback
            layer_checkable.add(r)
            if mu_n_cpu is None:
                # Baseline doesn't cover this layer.  Conservative: keep —
                # over-include rather than drop a real discriminative axis
                # for missing baseline data.
                layer_keep.add(r)
                continue
            projs = (C - mu_n_cpu) @ (row / row_norm)   # (K,) node projections
            # Straddle zero: both a negative and a positive projection ⇒ the
            # axis separates nodes across the baseline.  At K=2 this is the
            # bipolar ``proj_pos · proj_neg < 0`` opposite-sign test exactly.
            if float(projs.min()) < 0.0 and float(projs.max()) > 0.0:
                layer_keep.add(r)
        if layer_checkable:
            checkable[L] = layer_checkable
        if layer_keep:
            keep[L] = layer_keep
            any_pass = True
    if not any_pass:
        # Every checkable axis failed — probe degenerate on this model (the
        # diagnostics warning fires separately).  Keep the checkable set
        # rather than every layer; when nothing is checkable the fallback is
        # empty and the share-bake all-zero handler kicks in.
        warnings.warn(
            "DLS: no layers pass the discriminative check; falling back "
            "to keep-all-checkable.  Probe likely degenerate on this "
            "model — review the diagnostics warning above.",
            UserWarning,
            stacklevel=3,
        )
        return checkable
    return keep


def fold_directions_to_subspace(
    name: str,
    directions: dict[int, torch.Tensor],
    neutral_means: dict[int, torch.Tensor] | None,
    *,
    whitener: "Any | None" = None,
    label: str = "+",
    feature_space: str = "raw",
) -> "Any":  # -> saklas.core.manifold.Manifold
    """Fold an arbitrary per-layer *direction* into a neutral-anchored affine
    ``R = 1`` manifold (a one-pole ray).

    Where a bipolar concept's fit (`extraction.py`) folds a line through two
    pole centroids, a derived direction (a ``merge`` linear-combination, a
    `~`/`|` projection of one concept onto another) has no pole pair — just a
    direction to push along.
    So the subspace is neutral-anchored at ``mean = P_basis(ν_L) = (ν_L·d̂)d̂``
    (the neutral mean projected into the 1-D span, §5) with a single ``+`` pole
    node; steering ``along`` toward the pole pushes the running activation's
    ``d̂`` component away from neutral, and ``!`` (``along``-to-coord-0) slides
    it back to neutral, i.e. ablates the direction.

    Per layer: ``basis = d̂_L``; ``LayerSubspace.node_coords = [[‖d_L‖]]`` —
    the real coord of the pole, a step of ``‖d_L‖`` along ``d̂`` from the
    neutral origin (coord 0).  ``share = ‖d_L‖_M`` (whitened) / ``‖d_L‖₂``
    (Euclidean) — the direction magnitude itself is the budget (a single
    direction has no node cloud to take a μ-centered spread over, so this is
    *not* :func:`~saklas.core.manifold.subspace_share`'s √2-scaled form; the
    apply-time normalization handles the cross-layer scale either way).  No
    DLS — a derived direction has no polarity for the straddle test; the
    caller's layer set folds verbatim.  No origin store (coord 0, §2).
    ``neutral_means=None`` (CPU stubs) anchors at coord 0.
    """
    from saklas.core.manifold import (
        CustomDomain, LayerSubspace, Manifold,
    )

    present = sorted(directions)
    maha_w = (
        whitener
        if whitener is not None and whitener.covers_all(present)
        else None
    )

    # Every fresh tensor follows the directions' device so the folded manifold
    # is internally device-consistent.  The dispatch-time fold runs over a
    # *loaded* manifold's directions, which may live on the model device (MPS /
    # CUDA); a CPU-default ``node_coords`` would then mismatch the on-device
    # ``basis`` inside ``synthesize_subspace`` (``c_i @ B_i``).  ``ν`` is moved to
    # the same device too so the neutral-anchor projection stays co-located.
    dev = directions[present[0]].device if present else torch.device("cpu")

    layers: dict[int, "LayerSubspace"] = {}
    mahalanobis_share: dict[int, float] = {}
    for idx in present:
        d = directions[idx].to(torch.float32).reshape(-1)
        norm = float(d.norm())
        if norm <= 1e-12:
            continue
        basis = (d / norm).reshape(1, -1)          # (1, D) unit d̂
        nu = (
            neutral_means[idx].to(device=d.device, dtype=torch.float32).reshape(-1)
            if neutral_means is not None and idx in neutral_means
            else None
        )
        # Neutral-anchored: mean = P_basis(ν) (off-span part of ν dropped);
        # the pole sits at the real coord ‖d‖ along d̂ from the origin.
        mean = (nu @ basis.T) @ basis if nu is not None else torch.zeros_like(d)
        node_coords_L = torch.tensor([[norm]], dtype=torch.float32, device=d.device)
        layers[idx] = LayerSubspace.affine(mean, basis, node_coords=node_coords_L)
        mahalanobis_share[idx] = (
            float(maha_w.mahalanobis_norm(idx, d))
            if maha_w is not None else norm
        )

    # Shared display layout: a single ``+`` pole at coord 1 (the real per-layer
    # pole distance ‖d_L‖ lives on each ``LayerSubspace.node_coords``).
    node_coords = torch.tensor([[1.0]], dtype=torch.float32, device=dev)
    manifold = Manifold(
        name=name,
        domain=CustomDomain(1),
        node_labels=[label],
        node_coords=node_coords,
        layers=layers,
        feature_space=feature_space,
        mahalanobis_share=mahalanobis_share,
    )
    manifold.metadata["share_metric"] = (
        "mahalanobis" if maha_w is not None else "euclidean"
    )
    return manifold


def folded_vector_directions(manifold: "Any") -> dict[int, torch.Tensor]:
    """Baked-direction view of a folded (affine ``R = 1``) vector manifold.

    Returns ``{L: δ̂_L · share_L}`` — the steering-vector-equivalent baked
    directions: the per-layer unit direction scaled by its Mahalanobis
    share, i.e. the per-layer share-weighted direction (``‖baked_L‖ ∝
    ‖δ_L‖_M``), matching the (now-removed) DiM bake per layer.  The
    global per-concept scale differs (the folded share is the un-normalized
    ``‖δ_L‖_M``, not the ``ref_norm``-weighted normalized share today's bake
    folds in), so this view is *exact* for scale-invariant ops — per-layer and
    aggregate cosine, ``manifold compare``/``manifold why`` — and merely
    proportional for cross-concept magnitude (``merge`` / GGUF export, which
    migrate to read the folded artifact natively).

    Lets the unified folded Manifold back the legacy direction-math surface
    (``Profile``-returning ``extract()``, ``manifold compare``/``manifold why``) without a
    second *stored* representation — the concept's only on-disk artifact stays
    the folded Manifold; this is a downstream in-memory view.

    Raises :class:`ValueError` on a curved or multi-dim manifold — the
    single-direction view is only meaningful for a folded ``R = 1`` vector.
    """
    out: dict[int, torch.Tensor] = {}
    share_map = getattr(manifold, "mahalanobis_share", None) or {}
    for idx, sub in manifold.layers.items():
        if not sub.is_affine or sub.rank != 1:
            raise ValueError(
                "folded_vector_directions requires affine R=1 layers; "
                f"layer {idx} is rank {sub.rank}, affine={sub.is_affine}"
            )
        share = float(share_map.get(idx, 1.0))
        out[idx] = sub.basis.reshape(-1).to(torch.float32) * share
    return out


def save_profile(
    profile: dict[int, torch.Tensor],
    path: str | Path,
    metadata: dict[str, Any],
) -> None:
    """Save a baked vector profile as .safetensors with a slim .json sidecar.

    ``metadata`` must contain at minimum:
        method            - str, e.g. "difference_of_means" / "merge" / "layer_means"

    Optional keys honored:
        statements_sha256 - str, hash of source statements at extraction time
        components        - dict, merge provenance (method="merge" only)
        diagnostics       - dict[int, dict[str, float]], per-layer probe-quality
                            metrics (see ``_compute_layer_diagnostics``).
                            Persisted as ``diagnostics_by_layer`` on the
                            sidecar with stringified layer keys.

    The safetensors file contains keys ``"layer_{i}"`` for each active layer.
    Tensors are already baked (share pre-multiplied into magnitude) — the
    sidecar carries only method/saklas_version plus the optional fields above.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # fp32 write invariant: every saklas safetensor writer enforces fp32
    # on disk (matches gguf_io.py's ``.to(dtype=torch.float32)``), so the
    # stored dtype is a guarantee rather than a coincidence of the caller.
    tensors = {
        f"layer_{idx}": vec.to(dtype=torch.float32).contiguous().cpu()
        for idx, vec in profile.items()
    }
    save_file(tensors, str(path))

    from saklas import __version__ as _saklas_version
    from saklas.io.packs import PACK_FORMAT_VERSION
    sidecar: dict[str, Any] = {
        "format_version": PACK_FORMAT_VERSION,
        "method": metadata.get("method", "difference_of_means"),
        "saklas_version": _saklas_version,
    }
    if "statements_sha256" in metadata:
        sidecar["statements_sha256"] = metadata["statements_sha256"]
    if "components" in metadata:
        sidecar["components"] = metadata["components"]
    # v2.1: bake method records which scoring metric drove share allocation
    # (``"euclidean"`` = legacy ``||m||_2 / r``; ``"mahalanobis"`` =
    # ``||m||_M / r`` via per-layer activation covariance).  Loaders read
    # this only for diagnostics; the runtime hook reads tensor magnitudes
    # regardless of bake flavor.  Default ``"euclidean"`` is back-compat
    # for tensors written before the bake field existed.
    if "bake" in metadata:
        sidecar["bake"] = metadata["bake"]
    # SAE provenance — present only when extraction used an SAE backend.
    for key in ("sae_release", "sae_revision", "sae_ids_by_layer"):
        if key in metadata:
            sidecar[key] = metadata[key]
    # Transfer provenance — present only on transferred profiles
    # (method="procrustes_transfer").  ``alignment_map_hash`` pins the
    # specific Procrustes fit; ``transfer_quality_estimate`` is the
    # median per-layer R² across shared layers.
    for key in (
        "source_model_id",
        "alignment_map_hash",
        "transfer_quality_estimate",
    ):
        if key in metadata:
            sidecar[key] = metadata[key]
    # Diagnostics: stringify layer keys so the JSON round-trips through
    # standard parsers (JSON object keys must be strings).  Reader inverts.
    diagnostics = metadata.get("diagnostics")
    if diagnostics:
        sidecar["diagnostics_by_layer"] = {
            str(layer): {k: float(v) for k, v in metrics.items()}
            for layer, metrics in diagnostics.items()
        }

    from saklas.io.atomic import write_json_atomic
    meta_path = path.with_suffix(".json")
    write_json_atomic(meta_path, sidecar)

    log.info("Saved profile (%d layers) to %s", len(profile), path)


def load_profile(path: str | Path) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
    """Load a baked vector profile and its metadata.

    Dispatches on file extension: ``.safetensors`` reads the companion
    ``.json`` sidecar; ``.gguf`` reads the control-vector metadata embedded
    in the GGUF header (see :mod:`saklas.gguf_io`). Both paths yield the
    same ``(profile, metadata)`` shape — callers don't need to branch.

    Returns:
        (profile dict mapping layer_idx -> baked vector, metadata dict)
    """
    path = Path(path)
    if path.suffix == ".gguf":
        from saklas.io.gguf_io import read_gguf_profile
        return read_gguf_profile(path)

    tensors = load_file(str(path))
    meta_path = path.with_suffix(".json")
    with open(meta_path) as f:
        metadata = json.load(f)

    from saklas.io.packs import PACK_FORMAT_VERSION
    from saklas.core.profile import ProfileError
    fmt_ver = metadata.get("format_version", 1)
    if not isinstance(fmt_ver, int) or fmt_ver < PACK_FORMAT_VERSION:
        raise ProfileError(
            f"pack format is from saklas < 2.0 "
            f"(sidecar {meta_path} format_version={fmt_ver!r}, "
            f"need >= {PACK_FORMAT_VERSION}); "
            f"run `python scripts/upgrade_packs.py {path.parent}` to migrate"
        )

    profile = {int(key.split("_", 1)[1]): tensor for key, tensor in tensors.items()}

    # Invert the layer-key stringification done at save time so diagnostics
    # are addressable by ``int`` consistently with the profile dict.
    raw_diag = metadata.get("diagnostics_by_layer")
    if isinstance(raw_diag, dict) and raw_diag:
        # Malformed diagnostics leave the raw dict in place; downstream readers
        # can decide whether to fall back. The tensors themselves are still valid.
        with suppress(TypeError, ValueError):
            metadata["diagnostics"] = {
                int(layer): dict(metrics)
                for layer, metrics in raw_diag.items()
            }

    return profile, metadata


def project_profile(
    base: dict[int, torch.Tensor],
    onto: dict[int, torch.Tensor],
    operator: str,
    *,
    whitener: "Any | None" = None,
) -> dict[int, torch.Tensor]:
    """Per-layer projection of ``base`` against ``onto`` (LEACE only).

    LEACE-style projection in the Mahalanobis metric, per shared layer::

        coef = <base, onto>_M / <onto, onto>_M
        proj = coef * onto                # direction is ``onto``; metric is M

    The output direction is still ``onto``, but the *amount* removed is
    the component along ``onto`` measured in the whitened space.  For
    operator ``"|"``, this is the closed-form LEACE projector for a
    single direction (Belrose et al. 2023, arXiv 2306.03819) — provably
    erases linearly-decodable information along ``onto`` from ``base``
    with minimum collateral damage.

    Operator semantics:

    - ``operator == "~"``   returns ``proj``       (component of base aligned with onto).
    - ``operator == "|"`` returns ``base - proj`` (component of base orthogonal to onto).

    Layers in ``base`` without a matching layer in ``onto``: for ``"|"``
    they pass through unchanged (nothing to project away); for ``"~"`` they
    are dropped (projection onto an absent direction is undefined).

    Near-zero ``||onto|| < 1e-12`` layers are treated the same way: ``"|"``
    passes base through unchanged, ``"~"`` drops the layer.

    The whitener (a :class:`saklas.core.mahalanobis.LayerWhitener`) is
    **required** and must cover *every* projected layer (``base ∩ onto``),
    via :meth:`LayerWhitener.covers_all`.  There is no Euclidean path: a
    missing or non-covering whitener raises :class:`WhitenerError`.
    Result tensors are cast back to the source dtype of ``base``.
    """
    from saklas.core.mahalanobis import WhitenerError

    if operator not in ("~", "|"):
        raise ValueError(f"unknown projection operator: {operator!r}")
    # Mahalanobis-only: the whitener must cover every layer that actually
    # gets projected (``base ∩ onto``).  No Euclidean fallback — a missing
    # or partial whitener is an error.
    projected_layers = sorted(set(base) & set(onto))
    if whitener is None or not whitener.covers_all(projected_layers):
        raise WhitenerError(
            "project_profile requires a Mahalanobis whitener covering every "
            f"projected layer {projected_layers}; regenerate the neutral "
            "activation cache for this model (the Euclidean path is gone)"
        )
    out: dict[int, torch.Tensor] = {}
    for layer, base_t in base.items():
        onto_t = onto.get(layer)
        if onto_t is None:
            if operator == "|":
                out[layer] = base_t
            continue
        projected = whitener.leace_project(layer, base_t, onto_t, operator)
        # Drop the layer for ``~`` when ``onto`` is degenerate under
        # the Mahalanobis metric — leace_project returns a zero tensor
        # in that case.
        if operator == "~" and torch.all(projected == 0):
            continue
        out[layer] = projected
    if not out:
        raise ValueError(
            f"project_profile: no layers produced for operator {operator!r} "
            f"(base layers: {sorted(base.keys())}, "
            f"onto layers: {sorted(onto.keys())})"
        )
    return out
