"""CPU tests for the Jacobian lens: estimator correctness, merge, word→token.

The estimator test is the load-bearing one: on a tiny causal toy transformer
(frozen params, exactly like a saklas-loaded model) the fitted ``J_l`` must
match the exact averaged Jacobian computed with ``torch.autograd.functional``.
"""

from __future__ import annotations

import math
from typing import Any, cast

import pytest
import torch
from torch import nn

from saklas.core.jlens import (
    JacobianLens,
    JacobianLensError,
    MultiTokenWordError,
    fit_jacobian_lens,
    lens_logits,
    resolve_word_token,
    topk_logprobs,
)
from saklas.core.model import get_final_norm, get_unembedding
from tests._jlens_toys import TOY_D as _D
from tests._jlens_toys import TOY_VOCAB as _VOCAB
from tests._jlens_toys import CharTokenizer as _CharTokenizer
from tests._jlens_toys import ToyCausalLM as _TinyCausalLM
from tests._jlens_toys import frozen_toy as _frozen_model


def _layers(model: _TinyCausalLM) -> list[nn.Module]:
    return list(model.model.layers)


def _exact_jacobian(
    model: _TinyCausalLM, ids: torch.Tensor, source: int, *, skip_first: int
) -> torch.Tensor:
    """The reference estimand, via exact autograd on the block stack.

    ``J = mean_t Σ_{t' valid} ∂h_final[t']/∂h_source[t]`` over valid source
    positions t (causality zeroes the t' < t terms).
    """
    with torch.enable_grad():
        h = model.model.embed_tokens(ids)
        for block in model.model.layers[: source + 1]:
            h = block(h)
        h_source = h.detach()

        def rest(hs: torch.Tensor) -> torch.Tensor:
            out = hs
            for block in model.model.layers[source + 1 :]:
                out = block(out)
            return out

        full = torch.autograd.functional.jacobian(rest, h_source)
    # [1, T, d, 1, T, d] -> [T, d, T, d]
    full = full.squeeze(3).squeeze(0)
    seq_len = ids.shape[1]
    valid = range(skip_first, seq_len - 1)
    per_source = [sum(full[tp, :, t, :] for tp in valid) for t in valid]
    return torch.stack(per_source).mean(dim=0)


def test_estimator_matches_exact_jacobian() -> None:
    model = _frozen_model(n_layers=3)
    tokenizer = _CharTokenizer()
    prompt = "the quick brown fox js"  # 22 chars -> 5 valid positions
    skip = 16

    lens = fit_jacobian_lens(
        model, tokenizer, [prompt], _layers(model),
        dim_batch=4, skip_first=skip,
    )

    ids = tokenizer(prompt)["input_ids"]
    for source in (0, 1):
        exact = _exact_jacobian(model, ids, source, skip_first=skip)
        assert torch.allclose(lens.jacobians[source], exact, atol=1e-5), (
            f"layer {source}: estimator diverges from exact Jacobian"
        )


def test_batched_vjp_uses_single_prompt_forward_and_expected_grad_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _frozen_model(n_layers=3)
    tokenizer = _CharTokenizer()
    prompt = "the quick brown fox js"
    seen_batches: list[int] = []
    grad_calls = 0
    real_forward = model.forward
    real_grad = torch.autograd.grad

    def counted_forward(input_ids: torch.Tensor, use_cache: bool = False) -> Any:
        seen_batches.append(int(input_ids.shape[0]))
        return real_forward(input_ids=input_ids, use_cache=use_cache)

    def counted_grad(*args: Any, **kwargs: Any) -> Any:
        nonlocal grad_calls
        grad_calls += 1
        assert kwargs.get("is_grads_batched") is True
        return real_grad(*args, **kwargs)

    monkeypatch.setattr(model, "forward", counted_forward)
    monkeypatch.setattr(torch.autograd, "grad", counted_grad)

    fit_jacobian_lens(
        model, tokenizer, [prompt], _layers(model),
        dim_batch=4, skip_first=16, vjp_mode="batched",
    )

    assert seen_batches == [1]
    assert grad_calls == math.ceil(_D / 4)


def test_restricted_source_layers_match_exact_jacobian() -> None:
    """A band-restricted fit seeds at its lowest source layer (blocks below
    run graph-free) — the estimate must be unchanged by the truncation."""
    model = _frozen_model(n_layers=3)
    tokenizer = _CharTokenizer()
    prompt = "the quick brown fox js"
    skip = 16

    lens = fit_jacobian_lens(
        model, tokenizer, [prompt], _layers(model),
        source_layers=[1], dim_batch=4, skip_first=skip,
    )
    assert lens.source_layers == [1]

    ids = tokenizer(prompt)["input_ids"]
    exact = _exact_jacobian(model, ids, 1, skip_first=skip)
    assert torch.allclose(lens.jacobians[1], exact, atol=1e-5)


def test_replicated_vjp_mode_matches_exact_jacobian() -> None:
    model = _frozen_model(n_layers=3)
    tokenizer = _CharTokenizer()
    prompt = "the quick brown fox js"
    skip = 16

    lens = fit_jacobian_lens(
        model, tokenizer, [prompt], _layers(model),
        dim_batch=4, skip_first=skip, vjp_mode="replicated",
    )

    ids = tokenizer(prompt)["input_ids"]
    for source in (0, 1):
        exact = _exact_jacobian(model, ids, source, skip_first=skip)
        assert torch.allclose(lens.jacobians[source], exact, atol=1e-5)


def test_fit_averages_over_prompts_and_merge_agrees() -> None:
    model = _frozen_model(n_layers=2)
    tokenizer = _CharTokenizer()
    p1 = "a prompt that is long enough."
    p2 = "another, quite different one!!"

    joint = fit_jacobian_lens(model, tokenizer, [p1, p2], _layers(model), dim_batch=3)
    solo1 = fit_jacobian_lens(model, tokenizer, [p1], _layers(model), dim_batch=3)
    solo2 = fit_jacobian_lens(model, tokenizer, [p2], _layers(model), dim_batch=3)
    merged = JacobianLens.merge([solo1, solo2])

    assert joint.n_prompts == merged.n_prompts == 2
    for layer in joint.source_layers:
        assert torch.allclose(joint.jacobians[layer], merged.jacobians[layer], atol=1e-6)


def test_select_and_union_layers() -> None:
    j0 = torch.randn(_D, _D)
    j1 = torch.randn(_D, _D)
    j2 = torch.randn(_D, _D)
    lens = JacobianLens({0: j0, 1: j1}, n_prompts=3, d_model=_D)

    selected = lens.select_layers([1])
    assert selected.source_layers == [1]
    assert torch.equal(selected.jacobians[1], lens.jacobians[1])

    union = JacobianLens.union_layers([
        selected,
        JacobianLens({2: j2}, n_prompts=3, d_model=_D),
    ])
    assert union.source_layers == [1, 2]


def test_fit_skips_short_prompts_and_raises_when_all_short() -> None:
    model = _frozen_model(n_layers=2)
    tokenizer = _CharTokenizer()

    long_prompt = "a prompt that is long enough."
    lens = fit_jacobian_lens(
        model, tokenizer, ["tiny", long_prompt], _layers(model), dim_batch=3,
    )
    assert lens.n_prompts == 1

    with pytest.raises(JacobianLensError, match="no usable prompts"):
        fit_jacobian_lens(model, tokenizer, ["tiny"], _layers(model), dim_batch=3)


def test_fit_can_reuse_pretokenized_rows_without_tokenizer_call() -> None:
    model = _frozen_model(n_layers=2)
    tokenizer = _CharTokenizer()
    ids = tokenizer("a prompt that is long enough.")["input_ids"][0].tolist()

    class BombTokenizer:
        def __call__(self, *_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("fit should reuse input_id_rows")

    lens = fit_jacobian_lens(
        model, BombTokenizer(), ["already-tokenized"], _layers(model),
        dim_batch=3, input_id_rows=[ids],
    )

    assert lens.n_prompts == 1


def test_fit_checkpoint_callback_fires() -> None:
    model = _frozen_model(n_layers=2)
    tokenizer = _CharTokenizer()
    prompt = "a prompt that is long enough."
    seen: list[int] = []

    fit_jacobian_lens(
        model, tokenizer, [prompt] * 3, _layers(model),
        dim_batch=3, checkpoint_every=2, checkpoint_cb=lambda l: seen.append(l.n_prompts),
    )
    assert seen == [2]


def test_source_layers_must_precede_final() -> None:
    model = _frozen_model(n_layers=2)
    with pytest.raises(ValueError, match="source_layers"):
        fit_jacobian_lens(
            model, _CharTokenizer(), ["a prompt that is long enough."],
            _layers(model), source_layers=[1],
        )


def test_source_layers_must_not_be_empty() -> None:
    model = _frozen_model(n_layers=2)
    with pytest.raises(ValueError, match="at least one"):
        fit_jacobian_lens(
            model, _CharTokenizer(), ["a prompt that is long enough."],
            _layers(model), source_layers=[],
        )


def test_lens_logits_matches_model_at_identity_transport() -> None:
    """With J_l = I at the last source layer... the readout must equal running
    the remaining blocks' *absence* — i.e. norm+unembed of the residual itself
    (the logit-lens identity check)."""
    model = _frozen_model(n_layers=2)
    ids = _CharTokenizer()("a prompt that is long enough.")["input_ids"]
    with torch.no_grad():
        h = model.model.embed_tokens(ids)
        for block in model.model.layers:
            h = block(h)
        expected = model(input_ids=ids).logits

    lens = JacobianLens({1: torch.eye(_D)}, n_prompts=1, d_model=_D)
    out = lens_logits(
        lens, {1: h},
        unembed=get_unembedding(cast(Any, model)),
        final_norm=get_final_norm(cast(Any, model)),
    )
    assert torch.allclose(out[1], expected.squeeze(0).float(), atol=1e-5)


def test_transport_unknown_layer_raises() -> None:
    lens = JacobianLens({0: torch.eye(_D)}, n_prompts=1, d_model=_D)
    with pytest.raises(JacobianLensError):
        lens.transport(torch.zeros(_D), 3)


def test_accessors_on_toy_model() -> None:
    model = _frozen_model()
    w = get_unembedding(cast(Any, model))
    assert w.shape == (_VOCAB, _D)
    assert get_final_norm(cast(Any, model)) is model.model.norm


def test_token_direction_shape_and_math() -> None:
    J = torch.randn(2, _D, _D)
    lens = JacobianLens({0: J[0], 1: J[1]}, n_prompts=1, d_model=_D)
    unembed = torch.randn(_VOCAB, _D)
    dirs = lens.token_direction(5, unembed)
    assert set(dirs) == {0, 1}
    assert torch.allclose(dirs[1], unembed[5] @ J[1])


def test_token_direction_can_restrict_layers() -> None:
    J = torch.randn(2, _D, _D)
    lens = JacobianLens({0: J[0], 1: J[1]}, n_prompts=1, d_model=_D)
    unembed = torch.randn(_VOCAB, _D)
    dirs = lens.token_direction(5, unembed, layers=[1])
    assert set(dirs) == {1}
    assert torch.allclose(dirs[1], unembed[5] @ J[1])


def test_topk_logprobs_matches_full_log_softmax() -> None:
    logits = torch.randn(4, _VOCAB)
    vals, idxs = topk_logprobs(logits, 7)
    expected = torch.log_softmax(logits, dim=-1)
    exp_vals, exp_idxs = expected.topk(7, dim=-1)
    assert torch.equal(idxs, exp_idxs)
    assert torch.allclose(vals, exp_vals)


class _WordTokenizer:
    """encode/decode stub: vocabulary maps ' fake'->1, 'fake'->2, multi otherwise."""

    def __init__(self, single: dict[str, int], decode_map: dict[int, str]) -> None:
        self.single = single
        self.decode_map = decode_map

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        if text in self.single:
            return [self.single[text]]
        return [3, 4]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.decode_map.get(i, "?") for i in ids)


def test_resolve_word_token_prefers_leading_space() -> None:
    tok = _WordTokenizer({" fake": 1, "fake": 2}, {1: " fake", 2: "fake"})
    assert resolve_word_token(tok, "fake") == 1


def test_resolve_word_token_falls_back_to_bare() -> None:
    tok = _WordTokenizer({"fake": 2}, {2: "fake"})
    assert resolve_word_token(tok, "fake") == 2


def test_resolve_word_token_multi_token_raises() -> None:
    tok = _WordTokenizer({}, {3: "fa", 4: "ke"})
    with pytest.raises(MultiTokenWordError, match="pieces"):
        resolve_word_token(tok, "fake")


def test_resolve_word_token_decode_mismatch_raises() -> None:
    # single-token hit whose decode is a different word: must NOT be accepted
    tok = _WordTokenizer({" fake": 1, "fake": 2}, {1: " fakes", 2: "faked"})
    with pytest.raises(MultiTokenWordError):
        resolve_word_token(tok, "fake")


def test_merge_rejects_mismatched_lenses() -> None:
    a = JacobianLens({0: torch.eye(_D)}, n_prompts=1, d_model=_D)
    b = JacobianLens({1: torch.eye(_D)}, n_prompts=1, d_model=_D)
    with pytest.raises(ValueError, match="disagree"):
        JacobianLens.merge([a, b])
