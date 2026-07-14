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
    aggregate_readout,
    aggregate_readout_from_probabilities,
    fit_jacobian_lens,
    lens_logits,
    readout_probabilities,
    resolve_word_token,
    topk_logprobs,
    token_readout_stats,
    token_readout_stats_from_probabilities,
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


def test_fit_stops_before_final_norm_and_lm_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The target-layer hook should terminate the forward at the residual."""
    model = _frozen_model(n_layers=3)
    tokenizer = _CharTokenizer()
    calls = {"norm": 0, "head": 0}
    real_norm = model.model.norm.forward
    real_head = model.lm_head.forward

    def counted_norm(*args: Any, **kwargs: Any) -> Any:
        calls["norm"] += 1
        return real_norm(*args, **kwargs)

    def counted_head(*args: Any, **kwargs: Any) -> Any:
        calls["head"] += 1
        return real_head(*args, **kwargs)

    monkeypatch.setattr(model.model.norm, "forward", counted_norm)
    monkeypatch.setattr(model.lm_head, "forward", counted_head)
    fit_jacobian_lens(
        model, tokenizer, ["the quick brown fox js"], _layers(model),
        dim_batch=3,
    )
    assert calls == {"norm": 0, "head": 0}


def test_restricted_source_layers_match_exact_jacobian() -> None:
    """A restricted fit seeds at its lowest source output without changing J."""
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


def test_scalar_vjp_mode_matches_exact_jacobian_without_replication() -> None:
    model = _frozen_model(n_layers=3)
    tokenizer = _CharTokenizer()
    prompt = "the quick brown fox js"
    seen_batches: list[int] = []
    real_forward = model.forward

    def counted_forward(input_ids: torch.Tensor, **kwargs: Any) -> Any:
        seen_batches.append(int(input_ids.shape[0]))
        return real_forward(input_ids=input_ids, **kwargs)

    model.forward = counted_forward  # type: ignore[method-assign]
    lens = fit_jacobian_lens(
        model, tokenizer, [prompt], _layers(model),
        dim_batch=4, skip_first=16, vjp_mode="scalar",
    )

    ids = tokenizer(prompt)["input_ids"]
    for source in (0, 1):
        exact = _exact_jacobian(model, ids, source, skip_first=16)
        assert torch.allclose(lens.jacobians[source], exact, atol=1e-5)
    assert seen_batches and set(seen_batches) == {1}


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


def test_ragged_prompt_microbatch_matches_single_prompt_graphs() -> None:
    model = _frozen_model(n_layers=2)
    tokenizer = _CharTokenizer()
    prompts = [
        "short but still comfortably usable",
        "a substantially longer second prompt for padding coverage........",
        "medium prompt with another length....",
    ]
    batched = fit_jacobian_lens(
        model, tokenizer, prompts, _layers(model), dim_batch=3, prompt_batch=3,
    )
    singles = fit_jacobian_lens(
        model, tokenizer, prompts, _layers(model), dim_batch=3, prompt_batch=1,
    )
    for layer in batched.source_layers:
        assert torch.allclose(
            batched.jacobians[layer], singles.jacobians[layer], atol=1e-6,
        )


def test_oom_after_committed_rows_resumes_without_double_counting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import saklas.core.jlens as jlens_module

    model = _frozen_model(n_layers=2)
    tokenizer = _CharTokenizer()
    prompts = ["a prompt that is long enough."]
    expected = fit_jacobian_lens(
        model, tokenizer, prompts, _layers(model), dim_batch=3,
    )
    real_block = jlens_module._grad_row_block
    calls = 0
    injected = False

    def flaky_block(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls, injected
        calls += 1
        if calls == 2 and not injected:
            injected = True
            raise RuntimeError("synthetic out of memory")
        return real_block(*args, **kwargs)

    monkeypatch.setattr(jlens_module, "_grad_row_block", flaky_block)
    resumed = fit_jacobian_lens(
        model, tokenizer, prompts, _layers(model), dim_batch=3,
    )
    assert injected
    for layer in expected.source_layers:
        assert torch.allclose(
            resumed.jacobians[layer], expected.jacobians[layer], atol=1e-6,
        )


def test_committed_row_oom_at_dim_one_restarts_with_smaller_prompt_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import saklas.core.jlens as jlens_module

    model = _frozen_model(n_layers=2)
    tokenizer = _CharTokenizer()
    prompts = [
        "a first prompt that is long enough.",
        "a second prompt that is long enough.",
    ]
    expected = fit_jacobian_lens(
        model, tokenizer, prompts, _layers(model),
        dim_batch=1, prompt_batch=1,
    )
    real_block = jlens_module._grad_row_block
    calls = 0

    def _oom_once(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("DefaultCPUAllocator: can't allocate memory")
        return real_block(*args, **kwargs)

    monkeypatch.setattr(jlens_module, "_grad_row_block", _oom_once)
    resumed = fit_jacobian_lens(
        model, tokenizer, prompts, _layers(model),
        dim_batch=1, prompt_batch=2,
    )
    for layer in expected.source_layers:
        assert torch.allclose(
            resumed.jacobians[layer], expected.jacobians[layer], atol=1e-6,
        )


def test_late_committed_row_oom_splits_only_current_group_without_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A late width backoff must not erase already completed microbatches."""
    import saklas.core.jlens as jlens_module

    model = _frozen_model(n_layers=2)
    tokenizer = _CharTokenizer()
    prompts = [
        "the first prompt is comfortably long enough",
        "the second prompt is comfortably long enough",
        "the third prompt is comfortably long enough",
        "the fourth prompt is comfortably long enough",
    ]
    expected = fit_jacobian_lens(
        model, tokenizer, prompts, _layers(model),
        dim_batch=1, prompt_batch=1,
    )
    real_block = jlens_module._grad_row_block
    real_accumulate = jlens_module._accumulate_prompt_jacobian
    grad_calls = 0
    task_calls: list[tuple[int, int]] = []

    def _late_oom(*args: Any, **kwargs: Any) -> Any:
        nonlocal grad_calls
        grad_calls += 1
        # The toy hidden size is six: calls 1..6 complete the first B=2
        # microbatch, call 7 commits row zero of the second, and call 8 fails.
        if grad_calls == 8:
            raise RuntimeError("DefaultCPUAllocator: can't allocate memory")
        return real_block(*args, **kwargs)

    def _count_tasks(*args: Any, **kwargs: Any) -> Any:
        ids = args[1]
        task_calls.append((int(ids.shape[0]), int(kwargs.get("row_start", 0))))
        return real_accumulate(*args, **kwargs)

    monkeypatch.setattr(jlens_module, "_grad_row_block", _late_oom)
    monkeypatch.setattr(
        jlens_module, "_accumulate_prompt_jacobian", _count_tasks,
    )
    resumed = fit_jacobian_lens(
        model, tokenizer, prompts, _layers(model),
        dim_batch=1, prompt_batch=2,
    )

    # The first B=2 task is never replayed. Only the failed second task is
    # replaced by two B=1 suffix tasks sharing its committed row boundary.
    assert task_calls == [(2, 0), (2, 0), (1, 1), (1, 1)]
    assert grad_calls == 18  # 6 + (1 success + 1 OOM) + 2 * 5 suffix rows
    assert resumed.n_prompts == len(prompts)
    for layer in expected.source_layers:
        assert torch.allclose(
            resumed.jacobians[layer], expected.jacobians[layer], atol=1e-6,
        )


def test_cuda_oom_downgrades_transfer_buffers_to_one_slot() -> None:
    import saklas.core.jlens as jlens_module

    first_device = {0: object()}
    first_host = {0: object()}
    state: dict[str, Any] = {
        "stripe_rows": [first_device, {0: object()}],
        "host_stripes": [first_host, {0: object()}],
    }

    assert jlens_module._downgrade_cuda_stripe_buffers(
        state, torch.device("cuda"),
    )
    assert state["stripe_rows"] == [first_device]
    assert state["host_stripes"] == [first_host]
    assert not jlens_module._downgrade_cuda_stripe_buffers(
        state, torch.device("cuda"),
    )


def test_row_stripe_capacity_is_byte_bounded_and_never_splits_a_vjp() -> None:
    import saklas.core.jlens as jlens_module

    budget = 128 * 1024**2
    capacity = jlens_module._row_stripe_capacity(
        8192, 80, 8, byte_budget=budget,
    )

    assert 8 <= capacity < jlens_module._ROW_STRIPE
    assert capacity * 8192 * 80 * 4 <= budget
    # If one VJP block itself exceeds the budget, exactness wins: staging must
    # still hold that complete block and estimator dim-batch backoff owns it.
    assert jlens_module._row_stripe_capacity(
        8192, 80, 64, byte_budget=1024,
    ) == 64


def test_row_stripe_allocation_backoff_terminates_at_vjp_width() -> None:
    import saklas.core.jlens as jlens_module

    capacity = 51
    seen: list[int] = []
    while True:
        seen.append(capacity)
        smaller = jlens_module._smaller_row_stripe_capacity(capacity, 8)
        if smaller is None:
            break
        capacity = smaller

    assert seen == [51, 25, 12, 8]


def test_one_slot_oom_releases_staging_and_caps_retry_capacity() -> None:
    import saklas.core.jlens as jlens_module

    state: dict[str, Any] = {
        "stripe_rows": [{0: object()}],
        "host_stripes": [{0: object()}],
        "cuda_transfer_stream": object(),
        "stripe_capacity": 51,
    }

    assert jlens_module._shrink_device_stripe_buffers(
        state, torch.device("cuda"), 8,
    )
    assert state["stripe_rows"] is None
    assert state["host_stripes"] is None
    assert state["cuda_transfer_stream"] is None
    assert state["stripe_capacity_limit"] == 25
    assert not jlens_module._shrink_device_stripe_buffers(
        {"stripe_capacity": 8}, torch.device("mps"), 8,
    )


def test_repeated_oom_after_committed_rows_never_double_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import saklas.core.jlens as jlens_module

    model = _frozen_model(n_layers=2)
    tokenizer = _CharTokenizer()
    prompts = [
        "a first prompt that is long enough.",
        "a second prompt that is long enough.",
    ]
    expected = fit_jacobian_lens(
        model, tokenizer, prompts, _layers(model),
        dim_batch=2, prompt_batch=1,
    )
    real_block = jlens_module._grad_row_block
    calls = 0

    def _oom_twice(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls in {2, 3}:
            raise RuntimeError("synthetic out of memory")
        return real_block(*args, **kwargs)

    monkeypatch.setattr(jlens_module, "_grad_row_block", _oom_twice)
    resumed = fit_jacobian_lens(
        model, tokenizer, prompts, _layers(model),
        dim_batch=2, prompt_batch=2,
    )
    for layer in expected.source_layers:
        assert torch.allclose(
            resumed.jacobians[layer], expected.jacobians[layer], atol=1e-6,
        )


@pytest.mark.parametrize("fail_call", [1, 3])
def test_auto_scalar_fallback_stays_single_prompt_and_never_double_counts(
    monkeypatch: pytest.MonkeyPatch, fail_call: int,
) -> None:
    import saklas.core.jlens as jlens_module

    model = _frozen_model(n_layers=2)
    tokenizer = _CharTokenizer()
    prompts = [
        "a first prompt that is long enough.",
        "a second prompt that is also long enough..",
        "a third prompt for fallback continuation...",
    ]
    expected = fit_jacobian_lens(
        model, tokenizer, prompts, _layers(model),
        dim_batch=2, prompt_batch=3, vjp_mode="batched",
    )
    real_block = jlens_module._grad_row_block
    calls = 0
    injected = False

    def unsupported_once(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls, injected
        calls += 1
        if calls == fail_call and not injected:
            injected = True
            raise RuntimeError("vmap batching rule not implemented")
        return real_block(*args, **kwargs)

    monkeypatch.setattr(jlens_module, "_grad_row_block", unsupported_once)
    fallback = fit_jacobian_lens(
        model, tokenizer, prompts, _layers(model),
        dim_batch=2, prompt_batch=3, vjp_mode="auto",
    )
    assert injected
    assert fallback.n_prompts == len(prompts)
    for layer in expected.source_layers:
        assert torch.allclose(
            fallback.jacobians[layer], expected.jacobians[layer], atol=1e-6,
        )


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


def test_fit_checkpoint_accumulator_callback_fires() -> None:
    model = _frozen_model(n_layers=2)
    tokenizer = _CharTokenizer()
    prompt = "a prompt that is long enough."
    seen: list[int] = []

    fit_jacobian_lens(
        model, tokenizer, [prompt] * 4, _layers(model),
        dim_batch=3, prompt_batch=3, checkpoint_every=2,
        checkpoint_accumulator_cb=lambda _sums, count, _dim: seen.append(count),
    )
    # Cadence no longer fractures a healthy width-3 graph merely to stop at 2.
    assert seen == [3, 4]


def test_fit_resume_reuses_prefix_matrices_as_single_accumulator() -> None:
    tokenizer = _CharTokenizer()
    prompt = "a prompt that is long enough."
    base_model = _frozen_model(n_layers=2)
    base = fit_jacobian_lens(
        base_model, tokenizer, [prompt], _layers(base_model),
        dim_batch=3, prompt_batch=1,
    )
    pointers = {
        layer: tensor.data_ptr() for layer, tensor in base.jacobians.items()
    }

    resumed_model = _frozen_model(n_layers=2)
    resumed = fit_jacobian_lens(
        resumed_model, tokenizer, [prompt] * 2, _layers(resumed_model),
        dim_batch=3, prompt_batch=1, initial_lens=base,
    )
    full_model = _frozen_model(n_layers=2)
    full = fit_jacobian_lens(
        full_model, tokenizer, [prompt] * 3, _layers(full_model),
        dim_batch=3, prompt_batch=1,
    )

    assert resumed.n_prompts == 3
    for layer in resumed.source_layers:
        assert resumed.jacobians[layer].data_ptr() == pointers[layer]
        assert torch.allclose(resumed.jacobians[layer], full.jacobians[layer])


def test_fit_skips_terminal_periodic_checkpoint() -> None:
    model = _frozen_model(n_layers=2)
    tokenizer = _CharTokenizer()
    prompt = "a prompt that is long enough."
    seen: list[int] = []

    fit_jacobian_lens(
        model, tokenizer, [prompt] * 2, _layers(model),
        dim_batch=3, prompt_batch=2, checkpoint_every=2,
        suppress_terminal_checkpoint=True,
        checkpoint_accumulator_cb=lambda _sums, count, _dim: seen.append(count),
    )

    assert seen == []


def test_fit_cancellation_persists_completed_prefix_and_removes_hooks() -> None:
    import threading

    from saklas.core.jlens import JacobianLensCancelled

    model = _frozen_model(n_layers=2)
    event = threading.Event()
    saved: list[int] = []

    def _progress(_message: str) -> None:
        event.set()

    with pytest.raises(JacobianLensCancelled, match="after 1 prompts"):
        fit_jacobian_lens(
            model,
            _CharTokenizer(),
            ["a prompt that is long enough."] * 3,
            _layers(model),
            dim_batch=3,
            prompt_batch=1,
            checkpoint_every=3,
            checkpoint_accumulator_cb=(
                lambda _sums, count, _dim: saved.append(count)
            ),
            on_progress=_progress,
            cancel_event=event,
        )

    assert saved == [1]
    assert all(not layer._forward_hooks for layer in _layers(model))
    assert all(not layer._forward_pre_hooks for layer in _layers(model))


def test_fit_cancellation_interrupts_an_active_prompt_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A width-2 fit must not make cancel wait for both full prompt sweeps."""
    import threading

    import saklas.core.jlens as jlens_module
    from saklas.core.jlens import JacobianLensCancelled

    model = _frozen_model(n_layers=2)
    event = threading.Event()
    real_block = jlens_module._grad_row_block
    grad_calls = 0

    def _cancel_after_first_block(*args: Any, **kwargs: Any) -> Any:
        nonlocal grad_calls
        result = real_block(*args, **kwargs)
        grad_calls += 1
        event.set()
        return result

    monkeypatch.setattr(
        jlens_module, "_grad_row_block", _cancel_after_first_block,
    )
    with pytest.raises(JacobianLensCancelled, match="active prompt group"):
        fit_jacobian_lens(
            model,
            _CharTokenizer(),
            ["a prompt that is long enough."] * 2,
            _layers(model),
            dim_batch=1,
            prompt_batch=2,
            cancel_event=event,
        )

    assert grad_calls == 1
    assert all(not layer._forward_hooks for layer in _layers(model))
    assert all(not layer._forward_pre_hooks for layer in _layers(model))


def test_fit_cancellation_after_terminal_group_prevents_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancel in the final return window must not be silently ignored."""
    import threading

    import saklas.core.jlens as jlens_module
    from saklas.core.jlens import JacobianLensCancelled

    model = _frozen_model(n_layers=2)
    event = threading.Event()
    real_accumulate = jlens_module._accumulate_prompt_jacobian

    def _cancel_at_group_return(*args: Any, **kwargs: Any) -> Any:
        result = real_accumulate(*args, **kwargs)
        event.set()
        return result

    monkeypatch.setattr(
        jlens_module, "_accumulate_prompt_jacobian", _cancel_at_group_return,
    )
    with pytest.raises(JacobianLensCancelled, match="after the active"):
        fit_jacobian_lens(
            model,
            _CharTokenizer(),
            ["a prompt that is long enough."] * 2,
            _layers(model),
            dim_batch=3,
            prompt_batch=2,
            cancel_event=event,
        )

    assert all(not layer._forward_hooks for layer in _layers(model))
    assert all(not layer._forward_pre_hooks for layer in _layers(model))


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


def test_aggregate_readout_strength_is_mean_probability() -> None:
    logits = torch.randn(3, _VOCAB)
    depths = [0.4, 0.6, 0.8]
    rows = aggregate_readout(logits, depths, top_k=_VOCAB)
    probs = logits.softmax(dim=-1).mean(dim=0)
    got = {tok: s for tok, s, _, _ in rows}
    assert len(got) == _VOCAB
    for tok, s in got.items():
        assert s == pytest.approx(float(probs[tok]), abs=1e-6)
    # sorted by descending strength
    strengths = [s for _, s, _, _ in rows]
    assert strengths == sorted(strengths, reverse=True)


def test_aggregate_readout_reuses_calibrated_probabilities_exactly() -> None:
    logits = torch.randn(5, _VOCAB)
    depths = [0.41, 0.52, 0.63, 0.74, 0.85]
    expected = aggregate_readout(logits, depths, top_k=4)
    got = aggregate_readout_from_probabilities(
        readout_probabilities(logits), depths, top_k=4,
    )
    assert got == expected


def test_token_readout_stats_uses_exact_columns_without_full_probabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import saklas.core.jlens as jlens_module

    logits = torch.randn(
        4, _VOCAB, generator=torch.Generator().manual_seed(616),
    )
    depths = [0.41, 0.52, 0.63, 0.74]
    token_ids = [1, 4, 1]
    expected = token_readout_stats_from_probabilities(
        readout_probabilities(logits), depths, token_ids,
    )

    def _fail_full_probabilities(_logits: torch.Tensor) -> torch.Tensor:
        raise AssertionError("fixed-token stats should not full-softmax")

    monkeypatch.setattr(
        jlens_module, "readout_probabilities", _fail_full_probabilities,
    )

    got = token_readout_stats(logits, depths, token_ids)

    assert len(got) == len(expected)
    for got_row, expected_row in zip(got, expected):
        assert got_row[:3] == pytest.approx(expected_row[:3])
        assert got_row[3] == pytest.approx(expected_row[3])


def test_aggregate_readout_com_tracks_where_a_token_leads() -> None:
    # Token 0 dominates the earliest layer, token 1 the latest. Sharp
    # logits concentrate each layer's probability mass on its leader, so
    # each token's depth CoM sits at the layer(s) it leads.
    depths = [0.2, 0.5, 0.8]
    logits = torch.full((3, _VOCAB), -10.0)
    logits[0, 0] = 10.0
    logits[1, 3] = 10.0
    logits[2, 1] = 10.0
    rows = {tok: (com, spread) for tok, _, com, spread in
            aggregate_readout(logits, depths, top_k=_VOCAB)}
    com0, _ = rows[0]
    com1, _ = rows[1]
    assert com0 < 0.3          # leads only the early layer
    assert com1 > 0.7          # leads only the late layer
    assert com0 < rows[3][0] < com1  # mid-layer leader sits between


def test_aggregate_readout_single_layer_degenerates() -> None:
    logits = torch.randn(1, _VOCAB)
    rows = aggregate_readout(logits, [0.55], top_k=4)
    assert len(rows) == 4
    for _, strength, com, spread in rows:
        assert 0.0 <= strength <= 1.0
        assert com == pytest.approx(0.55, abs=1e-6)
        assert spread == pytest.approx(0.0, abs=1e-6)


def test_aggregate_readout_com_is_probability_mass_weighted() -> None:
    # The design point (2026-07-09): the band readout is sharp, and what
    # changes over depth is WHICH token leads — so depth CoM weights by
    # the per-layer probability itself, the same channel behind strength.
    # A fully-diffuse layer (uniform readout — nothing is being "said")
    # must not pin any token's depth: its vote is discounted by its own
    # lack of mass. The former within-layer salience handed the uniform
    # layer a FULL vote for every token (sal = 1.0 at the tied max),
    # reading token 0 early (com ≈ 0.33 here) even though its readout
    # probability concentrates late. (At the toy vocab of 13 the uniform
    # layer still carries p = 1/13 per token; at a real ~260k vocab the
    # discount is ~4e-6 and the pull toward the sharp layer is total.)
    depths = [0.3, 0.9]
    logits = torch.zeros(2, _VOCAB)   # layer 0 uniform (fully diffuse)
    logits[1, 1] = 12.0               # sharp late layer led by token 1
    logits[1, 0] = 9.0                # token 0 present but not leading
    p = logits.softmax(dim=-1)
    d = torch.tensor(depths)
    rows = {tok: com for tok, _, com, _ in
            aggregate_readout(logits, depths, top_k=_VOCAB)}
    for tok in (0, 1):
        expected = float((p[:, tok] * d).sum() / p[:, tok].sum())
        assert rows[tok] == pytest.approx(expected, abs=1e-6)
    # token 0's depth is pulled past the uniform layer toward its mass
    assert rows[0] > 0.5
    assert rows[1] > 0.8


def test_aggregate_readout_validates_shapes() -> None:
    with pytest.raises(ValueError):
        aggregate_readout(torch.randn(_VOCAB), [0.5], top_k=3)
    with pytest.raises(ValueError):
        aggregate_readout(torch.randn(2, _VOCAB), [0.5], top_k=3)
    with pytest.raises(ValueError):
        aggregate_readout(torch.randn(0, _VOCAB), [], top_k=3)


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
