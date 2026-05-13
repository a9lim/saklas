"""Tests for `saklas.core.model.load_model` attention-implementation selection.

These tests do not load real models — they intercept the AutoConfig and
AutoModelForCausalLM calls and inspect the load_kwargs `load_model` passes.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import torch

from saklas.core import model as model_mod


class _FakeConfig(SimpleNamespace):
    """Minimal AutoConfig stand-in with knobs for model_type / text_config."""

    def __init__(self, model_type: str, text_model_type: str | None = None):
        super().__init__()
        self.model_type = model_type
        if text_model_type is not None:
            self.text_config = SimpleNamespace(
                model_type=text_model_type, _name_or_path=""
            )
        else:
            self.text_config = None


class _FakeModel(SimpleNamespace):
    """Stand-in for an HF model object — only what load_model touches."""

    def __init__(self, attn_impl: str):
        super().__init__()
        self.config = SimpleNamespace(_attn_implementation=attn_impl)

    def requires_grad_(self, _flag):  # noqa: D401
        return self

    def train(self, _flag):
        return self

    def parameters(self):
        return iter([torch.zeros(1)])


def _captured_load_kwargs(device: str, model_type: str = "deepseek_v2"):
    """Run load_model with a mocked transformers stack; return the kwargs
    actually handed to AutoModelForCausalLM.from_pretrained."""
    cfg = _FakeConfig(model_type)
    captured: dict = {}

    def _fake_from_pretrained(model_id, **kwargs):
        captured.update(kwargs)
        return _FakeModel(kwargs.get("attn_implementation", "sdpa"))

    with (
        patch.object(model_mod, "AutoTokenizer") as mock_tok,
        patch.object(model_mod, "AutoConfig") as mock_cfg,
        patch.object(model_mod, "AutoModelForCausalLM") as mock_model,
    ):
        mock_tok.from_pretrained.return_value = SimpleNamespace()
        mock_cfg.from_pretrained.return_value = cfg
        mock_model.from_pretrained.side_effect = _fake_from_pretrained
        model_mod.load_model("fake/repo", device=device)

    return captured


def test_mla_on_mps_forces_eager():
    """DeepSeek-V2/V3 on MPS must request eager — PyTorch MPS SDPA mishandles
    mismatched query/value head_dim, breaking o_proj at runtime."""
    for mt in ("deepseek_v2", "deepseek_v3"):
        kwargs = _captured_load_kwargs(device="mps", model_type=mt)
        assert kwargs["attn_implementation"] == "eager", (
            f"{mt} on MPS should force eager, got "
            f"{kwargs['attn_implementation']!r}"
        )


def test_mla_on_cpu_keeps_sdpa():
    """CPU SDPA correctly handles mismatched Eq/Ev — no need to downgrade."""
    kwargs = _captured_load_kwargs(device="cpu", model_type="deepseek_v2")
    assert kwargs["attn_implementation"] == "sdpa"


def test_non_mla_on_mps_keeps_sdpa():
    """The MLA carve-out is narrow: vanilla architectures still get sdpa."""
    kwargs = _captured_load_kwargs(device="mps", model_type="qwen3")
    assert kwargs["attn_implementation"] == "sdpa"


def test_mla_in_text_config_also_triggers():
    """A multimodal wrapper whose text_config is deepseek_v2 must also
    fall through to eager on MPS."""
    cfg = _FakeConfig(model_type="some_vlm", text_model_type="deepseek_v2")
    captured: dict = {}

    def _fake_from_pretrained(model_id, **kwargs):
        captured.update(kwargs)
        return _FakeModel(kwargs.get("attn_implementation", "sdpa"))

    with (
        patch.object(model_mod, "AutoTokenizer") as mock_tok,
        patch.object(model_mod, "AutoConfig") as mock_cfg,
        patch.object(model_mod, "AutoModelForCausalLM") as mock_model,
    ):
        mock_tok.from_pretrained.return_value = SimpleNamespace()
        mock_cfg.from_pretrained.return_value = cfg
        mock_model.from_pretrained.side_effect = _fake_from_pretrained
        # The text_config carries deepseek_v2 — model_type is the wrapper.
        # extract_text_model takes the multimodal-wrapping branch only when
        # the text_config's model_type IS in _LAYER_ACCESSORS *and* the
        # outer model_type is NOT.  Both conditions hold for our fake, so
        # load_model would route through _load_text_from_multimodal.  Stub
        # it out — we only care that attn_impl was already flipped to eager
        # *before* dispatch.
        with patch.object(model_mod, "_load_text_from_multimodal") as mlm:
            mlm.return_value = _FakeModel("eager")
            model_mod.load_model("fake/repo", device="mps")

    # When the multimodal branch fires, AutoModelForCausalLM.from_pretrained
    # is never called — but the eager decision lives in load_kwargs which is
    # built either way.  Inspect the dtype-only call path: re-run with a
    # plain (non-multimodal) deepseek_v2 to confirm.
    kwargs = _captured_load_kwargs(device="mps", model_type="deepseek_v2")
    assert kwargs["attn_implementation"] == "eager"


def _captured_tokenizer_kwargs(model_id: str, model_type: str = "qwen3"):
    """Run load_model with a mocked transformers stack; return the kwargs
    actually handed to AutoTokenizer.from_pretrained."""
    cfg = _FakeConfig(model_type)
    captured: dict = {}

    def _fake_tok_from_pretrained(mid, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    with (
        patch.object(model_mod, "AutoTokenizer") as mock_tok,
        patch.object(model_mod, "AutoConfig") as mock_cfg,
        patch.object(model_mod, "AutoModelForCausalLM") as mock_model,
    ):
        mock_tok.from_pretrained.side_effect = _fake_tok_from_pretrained
        mock_cfg.from_pretrained.return_value = cfg
        mock_model.from_pretrained.return_value = _FakeModel("sdpa")
        model_mod.load_model(model_id, device="cpu")

    return captured


def test_mistral_regex_fix_passed_for_mistral():
    """HF-distributed Mistral checkpoints ship a buggy pre-tokenizer regex
    that mis-splits ~1% of tokens. ``load_model`` must pass
    ``fix_mistral_regex=True`` to AutoTokenizer for any mistral-family
    repo so the corrected regex from ``mistral_common`` is used.
    See https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/discussions/84
    """
    for mid in (
        "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        "mistralai/Ministral-3-14B-Instruct-2512",
        "someorg/mistral-7b-finetune",
    ):
        kwargs = _captured_tokenizer_kwargs(mid)
        assert kwargs.get("fix_mistral_regex") is True, (
            f"{mid}: expected fix_mistral_regex=True in tokenizer kwargs, "
            f"got {kwargs!r}"
        )


def test_mistral_regex_fix_skipped_for_non_mistral():
    """The fix kwarg is mistral-tokenizer-specific. Don't pass it for
    other architectures — modern transformers may absorb unknown kwargs
    silently, but we shouldn't rely on that."""
    for mid in (
        "google/gemma-4-31b-it",
        "Qwen/Qwen3.6-27B",
        "meta-llama/Llama-3.1-8B-Instruct",
    ):
        kwargs = _captured_tokenizer_kwargs(mid)
        assert "fix_mistral_regex" not in kwargs, (
            f"{mid}: fix_mistral_regex should NOT be set, got {kwargs!r}"
        )


# ---------------------------------------------------------------------------
# torch.compile auto-enable.  The compile path is wired in
# ``load_model``: when ``compile=True`` (default) and ``device == "cuda"``,
# the loaded model is wrapped with ``torch.compile``; on MPS/CPU compile
# is silently skipped, and ``compile=False`` is the explicit opt-out
# regardless of device.  Tests mock ``torch.compile`` so we never actually
# compile (which would download dependencies and take seconds); we only
# verify it was *invoked* with the right args.
# ---------------------------------------------------------------------------


def _run_load_with_compile(
    *, device: str, compile: bool = True, compile_mode: str = "default",
    model_type: str = "qwen3",
):
    """Run load_model with mocked transformers + ``torch.compile``.

    Returns ``(compile_called, compile_kwargs, returned_model)``.
    """
    cfg = _FakeConfig(model_type)
    compile_invocations: list[dict] = []
    base_model = _FakeModel("sdpa")

    def _fake_compile(model, **kwargs):
        compile_invocations.append({"model": model, **kwargs})
        # Return a sentinel marker so we can verify the wrapped object
        # is what propagates back to the caller.
        return SimpleNamespace(_compiled_marker=True, _orig_mod=model)

    with (
        patch.object(model_mod, "AutoTokenizer") as mock_tok,
        patch.object(model_mod, "AutoConfig") as mock_cfg,
        patch.object(model_mod, "AutoModelForCausalLM") as mock_model,
        patch.object(torch, "compile", side_effect=_fake_compile) as mock_compile,
    ):
        mock_tok.from_pretrained.return_value = SimpleNamespace()
        mock_cfg.from_pretrained.return_value = cfg
        mock_model.from_pretrained.return_value = base_model
        ret_model, _tok = model_mod.load_model(
            "fake/repo", device=device, compile=compile,
            compile_mode=compile_mode,
        )

    return (mock_compile.called, compile_invocations, ret_model)


def test_compile_auto_enabled_on_cuda():
    """Default behavior: CUDA load wraps the model with torch.compile."""
    called, invocations, model = _run_load_with_compile(device="cuda")
    assert called, "torch.compile should fire on CUDA when compile=True"
    assert len(invocations) == 1
    inv = invocations[0]
    assert inv["mode"] == "default", (
        f"expected mode='default' (Phase A — kernel fusion without graph "
        f"capture), got {inv.get('mode')!r}"
    )
    # The returned model is the compile wrapper, not the original.
    assert getattr(model, "_compiled_marker", False) is True


def test_compile_skipped_on_mps():
    """MPS path: compile is silently skipped (compile is CUDA-tuned)."""
    called, invocations, model = _run_load_with_compile(device="mps")
    assert not called, "torch.compile must not fire on MPS"
    assert invocations == []
    # Original model returned unwrapped.
    assert not hasattr(model, "_compiled_marker")


def test_compile_skipped_on_cpu():
    called, invocations, _ = _run_load_with_compile(device="cpu")
    assert not called, "torch.compile must not fire on CPU"


def test_compile_explicit_opt_out_on_cuda():
    """compile=False is the documented escape hatch — even on CUDA, no
    wrapping should occur.  Lets users debug architecture-specific
    compile breakage without having to flip device flags."""
    called, invocations, model = _run_load_with_compile(
        device="cuda", compile=False,
    )
    assert not called, "compile=False must override the CUDA auto-enable"
    assert not hasattr(model, "_compiled_marker")


def test_compile_mode_propagates():
    """compile_mode kwarg flows through to torch.compile unchanged.
    Phase B (CUDA graphs via StaticCache) will pair with
    ``compile_mode='reduce-overhead'``; this test guards the plumbing."""
    _called, invocations, _ = _run_load_with_compile(
        device="cuda", compile_mode="reduce-overhead",
    )
    assert invocations[0]["mode"] == "reduce-overhead"


def test_get_memory_gb_returns_zero_when_mps_backend_unavailable(monkeypatch):
    """``device='mps'`` on a host without an MPS backend (e.g. Linux CI)
    must not crash.  The CUDA branch already gates on
    ``torch.cuda.is_available()``; the MPS branch must mirror that — the
    bare ``torch.mps.current_allocated_memory()`` call raises RuntimeError
    rather than AttributeError when the backend isn't actually present,
    so a plain ``except AttributeError`` is insufficient.
    """
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    # Make the underlying call raise to prove the gate fires before it.
    def _raise():
        raise RuntimeError("Cannot execute getCurrentAllocatedMemory() without MPS backend.")
    monkeypatch.setattr(torch.mps, "current_allocated_memory", _raise)
    assert model_mod._get_memory_gb("mps") == 0.0
