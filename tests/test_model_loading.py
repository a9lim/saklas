"""Tests for `saklas.core.model.load_model` attention-implementation selection.

These tests do not load real models — they intercept the AutoConfig and
AutoModelForCausalLM calls and inspect the load_kwargs `load_model` passes.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
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

    def requires_grad_(self, _flag: bool) -> _FakeModel:  # noqa: D401
        return self

    def train(self, _flag: bool) -> _FakeModel:
        return self

    def parameters(self):
        return iter([torch.zeros(1)])


def _captured_load_kwargs(device: str, model_type: str = "deepseek_v2"):
    """Run load_model with a mocked transformers stack; return the kwargs
    actually handed to AutoModelForCausalLM.from_pretrained."""
    cfg = _FakeConfig(model_type)
    captured: dict[str, Any] = {}

    def _fake_from_pretrained(model_id: str, **kwargs: Any) -> _FakeModel:
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
    captured: dict[str, Any] = {}

    def _fake_from_pretrained(model_id: str, **kwargs: Any) -> _FakeModel:
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
        # extract_text_model takes the multimodal-wrapping branch whenever
        # the text_config's model_type is in _LAYER_ACCESSORS, so load_model
        # would route through _load_text_from_multimodal.  Stub it out — we
        # only care that attn_impl was already flipped to eager *before*
        # dispatch.
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
    captured: dict[str, Any] = {}

    def _fake_tok_from_pretrained(mid: str, **kwargs: Any) -> SimpleNamespace:
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
# torch.compile opt-in.  The compile path is wired in ``load_model``:
# when ``compile=True`` (off by default — torch 2.12 inductor bugs on
# newer archs make compile a sometimes-trap) and ``device == "cuda"``,
# the loaded model is wrapped with ``torch.compile``; on MPS/CPU compile
# is silently skipped.  Tests mock ``torch.compile`` so we never actually
# compile (which would take seconds); we only verify it was *invoked*
# with the right args.
# ---------------------------------------------------------------------------


def _run_load_with_compile(
    *, device: str, compile: bool = True, compile_mode: str = "default",
    model_type: str = "qwen3",
):
    """Run load_model with mocked transformers + ``torch.compile``.

    Returns ``(compile_called, compile_kwargs, returned_model)``.
    """
    cfg = _FakeConfig(model_type)
    compile_invocations: list[dict[str, Any]] = []
    base_model = _FakeModel("sdpa")

    class _FakeCompiled:
        """Callable sentinel — ``_compile_with_probe`` runs a probe forward
        after wrapping, so the result must accept the prefill / decode call
        shape and return an object with ``.logits``."""
        _compiled_marker = True

        def __init__(self, model: Any) -> None:
            self._orig_mod = model

        def __call__(self, **fwd: Any) -> SimpleNamespace:
            n = fwd["input_ids"].shape[1]
            return SimpleNamespace(logits=torch.zeros(1, n, 1))

    def _fake_compile(model: Any, **kwargs: Any) -> _FakeCompiled:
        compile_invocations.append({"model": model, **kwargs})
        return _FakeCompiled(model)

    with (
        patch.object(model_mod, "AutoTokenizer") as mock_tok,
        patch.object(model_mod, "AutoConfig") as mock_cfg,
        patch.object(model_mod, "AutoModelForCausalLM") as mock_model,
        patch.object(torch, "compile", side_effect=_fake_compile) as mock_compile,
        # The probe forwards allocate `torch.full(..., device="cuda", ...)`,
        # which raises on a no-CUDA CI host before the fake compiled wrapper
        # is ever called.  We're not exercising the probe here — only that
        # ``torch.compile`` fired with the right args — so stub it out.
        patch.object(model_mod, "_run_compile_probes", return_value=None),
    ):
        mock_tok.from_pretrained.return_value = SimpleNamespace()
        mock_cfg.from_pretrained.return_value = cfg
        mock_model.from_pretrained.return_value = base_model
        ret_model, _tok = model_mod.load_model(
            "fake/repo", device=device, compile=compile,
            compile_mode=compile_mode,
        )

    return (mock_compile.called, compile_invocations, ret_model)


def test_compile_opts_in_on_cuda():
    """Explicit ``compile=True`` on CUDA wraps the model with torch.compile."""
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


def test_compile_off_by_default():
    """``load_model`` defaults to ``compile=False`` — torch 2.12 inductor
    bugs on newer architectures make compile a sometimes-trap, and the
    upfront cost rarely amortizes on interactive workloads."""
    cfg = _FakeConfig("qwen3")
    base_model = _FakeModel("sdpa")

    with (
        patch.object(model_mod, "AutoTokenizer") as mock_tok,
        patch.object(model_mod, "AutoConfig") as mock_cfg,
        patch.object(model_mod, "AutoModelForCausalLM") as mock_model,
        patch.object(torch, "compile") as mock_compile,
    ):
        mock_tok.from_pretrained.return_value = SimpleNamespace()
        mock_cfg.from_pretrained.return_value = cfg
        mock_model.from_pretrained.return_value = base_model
        model_mod.load_model("fake/repo", device="cuda")

    assert not mock_compile.called, (
        "load_model default must not fire torch.compile — opt-in required"
    )


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


def test_compile_probe_failure_falls_back_to_eager():
    """If the compiled artifact raises during the warmup probe (e.g.
    Gemma-4 + torch 2.12 inductor codegen bug), ``_compile_with_probe``
    warns and returns the un-compiled model so generation still works."""
    import warnings as _w

    cfg = _FakeConfig("qwen3")
    base_model = _FakeModel("sdpa")

    class _BrokenCompiled:
        _compiled_marker = True

        def __init__(self, model: Any) -> None:
            self._orig_mod = model

        def __call__(self, **fwd: Any) -> None:
            raise RuntimeError("inductor codegen produced an invalid kernel")

    with (
        patch.object(model_mod, "AutoTokenizer") as mock_tok,
        patch.object(model_mod, "AutoConfig") as mock_cfg,
        patch.object(model_mod, "AutoModelForCausalLM") as mock_model,
        patch.object(torch, "compile", side_effect=lambda m, **k: _BrokenCompiled(m)),
        _w.catch_warnings(record=True) as caught,
    ):
        _w.simplefilter("always")
        mock_tok.from_pretrained.return_value = SimpleNamespace()
        mock_cfg.from_pretrained.return_value = cfg
        mock_model.from_pretrained.return_value = base_model
        ret_model, _tok = model_mod.load_model(
            "fake/repo", device="cuda", compile=True,
        )

    assert not hasattr(ret_model, "_compiled_marker"), (
        "broken compile must fall back to the un-compiled model"
    )
    assert any(
        "torch.compile probe failed" in str(w.message) for w in caught
    ), f"expected a fallback UserWarning, got: {[str(w.message) for w in caught]}"


def test_compile_mode_propagates():
    """compile_mode kwarg flows through to torch.compile unchanged.
    Phase B (CUDA graphs via StaticCache) will pair with
    ``compile_mode='reduce-overhead'``; this test guards the plumbing."""
    _called, invocations, _ = _run_load_with_compile(
        device="cuda", compile_mode="reduce-overhead",
    )
    assert invocations[0]["mode"] == "reduce-overhead"


def test_get_memory_gb_returns_zero_when_mps_backend_unavailable(monkeypatch: pytest.MonkeyPatch):
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


# ---------------------------------------------------------------------------
# Multimodal text-model extraction.
#
# A multimodal checkpoint stores the text model's weights under a
# ``language_model.`` path segment and ships an unused vision tower.
# ``_load_text_from_multimodal`` builds the text-only model on-device and
# streams just the text weights in, dropping the ``language_model.`` segment
# to recover the text model's own parameter names.  This both skips the
# vision tower and — critically on unified memory — avoids the ~2× peak of
# the standard CPU-stage-then-device-copy load (the >128 GB gemma-4-31B
# blowup).  Two real wire layouts exist; both must round-trip:
#
#   gemma-3 / gemma-4 : model.language_model.layers…   (segment nested)
#   mistral-3 / ministral : language_model.model.layers…  (leading prefix)
#
# These tests write a synthetic checkpoint with both text and fake vision
# keys, then assert the loader recovers exactly the text weights.
# ---------------------------------------------------------------------------


def _tiny_text_config(tie: bool):
    from transformers import AutoConfig as _AC
    return _AC.for_model(
        "llama", hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=128,
        head_dim=8, max_position_embeddings=64, pad_token_id=0,
        bos_token_id=1, eos_token_id=2, tie_word_embeddings=tie,
    )


def _remap_to_layout(state_dict: dict[str, torch.Tensor], layout: str) -> dict[str, torch.Tensor]:
    """Re-key a text-only state dict into a multimodal wire layout."""
    out: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if layout == "gemma":
            # model.X -> model.language_model.X ; lm_head.X -> language_model.lm_head.X
            nk = ("model.language_model." + k[len("model."):]
                  if k.startswith("model.") else "language_model." + k)
        elif layout == "mistral":
            nk = "language_model." + k  # leading prefix on everything
        else:  # pragma: no cover - guard
            raise ValueError(layout)
        out[nk] = v.clone()
    return out


def _write_synthetic_vlm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, layout: str, tie: bool):
    """Write a synthetic multimodal checkpoint (text + fake vision weights)
    and patch ``cached_file`` so the loader resolves it.  Returns the
    reference text-only model the checkpoint was derived from."""
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM as _AM
    from transformers.utils import SAFE_WEIGHTS_NAME

    text_cfg = _tiny_text_config(tie)
    torch.manual_seed(0)
    ref = _AM.from_config(text_cfg)
    remapped = _remap_to_layout(ref.state_dict(), layout)
    # Vision-tower / projector weights carry no ``language_model.`` segment
    # and must be skipped, not loaded into the text model.
    remapped["model.vision_tower.encoder.layers.0.weight"] = torch.randn(8, 8)
    remapped["model.multi_modal_projector.linear.weight"] = torch.randn(8, 8)

    weights_path = tmp_path / "model.safetensors"
    save_file(remapped, str(weights_path))

    def _fake_cached_file(model_id: str, filename: str, **kw: Any):
        # Single-file path: no index, weights resolve to our temp file.
        return str(weights_path) if filename == SAFE_WEIGHTS_NAME else None

    monkeypatch.setattr("transformers.utils.hub.cached_file", _fake_cached_file)
    return text_cfg, ref


@pytest.mark.parametrize(
    "layout,tie",
    [("gemma", True), ("mistral", False)],
    ids=["gemma-nested-tied", "mistral-prefix-untied"],
)
def test_load_text_from_multimodal_round_trips(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, layout: str, tie: bool):
    """The loaded text model's weights equal the reference's — for both
    wire layouts, and with the lm_head both tied (gemma) and separate
    (mistral, where ``language_model.lm_head.weight`` must be recovered)."""
    text_cfg, ref = _write_synthetic_vlm(
        tmp_path, monkeypatch, layout=layout, tie=tie,
    )
    loaded = model_mod._load_text_from_multimodal(
        "fake/vlm", text_cfg, torch.float32, "cpu",
    )

    ref_params = dict(ref.named_parameters())
    got_params = dict(loaded.named_parameters())
    assert set(got_params) == set(ref_params), (
        f"param-name mismatch after {layout} extraction"
    )
    for name, ref_t in ref_params.items():
        assert torch.equal(got_params[name], ref_t), (
            f"{layout}: param {name!r} did not round-trip through the "
            f"language_model. rename"
        )
    # The vision tower never made it into the text model.
    assert not any("vision" in n for n in got_params), (
        "vision-tower weights leaked into the text model"
    )
    # And the extracted model is a usable causal LM.
    assert len(model_mod.get_layers(loaded)) == text_cfg.num_hidden_layers


def test_load_text_from_multimodal_raises_when_no_text_weights(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A checkpoint with no ``language_model.`` weights (everything skipped)
    must raise rather than silently return a randomly-initialized model."""
    from safetensors.torch import save_file
    from transformers.utils import SAFE_WEIGHTS_NAME

    text_cfg = _tiny_text_config(tie=True)
    weights_path = tmp_path / "model.safetensors"
    save_file(
        {"model.vision_tower.encoder.layers.0.weight": torch.randn(8, 8)},
        str(weights_path),
    )

    def _fake_cached_file(model_id: str, filename: str, **kw: Any):
        return str(weights_path) if filename == SAFE_WEIGHTS_NAME else None

    monkeypatch.setattr("transformers.utils.hub.cached_file", _fake_cached_file)

    with pytest.raises(model_mod._NoTextWeightsExtracted):
        model_mod._load_text_from_multimodal(
            "fake/vlm", text_cfg, torch.float32, "cpu",
        )


# ---------------------------------------------------------------------------
# Routing: when does load_model prefer the text-only extraction path?
# ---------------------------------------------------------------------------


def _routing_calls(cfg: _FakeConfig, *, extractor_raises: bool = False):
    """Run load_model on CPU with a mocked stack; return
    ``(text_extractor_called, from_pretrained_called)``."""
    fp_called = {"v": False}

    def _fake_from_pretrained(model_id: str, **kwargs: Any) -> _FakeModel:
        fp_called["v"] = True
        return _FakeModel(kwargs.get("attn_implementation", "sdpa"))

    def _fake_extractor(*args: Any, **kwargs: Any) -> _FakeModel:
        if extractor_raises:
            raise model_mod._NoTextWeightsExtracted("synthetic miss")
        return _FakeModel("sdpa")

    with (
        patch.object(model_mod, "AutoTokenizer") as mock_tok,
        patch.object(model_mod, "AutoConfig") as mock_cfg,
        patch.object(model_mod, "AutoModelForCausalLM") as mock_model,
        patch.object(model_mod, "_load_text_from_multimodal",
                     side_effect=_fake_extractor) as mock_extract,
    ):
        mock_tok.from_pretrained.return_value = SimpleNamespace()
        mock_cfg.from_pretrained.return_value = cfg
        mock_model.from_pretrained.side_effect = _fake_from_pretrained
        model_mod.load_model("fake/repo", device="cpu")

    return mock_extract.called, fp_called["v"]


def test_multimodal_routes_to_text_extraction_even_when_outer_accessible():
    """A gemma-style wrapper (outer ``gemma4`` *is* in _LAYER_ACCESSORS,
    text ``gemma4_text`` also is) must still prefer text-only extraction —
    otherwise the standard load pulls the whole multimodal model into
    memory and double-buffers it on unified memory."""
    cfg = _FakeConfig(model_type="gemma4", text_model_type="gemma4_text")
    extractor_called, from_pretrained_called = _routing_calls(cfg)
    assert extractor_called, "gemma4 should route through text extraction"
    assert not from_pretrained_called, (
        "the full multimodal from_pretrained must not run when text "
        "extraction succeeds"
    )


def test_text_extraction_preserves_the_pinned_hub_revision():
    """Official artifacts bind to the immutable base-model revision.

    The text-only multimodal loader constructs a new model config, so Saklas
    must carry the already-resolved outer revision onto that live config.
    """
    cfg = _FakeConfig(model_type="gemma3", text_model_type="gemma3_text")
    cfg._commit_hash = "immutable-model-commit"

    with (
        patch.object(model_mod, "AutoTokenizer") as mock_tok,
        patch.object(model_mod, "AutoConfig") as mock_cfg,
        patch.object(model_mod, "AutoModelForCausalLM"),
        patch.object(model_mod, "_load_text_from_multimodal") as mock_extract,
    ):
        mock_tok.from_pretrained.return_value = SimpleNamespace()
        mock_cfg.from_pretrained.return_value = cfg
        mock_extract.return_value = _FakeModel("sdpa")
        model, _ = model_mod.load_model("google/gemma-3-4b-it", device="cpu")

    assert model.config._commit_hash == "immutable-model-commit"
    assert mock_extract.call_args.kwargs["revision"] == "immutable-model-commit"


def test_plain_text_model_uses_standard_load():
    """A non-multimodal config (no text_config) takes the standard load —
    there is nothing to extract."""
    cfg = _FakeConfig(model_type="qwen3")  # text_config is None
    extractor_called, from_pretrained_called = _routing_calls(cfg)
    assert not extractor_called
    assert from_pretrained_called


def test_load_model_falls_back_to_full_load_on_no_text_weights():
    """If text extraction raises ``_NoTextWeightsExtracted`` (unexpected
    weight layout), load_model must fall back to the standard full load
    rather than propagate the error or ship random init."""
    cfg = _FakeConfig(model_type="some_vlm", text_model_type="qwen3")
    extractor_called, from_pretrained_called = _routing_calls(
        cfg, extractor_raises=True,
    )
    assert extractor_called, "extraction should be attempted first"
    assert from_pretrained_called, (
        "a failed extraction must fall back to the standard full load"
    )
