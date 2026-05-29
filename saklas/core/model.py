"""Model loading utilities for activation steering."""

import logging
import warnings
from typing import Any, cast

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import PreTrainedModel, PreTrainedTokenizerBase

log = logging.getLogger(__name__)

_ORIG_HISTC = None
_ORIG_LDEXP = None


def _histc_mps_safe(input: torch.Tensor, bins: int = 100, min: float = 0, max: float = 0, *, out: torch.Tensor | None = None) -> torch.Tensor:
    assert _ORIG_HISTC is not None
    if input.device.type == "mps" and not input.is_floating_point():
        input = input.float()
    return _ORIG_HISTC(input, bins=bins, min=min, max=max, out=out)


def _ldexp_mps_safe(input: Any, other: Any, *, out: torch.Tensor | None = None) -> Any:
    assert _ORIG_LDEXP is not None
    if hasattr(input, "device") and input.device.type == "mps":
        in_cpu = input.cpu()
        other_cpu = other.cpu() if hasattr(other, "device") else other
        res = _ORIG_LDEXP(in_cpu, other_cpu)
        if out is not None:
            out.copy_(res.to(out.device))
            return out
        return res.to(input.device)
    return _ORIG_LDEXP(input, other, out=out) if out is not None \
        else _ORIG_LDEXP(input, other)


def patch_torch_for_mps() -> bool:
    """Install MPS-only torch workarounds lazily and idempotently.

    The patches are process-global because PyTorch exposes these as module
    functions, but saklas now installs them only when an MPS model load is
    actually requested.
    """
    global _ORIG_HISTC, _ORIG_LDEXP
    installed = False
    if getattr(torch.histc, "_saklas_mps_safe", False) is False:
        _ORIG_HISTC = torch.histc
        _histc_mps_safe._saklas_mps_safe = True  # pyright: ignore[reportFunctionMemberAccess]  # FunctionType stub forbids dynamic attrs
        torch.histc = _histc_mps_safe
        installed = True
    if getattr(torch.ldexp, "_saklas_mps_safe", False) is False:
        _ORIG_LDEXP = torch.ldexp
        _ldexp_mps_safe._saklas_mps_safe = True  # pyright: ignore[reportFunctionMemberAccess]  # FunctionType stub forbids dynamic attrs
        torch.ldexp = _ldexp_mps_safe
        installed = True
    return installed

def _MODEL_LAYERS(m: Any) -> Any: return m.model.layers
def _TRANSFORMER_H(m: Any) -> Any: return m.transformer.h
def _VLM_LANGUAGE_LAYERS(m: Any) -> Any: return m.model.language_model.layers

_LAYER_ACCESSORS = {
    # Llama family
    "llama": _MODEL_LAYERS,
    "llama4": _MODEL_LAYERS,
    "llama4_text": _MODEL_LAYERS,
    # Mistral / Mixtral / Ministral
    "mistral": _MODEL_LAYERS,
    "mistral4": _MODEL_LAYERS,
    "ministral": _MODEL_LAYERS,
    "ministral3": _MODEL_LAYERS,
    "mixtral": _MODEL_LAYERS,
    # Gemma family
    "gemma": _MODEL_LAYERS,
    "gemma2": _MODEL_LAYERS,
    "gemma3": _VLM_LANGUAGE_LAYERS,
    "gemma3_text": _MODEL_LAYERS,
    "gemma4": _VLM_LANGUAGE_LAYERS,
    "gemma4_text": _MODEL_LAYERS,
    "recurrent_gemma": _MODEL_LAYERS,
    # Phi family
    "phi": _MODEL_LAYERS,
    "phi3": _MODEL_LAYERS,
    "phimoe": _MODEL_LAYERS,
    # Qwen family
    "qwen": _TRANSFORMER_H,
    "qwen2": _MODEL_LAYERS,
    "qwen2_moe": _MODEL_LAYERS,
    "qwen3": _MODEL_LAYERS,
    "qwen3_moe": _MODEL_LAYERS,
    "qwen3_5": _MODEL_LAYERS,
    "qwen3_5_text": _MODEL_LAYERS,
    "qwen3_5_moe": _MODEL_LAYERS,
    # Cohere (Command-R)
    "cohere": _MODEL_LAYERS,
    "cohere2": _MODEL_LAYERS,
    # DeepSeek
    "deepseek_v2": _MODEL_LAYERS,
    "deepseek_v3": _MODEL_LAYERS,
    # Starcoder
    "starcoder2": _MODEL_LAYERS,
    # OLMo
    "olmo": _MODEL_LAYERS,
    "olmo2": _MODEL_LAYERS,
    "olmo3": _MODEL_LAYERS,
    "olmoe": _MODEL_LAYERS,
    # GLM (ChatGLM)
    "glm": _MODEL_LAYERS,
    "glm4": _MODEL_LAYERS,
    "glm4_moe_lite": _MODEL_LAYERS,
    # Granite (IBM)
    "granite": _MODEL_LAYERS,
    "granitemoe": _MODEL_LAYERS,
    # NVIDIA
    "nemotron": _MODEL_LAYERS,
    # StableLM
    "stablelm": _MODEL_LAYERS,
    # GPT-2 family
    "gpt2": _TRANSFORMER_H,
    "gpt_neo": _TRANSFORMER_H,
    "gptj": _TRANSFORMER_H,
    "gpt_bigcode": _TRANSFORMER_H,
    # Bloom / Falcon
    "bloom": _TRANSFORMER_H,
    "falcon": _TRANSFORMER_H,
    "falcon_h1": _MODEL_LAYERS,
    # GPT-NeoX / Pythia / GPT-OSS
    "gpt_neox": lambda m: m.gpt_neox.layers,
    "gpt_oss": _MODEL_LAYERS,
    # MPT / DBRX
    "mpt": lambda m: m.transformer.blocks,
    "dbrx": lambda m: m.transformer.blocks,
    # OPT
    "opt": lambda m: m.model.decoder.layers,
    # Talkie (vintage pre-1931 model; embedding-skip + per-block actgain decoder)
    "talkie": lambda m: m.model.blocks,
}

_SUPPORTED_TYPES = sorted(_LAYER_ACCESSORS)

# Architectures with end-to-end testing (smoke + session). Everything else in
# _LAYER_ACCESSORS is wired up optimistically — it may work, but has not been
# exercised. See CLAUDE.md "Architecture" section.
_TESTED_ARCHS: frozenset[str] = frozenset({
    "qwen2", "qwen3", "qwen3_5", "qwen3_5_text", "qwen3_5_moe",
    "gemma2", "gemma3", "gemma3_text", "gemma4", "gemma4_text",
    "mistral3", "ministral3", "gpt_oss", "llama", "glm",
    "talkie",
})
_warned: set[str] = set()


_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2,
               torch.float8_e4m3fnuz, torch.float8_e5m2fnuz)

# safetensors dtype tag -> torch dtype, for the manual (mmap-free) reader.
_ST_DTYPES: dict[str, torch.dtype] = {
    "F64": torch.float64, "F32": torch.float32, "F16": torch.float16,
    "BF16": torch.bfloat16, "I64": torch.int64, "I32": torch.int32,
    "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8,
    "BOOL": torch.bool, "F8_E4M3": torch.float8_e4m3fn, "F8_E5M2": torch.float8_e5m2,
}


def _read_safetensors_header(path: str) -> tuple[dict[str, Any], int]:
    """Parse a safetensors header: ``(tensor_metadata, data_section_offset)``.

    The format is an 8-byte little-endian header length, that many bytes
    of JSON (name → ``{dtype, shape, data_offsets}``), then the packed
    tensor data.  The ``__metadata__`` entry is dropped.
    """
    import json
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header, 8 + n


def _open_uncached(path: str) -> int:
    """Open ``path`` read-only, asking the OS not to cache its pages.

    On Apple Silicon the page cache shares the same physical RAM as MPS,
    so mmap-reading a 62 GB checkpoint fills the cache *on top of* the
    61 GB on-device model — ~2× the model size, which forces the unified
    pool into compression/swap.  ``F_NOCACHE`` (macOS) reads straight
    through without populating the unified buffer cache, holding the
    host-side transient to a single tensor.  Elsewhere the page cache is
    reclaimable and doesn't compete with discrete VRAM, so a plain fd is
    fine.
    """
    import os
    import sys
    fd = os.open(path, os.O_RDONLY)
    if sys.platform == "darwin":
        import fcntl
        try:
            fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
        except OSError:
            pass
    return fd


def _read_st_tensor(fd: int, header: dict[str, Any], data_start: int, name: str) -> torch.Tensor:
    """Read one tensor by name from an already-open safetensors fd.

    ``pread`` at the tensor's recorded offset, wrap the bytes as a CPU
    tensor (zero-copy view of the freshly-read buffer), reshape.  No mmap,
    so with an :func:`_open_uncached` fd the bytes never enter the page
    cache.
    """
    import os
    import warnings
    meta = header[name]
    start, end = meta["data_offsets"]
    nbytes = end - start
    offset = data_start + start
    # A single pread can't exceed INT_MAX bytes on macOS (EINVAL), and a
    # bf16 embedding is ~2.8 GB, so read in capped chunks.
    _CHUNK = 1 << 28  # 256 MiB
    chunks: list[bytes] = []
    got = 0
    while got < nbytes:
        block = os.pread(fd, min(_CHUNK, nbytes - got), offset + got)
        if not block:
            break
        chunks.append(block)
        got += len(block)
    buf = chunks[0] if len(chunks) == 1 else b"".join(chunks)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # frombuffer warns on read-only bytes
        flat = torch.frombuffer(buf, dtype=_ST_DTYPES[meta["dtype"]])
    return flat.reshape(meta["shape"])


class _NoTextWeightsExtracted(RuntimeError):
    """No text-model weights matched the ``language_model.`` layout.

    Raised by :func:`_load_text_from_multimodal` so the caller can fall
    back to a standard full-model load rather than return a model full of
    random initialization.
    """


def _load_text_from_multimodal(
    model_id: str,
    text_config: Any,
    dtype: torch.dtype,
    device: str,
):
    """Load just the text model from a multimodal checkpoint.

    Multimodal checkpoints store the text-model weights under a
    ``language_model.`` path *segment* — either as a leading prefix
    (Mistral-3 / Ministral: ``language_model.model.layers…``) or nested
    inside the composite model (Gemma-3 / Gemma-4:
    ``model.language_model.layers…``).  This function builds a text-only
    model directly on the target device, then streams each safetensors
    shard tensor-by-tensor, drops the ``language_model.`` segment to
    recover the text model's own parameter names, dequantizes FP8
    weights, and copies each into the preallocated parameter.  Vision
    tower / projector weights (no ``language_model.`` segment) are
    skipped.

    Two memory disciplines matter on Apple Silicon, where CPU and MPS
    share one unified-memory pool:

    * **On-device construction** — ``from_config`` under
      ``torch.device(device)`` allocates the model once, on the target
      device, so there is never a second full-model copy.  The standard
      ``from_pretrained(device_map={"": "mps"})`` path stages the entire
      state dict in CPU RAM *and* copies it to MPS before freeing the CPU
      side — ~2× the model size at peak (>128 GB for gemma-4-31B).
    * **Cache-bypassing reads** — each shard is read with ``pread`` on an
      :func:`_open_uncached` fd (``F_NOCACHE`` on macOS) rather than
      mmap'd via ``safe_open``.  mmap would fault the whole 62 GB
      checkpoint into the page cache, which on unified memory collides
      with the on-device model; reading straight through holds the
      host-side transient to a single tensor.

    Raises :class:`_NoTextWeightsExtracted` if nothing matched, so the
    caller can fall back to a full multimodal load.
    """
    import gc
    import json
    import os
    from transformers.utils import (
        cached_file,
        SAFE_WEIGHTS_INDEX_NAME,
        SAFE_WEIGHTS_NAME,
    )

    # Create directly on target device — avoids a CPU copy that would
    # spike RSS and eat into MPS's unified memory budget.
    with torch.device(device):
        model = AutoModelForCausalLM.from_config(text_config, dtype=dtype)

    # The text model's own parameter / buffer names → their (already
    # device-resident) tensors, so each shard tensor can be copy_'d in
    # place.  named_parameters() de-dupes tied weights, so a tied lm_head
    # is absent here and rides along on the shared embed_tokens storage.
    targets: dict[str, torch.Tensor] = dict(model.named_parameters())
    targets.update(model.named_buffers())

    # Prefer the sharded index; fall back to a single `model.safetensors`
    # for repos that ship consolidated weights (e.g. Ministral-3-3B).
    index_path = cached_file(
        model_id, SAFE_WEIGHTS_INDEX_NAME, _raise_exceptions_for_missing_entries=False
    )
    if index_path is not None:
        with open(index_path) as f:
            shard_names = sorted(set(json.load(f)["weight_map"].values()))
        # Resolve each shard through cached_file so HF hub downloads it
        # if it isn't already local (the index can land before the shards).
        shard_paths = [cached_file(model_id, name) for name in shard_names]
    else:
        single_path = cached_file(model_id, SAFE_WEIGHTS_NAME)
        shard_paths = [single_path]

    matched = 0
    with torch.no_grad():
        for sf in shard_paths:
            # cached_file returns str | None; None means a missing shard —
            # skip it. In practice the single-shard path raises before here.
            if sf is None:
                continue
            header, data_start = _read_safetensors_header(sf)
            # Read in on-disk offset order so an uncached fd still streams
            # sequentially (no per-tensor seek thrash).
            names = sorted(header, key=lambda nm: header[nm]["data_offsets"][0])
            fd = _open_uncached(sf)
            try:
                for name in names:
                    if "language_model." not in name:
                        continue  # vision tower / projector — skip
                    # Drop the language_model. segment to recover the text
                    # model's own parameter name. Works for both layouts:
                    #   language_model.model.layers… -> model.layers…
                    #   model.language_model.layers… -> model.layers…
                    key = name.replace("language_model.", "", 1)
                    if key.endswith(".weight_scale_inv") or key.endswith(".activation_scale"):
                        continue
                    target = targets.get(key)
                    if target is None:
                        continue
                    v = _read_st_tensor(fd, header, data_start, name)
                    # Dequantize FP8: real_weight = weight.to(dtype) * scale
                    if v.dtype in _FP8_DTYPES:
                        sk = name + "_scale_inv"
                        scale = (_read_st_tensor(fd, header, data_start, sk)
                                 if sk in header else None)
                        v = v.to(dtype) * scale.to(dtype) if scale is not None else v.to(dtype)
                    target.copy_(v)  # CPU→device cast + copy into preallocated param
                    matched += 1
                    del v
            finally:
                os.close(fd)
            if device == "mps":
                torch.mps.empty_cache()
            gc.collect()

    if matched == 0:
        raise _NoTextWeightsExtracted(
            f"no text-model weights matched the language_model. layout in {model_id!r}"
        )

    if device == "mps":
        torch.mps.empty_cache()

    return model


def _run_compile_probes(compiled: Any, model: PreTrainedModel, device: str | torch.device, bos_token_id: int, *, mode: str) -> None:
    """Trigger compilation of the call shapes saklas actually uses.

    For ``mode="reduce-overhead"`` saklas's session generates through
    a ``StaticCache`` with an explicit ``cache_position`` kwarg — that
    is a *distinct dynamo call shape* from a plain DynamicCache
    forward, so a DynamicCache probe will not surface inductor
    codegen bugs that fire only under the static-cache path
    (observed: ``TypeError: Pointer argument must be either uint64 or
    have data_ptr method`` on Qwen3.5 + torch 2.12).  We mimic both
    code paths here so any per-shape codegen failure raises during
    load, not on the user's first generation.

    Each forward is single-shot — we materialize a scalar after each
    call before releasing the output, so the CUDA-graph capture path
    has nothing to alias even under ``mode="reduce-overhead"``.  Two
    distinct prefill lengths force dynamo's automatic-shape promotion
    so the dynamic-shape artifact (which the user will hit) is also
    compiled here.
    """
    def _probe_dynamic(prefill_len: int) -> None:
        ids = torch.full(
            (1, prefill_len), bos_token_id, device=device, dtype=torch.long,
        )
        attn = torch.ones_like(ids)
        out = compiled(input_ids=ids, attention_mask=attn, use_cache=True)
        _ = out.logits[:, -1, :].argmax().item()

    def _probe_static(prefill_len: int) -> None:
        from saklas.core.cuda_graphs import make_static_cache
        dtype = next(model.parameters()).dtype
        try:
            cache = make_static_cache(
                model, max_cache_len=prefill_len + 2,
                device=device, dtype=dtype,
            )
        except Exception as cache_exc:
            # If StaticCache can't be built for this model, the
            # generation loop won't use it either — session
            # construction has already gated on
            # ``is_cuda_graphs_supported`` before settling on
            # ``mode="reduce-overhead"``.  In load_model called
            # standalone or from incomplete mocks the gate is
            # absent, so skip the static probe silently.
            log.info("static-cache probe skipped: %s", cache_exc)
            return
        # Prefill — saklas's generation passes attention_mask on prefill.
        ids = torch.full(
            (1, prefill_len), bos_token_id, device=device, dtype=torch.long,
        )
        attn = torch.ones_like(ids)
        cache_position = torch.arange(prefill_len, device=device)
        out = compiled(
            input_ids=ids, attention_mask=attn,
            past_key_values=cache, use_cache=True,
            cache_position=cache_position,
        )
        # Materialize next-token onto a fresh CPU-side tensor so the
        # decode forward below has nothing aliased to a cudagraph
        # buffer in ``out``.
        next_id = int(out.logits[:, -1, :].argmax().item())
        del out
        # Decode — saklas passes ``attention_mask=None`` on decode steps;
        # that is a *distinct* dynamo call signature from the prefill
        # call, so it gets a separate compile artifact.  Inductor
        # codegen bugs that only fire under the decode-shape path
        # (observed: Qwen3.5 + torch 2.12) won't be caught by a
        # prefill-only probe.
        next_ids = torch.tensor([[next_id]], device=device, dtype=torch.long)
        cache_position = torch.tensor([prefill_len], device=device)
        out2 = compiled(
            input_ids=next_ids, attention_mask=None,
            past_key_values=cache, use_cache=True,
            cache_position=cache_position,
        )
        _ = out2.logits[:, -1, :].argmax().item()

    # Two prefill lengths so dynamo promotes to dynamic-shape codegen.
    for n in (2, 16):
        _probe_dynamic(n)
    if mode == "reduce-overhead":
        for n in (2, 16):
            _probe_static(n)


def _compile_with_probe(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    device: str | torch.device,
    *,
    mode: str = "default",
):
    """``torch.compile`` the model and verify the compiled artifact runs.

    Triggers compilation of both prefill and decode shapes via a tiny
    probe forward pair so that inductor / Triton failures surface here
    rather than at the user's first generation. On failure, warns and
    returns the *un-compiled* model so generation still works.

    Forces ``torch._inductor.config.use_static_cuda_launcher = False``
    for the process before compiling. Torch 2.12's static CUDA launcher
    dereferences kernel args without validation, so an inductor codegen
    bug (e.g. passing a symbolic shape where a tensor pointer is
    expected — observed on Gemma-4) segfaults uncatchably. Routing
    through Triton's regular launcher instead turns that into a
    catchable ``TypeError``, which the fallback below can handle.
    """
    try:
        import torch._dynamo  # noqa: F401
    except ImportError:
        log.info("torch.compile unavailable (no _dynamo); skipping")
        return model

    import torch._inductor.config as _ic
    _ic.use_static_cuda_launcher = False

    log.info("Compiling model with torch.compile(mode=%r)", mode)
    compiled = torch.compile(model, mode=mode, dynamic=None)

    bos = (
        getattr(tokenizer, "bos_token_id", None)
        or getattr(tokenizer, "pad_token_id", None)
        or getattr(tokenizer, "eos_token_id", None)
        or 0
    )
    try:
        with torch.inference_mode():
            _run_compile_probes(compiled, model, device, bos, mode=mode)
    except Exception as exc:
        warnings.warn(
            f"torch.compile probe failed during warmup "
            f"({type(exc).__name__}: {str(exc)[:200]}); falling back to "
            "eager mode. Drop --compile (CLI) or compile=False "
            "(SaklasSession.from_pretrained) to silence this warning.",
            UserWarning,
            stacklevel=2,
        )
        log.info("torch.compile probe failed; using eager", exc_info=True)
        return model
    return compiled


def detect_device(requested: str = "auto") -> str:
    """Pick the best available device.

    'auto' probes in order: cuda > mps > cpu.
    An explicit value ('cuda', 'mps', 'cpu') is returned as-is.
    """
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _pick_dtype(device: str) -> torch.dtype:
    """Best default dtype for a device. bf16 on CUDA and MPS, fp32 on CPU.

    bf16 on MPS matters for models whose residual stream exceeds fp16 range
    (e.g. Gemma-3-4b-it hits ~1e5 by the final layer, well past fp16's 65504
    max). Modern PyTorch MPS handles bf16 natively.
    """
    if device in ("cuda", "mps"):
        return torch.bfloat16
    return torch.float32


def _resolve_dtype(dtype: torch.dtype | str | None, device: str) -> torch.dtype:
    if dtype is None:
        return _pick_dtype(device)
    if isinstance(dtype, str):
        return getattr(torch, dtype)
    return dtype


def load_model(
    model_id: str,
    quantize: str | None = None,
    device: str = "auto",
    dtype: torch.dtype | str | None = None,
    *,
    compile: bool = False,
    compile_mode: str = "default",
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a HuggingFace causal LM and its tokenizer.

    Args:
        model_id: Hub ID or local path.
        quantize: None for bf16/fp16/fp32, "4bit", or "8bit".
        device: "auto" (detect), "cuda", "mps", or "cpu".
        dtype: torch.dtype or string ("bfloat16"/"float16"/"float32"). None
            picks the device default — bf16 on CUDA/MPS, fp32 on CPU.
        compile: When True, wrap ``model`` with ``torch.compile`` after
            load on CUDA — fuses the per-layer kernels and amortizes
            launch overhead, typically 1.2–3× decode tok/s on small
            models when the compile succeeds.  Off by default because
            (a) torch 2.12's inductor has known codegen bugs on newer
            architectures (Gemma-4, Qwen3.5) that this loader's probe
            catches but still costs ~25–100s upfront, and (b) the
            speedup rarely amortizes on interactive workloads.  Pass
            ``True`` for sustained workloads where the per-token win
            pays back the compile cost.  Auto-skipped on MPS/CPU.
        compile_mode: torch.compile mode string passed through.
            ``"default"`` (default) does inductor kernel fusion without
            graph capture — composes cleanly with HF's growing
            ``DynamicCache``.  ``"reduce-overhead"`` adds internal
            CUDA-graph capture but expects fixed shapes (paired with
            :class:`transformers.StaticCache` in Phase B's
            ``cuda_graphs`` path; using it here without static cache
            would shape-recompile per decode step).
            ``"max-autotune"`` runs Triton autotune (long first-call
            latency, marginal gains for decode-shape workloads).

    Returns:
        (model, tokenizer) tuple.  Compiled models still expose the
        underlying ``transformers`` attribute graph through
        ``OptimizedModule.__getattr__``, so ``get_layers`` and
        ``get_model_info`` continue to work.
    """
    device = detect_device(device)
    if device == "mps":
        patch_torch_for_mps()
    resolved_dtype = _resolve_dtype(dtype, device)
    log.info("Device: %s", device)

    # HF-distributed Mistral checkpoints ship a buggy pre-tokenizer regex
    # that mis-splits ~1% of tokens (e.g. ``"'The'"`` → ``["'", "T", "he", "'"]``
    # instead of ``["'", "The", "'"]``).  ``fix_mistral_regex=True`` swaps in the
    # correct regex from ``mistral_common``.  Substring-match on the model_id
    # so it fires for any mistralai/* repo (Mistral-Small, Ministral, etc.) and
    # third-party finetunes whose name carries the family.
    # https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/discussions/84
    tokenizer_kwargs: dict[str, Any] = {"trust_remote_code": True}
    if "mistral" in model_id.lower():
        tokenizer_kwargs["fix_mistral_regex"] = True
    tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)

    # --- quantization config ---
    if quantize and device != "cuda":
        warnings.warn(
            f"bitsandbytes quantization ({quantize}) requires CUDA. "
            f"Ignoring --quantize on {device}, loading in {resolved_dtype}."
        )
        quantize = None

    # --- attention implementation ---
    attn_impl = "sdpa"  # safe default, works everywhere
    if device == "cuda":
        try:
            # Availability check only — presence of the package flips
            # transformers onto the flash-attention-2 kernel. We never
            # call into flash_attn ourselves.
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
        except ImportError:
            pass

    # --- device map ---
    # device_map="auto" requires accelerate and works best on CUDA.
    # For MPS/CPU, place the whole model on a single device.
    device_map = "auto" if device == "cuda" else {"": device}

    # --- trust_remote_code gating ---
    # Some repos (e.g. deepseek-ai/DeepSeek-V2-Lite-Chat) ship an
    # ``auto_map`` pointing to a stale ``modeling_*.py`` that breaks
    # against newer transformers.  When the architecture is already
    # supported natively, skip the custom code entirely.
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    probe_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    native_type = getattr(probe_config, "model_type", None)
    native_text_type = getattr(getattr(probe_config, "text_config", None),
                               "model_type", None)
    trust = not (
        (native_type and native_type in CONFIG_MAPPING)
        or (native_text_type and native_text_type in CONFIG_MAPPING)
    )

    # MLA architectures (DeepSeek-V2/V3) decompose attention into a
    # query/key head_dim = qk_nope_head_dim + qk_rope_head_dim (192 on
    # V2-Lite) and a separate value head_dim = v_head_dim (128).  PyTorch
    # 2.11's MPS scaled_dot_product_attention kernel returns the output
    # with the *query* head_dim instead of the value head_dim when the
    # two differ — attn_output ends up (B, T, n_heads * 192) instead of
    # (B, T, n_heads * 128), and o_proj rejects it with
    # ``linear(): input and weight.T shapes cannot be multiplied``.
    # CPU SDPA returns the correct shape; CUDA SDPA also handles this
    # via the flash backend.  Narrowest-correct fix: force eager on MPS
    # for these models.  The ``torch.histc`` MPS patch above already
    # covers the MoE-routing issue eager would otherwise hit.
    _MLA_TYPES = {"deepseek_v2", "deepseek_v3"}
    if device == "mps" and (
        native_type in _MLA_TYPES or native_text_type in _MLA_TYPES
    ):
        if attn_impl != "eager":
            log.info(
                "forcing eager attention on MPS for MLA model %r "
                "(PyTorch MPS SDPA mishandles mismatched q/v head_dim)",
                native_type or native_text_type,
            )
            attn_impl = "eager"

    # --- check for multimodal configs wrapping a text-only model ---
    # Some text-only models ship with a multimodal config whose
    # model_type isn't registered with AutoModelForCausalLM (e.g.
    # Ministral tagged as Mistral3).  If the config has a text_config
    # that IS a known causal-LM type, use that instead.
    load_kwargs: dict[str, Any] = dict(
        attn_implementation=attn_impl,
        trust_remote_code=trust,
        device_map=device_map,
    )
    if quantize == "4bit":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    elif quantize == "8bit":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        load_kwargs["dtype"] = resolved_dtype
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust)
    text_cfg = getattr(config, "text_config", None)
    # A non-None text_config means a multimodal wrapper. When it wraps a
    # text model saklas supports, load *only* the text submodel: it skips
    # the unused vision tower and — crucially — loads on-device shard-by-
    # shard so peak memory stays ~1× the model size instead of the ~2× the
    # standard CPU-stage-then-device-copy path costs on unified memory (the
    # >128 GB gemma-4-31B blowup). This fires even when the outer composite
    # model_type is itself in _LAYER_ACCESSORS (gemma3/gemma4): the full
    # multimodal model is still usable when handed to SaklasSession
    # directly, but the from_pretrained path prefers text-only.
    extract_text_model = (
        text_cfg is not None
        and getattr(text_cfg, "model_type", None) in _LAYER_ACCESSORS
    )

    model = None
    if extract_text_model:
        # Weights live under a "language_model." path segment that doesn't
        # match the text-only model's parameter names.  Load manually.
        # Propagate _name_or_path so cache paths resolve correctly.
        assert text_cfg is not None  # guaranteed by extract_text_model condition above
        if not getattr(text_cfg, "_name_or_path", ""):
            text_cfg._name_or_path = model_id
        log.info("extracting text model (%s) from multimodal checkpoint (%s)",
                 text_cfg.model_type, config.model_type)
        try:
            model = _load_text_from_multimodal(
                model_id, text_cfg, load_kwargs.get("dtype", resolved_dtype),
                device,
            )
        except _NoTextWeightsExtracted as e:
            # Unexpected weight layout — don't ship random init; fall through
            # to the standard full-model load below.
            log.warning("%s; falling back to full multimodal load", e)
            model = None

    if model is None:
        # --- standard load (with attention, dtype, and device fallbacks) ---
        def _try_load_with_fallbacks():
            try:
                return AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
            except ValueError as e:
                if "does not support an attention implementation" not in str(e):
                    raise
                log.info("attn_implementation %r unsupported, falling back to eager",
                         load_kwargs.get("attn_implementation"))
                load_kwargs["attn_implementation"] = "eager"
                return AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
            except Exception:
                if quantize is not None:
                    raise
                fallback = torch.float16 if device == "cuda" else torch.float32
                load_kwargs["dtype"] = fallback
                return AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)

        try:
            model = _try_load_with_fallbacks()
        except (RuntimeError, ValueError) as e:
            if device in ("cuda", "cpu") or "CONVERSION" not in str(e):
                raise
            log.info("weight conversion failed on %s, retrying on CPU", device)
            load_kwargs["device_map"] = {"": "cpu"}
            if "dtype" not in load_kwargs:
                load_kwargs["dtype"] = torch.float32
            model = _try_load_with_fallbacks()
            model = model.to(device)  # pyright: ignore[reportArgumentType]  # transformers stub: .to(str) overload missing

    model.requires_grad_(False)
    model.train(False)

    # --- memory report ---
    mem_gb = _get_memory_gb(device)
    if mem_gb > 0:
        log.info("Memory used: %.2f GB", mem_gb)

    # --- compile (CUDA-only auto-enable) ---
    # Wrapping last so the compile artifact closes over weights that
    # are already on-device, in their final dtype, and frozen.
    # ``OptimizedModule`` forwards attribute access via ``__getattr__``
    # so ``get_layers(model)`` / ``model.config`` keep working through
    # the wrapper; the caller never has to know whether compile fired.
    # Hooks added later via ``register_forward_hook`` on layer modules
    # (which live under ``model._orig_mod``) compose cleanly: compile
    # treats hooks as external state and recompiles only when the
    # hook's *Python-scalar* inputs change.  Pinning the angle scalars
    # to 0-dim tensors in :class:`SteeringHook._refresh_angular_cache`
    # keeps a single compiled artifact across α changes between
    # generations.
    if compile and device == "cuda":
        model = _compile_with_probe(model, tokenizer, device, mode=compile_mode)
    elif compile and device != "cuda":
        log.info(
            "compile=True but device=%s — skipping torch.compile "
            "(supported only on CUDA)",
            device,
        )

    return cast(PreTrainedModel, model), tokenizer


def _get_memory_gb(device: str) -> float:
    if device.startswith("cuda") and torch.cuda.is_available():
        return round(torch.cuda.memory_allocated() / 1024**3, 2)
    if device.startswith("mps") and torch.backends.mps.is_available():
        # `torch.mps` exists on every torch build but the underlying call
        # raises RuntimeError ("Cannot execute getCurrentAllocatedMemory()
        # without MPS backend") when the backend isn't actually present
        # (e.g. Linux CI with `device='mps'` requested by tests).  Gating
        # on `is_available()` mirrors the cuda branch and keeps the older
        # `AttributeError` swallow as a belt-and-suspenders fallback for
        # very old torch builds.
        try:
            return round(torch.mps.current_allocated_memory() / 1024**3, 2)
        except (AttributeError, RuntimeError):
            return 0.0
    return 0.0


def get_layers(model: PreTrainedModel) -> nn.ModuleList:
    """Return the sequential transformer blocks for a supported architecture."""
    model_type = model.config.model_type
    accessor = _LAYER_ACCESSORS.get(model_type)
    if accessor is None:
        raise ValueError(
            f"Unsupported model_type {model_type!r}. "
            f"Supported architectures: {', '.join(_SUPPORTED_TYPES)}"
        )
    if model_type not in _TESTED_ARCHS and model_type not in _warned:
        _warned.add(model_type)
        warnings.warn(
            f"architecture {model_type!r} is wired up but untested — "
            "report issues at https://github.com/a9lim/saklas",
            UserWarning,
            stacklevel=2,
        )
    return accessor(model)


def _text_config(model: PreTrainedModel) -> Any:
    """Return the text-specific config, handling multimodal wrappers."""
    cfg = model.config
    return getattr(cfg, "text_config", cfg)


def get_model_info(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase) -> dict[str, Any]:
    """Summary dict: model_type, num_layers, hidden_dim, device, dtype, vram_used_gb, param_count."""
    layers = get_layers(model)
    first_param = next(model.parameters())
    model_id = getattr(model.config, "_name_or_path", "unknown")
    param_count = model.num_parameters()
    return {
        "model_id": model_id,
        "model_type": model.config.model_type,
        "num_layers": len(layers),
        "hidden_dim": _text_config(model).hidden_size,
        "device": str(first_param.device),
        "dtype": str(first_param.dtype),
        "vram_used_gb": _get_memory_gb(str(first_param.device)),
        "param_count": param_count,
    }
