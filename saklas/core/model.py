"""Model loading utilities for activation steering."""

import logging
import warnings

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

log = logging.getLogger(__name__)

# MPS lacks the histogram kernel for integer tensors, which breaks
# torch.histc (used by MoE routing in transformers' grouped_mm path).
# The fix matches what transformers does for CPU: cast to float first.
_orig_histc = torch.histc


def _histc_mps_safe(input, bins=100, min=0, max=0, *, out=None):
    if input.device.type == "mps" and not input.is_floating_point():
        input = input.float()
    return _orig_histc(input, bins=bins, min=min, max=max, out=out)


torch.histc = _histc_mps_safe

def _MODEL_LAYERS(m): return m.model.layers
def _TRANSFORMER_H(m): return m.transformer.h
def _VLM_LANGUAGE_LAYERS(m): return m.model.language_model.layers

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
}

_SUPPORTED_TYPES = sorted(_LAYER_ACCESSORS)

# Architectures with end-to-end testing (smoke + session). Everything else in
# _LAYER_ACCESSORS is wired up optimistically — it may work, but has not been
# exercised. See CLAUDE.md "Architecture" section.
_TESTED_ARCHS: frozenset[str] = frozenset({
    "qwen2", "qwen3", "gemma2", "gemma3", "mistral3", "gpt_oss", "llama", "glm",
})
_warned: set[str] = set()


_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2,
               torch.float8_e4m3fnuz, torch.float8_e5m2fnuz)


def _load_text_from_multimodal(
    model_id: str,
    text_config,
    dtype: torch.dtype,
    device: str,
):
    """Load just the text model from a multimodal checkpoint.

    Multimodal checkpoints store text-model weights under a
    ``language_model.`` prefix.  This function creates a text-only
    model directly on the target device, then loads each safetensors
    shard, strips the prefix, dequantizes FP8 weights, and copies
    them in shard-by-shard.  Vision-tower weights are skipped.

    Creating on-device and loading per-shard keeps peak CPU RSS low
    (~6 GB vs ~30 GB), which matters on Apple Silicon where CPU and
    MPS share unified memory.
    """
    import gc
    import json
    from safetensors.torch import load_file
    from transformers.utils import (
        cached_file,
        SAFE_WEIGHTS_INDEX_NAME,
        SAFE_WEIGHTS_NAME,
    )

    # Create directly on target device — avoids a CPU copy that would
    # spike RSS and eat into MPS's unified memory budget.
    with torch.device(device):
        model = AutoModelForCausalLM.from_config(text_config, dtype=dtype)

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

    prefix = "language_model."

    for sf in shard_paths:
        shard = load_file(sf, device="cpu")
        mapped: dict[str, torch.Tensor] = {}

        for k, v in shard.items():
            if not k.startswith(prefix):
                continue
            key = k[len(prefix):]
            if key.endswith(".weight_scale_inv") or key.endswith(".activation_scale"):
                continue

            # Dequantize FP8: real_weight = weight.to(dtype) * scale
            if v.dtype in _FP8_DTYPES:
                scale = shard.get(k + "_scale_inv")
                if scale is not None:
                    v = v.to(dtype) * scale.to(dtype)
                else:
                    v = v.to(dtype)
            mapped[key] = v.to(device=device, dtype=dtype)

        if mapped:
            model.load_state_dict(mapped, strict=False)

        del shard, mapped
        gc.collect()

    if device == "mps":
        torch.mps.empty_cache()

    return model


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


def load_model(model_id: str, quantize=None, device="auto"):
    """Load a HuggingFace causal LM and its tokenizer.

    Args:
        model_id: Hub ID or local path.
        quantize: None for bf16/fp16/fp32, "4bit", or "8bit".
        device: "auto" (detect), "cuda", "mps", or "cpu".

    Returns:
        (model, tokenizer) tuple.
    """
    device = detect_device(device)
    log.info("Device: %s", device)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # --- quantization config ---
    if quantize and device != "cuda":
        warnings.warn(
            f"bitsandbytes quantization ({quantize}) requires CUDA. "
            f"Ignoring --quantize on {device}, loading in {_pick_dtype(device)}."
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

    # --- check for multimodal configs wrapping a text-only model ---
    # Some text-only models ship with a multimodal config whose
    # model_type isn't registered with AutoModelForCausalLM (e.g.
    # Ministral tagged as Mistral3).  If the config has a text_config
    # that IS a known causal-LM type, use that instead.
    load_kwargs: dict = dict(
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
        load_kwargs["dtype"] = _pick_dtype(device)
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust)
    text_cfg = getattr(config, "text_config", None)
    extract_text_model = (
        text_cfg is not None
        and getattr(text_cfg, "model_type", None) in _LAYER_ACCESSORS
        and getattr(config, "model_type", None) not in _LAYER_ACCESSORS
    )

    if extract_text_model:
        # Multimodal checkpoint wrapping a supported text-only model
        # (e.g. Ministral tagged as Mistral3).  Weights are stored
        # with a "language_model." prefix that doesn't match the
        # text-only model's parameter names.  Load manually.
        # Propagate _name_or_path so cache paths resolve correctly.
        if not getattr(text_cfg, "_name_or_path", ""):
            text_cfg._name_or_path = model_id
        log.info("extracting text model (%s) from multimodal checkpoint (%s)",
                 text_cfg.model_type, config.model_type)
        model = _load_text_from_multimodal(
            model_id, text_cfg, load_kwargs.get("dtype", _pick_dtype(device)),
            device,
        )
    else:
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
            model = model.to(device)

    model.requires_grad_(False)
    model.train(False)

    # --- memory report ---
    mem_gb = _get_memory_gb(device)
    if mem_gb > 0:
        log.info("Memory used: %.2f GB", mem_gb)

    return model, tokenizer


def _get_memory_gb(device: str) -> float:
    if device.startswith("cuda") and torch.cuda.is_available():
        return round(torch.cuda.memory_allocated() / 1024**3, 2)
    if device.startswith("mps"):
        try:
            return round(torch.mps.current_allocated_memory() / 1024**3, 2)
        except AttributeError:
            return 0.0
    return 0.0


def get_layers(model) -> nn.ModuleList:
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


def _text_config(model):
    """Return the text-specific config, handling multimodal wrappers."""
    cfg = model.config
    return getattr(cfg, "text_config", cfg)


def get_model_info(model, tokenizer) -> dict:
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
