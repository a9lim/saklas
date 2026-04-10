"""Model loading utilities for activation steering."""

import logging
import warnings

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

log = logging.getLogger(__name__)

torch._dynamo.config.skip_nnmodule_hook_guards = False

_MODEL_LAYERS = lambda m: m.model.layers  # noqa: E731
_TRANSFORMER_H = lambda m: m.transformer.h  # noqa: E731
_VLM_LANGUAGE_LAYERS = lambda m: m.model.language_model.layers  # noqa: E731

_LAYER_ACCESSORS = {
    # Llama family
    "llama": _MODEL_LAYERS,
    "llama4": _MODEL_LAYERS,
    "llama4_text": _MODEL_LAYERS,
    # Mistral / Mixtral
    "mistral": _MODEL_LAYERS,
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
    "olmoe": _MODEL_LAYERS,
    # GLM (ChatGLM)
    "glm": _MODEL_LAYERS,
    "glm4": _MODEL_LAYERS,
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
    # GPT-NeoX / Pythia
    "gpt_neox": lambda m: m.gpt_neox.layers,
    # MPT / DBRX
    "mpt": lambda m: m.transformer.blocks,
    "dbrx": lambda m: m.transformer.blocks,
    # OPT
    "opt": lambda m: m.model.decoder.layers,
}

_SUPPORTED_TYPES = sorted(_LAYER_ACCESSORS)


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
    """Best default dtype for a device. bf16 on CUDA, fp16 on MPS, fp32 on CPU."""
    if device == "cuda":
        return torch.bfloat16
    if device == "mps":
        return torch.float16  # MPS has limited bf16 support
    return torch.float32


def load_model(model_id: str, quantize=None, device="auto", no_compile=False):
    """Load a HuggingFace causal LM and its tokenizer.

    Args:
        model_id: Hub ID or local path.
        quantize: None for bf16/fp16/fp32, "4bit", or "8bit".
        device: "auto" (detect), "cuda", "mps", or "cpu".
        no_compile: Skip torch.compile if True.

    Returns:
        (model, tokenizer) tuple.
    """
    device = detect_device(device)
    print(f"  Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # --- quantization config ---
    quant_kwargs = {}
    if quantize and device != "cuda":
        warnings.warn(
            f"bitsandbytes quantization ({quantize}) requires CUDA. "
            f"Ignoring --quantize on {device}, loading in {_pick_dtype(device)}."
        )
        quantize = None

    if quantize == "4bit":
        quant_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    elif quantize == "8bit":
        quant_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        quant_kwargs["dtype"] = _pick_dtype(device)

    # --- attention implementation ---
    attn_impl = "sdpa"  # safe default, works everywhere
    if device == "cuda":
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
        except ImportError:
            pass

    # --- device map ---
    # device_map="auto" requires accelerate and works best on CUDA.
    # For MPS/CPU, place the whole model on a single device.
    if device == "cuda":
        device_kwargs = {"device_map": "auto"}
    else:
        device_kwargs = {"device_map": {"": device}}

    # --- load model ---
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            attn_implementation=attn_impl,
            trust_remote_code=True,
            **device_kwargs,
            **quant_kwargs,
        )
    except Exception:
        # bf16/fp16 unsupported — fall back through dtypes
        if quantize is None:
            fallback = torch.float16 if device == "cuda" else torch.float32
            quant_kwargs["dtype"] = fallback
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                attn_implementation=attn_impl,
                trust_remote_code=True,
                **device_kwargs,
                **quant_kwargs,
            )
        else:
            raise

    model.requires_grad_(False)
    model.train(False)

    # --- optional torch.compile ---
    # torch.compile with MPS backend is experimental; default to skip on non-CUDA
    should_compile = not no_compile and device == "cuda"
    if should_compile:
        try:
            model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
            log.info("torch.compile succeeded")
        except Exception as e:
            warnings.warn(f"torch.compile failed, continuing uncompiled: {e}")

    # --- memory report ---
    if device == "cuda" and torch.cuda.is_available():
        vram_bytes = torch.cuda.memory_allocated()
        print(f"  VRAM used: {vram_bytes / 1024**3:.2f} GB")
    elif device == "mps":
        try:
            mps_bytes = torch.mps.current_allocated_memory()
            print(f"  MPS memory used: {mps_bytes / 1024**3:.2f} GB")
        except AttributeError:
            pass  # older torch without mps memory tracking

    return model, tokenizer


def _get_memory_gb(device: str) -> float:
    if device.startswith("cuda") and torch.cuda.is_available():
        return round(torch.cuda.memory_allocated() / 1024**3, 2)
    if device == "mps":
        try:
            return round(torch.mps.current_allocated_memory() / 1024**3, 2)
        except AttributeError:
            return 0.0
    return 0.0


def get_layers(model) -> nn.ModuleList:
    """Return the sequential transformer blocks for a supported architecture."""
    model_type = getattr(model, "config", None) and model.config.model_type
    accessor = _LAYER_ACCESSORS.get(model_type)
    if accessor is None:
        raise ValueError(
            f"Unsupported model_type {model_type!r}. "
            f"Supported architectures: {', '.join(_SUPPORTED_TYPES)}"
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
    param_count = sum(p.numel() for p in model.parameters())
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
