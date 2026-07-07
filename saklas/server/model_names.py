"""Shared model-name and Ollama-alias helpers for server protocols."""

from __future__ import annotations

from typing import Any


# Manual overrides for HF ids whose canonical Ollama tags need to match
# Ollama's actual catalogue (e.g. Gemma-2-2b is ~2.6B params but Ollama
# advertises it as `gemma2:2b`), plus cases where we want to advertise the
# `:latest` tag or where model_type lacks the version number (Llama).
# If an HF id appears here, inference is skipped; overrides are authoritative.
_HF_TO_OLLAMA_ALIASES: dict[str, list[str]] = {
    # Llama 3.x: model_type is just "llama", no version suffix to infer from.
    "meta-llama/Llama-3.2-1B-Instruct": ["llama3.2:1b", "llama3.2:1b-instruct"],
    "meta-llama/Llama-3.2-3B-Instruct": ["llama3.2", "llama3.2:latest", "llama3.2:3b"],
    "meta-llama/Meta-Llama-3.1-8B-Instruct": ["llama3.1", "llama3.1:latest", "llama3.1:8b"],
    "meta-llama/Llama-3.3-70B-Instruct": ["llama3.3", "llama3.3:latest", "llama3.3:70b"],
    # Qwen: override to match Ollama's rounded size tags.
    "Qwen/Qwen2.5-0.5B-Instruct": ["qwen2.5:0.5b"],
    "Qwen/Qwen2.5-1.5B-Instruct": ["qwen2.5:1.5b"],
    "Qwen/Qwen2.5-3B-Instruct": ["qwen2.5:3b"],
    "Qwen/Qwen2.5-7B-Instruct": ["qwen2.5", "qwen2.5:latest", "qwen2.5:7b"],
    "Qwen/Qwen3-4B-Instruct": ["qwen3:4b"],
    "Qwen/Qwen3-8B": ["qwen3", "qwen3:latest", "qwen3:8b"],
    # Gemma: Ollama advertises rounded sizes.
    "google/gemma-2-2b-it": ["gemma2:2b"],
    "google/gemma-2-9b-it": ["gemma2", "gemma2:latest", "gemma2:9b"],
    "google/gemma-3-4b-it": ["gemma3", "gemma3:latest", "gemma3:4b"],
    # Mistral.
    "mistralai/Mistral-7B-Instruct-v0.3": ["mistral", "mistral:latest", "mistral:7b"],
    "mistralai/Ministral-8B-Instruct-2410": ["ministral:8b"],
    # Phi.
    "microsoft/Phi-3.5-mini-instruct": ["phi3.5", "phi3.5:latest"],
}


def _size_tag(params: int) -> str:
    """Render parameter count as an Ollama-style size tag: 3b, 1.5b, 27b."""
    if params <= 0:
        return ""
    if params >= 1_000_000_000:
        b = params / 1_000_000_000
        if b >= 10:
            return f"{round(b)}b"
        return f"{b:.1f}".rstrip("0").rstrip(".") + "b"
    if params >= 1_000_000:
        return f"{round(params / 1_000_000)}m"
    return ""


def _normalise_family(model_type: str) -> str:
    """Map an HF model_type to an Ollama-ish family name."""
    mt = (model_type or "").lower()
    for suffix in ("_text", "_moe", "forcausallm"):
        mt = mt.removesuffix(suffix)
    return mt


def _infer_aliases(model_info: dict[str, Any]) -> list[str]:
    family = _normalise_family(str(model_info.get("model_type", "")))
    size = _size_tag(int(model_info.get("param_count", 0) or 0))
    if not family or not size:
        return []
    return [f"{family}:{size}"]


def aliases_for(model_id: str, model_info: dict[str, Any]) -> list[str]:
    """Return Ollama-style aliases for a loaded model id and model_info."""
    overrides = _HF_TO_OLLAMA_ALIASES.get(model_id)
    if overrides:
        return list(overrides)
    return _infer_aliases(model_info)


def aliases_for_session(session: Any) -> list[str]:
    return aliases_for(str(session.model_id), dict(session.model_info))


def known_model_names(session: Any) -> set[str]:
    names = {str(session.model_id), *aliases_for_session(session)}
    return {n.lower() for n in names}
