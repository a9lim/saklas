"""Shared request helpers for OpenAI, Ollama, and native streaming routes."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

from saklas.core.errors import SaklasError
from saklas.core.results import TokenEvent
from saklas.core.sampling import SamplingConfig
from saklas.core.steering import Steering


class UnsupportedContentError(ValueError, SaklasError):
    """Non-text content parts submitted to a text-only endpoint."""

    def user_message(self) -> tuple[int, str]:
        return (400, str(self) or self.__class__.__name__)


def flatten_content(content: Any) -> str:
    """Concatenate text content parts for text-only endpoints.

    Accepts a string (passed through), a list of content parts (each a
    ``{"type": "text", "text": ...}`` dict or a bare string — non-text
    parts raise :class:`UnsupportedContentError`), ``None`` (→ ``""``,
    the Ollama convention), or any other scalar (stringified).
    """
    if isinstance(content, list):
        pieces: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                pieces.append(str(part.get("text", "")))
            elif isinstance(part, str):
                pieces.append(part)
            else:
                raise UnsupportedContentError(
                    "non-text content parts are not supported by this model"
                )
        return "".join(pieces)
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return str(content)


def probe_token_readings(event: TokenEvent) -> dict[str, Any] | None:
    """Serialize a token event's live probe readings, or ``None`` if absent."""
    readings = event.probe_readings
    if not readings:
        return None
    return {name: reading.to_dict() for name, reading in readings.items()}


def parse_request_steering(
    expr: str | None,
) -> tuple[Steering | None, bool]:
    """Parse a per-request steering expression string.

    Returns ``(req_steering, explicit_clear)``: ``None`` expression inherits
    the server default; an explicit empty / whitespace string clears it; a
    non-empty string parses through the shared grammar.
    """
    from saklas.core.steering_expr import parse_expr

    if expr is None:
        return None, False
    if not expr.strip():
        return None, True
    return parse_expr(expr), False


def merge_steering(
    req_steering: Steering | None,
    default_steering: Steering | None,
    explicit_clear: bool,
    thinking: bool | None,
) -> Steering | None:
    """Compose parsed request steering over the server default."""
    merged_alphas: dict[str, Any] = {}
    if default_steering is not None and not explicit_clear:
        merged_alphas.update(default_steering.alphas)
    if req_steering is not None:
        merged_alphas.update(req_steering.alphas)
    if not merged_alphas and thinking is None:
        # An explicit clear must reach the session as an *empty*
        # Steering, not None — None means "unset" there, and the cast
        # roster fills unset steering with the gen label's standing
        # recipe.  ``steering: ""`` is the escape hatch that must win.
        return Steering(alphas={}) if explicit_clear else None
    return Steering(alphas=merged_alphas, thinking=thinking)


def normalize_stop(
    stop: str | Sequence[str] | None,
) -> tuple[str, ...] | None:
    """Lower a protocol's ``stop`` field to the engine's tuple shape.

    Every protocol accepts either a bare string or a sequence of strings
    (OpenAI ``stop``, Ollama ``options.stop``, the native WS ``sampling.stop``
    list); the engine takes a tuple.  ``None`` and an empty sequence both mean
    "no stop sequences".
    """
    if stop is None:
        return None
    if isinstance(stop, str):
        return (stop,)
    normalized = tuple(str(s) for s in stop)
    return normalized or None


def logprob_count(
    logprobs: bool | int | None, top_logprobs: int | None = None,
) -> int | None:
    """Collapse the two OpenAI logprobs shapes to the engine's int count.

    ``/v1/chat/completions`` sends a bool plus a separate ``top_logprobs``
    count; ``/v1/completions`` sends the count directly as an int.  The engine
    takes one int (``0`` = chosen token only, ``None`` = disabled).
    """
    if isinstance(logprobs, bool):
        return (top_logprobs or 0) if logprobs else None
    if isinstance(logprobs, int):
        return logprobs
    return None


def build_sampling_config(
    *,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    max_tokens: int | None = None,
    seed: int | None = None,
    stop: str | Sequence[str] | None = None,
    logit_bias: dict[int, float] | None = None,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    logprobs: bool | int | None = None,
    top_logprobs: int | None = None,
    return_top_k: int = 0,
    return_probe_readings: bool = True,
    persist_per_layer_scores: bool = False,
    persist_subspace_coords: bool = False,
    user_role: str | None = None,
    assistant_role: str | None = None,
) -> SamplingConfig:
    """Build a :class:`SamplingConfig` from raw per-protocol request fields.

    The single construction site for all three protocols (OpenAI, Ollama, the
    native WebSocket), so the normalizations they share — ``stop`` in either
    string or sequence form, the two OpenAI ``logprobs`` shapes, empty role
    labels meaning "unset" — happen once instead of drifting per surface.
    """
    return SamplingConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        seed=seed,
        stop=normalize_stop(stop),
        logit_bias=logit_bias,
        presence_penalty=presence_penalty or 0.0,
        frequency_penalty=frequency_penalty or 0.0,
        logprobs=logprob_count(logprobs, top_logprobs),
        return_top_k=return_top_k,
        return_probe_readings=bool(return_probe_readings),
        persist_per_layer_scores=bool(persist_per_layer_scores),
        persist_subspace_coords=bool(persist_subspace_coords),
        user_role=user_role or None,
        assistant_role=assistant_role or None,
    )


def strict_model_enabled() -> bool:
    return os.environ.get("SAKLAS_STRICT_MODEL", "").lower() in (
        "1", "true", "yes", "on",
    )
