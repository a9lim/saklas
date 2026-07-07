"""Shared request helpers for OpenAI, Ollama, and native streaming routes."""

from __future__ import annotations

import os
from contextlib import suppress
from typing import Any, cast

from saklas.core.errors import SaklasError
from saklas.core.sampling import SamplingConfig
from saklas.core.steering import Steering
from saklas.server.streaming import probe_reading_aggregate as _stream_aggregate


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


def probe_reading_dict(
    session: Any,
    readings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Per-attached-probe wire dict from a readings mapping or session state."""
    monitor_names = set(session.monitor.probe_names)
    if readings is None:
        result = getattr(session, "last_result", None)
        readings = getattr(result, "readings", None) if result is not None else None
    if readings is None:
        readings = cast(dict[str, Any], session.build_readings())
    out: dict[str, Any] = {}
    for name, reading in readings.items():
        if name not in monitor_names:
            continue
        out[name] = reading.to_dict()
    return out


def probe_reading_aggregate(
    session: Any,
    result: Any | None = None,
) -> dict[str, Any]:
    """Per-attached-probe aggregate dict from ``result`` or ``session.last_result``."""
    if result is None:
        result = getattr(session, "last_result", None)
    return _stream_aggregate(session, result)


def probe_token_readings(event: Any) -> dict[str, Any] | None:
    """Serialize a token event's live probe readings, or ``None`` if absent."""
    readings = getattr(event, "probe_readings", None)
    if not readings:
        return None
    out: dict[str, Any] = {}
    for name, reading in readings.items():
        with suppress(Exception):
            out[name] = reading.to_dict()
    return out or None


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
        return None
    return Steering(alphas=merged_alphas, thinking=thinking)


def build_sampling_config(
    *,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None = None,
    max_tokens: int | None,
    seed: int | None,
    stop: tuple[str, ...] | None,
    logit_bias: dict[int, float] | None = None,
    presence_penalty: float,
    frequency_penalty: float,
    logprobs: int | None = None,
) -> SamplingConfig:
    """Build a :class:`SamplingConfig` from already-normalized fields."""
    return SamplingConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        seed=seed,
        stop=stop,
        logit_bias=logit_bias,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        logprobs=logprobs,
    )


def strict_model_enabled() -> bool:
    return os.environ.get("SAKLAS_STRICT_MODEL", "").lower() in (
        "1", "true", "yes", "on",
    )
