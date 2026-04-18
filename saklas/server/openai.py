"""OpenAI-shaped request models and helpers.

These live in ``saklas.server.app`` alongside ``create_app`` today; this
module re-exports them under a dedicated name so callers can
``from saklas.server.openai import ChatCompletionRequest`` without
reaching into ``app``.
"""

from saklas.server.app import (
    ChatCompletionRequest,
    ChatMessage,
    CompletionRequest,
    StreamOptions,
    UnsupportedContentError,
    _SamplingBase,
    _check_openai_model_strict,
    _openai_known_model_names,
    _render_logprobs_chat,
    _render_logprobs_completions,
    _sampling_kwargs,
    _stream_generation,
    _strict_model_enabled,
)

__all__ = [
    "ChatCompletionRequest",
    "ChatMessage",
    "CompletionRequest",
    "StreamOptions",
    "UnsupportedContentError",
    "_SamplingBase",
    "_sampling_kwargs",
    "_stream_generation",
    "_render_logprobs_chat",
    "_render_logprobs_completions",
    "_check_openai_model_strict",
    "_openai_known_model_names",
    "_strict_model_enabled",
]
