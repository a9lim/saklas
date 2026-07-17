"""Typed token-callback contract and consumer capability declaration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from saklas.core.results import TokenAlt


class TokenCallback(Protocol):
    def __call__(self, text: str, is_thinking: bool, token_id: int | None,
                 logprob: float | None, top_alts: list[TokenAlt] | None,
                 perplexity: float | None = None, /) -> None: ...


class StepTokenCallback(Protocol):
    """The decode loop's INTERNAL tap contract — ``generate_steered``'s
    ``on_token``.  The public :class:`TokenCallback` (what users hand to the
    session) stays six-argument; the trailing ``step_id`` is the loop-owned
    forward index (the same value that forward's ``step_callback`` /
    ``score_callback`` received), which the session tap uses to pair the
    instrument runs' step-keyed memos.  ``SaklasSession._token_tap`` absorbs
    it and invokes user callbacks with the public six arguments."""

    def __call__(self, text: str, is_thinking: bool, token_id: int | None,
                 logprob: float | None, top_alts: list[TokenAlt] | None,
                 perplexity: float | None, step_id: int, /) -> None: ...


@dataclass(frozen=True)
class TokenConsumerOptions:
    live_scores: bool = False
    per_layer_scores: bool = False
    lens_readout: bool = False
    sae_readout: bool = False
    perplexity: bool = False


@dataclass(frozen=True)
class TokenConsumer:
    callback: TokenCallback
    options: TokenConsumerOptions = TokenConsumerOptions()

    def __call__(self, text: str, is_thinking: bool, token_id: int | None,
                 logprob: float | None, top_alts: list[TokenAlt] | None,
                 perplexity: float | None = None) -> None:
        self.callback(text, is_thinking, token_id, logprob, top_alts, perplexity)


def consumer_options(callback: object | None) -> TokenConsumerOptions:
    return callback.options if isinstance(callback, TokenConsumer) else TokenConsumerOptions()
