"""Per-call sampling configuration for SaklasSession.generate.

Frozen dataclass holding the OpenAI-shaped sampling knobs that used to be
positional kwargs on ``session.generate``.  None on any field means "use the
session default" — the session's ``GenerationConfig`` still holds session-level
defaults (``max_new_tokens``, ``temperature``, ``top_p``, ``top_k``,
``system_prompt``) and ``_generate_core`` composes the two at entry without
mutating ``session.config``.

Pre-restructure this lives at the top level; cluster 8 will move it under
``saklas.core.sampling``.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    """Per-call sampling configuration.

    All fields default to ``None`` / neutral sentinels meaning "fall through
    to the session default".  ``merged_with`` composes two instances
    field-by-field, preferring the other's value where it differs from the
    default.
    """

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None  # new_tokens to generate
    seed: int | None = None
    stop: tuple[str, ...] | None = None
    logit_bias: dict[int, float] | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    logprobs: int | None = None  # 0 = chosen-only, >0 = top-k

    def __post_init__(self) -> None:
        # Accept list[str] from callers; store as tuple so the frozen
        # dataclass stays hashable.
        if self.stop is not None and not isinstance(self.stop, tuple):
            object.__setattr__(self, "stop", tuple(self.stop))

    # Default sentinels used by merged_with — matches the dataclass defaults.
    _DEFAULTS = {
        "temperature": None,
        "top_p": None,
        "top_k": None,
        "max_tokens": None,
        "seed": None,
        "stop": None,
        "logit_bias": None,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "logprobs": None,
    }

    def merged_with(self, other: "SamplingConfig | None") -> "SamplingConfig":
        """Return a new SamplingConfig where ``other``'s non-default fields win.

        Useful for server routes that hold a default SamplingConfig and accept
        per-request overrides from the request body.
        """
        if other is None:
            return self
        updates: dict[str, Any] = {}
        for f in fields(self):
            other_val = getattr(other, f.name)
            if other_val != self._DEFAULTS[f.name]:
                updates[f.name] = other_val
        if not updates:
            return self
        return replace(self, **updates)
