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
    # Opt-in: when True, GenerationResult.hidden_states is populated with
    # per-generated-token, per-layer residual-stream captures (all model
    # layers, CPU, detached). False keeps the fast path bit-identical to
    # today — HiddenCapture stays on probe-layer union.
    return_hidden: bool = False
    # Loom-path knob for top-K alternatives capture (saklas-native; the
    # OpenAI route uses ``logprobs`` instead and the two compose under
    # ``max`` inside ``_generate_core``).  ``0`` means logprob-only —
    # chosen-token logprob is captured whenever the engine already runs
    # log_softmax (any ``on_token`` consumer or an explicit ``logprobs``
    # request), so K=0 is the minimal-additive-cost default for the loom
    # path. ``K > 0`` additionally captures the top-K alternatives with
    # decoded text per :class:`TokenAlt`.  Clamped to ``[0, 256]`` in
    # ``__post_init__`` — beyond 256 is data nobody can act on and bytes
    # nobody wants on the wire.
    return_top_k: int = 0

    def __post_init__(self) -> None:
        # Accept list[str] from callers; store as tuple so the frozen
        # dataclass stays hashable. Type annotation narrows to tuple so
        # the isinstance check looks redundant to static analysis, but
        # it's a real runtime guard against lists slipping through.
        if self.stop is not None and not isinstance(self.stop, tuple):  # pyright: ignore[reportUnnecessaryIsInstance]
            object.__setattr__(self, "stop", tuple(self.stop))
        # Clamp return_top_k into the supported range. We do this rather
        # than raise because the field is exposed through both Python
        # (kwarg) and YAML/CLI surfaces; clamping silently keeps the
        # downstream engine math safe (slicing with a too-large K would
        # ValueError on torch.topk) and matches the rest of the
        # SamplingConfig discipline of preferring graceful coercion to
        # raises at construction.
        if self.return_top_k < 0:
            object.__setattr__(self, "return_top_k", 0)
        elif self.return_top_k > 256:
            object.__setattr__(self, "return_top_k", 256)

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
        "return_hidden": False,
        "return_top_k": 0,
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
