"""Custom assistant-role rendering for HF chat templates.

Most HF chat templates are Jinja with hardcoded role branches
(``if message.role == 'assistant'``).  Handing ``{"role": "pirate"}``
to ``apply_chat_template`` silently drops the message on
Qwen/GLM/GPT-OSS, raises on Mistral, and only works on Gemma (else
clause) and Llama (direct interpolation).  We need a strategy that
works uniformly across the architectures saklas supports.

The universal trick is **render-then-splice**: render the template with
the standard ``assistant`` role, then string-replace the per-family
role header at the rendered-string level, then tokenize.  This keeps
token-index stability for the saklas trigger system
(``thinking``/``response``/``prompt``/``generated`` boundary detection
works on the post-substitution string the tokenizer ingests).

The per-family role-header registry below is the bridge from
``model.config.model_type`` to the literal ``<header_before><label><header_after>``
byte sequence each family emits around the assistant role.  Mistral-3
is unsupported (positional ``[INST]/[/INST]`` markers, no role label
in the rendered string).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from saklas.core.errors import SaklasError


# ---------------------------------------------------------------------------
# Per-family role-header registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoleHeader:
    """The literal bytes a family's chat template emits around the assistant role.

    The splice key for a family is ``f"{before}{label}{after}"``; replacing
    ``label`` with a custom role string in that key (keeping ``before`` and
    ``after`` fixed) produces the substituted header.

    ``label`` is the *standard* role string the family uses for the assistant
    turn — almost always ``"assistant"``, but Gemma uses ``"model"``.  This is
    the string :func:`apply_with_role` looks for in the rendered template;
    the user-facing custom role replaces it.
    """

    before: str
    after: str
    label: str


@dataclass(frozen=True)
class SeatHeaders:
    """A family's headers for both seats (``None`` = substitution unsupported).

    One entry per ``model.config.model_type``.  The two seats are stored
    together because they are the *same* fact about a family — every live
    family except gpt-oss shares ``before``/``after`` across seats and differs
    only in ``label``, so splitting them into two parallel tables meant one
    family edit in two places with nothing enforcing agreement.
    """

    assistant: RoleHeader | None
    user: RoleHeader | None


def _seat_headers(
    before: str,
    after: str,
    *,
    assistant_label: str = "assistant",
    user_after: str | None = None,
) -> SeatHeaders:
    """Build both seats' headers from one family's delimiter pair.

    ``assistant_label`` is the family's standard assistant-turn label
    (``model`` on gemma); the user seat is always labelled ``user``.
    ``user_after`` overrides the user seat's trailing delimiter for the one
    family whose seats diverge (gpt-oss Harmony: ``<|start|>user<|message|>``
    against ``<|start|>assistant<|channel|>``).
    """
    return SeatHeaders(
        assistant=RoleHeader(before=before, after=after, label=assistant_label),
        user=RoleHeader(
            before=before,
            after=after if user_after is None else user_after,
            label="user",
        ),
    )


# Families with positional / label-free templates: there is no role label in
# the rendered string to swap, so the render-then-splice strategy doesn't apply
# to either seat (Mistral's ``[INST]``/``[/INST]``).
_NO_ROLE_LABEL = SeatHeaders(assistant=None, user=None)


# Map ``model.config.model_type`` → both seats' headers.  Sub-variant model
# types (``qwen3_text``, ``gemma3_text``, ...) inherit from their parent family
# — same chat template, same byte sequence.
def _build_seat_headers() -> dict[str, SeatHeaders]:
    qwen = _seat_headers("<|im_start|>", "\n")
    gemma = _seat_headers("<start_of_turn>", "\n", assistant_label="model")
    # Gemma-4 shipped a new turn-boundary scheme — ``<|turn>...<turn|>``
    # instead of the prior ``<start_of_turn>...<end_of_turn>``.  The label
    # stays ``model`` (gemma family convention), only the surrounding
    # delimiter changed.  Verified empirically against the
    # ``google/gemma-4-31b-it`` tokenizer's rendered chat template (2026-05).
    # The ``gemma4_unified`` variant (``gemma-4-12B-it``, 2026-06) uses the
    # byte-identical ``<|turn>model\n`` turn open, so it reuses this header;
    # its only addition is an empty ``<|channel>thought\n<channel|>`` reasoning
    # scaffold after the generation prompt, which the label swap rides over.
    gemma4 = _seat_headers("<|turn>", "\n", assistant_label="model")
    llama = _seat_headers("<|start_header_id|>", "<|end_header_id|>")
    glm = _seat_headers("<|", "|>")
    # gpt-oss is the one family whose seats diverge: the Harmony format renders
    # user turns as ``<|start|>user<|message|>`` (verified against the real
    # ``openai/gpt-oss-20b`` template, 2026-05).
    gpt_oss = _seat_headers("<|start|>", "<|channel|>", user_after="<|message|>")
    # Talkie-1930 renders turns as ``<|role|>content<|end|>`` — the same
    # ``<|...|>`` role-label shape as GLM (no post-marker newline).  Verified
    # against the ``a9lim/talkie-1930-13b-it-hf-cached`` chat template
    # (model_type "talkie", 2026-06).  The opt-out was "untested", not
    # "label-free" — the label is present, so render-then-splice applies.
    talkie = _seat_headers("<|", "|>")

    return {
        # Qwen family
        "qwen2": qwen,
        "qwen3": qwen,
        "qwen3_text": qwen,
        "qwen3_moe": qwen,
        "qwen3_5": qwen,
        "qwen3_5_text": qwen,
        "qwen3_5_moe": qwen,
        # Gemma family — note the assistant label is "model", not "assistant".
        # Gemma-4 uses a different delimiter than 2/3 (see ``gemma4`` above).
        "gemma2": gemma,
        "gemma3": gemma,
        "gemma3_text": gemma,
        "gemma4": gemma4,
        "gemma4_text": gemma4,
        "gemma4_unified": gemma4,
        "gemma4_unified_text": gemma4,
        # Llama family
        "llama": llama,
        # GLM (ChatGLM-4)
        "glm": glm,
        # GPT-OSS
        "gpt_oss": gpt_oss,
        # Talkie — <|role|> markers, GLM-shaped.
        "talkie": talkie,
        # Mistral: positional markers, no label to swap.  ``mistral`` is the
        # text sub-model a Mistral-3 checkpoint actually loads as (the
        # composite's ``text_config.model_type``), so it needs the same
        # explicit opt-out as the composite type — the lookup is exact.
        "mistral": _NO_ROLE_LABEL,
        "mistral3": _NO_ROLE_LABEL,
        "ministral3": _NO_ROLE_LABEL,
    }


SEAT_HEADERS: dict[str, SeatHeaders] = _build_seat_headers()

# Per-seat views over the one registry, kept because they are the lookup shape
# every call site already speaks (``_lookup_header`` takes a side's table).
ROLE_HEADERS: dict[str, RoleHeader | None] = {
    model_type: seats.assistant for model_type, seats in SEAT_HEADERS.items()
}
USER_ROLE_HEADERS: dict[str, RoleHeader | None] = {
    model_type: seats.user for model_type, seats in SEAT_HEADERS.items()
}


# ---------------------------------------------------------------------------
# Role-slug validation
# ---------------------------------------------------------------------------


# Same alphabet as ``io.paths._UNSAFE_VARIANT_CHARS`` — the slug discipline
# shared with the SAE release / transfer-source / role filename variants.  A
# custom role label is a user-provided string we splice into a rendered
# template and into a ``:role-<slug>`` selector suffix, so keeping it
# ``[a-z0-9._-]`` rules out whitespace, uppercase, and punctuation that would
# corrupt either surface.  At render time the underscore de-slugs to a space
# (:func:`render_role_label`), so ``someone_happy`` displays as
# ``someone happy``.
_ROLE_SLUG_RE = re.compile(r"^[a-z0-9._-]+$")


class RoleSubstitutionUnsupportedError(SaklasError, ValueError):
    """Raised when role substitution is requested for a family that doesn't
    support it.

    Two sub-cases share this error: the family is in :data:`ROLE_HEADERS`
    with value ``None`` (positional, label-free templates like Mistral-3),
    or the family is absent from the registry entirely (unknown or
    unsupported architecture).
    """

    def user_message(self) -> tuple[int, str]:
        return (400, str(self) or self.__class__.__name__)


class RoleTemplateDriftError(SaklasError, RuntimeError):
    """Raised when the rendered template does not contain the expected
    per-family role-header byte sequence.

    Hitting this means the registry's ``RoleHeader`` is no longer in sync
    with what the tokenizer's chat template actually emits — likely because
    HF/the model author shipped a template revision.  Silently no-oping
    would corrupt extraction (the baseline wouldn't match steering); we
    surface it as a bug signal instead.
    """

    def user_message(self) -> tuple[int, str]:
        return (500, str(self) or self.__class__.__name__)


class InvalidRoleError(SaklasError, ValueError):
    """Raised when the requested custom role label fails slug validation."""

    def user_message(self) -> tuple[int, str]:
        return (400, str(self) or self.__class__.__name__)


def validate_role(role: str) -> None:
    """Reject a cast/role label that isn't a legal slug.

    The module's public slug gate: every surface that accepts a user-authored
    role label (extraction ``--role``, per-node manifold roles, loom cast
    labels, scene turn labels) validates through this one function.
    """
    if not role:
        raise InvalidRoleError("role must be a non-empty string")
    if not _ROLE_SLUG_RE.match(role):
        raise InvalidRoleError(
            f"role {role!r} must match [a-z0-9._-]+ "
            f"(lowercase alphanumeric, dot/underscore/dash only)"
        )


# ---------------------------------------------------------------------------
# Splice helpers
# ---------------------------------------------------------------------------


def render_role_label(case_from: str, role: str) -> str:
    """Turn a role *slug* into the display label a render places in a header.

    De-slugs ``_`` → space (hyphens are kept verbatim), then mirrors the case
    of ``case_from`` so the label matches the casing its context emits.
    ``case_from`` is whatever lowercase-or-not reference string the caller
    has: the family's standard header label on the splice path
    (``assistant``/``model``/``user``), the seat name on the raw-marker path.
    Every live reference is lowercase, so today this is de-slug + lowercase;
    the case-mirror is forward-proofing for a family whose role header renders
    in a different case.
    """
    label = role.replace("_", " ")
    if case_from.isupper():
        return label.upper()
    if case_from.istitle():
        return label.title()
    return label


def _lookup_header(
    registry: dict[str, RoleHeader | None], model_type: str, *, side: str
) -> RoleHeader:
    """Resolve a family's :class:`RoleHeader` or raise an unsupported error."""
    if model_type not in registry:
        raise RoleSubstitutionUnsupportedError(
            f"{side}-role substitution is not supported for model_type "
            f"{model_type!r}: family is not in the role-header registry"
        )
    header = registry[model_type]
    if header is None:
        raise RoleSubstitutionUnsupportedError(
            f"{side}-role substitution is not supported for model_type "
            f"{model_type!r}: family uses positional or otherwise label-free "
            f"chat-template markers"
        )
    return header


def _splice_header(
    rendered: str, header: RoleHeader, role: str, *, model_type: str, strict: bool
) -> str:
    """Replace ``header``'s standard label with ``role`` in ``rendered``.

    ``strict=True`` raises :class:`RoleTemplateDriftError` when the header
    is absent (used for the assistant side — the generation prompt is
    always emitted, so a miss signals registry drift).  ``strict=False``
    treats a miss as a no-op (used for the user side — user turns are
    data-dependent, so their absence is legitimate, e.g. a system-only
    history or an assistant-prefill render).
    """
    pattern = f"{header.before}{header.label}{header.after}"
    if rendered.count(pattern) == 0:
        if strict:
            raise RoleTemplateDriftError(
                f"role header {pattern!r} not found in rendered template for "
                f"model_type {model_type!r}; the registry may have drifted "
                f"from the live chat template"
            )
        return rendered
    replacement = f"{header.before}{render_role_label(header.label, role)}{header.after}"
    return rendered.replace(pattern, replacement)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def apply_with_role(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    role: str | None,
    user_role: str | None = None,
    model_type: str,
    add_generation_prompt: bool = True,
    tokenize: bool = True,
    return_tensors: str | None = None,
    **chat_template_kwargs: Any,
) -> Any:
    """Render a chat template with custom assistant- and/or user-role labels.

    Strategy: render with the standard role labels, splice the custom
    label(s) at the string level, then optionally tokenize.  Token indices
    remain stable in the substituted prompt so saklas's
    ``thinking``/``response``/``prompt``/``generated`` boundary detection
    still works.

    Parameters
    ----------
    tokenizer:
        HF tokenizer (anything with ``apply_chat_template``).
    messages:
        Standard chat messages — ``[{"role": "user"|"system"|"assistant",
        "content": str}, ...]``.
    role:
        Custom *assistant*-role label (e.g. ``"oracle"``), or ``None`` to
        leave the family's standard assistant-role label in place.
    user_role:
        Custom *user*-role label (e.g. ``"captain"``), or ``None`` to leave
        the standard user label in place.  When both ``role`` and
        ``user_role`` are ``None`` this is a zero-overhead transparent
        pass-through to :meth:`apply_chat_template`.
    model_type:
        ``model.config.model_type`` — used to look up the family's
        :class:`RoleHeader` in :data:`ROLE_HEADERS` (assistant) and
        :data:`USER_ROLE_HEADERS` (user).
    add_generation_prompt:
        Forwarded to ``apply_chat_template``.
    tokenize:
        If True (default), tokenize the (possibly substituted) string and
        return token ids; if False return the rendered string.
    return_tensors:
        Forwarded to the tokenizer call when ``tokenize=True``.
    **chat_template_kwargs:
        Forwarded to ``apply_chat_template`` (e.g. ``enable_thinking``).

    Returns
    -------
    Either token ids / a tensor (``tokenize=True``) or the rendered string
    (``tokenize=False``).

    Raises
    ------
    InvalidRoleError:
        ``role`` / ``user_role`` does not match the role-slug alphabet.
    RoleSubstitutionUnsupportedError:
        ``model_type`` is absent from the relevant registry or maps to
        ``None`` for a requested side.
    RoleTemplateDriftError:
        The rendered template does not contain the family's expected
        assistant header — the registry has drifted from the live template.
        (A missing *user* header is tolerated, not raised.)
    """
    # Both None is the cheap default path.  Pass through verbatim — no
    # registry lookup, no string scan, no extra allocations.
    if role is None and user_role is None:
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=tokenize,
            return_tensors=return_tensors,
            **chat_template_kwargs,
        )

    # Validate slugs + resolve headers up front so an unsupported family or
    # bad slug fails before we render anything.
    assistant_header: RoleHeader | None = None
    if role is not None:
        validate_role(role)
        assistant_header = _lookup_header(ROLE_HEADERS, model_type, side="assistant")
    user_header: RoleHeader | None = None
    if user_role is not None:
        validate_role(user_role)
        user_header = _lookup_header(USER_ROLE_HEADERS, model_type, side="user")

    rendered = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=add_generation_prompt,
        tokenize=False,
        **chat_template_kwargs,
    )
    if not isinstance(rendered, str):
        # Defensive: tokenize=False should always return str for a single
        # conversation, but tokenizers occasionally return a list when
        # handed a list-of-conversations.  Don't try to be clever here.
        raise RoleTemplateDriftError(
            f"apply_chat_template returned {type(rendered).__name__} with "
            f"tokenize=False; role substitution expects a string"
        )

    spliced = rendered
    if assistant_header is not None and role is not None:
        spliced = _splice_header(
            spliced, assistant_header, role, model_type=model_type, strict=True
        )
    if user_header is not None and user_role is not None:
        spliced = _splice_header(
            spliced, user_header, user_role, model_type=model_type, strict=False
        )

    if not tokenize:
        return spliced

    # Tokenize the substituted string.  ``add_special_tokens=False`` because
    # apply_chat_template already emitted any BOS/EOS the template wants;
    # re-adding them would double-prefix.
    return tokenizer(
        spliced,
        return_tensors=return_tensors,
        add_special_tokens=False,
    )["input_ids"]


# ---------------------------------------------------------------------------
# Per-turn render (roleplay scaffold — each turn carries its own label)
# ---------------------------------------------------------------------------


def _splice_occurrences(
    rendered: str, sides: list[tuple[RoleHeader, list[str | None]]]
) -> str:
    """Replace each side's k-th header occurrence with ``labels[k]``.

    Occurrence *k* corresponds to the *k*-th turn of that side, because the
    chat template emits turns in order.  Graceful degradation in two
    directions: extra labels beyond the occurrences found are ignored (the
    gpt-oss case — its ``<|start|>assistant`` generation prompt has no
    ``<|channel|>``, so the trailing ``gen_role`` simply never lands), and
    occurrences beyond ``len(labels)`` are left standard (``None`` also
    leaves the standard label).

    Both sides splice in **one** left-to-right scan of the original string,
    which is what makes the cast label a free string.  Splicing the sides in
    sequence would let one side write the *other* side's standard header into
    the buffer — a user turn labelled ``assistant`` renders GLM's
    ``<|assistant|>`` — and the second pass would then claim that spliced
    header as its own occurrence 0, shifting every real assistant label by
    one.  A single scan only ever matches template bytes, never bytes this
    function wrote.
    """
    patterns = [
        (f"{header.before}{header.label}{header.after}", header, labels)
        for header, labels in sides
        if labels
    ]
    if not patterns:
        return rendered
    counters = [0] * len(patterns)
    parts: list[str] = []
    cursor = 0
    while True:
        # Earliest match across the sides; the longer pattern wins a tie so a
        # family whose one header is a prefix of the other can't mis-slice.
        best_idx = -1
        best_side = -1
        best_len = -1
        for i, (pattern, _header, _labels) in enumerate(patterns):
            idx = rendered.find(pattern, cursor)
            if idx < 0:
                continue
            if best_side < 0 or idx < best_idx or (
                idx == best_idx and len(pattern) > best_len
            ):
                best_idx, best_side, best_len = idx, i, len(pattern)
        if best_side < 0:
            parts.append(rendered[cursor:])
            break
        idx, side, width = best_idx, best_side, best_len
        pattern, header, labels = patterns[side]
        occ = counters[side]
        counters[side] = occ + 1
        parts.append(rendered[cursor:idx])
        label = labels[occ] if occ < len(labels) else None
        if label is None:
            parts.append(pattern)
        else:
            parts.append(
                f"{header.before}{render_role_label(header.label, label)}{header.after}"
            )
        cursor = idx + width
    return "".join(parts)


def apply_with_per_turn_roles(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    gen_role: str | None = None,
    model_type: str,
    add_generation_prompt: bool = True,
    tokenize: bool = True,
    return_tensors: str | None = None,
    **chat_template_kwargs: Any,
) -> Any:
    """Render a chat template where each turn carries its own role label.

    Unlike :func:`apply_with_role` (which splices one label across every
    turn of a side), this honors a per-message ``"label"`` key — the
    roleplay scaffold where turn 1 was sent as ``captain``, turn 2 generated
    as ``pirate``, etc.  ``gen_role`` is the label for the trailing
    generation-prompt assistant header (the turn about to be generated).

    Each message is ``{"role": ..., "content": ..., "label": str | None}``;
    a missing / ``None`` ``label`` keeps the standard role for that turn.
    When no message carries a label and ``gen_role`` is ``None`` this is a
    zero-overhead pass-through to :meth:`apply_chat_template`.

    Best-effort by design: a missing header occurrence is never a drift
    error (user turns are data-dependent and gpt-oss's generation prompt
    diverges from its historical assistant header).  Slug validation and
    family-support checks still raise.

    This is the fallback render path — it serves families whose chat template
    defeats the scene autopsy (GLM-4.7-Flash) and label-carrying renders the
    stitcher declines (a mid-conversation system turn).  A validated
    :class:`~saklas.core.scene.TurnGrammar` places labels in headers it
    *constructs* and never occurrence-matches at all.
    """
    user_labels = [m.get("label") for m in messages if m.get("role") == "user"]
    asst_labels = [m.get("label") for m in messages if m.get("role") == "assistant"]
    if add_generation_prompt:
        asst_labels = asst_labels + [gen_role]

    has_user = any(lbl for lbl in user_labels)
    has_asst = any(lbl for lbl in asst_labels)

    # Standard render always uses the canonical role strings — strip the
    # ``label`` key so the Jinja template's role branches still fire.
    clean = [
        {"role": m["role"], "content": m.get("content", "")} for m in messages
    ]

    if not has_user and not has_asst:
        return tokenizer.apply_chat_template(
            clean,
            add_generation_prompt=add_generation_prompt,
            tokenize=tokenize,
            return_tensors=return_tensors,
            **chat_template_kwargs,
        )

    user_header: RoleHeader | None = None
    asst_header: RoleHeader | None = None
    if has_user:
        for lbl in user_labels:
            if lbl is not None:
                validate_role(lbl)
        user_header = _lookup_header(USER_ROLE_HEADERS, model_type, side="user")
    if has_asst:
        for lbl in asst_labels:
            if lbl is not None:
                validate_role(lbl)
        asst_header = _lookup_header(ROLE_HEADERS, model_type, side="assistant")

    rendered = tokenizer.apply_chat_template(
        clean,
        add_generation_prompt=add_generation_prompt,
        tokenize=False,
        **chat_template_kwargs,
    )
    if not isinstance(rendered, str):
        raise RoleTemplateDriftError(
            f"apply_chat_template returned {type(rendered).__name__} with "
            f"tokenize=False; role substitution expects a string"
        )

    sides: list[tuple[RoleHeader, list[str | None]]] = []
    if user_header is not None:
        sides.append((user_header, user_labels))
    if asst_header is not None:
        sides.append((asst_header, asst_labels))
    rendered = _splice_occurrences(rendered, sides)

    if not tokenize:
        return rendered
    return tokenizer(
        rendered,
        return_tensors=return_tensors,
        add_special_tokens=False,
    )["input_ids"]
