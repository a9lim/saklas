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


# Map ``model.config.model_type`` → header, or ``None`` for families that
# can't support role substitution (positional templates, untested architectures).
#
# Sub-variant model types (``qwen3_text``, ``gemma3_text``, ...) inherit from
# their parent family — same chat template, same byte sequence.
def _build_role_headers() -> dict[str, RoleHeader | None]:
    qwen = RoleHeader(before="<|im_start|>", after="\n", label="assistant")
    gemma = RoleHeader(before="<start_of_turn>", after="\n", label="model")
    # Gemma-4 shipped a new turn-boundary scheme — ``<|turn>...<turn|>``
    # instead of the prior ``<start_of_turn>...<end_of_turn>``.  The label
    # stays ``model`` (gemma family convention), only the surrounding
    # delimiter changed.  Verified empirically against the
    # ``google/gemma-4-31b-it`` tokenizer's rendered chat template (2026-05).
    gemma4 = RoleHeader(before="<|turn>", after="\n", label="model")
    llama = RoleHeader(
        before="<|start_header_id|>", after="<|end_header_id|>", label="assistant"
    )
    glm = RoleHeader(before="<|", after="|>", label="assistant")
    gpt_oss = RoleHeader(
        before="<|start|>", after="<|channel|>", label="assistant"
    )

    table: dict[str, RoleHeader | None] = {
        # Qwen family
        "qwen2": qwen,
        "qwen3": qwen,
        "qwen3_text": qwen,
        "qwen3_moe": qwen,
        "qwen3_5": qwen,
        # Gemma family — note label="model", not "assistant".  Gemma-4
        # uses a different delimiter than 2/3 (see ``gemma4`` above).
        "gemma2": gemma,
        "gemma3": gemma,
        "gemma3_text": gemma,
        "gemma4": gemma4,
        "gemma4_text": gemma4,
        # Llama family
        "llama": llama,
        # GLM (ChatGLM-4)
        "glm": glm,
        # GPT-OSS
        "gpt_oss": gpt_oss,
        # Mistral: positional [INST]/[/INST] in the rendered string — there is
        # no role label to swap, so the strategy doesn't apply.  Likewise
        # talkie (vintage, untested for this feature; opt out explicitly).
        "mistral3": None,
        "ministral3": None,
        "talkie": None,
    }
    return table


ROLE_HEADERS: dict[str, RoleHeader | None] = _build_role_headers()


# ---------------------------------------------------------------------------
# Role-slug validation
# ---------------------------------------------------------------------------


# Same alphabet as ``io.paths._UNSAFE_VARIANT_CHARS`` — the slug discipline
# used for the SAE release and transfer-source filename variants.  A custom
# role label is similar: a user-provided string we splice into a rendered
# template and (downstream) into filenames.  Keeping it ``[a-z0-9._-]`` rules
# out whitespace, uppercase, and punctuation that would corrupt either
# surface.
_ROLE_SLUG_RE = re.compile(r"^[a-z0-9._-]+$")


class RoleSubstitutionUnsupportedError(SaklasError, ValueError):
    """Raised when role substitution is requested for a family that doesn't
    support it.

    Two sub-cases share this error: the family is in :data:`ROLE_HEADERS`
    with value ``None`` (positional templates like Mistral-3, opt-outs like
    talkie), or the family is absent from the registry entirely (unknown
    or unsupported architecture).
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


def _validate_role(role: str) -> None:
    if not role:
        raise InvalidRoleError("role must be a non-empty string")
    if not _ROLE_SLUG_RE.match(role):
        raise InvalidRoleError(
            f"role {role!r} must match [a-z0-9._-]+ "
            f"(lowercase alphanumeric, dot/underscore/dash only)"
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def apply_with_role(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    role: str | None,
    model_type: str,
    add_generation_prompt: bool = True,
    tokenize: bool = True,
    return_tensors: str | None = None,
    **chat_template_kwargs: Any,
) -> Any:
    """Render a chat template with a custom assistant-role label.

    Strategy: render with the standard assistant role, splice the role label
    at the string level, then optionally tokenize.  Token indices remain
    stable in the substituted prompt so saklas's
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
        Custom role label (e.g. ``"pirate"``), or ``None`` to render with
        the family's standard assistant-role label.  ``None`` is a
        zero-overhead transparent pass-through to
        :meth:`apply_chat_template`.
    model_type:
        ``model.config.model_type`` — used to look up the family's
        :class:`RoleHeader`.
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
        ``role`` does not match the role-slug alphabet.
    RoleSubstitutionUnsupportedError:
        ``model_type`` is absent from :data:`ROLE_HEADERS` or maps to
        ``None``.
    RoleTemplateDriftError:
        The rendered template does not contain the family's expected
        role header — the registry has drifted from the live template.
    """
    # role=None is the cheap default path.  Pass through verbatim — no
    # registry lookup, no string scan, no extra allocations.
    if role is None:
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=tokenize,
            return_tensors=return_tensors,
            **chat_template_kwargs,
        )

    _validate_role(role)

    if model_type not in ROLE_HEADERS:
        raise RoleSubstitutionUnsupportedError(
            f"role substitution is not supported for model_type {model_type!r}: "
            f"family is not in the role-header registry"
        )
    header = ROLE_HEADERS[model_type]
    if header is None:
        raise RoleSubstitutionUnsupportedError(
            f"role substitution is not supported for model_type {model_type!r}: "
            f"family uses positional or otherwise label-free chat-template markers"
        )

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

    pattern = f"{header.before}{header.label}{header.after}"
    replacement = f"{header.before}{role}{header.after}"
    hit_count = rendered.count(pattern)
    if hit_count == 0:
        raise RoleTemplateDriftError(
            f"role header {pattern!r} not found in rendered template for "
            f"model_type {model_type!r}; the registry may have drifted from "
            f"the live chat template"
        )
    spliced = rendered.replace(pattern, replacement)

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
