"""Tests for ``saklas.core.role_templates``.

Strategy: use a small ``FakeTokenizer`` shim with hand-written Jinja
templates that mirror the per-family chat-template shapes saklas
supports.  This is faster and more deterministic than tokenizer-only
HF downloads, and it lets us drive template-drift scenarios without
patching anything global.

The Jinja templates here are representative of the real ones — they
emit the exact byte sequence each family's role header uses, so the
splice logic in :func:`role_templates.apply_with_role` is exercised
against the same bytes it would meet in production.  Token-id values
are arbitrary; what matters is the surface text and the round-trip
behaviour.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jinja2
import pytest

from saklas.core.errors import SaklasError
from saklas.core.role_templates import (
    InvalidRoleError,
    ROLE_HEADERS,
    USER_ROLE_HEADERS,
    RoleHeader,
    RoleSubstitutionUnsupportedError,
    RoleTemplateDriftError,
    apply_with_role,
    apply_with_per_turn_roles,
)


# ---------------------------------------------------------------------------
# Per-family representative chat templates
# ---------------------------------------------------------------------------


# Mirrors the Qwen2/Qwen3 chat-template shape: ``<|im_start|>{role}\n{content}<|im_end|>``.
QWEN_TEMPLATE = (
    "{% for m in messages %}"
    "<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


# Gemma 2/3: ``<start_of_turn>{role}\n{content}<end_of_turn>`` — and the
# assistant role is labeled ``model``, not ``assistant``.
GEMMA_TEMPLATE = (
    "{% for m in messages %}"
    "{% set role = 'model' if m['role'] == 'assistant' else m['role'] %}"
    "<start_of_turn>{{ role }}\n{{ m['content'] }}<end_of_turn>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}"
)


# Gemma 4: same ``model`` label, new turn delimiters
# (``<|turn>...<turn|>`` instead of ``<start_of_turn>...<end_of_turn>``).
GEMMA4_TEMPLATE = (
    "{% for m in messages %}"
    "{% set role = 'model' if m['role'] == 'assistant' else m['role'] %}"
    "<|turn>{{ role }}\n{{ m['content'] }}<turn|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|turn>model\n{% endif %}"
)


# Llama-3: ``<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>``.
LLAMA_TEMPLATE = (
    "{% for m in messages %}"
    "<|start_header_id|>{{ m['role'] }}<|end_header_id|>\n\n"
    "{{ m['content'] }}<|eot_id|>"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}"
)


# GLM-4 chat-template-ish: ``<|{role}|>\n{content}``.
GLM_TEMPLATE = (
    "{% for m in messages %}"
    "<|{{ m['role'] }}|>\n{{ m['content'] }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
)


# GPT-OSS: ``<|start|>{role}<|channel|>{content}<|end|>``.
GPT_OSS_TEMPLATE = (
    "{% for m in messages %}"
    "<|start|>{{ m['role'] }}<|channel|>{{ m['content'] }}<|end|>"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|start|>assistant<|channel|>{% endif %}"
)


# Mistral-3-ish: positional [INST] / [/INST], no role label in the rendered
# string.  Included to test that the registry's None entry raises rather
# than silently no-ops.
MISTRAL_TEMPLATE = (
    "{% for m in messages %}"
    "{% if m['role'] == 'user' %}[INST] {{ m['content'] }} [/INST]"
    "{% else %}{{ m['content'] }}{% endif %}"
    "{% endfor %}"
)


# ---------------------------------------------------------------------------
# FakeTokenizer — minimal apply_chat_template + __call__ surface
# ---------------------------------------------------------------------------


@dataclass
class _FakeBatch:
    input_ids: Any  # list[int] or Tensor depending on return_tensors

    def __getitem__(self, key: str) -> Any:
        if key == "input_ids":
            return self.input_ids
        raise KeyError(key)


class FakeTokenizer:
    """Minimal stand-in for an HF tokenizer.

    Implements just enough of the surface that
    :func:`apply_with_role` needs: ``apply_chat_template`` (Jinja-rendered)
    and ``__call__`` (whitespace tokenization, deterministic id assignment).
    """

    def __init__(self, template: str):
        self._template = jinja2.Environment().from_string(template)

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool = True,
        tokenize: bool = True,
        return_tensors: str | None = None,
        **kwargs: Any,
    ) -> Any:
        rendered = self._template.render(
            messages=messages,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )
        if not tokenize:
            return rendered
        return self._tokenize(rendered, return_tensors=return_tensors)

    def __call__(
        self,
        text: str,
        *,
        return_tensors: str | None = None,
        add_special_tokens: bool = True,
    ) -> _FakeBatch:
        return _FakeBatch(input_ids=self._tokenize(text, return_tensors=return_tensors))

    @staticmethod
    def _tokenize(text: str, *, return_tensors: str | None = None) -> Any:
        # Stable deterministic mapping: token id is the hash of each
        # whitespace-split chunk mod a small constant.  Real values
        # don't matter for these tests.
        ids = [hash(chunk) % 50_000 for chunk in text.split()]
        if return_tensors == "pt":
            import torch

            return torch.tensor([ids])
        return ids


def _qwen_tok() -> FakeTokenizer:
    return FakeTokenizer(QWEN_TEMPLATE)


def _gemma_tok() -> FakeTokenizer:
    return FakeTokenizer(GEMMA_TEMPLATE)


def _gemma4_tok() -> FakeTokenizer:
    return FakeTokenizer(GEMMA4_TEMPLATE)


def _llama_tok() -> FakeTokenizer:
    return FakeTokenizer(LLAMA_TEMPLATE)


def _glm_tok() -> FakeTokenizer:
    return FakeTokenizer(GLM_TEMPLATE)


def _gpt_oss_tok() -> FakeTokenizer:
    return FakeTokenizer(GPT_OSS_TEMPLATE)


def _mistral_tok() -> FakeTokenizer:
    return FakeTokenizer(MISTRAL_TEMPLATE)


def _sample_messages():
    return [
        {"role": "system", "content": "be brief."},
        {"role": "user", "content": "hello"},
    ]


# ---------------------------------------------------------------------------
# Pass-through (role=None) — no extra work
# ---------------------------------------------------------------------------


def test_apply_with_role_none_is_passthrough_string():
    """role=None returns byte-for-byte identical output to apply_chat_template."""
    tok = _qwen_tok()
    messages = _sample_messages()
    direct = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    via_helper = apply_with_role(
        tok,
        messages,
        role=None,
        model_type="qwen3",
        tokenize=False,
    )
    assert via_helper == direct


def test_apply_with_role_none_is_passthrough_unknown_model_type():
    """role=None does not consult the registry — unknown model types pass through."""
    tok = _qwen_tok()
    messages = _sample_messages()
    # ``totally-fake`` would raise under role substitution, but role=None
    # is a transparent pass-through that never touches the registry.
    out = apply_with_role(
        tok,
        messages,
        role=None,
        model_type="totally-fake",
        tokenize=False,
    )
    expected = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    assert out == expected


def test_apply_with_role_none_unsupported_family_passes_through():
    """role=None with an explicitly-unsupported family (mistral3) still passes through."""
    tok = _mistral_tok()
    messages = [{"role": "user", "content": "hi"}]
    out = apply_with_role(
        tok,
        messages,
        role=None,
        model_type="mistral3",
        tokenize=False,
    )
    expected = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    # Mistral template ignores add_generation_prompt, so both renders match.
    # Use add_generation_prompt=False to align both calls.
    assert out == expected
    out_no_gen = apply_with_role(
        tok,
        messages,
        role=None,
        model_type="mistral3",
        tokenize=False,
        add_generation_prompt=False,
    )
    assert out_no_gen == expected


# ---------------------------------------------------------------------------
# Per-family splice
# ---------------------------------------------------------------------------


def test_apply_with_role_qwen3():
    """Qwen3: rendered prompt contains <|im_start|>pirate\\n exactly once
    and <|im_start|>assistant\\n zero times.
    """
    tok = _qwen_tok()
    messages = _sample_messages()
    out = apply_with_role(
        tok,
        messages,
        role="pirate",
        model_type="qwen3",
        tokenize=False,
    )
    assert out.count("<|im_start|>pirate\n") == 1
    assert "<|im_start|>assistant\n" not in out
    # User / system roles untouched — only the assistant header swaps.
    assert "<|im_start|>user\n" in out
    assert "<|im_start|>system\n" in out


def test_apply_with_role_qwen3_text_variant_inherits_parent():
    """Sub-variant model_types (qwen3_text) inherit their parent family's header."""
    tok = _qwen_tok()
    messages = _sample_messages()
    out = apply_with_role(
        tok,
        messages,
        role="pirate",
        model_type="qwen3_text",
        tokenize=False,
    )
    assert "<|im_start|>pirate\n" in out


def test_apply_with_role_gemma3_label_remap():
    """Gemma's pre-substitution label is 'model', not 'assistant' — the
    splice replaces the model marker so a custom role still lands.
    """
    tok = _gemma_tok()
    messages = _sample_messages()
    out = apply_with_role(
        tok,
        messages,
        role="pirate",
        model_type="gemma3",
        tokenize=False,
    )
    assert "<start_of_turn>pirate\n" in out
    # Critically: the *assistant turn marker* was the literal "model" label,
    # so it's now gone.  User-role markers untouched.
    assert "<start_of_turn>model\n" not in out
    assert "<start_of_turn>user\n" in out


def test_apply_with_role_gemma3_text_variant_inherits():
    """gemma3_text inherits the gemma family's <start_of_turn> RoleHeader."""
    tok = _gemma_tok()
    out = apply_with_role(
        tok,
        _sample_messages(),
        role="pirate",
        model_type="gemma3_text",
        tokenize=False,
    )
    assert "<start_of_turn>pirate\n" in out


def test_apply_with_role_gemma4_variants_use_new_delimiter():
    """gemma4 and gemma4_text use the <|turn> delimiter, distinct from gemma 2/3.

    Gemma 4 ships a new turn-boundary scheme; the registry carries a
    separate ``gemma4`` :class:`RoleHeader` and ``gemma4_text`` inherits
    that — not the gemma 2/3 ``<start_of_turn>`` header.
    """
    for mt in ("gemma4", "gemma4_text"):
        tok = _gemma4_tok()
        out = apply_with_role(
            tok,
            _sample_messages(),
            role="pirate",
            model_type=mt,
            tokenize=False,
        )
        assert "<|turn>pirate\n" in out, f"failed for {mt}"
        # The substituted assistant marker is gone; user-role markers untouched.
        assert "<|turn>model\n" not in out
        assert "<|turn>user\n" in out


def test_apply_with_role_llama3():
    """Llama-3: <|start_header_id|>pirate<|end_header_id|> appears."""
    tok = _llama_tok()
    messages = _sample_messages()
    out = apply_with_role(
        tok,
        messages,
        role="pirate",
        model_type="llama",
        tokenize=False,
    )
    assert "<|start_header_id|>pirate<|end_header_id|>" in out
    assert "<|start_header_id|>assistant<|end_header_id|>" not in out
    # Non-assistant headers (user, system) untouched.
    assert "<|start_header_id|>user<|end_header_id|>" in out


def test_apply_with_role_glm():
    """GLM-4: <|pirate|> appears, <|assistant|> does not."""
    tok = _glm_tok()
    messages = _sample_messages()
    out = apply_with_role(
        tok,
        messages,
        role="pirate",
        model_type="glm",
        tokenize=False,
    )
    assert "<|pirate|>" in out
    assert "<|assistant|>" not in out


def test_apply_with_role_gpt_oss():
    """GPT-OSS: <|start|>pirate<|channel|> appears, <|start|>assistant<|channel|> does not."""
    tok = _gpt_oss_tok()
    messages = _sample_messages()
    out = apply_with_role(
        tok,
        messages,
        role="pirate",
        model_type="gpt_oss",
        tokenize=False,
    )
    assert "<|start|>pirate<|channel|>" in out
    assert "<|start|>assistant<|channel|>" not in out


# ---------------------------------------------------------------------------
# Unsupported / drift / invalid role
# ---------------------------------------------------------------------------


def test_apply_with_role_mistral_raises():
    """Mistral-3 maps to None in the registry — substitution raises."""
    tok = _mistral_tok()
    messages = [{"role": "user", "content": "hi"}]
    with pytest.raises(RoleSubstitutionUnsupportedError):
        apply_with_role(
            tok,
            messages,
            role="pirate",
            model_type="mistral3",
            tokenize=False,
        )


def test_apply_with_role_ministral_raises():
    """ministral3 also opts out."""
    tok = _mistral_tok()
    with pytest.raises(RoleSubstitutionUnsupportedError):
        apply_with_role(
            tok,
            [{"role": "user", "content": "hi"}],
            role="pirate",
            model_type="ministral3",
            tokenize=False,
        )


def test_apply_with_role_talkie_raises():
    """talkie explicitly opts out (untested family)."""
    tok = _qwen_tok()  # template doesn't matter — registry blocks first
    with pytest.raises(RoleSubstitutionUnsupportedError):
        apply_with_role(
            tok,
            _sample_messages(),
            role="pirate",
            model_type="talkie",
            tokenize=False,
        )


def test_apply_with_role_unknown_model_type_raises():
    """An unknown model_type isn't silently treated as default-assistant."""
    tok = _qwen_tok()
    with pytest.raises(RoleSubstitutionUnsupportedError) as exc_info:
        apply_with_role(
            tok,
            _sample_messages(),
            role="pirate",
            model_type="totally-fake",
            tokenize=False,
        )
    # Clear error message — mentions the unknown model_type.
    assert "totally-fake" in str(exc_info.value)


def test_apply_with_role_template_drift_raises():
    """When the rendered template lacks the expected role-header bytes,
    splice raises rather than silently no-oping.
    """
    # A Qwen-shaped registry entry, but a tokenizer whose template emits
    # a *different* role header — the splice will find zero matches.
    drifted_template = (
        "{% for m in messages %}"
        "ROLE={{ m['role'] }} CONTENT={{ m['content'] }}\n"
        "{% endfor %}"
        "{% if add_generation_prompt %}ROLE=assistant{% endif %}"
    )
    tok = FakeTokenizer(drifted_template)
    with pytest.raises(RoleTemplateDriftError):
        apply_with_role(
            tok,
            _sample_messages(),
            role="pirate",
            model_type="qwen3",
            tokenize=False,
        )


def test_apply_with_role_invalid_role_slug_uppercase():
    """Uppercase / spaces / disallowed punctuation are rejected up front."""
    tok = _qwen_tok()
    with pytest.raises(InvalidRoleError):
        apply_with_role(
            tok,
            _sample_messages(),
            role="UPPERCASE PIRATE",
            model_type="qwen3",
            tokenize=False,
        )


def test_apply_with_role_invalid_role_empty():
    """Empty string is rejected (otherwise the splice would produce an empty header)."""
    tok = _qwen_tok()
    with pytest.raises(InvalidRoleError):
        apply_with_role(
            tok,
            _sample_messages(),
            role="",
            model_type="qwen3",
            tokenize=False,
        )


def test_apply_with_role_invalid_role_punctuation():
    """Punctuation outside ``[a-z0-9._-]`` is rejected (here: ``/``)."""
    tok = _qwen_tok()
    with pytest.raises(InvalidRoleError):
        apply_with_role(
            tok,
            _sample_messages(),
            role="ns/pirate",
            model_type="qwen3",
            tokenize=False,
        )


def test_apply_with_role_valid_role_with_underscore_and_dot():
    """``[a-z0-9._-]`` includes underscore, dot, dash — all valid.

    At render the slug de-slugs (``_`` → space) while dots/dashes are kept,
    so ``sea_dog`` displays as ``sea dog`` but ``captain.ahab`` / ``pirate-king``
    are spliced verbatim.
    """
    tok = _qwen_tok()
    for role, shown in (
        ("pirate", "pirate"),
        ("sea_dog", "sea dog"),
        ("captain.ahab", "captain.ahab"),
        ("pirate-king", "pirate-king"),
        ("p1rate", "p1rate"),
        ("someone_happy", "someone happy"),
    ):
        out = apply_with_role(
            tok,
            _sample_messages(),
            role=role,
            model_type="qwen3",
            tokenize=False,
        )
        assert f"<|im_start|>{shown}\n" in out


# ---------------------------------------------------------------------------
# Tokenize True / False
# ---------------------------------------------------------------------------


def test_apply_with_role_tokenize_false_returns_string():
    tok = _qwen_tok()
    out = apply_with_role(
        tok,
        _sample_messages(),
        role="pirate",
        model_type="qwen3",
        tokenize=False,
    )
    assert isinstance(out, str)


def test_apply_with_role_tokenize_true_returns_ids():
    tok = _qwen_tok()
    out = apply_with_role(
        tok,
        _sample_messages(),
        role="pirate",
        model_type="qwen3",
        tokenize=True,
    )
    # FakeTokenizer returns a list of ints when return_tensors is None.
    assert isinstance(out, list)
    assert all(isinstance(t, int) for t in out)


def test_apply_with_role_tokenize_true_with_pt_tensors():
    torch = pytest.importorskip("torch")
    tok = _qwen_tok()
    out = apply_with_role(
        tok,
        _sample_messages(),
        role="pirate",
        model_type="qwen3",
        tokenize=True,
        return_tensors="pt",
    )
    assert isinstance(out, torch.Tensor)
    assert out.ndim == 2  # (1, seq)


# ---------------------------------------------------------------------------
# Error hierarchy / family catch
# ---------------------------------------------------------------------------


def test_errors_subclass_saklas_error():
    """All three error types catch under except SaklasError, matching project convention."""
    for cls in (
        RoleSubstitutionUnsupportedError,
        RoleTemplateDriftError,
        InvalidRoleError,
    ):
        assert issubclass(cls, SaklasError), cls
    # And the stdlib parents are preserved per the errors.py contract.
    assert issubclass(RoleSubstitutionUnsupportedError, ValueError)
    assert issubclass(InvalidRoleError, ValueError)
    assert issubclass(RoleTemplateDriftError, RuntimeError)


def test_unsupported_error_caught_as_value_error():
    """Existing except ValueError sites still catch RoleSubstitutionUnsupportedError."""
    tok = _qwen_tok()
    with pytest.raises(ValueError):
        apply_with_role(
            tok,
            _sample_messages(),
            role="pirate",
            model_type="totally-fake",
            tokenize=False,
        )


# ---------------------------------------------------------------------------
# Registry sanity
# ---------------------------------------------------------------------------


def test_registry_covers_tested_archs():
    """Every model_type in core.model._TESTED_ARCHS has a registry entry
    (either a RoleHeader or an explicit None).  The registry must be
    exhaustive over the tested-arch set so saklas's first-class
    architectures all have a decision recorded.
    """
    from saklas.core.model import _TESTED_ARCHS

    for arch in _TESTED_ARCHS:
        # Sub-variants we explicitly route through their parent.  Build
        # the set of acceptable mappings: an exact registry hit, or a
        # parent-family hit reachable by stripping a known suffix.
        if arch in ROLE_HEADERS:
            continue
        # Allow text/moe suffix-stripping to the parent family.
        for suffix in ("_text", "_moe"):
            if arch.endswith(suffix) and arch[: -len(suffix)] in ROLE_HEADERS:
                break
        else:
            raise AssertionError(
                f"_TESTED_ARCHS member {arch!r} has no role_templates registry entry"
            )


def test_role_header_is_frozen():
    """RoleHeader is a frozen dataclass — registry entries can't be mutated post-import."""
    header = RoleHeader(before="x", after="y", label="z")
    with pytest.raises((AttributeError, Exception)):
        header.before = "mutated"  # pyright: ignore[reportAttributeAccessIssue]  # frozen dataclass


# ---------------------------------------------------------------------------
# build_chat_input per-turn role plumbing
# ---------------------------------------------------------------------------


def test_build_chat_input_with_gen_role():
    """``build_chat_input(gen_role=...)`` routes through the per-turn render
    so the generation prompt opens with the substituted assistant label.
    """
    from saklas.core.generation import build_chat_input

    tok: Any = _qwen_tok()
    tok.chat_template = QWEN_TEMPLATE

    messages = [{"role": "user", "content": "hello"}]
    input_ids = build_chat_input(
        tok, messages, system_prompt="be brief.",
        gen_role="pirate", model_type="qwen3",
    )
    rendered = apply_with_per_turn_roles(
        tok, [{"role": "system", "content": "be brief."}] + messages,
        gen_role="pirate", model_type="qwen3",
        tokenize=False,
    )
    assert "<|im_start|>pirate\n" in rendered
    assert "<|im_start|>assistant\n" not in rendered
    expected_ids = tok._tokenize(rendered, return_tensors="pt")
    import torch
    assert torch.equal(input_ids, expected_ids)


def test_build_chat_input_with_per_turn_labels():
    """Per-message ``label`` keys drive a faithful per-turn render."""
    from saklas.core.generation import build_chat_input

    tok: Any = _qwen_tok()
    tok.chat_template = QWEN_TEMPLATE

    messages = [
        {"role": "user", "content": "hi", "label": "captain"},
        {"role": "assistant", "content": "arr", "label": "pirate"},
        {"role": "user", "content": "more", "label": None},
    ]
    rendered = apply_with_per_turn_roles(
        tok, messages, gen_role="oracle", model_type="qwen3",
        tokenize=False,
    )
    assert "<|im_start|>captain\n" in rendered   # labeled user turn
    assert "<|im_start|>pirate\n" in rendered    # labeled assistant turn
    assert "<|im_start|>user\n" in rendered      # unlabeled user turn stays
    assert "<|im_start|>oracle\n" in rendered    # generation prompt
    # build_chat_input takes the per-turn path when labels are present.
    ids = build_chat_input(tok, messages, gen_role="oracle", model_type="qwen3")
    assert ids.shape[1] > 0


def test_build_chat_input_no_labels_unchanged():
    """No labels + ``gen_role=None`` is byte-identical to the plain path."""
    from saklas.core.generation import build_chat_input

    tok: Any = _qwen_tok()
    tok.chat_template = QWEN_TEMPLATE

    messages = [{"role": "user", "content": "hello"}]
    direct = build_chat_input(tok, messages, system_prompt="be brief.")
    via_none = build_chat_input(
        tok, messages, system_prompt="be brief.", gen_role=None,
    )
    import torch
    assert torch.equal(direct, via_none)


def test_build_chat_input_gen_role_requires_model_type():
    """``gen_role=`` without ``model_type=`` raises — the family lookup is
    required to find the right role-header bytes to splice.
    """
    from saklas.core.generation import build_chat_input

    tok: Any = _qwen_tok()
    tok.chat_template = QWEN_TEMPLATE

    with pytest.raises(ValueError, match="model_type"):
        build_chat_input(
            tok, [{"role": "user", "content": "hi"}],
            gen_role="pirate",
        )


# ---------------------------------------------------------------------------
# User-role splice (both-sides relabeling)
# ---------------------------------------------------------------------------


# gpt-oss's real Harmony format renders user turns with ``<|message|>`` (not
# the assistant ``<|channel|>``) — verified against ``openai/gpt-oss-20b``.
# The simplified GPT_OSS_TEMPLATE above collapses both to ``<|channel|>``, so
# the user-side splice needs a Harmony-accurate template to exercise the
# real ``<|start|>user<|message|>`` byte sequence.
GPT_OSS_HARMONY_TEMPLATE = (
    "{% for m in messages %}"
    "{% if m['role'] == 'assistant' %}"
    "<|start|>assistant<|channel|>final<|message|>{{ m['content'] }}<|end|>"
    "{% else %}"
    "<|start|>{{ m['role'] }}<|message|>{{ m['content'] }}<|end|>"
    "{% endif %}"
    "{% endfor %}"
    # Real Harmony emits a *bare* ``<|start|>assistant`` generation prompt
    # (no ``<|channel|>``), distinct from a completed assistant turn — so a
    # gen-prompt assistant relabel can't land there.
    "{% if add_generation_prompt %}<|start|>assistant{% endif %}"
)


@pytest.mark.parametrize(
    "tok_factory,model_type,expected",
    [
        (_qwen_tok, "qwen3", "<|im_start|>captain\n"),
        (_gemma_tok, "gemma3", "<start_of_turn>captain\n"),
        (_gemma4_tok, "gemma4", "<|turn>captain\n"),
        (_llama_tok, "llama", "<|start_header_id|>captain<|end_header_id|>"),
        (_glm_tok, "glm", "<|captain|>"),
        (
            lambda: FakeTokenizer(GPT_OSS_HARMONY_TEMPLATE),
            "gpt_oss",
            "<|start|>captain<|message|>",
        ),
    ],
)
def test_apply_with_user_role_per_family(tok_factory: Any, model_type: str, expected: str) -> None:
    """user_role relabels the user turn for every supported family."""
    tok = tok_factory()
    out = apply_with_role(
        tok,
        _sample_messages(),
        role=None,
        user_role="captain",
        model_type=model_type,
        tokenize=False,
    )
    assert expected in out


def test_apply_with_user_role_leaves_assistant_alone():
    """user_role swaps only the user header; the assistant generation
    prompt keeps its standard label when role is None."""
    tok = _qwen_tok()
    out = apply_with_role(
        tok,
        _sample_messages(),
        role=None,
        user_role="captain",
        model_type="qwen3",
        tokenize=False,
    )
    assert "<|im_start|>captain\n" in out
    assert "<|im_start|>user\n" not in out
    # Assistant generation prompt untouched.
    assert "<|im_start|>assistant\n" in out


def test_apply_with_both_roles_at_once():
    """role + user_role swap both headers independently."""
    tok = _qwen_tok()
    out = apply_with_role(
        tok,
        _sample_messages(),
        role="oracle",
        user_role="captain",
        model_type="qwen3",
        tokenize=False,
    )
    assert "<|im_start|>captain\n" in out          # user turn
    assert "<|im_start|>oracle\n" in out           # generation prompt
    assert "<|im_start|>user\n" not in out
    assert "<|im_start|>assistant\n" not in out


def test_apply_with_user_role_lenient_when_no_user_turn():
    """A missing user header is a no-op, not a drift error — user turns
    are data-dependent (here: a system-only history)."""
    tok = _qwen_tok()
    messages = [{"role": "system", "content": "be brief."}]
    out = apply_with_role(
        tok,
        messages,
        role=None,
        user_role="captain",
        model_type="qwen3",
        tokenize=False,
    )
    # No user turn rendered → captain never appears, and nothing raised.
    assert "captain" not in out
    assert "<|im_start|>system\n" in out


def test_apply_with_user_role_unsupported_family_raises():
    """user_role on a label-free family raises, like the assistant side."""
    tok = _mistral_tok()
    with pytest.raises(RoleSubstitutionUnsupportedError):
        apply_with_role(
            tok,
            _sample_messages(),
            role=None,
            user_role="captain",
            model_type="mistral3",
            tokenize=False,
        )


def test_apply_with_user_role_invalid_slug_raises():
    """user_role slug validation matches the assistant side."""
    tok = _qwen_tok()
    with pytest.raises(InvalidRoleError):
        apply_with_role(
            tok,
            _sample_messages(),
            role=None,
            user_role="Captain Ahab",
            model_type="qwen3",
            tokenize=False,
        )


def test_user_role_registry_covers_tested_archs():
    """Every ``_TESTED_ARCHS`` member has a USER_ROLE_HEADERS entry
    (RoleHeader or explicit None), mirroring the assistant registry."""
    from saklas.core.model import _TESTED_ARCHS

    for arch in _TESTED_ARCHS:
        if arch in USER_ROLE_HEADERS:
            continue
        for suffix in ("_text", "_moe"):
            if arch.endswith(suffix) and arch[: -len(suffix)] in USER_ROLE_HEADERS:
                break
        else:
            raise AssertionError(
                f"_TESTED_ARCHS member {arch!r} has no USER_ROLE_HEADERS entry"
            )


def test_user_role_registry_opt_outs_match_assistant():
    """The label-free families opt out of *both* sides."""
    for mt in ("mistral3", "ministral3", "talkie"):
        assert ROLE_HEADERS[mt] is None
        assert USER_ROLE_HEADERS[mt] is None


def test_build_chat_input_user_label_distinct_from_plain():
    """A per-turn user label produces a distinct render from the plain one
    (cache key keeps them separate)."""
    import torch

    from saklas.core.generation import build_chat_input

    tok: Any = _qwen_tok()
    tok.chat_template = QWEN_TEMPLATE

    plain = build_chat_input(tok, [{"role": "user", "content": "hello"}])
    with_user = build_chat_input(
        tok, [{"role": "user", "content": "hello", "label": "captain"}],
        model_type="qwen3",
    )
    assert not (
        plain.shape == with_user.shape and torch.equal(plain, with_user)
    )


def test_apply_with_per_turn_no_labels_is_passthrough():
    """No labels + no gen_role → byte-identical to apply_chat_template."""
    tok = _qwen_tok()
    messages = _sample_messages()
    direct = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    via = apply_with_per_turn_roles(
        tok, messages, gen_role=None, model_type="qwen3", tokenize=False)
    assert via == direct


def test_apply_with_per_turn_user_label_unsupported_family_raises():
    """A user label on a label-free family raises."""
    tok = _mistral_tok()
    with pytest.raises(RoleSubstitutionUnsupportedError):
        apply_with_per_turn_roles(
            tok,
            [{"role": "user", "content": "hi", "label": "captain"}],
            gen_role=None, model_type="mistral3", tokenize=False,
        )


def test_apply_with_per_turn_gpt_oss_gen_role_graceful():
    """gpt-oss's bare generation prompt has no <|channel|>, so gen_role
    no-ops there while history labels still land (graceful degradation)."""
    tok = FakeTokenizer(GPT_OSS_HARMONY_TEMPLATE)
    msgs = [
        {"role": "user", "content": "hi", "label": "captain"},
        {"role": "assistant", "content": "yo", "label": "pirate"},
        {"role": "user", "content": "more", "label": None},
    ]
    out = apply_with_per_turn_roles(
        tok, msgs, gen_role="oracle", model_type="gpt_oss", tokenize=False)
    assert "<|start|>captain<|message|>" in out   # user label lands
    assert "<|start|>pirate<|channel|>" in out    # history assistant lands
    assert "oracle" not in out                    # gen prompt no-ops


def test_apply_with_per_turn_ordering():
    """Two same-role turns get their own labels in render order."""
    tok = _qwen_tok()
    msgs = [
        {"role": "user", "content": "a", "label": "alice"},
        {"role": "assistant", "content": "x", "label": "bot"},
        {"role": "user", "content": "b", "label": "bob"},
    ]
    out = apply_with_per_turn_roles(
        tok, msgs, gen_role=None, model_type="qwen3", tokenize=False)
    # alice precedes bob in the rendered string.
    assert out.index("<|im_start|>alice\n") < out.index("<|im_start|>bob\n")
