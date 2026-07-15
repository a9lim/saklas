"""Tests for ``saklas.core.scene`` — the cast-model turn stitcher.

Strategy mirrors ``test_role_templates.py``: FakeTokenizer + hand-written
Jinja templates faithful to each family's byte shapes.  Two additional fakes
matter here: a *strict* gemma template (folds system into the first user turn
AND raises on non-alternating roles, like the real one) proving the stitcher
renders sequences the template itself refuses, and a qwen3-style
history-thinking stripper driving the "lasts one turn" policy.
"""
from __future__ import annotations

import pytest

from saklas.core.errors import SaklasError
from saklas.core.scene import (
    SceneGrammarError,
    SceneRenderError,
    SceneThinkingUnsupportedError,
    SceneTurn,
    extract_turn_grammar,
    render_scene,
    render_scene_raw,
    validate_turn_grammar,
)
from tests.test_role_templates import (
    FakeTokenizer,
    GEMMA4_TEMPLATE,
    GEMMA_TEMPLATE,
    GLM_TEMPLATE,
    LLAMA_TEMPLATE,
    MISTRAL_TEMPLATE,
    QWEN_TEMPLATE,
    TALKIE_TEMPLATE,
)


# ---------------------------------------------------------------------------
# Additional fakes
# ---------------------------------------------------------------------------


# Faithful-strict gemma: <bos> prelude, system folded into the first user
# turn (the real gemma has no system role), and non-alternating roles raise
# (``raise_exception`` is undefined in the plain jinja2 environment, so
# calling it throws — the same observable behavior as the real template).
GEMMA_STRICT_TEMPLATE = (
    "<bos>"
    "{% if messages[0]['role'] == 'system' %}"
    "{% set sys = messages[0]['content'] %}{% set msgs = messages[1:] %}"
    "{% else %}{% set sys = '' %}{% set msgs = messages %}{% endif %}"
    "{% for m in msgs %}"
    "{% set role = 'model' if m['role'] == 'assistant' else m['role'] %}"
    "{% set expected = 'user' if loop.index0 % 2 == 0 else 'model' %}"
    "{% if role != expected %}"
    "{{ raise_exception('roles must alternate') }}"
    "{% endif %}"
    "<start_of_turn>{{ role }}\n"
    "{% if loop.first and sys %}{{ sys }}\n\n{% endif %}"
    "{{ m['content'] }}<end_of_turn>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}"
)


# Qwen3-style: strips think blocks from assistant history and appends an
# empty think scaffold to the generation prompt (the gen_extra case).
QWEN_STRIP_TEMPLATE = (
    "{% for m in messages %}"
    "{% set c = m['content'] %}"
    "{% if m['role'] == 'assistant' and '</think>' in c %}"
    "{% set c = c.split('</think>')[-1] %}"
    "{% endif %}"
    "<|im_start|>{{ m['role'] }}\n{{ c }}<|im_end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    "{% endif %}"
)


def _grammar(template: str, model_type: str):
    tok = FakeTokenizer(template)
    return tok, extract_turn_grammar(tok, model_type)


# ---------------------------------------------------------------------------
# Autopsy + round-trip validation across families
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "template, model_type",
    [
        (QWEN_TEMPLATE, "qwen3"),
        (GEMMA_TEMPLATE, "gemma3"),
        (GEMMA_STRICT_TEMPLATE, "gemma3"),
        (GEMMA4_TEMPLATE, "gemma4"),
        (LLAMA_TEMPLATE, "llama"),
        (GLM_TEMPLATE, "glm"),
        (TALKIE_TEMPLATE, "talkie"),
        (QWEN_STRIP_TEMPLATE, "qwen3"),
    ],
    ids=["qwen", "gemma", "gemma-strict", "gemma4", "llama", "glm", "talkie",
         "qwen-strip"],
)
def test_round_trip_validation(template: str, model_type: str):
    """The stitcher reproduces the template's bytes on alternating input."""
    tok, grammar = _grammar(template, model_type)
    validate_turn_grammar(grammar, tok)


def test_mistral_is_unsupported():
    tok = FakeTokenizer(MISTRAL_TEMPLATE)
    with pytest.raises(SceneGrammarError):
        extract_turn_grammar(tok, "mistral3")


def test_unknown_family_is_unsupported():
    tok = FakeTokenizer(QWEN_TEMPLATE)
    with pytest.raises(SceneGrammarError):
        extract_turn_grammar(tok, "wumpus")


def test_scene_errors_are_saklas_errors():
    assert issubclass(SceneGrammarError, SaklasError)
    assert issubclass(SceneRenderError, SaklasError)
    assert issubclass(SceneThinkingUnsupportedError, SaklasError)


# ---------------------------------------------------------------------------
# Arbitrary seat sequences — the point of the module
# ---------------------------------------------------------------------------


def test_non_alternating_sequence_renders():
    """a/a/u/u renders through the stitcher; the template itself raises."""
    tok, grammar = _grammar(GEMMA_STRICT_TEMPLATE, "gemma3")

    # The template refuses this sequence outright.
    with pytest.raises(Exception):
        tok.apply_chat_template(
            [
                {"role": "assistant", "content": "one"},
                {"role": "assistant", "content": "two"},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )

    got = render_scene(
        grammar,
        [
            SceneTurn(seat="assistant", text="one"),
            SceneTurn(seat="assistant", text="two"),
            SceneTurn(seat="user", text="three"),
            SceneTurn(seat="user", text="four"),
        ],
        gen_seat="assistant",
    )
    assert got == (
        "<bos>"
        "<start_of_turn>model\none<end_of_turn>\n"
        "<start_of_turn>model\ntwo<end_of_turn>\n"
        "<start_of_turn>user\nthree<end_of_turn>\n"
        "<start_of_turn>user\nfour<end_of_turn>\n"
        "<start_of_turn>model\n"
    )


def test_seat_swap_is_pure_reassignment():
    """u/a vs a/u over the same text differ only in the seat headers."""
    _, grammar = _grammar(GEMMA_STRICT_TEMPLATE, "gemma3")
    ua = render_scene(grammar, [
        SceneTurn(seat="user", text="hello"),
        SceneTurn(seat="assistant", text="world"),
    ])
    au = render_scene(grammar, [
        SceneTurn(seat="assistant", text="hello"),
        SceneTurn(seat="user", text="world"),
    ])
    assert ua == (
        "<bos><start_of_turn>user\nhello<end_of_turn>\n"
        "<start_of_turn>model\nworld<end_of_turn>\n"
    )
    assert au == (
        "<bos><start_of_turn>model\nhello<end_of_turn>\n"
        "<start_of_turn>user\nworld<end_of_turn>\n"
    )


def test_unknown_seat_raises():
    _, grammar = _grammar(QWEN_TEMPLATE, "qwen3")
    with pytest.raises(SceneRenderError):
        render_scene(grammar, [SceneTurn(seat="narrator", text="x")])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Cast labels — constructed headers, N-party, collision safety
# ---------------------------------------------------------------------------


def test_cast_labels_in_constructed_headers():
    _, grammar = _grammar(GEMMA_STRICT_TEMPLATE, "gemma3")
    got = render_scene(
        grammar,
        [
            SceneTurn(seat="user", text="ahoy", label="captain"),
            SceneTurn(seat="assistant", text="arr", label="pirate"),
            SceneTurn(seat="user", text="*rustles*", label="deer"),
        ],
        gen_seat="assistant",
        gen_label="pirate",
    )
    assert got == (
        "<bos>"
        "<start_of_turn>captain\nahoy<end_of_turn>\n"
        "<start_of_turn>pirate\narr<end_of_turn>\n"
        "<start_of_turn>deer\n*rustles*<end_of_turn>\n"
        "<start_of_turn>pirate\n"
    )


def test_label_collision_with_standard_label_is_safe():
    """A user turn labeled ``model`` cannot corrupt turn attribution.

    The splice path's ``_splice_occurrences`` had an ordering hazard here (a
    substituted label creating fake occurrences of the other seat's
    pattern); construction places every label positionally, so the collision
    class does not exist.
    """
    _, grammar = _grammar(GEMMA_STRICT_TEMPLATE, "gemma3")
    got = render_scene(
        grammar,
        [
            SceneTurn(seat="user", text="one", label="model"),
            SceneTurn(seat="assistant", text="two", label="user"),
        ],
    )
    assert got == (
        "<bos>"
        "<start_of_turn>model\none<end_of_turn>\n"
        "<start_of_turn>user\ntwo<end_of_turn>\n"
    )


def test_label_slug_validation():
    _, grammar = _grammar(QWEN_TEMPLATE, "qwen3")
    with pytest.raises(SaklasError):
        render_scene(
            grammar, [SceneTurn(seat="user", text="x", label="Bad Label")]
        )


def test_label_deslug():
    """Underscores de-slug to spaces, mirroring the splice path."""
    _, grammar = _grammar(QWEN_TEMPLATE, "qwen3")
    got = render_scene(
        grammar,
        [SceneTurn(seat="user", text="aye", label="first_mate")],
    )
    assert "<|im_start|>first mate\naye<|im_end|>\n" in got


# ---------------------------------------------------------------------------
# Generation prompts on either seat
# ---------------------------------------------------------------------------


def test_generate_into_user_seat():
    _, grammar = _grammar(LLAMA_TEMPLATE, "llama")
    got = render_scene(
        grammar,
        [
            SceneTurn(seat="user", text="hi"),
            SceneTurn(seat="assistant", text="hello"),
        ],
        gen_seat="user",
        gen_label="deer",
    )
    assert got.endswith("<|start_header_id|>deer<|end_header_id|>\n\n")


def test_gen_extra_rides_assistant_seat_only():
    """A generation scaffold (qwen3 empty-think insert) lands on the
    assistant gen header and never on a user-seat one."""
    _, grammar = _grammar(QWEN_STRIP_TEMPLATE, "qwen3")
    assert grammar.gen_extra == "<think>\n\n</think>\n\n"
    asst = render_scene(
        grammar, [SceneTurn(seat="user", text="q")], gen_seat="assistant"
    )
    assert asst.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    user = render_scene(
        grammar, [SceneTurn(seat="assistant", text="a")], gen_seat="user"
    )
    assert user.endswith("<|im_start|>user\n")


# ---------------------------------------------------------------------------
# System conventions
# ---------------------------------------------------------------------------


def test_system_real_turn():
    _, grammar = _grammar(QWEN_TEMPLATE, "qwen3")
    got = render_scene(
        grammar,
        [SceneTurn(seat="user", text="hi")],
        system="be terse",
    )
    assert got.startswith("<|im_start|>system\nbe terse<|im_end|>\n")


def test_system_unsupported_family_degrades():
    """gemma-2 shape: the template refuses the system role; scene mode
    survives without system support and raises only on ``system=``."""
    template = (
        "<bos>"
        "{% for m in messages %}"
        "{% if m['role'] == 'system' %}"
        "{{ raise_exception('System role not supported') }}"
        "{% endif %}"
        "{% set role = 'model' if m['role'] == 'assistant' else m['role'] %}"
        "<start_of_turn>{{ role }}\n{{ m['content'] }}<end_of_turn>\n"
        "{% endfor %}"
        "{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}"
    )
    tok = FakeTokenizer(template)
    grammar = extract_turn_grammar(tok, "gemma2")
    assert not grammar.system_supported
    validate_turn_grammar(grammar, tok)
    turns = [SceneTurn(seat="user", text="hi")]
    assert render_scene(grammar, turns).startswith("<bos><start_of_turn>user")
    with pytest.raises(SceneRenderError):
        render_scene(grammar, turns, system="be terse")


def test_system_fold_first_turn_any_seat():
    """Convention 1: fold families prepend system into the first turn,
    whatever its seat — including an assistant-first scene the template
    has no answer for."""
    _, grammar = _grammar(GEMMA_STRICT_TEMPLATE, "gemma3")
    assert grammar.system is None
    assert grammar.system_fold_sep == "\n\n"
    got = render_scene(
        grammar,
        [
            SceneTurn(seat="assistant", text="I begin.", label="narrator"),
            SceneTurn(seat="user", text="Go on."),
        ],
        system="A play in one act.",
    )
    assert got == (
        "<bos>"
        "<start_of_turn>narrator\nA play in one act.\n\nI begin.<end_of_turn>\n"
        "<start_of_turn>user\nGo on.<end_of_turn>\n"
    )


# ---------------------------------------------------------------------------
# Thinking — per-turn input, family-convention history policy
# ---------------------------------------------------------------------------


def test_thinking_requires_delimiters():
    _, grammar = _grammar(QWEN_TEMPLATE, "qwen3")  # no think_delimiters
    with pytest.raises(SceneThinkingUnsupportedError):
        render_scene(
            grammar,
            [SceneTurn(seat="user", text="x", thinking="hmm")],
        )


def test_thinking_strip_family_lasts_one_turn():
    """On a strip family a thinking block renders only while its turn is
    the last before the generation header."""
    tok = FakeTokenizer(QWEN_STRIP_TEMPLATE)
    grammar = extract_turn_grammar(
        tok, "qwen3", think_delimiters=("<think>", "</think>")
    )
    assert grammar.strips_history_thinking

    # Last turn: thinking renders.
    live = render_scene(
        grammar,
        [
            SceneTurn(seat="user", text="q"),
            SceneTurn(seat="user", text="hm", thinking="let me think"),
        ],
        gen_seat="assistant",
    )
    assert "<think>let me think</think>hm" in live

    # Same turn one position earlier: stripped, matching the template's own
    # history convention.
    hist = render_scene(
        grammar,
        [
            SceneTurn(seat="user", text="hm", thinking="let me think"),
            SceneTurn(seat="assistant", text="a"),
        ],
        gen_seat="assistant",
    )
    assert "let me think" not in hist
    assert "<|im_start|>user\nhm<|im_end|>\n" in hist


def test_thinking_non_strip_family_persists():
    tok = FakeTokenizer(QWEN_TEMPLATE)
    grammar = extract_turn_grammar(
        tok, "qwen3", think_delimiters=("<think>", "</think>")
    )
    assert not grammar.strips_history_thinking
    got = render_scene(
        grammar,
        [
            SceneTurn(seat="assistant", text="a", thinking="early"),
            SceneTurn(seat="user", text="q"),
        ],
    )
    assert "<think>early</think>a" in got


# ---------------------------------------------------------------------------
# Raw-marker fallback
# ---------------------------------------------------------------------------


def test_raw_fallback_render():
    got = render_scene_raw(
        [
            SceneTurn(seat="user", text="ahoy", label="captain"),
            SceneTurn(seat="assistant", text="arr", label="pirate"),
            SceneTurn(seat="assistant", text="*rustles*", label="deer"),
        ],
        system="A forest clearing.",
        gen_seat="user",
        gen_label="first_mate",
    )
    assert got == (
        "A forest clearing.\n\n"
        "Captain: ahoy\n\n"
        "Pirate: arr\n\n"
        "Deer: *rustles*\n\n"
        "First mate:"
    )


def test_raw_fallback_default_labels():
    got = render_scene_raw(
        [SceneTurn(seat="user", text="hi")], gen_seat="assistant"
    )
    assert got == "User: hi\n\nAssistant:"
