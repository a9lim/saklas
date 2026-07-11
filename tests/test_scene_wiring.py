"""Wiring tests for the cast model: ``build_chat_input`` through the
stitcher, seat-aware loom nodes, and the WS ``generate_seat`` field.

The render-parity tests are the load-bearing ones: with a validated
grammar supplied, ``build_chat_input`` must produce byte-identical ids to
the legacy chat-template paths on every alternating render (the
extraction/steering baseline contract), while unlocking the shapes the
template refuses (non-alternating seats, user-seat generation prompts).
"""
from __future__ import annotations

from typing import Any, cast

import pytest
import torch

from saklas.core.loom import InvalidNodeOperationError, LoomTree
from saklas.core.generation import build_chat_input as _real_build_chat_input
from saklas.core.scene import (
    SceneRenderError,
    extract_turn_grammar,
    render_scene,
    SceneTurn,
    validate_turn_grammar,
)
from tests.test_role_templates import FakeTokenizer, QWEN_TEMPLATE
from tests.test_scene import GEMMA_STRICT_TEMPLATE


# Qwen3-style enable_thinking switch: the generation prompt carries an
# empty think scaffold only when thinking is disabled.
QWEN_THINK_TEMPLATE = (
    "{% for m in messages %}"
    "<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n"
    "{% if enable_thinking is defined and not enable_thinking %}"
    "<think>\n\n</think>\n\n"
    "{% endif %}"
    "{% endif %}"
)


def build_chat_input(tok: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
    # FakeTokenizer quacks like PreTrainedTokenizerBase for everything
    # build_chat_input touches; the cast keeps pyright out of the way.
    return _real_build_chat_input(cast(Any, tok), *args, **kwargs)


def _tok(template: str) -> FakeTokenizer:
    """FakeTokenizer with ``chat_template`` set — ``build_chat_input``
    branches on the attribute (absent = base model), so wiring tests must
    carry it like a real tokenizer does."""
    tok = FakeTokenizer(template)
    tok.chat_template = template  # type: ignore[attr-defined]
    return tok


def _ids(t: torch.Tensor) -> list[int]:
    return t[0].tolist()


# ---------------------------------------------------------------------------
# build_chat_input × scene grammar
# ---------------------------------------------------------------------------


def test_build_chat_input_scene_parity_plain():
    """scene= must not change any legacy render (alternating, no labels)."""
    tok = _tok(QWEN_TEMPLATE)
    grammar = extract_turn_grammar(tok, "qwen3")
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "again"},
    ]
    for system in (None, "be terse"):
        for gen in (True, False):
            legacy = build_chat_input(
                tok, messages, system, add_generation_prompt=gen,
            )
            scened = build_chat_input(
                tok, messages, system, add_generation_prompt=gen,
                scene=grammar,
            )
            assert _ids(legacy) == _ids(scened)


def test_build_chat_input_scene_parity_labeled():
    """Labeled renders match the splice path byte-for-byte."""
    tok = _tok(QWEN_TEMPLATE)
    grammar = extract_turn_grammar(tok, "qwen3")
    messages = [
        {"role": "user", "content": "ahoy", "label": "captain"},
        {"role": "assistant", "content": "arr", "label": "pirate"},
        {"role": "user", "content": "onward"},
    ]
    legacy = build_chat_input(
        tok, messages, gen_role="pirate", model_type="qwen3",
    )
    scened = build_chat_input(
        tok, messages, gen_role="pirate", model_type="qwen3", scene=grammar,
    )
    assert _ids(legacy) == _ids(scened)


def test_build_chat_input_non_alternating_needs_scene():
    """A seat sequence the template refuses renders only through the scene."""
    tok = _tok(GEMMA_STRICT_TEMPLATE)
    grammar = extract_turn_grammar(tok, "gemma3")
    messages = [
        {"role": "assistant", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"},
    ]
    with pytest.raises(Exception):
        build_chat_input(tok, messages)  # template raises on a/a
    ids = build_chat_input(tok, messages, scene=grammar)
    expected = render_scene(
        grammar,
        [SceneTurn(seat=m["role"], text=m["content"]) for m in messages],  # type: ignore[arg-type]
        gen_seat="assistant",
    )
    assert _ids(ids) == tok(expected, add_special_tokens=False).input_ids


def test_build_chat_input_gen_seat_user():
    tok = _tok(QWEN_TEMPLATE)
    grammar = extract_turn_grammar(tok, "qwen3")
    messages = [{"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"}]
    ids = build_chat_input(
        tok, messages, scene=grammar, gen_seat="user", gen_role="deer",
    )
    expected = render_scene(
        grammar,
        [SceneTurn(seat="user", text="hi"),
         SceneTurn(seat="assistant", text="hello")],
        gen_seat="user", gen_label="deer",
    )
    assert _ids(ids) == tok(expected, add_special_tokens=False).input_ids
    assert expected.endswith("<|im_start|>deer\n")


def test_build_chat_input_gen_seat_user_requires_scene():
    tok = _tok(QWEN_TEMPLATE)
    with pytest.raises(SceneRenderError):
        build_chat_input(
            tok, [{"role": "user", "content": "hi"}], gen_seat="user",
        )


def test_build_chat_input_thinking_gen_variants():
    """The enable_thinking switch selects the right gen appendix."""
    tok = _tok(QWEN_THINK_TEMPLATE)
    grammar = extract_turn_grammar(tok, "qwen3")
    validate_turn_grammar(grammar, tok)
    assert grammar.gen_extra == "<think>\n\n</think>\n\n"
    assert grammar.gen_extra_thinking == ""
    messages = [{"role": "user", "content": "hi"}]
    nothink = build_chat_input(tok, messages, thinking=False, scene=grammar)
    think = build_chat_input(tok, messages, thinking=True, scene=grammar)
    legacy_nothink = build_chat_input(tok, messages, thinking=False)
    legacy_think = build_chat_input(tok, messages, thinking=True)
    assert _ids(nothink) == _ids(legacy_nothink)
    assert _ids(think) == _ids(legacy_think)
    assert _ids(think) != _ids(nothink)


# ---------------------------------------------------------------------------
# Loom: seat-aware generated nodes
# ---------------------------------------------------------------------------


def test_begin_assistant_user_seat():
    tree = LoomTree()
    uid = tree.add_user_turn("hello")
    nid = tree.begin_assistant(uid, seat="user", role_label="deer")
    node = tree.get(nid)
    assert node.role == "user"
    assert node.role_label == "deer"


def test_begin_assistant_rejects_bad_seat():
    tree = LoomTree()
    uid = tree.add_user_turn("hello")
    with pytest.raises(InvalidNodeOperationError):
        tree.begin_assistant(uid, seat="system")


def test_user_seat_sibling_under_user_parent():
    """u/u — a generated user-seat node may hang under a user turn."""
    tree = LoomTree()
    uid = tree.add_user_turn("first")
    nid = tree.begin_assistant(uid, seat="user")
    tree.finalize_assistant(nid, text="second", finish_reason="stop")
    msgs = tree.messages_for(nid)
    assert [m["role"] for m in msgs] == ["user", "user"]


# ---------------------------------------------------------------------------
# WS: generate_seat field
# ---------------------------------------------------------------------------


def test_ws_generate_message_accepts_seat():
    from saklas.server.ws_models import WSGenerateMessage

    msg = WSGenerateMessage(type="generate", generate_seat="user")
    assert msg.generate_seat == "user"
    assert WSGenerateMessage(type="generate").generate_seat is None


def test_ws_generate_message_rejects_bad_seat():
    from pydantic import ValidationError

    from saklas.server.ws_models import WSGenerateMessage

    with pytest.raises(ValidationError):
        WSGenerateMessage(type="generate", generate_seat="narrator")  # type: ignore[arg-type]
