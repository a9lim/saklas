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
    SceneTurn,
    TurnGrammar,
    extract_turn_grammar,
    render_scene,
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


def test_ws_submit_message_uses_structural_roles():
    from saklas.server.ws_models import WSSubmitMessage

    msg = WSSubmitMessage(
        type="submit",
        text="answer",
        authored_role="assistant",
        generated_role="user",
    )
    assert msg.authored_role == "assistant"
    assert msg.generated_role == "user"


# ---------------------------------------------------------------------------
# Per-seat stop segments (convention 2)
# ---------------------------------------------------------------------------


def _grammar_with_closes(user_close: str, assistant_close: str) -> "TurnGrammar":
    from saklas.core.scene import SeatWrapper, TurnGrammar

    return TurnGrammar(
        model_type="fake",
        prelude="<bos>",
        user=SeatWrapper("<t>", "\n", user_close, "user"),
        assistant=SeatWrapper("<t>", "\n", assistant_close, "assistant"),
        system=None,
        system_fold_sep=None,
        gen_extra="",
    )


class _StopStub:
    def __init__(self, grammar: "TurnGrammar | None") -> None:
        self.scene_grammar = grammar


def _augment(
    grammar: "TurnGrammar | None",
    stop_list: list[str] | None,
    *,
    gen_seat: str = "user",
    raw: bool = False,
) -> list[str] | None:
    from typing import cast as _cast

    from saklas.core.session import SaklasSession

    return SaklasSession._seat_stop_augmentation(
        _cast(SaklasSession, _StopStub(grammar)), stop_list,
        gen_seat=gen_seat, raw=raw,
    )


def test_seat_stop_added_when_closes_differ():
    g = _grammar_with_closes("<|end|>\n", "<|return|>")
    assert _augment(g, None) == ["<|end|>"]
    # Composes with caller stops, no duplicates.
    assert _augment(g, ["xyz"]) == ["xyz", "<|end|>"]
    assert _augment(g, ["<|end|>"]) == ["<|end|>"]


def test_seat_stop_skipped_on_shared_close():
    g = _grammar_with_closes("<end_of_turn>\n", "<end_of_turn>\n")
    # Shared terminator → the EOS union covers it → no stop list at all
    # (None in, None out keeps the no-stop fast path).
    assert _augment(g, None) is None


def test_seat_stop_skipped_for_assistant_raw_and_no_grammar():
    g = _grammar_with_closes("<|end|>", "<|return|>")
    assert _augment(g, None, gen_seat="assistant") is None
    assert _augment(g, None, raw=True) is None
    assert _augment(None, None) is None


# ---------------------------------------------------------------------------
# Committed thinking: message-dict flow, cache key, loom storage, commit gate
# ---------------------------------------------------------------------------


def test_message_thinking_flows_through_build_chat_input():
    """A ``"thinking"`` key on a chat message reaches the stitcher, and
    renders that differ only in thinking never collide in the cache."""
    from tests.test_scene import QWEN_STRIP_TEMPLATE

    tok = _tok(QWEN_STRIP_TEMPLATE)
    grammar = extract_turn_grammar(
        tok, "qwen3", think_delimiters=("<think>", "</think>"),
    )
    base = [
        {"role": "user", "content": "q"},
        {"role": "user", "content": "hm"},
    ]
    thought = [
        {"role": "user", "content": "q"},
        {"role": "user", "content": "hm", "thinking": "let me think"},
    ]
    plain_ids = _ids(build_chat_input(tok, base, scene=grammar))
    think_ids = _ids(build_chat_input(tok, thought, scene=grammar))
    assert plain_ids != think_ids
    # The stitcher's own render of the same turns carries the block.
    expected_text = render_scene(
        grammar,
        [
            SceneTurn(seat="user", text="q"),
            SceneTurn(seat="user", text="hm", thinking="let me think"),
        ],
        gen_seat="assistant",
    )
    assert "<think>let me think</think>" in expected_text
    # Second call with identical inputs hits the cache and stays stable.
    assert _ids(build_chat_input(tok, thought, scene=grammar)) == think_ids


def test_loom_thinking_text_round_trips():
    tree = LoomTree()
    u = tree.add_user_turn("hm", thinking_text="let me think")
    assert tree.nodes[u].thinking_text == "let me think"
    # messages_for carries it only in the labeled (cast-render) shape.
    labeled = tree.messages_for(with_labels=True)
    assert labeled[0]["thinking"] == "let me think"
    plain = tree.messages_for()
    assert "thinking" not in plain[0]
    # finalize_assistant stamps it on generated/authored turns.
    a = tree.begin_assistant(u)
    tree.finalize_assistant(a, text="ok", thinking_text="planned")
    assert tree.nodes[a].thinking_text == "planned"
    # Serialization carries the field.
    d = tree.nodes[u].to_dict()
    assert d["thinking_text"] == "let me think"


def test_commit_thinking_gate():
    """Committed thinking is refused at commit time when the family
    can't render it (no grammar / no think delimiters)."""
    from typing import cast as _cast

    from saklas.core.scene import SceneThinkingUnsupportedError
    from saklas.core.session import SaklasSession
    from tests.test_scene import QWEN_STRIP_TEMPLATE

    class _CommitStub:
        # The real gate, borrowed the way test_loom's
        # _bind_commit_methods borrows session methods.
        _check_thinking_commit = SaklasSession._check_thinking_commit

        def __init__(self, grammar: Any) -> None:
            self.scene_grammar = grammar
            self.tree = LoomTree()

        def _check_user_send_target(self, parent_node_id: Any) -> None:
            return None

    def commit(stub: Any, thinking: str | None) -> str:
        return SaklasSession.append_user_turn(
            _cast(SaklasSession, stub), None, "hm", thinking=thinking,
        )

    # No grammar at all (fallback family).
    with pytest.raises(SceneThinkingUnsupportedError):
        commit(_CommitStub(None), "let me think")

    # Grammar without think delimiters (gemma-shaped).
    tok = _tok(GEMMA_STRICT_TEMPLATE)
    no_think = extract_turn_grammar(tok, "gemma3")
    with pytest.raises(SceneThinkingUnsupportedError):
        commit(_CommitStub(no_think), "let me think")

    # Think-capable grammar: commit lands with the block stored.
    tok2 = _tok(QWEN_STRIP_TEMPLATE)
    think_ok = extract_turn_grammar(
        tok2, "qwen3", think_delimiters=("<think>", "</think>"),
    )
    stub = _CommitStub(think_ok)
    node_id = commit(stub, "let me think")
    assert stub.tree.nodes[node_id].thinking_text == "let me think"
    # thinking=None never consults the grammar (stub with grammar=None).
    none_stub = _CommitStub(None)
    nid = commit(none_stub, None)
    assert none_stub.tree.nodes[nid].thinking_text is None


def test_authored_same_roles_coalesce_before_scene_seating_rules():
    """Same effective roles append in place on every renderer; scene mode
    still permits distinct-label same-seat turns that legacy templates reject."""
    from typing import cast as _cast

    from saklas.core.session import SaklasSession
    from tests.test_scene import QWEN_STRIP_TEMPLATE

    class _WordTok:
        def encode(self, text: str, **_: Any) -> list[int]:
            return [3000 + i for i, _w in enumerate(text.split())]

    class _SeatStub:
        _check_thinking_commit = SaklasSession._check_thinking_commit
        _check_user_send_target = SaklasSession._check_user_send_target

        def __init__(self, grammar: Any) -> None:
            self.scene_grammar = grammar
            self.tree = LoomTree()
            self._tokenizer = _WordTok()

    def as_sess(stub: Any) -> SaklasSession:
        return _cast(SaklasSession, stub)

    tok = _tok(QWEN_STRIP_TEMPLATE)
    grammar = extract_turn_grammar(tok, "qwen3")

    # Matching roles coalesce, regardless of renderer.
    stub = _SeatStub(grammar)
    u1 = SaklasSession.append_user_turn(as_sess(stub), None, "one")
    u2 = SaklasSession.append_user_turn(as_sess(stub), u1, " two")
    assert u2 == u1
    assert stub.tree.nodes[u1].text == "one two"
    a1 = SaklasSession.append_assistant_turn(as_sess(stub), u1, "three")
    a2 = SaklasSession.append_assistant_turn(as_sess(stub), a1, " four")
    assert a2 == a1
    assert stub.tree.nodes[a1].text == "three four"
    root_a = SaklasSession.append_assistant_turn(
        as_sess(stub), stub.tree.root_id, "first",
    )
    assert stub.tree.nodes[root_a].role == "assistant"

    # Distinct labels are distinct messages; scene mode can render the
    # resulting same-seat adjacency.
    u3 = SaklasSession.append_user_turn(
        as_sess(stub), u1, "other", role_label="narrator",
    )
    assert u3 != u1
    assert stub.tree.nodes[u3].parent_id == u1

    # Legacy templates coalesce matching labels but still reject distinct
    # same-seat messages.
    legacy = _SeatStub(None)
    lu = SaklasSession.append_user_turn(as_sess(legacy), None, "one")
    lu2 = SaklasSession.append_user_turn(as_sess(legacy), lu, " two")
    assert lu2 == lu
    with pytest.raises(InvalidNodeOperationError):
        SaklasSession.append_user_turn(
            as_sess(legacy), lu, "two", role_label="narrator",
        )
    la = SaklasSession.append_assistant_turn(as_sess(legacy), lu, "reply")
    la2 = SaklasSession.append_assistant_turn(as_sess(legacy), la, " again")
    assert la2 == la
    with pytest.raises(InvalidNodeOperationError):
        SaklasSession.append_assistant_turn(
            as_sess(legacy), la, "again", role_label="narrator",
        )
