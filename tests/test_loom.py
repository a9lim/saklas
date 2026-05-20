"""LoomTree unit tests — data model, mutations, persistence.

Covers the engine-side loom layer in isolation; concurrency interactions
with the gen lock live in ``test_loom_concurrency.py``.  These tests do
not load a model — :class:`LoomTree` is independent of HF state.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from saklas import (
    EventBus,
    InvalidNodeOperationError,
    LoomMutated,
    LoomNode,
    LoomTree,
    Recipe,
    SamplingConfig,
    UnknownNodeError,
    derive_seed_schedule,
)
from saklas.core.loom import TREE_FORMAT_VERSION


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _seed_tree() -> LoomTree:
    """Tree shaped like:

        root
         └── u1 ("hi")
              └── a1 ("hello")

    Returned with active = a1.
    """
    t = LoomTree()
    u1 = t.add_user_turn("hi")
    a1 = t.begin_assistant(u1, recipe=Recipe(steering="0.3 honest"))
    t.finalize_assistant(a1, text="hello", finish_reason="stop")
    return t


# ---------------------------------------------------------------------------
# Basic shape / read API
# ---------------------------------------------------------------------------


def test_fresh_tree_has_synthetic_root():
    t = LoomTree()
    assert t.root_id == t.active_node_id
    assert t.nodes[t.root_id].role == "system"
    assert t.nodes[t.root_id].text == ""
    assert t.children_of[t.root_id] == []


def test_add_user_turn_anchors_under_active():
    t = LoomTree()
    uid = t.add_user_turn("hello")
    assert t.active_node_id == uid
    assert t.nodes[uid].parent_id == t.root_id
    assert t.children_of[t.root_id] == [uid]


def test_begin_and_finalize_assistant():
    t = LoomTree()
    uid = t.add_user_turn("hello")
    aid = t.begin_assistant(uid, recipe=Recipe(steering="0.3 honest"))
    assert t.nodes[aid].role == "assistant"
    assert t.nodes[aid].recipe is not None
    assert t.nodes[aid].recipe.steering == "0.3 honest"
    assert t.active_node_id == aid
    t.finalize_assistant(
        aid, text="hi back",
        aggregate_readings={"angry": 0.1},
        applied_steering="0.3 honest",
        finish_reason="stop",
    )
    assert t.nodes[aid].text == "hi back"
    assert t.nodes[aid].aggregate_readings == {"angry": 0.1}
    assert t.nodes[aid].finish_reason == "stop"


def test_messages_for_active_path():
    t = _seed_tree()
    msgs = t.messages_for()
    assert msgs == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


def test_messages_for_skips_synthetic_root_by_default():
    t = LoomTree()
    msgs = t.messages_for()
    # Root has empty text and role=system; default include_system=False.
    assert msgs == []


def test_messages_for_include_system_true():
    t = LoomTree()
    msgs = t.messages_for(include_system=True)
    assert msgs == [{"role": "system", "content": ""}]


def test_messages_for_explicit_leaf():
    """Path-to-arbitrary-node lets surfaces render off-path branches."""
    t = LoomTree()
    u1 = t.add_user_turn("hi")
    a1 = t.begin_assistant(u1)
    t.finalize_assistant(a1, text="hello")
    # Branch a2 as a sibling of a1 under u1.
    a2 = t.branch(a1, "hi there")
    msgs = t.messages_for(a2)
    assert msgs == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hi there"},
    ]
    # active_path() still tracks the live cursor (now a2 after branch).
    assert t.active_node_id == a2


def test_descendants_dfs():
    t = LoomTree()
    u1 = t.add_user_turn("x")
    a1 = t.begin_assistant(u1)
    t.finalize_assistant(a1, text="A")
    a2 = t.branch(a1, "B")
    ids = {n.id for n in t.descendants(t.root_id)}
    assert ids == {u1, a1, a2}


def test_ancestors_iter():
    t = _seed_tree()
    a1 = t.active_node_id
    chain = list(t.ancestors_of(a1))
    # parent first (the user node), then root.
    assert len(chain) == 2
    assert t.nodes[chain[-1]].role == "system"  # root


def test_unknown_node_raises_404():
    t = LoomTree()
    with pytest.raises(UnknownNodeError) as exc:
        t.get("nope")
    code, _ = exc.value.user_message()
    assert code == 404


# ---------------------------------------------------------------------------
# Core primitive: edit
# ---------------------------------------------------------------------------


def test_edit_in_place_mutates_text_no_new_node():
    t = _seed_tree()
    aid = t.active_node_id
    rev_before = t.rev
    pre_count = len(t.nodes)
    t.edit(aid, "edited reply")
    assert t.nodes[aid].text == "edited reply"
    assert len(t.nodes) == pre_count
    assert t.nodes[aid].edit_count == 1
    assert t.nodes[aid].edited_at is not None
    assert t.rev > rev_before


def test_edit_increments_edit_count():
    t = _seed_tree()
    aid = t.active_node_id
    t.edit(aid, "v1")
    t.edit(aid, "v2")
    t.edit(aid, "v3")
    assert t.nodes[aid].edit_count == 3


def test_edit_root_refused():
    t = LoomTree()
    with pytest.raises(InvalidNodeOperationError):
        t.edit(t.root_id, "anything")


def test_edit_emits_event():
    t = _seed_tree()
    bus = EventBus()
    seen: list = []
    bus.subscribe(seen.append)
    t.attach_events(bus)
    t.edit(t.active_node_id, "edited")
    ops = [e.op for e in seen if isinstance(e, LoomMutated)]
    assert "edit" in ops


# ---------------------------------------------------------------------------
# Core primitive: branch
# ---------------------------------------------------------------------------


def test_branch_creates_sibling_and_preserves_original():
    t = _seed_tree()
    a1 = t.active_node_id
    parent = t.nodes[a1].parent_id
    a2 = t.branch(a1, "alternate reply")
    assert a2 != a1
    assert t.nodes[a2].parent_id == parent
    # Original preserved.
    assert t.nodes[a1].text == "hello"
    # New becomes active.
    assert t.active_node_id == a2
    # Sibling count under the user-parent is now 2.
    assert len(t.children_of[parent]) == 2


def test_branch_from_blank_text():
    t = _seed_tree()
    a1 = t.active_node_id
    a2 = t.branch(a1, "")
    assert t.nodes[a2].text == ""
    assert t.nodes[a2].role == t.nodes[a1].role


def test_branch_from_root_refused():
    t = LoomTree()
    with pytest.raises(InvalidNodeOperationError):
        t.branch(t.root_id, "anything")


def test_branch_make_active_false():
    t = _seed_tree()
    a1 = t.active_node_id
    a2 = t.branch(a1, "shadow", make_active=False)
    assert t.active_node_id == a1
    assert t.nodes[a2].parent_id == t.nodes[a1].parent_id


# ---------------------------------------------------------------------------
# Core primitive: navigate
# ---------------------------------------------------------------------------


def test_navigate_updates_active():
    t = _seed_tree()
    a1 = t.active_node_id
    a2 = t.branch(a1, "alt", make_active=False)
    t.navigate(a2)
    assert t.active_node_id == a2


def test_navigate_unknown_raises():
    t = LoomTree()
    with pytest.raises(UnknownNodeError):
        t.navigate("nope")


def test_navigate_to_self_is_noop():
    t = _seed_tree()
    rev_before = t.rev
    t.navigate(t.active_node_id)
    assert t.rev == rev_before  # no event, no rev bump


# ---------------------------------------------------------------------------
# Core primitive: delete_subtree
# ---------------------------------------------------------------------------


def test_delete_subtree_removes_node_and_descendants():
    t = LoomTree()
    u1 = t.add_user_turn("x")
    a1 = t.begin_assistant(u1)
    t.finalize_assistant(a1, text="A")
    a2 = t.branch(a1, "B", make_active=False)
    # Navigate away from a2's parent subtree.
    t.navigate(a1)
    pre = len(t.nodes)
    n = t.delete_subtree(a2)
    assert n == 1
    assert a2 not in t.nodes
    assert len(t.nodes) == pre - 1


def test_delete_subtree_ancestor_of_active_refused():
    t = _seed_tree()
    # Active is the assistant a1; its user-parent is an ancestor.
    user_parent = t.nodes[t.active_node_id].parent_id
    with pytest.raises(InvalidNodeOperationError):
        t.delete_subtree(user_parent)
    with pytest.raises(InvalidNodeOperationError):
        t.delete_subtree(t.active_node_id)


def test_delete_root_refused():
    t = LoomTree()
    with pytest.raises(InvalidNodeOperationError):
        t.delete_subtree(t.root_id)


def test_delete_subtree_cascades():
    t = LoomTree()
    u1 = t.add_user_turn("x")
    a1 = t.begin_assistant(u1)
    t.finalize_assistant(a1, text="A")
    u2 = t.add_user_turn("y", parent_id=a1)
    a2 = t.begin_assistant(u2)
    t.finalize_assistant(a2, text="B")
    t.navigate(t.root_id)
    # Delete the whole subtree rooted at u1.
    n = t.delete_subtree(u1)
    assert n == 4  # u1, a1, u2, a2
    assert t.children_of[t.root_id] == []


# ---------------------------------------------------------------------------
# Decoration ops
# ---------------------------------------------------------------------------


def test_star_toggles():
    t = _seed_tree()
    aid = t.active_node_id
    assert t.nodes[aid].starred is False
    t.star(aid, True)
    assert t.nodes[aid].starred is True
    t.star(aid, False)
    assert t.nodes[aid].starred is False


def test_annotate_sets_notes():
    t = _seed_tree()
    aid = t.active_node_id
    t.annotate(aid, "looked promising")
    assert t.nodes[aid].notes == "looked promising"


# ---------------------------------------------------------------------------
# Engine-level: reset, rewind
# ---------------------------------------------------------------------------


def test_reset_drops_everything():
    t = _seed_tree()
    old_root = t.root_id
    t.reset()
    # Fresh root; old root is gone.
    assert t.root_id != old_root
    assert t.active_node_id == t.root_id
    assert len(t.nodes) == 1  # synthetic root only


def test_rewind_non_destructive():
    t = _seed_tree()
    aid_before = t.active_node_id
    pre = len(t.nodes)
    t.rewind()
    assert t.active_node_id != aid_before
    # Tree shape preserved — rewound pair lives on as a dead branch.
    assert len(t.nodes) == pre
    # Active is now the root after rewinding past the user→assistant pair.
    assert t.active_node_id == t.root_id


def test_rewind_with_no_history_is_noop():
    t = LoomTree()
    rev_before = t.rev
    t.rewind()
    assert t.rev == rev_before
    assert t.active_node_id == t.root_id


# ---------------------------------------------------------------------------
# Active-node send semantics: user dedup
# ---------------------------------------------------------------------------


def test_add_user_turn_dedups_identical_text():
    t = LoomTree()
    u1 = t.add_user_turn("repeat me")
    a1 = t.begin_assistant(u1)
    t.finalize_assistant(a1, text="A")
    t.navigate(u1.__class__("noop") if False else t.root_id)  # noop branch
    # Re-send the same text from root: should resolve to existing u1.
    u1_again = t.add_user_turn("repeat me")
    assert u1_again == u1


def test_add_user_turn_no_dedup_when_text_differs():
    t = LoomTree()
    u1 = t.add_user_turn("first")
    t.navigate(t.root_id)
    u2 = t.add_user_turn("second")
    assert u1 != u2
    assert len(t.children_of[t.root_id]) == 2


# ---------------------------------------------------------------------------
# D15 — active-node send role table (decision 15 in docs/plans/loom.md)
#
# These tests verify the engine-side reject when the resolved parent for
# a new user turn is itself a user node — the "user turn already waiting
# for an assistant" case from the plan's send-semantics table.  The
# reject lives in ``session._generate_core``, so we exercise it through
# the session entry rather than the bare ``tree.add_user_turn`` (which
# is allowed to land a user-under-user for the auto-regen seed path).
# We monkey-patch the session shape minimally to avoid loading a model.
# ---------------------------------------------------------------------------


def _bind_check_user_send_target(tree: LoomTree):
    """Return a callable bound to a stub session exposing only ``tree``.

    ``SaklasSession._check_user_send_target`` only reaches into
    ``self.tree``; binding the unbound method onto a minimal stub
    sidesteps the model-load cost so we can unit-test D15.
    """
    from saklas.core.session import SaklasSession
    stub = type("S", (), {"tree": tree})()
    return SaklasSession._check_user_send_target.__get__(stub, type(stub))


def test_d15_engine_check_rejects_leaf_user_send():
    """A bare send when active is a leaf user raises per D15."""
    t = LoomTree()
    t.add_user_turn("waiting on assistant")
    # active = u1, a leaf user node
    check = _bind_check_user_send_target(t)
    with pytest.raises(InvalidNodeOperationError, match="cannot send a new user turn"):
        check(None)


def test_d15_engine_check_rejects_interior_user_send():
    """Same check fires for an interior user node."""
    t = LoomTree()
    u1 = t.add_user_turn("hi")
    a1 = t.begin_assistant(u1)
    t.finalize_assistant(a1, text="hello")
    u2 = t.add_user_turn("more")
    a2 = t.begin_assistant(u2)
    t.finalize_assistant(a2, text="world")
    # Navigate back to an interior user node.
    t.navigate(u1)
    check = _bind_check_user_send_target(t)
    with pytest.raises(InvalidNodeOperationError, match="cannot send a new user turn"):
        check(None)


def test_d15_engine_check_rejects_explicit_user_parent():
    """``parent_node_id`` pointing at a user node also raises."""
    t = LoomTree()
    u1 = t.add_user_turn("hi")
    # Active doesn't matter — explicit parent_node_id wins.
    check = _bind_check_user_send_target(t)
    with pytest.raises(InvalidNodeOperationError):
        check(u1)


def test_d15_engine_check_passes_for_assistant_leaf():
    """Sending from a leaf assistant node is the today-flow — no reject."""
    t = _seed_tree()
    assert t.nodes[t.active_node_id].role == "assistant"
    check = _bind_check_user_send_target(t)
    check(None)  # no raise


def test_d15_engine_check_passes_for_root():
    """Sending from the synthetic root (fresh tree) — no reject."""
    t = LoomTree()
    assert t.active_node_id == t.root_id
    check = _bind_check_user_send_target(t)
    check(None)  # no raise


def test_d15_engine_check_passes_for_interior_assistant():
    """Active is an interior assistant (user navigated back) — no reject."""
    t = LoomTree()
    u1 = t.add_user_turn("hi")
    a1 = t.begin_assistant(u1)
    t.finalize_assistant(a1, text="hello")
    u2 = t.add_user_turn("more")
    a2 = t.begin_assistant(u2)
    t.finalize_assistant(a2, text="world")
    t.navigate(a1)
    check = _bind_check_user_send_target(t)
    check(None)  # no raise


def test_d15_engine_check_passes_for_explicit_grandparent():
    """The regen flow: parent_node_id = user's parent (assistant) — no reject."""
    t = LoomTree()
    u1 = t.add_user_turn("hi")
    a1 = t.begin_assistant(u1)
    t.finalize_assistant(a1, text="hello")
    t.add_user_turn("more")
    check = _bind_check_user_send_target(t)
    check(a1)  # u2's parent is a1; regen passes a1 explicitly — no raise


# ---------------------------------------------------------------------------
# Authored turns — session.append_user_turn / append_assistant_turn
# ---------------------------------------------------------------------------
# These exercise the Ctrl+Enter "commit" primitives.  Both methods only
# touch ``self.tree`` and ``self._tokenizer``; binding them onto a minimal
# stub lets us unit-test the contract without loading a model.


def _bind_commit_methods(tree: LoomTree, tokenizer):
    from saklas.core.session import SaklasSession
    stub = type(
        "S", (),
        {
            "tree": tree,
            "_tokenizer": tokenizer,
            # ``append_user_turn`` reaches into ``_check_user_send_target``
            # for the D15 guard; bind the unbound method onto the same
            # stub so the lookup chain resolves.
            "_check_user_send_target":
                SaklasSession._check_user_send_target,
        },
    )()
    return (
        SaklasSession.append_user_turn.__get__(stub, type(stub)),
        SaklasSession.append_assistant_turn.__get__(stub, type(stub)),
    )


def test_append_user_turn_lands_under_root():
    from unittest.mock import MagicMock
    t = LoomTree()
    append_user, _ = _bind_commit_methods(t, MagicMock())
    new_id = append_user(None, "hi")
    assert t.nodes[new_id].role == "user"
    assert t.nodes[new_id].text == "hi"
    assert t.active_node_id == new_id


def test_append_user_turn_refuses_under_user_node():
    """D15 — sending a user turn under a user node is forbidden, same
    rule the normal-send path enforces."""
    from unittest.mock import MagicMock
    t = LoomTree()
    t.add_user_turn("hi")  # active = u1, a user node
    append_user, _ = _bind_commit_methods(t, MagicMock())
    with pytest.raises(InvalidNodeOperationError, match="cannot send a new user turn"):
        append_user(None, "more")


def test_append_user_turn_dedups_same_text_sibling():
    """``add_user_turn``'s dedup applies — re-issuing the same text under
    the same parent returns the existing sibling rather than growing
    the tree."""
    from unittest.mock import MagicMock
    t = LoomTree()
    u1 = t.add_user_turn("hi")
    a1 = t.begin_assistant(u1)
    t.finalize_assistant(a1, text="hello")
    t.navigate(a1)
    append_user, _ = _bind_commit_methods(t, MagicMock())
    u2 = append_user(None, "more")
    t.navigate(a1)
    u2b = append_user(None, "more")
    assert u2b == u2


def test_append_user_turn_empty_text_raises():
    from unittest.mock import MagicMock
    t = LoomTree()
    append_user, _ = _bind_commit_methods(t, MagicMock())
    with pytest.raises(InvalidNodeOperationError, match="non-empty"):
        append_user(None, "")


def test_append_assistant_turn_lands_under_user_with_tokenization():
    """Happy path — an authored assistant turn lands as a sibling under
    the user node, ``raw_token_ids`` is populated by the tokenizer, and
    ``recipe`` stays ``None`` as the implicit authored marker."""
    from unittest.mock import MagicMock
    t = LoomTree()
    u1 = t.add_user_turn("hi")
    tok = MagicMock()
    tok.encode.return_value = [10, 20, 30]
    _, append_assistant = _bind_commit_methods(t, tok)
    new_id = append_assistant(u1, "the reply")
    node = t.nodes[new_id]
    assert node.role == "assistant"
    assert node.text == "the reply"
    assert node.raw_token_ids == [10, 20, 30]
    assert node.recipe is None
    assert node.finish_reason == "stop"
    # Authored turns carry no per-token scores — clearing the empty
    # blobs that ``begin_assistant`` seeded matches the transcript-loaded
    # node shape so renderers treat them the same.
    assert node.tokens is None
    assert node.thinking_tokens is None
    assert t.active_node_id == new_id
    tok.encode.assert_called_once_with("the reply", add_special_tokens=False)


def test_append_assistant_turn_refuses_non_user_parent():
    from unittest.mock import MagicMock
    t = _seed_tree()  # leaf is an assistant
    aid = t.active_node_id
    _, append_assistant = _bind_commit_methods(t, MagicMock())
    with pytest.raises(InvalidNodeOperationError, match="not a user node"):
        append_assistant(aid, "reply")


def test_append_assistant_turn_empty_text_raises():
    from unittest.mock import MagicMock
    t = LoomTree()
    u1 = t.add_user_turn("hi")
    _, append_assistant = _bind_commit_methods(t, MagicMock())
    with pytest.raises(InvalidNodeOperationError, match="non-empty"):
        append_assistant(u1, "")


def test_append_assistant_turn_refuses_empty_tokenization():
    """Whitespace-only text that tokenizes to nothing isn't a turn —
    refuse rather than land a no-content authored node."""
    from unittest.mock import MagicMock
    t = LoomTree()
    u1 = t.add_user_turn("hi")
    tok = MagicMock()
    tok.encode.return_value = []
    _, append_assistant = _bind_commit_methods(t, tok)
    with pytest.raises(InvalidNodeOperationError, match="tokenized to an empty"):
        append_assistant(u1, "   ")


def test_append_assistant_turn_lands_sibling_of_in_flight_assistant():
    """An authored assistant can land alongside an in-flight streaming
    assistant under the same user — they're independent siblings."""
    from unittest.mock import MagicMock
    t = LoomTree()
    u1 = t.add_user_turn("hi")
    streaming = t.begin_assistant(u1)  # in-flight, not finalized
    tok = MagicMock()
    tok.encode.return_value = [99]
    _, append_assistant = _bind_commit_methods(t, tok)
    authored = append_assistant(u1, "manual")
    assert authored != streaming
    assert t.nodes[authored].parent_id == u1
    assert t.nodes[streaming].parent_id == u1


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_to_dict_round_trip(tmp_path: Path):
    t = _seed_tree()
    aid = t.active_node_id
    a2 = t.branch(aid, "alt")
    t.star(aid, True)
    path = tmp_path / "tree.json"
    t.save(path)
    raw = json.loads(path.read_text())
    assert raw["tree_format"] == TREE_FORMAT_VERSION
    # D22 / plan ":600" — header carries saklas_version so future
    # migrations can branch on the originating build.
    import saklas as _saklas
    assert raw["saklas_version"] == _saklas.__version__
    t2 = LoomTree.load(path)
    # Same structure.
    assert t2.root_id == t.root_id
    assert t2.active_node_id == t.active_node_id
    assert set(t2.nodes.keys()) == set(t.nodes.keys())
    assert t2.children_of == t.children_of
    # Per-node fidelity.
    assert t2.nodes[aid].starred is True
    assert t2.nodes[a2].text == "alt"
    assert t2.rev == t.rev


def test_save_load_round_trips_token_sidecar(tmp_path: Path):
    t = _seed_tree()
    aid = t.active_node_id
    t.append_token(
        aid,
        {
            "token_id": 42,
            "text": "hello",
            "logprob": -0.25,
            "top_alts": [{"id": 7, "text": "hi", "logprob": -1.5}],
        },
    )
    t.append_token(
        aid,
        {"token_id": 9, "text": "hmm", "logprob": -0.75},
        thinking=True,
    )

    path = tmp_path / "tree.json"
    t.save(path)
    raw = json.loads(path.read_text())
    assert raw["token_sidecar"] == "tree.tokens.json.gz"
    raw_node = next(n for n in raw["nodes"] if n["id"] == aid)
    assert "tokens" not in raw_node
    assert "thinking_tokens" not in raw_node

    sidecar = path.with_name("tree.tokens.json.gz")
    with gzip.open(sidecar, "rt", encoding="utf-8") as f:
        token_payload = json.load(f)
    assert token_payload["token_sidecar_format"] == 1
    assert token_payload["nodes"][aid]["tokens"][0]["token_id"] == 42

    t2 = LoomTree.load(path)
    assert t2.nodes[aid].tokens == t.nodes[aid].tokens
    assert t2.nodes[aid].thinking_tokens == t.nodes[aid].thinking_tokens


def test_recipe_round_trip():
    r = Recipe(
        steering="0.3 honest + 0.5 calm",
        sampling=SamplingConfig(
            temperature=0.7,
            max_tokens=128,
            seed=42,
            stop=("END",),
            logit_bias={123: -2.5},
            presence_penalty=0.2,
            frequency_penalty=0.1,
            logprobs=3,
            return_hidden=True,
            return_top_k=12,
        ),
        thinking=False,
        seed=42,
        probes=["angry.calm", "honest.deceptive"],
        probe_hashes={"angry.calm": "abc123", "honest.deceptive": "def456"},
    )
    rd = r.to_dict()
    r2 = Recipe.from_dict(rd)
    assert r2.steering == r.steering
    assert r2.thinking == r.thinking
    assert r2.seed == r.seed
    assert r2.probes == r.probes
    assert r2.probe_hashes == r.probe_hashes
    assert r2.sampling is not None
    assert r2.sampling.temperature == 0.7
    assert r2.sampling.max_tokens == 128
    assert r2.sampling.seed == 42
    assert r2.sampling.stop == ("END",)
    assert r2.sampling.logit_bias == {123: -2.5}
    assert r2.sampling.presence_penalty == 0.2
    assert r2.sampling.frequency_penalty == 0.1
    assert r2.sampling.logprobs == 3
    assert r2.sampling.return_hidden is True
    assert r2.sampling.return_top_k == 12


def test_children_order_preserved_through_save(tmp_path: Path):
    t = LoomTree()
    u1 = t.add_user_turn("x")
    a1 = t.begin_assistant(u1)
    t.finalize_assistant(a1, text="A")
    a2 = t.branch(a1, "B", make_active=False)
    a3 = t.branch(a1, "C", make_active=False)
    expected = [a1, a2, a3]
    assert t.children_of[u1] == expected
    path = tmp_path / "tree.json"
    t.save(path)
    t2 = LoomTree.load(path)
    assert t2.children_of[u1] == expected


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


def test_events_fire_on_mutations():
    bus = EventBus()
    seen: list[LoomMutated] = []
    bus.subscribe(lambda e: seen.append(e) if isinstance(e, LoomMutated) else None)
    t = LoomTree(events=bus)
    u1 = t.add_user_turn("x")
    a1 = t.begin_assistant(u1)
    t.finalize_assistant(a1, text="A")
    t.branch(a1, "B")
    t.edit(a1, "A-edited")
    t.star(a1, True)
    t.annotate(a1, "note")
    ops = [e.op for e in seen]
    assert "add_user" in ops
    assert "begin_assistant" in ops
    assert "finalize_assistant" in ops
    assert "branch" in ops
    assert "edit" in ops
    assert "star" in ops
    assert "note" in ops
    # Each mutation bumps rev monotonically.
    revs = [e.rev for e in seen]
    assert revs == sorted(revs)


def test_navigate_event_carries_active_node():
    bus = EventBus()
    seen: list[LoomMutated] = []
    bus.subscribe(lambda e: seen.append(e) if isinstance(e, LoomMutated) else None)
    t = LoomTree(events=bus)
    t.add_user_turn("x")
    seen.clear()
    t.navigate(t.root_id)
    assert seen and seen[0].op == "navigate"
    assert seen[0].active_node_id == t.root_id


# ---------------------------------------------------------------------------
# Seed schedule
# ---------------------------------------------------------------------------


def test_seed_schedule_deterministic():
    a = derive_seed_schedule(42, 4)
    b = derive_seed_schedule(42, 4)
    assert a == b


def test_seed_schedule_sensitive_to_base():
    a = derive_seed_schedule(42, 4)
    b = derive_seed_schedule(43, 4)
    assert a != b


def test_seed_schedule_unique_indices():
    seeds = derive_seed_schedule(42, 16)
    assert len(set(seeds)) == 16


def test_seed_schedule_n1_passes_base_through():
    assert derive_seed_schedule(42, 1) == [42]


def test_seed_schedule_none_base_returns_seed():
    seeds = derive_seed_schedule(None, 1)
    assert len(seeds) == 1
    assert 0 <= seeds[0] < (1 << 31)


def test_seed_schedule_invalid_n_raises():
    import pytest
    with pytest.raises(ValueError):
        derive_seed_schedule(42, 0)


# ---------------------------------------------------------------------------
# Filter predicate
# ---------------------------------------------------------------------------


def test_filter_predicate_returns_ids():
    t = LoomTree()
    u1 = t.add_user_turn("x")
    a1 = t.begin_assistant(u1)
    t.finalize_assistant(a1, text="A")
    a2 = t.branch(a1, "B", make_active=False)
    t.star(a2, True)
    starred = t.filter(lambda n: n.starred)
    assert starred == {a2}


# ---------------------------------------------------------------------------
# LoomNode level
# ---------------------------------------------------------------------------


def test_loom_node_to_from_dict_round_trip():
    n = LoomNode(
        id="n1", parent_id="p1", role="assistant", text="hi",
        aggregate_readings={"calm": 0.5}, starred=True, notes="meh",
        applied_steering="0.3 calm", finish_reason="stop",
        edit_count=2,
    )
    d = n.to_dict()
    n2 = LoomNode.from_dict(d)
    assert n2.id == n.id
    assert n2.parent_id == n.parent_id
    assert n2.role == n.role
    assert n2.text == n.text
    assert n2.aggregate_readings == n.aggregate_readings
    assert n2.starred is True
    assert n2.notes == "meh"
    assert n2.applied_steering == n.applied_steering
    assert n2.finish_reason == n.finish_reason
    assert n2.edit_count == n.edit_count


def test_loom_node_tokens_omitted_by_default():
    n = LoomNode(id="n1", parent_id="p1", role="assistant",
                 tokens=[{"id": 5, "text": "hi"}])
    d = n.to_dict()
    assert "tokens" not in d  # default: omitted
    d2 = n.to_dict(include_tokens=True)
    assert d2["tokens"] == [{"id": 5, "text": "hi"}]


# ---------------------------------------------------------------------------
# Format version guard
# ---------------------------------------------------------------------------


def test_load_refuses_future_format(tmp_path: Path):
    raw = {
        "tree_format": 9999,
        "root_id": "x",
        "active_node_id": "x",
        "nodes": [{"id": "x", "parent_id": None, "role": "system", "text": ""}],
        "children_of": {"x": []},
    }
    path = tmp_path / "tree.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(Exception):
        LoomTree.load(path)
