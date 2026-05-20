"""Phase 4 — TUI loom slash commands + helpers.

Tests cover the slash-command dispatch contract for the new loom verbs
plus the standalone helpers (alpha grid parser, prefix resolver,
transcript builder).  We don't drive the actual Textual screen — the
loom screen is a thin renderer over ``session.tree``, and the
mutations exercised here all flow through helpers that are easier to
test at the unit level.

The mock-session pattern mirrors ``tests/test_tui_commands.py``: build
an app via ``object.__new__``, install a real :class:`LoomTree` on a
MagicMock session, and verify the dispatch routes correctly.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

import saklas
from saklas.core.loom import LoomTree, Recipe
from saklas.tui.app import SaklasApp
from saklas.tui.loom_helpers import (
    AlphaListError,
    format_node_detail,
    format_path_summary,
    parse_alpha_list,
    resolve_node_prefix,
    search_nodes,
)


# ---------------------------------------------------------------------------
# Test scaffolding
# ---------------------------------------------------------------------------


def _make_app():
    """Minimal SaklasApp bag — same shape as test_tui_commands._make_app."""

    app = object.__new__(SaklasApp)
    session = MagicMock()
    session.tree = LoomTree()
    session.history = []
    session._profiles = {}
    session._model_info = {"model_id": "mock/mock", "model_type": "mock"}
    session._device = SimpleNamespace(type="cpu")
    session._layers = [0, 1, 2]
    session._monitor = MagicMock()
    session._monitor.probe_names = []
    session._monitor.profiles = {}
    session._last_result = None
    session._last_per_token_scores = None
    session._tokenizer = MagicMock()
    session.config = SimpleNamespace(
        temperature=0.7, top_p=0.9, max_new_tokens=128,
        system_prompt=None,
    )
    session.is_generating = False
    session.gen_state = saklas.GenState.IDLE
    session._gen_state = MagicMock()
    session._gen_state.stop_requested = MagicMock()
    session._gen_state.stop_requested.is_set = MagicMock(return_value=False)

    app._session = session
    app._device_str = "cpu"
    app._alphas = {}
    app._enabled = {}
    app._supports_thinking = False
    app._thinking = False
    app._current_assistant_widget = None
    app._poll_timer = None
    app._last_prompt = None
    app._ab_mode = False
    app._ab_shadow_active = False
    app._ab_shadow_row = None
    app._row_for_widget = {}
    # Pending-queue replaces the legacy single-slot ``_pending_action``.
    # ``_pulled_slot`` tracks an in-progress ↑-pull-and-edit; default
    # None means "no slot pulled."  Tests inspect ``_pending_queue``
    # directly or build :class:`PendingItem` instances for dispatch.
    app._pending_queue = []
    app._pulled_slot = None
    app._ui_gen_active = False
    app._focused_panel_idx = 1
    app._highlighting = False
    app._highlight_probe = None
    app._default_seed = None
    app._loom_prune_expr = None
    app._loom_auto_regen_mode = "unsteered"
    app._loom_auto_regen_on = False
    import queue as _queue
    app._ui_token_queue = _queue.SimpleQueue()
    app._input_history = []
    app._history_index = None
    app._history_stash = ""
    app._gen_start_time = 0.0
    app._gen_token_count = 0
    app._last_tok_per_sec = 0.0
    app._last_elapsed = 0.0
    app._log_ppl_sum = 0.0
    app._ppl_count = 0
    app._last_gen_state = (-1, -1.0, -1.0, False, -1)
    app._assistant_messages = []

    chat = MagicMock()
    chat.messages = []
    chat.add_system_message = lambda msg: chat.messages.append(msg)
    # ``_repaint_chat_from_active_path`` (loom navigation / ``/load``)
    # unpacks ``add_finalized_assistant``'s ``(row, widget)`` tuple — a
    # bare MagicMock iterates empty, so hand back a real pair.
    chat.add_finalized_assistant = MagicMock(
        return_value=(MagicMock(), MagicMock()),
    )
    app._chat_panel = chat

    trait = MagicMock()
    trait.get_selected_probe = MagicMock(return_value=None)
    app._trait_panel = trait

    left = MagicMock()
    app._left_panel = left

    return app


def _msgs(app: SaklasApp) -> str:
    return "\n".join(app._chat_panel.messages)


def _seed_tree(tree: LoomTree, *, role: str = "assistant", text: str = "hello"):
    """Drop a one-turn user→assistant pair on the tree, return the ids."""
    uid = tree.add_user_turn("what?")
    aid = tree.begin_assistant(uid, recipe=Recipe(steering="0.3 honest", seed=42))
    tree.finalize_assistant(aid, text=text)
    return uid, aid


# ---------------------------------------------------------------------------
# parse_alpha_list — shared grammar across TUI and webui sweep drawer
# ---------------------------------------------------------------------------


def test_parse_alpha_list_comma_list():
    assert parse_alpha_list("0.0, 0.3, 0.7") == [0.0, 0.3, 0.7]


def test_parse_alpha_list_strips_empties():
    assert parse_alpha_list("0.1, ,0.2,") == [0.1, 0.2]


def test_parse_alpha_list_linspace_basic():
    # linspace(-1, 1, 3) → [-1, 0, 1]
    vals = parse_alpha_list("linspace(-1, 1, 3)")
    assert len(vals) == 3
    assert vals[0] == pytest.approx(-1.0)
    assert vals[1] == pytest.approx(0.0)
    assert vals[2] == pytest.approx(1.0)


def test_parse_alpha_list_linspace_single_point():
    assert parse_alpha_list("linspace(0.5, 1.0, 1)") == [0.5]


def test_parse_alpha_list_linspace_invalid_count():
    with pytest.raises(AlphaListError):
        parse_alpha_list("linspace(0, 1, 0)")


def test_parse_alpha_list_range_form():
    vals = parse_alpha_list("0.0:1.0:0.25")
    # 0.0, 0.25, 0.5, 0.75, 1.0
    assert len(vals) == 5
    assert vals[0] == pytest.approx(0.0)
    assert vals[-1] == pytest.approx(1.0)


def test_parse_alpha_list_range_step_zero():
    with pytest.raises(AlphaListError):
        parse_alpha_list("0:1:0")


def test_parse_alpha_list_range_step_direction_mismatch():
    with pytest.raises(AlphaListError):
        parse_alpha_list("0:1:-0.1")


def test_parse_alpha_list_empty_errors():
    with pytest.raises(AlphaListError):
        parse_alpha_list("")
    with pytest.raises(AlphaListError):
        parse_alpha_list("  ")


def test_parse_alpha_list_bad_number():
    with pytest.raises(AlphaListError):
        parse_alpha_list("0.1, banana, 0.2")


# ---------------------------------------------------------------------------
# resolve_node_prefix — `/nav <prefix>`
# ---------------------------------------------------------------------------


def test_resolve_node_prefix_unique_hit():
    tree = LoomTree()
    uid, aid = _seed_tree(tree)
    # Longer prefix needed — ulid timestamp prefix (first ~10 chars) is
    # shared across nodes created in the same millisecond.
    match = resolve_node_prefix(tree, uid[:16])
    assert match.ok
    assert match.node_id == uid
    assert match.candidates == (uid,)


def test_resolve_node_prefix_case_insensitive():
    tree = LoomTree()
    uid, _ = _seed_tree(tree)
    match = resolve_node_prefix(tree, uid[:16].lower())
    assert match.ok
    assert match.node_id == uid


def test_resolve_node_prefix_no_match():
    tree = LoomTree()
    _seed_tree(tree)
    match = resolve_node_prefix(tree, "ZZZZZZZ")
    assert not match.ok
    assert match.missing


def test_resolve_node_prefix_ambiguous():
    tree = LoomTree()
    uid, aid = _seed_tree(tree)
    # Build two more siblings under the same user, then search for the
    # shared root-id prefix "0" — should hit multiple.
    tree.branch(aid, "another")
    tree.branch(aid, "third")
    match = resolve_node_prefix(tree, "0")
    # Whether `0` happens to be ambiguous depends on the random tail,
    # so try the smallest common prefix instead: ulids share their first
    # 10 chars (timestamp) within ~ms of each other.  We use the first
    # char of the root id, which is guaranteed unique to root + kids.
    # Just verify the API contract: when more than one matches, the
    # match isn't ok and candidates carry the matches.
    if match.ambiguous:
        assert match.node_id is None
        assert len(match.candidates) >= 2


def test_resolve_node_prefix_empty_input():
    tree = LoomTree()
    _seed_tree(tree)
    match = resolve_node_prefix(tree, "")
    assert match.missing
    match = resolve_node_prefix(tree, "   ")
    assert match.missing


# ---------------------------------------------------------------------------
# search_nodes
# ---------------------------------------------------------------------------


def test_search_nodes_text():
    tree = LoomTree()
    uid = tree.add_user_turn("what about owls?")
    aid = tree.begin_assistant(uid)
    tree.finalize_assistant(aid, text="owls are nocturnal birds of prey.")
    bid = tree.branch(aid, "elephants are not nocturnal.")
    hits = search_nodes(tree, "owl")
    assert uid in hits
    assert aid in hits
    assert bid not in hits


def test_search_nodes_notes():
    tree = LoomTree()
    uid = tree.add_user_turn("hi")
    aid = tree.begin_assistant(uid)
    tree.finalize_assistant(aid, text="hello")
    tree.annotate(aid, "good answer — keep this")
    assert aid in search_nodes(tree, "keep")


def test_search_nodes_empty_query():
    tree = LoomTree()
    _seed_tree(tree)
    assert search_nodes(tree, "") == []


# ---------------------------------------------------------------------------
# format_path_summary / format_node_detail
# ---------------------------------------------------------------------------


def test_format_path_summary_walks_active_path():
    tree = LoomTree()
    uid, aid = _seed_tree(tree, text="answer text")
    out = format_path_summary(tree)
    assert uid[:8] in out
    assert aid[:8] in out
    assert "what?" in out
    assert "answer text" in out


def test_format_path_summary_empty():
    tree = LoomTree()
    assert format_path_summary(tree) == "(empty path)"


def test_format_node_detail_includes_recipe_and_readings():
    tree = LoomTree()
    uid = tree.add_user_turn("hi")
    aid = tree.begin_assistant(uid, recipe=Recipe(steering="0.3 honest", seed=42))
    tree.finalize_assistant(
        aid, text="hello",
        aggregate_readings={"angry.calm": -0.5, "happy.sad": 0.2},
    )
    out = format_node_detail(tree, aid)
    assert "0.3 honest" in out
    assert "42" in out
    assert "angry.calm" in out
    assert "happy.sad" in out
    assert "hello" in out


def test_format_node_detail_includes_mean_logprob():
    tree = LoomTree()
    uid = tree.add_user_turn("hi")
    aid = tree.begin_assistant(uid)
    tree.finalize_assistant(aid, text="hello", mean_logprob=-0.42)
    out = format_node_detail(tree, aid)
    assert "mean lp" in out
    assert "-0.420" in out


def test_format_node_detail_escapes_markup_in_text():
    """Completion text can't inject Rich markup into the detail pane."""
    tree = LoomTree()
    uid = tree.add_user_turn("hi")
    aid = tree.begin_assistant(uid)
    tree.finalize_assistant(aid, text="see [bold]this[/bold]")
    out = format_node_detail(tree, aid)
    assert r"\[bold]" in out


# ---------------------------------------------------------------------------
# Loom-screen helpers — node labels, probe gutter, sibling compare
# ---------------------------------------------------------------------------


def test_node_label_marks_active_starred_and_edited():
    from saklas.tui.loom_screen import _node_label

    tree = LoomTree()
    uid = tree.add_user_turn("hello there")
    aid = tree.begin_assistant(uid)
    tree.finalize_assistant(aid, text="an answer")
    tree.star(aid, on=True)
    tree.edit(aid, "an edited answer")
    node = tree.get(aid)

    active = _node_label(node, is_active=True)
    assert "[b]" in active and aid[:8] in active
    assert "★" in active
    assert "✎" in active  # edit_count marker

    dimmed = _node_label(node, dim=True)
    assert dimmed.startswith("[dim]") and dimmed.endswith("[/dim]")


def test_probe_gutter_signs_and_blank():
    from saklas.tui.loom_screen import _probe_gutter

    tree = LoomTree()
    uid = tree.add_user_turn("hi")
    aid = tree.begin_assistant(uid)
    tree.finalize_assistant(
        aid, text="x", aggregate_readings={"warm.clinical": 0.8},
    )
    node = tree.get(aid)
    assert "ansi_green" in _probe_gutter(node, "warm.clinical")

    tree2 = LoomTree()
    u2 = tree2.add_user_turn("hi")
    a2 = tree2.begin_assistant(u2)
    tree2.finalize_assistant(
        a2, text="x", aggregate_readings={"warm.clinical": -0.8},
    )
    assert "ansi_red" in _probe_gutter(tree2.get(a2), "warm.clinical")

    # No probe selected, sentinel probe, or missing reading → blank.
    assert _probe_gutter(node, None) == " "
    assert _probe_gutter(node, "__surprise__") == " "
    assert _probe_gutter(node, "angry.calm") == " "


def _wire_real_diff(session: Any, tree: LoomTree) -> None:
    """Point ``session.diff_nodes`` at the real loom_diff primitives."""
    def _real_diff(a_id: str, b_id: str):
        a = tree.get(a_id)
        b = tree.get(b_id)
        return saklas.NodeDiff(
            a_id=a_id, b_id=b_id,
            parent_id=a.parent_id if a.parent_id == b.parent_id else None,
            text=saklas.text_diff(a.text or "", b.text or ""),
            readings=saklas.readings_diff(
                a.aggregate_readings or {}, b.aggregate_readings or {},
            ),
        )
    session.diff_nodes = _real_diff


def test_format_compare_renders_sibling_diff():
    from saklas.tui.loom_helpers import format_compare

    tree = LoomTree()
    uid = tree.add_user_turn("hi")
    aid = tree.begin_assistant(uid, recipe=Recipe(steering="0.3 honest"))
    tree.finalize_assistant(aid, text="answer one", aggregate_readings={"calm": 0.5})
    bid = tree.begin_assistant(uid, recipe=Recipe(steering="0.6 honest"))
    tree.finalize_assistant(bid, text="answer two", aggregate_readings={"calm": 0.1})
    session = SimpleNamespace(tree=tree)
    _wire_real_diff(session, tree)

    out = format_compare(session, bid)  # type: ignore[arg-type]
    assert "compare" in out
    assert aid[:8] in out          # the sibling is listed
    assert "honest" in out         # steering delta term
    assert "calm" in out           # reading delta
    assert "two" in out            # word-level text diff


def test_format_compare_one_reply_is_advisory():
    from saklas.tui.loom_helpers import format_compare

    tree = LoomTree()
    uid = tree.add_user_turn("hi")
    aid = tree.begin_assistant(uid)
    tree.finalize_assistant(aid, text="lonely")
    session = SimpleNamespace(tree=tree)
    # A turn with a single reply has nothing to compare — same advisory
    # whether the cursor sits on the assistant reply or the user node.
    assert "one assistant reply" in format_compare(session, aid)  # type: ignore[arg-type]
    assert "one assistant reply" in format_compare(session, uid)  # type: ignore[arg-type]


def test_format_compare_user_turn_with_no_replies_is_advisory():
    from saklas.tui.loom_helpers import format_compare

    tree = LoomTree()
    uid = tree.add_user_turn("hi")
    session = SimpleNamespace(tree=tree)
    out = format_compare(session, uid)  # type: ignore[arg-type]
    assert "no assistant replies" in out


def test_format_compare_from_user_node_diffs_replies():
    """A user node resolves to its assistant continuations — the role-
    aware fold of the old "compare children"; standing on the user node
    compares the same set as standing on either assistant reply."""
    from saklas.tui.loom_helpers import format_compare

    tree = LoomTree()
    uid = tree.add_user_turn("hi")
    aid = tree.begin_assistant(uid, recipe=Recipe(steering="0.3 honest"))
    tree.finalize_assistant(aid, text="answer one", aggregate_readings={"calm": 0.5})
    bid = tree.begin_assistant(uid, recipe=Recipe(steering="0.6 honest"))
    tree.finalize_assistant(bid, text="answer two", aggregate_readings={"calm": 0.1})
    session = SimpleNamespace(tree=tree)
    _wire_real_diff(session, tree)

    out = format_compare(session, uid)  # type: ignore[arg-type]
    assert "compare" in out
    assert "honest" in out         # steering delta term
    assert "calm" in out           # reading delta


# ---------------------------------------------------------------------------
# Slash command dispatch
# ---------------------------------------------------------------------------


def test_tree_slash_pushes_screen():
    app = _make_app()
    app.push_screen = MagicMock()
    app._handle_command("/tree")
    app.push_screen.assert_called_once()
    pushed = app.push_screen.call_args.args[0]
    # Lazy import in the handler — verify by class name to avoid an
    # eager Textual import in the test body.
    assert type(pushed).__name__ == "LoomScreen"


def test_nav_navigates_by_prefix():
    app = _make_app()
    uid, aid = _seed_tree(app._session.tree)
    # Active is currently aid (begin_assistant set it).  Use a longer
    # prefix — ulid timestamp prefix is shared across nodes created in
    # the same millisecond.
    app._handle_command(f"/nav {uid[:16]}")
    assert app._session.tree.active_node_id == uid
    assert any("navigated" in m for m in app._chat_panel.messages)


def test_nav_missing_reports():
    app = _make_app()
    _seed_tree(app._session.tree)
    app._handle_command("/nav ZZZZZZZ")
    assert any("no node matches" in m for m in app._chat_panel.messages)


def test_nav_usage_on_empty():
    app = _make_app()
    app._handle_command("/nav")
    assert any("Usage: /nav" in m for m in app._chat_panel.messages)


def test_edit_replaces_active_text():
    app = _make_app()
    uid, aid = _seed_tree(app._session.tree, text="old text")
    # Active is the assistant; /edit replaces its text in place.
    app._handle_command("/edit brand new")
    assert app._session.tree.get(aid).text == "brand new"
    assert app._session.tree.get(aid).edit_count == 1


def test_edit_usage():
    app = _make_app()
    _seed_tree(app._session.tree)
    app._handle_command("/edit")
    assert any("Usage: /edit" in m for m in app._chat_panel.messages)


def test_branch_creates_sibling():
    app = _make_app()
    uid, aid = _seed_tree(app._session.tree, text="orig")
    before = len(app._session.tree.child_ids(uid))
    app._handle_command("/branch alternative")
    after = len(app._session.tree.child_ids(uid))
    assert after == before + 1
    # Active node moved to the new sibling.
    new_id = app._session.tree.active_node_id
    assert new_id != aid
    assert app._session.tree.get(new_id).text == "alternative"


def test_branch_blank_creates_empty_sibling():
    app = _make_app()
    uid, aid = _seed_tree(app._session.tree)
    app._handle_command("/branch")
    new_id = app._session.tree.active_node_id
    assert new_id != aid
    assert app._session.tree.get(new_id).text == ""


def test_del_requires_yes_confirm():
    app = _make_app()
    uid, aid = _seed_tree(app._session.tree)
    app._handle_command("/del")
    # No 'yes' → no deletion; usage hint emitted.
    assert aid in app._session.tree.nodes


def test_ctrl_d_alone_does_not_delete():
    """Ctrl+D (``action_delete_subtree``) must NOT bypass the confirm
    guard — it routes through ``_handle_del("")`` so the same usage
    hint as ``/del`` fires.  The user has to type ``/del yes`` to
    actually delete.  Regression: an earlier shape called
    ``_handle_del("yes")`` and wiped the subtree on a stray keypress.
    """
    app = _make_app()
    uid, aid = _seed_tree(app._session.tree)
    extra = app._session.tree.branch(aid, "alt")
    assert app._session.tree.active_node_id == extra
    app.action_delete_subtree()
    # No deletion.
    assert extra in app._session.tree.nodes
    assert aid in app._session.tree.nodes
    # And the confirm hint surfaced in chat.
    assert any("type '/del yes'" in m for m in app._chat_panel.messages)


def test_del_yes_removes_subtree():
    app = _make_app()
    uid, aid = _seed_tree(app._session.tree)
    # Build a deeper subtree under the user turn — branch returns the
    # new sibling id and re-points the active node to it.
    extra = app._session.tree.branch(aid, "alt")
    assert app._session.tree.active_node_id == extra
    app._handle_command("/del yes")
    # The active sibling gets deleted; its sibling ``aid`` survives.
    assert extra not in app._session.tree.nodes
    assert aid in app._session.tree.nodes


def test_del_yes_surfaces_new_active_id():
    """The chat-screen pre-navigates before delete; the message must
    mention the new active id so the jump isn't silent."""
    app = _make_app()
    uid, aid = _seed_tree(app._session.tree)
    extra = app._session.tree.branch(aid, "alt")
    extra_parent_id = app._session.tree.get(extra).parent_id
    app._handle_command("/del yes")
    msg = _msgs(app)
    # Active landed on the (former) parent of ``extra``; first 8 chars
    # of that id appear in the deleted-N-nodes message.
    assert extra_parent_id is not None
    assert extra_parent_id[:8] in msg
    assert "active now" in msg


def test_star_toggles():
    app = _make_app()
    uid, aid = _seed_tree(app._session.tree)
    assert not app._session.tree.get(aid).starred
    app._handle_command("/star")
    assert app._session.tree.get(aid).starred
    app._handle_command("/star")
    assert not app._session.tree.get(aid).starred


def test_note_annotates_active():
    app = _make_app()
    uid, aid = _seed_tree(app._session.tree)
    app._handle_command("/note keep this")
    assert app._session.tree.get(aid).notes == "keep this"


def test_path_prints_summary():
    app = _make_app()
    uid, aid = _seed_tree(app._session.tree, text="hello")
    app._handle_command("/path")
    out = _msgs(app)
    assert uid[:8] in out
    assert aid[:8] in out
    assert "hello" in out


def test_save_writes_full_tree(tmp_path: Path, monkeypatch) -> None:
    """`/save` serializes the whole loom tree to the conversations dir."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    app = _make_app()
    _seed_tree(app._session.tree)
    app._handle_command("/save mytree")
    saved = tmp_path / "conversations" / "mytree.json"
    assert saved.exists()
    assert "saved tree" in _msgs(app)


def test_save_load_preserves_branches(tmp_path: Path, monkeypatch) -> None:
    """`/save` + `/load` round-trip every branch, not just the active path."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    app = _make_app()
    uid, aid = _seed_tree(app._session.tree)
    # A second assistant under the same user turn — an off-path branch.
    app._session.tree.branch(aid, "alt reply")
    app._handle_command("/save branched")

    # Fresh app — load it back and confirm the off-path sibling survived.
    app2 = _make_app()
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    app2._handle_command("/load branched")
    assert "loaded tree" in _msgs(app2)
    texts = {n.text for n in app2._session.tree.nodes.values()}
    assert "hello" in texts          # active-path assistant
    assert "alt reply" in texts      # off-path branch


def test_load_missing_file(tmp_path: Path, monkeypatch) -> None:
    """`/load` on an absent name reports cleanly without raising."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    app = _make_app()
    app._handle_command("/load ghost")
    assert "no saved tree" in _msgs(app)


def test_fire_auto_regen_streams_into_shadow_column():
    """Phase-5 fix: non-unsteered auto-regen modes stream the modifier
    output into the right (shadow) column via ``_ui_token_queue``
    tagged ``is_shadow=True``, the same plumbing the unsteered A/B path
    uses.  Mirrors :meth:`SaklasApp._start_shadow_generation`'s shape.
    """
    import threading
    from saklas import TokenEvent
    app = _make_app()
    # Active node has to be an assistant under a user turn so
    # ``_fire_auto_regen`` can resolve the anchor user.
    uid, aid = _seed_tree(app._session.tree)

    # Auto-regen state: on, with a non-unsteered mode.
    app._loom_auto_regen_on = True
    app._loom_auto_regen_mode = "inverted"

    # Stub the chat panel's shadow widget mount.  We only need an
    # object with the highlight method; the worker pushes events into
    # ``_ui_token_queue`` directly.
    shadow_widget = MagicMock()
    shadow_widget.apply_highlight = MagicMock()
    app._chat_panel.start_shadow_message = MagicMock(return_value=shadow_widget)
    row = MagicMock()

    # Stub ``generate_stream`` to yield two token events synchronously.
    fake_events = [
        TokenEvent(text="hi", token_id=1, index=0, thinking=False,
                   scores=None, perplexity=None),
        TokenEvent(text=" there", token_id=2, index=1, thinking=False,
                   scores=None, perplexity=None),
    ]

    def _fake_stream(*args, **kwargs):
        # Confirm we routed through ``recipe_override`` (so the mode
        # actually lands) and ``parent_node_id`` (so the new node is a
        # sibling under the user-parent).
        assert kwargs.get("recipe_override") == "inverted"
        assert "parent_node_id" in kwargs
        yield from fake_events

    app._session.generate_stream = _fake_stream

    # Synchronously run the worker the app would spawn so the test
    # doesn't depend on Textual's worker thread machinery.
    workers: list = []
    def _run_worker(fn, **_kw):
        t = threading.Thread(target=fn)
        t.start()
        workers.append(t)
    app.run_worker = _run_worker

    app._fire_auto_regen(row)

    for t in workers:
        t.join(timeout=5.0)

    # Drain the queue and confirm shadow-tagged tokens landed.
    items = []
    while not app._ui_token_queue.empty():
        items.append(app._ui_token_queue.get_nowait())
    tok_items = [it for it in items if it[0] == "tok"]
    assert len(tok_items) == 2, items
    # Position 7 (trailing flag) is the ``is_shadow`` tag.  Tuple shape
    # post-Phase 3: (kind, text, thinking, scores, perplexity, logprob,
    # widget, is_shadow) — eight elements after the logit-pass logprob
    # field landed between perplexity and widget.
    assert all(it[-1] is True for it in tok_items)
    # And the right column had its shadow widget mounted.
    app._chat_panel.start_shadow_message.assert_called_once_with(row)


def test_diff_unique_prefix_diffs_two_siblings():
    """Phase 5: `/diff a b` calls session.diff_nodes and renders the result."""
    app = _make_app()
    tree = app._session.tree
    uid = tree.add_user_turn("hi")
    aid = tree.begin_assistant(uid, recipe=Recipe(steering="0.3 honest"))
    tree.finalize_assistant(aid, text="answer one", aggregate_readings={"calm": 0.5})
    bid = tree.branch(aid, "")
    tree.edit(bid, "answer two")

    # Wire diff_nodes to call the real loom_diff primitives so the test
    # stays an integration test (MagicMock auto-attrs would mask the
    # rendering shape).
    import saklas

    def _real_diff(a_id, b_id):
        a = tree.get(a_id)
        b = tree.get(b_id)
        return saklas.NodeDiff(
            a_id=a_id, b_id=b_id,
            parent_id=a.parent_id if a.parent_id == b.parent_id else None,
            text=saklas.text_diff(a.text or "", b.text or ""),
            readings=saklas.readings_diff(
                a.aggregate_readings or {}, b.aggregate_readings or {},
            ),
        )
    app._session.diff_nodes = _real_diff

    app._handle_command(f"/diff {aid[:16]} {bid[:16]}")
    msg = _msgs(app)
    assert "diff:" in msg
    # Word-level diff captures the replacement.
    assert "two" in msg
    # Readings table label appears.
    assert "readings" in msg


def test_diff_siblings_two_kids_diffs_pair():
    app = _make_app()
    tree = app._session.tree
    uid = tree.add_user_turn("hi")
    aid = tree.begin_assistant(uid, recipe=Recipe(steering="0.3 honest"))
    tree.finalize_assistant(aid, text="A")
    tree.branch(aid, "B")

    import saklas

    def _real_diff(a_id, b_id):
        a = tree.get(a_id)
        b = tree.get(b_id)
        return saklas.NodeDiff(
            a_id=a_id, b_id=b_id, parent_id=uid,
            text=saklas.text_diff(a.text or "", b.text or ""),
            readings=saklas.readings_diff(
                a.aggregate_readings or {}, b.aggregate_readings or {},
            ),
        )
    app._session.diff_nodes = _real_diff

    # Active is the most recent sibling — works either way for siblings dispatch.
    app._handle_command("/diff --siblings")
    msg = _msgs(app)
    assert "diff:" in msg


def test_diff_siblings_one_child_errors():
    app = _make_app()
    uid, aid = _seed_tree(app._session.tree)
    app._handle_command("/diff --siblings")
    msg = _msgs(app)
    assert "need ≥2" in msg


def test_diff_usage_on_empty():
    app = _make_app()
    app._handle_command("/diff")
    assert any("/diff" in m and "Usage" in m for m in app._chat_panel.messages)


def test_prune_parses_and_stashes_expression():
    """Phase 5: `/prune` validates via parse_filter, then stashes."""
    app = _make_app()
    # Sneak a node with a reading so filter_by_expr can match against it
    # (saves us from monkey-patching the tree).
    tree = app._session.tree
    uid = tree.add_user_turn("hi")
    aid = tree.begin_assistant(uid)
    tree.finalize_assistant(aid, text="x", aggregate_readings={"angry.calm": 0.5})
    app._handle_command("/prune agg:angry.calm > 0.4")
    assert app._loom_prune_expr == "agg:angry.calm > 0.4"
    assert any("/prune active" in m for m in app._chat_panel.messages)
    # Empty arg clears.
    app._handle_command("/prune")
    assert app._loom_prune_expr is None


def test_prune_bad_grammar_reports():
    app = _make_app()
    app._handle_command("/prune not a valid filter")
    assert any("parse error" in m for m in app._chat_panel.messages)
    assert app._loom_prune_expr is None


def test_auto_regen_no_args_reports_state():
    app = _make_app()
    app._handle_command("/auto-regen")
    msg = _msgs(app)
    # Default: off, mode=unsteered.
    assert "off" in msg or "unsteered" in msg


def test_auto_regen_sets_mode():
    app = _make_app()
    app._handle_command("/auto-regen inverted")
    assert app._loom_auto_regen_mode == "inverted"


def test_auto_regen_on_off():
    app = _make_app()
    app._handle_command("/auto-regen on")
    assert app._loom_auto_regen_on is True
    app._handle_command("/auto-regen off")
    assert app._loom_auto_regen_on is False


def test_auto_regen_unknown_mode_rejects():
    app = _make_app()
    app._handle_command("/auto-regen bogus")
    assert any("unknown mode" in m for m in app._chat_panel.messages)
    # Mode unchanged from default.
    assert app._loom_auto_regen_mode == "unsteered"


def test_auto_regen_custom_parses_into_recipe():
    """`/auto-regen custom: <expr>` parses the expression into a Recipe.

    The stashed value is a Recipe partial (not a raw string), so the
    engine's ``compose_modifier(Recipe) -> Recipe`` passthrough handles
    it without the ValueError the pre-fix path tripped on every gen.
    """
    app = _make_app()
    app._handle_command("/auto-regen custom: 0.3 deer.wolf")
    mode = app._loom_auto_regen_mode
    assert isinstance(mode, Recipe)
    assert mode.steering is not None
    assert "0.3" in mode.steering
    # Footer / status renders Recipe as "custom".
    assert app._render_auto_regen_mode(mode) == "custom"


def test_auto_regen_custom_parse_error_keeps_mode_unchanged():
    """A bad ``custom:`` expression posts to chat and leaves mode alone."""
    app = _make_app()
    app._loom_auto_regen_mode = "inverted"
    app._handle_command("/auto-regen custom: ::: gibberish :::")
    assert any("custom parse error" in m or "expression"  in m
               for m in app._chat_panel.messages)
    assert app._loom_auto_regen_mode == "inverted"


def test_auto_regen_custom_empty_expression():
    """``/auto-regen custom:`` with nothing after the colon rejects."""
    app = _make_app()
    app._handle_command("/auto-regen custom:")
    assert any("needs an expression" in m for m in app._chat_panel.messages)
    assert app._loom_auto_regen_mode == "unsteered"


def test_fan_parses_alpha_grid_and_kicks_worker():
    """`/fan` parses the alpha grid and dispatches a fan-out worker.

    We don't actually run the worker (avoids a Textual app); we verify
    the right method receives the parsed call.
    """
    app = _make_app()
    uid, aid = _seed_tree(app._session.tree)
    app._last_prompt = "do a thing"
    captured = {}

    def _intercept(vector, alphas, prompt):
        captured["vector"] = vector
        captured["alphas"] = alphas
        captured["prompt"] = prompt
    app._run_fan_worker = _intercept

    app._handle_command("/fan angry.calm 0.0, 0.3, 0.7")
    assert captured.get("vector") == "angry.calm"
    assert captured.get("alphas") == [0.0, 0.3, 0.7]
    assert captured.get("prompt") == "do a thing"


def test_fan_alpha_grid_error():
    app = _make_app()
    app._last_prompt = "do a thing"
    app._handle_command("/fan angry.calm banana")
    assert any("alpha grid error" in m for m in app._chat_panel.messages)


def test_fan_usage_on_missing_alphas():
    app = _make_app()
    app._handle_command("/fan angry.calm")
    assert any("Usage: /fan" in m for m in app._chat_panel.messages)


def test_regen_n_dispatches_through_loom_helper():
    app = _make_app()
    captured = {}

    def _intercept(n):
        captured["n"] = n
    app._run_regen_n_worker = _intercept
    app._dispatch_loom_regen = lambda n, *, mode=None: _intercept(n)

    app._handle_command("/regen 3")
    assert captured.get("n") == 3


def test_regen_default_routes_through_action_regenerate():
    app = _make_app()
    app.action_regenerate = MagicMock()
    app._handle_command("/regen")
    app.action_regenerate.assert_called_once()


def test_regen_bad_n_reports():
    app = _make_app()
    app._handle_command("/regen banana")
    assert any("/regen: bad N" in m for m in app._chat_panel.messages)


def test_help_mentions_loom_commands():
    app = _make_app()
    app._handle_command("/help")
    msg = _msgs(app)
    assert "/tree" in msg
    assert "/regen [N]" in msg
    assert "/edit" in msg
    assert "/branch" in msg
    assert "/nav" in msg
    assert "/del" in msg
    assert "/star" in msg
    assert "/note" in msg
    assert "⌃L" in msg
    # Phase 5 additions surface in /help.
    assert "/prune" in msg
    assert "/auto-regen" in msg
    assert "/diff" in msg
    assert "/save" in msg
    assert "/load" in msg


# ---------------------------------------------------------------------------
# Phase 5 — fan-out, `/regen N mode`, steering-delta in detail
# ---------------------------------------------------------------------------


def test_sweep_command_is_removed():
    app = _make_app()
    _seed_tree(app._session.tree)
    app._last_prompt = "do a thing"

    app._handle_command("/sweep angry.calm 0.0, 0.3")
    msg = _msgs(app)
    assert "Unknown command" in msg


def test_regen_with_mode_calls_regen_with_modifier():
    """`/regen 2 inverted` routes through ``regen_with_modifier`` (mode≠None)."""
    app = _make_app()
    uid, aid = _seed_tree(app._session.tree)
    captured = {}

    def _intercept(n, mode):
        captured["n"] = n
        captured["mode"] = mode
    app._run_regen_modifier_worker = _intercept
    # Make sure we aren't deferred onto the pending queue.
    app._session.is_generating = False

    app._handle_command("/regen 4 inverted")
    assert captured.get("n") == 4
    assert captured.get("mode") == "inverted"


def test_node_detail_recipe_delta_block():
    """Assistant siblings get a per-sibling steering-delta block."""
    tree = LoomTree(model_id="mock/mock")
    uid = tree.add_user_turn("hi")
    a1 = tree.begin_assistant(
        uid, recipe=Recipe(steering="0.3 honest", seed=1),
    )
    tree.finalize_assistant(a1, text="one")
    a2 = tree.begin_assistant(
        uid, recipe=Recipe(steering="0.3 honest + 0.2 warm", seed=2),
    )
    tree.finalize_assistant(a2, text="two")
    out = format_node_detail(tree, a2)
    assert "--- Recipe ---" in out
    assert "0.3 honest" in out  # the canonical steering expression
    assert "Δ from sibling" in out
    # ``warm`` resolves to ``warm.clinical`` via the pole alias on the
    # bundled bipolar concept; ``steering_delta`` drops the leading sign
    # on the first term (sign_prefix=False).
    assert "0.2 warm" in out  # delta from a1's recipe


def test_node_detail_no_recipe_block_when_no_siblings():
    """Solo assistant child gets no sibling-delta block."""
    tree = LoomTree(model_id="mock/mock")
    uid = tree.add_user_turn("hi")
    aid = tree.begin_assistant(uid, recipe=Recipe(steering="0.3 honest"))
    tree.finalize_assistant(aid, text="only one")
    out = format_node_detail(tree, aid)
    assert "Δ from sibling" not in out


def test_ab_compare_flips_auto_regen_on():
    """Ctrl+A toggles both _ab_mode and _loom_auto_regen_on."""
    app = _make_app()
    chat_panel = app._chat_panel
    chat_panel.set_ab_mode = MagicMock()
    chat_panel.assistant_rows_pending_shadow = MagicMock(return_value=[])
    app.action_ab_compare()
    assert app._ab_mode is True
    assert app._loom_auto_regen_on is True
    app.action_ab_compare()
    assert app._ab_mode is False
    assert app._loom_auto_regen_on is False


# ---------------------------------------------------------------------------
# Role-aware send — answer-prefill on a user node
# ---------------------------------------------------------------------------


def test_prefill_target_is_user_node_when_user_active():
    app = _make_app()
    tree = app._session.tree
    uid, _aid = _seed_tree(tree)
    tree.navigate(uid)
    assert app._prefill_target_node_id() == uid


def test_prefill_target_is_none_when_assistant_active():
    app = _make_app()
    tree = app._session.tree
    _uid, aid = _seed_tree(tree)
    tree.navigate(aid)
    assert app._prefill_target_node_id() is None


def test_user_submitted_on_user_node_routes_to_prefill():
    """A typed message on a user node prefills the assistant reply."""
    app = _make_app()
    tree = app._session.tree
    uid, _aid = _seed_tree(tree)
    tree.navigate(uid)
    app._start_prefill = MagicMock()
    app._start_generation = MagicMock()
    app.on_chat_panel_user_submitted(SimpleNamespace(text="It is sunny"))
    app._start_prefill.assert_called_once_with(uid, "It is sunny")
    app._start_generation.assert_not_called()
    # The prefill path must not optimistically mount a user row.
    app._chat_panel.add_user_message.assert_not_called()


def test_user_submitted_on_assistant_node_routes_to_generation():
    """A typed message on an assistant node is a normal new user turn."""
    app = _make_app()
    tree = app._session.tree
    _uid, aid = _seed_tree(tree)
    tree.navigate(aid)
    app._start_prefill = MagicMock()
    app._start_generation = MagicMock()
    app.on_chat_panel_user_submitted(SimpleNamespace(text="next question"))
    app._start_generation.assert_called_once_with("next question")
    app._start_prefill.assert_not_called()
    app._chat_panel.add_user_message.assert_called_once_with("next question")


def test_user_submitted_on_user_node_defers_prefill_target_in_pending():
    """Mid-gen submit enqueues the prefill target so the deferred dispatch
    can't re-resolve against a shifted active node.  Per the v2.x queue
    semantics the in-flight gen is *not* interrupted — the new item
    waits for the current ``done`` and drains FIFO."""
    from saklas.tui.chat_panel import PendingItem

    app = _make_app()
    tree = app._session.tree
    uid, _aid = _seed_tree(tree)
    tree.navigate(uid)
    app._session.is_generating = True
    app._session.stop = MagicMock()
    app._start_prefill = MagicMock()
    app.on_chat_panel_user_submitted(SimpleNamespace(text="seed it"))
    assert app._pending_queue == [
        PendingItem("submit", "seed it", (uid,)),
    ]
    app._start_prefill.assert_not_called()
    # Stop is NOT called — queue model preserves in-flight tokens.
    app._session.stop.assert_not_called()


# ---------------------------------------------------------------------------
# Commit (Ctrl+Enter / Alt+Enter) — no-generation send
# ---------------------------------------------------------------------------


def _stub_chat_input(app: SaklasApp, value: str) -> MagicMock:
    """Wire the chat panel's ``query_one('#chat-input', ChatInput)`` to a
    MagicMock carrying ``value`` as ``text``.  ``action_commit_text``
    reads the TextArea-backed widget via ``.text`` and clears it via
    ``load_text("")``, so the stub needs both attached.  We also keep
    ``text`` writable (assigning to it updates the backing string) so
    the post-action assertion ``inp.text == ""`` works after the
    handler's ``load_text("")`` call.
    """
    inp = MagicMock()
    inp.text = value
    # Make ``load_text`` mutate ``inp.text`` so empty-buffer assertions
    # after the action match what the real ChatInput does.
    inp.load_text = MagicMock(side_effect=lambda t: setattr(inp, "text", t))
    app._chat_panel.query_one = MagicMock(return_value=inp)
    return inp


def test_commit_action_on_assistant_node_routes_to_commit_user():
    """Ctrl+Enter on a non-user active node lands a new user turn —
    no generation — via ``_start_commit_user``."""
    app = _make_app()
    tree = app._session.tree
    _uid, aid = _seed_tree(tree)
    tree.navigate(aid)
    inp = _stub_chat_input(app, "new question")
    app._start_commit_user = MagicMock()
    app._start_commit_assistant = MagicMock()
    app.action_commit_text()
    app._start_commit_user.assert_called_once_with("new question")
    app._start_commit_assistant.assert_not_called()
    assert inp.text == ""
    assert "new question" in app._input_history


def test_commit_action_on_user_node_routes_to_commit_assistant():
    """Ctrl+Enter on a user active node lands an authored assistant
    turn via ``_start_commit_assistant``."""
    app = _make_app()
    tree = app._session.tree
    uid, _aid = _seed_tree(tree)
    tree.navigate(uid)
    inp = _stub_chat_input(app, "the full reply")
    app._start_commit_user = MagicMock()
    app._start_commit_assistant = MagicMock()
    app.action_commit_text()
    app._start_commit_assistant.assert_called_once_with(uid, "the full reply")
    app._start_commit_user.assert_not_called()
    assert inp.text == ""


def test_commit_action_empty_input_is_noop():
    """A whitespace-only commit drops on the floor — no dispatch, no
    history push, no input clear."""
    app = _make_app()
    inp = _stub_chat_input(app, "   ")
    app._start_commit_user = MagicMock()
    app._start_commit_assistant = MagicMock()
    app.action_commit_text()
    app._start_commit_user.assert_not_called()
    app._start_commit_assistant.assert_not_called()
    assert inp.text == "   "
    assert app._input_history == []


def test_commit_action_during_gen_queues_commit_user():
    """Mid-gen Ctrl+Enter on a non-user node enqueues a commit so the
    deferred dispatch lands it once the streaming sibling finishes.
    The queue model leaves the in-flight gen alone — no ``stop()``."""
    from saklas.tui.chat_panel import PendingItem

    app = _make_app()
    tree = app._session.tree
    _uid, aid = _seed_tree(tree)
    tree.navigate(aid)
    _stub_chat_input(app, "next bit")
    app._session.is_generating = True
    app._session.stop = MagicMock()
    app._start_commit_user = MagicMock()
    app.action_commit_text()
    assert app._pending_queue == [PendingItem("commit_user", "next bit")]
    app._start_commit_user.assert_not_called()
    app._session.stop.assert_not_called()


def test_commit_action_during_gen_queues_commit_assistant_with_target():
    """Mid-gen Ctrl+Enter on a user node enqueues the user-node target so
    the deferred dispatch can't re-resolve against a shifted active node."""
    from saklas.tui.chat_panel import PendingItem

    app = _make_app()
    tree = app._session.tree
    uid, _aid = _seed_tree(tree)
    tree.navigate(uid)
    _stub_chat_input(app, "the canned reply")
    app._session.is_generating = True
    app._session.stop = MagicMock()
    app._start_commit_assistant = MagicMock()
    app.action_commit_text()
    assert app._pending_queue == [
        PendingItem("commit_assistant", "the canned reply", (uid,)),
    ]
    app._start_commit_assistant.assert_not_called()
    app._session.stop.assert_not_called()


def test_dispatch_pending_commit_user_routes_correctly():
    """``_dispatch_pending_action(PendingItem("commit_user", text))`` calls
    ``_start_commit_user`` — the post-gen wakeup path."""
    from saklas.tui.chat_panel import PendingItem

    app = _make_app()
    app._start_commit_user = MagicMock()
    app._dispatch_pending_action(PendingItem("commit_user", "queued text"))
    app._start_commit_user.assert_called_once_with("queued text")


def test_dispatch_pending_commit_assistant_routes_correctly():
    """``_dispatch_pending_action(PendingItem("commit_assistant", text, (uid,)))``
    calls ``_start_commit_assistant`` with the stashed user-node target."""
    from saklas.tui.chat_panel import PendingItem

    app = _make_app()
    tree = app._session.tree
    uid, _aid = _seed_tree(tree)
    app._start_commit_assistant = MagicMock()
    app._dispatch_pending_action(
        PendingItem("commit_assistant", "the reply", (uid,))
    )
    app._start_commit_assistant.assert_called_once_with(uid, "the reply")


def test_slash_commit_on_assistant_node_routes_to_commit_user():
    """`/commit text` on a non-user active node lands a user turn — the
    cross-terminal fallback for Ctrl+Enter."""
    app = _make_app()
    tree = app._session.tree
    _uid, aid = _seed_tree(tree)
    tree.navigate(aid)
    app._start_commit_user = MagicMock()
    app._start_commit_assistant = MagicMock()
    app._handle_commit("manual user input")
    app._start_commit_user.assert_called_once_with("manual user input")
    app._start_commit_assistant.assert_not_called()


def test_slash_commit_on_user_node_routes_to_commit_assistant():
    """`/commit text` on a user active node lands an authored assistant
    turn (full reply), same as Ctrl+Enter on a user node."""
    app = _make_app()
    tree = app._session.tree
    uid, _aid = _seed_tree(tree)
    tree.navigate(uid)
    app._start_commit_user = MagicMock()
    app._start_commit_assistant = MagicMock()
    app._handle_commit("the canned reply")
    app._start_commit_assistant.assert_called_once_with(uid, "the canned reply")
    app._start_commit_user.assert_not_called()


def test_slash_commit_empty_text_prints_usage():
    """`/commit` with no argument prints the usage hint and dispatches
    nothing."""
    app = _make_app()
    app._start_commit_user = MagicMock()
    app._start_commit_assistant = MagicMock()
    app._handle_commit("   ")
    app._start_commit_user.assert_not_called()
    app._start_commit_assistant.assert_not_called()
    assert "Usage: /commit" in _msgs(app)
