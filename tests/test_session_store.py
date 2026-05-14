"""Tests for the v2.3-minimal session persistence module."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from saklas.core.loom import LoomTree, Recipe
from saklas.io import session_store


@pytest.fixture
def saklas_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point ``$SAKLAS_HOME`` at a fresh temp dir for the test."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    return tmp_path


# ---------------------------------------------------------------------------
# default_session_id
# ---------------------------------------------------------------------------


def test_default_session_id_mints_and_persists(saklas_home: Path) -> None:
    sid1 = session_store.default_session_id()
    # ULID shape: 26 chars, Crockford-base32 alphabet only.
    assert len(sid1) == 26
    assert all(c in "0123456789ABCDEFGHJKMNPQRSTVWXYZ" for c in sid1)

    # Pointer file written.
    pointer = saklas_home / "sessions" / ".default"
    assert pointer.is_file()
    assert pointer.read_text().strip() == sid1

    # Calling again returns the same id.
    sid2 = session_store.default_session_id()
    assert sid2 == sid1


def test_default_session_id_tolerates_trailing_whitespace(saklas_home: Path) -> None:
    pointer = saklas_home / "sessions" / ".default"
    pointer.parent.mkdir(parents=True)
    pointer.write_text("01H8XGJWBWBAB1KMR1ZHFR8CD9\n\n\n")
    sid = session_store.default_session_id()
    assert sid == "01H8XGJWBWBAB1KMR1ZHFR8CD9"


# ---------------------------------------------------------------------------
# save/load roundtrip
# ---------------------------------------------------------------------------


def test_save_load_roundtrip(saklas_home: Path) -> None:
    tree = LoomTree(model_id="m1")
    u = tree.add_user_turn("hi")
    a = tree.begin_assistant(u, recipe=Recipe(steering="0.3 honest"))
    tree.finalize_assistant(a, text="hello", aggregate_readings={"x": 0.5})

    sid = "01TESTROUNDTRIP00000000000"
    session_store.save_tree(sid, tree)

    # File at the expected path.
    path = session_store.session_path(sid) / "tree.json"
    assert path.is_file()

    loaded = session_store.load_tree(sid)
    assert loaded is not None
    assert loaded.model_id == "m1"
    assert loaded.rev == tree.rev
    assert loaded.active_node_id == tree.active_node_id
    assert loaded.nodes[a].text == "hello"
    assert loaded.nodes[a].aggregate_readings == {"x": 0.5}
    assert loaded.nodes[a].recipe is not None
    assert loaded.nodes[a].recipe.steering == "0.3 honest"


def test_load_tree_missing_returns_none(saklas_home: Path) -> None:
    assert session_store.load_tree("01DOESNOTEXIST00000000000A") is None


def test_load_tree_corrupt_returns_none(saklas_home: Path) -> None:
    """A partially-written or otherwise unparseable ``tree.json``
    should return ``None`` so callers fall back to a fresh tree
    rather than crashing.  Simulates a crash mid-write before the
    atomic-replace step lands — the prior good state (if any) would
    be at ``<path>`` while the half-written bytes sit at ``<path>.tmp``;
    the read path only opens ``<path>`` so the orphan never confuses
    the load.  We verify the load is also defensive against a truly
    corrupted ``tree.json`` (e.g. truncation during a non-atomic
    write from an external tool).
    """
    sid = "01CORRUPTSESSION0000000000"
    path = session_store.session_path(sid) / "tree.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not valid json")
    assert session_store.load_tree(sid) is None


def test_atomic_write_orphan_tmp_does_not_corrupt_prior_good(saklas_home: Path) -> None:
    """Manually stage a half-written ``.tmp`` alongside a good ``tree.json``;
    confirm the loader returns the prior good version (the orphan is
    invisible to the load path).
    """
    tree = LoomTree(model_id="m_good")
    u = tree.add_user_turn("hello")
    tree.finalize_assistant(
        tree.begin_assistant(u),
        text="prior-good",
    )

    sid = "01PRIORGOOD0000000000000AA"
    session_store.save_tree(sid, tree)

    # Now drop a half-written tmp file in the same directory — simulating
    # a crash mid-write where the atomic-replace step didn't land.
    tmp = session_store.session_path(sid) / "tree.json.tmp"
    tmp.write_text("{half-written")

    # Loader reads the canonical path; the orphan tmp is ignored.
    loaded = session_store.load_tree(sid)
    assert loaded is not None
    # We can recover the prior-good assistant text.
    assistant_texts = [
        n.text for n in loaded.nodes.values()
        if n.role == "assistant"
    ]
    assert "prior-good" in assistant_texts


def test_save_overwrites_via_atomic(saklas_home: Path) -> None:
    """Two consecutive saves must end with the second tree's contents
    on disk — verifies the atomic-replace path is wired through.
    """
    sid = "01OVERWRITE00000000000000B"
    t1 = LoomTree(model_id="m")
    t1.add_user_turn("first")
    session_store.save_tree(sid, t1)

    t2 = LoomTree(model_id="m")
    t2.add_user_turn("second")
    session_store.save_tree(sid, t2)

    loaded = session_store.load_tree(sid)
    assert loaded is not None
    user_texts = [n.text for n in loaded.nodes.values() if n.role == "user"]
    assert "second" in user_texts
    assert "first" not in user_texts


# ---------------------------------------------------------------------------
# Honors SAKLAS_HOME
# ---------------------------------------------------------------------------


def test_session_path_honors_saklas_home(saklas_home: Path) -> None:
    expected = saklas_home / "sessions" / "01HONORSAKLASHOME00000000A"
    assert session_store.session_path("01HONORSAKLASHOME00000000A") == expected


def test_sessions_root_honors_saklas_home(saklas_home: Path) -> None:
    assert session_store.sessions_root() == saklas_home / "sessions"
