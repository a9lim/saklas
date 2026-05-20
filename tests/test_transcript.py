"""Tests for transcript export / import (v2.3 phase 5)."""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable

import pytest

from saklas import (
    LoomTree,
    ProbeRef,
    Recipe,
    SamplingConfig,
    Transcript,
    TranscriptFormatError,
    TranscriptModelMismatch,
    TranscriptProbeDriftError,
    TranscriptTurn,
)


# ---------------------------------------------------------------------------
# Synthetic session stub — avoids loading a real model.
# ---------------------------------------------------------------------------


class _Monitor:
    def __init__(self, probe_names: Iterable[str] = ()) -> None:
        self.probe_names = list(probe_names)
        self.profiles = {name: {0: f"prof:{name}"} for name in self.probe_names}


class _Config:
    def __init__(self, system_prompt: str | None = None) -> None:
        self.system_prompt = system_prompt


class _StubSession:
    """Tiny replacement for SaklasSession exposing the transcript API surface."""

    def __init__(
        self,
        *,
        model_id: str = "m1",
        system_prompt: str | None = None,
        probe_names: Iterable[str] = ("angry.calm", "honest.deceptive"),
        probe_hashes: dict[str, str] | None = None,
    ) -> None:
        self.model_id = model_id
        self.tree = LoomTree(model_id=model_id)
        self.config = _Config(system_prompt=system_prompt)
        self._monitor = _Monitor(probe_names=probe_names)
        self._probe_hash_cache = probe_hashes or {
            name: f"hash_of_{name}" for name in self._monitor.probe_names
        }

    def _probe_hash(self, name: str) -> str | None:
        return self._probe_hash_cache.get(name)


# ---------------------------------------------------------------------------
# Schema round-trip
# ---------------------------------------------------------------------------


def test_schema_round_trip():
    t = Transcript(
        model_id="google/gemma-3-4b-it",
        system_prompt="You are a helpful assistant.",
        probes=[
            ProbeRef(name="angry.calm", sha256="abc123"),
            ProbeRef(name="honest.deceptive", sha256="def456"),
        ],
        turns=[
            TranscriptTurn(role="user", text="What makes a good day?"),
            TranscriptTurn(
                role="assistant",
                text="Sunshine.",
                recipe=Recipe(
                    steering="0.3 honest.deceptive",
                    sampling=SamplingConfig(temperature=0.7, max_tokens=256),
                    seed=42,
                ),
                readings={"angry.calm": -0.12, "honest.deceptive": 0.41},
            ),
        ],
    )
    y = t.to_yaml()
    t2 = Transcript.from_yaml(y)
    assert t2.model_id == t.model_id
    assert t2.system_prompt == t.system_prompt
    assert len(t2.probes) == 2
    assert t2.probes[0].sha256 == "abc123"
    recipe = t2.turns[1].recipe
    assert recipe is not None
    assert recipe.steering == "0.3 honest.deceptive"
    assert t2.turns[1].readings["angry.calm"] == pytest.approx(-0.12)


def test_from_yaml_rejects_unknown_version():
    with pytest.raises(TranscriptFormatError, match="version"):
        Transcript.from_yaml("saklas_transcript: 99\nturns: []\n")


def test_from_yaml_rejects_non_mapping_root():
    with pytest.raises(TranscriptFormatError, match="must be a mapping"):
        Transcript.from_yaml("- foo\n- bar\n")


def test_save_load_round_trip(tmp_path: Path):
    t = Transcript(
        model_id="m", system_prompt=None, probes=[],
        turns=[
            TranscriptTurn(role="user", text="hi"),
            TranscriptTurn(role="assistant", text="hello"),
        ],
    )
    out = tmp_path / "trans.yaml"
    t.save(out)
    assert out.is_file()
    t2 = Transcript.load(out)
    assert len(t2.turns) == 2
    assert t2.turns[0].text == "hi"


# ---------------------------------------------------------------------------
# Build from session path
# ---------------------------------------------------------------------------


def test_from_path_walks_active_path():
    sess = _StubSession()
    u = sess.tree.add_user_turn("hi there")
    a = sess.tree.begin_assistant(u, recipe=Recipe(steering="0.3 honest.deceptive"))
    sess.tree.finalize_assistant(
        a, text="hello back",
        aggregate_readings={"angry.calm": 0.1},
    )
    t = Transcript.from_path(None, sess)
    assert t.model_id == "m1"
    roles = [turn.role for turn in t.turns]
    assert roles == ["user", "assistant"]
    assert t.turns[1].readings == {"angry.calm": 0.1}
    assert len(t.probes) == 2
    assert t.probes[0].sha256 == "hash_of_angry.calm"


# ---------------------------------------------------------------------------
# Three import modes
# ---------------------------------------------------------------------------


def _simple_transcript(*, model_id: str = "m1", with_assistant: bool = True) -> Transcript:
    turns = [TranscriptTurn(role="user", text="hi")]
    if with_assistant:
        turns.append(TranscriptTurn(role="assistant", text="hello"))
    turns.append(TranscriptTurn(role="user", text="how"))
    if with_assistant:
        turns.append(TranscriptTurn(role="assistant", text="fine"))
    return Transcript(
        model_id=model_id, system_prompt=None,
        probes=[], turns=turns,
    )


def test_import_default_attaches_under_root():
    sess = _StubSession()
    t = _simple_transcript()
    leaf = t.import_into(sess, mode="default")
    # Walk up — should hit root_id without crossing any existing user node.
    chain = [n.id for n in sess.tree.path_to(leaf)]
    assert sess.tree.root_id in chain


def test_import_here_attaches_under_active():
    sess = _StubSession()
    # Seed an existing path so active != root.
    u0 = sess.tree.add_user_turn("seed")
    a0 = sess.tree.begin_assistant(u0, recipe=Recipe())
    sess.tree.finalize_assistant(a0, text="seed-ans")

    t = _simple_transcript()
    leaf = t.import_into(sess, mode="here")
    chain = [n.id for n in sess.tree.path_to(leaf)]
    assert a0 in chain


def test_import_merge_finds_user_prefix():
    sess = _StubSession()
    # Pre-populate sess with the same first user-turn the transcript carries.
    u0 = sess.tree.add_user_turn("hi")
    a0 = sess.tree.begin_assistant(u0, recipe=Recipe())
    sess.tree.finalize_assistant(a0, text="prior-answer")

    t = _simple_transcript()
    leaf = t.import_into(sess, mode="merge")
    # Leaf path should descend from u0 (the shared "hi" user turn).
    chain = [n.id for n in sess.tree.path_to(leaf)]
    assert u0 in chain


def test_import_merge_skip_walks_existing_prefix_no_dupes():
    """Two-user-deep prefix merge: the imported tail must attach under
    the deepest matched user (u2), and the matched-prefix users (u1, u2)
    must NOT be re-attached as duplicates under it.

    Active path: ``root -> u1 -> a1 -> u2 -> a2``
    Transcript users: ``["u1_text", "u2_text", "u3_text"]`` (assistants
    between the matched users are dropped on the floor, matching the
    rest of the merge mode — the structural location is "already
    represented in the path", so the alternate assistant text would
    land at the wrong depth if attached under the merge anchor).
    Post-merge u3 should attach as a descendant of u2 (the deepest
    matched user), with no u1/u2 duplicates spawned under u2.
    """
    sess = _StubSession()
    u1 = sess.tree.add_user_turn("u1_text")
    a1 = sess.tree.begin_assistant(u1, recipe=Recipe())
    sess.tree.finalize_assistant(a1, text="a1_text")
    u2 = sess.tree.add_user_turn("u2_text", parent_id=a1)
    a2 = sess.tree.begin_assistant(u2, recipe=Recipe())
    sess.tree.finalize_assistant(a2, text="a2_text")

    # Transcript: matched-prefix [u1, u2] plus a fresh u3 tail.  No
    # assistants in the matched-prefix region — keeping the structure
    # symmetric with the task description while still exercising the
    # tail-attach path (an assistant follows u3 so we have a real
    # leaf node to assert on).
    t = Transcript(
        model_id="m1", system_prompt=None, probes=[],
        turns=[
            TranscriptTurn(role="user", text="u1_text"),
            TranscriptTurn(role="user", text="u2_text"),
            TranscriptTurn(role="user", text="u3_text"),
            TranscriptTurn(role="assistant", text="a3_text"),
        ],
    )
    leaf = t.import_into(sess, mode="merge")

    # u2's children: must contain a fresh user node "u3_text", with
    # NO duplicate "u1_text" / "u2_text" user nodes — the pre-fix
    # behaviour spawned those as children of the merge anchor when
    # the skip-walk re-called ``add_user_turn`` on matched prefixes.
    u2_children = sess.tree.children(u2)
    user_child_texts = [c.text for c in u2_children if c.role == "user"]
    assert "u3_text" in user_child_texts
    assert "u1_text" not in user_child_texts
    assert "u2_text" not in user_child_texts

    # The leaf assistant ("a3_text") lives at depth ``root -> u1 ->
    # a1 -> u2 -> u3_text -> a3_text`` — every matched-prefix node
    # is in the ancestor chain.
    chain = [n.id for n in sess.tree.path_to(leaf)]
    assert u1 in chain
    assert u2 in chain


def test_import_merge_falls_back_to_root_when_no_match():
    sess = _StubSession()
    u0 = sess.tree.add_user_turn("different")
    sess.tree.begin_assistant(u0, recipe=Recipe())

    t = _simple_transcript()
    leaf = t.import_into(sess, mode="merge")
    chain = [n.id for n in sess.tree.path_to(leaf)]
    assert sess.tree.root_id in chain


# ---------------------------------------------------------------------------
# Guard conditions
# ---------------------------------------------------------------------------


def test_guard_model_mismatch_refuses_merge():
    sess = _StubSession(model_id="m1")
    t = _simple_transcript(model_id="m2")
    with pytest.raises(TranscriptModelMismatch):
        t.import_into(sess, mode="merge")


def test_guard_model_mismatch_warns_on_default():
    sess = _StubSession(model_id="m1")
    t = _simple_transcript(model_id="m2")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t.import_into(sess, mode="default")
    assert any("model" in str(w.message) for w in caught)
    # First imported node carries a banner note.
    notes = []
    for n in sess.tree.descendants(sess.tree.root_id):
        if n.notes:
            notes.append(n.notes)
    assert any("model_mismatch" in n for n in notes)


def test_guard_system_prompt_mismatch_proceeds_with_warning():
    sess = _StubSession(system_prompt="be helpful")
    t = Transcript(
        model_id="m1", system_prompt="be sarcastic",
        probes=[],
        turns=[TranscriptTurn(role="user", text="hi"),
               TranscriptTurn(role="assistant", text="ok")],
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t.import_into(sess, mode="default")
    assert any("system" in str(w.message) for w in caught)


def test_guard_missing_probes_warns():
    sess = _StubSession(probe_names=("angry.calm",))
    t = Transcript(
        model_id="m1", system_prompt=None,
        probes=[ProbeRef(name="never_loaded", sha256="abc")],
        turns=[TranscriptTurn(role="user", text="hi"),
               TranscriptTurn(role="assistant", text="ok")],
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t.import_into(sess, mode="default")
    assert any("probes" in str(w.message).lower() for w in caught)


def test_guard_probe_drift_warns_then_strict_raises():
    sess = _StubSession(
        probe_names=("angry.calm",),
        probe_hashes={"angry.calm": "session_hash"},
    )
    t = Transcript(
        model_id="m1", system_prompt=None,
        probes=[ProbeRef(name="angry.calm", sha256="transcript_hash")],
        turns=[TranscriptTurn(role="user", text="hi"),
               TranscriptTurn(role="assistant", text="ok")],
    )
    # Non-strict warns + proceeds.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t.import_into(sess, mode="default")
    assert any("drift" in str(w.message).lower() for w in caught)

    # Strict raises.
    sess2 = _StubSession(
        probe_names=("angry.calm",),
        probe_hashes={"angry.calm": "session_hash"},
    )
    with pytest.raises(TranscriptProbeDriftError):
        t.import_into(sess2, mode="default", strict=True)


# ---------------------------------------------------------------------------
# CLI verb dispatch
# ---------------------------------------------------------------------------


def test_cli_verb_registered():
    from saklas.cli.parsers import _build_root_parser
    from saklas.cli.runners import _COMMAND_RUNNERS

    parser = _build_root_parser()
    ns = parser.parse_args([
        "experiment", "transcript", "run", "/tmp/x.yaml", "m1",
    ])
    assert ns.command == "experiment"
    assert ns.experiment_cmd == "transcript"
    assert ns.transcript_cmd == "run"
    assert ns.path == "/tmp/x.yaml"
    assert "experiment" in _COMMAND_RUNNERS
    assert "transcript" not in _COMMAND_RUNNERS
