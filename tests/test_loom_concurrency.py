"""LoomTree concurrency tests — gen reservation vs mutators.

The session sets ``_active_gen_reservation`` to the user-parent of the
streaming assistant node while a generation is in flight.  Its
``_loom_conflict_check`` consults this and raises
:class:`MutationDuringGenerationError` (HTTP 409) on conflicting ops.

These tests reproduce that protocol by wiring an equivalent conflict-
check function directly into a :class:`LoomTree`, so the rules can be
verified without loading a model.
"""
from __future__ import annotations

import pytest

from saklas import (
    LoomTree,
    MutationDuringGenerationError,
)


def _mk_tree_with_reservation():
    """Build a tree shaped like:

        root
         └── u1 ("hi")
              ├── a1 ("hello")          <-- gen target (in flight)
              └── a2 ("alt", finalized)

        plus a disjoint branch:

         └── u_other ("how are you")
              └── a_other ("fine")

    Reservation = u1 (the user-parent of the streaming assistant).
    """
    state = {"reservation": None}

    def conflict_check(node_id: str, op: str) -> None:
        reservation = state["reservation"]
        if reservation is None:
            return
        if op in ("add_user_turn", "begin_assistant", "finalize_assistant",
                  "branch", "star", "note", "navigate"):
            return
        if op == "reset":
            raise MutationDuringGenerationError(
                "cannot reset tree while a generation is in flight"
            )
        if (node_id == reservation
                or t.is_ancestor_of(reservation, node_id)
                or t.is_ancestor_of(node_id, reservation)):
            raise MutationDuringGenerationError(
                f"cannot {op} on node inside reservation {reservation}"
            )

    t = LoomTree()
    t.set_conflict_check(conflict_check)
    u1 = t.add_user_turn("hi")
    a1 = t.begin_assistant(u1)
    t.finalize_assistant(a1, text="hello")
    a2 = t.branch(a1, "alt")
    # Build a disjoint user→assistant pair under root.
    t.navigate(t.root_id)
    u_other = t.add_user_turn("how are you")
    a_other = t.begin_assistant(u_other)
    t.finalize_assistant(a_other, text="fine")
    # Set reservation = u1, mid-gen on a1's subtree.
    state["reservation"] = u1
    return t, state, u1, a1, a2, u_other, a_other


# ---------------------------------------------------------------------------
# Free ops (always allowed)
# ---------------------------------------------------------------------------


def test_branch_in_reservation_is_free():
    t, state, u1, a1, a2, _, _ = _mk_tree_with_reservation()
    # Branching off a sibling under the reserved subtree is allowed —
    # creating a sibling doesn't disturb the streaming target.
    new_id = t.branch(a1, "another alt")
    assert new_id in t.nodes
    assert t.nodes[new_id].parent_id == u1


def test_star_in_reservation_is_free():
    t, state, u1, a1, *_ = _mk_tree_with_reservation()
    t.star(a1, True)
    assert t.nodes[a1].starred is True


def test_annotate_in_reservation_is_free():
    t, state, u1, a1, *_ = _mk_tree_with_reservation()
    t.annotate(a1, "looked promising")
    assert t.nodes[a1].notes == "looked promising"


def test_navigate_during_gen_is_free():
    """Navigate-away leaves the streaming target attached; the user can
    navigate back at any time.  Phase 1 spec: 'navigate-away during gen
    leaves the gen attached to its original target'.
    """
    t, state, u1, a1, a2, u_other, a_other = _mk_tree_with_reservation()
    t.navigate(a_other)
    assert t.active_node_id == a_other


def test_branch_outside_reservation_is_free():
    t, state, _, _, _, _, a_other = _mk_tree_with_reservation()
    new_id = t.branch(a_other, "fine too")
    assert new_id in t.nodes


# ---------------------------------------------------------------------------
# Refused ops (mutations on the reservation)
# ---------------------------------------------------------------------------


def test_edit_in_reservation_refused():
    t, state, u1, a1, *_ = _mk_tree_with_reservation()
    with pytest.raises(MutationDuringGenerationError) as exc:
        t.edit(a1, "edited mid-gen")
    code, _ = exc.value.user_message()
    assert code == 409


def test_edit_on_user_parent_refused():
    """Editing the user-parent of the streaming assistant is refused too —
    the reservation covers the subtree rooted at the user parent."""
    t, state, u1, *_ = _mk_tree_with_reservation()
    with pytest.raises(MutationDuringGenerationError):
        t.edit(u1, "different prompt")


def test_edit_on_sibling_in_reservation_refused():
    """Even a non-target sibling inside the reservation refuses — phase 1
    spec: 'editing a non-target node in the reservation can corrupt token-
    score replay for downstream readers'."""
    t, state, u1, a1, a2, *_ = _mk_tree_with_reservation()
    with pytest.raises(MutationDuringGenerationError):
        t.edit(a2, "edited sibling")


def test_delete_subtree_intersecting_reservation_refused():
    t, state, u1, a1, *_ = _mk_tree_with_reservation()
    t.navigate(t.root_id)  # nav away first so it's not an ancestor of active
    with pytest.raises(MutationDuringGenerationError):
        t.delete_subtree(u1)


def test_reset_during_gen_refused():
    t, state, *_ = _mk_tree_with_reservation()
    with pytest.raises(MutationDuringGenerationError):
        t.reset()


# ---------------------------------------------------------------------------
# Free ops outside the reservation
# ---------------------------------------------------------------------------


def test_edit_outside_reservation_succeeds():
    t, state, _, _, _, _, a_other = _mk_tree_with_reservation()
    t.edit(a_other, "fine, you?")
    assert t.nodes[a_other].text == "fine, you?"


def test_delete_subtree_outside_reservation_succeeds():
    t, state, _, _, _, u_other, a_other = _mk_tree_with_reservation()
    t.navigate(t.root_id)
    n = t.delete_subtree(u_other)
    assert n == 2  # u_other + a_other


# ---------------------------------------------------------------------------
# Reservation lifecycle
# ---------------------------------------------------------------------------


def test_clearing_reservation_lifts_lock():
    t, state, u1, a1, *_ = _mk_tree_with_reservation()
    state["reservation"] = None
    # Now edits are free again.
    t.edit(a1, "post-gen edit")
    assert t.nodes[a1].text == "post-gen edit"


def test_reset_after_clearing_succeeds():
    t, state, *_ = _mk_tree_with_reservation()
    state["reservation"] = None
    t.reset()
    assert len(t.nodes) == 1  # synthetic root only


# ---------------------------------------------------------------------------
# 409 status code propagation
# ---------------------------------------------------------------------------


def test_mutation_during_generation_user_message_is_409():
    err = MutationDuringGenerationError("test")
    code, msg = err.user_message()
    assert code == 409
    assert "test" in msg
