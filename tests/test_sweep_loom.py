"""Tests for the sweep-as-loom-sibling shape (v2.3 phase 5 / decision 5).

These tests don't load a real model — they verify the deterministic
sibling layout and seed schedule on the engine side by stubbing out
``_generate_core``.  Per-row gen invocation is mocked; tree mutation
is real (loom + recipe stamping).
"""
from __future__ import annotations

import types
from typing import Any

import pytest

from saklas import (
    LoomTree,
    Recipe,
    SamplingConfig,
    derive_seed_schedule,
)
from saklas.core.results import GenerationResult


class _SweepStub:
    """Real LoomTree + stubbed ``_generate_core`` to skip the model load.

    We reuse :meth:`SaklasSession.generate_sweep` directly off the class
    by binding to this stub so the loom + sibling-bookkeeping logic
    runs verbatim.
    """

    def __init__(self) -> None:
        self.tree = LoomTree()
        self.calls: list[dict[str, Any]] = []
        self._gen_state = types.SimpleNamespace(
            stop_requested=types.SimpleNamespace(is_set=lambda: False),
        )

    def _generate_core(
        self,
        prompt: str,
        *,
        steering: Any = None,
        sampling: Any = None,
        stateless: bool = False,
        raw: bool = False,
        thinking: Any = None,
        parent_node_id: Any = None,
        on_token: Any = None,
        recipe_override: Any = None,
    ) -> GenerationResult:
        # Echo the call args for assertions.
        self.calls.append({
            "prompt": prompt, "steering": steering,
            "sampling": sampling, "parent_node_id": parent_node_id,
        })
        if not stateless:
            # Mimic the real path: spawn user + assistant under the
            # current active node so the tree shape matches a real run.
            uid = self.tree.add_user_turn(prompt, parent_id=parent_node_id)
            aid = self.tree.begin_assistant(
                uid,
                recipe=Recipe(steering=str(steering) if steering else None),
            )
            self.tree.finalize_assistant(
                aid, text=f"r{len(self.calls)}", applied_steering=str(steering),
            )
        return GenerationResult(
            text=f"r{len(self.calls)}",
            tokens=[], token_count=0, tok_per_sec=0.0, elapsed=0.001,
            readings={}, vectors={},
            applied_steering=str(steering) if steering else None,
        )


def test_sweep_lands_siblings_under_user_turn():
    from saklas.core.session import SaklasSession

    stub = _SweepStub()
    sweep = SaklasSession.generate_sweep.__get__(stub, _SweepStub)
    results = sweep(
        "say hi", sweep={"honest.deceptive": [0.0, 0.3, 0.6]},
        stateless=False,
    )
    assert len(results) == 3
    # All three sibling assistant nodes share the same user-parent.
    parents = set()
    for child_id in stub.tree.children_of[stub.tree.root_id]:
        parents.add(child_id)
    # The synthetic root has exactly one direct user child (the dedup'd
    # one) — sweep anchors siblings under it.
    assert len(parents) == 1
    user_id = next(iter(parents))
    assert stub.tree.nodes[user_id].role == "user"
    sibling_ids = stub.tree.children_of[user_id]
    # Exactly N sibling assistants under that user.
    assert len(sibling_ids) == 3
    for sid in sibling_ids:
        assert stub.tree.nodes[sid].role == "assistant"


def test_sweep_return_node_ids_returns_parallel_lists():
    from saklas.core.session import SaklasSession

    stub = _SweepStub()
    sweep = SaklasSession.generate_sweep.__get__(stub, _SweepStub)
    out = sweep(
        "go", sweep={"honest.deceptive": [0.1, 0.2]},
        stateless=False, return_node_ids=True,
    )
    assert isinstance(out, tuple)
    results, node_ids = out
    assert len(results) == 2
    assert len(node_ids) == 2
    for nid in node_ids:
        assert nid in stub.tree.nodes


def test_sweep_explicit_return_node_ids_false_warns():
    """Explicit ``return_node_ids=False`` emits a DeprecationWarning;
    omitting the kwarg (sentinel default) does NOT.  v2.4 will flip
    the default to the tuple shape.
    """
    import warnings

    from saklas.core.session import SaklasSession

    stub = _SweepStub()
    sweep = SaklasSession.generate_sweep.__get__(stub, _SweepStub)

    # Explicit False — warn.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = sweep(
            "go", sweep={"honest.deceptive": [0.1]},
            stateless=True, return_node_ids=False,
        )
    assert any(
        issubclass(w.category, DeprecationWarning)
        and "return_node_ids=False" in str(w.message)
        for w in caught
    )
    # Return shape unchanged in v2.3 — bare list, not tuple.
    assert isinstance(out, list)

    # Default (sentinel) — no warning.
    stub2 = _SweepStub()
    sweep2 = SaklasSession.generate_sweep.__get__(stub2, _SweepStub)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out2 = sweep2(
            "go", sweep={"honest.deceptive": [0.1]}, stateless=True,
        )
    assert not any(
        issubclass(w.category, DeprecationWarning) for w in caught
    )
    assert isinstance(out2, list)


def test_sweep_seed_schedule_is_deterministic():
    from saklas.core.session import SaklasSession

    stub = _SweepStub()
    sweep = SaklasSession.generate_sweep.__get__(stub, _SweepStub)
    sweep(
        "x", sweep={"honest.deceptive": [0.1, 0.2, 0.3]},
        sampling=SamplingConfig(seed=42),
        stateless=False,
    )
    seeds = [c["sampling"].seed for c in stub.calls]
    # Same fan-out under the same base seed reproduces — compare against
    # the derive_seed_schedule schedule applied per row.
    rebuilt = [derive_seed_schedule(42, i + 1)[-1] for i in range(3)]
    assert seeds == rebuilt


def test_sweep_default_return_shape_is_results_list():
    from saklas.core.session import SaklasSession

    stub = _SweepStub()
    sweep = SaklasSession.generate_sweep.__get__(stub, _SweepStub)
    results = sweep(
        "x", sweep={"honest.deceptive": [0.0]},
        stateless=False,
    )
    # Legacy shape: just the result list.
    assert isinstance(results, list)
    assert len(results) == 1
