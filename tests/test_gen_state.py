"""Tests for the typed ``GenState`` lifecycle on ``SaklasSession``.

GPU-gated for the same reason as ``test_session.py``: ``GenState``
transitions are exercised through real generation, which needs a model
loaded on a real device.  See Phase 6 of the audit-followups plan.
"""
from __future__ import annotations

import pytest
import torch

from saklas import GenState
from saklas.core.session import ConcurrentGenerationError

_HAS_GPU = torch.cuda.is_available() or torch.backends.mps.is_available()
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not _HAS_GPU,
        reason="No GPU backend available (neither CUDA nor MPS)",
    ),
]

MODEL_ID = "google/gemma-3-4b-it"


@pytest.fixture(scope="module")
def session():
    from saklas.core.session import SaklasSession
    s = SaklasSession.from_pretrained(MODEL_ID, device="auto", probes=["affect"])
    yield s
    s.close()


class TestGenStateTransitions:
    def test_idle_at_construction(self, session):
        assert session.gen_state is GenState.IDLE
        assert session.is_generating is False

    def test_running_during_generate(self, session):
        """Inside an ``on_token`` callback, the session must report
        ``RUNNING`` — that callback fires from the generation worker
        thread between the inner-try entry and exit."""
        session.clear_history()
        observed: list[GenState] = []

        def _tap(*args, **kwargs):
            observed.append(session.gen_state)

        from saklas.core.sampling import SamplingConfig
        session.generate(
            "Say hi.",
            sampling=SamplingConfig(max_tokens=4),
            on_token=_tap,
        )
        assert observed, "on_token never fired"
        assert all(s is GenState.RUNNING for s in observed), (
            f"expected all RUNNING, got {observed}"
        )

    def test_returns_to_idle_after_success(self, session):
        session.clear_history()
        from saklas.core.sampling import SamplingConfig
        session.generate(
            "Say hi.",
            sampling=SamplingConfig(max_tokens=4),
        )
        assert session.gen_state is GenState.IDLE
        assert session.is_generating is False

    def test_returns_to_idle_after_exception(self, session):
        """Worker-side failures must still drain the state machine.

        We trigger this by referencing an unregistered vector — the
        steering scope materialization raises before the inner try
        runs, but the outer ``finally`` still has to land us at
        ``IDLE`` for the next call to succeed.
        """
        session.clear_history()
        from saklas.core.sampling import SamplingConfig
        with pytest.raises(Exception):
            session.generate(
                "Say hi.",
                steering="0.3 definitely_not_a_real_concept_xyz",
                sampling=SamplingConfig(max_tokens=4),
            )
        assert session.gen_state is GenState.IDLE
        assert session.is_generating is False

        # And a fresh generation still works — the lock got released.
        session.generate(
            "Say hi.",
            sampling=SamplingConfig(max_tokens=4),
        )
        assert session.gen_state is GenState.IDLE


class TestConcurrentGuard:
    def test_concurrent_generate_rejected(self, session):
        """Re-entering generation from inside ``on_token`` must raise
        ``ConcurrentGenerationError`` — the typed state guard sits on
        top of the threading lock.
        """
        session.clear_history()
        from saklas.core.sampling import SamplingConfig

        captured: list[BaseException] = []

        def _reentry(*args, **kwargs):
            try:
                session.generate(
                    "nope",
                    sampling=SamplingConfig(max_tokens=2),
                )
            except ConcurrentGenerationError as e:
                captured.append(e)

        session.generate(
            "Say hi.",
            sampling=SamplingConfig(max_tokens=4),
            on_token=_reentry,
        )
        # The first emit should have triggered the guard at least once.
        assert captured, "expected ConcurrentGenerationError on re-entry"
        # And the outer call still finalized cleanly.
        assert session.gen_state is GenState.IDLE
