"""CPU tests for ablation dispatch through _SteeringContext / _rebuild_steering_hooks.

Builds a ``SaklasSession`` skeleton by hand (bypassing ``__init__`` via
``__new__``) so no model load is required.  Safe because these tests never
call generate() / extract() / anything that touches the model -- only the
steering-stack manipulation and hook-manager wiring is exercised.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import torch

from saklas.io import selectors as _sel
from saklas.core.events import EventBus
from saklas.core.hooks import SteeringManager
from saklas.core.session import SaklasSession, ProfileNotRegisteredError
from saklas.core.steering import Steering
from saklas.core.steering_composer import SteeringComposer
from saklas.core.steering_expr import AblationTerm
from saklas.core.triggers import Trigger
from tests._whitener import isotropic_whitener


@pytest.fixture(autouse=True)
def _isolated_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Keep parser pole-resolution from scanning the user's real vectors dir."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    _sel.invalidate()
    yield
    _sel.invalidate()


class _NoopModule(torch.nn.Module):
    def forward(self, x):  # pyright: ignore[reportMissingParameterType]  # noop stub, intentionally untyped
        return (x,)


def _skeleton_session() -> SaklasSession:
    import threading
    session = SaklasSession.__new__(SaklasSession)
    session._model = None  # pyright: ignore[reportAttributeAccessIssue]  # skeleton: bypasses __init__, _model accepts None here
    session._tokenizer = None  # pyright: ignore[reportAttributeAccessIssue]  # skeleton: _tokenizer accepts None here
    session._layers = torch.nn.ModuleList(
        [_NoopModule(), _NoopModule(), _NoopModule()]
    )
    session._device = torch.device("cpu")
    session._dtype = torch.float32
    session._profiles = {}
    session._layer_means = {1: torch.zeros(3)}
    session._steering = SteeringManager()
    session._steering_composer = SteeringComposer(session)
    # v2.2: _push_steering / _pop_steering acquire _gen_lock; skeleton
    # mode never runs gen so the lock is uncontended, but the ``with
    # self._gen_lock:`` block needs the attribute to exist.
    session._gen_lock = threading.RLock()
    # Phase guard the push/pop methods read to reject callback
    # reentry — skeleton sessions are always idle.
    from saklas.core.session import GenState
    session._gen_phase = GenState.IDLE
    session._internal_steering_pop = False
    session._whitener = isotropic_whitener([1], 3)
    session._compiled = False
    session._compiled_clean_eligible = False
    cast(Any, session)._monitor = type("_Monitor", (), {"probe_names": []})()
    session.events = EventBus()
    session._history = []  # pyright: ignore[reportAttributeAccessIssue]  # skeleton: _history is dynamically set
    return session


def test_session_steering_dispatches_ablation_to_manager():
    session = _skeleton_session()
    session._profiles["refusal"] = {1: torch.tensor([1.0, 0.0, 0.0])}
    session._layer_means[1] = torch.tensor([0.5, 0.0, 0.0])

    steering = Steering(alphas={
        "!refusal": AblationTerm(coeff=1.0, trigger=Trigger.BOTH, target="refusal"),
    })
    with session.steering(steering):
        # 4.0: ``!`` lowers to an ablation axis (target 0) inside the merged
        # affine subspace — one ``add_subspace`` entry per trigger group, an
        # ``subspace_inject`` group on the covered layer.
        assert session._steering.subspaces
        synth = next(iter(session._steering.subspaces.values()))["synth"]
        assert 1 in synth.layers
        assert 1 in session._steering.hooks
        assert session._steering.hooks[1].manifold_groups

    # Post-exit: steering cleared, hook detached.
    assert not session._steering.subspaces
    assert not session._steering.hooks


def test_session_steering_ablation_missing_profile_raises():
    session = _skeleton_session()
    steering = Steering(alphas={
        "!nonexistent": AblationTerm(
            coeff=1.0, trigger=Trigger.BOTH, target="nonexistent",
        ),
    })
    with pytest.raises(ProfileNotRegisteredError):
        with session.steering(steering):
            pass


def test_session_steering_string_with_ablation_end_to_end():
    """session.steering('!refusal') string form parses through to the manager."""
    session = _skeleton_session()
    session._profiles["refusal"] = {1: torch.tensor([1.0, 0.0, 0.0])}
    session._layer_means[1] = torch.tensor([0.5, 0.0, 0.0])

    with session.steering("!refusal"):
        assert session._steering.subspaces
        synth = next(iter(session._steering.subspaces.values()))["synth"]
        assert 1 in synth.layers
        assert session._steering.hooks[1].manifold_groups

    assert not session._steering.subspaces


@pytest.mark.parametrize("coeff", [0.15, 1.0, -0.3])
def test_ablation_coefficient_survives_affine_gain(coeff: float) -> None:
    """The shared affine gain must not amplify mean-ablation coefficients."""
    session = _skeleton_session()
    session._profiles["refusal"] = {1: torch.tensor([1.0, 0.0, 0.0])}

    steering = Steering(alphas={
        "!refusal": AblationTerm(
            coeff=coeff, trigger=Trigger.BOTH, target="refusal",
        ),
    })
    with session.steering(steering):
        hook = session._steering.hooks[1]
        assert len(hook.manifold_groups) == 1
        group = hook.manifold_groups[0]
        along = float(group[5])
        kernel_kappa = group[7]
        assert isinstance(kernel_kappa, torch.Tensor)
        assert along * float(kernel_kappa[0]) == pytest.approx(coeff)


def test_zero_ablation_coefficient_is_a_true_noop() -> None:
    session = _skeleton_session()
    session._profiles["refusal"] = {1: torch.tensor([1.0, 0.0, 0.0])}

    steering = Steering(alphas={
        "!refusal": AblationTerm(
            coeff=0.0, trigger=Trigger.BOTH, target="refusal",
        ),
    })
    with session.steering(steering):
        assert not session._steering.subspaces
        assert not session._steering.hooks
