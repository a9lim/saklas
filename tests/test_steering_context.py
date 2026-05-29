"""session.steering() context-manager semantics — stack flattening, events.

Model-loading is avoided by constructing a SaklasSession stub that only wires
up the pieces the context manager touches.  Hook installation is stubbed out
so nested enters/exits just twiddle the stack and fire events.

Every ``session.steering()`` call uses the unified expression grammar; dict
input is not accepted anywhere in the stack.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Generator

import pytest

from saklas.io import selectors as _sel
from saklas.core.events import EventBus, SteeringApplied, SteeringCleared
from saklas.core.session import SaklasSession, VectorNotRegisteredError
from saklas.core.steering import Steering
from saklas.core.triggers import Trigger


@pytest.fixture(autouse=True)
def _isolated_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Generator[None, None, None]:
    """Keep parser pole-resolution from scanning the user's real vectors dir."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    _sel.invalidate()
    yield
    _sel.invalidate()


class _Stub(SaklasSession):
    """Construct a session without touching any model/tokenizer machinery."""

    def __init__(self, profiles: dict) -> None:  # pyright: ignore[reportMissingTypeArgument]  # bare dict; stub doesn't constrain key/value types
        import threading
        self._profiles = dict(profiles)
        self._steering_stack = []
        self._steering_override_stack = []
        # Reentrant gen lock — _push_steering / _pop_steering acquire it
        # so out-of-band steering scope mutations don't race a mid-step
        # rebuild during generation (v2.2 fix).  Stub mode never runs
        # generation, so the lock is uncontended; we still need it to
        # exist as an attribute for the ``with self._gen_lock:`` block
        # to bind.
        self._gen_lock = threading.RLock()
        # Phase guard the push/pop methods consult to reject callback
        # reentry mid-gen — stubs are always idle.
        from saklas.core.session import GenState
        self._gen_phase = GenState.IDLE
        # Internal-cleanup bypass for the phase guard; stubs never run
        # gen so it stays False.
        self._internal_steering_pop = False
        # Session-level defaults consulted by _resolve_steering_override
        # and _resolve_projection_metric when the LIFO has no overrides.
        # Mirror the v2.1 SaklasSession defaults so the stub flows through
        # the same code paths the real session uses.
        self._injection_mode = "angular"
        from saklas.core.hooks import DEFAULT_THETA_MAX
        self._theta_max = DEFAULT_THETA_MAX
        self._projection_metric = "mahalanobis"
        # No whitener in stub mode — _materialize_projections silently
        # falls back to Euclidean per-layer when whitener is None.  Stub
        # the lazy ``whitener`` property to skip the model-dependent
        # neutral-activation build path.
        self._whitener = None
        self._layer_means = {}
        # Active assistant role (role-extraction Phase 7) — populated by
        # the steering scope's __enter__ when role-augmented terms agree
        # on a role, restored to None on exit.
        self._active_role: str | None = None
        self.events = EventBus()
        self._rebuild_calls: list[dict[str, float]] = []
        # SteeringStackEntry is tuple[float, Trigger] | AblationTerm | ManifoldTerm;
        # use Any so the append below accepts the full union without type errors.
        self._rebuild_entries: list[dict[str, Any]] = []

    def _rebuild_steering_hooks(self) -> None:
        flat = self._flatten_steering_stack()
        for name in flat:
            if name not in self._profiles:
                raise VectorNotRegisteredError(f"No vector registered for '{name}'")
        self._rebuild_entries.append(dict(flat))
        # The stub only registers plain (alpha, Trigger) entries; cast away the
        # union so pyright does not flag AblationTerm / ManifoldTerm as non-iterable.
        flat_any: dict[str, Any] = flat
        self._rebuild_calls.append(
            {name: alpha for name, (alpha, _trig) in flat_any.items()},
        )

    def _resolve_pole_aliases(self, entries):  # pyright: ignore[reportMissingParameterType]  # intentionally untyped stub override
        return {k: (float(v[0]), v[1]) for k, v in entries.items()}

    # Override the lazy whitener property so tests stay model-free —
    # the stub doesn't have a model/tokenizer to feed
    # ``_build_whitener_from_cache_or_compute``.  Returns ``None`` so
    # ``_materialize_projections`` falls back to Euclidean per-layer
    # transparently (same path real sessions hit when neutral
    # activations aren't cached yet).
    @property
    def whitener(self) -> None:
        return None


def test_single_scope_push_pop():
    s = _Stub({"angry.calm": None})
    events = []
    s.events.subscribe(events.append)
    with s.steering("0.5 angry.calm"):
        assert s._steering_stack == [{"angry.calm": (0.5, Trigger.BOTH)}]
    assert s._steering_stack == []
    assert len(s._rebuild_calls) == 2
    assert s._rebuild_calls[0] == {"angry.calm": 0.5}
    assert s._rebuild_calls[1] == {}
    kinds = [type(e).__name__ for e in events]
    assert kinds == ["SteeringApplied", "SteeringCleared"]


def test_nested_flattens_inner_wins():
    s = _Stub({"a": None, "b": None})
    with s.steering("0.3 a"):
        with s.steering("0.5 a + 0.1 b"):
            assert s._rebuild_calls[-1] == {"a": 0.5, "b": 0.1}
        assert s._rebuild_calls[-1] == {"a": 0.3}
    assert s._rebuild_calls[-1] == {}


def test_steering_accepts_steering_instance():
    s = _Stub({"a": None})
    with s.steering(Steering(alphas={"a": 0.2})):
        assert s._rebuild_calls[-1] == {"a": 0.2}


def test_unknown_vector_raises_on_enter():
    s = _Stub({"known": None})
    with pytest.raises(VectorNotRegisteredError):
        with s.steering("0.5 unknown"):
            pass
    # _push_steering rolls its entry back on rebuild failure, so the stack
    # is empty after a failed __enter__ and no SteeringApplied event fired.
    assert s._steering_stack == []

    events = []
    s2 = _Stub({"known": None})
    s2.events.subscribe(events.append)
    with pytest.raises(VectorNotRegisteredError):
        with s2.steering("0.5 unknown"):
            pass
    assert s2._steering_stack == []
    assert events == []


def test_failed_enter_under_outer_scope_preserves_outer():
    """An inner failed enter must not pop the outer scope's entry."""
    s = _Stub({"a": None})
    with s.steering("0.3 a"):
        with pytest.raises(VectorNotRegisteredError):
            with s.steering("0.5 unknown"):
                pass
        assert s._steering_stack == [{"a": (0.3, Trigger.BOTH)}]
        assert s._rebuild_calls[-1] == {"a": 0.3}
    assert s._steering_stack == []


def test_events_reflect_flattened_head():
    s = _Stub({"a": None, "b": None})
    events = []
    s.events.subscribe(events.append)
    with s.steering("0.3 a"):
        with s.steering("0.1 b"):
            pass
    applied = [e for e in events if isinstance(e, SteeringApplied)]
    cleared = [e for e in events if isinstance(e, SteeringCleared)]
    assert len(applied) == 3
    assert len(cleared) == 1
    assert applied[0].alphas == {"a": 0.3}
    assert applied[1].alphas == {"a": 0.3, "b": 0.1}
    assert applied[2].alphas == {"a": 0.3}


def test_steering_with_global_trigger_preserved_in_stack():
    """Steering(trigger=...) default flows through to the stack entries."""
    s = _Stub({"a": None})
    with s.steering(Steering(alphas={"a": 0.3}, trigger=Trigger.AFTER_THINKING)):
        assert s._steering_stack == [{"a": (0.3, Trigger.AFTER_THINKING)}]
    assert s._steering_stack == []


def test_steering_per_entry_trigger_preserved_in_stack():
    """Expression ``@after`` syntax attaches a per-entry trigger."""
    s = _Stub({"a": None, "b": None})
    with s.steering("0.3 a + 0.4 b@thinking"):
        entries = s._steering_stack[0]
        assert entries["a"] == (0.3, Trigger.BOTH)
        assert entries["b"] == (0.4, Trigger.THINKING_ONLY)


def test_nested_trigger_regimes_compose():
    """Nested steering scopes with distinct triggers flatten inner-wins."""
    s = _Stub({"a": None, "b": None})
    with s.steering(Steering(alphas={"a": 0.3}, trigger=Trigger.BOTH)):
        with s.steering(Steering(alphas={"b": 0.5}, trigger=Trigger.AFTER_THINKING)):
            inner = s._rebuild_entries[-1]
            assert inner["a"] == (0.3, Trigger.BOTH)
            assert inner["b"] == (0.5, Trigger.AFTER_THINKING)
        outer = s._rebuild_entries[-1]
        assert outer == {"a": (0.3, Trigger.BOTH)}


def test_steering_applied_event_carries_entries():
    s = _Stub({"a": None, "b": None})
    events = []
    s.events.subscribe(events.append)
    with s.steering("0.3 a"):
        applied = [e for e in events if isinstance(e, SteeringApplied)][-1]
        assert applied.alphas == {"a": 0.3}
        assert applied.entries == {"a": (0.3, Trigger.BOTH)}
    events.clear()
    with s.steering("0.3 a + 0.5 b@after"):
        applied = [e for e in events if isinstance(e, SteeringApplied)][-1]
        assert applied.alphas == {"a": 0.3, "b": 0.5}
        assert applied.entries["a"] == (0.3, Trigger.BOTH)
        assert applied.entries["b"] == (0.5, Trigger.AFTER_THINKING)


def test_pole_alias_sign_flip_preserves_trigger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Bare-pole alias ``wolf`` resolves to ``deer.wolf`` with sign -1;
    the trigger survives the sign flip through the grammar path."""
    from saklas.io import packs
    wolf_dir = tmp_path / "vectors" / "default" / "deer.wolf"
    wolf_dir.mkdir(parents=True)
    (wolf_dir / "statements.json").write_text("[]")
    meta = packs.PackMetadata(
        name="deer.wolf", description="x", version="1.0.0", license="MIT",
        tags=[], recommended_alpha=0.5, source="local",
        files={"statements.json": packs.hash_file(wolf_dir / "statements.json")},
    )
    meta.write(wolf_dir)
    _sel.invalidate()

    s = _Stub({"deer.wolf": None})
    with s.steering("0.4 wolf@after"):
        entries = s._steering_stack[0]
        assert entries["deer.wolf"] == (-0.4, Trigger.AFTER_THINKING)


def test_autoload_cache_hit_registers_bundled_vector(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """`_try_autoload_vector` is the cache-hit fast path used by the server
    route to let HTTP clients steer bundled concepts without a prior
    `POST /vectors`. Exercises the concept-scan + safetensors-load path
    against a synthetic fake concept folder."""
    import torch
    from saklas.core.session import SaklasSession
    from saklas.io import paths as io_paths
    from saklas.io import selectors as _sel_local
    _sel_local.invalidate()
    ns_dir = tmp_path / "vectors" / "local" / "myprobe"
    ns_dir.mkdir(parents=True)
    (ns_dir / "pack.json").write_text(
        '{"name":"myprobe","description":"test",'
        '"method":"contrastive_pca","tags":[],"files":{},'
        '"format_version":2,"version":"1.0.0","license":"unknown",'
        '"recommended_alpha":0.5,"long_description":"","source":"local"}'
    )
    from saklas.core.vectors import save_profile
    sid = io_paths.safe_model_id("fake/model")
    save_profile(
        {0: torch.randn(8), 1: torch.randn(8)},
        str(ns_dir / f"{sid}.safetensors"),
        metadata={"method": "test", "format_version": 2},
    )

    class _AutoloadStub(SaklasSession):
        def __init__(self):
            self._profiles = {}
            self._model_info = {"model_id": "fake/model"}
            self._device = torch.device("cpu")
            self._dtype = torch.float32

    s = _AutoloadStub()
    assert "myprobe" not in s._profiles
    s._try_autoload_vector("myprobe")
    assert "myprobe" in s._profiles
    assert 0 in s._profiles["myprobe"] and 1 in s._profiles["myprobe"]
    s._try_autoload_vector("no_such_concept")
    assert "no_such_concept" not in s._profiles


# ---------------------------------------------------------------------------
# Role-augmented steering (role-extraction Phase 7)
# ---------------------------------------------------------------------------


class TestRoleUnanimity:
    """``session.steering()`` enforces that every role-tagged term in one
    scope agrees on a single role, and surfaces the union to
    ``session._active_role`` so the generation prompt can splice the
    matching assistant-role label.
    """

    def test_role_unanimity_violation(self):
        from saklas.core.steering_expr import SteeringExprError
        s = _Stub({
            "honest.deceptive:role-pirate": None,
            "honest.deceptive:role-sage": None,
        })
        with pytest.raises(SteeringExprError) as info:
            with s.steering(
                "0.5 honest.deceptive:role-pirate "
                "+ 0.3 honest.deceptive:role-sage",
            ):
                pass
        # The error message names both conflicting roles so the user
        # doesn't have to guess which term they need to drop.
        msg = str(info.value)
        assert "pirate" in msg
        assert "sage" in msg

    def test_role_unanimity_same_role_ok(self):
        s = _Stub({
            "honest.deceptive:role-pirate": None,
            "angry.calm:role-pirate": None,
        })
        with s.steering(
            "0.5 honest.deceptive:role-pirate "
            "+ 0.3 angry.calm:role-pirate",
        ):
            # Active role is the shared label.
            assert s._active_role == "pirate"
            # Stack landed both entries — no parser rejection on the
            # unanimous-role path.
            entries = s._steering_stack[0]
            assert "honest.deceptive:role-pirate" in entries
            assert "angry.calm:role-pirate" in entries
        # Restored to no-role on scope exit.
        assert s._active_role is None

    def test_plain_and_role_mixing_warns(self):
        import warnings
        from saklas.core.errors import RoleBaselineMismatchWarning
        s = _Stub({
            "honest.deceptive": None,
            "angry.calm:role-pirate": None,
        })
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with s.steering(
                "0.5 honest.deceptive + 0.3 angry.calm:role-pirate",
            ):
                # Active role still reflects the role-tagged term.
                assert s._active_role == "pirate"
            mismatch = [
                w for w in caught
                if issubclass(w.category, RoleBaselineMismatchWarning)
            ]
            assert len(mismatch) >= 1
            assert "pirate" in str(mismatch[0].message)
        assert s._active_role is None

    def test_role_only_in_expression_inherits(self):
        """A pure role expression (no plain terms) lifts the role without
        warning."""
        import warnings
        from saklas.core.errors import RoleBaselineMismatchWarning
        s = _Stub({"honest.deceptive:role-pirate": None})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with s.steering("0.5 honest.deceptive:role-pirate"):
                assert s._active_role == "pirate"
            assert not [
                w for w in caught
                if issubclass(w.category, RoleBaselineMismatchWarning)
            ]
        assert s._active_role is None

    def test_nested_role_scopes_inner_wins(self):
        """An inner role scope overrides the outer's active_role for the
        inner scope only; the outer is restored on inner exit."""
        s = _Stub({
            "honest.deceptive:role-pirate": None,
            "angry.calm:role-sage": None,
        })
        with s.steering("0.5 honest.deceptive:role-pirate"):
            assert s._active_role == "pirate"
            with s.steering("0.3 angry.calm:role-sage"):
                assert s._active_role == "sage"
            assert s._active_role == "pirate"
        assert s._active_role is None
