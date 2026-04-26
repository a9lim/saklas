"""session.steering() context-manager semantics — stack flattening, events.

Model-loading is avoided by constructing a SaklasSession stub that only wires
up the pieces the context manager touches.  Hook installation is stubbed out
so nested enters/exits just twiddle the stack and fire events.

Every ``session.steering()`` call uses the unified expression grammar; dict
input is not accepted anywhere in the stack.
"""

from __future__ import annotations

import pytest

from saklas.io import selectors as _sel
from saklas.core.events import EventBus, SteeringApplied, SteeringCleared
from saklas.core.session import SaklasSession, VectorNotRegisteredError
from saklas.core.steering import Steering
from saklas.core.triggers import Trigger


@pytest.fixture(autouse=True)
def _isolated_home(monkeypatch, tmp_path):
    """Keep parser pole-resolution from scanning the user's real vectors dir."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    _sel.invalidate()
    yield
    _sel.invalidate()


class _Stub(SaklasSession):
    """Construct a session without touching any model/tokenizer machinery."""

    def __init__(self, profiles: dict) -> None:  # type: ignore[override]
        self._profiles = dict(profiles)
        self._steering_stack = []
        self.events = EventBus()
        self._rebuild_calls: list[dict[str, float]] = []
        self._rebuild_entries: list[dict[str, tuple[float, Trigger]]] = []

    def _rebuild_steering_hooks(self) -> None:  # type: ignore[override]
        flat = self._flatten_steering_stack()
        for name in flat:
            if name not in self._profiles:
                raise VectorNotRegisteredError(f"No vector registered for '{name}'")
        self._rebuild_entries.append(dict(flat))
        self._rebuild_calls.append(
            {name: alpha for name, (alpha, _trig) in flat.items()},
        )

    def _resolve_pole_aliases(self, entries):  # type: ignore[override]
        return {k: (float(v[0]), v[1]) for k, v in entries.items()}


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


def test_pole_alias_sign_flip_preserves_trigger(tmp_path, monkeypatch):
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


def test_autoload_cache_hit_registers_bundled_vector(monkeypatch, tmp_path):
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
        def __init__(self):  # type: ignore[override]
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
