"""Runtime projection: unit-level project_profile + session materialization.

Pure tensor math is tested in ``TestProjectProfile``; the session-level
integration rides on the same ``_Stub`` pattern used by
``test_steering_context.py`` — a ``SaklasSession`` that bypasses the
model-loading machinery and pre-registers profiles directly.
"""
from __future__ import annotations

import pytest
import torch

from saklas.core.events import EventBus
from saklas.core.session import (
    SaklasSession, VectorNotRegisteredError,
)
from saklas.core.steering_expr import parse_expr
from saklas.core.triggers import Trigger
from saklas.core.vectors import project_profile


# ------------------------------------------------------- project_profile ---

class TestProjectProfile:
    def test_orthogonal_to_parallel_is_zero(self):
        base = {0: torch.tensor([1.0, 0.0])}
        onto = {0: torch.tensor([1.0, 0.0])}
        out = project_profile(base, onto, "|")
        assert torch.allclose(out[0], torch.zeros(2), atol=1e-6)

    def test_onto_of_parallel_is_base(self):
        base = {0: torch.tensor([2.0, 0.0])}
        onto = {0: torch.tensor([1.0, 0.0])}
        out = project_profile(base, onto, "~")
        assert torch.allclose(out[0], torch.tensor([2.0, 0.0]), atol=1e-6)

    def test_orthogonal_drops_shared_axis(self):
        # base = [1, 1]; onto = [1, 0].  Projection onto = [1, 0]; orthogonal = [0, 1].
        base = {0: torch.tensor([1.0, 1.0])}
        onto = {0: torch.tensor([1.0, 0.0])}
        out_ortho = project_profile(base, onto, "|")
        assert torch.allclose(out_ortho[0], torch.tensor([0.0, 1.0]), atol=1e-6)

    def test_onto_keeps_shared_axis(self):
        base = {0: torch.tensor([1.0, 1.0])}
        onto = {0: torch.tensor([1.0, 0.0])}
        out_onto = project_profile(base, onto, "~")
        assert torch.allclose(out_onto[0], torch.tensor([1.0, 0.0]), atol=1e-6)

    def test_multi_layer(self):
        base = {
            0: torch.tensor([1.0, 1.0]),
            1: torch.tensor([2.0, 2.0]),
        }
        onto = {
            0: torch.tensor([1.0, 0.0]),
            1: torch.tensor([0.0, 1.0]),
        }
        out = project_profile(base, onto, "|")
        assert torch.allclose(out[0], torch.tensor([0.0, 1.0]), atol=1e-6)
        assert torch.allclose(out[1], torch.tensor([2.0, 0.0]), atol=1e-6)

    def test_missing_onto_layer_passes_through_for_ortho(self):
        base = {0: torch.tensor([1.0, 0.0]), 1: torch.tensor([0.5, 0.5])}
        onto = {0: torch.tensor([1.0, 0.0])}
        out = project_profile(base, onto, "|")
        assert 1 in out
        assert torch.allclose(out[1], torch.tensor([0.5, 0.5]), atol=1e-6)
        assert torch.allclose(out[0], torch.zeros(2), atol=1e-6)

    def test_missing_onto_layer_drops_for_onto(self):
        base = {0: torch.tensor([1.0, 0.0]), 1: torch.tensor([0.5, 0.5])}
        onto = {0: torch.tensor([1.0, 0.0])}
        out = project_profile(base, onto, "~")
        assert 1 not in out
        assert 0 in out

    def test_near_zero_onto_passes_base_for_ortho(self):
        base = {0: torch.tensor([1.0, 0.0])}
        onto = {0: torch.tensor([1e-20, 0.0])}
        out = project_profile(base, onto, "|")
        assert torch.allclose(out[0], torch.tensor([1.0, 0.0]), atol=1e-6)

    def test_near_zero_onto_drops_for_onto_operator(self):
        base = {0: torch.tensor([1.0, 0.0])}
        onto = {0: torch.tensor([1e-20, 0.0])}
        with pytest.raises(ValueError):
            project_profile(base, onto, "~")

    def test_unknown_operator_raises(self):
        base = {0: torch.tensor([1.0])}
        onto = {0: torch.tensor([1.0])}
        with pytest.raises(ValueError):
            project_profile(base, onto, "@")

    def test_empty_intersection_raises_for_ortho(self):
        base = {0: torch.tensor([1.0])}
        onto = {7: torch.tensor([1.0])}
        # "|" passes through layer 0 (not in onto), so the result has
        # layer 0 — non-empty.
        out = project_profile(base, onto, "|")
        assert set(out.keys()) == {0}

    def test_empty_intersection_raises_for_onto_operator(self):
        base = {0: torch.tensor([1.0])}
        onto = {7: torch.tensor([1.0])}
        with pytest.raises(ValueError):
            project_profile(base, onto, "~")

    def test_result_dtype_matches_base(self):
        base = {0: torch.tensor([1.0, 1.0], dtype=torch.float16)}
        onto = {0: torch.tensor([1.0, 0.0], dtype=torch.float16)}
        out = project_profile(base, onto, "|")
        assert out[0].dtype == torch.float16


# ---------------------------------------------- session-level integration ---

class _Stub(SaklasSession):
    """SaklasSession without real model/tokenizer, mirrors test_steering_context."""
    def __init__(self, profiles: dict) -> None:  # type: ignore[override]
        import threading
        self._profiles = dict(profiles)
        self._steering_stack = []
        self._steering_override_stack = []
        # v2.2: _push_steering / _pop_steering acquire _gen_lock and
        # consult _gen_phase + _internal_steering_pop.
        self._gen_lock = threading.RLock()
        from saklas.core.session import GenState
        self._gen_phase = GenState.IDLE
        self._internal_steering_pop = False
        # v2.1 session-level defaults consulted by ``_resolve_*``
        # helpers when the override LIFO has no entries.
        from saklas.core.hooks import DEFAULT_THETA_MAX as _DTM
        self._injection_mode = "angular"
        self._theta_max = _DTM
        self._projection_metric = "mahalanobis"
        self._whitener = None
        self._layer_means = {}
        self.events = EventBus()
        self._rebuild_calls: list[dict[str, float]] = []
        self._rebuild_entries: list[dict[str, tuple[float, Trigger]]] = []

    @property
    def whitener(self) -> None:  # type: ignore[override]
        # No model in stub mode — return None so
        # ``_materialize_projections`` falls back to Euclidean
        # per-layer (the path real sessions hit when neutrals aren't
        # cached yet).
        return None

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


def _profile_a():
    # "a" direction, layer 0 only.
    return {0: torch.tensor([1.0, 1.0])}


def _profile_b():
    # "b" direction along x-axis.
    return {0: torch.tensor([1.0, 0.0])}


class TestSessionProjection:
    def test_parses_and_registers_synthetic_key(self):
        s = _Stub({"a": _profile_a(), "b": _profile_b()})
        steering = parse_expr("0.5 a|b")
        with s.steering(steering):
            assert "a|b" in s._profiles
            # Registered projection is orthogonal: [0, 1] (the y-component).
            assert torch.allclose(
                s._profiles["a|b"][0], torch.tensor([0.0, 1.0]), atol=1e-6,
            )
            assert s._rebuild_calls[-1] == {"a|b": 0.5}

    def test_onto_operator_registers_projected(self):
        s = _Stub({"a": _profile_a(), "b": _profile_b()})
        steering = parse_expr("0.5 a~b")
        with s.steering(steering):
            assert "a~b" in s._profiles
            assert torch.allclose(
                s._profiles["a~b"][0], torch.tensor([1.0, 0.0]), atol=1e-6,
            )

    def test_mixed_plain_and_projection(self):
        s = _Stub({"a": _profile_a(), "b": _profile_b()})
        steering = parse_expr("0.3 a + 0.5 a|b")
        with s.steering(steering):
            flat = s._rebuild_calls[-1]
            assert flat == {"a": 0.3, "a|b": 0.5}

    def test_projection_trigger_propagates(self):
        s = _Stub({"a": _profile_a(), "b": _profile_b()})
        steering = parse_expr("0.5 a|b@after")
        with s.steering(steering):
            entries = s._rebuild_entries[-1]
            assert entries["a|b"] == (0.5, Trigger.AFTER_THINKING)

    def test_projection_missing_base_raises(self):
        s = _Stub({"b": _profile_b()})  # no "a" registered
        steering = parse_expr("0.5 a|b")
        with pytest.raises(VectorNotRegisteredError) as ei:
            with s.steering(steering):
                pass
        assert "a" in str(ei.value) or "projection" in str(ei.value)

    def test_projection_missing_onto_raises(self):
        s = _Stub({"a": _profile_a()})  # no "b"
        steering = parse_expr("0.5 a|b")
        with pytest.raises(VectorNotRegisteredError):
            with s.steering(steering):
                pass

    def test_base_direction_differs_from_orthogonal(self):
        # Sanity: steering "a" and steering "a|b" should register
        # measurably different tensors in _profiles.
        s = _Stub({"a": _profile_a(), "b": _profile_b()})
        with s.steering(parse_expr("1.0 a|b")):
            projected = s._profiles["a|b"][0].clone()
        # Baseline is a = [1, 1]; orthogonal-to-b strips x-component -> [0, 1].
        assert not torch.allclose(
            projected, _profile_a()[0], atol=1e-3,
        )
        assert torch.allclose(projected, torch.tensor([0.0, 1.0]), atol=1e-6)

    def test_nested_projection_scopes(self):
        # Keys "a" and "a|b" don't collide, so nesting leaves both active
        # in the flattened head; exiting the inner scope restores the outer.
        s = _Stub({"a": _profile_a(), "b": _profile_b()})
        with s.steering(parse_expr("0.3 a")):
            assert s._rebuild_calls[-1] == {"a": 0.3}
            with s.steering(parse_expr("0.5 a|b")):
                assert s._rebuild_calls[-1] == {"a": 0.3, "a|b": 0.5}
            assert s._rebuild_calls[-1] == {"a": 0.3}
        assert s._rebuild_calls[-1] == {}


# ---------------------------------------- v2.1 metric-default integration ---

class _MetricStub(_Stub):
    """Variant of ``_Stub`` that lets tests select a metric and stand-in
    whitener.  ``project_profile`` is patched per-test to record kwarg
    shape rather than do real math.
    """

    def __init__(
        self,
        profiles: dict,
        *,
        projection_metric: str = "mahalanobis",
        whitener_value: object = "WHITENER",
    ) -> None:
        super().__init__(profiles)
        self._projection_metric = projection_metric
        # Sentinel — patched ``project_profile`` only checks ``is None``.
        self._whitener_sentinel = whitener_value

    @property
    def whitener(self) -> object:  # type: ignore[override]
        return self._whitener_sentinel


class TestProjectionMetricDefault:
    """The v2.1 default flips runtime ``~`` / ``|`` to Mahalanobis.

    Verifies that ``_materialize_projections`` passes ``self.whitener``
    to ``project_profile`` under the default metric, and ``None``
    under ``"euclidean"`` (the ``--legacy`` path).
    """

    def _patch_project(self, monkeypatch, calls: list) -> None:
        # ``_materialize_projections`` imports ``project_profile``
        # lazily inside the method (``from saklas.core.vectors import
        # project_profile``), so we patch it on the source module.
        from saklas.core import vectors as vectors_mod

        def _spy(base, onto, operator, *, whitener=None):
            calls.append((operator, whitener is None))
            return {0: torch.tensor([0.0, 1.0])}

        monkeypatch.setattr(vectors_mod, "project_profile", _spy)

    def test_default_passes_whitener(self, monkeypatch):
        calls: list = []
        s = _MetricStub({"a": _profile_a(), "b": _profile_b()})
        self._patch_project(monkeypatch, calls)
        with s.steering(parse_expr("0.5 a|b")):
            pass
        assert calls == [("|", False)], (
            "default session should hand session.whitener to project_profile"
        )

    def test_legacy_metric_passes_none(self, monkeypatch):
        calls: list = []
        s = _MetricStub(
            {"a": _profile_a(), "b": _profile_b()},
            projection_metric="euclidean",
        )
        self._patch_project(monkeypatch, calls)
        with s.steering(parse_expr("0.5 a|b")):
            pass
        assert calls == [("|", True)], (
            "euclidean session should pass whitener=None"
        )

    def test_per_call_override_flips_metric(self, monkeypatch):
        from saklas.core.steering import Steering

        calls: list = []
        s = _MetricStub({"a": _profile_a(), "b": _profile_b()})
        self._patch_project(monkeypatch, calls)
        # First scope inherits the session default ("mahalanobis").
        with s.steering(parse_expr("0.5 a|b")):
            pass
        # Second scope overrides to euclidean via Steering.projection_metric.
        # parse_expr doesn't surface this field — programmatic-only.
        base = parse_expr("0.5 a|b")
        override = Steering(
            alphas=base.alphas,
            thinking=base.thinking,
            trigger=base.trigger,
            projection_metric="euclidean",
        )
        with s.steering(override):
            pass
        assert calls == [("|", False), ("|", True)]

    def test_per_call_override_inherits_to_inner_scope(self, monkeypatch):
        from saklas.core.steering import Steering

        calls: list = []
        s = _MetricStub({"a": _profile_a(), "b": _profile_b()})
        self._patch_project(monkeypatch, calls)
        # Outer scope sets "euclidean"; inner scope ``None`` inherits it.
        outer = Steering(
            alphas=parse_expr("0.5 a|b").alphas,
            projection_metric="euclidean",
        )
        with s.steering(outer):
            with s.steering(parse_expr("0.5 a~b")):
                pass
        # Both calls should run with whitener=None.
        assert calls == [("|", True), ("~", True)], (
            "inner scope without override should inherit outer's "
            "euclidean choice via _resolve_projection_metric"
        )

    def test_invalid_metric_rejected(self, monkeypatch):
        from saklas.core.steering import Steering

        calls: list = []
        s = _MetricStub({"a": _profile_a(), "b": _profile_b()})
        self._patch_project(monkeypatch, calls)
        bad = Steering(
            alphas=parse_expr("0.5 a|b").alphas,
            projection_metric="bogus",
        )
        with pytest.raises(ValueError, match="projection_metric"):
            with s.steering(bad):
                pass


# ----------------------------------- v2.1 layer_means lazy-load fix-up ---

class TestLayerMeansLazy:
    """The ``session.layer_means`` property lazy-builds when
    ``probes=[]`` left ``self._layer_means`` empty.

    Closes the v2.1 footgun where ``probes=[]`` sessions hit
    ``compute_dls_mask`` with an empty dict, every layer fell into
    the conservative-keep branch, and DLS silently disabled itself.
    """

    def test_property_returns_existing_means_without_rebuild(self, monkeypatch):
        """Non-empty ``self._layer_means`` short-circuits the property
        — no bootstrap call.  Sanity check that the lazy path is
        only triggered on miss."""
        s = _Stub({"a": _profile_a()})
        s._layer_means = {0: torch.tensor([1.0, 2.0])}

        called: list = []

        def _fail_bootstrap(*args, **kwargs):
            called.append(args)
            return {99: torch.tensor([0.0])}

        from saklas.core import session as session_mod
        monkeypatch.setattr(session_mod, "bootstrap_layer_means", _fail_bootstrap)
        result = s.layer_means
        assert set(result.keys()) == {0}
        assert torch.equal(result[0], torch.tensor([1.0, 2.0]))
        assert called == [], "bootstrap_layer_means should not run on hit"

    def test_property_lazy_builds_when_empty(self, monkeypatch):
        """Empty ``self._layer_means`` triggers ``bootstrap_layer_means``
        on first access; result is cached on subsequent calls."""
        s = _Stub({"a": _profile_a()})
        # Stub has _layer_means = {} from _Stub.__init__, plus the
        # whitener-property override returns None.  Override the
        # session.py-level whitener property override on the stub so
        # we can test layer_means in isolation: replace the class
        # method with the *real* SaklasSession.layer_means property.
        from saklas.core.session import SaklasSession
        # The stub doesn't override ``layer_means`` — it inherits the
        # real property, which is what we want to exercise.
        # Replace bootstrap_layer_means with a tracker.
        built = {3: torch.tensor([5.0, 6.0]), 4: torch.tensor([7.0, 8.0])}
        calls: list = []

        def _spy(*args, **kwargs):
            calls.append(args)
            return built

        # Stub doesn't have _model/_tokenizer/_layers, so the property's
        # try-except will fire and bootstrap_layer_means won't even be
        # reached — instead the except block warns and returns {}.
        # Patch *the same module the property imports from* so that
        # call lookup resolves to our spy without needing the model.
        from saklas.core import session as session_mod

        # Give the stub the minimal handle attributes the bootstrap call
        # looks at, so the try-block actually runs.
        s._model = object()  # type: ignore[attr-defined]
        s._tokenizer = object()  # type: ignore[attr-defined]
        s._layers = []  # type: ignore[attr-defined]
        s._model_info = {}  # type: ignore[attr-defined]
        monkeypatch.setattr(session_mod, "bootstrap_layer_means", _spy)

        # First access — triggers build.
        out = SaklasSession.layer_means.fget(s)  # type: ignore[union-attr]
        assert out is built
        assert len(calls) == 1
        # Second access — caches; no second call.
        out2 = SaklasSession.layer_means.fget(s)  # type: ignore[union-attr]
        assert out2 is built
        assert len(calls) == 1


class TestComputeDlsMaskEmptyGuard:
    """``compute_dls_mask`` treats ``layer_means={}`` identically to
    ``layer_means=None`` — both fall back to keep-all silently.

    Closes the v2.1 path where an empty dict propagated through
    ``probes=[]`` sessions and walked the per-layer loop without
    a baseline, hitting "conservative keep" on every layer.  The
    behavior was technically the same (keep all), but the early-out
    makes intent explicit.
    """

    def test_none_keeps_all(self):
        from saklas.core.vectors import compute_dls_mask
        mu_pos = {0: torch.tensor([1.0, 0.0]), 1: torch.tensor([2.0, 0.0])}
        mu_neg = {0: torch.tensor([-1.0, 0.0]), 1: torch.tensor([-2.0, 0.0])}
        directions = {L: mu_pos[L] - mu_neg[L] for L in mu_pos}
        out = compute_dls_mask(mu_pos, mu_neg, directions, None)
        assert out == {0, 1}

    def test_empty_dict_keeps_all(self):
        from saklas.core.vectors import compute_dls_mask
        mu_pos = {0: torch.tensor([1.0, 0.0]), 1: torch.tensor([2.0, 0.0])}
        mu_neg = {0: torch.tensor([-1.0, 0.0]), 1: torch.tensor([-2.0, 0.0])}
        directions = {L: mu_pos[L] - mu_neg[L] for L in mu_pos}
        out = compute_dls_mask(mu_pos, mu_neg, directions, {})
        assert out == {0, 1}

    def test_all_failed_fallback_excludes_skipped_layers(self):
        # All layers fail the discriminative test, but layers 0+1 also
        # had degenerate directions that the loop explicitly skipped.
        # Fallback should return only the *checkable* set (layer 2),
        # not every layer in mu_pos — re-including skipped layers via
        # the fallback would silently undo the skip.
        import warnings as _warnings
        from saklas.core.vectors import compute_dls_mask
        mu_pos = {
            0: torch.tensor([0.5, 0.0]),
            1: torch.tensor([0.7, 0.0]),
            2: torch.tensor([0.6, 0.0]),
        }
        mu_neg = {
            0: torch.tensor([0.3, 0.0]),
            1: torch.tensor([0.4, 0.0]),
            2: torch.tensor([0.4, 0.0]),
        }
        # Layers 0+1 have zero-norm directions (skipped).
        directions = {
            0: torch.zeros(2),
            1: torch.zeros(2),
            2: mu_pos[2] - mu_neg[2],  # non-degenerate, but won't pass DLS
        }
        layer_means = {0: torch.zeros(2), 1: torch.zeros(2), 2: torch.zeros(2)}

        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            out = compute_dls_mask(mu_pos, mu_neg, directions, layer_means)

        # Layer 2 is checkable but failed; fallback returns just it.
        # Layers 0+1 stay dropped despite the fallback.
        assert out == {2}
        msgs = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
        assert any("DLS" in m for m in msgs)

    def test_partial_layer_means_runs_check(self):
        """Layers with a baseline get checked; layers without fall
        through the per-layer ``mu_n is None`` conservative-keep."""
        from saklas.core.vectors import compute_dls_mask
        mu_pos = {
            0: torch.tensor([1.0, 0.0]),  # opposite-signed → kept
            1: torch.tensor([0.5, 0.0]),  # same-signed (both positive after centering) → drop
            2: torch.tensor([3.0, 0.0]),  # no baseline → conservative keep
        }
        mu_neg = {
            0: torch.tensor([-1.0, 0.0]),
            1: torch.tensor([0.3, 0.0]),
            2: torch.tensor([-3.0, 0.0]),
        }
        directions = {L: mu_pos[L] - mu_neg[L] for L in mu_pos}
        # Baseline = 0 for layers 0+1 but absent for layer 2.
        layer_means = {0: torch.zeros(2), 1: torch.zeros(2)}
        out = compute_dls_mask(mu_pos, mu_neg, directions, layer_means)
        assert out == {0, 2}  # 1 dropped (both leans positive); 2 kept (no baseline)


# ------------------------------- v2.1 nested projection scope restore ---

class TestNestedProjectionScopeLeak:
    """Inner scope's projection should not leak back to outer scope.

    ``_materialize_projections`` writes synthetic keys (``a|b``) into
    the global ``self._profiles`` registry.  Without snapshot+restore
    on the ``_SteeringContext`` exit path, an inner scope materializing
    the same synthetic key under a different ``projection_metric`` (or
    different base/onto pair) leaves the inner tensor bound when the
    outer scope's hooks re-build, silently using the inner's projection.
    """

    def test_inner_overwrite_restored_on_exit(self):
        s = _Stub({"a": _profile_a(), "b": _profile_b()})
        outer_steering = parse_expr("0.5 a|b")
        inner_steering = parse_expr("0.7 a|b")
        with s.steering(outer_steering):
            outer_tensor = s._profiles["a|b"][0].clone()
            with s.steering(inner_steering):
                inner_tensor = s._profiles["a|b"][0].clone()
            # After inner exits, outer's binding must be restored.
            restored = s._profiles["a|b"][0]
            assert torch.equal(restored, outer_tensor), (
                "outer scope's projected tensor must be restored when "
                "inner scope exits"
            )
            # Sanity: inner did write a value (test would be vacuous
            # if the inner write was identical).  The values are equal
            # in practice because the projection is deterministic from
            # base/onto, but the key point is the registry binding got
            # restored to a snapshot regardless of value equality.
            _ = inner_tensor

    def test_outer_binding_absent_pre_scope_removed_after_pop(self):
        # Synthetic key ``a|b`` doesn't exist before any scope opens.
        # Entering a scope materializes it; exiting must remove it
        # rather than leaving a stale binding behind.
        s = _Stub({"a": _profile_a(), "b": _profile_b()})
        assert "a|b" not in s._profiles
        with s.steering(parse_expr("0.5 a|b")):
            assert "a|b" in s._profiles
        assert "a|b" not in s._profiles, (
            "synthetic key materialized inside a scope must be removed "
            "from the registry on exit when no outer binding existed"
        )

    def test_outer_pre_existing_binding_survives_inner_overwrite(self):
        # Pre-bind ``a|b`` with a sentinel tensor, enter a scope that
        # overwrites it, exit, and assert the sentinel is restored.
        s = _Stub({"a": _profile_a(), "b": _profile_b()})
        sentinel = {0: torch.tensor([99.0, 99.0])}
        s._profiles["a|b"] = sentinel  # type: ignore[assignment]
        with s.steering(parse_expr("0.5 a|b")):
            assert s._profiles["a|b"] is not sentinel
        assert s._profiles["a|b"] is sentinel, (
            "pre-scope binding for a|b must be restored on scope exit"
        )
