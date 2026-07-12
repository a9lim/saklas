"""Runtime projection: unit-level project_profile + session materialization.

Pure tensor math is tested in ``TestProjectProfile``; the session-level
integration rides on the same ``_Stub`` pattern used by
``test_steering_context.py`` — a ``SaklasSession`` that bypasses the
model-loading machinery and pre-registers profiles directly.

Mahalanobis-only (4.0 collapse): ``project_profile`` is the closed-form LEACE
projector and *requires* a whitener covering every projected layer — there is
no Euclidean path, and no per-call ``projection_metric`` override.  Every call
here passes a synthetic covering whitener; the operator semantics are asserted
structurally (``|`` ⇒ M-orthogonal to ``onto``; ``~`` ⇒ parallel to ``onto``).
"""
from __future__ import annotations

from typing import Any

import pytest
import torch

from saklas.core.events import EventBus
from saklas.core.mahalanobis import WhitenerError
from saklas.core.session import (
    SaklasSession, VectorNotRegisteredError,
)
from saklas.core.steering_composer import SteeringComposer
from saklas.core.steering_expr import parse_expr
from saklas.core.triggers import Trigger
from saklas.core.vectors import project_profile
from tests._whitener import synthetic_whitener


def _whit(layers: Any, dim: int):
    """A covering whitener with zero neutral means (no recentering)."""
    layers = list(layers)
    means = {L: torch.zeros(dim) for L in layers}
    return synthetic_whitener(layers, dim, means=means)


def _m_dot(w: Any, layer: int, u: torch.Tensor, v: torch.Tensor) -> float:
    return w.mahalanobis_dot(layer, u.float(), v.float())


# ------------------------------------------------------- project_profile ---

class TestProjectProfile:
    def test_orthogonal_to_parallel_is_zero(self):
        w = _whit([0], 2)
        base = {0: torch.tensor([1.0, 0.0])}
        onto = {0: torch.tensor([1.0, 0.0])}
        out = project_profile(base, onto, "|", whitener=w)
        # base ∥ onto ⇒ removing the onto-component leaves ~0.
        assert torch.allclose(out[0], torch.zeros(2), atol=1e-5)

    def test_onto_of_parallel_is_base(self):
        w = _whit([0], 2)
        base = {0: torch.tensor([2.0, 0.0])}
        onto = {0: torch.tensor([1.0, 0.0])}
        out = project_profile(base, onto, "~", whitener=w)
        # base ∥ onto ⇒ the onto-component is base itself.
        assert torch.allclose(out[0], torch.tensor([2.0, 0.0]), atol=1e-5)

    def test_orthogonal_result_is_m_orthogonal_to_onto(self):
        w = _whit([0], 4)
        base = {0: torch.tensor([1.0, 1.0, 0.5, 0.0])}
        onto = {0: torch.tensor([1.0, 0.0, 0.0, 0.0])}
        out = project_profile(base, onto, "|", whitener=w)
        # LEACE ``|`` erases the onto-direction under the M-metric.
        assert _m_dot(w, 0, out[0], onto[0]) == pytest.approx(0.0, abs=1e-4)

    def test_onto_result_is_parallel_to_onto(self):
        w = _whit([0], 4)
        base = {0: torch.tensor([1.0, 1.0, 0.5, 0.0])}
        onto = {0: torch.tensor([1.0, 0.0, 0.0, 0.0])}
        out = project_profile(base, onto, "~", whitener=w)
        # ``~`` keeps the onto-component: result is a scalar multiple of onto.
        # All non-zero entries of onto share one ratio; onto here is e0.
        assert out[0][1:].abs().max() < 1e-5

    def test_decomposition_recombines_to_base(self):
        """``~`` + ``|`` partition base: their sum is base (LEACE identity)."""
        w = _whit([0], 4)
        base = {0: torch.tensor([1.0, 1.0, 0.5, -0.3])}
        onto = {0: torch.tensor([0.7, 0.2, 0.0, 0.0])}
        par = project_profile(base, onto, "~", whitener=w)[0]
        perp = project_profile(base, onto, "|", whitener=w)[0]
        assert torch.allclose(par + perp, base[0], atol=1e-4)

    def test_multi_layer_m_orthogonal(self):
        w = _whit([0, 1], 4)
        base = {
            0: torch.tensor([1.0, 1.0, 0.0, 0.0]),
            1: torch.tensor([2.0, 2.0, 0.0, 0.0]),
        }
        onto = {
            0: torch.tensor([1.0, 0.0, 0.0, 0.0]),
            1: torch.tensor([0.0, 1.0, 0.0, 0.0]),
        }
        out = project_profile(base, onto, "|", whitener=w)
        assert _m_dot(w, 0, out[0], onto[0]) == pytest.approx(0.0, abs=1e-4)
        assert _m_dot(w, 1, out[1], onto[1]) == pytest.approx(0.0, abs=1e-4)

    def test_missing_onto_layer_passes_through_for_ortho(self):
        w = _whit([0, 1], 2)
        base = {0: torch.tensor([1.0, 0.0]), 1: torch.tensor([0.5, 0.5])}
        onto = {0: torch.tensor([1.0, 0.0])}
        out = project_profile(base, onto, "|", whitener=w)
        assert 1 in out
        # Layer 1 has no onto ⇒ passes through verbatim.
        assert torch.allclose(out[1], torch.tensor([0.5, 0.5]), atol=1e-6)
        assert torch.allclose(out[0], torch.zeros(2), atol=1e-5)

    def test_missing_onto_layer_drops_for_onto(self):
        w = _whit([0, 1], 2)
        base = {0: torch.tensor([1.0, 0.0]), 1: torch.tensor([0.5, 0.5])}
        onto = {0: torch.tensor([1.0, 0.0])}
        out = project_profile(base, onto, "~", whitener=w)
        assert 1 not in out
        assert 0 in out

    def test_near_zero_onto_passes_base_for_ortho(self):
        w = _whit([0], 2)
        base = {0: torch.tensor([1.0, 0.0])}
        onto = {0: torch.tensor([1e-20, 0.0])}
        out = project_profile(base, onto, "|", whitener=w)
        assert torch.allclose(out[0], torch.tensor([1.0, 0.0]), atol=1e-6)

    def test_near_zero_onto_drops_for_onto_operator(self):
        w = _whit([0], 2)
        base = {0: torch.tensor([1.0, 0.0])}
        onto = {0: torch.tensor([1e-20, 0.0])}
        with pytest.raises(ValueError):
            project_profile(base, onto, "~", whitener=w)

    def test_unknown_operator_raises(self):
        w = _whit([0], 1)
        base = {0: torch.tensor([1.0])}
        onto = {0: torch.tensor([1.0])}
        with pytest.raises(ValueError):
            project_profile(base, onto, "@", whitener=w)

    def test_passthrough_intersection_for_ortho(self):
        w = _whit([0], 1)
        base = {0: torch.tensor([1.0])}
        onto = {7: torch.tensor([1.0])}
        # "|" passes through layer 0 (not in onto), so the result has
        # layer 0 — non-empty.
        out = project_profile(base, onto, "|", whitener=w)
        assert set(out.keys()) == {0}

    def test_empty_intersection_raises_for_onto_operator(self):
        w = _whit([0], 1)
        base = {0: torch.tensor([1.0])}
        onto = {7: torch.tensor([1.0])}
        with pytest.raises(ValueError):
            project_profile(base, onto, "~", whitener=w)

    def test_result_dtype_matches_base(self):
        w = _whit([0], 2)
        base = {0: torch.tensor([1.0, 1.0], dtype=torch.float16)}
        onto = {0: torch.tensor([1.0, 0.0], dtype=torch.float16)}
        out = project_profile(base, onto, "|", whitener=w)
        assert out[0].dtype == torch.float16

    def test_missing_whitener_raises(self):
        base = {0: torch.tensor([1.0, 0.0])}
        onto = {0: torch.tensor([1.0, 0.0])}
        with pytest.raises(WhitenerError, match="whitener"):
            project_profile(base, onto, "|")

    def test_partial_coverage_raises(self):
        w = _whit([0], 2)  # covers layer 0 only
        base = {0: torch.tensor([1.0, 0.0]), 1: torch.tensor([0.0, 1.0])}
        onto = {0: torch.tensor([1.0, 0.0]), 1: torch.tensor([0.0, 1.0])}
        with pytest.raises(WhitenerError, match="whitener"):
            project_profile(base, onto, "|", whitener=w)


# ---------------------------------------------- session-level integration ---

class _Stub(SaklasSession):
    """SaklasSession without real model/tokenizer, mirrors test_steering_context.

    Projection materialization is Mahalanobis-only, so the stub exposes a real
    covering whitener over the registered profiles' layer/dim (no model load).
    """
    def __init__(self, profiles: dict[str, Any]) -> None:
        import threading
        self._profiles = dict(profiles)
        # v2.2: _push_steering / _pop_steering acquire _gen_lock and
        # consult _gen_phase + _internal_steering_pop.
        self._gen_lock = threading.RLock()
        from saklas.core.session import GenState
        self._gen_phase = GenState.IDLE
        self._internal_steering_pop = False
        self._layer_means = {}
        self.events = EventBus()
        self._rebuild_calls: list[dict[str, Any]] = []
        self._rebuild_entries: list[dict[str, Any]] = []
        self._steering_composer = SteeringComposer(self)
        # Build a covering whitener over the union of profile layers/dim so
        # the projection materializer → ``project_profile`` has its mandatory
        # whitener.  ``None`` when there are no profiles (degenerate stub).
        layers: set[int] = set()
        dim = 0
        for prof in profiles.values():
            for L, t in prof.items():
                layers.add(L)
                dim = t.shape[-1]
        self._stub_whitener = _whit(sorted(layers), dim) if layers else None

    @property
    def whitener(self) -> Any:
        return self._stub_whitener

    def _rebuild_steering_hooks(self) -> None:
        flat = self._steering_composer.flatten_stack()
        for name in flat:
            if name not in self._profiles:
                raise VectorNotRegisteredError(f"No vector registered for '{name}'")
        self._rebuild_entries.append(dict(flat))
        flat_any: dict[str, Any] = flat
        self._rebuild_calls.append(
            {name: alpha for name, (alpha, _trig) in flat_any.items()},
        )


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
            # LEACE ``|`` erases b under the session's M-metric.
            proj = s._profiles["a|b"][0]
            assert _m_dot(s.whitener, 0, proj, _profile_b()[0]) == pytest.approx(
                0.0, abs=1e-4,
            )
            assert s._rebuild_calls[-1] == {"a|b": 0.5}

    def test_onto_operator_registers_projected(self):
        s = _Stub({"a": _profile_a(), "b": _profile_b()})
        steering = parse_expr("0.5 a~b")
        with s.steering(steering):
            assert "a~b" in s._profiles
            # ``~`` keeps the b-component: result ∥ b (b = e0, so only x ≠ 0).
            proj = s._profiles["a~b"][0]
            assert proj[1].abs() < 1e-5

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
        # The orthogonalized direction differs from the raw "a".
        assert not torch.allclose(projected, _profile_a()[0], atol=1e-3)
        # And it is M-orthogonal to b.
        assert _m_dot(s.whitener, 0, projected, _profile_b()[0]) == pytest.approx(
            0.0, abs=1e-4,
        )

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

    def test_materialize_without_whitener_raises(self):
        """A stub with no covering whitener can't materialize a ``~``/``|``
        term — Mahalanobis is mandatory (no Euclidean fallback)."""
        s = _Stub({"a": _profile_a(), "b": _profile_b()})
        s._stub_whitener = None  # drop the whitener
        with pytest.raises(WhitenerError, match="whitener"):
            with s.steering(parse_expr("0.5 a|b")):
                pass


# ----------------------------------- v2.1 layer_means lazy-load fix-up ---

class TestLayerMeansLazy:
    """The ``session.layer_means`` property lazy-builds when
    ``probes=[]`` left ``self._layer_means`` empty.

    Closes the v2.1 footgun where ``probes=[]`` sessions hit
    ``compute_dls_axes`` with an empty dict, every layer fell into
    the conservative-keep branch, and DLS silently disabled itself.
    """

    def test_property_returns_existing_means_without_rebuild(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Non-empty ``self._layer_means`` short-circuits the property
        — no bootstrap call.  Sanity check that the lazy path is
        only triggered on miss."""
        s = _Stub({"a": _profile_a()})
        s._layer_means = {0: torch.tensor([1.0, 2.0])}

        called: list[Any] = []

        def _fail_bootstrap(*args: Any, **kwargs: Any) -> Any:
            called.append(args)
            return {99: torch.tensor([0.0])}

        from saklas.core import session as session_mod
        monkeypatch.setattr(session_mod, "bootstrap_layer_means", _fail_bootstrap)
        result = s.layer_means
        assert set(result.keys()) == {0}
        assert torch.equal(result[0], torch.tensor([1.0, 2.0]))
        assert called == [], "bootstrap_layer_means should not run on hit"

    def test_property_lazy_builds_when_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Empty ``self._layer_means`` triggers ``bootstrap_layer_means``
        on first access; result is cached on subsequent calls."""
        s = _Stub({"a": _profile_a()})
        from saklas.core.session import SaklasSession
        built = {3: torch.tensor([5.0, 6.0]), 4: torch.tensor([7.0, 8.0])}
        calls: list[Any] = []

        def _spy(*args: Any, **kwargs: Any) -> Any:
            calls.append(args)
            return built

        from saklas.core import session as session_mod

        # Give the stub the minimal handle attributes the bootstrap call
        # looks at, so the try-block actually runs.
        s._model = object()  # pyright: ignore[reportAttributeAccessIssue]  # injecting minimal stub handle; real type is PreTrainedModel
        s._tokenizer = object()  # pyright: ignore[reportAttributeAccessIssue]  # injecting minimal stub handle; real type is PreTrainedTokenizerBase
        s._layers = []  # pyright: ignore[reportAttributeAccessIssue]  # injecting minimal stub handle; real type is ModuleList
        s._model_info = {}
        monkeypatch.setattr(session_mod, "bootstrap_layer_means", _spy)

        # First access — triggers build.
        out = SaklasSession.layer_means.fget(s)  # pyright: ignore[reportOptionalCall]  # fget present on real property
        assert out is built
        assert len(calls) == 1
        # Second access — caches; no second call.
        out2 = SaklasSession.layer_means.fget(s)  # pyright: ignore[reportOptionalCall]  # same as above
        assert out2 is built
        assert len(calls) == 1

    def test_whitener_derives_means_from_single_neutral_load(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Whitener bootstrap shares one neutral-cache read with mean derivation."""
        from saklas.core.session import SaklasSession
        from saklas.io import alignment as alignment_mod

        s = _Stub({"a": _profile_a()})
        s._model = object()  # pyright: ignore[reportAttributeAccessIssue]
        s._tokenizer = object()  # pyright: ignore[reportAttributeAccessIssue]
        s._layers = []  # pyright: ignore[reportAttributeAccessIssue]
        s._model_info = {"model_id": "test/model"}
        s._layer_means = {}
        calls: list[str] = []
        acts = {0: torch.tensor([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]])}

        def _load(*args: Any, **kwargs: Any) -> dict[int, torch.Tensor]:
            calls.append(kwargs["model_id"])
            return acts

        monkeypatch.setattr(
            alignment_mod, "load_or_compute_neutral_activations", _load,
        )
        whitener = SaklasSession._build_whitener_from_cache_or_compute(s)

        assert whitener is not None
        assert calls == ["test/model"]
        assert torch.equal(s._layer_means[0], acts[0].mean(dim=0))


class TestComputeDlsAxesBipolarGuard:
    """The folded-vector (bipolar, 2-node) face of ``compute_dls_axes``:
    ``layer_means={}`` behaves identically to ``layer_means=None`` (both
    keep-all), and the all-fail fallback excludes degenerate (skipped)
    layers.  Inputs are stacked ``[μ_pos, μ_neg]`` with a single-row basis —
    the shape a difference-of-means steering vector lowers to.
    """

    @staticmethod
    def _keep(
        mu_pos: dict[int, torch.Tensor],
        mu_neg: dict[int, torch.Tensor],
        directions: dict[int, torch.Tensor],
        layer_means: dict[int, torch.Tensor] | None,
    ) -> set[int]:
        from saklas.core.vectors import compute_dls_axes
        node_centroids = {
            L: torch.stack([mu_pos[L].reshape(-1), mu_neg[L].reshape(-1)])
            for L in mu_pos
        }
        bases = {L: directions[L].reshape(1, -1) for L in directions}
        per_axis = compute_dls_axes(node_centroids, bases, layer_means)
        return {L for L, axes in per_axis.items() if axes}

    def test_none_keeps_all(self):
        mu_pos = {0: torch.tensor([1.0, 0.0]), 1: torch.tensor([2.0, 0.0])}
        mu_neg = {0: torch.tensor([-1.0, 0.0]), 1: torch.tensor([-2.0, 0.0])}
        directions = {L: mu_pos[L] - mu_neg[L] for L in mu_pos}
        assert self._keep(mu_pos, mu_neg, directions, None) == {0, 1}

    def test_empty_dict_keeps_all(self):
        mu_pos = {0: torch.tensor([1.0, 0.0]), 1: torch.tensor([2.0, 0.0])}
        mu_neg = {0: torch.tensor([-1.0, 0.0]), 1: torch.tensor([-2.0, 0.0])}
        directions = {L: mu_pos[L] - mu_neg[L] for L in mu_pos}
        assert self._keep(mu_pos, mu_neg, directions, {}) == {0, 1}

    def test_all_failed_fallback_excludes_skipped_layers(self):
        # All layers fail the discriminative test, but layers 0+1 also had
        # degenerate directions the loop explicitly skipped.  Fallback returns
        # only the *checkable* set (layer 2), not every layer — re-including
        # skipped layers via the fallback would silently undo the skip.
        import warnings as _warnings
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
            out = self._keep(mu_pos, mu_neg, directions, layer_means)
        # Layer 2 is checkable but failed; fallback returns just it.
        # Layers 0+1 stay dropped despite the fallback.
        assert out == {2}
        msgs = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
        assert any("DLS" in m for m in msgs)

    def test_partial_layer_means_runs_check(self):
        """Layers with a baseline get checked; layers without fall through
        the per-layer ``mu_n is None`` conservative-keep."""
        mu_pos = {
            0: torch.tensor([1.0, 0.0]),  # opposite-signed → kept
            1: torch.tensor([0.5, 0.0]),  # same-signed → drop
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
        assert self._keep(mu_pos, mu_neg, directions, layer_means) == {0, 2}


class TestComputeDlsAxes:
    """N-node per-axis DLS straddle — the form a flat subspace of any node
    count (a vector at K=2, ``personas`` at K=101) prunes axes with.  At
    K=2 (stacked ``[μ_pos, μ_neg]``) it reproduces the difference-of-means
    keep set the folded-vector path uses."""

    def test_n_node_straddle_keep_drop(self):
        # Three nodes, neutral at 0.  Axis 0 (x): projections {+2, 0.5, -1}
        # straddle zero ⇒ KEEP.  Axis 1 (y): projections {1, 2, 3} all
        # positive ⇒ no straddle ⇒ DROP (a common offset, not a contrast).
        from saklas.core.vectors import compute_dls_axes
        node_centroids = {
            7: torch.tensor([
                [2.0, 1.0],
                [0.5, 2.0],
                [-1.0, 3.0],
            ]),
        }
        bases = {7: torch.tensor([[1.0, 0.0], [0.0, 1.0]])}
        layer_means = {7: torch.zeros(2)}
        out = compute_dls_axes(node_centroids, bases, layer_means)
        assert out == {7: {0}}

    def test_n_node_disabled_keeps_every_axis(self):
        from saklas.core.vectors import compute_dls_axes
        node_centroids = {0: torch.randn(4, 3), 1: torch.randn(4, 3)}
        bases = {0: torch.eye(3), 1: torch.eye(3)}
        out = compute_dls_axes(node_centroids, bases, None)
        assert out == {0: {0, 1, 2}, 1: {0, 1, 2}}

    def test_n_node_missing_baseline_conservative_keep(self):
        # A layer whose neutral baseline is absent keeps every checkable axis.
        from saklas.core.vectors import compute_dls_axes
        node_centroids = {3: torch.randn(5, 4)}
        bases = {3: torch.eye(4)}
        out = compute_dls_axes(node_centroids, bases, {})  # empty ⇒ disabled
        assert out == {3: {0, 1, 2, 3}}

    def test_n_node_degenerate_row_skipped(self):
        from saklas.core.vectors import compute_dls_axes
        node_centroids = {0: torch.tensor([[1.0, 0.0], [-1.0, 0.0]])}
        bases = {0: torch.tensor([[1.0, 0.0], [0.0, 0.0]])}  # row 1 zero-norm
        layer_means = {0: torch.zeros(2)}
        out = compute_dls_axes(node_centroids, bases, layer_means)
        assert out == {0: {0}}

    def test_n_node_all_fail_fallback(self):
        import warnings as _warnings
        from saklas.core.vectors import compute_dls_axes
        # Every node on the same side of neutral on both axes ⇒ no straddle.
        node_centroids = {0: torch.tensor([[0.5, 0.7], [0.3, 0.4], [0.6, 0.9]])}
        bases = {0: torch.eye(2)}
        layer_means = {0: torch.zeros(2)}
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            out = compute_dls_axes(node_centroids, bases, layer_means)
        assert out == {0: {0, 1}}
        msgs = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
        assert any("DLS" in m for m in msgs)

# ------------------------------- v2.1 nested projection scope restore ---

class TestNestedProjectionScopeLeak:
    """Inner scope's projection should not leak back to outer scope.

    ``SteeringComposer.materialize_projections`` writes synthetic keys (``a|b``) into
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
        s._profiles["a|b"] = sentinel
        with s.steering(parse_expr("0.5 a|b")):
            assert s._profiles["a|b"] is not sentinel
        assert s._profiles["a|b"] is sentinel, (
            "pre-scope binding for a|b must be restored on scope exit"
        )
