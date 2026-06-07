"""Batched generation: ``session.generate_batch`` / ``generate_sweep``.

Wrapper-loop approach — each prompt acquires the gen-lock, runs through
``_generate_core``, releases.  Tests cover ordering, sweep grid shape,
``applied_steering`` round-trip, and the experiment fan endpoint.

CPU-only.  Mock ``_generate_core`` so we exercise the wrapper logic
without spinning up a real model.
"""
from __future__ import annotations

import asyncio
import threading
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from saklas.core.results import GenerationResult, RunSet

if TYPE_CHECKING:
    from saklas.core.session import SaklasSession


# ---------------------------------------------------------------------------
# Helpers — mock just enough of SaklasSession for the batch path.
# ---------------------------------------------------------------------------


def _make_result(text: str, applied: str | None = None) -> GenerationResult:
    """Cheap GenerationResult stub."""
    return GenerationResult(
        text=text,
        tokens=[1, 2, 3],
        token_count=3,
        tok_per_sec=50.0,
        elapsed=0.06,
        readings={},
        vectors={},
        prompt_tokens=4,
        finish_reason="stop",
        applied_steering=applied,
    )


def _stub_generate_core(session: SaklasSession, *, capture: list[Any]) -> None:
    """Replace ``session._generate_core`` with a stub that records calls.

    Returns one ``GenerationResult`` per call carrying the call's index
    so tests can verify ordering.  Each call appends a ``(input,
    steering)`` tuple to ``capture`` for assertions on the wrapper's
    composition logic.
    """
    counter = {"n": 0}

    def _fake(input: Any, *, steering: Any = None, sampling: Any = None, stateless: bool = False, raw: bool = False, thinking: Any = None, on_token: Any = None, **kwargs: Any) -> GenerationResult:
        # ``kwargs`` swallows additions to ``_generate_core``'s signature
        # (v2.3 added ``parent_node_id`` and ``recipe_override``) so this
        # stub doesn't churn every time the core gains a new optional
        # keyword.
        idx = counter["n"]
        counter["n"] += 1
        capture.append({"input": input, "steering": steering})
        applied = steering if isinstance(steering, str) else None
        return _make_result(f"out_{idx}", applied=applied)

    session._generate_core = _fake


# ---------------------------------------------------------------------------
# session.generate_batch — pure unit tests against a real SaklasSession
# subclass that overrides _generate_core.
# ---------------------------------------------------------------------------


class TestGenerateBatch:
    def _session(self):
        # Construct a minimal SaklasSession by bypassing __init__; the
        # batch methods only need ``_generate_core`` to exist.
        from saklas.core.session import SaklasSession

        s = SaklasSession.__new__(SaklasSession)
        return s

    def test_returns_results_in_prompt_order(self) -> None:
        s = self._session()
        capture: list[Any] = []
        _stub_generate_core(s, capture=capture)

        prompts = ["alpha", "beta", "gamma"]
        results = s.generate_batch(prompts)

        assert len(results) == 3
        assert [r.text for r in results] == ["out_0", "out_1", "out_2"]
        assert [c["input"] for c in capture] == prompts

    def test_steering_passes_through_unchanged(self) -> None:
        s = self._session()
        capture: list[Any] = []
        _stub_generate_core(s, capture=capture)

        s.generate_batch(["p1", "p2"], steering="0.3 honest")

        assert all(c["steering"] == "0.3 honest" for c in capture)

    def test_on_result_callback_fires_per_completion(self) -> None:
        s = self._session()
        capture: list[Any] = []
        _stub_generate_core(s, capture=capture)

        seen: list[tuple[int, str]] = []
        s.generate_batch(
            ["p1", "p2", "p3"],
            on_result=lambda idx, result: seen.append((idx, result.text)),
        )

        assert seen == [(0, "out_0"), (1, "out_1"), (2, "out_2")]

    def test_empty_prompt_list_raises(self) -> None:
        s = self._session()
        with pytest.raises(ValueError, match="non-empty list"):
            s.generate_batch([])


# ---------------------------------------------------------------------------
# Prefix-cache eligibility — the predicates that decide whether a batch can
# share one prefill.  CPU-only: they only read steering triggers, no model.
# ---------------------------------------------------------------------------


class TestPrefixCacheEligibility:
    def _session(self):
        from saklas.core.session import SaklasSession

        return SaklasSession.__new__(SaklasSession)

    @pytest.mark.parametrize(
        "expr, expected_inactive",
        [
            (None, True),                          # no steering → reusable
            ("0.5 honest", False),                 # default BOTH steers prefill
            ("0.5 honest@response", True),         # response-phase: prompt untouched
            ("0.5 honest@generated", True),        # decode-only
            ("0.5 honest@after", True),            # after-N: prompt=False
            ("0.5 honest@prompt", False),          # prompt-phase steers prefill
            ("0.5 honest@both", False),            # explicit both
            ("0.5 honest@when:honest.deceptive>0.4", True),  # probe-gated: inactive in prefill
        ],
    )
    def test_steering_value_prefill_inactive(
        self, expr: "str | None", expected_inactive: bool,
    ) -> None:
        s = self._session()
        assert s._steering_value_prefill_inactive(expr) is expected_inactive

    def test_mixed_terms_active_if_any_steers_prefill(self) -> None:
        # One response-phase term + one default-BOTH term → prefill IS steered.
        s = self._session()
        assert s._steering_value_prefill_inactive("0.5 honest@response + 0.3 warm") is False

    def test_malformed_expression_treated_as_active(self) -> None:
        # A parse failure is conservatively prefill-active so the caller skips
        # caching and lets the normal path surface the error.
        s = self._session()
        assert s._steering_value_prefill_inactive("0.5 !nope~bad") is False

    def test_active_in_prefill_reads_the_live_stack(self) -> None:
        from saklas.core.triggers import Trigger

        s = self._session()
        s._steering_stack = []
        assert s._steering_active_in_prefill() is False
        # Response-phase entry (prompt=False) → prefill untouched.
        s._steering_stack = [{"honest": (0.5, Trigger.GENERATED_ONLY)}]
        assert s._steering_active_in_prefill() is False
        # Default BOTH entry → prefill steered.
        s._steering_stack = [{"honest": (0.5, Trigger.BOTH)}]
        assert s._steering_active_in_prefill() is True
        # A probe-gated trigger reports inactive during prefill.
        gated = Trigger.when("honest.deceptive", ">", 0.4)
        s._steering_stack = [{"honest": (0.5, gated)}]
        assert s._steering_active_in_prefill() is False

    def test_batch_common_prefix_detection_keeps_scalar_walk_on_cpu(self) -> None:
        import torch

        s = self._session()
        common = list(range(32))

        def _prepare_input(
            input: Any,
            raw: bool = False,
            thinking: bool = False,
            stateless: bool = False,
            parent_node_id: str | None = None,
            user_role: str | None = None,
            assistant_role: str | None = None,
            to_device: bool = True,
        ) -> torch.Tensor:
            suffix = 100 if input == "a" else 101
            return torch.tensor([common + [suffix]], dtype=torch.long)

        cast(Any, s)._prepare_input = _prepare_input

        prefix = s._batch_common_prefix_ids(["a", "b"], raw=False)

        assert prefix is not None
        assert prefix.device.type == "cpu"
        assert prefix.tolist() == common


# ---------------------------------------------------------------------------
# session.generate_sweep — Cartesian product, applied_steering receipts.
# ---------------------------------------------------------------------------


class TestGenerateSweep:
    def _session(self):
        from saklas.core.session import SaklasSession

        s = SaklasSession.__new__(SaklasSession)
        return s

    def test_single_concept_sweep_yields_one_per_alpha(self) -> None:
        s = self._session()
        capture: list[Any] = []
        _stub_generate_core(s, capture=capture)

        results = s.generate_sweep(
            "describe a sunset",
            sweep={"happy.sad": [-0.4, 0.0, 0.4]},
        )

        assert len(results) == 3
        # Each call's steering string carries the unique alpha + concept.
        steerings = [c["steering"] for c in capture]
        assert steerings == [
            "-0.4 happy.sad",
            "0.0 happy.sad",
            "0.4 happy.sad",
        ]

    def test_two_concept_grid_full_product(self) -> None:
        s = self._session()
        capture: list[Any] = []
        _stub_generate_core(s, capture=capture)

        s.generate_sweep(
            "test",
            sweep={"a": [0.0, 0.3], "b": [0.0, 0.5]},
        )

        # 2 * 2 = 4 results.
        assert len(capture) == 4
        # Order is itertools.product over (a, b): a varies slowest.
        steerings = [c["steering"] for c in capture]
        assert steerings == [
            "0.0 a + 0.0 b",
            "0.0 a + 0.5 b",
            "0.3 a + 0.0 b",
            "0.3 a + 0.5 b",
        ]

    def test_base_steering_composes_under_swept_terms(self) -> None:
        s = self._session()
        capture: list[Any] = []
        _stub_generate_core(s, capture=capture)

        s.generate_sweep(
            "test",
            sweep={"honest": [0.3, 0.6]},
            base_steering="0.2 warm",
        )

        steerings = [c["steering"] for c in capture]
        assert steerings == [
            "0.2 warm + 0.3 honest",
            "0.2 warm + 0.6 honest",
        ]

    def test_on_result_carries_alpha_values(self) -> None:
        s = self._session()
        capture: list[Any] = []
        _stub_generate_core(s, capture=capture)

        seen: list[tuple[int, dict[str, float]]] = []
        s.generate_sweep(
            "test",
            sweep={"a": [0.0, 0.3], "b": [0.5]},
            on_result=lambda idx, result, alphas: seen.append((idx, dict(alphas))),
        )

        assert seen == [
            (0, {"a": 0.0, "b": 0.5}),
            (1, {"a": 0.3, "b": 0.5}),
        ]

    def test_applied_steering_round_trips_canonical(self) -> None:
        s = self._session()
        capture: list[Any] = []
        _stub_generate_core(s, capture=capture)

        results = s.generate_sweep(
            "test", sweep={"honest": [0.4]},
        )

        # Stub propagates ``steering`` to ``applied_steering``; the
        # canonical receipt round-trips through generate_sweep.
        assert results[0].applied_steering == "0.4 honest"

    def test_empty_sweep_dict_raises(self) -> None:
        s = self._session()
        with pytest.raises(ValueError, match="non-empty"):
            s.generate_sweep("test", sweep={})

    def test_empty_alpha_list_raises(self) -> None:
        s = self._session()
        with pytest.raises(ValueError, match="non-empty list"):
            s.generate_sweep("test", sweep={"a": []})


# ---------------------------------------------------------------------------
# Server: POST /saklas/v1/sessions/{id}/experiments/fan.
# ---------------------------------------------------------------------------


def _mock_session_for_server():
    """Like ``test_saklas_api._mock_session`` but trimmed to what sweep needs."""
    session = MagicMock()
    session.model_id = "test/model"
    session.model_info = {"model_type": "gemma2", "num_layers": 4, "hidden_dim": 16}
    session._device = "cpu"
    session._dtype = "torch.bfloat16"
    session._created_ts = 1_700_000_000

    session.config = MagicMock()
    session.config.temperature = 1.0
    session.config.top_p = 0.9
    session.config.top_k = None
    session.config.max_new_tokens = 64
    session.config.system_prompt = None

    session.vectors = {}
    session.probes = {}
    session.history = []
    session._monitor = MagicMock()
    session._monitor.probe_names = []
    session._tokenizer = MagicMock()
    session._layers = []
    session._last_per_token_scores = None
    session.last_per_token_scores = None
    session.last_result = None
    session._gen_state = MagicMock()
    session._gen_state.finish_reason = "stop"
    session.lock = asyncio.Lock()

    session._trait_queues = []
    session._trait_lock = threading.Lock()
    session.register_trait_queue = lambda loop, q: session._trait_queues.append((loop, q))
    session.unregister_trait_queue = lambda loop, q: None

    session.events = MagicMock()
    session.events.subscribe = lambda cb: (lambda: None)
    session.events.emit = lambda event: None

    return session


@pytest.fixture
def fan_client():
    from saklas.server import create_app

    session = _mock_session_for_server()

    # Stub generate_sweep to return the standardized RunSet shape.
    def _fake_sweep(prompt: Any, sweep: Any, *, base_steering: Any = None, sampling: Any = None,
                   thinking: Any = None, stateless: bool = True, raw: bool = False, on_result: Any = None,
                   parent_node_id: Any = None, **kwargs: Any) -> RunSet:
        results: list[GenerationResult] = []
        node_ids: list[Any] = []
        grid: list[dict[str, float]] = []
        idx = 0
        # Simple linearization: walk the first concept's alphas.
        first_name, first_alphas = next(iter(sweep.items()))
        for alpha in first_alphas:
            r = _make_result(f"out_{idx}", applied=f"{alpha} {first_name}")
            results.append(r)
            node_ids.append(f"NODE_{idx}")
            grid.append({first_name: float(alpha)})
            if on_result is not None:
                on_result(idx, r, {first_name: alpha})
            idx += 1
        return RunSet(results, node_ids=node_ids, grid=grid, kind="fan")

    session.generate_sweep = _fake_sweep
    session.stop = MagicMock()

    app = create_app(session, default_steering=None)
    return session, TestClient(app)


class TestExperimentFanEndpoint:
    def test_fan_returns_rows_and_node_ids(self, fan_client: Any) -> None:
        _session, client = fan_client

        body = {
            "prompt": "describe a sunset",
            "grid": {"happy.sad": [-0.4, 0.0, 0.4]},
        }
        r = client.post(
            "/saklas/v1/sessions/default/experiments/fan",
            json=body,
        )
        assert r.status_code == 200
        payload = r.json()
        assert payload["kind"] == "fan"
        assert payload["total"] == 3
        assert payload["node_ids"] == ["NODE_0", "NODE_1", "NODE_2"]

        rows = payload["rows"]
        for i in range(3):
            row = rows[i]
            assert row["idx"] == i
            assert "happy.sad" in row["alpha_values"]
            assert row["node_id"] == f"NODE_{i}"
            assert "applied_steering" in row["result"]

    def test_fan_empty_grid_returns_400(self, fan_client: Any) -> None:
        _session, client = fan_client
        body = {"prompt": "x", "grid": {}}
        r = client.post("/saklas/v1/sessions/default/experiments/fan", json=body)
        assert r.status_code == 400

    def test_fan_empty_alpha_list_returns_400(self, fan_client: Any) -> None:
        _session, client = fan_client
        body = {"prompt": "x", "grid": {"a": []}}
        r = client.post("/saklas/v1/sessions/default/experiments/fan", json=body)
        assert r.status_code == 400

    def test_fan_unknown_session_returns_404(self, fan_client: Any) -> None:
        _session, client = fan_client
        body = {"prompt": "x", "grid": {"a": [0.0]}}
        r = client.post("/saklas/v1/sessions/missing/experiments/fan", json=body)
        assert r.status_code == 404
