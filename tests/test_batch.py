"""Batched generation: ``session.generate_batch`` / ``generate_sweep``.

Tests cover the serial wrapper contract plus compatible stateless fast paths:
ordering, sweep grid shape, ``applied_steering`` round-trip, and the experiment
fan endpoint.

CPU-only.  Mock ``_generate_core`` so we exercise the wrapper logic
without spinning up a real model.
"""
from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from saklas.core.results import GenerationResult, ProbeReading, RunSet
from saklas.core.steering_composer import SteeringComposer

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
        steering_alphas={},
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


class _NoopSteeringContext:
    def __enter__(self) -> "_NoopSteeringContext":
        return self

    def __exit__(self, *args: Any) -> None:
        return None


def _install_noop_steering(session: SaklasSession) -> None:
    session_any = cast(Any, session)
    session_any._profiles = {"a": {}}
    session_any.steering = lambda value: _NoopSteeringContext()
    session_any._snapshot_steering_alphas = lambda: {"a": 0.0}
    session_any._exit_internal_steering = (
        lambda steering_cm, *, swallow: steering_cm.__exit__(None, None, None)
    )


class _BatchTokenizer:
    pad_token_id = 0
    eos_token_id = 99
    name_or_path = "batch-test-tokenizer"
    vocab_size = 200
    chat_template = ""
    added_tokens_encoder: dict[str, int] = {}

    def decode(
        self,
        ids: list[int],
        skip_special_tokens: bool = True,
    ) -> str:
        del skip_special_tokens
        return " ".join(str(i) for i in ids)


class _BatchModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.config = SimpleNamespace(vocab_size=200)
        self.generation_config = SimpleNamespace(eos_token_id=99)

    def generate(self, **kwargs: Any):
        import torch

        self.calls.append(kwargs)
        input_ids = kwargs["input_ids"]
        tail = torch.tensor(
            [
                [10, 11, 99],
                [12, 13, 14],
                [15, 0, 0],
            ],
            dtype=torch.long,
            device=input_ids.device,
        )
        return torch.cat([input_ids, tail[: input_ids.shape[0]]], dim=1)


class _BatchProbeMonitor:
    def __init__(self) -> None:
        self.probe_names = ["mood"]
        self.scored: list[float] = []

    def probe_layers(self) -> set[int]:
        return {0}

    def enable_curved_warm(self, flag: bool) -> None:
        del flag

    def score_aggregate(
        self,
        hidden_per_layer: dict[int, Any],
    ) -> dict[str, ProbeReading]:
        value = float(hidden_per_layer[0].reshape(-1)[0])
        self.scored.append(value)
        return {"mood": ProbeReading(fraction=value, nearest=[], coords=(value,))}


class _ProbeBatchModel(_BatchModel):
    def __init__(self, layers: Any) -> None:
        super().__init__()
        self.layers = layers

    def generate(self, **kwargs: Any):
        import torch

        input_ids = kwargs["input_ids"]
        for values in ([1.0, 10.0, 100.0], [2.0, 20.0, 200.0], [3.0, 30.0, 300.0]):
            hidden = torch.tensor(
                values[: input_ids.shape[0]],
                dtype=torch.float32,
                device=input_ids.device,
            ).reshape(input_ids.shape[0], 1, 1)
            for layer in self.layers:
                layer(hidden)
        return super().generate(**kwargs)


def _fast_batch_session():
    import torch
    from saklas.core.generation import GenerationConfig, GenerationState
    from saklas.core.session import CaptureState, GenState, SaklasSession
    from saklas.core.triggers import TriggerContext

    s = SaklasSession.__new__(SaklasSession)
    model = _BatchModel()
    s_any = cast(Any, s)
    s_any._model = model
    s_any._tokenizer = _BatchTokenizer()
    s._device = torch.device("cpu")
    s_any._gen_lock = threading.Lock()
    s._gen_phase = GenState.IDLE
    s._gen_state = GenerationState()
    s_any._monitor = SimpleNamespace(probe_names=[])
    s._profiles = {}
    s._manifolds = {}
    s._default_return_top_k = 0
    s.config = GenerationConfig(
        max_new_tokens=3,
        temperature=0.0,
        top_p=1.0,
        top_k=None,
    )
    events = SimpleNamespace(emitted=[])
    events.emit = lambda event: events.emitted.append(event)
    s_any.events = events
    s_any._steering = SimpleNamespace(
        ctx=TriggerContext(),
        reset_manifold_feet=lambda: None,
        has_compiled_offsets=lambda: False,
        zero_compiled_offsets=lambda: None,
    )
    s._live_lens = None
    s._lens_probes = {}
    s._sae_probes = {}
    s._sae_layer = None
    s._trait_queues = []
    s._active_gen_reservation = None
    s._last_token_probe_payload = None
    s._capture_state = CaptureState()
    s._compiled_clean_eligible = False
    s._incremental_readings = []
    s._incremental_gate_scores = []
    s._steering_uses_compiled_offsets = False
    s._last_per_token_scores = None
    s._last_result = None
    s._internal_steering_pop = False
    s._active_role = None
    s._steering_composer = SteeringComposer(s)

    def _prepare_input(
        input: Any,
        raw: bool = False,
        thinking: bool = False,
        stateless: bool = False,
        parent_node_id: str | None = None,
        user_role: str | None = None,
        assistant_role: str | None = None,
        to_device: bool = True,
        gen_seat: str = "assistant",
    ):
        del raw, thinking, stateless, parent_node_id
        del user_role, assistant_role, to_device, gen_seat
        mapping = {
            "alpha": [1, 2],
            "beta": [3, 4, 5],
            "gamma": [6],
        }
        return torch.tensor([mapping[input]], dtype=torch.long)

    s._prepare_input = _prepare_input
    return s, model


def _probe_fast_batch_session():
    import torch
    from saklas.core.hooks import HiddenCapture

    s, _model = _fast_batch_session()
    s._layers = torch.nn.ModuleList([torch.nn.Identity()])
    s._capture = HiddenCapture()
    s_any = cast(Any, s)
    s_any._monitor = _BatchProbeMonitor()
    model = _ProbeBatchModel(s._layers)
    s_any._model = model
    return s, model


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
        s._steering_composer = SteeringComposer(s)
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

    def test_compatible_batch_uses_one_model_generate(self) -> None:
        s, model = _fast_batch_session()

        def _fail_generate_core(*args: Any, **kwargs: Any) -> GenerationResult:
            raise AssertionError("serial generation path should not run")

        s._generate_core = _fail_generate_core
        seen: list[tuple[int, str]] = []

        runset = s.generate_batch(
            ["alpha", "beta", "gamma"],
            thinking=False,
            on_result=lambda idx, result: seen.append((idx, result.text)),
        )

        assert len(model.calls) == 1
        call = model.calls[0]
        assert call["input_ids"].shape == (3, 3)
        assert call["attention_mask"].tolist() == [
            [0, 1, 1],
            [1, 1, 1],
            [0, 0, 1],
        ]
        assert [r.tokens for r in runset] == [[10, 11], [12, 13, 14], [15]]
        assert [r.finish_reason for r in runset] == ["stop", "length", "stop"]
        assert [r.text for r in runset] == ["10 11", "12 13 14", "15"]
        assert runset.grid == [
            {"prompt_index": 0},
            {"prompt_index": 1},
            {"prompt_index": 2},
        ]
        assert runset.node_ids == [None, None, None]
        assert seen == [(0, "10 11"), (1, "12 13 14"), (2, "15")]
        assert runset.metrics["batch_token_count"] == 6
        assert runset.metrics["batch_elapsed"] > 0.0
        assert "batch_tok_per_sec" in runset.metrics
        assert s.last_result is runset[-1]

    def test_greedy_batch_allows_seeded_sampling_config(self) -> None:
        from saklas.core.sampling import SamplingConfig

        s, model = _fast_batch_session()

        def _fail_generate_core(*args: Any, **kwargs: Any) -> GenerationResult:
            raise AssertionError("serial generation path should not run")

        s._generate_core = _fail_generate_core

        runset = s.generate_batch(
            ["alpha", "beta"],
            sampling=SamplingConfig(seed=123),
            thinking=False,
        )

        assert len(model.calls) == 1
        assert [r.tokens for r in runset] == [[10, 11], [12, 13, 14]]

    def test_stochastic_seeded_batch_stays_serial(self) -> None:
        from saklas.core.sampling import SamplingConfig

        s, model = _fast_batch_session()
        capture: list[Any] = []
        _stub_generate_core(s, capture=capture)

        runset = s.generate_batch(
            ["alpha", "beta"],
            sampling=SamplingConfig(seed=123, temperature=0.7),
            thinking=False,
        )

        assert len(model.calls) == 0
        assert [r.text for r in runset] == ["out_0", "out_1"]
        assert [c["input"] for c in capture] == ["alpha", "beta"]

    def test_probes_fall_back_to_serial_generation(self) -> None:
        s, model = _fast_batch_session()
        cast(Any, s._monitor).probe_names = ["mood"]
        capture: list[Any] = []
        _stub_generate_core(s, capture=capture)

        runset = s.generate_batch(["alpha", "beta"], thinking=False)

        assert len(model.calls) == 0
        assert [r.text for r in runset] == ["out_0", "out_1"]
        assert [c["input"] for c in capture] == ["alpha", "beta"]

    def test_probe_batch_fast_path_scores_per_row_aggregate(self) -> None:
        s, model = _probe_fast_batch_session()

        def _fail_generate_core(*args: Any, **kwargs: Any) -> GenerationResult:
            raise AssertionError("serial generation path should not run")

        s._generate_core = _fail_generate_core

        runset = s.generate_batch(["alpha", "beta", "gamma"], thinking=False)

        assert len(model.calls) == 1
        assert [r.tokens for r in runset] == [[10, 11], [12, 13, 14], [15]]
        assert [r.probe_readings["mood"].coords for r in runset] == [
            (2.0,),
            (30.0,),
            (100.0,),
        ]
        assert cast(Any, s._monitor).scored == [2.0, 30.0, 100.0]

    def test_probe_batch_fast_path_honors_return_probe_readings_false(self) -> None:
        from saklas.core.sampling import SamplingConfig

        s, model = _probe_fast_batch_session()

        def _fail_generate_core(*args: Any, **kwargs: Any) -> GenerationResult:
            raise AssertionError("serial generation path should not run")

        s._generate_core = _fail_generate_core

        runset = s.generate_batch(
            ["alpha", "beta", "gamma"],
            sampling=SamplingConfig(return_probe_readings=False),
            thinking=False,
        )

        assert len(model.calls) == 1
        assert [r.tokens for r in runset] == [[10, 11], [12, 13, 14], [15]]
        assert [r.probe_readings for r in runset] == [{}, {}, {}]
        assert cast(Any, s._monitor).scored == []

    def test_sae_readout_probe_batch_fast_path_scores_per_row_aggregate(self) -> None:
        from saklas.core.sae import MockSaeBackend

        s, model = _probe_fast_batch_session()
        cast(Any, s)._monitor = SimpleNamespace(probe_names=[])
        s._sae_backend = MockSaeBackend(layers=frozenset({0}), d_model=1, d_feature=1)
        s._sae_layer = 0
        s._sae_width = 1
        s._sae_feature_meta = {}
        s._sae_probes = {
            "sae/0": {
                "feature_id": 0,
                "layer": 0,
                "label": None,
                "max_act": None,
            },
        }

        def _fail_generate_core(*args: Any, **kwargs: Any) -> GenerationResult:
            raise AssertionError("serial generation path should not run")

        s._generate_core = _fail_generate_core

        runset = s.generate_batch(["alpha", "beta", "gamma"], thinking=False)

        assert len(model.calls) == 1
        assert [r.probe_readings["sae/0"].coords for r in runset] == [
            (2.0,),
            (30.0,),
            (100.0,),
        ]

    def test_sae_readout_probe_batch_fast_path_honors_return_probe_readings_false(self) -> None:
        from saklas.core.sampling import SamplingConfig

        s, model = _probe_fast_batch_session()
        cast(Any, s)._monitor = SimpleNamespace(probe_names=[])
        s._sae_layer = 0
        s._sae_probes = {
            "sae/0": {
                "feature_id": 0,
                "layer": 0,
                "label": None,
                "max_act": None,
            },
        }

        def _fail_generate_core(*args: Any, **kwargs: Any) -> GenerationResult:
            raise AssertionError("serial generation path should not run")

        s._generate_core = _fail_generate_core

        runset = s.generate_batch(
            ["alpha", "beta", "gamma"],
            sampling=SamplingConfig(return_probe_readings=False),
            thinking=False,
        )

        assert len(model.calls) == 1
        assert [r.probe_readings for r in runset] == [{}, {}, {}]

    def test_lens_readout_probe_batch_fast_path_scores_per_row_aggregate(self) -> None:
        s, model = _probe_fast_batch_session()
        s_any = cast(Any, s)
        s_any._monitor = SimpleNamespace(probe_names=[])
        s_any._lens_probes = {"jlens/g": {"token_id": 1, "layers": [0]}}
        s_any._lens_probe_layers = lambda: {0}

        def _score_lens_probes(
            hidden: dict[int, Any],
            **kwargs: Any,
        ) -> dict[str, ProbeReading]:
            del kwargs
            value = float(hidden[0].reshape(-1)[0])
            return {"jlens/g": ProbeReading(0.0, [], coords=(value,))}

        s_any._score_lens_probes = _score_lens_probes

        def _fail_generate_core(*args: Any, **kwargs: Any) -> GenerationResult:
            raise AssertionError("serial generation path should not run")

        s._generate_core = _fail_generate_core

        runset = s.generate_batch(["alpha", "beta", "gamma"], thinking=False)

        assert len(model.calls) == 1
        assert [r.probe_readings["jlens/g"].coords for r in runset] == [
            (2.0,),
            (30.0,),
            (100.0,),
        ]

    def test_lens_readout_probe_batch_fast_path_honors_return_probe_readings_false(self) -> None:
        from saklas.core.sampling import SamplingConfig

        s, model = _probe_fast_batch_session()
        s_any = cast(Any, s)
        s_any._monitor = SimpleNamespace(probe_names=[])
        s_any._lens_probes = {"jlens/g": {"token_id": 1, "layers": [0]}}
        s_any._lens_probe_layers = lambda: {0}

        def _score_lens_probes(*_args: Any, **_kwargs: Any) -> dict[str, ProbeReading]:
            raise AssertionError("return_probe_readings=False should skip lens probes")

        s_any._score_lens_probes = _score_lens_probes

        def _fail_generate_core(*args: Any, **kwargs: Any) -> GenerationResult:
            raise AssertionError("serial generation path should not run")

        s._generate_core = _fail_generate_core

        runset = s.generate_batch(
            ["alpha", "beta", "gamma"],
            sampling=SamplingConfig(return_probe_readings=False),
            thinking=False,
        )

        assert len(model.calls) == 1
        assert [r.probe_readings for r in runset] == [{}, {}, {}]

    def test_deterministic_fan_uses_batched_generation(self) -> None:
        s, model = _fast_batch_session()

        def _fail_generate_core(*args: Any, **kwargs: Any) -> GenerationResult:
            raise AssertionError("serial generation path should not run")

        s._generate_core = _fail_generate_core

        runset = s.generate("alpha", n=3, stateless=True, thinking=False)

        assert len(model.calls) == 1
        assert model.calls[0]["input_ids"].shape == (3, 2)
        assert runset.kind == "fan"
        assert runset.grid == [{}, {}, {}]
        assert runset.node_ids == [None, None, None]
        assert [r.tokens for r in runset] == [[10, 11], [12, 13, 14], [15]]

    def test_deterministic_fan_honors_return_probe_readings_false(self) -> None:
        from saklas.core.sampling import SamplingConfig

        s, model = _probe_fast_batch_session()

        def _fail_generate_core(*args: Any, **kwargs: Any) -> GenerationResult:
            raise AssertionError("serial generation path should not run")

        s._generate_core = _fail_generate_core

        runset = s.generate(
            "alpha",
            n=3,
            stateless=True,
            sampling=SamplingConfig(return_probe_readings=False),
            thinking=False,
        )

        assert len(model.calls) == 1
        assert runset.kind == "fan"
        assert [r.probe_readings for r in runset] == [{}, {}, {}]
        assert cast(Any, s._monitor).scored == []

    def test_deterministic_fan_with_seed_uses_batched_generation(self) -> None:
        from saklas.core.sampling import SamplingConfig

        s, model = _fast_batch_session()

        def _fail_generate_core(*args: Any, **kwargs: Any) -> GenerationResult:
            raise AssertionError("serial generation path should not run")

        s._generate_core = _fail_generate_core

        runset = s.generate(
            "alpha",
            n=3,
            sampling=SamplingConfig(seed=123),
            stateless=True,
            thinking=False,
        )

        assert len(model.calls) == 1
        assert runset.kind == "fan"
        assert [r.tokens for r in runset] == [[10, 11], [12, 13, 14], [15]]


# ---------------------------------------------------------------------------
# Prefix-cache eligibility — the predicates that decide whether a batch can
# share one prefill.  CPU-only: they only read steering triggers, no model.
# ---------------------------------------------------------------------------


class TestPrefixCacheEligibility:
    def _session(self):
        from saklas.core.session import SaklasSession
        session = SaklasSession.__new__(SaklasSession)
        session._steering_composer = SteeringComposer(session)
        return session

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
        s._steering_composer._stack = []
        assert s._steering_active_in_prefill() is False
        # Response-phase entry (prompt=False) → prefill untouched.
        s._steering_composer._stack = [{"honest": (0.5, Trigger.GENERATED_ONLY)}]
        assert s._steering_active_in_prefill() is False
        # Default BOTH entry → prefill steered.
        s._steering_composer._stack = [{"honest": (0.5, Trigger.BOTH)}]
        assert s._steering_active_in_prefill() is True
        # A probe-gated trigger reports inactive during prefill.
        gated = Trigger.when("honest.deceptive", ">", 0.4)
        s._steering_composer._stack = [{"honest": (0.5, gated)}]
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

    def test_static_prefix_hit_requires_static_eligibility_and_headroom(self) -> None:
        import torch
        from saklas.core.session import _PrefixCacheEntry

        s = self._session()
        cache = object()
        s._prefix_cache = _PrefixCacheEntry(
            prefix_ids_cpu=torch.tensor([1, 2]),
            past_key_values=cache,
            prefix_len=2,
            static=True,
            max_cache_len=5,
        )
        ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

        assert s._try_prefix_cache_hit(
            ids, static_eligible=False, required_max_new_tokens=1,
        ) is None
        assert s._try_prefix_cache_hit(
            ids, static_eligible=True, required_max_new_tokens=3,
        ) is None

        hit = s._try_prefix_cache_hit(
            ids, static_eligible=True, required_max_new_tokens=2,
        )
        assert hit is not None
        suffix, hit_cache, prefix_len, is_static = hit
        assert suffix.tolist() == [[3]]
        assert hit_cache is cache
        assert prefix_len == 2
        assert is_static is True

    def test_generation_loop_keeps_static_cache_on_static_prefix_hit(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import torch
        from saklas.core.generation import GenerationConfig
        from saklas.core.session import CaptureState, _PrefixCacheEntry

        class _Cache:
            def __init__(self) -> None:
                self.crops: list[int] = []

            def crop(self, length: int) -> None:
                self.crops.append(length)

        s = self._session()
        cache = _Cache()
        s._prefix_cache = _PrefixCacheEntry(
            prefix_ids_cpu=torch.tensor([1, 2]),
            past_key_values=cache,
            prefix_len=2,
            static=True,
            max_cache_len=8,
        )
        s._static_cache_active = True
        s_any = cast(Any, s)
        s_any._steering = SimpleNamespace(
            all_fast_path=lambda: True,
            static_steerable=lambda: False,
            ctx=None,
            hooks={},
        )
        s._steering_active_in_prefill = lambda: False
        s._steering_needs_probe_gating = lambda: False
        s_any._build_gating_score_callback = lambda: None
        s._compiled = False
        s._device = torch.device("cpu")
        s_any._model = object()
        s_any._tokenizer = object()
        s_any._gen_state = object()
        s._capture_state = CaptureState(persistent=False)
        s_any._capture = SimpleNamespace(
            ingest_persistent=lambda: None,
            fire_step_sink=lambda: None,
        )

        seen: dict[str, Any] = {}

        def _fake_generate(
            model: Any,
            tokenizer: Any,
            input_ids: torch.Tensor,
            config: GenerationConfig,
            state: Any,
            **kwargs: Any,
        ) -> list[int]:
            del model, tokenizer, config, state
            seen["input_ids"] = input_ids.clone()
            seen.update(kwargs)
            return [42]

        monkeypatch.setattr("saklas.core.session.generate_steered", _fake_generate)

        out, _elapsed = s._run_generation_loop(
            torch.tensor([[1, 2, 3]], dtype=torch.long),
            GenerationConfig(max_new_tokens=2),
            use_thinking=False,
            want_hidden=False,
            effective_tap=None,
            seed=None,
            stop_list=None,
            logit_bias=None,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            lp_count=None,
        )

        assert out == [42]
        assert seen["input_ids"].tolist() == [[3]]
        assert seen["past_key_values"] is cache
        assert seen["cache_position_offset"] == 2
        assert seen["use_static_cache"] is True
        assert cache.crops == [2]

    def test_cache_prefix_can_build_static_cache_entry(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import torch
        from saklas.core.generation import GenerationConfig

        s = self._session()
        s._static_cache_active = True
        s._device = torch.device("cpu")
        s.config = GenerationConfig(max_new_tokens=4)
        s._steering_composer._stack = []
        s._prefix_cache = None
        from saklas.core.session import GenState
        s._gen_phase = GenState.IDLE
        s._end_capture = lambda: None

        static_cache = object()
        made: dict[str, Any] = {}

        def _make_static_cache(model: Any, max_cache_len: int, device: Any, dtype: Any) -> object:
            made.update({
                "model": model,
                "max_cache_len": max_cache_len,
                "device": device,
                "dtype": dtype,
            })
            return static_cache

        monkeypatch.setattr(
            "saklas.core.cuda_graphs.make_static_cache", _make_static_cache,
        )

        class _Model:
            config = object()

            def __init__(self) -> None:
                self.calls: list[dict[str, Any]] = []
                self.param = torch.zeros(1, dtype=torch.float16)

            def parameters(self):
                return iter([self.param])

            def __call__(self, **kwargs: Any) -> Any:
                self.calls.append(kwargs)
                return SimpleNamespace(past_key_values=kwargs.get("past_key_values"))

        model = _Model()
        cast(Any, s)._model = model

        prefix_len = s.cache_prefix(
            torch.tensor([[4, 5, 6]], dtype=torch.long),
            max_new_tokens=7,
            prefer_static=True,
        )

        assert prefix_len == 3
        assert s._prefix_cache is not None
        assert s._prefix_cache.static is True
        assert s._prefix_cache.max_cache_len == 10
        assert s._prefix_cache.past_key_values is static_cache
        assert made == {
            "model": model,
            "max_cache_len": 10,
            "device": torch.device("cpu"),
            "dtype": torch.float16,
        }
        call = model.calls[0]
        assert call["past_key_values"] is static_cache
        assert call["cache_position"].tolist() == [0, 1, 2]


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

    def test_degenerate_greedy_sweep_uses_batched_generation(self) -> None:
        s, model = _fast_batch_session()
        _install_noop_steering(s)

        def _fail_generate_core(*args: Any, **kwargs: Any) -> GenerationResult:
            raise AssertionError("serial generation path should not run")

        s._generate_core = _fail_generate_core
        seen: list[tuple[int, str, dict[str, float]]] = []

        runset = s.generate_sweep(
            "alpha",
            sweep={"a": [0.0, 0.0, 0.0]},
            thinking=False,
            on_result=lambda idx, result, alphas: seen.append(
                (idx, result.text, dict(alphas)),
            ),
        )

        assert len(model.calls) == 1
        assert model.calls[0]["input_ids"].shape == (3, 2)
        assert runset.kind == "fan"
        assert runset.grid == [{"a": 0.0}, {"a": 0.0}, {"a": 0.0}]
        assert [r.tokens for r in runset] == [[10, 11], [12, 13, 14], [15]]
        assert seen == [
            (0, "10 11", {"a": 0.0}),
            (1, "12 13 14", {"a": 0.0}),
            (2, "15", {"a": 0.0}),
        ]

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
