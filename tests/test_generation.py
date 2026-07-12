"""CPU-only generation-loop regressions."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from saklas.core.generation import (
    GenerationConfig,
    GenerationState,
    _PenaltyState,
    generate_steered,
)
from saklas.core.results import GenerationResult, ProbeReading
from saklas.core.session import CaptureMode, CaptureState, GenState, SaklasSession
from saklas.core.steering import Steering


class _StopTokenizer:
    name_or_path = "stop-tokenizer"
    vocab_size = 4
    eos_token_id = 3
    added_tokens_encoder = {}
    all_special_ids = [3]

    _pieces = {
        0: "Hello",
        1: " STOP",
        2: " ignored",
        3: "",
    }

    def batch_decode(self, ids: Any) -> list[str]:
        return [self._pieces[row[0]] for row in ids]

    def decode(self, ids: Any, skip_special_tokens: bool = False) -> str:
        pieces = []
        for tid in ids:
            if skip_special_tokens and tid in self.all_special_ids:
                continue
            pieces.append(self._pieces[int(tid)])
        return "".join(pieces)


def _complete_finalizer_session(session: Any) -> None:
    """Install the current finalizer collaborator roster on a narrow stub."""
    if not hasattr(session, "_lens_probes"):
        session._lens_probes = {}
    if not hasattr(session, "_sae_probes"):
        session._sae_probes = {}
    session._scene_grammar = None
    session._scene_grammar_resolved = True


class _CurrentSessionStub(SaklasSession):
    """Narrow current-shape session whose lens is supplied in memory."""

    def __new__(cls) -> "_CurrentSessionStub":
        instance = super().__new__(cls)
        instance._lens_probes = {}
        instance._sae_probes = {}
        instance._live_lens = None
        instance._live_sae = None
        instance._live_lens_active_for_generation = True
        instance._live_sae_active_for_generation = True
        instance._generation_jlens = None
        instance._generation_jlens_active = False
        return instance

    @property
    def jlens(self) -> Any:
        return self._jlens


def _complete_capture_session(session: Any) -> None:
    """Install the current capture collaborator roster on a narrow stub."""
    if not hasattr(session, "_lens_probes"):
        session._lens_probes = {}
    if not hasattr(session, "_sae_probes"):
        session._sae_probes = {}
    if not hasattr(session, "_live_sae"):
        session._live_sae = None
    session._jlens = (
        object() if session._live_lens is not None or session._lens_probes else None
    )
    session._jlens_identity = None


class _EchoTokenizer:
    """Minimal tokenizer: encodes text to a tensor of its char codes."""

    name_or_path = "echo-tokenizer"

    def encode(self, text: str, return_tensors: str | None = None) -> Any:
        ids = [ord(c) for c in text]
        if return_tensors == "pt":
            return torch.tensor([ids], dtype=torch.long)
        return ids


def _decode_echo(ids: torch.Tensor) -> str:
    return "".join(chr(int(t)) for t in ids[0])


def test_prepare_generation_uses_session_thinking_default(monkeypatch: pytest.MonkeyPatch) -> None:
    session: Any = _CurrentSessionStub.__new__(_CurrentSessionStub)
    session._profiles = {}
    session._tokenizer = object()
    session._default_return_top_k = 0
    session.config = GenerationConfig(thinking=False)
    monkeypatch.setattr("saklas.core.session.supports_thinking", lambda _tok: True)

    _steering, use_thinking, *_rest = SaklasSession._prepare_generation_call(
        session, None, None, None,
    )
    assert use_thinking is False

    _steering, use_thinking, *_rest = SaklasSession._prepare_generation_call(
        session, Steering(alphas={}, thinking=True), None, None,
    )
    assert use_thinking is True

    _steering, use_thinking, *_rest = SaklasSession._prepare_generation_call(
        session, None, None, True,
    )
    assert use_thinking is True


def test_prepare_input_raw_feeds_flat_active_path():
    """raw=True walks the loom tree as flat text — no chat template, no
    role markers — and appends the call's own input."""
    from saklas.core.loom import LoomTree

    tree = LoomTree()
    u1 = tree.add_user_turn("once upon a ")
    a1 = tree.begin_assistant(u1)
    tree.finalize_assistant(a1, text="time")

    session: Any = _CurrentSessionStub.__new__(_CurrentSessionStub)
    session._tokenizer = _EchoTokenizer()
    session._device = torch.device("cpu")
    session.tree = tree

    # Non-stateless: prefix is the flattened active path; input rides on top.
    ids = SaklasSession._prepare_input(
        session, " the fox", raw=True, parent_node_id=a1,
    )
    assert _decode_echo(ids) == "once upon a time the fox"

    # Stateless: the tree is ignored — only the input string is encoded.
    ids = SaklasSession._prepare_input(
        session, " the fox", raw=True, stateless=True, parent_node_id=a1,
    )
    assert _decode_echo(ids) == " the fox"

    # Empty input is a bare continuation — just the flattened buffer.
    ids = SaklasSession._prepare_input(
        session, "", raw=True, parent_node_id=a1,
    )
    assert _decode_echo(ids) == "once upon a time"


class _StopModel:
    config = SimpleNamespace(vocab_size=4)
    generation_config = SimpleNamespace(eos_token_id=3)

    def __init__(self):
        self._tokens = [0, 1, 2]
        self._idx = 0

    def __call__(self, **_kwargs: Any) -> Any:
        tid = self._tokens[min(self._idx, len(self._tokens) - 1)]
        self._idx += 1
        logits = torch.full((1, 1, self.config.vocab_size), -100.0)
        logits[0, 0, tid] = 100.0
        return SimpleNamespace(logits=logits, past_key_values=object())


class _NoCacheModel(_StopModel):
    def __init__(self):
        super().__init__()
        self._tokens = [0, 1, 3]

    def __call__(self, **_kwargs: Any) -> Any:
        out = super().__call__(**_kwargs)
        out.past_key_values = None
        return out


class _SplitStopTokenizer(_StopTokenizer):
    name_or_path = "split-stop-tokenizer"
    vocab_size = 5
    _pieces = {
        0: "Hello S",
        1: "TOP",
        2: " ignored",
        3: "",
    }


class _SplitStopModel(_StopModel):
    config = SimpleNamespace(vocab_size=5)

    def __init__(self):
        self._tokens = [0, 1, 3]
        self._idx = 0


def test_stop_sequence_trimmed_text_is_final_result_text():
    model: Any = _StopModel()
    tokenizer: Any = _StopTokenizer()
    state = GenerationState()
    emitted: list[str] = []

    generated_ids = generate_steered(
        model,
        cast(Any, tokenizer),
        torch.tensor([[0]]),
        GenerationConfig(max_new_tokens=5, temperature=0.0),
        state,
        on_token=lambda text, *_args: emitted.append(text),
        stop=[" STOP"],
    )

    assert generated_ids == [0, 1]
    assert emitted == ["Hello"]
    assert state.finish_reason == "stop_sequence"
    assert state.response_text == "Hello"
    assert state.response_aggregate_index == 0

    session: Any = _CurrentSessionStub.__new__(_CurrentSessionStub)
    session._gen_state = state
    session._tokenizer = tokenizer
    session._monitor = SimpleNamespace(probe_names=[])
    session._capture = SimpleNamespace(stacked=lambda: {})
    session._capture_state = CaptureState(mode=CaptureMode.FULL)
    session._last_per_token_scores = None
    session._last_result = None
    session.build_readings = lambda: {}

    _complete_finalizer_session(session)
    result = SaklasSession._finalize_generation(
        session,
        generated_ids,
        elapsed=1.0,
        vector_snapshot={},
        stateless=True,
    )
    assert result.text == "Hello"


def test_stop_sequence_split_across_tokens_trims_final_text():
    model: Any = _SplitStopModel()
    tokenizer: Any = _SplitStopTokenizer()
    state = GenerationState()
    emitted: list[str] = []

    generated_ids = generate_steered(
        model,
        cast(Any, tokenizer),
        torch.tensor([[0]]),
        GenerationConfig(max_new_tokens=5, temperature=0.0),
        state,
        on_token=lambda text, *_args: emitted.append(text),
        stop=[" STOP"],
    )

    assert generated_ids == [0, 1]
    assert emitted == ["Hello S"]
    assert state.finish_reason == "stop_sequence"
    assert state.response_text == "Hello"
    assert state.response_aggregate_index == 0


def test_stop_sequence_probe_aggregate_uses_visible_endpoint():
    model: Any = _StopModel()
    tokenizer: Any = _StopTokenizer()
    state = GenerationState()
    generated_ids = generate_steered(
        model,
        cast(Any, tokenizer),
        torch.tensor([[0]]),
        GenerationConfig(max_new_tokens=5, temperature=0.0),
        state,
        on_token=lambda *_args: None,
        stop=[" STOP"],
    )

    visible = ProbeReading(coords=(0.25,), fraction=0.25, nearest=[])
    hidden_stop = ProbeReading(coords=(0.99,), fraction=0.99, nearest=[])

    class Capture:
        def stacked(self) -> dict[int, torch.Tensor]:
            raise AssertionError("incremental stop aggregate should not stack")

    class Monitor:
        probe_names = ("toy",)

        probe_names = ["toy"]

    session: Any = _CurrentSessionStub.__new__(_CurrentSessionStub)
    session._gen_state = state
    session._tokenizer = tokenizer
    session._monitor = Monitor()
    session._capture = Capture()
    session._capture_state = CaptureState(mode=CaptureMode.INCREMENTAL)
    session._incremental_readings = [{"toy": visible}, {"toy": hidden_stop}]
    session._last_per_token_scores = None
    session._last_result = None
    session.events = SimpleNamespace(emit=lambda _event: None)
    session.build_readings = lambda: {}

    _complete_finalizer_session(session)
    result = SaklasSession._finalize_generation(
        session,
        generated_ids,
        elapsed=1.0,
        vector_snapshot={},
        stateless=True,
    )

    assert result.text == "Hello"
    assert result.probe_readings == {"toy": visible}


def test_stop_sequence_only_tap_can_skip_full_token_table():
    class CountingStopTokenizer(_StopTokenizer):
        name_or_path = "stop-tokenizer-no-table"

        def __init__(self) -> None:
            self.batch_decode_calls = 0

        def batch_decode(self, ids: Any) -> list[str]:
            self.batch_decode_calls += 1
            return super().batch_decode(ids)

    model: Any = _StopModel()
    tokenizer = CountingStopTokenizer()
    state = GenerationState()
    emitted: list[str] = []

    generated_ids = generate_steered(
        model,
        cast(Any, tokenizer),
        torch.tensor([[0]]),
        GenerationConfig(max_new_tokens=5, temperature=0.0),
        state,
        on_token=lambda text, *_args: emitted.append(text),
        stop=[" STOP"],
        cache_token_text=False,
    )

    assert tokenizer.batch_decode_calls == 0
    assert generated_ids == [0, 1]
    assert emitted == ["Hello"]
    assert state.finish_reason == "stop_sequence"
    assert state.response_text == "Hello"


def test_finalize_reuses_scored_probe_aggregate() -> None:
    """Finalization should not rescore the same probe aggregate after
    ``score_per_token`` already returned the full ``ProbeReading``."""

    reading = ProbeReading(
        fraction=0.5,
        nearest=[("node", 0.25)],
        coords=(0.25,),
        fraction_per_layer={0: 0.5},
        coords_per_layer={0: (0.25,)},
    )

    class Capture:
        def __init__(self) -> None:
            self.calls = 0

        def stacked(self) -> dict[int, torch.Tensor]:
            self.calls += 1
            return {0: torch.randn(2, 4)}

    class Monitor:
        probe_names = ["toy"]

        def score_per_token(
            self,
            captured: dict[int, torch.Tensor],
            generated_ids: list[int],
            tokenizer: Any,
            *,
            accumulate: bool = True,
            aggregate_index: int | None = None,
        ) -> tuple[dict[str, ProbeReading], dict[str, list[float]]]:
            assert list(captured) == [0]
            assert generated_ids == [0, 1]
            assert accumulate is False
            assert aggregate_index == 1
            return {"toy": reading}, {"toy": [0.1, 0.25]}

        def score_aggregate(self, *_args: Any, **_kwargs: Any) -> None:
            raise AssertionError("aggregate was already scored")

    capture = Capture()
    state = GenerationState()
    state.finish_reason = "length"

    session: Any = _CurrentSessionStub.__new__(_CurrentSessionStub)
    session._gen_state = state
    session._tokenizer = _StopTokenizer()
    session._monitor = Monitor()
    session._capture = capture
    session._capture_state = CaptureState(mode=CaptureMode.FULL)
    session._last_per_token_scores = None
    session._last_result = None
    session.events = SimpleNamespace(emit=lambda _event: None)
    session.build_readings = lambda: {}

    _complete_finalizer_session(session)
    result = SaklasSession._finalize_generation(
        session,
        [0, 1],
        elapsed=1.0,
        vector_snapshot={},
        stateless=True,
    )

    assert capture.calls == 1
    assert result.probe_readings == {"toy": reading}
    assert session._last_per_token_scores == {"toy": [0.1, 0.25]}


def test_generate_stream_exposes_current_result() -> None:
    result = GenerationResult(
        text="ok", tokens=[7], token_count=1, tok_per_sec=1.0, elapsed=1.0,
    )

    def _fake_generate_core(*_args: Any, **kwargs: Any) -> GenerationResult:
        kwargs["on_token"]("ok", False, 7, None, None, None)
        return result

    session: Any = _CurrentSessionStub.__new__(_CurrentSessionStub)
    session._gen_state = GenerationState()
    session._monitor = SimpleNamespace(probe_names=[])
    session._last_token_probe_payload = {}
    session._generate_core = _fake_generate_core

    stream = session.generate_stream("prompt")
    events = list(stream)
    assert [e.text for e in events] == ["ok"]
    assert stream.result is result


def test_generate_stream_live_readouts_false_suppresses_readout_flags() -> None:
    result = GenerationResult(
        text="ok", tokens=[7], token_count=1, tok_per_sec=1.0, elapsed=1.0,
    )
    reading = ProbeReading(0.0, [], coords=(0.7,))
    flags: dict[str, bool] = {}

    session: Any = _CurrentSessionStub.__new__(_CurrentSessionStub)

    def _fake_generate_core(*_args: Any, **kwargs: Any) -> GenerationResult:
        on_token = kwargs["on_token"]
        flags["lens"] = on_token.options.lens_readout
        flags["sae"] = on_token.options.sae_readout
        session._last_token_probe_payload = {
                "probe_readings": {"sae/0": reading},
            "lens": {1: [("tok", 0.5)]},
            "lens_aggregate": [("tok", 0.5, 0.5, 0.0)],
            "sae": [(0, 1.0, None, None)],
        }
        on_token("ok", False, 7, None, None, None)
        return result

    session._gen_state = GenerationState()
    session._monitor = SimpleNamespace(
        probe_names=[],
        update_live=lambda _readings: None,
    )
    session._sae_probes = {"sae/0": {"feature_id": 0}}
    session._last_token_probe_payload = {}
    session._generate_core = _fake_generate_core

    events = list(session.generate_stream(
        "prompt", live_scores=True, live_readouts=False,
    ))

    assert flags == {"lens": False, "sae": False}
    assert events[0].probe_readings == {"sae/0": reading}
    assert events[0].lens_readout is None
    assert events[0].lens_aggregate is None
    assert events[0].sae_readout is None


def test_token_tap_skips_unconsumed_live_readout_helpers_and_empty_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import threading

    import saklas.core.token_payloads as token_payloads
    from saklas.core.triggers import TriggerContext

    session: Any = _CurrentSessionStub.__new__(_CurrentSessionStub)
    session._gen_lock = threading.Lock()
    session._gen_phase = GenState.IDLE
    session._gen_state = GenerationState()
    session._apply_cast_defaults = (
        lambda steering, sampling, thinking, **_kwargs: (
            steering,
            sampling,
            thinking,
        )
    )
    session._prepare_generation_call = lambda *_args: (
        None,
        False,
        GenerationConfig(max_new_tokens=1, temperature=0.0, top_p=1.0, top_k=None),
        None,
        None,
        None,
        None,
        0.0,
        0.0,
        None,
    )
    session._whitener = None
    session._seat_stop_augmentation = lambda stop_list, **_kwargs: stop_list
    session._monitor = SimpleNamespace(
        probe_names=[],
        set_subspace_coords=lambda _enabled: None,
        begin_live=lambda: None,
        end_live=lambda: None,
    )
    session._live_probe_scores = True
    session._trait_queues = []
    session._trait_lock = threading.Lock()
    composer = SimpleNamespace(
        _stack=[],
        steering_needs_probe_gating=lambda: False,
        gated_probe_keys=lambda: set(),
        gated_lens_probe_keys=lambda: set(),
        gated_sae_probe_keys=lambda: set(),
        gated_probe_names=lambda: set(),
    )
    session._steering_composer = composer
    session._start_loom_assistant = lambda *_args, **_kwargs: None
    session._snapshot_steering_alphas = lambda: {}
    session._generation_preamble = lambda *_args, **_kwargs: (
        torch.tensor([[1]], dtype=torch.long),
        False,
        1,
    )
    session.events = SimpleNamespace(emit=lambda _event: None)
    session._begin_capture = lambda **_kwargs: False
    session._steering = SimpleNamespace(
        ctx=TriggerContext(),
        reset_manifold_feet=lambda: None,
        has_compiled_offsets=lambda: False,
        zero_compiled_offsets=lambda: None,
    )
    session._capture_state = CaptureState()
    session._compiled = False
    session._static_cache_active = False
    session._capture_buffers = {}
    session._capture = SimpleNamespace(per_layer_buckets=lambda: {})
    session._incremental_readings = []
    session._incremental_gate_scores = []
    session._live_lens = {"layers": [0]}
    session._live_sae = {"layer": 0, "top_k": 3}
    session._live_lens_readout_step = lambda: (
        (_ for _ in ()).throw(AssertionError("lens readout not consumed"))
    )
    session._live_sae_readout_step = lambda: (
        (_ for _ in ()).throw(AssertionError("sae readout not consumed"))
    )
    session._last_lens_step_readings = None
    session._last_sae_step_readings = None
    session._last_token_probe_payload = {"stale": True}
    observed_payloads: list[Any] = []
    seen_tokens: list[str] = []
    payload_builder_calls = 0

    def _fail_empty_payload_build(**_kwargs: Any) -> Any:
        nonlocal payload_builder_calls
        payload_builder_calls += 1
        raise AssertionError("text-only token tap should not build empty payload")

    monkeypatch.setattr(
        token_payloads, "build_token_probe_payload", _fail_empty_payload_build,
    )

    def _run_generation_loop(
        _input_ids: Any,
        _gen_config: Any,
        *,
        effective_tap: Any,
        **_kwargs: Any,
    ) -> tuple[list[int], float]:
        assert effective_tap is not None
        effective_tap("x", False, 1, None, None, None)
        observed_payloads.append(session._last_token_probe_payload)
        return [1], 0.1

    session._run_generation_loop = _run_generation_loop
    session._finalize_generation = lambda *_args, **_kwargs: GenerationResult(
        text="x",
        tokens=[1],
        token_count=1,
        tok_per_sec=10.0,
        elapsed=0.1,
    )
    session._end_capture = lambda: None
    session._active_gen_reservation = None

    def _on_token(
        text: str,
        _thinking: bool,
        _tid: int | None,
        _lp: float | None,
        _top_alts: Any,
        _perplexity: float | None,
    ) -> None:
        seen_tokens.append(text)

    result = SaklasSession._generate_core(
        session,
        "prompt",
        stateless=True,
        on_token=_on_token,
    )

    assert result.text == "x"
    assert seen_tokens == ["x"]
    assert observed_payloads == [None]
    assert payload_builder_calls == 0


def test_finalize_incremental_probe_path_does_not_stack_capture() -> None:
    reading0 = ProbeReading(
        fraction=0.2,
        nearest=[("node", 0.4)],
        coords=(0.1,),
        fraction_per_layer={0: 0.2},
        coords_per_layer={0: (0.1,)},
    )
    reading1 = ProbeReading(
        fraction=0.5,
        nearest=[("node", 0.25)],
        coords=(0.25,),
        fraction_per_layer={0: 0.5},
        coords_per_layer={0: (0.25,)},
    )

    class Capture:
        def stacked(self) -> dict[int, torch.Tensor]:
            raise AssertionError("incremental probe finalization should not stack")

    class Monitor:
        probe_names = ["toy"]

    state = GenerationState()
    state.finish_reason = "length"

    session: Any = _CurrentSessionStub.__new__(_CurrentSessionStub)
    session._gen_state = state
    session._tokenizer = _StopTokenizer()
    session._monitor = Monitor()
    session._capture = Capture()
    session._capture_state = CaptureState(mode=CaptureMode.INCREMENTAL)
    session._incremental_readings = [
        {"toy": reading0},
        {"toy": reading1},
    ]
    session._last_per_token_scores = None
    session._last_result = None
    session.events = SimpleNamespace(emit=lambda _event: None)
    session.build_readings = lambda: {}

    _complete_finalizer_session(session)
    result = SaklasSession._finalize_generation(
        session,
        [0, 1],
        elapsed=1.0,
        vector_snapshot={},
        stateless=True,
    )

    assert result.probe_readings == {"toy": reading1}
    assert session._last_per_token_scores == {"toy": [0.1, 0.25]}


def test_finalize_lean_incremental_probe_path() -> None:
    """LEAN_INCREMENTAL: per-token coord stream from lean rows + full aggregate
    re-scored from the tail ring.

    Key invariants:
    1. The capture's ``stacked()`` is never called (lean uses the tail ring).
    2. The per-token coord stream is populated from ``_incremental_readings``
       (the lean per-token rows stored by the step sink).
    3. The aggregate in the result IS the full ``ProbeReading`` (with ``nearest``
       populated), NOT a lean coords-only row — ``_score_lean_incremental``
       re-scores via ``_score_aggregate_only`` → ``monitor.score_aggregate``.
    4. ``_last_per_token_scores`` holds the coord stream (not the lean reading).
    """

    # Lean per-token rows: only coords populated (nearest empty — the lean path).
    lean0 = ProbeReading(
        fraction=0.15,
        nearest=[],
        coords=(0.1,),
        fraction_per_layer={},
        coords_per_layer={},
    )
    lean1 = ProbeReading(
        fraction=0.3,
        nearest=[],
        coords=(0.3,),
        fraction_per_layer={},
        coords_per_layer={},
    )
    # Full aggregate reading from the tail ring re-score.
    full_agg = ProbeReading(
        fraction=0.3,
        nearest=[("node", 0.2)],
        coords=(0.3,),
        fraction_per_layer={0: 0.3},
        coords_per_layer={0: (0.3,)},
    )

    pooled_slice = {0: torch.randn(4)}

    class Capture:
        def stacked(self) -> dict[int, torch.Tensor]:
            raise AssertionError("LEAN_INCREMENTAL must not call stacked()")

        def tail_slice_at(self, _idx: int) -> dict[int, torch.Tensor]:
            return pooled_slice

    class Monitor:
        probe_names = ["toy"]
        _agg_calls = 0

        def enable_curved_warm(self, _flag: bool) -> None:
            pass

        def score_aggregate(
            self, captured: dict[int, torch.Tensor], **_kw: Any
        ) -> dict[str, ProbeReading]:
            self._agg_calls += 1
            assert captured is pooled_slice
            return {"toy": full_agg}

        def accumulate_readings(self, _vals: Any) -> None:
            pass

    monitor = Monitor()
    state = GenerationState()
    state.finish_reason = "length"

    session: Any = _CurrentSessionStub.__new__(_CurrentSessionStub)
    session._gen_state = state
    session._tokenizer = _StopTokenizer()
    session._monitor = monitor
    session._capture = Capture()
    session._capture_state = CaptureState(mode=CaptureMode.LEAN_INCREMENTAL)
    session._incremental_readings = [
        {"toy": lean0},
        {"toy": lean1},
    ]
    session._last_per_token_scores = None
    session._last_result = None
    session.events = SimpleNamespace(emit=lambda _event: None)
    session.build_readings = lambda: {}

    _complete_finalizer_session(session)
    result = SaklasSession._finalize_generation(
        session,
        [0, 1],
        elapsed=1.0,
        vector_snapshot={},
        stateless=True,
    )

    # Invariant 3: aggregate carries the full reading (nearest populated).
    assert result.probe_readings == {"toy": full_agg}
    assert result.probe_readings["toy"].nearest == [("node", 0.2)]
    # Invariant 2+4: per-token coord stream extracted from lean rows.
    assert session._last_per_token_scores == {"toy": [0.1, 0.3]}
    # Invariant 3: monitor.score_aggregate was called exactly once.
    assert monitor._agg_calls == 1


def test_finalize_gating_subset_probe_path() -> None:
    """GATING_SUBSET: per-token scoring is only for probe gates (subset scalars);
    the full-roster aggregate is pooled once from the tail ring at finalize.

    Key invariants:
    1. ``stacked()`` is never called.
    2. ``per_token`` in the result is empty (GATING_SUBSET shares the
       ``aggregate_only`` finalize path — no full per-token stream).
    3. The aggregate reading IS the full ``ProbeReading`` from ``score_aggregate``.
    4. ``_last_per_token_scores`` is None (no full per-token coord stream).
    """

    full_agg = ProbeReading(
        fraction=0.6,
        nearest=[("confident", 0.1)],
        coords=(0.6,),
        fraction_per_layer={0: 0.6},
        coords_per_layer={0: (0.6,)},
    )
    pooled_slice = {0: torch.randn(4)}

    class Capture:
        def stacked(self) -> dict[int, torch.Tensor]:
            raise AssertionError("GATING_SUBSET must not call stacked()")

        def tail_slice_at(self, _idx: int) -> dict[int, torch.Tensor]:
            return pooled_slice

    class Monitor:
        probe_names = ["toy"]
        _agg_calls = 0

        def enable_curved_warm(self, _flag: bool) -> None:
            pass

        def score_aggregate(
            self, captured: dict[int, torch.Tensor], **_kw: Any
        ) -> dict[str, ProbeReading]:
            self._agg_calls += 1
            assert captured is pooled_slice
            return {"toy": full_agg}

        def accumulate_readings(self, _vals: Any) -> None:
            pass

    monitor = Monitor()
    state = GenerationState()
    state.finish_reason = "length"

    session: Any = _CurrentSessionStub.__new__(_CurrentSessionStub)
    session._gen_state = state
    session._tokenizer = _StopTokenizer()
    session._monitor = monitor
    session._capture = Capture()
    # GATING_SUBSET: CaptureState.aggregate_only returns True for this mode.
    session._capture_state = CaptureState(
        mode=CaptureMode.GATING_SUBSET,
        gating_subset={"toy"},
        gating_keys={"toy"},
    )
    session._incremental_readings = []  # gate scores not used by finalize
    session._last_per_token_scores = None
    session._last_result = None
    session.events = SimpleNamespace(emit=lambda _event: None)
    session.build_readings = lambda: {}

    _complete_finalizer_session(session)
    result = SaklasSession._finalize_generation(
        session,
        [0, 1],
        elapsed=1.0,
        vector_snapshot={},
        stateless=True,
    )

    # Invariant 3: full aggregate reading from tail ring.
    assert result.probe_readings == {"toy": full_agg}
    assert result.probe_readings["toy"].nearest == [("confident", 0.1)]
    # Invariant 2+4: no per-token coord stream.
    assert session._last_per_token_scores is None
    # Invariant 3: score_aggregate called exactly once.
    assert monitor._agg_calls == 1


def test_finalize_reuses_one_aggregate_pool_for_monitor_lens_and_sae() -> None:
    pooled = {1: torch.ones(4)}

    class Capture:
        def __init__(self) -> None:
            self.tail_calls: list[int] = []

        def tail_slice_at(self, idx: int) -> dict[int, torch.Tensor]:
            self.tail_calls.append(idx)
            return pooled

        def stacked(self) -> dict[int, torch.Tensor]:
            raise AssertionError("shared aggregate pool should avoid stacked fallback")

    class Monitor:
        probe_names = ["toy"]

    capture = Capture()
    monitor_reading = ProbeReading(0.0, [], coords=(0.1,))
    lens_reading = ProbeReading(0.0, [], coords=(0.2,))
    sae_reading = ProbeReading(0.0, [], coords=(0.3,))
    expected_pooled = pooled
    seen: list[tuple[str, bool]] = []

    def score_aggregate_only(
        _generated_ids: list[int],
        *,
        accumulate: bool = True,
        pooled: dict[int, torch.Tensor] | None = None,
    ) -> dict[str, ProbeReading]:
        del accumulate
        seen.append(("monitor", pooled is expected_pooled))
        return {"toy": monitor_reading}

    def score_lens(
        _generated_ids: list[int],
        *,
        pooled: dict[int, torch.Tensor] | None = None,
    ) -> dict[str, ProbeReading]:
        seen.append(("lens", pooled is expected_pooled))
        return {"jlens/g": lens_reading}

    def score_sae(
        _generated_ids: list[int],
        *,
        pooled: dict[int, torch.Tensor] | None = None,
    ) -> dict[str, ProbeReading]:
        seen.append(("sae", pooled is expected_pooled))
        return {"sae/0": sae_reading}

    state = GenerationState()
    state.finish_reason = "length"

    session: Any = _CurrentSessionStub.__new__(_CurrentSessionStub)
    session._gen_state = state
    session._tokenizer = _StopTokenizer()
    session._monitor = Monitor()
    session._capture = capture
    session._capture_state = CaptureState(mode=CaptureMode.AGGREGATE_ONLY)
    session._lens_probes = {"jlens/g": {}}
    session._sae_probes = {"sae/0": {}}
    session._aggregate_forward_index = lambda _ids: 1
    session._score_aggregate_only = score_aggregate_only
    session._score_lens_probes_aggregate = score_lens
    session._score_sae_probes_aggregate = score_sae
    session._last_result = None
    session._last_per_token_scores = None
    session.events = SimpleNamespace(emit=lambda _event: None)

    _complete_finalizer_session(session)
    result = SaklasSession._finalize_generation(
        session,
        [0, 1],
        elapsed=1.0,
        vector_snapshot={},
        stateless=True,
    )

    assert capture.tail_calls == [1]
    assert seen == [("monitor", True), ("lens", True), ("sae", True)]
    assert result.probe_readings == {
        "toy": monitor_reading,
        "jlens/g": lens_reading,
        "sae/0": sae_reading,
    }


def test_gating_callback_backfills_exact_keys_hidden_by_top_n() -> None:
    """Full per-token readings can truncate label channels; gates still need the
    exact requested scalar keys."""

    from saklas.core.steering_composer import SteeringComposer

    class Capture:
        def latest_per_layer(self) -> dict[int, torch.Tensor]:
            return {0: torch.ones(4)}

    class Monitor:
        probe_names = ("toy",)

        def __init__(self) -> None:
            self.plan_calls: list[frozenset[str]] = []
            self.score_calls: list[tuple[str, frozenset[str]]] = []

        def flat_scalars(self, _readings: Any) -> dict[str, float]:
            return {"toy@nearest": -0.1}

        def plan_gate_scalars(
            self,
            gate_keys: set[str],
            *,
            probe_names: set[str] | None = None,
        ) -> tuple[str, frozenset[str]]:
            assert probe_names is None
            planned = frozenset(gate_keys)
            self.plan_calls.append(planned)
            return ("planned", planned)

        def score_planned_gate_scalars(
            self,
            _latest: dict[int, torch.Tensor],
            plan: tuple[str, frozenset[str]],
        ) -> dict[str, float]:
            self.score_calls.append(plan)
            assert plan == ("planned", frozenset({"toy@hidden"}))
            return {"toy@hidden": -2.0}

        def score_gate_scalars(
            self,
            _latest: dict[int, torch.Tensor],
            gate_keys: set[str],
            *,
            probe_names: set[str] | None = None,
        ) -> dict[str, float]:
            raise AssertionError("missing gate keys should use cached plans")

    monitor = Monitor()
    session: Any = SimpleNamespace(
        _capture=Capture(),
        _monitor=monitor,
        _capture_state=CaptureState(
            mode=CaptureMode.INCREMENTAL,
            gating_keys={"toy@nearest", "toy@hidden"},
        ),
        _incremental_readings=[{"toy": object()}],
        _incremental_gate_scores=[],
        _lens_probes={},
        _sae_probes={},
        _profiles={},
    )

    callback = SteeringComposer(session).build_gating_score_callback()
    scores = callback()
    scores_again = callback()

    assert scores == {"toy@nearest": -0.1, "toy@hidden": -2.0}
    assert scores_again == scores
    assert monitor.plan_calls == [frozenset({"toy@hidden"})]
    assert monitor.score_calls == [
        ("planned", frozenset({"toy@hidden"})),
        ("planned", frozenset({"toy@hidden"})),
    ]


@pytest.mark.parametrize(
    ("expr", "registry_attr", "score_attr", "expected"),
    [
        (
            "0.3 jlens/g@when:jlens/g>0.4",
            "_lens_probes",
            "_score_lens_gate_scalars",
            {"jlens/g": 0.7},
        ),
        (
            "0.3 sae/2@when:sae/2>0.4",
            "_sae_probes",
            "_score_sae_gate_scalars",
            {"sae/2": 0.8},
        ),
    ],
)
def test_readout_only_gates_skip_monitor_probe_scoring(
    expr: str,
    registry_attr: str,
    score_attr: str,
    expected: dict[str, float],
) -> None:
    from saklas.core.steering_composer import SteeringComposer
    from saklas.core.steering_expr import parse_expr

    class Capture:
        def latest_per_layer(self) -> dict[int, torch.Tensor]:
            raise AssertionError("monitor path should not inspect captures")

    class Monitor:
        probe_names = ("toy",)

        def flat_scalars(self, _readings: Any) -> dict[str, float]:
            raise AssertionError("readout-only gates should skip monitor scalars")

        def score_gate_scalars(self, *_args: Any, **_kwargs: Any) -> dict[str, float]:
            raise AssertionError("readout-only gates should skip monitor gates")

        def score_single_token(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
            raise AssertionError("readout-only gates should skip monitor probes")

    session: Any = SimpleNamespace(
        _capture=Capture(),
        _monitor=Monitor(),
        _capture_state=CaptureState(),
        _incremental_readings=[],
        _incremental_gate_scores=[],
        _lens_probes={},
        _sae_probes={},
    )
    setattr(session, registry_attr, {next(iter(expected)): {}})
    setattr(session, score_attr, lambda _keys: dict(expected))
    other_score_attr = (
        "_score_sae_gate_scalars"
        if score_attr == "_score_lens_gate_scalars"
        else "_score_lens_gate_scalars"
    )
    setattr(session, other_score_attr, lambda _keys: {})
    composer = SteeringComposer(session)
    composer._stack.append(cast(Any, dict(parse_expr(expr).alphas)))

    assert composer.build_gating_score_callback()() == expected


def test_mixed_monitor_and_lens_gates_score_only_monitor_gate_keys() -> None:
    from saklas.core.steering_composer import SteeringComposer
    from saklas.core.steering_expr import parse_expr

    class Capture:
        def latest_per_layer(self) -> dict[int, torch.Tensor]:
            return {0: torch.ones(4)}

    class Monitor:
        probe_names = ("toy", "other")

        def __init__(self) -> None:
            self.requested: set[str] | None = None

        def plan_gate_scalars(
            self,
            gate_keys: set[str],
            *,
            probe_names: set[str] | None = None,
        ) -> frozenset[str]:
            assert probe_names is None
            self.requested = set(gate_keys)
            return frozenset(gate_keys)

        def score_planned_gate_scalars(
            self,
            _latest: dict[int, torch.Tensor],
            plan: frozenset[str],
        ) -> dict[str, float]:
            assert plan == frozenset({"toy"})
            return {"toy": 0.2}

        def score_single_token(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
            raise AssertionError("exact monitor gate keys should avoid full scoring")

    monitor = Monitor()
    session: Any = SimpleNamespace(
        _capture=Capture(),
        _monitor=monitor,
        _capture_state=CaptureState(),
        _incremental_readings=[],
        _incremental_gate_scores=[],
        _lens_probes={"jlens/g": {}},
        _sae_probes={},
        _score_lens_gate_scalars=lambda _keys: {"jlens/g": 0.7},
        _score_sae_gate_scalars=lambda _keys: {},
    )
    composer = SteeringComposer(session)
    steering = parse_expr(
        "0.3 toy@when:toy>0.1 + 0.2 jlens/g@when:jlens/g>0.4",
    )
    composer._stack.append(cast(Any, dict(steering.alphas)))

    scores = composer.build_gating_score_callback()()

    assert scores == {"toy": 0.2, "jlens/g": 0.7}
    assert monitor.requested == {"toy"}


def test_stateless_zero_token_probe_result_does_not_use_history() -> None:
    """A stateless empty generation has no current-run probe aggregate."""

    class Capture:
        def stacked(self) -> dict[int, torch.Tensor]:
            raise AssertionError("no generated tokens should not stack capture")

    class Monitor:
        probe_names = ["toy"]

    state = GenerationState()
    state.finish_reason = "length"

    session: Any = _CurrentSessionStub.__new__(_CurrentSessionStub)
    session._gen_state = state
    session._tokenizer = _StopTokenizer()
    session._monitor = Monitor()
    session._capture = Capture()
    session._capture_state = CaptureState(mode=CaptureMode.FULL)
    session._last_per_token_scores = {"toy": [99.0]}
    session._last_result = None
    session.events = SimpleNamespace(emit=lambda _event: None)
    session.build_readings = lambda: {
        "toy": SimpleNamespace(mean=(42.0,), to_dict=lambda: {})
    }

    _complete_finalizer_session(session)
    result = SaklasSession._finalize_generation(
        session,
        [],
        elapsed=1.0,
        vector_snapshot={},
        stateless=True,
    )

    assert result.probe_readings == {}
    assert session._last_per_token_scores is None


def test_return_probe_readings_false_skips_probe_finalization() -> None:
    class Capture:
        def stacked(self) -> dict[int, torch.Tensor]:
            raise AssertionError("probe finalization disabled")

    class Monitor:
        probe_names = ["toy"]

    state = GenerationState()
    state.finish_reason = "length"

    session: Any = _CurrentSessionStub.__new__(_CurrentSessionStub)
    session._gen_state = state
    session._tokenizer = _StopTokenizer()
    session._monitor = Monitor()
    session._lens_probes = {"jlens/x": {}}
    session._sae_probes = {"sae/0": {}}
    session._capture = Capture()
    session._capture_state = CaptureState(mode=CaptureMode.FULL)
    session._last_per_token_scores = None
    session._last_result = None
    session.events = SimpleNamespace(emit=lambda _event: None)
    session.build_readings = lambda: {
        "toy": SimpleNamespace(mean=(42.0,), to_dict=lambda: {})
    }
    session._score_lens_probes_aggregate = lambda _ids: (
        (_ for _ in ()).throw(AssertionError("lens finalization disabled"))
    )
    session._score_sae_probes_aggregate = lambda _ids: (
        (_ for _ in ()).throw(AssertionError("sae finalization disabled"))
    )

    _complete_finalizer_session(session)
    result = SaklasSession._finalize_generation(
        session,
        [0],
        elapsed=1.0,
        vector_snapshot={},
        stateless=True,
        return_probe_readings=False,
    )

    assert result.probe_readings == {}


def test_lens_only_without_final_probe_aggregate_keeps_latest_tail() -> None:
    class Capture:
        def __init__(self) -> None:
            self.attached: list[int] | None = None
            self.aggregate_depth: int | None = None

        def clear(self) -> None:
            pass

        def attach(self, _layers: Any, layer_idxs: list[int]) -> None:
            self.attached = list(layer_idxs)

        def set_aggregate_tail(self, depth: int) -> None:
            self.aggregate_depth = depth

    class Monitor:
        probe_names: list[str] = []

        def probe_layers(self, _names: set[str] | None = None) -> set[int]:
            return set()

        def enable_curved_warm(self, _enabled: bool) -> None:
            pass

    capture = Capture()
    session: Any = _CurrentSessionStub.__new__(_CurrentSessionStub)
    session._layers = [object()] * 8
    session._monitor = Monitor()
    session._capture = capture
    session._capture_buffers = {}
    session._compiled_clean_eligible = False
    session._steering_uses_compiled_offsets = False
    session._live_lens = {"layers": [2, 4]}
    session._lens_probes = {}
    session._live_sae = None
    session._sae_probes = {}
    session._steering = SimpleNamespace(all_fast_path=lambda: True)

    _complete_capture_session(session)
    SaklasSession._begin_capture(
        session,
        need_per_token=False,
        final_probe_aggregate=False,
    )

    assert capture.attached == [2, 4]
    assert capture.aggregate_depth == 1


def test_dormant_lens_probe_without_final_aggregate_does_not_attach_capture() -> None:
    class Capture:
        def __init__(self) -> None:
            self.attached: list[int] | None = None

        def clear(self) -> None:
            pass

        def attach(self, _layers: Any, layer_idxs: list[int]) -> None:
            self.attached = list(layer_idxs)

    class Monitor:
        probe_names: list[str] = []

        def probe_layers(self, _names: set[str] | None = None) -> set[int]:
            return set()

    capture = Capture()
    session: Any = _CurrentSessionStub.__new__(_CurrentSessionStub)
    session._layers = [object()] * 8
    session._monitor = Monitor()
    session._capture = capture
    session._live_lens = None
    session._lens_probes = {"jlens/g": {"layers": [3]}}
    session._live_sae = None
    session._sae_probes = {}

    _complete_capture_session(session)
    attached = SaklasSession._begin_capture(
        session,
        need_per_token=False,
        final_probe_aggregate=False,
    )

    assert attached is False
    assert capture.attached is None
    assert session._capture_state.final_probe_aggregate is False


def test_lens_gate_without_final_aggregate_attaches_gated_probe_layers() -> None:
    class Capture:
        def __init__(self) -> None:
            self.attached: list[int] | None = None
            self.aggregate_depth: int | None = None

        def clear(self) -> None:
            pass

        def attach(self, _layers: Any, layer_idxs: list[int]) -> None:
            self.attached = list(layer_idxs)

        def set_aggregate_tail(self, depth: int) -> None:
            self.aggregate_depth = depth

    class Monitor:
        probe_names: list[str] = []

        def probe_layers(self, _names: set[str] | None = None) -> set[int]:
            return set()

        def enable_curved_warm(self, _enabled: bool) -> None:
            pass

    capture = Capture()
    session: Any = _CurrentSessionStub.__new__(_CurrentSessionStub)
    session._layers = [object()] * 8
    session._monitor = Monitor()
    session._capture = capture
    session._capture_buffers = {}
    session._compiled_clean_eligible = False
    session._steering_uses_compiled_offsets = False
    session._live_lens = None
    session._lens_probes = {
        "jlens/g": {"layers": [3]},
        "jlens/h": {"layers": [6]},
    }
    session._live_sae = None
    session._sae_probes = {}
    session._steering = SimpleNamespace(all_fast_path=lambda: True)

    _complete_capture_session(session)
    attached = SaklasSession._begin_capture(
        session,
        need_per_token=False,
        final_probe_aggregate=False,
        lens_gating_probe_keys={"jlens/g"},
    )

    assert attached is True
    assert capture.attached == [3]
    assert capture.aggregate_depth == 1


def test_sae_only_without_final_probe_aggregate_keeps_latest_tail() -> None:
    class Capture:
        def __init__(self) -> None:
            self.attached: list[int] | None = None
            self.aggregate_depth: int | None = None

        def clear(self) -> None:
            pass

        def attach(self, _layers: Any, layer_idxs: list[int]) -> None:
            self.attached = list(layer_idxs)

        def set_aggregate_tail(self, depth: int) -> None:
            self.aggregate_depth = depth

    class Monitor:
        probe_names: list[str] = []

        def probe_layers(self, _names: set[str] | None = None) -> set[int]:
            return set()

        def enable_curved_warm(self, _enabled: bool) -> None:
            pass

    capture = Capture()
    session: Any = _CurrentSessionStub.__new__(_CurrentSessionStub)
    session._layers = [object()] * 8
    session._monitor = Monitor()
    session._capture = capture
    session._capture_buffers = {}
    session._compiled_clean_eligible = False
    session._steering_uses_compiled_offsets = False
    session._live_lens = None
    session._lens_probes = {}
    session._live_sae = {"layer": 5, "top_k": 8}
    session._sae_probes = {}
    session._steering = SimpleNamespace(all_fast_path=lambda: True)

    _complete_capture_session(session)
    SaklasSession._begin_capture(
        session,
        need_per_token=False,
        final_probe_aggregate=False,
    )

    assert capture.attached == [5]
    assert capture.aggregate_depth == 1


def test_dormant_sae_probe_without_final_aggregate_does_not_attach_capture() -> None:
    class Capture:
        def __init__(self) -> None:
            self.attached: list[int] | None = None

        def clear(self) -> None:
            pass

        def attach(self, _layers: Any, layer_idxs: list[int]) -> None:
            self.attached = list(layer_idxs)

    class Monitor:
        probe_names: list[str] = []

        def probe_layers(self, _names: set[str] | None = None) -> set[int]:
            return set()

    capture = Capture()
    session: Any = _CurrentSessionStub.__new__(_CurrentSessionStub)
    session._layers = [object()] * 8
    session._monitor = Monitor()
    session._capture = capture
    session._live_lens = None
    session._lens_probes = {}
    session._live_sae = None
    session._sae_probes = {"sae/0": {"feature_id": 0}}
    session._sae_layer = 5

    _complete_capture_session(session)
    attached = SaklasSession._begin_capture(
        session,
        need_per_token=False,
        final_probe_aggregate=False,
    )

    assert attached is False
    assert capture.attached is None
    assert session._capture_state.final_probe_aggregate is False


def test_sae_gate_without_final_aggregate_attaches_sae_layer() -> None:
    class Capture:
        def __init__(self) -> None:
            self.attached: list[int] | None = None
            self.aggregate_depth: int | None = None

        def clear(self) -> None:
            pass

        def attach(self, _layers: Any, layer_idxs: list[int]) -> None:
            self.attached = list(layer_idxs)

        def set_aggregate_tail(self, depth: int) -> None:
            self.aggregate_depth = depth

    class Monitor:
        probe_names: list[str] = []

        def probe_layers(self, _names: set[str] | None = None) -> set[int]:
            return set()

        def enable_curved_warm(self, _enabled: bool) -> None:
            pass

    capture = Capture()
    session: Any = _CurrentSessionStub.__new__(_CurrentSessionStub)
    session._layers = [object()] * 8
    session._monitor = Monitor()
    session._capture = capture
    session._capture_buffers = {}
    session._compiled_clean_eligible = False
    session._steering_uses_compiled_offsets = False
    session._live_lens = None
    session._lens_probes = {}
    session._live_sae = None
    session._sae_probes = {"sae/0": {"feature_id": 0}}
    session._sae_layer = 5
    session._steering = SimpleNamespace(all_fast_path=lambda: True)

    _complete_capture_session(session)
    attached = SaklasSession._begin_capture(
        session,
        need_per_token=False,
        final_probe_aggregate=False,
        sae_gating_probe_keys={"sae/0"},
    )

    assert attached is True
    assert capture.attached == [5]
    assert capture.aggregate_depth == 1


def test_monitor_probe_without_final_aggregate_and_no_per_token_skips_capture() -> None:
    class Capture:
        def __init__(self) -> None:
            self.attached: list[int] | None = None

        def attach(self, *_args: Any, **_kwargs: Any) -> None:
            raise AssertionError("dormant monitor probes should not attach capture")

    class Monitor:
        probe_names = ["monitor"]

        def probe_layers(self, _names: set[str] | None = None) -> set[int]:
            raise AssertionError("dormant monitor probes should not widen capture")

    capture = Capture()
    session: Any = _CurrentSessionStub.__new__(_CurrentSessionStub)
    session._layers = [object()] * 8
    session._monitor = Monitor()
    session._capture = capture
    session._live_lens = None
    session._lens_probes = {}
    session._live_sae = None
    session._sae_probes = {}
    session._sae_layer = None

    _complete_capture_session(session)
    attached = SaklasSession._begin_capture(
        session,
        need_per_token=False,
        final_probe_aggregate=False,
    )

    assert attached is False
    assert capture.attached is None
    assert session._capture_state.final_probe_aggregate is False


def test_monitor_probe_final_aggregate_still_attaches_capture() -> None:
    class Capture:
        def __init__(self) -> None:
            self.attached: list[int] | None = None
            self.aggregate_depth: int | None = None

        def clear(self) -> None:
            pass

        def attach(self, _layers: Any, layer_idxs: list[int]) -> None:
            self.attached = list(layer_idxs)

        def set_aggregate_tail(self, depth: int) -> None:
            self.aggregate_depth = depth

    class Monitor:
        probe_names = ["monitor"]

        def __init__(self) -> None:
            self.layer_query: set[str] | None = None
            self.warm_enabled: bool | None = None

        def probe_layers(self, names: set[str] | None = None) -> set[int]:
            self.layer_query = names
            return {2, 4}

        def enable_curved_warm(self, enabled: bool) -> None:
            self.warm_enabled = enabled

    capture = Capture()
    monitor = Monitor()
    session: Any = _CurrentSessionStub.__new__(_CurrentSessionStub)
    session._layers = [object()] * 8
    session._monitor = monitor
    session._capture = capture
    session._capture_buffers = {}
    session._compiled_clean_eligible = False
    session._steering_uses_compiled_offsets = False
    session._live_lens = None
    session._lens_probes = {}
    session._live_sae = None
    session._sae_probes = {}
    session._sae_layer = None
    session._steering = SimpleNamespace(all_fast_path=lambda: True)

    _complete_capture_session(session)
    attached = SaklasSession._begin_capture(
        session,
        need_per_token=False,
        final_probe_aggregate=True,
    )

    assert attached is True
    assert monitor.layer_query is None
    assert capture.attached == [2, 4]
    assert capture.aggregate_depth is not None and capture.aggregate_depth > 1
    assert session._capture_state.mode is CaptureMode.AGGREGATE_ONLY
    assert monitor.warm_enabled is False


def test_gate_only_without_final_probe_aggregate_narrows_capture_layers() -> None:
    class Capture:
        def __init__(self) -> None:
            self.attached: list[int] | None = None
            self.incremental = False
            self.tail_with_sink = False

        def clear(self) -> None:
            pass

        def attach(self, _layers: Any, layer_idxs: list[int]) -> None:
            self.attached = list(layer_idxs)

        def set_incremental(self, _sink: Any) -> None:
            self.incremental = True

        def set_tail_with_sink(self, *_args: Any, **_kwargs: Any) -> None:
            self.tail_with_sink = True

    class Monitor:
        probe_names = ["gate", "other"]

        def __init__(self) -> None:
            self.layer_query: set[str] | None = None

        def probe_layers(self, names: set[str] | None = None) -> set[int]:
            self.layer_query = names
            return {2} if names == {"gate"} else {2, 5}

        def reset_curved_feet(self) -> None:
            pass

        def enable_curved_warm(self, _enabled: bool) -> None:
            pass

        def plan_gate_scalars(
            self,
            gate_keys: set[str],
            *,
            probe_names: set[str] | None = None,
        ) -> tuple[str, ...]:
            assert gate_keys == {"gate@x"}
            assert probe_names == {"gate"}
            return ("plan",)

        def score_planned_gate_scalars(
            self,
            _latest: dict[int, torch.Tensor],
            _plan: tuple[str, ...],
        ) -> dict[str, float]:
            return {}

    capture = Capture()
    monitor = Monitor()
    session: Any = _CurrentSessionStub.__new__(_CurrentSessionStub)
    session._layers = [object()] * 8
    session._monitor = monitor
    session._capture = capture
    session._capture_buffers = {}
    session._compiled_clean_eligible = False
    session._steering_uses_compiled_offsets = False
    session._live_lens = None

    _complete_capture_session(session)
    SaklasSession._begin_capture(
        session,
        need_per_token=True,
        gating_only_probes={"gate"},
        gating_probe_keys={"gate@x"},
        final_probe_aggregate=False,
    )

    assert monitor.layer_query == {"gate"}
    assert capture.attached == [2]
    assert capture.incremental is True
    assert capture.tail_with_sink is False
    assert session._capture_state.mode is CaptureMode.GATING_SUBSET
    assert session._capture_state.final_probe_aggregate is False


def test_gate_only_capture_reuses_preplanned_gate_scalars() -> None:
    plan = object()

    class Capture:
        def __init__(self) -> None:
            self.sink: Any | None = None

        def clear(self) -> None:
            pass

        def attach(self, _layers: Any, _layer_idxs: list[int]) -> None:
            pass

        def set_incremental(self, sink: Any) -> None:
            self.sink = sink

    class Monitor:
        probe_names = ["gate"]

        def __init__(self) -> None:
            self.plan_calls: list[tuple[set[str], set[str] | None]] = []
            self.score_plans: list[tuple[Any, ...]] = []

        def probe_layers(self, names: set[str] | None = None) -> set[int]:
            assert names == {"gate"}
            return {2}

        def reset_curved_feet(self) -> None:
            pass

        def enable_curved_warm(self, _enabled: bool) -> None:
            pass

        def plan_gate_scalars(
            self,
            gate_keys: set[str],
            *,
            probe_names: set[str] | None = None,
        ) -> tuple[Any, ...]:
            self.plan_calls.append((
                set(gate_keys),
                set(probe_names) if probe_names is not None else None,
            ))
            return (plan,)

        def score_planned_gate_scalars(
            self,
            _latest: dict[int, torch.Tensor],
            gate_plan: tuple[Any, ...],
        ) -> dict[str, float]:
            self.score_plans.append(gate_plan)
            return {"gate": 0.25}

        def score_gate_scalars(self, *_args: Any, **_kwargs: Any) -> dict[str, float]:
            raise AssertionError("GATING_SUBSET should reuse the planned scalars")

    capture = Capture()
    monitor = Monitor()
    session: Any = _CurrentSessionStub.__new__(_CurrentSessionStub)
    session._layers = [object()] * 8
    session._monitor = monitor
    session._capture = capture
    session._capture_buffers = {}
    session._compiled_clean_eligible = False
    session._steering_uses_compiled_offsets = False
    session._live_lens = None
    session._lens_probes = {}
    session._live_sae = None
    session._sae_probes = {}
    session._sae_layer = None
    session._steering = SimpleNamespace(all_fast_path=lambda: True)

    _complete_capture_session(session)
    SaklasSession._begin_capture(
        session,
        need_per_token=True,
        gating_only_probes={"gate"},
        gating_probe_keys={"gate"},
        final_probe_aggregate=False,
    )

    assert monitor.plan_calls == [({"gate"}, {"gate"})]
    assert capture.sink is not None
    capture.sink({2: torch.ones(4)})
    assert monitor.score_plans == [(plan,)]
    assert session._incremental_gate_scores == [{"gate": 0.25}]


def test_full_incremental_capture_deep_tail_only_for_readout_aggregate_layers() -> None:
    class Capture:
        def __init__(self) -> None:
            self.attached: list[int] | None = None
            self.tail_depth: int | None = None
            self.tail_layers: set[int] | None = None

        def clear(self) -> None:
            pass

        def attach(self, _layers: Any, layer_idxs: list[int]) -> None:
            self.attached = list(layer_idxs)

        def set_tail_with_sink(
            self,
            depth: int,
            _sink: Any,
            *,
            tail_layers: set[int] | None = None,
        ) -> None:
            self.tail_depth = depth
            self.tail_layers = set(tail_layers or set())

    class Monitor:
        probe_names = ["monitor"]

        def probe_layers(self, names: set[str] | None = None) -> set[int]:
            assert names is None
            return {0, 1}

        def reset_curved_feet(self) -> None:
            pass

        def enable_curved_warm(self, _enabled: bool) -> None:
            pass

    capture = Capture()
    session: Any = _CurrentSessionStub.__new__(_CurrentSessionStub)
    session._layers = [object()] * 8
    session._monitor = Monitor()
    session._capture = capture
    session._capture_buffers = {}
    session._compiled_clean_eligible = False
    session._steering_uses_compiled_offsets = False
    session._live_lens = None
    session._lens_probes = {"jlens/g": {"layers": {3}}}
    session._lens_probe_layers = lambda _names=None: {3}
    session._live_sae = None
    session._sae_probes = {"sae/5": {"feature_id": 5}}
    session._sae_layer = 5
    session._steering = SimpleNamespace(all_fast_path=lambda: True)

    _complete_capture_session(session)
    attached = SaklasSession._begin_capture(
        session,
        need_per_token=True,
        final_probe_aggregate=True,
    )

    assert attached is True
    assert capture.attached == [0, 1, 3, 5]
    assert capture.tail_depth is not None and capture.tail_depth > 1
    assert capture.tail_layers == {3, 5}
    assert session._capture_state.mode is CaptureMode.INCREMENTAL


def test_penalty_state_applies_sparse_counts_on_device():
    logits = torch.zeros(1, 8)
    state = _PenaltyState(max_tokens=4, device=logits.device, dtype=torch.float32)
    state.add(2)
    state.add(5)
    state.add(2)

    state.apply(logits, presence_penalty=0.5, frequency_penalty=0.25)

    assert logits[0, 2].item() == -1.0
    assert logits[0, 5].item() == -0.75
    assert logits[0, 1].item() == 0.0


def test_no_cache_fallback_does_not_cat_each_step(monkeypatch: pytest.MonkeyPatch) -> None:
    model: Any = _NoCacheModel()
    tokenizer: Any = _StopTokenizer()
    state = GenerationState()

    def _forbid_cat(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("no-cache fallback should use the preallocated buffer")

    monkeypatch.setattr(torch, "cat", _forbid_cat)
    with pytest.warns(UserWarning, match="no past_key_values"):
        generated_ids = generate_steered(
            model,
            tokenizer,
            torch.tensor([[0, 0]]),
            GenerationConfig(max_new_tokens=3, temperature=0.0),
            state,
        )

    assert generated_ids == [0, 1]
    assert state.finish_reason == "stop"
