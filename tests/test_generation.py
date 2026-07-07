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
from saklas.core.session import CaptureMode, CaptureState, SaklasSession


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


def test_prepare_input_raw_feeds_flat_active_path():
    """raw=True walks the loom tree as flat text — no chat template, no
    role markers — and appends the call's own input."""
    from saklas.core.loom import LoomTree

    tree = LoomTree()
    u1 = tree.add_user_turn("once upon a ")
    a1 = tree.begin_assistant(u1)
    tree.finalize_assistant(a1, text="time")

    session: Any = SaklasSession.__new__(SaklasSession)
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

    session: Any = SaklasSession.__new__(SaklasSession)
    session._gen_state = state
    session._tokenizer = tokenizer
    session._monitor = SimpleNamespace(probe_names=[])
    session._capture = SimpleNamespace(stacked=lambda: {})
    session._capture_state = CaptureState(mode=CaptureMode.FULL)
    session._last_per_token_scores = None
    session._last_result = None
    session.build_readings = lambda: {}

    result = SaklasSession._finalize_generation(
        session,
        "prompt",
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
        probe_names = ["toy"]

    session: Any = SaklasSession.__new__(SaklasSession)
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

    result = SaklasSession._finalize_generation(
        session,
        "prompt",
        generated_ids,
        elapsed=1.0,
        vector_snapshot={},
        stateless=True,
    )

    assert result.text == "Hello"
    assert result.probe_readings == {"toy": visible}
    assert result.readings["toy"].mean == (0.25,)


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

    session: Any = SaklasSession.__new__(SaklasSession)
    session._gen_state = state
    session._tokenizer = _StopTokenizer()
    session._monitor = Monitor()
    session._capture = capture
    session._capture_state = CaptureState(mode=CaptureMode.FULL)
    session._last_per_token_scores = None
    session._last_result = None
    session.events = SimpleNamespace(emit=lambda _event: None)
    session.build_readings = lambda: {}

    result = SaklasSession._finalize_generation(
        session,
        "prompt",
        [0, 1],
        elapsed=1.0,
        vector_snapshot={},
        stateless=True,
    )

    assert capture.calls == 1
    assert result.probe_readings == {"toy": reading}
    assert result.readings["toy"].mean == (0.25,)
    assert session._last_per_token_scores == {"toy": [0.1, 0.25]}


def test_generate_stream_exposes_current_result() -> None:
    result = GenerationResult(
        text="ok", tokens=[7], token_count=1, tok_per_sec=1.0, elapsed=1.0,
    )

    def _fake_generate_core(*_args: Any, **kwargs: Any) -> GenerationResult:
        kwargs["on_token"]("ok", False, 7, None, None, None)
        return result

    session: Any = SaklasSession.__new__(SaklasSession)
    session._gen_state = GenerationState()
    session._monitor = SimpleNamespace(probe_names=[])
    session._last_token_probe_payload = {}
    session._generate_core = _fake_generate_core

    stream = session.generate_stream("prompt")
    events = list(stream)
    assert [e.text for e in events] == ["ok"]
    assert stream.result is result


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

    session: Any = SaklasSession.__new__(SaklasSession)
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

    result = SaklasSession._finalize_generation(
        session,
        "prompt",
        [0, 1],
        elapsed=1.0,
        vector_snapshot={},
        stateless=True,
    )

    assert result.probe_readings == {"toy": reading1}
    assert result.readings["toy"].mean == (0.25,)
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

    session: Any = SaklasSession.__new__(SaklasSession)
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

    result = SaklasSession._finalize_generation(
        session,
        "prompt",
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

    session: Any = SaklasSession.__new__(SaklasSession)
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

    result = SaklasSession._finalize_generation(
        session,
        "prompt",
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
