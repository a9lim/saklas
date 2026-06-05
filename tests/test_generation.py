"""CPU-only generation-loop regressions."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from saklas.core.generation import (
    GenerationConfig,
    GenerationState,
    _PenaltyState,
    generate_steered,
)
from saklas.core.results import ProbeReading
from saklas.core.session import SaklasSession


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


def test_stop_sequence_trimmed_text_is_final_result_text():
    model: Any = _StopModel()
    tokenizer: Any = _StopTokenizer()
    state = GenerationState()
    emitted: list[str] = []

    generated_ids = generate_steered(
        model,
        tokenizer,
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

    session: Any = SaklasSession.__new__(SaklasSession)
    session._gen_state = state
    session._tokenizer = tokenizer
    session._monitor = SimpleNamespace(probe_names=[])
    session._capture = SimpleNamespace(stacked=lambda: {})
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
        ) -> tuple[dict[str, ProbeReading], dict[str, list[float]]]:
            assert list(captured) == [0]
            assert generated_ids == [0, 1]
            assert accumulate is False
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
    session._capture_incremental = False
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
    session._capture_incremental = True
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
