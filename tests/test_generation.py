"""CPU-only generation-loop regressions."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from saklas.core.generation import GenerationConfig, GenerationState, generate_steered
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

    def batch_decode(self, ids):
        return [self._pieces[row[0]] for row in ids]

    def decode(self, ids, skip_special_tokens=False):
        pieces = []
        for tid in ids:
            if skip_special_tokens and tid in self.all_special_ids:
                continue
            pieces.append(self._pieces[int(tid)])
        return "".join(pieces)


class _StopModel:
    config = SimpleNamespace(vocab_size=4)
    generation_config = SimpleNamespace(eos_token_id=3)

    def __init__(self):
        self._tokens = [0, 1, 2]
        self._idx = 0

    def __call__(self, **_kwargs):
        tid = self._tokens[min(self._idx, len(self._tokens) - 1)]
        self._idx += 1
        logits = torch.full((1, 1, self.config.vocab_size), -100.0)
        logits[0, 0, tid] = 100.0
        return SimpleNamespace(logits=logits, past_key_values=object())


def test_stop_sequence_trimmed_text_is_final_result_text():
    model = _StopModel()
    tokenizer = _StopTokenizer()
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

    session = SaklasSession.__new__(SaklasSession)
    session._gen_state = state
    session._tokenizer = tokenizer
    session._monitor = SimpleNamespace(probe_names=[])
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
