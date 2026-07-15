from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn


@pytest.fixture(autouse=True)
def _home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))


def _weights() -> dict[str, torch.Tensor]:
    return {
        "W_enc": torch.eye(3, 5),
        "W_dec": torch.eye(5, 3),
        "b_enc": torch.zeros(5),
        "b_dec": torch.ones(3),
    }


def test_local_sae_round_trip_and_backend() -> None:
    from saklas.core.sae import load_sae_backend
    from saklas.io.sae import list_sae_sources, load_active_sae_source
    from saklas.io.sae_artifacts import load_local_sae_manifest, save_local_sae

    manifest_path = save_local_sae(
        "org/model",
        "mine",
        _weights(),
        model_fingerprint="model-fp",
        model_source_fingerprint="source-fp",
        layer=2,
        corpus_spec="test",
        corpus_sha256="a" * 64,
        tokens_trained=100,
        seq_len=16,
        batch_size=2,
        learning_rate=1e-3,
        l1_coefficient=1e-3,
        dead_feature_threshold=1e-6,
    )
    assert manifest_path.parts[-4:] == ("sae", "local", "mine", "manifest.json")
    manifest = load_local_sae_manifest("org/model", "mine")
    assert manifest is not None and manifest["release"] == "local:mine"
    assert load_active_sae_source("org/model") == {
        "format_version": 1,
        "model_id": "org/model",
        "kind": "local",
        "name": "mine",
    }
    backend = load_sae_backend(
        "local:mine",
        model_id="org/model",
        device="cpu",
        dtype=torch.float32,
    )
    assert backend.layers == frozenset({2})
    hidden = torch.tensor([[2.0, 3.0, 4.0]])
    features = backend.encode_layer(2, hidden)
    assert features.shape == (1, 5)
    decoded = backend.decode_layer(2, features[0])
    assert decoded.shape == (3,)
    assert torch.equal(backend.feature_direction(2, 0), _weights()["W_dec"][0])
    assert list_sae_sources("org/model")[0]["source"] == "local:mine"


class _Tokenizer:
    pad_token_id = 0

    def __call__(self, text: str, **_kwargs: object) -> dict[str, list[int]]:
        return {"input_ids": [1 + (ord(char) % 11) for char in text]}


class _Block(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.linear = nn.Linear(width, width)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + torch.tanh(self.linear(hidden))


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(16, 4)
        self.layers = nn.ModuleList([_Block(4), _Block(4)])

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool = False,
    ) -> object:
        del attention_mask, use_cache
        hidden = self.embedding(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return type("Output", (), {"last_hidden_state": hidden})()


def test_native_sae_trainer_runs_without_saelens() -> None:
    from saklas.core.sae_training import train_residual_sae

    torch.manual_seed(0)
    model = _TinyModel().eval().requires_grad_(False)
    tensors, metrics = train_residual_sae(
        model,
        _Tokenizer(),
        list(model.layers),
        ["abcdefgh", "ijklmnop", "qrstuvwx"],
        layer=1,
        tokens=32,
        seq_len=8,
        batch_size=2,
        d_sae=7,
        learning_rate=1e-3,
        seed=7,
    )
    assert tensors["W_enc"].shape == (4, 7)
    assert tensors["W_dec"].shape == (7, 4)
    assert metrics["tokens_trained"] == 32
    assert metrics["d_sae"] == 7
    assert all(torch.isfinite(value).all() for value in tensors.values())


def test_native_sae_trainer_cancels_before_next_batch() -> None:
    import threading

    from saklas.core.sae_training import SaeTrainingCancelled, train_residual_sae

    cancel = threading.Event()
    cancel.set()
    model = _TinyModel().eval().requires_grad_(False)
    with pytest.raises(SaeTrainingCancelled):
        train_residual_sae(
            model,
            _Tokenizer(),
            list(model.layers),
            ["abcdefgh"],
            layer=1,
            tokens=8,
            seq_len=8,
            batch_size=1,
            d_sae=7,
            cancel_event=cancel,
        )
