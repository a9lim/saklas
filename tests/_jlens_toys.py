"""Shared tiny causal LM + tokenizer for the Jacobian-lens CPU tests.

The toy is HF-shaped just enough for the accessors and the fit: a frozen
``model.model.layers`` block stack with causal mixing, ``model.model.norm``,
``lm_head``/``get_output_embeddings``, and a ``config.model_type`` that
resolves through ``_LAYER_ACCESSORS``.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

TOY_D = 6
TOY_VOCAB = 13


class ToyBlock(nn.Module):
    """Residual block with a causal cumulative-mean mix standing in for attention."""

    def __init__(self, d: int, seed: int) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.w1 = nn.Parameter(torch.randn(d, d, generator=gen) / d**0.5)
        self.w2 = nn.Parameter(torch.randn(d, d, generator=gen) / d**0.5)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        steps = torch.arange(1, h.shape[1] + 1, device=h.device).view(1, -1, 1)
        mix = torch.cumsum(h, dim=1) / steps
        return h + torch.tanh(h @ self.w1 + mix @ self.w2)


class ToyDecoder(nn.Module):
    def __init__(self, n_layers: int) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(99)
        self.embed_tokens = nn.Embedding(TOY_VOCAB, TOY_D)
        with torch.no_grad():
            self.embed_tokens.weight.copy_(torch.randn(TOY_VOCAB, TOY_D, generator=gen))
        self.layers = nn.ModuleList(ToyBlock(TOY_D, seed=i) for i in range(n_layers))
        self.norm = nn.LayerNorm(TOY_D)


class ToyCausalLM(nn.Module):
    def __init__(self, n_layers: int = 3) -> None:
        super().__init__()
        self.model = ToyDecoder(n_layers)
        self.lm_head = nn.Linear(TOY_D, TOY_VOCAB, bias=False)
        with torch.no_grad():
            gen = torch.Generator().manual_seed(101)
            self.lm_head.weight.copy_(
                torch.randn(TOY_VOCAB, TOY_D, generator=gen)
            )
        self.config = SimpleNamespace(model_type="llama")

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def forward(self, input_ids: torch.Tensor, use_cache: bool = False):
        del use_cache
        h = self.model.embed_tokens(input_ids)
        for block in self.model.layers:
            h = block(h)
        return SimpleNamespace(logits=self.lm_head(self.model.norm(h)))


class CharTokenizer:
    """Char-code tokenizer whose encode/decode round-trip for 'a'..'m'."""

    def __call__(self, text: str, return_tensors: str = "pt") -> dict[str, torch.Tensor]:
        del return_tensors
        return {"input_ids": torch.tensor([self.encode(text)], dtype=torch.long)}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [(ord(c) - 97) % TOY_VOCAB for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(97 + int(i) % 26) for i in ids)


def frozen_toy(n_layers: int = 3) -> ToyCausalLM:
    model = ToyCausalLM(n_layers)
    model.requires_grad_(False)  # replicate load_model's freeze
    model.eval()
    return model
