"""SAE backend protocol and adapters.

The protocol is tiny on purpose: contrastive extraction needs only per-layer
encode/decode and the set of covered layers. The concrete ``SaeLensBackend``
adapter (added later) lives in the same module but imports ``sae_lens`` lazily,
inside its factory function — so installations without the ``[sae]`` extra
can still import ``saklas.core.sae`` (e.g. for the protocol type hint or the
mock).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

import torch


@runtime_checkable
class SaeBackend(Protocol):
    """Minimal surface for SAE-backed contrastive extraction."""

    release: str
    revision: str | None
    layers: frozenset[int]      # saklas 0-indexed transformer-block layers

    def encode_layer(self, idx: int, h: torch.Tensor) -> torch.Tensor:
        """Encode a batch of hidden states into sparse-feature space.

        Input shape: ``(N, d_model)``. Output shape: ``(N, d_feature)``.
        Caller guarantees ``idx in self.layers``.
        """
        ...

    def decode_layer(self, idx: int, f: torch.Tensor) -> torch.Tensor:
        """Decode a single feature-space direction back into model space.

        Input shape: ``(d_feature,)``. Output shape: ``(d_model,)``.
        Caller guarantees ``idx in self.layers``.
        """
        ...


# --- test helper ----------------------------------------------------------

@dataclass
class MockSaeBackend:
    """In-memory SAE backend for CPU-only tests.

    Default is identity encode/decode with ``d_feature == d_model``. Pass
    ``encode_fn`` / ``decode_fn`` for non-trivial layer-level transforms.
    """
    layers: frozenset[int]
    d_model: int
    d_feature: int | None = None
    release: str = "mock-release"
    revision: str | None = None
    encode_fn: Callable[[int, torch.Tensor], torch.Tensor] | None = None
    decode_fn: Callable[[int, torch.Tensor], torch.Tensor] | None = None

    def __post_init__(self):
        if self.d_feature is None:
            self.d_feature = self.d_model

    def encode_layer(self, idx: int, h: torch.Tensor) -> torch.Tensor:
        if self.encode_fn is not None:
            return self.encode_fn(idx, h)
        return h

    def decode_layer(self, idx: int, f: torch.Tensor) -> torch.Tensor:
        if self.decode_fn is not None:
            return self.decode_fn(idx, f)
        return f
