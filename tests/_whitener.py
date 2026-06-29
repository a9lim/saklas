"""Synthetic ``LayerWhitener`` + neutral means for CPU-only tests.

The Euclidean fit/score path was removed in 4.0: activation-space manifold
fits, ``~``/``|`` projections, and probe reads all *require* a whitener that
covers every scored layer (a missing/partial whitener is a hard error, not a
Euclidean fallback).  A whitener is cheap and synthesizable on CPU with no
model load — :meth:`LayerWhitener.from_neutral_activations` only needs two
in-memory ``{layer: tensor}`` dicts — so CPU tests build one over random
neutral activations rather than reaching for the (now-deleted) Euclidean path.

The returned ``means`` is the SAME dict the whitener centered on, so a stub
``ModelHandle`` can expose it as both ``whitener`` and ``layer_means`` and the
fit's neutral anchor agrees with the whitener's centering.
"""
from __future__ import annotations

from collections.abc import Iterable

import torch

from saklas.core.mahalanobis import LayerWhitener


def synthetic_means(
    layers: Iterable[int], dim: int, *, seed: int = 11,
) -> dict[int, torch.Tensor]:
    """Per-layer neutral means ``{layer: [dim]}`` — small, deterministic."""
    out: dict[int, torch.Tensor] = {}
    for L in layers:
        g = torch.Generator().manual_seed(seed + L)
        out[L] = torch.randn(dim, generator=g, dtype=torch.float32) * 0.1
    return out


def synthetic_whitener(
    layers: Iterable[int],
    dim: int,
    *,
    n: int = 64,
    means: dict[int, torch.Tensor] | None = None,
    seed: int = 11,
    ridge_scale: float = 1.0,
) -> LayerWhitener:
    """A ``LayerWhitener`` over synthetic anisotropic neutral activations.

    Covers exactly ``layers`` at hidden dim ``dim``.  Pass ``means`` to share
    the neutral anchor with the consuming handle's ``layer_means``.
    """
    layers = list(layers)
    means = means if means is not None else synthetic_means(layers, dim, seed=seed)
    acts: dict[int, torch.Tensor] = {}
    for L in layers:
        g = torch.Generator().manual_seed(seed * 7 + L)
        # Anisotropic covariance so the whitened metric differs from Euclidean
        # (a degenerate isotropic draw would make the two coincide).
        scale = 0.5 + torch.rand(dim, generator=g, dtype=torch.float32)
        X = torch.randn(n, dim, generator=g, dtype=torch.float32) * scale
        acts[L] = X + means[L].reshape(1, dim)
    return LayerWhitener.from_neutral_activations(
        acts, means, ridge_scale=ridge_scale,
    )


def rogue_whitener(
    layers: Iterable[int],
    dim: int,
    *,
    n: int = 256,
    seed: int = 11,
    rogue_count: int = 4,
    rogue_mag: float = 100.0,
) -> tuple[LayerWhitener, list[int]]:
    """A whitener with a few massive-activation (rogue) channels + the clean dims.

    Models the real activation-space condition the Fisher / whitened metric
    exists to suppress: a handful of channels at ``rogue_mag``x the background
    variance.  A robust topology read must be invariant to these — the metric
    divides them out — so the signal must be lifted into the returned
    ``clean_dims`` (rogue dims are background-only; placing signal there would be
    correctly *down-weighted*, a different test).  Returns ``(whitener,
    clean_dims)``.
    """
    layers = list(layers)
    means = {L: torch.zeros(dim, dtype=torch.float32) for L in layers}
    rogue = sorted({int(round(i * (dim - 1) / max(1, rogue_count - 1)))
                    for i in range(rogue_count)})
    clean = [d for d in range(dim) if d not in rogue]
    acts: dict[int, torch.Tensor] = {}
    for L in layers:
        g = torch.Generator().manual_seed(seed * 7 + L)
        scale = torch.ones(dim, dtype=torch.float32)
        for d in rogue:
            scale[d] = float(rogue_mag)
        acts[L] = torch.randn(n, dim, generator=g, dtype=torch.float32) * scale
    return LayerWhitener.from_neutral_activations(acts, means, ridge_scale=1.0), clean


def isotropic_whitener(
    layers: Iterable[int],
    dim: int,
    *,
    n: int = 512,
    seed: int = 11,
) -> LayerWhitener:
    """A ``LayerWhitener`` over (near-)isotropic zero-mean neutral activations.

    Σ ≈ σ²·I, so Σ⁻¹ ∝ I and the whitened readout reduces to the Euclidean
    one (scale cancels in cosine / fraction ratios).  Used where a Mahalanobis
    surface is now *mandatory* but the test asserts the geometric value that
    was Euclidean before the collapse — the isotropic metric reproduces it to
    a loose tolerance.  Zero means, so it doesn't recenter test vectors.
    """
    layers = list(layers)
    means = {L: torch.zeros(dim, dtype=torch.float32) for L in layers}
    acts: dict[int, torch.Tensor] = {}
    for L in layers:
        g = torch.Generator().manual_seed(seed * 7 + L)
        acts[L] = torch.randn(n, dim, generator=g, dtype=torch.float32)
    return LayerWhitener.from_neutral_activations(acts, means, ridge_scale=1.0)
