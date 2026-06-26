"""Behavior-space naturalness eval for manifold steering.

This module holds the *one* part of the manifold evaluation pipeline that
drives a live HuggingFace model: computing per-node output-distribution
centroids and per-step trajectory distributions both require a raw
``model(input_ids=...)`` forward, which belongs here rather than in
:mod:`saklas.core.manifold` (pure tensor math, no session/IO coupling).

The public surface mirrors Goodfire's "Manifold Steering" paper (arXiv
2605.05115, second half): fit a manifold over the model's *output*
distributions in Hellinger space, then measure how far a steered
generation's behavioral trajectory strays off it.

:func:`fit_behavior_manifold` and :func:`trajectory_naturalness` are
pure-tensor helpers that depend only on :mod:`saklas.core.manifold`
primitives.  :func:`compute_node_behavior_centroid` and
:func:`compute_trajectory_distributions` are the two functions that call
``model(...)`` directly and therefore live in this file.
"""
from __future__ import annotations

import torch

from saklas.core.errors import SaklasError
from saklas.core.manifold import (
    DEFAULT_N_COMPONENTS,
    LayerSubspace,
    ManifoldDomain,
    eval_rbf,
    fit_layer_subspace,
    invert_parameterization,
)


def to_hellinger(p: torch.Tensor) -> torch.Tensor:
    """Map a probability distribution to Hellinger space: ``p -> sqrt(p)``.

    In Hellinger space the L2 distance between two mapped distributions
    is their Hellinger distance, and the inner product is the
    Bhattacharyya coefficient -- which linearizes the simplex enough for
    a Euclidean PCA + RBF fit.
    """
    return p.clamp(min=0.0).sqrt()


def bhattacharyya_distance(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Bhattacharyya distance ``-ln sum sqrt(p*q)`` between distributions.

    ``p`` and ``q`` are probability distributions over the last axis.
    Returns the distance over any leading batch dims.
    """
    bc = (p.clamp(min=0.0) * q.clamp(min=0.0)).sqrt().sum(dim=-1)
    return -torch.log(bc.clamp(min=1e-12))


def fit_behavior_manifold(
    centroid_dists: torch.Tensor,
    node_params: torch.Tensor,
    *,
    n_components: int = DEFAULT_N_COMPONENTS,
) -> LayerSubspace:
    """Fit an RBF interpolant through per-node output-distribution centroids.

    ``centroid_dists`` is ``(K, V)`` -- one mean next-token distribution
    per manifold node -- and ``node_params`` is ``(K, m)`` the embedded
    domain coordinates.  The distributions are mapped to Hellinger space
    (``sqrt``) and the same :func:`~saklas.core.manifold.fit_layer_subspace`
    PCA + RBF fit is applied there.  Returns the fitted
    :class:`~saklas.core.manifold.LayerSubspace` (the behavior manifold).
    """
    sub, _ev = fit_layer_subspace(
        to_hellinger(centroid_dists), node_params, n_components=n_components,
    )
    return sub


def trajectory_naturalness(
    traj_dists: torch.Tensor,
    behavior: LayerSubspace,
    domain: ManifoldDomain,
    node_coords: torch.Tensor,
) -> torch.Tensor:
    """Per-step Bhattacharyya distance from a trajectory to a behavior manifold.

    ``traj_dists`` is ``(T, V)`` -- the sequence of next-token
    distributions a generation produced.  ``behavior`` is a behavior
    manifold from :func:`fit_behavior_manifold`, ``domain`` its
    :class:`~saklas.core.manifold.ManifoldDomain`, and ``node_coords``
    ``(K, n)`` the manifold's authoring node coordinates (warm-start seeds
    for the inverse map).  For each step the nearest point on the behavior
    manifold is found (in Hellinger space) and the Bhattacharyya distance
    to it is returned -- low means the step sits on the natural behavior
    manifold, high flags an off-manifold "teleportation" artifact.

    Returns a ``(T,)`` tensor of per-step distances.
    """
    h = to_hellinger(traj_dists)  # (T, V)
    coords = (h - behavior.mean) @ behavior.basis.T  # (T, R)
    pos, _ = invert_parameterization(behavior, domain, coords, node_coords)
    embedded = domain.embed(pos)  # (T, m)
    curve_coords = eval_rbf(
        *behavior.rbf_params(), behavior._normalize(embedded),
    )  # (T, R)
    curve_h = curve_coords @ behavior.basis + behavior.mean  # (T, V)
    # In Hellinger space ||h_a - h_b||^2 = 2 - 2*BC, so the Bhattacharyya
    # coefficient is BC = 1 - d^2/2 and the distance is -ln(BC).
    d2 = ((h - curve_h) ** 2).sum(dim=-1)
    bc = (1.0 - d2 / 2.0).clamp(min=1e-12)
    return -torch.log(bc)


def compute_node_behavior_centroid(
    model: object,
    tokenizer: object,
    device: torch.device,
    statements: list[str],
) -> torch.Tensor:
    """Mean next-token probability distribution over a node's statements.

    The behavior-space analogue of
    :func:`~saklas.core.manifold.compute_node_centroid`: each statement is
    run through the model and the softmax over the final-position logits is
    taken; the per-statement distributions are averaged.  Returns a ``(V,)``
    distribution in fp32 on CPU.
    """
    if not statements:
        raise ValueError("manifold node has no statements")
    is_mps = getattr(device, "type", None) == "mps"
    acc: torch.Tensor | None = None
    for text in statements:
        dist = _next_token_distribution(model, tokenizer, text, device)
        acc = dist if acc is None else acc + dist
        if is_mps:
            torch.mps.empty_cache()
    if acc is None:  # T1.6: guard a real invariant without relying on -O stripping
        raise SaklasError(
            "compute_node_behavior_centroid: accumulator is None after iterating"
            " non-empty statements list — this should not happen"
        )
    return acc / len(statements)


def compute_trajectory_distributions(
    model: object,
    tokenizer: object,
    device: torch.device,
    text: str,
) -> torch.Tensor:
    """Per-position next-token distributions for a generated ``text``.

    One forward pass over the full token sequence; returns ``(T, V)`` --
    the behavioral trajectory the eval scores against the behavior
    manifold.  Kept off the generation hot path: the steered text is
    produced first, then re-run once here.
    """
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)  # pyright: ignore[reportCallIssue]  # tokenizer typed as object
    ids = enc["input_ids"].to(device)
    if ids.shape[1] == 0:
        raise ValueError("trajectory text tokenized to zero tokens")
    with torch.inference_mode():
        logits = model(input_ids=ids, use_cache=False).logits  # pyright: ignore[reportCallIssue]  # model typed as object
    return torch.softmax(logits[0].float(), dim=-1).detach().to("cpu")


def _next_token_distribution(
    model: object, tokenizer: object, text: str, device: torch.device,
) -> torch.Tensor:
    """Softmax over the final-position logits for ``text`` -- ``(V,)`` fp32 CPU."""
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)  # pyright: ignore[reportCallIssue]  # tokenizer typed as object
    ids = enc["input_ids"]
    if ids.numel() == 0:
        bos = getattr(tokenizer, "bos_token_id", None) or 0
        ids = torch.tensor([[bos]])
    ids = ids.to(device)
    with torch.inference_mode():
        logits = model(input_ids=ids, use_cache=False).logits  # pyright: ignore[reportCallIssue]  # model typed as object
    return torch.softmax(logits[0, -1].float(), dim=-1).detach().to("cpu")


__all__ = [
    "to_hellinger",
    "bhattacharyya_distance",
    "fit_behavior_manifold",
    "trajectory_naturalness",
    "compute_node_behavior_centroid",
    "compute_trajectory_distributions",
]
