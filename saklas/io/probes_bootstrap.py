"""Bootstrap the per-model probe-centering baseline and the bundled probe roster.

A steering vector lives as a 2-node ``pca`` manifold (4.0), so the bundled
probe set is sourced by folding fitted manifolds (see
``SaklasSession._bootstrap_manifold_probes``); this module owns the two pieces
that feed it: the per-model probe-centering means (:func:`bootstrap_layer_means`,
derived from the neutral-activation cache) and the tag→manifold roster
(:func:`load_default_manifolds`).
"""

from __future__ import annotations

from typing import Any

import torch


def load_default_manifolds() -> dict[str, list[str]]:
    """Return {tag: [manifold_name, ...]} for bundled default/ manifolds.

    A steering vector lives as a 2-node ``pca`` manifold, tagged
    (``manifold.json::tags``) for category-grouped probe bootstrap.  Triggers
    first-run materialization of bundled manifolds.
    """
    from saklas.io.manifolds import (
        iter_manifold_folders, materialize_bundled_manifolds,
    )
    from saklas.io.templates import materialize_bundled_templates

    # Templates first — a bundled manifold may ``template_ref`` a bundled one.
    materialize_bundled_templates()
    materialize_bundled_manifolds()
    by_tag: dict[str, list[str]] = {}
    for _ns, mf in iter_manifold_folders(namespace="default"):
        for tag in mf.tags or []:
            by_tag.setdefault(tag, []).append(mf.name)
    return by_tag


def bootstrap_layer_means(
    model: Any, tokenizer: Any, layers: torch.nn.ModuleList, model_info: dict[str, Any],
) -> dict[int, torch.Tensor]:
    """Per-layer neutral mean activation for probe centering.

    Derived as ``X.mean(0)`` from the per-model neutral-activation cache
    (:func:`saklas.io.alignment.load_or_compute_neutral_activations`).  The
    neutral mean *is* the probe-centering baseline — same corpus, same
    last-content-token fp32 pooling the whitener's covariance is built from —
    so there is no separate ``layer_means`` forward pass or disk cache.  The
    neutral-activation cache is the single per-model artifact both the
    centering mean and the Mahalanobis whitener read; the means fall out of it
    for free, so a cold model pays one neutral-corpus forward loop instead of
    two.  Stale on ``neutral_statements.json`` drift via the neutral cache's
    own sha256 key.
    """
    from saklas.io.alignment import load_or_compute_neutral_activations

    model_id = model_info.get("model_id", "unknown")
    acts = load_or_compute_neutral_activations(
        model, tokenizer, layers, model_id=model_id,
    )
    return {idx: X.mean(dim=0) for idx, X in acts.items()}
