"""Bootstrap the per-model probe-centering baseline and the bundled probe roster.

A steering vector lives as a 2-node ``pca`` manifold (4.0), so the bundled
probe set is sourced by folding fitted manifolds (see
``SaklasSession._bootstrap_manifold_probes``); this module owns the two pieces
that feed it: the per-model layer-mean cache (:func:`bootstrap_layer_means`) and
the tag→manifold roster (:func:`load_default_manifolds`).
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from saklas.io.packs import Sidecar, hash_file
from saklas.io.paths import (
    model_dir,
    neutral_statements_path,
)
from saklas.core.vectors import (
    compute_layer_means,
    load_profile, save_profile,
)

log = logging.getLogger(__name__)

_LAYER_MEANS_NAME = "layer_means"


def load_default_manifolds() -> dict[str, list[str]]:
    """Return {tag: [manifold_name, ...]} for bundled default/ manifolds.

    A steering vector lives as a 2-node ``pca`` manifold, tagged
    (``manifold.json::tags``) for category-grouped probe bootstrap.  Triggers
    first-run materialization of bundled manifolds.
    """
    from saklas.io.manifolds import (
        iter_manifold_folders, materialize_bundled_manifolds,
    )

    materialize_bundled_manifolds()
    by_tag: dict[str, list[str]] = {}
    for _ns, mf in iter_manifold_folders(namespace="default"):
        for tag in mf.tags or []:
            by_tag.setdefault(tag, []).append(mf.name)
    return by_tag


def bootstrap_layer_means(
    model: Any, tokenizer: Any, layers: torch.nn.ModuleList, model_info: dict[str, Any],
) -> dict[int, torch.Tensor]:
    """Load or compute per-layer mean activations for probe centering.

    Stored at ~/.saklas/models/<safe_id>/layer_means.safetensors with a slim
    sidecar. Stale if neutral_statements.json has changed since extraction.
    """
    model_id = model_info.get("model_id", "unknown")
    md = model_dir(model_id)
    md.mkdir(parents=True, exist_ok=True)
    ts_path = md / f"{_LAYER_MEANS_NAME}.safetensors"
    sc_path = md / f"{_LAYER_MEANS_NAME}.json"

    current_ns_hash: str | None = None
    if neutral_statements_path().exists():
        current_ns_hash = hash_file(neutral_statements_path())

    if ts_path.exists() and sc_path.exists():
        try:
            sc = Sidecar.load(sc_path)
            if current_ns_hash is None or sc.statements_sha256 == current_ns_hash:
                profile, _ = load_profile(str(ts_path))
                log.debug("Loaded cached layer means")
                return profile
            log.info("Layer means stale (neutral_statements changed); recomputing")
        except Exception as e:
            log.warning("Corrupt layer means cache, recomputing: %s", e)

    log.info("Computing layer means (one-time per model)...")
    means = compute_layer_means(model, tokenizer, layers)
    save_profile(means, str(ts_path), {
        "method": "layer_means",
        "statements_sha256": current_ns_hash or "",
    })
    return means
