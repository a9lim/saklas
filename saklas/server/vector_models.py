"""Vector route schemas and serializers for native saklas routes."""

from __future__ import annotations

from operator import itemgetter
from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel, Field

from saklas.core.profile import Profile
from saklas.core.session import SaklasSession


class ExtractRequest(BaseModel):
    name: str
    source: Any = None
    baseline: str | None = None
    sae: str | None = None
    sae_revision: str | None = None
    role: str | None = None
    namespace: str | None = None
    force: bool = False
    auto_register: bool = Field(True, alias="register")

    model_config = {"populate_by_name": True}


class LoadVectorRequest(BaseModel):
    name: str
    source_path: str


class BakeVectorRequest(BaseModel):
    """Body for ``POST /saklas/v1/sessions/{id}/vectors/bake``."""

    name: str
    expression: str


def profile_to_json(name: str, profile: Profile) -> dict[str, Any]:
    layer_norms = [(idx, float(vec.norm().item())) for idx, vec in profile.items()]
    top = sorted(layer_norms, key=itemgetter(1), reverse=True)[:5]
    per_layer_norms = {str(idx): round(mag, 6) for idx, mag in sorted(layer_norms)}
    return {
        "name": name,
        "layers": profile.layers,
        "top_layers": [{"layer": idx, "magnitude": round(m, 4)} for idx, m in top],
        "per_layer_norms": per_layer_norms,
        "metadata": profile.metadata,
    }


def extract_registry_name(canonical: str, namespace: str | None) -> str:
    """Return the steerable key for a freshly extracted vector."""
    if namespace is None:
        return canonical
    if ":" in canonical:
        bare, suffix = canonical.rsplit(":", 1)
        return f"{namespace}/{bare}:{suffix}"
    return f"{namespace}/{canonical}"


def probe_profile_tensors(
    session: SaklasSession, name: str,
) -> dict[int, Any] | None:
    """Folded per-layer direction view of a vector probe, or ``None``."""
    manifold = session.monitor.manifolds.get(name)
    if manifold is None:
        return None
    from saklas.core.vectors import (
        folded_vector_directions,
        is_foldable_vector_manifold,
    )

    if not is_foldable_vector_manifold(manifold):
        return None
    return folded_vector_directions(manifold)


def coerce_corpora(source: Any) -> Any:
    """Normalize a JSON extract source into a concept name or two pole corpora."""
    if not isinstance(source, dict):
        return source

    if (
        isinstance(source.get("positive"), list)
        and isinstance(source.get("negative"), list)
    ):
        return (
            [str(s) for s in source["positive"]],
            [str(s) for s in source["negative"]],
        )

    raw_pairs: list[Any]
    if "pairs" in source:
        raw_pairs = list(source["pairs"])
    elif "positive" in source and "negative" in source:
        raw_pairs = [source]
    else:
        return source

    positive: list[str] = []
    negative: list[str] = []
    for idx, pair in enumerate(raw_pairs):
        if isinstance(pair, dict):
            if "positive" not in pair or "negative" not in pair:
                raise HTTPException(
                    400,
                    f"pairs[{idx}] must contain 'positive' and 'negative'",
                )
            positive.append(str(pair["positive"]))
            negative.append(str(pair["negative"]))
        elif isinstance(pair, (list, tuple)) and len(pair) == 2:
            positive.append(str(pair[0]))
            negative.append(str(pair[1]))
        else:
            raise HTTPException(
                400,
                f"pairs[{idx}] must be a [positive, negative] pair",
            )
    return positive, negative


# Backcompat aliases for the old ``saklas_api.py`` import surface.
_profile_to_json = profile_to_json
_extract_registry_name = extract_registry_name
_probe_profile_tensors = probe_profile_tensors
_coerce_corpora = coerce_corpora
