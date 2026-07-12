"""Vector route schemas and serializers for native saklas routes."""

from __future__ import annotations

from operator import itemgetter
from typing import Any

from pydantic import BaseModel, ConfigDict

from saklas.core.profile import Profile
from saklas.core.session import SaklasSession


class ExtractRequest(BaseModel):
    """Author and fit the current manifold representation of a concept."""

    model_config = ConfigDict(extra="forbid")

    concept: str
    baseline: str | None = None
    sae: str | None = None
    role: str | None = None
    namespace: str | None = None
    force: bool = False


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
