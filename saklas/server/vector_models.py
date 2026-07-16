"""Vector route schemas and serializers for native saklas routes."""

from __future__ import annotations

from typing import Any

from saklas.core.profile import Profile
from saklas.server.native_common import NativeRequest


class ExtractRequest(NativeRequest):
    """Author and fit the current manifold representation of a concept."""

    concept: str
    baseline: str | None = None
    sae: str | None = None
    role: str | None = None
    namespace: str | None = None
    force: bool = False


class BakeVectorRequest(NativeRequest):
    """Body for ``POST /saklas/v1/sessions/{id}/vectors/bake``."""

    name: str
    expression: str


def profile_to_json(name: str, profile: Profile) -> dict[str, Any]:
    return {
        "name": name,
        "layers": profile.layers,
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
