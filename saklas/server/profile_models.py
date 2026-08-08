"""Profile route schemas and serializers for native saklas routes."""

from __future__ import annotations

from typing import Literal

from saklas.core.profile import Profile
from saklas.server.native_common import NativeRequest
from saklas.server.response_models import VectorInfo


class ExtractRequest(NativeRequest):
    """Author and fit the current manifold representation of a concept.

    ``kind`` / ``custom_system`` are the same elicitation-framing pair
    ``POST /manifolds/generate`` carries, so the 2-node and multi-node
    authoring paths reach the same system templates.
    """

    concept: str
    baseline: str | None = None
    kind: Literal["abstract", "concrete", "custom"] = "abstract"
    custom_system: str | None = None
    sae: str | None = None
    role: str | None = None
    namespace: str | None = None
    force: bool = False


class BakeProfileRequest(NativeRequest):
    """Body for ``POST /saklas/v1/sessions/{id}/profiles/bake``."""

    name: str
    expression: str


def profile_to_json(name: str, profile: Profile) -> VectorInfo:
    return {
        "name": name,
        "layers": profile.layers,
        "metadata": profile.metadata,
    }


def extract_registry_name(canonical: str, namespace: str | None) -> str:
    """Return the steerable key for a freshly extracted profile."""
    if namespace is None:
        return canonical
    if ":" in canonical:
        bare, suffix = canonical.rsplit(":", 1)
        return f"{namespace}/{bare}:{suffix}"
    return f"{namespace}/{canonical}"
