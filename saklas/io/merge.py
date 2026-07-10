"""Offline direction merging into a corpus-less ``fit_mode="baked"`` manifold.

Merge expressions use the additive scalar subset of the shared steering
grammar: namespace-qualified terms joined by ``+`` / ``-``.  Dynamic terms and
projections are deliberately rejected.  In particular, live ``~`` / ``|``
projection is Mahalanobis-only and needs an identity-matched model whitener;
silently substituting an offline Euclidean projection would change semantics.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import torch

from saklas.core.errors import SaklasError
from saklas.io.paths import safe_model_id

log = logging.getLogger(__name__)


Profile = dict[int, torch.Tensor]


class MergeError(ValueError, SaklasError):
    def user_message(self) -> tuple[int, str]:
        return (400, str(self) or self.__class__.__name__)


def linear_sum(
    components: list[tuple[Profile, float]],
    *,
    strict: bool = False,
) -> Profile:
    """Compute ``merged[l] = sum_i alpha_i * vec_i[l]`` per touched layer.

    The layer set is the union, matching live expression composition: an
    absent component contributes zero rather than deleting the entire layer.
    ``strict=True`` rejects non-identical component coverage.
    """
    if len(components) < 1:
        raise MergeError("linear_sum requires at least one component")
    layer_sets = [set(p.keys()) for p, _ in components]
    union = set.union(*layer_sets)
    if not union:
        raise MergeError("merge components have no layers")
    if strict and any(layers != layer_sets[0] for layers in layer_sets[1:]):
        coverage = [sorted(layers) for layers in layer_sets]
        raise MergeError(
            "merge: component layer coverage differs under --strict: "
            f"{coverage}"
        )

    out: Profile = {}
    for layer in sorted(union):
        first_vec = next(profile[layer] for profile, _ in components if layer in profile)
        merged = torch.zeros_like(first_vec, dtype=torch.float32)
        for profile, alpha in components:
            if layer in profile:
                merged = (
                    merged
                    + float(alpha) * profile[layer].to(dtype=torch.float32)
                )
        out[layer] = merged
    return out


def _manifold_tensor_path(ns: str, name: str, sid: str, variant: Optional[str]) -> "Path | None":
    """Fitted-manifold tensor path for ``(ns, name, sid, variant)``, or ``None``.

    Role variants validate the canonical tensor's recorded baseline. Transfer
    and SAE variants select their concrete tensor suffix.
    """
    from saklas.io.paths import manifold_dir, tensor_filename

    transferred_from: str | None = None
    if variant in (None, "raw"):
        release: Optional[str] = None
    elif variant.startswith("sae-"):
        release = variant[len("sae-"):]
    elif variant.startswith("from-"):
        release = None
        transferred_from = variant[len("from-"):]
    elif variant.startswith("role-"):
        release = None
    else:
        return None
    path = manifold_dir(ns, name) / tensor_filename(
        sid, release=release, transferred_from=transferred_from,
        model_id_is_safe=True,
        transferred_from_is_safe=transferred_from is not None,
    )
    return path if path.exists() and path.with_suffix(".json").exists() else None


def _component_has_tensor_for(ns: str, name: str, sid: str, variant: Optional[str]) -> bool:
    """True when ``(ns, name)`` has a usable fitted manifold tensor for ``sid``."""
    try:
        _resolve_component(
            ns, name, sid, variant,
            f"{ns}/{name}" if variant is None else f"{ns}/{name}:{variant}",
        )
    except (MergeError, OSError, ValueError):
        return False
    return True


def _resolve_component(
    ns: str, name: str, sid: str, variant: Optional[str], coord: str,
) -> "tuple[Profile, dict[str, Any]]":
    """Load a merge component as ``(profile, source_tensor_path)``.

    Folds a fitted 2-node ``pca`` manifold down to a single direction
    (:func:`~saklas.core.vectors.folded_vector_directions`).  The returned path
    is the manifold's fitted tensor file, so provenance hashing is uniform.
    """
    mpath = _manifold_tensor_path(ns, name, sid, variant)
    if mpath is None:
        raise MergeError(
            f"component {coord} not installed (no fitted manifold for {sid})"
        )
    from saklas.core.manifold import load_manifold
    from saklas.core.vectors import folded_vector_directions
    manifold = load_manifold(mpath)
    metadata = dict(manifold.metadata)
    raw_roles = metadata.get("node_roles")
    roles = (
        list(raw_roles)
        if isinstance(raw_roles, list)
        else [None] * len(manifold.node_labels)
    )
    if variant is not None and variant.startswith("role-"):
        requested_role = variant[len("role-"):]
        if not roles or not all(role == requested_role for role in roles):
            raise MergeError(
                f"component {coord} was not fitted uniformly with role "
                f"{requested_role!r}"
            )
    elif variant in (None, "raw") and any(role is not None for role in roles):
        raise MergeError(
            f"component {coord} is role-baselined; select its :role-* alias"
        )
    if variant is not None and variant.startswith("from-"):
        from saklas.io.paths import safe_model_id

        requested_source = variant[len("from-"):]
        source_id = metadata.get("source_model_id")
        if (
            not isinstance(source_id, str)
            or safe_model_id(source_id).lower() != requested_source.lower()
        ):
            raise MergeError(
                f"component {coord} transfer provenance does not match "
                f"{requested_source!r}"
            )
    try:
        folded = folded_vector_directions(manifold)
    except Exception as e:
        raise MergeError(
            f"component {coord} is a manifold that does not fold to a single "
            f"steering direction (not a 2-node affine subspace): {e}"
        ) from e
    return folded, metadata


def _parse_merge_expr(expression: str) -> "list[_MergeTerm]":
    """Parse an additive merge expression into qualified scalar terms.

    Raises :class:`MergeError` on any parser-level issue or when a term
    uses a feature merge doesn't support (triggers, bare poles without
    namespaces).
    """
    from saklas.core.steering_expr import (
        SteeringExprError, _Parser, _lex,
    )

    if not expression or not expression.strip():
        raise MergeError("merge requires at least one component")

    try:
        toks = _lex(expression)
        terms = _Parser(toks).parse()
    except SteeringExprError as e:
        raise MergeError(f"merge expression: {e}") from e

    out: list[_MergeTerm] = []
    for term in terms:
        if term.ablation:
            raise MergeError(
                "merge expressions cannot bake dynamic mean-ablation terms ('!')"
            )
        if len(term.coeffs) != 1:
            raise MergeError(
                "merge expressions require one scalar coefficient per term"
            )
        if term.trigger is not None:
            raise MergeError(
                "merge expressions do not accept triggers "
                f"(got @{term.trigger})"
            )
        sel = term.selector
        if sel.manifold_position is not None:
            raise MergeError(
                "merge expressions cannot bake manifold-position ('%') terms"
            )
        if sel.operator is not None:
            raise MergeError(
                "offline merge cannot bake projection operators; live '~' and "
                "'|' require an identity-matched Mahalanobis whitener"
            )
        base = sel.base
        if base.namespace is None:
            raise MergeError(
                f"merge component '{base.concept}' must be namespace-qualified "
                f"(e.g. 'default/{base.concept}')"
            )
        out.append(_MergeTerm(
            ns=base.namespace,
            name=base.concept,
            variant=base.variant,
            coeff=term.coeff,
        ))
    return out


class _MergeTerm:
    __slots__ = ("coeff", "name", "ns", "variant")
    def __init__(
        self,
        ns: str,
        name: str,
        variant: Optional[str],
        coeff: float,
    ):
        self.ns = ns
        self.name = name
        self.variant = variant
        self.coeff = coeff

    @property
    def coord(self) -> str:
        base = f"{self.ns}/{self.name}"
        return base if self.variant in (None, "raw") else f"{base}:{self.variant}"

def _component_tensor_models(
    ns: Optional[str], name: str, variant: Optional[str],
) -> set[str]:
    """Models carrying the exact foldable tensor variant for a component."""
    if ns is None:
        raise MergeError(
            f"merge component '{name}' must be namespace-qualified "
            f"(e.g. 'default/{name}')"
        )
    from saklas.io.paths import manifold_dir, parse_tensor_filename
    mdir = manifold_dir(ns, name)
    models: set[str] = set()
    if mdir.is_dir():
        for tensor in mdir.glob("*.safetensors"):
            parsed = parse_tensor_filename(tensor.name)
            if parsed is None:
                continue
            safe_model, parsed_variant = parsed
            requested = None if variant in (None, "raw") else variant
            candidate = _manifold_tensor_path(ns, name, safe_model, requested)
            if (
                candidate is not None
                and candidate == tensor
                and _component_has_tensor_for(
                    ns, name, safe_model, requested,
                )
            ):
                models.add(safe_model)
    return models


def shared_models(expression: str) -> list[str]:
    """Return models for which every merge term has a tensor, sorted."""
    terms = _parse_merge_expr(expression)
    per: list[set[str]] = []
    for term in terms:
        per.append(_component_tensor_models(term.ns, term.name, term.variant))
    if not per:
        raise MergeError("no components provided")
    shared = set.intersection(*per)
    if not shared:
        raise MergeError(
            f"no shared models across {[t.coord for t in terms]}"
        )
    return sorted(shared)


def _term_desc(term: _MergeTerm) -> str:
    return f"{term.name} ({term.coeff})"


def merge_into_manifold(
    name: str,
    expression: str,
    model: Optional[str],
    *,
    force: bool = False,
    strict: bool = False,
    expected_model_fingerprint: str | None = None,
) -> Path:
    """Bake a merge while holding its destination folder transaction lock."""
    from saklas.io.manifold_folder import _locked_manifest
    from saklas.io.paths import manifold_dir

    with _locked_manifest(manifold_dir("local", name)):
        return _merge_into_manifold_locked(
            name, expression, model, force=force, strict=strict,
            expected_model_fingerprint=expected_model_fingerprint,
        )


def _merge_into_manifold_locked(
    name: str,
    expression: str,
    model: Optional[str],
    *,
    force: bool = False,
    strict: bool = False,
    expected_model_fingerprint: str | None = None,
) -> Path:
    """Merge an expression of installed directions into a baked manifold.

    Writes a corpus-less ``fit_mode="baked"`` manifold to
    ``~/.saklas/manifolds/local/<name>/`` — one fitted per-model tensor for
    every model the expression resolves on, all sharing one ``manifold.json``.
    ``expression`` uses the additive scalar subset of the shared steering
    grammar, for example ``0.5 default/happy - 0.3 default/sad``.

    Each component resolves to a per-layer ``dict[int, Tensor]`` direction (a
    fitted 2-node ``pca`` manifold folded down, or a legacy vector pack), the
    directions are linearly combined, and the result is folded to a one-pole
    ray (:func:`~saklas.core.vectors.fold_directions_to_subspace`) and frozen
    into the baked tensor.  Returns the manifold folder path.
    """
    from saklas.core.vectors import fold_directions_to_subspace
    from saklas.io.manifolds import (
        ManifoldFolder,
        create_baked_manifold_folder,
        save_baked_manifold_tensor,
    )
    from saklas.io.paths import manifold_dir

    terms = _parse_merge_expr(expression)

    dst = manifold_dir("local", name)
    if (dst / "manifold.json").exists() and not force:
        raise MergeError(f"{dst} exists; pass force=True to overwrite")

    if model is not None:
        sid_check = safe_model_id(model)
        target_models = [sid_check]
        for term in terms:
            if not _component_has_tensor_for(
                term.ns, term.name, sid_check, term.variant,
            ):
                raise MergeError(
                    f"component {term.coord} has no tensor for {model}"
                )
    else:
        target_models = shared_models(expression)

    description = f"Merged manifold: {' + '.join(_term_desc(t) for t in terms)}"

    prepared_outputs: list[tuple[str, Any, dict[str, Any], str]] = []
    for sid in target_models:
        profiles_and_alphas: list[tuple[Profile, float]] = []
        component_info: dict[str, dict[str, Any]] = {}
        component_fingerprints: set[str] = set()
        fingerprint_complete = True
        for term_index, term in enumerate(terms):
            profile, component_metadata = _resolve_component(
                term.ns, term.name, sid, term.variant, term.coord,
            )
            base_fingerprint = component_metadata.get("model_fingerprint")
            if isinstance(base_fingerprint, str):
                component_fingerprints.add(base_fingerprint)
            else:
                fingerprint_complete = False
            profiles_and_alphas.append((profile, term.coeff))
            component_info[f"{term_index}:{term.coord}"] = {
                "selector": term.coord,
                "alpha": term.coeff,
                "tensor_sha256": component_metadata.get("_tensor_sha256"),
            }

        merged = linear_sum(profiles_and_alphas, strict=strict)
        # Fold the derived direction into a corpus-less one-pole ray
        # (neutral_means=None — an offline merge has no model/whitener loaded,
        # so the subspace anchors at coord 0 and the share is the Euclidean
        # ‖merged_L‖, the same magnitude the components already carry baked in).
        manifold = fold_directions_to_subspace(name, merged, None, label="merged")
        baked_fingerprint = (
            next(iter(component_fingerprints))
            if fingerprint_complete and len(component_fingerprints) == 1
            else None
        )
        if baked_fingerprint is None:
            raise MergeError(
                f"cannot bake {name!r} for {sid}: component model "
                "fingerprints are missing or disagree; refit every component "
                "for the same loaded weights first"
            )
        if (
            expected_model_fingerprint is not None
            and baked_fingerprint != expected_model_fingerprint
        ):
            raise MergeError(
                f"cannot bake {name!r} for {sid}: components were fitted for "
                "different loaded weights; refit them for this session first"
            )
        prepared_outputs.append((
            sid, manifold, component_info, baked_fingerprint,
        ))

    # All models are resolved, integrity-checked, folded, and identity-checked
    # before the first destination mutation. A late model failure therefore
    # cannot leave a partial new folder or delete a prior good force target.
    folder: Optional[Path] = None
    for sid, manifold, component_info, baked_fingerprint in prepared_outputs:
        if folder is None:
            folder, _mf = create_baked_manifold_folder(
                "local", name, description, manifold, sid,
                method="merge", tags=["merge"], force=force,
                components=component_info,
                model_fingerprint=baked_fingerprint,
                model_id_is_safe=True,
            )
        else:
            tensor_path = save_baked_manifold_tensor(
                folder, manifold, sid, method="merge", components=component_info,
                model_fingerprint=baked_fingerprint,
                model_id_is_safe=True,
            )
            ManifoldFolder.load(
                folder, verify_manifest=False,
            ).update_file_hashes(
                tensor_path, tensor_path.with_suffix(".json"),
            )

    # ``target_models`` is non-empty (shared_models raises on an empty
    # intersection; the explicit-model branch is always a singleton).
    assert folder is not None
    return folder
