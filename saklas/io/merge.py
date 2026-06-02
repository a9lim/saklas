"""Offline direction merging: precompute a linear combination of existing
steering directions into a corpus-less ``fit_mode="baked"`` manifold.

Merge expressions use the shared steering grammar from
:mod:`saklas.core.steering_expr` — the same ``+`` / ``-`` / ``~`` /
``|`` / coefficient / projection syntax every other saklas surface
speaks.  Each component resolves to a per-layer ``dict[int, Tensor]``
direction by folding a fitted 2-node ``pca`` manifold down to a single
direction, the directions are linearly combined, and the result is
folded to a one-pole ray and frozen into a baked manifold under
``~/.saklas/manifolds/local/<name>/``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import torch

from saklas.core.errors import SaklasError
from saklas.io.packs import hash_file
from saklas.io.paths import safe_model_id

log = logging.getLogger(__name__)


Profile = dict[int, torch.Tensor]


class MergeError(ValueError, SaklasError):
    def user_message(self) -> tuple[int, str]:
        return (400, str(self) or self.__class__.__name__)


def project_away(a: Profile, b: Profile) -> Profile:
    """Return a new profile with b's direction projected out of a, per layer.

    Per-layer math (fp32)::

        result_L = a_L - (dot(a_L, b_L) / dot(b_L, b_L)) * b_L

    Layers where ``dot(b_L, b_L) < 1e-12`` are copied unchanged (near-zero b
    direction — no meaningful projection axis).  Only layers present in both
    profiles are projected; layers in a but not b are included unchanged.
    """
    out: Profile = {}
    for layer, a_t in a.items():
        if layer not in b:
            out[layer] = a_t
            continue
        a_f = a_t.to(dtype=torch.float32)
        b_f = b[layer].to(dtype=torch.float32)
        b_dot = torch.dot(b_f, b_f).item()
        if b_dot < 1e-12:
            out[layer] = a_t
        else:
            proj = (torch.dot(a_f, b_f) / b_dot) * b_f
            out[layer] = (a_f - proj).to(dtype=a_t.dtype)
    return out


def linear_sum(
    components: list[tuple[Profile, float]],
    *,
    strict: bool = False,
) -> Profile:
    """Compute merged[l] = sum_i alpha_i * vec_i[l] per layer.

    Layer set is the intersection of every component's layers. If
    ``strict`` is True, any non-common layers raise MergeError instead
    of being silently dropped.
    """
    if len(components) < 1:
        raise MergeError("linear_sum requires at least one component")
    layer_sets = [set(p.keys()) for p, _ in components]
    common = set.intersection(*layer_sets)
    if not common:
        raise MergeError("no common layers across components")

    union = set.union(*layer_sets)
    dropped = sorted(union - common)
    if dropped:
        if strict:
            raise MergeError(
                f"merge: layer intersection {len(common)}/{len(union)}; "
                f"refusing to drop layers {dropped} under --strict"
            )
        log.warning(
            "merge: layer intersection %d/%d; dropping layers %s",
            len(common), len(union), dropped,
        )

    out: Profile = {}
    for layer in sorted(common):
        first_vec = components[0][0][layer]
        merged = torch.zeros_like(first_vec, dtype=torch.float32)
        for profile, alpha in components:
            merged = merged + float(alpha) * profile[layer].to(dtype=torch.float32)
        out[layer] = merged
    return out


def _manifold_tensor_path(ns: str, name: str, sid: str, variant: Optional[str]) -> "Path | None":
    """Fitted-manifold tensor path for ``(ns, name, sid, variant)``, or ``None``.

    The 4.0 fold fallback: bundled & user concepts ship as 2-node ``pca``
    manifolds.  Only the ``raw`` / ``sae-<release>`` variants fold (role /
    transfer variants are vector-only).
    """
    from saklas.io.paths import manifold_dir, tensor_filename
    if variant in (None, "raw"):
        release: Optional[str] = None
    elif variant.startswith("sae-"):
        release = variant[len("sae-"):]
    else:
        return None
    path = manifold_dir(ns, name) / tensor_filename(sid, release=release)
    return path if path.exists() else None


def _component_has_tensor_for(ns: str, name: str, sid: str, variant: Optional[str]) -> bool:
    """True when ``(ns, name)`` has a usable fitted manifold tensor for ``sid``."""
    return _manifold_tensor_path(ns, name, sid, variant) is not None


def _resolve_component(
    ns: str, name: str, sid: str, variant: Optional[str], coord: str,
) -> "tuple[Profile, Path]":
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
    try:
        folded = folded_vector_directions(manifold)
    except Exception as e:
        raise MergeError(
            f"component {coord} is a manifold that does not fold to a single "
            f"steering direction (not a 2-node affine subspace): {e}"
        )
    return folded, mpath


def _parse_merge_expr(expression: str) -> "list[_MergeTerm]":
    """Parse a merge expression into a list of (ns, name, variant,
    coeff, operator, onto) terms.

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
        if term.trigger is not None:
            raise MergeError(
                "merge expressions do not accept triggers "
                f"(got @{term.trigger})"
            )
        sel = term.selector
        base = sel.base
        if base.namespace is None:
            raise MergeError(
                f"merge component '{base.concept}' must be namespace-qualified "
                f"(e.g. 'default/{base.concept}')"
            )
        onto_ns = onto_name = onto_variant = None
        op = None
        if sel.operator is not None:
            # Project-away (orthogonal) is ``|`` per the unified
            # grammar in saklas.core.steering_expr; ``~`` is the
            # *aligned* component (Gram-Schmidt onto, not away).
            # Pre-v2.1 merge accepted ``~`` and treated it as
            # project-away — silently inverting the user's intent
            # vs. every other surface.  Now ``|`` is the canonical
            # form; ``~`` raises with a hint pointing users at the
            # right operator (and at vector_merge if they actually
            # wanted the aligned component, which merge can't
            # currently express because the kept-aligned-only profile
            # would always have norm < the original).
            if sel.operator != "|":
                raise MergeError(
                    f"merge expressions support only '|' for "
                    f"project-away (got '{sel.operator}').  Note: pre-"
                    f"v2.1 merge accepted '~' for the same operation; "
                    f"the new spelling matches the unified grammar."
                )
            op = "|"
            onto = sel.onto
            assert onto is not None
            if onto.namespace is None:
                raise MergeError(
                    f"merge projection target '{onto.concept}' must be "
                    f"namespace-qualified (e.g. 'default/{onto.concept}')"
                )
            onto_ns, onto_name, onto_variant = (
                onto.namespace, onto.concept, onto.variant,
            )
        out.append(_MergeTerm(
            ns=base.namespace,
            name=base.concept,
            variant=base.variant,
            coeff=term.coeff,
            operator=op,
            onto_ns=onto_ns,
            onto_name=onto_name,
            onto_variant=onto_variant,
        ))
    return out


class _MergeTerm:
    __slots__ = (
        "coeff",
        "name",
        "ns",
        "onto_name",
        "onto_ns",
        "onto_variant",
        "operator",
        "variant",
    )
    def __init__(
        self,
        ns: str,
        name: str,
        variant: Optional[str],
        coeff: float,
        operator: Optional[str],
        onto_ns: Optional[str],
        onto_name: Optional[str],
        onto_variant: Optional[str],
    ):
        self.ns = ns
        self.name = name
        self.variant = variant
        self.coeff = coeff
        self.operator = operator
        self.onto_ns = onto_ns
        self.onto_name = onto_name
        self.onto_variant = onto_variant

    @property
    def coord(self) -> str:
        return f"{self.ns}/{self.name}"

    @property
    def onto_coord(self) -> "str | None":
        if self.onto_name is None:
            return None
        return f"{self.onto_ns}/{self.onto_name}"


def _component_tensor_models(ns: Optional[str], name: str) -> set[str]:
    """Models with a fitted ``raw`` manifold tensor for a component (the fold source)."""
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
            safe_model, variant = parsed
            if variant is None:  # raw fitted tensor → foldable
                models.add(safe_model)
    return models


def shared_models(expression: str) -> list[str]:
    """Return models for which every merge term has a tensor, sorted."""
    terms = _parse_merge_expr(expression)
    per: list[set[str]] = []
    for term in terms:
        per.append(_component_tensor_models(term.ns, term.name))
        if term.operator is not None:
            assert term.onto_ns is not None and term.onto_name is not None
            per.append(_component_tensor_models(term.onto_ns, term.onto_name))
    if not per:
        raise MergeError("no components provided")
    shared = set.intersection(*per)
    if not shared:
        raise MergeError(
            f"no shared models across {[t.coord for t in terms]}"
        )
    return sorted(shared)


def _term_desc(term: _MergeTerm) -> str:
    base = term.name
    if term.operator is not None:
        return f"{base}|{term.onto_name} ({term.coeff})"
    return f"{base} ({term.coeff})"


def merge_into_manifold(
    name: str,
    expression: str,
    model: Optional[str],
    *,
    force: bool = False,
    strict: bool = False,
) -> Path:
    """Merge an expression of installed directions into a baked manifold.

    Writes a corpus-less ``fit_mode="baked"`` manifold to
    ``~/.saklas/manifolds/local/<name>/`` — one fitted per-model tensor for
    every model the expression resolves on, all sharing one ``manifold.json``.
    ``expression`` uses the shared steering grammar:
    ``0.5 default/happy - 0.3 default/sad|default/calm`` (``|`` projects away).

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

    folder: Optional[Path] = None
    for sid in target_models:
        profiles_and_alphas: list[tuple[Profile, float]] = []
        component_info: dict[str, dict[str, Any]] = {}
        for term in terms:
            profile, base_path = _resolve_component(
                term.ns, term.name, sid, term.variant, term.coord,
            )
            if term.operator is not None:
                assert term.onto_ns is not None and term.onto_name is not None
                b_profile, _onto_path = _resolve_component(
                    term.onto_ns, term.onto_name, sid,
                    term.onto_variant, term.onto_coord or "",
                )
                profile = project_away(profile, b_profile)
            profiles_and_alphas.append((profile, term.coeff))
            component_info.setdefault(term.coord, {
                "alpha": term.coeff,
                "project_away": term.onto_coord,
                "tensor_sha256": hash_file(base_path),
            })

        merged = linear_sum(profiles_and_alphas, strict=strict)
        # Fold the derived direction into a corpus-less one-pole ray
        # (neutral_means=None — an offline merge has no model/whitener loaded,
        # so the subspace anchors at coord 0 and the share is the Euclidean
        # ‖merged_L‖, the same magnitude the components already carry baked in).
        manifold = fold_directions_to_subspace(name, merged, None, label="merged")
        if folder is None:
            folder, _mf = create_baked_manifold_folder(
                "local", name, description, manifold, sid,
                method="merge", tags=["merge"], force=force,
                components=component_info,
            )
        else:
            save_baked_manifold_tensor(
                folder, manifold, sid, method="merge", components=component_info,
            )

    # ``target_models`` is non-empty (shared_models raises on an empty
    # intersection; the explicit-model branch is always a singleton).
    assert folder is not None
    ManifoldFolder.load(folder).write_metadata()
    return folder
