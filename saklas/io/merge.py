"""Offline vector merging: precompute a linear combination of existing
steering vectors into a distributable single-vector pack.

Merge expressions use the shared steering grammar from
:mod:`saklas.core.steering_expr` — the same ``+`` / ``-`` / ``~`` /
``|`` / coefficient / projection syntax every other saklas surface
speaks.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Optional

import torch

from saklas.core.errors import SaklasError
from saklas.io.packs import (
    ConceptFolder, PackMetadata, hash_file,
)
from saklas.io.paths import concept_dir, safe_model_id
from saklas.core.vectors import load_profile, save_profile

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


def _resolve_coord(ns: Optional[str], name: str) -> ConceptFolder:
    if ns is None:
        raise MergeError(
            f"merge component '{name}' must be namespace-qualified "
            f"(e.g. 'default/{name}')"
        )
    folder = concept_dir(ns, name)
    if not folder.exists():
        raise MergeError(f"component {ns}/{name} not installed")
    return ConceptFolder.load(folder)


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
            if sel.operator != "~":
                raise MergeError(
                    f"merge expressions support only '~' for projection "
                    f"(got '{sel.operator}'). Use '~' for project-away."
                )
            op = "~"
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
        "ns", "name", "variant", "coeff",
        "operator", "onto_ns", "onto_name", "onto_variant",
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


def shared_models(expression: str) -> list[str]:
    """Return models for which every merge term has a tensor, sorted."""
    terms = _parse_merge_expr(expression)
    per: list[set[str]] = []
    for term in terms:
        cf = _resolve_coord(term.ns, term.name)
        per.append(set(cf.tensor_models()))
        if term.operator is not None:
            assert term.onto_ns is not None and term.onto_name is not None
            cf_b = _resolve_coord(term.onto_ns, term.onto_name)
            per.append(set(cf_b.tensor_models()))
    if not per:
        raise MergeError("no components provided")
    shared = set.intersection(*per)
    if not shared:
        raise MergeError(
            f"no shared models across {[t.coord for t in terms]}"
        )
    return sorted(shared)


def merge_into_pack(
    name: str,
    expression: str,
    model: Optional[str],
    *,
    force: bool = False,
    strict: bool = False,
) -> Path:
    """Create a merged tensors-only pack at ~/.saklas/vectors/local/<name>/.

    ``expression`` is a merge expression using the shared grammar:
    ``0.5 default/happy - 0.3 default/sad~default/calm``.
    """
    terms = _parse_merge_expr(expression)

    dst = concept_dir("local", name)
    if dst.exists() and not force:
        raise MergeError(f"{dst} exists; pass force=True to overwrite")
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    if model is not None:
        target_models = [safe_model_id(model)]
        for term in terms:
            cf = _resolve_coord(term.ns, term.name)
            if safe_model_id(model) not in cf.tensor_models():
                raise MergeError(
                    f"component {term.coord} has no tensor for {model}"
                )
    else:
        target_models = shared_models(expression)

    component_info: dict[str, dict[str, Any]] = {}
    files_map: dict[str, str] = {}

    for sid in target_models:
        profiles_and_alphas: list[tuple[Profile, float]] = []
        for term in terms:
            cf = _resolve_coord(term.ns, term.name)
            profile, _meta = load_profile(str(cf.tensor_path(sid)))
            if term.operator is not None:
                assert term.onto_ns is not None and term.onto_name is not None
                cf_b2 = _resolve_coord(term.onto_ns, term.onto_name)
                b_profile, _ = load_profile(str(cf_b2.tensor_path(sid)))
                profile = project_away(profile, b_profile)
            profiles_and_alphas.append((profile, term.coeff))
            component_info.setdefault(term.coord, {
                "alpha": term.coeff,
                "project_away": term.onto_coord,
                "tensor_sha256": hash_file(cf.tensor_path(sid)),
            })

        merged = linear_sum(profiles_and_alphas, strict=strict)
        ts_path = dst / f"{sid}.safetensors"
        save_profile(merged, str(ts_path), {
            "method": "merge",
            "components": component_info,
        })
        files_map[f"{sid}.safetensors"] = hash_file(ts_path)
        files_map[f"{sid}.json"] = hash_file(ts_path.with_suffix(".json"))

    def _term_desc(term: _MergeTerm) -> str:
        base = term.name
        if term.operator is not None:
            return f"{base}~{term.onto_name} ({term.coeff})"
        return f"{base} ({term.coeff})"

    desc = " + ".join(_term_desc(t) for t in terms)
    meta = PackMetadata(
        name=name,
        description=f"Merged pack: {desc}",
        version="1.0.0",
        license="AGPL-3.0-or-later",
        tags=["merge"],
        recommended_alpha=1.0,
        source="local",
        files=files_map,
    )
    meta.write(dst)
    return dst
