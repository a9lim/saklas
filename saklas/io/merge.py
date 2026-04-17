"""Offline vector merging: precompute a linear combination of existing
steering vectors into a distributable single-vector pack.

See docs/superpowers/specs/2026-04-12-story-a-portability-design.md §Component 6.
"""
from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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
    pass


@dataclass(frozen=True)
class ComponentSpec:
    """Parsed component: a coordinate, optional projection-removal target, and alpha."""
    coord: str
    project_away: str | None
    alpha: float


def parse_components(raw: str) -> list[ComponentSpec]:
    """Parse component grammar into a list of ComponentSpec.

    Grammar::

        component  = coord ":" alpha
                   | coord "~" coord ":" alpha
        components = component ("," component)*

    ``a~b:0.5`` means: project b's direction out of a, then scale by 0.5.
    Chained ``~`` (``a~b~c``) is rejected as a parse error.
    """
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    out: list[ComponentSpec] = []
    for part in parts:
        if ":" not in part:
            raise MergeError(f"component '{part}' missing :alpha")
        coord_part, alpha_s = part.rsplit(":", 1)
        coord_part = coord_part.strip()
        try:
            alpha = float(alpha_s)
        except ValueError as e:
            raise MergeError(f"component '{part}' alpha not a number: {e}") from e
        # Handle projection operator ~
        if "~" in coord_part:
            tilde_parts = coord_part.split("~")
            if len(tilde_parts) != 2:
                raise MergeError(
                    f"component '{part}': chained '~' is not allowed; "
                    f"use 'a~b:alpha' only"
                )
            coord, project_away = tilde_parts[0].strip(), tilde_parts[1].strip()
        else:
            coord = coord_part
            project_away = None
        out.append(ComponentSpec(coord=coord, project_away=project_away, alpha=alpha))
    if len(out) < 1:
        raise MergeError("merge requires at least one component")
    return out


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

    Since component vectors are already baked (share * ref_norm folded
    into the magnitude), a weighted sum preserves the layer-weighting
    semantics naturally — no re-scoring, no share redistribution. The
    merged tensor injects at apply time exactly as
    ``sum_i alpha_i * component_i`` would have.

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


def _resolve_coord(coord: str) -> ConceptFolder:
    if "/" not in coord:
        raise MergeError(f"component must be '<ns>/<concept>': {coord!r}")
    ns, name = coord.split("/", 1)
    folder = concept_dir(ns, name)
    if not folder.exists():
        raise MergeError(f"component {coord} not installed")
    return ConceptFolder.load(folder)


def shared_models(components: list[ComponentSpec]) -> list[str]:
    """Return models for which every component has a tensor, sorted."""
    per: list[set[str]] = []
    for comp in components:
        cf = _resolve_coord(comp.coord)
        per.append(set(cf.tensor_models()))
        # If a projection target is given, it also needs tensors for the same models.
        if comp.project_away is not None:
            cf_b = _resolve_coord(comp.project_away)
            per.append(set(cf_b.tensor_models()))
    if not per:
        raise MergeError("no components provided")
    shared = set.intersection(*per)
    if not shared:
        raise MergeError(
            f"no shared models across {[c.coord for c in components]}"
        )
    return sorted(shared)


def merge_into_pack(
    name: str,
    components: list[ComponentSpec],
    model: Optional[str],
    *,
    force: bool = False,
    strict: bool = False,
) -> Path:
    """Create a merged tensors-only pack at ~/.saklas/vectors/local/<name>/."""
    dst = concept_dir("local", name)
    if dst.exists() and not force:
        raise MergeError(f"{dst} exists; pass force=True to overwrite")
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    if model is not None:
        target_models = [safe_model_id(model)]
        for comp in components:
            cf = _resolve_coord(comp.coord)
            if safe_model_id(model) not in cf.tensor_models():
                raise MergeError(
                    f"component {comp.coord} has no tensor for {model}"
                )
    else:
        target_models = shared_models(components)

    component_info: dict[str, dict] = {}
    files_map: dict[str, str] = {}

    for sid in target_models:
        profiles_and_alphas: list[tuple[Profile, float]] = []
        for comp in components:
            cf = _resolve_coord(comp.coord)
            profile, _meta = load_profile(str(cf.tensor_path(sid)))
            if comp.project_away is not None:
                cf_b = _resolve_coord(comp.project_away)
                b_profile, _ = load_profile(str(cf_b.tensor_path(sid)))
                profile = project_away(profile, b_profile)
            profiles_and_alphas.append((profile, comp.alpha))
            component_info.setdefault(comp.coord, {
                "alpha": comp.alpha,
                "project_away": comp.project_away,
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

    def _comp_desc(comp: ComponentSpec) -> str:
        base = comp.coord.split("/")[-1]
        if comp.project_away is not None:
            return f"{base}~{comp.project_away.split('/')[-1]} ({comp.alpha})"
        return f"{base} ({comp.alpha})"

    desc = " + ".join(_comp_desc(c) for c in components)
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
