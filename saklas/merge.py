"""Offline vector merging: precompute a linear combination of existing
steering vectors into a distributable single-vector pack.

See docs/superpowers/specs/2026-04-12-story-a-portability-design.md §Component 6.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Optional

import torch

from saklas.packs import (
    ConceptFolder, PackMetadata, hash_file,
)
from saklas.paths import concept_dir, safe_model_id
from saklas.vectors import load_profile, save_profile

log = logging.getLogger(__name__)


Profile = dict[int, torch.Tensor]


class MergeError(ValueError):
    pass


def parse_components(raw: str) -> list[tuple[str, float]]:
    """Parse 'ns/name:alpha,ns/name:alpha' into a list of (coord, alpha)."""
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    out: list[tuple[str, float]] = []
    for part in parts:
        if ":" not in part:
            raise MergeError(f"component '{part}' missing :alpha")
        coord, alpha_s = part.rsplit(":", 1)
        try:
            out.append((coord.strip(), float(alpha_s)))
        except ValueError as e:
            raise MergeError(f"component '{part}' alpha not a number: {e}") from e
    if len(out) < 2:
        raise MergeError("merge requires at least two components")
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
    if len(components) < 2:
        raise MergeError("linear_sum requires at least two components")
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


def shared_models(components: list[tuple[str, float]]) -> list[str]:
    """Return models for which every component has a tensor, sorted."""
    per: list[set[str]] = []
    for coord, _alpha in components:
        cf = _resolve_coord(coord)
        per.append(set(cf.tensor_models()))
    if not per:
        raise MergeError("no components provided")
    shared = set.intersection(*per)
    if not shared:
        raise MergeError(f"no shared models across {[c for c, _ in components]}")
    return sorted(shared)


def merge_into_pack(
    name: str,
    components: list[tuple[str, float]],
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
        for coord, _alpha in components:
            cf = _resolve_coord(coord)
            if safe_model_id(model) not in cf.tensor_models():
                raise MergeError(f"component {coord} has no tensor for {model}")
    else:
        target_models = shared_models(components)

    component_info: dict[str, dict] = {}
    files_map: dict[str, str] = {}

    for sid in target_models:
        profiles_and_alphas: list[tuple[Profile, float]] = []
        for coord, alpha in components:
            cf = _resolve_coord(coord)
            profile, _meta = load_profile(str(cf.tensor_path(sid)))
            profiles_and_alphas.append((profile, alpha))
            component_info.setdefault(coord, {
                "alpha": alpha,
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

    desc = " + ".join(f"{c.split('/')[-1]} ({a})" for c, a in components)
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
