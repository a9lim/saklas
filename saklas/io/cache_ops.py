"""GGUF export for fitted manifolds + the model-hint resolver.

What remains of the old pack data layer after the 4.0 collapse: a steering
vector is a 2-node ``pca`` manifold, so the only interchange export left is
folding such a manifold to a single direction and writing a llama.cpp
control-vector GGUF.  Everything pack-shaped (install/refresh/clear/ls/search/
push + the ``ConceptRow``/``PackListResult`` result types) is gone — manifolds
own distribution now (:mod:`saklas.io.hf_manifolds`).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from saklas.io.paths import safe_model_id
from saklas.io.selectors import invalidate as _invalidate_selector_cache


def _resolve_model_hint(safe_id: str) -> str:
    """Derive llama.cpp's ``controlvector.model_hint`` from a safe_model_id.

    Strategy: load the base model's config via ``transformers.AutoConfig``
    (cache-first, network fallback) and return ``config.model_type``.  That's
    the same string llama.cpp's loader keys off when matching a control
    vector to a loaded model (e.g. ``"llama"``, ``"gemma2"``, ``"qwen2"``).

    Raises RuntimeError with actionable guidance if the config can't be
    resolved — callers should surface the ``--model-hint`` flag as the
    escape hatch.
    """
    hf_id = safe_id.replace("__", "/")
    try:
        from transformers import AutoConfig
    except ImportError as e:  # pragma: no cover — transformers is a hard dep
        raise RuntimeError(
            f"could not resolve model_hint for {hf_id!r}: transformers missing ({e})"
        ) from e
    try:
        cfg = AutoConfig.from_pretrained(hf_id, trust_remote_code=False)
    except Exception as e:
        raise RuntimeError(
            f"could not resolve model_hint for {hf_id!r}: {e}. "
            f"Pass --model-hint <arch> explicitly (e.g. 'llama', 'gemma2', 'qwen2')."
        ) from e
    mt = getattr(cfg, "model_type", None)
    if not mt:
        raise RuntimeError(
            f"{hf_id}: config has no model_type field; pass --model-hint explicitly"
        )
    return str(mt)


def export_gguf_manifold(
    ns: str,
    name: str,
    *,
    model_scope: Optional[str],
    output: Optional[str],
    model_hint: Optional[str],
) -> list[Path]:
    """Export a fitted 2-node ``pca`` manifold to GGUF by folding it to a vector.

    Sources each per-model profile by folding the manifold
    (:func:`~saklas.core.vectors.folded_vector_directions`), then writes a
    llama.cpp control-vector GGUF.  ``model_scope`` restricts to one base model;
    without it, every fitted ``raw`` tensor is exported (one ``.gguf`` per
    model).  ``output`` policy mirrors the old vector export:
      - single-model + ``.gguf`` path → write to exactly that path
      - single-model + directory → ``<dir>/<safe_model_id>.gguf``
      - multi-model → must be a directory (or None, meaning in-folder sibling)
      - None → write alongside the safetensors (rejected for bundled manifolds,
        whose folder is restored on refresh)
    """
    from saklas.io.gguf_io import write_gguf_profile
    from saklas.io.manifolds import ManifoldFolder, hash_manifold_files
    from saklas.io.paths import (
        manifold_dir, parse_tensor_filename, tensor_filename,
    )
    from saklas.core.manifold import load_manifold
    from saklas.core.vectors import folded_vector_directions

    mdir = manifold_dir(ns, name)
    mf = ManifoldFolder.load(mdir)

    raw_models: set[str] = set()
    for tensor in mdir.glob("*.safetensors"):
        parsed = parse_tensor_filename(tensor.name)
        if parsed is not None and parsed[1] is None:  # raw fitted → foldable
            raw_models.add(parsed[0])

    if model_scope is not None:
        sid = safe_model_id(model_scope)
        if sid not in raw_models:
            raise RuntimeError(
                f"{ns}/{name}: no fitted manifold tensor for {model_scope}"
            )
        targets = [sid]
    else:
        targets = sorted(raw_models)
        if not targets:
            raise RuntimeError(
                f"{ns}/{name}: no fitted manifold tensors to export"
            )

    out_path = Path(output) if output else None
    if out_path is not None and len(targets) > 1 and out_path.suffix == ".gguf":
        raise RuntimeError(
            "multi-model export needs a directory or no --output; "
            f"got file path {out_path}"
        )
    if out_path is None and (mf.source or "").startswith("bundled"):
        raise RuntimeError(
            f"{ns}/{name}: bundled manifold — in-place GGUF export would be lost "
            f"on next refresh. Pass --output <path> to write outside the folder."
        )

    written: list[Path] = []
    for sid in targets:
        manifold = load_manifold(mdir / tensor_filename(sid, release=None))
        try:
            profile = folded_vector_directions(manifold)
        except Exception as e:
            raise RuntimeError(
                f"{ns}/{name}: manifold does not fold to a single steering "
                f"direction (not a 2-node affine subspace): {e}"
            ) from e
        hint = model_hint or _resolve_model_hint(sid)
        if out_path is None:
            dest = mdir / f"{sid}.gguf"
        elif out_path.suffix == ".gguf":
            dest = out_path
        else:
            out_path.mkdir(parents=True, exist_ok=True)
            dest = out_path / f"{sid}.gguf"
        write_gguf_profile(profile, dest, model_hint=hint)
        written.append(dest)

    if any(p.parent == mdir for p in written):
        mf.write_metadata(files=hash_manifold_files(mdir))
        _invalidate_selector_cache()

    return written
