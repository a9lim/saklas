"""GGUF interchange: the control-vector codec plus the manifold export driver.

Two layers live here.  The **codec** — ``write_gguf_profile`` /
``read_gguf_profile`` — round-trips a per-layer direction dict through a
llama.cpp control-vector file.  The **driver** — ``export_gguf_manifold`` —
is the folder-level orchestrator over that codec: it folds a fitted 2-node
``pca`` manifold to a single direction and writes one ``.gguf`` per model,
with ``_resolve_model_hint`` deriving llama.cpp's architecture string.
GGUF is the only interchange export saklas emits.

Wire format matches the llama.cpp control-vector convention (same as
repeng's export_gguf):

  general.architecture     = "controlvector"
  controlvector.model_hint = <architecture string, e.g. "llama" / "gemma3">
  controlvector.layer_count = uint32
  direction.<layer_idx>     = float32 tensor, shape (hidden_dim,)

Because saklas bakes per-layer PCA shares into the stored direction
magnitudes at extraction time, the tensors written here are already in
llama.cpp's expected form: a uniform `--control-vector-scaled` scalar on
the consumer side reproduces saklas's layer-weighted injection without
any extra metadata slot.

Repeng-exported GGUFs are unit-normed (no share baking); reading them
yields a profile that injects uniformly across layers at apply time.
That's the correct behavior for repeng vectors — it's the semantic they
were exported with. Both paths go through the same loader.

The ``gguf`` Python package is an optional dependency; this module
lazy-imports it and raises a clear error if it's missing.
"""
from __future__ import annotations

from typing import Any, Optional

import logging
from pathlib import Path

import torch

from saklas.core.errors import SaklasError
from saklas.io.paths import safe_model_id, unsafe_model_id

log = logging.getLogger(__name__)

_ARCH = "controlvector"
_MODEL_HINT_KEY = f"{_ARCH}.model_hint"
_LAYER_COUNT_KEY = f"{_ARCH}.layer_count"
_TENSOR_PREFIX = "direction."


class GGUFNotInstalled(ImportError, SaklasError):
    """Raised when the optional ``gguf`` package is not available."""

    def user_message(self) -> tuple[int, str]:
        return (400, str(self) or self.__class__.__name__)


def _import_gguf():
    try:
        import gguf
    except ImportError as e:
        raise GGUFNotInstalled(
            "GGUF support requires the 'gguf' package. "
            "Install with: pip install saklas[gguf]"
        ) from e
    return gguf


def write_gguf_profile(
    profile: dict[int, torch.Tensor],
    path: str | Path,
    *,
    model_hint: str,
) -> Path:
    """Write a baked profile to a GGUF file at ``path``.

    Args:
        profile: layer_idx -> baked direction tensor (any dtype; cast to fp32).
        path: output file path.
        model_hint: architecture string for llama.cpp compatibility — usually
            ``model.config.model_type`` of the base model (e.g. "llama",
            "gemma3", "qwen2").

    Returns the written path.
    """
    gguf = _import_gguf()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    writer = gguf.GGUFWriter(str(path), _ARCH)
    writer.add_string(_MODEL_HINT_KEY, model_hint)
    writer.add_uint32(_LAYER_COUNT_KEY, len(profile))
    for layer_idx in sorted(profile.keys()):
        vec = profile[layer_idx].detach().to(dtype=torch.float32, device="cpu").contiguous()
        writer.add_tensor(f"{_TENSOR_PREFIX}{layer_idx}", vec.numpy())
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    log.info("Wrote GGUF profile (%d layers) to %s", len(profile), path)
    return path


def read_gguf_profile(path: str | Path) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
    """Read a GGUF control-vector file.

    Returns (profile, metadata) where metadata has a shape compatible with
    the safetensors sidecar: at minimum ``method`` and ``saklas_version``
    keys, plus ``model_hint`` from the GGUF header.

    Raises:
        GGUFNotInstalled: if the ``gguf`` package isn't installed.
        ValueError: if the file isn't a control-vector GGUF.
    """
    gguf = _import_gguf()
    path = Path(path)
    reader = gguf.GGUFReader(str(path))

    arch_field = reader.get_field("general.architecture")
    if arch_field is None or not len(arch_field.parts):
        log.warning("%s: missing general.architecture field", path)
    else:
        arch = str(bytes(arch_field.parts[-1]), encoding="utf-8", errors="replace")
        if arch != _ARCH:
            raise ValueError(
                f"{path}: general.architecture is {arch!r}, expected {_ARCH!r} "
                f"(is this actually a control-vector file?)"
            )

    hint_field = reader.get_field(_MODEL_HINT_KEY)
    if hint_field is None or not len(hint_field.parts):
        raise ValueError(f"{path}: missing {_MODEL_HINT_KEY} field")
    model_hint = str(bytes(hint_field.parts[-1]), encoding="utf-8", errors="replace")

    profile: dict[int, torch.Tensor] = {}
    for tensor in reader.tensors:
        if not tensor.name.startswith(_TENSOR_PREFIX):
            continue
        try:
            layer = int(tensor.name.split(".", 1)[1])
        except (IndexError, ValueError) as e:
            raise ValueError(
                f"{path}: invalid direction tensor name {tensor.name!r}"
            ) from e
        # tensor.data is a numpy memmap slice; copy to own the buffer.
        profile[layer] = torch.from_numpy(tensor.data.copy())

    if not profile:
        raise ValueError(f"{path}: no direction.* tensors found")

    metadata = {
        "method": "gguf_import",
        "saklas_version": _gguf_source_version(),
        "model_hint": model_hint,
    }
    return profile, metadata


def _gguf_source_version() -> str:
    """Return the saklas version string. Lazy import to avoid circularity."""
    from saklas import __version__
    return __version__


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
    hf_id = unsafe_model_id(safe_id)
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
    (:func:`~saklas.core.capture.folded_directions`), then writes a
    llama.cpp control-vector GGUF via :func:`write_gguf_profile`.
    ``model_scope`` restricts to one base model;
    without it, every fitted ``raw`` tensor is exported (one ``.gguf`` per
    model).  ``output`` policy:
      - single-model + ``.gguf`` path → write to exactly that path
      - single-model + directory → ``<dir>/<safe_model_id>.gguf``
      - multi-model → must be a directory (or None, meaning in-folder sibling)
      - None → write alongside the safetensors (rejected for bundled manifolds,
        whose folder is restored on refresh)
    """
    from saklas.io.manifolds import ManifoldFolder
    from saklas.io.paths import (
        manifold_dir, parse_tensor_filename, tensor_filename,
    )
    from saklas.io.manifold_tensors import load_manifold
    from saklas.core.capture import folded_directions

    mdir = manifold_dir(ns, name)
    # Preflight needs folder identity/source and fitted filenames only.  Each
    # selected raw tensor is integrity-checked by ``load_manifold`` below; do
    # not eagerly hash unrelated models, SAE variants, or prior GGUF exports.
    mf = ManifoldFolder.load(mdir, verify_manifest=False)

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
        manifold = load_manifold(
            mdir / tensor_filename(
                sid, release=None, model_id_is_safe=True,
            )
        )
        try:
            profile = folded_directions(manifold)
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

    # The export only extends the folder's integrity manifest with the new
    # ``.gguf`` entries; a manifold's namespace/name/tags/node labels are
    # untouched, so the selector index stays valid.
    if any(p.parent == mdir for p in written):
        mf.update_file_hashes(*(p for p in written if p.parent == mdir))

    return written
