"""GGUF read/write for baked steering-vector profiles.

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

import logging
from pathlib import Path

import torch

from saklas.core.errors import SaklasError

log = logging.getLogger(__name__)

_ARCH = "controlvector"
_MODEL_HINT_KEY = f"{_ARCH}.model_hint"
_LAYER_COUNT_KEY = f"{_ARCH}.layer_count"
_TENSOR_PREFIX = "direction."


class GGUFNotInstalled(ImportError, SaklasError):
    """Raised when the optional ``gguf`` package is not available."""


def _import_gguf():
    try:
        import gguf  # type: ignore
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


def read_gguf_profile(path: str | Path) -> tuple[dict[int, torch.Tensor], dict]:
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
