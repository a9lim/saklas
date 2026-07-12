"""Per-model Jacobian-lens artifact: save/load under ``models/<safe_id>/``.

The lens is a per-model transport (one ``J_l`` matrix per source layer), not a
per-concept artifact. Lenses fitted by Saklas are owned by Saklas and live in
the model's source registry:

    ~/.saklas/models/<safe_model_id>/jlens/local/default/
      jlens.layer-<L>.gen-<id>.safetensors
      manifest.json                                      # atomic shard pointer

fp32 on disk and in memory, matching saklas' fitted manifold, subspace, profile,
and neutral-activation artifacts. Keeping the estimator's fp32 accumulator
lossless across persistence also makes resumed fits numerically consistent with
uninterrupted fits.

The sidecar records the token-id corpus sha256 + loaded-model fingerprint so a
different corpus or mutable model revision reads as stale, and ``n_prompts`` so an interrupted fit can resume
(load → fit the remaining prompts → ``JacobianLens.merge`` → save).
"""

from __future__ import annotations

import json
import hashlib
import logging
import os
import struct
import uuid
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable, Iterable, Mapping
from typing import Any, cast

import torch
from safetensors import safe_open

from saklas.core.jlens import JacobianLens
from saklas.io.atomic import artifact_lock, fsync_directory, write_json_atomic

log = logging.getLogger(__name__)

LENS_FORMAT_VERSION = 6
_LENS_NAME = "jlens"
_LENS_CHECKPOINT_NAME = "jlens.partial"
_LENS_METHOD = "jlens_cotangent_sum"
LENS_CORPUS_PREPROCESS_VERSION = 1

_LENS_SIDECAR_FIELDS = {
    "format_version", "method", "n_prompts", "d_model", "source_layers",
    "dtype", "corpus_spec", "corpus_sha256", "corpus_hash_kind", "seq_len",
    "dim_batch", "skip_first_positions", "estimator_policy", "tensor_sha256",
    "tensor_files", "raw_corpus_sha256", "raw_prompt_count",
    "usable_prompt_count", "model_layer_count", "model_fingerprint",
    "model_source_fingerprint", "checkpoint", "base_n_prompts",
    "partial_n_prompts", "consumed_prefix_sha256",
}


@dataclass(frozen=True)
class _LensPayloadProof:
    """Transaction-scoped proof that one immutable pointer was byte-verified.

    Constructed only after a successful load or pointer publication.  Callers
    may reuse it while holding ``lens_fit_lock``; public/unverified callers pass
    no proof and retain full digest verification.
    """

    anchor: str
    pointer_sha256: str


def _lens_payload_proof(
    anchor: Path, sidecar: Mapping[str, Any],
) -> _LensPayloadProof:
    canonical = json.dumps(
        dict(sidecar), sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return _LensPayloadProof(
        anchor=str(anchor.resolve(strict=False)),
        pointer_sha256=hashlib.sha256(canonical).hexdigest(),
    )


def _proof_matches(
    proof: _LensPayloadProof | None,
    anchor: Path,
    sidecar: Mapping[str, Any],
) -> bool:
    return proof == _lens_payload_proof(anchor, sidecar) if proof is not None else False


def _json_pointer_matches(path: Path, payload: Mapping[str, Any]) -> bool:
    """Whether an exception escaped after this exact pointer was replaced."""
    try:
        with open(path) as handle:
            current = json.load(handle)
        return current == payload
    except (OSError, json.JSONDecodeError, TypeError):
        return False


def _lens_anchor_paths(model_id: str) -> tuple[Path, Path]:
    """Stable lock anchor and atomic durable pointer."""
    from saklas.io.lens_sources import local_lens_dir

    root = local_lens_dir(model_id)
    return root / f"{_LENS_NAME}.safetensors", root / "manifest.json"


def _checkpoint_anchor_paths(model_id: str) -> tuple[Path, Path]:
    """Stable lock anchor and atomic checkpoint pointer."""
    from saklas.io.lens_sources import local_lens_dir

    md = local_lens_dir(model_id)
    return (
        md / f"{_LENS_CHECKPOINT_NAME}.safetensors",
        md / "checkpoint.json",
    )

def _sidecar_tensor_path(anchor: Path, sidecar: Mapping[str, Any]) -> Path:
    """Resolve a representative generation for path-reporting compatibility."""
    shard_files = sidecar.get("tensor_files")
    if isinstance(shard_files, Mapping) and shard_files:
        try:
            first_key = min(shard_files, key=lambda key: int(str(key)))
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid lens tensor shard keys") from exc
        filename = shard_files[first_key]
        if (
            not isinstance(filename, str)
            or not filename
            or Path(filename).name != filename
        ):
            raise ValueError("invalid lens tensor shard filename")
        return anchor.parent / filename
    raise ValueError("lens sidecar has no tensor shard map")


def _sidecar_tensor_paths(
    anchor: Path, sidecar: Mapping[str, Any], source_layers: list[int], *,
    require_exact_keys: bool = True,
) -> dict[int, Path]:
    """Resolve every declared layer to its immutable tensor generation."""
    shard_files = sidecar.get("tensor_files")
    if not isinstance(shard_files, Mapping):
        raise ValueError("invalid lens tensor_files mapping")
    out: dict[int, Path] = {}
    for layer in source_layers:
        filename = shard_files.get(str(layer))
        if (
            not isinstance(filename, str)
            or not filename
            or Path(filename).name != filename
        ):
            raise ValueError(f"missing or invalid tensor shard for layer {layer}")
        out[layer] = anchor.parent / filename
    if (
        require_exact_keys
        and {str(layer) for layer in source_layers}
        != {str(key) for key in shard_files}
    ):
        raise ValueError("tensor shard keys do not match source_layers")
    return out


def _public_pointer_paths(
    anchor: Path, sidecar_path: Path,
) -> tuple[Path, Path]:
    """Best-effort current tensor path for reporting callers."""
    try:
        with open(sidecar_path) as handle:
            sidecar = json.load(handle)
        return _sidecar_tensor_path(anchor, sidecar), sidecar_path
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return anchor, sidecar_path


def lens_paths(model_id: str) -> tuple[Path, Path]:
    """Return a representative tensor shard and stable sidecar pointer path.

    The current format uses immutable per-layer generations and atomically
    switches ``jlens/local/default/manifest.json`` to the complete shard map. Use
    :func:`lens_tensor_paths` when every layer path is required.
    """
    return _public_pointer_paths(*_lens_anchor_paths(model_id))


def lens_checkpoint_paths(model_id: str) -> tuple[Path, Path]:
    """Return the current checkpoint generation and stable sidecar pointer."""
    return _public_pointer_paths(*_checkpoint_anchor_paths(model_id))


def lens_tensor_paths(
    model_id: str, sidecar: Mapping[str, Any], *, checkpoint: bool = False,
) -> dict[int, Path]:
    """Return every tensor generation named by a validated sidecar."""
    anchor, _ = (
        _checkpoint_anchor_paths(model_id) if checkpoint
        else _lens_anchor_paths(model_id)
    )
    layers = [int(layer) for layer in sidecar.get("source_layers", [])]
    return _sidecar_tensor_paths(anchor, sidecar, layers)


def lens_artifact_size(
    model_id: str, sidecar: Mapping[str, Any], *, checkpoint: bool = False,
) -> int:
    """Total bytes in the current pointer, retrying a stale caller snapshot."""
    del sidecar  # the stable pointer is authoritative under its anchor lock
    anchor, sc_path = (
        _checkpoint_anchor_paths(model_id) if checkpoint
        else _lens_anchor_paths(model_id)
    )
    with artifact_lock(anchor):
        current = _load_sidecar_at(
            model_id, anchor, sc_path,
            label="jlens checkpoint" if checkpoint else "jlens cache",
        )
        if current is None:
            return 0
        return sum(
            path.stat().st_size
            for path in set(_sidecar_tensor_paths(
                anchor, current,
                [int(layer) for layer in current["source_layers"]],
            ).values())
        )


def lens_payloads_match(
    model_id: str, sidecar: Mapping[str, Any], *, checkpoint: bool = False,
) -> bool:
    """Validate exact payload digests without materializing fp32 matrices."""
    from saklas.io.packs import hash_file

    try:
        paths = lens_tensor_paths(model_id, sidecar, checkpoint=checkpoint)
        expected = sidecar.get("tensor_sha256")
        if not isinstance(expected, Mapping):
            return False
        return all(
            isinstance(expected.get(str(layer)), str)
            and hash_file(path) == expected[str(layer)]
            for layer, path in paths.items()
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


@contextmanager
def lens_fit_lock(model_id: str):
    """Serialize the complete per-model fit/lifecycle transaction cross-process."""
    anchor, _ = _lens_anchor_paths(model_id)
    with artifact_lock(anchor.parent / "jlens.fit"):
        yield


def _new_layer_generation(anchor: Path, layer: int) -> Path:
    return anchor.with_name(
        f"{anchor.stem}.layer-{int(layer)}.gen-{uuid.uuid4().hex}{anchor.suffix}",
    )


def _referenced_tensor_names(model_folder: Path) -> set[str] | None:
    """Read both pointers, failing closed if an existing pointer is unreadable."""
    out: set[str] = set()
    for sidecar_path in (
        model_folder / "manifest.json",
        model_folder / "checkpoint.json",
    ):
        if not sidecar_path.exists():
            continue
        try:
            with open(sidecar_path) as handle:
                sidecar = json.load(handle)
            if not isinstance(sidecar, dict):
                return None
            shard_files = sidecar.get("tensor_files")
            if not isinstance(shard_files, Mapping):
                return None
            filenames = shard_files.values()
            for filename in filenames:
                if isinstance(filename, str) and Path(filename).name == filename:
                    out.add(filename)
        except (OSError, TypeError, json.JSONDecodeError):
            return None
    return out


def _cleanup_unreferenced_generations(model_folder: Path) -> None:
    """Remove unreferenced generations and crash-left streaming temporaries."""
    keep = _referenced_tensor_names(model_folder)
    if keep is None:
        # A transient pointer read failure cannot authorize deletion of any
        # potentially live multi-GiB generation.
        return
    candidates = (
        list(model_folder.glob("jlens*.gen-*.safetensors"))
        + list(model_folder.glob("jlens*.gen-*.safetensors.tmp"))
        + [
        model_folder / f"{_LENS_NAME}.safetensors",
        model_folder / f"{_LENS_CHECKPOINT_NAME}.safetensors",
        model_folder / f"{_LENS_NAME}.safetensors.tmp",
        model_folder / f"{_LENS_CHECKPOINT_NAME}.safetensors.tmp",
        ]
    )
    for path in candidates:
        if path.name not in keep:
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                # Pointer publication is already complete. Generation GC is
                # opportunistic and must not turn a valid fit into a failure.
                log.warning("could not remove old J-lens generation %s: %s", path, exc)


def cleanup_lens_artifacts(model_id: str) -> None:
    """Reap crash-left lens shards before a new fit allocates replacements."""
    anchor, _ = _lens_anchor_paths(model_id)
    with lens_fit_lock(model_id):
        _cleanup_unreferenced_generations(anchor.parent)


def lens_estimator_policy(*, skip_first: int | None = None) -> dict[str, Any]:
    """Fit semantics that must match before any final/checkpoint reuse."""
    from saklas.core.jlens import SKIP_FIRST_POSITIONS

    return {
        "method": _LENS_METHOD,
        "skip_first_positions": (
            SKIP_FIRST_POSITIONS if skip_first is None else int(skip_first)
        ),
        "corpus_preprocess_version": LENS_CORPUS_PREPROCESS_VERSION,
        "doc_chars": LENS_DOC_CHARS,
    }


def _load_sidecar_at(
    model_id: str, tensor_anchor: Path, sc_path: Path, *, label: str,
) -> dict[str, Any] | None:
    if not sc_path.exists():
        return None
    try:
        with open(sc_path) as f:
            sidecar = json.load(f)
        if not isinstance(sidecar, dict) or set(sidecar) != _LENS_SIDECAR_FIELDS:
            raise ValueError("lens sidecar does not match the current exact schema")
        ts_path = _sidecar_tensor_path(tensor_anchor, sidecar)
        if not ts_path.exists():
            return None
        version = sidecar.get("format_version")
        if (
            not isinstance(version, int)
            or isinstance(version, bool)
            or version != LENS_FORMAT_VERSION
        ):
            log.warning(
                "%s for %s has format_version %r (need exactly %d); ignoring "
                "— re-fit with `saklas lens fit`",
                label, model_id, version, LENS_FORMAT_VERSION,
            )
            return None
        if sidecar.get("estimator_policy") != lens_estimator_policy():
            log.warning(
                "%s for %s uses a different estimator/preprocessing policy; "
                "ignoring — re-fit with `saklas lens fit`", label, model_id,
            )
            return None
        d_model = sidecar["d_model"]
        source_layers_raw = sidecar["source_layers"]
        if (
            isinstance(d_model, bool)
            or not isinstance(d_model, int)
            or d_model <= 0
            or not isinstance(source_layers_raw, list)
            or not source_layers_raw
            or any(
                isinstance(layer, bool) or not isinstance(layer, int) or layer < 0
                for layer in source_layers_raw
            )
            or source_layers_raw != sorted(set(source_layers_raw))
            or sidecar["method"] != _LENS_METHOD
            or sidecar["dtype"] != "float32"
            or any(
                not isinstance(sidecar[key], str) or not sidecar[key]
                for key in ("corpus_spec", "corpus_sha256", "corpus_hash_kind")
            )
            or any(
                isinstance(sidecar[key], bool) or not isinstance(sidecar[key], int)
                or sidecar[key] < 0
                for key in ("n_prompts", "seq_len", "dim_batch", "skip_first_positions")
            )
            or not isinstance(sidecar["checkpoint"], bool)
        ):
            log.warning(
                "%s for %s has invalid sidecar shape metadata; ignoring "
                "— re-fit with `saklas lens fit`", label, model_id,
            )
            return None
        nullable_strings = (
            "raw_corpus_sha256", "model_fingerprint",
            "model_source_fingerprint", "consumed_prefix_sha256",
        )
        if any(
            sidecar[key] is not None and (
                not isinstance(sidecar[key], str) or not sidecar[key]
            )
            for key in nullable_strings
        ):
            raise ValueError("lens sidecar has invalid nullable string metadata")
        digest_fields = (
            "corpus_sha256", "raw_corpus_sha256", "consumed_prefix_sha256",
        )
        if any(
            sidecar[key] is not None and (
                len(sidecar[key]) != 64
                or any(char not in "0123456789abcdef" for char in sidecar[key])
            )
            for key in digest_fields
        ):
            raise ValueError("lens sidecar has invalid identity digest")
        nullable_ints = (
            "raw_prompt_count", "usable_prompt_count", "model_layer_count",
            "base_n_prompts", "partial_n_prompts",
        )
        if any(
            value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            )
            for value in (sidecar[key] for key in nullable_ints)
        ):
            raise ValueError("lens sidecar has invalid nullable integer metadata")
        if sidecar["checkpoint"]:
            if sidecar["base_n_prompts"] is None or sidecar["partial_n_prompts"] is None:
                raise ValueError("lens checkpoint is missing progress metadata")
            if (
                sidecar["partial_n_prompts"] > 0
                and sidecar["consumed_prefix_sha256"] is None
            ):
                raise ValueError("lens checkpoint is missing consumed-prefix identity")
            if (
                sidecar["partial_n_prompts"] == 0
                and sidecar["consumed_prefix_sha256"] is not None
            ):
                raise ValueError("zero-progress lens checkpoint carries prefix identity")
        elif any(
            sidecar[key] is not None
            for key in ("base_n_prompts", "partial_n_prompts", "consumed_prefix_sha256")
        ):
            raise ValueError("final lens carries checkpoint-only metadata")
        expected_digests = sidecar.get("tensor_sha256")
        expected_keys = {str(layer) for layer in sidecar["source_layers"]}
        if (
            not isinstance(expected_digests, Mapping)
            or {str(key) for key in expected_digests} != expected_keys
            or any(
                not isinstance(digest, str) or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)
                for digest in expected_digests.values()
            )
        ):
            raise ValueError("tensor digest keys do not match source_layers")
        tensor_paths = _sidecar_tensor_paths(
            tensor_anchor, sidecar, sidecar["source_layers"],
        )
        grouped: dict[Path, list[int]] = {}
        for layer, path in tensor_paths.items():
            if not path.exists():
                raise ValueError(f"missing tensor generation {path.name}")
            grouped.setdefault(path, []).append(layer)
        for path, path_layers in grouped.items():
            path_keys = [f"layer_{layer}" for layer in path_layers]
            with safe_open(str(path), framework="pt", device="cpu") as tensors:
                if sorted(tensors.keys()) != sorted(path_keys):
                    raise ValueError("tensor layer keys do not match sidecar")
                for key in path_keys:
                    view = tensors.get_slice(key)
                    if tuple(view.get_shape()) != (d_model, d_model):
                        raise ValueError(f"{key} shape does not match sidecar")
                    if view.get_dtype() != "F32":
                        raise ValueError(f"{key} is not float32")
        return sidecar
    except Exception as exc:
        log.warning("Corrupt %s sidecar for %s; ignoring: %s", label, model_id, exc)
        return None


def load_local_lens_sidecar(model_id: str) -> dict[str, Any] | None:
    """Load Saklas-owned local/default metadata without its matrices."""
    anchor, sc_path = _lens_anchor_paths(model_id)
    with artifact_lock(anchor):
        return _load_sidecar_at(model_id, anchor, sc_path, label="jlens cache")


def load_lens_sidecar(model_id: str) -> dict[str, Any] | None:
    """Load metadata for the active local or external lens source."""
    from saklas.io.lens_sources import (
        load_active_lens_source,
        load_external_lens_sidecar,
    )

    active = load_active_lens_source(model_id)
    if active is None:
        return None
    if active["kind"] == "huggingface":
        return load_external_lens_sidecar(model_id, active["name"])
    return load_local_lens_sidecar(model_id)


def load_lens_checkpoint_sidecar(model_id: str) -> dict[str, Any] | None:
    """Load validated checkpoint metadata without materializing its matrices."""
    anchor, sc_path = _checkpoint_anchor_paths(model_id)
    with artifact_lock(anchor):
        return _load_sidecar_at(
            model_id, anchor, sc_path, label="jlens checkpoint",
        )


def save_lens(
    lens: JacobianLens,
    model_id: str,
    *,
    corpus_spec: str,
    corpus_sha256: str,
    seq_len: int,
    dim_batch: int,
    skip_first: int,
    corpus_hash_kind: str = "text_v1",
    durable: bool = True,
    raw_corpus_sha256: str | None = None,
    raw_prompt_count: int | None = None,
    usable_prompt_count: int | None = None,
    model_layer_count: int | None = None,
    model_fingerprint: str | None = None,
    model_source_fingerprint: str | None = None,
    reuse_layers: Iterable[int] | None = None,
    _verified_reuse_proof: _LensPayloadProof | None = None,
) -> Path:
    """Persist a fitted lens and atomically point at its layer generations.

    ``reuse_layers`` names matrices known to be byte-identical to the current
    durable artifact (the missing-layer top-up path). Valid v6 shard pointers
    are carried forward; every other layer is converted and written anew.
    """
    anchor, sc_path = _lens_anchor_paths(model_id)
    with lens_fit_lock(model_id):
        path = _save_lens_at(
            lens, anchor, sc_path,
            corpus_spec=corpus_spec,
            corpus_sha256=corpus_sha256,
            seq_len=seq_len,
            dim_batch=dim_batch,
            skip_first=skip_first,
            corpus_hash_kind=corpus_hash_kind,
            durable=durable,
            raw_corpus_sha256=raw_corpus_sha256,
            raw_prompt_count=raw_prompt_count,
            usable_prompt_count=usable_prompt_count,
            model_layer_count=model_layer_count,
            model_fingerprint=model_fingerprint,
            model_source_fingerprint=model_source_fingerprint,
            reuse_layers=reuse_layers,
            verified_reuse_proof=_verified_reuse_proof,
        )
    from saklas.io.lens_sources import set_active_local_lens

    set_active_local_lens(model_id)
    return path


def save_lens_checkpoint_accumulator(
    sums: Mapping[int, torch.Tensor],
    n_prompts: int,
    d_model: int,
    model_id: str,
    *,
    base: JacobianLens | None,
    corpus_spec: str,
    corpus_sha256: str,
    seq_len: int,
    dim_batch: int,
    skip_first: int,
    corpus_hash_kind: str = "text_v1",
    raw_corpus_sha256: str | None = None,
    raw_prompt_count: int | None = None,
    usable_prompt_count: int | None = None,
    model_layer_count: int | None = None,
    model_fingerprint: str | None = None,
    model_source_fingerprint: str | None = None,
    consumed_prefix_sha256: str | None = None,
    _proof_out: list[_LensPayloadProof] | None = None,
) -> Path:
    """Write a self-contained checkpoint directly from raw estimator sums.

    Normalization and optional prefix merging happen one layer at a time while
    streaming fp32 rows, avoiding an extra full-fp32 ``JacobianLens``. The shard has
    ``base_n_prompts=0`` and can therefore survive any number of interruptions
    without depending on a separate full artifact.
    """
    anchor, sc_path = _checkpoint_anchor_paths(model_id)
    total_prompts = int(n_prompts) + (base.n_prompts if base is not None else 0)
    with lens_fit_lock(model_id), artifact_lock(anchor):
        return _save_lens_components(
            sums, total_prompts, d_model, anchor, sc_path,
            corpus_spec=corpus_spec,
            corpus_sha256=corpus_sha256,
            seq_len=seq_len,
            dim_batch=dim_batch,
            skip_first=skip_first,
            corpus_hash_kind=corpus_hash_kind,
            durable=True,
            raw_corpus_sha256=raw_corpus_sha256,
            raw_prompt_count=raw_prompt_count,
            usable_prompt_count=usable_prompt_count,
            model_layer_count=model_layer_count,
            model_fingerprint=model_fingerprint,
            model_source_fingerprint=model_source_fingerprint,
            raw_sum_count=int(n_prompts),
            average_base=base,
            extra_sidecar={
                "checkpoint": True,
                "base_n_prompts": 0,
                "partial_n_prompts": total_prompts,
                **(
                    {"consumed_prefix_sha256": consumed_prefix_sha256}
                    if consumed_prefix_sha256 is not None else {}
                ),
            },
            proof_out=_proof_out,
        )


def _save_lens_at(
    lens: JacobianLens,
    ts_path: Path,
    sc_path: Path,
    *,
    corpus_spec: str,
    corpus_sha256: str,
    seq_len: int,
    dim_batch: int,
    skip_first: int,
    corpus_hash_kind: str,
    durable: bool,
    raw_corpus_sha256: str | None = None,
    raw_prompt_count: int | None = None,
    usable_prompt_count: int | None = None,
    model_layer_count: int | None = None,
    model_fingerprint: str | None = None,
    model_source_fingerprint: str | None = None,
    reuse_layers: Iterable[int] | None = None,
    verified_reuse_proof: _LensPayloadProof | None = None,
    extra_sidecar: dict[str, Any] | None = None,
) -> Path:
    with artifact_lock(ts_path):
        return _save_lens_components(
            lens.jacobians, lens.n_prompts, lens.d_model, ts_path, sc_path,
            corpus_spec=corpus_spec,
            corpus_sha256=corpus_sha256,
            seq_len=seq_len,
            dim_batch=dim_batch,
            skip_first=skip_first,
            corpus_hash_kind=corpus_hash_kind,
            durable=durable,
            raw_corpus_sha256=raw_corpus_sha256,
            raw_prompt_count=raw_prompt_count,
            usable_prompt_count=usable_prompt_count,
            model_layer_count=model_layer_count,
            model_fingerprint=model_fingerprint,
            model_source_fingerprint=model_source_fingerprint,
            reuse_layers=reuse_layers,
            verified_reuse_proof=verified_reuse_proof,
            extra_sidecar=extra_sidecar,
        )


def _save_lens_components(
    jacobians: Mapping[int, torch.Tensor],
    n_prompts: int,
    d_model: int,
    ts_path: Path,
    sc_path: Path,
    *,
    corpus_spec: str,
    corpus_sha256: str,
    seq_len: int,
    dim_batch: int,
    skip_first: int,
    corpus_hash_kind: str,
    durable: bool,
    raw_corpus_sha256: str | None = None,
    raw_prompt_count: int | None = None,
    usable_prompt_count: int | None = None,
    model_layer_count: int | None = None,
    model_fingerprint: str | None = None,
    model_source_fingerprint: str | None = None,
    raw_sum_count: int | None = None,
    average_base: JacobianLens | None = None,
    reuse_layers: Iterable[int] | None = None,
    verified_reuse_proof: _LensPayloadProof | None = None,
    extra_sidecar: dict[str, Any] | None = None,
    proof_out: list[_LensPayloadProof] | None = None,
) -> Path:
    ts_path.parent.mkdir(parents=True, exist_ok=True)
    layer_ids = sorted(int(idx) for idx in jacobians)

    def _rows(idx: int, start: int, end: int) -> torch.Tensor:
        value = jacobians[idx][start:end].to(device="cpu", dtype=torch.float32)
        if raw_sum_count is not None:
            # Stripe-local arithmetic: checkpointing never clones a complete
            # fp32 layer or keeps a complete artifact mapping alive.
            value = value.clone()
            if average_base is not None:
                value.add_(
                    average_base.jacobians[idx][start:end],
                    alpha=average_base.n_prompts,
                )
            value.mul_(1.0 / max(int(n_prompts), 1))
        return value.contiguous()

    reuse_set = {int(layer) for layer in reuse_layers or ()}
    tensor_files: dict[str, str] = {}
    tensor_sha256: dict[str, str] = {}
    created: list[Path] = []
    if reuse_set:
        current = _load_sidecar_at(
            "current", ts_path, sc_path, label="jlens reuse source",
        )
        if (
            current is not None
            and int(current.get("format_version", 0)) == LENS_FORMAT_VERSION
            and int(current.get("d_model", -1)) == int(d_model)
            and int(current.get("n_prompts", -1)) == int(n_prompts)
            and current.get("corpus_sha256") == corpus_sha256
            and current.get("corpus_hash_kind") == corpus_hash_kind
            and int(current.get("seq_len", -1)) == int(seq_len)
            and current.get("model_fingerprint") == model_fingerprint
            and isinstance(current.get("tensor_sha256"), Mapping)
        ):
            current_layers = [int(layer) for layer in current["source_layers"]]
            current_paths = _sidecar_tensor_paths(
                ts_path, current, current_layers,
            )
            current_digests = current["tensor_sha256"]
            from saklas.io.packs import hash_file
            pointer_verified = _proof_matches(
                verified_reuse_proof, ts_path, current,
            )

            for layer in sorted(reuse_set & set(layer_ids) & set(current_layers)):
                digest = current_digests.get(str(layer))
                path = current_paths[layer]
                try:
                    payload_matches = (
                        isinstance(digest, str) and len(digest) == 64
                        and path.exists()
                        and (pointer_verified or hash_file(path) == digest)
                    )
                except OSError:
                    payload_matches = False
                if (
                    payload_matches and isinstance(digest, str)
                ):
                    tensor_files[str(layer)] = path.name
                    tensor_sha256[str(layer)] = digest
    try:
        for layer in layer_ids:
            if str(layer) in tensor_files:
                continue
            generation_path = _new_layer_generation(ts_path, layer)
            digest = _save_fp32_square_safetensors_atomic(
                generation_path, [layer], d_model, _rows, durable=durable,
            )
            created.append(generation_path)
            tensor_files[str(layer)] = generation_path.name
            tensor_sha256[str(layer)] = digest
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    sidecar: dict[str, Any] = {
        "format_version": LENS_FORMAT_VERSION,
        "method": _LENS_METHOD,
        "n_prompts": int(n_prompts),
        "d_model": int(d_model),
        "source_layers": sorted(int(layer) for layer in jacobians),
        "dtype": "float32",
        "corpus_spec": corpus_spec,
        "corpus_sha256": corpus_sha256,
        "corpus_hash_kind": corpus_hash_kind,
        "seq_len": seq_len,
        "dim_batch": dim_batch,
        "skip_first_positions": skip_first,
        "estimator_policy": lens_estimator_policy(skip_first=skip_first),
        "tensor_sha256": tensor_sha256,
        "tensor_files": tensor_files,
        "raw_corpus_sha256": raw_corpus_sha256,
        "raw_prompt_count": None if raw_prompt_count is None else int(raw_prompt_count),
        "usable_prompt_count": (
            None if usable_prompt_count is None else int(usable_prompt_count)
        ),
        "model_layer_count": (
            None if model_layer_count is None else int(model_layer_count)
        ),
        "model_fingerprint": model_fingerprint,
        "model_source_fingerprint": model_source_fingerprint,
        "checkpoint": False,
        "base_n_prompts": None,
        "partial_n_prompts": None,
        "consumed_prefix_sha256": None,
    }
    if extra_sidecar:
        sidecar.update(extra_sidecar)
    if set(sidecar) != _LENS_SIDECAR_FIELDS:
        raise ValueError("lens writer produced a non-canonical sidecar")
    try:
        # Every immutable layer generation is complete before this one atomic
        # pointer switch. A sidecar failure leaves the prior pointer and tensor
        # untouched, and the failed new generations are simply discarded.
        fsync_directory(sc_path.parent)
        write_json_atomic(sc_path, sidecar)
    except BaseException:
        # A signal may unwind just after the atomic replace. Never delete
        # immutable generations the live pointer already names.
        if not _json_pointer_matches(sc_path, sidecar):
            for path in created:
                path.unlink(missing_ok=True)
        raise
    # Make the new pointer rename durable before deleting any generation the
    # previous pointer named. Otherwise power loss can roll the directory back
    # to an old pointer whose shards GC already removed.
    fsync_directory(sc_path.parent)
    if proof_out is not None:
        proof_out[:] = [_lens_payload_proof(ts_path, sidecar)]
    _cleanup_unreferenced_generations(ts_path.parent)
    fsync_directory(sc_path.parent)
    return ts_path.parent / tensor_files[str(layer_ids[0])]


def _save_fp32_square_safetensors_atomic(
    path: Path,
    layers: list[int],
    d_model: int,
    rows: Callable[[int, int, int], torch.Tensor],
    *,
    durable: bool = True,
) -> str:
    """Stream square fp32 layer tensors into one atomic safetensors file.

    ``safetensors.torch.save_file`` requires a complete tensor mapping, which
    would make checkpoint RSS include another full lens beside the fp32 estimator.
    The wire format is deliberately simple: an 8-byte little-endian JSON-header
    length, a space-padded JSON header, then contiguous tensor payloads.  Writing
    256-row fp32 stripes avoids a complete in-memory snapshot. Current writers
    call this once per layer.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp") if path.suffix else path.with_name(
        path.name + ".tmp"
    )
    prefix, raw_header = _lens_safetensors_header(layers, d_model)
    digest = hashlib.sha256()
    try:
        with open(tmp, "wb", buffering=0) as f:
            f.write(prefix)
            digest.update(prefix)
            f.write(raw_header)
            digest.update(raw_header)
            for layer in sorted(layers):
                for start in range(0, d_model, 256):
                    block = rows(layer, start, min(start + 256, d_model))
                    expected = (min(start + 256, d_model) - start, d_model)
                    if block.dtype != torch.float32 or tuple(block.shape) != expected:
                        raise ValueError(
                            f"streamed layer {layer} rows have "
                            f"{tuple(block.shape)} {block.dtype}; expected "
                            f"{expected} float32"
                        )
                    # Both file.write and hashlib accept the buffer protocol;
                    # keep the stripe tensor alive and avoid allocating a
                    # second payload-sized ``bytes`` copy.
                    payload = memoryview(cast(Any, block.numpy())).cast("B")
                    f.write(payload)
                    digest.update(payload)
            if durable:
                os.fsync(f.fileno())
    except BaseException:
        with suppress(FileNotFoundError):
            tmp.unlink()
        raise
    os.replace(tmp, path)
    return digest.hexdigest()


def _lens_safetensors_header(
    layers: list[int], d_model: int,
) -> tuple[bytes, bytes]:
    """Canonical writer header bytes, shared by save and load-time digest."""
    bytes_per_layer = int(d_model) * int(d_model) * 4
    offset = 0
    header: dict[str, dict[str, object]] = {}
    for layer in sorted(layers):
        header[f"layer_{layer}"] = {
            "dtype": "F32",
            "shape": [int(d_model), int(d_model)],
            "data_offsets": [offset, offset + bytes_per_layer],
        }
        offset += bytes_per_layer
    raw_header = json.dumps(
        header, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    padded_len = (len(raw_header) + 7) // 8 * 8
    raw_header += b" " * (padded_len - len(raw_header))
    return struct.pack("<Q", padded_len), raw_header


def load_local_lens(model_id: str) -> tuple[JacobianLens, dict[str, Any]] | None:
    """Load the Saklas-owned local/default lens when usable.

    Self-healing like the neutral-activation cache: a wrong format version,
    non-finite tensors, or any parse failure logs a warning and reads as
    "no lens" (the caller decides whether to error or re-fit) rather than
    crashing the session.
    """
    loaded = _load_lens_verified(model_id)
    return None if loaded is None else (loaded[0], loaded[1])


def load_lens(model_id: str) -> tuple[JacobianLens, dict[str, Any]] | None:
    """Load the active local or externally managed lens source."""
    from saklas.io.lens_sources import (
        load_active_lens_source,
        load_external_lens,
    )

    active = load_active_lens_source(model_id)
    if active is None:
        return None
    if active["kind"] == "huggingface":
        return load_external_lens(model_id, active["name"])
    return load_local_lens(model_id)


def _load_lens_verified(
    model_id: str,
    *,
    requested_layers: set[int] | None = None,
) -> tuple[
    JacobianLens, dict[str, Any], _LensPayloadProof | None,
] | None:
    """Load a durable lens plus a proof reusable under the fit transaction."""
    anchor, sc_path = _lens_anchor_paths(model_id)
    with lens_fit_lock(model_id), artifact_lock(anchor):
        sidecar = _load_sidecar_at(
            model_id, anchor, sc_path, label="jlens cache",
        )
        loaded = _load_lens_at(
            model_id, anchor, sc_path, sidecar, label="jlens cache",
            requested_layers=requested_layers,
        )
        if loaded is None:
            return None
        lens, verified_sidecar = loaded
        _cleanup_unreferenced_generations(anchor.parent)
        durable_layers = {
            int(layer) for layer in verified_sidecar["source_layers"]
        }
        fully_verified = (
            requested_layers is None or requested_layers >= durable_layers
        )
        return lens, verified_sidecar, (
            _lens_payload_proof(anchor, verified_sidecar)
            if fully_verified else None
        )


def load_lens_checkpoint(model_id: str) -> tuple[JacobianLens, dict[str, Any]] | None:
    """Load a resumable partial lens shard, or ``None`` when absent/unusable."""
    anchor, sc_path = _checkpoint_anchor_paths(model_id)
    with lens_fit_lock(model_id), artifact_lock(anchor):
        sidecar = _load_sidecar_at(
            model_id, anchor, sc_path, label="jlens checkpoint",
        )
        loaded = _load_lens_at(
            model_id, anchor, sc_path, sidecar, label="jlens checkpoint",
        )
        if loaded is not None:
            _cleanup_unreferenced_generations(anchor.parent)
        return loaded


def promote_lens_checkpoint(
    model_id: str,
    *,
    n_prompts: int,
    source_layers: list[int],
    corpus_sha256: str,
    corpus_hash_kind: str,
    seq_len: int,
    d_model: int,
    model_fingerprint: str,
    _verified_proof: _LensPayloadProof | None = None,
) -> bool:
    """Promote a complete checkpoint to the durable lens without rewriting it.

    Terminal checkpoint cadence can coincide with successful fit completion.
    That shard is already the complete fp32 artifact, including any resumed
    base, so serializing the fp32 in-memory lens again would duplicate a
    potentially multi-GiB write. Promote only after cheap exact
    identity checks; callers fall back to :func:`save_lens` on ``False``.
    """
    checkpoint_anchor, checkpoint_sc = _checkpoint_anchor_paths(model_id)
    final_anchor, final_sc = _lens_anchor_paths(model_id)
    with lens_fit_lock(model_id), artifact_lock(checkpoint_anchor), artifact_lock(final_anchor):
        sidecar = _load_sidecar_at(
            model_id, checkpoint_anchor, checkpoint_sc, label="jlens checkpoint",
        )
        if (
            sidecar is None
            or int(sidecar.get("format_version", 0)) != LENS_FORMAT_VERSION
            or int(sidecar.get("n_prompts", -1)) != int(n_prompts)
            or [int(layer) for layer in sidecar.get("source_layers", [])]
            != sorted(int(layer) for layer in source_layers)
            or sidecar.get("corpus_sha256") != corpus_sha256
            or sidecar.get("corpus_hash_kind") != corpus_hash_kind
            or int(sidecar.get("seq_len", -1)) != int(seq_len)
            or int(sidecar.get("d_model", -1)) != int(d_model)
            or sidecar.get("model_fingerprint") != model_fingerprint
            or int(sidecar.get("base_n_prompts", -1)) != 0
        ):
            return False
        if not _proof_matches(
            _verified_proof, checkpoint_anchor, sidecar,
        ) and not lens_payloads_match(model_id, sidecar, checkpoint=True):
            return False
        checkpoint_tensors = set(_sidecar_tensor_paths(
            checkpoint_anchor, sidecar,
            [int(layer) for layer in sidecar["source_layers"]],
        ).values())
        # Re-fsync the checkpoint payloads before publishing the final pointer.
        for checkpoint_ts in checkpoint_tensors:
            with open(checkpoint_ts, "rb") as handle:
                os.fsync(handle.fileno())
        # The final pointer must follow both file and directory durability.
        fsync_directory(final_sc.parent)
        durable_sidecar = dict(sidecar)
        durable_sidecar.update({
            "checkpoint": False,
            "base_n_prompts": None,
            "partial_n_prompts": None,
            "consumed_prefix_sha256": None,
        })
        # Publish only the pointer. The immutable tensor stays referenced by the
        # checkpoint until this succeeds, so any sidecar failure preserves both
        # the prior durable lens and the complete resumable checkpoint.
        write_json_atomic(final_sc, durable_sidecar)
        fsync_directory(final_sc.parent)
        checkpoint_sc.unlink(missing_ok=True)
        fsync_directory(final_sc.parent)
        _cleanup_unreferenced_generations(final_anchor.parent)
        fsync_directory(final_sc.parent)
        from saklas.io.lens_sources import set_active_local_lens

        set_active_local_lens(model_id)
        return True


def _load_lens_at(
    model_id: str,
    tensor_anchor: Path,
    _sc_path: Path,
    sidecar: dict[str, Any] | None,
    *,
    label: str,
    requested_layers: set[int] | None = None,
) -> tuple[JacobianLens, dict[str, Any]] | None:
    if sidecar is None:
        return None
    try:
        d_model = int(sidecar["d_model"])
        durable_layers = [int(layer) for layer in sidecar["source_layers"]]
        if (
            requested_layers is not None
            and requested_layers - set(durable_layers)
        ):
            raise ValueError("requested layers are absent from lens")
        source_layers = (
            durable_layers if requested_layers is None
            else [layer for layer in durable_layers if layer in requested_layers]
        )
        if not source_layers:
            raise ValueError("requested layers are absent from lens")
        jacobians: dict[int, torch.Tensor] = {}
        tensor_paths = _sidecar_tensor_paths(
            tensor_anchor, sidecar, source_layers,
            require_exact_keys=(source_layers == durable_layers),
        )
        expected_digests = sidecar.get("tensor_sha256")
        if not isinstance(expected_digests, Mapping):
            log.warning(
                "%s for %s has no tensor digest map; ignoring — re-fit with "
                "`saklas lens fit`", label, model_id,
            )
            return None

        groups: list[tuple[Path, list[int], str]] = []
        for layer in source_layers:
            expected = expected_digests.get(str(layer))
            if not isinstance(expected, str) or len(expected) != 64:
                raise ValueError(f"layer {layer} has no valid tensor digest")
            groups.append((tensor_paths[layer], [layer], expected))

        for ts_path, tensor_layers, expected_digest in groups:
            prefix, raw_header = _lens_safetensors_header(
                tensor_layers, d_model,
            )
            digest = hashlib.sha256(prefix)
            digest.update(raw_header)
            with safe_open(str(ts_path), framework="pt", device="cpu") as tensors:
                actual_layers = sorted(
                    int(k.split("_", 1)[1]) for k in tensors.keys()
                )
                if actual_layers != tensor_layers:
                    log.warning(
                        "%s for %s has tensor layers %s but sidecar declares %s; "
                        "ignoring — re-fit with `saklas lens fit`",
                        label, model_id, actual_layers, tensor_layers,
                    )
                    return None
                for layer in tensor_layers:
                    raw = tensors.get_tensor(f"layer_{layer}")
                    if (
                        raw.dtype != torch.float32 or raw.ndim != 2
                        or tuple(raw.shape) != (d_model, d_model)
                    ):
                        log.warning(
                            "%s for %s layer %d has shape %s (need %dx%d); "
                            "ignoring — re-fit with `saklas lens fit`",
                            label, model_id, layer, tuple(raw.shape),
                            d_model, d_model,
                        )
                        return None
                    for start in range(0, d_model, 256):
                        block = raw[start:start + 256].contiguous()
                        digest.update(
                            memoryview(cast(Any, block.numpy())).cast("B"),
                        )
                    J = raw.to(torch.float32)
                    if not bool(torch.isfinite(J).all()):
                        log.warning(
                            "%s for %s contains non-finite values; ignoring — "
                            "re-fit with `saklas lens fit`", label, model_id,
                        )
                        return None
                    jacobians[layer] = J
            if digest.hexdigest() != expected_digest:
                log.warning(
                    "%s for %s failed tensor digest validation; ignoring — "
                    "re-fit with `saklas lens fit`", label, model_id,
                )
                return None
        lens = JacobianLens(
            jacobians,
            n_prompts=int(sidecar.get("n_prompts", 0)),
            d_model=d_model,
        )
        return lens, sidecar
    except Exception as exc:
        log.warning("Corrupt %s for %s; ignoring: %s", label, model_id, exc)
        return None


def _final_lens_subsumes_checkpoint(
    final: Mapping[str, Any], checkpoint: Mapping[str, Any],
) -> bool:
    """Whether the durable pointer contains every result in ``checkpoint``."""
    if checkpoint.get("checkpoint") is not True:
        return False
    semantic_keys = (
        "corpus_sha256",
        "corpus_hash_kind",
        "seq_len",
        "d_model",
        "model_fingerprint",
        "estimator_policy",
    )
    if any(final.get(key) != checkpoint.get(key) for key in semantic_keys):
        return False
    try:
        final_layers = {int(layer) for layer in final.get("source_layers", [])}
        checkpoint_layers = {
            int(layer) for layer in checkpoint.get("source_layers", [])
        }
        final_progress = int(final.get("n_prompts", -1))
        checkpoint_progress = int(checkpoint.get("base_n_prompts", 0)) + int(
            checkpoint.get("n_prompts", -1),
        )
    except (TypeError, ValueError):
        return False
    return bool(
        checkpoint_layers
        and final_layers >= checkpoint_layers
        and final_progress >= checkpoint_progress > 0
    )


def remove_subsumed_lens_checkpoint(
    model_id: str,
    *,
    verified_final_sidecar: Mapping[str, Any] | None = None,
) -> bool:
    """Reap a crash-left checkpoint only when the durable lens proves it redundant.

    Final publication and checkpoint unlink are necessarily two filesystem
    operations. A process death between them leaves two complete shard sets.
    This recovery path preserves a farther-ahead or semantically different
    checkpoint and validates the final payload before discarding the recovery
    point. Callers that just validated the exact current final sidecar may pass it
    to avoid hashing a multi-GiB lens twice.
    """
    final_anchor, final_sc = _lens_anchor_paths(model_id)
    checkpoint_anchor, checkpoint_sc = _checkpoint_anchor_paths(model_id)
    with (
        lens_fit_lock(model_id),
        artifact_lock(final_anchor),
        artifact_lock(checkpoint_anchor),
    ):
        checkpoint = _load_sidecar_at(
            model_id, checkpoint_anchor, checkpoint_sc,
            label="jlens checkpoint",
        )
        if checkpoint is None:
            return False
        final = _load_sidecar_at(
            model_id, final_anchor, final_sc, label="jlens cache",
        )
        if final is None or not _final_lens_subsumes_checkpoint(final, checkpoint):
            return False
        caller_verified_current = bool(
            verified_final_sidecar is not None
            and dict(verified_final_sidecar) == final
        )
        if not caller_verified_current and not lens_payloads_match(model_id, final):
            return False
        checkpoint_sc.unlink(missing_ok=True)
        # The sidecar is the atomic pointer.  Make its removal durable before
        # deleting any shard it could name, otherwise a crash can resurrect a
        # checkpoint pointer whose payload has already been collected.
        fsync_directory(final_anchor.parent)
        _cleanup_unreferenced_generations(final_anchor.parent)
        fsync_directory(final_anchor.parent)
        return True


def remove_lens_checkpoint(model_id: str) -> bool:
    """Delete a resumable checkpoint shard. Returns True when anything was removed."""
    anchor, sc_path = _checkpoint_anchor_paths(model_id)
    with lens_fit_lock(model_id), artifact_lock(anchor):
        before = set(anchor.parent.glob("jlens*.safetensors"))
        removed = sc_path.exists()
        sc_path.unlink(missing_ok=True)
        fsync_directory(anchor.parent)
        _cleanup_unreferenced_generations(anchor.parent)
        fsync_directory(anchor.parent)
        after = set(anchor.parent.glob("jlens*.safetensors"))
        return removed or before != after


def remove_lens(model_id: str) -> bool:
    """Delete a model's lens artifact. Returns True when anything was removed."""
    final_anchor, final_sc = _lens_anchor_paths(model_id)
    checkpoint_anchor, checkpoint_sc = _checkpoint_anchor_paths(model_id)
    with (
        lens_fit_lock(model_id),
        artifact_lock(final_anchor),
        artifact_lock(checkpoint_anchor),
    ):
        tensor_candidates = (
            list(final_anchor.parent.glob("jlens*.gen-*.safetensors"))
            + list(final_anchor.parent.glob("jlens*.gen-*.safetensors.tmp"))
            + [
                final_anchor,
                checkpoint_anchor,
                final_anchor.with_suffix(final_anchor.suffix + ".tmp"),
                checkpoint_anchor.with_suffix(checkpoint_anchor.suffix + ".tmp"),
            ]
        )
        candidates = [*tensor_candidates, final_sc, checkpoint_sc]
        removed = any(path.exists() for path in candidates)
        # Remove the atomic pointers first: after this logical deletion, a
        # best-effort tensor GC failure cannot expose a partially deleted lens.
        final_sc.unlink(missing_ok=True)
        checkpoint_sc.unlink(missing_ok=True)
        fsync_directory(final_anchor.parent)
        for path in tensor_candidates:
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                log.warning("could not remove J-lens tensor %s: %s", path, exc)
        fsync_directory(final_anchor.parent)
        from saklas.io.lens_sources import lens_active_path, load_active_lens_source

        active = load_active_lens_source(model_id)
        if active is not None and active["kind"] == "local":
            lens_active_path(model_id).unlink(missing_ok=True)
        return removed


#: Default web-text corpus for a lens fit (repo, config) — the paper-parity
#: pretraining-like sample.  Shared by CLI ``lens fit`` and the server's
#: ``POST .../lens/fit`` route.
DEFAULT_LENS_CORPUS = ("HuggingFaceFW/fineweb-edu", "sample-10BT")
#: Documents are sliced to this many characters before tokenization — the fit
#: truncates to ``seq_len`` tokens anyway, so tokenizing a full web page is
#: waste.
LENS_DOC_CHARS = 4000


def resolved_default_lens_corpus_spec() -> tuple[str, str]:
    """Return the default dataset's immutable Hub revision and corpus spec."""
    from saklas.core.jlens import JacobianLensError

    repo, config = DEFAULT_LENS_CORPUS
    try:
        from huggingface_hub import HfApi

        revision = HfApi().dataset_info(repo).sha
    except Exception as exc:
        raise JacobianLensError(
            f"could not resolve an immutable revision for {repo}: {exc}"
        ) from exc
    if not revision:
        raise JacobianLensError(
            f"Hugging Face returned no immutable dataset revision for {repo}"
        )
    return (
        str(revision),
        f"hf:{repo}/{config}@{revision};preprocess="
        f"{LENS_CORPUS_PREPROCESS_VERSION};doc_chars={LENS_DOC_CHARS}",
    )


def stream_default_lens_corpus(n: int) -> tuple[list[str], str]:
    """Stream ``n`` documents from the default web-text corpus.

    Returns ``(documents, corpus_spec)``.  Needs the optional ``datasets``
    dependency (``pip install 'saklas[hf]'``); raises
    :class:`~saklas.core.jlens.JacobianLensError` without it so both the CLI
    and the fit route surface the same actionable message.
    """
    from saklas.core.jlens import JacobianLensError

    try:
        from datasets import load_dataset
    except ImportError as e:
        raise JacobianLensError(
            "the default lens corpus streams via the optional `datasets` "
            "dependency — `pip install 'saklas[hf]'`, or supply a corpus "
            "file (one document per line)"
        ) from e
    repo, config = DEFAULT_LENS_CORPUS
    revision, corpus_spec = resolved_default_lens_corpus_spec()
    stream = load_dataset(
        repo, name=config, split="train", streaming=True, revision=revision,
    )
    docs: list[str] = []
    for row in stream:
        text = str(row.get("text", "")).strip()
        if text:
            docs.append(text[:LENS_DOC_CHARS])
        if len(docs) >= n:
            break
    return docs, corpus_spec
