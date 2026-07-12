"""Per-model Jacobian-lens artifact: save/load under ``models/<safe_id>/``.

The lens is a per-model transport (one ``J_l`` matrix per source layer), not a
per-concept artifact, so it lives next to the neutral-activation cache rather
than under ``manifolds/``:

    ~/.saklas/models/<safe_model_id>/jlens.safetensors   # layer_<idx>, fp16
    ~/.saklas/models/<safe_model_id>/jlens.json          # sidecar

fp16 on disk (the reference-repo convention — J entries are O(1), so range is
no constraint and fp16's extra mantissa bits beat bf16; this deliberately
differs from the neutral cache's fp32 invariant, which exists because that
cache feeds a covariance inversion). Promoted to fp32 on load.

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
from contextlib import suppress
from pathlib import Path
from collections.abc import Callable, Mapping
from typing import Any, cast

import torch
from safetensors import safe_open

from saklas.core.jlens import JacobianLens
from saklas.io.atomic import artifact_lock, write_json_atomic
from saklas.io.paths import model_dir

log = logging.getLogger(__name__)

LENS_FORMAT_VERSION = 3
_LENS_NAME = "jlens"
_LENS_CHECKPOINT_NAME = "jlens.partial"
_LENS_METHOD = "jlens_cotangent_sum"
LENS_CORPUS_PREPROCESS_VERSION = 1


def lens_paths(model_id: str) -> tuple[Path, Path]:
    """Return ``(safetensors_path, sidecar_path)`` for a model's lens."""
    md = model_dir(model_id)
    return md / f"{_LENS_NAME}.safetensors", md / f"{_LENS_NAME}.json"


def lens_checkpoint_paths(model_id: str) -> tuple[Path, Path]:
    """Return ``(safetensors_path, sidecar_path)`` for the resumable checkpoint."""
    md = model_dir(model_id)
    return (
        md / f"{_LENS_CHECKPOINT_NAME}.safetensors",
        md / f"{_LENS_CHECKPOINT_NAME}.json",
    )


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
    model_id: str, ts_path: Path, sc_path: Path, *, label: str,
) -> dict[str, Any] | None:
    if not (ts_path.exists() and sc_path.exists()):
        return None
    try:
        with open(sc_path) as f:
            sidecar = json.load(f)
        version = sidecar.get("format_version")
        if version != LENS_FORMAT_VERSION:
            log.warning(
                "%s for %s has format_version %r (need %d); ignoring "
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
        d_model = int(sidecar.get("d_model", 0) or 0)
        source_layers_raw = sidecar.get("source_layers")
        if (
            d_model <= 0
            or not isinstance(source_layers_raw, list)
            or not source_layers_raw
        ):
            log.warning(
                "%s for %s has invalid sidecar shape metadata; ignoring "
                "— re-fit with `saklas lens fit`", label, model_id,
            )
            return None
        # Normalize the layer values for metadata-only callers, matching
        # ``JacobianLens.source_layers`` without touching the safetensors file.
        sidecar = dict(sidecar)
        sidecar["source_layers"] = sorted(int(layer) for layer in source_layers_raw)
        sidecar["d_model"] = d_model
        expected_keys = [f"layer_{layer}" for layer in sidecar["source_layers"]]
        with safe_open(str(ts_path), framework="pt", device="cpu") as tensors:
            if sorted(tensors.keys()) != sorted(expected_keys):
                raise ValueError("tensor layer keys do not match sidecar")
            for key in expected_keys:
                view = tensors.get_slice(key)
                if tuple(view.get_shape()) != (d_model, d_model):
                    raise ValueError(f"{key} shape does not match sidecar")
                if view.get_dtype() != "F16":
                    raise ValueError(f"{key} is not float16")
        return sidecar
    except Exception as exc:
        log.warning("Corrupt %s sidecar for %s; ignoring: %s", label, model_id, exc)
        return None


def load_lens_sidecar(model_id: str) -> dict[str, Any] | None:
    """Load validated lens metadata without loading the tensor artifact."""
    ts_path, sc_path = lens_paths(model_id)
    with artifact_lock(ts_path):
        return _load_sidecar_at(model_id, ts_path, sc_path, label="jlens cache")


def load_lens_checkpoint_sidecar(model_id: str) -> dict[str, Any] | None:
    """Load validated checkpoint metadata without materializing its matrices."""
    ts_path, sc_path = lens_checkpoint_paths(model_id)
    with artifact_lock(ts_path):
        return _load_sidecar_at(
            model_id, ts_path, sc_path, label="jlens checkpoint",
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
) -> Path:
    """Persist a fitted lens (fp16 tensors + atomic JSON sidecar)."""
    ts_path, sc_path = lens_paths(model_id)
    _save_lens_at(
        lens, ts_path, sc_path,
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
    )
    return ts_path


def save_lens_checkpoint(
    partial: JacobianLens,
    model_id: str,
    *,
    base_n_prompts: int,
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
) -> Path:
    """Persist a resumable partial shard without rewriting the full lens."""
    ts_path, sc_path = lens_checkpoint_paths(model_id)
    _save_lens_at(
        partial, ts_path, sc_path,
        corpus_spec=corpus_spec,
        corpus_sha256=corpus_sha256,
        seq_len=seq_len,
        dim_batch=dim_batch,
        skip_first=skip_first,
        corpus_hash_kind=corpus_hash_kind,
        durable=False,
        raw_corpus_sha256=raw_corpus_sha256,
        raw_prompt_count=raw_prompt_count,
        usable_prompt_count=usable_prompt_count,
        model_layer_count=model_layer_count,
        model_fingerprint=model_fingerprint,
        model_source_fingerprint=model_source_fingerprint,
        extra_sidecar={
            "checkpoint": True,
            "base_n_prompts": int(base_n_prompts),
            "partial_n_prompts": partial.n_prompts,
        },
    )
    return ts_path


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
) -> Path:
    """Write a self-contained checkpoint directly from raw estimator sums.

    Normalization and optional prefix merging happen one layer at a time while
    converting to fp16, avoiding the extra full-fp32 ``JacobianLens`` that the
    compatibility checkpoint callback requires.  The resulting shard always has
    ``base_n_prompts=0`` and can therefore survive any number of interruptions
    without depending on a separate full artifact.
    """
    ts_path, sc_path = lens_checkpoint_paths(model_id)
    total_prompts = int(n_prompts) + (base.n_prompts if base is not None else 0)
    with artifact_lock(ts_path):
        _save_lens_components(
            sums, total_prompts, d_model, ts_path, sc_path,
            corpus_spec=corpus_spec,
            corpus_sha256=corpus_sha256,
            seq_len=seq_len,
            dim_batch=dim_batch,
            skip_first=skip_first,
            corpus_hash_kind=corpus_hash_kind,
            durable=False,
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
            },
        )
    return ts_path


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
    extra_sidecar: dict[str, Any] | None = None,
) -> None:
    with artifact_lock(ts_path):
        _save_lens_components(
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
    extra_sidecar: dict[str, Any] | None = None,
) -> None:
    ts_path.parent.mkdir(parents=True, exist_ok=True)
    layer_ids = sorted(int(idx) for idx in jacobians)

    def _rows(idx: int, start: int, end: int) -> torch.Tensor:
        value = jacobians[idx][start:end].to(device="cpu", dtype=torch.float32)
        if raw_sum_count is not None:
            # Stripe-local arithmetic: checkpointing never clones a complete
            # fp32 layer or keeps a complete fp16 artifact mapping alive.
            value = value.clone()
            if average_base is not None:
                value.add_(
                    average_base.jacobians[idx][start:end],
                    alpha=average_base.n_prompts,
                )
            value.mul_(1.0 / max(int(n_prompts), 1))
        return value.to(torch.float16).contiguous()

    tensor_sha256 = _save_fp16_square_safetensors_atomic(
        ts_path, layer_ids, d_model, _rows, durable=durable,
    )
    sidecar: dict[str, Any] = {
        "format_version": LENS_FORMAT_VERSION,
        "method": _LENS_METHOD,
        "n_prompts": int(n_prompts),
        "d_model": int(d_model),
        "source_layers": sorted(int(layer) for layer in jacobians),
        "dtype": "float16",
        "corpus_spec": corpus_spec,
        "corpus_sha256": corpus_sha256,
        "corpus_hash_kind": corpus_hash_kind,
        "seq_len": seq_len,
        "dim_batch": dim_batch,
        "skip_first_positions": skip_first,
        "estimator_policy": lens_estimator_policy(skip_first=skip_first),
        "tensor_sha256": tensor_sha256,
    }
    if raw_corpus_sha256 is not None:
        sidecar["raw_corpus_sha256"] = raw_corpus_sha256
    if raw_prompt_count is not None:
        sidecar["raw_prompt_count"] = int(raw_prompt_count)
    if usable_prompt_count is not None:
        sidecar["usable_prompt_count"] = int(usable_prompt_count)
    if model_layer_count is not None:
        sidecar["model_layer_count"] = int(model_layer_count)
    if model_fingerprint is not None:
        sidecar["model_fingerprint"] = model_fingerprint
    if model_source_fingerprint is not None:
        sidecar["model_source_fingerprint"] = model_source_fingerprint
    if extra_sidecar:
        sidecar.update(extra_sidecar)
    write_json_atomic(sc_path, sidecar)


def _save_fp16_square_safetensors_atomic(
    path: Path,
    layers: list[int],
    d_model: int,
    rows: Callable[[int, int, int], torch.Tensor],
    *,
    durable: bool = True,
) -> str:
    """Stream square fp16 layer tensors into one atomic safetensors file.

    ``safetensors.torch.save_file`` requires a complete tensor mapping, which
    made checkpoint RSS include every fp16 layer alongside the fp32 estimator.
    The wire format is deliberately simple: an 8-byte little-endian JSON-header
    length, a space-padded JSON header, then contiguous tensor payloads.  Writing
    256-row fp16 stripes preserves the public monolithic artifact without the
    complete in-memory snapshot.
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
                    if block.dtype != torch.float16 or tuple(block.shape) != expected:
                        raise ValueError(
                            f"streamed layer {layer} rows have "
                            f"{tuple(block.shape)} {block.dtype}; expected "
                            f"{expected} float16"
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
    bytes_per_layer = int(d_model) * int(d_model) * 2
    offset = 0
    header: dict[str, dict[str, object]] = {}
    for layer in sorted(layers):
        header[f"layer_{layer}"] = {
            "dtype": "F16",
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


def load_lens(model_id: str) -> tuple[JacobianLens, dict[str, Any]] | None:
    """Load a model's fitted lens, or ``None`` when absent or unusable.

    Self-healing like the neutral-activation cache: a wrong format version,
    non-finite tensors, or any parse failure logs a warning and reads as
    "no lens" (the caller decides whether to error or re-fit) rather than
    crashing the session.
    """
    ts_path, sc_path = lens_paths(model_id)
    with artifact_lock(ts_path):
        sidecar = _load_sidecar_at(
            model_id, ts_path, sc_path, label="jlens cache",
        )
        return _load_lens_at(
            model_id, ts_path, sc_path, sidecar, label="jlens cache",
        )


def load_lens_checkpoint(model_id: str) -> tuple[JacobianLens, dict[str, Any]] | None:
    """Load a resumable partial lens shard, or ``None`` when absent/unusable."""
    ts_path, sc_path = lens_checkpoint_paths(model_id)
    with artifact_lock(ts_path):
        sidecar = _load_sidecar_at(
            model_id, ts_path, sc_path, label="jlens checkpoint",
        )
        return _load_lens_at(
            model_id, ts_path, sc_path, sidecar, label="jlens checkpoint",
        )


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
) -> bool:
    """Promote a complete checkpoint to the durable lens without rewriting it.

    Terminal checkpoint cadence can coincide with successful fit completion.
    That shard is already the complete fp16 artifact, including any resumed
    base, so serializing the fp32 in-memory lens again would duplicate a
    potentially multi-GiB conversion and write.  Promote only after cheap exact
    identity checks; callers fall back to :func:`save_lens` on ``False``.
    """
    checkpoint_ts, checkpoint_sc = lens_checkpoint_paths(model_id)
    final_ts, final_sc = lens_paths(model_id)
    with artifact_lock(checkpoint_ts), artifact_lock(final_ts):
        sidecar = _load_sidecar_at(
            model_id, checkpoint_ts, checkpoint_sc, label="jlens checkpoint",
        )
        if (
            sidecar is None
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
        # The checkpoint writer deliberately skips fsync because most shards
        # are temporary. A promoted terminal shard becomes durable here.
        with open(checkpoint_ts, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(checkpoint_ts, final_ts)
        durable_sidecar = dict(sidecar)
        for key in ("checkpoint", "base_n_prompts", "partial_n_prompts"):
            durable_sidecar.pop(key, None)
        write_json_atomic(final_sc, durable_sidecar)
        checkpoint_sc.unlink(missing_ok=True)
        return True


def _load_lens_at(
    model_id: str,
    ts_path: Path,
    _sc_path: Path,
    sidecar: dict[str, Any] | None,
    *,
    label: str,
) -> tuple[JacobianLens, dict[str, Any]] | None:
    if sidecar is None:
        return None
    try:
        d_model = int(sidecar["d_model"])
        source_layers = [int(layer) for layer in sidecar["source_layers"]]
        expected_digest = sidecar.get("tensor_sha256")
        if not isinstance(expected_digest, str) or len(expected_digest) != 64:
            log.warning(
                "%s for %s has no tensor digest; ignoring — re-fit with "
                "`saklas lens fit`", label, model_id,
            )
            return None
        prefix, raw_header = _lens_safetensors_header(source_layers, d_model)
        digest = hashlib.sha256(prefix)
        digest.update(raw_header)
        jacobians: dict[int, torch.Tensor] = {}
        with safe_open(str(ts_path), framework="pt", device="cpu") as tensors:
            tensor_layers = sorted(
                int(k.split("_", 1)[1]) for k in tensors.keys()
            )
            if tensor_layers != source_layers:
                log.warning(
                    "%s for %s has tensor layers %s but sidecar declares %s; "
                    "ignoring — re-fit with `saklas lens fit`",
                    label, model_id, tensor_layers, source_layers,
                )
                return None
            # Convert and validate one layer at a time.  ``load_file`` kept the
            # complete fp16 mapping alive while also materializing the complete
            # fp32 lens, producing a needless ~1.5x artifact peak on resume.
            for layer in tensor_layers:
                raw = tensors.get_tensor(f"layer_{layer}")
                if (
                    raw.dtype != torch.float16 or raw.ndim != 2
                    or tuple(raw.shape) != (d_model, d_model)
                ):
                    log.warning(
                        "%s for %s layer %d has shape %s (need %dx%d); "
                        "ignoring — re-fit with `saklas lens fit`",
                        label, model_id, layer, tuple(raw.shape), d_model, d_model,
                    )
                    return None
                for start in range(0, d_model, 256):
                    block = raw[start:start + 256].contiguous()
                    digest.update(memoryview(cast(Any, block.numpy())).cast("B"))
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


def remove_lens_checkpoint(model_id: str) -> bool:
    """Delete a resumable checkpoint shard. Returns True when anything was removed."""
    ts_path, sc_path = lens_checkpoint_paths(model_id)
    with artifact_lock(ts_path):
        removed = False
        for path in (ts_path, sc_path):
            if path.exists():
                path.unlink()
                removed = True
        return removed


def remove_lens(model_id: str) -> bool:
    """Delete a model's lens artifact. Returns True when anything was removed."""
    removed = False
    for ts_path, sc_path in (lens_paths(model_id), lens_checkpoint_paths(model_id)):
        with artifact_lock(ts_path):
            for path in (ts_path, sc_path):
                if path.exists():
                    path.unlink()
                    removed = True
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
