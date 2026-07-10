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

The sidecar records the corpus spec + sha256 so a re-fit against a different
corpus reads as stale, and ``n_prompts`` so an interrupted fit can resume
(load → fit the remaining prompts → ``JacobianLens.merge`` → save).
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import suppress
from pathlib import Path
from collections.abc import Mapping
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from saklas.core.jlens import JacobianLens
from saklas.io.atomic import write_json_atomic
from saklas.io.paths import model_dir

log = logging.getLogger(__name__)

LENS_FORMAT_VERSION = 1
_LENS_NAME = "jlens"
_LENS_CHECKPOINT_NAME = "jlens.partial"
_LENS_METHOD = "jlens_cotangent_sum"


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
        return sidecar
    except Exception as exc:
        log.warning("Corrupt %s sidecar for %s; ignoring: %s", label, model_id, exc)
        return None


def load_lens_sidecar(model_id: str) -> dict[str, Any] | None:
    """Load validated lens metadata without loading the tensor artifact."""
    ts_path, sc_path = lens_paths(model_id)
    return _load_sidecar_at(model_id, ts_path, sc_path, label="jlens cache")


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
    extra_sidecar: dict[str, Any] | None = None,
) -> None:
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
    raw_sum_count: int | None = None,
    average_base: JacobianLens | None = None,
    extra_sidecar: dict[str, Any] | None = None,
) -> None:
    ts_path.parent.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, torch.Tensor] = {}
    for idx, J in jacobians.items():
        value = J
        if raw_sum_count is not None:
            value = J.clone()
            if average_base is not None:
                value.add_(
                    average_base.jacobians[idx], alpha=average_base.n_prompts,
                )
            value.mul_(1.0 / max(int(n_prompts), 1))
        tensors[f"layer_{idx}"] = value.contiguous().to(torch.float16).cpu()
    _save_safetensors_atomic(ts_path, tensors, durable=durable)
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
    }
    if raw_corpus_sha256 is not None:
        sidecar["raw_corpus_sha256"] = raw_corpus_sha256
    if raw_prompt_count is not None:
        sidecar["raw_prompt_count"] = int(raw_prompt_count)
    if usable_prompt_count is not None:
        sidecar["usable_prompt_count"] = int(usable_prompt_count)
    if model_layer_count is not None:
        sidecar["model_layer_count"] = int(model_layer_count)
    if extra_sidecar:
        sidecar.update(extra_sidecar)
    write_json_atomic(sc_path, sidecar)


def _save_safetensors_atomic(
    path: Path, tensors: dict[str, torch.Tensor], *, durable: bool = True,
) -> None:
    """Atomically replace a safetensors artifact, preserving the prior file on error."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp") if path.suffix else path.with_name(
        path.name + ".tmp"
    )
    try:
        save_file(tensors, str(tmp))
        if durable:
            fd = os.open(tmp, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    except BaseException:
        with suppress(FileNotFoundError):
            tmp.unlink()
        raise
    os.replace(tmp, path)


def load_lens(model_id: str) -> tuple[JacobianLens, dict[str, Any]] | None:
    """Load a model's fitted lens, or ``None`` when absent or unusable.

    Self-healing like the neutral-activation cache: a wrong format version,
    non-finite tensors, or any parse failure logs a warning and reads as
    "no lens" (the caller decides whether to error or re-fit) rather than
    crashing the session.
    """
    ts_path, sc_path = lens_paths(model_id)
    sidecar = load_lens_sidecar(model_id)
    return _load_lens_at(model_id, ts_path, sc_path, sidecar, label="jlens cache")


def load_lens_checkpoint(model_id: str) -> tuple[JacobianLens, dict[str, Any]] | None:
    """Load a resumable partial lens shard, or ``None`` when absent/unusable."""
    ts_path, sc_path = lens_checkpoint_paths(model_id)
    sidecar = _load_sidecar_at(model_id, ts_path, sc_path, label="jlens checkpoint")
    return _load_lens_at(model_id, ts_path, sc_path, sidecar, label="jlens checkpoint")


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
                J = tensors.get_tensor(f"layer_{layer}").to(torch.float32)
                if J.ndim != 2 or tuple(J.shape) != (d_model, d_model):
                    log.warning(
                        "%s for %s layer %d has shape %s (need %dx%d); "
                        "ignoring — re-fit with `saklas lens fit`",
                        label, model_id, layer, tuple(J.shape), d_model, d_model,
                    )
                    return None
                if not bool(torch.isfinite(J).all()):
                    log.warning(
                        "%s for %s contains non-finite values; ignoring — "
                        "re-fit with `saklas lens fit`", label, model_id,
                    )
                    return None
                jacobians[layer] = J
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
    removed = False
    for path in lens_checkpoint_paths(model_id):
        if path.exists():
            path.unlink()
            removed = True
    return removed


def remove_lens(model_id: str) -> bool:
    """Delete a model's lens artifact. Returns True when anything was removed."""
    removed = False
    for path in (*lens_paths(model_id), *lens_checkpoint_paths(model_id)):
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
    stream = load_dataset(repo, name=config, split="train", streaming=True)
    docs: list[str] = []
    for row in stream:
        text = str(row.get("text", "")).strip()
        if text:
            docs.append(text[:LENS_DOC_CHARS])
        if len(docs) >= n:
            break
    return docs, f"hf:{repo}/{config}"
