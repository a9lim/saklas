"""Cross-model probe alignment via per-layer Procrustes (v1.6).

Probes are extracted per (model, concept).  A probe extracted on one
model isn't directly usable on a different model — different tokenizers,
different hidden dims, different basis rotation in the residual stream.
This module fits a per-layer linear map between two models' neutral
activations (``Procrustes`` for matched dim, low-rank PCA-and-lift for
mismatched dim) and uses it to transfer a probe's per-layer baked
direction from source-space to target-space.

Public surface:

* :func:`load_or_compute_neutral_activations` — disk-cached per-model
  neutral-statement activations; ``[N=90, D]`` per layer, stored fp32.
  Its metadata-returning sibling lets dependent artifact builders reuse the
  sidecar validated in that same cache transaction instead of hashing the
  payload again merely to recover its identity.
* :func:`fit_alignment` — per-layer alignment map ``M_L : ℝ^D_src → ℝ^D_tgt``.
* :func:`transfer_profile` — apply the alignment map to a profile.
* :func:`alignment_cache_path` — disk cache for the fitted map keyed by
  the source model id.

The transferred profile lands at the target model's tensor path with a
``_from-<safe_src>`` suffix — uses the same variant-suffix machinery as
SAE variants, so the rest of saklas (selectors, packs, monitor) sees a
transferred probe as just another tensor on disk.
"""
from __future__ import annotations

import hashlib
import json
import math
import logging
import sys
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import torch
from safetensors import safe_open
from safetensors.torch import (
    load as load_safetensors,
    save as save_safetensors,
)

from saklas.core.errors import SaklasError
from saklas.core.profile import Profile
from saklas.io.atomic import fsync_directory, write_bytes_atomic, write_json_atomic
from saklas.io.paths import model_dir, safe_model_id
from saklas.io.packs import hash_file

log = logging.getLogger(__name__)

_NEUTRAL_ACTS_NAME = "neutral_activations"
_NEUTRAL_CACHE_FORMAT_VERSION = 4
_NEUTRAL_CAPTURE_VERSION = 1
_ALIGNMENT_CACHE_FORMAT_VERSION = 5

_NEUTRAL_SIDECAR_FIELDS = {
    "method", "format_version", "capture_version", "capture_sha256",
    "model_fingerprint", "model_source_fingerprint", "tensor_sha256",
    "tensor_files", "layers", "tensor_schema", "n_prompts", "n_layers",
}
_ALIGNMENT_SIDECAR_FIELDS = {
    "format_version", "method", "source_model_id", "target_model_id",
    "shared_layers", "tensor_schema", "source_neutral_identity",
    "target_neutral_identity", "tensor_files", "tensor_sha256",
    "quality_per_layer",
}


@dataclass(frozen=True)
class LayerAlignment:
    """Compact affine source->target activation map for one layer.

    ``linear(x) = left @ (right @ x)`` stores a rank-``r`` map in
    ``O(r * (D_src + D_tgt))`` rather than materializing ``D_tgt * D_src``.
    ``offset`` is applied only to activation *points*.  Directions intentionally
    use the linear component alone.
    """

    left: torch.Tensor       # (D_tgt, r)
    right: torch.Tensor      # (r, D_src)
    offset: torch.Tensor     # (D_tgt,)

    def __post_init__(self) -> None:
        if self.left.ndim != 2 or self.right.ndim != 2 or self.offset.ndim != 1:
            raise ValueError("alignment factors must be 2-D, 2-D, and 1-D")
        if self.left.shape[1] != self.right.shape[0]:
            raise ValueError("alignment factor ranks do not match")
        if self.left.shape[0] != self.offset.shape[0]:
            raise ValueError("alignment offset does not match target dimension")
        if self.left.shape[1] == 0:
            raise ValueError("alignment rank must be positive")
        device = self.left.device
        object.__setattr__(
            self, "left", self.left.to(device=device, dtype=torch.float32).contiguous(),
        )
        object.__setattr__(
            self, "right", self.right.to(device=device, dtype=torch.float32).contiguous(),
        )
        object.__setattr__(
            self, "offset", self.offset.to(device=device, dtype=torch.float32).contiguous(),
        )

    @property
    def source_dim(self) -> int:
        return int(self.right.shape[1])

    @property
    def target_dim(self) -> int:
        return int(self.left.shape[0])

    @property
    def rank(self) -> int:
        return int(self.left.shape[1])

    @property
    def shape(self) -> tuple[int, int]:
        return (self.target_dim, self.source_dim)

    @property
    def device(self) -> torch.device:
        return self.left.device

    def apply_vector(self, value: torch.Tensor) -> torch.Tensor:
        value = value.to(device=self.device, dtype=torch.float32)
        return self.left @ (self.right @ value)

    def apply_vectors(self, value: torch.Tensor) -> torch.Tensor:
        """Apply the linear component to row-major ``(..., D_src)`` vectors."""
        value = value.to(device=self.device, dtype=torch.float32)
        return (value @ self.right.transpose(0, 1)) @ self.left.transpose(0, 1)

    def apply_points(self, value: torch.Tensor) -> torch.Tensor:
        value = value.to(device=self.device, dtype=torch.float32)
        return self.apply_vectors(value) + self.offset

    def to_dense(self) -> torch.Tensor:
        return self.left @ self.right


AlignmentMap = Mapping[int, LayerAlignment]


# ---------------------------------------------------------------------------
# Per-model neutral activation cache.
# ---------------------------------------------------------------------------


def _neutral_acts_paths(model_id: str) -> tuple[Path, Path]:
    """Return the stable lock anchor and atomic pointer sidecar."""
    md = model_dir(model_id)
    return (
        md / f"{_NEUTRAL_ACTS_NAME}.safetensors",
        md / f"{_NEUTRAL_ACTS_NAME}.json",
    )


def _neutral_generation_path(anchor: Path, layer: int) -> Path:
    return anchor.with_name(
        f"{anchor.stem}.layer-{int(layer)}.gen-{uuid.uuid4().hex}{anchor.suffix}",
    )


def _neutral_shard_paths(
    anchor: Path, sidecar: Mapping[str, Any], layers: list[int],
) -> dict[int, Path]:
    files = sidecar.get("tensor_files")
    if not isinstance(files, Mapping):
        raise ValueError("neutral activation cache has no tensor shard map")
    if {str(layer) for layer in layers} != {str(key) for key in files}:
        raise ValueError("neutral tensor shard keys do not match layers")
    out: dict[int, Path] = {}
    for layer in layers:
        filename = files.get(str(layer))
        if (
            not isinstance(filename, str)
            or not filename
            or Path(filename).name != filename
        ):
            raise ValueError(f"invalid neutral shard for layer {layer}")
        out[layer] = anchor.parent / filename
    return out


def _json_pointer_matches(path: Path, payload: Mapping[str, Any]) -> bool:
    """Whether an exception escaped after this exact pointer was replaced."""
    try:
        with open(path) as handle:
            current = json.load(handle)
        return current == payload
    except (OSError, json.JSONDecodeError, TypeError):
        return False


def _cleanup_neutral_generations(
    anchor: Path, sidecar: Mapping[str, Any],
) -> None:
    files = sidecar.get("tensor_files")
    keep = (
        {str(filename) for filename in files.values() if isinstance(filename, str)}
        if isinstance(files, Mapping) else set()
    )
    for path in (
        *anchor.parent.glob(f"{anchor.stem}.layer-*.gen-*.safetensors"),
        *anchor.parent.glob(f"{anchor.stem}.layer-*.gen-*.safetensors.tmp"),
        anchor,
        anchor.with_suffix(anchor.suffix + ".tmp"),
    ):
        if path.name not in keep:
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                log.warning("could not remove old neutral generation %s: %s", path, exc)


@contextmanager
def neutral_fit_lock(model_id: str) -> Iterator[None]:
    """Single-flight lock for one model's expensive neutral forward pass."""
    from saklas.io.atomic import artifact_lock

    ts_path, _ = _neutral_acts_paths(model_id)
    with artifact_lock(ts_path.with_name(f"{ts_path.stem}.fit")):
        yield


def validate_neutral_cache_metadata(
    model_id: str, *, verify_payload: bool = True,
) -> dict[str, Any]:
    """Validate a neutral cache's identity and tensor header without paging data.

    The payload digest proves bytes, while ``safe_open`` checks the declared
    key/shape/dtype schema from the mmap header. This is the cheap preflight for
    exact transfer repeats; numerical finite checks remain in the materializing
    loader used by fits and whiteners.
    """
    from saklas.io.atomic import artifact_lock

    ts_path, _ = _neutral_acts_paths(model_id)
    with artifact_lock(ts_path):
        return _validate_neutral_cache_metadata_locked(
            model_id, verify_payload=verify_payload,
        )


def _validate_neutral_cache_metadata_locked(
    model_id: str, *, verify_payload: bool = True,
) -> dict[str, Any]:
    """Header/digest validation with the neutral pair lock already held."""
    ts_path, sc_path = _neutral_acts_paths(model_id)
    if not sc_path.exists():
        raise FileNotFoundError(f"neutral activation cache missing for {model_id}")
    with open(sc_path) as handle:
        sidecar = json.load(handle)
    if not isinstance(sidecar, dict) or set(sidecar) != _NEUTRAL_SIDECAR_FIELDS:
        raise ValueError("neutral activation cache sidecar has a non-current schema")
    version = sidecar["format_version"]
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version != _NEUTRAL_CACHE_FORMAT_VERSION
        or isinstance(sidecar["capture_version"], bool)
        or not isinstance(sidecar["capture_version"], int)
        or sidecar["capture_version"] != _NEUTRAL_CAPTURE_VERSION
        or sidecar["method"] != "neutral_activations"
        or any(
            not isinstance(sidecar[key], str) or not sidecar[key]
            for key in (
                "capture_sha256", "model_fingerprint",
                "model_source_fingerprint",
            )
        )
    ):
        raise ValueError("neutral activation cache identity mismatch")
    layers = sidecar["layers"]
    schema = sidecar["tensor_schema"]
    if (
        not isinstance(layers, list)
        or not layers
        or any(
            isinstance(layer, bool) or not isinstance(layer, int) or layer < 0
            for layer in layers
        )
        or layers != sorted(set(layers))
        or not isinstance(schema, dict)
        or set(schema) != {str(layer) for layer in layers}
        or isinstance(sidecar["n_prompts"], bool)
        or not isinstance(sidecar["n_prompts"], int)
        or sidecar["n_prompts"] <= 0
        or isinstance(sidecar["n_layers"], bool)
        or not isinstance(sidecar["n_layers"], int)
        or sidecar["n_layers"] != len(layers)
    ):
        raise ValueError("neutral activation cache sidecar has no layer schema")
    normalized_layers = layers
    expected_n = sidecar["n_prompts"]
    paths = _neutral_shard_paths(ts_path, sidecar, normalized_layers)
    digests = sidecar.get("tensor_sha256")
    if (
        not isinstance(digests, Mapping)
        or set(digests) != {str(layer) for layer in layers}
    ):
        raise ValueError("neutral activation cache has no tensor digests")
    grouped = {path: [layer] for layer, path in paths.items()}
    for layer, path in paths.items():
        digest = digests.get(str(layer))
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
            or not path.exists()
        ):
            raise ValueError(f"invalid neutral shard for layer {layer}")
        if verify_payload and hash_file(path) != digest:
            raise ValueError("neutral activation cache payload digest mismatch")
    for path, path_layers in grouped.items():
        with safe_open(str(path), framework="pt", device="cpu") as tensors:
            path_keys = {f"layer_{layer}" for layer in path_layers}
            if set(tensors.keys()) != path_keys:
                raise ValueError("neutral activation cache layer keys do not match sidecar")
            for key in tensors.keys():
                layer = int(key.split("_", 1)[1])
                spec = schema.get(str(layer), {})
                view = tensors.get_slice(key)
                shape = list(view.get_shape())
                if (
                    not isinstance(spec, dict)
                    or set(spec) != {"shape", "dtype"}
                    or not isinstance(spec["shape"], list)
                    or any(
                        isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0
                        for dim in spec["shape"]
                    )
                    or view.get_dtype() != "F32"
                    or spec.get("dtype") != "torch.float32"
                    or shape != spec.get("shape")
                    or len(shape) != 2
                    or shape[0] != expected_n
                ):
                    raise ValueError(
                        f"neutral activation cache layer {layer} header failed validation"
                    )
    return sidecar


def load_validated_neutral_cache(
    model_id: str,
    *,
    requested_layers: Iterable[int] | None = None,
) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
    """Materialize and fully validate a neutral cache without loading the model."""
    from saklas.io.atomic import artifact_lock

    ts_path, _ = _neutral_acts_paths(model_id)
    with artifact_lock(ts_path):
        sidecar = _validate_neutral_cache_metadata_locked(
            model_id, verify_payload=False,
        )
        all_layers = sorted(int(layer) for layer in sidecar["layers"])
        selected = (
            all_layers if requested_layers is None else sorted(
                set(all_layers) & {int(layer) for layer in requested_layers}
            )
        )
        paths = _neutral_shard_paths(ts_path, sidecar, all_layers)
        digests = sidecar["tensor_sha256"]
        tensors: dict[str, torch.Tensor] = {}
        for layer in selected:
            payload = paths[layer].read_bytes()
            if hashlib.sha256(payload).hexdigest() != digests[str(layer)]:
                raise ValueError("neutral activation cache payload digest mismatch")
            shard = load_safetensors(payload)
            tensors[f"layer_{layer}"] = shard[f"layer_{layer}"]
        # Reap superseded generations preserved by a signal immediately after
        # pointer replacement while the pair lock keeps references stable.
        _cleanup_neutral_generations(ts_path, sidecar)
        fsync_directory(ts_path.parent)
    out: dict[int, torch.Tensor] = {}
    schema = sidecar["tensor_schema"]
    expected_n = int(sidecar.get("n_prompts", -1))
    for key, tensor in tensors.items():
        layer = int(key.split("_", 1)[1])
        expected_shape = schema.get(str(layer), {}).get("shape")
        if (
            tensor.dtype != torch.float32
            or list(tensor.shape) != expected_shape
            or tensor.ndim != 2
            or int(tensor.shape[0]) != expected_n
            or not bool(torch.isfinite(tensor).all())
        ):
            raise ValueError(f"neutral activation cache layer {layer} failed validation")
        out[layer] = tensor
    return out, sidecar


def neutral_cache_identity(sidecar: dict[str, Any]) -> dict[str, Any]:
    """Stable identity subset embedded into dependent alignment artifacts."""
    keys = (
        "format_version", "capture_version", "model_fingerprint",
        "model_source_fingerprint",
        "capture_sha256", "tensor_sha256", "layers", "tensor_schema",
        "n_prompts",
    )
    required = {*keys, "tensor_files", "method"}
    if (
        not required <= set(sidecar)
        or sidecar["format_version"] != _NEUTRAL_CACHE_FORMAT_VERSION
        or sidecar["method"] != "neutral_activations"
    ):
        raise ValueError("neutral cache identity requires current validated metadata")
    identity = {key: sidecar[key] for key in keys}
    identity["tensor_files"] = sidecar["tensor_files"]
    return identity


def load_or_compute_neutral_activations(
    model: Any,
    tokenizer: Any,
    layers: Any,
    *,
    model_id: str,
    force: bool = False,
) -> dict[int, torch.Tensor]:
    """Single-flight neutral-cache load/capture, returning activations only."""
    activations, _sidecar = load_or_compute_neutral_activations_with_metadata(
        model, tokenizer, layers, model_id=model_id, force=force,
    )
    return activations


def load_or_compute_neutral_activations_with_metadata(
    model: Any,
    tokenizer: Any,
    layers: Any,
    *,
    model_id: str,
    force: bool = False,
) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
    """Return activations plus the sidecar proven in the same transaction.

    Alignment fitting needs both the materialized neutral rows and their stable
    cache identity.  Returning the already-validated sidecar avoids an immediate
    second full-file digest pass through
    :func:`validate_neutral_cache_metadata`.  The activation-only public API
    above remains the convenience surface for ordinary whitener/probe callers.
    """
    with neutral_fit_lock(model_id):
        return _load_or_compute_neutral_activations_with_metadata_locked(
            model, tokenizer, layers, model_id=model_id, force=force,
        )


def _load_or_compute_neutral_activations_with_metadata_locked(
    model: Any,
    tokenizer: Any,
    layers: Any,
    *,
    model_id: str,
    force: bool = False,
) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
    """Disk-cached per-statement activations for one model.

    Cache identity binds the exact rendered token rows and pooling positions,
    the loaded model weights, and the fitted layer set. The tensor payload is
    digest-verified before reuse. A stale cache is recomputed and replaced.

    Returns ``({layer_idx: [N, D] fp32 CPU tensor}, validated_sidecar)``.
    Stored fp32 on disk so the whitener covariance is built and inverted at
    full precision and the compute path (cache miss) and cache-hit path return
    bit-identical tensors (no precision seam — bf16's ~0.4% input error landed
    on exactly the inverted quantity).  fp16 was abandoned because its 65504
    ceiling overflows gemma-3's extreme late-layer channels to ±inf (poisoning
    ``λ`` / ``K``); fp32 has the range and, unlike the former bf16 store, no
    input-precision loss, at the cost of ~2× disk on this one cache.
    """
    from saklas.core.model import loaded_model_fingerprint
    from saklas.core.capture import (
        _neutral_pairs,
        _prepare_capture_batch,
        compute_neutral_activations,
    )

    ts_path, sc_path = _neutral_acts_paths(model_id)
    md = model_dir(model_id)
    md.mkdir(parents=True, exist_ok=True)

    pairs = _neutral_pairs()
    prompts = [prompt for prompt, _ in pairs]
    responses = [response for _, response in pairs]
    prepared = _prepare_capture_batch(
        tokenizer, prompts, responses,
    )
    rendered_payload = [
        {"input_ids": ids[0].tolist(), "content_end": int(content_end)}
        for ids, content_end in prepared
    ]
    capture_sha = hashlib.sha256(json.dumps(
        {"capture_version": _NEUTRAL_CAPTURE_VERSION, "rows": rendered_payload},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    model_fingerprint = loaded_model_fingerprint(model, model_id)
    base_model = getattr(model, "_orig_mod", model)
    model_source_fingerprint = getattr(
        base_model, "_saklas_source_fingerprint", None,
    )
    expected_layers = list(range(len(layers)))

    if not force and sc_path.exists():
        try:
            out, sc = load_validated_neutral_cache(model_id)
            identity_matches = (
                sc.get("format_version") == _NEUTRAL_CACHE_FORMAT_VERSION
                and sc.get("capture_version") == _NEUTRAL_CAPTURE_VERSION
                and sc.get("capture_sha256") == capture_sha
                and sc.get("model_fingerprint") == model_fingerprint
                and sc.get("model_source_fingerprint") == model_source_fingerprint
                and sc.get("layers") == expected_layers
            )
            if identity_matches:
                log.debug("Loaded cached neutral activations for %s", model_id)
                return out, sc
            else:
                log.info(
                    "Neutral activations stale (model/token/corpus/layer identity changed); "
                    "recomputing for %s",
                    model_id,
                )
        except Exception as e:
            log.warning("Corrupt neutral activations cache for %s, recomputing: %s", model_id, e)

    log.info("Computing neutral activations (one-time per model)...")
    activations = compute_neutral_activations(
        model, tokenizer, layers, rendered=prepared,
    )

    # Persist fp32 — the whitener covariance is built and inverted from these,
    # so the compute path (this fresh fp32 store) and the cache-hit path stay
    # bit-identical (no precision seam) and the inversion is full-precision.
    # fp32 carries the exponent range fp16 lacked (gemma-3's late layers exceed
    # the fp16 max of 65504 and overflowed to ±inf) without bf16's input loss,
    # at ~2× disk on this one cache.
    fp32 = {
        int(idx): t.contiguous().to(torch.float32).cpu()
        for idx, t in activations.items()
    }
    from saklas.io.atomic import artifact_lock

    with artifact_lock(ts_path):
        tensor_files: dict[str, str] = {}
        tensor_sha256: dict[str, str] = {}
        created: list[Path] = []
        try:
            for idx, tensor in sorted(fp32.items()):
                path = _neutral_generation_path(ts_path, idx)
                payload = save_safetensors({f"layer_{idx}": tensor})
                write_bytes_atomic(path, payload)
                created.append(path)
                tensor_files[str(idx)] = path.name
                tensor_sha256[str(idx)] = hashlib.sha256(payload).hexdigest()
        except BaseException:
            for path in created:
                path.unlink(missing_ok=True)
            raise
        sidecar: dict[str, Any] = {
            "method": "neutral_activations",
            "format_version": _NEUTRAL_CACHE_FORMAT_VERSION,
            "capture_version": _NEUTRAL_CAPTURE_VERSION,
            "capture_sha256": capture_sha,
            "model_fingerprint": model_fingerprint,
            "model_source_fingerprint": model_source_fingerprint,
            "tensor_sha256": tensor_sha256,
            "tensor_files": tensor_files,
            "layers": sorted(activations),
            "tensor_schema": {
                str(idx): {"shape": list(t.shape), "dtype": str(t.dtype)}
                for idx, t in sorted(activations.items())
            },
            "n_prompts": (
                next(iter(activations.values())).shape[0] if activations else 0
            ),
            "n_layers": len(activations),
        }
        if set(sidecar) != _NEUTRAL_SIDECAR_FIELDS:
            raise AssertionError("neutral writer produced a non-canonical sidecar")
        try:
            # Persist every new directory entry before publishing the pointer.
            # A crash can therefore expose either the complete old generation
            # or the complete new one, never a pointer to absent shards.
            fsync_directory(sc_path.parent)
            write_json_atomic(sc_path, sidecar)
        except BaseException:
            if not _json_pointer_matches(sc_path, sidecar):
                for path in created:
                    path.unlink(missing_ok=True)
            raise
        # The pointer rename is now authoritative. Preserve its generations if
        # this durability barrier itself fails; deleting them would break the
        # successfully replaced pointer immediately.
        fsync_directory(sc_path.parent)
        _cleanup_neutral_generations(ts_path, sidecar)
        fsync_directory(sc_path.parent)
    return activations, sidecar


# ---------------------------------------------------------------------------
# Alignment fit.
# ---------------------------------------------------------------------------


class AlignmentError(ValueError, SaklasError):
    """Raised when an alignment can't be fit (insufficient shared layers)."""

    def user_message(self) -> tuple[int, str]:
        return (422, str(self) or self.__class__.__name__)


def fit_alignment(
    src_acts: dict[int, torch.Tensor],
    tgt_acts: dict[int, torch.Tensor],
    *,
    min_shared_layers: int = 10,
    requested_layers: Iterable[int] | None = None,
    available_shared_layers: Iterable[int] | None = None,
) -> dict[int, LayerAlignment]:
    """Per-layer compact affine mapping ``A_L : ℝ^D_src → ℝ^D_tgt``.

    For each layer the two models share, fits ``A_L(x) = M_L x + b_L`` such
    that matched neutral activation *points* map source→target.  Directions
    use ``M_L`` alone, so translations never contaminate profile transfer.

    * **Same hidden dim:** evidence-supported orthogonal Procrustes, factored
      through economy QR row-space bases and a small ``<=N`` SVD.  No arbitrary
      unobserved ``D-N`` complement is invented.
    * **Different hidden dim:** minimum-norm least squares from the thin SVD of
      ``X_src``.  The result stays as rank-sized factors instead of a dense
      ``D_tgt × D_src`` matrix.

    The translation is ``b_L = mean_tgt - M_L mean_src``.  Returns compact
    :class:`LayerAlignment` objects over the shared-layer set; tensors are fp32
    on CPU.  Raises :class:`AlignmentError` when fewer than
    ``min_shared_layers`` layers are common to both models — partial
    alignment under that threshold tends to produce noisy transfers.
    """
    materialized_shared = set(src_acts.keys()) & set(tgt_acts.keys())
    available = (
        materialized_shared if available_shared_layers is None
        else {int(layer) for layer in available_shared_layers}
    )
    if len(available) < min_shared_layers:
        raise AlignmentError(
            f"alignment requires >= {min_shared_layers} shared layers; "
            f"src={sorted(src_acts.keys())}, tgt={sorted(tgt_acts.keys())}"
        )
    shared = sorted(
        materialized_shared if requested_layers is None else (
            materialized_shared & {int(layer) for layer in requested_layers}
        )
    )
    if not shared:
        raise AlignmentError("alignment has no requested layers shared by both models")

    out: dict[int, LayerAlignment] = {}
    for layer in shared:
        X_src = src_acts[layer].to(torch.float32)
        X_tgt = tgt_acts[layer].to(torch.float32)
        if X_src.shape[0] != X_tgt.shape[0]:
            raise AlignmentError(
                f"layer {layer}: src has N={X_src.shape[0]} prompts, "
                f"tgt has N={X_tgt.shape[0]}; expected matched."
            )

        # Center both sides — Procrustes is translation-invariant only
        # under shared origin.
        src_mean = X_src.mean(dim=0)
        tgt_mean = X_tgt.mean(dim=0)
        X_src_c = X_src - src_mean
        X_tgt_c = X_tgt - tgt_mean

        D_src = X_src_c.shape[1]
        D_tgt = X_tgt_c.shape[1]

        if D_src == D_tgt:
            # Low-rank orthogonal Procrustes.  Factor the two row spaces first,
            # then SVD only their <=N cross-core; the unobserved D-N complement
            # has no evidence and is deliberately not invented/persisted.
            Qs, Rs = torch.linalg.qr(X_src_c.transpose(0, 1), mode="reduced")
            Qt, Rt = torch.linalg.qr(X_tgt_c.transpose(0, 1), mode="reduced")
            U, S, Vh = torch.linalg.svd(Rt @ Rs.transpose(0, 1), full_matrices=False)
            tol = torch.finfo(S.dtype).eps * max(D_src, X_src_c.shape[0]) * float(S.max())
            rank = int((S > tol).sum())
            if rank == 0:
                raise AlignmentError(f"layer {layer}: centered alignment has rank 0")
            left = (Qt @ U[:, :rank]).contiguous()
            right = (Vh[:rank] @ Qs.transpose(0, 1)).contiguous()
        else:
            # Minimum-norm least squares in factored form:
            # M = X_tgt^T U S^-1 V^T for X_src = U S V^T.
            U, S, Vh = torch.linalg.svd(X_src_c, full_matrices=False)
            tol = torch.finfo(S.dtype).eps * max(X_src_c.shape) * float(S.max())
            rank = int((S > tol).sum())
            if rank == 0:
                raise AlignmentError(f"layer {layer}: centered alignment has rank 0")
            left = (
                X_tgt_c.transpose(0, 1) @ U[:, :rank]
            ).div(S[:rank].unsqueeze(0)).contiguous()
            right = Vh[:rank].contiguous()

        linear_src_mean = left @ (right @ src_mean)
        offset = (tgt_mean - linear_src_mean).contiguous()
        out[layer] = LayerAlignment(left, right, offset)
    return out


def alignment_quality(
    M: AlignmentMap,
    src_acts: dict[int, torch.Tensor],
    tgt_acts: dict[int, torch.Tensor],
) -> dict[int, float]:
    """Per-layer R² between the transferred src activations and tgt.

    For each layer ``L`` shared by ``M``, computes
    ``1 - ||X_tgt - X_src @ M_L^T||^2 / ||X_tgt||^2`` on centered data.
    Values near 1.0 mean the linear map captures the cross-model
    geometry; values near 0.0 mean transferred probes will be noisy.

    Surfaced through ``Sidecar.transfer_quality_estimate`` (median over
    shared layers) so users see one summary number per transfer; the full
    per-layer dict is exported in transfer sidecars and CLI JSON output for
    callers that want it.
    """
    out: dict[int, float] = {}
    for layer, M_L in M.items():
        X_src = src_acts[layer].to(torch.float32)
        X_tgt = tgt_acts[layer].to(torch.float32)
        X_src_c = X_src - X_src.mean(dim=0, keepdim=True)
        X_tgt_c = X_tgt - X_tgt.mean(dim=0, keepdim=True)
        # Translation cancels under centering; score the linear geometry.
        X_pred = M_L.apply_vectors(X_src_c)
        residual = (X_tgt_c - X_pred).pow(2).sum().item()
        total = X_tgt_c.pow(2).sum().item()
        if total <= 1e-12:
            out[layer] = 0.0
        else:
            out[layer] = float(1.0 - residual / total)
    return out


# ---------------------------------------------------------------------------
# Profile transfer.
# ---------------------------------------------------------------------------


def transfer_profile(
    profile: Profile,
    alignment_map: AlignmentMap,
    *,
    source_model_id: str,
    transfer_quality_estimate: float | None = None,
    whitener: "Any | None" = None,
) -> Profile:
    """Apply the alignment map to a source-space profile.

    For each layer in the profile that the alignment covers, computes
    ``v_tgt = M_L @ v_src``.  Layers not covered by the alignment are
    dropped — partial transfer is the only sensible behavior, since a
    direction with no map can't be lifted into target space.

    **Target-metric re-bake (mandatory).**  The source tensor is
    share-baked in the *source* model's metric — its per-layer Euclidean
    magnitude is the source Mahalanobis norm of the raw mean-diff (the
    unified subspace hook reads ``‖baked_L‖₂ / Σ‖baked‖₂`` back out as
    the layer share).  The orthogonal Procrustes map preserves Euclidean norm, so a
    bare transfer would carry the *source* cross-layer share into target
    space — where it no longer matches the target's anisotropy.  The
    ``whitener`` for the **target** model is **required** and must cover
    every transferred layer (all-or-nothing, mirroring the DiM-bake /
    monitor / manifold-fit gate); each layer is rescaled so its magnitude
    becomes its *target* Mahalanobis norm::

        v_tgt'_L = v_tgt_L · (‖v_tgt_L‖_M(target) / ‖v_tgt_L‖₂)

    The direction is untouched; only the per-layer magnitude — and hence
    the hook-recovered share — changes.  ``‖v_tgt_L‖₂`` carries
    the transported source-signal strength (the best target-signal proxy
    available without target contrastive pairs) and ``‖v̂_tgt_L‖_M(target)``
    applies the target anisotropy correction, so the composite is the
    target-metric analogue of a native DiM bake.  ``bake: "mahalanobis"``
    is stamped on the result.  A missing or non-covering whitener raises
    :class:`WhitenerError` — there is no Euclidean transfer.

    Carries provenance through ``Profile.metadata``:

    * ``method = "procrustes_transfer"``
    * ``source_model_id`` — HF coord of the source model.
    * ``transfer_quality_estimate`` — median R² across shared layers, if
      known.
    * ``bake`` — always ``"mahalanobis"``.

    Existing diagnostics fields on the source profile pass through
    unchanged; users can still reason about source-side separation when
    judging whether to trust a transferred probe.
    """
    if not alignment_map:
        from saklas.core.profile import ProfileError

        raise ProfileError("transfer_profile: alignment_map is empty")

    # Stage the transferred directions in fp32 (keyed by layer) plus the
    # original per-layer dtype, so the target re-bake operates at full
    # precision before the dtype restore.
    staged: dict[int, torch.Tensor] = {}
    orig_dtype: dict[int, torch.dtype] = {}
    for layer, src_vec in profile.items():
        M_L = alignment_map.get(layer)
        if M_L is None:
            continue
        staged[layer] = M_L.apply_vector(src_vec).cpu()
        orig_dtype[layer] = src_vec.dtype

    if not staged:
        from saklas.core.profile import ProfileError

        raise ProfileError(
            "transfer_profile: alignment covered no layers in the source profile"
        )

    # Mahalanobis-only target re-bake: the per-layer ‖·‖_M and ‖·‖₂ scales
    # differ by a 1/√λ_L factor that doesn't cancel from the cross-layer
    # share, so the target whitener must cover every transferred layer.
    # No Euclidean transfer — a missing / partial whitener is an error.
    from saklas.core.mahalanobis import WhitenerError

    if whitener is None or not whitener.covers_all(staged.keys()):
        raise WhitenerError(
            "transfer_profile requires a Mahalanobis whitener covering every "
            f"transferred layer {sorted(staged.keys())}; generate neutral "
            "activations for the TARGET model first (the Euclidean path is gone)"
        )
    for layer, v_tgt in staged.items():
        eucl = float(v_tgt.norm().item())
        if eucl < 1e-8:
            # Degenerate direction — leave it; rescaling a zero vector
            # is undefined and it carries no share anyway.
            continue
        m_norm = whitener.mahalanobis_norm(layer, v_tgt)
        staged[layer] = v_tgt * (m_norm / eucl)

    out_tensors = {
        layer: v.to(dtype=orig_dtype[layer]) for layer, v in staged.items()
    }

    metadata = dict(profile.metadata)
    metadata["method"] = "procrustes_transfer"
    metadata["source_model_id"] = source_model_id
    metadata["bake"] = "mahalanobis"
    if transfer_quality_estimate is not None:
        metadata["transfer_quality_estimate"] = float(transfer_quality_estimate)
    return Profile(out_tensors, metadata=metadata)


# ---------------------------------------------------------------------------
# Alignment-map disk cache.
# ---------------------------------------------------------------------------


def _alignment_anchor_paths(
    src_model_id: str, tgt_model_id: str,
) -> tuple[Path, Path]:
    """Stable lock anchor and atomic v5 pointer sidecar."""
    md = model_dir(tgt_model_id)
    al_dir = md / "alignments"
    src_safe = safe_model_id(src_model_id)
    return (al_dir / f"{src_safe}.safetensors", al_dir / f"{src_safe}.json")


def alignment_cache_path(src_model_id: str, tgt_model_id: str) -> tuple[Path, Path]:
    """Representative shard and stable pointer path for an alignment cache."""
    anchor, sidecar_path = _alignment_anchor_paths(src_model_id, tgt_model_id)
    try:
        with open(sidecar_path) as handle:
            sidecar = json.load(handle)
        files = sidecar.get("tensor_files")
        layers = sidecar.get("shared_layers")
        if isinstance(files, Mapping) and isinstance(layers, list) and layers:
            filename = files.get(str(min(int(layer) for layer in layers)))
            if isinstance(filename, str) and Path(filename).name == filename:
                return anchor.parent / filename, sidecar_path
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        pass
    return anchor, sidecar_path


def _alignment_generation_path(anchor: Path, layer: int) -> Path:
    return anchor.with_name(
        f"{anchor.stem}.layer-{int(layer)}.gen-{uuid.uuid4().hex}{anchor.suffix}",
    )


def _alignment_shard_paths(
    anchor: Path, sidecar: Mapping[str, Any], layers: list[int],
) -> dict[int, Path]:
    files = sidecar.get("tensor_files")
    if not isinstance(files, Mapping):
        raise ValueError("alignment cache has no tensor shard map")
    if {str(layer) for layer in layers} != {str(key) for key in files}:
        raise ValueError("alignment tensor shard keys do not match shared layers")
    out: dict[int, Path] = {}
    for layer in layers:
        filename = files.get(str(layer))
        if (
            not isinstance(filename, str)
            or not filename
            or Path(filename).name != filename
        ):
            raise ValueError(f"invalid alignment shard for layer {layer}")
        out[layer] = anchor.parent / filename
    return out


def _cleanup_alignment_generations(anchor: Path, sidecar: Mapping[str, Any]) -> None:
    files = sidecar.get("tensor_files")
    keep = (
        {str(filename) for filename in files.values() if isinstance(filename, str)}
        if isinstance(files, Mapping) else set()
    )
    for path in (
        *anchor.parent.glob(f"{anchor.stem}.layer-*.gen-*.safetensors"),
        *anchor.parent.glob(f"{anchor.stem}.layer-*.gen-*.safetensors.tmp"),
        anchor,
        anchor.with_suffix(anchor.suffix + ".tmp"),
    ):
        if path.name not in keep:
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                log.warning("could not remove old alignment generation %s: %s", path, exc)


@contextmanager
def alignment_fit_lock(src_model_id: str, tgt_model_id: str) -> Iterator[None]:
    """Single-flight a complete directional alignment fit, including loads."""
    from saklas.io.atomic import artifact_lock

    ts_path, _ = _alignment_anchor_paths(src_model_id, tgt_model_id)
    with artifact_lock(ts_path.with_name(f"{ts_path.stem}.fit")):
        yield


def save_alignment_map(
    M: AlignmentMap,
    src_model_id: str,
    tgt_model_id: str,
    *,
    source_identity: dict[str, Any],
    target_identity: dict[str, Any],
    quality_per_layer: dict[int, float] | None = None,
    extend: bool = False,
) -> Path:
    """Persist a fitted alignment map.

    Writes one bounded safetensors shard per layer, then atomically switches the
    JSON pointer.  Payload digests are computed from the exact bytes being
    written, so publication never rereads a complete shard.
    """
    anchor, sc_path = _alignment_anchor_paths(src_model_id, tgt_model_id)
    anchor.parent.mkdir(parents=True, exist_ok=True)

    normalized = {int(idx): value for idx, value in M.items()}
    if not normalized:
        raise ValueError("cannot save an empty alignment map")
    from saklas.io.atomic import artifact_lock

    with artifact_lock(anchor):
        tensor_files: dict[str, str] = {}
        tensor_sha256: dict[str, str] = {}
        tensor_schema: dict[str, Any] = {}
        merged_quality: dict[str, float | None] = {}
        if extend and sc_path.exists():
            try:
                with open(sc_path) as handle:
                    current = json.load(handle)
                if (
                    isinstance(current, dict)
                    and set(current) == _ALIGNMENT_SIDECAR_FIELDS
                    and current.get("format_version") == _ALIGNMENT_CACHE_FORMAT_VERSION
                    and current.get("method") == "sharded_factorized_affine_alignment"
                    and current.get("source_model_id") == src_model_id
                    and current.get("target_model_id") == tgt_model_id
                    and current.get("source_neutral_identity") == source_identity
                    and current.get("target_neutral_identity") == target_identity
                ):
                    current_layers = sorted(
                        int(layer) for layer in current.get("shared_layers", [])
                    )
                    current_paths = _alignment_shard_paths(
                        anchor, current, current_layers,
                    )
                    current_digests = current.get("tensor_sha256")
                    current_schema = current.get("tensor_schema")
                    if not isinstance(current_digests, Mapping) or not isinstance(
                        current_schema, Mapping,
                    ):
                        raise ValueError("alignment pointer has no shard metadata")
                    for layer in current_layers:
                        try:
                            digest = current_digests.get(str(layer))
                            spec = current_schema.get(str(layer))
                            path = current_paths[layer]
                            if (
                                not isinstance(digest, str) or len(digest) != 64
                                or not isinstance(spec, Mapping) or not path.exists()
                            ):
                                continue
                            # Header-only proof is enough here: this writer is
                            # under the artifact lock and carries the immutable
                            # generation forward byte-for-byte. The newly fitted
                            # mapping below replaces a damaged requested layer.
                            with safe_open(
                                str(path), framework="pt", device="cpu",
                            ) as shard:
                                if set(shard.keys()) != {"left", "right", "offset"}:
                                    continue
                            tensor_files[str(layer)] = path.name
                            tensor_sha256[str(layer)] = digest
                            tensor_schema[str(layer)] = dict(spec)
                        except (OSError, RuntimeError, ValueError, KeyError):
                            continue
                    raw_quality = current.get("quality_per_layer") or {}
                    if isinstance(raw_quality, Mapping):
                        merged_quality.update({
                            str(layer): float(value)
                            for layer, value in raw_quality.items()
                            if value is not None
                        })
            except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
                log.warning("could not extend current alignment generation: %s", exc)
                tensor_files.clear()
                tensor_sha256.clear()
                tensor_schema.clear()
                merged_quality.clear()
        created: list[Path] = []
        try:
            for idx, alignment in sorted(normalized.items()):
                path = _alignment_generation_path(anchor, idx)
                payload = save_safetensors({
                    "left": alignment.left.contiguous().cpu(),
                    "right": alignment.right.contiguous().cpu(),
                    "offset": alignment.offset.contiguous().cpu(),
                })
                write_bytes_atomic(path, payload)
                created.append(path)
                tensor_files[str(idx)] = path.name
                tensor_sha256[str(idx)] = hashlib.sha256(payload).hexdigest()
                tensor_schema[str(idx)] = {
                    "left": list(alignment.left.shape),
                    "right": list(alignment.right.shape),
                    "offset": list(alignment.offset.shape),
                }
        except BaseException:
            for path in created:
                path.unlink(missing_ok=True)
            raise
        sidecar: dict[str, Any] = {
            "format_version": _ALIGNMENT_CACHE_FORMAT_VERSION,
            "method": "sharded_factorized_affine_alignment",
            "source_model_id": src_model_id,
            "target_model_id": tgt_model_id,
            "shared_layers": sorted(int(layer) for layer in tensor_files),
            "tensor_schema": tensor_schema,
            "source_neutral_identity": source_identity,
            "target_neutral_identity": target_identity,
            "tensor_files": tensor_files,
            "tensor_sha256": tensor_sha256,
            "quality_per_layer": {},
        }
        if quality_per_layer:
            merged_quality.update({
                str(layer): round(float(q), 6)
                for layer, q in quality_per_layer.items()
            })
        for layer in sidecar["shared_layers"]:
            merged_quality.setdefault(str(layer), None)
        sidecar["quality_per_layer"] = merged_quality
        if set(sidecar) != _ALIGNMENT_SIDECAR_FIELDS:
            raise AssertionError("alignment writer produced a non-canonical sidecar")
        try:
            fsync_directory(sc_path.parent)
            write_json_atomic(sc_path, sidecar)
        except BaseException:
            if not _json_pointer_matches(sc_path, sidecar):
                for path in created:
                    path.unlink(missing_ok=True)
            raise
        # The pointer is now authoritative. If directory fsync itself fails,
        # preserve the generations it names; deleting them here would turn a
        # successfully replaced pointer into an immediately broken cache.
        fsync_directory(sc_path.parent)
        _cleanup_alignment_generations(anchor, sidecar)
        fsync_directory(sc_path.parent)
    return anchor.parent / tensor_files[str(min(int(layer) for layer in tensor_files))]


def load_alignment_map(
    src_model_id: str, tgt_model_id: str,
    *,
    source_identity: dict[str, Any],
    target_identity: dict[str, Any],
    requested_layers: Iterable[int] | None = None,
) -> tuple[dict[int, LayerAlignment], dict[str, Any]] | None:
    """Load all or selected factor shards after identity/header preflight.

    ``requested_layers`` materializes only the requested intersection with the
    cache's shared layers.  The returned sidecar remains the complete pointer.
    """

    anchor, sc_path = _alignment_anchor_paths(src_model_id, tgt_model_id)
    from saklas.io.atomic import artifact_lock

    with artifact_lock(anchor):
        if not sc_path.exists():
            return None
        try:
            with open(sc_path) as f:
                sidecar = json.load(f)
            if not isinstance(sidecar, dict) or set(sidecar) != _ALIGNMENT_SIDECAR_FIELDS:
                return None
            # Identity mismatch is the common stale-cache case.  Reject it before
            # hashing or materializing a potentially multi-GiB payload.
            if (
                sidecar.get("format_version") != _ALIGNMENT_CACHE_FORMAT_VERSION
                or sidecar.get("method") != "sharded_factorized_affine_alignment"
                or sidecar.get("source_model_id") != src_model_id
                or sidecar.get("target_model_id") != tgt_model_id
                or sidecar.get("source_neutral_identity") != source_identity
                or sidecar.get("target_neutral_identity") != target_identity
            ):
                return None
            expected_layers = sidecar["shared_layers"]
            schema = sidecar["tensor_schema"]
            if (
                not isinstance(expected_layers, list)
                or not expected_layers
                or any(
                    isinstance(layer, bool) or not isinstance(layer, int) or layer < 0
                    for layer in expected_layers
                )
                or expected_layers != sorted(set(expected_layers))
                or not isinstance(schema, dict)
                or set(schema) != {str(layer) for layer in expected_layers}
                or not isinstance(sidecar["quality_per_layer"], dict)
            ):
                return None
            all_layers = expected_layers
            paths = _alignment_shard_paths(anchor, sidecar, all_layers)
            digests = sidecar.get("tensor_sha256")
            layer_keys = {str(layer) for layer in all_layers}
            quality = sidecar["quality_per_layer"]
            if (
                not isinstance(digests, Mapping)
                or set(digests) != layer_keys
                or set(quality) != layer_keys
                or any(
                    value is not None and (
                        not isinstance(value, (int, float))
                        or isinstance(value, bool)
                        or not math.isfinite(float(value))
                    )
                    for value in quality.values()
                )
            ):
                return None
            # Validate declared headers independently before payload materialization.
            # One damaged layer must remain repairable without discarding the
            # pointer metadata for unrelated immutable factors.
            valid_headers: set[int] = set()
            for layer in all_layers:
                spec = schema.get(str(layer), {})
                digest = digests.get(str(layer))
                if (
                    not isinstance(spec, dict)
                    or set(spec) != {"left", "right", "offset"}
                    or any(
                        not isinstance(spec[part], list)
                        or not spec[part]
                        or any(
                            isinstance(dim, bool) or not isinstance(dim, int)
                            or dim <= 0 for dim in spec[part]
                        )
                        for part in ("left", "right", "offset")
                    )
                    or len(spec["left"]) != 2
                    or len(spec["right"]) != 2
                    or len(spec["offset"]) != 1
                    or spec["left"][1] != spec["right"][0]
                    or spec["left"][0] != spec["offset"][0]
                    or not isinstance(digest, str)
                    or len(digest) != 64
                    or any(char not in "0123456789abcdef" for char in digest)
                    or not paths[layer].exists()
                ):
                    continue
                try:
                    with safe_open(
                        str(paths[layer]), framework="pt", device="cpu",
                    ) as shard:
                        if set(shard.keys()) != {"left", "right", "offset"}:
                            continue
                        if any(
                            list(shard.get_slice(part).get_shape()) != spec.get(part)
                            or shard.get_slice(part).get_dtype() != "F32"
                            for part in ("left", "right", "offset")
                        ):
                            continue
                    valid_headers.add(layer)
                except (OSError, RuntimeError, ValueError, KeyError):
                    continue
            selected = (
                all_layers if requested_layers is None else sorted(
                    set(all_layers) & {int(layer) for layer in requested_layers}
                )
            )
            M: dict[int, LayerAlignment] = {}
            for layer in selected:
                if layer not in valid_headers:
                    continue
                try:
                    payload = paths[layer].read_bytes()
                    if hashlib.sha256(payload).hexdigest() != digests[str(layer)]:
                        continue
                    tensors = load_safetensors(payload)
                    left = tensors["left"]
                    right = tensors["right"]
                    offset = tensors["offset"]
                    if (
                        left.ndim != 2 or right.ndim != 2 or offset.ndim != 1
                        or left.shape[1] != right.shape[0]
                        or left.shape[0] != offset.shape[0]
                        or not all(
                            bool(torch.isfinite(t).all())
                            for t in (left, right, offset)
                        )
                    ):
                        continue
                    M[layer] = LayerAlignment(left, right, offset)
                except (OSError, RuntimeError, ValueError, KeyError):
                    continue
            _cleanup_alignment_generations(anchor, sidecar)
            return M, sidecar
        except Exception as e:
            log.warning(
                "Corrupt alignment map cache (%s → %s): %s",
                src_model_id, tgt_model_id, e,
            )
            return None


# ---------------------------------------------------------------------------
# Transfer-alignment orchestration
# ---------------------------------------------------------------------------
#
# Single-flight locking + cache-proof + retry-on-race orchestration that loads
# or fits a complete source->target Procrustes alignment and returns the
# identity-matched target metric.  Promoted out of ``cli/runners.py`` so it
# lives beside the primitives it wraps (``load_alignment_map`` /
# ``validate_neutral_cache_metadata`` / ``fit_alignment`` / ``save_alignment_map``).
# Concurrency semantics (locking order, proof-checking without payload hashing,
# roster narrowing, model-construction avoidance on cache hit, retry-on-race
# recursion) are load-bearing and documented in ``io/AGENTS.md``.


def _target_whitener_from_neutral_activations(
    activations: dict[int, Any],
) -> Any:
    """Build the transfer target metric from an already-proven row roster."""
    from saklas.core.mahalanobis import LayerWhitener

    return LayerWhitener.from_neutral_activations(
        activations,
        {
            layer: tensor.mean(dim=0)
            for layer, tensor in activations.items()
        },
    )


def load_or_fit_transfer_alignment(
    src_model: str,
    tgt_model: str,
    *,
    force: bool,
    label: str,
    requested_layers: "Sequence[int] | None" = None,
) -> tuple[
    dict[int, Any], dict[int, float], Path, dict[str, Any], dict[str, Any], Any,
    dict[int, Any],
]:
    """Single-flight a complete alignment plus its target metric."""
    with alignment_fit_lock(src_model, tgt_model):
        return _load_or_fit_transfer_alignment_locked(
            src_model, tgt_model, force=force, label=label,
            requested_layers=requested_layers,
        )


def _load_or_fit_transfer_alignment_locked(
    src_model: str,
    tgt_model: str,
    *,
    force: bool,
    label: str,
    requested_layers: "Sequence[int] | None" = None,
) -> tuple[
    dict[int, Any], dict[int, float], Path, dict[str, Any], dict[str, Any], Any,
    dict[int, Any],
]:
    """Load or fit an alignment and return its identity-matched target whitener."""
    from saklas.core.model import model_source_fingerprint
    from saklas.core.session import SaklasSession

    wanted_arg = (
        None if requested_layers is None
        else {int(layer) for layer in requested_layers}
    )

    def _proven_sidecar(model_id: str) -> dict[str, Any] | None:
        """Prove cache/source identity without hashing any tensor payload."""
        try:
            source = model_source_fingerprint(model_id)
            if source is None:
                return None
            sidecar = validate_neutral_cache_metadata(
                model_id, verify_payload=False,
            )
            if sidecar.get("model_source_fingerprint") != source:
                return None
            return sidecar
        except (OSError, RuntimeError, ValueError, TypeError, KeyError):
            return None

    def _load_proven_rows(
        model_id: str, layers: set[int] | None,
    ) -> tuple[dict[int, Any], dict[str, Any]] | None:
        before = _proven_sidecar(model_id)
        if before is None:
            return None
        try:
            rows, after = load_validated_neutral_cache(
                model_id, requested_layers=layers,
            )
        except (OSError, RuntimeError, ValueError, TypeError, KeyError):
            return None
        if neutral_cache_identity(before) != neutral_cache_identity(after):
            return None
        return rows, after

    def _load_or_capture_rows(
        model_id: str, layers: set[int] | None,
    ) -> tuple[dict[int, Any], dict[str, Any]]:
        cached_rows = _load_proven_rows(model_id, layers)
        if cached_rows is not None:
            return cached_rows
        # Single-flight starts before model construction, not merely before the
        # neutral forward. Two directional transfers sharing a cold model cache
        # therefore incur one large model load as well as one capture.
        with neutral_fit_lock(model_id):
            cached_rows = _load_proven_rows(model_id, layers)
            if cached_rows is not None:
                return cached_rows
            with SaklasSession.from_pretrained(
                model_id, device="auto", probes=[],
            ) as session:
                all_rows, sidecar = (
                    _load_or_compute_neutral_activations_with_metadata_locked(
                        session.model, session.tokenizer, session.layers,
                        model_id=model_id, force=False,
                    )
                )
            if layers is None:
                return all_rows, sidecar
            return (
                {layer: value for layer, value in all_rows.items() if layer in layers},
                sidecar,
            )

    src_sidecar = _proven_sidecar(src_model)
    tgt_sidecar = _proven_sidecar(tgt_model)
    src_seed_rows: dict[int, Any] | None = None
    tgt_seed_rows: dict[int, Any] | None = None
    if src_sidecar is None:
        src_seed_rows, src_sidecar = _load_or_capture_rows(src_model, None)
    if tgt_sidecar is None:
        tgt_seed_rows, tgt_sidecar = _load_or_capture_rows(tgt_model, None)

    src_identity = neutral_cache_identity(src_sidecar)
    tgt_identity = neutral_cache_identity(tgt_sidecar)
    expected_tgt_identity = tgt_identity
    available = set(int(layer) for layer in src_sidecar.get("layers", [])) & set(
        int(layer) for layer in tgt_sidecar.get("layers", [])
    )
    wanted = available if wanted_arg is None else (available & wanted_arg)
    if not wanted:
        sys.stderr.write(
            f"{label}: source and target have no requested shared layers\n"
        )
        sys.exit(1)

    # A cold cache fill necessarily captures the model's full neutral roster,
    # but a narrow transfer must not retain every layer from both models for
    # the rest of the transaction. Rebinding here drops unrequested tensor
    # owners as soon as shared coverage is known.
    if src_seed_rows is not None:
        src_seed_rows = {layer: src_seed_rows[layer] for layer in wanted}
    if tgt_seed_rows is not None:
        tgt_seed_rows = {layer: tgt_seed_rows[layer] for layer in wanted}

    cached = load_alignment_map(
        src_model, tgt_model,
        source_identity=src_identity, target_identity=tgt_identity,
        requested_layers=(set() if force else wanted),
    )
    cached_M: dict[int, Any] = {}
    cached_sidecar: dict[str, Any] = {}
    if cached is not None:
        cached_M, cached_sidecar = cached
    complete_hit = not force and set(cached_M) >= wanted
    if complete_hit:
        # The cached map proves the source side; only the target rows are still
        # needed to reconstruct the exact target metric.
        src_seed_rows = None

    # Target rows are always needed for the exact target-metric re-bake, but
    # only for layers the source profile can actually transfer.
    tgt_loaded = (
        ({layer: tgt_seed_rows[layer] for layer in wanted}, tgt_sidecar)
        if tgt_seed_rows is not None else _load_proven_rows(tgt_model, wanted)
    )
    if tgt_loaded is None:
        tgt_acts, tgt_sidecar = _load_or_capture_rows(tgt_model, wanted)
    else:
        tgt_acts, tgt_sidecar = tgt_loaded
    tgt_seed_rows = None
    if neutral_cache_identity(tgt_sidecar) != expected_tgt_identity:
        # A cache writer raced the metadata preflight. Restart this directional
        # transaction under the outer alignment fit lock with the new identity.
        return _load_or_fit_transfer_alignment_locked(
            src_model, tgt_model, force=force, label=label,
            requested_layers=requested_layers,
        )

    raw_q = cached_sidecar.get("quality_per_layer") or {}
    quality_per_layer = {int(k): float(v) for k, v in raw_q.items()}
    map_path, _ = alignment_cache_path(src_model, tgt_model)
    if complete_hit:
        selected_quality = {
            layer: quality_per_layer[layer]
            for layer in sorted(wanted) if layer in quality_per_layer
        }
        return (
            {layer: cached_M[layer] for layer in sorted(wanted)},
            selected_quality, map_path, src_identity, expected_tgt_identity,
            _target_whitener_from_neutral_activations(tgt_acts),
            {layer: tensor.mean(dim=0) for layer, tensor in tgt_acts.items()},
        )

    missing = wanted if force else (wanted - set(cached_M))
    src_loaded = (
        ({layer: src_seed_rows[layer] for layer in missing}, src_sidecar)
        if src_seed_rows is not None else _load_proven_rows(src_model, missing)
    )
    if src_loaded is None:
        src_acts, src_sidecar = _load_or_capture_rows(src_model, missing)
    else:
        src_acts, src_sidecar = src_loaded
    src_seed_rows = None
    if neutral_cache_identity(src_sidecar) != src_identity:
        return _load_or_fit_transfer_alignment_locked(
            src_model, tgt_model, force=force, label=label,
            requested_layers=requested_layers,
        )
    tgt_missing = {layer: tgt_acts[layer] for layer in missing}
    try:
        fitted = fit_alignment(
            src_acts, tgt_missing,
            requested_layers=missing,
            available_shared_layers=available,
        )
    except AlignmentError as e:
        sys.stderr.write(f"{label}: {e}\n")
        sys.exit(1)
    fitted_quality = alignment_quality(fitted, src_acts, tgt_missing)
    quality_per_layer.update(fitted_quality)
    map_path = save_alignment_map(
        fitted, src_model, tgt_model,
        source_identity=src_identity, target_identity=tgt_identity,
        quality_per_layer=fitted_quality, extend=bool(cached_sidecar),
    )
    result_M = dict(cached_M)
    result_M.update(fitted)
    selected_quality = {
        layer: quality_per_layer[layer]
        for layer in sorted(wanted) if layer in quality_per_layer
    }
    return (
        {layer: result_M[layer] for layer in sorted(wanted)},
        selected_quality, map_path, src_identity, expected_tgt_identity,
        _target_whitener_from_neutral_activations(tgt_acts),
        {layer: tensor.mean(dim=0) for layer, tensor in tgt_acts.items()},
    )
