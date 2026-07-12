"""Profile: the ergonomic wrapper around a baked steering-vector dict.

The native wire format is a safetensors file with one fp32 vector per active
layer plus an exact-version JSON
sidecar (safetensors path) or a llama.cpp control-vector GGUF (gguf path).
This class is purely the Python-level surface the rest of saklas uses so
that callers stop passing bare ``dict[int, Tensor]`` around.

The underlying tensors are "baked": the per-layer Mahalanobis share is
folded into the magnitude at extraction time (see
``extraction.ManifoldExtractionPipeline`` /
``vectors.fold_directions_to_subspace``).  A ``Profile`` is just a thin
wrapper; the dict stays the canonical shape at rest and the unified
subspace kernel reads the baked magnitudes as per-layer weights.
"""

from __future__ import annotations

import json
import hashlib
import logging
import math
import pathlib
import re
from typing import Any, Iterable, Iterator, Literal, Mapping, overload

import torch
from safetensors.torch import load as load_safetensors, save as save_safetensors

from saklas.core.errors import SaklasError

log = logging.getLogger(__name__)

_PROFILE_SIDECAR_FIELDS = {
    "format_version", "method", "saklas_version", "statements_sha256",
    "components", "bake", "sae_release", "sae_revision",
    "sae_ids_by_layer", "source_model_id", "alignment_map_hash",
    "transfer_quality_estimate", "diagnostics_by_layer",
    "tensor_sha256",
}
_PROFILE_METADATA_FIELDS = _PROFILE_SIDECAR_FIELDS - {
    "format_version", "saklas_version", "diagnostics_by_layer", "tensor_sha256",
} | {"diagnostics"}

_PROFILE_METHODS = frozenset({
    "profile", "merge", "contrastive_pca", "procrustes_transfer",
})


class ProfileError(ValueError, SaklasError):
    """Raised on invalid Profile operations (missing layer, empty, etc.)."""

    def user_message(self) -> tuple[int, str]:
        return (400, str(self) or self.__class__.__name__)


def _validate_profile_sidecar(
    data: Any, *, tensor_sha256: str | None = None,
    layers: set[int] | None = None,
) -> None:
    if not isinstance(data, dict) or set(data) != _PROFILE_SIDECAR_FIELDS:
        raise ProfileError("profile sidecar does not match the current exact schema")
    method = data["method"]
    if method not in _PROFILE_METHODS:
        raise ProfileError(f"profile sidecar has invalid method {method!r}")
    from saklas.io.packs import PROFILE_FORMAT_VERSION

    if (
        isinstance(data["format_version"], bool)
        or data["format_version"] != PROFILE_FORMAT_VERSION
        or not isinstance(data["saklas_version"], str)
        or not data["saklas_version"]
    ):
        raise ProfileError("profile sidecar has invalid format identity")
    digest = data["tensor_sha256"]
    if (
        not isinstance(digest, str) or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
        or tensor_sha256 is not None and digest != tensor_sha256
    ):
        raise ProfileError("profile sidecar has invalid tensor sha256")
    diag = data["diagnostics_by_layer"]
    if diag is not None:
        if not isinstance(diag, dict):
            raise ProfileError("profile diagnostics_by_layer must be an object")
        for layer, metrics in diag.items():
            if not re.fullmatch(r"0|[1-9][0-9]*", layer) or not isinstance(metrics, dict):
                raise ProfileError("profile diagnostics_by_layer has invalid layer schema")
            if any(
                not isinstance(name, str) or not name
                or isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for name, value in metrics.items()
            ):
                raise ProfileError("profile diagnostics_by_layer has invalid metrics")
        if layers is not None and {int(layer) for layer in diag} != layers:
            raise ProfileError("profile diagnostics_by_layer does not cover tensor layers")
    sae_ids = data["sae_ids_by_layer"]
    if sae_ids is not None and (
        not isinstance(sae_ids, dict) or any(
            not isinstance(key, str) or not re.fullmatch(r"0|[1-9][0-9]*", key)
            or not isinstance(value, str) or not value
            for key, value in sae_ids.items()
        )
    ):
        raise ProfileError("profile sae_ids_by_layer is invalid")
    if sae_ids is not None and layers is not None and {
        int(layer) for layer in sae_ids
    } != layers:
        raise ProfileError("profile sae_ids_by_layer does not cover tensor layers")
    components = data["components"]
    if components is not None:
        if not isinstance(components, dict) or not components:
            raise ProfileError("profile components must be a non-empty object")
        for key, row in components.items():
            if (
                not isinstance(key, str) or not key or not isinstance(row, dict)
                or set(row) != {"selector", "alpha", "tensor_sha256"}
                or not isinstance(row["selector"], str) or not row["selector"]
                or isinstance(row["alpha"], bool)
                or not isinstance(row["alpha"], (int, float))
                or not math.isfinite(float(row["alpha"]))
                or not isinstance(row["tensor_sha256"], str)
                or not re.fullmatch(r"[0-9a-f]{64}", row["tensor_sha256"])
            ):
                raise ProfileError("profile components has invalid provenance")
    if method == "merge" and components is None:
        raise ProfileError("merge profile requires component provenance")
    if method != "merge" and components is not None:
        raise ProfileError("non-merge profile carries component provenance")
    nullable_strings = {
        "statements_sha256", "bake", "sae_release", "sae_revision",
        "source_model_id", "alignment_map_hash",
    }
    if any(
        data[key] is not None and (
            not isinstance(data[key], str) or not data[key]
        ) for key in nullable_strings
    ):
        raise ProfileError("profile sidecar has invalid string provenance")
    for key in ("statements_sha256", "alignment_map_hash"):
        value = data[key]
        if value is not None and not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ProfileError(f"profile sidecar field {key!r} is not a sha256")
    transfer_fields = (
        data["source_model_id"], data["alignment_map_hash"],
        data["transfer_quality_estimate"],
    )
    if method == "procrustes_transfer":
        if transfer_fields[0] is None or transfer_fields[1] is None:
            raise ProfileError("transfer profile has incomplete provenance")
    elif any(value is not None for value in transfer_fields):
        raise ProfileError("non-transfer profile carries transfer provenance")
    if data["transfer_quality_estimate"] is not None and (
        isinstance(data["transfer_quality_estimate"], bool)
        or not isinstance(data["transfer_quality_estimate"], (int, float))
        or not math.isfinite(float(data["transfer_quality_estimate"]))
    ):
        raise ProfileError("profile transfer quality must be finite")
    sae_present = any(
        data[key] is not None for key in ("sae_release", "sae_revision", "sae_ids_by_layer")
    )
    if sae_present and (
        data["sae_release"] is None or data["sae_ids_by_layer"] is None
    ):
        raise ProfileError("profile has incomplete SAE provenance")


def save_profile(
    profile: Mapping[int, torch.Tensor],
    path: str | pathlib.Path,
    metadata: dict[str, Any],
) -> None:
    """Save a baked vector profile as .safetensors with a slim .json sidecar.

    ``metadata`` must contain at minimum:
        method            - one of the current safetensor profile producers:
                            "profile" / "merge" / "contrastive_pca" /
                            "procrustes_transfer"

    Optional keys honored:
        statements_sha256 - str, hash of the source neutral corpus at write time
        components        - dict, merge provenance (method="merge" only)
        diagnostics       - dict[int, dict[str, float]], per-layer probe-quality
                            metrics (see extraction diagnostics).
                            Persisted as ``diagnostics_by_layer`` on the
                            sidecar with stringified layer keys.

    The safetensors file contains keys ``"layer_{i}"`` for each active layer.
    Tensors are already baked (share pre-multiplied into magnitude) — the
    sidecar carries only method/saklas_version plus the optional fields above.
    """
    method = metadata.get("method")
    if not isinstance(method, str) or not method:
        raise ProfileError("profile metadata requires a non-empty 'method' string")
    unknown_metadata = set(metadata) - _PROFILE_METADATA_FIELDS
    if unknown_metadata:
        raise ProfileError(
            f"profile metadata has unknown field(s): {sorted(unknown_metadata)}"
        )
    if not profile:
        raise ProfileError("profile requires at least one layer tensor")
    for layer, tensor in profile.items():
        layer_value: Any = layer
        if (
            isinstance(layer_value, bool)
            or not isinstance(layer_value, int)
            or layer_value < 0
        ):
            raise ProfileError(f"profile layer must be a non-negative int: {layer!r}")
        if tensor.ndim != 1 or tensor.numel() == 0:
            raise ProfileError(f"profile layer {layer} must be a non-empty rank-1 tensor")
        if not bool(torch.isfinite(tensor).all().item()):
            raise ProfileError(f"profile layer {layer} must contain only finite values")

    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # fp32 write invariant: every saklas safetensor writer enforces fp32
    # on disk (matches gguf_io.py's ``.to(dtype=torch.float32)``), so the
    # stored dtype is a guarantee rather than a coincidence of the caller.
    tensors = {
        f"layer_{idx}": vec.to(dtype=torch.float32).contiguous().cpu()
        for idx, vec in profile.items()
    }
    tensor_bytes = save_safetensors(tensors)
    tensor_digest = hashlib.sha256(tensor_bytes).hexdigest()

    from saklas import __version__ as _saklas_version
    from saklas.io.packs import PROFILE_FORMAT_VERSION

    sidecar: dict[str, Any] = {
        "format_version": PROFILE_FORMAT_VERSION,
        "method": method,
        "saklas_version": _saklas_version,
        "statements_sha256": metadata.get("statements_sha256"),
        "components": metadata.get("components"),
        "bake": metadata.get("bake"),
        "sae_release": metadata.get("sae_release"),
        "sae_revision": metadata.get("sae_revision"),
        "sae_ids_by_layer": metadata.get("sae_ids_by_layer"),
        "source_model_id": metadata.get("source_model_id"),
        "alignment_map_hash": metadata.get("alignment_map_hash"),
        "transfer_quality_estimate": metadata.get("transfer_quality_estimate"),
        "diagnostics_by_layer": None,
        "tensor_sha256": tensor_digest,
    }
    # Diagnostics: stringify layer keys so the JSON round-trips through
    # standard parsers (JSON object keys must be strings).  Reader inverts.
    diagnostics = metadata.get("diagnostics")
    if diagnostics is not None:
        if not isinstance(diagnostics, dict):
            raise ProfileError("profile metadata 'diagnostics' must be an object")
        sidecar["diagnostics_by_layer"] = {
            str(layer): dict(metrics)
            for layer, metrics in diagnostics.items()
        }

    _validate_profile_sidecar(
        sidecar, tensor_sha256=tensor_digest, layers=set(profile),
    )
    from saklas.io.atomic import artifact_lock, write_bytes_atomic, write_json_atomic

    meta_path = path.with_suffix(".json")
    with artifact_lock(path):
        write_bytes_atomic(path, tensor_bytes)
        write_json_atomic(meta_path, sidecar)

    log.info("Saved profile (%d layers) to %s", len(profile), path)


def load_profile(path: str | pathlib.Path) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
    """Load a baked vector profile and its metadata.

    Dispatches on file extension: ``.safetensors`` reads the companion
    ``.json`` sidecar; ``.gguf`` reads the control-vector metadata embedded
    in the GGUF header (see :mod:`saklas.io.gguf_io`). Both paths yield the
    same ``(profile, metadata)`` shape — callers don't need to branch.
    """
    path = pathlib.Path(path)
    if path.suffix == ".gguf":
        from saklas.io.gguf_io import read_gguf_profile

        return read_gguf_profile(path)

    meta_path = path.with_suffix(".json")
    from saklas.io.atomic import artifact_lock

    with artifact_lock(path):
        tensor_bytes = path.read_bytes()
        with open(meta_path) as f:
            metadata = json.load(f)
    tensor_digest = hashlib.sha256(tensor_bytes).hexdigest()
    tensors = load_safetensors(tensor_bytes)


    if not tensors:
        raise ProfileError(f"profile tensor {path} has no layers")
    profile: dict[int, torch.Tensor] = {}
    for key, tensor in tensors.items():
        if not re.fullmatch(r"layer_(0|[1-9][0-9]*)", key):
            raise ProfileError(f"profile tensor {path} has invalid key {key!r}")
        layer = int(key.removeprefix("layer_"))
        if tensor.dtype != torch.float32 or tensor.ndim != 1 or tensor.numel() == 0:
            raise ProfileError(
                f"profile tensor {path} layer {layer} must be non-empty rank-1 fp32"
            )
        if not bool(torch.isfinite(tensor).all().item()):
            raise ProfileError(f"profile tensor {path} layer {layer} is non-finite")
        profile[layer] = tensor
    _validate_profile_sidecar(
        metadata, tensor_sha256=tensor_digest, layers=set(profile),
    )

    # Invert the layer-key stringification done at save time so diagnostics
    # are addressable by ``int`` consistently with the profile dict.
    raw_diag = metadata["diagnostics_by_layer"]
    if raw_diag is not None:
        metadata["diagnostics"] = {
            int(layer): dict(metrics) for layer, metrics in raw_diag.items()
        }

    return profile, metadata


class Profile:
    """Steering direction set: one baked tensor per transformer layer.

    Wraps ``dict[int, torch.Tensor]`` with the same mapping interface
    (``__getitem__``, ``items``, ``keys``, ``values``, ``__iter__``,
    ``__len__``, ``__contains__``) plus a typed public surface
    (``layers``, ``metadata``, ``weight_at``, ``save``/``load``,
    ``merged``, ``projected_away``, ``cosine_similarity``).
    """

    __slots__ = ("_metadata", "_tensors")

    def __init__(
        self,
        tensors: Mapping[int, torch.Tensor],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        tensors_in: Any = tensors
        if not isinstance(tensors_in, Mapping):
            raise ProfileError(
                f"Profile(tensors) must be a mapping, got {type(tensors).__name__}"
            )
        if not tensors:
            raise ProfileError("Profile must contain at least one layer")
        out: dict[int, torch.Tensor] = {}
        ref_dtype: torch.dtype | None = None
        for layer, t in tensors.items():
            layer_in: Any = layer
            tensor_in: Any = t
            if not isinstance(layer_in, int):
                raise ProfileError(
                    f"Profile layer key must be int, got {type(layer).__name__}"
                )
            if not isinstance(tensor_in, torch.Tensor):
                raise ProfileError(
                    f"Profile value at layer {layer} must be torch.Tensor, "
                    f"got {type(t).__name__}"
                )
            if ref_dtype is None:
                ref_dtype = t.dtype
            out[layer] = t
        self._tensors: dict[int, torch.Tensor] = out
        self._metadata: dict[str, Any] = dict(metadata or {})

    # Mapping surface -----------------------------------------------------

    def __getitem__(self, layer: int) -> torch.Tensor:
        return self._tensors[layer]

    def __iter__(self) -> Iterator[int]:
        return iter(self._tensors)

    def __len__(self) -> int:
        return len(self._tensors)

    def __contains__(self, layer: object) -> bool:
        return layer in self._tensors

    def items(self):
        return self._tensors.items()

    def keys(self):
        return self._tensors.keys()

    def values(self):
        return self._tensors.values()

    # Public surface ------------------------------------------------------

    @property
    def layers(self) -> list[int]:
        """Sorted list of layer indices present in this profile."""
        return sorted(self._tensors.keys())

    @property
    def metadata(self) -> dict[str, Any]:
        """Copy of the metadata dict carried alongside the tensors."""
        return dict(self._metadata)

    @property
    def diagnostics(self) -> dict[int, dict[str, float]] | None:
        """Per-layer probe-quality metrics, when available.

        Keys are the same layer indices the profile carries; values are
        small dicts of metric-name → float (``evr``,
        ``intra_pair_variance_mean`` / ``_std``, ``inter_pair_alignment``,
        ``diff_principal_projection``).  Returns ``None`` when the profile
        was extracted before diagnostics existed (saklas < 1.6) or loaded
        from a sidecar that didn't carry them — callers should branch on
        ``has_diagnostics`` first.
        """
        diag = self._metadata.get("diagnostics")
        if not isinstance(diag, dict) or not diag:
            return None
        # Defensive copy: callers shouldn't be able to mutate the cached
        # metric dicts through this surface.
        return {int(L): dict(metrics) for L, metrics in diag.items()}

    @property
    def has_diagnostics(self) -> bool:
        """True iff this profile carries per-layer diagnostic metrics."""
        diag = self._metadata.get("diagnostics")
        return isinstance(diag, dict) and bool(diag)

    def as_dict(self) -> dict[int, torch.Tensor]:
        """Return the underlying dict (shared reference, not a copy).

        Internal helper for call sites that work on the raw tensor dict
        (hooks, merge.linear_sum, monitor). Do not mutate.
        """
        return self._tensors

    def weight_at(self, layer: int) -> torch.Tensor:
        """Return the baked direction at ``layer``; raise ProfileError if missing."""
        try:
            return self._tensors[layer]
        except KeyError as e:
            raise ProfileError(
                f"Profile has no tensor for layer {layer}; "
                f"available: {self.layers}"
            ) from e

    def save(
        self,
        path: str | pathlib.Path,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save as safetensors + slim JSON sidecar.

        Metadata passed here overrides / augments the profile's own
        ``self.metadata``; the sidecar carries the current
        ``PROFILE_FORMAT_VERSION``.
        """
        merged: dict[str, Any] = dict(self._metadata)
        if metadata:
            merged.update(metadata)
        merged.setdefault("method", "profile")
        save_profile(self._tensors, path, merged)

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "Profile":
        """Load from safetensors (+ sidecar) or gguf.

        Dispatches on file extension. Safetensors sidecars with a
        any ``format_version`` other than ``PROFILE_FORMAT_VERSION`` raise
        :class:`ProfileError`. GGUF
        files carry metadata in-header and are exempt from the
        format_version gate.
        """
        tensors, meta = load_profile(path)
        return cls(tensors, metadata=meta)

    def to_gguf(self, path: str | pathlib.Path, *, model_hint: str) -> None:
        """Write as llama.cpp control-vector GGUF.

        Baked share/ref_norm magnitudes carry through unchanged — llama.cpp's
        uniform ``--control-vector-scaled`` scalar reproduces saklas's
        per-layer weighting without needing a per-layer metadata slot.
        """
        from saklas.io.gguf_io import write_gguf_profile

        write_gguf_profile(self._tensors, path, model_hint=model_hint)

    @classmethod
    def merged(
        cls,
        components: Iterable[tuple["Profile", float]],
        *,
        strict: bool = False,
    ) -> "Profile":
        """Linear combination: ``sum(alpha_i * profile_i)`` per layer.

        Delegates to :func:`saklas.io.merge.linear_sum`. Layer coverage is the
        union, matching live expression composition (an absent term contributes
        zero); ``strict=True`` requires identical coverage.
        """
        from saklas.io.merge import linear_sum

        pairs = [(p.as_dict(), float(a)) for p, a in components]
        if len(pairs) < 2:
            raise ProfileError("Profile.merged requires at least two components")
        merged_dict = linear_sum(pairs, strict=strict)
        return cls(merged_dict, metadata={"method": "merge"})

    def merged_with(
        self,
        other: "Profile",
        *,
        weights: tuple[float, float] = (1.0, 1.0),
        strict: bool = False,
    ) -> "Profile":
        """Binary merge convenience wrapping :meth:`merged`."""
        return type(self).merged(
            [(self, weights[0]), (other, weights[1])], strict=strict,
        )

    def promoted_to(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "Profile":
        """Return a new Profile with tensors cast to ``device``/``dtype``.

        No-op layers (already matching) are reused by reference. The
        current instance is never mutated.
        """
        if device is None and dtype is None:
            return self
        target_device = torch.device(device) if device is not None else None
        out: dict[int, torch.Tensor] = {}
        for idx, t in self._tensors.items():
            dev_ok = target_device is None or t.device == target_device
            dt_ok = dtype is None or t.dtype == dtype
            if dev_ok and dt_ok:
                out[idx] = t
            else:
                out[idx] = t.to(
                    device=target_device if target_device is not None else t.device,
                    dtype=dtype if dtype is not None else t.dtype,
                )
        return type(self)(out, metadata=self._metadata)

    def projected_away(self, other: "Profile") -> "Profile":
        """Return a new Profile with *other*'s direction projected out, per layer.

        Per-layer math (fp32)::

            result_L = self_L - (dot(self_L, other_L) / dot(other_L, other_L)) * other_L

        Only layers present in *both* profiles are projected; layers in
        ``self`` but not ``other`` are included unchanged.  Near-zero
        ``dot(other_L, other_L) < 1e-12`` layers are copied unchanged.

        Raises :class:`ProfileError` when no layers are shared between
        ``self`` and ``other``.
        """
        shared = set(self._tensors) & set(other._tensors)
        if not shared:
            raise ProfileError(
                "projected_away: no shared layers between the two profiles"
            )
        out: dict[int, torch.Tensor] = {}
        for layer, a_t in self._tensors.items():
            if layer not in other._tensors:
                out[layer] = a_t
                continue
            a_f = a_t.to(dtype=torch.float32)
            b_f = other._tensors[layer].to(dtype=torch.float32)
            b_dot = torch.dot(b_f, b_f).item()
            if b_dot < 1e-12:
                out[layer] = a_t
            else:
                proj = (torch.dot(a_f, b_f) / b_dot) * b_f
                out[layer] = (a_f - proj).to(dtype=a_t.dtype)
        return type(self)(out, metadata=self._metadata)

    @overload
    def cosine_similarity(
        self, other: "Profile", *, per_layer: Literal[False] = ...,
        whitener: "Any | None" = ...,
    ) -> float: ...
    @overload
    def cosine_similarity(
        self, other: "Profile", *, per_layer: Literal[True],
        whitener: "Any | None" = ...,
    ) -> dict[int, float]: ...

    def cosine_similarity(
        self,
        other: "Profile",
        *,
        per_layer: bool = False,
        whitener: "Any | None" = None,
    ) -> "float | dict[int, float]":
        """Cosine similarity against *other* (Mahalanobis only).

        **Aggregate** (default): Mahalanobis-norm-weighted cosine over the
        layer intersection.  Weighting matches the monitor regime.

        **Per-layer** (``per_layer=True``): Mahalanobis cosine per shared
        layer.

        The metric is Mahalanobis cosine ``<u, v>_M = u^T Σ^{-1} v`` —
        predicts cross-domain probe generalization better than plain
        cosine on activation distributions with strongly anisotropic
        covariance.

        The ``whitener`` (a :class:`LayerWhitener`) is **required** and
        must cover *every* shared layer, via
        :meth:`LayerWhitener.covers_all`.  There is no Euclidean path: a
        missing or non-covering whitener raises :class:`WhitenerError`.

        Raises :class:`ProfileError` when no layers are shared.
        """
        from saklas.core.mahalanobis import WhitenerError

        shared = sorted(set(self._tensors) & set(other._tensors))
        if not shared:
            raise ProfileError(
                "cosine_similarity: no shared layers between the two profiles"
            )

        # Cross-device pairs (e.g. an actively-steered profile hooked on
        # the model device against a disk-loaded peer on CPU) would crash
        # the dot below; resolve to CPU once and reuse for both code paths.
        # CPU is also where LayerWhitener applies its Woodbury factors.
        def _aligned(a_t: torch.Tensor, b_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return a_t.float().cpu(), b_t.float().cpu()

        # Mahalanobis-only: the whitener must cover every shared layer.
        # No Euclidean fallback — a missing or partial whitener is an error.
        if whitener is None or not whitener.covers_all(shared):
            raise WhitenerError(
                "cosine_similarity requires a Mahalanobis whitener covering "
                f"every shared layer {shared}; regenerate the neutral "
                "activation cache for this model (the Euclidean path is gone)"
            )

        if per_layer:
            out: dict[int, float] = {}
            for L in shared:
                a, b = _aligned(self._tensors[L], other._tensors[L])
                out[L] = whitener.mahalanobis_cosine(L, a, b)
            return out

        # Mahalanobis-norm-weighted aggregate: directions whose typical
        # activations don't cover them dominate the average less, mirroring
        # the monitor regime.
        num = 0.0
        den = 0.0
        for L in shared:
            a, b = _aligned(self._tensors[L], other._tensors[L])
            si_a = whitener.apply_inv(L, a).float()
            si_b = whitener.apply_inv(L, b).float()
            aa = max(float(torch.dot(a.reshape(-1), si_a.reshape(-1)).item()), 0.0)
            bb = max(float(torch.dot(b.reshape(-1), si_b.reshape(-1)).item()), 0.0)
            if aa < 1e-12 or bb < 1e-12:
                continue
            num += float(torch.dot(a.reshape(-1), si_b.reshape(-1)).item())
            den += math.sqrt(aa * bb)
        if den < 1e-12:
            raise ProfileError(
                "cosine_similarity: every shared layer has near-zero "
                "magnitude under the requested metric"
            )
        return num / den

    def __repr__(self) -> str:
        layers = self.layers
        layer_desc = str(layers) if len(layers) <= 4 else f"[{layers[0]}..{layers[-1]}] ({len(layers)} layers)"
        first = next(iter(self._tensors.values()))
        return (
            f"Profile({layer_desc}, dtype={first.dtype}, "
            f"device={first.device})"
        )
