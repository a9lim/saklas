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

# ``ProfileError`` is defined in :mod:`saklas.core.errors` (the single home
# for the engine's error taxonomy) and re-exported here so
# ``from saklas.core.profile import ProfileError`` keeps working.
from saklas.core.errors import ProfileError as ProfileError

log = logging.getLogger(__name__)

# The exact on-disk sidecar schema.  Four identity fields plus one
# free-form ``provenance`` object; a producer that wants to record
# anything else puts it there rather than growing this set.
_PROFILE_SIDECAR_FIELDS = {
    "format_version", "saklas_version", "method", "tensor_sha256",
    "provenance",
}

# Writer-stamped identity.  ``load_profile`` returns these alongside the
# provenance so callers can read the format identity, and ``save_profile``
# ignores them on the way back in (re-stamping fresh values) — that is what
# makes ``Profile.load(p).save(q)`` a round-trip rather than a schema error.
_PROFILE_STAMPED_FIELDS = frozenset({
    "format_version", "saklas_version", "tensor_sha256",
})

# The live producers of a profile sidecar: a bare fold / hand-built
# ``Profile.save`` (``profile``), ``SaklasSession.extract``'s folded 2-node
# manifold view (``manifold_pca``), and ``Profile.merged`` (``merge``).
_PROFILE_METHODS = frozenset({"profile", "manifold_pca", "merge"})

# Provenance leaf types.  JSON-safe scalars only, plus lists/objects of
# them; object keys must be strings so the blob round-trips byte-identically
# through ``json``.
_PROVENANCE_SCALARS = (str, int, float)
_PROVENANCE_MAX_DEPTH = 8


def _validate_provenance(value: Any, *, depth: int = 0) -> None:
    """Reject anything that would not survive a JSON round-trip.

    The provenance blob is free-form by design — the schema deliberately
    does not enumerate producer keys — but it still has to come back
    byte-identical, so object keys must be strings, floats must be finite
    (``NaN``/``Infinity`` are not JSON), and the nesting has to terminate.
    """
    if depth > _PROVENANCE_MAX_DEPTH:
        raise ProfileError("profile provenance nests too deeply")
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, _PROVENANCE_SCALARS):
        if isinstance(value, float) and not math.isfinite(value):
            raise ProfileError("profile provenance floats must be finite")
        return
    if isinstance(value, list):
        for item in value:
            _validate_provenance(item, depth=depth + 1)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ProfileError(
                    "profile provenance object keys must be strings; "
                    f"got {type(key).__name__}"
                )
            _validate_provenance(item, depth=depth + 1)
        return
    raise ProfileError(
        f"profile provenance holds a non-JSON value of type {type(value).__name__}"
    )


def _validate_profile_sidecar(
    data: Any, *, tensor_sha256: str | None = None,
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
    provenance = data["provenance"]
    if not isinstance(provenance, dict):
        raise ProfileError("profile provenance must be an object")
    _validate_provenance(provenance)


def save_profile(
    profile: Mapping[int, torch.Tensor],
    path: str | pathlib.Path,
    metadata: dict[str, Any],
) -> None:
    """Save a baked vector profile as .safetensors with a slim .json sidecar.

    ``metadata["method"]`` names the producer and must be one of
    :data:`_PROFILE_METHODS` — ``"profile"`` (a hand-built or folded
    profile), ``"manifold_pca"`` (the folded 2-node manifold
    :meth:`SaklasSession.extract` returns) or ``"merge"``
    (:meth:`Profile.merged`).

    Every *other* metadata key lands verbatim in the sidecar's free-form
    ``provenance`` object; it only has to be JSON-safe (see
    :func:`_validate_provenance`).  The writer-stamped identity fields
    (``format_version`` / ``saklas_version`` / ``tensor_sha256``) are
    ignored on input and re-stamped, so metadata handed back by
    :func:`load_profile` saves again unchanged.

    The safetensors file contains keys ``"layer_{i}"`` for each active layer.
    Tensors are already baked (share pre-multiplied into magnitude).
    """
    method = metadata.get("method")
    if not isinstance(method, str) or not method:
        raise ProfileError("profile metadata requires a non-empty 'method' string")
    if method not in _PROFILE_METHODS:
        raise ProfileError(
            f"profile metadata has invalid method {method!r}; "
            f"expected one of {sorted(_PROFILE_METHODS)}"
        )
    provenance = {
        key: value for key, value in metadata.items()
        if key != "method" and key not in _PROFILE_STAMPED_FIELDS
    }
    _validate_provenance(provenance)
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
        "saklas_version": _saklas_version,
        "method": method,
        "tensor_sha256": tensor_digest,
        "provenance": provenance,
    }
    _validate_profile_sidecar(sidecar, tensor_sha256=tensor_digest)
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

    The returned metadata is the sidecar's identity fields plus the
    ``provenance`` blob flattened back to the top level — i.e. exactly the
    shape :func:`save_profile` accepts, so load → save round-trips.
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
    _validate_profile_sidecar(metadata, tensor_sha256=tensor_digest)

    # Flatten ``provenance`` back to the top level.  Producer keys are what
    # callers actually read, and the flat shape is what ``save_profile``
    # takes, so a loaded Profile saves again without translation.  The
    # nesting exists only on disk, to keep the wire schema exact.
    out_metadata: dict[str, Any] = {
        key: metadata[key] for key in ("format_version", "saklas_version", "method",
                                       "tensor_sha256")
    }
    out_metadata.update(metadata["provenance"])
    return profile, out_metadata


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
        """Per-layer fit-quality metrics carried in provenance, if any.

        Reads a ``diagnostics`` key off :attr:`metadata` — layer index →
        ``{metric_name: float}``.  Nothing in saklas writes one: the unified
        pipeline's diagnostics (``PcaDiagnostics`` / ``SpectralDiagnostics``)
        ride the *manifold* sidecar instead, a separate channel surfaced by
        ``manifold show`` / ``pack show``.  This surface therefore returns
        ``None`` for every profile saklas hands out and exists for callers
        that stash their own per-layer metrics; branch on
        :attr:`has_diagnostics` first.
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

        Delegates to :func:`saklas.io.bake.linear_sum`. Layer coverage is the
        union, matching live expression composition (an absent term contributes
        zero); ``strict=True`` requires identical coverage.
        """
        from saklas.io.bake import linear_sum

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
