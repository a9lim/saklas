"""Profile: the ergonomic wrapper around a baked steering-vector dict.

Wire format stays identical to what :mod:`saklas.vectors` writes today —
a safetensors file with one tensor per active layer plus a slim JSON
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
import logging
import math
import pathlib
from typing import Any, Iterable, Iterator, Literal, Mapping, overload

import torch
from safetensors.torch import load_file, save_file

from saklas.core.errors import SaklasError

log = logging.getLogger(__name__)


class ProfileError(ValueError, SaklasError):
    """Raised on invalid Profile operations (missing layer, empty, etc.)."""

    def user_message(self) -> tuple[int, str]:
        return (400, str(self) or self.__class__.__name__)


def save_profile(
    profile: Mapping[int, torch.Tensor],
    path: str | pathlib.Path,
    metadata: dict[str, Any],
) -> None:
    """Save a baked vector profile as .safetensors with a slim .json sidecar.

    ``metadata`` must contain at minimum:
        method            - str, e.g. "merge" / "procrustes_transfer" /
                            "neutral_activations" / "gguf_import"

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

    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # fp32 write invariant: every saklas safetensor writer enforces fp32
    # on disk (matches gguf_io.py's ``.to(dtype=torch.float32)``), so the
    # stored dtype is a guarantee rather than a coincidence of the caller.
    tensors = {
        f"layer_{idx}": vec.to(dtype=torch.float32).contiguous().cpu()
        for idx, vec in profile.items()
    }
    save_file(tensors, str(path))

    from saklas import __version__ as _saklas_version
    from saklas.io.packs import PROFILE_FORMAT_VERSION

    sidecar: dict[str, Any] = {
        "format_version": PROFILE_FORMAT_VERSION,
        "method": method,
        "saklas_version": _saklas_version,
    }
    if "statements_sha256" in metadata:
        sidecar["statements_sha256"] = metadata["statements_sha256"]
    if "components" in metadata:
        sidecar["components"] = metadata["components"]
    # The optional bake method records which scoring metric drove share
    # allocation. Loaders read this only for diagnostics; tensor magnitudes
    # already carry the runtime weights.
    if "bake" in metadata:
        sidecar["bake"] = metadata["bake"]
    # SAE provenance — present only when extraction used an SAE backend.
    for key in ("sae_release", "sae_revision", "sae_ids_by_layer"):
        if key in metadata:
            sidecar[key] = metadata[key]
    # Transfer provenance — present only on transferred profiles
    # (method="procrustes_transfer").  ``alignment_map_hash`` pins the
    # specific Procrustes fit; ``transfer_quality_estimate`` is the
    # median per-layer R² across shared layers.
    for key in (
        "source_model_id",
        "alignment_map_hash",
        "transfer_quality_estimate",
    ):
        if key in metadata:
            sidecar[key] = metadata[key]
    # Diagnostics: stringify layer keys so the JSON round-trips through
    # standard parsers (JSON object keys must be strings).  Reader inverts.
    diagnostics = metadata.get("diagnostics")
    if diagnostics:
        sidecar["diagnostics_by_layer"] = {
            str(layer): {k: float(v) for k, v in metrics.items()}
            for layer, metrics in diagnostics.items()
        }

    from saklas.io.atomic import write_json_atomic

    meta_path = path.with_suffix(".json")
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

    tensors = load_file(str(path))
    meta_path = path.with_suffix(".json")
    with open(meta_path) as f:
        metadata = json.load(f)

    from saklas.io.packs import PROFILE_FORMAT_VERSION

    if not isinstance(metadata, dict):
        raise ProfileError(f"profile sidecar {meta_path} must be a JSON object")
    fmt_ver = metadata.get("format_version")
    if (
        not isinstance(fmt_ver, int)
        or isinstance(fmt_ver, bool)
        or fmt_ver != PROFILE_FORMAT_VERSION
    ):
        raise ProfileError(
            f"profile sidecar {meta_path} has format_version={fmt_ver!r}; "
            f"need exactly {PROFILE_FORMAT_VERSION}. Regenerate it with "
            "the current saklas"
        )
    if not isinstance(metadata.get("method"), str) or not metadata["method"]:
        raise ProfileError(
            f"profile sidecar {meta_path} requires a non-empty 'method' string"
        )

    profile = {int(key.split("_", 1)[1]): tensor for key, tensor in tensors.items()}

    # Invert the layer-key stringification done at save time so diagnostics
    # are addressable by ``int`` consistently with the profile dict.
    raw_diag = metadata.get("diagnostics_by_layer")
    if raw_diag is not None:
        if not isinstance(raw_diag, dict):
            raise ProfileError(
                f"profile sidecar {meta_path} diagnostics_by_layer must be an object"
            )
        try:
            metadata["diagnostics"] = {
                int(layer): {
                    str(name): float(value)
                    for name, value in metrics.items()
                }
                for layer, metrics in raw_diag.items()
                if isinstance(metrics, dict)
            }
        except (TypeError, ValueError) as exc:
            raise ProfileError(
                f"profile sidecar {meta_path} has malformed diagnostics_by_layer"
            ) from exc
        if len(metadata["diagnostics"]) != len(raw_diag):
            raise ProfileError(
                f"profile sidecar {meta_path} has malformed diagnostics_by_layer"
            )

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
