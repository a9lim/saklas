"""Profile: the ergonomic wrapper around a baked steering-vector dict.

Wire format stays identical to what :mod:`saklas.vectors` writes today —
a safetensors file with one tensor per active layer plus a slim JSON
sidecar (safetensors path) or a llama.cpp control-vector GGUF (gguf path).
This class is purely the Python-level surface the rest of saklas uses so
that callers stop passing bare ``dict[int, Tensor]`` around.

The underlying tensors are "baked": share and reference norm are folded
into the magnitude at extraction time (see ``vectors.extract_contrastive``),
so the steering hook collapses to ``alpha * _STEER_GAIN * sum(baked)``.
A ``Profile`` is just a thin wrapper; the dict stays the canonical shape
at rest.
"""

from __future__ import annotations

import pathlib
from typing import Iterable, Iterator, Mapping

import torch

from saklas.core.errors import SaklasError


class ProfileError(ValueError, SaklasError):
    """Raised on invalid Profile operations (missing layer, empty, etc.)."""


class Profile:
    """Steering direction set: one baked tensor per transformer layer.

    Wraps ``dict[int, torch.Tensor]``. Dict-compat surface is intentional
    for the migration from v1.x — existing producers of bare dicts can be
    wrapped without touching their internals.
    """

    __slots__ = ("_tensors", "_metadata")

    def __init__(
        self,
        tensors: Mapping[int, torch.Tensor],
        *,
        metadata: dict | None = None,
    ) -> None:
        if not isinstance(tensors, Mapping):
            raise ProfileError(
                f"Profile(tensors) must be a mapping, got {type(tensors).__name__}"
            )
        if not tensors:
            raise ProfileError("Profile must contain at least one layer")
        out: dict[int, torch.Tensor] = {}
        ref_dtype: torch.dtype | None = None
        for layer, t in tensors.items():
            if not isinstance(layer, int):
                raise ProfileError(
                    f"Profile layer key must be int, got {type(layer).__name__}"
                )
            if not isinstance(t, torch.Tensor):
                raise ProfileError(
                    f"Profile value at layer {layer} must be torch.Tensor, "
                    f"got {type(t).__name__}"
                )
            if ref_dtype is None:
                ref_dtype = t.dtype
            out[layer] = t
        self._tensors: dict[int, torch.Tensor] = out
        self._metadata: dict = dict(metadata or {})

    # Dict-compat surface -------------------------------------------------

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
    def metadata(self) -> dict:
        """Copy of the metadata dict carried alongside the tensors."""
        return dict(self._metadata)

    def as_dict(self) -> dict[int, torch.Tensor]:
        """Return the underlying dict (shared reference, not a copy).

        Internal helper for call sites that still speak the bare-dict
        wire format (hooks, merge.linear_sum, monitor). Do not mutate.
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
        metadata: dict | None = None,
    ) -> None:
        """Save as safetensors + slim JSON sidecar.

        The sidecar carries ``format_version = 2`` so future readers can
        refuse v1.x packs. Metadata passed here overrides / augments the
        profile's own ``self.metadata``.
        """
        from saklas.core.vectors import save_profile as _save

        merged: dict = dict(self._metadata)
        if metadata:
            merged.update(metadata)
        _save(self._tensors, str(path), merged)

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "Profile":
        """Load from safetensors (+ sidecar) or gguf.

        Dispatches on file extension. Safetensors sidecars with missing or
        ``format_version < 2`` raise :class:`ProfileError` pointing at the
        migration script. GGUF files carry metadata in-header and are
        exempt from the format_version gate.
        """
        from saklas.core.vectors import load_profile as _load

        tensors, meta = _load(str(path))
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

        Delegates to :func:`saklas.merge.linear_sum`. Layer set is the
        intersection of every component; ``strict=True`` raises on drop.
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

    def cosine_similarity(
        self,
        other: "Profile",
        *,
        per_layer: bool = False,
    ) -> "float | dict[int, float]":
        """Cosine similarity against *other*.

        **Aggregate** (default): magnitude-weighted cosine over the layer
        intersection.  Weighting by ``||baked||`` matches the monitor regime.

        **Per-layer** (``per_layer=True``): raw cosine per shared layer.

        Raises :class:`ProfileError` when no layers are shared.
        """
        shared = sorted(set(self._tensors) & set(other._tensors))
        if not shared:
            raise ProfileError(
                "cosine_similarity: no shared layers between the two profiles"
            )

        if per_layer:
            out: dict[int, float] = {}
            for L in shared:
                a, b = self._tensors[L].float(), other._tensors[L].float()
                cos = (torch.dot(a, b) / (a.norm() * b.norm())).item()
                out[L] = cos
            return out

        # Magnitude-weighted aggregate.
        num = 0.0
        den = 0.0
        for L in shared:
            a, b = self._tensors[L].float(), other._tensors[L].float()
            na, nb = a.norm().item(), b.norm().item()
            cos = (torch.dot(a, b) / (na * nb)).item()
            w = na * nb
            num += w * cos
            den += w
        return num / den

    def __repr__(self) -> str:
        layers = self.layers
        if len(layers) <= 4:
            layer_desc = str(layers)
        else:
            layer_desc = f"[{layers[0]}..{layers[-1]}] ({len(layers)} layers)"
        first = next(iter(self._tensors.values()))
        return (
            f"Profile({layer_desc}, dtype={first.dtype}, "
            f"device={first.device})"
        )
