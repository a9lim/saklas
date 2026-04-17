"""SAE backend protocol and adapters.

The protocol is tiny on purpose: contrastive extraction needs only per-layer
encode/decode and the set of covered layers. The concrete ``SaeLensBackend``
adapter (added later) lives in the same module but imports ``sae_lens`` lazily,
inside its factory function — so installations without the ``[sae]`` extra
can still import ``saklas.core.sae`` (e.g. for the protocol type hint or the
mock).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol, runtime_checkable

import torch


@runtime_checkable
class SaeBackend(Protocol):
    """Minimal surface for SAE-backed contrastive extraction."""

    release: str
    revision: str | None
    layers: frozenset[int]      # saklas 0-indexed transformer-block layers

    def encode_layer(self, idx: int, h: torch.Tensor) -> torch.Tensor:
        """Encode a batch of hidden states into sparse-feature space.

        Input shape: ``(N, d_model)``. Output shape: ``(N, d_feature)``.
        Caller guarantees ``idx in self.layers``.
        """
        ...

    def decode_layer(self, idx: int, f: torch.Tensor) -> torch.Tensor:
        """Decode a single feature-space direction back into model space.

        Input shape: ``(d_feature,)``. Output shape: ``(d_model,)``.
        Caller guarantees ``idx in self.layers``.
        """
        ...


# --- test helper ----------------------------------------------------------

@dataclass
class MockSaeBackend:
    """In-memory SAE backend for CPU-only tests.

    Default is identity encode/decode with ``d_feature == d_model``. Pass
    ``encode_fn`` / ``decode_fn`` for non-trivial layer-level transforms.
    """
    layers: frozenset[int]
    d_model: int
    d_feature: int | None = None
    release: str = "mock-release"
    revision: str | None = None
    encode_fn: Callable[[int, torch.Tensor], torch.Tensor] | None = None
    decode_fn: Callable[[int, torch.Tensor], torch.Tensor] | None = None

    def __post_init__(self):
        if self.d_feature is None:
            self.d_feature = self.d_model

    def encode_layer(self, idx: int, h: torch.Tensor) -> torch.Tensor:
        if self.encode_fn is not None:
            return self.encode_fn(idx, h)
        return h

    def decode_layer(self, idx: int, f: torch.Tensor) -> torch.Tensor:
        if self.decode_fn is not None:
            return self.decode_fn(idx, f)
        return f


# --- SAELens-backed concrete adapter --------------------------------------

@dataclass
class SaeLensBackend:
    """SAELens-backed concrete ``SaeBackend``.

    Per-layer SAE modules are held in a dict keyed by layer index. The
    registry-level resolution of `release → per-layer sae_ids` is performed
    once at load time (see :func:`load_sae_backend`); this class just
    dispatches encode/decode to the right per-layer module.
    """
    release: str
    revision: str | None
    layers: frozenset[int]
    _saes_by_layer: dict[int, "object"] = field(repr=False)
    sae_ids_by_layer: dict[str, str] = field(default_factory=dict)

    def encode_layer(self, idx: int, h: torch.Tensor) -> torch.Tensor:
        return self._saes_by_layer[idx].encode(h)

    def decode_layer(self, idx: int, f: torch.Tensor) -> torch.Tensor:
        return self._saes_by_layer[idx].decode(f)


def load_sae_backend(
    release: str,
    *,
    revision: str | None = None,
    model_id: str,
    device: str | torch.device,
    dtype: torch.dtype | None = None,
) -> SaeLensBackend:
    """Resolve a SAELens release to a fully-populated :class:`SaeLensBackend`.

    Raises:
        SaeBackendImportError: ``sae_lens`` not installed.
        SaeReleaseNotFoundError: release not in the SAELens registry.
        SaeModelMismatchError: release's base model != requested ``model_id``.
    """
    from saklas.core.errors import (
        SaeBackendImportError,
        SaeReleaseNotFoundError,
        SaeModelMismatchError,
    )

    try:
        import sae_lens  # type: ignore[import-not-found]
    except ImportError:
        raise SaeBackendImportError(
            f"requested SAE release '{release}' but `sae_lens` is not "
            f"installed. Install with `pip install -e \".[sae]\"`."
        )
    if sae_lens is None:
        # Tests (and some environments) may None-shadow the module.
        raise SaeBackendImportError(
            f"requested SAE release '{release}' but `sae_lens` is not "
            f"installed. Install with `pip install -e \".[sae]\"`."
        )

    registry = sae_lens.get_pretrained_saes_directory()
    if release not in registry:
        import difflib
        all_releases = list(registry.keys())
        nearby = difflib.get_close_matches(release, all_releases, n=10, cutoff=0.3)
        # Fall back to listing a sample of available releases when no fuzzy
        # match trips the cutoff — users still need a discoverability hint.
        if not nearby:
            nearby = all_releases[:10]
        raise SaeReleaseNotFoundError(
            f"SAE release '{release}' not found in SAELens registry. "
            f"Near matches: {nearby or '(none)'}"
        )

    entry = registry[release]
    registry_model = entry.get("model") or entry.get("model_name")
    if registry_model and model_id and not _model_names_match(registry_model, model_id):
        raise SaeModelMismatchError(
            f"SAE release '{release}' was trained on '{registry_model}' but "
            f"the saklas session is loaded with '{model_id}'"
        )

    saes_map = entry.get("saes_map", {})
    if not saes_map:
        raise SaeReleaseNotFoundError(
            f"SAE release '{release}' has no saes_map in the registry"
        )

    layers: set[int] = set()
    saes_by_layer: dict[int, object] = {}
    ids_by_layer: dict[str, str] = {}
    for sae_id, hook_layer in _canonical_layer_map(saes_map).items():
        sae, cfg_dict, _sparsity = sae_lens.SAE.from_pretrained(
            release=release,
            sae_id=sae_id,
            device=str(device),
        )
        # Prefer the cfg's hook_layer if the registry and cfg disagree.
        layer_idx = int(cfg_dict.get("hook_layer", hook_layer))
        layers.add(layer_idx)
        saes_by_layer[layer_idx] = sae
        ids_by_layer[str(layer_idx)] = sae_id

    if dtype is not None:
        for sae in saes_by_layer.values():
            if hasattr(sae, "to"):
                sae.to(dtype=dtype)

    return SaeLensBackend(
        release=release,
        revision=revision,
        layers=frozenset(layers),
        _saes_by_layer=saes_by_layer,
        sae_ids_by_layer=ids_by_layer,
    )


def _canonical_layer_map(saes_map: dict) -> dict[str, int]:
    """Pick one sae_id per layer. Prefer narrowest width; warn on ambiguity.

    SAELens releases that ship multiple SAEs per layer (different widths, L0s)
    expose all of them via ``saes_map``. Canonical sub-releases (e.g.
    ``gemma-scope-2b-pt-res-canonical``) already commit to one per layer;
    other releases surface multiple. We bucket by ``hook_layer``, pick the
    lexicographically smallest sae_id per bucket, and warn when we had to
    choose — so users can override by picking a different release.
    """
    import re
    import warnings

    buckets: dict[int, list[tuple[str, int]]] = {}
    for sae_id, hook_layer in saes_map.items():
        layer_int: int | None = None
        try:
            layer_int = int(hook_layer)
        except (ValueError, TypeError):
            m = re.search(r"layer[_-]?(\d+)", sae_id)
            if m:
                layer_int = int(m.group(1))
        if layer_int is None:
            continue
        buckets.setdefault(layer_int, []).append((sae_id, layer_int))

    out: dict[str, int] = {}
    for layer_int, candidates in buckets.items():
        candidates.sort(key=lambda t: t[0])
        chosen_id, chosen_layer = candidates[0]
        out[chosen_id] = chosen_layer
        if len(candidates) > 1:
            other_ids = [c[0] for c in candidates[1:]]
            warnings.warn(
                f"SAE layer {layer_int}: multiple SAEs in registry; chose "
                f"'{chosen_id}'. Others available: {other_ids}",
                stacklevel=3,
            )
    return out


def _model_names_match(a: str, b: str) -> bool:
    """Lenient equality between HF model ids and SAELens short names.

    SAELens ``cfg.model_name`` may be a short name (e.g. ``gpt2-small``);
    saklas callers pass full HF ids (``openai-community/gpt2``). We match
    case-insensitively and treat one substring-containing-the-other as a
    match so typical short/long pairings line up.
    """
    a_short = a.split("/")[-1].lower()
    b_short = b.split("/")[-1].lower()
    return a_short == b_short or a_short in b_short or b_short in a_short
