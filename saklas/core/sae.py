"""SAE backend protocol and adapters.

The protocol is tiny on purpose: contrastive extraction needs only per-layer
encode/decode and the set of covered layers. The concrete ``SaeLensBackend``
adapter (added later) lives in the same module but imports ``sae_lens`` lazily,
inside its factory function — so installations without the ``[sae]`` extra
can still import ``saklas.core.sae`` (e.g. for the protocol type hint or the
mock).
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

import torch


def _commit_pinned_converter(converter: Any, revision: str) -> Any:
    """Clone a SAELens converter with every Hub read pinned to one commit.

    SAELens 6 exposes ``converter=`` but not ``revision=``. Its built-in
    converters share one module-global ``hf_hub_download``/``hf_hub_url``.
    Cloning those same-module helper functions into an isolated globals dict
    lets saklas inject an immutable revision without process-global monkeypatches.
    """
    import inspect
    from types import FunctionType

    if not inspect.isfunction(converter):
        raise TypeError("SAELens converter is not a clonable Python function")
    source_globals = converter.__globals__
    original_download = source_globals.get("hf_hub_download")
    original_url = source_globals.get("hf_hub_url")
    if not callable(original_download):
        raise TypeError("SAELens converter has no interceptable Hub downloader")

    def _download(*args: Any, **kwargs: Any) -> Any:
        kwargs["revision"] = revision
        return original_download(*args, **kwargs)

    def _url(*args: Any, **kwargs: Any) -> Any:
        if not callable(original_url):
            raise TypeError("SAELens converter has no interceptable Hub URL builder")
        kwargs["revision"] = revision
        return original_url(*args, **kwargs)

    cloned_globals = dict(source_globals)
    cloned_globals["hf_hub_download"] = _download
    if original_url is not None:
        cloned_globals["hf_hub_url"] = _url
    for name, value in source_globals.items():
        if inspect.isfunction(value) and value.__globals__ is source_globals:
            clone = FunctionType(
                value.__code__, cloned_globals, value.__name__,
                value.__defaults__, value.__closure__,
            )
            clone.__kwdefaults__ = value.__kwdefaults__
            cloned_globals[name] = clone
    return cloned_globals.get(converter.__name__, converter)


@runtime_checkable
class SaeBackend(Protocol):
    """Minimal surface for SAE-backed contrastive extraction."""

    release: str
    revision: str | None
    fingerprint: str | None
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

    def feature_count(self, idx: int) -> int:
        """Number of sparse features at ``idx``."""
        ...

    def feature_direction(self, idx: int, feature_id: int) -> torch.Tensor:
        """Decoder row for one sparse feature, in model residual space."""
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
    fingerprint: str | None = None
    encode_fn: Callable[[int, torch.Tensor], torch.Tensor] | None = None
    decode_fn: Callable[[int, torch.Tensor], torch.Tensor] | None = None
    sae_ids_by_layer: dict[str, str] = field(default_factory=dict)
    repo_id: str | None = None
    neuronpedia_id: str | None = None
    neuronpedia_ids_by_layer: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.d_feature is None:
            self.d_feature = self.d_model
        if not self.sae_ids_by_layer:
            self.sae_ids_by_layer = {
                str(layer): f"mock-layer-{layer}" for layer in self.layers
            }
        if (
            self.fingerprint is None
            and self.encode_fn is None
            and self.decode_fn is None
        ):
            import hashlib
            import json

            self.fingerprint = hashlib.sha256(json.dumps({
                "backend": "mock-identity",
                "release": self.release,
                "revision": self.revision,
                "layers": sorted(self.layers),
                "d_model": self.d_model,
                "d_feature": self.d_feature,
            }, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()

    def encode_layer(self, idx: int, h: torch.Tensor) -> torch.Tensor:
        if self.encode_fn is not None:
            return self.encode_fn(idx, h)
        return h

    def decode_layer(self, idx: int, f: torch.Tensor) -> torch.Tensor:
        if self.decode_fn is not None:
            return self.decode_fn(idx, f)
        return f

    def feature_count(self, idx: int) -> int:
        if idx not in self.layers:
            raise KeyError(idx)
        assert self.d_feature is not None
        return int(self.d_feature)

    def feature_direction(self, idx: int, feature_id: int) -> torch.Tensor:
        width = self.feature_count(idx)
        if not 0 <= feature_id < width:
            raise IndexError(feature_id)
        one_hot = torch.zeros(width)
        one_hot[feature_id] = 1
        return self.decode_layer(idx, one_hot)


# --- SAELens-backed concrete adapter --------------------------------------

@dataclass
class SaeLensBackend:
    """SAELens-backed concrete ``SaeBackend``.

    Registry resolution is eager and weight loading is lazy.  Manifold fitting
    visits one layer at a time, so retaining every SAE in a release can consume
    many times the base model's memory for no benefit.  This adapter keeps only
    the most recently used layer resident; an encode/decode pair for the same
    layer shares that module, then the next layer replaces it.
    """
    release: str
    revision: str | None
    fingerprint: str | None
    layers: frozenset[int]
    _loader: Callable[[int], Any] = field(repr=False)
    sae_ids_by_layer: dict[str, str] = field(default_factory=dict)
    repo_id: str | None = None
    neuronpedia_id: str | None = None
    neuronpedia_ids_by_layer: dict[str, str] = field(default_factory=dict)
    _active_layer: int | None = field(default=None, init=False, repr=False)
    _active_sae: Any | None = field(default=None, init=False, repr=False)

    def _sae(self, idx: int) -> Any:
        if idx not in self.layers:
            raise KeyError(f"SAE release {self.release!r} does not cover layer {idx}")
        if self._active_layer != idx or self._active_sae is None:
            self._active_sae = self._loader(idx)
            self._active_layer = idx
        return self._active_sae

    def encode_layer(self, idx: int, h: torch.Tensor) -> torch.Tensor:
        return self._sae(idx).encode(h)

    def decode_layer(self, idx: int, f: torch.Tensor) -> torch.Tensor:
        return self._sae(idx).decode(f)

    def feature_count(self, idx: int) -> int:
        sae = self._sae(idx)
        cfg: Any = getattr(sae, "cfg", None)
        width = (
            cfg.get("d_sae")
            if hasattr(cfg, "get")
            else getattr(cfg, "d_sae", None)
        )
        if width is None:
            w_dec: Any = getattr(sae, "W_dec", None)
            if not isinstance(w_dec, torch.Tensor) or w_dec.ndim != 2:
                raise ValueError(
                    f"SAE release {self.release!r} does not expose d_sae/W_dec"
                )
            width = w_dec.shape[0]
        return int(width)

    def feature_direction(self, idx: int, feature_id: int) -> torch.Tensor:
        sae = self._sae(idx)
        width = self.feature_count(idx)
        if not 0 <= feature_id < width:
            raise IndexError(feature_id)
        w_dec: Any = getattr(sae, "W_dec", None)
        if isinstance(w_dec, torch.Tensor) and w_dec.ndim == 2:
            return w_dec[feature_id]
        # Compatibility fallback for an adapter that exposes only decode().
        first = next(iter(sae.parameters()), None) if hasattr(sae, "parameters") else None
        one_hot = torch.zeros(
            width,
            device=getattr(first, "device", None),
            dtype=getattr(first, "dtype", None),
        )
        one_hot[feature_id] = 1
        return sae.decode(one_hot)


def select_runtime_layer(
    available: frozenset[int] | set[int],
    n_layers: int,
    requested: int | None = None,
) -> int:
    """Choose the one resident SAE hook layer for the v1 live runtime.

    Explicit selection wins.  Otherwise choose the available layer nearest
    65% depth, preferring the model's 40–90% workspace band.  This is stable
    across launches and keeps the unqualified ``sae/<id>`` grammar singular.
    """
    choices = sorted(int(layer) for layer in available)
    if not choices:
        from saklas.core.errors import SaeCoverageError
        raise SaeCoverageError("SAE release covers no model layers")
    if requested is not None:
        if requested not in choices:
            from saklas.core.errors import SaeCoverageError
            raise SaeCoverageError(
                f"SAE release does not cover layer {requested}; available: {choices}"
            )
        return requested
    denom = max(n_layers - 1, 1)
    band = [layer for layer in choices if 0.4 <= layer / denom <= 0.9]
    pool = band or choices
    target = 0.65 * denom
    return min(pool, key=lambda layer: (abs(layer - target), layer))


def list_sae_releases(model_id: str) -> list[dict[str, Any]]:
    """Return SAELens registry suggestions compatible with ``model_id``.

    Metadata-only: no SAE weights are loaded and no model forward runs. Saklas
    captures transformer-block outputs, so registry families that explicitly
    target attention/MLP internals or transcoders are not live-runtime
    compatible and are omitted. Unknown naming schemes remain discoverable and
    are validated from their loaded SAE metadata before becoming resident.
    """
    from saklas.core.errors import SaeBackendImportError

    try:
        import sae_lens
    except ImportError as exc:
        raise SaeBackendImportError(
            "SAE release discovery requires `sae_lens`; install saklas[sae]"
        ) from exc
    loader = getattr(sae_lens, "get_pretrained_saes_directory", None)
    if loader is None:
        from sae_lens.loading.pretrained_saes_directory import (  # pyright: ignore[reportMissingImports]
            get_pretrained_saes_directory,
        )
        loader = get_pretrained_saes_directory
    rows: list[dict[str, Any]] = []
    for release, entry in loader().items():
        def value(name: str, default: Any = None) -> Any:
            return entry.get(name, default) if isinstance(entry, Mapping) else getattr(entry, name, default)

        release_model = value("model") or value("model_name")
        if release_model and model_id and not _model_names_match(str(release_model), model_id):
            continue
        hook_kind = _release_hook_kind(str(release))
        if hook_kind in {"attention", "mlp", "transcoder"}:
            continue
        layer_map = _canonical_layer_map(value("saes_map", {}), warn=False)
        rows.append({
            "release": str(release),
            "model": str(release_model) if release_model else None,
            "layers": sorted(set(int(layer) for layer in layer_map.values())),
            "repo_id": value("repo_id"),
            "neuronpedia": bool(value("neuronpedia_id")),
        })
    rows.sort(key=lambda row: row["release"])
    return rows


def _release_hook_kind(release: str) -> str | None:
    """Infer explicit SAELens hook families from conventional release names.

    This is only a metadata-time filter for the release picker. The loaded SAE
    remains authoritative and is checked by :func:`_validate_residual_hook`.
    """
    import re

    tokens = set(filter(None, re.split(r"[^a-z0-9]+", release.lower())))
    if "transcoder" in tokens or "transcoders" in tokens:
        return "transcoder"
    if "att" in tokens or "attn" in tokens or "attention" in tokens:
        return "attention"
    if "mlp" in tokens:
        return "mlp"
    if "res" in tokens or "resid" in tokens or "residual" in tokens:
        return "residual"
    return None


def _metadata_value(metadata: Any, name: str) -> Any:
    if metadata is None:
        return None
    if hasattr(metadata, "get"):
        return metadata.get(name)
    return getattr(metadata, name, None)


def _validate_residual_hook(sae: Any, release: str, layer_idx: int) -> None:
    """Reject a loaded SAE trained anywhere other than block residual-post.

    Saklas's capture layer is the transformer block output. Feeding that tensor
    to an attention/MLP SAE can either fail on width (the common case) or,
    worse, silently produce meaningless activations when widths happen to agree.
    """
    from saklas.core.errors import SaeCoverageError

    cfg: Any = getattr(sae, "cfg", None)
    metadata = (
        cfg.get("metadata")
        if hasattr(cfg, "get")
        else getattr(cfg, "metadata", None)
    )
    hook_name = (
        _metadata_value(metadata, "hook_name")
        or _metadata_value(metadata, "hf_hook_name")
        or (
            cfg.get("hook_name")
            if hasattr(cfg, "get")
            else getattr(cfg, "hook_name", None)
        )
    )
    if not isinstance(hook_name, str) or not hook_name:
        return
    normalized = hook_name.lower()
    is_residual_post = (
        normalized.endswith(".hook_resid_post")
        or normalized.endswith(f"layers.{layer_idx}.output")
    )
    if not is_residual_post:
        raise SaeCoverageError(
            f"SAE release {release!r} targets {hook_name!r}, but saklas SAE "
            "capture requires a residual-post/block-output release (for "
            "Gemma Scope, choose the corresponding '-res' release)"
        )


def validate_residual_width(
    backend: SaeBackend, layer: int, model_width: int,
) -> None:
    """Ensure one SAE decoder row lives in the model residual stream."""
    from saklas.core.errors import SaeCoverageError

    direction = backend.feature_direction(layer, 0)
    sae_width = int(direction.numel())
    if sae_width != int(model_width):
        raise SaeCoverageError(
            f"SAE release {backend.release!r} at layer {layer} has activation "
            f"width {sae_width}, but the model residual stream has width "
            f"{model_width}; choose a residual-post/block-output SAE release"
        )


def sae_device_str(device: str | torch.device) -> str:
    """Device string for the SAELens loader.

    safetensors (and SAELens's own device plumbing) reject the indexed MPS
    form a live model reports (``str(model.device) == "mps:0"``); MPS has
    exactly one device, so the bare form is equivalent.
    """
    text = str(device)
    return "mps" if text.startswith("mps") else text


def load_sae_backend(
    release: str,
    *,
    revision: str | None = None,
    model_id: str,
    device: str | torch.device,
    dtype: torch.dtype | None = None,
) -> SaeLensBackend:
    """Resolve a SAELens release to a lazy :class:`SaeLensBackend`.

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
        import sae_lens
    except ImportError as e:
        raise SaeBackendImportError(
            f"requested SAE release '{release}' but `sae_lens` is not "
            f"installed. Install with `pip install -e \".[sae]\"`."
        ) from e
    if sae_lens is None:
        # Tests (and some environments) may None-shadow the module.
        raise SaeBackendImportError(
            f"requested SAE release '{release}' but `sae_lens` is not "
            f"installed. Install with `pip install -e \".[sae]\"`."
        )

    registry_loader = getattr(sae_lens, "get_pretrained_saes_directory", None)
    if registry_loader is None:
        # SAELens 6.x moved the registry helper out of the package root.
        from sae_lens.loading.pretrained_saes_directory import (  # pyright: ignore[reportMissingImports]
            get_pretrained_saes_directory,
        )

        registry_loader = get_pretrained_saes_directory
    registry = registry_loader()
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
    def _entry_value(name: str, default: Any = None) -> Any:
        if isinstance(entry, Mapping):
            return entry.get(name, default)
        return getattr(entry, name, default)

    registry_model = _entry_value("model") or _entry_value("model_name")
    if registry_model and model_id and not _model_names_match(registry_model, model_id):
        raise SaeModelMismatchError(
            f"SAE release '{release}' was trained on '{registry_model}' but "
            f"the saklas session is loaded with '{model_id}'"
        )

    saes_map = _entry_value("saes_map", {})
    if not saes_map:
        raise SaeReleaseNotFoundError(
            f"SAE release '{release}' has no saes_map in the registry"
        )

    ids_by_layer_int = {
        int(hook_layer): sae_id
        for sae_id, hook_layer in _canonical_layer_map(saes_map).items()
    }
    ids_by_layer = {
        str(layer_idx): sae_id
        for layer_idx, sae_id in ids_by_layer_int.items()
    }
    repo_id = _entry_value("repo_id")
    neuronpedia_raw = _entry_value("neuronpedia_id")
    neuronpedia_by_layer: dict[str, str] = {}
    if isinstance(neuronpedia_raw, Mapping):
        for layer_idx, sae_id in ids_by_layer_int.items():
            value = neuronpedia_raw.get(sae_id)
            if isinstance(value, str) and value:
                neuronpedia_by_layer[str(layer_idx)] = value
    elif isinstance(neuronpedia_raw, str) and neuronpedia_raw:
        # Single-SAE releases occasionally expose one scalar id.
        for layer_idx in ids_by_layer_int:
            neuronpedia_by_layer[str(layer_idx)] = neuronpedia_raw
    resolved_commit: str | None = None
    if isinstance(repo_id, str) and repo_id:
        try:
            from huggingface_hub import HfApi

            resolved_commit = HfApi().model_info(
                repo_id, revision=revision, timeout=3,
            ).sha
        except Exception:
            # Without an immutable Hub identity, SAE feature transforms are
            # not provably reusable. The backend remains usable, but fitted
            # tensor cache hits are disabled by ``fingerprint=None``.
            resolved_commit = None
        if (
            resolved_commit is None
            and isinstance(revision, str)
            and len(revision) == 40
            and all(char in "0123456789abcdefABCDEF" for char in revision)
        ):
            # A caller-supplied full commit is already immutable even if the
            # metadata lookup is offline.
            resolved_commit = revision.lower()
    elif not repo_id:
        # A registry map names transforms but does not prove their bytes. Local
        # and private converters may resolve the same ids to mutable files, so
        # only an integration-supplied immutable content fingerprint permits
        # fitted-artifact reuse. Without one the backend remains usable and the
        # extraction cache deliberately misses.
        explicit_fingerprint = (
            _entry_value("content_fingerprint")
            or _entry_value("fingerprint")
        )
        if isinstance(explicit_fingerprint, str) and explicit_fingerprint:
            resolved_commit = explicit_fingerprint
    fingerprint = None
    if resolved_commit is not None:
        import hashlib
        import json

        fingerprint = hashlib.sha256(json.dumps({
            "release": release,
            "repo_id": repo_id,
            "resolved_commit": resolved_commit,
            "sae_ids_by_layer": ids_by_layer,
            "sae_lens_version": getattr(sae_lens, "__version__", None),
            "conversion_func": _entry_value("conversion_func"),
            "config_overrides": _entry_value("config_overrides"),
            "dtype": str(dtype),
        }, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    import inspect

    loader = getattr(  # SAELens 6.x's tuple-returning compatibility loader.
        sae_lens.SAE, "from_pretrained_with_cfg_and_sparsity", None,
    ) or sae_lens.SAE.from_pretrained  # pyright: ignore[reportAttributeAccessIssue]  # optional dep
    loader_parameters = inspect.signature(loader).parameters
    supports_revision = "revision" in loader_parameters
    pinned_converter: Any | None = None
    if isinstance(repo_id, str) and repo_id and resolved_commit is not None:
        if not supports_revision and "converter" in loader_parameters:
            try:
                from sae_lens.loading.pretrained_sae_loaders import (  # pyright: ignore[reportMissingImports]
                    NAMED_PRETRAINED_SAE_LOADERS,
                    get_conversion_loader_name,
                )

                converter = NAMED_PRETRAINED_SAE_LOADERS[
                    get_conversion_loader_name(release)
                ]
                pinned_converter = _commit_pinned_converter(
                    converter, resolved_commit,
                )
            except Exception as exc:
                raise ValueError(
                    f"SAELens {getattr(sae_lens, '__version__', 'installed')} "
                    f"cannot bind release {release!r} to immutable commit "
                    f"{resolved_commit}: {exc}"
                ) from exc
        elif not supports_revision:
            raise ValueError(
                f"SAELens {getattr(sae_lens, '__version__', 'installed')} "
                "exposes neither revision= nor a pinnable converter=; refusing "
                "an identity-unsafe SAE fit"
            )

    def _load_layer(layer_idx: int) -> Any:
        from saklas.core.errors import SaeCoverageError

        sae_id = ids_by_layer_int[layer_idx]
        if isinstance(repo_id, str) and repo_id and resolved_commit is None:
            raise ValueError(
                f"cannot resolve an immutable Hub commit for SAE release "
                f"{release!r}; refusing to mix or persist mutable weights"
            )
        load_kwargs: dict[str, Any] = {
            "release": release,
            "sae_id": sae_id,
            "device": sae_device_str(device),
        }
        if dtype is not None and "dtype" in inspect.signature(loader).parameters:
            load_kwargs["dtype"] = str(dtype).removeprefix("torch.")
        if supports_revision:
            load_revision = resolved_commit if repo_id else revision
            if load_revision is not None:
                load_kwargs["revision"] = load_revision
        elif pinned_converter is not None:
            load_kwargs["converter"] = pinned_converter
        loaded = loader(**load_kwargs)
        if isinstance(loaded, tuple):
            sae = loaded[0]
            cfg_dict = loaded[1] if len(loaded) > 1 else getattr(sae, "cfg", {})
        else:
            sae = loaded
            cfg_dict = getattr(sae, "cfg", {})
        cfg_dict_any: Any = cfg_dict  # sae_lens SAEConfig has no stubs; treat as Any
        loaded_layer = int(
            cfg_dict_any.get("hook_layer", layer_idx)
            if hasattr(cfg_dict_any, "get")
            else getattr(cfg_dict_any, "hook_layer", layer_idx)
        )
        if loaded_layer != layer_idx:
            raise SaeCoverageError(
                f"SAE registry maps {sae_id!r} to layer {layer_idx}, but its "
                f"loaded config reports layer {loaded_layer}"
            )
        _validate_residual_hook(sae, release, layer_idx)
        if dtype is not None:
            if hasattr(sae, "to"):
                sae_any: Any = sae
                sae_any.to(dtype=dtype)
        return sae

    return SaeLensBackend(
        release=release,
        revision=revision,
        fingerprint=fingerprint,
        layers=frozenset(ids_by_layer_int),
        _loader=_load_layer,
        sae_ids_by_layer=ids_by_layer,
        repo_id=repo_id if isinstance(repo_id, str) else None,
        neuronpedia_id=(
            next(iter(neuronpedia_by_layer.values()), None)
        ),
        neuronpedia_ids_by_layer=neuronpedia_by_layer,
    )


def _canonical_layer_map(
    saes_map: dict[str, Any],
    *,
    warn: bool = True,
) -> dict[str, int]:
    """Pick one SAE per layer: narrowest width, then sparsest L0.

    SAELens releases that ship multiple SAEs per layer (different widths, L0s)
    expose all of them via ``saes_map``. Canonical sub-releases (e.g.
    ``gemma-scope-2b-pt-res-canonical``) already commit to one per layer;
    other releases surface multiple. We bucket by ``hook_layer``, pick the
    numerically narrowest width per bucket, then the numerically smallest
    average-L0 (or ``small`` / ``medium`` / ``big`` in that order), and warn
    when we had to choose — so users can override by picking a different
    release.  The final id tie-break makes registry ordering irrelevant.
    """
    import re
    import warnings

    def _candidate_rank(sae_id: str) -> tuple[int, int, str]:
        width_match = re.search(
            r"(?:^|[/_.-])width[_-]?(\d+)([km])?(?:$|[/_.-])",
            sae_id,
            flags=re.IGNORECASE,
        )
        if width_match is None:
            width = 2**63 - 1
        else:
            multiplier = {"": 1, "k": 1_000, "m": 1_000_000}[
                width_match.group(2).lower()
            ]
            width = int(width_match.group(1)) * multiplier
        l0_match = re.search(
            r"(?:average[_-])?l0[_-]?(\d+|small|medium|big)(?:$|[/_.-])",
            sae_id,
            flags=re.IGNORECASE,
        )
        if l0_match is None:
            l0 = 2**63 - 1
        else:
            raw_l0 = l0_match.group(1).lower()
            l0 = (
                {"small": 0, "medium": 1, "big": 2}[raw_l0]
                if not raw_l0.isdigit()
                else int(raw_l0)
            )
        return width, l0, sae_id

    buckets: dict[int, list[tuple[str, int]]] = {}
    for sae_id, hook_layer in saes_map.items():
        layer_int: int | None = None
        try:
            layer_int = int(hook_layer)
        except (ValueError, TypeError):
            candidates = (sae_id, str(hook_layer))
            patterns = (
                r"(?:^|[/_.-])layer[_-]?(\d+)(?:$|[/_.-])",
                r"(?:^|[/_.-])blocks?[._/-](\d+)(?:$|[/_.-])",
                r"(?:^|[/_.-])l(\d+)[am]?(?:$|[/_.-])",
            )
            for candidate in candidates:
                match = next((
                    found
                    for pattern in patterns
                    if (found := re.search(pattern, candidate, flags=re.IGNORECASE))
                ), None)
                if match is not None:
                    layer_int = int(match.group(1))
                    break
        if layer_int is None:
            continue
        buckets.setdefault(layer_int, []).append((sae_id, layer_int))

    out: dict[str, int] = {}
    for layer_int, candidates in buckets.items():
        candidates.sort(key=lambda candidate: _candidate_rank(candidate[0]))
        chosen_id, chosen_layer = candidates[0]
        out[chosen_id] = chosen_layer
        if warn and len(candidates) > 1:
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
