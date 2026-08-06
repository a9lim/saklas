"""StaticCache support — the ``torch.compile`` enabler, on every backend.

:class:`transformers.StaticCache` pre-allocates fixed-shape K/V buffers, so
kernel shapes stay constant across the decode loop where the default
``DynamicCache`` grows one position per step.  That is what lets inductor reuse
a single trace instead of re-specializing every token, and it is **not**
CUDA-specific: the MPS path takes it too (measured ~+16% eager, and it unlocks
the ~1.7× ``compile`` win on top).  CUDA graphs are the *superset* — with
``cuda_graphs=True`` on CUDA, ``torch.compile(mode="reduce-overhead")`` can
additionally capture graphs for the inference-shape regions.

This module owns *detection*, *construction*, and *fallback*:

- :func:`is_static_cache_supported` — the device-agnostic viability probe,
  cached by underlying module id (through ``torch.compile``'s ``_orig_mod``
  wrapper), device, and dtype.  Every StaticCache-eligible backend consults it.
- :func:`is_cuda_graphs_supported` — the CUDA-only gate on top of it, which
  decides ``reduce-overhead`` vs the ``default`` (fusion-only) compile mode.
- :func:`make_static_cache` — the single factory used by
  :mod:`saklas.core.generation`, :mod:`saklas.core.session`, and
  :mod:`saklas.core.model`.  It early-initializes the layer buffers so
  Transformers can mark stable K/V addresses outside Dynamo, and flags the
  sliding layers that cannot slide for this generation so
  :func:`_patch_static_sliding_mask`'s constant mask keeps them in the compiled
  graph.
- :func:`warn_once` — logs a CUDA-graph fallback reason once per model.

The cache pass-through itself (sizing, per-step ``cache_position``) happens at
the call sites; the policy lives here so the eager path stays uncluttered.
Callers keep an identity-stable cache, reset it between generations, and grow it
only when ``prompt_len + max_new_tokens + cache_position_offset`` exceeds its
capacity; an unsupported model falls back transparently to DynamicCache.

Steering eligibility is decided at the steering layer, not here
(``SteeringManager.all_fast_path`` / ``static_steerable``): curved, gated, and
phase-triggered hooks read mutating per-step state, so they route to the eager
DynamicCache path.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from transformers import PreTrainedModel

log = logging.getLogger(__name__)


# Logged-once gate so the fallback reason surfaces on the first generation
# of a session but doesn't spam the per-token loop.  Keyed by ``id(model)``
# rather than model_id since the user may run multiple sessions on different
# weights of the same checkpoint.
_warned_models: set[int] = set()
_support_cache: dict[
    tuple[int, str, torch.dtype | None], tuple[bool, str | None]
] = {}


def _support_cache_key(
    model: "PreTrainedModel | torch.nn.Module",
    device: torch.device | str,
) -> tuple[int, str, torch.dtype | None]:
    base = getattr(model, "_orig_mod", model)
    dtype: torch.dtype | None = None
    try:
        dtype = next(base.parameters()).dtype
    except Exception:
        dtype = None
    return id(base), str(device), dtype


def is_static_cache_supported(
    model: "PreTrainedModel | torch.nn.Module",
    device: torch.device | str,
) -> tuple[bool, str | None]:
    """Probe whether :class:`transformers.StaticCache` is viable here.

    **Device-agnostic** — unlike :func:`is_cuda_graphs_supported`, this does
    not require CUDA.  StaticCache (pre-allocated, fixed-shape K/V) is the
    enabler for ``torch.compile`` on *any* backend: fixed kernel shapes across
    the decode loop let inductor reuse one trace instead of re-specializing as a
    ``DynamicCache`` grows.  The MPS path calls this directly and pairs it with
    ``default`` compile mode; the fixed-shape benefit there is real (measured
    ~+16% eager, and it unlocks the ~1.7x ``compile`` win on top).

    Returns ``(supported, reason)``.  Checks: (1) ``StaticCache`` importable
    (transformers ≥ 4.40); (2) it constructs against the model config with a
    1-token capacity — some architectures (MLA variants, certain custom
    modeling files) raise here even when ``DynamicCache`` works.  Cached by
    ``(module id, device, dtype)``.
    """
    cache_key = _support_cache_key(model, device)
    cached = _support_cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        from transformers import StaticCache
    except ImportError:
        result = False, "transformers does not expose StaticCache (need >=4.40)"
        _support_cache[cache_key] = result
        return result

    # Probe construction with a 1-token cache.  Catch broadly because
    # architecture-specific issues raise a wide variety of errors:
    # AttributeError on missing config fields, ValueError on shape
    # mismatches, NotImplementedError on unsupported attention layouts.
    try:
        cfg = model.config
        dtype = cache_key[2] or (
            next(model.parameters()).dtype
            if hasattr(model, "parameters")
            else torch.bfloat16
        )
        probe = StaticCache(
            cfg,  # pyright: ignore[reportArgumentType]  # transformers stub types model.config as PreTrainedConfig|Tensor|Module
            max_cache_len=1,
            device=device,
            dtype=dtype,
        )
        # Touch a layer to make sure the buffers actually allocated; some
        # configs accept the constructor but trip on first slice.
        if hasattr(probe, "layers") and len(probe.layers) == 0:
            result = False, "StaticCache built zero layer buffers"
            _support_cache[cache_key] = result
            return result
        del probe
    except Exception as e:
        result = (
            False,
            f"StaticCache construction failed: {type(e).__name__}: {e}",
        )
        _support_cache[cache_key] = result
        return result

    result = True, None
    _support_cache[cache_key] = result
    return result


def is_cuda_graphs_supported(
    model: "PreTrainedModel | torch.nn.Module",
    device: torch.device | str,
) -> tuple[bool, str | None]:
    """Probe whether StaticCache + **CUDA-graph** capture is viable.

    The CUDA-specific superset of :func:`is_static_cache_supported`: CUDA-graph
    capture (via ``torch.compile(mode="reduce-overhead")``) only fires on CUDA,
    so this gates device first, then defers to the device-agnostic StaticCache
    probe.  ``__init__`` consults it to decide ``reduce-overhead`` vs the
    ``default`` (fusion-only) compile mode; the MPS/CPU fast path uses
    :func:`is_static_cache_supported` directly with ``default`` mode.
    """
    dev_str = str(device)
    dev_type = getattr(device, "type", "") if hasattr(device, "type") else dev_str
    if dev_type != "cuda" and not dev_str.startswith("cuda"):
        return False, f"device={dev_str!r} (CUDA-only)"
    return is_static_cache_supported(model, device)


_static_sliding_mask_patched = False


def _patch_static_sliding_mask() -> None:
    """Make a non-sliding StaticSlidingWindowLayer's ``get_mask_sizes`` constant.

    The hybrid-cache recompile storm: ``StaticSlidingWindowLayer.get_mask_sizes``
    branches on ``cumulative_length_int`` — a Python int that increments every
    decode step — so inside a ``torch.compile`` graph dynamo specializes on its
    value and recompiles every token until it hits ``recompile_limit`` and falls
    back to eager.  But when the cache never slides (the whole generation fits the
    static buffer — ``total_context <= max_cache_len``), ``get_mask_sizes`` is a
    *constant* ``(max_cache_len, 0)`` for every decode step: ``is_full`` stays
    False and ``kv_offset`` stays 0, so the original's ``else`` branch already
    returns exactly this.  Returning it directly — gated on the per-cache
    ``_saklas_static_mask`` flag :func:`make_static_cache` sets only when no slide
    can occur — is byte-identical to the original in that regime, but drops the
    per-step int guard so the mask stays in the compiled graph with no recompile.
    A sliding cache (long context) keeps the original dynamic path.  Idempotent."""
    global _static_sliding_mask_patched
    if _static_sliding_mask_patched:
        return
    try:
        from transformers.cache_utils import StaticSlidingWindowLayer
    except Exception:
        _static_sliding_mask_patched = True  # nothing to patch; don't retry
        return

    _orig = StaticSlidingWindowLayer.get_mask_sizes

    def get_mask_sizes(self: Any, query_length: int) -> tuple[int, int]:
        if getattr(self, "_saklas_static_mask", False):
            return self.max_cache_len, 0
        return _orig(self, query_length)

    get_mask_sizes._saklas_orig = _orig  # type: ignore[attr-defined]
    StaticSlidingWindowLayer.get_mask_sizes = get_mask_sizes  # type: ignore[method-assign]
    _static_sliding_mask_patched = True


def make_static_cache(
    model: "PreTrainedModel | torch.nn.Module",
    max_cache_len: int,
    device: torch.device | str,
    dtype: torch.dtype,
):
    """Build a StaticCache sized to ``max_cache_len`` total positions.

    Pre-allocates the per-layer K/V buffers up front so the decode loop
    sees no allocator activity.  Caller passes ``max_cache_len ≥
    prompt_len + max_new_tokens + cache_position_offset``; sizing too
    tight causes the model to OOM the cache mid-generation.

    Called on **every** StaticCache-eligible backend, not only CUDA.  Raises
    whatever the StaticCache constructor raises; callers that want graceful
    fallback probe first — :func:`is_static_cache_supported` on any device,
    :func:`is_cuda_graphs_supported` when they also need graph capture.
    """
    from transformers import StaticCache
    _patch_static_sliding_mask()
    cache = StaticCache(
        model.config,  # pyright: ignore[reportArgumentType]  # transformers stub types model.config as PreTrainedConfig|Tensor|Module
        max_cache_len=max_cache_len,
        device=device,
        dtype=dtype,
    )
    # Transformers 5.x otherwise initializes every cache layer on its first
    # ``update`` *inside* the compiled prefill.  Besides allocating the K/V
    # buffers in the graph, that moves ``cumulative_length`` from CPU to CUDA
    # there, so inductor rejects CUDA-graph capture (one CPU input plus three
    # unmarked mutations per layer) and recompiles every fresh cache until the
    # recompile limit falls back to eager.  Initialize from the text config
    # before the compiled call so transformers can mark the stable K/V tensor
    # addresses and the advertised CUDA-graph path actually captures.
    #
    # Keep a lazy fallback for unusual custom architectures whose cache
    # geometry cannot be described by the standard GQA fields.  StaticCache is
    # still correct for those models; it merely retains the older fusion-only
    # behavior instead of turning an optimization attempt into a generation
    # failure.
    early_init = getattr(cache, "early_initialization", None)
    if early_init is not None:
        config = getattr(model, "config", None)
        get_text_config = getattr(config, "get_text_config", None)
        if callable(get_text_config):
            try:
                config = get_text_config()
            except (AttributeError, RuntimeError, TypeError, ValueError):
                log.debug(
                    "Could not resolve text config for StaticCache early "
                    "initialization; retaining lazy initialization",
                    exc_info=True,
                )
        attention_heads = getattr(config, "num_attention_heads", None)
        kv_heads = getattr(config, "num_key_value_heads", None)
        if kv_heads is None:
            kv_heads = attention_heads
        head_dim = getattr(config, "head_dim", None)
        hidden_size = getattr(config, "hidden_size", None)
        if (
            head_dim is None
            and isinstance(hidden_size, int)
            and isinstance(attention_heads, int)
            and attention_heads > 0
        ):
            head_dim = hidden_size // attention_heads
        if (
            isinstance(kv_heads, int)
            and kv_heads > 0
            and isinstance(head_dim, int)
            and head_dim > 0
        ):
            try:
                early_init(
                    batch_size=1,
                    num_heads=kv_heads,
                    head_dim=head_dim,
                    dtype=dtype,
                    device=torch.device(device),
                )
            except (AttributeError, RuntimeError, TypeError, ValueError):
                log.debug(
                    "StaticCache early initialization failed; retaining lazy "
                    "initialization",
                    exc_info=True,
                )
    # Flag each sliding layer that can't slide for this generation (the whole
    # ``max_cache_len`` context fits its static buffer) so the patched
    # ``get_mask_sizes`` returns the guard-free constant and the hybrid-cache
    # recompile storm doesn't fire.  A sliding layer caps its buffer to
    # ``min(sliding_window, max_cache_len)``, so it never slides exactly when the
    # requested total is within that buffer.
    for layer in getattr(cache, "layers", []):
        if getattr(layer, "is_sliding", False):
            buf = getattr(layer, "max_cache_len", None)
            if buf is not None:
                layer._saklas_static_mask = max_cache_len <= buf
    return cache


def warn_once(model: "PreTrainedModel | torch.nn.Module", reason: str) -> None:
    """Log the fallback reason for a model exactly once per session lifetime.

    Used by the session at first-generation time to surface why CUDA
    graphs are off (architecture quirk, transformers too old, etc.)
    without spamming the per-step loop.
    """
    key = id(model)
    if key in _warned_models:
        return
    _warned_models.add(key)
    log.info("CUDA graphs disabled: %s", reason)
