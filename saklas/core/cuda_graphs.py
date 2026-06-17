"""CUDA-graphs / StaticCache support.

When enabled (``cuda_graphs=True`` + device==cuda + supported architecture
+ fast-path-eligible steering), generation routes through
:class:`transformers.StaticCache` instead of the default ``DynamicCache``.
Static caches don't grow per step, so kernel shapes stay fixed across the
decode loop, which lets ``torch.compile(mode="reduce-overhead")`` capture
CUDA graphs internally for the inference-shape regions.

This module owns *detection* and *fallback*: probing whether StaticCache
construction succeeds on a given model + device, caching that answer, and
exposing a single factory used by :mod:`saklas.core.generation` and
:mod:`saklas.core.session`.  The actual cache pass-through (allocating,
sizing, per-step ``cache_position``) happens at the call sites; we keep
the policy in one place so the eager path stays uncluttered.

Caller responsibilities:
- Call :func:`is_cuda_graphs_supported` during construction; it caches by
  underlying module id (through ``torch.compile``'s ``_orig_mod`` wrapper),
  device, and dtype, then the session stores the boolean result.
- On supported sessions, build a fresh StaticCache via
  :func:`make_static_cache` per generation, sized to
  ``prompt_len + max_new_tokens + cache_position_offset``.
- On unsupported sessions, fall back transparently to DynamicCache —
  the eager loop is unchanged.

Slow-path steering (probe gates, multi-trigger, ablation under CTX
mutation) bypasses StaticCache: the hooks read mutating state per step,
which CUDA-graph capture can't track without recapture overhead.  The
eligibility check lives at the steering layer, not here.
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
    ``DynamicCache`` grows.  On MPS the fixed-shape benefit is real (measured
    ~+16% eager, and it unlocks the ~1.7x ``compile`` win on top), not the
    "negligible" the old CUDA-only gate assumed.

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

    Raises whatever the StaticCache constructor raises.  Callers that
    want graceful fallback should call :func:`is_cuda_graphs_supported`
    first and check the boolean.
    """
    from transformers import StaticCache
    _patch_static_sliding_mask()
    cache = StaticCache(
        model.config,  # pyright: ignore[reportArgumentType]  # transformers stub types model.config as PreTrainedConfig|Tensor|Module
        max_cache_len=max_cache_len,
        device=device,
        dtype=dtype,
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
