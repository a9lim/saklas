"""CUDA-graphs / StaticCache support (Phase B of v2.2's perf push).

When enabled (``cuda_graphs=True`` + device==cuda + supported architecture
+ fast-path-eligible steering), generation routes through
:class:`transformers.StaticCache` instead of the default ``DynamicCache``.
Static caches don't grow per step, so kernel shapes stay fixed across the
decode loop, which lets ``torch.compile(mode="reduce-overhead")`` capture
CUDA graphs internally for the inference-shape regions.

This module owns *detection* and *fallback*: probing whether StaticCache
construction succeeds on a given model + device, and exposing a single
factory used by :mod:`saklas.core.generation` and
:mod:`saklas.core.session`.  The actual cache pass-through (allocating,
sizing, per-step ``cache_position``) happens at the call sites; we keep
the policy in one place so the eager path stays uncluttered.

Caller responsibilities:
- Call :func:`is_cuda_graphs_supported` once per session at construction
  time; cache the result on the session.
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
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from transformers import PreTrainedModel

log = logging.getLogger(__name__)


# Logged-once gate so the fallback reason surfaces on the first generation
# of a session but doesn't spam the per-token loop.  Keyed by ``id(model)``
# rather than model_id since the user may run multiple sessions on different
# weights of the same checkpoint.
_warned_models: set[int] = set()


def is_cuda_graphs_supported(
    model: "PreTrainedModel | torch.nn.Module",
    device: torch.device | str,
) -> tuple[bool, str | None]:
    """Probe whether StaticCache + CUDA-graph capture is viable.

    Returns ``(supported, reason)``.  When ``supported=False``, ``reason``
    is a short human-readable string explaining the gate that failed —
    logged once per model so the user sees why the fast path is off
    without spamming subsequent generations.

    Checks (in order):

    1. Device is CUDA.  StaticCache works on CPU/MPS too but graph capture
       only fires on CUDA, and on MPS the static-shape benefit is
       negligible against the device-context overhead.
    2. ``transformers.StaticCache`` is importable (transformers ≥ 4.40).
    3. StaticCache constructs successfully against the model's config
       with a 1-token capacity.  Some architectures (notably MLA variants
       and certain custom modeling files) raise here even when
       DynamicCache works.
    """
    dev_str = str(device)
    dev_type = getattr(device, "type", "") if hasattr(device, "type") else dev_str
    if dev_type != "cuda" and not dev_str.startswith("cuda"):
        return False, f"device={dev_str!r} (CUDA-only)"

    try:
        from transformers import StaticCache
    except ImportError:
        return False, "transformers does not expose StaticCache (need >=4.40)"

    # Probe construction with a 1-token cache.  Catch broadly because
    # architecture-specific issues raise a wide variety of errors:
    # AttributeError on missing config fields, ValueError on shape
    # mismatches, NotImplementedError on unsupported attention layouts.
    try:
        cfg = model.config  # type: ignore[union-attr]
        dtype = next(model.parameters()).dtype if hasattr(model, "parameters") else torch.bfloat16
        probe = StaticCache(
            cfg,  # type: ignore[arg-type]
            max_cache_len=1,
            device=device,
            dtype=dtype,
        )
        # Touch a layer to make sure the buffers actually allocated; some
        # configs accept the constructor but trip on first slice.
        if hasattr(probe, "layers") and len(probe.layers) == 0:
            return False, "StaticCache built zero layer buffers"
        del probe
    except Exception as e:
        return False, f"StaticCache construction failed: {type(e).__name__}: {e}"

    return True, None


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
    return StaticCache(
        model.config,  # type: ignore[arg-type,union-attr]
        max_cache_len=max_cache_len,
        device=device,
        dtype=dtype,
    )


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
