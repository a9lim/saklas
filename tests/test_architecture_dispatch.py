"""Per-architecture structural + hot-path coverage.

`saklas.core.model._LAYER_ACCESSORS` maps each ``model_type`` to the
attribute path of its transformer-block list.  Those mappings are the
load-bearing assumption behind *everything* saklas does — capture,
steering, monitoring — yet most entries have never been exercised
against the architecture's *actual* module tree.  When ``transformers``
refactors a model's internals (renames ``model.layers``, wraps the text
model under a composite, returns a bare tensor instead of a tuple from a
decoder block), the accessor silently points at the wrong place and
every downstream feature breaks at once.

These tests pin the contract for a curated set of families — the ones a
user is most likely to load — with **zero downloads**.  Architectures
are built from tiny synthetic configs: the meta-device path allocates no
weights at all, and the real-CPU path builds a 4-layer / hidden-64 model
that forwards in milliseconds.  Random weights make the *outputs*
meaningless, but the *plumbing* is exactly the production code path, so a
broken accessor or a changed block-output shape fails here first.

The three guarantees, per architecture:

* ``get_layers`` resolves to the full block list (accessor is correct).
* saklas's own :class:`HiddenCapture` reads a residual-stream vector of
  shape ``(hidden_size,)`` from every layer (the block returns the
  residual at ``output`` / ``output[0]`` as the hooks assume).
* a :class:`SteeringManager` vector at a middle layer actually changes
  the model's logits (the write path is wired to this block type).

To widen coverage, append a ``model_type`` to :data:`GUARANTEED_ARCHS`
once it builds from a tiny config — the parametrization does the rest.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM

import pytest

from saklas.core.model import _LAYER_ACCESSORS, get_layers
from saklas.core.hooks import HiddenCapture, SteeringManager
from saklas.core.manifold import synthesize_subspace
from saklas.core.triggers import Trigger


# Families saklas guarantees end-to-end on CPU.  Covers every architecture
# group named in the README's "Tested architectures" set plus the
# optimistically-wired entries users have asked about (granite, olmo).
# Each must build from the tiny config below; the dry-run in the test
# suite proves all three operations succeed.
GUARANTEED_ARCHS: list[str] = [
    "llama",
    "qwen2",
    "qwen3",
    "glm",
    "glm4",
    "granite",
    "granitemoe",
    "olmo2",
    "olmo3",
    "gpt_oss",
    "mistral",
    "ministral3",
    "gemma2",
]

# Deliberately tiny so the real-CPU forward is instant.  pad/bos/eos are
# pinned inside the vocab — several configs (GLM) default a special-token
# id past a small vocab and trip ``Embedding``'s ``padding_idx`` assert.
_TINY_CONFIG = dict(
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    vocab_size=512,
    head_dim=16,
    max_position_embeddings=128,
    pad_token_id=0,
    bos_token_id=1,
    eos_token_id=2,
)

# Build each real model once and reuse across the capture / steer tests —
# random init is the same per process, and every forward below is
# read-only (steering uses clones and removes its hooks), so reuse is safe.
_REAL_CACHE: dict[str, tuple[Any, Any]] = {}


def _tiny_config(model_type: str):
    return AutoConfig.for_model(model_type, **_TINY_CONFIG)


def _real_model(model_type: str):
    """Return a cached ``(model, config)`` built on CPU with random init."""
    if model_type not in _REAL_CACHE:
        cfg = _tiny_config(model_type)
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_config(cfg).eval()
        _REAL_CACHE[model_type] = (model, cfg)
    return _REAL_CACHE[model_type]


def test_guaranteed_archs_are_in_accessor_table():
    """The curated set may not silently drift out of ``_LAYER_ACCESSORS``.

    A model dropped from the table would raise on load with an
    "unsupported architecture" error; catch that here rather than in a
    user's session.
    """
    missing = [mt for mt in GUARANTEED_ARCHS if mt not in _LAYER_ACCESSORS]
    assert not missing, f"guaranteed archs missing from _LAYER_ACCESSORS: {missing}"


@pytest.mark.parametrize("model_type", GUARANTEED_ARCHS)
def test_get_layers_resolves_full_block_list(model_type: str):
    """``get_layers`` returns every transformer block, on the meta device.

    Meta instantiation allocates no weights — this is a pure structural
    check that the accessor path matches the architecture's real module
    tree for the current ``transformers`` version.
    """
    cfg = _tiny_config(model_type)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg)
    layers = get_layers(model)
    assert isinstance(layers, nn.ModuleList), (
        f"{model_type}: get_layers returned {type(layers).__name__}, "
        f"expected nn.ModuleList"
    )
    assert len(layers) == cfg.num_hidden_layers, (
        f"{model_type}: accessor found {len(layers)} blocks, config declares "
        f"{cfg.num_hidden_layers} — accessor points at the wrong attribute"
    )


@pytest.mark.parametrize("model_type", GUARANTEED_ARCHS)
def test_hidden_capture_reads_residual_per_layer(model_type: str):
    """saklas's :class:`HiddenCapture` recovers a ``(hidden_size,)`` vector
    from every layer.

    This is the read-side contract every probe / monitor depends on: the
    decoder block exposes the residual stream at ``output`` (bare tensor,
    current transformers) or ``output[0]`` (legacy tuple), and the
    last-token slice has shape ``(hidden_size,)``.
    """
    model, cfg = _real_model(model_type)
    layers = get_layers(model)
    capture = HiddenCapture()
    capture.attach(layers, list(range(len(layers))))
    try:
        with torch.no_grad():
            model(input_ids=torch.tensor([[1, 5, 9, 13]]))
    finally:
        latest = capture.latest_per_layer()
        capture.detach()

    assert set(latest) == set(range(len(layers))), (
        f"{model_type}: captured layers {sorted(latest)}, expected all "
        f"{len(layers)}"
    )
    for idx, vec in latest.items():
        assert tuple(vec.shape) == (cfg.hidden_size,), (
            f"{model_type} layer {idx}: captured shape {tuple(vec.shape)}, "
            f"expected ({cfg.hidden_size},) — block output is not the "
            f"residual stream the hooks assume"
        )


@pytest.mark.parametrize("model_type", GUARANTEED_ARCHS)
def test_steering_vector_changes_logits(model_type: str):
    """A :class:`SteeringManager` vector at a middle layer perturbs the
    model's output.

    Confirms the *write* side of the hook is wired to this block type:
    injecting at layer ``L`` must propagate through ``L+1..final`` and
    move the logits.  Random weights make the magnitude meaningless, but a
    no-op would mean the hook never modified the residual.
    """
    model, cfg = _real_model(model_type)
    layers = get_layers(model)
    ids = torch.tensor([[1, 5, 9, 13]])

    with torch.no_grad():
        baseline = model(input_ids=ids).logits.clone()

    torch.manual_seed(1)
    direction = torch.randn(cfg.hidden_size)
    direction = direction / direction.norm()

    # 4.0: a vector lowers to a rank-1 push fragment synthesized into one
    # merged affine subspace, then ``add_subspace`` → ``subspace_inject``.
    L = len(layers) // 2
    synth = synthesize_subspace(
        push=[({L: direction.reshape(1, -1)}, {L: torch.tensor([1.0])}, 1.0)],
        ablate=[], neutral_means={L: torch.zeros(cfg.hidden_size)},
    )
    mgr = SteeringManager()
    mgr.add_subspace("probe", synth, trigger=Trigger.BOTH)
    mgr.apply_to_model(layers, device=torch.device("cpu"), dtype=torch.float32)
    try:
        with torch.no_grad():
            steered = model(input_ids=ids).logits
    finally:
        mgr.clear_all()

    assert not torch.allclose(baseline, steered, atol=1e-5), (
        f"{model_type}: steering at layer {len(layers) // 2} left the logits "
        f"unchanged — the hook did not modify the residual stream"
    )
