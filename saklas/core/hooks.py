"""Steering hooks for activation steering on transformer models."""

from __future__ import annotations

import torch


class HiddenCapture:
    """Accumulates the last-position hidden state at each hooked layer on every
    forward pass. Paired with a KV-cached generation loop, one capture per step
    gives N captures for N generated tokens: capture[k] is the state that
    produced token t_k.

    The first capture (step 0, prompt forward) is the state at the last prompt
    token — the state that selected t_0. Subsequent steps feed one generated
    token at a time; each hidden state is the model's state that selected the
    following token. The k-th capture is thus semantically "the activation that
    produced generated token k."

    Hot-path discipline: hooks copy a (dim,) slice via ``detach().clone()``
    (device-local, no sync) and append to a per-layer Python list. Stacking and
    fp32 casting happen after detach, not in the hot path.
    """

    def __init__(self) -> None:
        self._per_layer: dict[int, list[torch.Tensor]] = {}
        self._handles: list = []

    def attach(
        self, layers: "torch.nn.ModuleList", layer_indices: list[int]
    ) -> None:
        self._per_layer = {idx: [] for idx in layer_indices}
        self._handles = []
        for idx in layer_indices:
            bucket = self._per_layer[idx]

            def _make(bucket_ref):
                def _hook(module, input, output):
                    h = output if isinstance(output, torch.Tensor) else output[0]
                    bucket_ref.append(h[0, -1, :].detach().clone())
                return _hook

            self._handles.append(layers[idx].register_forward_hook(_make(bucket)))

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def clear(self) -> None:
        self._per_layer = {}
        self._handles = []

    def stacked(self) -> dict[int, torch.Tensor]:
        """Return per-layer ``(n_captures, dim)`` tensors in the capture dtype.

        Scoring code casts to fp32 via the monitor's normalize helper.
        """
        out: dict[int, torch.Tensor] = {}
        for idx, bucket in self._per_layer.items():
            if bucket:
                out[idx] = torch.stack(bucket)
        return out


class SteeringHook:
    """Pre-composed steering vector for a single layer."""

    def __init__(self) -> None:
        self.composed: torch.Tensor | None = None
        self._handle = None

    def recompose(
        self,
        vectors: list[tuple[torch.Tensor, float]],
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Pre-compose all vectors for this layer into a single tensor.

        ``vectors`` is a list of ``(baked_direction, effective_alpha)`` pairs;
        multiple entries occur when several named profiles contribute to the
        same layer (different alphas) and are summed linearly.
        """
        if not vectors:
            self.composed = None
            return
        # All-zero alphas → no perturbation; skip the matmul so that
        # 0 * NaN (from a bad extraction) doesn't inject NaN into hooks.
        if all(alpha == 0.0 for _, alpha in vectors):
            self.composed = None
            return
        stacked = torch.stack(
            [vec.to(device=device, dtype=dtype) for vec, _ in vectors]
        )
        alphas = torch.tensor([alpha for _, alpha in vectors], device=device, dtype=dtype)
        self.composed = (alphas.unsqueeze(1) * stacked).sum(dim=0)

    def hook_fn(self, module, input, output):
        if self.composed is None:
            return output
        hidden = output if isinstance(output, torch.Tensor) else output[0]
        # Norm preservation: rescale each position back to its pre-injection
        # magnitude after adding the steering vector.  Keeps the residual
        # stream norm on its natural trajectory, which prevents the
        # "crank alpha → logit explosion → gibberish" failure mode at high
        # user alpha.  Per-token in the unmodified add step, so only the
        # magnitude is clamped — the direction still moves toward the
        # steering pole proportionally to alpha.
        #
        # vector_norm(dtype=fp32) upcasts the accumulator without
        # materializing an fp32 copy of the hidden tensor, which matters
        # at hidden_dim >= 2048 where fp16 sum-of-squares overflows.
        norm_pre = torch.linalg.vector_norm(
            hidden, dim=-1, keepdim=True, dtype=torch.float32,
        )
        hidden.add_(self.composed)
        norm_post = torch.linalg.vector_norm(
            hidden, dim=-1, keepdim=True, dtype=torch.float32,
        ).clamp_(min=1e-6)
        hidden.mul_((norm_pre / norm_post).to(hidden.dtype))
        return output

    def attach(self, layer_module: torch.nn.Module) -> None:
        """Register forward hook on a layer module."""
        self._handle = layer_module.register_forward_hook(self.hook_fn)

    def detach(self) -> None:
        """Remove the forward hook."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


# Global gain that pins the user-facing alpha scale.  Per-layer shares
# (score_i / sum(scores)) are baked into the stored direction magnitudes
# at extraction time, so the hook math collapses to a single flat scalar:
#
#     effective_injection = user_alpha * _STEER_GAIN * baked_direction_i
#
# The same two invariances fall out of the baking step (they just moved
# from apply-time to extract-time):
#
#   - Layer-count invariance: total injection is independent of n_layers,
#     so models of different depths hit the same behavioral effect at the
#     same user alpha.  Without this, deeper models over-inject (e.g.
#     gemma-4-E4B at 42 layers vs gemma-4-31B at 60 layers would drift
#     by ~1.5× in coherent α).
#
#   - Score-magnitude invariance: absolute PCA scores vary wildly between
#     architectures (diffuse Llama-3.2-3B ≈0.07, sharp gemma-3-4b ≈0.25
#     for the same pairs), but only the *relative* per-layer shares
#     matter here — high-signal layers still get proportionally more
#     push within a profile.
#
# The gain is calibrated so that a user alpha of ~0.5 lands in the
# coherent steering band on the reference model (gemma-4-31B-it) for
# the bundled 21-probe pack.  Raising the gain shifts every model's
# coherent α lower; lowering does the opposite.  Smaller or non-standard-
# geometry models (MatFormer, MoE, heavily safety-trained) may still need
# proportionally higher alpha due to residual architectural effects
# (activation magnitude, attention layout) this normalization doesn't
# capture.
_STEER_GAIN = 3.5


class SteeringManager:
    """Manages multiple SteeringHooks across model layers."""

    def __init__(self) -> None:
        self.hooks: dict[int, SteeringHook] = {}
        self.vectors: dict[str, dict] = {}

    def add_vector(
        self,
        name: str,
        profile: dict[int, torch.Tensor],
        alpha: float,
    ) -> None:
        self.vectors[name] = {
            "profile": profile,
            "alpha": alpha,
        }

    def apply_to_model(
        self,
        model_layers: torch.nn.ModuleList,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Group vectors by layer, recompose hooks, attach to model."""
        by_layer: dict[int, list[tuple[torch.Tensor, float]]] = {}
        for v in self.vectors.values():
            effective_alpha = v["alpha"] * _STEER_GAIN
            for layer_idx, vec in v["profile"].items():
                by_layer.setdefault(layer_idx, []).append((vec, effective_alpha))

        # Detach hooks for layers that no longer have vectors
        for idx in list(self.hooks):
            if idx not in by_layer:
                self.hooks[idx].detach()
                del self.hooks[idx]

        # Recompose and attach for each active layer
        for idx, pairs in by_layer.items():
            if idx not in self.hooks:
                hook = SteeringHook()
                hook.attach(model_layers[idx])
                self.hooks[idx] = hook

            self.hooks[idx].recompose(pairs, device, dtype)

    def clear_all(self) -> None:
        """Detach all hooks and clear vectors."""
        for hook in self.hooks.values():
            hook.detach()
        self.hooks.clear()
        self.vectors.clear()

