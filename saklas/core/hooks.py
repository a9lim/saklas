"""Steering hooks for activation steering on transformer models."""

from __future__ import annotations

import torch

from saklas.core.triggers import Trigger, TriggerContext


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
    """Pre-composed steering vectors for a single layer, grouped by trigger.

    Fast path (``Trigger.BOTH`` only): a single composed tensor is added
    unconditionally at hook time — no per-step trigger check.

    Slow path (any non-``BOTH`` trigger): entries are grouped by trigger
    equality into ``composed_groups``; each group has its own pre-composed
    tensor. At hook time, only groups whose ``trigger.active(ctx)`` returns
    True contribute, and the norm-preservation rescale wraps the conditional
    sum. Groups that would sum to zero are dropped at recompose time so the
    hot path never pays for dead weight.
    """

    def __init__(self) -> None:
        # Populated on the fast path (BOTH only). Mutually exclusive with
        # ``composed_groups`` — recompose sets exactly one of them.
        self.composed: torch.Tensor | None = None
        # Slow path: list of (trigger, composed_tensor) pairs. Iterated per
        # hook call; each group's trigger is consulted against ``_ctx``.
        self.composed_groups: list[tuple[Trigger, torch.Tensor]] = []
        # Shared mutable context threaded in by SteeringManager.  Read-only
        # from the hook's perspective; the generation loop mutates fields.
        self._ctx: TriggerContext | None = None
        self._handle = None

    def recompose(
        self,
        entries: list[tuple[torch.Tensor, float, Trigger]],
        device: torch.device,
        dtype: torch.dtype,
        ctx: TriggerContext,
    ) -> None:
        """Pre-compose per-trigger groups of steering vectors for this layer.

        ``entries`` is a list of ``(baked_direction, effective_alpha,
        trigger)`` triples; entries sharing a trigger value (dataclass
        equality) collapse into one composed tensor.  ``ctx`` is the
        shared per-generation TriggerContext mutated by the generation
        loop and read here at hook-fire time.
        """
        self._ctx = ctx
        if not entries:
            self.composed = None
            self.composed_groups = []
            return

        # Group by trigger value (Trigger is a frozen dataclass, hashable).
        # Preserve the first-seen trigger instance per group so equality-
        # equivalent but distinct Trigger objects still share storage.
        groups: dict[Trigger, list[tuple[torch.Tensor, float]]] = {}
        for vec, alpha, trig in entries:
            groups.setdefault(trig, []).append((vec, alpha))

        composed_groups: list[tuple[Trigger, torch.Tensor]] = []
        for trig, vecs in groups.items():
            # All-zero alphas → group contributes nothing; skip the matmul
            # so that a stale entry with alpha=0 doesn't inject NaN on any
            # bad-extraction vectors it carries.
            if all(alpha == 0.0 for _, alpha in vecs):
                continue
            stacked = torch.stack(
                [v.to(device=device, dtype=dtype) for v, _ in vecs]
            )
            alphas_t = torch.tensor(
                [alpha for _, alpha in vecs], device=device, dtype=dtype,
            )
            composed = (alphas_t.unsqueeze(1) * stacked).sum(dim=0)
            composed_groups.append((trig, composed))

        if not composed_groups:
            self.composed = None
            self.composed_groups = []
            return

        # Fast-path collapse: all contributions use Trigger.BOTH (or an
        # equality-equivalent default Trigger()).  One tensor, no per-step
        # .active() check.
        if len(composed_groups) == 1 and composed_groups[0][0] == Trigger.BOTH:
            self.composed = composed_groups[0][1]
            self.composed_groups = []
        else:
            self.composed = None
            self.composed_groups = composed_groups

    def hook_fn(self, module, input, output):
        # Fast path: single composed tensor, no trigger check,
        # unconditional norm preservation.
        if self.composed is not None:
            hidden = output if isinstance(output, torch.Tensor) else output[0]
            norm_pre = torch.linalg.vector_norm(
                hidden, dim=-1, keepdim=True, dtype=torch.float32,
            )
            hidden.add_(self.composed)
            norm_post = torch.linalg.vector_norm(
                hidden, dim=-1, keepdim=True, dtype=torch.float32,
            ).clamp_(min=1e-6)
            hidden.mul_((norm_pre / norm_post).to(hidden.dtype))
            return output

        # Slow path: trigger-gated groups. Pre-check whether any group
        # fires this step — if none do, skip the norm capture entirely.
        groups = self.composed_groups
        if not groups:
            return output
        ctx = self._ctx
        # ctx should always be non-None when composed_groups is populated
        # (SteeringManager.apply_to_model threads it in via recompose),
        # but defend against a stale-handle case by treating missing ctx
        # as "no triggers fire" rather than panicking in the hot path.
        if ctx is None:
            return output

        # First pass: does anything fire? Cheap — a handful of int compares
        # per group, no tensor work. Saves the fp32 norm round-trip when
        # no group is currently active (e.g. AFTER_THINKING during prefill).
        any_active = False
        for trig, _ in groups:
            if trig.active(ctx):
                any_active = True
                break
        if not any_active:
            return output

        hidden = output if isinstance(output, torch.Tensor) else output[0]
        norm_pre = torch.linalg.vector_norm(
            hidden, dim=-1, keepdim=True, dtype=torch.float32,
        )
        for trig, composed in groups:
            if trig.active(ctx):
                hidden.add_(composed)
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
#
# Recalibrated from 3.5 → 2.0 when extract_contrastive gained the
# drop_edges=(2, 2) default.  Edge-drop removes L0/L1 and L_N-2/L_N-1
# from the share distribution (early layers carry tokenization/lexical
# features; late layers are unembedding-aligned — steering either corrupts
# surface form rather than latent meaning).  Remaining middle layers'
# shares inflate ~10-15% after redistribution, and the post-drop coherence
# ratio rises (directions align tighter), so per-α directional rotation
# increases.  2.0 pushes the cliff above α≈0.9 on the reference model,
# giving users a wide coherent band (~0.3-0.85) to dial in steering
# intensity and leaving generous headroom for the long tail of untested
# architectures.
_STEER_GAIN = 2.0


class SteeringManager:
    """Manages multiple SteeringHooks across model layers.

    Owns the per-generation :class:`TriggerContext` consumed by every
    attached :class:`SteeringHook`.  The generation loop mutates the
    context's fields at lifecycle boundaries (prefill → decode, thinking
    transitions, per-step counter); hooks read them to decide which
    trigger-gated groups contribute at each forward.
    """

    def __init__(self) -> None:
        self.hooks: dict[int, SteeringHook] = {}
        self.vectors: dict[str, dict] = {}
        self.ctx: TriggerContext = TriggerContext()

    def add_vector(
        self,
        name: str,
        profile: dict[int, torch.Tensor],
        alpha: float,
        trigger: Trigger = Trigger.BOTH,
    ) -> None:
        self.vectors[name] = {
            "profile": profile,
            "alpha": alpha,
            "trigger": trigger,
        }

    def apply_to_model(
        self,
        model_layers: torch.nn.ModuleList,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Group vectors by layer, recompose hooks, attach to model."""
        by_layer: dict[int, list[tuple[torch.Tensor, float, Trigger]]] = {}
        for v in self.vectors.values():
            effective_alpha = v["alpha"] * _STEER_GAIN
            trigger = v.get("trigger", Trigger.BOTH)
            for layer_idx, vec in v["profile"].items():
                by_layer.setdefault(layer_idx, []).append(
                    (vec, effective_alpha, trigger),
                )

        # Detach hooks for layers that no longer have vectors
        for idx in list(self.hooks):
            if idx not in by_layer:
                self.hooks[idx].detach()
                del self.hooks[idx]

        # Recompose and attach for each active layer
        for idx, entries in by_layer.items():
            if idx not in self.hooks:
                hook = SteeringHook()
                hook.attach(model_layers[idx])
                self.hooks[idx] = hook

            self.hooks[idx].recompose(entries, device, dtype, self.ctx)

    def clear_all(self) -> None:
        """Detach all hooks and clear vectors."""
        for hook in self.hooks.values():
            hook.detach()
        self.hooks.clear()
        self.vectors.clear()
