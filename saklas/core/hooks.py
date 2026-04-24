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
    """Pre-composed steering vectors and ablation data for a single layer.

    Fast path (``Trigger.BOTH`` additive only, no ablation): a single
    composed tensor is added unconditionally at hook time — no per-step
    trigger check.

    Slow path: entries are grouped by trigger equality into
    ``composed_groups`` (additive) and ``ablation_groups`` (mean
    replacement). At hook time, ablation groups fire first, then additive
    groups; the unconditional norm-preservation rescale wraps the combined
    op.
    """

    def __init__(self) -> None:
        # Populated on the fast path (BOTH only, no ablation). Mutually
        # exclusive with ``composed_groups``/``ablation_groups`` — recompose
        # sets exactly one code path live.
        self.composed: torch.Tensor | None = None
        # Slow path: list of (trigger, composed_tensor) pairs. Iterated per
        # hook call; each group's trigger is consulted against ``_ctx``.
        self.composed_groups: list[tuple[Trigger, torch.Tensor]] = []
        # Ablation groups: (Trigger, D_unit [K,dim], m [K], alpha [K]).
        # D_unit rows are per-direction unit vectors; m[k] = μ_L · d̂_k;
        # alpha[k] is the user coefficient (no _STEER_GAIN — ablation is
        # a conservative replace, not a tunable push).
        self.ablation_groups: list[
            tuple[Trigger, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = []
        # Shared mutable context threaded in by SteeringManager.  Read-only
        # from the hook's perspective; the generation loop mutates fields.
        self._ctx: TriggerContext | None = None
        self._handle = None

    def recompose(
        self,
        additive_entries: list[tuple[torch.Tensor, float, Trigger]],
        ablation_entries: list[tuple[torch.Tensor, torch.Tensor, float, Trigger]],
        device: torch.device,
        dtype: torch.dtype,
        ctx: TriggerContext,
    ) -> None:
        """Pre-compose additive and ablation state for this layer.

        ``additive_entries`` are ``(baked_direction, effective_alpha,
        trigger)`` triples; entries sharing a trigger value (dataclass
        equality) collapse into one composed tensor.  ``ablation_entries``
        are ``(baked_direction, layer_mean, user_alpha, trigger)``
        quadruples; per-trigger groups collapse into one stacked-direction
        matrix with companion mean-scalar and coefficient vectors.  ``ctx``
        is the shared per-generation TriggerContext mutated by the
        generation loop and read here at hook-fire time.
        """
        self._ctx = ctx

        # --- additive grouping (existing semantics) ---
        add_groups: dict[Trigger, list[tuple[torch.Tensor, float]]] = {}
        for vec, alpha, trig in additive_entries:
            add_groups.setdefault(trig, []).append((vec, alpha))

        composed_groups: list[tuple[Trigger, torch.Tensor]] = []
        for trig, vecs in add_groups.items():
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

        # --- ablation grouping ---
        abl_groups: dict[
            Trigger, list[tuple[torch.Tensor, torch.Tensor, float]]
        ] = {}
        for baked, layer_mean, alpha, trig in ablation_entries:
            # Zero alpha ⇒ no-op ablation; drop at compose time so the hot
            # path never iterates dead rows.
            if alpha == 0.0:
                continue
            abl_groups.setdefault(trig, []).append((baked, layer_mean, alpha))

        ablation_groups: list[
            tuple[Trigger, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = []
        for trig, rows in abl_groups.items():
            # Compute each unit direction + mean scalar in fp32 for
            # stability (fp16 sum-of-squares overflows at hidden_dim ≥ 2048,
            # and mean projections can be close to zero), then cast to hook
            # dtype for the hot path.
            d_units_f32: list[torch.Tensor] = []
            m_vals_f32: list[torch.Tensor] = []
            alphas_list: list[float] = []
            for baked, layer_mean, alpha in rows:
                b32 = baked.to(device=device, dtype=torch.float32)
                m32 = layer_mean.to(device=device, dtype=torch.float32)
                n = torch.linalg.vector_norm(b32).clamp(min=1e-12)
                d_hat = b32 / n
                d_units_f32.append(d_hat)
                m_vals_f32.append((m32 * d_hat).sum())
                alphas_list.append(alpha)
            D_unit = torch.stack(d_units_f32).to(dtype=dtype)
            m = torch.stack(m_vals_f32).to(dtype=dtype)
            alpha_vec = torch.tensor(alphas_list, device=device, dtype=dtype)
            ablation_groups.append((trig, D_unit, m, alpha_vec))

        self.ablation_groups = ablation_groups

        # --- fast-path collapse decision ---
        if not composed_groups and not ablation_groups:
            self.composed = None
            self.composed_groups = []
            return

        # Fast path only when the single contributor is additive/BOTH and
        # no ablation is attached.  Any ablation forces the slow path so
        # the hook_fn rewrite (Task 8) can sequence ablation-then-additive.
        if (
            not ablation_groups
            and len(composed_groups) == 1
            and composed_groups[0][0] == Trigger.BOTH
        ):
            self.composed = composed_groups[0][1]
            self.composed_groups = []
        else:
            self.composed = None
            self.composed_groups = composed_groups

    def hook_fn(self, module, input, output):
        # Fast path: single composed additive tensor, no ablation, no
        # trigger check — unconditional norm preservation.
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

        add_groups = self.composed_groups
        abl_groups = self.ablation_groups
        if not add_groups and not abl_groups:
            return output
        ctx = self._ctx
        if ctx is None:
            return output

        # Cheap pre-check: any group active this step? Skip the fp32 norm
        # capture entirely if not (e.g. AFTER_THINKING during prefill).
        any_active = False
        for trig, *_ in abl_groups:
            if trig.active(ctx):
                any_active = True
                break
        if not any_active:
            for trig, _ in add_groups:
                if trig.active(ctx):
                    any_active = True
                    break
        if not any_active:
            return output

        hidden = output if isinstance(output, torch.Tensor) else output[0]
        norm_pre = torch.linalg.vector_norm(
            hidden, dim=-1, keepdim=True, dtype=torch.float32,
        )

        # Ablation first: replace the component along each d̂ with the
        # neutral-baseline mean (α · (h·d̂ - μ·d̂) subtracted per direction).
        for trig, D_unit, m, alpha_vec in abl_groups:
            if not trig.active(ctx):
                continue
            coeffs = hidden @ D_unit.T
            coeffs.sub_(m).mul_(alpha_vec)
            hidden.sub_(coeffs @ D_unit)

        # Additive second: inject into the already-cleaned residual stream.
        for trig, composed in add_groups:
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

            self.hooks[idx].recompose(
                additive_entries=entries,
                ablation_entries=[],
                device=device,
                dtype=dtype,
                ctx=self.ctx,
            )

    def clear_all(self) -> None:
        """Detach all hooks and clear vectors."""
        for hook in self.hooks.values():
            hook.detach()
        self.hooks.clear()
        self.vectors.clear()
