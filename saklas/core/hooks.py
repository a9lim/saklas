"""Steering hooks for activation steering on transformer models."""

from __future__ import annotations

import math
from typing import Literal

import torch

from saklas.core.triggers import Trigger, TriggerContext


# Default rotation cap for angular injection.  ``π/2`` (90°) means user
# α=1 fully aligns the residual with the concept direction (saturating,
# no past-rotation flip).  Per the user's design choice on the
# prototype plan; users can override per-session.
DEFAULT_THETA_MAX: float = math.pi / 2

InjectionMode = Literal["angular", "additive"]

# Per-position guard for the angular Givens rotation: when the hidden
# state is already aligned (or anti-aligned) with the steering direction,
# the in-plane orthogonal axis is undefined and the rotation has no
# meaningful effect.  Below this magnitude on the orthogonal component,
# we leave the position unchanged (avoids the
# ``cos_t * h_unit + sin_t * 0 = cos_t * h_unit`` pathology that would
# silently shrink the residual norm).
_ANGULAR_PERP_EPSILON: float = 1e-6


def _angular_inplace(
    hidden: torch.Tensor,
    d_hat: torch.Tensor | None,
    cos_t: float,
    sin_t: float,
) -> None:
    """Rotate ``hidden`` in-place toward ``d_hat`` by the cached angle.

    The rotation lives in the 2D subspace spanned by each position's
    ``hidden[i]`` and the global direction ``d_hat``, with angle ``θ``
    encoded by the Python scalars ``cos_t`` / ``sin_t``.  Norm is
    preserved exactly: rotation is unitary in the (h_unit, e_perp)
    basis, and we restore the original per-position magnitude.

    No-op when ``d_hat is None`` (degenerate composed tensor) or when
    ``sin_t == 0`` (θ=0; identity rotation).  Per-position positions
    whose orthogonal component to ``d_hat`` falls below
    :data:`_ANGULAR_PERP_EPSILON` are left untouched.
    """
    if d_hat is None or sin_t == 0.0:
        return

    h_f32 = hidden.to(torch.float32)
    h_norm = torch.linalg.vector_norm(
        h_f32, dim=-1, keepdim=True,
    ).clamp_(min=1e-12)
    h_unit = h_f32 / h_norm

    d_hat_f32 = d_hat.to(torch.float32)
    cos0 = (h_unit * d_hat_f32).sum(dim=-1, keepdim=True)
    # ``d_perp`` is the in-plane vector orthogonal to ``h_unit`` pointing
    # toward ``d_hat``; its magnitude equals sin(α_h) where α_h is the
    # current angle between h and d̂.
    d_perp = d_hat_f32 - cos0 * h_unit
    d_perp_norm = torch.linalg.vector_norm(d_perp, dim=-1, keepdim=True)
    safe_norm = d_perp_norm.clamp(min=_ANGULAR_PERP_EPSILON)
    d_perp_unit = d_perp / safe_norm

    rotated_unit = cos_t * h_unit + sin_t * d_perp_unit
    # Where the perpendicular is below threshold, the rotation is
    # ill-defined; preserve h_unit there (no rotation has effect when
    # h is already on the steering axis).
    near_aligned = d_perp_norm < _ANGULAR_PERP_EPSILON
    if near_aligned.any():
        rotated_unit = torch.where(near_aligned, h_unit, rotated_unit)

    hidden.copy_((rotated_unit * h_norm).to(hidden.dtype))


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

    def latest_per_layer(self) -> dict[int, torch.Tensor]:
        """Return the most-recent capture per layer as ``[D]`` tensors.

        Used by the per-step probe-gate scorer in ``generate_steered``:
        feeds ``TraitMonitor.score_single_token`` with the latest
        hidden-state slice per layer so probe gates can consult last-
        step monitor readings.  Layers with no captures are omitted —
        the monitor handles missing layers as zero-weight contributors.
        Zero allocation other than the dict itself; the underlying
        tensors are the same `[D]` slices the hot path stored.
        """
        out: dict[int, torch.Tensor] = {}
        for idx, bucket in self._per_layer.items():
            if bucket:
                out[idx] = bucket[-1]
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

    def __init__(
        self,
        *,
        injection_mode: InjectionMode = "angular",
        theta_max: float = DEFAULT_THETA_MAX,
    ) -> None:
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
        # Injection-mode state.  Angular cache is populated only on the
        # fast path; the slow path computes per-fire from active groups.
        self.injection_mode: InjectionMode = injection_mode
        self.theta_max: float = theta_max
        # Angular fast-path cache (None unless angular + fast path).
        self._d_hat: torch.Tensor | None = None
        self._theta: float = 0.0
        self._cos_t: float = 1.0
        self._sin_t: float = 0.0
        # Slow-path angular: per-group α-budget (Σ_i |α_i| × ||baked_i||)
        # parallel to ``composed_groups`` so the hot path can sum-then-
        # rotate without re-traversing the entries list.
        self.angular_strengths: list[tuple[Trigger, float]] = []
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
        *,
        injection_mode: InjectionMode | None = None,
        theta_max: float | None = None,
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

        ``injection_mode`` and ``theta_max`` override the values stamped
        on the hook at construction.  Manager passes them on every
        ``apply_to_model`` so that flipping the session-level mode
        between calls re-warms the angular cache without rebuilding
        hook objects.

        For ``injection_mode="additive"`` the per-trigger ``composed``
        tensor is ``Σ_i α_i × baked_i`` (summed addend) — this is
        bit-identical to the v1.x hook math.  For ``"angular"`` we
        additionally cache a unit direction ``d̂`` and a Python-scalar
        rotation angle so the hot path can run the Givens rotation with
        no extra GPU ops or sync points.
        """
        self._ctx = ctx
        if injection_mode is not None:
            self.injection_mode = injection_mode
        if theta_max is not None:
            self.theta_max = theta_max

        # --- additive grouping (existing semantics) ---
        add_groups: dict[Trigger, list[tuple[torch.Tensor, float]]] = {}
        for vec, alpha, trig in additive_entries:
            add_groups.setdefault(trig, []).append((vec, alpha))

        composed_groups: list[tuple[Trigger, torch.Tensor]] = []
        # Parallel α-strength per group for angular's α/θ map.  Kept in
        # lock-step with ``composed_groups`` so slow-path sum-then-rotate
        # can recover the per-fire rotation magnitude from active groups.
        angular_strengths: list[tuple[Trigger, float]] = []
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

            # Angular rotation magnitude:
            #   strength = ||Σ_i α_i × (baked_i / ||baked_i||)||
            # Collapses to ``|α|`` for single-term (the obvious α → θ map),
            # captures cancellation when terms point opposing directions
            # (||·|| < Σ|α_i|), and stays concept-agnostic — the per-layer
            # ``||baked||`` weights set the *direction* d̂ via the
            # share-weighted ``composed``, but they don't inflate the
            # rotation magnitude itself.
            stacked_f32 = stacked.to(torch.float32)
            baked_norms = torch.linalg.vector_norm(
                stacked_f32, dim=-1,
            ).clamp(min=1e-12)
            unit_stacked = stacked_f32 / baked_norms.unsqueeze(-1)
            alphas_f32 = alphas_t.to(torch.float32)
            composed_unit_sum = (
                alphas_f32.unsqueeze(1) * unit_stacked
            ).sum(dim=0)
            strength_t = torch.linalg.vector_norm(composed_unit_sum)
            angular_strengths.append((trig, float(strength_t.item())))

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
        self.angular_strengths = angular_strengths

        # --- fast-path collapse decision ---
        if not composed_groups and not ablation_groups:
            self.composed = None
            self.composed_groups = []
            self._d_hat = None
            return

        # Fast path only when the single contributor is additive/BOTH and
        # no ablation is attached.  Any ablation forces the slow path so
        # the hook_fn can sequence ablation-then-additive.
        if (
            not ablation_groups
            and len(composed_groups) == 1
            and composed_groups[0][0] == Trigger.BOTH
        ):
            self.composed = composed_groups[0][1]
            self.composed_groups = []
            # Pre-compute angular fast-path cache so the hot path skips
            # the GPU sync of ``vector_norm(composed).item()``.
            self._refresh_angular_cache(angular_strengths[0][1])
        else:
            self.composed = None
            self.composed_groups = composed_groups
            self._d_hat = None

    def _refresh_angular_cache(self, theta_strength: float) -> None:
        """Populate ``_d_hat`` / ``_theta`` / ``_cos_t`` / ``_sin_t``
        from the current ``self.composed`` tensor.

        Only called on the fast path (single-group BOTH).  The slow
        path computes per-fire from active groups.

        ``theta_strength = ||Σ_i α_i × (baked_i / ||baked_i||)||`` is the
        rotation magnitude (clamped to ``θ_max``).  The caller passes it
        from the parallel ``angular_strengths`` list so we don't
        re-traverse the entries.  Falls back to identity (no rotation)
        when either ``composed`` or the strength is degenerate.
        """
        composed = self.composed
        if composed is None or theta_strength <= 1e-12:
            self._d_hat = None
            self._theta = 0.0
            self._cos_t = 1.0
            self._sin_t = 0.0
            return
        c_f32 = composed.detach().to(torch.float32)
        c_norm_t = torch.linalg.vector_norm(c_f32)
        c_norm = float(c_norm_t.item())
        if c_norm <= 1e-12:
            # All α-weighted terms cancelled out; treat as no-op.
            self._d_hat = None
            self._theta = 0.0
            self._cos_t = 1.0
            self._sin_t = 0.0
            return
        # ``d_hat`` lives on the same device/dtype as ``composed`` so the
        # hot-path projection broadcasts cleanly against ``hidden``.
        # We use the share-baked composed tensor here — high-share layers
        # contribute more to the rotation *target*, but the rotation
        # *magnitude* stays concept-agnostic via ``theta_strength``.
        self._d_hat = (c_f32 / c_norm).to(dtype=composed.dtype)
        ratio = theta_strength
        if ratio > 1.0:
            ratio = 1.0
        self._theta = ratio * self.theta_max
        self._cos_t = math.cos(self._theta)
        self._sin_t = math.sin(self._theta)

    def hook_fn(self, module, input, output):
        # Fast path: single composed additive tensor, no ablation, no
        # trigger check.
        if self.composed is not None:
            hidden = output if isinstance(output, torch.Tensor) else output[0]
            if self.injection_mode == "angular":
                _angular_inplace(
                    hidden,
                    self._d_hat,
                    self._cos_t,
                    self._sin_t,
                )
            else:
                # Additive (legacy): in-place add + norm-preserving rescale.
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

        if self.injection_mode == "additive":
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

        # --- angular slow path ---
        # Ablation runs unchanged (mean-replace lives in additive space).
        # Then sum active additive groups into one composed tensor and
        # rotate once toward its direction.  Skipping the post-additive
        # norm rescale because rotation is exactly norm-preserving.
        for trig, D_unit, m, alpha_vec in abl_groups:
            if not trig.active(ctx):
                continue
            coeffs = hidden @ D_unit.T
            coeffs.sub_(m).mul_(alpha_vec)
            hidden.sub_(coeffs @ D_unit)

        active_composed: torch.Tensor | None = None
        active_strength: float = 0.0
        for (trig, composed), (_, strength) in zip(
            add_groups, self.angular_strengths,
        ):
            if not trig.active(ctx):
                continue
            active_composed = (
                composed if active_composed is None else active_composed + composed
            )
            # Approximation for cross-trigger strengths: sum.  Triggers
            # are typically mutually exclusive (BEFORE_THINKING vs
            # AFTER_THINKING) so this is the cooperating-cooperating
            # case; opposing trigger groups would over-rotate, but
            # constructing one in practice requires explicit user intent.
            active_strength += strength

        if active_composed is None or active_strength <= 1e-12:
            return output

        c_f32 = active_composed.to(torch.float32)
        c_norm_t = torch.linalg.vector_norm(c_f32)
        c_norm = float(c_norm_t.item())
        if c_norm <= 1e-12:
            return output
        d_hat = (c_f32 / c_norm).to(dtype=hidden.dtype)
        ratio = active_strength
        if ratio > 1.0:
            ratio = 1.0
        theta = ratio * self.theta_max
        _angular_inplace(hidden, d_hat, math.cos(theta), math.sin(theta))
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

    def __init__(
        self,
        *,
        injection_mode: InjectionMode = "angular",
        theta_max: float = DEFAULT_THETA_MAX,
    ) -> None:
        self.hooks: dict[int, SteeringHook] = {}
        self.vectors: dict[str, dict] = {}
        self.ablations: dict[str, dict] = {}
        self.ctx: TriggerContext = TriggerContext()
        self.injection_mode: InjectionMode = injection_mode
        self.theta_max: float = theta_max

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

    def add_ablation(
        self,
        name: str,
        profile: dict[int, torch.Tensor],
        alpha: float,
        trigger: Trigger,
        layer_means: dict[int, torch.Tensor],
    ) -> None:
        """Register a mean-replacement ablation target.

        At ``apply_to_model`` time, for every layer present in both
        ``profile`` and ``layer_means``, a per-layer
        ``(baked_direction, layer_mean, alpha, trigger)`` entry is attached
        to the corresponding :class:`SteeringHook`.  Profile layers missing
        from ``layer_means`` are silently skipped — ablation without a
        baseline mean is undefined.
        """
        self.ablations[name] = {
            "profile": profile,
            "alpha": alpha,
            "trigger": trigger,
            "layer_means": layer_means,
        }

    def apply_to_model(
        self,
        model_layers: torch.nn.ModuleList,
        device: torch.device,
        dtype: torch.dtype,
        *,
        injection_mode: InjectionMode | None = None,
        theta_max: float | None = None,
    ) -> None:
        """Group entries by layer, recompose hooks, attach to model.

        ``injection_mode`` and ``theta_max`` override the manager-level
        values stamped at construction.  Defaulting both to ``None``
        means "use whatever was set on the manager"; callers (the
        session) pass concrete values when the per-call steering carries
        an override.

        Under ``injection_mode="additive"`` the user α is multiplied by
        :data:`_STEER_GAIN` so the v1.x steering scale is reproduced
        bit-identically.  Under ``"angular"`` the gain is dropped — α
        maps directly to a rotation angle (the user's reason for
        switching modes in the first place).
        """
        if injection_mode is not None:
            self.injection_mode = injection_mode
        if theta_max is not None:
            self.theta_max = theta_max

        additive_by_layer: dict[
            int, list[tuple[torch.Tensor, float, Trigger]]
        ] = {}
        for v in self.vectors.values():
            profile = v["profile"]
            user_alpha = v["alpha"]
            trigger = v.get("trigger", Trigger.BOTH)

            if self.injection_mode == "angular":
                # Share-weight α per layer so per-layer θ scales with
                # ``share_L = ||baked_L|| / Σ_L' ||baked_L'||``.  Without
                # this, every layer rotates by the same |α|·θ_max and the
                # rotations compound through the residual stream — for an
                # N-layer model the cumulative rotation lands at ~N·|α|·
                # θ_max, which crashes coherence at any nontrivial α.
                # With share-weighting, the per-layer ``Σ θ_L`` collapses
                # to ``|α|·θ_max``, matching the additive-path intuition
                # that high-signal layers carry most of the steering and
                # the user-facing α is bounded in the same band.
                norms_total = 0.0
                norms: dict[int, float] = {}
                for layer_idx, vec in profile.items():
                    n = float(
                        torch.linalg.vector_norm(
                            vec.to(torch.float32),
                        ).item()
                    )
                    norms[layer_idx] = n
                    norms_total += n
                if norms_total <= 1e-12:
                    # Profile is degenerate (all-zero); skip silently.
                    continue
                for layer_idx, vec in profile.items():
                    share_L = norms[layer_idx] / norms_total
                    additive_by_layer.setdefault(layer_idx, []).append(
                        (vec, user_alpha * share_L, trigger),
                    )
            else:
                # Additive (legacy): the v1.x ``α × _STEER_GAIN`` math.
                # Share is already baked into ``||baked_L||`` so per-layer
                # weighting falls out of the magnitude — no extra scaling
                # needed at apply time.
                effective_alpha = user_alpha * _STEER_GAIN
                for layer_idx, vec in profile.items():
                    additive_by_layer.setdefault(layer_idx, []).append(
                        (vec, effective_alpha, trigger),
                    )

        ablation_by_layer: dict[
            int, list[tuple[torch.Tensor, torch.Tensor, float, Trigger]]
        ] = {}
        for a in self.ablations.values():
            alpha = a["alpha"]
            trigger = a["trigger"]
            means = a["layer_means"]
            for layer_idx, vec in a["profile"].items():
                if layer_idx not in means:
                    continue
                ablation_by_layer.setdefault(layer_idx, []).append(
                    (vec, means[layer_idx], alpha, trigger),
                )

        active_layers = set(additive_by_layer) | set(ablation_by_layer)

        # Detach hooks for layers that no longer have any contribution.
        for idx in list(self.hooks):
            if idx not in active_layers:
                self.hooks[idx].detach()
                del self.hooks[idx]

        for idx in active_layers:
            if idx not in self.hooks:
                hook = SteeringHook(
                    injection_mode=self.injection_mode,
                    theta_max=self.theta_max,
                )
                hook.attach(model_layers[idx])
                self.hooks[idx] = hook
            self.hooks[idx].recompose(
                additive_entries=additive_by_layer.get(idx, []),
                ablation_entries=ablation_by_layer.get(idx, []),
                injection_mode=self.injection_mode,
                theta_max=self.theta_max,
                device=device, dtype=dtype, ctx=self.ctx,
            )

    def clear_all(self) -> None:
        """Detach all hooks and clear vectors + ablations."""
        for hook in self.hooks.values():
            hook.detach()
        self.hooks.clear()
        self.vectors.clear()
        self.ablations.clear()
