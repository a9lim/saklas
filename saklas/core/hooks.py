"""Steering hooks for activation steering on transformer models."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, Literal

import torch

from saklas.core.manifold import (
    eval_rbf,
    rotate_toward,
    subspace_replace,
    subspace_rotate,
)
from saklas.core.triggers import Trigger, TriggerContext


# Default rotation cap for angular injection.  ``π/2`` (90°) means user
# α=1 fully aligns the residual with the concept direction (saturating,
# no past-rotation flip).  Per the user's design choice on the
# prototype plan; users can override per-session.
DEFAULT_THETA_MAX: float = math.pi / 2

InjectionMode = Literal["angular", "additive"]


def _angular_inplace(
    hidden: torch.Tensor,
    d_hat: torch.Tensor | None,
    cos_t: float | torch.Tensor | None,
    sin_t: float | torch.Tensor | None,
) -> None:
    """Rotate ``hidden`` in-place toward ``d_hat`` by the cached angle.

    The rotation lives in the 2D subspace spanned by each position's
    ``hidden[i]`` and the global direction ``d_hat``, with angle ``θ``
    encoded by ``cos_t`` / ``sin_t``.  Both may be Python floats (the
    direct-call test path) or 0-dim tensors (the hook fast path under
    ``torch.compile``: pinning the angle to a tensor input avoids
    recompiles when α changes between generations and avoids constant-
    folding the angle into a CUDA-graph capture).

    Norm is preserved exactly: rotation is unitary in the
    ``(h_unit, e_perp)`` basis, and we restore the original per-position
    magnitude.

    No-op when ``d_hat is None`` (degenerate composed tensor —
    :meth:`SteeringHook._refresh_angular_cache` sets this when α is
    effectively zero, so the ``sin_t == 0`` Python guard the v2.1 entry
    used to carry is unnecessary and would force a CPU sync if ``sin_t``
    is a tensor).  Positions already (anti-)aligned with ``d_hat`` are
    left untouched by :func:`~saklas.core.manifold.rotate_toward`'s own
    near-aligned guard.

    The rotation itself is :func:`~saklas.core.manifold.rotate_toward` —
    the Givens kernel shared with manifold angular steering
    (:func:`~saklas.core.manifold.subspace_rotate`).  This path applies it
    to the *full* unit activation toward the concept direction ``d̂``;
    manifold steering applies the same kernel to the centered in-subspace
    component.  Only the magnitude extraction (``‖h‖`` here) and reassembly
    differ.
    """
    if d_hat is None or cos_t is None or sin_t is None:
        return

    h_f32 = hidden.to(torch.float32)
    h_norm = torch.linalg.vector_norm(
        h_f32, dim=-1, keepdim=True,
    ).clamp_(min=1e-12)
    h_unit = h_f32 / h_norm

    d_hat_f32 = d_hat.to(torch.float32)
    # ``w_norm`` (the near-aligned tell) is unused on the vector path —
    # rotate_toward's internal ``where`` is the whole guard needed here;
    # only manifold steering folds it into a wider degeneracy condition.
    rotated_unit, _w_norm = rotate_toward(h_unit, d_hat_f32, cos_t, sin_t)

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

    Incremental mode (``set_incremental``): for the common monitored case
    the full ``[T, D]`` stack is never needed — the session scores each
    token as it is produced and keeps only the per-token score rows. In
    this mode each per-layer hook OVERWRITES its bucket (length-1) instead
    of appending, so device memory stays O(layers·D) for the whole
    generation, and a ``step_sink`` callback fires once per forward (after
    the highest hooked layer stores its slice) with the latest per-layer
    slice. ``latest_per_layer()`` and the ``bucket[-1]`` reads the
    streaming tap relies on keep working — a length-1 bucket's ``[-1]`` is
    still the latest. Non-incremental mode is byte-identical to the append
    path: ``stacked()`` returns the full ``[T, D]``.
    """

    def __init__(self) -> None:
        self._per_layer: dict[int, list[torch.Tensor]] = {}
        self._handles: list[Any] = []
        # Incremental-mode state. ``_incremental`` flips the per-layer
        # hook from append to overwrite; ``_step_sink`` is invoked once
        # per forward after the highest hooked layer (``_max_layer``)
        # stores this step's slice. All three reset on attach/clear.
        self._incremental: bool = False
        self._step_sink: Callable[[dict[int, torch.Tensor]], None] | None = None
        self._max_layer: int | None = None

    def attach(
        self, layers: "torch.nn.ModuleList", layer_indices: list[int]
    ) -> None:
        self._per_layer = {idx: [] for idx in layer_indices}
        self._handles = []
        # Attach resets incremental state — a fresh capture starts in the
        # append (full-retention) mode. ``set_incremental`` must be called
        # after attach to opt into incremental scoring for this gen.
        self._incremental = False
        self._step_sink = None
        self._max_layer = None
        for idx in layer_indices:
            bucket = self._per_layer[idx]

            def _make(bucket_ref: list[torch.Tensor], layer_idx: int) -> Any:
                def _hook(module: Any, input: Any, output: Any) -> None:
                    h = output if isinstance(output, torch.Tensor) else output[0]
                    slice_ = h[0, -1, :].detach().clone()
                    if self._incremental:
                        # Overwrite — keep the bucket length-1 so device
                        # memory stays O(layers·D). ``[-1]`` reads (tap,
                        # latest_per_layer) still return the latest slice.
                        bucket_ref[:] = (slice_,)
                        # The highest hooked layer fires last in the
                        # forward (forward hooks run in layer-execution
                        # order), so by the time it stores its slice every
                        # hooked layer holds this step's value. Score now.
                        if layer_idx == self._max_layer and self._step_sink is not None:
                            self._step_sink(self.latest_per_layer())
                    else:
                        bucket_ref.append(slice_)
                return _hook

            self._handles.append(
                layers[idx].register_forward_hook(_make(bucket, idx)),
            )

    def set_incremental(
        self, step_sink: Callable[[dict[int, torch.Tensor]], None],
    ) -> None:
        """Enable incremental mode: overwrite buckets + per-step scoring.

        Must be called after :meth:`attach`. Flips the per-layer hook from
        append to length-1 overwrite and records the highest hooked layer
        as the per-forward sink trigger. ``step_sink`` receives
        :meth:`latest_per_layer` once per forward, after every hooked
        layer has stored this step's slice.
        """
        self._incremental = True
        self._step_sink = step_sink
        self._max_layer = max(self._per_layer) if self._per_layer else None

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def clear(self) -> None:
        self._per_layer = {}
        self._handles = []
        self._incremental = False
        self._step_sink = None
        self._max_layer = None

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
        # Manifold groups: (Trigger, basis [R,dim], mean [dim],
        # target [dim], alpha).  ``target`` is the spline point at the
        # term's fixed position ``t`` — precomputed at recompose time, so
        # the hot path only does the subspace-replace matmuls.  Any
        # manifold group forces the slow path.
        self.manifold_groups: list[
            tuple[Trigger, torch.Tensor, torch.Tensor, torch.Tensor, float]
        ] = []
        # Injection-mode state.  Angular cache is populated only on the
        # fast path; the slow path computes per-fire from active groups.
        self.injection_mode: InjectionMode = injection_mode
        self.theta_max: float = theta_max
        # Angular fast-path cache (None unless angular + fast path).
        # ``_cos_t`` / ``_sin_t`` are 0-dim tensors when the cache is
        # warm (allocated lazily on the layer's device in
        # ``_refresh_angular_cache``) so that ``torch.compile`` sees a
        # single tensor input across recomposes — Python-scalar inputs
        # would burn a guard per α and force recompile on every
        # generation.  Allocated once via ``empty(())`` and refreshed
        # in-place via ``copy_`` against a CPU staging tensor.  When
        # the cache is cold (no fast-path steering active) the
        # tensors stay as ``None`` and the hot path early-exits on
        # ``self._d_hat is None``.
        self._d_hat: torch.Tensor | None = None
        self._theta: float = 0.0
        self._cos_t: torch.Tensor | None = None
        self._sin_t: torch.Tensor | None = None
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
        manifold_entries: "list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, Trigger]] | None" = None,
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

        # --- manifold grouping ---
        # ``target`` is the spline point at the term's fixed position,
        # already computed by the manager; the hot path only runs the
        # subspace-replace matmuls.  Zero-alpha entries drop here.
        manifold_groups: list[
            tuple[Trigger, torch.Tensor, torch.Tensor, torch.Tensor, float]
        ] = []
        for basis, mean, target, alpha, trig in (manifold_entries or []):
            if alpha == 0.0:
                continue
            manifold_groups.append((
                trig,
                basis.to(device=device, dtype=dtype),
                mean.to(device=device, dtype=dtype),
                target.to(device=device, dtype=dtype),
                float(alpha),
            ))
        self.manifold_groups = manifold_groups

        # --- fast-path collapse decision ---
        if not composed_groups and not ablation_groups and not manifold_groups:
            self.composed = None
            self.composed_groups = []
            self._d_hat = None
            return

        # Fast path only when the single contributor is additive/BOTH and
        # no ablation or manifold group is attached.  Either forces the
        # slow path so the hook_fn can sequence the ops.
        if (
            not ablation_groups
            and not manifold_groups
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

        ``_cos_t`` / ``_sin_t`` are 0-dim tensors on the layer's device
        (allocated lazily on first non-degenerate refresh, reused via
        in-place ``copy_`` thereafter).  Pinning them as tensor inputs
        keeps the hook compatible with ``torch.compile`` and CUDA-graph
        capture across α changes — Python scalars would force recompile
        per generation.
        """
        composed = self.composed
        if composed is None or theta_strength <= 1e-12:
            self._d_hat = None
            self._theta = 0.0
            return
        c_f32 = composed.detach().to(torch.float32)
        c_norm_t = torch.linalg.vector_norm(c_f32)
        c_norm = float(c_norm_t.item())
        if c_norm <= 1e-12:
            # All α-weighted terms cancelled out; treat as no-op.
            self._d_hat = None
            self._theta = 0.0
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
        cos_v = math.cos(self._theta)
        sin_v = math.sin(self._theta)
        # Lazy alloc on the composed tensor's device; subsequent
        # refreshes mutate in place so the tensor *address* the
        # captured graph or compiled artifact saw stays valid.
        # Locals so pyright narrows the post-alloc fill_ calls — the
        # ``self.<attr>`` writes don't propagate the type narrowing
        # back to the attribute reads.
        cos_buf = self._cos_t
        sin_buf = self._sin_t
        if cos_buf is None or cos_buf.device != composed.device:
            cos_buf = torch.empty(
                (), device=composed.device, dtype=torch.float32,
            )
            sin_buf = torch.empty(
                (), device=composed.device, dtype=torch.float32,
            )
            self._cos_t = cos_buf
            self._sin_t = sin_buf
        assert sin_buf is not None  # paired alloc above
        cos_buf.fill_(cos_v)
        sin_buf.fill_(sin_v)

    def hook_fn(self, module: Any, input: Any, output: Any) -> Any:
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
        mfld_groups = self.manifold_groups
        if not add_groups and not abl_groups and not mfld_groups:
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
            for grp in mfld_groups:
                if grp[0].active(ctx):
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
            # Manifold replace runs last — it is a destructive overwrite
            # of the in-subspace component, so it dominates at the layers
            # it covers; running after additive/ablation makes that
            # ordering explicit.
            self._apply_manifold_groups(hidden, ctx)
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

        if active_composed is not None and active_strength > 1e-12:
            c_f32 = active_composed.to(torch.float32)
            c_norm_t = torch.linalg.vector_norm(c_f32)
            c_norm = float(c_norm_t.item())
            if c_norm > 1e-12:
                d_hat = (c_f32 / c_norm).to(dtype=hidden.dtype)
                ratio = active_strength
                if ratio > 1.0:
                    ratio = 1.0
                theta = ratio * self.theta_max
                _angular_inplace(
                    hidden, d_hat, math.cos(theta), math.sin(theta),
                )

        # Manifold replace runs last (see the additive branch).
        self._apply_manifold_groups(hidden, ctx)
        return output

    def _apply_manifold_groups(
        self, hidden: torch.Tensor, ctx: TriggerContext,
    ) -> None:
        """Apply every active manifold group as a subspace blend or rotation.

        Dispatches on :attr:`injection_mode`:

        - ``"additive"`` -> :func:`subspace_replace`: blend ``hidden``'s
          projection onto the manifold's PCA subspace toward the
          precomputed spline target by ``alpha`` and restore the original
          per-position norm.  Destructive in-subspace overwrite.
        - ``"angular"`` -> :func:`subspace_rotate`: Givens-rotate
          ``hidden``'s in-subspace component toward the target by
          ``α · θ_max`` inside the subspace plane; the orthogonal
          residual is kept verbatim, ``||hidden - mean||`` is preserved
          by construction, no rescale.

        Runs after ablation and additive so the in-subspace operation
        wins at the layers it covers, regardless of mode.
        """
        if not self.manifold_groups:
            return
        if self.injection_mode == "angular":
            theta_max = self.theta_max
            for trig, basis, mean, target, alpha in self.manifold_groups:
                if not trig.active(ctx):
                    continue
                hidden.copy_(
                    subspace_rotate(
                        hidden, mean, basis, target, alpha, theta_max,
                    )
                )
        else:
            for trig, basis, mean, target, alpha in self.manifold_groups:
                if not trig.active(ctx):
                    continue
                hidden.copy_(
                    subspace_replace(hidden, mean, basis, target, alpha)
                )

    def attach(self, layer_module: torch.nn.Module) -> None:
        """Register forward hook on a layer module."""
        self._handle = layer_module.register_forward_hook(self.hook_fn)

    def detach(self) -> None:
        """Remove the forward hook."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


# Gain that pins the user-facing alpha scale, applied **only under
# additive mode** (the default is angular, which maps α directly to a
# rotation angle and takes no gain).  Per-layer shares
# (score_L / Σ scores) are baked into the stored direction magnitudes at
# extraction time, so the additive hook math collapses to a single flat
# scalar:
#
#     effective_injection = user_alpha * _STEER_GAIN * baked_direction_L
#
# (Angular mode never multiplies by ``_STEER_GAIN``: its per-layer budget
# is ``user_alpha · share_L``, water-filled to ≤ θ_max per layer, and the
# share is read straight back off ``||baked_L||`` — see
# ``SteeringManager.apply_to_model``.)
#
# The same two invariances fall out of the baking step (they just moved
# from apply-time to extract-time), and hold for both injection modes
# because both read the per-layer share off the baked magnitude:
#
#   - Layer-count invariance: total injection is independent of n_layers,
#     so models of different depths hit the same behavioral effect at the
#     same user alpha.  Without this, deeper models over-inject (e.g.
#     gemma-4-E4B at 42 layers vs gemma-4-31B at 60 layers would drift
#     by ~1.5× in coherent α).
#
#   - Score-magnitude invariance: absolute extraction scores vary wildly
#     between architectures (diffuse Llama-3.2-3B ≈0.07, sharp gemma-3-4b
#     ≈0.25 for the same pairs), but only the *relative* per-layer shares
#     matter here — high-signal layers still get proportionally more push
#     within a profile.
#
# The gain is calibrated so that a user alpha of ~0.5 lands in the
# coherent additive-steering band on the reference model (gemma-4-31B-it)
# across the 26 bundled concepts.  Raising the gain shifts every model's
# coherent α lower; lowering does the opposite.  Smaller or non-standard-
# geometry models (MatFormer, MoE, heavily safety-trained) may still need
# proportionally higher alpha due to residual architectural effects
# (activation magnitude, attention layout) this normalization doesn't
# capture.
#
# History: recalibrated from 3.5 → 2.0 alongside discriminative layer
# selection (DLS, Dang & Ngo 2026 — the share-bake gate that replaced the
# earlier fixed edge-drop heuristic).  DLS keeps only the layers where the
# pos / neg projections straddle the neutral baseline (polarity, not
# intensity), which both trims early tokenization / late unembedding
# layers *and* tightens the retained directions' cross-layer coherence —
# so per-α directional rotation rises and 2.0 lands the additive cliff
# above α≈0.9 on the reference model, giving a wide coherent band
# (~0.3-0.85) with headroom for the untested-architecture tail.
_STEER_GAIN = 2.0


# Per-mode *base* gains for manifold steering, analogous to
# :data:`_STEER_GAIN` for vector steering.  Manifold injection moves only
# ``h_par`` (the projection of ``h`` into the manifold's R-dim affine
# subspace) — a fraction of ``||h||`` that varies with subspace dimension
# and with whether the subspace is rogue-heavy (Euclidean PCA) or
# de-rogued (whitened/Fisher PCA).  The per-α behavioral magnitude
# therefore scales with that fraction, which would make a 16-dim fit
# steer harder than a 4-dim fit, and a Euclidean fit harder than a
# whitened one, at the same user α.
#
# ``apply_to_model`` removes that dependence by dividing the base gain by
# the manifold's share-weighted *lever* ``N = Σ_L share_L · f_L`` where
# ``f_L = E_neutral[||h_par_c||/||h||]`` is baked at fit time (see
# :func:`saklas.core.manifold.layer_lever`).  So the *effective* gain is
# ``base / N``: a small-lever fit (de-rogued, or small R) gets a
# proportionally larger gain, equalizing the per-α effect across subspace
# dimension and selection metric.  The base is thus a pure, lever-
# normalized scale — calibrated (sweep on the bundled whitened
# ``personas`` fit) so ``α ≈ 0.5`` lands in the coherent persona band on
# the reference model regardless of the fit's R or metric.
#
# Per-mode because the operators are differently lossy: angular
# (``subspace_rotate``) rotates ``h_par`` by an exact angle, no rescale;
# additive (``subspace_replace``) blends then norm-restores, which
# partially undoes the move, so it needs ~2× the budget — stamped as
# ``angular × _STEER_GAIN`` to keep the vector/manifold relationship
# consistent.  The lever ``N`` subsumes the former additive-only
# ``1/√EV`` quality factor: ``N`` measures the actual steering lever
# directly (and for both modes), where EV was only a proxy for it.
#
# Calibrated on the gemma-4-31b whitened-``personas`` α/R sweep: with the
# lever normalization, ``base = 2.0`` puts ``α ≈ 0.5`` at the mild-coherent
# point and ``α ≈ 1.0`` at strong persona expression — so the vector-
# comparable "``α ≈ 0.5``, tune per target" idiom carries over unchanged,
# with headroom to push (α is clamped to [0, 1], so the base is the
# strength ceiling).  The lever normalization makes this hold across
# subspace dimension and selection metric; per-persona strength variance
# remains (a hard persona like ``caveman`` peaks near its coherence edge
# at α≈1, where a robust one like ``hacker`` still has room — tune α down
# per target).
_MANIFOLD_GAIN_ANGULAR = 2.0
_MANIFOLD_GAIN_ADDITIVE = _MANIFOLD_GAIN_ANGULAR * _STEER_GAIN  # = 4.0

# Floor on the share-weighted lever ``N`` so a near-degenerate (tiny-
# lever) fit can't send the effective gain ``base / N`` to infinity.
_MIN_MANIFOLD_LEVER = 0.01

# One-time saturation warnings, keyed by manifold name.
_warned_manifold_saturated: set[str] = set()


def _redistribute_budget(
    per_layer_raw: dict[int, float],
) -> dict[int, float]:
    """Water-fill a per-layer angular budget so no layer exceeds θ_max.

    Both angular steering paths express a per-layer rotation budget in
    ``θ_max`` units (1.0 ⇒ a full ``θ_max`` rotation at that layer).  A
    peaked per-layer share can push a single layer's budget above 1.0;
    the historical code clamped that layer at 1.0 and silently *dropped*
    the excess, which loses cumulative steering budget on a saturated
    layer.  This helper instead conserves the budget: cap each layer at
    1.0 and redistribute the trimmed excess across the still-uncapped
    layers proportional to their current value, iterating until no new
    layer caps (or every layer is capped).

    The cumulative budget is preserved up to the hard ceiling: the
    returned values sum to ``min(Σ raw, n_layers)`` — when the requested
    total fits under ``n_layers · θ_max`` every bit of budget lands on
    some layer; when it overflows (more total than the layers can hold)
    every layer saturates at 1.0.

    Apply-time only (one host-side pass over the layer set per
    ``apply_to_model``), never the per-token hot path; pure Python floats
    so there is no device sync.  Negative raw budgets (an inverted /
    sign-flipped term) are passed through by magnitude: the cap and the
    redistribution operate on ``|raw|`` and the original sign is
    re-attached, so a negative term rotates the other way without
    breaking the water-fill bookkeeping.
    """
    if not per_layer_raw:
        return {}
    signs: dict[int, float] = {
        L: (-1.0 if v < 0.0 else 1.0) for L, v in per_layer_raw.items()
    }
    # Work on magnitudes; re-attach sign at the end.
    capped: dict[int, float] = {L: abs(v) for L, v in per_layer_raw.items()}
    uncapped: set[int] = set(capped)
    # Iterate: pull every over-budget layer down to 1.0, pool the trimmed
    # excess, and spread it over the layers still below 1.0 in proportion
    # to their current (sub-cap) value.  Each pass can only push more
    # layers to the cap, so it terminates in ≤ n_layers iterations.
    while True:
        excess = 0.0
        newly_capped: list[int] = []
        for L in list(uncapped):
            if capped[L] > 1.0:
                excess += capped[L] - 1.0
                capped[L] = 1.0
                newly_capped.append(L)
        for L in newly_capped:
            uncapped.discard(L)
        if excess <= 1e-12 or not uncapped:
            break
        pool = sum(capped[L] for L in uncapped)
        if pool <= 1e-12:
            # All remaining layers are at zero budget — nothing to soak
            # up the excess; it is dropped (cumulative budget already
            # equals n_capped, the hard ceiling for this configuration).
            break
        for L in uncapped:
            capped[L] += excess * (capped[L] / pool)
    return {L: signs[L] * capped[L] for L in per_layer_raw}


def _profile_layer_shares(profile: dict[int, torch.Tensor]) -> dict[int, float]:
    norms: dict[int, float] = {}
    total = 0.0
    for layer_idx, vec in profile.items():
        n = float(torch.linalg.vector_norm(vec.to(torch.float32)).item())
        norms[layer_idx] = n
        total += n
    if total <= 1e-12:
        return {}
    return {layer_idx: n / total for layer_idx, n in norms.items()}


def _manifold_layer_shares(manifold: Any) -> dict[int, float]:
    # Prefer the whitened (Mahalanobis) per-layer share baked at fit time —
    # the subspace-restricted analogue of vector steering's ``‖d‖_M`` bake
    # score (see ``LayerWhitener.subspace_gram`` /
    # ``ManifoldExtractionPipeline.fit``).  Requires *full* layer coverage:
    # the share is a cross-layer-normalized weight, so mixing whitened and
    # Euclidean scalars across layers would compare incommensurable
    # metrics.  When the baked share is absent (no whitener at fit time —
    # CPU test stubs) or partial, fall back to the Euclidean centroid-
    # spread ``‖coords‖_F``.
    baked = getattr(manifold, "mahalanobis_share", None)
    if baked and all(layer_idx in baked for layer_idx in manifold.layers):
        layer_scores: dict[int, float] = {
            layer_idx: float(baked[layer_idx]) for layer_idx in manifold.layers
        }
    else:
        layer_scores = {}
        for layer_idx, sub in manifold.layers.items():
            node_coords = eval_rbf(
                sub.node_params, sub.rbf_weights, sub.poly_coeffs,
                sub.node_params,
            )  # (K, R) — exact centered coords at the fit nodes
            layer_scores[layer_idx] = float(
                torch.linalg.vector_norm(node_coords).item()
            )
    total_score = sum(layer_scores.values())
    if total_score <= 1e-12:
        n_layers = max(1, len(manifold.layers))
        return {L: 1.0 / n_layers for L in manifold.layers}
    return {L: s / total_score for L, s in layer_scores.items()}


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
        self.vectors: dict[str, dict[str, Any]] = {}
        self.ablations: dict[str, dict[str, Any]] = {}
        self.manifolds: dict[str, dict[str, Any]] = {}
        self.ctx: TriggerContext = TriggerContext()
        self.injection_mode: InjectionMode = injection_mode
        self.theta_max: float = theta_max

    def all_fast_path(self) -> bool:
        """True iff every attached hook is in fast-path mode.

        Fast path = single ``Trigger.BOTH`` additive group with no
        ablation, collapsed by ``SteeringHook.recompose`` into a stable
        ``self.composed`` tensor.  In this regime the hook fires
        unconditional in-place ops with no read of the mutable
        :class:`TriggerContext` per step — which is the precondition
        for ``torch.compile(mode="reduce-overhead")`` graph capture and
        for the StaticCache routing in :mod:`saklas.core.cuda_graphs`.

        Slow path = trigger groups, ablation, or probe gates: hooks
        consult ``ctx`` per fire so a captured graph can't represent the
        per-step decision.  Empty manager (no hooks attached) returns
        True — there's nothing to break, and unsteered generations are
        the cheapest case.
        """
        for hook in self.hooks.values():
            if hook.composed is None:
                return False
        return True

    def add_vector(
        self,
        name: str,
        profile: dict[int, torch.Tensor],
        alpha: float,
        trigger: Trigger = Trigger.BOTH,
    ) -> None:
        # Shares are computed lazily in ``apply_to_model`` — only the
        # angular path reads them, and ``_profile_layer_shares`` does a
        # host ``.item()`` sync per layer.  Additive reads ``||baked_L||``
        # back out of the tensor magnitude and never touches shares, so
        # eager computation here would burn the sync on every add for
        # nothing under the default-after-mode-flip additive case.
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

    def add_manifold(
        self,
        name: str,
        manifold: object,
        position: tuple[float, ...] | str,
        alpha: float,
        trigger: Trigger = Trigger.BOTH,
    ) -> None:
        """Register a manifold-steering term.

        At ``apply_to_model`` time, for every layer the manifold covers,
        the manifold point at ``position`` is precomputed and a per-layer
        ``(basis, mean, target, alpha, trigger)`` entry is attached to the
        corresponding :class:`SteeringHook`.

        ``position`` is either a tuple of authoring coordinates (coord
        form) or a node-label string (label form, sugar for that node's
        coords).  Labels are resolved through
        :meth:`Manifold.resolve_position` here so the downstream
        ``manifolds`` dict always carries a plain coord tuple.  An
        unknown label raises
        :class:`saklas.core.manifold.UnknownManifoldLabelError`; arity
        mismatches against the manifold's domain (only meaningful for
        coord-form input) raise
        :class:`saklas.core.steering_expr.SteeringExprError`.

        ``alpha`` is the blend fraction (clamped to ``[0, 1]``); no
        ``_STEER_GAIN`` applies — it is a fraction, not an additive push.
        """
        resolve = getattr(manifold, "resolve_position", None)
        if resolve is not None:
            resolved = resolve(position)
        elif isinstance(position, str):
            # Defensive: a manifold-shaped object without
            # ``resolve_position`` (e.g. a test double) can't resolve
            # labels.  Raise rather than guess.
            raise TypeError(
                f"manifold {name!r} cannot resolve a label-form position "
                f"({position!r}) — the manifold lacks resolve_position()"
            )
        else:
            resolved = tuple(float(c) for c in position)
        domain = getattr(manifold, "domain", None)
        if domain is not None and len(resolved) != domain.intrinsic_dim:
            from saklas.core.errors import ManifoldArityError
            raise ManifoldArityError(
                f"manifold {name!r} has a {domain.intrinsic_dim}-dimensional "
                f"domain but the steering position has {len(resolved)} "
                f"coordinate(s)"
            )
        self.manifolds[name] = {
            "manifold": manifold,
            "position": tuple(float(c) for c in resolved),
            "alpha": alpha,
            "trigger": trigger,
            "shares": _manifold_layer_shares(manifold),
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
                # Compute shares here (not at add_vector) — this is the
                # only path that reads them, and the computation host-syncs
                # per layer.
                shares: dict[int, float] = _profile_layer_shares(profile)
                if not shares:
                    # Profile is degenerate (all-zero); skip silently.
                    continue
                # Raw per-layer budget in θ_max units (1.0 ⇒ full θ_max
                # rotation at that layer).  Water-fill so a peaked-share
                # layer never rotates past θ_max while the trimmed excess
                # is redistributed to unsaturated layers — preserving the
                # cumulative budget Σ_L θ_L = min(|α|, n_layers)·θ_max
                # instead of silently dropping it on the capped layer.
                raw_budget: dict[int, float] = {
                    layer_idx: user_alpha * shares.get(layer_idx, 0.0)
                    for layer_idx in profile
                }
                capped_budget = _redistribute_budget(raw_budget)
                for layer_idx, vec in profile.items():
                    additive_by_layer.setdefault(layer_idx, []).append(
                        (vec, capped_budget.get(layer_idx, 0.0), trigger),
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

        # Manifold entries: precompute the manifold target per layer once.
        # ``position`` is fixed for the whole generation, so the RBF eval
        # never reaches the hot path.  Only one manifold may cover a
        # given layer — two would each destructively overwrite the
        # in-subspace component, so their composition is ill-defined.
        #
        # Per-layer α is share-weighted, exactly analogous to vector
        # steering's ``share_L = ||baked_L|| / Σ ||baked||``.  The
        # manifold's per-layer "signal strength" is the Frobenius norm
        # of the centered node centroids in subspace coords — i.e. how
        # widely the per-node centroids spread inside this layer's
        # affine PCA subspace, which is the manifold analogue of "how
        # discriminative is this layer for the steered concept."  Those
        # shares are cached when the manifold is registered (by evaluating
        # the RBF at its fit nodes once), so alpha changes do not repeat
        # host syncs or small RBF evaluations.  Layers with weak persona
        # separation get a small slice of α; the cumulative budget
        # ``Σ_L α · share_L = α`` regardless of how many layers the
        # manifold covers, so the user-facing α-regime is
        # layer-count-invariant and matches the vector-steering
        # idiom.  Degenerate (all-zero) shares fall back to uniform
        # ``1/N`` so behavior stays defined even on a pathological fit.
        manifold_by_layer: dict[
            int,
            list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, Trigger]],
        ] = {}
        manifold_owner: dict[int, str] = {}
        for mname, m in self.manifolds.items():
            manifold = m["manifold"]
            position = m["position"]
            trigger = m["trigger"]
            alpha = max(0.0, min(1.0, float(m["alpha"])))

            shares: dict[int, float] = m.get("shares", {})

            # Dispatch the per-mode base gain.  Per-call
            # ``Steering.injection_mode`` overrides flow through
            # ``self.injection_mode`` already.
            if self.injection_mode == "angular":
                base_gain = _MANIFOLD_GAIN_ANGULAR
            else:
                base_gain = _MANIFOLD_GAIN_ADDITIVE

            # Lever normalization (replaces the former additive-only
            # ``1/√EV`` quality factor — see the gain constants' docstring).
            # The per-α behavioral magnitude scales with the manifold's
            # *lever* ``f_L = E_neutral[||h_par_c||/||h||]``: a bigger
            # subspace (larger R) or a rogue-heavy Euclidean fit captures
            # more of ``||h||`` ⇒ steers harder at the same α.  Dividing the
            # gain by the share-weighted lever ``N = Σ_L share_L·f_L`` makes
            # the effect invariant to subspace dimension *and* to
            # subspace-selection metric, so a 4-dim and a 16-dim fit — or a
            # Euclidean and a whitened fit — reach comparable behavior at the
            # same user α.  Baked all-or-nothing alongside the whitened
            # share; a fit without it (no neutral activations at fit time)
            # uses ``N = 1`` (the un-normalized base gain) for that
            # degenerate path.
            lever = getattr(manifold, "lever", None)
            if lever and all(L in lever for L in manifold.layers):
                N = sum(shares[L] * lever[L] for L in manifold.layers)
                gain = base_gain / max(N, _MIN_MANIFOLD_LEVER)
            else:
                gain = base_gain

            raw_alpha: dict[int, float] = {
                layer_idx: alpha * shares[layer_idx] * gain
                for layer_idx in manifold.layers
            }
            # Angular mode reads ``effective_alpha`` as a rotation budget
            # in θ_max units (``subspace_rotate`` computes ``θ =
            # effective_alpha · θ_max``).  A peaked-share layer can push
            # that budget past 1.0 — i.e. rotate ``h_par`` more than θ_max,
            # which ``subspace_rotate`` does *not* clamp.  Water-fill the
            # per-layer budget so no layer rotates past θ_max and the
            # trimmed excess is redistributed to unsaturated layers,
            # preserving the cumulative rotation budget.  Additive mode is
            # a blend fraction with its own per-position norm restore, so
            # there is no θ_max ceiling to enforce — leave it as the raw
            # share-weighted value.
            if self.injection_mode == "angular":
                eff_alpha = _redistribute_budget(raw_alpha)
                # Saturation tell: water-fill caps each layer at θ_max, so
                # the cumulative budget maxes at ``n_layers``.  If it pins
                # near that ceiling, the manifold's lever is maxed — the
                # rotation operator can't push the persona harder no matter
                # how much α rises (switch to additive, or raise θ_max).
                # This is the small-lever regime a heavily de-rogued fit can
                # hit; warn once so a flat-at-high-α sweep isn't mistaken
                # for a calibration bug.
                n_lay = len(eff_alpha)
                if (
                    alpha > 0.0
                    and n_lay > 0
                    and sum(eff_alpha.values()) >= 0.98 * n_lay
                    and mname not in _warned_manifold_saturated
                ):
                    _warned_manifold_saturated.add(mname)
                    import warnings
                    warnings.warn(
                        f"manifold {mname!r} saturates the angular rotation "
                        f"budget at α={alpha:.2f} (every layer at θ_max): the "
                        f"rotation operator is maxed (small in-subspace lever "
                        f"and/or high α·gain), so it can't steer harder past "
                        f"this point. Use additive mode or raise theta_max to "
                        f"push further.",
                        stacklevel=2,
                    )
            else:
                eff_alpha = raw_alpha

            for layer_idx, sub in manifold.layers.items():
                if layer_idx in manifold_owner:
                    from saklas.core.errors import OverlappingManifoldError
                    raise OverlappingManifoldError(
                        f"manifolds '{manifold_owner[layer_idx]}' and "
                        f"'{mname}' both cover layer {layer_idx}; manifold "
                        f"steering allows only one manifold per layer"
                    )
                manifold_owner[layer_idx] = mname
                target = manifold.manifold_point(layer_idx, position)
                manifold_by_layer.setdefault(layer_idx, []).append(
                    (sub.basis, sub.mean, target, eff_alpha[layer_idx], trigger),
                )

        active_layers = (
            set(additive_by_layer)
            | set(ablation_by_layer)
            | set(manifold_by_layer)
        )

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
                manifold_entries=manifold_by_layer.get(idx, []),
                injection_mode=self.injection_mode,
                theta_max=self.theta_max,
                device=device, dtype=dtype, ctx=self.ctx,
            )

    def clear_all(self) -> None:
        """Detach all hooks and clear vectors + ablations + manifolds."""
        for hook in self.hooks.values():
            hook.detach()
        self.hooks.clear()
        self.vectors.clear()
        self.ablations.clear()
        self.manifolds.clear()
