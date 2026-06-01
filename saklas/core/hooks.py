"""Steering hooks for activation steering on transformer models."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from saklas.core.manifold import (
    CustomDomain,
    LayerSubspace,
    ManifoldDomain,
    SynthesizedSubspace,
    _ortho_basis,
    eval_rbf,
    inject_three_op,
)
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
    """Per-layer steering state: zero or more subspace / manifold groups.

    The 4.0 unified backend.  Every steering term — vectors, poles,
    ``~``/``|`` projections, ``!`` ablations, affine and curved ``%`` — lowers
    to a per-layer :func:`inject_three_op` group: the dispatch-synthesized
    merged affine subspace, plus zero or more mutually-orthogonal curved
    manifolds.  There is no additive/angular vector fast path any more, so a
    steered layer always runs the (ctx-consulting) slow hook — per-step
    triggers and probe gates work uniformly, at the cost of the StaticCache /
    graph-capture path that the old composed-tensor fast path enabled.
    """

    def __init__(self) -> None:
        # Subspace / manifold groups: (Trigger, subspace, domain,
        # target_coord [n], origin_coord [n], along, onto).  The merged affine
        # subspace (from the dispatch synthesizer) and each curved manifold are
        # both groups here; ``inject_three_op``'s ``is_affine`` branch picks the
        # analytic-vs-foot-following path.  ``target_coord`` / ``origin_coord``
        # are authoring coordinates; ``along`` / ``onto`` are the per-layer
        # effective coefficients (share-weighted + lever-normalized at apply
        # time).  See :meth:`_apply_manifold_groups`.
        self.manifold_groups: list[
            tuple[
                Trigger, LayerSubspace, ManifoldDomain,
                torch.Tensor, torch.Tensor, float, float,
            ]
        ] = []
        # Per-token nearest-point foot state, parallel to ``manifold_groups``
        # (``None`` = cold, seed at the origin).  Affine groups ignore it (the
        # foot is ``q`` exactly); curved groups warm-start the Gauss-Newton
        # follower from it.  ``inject_three_op`` returns the refined foot each
        # fire; we stash the *last position* of it as the next token's warm
        # start.  Reset at recompose and by
        # :meth:`SteeringManager.reset_manifold_feet` at each generation start.
        self._manifold_feet: list[torch.Tensor | None] = []
        # Shared mutable context threaded in by SteeringManager.  Read-only
        # from the hook's perspective; the generation loop mutates fields.
        self._ctx: TriggerContext | None = None
        self._handle = None

    def recompose(
        self,
        manifold_entries: "list[tuple[LayerSubspace, ManifoldDomain, torch.Tensor, torch.Tensor, float, float, Trigger]] | None",
        ctx: TriggerContext,
        *,
        device: torch.device,
    ) -> None:
        """Pre-compose this layer's subspace / manifold groups.

        Each entry is ``(subspace, domain, target_coord, origin_coord, along,
        onto, trigger)`` — the merged affine subspace (one entry, possibly per
        active trigger group) and each curved manifold are both groups here.
        ``ctx`` is the shared per-generation :class:`TriggerContext` the
        generation loop mutates and the hook reads at fire time.

        Subspace tensors are cast to **fp32** (the RBF / Gauss-Newton math is
        fp32 regardless of the model dtype; ``inject_three_op`` casts the
        result back to ``hidden.dtype`` per fire).  An entry with both
        coefficients zero is a no-op and drops here.  A new group set
        cold-starts every foot-follower.
        """
        self._ctx = ctx

        # --- subspace / manifold grouping ---
        # ``target_coord`` / ``origin_coord`` are authoring coordinates; the
        # subspace tensors stay **fp32** (the RBF / Gauss-Newton math is fp32
        # regardless of the model dtype, and quantizing ``node_params`` /
        # ``rbf_weights`` to bf16 would wreck the interpolant precision).
        # ``inject_three_op`` re-casts internally, so fp32 here is the precise
        # carrier, not an extra cost.  An entry with both coefficients
        # zero is a no-op and drops here.
        manifold_groups: list[
            tuple[
                Trigger, LayerSubspace, ManifoldDomain,
                torch.Tensor, torch.Tensor, float, float,
            ]
        ] = []
        for sub, domain, target, origin, along, onto, trig in (
            manifold_entries or []
        ):
            if along == 0.0 and onto == 0.0:
                continue
            manifold_groups.append((
                trig,
                sub.to(device=device, dtype=torch.float32),
                domain,
                target.to(device=device, dtype=torch.float32),
                origin.to(device=device, dtype=torch.float32),
                float(along),
                float(onto),
            ))
        self.manifold_groups = manifold_groups
        # New group set ⇒ cold-start every foot-follower (seed at origin).
        self._manifold_feet = [None] * len(manifold_groups)

    def hook_fn(self, module: Any, input: Any, output: Any) -> Any:
        groups = self.manifold_groups
        if not groups:
            return output
        ctx = self._ctx
        if ctx is None:
            return output
        # Cheap pre-check: any group active this step?  Skip the work entirely
        # if not (e.g. an ``AFTER_THINKING`` group during prefill).
        if not any(grp[0].active(ctx) for grp in groups):
            return output
        hidden = output if isinstance(output, torch.Tensor) else output[0]
        self._apply_manifold_groups(hidden, ctx)
        return output

    def _apply_manifold_groups(
        self, hidden: torch.Tensor, ctx: TriggerContext,
    ) -> None:
        """Apply every active manifold group via the unified kernel.

        Each group runs :func:`inject_three_op` — the single along/onto
        injection that replaced the angular/additive mode split.  The two
        per-layer coefficients are already share-weighted + lever-normalized at
        :meth:`SteeringManager.apply_to_model` time, so the hot path just routes
        them through the kernel and threads the per-token foot.

        **Foot-following.**  The nearest-point foot on ``M`` is a function of
        the running activation, so we track it across tokens instead of
        re-solving from scratch.  ``self._manifold_feet[i]`` holds the previous
        token's refined foot (``None`` = cold).  Cold ⇒ seed at the origin ``O``
        and take :data:`_MANIFOLD_COLD_GN_STEPS` Gauss-Newton steps (the prefill
        fire converges the foot across the whole prompt window); warm ⇒ one
        step from the carried foot.  After each fire we stash the *last
        position's* foot (``foot[..., -1:, :]``) — broadcasting the single
        decode position forward, and carrying the last prompt position from
        prefill into the first decode step.

        The only steering path in 4.0 — the merged affine subspace and every
        curved manifold are both groups here, dispatched by the kernel's
        ``is_affine`` branch (analytic slide vs foot-following GN).
        """
        if not self.manifold_groups:
            return
        for i, (
            trig, sub, domain, target, origin, along, onto,
        ) in enumerate(self.manifold_groups):
            if not trig.active(ctx):
                continue
            lead = hidden.shape[:-1]
            seed = self._manifold_feet[i]
            # Warm only when the carried foot's leading shape broadcasts onto
            # this fire (B unchanged, one decode position).  A prefill→decode
            # transition stashes ``(B, 1, n)``, which matches the ``(B, 1, …)``
            # decode hidden; any other mismatch (re-prefill, batch change) falls
            # back to a cold seed rather than a shape error.
            if seed is not None and seed.shape[:-1] == lead:
                foot_seed = seed
                gn_steps = 1
            else:
                n = int(origin.shape[-1])
                foot_seed = origin.reshape(
                    (1,) * len(lead) + (n,)
                ).expand(*lead, n)
                gn_steps = _MANIFOLD_COLD_GN_STEPS
            h_new, foot = inject_three_op(
                hidden, sub, domain, target, foot_seed,
                along, onto, gn_steps=gn_steps,
            )
            hidden.copy_(h_new)
            # Carry the last position's foot forward (decode keeps its single
            # position; prefill hands its final prompt position to decode).
            self._manifold_feet[i] = foot[..., -1:, :].detach()

    def attach(self, layer_module: torch.nn.Module) -> None:
        """Register forward hook on a layer module."""
        self._handle = layer_module.register_forward_hook(self.hook_fn)

    def detach(self) -> None:
        """Remove the forward hook."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


# *Base* gain for the unified subspace/manifold steering backend — one gain
# for every term (vectors, poles, projections, ablations, affine and curved
# ``%``), since 4.0 lowers them all to the one along/onto kernel.  Each op moves
# a piece of ``h`` whose size is a fraction of ``||h||`` that varies with
# subspace dimension ``R`` and with whether the subspace is rogue-heavy
# (Euclidean PCA) or de-rogued (whitened/Fisher PCA); the per-α magnitude
# would otherwise scale with that fraction (a 16-dim fit steering harder
# than a 4-dim fit, a Euclidean fit harder than a whitened one).
#
# ``apply_to_model`` removes that dependence by dividing the base gain by
# the manifold's share-weighted *lever* ``N = Σ_L share_L · f_L`` where
# ``f_L = E_neutral[||h_par_c||/||h||]`` is baked at fit time (see
# :func:`saklas.core.manifold.layer_lever`).  So the *effective* gain is
# ``base / N``: a small-lever fit (de-rogued, or small R) gets a
# proportionally larger gain, equalizing the per-α effect across subspace
# dimension and selection metric.  Because ``f_L`` ≈ the in-subspace norm
# fraction (≈ ``1/n_layers`` scale) and the shares sum to 1, ``base / N`` ≈
# ``n_layers``, so the per-layer ``coeff · share_L · (base/N)`` lands ≈
# ``coeff`` on average — i.e. ``along/onto`` read as ≈ [0, 1]
# fractions per layer, concentrated at the high-share layers.  ``N``
# subsumes the former additive-only ``1/√EV`` quality factor (EV was only a
# proxy for the lever; now recorded as a fit-quality diagnostic only).
#
# Calibrated on the gemma-3-4b whitened-``circumplex`` MPS sweep (the three-op
# Phase-1 validation): the coherent ``along`` band runs to effective ~0.4 with
# collapse beyond, and the lever normalization makes effective ≈ 0.4·user_α, so
# ``base = 1.0`` lands ``α ≈ 0.5`` at the mild-coherent point and ``α ≈ 1.0`` at
# the strong/coherence-edge (the vector-comparable "``α ≈ 0.5``, tune per
# target" idiom; α is clamped to [0, 1] so the base is the strength ceiling).
# Per-persona strength variance persists (``happy`` collapses near α≈1 where
# ``elated`` still holds — tune down per target).  Recalibrated 2.0 → 1.0 from
# the old rotation-kernel value: the geodesic-lerp ``along`` over-steers at the
# old gain.  NOTE (phase-1 sweep item): the two ops share this base; the
# collapse op (``onto``) may want its own (smaller) base if the share-weighted
# concentration proves too aggressive — splitting it is a one-constant change
# here.  The gemma-4-31b sweep (#8) refines the number.
_MANIFOLD_GAIN = 1.0

# Floor on the share-weighted lever ``N`` so a near-degenerate (tiny-
# lever) fit can't send the effective gain ``base / N`` to infinity.
_MIN_MANIFOLD_LEVER = 0.01

# Max |cosine| between two *curved* manifold subspaces sharing a layer before
# they are deemed overlapping (``OverlappingManifoldError``).  Curved manifolds
# that share a layer must be (near-)orthogonal — each overwrites its own
# in-subspace component, so overlapping spans would clobber each other.  The
# merged affine subspace is instead always orthogonalized against the curved
# spans (``_orthogonalize_affine_against``), so affine-vs-curved never raises.
_CURVED_ORTHO_TOL = 1e-3


def _orthogonalize_affine_against(
    sub: LayerSubspace,
    target: torch.Tensor,
    curved_basis: torch.Tensor,
) -> "tuple[LayerSubspace, torch.Tensor] | None":
    """Project a merged affine subspace out of the curved manifolds' span.

    Strips the curved-span component from the affine basis rows *and* the push
    displacement so the merged affine subspace and the orthogonal curved
    manifolds at a layer operate on disjoint directions — the curved manifold
    wins the shared directions (ARCHITECTURE §6 precedence), the affine slide
    handles the complement.  ``sub`` is the synthesized affine ``LayerSubspace``
    (orthonormal ``basis``), ``target`` its ``(R,)`` push coord, ``curved_basis``
    the stacked ``(Rc, D)`` orthonormal rows of every curved manifold at this
    layer.  Returns the re-orthonormalized ``(subspace', target')`` or ``None``
    when the affine span lies entirely inside the curved span (nothing left to
    steer there).
    """
    basis = sub.basis.to(torch.float32)              # (R, D)
    cb = curved_basis.to(torch.float32)              # (Rc, D)
    delta = basis.T @ target.to(torch.float32)       # (D,) world push displacement
    residual = basis - (basis @ cb.T) @ cb           # (R, D) rows ⟂ curved span
    new_basis, _kept = _ortho_basis(list(residual))
    if new_basis.shape[0] == 0:
        return None
    delta_perp = delta - (delta @ cb.T) @ cb         # drop the curved-span part
    new_target = new_basis @ delta_perp              # (R',)
    return LayerSubspace.affine(sub.mean, new_basis), new_target

# Gauss-Newton steps taken on a *cold* foot (seed at origin ``O``).  The
# warm path takes one step per token; the cold fire — the prefill window,
# or the first decode step under a non-prompt trigger — converges the foot
# better with a handful so the early tokens steer from an accurate foot.
# Cheap: O(R) per position, off the model-forward critical path.
_MANIFOLD_COLD_GN_STEPS = 4

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
            _np, _rw, _pc = sub.rbf_params()
            node_coords = eval_rbf(
                _np, _rw, _pc, _np,
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

    def __init__(self) -> None:
        self.hooks: dict[int, SteeringHook] = {}
        self.manifolds: dict[str, dict[str, Any]] = {}
        # Dispatch-synthesized merged affine subspaces (one per active trigger
        # group), the 4.0 unified vector/pole/projection/ablation/affine-``%``
        # backend.  Each value is ``{synth, lever, trigger}``; ``apply_to_model``
        # lowers them to per-layer ``inject_three_op`` entries alongside curved
        # manifolds.
        self.subspaces: dict[str, dict[str, Any]] = {}
        self.ctx: TriggerContext = TriggerContext()

    def all_fast_path(self) -> bool:
        """True iff no steering hook is attached (the unsteered path).

        The 4.0 backend lowers every steering term to a slow (ctx-consulting)
        :func:`inject_three_op` group, so there is no longer a composed-tensor
        fast path.  An attached hook therefore always forces the slow path,
        which is the StaticCache / ``torch.compile`` graph-capture
        ineligibility signal :mod:`saklas.core.cuda_graphs` reads — graph
        capture stays available only for unsteered generation (no hooks),
        the cheapest case.
        """
        return not self.hooks

    def add_manifold(
        self,
        name: str,
        manifold: object,
        position: tuple[float, ...] | str,
        along: float,
        onto: float,
        trigger: Trigger = Trigger.BOTH,
    ) -> None:
        """Register a manifold-steering term.

        At ``apply_to_model`` time, for every layer the manifold covers, the
        per-layer subspace + domain + authoring ``target`` / ``origin`` coords
        are attached to the corresponding :class:`SteeringHook` along with the
        share-weighted + lever-normalized per-layer ``along`` / ``onto``
        coefficients; the hot path runs :func:`inject_three_op`.

        ``position`` is either a tuple of authoring coordinates (coord form)
        or a node-label string (label form, sugar for that node's coords).
        Labels are resolved through :meth:`Manifold.resolve_position` here so
        the downstream ``manifolds`` dict always carries a plain coord tuple.
        An unknown label raises
        :class:`saklas.core.manifold.UnknownManifoldLabelError`; arity
        mismatches against the manifold's domain (only meaningful for
        coord-form input) raise
        :class:`saklas.core.steering_expr.SteeringExprError`.

        ``along`` / ``onto`` are the user coefficients (each clamped to
        ``[0, 1]`` at apply time): ``along`` slides the foot toward
        ``position`` geodesically, ``onto`` collapses the off-manifold
        in-subspace residual.  The off-subspace residual is always kept
        verbatim (the old ``toward`` op is removed).
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
            "along": float(along),
            "onto": float(onto),
            "trigger": trigger,
            "shares": _manifold_layer_shares(manifold),
        }

    def add_subspace(
        self,
        name: str,
        synth: SynthesizedSubspace,
        *,
        lever: dict[int, float] | None = None,
        trigger: Trigger = Trigger.BOTH,
    ) -> None:
        """Register a dispatch-synthesized merged affine subspace (4.0).

        ``synth`` (one per active trigger group) carries the per-layer affine
        :class:`LayerSubspace`, the ``along`` ``target_coord`` (every active
        push term's coeff-scaled pole already composed in), and the
        un-normalized per-layer budget ``share`` (``‖Δ_L‖``).  ``lever`` is the
        per-layer steering lever ``f_L = E_neutral[‖h_par_c‖/‖h‖]`` the
        *session* computes at dispatch over the cached neutral activations
        (:func:`saklas.core.manifold.layer_lever`); ``None`` ⇒ ``N = 1`` (no
        lever normalization — the degenerate / CPU-stub path).

        At :meth:`apply_to_model` each layer becomes a per-layer
        ``(subspace, CustomDomain(R_L), target_coord, origin=0, eff_along,
        onto=0)`` entry routed through the same :func:`inject_three_op` hot
        path as a curved manifold — the affine analytic shortcut slides the
        in-subspace component toward ``target_coord`` with
        ``eff_along_L = clamp(share_L · base_gain / N, 0, 1)``.  No ``onto``
        (the surface fills its span) and no θ_max water-fill (the slide to a
        neutral-anchored real-coord target is intrinsically bounded — the foot
        lands at most on the composed pole at ``eff_along = 1``); the budget is
        a plain per-layer clamp.
        """
        self.subspaces[name] = {
            "synth": synth,
            "lever": lever,
            "trigger": trigger,
        }

    def apply_to_model(
        self,
        model_layers: torch.nn.ModuleList,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Group entries by layer, recompose hooks, attach to model.

        Lowers every registered term — the dispatch-synthesized merged affine
        subspace(s) and each curved manifold — to per-layer
        :func:`inject_three_op` groups, orthogonalizing the affine subspace
        against the curved manifolds so they compose with zero cross-talk,
        then recomposing the per-layer hooks.  ``dtype`` is the model dtype the
        hook casts the fp32 subspace result back to.
        """

        # Manifold entries: stamp the per-layer subspace + domain + authoring
        # ``target`` / ``origin`` coords and the two share-weighted,
        # lever-normalized op coefficients.  The kernel slides the foot in
        # *coordinate* space, so there is no fixed world-target precompute —
        # only the (layer-independent) authoring coords.  Two *curved* manifolds
        # may share a layer only if their subspaces are (near-)orthogonal
        # (``_CURVED_ORTHO_TOL``); overlapping ones raise
        # ``OverlappingManifoldError`` (each would clobber the other's
        # in-subspace component).  ``curved_basis_by_layer`` accumulates the
        # curved spans so the merged affine subspace can be orthogonalized
        # against them below.
        #
        # **Gain (share-weight every op).**  Each of ``along`` / ``onto``
        # gets the same per-layer factor ``share_L · (base/N)``:
        # ``share_L`` (whitened Mahalanobis share, else Euclidean
        # centroid-spread) is how discriminative the manifold is at that layer;
        # ``N = Σ_L share_L·f_L`` is the share-weighted lever that makes the
        # per-α effect invariant to subspace dimension and selection metric
        # (see the gain constants' docstring).  Because ``base/N ≈ n_layers``,
        # the per-layer ``coeff · share_L · (base/N)`` lands ≈ ``coeff`` on
        # average — so the two coefficients read as ≈ [0, 1] per-layer
        # fractions, concentrated at the high-share layers.
        #
        # ``along`` is a displacement budget → **water-filled** (cap 1.0/layer,
        # excess redistributed) so a peaked-share layer never slides past the
        # target and the cumulative budget is conserved.  ``onto`` is a bounded
        # collapse fraction → **clamped** to ``[0, 1]`` per layer (saturation at
        # a layer just means "fully collapsed there"; there is nothing to
        # redistribute).
        manifold_by_layer: dict[
            int,
            list[tuple[
                LayerSubspace, ManifoldDomain,
                torch.Tensor, torch.Tensor, float, float, Trigger,
            ]],
        ] = {}
        curved_owner: dict[int, str] = {}
        curved_basis_by_layer: dict[int, torch.Tensor] = {}
        for mname, m in self.manifolds.items():
            manifold = m["manifold"]
            position = m["position"]
            trigger = m["trigger"]
            along = max(0.0, min(1.0, float(m["along"])))
            onto = max(0.0, min(1.0, float(m["onto"])))

            shares: dict[int, float] = m.get("shares", {})

            # Lever-normalized base gain (one base now — the along/onto split
            # replaced the per-mode angular/additive pair).  ``N`` is baked
            # all-or-nothing alongside the whitened share; a fit without it
            # uses ``N = 1`` (the un-normalized base) for that degenerate path.
            lever = getattr(manifold, "lever", None)
            if lever and all(L in lever for L in manifold.layers):
                N = sum(shares[L] * lever[L] for L in manifold.layers)
                gain = _MANIFOLD_GAIN / max(N, _MIN_MANIFOLD_LEVER)
            else:
                gain = _MANIFOLD_GAIN

            # ``along`` water-fills (rotation/displacement budget, cap 1.0);
            # ``onto`` clamps per layer (collapse fraction).
            raw_along: dict[int, float] = {
                L: along * shares[L] * gain for L in manifold.layers
            }
            eff_along = _redistribute_budget(raw_along)
            eff_onto = {
                L: max(0.0, min(1.0, onto * shares[L] * gain))
                for L in manifold.layers
            }

            # Saturation tell for ``along``: water-fill caps each layer at 1.0,
            # so the cumulative budget maxes at ``n_layers``.  Pinned near that
            # ceiling ⇒ every layer slides fully onto the target — the slide
            # operator is maxed (small in-subspace lever and/or high α·gain),
            # so ``along`` can't push harder.  Warn once so a flat-at-high-α
            # sweep isn't mistaken for a calibration bug.
            n_lay = len(eff_along)
            if (
                along > 0.0
                and n_lay > 0
                and sum(eff_along.values()) >= 0.98 * n_lay
                and mname not in _warned_manifold_saturated
            ):
                _warned_manifold_saturated.add(mname)
                import warnings
                warnings.warn(
                    f"manifold {mname!r} saturates the 'along' displacement "
                    f"budget at along={along:.2f} (every layer fully on the "
                    f"target): the slide operator is maxed (small in-subspace "
                    f"lever and/or high coeff·gain), so it can't push the "
                    f"position harder past this point.",
                    stacklevel=2,
                )

            # Target is layer-independent (one authoring position); clamp it
            # into the domain once.  The cold-start origin seed ``O_L`` is
            # per-layer (each layer's neutral foot) — picked inside the loop.
            domain = manifold.domain
            n_dim = domain.intrinsic_dim
            target_coord = domain.clamp_position(
                torch.tensor([float(c) for c in position], dtype=torch.float32)
            )
            mfld_origins = getattr(manifold, "origin", None) or {}

            for layer_idx, sub in manifold.layers.items():
                B_new = sub.basis.to(torch.float32)
                prev = curved_basis_by_layer.get(layer_idx)
                if prev is not None:
                    # Two curved manifolds share this layer — compose only if
                    # their subspaces are (near-)orthogonal.
                    cross = float((B_new @ prev.T).abs().max().item())
                    if cross > _CURVED_ORTHO_TOL:
                        from saklas.core.errors import OverlappingManifoldError
                        raise OverlappingManifoldError(
                            f"manifolds '{curved_owner[layer_idx]}' and "
                            f"'{mname}' both cover layer {layer_idx} with "
                            f"non-orthogonal subspaces (max |cosine| = "
                            f"{cross:.3f} > {_CURVED_ORTHO_TOL}); curved "
                            f"manifolds sharing a layer must be orthogonal"
                        )
                    curved_basis_by_layer[layer_idx] = torch.cat([prev, B_new])
                else:
                    curved_basis_by_layer[layer_idx] = B_new
                    curved_owner[layer_idx] = mname
                # ``O_L`` (this layer's neutral foot) or ``zeros(n)`` when the
                # layer baked no origin (CPU stub / pre-origin fit).
                O_L = mfld_origins.get(layer_idx)
                if O_L is None:
                    origin_coord = torch.zeros(n_dim, dtype=torch.float32)
                else:
                    origin_coord = domain.clamp_position(
                        O_L.reshape(-1).to(torch.float32)
                    )
                manifold_by_layer.setdefault(layer_idx, []).append((
                    sub, domain, target_coord, origin_coord,
                    eff_along[layer_idx], eff_onto[layer_idx], trigger,
                ))

        # Dispatch-synthesized merged affine subspaces (4.0 unified backend).
        # Each ``synth`` is already neutral-anchored with its ``along`` target
        # composed from every active push term's coeff-scaled pole; here we only
        # set the per-layer slide budget ``eff_along_L = clamp(share_L·base/N)``
        # and lower each layer to a ``CustomDomain(R_L)`` ``inject_three_op``
        # entry (the affine analytic shortcut — no GN / RBF / foot solve).  The
        # target carries the strength, so there is no separate user-α multiply
        # here and no θ_max water-fill (the slide is intrinsically bounded — it
        # lands at most on the composed pole); ``onto = 0`` (the surface fills
        # its span).  Curved-vs-affine orthogonalization + the relaxed overlap
        # check land with the session dispatch flip (Step 5b); here a subspace
        # simply joins ``manifold_by_layer``.
        for s in self.subspaces.values():
            synth: SynthesizedSubspace = s["synth"]
            sub_trigger: Trigger = s["trigger"]
            lever: dict[int, float] | None = s["lever"]
            layer_set = list(synth.layers)
            if not layer_set:
                continue

            # Normalize the per-layer budget share across the subspace's layers
            # (Σ_L share_L = 1) so the slide is layer-count-invariant.
            raw_share = {L: float(synth.share.get(L, 0.0)) for L in layer_set}
            total_share = sum(raw_share.values())
            if total_share <= 1e-12:
                shares = {L: 1.0 / len(layer_set) for L in layer_set}
            else:
                shares = {L: raw_share[L] / total_share for L in layer_set}

            # Lever-normalized base gain (``N = Σ_L share_L·f_L``; ``N = 1``
            # without a lever — the degenerate / CPU-stub path).  Mirrors the
            # curved-manifold gain so the per-α effect is R-/metric-invariant
            # across a rank-1 vector and a rank-8 ``personas%`` term.
            if lever and all(L in lever for L in layer_set):
                N = sum(shares[L] * lever[L] for L in layer_set)
                gain = _MANIFOLD_GAIN / max(N, _MIN_MANIFOLD_LEVER)
            else:
                gain = _MANIFOLD_GAIN

            for L in layer_set:
                sub_L = synth.layers[L]
                sub_target = synth.target_coord[L].to(torch.float32)
                # Orthogonalize the affine subspace against any curved manifold
                # sharing this layer (curved wins the shared directions); drop
                # the layer if the affine span lies entirely inside the curved
                # span (nothing left for the merged subspace to steer there).
                curved = curved_basis_by_layer.get(L)
                if curved is not None:
                    res = _orthogonalize_affine_against(sub_L, sub_target, curved)
                    if res is None:
                        continue
                    sub_L, sub_target = res
                r_l = sub_L.rank
                sub_domain = CustomDomain(r_l)
                # Affine origin is span-coord 0 (neutral → coord 0, §5); the
                # foot seed / cold-start is unused on the affine shortcut.
                sub_origin = torch.zeros(r_l, dtype=torch.float32)
                eff_along_L = max(0.0, min(1.0, shares[L] * gain))
                manifold_by_layer.setdefault(L, []).append((
                    sub_L, sub_domain, sub_target, sub_origin,
                    eff_along_L, 0.0, sub_trigger,
                ))

        active_layers = set(manifold_by_layer)

        # Detach hooks for layers that no longer have any contribution.
        for idx in list(self.hooks):
            if idx not in active_layers:
                self.hooks[idx].detach()
                del self.hooks[idx]

        for idx in active_layers:
            if idx not in self.hooks:
                hook = SteeringHook()
                hook.attach(model_layers[idx])
                self.hooks[idx] = hook
            self.hooks[idx].recompose(
                manifold_by_layer.get(idx, []),
                self.ctx,
                device=device,
            )

    def reset_manifold_feet(self) -> None:
        """Cold-start every hook's per-token foot at the next forward.

        The foot-follower carries the nearest-point foot across decode steps
        as a warm start; that state is per-*generation*.  The session calls
        this at each generation start (alongside ``ctx.reset()``) so a new run
        re-seeds at the origin ``O`` instead of inheriting the previous run's
        final foot.  Hooks with no manifold group are unaffected (empty list).
        """
        for hook in self.hooks.values():
            hook._manifold_feet = [None] * len(hook.manifold_groups)

    def clear_all(self) -> None:
        """Detach all hooks and clear manifolds + subspaces."""
        for hook in self.hooks.values():
            hook.detach()
        self.hooks.clear()
        self.manifolds.clear()
        self.subspaces.clear()
