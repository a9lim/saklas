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
    subspace_inject,
)
from saklas.core.triggers import Trigger, TriggerContext


def _trigger_active(trigger: Trigger, ctx: TriggerContext) -> bool:
    """Hot-path trigger predicate with the default preset as a true no-op."""
    return trigger is Trigger.BOTH or trigger.active(ctx)


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

    Incremental mode (``set_incremental``): for the gated / live-stream
    monitored case the full ``[T, D]`` stack is never needed — the session
    scores each token as it is produced and keeps only the per-token score
    rows. In this mode each per-layer hook OVERWRITES its bucket (length-1)
    instead of appending, so device memory stays O(layers·D) for the whole
    generation, and a ``step_sink`` callback fires once per forward (after
    the highest hooked layer stores its slice) with the latest per-layer
    slice. ``latest_per_layer()`` and the ``bucket[-1]`` reads the
    streaming tap relies on keep working — a length-1 bucket's ``[-1]`` is
    still the latest.

    Aggregate-only mode (``set_aggregate_tail``): when the caller needs only
    the end-of-gen aggregate (no probe gate, no per-token stream — e.g.
    stateless server scoring), the hook keeps a bounded *ring* of the last
    ``depth`` slices per layer and runs NO step sink, so the decode loop pays
    zero per-token monitor scoring (T scorings → 1 at finalize). The session
    pools the last content token via :meth:`tail_slice_at`; the ring is deep
    enough to walk back past trailing special tokens (EOS / end-of-turn).

    Non-incremental mode is byte-identical to the append path: ``stacked()``
    returns the full ``[T, D]`` (used for ``return_hidden``).
    """

    def __init__(self) -> None:
        self._per_layer: dict[int, list[torch.Tensor]] = {}
        self._handles: list[Any] = []
        # Incremental-mode state. ``_incremental`` flips the per-layer
        # hook from append to overwrite; ``_step_sink`` is invoked once
        # per forward after the highest hooked layer (``_max_layer``)
        # stores this step's slice. All reset on attach/clear.
        self._incremental: bool = False
        self._step_sink: Callable[[dict[int, torch.Tensor]], None] | None = None
        self._max_layer: int | None = None
        # Bounded-tail (aggregate-only) state: when ``_tail_depth > 1`` the
        # incremental hook keeps the last ``_tail_depth`` slices per layer (a
        # ring) instead of length-1, and runs *no* step sink — used when the
        # caller needs only the end-of-gen aggregate, not the per-token stream,
        # so the expensive per-token scoring is skipped entirely and the
        # aggregate is scored once at finalize from the retained tail (deep
        # enough to walk back past trailing special tokens to the last content
        # token). ``_forward_count`` counts decode forwards (incremented once
        # per forward at the max layer) so ``tail_slice_at`` can map a
        # generated-token index to its ring slot.
        self._tail_depth: int = 1
        self._forward_count: int = 0
        # Persistent compile-clean capture source: when capture rides the
        # always-on pre-compile capture hooks (``install_persistent_capture_hooks``)
        # instead of transient per-gen hooks, this maps each captured layer to its
        # persistent ``(D,)`` buffer (the latest-slice the hook ``copy_``-ed this
        # forward).  ``attach_persistent`` populates it (no transient hook is
        # registered); :meth:`ingest_persistent` reads it post-forward to drive the
        # same accumulation + step-sink the in-hook path does.  Empty ⇒ the normal
        # transient-hook path.
        self._persistent_buffers: dict[int, torch.Tensor] = {}

    def attach(
        self, layers: "torch.nn.ModuleList", layer_indices: list[int]
    ) -> None:
        self._per_layer = {idx: [] for idx in layer_indices}
        self._handles = []
        # Attach resets incremental state — a fresh capture starts in the
        # append (full-retention) mode. ``set_incremental`` /
        # ``set_aggregate_tail`` must be called after attach to opt into
        # incremental scoring / bounded-tail capture for this gen.
        self._incremental = False
        self._step_sink = None
        self._max_layer = None
        self._tail_depth = 1
        self._forward_count = 0
        self._persistent_buffers = {}
        for idx in layer_indices:
            bucket = self._per_layer[idx]

            def _make(bucket_ref: list[torch.Tensor], layer_idx: int) -> Any:
                def _hook(module: Any, input: Any, output: Any) -> None:
                    h = output if isinstance(output, torch.Tensor) else output[0]
                    src = h[0, -1, :].detach()
                    if self._incremental:
                        if self._tail_depth <= 1:
                            # Overwrite into a single preallocated (D,) buffer per
                            # layer — ``copy_`` the latest slice in instead of
                            # allocating a fresh clone every step, so the per-token
                            # decode loop does zero capture allocation (only the
                            # first fire allocates).  Device memory stays
                            # O(layers·D) and the bucket stays length-1, so
                            # ``[-1]`` reads (tap, latest_per_layer) still return
                            # the latest slice.  Safe because every consumer (step
                            # sink, gate callback, token tap) reads the slice
                            # synchronously after this forward and before the next
                            # overwrites it.
                            if bucket_ref:
                                bucket_ref[0].copy_(src)
                            else:
                                bucket_ref.append(src.clone())
                        else:
                            # Bounded-tail (aggregate-only) ring: keep the last
                            # ``_tail_depth`` slices so finalize can pool the last
                            # *content* token (walking back past trailing
                            # specials).  O(tail_depth·layers·D), no per-token
                            # scoring.
                            bucket_ref.append(src.clone())
                            if len(bucket_ref) > self._tail_depth:
                                bucket_ref.pop(0)
                        # The highest hooked layer fires last in the forward
                        # (forward hooks run in layer-execution order), so by the
                        # time it stores its slice every hooked layer holds this
                        # step's value.  Count the forward (drives the tail ring's
                        # ``tail_slice_at`` mapping); the per-token monitor scoring
                        # that used to fire *here* now runs post-forward via
                        # :meth:`fire_step_sink` (FIX F1) so the host-side score
                        # read no longer drains the device pipeline mid-forward at
                        # the max probe layer — the remaining transformer layers +
                        # LM head stay queued.
                        if layer_idx == self._max_layer:
                            self._forward_count += 1
                    else:
                        # Full-retention mode: each step is a distinct clone so
                        # ``stacked()`` can build the [T, D] history.
                        bucket_ref.append(src.clone())
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
        as the per-forward counter trigger. ``step_sink`` receives
        :meth:`latest_per_layer` once per forward via :meth:`fire_step_sink`,
        which ``generate_steered`` invokes **post-forward** (FIX F1) — not from
        inside the capture hook — so the score read doesn't drain the device
        pipeline mid-forward.
        """
        self._incremental = True
        self._tail_depth = 1
        self._step_sink = step_sink
        self._max_layer = max(self._per_layer) if self._per_layer else None

    def set_aggregate_tail(self, depth: int) -> None:
        """Enable bounded-tail capture (aggregate-only, no per-token scoring).

        Must be called after :meth:`attach`. Keeps the last ``depth`` slices
        per layer (a ring) and installs *no* step sink, so the decode loop pays
        no per-token monitor scoring; the session pools the last content token
        at finalize via :meth:`tail_slice_at`. ``depth`` must exceed the most
        trailing special tokens a generation can append after its last content
        token (EOS, end-of-turn) so the walk-back lands inside the ring.
        """
        self._incremental = True
        self._step_sink = None
        self._tail_depth = max(1, int(depth))
        self._max_layer = max(self._per_layer) if self._per_layer else None

    def attach_persistent(
        self, layer_indices: list[int], buffers: dict[int, torch.Tensor],
    ) -> None:
        """Compile-clean capture: accumulate from persistent buffers, no hooks.

        The always-on pre-compile capture hooks
        (:func:`install_persistent_capture_hooks`) ``copy_`` each layer's latest
        ``(D,)`` slice into ``buffers[L]`` every forward — fused into the compiled
        graph, so they don't break it (a transient per-gen ``register_forward_hook``
        would).  This sets up the per-layer buckets for the probe subset and
        records the buffer source, but registers **no** transient hook.  The
        generation loop calls :meth:`ingest_persistent` post-forward to run the
        same accumulation (length-1 / tail ring / full stack) + step sink the
        in-hook path runs.  Mode setters (:meth:`set_incremental` /
        :meth:`set_aggregate_tail`) apply identically afterward.
        """
        self._per_layer = {idx: [] for idx in layer_indices}
        self._handles = []
        self._incremental = False
        self._step_sink = None
        self._max_layer = None
        self._tail_depth = 1
        self._forward_count = 0
        self._persistent_buffers = {
            idx: buffers[idx] for idx in layer_indices if idx in buffers
        }

    def ingest_persistent(self) -> None:
        """Post-forward accumulation from the persistent buffers (compiled path).

        The compile-clean mirror of the in-hook accumulation: for each captured
        layer, append / overwrite / ring its latest persistent slice exactly as
        :meth:`attach`'s hook would, advance the forward counter, then fire the
        step sink once (the FIX F1 post-forward scoring point).  Wired as the
        decode loop's ``step_callback`` when capture is persistent, so the call
        order (forward → ingest → gate scoring) and the resulting bucket shapes
        are byte-identical to the transient path — every downstream consumer
        (``tail_slice_at`` / ``stacked`` / ``latest_per_layer`` / the step sink's
        stored rows) is unchanged.  No-op unless attached via
        :meth:`attach_persistent`.
        """
        if not self._persistent_buffers:
            return
        for idx, bucket in self._per_layer.items():
            src = self._persistent_buffers.get(idx)
            if src is None:
                continue
            if self._incremental:
                if self._tail_depth <= 1:
                    # Length-1 overwrite: clone once, then ``copy_`` the latest
                    # persistent slice in each step (zero steady-state allocation,
                    # bucket stays length-1 so ``[-1]`` reads the latest).
                    if bucket:
                        bucket[0].copy_(src)
                    else:
                        bucket.append(src.clone())
                else:
                    # Bounded-tail ring: keep the last ``_tail_depth`` slices so
                    # finalize can pool the last *content* token.
                    bucket.append(src.clone())
                    if len(bucket) > self._tail_depth:
                        bucket.pop(0)
            else:
                # Full-retention: a distinct clone per step so ``stacked()`` can
                # build the [T, D] history.
                bucket.append(src.clone())
        self._forward_count += 1
        if self._step_sink is not None:
            self._step_sink(self.latest_per_layer())

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []
        # Drop the persistent-buffer source so a later transient capture (or a
        # no-probe gen) can't read stale slices through ``ingest_persistent``.
        self._persistent_buffers = {}

    def clear(self) -> None:
        self._per_layer = {}
        self._handles = []
        self._incremental = False
        self._step_sink = None
        self._max_layer = None
        self._tail_depth = 1
        self._forward_count = 0
        self._persistent_buffers = {}

    def tail_slice_at(self, forward_index: int) -> dict[int, torch.Tensor]:
        """Per-layer ``[D]`` slice for decode ``forward_index`` from the tail ring.

        Used by the aggregate-only finalize: ``forward_index`` is the
        generated-token index of the last content token (generated token ``k``
        was produced by forward ``k``), and the ring holds the last
        ``len(bucket)`` forwards ending at ``_forward_count - 1``.  The position
        is clamped into the ring, so a walk-back longer than the tail depth
        (more trailing specials than ``depth``) degrades to the oldest retained
        slice rather than indexing out of range.
        """
        out: dict[int, torch.Tensor] = {}
        F = self._forward_count
        for idx, bucket in self._per_layer.items():
            if not bucket:
                continue
            start = F - len(bucket)            # forward index of bucket[0]
            pos = forward_index - start
            pos = max(0, min(pos, len(bucket) - 1))
            out[idx] = bucket[pos]
        return out

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
        feeds ``Monitor.score_single_token`` with the latest
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

    def fire_step_sink(self) -> None:
        """Run the per-token step sink once, **after** the model forward (FIX F1).

        Invoked by ``generate_steered`` post-``model()`` rather than from inside
        the capture hook at the max probe layer.  Scoring here keeps the
        device→host sync the sink's score read incurs out of the *middle* of the
        forward pass, so the remaining transformer layers + LM head stay queued
        on the device instead of draining the pipeline at the highest hooked
        layer.  No-op when no sink is installed (aggregate-only / full-retention /
        no-probe captures).  Reads the latest per-layer slice the forward just
        stored — valid until the next forward overwrites the length-1 buffer (or
        rotates the tail ring).
        """
        if self._step_sink is not None:
            self._step_sink(self.latest_per_layer())


class SteeringHook:
    """Per-layer steering state: zero or more subspace / manifold groups.

    The 4.0 unified backend.  Every steering term — vectors, poles,
    ``~``/``|`` projections, ``!`` ablations, affine and curved ``%`` — lowers
    to a per-layer :func:`subspace_inject` group: the dispatch-synthesized
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
        # both groups here; ``subspace_inject``'s ``is_affine`` branch picks the
        # analytic-vs-foot-following path.  ``target_coord`` / ``origin_coord``
        # are authoring coordinates; ``along`` / ``onto`` are the per-layer
        # effective coefficients (share-weighted at apply time).  See
        # :meth:`_apply_manifold_groups`.
        self.manifold_groups: list[
            tuple[
                Trigger, LayerSubspace, ManifoldDomain,
                torch.Tensor, torch.Tensor, float, float,
                "float | torch.Tensor",
            ]
        ] = []
        # Per-token nearest-point foot state, parallel to ``manifold_groups``
        # (``None`` = cold, seed at the origin).  Affine groups ignore it (the
        # foot is ``q`` exactly); curved groups warm-start the Gauss-Newton
        # follower from it.  ``subspace_inject`` returns the refined foot each
        # fire; we stash the *last position* of it as the next token's warm
        # start.  Reset at recompose and by
        # :meth:`SteeringManager.reset_manifold_feet` at each generation start.
        self._manifold_feet: list[torch.Tensor | None] = []
        self._all_groups_always_active = False
        # Fast-path payload for the dominant steering case — exactly one affine
        # group, always-active (``Trigger.BOTH``).  When set,
        # ``(sub, domain, target, origin, along, onto, mean_proj, kappa)`` is the
        # unpacked single group, so ``hook_fn`` runs one ``subspace_inject`` +
        # ``copy_`` with no per-fire group loop, no trigger re-check, and no
        # foot-seed branch.  ``mean_proj = mean·basisᵀ`` (the ``(R,)`` reduced
        # projection of the layer mean) is precomputed here so the affine
        # shortcut skips both the full-width ``centered = h − mean`` temporary
        # and the per-fire matvec.  ``None`` ⇒ general path (multi-group,
        # gated, or any curved group).
        self._single_affine_fast: tuple[
            LayerSubspace, ManifoldDomain,
            torch.Tensor, torch.Tensor, float, float, torch.Tensor,
            "float | torch.Tensor",
        ] | None = None
        # Constant-add fast path for a **pure-push** affine group (all-zero
        # κ — no ``!`` ablation).  When such a group's foot translates by the
        # fixed offset ``along·target`` (κ=0 ⇒ ``p_new − q = along·target``,
        # independent of ``h``), the entire ``subspace_inject`` affine branch
        # reduces to ``h_new = h + c`` with ``c = (along·target)@basis`` a fixed
        # per-layer world vector — the projection ``q`` is computed only to be
        # cancelled, and the foot it returns is discarded for affine groups.  So
        # the offset is precomputed once here and the hot path does a single
        # in-place ``hidden.add_(c)`` (one kernel, zero alloc, no fp32 cast, no
        # ``copy_``).  ``_const_single`` is the offset for the single-affine-fast
        # case (when that group is pure-push), parallel to
        # ``_single_affine_fast``; ``_const_groups`` is the per-group offset
        # list parallel to ``self.manifold_groups`` (``None`` where a group is
        # curved or carries a non-zero κ, i.e. needs the full kernel because the
        # injection then depends on ``h``).  Both ``None`` ⇒ no group qualified.
        self._const_single: torch.Tensor | None = None
        self._const_groups: list[torch.Tensor | None] = []
        # Shared mutable context threaded in by SteeringManager.  Read-only
        # from the hook's perspective; the generation loop mutates fields.
        self._ctx: TriggerContext | None = None
        self._handle = None

    def recompose(
        self,
        manifold_entries: "list[tuple[LayerSubspace, ManifoldDomain, torch.Tensor, torch.Tensor, float, float, float | torch.Tensor, Trigger]] | None",
        ctx: TriggerContext,
        *,
        device: torch.device,
        dtype: "torch.dtype | None" = None,
    ) -> None:
        """Pre-compose this layer's subspace / manifold groups.

        Each entry is ``(subspace, domain, target_coord, origin_coord, along,
        onto, kappa, trigger)`` — the merged affine subspace (one entry, possibly
        per active trigger group) and each curved manifold are both groups here.
        ``ctx`` is the shared per-generation :class:`TriggerContext` the
        generation loop mutates and the hook reads at fire time.

        Subspace tensors are cast to **fp32** (the RBF / Gauss-Newton math is
        fp32 regardless of the model dtype; ``subspace_inject`` returns its
        fp32 result and :meth:`_apply_manifold_groups`'s ``hidden.copy_``
        downcasts to the model dtype on the write, so there is no per-fire
        model-dtype temporary).  An entry with both coefficients zero is a
        no-op and drops here.  A new group set cold-starts every
        foot-follower.

        ``dtype`` is the model's residual-stream dtype (threaded in from
        :meth:`SteeringManager.apply_to_model`).  It is used only to pre-cast the
        constant-add fast path's offset (see :meth:`_arm_constant_add`) so the
        hot path can ``hidden.add_`` an already-model-dtype tensor with no
        per-fire cast temporary.  ``None`` (the direct-``recompose`` test seam)
        skips that pre-cast and keeps the offset fp32 — still correct, since the
        in-place ``add_`` keeps ``hidden``'s dtype regardless.
        """
        self._ctx = ctx

        # --- subspace / manifold grouping ---
        # ``target_coord`` / ``origin_coord`` are authoring coordinates; the
        # subspace tensors stay **fp32** (the RBF / Gauss-Newton math is fp32
        # regardless of the model dtype, and quantizing ``node_params`` /
        # ``rbf_weights`` to bf16 would wreck the interpolant precision).
        # ``subspace_inject`` re-casts internally, so fp32 here is the precise
        # carrier, not an extra cost.  An entry with both coefficients
        # zero is a no-op and drops here.
        manifold_groups: list[
            tuple[
                Trigger, LayerSubspace, ManifoldDomain,
                torch.Tensor, torch.Tensor, float, float,
                "float | torch.Tensor",
            ]
        ] = []
        for sub, domain, target, origin, along, onto, kappa, trig in (
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
                kappa.to(device=device, dtype=torch.float32)
                if isinstance(kappa, torch.Tensor) else float(kappa),
            ))
        self.manifold_groups = manifold_groups
        self._all_groups_always_active = all(
            group[0] is Trigger.BOTH for group in manifold_groups
        )
        # Arm the single-affine-group fast path for the dominant case (one
        # always-active affine group — the merged folded-vector/pole/affine-``%``
        # subspace of a plain ``BOTH`` steering scope).  Curved groups, gated
        # triggers, or multiple groups fall back to the general loop.
        self._single_affine_fast = None
        if (
            len(manifold_groups) == 1
            and self._all_groups_always_active
            and manifold_groups[0][1].is_affine
        ):
            _trig, _sub, _dom, _tgt, _org, _alo, _ont, _kap = manifold_groups[0]
            # Precompute mean·basisᵀ ((R,) fp32 on device — ``_sub`` was cast in
            # the group append above) so the affine shortcut never materializes
            # the full-width ``centered`` temp per fire.
            _mp = _sub.mean @ _sub.basis.T
            self._single_affine_fast = (
                _sub, _dom, _tgt, _org, _alo, _ont, _mp, _kap,
            )
        # Arm the *constant-add* fast path on top of (or in place of) the above
        # for **pure-push** affine groups (all-zero κ — no ``!`` ablation).  Such
        # a group's injection is ``h_new = h + c`` with ``c = (along·target)@basis``
        # a fixed per-layer world vector (κ=0 ⇒ ``p_new − q = along·target``,
        # independent of ``h``), so the projection ``q`` inside ``subspace_inject``
        # is pure waste — computed only to cancel.  Precompute ``c`` once here (in
        # the model dtype when ``dtype`` is known, so the hot path's in-place
        # ``add_`` needs no per-fire cast temporary) and let ``hook_fn`` /
        # ``_apply_manifold_groups`` short-circuit to a single ``hidden.add_(c)``.
        # ``_const_single`` mirrors the single-affine-fast slot; ``_const_groups``
        # is the per-group list (``None`` where a group is curved or has non-zero
        # κ — the kernel is still needed there, since the ablation term
        # ``p_new − q = −along·κ·q`` depends on ``h``).
        if self._single_affine_fast is not None:
            _g = manifold_groups[0]
            self._const_single = self._pure_push_constant(
                _g[1], _g[3], _g[5], _g[7], device=device, dtype=dtype,
            )
        else:
            self._const_single = None
        self._const_groups = [
            self._pure_push_constant(
                sub, target, along, kappa, device=device, dtype=dtype,
            )
            if sub.is_affine else None
            for (_t, sub, _d, target, _o, along, _on, kappa) in manifold_groups
        ]
        # New group set ⇒ cold-start every foot-follower (seed at origin).
        self._manifold_feet = [None] * len(manifold_groups)

    @staticmethod
    def _pure_push_constant(
        sub: LayerSubspace,
        target: torch.Tensor,
        along: float,
        kappa: "float | torch.Tensor",
        *,
        device: torch.device,
        dtype: "torch.dtype | None",
    ) -> "torch.Tensor | None":
        """Precompute the constant world offset for a pure-push affine group.

        Returns ``c = (along·target) @ basis`` ((D,) per-layer offset) when the
        per-axis collapse mask ``kappa`` is all zero (a push-only term: vectors,
        poles, ``~``/``|`` projections, affine ``%``, and merges of them) — the
        case where the affine injection is exactly ``h_new = h + c`` (see
        :meth:`recompose`).  ``None`` for a mixed push+ablate group (any
        ``κ ≠ 0``), whose injection ``p_new − q = along·(target − κ·q)`` depends
        on ``q`` (hence ``h``) and so must keep the full kernel.

        The offset is cast to the model ``dtype`` when known so the hot-path
        ``hidden.add_`` is a single fused kernel with no per-fire cast; ``None``
        ``dtype`` keeps it fp32 (still correct — in-place ``add_`` keeps the
        destination dtype).  fp32 throughout the math (the ``@`` runs before the
        optional downcast).
        """
        # All-zero κ test: scalar ``0.0`` ⇒ pure push; a ``(R,)`` mask ⇒ pure
        # push iff every entry is zero.  Any ablation axis (κ≠0) disqualifies —
        # the injection then depends on ``h``.
        if isinstance(kappa, torch.Tensor):
            if bool(kappa.any()):
                return None
        elif float(kappa) != 0.0:
            return None
        basis = sub.basis.to(device=device, dtype=torch.float32)   # (R, D)
        tgt = target.to(device=device, dtype=torch.float32)        # (R,)
        c = (along * tgt) @ basis                                  # (D,) world offset
        # ``norm_cap`` (``‖h_new‖ ≤ 3·‖h‖``) is deliberately **skipped** here.
        # For the constant-add path it only ever guarded curved-RBF off-domain
        # extrapolation (``clamp_position`` keeps affine ``p_new`` in-box, and a
        # flat affine fit has no RBF to extrapolate — the affine-branch comment
        # in ``subspace_inject`` calls the cap "belt-and-suspenders, not the norm
        # semantic" for exactly this reason): a bounded steering offset ``c``
        # added to a large-norm residual-stream ``h`` cannot plausibly push
        # ``‖h + c‖`` past ``3·‖h‖``.  Dropping it is the point of the fix —
        # it removes the two per-fire norm reductions the cap needs.
        return c.to(dtype) if dtype is not None else c

    def hook_fn(self, module: Any, input: Any, output: Any) -> Any:
        # Constant-add fast path: one always-active **pure-push** affine group
        # (the dominant steering case — any push term / merge with no ``!``
        # ablation).  The whole injection is ``h += c`` with ``c`` the
        # precomputed per-layer offset, so it is a single in-place ``add_``: no
        # ``subspace_inject`` call, no projection, no fp32 cast, no ``copy_``.
        const = self._const_single
        if const is not None:
            hidden = output if isinstance(output, torch.Tensor) else output[0]
            hidden.add_(const)
            return output
        # Fast path: one always-active affine group (the common steering case).
        # Skips the group loop, the trigger re-check, and the foot-seed branch;
        # the analytic affine slide consults no per-step ctx, so it is correct
        # whether or not ``ctx`` is set.  Behaviorally identical to the general
        # path for this group shape (one ``subspace_inject`` + ``copy_``).
        # Reached only by a mixed push+ablate affine group (κ≠0 — the constant
        # add above didn't arm), which still needs the kernel.
        fast = self._single_affine_fast
        if fast is not None:
            hidden = output if isinstance(output, torch.Tensor) else output[0]
            sub, domain, target, origin, along, onto, mean_proj, kappa = fast
            h_new, _foot = subspace_inject(
                hidden, sub, domain, target, origin, along, onto,
                gn_steps=1, mean_proj=mean_proj, kappa=kappa,
            )
            hidden.copy_(h_new)
            return output
        groups = self.manifold_groups
        if not groups:
            return output
        ctx = self._ctx
        if ctx is None:
            return output
        # Cheap pre-check: any group active this step?  Skip the work entirely
        # if not (e.g. an ``AFTER_THINKING`` group during prefill).
        if (
            not self._all_groups_always_active
            and not any(_trigger_active(grp[0], ctx) for grp in groups)
        ):
            return output
        hidden = output if isinstance(output, torch.Tensor) else output[0]
        self._apply_manifold_groups(hidden, ctx)
        return output

    def _apply_manifold_groups(
        self, hidden: torch.Tensor, ctx: TriggerContext,
    ) -> None:
        """Apply every active manifold group via the unified kernel.

        Each group runs :func:`subspace_inject` — the single along/onto
        injection that replaced the angular/additive mode split.  The two
        per-layer coefficients are already share-weighted at
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
        lead = hidden.shape[:-1]
        all_groups_always_active = self._all_groups_always_active
        const_groups = self._const_groups
        for i, (
            trig, sub, domain, target, origin, along, onto, kappa,
        ) in enumerate(self.manifold_groups):
            if not all_groups_always_active and not _trigger_active(trig, ctx):
                continue
            if sub.is_affine:
                # Pure-push affine group (all-zero κ): the injection is a fixed
                # ``h += c`` — a single in-place ``add_`` of the precomputed
                # offset, no projection / kernel call / ``copy_``.  A mixed
                # push+ablate group (``const`` is ``None``) keeps the full kernel
                # (its ablation term depends on ``h``).  This path is reached by a
                # *mixed affine+curved* scope, whose affine group can still be
                # pure-push.
                const = const_groups[i]
                if const is not None:
                    hidden.add_(const)
                    continue
                h_new, _foot = subspace_inject(
                    hidden, sub, domain, target, origin,
                    along, onto, gn_steps=1, kappa=kappa,
                )
                hidden.copy_(h_new)
                continue
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
            h_new, foot = subspace_inject(
                hidden, sub, domain, target, foot_seed,
                along, onto, gn_steps=gn_steps, origin=origin,
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
# ``%``), since 4.0 lowers them all to the one along/onto kernel.
#
# Each op moves a piece of ``h`` whose *magnitude* is carried by the target's
# **neutral-anchored real coords** (Step 3): a high-signal layer's pole sits
# farther from the origin, so sliding ``along`` toward it displaces more there.
# The per-layer share (``_manifold_layer_shares`` / ``synth.share``) is
# normalized to **mean 1** (``Σ_L share_L = n_layers``), so
# ``eff_along_L = share_L · base_gain`` reads as a clean per-layer *slide
# fraction* ≈ ``base_gain`` on a typical layer — above ``base`` on the
# high-spread layers, below it on the flat ones — and is **n_layers-invariant**:
# one covered layer and a 30-layer fit both put ≈ ``base`` of slide on each
# contributing layer (a 4-dim and a 16-dim fit reach comparable behavior at the
# same α; an ``A ⊂ B`` nested subspace steers its shared axis identically).
#
# There is **no lever / ``N`` correction** (torn out in Step 8): it
# double-counted the magnitude the de-rogued real coords already carry, broke
# the A⊂B consistency above, and blew the gain up on whitened low-rank fits
# (``N ≈ 1e-4`` → ``base/N`` saturating every layer).  And **no ``[0, 1]`` clamp
# / water-fill** on ``along``: a high-signal layer is *meant* to overshoot past
# the pole; the affine/RBF ``norm_cap = 3·‖h‖`` inside ``subspace_inject`` is
# the only bound.  ``onto`` stays clamped ``[0, 1]`` (a residual-shrink fraction
# beyond 1 would overshoot through the zero-thickness wire or σ-tube).
#
# This is the **onto** (off-surface collapse) gain only: ``eff_onto_L =
# clamp(onto · share_L · _MANIFOLD_ONTO_GAIN, 0, 1)``.  On a legacy zero-thickness
# curved fit the kernel scales the off-surface residual by ``(1 − eff_onto)``; on
# a fuzzy σ-field fit it instead shrinks residual norm toward the local tube
# thickness.  That residual carries the per-token content variation, so combined
# with a directional ``along`` push too much onto erases the spread and degenerates
# into looping — the exact failure the translate-not-collapse ``along`` design
# avoids, reintroduced by over-shrinking the residual that held the spread.
# Calibrated on the gemma-4-12b ``emotions%dominant`` onto sweep — the affect
# manifold under a curved fit, along fixed at 0.3:
# at the old ``1.0`` even ``onto = 0.5`` fragmented and ``onto = 1.0`` collapsed to
# ``!!!``; ``0.5`` puts the recommended ``onto ≈ 0.5`` at a clean sweet spot and
# keeps ``onto = 1.0`` a strong-but-coherent ceiling, while below ``~0.3`` the
# [0, 1] knob saturates into no dynamic range (``onto = 0.5 ≈ 1.0``).  A [0, 1]
# dial whose top emits garbage is a bad dial, so ``onto = 1.0`` is deliberately the
# coherent maximum, not the over-steer edge.
_MANIFOLD_ONTO_GAIN = 0.5

# --- translate gain (prototype) ----------------------------------------------
# The injection *translates* the in-subspace foot by a fixed offset toward the
# target rather than collapsing every token's foot onto it (see
# ``subspace_inject`` / ``ManifoldDomain.translate_foot``): the fixed offset
# preserves the per-token in-subspace spread, which the kernel ablation showed
# keeps strong steer coherent (collapse → looping degeneration).
#
# Translate is unbounded where collapse saturated (a fixed offset compounds
# across layers rather than landing *on* the target), so the slide gain runs ~an
# order of magnitude below the old collapse gain.  A typical layer gets
# ``eff_along ≈ _SUBSPACE_GAIN``.
#
# **Whitened-normalization recalibration.**  ``synthesize_subspace`` now emits a
# *whitened-unit* affine target (``‖target@basis‖_M = 1`` per push term, magnitude
# carried by α), so the avg per-layer whitened push is simply ``GAIN·α`` — the same
# for every target.  Under the prior raw-Euclidean target the push was
# ``eff_along · ‖node_coords‖₂``, which baked in each node's distance from neutral
# (caveman ~17, formal ~0.3, a ~100× spread); the old ``0.125`` was tuned against
# that ~17 scale, so on the unit scale it under-pushes ~100×.
#
# Live-calibrated on a gemma-4-12b α-sweep (thinking off; ``formal.casual%formal``,
# ``personas%caveman``, ``personas%hacker``).  In **effective gain ``E = GAIN·α``**:
# the dead→expressive transition is ``E ≈ 7-8`` for every target, but the coherence
# *ceiling* (token-soup looping past it) varies ~2× by target — hacker shatters at
# ``E ≈ 12``, caveman at ``E ≈ 17``, formal still coherent past ``E ≈ 22``.  There
# is **no single E** that makes caveman full AND keeps hacker coherent; this is the
# §10 per-persona coherence variance, which a scalar gain cannot unify (it is a
# steering-access property of the fit, not the geometric scale the whitened
# normalization fixed).
#
# So the calibration is **coherence-first**: put the recommended ``α ≈ 0.5`` at
# ``E ≈ 8`` — the strongest setting where *every* target (including the fragile
# hacker) is clearly register-shifted yet coherent — giving ``GAIN = 16``.  ``α ≈
# 1.0`` then lands at ``E ≈ 16`` (the documented "strong / over-steered" zone:
# robust concepts still coherent, hard personas break — dial α down per target).
# This is the payoff of the whitened normalization: pre-fix, ``0.5 formal%formal``
# did nothing while ``0.5 personas%caveman`` slammed; now ``0.5`` of *either* lands
# in the same coherent band.  (Tune up toward ``E ≈ 12`` — α ≈ 0.75 — for fuller
# persona expression where the target tolerates it.)  ``_MANIFOLD_ONTO_GAIN`` stays the
# gain for ``onto`` only (the off-surface collapse share-weight).
_SUBSPACE_GAIN = 16.0

# --- curved-path translate gain (separate from the affine gain above) --------
# The ``_SUBSPACE_GAIN = 16`` above is the **affine** path's free-magnitude gain
# (whitened-unit target, ``norm_cap``-bounded).  The **curved** path is different
# in kind: its target is the node's *raw domain coordinates* and ``subspace_inject``
# translates the foot by ``eff_along·(target − origin)`` (``domain.translate_foot``),
# so ``eff_along`` is a **fraction of the way to the node** (``1.0`` lands on it),
# not a free magnitude.  ``norm_cap = 3·‖h‖`` bounds the off-domain RBF
# extrapolation, so a curved fit doesn't detonate the way the affine path would —
# instead, *past* ``eff_along ≈ 1`` the foot keeps translating past the node.
#
# Live-calibrated on a CLEAN gemma-4-12b ``months_loop%january`` α-sweep
# (stateless=True — earlier sweeps were confounded by conversation accumulation;
# see ``scripts/curved_gain_calibrate.py`` / ``months_low_gain.py``).  ``along=1.0``
# at gain ``4`` (``eff_along ≈ 4``) lands the vivid coherent winter sweet spot
# ("skeletal trees heavy with frost", consistent across seeds); gain ``2`` is milder
# but clearly cold; the prior ``1.5`` was too weak (only a "cool prickle").
#
# CAVEAT — non-monotonic above the sweet spot on PERIODIC fits (a9 will revisit).
# ``months_loop`` is a period-12 ring; ``translate_foot`` wraps, and ``eff_along`` is
# share-weighted (per-layer ``share ∈ [0.19, 1.47]``), so past ``~1`` each layer
# orbits the loop at its own rate and the layers land on *different* months → the
# seasonal signal washes out / scatters rather than intensifying (foot-ring
# geometry: ``scripts/months_foot_ring.py``).  ``4`` rides a coincidentally-coherent
# part of the wrap; it is NOT a "stronger = more january" magnitude.  The principled
# fix (deferred) is to clamp the curved ``eff_along`` to ``[0,1]`` and stop
# share-weighting it on periodic domains so ``along=1`` lands every layer exactly on
# the node.  Prototype, like its affine sibling.
_MANIFOLD_ALONG_GAIN = 4.0

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
    kappa: torch.Tensor,
    curved_basis: torch.Tensor,
) -> "tuple[LayerSubspace, torch.Tensor, torch.Tensor] | None":
    """Project a merged affine subspace out of the curved manifolds' span.

    Strips the curved-span component from the affine basis rows *and* the push
    displacement so the merged affine subspace and the orthogonal curved
    manifolds at a layer operate on disjoint directions — the curved manifold
    wins the shared directions (ARCHITECTURE §6 precedence), the affine slide
    handles the complement.  ``sub`` is the synthesized affine ``LayerSubspace``
    (orthonormal ``basis``), ``target`` its ``(R,)`` push coord, ``curved_basis``
    the stacked ``(Rc, D)`` orthonormal rows of every curved manifold at this
    layer.  Carries the per-axis collapse mask ``kappa`` through the
    re-orthonormalization (each new axis inherits its squared projection onto the
    old ablate span).  Returns the re-orthonormalized ``(subspace', target',
    kappa')`` or ``None`` when the affine span lies entirely inside the curved
    span (nothing left to steer there).
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
    # Carry the per-axis collapse mask κ through the re-orthonormalization: each
    # new axis inherits ``Σ_i κ_i (new_axis · old_basis_i)²`` — its squared
    # projection onto the old ablate span.  ``M = new_basis @ basisᵀ`` (R', R).
    M = new_basis @ basis.T                          # (R', R)
    new_kappa = (M * M) @ kappa.to(torch.float32)    # (R',)
    return LayerSubspace.affine(sub.mean, new_basis), new_target, new_kappa

# Gauss-Newton steps taken on a *cold* foot (seed at origin ``O``).  The
# warm path takes one step per token; the cold fire — the prefill window,
# or the first decode step under a non-prompt trigger — converges the foot
# better with a handful so the early tokens steer from an accurate foot.
# Cheap: O(R) per position, off the model-forward critical path.
_MANIFOLD_COLD_GN_STEPS = 4


def _normalize_shares_mean1(raw: dict[int, float]) -> dict[int, float]:
    """Normalize per-layer share scores to **mean 1** (``Σ_L share_L = n_layers``).

    So ``eff_along_L = share_L · base`` is a clean per-layer slide fraction
    ≈ ``base`` on a typical layer and n_layers-invariant (see
    ``_MANIFOLD_ONTO_GAIN``).  Degenerate guard: an all-zero / near-zero total
    falls back to a uniform ``1.0`` per layer.
    """
    n_layers = max(1, len(raw))
    total = sum(raw.values())
    if total <= 1e-12:
        return dict.fromkeys(raw, 1.0)
    return {L: s / total * n_layers for L, s in raw.items()}


def _manifold_layer_shares(manifold: Any) -> dict[int, float]:
    # Prefer the whitened (Mahalanobis) per-layer share baked at fit time —
    # the subspace-restricted analogue of vector steering's ``‖d‖_M`` bake
    # score (see ``LayerWhitener.subspace_gram`` /
    # ``ManifoldExtractionPipeline.fit``).  Requires *full* layer coverage:
    # the share is a cross-layer-normalized weight, so mixing whitened and
    # Euclidean scalars across layers would compare incommensurable
    # metrics.  When the baked share is absent (no whitener at fit time —
    # CPU test stubs) or partial, fall back to the Euclidean centroid-
    # spread ``‖coords‖_F``.  Normalized to **mean 1** (``Σ_L share_L =
    # n_layers``, not 1) so ``eff_along_L = share_L · base_gain`` is a clean
    # per-layer slide fraction ≈ ``base`` on a typical layer and
    # n_layers-invariant — see ``_MANIFOLD_ONTO_GAIN``.
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
    return _normalize_shares_mean1(layer_scores)


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
        # backend.  Each value is ``{synth, trigger}``; ``apply_to_model``
        # lowers them to per-layer ``subspace_inject`` entries alongside curved
        # manifolds.
        self.subspaces: dict[str, dict[str, Any]] = {}
        self.ctx: TriggerContext = TriggerContext()
        # Persistent compile-clean steering path (MPS torch.compile).  A single
        # branchless ``hidden.add_(offset)`` hook per layer, attached ONCE before
        # compile (so it is traced into the captured graph) and never
        # re-registered — the per-gen steering is pushed by updating the
        # ``(D,)`` offset buffers *in place* (``copy_``), which torch.compile
        # does not retrace on (tensor value, not identity/structure).  Only the
        # static-affine pure-push case lowers here (``compute_static_offsets``);
        # curved / gated / ablation steering keeps the transient ``hooks`` path
        # on the eager model.  ``_compiled_offsets`` maps layer → buffer;
        # populated by :meth:`adopt_compiled_offsets` at session construction.
        self._compiled_offsets: dict[int, torch.Tensor] = {}
        self._compiled_offset_handles: list[Any] = []

    def all_fast_path(self) -> bool:
        """True iff no steering hook is attached (the unsteered path).

        The cheapest case: nothing mutates the residual stream, so StaticCache
        / ``torch.compile`` graph capture is unconditionally eligible.  A
        ctx-consulting / curved / gated hook still forces the eager
        DynamicCache path; the *static-affine* steered case is captured by the
        separate :meth:`static_steerable` signal below.
        """
        return not self.hooks

    def static_steerable(self) -> bool:
        """True iff every attached hook is the static single-affine fast path.

        The precondition for routing *steered* generation through StaticCache /
        CUDA-graph capture: each steered layer carries exactly one always-active
        (``Trigger.BOTH``) affine group — the analytic subspace slide
        (:attr:`SteeringHook._single_affine_fast`) consults no per-step
        ``TriggerContext`` and threads no foot state, so its injection is a
        fixed sequence of tensor ops, identical every decode step.  StaticCache
        never bypasses forward hooks (the hook fires on every forward, writing
        into the preallocated K/V buffers), so the steering applies unchanged;
        the analytic affine ops are also traceable, so ``torch.compile``
        graph capture can fold them into the captured region (and
        ``_compile_with_probe``'s warmup falls back to eager if a given arch
        breaks capture).  Any curved manifold, probe gate, or phase-gated
        trigger leaves a hook on the general (ctx-consulting) path and
        disqualifies the whole generation.  False when unsteered — that is
        :meth:`all_fast_path`'s (cheaper) case.
        """
        hooks = self.hooks
        return bool(hooks) and all(
            h._single_affine_fast is not None for h in hooks.values()
        )

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
        share-weighted per-layer ``along`` / ``onto`` coefficients; the hot path
        runs :func:`subspace_inject`.

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
        trigger: Trigger = Trigger.BOTH,
    ) -> None:
        """Register a dispatch-synthesized merged affine subspace (4.0).

        ``synth`` (one per active trigger group) carries the per-layer affine
        :class:`LayerSubspace`, the ``along`` ``target_coord`` (every active
        push term's coeff-scaled pole already composed in), and the
        un-normalized per-layer budget ``share`` (``‖Δ_L‖``).

        At :meth:`apply_to_model` each layer becomes a per-layer
        ``(subspace, CustomDomain(R_L), target_coord, origin=0, eff_along,
        onto=0)`` entry routed through the same :func:`subspace_inject` hot
        path as a curved manifold — the affine analytic shortcut slides the
        in-subspace component toward ``target_coord`` with
        ``eff_along_L = share_L · base_gain`` (``share_L`` mean-1 normalized; no
        lever / ``N``, no ``[0, 1]`` clamp — the de-rogued real-coord target
        carries the magnitude and the ``norm_cap`` inside ``subspace_inject``
        bounds an over-share layer's overshoot).  ``onto = 0`` (the surface
        fills its span).
        """
        self.subspaces[name] = {
            "synth": synth,
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
        :func:`subspace_inject` groups, orthogonalizing the affine subspace
        against the curved manifolds so they compose with zero cross-talk,
        then recomposing the per-layer hooks.  ``dtype`` is the model dtype the
        hook casts the fp32 subspace result back to.
        """

        # Manifold entries: stamp the per-layer subspace + domain + authoring
        # ``target`` / ``origin`` coords and the two share-weighted op
        # coefficients.  The kernel slides the foot in
        # *coordinate* space, so there is no fixed world-target precompute —
        # only the (layer-independent) authoring coords.  Two *curved* manifolds
        # may share a layer only if their subspaces are (near-)orthogonal
        # (``_CURVED_ORTHO_TOL``); overlapping ones raise
        # ``OverlappingManifoldError`` (each would clobber the other's
        # in-subspace component).  ``curved_basis_by_layer`` accumulates the
        # curved spans so the merged affine subspace can be orthogonalized
        # against them below.
        #
        # **Gain (share-weight every op).**  Each of ``along`` / ``onto`` gets
        # the same per-layer factor ``share_L · base``: ``share_L`` (whitened
        # Mahalanobis share, else Euclidean centroid-spread) is how
        # discriminative the manifold is at that layer, normalized to mean 1
        # (``Σ_L share_L = n_layers``); ``base`` is the one gain constant.  No
        # lever / ``N`` and no water-fill (both torn out in Step 8 — see the
        # ``_MANIFOLD_ONTO_GAIN`` docstring): ``along`` is left un-clamped so a
        # high-share layer overshoots past the target (the ``norm_cap`` inside
        # ``subspace_inject`` is the only bound), while ``onto`` stays clamped
        # ``[0, 1]`` per layer (beyond 1 would overshoot through the wire/tube).
        manifold_by_layer: dict[
            int,
            list[tuple[
                LayerSubspace, ManifoldDomain,
                torch.Tensor, torch.Tensor, float, float,
                "float | torch.Tensor", Trigger,
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

            # Curved-path **fraction** gain (NOT the affine magnitude gain):
            # ``eff_along`` is the fraction of the way to the node, so it must stay
            # near [0, ~2] or the RBF extrapolates off-domain (see
            # ``_MANIFOLD_ALONG_GAIN``).  ``onto`` clamps per layer.
            eff_along = {
                L: along * shares[L] * _MANIFOLD_ALONG_GAIN
                for L in manifold.layers
            }
            eff_onto = {
                L: max(0.0, min(1.0, onto * shares[L] * _MANIFOLD_ONTO_GAIN))
                for L in manifold.layers
            }

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
                    eff_along[layer_idx], eff_onto[layer_idx], 0.0, trigger,
                ))  # κ = 0: curved manifolds are push-only (pure translate)

        # Dispatch-synthesized merged affine subspaces (4.0 unified backend).
        # Each ``synth`` is already neutral-anchored with its ``along`` target
        # composed from every active push term's coeff-scaled pole; here we only
        # set the per-layer slide budget ``eff_along_L = share_L · base`` (mean-1
        # share, no lever / clamp — Step 8) and lower each layer to a
        # ``CustomDomain(R_L)`` ``subspace_inject`` entry (the affine analytic
        # shortcut — no GN / RBF / foot solve).  The target carries the
        # strength, so there is no separate user-α multiply here; ``onto = 0``
        # (the surface fills its span).  Curved-vs-affine orthogonalization +
        # the relaxed overlap check land with the session dispatch flip (Step
        # 5b); here a subspace simply joins ``manifold_by_layer``.
        for s in self.subspaces.values():
            synth: SynthesizedSubspace = s["synth"]
            sub_trigger: Trigger = s["trigger"]
            layer_set = list(synth.layers)
            if not layer_set:
                continue

            # Normalize the per-layer budget share to **mean 1**
            # (``Σ_L share_L = n_layers``) so ``eff_along_L = share_L · base``
            # is a clean per-layer slide fraction and n_layers-invariant (one
            # covered layer and a 30-layer fit both put ≈ ``base`` of slide on
            # each contributing layer; A⊂B steers its shared axis identically).
            raw_share = {L: float(synth.share.get(L, 0.0)) for L in layer_set}
            shares = _normalize_shares_mean1(raw_share)

            for L in layer_set:
                sub_L = synth.layers[L]
                sub_target = synth.target_coord[L].to(torch.float32)
                # Per-axis collapse mask κ (0 push / translate, 1 ablate /
                # collapse) — default all-translate when a synth predates it.
                sub_kappa = synth.kappa.get(L)
                sub_kappa = (
                    sub_kappa.to(torch.float32) if sub_kappa is not None
                    else torch.zeros(sub_L.rank, dtype=torch.float32)
                )
                # Orthogonalize the affine subspace against any curved manifold
                # sharing this layer (curved wins the shared directions); κ rides
                # through the re-orthonormalization.  Drop the layer if the affine
                # span lies entirely inside the curved span (nothing left here).
                curved = curved_basis_by_layer.get(L)
                if curved is not None:
                    res = _orthogonalize_affine_against(
                        sub_L, sub_target, sub_kappa, curved,
                    )
                    if res is None:
                        continue
                    sub_L, sub_target, sub_kappa = res
                r_l = sub_L.rank
                sub_domain = CustomDomain(r_l)
                # Affine origin is span-coord 0 (neutral → coord 0, §5); the
                # foot seed / cold-start is unused on the affine shortcut.
                sub_origin = torch.zeros(r_l, dtype=torch.float32)
                # No lever / ``N`` and no ``[0, 1]`` clamp (Step 8): the
                # de-rogued real-coord target carries the magnitude; the per-axis
                # share-weighted ``eff_along`` is unclamped (``norm_cap`` bounds
                # it in ``subspace_inject``).
                eff_along_L = shares[L] * _SUBSPACE_GAIN
                manifold_by_layer.setdefault(L, []).append((
                    sub_L, sub_domain, sub_target, sub_origin,
                    eff_along_L, 0.0, sub_kappa, sub_trigger,
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
                dtype=dtype,
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

    # -- Persistent compile-clean offset path (MPS torch.compile) --------------

    def adopt_compiled_offsets(
        self, buffers: dict[int, torch.Tensor], handles: list[Any],
    ) -> None:
        """Adopt the persistent per-layer offset buffers + hook handles.

        Built by :func:`install_persistent_offset_hooks` in ``from_pretrained``
        *before* ``torch.compile`` so the branchless ``add_(offset)`` hooks are
        traced into the captured graph.  The session hands them here so the
        manager can push static-affine steering by updating the buffers in place.
        """
        self._compiled_offsets = buffers
        self._compiled_offset_handles = handles

    def has_compiled_offsets(self) -> bool:
        return bool(self._compiled_offsets)

    def compute_static_offsets(self) -> "dict[int, torch.Tensor] | None":
        """Per-layer constant steering offsets, or ``None`` if not compile-clean.

        Returns ``{}`` for unsteered (all buffers zero), a ``{layer: (D,)}`` map
        for the static-affine pure-push case (every subspace is an always-active
        ``Trigger.BOTH`` push with zero ablation mask κ and no curved manifold),
        and ``None`` for anything that needs the per-token kernel (curved ``%``,
        a probe gate / phase trigger, or an ``!`` ablation).  The offset is the
        same world vector the transient :meth:`SteeringHook._pure_push_constant`
        fast path adds (``(share_L·gain·target) @ basis``), so the compiled and
        eager pushes are numerically identical.
        """
        if self.manifolds:
            return None  # a curved manifold isn't a constant add
        offsets: dict[int, torch.Tensor] = {}
        for s in self.subspaces.values():
            synth: SynthesizedSubspace = s["synth"]
            if s["trigger"] is not Trigger.BOTH:
                return None  # gated / phased — needs the ctx-consulting hook
            layer_set = list(synth.layers)
            if not layer_set:
                continue
            raw_share = {L: float(synth.share.get(L, 0.0)) for L in layer_set}
            shares = _normalize_shares_mean1(raw_share)
            for L in layer_set:
                sub_L = synth.layers[L]
                kappa = synth.kappa.get(L)
                if kappa is not None and bool((kappa.abs() > 0).any()):
                    return None  # ablation: injection depends on h, not a const
                target = synth.target_coord[L].to(torch.float32)
                basis = sub_L.basis.to(torch.float32)              # (R, D)
                eff_along = shares[L] * _SUBSPACE_GAIN
                c = (eff_along * target) @ basis                   # (D,)
                offsets[L] = offsets[L] + c if L in offsets else c
        return offsets

    def write_compiled_offsets(self, offsets: dict[int, torch.Tensor]) -> None:
        """Push ``offsets`` into the persistent buffers in place (``copy_``).

        Layers absent from ``offsets`` are zeroed, so a stale push can't leak
        into a layer the current steering doesn't touch.  In-place is load-
        bearing: reassigning the buffer tensor would change its identity and
        force a torch.compile retrace.
        """
        for L, buf in self._compiled_offsets.items():
            c = offsets.get(L)
            if c is None:
                buf.zero_()
            else:
                buf.copy_(c)   # copy_ casts to the buffer's device/dtype

    def zero_compiled_offsets(self) -> None:
        """Zero every persistent offset buffer (no steering push)."""
        for buf in self._compiled_offsets.values():
            buf.zero_()

    def detach_transient_hooks(self) -> None:
        """Detach the ctx-consulting steering hooks, keeping synthesis state.

        Used by the compiled offset path: the per-layer push is carried by the
        persistent offset buffers, so the transient :class:`SteeringHook`s must
        not also fire (they would double-apply on the eager fallback, and their
        attach/detach churn recompiles the graph).  ``subspaces`` / ``manifolds``
        are left intact (already consumed into the offsets; cleared on scope
        pop).
        """
        for hook in self.hooks.values():
            hook.detach()
        self.hooks.clear()

    def detach_compiled_offsets(self) -> None:
        """Remove the persistent offset hooks (compile-failure / teardown)."""
        for h in self._compiled_offset_handles:
            h.remove()
        self._compiled_offset_handles = []
        self._compiled_offsets = {}


def install_persistent_offset_hooks(
    layers: "torch.nn.ModuleList",
    hidden_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> "tuple[dict[int, torch.Tensor], list[Any]]":
    """Attach one branchless ``hidden.add_(offset)`` forward hook per layer.

    The persistent compile-clean steering path: each layer gets a preallocated
    ``(hidden_size,)`` offset buffer (zero-initialized) and a hook that adds it
    in place.  Call **before** ``torch.compile`` so the add is captured in the
    traced graph; per-generation steering then updates the buffers in place
    (:meth:`SteeringManager.write_compiled_offsets`) without retracing.  Returns
    ``(buffers, handles)`` for :meth:`SteeringManager.adopt_compiled_offsets`.
    """
    buffers: dict[int, torch.Tensor] = {}
    handles: list[Any] = []

    def _make(buf: torch.Tensor) -> Any:
        def _hook(module: Any, inp: Any, out: Any) -> Any:
            h = out if isinstance(out, torch.Tensor) else out[0]
            h.add_(buf)
            return out
        return _hook

    for idx in range(len(layers)):
        buf = torch.zeros(hidden_size, device=device, dtype=dtype)
        buffers[idx] = buf
        handles.append(layers[idx].register_forward_hook(_make(buf)))
    return buffers, handles


def install_persistent_capture_hooks(
    layers: "torch.nn.ModuleList",
    hidden_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> "tuple[dict[int, torch.Tensor], list[Any]]":
    """Attach one branchless ``buf.copy_(hidden[0, -1, :])`` forward hook per layer.

    The persistent compile-clean *capture* path (the read-side analogue of
    :func:`install_persistent_offset_hooks`): each layer gets a preallocated
    ``(hidden_size,)`` buffer and a hook that copies its last-position hidden
    slice in.  Call **before** ``torch.compile`` so the ``copy_`` is fused into
    the traced graph; a probed generation then reads the latest slice per forward
    via :meth:`HiddenCapture.ingest_persistent` (post-forward, host side) instead
    of registering transient capture hooks that would graph-break / recompile.

    Hooks cover **every** layer (the probe roster isn't known until after compile,
    so capture a fixed superset and let the monitor read its subset).  The
    ``copy_`` is a pure write with no effect on the model output, so the hooks are
    always safe to leave installed — unprobed and eager-fallback gens just ignore
    the buffers.  Returns ``(buffers, handles)`` for the session to adopt.
    """
    buffers: dict[int, torch.Tensor] = {}
    handles: list[Any] = []

    def _make(buf: torch.Tensor) -> Any:
        def _hook(module: Any, inp: Any, out: Any) -> Any:
            h = out if isinstance(out, torch.Tensor) else out[0]
            buf.copy_(h[0, -1, :])
            return out
        return _hook

    for idx in range(len(layers)):
        buf = torch.zeros(hidden_size, device=device, dtype=dtype)
        buffers[idx] = buf
        handles.append(layers[idx].register_forward_hook(_make(buf)))
    return buffers, handles
