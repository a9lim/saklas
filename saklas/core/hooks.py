"""Steering hooks for activation steering on transformer models."""

from __future__ import annotations

import math

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import torch

from saklas.core.manifold import (
    BoxDomain,
    CustomDomain,
    LayerSubspace,
    Manifold,
    ManifoldDomain,
    SynthesizedSubspace,
    _ortho_basis,
    subspace_inject,
)
from saklas.core.triggers import Trigger, TriggerContext


def _trigger_active(trigger: Trigger, ctx: TriggerContext) -> bool:
    """Hot-path trigger predicate with the default preset as a true no-op."""
    return trigger is Trigger.BOTH or trigger.active(ctx)


def _affine_push_offset(
    subspace: LayerSubspace,
    target: torch.Tensor,
    eff_along: float,
    *,
    device: "torch.device | None" = None,
) -> torch.Tensor:
    """The world-space constant a pure-push affine slide adds at one layer.

    ``c = (eff_along · target) @ basis`` — the ``(D,)`` displacement the affine
    branch of :func:`subspace_inject` produces when the per-axis collapse mask
    ``κ`` is all zero (``p_new − q = eff_along·target``, independent of ``h``).
    Both consumers of a lowered affine term contract through here: the transient
    hook's constant-add fast path (:meth:`SteeringHook._pure_push_constant`) and
    the persistent compiled-offset buffers
    (:meth:`SteeringManager.compute_static_offsets`), so the compiled and eager
    pushes are the same tensor expression rather than two transcriptions of it.

    fp32 throughout (the project-wide norm/accumulation invariant); ``device``
    ``None`` keeps the inputs where they are.
    """
    basis = subspace.basis.to(device=device, dtype=torch.float32)   # (R, D)
    tgt = target.to(device=device, dtype=torch.float32)             # (R,)
    return (eff_along * tgt) @ basis                                # (D,)


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

    Hot-path discipline: the hook does device work only — a ``(dim,)`` slice
    ``detach()``-ed and either ``copy_``-ed into a preallocated per-layer
    buffer (incremental retention, zero steady-state allocation) or cloned
    into a per-layer Python list (the tail ring and full retention, where each
    step needs a distinct tensor). Stacking, fp32 casting, and every host read
    happen after the forward, not inside it.

    Retention is armed by :meth:`set_retention` (or one of its three named
    aliases) after :meth:`attach`, along two orthogonal axes:

    - **depth 1** — each per-layer hook OVERWRITES a single preallocated
      ``(D,)`` buffer instead of appending, so device memory stays
      O(layers·D) for the whole generation and the decode loop allocates
      nothing. ``latest_per_layer()`` and the ``bucket[-1]`` reads the
      streaming tap relies on keep working — a length-1 bucket's ``[-1]`` is
      still the latest.
    - **depth > 1** — a bounded *ring* of the last ``depth`` slices per layer,
      so finalize can pool the last *content* token via
      :meth:`tail_slice_at`; the ring is deep enough to walk back past
      trailing special tokens (EOS / end-of-turn).
    - **step_sink set / unset** — with a sink, one callback fires per forward
      (via :meth:`fire_step_sink` / :meth:`ingest_persistent`) carrying the
      latest per-layer slice, and the session scores that token. Without one,
      the decode loop pays zero per-token scoring: T scorings collapse to a
      single pooled read at finalize.

    Unarmed (the state :meth:`attach` leaves) is full retention: a distinct
    clone per step, so ``stacked()`` returns the full ``[T, D]`` — the
    ``return_hidden`` path.
    """

    def __init__(self) -> None:
        self._per_layer: dict[int, list[torch.Tensor]] = {}
        self._handles: list[Any] = []
        # Incremental-mode state. ``_incremental`` flips the per-layer
        # hook from append to overwrite; ``_step_sink`` is invoked once
        # per forward after the highest hooked layer (``_max_layer``)
        # stores this step's slice. All reset on attach/clear.  The sink
        # receives ``(step_id, latest_per_layer)`` — the decode loop's
        # forward index rides along so per-token scorers can prime the
        # instrument runs' step-keyed observe memos.
        self._incremental: bool = False
        self._step_sink: (
            Callable[[int, dict[int, torch.Tensor]], None] | None
        ) = None
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
        self._tail_layers: frozenset[int] | None = None
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
        # Batched aggregate-only capture: ``generate_batch`` can delegate the
        # decode loop to ``transformers.generate`` and still retain the final
        # per-row probe state by storing one ``[B, D]`` tail slice per layer per
        # forward.  The normal single-row paths keep using ``_per_layer``.
        self._batch_per_layer: dict[int, list[torch.Tensor]] = {}
        self._batch_forward_count: int = 0
        # Selective prompt capture.  Decode-time measurements retain only the
        # last position, but a loom-attached generation may also ask for the
        # producer positions of visible authored tokens from the first prefill
        # forward.  Keeping only those rows avoids cloning the full [T, D]
        # prompt at every measured layer.
        self._prompt_positions: tuple[int, ...] = ()
        self._prompt_per_layer: dict[int, torch.Tensor] = {}

    def attach(
        self, layers: "torch.nn.ModuleList", layer_indices: list[int]
    ) -> None:
        self._per_layer = {idx: [] for idx in layer_indices}
        self._handles = []
        # Attach resets incremental state — a fresh capture starts in the
        # append (full-retention) mode. ``set_retention`` (or one of its named
        # aliases) must be called after attach to opt into incremental scoring
        # / bounded-tail capture for this gen.
        self._incremental = False
        self._step_sink = None
        self._max_layer = None
        self._tail_depth = 1
        self._tail_layers = None
        self._forward_count = 0
        self._persistent_buffers = {}
        self._prompt_positions = ()
        self._prompt_per_layer = {}
        for idx in layer_indices:
            bucket = self._per_layer[idx]

            def _make(bucket_ref: list[torch.Tensor], layer_idx: int) -> Any:
                def _hook(module: Any, input: Any, output: Any) -> None:
                    h = output if isinstance(output, torch.Tensor) else output[0]
                    if (
                        self._prompt_positions
                        and layer_idx not in self._prompt_per_layer
                        and int(h.shape[1]) > self._prompt_positions[-1]
                    ):
                        positions = torch.tensor(
                            self._prompt_positions,
                            device=h.device,
                            dtype=torch.long,
                        )
                        self._prompt_per_layer[layer_idx] = (
                            h[0].index_select(0, positions).detach().clone()
                        )
                    src = h[0, -1, :].detach()
                    if self._incremental:
                        keep_deep_tail = (
                            self._tail_depth > 1
                            and (
                                self._tail_layers is None
                                or layer_idx in self._tail_layers
                            )
                        )
                        if not keep_deep_tail:
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
                        # step's value.  Count the forward — it drives the tail
                        # ring's ``tail_slice_at`` mapping.  The step sink is
                        # deliberately NOT run here: it fires post-forward from
                        # :meth:`fire_step_sink`, so its host-side score read
                        # never drains the device pipeline mid-forward and the
                        # remaining transformer layers + LM head stay queued.
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

    def attach_batch_tail(
        self,
        layers: "torch.nn.ModuleList",
        layer_indices: list[int],
        *,
        depth: int,
    ) -> None:
        """Capture batched ``[B, D]`` last-position slices in a bounded ring.

        Used by the ``generate_batch`` fast path for stateless aggregate probe
        reads.  It deliberately does not support per-token sinks, hidden-state
        returns, or persistent compile-clean buffers; those remain on the custom
        single-row loop where their per-step semantics are already explicit.
        """
        self.clear()
        self._batch_per_layer = {idx: [] for idx in layer_indices}
        tail_depth = max(1, int(depth))
        for idx in layer_indices:
            bucket = self._batch_per_layer[idx]

            def _make(bucket_ref: list[torch.Tensor], layer_idx: int) -> Any:
                def _hook(module: Any, input: Any, output: Any) -> None:
                    h = output if isinstance(output, torch.Tensor) else output[0]
                    src = h[:, -1, :].detach().clone()
                    bucket_ref.append(src)
                    if len(bucket_ref) > tail_depth:
                        bucket_ref.pop(0)
                    if layer_idx == layer_indices[-1]:
                        self._batch_forward_count += 1
                return _hook

            self._handles.append(
                layers[idx].register_forward_hook(_make(bucket, idx)),
            )

    def set_retention(
        self,
        *,
        depth: int = 1,
        step_sink: "Callable[[int, dict[int, torch.Tensor]], None] | None" = None,
        tail_layers: "set[int] | frozenset[int] | None" = None,
    ) -> None:
        """Arm incremental capture: how much to retain, and who reads it.

        The single retention setter — must be called after :meth:`attach`.  It
        flips the per-layer hook out of the append (full-retention) mode and
        sets the two orthogonal choices the four capture modes pick between:

        - ``depth`` — how many trailing slices each layer keeps.  ``1`` (the
          default) is a single preallocated ``(D,)`` buffer per layer,
          ``copy_``-ed in place every forward, so the decode loop allocates
          nothing and ``bucket[-1]`` reads stay valid.  ``> 1`` keeps a ring of
          the last ``depth`` slices so finalize can pool the last *content*
          token via :meth:`tail_slice_at` (deep enough to walk back past
          trailing specials: EOS, end-of-turn).
        - ``step_sink`` — the per-token reader, or ``None`` for no per-token
          scoring at all.  It receives ``(step_id, latest_per_layer)`` once per
          forward through :meth:`fire_step_sink` (or
          :meth:`ingest_persistent`), which the decode loop calls after the
          model forward returns.

        ``tail_layers`` optionally restricts the deep ring to the layers a
        final readout aggregate can consume; the rest stay length-1
        latest-slice buffers and are omitted from :meth:`tail_slice_at`.

        The highest hooked layer is recorded as the per-forward counter
        trigger; it fires last in a forward, so by then every hooked layer
        holds this step's value.
        """
        self._incremental = True
        self._tail_depth = max(1, int(depth))
        self._tail_layers = (
            frozenset(int(layer) for layer in tail_layers)
            if tail_layers is not None
            else None
        )
        self._step_sink = step_sink
        self._max_layer = max(self._per_layer) if self._per_layer else None

    def set_incremental(
        self, step_sink: Callable[[int, dict[int, torch.Tensor]], None],
    ) -> None:
        """Length-1 buffers + per-token scoring (the INCREMENTAL mode)."""
        self.set_retention(depth=1, step_sink=step_sink)

    def set_aggregate_tail(self, depth: int) -> None:
        """Bounded tail ring, no per-token scoring (the AGGREGATE_ONLY mode)."""
        self.set_retention(depth=depth)

    def set_tail_with_sink(
        self,
        depth: int,
        step_sink: Callable[[int, dict[int, torch.Tensor]], None],
        *,
        tail_layers: set[int] | frozenset[int] | None = None,
    ) -> None:
        """Tail ring PLUS a per-token sink (GATING_SUBSET / LEAN_INCREMENTAL).

        Both of those modes score a *partial* reading per token (a gated
        subset's scalars, or ``coords_only`` rows) and still pool the FULL
        roster once at finalize from the retained ring.
        """
        self.set_retention(
            depth=depth, step_sink=step_sink, tail_layers=tail_layers,
        )

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
        in-hook path runs.  :meth:`set_retention` applies identically
        afterward.
        """
        self._per_layer = {idx: [] for idx in layer_indices}
        self._handles = []
        self._incremental = False
        self._step_sink = None
        self._max_layer = None
        self._tail_depth = 1
        self._tail_layers = None
        self._forward_count = 0
        self._persistent_buffers = {
            idx: buffers[idx] for idx in layer_indices if idx in buffers
        }

    def ingest_persistent(self, step_id: int) -> None:
        """Post-forward accumulation from the persistent buffers (compiled path).

        The compile-clean mirror of the in-hook accumulation: for each captured
        layer, append / overwrite / ring its latest persistent slice exactly as
        :meth:`attach`'s hook would, advance the forward counter, then fire the
        step sink once — the same post-forward scoring point
        :meth:`fire_step_sink` is on the transient path.  Wired as the
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
                keep_deep_tail = (
                    self._tail_depth > 1
                    and (
                        self._tail_layers is None
                        or idx in self._tail_layers
                    )
                )
                if not keep_deep_tail:
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
            self._step_sink(step_id, self.latest_per_layer())

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []
        # Drop the persistent-buffer source so a later transient capture (or a
        # no-probe gen) can't read stale slices through ``ingest_persistent``.
        self._persistent_buffers = {}
        self._prompt_positions = ()
        self._prompt_per_layer = {}

    def clear(self) -> None:
        self._per_layer = {}
        self._batch_per_layer = {}
        self._handles = []
        self._incremental = False
        self._step_sink = None
        self._max_layer = None
        self._tail_depth = 1
        self._tail_layers = None
        self._forward_count = 0
        self._batch_forward_count = 0
        self._persistent_buffers = {}
        self._prompt_positions = ()
        self._prompt_per_layer = {}

    def set_prompt_positions(self, positions: list[int]) -> None:
        """Retain selected rows from the first multi-token prefill forward.

        Positions are local to the ``input_ids`` passed into that forward.  A
        caller using a KV prefix therefore subtracts the cached prefix length
        before arming the capture.  Persistent latest-slice buffers cannot
        provide prompt rows; the session forces this mode onto transient hooks.
        """
        normalized = tuple(sorted({int(pos) for pos in positions if pos >= 0}))
        self._prompt_positions = normalized
        self._prompt_per_layer = {}

    def prompt_stacked(self) -> dict[int, torch.Tensor]:
        """Return selected prefill rows as ``{layer: [positions, D]}``."""
        return dict(self._prompt_per_layer)

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
            if self._tail_layers is not None and idx not in self._tail_layers:
                continue
            start = F - len(bucket)            # forward index of bucket[0]
            pos = forward_index - start
            pos = max(0, min(pos, len(bucket) - 1))
            out[idx] = bucket[pos]
        return out

    def batch_tail_slice_at(
        self,
        row_index: int,
        forward_index: int,
    ) -> dict[int, torch.Tensor]:
        """Per-layer ``[D]`` slice for one row from the batched tail ring."""
        out: dict[int, torch.Tensor] = {}
        F = self._batch_forward_count
        for idx, bucket in self._batch_per_layer.items():
            if not bucket:
                continue
            start = F - len(bucket)
            pos = forward_index - start
            pos = max(0, min(pos, len(bucket) - 1))
            batch_slice = bucket[pos]
            if row_index < 0 or row_index >= int(batch_slice.shape[0]):
                continue
            out[idx] = batch_slice[row_index]
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

    def per_layer_buckets(self) -> dict[int, list[torch.Tensor]]:
        """The raw per-layer capture buckets (``{layer: [slice, …]}``).

        The public accessor for the streaming-tap read that builds the latest
        per-layer ``[D]`` dict from each non-empty bucket's ``[-1]``.  A plain
        attribute return — **no per-token cost** (it is read once per token on the
        WS path) — the caller does the ``[-1]`` selection so a length-1 / tail-ring
        bucket reads identically.  (:meth:`latest_per_layer` is the packaged form;
        this exposes the buckets for callers that filter on emptiness themselves.)
        """
        return self._per_layer

    def is_transient(self) -> bool:
        """True iff transient per-gen forward hooks are registered (vs persistent).

        The compiled-clean routing gate: the captured graph was traced with only
        the always-on persistent capture/offset hooks present, so it stays valid
        exactly when no *transient* ``register_forward_hook`` was installed —
        ``attach_persistent`` (no transient hook) and the no-probe path both leave
        ``_handles`` empty.  ``attach`` registers one transient hook per layer.
        """
        return bool(self._handles)

    def fire_step_sink(self, step_id: int) -> None:
        """Run the per-token step sink once, **after** the model forward.

        Invoked by ``generate_steered`` post-``model()`` rather than from inside
        the capture hook at the max probe layer.  Scoring here keeps the
        device→host sync the sink's score read incurs out of the *middle* of the
        forward pass, so the remaining transformer layers + LM head stay queued
        on the device instead of draining the pipeline at the highest hooked
        layer.  No-op when no sink is installed (aggregate-only / full-retention /
        no-probe captures).  Reads the latest per-layer slice the forward just
        stored — valid until the next forward overwrites the length-1 buffer (or
        rotates the tail ring).  ``step_id`` is the loop's forward index,
        forwarded to the sink for step-keyed memo priming.
        """
        if self._step_sink is not None:
            self._step_sink(step_id, self.latest_per_layer())


class SteeringHook:
    """Per-layer steering state: zero or more subspace / manifold groups.

    Every steering term — vectors, poles, ``~``/``|`` projections, ``!``
    ablations, affine and curved ``%`` — lowers to a per-layer
    :func:`subspace_inject` group: the dispatch-synthesized merged affine
    subspace, plus zero or more mutually-orthogonal curved manifolds.

    :meth:`hook_fn` dispatches on the shape :meth:`recompose` armed, cheapest
    first, and the first three consult no :class:`TriggerContext` at all:

    1. :attr:`_const_single` — one always-active pure-push affine group: a
       single in-place ``hidden.add_(c)``.
    2. :attr:`_single_affine_lowrank` — the same group with an ``!`` ablation:
       the fixed push plus a projection restricted to the nonzero-κ rows.
    3. :attr:`_single_affine_fast` — the general single-affine fallback: one
       ``subspace_inject`` + ``copy_``, no group loop, no foot state.
    4. the general path — multiple groups, any curved manifold, or a gated /
       phased trigger: read ``_ctx``, skip inactive groups, thread the
       per-token foot.

    Cases 1–3 are a fixed tensor-op sequence, identical on every decode step,
    which is what makes a steered generation StaticCache / ``torch.compile``
    eligible (:meth:`SteeringManager.static_steerable`).  Case 4 keeps per-step
    triggers and probe gates dynamic and forces the eager DynamicCache path.
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
        # Model-dtype low-rank fast path for the single-affine mixed
        # push+ablation case.  The full ``subspace_inject`` fallback casts the
        # entire residual stream to fp32 and copies a full-width result back.
        # For an affine subspace the update is exactly low-rank.  Split its
        # fixed push from its activation-dependent ablation at recompose time:
        # ``h += c_push - along·(κₐ⊙(hBₐᵀ-μBₐᵀ))Bₐ``.  ``Bₐ`` contains
        # only nonzero-κ rows, so a push + one ablation (the usual mixed
        # expression) projects one axis instead of the entire merged span.  The
        # rank-1 hot path below further lowers that projection to an elementwise
        # dot + axpy, which is about 3x cheaper than two tiny MPS matmuls.
        self._single_affine_lowrank: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
            torch.Tensor, float,
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
                cast(torch.Tensor, kappa).to(device=device, dtype=torch.float32)
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
            self._single_affine_lowrank = (
                self._mixed_affine_lowrank(
                    _g[1], _g[3], _g[5], _g[7],
                    mean_proj=self._single_affine_fast[6],
                    device=device,
                    dtype=dtype,
                )
                if self._const_single is None else None
            )
        else:
            self._const_single = None
            self._single_affine_lowrank = None
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

        Returns :func:`_affine_push_offset` — ``c = (along·target) @ basis``,
        a ``(D,)`` per-layer offset — when the per-axis collapse mask ``kappa``
        is all zero (a push-only term: vectors, poles, ``~``/``|`` projections,
        affine ``%``, and merges of them), the case where the affine injection
        is exactly ``h_new = h + c`` (see :meth:`recompose`).  ``None`` for a
        mixed push+ablate group (any ``κ ≠ 0``), whose injection
        ``p_new − q = along·(target − κ·q)`` depends on ``q`` (hence ``h``) and
        so must keep the full kernel.

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
            kappa_tensor = cast(torch.Tensor, kappa)
            if bool(kappa_tensor.any()):
                return None
        elif float(kappa) != 0.0:
            return None
        c = _affine_push_offset(sub, target, along, device=device)
        # The constant-add path carries no ``norm_cap`` (``‖h_new‖ ≤ 3·‖h‖``);
        # that guard rides the curved path, where the RBF can extrapolate
        # off-domain.  A flat affine fit has no RBF and ``clamp_position`` keeps
        # its ``p_new`` in-box, so a bounded offset ``c`` added to a large-norm
        # residual-stream ``h`` cannot push ``‖h + c‖`` past ``3·‖h‖``.  Its
        # absence is what makes this path one kernel with no norm reductions.
        return c.to(dtype) if dtype is not None else c

    @staticmethod
    def _mixed_affine_lowrank(
        sub: LayerSubspace,
        target: torch.Tensor,
        along: float,
        kappa: "float | torch.Tensor",
        *,
        mean_proj: torch.Tensor,
        device: torch.device,
        dtype: "torch.dtype | None",
    ) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float] | None":
        """Precompute the fixed push and compact ablation-only projection.

        The exact affine update is ``along·target@B -
        along·(κ⊙q)@B``.  Its first term is constant and rows where
        ``κ == 0`` never contribute to the second, so carrying the complete
        merged rank through both hot-path matmuls is avoidable duplication.
        """
        if dtype is None:
            return None
        if isinstance(kappa, torch.Tensor):
            kappa_tensor = cast(torch.Tensor, kappa)
            if not bool(kappa_tensor.any()):
                return None
            kappa_f32 = kappa_tensor.to(device=device, dtype=torch.float32)
        else:
            if float(kappa) == 0.0:
                return None
            kappa_f32 = torch.full_like(
                target, float(kappa), device=device, dtype=torch.float32,
            )
        basis_f32 = sub.basis.to(device=device, dtype=torch.float32)  # (R, D)
        target_f32 = target.to(device=device, dtype=torch.float32)    # (R,)
        const = ((float(along) * target_f32) @ basis_f32).to(dtype)   # (D,)
        active = kappa_f32 != 0
        ablate_basis = basis_f32[active].to(dtype).contiguous()       # (Ra, D)
        ablate_basis_t = ablate_basis.T.contiguous()                  # (D, Ra)
        ablate_kappa = kappa_f32[active].to(dtype)                    # (Ra,)
        ablate_mean_proj = mean_proj.to(device=device, dtype=torch.float32)[
            active
        ].to(dtype)                                                    # (Ra,)
        return (
            ablate_basis, ablate_basis_t, ablate_kappa,
            ablate_mean_proj, const, float(along),
        )

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
        lowrank = self._single_affine_lowrank
        if lowrank is not None:
            hidden = output if isinstance(output, torch.Tensor) else output[0]
            basis, basis_t, kappa, mean_proj, const, along = lowrank
            if hidden.dtype != basis.dtype:
                # ``recompose`` casts to the dtype ``apply_to_model`` threaded in
                # (``next(model.parameters()).dtype``), which a wrapper model can
                # disagree with at the block output — a multimodal / mixed-
                # precision stack whose language blocks emit a narrower dtype
                # than the wrapper's first parameter.  Recast once and **store
                # it back**, so the hot path pays the five allocations on the
                # first fire after a recompose and none thereafter (the hot-path
                # rule is zero steady-state allocation; a per-fire recast would
                # break it on exactly the models this branch protects).
                basis = basis.to(hidden.dtype)
                basis_t = basis.T.contiguous()
                kappa = kappa.to(hidden.dtype)
                mean_proj = mean_proj.to(hidden.dtype)
                const = const.to(hidden.dtype)
                self._single_affine_lowrank = (
                    basis, basis_t, kappa, mean_proj, const, along,
                )
            if basis.shape[0] == 1:
                # MPS dispatch dominates a rank-1 GEMM.  Express the same
                # projection as a dot + axpy so Metal can use elementwise
                # kernels and avoid allocating the full merged-rank ``q``.
                row = basis[0]
                q = (hidden * row).sum(dim=-1, keepdim=True) - mean_proj[0]
                delta = const - (along * kappa[0]) * q * row
            else:
                q = hidden @ basis_t - mean_proj
                delta = const - (along * (kappa * q)) @ basis
            hidden.add_(delta)
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

        Each group runs :func:`subspace_inject`, the one along/onto injection.
        The two per-layer coefficients are already share-weighted at
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

        The merged affine subspace and every curved manifold are both groups
        here, dispatched by the kernel's ``is_affine`` branch (analytic slide
        vs foot-following GN).
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


# --- steering gains ----------------------------------------------------------
# Three constants, one per operation/path.  They all ride the same per-layer
# share weighting, whose contract lives once in ``_normalize_shares_mean1``
# below — read that first; each comment here covers only its own constant.
#
#   affine ``along``   ``eff_along_L = share_L · _SUBSPACE_GAIN``
#   curved ``along``   ``eff_along_L = along · share_L · _MANIFOLD_ALONG_GAIN``
#                      (periodic ``BoxDomain``: unweighted + clamped to [0, 1])
#   ``onto``           ``eff_onto_L  = clamp(onto · share_L · _MANIFOLD_ONTO_GAIN, 0, 1)``

# **Affine ``along`` gain** (prototype).  The injection *translates* the
# in-subspace foot by a fixed offset toward the target rather than collapsing
# every token's foot onto it (see ``subspace_inject`` /
# ``ManifoldDomain.translate_foot``): the fixed offset preserves the per-token
# in-subspace spread, which the kernel ablation showed keeps strong steer
# coherent (collapse → looping degeneration).  A typical layer gets
# ``eff_along ≈ _SUBSPACE_GAIN``.
#
# ``synthesize_subspace`` emits a *whitened-unit* affine target
# (``‖target@basis‖_M = 1`` per push term, magnitude carried by α), so the avg
# per-layer whitened push is simply ``GAIN·α`` — the same for every target.
# That is what makes one scalar gain calibratable at all: steering by the
# raw-Euclidean node distance baked in each node's distance from neutral
# (caveman ~17, formal ~0.3, a ~100× spread across targets).
#
# Live-calibrated on a gemma-4-12b α-sweep (thinking off; ``formal.casual%formal``,
# ``personas%caveman``, ``personas%hacker``).  In **effective gain ``E = GAIN·α``**:
# the dead→expressive transition is ``E ≈ 7-8`` for every target, but the coherence
# *ceiling* (token-soup looping past it) varies ~2× by target — hacker shatters at
# ``E ≈ 12``, caveman at ``E ≈ 17``, formal still coherent past ``E ≈ 22``.  There
# is **no single E** that makes caveman full AND keeps hacker coherent; this is the
# §10 per-persona coherence variance, which a scalar gain cannot unify (it is a
# steering-access property of the fit, not a geometric scale the whitened
# normalization could remove).
#
# So the calibration is **coherence-first**: put the recommended ``α ≈ 0.5`` at
# ``E ≈ 8`` — the strongest setting where *every* target (including the fragile
# hacker) is clearly register-shifted yet coherent — giving ``GAIN = 16``.  ``α ≈
# 1.0`` then lands at ``E ≈ 16`` (the documented "strong / over-steered" zone:
# robust concepts still coherent, hard personas break — dial α down per target).
# ``0.5`` of a tight concept and ``0.5`` of a far persona both land in that band.
# (Tune up toward ``E ≈ 12`` — α ≈ 0.75 — for fuller persona expression where the
# target tolerates it.)
_SUBSPACE_GAIN = 16.0

# **Curved ``along`` gain** — different in kind from the affine gain above.  The
# curved target is the node's *raw domain coordinates* and ``subspace_inject``
# translates the foot by ``eff_along·(target − origin)``
# (``domain.translate_foot``), so ``eff_along`` is a **fraction of the way to the
# node** (``1.0`` lands on it), not a free magnitude.  ``norm_cap = 3·‖h‖``
# bounds the off-domain RBF extrapolation, so a curved fit doesn't detonate the
# way the affine path would — instead, *past* ``eff_along ≈ 1`` the foot keeps
# translating past the node.
#
# Live-calibrated on a clean, stateless gemma-4-12b ``months_loop%january``
# α-sweep: ``along=1.0`` at gain ``4`` (``eff_along ≈ 4``) lands the vivid
# coherent winter sweet spot ("skeletal trees heavy with frost", consistent
# across seeds); gain ``2`` is milder but clearly cold; ``1.5`` is too weak (only
# a "cool prickle").
#
# Periodic ``BoxDomain`` fits are a distinct case: the runtime drops
# share-weighting and clamps ``eff_along = along · gain`` to ``[0, 1]`` uniformly
# across layers, because share ∈ [0.19, 1.47] × gain 4 sends many layers past 1
# on a ring and each would then wrap to a different node, scattering the signal.
# ``along = 1`` therefore lands every layer on the target node.  Non-periodic
# curved fits keep the share-weighted, unclamped translation.
_MANIFOLD_ALONG_GAIN = 4.0

# **``onto`` gain** — the off-surface collapse only, on the curved path.  With no
# σ-field the kernel scales the off-surface residual by ``(1 − eff_onto)``; on a
# fuzzy σ-field fit it instead shrinks the residual norm toward the local tube
# thickness.  That residual carries the per-token content variation, so combined
# with a directional ``along`` push too much ``onto`` erases the spread and
# degenerates into looping — the same failure the translate-not-collapse
# ``along`` design avoids, reintroduced by over-shrinking the residual that held
# the spread.
#
# Calibrated on the gemma-4-12b ``emotions%dominant`` onto sweep (the affect
# manifold under a curved fit, along fixed at 0.3): at ``1.0`` even ``onto = 0.5``
# fragmented and ``onto = 1.0`` collapsed to ``!!!``; ``0.5`` puts the recommended
# ``onto ≈ 0.5`` at a clean sweet spot and keeps ``onto = 1.0`` a
# strong-but-coherent ceiling, while below ``~0.3`` the [0, 1] knob saturates into
# no dynamic range (``onto = 0.5 ≈ 1.0``).  A [0, 1] dial whose top emits garbage
# is a bad dial, so ``onto = 1.0`` is deliberately the coherent maximum, not the
# over-steer edge.
_MANIFOLD_ONTO_GAIN = 0.5

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

    The share weighting every steering gain rides — the one place its contract
    lives.  A raw share (``synth.share`` = the whitened push displacement
    ``‖Δ_L‖_M``, or ``_manifold_layer_shares`` = the baked
    ``mahalanobis_share``) says how much signal a layer carries; mean-1
    normalization turns that into a *relative* weight, so
    ``eff_L = share_L · gain`` reads as a clean per-layer slide fraction ≈
    ``gain`` on a typical layer — above it on the high-signal layers, below it
    on the flat ones.

    The property that buys is **n_layers-invariance**: one covered layer and a
    30-layer fit both put ≈ ``gain`` of slide on each contributing layer, so a
    4-dim and a 16-dim fit reach comparable behavior at the same α and an
    ``A ⊂ B`` nested subspace steers its shared axis identically.  (Normalizing
    to sum 1 instead would divide the strength by the layer count; a per-fit
    lever correction would double-count the magnitude the whitened target
    already carries and blow up on low-rank fits.)

    ``along`` is **not** clamped afterwards: a high-signal layer is *meant* to
    overshoot past the target, and the bounded whitened-unit target (affine) or
    ``norm_cap = 3·‖h‖`` inside ``subspace_inject`` (curved) is what holds it.
    The two exceptions are documented at their gains — a periodic ``BoxDomain``
    curved fit drops this weighting entirely, and ``onto`` stays clamped
    ``[0, 1]`` (a residual-shrink fraction beyond 1 would overshoot through the
    zero-thickness wire or σ-tube).

    Inputs must be finite and strictly positive — exact current geometry, never
    a degenerate placeholder.
    """
    if not raw or any(not math.isfinite(s) or s <= 0.0 for s in raw.values()):
        raise ValueError("layer shares must be a nonempty finite positive mapping")
    n_layers = len(raw)
    total = sum(raw.values())
    return {L: s / total * n_layers for L, s in raw.items()}


def _manifold_layer_shares(manifold: Manifold) -> dict[int, float]:
    # A curved manifold's per-layer share is the whitened (Mahalanobis) weight
    # baked at fit time — the subspace-restricted analogue of vector steering's
    # ``‖d‖_M`` bake score (see ``LayerWhitener.subspace_gram`` /
    # ``ManifoldExtractionPipeline.fit``).  *Full* layer coverage is required:
    # the share is a cross-layer-normalized weight, so mixing whitened and
    # Euclidean scalars across layers would compare incommensurable metrics.
    # ``_normalize_shares_mean1`` then turns it into the relative per-layer
    # weight the gains ride (see its docstring).
    baked = manifold.mahalanobis_share
    missing = set(manifold.layers) - set(baked)
    extra = set(baked) - set(manifold.layers)
    if missing or extra:
        raise ValueError(
            f"manifold {manifold.name!r} has non-canonical Mahalanobis shares "
            f"(missing={sorted(missing)}, extra={sorted(extra)})"
        )
    layer_scores = {
        layer_idx: float(baked[layer_idx]) for layer_idx in manifold.layers
    }
    return _normalize_shares_mean1(layer_scores)


@dataclass(frozen=True)
class LoweredAffine:
    """One layer's merged affine steering term in runtime form.

    The output of :func:`_lower_affine_subspaces` — a
    :class:`~saklas.core.manifold.SynthesizedSubspace` layer reduced to exactly
    what the injection needs: the (possibly curved-orthogonalized) affine
    ``subspace``, its ``(R,)`` push ``target`` (every active push term's
    coeff-scaled fragment already composed in by ``synthesize_subspace``), the
    per-layer slide budget ``eff_along = share_L · _SUBSPACE_GAIN`` with
    ``share_L`` mean-1 normalized, and the gain-compensated per-axis collapse
    mask ``kappa`` (``requested_κ / eff_along``, so the kernel's ``along · κ``
    product is exactly the user's ablation coefficient).
    """

    subspace: LayerSubspace
    target: torch.Tensor
    eff_along: float
    kappa: torch.Tensor

    @property
    def is_pure_push(self) -> bool:
        """True when no axis ablates, i.e. the injection is a constant add.

        ``κ = 0`` on every axis ⇒ ``p_new − q = eff_along·target`` independent
        of ``h``, which is what both constant-add paths
        (:meth:`SteeringHook._pure_push_constant`, the compiled offset buffers)
        require.  Gain compensation divides by a strictly positive
        ``eff_along``, so this reads the same as testing the requested κ.
        """
        return not bool(self.kappa.any())


def _lower_affine_subspaces(
    synth: SynthesizedSubspace,
    *,
    curved_basis_by_layer: "dict[int, torch.Tensor] | None" = None,
) -> dict[int, LoweredAffine]:
    """Lower one synthesized affine subspace to its per-layer runtime terms.

    **The** affine lowering: mean-1 share normalization, optional
    orthogonalization against the curved manifolds sharing a layer,
    ``eff_along_L = share_L · _SUBSPACE_GAIN``, and the κ gain compensation.
    Both consumers go through here — :meth:`SteeringManager.apply_to_model`
    builds transient hook entries from the result and
    :meth:`SteeringManager.compute_static_offsets` contracts it into the
    persistent compiled offset buffers — so compiled and eager generations
    steer by construction, not by two implementations agreeing.

    ``curved_basis_by_layer`` supplies the stacked orthonormal rows of every
    curved manifold at a layer; the affine span is projected out of them
    (curved wins the shared directions, ARCHITECTURE §6) and the layer is
    dropped entirely when nothing is left to steer there.  ``None`` (the
    compiled path, which only lowers when there is no curved manifold at all)
    skips that step.

    Runs at compose time, once per steering push — never per token.
    """
    layer_set = list(synth.layers)
    if not layer_set:
        return {}
    # ``synth.share`` is the un-normalized whitened push displacement ``‖Δ_L‖_M``
    # per layer; ``_normalize_shares_mean1`` turns it into the relative weight
    # the gain multiplies (its docstring owns that contract).
    shares = _normalize_shares_mean1(
        {L: float(synth.share[L]) for L in layer_set}
    )
    lowered: dict[int, LoweredAffine] = {}
    for L in layer_set:
        sub_L = synth.layers[L]
        target = synth.target_coord[L].to(torch.float32)
        # Per-axis requested ablation coefficient (0 on push axes).
        # Orthogonalization runs on these user-space coefficients first, then
        # the single scalar division below preserves the resulting operator
        # exactly.
        requested_kappa = synth.kappa[L].to(torch.float32)
        curved = (
            curved_basis_by_layer.get(L)
            if curved_basis_by_layer is not None
            else None
        )
        if curved is not None:
            res = _orthogonalize_affine_against(
                sub_L, target, requested_kappa, curved,
            )
            if res is None:
                # The affine span lies entirely inside the curved span —
                # nothing left to steer at this layer.
                continue
            sub_L, target, requested_kappa = res
        # No lever / ``N`` and no ``[0, 1]`` clamp: the de-rogued whitened-unit
        # target carries the magnitude, so a high-share layer is *meant* to
        # overshoot past the target.
        eff_along = shares[L] * _SUBSPACE_GAIN
        # ``subspace_inject`` multiplies κ by ``along``; dividing by the affine
        # push gain here keeps ``0.15 !x`` at 15% rather than ``16×`` that.
        lowered[L] = LoweredAffine(
            subspace=sub_L,
            target=target,
            eff_along=eff_along,
            kappa=requested_kappa / eff_along,
        )
    return lowered


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
        # group) — where vectors, poles, ``~``/``|`` projections, ``!``
        # ablations, and affine ``%`` all land.  Each value is
        # ``{synth, trigger}``; ``apply_to_model`` lowers them to per-layer
        # ``subspace_inject`` entries alongside the curved manifolds.
        self.subspaces: dict[str, dict[str, Any]] = {}
        self.ctx: TriggerContext = TriggerContext()
        # Persistent compile-clean steering path (CUDA/MPS torch.compile). A single
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
        """True iff no **transient** steering hook is attached.

        Two states satisfy it: genuinely unsteered, and steering that lowered
        to the persistent compiled offset buffers (the composer calls
        :meth:`detach_transient_hooks` after
        :meth:`write_compiled_offsets`, so ``hooks`` is empty while the
        branchless ``add_(offset)`` hooks are actively steering).  Either way
        no ctx-consulting hook can fire, so StaticCache / ``torch.compile``
        graph capture is eligible — which is what this signal is for.

        It is therefore **not** an "is this generation unsteered" test.
        Callers that need that distinction read the session's
        ``_steering_uses_compiled_offsets`` alongside it.  A curved / gated /
        phased hook stays on the transient list and forces the eager
        DynamicCache path; the *static-affine* transient case is the separate
        :meth:`static_steerable` signal below.
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
        manifold: Manifold,
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
        in-subspace residual.  The off-*subspace* residual is always kept
        verbatim — that is what lets a vector and N orthogonal manifolds
        compose with zero cross-talk.
        """
        manifold.validate_runtime_geometry()
        resolved = manifold.resolve_position(position)
        domain = manifold.domain
        if len(resolved) != domain.intrinsic_dim:
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
        """Register a dispatch-synthesized merged affine subspace.

        ``synth`` (one per active trigger group) carries the per-layer affine
        :class:`LayerSubspace`, the ``along`` ``target_coord`` (every active
        push term's coeff-scaled pole already composed in), and the
        un-normalized per-layer budget ``share`` (``‖Δ_L‖_M``).

        :func:`_lower_affine_subspaces` reduces it to per-layer
        :class:`LoweredAffine` terms, which :meth:`apply_to_model` wraps as
        ``(subspace, CustomDomain(R_L), target_coord, origin=0, eff_along,
        onto=0)`` entries routed through the same :func:`subspace_inject` hot
        path as a curved manifold — the affine analytic shortcut slides the
        in-subspace component toward ``target_coord``.  ``onto = 0`` (the
        surface fills its span).
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
        # **Gain.**  Both curved coefficients are share-weighted by the manifold's
        # mean-1 per-layer share (``_normalize_shares_mean1`` owns that contract)
        # and scaled by their own constant — ``_MANIFOLD_ALONG_GAIN`` for
        # ``along``, ``_MANIFOLD_ONTO_GAIN`` for ``onto``.  ``along`` is left
        # un-clamped so a high-share layer overshoots past the target (the
        # ``norm_cap`` inside ``subspace_inject`` is the only bound); ``onto``
        # clamps to ``[0, 1]`` per layer (beyond 1 would overshoot through the
        # wire/tube).
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
            # ``eff_along`` is the fraction of the way to the node, so it must
            # stay near [0, ~2] or the RBF extrapolates off-domain (see
            # ``_MANIFOLD_ALONG_GAIN``).  ``onto`` clamps per layer.
            #
            # Periodic (loop) domains take the other branch: share-weighting is
            # dropped and ``eff_along`` clamps to [0, 1] uniformly, so no layer
            # wraps past the target node.  (Share ∈ [0.19, 1.47] × gain 4 sends
            # many layers past 1 on a ring, and each would then land on a
            # different node — scattering the signal.)  Non-periodic curved fits
            # keep the share-weighted, unclamped translation.
            domain = manifold.domain
            _is_periodic = isinstance(domain, BoxDomain) and any(
                ax.periodic for ax in domain.axes
            )
            if _is_periodic:
                eff_along = {
                    L: max(0.0, min(1.0, along * _MANIFOLD_ALONG_GAIN))
                    for L in manifold.layers
                }
            else:
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
            target_coord = domain.clamp_position(
                torch.tensor([float(c) for c in position], dtype=torch.float32)
            )
            mfld_origins = manifold.origin

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
                if layer_idx not in mfld_origins:
                    raise ValueError(
                        f"curved manifold {manifold.name!r} has no neutral "
                        f"origin for layer {layer_idx}"
                    )
                origin_coord = domain.clamp_position(
                    mfld_origins[layer_idx].reshape(-1).to(torch.float32)
                )
                manifold_by_layer.setdefault(layer_idx, []).append((
                    sub, domain, target_coord, origin_coord,
                    eff_along[layer_idx], eff_onto[layer_idx], 0.0, trigger,
                ))  # κ = 0: curved manifolds are push-only (pure translate)

        # Dispatch-synthesized merged affine subspaces.  Each ``synth`` is
        # already neutral-anchored with its ``along`` target composed from every
        # active push term's coeff-scaled pole, so all that remains is
        # :func:`_lower_affine_subspaces` (the shared share-normalization /
        # gain / κ-compensation / curved-orthogonalization step, also consumed
        # by :meth:`compute_static_offsets`) and wrapping each layer as a
        # ``CustomDomain(R_L)`` ``subspace_inject`` entry — the affine analytic
        # shortcut, no GN / RBF / foot solve.  The target carries the strength,
        # so there is no separate user-α multiply here; ``onto = 0`` (the
        # surface fills its span).
        for s in self.subspaces.values():
            synth: SynthesizedSubspace = s["synth"]
            sub_trigger: Trigger = s["trigger"]
            lowered = _lower_affine_subspaces(
                synth, curved_basis_by_layer=curved_basis_by_layer,
            )
            for L, low in lowered.items():
                r_l = low.subspace.rank
                # Affine origin is span-coord 0 (neutral → coord 0, §5); the
                # foot seed / cold-start is unused on the affine shortcut.
                sub_origin = torch.zeros(r_l, dtype=torch.float32)
                manifold_by_layer.setdefault(L, []).append((
                    low.subspace, CustomDomain(r_l), low.target, sub_origin,
                    low.eff_along, 0.0, low.kappa, sub_trigger,
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

    # -- Persistent compile-clean offset path (CUDA/MPS torch.compile) ---------

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
        a probe gate / phase trigger, or an ``!`` ablation).

        Compiled/eager parity is structural: the terms come from the same
        :func:`_lower_affine_subspaces` :meth:`apply_to_model` consumes, and the
        offset is contracted by the same :func:`_affine_push_offset` the
        transient :meth:`SteeringHook._pure_push_constant` fast path adds — so a
        change to the share normalization, the gain, or the κ semantics moves
        both paths together.  ``curved_basis_by_layer`` is omitted from the
        lowering because a curved manifold disqualifies this path outright.
        """
        if self.manifolds:
            return None  # a curved manifold isn't a constant add
        offsets: dict[int, torch.Tensor] = {}
        for s in self.subspaces.values():
            synth: SynthesizedSubspace = s["synth"]
            if s["trigger"] is not Trigger.BOTH:
                return None  # gated / phased — needs the ctx-consulting hook
            for L, low in _lower_affine_subspaces(synth).items():
                if not low.is_pure_push:
                    return None  # ablation: injection depends on h, not a const
                c = _affine_push_offset(
                    low.subspace, low.target, low.eff_along,
                )
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
