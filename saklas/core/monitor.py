from __future__ import annotations

import re
from collections import deque
from typing import TYPE_CHECKING, Any, cast

import torch

from saklas.core.manifold import BoxDomain, invert_parameterization
from saklas.core.monitor_attach import (
    DEFAULT_NEAREST_TOP_N,
    NEUTRAL_LABEL,
    _FRACTION_EPSILON,
    _MIN_SHARE_WEIGHT,
    _woodbury_apply,
    AttachedManifoldProbe,  # re-exported: tests import from saklas.core.monitor
    _build_whitened_factors,
    _attach_manifold_probe,
    _layer_geometry,        # re-exported: tests import from saklas.core.monitor
)
from saklas.core.results import ProbeReading

if TYPE_CHECKING:
    from saklas.core.manifold import Manifold

_MAX_HISTORY = 8

_EMPTY_STATS = {"count": 0, "sum": 0.0, "sum_sq": 0.0,
                "min": float("inf"), "max": float("-inf")}


class Monitor:
    """Reads probes as whitened subspace coordinates — flat and curved alike.

    **One** read shape, two execution paths.  Each probe is a
    :class:`~saklas.core.manifold.Manifold`, and every read — live per token
    (gate / stream) and the end-of-generation aggregate — produces one full
    :class:`ProbeReading` (``coords`` + ``fraction`` + ``nearest`` +
    ``residual`` plus their ``*_per_layer`` traces), with no flat/curved field
    asymmetry: a flat (affine) probe recovers ``coords`` through the affine
    reduced→domain map (off-surface ``residual`` identically 0), a curved probe
    through the iterative :func:`invert_parameterization` foot solve (real
    off-surface ``residual``).  The full per-token information (nearest, curved
    coords, residual, per-layer) is the research-tool priority.

    Execution is still no-redundancy on the hot path.  The whole *flat* roster
    is scored together per layer in :meth:`_score_flat_batched` — one ``Σ⁻¹h``
    Woodbury apply + stacked / block-diagonal matmuls + a **single** host
    transfer for the roster — instead of re-running the apply and the R-dim
    solve per probe per layer.  *Curved* probes keep the per-probe foot solve
    (:meth:`_score_probe_full`), warm-started across decode tokens from the
    previous token's foot (:meth:`enable_curved_warm`) on the sequential live
    path.  And when nothing consumes a per-token reading (no gate, no stream),
    the session skips per-token scoring entirely and pools the aggregate once
    at finalize — see ``SaklasSession._begin_capture`` /
    :meth:`score_aggregate`.

    Coordinates are **domain-frame**: each layer's reduced activation
    coordinate is inverted into the shared domain *before* EV-averaging,
    because raw per-layer coords live in per-layer ``‖δ_L‖`` units and don't
    average coherently.  At rank-1 this is the pole-normalized coordinate:
    ``1.0`` sits at the positive node, signed, unbounded past it.

    The Mahalanobis whitener is mandatory: every probed layer must be
    covered, built per probe at attach (:func:`_build_whitened_factors`) —
    a missing / non-covering whitener raises :class:`WhitenerError`, there is
    no Euclidean readout (on real LMs the Euclidean metric is rogue-dominated,
    a wrong answer not a degraded one).

    TUI-facing scalar helpers (``get_stats`` / ``get_sparkline`` /
    ``get_current_and_previous``) report coordinate **axis 0** so the
    untouched trait panel keeps working; the full per-axis + per-layer data
    flows through the :class:`ProbeReading` surface.
    """

    @staticmethod
    def _empty_stats() -> dict[str, Any]:
        return dict(_EMPTY_STATS)

    def __init__(self, probe_manifolds: dict[str, "Manifold"] | None = None,
                 layer_means: dict[int, torch.Tensor] | None = None,
                 whitener: Any = None):
        """
        probe_manifolds: maps probe name -> flat :class:`Manifold` (rank-1
            concept axis or rank-R discover fit).  Ad-hoc baked directions
            are wrapped into a 1-pole affine manifold by the session before
            registration (``fold_directions_to_subspace``).
        layer_means: maps layer_idx -> neutral mean activation (kept for
            init-signature parity with :class:`Monitor`; the
            coordinate readout centers on each fit's own ``LayerSubspace.mean``,
            not the global layer mean, so this is not consulted on the hot
            path — only exposed via the ``layer_means`` property).
        whitener: the :class:`~saklas.core.mahalanobis.LayerWhitener`;
            mandatory at scoring time (covers every probed layer or raise).
        """
        self._probes: dict[str, AttachedManifoldProbe] = {}
        self._layer_means: dict[int, torch.Tensor] = dict(layer_means) if layer_means else {}
        self._whitener: Any = whitener
        self._whitener_factor_cache: dict[
            tuple[int, str, torch.dtype],
            tuple[torch.Tensor, torch.Tensor, float],
        ] = {}

        # Cross-probe batched flat-read cache (the hot-path no-redundancy
        # primitive).  For every *flat* (affine) probe the per-layer geometry
        # is identical algebra (``_layer_geometry``) over a shared activation,
        # so the whole flat roster is scored per layer with one ``Σ⁻¹h`` Woodbury
        # apply + a handful of stacked / block-diagonal matmuls instead of
        # re-running the apply and the R-dim solve once per probe per layer (the
        # 17×-redundant cost that dominated decode once a probe roster was
        # attached).  Curved probes keep the per-probe foot solve.  Built lazily
        # per (device, roster, whitener); invalidated on any of the three
        # changing.  ``_flat_layer_cache[L]`` holds the stacked factors for the
        # flat probes covering layer ``L``; ``_flat_keys`` / ``_curved_keys``
        # split the roster.
        self._flat_layer_cache: dict[int, dict[str, Any]] = {}
        self._flat_global: dict[str, Any] = {}
        self._flat_keys: tuple[str, ...] = ()
        self._curved_keys: tuple[str, ...] = ()
        self._flat_cache_device: torch.device | None = None
        # Dirty flag set by ``_invalidate_flat_cache`` on every roster / whitener
        # mutation (FIX F5).  ``_ensure_flat_cache`` checks it instead of
        # rebuilding ``tuple(self._probes.keys())`` + comparing it every token —
        # the cache only changes when a mutation flips this, so the per-token
        # guard is a bool + a device compare, not a per-token tuple alloc.
        self._flat_cache_dirty: bool = True

        # Per-coordinate history + summary stats.  ``history`` holds the
        # per-generation aggregate coordinate tuple; ``_stats`` is axis-0
        # scalar stats (TUI compat) plus the per-axis accumulators the
        # session reads for the vectorized ``ProbeReadings``.
        self.history: dict[str, deque[tuple[float, ...]]] = {
            n: deque(maxlen=_MAX_HISTORY) for n in self._probes
        }
        self._stats: dict[str, dict[str, Any]] = {
            n: self._empty_stats() for n in self._probes
        }

        # Aggregate path sets _pending_aggregate; per-token path sets _pending_per_token.
        # has_pending_data() returns aggregate readiness — the TUI uses it to refresh
        # trait readings after a measure() call.
        self._pending_aggregate = False
        self._pending_per_token = False

        # Live running mean (per coordinate axis) during streaming generation.
        self._live_values: dict[str, list[float]] = {}
        self._live_count: int = 0
        self._live_pending: bool = False

        # Curved-probe per-token foot-follow (read-side analogue of the steering
        # foot-follower).  When ``_curved_warm`` is set — only the sequential
        # incremental live-scoring path enables it, via ``enable_curved_warm`` —
        # ``_score_probe_full`` seeds each curved probe's ``invert_parameterization``
        # from the previous token's foot (keyed ``(probe_name, layer_idx)``) and
        # stashes the refined foot for the next token, cutting the cold
        # 12-iter/3-restart solve to ``warm_iter`` steps over two restarts.  Off
        # by default and for one-off reads (``score_aggregate`` / ``score_hidden``
        # / the non-incremental stream), so those stay bit-for-bit the cold path.
        self._curved_feet: dict[tuple[str, int], torch.Tensor] = {}
        self._curved_warm: bool = False

        # Probe-inspector live trail: when set, every full read also stamps each
        # probe's per-layer whitened query coords (``cdist_query``) onto the
        # ``ProbeReading.subspace_coords_per_layer`` so the dashboard can plot the
        # current hidden state (+ a fading trajectory trail) in the same whitened
        # frame as the geometry endpoint's ``node_white``.  Off by default — the
        # session flips it on only while a client requests it
        # (``persist_subspace_coords``), so the default hot path never pays the
        # post-pass.
        self._emit_subspace_coords: bool = False

        if probe_manifolds:
            for _name, _m in probe_manifolds.items():
                self.add_probe(_name, _m)

    @property
    def probe_names(self) -> list[str]:
        """Probe names in insertion order."""
        return list(self._probes.keys())

    @property
    def manifolds(self) -> dict[str, "Manifold"]:
        """Attached probe manifolds: name -> flat :class:`Manifold`."""
        return {n: p.manifold for n, p in self._probes.items()}

    def probe_layers(self, names: set[str] | None = None) -> set[int]:
        """Union of fit-layer indices across attached probes.

        The capture-widening signal the session uses to retain every layer
        a probe reads (peer of :meth:`Monitor.attached_layers`).  ``names``
        narrows the union for gate-only control calls that do not need a final
        full-roster probe aggregate.
        """
        out: set[int] = set()
        probes = (
            (p for n, p in self._probes.items() if n in names)
            if names is not None else self._probes.values()
        )
        for probe in probes:
            out.update(probe.manifold.layers.keys())
        return out

    @property
    def layer_means(self) -> dict[int, torch.Tensor]:
        return self._layer_means

    @layer_means.setter
    def layer_means(self, value: dict[int, torch.Tensor]) -> None:
        value_in: Any = value
        if value_in is not None and not isinstance(value_in, dict):
            raise TypeError(f"layer_means must be a dict, got {type(value).__name__}")
        self._layer_means = dict(value) if value else {}

    @property
    def whitener(self) -> Any:
        """The wired :class:`LayerWhitener` (required for reads)."""
        return self._whitener

    def set_whitener(self, whitener: Any) -> None:
        """Wire (or clear) the Mahalanobis whitener and invalidate the cache.

        Idempotent on the same instance — re-wiring the identical whitener
        is a no-op.  Any change flushes the shared Woodbury-factor cache and
        rebuilds each attached probe's per-layer whitened factors against the new
        covariance (manifold reads are Mahalanobis-only, so a whitener that
        doesn't cover an attached probe's fit layers raises here).
        """
        if whitener is self._whitener:
            return
        self._whitener = whitener
        self._whitener_factor_cache.clear()
        self._invalidate_flat_cache()
        for probe in self._probes.values():
            probe.whitened = _build_whitened_factors(
                whitener, probe, factor_cache=self._whitener_factor_cache,
            )

    def enable_curved_warm(self, flag: bool) -> None:
        """Enable/disable curved-probe per-token foot warm-starting.

        The session flips this on only for the sequential incremental
        live-scoring path (one ``score_single_token`` per decode step) and off
        otherwise, so one-off / out-of-order reads (``score_aggregate``,
        ``score_hidden``, the non-incremental per-token stream) stay on the cold
        ``invert_parameterization`` path and remain bit-for-bit reproducible.
        """
        self._curved_warm = bool(flag)

    def set_subspace_coords(self, flag: bool) -> None:
        """Enable/disable the probe-inspector live whitened-coords post-pass.

        Flipped on by the session for the duration of a generation when a client
        sets ``persist_subspace_coords`` (the dashboard inspector being open), and
        reset off at teardown.  Gates the only added work in :meth:`_score_full`,
        so with it off the read path is byte-for-byte the legacy path.
        """
        self._emit_subspace_coords = bool(flag)

    def reset_curved_feet(self) -> None:
        """Cold-start every curved probe's per-token foot (call per generation).

        Peer of ``SteeringManager.reset_manifold_feet`` — the carried foot is
        per-generation state, so a new run must re-seed from the nearest node
        rather than inherit the previous run's final foot.
        """
        self._curved_feet.clear()

    def _invalidate_flat_cache(self) -> None:
        """Drop the cross-probe batched flat-read cache.

        Called whenever the roster, whitener, or scoring device changes — the
        stacked / block-diagonal factors are rebuilt lazily on the next score.
        """
        self._flat_layer_cache = {}
        self._flat_global = {}
        self._flat_keys = ()
        self._curved_keys = ()
        self._flat_cache_device = None
        self._flat_cache_dirty = True

    def _score_probe_full(
        self,
        probe: "AttachedManifoldProbe",
        hidden_per_layer: dict[int, torch.Tensor],
        sih_cache: dict[int, torch.Tensor] | None = None,
        *,
        coords_only: bool = False,
    ) -> ProbeReading:
        """Full per-probe reading for one state — flat or curved, all fields.

        ``coords_only`` (FIX F2): when a live consumer reads only the axis-0
        coord / fraction, skip the per-token nearest-distance ranking, the soft
        assignment, the off-surface residual / tube membership, and the per-layer
        trace transfer — return a reading carrying just ``coords`` + ``fraction``
        (``nearest``/``assignment`` empty, ``residual`` 0, ``membership`` 1.0).
        The full aggregate is re-scored once at finalize, so nothing is lost.

        Loops the probe's shared fit layers through :func:`_layer_geometry`
        and **share**-weights across them (the layer carrying the most
        Mahalanobis steering budget is also the most reliable to read from).
        A flat (affine) probe recovers ``coords`` through the affine
        reduced→domain map (off-surface ``residual`` 0); a curved probe
        through the :func:`invert_parameterization` foot solve (real
        off-surface ``residual``).  Returns the cross-layer share-weighted
        ``coords`` / ``fraction`` / ``nearest`` / ``residual`` plus their
        per-layer traces.  This is the single geometry primitive every read
        entry point (live per token, aggregate) shares, so the aggregate at a
        token index is bit-identical to the live read at that token.

        ``sih_cache`` (optional, keyed by layer) shares the probe-independent
        ``Σ⁻¹h`` apply across every curved probe in one ``_score_full`` pass; it
        is valid only for the current state, so callers pass a fresh dict per
        token / pool.
        """
        manifold = probe.manifold
        sw = probe.share_weights
        is_affine = probe.is_affine
        shared = [idx for idx in manifold.layers if idx in hidden_per_layer]
        if not shared:
            return ProbeReading(fraction=0.0, nearest=[], coords=())
        total_w = sum(sw.get(idx, 0.0) for idx in shared)
        if total_w <= _MIN_SHARE_WEIGHT:
            w_shared = {idx: 1.0 / len(shared) for idx in shared}
        else:
            w_shared = {idx: sw.get(idx, 0.0) / total_w for idx in shared}

        K = probe.node_values_reduced[shared[0]].shape[0]
        inject_neutral = probe.inject_neutral
        Kc = K + (1 if inject_neutral else 0)   # candidate count incl. neutral
        n_dim = manifold.domain.intrinsic_dim
        dist_acc_t: torch.Tensor | None = None
        # Per-layer terms + running share-weighted means are accumulated as
        # on-device tensors and pulled to the host in a *single* transfer per
        # probe at the end.  The earlier code called ``float(... .item())`` /
        # ``.tolist()`` per layer, which on MPS is one command-buffer stall per
        # layer per probe per token — the dominant decode-loop cost once a probe
        # roster is attached (~31 host syncs/probe × 17 probes ≈ 140 ms/token on
        # an M-series GPU).  Keeping everything on device until one batched
        # ``.cpu()`` preserves every field of the full reading (the per-token
        # info priority) at ~1 sync/probe.
        frac_terms: list[torch.Tensor] = []        # each (1,)
        resid_terms: list[torch.Tensor] = []       # each (1,)
        coord_terms: list[torch.Tensor] = []       # each (n_dim,)
        layer_order: list[int] = []
        frac_mean_t: torch.Tensor | None = None    # (1,)
        resid_mean_t: torch.Tensor | None = None   # (1,)
        coords_mean_t: torch.Tensor | None = None  # (n_dim,)
        mem_mean_t: torch.Tensor | None = None     # (1,) tube-fit membership

        for layer_idx in shared:
            sub = manifold.layers[layer_idx]
            h = hidden_per_layer[layer_idx].to(torch.float32)
            if h.ndim > 1:
                h = h.reshape(-1, h.shape[-1])[-1]
            wh = probe.whitened[layer_idx]
            frac_t, cdist_query, invert_query, _cdist_nodes = _layer_geometry(
                probe, layer_idx, h, sih_cache,
            )
            w = w_shared[layer_idx]

            if not coords_only:
                # Neutral competes as a virtual candidate at the ``K``-th row of
                # the precomputed ``node_white_aug`` (FIX F4 — appended once at
                # attach, not re-``cat``-ed per token): its whitened coord is
                # ``wh.neutral_white`` (0 for an affine fit, the baked origin for
                # a curved one), so the same cdist yields its distance with no
                # special-casing downstream.  Skipped entirely under ``coords_only``
                # (FIX F2) — nearest / assignment aren't read.
                dists = torch.linalg.vector_norm(
                    wh.node_white_aug - cdist_query.reshape(1, -1), dim=-1,
                )
                weighted_dists = dists * w
                dist_acc_t = (
                    weighted_dists
                    if dist_acc_t is None
                    else dist_acc_t + weighted_dists.to(dist_acc_t.device)
                )

            if is_affine:
                # Affine map: coords are exact, no off-subspace residual.
                pos_t = (
                    wh.coord_S @ invert_query + wh.coord_b
                    if wh.coord_S is not None and wh.coord_b is not None
                    else invert_query
                )
                resid_t = frac_t.new_zeros(1)
                mem_t = frac_t.new_ones(1)   # surface fills subspace ⇒ full fit
            else:
                # Curved probe: warm-start the nearest-point foot from the
                # previous token's foot when the session enabled it for this
                # (sequential) live-scoring pass; otherwise solve cold.  The foot
                # solve is needed for ``coords`` regardless, so it always runs;
                # only the residual / membership it feeds are ``coords_only``-gated.
                foot_key = (probe.name, layer_idx)
                warm = self._curved_feet.get(foot_key) if self._curved_warm else None
                pos, res = invert_parameterization(
                    sub, manifold.domain, invert_query.reshape(1, -1),
                    manifold.node_coords, warm_start=warm,
                )
                if self._curved_warm:
                    self._curved_feet[foot_key] = pos.detach()
                pos_t = pos.reshape(-1)
                if coords_only:
                    resid_t = frac_t.new_zeros(1)
                    mem_t = frac_t.new_ones(1)
                else:
                    par_norm = torch.linalg.vector_norm(invert_query)
                    res_flat = res.reshape(-1)[:1]
                    # norm_residual = res / ‖query‖, or 0 if the query is ~zero —
                    # kept as a tensor (no ``.item()``) via a branchless ``where``.
                    resid_t = torch.where(
                        par_norm < _FRACTION_EPSILON,
                        torch.zeros_like(res_flat),
                        res_flat / par_norm.clamp(min=_FRACTION_EPSILON),
                    )
                    # Tube-fit membership ``exp(−res²/2σ²)`` at the foot — both
                    # ``res`` and ``σ`` are raw reduced-space (un-whitened)
                    # distances, so the ratio is unit-consistent.  No σ-field ⇒
                    # no tube ⇒ full membership (1.0).
                    if sub.has_sigma:
                        sig_foot = sub.sigma_at(
                            manifold.domain.embed(pos.reshape(1, -1)),
                        ).reshape(1)
                        mem_t = torch.exp(
                            -(res_flat ** 2)
                            / (2.0 * (sig_foot ** 2).clamp(min=_FRACTION_EPSILON))
                        )
                    else:
                        mem_t = frac_t.new_ones(1)

            frac_t = frac_t.reshape(1)
            coord_t = pos_t.reshape(-1)[:n_dim]
            if coord_t.numel() < n_dim:
                coord_t = torch.cat(
                    [coord_t, coord_t.new_zeros(n_dim - coord_t.numel())],
                )
            frac_mean_t = (
                w * frac_t if frac_mean_t is None else frac_mean_t + w * frac_t
            )
            coords_mean_t = (
                w * coord_t if coords_mean_t is None else coords_mean_t + w * coord_t
            )
            if coords_only:
                # coords + fraction means are all the lean reading needs (FIX F2);
                # skip the per-layer trace lists + residual / membership means.
                continue
            resid_t = resid_t.reshape(1)
            mem_t = mem_t.reshape(1)
            layer_order.append(layer_idx)
            frac_terms.append(frac_t)
            resid_terms.append(resid_t)
            coord_terms.append(coord_t)
            resid_mean_t = (
                w * resid_t if resid_mean_t is None else resid_mean_t + w * resid_t
            )
            mem_mean_t = (
                w * mem_t if mem_mean_t is None else mem_mean_t + w * mem_t
            )

        if coords_only:
            # Lean reading: one host transfer of (fraction, coords) (FIX F2).
            if frac_mean_t is None or coords_mean_t is None:
                return ProbeReading(fraction=0.0, nearest=[], coords=())
            lean = torch.cat(
                [frac_mean_t.reshape(-1), coords_mean_t.reshape(-1)],
            ).detach().cpu().tolist()
            return ProbeReading(
                fraction=lean[0],
                nearest=[],
                coords=tuple(lean[1:1 + n_dim]),
                residual=0.0,
                assignment=[],
                membership=1.0,
            )

        requested_top_n = int(probe.top_n)
        top_n = requested_top_n if requested_top_n >= 0 else Kc + requested_top_n
        top_n = min(max(top_n, 0), Kc)
        nearest_dist_t: torch.Tensor | None = None
        nearest_idx_t: torch.Tensor | None = None
        if top_n and dist_acc_t is not None:
            label_scale = float(probe.label_scale)
            # Rank by **raw** whitened distance (so ``nearest`` is literally the
            # nearest node, distinct from the density-aware ``assignment``), then
            # report it in units of the probe's typical label spacing
            # ``label_scale`` (median node nearest-neighbor distance) — a single
            # per-probe constant, so it rescales the value without reordering.
            # Raw whitened distance spans ~60× across fits (a node sits 1.15..72
            # from neutral), so a bare negated-distance ``@label`` gate was not
            # portable; ``d / label_scale`` ≈ "typical label-spacings away"
            # transfers across probes.  Assignment keeps raw ``dist_acc_t`` (it
            # needs raw ``d`` for the Gaussian ``−d²/2τ²``).
            nearest_dist_raw, nearest_idx_raw = torch.topk(
                dist_acc_t, k=top_n, largest=False, sorted=True,
            )
            nearest_dist_t = cast(torch.Tensor, nearest_dist_raw) / label_scale
            nearest_idx_t = cast(torch.Tensor, nearest_idx_raw)
        # Soft assignment: softmax(−d²/(2τ²) − R·log(τ)) — a proper isotropic
        # R-D Gaussian-mixture posterior with uniform node prior.  The
        # ``logvol_bias`` term is the missing Gaussian normalization; without it
        # the bare softmax sends mass to whichever ``τ`` is largest regardless of
        # distance (broadest-node-wins).
        assign_prob_t: torch.Tensor | None = None
        assign_idx_t: torch.Tensor | None = None
        if (
            top_n
            and dist_acc_t is not None
            and probe.assign_bandwidth is not None
            and probe.assign_logvol_bias is not None
        ):
            tau = probe.assign_bandwidth.to(dist_acc_t.device, torch.float32)
            lvb = probe.assign_logvol_bias.to(dist_acc_t.device, torch.float32)
            if tau.numel() == dist_acc_t.numel() == lvb.numel():
                logits = (
                    -(dist_acc_t ** 2) / (2.0 * (tau ** 2).clamp(min=_FRACTION_EPSILON))
                    + lvb
                )
                probs = torch.softmax(logits, dim=0)
                assign_prob_t, assign_idx_t = torch.topk(
                    probs, k=top_n, largest=True, sorted=True,
                )

        # --- one device→host transfer for the whole probe reading ---
        # Means (1 + 1 + n_dim) ‖ per-layer traces (L + L + L·n_dim) ‖ nearest
        # (top_n dists + top_n indices-as-float).  ``.cpu().tolist()`` once.
        # ``shared`` is non-empty here (empty returns early above), so the loop
        # ran and the running means are populated.  Guard with an explicit early
        # return (not assert) so ``python -O`` / release builds see the same
        # behaviour — a downstream torch.cat on a None tensor would TypeError.
        if (
            frac_mean_t is None
            or resid_mean_t is None
            or coords_mean_t is None
            or mem_mean_t is None
        ):
            return ProbeReading(fraction=0.0, nearest=[], coords=())
        L = len(layer_order)
        parts: list[torch.Tensor] = [
            frac_mean_t, resid_mean_t, coords_mean_t, mem_mean_t,
            torch.cat(frac_terms), torch.cat(resid_terms),
            torch.stack(coord_terms, 0).reshape(-1),
        ]
        if nearest_dist_t is not None and nearest_idx_t is not None:
            parts.append(nearest_dist_t)
            parts.append(nearest_idx_t.to(torch.float32))
        if assign_prob_t is not None and assign_idx_t is not None:
            parts.append(assign_prob_t)
            parts.append(assign_idx_t.to(torch.float32))
        flat = torch.cat([p.reshape(-1) for p in parts]).detach().cpu().tolist()

        o = 0
        frac_mean = flat[o]
        o += 1
        residual_mean = flat[o]
        o += 1
        coords_mean = flat[o:o + n_dim]
        o += n_dim
        membership = flat[o]
        o += 1
        frac_layer_vals = flat[o:o + L]
        o += L
        resid_layer_vals = flat[o:o + L]
        o += L
        coord_layer_vals = flat[o:o + L * n_dim]
        o += L * n_dim
        frac_per_layer = {layer_order[i]: frac_layer_vals[i] for i in range(L)}
        residual_per_layer = {
            layer_order[i]: resid_layer_vals[i] for i in range(L)
        }
        coords_per_layer = {
            layer_order[i]: tuple(coord_layer_vals[i * n_dim:(i + 1) * n_dim])
            for i in range(L)
        }
        def _label(idx: int) -> str:
            # Index ``K`` is the synthetic neutral candidate (when injected);
            # everything below indexes a real corpus node.
            return (
                NEUTRAL_LABEL
                if inject_neutral and idx == K
                else manifold.node_labels[idx]
            )

        if nearest_dist_t is not None:
            tn = int(nearest_dist_t.numel())
            nd_vals = flat[o:o + tn]
            o += tn
            ni_vals = flat[o:o + tn]
            o += tn
            nearest = [
                (_label(int(round(ni_vals[j]))), nd_vals[j])
                for j in range(tn)
            ]
        elif top_n:
            nearest = [(_label(k), 0.0) for k in range(top_n)]
        else:
            nearest = []

        assignment: list[tuple[str, float]] = []
        if assign_prob_t is not None:
            ta = int(assign_prob_t.numel())
            ap_vals = flat[o:o + ta]
            o += ta
            ai_vals = flat[o:o + ta]
            o += ta
            assignment = [
                (_label(int(round(ai_vals[j]))), ap_vals[j])
                for j in range(ta)
            ]

        return ProbeReading(
            fraction=frac_mean,
            nearest=nearest,
            coords=tuple(coords_mean),
            residual=residual_mean,
            fraction_per_layer=frac_per_layer,
            coords_per_layer=coords_per_layer,
            residual_per_layer=residual_per_layer,
            assignment=assignment,
            membership=membership,
        )

    def _score_full(
        self, hidden_per_layer: dict[int, torch.Tensor],
        *, only: set[str] | None = None, coords_only: bool = False,
    ) -> dict[str, ProbeReading]:
        """Full readings for every attached probe at one state.

        **Cross-probe batched.**  Every *flat* (affine) probe shares the same
        per-layer geometry algebra over a common activation, so the whole flat
        roster is scored together — one ``Σ⁻¹h`` Woodbury apply + a few
        stacked / block-diagonal matmuls per layer — in :meth:`_score_flat_batched`,
        instead of re-running the apply and the R-dim solve once per probe per
        layer.  Curved probes keep the per-probe foot solve
        (:meth:`_score_probe_full`).  Every field of the full
        :class:`ProbeReading` is populated either way, and the batched result is
        bit-identical (to float tolerance) to the per-probe path the aggregate
        uses.

        ``only`` (a set of probe names) scores just that subset, routing every
        probe in it — flat or curved — through the per-probe :meth:`_score_probe_full`
        path (correct for both shapes, and cheap for the 1-2 probe subset a
        gate-only step needs, where building the full batched roster would be
        pure waste).  ``only=None`` is the full-roster behavior, byte-identical
        to before.
        """
        out: dict[str, ProbeReading] = {}
        if not hidden_per_layer or not self._probes:
            return out
        # ``Σ⁻¹h`` is probe-independent (depends only on the layer's hidden state
        # + the shared whitener), so it is computed at most once per layer and
        # reused across every curved probe in this pass.  Valid only for this
        # ``hidden_per_layer`` — a fresh dict per call.
        sih_cache: dict[int, torch.Tensor] = {}
        if only is not None:
            # Gate-only subset: per-probe path for both flat and curved.
            for name in self._probes:
                if name in only:
                    out[name] = self._score_probe_full(
                        self._probes[name], hidden_per_layer, sih_cache,
                        coords_only=coords_only,
                    )
            return out
        device = next(iter(hidden_per_layer.values())).device
        self._ensure_flat_cache(device)
        if self._flat_keys:
            out.update(self._score_flat_batched(
                hidden_per_layer, sih_cache, coords_only=coords_only,
            ))
        for name in self._curved_keys:
            out[name] = self._score_probe_full(
                self._probes[name], hidden_per_layer, sih_cache,
                coords_only=coords_only,
            )
        # Defensive: any probe the cache didn't classify (it always should) is
        # scored per-probe so nothing silently drops from the readings.
        for name, probe in self._probes.items():
            if name not in out:
                out[name] = self._score_probe_full(
                    probe, hidden_per_layer, sih_cache,
                    coords_only=coords_only,
                )
        # Probe-inspector live trail (gated): stamp each probe's per-layer whitened
        # query coords onto its reading.  Isolated post-pass — the hot batched
        # flat cache is untouched; runs only while a client opted in.
        if self._emit_subspace_coords:
            for name, probe in self._probes.items():
                reading = out.get(name)
                if reading is not None:
                    reading.subspace_coords_per_layer = (
                        self._subspace_coords_for(probe, hidden_per_layer)
                    )
        # Preserve probe insertion order in the returned dict.
        return {name: out[name] for name in self._probes if name in out}

    def _score_probe_gate_scalars(
        self,
        probe: "AttachedManifoldProbe",
        hidden_per_layer: dict[int, torch.Tensor],
        gate_keys: set[str],
        sih_cache: dict[int, torch.Tensor] | None = None,
    ) -> dict[str, float]:
        """Score only the exact scalar channels a probe gate consumes.

        This is intentionally narrower than :meth:`_score_probe_full`: it is used
        only by the gate-only decode path, where no UI/API consumer needs a full
        :class:`ProbeReading`. Curved ``:fraction`` and ``@label`` / ``~label``
        gates can skip the nearest-foot solve entirely; coord and membership
        gates still run the geometry they semantically require.
        """
        name = probe.name
        suffixes = {
            key[len(name):]: key for key in gate_keys
            if key == name
            or key.startswith(f"{name}[")
            or key.startswith(f"{name}:")
            or key.startswith(f"{name}@")
            or key.startswith(f"{name}~")
        }
        if not suffixes:
            return {}

        coord_axes: dict[int, str] = {}
        if "" in suffixes:
            coord_axes[0] = suffixes[""]
        for suffix, key in suffixes.items():
            if suffix.startswith("[") and suffix.endswith("]"):
                try:
                    axis = int(suffix[1:-1])
                except ValueError:
                    continue
                if axis >= 0:
                    coord_axes[axis] = key
        fraction_key = suffixes.get(":fraction")
        membership_key = suffixes.get(":membership")
        dist_labels = {
            suffix[1:]: key for suffix, key in suffixes.items()
            if suffix.startswith("@") and len(suffix) > 1
        }
        assign_labels = {
            suffix[1:]: key for suffix, key in suffixes.items()
            if suffix.startswith("~") and len(suffix) > 1
        }

        need_coords = bool(coord_axes)
        need_fraction = fraction_key is not None
        need_membership = membership_key is not None
        need_nearest = bool(dist_labels)
        need_assignment = bool(assign_labels)
        need_dist = bool(need_nearest or need_assignment)
        if not (need_coords or need_fraction or need_membership or need_dist):
            return {}

        manifold = probe.manifold
        sw = probe.share_weights
        is_affine = probe.is_affine
        shared = [idx for idx in manifold.layers if idx in hidden_per_layer]
        if not shared:
            return {}
        total_w = sum(sw.get(idx, 0.0) for idx in shared)
        if total_w <= _MIN_SHARE_WEIGHT:
            w_shared = {idx: 1.0 / len(shared) for idx in shared}
        else:
            w_shared = {idx: sw.get(idx, 0.0) / total_w for idx in shared}

        K = probe.node_values_reduced[shared[0]].shape[0]
        inject_neutral = probe.inject_neutral
        n_dim = manifold.domain.intrinsic_dim
        frac_mean_t: torch.Tensor | None = None
        coords_mean_t: torch.Tensor | None = None
        mem_mean_t: torch.Tensor | None = None
        dist_acc_t: torch.Tensor | None = None

        for layer_idx in shared:
            sub = manifold.layers[layer_idx]
            h = hidden_per_layer[layer_idx].to(torch.float32)
            if h.ndim > 1:
                h = h.reshape(-1, h.shape[-1])[-1]
            wh = probe.whitened[layer_idx]
            frac_t, cdist_query, invert_query, _cdist_nodes = _layer_geometry(
                probe, layer_idx, h, sih_cache,
            )
            w = w_shared[layer_idx]

            if need_fraction:
                frac_t = frac_t.reshape(1)
                frac_mean_t = (
                    w * frac_t if frac_mean_t is None else frac_mean_t + w * frac_t
                )

            if need_dist:
                # Neutral rides the precomputed ``node_white_aug`` (FIX F4 — no
                # per-token ``cat``).
                dists = torch.linalg.vector_norm(
                    wh.node_white_aug - cdist_query.reshape(1, -1), dim=-1,
                )
                weighted_dists = dists * w
                dist_acc_t = (
                    weighted_dists
                    if dist_acc_t is None
                    else dist_acc_t + weighted_dists.to(dist_acc_t.device)
                )

            if need_coords or need_membership:
                if is_affine:
                    pos_t = (
                        wh.coord_S @ invert_query + wh.coord_b
                        if wh.coord_S is not None and wh.coord_b is not None
                        else invert_query
                    )
                    mem_t = frac_t.new_ones(1)
                else:
                    foot_key = (probe.name, layer_idx)
                    warm = self._curved_feet.get(foot_key) if self._curved_warm else None
                    pos, res = invert_parameterization(
                        sub, manifold.domain, invert_query.reshape(1, -1),
                        manifold.node_coords, warm_start=warm,
                    )
                    if self._curved_warm:
                        self._curved_feet[foot_key] = pos.detach()
                    pos_t = pos.reshape(-1)
                    if need_membership and sub.has_sigma:
                        res_flat = res.reshape(-1)[:1]
                        sig_foot = sub.sigma_at(
                            manifold.domain.embed(pos.reshape(1, -1)),
                        ).reshape(1)
                        mem_t = torch.exp(
                            -(res_flat ** 2)
                            / (2.0 * (sig_foot ** 2).clamp(min=_FRACTION_EPSILON))
                        )
                    else:
                        mem_t = frac_t.new_ones(1)

                if need_coords:
                    coord_t = pos_t.reshape(-1)[:n_dim]
                    if coord_t.numel() < n_dim:
                        coord_t = torch.cat(
                            [coord_t, coord_t.new_zeros(n_dim - coord_t.numel())],
                        )
                    coords_mean_t = (
                        w * coord_t
                        if coords_mean_t is None
                        else coords_mean_t + w * coord_t
                    )
                if need_membership:
                    mem_t = mem_t.reshape(1)
                    mem_mean_t = (
                        w * mem_t
                        if mem_mean_t is None
                        else mem_mean_t + w * mem_t
                    )

        label_to_idx = {label: idx for idx, label in enumerate(manifold.node_labels)}
        if inject_neutral:
            label_to_idx[NEUTRAL_LABEL] = K
        dist_requests = [
            (key, label_to_idx[label])
            for label, key in dist_labels.items()
            if label in label_to_idx
        ]
        assign_requests = [
            (key, label_to_idx[label])
            for label, key in assign_labels.items()
            if label in label_to_idx
        ]

        out: dict[str, float] = {}
        parts: list[torch.Tensor] = []
        slots: list[tuple[str, Any]] = []
        if need_fraction and frac_mean_t is not None and fraction_key is not None:
            parts.append(frac_mean_t.reshape(1))
            slots.append(("scalar", fraction_key))
        if need_coords and coords_mean_t is not None:
            for axis, key in sorted(coord_axes.items()):
                if axis < int(coords_mean_t.numel()):
                    parts.append(coords_mean_t[axis].reshape(1))
                    slots.append(("scalar", key))
        if need_membership and mem_mean_t is not None and membership_key is not None:
            parts.append(mem_mean_t.reshape(1))
            slots.append(("scalar", membership_key))

        if need_nearest and dist_requests and dist_acc_t is not None:
            idx_t = torch.tensor(
                [idx for _key, idx in dist_requests],
                device=dist_acc_t.device,
                dtype=torch.long,
            )
            parts.append(dist_acc_t.index_select(0, idx_t) / probe.label_scale)
            slots.append(("nearest_exact", [key for key, _idx in dist_requests]))
        if (
            need_assignment
            and assign_requests
            and dist_acc_t is not None
            and probe.assign_bandwidth is not None
            and probe.assign_logvol_bias is not None
        ):
            tau = probe.assign_bandwidth.to(dist_acc_t.device, torch.float32)
            lvb = probe.assign_logvol_bias.to(dist_acc_t.device, torch.float32)
            if tau.numel() == dist_acc_t.numel() == lvb.numel():
                logits = (
                    -(dist_acc_t ** 2)
                    / (2.0 * (tau ** 2).clamp(min=_FRACTION_EPSILON))
                    + lvb
                )
                probs = torch.softmax(logits, dim=0)
                idx_t = torch.tensor(
                    [idx for _key, idx in assign_requests],
                    device=probs.device,
                    dtype=torch.long,
                )
                parts.append(probs.index_select(0, idx_t))
                slots.append(("assignment_exact", [key for key, _idx in assign_requests]))

        if not parts:
            return out
        flat = torch.cat([p.reshape(-1) for p in parts]).detach().cpu().tolist()
        pos = 0
        for kind, meta in slots:
            if kind == "scalar":
                out[meta] = float(flat[pos])
                pos += 1
            elif kind == "nearest_exact":
                keys = list(meta)
                vals = flat[pos:pos + len(keys)]
                pos += len(keys)
                for key, dist in zip(keys, vals, strict=True):
                    out[key] = -float(dist)
            elif kind == "assignment_exact":
                keys = list(meta)
                vals = flat[pos:pos + len(keys)]
                pos += len(keys)
                for key, prob in zip(keys, vals, strict=True):
                    out[key] = float(prob)
        return out

    def score_gate_scalars(
        self,
        hidden_per_layer: dict[int, torch.Tensor],
        gate_keys: set[str],
    ) -> dict[str, float]:
        """Return exact gate scalar keys without building full readings."""
        if not hidden_per_layer or not self._probes or not gate_keys:
            return {}
        # Only the probes a gate key actually references do any work (FIX F3):
        # derive their names once (the prefix before the first ``[`` / ``:`` /
        # ``@`` / ``~`` channel marker — ``NAME_REGEX`` forbids those in a name)
        # and dispatch ``_score_probe_gate_scalars`` for those alone, instead of
        # running it (and its per-probe suffix-dict build) for every probe in the
        # roster every token.
        names = {re.split(r"[\[:@~]", k, maxsplit=1)[0] for k in gate_keys}
        sih_cache: dict[int, torch.Tensor] = {}
        out: dict[str, float] = {}
        for name in names:
            probe = self._probes.get(name)
            if probe is None:
                continue
            out.update(
                self._score_probe_gate_scalars(
                    probe, hidden_per_layer, gate_keys, sih_cache,
                )
            )
        return out

    def _subspace_coords_for(
        self,
        probe: "AttachedManifoldProbe",
        hidden_per_layer: dict[int, torch.Tensor],
    ) -> dict[int, tuple[float, ...]]:
        """Per-layer whitened query coords for one probe (inspector trail).

        Reuses :func:`_layer_geometry` — the same primitive the read path runs —
        so the live point lands in the identical whitened frame as the geometry
        endpoint's ``node_white`` / ``neutral_white``.  One batched host transfer
        per probe; only reached when ``_emit_subspace_coords`` is set.  Rank can
        vary per layer (flat DLS prune), so each layer's slice length is tracked
        explicitly rather than assumed uniform.
        """
        rows: list[torch.Tensor] = []
        order: list[int] = []
        lens: list[int] = []
        for layer_idx in probe.manifold.layers:
            if layer_idx not in hidden_per_layer:
                continue
            if probe.whitened.get(layer_idx) is None:
                continue
            h = hidden_per_layer[layer_idx].to(torch.float32)
            if h.ndim > 1:
                h = h.reshape(-1, h.shape[-1])[-1]
            _, cdist_query, _, _ = _layer_geometry(probe, layer_idx, h)
            cq = cdist_query.reshape(-1)
            rows.append(cq)
            order.append(layer_idx)
            lens.append(int(cq.numel()))
        if not rows:
            return {}
        flat = torch.cat(rows).detach().cpu().tolist()
        out: dict[int, tuple[float, ...]] = {}
        pos = 0
        for i, ln in enumerate(lens):
            out[order[i]] = tuple(flat[pos:pos + ln])
            pos += ln
        return out

    def _ensure_flat_cache(self, device: torch.device) -> None:
        """Build/refresh the cross-probe batched flat-read cache on ``device``.

        Splits the roster into flat (``is_affine``) and curved probes, then
        precomputes (a) a **global** per-probe layout — coord-offsets, node
        counts, top-N, labels, a padded-node validity mask, and ``Rmax`` /
        ``Kmax`` — and (b) per layer the stacked / block-diagonal factors plus
        the scatter indices that let one batched sweep accumulate straight into
        global per-probe slots.  Per layer ``L``:

        * ``X`` / ``K_inv`` / ``lam`` — shared Woodbury factors (one ``Σ⁻¹h``
          per layer covers every flat probe; the whitener is identical),
        * ``basis_stack`` ``(ΣR, D)`` / ``gmean_stack`` ``(ΣR,)`` /
          ``means_stack`` ``(P_L, D)`` / ``mm_stack`` ``(P_L,)`` — the stacked
          ``B Σ⁻¹``, the ``B Σ⁻¹ μ`` shift, and the fraction-denominator cross
          terms,
        * ``Mblk`` / ``chol_blk`` / ``coordS_blk`` (block-diagonal) — per-probe
          ``M_R⁻¹``, whitened-distance Cholesky, and the affine reduced→domain
          map, each one block-diagonal matmul,
        * ``cols`` ``(P_L,)`` / ``coords_gidx`` ``(Σnd_L,)`` / ``wt_perdim`` /
          ``cq_scatter`` ``(ΣR_L,)`` — scatter indices into the global
          ``frac_acc`` / ``coords_acc`` / ``dist_acc`` slots and into the padded
          ``(P_L, Rmax)`` whitened query,
        * ``node_white_pad`` ``(P_L, Kmax, Rmax)`` — padded whitened node coords
          (real dims filled, the rest 0) so the per-layer nearest distance is one
          batched ``(P_L, Kmax)`` norm,
        * host ``present_cols`` / ``present_nd`` — for reconstructing the
          per-layer traces after the single transfer.

        Off the hot path; rebuilt only when the device, roster, or whitener
        changes (the cache-key guard short-circuits otherwise).
        """
        # Per-token guard (FIX F5): a bool + a device compare.  ``_flat_cache_dirty``
        # is flipped True by ``_invalidate_flat_cache`` on every roster / whitener
        # change, so a clean cache on the same device short-circuits without
        # rebuilding + comparing a ``tuple(self._probes.keys())`` each token.
        if not self._flat_cache_dirty and self._flat_cache_device == device:
            return
        self._flat_cache_device = device
        self._flat_cache_dirty = False
        flat_names = [
            n for n, p in self._probes.items()
            if p.is_affine and p.manifold.layers
        ]
        self._flat_keys = tuple(flat_names)
        self._curved_keys = tuple(
            n for n, p in self._probes.items()
            if not p.is_affine and p.manifold.layers
        )
        if not flat_names:
            self._flat_layer_cache = {}
            self._flat_global = {}
            return

        # --- global per-probe layout ---
        P = len(flat_names)
        nd_list = [
            int(self._probes[n].manifold.domain.intrinsic_dim) for n in flat_names
        ]
        K_list = [
            int(self._probes[n].manifold.node_coords.shape[0]) for n in flat_names
        ]
        # Neutral competes as a virtual candidate at column ``K`` per injecting
        # probe.  A flat fit is neutral-anchored (origin = reduced 0), and
        # ``node_pad`` is zero-initialized, so the reserved column already holds
        # ``neutral_white`` — its distance falls out of the existing padded norm
        # as ``‖cq‖`` with no hot-loop change.  ``Kc_list`` is the candidate
        # count (nodes + neutral) per probe.
        inj_list = [bool(self._probes[n].inject_neutral) for n in flat_names]
        Kc_list = [K + (1 if inj else 0) for K, inj in zip(K_list, inj_list, strict=True)]
        top_n_list: list[int] = []
        for n, Kc in zip(flat_names, Kc_list, strict=True):
            req = int(self._probes[n].top_n)
            tn = req if req >= 0 else Kc + req
            top_n_list.append(min(max(tn, 0), Kc))
        topk_width = max(top_n_list, default=0)
        nd_off: list[tuple[int, int]] = []
        off = 0
        for nd in nd_list:
            nd_off.append((off, nd))
            off += nd
        nd_total = off
        Kmax = max(Kc_list)
        Rmax = max(
            int(sub.basis.shape[0])
            for n in flat_names
            for sub in self._probes[n].whitened.values()
        )
        valid_mask = torch.zeros((P, Kmax), dtype=torch.bool, device=device)
        for ci, Kc in enumerate(Kc_list):
            valid_mask[ci, :Kc] = True
        # Padded per-candidate soft-assignment bandwidth ``τ`` and the precomputed
        # Gaussian log-volume bias ``−R·log(τ)`` ``(P, Kmax)`` each.  Filled from
        # each probe's precomputed ``assign_bandwidth`` / ``assign_logvol_bias``
        # (invalid/pad columns stay at the neutral defaults — band=1.0, bias=0 —
        # but are masked to −inf logit by ``valid_mask`` in the softmax, so their
        # value is irrelevant).  A probe without a bandwidth (degenerate) keeps
        # the defaults and is suppressed from the assignment downstream via
        # ``band_present``.
        band_pad = torch.ones((P, Kmax), dtype=torch.float32, device=device)
        logvol_pad = torch.zeros((P, Kmax), dtype=torch.float32, device=device)
        band_present_list = [False] * P
        for ci, n in enumerate(flat_names):
            bw = self._probes[n].assign_bandwidth
            lvb = self._probes[n].assign_logvol_bias
            if (
                bw is not None and lvb is not None
                and bw.numel() == Kc_list[ci] and lvb.numel() == Kc_list[ci]
            ):
                band_pad[ci, :bw.numel()] = bw.to(device, torch.float32)
                logvol_pad[ci, :lvb.numel()] = lvb.to(device, torch.float32)
                band_present_list[ci] = True
        # Per-probe ``@label`` distance scale ``(P,)`` — the robust median node
        # spacing each probe carries.  The reported nearest distance is divided
        # by it (a per-probe constant ⇒ preserves the raw ranking) to make the
        # ``@label`` gate threshold portable across probes.
        label_scale = torch.tensor(
            [float(self._probes[n].label_scale) for n in flat_names],
            device=device, dtype=torch.float32,
        ).clamp(min=_FRACTION_EPSILON)
        labels: list[list[str]] = []
        for n, inj in zip(flat_names, inj_list, strict=True):
            row = list(self._probes[n].manifold.node_labels)
            if inj:
                row.append(NEUTRAL_LABEL)   # index K → neutral
            labels.append(row)
        nd_counts = torch.tensor(nd_list, device=device, dtype=torch.long)
        self._flat_global = {
            "P": P, "nd_list": nd_list, "nd_off": nd_off, "nd_total": nd_total,
            "K_list": K_list, "top_n_list": top_n_list,
            "labels": labels,
            "Kmax": Kmax, "Rmax": Rmax,
            "topk_width": topk_width,
            "valid_mask": valid_mask, "nd_counts": nd_counts,
            "band_pad": band_pad, "logvol_pad": logvol_pad,
            "band_present_list": band_present_list,
            "band_present_any": any(band_present_list),
            "label_scale": label_scale,
        }

        layer_members: dict[int, list[int]] = {}
        for ci, n in enumerate(flat_names):
            for layer_idx in self._probes[n].manifold.layers:
                layer_members.setdefault(layer_idx, []).append(ci)

        cache: dict[int, dict[str, Any]] = {}
        for layer_idx, cis in layer_members.items():
            present = [flat_names[ci] for ci in cis]
            whs = [self._probes[n].whitened[layer_idx] for n in present]
            X = whs[0].X.to(device)
            K_inv = whs[0].K_inv.to(device)
            lam = whs[0].lam
            basis_stack = torch.cat([w.basis.to(device) for w in whs], dim=0)
            means_stack = torch.stack([w.mean.to(device) for w in whs], dim=0)
            simeans = [
                _woodbury_apply(w.mean.to(device), X, K_inv, lam) for w in whs
            ]
            gmean_stack = torch.cat(
                [w.basis.to(device) @ sm
                 for w, sm in zip(whs, simeans, strict=True)],
                dim=0,
            )
            mm_stack = torch.stack(
                [(w.mean.to(device) * sm).sum()
                 for w, sm in zip(whs, simeans, strict=True)],
            )
            Mblk = torch.block_diag(*[w.m_r_inv.to(device) for w in whs])
            chol_blk = torch.block_diag(*[w.chol.to(device) for w in whs])
            coordS_blk = torch.block_diag(*[
                w.coord_S.to(device) if w.coord_S is not None
                else torch.eye(
                    int(w.basis.shape[0]), device=device, dtype=torch.float32,
                )
                for w in whs
            ])
            coordb_stack = torch.cat([
                w.coord_b.to(device) if w.coord_b is not None
                else torch.zeros(
                    int(w.basis.shape[0]), device=device, dtype=torch.float32,
                )
                for w in whs
            ])
            wt = torch.tensor(
                [float(self._probes[n].share_weights.get(layer_idx, 0.0))
                 for n in present],
                device=device, dtype=torch.float32,
            )
            # scatter indices + padded node coords
            seg_ids_list: list[int] = []
            cq_scatter_list: list[int] = []
            coords_gidx_list: list[int] = []
            wt_perdim_list: list[float] = []
            present_nd: list[int] = []
            node_pad = torch.zeros((len(present), Kmax, Rmax),
                                   device=device, dtype=torch.float32)
            for li, (n, w) in enumerate(zip(present, whs, strict=True)):
                Rp = int(w.basis.shape[0])
                seg_ids_list.extend([li] * Rp)
                cq_scatter_list.extend(range(li * Rmax, li * Rmax + Rp))
                ndp = nd_list[cis[li]]
                present_nd.append(ndp)
                g_off = nd_off[cis[li]][0]
                coords_gidx_list.extend(range(g_off, g_off + ndp))
                wt_perdim_list.extend([li] * ndp)  # local index → weight gather
                nw = w.node_white.to(device)                       # (K_p, Rp)
                node_pad[li, :nw.shape[0], :Rp] = nw
            wt_perdim = wt[torch.tensor(
                wt_perdim_list, device=device, dtype=torch.long,
            )] if wt_perdim_list else wt.new_zeros(0)
            cache[layer_idx] = {
                "X": X, "K_inv": K_inv, "lam": lam,
                "basis_stack": basis_stack, "means_stack": means_stack,
                "gmean_stack": gmean_stack, "mm_stack": mm_stack,
                "Mblk": Mblk, "chol_blk": chol_blk,
                "coordS_blk": coordS_blk, "coordb_stack": coordb_stack,
                "wt": wt,
                "seg_ids": torch.tensor(seg_ids_list, device=device, dtype=torch.long),
                "cq_scatter": torch.tensor(cq_scatter_list, device=device, dtype=torch.long),
                "coords_gidx": torch.tensor(coords_gidx_list, device=device, dtype=torch.long),
                "wt_perdim": wt_perdim,
                "cols": torch.tensor(cis, device=device, dtype=torch.long),
                "cols_list": list(cis),
                "present_nd": present_nd,
                "node_pad": node_pad,
            }
        self._flat_layer_cache = cache

    def _score_flat_batched(
        self, hidden_per_layer: dict[int, torch.Tensor],
        sih_cache: dict[int, torch.Tensor] | None = None,
        *,
        coords_only: bool = False,
    ) -> dict[str, ProbeReading]:
        """Score the whole flat roster in one batched sweep + one host transfer.

        Phase 1 (per layer, batched over every flat probe): one ``Σ⁻¹h``
        Woodbury apply, stacked / block-diagonal matmuls for reduced coords,
        domain coords, fraction, and a padded ``(P_L, Kmax)`` nearest-distance
        norm — each scattered straight into global per-probe accumulators
        (``frac_acc`` / ``coords_acc`` / ``dist_acc`` / ``wtsum``).  Phase 2
        (device): share-normalize, one global ``topk`` sized to the largest requested
        ``top_n`` for nearest, and **one**
        ``.cpu()`` for the entire roster (means + per-layer traces + nearest).
        Phase 3 (host): slice the flat blob back into per-probe
        :class:`ProbeReading`s.  Values match the per-probe path (the aggregate
        uses) to float tolerance.

        ``sih_cache`` (optional, keyed by layer) is the per-pass ``Σ⁻¹h`` cache
        shared with the curved per-probe path: the apply depends only on the
        layer's hidden state + the shared whitener (identical Woodbury factors),
        so a mixed flat+curved roster computes ``Σ⁻¹h`` once per layer total.
        """
        gl = self._flat_global
        cache = self._flat_layer_cache
        if not gl or not cache:
            return {}
        P = gl["P"]
        Kmax = gl["Kmax"]
        Rmax = gl["Rmax"]
        topk_width = gl["topk_width"]
        flat_names = self._flat_keys
        device = gl["valid_mask"].device

        frac_acc = torch.zeros(P, device=device, dtype=torch.float32)
        wtsum = torch.zeros(P, device=device, dtype=torch.float32)
        coords_acc = torch.zeros(gl["nd_total"], device=device, dtype=torch.float32)
        dist_acc = torch.zeros((P, Kmax), device=device, dtype=torch.float32)
        # Per-layer trace pieces (raw, unweighted) for the single transfer.
        trace_frac: list[torch.Tensor] = []
        trace_coords: list[torch.Tensor] = []
        trace_layers: list[int] = []
        seen: set[int] = set()

        for layer_idx, h in hidden_per_layer.items():
            ent = cache.get(layer_idx)
            if ent is None:
                continue
            hf = h.to(torch.float32)
            if hf.ndim > 1:
                hf = hf.reshape(-1, hf.shape[-1])[-1]
            # Σ⁻¹h — probe-independent, so share it with the curved per-probe
            # path via the per-pass cache (same Woodbury factors per layer).
            sih = None if sih_cache is None else sih_cache.get(layer_idx)
            if sih is None:
                sih = _woodbury_apply(hf, ent["X"], ent["K_inv"], ent["lam"])
                if sih_cache is not None:
                    sih_cache[layer_idx] = sih
            h_sih = (hf * sih).sum()
            g_all = ent["basis_stack"] @ sih - ent["gmean_stack"]   # (ΣR,)
            c_all = ent["Mblk"] @ g_all                             # (ΣR,)
            coords_all = ent["coordS_blk"] @ c_all + ent["coordb_stack"]  # (Σnd_L,)
            hsm = ent["means_stack"] @ sih                          # (P_L,)
            cols = ent["cols"]
            n_present = cols.shape[0]
            par2 = torch.zeros(
                n_present, device=device, dtype=torch.float32,
            ).index_add_(0, ent["seg_ids"], g_all * c_all)          # (P_L,)
            x_m2 = (h_sih - 2.0 * hsm + ent["mm_stack"]).clamp(min=_FRACTION_EPSILON)
            frac = (par2.clamp(min=0.0).sqrt() / x_m2.sqrt()).clamp(0.0, 1.0)  # (P_L,)

            wt = ent["wt"]
            frac_acc.index_add_(0, cols, wt * frac)
            wtsum.index_add_(0, cols, wt)
            coords_acc.index_add_(0, ent["coords_gidx"], ent["wt_perdim"] * coords_all)
            seen.update(ent["cols_list"])
            if coords_only:
                # FIX F2: a live consumer that reads only the axis-0 coord /
                # fraction (the trait stream, the loom probe row) doesn't need the
                # nearest-distance norm over ``Kmax`` candidates, the assignment
                # softmax, or the host-side per-layer trace reconstruction — skip
                # them.  The full aggregate (which DOES need them) is re-scored
                # once from the tail ring at finalize.
                continue
            cq_all = c_all @ ent["chol_blk"]                        # (ΣR,)
            # Nearest: scatter cq into a padded (P_L, Rmax) query, batched norm.
            cq_pad = torch.zeros(n_present * Rmax, device=device, dtype=torch.float32)
            cq_pad = cq_pad.index_copy_(0, ent["cq_scatter"], cq_all).reshape(n_present, Rmax)
            dist = torch.linalg.vector_norm(
                ent["node_pad"] - cq_pad.unsqueeze(1), dim=-1,
            )                                                       # (P_L, Kmax)
            dist_acc.index_add_(0, cols, dist * wt.unsqueeze(1))
            trace_frac.append(frac)
            trace_coords.append(coords_all)
            trace_layers.append(layer_idx)

        # --- device finalize (share-normalize + one bounded global topk) ---
        wtsum_safe = wtsum.clamp(min=_FRACTION_EPSILON)
        frac_final = frac_acc / wtsum_safe
        coords_final = coords_acc / wtsum_safe.repeat_interleave(
            gl["nd_counts"],
        ).clamp(min=_FRACTION_EPSILON)
        if coords_only:
            # Minimal blob: per-probe fraction + domain coords only (FIX F2).
            blob = torch.cat([frac_final, coords_final]).detach().cpu().tolist()
            frac_v = blob[:P]
            coords_v = blob[P:P + gl["nd_total"]]
            nd_off = gl["nd_off"]
            out: dict[str, ProbeReading] = {}
            for ci, name in enumerate(flat_names):
                if ci not in seen:
                    out[name] = ProbeReading(fraction=0.0, nearest=[], coords=())
                    continue
                g_off, nd = nd_off[ci]
                out[name] = ProbeReading(
                    fraction=frac_v[ci],
                    nearest=[],
                    coords=tuple(coords_v[g_off:g_off + nd]),
                    residual=0.0,
                    assignment=[],
                    membership=1.0,
                )
            return out
        dist_final = (dist_acc / wtsum_safe.unsqueeze(1)).masked_fill(
            ~gl["valid_mask"], float("inf"),
        )
        if topk_width:
            # Rank by **raw** distance (``nearest`` stays literally nearest), then
            # divide the reported top-N distances by each probe's ``label_scale``
            # (a per-probe constant ⇒ no reordering) for a portable ``@label``
            # gate — distance in "typical label-spacings"; see ``_score_probe_full``.
            # Assignment below keeps the raw ``dist_final``.
            nd_sorted, ni_sorted = torch.topk(
                dist_final, k=topk_width, largest=False, sorted=True,
            )                                                       # (P, topk_width)
            nd_sorted = nd_sorted / gl["label_scale"].unsqueeze(1)
        else:
            nd_sorted = dist_final.new_zeros((P, 0))
            ni_sorted = torch.empty((P, 0), device=device, dtype=torch.long)
        # Soft assignment: softmax(−d²/(2τ²) − R·log(τ)) per probe over valid
        # candidates (invalid columns are +inf distance ⇒ −inf logit ⇒ 0
        # probability), top-``topk_width`` by probability.  ``band_pad`` is the
        # precomputed per-candidate bandwidth, ``logvol_pad`` the precomputed
        # ``−R·log(τ)`` Gaussian normalization bias; together they make a proper
        # isotropic R-D mixture posterior (vs. the bare ``−d²/2τ²`` softmax's
        # broadest-node-wins pathology).  Rows without a bandwidth get the
        # neutral defaults (band=1, bias=0) so the blob shape is static
        # (suppressed per-probe on the host via ``band_present``).
        if topk_width and gl["band_present_any"]:
            assign_logits = -(dist_final ** 2) / (
                2.0 * (gl["band_pad"] ** 2).clamp(min=_FRACTION_EPSILON)
            ) + gl["logvol_pad"]
            assign_probs = torch.softmax(assign_logits, dim=1)      # (P, Kmax)
            ap_sorted, ai_sorted = torch.topk(
                assign_probs, k=topk_width, largest=True, sorted=True,
            )                                                       # (P, topk_width)
        else:
            ap_sorted = dist_final.new_zeros((P, topk_width))
            ai_sorted = torch.zeros((P, topk_width), device=device, dtype=torch.long)

        all_frac = (
            torch.cat(trace_frac) if trace_frac
            else frac_final.new_zeros(0)
        )
        all_coords = (
            torch.cat(trace_coords) if trace_coords
            else coords_final.new_zeros(0)
        )
        blob = torch.cat([
            frac_final, coords_final,
            nd_sorted.reshape(-1), ni_sorted.to(torch.float32).reshape(-1),
            ap_sorted.reshape(-1), ai_sorted.to(torch.float32).reshape(-1),
            all_frac, all_coords,
        ]).detach().cpu().tolist()

        # --- host reconstruction (one slice walk; no per-probe sync) ---
        nd_off = gl["nd_off"]
        top_n_list = gl["top_n_list"]
        labels = gl["labels"]
        o = 0
        frac_v = blob[o:o + P]
        o += P
        coords_v = blob[o:o + gl["nd_total"]]
        o += gl["nd_total"]
        nd_v = blob[o:o + P * topk_width]
        o += P * topk_width
        ni_v = blob[o:o + P * topk_width]
        o += P * topk_width
        ap_v = blob[o:o + P * topk_width]
        o += P * topk_width
        ai_v = blob[o:o + P * topk_width]
        o += P * topk_width
        band_present = gl["band_present_list"]
        # per-layer trace offsets within all_frac / all_coords
        frac_per_layer_acc: list[dict[int, float]] = [{} for _ in range(P)]
        coords_per_layer_acc: list[dict[int, tuple[float, ...]]] = [{} for _ in range(P)]
        residual_per_layer_acc: list[dict[int, float]] = [{} for _ in range(P)]
        fo = o
        co = o + all_frac.numel()
        for layer_idx in trace_layers:
            ent = cache[layer_idx]
            cols_list = ent["cols_list"]
            present_nd = ent["present_nd"]
            ndo = co
            for li, ci in enumerate(cols_list):
                frac_per_layer_acc[ci][layer_idx] = blob[fo + li]
                ndp = present_nd[li]
                coords_per_layer_acc[ci][layer_idx] = tuple(blob[ndo:ndo + ndp])
                residual_per_layer_acc[ci][layer_idx] = 0.0
                ndo += ndp
            fo += len(cols_list)
            co += sum(present_nd)

        out: dict[str, ProbeReading] = {}
        for ci, name in enumerate(flat_names):
            if ci not in seen:
                out[name] = ProbeReading(fraction=0.0, nearest=[], coords=())
                continue
            g_off, nd = nd_off[ci]
            top_n = top_n_list[ci]
            row = ci * topk_width
            nearest = [
                (labels[ci][int(round(ni_v[row + j]))], nd_v[row + j])
                for j in range(top_n)
            ]
            assignment: list[tuple[str, float]] = []
            if band_present[ci]:
                assignment = [
                    (labels[ci][int(round(ai_v[row + j]))], ap_v[row + j])
                    for j in range(top_n)
                ]
            out[name] = ProbeReading(
                fraction=frac_v[ci],
                nearest=nearest,
                coords=tuple(coords_v[g_off:g_off + nd]),
                residual=0.0,
                fraction_per_layer=frac_per_layer_acc[ci],
                coords_per_layer=coords_per_layer_acc[ci],
                residual_per_layer=residual_per_layer_acc[ci],
                assignment=assignment,
                membership=1.0,
            )
        return out

    def _score_tokens(
        self,
        hidden_per_layer: dict[int, torch.Tensor],
        accumulate: bool = True,
        *,
        only: set[str] | None = None,
        coords_only: bool = False,
    ) -> dict[str, ProbeReading]:
        """Score every probe → ``{name: ProbeReading}`` (full reading).

        ``accumulate`` folds the cross-layer coords into history/stats (the
        in-flight per-token path passes False).  ``only`` restricts scoring to
        a subset (gate-only per-token path); a subset read never accumulates
        (the cross-gen stats want the full roster, not the gated probes alone).
        ``coords_only`` (FIX F2) returns the lean coords+fraction reading.
        """
        out = self._score_full(
            hidden_per_layer, only=only, coords_only=coords_only,
        )
        if out and accumulate and only is None:
            self._apply_accumulate(out)
        return out

    def _apply_accumulate(
        self, readings: dict[str, ProbeReading],
    ) -> None:
        """Fold per-probe aggregate coords into history + per-axis stats.

        ``history`` stores the full coordinate tuple; ``_stats`` keeps
        axis-0 scalar accumulators (TUI compat) plus per-axis ``sum`` /
        ``sum_sq`` / ``min`` / ``max`` lists for the session's vectorized
        :class:`ProbeReadings`.
        """
        for name, reading in readings.items():
            if name not in self.history:
                continue
            coords = reading.coords or (0.0,)
            self.history[name].append(tuple(coords))
            s = self._stats[name]
            s["count"] += 1
            v0 = coords[0]
            s["sum"] += v0
            s["sum_sq"] += v0 * v0
            if v0 < s["min"]:
                s["min"] = v0
            if v0 > s["max"]:
                s["max"] = v0
            # Per-axis accumulators (lazily sized to the coord rank).
            axes = s.setdefault("axes", [])
            for i, v in enumerate(coords):
                if i >= len(axes):
                    axes.append({"sum": 0.0, "sum_sq": 0.0,
                                 "min": float("inf"), "max": float("-inf")})
                a = axes[i]
                a["sum"] += v
                a["sum_sq"] += v * v
                if v < a["min"]:
                    a["min"] = v
                if v > a["max"]:
                    a["max"] = v
        self._pending_aggregate = True

    def accumulate_readings(
        self, readings: dict[str, ProbeReading],
    ) -> None:
        """Fold an already-scored aggregate into history/stats.

        The normal stack-scoring path scores and accumulates in one call.  The
        incremental capture path scores each token live, so finalization already
        has the aggregate ``ProbeReading`` and should not rescore just to update
        cross-generation stats.
        """
        self._apply_accumulate(readings)

    def score_single_token(
        self, hidden_per_layer: dict[int, torch.Tensor],
        *, only: set[str] | None = None, coords_only: bool = False,
    ) -> dict[str, "ProbeReading"]:
        """Per-probe full reading for a single token (no accumulate).

        The live per-token read source: returns ``{name: ProbeReading}``
        with ``coords`` (domain frame) + ``fraction`` + ``nearest`` +
        ``residual`` and their per-layer traces, every probe (flat or curved).
        Does NOT touch history/stats — the in-flight gate/stream path must
        not corrupt the session-level accumulators — but flips the
        per-token pending flag so the TUI/webui can poll for a fresh reading.

        ``only`` (a set of probe names) scores just that subset, returning
        ``{name: ProbeReading}`` for the subset alone — the gate-only per-token
        path, where the step sink consumes just the gated probes' scalars and
        the big-K roster's nearest-distance work is pure waste.  ``only=None``
        keeps the byte-identical full-roster behavior.  ``coords_only`` (FIX F2)
        returns the lean coords+fraction reading (no nearest / assignment /
        per-layer trace) for the axis-0-only live consumers.
        """
        out = self._score_tokens(
            hidden_per_layer, accumulate=False, only=only,
            coords_only=coords_only,
        )
        if out:
            self._pending_per_token = True
        return out

    def score_single_token_per_layer(
        self,
        hidden_per_layer: dict[int, torch.Tensor],
    ) -> dict[int, dict[str, float]]:
        """Per-layer × per-probe domain coordinate (axis 0) for a single token.

        ``{layer_idx: {probe_name: coord}}`` — the un-aggregated axis-0
        domain coordinate per layer, a view over the full reading's
        ``coords_per_layer``.  Now covers curved probes too (their per-layer
        coords fall out of the foot solve), not just the affine roster.
        """
        readings = self._score_full(hidden_per_layer)
        out: dict[int, dict[str, float]] = {}
        for name, reading in readings.items():
            for layer_idx, coord in reading.coords_per_layer.items():
                out.setdefault(layer_idx, {})[name] = coord[0] if coord else 0.0
        return out

    def measure_from_hidden(
        self, hidden_per_layer: dict[int, torch.Tensor], accumulate: bool = True,
    ) -> dict[str, "ProbeReading"]:
        """Score probes from pre-captured hidden states (no forward pass).

        Use when hidden states have already been captured during generation
        (e.g. via capture hooks), avoiding a redundant forward pass.
        """
        return self._score_tokens(hidden_per_layer, accumulate=accumulate)

    @staticmethod
    def flat_scalars(
        readings: dict[str, "ProbeReading"],
    ) -> dict[str, float]:
        """Flatten per-probe readings into namespaced gate-callback scalars.

        The single gate emitter for every probe shape.  For each probe emits
        the bare ``"<probe>"`` aliased to coordinate axis 0 (so
        ``@when:angry.calm > 0.4`` reads the single coord of a 2-node concept
        unchanged), one ``"<probe>[<i>]"`` per coordinate axis (so
        ``@when:personas[3] > 0.4`` indexes an axis), ``"<probe>:fraction"``,
        and one ``"<probe>@<label>"`` per nearest node as ``-distance`` (so
        larger means closer and ``@when:emotions@happy > -0.1`` reads like a
        similarity gate).  Under the unified full reading every probe — flat
        and curved — carries coords *and* nearest, so flat probes now expose
        ``@label`` similarity gates too (e.g. ``@when:personas@hacker``) and
        the gate grammar is uniform.

        **Fuzzy channels** (additive — the ``@label`` distance gates are
        untouched): one ``"<probe>~<label>"`` per assigned node as the
        soft-assignment *probability* (``@when:personas~hacker > 0.5`` — a
        normalized, in-``[0,1]`` membership gate, unlike the unbounded
        ``-distance``), and ``"<probe>:membership"`` as the graded tube-fit
        (``@when:emotions:membership > 0.6`` — high when the activation sits
        inside the manifold's learned thickness).  All shapes merge directly into
        ``TriggerContext.probe_scores``.
        """
        out: dict[str, float] = {}
        for name, reading in readings.items():
            coords = reading.coords
            if coords:
                out[name] = coords[0]
                for i, c in enumerate(coords):
                    out[f"{name}[{i}]"] = c
            out[f"{name}:fraction"] = reading.fraction
            for label, dist in reading.nearest:
                out[f"{name}@{label}"] = -dist
            for label, prob in reading.assignment:
                out[f"{name}~{label}"] = prob
            out[f"{name}:membership"] = reading.membership
        return out

    def _per_token_coord_stream(
        self, captured: dict[int, torch.Tensor], n: int,
    ) -> dict[str, list[float]]:
        """Axis-0 domain-coordinate stream per probe over ``n`` tokens.

        Each token is scored ``coords_only`` (the flat affine map + curved foot
        solve that axis 0 needs, skipping the big-K nearest norm, soft
        assignment, and per-layer host reconstruction this stream discards), so
        curved probes still carry a real per-token coord stream (not zeros).
        Row ``i`` reads ``captured[L][i]`` (guarded), so an EOS overshoot is
        ignored and a short capture leaves trailing zeros.
        """
        per_token: dict[str, list[float]] = {name: [0.0] * n for name in self._probes}
        if not self._probes:
            return per_token
        for i in range(n):
            tok = {L: h[i] for L, h in captured.items() if h.shape[0] > i}
            if not tok:
                continue
            for name, reading in self._score_full(tok, coords_only=True).items():
                per_token[name][i] = reading.coords[0] if reading.coords else 0.0
        return per_token

    def score_per_token(
        self,
        captured: dict[int, torch.Tensor],
        generated_ids: list[int],
        tokenizer: Any,
        *,
        accumulate: bool = True,
        aggregate_index: int | None = None,
    ) -> tuple[dict[str, "ProbeReading"], dict[str, list[float]]]:
        """Score probes per generated token using pre-captured hidden states.

        ``captured[layer_idx]`` is a ``(n, dim)`` stack where row ``k`` is
        the hidden state that produced generated token ``k``.  Returns
        ``(aggregate_readings, per_token_coord_stream)``: the aggregate is
        the full :class:`ProbeReading` per probe pooled at the last
        non-special token (updates history when ``accumulate``); the stream
        is the per-token axis-0 domain coordinate.
        """
        n = len(generated_ids)
        empty_agg = {
            name: ProbeReading(fraction=0.0, nearest=[], coords=())
            for name in self._probes
        }
        if n == 0 or not captured:
            return empty_agg, {name: [] for name in self._probes}

        if aggregate_index is not None and int(aggregate_index) < 0:
            agg = empty_agg
            per_token = self._per_token_coord_stream(captured, n)
            if accumulate:
                self._pending_per_token = True
            return agg, per_token
        if aggregate_index is None:
            from saklas.core.vectors import last_content_index
            agg_idx = last_content_index(generated_ids, tokenizer)
        else:
            agg_idx = max(0, min(int(aggregate_index), n - 1))
        agg_hidden = {
            layer_idx: h[agg_idx] for layer_idx, h in captured.items()
            if h.shape[0] > agg_idx
        }
        agg = self._score_tokens(agg_hidden, accumulate=accumulate)
        per_token = self._per_token_coord_stream(captured, n)
        if accumulate:
            self._pending_per_token = True
        return agg, per_token

    def score_stack(
        self,
        captured: dict[int, torch.Tensor],
        *,
        agg_index: int | None = None,
        accumulate: bool = False,
    ) -> tuple[dict[str, "ProbeReading"], dict[str, list[float]]]:
        """Score probes over a pre-captured ``[T, D]`` stack per layer.

        Like :meth:`score_per_token` but without ``generated_ids`` /
        tokenizer — the caller has already chosen the meaningful rows.
        Aggregate is pooled from row ``agg_index`` (default the last row).
        ``accumulate`` defaults to ``False`` (ad-hoc researcher probing,
        not the in-flight loop).
        """
        empty_agg = {
            name: ProbeReading(fraction=0.0, nearest=[], coords=())
            for name in self._probes
        }
        if not captured:
            return empty_agg, {name: [] for name in self._probes}

        any_h = next(iter(captured.values()))
        if any_h.ndim != 2:
            raise ValueError(
                f"score_stack expects [T, D] per layer; got shape {tuple(any_h.shape)}",
            )
        n = any_h.shape[0]
        if n == 0:
            return empty_agg, {name: [] for name in self._probes}

        # Uniform T across layers — mixed lengths are a caller bug.
        for layer_idx, h in captured.items():
            if h.ndim != 2 or h.shape[0] != n:
                raise ValueError(
                    f"score_stack: layer {layer_idx} has shape {tuple(h.shape)}, "
                    f"expected [{n}, D]",
                )

        agg_row = n - 1 if agg_index is None else int(agg_index)
        if not (0 <= agg_row < n):
            raise ValueError(
                f"score_stack: agg_index={agg_index} out of range for T={n}",
            )

        # T-uniformity validated above and agg_row < n; no per-layer
        # shape guard is needed (unlike score_per_token, which operates
        # on raw hook captures that can be ragged by one around EOS).
        agg_hidden = {
            layer_idx: h[agg_row] for layer_idx, h in captured.items()
        }
        agg = self._score_tokens(agg_hidden, accumulate=accumulate)
        per_token = self._per_token_coord_stream(captured, n)
        if accumulate:
            self._pending_per_token = True
        return agg, per_token

    def has_pending_data(self) -> bool:
        """True iff an aggregate measurement is waiting to be consumed."""
        return self._pending_aggregate or self._live_pending

    def has_pending_per_token(self) -> bool:
        return self._pending_per_token

    def consume_pending(self) -> None:
        """Mark aggregate pending data as consumed (called by TUI after reading)."""
        self._pending_aggregate = False
        self._live_pending = False

    def consume_pending_per_token(self) -> None:
        self._pending_per_token = False

    def begin_live(self) -> None:
        """Reset the live running-mean accumulator at the start of a gen."""
        self._live_values = {}
        self._live_count = 0
        self._live_pending = False

    def update_live(self, readings: dict[str, "ProbeReading"]) -> None:
        """Fold one token's per-probe coordinate axis 0 into the running mean.

        Accepts the per-token readings the score callback already produces;
        the live mean tracks coordinate axis 0 (the TUI-facing scalar).
        """
        self._live_count += 1
        c = self._live_count
        for name, reading in readings.items():
            v = reading.coords[0] if reading.coords else 0.0
            prev = self._live_values.get(name, [0.0])
            self._live_values[name] = [prev[0] + (v - prev[0]) / c]
        self._live_pending = True

    def end_live(self) -> None:
        """Drop the live running mean so reads fall back to history."""
        self._live_values = {}
        self._live_count = 0
        self._live_pending = False

    def get_current_and_previous(self) -> tuple[dict[str, float], dict[str, float]]:
        """Axis-0 scalar current/previous reading per probe (TUI compat)."""
        current: dict[str, float] = {}
        previous: dict[str, float] = {}
        for name in self._probes:
            hist = self.history[name]
            live = self._live_values.get(name)
            if live is not None:
                current[name] = live[0]
                previous[name] = hist[-1][0] if hist else live[0]
            elif len(hist) >= 2:
                current[name] = hist[-1][0]
                previous[name] = hist[-2][0]
            elif hist:
                current[name] = hist[-1][0]
                previous[name] = hist[-1][0]
            else:
                current[name] = 0.0
                previous[name] = 0.0
        return current, previous

    def get_stats(self, name: str) -> dict[str, Any]:
        return self._stats.get(name, self._empty_stats())

    def axis_stats(self, name: str) -> list[dict[str, Any]]:
        """Per-coordinate-axis accumulators for the vectorized readings.

        ``[{sum, sum_sq, min, max}, ...]`` aligned with the coordinate
        axes; empty until the probe has accumulated a reading.  The
        session reads this for the per-axis :class:`ProbeReadings`.
        """
        return self._stats.get(name, {}).get("axes", [])

    def get_sparkline(self, name: str) -> str:
        blocks = " ▁▂▃▄▅▆▇█"
        # Axis-0 coordinate history (TUI compat).
        values = [c[0] for c in self.history[name]] if name in self.history else []
        if not values:
            return ""
        lo, hi = min(values), max(values)
        span = hi - lo if hi != lo else 1.0
        return "".join(blocks[min(8, max(0, int((v - lo) / span * 8)))] for v in values)

    def add_probe(self, name: str, manifold: "Manifold", *, top_n: int = DEFAULT_NEAREST_TOP_N):
        """Register a :class:`Manifold` probe — flat (any rank) or curved.

        Pre-caches the per-layer node values + EV weights + whitened factors
        (and, for a flat fit, the affine reduced→domain coord map) via
        :func:`_attach_manifold_probe`; the wired whitener must cover the
        fit's layers.  Flat vs curved is cached once on the attached probe.
        """
        is_new = name not in self._probes
        self._probes[name] = _attach_manifold_probe(
            name, manifold, top_n=top_n, whitener=self._whitener,
            factor_cache=self._whitener_factor_cache,
        )
        self._invalidate_flat_cache()
        if is_new:
            self.history[name] = deque(maxlen=_MAX_HISTORY)
            self._stats[name] = self._empty_stats()

    def remove_probe(self, name: str):
        self._probes.pop(name, None)
        self._whitener_factor_cache.clear()
        self._invalidate_flat_cache()
        if name in self.history:
            del self.history[name]
        if name in self._stats:
            del self._stats[name]

    def attached_probes(self) -> dict[str, "AttachedManifoldProbe"]:
        """Live attached-probe map (read-only view) — name → probe."""
        return dict(self._probes)

    def attached_layers(self) -> set[int]:
        """Union of fit-layer indices across every attached probe.

        Alias of :meth:`probe_layers` (capture-widening signal); kept for the
        server/TUI surfaces that consumed the former ``ManifoldMonitor``.
        """
        return self.probe_layers()

    def reset(self) -> None:
        """Drop every attached probe (TUI ``/unsteer``-style reset)."""
        self._probes.clear()
        self.history.clear()
        self._stats.clear()
        self._whitener_factor_cache.clear()
        self._invalidate_flat_cache()
        self._pending_aggregate = False
        self._pending_per_token = False

    def reset_history(self):
        for name in self._probes:
            self.history[name].clear()
            self._stats[name] = self._empty_stats()
        self._pending_aggregate = False
        self._pending_per_token = False

    def score_aggregate(
        self,
        captured_per_layer: dict[int, torch.Tensor],
        *,
        agg_index: int | None = None,
    ) -> dict[str, ProbeReading]:
        """End-of-generation aggregate over pooled captures — all probes.

        ``captured_per_layer[L]`` is the per-layer ``[T, D]`` capture stack.
        The pooled activation is the **last non-special token** (``agg_index``,
        default the final row) — the same single-state discipline extraction
        uses, not a trajectory mean.  It routes through the same full scorer
        as live single-token reads, so flat probe rosters use the batched
        Woodbury path while curved probes keep their per-probe foot solve.
        """
        out: dict[str, ProbeReading] = {}
        if not captured_per_layer or not self._probes:
            return out

        pooled: dict[int, torch.Tensor] = {}
        for layer_idx, stack in captured_per_layer.items():
            if stack is None or stack.numel() == 0:
                continue
            if stack.ndim == 1:
                pooled[layer_idx] = stack.to(torch.float32)
            else:
                t = stack.shape[0]
                row = t - 1 if agg_index is None else max(0, min(agg_index, t - 1))
                pooled[layer_idx] = stack.to(torch.float32)[row]

        return self._score_full(pooled)

    def probe_geometry(
        self, name: str, *, grid_resolution: int = 24,
    ) -> dict[str, Any]:
        """Static geometry payload for one attached probe (inspector plot).

        Everything in the **whitened frame** the reads use: per-layer node
        centroids (``wh.node_white``), the neutral anchor (``wh.neutral_white``),
        a top-3 PCA rotation of the node cloud when ``rank >= 3``, and — for a
        curved fit of intrinsic dim 1/2 embedded in a higher-rank subspace — the
        manifold curve/surface sampled and whitened into the same frame.  Pairs
        with the per-token ``ProbeReading.subspace_coords_per_layer`` (the live
        point + trail), which carries the same ``cdist_query`` whitened coord.
        CPU/fp32 for the SVD + grid (MPS-fallback discipline); off the hot path.
        """
        probe = self._probes.get(name)
        if probe is None:
            raise KeyError(name)
        manifold = probe.manifold
        domain = manifold.domain
        n = int(domain.intrinsic_dim)
        layers_out: dict[str, Any] = {}
        ranks: set[int] = set()
        for layer_idx in sorted(manifold.layers):
            wh = probe.whitened.get(layer_idx)
            if wh is None:
                continue
            sub = manifold.layers[layer_idx]
            node_white = wh.node_white.detach().to(torch.float32).cpu()   # (K, R)
            R = int(node_white.shape[1])
            ranks.add(R)
            neutral_white = (
                wh.neutral_white.detach().to(torch.float32).cpu().reshape(-1)
            )
            pca_rotation: list[list[float]] | None = None
            ev_pcs: list[float] | None = None
            if R >= 3 and node_white.shape[0] >= 2:
                centered = node_white - node_white.mean(dim=0, keepdim=True)
                try:
                    _u, s, vh = torch.linalg.svd(centered, full_matrices=False)
                    pca_rotation = vh[:3].transpose(0, 1).contiguous().tolist()
                    var = s ** 2
                    tot = float(var.sum().clamp(min=1e-12))
                    ev_pcs = [float(v) / tot for v in var[:3].tolist()]
                except Exception:
                    pca_rotation, ev_pcs = None, None
            overlay: dict[str, Any] | None = None
            if (
                (not sub.is_affine)
                and n in (1, 2)
                and n < R
                and isinstance(domain, BoxDomain)
            ):
                overlay = self._sample_overlay(sub, domain, wh, n, grid_resolution)
            layers_out[str(layer_idx)] = {
                "layer": layer_idx,
                "rank": R,
                "intrinsic_dim": n,
                "is_affine": bool(sub.is_affine),
                "node_white": node_white.tolist(),
                "neutral_white": neutral_white.tolist(),
                "pca_rotation": pca_rotation,
                "explained_variance_pcs": ev_pcs,
                "mahalanobis_share": float(
                    manifold.mahalanobis_share.get(layer_idx, 0.0),
                ),
                "overlay": overlay,
            }
        return {
            "name": name,
            "manifold": manifold.name,
            "intrinsic_dim": n,
            "is_affine": bool(probe.is_affine),
            "node_labels": list(manifold.node_labels),
            "rank_uniform": len(ranks) <= 1,
            "layers": layers_out,
        }

    @staticmethod
    def _sample_overlay(
        sub: Any, domain: BoxDomain, wh: Any, n: int, res: int,
    ) -> dict[str, Any] | None:
        """Sample a curved manifold's curve (n=1) / surface (n=2), whitened.

        Returns the sampled points already in the ``node_white`` frame
        (``r @ chol``), so the frontend overlays them on the same plot without a
        second transform.  Off the hot path — CPU/fp32 throughout.
        """
        cpu = torch.device("cpu")
        sub_c = sub.to(device=cpu, dtype=torch.float32)
        chol = wh.chol.detach().to(torch.float32).cpu()       # (R, R)
        basis = sub_c.basis                                   # (R, D)
        mean = sub_c.mean                                     # (D,)
        axes = domain.axes

        def _axis_grid(ax: Any) -> torch.Tensor:
            if ax.periodic:
                return torch.linspace(
                    0.0, float(ax.period), res + 1, dtype=torch.float32,
                )[:-1]
            return torch.linspace(
                float(ax.lo), float(ax.hi), res, dtype=torch.float32,
            )

        def _whiten(coords: torch.Tensor) -> torch.Tensor:
            world = sub_c.eval_at(domain.embed(domain.clamp_position(coords)))
            reduced = (world - mean) @ basis.transpose(0, 1)  # (.., R)
            return reduced @ chol                             # (.., R) whitened

        if n == 1:
            g = _axis_grid(axes[0]).reshape(-1, 1)            # (S, 1)
            return {"kind": "curve", "points": _whiten(g).tolist()}
        gu, gv = _axis_grid(axes[0]), _axis_grid(axes[1])
        uu, vv = torch.meshgrid(gu, gv, indexing="ij")
        grid = torch.stack([uu.reshape(-1), vv.reshape(-1)], dim=1)  # (nu*nv, 2)
        return {
            "kind": "surface",
            "points": _whiten(grid).tolist(),
            "grid_shape": [int(gu.numel()), int(gv.numel())],
        }
