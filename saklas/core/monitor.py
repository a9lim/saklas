from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from saklas.core.mahalanobis import WhitenerError
from saklas.core.manifold import invert_parameterization
from saklas.core.results import ProbeReading

if TYPE_CHECKING:
    from saklas.core.manifold import Manifold

_MAX_HISTORY = 8

_EMPTY_STATS = {"count": 0, "sum": 0.0, "sum_sq": 0.0,
                "min": float("inf"), "max": float("-inf")}

# Default top-N nearest-node count for manifold probes.  Per-probe
# override available on ``Monitor.add_probe``.
DEFAULT_NEAREST_TOP_N: int = 3

# Synthetic label for the neutral anchor in the nearest-node readout.  Every
# fit is neutral-anchored (the per-model neutral mean is the frame origin), so
# neutral is a *point* in the same whitened metric the nodes live in — not a
# stored corpus node.  It competes in the ``nearest`` ranking as a virtual
# candidate (computed, never written to ``node_labels`` / ``node_coords``): when
# the running activation sits closer to the origin than to any node, ``nearest``
# reports ``("neutral", dist)`` and ``flat_scalars`` exposes the uniform
# ``<probe>@neutral`` gate channel.  Suppressed when a manifold already carries a
# real node with this label (the corpus node owns the name).
NEUTRAL_LABEL: str = "neutral"

# Floor for the EV weight on a manifold layer with degenerate fit
# quality — keeps the EV-weighted aggregation from collapsing to NaN
# on a manifold whose every layer reports EV ≈ 0.  Matches the
# ``min_ev`` floor on additive-mode manifold steering's quality_factor.
_MIN_EV_WEIGHT: float = 1e-6

# Guard against division by zero in the subspace_fraction / cosine
# denominator (a zero or near-zero activation norm).
_FRACTION_EPSILON: float = 1e-8


def _woodbury_apply(
    v: torch.Tensor, X: torch.Tensor, K: torch.Tensor, lam: float,
) -> torch.Tensor:
    """On-device ``Σ_reg⁻¹ v = (1/λ)(v − Xᵀ K (X v))`` (Woodbury).

    Shared by :class:`Monitor` and :class:`Monitor`.  Kept a
    plain module-level function (not a method) so the per-token hot path is
    a global lookup with no class attribute resolution — the hot-path
    companion to :meth:`LayerWhitener.apply_inv` (which force-promotes
    fp32/CPU and is wrong here).  ``v`` is ``[D]`` or ``[n, D]``; ``X`` is
    ``[N, D]``, ``K`` is ``[N, N]``, all on the same device/dtype.  Pure
    matmuls, no host sync.  For a ``[n, D]`` batch this is
    ``(1/λ)(V − (V Xᵀ) K X)``; for ``[D]`` it broadcasts the same way.
    """
    Xv = v @ X.transpose(0, 1)        # [..., N]
    KXv = Xv @ K.transpose(0, 1)      # [..., N]  (K symmetric; t() = K)
    return (v - KXv @ X) / lam        # [..., D]


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
        self._flat_cache_sig: tuple[str, ...] = ()
        self._flat_cache_wid: int | None = None

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

    def probe_layers(self) -> set[int]:
        """Union of fit-layer indices across every attached probe.

        The capture-widening signal the session uses to retain every layer
        a probe reads (peer of :meth:`Monitor.attached_layers`).
        """
        out: set[int] = set()
        for probe in self._probes.values():
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
        self._flat_cache_sig = ()
        self._flat_cache_wid = None

    def _score_probe_full(
        self,
        probe: "AttachedManifoldProbe",
        hidden_per_layer: dict[int, torch.Tensor],
    ) -> ProbeReading:
        """Full per-probe reading for one state — flat or curved, all fields.

        Loops the probe's shared fit layers through :func:`_layer_geometry`
        and EV-weights across them.  A flat (affine) probe recovers ``coords``
        through the affine reduced→domain map (off-surface ``residual`` 0); a
        curved probe through the :func:`invert_parameterization` foot solve
        (real off-surface ``residual``).  Returns the cross-layer EV-weighted
        ``coords`` / ``fraction`` / ``nearest`` / ``residual`` plus their
        per-layer traces.  This is the single geometry primitive every read
        entry point (live per token, aggregate) shares, so the aggregate at a
        token index is bit-identical to the live read at that token.
        """
        manifold = probe.manifold
        ev = probe.ev_weights
        is_affine = probe.is_affine
        shared = [idx for idx in manifold.layers if idx in hidden_per_layer]
        if not shared:
            return ProbeReading(fraction=0.0, nearest=[], coords=())
        total_w = sum(ev.get(idx, 0.0) for idx in shared)
        if total_w <= _MIN_EV_WEIGHT:
            w_shared = {idx: 1.0 / len(shared) for idx in shared}
        else:
            w_shared = {idx: ev.get(idx, 0.0) / total_w for idx in shared}

        K = probe.node_values_reduced[shared[0]].shape[0]
        inject_neutral = probe.inject_neutral
        Kc = K + (1 if inject_neutral else 0)   # candidate count incl. neutral
        n_dim = manifold.domain.intrinsic_dim
        dist_acc_t: torch.Tensor | None = None
        # Per-layer terms + running EV-weighted means are accumulated as
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
            frac_t, cdist_query, invert_query, cdist_nodes = _layer_geometry(
                probe, layer_idx, h,
            )
            w = w_shared[layer_idx]

            # Neutral competes as a virtual candidate appended after the K real
            # nodes (index ``K``): its whitened coord is ``wh.neutral_white``
            # (0 for an affine fit, the baked origin for a curved one), so the
            # same cdist yields its distance with no special-casing downstream.
            if inject_neutral:
                cdist_nodes = torch.cat(
                    [cdist_nodes, wh.neutral_white.reshape(1, -1)], dim=0,
                )
            dists = torch.linalg.vector_norm(
                cdist_nodes - cdist_query.reshape(1, -1), dim=-1,
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
                # (sequential) live-scoring pass; otherwise solve cold.
                foot_key = (probe.name, layer_idx)
                warm = self._curved_feet.get(foot_key) if self._curved_warm else None
                pos, res = invert_parameterization(
                    sub, manifold.domain, invert_query.reshape(1, -1),
                    manifold.node_coords, warm_start=warm,
                )
                if self._curved_warm:
                    self._curved_feet[foot_key] = pos.detach()
                pos_t = pos.reshape(-1)
                par_norm = torch.linalg.vector_norm(invert_query)
                res_flat = res.reshape(-1)[:1]
                # norm_residual = res / ‖query‖, or 0 if the query is ~zero —
                # kept as a tensor (no ``.item()``) via a branchless ``where``.
                resid_t = torch.where(
                    par_norm < _FRACTION_EPSILON,
                    torch.zeros_like(res_flat),
                    res_flat / par_norm.clamp(min=_FRACTION_EPSILON),
                )
                # Tube-fit membership ``exp(−res²/2σ²)`` at the foot — both ``res``
                # and ``σ`` are raw reduced-space (un-whitened) distances, so the
                # ratio is unit-consistent.  No σ-field ⇒ no tube ⇒ full
                # membership (1.0), matching the legacy "no fuzziness" read.
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
            resid_t = resid_t.reshape(1)
            mem_t = mem_t.reshape(1)
            coord_t = pos_t.reshape(-1)[:n_dim]
            if coord_t.numel() < n_dim:
                coord_t = torch.cat(
                    [coord_t, coord_t.new_zeros(n_dim - coord_t.numel())],
                )

            layer_order.append(layer_idx)
            frac_terms.append(frac_t)
            resid_terms.append(resid_t)
            coord_terms.append(coord_t)
            frac_mean_t = (
                w * frac_t if frac_mean_t is None else frac_mean_t + w * frac_t
            )
            resid_mean_t = (
                w * resid_t if resid_mean_t is None else resid_mean_t + w * resid_t
            )
            coords_mean_t = (
                w * coord_t if coords_mean_t is None else coords_mean_t + w * coord_t
            )
            mem_mean_t = (
                w * mem_t if mem_mean_t is None else mem_mean_t + w * mem_t
            )

        requested_top_n = int(probe.top_n)
        top_n = requested_top_n if requested_top_n >= 0 else Kc + requested_top_n
        top_n = min(max(top_n, 0), Kc)
        nearest_dist_t: torch.Tensor | None = None
        nearest_idx_t: torch.Tensor | None = None
        if top_n and dist_acc_t is not None:
            nearest_dist_t, nearest_idx_t = torch.topk(
                dist_acc_t, k=top_n, largest=False, sorted=True,
            )
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
        # ran and the running means are populated.
        assert (
            frac_mean_t is not None
            and resid_mean_t is not None
            and coords_mean_t is not None
            and mem_mean_t is not None
        )
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
        """
        out: dict[str, ProbeReading] = {}
        if not hidden_per_layer or not self._probes:
            return out
        device = next(iter(hidden_per_layer.values())).device
        self._ensure_flat_cache(device)
        if self._flat_keys:
            out.update(self._score_flat_batched(hidden_per_layer))
        for name in self._curved_keys:
            out[name] = self._score_probe_full(self._probes[name], hidden_per_layer)
        # Defensive: any probe the cache didn't classify (it always should) is
        # scored per-probe so nothing silently drops from the readings.
        for name, probe in self._probes.items():
            if name not in out:
                out[name] = self._score_probe_full(probe, hidden_per_layer)
        # Preserve probe insertion order in the returned dict.
        return {name: out[name] for name in self._probes if name in out}

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
        * ``cols`` ``(P_L,)`` / ``coords_gidx`` ``(Σnd_L,)`` / ``ev_perdim`` /
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
        sig = tuple(self._probes.keys())
        if (
            self._flat_cache_device == device
            and self._flat_cache_sig == sig
            and self._flat_cache_wid == id(self._whitener)
        ):
            return
        self._flat_cache_device = device
        self._flat_cache_sig = sig
        self._flat_cache_wid = id(self._whitener)
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
        }

        layer_members: dict[int, list[int]] = {}
        for ci, n in enumerate(flat_names):
            for layer_idx in self._probes[n].manifold.layers:
                layer_members.setdefault(layer_idx, []).append(ci)

        cache: dict[int, dict[str, Any]] = {}
        for layer_idx, cis in layer_members.items():
            present = [flat_names[ci] for ci in cis]
            whs = [self._probes[n].whitened[layer_idx] for n in present]
            X, K_inv, lam = whs[0].X, whs[0].K_inv, whs[0].lam
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
            ev = torch.tensor(
                [float(self._probes[n].ev_weights.get(layer_idx, 0.0))
                 for n in present],
                device=device, dtype=torch.float32,
            )
            # scatter indices + padded node coords
            seg_ids_list: list[int] = []
            cq_scatter_list: list[int] = []
            coords_gidx_list: list[int] = []
            ev_perdim_list: list[float] = []
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
                ev_perdim_list.extend([li] * ndp)  # local index → ev gather
                nw = w.node_white.to(device)                       # (K_p, Rp)
                node_pad[li, :nw.shape[0], :Rp] = nw
            ev_perdim = ev[torch.tensor(
                ev_perdim_list, device=device, dtype=torch.long,
            )] if ev_perdim_list else ev.new_zeros(0)
            cache[layer_idx] = {
                "X": X, "K_inv": K_inv, "lam": lam,
                "basis_stack": basis_stack, "means_stack": means_stack,
                "gmean_stack": gmean_stack, "mm_stack": mm_stack,
                "Mblk": Mblk, "chol_blk": chol_blk,
                "coordS_blk": coordS_blk, "coordb_stack": coordb_stack,
                "ev": ev,
                "seg_ids": torch.tensor(seg_ids_list, device=device, dtype=torch.long),
                "cq_scatter": torch.tensor(cq_scatter_list, device=device, dtype=torch.long),
                "coords_gidx": torch.tensor(coords_gidx_list, device=device, dtype=torch.long),
                "ev_perdim": ev_perdim,
                "cols": torch.tensor(cis, device=device, dtype=torch.long),
                "cols_list": list(cis),
                "present_nd": present_nd,
                "node_pad": node_pad,
            }
        self._flat_layer_cache = cache

    def _score_flat_batched(
        self, hidden_per_layer: dict[int, torch.Tensor],
    ) -> dict[str, ProbeReading]:
        """Score the whole flat roster in one batched sweep + one host transfer.

        Phase 1 (per layer, batched over every flat probe): one ``Σ⁻¹h``
        Woodbury apply, stacked / block-diagonal matmuls for reduced coords,
        domain coords, fraction, and a padded ``(P_L, Kmax)`` nearest-distance
        norm — each scattered straight into global per-probe accumulators
        (``frac_acc`` / ``coords_acc`` / ``dist_acc`` / ``evsum``).  Phase 2
        (device): EV-normalize, one global ``topk`` sized to the largest requested
        ``top_n`` for nearest, and **one**
        ``.cpu()`` for the entire roster (means + per-layer traces + nearest).
        Phase 3 (host): slice the flat blob back into per-probe
        :class:`ProbeReading`s.  Values match the per-probe path (the aggregate
        uses) to float tolerance.
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
        evsum = torch.zeros(P, device=device, dtype=torch.float32)
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
            sih = _woodbury_apply(hf, ent["X"], ent["K_inv"], ent["lam"])
            h_sih = (hf * sih).sum()
            g_all = ent["basis_stack"] @ sih - ent["gmean_stack"]   # (ΣR,)
            c_all = ent["Mblk"] @ g_all                             # (ΣR,)
            cq_all = c_all @ ent["chol_blk"]                        # (ΣR,)
            coords_all = ent["coordS_blk"] @ c_all + ent["coordb_stack"]  # (Σnd_L,)
            hsm = ent["means_stack"] @ sih                          # (P_L,)
            cols = ent["cols"]
            n_present = cols.shape[0]
            par2 = torch.zeros(
                n_present, device=device, dtype=torch.float32,
            ).index_add_(0, ent["seg_ids"], g_all * c_all)          # (P_L,)
            x_m2 = (h_sih - 2.0 * hsm + ent["mm_stack"]).clamp(min=_FRACTION_EPSILON)
            frac = (par2.clamp(min=0.0).sqrt() / x_m2.sqrt()).clamp(0.0, 1.0)  # (P_L,)

            # Nearest: scatter cq into a padded (P_L, Rmax) query, batched norm.
            cq_pad = torch.zeros(n_present * Rmax, device=device, dtype=torch.float32)
            cq_pad = cq_pad.index_copy_(0, ent["cq_scatter"], cq_all).reshape(n_present, Rmax)
            dist = torch.linalg.vector_norm(
                ent["node_pad"] - cq_pad.unsqueeze(1), dim=-1,
            )                                                       # (P_L, Kmax)

            ev = ent["ev"]
            frac_acc.index_add_(0, cols, ev * frac)
            evsum.index_add_(0, cols, ev)
            coords_acc.index_add_(0, ent["coords_gidx"], ent["ev_perdim"] * coords_all)
            dist_acc.index_add_(0, cols, dist * ev.unsqueeze(1))
            trace_frac.append(frac)
            trace_coords.append(coords_all)
            trace_layers.append(layer_idx)
            seen.update(ent["cols_list"])

        # --- device finalize (EV-normalize + one bounded global topk) ---
        evsum_safe = evsum.clamp(min=_FRACTION_EPSILON)
        frac_final = frac_acc / evsum_safe
        coords_final = coords_acc / evsum_safe.repeat_interleave(
            gl["nd_counts"],
        ).clamp(min=_FRACTION_EPSILON)
        dist_final = (dist_acc / evsum_safe.unsqueeze(1)).masked_fill(
            ~gl["valid_mask"], float("inf"),
        )
        if topk_width:
            nd_sorted, ni_sorted = torch.topk(
                dist_final, k=topk_width, largest=False, sorted=True,
            )                                                       # (P, topk_width)
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
    ) -> dict[str, ProbeReading]:
        """Score every probe → ``{name: ProbeReading}`` (full reading).

        ``accumulate`` folds the cross-layer coords into history/stats (the
        in-flight per-token path passes False).
        """
        out = self._score_full(hidden_per_layer)
        if out and accumulate:
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
    ) -> dict[str, "ProbeReading"]:
        """Per-probe full reading for a single token (no accumulate).

        The live per-token read source: returns ``{name: ProbeReading}``
        with ``coords`` (domain frame) + ``fraction`` + ``nearest`` +
        ``residual`` and their per-layer traces, every probe (flat or curved).
        Does NOT touch history/stats — the in-flight gate/stream path must
        not corrupt the session-level accumulators — but flips the
        per-token pending flag so the TUI/webui can poll for a fresh reading.
        """
        out = self._score_tokens(hidden_per_layer, accumulate=False)
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
        larger means closer and ``@when:pad@happy > -0.1`` reads like a
        similarity gate).  Under the unified full reading every probe — flat
        and curved — carries coords *and* nearest, so flat probes now expose
        ``@label`` similarity gates too (e.g. ``@when:personas@hacker``) and
        the gate grammar is uniform.

        **Fuzzy channels** (additive — the ``@label`` distance gates are
        untouched): one ``"<probe>~<label>"`` per assigned node as the
        soft-assignment *probability* (``@when:personas~hacker > 0.5`` — a
        normalized, in-``[0,1]`` membership gate, unlike the unbounded
        ``-distance``), and ``"<probe>:membership"`` as the graded tube-fit
        (``@when:pad:membership > 0.6`` — high when the activation sits inside
        the manifold's learned thickness).  All shapes merge directly into
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

        Each token's full reading is computed (flat affine map + curved foot
        solve alike) and its cross-layer axis-0 coordinate extracted, so
        curved probes now carry a real per-token coord stream (not zeros).
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
            for name, reading in self._score_full(tok).items():
                per_token[name][i] = reading.coords[0] if reading.coords else 0.0
        return per_token

    def score_per_token(
        self,
        captured: dict[int, torch.Tensor],
        generated_ids: list[int],
        tokenizer: Any,
        *,
        accumulate: bool = True,
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

        from saklas.core.vectors import last_content_index
        agg_idx = last_content_index(generated_ids, tokenizer)
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


@dataclass
class AttachedManifoldProbe:
    """One manifold registered on a :class:`Monitor`.

    Pairs the loaded :class:`Manifold` artifact with the per-layer cache
    the monitor uses on the hot path: ``node_values_reduced`` is the
    per-layer ``(K, R)`` tensor of node activations in subspace coords —
    ``sub.node_coords`` for a flat fit (the M-projected node coords directly,
    pruned-basis-consistent), else ``(sub.eval_at(domain.embed(node_coords)) -
    sub.mean) @ sub.basis.T`` for a curved fit (RBF surface eval) —
    pre-computed once at attach so per-token distance computations are one
    batched cdist in R-dim per layer.  ``ev_weights`` is the per-layer EV
    ratio used to weight cross-layer aggregation; floored at
    :data:`_MIN_EV_WEIGHT` so a degenerate layer doesn't crash the
    aggregator.
    """

    name: str
    manifold: "Manifold"
    top_n: int = DEFAULT_NEAREST_TOP_N
    is_affine: bool = True
    # Whether the neutral anchor competes as a virtual candidate in this
    # probe's ``nearest`` ranking (see :data:`NEUTRAL_LABEL`).  Set at attach to
    # ``NEUTRAL_LABEL not in manifold.node_labels`` so a real node named
    # ``neutral`` keeps sole ownership of the label.
    inject_neutral: bool = True
    # Per-layer cache, indexed by layer index — same set of layers as
    # ``manifold.layers``.
    node_values_reduced: dict[int, torch.Tensor] = field(default_factory=dict)
    # Per-layer EV weight, normalized to sum to 1 across attached layers.
    ev_weights: dict[int, float] = field(default_factory=dict)
    # Per-candidate soft-assignment bandwidth ``τ`` ``(Kc,)`` in the **whitened**
    # metric (the metric the nearest-node distances live in), EV-weighted across
    # layers, candidate order = nodes then the neutral anchor (when injected).
    # A curved fit's within-node σ-field mapped into the whitened metric (×
    # ``√(tr(M_R)/R)``, the isotropic-σ scale); a flat fit's local layout scale
    # (each node's nearest-neighbor whitened distance).  Drives the
    # ``softmax(−d²/2τ² − R·log(τ))`` soft assignment.  ``None`` until the
    # post-attach bandwidth pass runs (an empty / degenerate manifold leaves it
    # ``None`` ⇒ assignment empty, argmax ``nearest`` unaffected).
    assign_bandwidth: torch.Tensor | None = None
    # Per-candidate Gaussian log-volume bias ``−R·log(τ_k)`` ``(Kc,)`` —
    # precomputed at attach so the hot-path logit is
    # ``−d²/(2τ²) + logvol_bias`` (one add, no per-token ``log``).  This is the
    # missing Gaussian normalization that turns the bare ``softmax(−d²/2τ²)``
    # into a proper isotropic R-D mixture posterior with a uniform node prior:
    # the bias penalizes diffuse-``τ`` candidates by their log-volume so a wide
    # node can't swallow probability from far away (the "broadest-node-wins"
    # pathology the bare form has).  ``R`` is the manifold's per-layer subspace
    # rank — the effective dimension of the space ``τ`` measures.
    assign_logvol_bias: torch.Tensor | None = None
    # Per-layer Mahalanobis bundle, populated at attach.  The wired whitener
    # must cover every layer of this manifold (all-or-nothing per probe), else
    # ``_build_whitened`` raises ``WhitenerError`` — there is no Euclidean
    # readout.  Each entry is ``_LayerWhiten`` — the precomputed factors that
    # turn the per-token fraction + nearest-node distance into their whitened
    # forms (M-orthogonal subspace projection + Mahalanobis distance).  Empty
    # only for an empty manifold (no layers to read).
    whitened: dict[int, "_LayerWhiten"] = field(default_factory=dict)


@dataclass
class _LayerWhiten:
    """Precomputed per-layer Mahalanobis factors for a manifold probe.

    Built at attach from the wired :class:`LayerWhitener` + the layer's
    subspace.  ``m_r_inv`` and ``chol`` are the ``(R, R)`` inverse and
    lower-Cholesky factor of the subspace-restricted inverse covariance
    ``M_R = B Σ⁻¹ Bᵀ`` (:meth:`LayerWhitener.subspace_gram`); ``node_white``
    is the ``(K, R)`` node coords transformed into the whitened metric
    (``v_reduced @ chol``) so a plain cdist against the transformed query
    yields the true Mahalanobis distance restricted to the subspace.
    ``(X, K_inv, lam)`` are the on-device Woodbury factors for the per-token
    ``Σ⁻¹ x`` apply.  All tensors live on the manifold's fit device.
    """

    m_r_inv: torch.Tensor      # (R, R) = (B Σ⁻¹ Bᵀ)⁻¹
    chol: torch.Tensor         # (R, R) lower-tri, M_R = chol @ cholᵀ
    node_white: torch.Tensor   # (K, R) = node_values_reduced @ chol
    # Neutral anchor in the same whitened metric as ``node_white`` (R,).  For a
    # neutral-anchored affine fit the frame origin *is* neutral, so this is the
    # zero vector; a curved fit centers on the PCA-frame mean, so it is the baked
    # per-layer ``origin`` (the authoring-coord foot of the neutral mean) mapped
    # through ``eval_at`` → basis → ``chol``.  The neutral candidate's distance
    # is then ``‖cdist_query − neutral_white‖``, identical machinery to a node.
    neutral_white: torch.Tensor  # (R,)
    mean: torch.Tensor         # (D,) fit mean on the scoring device
    basis: torch.Tensor        # (R, D) basis on the scoring device
    X: torch.Tensor            # (N, D) centered neutral observations
    K_inv: torch.Tensor        # (N, N) Woodbury inverse
    lam: float                 # ridge λ
    # Affine reduced→domain coordinate map (flat probes only; ``None`` for a
    # curved fit, which recovers coords through ``invert_parameterization``).
    # ``dom = c @ coord_S.T + coord_b`` sends the whitened M-orthogonal
    # reduced coords ``c = M_R⁻¹ B Σ⁻¹ x`` to the fit's domain frame, fit by
    # least squares so each node's reduced coord maps to its
    # ``node_coords`` (the rank-1 case reproduces the old slope/intercept).
    coord_S: torch.Tensor | None = None   # (n_dim, R)
    coord_b: torch.Tensor | None = None   # (n_dim,)


# ----------------------------------------------------------------------------
# Shared subspace-read machinery — used by both read paths of the unified
# :class:`Monitor`.  Flat (affine) probes get an analytic coordinate readout;
# curved probes foot-solve against the aggregate.  Both share the
# whitened-factor build, the per-layer geometry, and the attach-time node cache
# — the read-side peer of the steering split (one ``subspace_inject`` kernel,
# ``SteeringManager.{subspaces, manifolds}``).
# ----------------------------------------------------------------------------


def _probe_is_affine_for_manifold(manifold: "Manifold") -> bool:
    """True iff a manifold's fit is flat (affine) — batched coordinate readout.

    A manifold's layers are uniformly affine (``pca`` / 2-node concept) or
    curved (``spectral`` / ``authored``), so the first fitted layer decides.
    An empty manifold is treated as affine (it scores to nothing anyway).
    """
    for sub in manifold.layers.values():
        return bool(sub.is_affine)
    return True


def _probe_is_affine(probe: "AttachedManifoldProbe") -> bool:
    """Back-compat wrapper for tests / callers that ask about an attached probe."""
    return probe.is_affine


def _affine_coord_map(
    sub: Any, manifold: "Manifold", m_r_inv: torch.Tensor,
    X: torch.Tensor, K_inv: torch.Tensor, lam: float,
) -> "tuple[torch.Tensor, torch.Tensor]":
    """Least-squares affine map: whitened M-proj reduced coords → domain.

    Returns ``(coord_S (n_dim, R), coord_b (n_dim,))`` such that
    ``dom = c @ coord_Sᵀ + coord_b`` sends each node's whitened M-orthogonal
    reduced coords ``c_k = M_R⁻¹ B Σ⁻¹ (v_k − mean)`` to its domain
    ``node_coords`` — the rank-R generalization of the old rank-1 2-node
    slope/intercept (which it reproduces exactly at R=1, K=2).  The small
    solve runs on CPU (MPS has ``linalg.lstsq`` gaps); the per-node
    ``Σ⁻¹`` apply rides the device Woodbury.
    """
    dev = X.device
    sub_f = sub.to(device=dev, dtype=torch.float32)
    basis = sub_f.basis                                   # (R, D)
    mean = sub_f.mean                                     # (D,)
    R = int(basis.shape[0])
    node_coords = manifold.node_coords.to(device=dev, dtype=torch.float32)
    K = int(node_coords.shape[0])
    n_dim = int(manifold.domain.intrinsic_dim)
    embedded = manifold.domain.embed(
        manifold.domain.clamp_position(node_coords)
    ).to(device=dev, dtype=torch.float32)
    if sub_f.node_coords is not None:
        # True per-layer node activation in-subspace: ``node_coords @ basis``
        # (== ``eval_at`` − mean for a flat fit, but pruned-basis-consistent —
        # the shared domain layout would mis-dimension a DLS-pruned layer).
        v_centered = sub_f.node_coords.to(device=dev, dtype=torch.float32) @ basis
    else:
        v_centered = sub_f.eval_at(embedded) - mean      # (K, D) fallback
    si_vc = _woodbury_apply(v_centered, X, K_inv, lam)    # (K, D) = Σ⁻¹ v_centered
    g_nodes = si_vc @ basis.T                             # (K, R) = B Σ⁻¹ v_centered
    c_nodes = (g_nodes @ m_r_inv.T).cpu()                # (K, R) whitened M-proj coords
    dc = node_coords.reshape(K, -1)[:, :n_dim].cpu()     # (K, n_dim) domain target
    if K == 1:
        # Monopolar ray (R == 1): anchor through the origin (neutral → 0),
        # the single node → its domain coord; minimal-norm S, b = 0.
        c0 = c_nodes[0]
        denom = (c0 @ c0).clamp(min=_FRACTION_EPSILON)
        coord_S = torch.outer(dc[0], c0) / denom         # (n_dim, R)
        coord_b = torch.zeros(n_dim, dtype=torch.float32)
    else:
        ones = torch.ones((K, 1), dtype=torch.float32)
        c1 = torch.cat([c_nodes, ones], dim=1)           # (K, R+1)
        sol = torch.linalg.lstsq(c1, dc).solution        # (R+1, n_dim)
        coord_S = sol[:R].T.contiguous()                 # (n_dim, R)
        coord_b = sol[R].contiguous()                    # (n_dim,)
    return (
        coord_S.to(device=dev, dtype=torch.float32),
        coord_b.to(device=dev, dtype=torch.float32),
    )


def _build_whitened_factors(
    whitener: Any, probe: "AttachedManifoldProbe",
    *,
    factor_cache: dict[
        tuple[int, str, torch.dtype],
        tuple[torch.Tensor, torch.Tensor, float],
    ] | None = None,
) -> dict[int, "_LayerWhiten"]:
    """Per-layer :class:`_LayerWhiten` map for a probe (Mahalanobis-only).

    The wired whitener is **required** and must cover **every** fit layer
    (all-or-nothing per probe).  A missing/non-covering whitener raises
    :class:`WhitenerError` — there is no Euclidean readout.  Off the hot
    path; runs once at attach / on ``set_whitener``.
    """
    manifold = probe.manifold
    if not manifold.layers:
        return {}
    layers = list(manifold.layers.keys())
    if whitener is None or not whitener.covers_all(layers):
        raise WhitenerError(
            "subspace probe reads require a Mahalanobis whitener covering "
            f"every fit layer {sorted(layers)}; regenerate the neutral "
            "activation cache for this model (the Euclidean path is gone)"
        )
    out: dict[int, _LayerWhiten] = {}
    for layer_idx, sub in manifold.layers.items():
        v_reduced = probe.node_values_reduced.get(layer_idx)
        if v_reduced is None:
            raise WhitenerError(
                f"subspace probe cache missing reduced node coords for "
                f"layer {layer_idx}; rebuild the probe before scoring"
            )
        dev = v_reduced.device
        basis = sub.basis.to(device=torch.device("cpu"), dtype=torch.float32)
        # M_R = B Σ⁻¹ Bᵀ (R, R), PD for an orthonormal B and ridge-PD Σ.
        m_r = whitener.subspace_gram(layer_idx, basis)  # CPU fp32
        R = m_r.shape[0]
        try:
            chol = torch.linalg.cholesky(m_r)
        except torch.linalg.LinAlgError:
            # Defensive jitter for a near-singular subspace gram.
            eye = torch.eye(R, dtype=m_r.dtype)
            jitter = 1e-8 * float(m_r.diagonal().mean().clamp_min(1e-12))
            chol = torch.linalg.cholesky(m_r + jitter * eye)
        m_r_inv = torch.cholesky_inverse(chol)
        cache_key = (layer_idx, str(dev), torch.float32)
        if factor_cache is not None and cache_key in factor_cache:
            X, K_inv, lam = factor_cache[cache_key]
        else:
            X, K_inv, lam = whitener.woodbury_factors(
                layer_idx, device=dev, dtype=torch.float32,
            )
            if factor_cache is not None:
                factor_cache[cache_key] = (X, K_inv, lam)
        chol_dev = chol.to(device=dev, dtype=torch.float32)
        m_r_inv_dev = m_r_inv.to(device=dev, dtype=torch.float32)
        # Flat probes carry the affine reduced→domain coord map (the rank-R
        # generalization of the rank-1 slope/intercept); a curved fit leaves
        # it ``None`` and recovers coords via ``invert_parameterization``.
        coord_S = coord_b = None
        if sub.is_affine:
            coord_S, coord_b = _affine_coord_map(
                sub, manifold, m_r_inv_dev, X, K_inv, lam,
            )
        # Neutral anchor in the whitened metric.  Affine: the frame is
        # neutral-anchored, so neutral is reduced coord 0.  Curved: map the
        # baked per-layer ``origin`` (authoring-coord foot of the neutral mean)
        # through the same eval_at → basis → chol pipeline as a node.  A curved
        # fit without a stored origin falls back to the subspace mean (0).
        if sub.is_affine:
            neutral_white = torch.zeros(R, device=dev, dtype=torch.float32)
        else:
            o = manifold.origin.get(layer_idx)
            if o is None:
                neutral_white = torch.zeros(R, device=dev, dtype=torch.float32)
            else:
                o_dom = o.to(device=dev, dtype=torch.float32).reshape(1, -1)
                emb = manifold.domain.embed(
                    manifold.domain.clamp_position(o_dom)
                ).to(device=dev, dtype=torch.float32)
                v_centered = sub.eval_at(emb) - sub.mean.to(
                    device=dev, dtype=torch.float32,
                )
                v_red = v_centered @ sub.basis.to(
                    device=dev, dtype=torch.float32,
                ).T                                  # (1, R)
                neutral_white = (v_red @ chol_dev).reshape(-1)
        out[layer_idx] = _LayerWhiten(
            m_r_inv=m_r_inv_dev,
            chol=chol_dev,
            node_white=(v_reduced.to(torch.float32) @ chol_dev),
            neutral_white=neutral_white,
            mean=sub.mean.to(device=dev, dtype=torch.float32),
            basis=sub.basis.to(device=dev, dtype=torch.float32),
            X=X, K_inv=K_inv, lam=lam,
            coord_S=coord_S, coord_b=coord_b,
        )
    return out


def _attach_manifold_probe(
    name: str,
    manifold: "Manifold",
    *,
    top_n: int = DEFAULT_NEAREST_TOP_N,
    whitener: Any = None,
    factor_cache: dict[
        tuple[int, str, torch.dtype],
        tuple[torch.Tensor, torch.Tensor, float],
    ] | None = None,
) -> "AttachedManifoldProbe":
    """Build an :class:`AttachedManifoldProbe`: node values + EV weights + whitened.

    Pre-caches, once at attach, the per-layer reduced ``(K, R)`` node
    activations (hot-path cdist working space) and the normalized per-layer
    EV weights, then the Mahalanobis bundle via :func:`_build_whitened_factors`
    (the whitener must cover the fit's layers).
    """
    if not manifold.layers:
        raise ValueError(f"manifold {manifold.name!r} carries no fitted layers")
    if manifold.node_coords.numel() == 0 or not manifold.node_labels:
        raise ValueError(
            f"manifold {manifold.name!r} carries no node coords / labels"
        )
    node_values_reduced: dict[int, torch.Tensor] = {}
    ev_weights_raw: dict[int, float] = {}
    coords = manifold.node_coords.to(torch.float32)
    clamped = manifold.domain.clamp_position(coords)
    embedded = manifold.domain.embed(clamped)  # (K, m)
    for layer_idx, sub in manifold.layers.items():
        sub_f32 = sub.to(device=sub.mean.device, dtype=torch.float32)
        if sub_f32.is_affine and sub_f32.node_coords is not None:
            # Flat fit: the per-layer reduced node coords ARE the cdist working
            # space — analytically ``c_k = node_coords`` in the same M-projected
            # frame the running-activation query uses.  They're already pruned
            # in lockstep with a DLS-pruned basis, whereas feeding the shared
            # (intrinsic_dim) domain layout through ``eval_at`` mis-dimensions a
            # pruned layer (basis rank R < intrinsic_dim ⇒ ``embedded @ basis``
            # shape mismatch) *and* conflates the abstract layout frame with the
            # per-layer activation frame.
            v_reduced = sub_f32.node_coords.to(torch.float32)  # (K, R)
        else:
            embedded_dev = embedded.to(
                device=sub_f32.mean.device, dtype=torch.float32,
            )
            v_world = sub_f32.eval_at(embedded_dev)  # (K, D)
            v_centered = v_world - sub_f32.mean
            v_reduced = v_centered @ sub_f32.basis.T  # (K, R)
        node_values_reduced[layer_idx] = v_reduced.contiguous()
        ev_weights_raw[layer_idx] = float(
            manifold.explained_variance.get(layer_idx, 1.0)
        )
    total = sum(max(_MIN_EV_WEIGHT, w) for w in ev_weights_raw.values())
    if total <= 0.0:
        n_layers = max(1, len(ev_weights_raw))
        ev_weights = dict.fromkeys(ev_weights_raw, 1.0 / n_layers)
    else:
        ev_weights = {
            idx: max(_MIN_EV_WEIGHT, w) / total
            for idx, w in ev_weights_raw.items()
        }
    probe = AttachedManifoldProbe(
        name=name,
        manifold=manifold,
        top_n=int(top_n),
        is_affine=_probe_is_affine_for_manifold(manifold),
        inject_neutral=NEUTRAL_LABEL not in manifold.node_labels,
        node_values_reduced=node_values_reduced,
        ev_weights=ev_weights,
    )
    probe.whitened = _build_whitened_factors(
        whitener, probe, factor_cache=factor_cache,
    )
    bw, lvb = _compute_assign_bandwidth(probe, embedded)
    probe.assign_bandwidth = bw
    probe.assign_logvol_bias = lvb
    return probe


def _compute_assign_bandwidth(
    probe: "AttachedManifoldProbe", embedded: torch.Tensor,
) -> "tuple[torch.Tensor | None, torch.Tensor | None]":
    """Per-candidate ``(τ, −R·log(τ))`` ``(Kc,)`` each, EV-weighted.

    ``τ`` is the soft-assignment bandwidth in the **whitened** metric the
    nearest-node distances use, candidate order = nodes then the neutral anchor
    (when ``inject_neutral``).  Two sources for ``τ``, per the chosen within-
    node-thickness variance:

    * **curved + σ-field**: each node's reduced within-node thickness
      ``σ(z)`` (:meth:`LayerSubspace.sigma_at`) mapped into the whitened metric by
      the isotropic scale ``√(tr(M_R)/R)`` (``M_R = chol cholᵀ``), so a
      reduced-space σ is comparable to the chol-whitened distances; the neutral
      anchor takes the per-layer median node σ.
    * **flat / no σ-field**: each node's *local layout scale* — its
      nearest-neighbor distance among the whitened node coords ``node_white`` —
      so a dense cluster assigns sharply and an isolated node softly; the neutral
      anchor takes its own nearest-node distance.

    The second return ``−R·log(τ)`` is the precomputed Gaussian log-volume
    bias for the soft-assignment logit: the proper isotropic R-D Gaussian
    posterior is ``softmax(−d²/(2τ²) − R·log(τ))`` (the ``−R·log(τ)`` is the
    log of the normalization ``(2πτ²)^(-R/2)`` with the constant dropped — it
    cancels in softmax).  Without this term the bare ``−d²/2τ²`` softmax has a
    *broadest-node-wins* pathology: a wide-``τ`` candidate's logit sits near 0
    while crisp-``τ`` candidates have strongly-negative logits, so the diffuse
    node swallows probability regardless of distance.  Precomputed once at
    attach so the hot path is a single add.  ``R`` is the manifold's per-layer
    subspace rank (one number per probe — fits are rank-uniform across layers).

    EV-weighted across layers (same weights as every other cross-layer read),
    floored positive.  Returns ``(None, None)`` for a degenerate manifold (no
    layers / single node), which leaves the assignment empty without disturbing
    ``nearest``.
    """
    shared = list(probe.manifold.layers.keys())
    if not shared:
        return None, None
    K = probe.node_values_reduced[shared[0]].shape[0]
    if K < 1:
        return None, None
    ev = probe.ev_weights
    acc: torch.Tensor | None = None
    wsum = 0.0
    for layer_idx in shared:
        wh = probe.whitened.get(layer_idx)
        if wh is None:
            continue
        sub = probe.manifold.layers[layer_idx]
        node_white = wh.node_white.to(torch.float32)        # (K, R)
        R = int(node_white.shape[1])
        if (not sub.is_affine) and sub.has_sigma:
            # σ-field (reduced units) → whitened via the isotropic scale.
            emb = embedded.to(device=node_white.device, dtype=torch.float32)
            sig_reduced = sub.sigma_at(emb).reshape(-1).to(torch.float32)  # (K,)
            m_r = wh.chol @ wh.chol.transpose(-1, -2)        # M_R (R, R)
            scale = float((torch.diagonal(m_r).sum() / max(R, 1)).clamp(min=1e-12).sqrt())
            band = sig_reduced * scale                       # (K,) whitened
            neutral_band = band.median()
        else:
            # Local layout scale: each node's nearest-neighbor whitened distance.
            if K >= 2:
                dmat = torch.cdist(node_white, node_white)   # (K, K)
                dmat = dmat + torch.eye(
                    K, device=dmat.device, dtype=dmat.dtype,
                ) * 1e9                                       # mask self
                band = dmat.min(dim=1).values                # (K,)
            else:
                band = node_white.new_ones(K)
            if probe.inject_neutral:
                nd = torch.linalg.vector_norm(
                    node_white - wh.neutral_white.reshape(1, -1), dim=-1,
                )                                            # (K,)
                neutral_band = nd.min()
            else:
                neutral_band = band.median() if K else node_white.new_tensor(1.0)
        cand = (
            torch.cat([band, neutral_band.reshape(1)])
            if probe.inject_neutral else band
        )                                                    # (Kc,)
        w = float(ev.get(layer_idx, 0.0))
        acc = cand * w if acc is None else acc + cand * w
        wsum += w
    if acc is None:
        return None, None
    if wsum > _MIN_EV_WEIGHT:
        acc = acc / wsum
    # Floor positive so the softmax denominator never divides by ~0.
    med = float(acc.median().clamp(min=1e-6))
    tau = acc.clamp(min=1e-3 * med).to(torch.float32)
    # Gaussian log-volume bias ``−R·log(τ)`` for the soft-assignment logit.
    # ``R`` = the manifold's per-layer subspace rank (rank-uniform across a
    # fit's layers), the effective dimension of the space the bandwidth ``τ``
    # lives in.  Precomputed here so the hot path adds a single scalar per
    # candidate with no per-token ``log()``.
    R = int(next(iter(probe.manifold.layers.values())).rank)
    logvol_bias = (-float(R) * torch.log(tau)).to(torch.float32)
    return tau, logvol_bias


def _layer_geometry(
    probe: "AttachedManifoldProbe",
    layer_idx: int,
    h: torch.Tensor,
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]":
    """Per-layer readout pieces, shared by the per-token + aggregate paths.

    Returns ``(frac, cdist_query, invert_query, cdist_nodes)``:

    * ``frac`` — scalar tensor in ``[0, 1]``, the whitened **M-orthogonal**
      in-subspace energy share ``sqrt(gᵀ M_R⁻¹ g) / ‖x‖_M`` with
      ``g = B Σ⁻¹ x``.
    * ``cdist_query`` — ``(1, R)`` query (``Lᵀc``) so a plain cdist against
      ``cdist_nodes`` is the Mahalanobis distance.
    * ``invert_query`` — ``(R,)`` M-orthogonal projection coords
      ``c = M_R⁻¹ g`` for :func:`invert_parameterization`.
    * ``cdist_nodes`` — ``(K, R)`` whitened node coords.
    """
    wh = probe.whitened.get(layer_idx)
    if wh is None:
        raise WhitenerError(
            f"subspace probe read missing whitened factors for layer "
            f"{layer_idx}; rebuild the probe (the Euclidean path is gone)"
        )
    mean = wh.mean
    basis = wh.basis
    x = h - mean
    sx = _woodbury_apply(x, wh.X, wh.K_inv, wh.lam)  # Σ⁻¹ x  (D,)
    x_mnorm = torch.sqrt(
        (x * sx).sum().clamp(min=0.0)
    ).clamp(min=_FRACTION_EPSILON)
    g = basis @ sx                       # (R,) = B Σ⁻¹ x
    c = wh.m_r_inv @ g                   # (R,) = M_R⁻¹ g  (M-proj coords)
    par_mnorm = torch.sqrt((g * c).sum().clamp(min=0.0))  # ‖P_M x‖_M
    frac = (par_mnorm / x_mnorm).clamp(min=0.0, max=1.0)
    cdist_query = (c.reshape(1, -1) @ wh.chol)  # (1, R) — Lᵀc as row
    return frac, cdist_query, c, wh.node_white
