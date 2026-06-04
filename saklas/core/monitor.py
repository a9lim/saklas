from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from saklas.core.mahalanobis import WhitenerError
from saklas.core.manifold import invert_parameterization
from saklas.core.results import ManifoldAggregate, ManifoldTokenReading

if TYPE_CHECKING:
    from saklas.core.manifold import Manifold

_MAX_HISTORY = 8

_EMPTY_STATS = {"count": 0, "sum": 0.0, "sum_sq": 0.0,
                "min": float("inf"), "max": float("-inf")}

# Default top-N nearest-node count for manifold probes.  Per-probe
# override available on ``ManifoldMonitor.add_probe``.
DEFAULT_NEAREST_TOP_N: int = 3

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

    Shared by :class:`TraitMonitor` and :class:`ManifoldMonitor`.  Kept a
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


class TraitMonitor:
    """Reads flat-subspace probes as whitened coordinates (the unified readout).

    Each probe is a flat (affine) :class:`~saklas.core.manifold.Manifold` — a
    2-node concept axis is the rank-1 case, a multi-node discover fit
    (``personas`` / ``cultural`` / ``register``) the rank-R case.  Per token
    the monitor reports the same shape :class:`ManifoldMonitor` does —
    ``coords`` (the whitened in-subspace position in the fit's *domain*
    frame, EV-weighted across layers), ``fraction`` (the in-subspace energy
    share ∈ [0, 1]), and ``nearest`` (top-N node labels) — but takes the
    **flat fast path**: the per-layer subspace inverse is affine, so coords
    are exact and cheap enough to fill per token, and every rank-1 probe is
    scored in one stacked matmul per layer (no per-probe Python loop on the
    hot path).

    Coordinates are **domain-frame**, matching
    :meth:`ManifoldMonitor.score_aggregate` — which inverts each layer's
    reduced activation coordinate into the shared domain *before*
    EV-averaging, because raw per-layer coords live in per-layer ``‖δ_L‖``
    units and don't average coherently.  At rank-1 this is the
    pole-normalized coordinate: ``1.0`` sits at the positive node, signed,
    unbounded past it.

    The Mahalanobis whitener is mandatory: every probed layer must be
    covered (``covers_all``) or :meth:`_ensure_cache` raises
    :class:`WhitenerError` — there is no Euclidean readout (on real LMs the
    Euclidean metric is rogue-dominated, a wrong answer not a degraded one).

    TUI-facing scalar helpers (``get_stats`` / ``get_sparkline`` /
    ``get_current_and_previous``) report coordinate **axis 0** so the
    untouched trait panel keeps working; the full per-axis data flows
    through the :class:`ManifoldTokenReading` / :class:`ManifoldAggregate`
    surfaces.
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
            init-signature parity with :class:`ManifoldMonitor`; the
            coordinate readout centers on each fit's own ``LayerSubspace.mean``,
            not the global layer mean, so this is not consulted on the hot
            path — only exposed via the ``layer_means`` property).
        whitener: the :class:`~saklas.core.mahalanobis.LayerWhitener`;
            mandatory at scoring time (covers every probed layer or raise).
        """
        self._probes: dict[str, AttachedManifoldProbe] = {}
        self._layer_means: dict[int, torch.Tensor] = dict(layer_means) if layer_means else {}
        self._whitener: Any = whitener

        # Per-layer batched cache for the rank-1 fast path, keyed by layer.
        # Each entry stacks every rank-1 probe covering that layer so one
        # matmul yields all their reduced coords:
        #   A[P1, D]      : Σ⁻¹ d̂_p              (so g = A @ h − cmean)
        #   cmean[P1]     : d̂_p · Σ⁻¹ mean_p     (= d̂_p Σ⁻¹ mean_p)
        #   mR[P1]        : d̂_p · Σ⁻¹ d̂_p        (= ‖d̂_p‖²_M)
        #   simean[P1, D] : Σ⁻¹ mean_p           (fraction cross term h·Σ⁻¹mean)
        #   mm[P1]        : mean_p · Σ⁻¹ mean_p   (fraction const)
        #   slope[P1], intercept[P1] : affine reduced→domain map (rank-1)
        #   ev[P1]        : per-layer EV weight, normalized mean-1 per probe
        #   X, K, lam     : shared Woodbury factors for Σ⁻¹ h (fraction denom)
        #   probe_idx[P1] : column → index into self._fast_keys
        # Built lazily; invalidated on probe add/remove or whitener change.
        self._fast_cache: dict[int, dict[str, Any]] = {}
        # Insertion-ordered rank-1 probe names (the fast-path column order).
        self._fast_keys: tuple[str, ...] = ()
        # Rank-R (>1) probe names — scored on the per-probe slow path.
        self._slow_keys: tuple[str, ...] = ()
        self._cache_device: torch.device | None = None
        self._cache_probe_keys: tuple[str, ...] = ()
        self._cache_whitener_id: int | None = id(whitener)

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
        a probe reads (peer of :meth:`ManifoldMonitor.attached_layers`).
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
        is a no-op.  Any change flushes the batched fast cache and rebuilds
        each attached probe's per-layer whitened factors against the new
        covariance (manifold reads are Mahalanobis-only, so a whitener that
        doesn't cover an attached probe's fit layers raises here).
        """
        if whitener is self._whitener:
            return
        self._whitener = whitener
        self._invalidate_cache()
        self._cache_whitener_id = id(whitener)
        for probe in self._probes.values():
            probe.whitened = _build_whitened_factors(whitener, probe)

    def _invalidate_cache(self) -> None:
        self._fast_cache = {}
        self._fast_keys = ()
        self._slow_keys = ()
        self._cache_device = None
        self._cache_probe_keys = ()

    def _ensure_cache(self, device: torch.device) -> None:
        """Build/refresh the batched rank-1 fast cache on ``device``.

        Every rank-1 (2-node concept) probe covering a layer is stacked so
        the whole roster's reduced coordinates come from one matmul per
        layer; rank-R (>1) probes go to ``_slow_keys`` and score per-probe.
        Per (layer, stacked probe) the cache precomputes, off the hot path:

          A[P1ₗ, D]      = Σ⁻¹ d̂          (so g = A @ h − cmean)
          SImean[P1ₗ, D] = Σ⁻¹ mean        (fraction cross term h·Σ⁻¹mean)
          cmean[P1ₗ]     = d̂ · Σ⁻¹ mean
          mR[P1ₗ]        = d̂ · Σ⁻¹ d̂      (= ‖d̂‖²_M; reduced coord = g/mR)
          mm[P1ₗ]        = mean · Σ⁻¹ mean (fraction const)
          slope/intercept[P1ₗ] : affine reduced→domain map (the two nodes)
          ev[P1ₗ]        : per-layer EV weight
          cols[P1ₗ]      : column → global ``_fast_keys`` index
          X, K, lam      : shared Woodbury factors (Σ⁻¹ h for the fraction)

        The Mahalanobis whitener must cover every probed layer or this
        raises :class:`WhitenerError` (no Euclidean readout).
        """
        probe_keys = tuple(self._probes.keys())
        if (
            self._cache_device == device
            and self._cache_probe_keys == probe_keys
            and self._cache_whitener_id == id(self._whitener)
            and (self._fast_cache or not self._fast_keys)
        ):
            return

        self._cache_device = device
        self._cache_whitener_id = id(self._whitener)
        self._cache_probe_keys = probe_keys
        whitener = self._whitener

        all_layers = sorted({
            L for p in self._probes.values() for L in p.manifold.layers
        })
        if all_layers and (
            whitener is None or not whitener.covers_all(all_layers)
        ):
            raise WhitenerError(
                "probe scoring requires a Mahalanobis whitener covering every "
                f"probed layer {all_layers}; regenerate the neutral activation "
                "cache for this model (the Euclidean path is gone)"
            )

        fast_names = [
            n for n, p in self._probes.items()
            if p.manifold.layers and _probe_rank(p) == 1
        ]
        slow_names = [
            n for n, p in self._probes.items()
            if p.manifold.layers and _probe_rank(p) > 1
        ]
        self._fast_keys = tuple(fast_names)
        self._slow_keys = tuple(slow_names)

        # layer -> list of (global fast-col idx) covering it.
        layer_members: dict[int, list[int]] = {}
        for ci, name in enumerate(fast_names):
            for L in self._probes[name].manifold.layers:
                layer_members.setdefault(L, []).append(ci)

        new_cache: dict[int, dict[str, Any]] = {}
        for L, cols in layer_members.items():
            X, K, lam = whitener.woodbury_factors(
                L, device=device, dtype=torch.float32,
            )
            P1 = len(cols)
            first_sub = self._probes[fast_names[cols[0]]].manifold.layers[L]
            D = first_sub.basis.shape[-1]
            A = torch.zeros((P1, D), device=device, dtype=torch.float32)
            SImean = torch.zeros((P1, D), device=device, dtype=torch.float32)
            cmean = torch.zeros((P1,), device=device, dtype=torch.float32)
            mR = torch.ones((P1,), device=device, dtype=torch.float32)
            mm = torch.zeros((P1,), device=device, dtype=torch.float32)
            slope = torch.zeros((P1,), device=device, dtype=torch.float32)
            intercept = torch.zeros((P1,), device=device, dtype=torch.float32)
            ev = torch.zeros((P1,), device=device, dtype=torch.float32)
            for row, ci in enumerate(cols):
                probe = self._probes[fast_names[ci]]
                manifold = probe.manifold
                sub = manifold.layers[L]
                d = sub.basis.reshape(-1).to(device=device, dtype=torch.float32)
                mean = sub.mean.reshape(-1).to(device=device, dtype=torch.float32)
                sd = _woodbury_apply(d, X, K, lam)      # Σ⁻¹ d̂
                sm = _woodbury_apply(mean, X, K, lam)    # Σ⁻¹ mean
                A[row] = sd
                SImean[row] = sm
                cmean[row] = torch.dot(d, sm)
                mR_row = torch.dot(d, sd).clamp(min=1e-12)
                mR[row] = mR_row
                mm[row] = torch.dot(mean, sm)
                # Affine reduced→domain map.  The read coordinate ``c =
                # M_R⁻¹ B Σ⁻¹ x`` is the *whitened* (GLS) reduced coord, so
                # the reference ``rc`` must be each node's whitened read
                # coord — NOT the Euclidean ``sub.node_coords`` (the two
                # frames coincide only under isotropic Σ; on a real LM they
                # differ).  ``dom = slope·c + intercept`` then sends node k
                # to its domain coord ``dc[k]`` exactly.
                emb = manifold.domain.embed(
                    manifold.domain.clamp_position(
                        manifold.node_coords.to(torch.float32)
                    )
                ).to(device=device, dtype=torch.float32)
                x_nodes = sub.eval_at(emb).to(torch.float32) - mean  # (K, D)
                rc = (x_nodes @ sd) / mR_row                          # (K,) whitened
                ncd = manifold.node_coords
                dc = ncd.reshape(ncd.shape[0], -1)[:, 0].to(torch.float32)
                if rc.shape[0] >= 2:
                    # Bipolar (2-node) concept: line through the two nodes.
                    drc = float(rc[0] - rc[1])
                    if abs(drc) < 1e-12:
                        slope[row] = 1.0
                        intercept[row] = 0.0
                    else:
                        s = (dc[0] - dc[1]) / (rc[0] - rc[1])
                        slope[row] = s
                        intercept[row] = dc[0] - s * rc[0]
                else:
                    # Monopolar ray (1-node fold, e.g. ``agentic`` or an ad-hoc
                    # ``probe()`` direction): neutral is the reduced-coord-0
                    # origin → domain 0, the single pole node → ``dc[0]``.
                    r0 = float(rc[0])
                    if abs(r0) < 1e-12:
                        slope[row] = 1.0
                    else:
                        slope[row] = dc[0] / rc[0]
                    intercept[row] = 0.0
                ev[row] = float(probe.ev_weights.get(L, 1.0))
            new_cache[L] = {
                "A": A, "SImean": SImean, "cmean": cmean, "mR": mR, "mm": mm,
                "slope": slope, "intercept": intercept, "ev": ev,
                "cols": torch.tensor(cols, device=device, dtype=torch.long),
                "X": X, "K": K, "lam": lam,
            }
        self._fast_cache = new_cache

    def _score_fast(
        self, hidden_per_layer: dict[int, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rank-1 fast path: ``(domain_coords[P1], fraction[P1])`` on-device.

        One matmul + one Woodbury apply per layer; EV-weighted across layers
        (re-normalized by the realized weight so a missing layer doesn't
        skew the result).  Zero ``.item()`` / ``.cpu()`` — the hot-path
        no-sync primitive.  ``coords`` is the domain-frame coordinate
        (pole-normalized at rank-1), ``fraction`` the in-subspace energy
        share in ``[0, 1]``.
        """
        device = self._cache_device or torch.device("cpu")
        P1 = len(self._fast_keys)
        coords = torch.zeros((P1,), device=device, dtype=torch.float32)
        frac = torch.zeros((P1,), device=device, dtype=torch.float32)
        evsum = torch.zeros((P1,), device=device, dtype=torch.float32)
        if P1 == 0:
            return coords, frac
        for L, h in hidden_per_layer.items():
            entry = self._fast_cache.get(L)
            if entry is None:
                continue
            hf = h.reshape(-1).to(torch.float32) if h.ndim > 1 else h.to(torch.float32)
            cols = entry["cols"]
            ev = entry["ev"]
            g = entry["A"] @ hf - entry["cmean"]            # reduced·mR
            mRL = entry["mR"]
            c = g / mRL                                     # reduced coord
            dom = entry["slope"] * c + entry["intercept"]   # domain coord
            coords.index_add_(0, cols, ev * dom)
            evsum.index_add_(0, cols, ev)
            # Fraction: ‖P_M x‖_M / ‖x‖_M.
            sih = _woodbury_apply(hf, entry["X"], entry["K"], entry["lam"])
            hSih = torch.dot(hf, sih).clamp(min=0.0)        # hᵀΣ⁻¹h
            hsm = entry["SImean"] @ hf                      # h·Σ⁻¹mean
            xM2 = (hSih - 2.0 * hsm + entry["mm"]).clamp(min=1e-12)
            par = g.abs() / mRL.sqrt()                      # ‖P_M x‖_M = |g|/√mR
            fL = (par / xM2.sqrt()).clamp(0.0, 1.0)
            frac.index_add_(0, cols, ev * fL)
        evsum = evsum.clamp(min=1e-8)
        return coords / evsum, frac / evsum

    def _score_slow(
        self, name: str, hidden_per_layer: dict[int, torch.Tensor],
    ) -> ManifoldTokenReading:
        """Rank-R per-probe path: domain coords + fraction + nearest.

        Mirrors :meth:`ManifoldMonitor.score_aggregate`'s per-layer
        invert-then-EV-mean, run on the single pooled hidden state.  Used
        for the (rare) multi-node flat probes attached to this monitor;
        those force full-retention capture so they never ride the no-sync
        incremental row.
        """
        probe = self._probes[name]
        manifold = probe.manifold
        shared = [L for L in manifold.layers if L in hidden_per_layer]
        if not shared:
            return ManifoldTokenReading(fraction=0.0, nearest=[], coords=())
        total_w = sum(probe.ev_weights.get(L, 0.0) for L in shared)
        if total_w <= _MIN_EV_WEIGHT:
            w_shared = {L: 1.0 / len(shared) for L in shared}
        else:
            w_shared = {L: probe.ev_weights.get(L, 0.0) / total_w for L in shared}

        K = probe.node_values_reduced[shared[0]].shape[0]
        n_dim = manifold.domain.intrinsic_dim
        dist_acc = [0.0] * K
        frac_mean = 0.0
        coords_mean: list[float] | None = None
        for L in shared:
            sub = manifold.layers[L]
            h = hidden_per_layer[L]
            h = h.reshape(-1, h.shape[-1])[-1] if h.ndim > 1 else h
            frac_t, cdist_query, invert_query, cdist_nodes = _layer_geometry(
                probe, L, h,
            )
            w = w_shared[L]
            frac_mean += w * float(frac_t.item())
            dists = torch.cdist(cdist_query, cdist_nodes).reshape(-1).cpu().tolist()
            for k in range(K):
                dist_acc[k] += w * dists[k]
            # ``invert_parameterization`` recovers domain coords from the
            # reduced query, but raises on a flat (affine) subspace — its
            # RBF path has no affine branch.  A flat rank-R probe therefore
            # reports fraction + nearest with empty coords (the affine
            # multi-dim coordinate invert is a follow-up); a curved probe
            # gets full coords.
            try:
                pos, _res = invert_parameterization(
                    sub, manifold.domain, invert_query.reshape(1, -1),
                    manifold.node_coords,
                )
            except ValueError:
                continue
            coord_tup = [float(c) for c in pos.reshape(-1).tolist()[:n_dim]]
            if coords_mean is None:
                coords_mean = [w * c for c in coord_tup]
            else:
                for i, c in enumerate(coord_tup):
                    if i < len(coords_mean):
                        coords_mean[i] += w * c
        order = sorted(range(K), key=lambda k: dist_acc[k])
        nearest = [(manifold.node_labels[k], dist_acc[k]) for k in order[: probe.top_n]]
        return ManifoldTokenReading(
            fraction=frac_mean,
            nearest=nearest,
            coords=tuple(coords_mean or ()),
        )

    def _score_tokens(
        self,
        hidden_per_layer: dict[int, torch.Tensor],
        accumulate: bool = True,
    ) -> dict[str, ManifoldTokenReading]:
        """Score every probe → ``{name: ManifoldTokenReading}``.

        Fast (rank-1) probes are batched; slow (rank-R) probes scored
        per-probe.  ``accumulate`` folds the aggregate coords into
        history/stats (the in-flight per-token path passes False).
        """
        out: dict[str, ManifoldTokenReading] = {}
        if not hidden_per_layer or not self._probes:
            for name in self._probes:
                out[name] = ManifoldTokenReading(fraction=0.0, nearest=[], coords=())
            if accumulate:
                self._apply_accumulate(out)
            return out

        device = next(iter(hidden_per_layer.values())).device
        self._ensure_cache(device)

        if self._fast_keys:
            coords_t, frac_t = self._score_fast(hidden_per_layer)
            coords_l = coords_t.cpu().tolist()
            frac_l = frac_t.cpu().tolist()
            for i, name in enumerate(self._fast_keys):
                out[name] = ManifoldTokenReading(
                    fraction=frac_l[i], nearest=[], coords=(coords_l[i],),
                )
        for name in self._slow_keys:
            out[name] = self._score_slow(name, hidden_per_layer)

        if accumulate:
            self._apply_accumulate(out)
        return out

    def _apply_accumulate(
        self, readings: dict[str, ManifoldTokenReading],
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

    def score_single_token(
        self, hidden_per_layer: dict[int, torch.Tensor],
    ) -> dict[str, "ManifoldTokenReading"]:
        """Per-probe coordinate readings for a single token (no accumulate).

        The live per-token read source: returns ``{name: ManifoldTokenReading}``
        with ``coords`` (domain frame), ``fraction``, ``nearest`` per probe.
        Does NOT touch history/stats — the in-flight gate/stream path must
        not corrupt the session-level accumulators.
        """
        return self._score_tokens(hidden_per_layer, accumulate=False)

    def score_single_token_tensor(
        self, hidden_per_layer: dict[int, torch.Tensor],
    ) -> torch.Tensor:
        """On-device ``[P1]`` axis-0 coordinate row for one token — no host sync.

        The rank-1 fast-path coordinate for each rank-1 probe (in
        :attr:`_fast_keys` order), returned on-device with zero ``.item()``
        / ``.cpu()``.  Powers the session's incremental per-token capture:
        ``_finalize_generation`` stacks one row per generated token and
        syncs once, reproducing :meth:`score_per_token` bit-for-bit while
        keeping only the latest hidden per layer in device memory.  Rank-R
        probes force full-retention capture, so they never ride this row.

        Empty-hidden / no-fast-probe case returns a zeros ``[P1]`` tensor on
        the cache device (else CPU).
        """
        n_fast = len(self._fast_keys)
        if not hidden_per_layer or (
            n_fast == 0 and not self._probes
        ):
            device = self._cache_device or torch.device("cpu")
            return torch.zeros((n_fast,), device=device, dtype=torch.float32)
        device = next(iter(hidden_per_layer.values())).device
        self._ensure_cache(device)
        coords_t, _frac_t = self._score_fast(hidden_per_layer)
        return coords_t

    def score_single_token_per_layer(
        self,
        hidden_per_layer: dict[int, torch.Tensor],
    ) -> dict[int, dict[str, float]]:
        """Per-layer × per-probe domain coordinate for a single token.

        Returns the per-layer (un-aggregated) rank-1 domain coordinate for
        each rank-1 probe covering that layer — the coordinate analogue of
        the old per-layer cosine heatmap.  ``{layer_idx: {probe_name:
        coord}}``; layers no probe covers are omitted.  Rank-R probes are
        not expanded here (their per-layer coords need the per-probe
        invert); the heatmap is a rank-1 roster view.
        """
        if not hidden_per_layer or not self._fast_keys:
            return {}
        device = next(iter(hidden_per_layer.values())).device
        self._ensure_cache(device)
        out: dict[int, dict[str, float]] = {}
        for layer_idx, h in hidden_per_layer.items():
            entry = self._fast_cache.get(layer_idx)
            if entry is None:
                continue
            hf = h.reshape(-1).to(torch.float32) if h.ndim > 1 else h.to(torch.float32)
            g = entry["A"] @ hf - entry["cmean"]
            c = g / entry["mR"]
            dom = (entry["slope"] * c + entry["intercept"]).cpu().tolist()
            cols = entry["cols"].cpu().tolist()
            out[layer_idx] = {
                self._fast_keys[ci]: dom[row] for row, ci in enumerate(cols)
            }
        return out

    def measure_from_hidden(
        self, hidden_per_layer: dict[int, torch.Tensor], accumulate: bool = True,
    ) -> dict[str, "ManifoldTokenReading"]:
        """Score probes from pre-captured hidden states (no forward pass).

        Use when hidden states have already been captured during generation
        (e.g. via capture hooks), avoiding a redundant forward pass.
        """
        return self._score_tokens(hidden_per_layer, accumulate=accumulate)

    @staticmethod
    def flat_scalars(
        readings: dict[str, "ManifoldTokenReading"],
    ) -> dict[str, float]:
        """Flatten per-probe readings into namespaced gate-callback scalars.

        For each probe emits the bare ``"<probe>"`` aliased to coordinate
        axis 0 (so ``@when:angry.calm > 0.4`` reads the single coord of a
        2-node concept unchanged), one ``"<probe>[<i>]"`` per coordinate
        axis (so ``@when:personas[3] > 0.4`` indexes an axis), and
        ``"<probe>:fraction"``.  Merges directly into
        ``TriggerContext.probe_scores`` alongside the manifold-probe
        scalars from :meth:`ManifoldMonitor.flat_scalars`.
        """
        out: dict[str, float] = {}
        for name, reading in readings.items():
            coords = reading.coords
            out[name] = coords[0] if coords else 0.0
            for i, c in enumerate(coords):
                out[f"{name}[{i}]"] = c
            out[f"{name}:fraction"] = reading.fraction
        return out

    def _per_token_coord_stream(
        self, captured: dict[int, torch.Tensor], n: int,
    ) -> dict[str, list[float]]:
        """Axis-0 domain-coordinate stream per probe over ``n`` tokens.

        Fast (rank-1) probes are accumulated on-device and synced once;
        rank-R probes fall back to axis 0 of their per-token reading.  Row
        ``i`` reads ``captured[L][i]`` (guarded), so an EOS overshoot is
        ignored and a short capture leaves trailing zeros.
        """
        per_token: dict[str, list[float]] = {name: [0.0] * n for name in self._probes}
        if self._fast_keys:
            rows: list[torch.Tensor] = []
            for i in range(n):
                tok = {L: h[i] for L, h in captured.items() if h.shape[0] > i}
                coords_t, _ = self._score_fast(tok)
                rows.append(coords_t)
            stacked = torch.stack(rows, 0).cpu().tolist() if rows else []
            for j, name in enumerate(self._fast_keys):
                per_token[name] = [stacked[i][j] for i in range(n)]
        for name in self._slow_keys:
            for i in range(n):
                tok = {L: h[i] for L, h in captured.items() if h.shape[0] > i}
                r = self._score_slow(name, tok)
                per_token[name][i] = r.coords[0] if r.coords else 0.0
        return per_token

    def score_per_token(
        self,
        captured: dict[int, torch.Tensor],
        generated_ids: list[int],
        tokenizer: Any,
        *,
        accumulate: bool = True,
    ) -> tuple[dict[str, "ManifoldTokenReading"], dict[str, list[float]]]:
        """Score probes per generated token using pre-captured hidden states.

        ``captured[layer_idx]`` is a ``(n, dim)`` stack where row ``k`` is
        the hidden state that produced generated token ``k``.  Returns
        ``(aggregate_readings, per_token_coord_stream)``: the aggregate is
        the full :class:`ManifoldTokenReading` per probe pooled at the last
        non-special token (updates history when ``accumulate``); the stream
        is the per-token axis-0 domain coordinate.
        """
        n = len(generated_ids)
        empty_agg = {
            name: ManifoldTokenReading(fraction=0.0, nearest=[], coords=())
            for name in self._probes
        }
        if n == 0 or not captured:
            return empty_agg, {name: [] for name in self._probes}

        any_h = next(iter(captured.values()))
        self._ensure_cache(any_h.device)

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
    ) -> tuple[dict[str, "ManifoldTokenReading"], dict[str, list[float]]]:
        """Score probes over a pre-captured ``[T, D]`` stack per layer.

        Like :meth:`score_per_token` but without ``generated_ids`` /
        tokenizer — the caller has already chosen the meaningful rows.
        Aggregate is pooled from row ``agg_index`` (default the last row).
        ``accumulate`` defaults to ``False`` (ad-hoc researcher probing,
        not the in-flight loop).
        """
        empty_agg = {
            name: ManifoldTokenReading(fraction=0.0, nearest=[], coords=())
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

        self._ensure_cache(any_h.device)

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

    def update_live(self, readings: dict[str, "ManifoldTokenReading"]) -> None:
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
        """Register a flat :class:`Manifold` probe (rank-1 or rank-R).

        Pre-caches the per-layer node values + EV weights + whitened
        factors (shared with :class:`ManifoldMonitor` via
        :func:`_attach_manifold_probe`); the wired whitener must cover the
        fit's layers.
        """
        is_new = name not in self._probes
        self._probes[name] = _attach_manifold_probe(
            name, manifold, top_n=top_n, whitener=self._whitener,
        )
        if is_new:
            self.history[name] = deque(maxlen=_MAX_HISTORY)
            self._stats[name] = self._empty_stats()
        self._invalidate_cache()

    def remove_probe(self, name: str):
        self._probes.pop(name, None)
        if name in self.history:
            del self.history[name]
        if name in self._stats:
            del self._stats[name]
        self._invalidate_cache()

    def reset_history(self):
        for name in self._probes:
            self.history[name].clear()
            self._stats[name] = self._empty_stats()
        self._pending_aggregate = False
        self._pending_per_token = False


@dataclass
class AttachedManifoldProbe:
    """One manifold registered on a :class:`ManifoldMonitor`.

    Pairs the loaded :class:`Manifold` artifact with the per-layer cache
    the monitor uses on the hot path: ``node_values_reduced`` is the
    per-layer ``(K, R)`` tensor of node activations in subspace coords
    (``(sub.eval_at(domain.embed(node_coords)) - sub.mean) @ sub.basis.T``),
    pre-computed once at attach so per-token distance computations are one
    batched cdist in R-dim per layer.  ``ev_weights`` is the per-layer EV
    ratio used to weight cross-layer aggregation; floored at
    :data:`_MIN_EV_WEIGHT` so a degenerate layer doesn't crash the
    aggregator.
    """

    name: str
    manifold: "Manifold"
    top_n: int = DEFAULT_NEAREST_TOP_N
    # Per-layer cache, indexed by layer index — same set of layers as
    # ``manifold.layers``.
    node_values_reduced: dict[int, torch.Tensor] = field(default_factory=dict)
    # Per-layer EV weight, normalized to sum to 1 across attached layers.
    ev_weights: dict[int, float] = field(default_factory=dict)
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
    X: torch.Tensor            # (N, D) centered neutral observations
    K_inv: torch.Tensor        # (N, N) Woodbury inverse
    lam: float                 # ridge λ


# ----------------------------------------------------------------------------
# Shared subspace-read machinery — used by BOTH monitors.  TraitMonitor (flat,
# batched, coordinate readout) and ManifoldMonitor (per-probe, curved-capable)
# are the read-side peers of the steering split (one ``subspace_inject`` kernel,
# ``SteeringManager.{subspaces, manifolds}``): they share the whitened-factor
# build, the per-layer geometry, and the attach-time node cache, and diverge
# only in how they aggregate (TraitMonitor stacks rank-1 probes into one matmul
# and inverts affinely; ManifoldMonitor loops per probe and foot-solves curved
# fits in the aggregate).
# ----------------------------------------------------------------------------


def _probe_rank(probe: "AttachedManifoldProbe") -> int:
    """Subspace dimension ``R`` of a flat probe (its per-layer basis rows)."""
    for sub in probe.manifold.layers.values():
        return int(sub.basis.shape[0])
    return 0


def _build_whitened_factors(
    whitener: Any, probe: "AttachedManifoldProbe",
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
        X, K_inv, lam = whitener.woodbury_factors(
            layer_idx, device=dev, dtype=torch.float32,
        )
        chol_dev = chol.to(device=dev, dtype=torch.float32)
        out[layer_idx] = _LayerWhiten(
            m_r_inv=m_r_inv.to(device=dev, dtype=torch.float32),
            chol=chol_dev,
            node_white=(v_reduced.to(torch.float32) @ chol_dev),
            X=X, K_inv=K_inv, lam=lam,
        )
    return out


def _attach_manifold_probe(
    name: str,
    manifold: "Manifold",
    *,
    top_n: int = DEFAULT_NEAREST_TOP_N,
    whitener: Any = None,
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
        node_values_reduced=node_values_reduced,
        ev_weights=ev_weights,
    )
    probe.whitened = _build_whitened_factors(whitener, probe)
    return probe


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
    sub = probe.manifold.layers[layer_idx]
    mean = sub.mean.to(device=h.device, dtype=torch.float32)
    basis = sub.basis.to(device=h.device, dtype=torch.float32)
    wh = probe.whitened.get(layer_idx)
    if wh is None:
        raise WhitenerError(
            f"subspace probe read missing whitened factors for layer "
            f"{layer_idx}; rebuild the probe (the Euclidean path is gone)"
        )
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


class ManifoldMonitor:
    """Read-side counterpart to manifold steering.

    For each attached manifold the monitor exposes three channels: the
    fraction of the centered activation that lives in the manifold's PCA
    subspace (hot path, scored per token), the top-N node labels nearest
    the running activation in activation space (hot path, EV-weighted
    across layers), and the inverse-projection of the pooled
    end-of-generation activation onto the manifold (slow path, run once
    in ``_finalize_generation``).

    Peer to :class:`TraitMonitor` — vector probes and manifold probes
    compose independently; ``flat_scalars`` flattens the rich per-token
    readings into namespaced scalars (``<probe>:fraction`` and
    ``<probe>@<label>``, the latter as ``-distance`` so larger means
    closer) that merge cleanly into ``TriggerContext.probe_scores`` for
    ``@when:`` gates.

    Hot-path discipline: no ``.item()`` per token, fp32 norms (fp16
    sum-of-squares overflows at hidden_dim >= 2048), pre-cached node
    values in subspace coords so per-token distance computations are one
    batched cdist in R-dim per layer.  ``layer_means`` is accepted for
    parity with :class:`TraitMonitor`'s init signature but not consulted
    — manifold fraction uses the per-fit ``LayerSubspace.mean`` (the
    centroid of the node centroids), not the global layer mean.
    """

    def __init__(
        self,
        layer_means: dict[int, torch.Tensor] | None = None,
        *,
        whitener: Any = None,
    ) -> None:
        # ``layer_means`` is accepted for parity with TraitMonitor's init
        # signature (the session wires both monitors the same way) but is
        # intentionally ignored — the manifold-side math never reads global
        # layer means; fraction centers on each fit's per-layer mean
        # (``LayerSubspace.mean``).
        del layer_means
        self._probes: dict[str, AttachedManifoldProbe] = {}
        # Mandatory Mahalanobis whitener (set lazily by the session).  The
        # per-token fraction + nearest-node distance are the whitened forms
        # — the read-side analogue of the whitened/Fisher subspace the
        # manifold was *fitted* in, and the peer of ``TraitMonitor``'s
        # whitened cosine.  ``_build_whitened`` raises if it doesn't cover
        # an attached manifold's fit layers (there is no Euclidean readout).
        self._whitener: Any = whitener
        # Mirror TraitMonitor's pending-flag pattern so the TUI / webui
        # streaming surfaces can poll for "is a new per-token reading
        # available."
        self._pending_per_token: bool = False

    @property
    def probe_names(self) -> list[str]:
        """Attached manifold-probe names in insertion order."""
        return list(self._probes.keys())

    @property
    def whitener(self) -> Any:
        """The wired :class:`LayerWhitener` (required for reads)."""
        return self._whitener

    def set_whitener(self, whitener: Any) -> None:
        """Wire (or clear) the Mahalanobis whitener and rebuild probe caches.

        Idempotent on the same instance.  Any change re-derives the
        per-probe :class:`_LayerWhiten` factors; manifold reads are
        Mahalanobis-only, so a new whitener that doesn't cover an attached
        manifold's layers raises :class:`WhitenerError` here.  Mirrors
        :meth:`TraitMonitor.set_whitener` — the session pushes the
        lazily-built whitener through here.
        """
        if whitener is self._whitener:
            return
        self._whitener = whitener
        for probe in self._probes.values():
            probe.whitened = _build_whitened_factors(self._whitener, probe)

    def attached_probes(self) -> dict[str, AttachedManifoldProbe]:
        """Return the live attached-probe map (read-only view)."""
        return dict(self._probes)

    def attached_layers(self) -> set[int]:
        """Union of layer indices across every attached manifold.

        Used by ``session._begin_capture`` to widen the per-token
        hidden-state capture so manifold scoring sees every layer the
        manifold covers, not just the vector-probe layer union.
        """
        out: set[int] = set()
        for probe in self._probes.values():
            out.update(probe.manifold.layers.keys())
        return out

    def add_probe(
        self,
        name: str,
        manifold: "Manifold",
        *,
        top_n: int = DEFAULT_NEAREST_TOP_N,
    ) -> None:
        """Register a manifold under ``name`` and pre-cache per-layer node values.

        ``top_n`` controls the ``nearest`` list length; default
        :data:`DEFAULT_NEAREST_TOP_N` (3).  Delegates the attach-time cache
        build (reduced node activations, EV weights, whitened factors) to
        the shared :func:`_attach_manifold_probe`; the wired whitener must
        cover the fit's layers.
        """
        self._probes[name] = _attach_manifold_probe(
            name, manifold, top_n=top_n, whitener=self._whitener,
        )

    def remove_probe(self, name: str) -> None:
        self._probes.pop(name, None)

    def reset(self) -> None:
        """Clear every attached manifold probe (TUI ``/unsteer``-style reset)."""
        self._probes.clear()
        self._pending_per_token = False

    def has_pending_per_token(self) -> bool:
        return self._pending_per_token

    def consume_pending_per_token(self) -> None:
        self._pending_per_token = False

    # ----------------------------------------------- hot-path scoring ---

    def score_single_token(
        self,
        hidden_per_layer: dict[int, torch.Tensor],
    ) -> dict[str, ManifoldTokenReading]:
        """Per-token manifold readings — hot path.

        For each attached probe, walks the shared layer set between
        ``hidden_per_layer`` and the probe's manifold layers, computes
        the per-layer subspace fraction and per-node distance, then
        EV-weighted-aggregates across layers.  Returns one
        :class:`ManifoldTokenReading` per probe; the empty
        ``hidden_per_layer`` case returns an empty dict.

        Each ``ManifoldTokenReading.fraction`` is in ``[0, 1]``; the
        nearest list is top-N ``(label, distance)`` ascending by
        distance.
        """
        out: dict[str, ManifoldTokenReading] = {}
        if not hidden_per_layer or not self._probes:
            return out

        for name, probe in self._probes.items():
            manifold = probe.manifold
            ev = probe.ev_weights
            shared = [
                idx for idx in manifold.layers
                if idx in hidden_per_layer
            ]
            if not shared:
                out[name] = ManifoldTokenReading(fraction=0.0, nearest=[])
                continue
            # EV weights restricted to the shared set, re-normalized.
            total_w = sum(ev.get(idx, 0.0) for idx in shared)
            if total_w <= _MIN_EV_WEIGHT:
                w_shared = {idx: 1.0 / len(shared) for idx in shared}
            else:
                w_shared = {idx: ev.get(idx, 0.0) / total_w for idx in shared}

            # Per-node EV-weighted distance accumulated ON-DEVICE as a
            # ``[K]`` tensor (K on the order of tens) and the EV-weighted
            # fraction as an on-device scalar.  Mirrors ``TraitMonitor``'s
            # single-sync discipline: zero ``.item()``/``.tolist()`` inside
            # the per-layer loop, one ``.cpu().tolist()`` + one ``.item()``
            # per probe after the loop.
            K = probe.node_values_reduced[shared[0]].shape[0]
            dist_acc: torch.Tensor | None = None
            frac_acc: torch.Tensor | None = None
            for layer_idx in shared:
                h = hidden_per_layer[layer_idx].to(torch.float32)
                if h.ndim > 1:
                    # Hot-path captures are 1-D ``[D]`` slices, but
                    # defensively flatten to the last dim.
                    h = h.reshape(-1, h.shape[-1])[-1]
                w = w_shared[layer_idx]
                # ``_layer_geometry`` returns the whitened M-orthogonal
                # fraction + the cdist query/nodes in matching (Mahalanobis)
                # coords.
                frac, cdist_query, _invert, cdist_nodes = _layer_geometry(
                    probe, layer_idx, h,
                )
                frac_contrib = w * frac
                frac_acc = (
                    frac_contrib if frac_acc is None
                    else frac_acc + frac_contrib
                )
                dists = torch.cdist(cdist_query, cdist_nodes).reshape(-1)  # (K,)
                contrib = w * dists
                dist_acc = (
                    contrib if dist_acc is None else dist_acc + contrib
                )
            # Single sync per probe: pull the accumulated distances + the
            # fraction off-device together, then sort host-side.
            assert dist_acc is not None and frac_acc is not None  # shared ≠ []
            dist_list = dist_acc.cpu().tolist()
            frac_sum = float(frac_acc.item())
            # Top-N by ascending EV-weighted distance.
            order = sorted(range(K), key=lambda k: dist_list[k])
            top = order[: probe.top_n]
            nearest = [
                (manifold.node_labels[k], dist_list[k]) for k in top
            ]
            out[name] = ManifoldTokenReading(
                fraction=frac_sum,
                nearest=nearest,
            )
        self._pending_per_token = True
        return out

    def flat_scalars(
        self,
        readings: dict[str, ManifoldTokenReading],
    ) -> dict[str, float]:
        """Flatten per-probe readings into namespaced gate-callback scalars.

        Emits ``"<probe>:fraction"`` for the subspace-fraction channel
        and ``"<probe>@<label>"`` for each node-distance channel; the
        latter uses ``-distance`` so the convention "larger = closer"
        holds across both channel types and ``@when:<probe>@<label> > x``
        reads like a similarity gate.  The output dict merges directly
        into ``TriggerContext.probe_scores`` alongside vector-probe
        scalars from :class:`TraitMonitor`.
        """
        out: dict[str, float] = {}
        for name, reading in readings.items():
            out[f"{name}:fraction"] = reading.fraction
            for label, dist in reading.nearest:
                out[f"{name}@{label}"] = -dist
        return out

    # ----------------------------------------------- slow-path aggregate ---

    def score_aggregate(
        self,
        captured_per_layer: dict[int, torch.Tensor],
        *,
        agg_index: int | None = None,
    ) -> dict[str, ManifoldAggregate]:
        """End-of-generation manifold aggregates over pooled captures.

        ``captured_per_layer[L]`` is the per-layer ``[T, D]`` stack of
        per-token captures the session collected during generation.  The
        per-layer pooled activation is the hidden state at the **last
        non-special token** (``agg_index``, default the final row) — the
        same single-state discipline extraction and the vector-probe
        aggregate use, *not* a mean across the trajectory (that lives in
        :meth:`score_single_token`).  The session passes the
        ``last_content_index`` walkback so trailing special / structural
        tokens never pull the reported aggregate.  From that pooled state
        the aggregator computes (1) per-layer + EV-weighted subspace
        fraction, (2) the EV-weighted nearest-node vote (same shape as
        the per-token channel, top-N), and (3) per-layer
        ``invert_parameterization`` to recover authoring coords +
        normalized residual.  Coords are EV-weighted-meaned across
        layers (they share the manifold's domain so the mean is
        meaningful); residual is reported as
        ``dist_L / ||h_par_c_L||`` per layer and EV-weighted-meaned.
        """
        out: dict[str, ManifoldAggregate] = {}
        if not captured_per_layer or not self._probes:
            return out

        # Pool per-layer once — every probe reads the same captures.
        # Select the last-non-special row (``agg_index``, default the
        # final token) rather than averaging the trajectory, so the
        # reported aggregate matches extraction and the vector aggregate.
        pooled: dict[int, torch.Tensor] = {}
        for layer_idx, stack in captured_per_layer.items():
            if stack is None or stack.numel() == 0:
                continue
            if stack.ndim == 1:
                pooled[layer_idx] = stack.to(torch.float32)
            else:
                T = stack.shape[0]
                row = T - 1 if agg_index is None else max(0, min(agg_index, T - 1))
                pooled[layer_idx] = stack.to(torch.float32)[row]

        for name, probe in self._probes.items():
            manifold = probe.manifold
            ev = probe.ev_weights
            shared = [
                idx for idx in manifold.layers
                if idx in pooled
            ]
            if not shared:
                out[name] = ManifoldAggregate(
                    fraction_mean=0.0,
                    fraction_per_layer={},
                    nearest=[],
                    coords=(),
                    coords_per_layer={},
                    residual_mean=0.0,
                    residual_per_layer={},
                )
                continue
            total_w = sum(ev.get(idx, 0.0) for idx in shared)
            if total_w <= _MIN_EV_WEIGHT:
                w_shared = {idx: 1.0 / len(shared) for idx in shared}
            else:
                w_shared = {idx: ev.get(idx, 0.0) / total_w for idx in shared}

            K = probe.node_values_reduced[shared[0]].shape[0]
            dist_acc = [0.0] * K
            frac_per_layer: dict[int, float] = {}
            coords_per_layer: dict[int, tuple[float, ...]] = {}
            residual_per_layer: dict[int, float] = {}
            frac_mean = 0.0
            residual_mean = 0.0
            coords_mean: list[float] | None = None
            n_dim = manifold.domain.intrinsic_dim

            for layer_idx in shared:
                sub = manifold.layers[layer_idx]
                h = pooled[layer_idx].to(torch.float32)
                if h.ndim > 1:
                    h = h.reshape(-1, h.shape[-1])[-1]
                # Shared metric branch: ``frac`` is the (whitened
                # M-orthogonal or Euclidean) in-subspace share; ``cdist_*``
                # are matched-space nearest-node coords; ``invert_query`` is
                # the reduced-coord query for ``invert_parameterization``
                # (the M-orthogonal projection coords when whitened).
                frac_t, cdist_query, invert_query, cdist_nodes = (
                    _layer_geometry(probe, layer_idx, h)
                )
                frac = float(frac_t.item())
                frac_per_layer[layer_idx] = frac
                w = w_shared[layer_idx]
                frac_mean += w * frac

                dists = torch.cdist(cdist_query, cdist_nodes).reshape(-1)
                dist_list = dists.cpu().tolist()
                for k in range(K):
                    dist_acc[k] += w * dist_list[k]

                # Inverse projection — coords + residual per layer.
                pos, res = invert_parameterization(
                    sub, manifold.domain, invert_query.reshape(1, -1),
                    manifold.node_coords,
                )
                pos_t = pos.reshape(-1)
                res_val = float(res.reshape(-1)[0].item())
                # Normalize residual by the in-subspace reduced-coord
                # magnitude (‖h_par_c‖ in the Euclidean case, ‖M-proj
                # coords‖ when whitened) so the number is comparable across
                # layers / generations.
                par_norm_val = float(
                    torch.linalg.vector_norm(invert_query).item()
                )
                norm_residual = (
                    0.0
                    if par_norm_val < _FRACTION_EPSILON
                    else res_val / par_norm_val
                )
                residual_per_layer[layer_idx] = norm_residual
                residual_mean += w * norm_residual
                coord_tup = tuple(
                    float(c) for c in pos_t.tolist()[:n_dim]
                )
                coords_per_layer[layer_idx] = coord_tup
                if coords_mean is None:
                    coords_mean = [w * c for c in coord_tup]
                else:
                    for i, c in enumerate(coord_tup):
                        if i < len(coords_mean):
                            coords_mean[i] += w * c

            order = sorted(range(K), key=lambda k: dist_acc[k])
            top = order[: probe.top_n]
            nearest = [
                (manifold.node_labels[k], dist_acc[k]) for k in top
            ]
            coords_tuple = (
                tuple(coords_mean) if coords_mean is not None else ()
            )
            out[name] = ManifoldAggregate(
                fraction_mean=frac_mean,
                fraction_per_layer=frac_per_layer,
                nearest=nearest,
                coords=coords_tuple,
                coords_per_layer=coords_per_layer,
                residual_mean=residual_mean,
                residual_per_layer=residual_per_layer,
            )
        return out
