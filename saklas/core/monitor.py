from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from saklas.core.mahalanobis import WhitenerError
from saklas.core.manifold import invert_parameterization
from saklas.core.results import ManifoldReading

if TYPE_CHECKING:
    from saklas.core.manifold import Manifold

_MAX_HISTORY = 8

_EMPTY_STATS = {"count": 0, "sum": 0.0, "sum_sq": 0.0,
                "min": float("inf"), "max": float("-inf")}

# Default top-N nearest-node count for manifold probes.  Per-probe
# override available on ``Monitor.add_probe``.
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

    **One** read path.  Each probe is a
    :class:`~saklas.core.manifold.Manifold`, and every read — live per token
    (gate / stream) and the end-of-generation aggregate — runs the same
    per-probe per-layer geometry (:func:`_layer_geometry`) and produces one
    full :class:`ManifoldReading`.  There is no batched-affine fast path and
    no flat/curved field asymmetry: a flat (affine) probe recovers ``coords``
    through the affine reduced→domain map (off-surface ``residual`` is
    identically 0), a curved probe through the iterative
    :func:`invert_parameterization` foot solve (the off-surface ``residual``
    is real).  Both report ``coords`` + ``fraction`` + ``nearest`` +
    ``residual`` plus their ``*_per_layer`` traces, every token.  This trades
    the old hot-path batching for full per-token information — the project is
    a research tool, and the per-token nearest / curved coords / residual /
    per-layer data is worth the throughput.

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
    flows through the :class:`ManifoldReading` surface.
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
        is a no-op.  Any change flushes the batched fast cache and rebuilds
        each attached probe's per-layer whitened factors against the new
        covariance (manifold reads are Mahalanobis-only, so a whitener that
        doesn't cover an attached probe's fit layers raises here).
        """
        if whitener is self._whitener:
            return
        self._whitener = whitener
        for probe in self._probes.values():
            probe.whitened = _build_whitened_factors(whitener, probe)

    def _score_probe_full(
        self,
        probe: "AttachedManifoldProbe",
        hidden_per_layer: dict[int, torch.Tensor],
    ) -> ManifoldReading:
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
        is_affine = _probe_is_affine(probe)
        shared = [idx for idx in manifold.layers if idx in hidden_per_layer]
        if not shared:
            return ManifoldReading(fraction=0.0, nearest=[], coords=())
        total_w = sum(ev.get(idx, 0.0) for idx in shared)
        if total_w <= _MIN_EV_WEIGHT:
            w_shared = {idx: 1.0 / len(shared) for idx in shared}
        else:
            w_shared = {idx: ev.get(idx, 0.0) / total_w for idx in shared}

        K = probe.node_values_reduced[shared[0]].shape[0]
        n_dim = manifold.domain.intrinsic_dim
        dist_acc = [0.0] * K
        frac_per_layer: dict[int, float] = {}
        coords_per_layer: dict[int, tuple[float, ...]] = {}
        residual_per_layer: dict[int, float] = {}
        frac_mean = 0.0
        residual_mean = 0.0
        coords_mean: list[float] | None = None

        for layer_idx in shared:
            sub = manifold.layers[layer_idx]
            h = hidden_per_layer[layer_idx].to(torch.float32)
            if h.ndim > 1:
                h = h.reshape(-1, h.shape[-1])[-1]
            frac_t, cdist_query, invert_query, cdist_nodes = _layer_geometry(
                probe, layer_idx, h,
            )
            frac = float(frac_t.item())
            frac_per_layer[layer_idx] = frac
            w = w_shared[layer_idx]
            frac_mean += w * frac

            dists = torch.cdist(cdist_query, cdist_nodes).reshape(-1)
            dist_list = dists.cpu().tolist()
            for k in range(K):
                dist_acc[k] += w * dist_list[k]

            if is_affine:
                # Affine map: coords are exact, no off-subspace residual.
                wh = probe.whitened[layer_idx]
                pos_t = (
                    wh.coord_S @ invert_query + wh.coord_b
                    if wh.coord_S is not None and wh.coord_b is not None
                    else invert_query
                )
                norm_residual = 0.0
            else:
                pos, res = invert_parameterization(
                    sub, manifold.domain, invert_query.reshape(1, -1),
                    manifold.node_coords,
                )
                pos_t = pos.reshape(-1)
                res_val = float(res.reshape(-1)[0].item())
                par_norm_val = float(
                    torch.linalg.vector_norm(invert_query).item()
                )
                norm_residual = (
                    0.0 if par_norm_val < _FRACTION_EPSILON
                    else res_val / par_norm_val
                )
            residual_per_layer[layer_idx] = norm_residual
            residual_mean += w * norm_residual
            coord_tup = tuple(
                float(c) for c in pos_t.reshape(-1).tolist()[:n_dim]
            )
            coords_per_layer[layer_idx] = coord_tup
            if coords_mean is None:
                coords_mean = [w * c for c in coord_tup]
            else:
                for i, c in enumerate(coord_tup):
                    if i < len(coords_mean):
                        coords_mean[i] += w * c

        order = sorted(range(K), key=lambda k: dist_acc[k])
        nearest = [
            (manifold.node_labels[k], dist_acc[k]) for k in order[: probe.top_n]
        ]
        return ManifoldReading(
            fraction=frac_mean,
            nearest=nearest,
            coords=tuple(coords_mean) if coords_mean is not None else (),
            residual=residual_mean,
            fraction_per_layer=frac_per_layer,
            coords_per_layer=coords_per_layer,
            residual_per_layer=residual_per_layer,
        )

    def _score_full(
        self, hidden_per_layer: dict[int, torch.Tensor],
    ) -> dict[str, ManifoldReading]:
        """Full readings for every attached probe at one state.

        One per-probe :func:`_layer_geometry` pass — flat and curved alike,
        every field populated.  No batched fast path: the research-tool
        priority is full per-token information, not throughput.
        """
        out: dict[str, ManifoldReading] = {}
        if not hidden_per_layer or not self._probes:
            return out
        for name, probe in self._probes.items():
            out[name] = self._score_probe_full(probe, hidden_per_layer)
        return out

    def _score_tokens(
        self,
        hidden_per_layer: dict[int, torch.Tensor],
        accumulate: bool = True,
    ) -> dict[str, ManifoldReading]:
        """Score every probe → ``{name: ManifoldReading}`` (full reading).

        ``accumulate`` folds the cross-layer coords into history/stats (the
        in-flight per-token path passes False).
        """
        out = self._score_full(hidden_per_layer)
        if out and accumulate:
            self._apply_accumulate(out)
        return out

    def _apply_accumulate(
        self, readings: dict[str, ManifoldReading],
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
    ) -> dict[str, "ManifoldReading"]:
        """Per-probe full reading for a single token (no accumulate).

        The live per-token read source: returns ``{name: ManifoldReading}``
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
    ) -> dict[str, "ManifoldReading"]:
        """Score probes from pre-captured hidden states (no forward pass).

        Use when hidden states have already been captured during generation
        (e.g. via capture hooks), avoiding a redundant forward pass.
        """
        return self._score_tokens(hidden_per_layer, accumulate=accumulate)

    @staticmethod
    def flat_scalars(
        readings: dict[str, "ManifoldReading"],
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
        the gate grammar is uniform.  All shapes merge directly into
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
    ) -> tuple[dict[str, "ManifoldReading"], dict[str, list[float]]]:
        """Score probes per generated token using pre-captured hidden states.

        ``captured[layer_idx]`` is a ``(n, dim)`` stack where row ``k`` is
        the hidden state that produced generated token ``k``.  Returns
        ``(aggregate_readings, per_token_coord_stream)``: the aggregate is
        the full :class:`ManifoldReading` per probe pooled at the last
        non-special token (updates history when ``accumulate``); the stream
        is the per-token axis-0 domain coordinate.
        """
        n = len(generated_ids)
        empty_agg = {
            name: ManifoldReading(fraction=0.0, nearest=[], coords=())
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
    ) -> tuple[dict[str, "ManifoldReading"], dict[str, list[float]]]:
        """Score probes over a pre-captured ``[T, D]`` stack per layer.

        Like :meth:`score_per_token` but without ``generated_ids`` /
        tokenizer — the caller has already chosen the meaningful rows.
        Aggregate is pooled from row ``agg_index`` (default the last row).
        ``accumulate`` defaults to ``False`` (ad-hoc researcher probing,
        not the in-flight loop).
        """
        empty_agg = {
            name: ManifoldReading(fraction=0.0, nearest=[], coords=())
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

    def update_live(self, readings: dict[str, "ManifoldReading"]) -> None:
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
        fit's layers.  Flat vs curved is decided per probe at read time
        (:func:`_probe_is_affine`), not cached.
        """
        is_new = name not in self._probes
        self._probes[name] = _attach_manifold_probe(
            name, manifold, top_n=top_n, whitener=self._whitener,
        )
        if is_new:
            self.history[name] = deque(maxlen=_MAX_HISTORY)
            self._stats[name] = self._empty_stats()

    def remove_probe(self, name: str):
        self._probes.pop(name, None)
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
    ) -> dict[str, ManifoldReading]:
        """End-of-generation aggregate over pooled captures — all probes.

        ``captured_per_layer[L]`` is the per-layer ``[T, D]`` capture stack.
        The pooled activation is the **last non-special token** (``agg_index``,
        default the final row) — the same single-state discipline extraction
        uses, not a trajectory mean.  The aggregate is literally the unified
        per-token :meth:`_score_probe_full` read evaluated at the pooled token,
        so it is the same :class:`ManifoldReading` shape the live stream
        carries (and bit-identical to the live read at that token index).
        """
        out: dict[str, ManifoldReading] = {}
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

        for name, probe in self._probes.items():
            out[name] = self._score_probe_full(probe, pooled)
        return out


@dataclass
class AttachedManifoldProbe:
    """One manifold registered on a :class:`Monitor`.

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
# :class:`Monitor`.  The flat (affine) path stacks every flat probe — any rank
# — into one matmul + one shared Woodbury ``Σ⁻¹h`` per layer and inverts
# affinely; the curved path loops per probe and foot-solves in the aggregate.
# Both share the whitened-factor build, the per-layer geometry, and the
# attach-time node cache — the read-side peer of the steering split (one
# ``subspace_inject`` kernel, ``SteeringManager.{subspaces, manifolds}``).
# ----------------------------------------------------------------------------


def _probe_is_affine(probe: "AttachedManifoldProbe") -> bool:
    """True iff the probe's fit is flat (affine) — batched coordinate readout.

    A manifold's layers are uniformly affine (``pca`` / 2-node concept) or
    curved (``spectral`` / ``authored``), so the first fitted layer decides.
    An empty manifold is treated as affine (it scores to nothing anyway).
    """
    for sub in probe.manifold.layers.values():
        return bool(sub.is_affine)
    return True


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
    v_centered = sub_f.eval_at(embedded) - mean          # (K, D)
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
        m_r_inv_dev = m_r_inv.to(device=dev, dtype=torch.float32)
        # Flat probes carry the affine reduced→domain coord map (the rank-R
        # generalization of the rank-1 slope/intercept); a curved fit leaves
        # it ``None`` and recovers coords via ``invert_parameterization``.
        coord_S = coord_b = None
        if sub.is_affine:
            coord_S, coord_b = _affine_coord_map(
                sub, manifold, m_r_inv_dev, X, K_inv, lam,
            )
        out[layer_idx] = _LayerWhiten(
            m_r_inv=m_r_inv_dev,
            chol=chol_dev,
            node_white=(v_reduced.to(torch.float32) @ chol_dev),
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
