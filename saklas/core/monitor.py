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
    """Monitors model activations against a library of probe vectors.

    Each probe has a profile (dict mapping layer_idx -> baked direction).
    After generation, a single forward pass over the generated text
    pools the last content token's hidden state at each layer.  Per-layer
    probe similarity is the whitened (Mahalanobis) cosine
    ``⟨V, h_c⟩_M / (||V||_M ||h_c||_M)`` when a :class:`LayerWhitener`
    is wired and covers the layer — matching the Mahalanobis metric the
    default DiM extraction bakes and the ``~`` / ``|`` projection
    defaults to — and falls back to plain Euclidean cosine on layers the
    whitener doesn't cover (no whitener wired ⇒ all layers Euclidean).
    Similarities are weighted by the baked magnitude ``||baked_L||₂``
    (which encodes share * ref_norm — the same "how much does this layer
    steer per unit alpha" quantity; the bake already folded the
    Mahalanobis score into that magnitude, so the weight stays Euclidean
    to avoid double-counting), giving one value per probe per generation.
    """

    @staticmethod
    def _empty_stats() -> dict[str, Any]:
        return dict(_EMPTY_STATS)

    def __init__(self, probe_profiles: dict[str, dict[int, torch.Tensor]],
                 layer_means: dict[int, torch.Tensor] | None = None,
                 whitener: Any = None):
        """
        probe_profiles: maps probe name -> profile dict (layer_idx -> baked vector)
        layer_means: maps layer_idx -> mean activation vector for centering
        whitener: optional :class:`~saklas.core.mahalanobis.LayerWhitener`;
            when set, per-layer probe similarity switches to the whitened
            (Mahalanobis) cosine on every covered layer.  ``None`` keeps
            the legacy Euclidean cosine everywhere (the legitimate
            fallback, mirroring ``project_profile`` / ``subspace compare``).
        """
        self._raw_profiles: dict[str, dict[int, torch.Tensor]] = dict(probe_profiles)
        self._layer_means: dict[int, torch.Tensor] = dict(layer_means) if layer_means else {}
        self._whitener: Any = whitener

        # Per-layer stacked cache, inverted from the previous {probe: {layer: ...}}
        # form so one matmul scores every probe against a hidden state in a single
        # kernel launch. For each layer that any probe covers:
        #   V[P, D]  : unit-normed probe directions (zeros for probes missing L)
        #   W[P]     : per-probe weight at layer L (= ||baked_L||, 0 if missing)
        # The denominator per probe is built on-device as sum of W across layers
        # where hidden is also present; no .item() on the hot path.
        # Structure: {layer_idx: (V_stacked, W_stacked)}.
        self._layer_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        # Per-layer Mahalanobis cache, populated only for layers the
        # whitener covers.  Each entry holds the precomputed (off-hot-path)
        # whitened probe directions + their Mahalanobis norms plus the
        # on-device Woodbury factors used to whiten the per-token hidden:
        #   Vinv[P, D]  : Σ⁻¹ V (so ⟨V, h_c⟩_M = Vinv @ h_c)
        #   V_Mnorm[P]  : ||V||_M (zeros for probes missing L)
        #   X[N, D], K[N, N], lam : Woodbury factors for Σ⁻¹ h_c
        # Layers absent from this dict score Euclidean via _layer_cache.
        self._layer_white: dict[
            int,
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float],
        ] = {}
        # Probe -> index into the P axis (stable, insertion order).
        self._probe_index: dict[str, int] = {}
        self._cache_device: torch.device | None = None
        self._cache_probe_keys: tuple[str, ...] = ()
        # Identity of the whitener the cache was built against — a change
        # (set_whitener) invalidates the Mahalanobis cache.
        self._cache_whitener_id: int | None = id(whitener)
        # Cache of layer_means cast to float32 on cache_device.
        self._mean_cache: dict[int, torch.Tensor] = {}

        self.history: dict[str, deque[float]] = {n: deque(maxlen=_MAX_HISTORY) for n in self._raw_profiles}
        self._stats: dict[str, dict[str, Any]] = {n: self._empty_stats() for n in self._raw_profiles}

        # Aggregate path sets _pending_aggregate; per-token path sets _pending_per_token.
        # has_pending_data() returns aggregate readiness — the TUI uses it to refresh
        # trait readings after a measure() call.
        self._pending_aggregate = False
        self._pending_per_token = False

        # Live running mean during streaming generation. ``update_live`` is
        # called by ``generate_stream`` per emitted token; ``begin_live``
        # resets the accumulator at gen start; ``end_live`` clears so
        # post-gen reads fall back to the canonical history aggregate.
        self._live_values: dict[str, float] = {}
        self._live_count: int = 0
        self._live_pending: bool = False

    @property
    def probe_names(self) -> list[str]:
        """Probe names in insertion order."""
        return list(self._raw_profiles.keys())

    @property
    def profiles(self) -> dict[str, dict[int, torch.Tensor]]:
        """Probe profiles: name -> {layer_idx: baked vector}."""
        return self._raw_profiles

    @property
    def layer_means(self) -> dict[int, torch.Tensor]:
        return self._layer_means

    @layer_means.setter
    def layer_means(self, value: dict[int, torch.Tensor]) -> None:
        value_in: Any = value
        if value_in is not None and not isinstance(value_in, dict):
            raise TypeError(f"layer_means must be a dict, got {type(value).__name__}")
        self._layer_means = dict(value) if value else {}
        # Invalidate mean cache; v_unit cache is independent of means.
        self._mean_cache = {}

    @property
    def whitener(self) -> Any:
        """The wired :class:`LayerWhitener` (required for reads)."""
        return self._whitener

    def set_whitener(self, whitener: Any) -> None:
        """Wire (or clear) the Mahalanobis whitener and invalidate the cache.

        Idempotent on the same instance — re-wiring the identical whitener
        is a no-op so a session that hands the monitor its lazily-built
        whitener twice doesn't pay a needless rebuild.  Any change flushes
        the Mahalanobis cache so the next scoring call rebuilds the
        whitened probe directions against the new covariance.
        """
        if whitener is self._whitener:
            return
        self._whitener = whitener
        self._layer_white = {}
        self._layer_cache = {}
        self._cache_device = None
        self._cache_probe_keys = ()
        self._cache_whitener_id = id(whitener)

    def _ensure_cache(self, device: torch.device) -> None:
        """Build/refresh the per-device float32 cache of stacked probe matrices + means.

        Builds one ``(V[P,D], W[P])`` pair per layer that any probe covers. ``V`` holds
        unit-normed directions (rows for probes missing that layer are zero, which
        produces zero similarity — correct because ``W`` at that slot is also zero and
        the denominator mask is shared). ``W[p] = ||baked_p_L||`` for probes that own
        the layer, else 0. No ``.item()`` calls — norms stay on-device.

        When a whitener is wired, every layer it covers additionally gets a
        Mahalanobis entry in ``_layer_white`` — the whitened probe
        directions ``Σ⁻¹V`` and their Mahalanobis norms ``||V||_M``
        (precomputed here, off the hot path) plus the on-device Woodbury
        factors ``(X, K, λ)`` so the per-token ``Σ⁻¹ h_c`` apply runs
        inline without routing through the CPU-forcing
        :meth:`LayerWhitener.apply_inv`.
        """
        probe_keys = tuple(self._raw_profiles.keys())
        if (
            self._cache_device == device
            and self._cache_probe_keys == probe_keys
            and self._mean_cache.keys() == self._layer_means.keys()
            and self._cache_whitener_id == id(self._whitener)
            and self._layer_cache
        ):
            return

        self._cache_device = device
        self._cache_whitener_id = id(self._whitener)
        self._probe_index = {name: i for i, name in enumerate(probe_keys)}
        self._cache_probe_keys = probe_keys
        n_probes = len(probe_keys)

        # Union of layers across all probes, plus a per-layer probe membership map.
        layer_members: dict[int, list[tuple[int, torch.Tensor]]] = {}
        dim_for_layer: dict[int, int] = {}
        for pi, name in enumerate(probe_keys):
            for layer_idx, vec in self._raw_profiles[name].items():
                v = vec.to(device=device, dtype=torch.float32)
                layer_members.setdefault(layer_idx, []).append((pi, v))
                dim_for_layer[layer_idx] = v.shape[-1]

        whitener = self._whitener
        new_layer_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        new_layer_white: dict[
            int,
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float],
        ] = {}
        for layer_idx, members in layer_members.items():
            dim = dim_for_layer[layer_idx]
            V = torch.zeros((n_probes, dim), device=device, dtype=torch.float32)
            W = torch.zeros((n_probes,), device=device, dtype=torch.float32)
            for pi, v in members:
                vn = v.norm().clamp(min=1e-8)
                V[pi] = v / vn
                # Keep weight on-device; sync cost deferred to the final result.
                W[pi] = vn
            new_layer_cache[layer_idx] = (V, W)

        # Mahalanobis read is MANDATORY over the scored-layer set, via the
        # ``LayerWhitener.covers_all`` gate vector extraction / manifold
        # fitting / ``subspace compare`` already use: every probed layer
        # must be whitened or the read raises.  There is no Euclidean path.
        # ``covers_all`` is trustworthy as "finite, usable factors on every
        # covered layer": ``LayerWhitener.from_neutral_activations`` skips
        # any layer whose centered activations or regularized inverse come
        # back non-finite (the degenerate-whitener case — e.g. gemma-3's
        # late-layer fp16 overflow — leaves the layer uncovered), so a
        # covered set is a clean set and no local finite-check is needed
        # here.  The per-probe whitening (``apply_inv``, P × O(ND)) is an
        # off-hot-path cache-build cost; the hot path whitens the per-token
        # hidden once per layer.
        scored_layers = list(layer_members.keys())
        if scored_layers and (
            whitener is None or not whitener.covers_all(scored_layers)
        ):
            raise WhitenerError(
                "probe scoring requires a Mahalanobis whitener covering every "
                f"probed layer {sorted(scored_layers)}; regenerate the neutral "
                "activation cache for this model (the Euclidean path is gone)"
            )
        if scored_layers:
            for layer_idx in scored_layers:
                X, K, lam = whitener.woodbury_factors(
                    layer_idx, device=device, dtype=torch.float32,
                )
                V, _W = new_layer_cache[layer_idx]
                dim = V.shape[-1]
                Vinv = torch.zeros(
                    (n_probes, dim), device=device, dtype=torch.float32,
                )
                # Floor ‖V‖_M at 1e-8 for *every* row, including probes that
                # don't own this layer (their Vinv row stays zero).  Without
                # the floor on the missing-probe rows the per-layer division
                # ``num / (V_Mnorm · h_Mnorm)`` would be ``0 / 0 = NaN``
                # there, and ``W[pi]=0 · NaN = NaN`` would poison the
                # accumulated numerator.  With the floor the missing-probe
                # sim is a clean ``0 / (1e-8 · h_Mnorm) = 0`` — matching the
                # Euclidean path's zero-row behavior.
                V_Mnorm = torch.full(
                    (n_probes,), 1e-8, device=device, dtype=torch.float32,
                )
                for pi, _v in layer_members[layer_idx]:
                    # V[pi] is the unit Euclidean direction; whiten it once.
                    vw = _woodbury_apply(V[pi], X, K, lam)
                    Vinv[pi] = vw
                    # ||V||_M = sqrt(Vᵀ Σ⁻¹ V) = sqrt(V · vw).
                    V_Mnorm[pi] = torch.sqrt(
                        torch.dot(V[pi], vw).clamp(min=0.0)
                    ).clamp(min=1e-8)
                new_layer_white[layer_idx] = (Vinv, V_Mnorm, X, K, lam)

        self._layer_cache = new_layer_cache
        self._layer_white = new_layer_white

        self._mean_cache = {
            idx: m.to(device=device, dtype=torch.float32)
            for idx, m in self._layer_means.items()
        }

    def _layer_sims(self, layer_idx: int, h: torch.Tensor) -> torch.Tensor:
        """Per-layer probe similarity row(s) for a hidden state.

        ``h`` is ``[D]`` (single token) or ``[n, D]`` (token stack); the
        returned tensor is ``[P]`` or ``[n, P]`` respectively.  Centers
        ``h`` by the layer mean, then computes the Mahalanobis cosine
        ``⟨V, h_c⟩_M / (||V||_M ||h_c||_M)`` (the whitener covers every
        scored layer — ``_ensure_cache`` raises otherwise).  The caller
        multiplies by the per-layer weight ``W`` and accumulates.
        """
        mean = self._mean_cache.get(layer_idx)
        h_c = h - mean if mean is not None else h
        Vinv, V_Mnorm, X, K, lam = self._layer_white[layer_idx]
        # ⟨V, h_c⟩_M = (Σ⁻¹ V) · h_c = Vinv · h_c.
        num = (
            h_c @ Vinv.transpose(0, 1) if h_c.ndim > 1 else Vinv @ h_c
        )                                              # [n,P] or [P]
        # ||h_c||_M = sqrt(h_cᵀ Σ⁻¹ h_c); one Woodbury apply per token.
        h_inv = _woodbury_apply(h_c, X, K, lam)
        h_Mnorm = torch.sqrt(
            (h_c * h_inv).sum(dim=-1, keepdim=True).clamp(min=0.0)
        ).clamp(min=1e-8)                              # [n,1] or [1]
        return num / (V_Mnorm * h_Mnorm)

    def _score_probes(self, hidden_per_layer: dict[int, torch.Tensor], accumulate: bool = True) -> dict[str, float]:
        """Score all probes against hidden states.

        When ``accumulate`` is True (default), history and stats are updated.
        When False, the call is read-only — useful for stateless API requests
        that must not mutate session-level probe accumulators.
        """
        probe_keys = self._cache_probe_keys if self._cache_device is not None else tuple(self._raw_profiles.keys())
        if not hidden_per_layer:
            vals = dict.fromkeys(self._raw_profiles, 0.0)
            if accumulate:
                self._apply_accumulate(vals)
            return vals

        device = next(iter(hidden_per_layer.values())).device
        self._ensure_cache(device)
        probe_keys = self._cache_probe_keys
        n_probes = len(probe_keys)

        num = torch.zeros((n_probes,), device=device, dtype=torch.float32)
        den = torch.zeros((n_probes,), device=device, dtype=torch.float32)
        for layer_idx, h in hidden_per_layer.items():
            entry = self._layer_cache.get(layer_idx)
            if entry is None:
                continue
            _V, W = entry  # W: (P,)
            sims = self._layer_sims(layer_idx, h.float())  # (P,)
            num.add_(W * sims)
            den.add_(W)
        den.clamp_(min=1e-8)
        result = (num / den).cpu().tolist()  # single sync
        vals = {name: result[i] for i, name in enumerate(probe_keys)}
        # Probes that exist but weren't in the cache (shouldn't happen post-ensure)
        # still need a zero default to keep the output key set stable.
        for name in self._raw_profiles:
            vals.setdefault(name, 0.0)

        if accumulate:
            self._apply_accumulate(vals)
        return vals

    def _apply_accumulate(self, vals: dict[str, float]) -> None:
        for name, val in vals.items():
            if name not in self.history:
                continue
            self.history[name].append(val)
            s = self._stats[name]
            s["count"] += 1
            s["sum"] += val
            s["sum_sq"] += val * val
            if val < s["min"]:
                s["min"] = val
            if val > s["max"]:
                s["max"] = val
        self._pending_aggregate = True

    def score_single_token(self, hidden_per_layer: dict[int, torch.Tensor]) -> dict[str, float]:
        """Score all probes against a single token's hidden states.

        Like :meth:`measure_from_hidden` but does NOT accumulate into
        history/stats.  Designed for inline per-token scoring during live
        SSE streaming where accumulation would corrupt the session-level
        probe accumulators.
        """
        return self._score_probes(hidden_per_layer, accumulate=False)

    def score_single_token_tensor(
        self, hidden_per_layer: dict[int, torch.Tensor],
    ) -> torch.Tensor:
        """On-device ``[P]`` aggregate row for one token — no host sync.

        Exactly the math :meth:`_score_probes` does up to ``num/den``
        (per-layer ``V @ h_unit`` weighted accumulate, ``den.clamp_(min=
        1e-8)``), but returns the result as an on-device tensor in
        ``self._cache_probe_keys`` order instead of ``.cpu().tolist()``.
        Read-only: no history/stats accumulation, zero ``.item()``/
        ``.cpu()`` calls.

        Powers the session's incremental per-token capture path
        (``_finalize_generation`` stacks one row per generated token and
        does a single ``.cpu().tolist()`` at the end, reproducing
        :meth:`score_per_token` bit-for-bit while keeping only the latest
        hidden per layer in device memory).

        The empty-hidden / no-probes case returns a zeros ``[P]`` tensor
        on a sensible device (the cache device if a cache has been built,
        else CPU).
        """
        probe_keys = (
            self._cache_probe_keys
            if self._cache_device is not None
            else tuple(self._raw_profiles.keys())
        )
        n_probes = len(probe_keys)
        if not hidden_per_layer or n_probes == 0:
            device = self._cache_device or torch.device("cpu")
            return torch.zeros((n_probes,), device=device, dtype=torch.float32)

        device = next(iter(hidden_per_layer.values())).device
        self._ensure_cache(device)
        probe_keys = self._cache_probe_keys
        n_probes = len(probe_keys)

        num = torch.zeros((n_probes,), device=device, dtype=torch.float32)
        den = torch.zeros((n_probes,), device=device, dtype=torch.float32)
        for layer_idx, h in hidden_per_layer.items():
            entry = self._layer_cache.get(layer_idx)
            if entry is None:
                continue
            _V, W = entry  # W: (P,)
            sims = self._layer_sims(layer_idx, h.float())  # (P,)
            num.add_(W * sims)
            den.add_(W)
        den.clamp_(min=1e-8)
        return num / den

    def score_single_token_per_layer(
        self,
        hidden_per_layer: dict[int, torch.Tensor],
    ) -> dict[int, dict[str, float]]:
        """Per-layer × per-probe cosine scores for a single token.

        Same input shape as :meth:`score_single_token` but returns the
        raw per-layer cosines instead of the magnitude-weighted aggregate.
        Powers the web UI's per-token × per-layer × per-probe heatmap
        inspector — surfaces what each probe was reading at each layer
        for each generated token, not just the rolled-up score.

        Output: ``{layer_idx: {probe_name: cosine}}``.  Layers absent
        from the cache (no probe covers them) are omitted from the
        output entirely; missing-probe entries land as 0.0 to keep the
        per-layer dict shape stable.

        Cost: one matmul per layer × one ``.cpu().tolist()`` per layer.
        Heavier than the aggregate ``score_single_token`` (one matmul
        total + one sync); only called when at least one WS subscriber
        is consuming the per-layer payload.  Probe-key set is the union
        of registered probes — same as the aggregate path.
        """
        if not hidden_per_layer or not self._raw_profiles:
            return {}

        device = next(iter(hidden_per_layer.values())).device
        self._ensure_cache(device)
        probe_keys = self._cache_probe_keys
        if not probe_keys:
            return {}

        out: dict[int, dict[str, float]] = {}
        for layer_idx, h in hidden_per_layer.items():
            entry = self._layer_cache.get(layer_idx)
            if entry is None:
                continue
            sims = self._layer_sims(layer_idx, h.float()).cpu().tolist()  # one sync per layer
            out[layer_idx] = {
                name: sims[i] for i, name in enumerate(probe_keys)
            }
            # Probes registered but not covering this layer fall to 0
            # — keeps every layer's dict the same shape regardless of
            # which subset of probes the layer's cache holds.
            for name in self._raw_profiles:
                out[layer_idx].setdefault(name, 0.0)
        return out

    def measure_from_hidden(self, hidden_per_layer: dict[int, torch.Tensor], accumulate: bool = True) -> dict[str, float]:
        """Score probes from pre-captured hidden states (no forward pass).

        Use when hidden states have already been captured during generation
        (e.g. via capture hooks), avoiding a redundant forward pass.
        """
        return self._score_probes(hidden_per_layer, accumulate=accumulate)

    def score_per_token(
        self,
        captured: dict[int, torch.Tensor],
        generated_ids: list[int],
        tokenizer: Any,
        *,
        accumulate: bool = True,
    ) -> tuple[dict[str, float], dict[str, list[float]]]:
        """Score probes per generated token using pre-captured hidden states.

        ``captured[layer_idx]`` must be a ``(n, dim)`` tensor where row ``k``
        is the hidden state that produced generated token ``k`` (``n ==
        len(generated_ids)``). Typically populated by a ``HiddenCapture`` hook
        running in lockstep with the generation loop, so no extra forward
        pass is needed.

        Returns ``(aggregate_vals, per_token_scores)``. The aggregate is
        pooled from the last non-special generated token and updates history
        when ``accumulate`` is True. Per-token scores cover all
        ``len(generated_ids)`` rows.
        """
        n = len(generated_ids)
        empty_agg = dict.fromkeys(self._raw_profiles, 0.0)
        if n == 0 or not captured:
            return empty_agg, {name: [] for name in self._raw_profiles}

        any_h = next(iter(captured.values()))
        self._ensure_cache(any_h.device)

        # Aggregate pool: last non-special generated token — the one
        # canonical walkback (all_special_ids + added_tokens_encoder).
        from saklas.core.vectors import last_content_index
        agg_idx = last_content_index(generated_ids, tokenizer)
        agg_hidden = {
            layer_idx: h[agg_idx] for layer_idx, h in captured.items()
            if h.shape[0] > agg_idx
        }
        agg_vals = self._score_probes(agg_hidden, accumulate=accumulate)

        probe_keys = self._cache_probe_keys
        n_probes = len(probe_keys)
        device = any_h.device
        num = torch.zeros((n, n_probes), device=device, dtype=torch.float32)
        den = torch.zeros((n_probes,), device=device, dtype=torch.float32)
        for layer_idx, h in captured.items():
            # Captures may overshoot generated_ids by one when generation
            # terminates on an EOS token: the model forward fires (capture
            # +1) and then the loop breaks without appending the EOS to
            # generated_ids. Trim trailing extras to align capture[i] with
            # generated_ids[i]. Skip if we somehow have fewer than n.
            if h.shape[0] < n:
                continue
            if h.shape[0] > n:
                h = h[:n]
            entry = self._layer_cache.get(layer_idx)
            if entry is None:
                continue
            _V, W = entry  # W: (P,)
            sims = self._layer_sims(layer_idx, h.float())  # (n, P)
            num.add_(sims * W)  # broadcast over n
            den.add_(W)
        den.clamp_(min=1e-8)
        result = (num / den).cpu().tolist()  # single sync: list[n] of list[P]
        per_token: dict[str, list[float]] = {name: [] for name in self._raw_profiles}
        for i, name in enumerate(probe_keys):
            per_token[name] = [row[i] for row in result]
        for name in self._raw_profiles:
            if not per_token[name]:
                per_token[name] = [0.0] * n

        # Mirror score_stack's gate: the pending-per-token flag is TUI
        # state, and stateless callers (server path with accumulate=False)
        # must not leak into it.
        if accumulate:
            self._pending_per_token = True
        return agg_vals, per_token

    def score_stack(
        self,
        captured: dict[int, torch.Tensor],
        *,
        agg_index: int | None = None,
        accumulate: bool = False,
    ) -> tuple[dict[str, float], dict[str, list[float]]]:
        """Score probes over a pre-captured ``[T, D]`` stack per layer.

        Unlike :meth:`score_per_token`, this entry point does not require
        ``generated_ids`` or a tokenizer — the caller has already decided
        which rows are meaningful (e.g. already trimmed trailing special
        tokens if they cared). Aggregate is pooled from row ``agg_index``
        (defaults to the last row of each layer).

        Returns ``(aggregate_vals, per_token_scores)``. ``accumulate``
        defaults to ``False`` here (opposite of :meth:`score_per_token`)
        because this path is for ad-hoc researcher probing, not the
        in-flight generation loop.
        """
        empty_agg = dict.fromkeys(self._raw_profiles, 0.0)
        if not captured:
            return empty_agg, {name: [] for name in self._raw_profiles}

        any_h = next(iter(captured.values()))
        if any_h.ndim != 2:
            raise ValueError(
                f"score_stack expects [T, D] per layer; got shape {tuple(any_h.shape)}",
            )
        n = any_h.shape[0]
        if n == 0:
            return empty_agg, {name: [] for name in self._raw_profiles}

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
        agg_vals = self._score_probes(agg_hidden, accumulate=accumulate)

        probe_keys = self._cache_probe_keys
        n_probes = len(probe_keys)
        device = any_h.device
        num = torch.zeros((n, n_probes), device=device, dtype=torch.float32)
        den = torch.zeros((n_probes,), device=device, dtype=torch.float32)
        for layer_idx, h in captured.items():
            entry = self._layer_cache.get(layer_idx)
            if entry is None:
                continue
            _V, W = entry
            sims = self._layer_sims(layer_idx, h.float())  # (n, P)
            num.add_(sims * W)
            den.add_(W)
        den.clamp_(min=1e-8)
        result = (num / den).cpu().tolist()
        per_token: dict[str, list[float]] = {name: [] for name in self._raw_profiles}
        for i, name in enumerate(probe_keys):
            per_token[name] = [row[i] for row in result]
        for name in self._raw_profiles:
            if not per_token[name]:
                per_token[name] = [0.0] * n

        # Only flip the pending flag when the caller is actually
        # feeding this into history. Ad-hoc researcher calls
        # (accumulate=False, the default here) must not surface as
        # pending data on the monitor's TUI-facing consumer side —
        # score_per_token sets this unconditionally because it's
        # always in-flight; score_stack is not.
        if accumulate:
            self._pending_per_token = True
        return agg_vals, per_token

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

    def update_live(self, scores: dict[str, float]) -> None:
        """Fold one token's per-probe scores into the running mean."""
        self._live_count += 1
        c = self._live_count
        for name, v in scores.items():
            prev = self._live_values.get(name, 0.0)
            self._live_values[name] = prev + (v - prev) / c
        self._live_pending = True

    def end_live(self) -> None:
        """Drop the live running mean so reads fall back to history."""
        self._live_values = {}
        self._live_count = 0
        self._live_pending = False

    def get_current_and_previous(self) -> tuple[dict[str, float], dict[str, float]]:
        current = {}
        previous = {}
        for name in self._raw_profiles:
            hist = self.history[name]
            if name in self._live_values:
                # Mid-gen: live running mean wins; previous = last canonical
                # aggregate from history (or live itself if no history).
                current[name] = self._live_values[name]
                previous[name] = hist[-1] if hist else self._live_values[name]
            elif len(hist) >= 2:
                current[name] = hist[-1]
                previous[name] = hist[-2]
            elif hist:
                current[name] = hist[-1]
                previous[name] = hist[-1]
            else:
                current[name] = 0.0
                previous[name] = 0.0
        return current, previous

    def get_stats(self, name: str) -> dict[str, Any]:
        return self._stats.get(name, self._empty_stats())

    def get_sparkline(self, name: str) -> str:
        blocks = " ▁▂▃▄▅▆▇█"
        values = self.history[name]
        if not values:
            return ""
        lo, hi = min(values), max(values)
        span = hi - lo if hi != lo else 1.0
        return "".join(blocks[min(8, max(0, int((v - lo) / span * 8)))] for v in values)

    def add_probe(self, name: str, profile: dict[int, torch.Tensor]):
        is_new = name not in self._raw_profiles
        self._raw_profiles[name] = profile
        if is_new:
            self.history[name] = deque(maxlen=_MAX_HISTORY)
            self._stats[name] = self._empty_stats()
        # Invalidate stacked cache; rebuilt on next scoring call.
        self._layer_cache = {}
        self._cache_device = None
        self._cache_probe_keys = ()

    def remove_probe(self, name: str):
        if name in self._raw_profiles:
            del self._raw_profiles[name]
        if name in self.history:
            del self.history[name]
        if name in self._stats:
            del self._stats[name]
        self._layer_cache = {}
        self._cache_device = None
        self._cache_probe_keys = ()

    def reset_history(self):
        for name in self._raw_profiles:
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
    # Cached node coords on the device.  ``(K, n)`` shared across layers.
    embedded_node_coords: torch.Tensor | None = None
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
            probe.whitened = self._build_whitened(probe.manifold, probe)

    def _build_whitened(
        self, manifold: "Manifold", probe: AttachedManifoldProbe,
    ) -> dict[int, _LayerWhiten]:
        """Build the per-layer :class:`_LayerWhiten` map for a probe.

        The wired whitener is **required** and must cover **every** fit
        layer of ``manifold`` (all-or-nothing per probe, mirroring the
        fraction/distance metric gate the fit + DiM bake use).  There is no
        Euclidean readout: a missing or non-covering whitener raises
        :class:`WhitenerError`.  An empty ``manifold.layers`` yields an
        empty map (nothing to read).  Off the hot path — runs once at
        attach / on ``set_whitener``.
        """
        whitener = self._whitener
        if not manifold.layers:
            return {}
        layers = list(manifold.layers.keys())
        if whitener is None or not whitener.covers_all(layers):
            raise WhitenerError(
                "manifold probe reads require a Mahalanobis whitener covering "
                f"every fit layer {sorted(layers)}; regenerate the neutral "
                "activation cache for this model (the Euclidean path is gone)"
            )
        out: dict[int, _LayerWhiten] = {}
        for layer_idx, sub in manifold.layers.items():
            v_reduced = probe.node_values_reduced.get(layer_idx)
            if v_reduced is None:
                raise WhitenerError(
                    f"manifold probe cache missing reduced node coords for "
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

    def _layer_geometry(
        self,
        probe: AttachedManifoldProbe,
        layer_idx: int,
        h: torch.Tensor,
    ) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]":
        """Per-layer readout pieces, shared by the per-token + aggregate paths.

        Returns ``(frac, cdist_query, invert_query, cdist_nodes)``:

        * ``frac`` — scalar tensor in ``[0, 1]``, the in-subspace energy
          share.  Euclidean: ``‖h_par_c‖ / ‖h − mean‖``.  Whitened: the
          **M-orthogonal** share ``‖P_M(h−mean)‖_M / ‖h−mean‖_M`` =
          ``sqrt(gᵀ M_R⁻¹ g) / ‖x‖_M`` with ``g = B Σ⁻¹ x`` (an M-norm
          contraction, so still ``≤ 1``).
        * ``cdist_query`` — ``(1, R)`` query for the nearest-node cdist, in
          the metric space matching ``cdist_nodes`` (Euclidean reduced
          coords, or ``c @ chol`` so a plain cdist is the Mahalanobis
          distance).
        * ``invert_query`` — ``(R,)`` query in the RBF's reduced-coord
          space for ``invert_parameterization`` (Euclidean projection
          coords, or the M-orthogonal projection coords ``c = M_R⁻¹ g``).
        * ``cdist_nodes`` — ``(K, R)`` node coords in the same metric space
          as ``cdist_query``.
        """
        sub = probe.manifold.layers[layer_idx]
        mean = sub.mean.to(device=h.device, dtype=torch.float32)
        basis = sub.basis.to(device=h.device, dtype=torch.float32)
        wh = probe.whitened.get(layer_idx)
        if wh is None:
            # Mahalanobis-only: ``_build_whitened`` populates every fit
            # layer or raises, so a missing entry means the probe cache is
            # out of sync — never an intentional Euclidean read.
            raise WhitenerError(
                f"manifold probe read missing whitened factors for layer "
                f"{layer_idx}; rebuild the probe (the Euclidean path is gone)"
            )
        # Whitened: M-orthogonal subspace projection + Mahalanobis distance.
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
        :data:`DEFAULT_NEAREST_TOP_N` (3).  The pre-cache runs once at
        attach: for every layer the manifold covers, compute the per-node
        reduced (in-subspace) activation ``(K, R)`` for the per-token
        cdist and the normalized EV weight (per-layer EV ratio over the
        layer-union sum, floored at :data:`_MIN_EV_WEIGHT`).
        """
        if not manifold.layers:
            raise ValueError(
                f"manifold {manifold.name!r} carries no fitted layers"
            )
        if manifold.node_coords.numel() == 0 or not manifold.node_labels:
            raise ValueError(
                f"manifold {manifold.name!r} carries no node coords / labels"
            )
        node_values_reduced: dict[int, torch.Tensor] = {}
        ev_weights_raw: dict[int, float] = {}
        # Cache the embedded node coords once — they ride with the
        # manifold's domain, not per layer.
        coords = manifold.node_coords.to(torch.float32)
        clamped = manifold.domain.clamp_position(coords)
        embedded = manifold.domain.embed(clamped)  # (K, m)
        for layer_idx, sub in manifold.layers.items():
            sub_f32 = sub.to(device=sub.mean.device, dtype=torch.float32)
            embedded_dev = embedded.to(
                device=sub_f32.mean.device, dtype=torch.float32,
            )
            # Reduced (in-subspace) per-node activation: ``(K, R)`` — the
            # working space for the hot-path cdist against the running
            # activation's in-subspace projection.  Centered through
            # ``sub.mean`` so it matches the centered ``h_par_c`` slice
            # ``decompose`` returns.
            v_world = sub_f32.eval_at(embedded_dev)  # (K, D)
            v_centered = v_world - sub_f32.mean
            v_reduced = v_centered @ sub_f32.basis.T  # (K, R)
            node_values_reduced[layer_idx] = v_reduced.contiguous()
            ev_weights_raw[layer_idx] = float(
                manifold.explained_variance.get(layer_idx, 1.0)
            )
        # Normalize EV weights to sum to 1 across attached layers, with
        # the per-layer floor.  Empty EV dict (pre-v4 manifolds) falls
        # back to uniform weighting.
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
            embedded_node_coords=embedded,
            ev_weights=ev_weights,
        )
        # Build the Mahalanobis bundle (mandatory): the wired whitener must
        # cover this manifold's layers (all-or-nothing per probe), else
        # ``_build_whitened`` raises — there is no Euclidean readout.
        probe.whitened = self._build_whitened(manifold, probe)
        self._probes[name] = probe

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
                frac, cdist_query, _invert, cdist_nodes = self._layer_geometry(
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
                    self._layer_geometry(probe, layer_idx, h)
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
                if par_norm_val < _FRACTION_EPSILON:
                    norm_residual = 0.0
                else:
                    norm_residual = res_val / par_norm_val
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
