from collections import deque

import torch

_MAX_HISTORY = 8

_EMPTY_STATS = {"count": 0, "sum": 0.0, "sum_sq": 0.0,
                "min": float("inf"), "max": float("-inf")}


class TraitMonitor:
    """Monitors model activations against a library of probe vectors.

    Each probe has a profile (dict mapping layer_idx -> baked direction).
    After generation, a single forward pass over the generated text
    pools the last content token's hidden state at each layer.
    Mean-centered cosine similarities against probe vectors, weighted by
    the baked magnitude ||baked_i|| (which encodes share * ref_norm — the
    same "how much does this layer steer per unit alpha" quantity),
    give one value per probe per generation.
    """

    @staticmethod
    def _empty_stats() -> dict:
        return dict(_EMPTY_STATS)

    def __init__(self, probe_profiles: dict[str, dict[int, torch.Tensor]],
                 layer_means: dict[int, torch.Tensor] | None = None):
        """
        probe_profiles: maps probe name -> profile dict (layer_idx -> baked vector)
        layer_means: maps layer_idx -> mean activation vector for centering
        """
        self._raw_profiles: dict[str, dict[int, torch.Tensor]] = dict(probe_profiles)
        self._layer_means: dict[int, torch.Tensor] = dict(layer_means) if layer_means else {}

        # Per-layer stacked cache, inverted from the previous {probe: {layer: ...}}
        # form so one matmul scores every probe against a hidden state in a single
        # kernel launch. For each layer that any probe covers:
        #   V[P, D]  : unit-normed probe directions (zeros for probes missing L)
        #   W[P]     : per-probe weight at layer L (= ||baked_L||, 0 if missing)
        # The denominator per probe is built on-device as sum of W across layers
        # where hidden is also present; no .item() on the hot path.
        # Structure: {layer_idx: (V_stacked, W_stacked)}.
        self._layer_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        # Probe -> index into the P axis (stable, insertion order).
        self._probe_index: dict[str, int] = {}
        self._cache_device: torch.device | None = None
        self._cache_probe_keys: tuple[str, ...] = ()
        # Cache of layer_means cast to float32 on cache_device.
        self._mean_cache: dict[int, torch.Tensor] = {}

        self.history: dict[str, deque[float]] = {n: deque(maxlen=_MAX_HISTORY) for n in self._raw_profiles}
        self._stats: dict[str, dict] = {n: self._empty_stats() for n in self._raw_profiles}

        # Aggregate path sets _pending_aggregate; per-token path sets _pending_per_token.
        # has_pending_data() returns aggregate readiness — the TUI uses it to refresh
        # trait readings after a measure() call.
        self._pending_aggregate = False
        self._pending_per_token = False

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
        if value is not None and not isinstance(value, dict):
            raise TypeError(f"layer_means must be a dict, got {type(value).__name__}")
        self._layer_means = dict(value) if value else {}
        # Invalidate mean cache; v_unit cache is independent of means.
        self._mean_cache = {}

    def _ensure_cache(self, device: torch.device) -> None:
        """Build/refresh the per-device float32 cache of stacked probe matrices + means.

        Builds one ``(V[P,D], W[P])`` pair per layer that any probe covers. ``V`` holds
        unit-normed directions (rows for probes missing that layer are zero, which
        produces zero similarity — correct because ``W`` at that slot is also zero and
        the denominator mask is shared). ``W[p] = ||baked_p_L||`` for probes that own
        the layer, else 0. No ``.item()`` calls — norms stay on-device.
        """
        probe_keys = tuple(self._raw_profiles.keys())
        if (
            self._cache_device == device
            and self._cache_probe_keys == probe_keys
            and self._mean_cache.keys() == self._layer_means.keys()
            and self._layer_cache
        ):
            return

        self._cache_device = device
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

        new_layer_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
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
        self._layer_cache = new_layer_cache

        self._mean_cache = {
            idx: m.to(device=device, dtype=torch.float32)
            for idx, m in self._layer_means.items()
        }

    def _normalize_hidden(self, layer_idx: int, h: torch.Tensor) -> torch.Tensor:
        """Center (per layer_means) and L2-normalize a hidden-state tensor.

        Works for both 1D (dim,) and 2D (seq, dim) — mean broadcasts over
        rows, and ``dim=-1, keepdim=True`` is a no-op reshape for 1D.
        """
        mean = self._mean_cache.get(layer_idx)
        if mean is not None:
            h = h - mean
        hn = h.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return h / hn

    def _score_probes(self, hidden_per_layer: dict[int, torch.Tensor], accumulate: bool = True) -> dict[str, float]:
        """Score all probes against hidden states.

        When ``accumulate`` is True (default), history and stats are updated.
        When False, the call is read-only — useful for stateless API requests
        that must not mutate session-level probe accumulators.
        """
        probe_keys = self._cache_probe_keys if self._cache_device is not None else tuple(self._raw_profiles.keys())
        if not hidden_per_layer:
            vals = {name: 0.0 for name in self._raw_profiles}
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
            V, W = entry  # V: (P, D), W: (P,)
            h_unit = self._normalize_hidden(layer_idx, h.float())  # (D,)
            sims = V @ h_unit  # (P,)
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

    def measure(self, model, tokenizer, layers, text: str, device=None, accumulate: bool = True) -> dict[str, float]:
        """Run one forward pass over *text* and compute probe similarities.

        Pools the last content token's hidden state per layer (same as
        extraction), mean-centers, then computes score-weighted cosine
        similarities against all probes. When ``accumulate`` is False,
        history and stats are left untouched.
        """
        from saklas.core.vectors import _encode_and_capture_all

        if device is None:
            device = next(model.parameters()).device

        hidden_per_layer = _encode_and_capture_all(model, tokenizer, text, layers, device)
        return self._score_probes(hidden_per_layer, accumulate=accumulate)

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
        tokenizer,
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
        empty_agg = {name: 0.0 for name in self._raw_profiles}
        if n == 0 or not captured:
            return empty_agg, {name: [] for name in self._raw_profiles}

        any_h = next(iter(captured.values()))
        self._ensure_cache(any_h.device)

        # Aggregate pool: last non-special generated token.
        special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
        agg_idx = n - 1
        while agg_idx > 0 and int(generated_ids[agg_idx]) in special_ids:
            agg_idx -= 1
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
            if h.shape[0] != n:
                continue
            entry = self._layer_cache.get(layer_idx)
            if entry is None:
                continue
            V, W = entry  # V: (P, D), W: (P,)
            h_unit = self._normalize_hidden(layer_idx, h.float())  # (n, D)
            sims = h_unit @ V.t()  # (n, P)
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

        self._pending_per_token = True
        return agg_vals, per_token

    def has_pending_data(self) -> bool:
        """True iff an aggregate measurement is waiting to be consumed."""
        return self._pending_aggregate

    def has_pending_per_token(self) -> bool:
        return self._pending_per_token

    def consume_pending(self) -> None:
        """Mark aggregate pending data as consumed (called by TUI after reading)."""
        self._pending_aggregate = False

    def consume_pending_per_token(self) -> None:
        self._pending_per_token = False

    def get_current_and_previous(self) -> tuple[dict[str, float], dict[str, float]]:
        current = {}
        previous = {}
        for name in self._raw_profiles:
            hist = self.history[name]
            if len(hist) >= 2:
                current[name] = hist[-1]
                previous[name] = hist[-2]
            elif hist:
                current[name] = hist[-1]
                previous[name] = hist[-1]
            else:
                current[name] = 0.0
                previous[name] = 0.0
        return current, previous

    def get_stats(self, name: str) -> dict:
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
