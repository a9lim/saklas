from collections import deque

import torch

_MAX_HISTORY = 8

_EMPTY_STATS = {"count": 0, "sum": 0.0, "sum_sq": 0.0,
                "min": float("inf"), "max": float("-inf")}


class TraitMonitor:
    """Monitors model activations against a library of probe vectors.

    Each probe has a profile (dict mapping layer_idx -> (vector, score)).
    After generation, a single forward pass over the generated text
    pools the last content token's hidden state at each layer.
    Mean-centered cosine similarities against probe vectors, weighted by
    score, give one value per probe per generation.
    """

    @staticmethod
    def _empty_stats() -> dict:
        return dict(_EMPTY_STATS)

    def __init__(self, probe_profiles: dict[str, dict[int, tuple[torch.Tensor, float]]],
                 layer_means: dict[int, torch.Tensor] | None = None):
        """
        probe_profiles: maps probe name -> profile dict (layer_idx -> (vector, score))
        layer_means: maps layer_idx -> mean activation vector for centering
        """
        self._raw_profiles: dict[str, dict[int, tuple[torch.Tensor, float]]] = dict(probe_profiles)
        self._layer_means: dict[int, torch.Tensor] = dict(layer_means) if layer_means else {}

        # Cache of probe vectors pre-normalized to unit float32 on a specific device.
        # Structure: {probe_name: {layer_idx: v_unit_tensor}}, plus a single _cache_device.
        self._v_unit_cache: dict[str, dict[int, torch.Tensor]] = {}
        self._cache_device: torch.device | None = None
        # Cache of layer_means cast to float32 on cache_device.
        self._mean_cache: dict[int, torch.Tensor] = {}

        self.history: dict[str, deque[float]] = {n: deque(maxlen=_MAX_HISTORY) for n in self._raw_profiles}
        self._stats: dict[str, dict] = {n: self._empty_stats() for n in self._raw_profiles}

        # Set after measure() — signals TUI to refresh
        self._pending = False

    @property
    def probe_names(self) -> list[str]:
        """Probe names in insertion order."""
        return list(self._raw_profiles.keys())

    @property
    def profiles(self) -> dict[str, dict[int, tuple[torch.Tensor, float]]]:
        """Probe profiles: name -> {layer_idx: (vector, score)}."""
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
        """Build/refresh the per-device float32 cache of unit probe vectors + means."""
        if self._cache_device == device and self._v_unit_cache.keys() == self._raw_profiles.keys():
            # Also verify inner layer sets match (probe replaced via add_probe invalidates).
            ok = True
            for name, prof in self._raw_profiles.items():
                cached = self._v_unit_cache.get(name)
                if cached is None or cached.keys() != prof.keys():
                    ok = False
                    break
            if ok and self._mean_cache.keys() == self._layer_means.keys():
                return

        self._cache_device = device
        new_cache: dict[int, dict[int, torch.Tensor]] = {}
        for name, prof in self._raw_profiles.items():
            per_layer: dict[int, torch.Tensor] = {}
            for layer_idx, (vec, _score) in prof.items():
                v = vec.to(device=device, dtype=torch.float32)
                vn = v.norm().clamp(min=1e-8)
                per_layer[layer_idx] = v / vn
            new_cache[name] = per_layer
        self._v_unit_cache = new_cache

        self._mean_cache = {
            idx: m.to(device=device, dtype=torch.float32)
            for idx, m in self._layer_means.items()
        }

    def _score_probes(self, hidden_per_layer: dict[int, torch.Tensor]):
        """Score all probes against hidden states and update history/stats."""
        # Pick a device from any hidden state (they share device).
        device = None
        for h in hidden_per_layer.values():
            device = h.device
            break
        if device is not None:
            self._ensure_cache(device)

        # Pre-cast each layer's hidden state once; compute centered h / ||h||.
        h_unit_per_layer: dict[int, torch.Tensor] = {}
        for layer_idx, h_raw in hidden_per_layer.items():
            h = h_raw.float()
            mean = self._mean_cache.get(layer_idx)
            if mean is not None:
                h = h - mean
            hn = h.norm().clamp(min=1e-8)
            h_unit_per_layer[layer_idx] = h / hn

        for name in self._raw_profiles:
            total_w = 0.0
            weighted_sim = 0.0
            v_unit_layers = self._v_unit_cache.get(name, {})
            for layer_idx, (_vec, score) in self._raw_profiles[name].items():
                h_unit = h_unit_per_layer.get(layer_idx)
                if h_unit is None:
                    continue
                total_w += score
                v_unit = v_unit_layers[layer_idx]
                cos = (h_unit @ v_unit)
                weighted_sim += score * cos.item()
            total_w = max(total_w, 1e-8)
            val = weighted_sim / total_w
            self.history[name].append(val)
            s = self._stats[name]
            s["count"] += 1
            s["sum"] += val
            s["sum_sq"] += val * val
            if val < s["min"]:
                s["min"] = val
            if val > s["max"]:
                s["max"] = val

        self._pending = True

    def measure(self, model, tokenizer, layers, text: str, device=None):
        """Run one forward pass over *text* and compute probe similarities.

        Pools the last content token's hidden state per layer (same as
        extraction), mean-centers, then computes score-weighted cosine
        similarities against all probes.
        """
        from saklas.vectors import _encode_and_capture_all

        if device is None:
            device = next(model.parameters()).device

        hidden_per_layer = _encode_and_capture_all(model, tokenizer, text, layers, device)
        self._score_probes(hidden_per_layer)

    def measure_from_hidden(self, hidden_per_layer: dict[int, torch.Tensor]):
        """Score probes from pre-captured hidden states (no forward pass).

        Use when hidden states have already been captured during generation
        (e.g. via capture hooks), avoiding a redundant forward pass.
        """
        self._score_probes(hidden_per_layer)

    def has_pending_data(self) -> bool:
        return self._pending

    def consume_pending(self) -> None:
        """Mark pending data as consumed (called by TUI after reading)."""
        self._pending = False

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

    def add_probe(self, name: str, profile: dict[int, tuple[torch.Tensor, float]]):
        is_new = name not in self._raw_profiles
        self._raw_profiles[name] = profile
        if is_new:
            self.history[name] = deque(maxlen=_MAX_HISTORY)
            self._stats[name] = self._empty_stats()
        # Drop any stale cached unit vectors for this probe; rebuilt lazily.
        self._v_unit_cache.pop(name, None)

    def remove_probe(self, name: str):
        if name in self._raw_profiles:
            del self._raw_profiles[name]
        if name in self.history:
            del self.history[name]
        if name in self._stats:
            del self._stats[name]
        self._v_unit_cache.pop(name, None)

    def reset_history(self):
        for name in self._raw_profiles:
            self.history[name].clear()
            self._stats[name] = self._empty_stats()
        self._pending = False
