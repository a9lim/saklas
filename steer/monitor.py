from collections import deque

import torch

_MAX_HISTORY = 8


class TraitMonitor:
    """Monitors model activations against a library of probe vectors.

    Each probe has a profile (dict mapping layer_idx -> (vector, score)).
    After generation, a single forward pass over the generated text
    produces attention-weighted hidden states per layer.  Mean-centered
    cosine similarities against probe vectors, weighted by score, give
    one value per probe per generation.
    """

    @staticmethod
    def _empty_stats() -> dict:
        return {"count": 0, "sum": 0.0, "sum_sq": 0.0,
                "min": float("inf"), "max": float("-inf"),
                "first": 0.0, "last": 0.0}

    def __init__(self, probe_profiles: dict[str, dict[int, tuple[torch.Tensor, float]]],
                 layer_means: dict[int, torch.Tensor] | None = None):
        """
        probe_profiles: maps probe name -> profile dict (layer_idx -> (vector, score))
        layer_means: maps layer_idx -> mean activation vector for centering
        """
        self.probe_names: list[str] = list(probe_profiles.keys())
        self._raw_profiles: dict[str, dict[int, tuple[torch.Tensor, float]]] = dict(probe_profiles)
        self._layer_means: dict[int, torch.Tensor] = dict(layer_means) if layer_means else {}

        self._probe_col: dict[str, int] = {n: i for i, n in enumerate(self.probe_names)}

        self.history: dict[str, deque[float]] = {n: deque(maxlen=_MAX_HISTORY) for n in self.probe_names}
        self._stats: dict[str, dict] = {n: self._empty_stats() for n in self.probe_names}

        # Set after measure() — signals TUI to refresh
        self._pending = False

    def measure(self, model, tokenizer, layers, text: str, device=None):
        """Run one forward pass over *text* and compute probe similarities.

        Uses attention-weighted pooling (same as extraction) to produce
        one hidden-state vector per layer, mean-centers, then computes
        score-weighted cosine similarities against all probes.
        """
        from steer.vectors import _encode_and_capture_all

        if device is None:
            device = next(model.parameters()).device

        hidden_per_layer = _encode_and_capture_all(model, tokenizer, text, layers, device)

        num_probes = len(self.probe_names)
        sims = torch.zeros(num_probes)

        # Total weight per probe for normalization
        total_weights = torch.zeros(num_probes)
        for name in self.probe_names:
            col = self._probe_col[name]
            for _idx, (_vec, score) in self._raw_profiles[name].items():
                total_weights[col] += score

        for name in self.probe_names:
            col = self._probe_col[name]
            for layer_idx, (vec, score) in self._raw_profiles[name].items():
                if layer_idx not in hidden_per_layer:
                    continue
                h = hidden_per_layer[layer_idx].float()
                mean = self._layer_means.get(layer_idx)
                if mean is not None:
                    h = h - mean.to(h.device).float()
                v = vec.to(h.device).float()
                cos = (h @ v) / (h.norm().clamp(min=1e-8) * v.norm().clamp(min=1e-8))
                sims[col] += score * cos.item()

        # Normalize by total weight
        total_weights.clamp_(min=1e-8)
        sims /= total_weights
        values = sims.tolist()

        for name in self.probe_names:
            col = self._probe_col[name]
            val = values[col]
            self.history[name].append(val)
            s = self._stats[name]
            if s["count"] == 0:
                s["first"] = val
            s["count"] += 1
            s["sum"] += val
            s["sum_sq"] += val * val
            if val < s["min"]:
                s["min"] = val
            if val > s["max"]:
                s["max"] = val
            s["last"] = val

        self._pending = True

    def has_pending_data(self) -> bool:
        return self._pending

    def consume_pending(self) -> None:
        """Mark pending data as consumed (called by TUI after reading)."""
        self._pending = False

    def get_current_and_previous(self) -> tuple[dict[str, float], dict[str, float]]:
        current = {}
        previous = {}
        for name in self.probe_names:
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
        self._raw_profiles[name] = profile
        if name not in self.probe_names:
            self.probe_names.append(name)
            self._probe_col[name] = len(self._probe_col)
            self.history[name] = deque(maxlen=_MAX_HISTORY)
            self._stats[name] = self._empty_stats()

    def remove_probe(self, name: str):
        if name in self._raw_profiles:
            del self._raw_profiles[name]
        if name in self._probe_col:
            del self._probe_col[name]
        if name in self.probe_names:
            self.probe_names.remove(name)
        if name in self.history:
            del self.history[name]
        if name in self._stats:
            del self._stats[name]
        self._probe_col = {n: i for i, n in enumerate(self.probe_names)}

    def reset_history(self):
        for name in self.probe_names:
            self.history[name] = deque(maxlen=_MAX_HISTORY)
            self._stats[name] = self._empty_stats()
        self._pending = False
