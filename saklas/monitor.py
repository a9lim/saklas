from collections import deque

import torch

_MAX_HISTORY = 8


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
        return {"count": 0, "sum": 0.0, "sum_sq": 0.0,
                "min": float("inf"), "max": float("-inf")}

    def __init__(self, probe_profiles: dict[str, dict[int, tuple[torch.Tensor, float]]],
                 layer_means: dict[int, torch.Tensor] | None = None):
        """
        probe_profiles: maps probe name -> profile dict (layer_idx -> (vector, score))
        layer_means: maps layer_idx -> mean activation vector for centering
        """
        self._raw_profiles: dict[str, dict[int, tuple[torch.Tensor, float]]] = dict(probe_profiles)
        self._layer_means: dict[int, torch.Tensor] = dict(layer_means) if layer_means else {}

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
        self._layer_means = dict(value) if value else {}

    def _score_probes(self, hidden_per_layer: dict[int, torch.Tensor]):
        """Score all probes against hidden states and update history/stats."""
        for name in self._raw_profiles:
            total_w = 0.0
            weighted_sim = 0.0
            for layer_idx, (vec, score) in self._raw_profiles[name].items():
                if layer_idx not in hidden_per_layer:
                    continue
                total_w += score
                h = hidden_per_layer[layer_idx].float()
                mean = self._layer_means.get(layer_idx)
                if mean is not None:
                    h = h - mean.to(h.device)
                v = vec.to(h.device).float()
                cos = (h @ v) / (h.norm() * v.norm()).clamp(min=1e-8)
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

    def remove_probe(self, name: str):
        if name in self._raw_profiles:
            del self._raw_profiles[name]
        if name in self.history:
            del self.history[name]
        if name in self._stats:
            del self._stats[name]

    def reset_history(self):
        for name in self._raw_profiles:
            self.history[name].clear()
            self._stats[name] = self._empty_stats()
        self._pending = False
