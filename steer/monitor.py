from collections import deque

import torch
import torch.nn as nn

_MAX_HISTORY = 8


class TraitMonitor:
    """Monitors model activations against a library of probe vectors.

    Each probe has a profile (dict mapping layer_idx -> (vector, score)).
    Monitoring uses all layers in each probe's profile, weighted by score.
    Hooks are grouped by layer; each active layer has one hook serving
    the subset of probes that include that layer in their profile.
    A shared accumulator aggregates weighted cosine similarities across
    layers, producing one value per probe per token.
    """

    @staticmethod
    def _empty_stats() -> dict:
        return {"count": 0, "sum": 0.0, "sum_sq": 0.0,
                "min": float("inf"), "max": float("-inf"),
                "first": 0.0, "last": 0.0}

    def __init__(self, probe_profiles: dict[str, dict[int, tuple[torch.Tensor, float]]]):
        """
        probe_profiles: maps probe name -> profile dict (layer_idx -> (vector, score))
        """
        self.probe_names: list[str] = list(probe_profiles.keys())
        self._raw_profiles: dict[str, dict[int, tuple[torch.Tensor, float]]] = dict(probe_profiles)

        # Per-layer hook state, populated by attach()
        self._layer_hooks: dict[int, dict] = {}
        self._attached = False

        # Shared accumulator state (populated by attach)
        self._accum: torch.Tensor | None = None
        self._total_weights: torch.Tensor | None = None
        self._buf_idx: int = 0
        self._max_tokens: int = 2048

        # Probe name -> index in accumulator columns
        self._probe_col: dict[str, int] = {n: i for i, n in enumerate(self.probe_names)}

        self.history: dict[str, deque[float]] = {n: deque(maxlen=_MAX_HISTORY) for n in self.probe_names}
        self._stats: dict[str, dict] = {n: self._empty_stats() for n in self.probe_names}

    def attach(self, model_layers: nn.ModuleList, device, dtype, max_tokens=2048):
        """Group probes by profile layers, build per-layer hooks with shared accumulator."""
        self._detach_hooks()
        self._max_tokens = max_tokens
        num_probes = len(self.probe_names)

        # Shared accumulator: (max_tokens, num_probes) — all hooks write into this
        self._accum = torch.zeros(max_tokens, num_probes, device=device, dtype=dtype)
        self._buf_idx = 0

        # Precompute total weight per probe (sum of scores across profile layers)
        total_weights = torch.zeros(num_probes, device=device, dtype=dtype)
        for name in self.probe_names:
            col = self._probe_col[name]
            for _layer_idx, (_vec, score) in self._raw_profiles[name].items():
                total_weights[col] += score
        self._total_weights = total_weights.clamp(min=1e-8)

        # Group probes by layer: layer_idx -> list of (probe_col, vector, score)
        layers_to_probes: dict[int, list[tuple[int, torch.Tensor, float]]] = {}
        for name in self.probe_names:
            col = self._probe_col[name]
            for layer_idx, (vec, score) in self._raw_profiles[name].items():
                layers_to_probes.setdefault(layer_idx, []).append((col, vec, score))

        # The highest layer index advances buf_idx after writing —
        # layers fire in ascending order during forward pass, so this
        # is always the last hook to run for a given token.
        max_layer_idx = max(layers_to_probes) if layers_to_probes else -1

        for layer_idx, entries in layers_to_probes.items():
            cols = [col for col, _, _ in entries]
            vecs = []
            scores = []
            for _, vec, score in entries:
                vecs.append(vec.to(device=device, dtype=dtype))
                scores.append(score)
            probe_matrix = torch.stack(vecs)  # (P_k, D)
            norms = probe_matrix.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            probe_matrix_normed = probe_matrix / norms
            score_weights = torch.tensor(scores, device=device, dtype=dtype)  # (P_k,)

            hook_state = {
                "probe_matrix_normed": probe_matrix_normed,
                "score_weights": score_weights,
                "cols": cols,
                "handle": None,
            }

            is_last = layer_idx == max_layer_idx

            def _make_hook(state, accum, col_indices, advances_idx):
                col_idx = torch.tensor(col_indices, device=device, dtype=torch.long)
                sw = state["score_weights"]
                pm = state["probe_matrix_normed"]

                def _hook(module, input, output):
                    """Hot path. One matmul, weighted scatter into shared accumulator."""
                    hidden = output[0] if isinstance(output, tuple) else output
                    last_state = hidden[0, -1]  # (D,)
                    buf_idx = self._buf_idx
                    if buf_idx < accum.shape[0]:
                        dots = pm @ last_state  # (P_k,)
                        weighted = sw * dots / last_state.norm().clamp(min=1e-8)  # (P_k,)
                        accum[buf_idx, col_idx] += weighted
                    if advances_idx:
                        self._buf_idx += 1
                    return None
                return _hook

            hook_state["handle"] = model_layers[layer_idx].register_forward_hook(
                _make_hook(hook_state, self._accum, cols, is_last)
            )
            self._layer_hooks[layer_idx] = hook_state

        self._attached = True

    def has_pending_data(self) -> bool:
        """True if the shared accumulator has unflushed data."""
        return self._buf_idx > 0

    def flush_to_cpu(self):
        """Batch-transfer accumulator to CPU history. Call from TUI poll, not per token."""
        buf_idx = self._buf_idx
        if buf_idx == 0 or self._accum is None:
            return

        was_full = buf_idx >= self._accum.shape[0]

        # Divide by total weights to get weighted average, then transfer to CPU
        cpu_data = (self._accum[:buf_idx] / self._total_weights.unsqueeze(0)).float().cpu()
        n_tokens = cpu_data.shape[0]

        # Vectorize stats across probe dimension
        sums = cpu_data.sum(dim=0)
        sum_sqs = (cpu_data ** 2).sum(dim=0)
        mins = cpu_data.min(dim=0).values
        maxs = cpu_data.max(dim=0).values
        firsts = cpu_data[0]
        lasts = cpu_data[-1]
        sums_list = sums.tolist()
        sum_sqs_list = sum_sqs.tolist()
        mins_list = mins.tolist()
        maxs_list = maxs.tolist()
        firsts_list = firsts.tolist()
        lasts_list = lasts.tolist()
        tail_data = cpu_data[max(0, n_tokens - _MAX_HISTORY):]

        for name in self.probe_names:
            i = self._probe_col[name]
            self.history[name] = deque(tail_data[:, i].tolist(), maxlen=_MAX_HISTORY)
            s = self._stats[name]
            if s["count"] == 0:
                s["first"] = firsts_list[i]
            s["count"] += n_tokens
            s["sum"] += sums_list[i]
            s["sum_sq"] += sum_sqs_list[i]
            col_min, col_max = mins_list[i], maxs_list[i]
            if col_min < s["min"]:
                s["min"] = col_min
            if col_max > s["max"]:
                s["max"] = col_max
            s["last"] = lasts_list[i]

        # Reset accumulator
        self._accum[:buf_idx].zero_()
        self._buf_idx = 0
        if was_full:
            device, dtype = self._accum.device, self._accum.dtype
            new_size = self._accum.shape[0] * 2
            self._accum = torch.zeros(new_size, self._accum.shape[1], device=device, dtype=dtype)

    def get_current_and_previous(self) -> tuple[dict[str, float], dict[str, float]]:
        """Latest and second-to-last similarity for each probe. Caller must flush_to_cpu() first."""
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
        """Return pre-computed running stats for a probe."""
        return self._stats.get(name, self._empty_stats())

    def get_sparkline(self, name: str) -> str:
        """Unicode sparkline of recent history. Caller must flush_to_cpu() first."""
        blocks = " ▁▂▃▄▅▆▇█"
        values = self.history[name]
        if not values:
            return ""
        lo, hi = min(values), max(values)
        span = hi - lo if hi != lo else 1.0
        return "".join(blocks[min(8, max(0, int((v - lo) / span * 8)))] for v in values)

    def _rebuild_from_model_layers(self, model_layers, device, dtype):
        """Full rebuild: detach, regroup, re-attach all hooks."""
        max_tokens = self._max_tokens
        if self._accum is not None:
            max_tokens = max(max_tokens, self._accum.shape[0])
        self._detach_hooks()
        self.attach(model_layers, device, dtype, max_tokens=max_tokens)

    def add_probe(self, name: str, profile: dict[int, tuple[torch.Tensor, float]],
                  model_layers=None, device=None, dtype=None):
        """Add a probe dynamically. Rebuilds layer hooks if attached."""
        self._raw_profiles[name] = profile
        if name not in self.probe_names:
            self.probe_names.append(name)
            self._probe_col[name] = len(self._probe_col)
            self.history[name] = deque(maxlen=_MAX_HISTORY)
            self._stats[name] = self._empty_stats()
        if self._attached and model_layers is not None and device is not None:
            self._rebuild_from_model_layers(model_layers, device, dtype)

    def remove_probe(self, name: str, model_layers=None, device=None, dtype=None):
        """Remove a probe. Rebuilds layer hooks if attached."""
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
        # Rebuild column indices
        self._probe_col = {n: i for i, n in enumerate(self.probe_names)}
        if self._attached and model_layers is not None and device is not None and self.probe_names:
            self._rebuild_from_model_layers(model_layers, device, dtype)

    def reset_history(self):
        """Clear all history (e.g., on new generation)."""
        for name in self.probe_names:
            self.history[name] = deque(maxlen=_MAX_HISTORY)
            self._stats[name] = self._empty_stats()
        if self._accum is not None:
            self._accum.zero_()
        self._buf_idx = 0

    def _detach_hooks(self):
        """Remove all layer hooks."""
        for state in self._layer_hooks.values():
            if state["handle"] is not None:
                state["handle"].remove()
        self._layer_hooks.clear()

    def detach(self):
        self.flush_to_cpu()
        self._detach_hooks()
        self._attached = False
