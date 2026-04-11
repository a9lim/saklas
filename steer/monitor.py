from collections import deque

import torch
import torch.nn as nn

_MAX_HISTORY = 8


def _peak_layer(profile: dict[int, tuple[torch.Tensor, float]]) -> int:
    """Return the layer index with the highest score in a profile."""
    return max(profile, key=lambda k: profile[k][1])


class TraitMonitor:
    """Monitors model activations against a library of probe vectors.

    Each probe has a profile (dict mapping layer_idx -> (vector, score)).
    Monitoring uses the peak layer per probe — the layer with the strongest
    contrastive signal. Hooks are grouped by layer so each active layer
    has one hook serving a subset of probes.
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
        self._probe_layer: dict[str, int] = {
            name: _peak_layer(prof) for name, prof in probe_profiles.items()
        }

        # Per-layer hook state, populated by attach()
        self._layer_hooks: dict[int, dict] = {}
        self._attached = False

        self.history: dict[str, deque[float]] = {n: deque(maxlen=_MAX_HISTORY) for n in self.probe_names}
        self._stats: dict[str, dict] = {n: self._empty_stats() for n in self.probe_names}

    def attach(self, model_layers: nn.ModuleList, device, dtype, max_tokens=2048):
        """Group probes by peak layer, build per-layer hooks."""
        self._detach_hooks()

        # Group probes by their peak layer
        layers_to_probes: dict[int, list[str]] = {}
        for name in self.probe_names:
            layer_idx = self._probe_layer[name]
            layers_to_probes.setdefault(layer_idx, []).append(name)

        for layer_idx, probe_names in layers_to_probes.items():
            # Build probe matrix for this layer using the vector at the peak layer
            vecs = []
            for name in probe_names:
                vec, _score = self._raw_profiles[name][layer_idx]
                vecs.append(vec.to(device=device, dtype=dtype))
            probe_matrix = torch.stack(vecs)  # (P_k, D)
            norms = probe_matrix.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            probe_matrix_normed = probe_matrix / norms

            gpu_buffer = torch.zeros(max_tokens, len(probe_names), device=device, dtype=dtype)

            hook_state = {
                "probe_matrix_normed": probe_matrix_normed,
                "gpu_buffer": gpu_buffer,
                "buf_idx": 0,
                "handle": None,
                "probe_names": probe_names,
            }

            def _make_hook(state):
                def _hook(module, input, output):
                    """Hot path. One matmul, no .item(), no CPU sync."""
                    hidden = output[0] if isinstance(output, tuple) else output
                    last_state = hidden[0, -1]  # (D,)
                    if state["buf_idx"] < state["gpu_buffer"].shape[0]:
                        dots = state["probe_matrix_normed"] @ last_state
                        state["gpu_buffer"][state["buf_idx"]] = dots / last_state.norm().clamp(min=1e-8)
                        state["buf_idx"] += 1
                    return None  # read-only hook
                return _hook

            hook_state["handle"] = model_layers[layer_idx].register_forward_hook(_make_hook(hook_state))
            self._layer_hooks[layer_idx] = hook_state

        self._attached = True

    def has_pending_data(self) -> bool:
        """True if any layer's GPU buffer has unflushed data."""
        return any(state["buf_idx"] > 0 for state in self._layer_hooks.values())

    def flush_to_cpu(self):
        """Batch-transfer GPU buffers to CPU history. Call from TUI poll, not per token."""
        for state in self._layer_hooks.values():
            buf_idx = state["buf_idx"]
            if buf_idx == 0:
                continue

            was_full = buf_idx >= state["gpu_buffer"].shape[0]
            cpu_data = state["gpu_buffer"][:buf_idx].float().cpu()
            n_tokens = cpu_data.shape[0]
            probe_names = state["probe_names"]

            # Vectorize across probe dimension
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

            for i, name in enumerate(probe_names):
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

            state["buf_idx"] = 0
            if was_full:
                state["gpu_buffer"] = torch.zeros(
                    state["gpu_buffer"].shape[0] * 2, state["gpu_buffer"].shape[1],
                    device=state["gpu_buffer"].device, dtype=state["gpu_buffer"].dtype,
                )

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
        # Preserve buffer sizes
        max_tokens = 2048
        for state in self._layer_hooks.values():
            max_tokens = max(max_tokens, state["gpu_buffer"].shape[0])
        self._detach_hooks()
        self.attach(model_layers, device, dtype, max_tokens=max_tokens)

    def add_probe(self, name: str, profile: dict[int, tuple[torch.Tensor, float]],
                  model_layers=None, device=None, dtype=None):
        """Add a probe dynamically. Rebuilds layer hooks if attached."""
        self._raw_profiles[name] = profile
        self._probe_layer[name] = _peak_layer(profile)
        if name not in self.probe_names:
            self.probe_names.append(name)
            self.history[name] = deque(maxlen=_MAX_HISTORY)
            self._stats[name] = self._empty_stats()
        if self._attached and model_layers is not None and device is not None:
            self._rebuild_from_model_layers(model_layers, device, dtype)

    def remove_probe(self, name: str, model_layers=None, device=None, dtype=None):
        """Remove a probe. Rebuilds layer hooks if attached."""
        if name in self._raw_profiles:
            del self._raw_profiles[name]
        if name in self._probe_layer:
            del self._probe_layer[name]
        if name in self.probe_names:
            self.probe_names.remove(name)
        if name in self.history:
            del self.history[name]
        if name in self._stats:
            del self._stats[name]
        if self._attached and model_layers is not None and device is not None and self.probe_names:
            self._rebuild_from_model_layers(model_layers, device, dtype)

    def reset_history(self):
        """Clear all history (e.g., on new generation)."""
        for name in self.probe_names:
            self.history[name] = deque(maxlen=_MAX_HISTORY)
            self._stats[name] = self._empty_stats()
        for state in self._layer_hooks.values():
            state["buf_idx"] = 0

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
