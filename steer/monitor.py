from collections import deque

import torch
import torch.nn as nn

_MAX_HISTORY = 8


class TraitMonitor:
    """Monitors model activations against a library of probe vectors."""

    @staticmethod
    def _empty_stats() -> dict:
        return {"count": 0, "sum": 0.0, "sum_sq": 0.0,
                "min": float("inf"), "max": float("-inf"),
                "first": 0.0, "last": 0.0}

    def __init__(self, probe_dict: dict[str, torch.Tensor], monitor_layer_idx: int):
        """
        probe_dict: maps probe name -> unit vector (hidden_dim,)
        monitor_layer_idx: which layer to hook
        """
        self.probe_names: list[str] = list(probe_dict.keys())
        self.monitor_layer_idx = monitor_layer_idx
        self._handle = None
        self._probe_matrix_normed: torch.Tensor | None = None
        self._raw_probes = probe_dict
        self._gpu_buffer: torch.Tensor | None = None
        self._buf_idx: int = 0
        self.history: dict[str, deque[float]] = {n: deque(maxlen=_MAX_HISTORY) for n in self.probe_names}
        self._stats: dict[str, dict] = {n: self._empty_stats() for n in self.probe_names}

    def attach(self, model_layers: nn.ModuleList, device, dtype, max_tokens=2048):
        """Pre-stack probes into matrix, pre-allocate GPU buffer, register hook."""
        vecs = [self._raw_probes[n].to(device=device, dtype=dtype) for n in self.probe_names]
        probe_matrix = torch.stack(vecs)  # (P, D)
        norms = probe_matrix.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        self._probe_matrix_normed = probe_matrix / norms  # (P, D) unit vectors
        self._gpu_buffer = torch.zeros(max_tokens, len(self.probe_names), device=device, dtype=dtype)
        self._buf_idx = 0
        layer = model_layers[self.monitor_layer_idx]
        self._handle = layer.register_forward_hook(self._hook)

    def _hook(self, module, input, output):
        """Hot path. One matmul, no .item(), no CPU sync."""
        hidden = output[0] if isinstance(output, tuple) else output
        last_state = hidden[0, -1]  # (D,) — last token of batch dim 0
        if self._buf_idx < self._gpu_buffer.shape[0]:
            normed_state = last_state / last_state.norm().clamp(min=1e-8)
            self._gpu_buffer[self._buf_idx] = self._probe_matrix_normed @ normed_state
            self._buf_idx += 1
        return None  # read-only hook

    def has_pending_data(self) -> bool:
        """True if the GPU buffer has unflushed data."""
        return self._buf_idx > 0

    def flush_to_cpu(self):
        """Batch-transfer GPU buffer to CPU history. Call from TUI poll, not per token."""
        if self._buf_idx == 0:
            return
        was_full = self._buf_idx >= self._gpu_buffer.shape[0]
        cpu_data = self._gpu_buffer[:self._buf_idx].float().cpu()
        n_tokens = cpu_data.shape[0]
        # Vectorize across probe dimension
        sums = cpu_data.sum(dim=0)          # (P,)
        sum_sqs = (cpu_data ** 2).sum(dim=0)  # (P,)
        mins = cpu_data.min(dim=0).values   # (P,)
        maxs = cpu_data.max(dim=0).values   # (P,)
        firsts = cpu_data[0]                # (P,)
        lasts = cpu_data[-1]                # (P,)
        for i, name in enumerate(self.probe_names):
            self.history[name].extend(cpu_data[:, i].tolist())
            s = self._stats[name]
            if s["count"] == 0:
                s["first"] = firsts[i].item()
            s["count"] += n_tokens
            s["sum"] += sums[i].item()
            s["sum_sq"] += sum_sqs[i].item()
            col_min, col_max = mins[i].item(), maxs[i].item()
            if col_min < s["min"]:
                s["min"] = col_min
            if col_max > s["max"]:
                s["max"] = col_max
            s["last"] = lasts[i].item()
        self._buf_idx = 0
        if was_full:
            self._gpu_buffer = torch.empty(
                self._gpu_buffer.shape[0] * 2, self._gpu_buffer.shape[1],
                device=self._gpu_buffer.device, dtype=self._gpu_buffer.dtype,
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

    def get_sparkline(self, name: str, width: int = _MAX_HISTORY) -> str:
        """Unicode sparkline of recent history. Caller must flush_to_cpu() first."""
        blocks = " ▁▂▃▄▅▆▇█"
        values = self.history[name][-width:]
        if not values:
            return ""
        lo, hi = min(values), max(values)
        span = hi - lo if hi != lo else 1.0
        return "".join(blocks[min(8, max(0, int((v - lo) / span * 8)))] for v in values)

    def _rebuild_probe_matrix(self, device, dtype):
        """Rebuild the normalized probe matrix and resize the GPU buffer."""
        self.flush_to_cpu()
        vecs = [self._raw_probes[n].to(device=device, dtype=dtype) for n in self.probe_names]
        probe_matrix = torch.stack(vecs)
        norms = probe_matrix.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        self._probe_matrix_normed = probe_matrix / norms
        self._gpu_buffer = torch.zeros(
            self._gpu_buffer.shape[0], len(self.probe_names),
            device=device, dtype=dtype,
        )

    def add_probe(self, name: str, vector: torch.Tensor, device=None, dtype=None):
        """Add a probe dynamically. Rebuilds the probe matrix."""
        self._raw_probes[name] = vector
        if name not in self.probe_names:
            self.probe_names.append(name)
            self.history[name] = deque(maxlen=_MAX_HISTORY)
            self._stats[name] = self._empty_stats()
        if self._handle is not None and device is not None:
            self._rebuild_probe_matrix(device, dtype)

    def remove_probe(self, name: str, device=None, dtype=None):
        """Remove a probe. Rebuilds the probe matrix and GPU buffer."""
        if name in self._raw_probes:
            del self._raw_probes[name]
        if name in self.probe_names:
            self.probe_names.remove(name)
        if name in self.history:
            del self.history[name]
        if name in self._stats:
            del self._stats[name]
        if self._handle is not None and device is not None and self.probe_names:
            self._rebuild_probe_matrix(device, dtype)

    def reset_history(self):
        """Clear all history (e.g., on new generation)."""
        for name in self.probe_names:
            self.history[name] = deque(maxlen=_MAX_HISTORY)
            self._stats[name] = self._empty_stats()
        self._buf_idx = 0

    def detach(self):
        if self._handle:
            self._handle.remove()
            self._handle = None
        self.flush_to_cpu()
