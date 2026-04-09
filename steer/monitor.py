import torch
import torch.nn as nn
import numpy as np


class TraitMonitor:
    """Monitors model activations against a library of probe vectors."""

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
        self.history: dict[str, list[float]] = {n: [] for n in self.probe_names}

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
        last_state = hidden[0, -1, :]  # (D,) — last token of batch dim 0
        norm = last_state.norm()
        if self._buf_idx < self._gpu_buffer.shape[0]:
            if norm > 1e-8:
                normed_state = last_state / norm
                self._gpu_buffer[self._buf_idx] = self._probe_matrix_normed @ normed_state
            else:
                self._gpu_buffer[self._buf_idx].zero_()
            self._buf_idx += 1
        return None  # read-only hook

    def flush_to_cpu(self):
        """Batch-transfer GPU buffer to CPU history. Call from TUI poll, not per token."""
        if self._buf_idx == 0:
            return
        cpu_data = self._gpu_buffer[:self._buf_idx].float().cpu().numpy()
        for i, name in enumerate(self.probe_names):
            self.history[name].extend(cpu_data[:, i].tolist())
        self._buf_idx = 0

    def get_current(self) -> dict[str, float]:
        """Latest similarity for each probe."""
        if self._buf_idx > 0:
            last = self._gpu_buffer[self._buf_idx - 1].float().cpu().numpy()
            return {name: float(last[i]) for i, name in enumerate(self.probe_names)}
        return {name: (hist[-1] if hist else 0.0) for name, hist in self.history.items()}

    def get_previous(self) -> dict[str, float]:
        """Second-to-last similarity (for direction arrows)."""
        if self._buf_idx > 1:
            prev = self._gpu_buffer[self._buf_idx - 2].float().cpu().numpy()
            return {name: float(prev[i]) for i, name in enumerate(self.probe_names)}
        # Fall back to CPU history
        result = {}
        for name in self.probe_names:
            hist = self.history[name]
            if len(hist) >= 2:
                result[name] = hist[-2]
            elif self._buf_idx == 1 and hist:
                result[name] = hist[-1]
            else:
                result[name] = 0.0
        return result

    def get_sparkline(self, name: str, width: int = 64) -> str:
        """Unicode sparkline of recent history. Caller must flush_to_cpu() first."""
        blocks = " ▁▂▃▄▅▆▇█"
        raw = self.history[name][-width:]
        values = [v for v in raw if v == v]  # drop NaN (NaN != NaN)
        if not values:
            return ""
        lo, hi = min(values), max(values)
        span = hi - lo if hi != lo else 1.0
        return "".join(blocks[min(8, max(0, int((v - lo) / span * 8)))] for v in values)

    def add_probe(self, name: str, vector: torch.Tensor, device=None, dtype=None):
        """Add a probe dynamically. Rebuilds the probe matrix."""
        self._raw_probes[name] = vector
        if name not in self.probe_names:
            self.probe_names.append(name)
            self.history[name] = []
        if self._handle is not None and device is not None:
            # Rebuild probe matrix
            vecs = [self._raw_probes[n].to(device=device, dtype=dtype) for n in self.probe_names]
            probe_matrix = torch.stack(vecs)
            norms = probe_matrix.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            self._probe_matrix_normed = probe_matrix / norms
            # Resize GPU buffer if needed
            old_buf = self._gpu_buffer
            self._gpu_buffer = torch.zeros(old_buf.shape[0], len(self.probe_names), device=device, dtype=dtype)
            if self._buf_idx > 0:
                self._gpu_buffer[:self._buf_idx, :old_buf.shape[1]] = old_buf[:self._buf_idx]

    def remove_probe(self, name: str, device=None, dtype=None):
        """Remove a probe. Rebuilds the probe matrix."""
        if name in self._raw_probes:
            del self._raw_probes[name]
        if name in self.probe_names:
            self.probe_names.remove(name)
        if name in self.history:
            del self.history[name]
        if self._handle is not None and device is not None and self.probe_names:
            vecs = [self._raw_probes[n].to(device=device, dtype=dtype) for n in self.probe_names]
            probe_matrix = torch.stack(vecs)
            norms = probe_matrix.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            self._probe_matrix_normed = probe_matrix / norms

    def reset_history(self):
        """Clear all history (e.g., on new generation)."""
        self.flush_to_cpu()
        for name in self.probe_names:
            self.history[name] = []
        self._buf_idx = 0

    def detach(self):
        if self._handle:
            self._handle.remove()
            self._handle = None
        self.flush_to_cpu()
