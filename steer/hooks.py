"""Steering hooks for activation steering on transformer models."""

from __future__ import annotations

import torch


class SteeringHook:
    """Pre-composed steering vector for a single layer."""

    def __init__(self) -> None:
        self.composed: torch.Tensor | None = None
        self._handle = None

    def recompose(
        self,
        vectors: list[tuple[torch.Tensor, float]],
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Pre-compose all vectors for this layer into a single tensor."""
        if not vectors:
            self.composed = None
            return
        # All-zero alphas → no perturbation; skip the matmul so that
        # 0 * NaN (from a bad extraction) doesn't inject NaN into hooks.
        if all(alpha == 0.0 for _, alpha in vectors):
            self.composed = None
            return
        alphas = torch.tensor([alpha for _, alpha in vectors], device=device, dtype=dtype)
        stacked = torch.stack([vec.to(device=device, dtype=dtype) for vec, _ in vectors])
        self.composed = (alphas.unsqueeze(1) * stacked).sum(dim=0)

    def hook_fn(self, module, input, output):
        if self.composed is not None:
            if isinstance(output, torch.Tensor):
                output.add_(self.composed)
                return output
            output[0].add_(self.composed)
        return output

    def attach(self, layer_module: torch.nn.Module) -> None:
        """Register forward hook on a layer module."""
        self._handle = layer_module.register_forward_hook(self.hook_fn)

    def detach(self) -> None:
        """Remove the forward hook."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def orthogonalize_vectors(vectors: list[torch.Tensor]) -> list[torch.Tensor]:
    """Gram-Schmidt orthogonalization on a list of unit vectors.

    Returns orthogonalized unit vectors. Skips near-zero vectors that
    collapse during projection.
    """
    result: list[torch.Tensor] = []
    for v in vectors:
        u = v.clone()
        for basis in result:
            u = u - torch.dot(u, basis) * basis
        norm = u.norm()
        if norm > 1e-8:
            result.append(u / norm)
    return result


class SteeringManager:
    """Manages multiple SteeringHooks across model layers."""

    def __init__(self) -> None:
        self.hooks: dict[int, SteeringHook] = {}
        self.vectors: list[dict] = []
        self._name_idx: dict[str, int] = {}

    def _rebuild_index(self) -> None:
        self._name_idx = {v["name"]: i for i, v in enumerate(self.vectors)}

    def _find(self, name: str) -> dict | None:
        idx = self._name_idx.get(name)
        return self.vectors[idx] if idx is not None else None

    def add_vector(
        self,
        name: str,
        profile: dict[int, tuple[torch.Tensor, float]],
        alpha: float,
    ) -> None:
        self._name_idx[name] = len(self.vectors)
        self.vectors.append(
            {
                "name": name,
                "profile": profile,
                "alpha": alpha,
                "enabled": True,
            }
        )

    def remove_vector(self, name: str) -> None:
        self.vectors = [v for v in self.vectors if v["name"] != name]
        self._rebuild_index()

    def set_alpha(self, name: str, alpha: float) -> None:
        v = self._find(name)
        if v is not None:
            v["alpha"] = alpha

    def toggle_vector(self, name: str) -> None:
        v = self._find(name)
        if v is not None:
            v["enabled"] = not v["enabled"]

    def apply_to_model(
        self,
        model_layers: torch.nn.ModuleList,
        device: torch.device,
        dtype: torch.dtype,
        orthogonalize: bool = False,
    ) -> None:
        """Group enabled vectors by layer, recompose hooks, attach to model."""
        # Group enabled vectors by layer via their profiles
        by_layer: dict[int, list[tuple[torch.Tensor, float]]] = {}
        for v in self.vectors:
            if v["enabled"]:
                for layer_idx, (vec, score) in v["profile"].items():
                    effective_alpha = v["alpha"] * score
                    by_layer.setdefault(layer_idx, []).append((vec, effective_alpha))

        # Detach hooks for layers that no longer have vectors
        for idx in list(self.hooks):
            if idx not in by_layer:
                self.hooks[idx].detach()
                del self.hooks[idx]

        # Recompose and attach for each active layer
        for idx, pairs in by_layer.items():
            if orthogonalize and len(pairs) > 1:
                raw_vectors = [vec for vec, _ in pairs]
                alphas = [alpha for _, alpha in pairs]
                raw_vectors = orthogonalize_vectors(raw_vectors)
                alphas = alphas[: len(raw_vectors)]
                pairs = list(zip(raw_vectors, alphas))

            if idx not in self.hooks:
                self.hooks[idx] = SteeringHook()

            hook = self.hooks[idx]
            hook.recompose(pairs, device, dtype)
            # Re-attach (detach first to avoid duplicate hooks)
            hook.detach()
            hook.attach(model_layers[idx])

    def clear_all(self) -> None:
        """Detach all hooks and clear vectors."""
        for hook in self.hooks.values():
            hook.detach()
        self.hooks.clear()
        self.vectors.clear()
        self._name_idx.clear()

    def get_active_vectors(self) -> list[dict]:
        """Return all vector configs (for TUI display)."""
        return list(self.vectors)
