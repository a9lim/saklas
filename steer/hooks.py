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
        stacked = torch.stack([vec.to(device=device, dtype=dtype) for vec, _ in vectors])
        alphas = torch.tensor([alpha for _, alpha in vectors], device=device, dtype=dtype)
        self.composed = (alphas.unsqueeze(1) * stacked).sum(dim=0)

    def hook_fn(self, module, input, output):
        if self.composed is not None:
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
            u = u - torch.dot(u.flatten(), basis.flatten()) * basis
        norm = u.norm()
        if norm > 1e-8:
            result.append(u / norm)
    return result


def pairwise_cosine_similarity(
    vectors: list[torch.Tensor],
) -> list[tuple[int, int, float]]:
    """Pairwise cosine similarity for all vector pairs.

    Returns list of (i, j, cosine_sim) tuples. Useful for showing
    interference between active steering vectors.
    """
    pairs: list[tuple[int, int, float]] = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            a = vectors[i].flatten()
            b = vectors[j].flatten()
            cos = torch.dot(a, b) / (a.norm() * b.norm() + 1e-8)
            pairs.append((i, j, cos.item()))
    return pairs


class SteeringManager:
    """Manages multiple SteeringHooks across model layers."""

    def __init__(self) -> None:
        self.hooks: dict[int, SteeringHook] = {}
        self.vectors: list[dict] = []

    def add_vector(
        self,
        name: str,
        vector: torch.Tensor,
        alpha: float,
        layer_idx: int,
    ) -> None:
        self.vectors.append(
            {
                "name": name,
                "vector": vector,
                "alpha": alpha,
                "layer_idx": layer_idx,
                "enabled": True,
            }
        )

    def remove_vector(self, name: str) -> None:
        self.vectors = [v for v in self.vectors if v["name"] != name]

    def set_alpha(self, name: str, alpha: float) -> None:
        for v in self.vectors:
            if v["name"] == name:
                v["alpha"] = alpha
                return

    def set_layer(
        self,
        name: str,
        layer_idx: int,
        model_layers: torch.nn.ModuleList,
    ) -> None:
        for v in self.vectors:
            if v["name"] == name:
                old_idx = v["layer_idx"]
                v["layer_idx"] = layer_idx
                if old_idx != layer_idx and old_idx in self.hooks:
                    self.hooks[old_idx].detach()
                    del self.hooks[old_idx]
                return

    def toggle_vector(self, name: str) -> None:
        for v in self.vectors:
            if v["name"] == name:
                v["enabled"] = not v["enabled"]
                return

    def apply_to_model(
        self,
        model_layers: torch.nn.ModuleList,
        device: torch.device,
        dtype: torch.dtype,
        orthogonalize: bool = False,
    ) -> None:
        """Group enabled vectors by layer, recompose hooks, attach to model."""
        # Group enabled vectors by layer
        by_layer: dict[int, list[dict]] = {}
        for v in self.vectors:
            if v["enabled"]:
                by_layer.setdefault(v["layer_idx"], []).append(v)

        # Detach hooks for layers that no longer have vectors
        for idx in list(self.hooks):
            if idx not in by_layer:
                self.hooks[idx].detach()
                del self.hooks[idx]

        # Recompose and attach for each active layer
        for idx, vecs in by_layer.items():
            raw_vectors = [v["vector"] for v in vecs]
            alphas = [v["alpha"] for v in vecs]

            if orthogonalize and len(raw_vectors) > 1:
                raw_vectors = orthogonalize_vectors(raw_vectors)
                # Pad alphas if orthogonalization dropped vectors
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

    def get_active_vectors(self) -> list[dict]:
        """Return all vector configs (for TUI display)."""
        return list(self.vectors)
