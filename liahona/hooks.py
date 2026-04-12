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
        stacked = torch.stack(
            [vec.to(device=device, dtype=dtype) for vec, _ in vectors]
        )
        alphas = torch.tensor([alpha for _, alpha in vectors], device=device, dtype=dtype)
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
    """QR-based orthogonalization on a list of vectors.

    Returns orthonormal vectors preserving each input's orientation.
    Drops degenerate directions whose R-diagonal magnitude falls below
    threshold. Single-kernel QR replaces sequential Gram-Schmidt.
    """
    if not vectors:
        return []
    if len(vectors) == 1:
        norm = vectors[0].norm()
        return [vectors[0] / norm] if norm > 1e-8 else []
    orig_dtype = vectors[0].dtype
    stacked = torch.stack(vectors).float()           # (N, dim)
    Q, R = torch.linalg.qr(stacked.T)                # Q: (dim, N)
    diag = R.diag()                                   # (N,)
    keep = diag.abs() > 1e-8
    # QR can flip column signs relative to input; R's diagonal sign
    # encodes the flip.  Correct so alphas keep their meaning.
    signs = diag.sign()
    return [
        (Q[:, i] * signs[i]).to(orig_dtype)
        for i in range(min(Q.shape[1], len(vectors)))
        if keep[i]
    ]


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

        # Layer-scalar compensation: compute reverse cumulative product of
        # subsequent scalars so we know the attenuation each layer's
        # perturbation will suffer before reaching the LM head.
        n_layers = len(model_layers)
        layer_scalar_comp: dict[int, float] | None = None
        if by_layer and getattr(model_layers[0], "layer_scalar", None) is not None:
            # subsequent_prod[i] = product of layer_scalar at layers i+1..n-1
            subsequent_prod = [1.0] * (n_layers + 1)
            for i in range(n_layers - 1, -1, -1):
                s_buf = getattr(model_layers[i], "layer_scalar", None)
                subsequent_prod[i] = subsequent_prod[i + 1] * (s_buf.item() if s_buf is not None else 1.0)
            # Compensation = 1 / subsequent_prod[idx+1], capped.
            # Last layer: subsequent_prod[n] = 1.0 → comp = 1.0 (no boost).
            _MAX_SCALAR_COMP = 4.0
            layer_scalar_comp = {}
            for idx in by_layer:
                atten = subsequent_prod[idx + 1] if idx + 1 <= n_layers else 1.0
                if atten < 1.0 and atten > 0:
                    layer_scalar_comp[idx] = min(1.0 / atten, _MAX_SCALAR_COMP)

        # Recompose and attach for each active layer
        for idx, pairs in by_layer.items():
            if orthogonalize and len(pairs) > 1:
                raw_vectors = [vec for vec, _ in pairs]
                alphas = [alpha for _, alpha in pairs]
                raw_vectors = orthogonalize_vectors(raw_vectors)
                alphas = alphas[: len(raw_vectors)]
                pairs = list(zip(raw_vectors, alphas))

            # Compensate for per-layer output scaling (Gemma 4 layer_scalar).
            # A perturbation at layer N is attenuated by the product of all
            # subsequent layers' scalars before reaching the LM head.  Boost
            # by the inverse of that product (capped) so steering propagates
            # at the intended strength.  The last layer gets comp=1.0 since
            # its perturbation feeds directly into the LM head.
            if layer_scalar_comp and idx in layer_scalar_comp:
                comp = layer_scalar_comp[idx]
                pairs = [(vec, alpha * comp) for vec, alpha in pairs]

            if idx not in self.hooks:
                hook = SteeringHook()
                hook.attach(model_layers[idx])
                self.hooks[idx] = hook

            self.hooks[idx].recompose(pairs, device, dtype)

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
