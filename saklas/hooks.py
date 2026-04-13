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


_DEGEN_THRESHOLD = 1e-8

# Reference mean PCA score used to anchor per-profile score normalization.
# Chosen to match the typical mean score of well-concentrated profiles
# (gemma-3-4b-it, Qwen-3.5-4B, Ministral-3-8B) so their recommended alphas
# carry over unchanged.  Diffuse-geometry models like Llama-3.2-3B-Instruct
# score much lower (~0.07); normalization divides by their profile mean and
# re-multiplies by this constant, bringing them onto the same alpha scale.
_REF_SCORE = 1.0 / 32.0


def orthogonalize_vectors(
    vectors: list[torch.Tensor],
    alphas: list[float] | None = None,
) -> list[torch.Tensor] | list[tuple[torch.Tensor, float]]:
    """QR-based orthogonalization on a list of vectors.

    Returns orthonormal vectors preserving each input's orientation.
    Drops degenerate directions whose R-diagonal magnitude falls below
    threshold. Single-kernel QR replaces sequential Gram-Schmidt.
    """
    if not vectors:
        return []
    if len(vectors) == 1:
        norm = vectors[0].norm()
        if norm <= _DEGEN_THRESHOLD:
            return []
        v = vectors[0] / norm
        if alphas is not None:
            return [(v, alphas[0])]
        return [v]
    orig_dtype = vectors[0].dtype
    stacked = torch.stack(vectors).float()           # (N, dim)
    Q, R = torch.linalg.qr(stacked.T)                # Q: (dim, N)
    diag = R.diag()                                   # (N,)
    keep = diag.abs() > _DEGEN_THRESHOLD
    # QR can flip column signs relative to input; R's diagonal sign
    # encodes the flip.  Correct so alphas keep their meaning.
    signs = diag.sign()
    if alphas is not None:
        return [
            ((Q[:, i] * signs[i]).to(orig_dtype), alphas[i])
            for i in range(len(vectors))
            if keep[i]
        ]
    return [
        (Q[:, i] * signs[i]).to(orig_dtype)
        for i in range(len(vectors))
        if keep[i]
    ]


class SteeringManager:
    """Manages multiple SteeringHooks across model layers."""

    def __init__(self) -> None:
        self.hooks: dict[int, SteeringHook] = {}
        self.vectors: dict[str, dict] = {}

    def add_vector(
        self,
        name: str,
        profile: dict[int, tuple[torch.Tensor, float]],
        alpha: float,
    ) -> None:
        self.vectors[name] = {
            "profile": profile,
            "alpha": alpha,
        }

    def apply_to_model(
        self,
        model_layers: torch.nn.ModuleList,
        device: torch.device,
        dtype: torch.dtype,
        orthogonalize: bool = False,
    ) -> None:
        """Group vectors by layer, recompose hooks, attach to model."""
        by_layer: dict[int, list[tuple[torch.Tensor, float]]] = {}
        for v in self.vectors.values():
            profile = v["profile"]
            # Mean-normalize per-profile so alpha means the same thing
            # across models.  Raw PCA scores (explained-variance-ratio)
            # vary wildly between architectures — diffuse small models
            # like Llama-3.2-3B-Instruct score ~0.07 while gemma-3-4b
            # scores ~0.25 for the same statements, turning `alpha*score`
            # into a model-dependent gate.  Dividing by the profile mean
            # and re-scaling to _REF_SCORE preserves relative per-layer
            # emphasis (high-signal layers still get more) while keeping
            # existing recommended alphas usable on well-behaved models.
            scores = [score for _, score in profile.values()]
            mean_score = sum(scores) / len(scores) if scores else 1.0
            if mean_score <= 0.0:
                mean_score = 1.0
            scale = _REF_SCORE / mean_score
            for layer_idx, (vec, score) in profile.items():
                effective_alpha = v["alpha"] * score * scale
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
                raw_alphas = [alpha for _, alpha in pairs]
                pairs = orthogonalize_vectors(raw_vectors, raw_alphas)

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

