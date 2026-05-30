"""Mahalanobis whitening: cosine similarity and LEACE-style projection.

Two extraction-time / analysis-time primitives keyed off the per-model
activation covariance:

* **Mahalanobis cosine** — cosine similarity in the whitened metric
  ``<u, v>_M = u^T Σ^{-1} v``.  Equivalent to plain cosine after applying
  the whitening transform ``W = Σ^{-1/2}``.  Predicts cross-domain probe
  generalization much better than plain cosine when activations have
  strongly anisotropic covariance — the metric automatically downweights
  alignment along high-variance directions where "alignment" is mostly
  shared base-rate variation rather than concept overlap.

* **LEACE projection** — the closed-form projector from Belrose et al.
  (NeurIPS 2023, arXiv 2306.03819) for a single concept direction::

        P_LEACE x = x - (x^T Σ^{-1} d / d^T Σ^{-1} d) · d

  Removes all linearly-decodable concept information along ``d`` from
  ``x`` with minimum collateral damage in the covariance norm.  Reduces
  to plain Gram-Schmidt projection when ``Σ = I``.

Storage discipline: **no new persistent cache.**  :class:`LayerWhitener`
is built from the existing ``layer_means.safetensors`` and
``neutral_activations.safetensors`` caches under ``~/.saklas/models/<id>/``.
The 90 neutral statements give ``X ∈ ℝ^(N=90, D)`` per layer; ``Σ`` is
rank-deficient (rank ≤ N-1 = 89), so we ridge-regularize with
``λ_L = (||X_L||_F² / (N · D)) · ridge_scale`` (mean diagonal of the
sample covariance — a standard shrinkage target for the high-D, small-N
regime).  The full ``D × D`` inverse is never materialized; we apply
``Σ^{-1} v`` via the Woodbury identity::

        Σ_reg^{-1} v = (1/λ) [v - X^T K (X v)]
        K = (NλI_N + X X^T)^{-1}     # small N×N inverse, cached

Cost per ``Σ^{-1} v``: ``O(N D)`` (one ``X v`` matvec, one ``X^T (Kx)``
matvec).  Storage cost: ``X`` already on disk; ``K`` is ``N × N`` (≈32 KB
per layer at N=90), held in memory only.

Intended consumers:

* :meth:`saklas.core.profile.Profile.cosine_similarity` — pass a
  :class:`LayerWhitener` to switch to the Mahalanobis metric.
* :func:`saklas.core.vectors.project_profile` — pass a whitener to
  switch ``|`` / ``~`` to LEACE-flavored projection.

Both call sites accept ``whitener=None`` (default) for back-compat
Euclidean math.
"""

from __future__ import annotations

import logging
import math
from typing import Iterable, Mapping

import torch

from saklas.core.errors import SaklasError

log = logging.getLogger(__name__)


class WhitenerError(ValueError, SaklasError):
    """Raised when whitener construction or lookup fails."""

    def user_message(self) -> tuple[int, str]:
        return (400, str(self) or self.__class__.__name__)


# Numerical guard for Mahalanobis denominators.  Mirrors the ``1e-12``
# guard used by ``project_profile`` and ``Profile.projected_away`` so
# behavior at the near-zero edge stays consistent across metrics.
_NEAR_ZERO_DENOM = 1e-12


# Default ridge multiplier.  ``ridge_scale=1.0`` sets λ to the mean
# diagonal of the un-regularized sample covariance — neither aggressive
# (would mask real anisotropy) nor sample-noisy (small λ over a rank-89
# Σ extrapolates badly along null-space directions).  Tunable via
# :meth:`LayerWhitener.from_neutral_activations`.
DEFAULT_RIDGE_SCALE = 1.0


class LayerWhitener:
    """Per-layer Mahalanobis primitives over centered neutral activations.

    Holds the centered observations ``X_L ∈ ℝ^(N, D_L)`` and the small
    Woodbury inverses ``K_L = (NλI + X X^T)^{-1} ∈ ℝ^(N, N)`` per layer.
    All computations route through :meth:`apply_inv`, which returns
    ``Σ_reg^{-1} v`` without materializing ``D × D`` inverses.

    Layers without neutral activations are simply absent — callers that
    iterate via ``in`` / :attr:`layers` see only the covered set.
    """

    __slots__ = ("_X", "_K", "_lambda", "_dim")

    def __init__(
        self,
        centered: Mapping[int, torch.Tensor],
        small_inv: Mapping[int, torch.Tensor],
        ridge: Mapping[int, float],
    ) -> None:
        if not centered:
            raise WhitenerError("LayerWhitener requires at least one layer")
        if set(centered) != set(small_inv) or set(centered) != set(ridge):
            raise WhitenerError(
                "centered / small_inv / ridge must cover the same layer set"
            )
        self._X: dict[int, torch.Tensor] = dict(centered)
        self._K: dict[int, torch.Tensor] = dict(small_inv)
        self._lambda: dict[int, float] = dict(ridge)
        # Cache hidden-dim per layer for input-shape validation.  Allowing
        # mismatched dims would silently corrupt the matvec.
        self._dim: dict[int, int] = {
            L: int(t.shape[-1]) for L, t in self._X.items()
        }

    # ---- construction --------------------------------------------------

    @classmethod
    def from_neutral_activations(
        cls,
        neutral_activations: Mapping[int, torch.Tensor],
        layer_means: Mapping[int, torch.Tensor],
        *,
        ridge_scale: float = DEFAULT_RIDGE_SCALE,
    ) -> "LayerWhitener":
        """Build a whitener from raw per-layer neutral activations.

        ``neutral_activations`` is ``{layer: [N, D] fp32 tensor}`` (the
        shape produced by :func:`saklas.core.vectors.compute_neutral_activations`).
        ``layer_means`` is ``{layer: [D]}`` — typically the cached means
        from :func:`saklas.io.probes_bootstrap.bootstrap_layer_means`.

        Layers present in ``neutral_activations`` but not in
        ``layer_means`` are skipped (centering would be undefined).
        Subtracts the mean per layer, computes the regularized covariance
        on-the-fly, and inverts the small ``N × N`` Woodbury kernel
        ``NλI + X X^T``.  Everything stays on CPU in fp32; this is one-
        shot setup, not a hot-path operation.
        """
        if ridge_scale <= 0:
            raise WhitenerError(
                f"ridge_scale must be positive, got {ridge_scale!r}"
            )
        centered: dict[int, torch.Tensor] = {}
        small_inv: dict[int, torch.Tensor] = {}
        ridge: dict[int, float] = {}
        for L, X_raw in neutral_activations.items():
            if L not in layer_means:
                continue
            mu = layer_means[L].to(dtype=torch.float32, device="cpu").reshape(-1)
            X = X_raw.to(dtype=torch.float32, device="cpu")
            if X.dim() != 2 or X.shape[1] != mu.shape[0]:
                raise WhitenerError(
                    f"layer {L}: neutral activations shape {tuple(X.shape)} "
                    f"does not match mean dim {mu.shape[0]}"
                )
            X_c = X - mu  # (N, D)
            n, d = X_c.shape
            # λ_L = mean diagonal of the un-regularized sample covariance
            # Σ̂ = X^T X / N.  trace(Σ̂) = ||X||_F² / N, divided by D → mean
            # eigenvalue of Σ̂ averaged over the standard basis.  Stable
            # for any ``ridge_scale > 0``; degenerates only when ``X`` is
            # exactly zero (covered by the all-zero guard below).
            frob_sq = float(torch.sum(X_c * X_c).item())
            if frob_sq <= _NEAR_ZERO_DENOM:
                # Activations are degenerate at this layer (e.g. a layer
                # the model zeros out).  Fall back to λ=1.0 — Mahalanobis
                # collapses to plain Euclidean for this layer, which is
                # the only sensible behavior.
                lam = 1.0
            else:
                lam = (frob_sq / (n * d)) * ridge_scale
            # K = (NλI + X X^T)^{-1}, fp32, on CPU.  The ``(NλI + X X^T)``
            # matrix is symmetric PD by construction (Gram matrix plus a
            # positive multiple of identity) — Cholesky is the right
            # solver, but ``torch.linalg.inv`` is fine at N=90 and avoids
            # threading a separate Cholesky through every call site.
            G = X_c @ X_c.transpose(0, 1)  # (N, N) Gram
            G.diagonal().add_(n * lam)
            K = torch.linalg.inv(G)
            centered[L] = X_c
            small_inv[L] = K
            ridge[L] = lam
        if not centered:
            raise WhitenerError(
                "no layers shared between neutral_activations and layer_means"
            )
        return cls(centered, small_inv, ridge)

    @classmethod
    def from_cache(
        cls,
        model_id: str,
        *,
        ridge_scale: float = DEFAULT_RIDGE_SCALE,
    ) -> "LayerWhitener":
        """Load a whitener from disk caches alone (no model load).

        Reads ``layer_means.safetensors`` and ``neutral_activations.safetensors``
        from ``~/.saklas/models/<safe_id>/``.  Raises :class:`WhitenerError`
        with a populating-command hint when either cache is missing —
        ``vector compare`` and similar offline tools shouldn't silently
        load a model.

        Use :meth:`from_neutral_activations` instead when both tensors
        are already in memory (e.g. inside a live ``SaklasSession``).
        """
        from safetensors.torch import load_file
        from saklas.io.paths import model_dir

        md = model_dir(model_id)
        means_path = md / "layer_means.safetensors"
        acts_path = md / "neutral_activations.safetensors"
        missing: list[str] = []
        if not means_path.is_file():
            missing.append(means_path.name)
        if not acts_path.is_file():
            missing.append(acts_path.name)
        if missing:
            raise WhitenerError(
                f"whitener cache missing for {model_id} "
                f"(expected {', '.join(missing)} under {md}); "
                f"populate via any flow that loads the model + neutral "
                f"activations (e.g. `saklas vector transfer --to {model_id}` "
                f"once, or run any session-level extract on this model)"
            )
        means_raw = load_file(str(means_path))
        acts_raw = load_file(str(acts_path))
        # Both files key tensors by ``layer_<idx>`` (alignment.py and
        # vectors.save_profile shape).  ``layer_means`` are stored fp32;
        # ``neutral_activations`` are stored fp16 to halve disk usage and
        # promoted to fp32 here because the small N×N inverse doesn't
        # tolerate fp16 condition number.
        means = {
            int(k.split("_", 1)[1]): v.to(dtype=torch.float32)
            for k, v in means_raw.items()
        }
        acts = {
            int(k.split("_", 1)[1]): v.to(dtype=torch.float32)
            for k, v in acts_raw.items()
        }
        return cls.from_neutral_activations(
            acts, means, ridge_scale=ridge_scale,
        )

    # ---- introspection -------------------------------------------------

    @property
    def layers(self) -> set[int]:
        """Layer indices the whitener covers."""
        return set(self._X)

    def covers(self, layer: int) -> bool:
        return layer in self._X

    def covers_all(self, layers: "Iterable[int]") -> bool:
        """True iff every layer in ``layers`` is covered.

        Backs the *all-or-nothing* metric gate shared by vector extraction
        and manifold fitting: the per-layer Mahalanobis and Euclidean
        scores live on different scales (``‖·‖_M`` carries a ``1/√λ_L``
        factor that ``‖·‖_2`` doesn't), so mixing metrics across the layers
        of one cross-layer-normalized share weight would compare
        incommensurable magnitudes.  Callers fall back to all-Euclidean
        when this returns ``False`` rather than whitening only the covered
        subset.  Empty ``layers`` → ``True`` (vacuous).
        """
        return all(layer in self._X for layer in layers)

    def ridge(self, layer: int) -> float:
        """Per-layer regularization parameter ``λ_L``."""
        if layer not in self._lambda:
            raise WhitenerError(f"whitener does not cover layer {layer}")
        return self._lambda[layer]

    def __contains__(self, layer: object) -> bool:
        return layer in self._X

    def __repr__(self) -> str:
        n_layers = len(self._X)
        if n_layers == 0:
            return "LayerWhitener(empty)"
        first = next(iter(self._X.values()))
        return (
            f"LayerWhitener(layers={n_layers}, "
            f"N={first.shape[0]}, dtype=float32)"
        )

    # ---- core math -----------------------------------------------------

    def apply_inv(self, layer: int, v: torch.Tensor) -> torch.Tensor:
        """Return ``Σ_L^{-1} v`` via Woodbury (no ``D × D`` inverse).

        Math::

            Σ_reg^{-1} v = (1/λ) [v - X^T K (X v)]
            K = (NλI + X X^T)^{-1}

        ``v`` is promoted to fp32 on CPU regardless of input dtype/device
        — ``LayerWhitener`` is an analysis-time tool, not a hot path.
        Output matches ``v``'s original dtype.
        """
        if layer not in self._X:
            raise WhitenerError(f"whitener does not cover layer {layer}")
        v_in_dtype = v.dtype
        v32 = v.to(dtype=torch.float32, device="cpu").reshape(-1)
        if v32.shape[0] != self._dim[layer]:
            raise WhitenerError(
                f"layer {layer}: input dim {v32.shape[0]} does not match "
                f"whitener dim {self._dim[layer]}"
            )
        X = self._X[layer]
        K = self._K[layer]
        lam = self._lambda[layer]
        # X v  ∈ ℝ^N ; K (X v)  ∈ ℝ^N ; X^T (K X v)  ∈ ℝ^D
        Xv = X @ v32  # (N,)
        KXv = K @ Xv  # (N,)
        out = (v32 - X.transpose(0, 1) @ KXv) / lam
        return out.to(dtype=v_in_dtype)

    def subspace_gram(self, layer: int, basis: torch.Tensor) -> torch.Tensor:
        """Return the reduced-space inverse-covariance Gram ``B Σ_L^{-1} Bᵀ``.

        ``basis`` is ``(R, D)`` with rows spanning a subspace of the
        layer's activation space — typically a manifold's per-layer PCA
        frame.  The returned ``(R, R)`` matrix is the inverse covariance
        *restricted* to that subspace's reduced coordinates: for a reduced
        vector ``c ∈ ℝ^R`` whose model-space realization is ``Bᵀ c``,

            cᵀ (B Σ^{-1} Bᵀ) c = ‖Bᵀ c‖²_M = (Bᵀ c)ᵀ Σ^{-1} (Bᵀ c).

        Manifold steering uses this to whiten its per-layer share weight —
        the subspace-restricted analogue of the vector path's ``‖d‖_M``
        bake score.  When ``Σ = I`` and ``B`` is orthonormal it reduces to
        ``B Bᵀ = I_R``, so the whitened share collapses to the Euclidean
        spread (``‖coords‖_F``) the unwhitened path computes — the natural
        sanity check.

        Computed via the same Woodbury identity as :meth:`apply_inv`,
        vectorized over the ``R`` basis rows (one ``X Bᵀ`` matvec, one
        ``K``-apply, one ``Xᵀ`` matvec) so the cost is ``O(N D R)`` with no
        ``D × D`` inverse::

            Σ_reg^{-1} Bᵀ = (1/λ)[Bᵀ − Xᵀ K (X Bᵀ)]

        Promoted to fp32 on CPU; this is fit-time setup, not a hot path.
        """
        if layer not in self._X:
            raise WhitenerError(f"whitener does not cover layer {layer}")
        B = basis.to(dtype=torch.float32, device="cpu")
        if B.dim() != 2 or B.shape[1] != self._dim[layer]:
            raise WhitenerError(
                f"layer {layer}: basis shape {tuple(B.shape)} does not match "
                f"whitener dim {self._dim[layer]}"
            )
        X = self._X[layer]      # (N, D)
        K = self._K[layer]      # (N, N)
        lam = self._lambda[layer]
        Bt = B.transpose(0, 1)          # (D, R)
        XBt = X @ Bt                    # (N, R)
        KXBt = K @ XBt                  # (N, R)
        inv_Bt = (Bt - X.transpose(0, 1) @ KXBt) / lam  # (D, R) = Σ^{-1} Bᵀ
        gram = B @ inv_Bt               # (R, R)
        # The true Gram is symmetric PSD; symmetrize away finite-precision
        # asymmetry so downstream quadratic forms can't pick up drift.
        return 0.5 * (gram + gram.transpose(0, 1))

    def mahalanobis_dot(
        self, layer: int, u: torch.Tensor, v: torch.Tensor,
    ) -> float:
        """Return the scalar ``u^T Σ_L^{-1} v``."""
        u32 = u.to(dtype=torch.float32, device="cpu").reshape(-1)
        if u32.shape[0] != self._dim[layer]:
            raise WhitenerError(
                f"layer {layer}: input dim {u32.shape[0]} does not match "
                f"whitener dim {self._dim[layer]}"
            )
        return float(torch.dot(u32, self.apply_inv(layer, v).float()).item())

    def mahalanobis_norm(self, layer: int, v: torch.Tensor) -> float:
        """Return ``sqrt(v^T Σ_L^{-1} v)``."""
        sq = self.mahalanobis_dot(layer, v, v)
        # Tiny negative drift from finite-precision can leak in for
        # near-zero vectors; clamp before sqrt.
        return math.sqrt(max(sq, 0.0))

    def mahalanobis_cosine(
        self, layer: int, u: torch.Tensor, v: torch.Tensor,
    ) -> float:
        """Whitened cosine ``<u, v>_M / (||u||_M · ||v||_M)``.

        Returns ``0.0`` when either argument has near-zero Mahalanobis
        norm — same convention as plain cosine on near-zero vectors.
        """
        nu = self.mahalanobis_norm(layer, u)
        nv = self.mahalanobis_norm(layer, v)
        if nu < _NEAR_ZERO_DENOM or nv < _NEAR_ZERO_DENOM:
            return 0.0
        return self.mahalanobis_dot(layer, u, v) / (nu * nv)

    def leace_project(
        self,
        layer: int,
        base: torch.Tensor,
        onto: torch.Tensor,
        operator: str,
    ) -> torch.Tensor:
        """LEACE-style projection of ``base`` against ``onto``.

        For each shared layer (fp32 internal)::

            coef = <base, onto>_M / <onto, onto>_M
            proj = coef * onto

        - ``operator == "~"``  →  ``proj``  (Mahalanobis-aligned component).
        - ``operator == "|"``  →  ``base - proj``  (LEACE erasure of ``onto``).

        Reduces to plain Gram-Schmidt projection when ``Σ = I`` — useful
        sanity check.  Removes *linearly-decodable* information along
        ``onto`` rather than raw Euclidean overlap; for a single direction
        that's exactly the LEACE projector from Belrose et al. (2023).

        Output dtype matches ``base``.
        """
        if operator not in ("~", "|"):
            raise ValueError(f"unknown projection operator: {operator!r}")
        if layer not in self._X:
            raise WhitenerError(f"whitener does not cover layer {layer}")
        base_dtype = base.dtype
        base32 = base.to(dtype=torch.float32, device="cpu").reshape(-1)
        onto32 = onto.to(dtype=torch.float32, device="cpu").reshape(-1)
        m_oo = self.mahalanobis_dot(layer, onto32, onto32)
        if m_oo < _NEAR_ZERO_DENOM:
            # ``onto`` has effectively zero Mahalanobis norm — no
            # projection is well-defined.  Mirror ``project_profile``'s
            # near-zero behavior: ``|`` passes ``base`` through, ``~``
            # produces zero (caller decides whether to drop the layer).
            if operator == "|":
                return base32.to(dtype=base_dtype)
            return torch.zeros_like(base32).to(dtype=base_dtype)
        m_bo = self.mahalanobis_dot(layer, base32, onto32)
        coef = m_bo / m_oo
        proj = coef * onto32
        if operator == "~":
            return proj.to(dtype=base_dtype)
        return (base32 - proj).to(dtype=base_dtype)


__all__ = [
    "DEFAULT_RIDGE_SCALE",
    "LayerWhitener",
    "WhitenerError",
]
