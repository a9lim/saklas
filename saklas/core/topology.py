"""Coordinate discovery and topology selection for discover-mode manifolds.

A discover-mode manifold folder supplies labeled node corpora and nothing
else: the coordinate system is *derived* per-model at fit time from the
node centroids.  This module owns that derivation and the geometry choice
that precedes it.

Two layers:

* **Coordinate derivation** — :func:`derive_pca_coords` (eigendecompose
  the consensus Gram, keep the smallest variance-crossing prefix) and
  :func:`derive_spectral_coords` (Laplacian eigenmaps over a symmetric
  k-NN graph, dimension picked at the eigenvalue-ratio cliff), dispatched
  by :func:`discover_coords`.  Both embed the same layer-agnostic
  consensus geometry; :func:`neutral_layout_coord` places the per-model
  neutral mean into a flat layout by landmark MDS so ``% 0,…,0`` reads as
  neutral.
* **Topology selection** (``fit_mode="auto"``) — :func:`select_topology`
  makes two deliberately decoupled decisions: flat-vs-curved by GCV in a
  shared whitened-reduced metric, and periodic axes by Vietoris-Rips H1
  persistent homology (:func:`_rips_h1_persistence` /
  :func:`_count_persistent_loops`) coordinated off the spectral
  eigenpairs, with :func:`_faint_cycle_coords` as the guarded
  single-cycle fallback for rings too thin for PH's hole-size threshold.

Everything here is pure tensor math on the ``(K, K)`` node scatter — no
model, no IO.  It depends on :mod:`saklas.core.manifold` for the domain
types and the per-layer fit primitives, never the other way round;
``extraction.py::_fit_locked`` is the one production consumer.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from saklas.core.manifold import (
    BoxAxis,
    BoxDomain,
    CustomDomain,
    ManifoldDomain,
    RbfFitPlan,
    _gcv_select_lambda,
    _pca_basis,
    _rbf_poised,
    _rbf_smoother_matrix,
    prepare_rbf_fit_plan,
)

if TYPE_CHECKING:
    from saklas.core.mahalanobis import LayerWhitener

log = logging.getLogger(__name__)


# ======================================================= coord discovery ===
#
# Derive node coordinates from per-node activation centroids when the user
# hasn't authored a coordinate system.  PCA is the safe linear default
# (reproduces the flat-subspace layout a user would author themselves once
# they knew the answer).  Spectral (Laplacian eigenmaps) recovers
# curved-manifold topology that PCA flattens.  Both feed the same downstream
# machinery: ``CustomDomain(k)`` with identity embedding, then per-layer
# ``manifold.fit_layer_subspace`` exactly as for authored manifolds.


@dataclass(frozen=True)
class PcaDiagnostics:
    """Diagnostics for a PCA coordinate derivation.

    Surfaced on the per-model sidecar so a user inspecting a fitted
    manifold can tell whether the chosen ``k`` was well-supported by
    the data (a sharp variance plateau) or compromised (variance still
    rising at the cap).
    """

    per_component_variance: torch.Tensor  # (max_dim,) normalized to sum to 1
    cumulative_variance: torch.Tensor     # (max_dim,)
    picked_k: int
    threshold: float


@dataclass(frozen=True)
class SpectralDiagnostics:
    """Diagnostics for a spectral (Laplacian-eigenmaps) coordinate derivation.

    The spectral gap is the one knob that says "the data has a clean
    k-dimensional structure" (large gap at ``picked_k``) versus "no
    clean cut, pick a dim by hand" (gaps flat across the candidate
    range).  Bandwidth + ``k_nn`` are recorded for reproducibility:
    both default to data-driven values (median k-NN distance,
    ``max(5, ceil(log K))``) when the user doesn't override.
    """

    eigenvalues: torch.Tensor  # (kept_count,) non-trivial spectrum, ascending
    picked_k: int
    # ``picked_k`` is chosen by the eigenvalue-*ratio* heuristic
    # (``argmax(λ_{k+1} / λ_k)`` over ``[1, max_dim]``), not the spec's
    # original absolute-gap form -- the absolute gap scales like k² on
    # S¹ (continuous-limit eigenvalues are quadratic in k) and pushes
    # the picker toward larger k.  ``gap_index`` is kept as an alias of
    # ``picked_k`` for diagnostic-render call sites that pre-date the
    # rename; ``gap_magnitude`` is still the *absolute* gap at
    # ``picked_k`` for the inspector's bar-chart annotation.
    gap_index: int             # == picked_k
    gap_magnitude: float       # eigenvalues[picked_k] - eigenvalues[picked_k - 1]
    bandwidth: float           # heat-kernel sigma actually used
    k_nn: int                  # number of nearest neighbors actually used
    component_count: int       # always 1 on success (disconnected graphs raise)
    # Authored-dimensionality floor.  ``heuristic_k`` is what the
    # eigenvalue-ratio cliff picked on its own; when ``min_dim`` is set and
    # exceeds it, ``picked_k`` is floored to ``min(min_dim, cap)`` and
    # ``pinned`` is True.  The cliff *undershoots* for a manifold whose
    # strongest mode dominates the spectrum (a small Fiedler value forces an
    # early ratio cliff), so the floor lets a known geometry (PAD's P×A×D)
    # survive a re-derivation that would otherwise collapse it.
    heuristic_k: int = 0       # ratio-cliff pick before the floor
    min_dim: int | None = None # author-declared floor (None = pure heuristic)
    pinned: bool = False       # True iff the floor raised picked_k


def derive_pca_coords(
    gram: torch.Tensor,
    *,
    max_dim: int = 8,
    var_threshold: float = 0.70,
) -> tuple[torch.Tensor, PcaDiagnostics]:
    """Derive node coordinates from the eigendecomposition of a centroid Gram.

    ``gram`` is the ``(K, K)`` symmetric-PSD Gram of the K node centroids.
    For a single layer it is ``X̃ X̃ᵀ`` (``X̃`` = node-mean-centered
    centroids), whose eigendecomposition is exactly the PCA of those
    centroids — eigenvalues are the component variances, eigenvectors
    scaled by ``√λ`` are the PCA scores ``U S``.  For the layer-agnostic
    fit it is the **signal-weighted consensus** Gram
    ``mean_L X̃_L Σ_L⁻¹ X̃_Lᵀ`` (whitened, averaged over every fit layer):
    the ``(K, K)`` Gram is the layer-invariant object, so averaging it is
    what lets the coordinate layout draw on all layers at once instead of
    one arbitrary reference layer.  A layer where the nodes aren't
    separated contributes a near-zero whitened Gram, so it drops out of
    the average on its own — no need to pick a layer band.

    Returns ``(coords, diagnostics)`` where ``coords`` is ``(K, k)`` and
    ``k`` is the smallest prefix whose cumulative variance crosses
    ``var_threshold``, capped at ``max_dim`` and floored at 1.

    Pure tensor, fp32, dependency-free.
    """
    gram = gram.to(torch.float32)
    K = gram.shape[0]
    if gram.dim() != 2 or gram.shape[1] != K:
        raise ValueError(
            f"PCA coord derivation needs a square (K, K) Gram, got shape "
            f"{tuple(gram.shape)}"
        )
    if K < 2:
        raise ValueError(
            f"PCA coord derivation needs >= 2 centroids, got {K}"
        )
    # Symmetrize away finite-precision drift, then eigendecompose.  eigh is
    # the Gram analogue of the old SVD-of-centered-centroids: eigenvalues =
    # S², eigenvectors = U (up to per-axis sign, immaterial for a layout).
    gram = 0.5 * (gram + gram.transpose(0, 1))
    evals, evecs = torch.linalg.eigh(gram)  # ascending
    # Descending order (PCA convention); clamp tiny-negative eigenvalues
    # that fp drift can leak past a genuinely PSD Gram.
    evals = evals.flip(0).clamp(min=0.0)
    evecs = evecs.flip(1)
    # Variance fractions are λ_i / Σ_i λ_i — metric-invariant and unaffected
    # by the (sum-vs-mean) scaling of the consensus average.
    total = evals.sum().clamp(min=1e-12)
    var_frac = evals / total                     # (K,)
    cum_var = torch.cumsum(var_frac, dim=0)      # (K,)
    cap = min(max_dim, var_frac.shape[0])
    # Smallest k such that cum_var[k-1] >= threshold; default to cap.
    over = (cum_var[:cap] >= var_threshold).nonzero(as_tuple=False)
    picked_k = int(over[0].item()) + 1 if over.numel() > 0 else cap
    picked_k = max(1, min(picked_k, cap))

    coords = evecs[:, :picked_k] * evals[:picked_k].sqrt()  # (K, picked_k) = U S
    diagnostics = PcaDiagnostics(
        per_component_variance=var_frac[:cap].detach().clone(),
        cumulative_variance=cum_var[:cap].detach().clone(),
        picked_k=picked_k,
        threshold=float(var_threshold),
    )
    return coords.contiguous(), diagnostics


def neutral_layout_coord(
    node_coords: torch.Tensor,
    neutral_cross_gram: torch.Tensor,
) -> torch.Tensor:
    """Project the neutral baseline into a consensus-PCA node layout.

    :func:`derive_pca_coords` returns ``node_coords`` ``(K, k)`` centered on the
    **node mean** (PCA removes it), so the layout origin is the node centroid,
    not the neutral baseline.  This is the classical-MDS / kernel-PCA
    out-of-sample extension that locates neutral in the *same* layout:

    - ``node_coords`` are the ``U S`` scores, so ``node_coords @ node_coordsᵀ``
      reproduces the rank-``k`` consensus Gram ``Ḡ`` the layout was built from.
    - ``neutral_cross_gram`` ``(K,)`` is neutral's matching cross-Gram column —
      its node-mean-centered, whitened inner product with each node centroid in
      the *same* layer-averaged metric ``Ḡ`` uses
      (``gᵢ = mean_L (ν_L − μ_L)ᵀ Σ_L⁻¹ (c_{L,i} − μ_L)``).

    The landmark coordinate is then ``cₙ = node_coords⁺ g`` (``⁺`` the
    pseudo-inverse: ``cₙ[r] = (1/√λ_r) Σᵢ Uᵢᵣ gᵢ``).  Subtracting it from
    ``node_coords`` re-anchors the layout so neutral sits at the origin — a pure
    translation that leaves the inter-node geometry (and, via the
    translation-invariant cardinal weights, every steering target) unchanged.
    Returns ``(k,)`` fp32.
    """
    nc = node_coords.to(torch.float32)
    g = neutral_cross_gram.to(torch.float32).reshape(-1)
    if g.shape[0] != nc.shape[0]:
        raise ValueError(
            f"neutral cross-Gram has {g.shape[0]} entries but the layout has "
            f"{nc.shape[0]} nodes"
        )
    return (torch.linalg.pinv(nc) @ g).contiguous()


def _knn_adjacency(
    distances: torch.Tensor, k_nn: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric k-NN graph from a pairwise distance matrix.

    Returns ``(mask, neighbor_dists)`` where ``mask`` is a ``(K, K)``
    bool tensor (True for retained edges, no self-loops) and
    ``neighbor_dists`` is the flat 1-D tensor of edge distances actually
    kept — useful for the median-bandwidth default.

    Edges are symmetrized by **union** (``i ↔ j`` if ``j`` is in
    ``knn(i)`` OR ``i`` is in ``knn(j)``).  Union is the standard
    convention for Laplacian eigenmaps; intersection drops too many
    edges and tends to disconnect borderline points.
    """
    K = distances.shape[0]
    # Include self-distance in the top-k+1 call, then strip the self
    # entry — guaranteed at position 0 because diag is zero.
    k_eff = min(k_nn + 1, K)
    _, idx = torch.topk(distances, k=k_eff, dim=1, largest=False)
    # Build directed mask: row i has 1 in column idx[i, j] for j>=1.
    directed = torch.zeros(K, K, dtype=torch.bool, device=distances.device)
    rows = torch.arange(K, device=distances.device).unsqueeze(1).expand(-1, k_eff)
    directed[rows, idx] = True
    directed.fill_diagonal_(False)
    # Symmetrize via union.
    mask = directed | directed.T
    neighbor_dists = distances[mask]
    return mask, neighbor_dists


class _DSU:
    """Disjoint-set forest with path halving over the node ids ``0..n-1``.

    The single union-find behind every graph reduction in the topology
    block: connected-component counting on a k-NN adjacency mask, the H0
    (edge) half of the Vietoris-Rips boundary reduction, and the Kruskal
    MST that fixes the persistence connectivity scale ``ε_c``.  Node counts
    are tens to hundreds, so path halving alone is ample — no rank/size
    balancing, matching what each hand-rolled copy did.
    """

    __slots__ = ("_parent",)

    def __init__(self, n: int) -> None:
        self._parent = list(range(n))

    def find(self, x: int) -> int:
        parent = self._parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        """Merge the sets holding ``a`` and ``b``; True iff they differed.

        The boolean is the H0 signal both persistence callers need: True
        means the edge was a tree edge (it killed a component), False means
        both endpoints were already joined (the edge births a 1-cycle).
        """
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self._parent[ra] = rb
        return True


def _connected_components(mask: torch.Tensor) -> int:
    """Number of connected components in an undirected adjacency mask.

    Plain union-find on a small ``(K, K)`` bool matrix — ``K`` is on
    the order of tens to hundreds, so a quadratic scan over the upper
    triangle is fine and avoids both scipy and the eigenvalue-counting
    tolerance choice.
    """
    K = mask.shape[0]
    dsu = _DSU(K)
    # Only need the upper triangle since the mask is symmetric.
    rows, cols = mask.triu(diagonal=1).nonzero(as_tuple=True)
    for r, c in zip(rows.tolist(), cols.tolist(), strict=True):
        dsu.union(r, c)
    return len({dsu.find(i) for i in range(K)})


def _laplacian_eigen(
    gram: torch.Tensor,
    *,
    k_nn: int | None = None,
    bandwidth: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int, float]:
    """Normalized-Laplacian eigenmaps core shared by every spectral topology.

    Builds the symmetric k-NN heat-kernel graph over the distances read off
    ``gram`` (``d²_ij = G_ii + G_jj − 2 G_ij``), forms the normalized
    Laplacian ``L = I − D^{-1/2} W D^{-1/2}``, eigendecomposes it, and drops
    the trivial (constant-mode) eigenpair.  Returns ``(nontrivial_vals
    (K-1,), nontrivial_vecs (K, K-1), k_nn, bandwidth)`` — the spectral
    embedding :func:`derive_spectral_coords` reads as flat coordinates and
    :func:`_detect_periodic_axes` reads as ``(cos, sin)`` angle pairs for the
    loops persistent homology has counted.

    The first ``n`` nontrivial eigenfunctions of data sampled from a manifold
    are its lowest Laplace–Beltrami modes — on a circle the ``(cos θ, sin θ)``
    pair — which is what lets the periodic detection read an angle coordinate
    straight off this embedding once a loop is confirmed.

    Raises on a < 4-node heap (no graph) or a disconnected k-NN graph (no
    single embedding) — the geometric preconditions every spectral topology
    shares.  ``derive_spectral_coords`` validates first with its own
    messages, so those callers never reach these.
    """
    gram = gram.to(torch.float32)
    K = gram.shape[0]
    if gram.dim() != 2 or gram.shape[1] != K:
        raise ValueError(
            f"spectral embedding needs a square (K, K) Gram, got shape "
            f"{tuple(gram.shape)}"
        )
    if K < 4:
        raise ValueError(
            f"spectral embedding needs >= 4 centroids to form a k-NN graph, "
            f"got {K}"
        )
    if k_nn is None:
        k_nn = max(5, math.ceil(math.log(K)))
    k_nn = max(1, min(k_nn, K - 1))

    # Pairwise distances from the Gram: d²_ij = G_ii + G_jj − 2 G_ij.  Clamp
    # the (PSD-up-to-fp-drift) squared distances off negative before sqrt.
    gram = 0.5 * (gram + gram.transpose(0, 1))
    diag = gram.diagonal()
    sq = diag.unsqueeze(0) + diag.unsqueeze(1) - 2.0 * gram
    distances = sq.clamp(min=0.0).sqrt()
    distances.fill_diagonal_(0.0)
    mask, neighbor_dists = _knn_adjacency(distances, k_nn)

    components = _connected_components(mask)
    if components > 1:
        raise ValueError(
            f"spectral embedding: k-NN graph has {components} connected "
            f"components (need 1). Raise k_nn or switch to PCA."
        )

    if bandwidth is None:
        if neighbor_dists.numel() == 0:
            raise ValueError("spectral embedding: k-NN graph has no edges")
        bandwidth = float(neighbor_dists.median().item())
        if bandwidth <= 0.0:
            # All-zero neighbor distances would NaN out the heat kernel.
            bandwidth = 1e-6
    bandwidth = float(bandwidth)

    # Heat-kernel weights on the symmetric k-NN edge set.
    W = torch.zeros_like(distances)
    sq = (distances * distances) / (2.0 * bandwidth * bandwidth)
    W = torch.where(mask, torch.exp(-sq), W)
    W.fill_diagonal_(0.0)

    deg = W.sum(dim=1)
    d_inv_sqrt = deg.clamp(min=1e-12).rsqrt()
    # L_sym = I - D^{-1/2} W D^{-1/2}
    L = -W * d_inv_sqrt.unsqueeze(0) * d_inv_sqrt.unsqueeze(1)
    L.fill_diagonal_(0.0)
    L = L + torch.eye(K, dtype=L.dtype, device=L.device)

    eigvals, eigvecs = torch.linalg.eigh(L)  # ascending
    # Drop the smallest eigenvalue (~0 for a connected graph); its eigenvector
    # is D^{1/2}1, carrying no embedding information.
    return eigvals[1:], eigvecs[:, 1:], int(k_nn), float(bandwidth)


def derive_spectral_coords(
    gram: torch.Tensor,
    *,
    max_dim: int = 8,
    min_dim: int | None = None,
    k_nn: int | None = None,
    bandwidth: float | None = None,
    _eigen_result: tuple[torch.Tensor, torch.Tensor, int, float] | None = None,
) -> tuple[torch.Tensor, SpectralDiagnostics]:
    """Derive node coordinates from a Laplacian-eigenmaps spectral embedding.

    ``gram`` is the ``(K, K)`` symmetric-PSD centroid Gram (see
    :func:`derive_pca_coords`); the pairwise distances the graph is built
    from are read straight off it, ``d²_ij = G_ii + G_jj − 2 G_ij``.  For a
    single layer this is the plain Euclidean distance ``‖c_i − c_j‖``; for
    the layer-agnostic fit, where ``gram`` is the layer-averaged whitened
    consensus, it is the mean per-layer **Mahalanobis** distance — which is
    exactly the squared distance in the concatenated-whitened space, the
    same geometry :func:`derive_pca_coords` embeds linearly.  (``mean_L`` of
    ``diag(G_L) ⊕ diag(G_L) − 2 G_L`` equals ``diag(Ḡ) ⊕ diag(Ḡ) − 2 Ḡ``
    because the diagonal and the Gram are both linear in the layer average,
    so the consensus distance is recoverable from the consensus Gram alone.)

    Build a symmetric k-NN graph over those distances,
    heat-kernel weights ``W_ij = exp(-d_ij^2 / (2 sigma^2))``, the
    normalized Laplacian ``L = I - D^{-1/2} W D^{-1/2}``, eigendecompose
    via :func:`torch.linalg.eigh`, drop the smallest (trivial) eigenvalue,
    and take the next ``k`` eigenvector entries as coordinates.

    ``k`` is chosen by the **eigenvalue-ratio** heuristic: the index
    that maximizes ``λ_{k+1} / λ_k`` for ``1 <= k <= max_dim``.  This
    captures the structural cliff between "signal" and "noise"
    eigenvalues robustly across topologies — on a circle the cos/sin
    pair at the lowest frequency produces a clean ratio cliff between
    ``λ_2`` and ``λ_3``, picking ``k=2``.  Absolute gaps
    ``λ_{k+1} - λ_k`` would over-pick on S^1 because the eigenvalues
    scale ~ ``k²`` in the continuous limit, so the largest absolute
    gap lands at high ``k`` rather than the structural cliff.

    ``min_dim`` floors the picked dimension: when an author declares the
    intrinsic dimensionality (e.g. PAD is P×A×D, 3-D by construction), the
    ratio cliff can *undershoot* if one mode dominates the spectrum — a
    small first non-trivial (Fiedler) eigenvalue makes ``λ_2 / λ_1`` the
    largest ratio, picking ``k=1`` regardless of the true geometry. The
    floor raises ``picked_k`` to ``min(min_dim, cap)`` (it can't exceed the
    usable eigenvector budget) and the diagnostics record both the
    heuristic pick and that the floor fired. ``None`` (default) leaves the
    pick to the heuristic alone.

    Defaults: ``k_nn = max(5, ceil(log K))``, ``bandwidth = median
    k-NN distance``.  Both are recorded in the diagnostics.

    A disconnected k-NN graph raises :class:`ValueError` with the
    component count — a degenerate heap with isolated centroids cannot
    be embedded by Laplacian eigenmaps.  Recommend the user raise
    ``k_nn`` or switch to PCA.

    Stays inside saklas's no-scipy rule.  Noisy below roughly 50 nodes
    (too few centroids for stable heat-kernel weights, spectral gap
    collapses into the eigenvalue noise floor); PCA is the right
    default at bundled-heap sizes.
    """
    gram = gram.to(torch.float32)
    K = gram.shape[0]
    if gram.dim() != 2 or gram.shape[1] != K:
        raise ValueError(
            f"spectral coord derivation needs a square (K, K) Gram, got "
            f"shape {tuple(gram.shape)}"
        )
    if K < 4:
        # Need at least 4 nodes to form any kind of k-NN graph and have
        # a candidate gap range.  Below that the heuristics are pure
        # noise; raise early rather than ship a meaningless embedding.
        raise ValueError(
            f"spectral coord derivation needs >= 4 centroids, got {K}"
        )
    # Graph build + normalized-Laplacian eigendecomposition, shared with the
    # sphere/torus spectral derivations.  The square + ``K < 4`` validation
    # above stays here so the spectral-specific error messages are preserved;
    # ``_laplacian_eigen`` re-checks defensively for its other callers.
    if _eigen_result is None:
        nontrivial_vals, nontrivial_vecs, k_nn, bandwidth = _laplacian_eigen(
            gram, k_nn=k_nn, bandwidth=bandwidth,
        )
    else:
        nontrivial_vals, nontrivial_vecs, k_nn, bandwidth = _eigen_result

    # Pick k by the eigenvalue-ratio heuristic.  For each candidate
    # ``k`` in ``[1, cap]`` the ratio ``nontrivial[k] / nontrivial[k-1]``
    # measures the multiplicative cliff between "kept" and "dropped"
    # eigenvalues — large when ``k`` separates a structural cluster of
    # low eigenvalues from a clearly higher group, near 1 when the
    # spectrum is smooth.  Picking the argmax-ratio is the standard
    # spectral-gap heuristic; the alternative absolute gap
    # ``nontrivial[k] - nontrivial[k-1]`` over-picks on S^1 because
    # eigenvalues scale ~ ``k²``.
    cap = min(max_dim, nontrivial_vals.shape[0] - 1, K - 2)
    cap = max(1, cap)
    if cap == 1:
        picked_k = 1
    else:
        # Ratio at "use k dims" = nontrivial[k] / nontrivial[k-1].
        # Clamp the denominator off zero — a vanishing eigenvalue at
        # ``k-1`` already means the graph is near-disconnected; the
        # ratio there is meaningless but mustn't NaN out the argmax.
        denom = nontrivial_vals[:cap].clamp(min=1e-12)
        ratios = nontrivial_vals[1:cap + 1] / denom
        picked_k = int(ratios.argmax().item()) + 1

    # Authored-dimensionality floor: honor a declared intrinsic dim over the
    # ratio cliff (which undershoots when one mode dominates — see docstring).
    # The floor can't exceed the usable eigenvector budget (``cap``).
    heuristic_k = picked_k
    if min_dim is not None:
        picked_k = max(picked_k, min(int(min_dim), cap))
    pinned = picked_k != heuristic_k

    # Absolute gap at the *final* picked_k (diagnostic annotation); 0 when
    # picked_k saturates the available spectrum so there's no λ_{k+1}.
    gap_magnitude = float(
        (nontrivial_vals[picked_k] - nontrivial_vals[picked_k - 1]).item()
        if picked_k < nontrivial_vals.shape[0]
        else 0.0
    )

    coords = nontrivial_vecs[:, :picked_k].contiguous()  # (K, picked_k)

    kept = min(max_dim + 5, nontrivial_vals.shape[0])
    diagnostics = SpectralDiagnostics(
        eigenvalues=nontrivial_vals[:kept].detach().clone(),
        picked_k=picked_k,
        gap_index=picked_k,
        gap_magnitude=gap_magnitude,
        bandwidth=bandwidth,
        k_nn=k_nn,
        component_count=1,
        heuristic_k=heuristic_k,
        min_dim=int(min_dim) if min_dim is not None else None,
        pinned=pinned,
    )
    return coords, diagnostics


def discover_coords(
    gram: torch.Tensor,
    method: str,
    **kwargs: Any,
) -> tuple[torch.Tensor, PcaDiagnostics | SpectralDiagnostics]:
    """Dispatch to :func:`derive_pca_coords` or :func:`derive_spectral_coords`.

    Exists so the pipeline doesn't branch on method strings; downstream
    code builds the consensus ``(K, K)`` Gram once and calls
    ``discover_coords(gram, method, **fit_kwargs)``, and the diagnostics
    ride into the sidecar through the same seam regardless of method.  Both
    methods embed the *same* layer-averaged whitened Gram — PCA linearly
    (eigendecomposition), spectral locally (k-NN Laplacian eigenmaps over
    the distances read off it).
    """
    if method == "pca":
        return derive_pca_coords(gram, **kwargs)
    if method == "spectral":
        return derive_spectral_coords(gram, **kwargs)
    raise ValueError(
        f"unknown discover method {method!r} (expected 'pca' | 'spectral')"
    )


# ===================================================== topology selection ===
#
# ``fit_mode="auto"`` picks the discover geometry instead of the user declaring
# it, in two decoupled decisions (the decoupling is what dodges the dimension
# bias that sinks naive reconstruction scoring — more spectral coordinates
# always "fit" better, so a single score always crowns the highest-dim
# candidate):
#
#   (a) flat vs curved — compare the flat affine (``pca``) and curved RBF
#       (``spectral``) fits, each at its own intrinsic dim, by GCV in a shared
#       whitened-reduced metric.  GCV's effective-dof penalty makes this a fair
#       model selection rather than a coordinate-count race.
#   (b) periodic axes — Vietoris–Rips H1 *persistent homology* counts the loops
#       (topologically robust: a circle and an ellipse both read as one loop, a
#       2-torus as two, a blob/arc/sphere as none), and the spectral eigenpairs
#       coordinate them.  A detected loop routes to a periodic ``BoxDomain``.
#
# Spheres are authored-only — ``S^n`` is a speculative topology that's the least
# reliable to detect from few centroids, so it is not an auto candidate.


@dataclass(frozen=True)
class TopologyCandidate:
    """One scored candidate in :func:`select_topology`'s ranking."""

    name: str            # display name, e.g. "flat-pca", "spectral", "torus-T1"
    fit_mode: str        # resolved fit_mode: "pca" (flat) | "spectral" (curved)
    intrinsic_dim: int
    score: float         # summed GCV (lower = better); inf if unscorable
    viable: bool
    reason: str = ""     # why excluded / how chosen (e.g. periodic detection note)


@dataclass(frozen=True)
class TopologyChoice:
    """The winning topology + the full ranked candidate field (for the sidecar)."""

    winner_name: str
    fit_mode: str                  # "pca" | "spectral"
    coords: torch.Tensor           # (K, n) winner intrinsic coords
    domain: ManifoldDomain         # CustomDomain | BoxDomain (periodic)
    candidates: tuple[TopologyCandidate, ...]   # ranked best-first
    # The winner's coordinate diagnostics — ``PcaDiagnostics`` for a flat
    # winner, ``SpectralDiagnostics`` for a curved / periodic one (the
    # Laplacian eigenpairs the spectral/periodic embedding rode), or ``None``
    # when unavailable.  Lets an ``auto`` fit emit a diagnostics block so the
    # inspector renders the same bars a pinned ``pca``/``spectral`` fit does.
    diagnostics: "PcaDiagnostics | SpectralDiagnostics | None" = None
    # Curved winner's already-normalized, factorized layout plan.  Auto-mode
    # spends this work while scoring the candidate; the final per-layer fit can
    # reuse it instead of repeating QR/eigh/LU over the same node geometry.
    rbf_plan: "RbfFitPlan | None" = None
    fisher_bases: "dict[int, torch.Tensor]" = field(default_factory=dict)


def _gcv_value(rss: float, edf: float, K: int) -> float:
    """The GCV score ``K · RSS / (K − edf)²`` — RSS penalized by effective dof.

    The ``(K − edf)²`` denominator is what gives model selection its parsimony:
    a higher-dimensional / more flexible candidate drives ``edf`` up, shrinking
    the denominator and *raising* its GCV unless the extra flexibility buys a
    commensurate RSS drop.  This is what stops a naive reconstruction error
    from always preferring the highest-dimensional topology.  ``edf ≥ K``
    (a saturated fit) returns ``+inf`` — it explains nothing out of sample.
    """
    slack = K - edf
    if slack <= 0.0:
        return math.inf
    return K * rss / (slack * slack)


def _ols_gcv_score(coords: torch.Tensor, targets: dict[int, torch.Tensor]) -> float:
    """Summed GCV of the affine (flat) map ``coords → target``.

    The flat candidate's surface is affine in its coordinates; its hat is the
    OLS projection ``H = C̃ (C̃ᵀC̃)⁺ C̃ᵀ`` (``C̃ = [1|coords]``) with effective
    dof ``edf = tr H = rank(C̃) = dim+1``.  Per-layer GCV (:func:`_gcv_value`)
    summed over layers — comparable to the curved candidates' GCV.
    """
    K = coords.shape[0]
    C = torch.cat([torch.ones(K, 1, dtype=torch.float32), coords.to(torch.float32)], dim=1)
    H = C @ torch.linalg.pinv(C.transpose(0, 1) @ C) @ C.transpose(0, 1)  # (K, K)
    edf = float(H.diagonal().sum().item())
    total = 0.0
    for y in targets.values():
        rss = float((y - H @ y).pow(2).sum().item())
        total += _gcv_value(rss, edf, K)
    return total


def _rbf_gcv_score(
    node_params: torch.Tensor,
    targets: dict[int, torch.Tensor],
    *,
    smoothing: float | str,
    plan: RbfFitPlan | None = None,
) -> float:
    """Summed GCV of the penalized RBF surface over a layout.

    The curved candidates (``spectral`` / ``S^n`` / ``T^d``) fit an ``r**3``
    RBF to the (unit-box-normalized) embedded coordinates.  Each layer's ``λ``
    is GCV-selected (:func:`_gcv_select_lambda` returns its GCV directly), and
    the score sums those per-layer GCVs — so every candidate is judged at its
    own best smoothing, the standard model-selection comparison.  Raises
    (propagating poisedness) if the layout can't carry an RBF — the caller
    marks that candidate non-viable.
    """
    _rbf_poised(node_params)
    lo = node_params.min(dim=0).values
    hi = node_params.max(dim=0).values
    norm = (node_params - lo) / (hi - lo).clamp(min=1e-9)
    K = norm.shape[0]
    plan = plan or prepare_rbf_fit_plan(norm, smoothing=smoothing)
    E, Q = plan.E, plan.Q
    fixed_smoother: torch.Tensor | None = None
    fixed_edf = 0.0
    if smoothing != "auto":
        denom_e = K * K - K
        e_scale = float(E.abs().sum() / denom_e) if denom_e > 0 else 1.0
        lam = float(smoothing) * (e_scale if e_scale > 0.0 else 1.0)
        fixed_smoother = _rbf_smoother_matrix(E, Q, lam)
        fixed_edf = float(fixed_smoother.diagonal().sum().item())
    total = 0.0
    for y in targets.values():
        if smoothing == "auto":
            _lam, _edf, gcv = _gcv_select_lambda(E, Q, y, plan=plan)
        else:
            assert fixed_smoother is not None
            rss = float((y - fixed_smoother @ y).pow(2).sum().item())
            gcv = _gcv_value(rss, fixed_edf, K)
        total += gcv
    return total


_HARMONIC_COHERENCE = 0.80        # |⟨z_new, z_acc^m⟩| above this ⇒ a harmonic, not a new axis
_HARMONIC_MAX_ORDER = 5


def _is_angular_harmonic(theta_new: torch.Tensor, accepted: list[torch.Tensor]) -> bool:
    """True if ``theta_new`` is an integer harmonic of an already-accepted angle.

    A single circle's Laplacian spectrum is ``cos kθ, sin kθ`` for ``k = 1, 2,
    …`` — every harmonic pair looks like a clean circle, so a naive scan would
    count one circle as a high-dimensional torus.  Two angles are the *same*
    circular axis when one is an integer multiple of the other: in the complex
    phase ``z = e^{iθ}``, ``θ_new = m·θ_acc`` ⇔ ``z_new = z_acc^m`` ⇔ the
    coherence ``|mean(z_new · conj(z_acc^m))| ≈ 1``.  Checks ``m = 1…5`` and the
    conjugate (opposite winding); a hit means ``theta_new`` is a harmonic of an
    existing fundamental and is *not* an independent periodic axis.
    """
    z_new = torch.polar(torch.ones_like(theta_new), theta_new)  # e^{iθ_new}
    for theta_acc in accepted:
        z_acc = torch.polar(torch.ones_like(theta_acc), theta_acc)
        for m in range(1, _HARMONIC_MAX_ORDER + 1):
            zm = z_acc.pow(m)
            coh = (z_new * zm.conj()).mean().abs()
            coh_conj = (z_new * zm).mean().abs()  # opposite winding
            if float(torch.maximum(coh, coh_conj).item()) >= _HARMONIC_COHERENCE:
                return True
    return False


def _rips_h1_persistence(
    distances: torch.Tensor,
    eps_max: float,
    *,
    max_triangles: int = 500_000,
) -> list[tuple[float, float]]:
    """H1 persistence pairs of the Vietoris–Rips filtration up to ``eps_max``.

    The robust, ellipse- and noise-tolerant **loop counter**: standard boundary-
    matrix reduction over the chain complex (vertices → edges → triangles,
    ordered by filtration value, ties broken by dimension).  A column reduces to
    a unique low; a triangle that lows out on an edge *kills* that edge's 1-cycle
    (a finite H1 pair ``(birth_edge_len, death_triangle_len)``), and a 1-cycle
    edge never killed up to ``eps_max`` is *essential* (death ``= ∞``).  Counting
    H1 classes with large persistence is the topological signal "there is a
    loop here" — invariant to the metric distortion (a circle vs. an ellipse)
    that breaks the eigenpair-geometry heuristics.

    Bounded for tractability: only simplices with all edges ``≤ eps_max`` enter
    the filtration, and the triangle list is capped at ``max_triangles``.  The
    cap is a **performance bound, not a free one**: truncating the
    largest-filtration triangles drops the boundaries that would *fill* the
    larger cycles, so a cycle can be left born-but-unfillable and miscounted as
    essential.  This manufactured a spurious 8-torus on the 107-node
    ``personas`` heap: an outlier-inflated ``eps_max`` made the Rips complex
    *complete* (every pair within the ceiling), whose true H1 is 0, but
    ``C(107,3) ≈ 198k`` triangles overran the old ``150k`` cap so ~650 cycles
    were left unfilled and miscounted.  The cap is therefore set high enough
    (``500k > C(143,3)``) to keep *every* triangle of a (near-)complete complex
    across the supported ``7 ≤ K ≤ 128`` periodic-detection regime, so
    truncation never manufactures a loop there; it only backstops a
    pathologically large-``K`` heap (where periodic detection isn't meaningful
    anyway).  Pure Python over small index sets; ``K`` is tens-to-low-hundreds
    and this runs once at fit time.  Returns the list of ``(birth, death)`` H1
    pairs.
    """
    K = int(distances.shape[0])
    # Edges with length <= eps_max, in a total filtration order.
    iu = torch.triu_indices(K, K, offset=1)
    lens_all = distances[iu[0], iu[1]]
    keep = lens_all <= eps_max
    ei = iu[0][keep].tolist()
    ej = iu[1][keep].tolist()
    el = lens_all[keep].tolist()
    order = sorted(range(len(el)), key=lambda x: (el[x], ei[x], ej[x]))
    ei = [ei[o] for o in order]
    ej = [ej[o] for o in order]
    el = [el[o] for o in order]
    E = len(el)
    edge_id: dict[tuple[int, int], int] = {(ei[x], ej[x]): x for x in range(E)}
    # Global simplex indices: vertices 0..K-1, edges K..K+E-1 (in filtration
    # order, so edge global index already respects the filtration).
    def eg(idx: int) -> int:
        return K + idx
    adj: list[set[int]] = [set() for _ in range(K)]
    for x in range(E):
        adj[ei[x]].add(ej[x])
        adj[ej[x]].add(ei[x])
    # Triangles: i<j<k with all three edges present; filtration = longest edge.
    dist = distances
    triangles: list[tuple[float, int, int, int]] = []
    for x in range(E):
        i, j = ei[x], ej[x]
        for k in adj[i] & adj[j]:
            if k > j:  # canonical i<j<k → each triangle once
                fl = max(el[x], float(dist[i, k].item()), float(dist[j, k].item()))
                triangles.append((fl, i, j, k))
    triangles.sort(key=lambda t: (t[0], t[1], t[2], t[3]))
    if len(triangles) > max_triangles:
        triangles = triangles[:max_triangles]

    # Reduction.  Vertices and edges first (H0); their positive (cycle-creating)
    # edges are found by union-find — equivalent to reducing the edge columns
    # over vertex rows, and cheaper.
    dsu = _DSU(K)
    positive_edges: set[int] = set()  # edge filtration indices that create a 1-cycle
    for x in range(E):
        # A tree edge kills an H0 component; a redundant edge births a loop.
        if not dsu.union(ei[x], ej[x]):
            positive_edges.add(x)

    # H1 deaths: reduce triangle columns over edge rows (global edge indices, so
    # the pivot respects filtration order).  A reduced low on a positive edge
    # pairs that loop's birth with this triangle's death.
    # Python integers are compact C-level bitsets over edge rows.  Symmetric
    # difference becomes one XOR and the pivot is ``bit_length()-1``; this is
    # the identical GF(2) reduction without allocating/copying Python sets for
    # every triangle column on dense auto-topology heaps.
    low_inv: dict[int, int] = {}          # pivot edge-global → reduced bit column
    killed: dict[int, float] = {}         # edge filtration index → death filtration
    for fl, i, j, k in triangles:
        col = (
            (1 << eg(edge_id[(min(i, j), max(i, j))]))
            | (1 << eg(edge_id[(min(i, k), max(i, k))]))
            | (1 << eg(edge_id[(min(j, k), max(j, k))]))
        )
        while col:
            piv = col.bit_length() - 1
            if piv in low_inv:
                col ^= low_inv[piv]
            else:
                break
        if not col:
            continue  # triangle creates an H2 void — irrelevant to H1
        piv = col.bit_length() - 1
        low_inv[piv] = col
        edge_idx = piv - K
        if edge_idx in positive_edges and edge_idx not in killed:
            killed[edge_idx] = fl

    pairs: list[tuple[float, float]] = []
    for idx in positive_edges:
        birth = el[idx]
        death = killed.get(idx, math.inf)
        pairs.append((birth, death))
    return pairs


def _count_persistent_loops(
    distances: torch.Tensor,
    *,
    persistence_frac: float = 0.5,
    max_dim: int = 8,
) -> int:
    """Number of significant H1 loops in the Rips filtration of ``distances``.

    Sets the filtration ceiling at ``2x`` the minimum-spanning-tree's longest
    edge — the connectivity scale ``eps_c`` is where the loop *closes*, and the
    ``2 eps_c`` window is wide enough to birth it yet narrow enough that
    cross-chords (which would slice an elongated loop into spurious sub-loops)
    haven't formed.  In that window a genuine ``S^1`` / ``T^d`` loop stays
    **essential** — a 1-D hole never fills until the whole structure does, at
    ``eps`` far above ``eps_c`` — while a 2-D surface's holes (a sphere) and
    noise loops are *finite*, born and filled within a few ``eps_c``.  Counting
    only essential loops (unfilled at ``eps_max``) whose persistence ``eps_max −
    birth`` clears ``persistence_frac · eps_c`` therefore separates a true loop
    from a sphere's transient surface holes and from noise, and is robust to the
    metric distortion (circle vs. elongated ellipse) that defeats eigenpair
    geometry.  Circle / ellipse / noisy circle: 1.  ``T^d``: ``d``.  Blob, arc,
    line: 0.  A 2-D *spherical* surface is out of scope (``S^n`` is authored-
    only, not auto-selected) and at some sampling densities can leave one
    surface hole essential inside the window — a known, accepted false positive
    for an unsupported auto topology, not a target case.
    """
    K = int(distances.shape[0])
    if K < 4:
        return 0
    # Connectivity scale ε_c: the largest edge in the MST (Kruskal).
    iu = torch.triu_indices(K, K, offset=1)
    lens = distances[iu[0], iu[1]]
    order = torch.argsort(lens)
    dsu = _DSU(K)
    eps_c = 0.0
    joined = 0
    src, dst = iu[0].tolist(), iu[1].tolist()
    for o in order.tolist():
        if dsu.union(src[o], dst[o]):
            eps_c = float(lens[o].item())
            joined += 1
            if joined == K - 1:
                break
    if eps_c <= 0.0:
        return 0
    eps_max = 2.0 * eps_c
    # At the ceiling a complete Rips graph has the full simplex as its clique
    # complex, hence trivial H1.  This is exactly the outlier-inflated personas
    # case and lets the loop *counter* skip constructing/reducing O(K^3)
    # triangles.  ``_rips_h1_persistence`` itself keeps its full finite-pair
    # contract; this shortcut is valid here because the caller only counts
    # classes still essential at ``eps_max``.
    if bool((lens <= eps_max).all()):
        return 0
    pairs = _rips_h1_persistence(distances, eps_max)
    threshold = persistence_frac * eps_c
    count = 0
    for birth, death in pairs:
        # Essential only (a finite death means a 2-D surface hole / noise loop
        # that filled inside the window — not a real 1-D cycle).
        if math.isfinite(death):
            continue
        if (eps_max - birth) >= threshold:
            count += 1
    return min(count, max_dim)


# Single-cycle fallback thresholds (complement H1 persistence, which only counts
# *fat* loops — a thin ring slips under it).  The detector covers two sampling
# regimes: a **uniform** ring (near-equidistant nodes — a faint thin loop) and a
# **clustered** ring (tight clumps spaced around the loop — the seasonal sampling
# real concept families have: months→seasons, days→weekday/weekend).  Validated
# for sensitivity (synthetic faint + clustered rings, real day-of-week centroids)
# and specificity (~0% false-positive on random Gaussian heaps K>=9; lines, open
# arcs, grids, branched theta/Y, and high-D persona-style fans all rejected).
_CYCLE_MIN_NODES = 7        # below this, too few points to detect a ring reliably
_CYCLE_MAX_NODES = 128      # above this, robust loops are visible to H1; cap the tour
_CYCLE_MAX_DEGREE = 3       # uniform path: symmetric 2-NN degree of a 1-D loop
_CYCLE_CLOSURE_MAX = 2.0    # uniform path: max/median tour edge (no single long edge)
_CYCLE_RECALL_MIN = 0.90    # uniform path: tour-neighbour top-2 recall (loop is local)
_CYCLE_CONTRAST_MIN = 1.08  # mean d(sep=2)/d(sep=1): genuine cyclic growth (both paths)
# Clustered-ring path: tight clumps make the tour edges *bimodal* (tiny intra-
# cluster, large inter-cluster), so closure/recall fail — but the inter-cluster
# gaps are >=2, mutually regular, and the loop has a real far antipode.  These
# reject what would otherwise leak in: an open arc (exactly 1 gap), a blob/fan (no
# antipode), and a 2-D grid (gaps are noise-marginal, not decisively bimodal).
_CYCLE_MAX_DEGREE_CLUSTER = 4   # a tight clump pushes a node to degree 4 (vs 1-D's 2-3)
_CYCLE_LARGE_FACTOR = 2.5       # a tour edge is an inter-cluster gap past this x small-scale
_CYCLE_GAPS_MIN = 2             # >=2 regular gaps => a closed cycle of clumps (1 => arc)
_CYCLE_GAP_REG_MAX = 2.5        # the gaps must be mutually comparable (an even-ish ring)
_CYCLE_ANTIPODE_MIN = 2.5       # tour-antipode/tour-neighbour distance: a ring has a far side
_CYCLE_BIMODAL_MIN = 3.5        # the *smallest* gap must clear this x small-scale: a clustered
                                # ring is *decisively* bimodal (tight clumps, big gaps, no edge
                                # in between), screening a diffuse low-D cloud whose many
                                # accidental long edges only marginally clear _CYCLE_LARGE_FACTOR


def _nn_tour(dist: torch.Tensor) -> list[int]:
    """Greedy nearest-neighbour tour (best of all starts) + 2-opt — a cheap,
    deterministic TSP heuristic recovering a candidate cyclic order over the K
    centroids.  ``O(K^3)``; the caller gates ``K <= _CYCLE_MAX_NODES``."""
    D = dist.tolist()
    K = len(D)
    best_tour: list[int] | None = None
    best_len = math.inf
    for start in range(K):
        unvisited = set(range(K))
        unvisited.discard(start)
        tour = [start]
        while unvisited:
            row = D[tour[-1]]
            nxt = min(unvisited, key=row.__getitem__)
            tour.append(nxt)
            unvisited.discard(nxt)
        length = sum(D[tour[i]][tour[(i + 1) % K]] for i in range(K))
        if length < best_len:
            best_len, best_tour = length, tour
    tour = best_tour if best_tour is not None else list(range(K))
    improved = True
    while improved:
        improved = False
        for i in range(K - 1):
            for j in range(i + 2, K):
                if i == 0 and j == K - 1:
                    continue  # the closing edge — reversing it is a no-op
                a, b = tour[i], tour[i + 1]
                c, e = tour[j], tour[(j + 1) % K]
                if D[a][b] + D[c][e] > D[a][c] + D[b][e] + 1e-9:
                    tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                    improved = True
    return tour


def _faint_cycle_coords(distances: torch.Tensor) -> torch.Tensor | None:
    """Recover a single periodic coordinate for an ``S^1`` that H1 misses.

    Vietoris–Rips H1 persistence counts loops by *hole size*, so two kinds of
    ring slip under its threshold even though both are unambiguously cyclic by
    graph topology.  This complementary test runs only when persistence found
    nothing, and recovers the cyclic order in either of two sampling regimes:

    **Uniform** — a faint ring: a small cyclic modulation on a near-equidistant
    heap (e.g. day-of-week centroids at ~16% modulation).  Too thin a hole for
    H1, but near-equidistant, so the original guards fire: **1-D** (symmetric
    2-NN max degree ``<= _CYCLE_MAX_DEGREE``), **closed** (greedy+2-opt tour
    edges near-uniform, ``max/median < _CYCLE_CLOSURE_MAX`` — a line/arc needs
    one long closing edge), and **local** (each node's two tour-neighbours among
    its two nearest, ``recall >= _CYCLE_RECALL_MIN``).

    **Clustered** — tight clumps spaced around the loop, the sampling real
    concept families have (months→seasons, days→weekday/weekend).  Here the tour
    edges are **bimodal** (tiny intra-cluster, large inter-cluster), so closure
    and recall both fail though the loop is real.  It is accepted when the
    inter-cluster gaps (edges ``> _CYCLE_LARGE_FACTOR`` × the small-edge scale)
    are (a) ``>= _CYCLE_GAPS_MIN`` in number, (b) mutually regular
    (``max/min <= _CYCLE_GAP_REG_MAX``), and (c) the loop has a real far antipode
    (tour-antipode/tour-neighbour mean distance ``>= _CYCLE_ANTIPODE_MIN``).
    Each guard rejects a distinct impostor that the bimodality alone would admit:
    an **open arc** has exactly one large edge (its closing chord), a **blob/fan**
    has no antipode, and a **2-D grid**'s gaps are noise-marginal — not decisively
    bimodal — so they don't clear ``_CYCLE_LARGE_FACTOR``.  Both regimes also
    require **graded** growth (mean ``d(sep=2)/d(sep=1) >= _CYCLE_CONTRAST_MIN``)
    and ``1-D``-ness (degree ``<= _CYCLE_MAX_DEGREE_CLUSTER``, looser than the
    uniform path's ``_CYCLE_MAX_DEGREE`` because a tight clump reaches degree 4).

    Returns the per-node angle ``(K,)`` = ``2*pi*tour_rank/K`` (a uniform ``S^1``
    parameterisation in the recovered cyclic *order* — exact spacing is dropped,
    which is fine: the loop's topology, not its metric, is what the periodic
    domain needs), or ``None``.  The clustered path's bimodal-gap test trades two
    documented false-negatives — a very-loose cluster heap approaching uniform,
    and an eccentric ellipse ``> 6:1`` (its gaps aren't decisively bimodal) — for
    a ~0% false-positive rate that holds against grids, fans, arcs and blobs.
    """
    K = int(distances.shape[0])
    if K < _CYCLE_MIN_NODES or K > _CYCLE_MAX_NODES:
        return None
    dist = distances.detach().to(torch.float32)
    # 1-D filter (shared, at the looser clustered bound): a high-D fan has degree
    # >> 4 and is rejected before any tour runs; the uniform path tightens to 3.
    nn2 = dist.argsort(dim=1)[:, 1:3].tolist()
    deg: dict[int, int] = {}
    edges_uv: set[tuple[int, int]] = set()
    for i, row in enumerate(nn2):
        for j in row:
            edges_uv.add((i, j) if i < j else (j, i))
    for u, v in edges_uv:
        deg[u] = deg.get(u, 0) + 1
        deg[v] = deg.get(v, 0) + 1
    max_degree = max(deg.values())
    if max_degree > _CYCLE_MAX_DEGREE_CLUSTER:
        return None
    # Recover the cyclic order (shared by both regimes).
    tour = _nn_tour(dist)
    D = dist.tolist()
    pos = [0] * K
    for i, n in enumerate(tour):
        pos[n] = i
    tour_edges = [D[tour[i]][tour[(i + 1) % K]] for i in range(K)]
    sorted_edges = sorted(tour_edges)
    median = (sorted_edges[K // 2] if K % 2
              else 0.5 * (sorted_edges[K // 2 - 1] + sorted_edges[K // 2]))
    if median <= 0.0:
        return None
    # graded (shared): mean distance grows from cyclic separation 1 to 2 — a real
    # cyclic order, not a flat simplex.
    s1 = s1n = s2 = s2n = 0.0
    for a in range(K):
        for b in range(a + 1, K):
            sep = min((pos[a] - pos[b]) % K, (pos[b] - pos[a]) % K)
            if sep == 1:
                s1 += D[a][b]
                s1n += 1
            elif sep == 2:
                s2 += D[a][b]
                s2n += 1
    if s1n == 0 or s2n == 0:
        return None
    if (s2 / s2n) / max(s1 / s1n, 1e-9) < _CYCLE_CONTRAST_MIN:
        return None
    # Uniform regime: near-equidistant nodes — tight closure + local recall.
    nn2_set = [set(row) for row in nn2]
    hits = sum(len({tour[(i - 1) % K], tour[(i + 1) % K]} & nn2_set[n])
               for i, n in enumerate(tour))
    recall = hits / (2 * K)
    closure = sorted_edges[-1] / median
    uniform_ok = (max_degree <= _CYCLE_MAX_DEGREE
                  and closure < _CYCLE_CLOSURE_MAX
                  and recall >= _CYCLE_RECALL_MIN)
    # Clustered regime: >=2 regular, decisively-bimodal inter-cluster gaps + a
    # real far antipode.  The bimodality strength (smallest gap >> small scale) is
    # what screens a diffuse low-D random cloud, whose tour throws off many edges
    # that only marginally clear _CYCLE_LARGE_FACTOR.
    small = sorted_edges[:max(1, K // 2)]
    small_scale = small[len(small) // 2]
    gaps = [e for e in tour_edges if e > _CYCLE_LARGE_FACTOR * small_scale]
    half = K // 2
    near = sum(tour_edges) / K
    far = sum(D[tour[i]][tour[(i + half) % K]] for i in range(K)) / K
    antipode = far / max(near, 1e-9)
    clustered_ok = (len(gaps) >= _CYCLE_GAPS_MIN
                    and min(gaps) >= _CYCLE_BIMODAL_MIN * small_scale
                    and max(gaps) / min(gaps) <= _CYCLE_GAP_REG_MAX
                    and antipode >= _CYCLE_ANTIPODE_MIN)
    if not (uniform_ok or clustered_ok):
        return None
    # Uniform S^1 coordinate in the recovered cyclic order.
    angles = torch.zeros(K, dtype=torch.float32)
    for i, n in enumerate(tour):
        angles[n] = 2.0 * math.pi * i / K
    return angles


def _detect_periodic_axes(
    distances: torch.Tensor,
    eigvecs: torch.Tensor,
    *,
    max_dim: int,
    persistence_frac: float = 0.5,
) -> tuple[torch.Tensor, int] | None:
    """Detect periodic axes — part (b), persistent homology + spectral angles.

    Two stages with a clean division of labour:

    1. **Count** the loops with :func:`_count_persistent_loops` (Vietoris–Rips
       H1 persistence).  This is the topologically robust part: it sees a circle
       *and* an ellipse as one loop, a 2-torus as two, and a blob / open arc as
       none — immune to the metric distortion that defeats eigenpair geometry.

    2. **Coordinate** each detected loop from the spectral eigenmap.  An ellipse's
       ``atan2`` of its fundamental eigenpair still winds once around the loop
       (monotonically, if non-uniformly), so it is a valid periodic coordinate —
       the count from stage 1 caps how many independent eigenpair-angles to take,
       which is exactly what stops a single circle's ``cos kθ`` harmonics from
       being miscounted (:func:`_is_angular_harmonic` skips harmonics of an
       already-taken axis).

    When persistence counts **zero** loops, a complementary single-cycle
    fallback (:func:`_faint_cycle_coords`) runs: PH measures hole *size*, so a
    thin ring (a faint cyclic modulation on a near-equidistant heap — e.g.
    day-of-week centroids) slips under its persistence threshold even though it
    is unambiguously cyclic by graph topology.  The fallback recovers one
    periodic axis from a guarded tour test, or confirms there is no cycle.

    Returns ``(angles (K, d), loop_count)`` for the ``d`` periodic axes, or
    ``None`` if no loop persists.
    """
    d = _count_persistent_loops(
        distances, persistence_frac=persistence_frac, max_dim=max_dim,
    )
    if d < 1:
        # PH found no *fat* loop; try the faint single-cycle fallback before
        # conceding there is no periodicity.
        theta = _faint_cycle_coords(distances)
        if theta is None:
            return None
        return theta.unsqueeze(1), 1
    avail = int(eigvecs.shape[1])
    angles: list[torch.Tensor] = []
    p = 0
    # Take the d lowest *independent* eigenpair angles (skip harmonics of an
    # axis already taken); PH count d is the ground truth for how many.
    while p * 2 + 1 < avail and len(angles) < d:
        cos_j = eigvecs[:, 2 * p]
        sin_j = eigvecs[:, 2 * p + 1]
        theta = torch.atan2(sin_j, cos_j)
        if not _is_angular_harmonic(theta, angles):
            angles.append(theta)
        p += 1
    if not angles:
        return None
    return torch.stack(angles, dim=1).contiguous(), d


def select_topology(
    stacks: dict[int, torch.Tensor],
    layer_grams: dict[int, torch.Tensor],
    consensus_gram: torch.Tensor,
    *,
    whitener: "LayerWhitener",
    whitened_rows: dict[int, torch.Tensor] | None = None,
    max_dim: int = 8,
    smoothing: float | str = "auto",
    score_dim: int | None = None,
    k_nn: int | None = None,
    bandwidth: float | None = None,
    persistence_frac: float = 0.5,
) -> TopologyChoice:
    """Pick the manifold topology for a discover heap — flat vs curved vs periodic.

    Two decisions, deliberately decoupled to avoid the dimension bias that
    sinks naive reconstruction scoring (where more spectral coordinates always
    "fit" better, so the highest-dim candidate always wins):

    **(a) flat vs curved** — compare the flat affine fit (``pca`` mode, at its
    PCA variance-threshold dim) against the curved RBF fit (``spectral`` mode,
    floored to the *same* dim) by **GCV** in a shared per-layer whitened / Fisher
    reduced metric (``targets[L] = X̃_L · basis_Lᵀ``).  GCV's ``(K − edf)²``
    denominator charges each mode for its own effective dof, so a more flexible
    candidate must earn its flexibility — a fair model-selection comparison
    rather than a coordinate-count race.  The curved candidate's dim is floored
    to the flat candidate's (``min_dim=k_flat``) because the spectral
    eigenvalue-ratio cliff systematically *undershoots* (one dominant Fiedler
    mode picks ``k=1``); without the floor the curved fit competes starved of
    coordinates and a curved manifold linearly embedded in a ``k_flat``-plane is
    mislabelled flat — the flat affine fit reconstructs it near-perfectly while
    the under-dimensioned curved fit cannot, even though a dim-matched curved fit
    *wins*.

    **(b) periodic axes** — independently, :func:`_detect_periodic_axes` counts
    loops by Vietoris–Rips H1 *persistent homology* (a circle and an ellipse both
    read as one loop, a 2-torus as two, a blob/arc as none), with a guarded
    single-cycle fallback (:func:`_faint_cycle_coords`) for faint rings whose
    hole is too thin to clear the persistence threshold.  Detected circles are
    *topology*, not surface shape — a circle can be linearly embedded yet still
    needs a periodic domain to steer around rather than across — so a confident
    detection routes to a periodic ``BoxDomain`` (the curved path).  Spheres are
    **not** auto-selected (a speculative topology that's the least reliable to
    detect from few centroids); ``S^n`` is available as an *authored* domain only.

    Returns a :class:`TopologyChoice` carrying the winner's ``fit_mode`` /
    ``coords`` / ``domain`` plus the ranked candidate field for the sidecar.
    ``whitener`` must cover every fit layer (the caller gates on
    ``covers_all``).
    """
    fit_layers = sorted(layer_grams.keys())
    K = consensus_gram.shape[0]
    R = score_dim if score_dim is not None else min(max_dim, K - 1)
    R = max(1, R)

    # Shared per-layer whitened/Fisher reduced targets — every candidate
    # predicts the *same* y in the *same* metric, so the comparison isolates
    # the coordinate geometry + surface, not the basis (common-mode).
    targets: dict[int, torch.Tensor] = {}
    fisher_bases: dict[int, torch.Tensor] = {}
    for L in fit_layers:
        X = stacks[L].to(torch.float32)
        X = X - X.mean(dim=0, keepdim=True)
        basis, _ev = _pca_basis(
            X, n_components=R, whitener=whitener, layer=L,
            whitened_gram=layer_grams[L],
            whitened_rows=(
                whitened_rows.get(L) if whitened_rows is not None else None
            ),
        )
        fisher_bases[L] = basis
        targets[L] = (X @ basis.transpose(0, 1)).contiguous()      # (K, R)

    candidates: list[TopologyCandidate] = []

    # (a) Flat (pca) — always viable for K >= 2.
    coords_flat, pca_diag = derive_pca_coords(consensus_gram, max_dim=max_dim)
    k_flat = int(coords_flat.shape[1])
    gcv_flat = _ols_gcv_score(coords_flat, targets)
    candidates.append(TopologyCandidate("flat-pca", "pca", k_flat, gcv_flat, True))

    # (a) Curved Euclidean (spectral) — may fail on a tiny / disconnected heap.
    gcv_curved = math.inf
    curved: tuple[torch.Tensor, ManifoldDomain, RbfFitPlan] | None = None
    spec_diag: object | None = None
    spectral_eigen: tuple[torch.Tensor, torch.Tensor, int, float] | None = None
    try:
        # Floor the curved candidate's intrinsic dim to the flat PCA dim so flat
        # and curved are compared at *matched expressiveness*.  The spectral
        # eigenvalue-ratio cliff systematically undershoots (its documented
        # failure mode: one dominant Fiedler mode makes λ₂/λ₁ the largest ratio,
        # picking k=1 regardless of the true geometry), which starves the curved
        # RBF of coordinates and biases the GCV comparison toward flat — a curved
        # manifold linearly embedded in a k_flat-plane is reconstructed near-
        # perfectly by the flat k_flat-affine fit, but the under-dimensioned curved
        # fit can't match it and loses on reconstruction it would *win* at matched
        # dim.  ``min_dim`` is the floor :func:`derive_spectral_coords` already
        # carries for exactly this undershoot; wiring it into auto-mode is what
        # makes the flat-vs-curved verdict trustworthy rather than an artifact of
        # the dim mismatch.  (The periodic path sets its own dim from the H1 loop
        # count, so it is unaffected.)
        spectral_eigen = _laplacian_eigen(
            consensus_gram, k_nn=k_nn, bandwidth=bandwidth,
        )
        coords_spec, spec_diag = derive_spectral_coords(
            consensus_gram, max_dim=max_dim, min_dim=k_flat,
            k_nn=k_nn, bandwidth=bandwidth,
            _eigen_result=spectral_eigen,
        )
        k_spec = int(coords_spec.shape[1])
        if (2 * k_spec + 1) > K:
            raise ValueError(f"poisedness floor 2n+1={2 * k_spec + 1} exceeds K={K}")
        spec_plan = prepare_rbf_fit_plan(
            coords_spec.to(torch.float32), smoothing=smoothing,
        )
        gcv_curved = _rbf_gcv_score(
            coords_spec.to(torch.float32), targets, smoothing=smoothing,
            plan=spec_plan,
        )
        curved = (coords_spec, CustomDomain(k_spec), spec_plan)
        candidates.append(TopologyCandidate("spectral", "spectral", k_spec, gcv_curved, True))
    except (ValueError, RuntimeError) as e:  # _LinAlgError ⊂ RuntimeError
        candidates.append(TopologyCandidate("spectral", "spectral", 0, math.inf, False, str(e)))

    # (b) Periodic (circle / torus) detection — persistent homology counts the
    # loops (ellipse/noise-robust), spectral eigenpairs coordinate them.
    periodic: tuple[
        torch.Tensor, ManifoldDomain, float, RbfFitPlan,
    ] | None = None
    try:
        if spectral_eigen is None:
            spectral_eigen = _laplacian_eigen(
                consensus_gram, k_nn=k_nn, bandwidth=bandwidth,
            )
        _vals, eigvecs, _knn, _bw = spectral_eigen
        # Whitened pairwise distances off the consensus Gram (same metric the
        # eigenmap embeds): d²_ij = G_ii + G_jj − 2 G_ij.
        cg = 0.5 * (consensus_gram + consensus_gram.transpose(0, 1))
        diag = cg.diagonal()
        d2 = diag.unsqueeze(0) + diag.unsqueeze(1) - 2.0 * cg
        distances = d2.clamp(min=0.0).sqrt()
        distances.fill_diagonal_(0.0)
        detected = _detect_periodic_axes(
            distances, eigvecs, max_dim=max_dim,
            persistence_frac=persistence_frac,
        )
        if detected is not None:
            p_coords, n_loops = detected
            d = int(p_coords.shape[1])
            if (2 * d + 1) <= K:
                axes = [
                    BoxAxis(f"theta{i}", periodic=True, period=2.0 * math.pi)
                    for i in range(d)
                ]
                p_domain = BoxDomain(axes)
                p_params = p_domain.embed(p_coords).to(torch.float32)
                periodic_plan = prepare_rbf_fit_plan(
                    p_params, smoothing=smoothing,
                )
                gcv_p = _rbf_gcv_score(
                    p_params, targets, smoothing=smoothing,
                    plan=periodic_plan,
                )
                note = f"H1 persistent loops = {n_loops}"
                candidates.append(TopologyCandidate(
                    f"torus-T{d}", "spectral", d, gcv_p, True, note,
                ))
                periodic = (p_coords, p_domain, gcv_p, periodic_plan)
    except (ValueError, RuntimeError):
        pass  # no clean eigenmap ⇒ no periodic candidate

    # Decision.  A confidently-detected periodic topology wins outright: the
    # circularity test (constant radius + full coverage + harmonic dedup) is a
    # strong, conservative geometric signal, and periodicity is the correct
    # steering geometry *even when a flat plane reconstructs the centroids
    # better* — a linearly-embedded circle lives in a 2-plane, so flat always
    # wins reconstruction, yet you still want to steer *around* the loop, not
    # across the chord.  Gating periodicity on GCV-vs-flat would therefore
    # always reject the correct circle; instead the geometric test is trusted,
    # guarded only against a degenerate (non-finite) periodic fit.  Absent a
    # circle, the lower-GCV of flat vs curved wins.
    if periodic is not None and math.isfinite(periodic[2]):
        p_coords, p_domain, _gcv_p, winner_plan = periodic
        win_name = f"torus-T{int(p_coords.shape[1])}"
        win_mode, win_coords, win_domain = "spectral", p_coords, p_domain
        win_diag = spec_diag  # periodic rides the spectral eigenpairs
    elif curved is not None and gcv_curved < gcv_flat:
        win_name = "spectral"
        win_mode, win_coords, win_domain = "spectral", curved[0], curved[1]
        win_diag = spec_diag
        winner_plan = curved[2]
    else:
        win_name = "flat-pca"
        win_mode, win_coords, win_domain = "pca", coords_flat, CustomDomain(k_flat)
        win_diag = pca_diag
        winner_plan = None

    candidates.sort(key=lambda c: (not c.viable, c.score))
    return TopologyChoice(
        winner_name=win_name,
        fit_mode=win_mode,
        coords=win_coords,
        domain=win_domain,
        candidates=tuple(candidates),
        diagnostics=win_diag,
        rbf_plan=winner_plan,
        fisher_bases=fisher_bases,
    )


__all__ = [
    "PcaDiagnostics",
    "SpectralDiagnostics",
    "TopologyCandidate",
    "TopologyChoice",
    "derive_pca_coords",
    "derive_spectral_coords",
    "discover_coords",
    "neutral_layout_coord",
    "select_topology",
]
