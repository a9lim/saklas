"""Manifold steering primitives — arbitrary-dimensional, arbitrary-topology.

Implements the activation-manifold half of Goodfire's "Manifold Steering"
(arXiv 2605.05115) for saklas, generalized from a 1-D curve to a manifold
of arbitrary intrinsic dimension and topology.

A linear A->B steering vector cuts a straight chord through activation
space; that chord passes through low-density off-manifold regions, which
shows up behaviorally as "teleportation" (unnatural intermediate states)
and diversity collapse. Interpolating instead through per-concept
activation *centroids* keeps the trajectory on the learned manifold.

Geometry lives in a :class:`ManifoldDomain`: an embedding of an
``n``-dimensional intrinsic manifold into ``R^m`` plus a distance
function. :class:`BoxDomain` covers Euclidean boxes/disks, cylinders and
tori (per-axis periodicity); :class:`SphereDomain` covers ``S^n``;
:class:`CustomDomain` is the escape hatch for non-orientable or otherwise
exotic surfaces given an explicit immersion (at chordal, not geodesic,
fidelity — and on a non-orientable domain a :meth:`Manifold.tangent`
frame cannot be globally combed across the orientation flip).

The per-layer interpolant is a single ``r**3`` polyharmonic RBF with an
affine polynomial term, valid in every dimension. At ``n == 1`` with an
open axis it reproduces the natural cubic spline exactly (the 1-D
order-2 polyharmonic spline *is* the natural cubic), so this module
subsumes the former cubic-spline machinery rather than running beside it.

This module is pure tensor math (fp32, no session/IO coupling), mirroring
how :mod:`saklas.core.vectors` holds the low-level extraction primitives.
The RBF fit solves a small dense symmetric-indefinite saddle system with
``torch.linalg.solve`` -- node counts are tiny (on the order of ten to
thirty) and fitting is a one-shot operation, not a hot path. scipy is not
pulled in. ``eval_rbf`` and :func:`subspace_replace` are the only
functions reachable from the generation hot path; both are
allocation-light and free of host syncs.
"""
from __future__ import annotations

import json
import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch
from safetensors.torch import load_file, save_file

log = logging.getLogger(__name__)


# Default PCA width for a fitted manifold subspace.  Matches the paper's
# 64-dimensional reduction.  Clamped down to ``min(64, K-1, rank)`` when
# the node count ``K`` is small -- ``K`` centered centroids span a space
# of rank at most ``K-1``.
DEFAULT_N_COMPONENTS = 64

# Grid resolution for the inverse parameterization (nearest-point
# projection onto the manifold).  Used only by the naturalness eval,
# never by the steering hot path -- the hook steers to a fixed position.
DEFAULT_INVERSION_RESOLUTION = 512


# ================================================================ domains ===
#
# A manifold's geometry is an embedding of an n-dimensional intrinsic
# manifold into R^m plus a distance function.  The RBF interpolant only
# ever needs pairwise distances between embedded points and an embedding
# map, so any topology that can be embedded -- box, cylinder, torus,
# sphere, or an explicit immersion of a non-orientable surface -- is
# expressible without touching the interpolant.


class ManifoldDomain(ABC):
    """Embedding of an n-D intrinsic manifold into R^m, plus a metric.

    Subclasses define :meth:`embed` (authoring coords -> embedded
    coords), :meth:`embed_jacobian`, :meth:`clamp_position` and
    :meth:`sample_positions`.  :meth:`distance` defaults to the chordal
    (Euclidean-in-embedding) metric and is rarely overridden -- a
    periodic axis embedded as a circle already wraps correctly under the
    chordal metric.
    """

    @property
    @abstractmethod
    def intrinsic_dim(self) -> int:
        """Dimension ``n`` of the intrinsic manifold (number of authoring axes)."""

    @property
    @abstractmethod
    def embed_dim(self) -> int:
        """Dimension ``m`` of the embedding space."""

    @abstractmethod
    def embed(self, coords: torch.Tensor) -> torch.Tensor:
        """Map authoring coords ``(.., n)`` to embedded coords ``(.., m)``."""

    @abstractmethod
    def embed_jacobian(self, coords: torch.Tensor) -> torch.Tensor:
        """Jacobian ``d embed / d coords`` at a single point: ``(n,) -> (m, n)``."""

    @abstractmethod
    def clamp_position(self, coords: torch.Tensor) -> torch.Tensor:
        """Clamp open axes to range, wrap periodic axes; ``(.., n) -> (.., n)``."""

    @abstractmethod
    def sample_positions(self, resolution: int) -> torch.Tensor:
        """A grid of authoring coords covering the domain: ``(G, n)``."""

    @abstractmethod
    def to_spec(self) -> dict[str, Any]:
        """Serialize to the ``manifold.json`` tagged-union ``domain`` object."""

    def distance(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Chordal distance between embedded points ``(.., m)`` -> ``(..,)``."""
        return torch.linalg.vector_norm(a - b, dim=-1)


@dataclass(frozen=True)
class BoxAxis:
    """One axis of a :class:`BoxDomain` -- open (an interval) or periodic."""

    name: str
    periodic: bool
    period: float = 1.0
    lo: float = 0.0
    hi: float = 1.0


class BoxDomain(ManifoldDomain):
    """A product of intervals and circles: box/disk, cylinder, n-torus.

    Each axis is open (contributes its raw coordinate) or periodic (a
    circle of the given ``period``, embedded as ``(cos, sin)``).  All
    axes open -> a Euclidean box; all periodic -> the n-torus; a mix ->
    a cylinder.  ``n == 1`` all-open reproduces the natural cubic spline;
    ``n == 1`` periodic is the closed-loop manifold.
    """

    def __init__(self, axes: Sequence[BoxAxis]):
        self._axes = tuple(axes)
        if not self._axes:
            raise ValueError("BoxDomain needs at least one axis")
        m = 0
        for ax in self._axes:
            m += 2 if ax.periodic else 1
        self._embed_dim = m

    @property
    def axes(self) -> tuple[BoxAxis, ...]:
        return self._axes

    @property
    def intrinsic_dim(self) -> int:
        return len(self._axes)

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def embed(self, coords: torch.Tensor) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        for i, ax in enumerate(self._axes):
            ci = coords[..., i]
            if ax.periodic:
                w = 2.0 * math.pi / ax.period
                parts.append(torch.cos(w * ci))
                parts.append(torch.sin(w * ci))
            else:
                parts.append(ci)
        return torch.stack(parts, dim=-1)

    def embed_jacobian(self, coords: torch.Tensor) -> torch.Tensor:
        n = self.intrinsic_dim
        m = self.embed_dim
        J = torch.zeros(m, n, dtype=coords.dtype, device=coords.device)
        out = 0
        for i, ax in enumerate(self._axes):
            if ax.periodic:
                w = 2.0 * math.pi / ax.period
                ci = coords[i]
                J[out, i] = -w * torch.sin(w * ci)
                J[out + 1, i] = w * torch.cos(w * ci)
                out += 2
            else:
                J[out, i] = 1.0
                out += 1
        return J

    def clamp_position(self, coords: torch.Tensor) -> torch.Tensor:
        out = coords.clone()
        for i, ax in enumerate(self._axes):
            if ax.periodic:
                out[..., i] = torch.remainder(coords[..., i], ax.period)
            else:
                out[..., i] = coords[..., i].clamp(min=ax.lo, max=ax.hi)
        return out

    def sample_positions(self, resolution: int) -> torch.Tensor:
        axis_grids: list[torch.Tensor] = []
        for ax in self._axes:
            if ax.periodic:
                # Exclude the wrap endpoint -- it duplicates 0.
                axis_grids.append(
                    torch.linspace(0.0, ax.period, resolution + 1)[:-1]
                )
            else:
                axis_grids.append(
                    torch.linspace(ax.lo, ax.hi, resolution)
                )
        mesh = torch.meshgrid(*axis_grids, indexing="ij")
        return torch.stack([g.reshape(-1) for g in mesh], dim=-1)

    def to_spec(self) -> dict[str, Any]:
        return {
            "type": "box",
            "axes": [
                {
                    "name": ax.name,
                    "periodic": ax.periodic,
                    "period": ax.period,
                    "lo": ax.lo,
                    "hi": ax.hi,
                }
                for ax in self._axes
            ],
        }


class SphereDomain(ManifoldDomain):
    """The n-sphere ``S^n``, embedded as unit vectors in ``R^(n+1)``.

    Authoring coords are hyperspherical angles ``(phi_0, .., phi_{n-1})``;
    ``phi_0..phi_{n-2}`` are polar angles in ``[0, pi]`` and ``phi_{n-1}``
    is the azimuth in ``[0, 2*pi)``.  For ``S^2`` this is
    ``(colatitude, longitude)``.  The metric is chordal (Euclidean of the
    unit-vector embeddings) -- chosen over great-circle because chordal
    ``r**3`` stays conditionally positive definite in the ambient
    ``R^(n+1)``; a great-circle metric is a documented future option.
    """

    def __init__(self, dim: int):
        if dim < 1:
            raise ValueError(f"SphereDomain needs dim >= 1, got {dim}")
        self._dim = dim

    @property
    def intrinsic_dim(self) -> int:
        return self._dim

    @property
    def embed_dim(self) -> int:
        return self._dim + 1

    def embed(self, coords: torch.Tensor) -> torch.Tensor:
        sins = torch.sin(coords)
        coss = torch.cos(coords)
        running = torch.ones(
            coords.shape[:-1], dtype=coords.dtype, device=coords.device,
        )
        parts: list[torch.Tensor] = []
        for k in range(self._dim):
            parts.append(running * coss[..., k])
            running = running * sins[..., k]
        parts.append(running)
        return torch.stack(parts, dim=-1)

    def embed_jacobian(self, coords: torch.Tensor) -> torch.Tensor:
        n = self._dim
        sins = torch.sin(coords)
        coss = torch.cos(coords)
        J = torch.zeros(n + 1, n, dtype=coords.dtype, device=coords.device)
        one = torch.ones((), dtype=coords.dtype, device=coords.device)
        for k in range(n + 1):
            for l in range(n):
                if k < n:
                    if l > k:
                        continue
                    if l == k:
                        prefix = one
                        for i in range(k):
                            prefix = prefix * sins[i]
                        J[k, l] = -prefix * sins[k]
                    else:
                        term = one
                        for i in range(k):
                            term = term * (coss[l] if i == l else sins[i])
                        J[k, l] = term * coss[k]
                else:
                    term = one
                    for i in range(n):
                        term = term * (coss[l] if i == l else sins[i])
                    J[k, l] = term
        return J

    def clamp_position(self, coords: torch.Tensor) -> torch.Tensor:
        out = coords.clone()
        for i in range(self._dim - 1):
            out[..., i] = coords[..., i].clamp(min=0.0, max=math.pi)
        out[..., self._dim - 1] = torch.remainder(
            coords[..., self._dim - 1], 2.0 * math.pi
        )
        return out

    def sample_positions(self, resolution: int) -> torch.Tensor:
        axis_grids: list[torch.Tensor] = []
        for i in range(self._dim):
            if i < self._dim - 1:
                axis_grids.append(torch.linspace(0.0, math.pi, resolution))
            else:
                axis_grids.append(
                    torch.linspace(0.0, 2.0 * math.pi, resolution + 1)[:-1]
                )
        mesh = torch.meshgrid(*axis_grids, indexing="ij")
        return torch.stack([g.reshape(-1) for g in mesh], dim=-1)

    def to_spec(self) -> dict[str, Any]:
        return {"type": "sphere", "dim": self._dim}


class CustomDomain(ManifoldDomain):
    """An explicit immersion: authoring coords *are* the embedding coords.

    The escape hatch for topologies the structured domains do not cover
    -- Moebius strip, Klein bottle, ``RP^2`` -- reachable by authoring an
    explicit set of embedding coordinates per node.  The metric is
    chordal, so intrinsic (geodesic) distances are approximated; near a
    self-near-approaching seam this distorts the fit.  On a non-orientable
    immersion a :meth:`Manifold.tangent` frame cannot be combed
    consistently around the surface (the orientation flips).
    """

    def __init__(self, embed_dim: int, bounds: Sequence[Sequence[float]] | None = None):
        if embed_dim < 1:
            raise ValueError(f"CustomDomain needs embed_dim >= 1, got {embed_dim}")
        self._embed_dim = embed_dim
        self._bounds = (
            tuple((float(lo), float(hi)) for lo, hi in bounds)
            if bounds is not None
            else None
        )

    @property
    def intrinsic_dim(self) -> int:
        return self._embed_dim

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def embed(self, coords: torch.Tensor) -> torch.Tensor:
        return coords

    def embed_jacobian(self, coords: torch.Tensor) -> torch.Tensor:
        return torch.eye(
            self._embed_dim, dtype=coords.dtype, device=coords.device,
        )

    def clamp_position(self, coords: torch.Tensor) -> torch.Tensor:
        return coords

    def sample_positions(self, resolution: int) -> torch.Tensor:
        bounds = self._bounds or tuple((0.0, 1.0) for _ in range(self._embed_dim))
        axis_grids = [
            torch.linspace(lo, hi, resolution) for lo, hi in bounds
        ]
        mesh = torch.meshgrid(*axis_grids, indexing="ij")
        return torch.stack([g.reshape(-1) for g in mesh], dim=-1)

    def to_spec(self) -> dict[str, Any]:
        spec: dict[str, Any] = {"type": "custom", "embed_dim": self._embed_dim}
        if self._bounds is not None:
            spec["bounds"] = [[lo, hi] for lo, hi in self._bounds]
        return spec


def domain_from_spec(spec: dict[str, Any]) -> ManifoldDomain:
    """Build a :class:`ManifoldDomain` from a ``manifold.json`` domain object."""
    kind = spec.get("type")
    if kind == "box":
        axes = [
            BoxAxis(
                name=str(a.get("name", f"axis{i}")),
                periodic=bool(a.get("periodic", False)),
                period=float(a.get("period", 1.0)),
                lo=float(a.get("lo", 0.0)),
                hi=float(a.get("hi", 1.0)),
            )
            for i, a in enumerate(spec.get("axes", []))
        ]
        return BoxDomain(axes)
    if kind == "sphere":
        return SphereDomain(int(spec["dim"]))
    if kind == "custom":
        return CustomDomain(
            int(spec["embed_dim"]), bounds=spec.get("bounds"),
        )
    raise ValueError(f"unknown manifold domain type {kind!r}")


# ======================================================== RBF interpolant ===
#
# Per layer the interpolant is one r**3 polyharmonic RBF with an affine
# polynomial term.  The kernel phi(r) = r**3 is conditionally positive
# definite of order 2 in every dimension; with the affine term it gives a
# smooth (C^2) interpolant, and at n=1 over an open axis it is exactly
# the natural cubic spline.


def fit_rbf_interpolant(
    node_params: torch.Tensor,
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit an ``r**3`` polyharmonic RBF + affine polynomial.

    ``node_params`` is ``(K, m)`` -- the (normalized) embedded node
    coordinates -- and ``values`` is ``(K, R)`` -- the value to
    interpolate at each node.  Returns ``(rbf_weights (K, R),
    poly_coeffs (m+1, R))``.

    Solves the symmetric saddle system::

        [ A   Q ] [ w ]   [ values ]
        [ Q^T 0 ] [ c ] = [ 0      ]

    where ``A_ij = ||p_i - p_j||**3``, ``Q = [1 | node_params]``.  The
    matrix is symmetric *indefinite* (the zero block forces negative
    eigenvalues), so it is solved with ``torch.linalg.solve`` (LU) --
    never Cholesky.  It is nonsingular exactly when the node coordinates
    are *poised* for affine interpolation: their affine rank equals
    ``m``.  A rank-deficient set (all collinear in 2-D, coplanar in 3-D,
    ...) raises ``ValueError`` naming the poisedness failure.
    """
    node_params = node_params.to(torch.float32)
    values = values.to(torch.float32)
    K, m = node_params.shape
    R = values.shape[1]
    if K < m + 1:
        raise ValueError(
            f"RBF poisedness failure: {K} nodes cannot determine an affine "
            f"term in {m} embedding dimensions (need >= {m + 1})"
        )
    centered = node_params - node_params.mean(dim=0, keepdim=True)
    if int(torch.linalg.matrix_rank(centered)) != m:
        raise ValueError(
            f"RBF poisedness failure: the {K} node coordinates do not "
            f"affinely span the {m}-dim embedding space (they lie in a "
            f"lower-dimensional flat); spread the nodes across every axis"
        )

    dist = torch.cdist(node_params, node_params)
    A = dist.pow(3)
    ones = torch.ones(K, 1, dtype=torch.float32)
    Q = torch.cat([ones, node_params], dim=1)  # (K, m+1)
    top = torch.cat([A, Q], dim=1)                       # (K, K+m+1)
    bot = torch.cat(
        [Q.T, torch.zeros(m + 1, m + 1, dtype=torch.float32)], dim=1,
    )                                                    # (m+1, K+m+1)
    M = torch.cat([top, bot], dim=0)                     # (K+m+1, square)
    rhs = torch.cat(
        [values, torch.zeros(m + 1, R, dtype=torch.float32)], dim=0,
    )
    sol = torch.linalg.solve(M, rhs)
    return sol[:K].contiguous(), sol[K:].contiguous()


def eval_rbf(
    node_params: torch.Tensor,
    rbf_weights: torch.Tensor,
    poly_coeffs: torch.Tensor,
    query: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the RBF interpolant at ``query``.

    ``node_params`` ``(K, m)``, ``rbf_weights`` ``(K, R)``,
    ``poly_coeffs`` ``(m+1, R)``; ``query`` is ``(.., m)`` (already
    embedded and normalized).  Returns ``(.., R)``.

    Hot-path safe: no ``.item()``, no host sync.
    """
    diff = query.unsqueeze(-2) - node_params  # (.., K, m)
    r = torch.linalg.vector_norm(diff, dim=-1)  # (.., K)
    phi = r.pow(3)
    rbf_part = torch.matmul(phi, rbf_weights)  # (.., R)
    aug = torch.cat(
        [torch.ones_like(query[..., :1]), query], dim=-1,
    )  # (.., m+1)
    poly_part = torch.matmul(aug, poly_coeffs)  # (.., R)
    return rbf_part + poly_part


def eval_rbf_jacobian(
    node_params: torch.Tensor,
    rbf_weights: torch.Tensor,
    poly_coeffs: torch.Tensor,
    query: torch.Tensor,
) -> torch.Tensor:
    """Analytic Jacobian ``d s / d query`` at a single point.

    ``query`` is ``(m,)``; returns ``(R, m)``.  The kernel derivative is
    ``d/dx[r**3] = 3 r (x - p_j)``; the polynomial contributes its linear
    coefficients.  No autograd.
    """
    diff = query - node_params  # (K, m)
    r = torch.linalg.vector_norm(diff, dim=-1)  # (K,)
    grad_phi = 3.0 * r.unsqueeze(-1) * diff  # (K, m)
    j_rbf = rbf_weights.T @ grad_phi  # (R, m)
    j_poly = poly_coeffs[1:].T  # (R, m)
    return j_rbf + j_poly


# ============================================================== subspaces ===

@dataclass
class LayerSubspace:
    """Per-layer reduced frame and RBF interpolant for one manifold.

    ``mean`` and ``basis`` define an affine PCA subspace of the
    activation space; ``node_params`` / ``rbf_weights`` / ``poly_coeffs``
    define the ``r**3`` RBF interpolant from the manifold's embedded
    domain coordinates to that subspace's reduced coordinates.
    ``coord_offset`` / ``coord_scale`` carry the unit-box normalization
    applied to the embedded coordinates before the fit (the RBF kernel
    amplifies coordinate scale, so normalization is mandatory).
    """

    mean: torch.Tensor          # (D,)   centering mean over the node centroids
    basis: torch.Tensor         # (R, D) orthonormal PCA rows
    node_params: torch.Tensor   # (K, m) normalized embedded node coordinates
    rbf_weights: torch.Tensor   # (K, R) RBF weights
    poly_coeffs: torch.Tensor   # (m+1, R) affine polynomial coefficients
    coord_offset: torch.Tensor  # (m,)   unit-box normalization offset
    coord_scale: torch.Tensor   # (m,)   unit-box normalization scale

    @property
    def rank(self) -> int:
        return int(self.basis.shape[0])

    def to(self, *, device: torch.device, dtype: torch.dtype) -> "LayerSubspace":
        """Return a copy with every tensor on ``device`` in ``dtype``."""
        return LayerSubspace(
            mean=self.mean.to(device=device, dtype=dtype),
            basis=self.basis.to(device=device, dtype=dtype),
            node_params=self.node_params.to(device=device, dtype=dtype),
            rbf_weights=self.rbf_weights.to(device=device, dtype=dtype),
            poly_coeffs=self.poly_coeffs.to(device=device, dtype=dtype),
            coord_offset=self.coord_offset.to(device=device, dtype=dtype),
            coord_scale=self.coord_scale.to(device=device, dtype=dtype),
        )

    def _normalize(self, embedded: torch.Tensor) -> torch.Tensor:
        return (embedded - self.coord_offset) / self.coord_scale

    def eval_at(self, embedded: torch.Tensor) -> torch.Tensor:
        """World-space activation ``(.., D)`` at embedded domain coords ``(.., m)``."""
        reduced = eval_rbf(
            self.node_params, self.rbf_weights, self.poly_coeffs,
            self._normalize(embedded),
        )
        return reduced @ self.basis + self.mean

    def jacobian_at(self, embedded: torch.Tensor) -> torch.Tensor:
        """Activation Jacobian ``d activation / d embedded``: ``(m,) -> (D, m)``."""
        j_norm = eval_rbf_jacobian(
            self.node_params, self.rbf_weights, self.poly_coeffs,
            self._normalize(embedded),
        )  # (R, m) w.r.t. normalized coords
        j_embedded = j_norm / self.coord_scale  # chain through the normalization
        return self.basis.T @ j_embedded  # (D, m)


def fit_layer_subspace(
    centroids: torch.Tensor,
    node_params: torch.Tensor,
    *,
    n_components: int = DEFAULT_N_COMPONENTS,
) -> LayerSubspace:
    """Fit a PCA subspace + RBF interpolant for one layer.

    ``centroids`` is ``(K, D)`` -- one per-node mean activation -- and
    ``node_params`` is ``(K, m)`` -- the corresponding embedded domain
    coordinates (raw, un-normalized).  The activations are centered and
    reduced to ``R = min(n_components, K-1, rank)`` principal components;
    the embedded coordinates are normalized to the unit box; an ``r**3``
    RBF is fitted from the normalized coordinates to the reduced
    activations.
    """
    centroids = centroids.to(torch.float32)
    node_params = node_params.to(torch.float32)
    K = centroids.shape[0]
    if K < 3:
        raise ValueError(f"a manifold needs >= 3 nodes to fit, got {K}")
    mean = centroids.mean(dim=0)
    X = centroids - mean  # (K, D)
    _, S, Vh = torch.linalg.svd(X, full_matrices=False)
    rank = int((S > 1e-6 * S[0].clamp(min=1e-12)).sum().item())
    R = max(1, min(n_components, K - 1, rank))
    basis = Vh[:R].contiguous()  # (R, D)
    coords = X @ basis.T          # (K, R) -- the values the RBF interpolates

    lo = node_params.min(dim=0).values
    hi = node_params.max(dim=0).values
    coord_offset = lo
    coord_scale = (hi - lo).clamp(min=1e-9)
    normalized = (node_params - coord_offset) / coord_scale

    rbf_weights, poly_coeffs = fit_rbf_interpolant(normalized, coords)
    return LayerSubspace(
        mean=mean, basis=basis, node_params=normalized,
        rbf_weights=rbf_weights, poly_coeffs=poly_coeffs,
        coord_offset=coord_offset, coord_scale=coord_scale,
    )


# =============================================================== manifold ===

@dataclass
class Manifold:
    """A fitted manifold: a domain + per-layer subspaces, plus identity.

    The in-memory analogue of :class:`saklas.core.profile.Profile` for
    manifold steering.  Built by the manifold extraction pipeline and
    consumed by the session and the steering hooks.

    ``feature_space`` is ``"raw"`` for a manifold fitted directly on
    residual-stream activations, or ``"sae-<release>"`` when fitted in an
    SAE feature space (the stored :class:`LayerSubspace` values are then
    already decoded back to model space for runtime use).
    """

    name: str
    domain: ManifoldDomain
    node_labels: list[str]
    node_coords: torch.Tensor   # (K, n) authoring coordinates per node
    layers: dict[int, LayerSubspace]
    feature_space: str = "raw"
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def layer_indices(self) -> list[int]:
        return sorted(self.layers)

    def to(self, *, device: torch.device, dtype: torch.dtype) -> "Manifold":
        """Return a copy with every layer tensor on ``device`` in ``dtype``."""
        return Manifold(
            name=self.name,
            domain=self.domain,
            node_labels=list(self.node_labels),
            node_coords=self.node_coords.to(device=device, dtype=dtype),
            layers={
                idx: sub.to(device=device, dtype=dtype)
                for idx, sub in self.layers.items()
            },
            feature_space=self.feature_space,
            metadata=dict(self.metadata),
        )

    def _position_tensor(
        self,
        position: "float | Sequence[float] | torch.Tensor",
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Coerce ``float | Sequence[float] | Tensor`` to a ``(n,)`` tensor."""
        n = self.domain.intrinsic_dim
        if isinstance(position, torch.Tensor):
            pos = position.to(device=device, dtype=dtype).reshape(-1)
        else:
            if isinstance(position, (int, float)):
                position = (float(position),)
            pos = torch.tensor(
                [float(c) for c in position], dtype=dtype, device=device,
            )
        if pos.shape[0] != n:
            raise ValueError(
                f"manifold {self.name!r} has intrinsic dimension {n}; "
                f"position has {pos.shape[0]} coordinate(s)"
            )
        return pos

    def manifold_point(
        self, layer: int, position: "float | Sequence[float] | torch.Tensor",
    ) -> torch.Tensor:
        """World-space activation ``(D,)`` at authoring coords ``position``."""
        sub = self.layers[layer]
        pos = self._position_tensor(
            position, device=sub.mean.device, dtype=sub.mean.dtype,
        )
        embedded = self.domain.embed(self.domain.clamp_position(pos))
        return sub.eval_at(embedded)

    def tangent(
        self, layer: int, position: "float | Sequence[float] | torch.Tensor",
    ) -> torch.Tensor:
        """Per-axis steering directions ``(n, D)`` at authoring coords ``position``.

        Row ``k`` is ``d activation / d position_k`` -- the local
        direction in activation space along authoring axis ``k`` (e.g.
        the valence and arousal directions of an affect manifold).  The
        analytic RBF Jacobian chained through the domain's embedding
        Jacobian; no autograd.
        """
        sub = self.layers[layer]
        pos = self._position_tensor(
            position, device=sub.mean.device, dtype=sub.mean.dtype,
        )
        embedded = self.domain.embed(pos)
        j_act_embedded = sub.jacobian_at(embedded)        # (D, m)
        embed_jac = self.domain.embed_jacobian(pos)        # (m, n)
        j_act_authoring = j_act_embedded @ embed_jac       # (D, n)
        return j_act_authoring.T.contiguous()              # (n, D)


def subspace_replace(
    h: torch.Tensor,
    mean: torch.Tensor,
    basis: torch.Tensor,
    target: torch.Tensor,
    alpha: float | torch.Tensor,
) -> torch.Tensor:
    """Soft subspace-replace: blend ``h``'s in-subspace component to ``target``.

    ``h`` is ``(.., D)`` running activations; ``mean`` ``(D,)`` and
    ``basis`` ``(R, D)`` define the manifold's affine PCA subspace;
    ``target`` ``(D,)`` is the manifold point to steer toward; ``alpha``
    is the blend fraction.

    The activation decomposes as ``h = h_par + h_perp`` where ``h_par``
    is the reconstruction inside the subspace and ``h_perp`` the
    orthogonal residual.  Only ``h_par`` is moved -- toward ``target`` --
    and the residual is kept verbatim::

        h' = h + alpha * (target - h_par)

    so ``alpha=0`` is the identity and ``alpha=1`` snaps the in-subspace
    component fully onto the manifold.  The result is rescaled to ``h``'s
    original per-position norm, matching every other saklas injection
    hook.  This generalizes the ``~`` / ``|`` line-projection operators
    from a 1-D direction to an R-dimensional subspace.

    Returns a new tensor; the caller copies it in place.  fp32
    intermediates throughout (fp16 sum-of-squares overflows at large
    hidden dim).
    """
    h_f32 = h.to(torch.float32)
    mean_f32 = mean.to(torch.float32)
    basis_f32 = basis.to(torch.float32)
    target_f32 = target.to(torch.float32)

    centered = h_f32 - mean_f32
    coords = centered @ basis_f32.T          # (.., R)
    h_par = coords @ basis_f32 + mean_f32    # (.., D) in-subspace part
    delta = target_f32 - h_par               # (.., D) in-subspace correction
    h_new = h_f32 + alpha * delta

    norm_pre = torch.linalg.vector_norm(h_f32, dim=-1, keepdim=True)
    norm_post = torch.linalg.vector_norm(
        h_new, dim=-1, keepdim=True,
    ).clamp(min=1e-6)
    h_new = h_new * (norm_pre / norm_post)
    return h_new.to(h.dtype)


# ======================================================= coord discovery ===
#
# Auto-fitting: derive node coordinates from per-node activation
# centroids when the user hasn't authored a coordinate system.  PCA is
# the safe linear default (reproduces the flat-subspace layout a user
# would author themselves once they knew the answer).  Spectral
# (Laplacian eigenmaps) recovers curved-manifold topology that PCA
# flattens.  Both feed the same downstream machinery:
# ``CustomDomain(k)`` with identity embedding, then per-layer
# :func:`fit_layer_subspace` exactly as for authored manifolds.


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


def derive_pca_coords(
    centroids: torch.Tensor,
    *,
    max_dim: int = 8,
    var_threshold: float = 0.70,
) -> tuple[torch.Tensor, PcaDiagnostics]:
    """Derive node coordinates from centroid PCA.

    ``centroids`` is ``(K, D)`` — one per-node mean activation in the
    chosen reference layer.  Returns ``(coords, diagnostics)`` where
    ``coords`` is ``(K, k)`` and ``k`` is the smallest prefix whose
    cumulative variance crosses ``var_threshold``, capped at
    ``max_dim`` and floored at 1.

    Reproduces the flat-subspace layout a user would author themselves
    once they knew the answer.  Pure tensor, fp32, dependency-free.
    """
    centroids = centroids.to(torch.float32)
    K = centroids.shape[0]
    if K < 2:
        raise ValueError(
            f"PCA coord derivation needs >= 2 centroids, got {K}"
        )
    centered = centroids - centroids.mean(dim=0, keepdim=True)
    # full_matrices=False gives U: (K, min(K,D)), S: (min(K,D),),
    # Vh: (min(K,D), D).  We only need U @ diag(S) as the scores.
    U, S, _ = torch.linalg.svd(centered, full_matrices=False)
    # Variance fractions are (S^2)_i / sum_i (S^2)_i — metric-invariant
    # and unaffected by sample-size scaling, so no (K-1) divisor.
    var = S.pow(2)
    total = var.sum().clamp(min=1e-12)
    var_frac = var / total                       # (min(K,D),)
    cum_var = torch.cumsum(var_frac, dim=0)      # (min(K,D),)
    cap = min(max_dim, var_frac.shape[0])
    # Smallest k such that cum_var[k-1] >= threshold; default to cap.
    over = (cum_var[:cap] >= var_threshold).nonzero(as_tuple=False)
    picked_k = int(over[0].item()) + 1 if over.numel() > 0 else cap
    picked_k = max(1, min(picked_k, cap))

    coords = U[:, :picked_k] * S[:picked_k]      # (K, picked_k)
    diagnostics = PcaDiagnostics(
        per_component_variance=var_frac[:cap].detach().clone(),
        cumulative_variance=cum_var[:cap].detach().clone(),
        picked_k=picked_k,
        threshold=float(var_threshold),
    )
    return coords.contiguous(), diagnostics


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


def _connected_components(mask: torch.Tensor) -> int:
    """Number of connected components in an undirected adjacency mask.

    Plain union-find on a small ``(K, K)`` bool matrix — ``K`` is on
    the order of tens to hundreds, so a quadratic scan over the upper
    triangle is fine and avoids both scipy and the eigenvalue-counting
    tolerance choice.
    """
    K = mask.shape[0]
    parent = list(range(K))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Only need the upper triangle since the mask is symmetric.
    rows, cols = mask.triu(diagonal=1).nonzero(as_tuple=True)
    for r, c in zip(rows.tolist(), cols.tolist()):
        union(r, c)
    return len({find(i) for i in range(K)})


def derive_spectral_coords(
    centroids: torch.Tensor,
    *,
    max_dim: int = 8,
    k_nn: int | None = None,
    bandwidth: float | None = None,
) -> tuple[torch.Tensor, SpectralDiagnostics]:
    """Derive node coordinates from a Laplacian-eigenmaps spectral embedding.

    Build a symmetric k-NN graph over the ``(K, D)`` centroids,
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
    centroids = centroids.to(torch.float32)
    K, _ = centroids.shape
    if K < 4:
        # Need at least 4 nodes to form any kind of k-NN graph and have
        # a candidate gap range.  Below that the heuristics are pure
        # noise; raise early rather than ship a meaningless embedding.
        raise ValueError(
            f"spectral coord derivation needs >= 4 centroids, got {K}"
        )
    if k_nn is None:
        k_nn = max(5, math.ceil(math.log(K)))
    k_nn = max(1, min(k_nn, K - 1))

    distances = torch.cdist(centroids, centroids)
    mask, neighbor_dists = _knn_adjacency(distances, k_nn)

    components = _connected_components(mask)
    if components > 1:
        raise ValueError(
            f"spectral coord derivation: k-NN graph has {components} "
            f"connected components (need 1). Raise --k-nn or switch to "
            f"--method pca."
        )

    if bandwidth is None:
        if neighbor_dists.numel() == 0:
            raise ValueError(
                "spectral coord derivation: k-NN graph has no edges"
            )
        bandwidth = float(neighbor_dists.median().item())
        if bandwidth <= 0.0:
            # All-zero neighbor distances would NaN out the heat kernel.
            # Falls back to a small positive sentinel; the caller's data
            # is degenerate but we'd rather return something than crash.
            bandwidth = 1e-6
    bandwidth = float(bandwidth)

    # Heat-kernel weights on the symmetric k-NN edge set.
    W = torch.zeros_like(distances)
    sq = (distances * distances) / (2.0 * bandwidth * bandwidth)
    W = torch.where(mask, torch.exp(-sq), W)
    # Belt-and-braces: kill any residual self-loops.
    W.fill_diagonal_(0.0)

    deg = W.sum(dim=1)
    # _connected_components above guarantees no isolated vertex, so
    # deg > 0 everywhere; the clamp is purely defensive against fp32
    # underflow on huge bandwidths.
    d_inv_sqrt = deg.clamp(min=1e-12).rsqrt()
    # L_sym = I - D^{-1/2} W D^{-1/2}
    L = -W * d_inv_sqrt.unsqueeze(0) * d_inv_sqrt.unsqueeze(1)
    L.fill_diagonal_(0.0)
    L = L + torch.eye(K, dtype=L.dtype, device=L.device)

    eigvals, eigvecs = torch.linalg.eigh(L)  # ascending

    # Drop the smallest eigenvalue (~0 for a connected graph).  The
    # corresponding eigenvector is D^{1/2}1, not constant — it carries
    # no embedding information.
    nontrivial_vals = eigvals[1:]
    nontrivial_vecs = eigvecs[:, 1:]

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
        gap_magnitude = float(
            (nontrivial_vals[1] - nontrivial_vals[0]).item()
            if nontrivial_vals.shape[0] > 1
            else 0.0
        )
    else:
        # Ratio at "use k dims" = nontrivial[k] / nontrivial[k-1].
        # Clamp the denominator off zero — a vanishing eigenvalue at
        # ``k-1`` already means the graph is near-disconnected; the
        # ratio there is meaningless but mustn't NaN out the argmax.
        denom = nontrivial_vals[:cap].clamp(min=1e-12)
        ratios = nontrivial_vals[1:cap + 1] / denom
        picked_k = int(ratios.argmax().item()) + 1
        gap_magnitude = float(
            (nontrivial_vals[picked_k] - nontrivial_vals[picked_k - 1]).item()
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
    )
    return coords, diagnostics


def discover_coords(
    centroids: torch.Tensor,
    method: str,
    **kwargs: Any,
) -> tuple[torch.Tensor, PcaDiagnostics | SpectralDiagnostics]:
    """Dispatch to :func:`derive_pca_coords` or :func:`derive_spectral_coords`.

    Exists so the pipeline doesn't branch on method strings; downstream
    code calls ``discover_coords(centroids, method, **fit_kwargs)``
    once and the diagnostics ride into the sidecar through the same
    seam regardless of method.
    """
    if method == "pca":
        return derive_pca_coords(centroids, **kwargs)
    if method == "spectral":
        return derive_spectral_coords(centroids, **kwargs)
    raise ValueError(
        f"unknown discover method {method!r} (expected 'pca' | 'spectral')"
    )


# ------------------------------------------------------ centroid capture ---

def compute_node_centroid(
    model: object,
    tokenizer: object,
    layers: "torch.nn.ModuleList",
    device: torch.device,
    statements: list[str],
) -> dict[int, torch.Tensor]:
    """Mean per-layer pooled activation over a manifold node's statements.

    Reuses :func:`saklas.core.vectors._encode_and_capture_all` -- the same
    chat-templated, last-content-token, fp32 pooling discipline that
    backs :func:`saklas.core.vectors.compute_layer_means` -- and the same
    MPS ``empty_cache`` discipline between forward passes.

    Returns ``{layer_idx: centroid (D,)}`` in fp32 on CPU.
    """
    from saklas.core.vectors import _encode_and_capture_all

    if not statements:
        raise ValueError("manifold node has no statements")

    n_layers = len(layers)
    sums: dict[int, torch.Tensor] = {}
    is_mps = getattr(device, "type", None) == "mps"

    for text in statements:
        per_layer = _encode_and_capture_all(
            model, tokenizer, text, layers, device,
        )
        if not sums:
            for idx in range(n_layers):
                sums[idx] = per_layer[idx].detach().to("cpu", torch.float32)
        else:
            for idx in range(n_layers):
                sums[idx] += per_layer[idx].detach().to("cpu", torch.float32)
        del per_layer
        if is_mps:
            torch.mps.empty_cache()

    n = len(statements)
    return {idx: sums[idx] / n for idx in range(n_layers)}


# ------------------------------------------------------------- save/load ---

def save_manifold(
    manifold: Manifold, path: str | Path, metadata: dict[str, object],
) -> None:
    """Save a fitted manifold as ``.safetensors`` + a ``.json`` sidecar.

    The safetensors payload carries, per layer ``L``: ``layer_<L>.mean``,
    ``layer_<L>.basis``, ``layer_<L>.node_params``,
    ``layer_<L>.rbf_weights``, ``layer_<L>.poly_coeffs``,
    ``layer_<L>.coord_offset``, ``layer_<L>.coord_scale``; plus one
    shared bare ``node_coords`` tensor.  The sidecar carries the manifold
    identity (name, domain spec, ordered node labels, feature space) plus
    the provenance fields in ``metadata``.
    """
    from saklas.io.manifolds import MANIFOLD_FORMAT_VERSION

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tensors: dict[str, torch.Tensor] = {
        "node_coords": manifold.node_coords.contiguous().cpu(),
    }
    for idx, sub in manifold.layers.items():
        tensors[f"layer_{idx}.mean"] = sub.mean.contiguous().cpu()
        tensors[f"layer_{idx}.basis"] = sub.basis.contiguous().cpu()
        tensors[f"layer_{idx}.node_params"] = sub.node_params.contiguous().cpu()
        tensors[f"layer_{idx}.rbf_weights"] = sub.rbf_weights.contiguous().cpu()
        tensors[f"layer_{idx}.poly_coeffs"] = sub.poly_coeffs.contiguous().cpu()
        tensors[f"layer_{idx}.coord_offset"] = sub.coord_offset.contiguous().cpu()
        tensors[f"layer_{idx}.coord_scale"] = sub.coord_scale.contiguous().cpu()
    save_file(tensors, str(path))

    from saklas import __version__ as _saklas_version

    sidecar: dict[str, object] = {
        "format_version": MANIFOLD_FORMAT_VERSION,
        "method": metadata.get("method", "manifold_pca"),
        "saklas_version": _saklas_version,
        "name": manifold.name,
        "domain": manifold.domain.to_spec(),
        "node_labels": list(manifold.node_labels),
        "node_count": len(manifold.node_labels),
        "feature_space": manifold.feature_space,
    }
    for key in (
        "nodes_sha256", "sae_release", "sae_revision", "sae_ids_by_layer",
        # Discover-mode fields.  ``fit_mode`` discriminates authored vs
        # discover at read time; ``hyperparams`` records the knobs the
        # fitter was called with (max_dim / var_threshold / k_nn /
        # bandwidth) for reproducibility; ``diagnostics`` carries the
        # per-method PCA variance bars or spectral spectrum for the
        # CLI / webui inspector.  All absent for authored fits.
        "fit_mode", "hyperparams", "diagnostics",
    ):
        if key in metadata:
            sidecar[key] = metadata[key]

    from saklas.io.atomic import write_json_atomic

    write_json_atomic(path.with_suffix(".json"), sidecar)
    log.info("Saved manifold %r (%d layers) to %s",
             manifold.name, len(manifold.layers), path)


def load_manifold(path: str | Path) -> Manifold:
    """Load a fitted manifold and its sidecar metadata."""
    path = Path(path)
    tensors = load_file(str(path))
    with open(path.with_suffix(".json")) as f:
        sidecar = json.load(f)

    # The shared, layer-agnostic node_coords tensor — pop it before the
    # per-layer key split, which assumes ``layer_<idx>.<field>``.
    node_coords = tensors.pop("node_coords", None)

    by_layer: dict[int, dict[str, torch.Tensor]] = {}
    for key, tensor in tensors.items():
        head, field_name = key.rsplit(".", 1)
        idx = int(head.split("_", 1)[1])
        by_layer.setdefault(idx, {})[field_name] = tensor

    layers: dict[int, LayerSubspace] = {}
    for idx, parts in by_layer.items():
        layers[idx] = LayerSubspace(
            mean=parts["mean"],
            basis=parts["basis"],
            node_params=parts["node_params"],
            rbf_weights=parts["rbf_weights"],
            poly_coeffs=parts["poly_coeffs"],
            coord_offset=parts["coord_offset"],
            coord_scale=parts["coord_scale"],
        )

    domain = domain_from_spec(sidecar["domain"])
    if node_coords is None:
        node_coords = torch.zeros(0, domain.intrinsic_dim)

    return Manifold(
        name=sidecar.get("name", path.parent.name),
        domain=domain,
        node_labels=list(sidecar.get("node_labels", [])),
        node_coords=node_coords,
        layers=layers,
        feature_space=sidecar.get("feature_space", "raw"),
        metadata=sidecar,
    )


def invert_parameterization(
    subspace: LayerSubspace,
    domain: ManifoldDomain,
    query: torch.Tensor,
    *,
    resolution: int = DEFAULT_INVERSION_RESOLUTION,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Nearest-point projection of ``query`` onto the fitted manifold.

    Returns ``(positions, dist)``: ``positions`` ``(.., n)`` are the
    authoring coordinates whose interpolant value minimizes the Euclidean
    distance to each query in reduced-coordinate space, ``dist`` ``(..,)``
    is that distance.  ``query`` is ``(.., R)``.

    A dense grid scan over the domain.  The grid is ``resolution`` per
    axis for ``n <= 2``, coarsened for higher ``n`` (the grid cost is
    ``resolution**n``) -- the result is then *approximate* for ``n >= 3``.
    This is the paper's ``s^-1`` map; it is used only by the naturalness
    eval, never by the steering hot path, so a coarse landing only adds
    noise to the metric.
    """
    n = domain.intrinsic_dim
    if n <= 2:
        res = resolution
    elif n == 3:
        res = 24
    else:
        res = 8
        log.info(
            "invert_parameterization: n=%d domain, grid coarsened to %d "
            "per axis — naturalness metric is approximate", n, res,
        )

    cand = domain.sample_positions(res).to(
        device=query.device, dtype=query.dtype,
    )  # (G, n)
    embedded = domain.embed(cand)  # (G, m)
    normalized = subspace._normalize(embedded)
    vals = eval_rbf(
        subspace.node_params, subspace.rbf_weights, subspace.poly_coeffs,
        normalized,
    )  # (G, R)

    flat = query.reshape(-1, query.shape[-1])  # (N, R)
    d = torch.cdist(flat, vals)  # (N, G)
    best = d.argmin(dim=-1)  # (N,)
    pos = cand[best]  # (N, n)
    dist = d.gather(1, best.unsqueeze(1)).squeeze(1)  # (N,)
    return (
        pos.reshape(*query.shape[:-1], n),
        dist.reshape(query.shape[:-1]),
    )


# ================================================ behavior-space manifold ===
#
# The naturalness eval (Goodfire's paper, second half): fit a manifold
# over the model's *output* distributions, then measure how far a steered
# generation's behavioral trajectory strays off it.  Output distributions
# are mapped to Hellinger space via ``p -> sqrt(p)`` -- there the ordinary
# Euclidean distance is the Hellinger distance, which linearizes the
# probability simplex so the same RBF machinery applies.

def to_hellinger(p: torch.Tensor) -> torch.Tensor:
    """Map a probability distribution to Hellinger space: ``p -> sqrt(p)``.

    In Hellinger space the L2 distance between two mapped distributions
    is their Hellinger distance, and the inner product is the
    Bhattacharyya coefficient -- which linearizes the simplex enough for
    a Euclidean PCA + RBF fit.
    """
    return p.clamp(min=0.0).sqrt()


def bhattacharyya_distance(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Bhattacharyya distance ``-ln sum sqrt(p*q)`` between distributions.

    ``p`` and ``q`` are probability distributions over the last axis.
    Returns the distance over any leading batch dims.
    """
    bc = (p.clamp(min=0.0) * q.clamp(min=0.0)).sqrt().sum(dim=-1)
    return -torch.log(bc.clamp(min=1e-12))


def fit_behavior_manifold(
    centroid_dists: torch.Tensor,
    node_params: torch.Tensor,
    *,
    n_components: int = DEFAULT_N_COMPONENTS,
) -> LayerSubspace:
    """Fit an RBF interpolant through per-node output-distribution centroids.

    ``centroid_dists`` is ``(K, V)`` -- one mean next-token distribution
    per manifold node -- and ``node_params`` is ``(K, m)`` the embedded
    domain coordinates.  The distributions are mapped to Hellinger space
    (``sqrt``) and the same :func:`fit_layer_subspace` PCA + RBF fit is
    applied there.  Returns the fitted :class:`LayerSubspace` (the
    behavior manifold).
    """
    return fit_layer_subspace(
        to_hellinger(centroid_dists), node_params, n_components=n_components,
    )


def trajectory_naturalness(
    traj_dists: torch.Tensor,
    behavior: LayerSubspace,
    domain: ManifoldDomain,
) -> torch.Tensor:
    """Per-step Bhattacharyya distance from a trajectory to a behavior manifold.

    ``traj_dists`` is ``(T, V)`` -- the sequence of next-token
    distributions a generation produced.  ``behavior`` is a behavior
    manifold from :func:`fit_behavior_manifold`, ``domain`` its
    :class:`ManifoldDomain`.  For each step the nearest point on the
    behavior manifold is found (in Hellinger space) and the Bhattacharyya
    distance to it is returned -- low means the step sits on the natural
    behavior manifold, high flags an off-manifold "teleportation"
    artifact.

    Returns a ``(T,)`` tensor of per-step distances.
    """
    h = to_hellinger(traj_dists)  # (T, V)
    coords = (h - behavior.mean) @ behavior.basis.T  # (T, R)
    pos, _ = invert_parameterization(behavior, domain, coords)
    embedded = domain.embed(pos)  # (T, m)
    curve_coords = eval_rbf(
        behavior.node_params, behavior.rbf_weights, behavior.poly_coeffs,
        behavior._normalize(embedded),
    )  # (T, R)
    curve_h = curve_coords @ behavior.basis + behavior.mean  # (T, V)
    # In Hellinger space ||h_a - h_b||^2 = 2 - 2*BC, so the Bhattacharyya
    # coefficient is BC = 1 - d^2/2 and the distance is -ln(BC).
    d2 = ((h - curve_h) ** 2).sum(dim=-1)
    bc = (1.0 - d2 / 2.0).clamp(min=1e-12)
    return -torch.log(bc)


def compute_node_behavior_centroid(
    model: object,
    tokenizer: object,
    device: torch.device,
    statements: list[str],
) -> torch.Tensor:
    """Mean next-token probability distribution over a node's statements.

    The behavior-space analogue of :func:`compute_node_centroid`: each
    statement is run through the model and the softmax over the
    final-position logits is taken; the per-statement distributions are
    averaged.  Returns a ``(V,)`` distribution in fp32 on CPU.
    """
    if not statements:
        raise ValueError("manifold node has no statements")
    is_mps = getattr(device, "type", None) == "mps"
    acc: torch.Tensor | None = None
    for text in statements:
        dist = _next_token_distribution(model, tokenizer, text, device)
        acc = dist if acc is None else acc + dist
        if is_mps:
            torch.mps.empty_cache()
    assert acc is not None
    return acc / len(statements)


def compute_trajectory_distributions(
    model: object,
    tokenizer: object,
    device: torch.device,
    text: str,
) -> torch.Tensor:
    """Per-position next-token distributions for a generated ``text``.

    One forward pass over the full token sequence; returns ``(T, V)`` --
    the behavioral trajectory the eval scores against the behavior
    manifold.  Kept off the generation hot path: the steered text is
    produced first, then re-run once here.
    """
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)  # type: ignore[operator]
    ids = enc["input_ids"].to(device)
    if ids.shape[1] == 0:
        raise ValueError("trajectory text tokenized to zero tokens")
    with torch.inference_mode():
        logits = model(input_ids=ids, use_cache=False).logits  # type: ignore[operator]
    return torch.softmax(logits[0].float(), dim=-1).detach().to("cpu")


def _next_token_distribution(
    model: object, tokenizer: object, text: str, device: torch.device,
) -> torch.Tensor:
    """Softmax over the final-position logits for ``text`` -- ``(V,)`` fp32 CPU."""
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)  # type: ignore[operator]
    ids = enc["input_ids"]
    if ids.numel() == 0:
        bos = getattr(tokenizer, "bos_token_id", None) or 0
        ids = torch.tensor([[bos]])
    ids = ids.to(device)
    with torch.inference_mode():
        logits = model(input_ids=ids, use_cache=False).logits  # type: ignore[operator]
    return torch.softmax(logits[0, -1].float(), dim=-1).detach().to("cpu")


__all__ = [
    "DEFAULT_N_COMPONENTS",
    "DEFAULT_INVERSION_RESOLUTION",
    "ManifoldDomain",
    "BoxDomain",
    "SphereDomain",
    "CustomDomain",
    "domain_from_spec",
    "LayerSubspace",
    "Manifold",
    "fit_rbf_interpolant",
    "eval_rbf",
    "eval_rbf_jacobian",
    "fit_layer_subspace",
    "subspace_replace",
    "PcaDiagnostics",
    "SpectralDiagnostics",
    "derive_pca_coords",
    "derive_spectral_coords",
    "discover_coords",
    "compute_node_centroid",
    "save_manifold",
    "load_manifold",
    "invert_parameterization",
    "to_hellinger",
    "bhattacharyya_distance",
    "fit_behavior_manifold",
    "trajectory_naturalness",
    "compute_node_behavior_centroid",
    "compute_trajectory_distributions",
]
