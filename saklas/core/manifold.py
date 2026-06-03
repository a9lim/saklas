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
pulled in. ``eval_rbf``, :func:`eval_rbf_jacobian`, :func:`_gn_step` and
:func:`subspace_inject` are the functions reachable from the generation
hot path (``subspace_inject`` is the unified along/onto subspace/manifold
injection — the single steering backend); all are allocation-light and
free of host syncs.
"""
from __future__ import annotations

import json
import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import torch
from safetensors.torch import load_file, save_file

from saklas.core.errors import SaklasError

if TYPE_CHECKING:
    from saklas.core.mahalanobis import LayerWhitener

log = logging.getLogger(__name__)


class UnknownManifoldLabelError(KeyError, SaklasError):
    """Raised when a manifold position payload names an unknown node label.

    Produced by :meth:`Manifold.resolve_position` (and the nearest-node
    helpers, which short-circuit on labels) when the label is not in
    :attr:`Manifold.node_labels`.  Surfaces a 404-shaped error at the
    HTTP layer through the shared :class:`SaklasError` MRO; CLI / TUI
    handlers print the message and recover.
    """

    def user_message(self) -> tuple[int, str]:
        return (404, str(self))


# Default PCA width for a fitted manifold subspace.  Matches the paper's
# 64-dimensional reduction.  Clamped down to ``min(64, K-1, rank)`` when
# the node count ``K`` is small -- ``K`` centered centroids span a space
# of rank at most ``K-1``.
DEFAULT_N_COMPONENTS = 64

# Levenberg-Marquardt settings for the inverse parameterization
# (nearest-point projection of an activation onto the fitted manifold).
# Used only by the naturalness eval and the read-side ``ManifoldMonitor``
# aggregate -- never the steering hot path, which steers to a fixed
# position.  The solve is warm-started from the nearest fit node(s) and
# Marquardt-damped, so a fixed dozen iterations converges in
# authoring-coord space *independent of intrinsic dimension*.  This
# replaces a grid scan that was O(resolution**n): even the n=2 path ran
# 512**2 = 262k RBF evals per layer, and the budget-capped high-n path
# degraded to a 4-point-per-axis landing on the bundled 8-D ``personas``.
DEFAULT_INVERSION_MAX_ITER = 12
DEFAULT_INVERSION_RESTARTS = 3
DEFAULT_INVERSION_DAMPING = 1e-3
# Absolute floor added to the LM normal-equation diagonal so a locally
# rank-deficient Jacobian (a fold/pinch, or a flat authoring direction)
# still yields a solvable, well-conditioned system.
_INVERSION_DIAG_FLOOR = 1e-9


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
    coords), :meth:`embed_jacobian` and :meth:`clamp_position`.
    :meth:`distance` defaults to the chordal (Euclidean-in-embedding)
    metric and is rarely overridden -- a periodic axis embedded as a
    circle already wraps correctly under the chordal metric.
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
        """Jacobian ``d embed / d coords``: ``(.., n) -> (.., m, n)``.

        Batch-generic over leading dims; the bare ``(n,)`` point returns
        ``(m, n)`` for :meth:`Manifold.tangent`.
        """

    @abstractmethod
    def clamp_position(self, coords: torch.Tensor) -> torch.Tensor:
        """Clamp open axes to range, wrap periodic axes; ``(.., n) -> (.., n)``."""

    @abstractmethod
    def to_spec(self) -> dict[str, Any]:
        """Serialize to the ``manifold.json`` tagged-union ``domain`` object."""

    def distance(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Chordal distance between embedded points ``(.., m)`` -> ``(..,)``."""
        return torch.linalg.vector_norm(a - b, dim=-1)

    def geodesic(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        frac: float | torch.Tensor,
    ) -> torch.Tensor:
        """Point ``frac`` of the way along the geodesic ``a -> b`` in *authoring*
        coords. ``a``/``b`` are ``(.., n)``; ``frac`` is a scalar or ``(.., 1)``;
        returns ``(.., n)`` clamped to the domain.

        The default is the straight coordinate lerp -- correct for any flat,
        non-wrapping domain (``CustomDomain``, and the open-axis part of a
        ``BoxDomain``). :class:`BoxDomain` overrides it for periodic axes
        (wrap-aware shortest arc) and :class:`SphereDomain` for great-circle
        slerp. This is the operator the two-op ``along`` step slides the
        projected foot through, so the path stays *on the surface* rather than
        cutting the ambient chord the old additive injection took.
        """
        return self.clamp_position(a + frac * (b - a))


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
        batch = coords.shape[:-1]
        J = torch.zeros(*batch, m, n, dtype=coords.dtype, device=coords.device)
        out = 0
        for i, ax in enumerate(self._axes):
            if ax.periodic:
                w = 2.0 * math.pi / ax.period
                ci = coords[..., i]
                J[..., out, i] = -w * torch.sin(w * ci)
                J[..., out + 1, i] = w * torch.cos(w * ci)
                out += 2
            else:
                J[..., out, i] = 1.0
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

    def geodesic(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        frac: float | torch.Tensor,
    ) -> torch.Tensor:
        """Per-axis lerp; periodic axes take the wrap-aware shortest arc.

        Open axes lerp straight (then clamp to range). Periodic axes route the
        delta through the signed minimal representative
        ``remainder(Δ + period/2, period) - period/2`` so the slide crosses the
        seam when that is the shorter path around the circle, then ``clamp``
        wraps the result back into ``[0, period)``.
        """
        delta = b - a
        for i, ax in enumerate(self._axes):
            if ax.periodic:
                half = ax.period / 2.0
                delta[..., i] = (
                    torch.remainder(delta[..., i] + half, ax.period) - half
                )
        return self.clamp_position(a + frac * delta)

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
        batch = coords.shape[:-1]
        sins = torch.sin(coords)  # (.., n)
        coss = torch.cos(coords)  # (.., n)
        J = torch.zeros(
            *batch, n + 1, n, dtype=coords.dtype, device=coords.device,
        )
        one = torch.ones(batch, dtype=coords.dtype, device=coords.device)
        for k in range(n + 1):
            for l in range(n):
                if k < n:
                    if l > k:
                        continue
                    if l == k:
                        prefix = one
                        for i in range(k):
                            prefix = prefix * sins[..., i]
                        J[..., k, l] = -prefix * sins[..., k]
                    else:
                        term = one
                        for i in range(k):
                            term = term * (
                                coss[..., l] if i == l else sins[..., i]
                            )
                        J[..., k, l] = term * coss[..., k]
                else:
                    term = one
                    for i in range(n):
                        term = term * (
                            coss[..., l] if i == l else sins[..., i]
                        )
                    J[..., k, l] = term
        return J

    def clamp_position(self, coords: torch.Tensor) -> torch.Tensor:
        out = coords.clone()
        for i in range(self._dim - 1):
            out[..., i] = coords[..., i].clamp(min=0.0, max=math.pi)
        out[..., self._dim - 1] = torch.remainder(
            coords[..., self._dim - 1], 2.0 * math.pi
        )
        return out

    def _unembed(self, e: torch.Tensor) -> torch.Tensor:
        """Recover hyperspherical angles ``(.., n)`` from a unit vector ``(.., n+1)``.

        Inverse of :meth:`embed`. The polar angles ``phi_0..phi_{n-2}`` come from
        ``atan2(||tail||, e_k) in [0, pi]``; the azimuth ``phi_{n-1}`` uses the
        signed ``atan2(e_n, e_{n-1})`` so the full circle is recovered (the
        polar formula would fold it into ``[0, pi]``). ``clamp_position`` wraps
        the azimuth into ``[0, 2*pi)``.
        """
        n = self._dim
        angles: list[torch.Tensor] = []
        for k in range(n):
            if k < n - 1:
                tail = torch.linalg.vector_norm(e[..., k + 1:], dim=-1)
                angles.append(torch.atan2(tail, e[..., k]))
            else:
                angles.append(torch.atan2(e[..., n], e[..., n - 1]))
        return torch.stack(angles, dim=-1)

    def geodesic(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        frac: float | torch.Tensor,
    ) -> torch.Tensor:
        """Great-circle slerp between two angle tuples, via the unit-vector embed.

        Embeds both endpoints to unit vectors, spherical-linear-interpolates in
        the ambient ``R^(n+1)`` (with a linear fallback when the two points
        nearly coincide, ``sin(omega) -> 0``), renormalizes, and unembeds back
        to angles. The result is the on-sphere path, not the chord through the
        ball.
        """
        ea = self.embed(a)
        eb = self.embed(b)
        dot = (ea * eb).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
        omega = torch.arccos(dot)                       # (.., 1)
        sin_omega = torch.sin(omega)
        small = sin_omega.abs() < 1e-6
        denom = sin_omega.clamp_min(1e-9)
        w_a = torch.sin((1.0 - frac) * omega) / denom
        w_b = torch.sin(frac * omega) / denom
        e = w_a * ea + w_b * eb
        e_lin = (1.0 - frac) * ea + frac * eb           # omega -> 0 fallback
        e = torch.where(small, e_lin, e)
        e = e / torch.linalg.vector_norm(e, dim=-1, keepdim=True).clamp_min(1e-9)
        return self.clamp_position(self._unembed(e))

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
        eye = torch.eye(
            self._embed_dim, dtype=coords.dtype, device=coords.device,
        )
        batch = coords.shape[:-1]
        if batch:
            eye = eye.expand(*batch, self._embed_dim, self._embed_dim)
        return eye

    def clamp_position(self, coords: torch.Tensor) -> torch.Tensor:
        return coords

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
    if m + 1 > K:
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
    """Analytic Jacobian ``d s / d query``.

    ``query`` is ``(.., m)`` (a single ``(m,)`` point or any batch of
    them); returns ``(.., R, m)``.  The kernel derivative is
    ``d/dx[r**3] = 3 r (x - p_j)``; the polynomial contributes its linear
    coefficients.  No autograd.  Batch-generic so the inverse
    parameterization can Jacobian a whole ``(N, S)`` fan of LM iterates in
    one call; the bare ``(m,)`` path still returns ``(R, m)`` for
    :meth:`LayerSubspace.jacobian_at`.
    """
    diff = query.unsqueeze(-2) - node_params  # (.., K, m)
    r = torch.linalg.vector_norm(diff, dim=-1)  # (.., K)
    grad_phi = 3.0 * r.unsqueeze(-1) * diff  # (.., K, m)
    j_rbf = torch.einsum("kr,...km->...rm", rbf_weights, grad_phi)  # (.., R, m)
    j_poly = poly_coeffs[1:].T  # (R, m) -- broadcasts over leading dims
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

    **Affine (flat) case.**  When ``node_params`` / ``rbf_weights`` /
    ``poly_coeffs`` are ``None`` the subspace carries no RBF surface — the
    "surface" *is* the whole affine subspace (a folded steering vector at
    ``n = R``), so the reduced coordinates equal the authoring coordinates
    by an identity map and ``manifold_point(c) = mean + c @ basis``.  The
    affine case has ``H_n ≡ 0`` (the surface fills its subspace), so
    ``subspace_inject`` takes an analytic shortcut that skips the
    Gauss-Newton foot solve, the RBF eval, and the tangent Gram-solve —
    load-bearing for throughput, since a folded vector is the common
    steering case and the curved per-token solve would blow the
    throughput invariant.  Build one via :meth:`affine`; query via
    :attr:`is_affine`.
    """

    mean: torch.Tensor                  # (D,)   centering mean over the node centroids
    basis: torch.Tensor                 # (R, D) orthonormal PCA rows
    node_params: torch.Tensor | None    # (K, m) normalized embedded coords; None = affine
    rbf_weights: torch.Tensor | None    # (K, R) RBF weights; None = affine
    poly_coeffs: torch.Tensor | None    # (m+1, R) affine polynomial coeffs; None = affine
    coord_offset: torch.Tensor          # (m,)   unit-box normalization offset
    coord_scale: torch.Tensor           # (m,)   unit-box normalization scale
    node_coords: torch.Tensor | None = None
    # (K, R) per-layer **real, neutral-anchored** reduced node coordinates
    # ``(c_i − ν*)·basisᵀ`` — affine (flat) subspaces only; ``None`` on a
    # curved subspace (which carries the shared ``Manifold.node_coords`` +
    # RBF instead).  This is the per-layer steer target source: a flat
    # subspace's pole / node sits at distance ∝ ‖δ_L‖ from the neutral
    # origin here, so the synthesizer reads ``node_coords[index]`` per layer
    # as the ``along`` target.  The shared ``Manifold.node_coords`` stays the
    # label/display layout; *these* are the geometry (§5 neutral-anchor).

    @property
    def rank(self) -> int:
        return int(self.basis.shape[0])

    @property
    def is_affine(self) -> bool:
        """True for a flat (no-RBF) subspace — the surface fills its span.

        The analytic-shortcut marker: ``subspace_inject`` and ``eval_at``
        branch on this to skip all RBF / Gauss-Newton machinery.  A folded
        steering vector (``n = R``, the chord) is built affine; every
        RBF-fitted manifold (curved, or merely space-filling like a discover
        fit at ``R = n``) is not.
        """
        return self.node_params is None

    @classmethod
    def affine(
        cls,
        mean: torch.Tensor,
        basis: torch.Tensor,
        *,
        node_coords: torch.Tensor | None = None,
    ) -> "LayerSubspace":
        """Build a flat (affine, no-RBF) subspace from ``mean`` + ``basis``.

        The authoring coordinates map to reduced coordinates by identity
        (``coord_offset = 0``, ``coord_scale = 1``), so
        ``eval_at(c) = c @ basis + mean`` exactly.  ``basis`` is ``(R, D)``;
        the implied intrinsic dimension is ``n = R`` (the surface fills the
        span).  Backs the folded-vector / flat-subspace representation
        (Phase 2 §1).  ``node_coords`` ``(K, R)`` carries the per-layer real
        neutral-anchored node positions (the steer-target source — §5);
        ``None`` for a bare span with no associated nodes.
        """
        r = int(basis.shape[0])
        ref = mean
        return cls(
            mean=mean,
            basis=basis,
            node_params=None,
            rbf_weights=None,
            poly_coeffs=None,
            coord_offset=torch.zeros(r, device=ref.device, dtype=ref.dtype),
            coord_scale=torch.ones(r, device=ref.device, dtype=ref.dtype),
            node_coords=node_coords,
        )

    def select_axes(self, kept: Sequence[int]) -> "LayerSubspace":
        """Restrict an affine subspace to a subset of its basis rows (DLS prune).

        Slices ``basis`` + ``node_coords`` to the ``kept`` axis indices and
        recomputes ``mean`` as the projection of the *same* anchor into the
        reduced span (``mean' = P_{basis[kept]}(anchor)``) — recovered from the
        stored ``mean`` without the raw anchor, since the anchor's per-axis
        coords are ``mean @ basisᵀ`` (the basis is orthonormal).  Affine only
        — a curved subspace's per-axis pruning would force an RBF re-fit, so
        the split is flat-yes / curved-no (§5).  ``kept`` indexes into
        ``[0, rank)``; an empty ``kept`` is a caller error (drop the layer).

        Bakes the DLS keep set into the stored basis at fit time (the N-node
        analogue of the folded vector dropping a non-discriminative *layer*),
        so the steer/apply path needs no separate per-axis mask.
        """
        if not self.is_affine:
            raise ValueError(
                "select_axes is affine-only (curved has no per-axis DLS)"
            )
        idx = list(kept)
        if not idx:
            raise ValueError(
                "select_axes: empty kept set — drop the layer instead"
            )
        basis_k = self.basis[idx].contiguous()                 # (|kept|, D)
        anchor_coords = self.mean @ self.basis.T               # (R,) anchor coords
        mean_k = anchor_coords[idx] @ basis_k                  # P_{basis[kept]}(anchor)
        nc_k = (
            self.node_coords[:, idx].contiguous()
            if self.node_coords is not None else None
        )
        return LayerSubspace.affine(mean_k, basis_k, node_coords=nc_k)

    def rbf_params(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The validated ``(node_params, rbf_weights, poly_coeffs)`` triple.

        Raises on an affine (flat) subspace — every RBF call site operates on
        the curved path, and the affine path has analytic equivalents that
        must never reach the interpolant.  Doubles as a guardrail: a stray
        affine subspace routed through the RBF machinery raises loudly
        instead of silently dereferencing ``None``.
        """
        np_, rw, pc = self.node_params, self.rbf_weights, self.poly_coeffs
        if np_ is None or rw is None or pc is None:
            raise ValueError(
                "LayerSubspace.rbf_params() called on an affine (flat) "
                "subspace; the affine path has analytic equivalents and must "
                "not reach the RBF interpolant"
            )
        return np_, rw, pc

    def to(self, *, device: torch.device, dtype: torch.dtype) -> "LayerSubspace":
        """Return a copy with every tensor on ``device`` in ``dtype``."""
        def _cast(t: torch.Tensor | None) -> torch.Tensor | None:
            return None if t is None else t.to(device=device, dtype=dtype)
        return LayerSubspace(
            mean=self.mean.to(device=device, dtype=dtype),
            basis=self.basis.to(device=device, dtype=dtype),
            node_params=_cast(self.node_params),
            rbf_weights=_cast(self.rbf_weights),
            poly_coeffs=_cast(self.poly_coeffs),
            coord_offset=self.coord_offset.to(device=device, dtype=dtype),
            coord_scale=self.coord_scale.to(device=device, dtype=dtype),
            node_coords=_cast(self.node_coords),
        )

    def _normalize(self, embedded: torch.Tensor) -> torch.Tensor:
        return (embedded - self.coord_offset) / self.coord_scale

    def eval_at(self, embedded: torch.Tensor) -> torch.Tensor:
        """World-space activation ``(.., D)`` at embedded domain coords ``(.., m)``."""
        if self.is_affine:
            # Flat: reduced coords == authoring coords (identity map), the
            # surface fills the subspace ⇒ pure affine, no RBF eval.
            return embedded @ self.basis + self.mean
        reduced = eval_rbf(*self.rbf_params(), self._normalize(embedded))
        return reduced @ self.basis + self.mean

    def jacobian_at(self, embedded: torch.Tensor) -> torch.Tensor:
        """Activation Jacobian ``d activation / d embedded``: ``(m,) -> (D, m)``."""
        if self.is_affine:
            # d(embedded @ basis + mean)/d embedded = basis.T, position-
            # independent; broadcast across any leading batch dims.
            jac = self.basis.transpose(-1, -2)  # (D, R) == (D, m)
            if embedded.ndim > 1:
                jac = jac.expand(*embedded.shape[:-1], *jac.shape)
            return jac
        j_norm = eval_rbf_jacobian(
            *self.rbf_params(), self._normalize(embedded),
        )  # (R, m) w.r.t. normalized coords
        j_embedded = j_norm / self.coord_scale  # chain through the normalization
        return self.basis.T @ j_embedded  # (D, m)


def _pca_basis(
    X: torch.Tensor,
    *,
    n_components: int = DEFAULT_N_COMPONENTS,
    whitener: "LayerWhitener | None" = None,
    layer: int | None = None,
) -> tuple[torch.Tensor, float]:
    """μ-centered PCA basis — Euclidean (default) or whitened/Fisher.

    ``X`` is the ``(K, D)`` **μ-centered** centroid scatter (the caller has
    already subtracted the centroid mean — never the neutral anchor, per the
    §5 basis caveat: anchor-centering the scatter injects the neutral offset
    as a spurious axis and breaks PCA@2 ≡ DiM).  Returns ``(basis,
    ev_ratio)``: ``basis`` is ``(R, D)`` orthonormal rows,
    ``R = min(n_components, K-1, rank)``.

    With ``whitener``/``layer`` set (the caller having gated all-or-nothing
    on ``covers_all``) the basis is the whitened/Fisher discriminant — the
    generalized eigenproblem ``(S_b, Σ)`` via the low-rank Woodbury ``Σ⁻¹``
    (``G = X Σ⁻¹ Xᵀ``, eigvecs ``a``, directions ``Σ⁻¹ Xᵀ a``),
    re-expressed in a Euclidean-orthonormal basis via QR (span-preserving) so
    the steering hot path is untouched; ``ev_ratio`` is the retained fraction
    of whitened between-variance.  Otherwise ordinary SVD of ``X`` and the
    raw inter-node variance ratio.  Shared by :func:`fit_layer_subspace`
    (curved RBF) and :func:`fit_affine_subspace` (flat) so both pick the
    basis identically — only what's built on top differs.
    """
    K = int(X.shape[0])
    if whitener is not None and layer is not None:
        # Whitened (Fisher) PCA — generalized eigenproblem (S_b, Σ) via the
        # whitener's low-rank Woodbury Σ⁻¹.  ``G = X Σ⁻¹ Xᵀ`` is K×K; its
        # eigvecs ``a`` give the discriminant directions ``v_r = Σ⁻¹ Xᵀ a_r``.
        G = whitener.subspace_gram(layer, X)            # (K, K) = X Σ⁻¹ Xᵀ
        mu, A = torch.linalg.eigh(G)                    # ascending
        mu_pos = mu.clamp_min(0.0)
        rank = int((mu_pos > 1e-6 * mu_pos[-1].clamp(min=1e-12)).sum().item())
        R = max(1, min(n_components, K - 1, rank))
        top = torch.argsort(mu, descending=True)[:R]
        XtA = X.transpose(0, 1) @ A[:, top]             # (D, R) = Xᵀ a_r
        directions = torch.stack(
            [whitener.apply_inv(layer, XtA[:, j]) for j in range(R)]
        )                                               # (R, D) = Σ⁻¹ Xᵀ a_r
        # QR → orthonormal column span identical to the discriminant span;
        # transpose back to (R, D) rows the LayerSubspace expects.
        basis = torch.linalg.qr(
            directions.transpose(0, 1)
        ).Q.transpose(0, 1).contiguous()                # (R, D)
        total_w = float(mu_pos.sum().item())
        retained_w = float(mu_pos[top].sum().item())
        ev_ratio = retained_w / total_w if total_w > 1e-12 else 1.0
    else:
        # Euclidean PCA — ordinary SVD of the centered centroids (no whitener
        # wired, or partial layer coverage at the call site).
        _, S, Vh = torch.linalg.svd(X, full_matrices=False)
        rank = int((1e-6 * S[0].clamp(min=1e-12) < S).sum().item())
        R = max(1, min(n_components, K - 1, rank))
        basis = Vh[:R].contiguous()                     # (R, D)
        # Per-layer EV ratio.  Falls back to 1.0 on a degenerate (all-zero
        # singular value) layer rather than NaN.
        total_var = float(S.pow(2).sum().item())
        retained_var = float(S[:R].pow(2).sum().item())
        ev_ratio = retained_var / total_var if total_var > 1e-12 else 1.0
    return basis, ev_ratio


def fit_affine_subspace(
    centroids: torch.Tensor,
    *,
    neutral_mean: torch.Tensor | None = None,
    n_components: int = DEFAULT_N_COMPONENTS,
    whitener: "LayerWhitener | None" = None,
    layer: int | None = None,
    orient_to: int | None = 0,
) -> tuple[LayerSubspace, torch.Tensor, float]:
    """Fit a flat (affine, no-RBF) subspace from per-node centroids (§5).

    The flat half of the unified fit: ``fit_mode=pca`` produces these, and a
    steering vector is the ``K = 2`` case.  Derives the per-layer basis by
    **μ-centered** PCA (Euclidean default, whitened/Fisher when the whitener
    covers ``layer``), then **neutral-anchors** the frame:

    - ``anchor = neutral_mean`` if given, else the centroid mean ``μ`` (the
      degenerate fallback when no neutral baseline is available — CPU stubs).
    - ``mean = P_basis(anchor) = (anchor·basisᵀ)·basis`` — the anchor's
      projection *into the span*, dropping its off-span component (§5: keeps
      the residual / read-side fraction clean; the dropped part provably
      cancels in the steered output anyway).
    - ``node_coords = (centroids − anchor)·basisᵀ`` ``(K, R)`` — **real**,
      anchor-relative reduced coords.  Neutral → coord 0 by construction, so
      the affine origin is implicitly 0 (no stored origin).  A node sits at
      distance ∝ ‖δ_L‖ from the origin, so ``along`` displaces more where the
      concept signal is bigger — the intrinsic per-layer lever.

    **Basis caveat (do NOT break PCA@2 ≡ DiM).**  The basis comes from the
    **μ-centered** scatter, *not* the anchor-centered one: at ``K = 2`` the
    μ-centered SVD's sole axis is ``δ̂ = unit(c₀ − c₁)`` (difference-of-means
    exactly), while anchor-centering would inject ``(μ − ν)`` as a spurious
    axis.  Frame (mean + coords) anchors at neutral; basis stays μ-centered.

    ``orient_to`` flips each basis row so node ``orient_to``'s μ-centered
    projection is non-negative — a deterministic sign convention that makes
    the ``K = 2`` / node-0-is-pos case reproduce the DiM ``+δ̂`` orientation
    (``orient_to=None`` leaves the raw SVD/QR sign).

    Returns ``(LayerSubspace.affine(mean, basis, node_coords), mu_coords,
    ev_ratio)`` where ``mu_coords = (centroids − μ)·basisᵀ`` is the
    *μ-centered* reduced coords the caller feeds :func:`subspace_share` for
    the anchor-independent budget weight (coords carry the Euclidean
    position, share carries the Mahalanobis budget — §5).
    """
    centroids = centroids.to(torch.float32)
    K = int(centroids.shape[0])
    if K < 2:
        raise ValueError(f"an affine subspace needs >= 2 nodes, got {K}")
    mu = centroids.mean(dim=0)
    X = centroids - mu  # (K, D) μ-centered
    basis, ev_ratio = _pca_basis(
        X, n_components=n_components, whitener=whitener, layer=layer,
    )
    if orient_to is not None:
        proj = basis @ (centroids[orient_to] - mu)      # (R,)
        signs = torch.where(proj < 0, -1.0, 1.0)        # flip rows facing away
        basis = (basis * signs.unsqueeze(1)).contiguous()
    if neutral_mean is not None:
        anchor = neutral_mean.to(device=centroids.device, dtype=torch.float32).reshape(-1)
    else:
        anchor = mu
    mean = (anchor @ basis.T) @ basis                   # P_basis(anchor) (D,)
    node_coords = (centroids - anchor) @ basis.T        # (K, R) anchor-relative
    mu_coords = X @ basis.T                             # (K, R) μ-centered (share)
    sub = LayerSubspace.affine(mean, basis, node_coords=node_coords)
    return sub, mu_coords, ev_ratio


def subspace_share(
    mu_coords: torch.Tensor,
    basis: torch.Tensor,
    *,
    whitener: "LayerWhitener | None" = None,
    layer: int | None = None,
) -> float:
    """Per-layer budget share — the μ-centered (anchor-independent) spread.

    ``share_L = sqrt(Σ_k coords_kᵀ M_R coords_k)`` whitened (``M_R = B Σ⁻¹ Bᵀ``
    via ``subspace_gram``), else ``‖coords‖_F`` Euclidean — the whitened /
    Euclidean spread of the node centroids around their *own* mean, restricted
    to the subspace.  Drives the apply-time cross-layer budget normalization
    (``Σ_L share_L = 1``).  **Anchor-independent** (μ-centered, not
    neutral-centered): the budget measures *signal spread*, not where neutral
    happens to sit.  At ``K = 2`` / ``R = 1`` this is ``‖δ_L‖_M / √2``
    (whitened) or ``‖δ_L‖₂ / √2`` (Euclidean) — proportional to the DiM bake
    share, so the *normalized* per-layer profile is the DiM one exactly (the
    √2 cancels).  ``mu_coords`` is the second return of
    :func:`fit_affine_subspace`, or ``(centroids − μ)·basisᵀ`` for a curved
    fit (the μ-centered node values, == ``eval_rbf(node_params)`` at the fit
    nodes).
    """
    mu_coords = mu_coords.to(torch.float32)
    if mu_coords.ndim == 1:
        mu_coords = mu_coords.reshape(-1, 1)
    if whitener is not None and layer is not None:
        M_R = whitener.subspace_gram(layer, basis.to(torch.float32))  # (R, R)
        quad = float((mu_coords @ M_R * mu_coords).sum().clamp_min(0.0).item())
        return quad ** 0.5
    return float(torch.linalg.norm(mu_coords).item())


def fit_layer_subspace(
    centroids: torch.Tensor,
    node_params: torch.Tensor,
    *,
    n_components: int = DEFAULT_N_COMPONENTS,
    whitener: "LayerWhitener | None" = None,
    layer: int | None = None,
    neutral_mean: torch.Tensor | None = None,
) -> tuple[LayerSubspace, float]:
    """Fit a PCA subspace + RBF interpolant for one layer (curved).

    ``centroids`` is ``(K, D)`` -- one per-node mean activation -- and
    ``node_params`` is ``(K, m)`` -- the corresponding embedded domain
    coordinates (raw, un-normalized).  The activations are centered and
    reduced to ``R = min(n_components, K-1, rank)`` principal components;
    the embedded coordinates are normalized to the unit box; an ``r**3``
    RBF is fitted from the normalized coordinates to the reduced
    activations.

    **Basis selection — Euclidean vs whitened (Fisher) PCA.**  With
    ``whitener=None`` (the default) the subspace is ordinary PCA of the
    centered centroids: it maximizes raw between-node variance
    ``vᵀ S_b v``.  On real LMs that objective *chases massive-activation
    channels* — they carry the most raw variance regardless of whether
    they carry node signal — so the fitted subspace, its ``mean``, and the
    resulting steering direction all end up dominated by a handful of rogue
    dims, producing unstable per-step norms under steering.

    Passing a ``whitener`` (covering ``layer``) switches to **whitened /
    Fisher PCA**: it maximizes the *ratio* ``vᵀ S_b v / vᵀ Σ v`` where
    ``Σ`` is the neutral-background covariance — the LDA objective.  A
    rogue dim has enormous background variance ``vᵀ Σ v``, so it is divided
    down to nothing — the exact cancellation difference-of-means steering
    gets for free by differencing two means.  The directions that survive
    are where nodes separate *more than the background fluctuates*, i.e.
    the genuine concept signal.  Solved as the generalized eigenproblem
    ``(S_b, Σ)`` via the whitener's low-rank Woodbury ``Σ⁻¹``: eigvecs
    ``a`` of ``G = X Σ⁻¹ Xᵀ`` (``K×K``), directions ``v_r = Σ⁻¹ Xᵀ a_r``.
    The result is re-expressed in a **Euclidean-orthonormal** basis (QR,
    span-preserving) so :func:`decompose` / :func:`subspace_inject` — the
    steering hot path — are untouched; only *which* subspace they operate
    in moves.  The de-rogued subspace barely
    overlaps the rogue-dominated ``mean``, so the angular norm artifact
    collapses for free (no explicit norm-restore needed).  The caller
    gates this all-or-nothing on ``whitener.covers_all`` over the fit
    layers, mirroring the DiM-bake / monitor / share gates.

    **Neutral-anchor (§5).**  ``neutral_mean`` (the layer's neutral baseline)
    anchors the frame: ``mean = P_basis(neutral)`` and the RBF interpolates
    **neutral-relative** reduced coords, so neutral lands at reduced-coord 0
    and the surface passes through each centroid's in-span projection
    ``eval_at(node_i) = P_basis(centroid_i)``.  ``None`` falls back to the
    centroid mean ``μ`` as the anchor (the degenerate path — CPU stubs / no
    neutral cache).  The basis is always derived from the **μ-centered**
    scatter regardless (the basis caveat); only the anchor moves.  The
    dropped off-anchor component cancels in the steered output.

    Returns ``(LayerSubspace, explained_variance_ratio)``.  Under
    Euclidean PCA the EV ratio is ``Σ σ_i² (retained) / Σ σ_i² (all)`` —
    the fraction of raw inter-node variance retained.  Under whitened PCA
    it is the fraction of *whitened* between-variance retained
    (``Σ μ (retained) / Σ μ (all)`` over the generalized eigenvalues) —
    the metric-appropriate fit-quality signal there.  The same diagnostic
    is persisted for fit inspection and share baking; computed from the
    decomposition that already runs, so this is free.
    """
    centroids = centroids.to(torch.float32)
    node_params = node_params.to(torch.float32)
    K = centroids.shape[0]
    if K < 3:
        raise ValueError(f"a manifold needs >= 3 nodes to fit, got {K}")
    mu = centroids.mean(dim=0)
    X = centroids - mu  # (K, D) μ-centered (basis caveat: never anchor-center)

    # Basis selection (Euclidean / whitened-Fisher) is shared with the flat
    # ``fit_affine_subspace`` via ``_pca_basis`` so both pick the subspace
    # identically; only what's built on top (RBF surface vs. analytic affine)
    # differs.
    basis, ev_ratio = _pca_basis(
        X, n_components=n_components, whitener=whitener, layer=layer,
    )
    # Neutral-anchor the frame (§5): ``mean = P_basis(anchor)`` and the RBF
    # interpolates **anchor-relative** reduced coords, so neutral lands at
    # reduced-coord 0 and ``eval_at(node_i) = P_basis(centroid_i)`` (the
    # R-dim surface passes through the centroids' in-span projections).  The
    # anchor is the supplied neutral mean, else the centroid mean ``μ`` (the
    # degenerate fallback — CPU stubs / no-neutral cache).  The off-anchor
    # component dropped by the projection provably cancels in the steered
    # output; projecting only cleans the residual / read-side fraction.
    if neutral_mean is not None:
        anchor = neutral_mean.to(device=centroids.device, dtype=torch.float32).reshape(-1)
    else:
        anchor = mu
    mean = (anchor @ basis.T) @ basis           # P_basis(anchor) (D,)
    coords = (centroids - anchor) @ basis.T     # (K, R) anchor-relative RBF targets

    lo = node_params.min(dim=0).values
    hi = node_params.max(dim=0).values
    coord_offset = lo
    coord_scale = (hi - lo).clamp(min=1e-9)
    normalized = (node_params - coord_offset) / coord_scale

    rbf_weights, poly_coeffs = fit_rbf_interpolant(normalized, coords)
    sub = LayerSubspace(
        mean=mean, basis=basis, node_params=normalized,
        rbf_weights=rbf_weights, poly_coeffs=poly_coeffs,
        coord_offset=coord_offset, coord_scale=coord_scale,
    )
    return sub, ev_ratio


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
    # Per-node assistant-role substitution recorded at fit time, aligned
    # with ``node_labels``.  ``None`` for a given node = "pooled under
    # the standard assistant baseline" (the legacy shape, what every
    # non-role manifold carries).  Used by
    # :meth:`Manifold.nearest_node_role` for role-paired steering.
    node_roles: list[str | None] = field(default_factory=list)
    # Per-node conceptual ``kind`` — ``"abstract"`` (a trait/quality, e.g.
    # ``happy``) or ``"concrete"`` (an entity, e.g. ``pirate``), aligned with
    # ``node_labels``.  ``None`` = unspecified.  A *generation-time* attribute
    # only: it selects the system template and the elicitation role label
    # (``someone {label}`` vs ``{label}``) when authoring a node's
    # conversational corpus.  It does NOT feed the fit — extraction pools in
    # standard-assistant space (swap-back) regardless — so it is carried for
    # provenance / regeneration, not consumed by ``compute_node_centroid``.
    node_kinds: list[str | None] = field(default_factory=list)
    # Per-layer PCA explained-variance ratio recorded at fit time.
    # ``explained_variance[L] = Σ σ²[:R] / Σ σ² (all)`` where ``R`` is
    # the retained subspace rank — a per-layer fit-quality signal used
    # by additive-mode manifold steering to normalize per-α behavioral
    # magnitude across manifolds of varying quality.  Empty dict on
    # pre-v4 manifolds (no EV recorded at fit time) — the additive
    # normalization falls back to "no quality correction" so legacy
    # fits keep their pre-v4 behavior.
    explained_variance: dict[int, float] = field(default_factory=dict)
    # Per-layer Mahalanobis share weight recorded at fit time when a
    # whitener was available — ``share_L = ‖Bᵀ coords_k‖_M`` summed over
    # nodes, the subspace-restricted analogue of vector steering's
    # ``‖d‖_M`` bake score (see ``LayerWhitener.subspace_gram``).  When
    # populated *and* covering every layer, ``hooks._manifold_layer_shares``
    # uses it in place of the Euclidean centroid-spread; an empty dict
    # (no whitener at fit time, e.g. CPU test stubs, or partial layer
    # coverage) falls back to the Euclidean ``‖coords‖_F`` weighting.
    # These are raw per-layer scalars — the apply-time normalization to
    # ``Σ_L share_L = 1`` lives in ``_manifold_layer_shares``.
    mahalanobis_share: dict[int, float] = field(default_factory=dict)
    # Origin ``O_L`` — the **per-layer** foot of the neutral mean on ``M``, in
    # authoring coordinates ``(n,)``, keyed by layer.  Always a point *on* the
    # manifold (each is an ``invert_parameterization`` result), so always affine;
    # there is no linear/affine field.  Two roles in two-op steering: the
    # cold-start seed for that layer's per-token nearest-point foot-follower
    # (``subspace_inject``), and the slide-to target of the ``!`` operator
    # (Phase 2).  Per-layer rather than a single shared coord because each layer
    # embeds the shared authoring coords into activation space differently (its
    # own PCA + RBF), so neutral's foot genuinely differs by depth — and the
    # hot-path follower runs one Gauss-Newton step from a *single* seed with no
    # restarts (unlike ``invert_parameterization``), so a per-layer seed in the
    # right basin avoids a wrong-basin foot on periodic / curved domains.  Empty
    # dict on a fit with no neutral means available (CPU test stubs); the apply
    # path then seeds that layer's foot at the coord-space origin ``zeros(n)``.
    origin: dict[int, torch.Tensor] = field(default_factory=dict)

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
            node_roles=list(self.node_roles),
            node_kinds=list(self.node_kinds),
            explained_variance=dict(self.explained_variance),
            mahalanobis_share=dict(self.mahalanobis_share),
            origin={
                L: o.to(device=device, dtype=dtype)
                for L, o in self.origin.items()
            },
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

    def resolve_position(
        self,
        position: "float | Sequence[float] | str | torch.Tensor",
    ) -> tuple[float, ...]:
        """Coerce a position payload to a coord tuple.

        Two input shapes are accepted (parser produces both):

        - A coord payload (tuple, list, float, or 1-D tensor) — passthrough
          to a plain coord tuple with arity unchanged.  Arity validation
          against the domain's intrinsic dimension happens downstream in
          :meth:`SteeringManager.add_manifold`.
        - A node-label string (``"pirate"``) — sugar for "the coords of
          the node labeled <s>".  The label is looked up in
          :attr:`node_labels` and the matching row of
          :attr:`node_coords` is returned.  An unknown label raises
          :class:`UnknownManifoldLabelError`.

        Label form makes ``persona%pirate`` a first-class steering term
        in the shared grammar; the bare-name resolver (Phase C) builds
        on the same lookup.
        """
        if isinstance(position, str):
            try:
                idx = self.node_labels.index(position)
            except ValueError:
                raise UnknownManifoldLabelError(
                    f"manifold {self.name!r} has no node labeled "
                    f"{position!r}; known labels: "
                    f"{sorted(self.node_labels)}"
                ) from None
            row = self.node_coords[idx]
            return tuple(float(c) for c in row.tolist())
        if isinstance(position, torch.Tensor):
            return tuple(float(c) for c in position.reshape(-1).tolist())
        if isinstance(position, (int, float)):
            return (float(position),)
        return tuple(float(c) for c in position)

    def nearest_node_index(
        self, position: "float | Sequence[float] | str | torch.Tensor",
    ) -> int:
        """Index of the node whose authoring coords lie nearest ``position``.

        Distance is the domain's chordal distance between embedded
        points (the same metric the fit pipeline uses for poisedness
        + the RBF kernel), so periodic axes wrap correctly and a
        sphere is measured on its chord.  ``position`` may be a coord
        payload or a node label — labels short-circuit to the matching
        node index without a distance computation.

        Raises ``ValueError`` when the manifold has no nodes recorded
        (a fitted manifold always carries ``node_coords`` from disk).
        """
        if self.node_coords.numel() == 0 or not self.node_labels:
            raise ValueError(
                f"manifold {self.name!r} carries no node coords — cannot "
                f"resolve a nearest node"
            )
        if isinstance(position, str):
            # A node label trivially is the nearest node to itself.
            # Bypass the geometry; surface UnknownManifoldLabelError on
            # a typo through the same channel as resolve_position.
            try:
                return self.node_labels.index(position)
            except ValueError:
                raise UnknownManifoldLabelError(
                    f"manifold {self.name!r} has no node labeled "
                    f"{position!r}; known labels: "
                    f"{sorted(self.node_labels)}"
                ) from None
        # Coerce + embed both sides through the domain so the metric is
        # consistent.  CPU/fp32 — this runs once per scope entry, off
        # the hot path; we don't need to chase the model's device.
        ref_dtype = self.node_coords.dtype
        pos = self._position_tensor(
            position, device=self.node_coords.device, dtype=ref_dtype,
        )
        pos = self.domain.clamp_position(pos)
        embedded_pos = self.domain.embed(pos)                       # (m,)
        clamped_nodes = self.domain.clamp_position(self.node_coords)
        embedded_nodes = self.domain.embed(clamped_nodes)           # (K, m)
        dists = self.domain.distance(embedded_pos, embedded_nodes)  # (K,)
        return int(torch.argmin(dists).item())

    def nearest_node_label(
        self, position: "float | Sequence[float] | str | torch.Tensor",
    ) -> str:
        """Label of the node nearest ``position``."""
        return self.node_labels[self.nearest_node_index(position)]

    def nearest_node_role(
        self, position: "float | Sequence[float] | str | torch.Tensor",
    ) -> str | None:
        """Role of the node nearest ``position`` — or ``None``.

        Returns ``None`` when the nearest node carries no role (legacy
        nodes, or nodes that opted out of role substitution).  The
        return value rides through to ``session._active_role`` so the
        generation prefill re-applies the substitution at decode time,
        producing role-paired manifold steering (Phase A.3).

        ``node_roles`` may be empty on a legacy fitted manifold whose
        sidecar predates the per-node-role schema — that case also
        returns ``None`` (treat the whole manifold as standard
        assistant baseline).
        """
        if not self.node_roles:
            return None
        idx = self.nearest_node_index(position)
        if idx >= len(self.node_roles):
            return None
        return self.node_roles[idx]

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


# Per-position guard against a degenerate normal-transport renorm in
# :func:`subspace_inject` (the off-manifold residual collapsing onto the
# tangent at a new foot near a fold): below this the transported residual
# is dropped rather than fabricated from a near-zero direction.
_ROTATE_EPSILON: float = 1e-6


def decompose(
    h: torch.Tensor,
    mean: torch.Tensor,
    basis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decompose a centered activation into in-subspace + orthogonal parts.

    ``h`` is ``(.., D)``, ``mean`` ``(D,)``, ``basis`` ``(R, D)``.
    Returns ``(h_par_c, h_perp)``: ``h_par_c`` is the reconstruction of
    the centered activation inside the manifold's affine subspace,
    ``h_perp`` is the orthogonal residual; together they sum to
    ``h - mean`` exactly.

    The shared decomposition step that backs :func:`subspace_inject` and
    the read-side ``ManifoldMonitor``.  All
    three intermediates are kept in the input's dtype — callers that need
    fp32 (the injection functions, the monitor) cast their inputs first.
    """
    centered = h - mean
    coords = centered @ basis.T          # (.., R)
    h_par_c = coords @ basis             # (.., D)
    h_perp = centered - h_par_c          # (.., D)
    return h_par_c, h_perp


def _ortho_basis(
    dirs: Sequence[torch.Tensor], *, eps: float = 1e-6,
) -> tuple[torch.Tensor, list[int]]:
    """Ordered Gram-Schmidt orthonormalization of a list of ``(D,)`` directions.

    Processes ``dirs`` in order; each is unit-normalized first (a basis of the
    *span* — input magnitudes are irrelevant), then a direction whose residual
    norm after removing its projection onto the already-accepted rows falls
    below ``eps`` is dropped — collinear / duplicate directions add no axis.
    Working in unit space makes ``eps`` a scale-free relative tolerance (fp32
    leaves a ~1e-7 residual on a truly-parallel direction, so the ``1e-6``
    default catches it).  Returns ``(B, kept)`` where ``B`` is the ``(R, D)``
    orthonormal basis (``R`` = retained rank) and ``kept`` the indices of
    ``dirs`` that became rows, in order.  fp32 throughout; ``R = 0``
    (all-degenerate) yields an empty basis.

    Ordering is load-bearing for :func:`synthesize_subspace`: feeding *push*
    directions before *ablation* ones keeps the push displacement inside the
    earlier rows, so the ablation-only axes (the orthogonal complement) carry a
    target coordinate of ~0 automatically.
    """
    rows: list[torch.Tensor] = []
    kept: list[int] = []
    for i, d in enumerate(dirs):
        d = d.to(torch.float32)
        dn = float(torch.linalg.vector_norm(d))
        if dn < 1e-12:                 # exact-zero input — no direction
            continue
        v = d / dn                     # work in unit space (scale-free residual)
        for b in rows:
            v = v - (v @ b) * b
        nv = float(torch.linalg.vector_norm(v))
        if nv < eps:                   # collinear with an accepted row
            continue
        rows.append(v / nv)
        kept.append(i)
    if not rows:
        d0 = dirs[0]
        return d0.new_zeros((0, d0.shape[-1]), dtype=torch.float32), kept
    return torch.stack(rows), kept


@dataclass
class SynthesizedSubspace:
    """A per-layer affine subspace synthesized from an active steering term set.

    The dispatch-time analogue of a fitted :class:`Manifold`: instead of loading
    one artifact, the session composes the whole active steering expression into
    a single per-layer subspace + ``along`` target.  This is what lets steering
    keep its superposition semantics under the two-op kernel — one manifold per
    layer holds, because dispatch builds exactly *one* derived subspace per layer
    from every active term (rather than one manifold per concept, which would
    collide at shared layers, ``OverlappingManifoldError``).

    Per layer the subspace spans the union of every active term's directions
    (push + ablation); the ``target_coord`` slides the foot toward each *push*
    term's (signed, coeff-scaled) target on its own axes and toward the origin
    (``0``) on each *ablation* axis, so one geodesic ``along`` slide both pushes
    the concept subspaces and removes the ablated ones.  ``share`` is the
    un-normalized per-layer budget weight — the world push-displacement
    magnitude ``‖Δ_L‖`` (a pure-ablation layer weights by the summed ablation
    magnitude instead); the apply path normalizes it across layers.

    Fields are keyed by layer index; only layers carrying at least one
    non-degenerate active direction (and present in ``neutral_means``) appear.
    """

    layers: dict[int, "LayerSubspace"]        # affine: mean = neutral_L, basis = ortho span
    target_coord: dict[int, torch.Tensor]     # (R_L,) the along target (poles / 0)
    share: dict[int, float]                   # ‖Δ_L‖, un-normalized budget weight


def synthesize_subspace(
    push: Sequence[
        tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], float]
    ],
    ablate: Sequence[dict[int, torch.Tensor]],
    neutral_means: dict[int, torch.Tensor],
    *,
    eps: float = 1e-9,
) -> SynthesizedSubspace:
    """Compose an active steering term set into one affine subspace per layer.

    Each **push** term is an affine subspace fragment
    ``(basis_rows, coord_target, coeff)``:

    - ``basis_rows`` — per-layer ``(R_i, D)`` orthonormal rows (a steering
      *vector* is rank-1, ``(1, D)``; ``personas%pirate`` is rank-8, ``(8, D)``).
    - ``coord_target`` — per-layer ``(R_i,)`` target position *in that fragment's
      own basis* (a pole / node coordinate, origin-relative).
    - ``coeff`` — the signed strength (the blend fraction / α).

    Each **ablation** term is a per-layer ``(R_i, D)`` (or ``(D,)``) direction
    set to remove; its target is the origin (``0``).  ``neutral_means`` supplies
    each layer's anchor (``mean``); a layer absent from it is skipped (no
    anchor ⇒ no subspace).

    Per layer (over the union of layers any term touches):

    - Flatten every present push term's basis rows, then every ablation term's
      rows, into one ordered list (push first); orthonormalize the union
      (:func:`_ortho_basis`) → the merged ``(R, D)`` basis ``B``.
    - World push displacement ``Δ = Σ_push coeffᵢ·(coord_targetᵢ @ basis_rowsᵢ)``
      — each fragment's own ``(R_i,) @ (R_i, D) = (D,)`` world vector, scaled by
      its coeff.  ``target = B @ Δ`` is its coordinate in the merged basis.
      Because ``Δ`` lives in the push span and the ablation-only axes are its
      orthogonal complement, those axes get ``target ≈ 0`` for free — sliding
      the foot toward ``target`` pushes the concepts *and* collapses the ablated
      directions in one op.
    - ``share = ‖Δ‖`` (the world displacement magnitude); a pure-ablation layer
      weights by the summed ablation-row magnitude instead.

    Because the basis is orthonormal, ``‖target‖ = ‖Δ‖`` for a push, so the
    per-layer budget weight and the steered coordinate sit on one consistent
    scale (unlike the rank-1-only predecessor, which mixed a unit target with a
    baked-magnitude share).  Real per-layer coords (Step 3) restore the
    ``∝ ‖δ_L‖`` magnitude that the high-signal layers should carry.

    The strengths live in ``target`` (per-axis), not in a single ``along`` — the
    caller picks ``along`` (the overall slide, the existing manifold-``%``
    knob) and the per-layer (mean-1) share normalization at apply time.  Pure
    tensor, fp32, no model/IO coupling — the dispatch synthesizer (which routes
    the result through ``subspace_inject`` with a ``CustomDomain(R)`` per layer)
    is the only consumer.
    """
    all_layers: set[int] = set()
    for basis_dirs, _coords, _c in push:
        all_layers |= basis_dirs.keys()
    for dirs in ablate:
        all_layers |= dirs.keys()

    layers: dict[int, "LayerSubspace"] = {}
    target_coord: dict[int, torch.Tensor] = {}
    share: dict[int, float] = {}

    for L in sorted(all_layers):
        if L not in neutral_means:
            continue
        mean = neutral_means[L].to(torch.float32).reshape(-1)

        # Push fragments present at this layer: their basis rows (for the span)
        # and their world displacement vector ``coeff · (coords @ basis)``.
        push_rows: list[torch.Tensor] = []          # individual (D,) basis rows
        push_world: list[torch.Tensor] = []         # per-fragment (D,) displacement
        for basis_dirs, coord_dirs, coeff in push:
            B_i = basis_dirs.get(L)
            if B_i is None:
                continue
            B_i = B_i.to(torch.float32)
            if B_i.ndim == 1:
                B_i = B_i.reshape(1, -1)
            if float(torch.linalg.matrix_norm(B_i)) < eps:
                continue
            c_i = coord_dirs.get(L)
            if c_i is None:
                # No target coords for this layer ⇒ no displacement, but the
                # rows still join the span (a degenerate push = ablation).
                c_i = B_i.new_zeros(B_i.shape[0])
            c_i = c_i.to(torch.float32).reshape(-1)
            push_rows.extend(B_i)
            push_world.append(float(coeff) * (c_i @ B_i))   # (D,)

        ablate_rows: list[torch.Tensor] = []        # individual (D,) basis rows
        ablate_raw: list[torch.Tensor] = []         # raw rows (magnitude → share)
        for dirs in ablate:
            d = dirs.get(L)
            if d is None:
                continue
            d = d.to(torch.float32)
            if d.ndim == 1:
                d = d.reshape(1, -1)
            for row in d:
                if float(torch.linalg.vector_norm(row)) < eps:
                    continue
                ablate_rows.append(row)
                ablate_raw.append(row)

        ordered = push_rows + ablate_rows
        if not ordered:
            continue
        # ``_ortho_basis`` uses its own scale-free dependency tolerance; ``eps``
        # here is only the degenerate-direction prefilter (applied above).
        basis, _kept = _ortho_basis(ordered)
        if basis.shape[0] == 0:
            continue

        if push_world:
            delta = torch.stack(push_world).sum(0)
            share_L = float(torch.linalg.vector_norm(delta))
        else:
            delta = torch.zeros_like(mean)
            ablate_sum = torch.stack(ablate_raw).sum(0)
            share_L = float(torch.linalg.vector_norm(ablate_sum))

        target_coord[L] = basis @ delta          # (R,) ablation axes ≈ 0
        layers[L] = LayerSubspace.affine(mean=mean, basis=basis)
        share[L] = share_L

    return SynthesizedSubspace(
        layers=layers, target_coord=target_coord, share=share,
    )


_TANGENT_GRAM_RIDGE = 1e-6


def subspace_inject(
    h: torch.Tensor,
    subspace: LayerSubspace,
    domain: ManifoldDomain,
    target_coord: torch.Tensor,
    foot_seed: torch.Tensor,
    along: "float | torch.Tensor",
    onto: "float | torch.Tensor",
    *,
    gn_steps: int = 1,
    norm_cap: float = 3.0,
    damping: float = DEFAULT_INVERSION_DAMPING,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The unified two-operation manifold injection (replaces angular/additive).

    Decomposes ``h = mean + H_m + H_n + H_o`` against the layer's affine
    subspace and its RBF surface ``M``, then applies two near-orthogonal
    operations, each a coefficient in ``[0, 1]``:

    - **along** (``a``): slide the projected foot from its current position
      ``p*`` toward ``target_coord`` *geodesically in authoring-coord space*
      (``domain.geodesic``), and transport the off-manifold residual ``H_n`` to
      stay normal at the new foot (project onto the new tangent's normal space,
      renorm to the preserved ``‖H_n‖``).  Tangential / directional; leaves
      ``H_o`` untouched.  Replaces the old additive chord — by sliding *on the
      surface* it never cuts through the off-manifold low-density region.
    - **onto** (``o``): scale ``H_n`` by ``(1 − o)`` — collapse onto the surface
      within the subspace.  Vacuous when the surface fills its subspace
      (``H_n ≈ 0``).

    The off-*subspace* residual ``H_o`` is **always kept verbatim** — the old
    third op (``toward``, which scaled ``H_o``) is removed.  It scaled the
    orthogonal complement of *this* subspace, i.e. every composing neighbor's
    span, so it broke orthogonal composition and never cohered as a knob; the
    R-dim ``~``/``|`` semantics are recovered by routing those operators into the
    merged affine subspace as push/ablation axes instead.

    All subspace arithmetic runs in **reduced (R-dim) coordinates**; because
    ``basis`` is orthonormal, ``‖H_n_reduced‖ = ‖H_n‖`` exactly, so the
    transport's norm-preservation holds and the cost is ``O(R)`` not ``O(D)``.
    fp32 throughout (fp16 sum-of-squares overflows at large ``D``).

    ``foot_seed`` ``(.., n)`` is the warm start for the nearest-point foot — the
    previous token's foot on the hot path, refined by ``gn_steps`` shared
    :func:`_gn_step` iterations (one per token in steady state).  Returns
    ``(h_new, foot)`` where ``foot`` ``(.., n)`` is *this* token's refined
    pre-slide foot, to seed the next token.

    Order is fixed **along → onto**: the transport (along) must run before
    ``onto`` scales the transported residual.  No global norm preservation —
    ``onto`` is *meant* to shrink ``‖h − mean‖``, and the apply-time (mean-1)
    share normalization controls the per-α magnitude; a soft cap
    ``‖h_new‖ ≤ norm_cap·‖h‖`` guards only against off-domain RBF extrapolation
    blowup.
    """
    h_f32 = h.to(torch.float32)
    mean = subspace.mean.to(torch.float32)            # (D,)
    basis = subspace.basis.to(torch.float32)          # (R, D)
    target = target_coord.to(device=h_f32.device, dtype=torch.float32)

    centered = h_f32 - mean                            # (.., D)
    q = centered @ basis.T                             # (.., R) reduced coords of h_par
    h_par = q @ basis                                  # (.., D) in-subspace reconstruction
    h_perp = centered - h_par                          # (.., D) = H_o

    if subspace.is_affine:
        # --- flat (folded-vector) shortcut --------------------------------
        # The surface fills the subspace, so reduced coords *are* authoring
        # coords (identity map): the foot is ``q`` exactly, ``H_n ≡ 0`` and
        # ``onto`` is vacuous (ignored).  No GN solve, no RBF eval, no
        # tangent Gram-solve — the cost the common steering case can't pay.
        # ALONG slides ``q`` toward ``target`` geodesically (CustomDomain ⇒
        # linear); the off-subspace residual ``H_o`` is kept verbatim.
        p_new = domain.geodesic(q, target, along)      # (.., n==R)
        new_par = p_new @ basis                        # (.., D)
        new_perp = h_perp                              # (.., D) kept verbatim
        h_new = mean + new_par + new_perp
        norm_pre = torch.linalg.vector_norm(h_f32, dim=-1, keepdim=True)
        norm_post = torch.linalg.vector_norm(h_new, dim=-1, keepdim=True)
        cap = norm_cap * norm_pre
        h_new = torch.where(
            norm_post > cap, h_new * (cap / norm_post.clamp(min=1e-6)), h_new,
        )
        return h_new.to(h.dtype), q

    np_, rw, pc = subspace.rbf_params()
    np_ = np_.to(torch.float32)
    rw = rw.to(torch.float32)
    pc = pc.to(torch.float32)
    offset = subspace.coord_offset.to(torch.float32)  # (m,)
    scale = subspace.coord_scale.to(torch.float32)    # (m,)

    # --- foot p* : nearest point on M to q, warm-started from foot_seed ---
    p = foot_seed
    for _ in range(int(gn_steps)):
        p, _ = _gn_step(p, q, np_, rw, pc, offset, scale, domain, damping)

    foot_red = eval_rbf(np_, rw, pc, (domain.embed(p) - offset) / scale)  # (.., R)
    Hn_red = q - foot_red                              # (.., R) off-manifold-in-subspace (reduced)
    Hn_norm = torch.linalg.vector_norm(Hn_red, dim=-1, keepdim=True)      # (.., 1)

    # --- ALONG: geodesic slide of the foot, with H_n transport ---
    p_new = domain.geodesic(p, target, along)          # (.., n)
    emb_new = (domain.embed(p_new) - offset) / scale   # (.., m)
    foot_new_red = eval_rbf(np_, rw, pc, emb_new)      # (.., R)

    # World-reduced tangent columns of M at p_new: dRBF/dcoord chained through
    # the embedding Jacobian (the reduced-space analogue of Manifold.tangent).
    tangent = (
        eval_rbf_jacobian(np_, rw, pc, emb_new) / scale  # (.., R, m)
    ) @ domain.embed_jacobian(p_new)                     # (.., R, n)
    # Project H_n onto the normal space at p_new (remove its tangent component).
    gram = tangent.transpose(-1, -2) @ tangent           # (.., n, n)
    n_dim = gram.shape[-1]
    eye = torch.eye(n_dim, device=gram.device, dtype=gram.dtype)
    rhs = tangent.transpose(-1, -2) @ Hn_red.unsqueeze(-1)  # (.., n, 1)
    bsz = gram.shape[:-2]
    coeff = torch.linalg.solve(
        (gram + _TANGENT_GRAM_RIDGE * eye).reshape(-1, n_dim, n_dim).contiguous(),
        rhs.reshape(-1, n_dim, 1).contiguous(),
    ).reshape(*bsz, n_dim, 1)
    Hn_tan = (tangent @ coeff).squeeze(-1)               # (.., R) tangent component
    Hn_normal = Hn_red - Hn_tan                          # (.., R) normal at p_new
    Hn_normal_norm = torch.linalg.vector_norm(Hn_normal, dim=-1, keepdim=True)
    # Renorm to the preserved ‖H_n‖; if the projection collapsed (H_n went
    # tangent at the new foot — a large slide near a fold), drop it rather than
    # fabricate a direction (risk-2 mitigation), via a hot-path-safe ``where``.
    Hn_trans = Hn_normal * (Hn_norm / Hn_normal_norm.clamp(min=_ROTATE_EPSILON))
    Hn_trans = torch.where(
        Hn_normal_norm < _ROTATE_EPSILON, torch.zeros_like(Hn_trans), Hn_trans,
    )

    # --- ONTO: scale the transported off-manifold residual ---
    Hn_final = (1.0 - onto) * Hn_trans                   # (.., R)

    new_par = (foot_new_red + Hn_final) @ basis          # (.., D) back to world

    # The off-subspace residual ``H_o`` is kept verbatim (the old ``toward`` op
    # that scaled it is removed — it scaled the orthogonal complement of this
    # subspace, breaking orthogonal composition with neighboring terms).
    new_perp = h_perp                                    # (.., D) kept verbatim

    h_new = mean + new_par + new_perp                    # (.., D)

    # Soft safety cap: only fires on pathological off-domain RBF extrapolation
    # (clamp_position keeps p_new in-box for open axes, so this is belt-and-
    # suspenders, not the norm semantic — onto is allowed to shrink ‖h‖).
    norm_pre = torch.linalg.vector_norm(h_f32, dim=-1, keepdim=True)
    norm_post = torch.linalg.vector_norm(h_new, dim=-1, keepdim=True)
    cap = norm_cap * norm_pre
    h_new = torch.where(
        norm_post > cap, h_new * (cap / norm_post.clamp(min=1e-6)), h_new,
    )
    return h_new.to(h.dtype), p


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
    model: torch.nn.Module,
    tokenizer: object,
    layers: torch.nn.ModuleList,
    device: torch.device,
    responses: list[str],
    prompts: list[str],
    *,
    role: str | None = None,
    model_type: str | None = None,
) -> dict[int, torch.Tensor]:
    """Mean per-layer pooled activation over a manifold node's responses.

    Conversational (4.0 / A2) pooling: a node corpus is a list of in-character
    *responses* to the shared baseline *prompts*, aligned positionally as
    ``responses[i] -> prompts[i % len(prompts)]``.  Each is captured as a
    ``[user: prompt, assistant: response]`` pair (no system, standard label)
    via :func:`saklas.core.vectors._encode_and_capture_all` -- the same
    last-content-token, fp32 pooling discipline that backs
    :func:`saklas.core.vectors.compute_layer_means`, with the same MPS
    ``empty_cache`` discipline between forward passes.

    ``role`` (optional): substitute a custom assistant-role label into the
    chat template via :func:`saklas.core.role_templates.apply_with_role`, so
    the pooled centroid lives in persona-baseline activation space instead of
    the standard assistant (swap-back) baseline.  Requires ``model_type``.
    ``role=None`` is the swap-back default.

    Returns ``{layer_idx: centroid (D,)}`` in fp32 on CPU.
    """
    from saklas.core.vectors import _encode_and_capture_all

    if not responses:
        raise ValueError("manifold node has no responses")
    if not prompts:
        raise ValueError("conversational capture needs at least one baseline prompt")
    k = len(prompts)
    if len(responses) % k != 0:
        raise ValueError(
            f"node corpus ({len(responses)} responses) must be a multiple of "
            f"the baseline prompt set ({k}); responses align response[i] -> "
            f"prompt[i % k]"
        )

    n_layers = len(layers)
    sums: dict[int, torch.Tensor] = {}
    is_mps = getattr(device, "type", None) == "mps"

    for i, response in enumerate(responses):
        per_layer = _encode_and_capture_all(
            model, tokenizer, prompts[i % k], response, layers, device,
            role=role, model_type=model_type,
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

    n = len(responses)
    return {idx: sums[idx] / n for idx in range(n_layers)}


# ------------------------------------------------------------- save/load ---

def save_manifold(
    manifold: Manifold, path: str | Path, metadata: dict[str, object],
) -> None:
    """Save a fitted manifold as ``.safetensors`` + a ``.json`` sidecar.

    The safetensors payload carries, per layer ``L``: ``layer_<L>.mean``,
    ``layer_<L>.basis``, ``layer_<L>.node_params``,
    ``layer_<L>.rbf_weights``, ``layer_<L>.poly_coeffs``,
    ``layer_<L>.coord_offset``, ``layer_<L>.coord_scale``; plus a shared
    bare ``node_coords`` tensor.  The sidecar carries the manifold identity
    (name, domain spec, ordered node labels, feature space), the per-layer
    ``origin_per_layer`` (the authoring-coordinate foot of the neutral mean,
    keyed by layer), plus the provenance fields in ``metadata``.

    **Affine (flat / folded-vector) layers** write only ``layer_<L>.mean`` +
    ``layer_<L>.basis`` — there is no RBF triple and the coord normalization
    is identity (reconstructed from the basis shape by
    :meth:`LayerSubspace.affine` on load).  The *absence* of ``node_params``
    on disk is the read-side affine marker ``load_manifold`` keys on, so a
    folded-vector artifact and a fitted manifold share one save/load path.
    """
    from saklas.io.manifolds import MANIFOLD_FORMAT_VERSION

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Every saklas safetensor artifact is stored fp32 (cast at the writer);
    # fits already produce fp32, so the casts are idempotent guarantees.
    tensors: dict[str, torch.Tensor] = {
        "node_coords": manifold.node_coords.contiguous().to(torch.float32).cpu(),
    }
    for idx, sub in manifold.layers.items():
        tensors[f"layer_{idx}.mean"] = sub.mean.contiguous().to(torch.float32).cpu()
        tensors[f"layer_{idx}.basis"] = sub.basis.contiguous().to(torch.float32).cpu()
        if sub.is_affine:
            # Flat (folded-vector / subspace) layer: no RBF surface, and the
            # coord normalization is identity — rebuilt from the basis shape
            # by ``LayerSubspace.affine`` on load.  Persist mean + basis (the
            # *absence* of ``node_params`` on disk is the affine marker), plus
            # the per-layer **real, neutral-anchored** node coords ``(K, R)``
            # when present — the steer-target source (§5).  Older affine
            # artifacts without it load with ``node_coords=None``.
            if sub.node_coords is not None:
                tensors[f"layer_{idx}.node_coords"] = (
                    sub.node_coords.contiguous().to(torch.float32).cpu()
                )
            continue
        np_, rw, pc = sub.rbf_params()
        tensors[f"layer_{idx}.node_params"] = np_.contiguous().to(torch.float32).cpu()
        tensors[f"layer_{idx}.rbf_weights"] = rw.contiguous().to(torch.float32).cpu()
        tensors[f"layer_{idx}.poly_coeffs"] = pc.contiguous().to(torch.float32).cpu()
        tensors[f"layer_{idx}.coord_offset"] = sub.coord_offset.contiguous().to(torch.float32).cpu()
        tensors[f"layer_{idx}.coord_scale"] = sub.coord_scale.contiguous().to(torch.float32).cpu()
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
    # Per-layer explained-variance ratio (v4+).  Stored as
    # ``{str(idx): float}`` for JSON compatibility; loaded back into
    # an ``int``-keyed dict.  Older fits without EV simply skip the
    # field and load with an empty dict downstream.
    if manifold.explained_variance:
        sidecar["explained_variance_per_layer"] = {
            str(idx): float(v)
            for idx, v in manifold.explained_variance.items()
        }
    # Per-layer Mahalanobis share weight (whitened bake-score analogue).
    # Stored as ``{str(idx): float}`` like EV; absent when no whitener was
    # available at fit time, in which case the apply-time share weighting
    # falls back to the Euclidean centroid-spread.
    if manifold.mahalanobis_share:
        sidecar["mahalanobis_share_per_layer"] = {
            str(idx): float(v)
            for idx, v in manifold.mahalanobis_share.items()
        }
    # Per-layer origin ``O_L`` (the authoring-coordinate foot of the neutral
    # mean) — the cold-start foot seed.  Stored ``{str(L): [coord, ...]}`` like
    # the other per-layer bakes; absent on a fit with no neutral means.
    if manifold.origin:
        sidecar["origin_per_layer"] = {
            str(idx): [float(c) for c in o.reshape(-1).tolist()]
            for idx, o in manifold.origin.items()
        }
    for key in (
        "nodes_sha256", "sae_release", "sae_revision", "sae_ids_by_layer",
        # Share-weighting metric ("mahalanobis" / "euclidean") — the
        # manifold analogue of the vector sidecar's ``bake`` field.
        "share_metric",
        # PCA subspace-selection metric ("mahalanobis" => whitened/Fisher
        # PCA, "euclidean" => ordinary centroid PCA).  Provenance only —
        # the fitted basis is baked into the tensor, so the runtime hot
        # path needs nothing from this field; surfaced by `vector manifold
        # show` and the inspector.
        "subspace_metric",
        # Discover-mode fields.  ``fit_mode`` discriminates authored vs
        # discover at read time; ``hyperparams`` records the knobs the
        # fitter was called with (max_dim / var_threshold / k_nn /
        # bandwidth) for reproducibility; ``diagnostics`` carries the
        # per-method PCA variance bars or spectral spectrum for the
        # CLI / webui inspector.  All absent for authored fits.
        "fit_mode", "hyperparams", "diagnostics",
        # Per-node role-augmented fit metadata.  Aligned with
        # ``node_labels`` index-by-index; ``None`` for a given node means
        # "pooled under the standard assistant baseline" (the legacy
        # shape, what every pre-role-differential manifold carries).
        # Absent on non-role manifolds — the pipeline only stamps it
        # when at least one node opts into role substitution.
        "node_roles",
        # Per-node conceptual kind ("abstract"/"concrete"), aligned with
        # ``node_labels``.  Generation-time provenance (system template +
        # elicitation role label); absent when no node carries a kind.
        "node_kinds",
        # Merge provenance ({coord: {alpha, project_away, tensor_sha256}}),
        # carried on a ``fit_mode="baked"`` manifold produced by
        # :func:`saklas.io.merge.merge_into_manifold`.  Informational only — a
        # baked manifold never re-fits, so nothing branches on it; surfaced
        # by the inspector the way the legacy pack sidecar's ``components``
        # was.  Absent on every non-merge fit.
        "components",
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
        if "node_params" not in parts:
            # Affine (flat / folded-vector) layer — only mean + basis on disk
            # (the coord normalization is identity, rebuilt from the basis
            # shape), plus the per-layer real node coords when the writer
            # stamped them.  Read side of ``save_manifold``'s affine branch.
            layers[idx] = LayerSubspace.affine(
                mean=parts["mean"], basis=parts["basis"],
                node_coords=parts.get("node_coords"),
            )
            continue
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

    ev_raw = sidecar.get("explained_variance_per_layer") or {}
    explained_variance: dict[int, float] = {
        int(k): float(v) for k, v in ev_raw.items()
    }

    maha_raw = sidecar.get("mahalanobis_share_per_layer") or {}
    mahalanobis_share: dict[int, float] = {
        int(k): float(v) for k, v in maha_raw.items()
    }

    # Per-layer origin ``O_L`` (authoring-coordinate foot of neutral).  Absent
    # on a pre-origin fit → empty dict; the apply path seeds at ``zeros(n)``.
    origin_raw = sidecar.get("origin_per_layer") or {}
    origin: dict[int, torch.Tensor] = {
        int(k): torch.tensor([float(c) for c in v], dtype=torch.float32)
        for k, v in origin_raw.items()
    }

    return Manifold(
        name=sidecar.get("name", path.parent.name),
        domain=domain,
        node_labels=list(sidecar.get("node_labels", [])),
        node_coords=node_coords,
        layers=layers,
        feature_space=sidecar.get("feature_space", "raw"),
        metadata=sidecar,
        # ``node_roles`` is absent on non-role manifolds (every
        # pre-Phase-A fit); the loaded list stays empty in that case.
        node_roles=list(sidecar.get("node_roles", [])),
        # ``node_kinds`` is absent on manifolds authored without the
        # abstract/concrete distinction; the loaded list stays empty then.
        node_kinds=list(sidecar.get("node_kinds", [])),
        explained_variance=explained_variance,
        mahalanobis_share=mahalanobis_share,
        origin=origin,
    )


def _gn_step(
    p: torch.Tensor,
    q: torch.Tensor,
    node_params: torch.Tensor,
    rbf_weights: torch.Tensor,
    poly_coeffs: torch.Tensor,
    coord_offset: torch.Tensor,
    coord_scale: torch.Tensor,
    domain: ManifoldDomain,
    damping: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One damped Gauss-Newton (Levenberg-Marquardt) step of the nearest-point
    inversion ``argmin_p ||s(p) - q||``.

    ``p`` ``(.., n)`` are authoring coords, ``q`` ``(.., R)`` the reduced-space
    targets (broadcastable against ``p``'s leading dims -- the inversion fans
    ``p`` over ``S`` restarts against a ``(.., 1, R)`` query). The remaining
    args are the subspace's RBF tensors **already cast to ``p``'s device/dtype**
    (the caller hoists that out of the loop). Returns ``(p_new, resid_norm)``:
    the clamped post-step position and the *pre-step* reduced-space residual
    norm ``(..,)`` -- the latter lets a warm-started caller (the steering
    foot-follower) gate on whether a single step actually reduced the residual.

    Shared by :func:`invert_parameterization` (looped ``max_iter`` times over
    ``S`` warm starts) and the two-op steering kernel (one step per token,
    warm-started from the previous foot), so the LM math lives in exactly one
    place.
    """
    emb = (domain.embed(p) - coord_offset) / coord_scale          # (.., m)
    resid = eval_rbf(node_params, rbf_weights, poly_coeffs, emb) - q  # (.., R)
    j_auth = (
        eval_rbf_jacobian(node_params, rbf_weights, poly_coeffs, emb)
        / coord_scale                                             # (.., R, m)
    ) @ domain.embed_jacobian(p)                                  # (.., R, n)
    jt = j_auth.transpose(-1, -2)                                 # (.., n, R)
    jtj = jt @ j_auth                                             # (.., n, n)
    jtr = jt @ resid.unsqueeze(-1)                                # (.., n, 1)
    diag = torch.diagonal(jtj, dim1=-2, dim2=-1)                  # (.., n)
    reg = torch.diag_embed(
        damping * diag.clamp(min=_INVERSION_DIAG_FLOOR) + _INVERSION_DIAG_FLOOR
    )
    A = jtj + reg                                                 # (.., n, n)
    n_dim = A.shape[-1]
    bsz = A.shape[:-2]
    step = torch.linalg.solve(
        A.reshape(-1, n_dim, n_dim).contiguous(),
        jtr.reshape(-1, n_dim, 1).contiguous(),
    ).reshape(*bsz, n_dim)                                        # (.., n)
    p_new = domain.clamp_position(p - step)
    resid_norm = torch.linalg.vector_norm(resid, dim=-1)         # (..,)
    return p_new, resid_norm


def invert_parameterization(
    subspace: LayerSubspace,
    domain: ManifoldDomain,
    query: torch.Tensor,
    node_coords: torch.Tensor,
    *,
    max_iter: int = DEFAULT_INVERSION_MAX_ITER,
    n_restarts: int = DEFAULT_INVERSION_RESTARTS,
    damping: float = DEFAULT_INVERSION_DAMPING,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Nearest-point projection of ``query`` onto the fitted manifold.

    Returns ``(positions, dist)``: ``positions`` ``(.., n)`` are the
    authoring coordinates whose interpolant value minimizes the Euclidean
    distance to each query in reduced-coordinate space, ``dist`` ``(..,)``
    is that distance.  ``query`` is ``(.., R)``; ``node_coords`` ``(K, n)``
    are the manifold's authoring node coordinates (the warm-start seeds).

    This is the paper's ``s^-1`` map.  It minimizes ``||s(p) - q||`` over
    authoring coords ``p`` by damped Gauss-Newton (Levenberg-Marquardt):
    seed each query at its nearest fit node(s) in reduced space, then take
    a fixed number of LM steps using the analytic RBF Jacobian chained
    through the domain's embedding Jacobian, projecting back onto the
    domain (``clamp_position`` -- open-clamp, periodic-wrap, sphere-retract)
    after each step.  Cost is independent of intrinsic dimension, unlike
    the former ``resolution**n`` grid scan, and the landing is continuous
    rather than quantized to a grid.  ``n_restarts`` warm starts from the
    top nearest nodes guard against a fold/periodic local minimum; the best
    final residual per query wins.  Used only by the naturalness eval and
    ``ManifoldMonitor.score_aggregate`` -- never the steering hot path.
    """
    n = domain.intrinsic_dim
    R = query.shape[-1]
    lead = query.shape[:-1]
    device = query.device
    dtype = query.dtype if query.dtype.is_floating_point else torch.float32
    flat = query.reshape(-1, R).to(dtype)  # (N, R)
    N = flat.shape[0]

    # Subspace pieces on the query's device/dtype -- the caller may hand
    # us a subspace still resident on its load device (the read-side
    # aggregate moves only ``mean``/``basis``).
    np_, rw, pc = subspace.rbf_params()
    np_ = np_.to(device=device, dtype=dtype)  # (K, m)
    rw = rw.to(device=device, dtype=dtype)
    pc = pc.to(device=device, dtype=dtype)
    offset = subspace.coord_offset.to(device=device, dtype=dtype)  # (m,)
    scale = subspace.coord_scale.to(device=device, dtype=dtype)  # (m,)

    node_coords = node_coords.to(device=device, dtype=dtype)  # (K, n)
    K = node_coords.shape[0]
    restarts = max(1, min(int(n_restarts), K))

    def _eval(p: torch.Tensor) -> torch.Tensor:
        # Inline normalize (avoids ``subspace._normalize`` reaching for the
        # subspace's own offset/scale on a possibly-foreign device).
        return eval_rbf(np_, rw, pc, (domain.embed(p) - offset) / scale)

    # Reduced values at the fit nodes -- the RBF is exact at ``node_params``
    # so this recovers the per-node centroids in reduced coords without a
    # stored field.  Used to pick each query's nearest node(s) as seeds.
    node_vals = eval_rbf(np_, rw, pc, np_)  # (K, R)
    seed_idx = torch.cdist(flat, node_vals).topk(
        restarts, dim=-1, largest=False,
    ).indices  # (N, S)
    p = domain.clamp_position(node_coords[seed_idx])  # (N, S, n)
    q = flat.unsqueeze(1)  # (N, 1, R) -- broadcasts over the S restarts

    # Each step shares the LM body with the steering foot-follower via
    # ``_gn_step``; ``q`` is ``(N, 1, R)`` and broadcasts over the ``S``
    # restarts.  The internal ``reshape(-1, n, n)`` there also dodges
    # ``torch.linalg.solve``'s size-1-leading-batch out-resize warning on MPS.
    for _ in range(int(max_iter)):
        p, _ = _gn_step(p, q, np_, rw, pc, offset, scale, domain, damping)

    # Best restart per query by final reduced-space residual norm.
    final_res = torch.linalg.vector_norm(_eval(p) - q, dim=-1)  # (N, S)
    best = final_res.argmin(dim=-1)  # (N,)
    pos = p.gather(1, best[:, None, None].expand(N, 1, n)).squeeze(1)  # (N, n)
    dist = final_res.gather(1, best[:, None]).squeeze(1)  # (N,)
    return (
        pos.reshape(*lead, n),
        dist.reshape(lead),
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
    sub, _ev = fit_layer_subspace(
        to_hellinger(centroid_dists), node_params, n_components=n_components,
    )
    return sub


def trajectory_naturalness(
    traj_dists: torch.Tensor,
    behavior: LayerSubspace,
    domain: ManifoldDomain,
    node_coords: torch.Tensor,
) -> torch.Tensor:
    """Per-step Bhattacharyya distance from a trajectory to a behavior manifold.

    ``traj_dists`` is ``(T, V)`` -- the sequence of next-token
    distributions a generation produced.  ``behavior`` is a behavior
    manifold from :func:`fit_behavior_manifold`, ``domain`` its
    :class:`ManifoldDomain`, and ``node_coords`` ``(K, n)`` the manifold's
    authoring node coordinates (warm-start seeds for the inverse map).
    For each step the nearest point on the behavior manifold is found (in
    Hellinger space) and the Bhattacharyya distance to it is returned --
    low means the step sits on the natural behavior manifold, high flags
    an off-manifold "teleportation" artifact.

    Returns a ``(T,)`` tensor of per-step distances.
    """
    h = to_hellinger(traj_dists)  # (T, V)
    coords = (h - behavior.mean) @ behavior.basis.T  # (T, R)
    pos, _ = invert_parameterization(behavior, domain, coords, node_coords)
    embedded = domain.embed(pos)  # (T, m)
    curve_coords = eval_rbf(
        *behavior.rbf_params(), behavior._normalize(embedded),
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
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)  # pyright: ignore[reportCallIssue]  # tokenizer typed as object
    ids = enc["input_ids"].to(device)
    if ids.shape[1] == 0:
        raise ValueError("trajectory text tokenized to zero tokens")
    with torch.inference_mode():
        logits = model(input_ids=ids, use_cache=False).logits  # pyright: ignore[reportCallIssue]  # model typed as object
    return torch.softmax(logits[0].float(), dim=-1).detach().to("cpu")


def _next_token_distribution(
    model: object, tokenizer: object, text: str, device: torch.device,
) -> torch.Tensor:
    """Softmax over the final-position logits for ``text`` -- ``(V,)`` fp32 CPU."""
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)  # pyright: ignore[reportCallIssue]  # tokenizer typed as object
    ids = enc["input_ids"]
    if ids.numel() == 0:
        bos = getattr(tokenizer, "bos_token_id", None) or 0
        ids = torch.tensor([[bos]])
    ids = ids.to(device)
    with torch.inference_mode():
        logits = model(input_ids=ids, use_cache=False).logits  # pyright: ignore[reportCallIssue]  # model typed as object
    return torch.softmax(logits[0, -1].float(), dim=-1).detach().to("cpu")


__all__ = [
    "DEFAULT_N_COMPONENTS",
    "DEFAULT_INVERSION_MAX_ITER",
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
    "decompose",
    "subspace_inject",
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
