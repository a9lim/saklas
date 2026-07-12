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
import os
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence, cast

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from saklas.core.errors import SaklasError, is_out_of_memory_error

if TYPE_CHECKING:
    from saklas.core.mahalanobis import LayerWhitener

log = logging.getLogger(__name__)

# Numerical fitting semantics are independent of the folder/tensor wire
# format. Bump this whenever PCA/Fisher selection, topology choice, RBF/sigma
# fitting, DLS, or share allocation changes incompatibly.
MANIFOLD_FIT_POLICY_VERSION = 1


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
# Used only by the naturalness eval and the read-side ``Monitor``
# aggregate -- never the steering hot path, which steers to a fixed
# position.  The solve is warm-started from the nearest fit node(s) and
# Marquardt-damped, so a fixed dozen iterations converges in
# authoring-coord space *independent of intrinsic dimension*.  This
# replaces a grid scan that was O(resolution**n): even the n=2 path ran
# 512**2 = 262k RBF evals per layer, and the budget-capped high-n path
# degraded to a 4-point-per-axis landing on the bundled 8-D ``personas``.
DEFAULT_INVERSION_MAX_ITER = 12
DEFAULT_INVERSION_RESTARTS = 3
# Warm-started inversion (curved-probe per-token foot-follow): when the caller
# hands a previous foot as ``warm_start``, the activation has drifted only one
# decode step, so the carried foot is already near this token's nearest point —
# a handful of LM steps from it (plus one nearest-node restart as a basin-jump
# safety net) converges where the cold 12-iter / 3-restart search would.  This
# is the read-side analogue of the steering foot-follower's one-warm-step path.
DEFAULT_INVERSION_WARM_ITER = 4
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

    def _tangent(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Tangent at ``a`` pointing toward ``b`` in authoring coords.

        Default is the linear ``b - a`` (correct for any flat, non-wrapping
        domain — :class:`CustomDomain` and the open-axis part of a
        :class:`BoxDomain`).  :class:`BoxDomain` overrides it for periodic axes
        (wrap-aware minimal arc).  Consumed by :meth:`translate_foot`.
        """
        return b - a

    def translate_foot(
        self,
        p: torch.Tensor,
        origin: torch.Tensor,
        target: torch.Tensor,
        frac: float | torch.Tensor,
    ) -> torch.Tensor:
        """Translate the foot ``p`` ``(.., n)`` by the fixed offset toward target.

        The ``along`` step.  Shift *every* token's foot by the same displacement
        — the neutral→target tangent ``target − origin``, parallel-transported to
        ``p``, scaled by ``frac`` — rather than sliding each foot onto the
        absolute ``target``.  The fixed offset **preserves the per-token
        in-subspace spread**, which the kernel ablation showed is what keeps
        strong steer coherent (collapsing onto one target erases that spread and
        degenerates into looping).

        ``p_new = clamp(p + frac · transport_{origin→p}(target − origin))``.

        Flat default — transport is the identity, so the offset is the plain
        ``_tangent(origin, target)``.  :class:`SphereDomain` overrides for
        curvature; :class:`BoxDomain` inherits this but supplies a wrap-aware
        ``_tangent``.  For the affine frame (``origin = 0``) this is
        ``p + frac · target``.
        """
        return self.clamp_position(p + frac * self._tangent(origin, target))


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

    def _tangent(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Linear delta, but periodic axes take the wrap-aware minimal arc.

        The translate connection on a torus/cylinder is flat (a fixed
        authoring-coord offset is parallel along every axis), so
        :meth:`ManifoldDomain.translate_foot` is inherited unchanged — it only
        needs this wrap-aware tangent for the periodic axes.
        """
        delta = b - a
        for i, ax in enumerate(self._axes):
            if ax.periodic:
                half = ax.period / 2.0
                delta[..., i] = (
                    torch.remainder(delta[..., i] + half, ax.period) - half
                )
        return delta

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

    @staticmethod
    def _sphere_log(ea: torch.Tensor, eb: torch.Tensor) -> torch.Tensor:
        """Log map at unit vector ``ea`` toward ``eb`` — tangent in ``R^(n+1)``."""
        dot = (ea * eb).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
        omega = torch.arccos(dot)
        u = eb - dot * ea
        un = torch.linalg.vector_norm(u, dim=-1, keepdim=True)
        return torch.where(
            un < 1e-9, torch.zeros_like(u), omega * u / un.clamp_min(1e-9),
        )

    @staticmethod
    def _sphere_exp(ea: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Exp map at ``ea`` of tangent ``v`` — a unit vector in ``R^(n+1)``."""
        theta = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
        e = torch.cos(theta) * ea + torch.sin(theta) * v / theta.clamp_min(1e-9)
        return torch.where(theta < 1e-9, ea.expand_as(e), e)

    @staticmethod
    def _sphere_transport(
        v: torch.Tensor, ea: torch.Tensor, eb: torch.Tensor,
    ) -> torch.Tensor:
        """Parallel-transport tangent ``v`` (at ``ea``) along the geodesic to ``eb``."""
        c = (ea * eb).sum(dim=-1, keepdim=True)
        coeff = (v * eb).sum(dim=-1, keepdim=True) / (1.0 + c).clamp_min(1e-9)
        return v - coeff * (ea + eb)

    def translate_foot(
        self,
        p: torch.Tensor,
        origin: torch.Tensor,
        target: torch.Tensor,
        frac: float | torch.Tensor,
    ) -> torch.Tensor:
        """Curvature-correct translate in the unit-vector embedding.

        Parallel-transports the neutral→target tangent ``log_origin(target)`` to
        ``p`` (the flat ``target − origin`` is *not* parallel on a curved sphere)
        and exponentiates the ``frac``-scaled offset.  See
        :meth:`ManifoldDomain.translate_foot`.
        """
        ep, eo, et = self.embed(p), self.embed(origin), self.embed(target)
        offset = self._sphere_transport(self._sphere_log(eo, et), eo, ep)
        e_new = self._sphere_exp(ep, frac * offset)
        return self.clamp_position(self._unembed(e_new))

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


def _rbf_poised(node_params: torch.Tensor) -> tuple[int, int]:
    """Validate affine poisedness for an RBF fit; return ``(K, m)``.

    Mirrors the checks at the head of :func:`fit_rbf_interpolant` so the
    smoothing path raises the *same* ``ValueError`` messages on a
    rank-deficient layout (the penalty conditions the kernel block but the
    constraint ``Qᵀw = 0`` still needs ``Q`` full column rank for the
    polynomial coefficients to be determined).
    """
    K, m = node_params.shape
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
    return K, m


def _rbf_saddle(
    A: torch.Tensor, Q: torch.Tensor, rhs_top: torch.Tensor,
) -> torch.Tensor:
    """Solve ``[[A, Q],[Qᵀ, 0]] [x; c] = [rhs_top; 0]`` and return ``[x; c]``.

    The shared saddle assembler for the smoothing path.  ``A`` is the
    (possibly penalized) ``(K, K)`` kernel block, ``Q`` the ``(K, m+1)``
    polynomial block, ``rhs_top`` the ``(K, P)`` top right-hand side (node
    values for a fit; ``I_K`` for the smoother matrix).  Symmetric-indefinite
    ⇒ LU (``torch.linalg.solve``), never Cholesky — same as
    :func:`fit_rbf_interpolant`.
    """
    mp1 = Q.shape[1]
    P = rhs_top.shape[1]
    top = torch.cat([A, Q], dim=1)                                  # (K, K+m+1)
    bot = torch.cat([Q.transpose(0, 1), torch.zeros(mp1, mp1, dtype=A.dtype)], dim=1)
    M = torch.cat([top, bot], dim=0)                                # (K+m+1, square)
    rhs = torch.cat([rhs_top, torch.zeros(mp1, P, dtype=A.dtype)], dim=0)
    return torch.linalg.solve(M, rhs)


@dataclass(frozen=True)
class RbfFitPlan:
    """Fit-wide geometry shared by every layer over one node layout.

    ``node_params`` is already unit-box normalized.  The kernel/polynomial
    blocks, Demmler-Reinsch eigensystem, λ grid, and (for an exact/fixed-λ
    surface) saddle LU depend only on this geometry—not on a layer's activation
    targets.  Building them once turns the per-layer fit into RHS work instead
    of repeating cubic QR/eigh/factorization for every layer and again for the
    sigma field.
    """

    node_params: torch.Tensor
    coord_offset: torch.Tensor
    coord_scale: torch.Tensor
    E: torch.Tensor
    Q: torch.Tensor
    grid: torch.Tensor
    q2: torch.Tensor
    gamma: torch.Tensor
    eigenvectors: torch.Tensor
    fixed_lambda: float | None
    fixed_lu: torch.Tensor | None
    fixed_pivots: torch.Tensor | None


def prepare_rbf_fit_plan(
    node_params: torch.Tensor,
    *,
    smoothing: float | str | None,
) -> RbfFitPlan:
    """Precompute layout-only RBF work for a multi-layer curved fit."""
    raw = node_params.to(device="cpu", dtype=torch.float32)
    _rbf_poised(raw)
    lo = raw.min(dim=0).values
    hi = raw.max(dim=0).values
    scale = (hi - lo).clamp(min=1e-9)
    normalized = ((raw - lo) / scale).contiguous()
    K = int(normalized.shape[0])
    E = torch.cdist(normalized, normalized).pow(3)
    Q = torch.cat([torch.ones(K, 1, dtype=torch.float32), normalized], dim=1)
    denom = K * K - K
    e_scale = float(E.abs().sum() / denom) if denom > 0 else 1.0
    if not math.isfinite(e_scale) or e_scale <= 0.0:
        e_scale = 1.0
    grid = e_scale * torch.logspace(-6.0, 3.0, 40, dtype=E.dtype)
    mp1 = int(Q.shape[1])
    if K > mp1:
        q_full, _ = torch.linalg.qr(Q, mode="complete")
        q2 = q_full[:, mp1:]
        g = q2.transpose(0, 1) @ E @ q2
        g = 0.5 * (g + g.transpose(0, 1))
        gamma, eigenvectors = torch.linalg.eigh(g)
        gamma = gamma.clamp_min(0.0)
    else:
        q2 = torch.empty(K, 0, dtype=E.dtype)
        gamma = torch.empty(0, dtype=E.dtype)
        eigenvectors = torch.empty(0, 0, dtype=E.dtype)

    fixed_lambda: float | None = None
    if smoothing is None:
        fixed_lambda = 0.0
    elif isinstance(smoothing, (int, float)):
        fixed_lambda = float(smoothing) * e_scale
    fixed_lu: torch.Tensor | None = None
    fixed_pivots: torch.Tensor | None = None
    if fixed_lambda is not None:
        A = E + fixed_lambda * torch.eye(K, dtype=E.dtype)
        mp1 = int(Q.shape[1])
        M = torch.cat([
            torch.cat([A, Q], dim=1),
            torch.cat([
                Q.transpose(0, 1),
                torch.zeros(mp1, mp1, dtype=E.dtype),
            ], dim=1),
        ], dim=0)
        fixed_lu, fixed_pivots = torch.linalg.lu_factor(M)
    return RbfFitPlan(
        node_params=normalized,
        coord_offset=lo,
        coord_scale=scale,
        E=E,
        Q=Q,
        grid=grid,
        q2=q2,
        gamma=gamma,
        eigenvectors=eigenvectors,
        fixed_lambda=fixed_lambda,
        fixed_lu=fixed_lu,
        fixed_pivots=fixed_pivots,
    )


def _rbf_smoother_matrix(
    E: torch.Tensor, Q: torch.Tensor, lam: float,
) -> torch.Tensor:
    """The ``(K, K)`` smoother (hat) matrix ``S_λ`` mapping node values to fits.

    ``ŷ = S_λ y = E w + Q c`` where ``(w, c)`` solve the penalized saddle
    ``[[E+λI, Q],[Qᵀ, 0]] [w; c] = [y; 0]``.  Built by solving the saddle
    against ``[I_K; 0]`` — the columns of the inverse that map ``y`` into
    ``(w, c)`` — then composing ``S = E·M11 + Q·M21``.  ``tr S_λ`` is the
    effective degrees of freedom (``K`` at ``λ=0`` ⇒ exact interpolation;
    ``m+1`` as ``λ→∞`` ⇒ the polynomial trend), and ``I − S_λ`` is the
    residual operator behind GCV / leave-one-out.
    """
    K = E.shape[0]
    A = E + lam * torch.eye(K, dtype=E.dtype)
    eye = torch.eye(K, dtype=E.dtype)
    sol = _rbf_saddle(A, Q, eye)            # (K+m+1, K)
    W = sol[:K]                              # M11  (K, K)
    C = sol[K:]                              # M21  (m+1, K)
    return E @ W + Q @ C                     # (K, K)


def _gcv_select_lambda(
    E: torch.Tensor, Q: torch.Tensor, values: torch.Tensor,
    *, n_grid: int = 40, plan: RbfFitPlan | None = None,
) -> tuple[float, float, float]:
    """Pick the smoothing ``λ`` minimizing generalized cross-validation.

    ``V(λ) = K · ‖(I − S_λ) Y‖²_F / [tr(I − S_λ)]²`` over a log-spaced grid
    scaled by the mean kernel magnitude (so ``λ`` is dimensionless against
    ``E``).  The smoother ``S_λ`` is shared across the ``R`` output columns,
    so the multi-output GCV is the single shared-``S`` form with a Frobenius
    RSS — the ``K`` factor and the (common) ``1/K`` normalizations are
    constants in ``λ`` and don't move the argmin; kept for a comparable
    scalar.  ``λ = 0`` (exact interpolation) is excluded: there ``S = I`` so
    ``tr(I − S) = 0`` and GCV is the indeterminate ``0/0`` — the smallest
    grid ``λ`` is the near-interpolating limit.  Returns
    ``(λ*, edf = tr S_{λ*}, V(λ*))``.

    Demmler–Reinsch form (derived from the same saddle ``_rbf_smoother_matrix``
    solves: ``ŷ = y − λw``, ``w = Q2 z``, ``(G + λI) z = Q2ᵀ y``):
    ``I − S_λ = λ·Q2 (G + λI)⁻¹ Q2ᵀ`` with ``Q2`` an orthonormal basis for
    ``null(Qᵀ)`` (a complete QR of ``Q``) and ``G = Q2ᵀ E Q2`` the reduced
    kernel.  One ``eigh(G) = UΛUᵀ`` collapses every grid point to scalar
    evals of ``γⱼ/(γⱼ+λ)`` — ``tr(I−S_λ) = Σⱼ λ/(γⱼ+λ)`` and
    ``‖(I−S_λ)Y‖²_F = Σ_{j,r} [λ/(γⱼ+λ)]² bⱼᵣ²`` with ``b = Uᵀ Q2ᵀ Y`` — so the
    sweep is one QR + one eigh + vectorized scalars instead of ``n_grid``
    ``(K+m+1)``-saddle solves (the dominant cost of an ``auto`` fit on a large
    heap).  Selection is identical to the smoother-matrix loop.
    """
    K = int(E.shape[0])
    mp1 = int(Q.shape[1])
    null_dim = K - mp1
    if plan is not None and n_grid == 40:
        grid = plan.grid
    else:
        # Scale the grid by the mean off-diagonal kernel magnitude (diag(E) = 0),
        # so the search range is invariant to coordinate scale.
        denom = K * K - K
        e_scale = float(E.abs().sum() / denom) if denom > 0 else 1.0
        if not math.isfinite(e_scale) or e_scale <= 0.0:
            e_scale = 1.0
        grid = e_scale * torch.logspace(-6.0, 3.0, n_grid, dtype=E.dtype)
    if null_dim <= 0:
        # Q full-rank square ⇒ no penalized null space; the polynomial fits
        # every node exactly (S = I) and GCV is the indeterminate 0/0 over the
        # whole grid.  Match the all-skipped loop: smallest λ at interp edf.
        return float(grid[0].item()), float(K), math.inf
    # Q2: an orthonormal basis of null(Qᵀ) from a complete QR of Q.
    if plan is not None:
        q2 = plan.q2
        gamma = plan.gamma
        u = plan.eigenvectors
    else:
        q_full, _ = torch.linalg.qr(Q, mode="complete")    # (K, K)
        q2 = q_full[:, mp1:]                                # (K, null_dim)
        g = q2.transpose(0, 1) @ E @ q2                     # (null_dim, null_dim)
        g = 0.5 * (g + g.transpose(0, 1))                   # symmetrize vs roundoff
        gamma, u = torch.linalg.eigh(g)                     # γⱼ ≥ 0 (cond. PSD)
        gamma = gamma.clamp_min(0.0)
    b = u.transpose(0, 1) @ (q2.transpose(0, 1) @ values)  # (null_dim, R)
    b_sq = b.pow(2).sum(dim=1)                              # Σ_r bⱼᵣ²  (null_dim,)
    # ratios[i, j] = λᵢ / (γⱼ + λᵢ) — the eigenvalues of (I − S_{λᵢ}).
    ratios = grid.unsqueeze(1) / (gamma.unsqueeze(0) + grid.unsqueeze(1))
    tr = ratios.sum(dim=1)                                  # tr(I − S_λ)  (n_grid,)
    rss = (ratios.pow(2) * b_sq.unsqueeze(0)).sum(dim=1)    # ‖(I − S_λ)Y‖²_F
    gcv = torch.where(tr > 0.0, K * rss / (tr * tr), torch.full_like(tr, math.inf))
    best_idx = int(torch.argmin(gcv).item())               # first (smallest λ) on ties
    return (
        float(grid[best_idx].item()),
        float(K) - float(tr[best_idx].item()),             # edf = tr S_{λ*}
        float(gcv[best_idx].item()),
    )


def fit_rbf_smoothed(
    node_params: torch.Tensor,
    values: torch.Tensor,
    *,
    smoothing: float | str | None = "auto",
    plan: RbfFitPlan | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Fit a *penalized* ``r**3`` polyharmonic RBF — the smoothing generalization.

    The thin-plate / Duchon smoothing spline: minimize
    ``‖y − Ew − Qc‖² + λ wᵀEw`` subject to ``Qᵀw = 0``, whose stationarity is
    the penalized saddle::

        [ E + λI   Q ] [w]   [y]
        [ Qᵀ       0 ] [c] = [0]

    ``E_ij = ‖p_i − p_j‖³``, ``Q = [1 | node_params]``.  At ``λ = 0`` this is
    exactly :func:`fit_rbf_interpolant` (the no-penalty / kernel-ridge limit
    ``(E)w + Qc = y``), so the surface interpolates the node values; at
    ``λ > 0`` it *shrinks* toward the affine polynomial trend, trading
    exactness for a smoother surface that doesn't chase noise in the
    centroids.  This is the discover-mode counterpart to authored manifolds'
    exact interpolation (where node = exact steering target is the contract).

    ``smoothing`` is ``"auto"`` (GCV-select ``λ`` over a log grid — the
    default for a noisy discover fit), ``0`` / ``None`` (exact, delegates to
    :func:`fit_rbf_interpolant` for a byte-identical result), or a float
    (a fixed ``λ`` on the mean-kernel-magnitude scale, for advanced control).

    Returns ``(rbf_weights (K, R), poly_coeffs (m+1, R), info)`` where
    ``info`` carries ``{"lambda", "edf", "gcv"}`` (the chosen ``λ``, the
    effective dof ``tr S_λ``, and the GCV score — ``gcv`` is ``-1`` for the
    exact / fixed-``λ`` paths that don't run the search).  The weight shapes
    are identical to :func:`fit_rbf_interpolant`, so :func:`eval_rbf` and the
    steering hot path are unchanged — only the coefficient *values* shrink.

    CPU / fp32: the saddle solve is symmetric-indefinite (MPS-unsafe), and
    this runs once per layer at fit time, off the hot path.
    """
    node_params = node_params.to(device="cpu", dtype=torch.float32)
    values = values.to(device="cpu", dtype=torch.float32)
    # Exact path: delegate so ``λ = 0`` reproduces ``fit_rbf_interpolant``
    # bit-for-bit (the cardinal-weight + interpolation tests pin this).
    if smoothing is None or (isinstance(smoothing, (int, float)) and float(smoothing) == 0.0):
        if plan is not None and plan.fixed_lambda == 0.0:
            assert plan.fixed_lu is not None and plan.fixed_pivots is not None
            mp1 = plan.Q.shape[1]
            rhs = torch.cat([
                values,
                torch.zeros(mp1, values.shape[1], dtype=values.dtype),
            ], dim=0)
            sol = torch.linalg.lu_solve(plan.fixed_lu, plan.fixed_pivots, rhs)
            w, c = sol[:node_params.shape[0]], sol[node_params.shape[0]:]
        else:
            w, c = fit_rbf_interpolant(node_params, values)
        return w, c, {"lambda": 0.0, "edf": float(node_params.shape[0]), "gcv": -1.0}

    K, _ = _rbf_poised(node_params)
    if plan is not None:
        E, Q = plan.E, plan.Q
    else:
        dist = torch.cdist(node_params, node_params)
        E = dist.pow(3)
        Q = torch.cat([torch.ones(K, 1, dtype=torch.float32), node_params], dim=1)

    if smoothing == "auto":
        lam, edf, gcv = _gcv_select_lambda(E, Q, values, plan=plan)
    elif isinstance(smoothing, (int, float)):
        denom = K * K - K
        e_scale = float(E.abs().sum() / denom) if denom > 0 else 1.0
        lam = float(smoothing) * (e_scale if e_scale > 0.0 else 1.0)
        S = _rbf_smoother_matrix(E, Q, lam)
        edf = float(S.diagonal().sum().item())
        gcv = -1.0
    else:
        raise ValueError(
            f"smoothing must be 'auto', a float, or 0/None; got {smoothing!r}"
        )

    if (
        plan is not None
        and plan.fixed_lambda is not None
        and math.isclose(lam, plan.fixed_lambda, rel_tol=0.0, abs_tol=0.0)
    ):
        assert plan.fixed_lu is not None and plan.fixed_pivots is not None
        mp1 = Q.shape[1]
        rhs = torch.cat([
            values,
            torch.zeros(mp1, values.shape[1], dtype=values.dtype),
        ], dim=0)
        sol = torch.linalg.lu_solve(plan.fixed_lu, plan.fixed_pivots, rhs)
    else:
        A = E + lam * torch.eye(K, dtype=torch.float32)
        sol = _rbf_saddle(A, Q, values)
    w, c = sol[:K].contiguous(), sol[K:].contiguous()
    return w, c, {"lambda": float(lam), "edf": float(edf), "gcv": float(gcv)}


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


def rbf_cardinal_weights(
    node_coords: torch.Tensor, query: torch.Tensor,
) -> torch.Tensor:
    """Cardinal ``r**3``-RBF interpolation weights ``w(z) ∈ ℝ^K`` over a layout.

    Given ``node_coords`` ``(K, n)`` -- a manifold's authoring node layout --
    and a ``query`` ``(n,)``, return weights ``(K,)`` such that for *any*
    per-node values ``Y`` ``(K, R)`` the ``r**3``-polyharmonic RBF interpolant
    of ``Y`` evaluated at ``query`` equals ``w @ Y``.  The weights are the
    layer-agnostic cardinal functions of the layout: they depend only on the
    node coordinates and the query, not on what is interpolated, so one solve
    serves every layer.  They are **exact at the nodes** (``w = e_i`` at node
    ``i``, since RBF interpolation reproduces the sampled values) and form a
    partition of unity (``Σ w = 1``, since the affine polynomial reproduces
    constants).

    This is the flat-manifold coord-form analogue of a curved fit's RBF
    surface: applying ``w`` to a flat fit's per-layer real reduced node coords
    reproduces label-form steering at the nodes and interpolates the per-layer
    target between them off-node — staying within the flat subspace while
    following the learned layout rather than a straight chord.

    Computed on CPU / fp32 (the indefinite saddle solve in
    :func:`fit_rbf_interpolant` is MPS-unsafe and this runs once per steering
    compose, off the hot path).  Unit-box-normalized for kernel conditioning,
    matching :func:`fit_layer_subspace`.  Propagates the ``ValueError`` from
    :func:`fit_rbf_interpolant` when the layout is not affinely poised (no
    interpolant exists — the caller re-raises it as ``SteeringExprError``
    advising the user to steer by node label instead).
    """
    node_coords = node_coords.detach().to(device="cpu", dtype=torch.float32)
    query = query.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    lo = node_coords.min(dim=0).values
    hi = node_coords.max(dim=0).values
    scale = (hi - lo).clamp(min=1e-9)
    nc_norm = (node_coords - lo) / scale
    q_norm = (query - lo) / scale                       # (n,)
    eye = torch.eye(node_coords.shape[0], dtype=torch.float32)
    rbf_weights, poly_coeffs = fit_rbf_interpolant(nc_norm, eye)
    return eval_rbf(nc_norm, rbf_weights, poly_coeffs, q_norm)  # (K,)


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
    sigma_rbf_weights: torch.Tensor | None = None
    sigma_poly_coeffs: torch.Tensor | None = None
    # The **fuzzy-manifold σ-field** (curved subspaces only; ``None`` =
    # zero-thickness wire = legacy behavior).  A *separate* ``r**3`` RBF over
    # the **same** normalized ``node_params`` that interpolates per-node
    # ``log σ`` — the within-node off-surface activation spread (the corpus a
    # node produces scatters off the mean surface; ``σ`` is that scatter's
    # normal-projected std, a tube thickness).  Kept separate from the surface
    # RBF (rather than appended as an extra value column) so the ``(R,)``-shape
    # contracts the surface consumers rely on are untouched; ``sigma_at`` is the
    # one extra ``eval_rbf`` (``O(K)``) paid only on the already-slow curved
    # path.  ``sigma_rbf_weights`` is ``(K, 1)``, ``sigma_poly_coeffs`` is
    # ``(m+1, 1)``.  ``None`` (affine, legacy, or a curved fit predating the
    # σ-field) ⇒ ``sigma_at`` returns ``0`` ⇒ soft-onto degenerates to the hard
    # collapse and the read bandwidth degenerates to argmax — exact legacy.

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
        return cls(
            mean=mean,
            basis=basis,
            node_params=None,
            rbf_weights=None,
            poly_coeffs=None,
            coord_offset=torch.zeros(r, device=mean.device, dtype=mean.dtype),
            coord_scale=torch.ones(r, device=mean.device, dtype=mean.dtype),
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
            sigma_rbf_weights=_cast(self.sigma_rbf_weights),
            sigma_poly_coeffs=_cast(self.sigma_poly_coeffs),
        )

    @property
    def has_sigma(self) -> bool:
        """True iff this subspace carries a fuzzy-manifold σ-field.

        A curved subspace fitted with the within-node spread pass; ``False``
        for affine fits, legacy curved fits, and any subspace where the σ-RBF
        wasn't built.  Gates :meth:`sigma_at` (which returns ``0`` when absent)
        so every σ consumer degrades to the exact zero-thickness behavior.
        """
        return self.sigma_rbf_weights is not None and self.sigma_poly_coeffs is not None

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

    def sigma_at(self, embedded: torch.Tensor) -> torch.Tensor:
        """Within-node off-surface spread ``σ`` at embedded coords ``(.., m)``.

        Interpolates the per-node ``log σ`` field through the σ-RBF over the
        same normalized ``node_params``, then exponentiates — so the result is
        a positive ``(..,)`` thickness in the layer's reduced-coordinate units
        (the same units ``H_n`` is measured in, since the basis is
        orthonormal).  Returns an all-zeros ``(..,)`` when the subspace carries
        no σ-field (:attr:`has_sigma` false): affine fits, legacy curved fits,
        the degenerate-but-safe path that makes every σ consumer exact-legacy.
        Hot-path safe (one extra ``eval_rbf``, no ``.item()`` / host sync).
        """
        lead = embedded.shape[:-1]
        sw, sp, np_ = self.sigma_rbf_weights, self.sigma_poly_coeffs, self.node_params
        if sw is None or sp is None or np_ is None:
            return torch.zeros(lead, device=embedded.device, dtype=embedded.dtype)
        dev, dt = embedded.device, embedded.dtype
        norm = (embedded - self.coord_offset.to(dev, dt)) / self.coord_scale.to(dev, dt)
        log_sigma = eval_rbf(
            np_.to(dev, dt), sw.to(dev, dt), sp.to(dev, dt), norm,
        )  # (.., 1)
        return torch.exp(log_sigma).squeeze(-1)  # (..,)

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
    whitened_gram: torch.Tensor | None = None,
    whitened_rows: torch.Tensor | None = None,
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

    ``whitened_gram`` may provide a precomputed ``X Σ⁻¹ Xᵀ`` for fit
    callers that already built the same Gram for diagnostics or discover
    coordinate derivation. ``whitened_rows`` is the matching precomputed
    ``Σ⁻¹X`` row batch; when both are supplied the Fisher directions are
    ``Aᵀ(Σ⁻¹X)`` directly, so Gram construction, PCA, and neutral-layout
    anchoring share one Woodbury application over the node scatter.
    """
    K = int(X.shape[0])
    if whitener is not None and layer is not None:
        # Whitened (Fisher) PCA — generalized eigenproblem (S_b, Σ) via the
        # whitener's low-rank Woodbury Σ⁻¹.  ``G = X Σ⁻¹ Xᵀ`` is K×K; its
        # eigvecs ``a`` give the discriminant directions ``v_r = Σ⁻¹ Xᵀ a_r``.
        G = (
            whitened_gram.to(dtype=torch.float32, device="cpu")
            if whitened_gram is not None
            else whitener.subspace_gram(layer, X)
        )                                               # (K, K) = X Σ⁻¹ Xᵀ
        if G.shape != (K, K):
            raise ValueError(
                f"whitened_gram shape {tuple(G.shape)} does not match "
                f"centered scatter shape ({K}, {K})"
            )
        mu, A = torch.linalg.eigh(G)                    # ascending
        mu_pos = mu.clamp_min(0.0)
        rank = int((mu_pos > 1e-6 * mu_pos[-1].clamp(min=1e-12)).sum().item())
        R = max(1, min(n_components, K - 1, rank))
        top = torch.argsort(mu, descending=True)[:R]
        if whitened_rows is not None:
            sinv_x = whitened_rows.to(dtype=torch.float32, device="cpu")
            if sinv_x.shape != X.shape:
                raise ValueError(
                    f"whitened_rows shape {tuple(sinv_x.shape)} does not match "
                    f"centered scatter shape {tuple(X.shape)}"
                )
            directions = A[:, top].transpose(0, 1) @ sinv_x
        else:
            XtA = X.transpose(0, 1) @ A[:, top]         # (D, R) = Xᵀ a_r
            directions = whitener.apply_inv(
                layer, XtA.transpose(0, 1).contiguous(),
            )                                           # (R, D) = Σ⁻¹ Xᵀ a_r
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
    whitened_gram: torch.Tensor | None = None,
    whitened_rows: torch.Tensor | None = None,
    orient_to: int | None = 0,
    basis_override: torch.Tensor | None = None,
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
    if basis_override is None:
        basis, ev_ratio = _pca_basis(
            X, n_components=n_components, whitener=whitener, layer=layer,
            whitened_gram=whitened_gram, whitened_rows=whitened_rows,
        )
    else:
        basis = basis_override[: min(n_components, basis_override.shape[0])].to(
            device="cpu", dtype=torch.float32,
        ).contiguous()
        ev_ratio = 1.0  # diagnostic is unused by planned pipeline callers
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
    whitened_gram: torch.Tensor | None = None,
    whitened_rows: torch.Tensor | None = None,
    smoothing: float | str | None = None,
    rbf_info: dict[str, float] | None = None,
    rbf_plan: RbfFitPlan | None = None,
    basis_override: torch.Tensor | None = None,
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
    if basis_override is None:
        basis, ev_ratio = _pca_basis(
            X, n_components=n_components, whitener=whitener, layer=layer,
            whitened_gram=whitened_gram, whitened_rows=whitened_rows,
        )
    else:
        basis = basis_override[: min(n_components, basis_override.shape[0])].to(
            device="cpu", dtype=torch.float32,
        ).contiguous()
        ev_ratio = 1.0  # diagnostic is unused by planned pipeline callers
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

    plan = rbf_plan or prepare_rbf_fit_plan(
        node_params, smoothing=smoothing,
    )
    coord_offset = plan.coord_offset
    coord_scale = plan.coord_scale
    normalized = plan.node_params

    # Exact interpolation by default (``smoothing=None``) — every existing
    # caller (authored fits, the behavior-manifold naturalness fit, the test
    # suite) keeps the byte-identical interpolant.  ``smoothing`` set (the
    # discover ``spectral`` path) opts into the penalized fit; the chosen
    # ``λ``/edf flow back through the optional ``rbf_info`` out-dict for the
    # sidecar, leaving the 2-tuple return arity untouched.
    if smoothing is None:
        rbf_weights, poly_coeffs, _ = fit_rbf_smoothed(
            normalized, coords, smoothing=0.0, plan=plan,
        )
    else:
        rbf_weights, poly_coeffs, _info = fit_rbf_smoothed(
            normalized, coords, smoothing=smoothing, plan=plan,
        )
        if rbf_info is not None:
            rbf_info.update(_info)
    sub = LayerSubspace(
        # Geometry is shared through ``rbf_plan`` while fitting, but each layer's
        # persistent payload owns these tiny tensors: safetensors deliberately
        # rejects aliases across keys.
        mean=mean, basis=basis, node_params=normalized.clone(),
        rbf_weights=rbf_weights, poly_coeffs=poly_coeffs,
        coord_offset=coord_offset.clone(), coord_scale=coord_scale.clone(),
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
    # Per-layer Mahalanobis share weight recorded at fit time when a
    # whitener was available — ``share_L = ‖Bᵀ coords_k‖_M`` summed over
    # nodes, the subspace-restricted analogue of vector steering's
    # ``‖d‖_M`` bake score (see ``LayerWhitener.subspace_gram``).  When
    # populated *and* covering every layer, ``hooks._manifold_layer_shares``
    # uses it in place of the Euclidean centroid-spread; an empty dict
    # (no whitener at fit time, e.g. CPU test stubs, or partial layer
    # coverage) falls back to the Euclidean ``‖coords‖_F`` weighting.
    # These are raw per-layer scalars with two normalized consumers: the
    # apply-time **steer** weight (normalized to mean 1, ``Σ_L share_L =
    # n_layers``) in ``_manifold_layer_shares``, and the **read** weight
    # (normalized to sum 1) the unified ``Monitor`` uses to combine each
    # layer's geometry into one cross-layer reading — the layer carrying
    # the most steering budget is also the most reliable to read from, so
    # one quantity drives both sides.
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
            pos = cast(torch.Tensor, position).to(device=device, dtype=dtype).reshape(-1)
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
            tensor_position = cast(torch.Tensor, position)
            return tuple(float(c) for c in tensor_position.reshape(-1).tolist())
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


# Per-position guard for the principal-angle residual transport in
# :func:`_frame_rotation_transport`: a principal plane whose ``sin θ`` is below
# this is treated as a no-op rotation (its in-plane direction ``nᵢ`` is a 0/0),
# which is also exactly the ``p_new == p`` identity-at-rest case.
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
    the read-side ``Monitor``.  All
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
    (push + ablation).  The ``along`` step **translates** the foot by the
    ``target_coord`` offset on each *push* axis (preserving the per-token
    spread → coherent strong steer) and **collapses** the foot to ``0`` on each
    *ablation* axis (removing the ablated component).  ``kappa`` is the per-axis
    blend that selects which is which (``0`` push / translate, ``1`` ablate /
    collapse): the kernel applies ``target − κ⊙q``, so push axes ignore ``q``
    (fixed offset) while ablation axes drive ``q`` toward ``0``.  ``share`` is the
    un-normalized per-layer budget weight — the push-displacement magnitude
    ``‖Δ_L‖_M`` (whitened) when a covering ``whitener`` is supplied, else the
    raw-Euclidean ``‖Δ_L‖₂``; a pure-ablation layer weights by the summed
    ablation magnitude instead.  The apply path normalizes it across layers
    (mean-1).  ``target_coord`` is correspondingly a whitened-unit direction on
    the whitened path (magnitude carried by ``share``) or the raw reduced
    displacement on the Euclidean fallback.

    Fields are keyed by layer index; only layers carrying at least one
    non-degenerate active direction (and present in ``neutral_means``) appear.
    """

    layers: dict[int, "LayerSubspace"]        # affine: mean = neutral_L, basis = ortho span
    target_coord: dict[int, torch.Tensor]     # (R_L,) the along target (poles / 0)
    share: dict[int, float]                   # ‖Δ_L‖, un-normalized budget weight
    kappa: dict[int, torch.Tensor] = field(   # (R_L,) per-axis: 0 push (translate), 1 ablate (collapse)
        default_factory=dict
    )


def synthesize_subspace(
    push: Sequence[
        tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], float]
    ],
    ablate: Sequence[dict[int, torch.Tensor]],
    neutral_means: dict[int, torch.Tensor],
    *,
    whitener: "LayerWhitener | None" = None,
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
      orthogonal complement, those axes get ``target ≈ 0``.
    - Per-axis collapse mask ``kappa`` ``(R,)`` — ``0`` on the push span, ``1`` on
      the ablation-only complement (``κ = 1 − ‖proj onto push span‖²``).  The
      kernel *translates* push axes by the fixed offset but *collapses* ``κ=1``
      axes toward 0 (``p_new = q + a·(target − κ·q)``): post translate-not-
      collapse a ``target ≈ 0`` alone no longer removes an ablated direction (a
      pure translate by 0 is a no-op), so ``κ`` is what carries the ablation.
    - ``share = ‖Δ‖`` (the world displacement magnitude); a pure-ablation layer
      weights by the summed ablation-row magnitude instead.

    **Whitened normalization (``whitener`` given).**  The push magnitude used to
    be the *raw-Euclidean* node displacement ``‖Δ_L‖₂``, which is not a
    scale-stable steering unit: it is the un-whitened (rogue-dominated) distance
    from neutral to the target node, and it spans ~100× across targets purely by
    where each centroid happens to sit (a tight bipolar pole sits ~0.3 from
    neutral, a far persona centroid ~17), so a single ``along`` gain over-pushed
    far nodes into incoherence and left near ones doing nothing.  When the
    ``whitener`` covers every synthesized layer the push is instead normalized in
    the Mahalanobis metric ``M_R = B Σ⁻¹ Bᵀ`` (the engine-wide read/fit metric):

    - ``share = ‖Δ‖_M`` (whitened displacement) — the cross-layer profile weights
      by *whitened* signal, matching the baked ``mahalanobis_share`` rather than
      the raw activation distance.
    - ``target = (B @ Δ) / ‖Δ‖_M`` — a **whitened-unit** direction (``‖B@target‖_M
      = 1``), so the apply path's ``eff_along_L = mean1(share)·gain`` puts the same
      *whitened* slide on every target.  The push still aims at the node centroid
      (whitening a direction toward a fixed point is metric-invariant — only the
      calibration changes), it is just measured in std-units instead of the
      raw-Euclidean scale.  Every target then receives one uniform whitened
      budget (``Σ_L eff_along_L = gain·n_layers``), distributed across layers by
      where its signal lives; ``along`` becomes a scale-stable strength knob.

    This is all-or-nothing (Mahalanobis-only): a partially-covering / absent
    whitener falls back to the raw-Euclidean ``‖Δ‖₂`` path below (CPU stubs,
    degenerate fits), never mixing the two metrics across layers within one
    steer (a mixed cross-layer profile would be meaningless under the mean-1
    normalization).

    Because the basis is orthonormal, ``‖target‖₂ = ‖Δ‖₂`` on the Euclidean
    fallback (the per-layer budget weight and the steered coordinate sit on one
    scale); the whitened path instead decouples them — ``share`` carries the
    per-layer magnitude, ``target`` carries only the unit direction.

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

    # All-or-nothing whitened normalization: only when the whitener covers every
    # layer that will actually be synthesized (present in ``neutral_means``).  A
    # mixed whitened/Euclidean cross-layer profile is meaningless under the
    # apply-time mean-1 share normalization, so gate the whole synth on one
    # ``covers_all`` rather than per-layer.
    present_layers = sorted(L for L in all_layers if L in neutral_means)
    maha = (
        whitener
        if whitener is not None
        and present_layers
        and whitener.covers_all(present_layers)
        else None
    )

    layers: dict[int, "LayerSubspace"] = {}
    target_coord: dict[int, torch.Tensor] = {}
    share: dict[int, float] = {}
    kappa: dict[int, torch.Tensor] = {}

    for L in sorted(all_layers):
        if L not in neutral_means:
            continue
        mean = neutral_means[L].to(torch.float32).reshape(-1)

        # Push fragments present at this layer: their basis rows (for the span)
        # and per fragment the ``(coeff, world_dir)`` pair — ``world_dir = coords
        # @ basis`` is the raw (coeff-free) neutral→node displacement; ``coeff``
        # is kept *separate* so the whitened path can unit-normalize the
        # direction (strip the node's raw-Euclidean distance) while still scaling
        # by the user strength.
        push_rows: list[torch.Tensor] = []          # individual (D,) basis rows
        push_frags: list[tuple[float, torch.Tensor]] = []   # (coeff, world_dir (D,))
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
            push_frags.append((float(coeff), c_i @ B_i))    # (D,) raw world dir

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

        if push_frags:
            # Raw coeff-weighted displacement — its (whitened) magnitude is the
            # per-layer **profile** weight (``share``); the absolute node-distance
            # scale cancels under the apply-time mean-1 normalization, leaving
            # only the relative across-layer shape (steer where the signal is).
            raw_delta = torch.stack([cf * wd for cf, wd in push_frags]).sum(0)
            if maha is not None:
                share_L = float(maha.mahalanobis_norm(L, raw_delta))
                # Target = Σ_i coeff_i · (B @ world_dir_i)/‖world_dir_i‖_M — each
                # fragment a **whitened-unit** direction (node-distance stripped,
                # so scale-stable across targets) scaled by its user coeff (the
                # strength knob the mean-1 ``share`` would otherwise cancel).
                tc = basis.new_zeros(basis.shape[0])
                for cf, wd in push_frags:
                    wn = float(maha.mahalanobis_norm(L, wd))
                    if wn > eps:
                        tc = tc + (cf / wn) * (basis @ wd)
                if float(torch.linalg.vector_norm(tc)) < eps:
                    # Every fragment degenerate in the whitened metric here —
                    # fall back to the raw reduced target so the layer still steers.
                    tc = basis @ raw_delta
                    share_L = float(torch.linalg.vector_norm(raw_delta))
            else:
                tc = basis @ raw_delta                    # Euclidean fallback
                share_L = float(torch.linalg.vector_norm(raw_delta))
            target_coord[L] = tc                          # ablation axes ≈ 0
        else:
            ablate_sum = torch.stack(ablate_raw).sum(0)
            share_L = (
                float(maha.mahalanobis_norm(L, ablate_sum))
                if maha is not None
                else float(torch.linalg.vector_norm(ablate_sum))
            )
            target_coord[L] = basis.new_zeros(basis.shape[0])   # (R,) all ≈ 0
        # Per-axis collapse weight κ: 0 on the push span (translate — preserve
        # the per-token in-subspace spread), 1 on the ablate-only complement
        # (collapse the component to 0).  Derived by projecting each merged-basis
        # axis onto the push span (``κ = 1 − ‖proj‖²``), so it is robust to the
        # orthonormalization order.  Pure push → all 0; pure ablation → all 1.
        if push_rows and ablate_rows:
            B_push, _ = _ortho_basis(push_rows)             # (k_push, D)
            proj = basis @ B_push.T                          # (R, k_push)
            kappa[L] = (1.0 - (proj * proj).sum(-1)).clamp(0.0, 1.0)  # (R,)
        elif ablate_rows:
            kappa[L] = basis.new_ones(basis.shape[0])
        else:
            kappa[L] = basis.new_zeros(basis.shape[0])
        layers[L] = LayerSubspace.affine(mean=mean, basis=basis)
        share[L] = share_L

    return SynthesizedSubspace(
        layers=layers, target_coord=target_coord, share=share, kappa=kappa,
    )


def _soft_norm_cap(
    h_new: torch.Tensor, h_f32: torch.Tensor, norm_cap: float,
) -> torch.Tensor:
    """Soft cap ``‖h_new‖ ≤ norm_cap·‖h‖`` — the off-domain RBF-extrapolation
    blowup guard shared by the affine shortcut and the curved injection path.

    Operates **in place** on ``h_new`` (returning it for call-site clarity):
    both call sites pass a freshly-allocated sum (``h_f32 + Δ`` / ``mean +
    new_par + new_perp``), never an alias of the function input ``h``, so the
    in-place ``mul_`` by a ``(.., 1)`` per-row scale is safe.

    The scale is ``min(1, norm_cap·‖h‖ / ‖h_new‖)`` expressed as a single
    ``clamp(max=1.0)`` on the ratio — when ``‖h_new‖ ≤ cap`` the ratio is ``≥1``
    and clamps to a no-op ``1``; when it overshoots, the ratio is the shrink
    factor ``cap/‖h_new‖``.  This is identical to the old
    ``where(post > cap, cap/post, 1)`` form on every non-degenerate ``h`` but
    skips both the full-width ``torch.ones_like`` temporary *and* the
    ``torch.where`` select the guard used to allocate every fire (only the
    all-zero-``h`` corner differs — scale 0 vs 1 — and there ``h_new ≈ 0`` so
    the product is ``0`` either way)."""
    norm_pre = torch.linalg.vector_norm(h_f32, dim=-1, keepdim=True)
    norm_post = torch.linalg.vector_norm(h_new, dim=-1, keepdim=True)
    scale = (
        norm_cap * norm_pre / norm_post.clamp(min=1e-6)
    ).clamp_(max=1.0)                              # (.., 1) — not full-width
    return h_new.mul_(scale)


def _orthonormalize_columns(
    m: torch.Tensor, *, eps: float = _ROTATE_EPSILON,
) -> torch.Tensor:
    """Modified Gram-Schmidt orthonormalization of the columns of ``m`` (.., R, n).

    Returns ``(.., R, n)`` with orthonormal columns spanning the same range.
    Pure matmul / elementwise — **no** ``torch.linalg.qr``, which is
    unimplemented on the MPS backend (no autograd-fallback either), so this runs
    natively on every device.  ``n`` is the small intrinsic dim (≤ a handful),
    so the Python loop is cheap.  A column that collapses to ~0 after
    orthogonalization (a rank-deficient frame at a fold) is zeroed rather than
    amplified by the norm division — the downstream principal-angle SVD treats
    the resulting zero overlap row as a 90° angle, i.e. no transport in it.
    """
    cols: list[torch.Tensor] = []
    for i in range(m.shape[-1]):
        v = m[..., i]                                       # (.., R)
        for u in cols:
            v = v - (u * v).sum(dim=-1, keepdim=True) * u
        norm = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
        v = torch.where(norm > eps, v / norm.clamp(min=eps), torch.zeros_like(v))
        cols.append(v)
    return torch.stack(cols, dim=-1)                        # (.., R, n)


def _svd_mps_safe(
    a: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``torch.linalg.svd`` with an explicit CPU hop on MPS.

    ``aten::linalg_svd`` is unimplemented on Metal and silently CPU-falls-back
    with a per-call warning; doing the hop explicitly suppresses the warning and
    keeps it a tiny ``(.., n, n)`` round-trip (``n`` = intrinsic dim).  Native on
    CUDA / CPU.
    """
    if a.device.type == "mps":
        u, s, vh = torch.linalg.svd(a.cpu())
        return u.to(a.device), s.to(a.device), vh.to(a.device)
    return torch.linalg.svd(a)


def _frame_rotation_transport(
    Hn: torch.Tensor,      # (.., R) off-surface residual to transport
    j_old: torch.Tensor,   # (.., R, n) tangent at the old foot p
    j_new: torch.Tensor,   # (.., R, n) tangent at the new foot p_new
) -> torch.Tensor:
    """Transport ``Hn`` from the tangent frame at the old foot to the frame at
    the new foot by the minimal (principal-angle) orthogonal rotation between
    the two tangent subspaces.

    The rotation maps ``span(j_old) → span(j_new)`` (and hence their normal
    complements) by rotating each pair of principal vectors ``aᵢ → bᵢ`` in its
    own plane, identity on the orthogonal complement of both frames.  Two
    properties this guarantees, which the former project-onto-normal + renorm
    violated:

    - **Identity at rest.**  When the foot doesn't move (``p_new == p``, i.e.
      ``along == 0``) the frames coincide, every principal angle is 0, and the
      rotation is *exactly* the identity — so ``subspace_inject`` returns its
      input untouched regardless of foot-solve accuracy.  (The old code
      reprojected ``Hn`` onto the normal space every fire, corrupting any
      off-neutral activation by the residual's tangential part — which never
      vanishes at an approximate foot — independent of the slide.)
    - **No information loss.**  A rotation preserves ``‖Hn‖`` and discards no
      component; the residual's full content rides the frame as it turns.

    All ``O(R·n)`` per position, no ``(R, R)`` matrix materialized.  fp32.
    """
    # Orthonormal bases of the two tangent spaces (basis-choice-arbitrary, but
    # the principal-angle SVD below is invariant to that choice).  Modified
    # Gram-Schmidt, not ``torch.linalg.qr`` — the latter is unimplemented on MPS.
    qa = _orthonormalize_columns(j_old)            # (.., R, n)
    qb = _orthonormalize_columns(j_new)            # (.., R, n)
    n = qa.shape[-1]
    if n == 1:
        # --- on-device n=1 closed form (the common curved topology: a 1-D
        # ring/arc) ----------------------------------------------------------
        # A 1×1 overlap ``[c0] = qaᵀqb`` has a trivial SVD: the singular value is
        # ``|c0|`` and the sign rides into U/V.  Rather than route a 1×1 matrix
        # through ``_svd_mps_safe`` (which hops to CPU on MPS every fire — the
        # whole point of this branch is to stay on-device), build the principal
        # directions / cosine directly: pick ``b = ±qb`` so the cosine is
        # ``c = |c0| ≥ 0`` (exactly what SVD's non-negative singular value gives,
        # the sign absorbed into V).  ``a = qa``, ``b = sign(c0)·qb`` then
        # satisfy ``aᵀb = |c0| = c`` — bit-for-bit the (a, b, c) the SVD path
        # produces at n=1, so the shared planar-rotation tail below is identical.
        c0 = (qa * qb).sum(dim=-2, keepdim=True)   # (.., 1, 1) = qaᵀqb
        sign = torch.where(
            c0 >= 0.0, torch.ones_like(c0), -torch.ones_like(c0),
        )                                          # (.., 1, 1) sign(c0), +1 at 0
        a = qa                                     # (.., R, 1) principal dir in span(qa)
        b = qb * sign                              # (.., R, 1) flipped so aᵀb ≥ 0
        c = c0.squeeze(-2).abs().clamp(0.0, 1.0)   # (.., 1) cos θ = |c0|
    else:
        # Principal angles between the subspaces: SVD of the n×n frame overlap.
        # Σ = cos θᵢ; the principal directions pair up orthogonally across i
        # (aᵢᵀbⱼ = cos θᵢ · δᵢⱼ), so the subspace rotation is a product of
        # *independent* planar rotations aᵢ → bᵢ.  n≥2 keeps the CPU-hopped SVD
        # (``linalg.svd`` is unimplemented on Metal); only the tiny n=1 case
        # (the dominant curved topology) earned a hand-written on-device path.
        u, s_cos, vh = _svd_mps_safe(qa.transpose(-1, -2) @ qb)  # (..,n,n),(..,n),(..,n,n)
        a = qa @ u                                 # (.., R, n) principal dirs in span(qa)
        b = qb @ vh.transpose(-1, -2)              # (.., R, n) principal dirs in span(qb)
        c = s_cos.clamp(-1.0, 1.0)                 # (.., n) cos θᵢ
    s = (1.0 - c * c).clamp(min=0.0).sqrt()        # (.., n) sin θᵢ
    # nᵢ = unit(bᵢ − cᵢ aᵢ): the in-(aᵢ,bᵢ)-plane direction ⊥ aᵢ that aᵢ rotates
    # toward (‖bᵢ − cᵢ aᵢ‖ = sᵢ).  Guard the sᵢ≈0 (no-rotation) planes — their
    # nᵢ is a 0/0 and their planar rotation is the identity anyway.
    perp = b - c.unsqueeze(-2) * a                 # (.., R, n)
    perp_norm = torch.linalg.vector_norm(perp, dim=-2, keepdim=True)  # (.., 1, n)
    n_dir = perp / perp_norm.clamp(min=_ROTATE_EPSILON)  # (.., R, n) unit nᵢ
    # Hn's components in each plane and the planar-rotation change
    # (aᵢ → cᵢaᵢ + sᵢnᵢ, nᵢ → −sᵢaᵢ + cᵢnᵢ):
    #   Δ = Σᵢ [(cᵢ−1)αᵢ − sᵢβᵢ] aᵢ + [sᵢαᵢ + (cᵢ−1)βᵢ] nᵢ.
    alpha = (a * Hn.unsqueeze(-1)).sum(dim=-2)     # (.., n)  aᵢᵀ Hn
    beta = (n_dir * Hn.unsqueeze(-1)).sum(dim=-2)  # (.., n)  nᵢᵀ Hn
    active = (s > _ROTATE_EPSILON).to(Hn.dtype)    # (.., n) skip sᵢ≈0 planes
    d_a = ((c - 1.0) * alpha - s * beta) * active  # (.., n)
    d_n = (s * alpha + (c - 1.0) * beta) * active  # (.., n)
    delta = (
        (a * d_a.unsqueeze(-2)).sum(dim=-1) + (n_dir * d_n.unsqueeze(-2)).sum(dim=-1)
    )                                              # (.., R)
    return Hn + delta


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
    mean_proj: "torch.Tensor | None" = None,
    origin: "torch.Tensor | None" = None,
    kappa: "float | torch.Tensor" = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The unified two-operation manifold injection (replaces angular/additive).

    Decomposes ``h = mean + H_m + H_n + H_o`` against the layer's affine
    subspace and its RBF surface ``M``, then applies two near-orthogonal
    operations, each a coefficient in ``[0, 1]``:

    - **along** (``a``): **translate** the projected foot ``p*`` by the fixed
      neutral→target offset (scaled by ``a``) via
      :meth:`ManifoldDomain.translate_foot` — every token's foot shifts by the
      same displacement, which preserves the per-token in-subspace spread and
      keeps strong steer coherent (collapsing each foot onto the absolute target
      instead erases that spread and degenerates into looping).  Then transport
      the off-manifold residual ``H_n`` from the tangent frame at the old foot to
      the frame at the new foot by the minimal orthogonal (principal-angle)
      rotation between the two tangent subspaces
      (:func:`_frame_rotation_transport`) — *exactly* the identity when the foot
      didn't move, so the curved path is identity at ``along == 0`` regardless of
      foot-solve accuracy; norm-preserving and lossless (the former
      project-onto-normal + renorm discarded the residual's tangential part every
      fire, corrupting any off-neutral activation independent of the slide).
      Tangential / directional; leaves ``H_o`` untouched.  By moving *on the
      surface* it never cuts through the off-manifold low-density region.

    ``origin`` is the neutral foot — the translate reference; it is coord 0 for
    an affine fit, so the affine path ignores it.  ``kappa`` is the per-axis
    collapse blend on the affine path: scalar ``0.0`` ⇒ pure translate, or a
    ``(R,)`` mask (from :func:`synthesize_subspace`) whose ``κ=1`` ablation axes
    collapse toward 0 while ``κ=0`` push axes translate.  (Curved manifolds are
    push-only, so they take the scalar-0 / pure-translate :meth:`translate_foot`.)
    - **onto** (``o``): collapse the off-surface residual ``H_n`` toward the
      surface within the subspace.  With no σ-field (zero-thickness wire) this
      scales ``H_n`` by ``(1 − o)`` — at ``o = 1`` the activation lands on the
      mean surface.  With a fuzzy-manifold σ-field (:meth:`LayerSubspace.sigma_at`)
      it instead shrinks ``‖H_n‖`` toward the local within-node thickness
      ``σ(z)`` (the *tube*), so ``o = 1`` lands one-σ off the wire — a sample-like
      point on the surface's typical set — direction preserved and never
      expanding a residual already inside the tube.  Vacuous when the surface
      fills its subspace (``H_n ≈ 0``, every affine term).

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

    if subspace.is_affine:
        # --- flat (folded-vector) shortcut --------------------------------
        # The surface fills the subspace, so reduced coords *are* authoring
        # coords (identity map): the foot is ``q`` exactly, ``H_n ≡ 0`` and
        # ``onto`` is vacuous (ignored).  No GN solve, no RBF eval, no
        # tangent Gram-solve — the cost the common steering case can't pay.
        #
        # ``q = (h − mean)·basisᵀ = h·basisᵀ − mean·basisᵀ`` is computed
        # **without** the full-width ``centered = h − mean`` temporary: the
        # ``(R,)`` ``mean_proj = mean·basisᵀ`` is a tiny reduced-space constant
        # (one dot product at R=1).  Hot callers (the single-affine fast hook)
        # precompute it once at ``recompose`` and thread it in; everyone else
        # gets the inline ``mean @ basis.T``.  ALONG translates ``q`` by the
        # fixed offset ``along·target``; the off-subspace residual ``H_o`` is
        # kept verbatim (``h_new = h + Δ·basis`` is a fresh tensor, never an
        # in-place mutation of ``h``).
        mp = mean_proj if mean_proj is not None else (mean @ basis.T)  # (R,)
        q = h_f32 @ basis.T - mp                        # (.., R)
        # Per-axis κ-blend of the foot toward ``target`` (``origin`` is span-coord
        # 0 in the ν-anchored affine frame; an affine fit is always on the flat
        # ``CustomDomain`` — identity clamp — so this analytic is exact):
        #   push axis   (κ=0): ``q + along·target``        — translate by the
        #     fixed offset, preserving the per-token in-subspace spread (coherent);
        #   ablate axis (κ=1): ``q + along·(0 − q)``       — collapse the
        #     component toward 0 (remove the ablated direction).
        # ``kappa`` is a scalar (``0.0`` ⇒ pure translate) or a per-axis ``(R,)``
        # mask from ``synthesize_subspace`` (0 push / 1 ablate).
        p_new = q + along * (target - kappa * q)        # (.., n==R)
        h_new = h_f32 + ((p_new - q) @ basis)           # keep H_o verbatim
        # No ``_soft_norm_cap`` on the affine branch.  The cap only ever guarded
        # off-domain RBF *extrapolation* blowup; a flat affine fit has no RBF
        # surface (``clamp_position`` is identity on the flat ``CustomDomain``),
        # so ``p_new`` is always in-frame and the displacement ``(p_new − q)@basis``
        # is a bounded steering offset added to a large-norm residual stream — it
        # cannot plausibly push ``‖h_new‖`` past ``3·‖h‖``.  This is the same
        # reasoning (and now the same behavior) as the constant-add fast path in
        # ``SteeringHook._pure_push_constant``, which already drops the cap; the
        # mixed push+ablate (κ≠0) affine term only *shrinks* the ablated axes, so
        # it can't grow the norm past the pure-push case either.  Dropping it
        # removes the two per-fire full-width norm reductions on the affine kernel
        # path (curved fits keep the cap below).
        # Return fp32; the caller's ``hidden.copy_(h_new)`` downcasts to the
        # model dtype on the copy, so an explicit ``.to(h.dtype)`` here would
        # only allocate a redundant full-width model-dtype temporary per fire.
        return h_new, q

    centered = h_f32 - mean                            # (.., D)
    q = centered @ basis.T                             # (.., R) reduced coords of h_par
    h_par = q @ basis                                  # (.., D) in-subspace reconstruction
    h_perp = centered - h_par                          # (.., D) = H_o

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

    along_zero = not isinstance(along, torch.Tensor) and float(along) == 0.0
    onto_zero = not isinstance(onto, torch.Tensor) and float(onto) == 0.0

    emb_old_raw = domain.embed(p)                      # (.., m) un-normalized
    emb_old = (emb_old_raw - offset) / scale           # (.., m)
    foot_red = eval_rbf(np_, rw, pc, emb_old)          # (.., R)
    Hn_red = q - foot_red                              # (.., R) off-manifold-in-subspace (reduced)

    # --- ALONG: translate the foot along the surface ---
    # Translate p* by the parallel-transported neutral→target offset (scaled by
    # ``along``), preserving the per-token foot spread rather than collapsing
    # every foot onto the absolute target.  ``origin`` is this layer's neutral
    # foot; falls back to coord 0.
    if along_zero:
        p_new = p
        emb_new_raw = emb_old_raw
        foot_new_red = foot_red
        Hn_trans = Hn_red
    else:
        org = origin if origin is not None else torch.zeros_like(target)
        p_new = domain.translate_foot(p, org, target, along)  # (.., n)
        emb_new_raw = domain.embed(p_new)                  # (.., m) un-normalized
        emb_new = (emb_new_raw - offset) / scale           # (.., m)
        foot_new_red = eval_rbf(np_, rw, pc, emb_new)      # (.., R)
        # World-reduced tangent columns of M at the old/new feet: dRBF/dcoord
        # chained through the embedding Jacobian (the reduced-space analogue of
        # Manifold.tangent).  Needed only when the foot actually moves; when
        # along==0 the transport is exactly identity, so skipping this avoids the
        # n>=2 MPS SVD CPU hop and two Jacobian evaluations.
        j_old = (
            eval_rbf_jacobian(np_, rw, pc, emb_old) / scale  # (.., R, m)
        ) @ domain.embed_jacobian(p)                         # (.., R, n)
        j_new = (
            eval_rbf_jacobian(np_, rw, pc, emb_new) / scale  # (.., R, m)
        ) @ domain.embed_jacobian(p_new)                     # (.., R, n)

        # Transport the off-surface residual from the frame at p to the frame at
        # p_new by the minimal orthogonal (principal-angle) rotation between the
        # two tangent subspaces.  See :func:`_frame_rotation_transport`.
        Hn_trans = _frame_rotation_transport(Hn_red, j_old, j_new)  # (.., R)

    # --- ONTO: collapse the transported off-manifold residual toward the tube ---
    # Legacy (zero-thickness wire, σ-field absent): scale ``H_n`` by ``(1 − o)``
    # — at ``o = 1`` the activation lands *on* the mean surface.  Fuzzy
    # (σ-field present): shrink ``‖H_n‖`` toward the local within-node thickness
    # ``σ(z)`` instead of toward 0, so ``o = 1`` lands one-σ off the wire (the
    # surface's *typical set*, a sample-like point) rather than on the idealized
    # centroid — the within-concept variety the hard collapse erases is exactly
    # what drives the strong-push mode-collapse the open-frontier note flags.
    # The residual *direction* is preserved (only its magnitude is rescaled),
    # and a token already inside the tube (``‖H_n‖ ≤ σ``) is never *expanded*
    # (the ``(·)_+`` clamp).  ``σ(z) = 0`` ⇒ ``shrink = (1)_+ = 1`` ⇒ the exact
    # ``(1 − o)`` legacy collapse.
    if onto_zero:
        Hn_final = Hn_trans
    else:
        sigma = subspace.sigma_at(emb_new_raw)               # (..,) tube thickness σ(z)
        hn_norm = torch.linalg.vector_norm(Hn_trans, dim=-1)  # (..,)
        shrink = torch.clamp(
            1.0 - sigma / hn_norm.clamp(min=1e-6), min=0.0,
        )                                                     # (..,)
        onto_scale = 1.0 - onto * shrink                     # (..,) in [1−o, 1]
        Hn_final = onto_scale.unsqueeze(-1) * Hn_trans        # (.., R)

    new_par = (foot_new_red + Hn_final) @ basis          # (.., D) back to world

    # The off-subspace residual ``H_o`` is kept verbatim (the old ``toward`` op
    # that scaled it is removed — it scaled the orthogonal complement of this
    # subspace, breaking orthogonal composition with neighboring terms).
    new_perp = h_perp                                    # (.., D) kept verbatim

    h_new = mean + new_par + new_perp                    # (.., D)

    # Soft safety cap: only fires on pathological off-domain RBF extrapolation
    # (clamp_position keeps p_new in-box for open axes, so this is belt-and-
    # suspenders, not the norm semantic — onto is allowed to shrink ‖h‖).
    h_new = _soft_norm_cap(h_new, h_f32, norm_cap)
    # Return fp32 (see the affine branch): the caller's ``hidden.copy_`` does
    # the model-dtype downcast, so no per-fire ``.to(h.dtype)`` temporary.
    return h_new, p


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
    for r, c in zip(rows.tolist(), cols.tolist(), strict=True):
        union(r, c)
    return len({find(i) for i in range(K)})


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
    diagnostics: "object | None" = None
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
    parent = list(range(K))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    positive_edges: set[int] = set()  # edge filtration indices that create a 1-cycle
    for x in range(E):
        ri, rj = find(ei[x]), find(ej[x])
        if ri != rj:
            parent[ri] = rj      # tree edge — kills an H0 component
        else:
            positive_edges.add(x)  # both endpoints already joined — births a loop

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
    parent = list(range(K))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    eps_c = 0.0
    joined = 0
    src, dst = iu[0].tolist(), iu[1].tolist()
    for o in order.tolist():
        ra, rb = find(src[o]), find(dst[o])
        if ra != rb:
            parent[ra] = rb
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


# ------------------------------------------------------ centroid capture ---

def compute_node_centroid(
    model: torch.nn.Module,
    tokenizer: object,
    layers: torch.nn.ModuleList,
    device: torch.device,
    responses: list[str],
    prompts: "list[str] | list[list[dict[str, str]]]",
    *,
    role: str | None = None,
    model_type: str | None = None,
    layer_indices: Sequence[int] | None = None,
) -> dict[int, torch.Tensor]:
    """Mean per-layer pooled activation over a manifold node's responses.

    Conversational (4.0 / A2) pooling: a node corpus is a list of in-character
    *responses* to the shared baseline *prompts*, aligned positionally as
    ``responses[i] -> prompts[i % len(prompts)]``.  Each is captured as a
    ``[system: length directive, user: prompt, assistant: response]`` turn
    (the shared brevity directive only -- *not* the generation-time persona --
    standard label) via :func:`saklas.core.vectors._encode_and_capture_all`,
    matching the framing the corpus was generated under so it isn't
    out-of-distribution and cancels as common-mode against the neutral baseline.
    Same last-content-token, fp32 pooling discipline that backs
    :func:`saklas.core.vectors.compute_neutral_activations`, with the same MPS
    ``empty_cache`` discipline between forward passes.

    ``role`` (optional): substitute a custom assistant-role label into the
    chat template via :func:`saklas.core.role_templates.apply_with_role`, so
    the pooled centroid lives in persona-baseline activation space instead of
    the standard assistant (swap-back) baseline.  Requires ``model_type``.
    ``role=None`` is the swap-back default.

    ``layer_indices`` narrows capture to a subset; ``None`` captures every layer.
    Returns ``{layer_idx: centroid (D,)}`` in fp32 on CPU.
    """
    from saklas.core.vectors import _CAPTURE_BATCH, _encode_and_capture_all_batch

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

    capture_layers = (
        list(range(len(layers)))
        if layer_indices is None
        else [int(idx) for idx in layer_indices]
    )
    sums: dict[int, torch.Tensor] = {}
    is_mps = getattr(device, "type", None) == "mps"
    n = len(responses)
    # Each response pairs with ``prompts[i % k]`` (the A2 round-robin alignment);
    # capture is batched in chunks of ``_CAPTURE_BATCH`` (one forward per chunk,
    # MPS flush amortized) and reduced to a running per-layer sum, so peak memory
    # is one chunk's pooled ``(B, D)`` per layer rather than the whole corpus.
    aligned_prompts = [prompts[i % k] for i in range(n)]

    for start in range(0, n, _CAPTURE_BATCH):
        end = min(start + _CAPTURE_BATCH, n)
        per_layer = _encode_and_capture_all_batch(
            model, tokenizer,
            aligned_prompts[start:end], responses[start:end],
            layers, device, role=role, model_type=model_type,
            layer_indices=capture_layers,
        )
        for idx in capture_layers:
            chunk_sum = per_layer[idx].detach().sum(dim=0).to("cpu", torch.float32)
            if idx in sums:
                sums[idx] += chunk_sum
            else:
                sums[idx] = chunk_sum
        del per_layer
        if is_mps:
            torch.mps.empty_cache()

    return {idx: sums[idx] / n for idx in capture_layers}


def compute_node_activation_rows(
    model: torch.nn.Module,
    tokenizer: object,
    layers: torch.nn.ModuleList,
    device: torch.device,
    responses: list[str],
    prompts: "list[str] | list[list[dict[str, str]]]",
    *,
    role: str | None = None,
    model_type: str | None = None,
    layer_indices: Sequence[int] | None = None,
) -> dict[int, torch.Tensor]:
    """Per-response pooled activations for one node, fp32 on CPU.

    This is the row-retaining sibling of :func:`compute_node_centroid`: same
    conversational rendering, batching, last-content pooling, role handling, and
    MPS cache discipline, but it returns ``{layer: (N, D)}`` instead of reducing
    to a centroid.  Curved raw manifold fits use it to derive both centroids and
    the fuzzy-manifold σ-field from one capture pass.
    """
    from saklas.core.vectors import _CAPTURE_BATCH, _encode_and_capture_all_batch

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

    capture_layers = (
        list(range(len(layers)))
        if layer_indices is None
        else [int(idx) for idx in layer_indices]
    )
    rows_by_layer: dict[int, torch.Tensor] = {}
    is_mps = getattr(device, "type", None) == "mps"
    n = len(responses)
    aligned_prompts = [prompts[i % k] for i in range(n)]

    for start in range(0, n, _CAPTURE_BATCH):
        end = min(start + _CAPTURE_BATCH, n)
        per_layer = _encode_and_capture_all_batch(
            model, tokenizer,
            aligned_prompts[start:end], responses[start:end],
            layers, device, role=role, model_type=model_type,
            layer_indices=capture_layers,
        )
        for idx in capture_layers:
            chunk = per_layer[idx].detach().to("cpu", torch.float32)
            if idx not in rows_by_layer:
                rows_by_layer[idx] = torch.empty(
                    n, chunk.shape[1], dtype=torch.float32,
                )
            rows_by_layer[idx][start:end].copy_(chunk)
        del per_layer
        if is_mps:
            torch.mps.empty_cache()

    return rows_by_layer


class ActivationRowStore:
    """Temporary layer-major pooled-row spool for curved manifold fitting.

    A curved fit needs per-response rows only after the centroid-derived basis
    exists.  Keeping ``nodes × responses × layers × d_model`` in resident fp32
    is a multi-GiB cliff, while dropping it makes auto-curved fitting repeat the
    complete model pass.  This store writes one mmap-backed tensor per layer in
    the model output dtype, indexed by the original flat corpus row.  fp16/bf16
    storage is lossless relative to the source residual; fp32 promotion happens
    only when covariance math consumes a node slice.
    """

    def __init__(self, node_sizes: Sequence[int]) -> None:
        self.node_sizes = [int(size) for size in node_sizes]
        self.offsets: list[int] = []
        offset = 0
        for size in self.node_sizes:
            self.offsets.append(offset)
            offset += size
        self.total_rows = offset
        self._tmp: tempfile.TemporaryDirectory[str] | None = (
            tempfile.TemporaryDirectory(prefix="saklas-manifold-rows-")
        )
        self._layers: dict[int, torch.Tensor] = {}
        self._owners: list[ActivationRowStore] = []
        self._closed = False

    @classmethod
    def load(
        cls, path: Path, node_sizes: Sequence[int], *,
        layer_indices: Sequence[int] | None = None,
    ) -> "ActivationRowStore":
        store = cls.__new__(cls)
        store.node_sizes = [int(size) for size in node_sizes]
        store.offsets = []
        offset = 0
        for size in store.node_sizes:
            store.offsets.append(offset)
            offset += size
        store.total_rows = offset
        selected = (
            None if layer_indices is None
            else {int(idx) for idx in layer_indices}
        )
        store._layers = {}
        with safe_open(str(path), framework="pt", device="cpu") as tensors:
            layer_keys: dict[int, str] = {}
            for key in tensors.keys():
                if not key.startswith("layer_"):
                    raise ValueError(
                        f"invalid activation-row cache key {key!r} at {path}"
                    )
                try:
                    idx = int(key.split("_", 1)[1])
                except ValueError as exc:
                    raise ValueError(
                        f"invalid activation-row cache key {key!r} at {path}"
                    ) from exc
                shape = tuple(int(dim) for dim in tensors.get_slice(key).get_shape())
                if len(shape) != 2 or shape[0] != store.total_rows:
                    raise ValueError(
                        f"invalid activation-row cache tensor {key!r} at {path}"
                    )
                layer_keys[idx] = key
                if selected is None or idx in selected:
                    store._layers[idx] = tensors.get_tensor(key)
            if not layer_keys:
                raise ValueError(f"empty activation-row cache at {path}")
        if not store._layers or any(
            rows.ndim != 2 or int(rows.shape[0]) != store.total_rows
            for rows in store._layers.values()
        ):
            raise ValueError(f"invalid activation-row cache at {path}")
        if selected is not None and set(store._layers) != selected:
            raise ValueError(
                f"activation-row cache at {path} has layers "
                f"{sorted(store._layers)}, need {sorted(selected)}"
            )
        store._tmp = None
        store._owners = []
        store._closed = False
        return store

    @classmethod
    def load_shards(
        cls,
        paths: "dict[int, Path]",
        node_sizes: Sequence[int],
    ) -> "ActivationRowStore":
        """Load independently persisted per-layer activation-row shards.

        Capture-cache v4 stores one safetensors file per layer so a scoped fit
        neither maps nor rewrites unrelated multi-GiB row payloads.  Every
        shard must carry exactly its named ``layer_<L>`` tensor; the caller
        validates the tensor's exact digest against the atomically published
        capture metadata.
        """
        if not paths:
            raise ValueError("activation-row shard selection must not be empty")
        store = cls.__new__(cls)
        store.node_sizes = [int(size) for size in node_sizes]
        store.offsets = []
        offset = 0
        for size in store.node_sizes:
            store.offsets.append(offset)
            offset += size
        store.total_rows = offset
        store._layers = {}
        for idx, path in sorted(paths.items()):
            expected_key = f"layer_{int(idx)}"
            with safe_open(str(path), framework="pt", device="cpu") as tensors:
                keys = list(tensors.keys())
                if keys != [expected_key]:
                    raise ValueError(
                        f"activation-row shard at {path} has keys {keys}, "
                        f"expected [{expected_key!r}]"
                    )
                shape = tuple(
                    int(dim) for dim in tensors.get_slice(expected_key).get_shape()
                )
                if len(shape) != 2 or shape[0] != store.total_rows:
                    raise ValueError(
                        f"invalid activation-row shard tensor {expected_key!r} "
                        f"at {path}"
                    )
                store._layers[int(idx)] = tensors.get_tensor(expected_key)
        store._tmp = None
        store._owners = []
        store._closed = False
        return store

    @classmethod
    def combine_disjoint(
        cls, stores: Sequence["ActivationRowStore"],
    ) -> "ActivationRowStore":
        """Combine disjoint layer stores by view, transferring ownership.

        Sharded cache top-ups often pair many immutable cached layers with one
        newly captured temporary layer. Copying all ``N x D`` rows into a third
        mmap merely to present one layer-major roster doubles multi-GiB I/O.
        This composite aliases the existing tensors and keeps their stores
        alive until the composite closes; no payload bytes move.
        """
        if not stores:
            raise ValueError("activation-row combine needs at least one store")
        node_sizes = list(stores[0].node_sizes)
        if any(store._closed for store in stores):
            raise RuntimeError("cannot combine a closed activation-row store")
        if any(store.node_sizes != node_sizes for store in stores[1:]):
            raise ValueError("activation-row stores must share node sizes")
        layers: dict[int, torch.Tensor] = {}
        for store in stores:
            overlap = set(layers) & set(store._layers)
            if overlap:
                raise ValueError(
                    f"activation-row stores overlap layers {sorted(overlap)}"
                )
            layers.update(store._layers)
        combined = cls.__new__(cls)
        combined.node_sizes = node_sizes
        combined.offsets = list(stores[0].offsets)
        combined.total_rows = stores[0].total_rows
        combined._tmp = None
        combined._layers = layers
        combined._owners = list(stores)
        combined._closed = False
        return combined

    def _layer(self, idx: int, *, dim: int, dtype: torch.dtype) -> torch.Tensor:
        existing = self._layers.get(idx)
        if existing is not None:
            if existing.shape != (self.total_rows, dim) or existing.dtype != dtype:
                raise ValueError(f"activation-row shape/dtype changed at layer {idx}")
            return existing
        if self._closed:
            raise RuntimeError("activation-row store is closed")
        assert self._tmp is not None
        path = Path(self._tmp.name) / f"layer_{idx}.bin"
        rows = torch.from_file(
            str(path), shared=True, size=self.total_rows * dim, dtype=dtype,
        ).reshape(self.total_rows, dim)
        self._layers[idx] = rows
        return rows

    def write(
        self, idx: int, flat_indices: torch.Tensor, rows: torch.Tensor,
    ) -> None:
        host = rows.detach().to(device="cpu")
        target = self._layer(idx, dim=int(host.shape[1]), dtype=host.dtype)
        target.index_copy_(0, flat_indices.to(dtype=torch.long, device="cpu"), host)

    def node_rows(self, node_idx: int) -> dict[int, torch.Tensor]:
        start = self.offsets[node_idx]
        end = start + self.node_sizes[node_idx]
        return {idx: rows[start:end] for idx, rows in self._layers.items()}

    @property
    def layer_indices(self) -> list[int]:
        return sorted(self._layers)

    def flat_rows(self, idx: int) -> torch.Tensor:
        return self._layers[idx]

    def persist(self, path: Path) -> None:
        if self._closed:
            raise RuntimeError("activation-row store is closed")
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
        )
        os.close(fd)
        tmp = Path(tmp_name)
        try:
            save_file(
                {
                    f"layer_{idx}": rows.contiguous()
                    for idx, rows in self._layers.items()
                },
                str(tmp),
            )
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)

    def __iter__(self) -> Iterator[dict[int, torch.Tensor]]:
        for node_idx in range(len(self.node_sizes)):
            yield self.node_rows(node_idx)

    def close(self) -> None:
        if self._closed:
            return
        self._layers.clear()
        owners = getattr(self, "_owners", [])
        self._owners = []
        for owner in owners:
            owner.close()
        if self._tmp is not None:
            self._tmp.cleanup()
        self._closed = True

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def compute_manifold_node_stats(
    model: torch.nn.Module,
    tokenizer: object,
    layers: torch.nn.ModuleList,
    device: torch.device,
    node_groups: "Sequence[tuple[str, list[str]]]",
    prompts: "Sequence[str | list[dict[str, str]]]",
    *,
    roles: "Sequence[str | None]",
    model_type: str | None = None,
    layer_indices: Sequence[int] | None = None,
    retain_rows: bool = False,
    prepared_rows: "Sequence[tuple[torch.Tensor, int]] | None" = None,
    capture_context: "Any | None" = None,
) -> tuple[dict[int, torch.Tensor], ActivationRowStore | None]:
    """Fit-wide batched capture for all manifold nodes.

    Rows carry their node/within-node indices through one stream, so short
    template corpora share batches across node boundaries instead of paying one
    underfilled model forward per node.  Standard 48-response nodes retain the
    same chunking, while OOM backoff halves the active batch and cautiously grows
    it again. Returns layer-major ``(K, D)`` fp32 centroid stacks plus optional
    retained rows in the capture source dtype.  Keeping the result layer-major
    matches every fit consumer and avoids rebuilding each stack from ``K`` row
    views immediately after capture.
    """
    from saklas.core.vectors import (
        _CAPTURE_BATCH_MAX,
        _encode_and_capture_all_batch,
        _prepare_capture_batch,
    )

    if not prompts:
        raise ValueError("conversational capture needs at least one baseline prompt")
    if len(node_groups) != len(roles):
        raise ValueError("node_groups and roles must be aligned")
    capture_layers = (
        list(range(len(layers)))
        if layer_indices is None else [int(idx) for idx in layer_indices]
    )
    k = len(prompts)
    flat: list[
        tuple[int, int, str | list[dict[str, str]], str, str | None]
    ] = []
    node_sizes: list[int] = []
    for node_idx, ((_label, responses), role) in enumerate(
        zip(node_groups, roles, strict=True),
    ):
        if not responses or len(responses) % k != 0:
            raise ValueError(
                f"node corpus ({len(responses)} responses) must be a non-empty "
                f"multiple of the baseline prompt set ({k})"
            )
        node_sizes.append(len(responses))
        flat.extend(
            (node_idx, row_idx, prompts[row_idx % k], response, role)
            for row_idx, response in enumerate(responses)
        )

    sums: dict[int, torch.Tensor] = {}
    retained = ActivationRowStore(node_sizes) if retain_rows else None
    # Render/tokenize once, then group similar lengths so right-padding does not
    # make every row pay the longest response's quadratic attention cost.  Tiny
    # CPU test tokenizers are intentionally non-callable and keep the legacy
    # seam; production HF tokenizers always take the prepared path.
    prepared: list[tuple[torch.Tensor, int]] | None = (
        list(prepared_rows) if prepared_rows is not None else None
    )
    if prepared is not None and len(prepared) != len(flat):
        raise ValueError("prepared_rows must align with the flattened manifold corpus")
    if prepared is None and callable(tokenizer):
        prepared = _prepare_capture_batch(
            tokenizer,
            [row[2] for row in flat],
            [row[3] for row in flat],
            device,
            roles=[row[4] for row in flat],
            model_type=model_type,
        )
    order = (
        sorted(range(len(flat)), key=lambda i: int(prepared[i][0].shape[1]))
        if prepared is not None else list(range(len(flat)))
    )
    # Optimistically try the proven maximum and halve on OOM. Starting at the
    # old conservative 16 made a 107-node fit spend four successful forwards
    # merely climbing to the width that the same device had already shown it
    # could run.
    active_batch = _CAPTURE_BATCH_MAX
    start = 0
    is_mps = getattr(device, "type", None) == "mps"
    while start < len(order):
        end = min(start + active_batch, len(order))
        chunk_indices = order[start:end]
        chunk = [flat[i] for i in chunk_indices]
        try:
            per_layer = _encode_and_capture_all_batch(
                model, tokenizer,
                [row[2] for row in chunk],
                [row[3] for row in chunk],
                layers, device,
                roles=[row[4] for row in chunk],
                model_type=model_type,
                layer_indices=capture_layers,
                rendered=(
                    [prepared[i] for i in chunk_indices]
                    if prepared is not None else None
                ),
                promote_pooled=not retain_rows,
                capture_context=capture_context,
            )
        except RuntimeError as exc:
            if not is_out_of_memory_error(exc) or active_batch <= 1:
                raise
            active_batch = max(1, active_batch // 2)
            if is_mps:
                torch.mps.empty_cache()
            elif getattr(device, "type", None) == "cuda":
                torch.cuda.empty_cache()
            continue

        node_ids_cpu = torch.tensor([row[0] for row in chunk], dtype=torch.long)
        flat_indices_cpu = torch.tensor(chunk_indices, dtype=torch.long)
        unique = inverse = None
        if retained is None:
            node_ids = node_ids_cpu.to(device)
            unique, inverse = torch.unique(
                node_ids, sorted=True, return_inverse=True,
            )
        for idx in capture_layers:
            captured = per_layer[idx].detach()
            if idx not in sums:
                sums[idx] = torch.zeros(
                    len(node_groups), captured.shape[1], dtype=torch.float32,
                )
            if retained is not None:
                # Raw rows must cross the boundary for the later sigma fit, but
                # stay in source dtype in the mmap.  Accumulate their promoted
                # host values while they are already present.
                host = captured.to(device="cpu")
                retained.write(idx, flat_indices_cpu, host)
                sums[idx].index_add_(
                    0, node_ids_cpu, host.to(torch.float32),
                )
            else:
                # Centroid-only fits need one partial sum per node, not every
                # response row.  Reduce on-device in fp32 and transfer U×D
                # (usually 1×D) instead of B×D.
                assert unique is not None and inverse is not None
                partial = torch.zeros(
                    unique.numel(), captured.shape[1],
                    dtype=torch.float32, device=captured.device,
                )
                partial.index_add_(0, inverse, captured.to(torch.float32))
                sums[idx].index_add_(
                    0, unique.to(device="cpu"), partial.to(device="cpu"),
                )
        del per_layer
        start = end
        if is_mps:
            torch.mps.empty_cache()

    divisors = torch.tensor(node_sizes, dtype=torch.float32).reshape(-1, 1)
    centroids = {
        idx: (sums[idx] / divisors).contiguous() for idx in capture_layers
    }
    return centroids, retained


def compute_node_reduced_covariance_from_rows(
    activation_rows: dict[int, torch.Tensor],
    layer_subs: "dict[int, LayerSubspace]",
) -> dict[int, torch.Tensor]:
    """Within-node reduced covariance from retained pooled activations.

    ``activation_rows`` must carry the same ``{layer: (N, D)}`` rows returned by
    :func:`compute_node_activation_rows`.  The result is identical to
    :func:`compute_node_reduced_covariance` without re-running the model.
    """
    covs: dict[int, torch.Tensor] = {}
    for idx, sub in layer_subs.items():
        rows = activation_rows[idx].to("cpu", torch.float32)
        n = int(rows.shape[0])
        mean = sub.mean.to(torch.float32)
        basis = sub.basis.to(torch.float32)
        z = (rows - mean) @ basis.T
        if n <= 1:
            covs[idx] = torch.zeros(
                (sub.rank, sub.rank), dtype=torch.float32,
            )
            continue
        centered = z - z.mean(dim=0, keepdim=True)
        covs[idx] = centered.T @ centered / float(n - 1)
    return covs


def compute_store_reduced_covariances(
    store: ActivationRowStore,
    layer_subs: "dict[int, LayerSubspace]",
    *,
    row_chunk: int = 2048,
) -> list[dict[int, torch.Tensor]]:
    """Project a layer-major activation spool into per-node covariances.

    The legacy helper above is useful for one standalone node, but iterating it
    over an :class:`ActivationRowStore` visits a layer-major mmap in node-major
    order and launches one ``(N_node,D) @ (D,R)`` projection per node and layer.
    This fit-wide sibling streams each layer once in bounded row chunks, performs
    large projection GEMMs, and segments only the small ``(N,R)`` results by the
    store's contiguous node boundaries.  Covariance is translation-invariant, so
    projecting raw rows is exactly equivalent to subtracting ``sub.mean`` first
    and then centering the reduced rows.
    """
    if row_chunk <= 0:
        raise ValueError("row_chunk must be > 0")
    if not set(layer_subs) <= set(store.layer_indices):
        raise ValueError(
            "activation-row store must cover every fitted subspace layer"
        )

    n_nodes = len(store.node_sizes)
    out: list[dict[int, torch.Tensor]] = [dict() for _ in range(n_nodes)]
    boundaries = [
        (offset, offset + size)
        for offset, size in zip(store.offsets, store.node_sizes, strict=True)
    ]
    for idx, sub in layer_subs.items():
        basis_t = sub.basis.to(device="cpu", dtype=torch.float32).transpose(0, 1)
        rank = sub.rank
        rows = store.flat_rows(idx)
        reduced_rows = torch.empty(store.total_rows, rank, dtype=torch.float32)
        mean = sub.mean.to(device="cpu", dtype=torch.float32)
        for start in range(0, store.total_rows, row_chunk):
            end = min(start + row_chunk, store.total_rows)
            centered_chunk = rows[start:end].to(torch.float32)
            centered_chunk.sub_(mean)
            reduced_rows[start:end] = centered_chunk @ basis_t

        for k, (start, end) in enumerate(boundaries):
            size = end - start
            if size <= 1:
                cov = torch.zeros(rank, rank, dtype=torch.float32)
            else:
                node_reduced = reduced_rows[start:end]
                centered = node_reduced - node_reduced.mean(dim=0, keepdim=True)
                cov = centered.transpose(0, 1) @ centered / float(size - 1)
            out[k][idx] = cov
    return out


def compute_node_reduced_covariance(
    model: torch.nn.Module,
    tokenizer: object,
    layers: torch.nn.ModuleList,
    device: torch.device,
    responses: list[str],
    prompts: "list[str] | list[list[dict[str, str]]]",
    layer_subs: "dict[int, LayerSubspace]",
    *,
    role: str | None = None,
    model_type: str | None = None,
) -> dict[int, torch.Tensor]:
    """Within-node **reduced** covariance ``(R, R)`` per layer for one node.

    The fuzzy-manifold fallback pass (curved fits only).  Re-captures the node's
    corpus exactly as :func:`compute_node_centroid` does — same conversational
    ``[directive, prompt, response]`` framing, same last-content-token fp32
    pooling — but instead of pooling to the centroid it projects every sample
    through each fitted layer's affine frame
    (``Z = (h − mean) · basisᵀ``, ``(B, R)``) and accumulates the reduced
    first/second moments, returning ``{layer: cov (R, R)}`` (sample covariance,
    ``count − 1`` denominator; zeros for a single-sample node).

    This needs the per-layer ``mean``/``basis`` (hence ``layer_subs``), which
    only exist *after* the surface fit.  When the extraction pipeline already
    retained first-pass activation rows it uses
    :func:`compute_node_reduced_covariance_from_rows` instead; this function is
    the low-memory fallback for paths that did not retain them (notably
    auto-topology fits that resolved curved after centroid pooling).  The
    off-surface reduction of these covariances into per-node ``σ`` lives in the
    extraction pipeline.  fp32 on CPU, same MPS ``empty_cache`` discipline.
    """
    from saklas.core.vectors import _CAPTURE_BATCH, _encode_and_capture_all_batch

    if not responses:
        raise ValueError("manifold node has no responses")
    if not prompts:
        raise ValueError("conversational capture needs at least one baseline prompt")
    k = len(prompts)
    if len(responses) % k != 0:
        raise ValueError(
            f"node corpus ({len(responses)} responses) must be a multiple of "
            f"the baseline prompt set ({k})"
        )
    is_mps = getattr(device, "type", None) == "mps"
    n = len(responses)
    aligned_prompts = [prompts[i % k] for i in range(n)]

    # Per-layer reduced accumulators: count, Σz (R,), Σ z zᵀ (R, R).
    sum_z: dict[int, torch.Tensor] = {}
    sum_zz: dict[int, torch.Tensor] = {}
    means_basis = {
        idx: (sub.mean.to(torch.float32), sub.basis.to(torch.float32))
        for idx, sub in layer_subs.items()
    }

    for start in range(0, n, _CAPTURE_BATCH):
        end = min(start + _CAPTURE_BATCH, n)
        per_layer = _encode_and_capture_all_batch(
            model, tokenizer,
            aligned_prompts[start:end], responses[start:end],
            layers, device, role=role, model_type=model_type,
            layer_indices=sorted(layer_subs),
        )
        for idx, (mean, basis) in means_basis.items():
            h = per_layer[idx].detach().to("cpu", torch.float32)  # (B, D)
            z = (h - mean) @ basis.T                              # (B, R)
            sz = z.sum(dim=0)                                     # (R,)
            szz = z.T @ z                                         # (R, R)
            if idx in sum_z:
                sum_z[idx] += sz
                sum_zz[idx] += szz
            else:
                sum_z[idx] = sz
                sum_zz[idx] = szz
        del per_layer
        if is_mps:
            torch.mps.empty_cache()

    covs: dict[int, torch.Tensor] = {}
    for idx in layer_subs:
        cnt = float(n)
        mu = sum_z[idx] / cnt                                     # (R,)
        # Sample covariance Σzzᵀ/cnt − μμᵀ, unbiased (cnt−1) scaling.
        cov = sum_zz[idx] / cnt - torch.outer(mu, mu)            # (R, R)
        if n > 1:
            cov = cov * (cnt / (cnt - 1.0))
        covs[idx] = cov
    return covs


def _reduced_tangents(
    sub: "LayerSubspace", domain: "ManifoldDomain", coords: torch.Tensor,
) -> torch.Tensor:
    """Batched reduced-space surface tangents ``(K, R, n)``.

    Computes the RBF Jacobian chained through the domain's embedding Jacobian
    for every node at once — ``embed`` / ``embed_jacobian`` are batch-generic
    and ``eval_rbf_jacobian`` is vectorized, so the σ-field pass covers all K
    node tangent frames for a layer in one tensor sweep.
    """
    np_, rw, pc = sub.rbf_params()
    coords_f = coords.to(torch.float32)
    emb = sub._normalize(domain.embed(coords_f))                  # (K, m)
    j_red = eval_rbf_jacobian(np_, rw, pc, emb) / sub.coord_scale  # (K, R, m)
    return j_red @ domain.embed_jacobian(coords_f)                # (K, R, n)


# test-only: production code uses the batched _off_surface_vars; this scalar
# form is exercised directly in tests/test_manifold_math.py.
def _off_surface_var(
    cov: torch.Tensor, tangent: torch.Tensor, R: int, n: int,
) -> float:
    """Mean within-node variance in the off-surface (normal) directions.

    ``cov`` is the node's reduced ``(R, R)`` within-node covariance, ``tangent``
    its ``(R, n)`` surface tangent.  Projects ``cov`` onto the normal complement
    ``P = I − t(tᵀt)⁺tᵀ`` and returns ``tr(P cov)/(R − n)`` — the part of the
    node's scatter that lives *off* the mean surface, which is what the tube
    thickness should be (tangential scatter is the node sliding *along* the
    surface, expected and not thickness).  Degenerates to the full isotropic
    ``tr(cov)/R`` when the surface fills its subspace (``R ≤ n``, no normal
    complement).  Clamped non-negative (sample-covariance round-off).
    """
    if R <= n:
        return float(torch.diagonal(cov).sum() / max(R, 1))
    tt = tangent.T @ tangent                                   # (n, n)
    proj = tangent @ torch.linalg.pinv(tt) @ tangent.T          # (R, R) onto tangent
    normal = torch.eye(R, dtype=cov.dtype) - proj               # onto normal complement
    var = torch.trace(normal @ cov) / float(R - n)
    return float(var.clamp(min=0.0))


def _off_surface_vars(
    covs: torch.Tensor, tangents: torch.Tensor, R: int, n: int,
) -> torch.Tensor:
    """Batched counterpart to :func:`_off_surface_var`.

    ``covs`` is ``(K, R, R)`` and ``tangents`` is ``(K, R, n)``.  Returns one
    non-negative off-surface variance per node, keeping the small pseudoinverse
    solve batched on CPU during σ-field fitting.
    """
    if R <= n:
        return torch.diagonal(covs, dim1=-2, dim2=-1).sum(dim=-1) / max(R, 1)
    tt = tangents.transpose(-1, -2) @ tangents                  # (K, n, n)
    proj = tangents @ torch.linalg.pinv(tt) @ tangents.transpose(-1, -2)
    normal = torch.eye(R, dtype=covs.dtype, device=covs.device) - proj
    var = torch.einsum("kij,kji->k", normal, covs) / float(R - n)
    return var.clamp(min=0.0)


def fit_sigma_field(
    layer_subs: "dict[int, LayerSubspace]",
    domain: "ManifoldDomain",
    node_coords: torch.Tensor,
    node_covs: "list[dict[int, torch.Tensor]]",
    *,
    smoothing: float | str | None = "auto",
    floor_frac: float = 1e-3,
    rbf_plan: RbfFitPlan | None = None,
) -> dict[int, dict[str, float]]:
    """Attach a fuzzy-manifold ``log σ`` RBF to each curved layer (mutates them).

    Reduces the per-node within-node covariances (from
    :func:`compute_node_reduced_covariance`) to one off-surface ``σ`` per node
    per layer (:func:`_off_surface_var`), then fits a *separate* penalized
    ``r**3`` RBF over the **same normalized** ``node_params`` interpolating the
    per-node ``log σ`` and writes it onto ``sub.sigma_rbf_weights`` /
    ``sub.sigma_poly_coeffs``.  Returns ``{layer: {"sigma_mean", "sigma_min",
    "sigma_max", "lambda"}}`` for the sidecar.

    ``floor_frac`` floors each layer's per-node ``σ²`` at ``floor_frac × median``
    so a degenerate (single-sample / collapsed) node can't drive ``log σ → −∞``.
    Smoothing defaults to GCV ``"auto"`` — the σ-field is noisier than the mean
    surface (a second-moment estimate from ~48 samples), so a regularized
    interpolant is the right default; ``0`` makes it exact at the nodes.
    """
    K = int(node_coords.shape[0])
    n = int(domain.intrinsic_dim)
    coords_f = node_coords.to(torch.float32)
    info: dict[int, dict[str, float]] = {}
    if rbf_plan is None and layer_subs:
        first = next(iter(layer_subs.values()))
        np_, _rw, _pc = first.rbf_params()
        rbf_plan = prepare_rbf_fit_plan(np_, smoothing=smoothing)
    for idx, sub in layer_subs.items():
        R = sub.rank
        covs = torch.stack(
            [node_covs[kidx][idx].to(torch.float32) for kidx in range(K)],
            dim=0,
        )                                                        # (K, R, R)
        tangents = _reduced_tangents(sub, domain, coords_f)      # (K, R, n)
        raw = _off_surface_vars(covs, tangents, R, n).to(torch.float32)
        floor = floor_frac * float(raw.median().clamp(min=1e-12))
        sigma = raw.clamp(min=floor).sqrt()                      # (K,) σ (std)
        log_sigma = torch.log(sigma).reshape(K, 1)               # (K, 1)
        np_, _rw, _pc = sub.rbf_params()
        w, c, rinfo = fit_rbf_smoothed(
            np_.to(torch.float32), log_sigma, smoothing=smoothing,
            plan=rbf_plan,
        )
        sub.sigma_rbf_weights = w
        sub.sigma_poly_coeffs = c
        info[idx] = {
            "sigma_mean": float(sigma.mean()),
            "sigma_min": float(sigma.min()),
            "sigma_max": float(sigma.max()),
            "lambda": float(rinfo.get("lambda", 0.0)),
        }
    return info


# ------------------------------------------------------------- save/load ---

def _replace_manifold_file(source: Path, target: Path) -> None:
    """Atomic replace seam used by publication failure-injection tests."""

    os.replace(source, target)

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
        # Fuzzy-manifold σ-field (curved fits with the within-node spread pass).
        # The log-σ RBF over the same normalized ``node_params``; its *absence*
        # on disk is the read-side "zero-thickness wire" marker (legacy curved
        # fits and SAE fits skip it), so a fuzzy and a legacy manifold share one
        # save/load path exactly as affine/curved do.
        if sub.sigma_rbf_weights is not None and sub.sigma_poly_coeffs is not None:
            tensors[f"layer_{idx}.sigma_rbf_weights"] = (
                sub.sigma_rbf_weights.contiguous().to(torch.float32).cpu()
            )
            tensors[f"layer_{idx}.sigma_poly_coeffs"] = (
                sub.sigma_poly_coeffs.contiguous().to(torch.float32).cpu()
            )
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
        "nodes_sha256", "sae_release", "sae_revision", "sae_fingerprint",
        "sae_ids_by_layer",
        "sae_full_coverage",
        "model_fingerprint", "capture_sha256", "fitted_layers",
        "fit_policy_version",
        # Share-weighting metric ("mahalanobis" / "euclidean") — the
        # manifold analogue of the vector sidecar's ``bake`` field.
        "share_metric",
        # PCA subspace-selection metric ("mahalanobis" => whitened/Fisher
        # PCA, "euclidean" => ordinary centroid PCA).  Provenance only —
        # the fitted basis is baked into the tensor, so the runtime hot
        # path needs nothing from this field; surfaced by `manifold show`
        # and the inspector.
        "subspace_metric",
        # Per-layer whitened between-node spread ``{str(L): tr(G_L)}`` — the
        # concept's signal-concentration profile across the stack (the
        # consensus Gram's per-layer summand traces).  Diagnostic only; absent
        # on fits that predate it (loads as an empty dict).  Surfaced by
        # `manifold show`.
        "node_spread_per_layer",
        # Penalized-RBF provenance ``{str(L): {lambda, edf, gcv}}`` (curved
        # discover fits) and the fuzzy-manifold σ-field summary
        # ``{str(L): {sigma_mean, sigma_min, sigma_max, lambda}}`` (curved fits
        # run with the within-node spread pass).  Diagnostic only — the σ-RBF
        # tensors themselves ride the safetensors; these are the inspector
        # summaries.  Absent on fits without them (load as empty dicts).
        "rbf_smoothing_per_layer",
        "sigma_field_per_layer",
        # Discover-mode fields.  ``fit_mode`` discriminates authored vs
        # discover at read time; ``hyperparams`` records the knobs the
        # fitter was called with (max_dim / var_threshold / k_nn /
        # bandwidth) for reproducibility; ``diagnostics`` carries the
        # per-method PCA variance bars or spectral spectrum for the
        # CLI / webui inspector.  All absent for authored fits.
        "fit_mode", "hyperparams", "diagnostics",
        # Topology-selection provenance (``fit_mode="auto"`` only): the
        # geometry ``select_topology`` resolved to (``resolved_fit_mode`` ∈
        # pca/spectral + the winning ``topology_winner`` name) and the full
        # ranked candidate field (``topology_candidates`` — each
        # ``{name, fit_mode, intrinsic_dim, score, viable, reason}``, the
        # GCV scores behind the flat-vs-curved-vs-periodic decision).  Built
        # in ``extraction.py``; absent for pinned (non-auto) discover and
        # authored fits.
        "resolved_fit_mode", "topology_winner", "topology_candidates",
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
        # Merge provenance ({coord: {alpha, tensor_sha256}}),
        # carried on a ``fit_mode="baked"`` manifold produced by
        # :func:`saklas.io.merge.merge_into_manifold`.  Informational only — a
        # baked manifold never re-fits, so nothing branches on it; surfaced
        # by the inspector the way the legacy pack sidecar's ``components``
        # was.  Absent on every non-merge fit.
        "components", "bake_policy",
        # Cross-model transfer provenance. Persisted in the initial sidecar so
        # pair publication has no post-save patch crash window.
        "source_model_id", "source_model_fingerprint",
        "transfer_quality_estimate",
    ):
        if key in metadata:
            sidecar[key] = metadata[key]

    from saklas.io.manifold_folder import manifold_pair_lock

    with manifold_pair_lock(path):
        sidecar_path = path.with_suffix(".json")
        tensor_fd, tensor_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
        )
        os.close(tensor_fd)
        sidecar_fd, sidecar_name = tempfile.mkstemp(
            prefix=f".{sidecar_path.name}.", suffix=".tmp", dir=path.parent,
        )
        tensor_tmp = Path(tensor_name)
        sidecar_tmp = Path(sidecar_name)
        try:
            # Stage and fsync *both* payloads before replacing either canonical
            # path.  Commit the sidecar first: an interruption on a first fit
            # then leaves an ignored orphan sidecar, never the tensor-without-
            # sidecar shape the folder loader must reject.  On replacement an
            # interrupted mixed pair fails its old manifest proof and the next
            # target fit deterministically overwrites it.
            save_file(tensors, str(tensor_tmp))
            with open(tensor_tmp, "rb") as handle:
                os.fsync(handle.fileno())
            sidecar_bytes = (json.dumps(sidecar, indent=2) + "\n").encode()
            with os.fdopen(sidecar_fd, "wb", closefd=True) as handle:
                sidecar_fd = -1
                handle.write(sidecar_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            _replace_manifold_file(sidecar_tmp, sidecar_path)
            _replace_manifold_file(tensor_tmp, path)
            # Make both directory entries durable before the manifest CAS can
            # publish their hashes as the current pair.
            try:
                dir_fd = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
            except OSError:
                # Directory fsync is unavailable on some platforms; both files
                # themselves are still fsynced and retry recovery remains safe.
                pass
        finally:
            if sidecar_fd >= 0:
                os.close(sidecar_fd)
            tensor_tmp.unlink(missing_ok=True)
            sidecar_tmp.unlink(missing_ok=True)
    log.info("Saved manifold %r (%d layers) to %s",
             manifold.name, len(manifold.layers), path)


def load_manifold(
    path: str | Path, *, verify_manifest: bool = True,
) -> Manifold:
    """Load a fitted manifold, verifying its targeted manifest pair by default."""
    from saklas.io.manifold_folder import _locked_manifest, manifold_pair_lock

    resolved = Path(path)
    if verify_manifest:
        # Global lock order for fitted artifacts is folder -> tensor pair.
        # Fit/publication already uses this order; readers must match it or a
        # reader holding the pair can deadlock a fitter holding the folder.
        with _locked_manifest(resolved.parent):
            with manifold_pair_lock(resolved):
                return _load_manifold_locked(
                    resolved, verify_manifest=verify_manifest,
                )
    with manifold_pair_lock(resolved):
        return _load_manifold_locked(resolved, verify_manifest=False)


def _load_manifold_locked(
    path: str | Path, *, verify_manifest: bool = True,
) -> Manifold:
    """Read one fitted manifold while its tensor/sidecar pair lock is held."""
    path = Path(path)
    manifest_path = path.parent / "manifold.json"
    verified_tensor_sha256: str | None = None
    if verify_manifest and manifest_path.exists():
        from saklas.io.manifold_folder import ManifoldFormatError
        from saklas.io.packs import verify_integrity

        try:
            with open(manifest_path) as handle:
                manifest = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise ManifoldFormatError(
                f"manifold manifest is unreadable at {manifest_path}: {exc}"
            ) from exc
        from saklas.io.manifolds import MANIFOLD_FORMAT_VERSION

        if manifest.get("format_version") != MANIFOLD_FORMAT_VERSION:
            raise ManifoldFormatError(
                f"manifold format_version={manifest.get('format_version')!r}; "
                f"need exactly {MANIFOLD_FORMAT_VERSION}"
            )
        fit_mode = manifest.get("fit_mode", "authored")
        if not isinstance(manifest.get("name"), str) or fit_mode not in {
            "authored", "pca", "spectral", "auto", "baked",
        }:
            raise ManifoldFormatError("manifold manifest identity is incomplete")
        files = manifest.get("files", {})
        if not isinstance(files, dict):
            raise ManifoldFormatError("manifold integrity manifest is not an object")
        pair_names = (path.name, path.with_suffix(".json").name)
        missing = [name for name in pair_names if name not in files]
        if missing:
            raise ManifoldFormatError(
                f"manifold integrity manifest has no proof for {missing}"
            )
        expected = {name: files[name] for name in pair_names}
        ok, bad = verify_integrity(path.parent, expected)
        if not ok:
            raise ManifoldFormatError(
                f"manifold integrity check failed for {path.name}: {bad}"
            )
        verified_tensor_sha256 = str(expected[path.name])
    tensors = load_file(str(path))
    with open(path.with_suffix(".json")) as f:
        sidecar = json.load(f)
    if verified_tensor_sha256 is not None:
        sidecar["_tensor_sha256"] = verified_tensor_sha256
    if (
        verify_manifest
        and manifest_path.exists()
        and sidecar.get("fit_mode", "authored") != "baked"
    ):
        from saklas.io.manifold_folder import ManifoldFolder, ManifoldFormatError

        # ``load_manifold`` holds the folder lock outside the pair lock.
        current_nodes = ManifoldFolder.load(
            path.parent, verify_manifest=False,
        ).nodes_sha256()
        if sidecar.get("nodes_sha256") != current_nodes:
            raise ManifoldFormatError(
                f"fitted manifold {path.name} is stale for the live corpus/"
                "domain/template inputs; refit before loading"
            )
        if sidecar.get("fit_policy_version") != MANIFOLD_FIT_POLICY_VERSION:
            raise ManifoldFormatError(
                f"fitted manifold {path.name} uses an older numerical fit "
                "policy; refit it with the current saklas"
            )
    if verify_manifest and sidecar.get("method") == "merge":
        from saklas.io.manifold_folder import MERGE_BAKE_POLICY

        if sidecar.get("bake_policy") != MERGE_BAKE_POLICY:
            from saklas.io.manifold_folder import ManifoldFormatError

            raise ManifoldFormatError(
                "legacy projected bake has no current additive/Mahalanobis "
                "policy stamp; rebake the expression"
            )
        components = sidecar.get("components")
        if isinstance(components, dict) and any(
            isinstance(info, dict) and info.get("project_away") is not None
            for info in components.values()
        ):
            from saklas.io.manifold_folder import ManifoldFormatError

            raise ManifoldFormatError(
                "legacy projected bake used Euclidean projection; rebake the "
                "expression under the current Mahalanobis-only policy"
            )

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
            # Fuzzy-manifold σ-field — present only on curved fits run with the
            # within-node spread pass; absent ⇒ ``None`` ⇒ zero-thickness wire
            # (the exact legacy curved behavior).
            sigma_rbf_weights=parts.get("sigma_rbf_weights"),
            sigma_poly_coeffs=parts.get("sigma_poly_coeffs"),
        )

    domain = domain_from_spec(sidecar["domain"])
    if node_coords is None:
        node_coords = torch.zeros(0, domain.intrinsic_dim)

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
        mahalanobis_share=mahalanobis_share,
        origin=origin,
    )


def transfer_manifold_subspaces(
    src: Manifold,
    alignment: dict[int, torch.Tensor],
    *,
    whitener: "LayerWhitener | None",
    from_model: str,
    to_model: str,
) -> Manifold:
    """Map a fitted manifold's per-layer subspaces into a target model's space.

    The pure-tensor core of the cross-model Procrustes transfer (the folder
    read/write orchestration stays in :func:`saklas.io.manifold_lifecycle.
    transfer_manifold`).  Takes the already-loaded **source** ``Manifold``, a
    per-layer alignment map ``{layer: M_L}`` (``v_tgt = M_L @ v_src``, the shape
    :func:`saklas.io.alignment.fit_alignment` produces), and the **target**
    model's whitener, and returns a new ``Manifold`` whose subspaces live in
    target space.

    Each covered layer's affine subspace maps as ``mean_tgt = M_L @ mean_src``
    and ``basis_tgt = basis_src @ M_Lᵀ`` (each basis row transforms like a
    vector).  Layers the alignment doesn't cover are dropped.  The RBF
    interpolant fields (``node_params`` / ``rbf_weights`` / ``poly_coeffs`` /
    ``coord_offset`` / ``coord_scale``) and the shared ``node_coords`` live in
    subspace/authoring-coordinate space, not model space, so they ride through
    untouched — the subspace relocates via the transformed ``mean``/``basis``
    and the in-subspace parameterization is invariant.

    **Target-metric share re-bake (mandatory).**  The source fit's per-layer
    Mahalanobis ``share`` is a per-model quantity (``Σ`` belongs to
    ``from_model``), so it can't carry across.  The target ``whitener`` is
    **required** and must cover every transferred layer (all-or-nothing,
    mirroring the fit gate); the share is recomputed in target space via
    :func:`subspace_share` (``sqrt(Σ_k coordsᵀ (B_tgt Σ_tgt⁻¹ B_tgtᵀ) coords)``
    — the same formula the fit pipeline bakes).  A missing or non-covering
    whitener raises :class:`~saklas.core.mahalanobis.WhitenerError`; there is no
    Euclidean rebake.  ``origin`` (the per-layer foot of the *source* neutral
    mean) is per-model too, so it is cleared — the apply path falls back to a
    zero-coord seed per layer.

    Folder-level format guards (empty alignment, source fit missing) are the
    caller's concern; this function raises only :class:`~saklas.core.
    mahalanobis.WhitenerError` (missing / partial target whitener) and
    ``ValueError`` when ``alignment`` covers none of the source's fitted layers.
    """
    from dataclasses import replace as _dc_replace

    from saklas.core.mahalanobis import WhitenerError

    # Map each covered layer's subspace into target space.  ``M_L`` is
    # ``(D_tgt, D_src)`` so ``mean_tgt = M_L @ mean_src`` and each basis row
    # transforms the same way → ``basis_tgt = basis_src @ M_L^T``.
    new_layers: dict[int, LayerSubspace] = {}
    for layer, sub in src.layers.items():
        M_L = alignment.get(layer)
        if M_L is None:
            continue
        M = M_L.to(dtype=torch.float32)
        mean_f = sub.mean.to(torch.float32)
        basis_f = sub.basis.to(torch.float32)
        mean_tgt = (M @ mean_f).to(dtype=sub.mean.dtype)
        basis_tgt = (basis_f @ M.transpose(0, 1)).to(dtype=sub.basis.dtype)
        new_layers[layer] = _dc_replace(sub, mean=mean_tgt, basis=basis_tgt)

    if not new_layers:
        raise ValueError(
            f"alignment for {from_model!r} → {to_model!r} covered none of the "
            f"source manifold's fitted layers ({sorted(src.layers)})"
        )

    # The source model's Mahalanobis share is per-model (Σ and the neutral
    # activations are both ``from_model`` quantities), so it's invalid in
    # ``to_model`` space.  The **target** whitener is mandatory and must cover
    # every transferred layer (all-or-nothing, mirroring the fit gate);
    # recompute the share in target space.  No Euclidean rebake — a missing /
    # partial whitener is an error.
    if whitener is None or not whitener.covers_all(new_layers.keys()):
        raise WhitenerError(
            "manifold transfer requires a Mahalanobis whitener covering every "
            f"transferred layer {sorted(new_layers.keys())}; generate neutral "
            "activations for the TARGET model first (the Euclidean path is gone)"
        )

    new_share: dict[int, float] = {}
    for layer, sub_tgt in new_layers.items():
        sub_f = sub_tgt.to(device=torch.device("cpu"), dtype=torch.float32)
        # ``coords`` are the reduced node values in subspace-coordinate space —
        # invariant under the model-space alignment, so identical to the source
        # fit.  ``subspace_share`` computes the μ-centered whitened spread
        # ``sqrt(Σ_k c_kᵀ M_R c_k)`` (``M_R = B_tgt Σ_tgt⁻¹ B_tgtᵀ`` via
        # ``subspace_gram``, the *target* Σ⁻¹ restricted to the transferred
        # basis) — the same formula the fit pipeline bakes, now in target space.
        # It μ-centers internally only if fed μ-centered coords, so do the
        # centering here: flat fits carry neutral-anchored real coords in
        # ``node_coords``; curved fits read μ-centered node values off the RBF.
        if sub_f.is_affine:
            coords = sub_f.node_coords  # (K, R) neutral-anchored
            if coords is None:  # affine ⇒ node_coords set; sidecar corruption guard
                raise SaklasError(
                    "transfer_manifold_subspaces: affine LayerSubspace has"
                    " node_coords=None — the saved manifold sidecar may be corrupt"
                )
        else:
            _np, _rw, _pc = sub_f.rbf_params()
            coords = eval_rbf(_np, _rw, _pc, _np)  # (K, R)
        mu_coords = coords - coords.mean(dim=0, keepdim=True)  # μ-center
        new_share[layer] = subspace_share(
            mu_coords, sub_f.basis, whitener=whitener, layer=layer,
        )

    return _dc_replace(
        src, layers=new_layers,
        mahalanobis_share=new_share,
        # ``origin`` is the per-layer foot of the *source* model's neutral mean
        # — a per-model quantity invalid in target space (same reason the share
        # is cleared); the apply path falls back to a zero-coord seed per layer.
        origin={},
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
    warm_start: torch.Tensor | None = None,
    warm_iter: int = DEFAULT_INVERSION_WARM_ITER,
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
    final residual per query wins.  Used by the naturalness eval,
    ``Monitor.score_aggregate``, and the curved-probe per-token read --
    never the steering hot path.

    ``warm_start`` ``(.., n)`` (the curved-probe foot-follower) seeds from a
    carried previous-token foot instead of the nearest-node scan: the read
    runs ``warm_iter`` LM steps over just two restarts -- the carried foot and
    the single nearest fit node (a cheap basin-jump safety net) -- because a
    one-decode-step activation drift leaves the foot already near this token's
    nearest point.  The best-residual restart still wins, so a genuine basin
    jump falls back to the nearest-node chain.  ``None`` is the cold path
    (``n_restarts`` nearest-node seeds, ``max_iter`` steps), bit-for-bit
    unchanged.
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

    def _eval(p: torch.Tensor) -> torch.Tensor:
        # Inline normalize (avoids ``subspace._normalize`` reaching for the
        # subspace's own offset/scale on a possibly-foreign device).
        return eval_rbf(np_, rw, pc, (domain.embed(p) - offset) / scale)

    # Reduced values at the fit nodes -- the RBF is exact at ``node_params``
    # so this recovers the per-node centroids in reduced coords without a
    # stored field.  Used to pick each query's nearest node(s) as seeds.
    node_vals = eval_rbf(np_, rw, pc, np_)  # (K, R)
    if warm_start is not None:
        # Warm path: seed from the carried foot + the single nearest fit node
        # (basin-jump safety net), and take only ``warm_iter`` LM steps.
        ws = warm_start.to(device=device, dtype=dtype).reshape(N, n)  # (N, n)
        near1 = torch.cdist(flat, node_vals).topk(
            1, dim=-1, largest=False,
        ).indices.squeeze(-1)  # (N,)
        p = domain.clamp_position(
            torch.stack([ws, node_coords[near1]], dim=1)  # (N, 2, n)
        )
        iters = int(warm_iter)
    else:
        restarts = max(1, min(int(n_restarts), K))
        seed_idx = torch.cdist(flat, node_vals).topk(
            restarts, dim=-1, largest=False,
        ).indices  # (N, S)
        p = domain.clamp_position(node_coords[seed_idx])  # (N, S, n)
        iters = int(max_iter)
    q = flat.unsqueeze(1)  # (N, 1, R) -- broadcasts over the S restarts

    # Each step shares the LM body with the steering foot-follower via
    # ``_gn_step``; ``q`` is ``(N, 1, R)`` and broadcasts over the ``S``
    # restarts.  The internal ``reshape(-1, n, n)`` there also dodges
    # ``torch.linalg.solve``'s size-1-leading-batch out-resize warning on MPS.
    for _ in range(iters):
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


def manifold_is_affine(manifold: "Manifold") -> bool:
    """True iff every layer subspace is flat — an affine ``%`` joins the merge.

    A fit is all-affine (``fit_mode=pca``) or all-curved (authored / spectral);
    a curved ``%`` gets its own two-op instead.
    """
    layers = getattr(manifold, "layers", None)
    if not layers:
        return False
    return all(sub.is_affine for sub in layers.values())


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
    "fit_rbf_smoothed",
    "RbfFitPlan",
    "prepare_rbf_fit_plan",
    "eval_rbf",
    "eval_rbf_jacobian",
    "rbf_cardinal_weights",
    "fit_layer_subspace",
    "decompose",
    "subspace_inject",
    "PcaDiagnostics",
    "SpectralDiagnostics",
    "TopologyCandidate",
    "TopologyChoice",
    "derive_pca_coords",
    "derive_spectral_coords",
    "discover_coords",
    "neutral_layout_coord",
    "select_topology",
    "compute_node_centroid",
    "compute_node_activation_rows",
    "compute_manifold_node_stats",
    "compute_node_reduced_covariance_from_rows",
    "compute_store_reduced_covariances",
    "save_manifold",
    "load_manifold",
    "invert_parameterization",
    "manifold_is_affine",
]
