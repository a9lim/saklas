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
how :mod:`saklas.core.capture` holds the low-level extraction primitives.
The RBF fit solves a small dense symmetric-indefinite saddle system with
``torch.linalg.solve`` -- node counts are tiny (on the order of ten to
thirty) and fitting is a one-shot operation, not a hot path. scipy is not
pulled in. ``eval_rbf``, :func:`eval_rbf_jacobian`, :func:`_gn_step` and
:func:`subspace_inject` are the functions reachable from the generation
hot path (``subspace_inject`` is the unified along/onto subspace/manifold
injection — the single steering backend); all are allocation-light and
free of host syncs.

Discover-mode coordinate derivation and ``fit_mode="auto"`` topology
selection live next door in :mod:`saklas.core.topology`, which imports the
domains and per-layer fit primitives from here.  The on-disk tensor codec
and the activation-row spool live in :mod:`saklas.io.manifold_tensors`.
"""
from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence, cast

import torch

from saklas.core.errors import SaklasError, is_out_of_memory_error

if TYPE_CHECKING:
    from saklas.core.mahalanobis import LayerWhitener
    from saklas.io.alignment import LayerAlignment
    # The activation-row spool + the manifold tensor codec live in the io
    # layer (``io.manifold_tensors``) now; the fit-capture math here still
    # produces/consumes the spool, so it's a forward-referenced type +
    # lazy runtime import (the established core -> io reference pattern).
    from saklas.io.manifold_tensors import ActivationRowStore

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
    HTTP layer through the shared :class:`SaklasError` MRO; CLI handlers print
    the message and recover.
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
        return {
            "type": "custom",
            "embed_dim": self._embed_dim,
            "bounds": (
                None
                if self._bounds is None
                else [[lo, hi] for lo, hi in self._bounds]
            ),
        }


def validate_domain_spec(spec: Any) -> dict[str, Any]:
    """Validate the exact current tagged-union domain wire shape."""
    if not isinstance(spec, dict):
        raise ValueError("manifold domain must be an object")
    kind = spec.get("type")
    if kind == "box":
        if set(spec) != {"type", "axes"} or not isinstance(spec["axes"], list) or not spec["axes"]:
            raise ValueError("box domain requires exactly type + non-empty axes")
        for index, axis in enumerate(spec["axes"]):
            if not isinstance(axis, dict) or set(axis) != {
                "name", "periodic", "period", "lo", "hi",
            }:
                raise ValueError(f"box axis {index} has a non-current schema")
            if not isinstance(axis["name"], str) or not axis["name"]:
                raise ValueError(f"box axis {index} name must be non-empty")
            if not isinstance(axis["periodic"], bool):
                raise ValueError(f"box axis {index} periodic must be bool")
            for key in ("period", "lo", "hi"):
                value = axis[key]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    raise ValueError(f"box axis {index} {key} must be finite")
            if float(axis["period"]) <= 0 or float(axis["hi"]) <= float(axis["lo"]):
                raise ValueError(f"box axis {index} has invalid bounds/period")
    elif kind == "sphere":
        if set(spec) != {"type", "dim"} or isinstance(spec["dim"], bool) or not isinstance(spec["dim"], int) or spec["dim"] < 1:
            raise ValueError("sphere domain requires exactly type + positive integer dim")
    elif kind == "custom":
        if set(spec) != {"type", "embed_dim", "bounds"}:
            raise ValueError("custom domain requires exactly type, embed_dim, bounds")
        dim = spec["embed_dim"]
        if isinstance(dim, bool) or not isinstance(dim, int) or dim < 1:
            raise ValueError("custom domain embed_dim must be a positive integer")
        bounds = spec["bounds"]
        if bounds is not None and (
            not isinstance(bounds, list)
            or len(bounds) != dim
            or any(
                not isinstance(row, list)
                or len(row) != 2
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) for v in row)
                or float(row[1]) <= float(row[0])
                for row in bounds
            )
        ):
            raise ValueError("custom domain bounds must be null or one finite [lo, hi] pair per dimension")
    else:
        raise ValueError(f"unknown manifold domain type {kind!r}")
    return spec


def normalize_domain_spec(spec: Any) -> dict[str, Any]:
    """Normalize ergonomic authoring input into the exact persisted union."""
    if not isinstance(spec, dict):
        raise ValueError("manifold domain must be an object")
    kind = spec.get("type")
    if kind == "box" and set(spec) == {"type", "axes"} and isinstance(spec["axes"], list):
        normalized = {
            "type": "box",
            "axes": [
                {
                    "name": axis.get("name", f"axis{index}"),
                    "periodic": axis.get("periodic", False),
                    "period": axis.get("period", 1.0),
                    "lo": axis.get("lo", 0.0),
                    "hi": axis.get("hi", 1.0),
                }
                for index, axis in enumerate(spec["axes"])
                if isinstance(axis, dict)
            ],
        }
    elif kind == "custom" and set(spec) <= {"type", "embed_dim", "bounds"}:
        normalized = {**spec, "bounds": spec.get("bounds")}
    else:
        normalized = dict(spec)
    return validate_domain_spec(normalized)


def domain_from_spec(spec: dict[str, Any]) -> ManifoldDomain:
    """Build a :class:`ManifoldDomain` from a ``manifold.json`` domain object."""
    spec = validate_domain_spec(spec)
    kind = spec["type"]
    if kind == "box":
        axes = [
            BoxAxis(
                name=a["name"],
                periodic=a["periodic"],
                period=float(a["period"]),
                lo=float(a["lo"]),
                hi=float(a["hi"]),
            )
            for a in spec["axes"]
        ]
        return BoxDomain(axes)
    if kind == "sphere":
        return SphereDomain(int(spec["dim"]))
    if kind == "custom":
        return CustomDomain(
            spec["embed_dim"], bounds=spec["bounds"],
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
    ``n = R``).  Ordinary fits use the identity authoring→reduced map;
    rectangular cross-model transfers may carry an explicit ``affine_map`` so
    ``manifold_point(c) = mean + c @ affine_map @ basis`` while ``basis`` stays
    orthonormal.  The
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
    affine_map: torch.Tensor | None = None
    # (m, R) authoring-to-reduced coordinate map for affine subspaces.  ``None``
    # is the canonical identity map (m == R). Cross-model transfer may need a
    # non-isometric map after orthonormalizing a rectangular mapped basis; this
    # explicit factor preserves the exact world surface without weakening the
    # runtime's orthonormal-basis invariant.
    sigma_rbf_weights: torch.Tensor | None = None
    sigma_poly_coeffs: torch.Tensor | None = None
    # The **fuzzy-manifold σ-field** (raw curved subspaces only). A separate
    # ``r**3`` RBF over
    # the **same** normalized ``node_params`` that interpolates per-node
    # ``log σ`` — the within-node off-surface activation spread (the corpus a
    # node produces scatters off the mean surface; ``σ`` is that scatter's
    # normal-projected std, a tube thickness).  Kept separate from the surface
    # RBF (rather than appended as an extra value column) so the ``(R,)``-shape
    # contracts the surface consumers rely on are untouched; ``sigma_at`` is the
    # one extra ``eval_rbf`` (``O(K)``) paid only on the already-slow curved
    # path. ``sigma_rbf_weights`` is ``(K, 1)``, ``sigma_poly_coeffs`` is
    # ``(m+1, 1)``. Affine and SAE-curved fits do not model a raw-activation
    # tube and therefore carry neither sigma tensor.

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

    def validate_structure(
        self, *, feature_space: str, expected_node_count: int,
    ) -> None:
        """Validate the exact affine/curved discriminated tensor shape."""
        if self.mean.ndim != 1:
            raise ValueError("LayerSubspace mean must have shape (D,)")
        if self.basis.ndim != 2 or self.basis.shape[0] < 1:
            raise ValueError("LayerSubspace basis must have shape (R, D) with R >= 1")
        rank, dim = self.basis.shape
        if self.mean.shape[0] != dim:
            raise ValueError("LayerSubspace mean dimension must match basis width")
        if self.coord_offset.ndim != 1 or self.coord_scale.ndim != 1:
            raise ValueError("LayerSubspace coordinate normalization must be vectors")
        if self.coord_offset.shape != self.coord_scale.shape:
            raise ValueError("LayerSubspace coord_offset/coord_scale shapes must match")
        embed_dim = int(self.coord_offset.shape[0])
        surface = (self.node_params, self.rbf_weights, self.poly_coeffs)
        surface_present = tuple(value is not None for value in surface)
        sigma_present = (
            self.sigma_rbf_weights is not None,
            self.sigma_poly_coeffs is not None,
        )
        if any(surface_present) and not all(surface_present):
            raise ValueError("curved LayerSubspace requires the complete RBF triple")
        if sigma_present[0] != sigma_present[1]:
            raise ValueError("LayerSubspace sigma tensors must be present as a pair")
        if not any(surface_present):
            if self.node_coords is None:
                raise ValueError("affine LayerSubspace requires node_coords")
            if self.node_coords.shape != (expected_node_count, rank):
                raise ValueError(
                    "affine LayerSubspace node_coords must have shape (K, R)"
                )
            if self.affine_map is None:
                if embed_dim != rank:
                    raise ValueError(
                        "affine LayerSubspace without affine_map requires m == R"
                    )
            elif self.affine_map.shape != (embed_dim, rank):
                raise ValueError("affine_map must have shape (m, R)")
            if any(sigma_present):
                raise ValueError("affine LayerSubspace cannot carry a sigma field")
            return
        if self.node_coords is not None or self.affine_map is not None:
            raise ValueError(
                "curved LayerSubspace cannot carry affine node_coords/affine_map"
            )
        if feature_space == "raw" and not all(sigma_present):
            raise ValueError("raw curved LayerSubspace requires a sigma field")
        assert self.node_params is not None
        assert self.rbf_weights is not None
        assert self.poly_coeffs is not None
        if self.node_params.shape != (expected_node_count, embed_dim):
            raise ValueError("curved node_params must have shape (K, m)")
        if self.rbf_weights.shape != (expected_node_count, rank):
            raise ValueError("curved rbf_weights must have shape (K, R)")
        if self.poly_coeffs.shape != (embed_dim + 1, rank):
            raise ValueError("curved poly_coeffs must have shape (m + 1, R)")
        if all(sigma_present):
            assert self.sigma_rbf_weights is not None
            assert self.sigma_poly_coeffs is not None
            if self.sigma_rbf_weights.shape != (expected_node_count, 1):
                raise ValueError("sigma_rbf_weights must have shape (K, 1)")
            if self.sigma_poly_coeffs.shape != (embed_dim + 1, 1):
                raise ValueError("sigma_poly_coeffs must have shape (m + 1, 1)")

    @classmethod
    def affine(
        cls,
        mean: torch.Tensor,
        basis: torch.Tensor,
        *,
        node_coords: torch.Tensor | None = None,
        affine_map: torch.Tensor | None = None,
    ) -> "LayerSubspace":
        """Build a flat (affine, no-RBF) subspace from ``mean`` + ``basis``.

        By default authoring coordinates map to reduced coordinates by identity
        (``coord_offset = 0``, ``coord_scale = 1``), so
        ``eval_at(c) = c @ basis + mean`` exactly.  ``affine_map`` optionally
        supplies an ``(m, R)`` map after cross-model reparameterization.
        ``basis`` is ``(R, D)``;
        the implied intrinsic dimension is ``n = R`` (the surface fills the
        span).  Backs the folded-vector / flat-subspace representation
        (Phase 2 §1).  ``node_coords`` ``(K, R)`` carries the per-layer real
        neutral-anchored node positions (the steer-target source — §5);
        ``None`` for a bare span with no associated nodes.
        """
        r = int(basis.shape[0])
        m = r
        if affine_map is not None:
            if affine_map.ndim != 2 or affine_map.shape[1] != r:
                raise ValueError(
                    f"affine_map must have shape (m, {r}), got "
                    f"{tuple(affine_map.shape)}"
                )
            m = int(affine_map.shape[0])
        return cls(
            mean=mean,
            basis=basis,
            node_params=None,
            rbf_weights=None,
            poly_coeffs=None,
            coord_offset=torch.zeros(m, device=mean.device, dtype=mean.dtype),
            coord_scale=torch.ones(m, device=mean.device, dtype=mean.dtype),
            node_coords=node_coords,
            affine_map=affine_map,
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
        amap_k = (
            self.affine_map[:, idx].contiguous()
            if self.affine_map is not None else None
        )
        return LayerSubspace.affine(
            mean_k, basis_k, node_coords=nc_k, affine_map=amap_k,
        )

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
            affine_map=_cast(self.affine_map),
            sigma_rbf_weights=_cast(self.sigma_rbf_weights),
            sigma_poly_coeffs=_cast(self.sigma_poly_coeffs),
        )

    @property
    def has_sigma(self) -> bool:
        """True iff this subspace carries a fuzzy-manifold σ-field.

        A raw curved subspace fitted with the within-node spread pass; ``False``
        for affine and SAE-curved fits, which do not model raw tube density.
        """
        return self.sigma_rbf_weights is not None and self.sigma_poly_coeffs is not None

    def _normalize(self, embedded: torch.Tensor) -> torch.Tensor:
        return (embedded - self.coord_offset) / self.coord_scale

    def eval_at(self, embedded: torch.Tensor) -> torch.Tensor:
        """World-space activation ``(.., D)`` at embedded domain coords ``(.., m)``."""
        if self.is_affine:
            # Flat: the canonical representation has an identity
            # authoring→reduced map.  Rectangular cross-model transfer may
            # carry an explicit map after orthonormalizing the target basis.
            reduced = embedded if self.affine_map is None else embedded @ self.affine_map
            return reduced @ self.basis + self.mean
        reduced = eval_rbf(*self.rbf_params(), self._normalize(embedded))
        return reduced @ self.basis + self.mean

    def sigma_at(self, embedded: torch.Tensor) -> torch.Tensor:
        """Within-node off-surface spread ``σ`` at embedded coords ``(.., m)``.

        Interpolates the per-node ``log σ`` field through the σ-RBF over the
        same normalized ``node_params``, then exponentiates — so the result is
        a positive ``(..,)`` thickness in the layer's reduced-coordinate units
        (the same units ``H_n`` is measured in, since the basis is
        orthonormal).  Returns an all-zeros ``(..,)`` when the subspace carries
        no σ-field (:attr:`has_sigma` false): affine and SAE-curved fits.
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
            # d(embedded @ A @ basis + mean)/d embedded = basis.T @ A.T,
            # position-independent.  ``A`` is identity on ordinary fits.
            jac = self.basis.transpose(-1, -2)
            if self.affine_map is not None:
                jac = jac @ self.affine_map.transpose(-1, -2)
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
    fit_result: dict[str, torch.Tensor] | None = None,
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
    # Project the μ-centered roster exactly once. This is the historical share
    # computation order; anchor-relative RBF targets differ only by one reduced
    # offset, so they do not need a second K×D by D×R product.
    mu_coords = X @ basis.T
    coords = mu_coords + ((mu - anchor) @ basis.T)
    if fit_result is not None:
        fit_result["mu_coords"] = mu_coords

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

class _OmittedNodeRoster(list[str | None]):
    """Distinct default: an explicitly supplied empty roster is malformed."""


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
    node_roles: list[str | None] = field(default_factory=_OmittedNodeRoster)
    # Per-node conceptual ``kind`` — ``"abstract"`` (a trait/quality, e.g.
    # ``happy``) or ``"concrete"`` (an entity, e.g. ``pirate``), aligned with
    # ``node_labels``.  ``None`` = unspecified.  A *generation-time* attribute
    # only: it selects the system template and the elicitation role label
    # (``someone {label}`` vs ``{label}``) when authoring a node's
    # conversational corpus.  It does NOT feed the fit — extraction pools in
    # standard-assistant space (swap-back) regardless — so it is carried for
    # provenance / regeneration, not consumed by the fit-time capture pass.
    node_kinds: list[str | None] = field(default_factory=_OmittedNodeRoster)
    # Per-layer Mahalanobis share weight recorded at fit time —
    # ``share_L = ‖Bᵀ coords_k‖_M`` summed over
    # nodes, the subspace-restricted analogue of vector steering's
    # ``‖d‖_M`` bake score (see ``LayerWhitener.subspace_gram``). Coverage is
    # exact: every fitted layer has one share value.
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

    def __post_init__(self) -> None:
        # The programmatic constructor emits the same exact full-length roster
        # as the current wire format. ``None`` is a real per-node value; an
        # omitted constructor argument is not a second in-memory shape.
        if isinstance(self.node_roles, _OmittedNodeRoster):
            self.node_roles = [None] * len(self.node_labels)
        if isinstance(self.node_kinds, _OmittedNodeRoster):
            self.node_kinds = [None] * len(self.node_labels)

    @property
    def layer_indices(self) -> list[int]:
        return sorted(self.layers)

    def validate_runtime_geometry(self) -> None:
        """Require the complete fitted geometry consumed by live steering.

        Persistence readers validate their wire schema; this closes the same
        contract over programmatically constructed objects before they enter a
        hot path.  Optionality remains structural only: affine layers have no
        RBF or tube, and SAE-space curved fits deliberately have no raw-space
        tube.  Missing bakes are never interpreted as an older representation.
        """
        if not self.layers:
            raise ValueError(f"manifold {self.name!r} has no fitted layers")
        node_count = len(self.node_labels)
        intrinsic_dim = int(self.domain.intrinsic_dim)
        if self.node_coords.ndim != 2 or self.node_coords.shape != (
            node_count, intrinsic_dim,
        ):
            raise ValueError(
                f"manifold {self.name!r} node_coords must have shape (K, n)"
            )
        for field_name, values in (
            ("node_roles", self.node_roles),
            ("node_kinds", self.node_kinds),
        ):
            if len(values) != len(self.node_labels):
                raise ValueError(
                    f"manifold {self.name!r} {field_name} must align exactly "
                    "with node_labels"
                )
        layer_keys = set(self.layers)
        if set(self.mahalanobis_share) != layer_keys:
            raise ValueError(
                f"manifold {self.name!r} Mahalanobis shares must cover exactly "
                "its fitted layers"
            )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))  # pyright: ignore[reportUnnecessaryIsInstance]  # runtime-loaded artifacts can violate annotations
            or not math.isfinite(value)
            or value <= 0.0
            for value in self.mahalanobis_share.values()
        ):
            raise ValueError(
                f"manifold {self.name!r} Mahalanobis shares must be finite and positive"
            )
        curved = {idx: sub for idx, sub in self.layers.items() if not sub.is_affine}
        if curved and len(curved) != len(self.layers):
            raise ValueError(
                f"manifold {self.name!r} cannot mix affine and curved layers"
            )
        if set(self.origin) != set(curved):
            raise ValueError(
                f"manifold {self.name!r} origins must cover exactly its curved layers"
            )
        for idx, value in self.origin.items():
            if value.ndim != 1 or value.shape != (intrinsic_dim,):
                raise ValueError(
                    f"manifold {self.name!r} origin for layer {idx} must "
                    "have shape (n,)"
                )
        for idx, sub in self.layers.items():
            try:
                sub.validate_structure(
                    feature_space=self.feature_space,
                    expected_node_count=len(self.node_labels),
                )
            except ValueError as exc:
                raise ValueError(
                    f"manifold {self.name!r} layer {idx}: {exc}"
                ) from exc

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

        Returns ``None`` when the nearest node opts out of role substitution. The
        return value rides through to ``session._active_role`` so the
        generation prefill re-applies the substitution at decode time,
        producing role-paired manifold steering (Phase A.3). The role roster is
        exact and aligned with :attr:`node_labels`.
        """
        idx = self.nearest_node_index(position)
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
    spread → coherent strong steer) and **collapses** the foot toward ``0`` on
    each *ablation* axis (removing the requested fraction of that component).
    ``kappa`` stores the requested per-axis ablation coefficient: ``0`` on push
    axes, ``1`` for full mean replacement, fractional for partial ablation, and
    signed/outside ``[0, 1]`` for the grammar's extrapolating forms.  The hook
    divides it by the affine push gain before calling the shared kernel, so the
    kernel's ``along · kappa`` product is exactly the user coefficient rather
    than the affine steering gain.  ``share`` is the
    un-normalized per-layer budget weight — the push-displacement magnitude
    ``‖Δ_L‖_M`` under the required covering whitener; a pure-ablation layer
    weights by the summed
    ablation magnitude instead.  The apply path normalizes it across layers
    (mean-1).  ``target_coord`` is correspondingly a whitened-unit direction on
    the whitened path, with magnitude carried by ``share``.

    Fields are keyed by layer index; only layers carrying at least one
    non-degenerate active direction (and present in ``neutral_means``) appear.
    """

    layers: dict[int, "LayerSubspace"]        # affine: mean = neutral_L, basis = ortho span
    target_coord: dict[int, torch.Tensor]     # (R_L,) the along target (poles / 0)
    share: dict[int, float]                   # ‖Δ_L‖, un-normalized budget weight
    kappa: dict[int, torch.Tensor] = field(   # (R_L,) per-axis requested ablation coefficient
        default_factory=dict
    )

    def __post_init__(self) -> None:
        # An empty synth is the explicit no-op result when every active
        # fragment cancels or degenerates; SteeringManager skips it.
        keys = set(self.layers)
        for name, mapping in (
            ("target_coord", self.target_coord),
            ("share", self.share),
            ("kappa", self.kappa),
        ):
            if set(mapping) != keys:
                raise ValueError(
                    f"SynthesizedSubspace {name} must cover exactly its layers"
                )
        for layer, sub in self.layers.items():
            if not sub.is_affine:
                raise ValueError("SynthesizedSubspace layers must be affine")
            if sub.mean.ndim != 1:
                raise ValueError("synthesized mean must have shape (D,)")
            if sub.basis.ndim != 2 or sub.basis.shape[0] < 1:
                raise ValueError("synthesized basis must have shape (R, D), R >= 1")
            rank = sub.rank
            dim = int(sub.basis.shape[1])
            if sub.mean.shape != (dim,):
                raise ValueError("synthesized mean width must match basis width")
            if sub.coord_offset.shape != (rank,) or sub.coord_scale.shape != (rank,):
                raise ValueError(
                    "synthesized coord_offset/coord_scale must have shape (R,)"
                )
            if (
                sub.node_coords is not None
                or sub.affine_map is not None
                or sub.sigma_rbf_weights is not None
                or sub.sigma_poly_coeffs is not None
                or sub.node_params is not None
                or sub.rbf_weights is not None
                or sub.poly_coeffs is not None
            ):
                raise ValueError(
                    "synthesized affine spans cannot carry artifact surface fields"
                )
            gram = sub.basis @ sub.basis.T
            identity = torch.eye(rank, dtype=gram.dtype, device=gram.device)
            if not torch.allclose(gram, identity, atol=1e-5, rtol=1e-5):
                raise ValueError("synthesized basis rows must be orthonormal")
            if self.target_coord[layer].shape != (rank,):
                raise ValueError("SynthesizedSubspace target_coord must have shape (R,)")
            if self.kappa[layer].shape != (rank,):
                raise ValueError("SynthesizedSubspace kappa must have shape (R,)")
            # Do not concatenate these device tensors just to validate them.
            # PyTorch 2.12's MPS ``cat`` can segfault in the Metal blit path
            # for the mixed view/stride shapes produced by dispatch-time
            # synthesis, taking the whole server down before generation.  Four
            # cold-path finite reductions are cheap and keep validation on the
            # tensors' native device without allocating a joined buffer.
            finite_parts = (
                sub.mean,
                sub.basis,
                self.target_coord[layer],
                self.kappa[layer],
            )
            if not all(bool(torch.isfinite(part).all()) for part in finite_parts):
                raise ValueError("SynthesizedSubspace tensors must be finite")
            if (
                isinstance(self.share[layer], bool)
                or not isinstance(self.share[layer], (int, float))  # pyright: ignore[reportUnnecessaryIsInstance]  # runtime-loaded values can violate annotations
                or not math.isfinite(self.share[layer])
                or self.share[layer] <= 0.0
            ):
                raise ValueError(
                    "SynthesizedSubspace shares must be finite and positive"
                )


def synthesize_subspace(
    push: Sequence[
        tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], float]
    ],
    ablate: Sequence[tuple[dict[int, torch.Tensor], float]],
    neutral_means: dict[int, torch.Tensor],
    *,
    whitener: "LayerWhitener",
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

    Each **ablation** term is ``(directions, coeff)``: a per-layer ``(R_i, D)``
    (or ``(D,)``) direction set plus the signed fraction to remove.  Its target
    is the origin (``0``).  ``neutral_means`` supplies each layer's anchor
    (``mean``); it must cover every participating layer.

    Per layer (over the union of layers any term touches):

    - Orthonormalize the push span, project ablation directions out of it (push
      wins a shared direction), and build the reduced symmetric ablation
      operator ``A = Σ coeffᵢ uᵢuᵢᵀ`` on the remaining span.  Diagonalizing
      ``A`` gives an orthonormal ablation basis and one exact collapse
      coefficient per axis, including partial, repeated, and negative terms.
    - World push displacement ``Δ = Σ_push coeffᵢ·(coord_targetᵢ @ basis_rowsᵢ)``
      — each fragment's own ``(R_i,) @ (R_i, D) = (D,)`` world vector, scaled by
      its coeff.  ``target = B @ Δ`` is its coordinate in the merged basis.
      Because ``Δ`` lives in the push span and the ablation-only axes are its
      orthogonal complement, those axes get ``target ≈ 0``.
    - Per-axis ``kappa`` ``(R,)`` — ``0`` on the push span and the eigenvalues of
      ``A`` on the ablation-only complement.  The apply path gain-compensates
      these coefficients before the kernel evaluates
      ``p_new = q + along·(target − kappa·q)``.
    - ``share = ‖Δ‖`` (the world displacement magnitude); a pure-ablation layer
      uses a positive magnitude proxy because gain compensation makes the
      resulting collapse independent of cross-layer share normalization.

    **Whitened normalization (Mahalanobis-only).**  ``whitener`` is required and
    must cover every synthesized layer; the push is normalized in the Mahalanobis
    metric ``M_R = B Σ⁻¹ Bᵀ`` (the engine-wide read/fit metric):

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
      where its signal lives; ``along`` is a scale-stable strength knob.

    The metric gate is all-or-nothing: missing neutral means or partial whitener
    coverage raise :class:`~saklas.core.mahalanobis.WhitenerError` before
    synthesis, so a cross-layer profile can never mix metrics and there is no
    Euclidean path.  ``share`` carries the per-layer magnitude and ``target``
    carries only the whitened-unit direction.

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
    for dirs, _coeff in ablate:
        all_layers |= dirs.keys()

    from saklas.core.mahalanobis import WhitenerError

    present_layers = sorted(all_layers)
    missing_means = sorted(all_layers - set(neutral_means))
    if missing_means:
        raise WhitenerError(
            f"steering synthesis requires neutral means for every layer; "
            f"missing {missing_means}"
        )
    if not whitener.covers_all(present_layers):
        raise WhitenerError(
            "steering synthesis requires a Mahalanobis whitener covering "
            f"every layer {present_layers}"
        )
    maha = whitener

    layers: dict[int, "LayerSubspace"] = {}
    target_coord: dict[int, torch.Tensor] = {}
    share: dict[int, float] = {}
    kappa: dict[int, torch.Tensor] = {}

    for L in sorted(all_layers):
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
            if abs(float(coeff)) < eps:
                continue
            B_i = basis_dirs.get(L)
            if B_i is None:
                continue
            # ``synthesize_subspace`` is the join point for profiles from
            # several artifact families.  Fitted manifolds already follow the
            # model device, while external J-lens shards and provider SAE
            # payloads are CPU-backed and may be promoted lazily.  Treat the
            # neutral mean as the canonical per-layer runtime device and
            # co-locate every fragment here; merely changing dtype preserves a
            # stray CPU device and makes a valid mixed J-lens + SAE recipe fail
            # before its first token.
            B_i = B_i.to(device=mean.device, dtype=torch.float32)
            if B_i.ndim == 1:
                B_i = B_i.reshape(1, -1)
            if float(torch.linalg.matrix_norm(B_i)) < eps:
                continue
            c_i = coord_dirs.get(L)
            if c_i is None:
                # No target coords for this layer ⇒ no displacement, but the
                # rows still join the span (a degenerate push = ablation).
                c_i = B_i.new_zeros(B_i.shape[0])
            c_i = c_i.to(device=mean.device, dtype=torch.float32).reshape(-1)
            push_rows.extend(B_i)
            push_frags.append((float(coeff), c_i @ B_i))    # (D,) raw world dir

        ablate_frags: list[tuple[float, torch.Tensor]] = []
        ablate_share = 0.0
        for dirs, coeff in ablate:
            coeff_f = float(coeff)
            if abs(coeff_f) < eps:
                continue
            d = dirs.get(L)
            if d is None:
                continue
            d = d.to(device=mean.device, dtype=torch.float32)
            if d.ndim == 1:
                d = d.reshape(1, -1)
            for row in d:
                if float(torch.linalg.vector_norm(row)) < eps:
                    continue
                row_norm = float(torch.linalg.vector_norm(row))
                unit = row / row_norm
                ablate_frags.append((coeff_f, unit))
                # A strictly positive proxy is sufficient here: hook-time gain
                # compensation cancels share from the actual ablation amount.
                ablate_share += abs(coeff_f) * max(
                    float(maha.mahalanobis_norm(L, row)), eps,
                )

        # Build the active push target in world coordinates first.  If all
        # pushes cancel, they must not suppress an ablation on the same axis.
        world_target = mean.new_zeros(mean.shape)
        raw_delta = mean.new_zeros(mean.shape)
        for cf, wd in push_frags:
            wn = float(maha.mahalanobis_norm(L, wd))
            raw_delta = raw_delta + cf * wd
            if wn > eps:
                world_target = world_target + (cf / wn) * wd
        has_push = float(torch.linalg.vector_norm(world_target)) >= eps

        if has_push:
            B_push, _ = _ortho_basis(push_rows)
        else:
            B_push = mean.new_zeros((0, mean.numel()))

        # Push wins shared directions.  The remaining ablation operator is
        # diagonalized in its own small span so coefficients survive exactly;
        # a per-axis mask cannot represent unequal non-orthogonal terms.
        projected_ablate: list[tuple[float, torch.Tensor]] = []
        for coeff_f, unit in ablate_frags:
            residual = unit
            if B_push.shape[0]:
                residual = residual - (residual @ B_push.T) @ B_push
            rn = float(torch.linalg.vector_norm(residual))
            if rn >= eps:
                projected_ablate.append((coeff_f, residual / rn))

        B_ab = mean.new_zeros((0, mean.numel()))
        ablate_eigenvalues = mean.new_zeros((0,))
        if projected_ablate:
            B_ab_span, _ = _ortho_basis([row for _cf, row in projected_ablate])
            reduced = B_ab_span.new_zeros((B_ab_span.shape[0], B_ab_span.shape[0]))
            for coeff_f, row in projected_ablate:
                coord = B_ab_span @ row
                reduced = reduced + coeff_f * torch.outer(coord, coord)
            # ``torch.linalg.eigh`` is still unavailable on MPS.  This is a
            # tiny rank-by-rank compose-time matrix (never a decode hot-path
            # operation), so diagonalize it on CPU and return the result to
            # the originating device.  Keeping this explicit also avoids
            # requiring users to opt into PyTorch's process-wide MPS fallback
            # just to author an ablation term.
            eig_device = reduced.device
            eigenvalues, eigenvectors = torch.linalg.eigh(reduced.cpu())
            eigenvalues = eigenvalues.to(eig_device)
            eigenvectors = eigenvectors.to(eig_device)
            keep = eigenvalues.abs() >= eps
            if bool(keep.any()):
                ablate_eigenvalues = eigenvalues[keep]
                B_ab = eigenvectors[:, keep].T @ B_ab_span

        if B_push.shape[0] == 0 and B_ab.shape[0] == 0:
            continue
        basis = torch.cat((B_push, B_ab), dim=0)

        if has_push:
            # Raw coeff-weighted displacement — its (whitened) magnitude is the
            # per-layer **profile** weight (``share``); the absolute node-distance
            # scale cancels under the apply-time mean-1 normalization, leaving
            # only the relative across-layer shape (steer where the signal is).
            share_L = float(maha.mahalanobis_norm(L, raw_delta))
            if share_L < eps:
                share_L = sum(
                    abs(cf) * float(maha.mahalanobis_norm(L, wd))
                    for cf, wd in push_frags
                )
            target_coord[L] = basis @ world_target       # ablation axes ≈ 0
        else:
            share_L = ablate_share
            target_coord[L] = basis.new_zeros(basis.shape[0])   # (R,) all ≈ 0
        if share_L < eps:
            continue
        kappa[L] = torch.cat((
            basis.new_zeros(B_push.shape[0]),
            ablate_eigenvalues.to(dtype=basis.dtype, device=basis.device),
        ))
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
    factor ``cap/‖h_new‖``.  The clamp form is deliberate: a
    ``where(post > cap, cap/post, 1)`` select would allocate a full-width
    ``ones_like`` temporary every fire.  The two forms agree on every
    non-degenerate ``h`` (they differ only at all-zero ``h`` — scale 0 vs 1 —
    where ``h_new ≈ 0`` makes the product ``0`` either way)."""
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
    load-bearing guarantees follow:

    - **Identity at rest.**  When the foot doesn't move (``p_new == p``, i.e.
      ``along == 0``) the frames coincide, every principal angle is 0, and the
      rotation is *exactly* the identity — so ``subspace_inject`` returns its
      input untouched regardless of foot-solve accuracy.  Anything that
      reprojects ``Hn`` onto the normal space instead corrupts an off-neutral
      activation by the residual's tangential part, which never vanishes at an
      approximate foot; the guarantee is why this is a rotation.
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
    """The unified two-operation manifold injection — saklas's one steering kernel.

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
      foot-solve accuracy; norm-preserving and lossless.  Tangential /
      directional; leaves ``H_o`` untouched.  By moving *on the surface* it never
      cuts through the off-manifold low-density region.

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

    The off-*subspace* residual ``H_o`` is **always kept verbatim**.  That is the
    composition invariant: ``H_o`` is the orthogonal complement of *this*
    subspace, i.e. every composing neighbor's span, so touching it would couple
    otherwise-orthogonal terms.  ``~``/``|`` semantics live inside the subspace
    instead, as push/ablation axes of the merged affine basis.

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
        # The surface fills the subspace, so the injected target is already in
        # the layer's orthonormal reduced frame: the foot is ``q`` exactly,
        # ``H_n ≡ 0`` and ``onto`` is vacuous (ignored).  Ordinary fits have an
        # identity authoring→reduced map; rectangular transfers bake their
        # explicit map into the resolved per-layer target before this kernel.
        # No GN solve, no RBF eval, no tangent Gram-solve — the cost the common
        # steering case can't pay.
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
    # Zero-thickness wire (σ-field absent — flat/SAE fits, which carry no tube):
    # scale ``H_n`` by ``(1 − o)`` — at ``o = 1`` the activation lands *on* the
    # mean surface.  Fuzzy (σ-field present): shrink ``‖H_n‖`` toward the local
    # within-node thickness ``σ(z)`` instead of toward 0, so ``o = 1`` lands
    # one-σ off the wire (the surface's *typical set*, a sample-like point)
    # rather than on the idealized centroid — the within-concept variety a hard
    # collapse erases is exactly what drives the strong-push mode-collapse the
    # open-frontier note flags.  The residual *direction* is preserved (only its
    # magnitude is rescaled), and a token already inside the tube
    # (``‖H_n‖ ≤ σ``) is never *expanded* (the ``(·)_+`` clamp).  ``σ(z) = 0``
    # ⇒ ``shrink = (1)_+ = 1`` ⇒ exactly the zero-thickness ``(1 − o)`` collapse.
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


# ------------------------------------------------------ centroid capture ---

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
    from saklas.core.capture import (
        _CAPTURE_BATCH_MAX,
        _encode_and_capture_all_batch,
        _prepare_capture_batch,
    )
    from saklas.io.manifold_tensors import ActivationRowStore

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
    # The accumulators already have the final layer-major K×D shape and sole
    # ownership here. Normalize them in place instead of allocating a second
    # complete fp32 centroid roster at the capture/fitting boundary.
    for idx in capture_layers:
        sums[idx].div_(divisors)
    return sums, retained


def compute_node_reduced_covariance_from_rows(
    activation_rows: dict[int, torch.Tensor],
    layer_subs: "dict[int, LayerSubspace]",
) -> dict[int, torch.Tensor]:
    """Within-node reduced covariance from retained pooled activations.

    ``activation_rows`` carries one node's ``{layer: (N, D)}`` pooled rows —
    the per-response activations :func:`compute_manifold_node_stats` retains
    when ``retain_rows=True``.  Projects them through each fitted layer's
    affine frame (``Z = (h − mean) · basisᵀ``) and returns the sample
    covariance ``{layer: (R, R)}`` (``N − 1`` denominator; zeros for a
    single-sample node).  The standalone sibling of
    :func:`compute_store_reduced_covariances`, which streams the same math
    over a whole disk-backed roster.
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
        centered_buffer = torch.empty(
            min(row_chunk, store.total_rows), rows.shape[1], dtype=torch.float32,
        )
        for start in range(0, store.total_rows, row_chunk):
            end = min(start + row_chunk, store.total_rows)
            centered = centered_buffer[:end - start]
            # Center before projection for numerical stability at the large
            # common-mode offsets residual streams can carry. ``out=`` keeps
            # the operation bounded and out-of-place even when the source row
            # store is already fp32 (where ``.to(float32)`` would alias it).
            torch.sub(rows[start:end], mean, out=centered)
            reduced_rows[start:end] = centered @ basis_t

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
    ``P = I − tt⁺`` and returns ``tr(P cov)/(R − rank(t))`` — the part of the
    node's scatter that lives *off* the mean surface, which is what the tube
    thickness should be (tangential scatter is the node sliding *along* the
    surface, expected and not thickness).  Degenerates to the full isotropic
    ``tr(cov)/R`` only when the actual tangent range fills the reduced subspace.
    Clamped non-negative (sample-covariance round-off).
    """
    if cov.shape != (R, R):
        raise ValueError(
            f"covariance must have shape ({R}, {R}), got {tuple(cov.shape)}"
        )
    if tangent.shape != (R, n):
        raise ValueError(
            f"tangent must have shape ({R}, {n}), got {tuple(tangent.shape)}"
        )
    return float(
        _off_surface_vars(
            cov.reshape(1, R, R), tangent.reshape(1, R, n), R, n,
        )[0]
    )


def _off_surface_vars(
    covs: torch.Tensor, tangents: torch.Tensor, R: int, n: int,
) -> torch.Tensor:
    """Batched counterpart to :func:`_off_surface_var`.

    ``covs`` is ``(K, R, R)`` and ``tangents`` is ``(K, R, n)``.  Returns one
    non-negative off-surface variance per node, keeping the economy SVD batched
    on CPU during σ-field fitting.
    """
    if covs.ndim != 3 or covs.shape[-2:] != (R, R):
        raise ValueError(
            f"covariances must have shape (K, {R}, {R}), got {tuple(covs.shape)}"
        )
    if tangents.shape != (covs.shape[0], R, n):
        raise ValueError(
            f"tangents must have shape ({covs.shape[0]}, {R}, {n}), "
            f"got {tuple(tangents.shape)}"
        )
    # One economy SVD supplies both the numerical rank and the projector onto
    # the *actual* local tangent range.  An RBF surface may have a fold/pinch
    # where rank(T) < n; using nominal ``R-n`` there overstates tube variance.
    # Match ``torch.linalg.matrix_rank``'s default relative tolerance so this
    # fused path preserves the old rank boundary without a second decomposition.
    U, singular_values, _ = torch.linalg.svd(tangents, full_matrices=False)
    relative_tol = (
        singular_values[..., :1]
        * (max(R, n) * torch.finfo(singular_values.dtype).eps)
    )
    active = singular_values > relative_tol
    tangent_rank = active.sum(dim=-1)
    tangent_frame = U * active.unsqueeze(-2)
    proj = tangent_frame @ U.transpose(-1, -2)
    normal = torch.eye(R, dtype=covs.dtype, device=covs.device) - proj
    normal_dof = R - tangent_rank
    normal_var = torch.einsum("kij,kji->k", normal, covs) / normal_dof.clamp_min(1)
    # A full-row-rank tangent fills the fitted subspace: there is no normal
    # direction to average. Preserve the established isotropic fallback.
    isotropic = torch.diagonal(covs, dim1=-2, dim2=-1).sum(dim=-1) / max(R, 1)
    return torch.where(normal_dof > 0, normal_var, isotropic).clamp(min=0.0)


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
    :func:`compute_store_reduced_covariances` or
    :func:`compute_node_reduced_covariance_from_rows`) to one off-surface ``σ`` per node
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


def transfer_manifold_subspaces(
    src: Manifold,
    alignment: Mapping[int, "LayerAlignment"],
    *,
    whitener: "LayerWhitener | None",
    target_layer_means: Mapping[int, torch.Tensor],
    from_model: str,
    to_model: str,
) -> Manifold:
    """Map a fitted manifold's per-layer subspaces into a target model's space.

    The pure-tensor core of the cross-model Procrustes transfer (the folder
    read/write orchestration stays in :func:`saklas.io.manifold_lifecycle.
    transfer_manifold`).  Takes the already-loaded **source** ``Manifold``, a
    per-layer affine alignment map (the compact factorized
    :class:`saklas.io.alignment.LayerAlignment` returned by
    :func:`saklas.io.alignment.fit_alignment`), and the **target** model's
    whitener, and returns a new
    ``Manifold`` whose subspaces live in target space.

    Means and manifold points use the affine map (linear factor plus fitted
    translation); basis rows and direction profiles use only its linear factor.
    The mapped rows are QR-reparameterized to an orthonormal target frame and
    every affine/RBF reduced-coordinate coefficient is transformed by the exact
    companion matrix, preserving world points. A collapsed source span is
    rejected. Curved transfer requires that companion map to be an isometry:
    a non-isometric map turns the scalar tube thickness anisotropic, which the
    current manifold representation cannot encode. Layers the alignment doesn't
    cover drop.

    **Target-metric share re-bake (mandatory).**  The source fit's per-layer
    Mahalanobis ``share`` is a per-model quantity (``Σ`` belongs to
    ``from_model``), so it can't carry across.  The target ``whitener`` is
    **required** and must cover every transferred layer (all-or-nothing,
    mirroring the fit gate); the share is recomputed in target space via
    :func:`subspace_share` (``sqrt(Σ_k coordsᵀ (B_tgt Σ_tgt⁻¹ B_tgtᵀ) coords)``
    — the same formula the fit pipeline bakes).  A missing or non-covering
    whitener raises :class:`~saklas.core.mahalanobis.WhitenerError`; there is no
    Euclidean rebake. For curved layers, the target neutral mean is projected
    into the transferred frame and inverted on the transferred surface,
    yielding the target-model origin directly.

    Folder-level format guards (empty alignment, source fit missing) are the
    caller's concern; this function raises only :class:`~saklas.core.
    mahalanobis.WhitenerError` (missing / partial target whitener) and
    ``ValueError`` for unrepresentable geometry or when ``alignment`` covers
    none of the source's fitted layers.
    """
    from dataclasses import replace as _dc_replace

    from saklas.core.mahalanobis import WhitenerError

    src.validate_runtime_geometry()

    # Map each covered layer's subspace into target space.  ``M_L`` is
    # ``(D_tgt, D_src)`` so ``mean_tgt = M_L @ mean_src`` and each basis row
    # transforms the same way → ``basis_tgt = basis_src @ M_L^T``.
    new_layers: dict[int, LayerSubspace] = {}
    for layer, sub in src.layers.items():
        M_L = alignment.get(layer)
        if M_L is None:
            continue
        mean_f = sub.mean.to(torch.float32)
        basis_f = sub.basis.to(torch.float32)
        raw_basis = M_L.apply_vectors(basis_f)
        mean_tgt_f = M_L.apply_points(mean_f)

        rank = int(raw_basis.shape[0])
        if raw_basis.ndim != 2 or raw_basis.shape[1] < rank:
            raise ValueError(
                f"alignment for layer {layer} cannot carry rank-{rank} source "
                f"subspace into target shape {tuple(raw_basis.shape)}"
            )
        # Reparameterize B_raw = A @ B_ortho.  Runtime projection/injection
        # requires orthonormal basis rows, while multiplying every reduced output
        # by A preserves each mapped world-space manifold point exactly.
        raw_gram = raw_basis @ raw_basis.transpose(0, 1)
        identity = torch.eye(rank, dtype=raw_gram.dtype, device=raw_gram.device)
        if torch.allclose(raw_gram, identity, atol=1e-5, rtol=1e-5):
            # Preserve an already-orthonormal map byte-for-byte (identity and
            # square Procrustes rotations need no coordinate change).
            basis_tgt_f = raw_basis.contiguous()
            reduced_map = identity
        else:
            Q, R = torch.linalg.qr(raw_basis.transpose(0, 1), mode="reduced")
            # Unpivoted QR diagonals are not rank-revealing for oblique
            # dependencies. Rank-test only this small R×R factor with singular
            # values before publishing a supposedly orthonormal frame.
            singular_values = torch.linalg.svdvals(R)
            tol = (
                torch.finfo(R.dtype).eps * max(raw_basis.shape)
                * float(singular_values.max())
            )
            if (
                singular_values.numel() != rank
                or bool((singular_values <= tol).any())
            ):
                raise ValueError(
                    f"alignment for layer {layer} collapses the rank-{rank} manifold "
                    "subspace; refusing a rank-deficient transfer"
                )
            basis_tgt_f = Q.transpose(0, 1).contiguous()
            reduced_map = R.transpose(0, 1).contiguous()
        kwargs: dict[str, Any] = {
            "mean": mean_tgt_f.to(dtype=sub.mean.dtype),
            "basis": basis_tgt_f.to(dtype=sub.basis.dtype),
        }
        if sub.is_affine:
            source_map = (
                torch.eye(
                    sub.rank, dtype=torch.float32, device=reduced_map.device,
                )
                if sub.affine_map is None
                else sub.affine_map.to(
                    device=reduced_map.device, dtype=torch.float32,
                )
            )
            kwargs["affine_map"] = (source_map @ reduced_map).to(
                dtype=sub.basis.dtype,
            )
            kwargs["node_coords"] = (
                None if sub.node_coords is None
                else (sub.node_coords.to(torch.float32) @ reduced_map).to(
                    dtype=sub.node_coords.dtype,
                )
            )
        else:
            assert sub.rbf_weights is not None and sub.poly_coeffs is not None
            kwargs["rbf_weights"] = (
                sub.rbf_weights.to(torch.float32) @ reduced_map
            ).to(dtype=sub.rbf_weights.dtype)
            kwargs["poly_coeffs"] = (
                sub.poly_coeffs.to(torch.float32) @ reduced_map
            ).to(dtype=sub.poly_coeffs.dtype)
            # A scalar isotropic tube thickness is invariant only under an
            # isometry of this reduced frame. The current representation has no
            # anisotropic tube field, so reject an unrepresentable transfer.
            gram = reduced_map @ reduced_map.transpose(0, 1)
            isometric = torch.allclose(
                gram, torch.eye(rank, dtype=gram.dtype, device=gram.device),
                atol=1e-4, rtol=1e-4,
            )
            if not isometric:
                raise ValueError(
                    f"alignment for layer {layer} is non-isometric in curved "
                    "manifold coordinates; anisotropic tube transfer is not "
                    "representable by the current scalar sigma field"
                )
        new_layers[layer] = _dc_replace(sub, **kwargs)

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
    curved_layers = {
        layer for layer, subspace in new_layers.items() if not subspace.is_affine
    }
    missing_means = sorted(curved_layers - set(target_layer_means))
    if missing_means:
        raise WhitenerError(
            "manifold transfer requires target neutral means covering every "
            f"transferred layer; missing {missing_means}"
        )

    new_share: dict[int, float] = {}
    for layer, sub_tgt in new_layers.items():
        sub_f = sub_tgt.to(device=torch.device("cpu"), dtype=torch.float32)
        # ``coords`` are the reduced node values in subspace-coordinate space.
        # For a K=1 affine folded ray there is no node cloud to μ-center: its
        # share is the target-metric norm of the actual neutral→pole world
        # direction. For K>=2 affine and every curved fit, ``subspace_share``
        # computes the μ-centered whitened spread
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
            if coords.shape[0] == 1:
                world_direction = coords[0] @ sub_f.basis
                new_share[layer] = float(
                    whitener.mahalanobis_norm(layer, world_direction)
                )
                continue
        else:
            _np, _rw, _pc = sub_f.rbf_params()
            coords = eval_rbf(_np, _rw, _pc, _np)  # (K, R)
        mu_coords = coords - coords.mean(dim=0, keepdim=True)  # μ-center
        new_share[layer] = subspace_share(
            mu_coords, sub_f.basis, whitener=whitener, layer=layer,
        )

    new_origin: dict[int, torch.Tensor] = {}
    for layer, sub_tgt in new_layers.items():
        if sub_tgt.is_affine:
            continue
        sub_f = sub_tgt.to(device=torch.device("cpu"), dtype=torch.float32)
        target_mean = target_layer_means[layer].to(
            device="cpu", dtype=torch.float32,
        ).reshape(-1)
        if target_mean.shape != sub_f.mean.shape:
            raise ValueError(
                f"target neutral mean for layer {layer} has shape "
                f"{tuple(target_mean.shape)}, expected {tuple(sub_f.mean.shape)}"
            )
        query = (target_mean - sub_f.mean) @ sub_f.basis.T
        origin, _distance = invert_parameterization(
            sub_f, src.domain, query, src.node_coords.to(torch.float32),
        )
        new_origin[layer] = origin.reshape(-1).to(torch.float32)

    transferred = _dc_replace(
        src, layers=new_layers,
        mahalanobis_share=new_share,
        origin=new_origin,
    )
    transferred.validate_runtime_geometry()
    return transferred


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
    manifold.validate_runtime_geometry()
    return next(iter(manifold.layers.values())).is_affine


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
    "compute_manifold_node_stats",
    "compute_node_reduced_covariance_from_rows",
    "compute_store_reduced_covariances",
    "invert_parameterization",
    "manifold_is_affine",
]
