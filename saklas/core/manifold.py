"""Spline-based manifold steering primitives.

Implements the activation-manifold half of Goodfire's "Manifold Steering"
(arXiv 2605.05115) for saklas: instead of a single linear steering
direction, fit a cubic spline through per-concept activation *centroids*
in a low-dimensional PCA subspace, then steer by moving the running
activation's in-subspace component onto a chosen point of that curve.

A linear A->B steering vector cuts a straight chord through activation
space; that chord passes through low-density off-manifold regions, which
shows up behaviorally as "teleportation" (unnatural intermediate states)
and diversity collapse. A spline through the ordered centroids stays on
the learned manifold.

Scope note: the paper also fits 2-D manifolds with thin-plate splines.
saklas manifolds are an ordered (or cyclic) sequence of labeled nodes, so
the manifold is intrinsically a 1-parameter path -- parameter ``t`` in
``[0, 1]`` -- regardless of the embedding dimension. Only 1-D-domain
splines are fitted here; thin-plate splines are deliberately out of scope.
A future 2-D extension would add them rather than reshape this module.

This module is pure tensor math (fp32, no session/IO coupling), mirroring
how :mod:`saklas.core.vectors` holds the low-level extraction primitives.
The spline solves use dense ``torch.linalg.solve`` -- knot counts are tiny
(a manifold has on the order of ten to thirty nodes) and fitting is a
one-shot operation, not a hot path. ``eval_cubic`` and
:func:`subspace_replace` are the only functions reachable from the
generation hot path; both are allocation-light and free of host syncs.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

log = logging.getLogger(__name__)


# Default PCA width for a fitted manifold subspace.  Matches the paper's
# 64-dimensional reduction.  Clamped down to ``min(64, K-1, rank)`` when
# the node count ``K`` is small -- ``K`` centered centroids span a space
# of rank at most ``K-1``.
DEFAULT_N_COMPONENTS = 64

# Grid resolution for the inverse parameterization (nearest-point
# projection onto the curve).  Used only by the naturalness eval, never
# by the steering hot path -- the hook steers to a fixed ``t``.
DEFAULT_INVERSION_RESOLUTION = 512


# --------------------------------------------------------------- splines ---

def solve_natural_cubic(t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Second-derivative coefficients ``M`` for a natural cubic spline.

    ``t`` is ``(K,)`` strictly increasing knot parameters; ``y`` is
    ``(K, D)`` knot values.  Returns ``M`` of shape ``(K, D)`` with the
    natural boundary condition ``M[0] == M[K-1] == 0``.

    The interior second derivatives solve the standard tridiagonal
    system (e.g. Press et al., *Numerical Recipes*, natural cubic
    spline).  Solved densely -- ``K`` is tiny.
    """
    t = t.to(torch.float32)
    y = y.to(torch.float32)
    K = t.shape[0]
    if K < 2:
        raise ValueError(f"natural cubic spline needs >= 2 knots, got {K}")
    M = torch.zeros_like(y)
    if K == 2:
        # One interval: M stays zero -> the spline is the straight chord.
        return M
    h = t[1:] - t[:-1]  # (K-1,)
    if torch.any(h <= 0):
        raise ValueError("spline knot parameters must be strictly increasing")
    n = K - 2  # number of interior knots / unknowns
    A = torch.zeros((n, n), dtype=torch.float32)
    for i in range(n):
        A[i, i] = 2.0 * (h[i] + h[i + 1])
        if i > 0:
            A[i, i - 1] = h[i]
        if i < n - 1:
            A[i, i + 1] = h[i + 1]
    # RHS rows track interior knots 1..K-2.
    rhs = 6.0 * (
        (y[2:] - y[1:-1]) / h[1:].unsqueeze(-1)
        - (y[1:-1] - y[:-2]) / h[:-1].unsqueeze(-1)
    )  # (n, D)
    M[1:-1] = torch.linalg.solve(A, rhs)
    return M


def solve_periodic_cubic(t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Second-derivative coefficients ``M`` for a periodic cubic spline.

    ``t`` is ``(K+1,)`` knot parameters where the final knot wraps the
    curve closed; ``y`` is ``(K+1, D)`` with ``y[K] == y[0]``.  Returns
    ``M`` of shape ``(K+1, D)`` with ``M[K] == M[0]`` -- the closed
    curve has matching value, first and second derivative at the seam.

    The system over the ``K`` distinct knots is cyclic tridiagonal;
    solved densely (``K`` is tiny).
    """
    t = t.to(torch.float32)
    y = y.to(torch.float32)
    P = t.shape[0]
    K = P - 1  # distinct knots; knot K wraps to knot 0
    if K < 3:
        raise ValueError(
            f"periodic cubic spline needs >= 3 distinct knots, got {K}"
        )
    h = t[1:] - t[:-1]  # (K,) interval lengths, including the wrap segment
    if torch.any(h <= 0):
        raise ValueError("spline knot parameters must be strictly increasing")
    D = y.shape[1]
    A = torch.zeros((K, K), dtype=torch.float32)
    rhs = torch.zeros((K, D), dtype=torch.float32)
    for i in range(K):
        hm = h[(i - 1) % K]  # length of the interval before knot i
        hi = h[i]            # length of the interval after knot i
        A[i, i] = 2.0 * (hm + hi)
        A[i, (i - 1) % K] += hm
        A[i, (i + 1) % K] += hi
        # y indices over the distinct knots, wrapping through knot 0.
        y_prev = y[(i - 1) % K]
        y_here = y[i]
        y_next = y[(i + 1) % K]
        rhs[i] = 6.0 * ((y_next - y_here) / hi - (y_here - y_prev) / hm)
    M_distinct = torch.linalg.solve(A, rhs)  # (K, D)
    M = torch.empty_like(y)
    M[:K] = M_distinct
    M[K] = M_distinct[0]  # close the seam
    return M


def eval_cubic(
    t_knots: torch.Tensor,
    y: torch.Tensor,
    M: torch.Tensor,
    s: torch.Tensor,
) -> torch.Tensor:
    """Evaluate a cubic spline at parameters ``s``.

    ``t_knots`` is ``(P,)``; ``y`` and ``M`` are ``(P, D)`` (knot values
    and second derivatives, as produced by :func:`solve_natural_cubic` /
    :func:`solve_periodic_cubic`).  ``s`` is any shape ``(..,)``; the
    return is ``(.., D)``.

    Hot-path safe: vectorized via ``searchsorted``, no ``.item()`` and no
    host sync.  ``s`` is clamped into ``[t_knots[0], t_knots[-1]]`` so
    out-of-range positions saturate at the curve endpoints.
    """
    P = t_knots.shape[0]
    t_knots = t_knots.to(s.dtype)
    y = y.to(s.dtype)
    M = M.to(s.dtype)
    s_clamped = s.clamp(min=t_knots[0], max=t_knots[-1])
    # Interval index in [0, P-2]: searchsorted gives the insertion point.
    idx = torch.searchsorted(t_knots, s_clamped, right=True) - 1
    idx = idx.clamp(min=0, max=P - 2)
    idx1 = idx + 1
    t0 = t_knots[idx]
    t1 = t_knots[idx1]
    h = (t1 - t0).clamp(min=1e-12)
    a = ((t1 - s_clamped) / h).unsqueeze(-1)
    b = ((s_clamped - t0) / h).unsqueeze(-1)
    h2 = (h * h).unsqueeze(-1)
    y0 = y[idx]
    y1 = y[idx1]
    M0 = M[idx]
    M1 = M[idx1]
    return (
        a * y0
        + b * y1
        + ((a.pow(3) - a) * M0 + (b.pow(3) - b) * M1) * h2 / 6.0
    )


def invert_parameterization(
    t_knots: torch.Tensor,
    y: torch.Tensor,
    M: torch.Tensor,
    query: torch.Tensor,
    *,
    resolution: int = DEFAULT_INVERSION_RESOLUTION,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Nearest-point projection of ``query`` onto the spline curve.

    Returns ``(s, dist)``: ``s`` is the curve parameter minimizing the
    Euclidean distance from the curve to each query, ``dist`` is that
    distance.  ``query`` is ``(.., D)``; both returns are ``(..,)``.

    A dense grid scan (``resolution`` samples) followed by a local
    three-point parabola refinement for sub-grid precision.  This is the
    paper's ``s^-1`` map.  It is used only by the naturalness eval, never
    by the steering hot path -- the hook steers to a fixed ``t`` -- so the
    occasional wrong-basin landing on a near-self-intersecting curve only
    adds noise to the metric.
    """
    s_grid = torch.linspace(
        float(t_knots[0]), float(t_knots[-1]), resolution, dtype=query.dtype,
    )
    curve = eval_cubic(t_knots, y, M, s_grid)  # (resolution, D)
    flat = query.reshape(-1, query.shape[-1])  # (N, D)
    # Pairwise squared distance: (N, resolution).
    d2 = torch.cdist(flat, curve).pow(2)
    best = d2.argmin(dim=-1)  # (N,)
    # Three-point parabola refinement around the grid minimum.
    lo = (best - 1).clamp(min=0, max=resolution - 1)
    hi = (best + 1).clamp(min=0, max=resolution - 1)
    f_lo = d2.gather(1, lo.unsqueeze(1)).squeeze(1)
    f_mid = d2.gather(1, best.unsqueeze(1)).squeeze(1)
    f_hi = d2.gather(1, hi.unsqueeze(1)).squeeze(1)
    denom = f_lo - 2.0 * f_mid + f_hi
    # Parabola vertex offset in grid units, in (-0.5, 0.5).  Only a
    # convex triple (denom > 0) brackets a real minimum; collinear or
    # concave triples (denom <= 0) fall back to the grid point itself.
    convex = denom > 1e-12
    denom_safe = torch.where(convex, denom, torch.ones_like(denom))
    offset = torch.where(
        convex,
        0.5 * (f_lo - f_hi) / denom_safe,
        torch.zeros_like(denom),
    )
    offset = offset.clamp(min=-0.5, max=0.5)
    step = (s_grid[-1] - s_grid[0]) / (resolution - 1)
    s = s_grid[best] + offset * step
    s = s.clamp(min=float(t_knots[0]), max=float(t_knots[-1]))
    refined = eval_cubic(t_knots, y, M, s)  # (N, D)
    dist = torch.linalg.vector_norm(flat - refined, dim=-1)
    return s.reshape(query.shape[:-1]), dist.reshape(query.shape[:-1])


# ------------------------------------------------------------- subspaces ---

@dataclass
class LayerSubspace:
    """Per-layer reduced-coordinate frame and spline for one manifold.

    ``mean`` and ``basis`` define an affine PCA subspace; ``t_knots``,
    ``coords`` and ``spline_M`` define the cubic spline through the node
    centroids expressed in that subspace's reduced coordinates.

    For a cyclic manifold the knot arrays carry ``K+1`` rows -- the final
    row wraps back to the first -- so :func:`eval_cubic` treats natural
    and periodic splines uniformly.
    """

    mean: torch.Tensor      # (D,) centering mean over the node centroids
    basis: torch.Tensor     # (R, D) orthonormal PCA rows, R = min(64, K-1, rank)
    t_knots: torch.Tensor   # (P,) chord-length knot parameters in [0, 1]
    coords: torch.Tensor    # (P, R) centroid coordinates in PCA space
    spline_M: torch.Tensor  # (P, R) spline second-derivative coefficients

    @property
    def rank(self) -> int:
        return int(self.basis.shape[0])

    def to(self, *, device: torch.device, dtype: torch.dtype) -> "LayerSubspace":
        """Return a copy with every tensor on ``device`` in ``dtype``."""
        return LayerSubspace(
            mean=self.mean.to(device=device, dtype=dtype),
            basis=self.basis.to(device=device, dtype=dtype),
            t_knots=self.t_knots.to(device=device, dtype=dtype),
            coords=self.coords.to(device=device, dtype=dtype),
            spline_M=self.spline_M.to(device=device, dtype=dtype),
        )

    def spline_point(self, s: torch.Tensor | float) -> torch.Tensor:
        """World-space activation ``(D,)`` at curve parameter ``s``.

        Evaluates the spline in reduced coordinates, then lifts back to
        the ambient activation space via ``coords @ basis + mean``.
        """
        if not isinstance(s, torch.Tensor):
            # Materialize on the subspace's device — a CPU scalar would
            # mix with device-resident knots inside ``eval_cubic`` once
            # the manifold has been promoted to a CUDA/MPS session.
            s = torch.tensor(
                float(s), dtype=self.coords.dtype, device=self.coords.device,
            )
        reduced = eval_cubic(self.t_knots, self.coords, self.spline_M, s)
        return reduced @ self.basis + self.mean


def fit_layer_subspace(
    centroids: torch.Tensor,
    *,
    cyclic: bool,
    n_components: int = DEFAULT_N_COMPONENTS,
) -> LayerSubspace:
    """Fit a PCA subspace + cubic spline through one layer's centroids.

    ``centroids`` is ``(K, D)`` -- one per-node mean activation, in node
    order.  The data is centered, reduced to ``R = min(n_components,
    K-1, rank)`` principal components, and a cubic spline is fitted
    through the reduced coordinates with a chord-length parameterization.

    ``cyclic=True`` fits a periodic spline (the last node connects back
    to the first); ``cyclic=False`` fits a natural cubic spline.
    """
    centroids = centroids.to(torch.float32)
    K = centroids.shape[0]
    if K < 3:
        raise ValueError(
            f"a manifold needs >= 3 nodes to fit a curve, got {K}"
        )
    mean = centroids.mean(dim=0)
    X = centroids - mean  # (K, D)
    # Economy SVD: Vh rows are principal directions, S the singular values.
    _, S, Vh = torch.linalg.svd(X, full_matrices=False)
    rank = int((S > 1e-6 * S[0].clamp(min=1e-12)).sum().item())
    R = max(1, min(n_components, K - 1, rank))
    basis = Vh[:R].contiguous()  # (R, D)
    coords = X @ basis.T          # (K, R)

    if cyclic:
        # Append the wrap segment: the curve returns to node 0.
        coords_eval = torch.cat([coords, coords[:1]], dim=0)  # (K+1, R)
        seg = torch.linalg.vector_norm(
            coords_eval[1:] - coords_eval[:-1], dim=-1,
        )  # (K,)
        t_knots = _chord_param(seg)
        spline_M = solve_periodic_cubic(t_knots, coords_eval)
        coords_out = coords_eval
    else:
        seg = torch.linalg.vector_norm(
            coords[1:] - coords[:-1], dim=-1,
        )  # (K-1,)
        t_knots = _chord_param(seg)
        spline_M = solve_natural_cubic(t_knots, coords)
        coords_out = coords

    return LayerSubspace(
        mean=mean, basis=basis, t_knots=t_knots,
        coords=coords_out, spline_M=spline_M,
    )


def _chord_param(segments: torch.Tensor) -> torch.Tensor:
    """Cumulative chord-length knot parameters normalized to ``[0, 1]``.

    ``segments`` holds the consecutive inter-knot distances.  Degenerate
    (all-zero) inputs fall back to a uniform parameterization so the
    spline solve still receives strictly increasing knots.
    """
    total = float(segments.sum())
    if total <= 1e-12:
        n = segments.shape[0] + 1
        return torch.linspace(0.0, 1.0, n, dtype=torch.float32)
    # Guard individual zero-length segments (duplicate centroids) with a
    # tiny floor so knots stay strictly increasing.
    seg = segments.clamp(min=1e-9)
    cumulative = torch.cat([
        torch.zeros(1, dtype=torch.float32),
        torch.cumsum(seg, dim=0),
    ])
    return cumulative / cumulative[-1]


# -------------------------------------------------------------- manifold ---

@dataclass
class Manifold:
    """A fitted manifold: per-layer subspaces + splines, plus identity.

    The in-memory analogue of :class:`saklas.core.profile.Profile` for
    manifold steering.  Built by the manifold extraction pipeline and
    consumed by the session and the steering hooks.

    ``feature_space`` is ``"raw"`` for a manifold fitted directly on
    residual-stream activations, or ``"sae-<release>"`` when fitted in an
    SAE feature space (in which case the stored :class:`LayerSubspace`
    values are already decoded back to model space for runtime use).
    """

    name: str
    cyclic: bool
    node_labels: list[str]
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
            cyclic=self.cyclic,
            node_labels=list(self.node_labels),
            layers={
                idx: sub.to(device=device, dtype=dtype)
                for idx, sub in self.layers.items()
            },
            feature_space=self.feature_space,
            metadata=dict(self.metadata),
        )

    def spline_point(self, layer: int, s: torch.Tensor | float) -> torch.Tensor:
        """World-space activation at curve parameter ``s`` for one layer."""
        return self.layers[layer].spline_point(s)


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
    ``target`` ``(D,)`` is the spline point to steer toward; ``alpha`` is
    the blend fraction.

    The activation decomposes as ``h = h_par + h_perp`` where ``h_par``
    is the reconstruction inside the subspace and ``h_perp`` the
    orthogonal residual.  Only ``h_par`` is moved -- toward ``target`` --
    and the residual is kept verbatim::

        h' = h + alpha * (target - h_par)

    so ``alpha=0`` is the identity and ``alpha=1`` snaps the in-subspace
    component fully onto the spline.  The result is rescaled to ``h``'s
    original per-position norm, matching every other saklas injection
    hook.  This generalizes the ``~`` / ``|`` line-projection operators
    from a 1-D direction to an R-dimensional curve.

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
    ``layer_<L>.basis``, ``layer_<L>.t_knots``, ``layer_<L>.coords``,
    ``layer_<L>.spline_M``.  The sidecar carries the manifold identity
    (name, cyclic flag, ordered node labels, feature space) plus the
    provenance fields in ``metadata`` (``method``, ``nodes_sha256``, and
    optional SAE keys).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tensors: dict[str, torch.Tensor] = {}
    for idx, sub in manifold.layers.items():
        tensors[f"layer_{idx}.mean"] = sub.mean.contiguous().cpu()
        tensors[f"layer_{idx}.basis"] = sub.basis.contiguous().cpu()
        tensors[f"layer_{idx}.t_knots"] = sub.t_knots.contiguous().cpu()
        tensors[f"layer_{idx}.coords"] = sub.coords.contiguous().cpu()
        tensors[f"layer_{idx}.spline_M"] = sub.spline_M.contiguous().cpu()
    save_file(tensors, str(path))

    from saklas import __version__ as _saklas_version
    from saklas.io.packs import PACK_FORMAT_VERSION

    sidecar: dict[str, object] = {
        "format_version": PACK_FORMAT_VERSION,
        "method": metadata.get("method", "manifold_pca"),
        "saklas_version": _saklas_version,
        "name": manifold.name,
        "cyclic": manifold.cyclic,
        "node_labels": list(manifold.node_labels),
        "node_count": len(manifold.node_labels),
        "feature_space": manifold.feature_space,
    }
    for key in (
        "nodes_sha256", "sae_release", "sae_revision", "sae_ids_by_layer",
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

    by_layer: dict[int, dict[str, torch.Tensor]] = {}
    for key, tensor in tensors.items():
        # key shape: "layer_<idx>.<field>"
        head, field_name = key.rsplit(".", 1)
        idx = int(head.split("_", 1)[1])
        by_layer.setdefault(idx, {})[field_name] = tensor

    layers: dict[int, LayerSubspace] = {}
    for idx, parts in by_layer.items():
        layers[idx] = LayerSubspace(
            mean=parts["mean"],
            basis=parts["basis"],
            t_knots=parts["t_knots"],
            coords=parts["coords"],
            spline_M=parts["spline_M"],
        )

    return Manifold(
        name=sidecar.get("name", path.parent.name),
        cyclic=bool(sidecar.get("cyclic", False)),
        node_labels=list(sidecar.get("node_labels", [])),
        layers=layers,
        feature_space=sidecar.get("feature_space", "raw"),
        metadata=sidecar,
    )


# ------------------------------------------------ behavior-space manifold ---
#
# The naturalness eval (Goodfire's paper, second half): fit a manifold
# over the model's *output* distributions, then measure how far a steered
# generation's behavioral trajectory strays off it.  Output distributions
# are mapped to Hellinger space via ``p -> sqrt(p)`` -- there the ordinary
# Euclidean distance is the Hellinger distance, which linearizes the
# probability simplex so the same cubic-spline machinery applies.

def to_hellinger(p: torch.Tensor) -> torch.Tensor:
    """Map a probability distribution to Hellinger space: ``p -> sqrt(p)``.

    In Hellinger space the L2 distance between two mapped distributions
    is their Hellinger distance, and the inner product is the
    Bhattacharyya coefficient -- which linearizes the simplex enough for
    a Euclidean PCA + spline fit.
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
    *,
    cyclic: bool,
    n_components: int = DEFAULT_N_COMPONENTS,
) -> LayerSubspace:
    """Fit a spline through per-node output-distribution centroids.

    ``centroid_dists`` is ``(K, V)`` -- one mean next-token distribution
    per manifold node, in node order.  The distributions are mapped to
    Hellinger space (``sqrt``) and the same :func:`fit_layer_subspace`
    PCA + cubic-spline fit is applied there.  Returns the fitted
    :class:`LayerSubspace` (the behavior manifold).
    """
    return fit_layer_subspace(
        to_hellinger(centroid_dists), cyclic=cyclic, n_components=n_components,
    )


def trajectory_naturalness(
    traj_dists: torch.Tensor,
    behavior: LayerSubspace,
) -> torch.Tensor:
    """Per-step Bhattacharyya distance from a trajectory to a behavior manifold.

    ``traj_dists`` is ``(T, V)`` -- the sequence of next-token
    distributions a generation produced.  ``behavior`` is a behavior
    manifold from :func:`fit_behavior_manifold`.  For each step the
    nearest point on the behavior spline is found (in Hellinger space)
    and the Bhattacharyya distance to it is returned -- low means the
    step sits on the natural behavior manifold, high flags an
    off-manifold "teleportation" artifact.

    Returns a ``(T,)`` tensor of per-step distances.
    """
    h = to_hellinger(traj_dists)  # (T, V)
    coords = (h - behavior.mean) @ behavior.basis.T  # (T, R)
    s, _dist = invert_parameterization(
        behavior.t_knots, behavior.coords, behavior.spline_M, coords,
    )
    curve_coords = eval_cubic(
        behavior.t_knots, behavior.coords, behavior.spline_M, s,
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
    "LayerSubspace",
    "Manifold",
    "solve_natural_cubic",
    "solve_periodic_cubic",
    "eval_cubic",
    "invert_parameterization",
    "fit_layer_subspace",
    "subspace_replace",
    "compute_node_centroid",
    "save_manifold",
    "load_manifold",
    "to_hellinger",
    "bhattacharyya_distance",
    "fit_behavior_manifold",
    "trajectory_naturalness",
    "compute_node_behavior_centroid",
    "compute_trajectory_distributions",
]
