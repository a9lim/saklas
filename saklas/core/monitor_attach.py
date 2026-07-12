"""Attach-time probe algebra for :class:`~saklas.core.monitor.Monitor`.

This module holds the once-per-``add_probe`` setup work that is entirely off
the per-token hot path: the :class:`AttachedManifoldProbe` and
:class:`_LayerWhiten` dataclasses, the Mahalanobis factor builder
(:func:`_build_whitened_factors`), the node-cache + share-weight builder
(:func:`_attach_manifold_probe`), the soft-assignment bandwidth computation
(:func:`_compute_assign_bandwidth`), and the per-layer geometry primitive
(:func:`_layer_geometry`).

The *only* item here that the hot path calls directly is
:func:`_layer_geometry` — a module-level function call with no class
attribute resolution overhead.  :mod:`saklas.core.monitor` imports it (and the
supporting constants / dataclasses) back from here; this module must **not**
import from :mod:`saklas.core.monitor` (no circular dependency).
"""

from __future__ import annotations

import math

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from saklas.core.mahalanobis import WhitenerError

if TYPE_CHECKING:
    from saklas.core.manifold import Manifold

# ---------------------------------------------------------------------------
# Module-level constants re-exported to monitor.py
# ---------------------------------------------------------------------------

# Default top-N nearest-node count for manifold probes.  Per-probe
# override available on ``Monitor.add_probe``.
DEFAULT_NEAREST_TOP_N: int = 3

# Synthetic label for the neutral anchor in the nearest-node readout.  Every
# fit is neutral-anchored (the per-model neutral mean is the frame origin), so
# neutral is a *point* in the same whitened metric the nodes live in — not a
# stored corpus node.  It competes in the ``nearest`` ranking as a virtual
# candidate (computed, never written to ``node_labels`` / ``node_coords``): when
# the running activation sits closer to the origin than to any node, ``nearest``
# reports ``("neutral", dist)`` and ``flat_scalars`` exposes the uniform
# ``<probe>@neutral`` gate channel.  Suppressed when a manifold already carries a
# real node with this label (the corpus node owns the name).
NEUTRAL_LABEL: str = "neutral"

# Floor for the per-layer share weight on a manifold layer with a
# degenerate (near-zero) Mahalanobis share — keeps the share-weighted
# cross-layer aggregation from collapsing to NaN on a manifold whose
# every layer reports share ≈ 0.  Negligible against a real per-layer
# share (~O(1) whitened units), so it doesn't de-peak the weighting.
_MIN_SHARE_WEIGHT: float = 1e-6

# Guard against division by zero in the subspace_fraction / cosine
# denominator (a zero or near-zero activation norm).
_FRACTION_EPSILON: float = 1e-8


# ---------------------------------------------------------------------------
# Woodbury helper — shared by attach-time factor build and the per-token
# hot path in monitor.py (imported back there).
# ---------------------------------------------------------------------------

def _woodbury_apply(
    v: torch.Tensor, X: torch.Tensor, K: torch.Tensor, lam: float,
) -> torch.Tensor:
    """On-device ``Σ_reg⁻¹ v = (1/λ)(v − Xᵀ K (X v))`` (Woodbury).

    Shared by the per-probe and batched flat scoring paths.  Kept a
    plain module-level function (not a method) so the per-token hot path is
    a global lookup with no class attribute resolution — the hot-path
    companion to :meth:`LayerWhitener.apply_inv` (which force-promotes
    fp32/CPU and is wrong here).  ``v`` is ``[D]`` or ``[n, D]``; ``X`` is
    ``[N, D]``, ``K`` is ``[N, N]``, all on the same device/dtype.  Pure
    matmuls, no host sync.  For a ``[n, D]`` batch this is
    ``(1/λ)(V − (V Xᵀ) K X)``; for ``[D]`` it broadcasts the same way.
    """
    Xv = v @ X.transpose(0, 1)        # [..., N]
    KXv = Xv @ K.transpose(0, 1)      # [..., N]  (K symmetric; t() = K)
    return (v - KXv @ X) / lam        # [..., D]


# ---------------------------------------------------------------------------
# Attach-time algebra — runs once per add_probe, off the hot path
# ---------------------------------------------------------------------------

@dataclass
class AttachedManifoldProbe:
    """One manifold registered on a :class:`~saklas.core.monitor.Monitor`.

    Pairs the loaded :class:`~saklas.core.manifold.Manifold` artifact with
    the per-layer cache the monitor uses on the hot path:
    ``node_values_reduced`` is the per-layer ``(K, R)`` tensor of node
    activations in subspace coords — ``sub.node_coords`` for a flat fit (the
    M-projected node coords directly, pruned-basis-consistent), else
    ``(sub.eval_at(domain.embed(node_coords)) - sub.mean) @ sub.basis.T`` for
    a curved fit (RBF surface eval) — pre-computed once at attach so per-token
    distance computations are one batched cdist in R-dim per layer.
    ``share_weights`` is the per-layer Mahalanobis share (the steering budget)
    used to weight cross-layer aggregation — the layer that carries the most
    push is the most reliable to read from; floored at
    :data:`_MIN_SHARE_WEIGHT` so a degenerate layer doesn't crash the
    aggregator.
    """

    name: str
    manifold: "Manifold"
    # Candidate order and exact lookup are attachment invariants, built once
    # from corpus nodes plus the optional synthetic neutral anchor.
    candidate_labels: tuple[str, ...]
    label_to_candidate_idx: dict[str, int]
    top_n: int = DEFAULT_NEAREST_TOP_N
    is_affine: bool = True
    # Whether the neutral anchor competes as a virtual candidate in this
    # probe's ``nearest`` ranking (see :data:`NEUTRAL_LABEL`).  Set at attach to
    # ``NEUTRAL_LABEL not in manifold.node_labels`` so a real node named
    # ``neutral`` keeps sole ownership of the label.
    inject_neutral: bool = True
    # Candidate order for exact ``@label`` / ``~label`` gate reads: corpus nodes
    # followed by the synthetic neutral anchor when injected.  The mapping is
    # built once at attach (duplicate corpus labels resolve to their last
    # occurrence, matching the historic per-token dict comprehension).
    # Per-layer cache, indexed by layer index — same set of layers as
    # ``manifold.layers``.
    node_values_reduced: dict[int, torch.Tensor] = field(default_factory=dict)
    # Per-layer Mahalanobis-share weight, normalized to sum to 1 across
    # attached layers.
    share_weights: dict[int, float] = field(default_factory=dict)
    # Per-candidate soft-assignment bandwidth ``τ`` ``(Kc,)`` in the **whitened**
    # metric (the metric the nearest-node distances live in), share-weighted across
    # layers, candidate order = nodes then the neutral anchor (when injected).
    # A curved fit's within-node σ-field mapped into the whitened metric (×
    # ``√(tr(M_R)/R)``, the isotropic-σ scale); a flat fit's local layout scale
    # (each node's nearest-neighbor whitened distance).  Drives the
    # ``softmax(−d²/2τ² − R·log(τ))`` soft assignment.  ``None`` until the
    # post-attach bandwidth pass runs (an empty / degenerate manifold leaves it
    # ``None`` ⇒ assignment empty, argmax ``nearest`` unaffected).
    assign_bandwidth: torch.Tensor | None = None
    # Per-candidate Gaussian log-volume bias ``−R·log(τ_k)`` ``(Kc,)`` —
    # precomputed at attach so the hot-path logit is
    # ``−d²/(2τ²) + logvol_bias`` (one add, no per-token ``log``).  This is the
    # missing Gaussian normalization that turns the bare ``softmax(−d²/2τ²)``
    # into a proper isotropic R-D mixture posterior with a uniform node prior:
    # the bias penalizes diffuse-``τ`` candidates by their log-volume so a wide
    # node can't swallow probability from far away (the "broadest-node-wins"
    # pathology the bare form has).  ``R`` is the manifold's per-layer subspace
    # rank — the effective dimension of the space ``τ`` measures.
    assign_logvol_bias: torch.Tensor | None = None
    # Robust per-probe scale for the ``nearest`` / ``@label`` distance: the
    # **median node nearest-neighbor whitened spacing** (a single scalar, not the
    # per-candidate ``τ`` above).  The reported nearest distance is divided by it
    # so ``@label`` reads "how many typical label-spacings away" — portable across
    # probes (raw whitened distance spans ~60× by fit), while staying a *uniform*
    # rescale that preserves the raw nearest **ranking** (``nearest`` stays
    # literally nearest, distinct from the density-aware ``assignment``).  A
    # single-node fit / degenerate manifold leaves it ``1.0`` (raw distance).
    label_scale: float = 1.0
    # Per-layer Mahalanobis bundle, populated at attach.  The wired whitener
    # must cover every layer of this manifold (all-or-nothing per probe), else
    # ``_build_whitened`` raises ``WhitenerError`` — there is no Euclidean
    # readout.  Each entry is ``_LayerWhiten`` — the precomputed factors that
    # turn the per-token fraction + nearest-node distance into their whitened
    # forms (M-orthogonal subspace projection + Mahalanobis distance).  Empty
    # only for an empty manifold (no layers to read).
    whitened: dict[int, "_LayerWhiten"] = field(default_factory=dict)


@dataclass
class _LayerWhiten:
    """Precomputed per-layer Mahalanobis factors for a manifold probe.

    Built at attach from the wired :class:`~saklas.core.mahalanobis.LayerWhitener`
    + the layer's subspace.  ``m_r_inv`` and ``chol`` are the ``(R, R)``
    inverse and lower-Cholesky factor of the subspace-restricted inverse
    covariance ``M_R = B Σ⁻¹ Bᵀ``
    (:meth:`~saklas.core.mahalanobis.LayerWhitener.subspace_gram`);
    ``node_white`` is the ``(K, R)`` node coords transformed into the whitened
    metric (``v_reduced @ chol``) so a plain cdist against the transformed
    query yields the true Mahalanobis distance restricted to the subspace.
    ``(X, K_inv, lam)`` are the on-device Woodbury factors for the per-token
    ``Σ⁻¹ x`` apply.  All tensors live on the manifold's fit device.
    """

    m_r_inv: torch.Tensor      # (R, R) = (B Σ⁻¹ Bᵀ)⁻¹
    chol: torch.Tensor         # (R, R) lower-tri, M_R = chol @ cholᵀ
    node_white: torch.Tensor   # (K, R) = node_values_reduced @ chol
    # Neutral anchor in the same whitened metric as ``node_white`` (R,).  For a
    # neutral-anchored affine fit the frame origin *is* neutral, so this is the
    # zero vector; a curved fit centers on the PCA-frame mean, so it is the baked
    # per-layer ``origin`` (the authoring-coord foot of the neutral mean) mapped
    # through ``eval_at`` → basis → ``chol``.  The neutral candidate's distance
    # is then ``‖cdist_query − neutral_white‖``, identical machinery to a node.
    neutral_white: torch.Tensor  # (R,)
    # ``node_white`` with the neutral anchor appended as the ``K``-th candidate
    # row **iff** the probe injects neutral — i.e. exactly the
    # ``(Kc, R)`` cdist target the per-probe read needs.  Precomputed at attach
    # so the per-token curved/gate path does one ``vector_norm`` against a
    # standing tensor instead of re-``cat``-ing ``neutral_white`` onto
    # ``node_white`` every token.  Equals ``node_white`` when neutral is not
    # injected.
    node_white_aug: torch.Tensor  # (Kc, R)
    mean: torch.Tensor         # (D,) fit mean on the scoring device
    basis: torch.Tensor        # (R, D) basis on the scoring device
    X: torch.Tensor            # (N, D) centered neutral observations
    K_inv: torch.Tensor        # (N, N) Woodbury inverse
    lam: float                 # ridge λ
    # Precomputed ``Σ⁻¹ mean`` (D,) — the per-probe-per-layer centering term.
    # ``_woodbury_apply`` is exactly linear, so the per-token centered apply
    # ``Σ⁻¹(h − mean)`` is ``Σ⁻¹h − s_mean``: one subtract against a shared
    # ``Σ⁻¹h`` (the same for every probe at a layer, cached in ``_score_full``)
    # instead of a full Woodbury apply per probe.  Baked at attach (and rebuilt
    # on ``set_whitener``), since ``mean`` and the whitener are both constant.
    s_mean: torch.Tensor       # (D,) = Σ⁻¹ mean
    # Affine reduced→domain coordinate map (flat probes only; ``None`` for a
    # curved fit, which recovers coords through ``invert_parameterization``).
    # ``dom = c @ coord_S.T + coord_b`` sends the whitened M-orthogonal
    # reduced coords ``c = M_R⁻¹ B Σ⁻¹ x`` to the fit's domain frame, fit by
    # least squares so each node's reduced coord maps to its
    # ``node_coords`` (the rank-1 case reproduces the old slope/intercept).
    coord_S: torch.Tensor | None = None   # (n_dim, R)
    coord_b: torch.Tensor | None = None   # (n_dim,)


# ----------------------------------------------------------------------------
# Shared subspace-read machinery — used by both read paths of the unified
# Monitor.  Flat (affine) probes get an analytic coordinate readout;
# curved probes foot-solve against the aggregate.  Both share the
# whitened-factor build, the per-layer geometry, and the attach-time node
# cache — the read-side peer of the steering split (one subspace_inject kernel,
# SteeringManager.{subspaces, manifolds}).
# ----------------------------------------------------------------------------


def _probe_is_affine_for_manifold(manifold: "Manifold") -> bool:
    """True iff a manifold's fit is flat (affine) — batched coordinate readout.

    A manifold's layers are uniformly affine (``pca`` / 2-node concept) or
    curved (``spectral`` / ``authored``), so the first fitted layer decides.
    An empty manifold is treated as affine (it scores to nothing anyway).
    """
    from saklas.core.manifold import manifold_is_affine

    return manifold_is_affine(manifold)


def _affine_coord_map(
    sub: Any, manifold: "Manifold", m_r_inv: torch.Tensor,
    X: torch.Tensor, K: torch.Tensor, lam: float,
) -> "tuple[torch.Tensor, torch.Tensor]":
    """Least-squares affine map: whitened M-proj reduced coords → domain.

    Returns ``(coord_S (n_dim, R), coord_b (n_dim,))`` such that
    ``dom = c @ coord_Sᵀ + coord_b`` sends each node's whitened M-orthogonal
    reduced coords ``c_k = M_R⁻¹ B Σ⁻¹ (v_k − mean)`` to its domain
    ``node_coords`` — the rank-R generalization of the old rank-1 2-node
    slope/intercept (which it reproduces exactly at R=1, K=2).  The small
    solve runs on CPU (MPS has ``linalg.lstsq`` gaps); the per-node
    ``Σ⁻¹`` apply rides the device Woodbury.
    """
    dev = X.device
    sub_f = sub.to(device=dev, dtype=torch.float32)
    basis = sub_f.basis                                   # (R, D)
    R = int(basis.shape[0])
    node_coords = manifold.node_coords.to(device=dev, dtype=torch.float32)
    K_nodes = int(node_coords.shape[0])
    n_dim = int(manifold.domain.intrinsic_dim)
    assert sub_f.node_coords is not None
    v_centered = sub_f.node_coords.to(device=dev, dtype=torch.float32) @ basis
    si_vc = _woodbury_apply(v_centered, X, K, lam)       # (K, D) = Σ⁻¹ v_centered
    g_nodes = si_vc @ basis.T                             # (K, R) = B Σ⁻¹ v_centered
    c_nodes = (g_nodes @ m_r_inv.T).cpu()                # (K, R) whitened M-proj coords
    dc = node_coords.reshape(K_nodes, -1)[:, :n_dim].cpu()  # (K, n_dim) domain target
    if K_nodes == 1:
        # Monopolar ray (R == 1): anchor through the origin (neutral → 0),
        # the single node → its domain coord; minimal-norm S, b = 0.
        c0 = c_nodes[0]
        denom = (c0 @ c0).clamp(min=_FRACTION_EPSILON)
        coord_S = torch.outer(dc[0], c0) / denom         # (n_dim, R)
        coord_b = torch.zeros(n_dim, dtype=torch.float32)
    else:
        ones = torch.ones((K_nodes, 1), dtype=torch.float32)
        c1 = torch.cat([c_nodes, ones], dim=1)           # (K, R+1)
        sol = torch.linalg.lstsq(c1, dc).solution        # (R+1, n_dim)
        coord_S = sol[:R].T.contiguous()                 # (n_dim, R)
        coord_b = sol[R].contiguous()                    # (n_dim,)
    return (
        coord_S.to(device=dev, dtype=torch.float32),
        coord_b.to(device=dev, dtype=torch.float32),
    )


def _build_whitened_factors(
    whitener: Any, probe: "AttachedManifoldProbe",
    *,
    factor_cache: dict[
        tuple[int, str, torch.dtype],
        tuple[torch.Tensor, torch.Tensor, float],
    ] | None = None,
) -> dict[int, "_LayerWhiten"]:
    """Per-layer :class:`_LayerWhiten` map for a probe (Mahalanobis-only).

    The wired whitener is **required** and must cover **every** fit layer
    (all-or-nothing per probe).  A missing/non-covering whitener raises
    :class:`~saklas.core.mahalanobis.WhitenerError` — there is no Euclidean
    readout.  Off the hot path; runs once at attach / on ``set_whitener``.
    """
    manifold = probe.manifold
    if not manifold.layers:
        return {}
    layers = list(manifold.layers.keys())
    if whitener is None or not whitener.covers_all(layers):
        raise WhitenerError(
            "subspace probe reads require a Mahalanobis whitener covering "
            f"every fit layer {sorted(layers)}; regenerate the neutral "
            "activation cache for this model (the Euclidean path is gone)"
        )
    out: dict[int, _LayerWhiten] = {}
    for layer_idx, sub in manifold.layers.items():
        v_reduced = probe.node_values_reduced.get(layer_idx)
        if v_reduced is None:
            raise WhitenerError(
                f"subspace probe cache missing reduced node coords for "
                f"layer {layer_idx}; rebuild the probe before scoring"
            )
        dev = v_reduced.device
        basis = sub.basis.to(device=torch.device("cpu"), dtype=torch.float32)
        # M_R = B Σ⁻¹ Bᵀ (R, R), PD for an orthonormal B and ridge-PD Σ.
        m_r = whitener.subspace_gram(layer_idx, basis)  # CPU fp32
        R = m_r.shape[0]
        try:
            chol = torch.linalg.cholesky(m_r)
        except torch.linalg.LinAlgError:
            # Defensive jitter for a near-singular subspace gram.
            eye = torch.eye(R, dtype=m_r.dtype)
            jitter = 1e-8 * float(m_r.diagonal().mean().clamp_min(1e-12))
            chol = torch.linalg.cholesky(m_r + jitter * eye)
        m_r_inv = torch.cholesky_inverse(chol)
        cache_key = (layer_idx, str(dev), torch.float32)
        if factor_cache is not None and cache_key in factor_cache:
            X, K_inv, lam = factor_cache[cache_key]
        else:
            X, K_inv, lam = whitener.woodbury_factors(
                layer_idx, device=dev, dtype=torch.float32,
            )
            if factor_cache is not None:
                factor_cache[cache_key] = (X, K_inv, lam)
        chol_dev = chol.to(device=dev, dtype=torch.float32)
        m_r_inv_dev = m_r_inv.to(device=dev, dtype=torch.float32)
        # Flat probes carry the affine reduced→domain coord map (the rank-R
        # generalization of the rank-1 slope/intercept); a curved fit leaves
        # it ``None`` and recovers coords via ``invert_parameterization``.
        coord_S = coord_b = None
        if sub.is_affine:
            coord_S, coord_b = _affine_coord_map(
                sub, manifold, m_r_inv_dev, X, K_inv, lam,
            )
        # Neutral anchor in the whitened metric.  Affine: the frame is
        # neutral-anchored, so neutral is reduced coord 0.  Curved: map the
        # baked per-layer ``origin`` (authoring-coord foot of the neutral mean)
        # through the same eval_at → basis → chol pipeline as a node.
        if sub.is_affine:
            neutral_white = torch.zeros(R, device=dev, dtype=torch.float32)
        else:
            o = manifold.origin[layer_idx]
            o_dom = o.to(device=dev, dtype=torch.float32).reshape(1, -1)
            emb = manifold.domain.embed(
                manifold.domain.clamp_position(o_dom)
            ).to(device=dev, dtype=torch.float32)
            v_centered = sub.eval_at(emb) - sub.mean.to(
                device=dev, dtype=torch.float32,
            )
            v_red = v_centered @ sub.basis.to(
                device=dev, dtype=torch.float32,
            ).T                                  # (1, R)
            neutral_white = (v_red @ chol_dev).reshape(-1)
        mean_dev = sub.mean.to(device=dev, dtype=torch.float32)
        # Precompute ``Σ⁻¹ mean`` once (constant in mean + whitener), so the
        # per-token ``_layer_geometry`` centers via ``sih − s_mean`` against the
        # shared ``Σ⁻¹h`` rather than a full per-probe Woodbury apply.
        s_mean = _woodbury_apply(mean_dev, X, K_inv, lam)
        node_white = v_reduced.to(torch.float32) @ chol_dev   # (K, R)
        # Append the neutral anchor as the ``K``-th candidate once, here, so the
        # per-token read does one ``vector_norm`` against a standing
        # ``(Kc, R)`` tensor instead of a per-token ``cat`` (FIX F4).
        node_white_aug = (
            torch.cat([node_white, neutral_white.reshape(1, -1)], dim=0)
            if probe.inject_neutral else node_white
        )
        out[layer_idx] = _LayerWhiten(
            m_r_inv=m_r_inv_dev,
            chol=chol_dev,
            node_white=node_white,
            neutral_white=neutral_white,
            node_white_aug=node_white_aug,
            mean=mean_dev,
            basis=sub.basis.to(device=dev, dtype=torch.float32),
            X=X, K_inv=K_inv, lam=lam,
            s_mean=s_mean,
            coord_S=coord_S, coord_b=coord_b,
        )
    return out


def _attach_manifold_probe(
    name: str,
    manifold: "Manifold",
    *,
    top_n: int = DEFAULT_NEAREST_TOP_N,
    whitener: Any = None,
    factor_cache: dict[
        tuple[int, str, torch.dtype],
        tuple[torch.Tensor, torch.Tensor, float],
    ] | None = None,
) -> "AttachedManifoldProbe":
    """Build an :class:`AttachedManifoldProbe`: node values + share weights + whitened.

    Pre-caches, once at attach, the per-layer reduced ``(K, R)`` node
    activations (hot-path cdist working space) and the normalized per-layer
    Mahalanobis-share weights, then the Mahalanobis bundle via
    :func:`_build_whitened_factors` (the whitener must cover the fit's layers).
    The share is baked at fit under the same whitener the monitor requires, so
    it is present for every probed layer; an absent share (CPU stub / partial
    fit) falls back to uniform weighting via the floor.
    """
    manifold.validate_runtime_geometry()
    if manifold.node_coords.numel() == 0 or not manifold.node_labels:
        raise ValueError(
            f"manifold {manifold.name!r} carries no node coords / labels"
        )
    node_values_reduced: dict[int, torch.Tensor] = {}
    share_weights_raw: dict[int, float] = {}
    coords = manifold.node_coords.to(torch.float32)
    clamped = manifold.domain.clamp_position(coords)
    embedded = manifold.domain.embed(clamped)  # (K, m)
    for layer_idx, sub in manifold.layers.items():
        sub_f32 = sub.to(device=sub.mean.device, dtype=torch.float32)
        if sub_f32.is_affine and sub_f32.node_coords is not None:
            # Flat fit: the per-layer reduced node coords ARE the cdist working
            # space — analytically ``c_k = node_coords`` in the same M-projected
            # frame the running-activation query uses.  They're already pruned
            # in lockstep with a DLS-pruned basis, whereas feeding the shared
            # (intrinsic_dim) domain layout through ``eval_at`` mis-dimensions a
            # pruned layer (basis rank R < intrinsic_dim ⇒ ``embedded @ basis``
            # shape mismatch) *and* conflates the abstract layout frame with the
            # per-layer activation frame.
            v_reduced = sub_f32.node_coords.to(torch.float32)  # (K, R)
        else:
            embedded_dev = embedded.to(
                device=sub_f32.mean.device, dtype=torch.float32,
            )
            v_world = sub_f32.eval_at(embedded_dev)  # (K, D)
            v_centered = v_world - sub_f32.mean
            v_reduced = v_centered @ sub_f32.basis.T  # (K, R)
        node_values_reduced[layer_idx] = v_reduced.contiguous()
        share_weights_raw[layer_idx] = float(manifold.mahalanobis_share[layer_idx])
    if any(
        not math.isfinite(value) or value <= 0.0
        for value in share_weights_raw.values()
    ):
        raise ValueError("manifold probe shares must be finite and positive")
    total = sum(max(_MIN_SHARE_WEIGHT, w) for w in share_weights_raw.values())
    share_weights = {
        idx: max(_MIN_SHARE_WEIGHT, w) / total
        for idx, w in share_weights_raw.items()
    }
    inject_neutral = NEUTRAL_LABEL not in manifold.node_labels
    candidate_labels = tuple(manifold.node_labels)
    label_to_candidate_idx = {
        label: idx for idx, label in enumerate(manifold.node_labels)
    }
    if inject_neutral:
        label_to_candidate_idx[NEUTRAL_LABEL] = len(manifold.node_labels)
        candidate_labels = (*candidate_labels, NEUTRAL_LABEL)
    probe = AttachedManifoldProbe(
        name=name,
        manifold=manifold,
        top_n=int(top_n),
        is_affine=_probe_is_affine_for_manifold(manifold),
        inject_neutral=inject_neutral,
        candidate_labels=candidate_labels,
        label_to_candidate_idx=label_to_candidate_idx,
        node_values_reduced=node_values_reduced,
        share_weights=share_weights,
    )
    probe.whitened = _build_whitened_factors(
        whitener, probe, factor_cache=factor_cache,
    )
    bw, lvb = _compute_assign_bandwidth(probe, embedded)
    # Park the soft-assignment factors on the scoring device once (FIX F4), so
    # the per-token ``probe.assign_bandwidth.to(dist_acc_t.device, fp32)`` in the
    # curved/gate read is a no-op (``.to`` returns self on a device+dtype match)
    # rather than a per-token host→device copy of the ``(Kc,)`` tensors.
    probe_device = next(iter(node_values_reduced.values())).device
    probe.assign_bandwidth = (
        bw.to(device=probe_device, dtype=torch.float32) if bw is not None else None
    )
    probe.assign_logvol_bias = (
        lvb.to(device=probe_device, dtype=torch.float32) if lvb is not None else None
    )
    # Robust per-probe ``@label`` distance scale: the median of the *node*
    # bandwidths (``bw`` is candidate-order nodes-then-neutral; the node entries
    # are each node's nearest-neighbor whitened spacing, share-weighted across
    # layers).  Median, not per-candidate, so one crowded pair can't make a label
    # weirdly ungateable; a single scalar, so it rescales the reported distance
    # without reordering ``nearest``.  K=1 / no bandwidth ⇒ 1.0 (raw distance).
    n_nodes = len(manifold.node_labels)
    if bw is not None and bw.numel() >= 1 and n_nodes >= 1:
        node_bw = bw[:n_nodes] if bw.numel() >= n_nodes else bw
        if node_bw.numel():
            probe.label_scale = float(
                node_bw.median().clamp(min=_FRACTION_EPSILON)
            )
    return probe


def _compute_assign_bandwidth(
    probe: "AttachedManifoldProbe", embedded: torch.Tensor,
) -> "tuple[torch.Tensor | None, torch.Tensor | None]":
    """Per-candidate ``(τ, −R·log(τ))`` ``(Kc,)`` each, share-weighted.

    ``τ`` is the soft-assignment bandwidth in the **whitened** metric the
    nearest-node distances use, candidate order = nodes then the neutral anchor
    (when ``inject_neutral``).  Two sources for ``τ``, per the chosen within-
    node-thickness variance:

    * **curved + σ-field**: each node's reduced within-node thickness
      ``σ(z)`` (:meth:`~saklas.core.manifold.LayerSubspace.sigma_at`) mapped
      into the whitened metric by the isotropic scale ``√(tr(M_R)/R)``
      (``M_R = chol cholᵀ``), so a reduced-space σ is comparable to the
      chol-whitened distances; the neutral anchor takes the per-layer median
      node σ.
    * **flat / no σ-field**: each node's *local layout scale* — its
      nearest-neighbor distance among the whitened node coords ``node_white`` —
      so a dense cluster assigns sharply and an isolated node softly; the neutral
      anchor takes its own nearest-node distance.

    The second return ``−R·log(τ)`` is the precomputed Gaussian log-volume
    bias for the soft-assignment logit: the proper isotropic R-D Gaussian
    posterior is ``softmax(−d²/(2τ²) − R·log(τ))`` (the ``−R·log(τ)`` is the
    log of the normalization ``(2πτ²)^(-R/2)`` with the constant dropped — it
    cancels in softmax).  Without this term the bare ``−d²/2τ²`` softmax has a
    *broadest-node-wins* pathology: a wide-``τ`` candidate's logit sits near 0
    while crisp-``τ`` candidates have strongly-negative logits, so the diffuse
    node swallows probability regardless of distance.  Precomputed once at
    attach so the hot path is a single add.  ``R`` is the manifold's per-layer
    subspace rank (one number per probe — fits are rank-uniform across layers).

    Share-weighted across layers (same weights as every other cross-layer
    read), floored positive.  Returns ``(None, None)`` for a degenerate
    manifold (no layers / single node), which leaves the assignment empty
    without disturbing ``nearest``.
    """
    shared = list(probe.manifold.layers.keys())
    if not shared:
        return None, None
    K = probe.node_values_reduced[shared[0]].shape[0]
    if K < 1:
        return None, None
    sw = probe.share_weights
    acc: torch.Tensor | None = None
    wsum = 0.0
    for layer_idx in shared:
        wh = probe.whitened.get(layer_idx)
        if wh is None:
            continue
        sub = probe.manifold.layers[layer_idx]
        node_white = wh.node_white.to(torch.float32)        # (K, R)
        R = int(node_white.shape[1])
        if (not sub.is_affine) and sub.has_sigma:
            # σ-field (reduced units) → whitened via the isotropic scale.
            emb = embedded.to(device=node_white.device, dtype=torch.float32)
            sig_reduced = sub.sigma_at(emb).reshape(-1).to(torch.float32)  # (K,)
            m_r = wh.chol @ wh.chol.transpose(-1, -2)        # M_R (R, R)
            scale = float((torch.diagonal(m_r).sum() / max(R, 1)).clamp(min=1e-12).sqrt())
            band = sig_reduced * scale                       # (K,) whitened
            neutral_band = band.median()
        else:
            # Local layout scale: each node's nearest-neighbor whitened distance.
            if K >= 2:
                dmat = torch.cdist(node_white, node_white)   # (K, K)
                dmat = dmat + torch.eye(
                    K, device=dmat.device, dtype=dmat.dtype,
                ) * 1e9                                       # mask self
                band = dmat.min(dim=1).values                # (K,)
            else:
                band = node_white.new_ones(K)
            if probe.inject_neutral:
                nd = torch.linalg.vector_norm(
                    node_white - wh.neutral_white.reshape(1, -1), dim=-1,
                )                                            # (K,)
                neutral_band = nd.min()
            else:
                neutral_band = band.median() if K else node_white.new_tensor(1.0)
        cand = (
            torch.cat([band, neutral_band.reshape(1)])
            if probe.inject_neutral else band
        )                                                    # (Kc,)
        w = float(sw[layer_idx])
        acc = cand * w if acc is None else acc + cand * w
        wsum += w
    if acc is None:
        return None, None
    acc_t = acc
    if wsum > _MIN_SHARE_WEIGHT:
        acc_t = acc_t / wsum
    # Floor positive so the softmax denominator never divides by ~0.
    med = float(acc_t.median().clamp(min=1e-6))
    tau = acc_t.clamp(min=1e-3 * med).to(torch.float32)
    # Gaussian log-volume bias ``−R·log(τ)`` for the soft-assignment logit.
    # ``R`` = the manifold's per-layer subspace rank (rank-uniform across a
    # fit's layers), the effective dimension of the space the bandwidth ``τ``
    # lives in.  Precomputed here so the hot path adds a single scalar per
    # candidate with no per-token ``log()``.
    R = int(next(iter(probe.manifold.layers.values())).rank)
    logvol_bias = (-float(R) * torch.log(tau)).to(torch.float32)
    return tau, logvol_bias


def _layer_geometry(
    probe: "AttachedManifoldProbe",
    layer_idx: int,
    h: torch.Tensor,
    sih_cache: dict[int, torch.Tensor] | None = None,
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]":
    """Per-layer readout pieces, shared by the per-token + aggregate paths.

    Returns ``(frac, cdist_query, invert_query, cdist_nodes)``:

    * ``frac`` — scalar tensor in ``[0, 1]``, the whitened **M-orthogonal**
      in-subspace energy share ``sqrt(gᵀ M_R⁻¹ g) / ‖x‖_M`` with
      ``g = B Σ⁻¹ x``.
    * ``cdist_query`` — ``(1, R)`` query (``Lᵀc``) so a plain cdist against
      ``cdist_nodes`` is the Mahalanobis distance.
    * ``invert_query`` — ``(R,)`` M-orthogonal projection coords
      ``c = M_R⁻¹ g`` for :func:`~saklas.core.manifold.invert_parameterization`.
    * ``cdist_nodes`` — ``(K, R)`` whitened node coords.

    ``Σ⁻¹h`` depends only on the layer's hidden state + the shared whitener,
    not on the probe, so within one :meth:`Monitor._score_full` pass (where ``h``
    is fixed) it is computed once per layer and reused across every curved probe
    at that layer.  ``sih_cache`` (keyed by ``layer_idx``, valid only for the
    current ``h``) carries it; on a miss we compute and store it.  The centering
    is then ``Σ⁻¹(h − mean) = Σ⁻¹h − Σ⁻¹mean = sih − wh.s_mean`` (one subtract,
    ``_woodbury_apply`` being exactly linear), instead of a full per-probe apply.
    ``None`` (one-off reads) computes ``Σ⁻¹h`` locally — bit-identical algebra.
    """
    wh = probe.whitened.get(layer_idx)
    if wh is None:
        raise WhitenerError(
            f"subspace probe read missing whitened factors for layer "
            f"{layer_idx}; rebuild the probe (the Euclidean path is gone)"
        )
    device = h.device
    mean = wh.mean.to(device)
    basis = wh.basis.to(device)
    X = wh.X.to(device)
    K_inv = wh.K_inv.to(device)
    s_mean = wh.s_mean.to(device)
    x = h - mean
    # Σ⁻¹h is probe-independent — share it across the layer's curved probes via
    # the per-pass cache, then center by subtracting the precomputed Σ⁻¹mean.
    if sih_cache is not None:
        sih = sih_cache.get(layer_idx)
        if sih is None:
            sih = _woodbury_apply(h, X, K_inv, wh.lam)  # Σ⁻¹ h  (D,)
            sih_cache[layer_idx] = sih
    else:
        sih = _woodbury_apply(h, X, K_inv, wh.lam)       # Σ⁻¹ h  (D,)
    sx = sih - s_mean                  # Σ⁻¹ x = Σ⁻¹h − Σ⁻¹mean  (D,)
    x_mnorm = torch.sqrt(
        (x * sx).sum().clamp(min=0.0)
    ).clamp(min=_FRACTION_EPSILON)
    g = basis @ sx                       # (R,) = B Σ⁻¹ x
    c = wh.m_r_inv @ g                   # (R,) = M_R⁻¹ g  (M-proj coords)
    par_mnorm = torch.sqrt((g * c).sum().clamp(min=0.0))  # ‖P_M x‖_M
    frac = (par_mnorm / x_mnorm).clamp(min=0.0, max=1.0)
    cdist_query = (c.reshape(1, -1) @ wh.chol)  # (1, R) — Lᵀc as row
    return frac, cdist_query, c, wh.node_white
