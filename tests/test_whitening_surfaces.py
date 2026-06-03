"""Whitening surfaces added in the tier-1/tier-2 sweep.

Pure-CPU verification of the four new Mahalanobis integration points, all
checked against the *definitional* formula (not just "runs"):

* ``ManifoldMonitor`` whitened readout — the M-orthogonal subspace
  projection fraction + Mahalanobis nearest-node distance, vs a reference
  implementation built from ``LayerWhitener.apply_inv`` / ``subspace_gram``.
* ``transfer_profile`` target-metric re-bake — per-layer magnitude becomes
  the target Mahalanobis norm; all-or-nothing gate; ``bake`` provenance.
"""
from __future__ import annotations

import pytest
import torch

from saklas.core.mahalanobis import LayerWhitener
from saklas.core.manifold import (
    BoxAxis,
    BoxDomain,
    LayerSubspace,
    Manifold,
    fit_layer_subspace,
)
from saklas.core.monitor import ManifoldMonitor
from tests._whitener import isotropic_whitener


# --------------------------------------------------------------------------- #
# Shared helpers                                                              #
# --------------------------------------------------------------------------- #


def _make_whitener(
    *,
    layers: tuple[int, ...],
    d: int,
    n: int = 200,
    cov_scale: torch.Tensor | None = None,
    seed: int = 7,
    ridge_scale: float = 1.0,
) -> LayerWhitener:
    """Whitener over synthetic zero-mean neutral activations.

    ``cov_scale`` (a length-``d`` vector) gives a known anisotropic
    covariance; ``None`` draws isotropic N(0, I).
    """
    acts: dict[int, torch.Tensor] = {}
    for L in layers:
        g = torch.Generator().manual_seed(seed + L)
        X = torch.randn(n, d, generator=g)
        if cov_scale is not None:
            X = X * cov_scale.reshape(1, d)
        acts[L] = X.to(torch.float32)
    means = {L: torch.zeros(d) for L in layers}
    return LayerWhitener.from_neutral_activations(
        acts, means, ridge_scale=ridge_scale,
    )


def _toy_manifold(*, dim: int = 8, n_layers: int = 1, seed: int = 0) -> Manifold:
    """1-D BoxDomain manifold, 3 nodes at [-1, 0, 1], centroids in (e0, e1)."""
    torch.manual_seed(seed)
    domain = BoxDomain([BoxAxis("u", periodic=False, lo=-1.0, hi=1.0)])
    coords = torch.tensor([[-1.0], [0.0], [1.0]])
    e0 = torch.zeros(dim)
    e0[0] = 1.0
    e1 = torch.zeros(dim)
    e1[1] = 1.0
    layers: dict[int, LayerSubspace] = {}
    ev: dict[int, float] = {}
    for layer_idx in range(n_layers):
        scale = 1.0 + 0.5 * layer_idx
        centroids = torch.stack([-scale * e0, torch.zeros(dim), scale * e0])
        centroids = centroids + 0.01 * torch.stack([-e1, torch.zeros(dim), e1])
        sub, ev_ratio = fit_layer_subspace(centroids, domain.embed(coords))
        layers[layer_idx] = sub
        ev[layer_idx] = ev_ratio
    return Manifold(
        name="toy", domain=domain, node_labels=["a", "b", "c"],
        node_coords=coords, layers=layers, explained_variance=ev,
    )


# --------------------------------------------------------------------------- #
# ManifoldMonitor whitened readout                                            #
# --------------------------------------------------------------------------- #


class TestManifoldMonitorWhitened:
    def test_whitened_cache_built_all_or_nothing(self) -> None:
        from saklas.core.mahalanobis import WhitenerError

        m = _toy_manifold(dim=8, n_layers=2)
        w = _make_whitener(layers=(0, 1), d=8)
        mon = ManifoldMonitor(whitener=w)
        mon.add_probe("toy", m)
        probe = mon.attached_probes()["toy"]
        assert set(probe.whitened.keys()) == {0, 1}

        # Whitener missing a layer → hard error (Mahalanobis is mandatory;
        # no Euclidean fallback).
        w_partial = _make_whitener(layers=(0,), d=8)
        mon2 = ManifoldMonitor(whitener=w_partial)
        with pytest.raises(WhitenerError, match="whitener"):
            mon2.add_probe("toy", m)

    def test_fraction_matches_reference_and_in_range(self) -> None:
        m = _toy_manifold(dim=8, n_layers=1)
        # Anisotropic Σ so the whitened metric genuinely differs from L2.
        cov = torch.tensor([1.0, 1.0, 4.0, 4.0, 0.5, 0.5, 2.0, 2.0])
        w = _make_whitener(layers=(0,), d=8, cov_scale=cov)
        mon = ManifoldMonitor(whitener=w)
        mon.add_probe("toy", m)
        probe = mon.attached_probes()["toy"]
        sub = m.layers[0]

        torch.manual_seed(3)
        for _ in range(5):
            h = torch.randn(8)
            frac_t, cdist_q, invert_q, cdist_nodes = mon._layer_geometry(
                probe, 0, h,
            )
            frac = float(frac_t.item())
            assert 0.0 <= frac <= 1.0 + 1e-6

            # Reference M-orthogonal projection fraction.
            x = h - sub.mean
            sx = w.apply_inv(0, x)
            x_m = float(torch.sqrt((x * sx).sum().clamp_min(0.0)))
            g = sub.basis @ sx
            m_r = w.subspace_gram(0, sub.basis)
            c_ref = torch.linalg.solve(m_r, g)
            par_m = float(torch.sqrt((g * c_ref).sum().clamp_min(0.0)))
            frac_ref = par_m / max(x_m, 1e-12)
            assert frac == pytest.approx(frac_ref, abs=1e-4)

            # M-projection coords feed invert_parameterization.
            assert torch.allclose(invert_q, c_ref, atol=1e-4)

            # Mahalanobis nearest-node distances: cdist in whitened space
            # equals sqrt((c - v_k)ᵀ M_R (c - v_k)).
            dists = torch.cdist(cdist_q, cdist_nodes).reshape(-1)
            v_reduced = probe.node_values_reduced[0]
            for k in range(v_reduced.shape[0]):
                delta = c_ref - v_reduced[k]
                ref_d = float(torch.sqrt((delta @ m_r @ delta).clamp_min(0.0)))
                assert float(dists[k]) == pytest.approx(ref_d, abs=1e-4)

    def test_isotropic_reduces_to_euclidean(self) -> None:
        """Under an isotropic whitener (Σ ≈ I) the M-orthogonal fraction
        equals the plain Euclidean projection share.

        The Euclidean ``ManifoldMonitor`` path was removed in the 4.0
        Mahalanobis-only collapse, so the reference is computed by hand:
        the L2-orthogonal projection share ``‖x_par‖₂ / ‖x‖₂`` of the
        centered hidden onto the subspace basis.  With Σ ≈ I the whitened
        readout must reproduce it (the synthetic isotropic Σ isn't exactly
        I, hence the loose tolerance)."""
        m = _toy_manifold(dim=8, n_layers=1)
        iso = isotropic_whitener((0,), 8, n=4000)  # Σ ≈ I
        mon = ManifoldMonitor(whitener=iso)
        mon.add_probe("toy", m)
        p = mon.attached_probes()["toy"]
        sub = m.layers[0]

        torch.manual_seed(11)
        for _ in range(5):
            h = torch.randn(8)
            f = float(mon._layer_geometry(p, 0, h)[0].item())
            assert 0.0 <= f <= 1.0 + 1e-6

            # Euclidean reference: L2-orthogonal projection energy share onto
            # the subspace basis (Σ⁻¹ → I in the whitened formula).
            x = h - sub.mean
            g_e = sub.basis @ x
            m_r_e = sub.basis @ sub.basis.T
            c_e = torch.linalg.solve(m_r_e, g_e)
            par_e = float(torch.sqrt((g_e * c_e).sum().clamp_min(0.0)))
            frac_e = par_e / max(float(x.norm()), 1e-12)
            # Σ ≈ I ⇒ M-orthogonal projection ≈ Euclidean projection.
            assert f == pytest.approx(frac_e, abs=2e-2)

    def test_set_whitener_rebuilds_per_probe_cache(self) -> None:
        """``set_whitener`` rebuilds the per-probe whitened factors over the
        new covariance (Mahalanobis-only — there is no Euclidean readout)."""
        m = _toy_manifold(dim=8, n_layers=1)
        w1 = _make_whitener(layers=(0,), d=8, seed=1)
        mon = ManifoldMonitor(whitener=w1)
        mon.add_probe("toy", m)
        assert set(mon.attached_probes()["toy"].whitened.keys()) == {0}
        # Swap in a different covering whitener → cache rebuilt for layer 0.
        w2 = _make_whitener(layers=(0,), d=8, seed=2)
        mon.set_whitener(w2)
        assert set(mon.attached_probes()["toy"].whitened.keys()) == {0}


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.dot(a / a.norm(), b / b.norm()).item())


# --------------------------------------------------------------------------- #
# transfer_profile target-metric re-bake                                      #
# --------------------------------------------------------------------------- #


class TestTransferProfileRebake:
    def _src_profile(self, d: int, layers: tuple[int, ...]):
        from saklas.core.profile import Profile
        torch.manual_seed(5)
        return Profile({L: torch.randn(d) for L in layers})

    def test_rebake_magnitude_is_target_mahalanobis_norm(self) -> None:
        from saklas.io.alignment import transfer_profile
        d, layers = 6, (0, 1, 2)
        src = self._src_profile(d, layers)
        # Identity alignment (orthogonal) so v_tgt == v_src — isolates the
        # re-bake effect from the rotation.
        align = {L: torch.eye(d) for L in layers}
        w = _make_whitener(
            layers=layers, d=d,
            cov_scale=torch.tensor([1.0, 3.0, 0.5, 2.0, 1.0, 4.0]),
        )
        out = transfer_profile(
            src, align, source_model_id="src/m", whitener=w,
        )
        assert out.metadata["bake"] == "mahalanobis"
        for L in layers:
            v_tgt = align[L] @ src[L].float()
            expected_norm = w.mahalanobis_norm(L, v_tgt)
            assert float(out[L].norm()) == pytest.approx(expected_norm, abs=1e-4)
            # Direction preserved (only magnitude re-scaled).
            assert abs(_cos(out[L].float(), src[L].float())) == pytest.approx(
                1.0, abs=1e-5,
            )

    def test_no_whitener_raises(self) -> None:
        """Transfer is Mahalanobis-only: no target whitener is a hard error
        (the Euclidean transfer path is gone)."""
        from saklas.core.mahalanobis import WhitenerError
        from saklas.io.alignment import transfer_profile
        d, layers = 6, (0, 1, 2)
        src = self._src_profile(d, layers)
        align = {L: torch.eye(d) for L in layers}
        with pytest.raises(WhitenerError, match="TARGET"):
            transfer_profile(src, align, source_model_id="src/m")

    def test_partial_coverage_raises(self) -> None:
        from saklas.core.mahalanobis import WhitenerError
        from saklas.io.alignment import transfer_profile
        d, layers = 6, (0, 1, 2)
        src = self._src_profile(d, layers)
        align = {L: torch.eye(d) for L in layers}
        w_partial = _make_whitener(layers=(0, 1), d=d)  # missing layer 2
        with pytest.raises(WhitenerError, match="TARGET"):
            transfer_profile(
                src, align, source_model_id="src/m", whitener=w_partial,
            )
