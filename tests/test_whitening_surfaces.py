"""Whitening surfaces added in the tier-1/tier-2 sweep.

Pure-CPU verification of the four new Mahalanobis integration points, all
checked against the *definitional* formula (not just "runs"):

* ``ManifoldMonitor`` whitened readout — the M-orthogonal subspace
  projection fraction + Mahalanobis nearest-node distance, vs a reference
  implementation built from ``LayerWhitener.apply_inv`` / ``subspace_gram``.
* ``extract_contrastive`` whitened/Fisher PCA — reduces to ordinary PCA
  under an isotropic Σ, de-rogues under an anisotropic one, falls back to
  Euclidean on partial coverage.
* ``transfer_profile`` target-metric re-bake — per-layer magnitude becomes
  the target Mahalanobis norm; all-or-nothing gate; ``bake`` provenance.
"""
from __future__ import annotations

from typing import Any

import pytest
import torch

from saklas.core import vectors as V
from saklas.core.mahalanobis import LayerWhitener
from saklas.core.manifold import (
    BoxAxis,
    BoxDomain,
    LayerSubspace,
    Manifold,
    fit_layer_subspace,
)
from saklas.core.monitor import ManifoldMonitor


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
        m = _toy_manifold(dim=8, n_layers=2)
        w = _make_whitener(layers=(0, 1), d=8)
        mon = ManifoldMonitor(whitener=w)
        mon.add_probe("toy", m)
        probe = mon.attached_probes()["toy"]
        assert set(probe.whitened.keys()) == {0, 1}

        # Whitener missing a layer → no whitened cache (Euclidean fallback).
        w_partial = _make_whitener(layers=(0,), d=8)
        mon2 = ManifoldMonitor(whitener=w_partial)
        mon2.add_probe("toy", m)
        assert mon2.attached_probes()["toy"].whitened == {}

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
        m = _toy_manifold(dim=8, n_layers=1)
        iso = _make_whitener(layers=(0,), d=8, n=4000)  # Σ ≈ I
        mon_w = ManifoldMonitor(whitener=iso)
        mon_w.add_probe("toy", m)
        mon_e = ManifoldMonitor()  # Euclidean
        mon_e.add_probe("toy", m)
        pw = mon_w.attached_probes()["toy"]
        pe = mon_e.attached_probes()["toy"]

        torch.manual_seed(11)
        for _ in range(5):
            h = torch.randn(8)
            fw = float(mon_w._layer_geometry(pw, 0, h)[0].item())
            fe = float(mon_e._layer_geometry(pe, 0, h)[0].item())
            # Σ ≈ I ⇒ M-orthogonal projection ≈ Euclidean projection.
            assert fw == pytest.approx(fe, abs=0.05)

    def test_set_whitener_flips_readout(self) -> None:
        m = _toy_manifold(dim=8, n_layers=1)
        mon = ManifoldMonitor()  # start Euclidean
        mon.add_probe("toy", m)
        assert mon.attached_probes()["toy"].whitened == {}
        w = _make_whitener(layers=(0,), d=8)
        mon.set_whitener(w)
        assert set(mon.attached_probes()["toy"].whitened.keys()) == {0}
        mon.set_whitener(None)
        assert mon.attached_probes()["toy"].whitened == {}


# --------------------------------------------------------------------------- #
# Whitened / Fisher PCA extraction                                            #
# --------------------------------------------------------------------------- #


class _FakeModel(torch.nn.Module):
    def parameters(self, recurse: bool = True):  # type: ignore[override]
        yield torch.zeros(1)


class _FakeTok:
    pass


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.dot(a / a.norm(), b / b.norm()).item())


def _rogue_encoder(d: int, sig_axis: int, rogue_axis: int, rogue_sigma: float):
    """Encoder stub: per-pair diff = e[sig] + s·e[rogue], s ~ N(0, σ²).

    The *consistent* class signal is on ``sig_axis``; ``rogue_axis`` carries
    large per-pair variance but no consistent signal — exactly the regime
    where ordinary PCA of the diffs chases the rogue axis and whitened PCA
    (with Σ large on the rogue axis) recovers the signal axis.
    """
    def _enc(model: Any, tok: Any, text: str, layers: Any, device: Any,
             **_kw: Any) -> dict[int, torch.Tensor]:
        n = len(layers)
        idx = int(text.split("_")[-1])
        g = torch.Generator().manual_seed(1000 + idx)
        s = float(torch.randn(1, generator=g).item()) * rogue_sigma
        sign = 1.0 if text.startswith("pos") else -1.0
        out: dict[int, torch.Tensor] = {}
        for L in range(n):
            v = torch.zeros(d)
            v[sig_axis] = 0.5 * sign
            v[rogue_axis] = 0.5 * sign * s
            out[L] = v
        return out
    return _enc


class TestWhitenedPCAExtraction:
    def test_isotropic_matches_ordinary_pca(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Clean separable signal along axis 0; isotropic Σ ⇒ whitened PCA
        # ≈ ordinary PCA direction.
        monkeypatch.setattr(
            V, "_encode_and_capture_all",
            _rogue_encoder(d=4, sig_axis=0, rogue_axis=1, rogue_sigma=0.0),
        )
        pairs = [{"positive": f"pos_{i}", "negative": f"neg_{i}"} for i in range(12)]
        layers = [object()] * 3
        plain, _ = V.extract_contrastive(
            _FakeModel(), _FakeTok(), pairs, layers=layers,  # type: ignore[arg-type]
            device=torch.device("cpu"), dls=False,
        )
        iso = _make_whitener(layers=(0, 1, 2), d=4, n=4000)
        whit, _ = V.extract_contrastive(
            _FakeModel(), _FakeTok(), pairs, layers=layers,  # type: ignore[arg-type]
            device=torch.device("cpu"), dls=False, whitener=iso,
        )
        for L in range(3):
            assert abs(_cos(plain[L], whit[L])) > 0.99

    def test_anisotropic_derogues_direction(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Signal on axis 0, large per-pair rogue variance on axis 1.
        monkeypatch.setattr(
            V, "_encode_and_capture_all",
            _rogue_encoder(d=4, sig_axis=0, rogue_axis=1, rogue_sigma=3.0),
        )
        pairs = [{"positive": f"pos_{i}", "negative": f"neg_{i}"} for i in range(24)]
        layers = [object()] * 2
        plain, _ = V.extract_contrastive(
            _FakeModel(), _FakeTok(), pairs, layers=layers,  # type: ignore[arg-type]
            device=torch.device("cpu"), dls=False,
        )
        # Σ huge on the rogue axis (axis 1) so whitened PCA divides it out.
        cov = torch.tensor([1.0, 6.0, 1.0, 1.0])
        w = _make_whitener(layers=(0, 1), d=4, cov_scale=cov, n=2000)
        whit, _ = V.extract_contrastive(
            _FakeModel(), _FakeTok(), pairs, layers=layers,  # type: ignore[arg-type]
            device=torch.device("cpu"), dls=False, whitener=w,
        )
        e0 = torch.tensor([1.0, 0.0, 0.0, 0.0])
        for L in range(2):
            # Whitened direction sits closer to the true signal axis than
            # the rogue-chasing ordinary PCA direction.
            assert abs(_cos(whit[L], e0)) > abs(_cos(plain[L], e0))

    def test_partial_coverage_falls_back_to_euclidean(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            V, "_encode_and_capture_all",
            _rogue_encoder(d=4, sig_axis=0, rogue_axis=1, rogue_sigma=1.0),
        )
        pairs = [{"positive": f"pos_{i}", "negative": f"neg_{i}"} for i in range(12)]
        layers = [object()] * 3
        plain, _ = V.extract_contrastive(
            _FakeModel(), _FakeTok(), pairs, layers=layers,  # type: ignore[arg-type]
            device=torch.device("cpu"), dls=False,
        )
        # Whitener covers only layers {0, 1} — not all 3 → Euclidean for all.
        w_partial = _make_whitener(layers=(0, 1), d=4)
        gated, _ = V.extract_contrastive(
            _FakeModel(), _FakeTok(), pairs, layers=layers,  # type: ignore[arg-type]
            device=torch.device("cpu"), dls=False, whitener=w_partial,
        )
        for L in range(3):
            assert torch.allclose(plain[L], gated[L], atol=1e-5)


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

    def test_no_whitener_is_plain_euclidean_transfer(self) -> None:
        from saklas.io.alignment import transfer_profile
        d, layers = 6, (0, 1, 2)
        src = self._src_profile(d, layers)
        align = {L: torch.eye(d) for L in layers}
        out = transfer_profile(src, align, source_model_id="src/m")
        assert out.metadata["bake"] == "euclidean"
        for L in layers:
            assert torch.allclose(out[L].float(), src[L].float(), atol=1e-5)

    def test_partial_coverage_gate_skips_rebake(self) -> None:
        from saklas.io.alignment import transfer_profile
        d, layers = 6, (0, 1, 2)
        src = self._src_profile(d, layers)
        align = {L: torch.eye(d) for L in layers}
        w_partial = _make_whitener(layers=(0, 1), d=d)  # missing layer 2
        out = transfer_profile(
            src, align, source_model_id="src/m", whitener=w_partial,
        )
        assert out.metadata["bake"] == "euclidean"
        for L in layers:
            assert torch.allclose(out[L].float(), src[L].float(), atol=1e-5)
