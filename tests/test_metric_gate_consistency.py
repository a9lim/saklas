"""All-or-nothing metric gate consistency across the whitener surfaces.

Both :meth:`Profile.cosine_similarity` and :func:`project_profile` switch
to the Mahalanobis metric **only** when the whitener covers *every*
relevant layer (``LayerWhitener.covers_all``).  Partial coverage must
fall back to Euclidean for *all* layers — never a per-layer mix — because
the per-layer ``‖·‖_M`` and ``‖·‖_2`` scales differ by a ``1/√λ_L``
factor that doesn't cancel across the cross-layer aggregate.

These are CPU-only — no model load.  A whitener is built over a strict
subset of layers ({0, 1}) while the profiles span a superset ({0, 1, 2}),
so partial coverage is exercised directly, and a full-coverage whitener
({0, 1, 2}) confirms the Mahalanobis path actually engages.
"""

from __future__ import annotations

import torch

from saklas.core.mahalanobis import LayerWhitener
from saklas.core.profile import Profile
from saklas.core.vectors import project_profile


_D = 8
_N = 16
_LAYERS = (0, 1, 2)


def _whitener(layers: tuple[int, ...], *, seed: int = 0) -> LayerWhitener:
    """Build a small whitener over synthetic neutral activations."""
    torch.manual_seed(seed)
    neutral = {L: torch.randn(_N, _D, dtype=torch.float32) for L in layers}
    means = {L: neutral[L].mean(dim=0) for L in layers}
    return LayerWhitener.from_neutral_activations(neutral, means)


def _profile(seed: int) -> Profile:
    torch.manual_seed(seed)
    return Profile({L: torch.randn(_D, dtype=torch.float32) for L in _LAYERS})


# ---- Profile.cosine_similarity ----------------------------------------


def test_cosine_partial_coverage_matches_euclidean():
    """Partial whitener coverage → identical to whitener=None (no whiten)."""
    a, b = _profile(1), _profile(2)
    partial = _whitener((0, 1), seed=7)  # covers a strict subset of {0,1,2}

    euclid = a.cosine_similarity(b, whitener=None)
    gated = a.cosine_similarity(b, whitener=partial)
    assert gated == euclid


def test_cosine_full_coverage_diverges_from_euclidean():
    """Full whitener coverage → Mahalanobis path, differs from Euclidean."""
    a, b = _profile(1), _profile(2)
    full = _whitener(_LAYERS, seed=7)

    euclid = a.cosine_similarity(b, whitener=None)
    whitened = a.cosine_similarity(b, whitener=full)
    assert whitened != euclid


def test_cosine_per_layer_partial_coverage_matches_euclidean():
    """The per-layer dict path honors the same all-or-nothing gate."""
    a, b = _profile(1), _profile(2)
    partial = _whitener((0, 1), seed=7)

    euclid = a.cosine_similarity(b, per_layer=True, whitener=None)
    gated = a.cosine_similarity(b, per_layer=True, whitener=partial)
    assert gated == euclid


def test_cosine_per_layer_full_coverage_diverges():
    """Per-layer full coverage whitens every layer (each entry shifts)."""
    a, b = _profile(1), _profile(2)
    full = _whitener(_LAYERS, seed=7)

    euclid = a.cosine_similarity(b, per_layer=True, whitener=None)
    whitened = a.cosine_similarity(b, per_layer=True, whitener=full)
    assert set(whitened) == set(euclid)
    assert all(whitened[L] != euclid[L] for L in euclid)


# ---- project_profile --------------------------------------------------


def _project_dicts(
    base: Profile, onto: Profile, operator: str,
    whitener: LayerWhitener | None,
) -> dict[int, torch.Tensor]:
    return project_profile(
        dict(base), dict(onto), operator, whitener=whitener
    )


def _same_projection(
    p: dict[int, torch.Tensor], q: dict[int, torch.Tensor],
) -> bool:
    if set(p) != set(q):
        return False
    return all(torch.equal(p[L], q[L]) for L in p)


def test_project_partial_coverage_matches_gram_schmidt():
    """Partial whitener coverage → identical to whitener=None projection."""
    base, onto = _profile(3), _profile(4)
    partial = _whitener((0, 1), seed=7)

    euclid = _project_dicts(base, onto, "|", None)
    gated = _project_dicts(base, onto, "|", partial)
    assert _same_projection(gated, euclid)


def test_project_full_coverage_diverges_from_gram_schmidt():
    """Full whitener coverage → LEACE projection, differs from Euclidean."""
    base, onto = _profile(3), _profile(4)
    full = _whitener(_LAYERS, seed=7)

    euclid = _project_dicts(base, onto, "|", None)
    leace = _project_dicts(base, onto, "|", full)
    assert set(leace) == set(euclid)
    assert not _same_projection(leace, euclid)


def test_project_tilde_partial_coverage_matches_gram_schmidt():
    """The ``~`` operator honors the same all-or-nothing gate."""
    base, onto = _profile(3), _profile(4)
    partial = _whitener((0, 1), seed=7)

    euclid = _project_dicts(base, onto, "~", None)
    gated = _project_dicts(base, onto, "~", partial)
    assert _same_projection(gated, euclid)
