"""Unit tests for the Profile wrapper class."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from saklas.core.errors import SaklasError
from saklas.core.profile import Profile, ProfileError
from saklas.io.packs import PACK_FORMAT_VERSION


def _mk(layers: Any = (0, 5, 10), dim: int = 8, dtype: Any = torch.float32) -> dict[int, torch.Tensor]:
    return {i: torch.randn(dim, dtype=dtype) for i in layers}


def test_mro_profile_error_is_saklas_and_value_error():
    assert issubclass(ProfileError, SaklasError)
    assert issubclass(ProfileError, ValueError)


def test_construct_and_dict_surface():
    p = Profile(_mk())
    assert p.layers == [0, 5, 10]
    assert len(p) == 3
    assert 5 in p
    assert 99 not in p
    # items/keys/values
    assert set(p.keys()) == {0, 5, 10}
    assert [t.shape for t in p.values()] == [torch.Size([8])] * 3
    for idx, t in p.items():
        assert isinstance(idx, int) and isinstance(t, torch.Tensor)
    # __getitem__
    assert p[0].shape == (8,)


def test_empty_profile_rejected():
    with pytest.raises(ProfileError, match="at least one layer"):
        Profile({})


def test_bad_key_type_rejected():
    with pytest.raises(ProfileError, match="layer key must be int"):
        Profile({"0": torch.zeros(4)})  # pyright: ignore[reportArgumentType]  # str key, intentionally wrong type for the test


def test_bad_value_type_rejected():
    with pytest.raises(ProfileError, match="must be torch.Tensor"):
        Profile({0: [1.0, 2.0]})  # pyright: ignore[reportArgumentType]  # list value, intentionally wrong type for the test


def test_weight_at_missing_raises_profile_error():
    p = Profile(_mk())
    with pytest.raises(ProfileError, match="no tensor for layer 99"):
        p.weight_at(99)
    # present layer just returns the tensor
    assert torch.equal(p.weight_at(5), p[5])


def test_metadata_is_copy():
    meta = {"method": "contrastive_pca", "saklas_version": "1.4.0"}
    p = Profile(_mk(), metadata=meta)
    out = p.metadata
    out["method"] = "mutated"
    assert p.metadata["method"] == "contrastive_pca"


def test_save_load_roundtrip(tmp_path: Path):
    p = Profile(_mk(layers=(0, 3)))
    path = tmp_path / "cv.safetensors"
    p.save(path, metadata={"method": "contrastive_pca"})
    assert path.exists()
    assert path.with_suffix(".json").exists()

    loaded = Profile.load(path)
    assert loaded.layers == [0, 3]
    for idx in loaded.layers:
        assert torch.allclose(loaded[idx], p[idx])
    assert loaded.metadata["method"] == "contrastive_pca"
    assert loaded.metadata["format_version"] == PACK_FORMAT_VERSION


def test_merged_intersection_semantics():
    a = Profile({0: torch.ones(4), 1: torch.ones(4), 2: torch.ones(4)})
    b = Profile({1: torch.ones(4) * 2, 2: torch.ones(4) * 2, 3: torch.ones(4) * 2})
    merged = Profile.merged([(a, 1.0), (b, 0.5)])
    # intersection = {1, 2}
    assert merged.layers == [1, 2]
    # 1*1 + 0.5*2 = 2
    assert torch.allclose(merged[1], torch.full((4,), 2.0))
    assert torch.allclose(merged[2], torch.full((4,), 2.0))


def test_merged_strict_refuses_drop():
    a = Profile({0: torch.ones(4), 1: torch.ones(4)})
    b = Profile({1: torch.ones(4), 2: torch.ones(4)})
    from saklas.io.merge import MergeError
    with pytest.raises(MergeError):
        Profile.merged([(a, 1.0), (b, 1.0)], strict=True)


def test_merged_with_binary_convenience():
    a = Profile({0: torch.ones(4)})
    b = Profile({0: torch.ones(4) * 3})
    out = a.merged_with(b, weights=(1.0, 2.0))
    # 1*1 + 2*3 = 7
    assert torch.allclose(out[0], torch.full((4,), 7.0))


def test_merged_requires_two_components():
    p = Profile({0: torch.ones(4)})
    with pytest.raises(ProfileError, match="at least two"):
        Profile.merged([(p, 1.0)])


def test_promoted_to_dtype_flip_noop_when_matching():
    p = Profile(_mk(dtype=torch.float32))
    same = p.promoted_to(dtype=torch.float32)
    # same instance layers untouched
    for idx in p.layers:
        assert same[idx] is p[idx]

    flipped = p.promoted_to(dtype=torch.float16)
    for idx in p.layers:
        assert flipped[idx].dtype == torch.float16
        # source not mutated
        assert p[idx].dtype == torch.float32


def test_promoted_to_no_args_returns_self():
    p = Profile(_mk())
    assert p.promoted_to() is p


def test_repr_contains_layer_info():
    p = Profile(_mk(layers=range(10)))
    r = repr(p)
    assert "Profile(" in r
    assert "10 layers" in r


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------

# Mahalanobis-only (4.0 collapse): ``cosine_similarity`` requires a covering
# whitener — there is no Euclidean path.  The whitener is built with zero
# neutral means so it doesn't recenter the test vectors; the parallel /
# anti-parallel ±1 invariants hold under any positive-definite metric, while
# the orthogonal case is rewritten against *Mahalanobis* orthogonality.

def _cov_whitener(layers: Any, dim: int):
    from tests._whitener import synthetic_whitener, synthetic_means
    return synthetic_whitener(
        layers, dim, means=synthetic_means(layers, dim, seed=0),
    )


def test_cosine_similarity_identical_profiles():
    """Identical profiles should have whitened cosine 1.0 (metric-invariant)."""
    tensors = _mk(layers=(0, 1, 2), dim=16)
    a = Profile(tensors)
    b = Profile({k: v.clone() for k, v in tensors.items()})
    w = _cov_whitener((0, 1, 2), 16)
    assert a.cosine_similarity(b, whitener=w) == pytest.approx(1.0, abs=1e-5)


def test_cosine_similarity_opposite_profiles():
    """Negated profiles should have whitened cosine -1.0 (metric-invariant)."""
    tensors = _mk(layers=(0, 1, 2), dim=16)
    a = Profile(tensors)
    b = Profile({k: -v for k, v in tensors.items()})
    w = _cov_whitener((0, 1, 2), 16)
    assert a.cosine_similarity(b, whitener=w) == pytest.approx(-1.0, abs=1e-5)


def test_cosine_similarity_mahalanobis_orthogonal_profiles():
    """Profiles M-orthogonal per layer have whitened cosine 0.0.

    Build ``b_L`` Euclidean-orthogonal to ``Σ⁻¹ a_L`` so the Mahalanobis dot
    ``a_L^T Σ⁻¹ b_L`` vanishes — the whitened analogue of the old
    Euclidean-orthogonal construction.
    """
    layers = (0, 1, 2)
    w = _cov_whitener(layers, 16)
    a_tensors, b_tensors = {}, {}
    for L in layers:
        v = torch.randn(16)
        sv = w.apply_inv(L, v)          # Σ⁻¹ a_L
        u = torch.randn(16)
        # Remove the component of u along Σ⁻¹a so <u, Σ⁻¹a> = 0, i.e.
        # <a, u>_M = a^T Σ⁻¹ u = 0 (Σ⁻¹ symmetric).
        u = u - (u @ sv) / (sv @ sv) * sv
        a_tensors[L] = v
        b_tensors[L] = u
    a = Profile(a_tensors)
    b = Profile(b_tensors)
    assert a.cosine_similarity(b, whitener=w) == pytest.approx(0.0, abs=1e-4)


def test_cosine_similarity_partial_layer_overlap():
    """Only shared layers contribute to the similarity."""
    shared = torch.randn(8)
    a = Profile({0: shared.clone(), 1: torch.randn(8)})
    b = Profile({0: shared.clone(), 2: torch.randn(8)})
    w = _cov_whitener((0,), 8)
    # Only layer 0 overlaps, and it's identical -> 1.0 (metric-invariant).
    assert a.cosine_similarity(b, whitener=w) == pytest.approx(1.0, abs=1e-5)


def test_cosine_similarity_empty_intersection_raises():
    """No shared layers should raise ProfileError."""
    a = Profile({0: torch.randn(8)})
    b = Profile({1: torch.randn(8)})
    w = _cov_whitener((0, 1), 8)
    with pytest.raises(ProfileError, match="no shared layers"):
        a.cosine_similarity(b, whitener=w)


def test_cosine_similarity_missing_whitener_raises():
    """Mahalanobis-only: a missing / non-covering whitener is a hard error."""
    from saklas.core.mahalanobis import WhitenerError

    tensors = _mk(layers=(0, 1), dim=8)
    a = Profile(tensors)
    b = Profile({k: v.clone() for k, v in tensors.items()})
    with pytest.raises(WhitenerError, match="whitener"):
        a.cosine_similarity(b)
    # A whitener covering only some shared layers is also rejected.
    partial = _cov_whitener((0,), 8)
    with pytest.raises(WhitenerError, match="whitener"):
        a.cosine_similarity(b, whitener=partial)


def test_cosine_similarity_per_layer():
    """per_layer=True returns a dict of per-layer whitened cosines."""
    tensors = _mk(layers=(0, 5, 10), dim=16)
    a = Profile(tensors)
    b = Profile({k: v.clone() for k, v in tensors.items()})
    w = _cov_whitener((0, 5, 10), 16)
    result = a.cosine_similarity(b, per_layer=True, whitener=w)
    assert isinstance(result, dict)
    assert set(result.keys()) == {0, 5, 10}
    for v in result.values():
        assert v == pytest.approx(1.0, abs=1e-5)


def test_cosine_similarity_per_layer_partial_overlap():
    """per_layer=True only includes shared layers."""
    a = Profile({0: torch.randn(8), 1: torch.randn(8)})
    b = Profile({1: torch.randn(8), 2: torch.randn(8)})
    w = _cov_whitener((1,), 8)
    result: dict[int, float] = a.cosine_similarity(b, per_layer=True, whitener=w)
    assert set(result.keys()) == {1}


# ---------------------------------------------------------------------------
# Profile.projected_away
# ---------------------------------------------------------------------------

def test_projected_away_orthogonality():
    """Result at each shared layer should be orthogonal to other."""
    a = Profile({
        0: torch.tensor([1.0, 2.0, 0.0]),
        1: torch.tensor([3.0, 1.0, 5.0]),
    })
    b = Profile({
        0: torch.tensor([1.0, 0.0, 0.0]),
        1: torch.tensor([0.0, 1.0, 0.0]),
    })
    result = a.projected_away(b)
    for layer in [0, 1]:
        r_f = result[layer].float()
        b_f = b[layer].float()
        dot = torch.dot(r_f, b_f).item()
        assert abs(dot) < 1e-5, f"Layer {layer} not orthogonal: dot={dot}"


def test_projected_away_layer_not_in_other_unchanged():
    """Layers in self but not in other are included unchanged."""
    a = Profile({0: torch.tensor([1.0, 2.0]), 1: torch.tensor([3.0, 4.0])})
    b = Profile({1: torch.tensor([1.0, 0.0])})
    result = a.projected_away(b)
    assert set(result.keys()) == {0, 1}
    assert torch.allclose(result[0], a[0])


def test_projected_away_near_zero_b_skipped():
    """Near-zero b layer (dot < 1e-12) is skipped; self layer copied as-is."""
    a = Profile({0: torch.tensor([1.0, 2.0])})
    b = Profile({0: torch.tensor([0.0, 0.0])})
    result = a.projected_away(b)
    assert torch.allclose(result[0], a[0])


def test_projected_away_empty_intersection_raises():
    """No shared layers should raise ProfileError."""
    a = Profile({0: torch.randn(4)})
    b = Profile({1: torch.randn(4)})
    with pytest.raises(ProfileError, match="no shared layers"):
        a.projected_away(b)


def test_projected_away_full_projection():
    """Projecting a vector fully aligned with b yields near-zero result."""
    v = torch.tensor([3.0, 0.0, 0.0])
    a = Profile({0: v.clone()})
    b = Profile({0: torch.tensor([1.0, 0.0, 0.0])})
    result = a.projected_away(b)
    assert torch.allclose(result[0].float(), torch.zeros(3), atol=1e-6)


def test_projected_away_preserves_metadata():
    """Metadata from self is preserved in the output Profile."""
    a = Profile({0: torch.tensor([1.0, 0.0])}, metadata={"method": "test"})
    b = Profile({0: torch.tensor([1.0, 0.0])})
    result = a.projected_away(b)
    assert result.metadata.get("method") == "test"


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------

def test_repr_short_form_lists_layers():
    """<=4 layers renders the layer list verbatim."""
    p = Profile(_mk(layers=(0, 3)))
    r = repr(p)
    assert "[0, 3]" in r


def test_repr_long_form_shows_range_and_count():
    """>4 layers collapses to ``[first..last] (n layers)``."""
    p = Profile(_mk(layers=range(8)))
    r = repr(p)
    assert "0..7" in r
    assert "8 layers" in r


# ---------------------------------------------------------------------------
# save metadata override
# ---------------------------------------------------------------------------

def test_save_metadata_override_merges_on_top_of_self_metadata(tmp_path: Path):
    """``metadata=`` kwarg to ``save`` overrides matching keys in ``self.metadata``.

    The wire sidecar is deliberately slim (see ``saklas.core.vectors.save_profile``)
    — only the allowlisted provenance fields round-trip.  We verify the merge
    via ``method`` (override wins) and ``statements_sha256`` (self-metadata
    field that the override doesn't touch).
    """
    p = Profile(
        _mk(layers=(0, 1)),
        metadata={"method": "contrastive_pca", "statements_sha256": "selfhash"},
    )
    path = tmp_path / "cv.safetensors"
    p.save(path, metadata={"method": "overridden"})
    loaded = Profile.load(path)
    assert loaded.metadata["method"] == "overridden"
    assert loaded.metadata["statements_sha256"] == "selfhash"


# ---------------------------------------------------------------------------
# SaklasError hierarchy — the v2 guarantee that every custom error
# reparents to SaklasError while preserving its stdlib MRO.
# ---------------------------------------------------------------------------

def test_saklas_error_family_mro_contract():
    """Every saklas-raised exception is a SaklasError AND its stdlib parent."""
    from saklas.core.errors import (
        AmbiguousVariantError,
        SaeBackendImportError,
        SaeCoverageError,
        SaeModelMismatchError,
        SaeReleaseNotFoundError,
        UnknownVariantError,
    )
    cases: list[tuple[type[Exception], type[Exception]]] = [
        (ProfileError, ValueError),
        (SaeBackendImportError, ImportError),
        (SaeReleaseNotFoundError, ValueError),
        (SaeModelMismatchError, ValueError),
        (SaeCoverageError, ValueError),
        (AmbiguousVariantError, ValueError),
        (UnknownVariantError, KeyError),
    ]
    for exc, stdlib_parent in cases:
        assert issubclass(exc, SaklasError), exc
        assert issubclass(exc, stdlib_parent), exc
