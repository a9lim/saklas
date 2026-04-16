"""Unit tests for the Profile wrapper class."""
from __future__ import annotations

import pytest
import torch

from saklas.core.errors import SaklasError
from saklas.core.profile import Profile, ProfileError


def _mk(layers=(0, 5, 10), dim=8, dtype=torch.float32):
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
        Profile({"0": torch.zeros(4)})


def test_bad_value_type_rejected():
    with pytest.raises(ProfileError, match="must be torch.Tensor"):
        Profile({0: [1.0, 2.0]})  # type: ignore[arg-type]


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


def test_save_load_roundtrip(tmp_path):
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
    assert loaded.metadata["format_version"] == 2


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

def test_cosine_similarity_identical_profiles():
    """Identical profiles should have cosine similarity 1.0."""
    tensors = _mk(layers=(0, 1, 2), dim=16)
    a = Profile(tensors)
    b = Profile({k: v.clone() for k, v in tensors.items()})
    assert a.cosine_similarity(b) == pytest.approx(1.0, abs=1e-5)


def test_cosine_similarity_opposite_profiles():
    """Negated profiles should have cosine similarity -1.0."""
    tensors = _mk(layers=(0, 1, 2), dim=16)
    a = Profile(tensors)
    b = Profile({k: -v for k, v in tensors.items()})
    assert a.cosine_similarity(b) == pytest.approx(-1.0, abs=1e-5)


def test_cosine_similarity_orthogonal_profiles():
    """Orthogonal profiles should have cosine similarity 0.0."""
    # Construct two orthogonal vectors per layer via Gram-Schmidt.
    layers = (0, 1, 2)
    a_tensors, b_tensors = {}, {}
    for L in layers:
        v = torch.randn(16)
        u = torch.randn(16)
        # Remove component of v from u -> orthogonal.
        u = u - (u @ v) / (v @ v) * v
        a_tensors[L] = v
        b_tensors[L] = u
    a = Profile(a_tensors)
    b = Profile(b_tensors)
    assert a.cosine_similarity(b) == pytest.approx(0.0, abs=1e-4)


def test_cosine_similarity_partial_layer_overlap():
    """Only shared layers contribute to the similarity."""
    shared = torch.randn(8)
    a = Profile({0: shared.clone(), 1: torch.randn(8)})
    b = Profile({0: shared.clone(), 2: torch.randn(8)})
    # Only layer 0 overlaps, and it's identical -> 1.0.
    assert a.cosine_similarity(b) == pytest.approx(1.0, abs=1e-5)


def test_cosine_similarity_empty_intersection_raises():
    """No shared layers should raise ProfileError."""
    a = Profile({0: torch.randn(8)})
    b = Profile({1: torch.randn(8)})
    with pytest.raises(ProfileError, match="no shared layers"):
        a.cosine_similarity(b)


def test_cosine_similarity_per_layer():
    """per_layer=True returns a dict of per-layer cosines."""
    tensors = _mk(layers=(0, 5, 10), dim=16)
    a = Profile(tensors)
    b = Profile({k: v.clone() for k, v in tensors.items()})
    result = a.cosine_similarity(b, per_layer=True)
    assert isinstance(result, dict)
    assert set(result.keys()) == {0, 5, 10}
    for v in result.values():
        assert v == pytest.approx(1.0, abs=1e-5)


def test_cosine_similarity_per_layer_partial_overlap():
    """per_layer=True only includes shared layers."""
    a = Profile({0: torch.randn(8), 1: torch.randn(8)})
    b = Profile({1: torch.randn(8), 2: torch.randn(8)})
    result = a.cosine_similarity(b, per_layer=True)
    assert set(result.keys()) == {1}
