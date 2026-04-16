import pytest
import torch

from saklas.io import merge, packs
from saklas.core.vectors import save_profile


def _cs(coord, alpha, project_away=None):
    return merge.ComponentSpec(coord=coord, project_away=project_away, alpha=alpha)


def test_parse_components_two():
    out = merge.parse_components("default/happy:0.3,a9lim/archaic:0.4")
    assert out == [_cs("default/happy", 0.3), _cs("a9lim/archaic", 0.4)]


def test_parse_components_three():
    out = merge.parse_components("a:0.1,b:0.2,c:0.3")
    assert [c.coord for c in out] == ["a", "b", "c"]
    assert [c.alpha for c in out] == [0.1, 0.2, 0.3]


def test_parse_components_requires_alpha():
    with pytest.raises(merge.MergeError, match="alpha"):
        merge.parse_components("a,b:0.2")


def test_parse_components_single_accepted():
    """Single component is now valid (minimum dropped to 1)."""
    out = merge.parse_components("a:0.5")
    assert out == [_cs("a", 0.5)]


def test_parse_components_projection():
    """a~b:0.5 parses to ComponentSpec with project_away set."""
    out = merge.parse_components("default/happy~default/sad:0.5")
    assert len(out) == 1
    assert out[0].coord == "default/happy"
    assert out[0].project_away == "default/sad"
    assert out[0].alpha == 0.5


def test_parse_components_projection_mixed():
    """Mix of plain and projection components."""
    out = merge.parse_components("a~b:0.5,c:1.0")
    assert out[0] == _cs("a", 0.5, "b")
    assert out[1] == _cs("c", 1.0)


def test_parse_components_chained_tilde_rejected():
    """a~b~c:0.5 is a parse error."""
    with pytest.raises(merge.MergeError, match="chained"):
        merge.parse_components("a~b~c:0.5")


def test_parse_components_bare_projection_rejected():
    """a~b without :alpha is rejected."""
    with pytest.raises(merge.MergeError, match="alpha"):
        merge.parse_components("a~b")


def test_linear_sum_equal_layers():
    a = {0: torch.tensor([1.0, 0.0]), 1: torch.tensor([0.0, 1.0])}
    b = {0: torch.tensor([2.0, 0.0]), 1: torch.tensor([0.0, 2.0])}
    out = merge.linear_sum([(a, 0.5), (b, 0.25)])
    assert torch.allclose(out[0], torch.tensor([1.0, 0.0]))
    assert torch.allclose(out[1], torch.tensor([0.0, 1.0]))


def test_linear_sum_layer_intersection():
    a = {0: torch.tensor([1.0, 0.0]),
         1: torch.tensor([0.0, 1.0]),
         2: torch.tensor([1.0, 1.0])}
    b = {1: torch.tensor([0.0, 2.0]),
         2: torch.tensor([2.0, 2.0])}
    out = merge.linear_sum([(a, 1.0), (b, 1.0)])
    assert sorted(out.keys()) == [1, 2]


def test_linear_sum_empty_intersection_raises():
    a = {0: torch.tensor([1.0])}
    b = {1: torch.tensor([2.0])}
    with pytest.raises(merge.MergeError, match="no common layers"):
        merge.linear_sum([(a, 1.0), (b, 1.0)])


def _make_concept_with_tensors(tmp_path, ns, name, model_tensors):
    d = tmp_path / "vectors" / ns / name
    d.mkdir(parents=True)
    (d / "statements.json").write_text("[]")
    files = {"statements.json": packs.hash_file(d / "statements.json")}
    for model_id, profile in model_tensors.items():
        ts = d / f"{model_id}.safetensors"
        save_profile(profile, str(ts), {"method": "contrastive_pca"})
        files[f"{model_id}.safetensors"] = packs.hash_file(ts)
        files[f"{model_id}.json"] = packs.hash_file(ts.with_suffix(".json"))
    packs.PackMetadata(
        name=name, description="x", version="1.0.0", license="MIT",
        tags=[], recommended_alpha=0.5, source="local",
        files=files,
    ).write(d)
    return d


def test_shared_models_intersection(monkeypatch, tmp_path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    profile = {0: torch.tensor([1.0])}
    _make_concept_with_tensors(tmp_path, "default", "happy",
                                {"gemma": profile, "qwen": profile})
    _make_concept_with_tensors(tmp_path, "a9lim", "archaic",
                                {"gemma": profile})
    shared = merge.shared_models([_cs("default/happy", 0.5), _cs("a9lim/archaic", 0.5)])
    assert shared == ["gemma"]


def test_shared_models_empty_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    profile = {0: torch.tensor([1.0])}
    _make_concept_with_tensors(tmp_path, "default", "happy", {"gemma": profile})
    _make_concept_with_tensors(tmp_path, "a9lim", "archaic", {"qwen": profile})
    with pytest.raises(merge.MergeError, match="no shared models"):
        merge.shared_models([_cs("default/happy", 0.5), _cs("a9lim/archaic", 0.5)])


def test_merge_into_pack_single_model(monkeypatch, tmp_path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    p1 = {0: torch.tensor([1.0, 0.0])}
    p2 = {0: torch.tensor([0.0, 2.0])}
    _make_concept_with_tensors(tmp_path, "default", "happy", {"gemma": p1})
    _make_concept_with_tensors(tmp_path, "a9lim", "archaic", {"gemma": p2})
    dst = merge.merge_into_pack(
        "bard",
        components=[_cs("default/happy", 0.5), _cs("a9lim/archaic", 0.25)],
        model=None,
        force=False,
    )
    assert dst == tmp_path / "vectors" / "local" / "bard"
    m = packs.PackMetadata.load(dst)
    assert m.name == "bard"
    assert m.tags == ["merge"]
    assert m.source == "local"
    assert (dst / "gemma.safetensors").is_file()
    sc = packs.Sidecar.load(dst / "gemma.json")
    assert sc.method == "merge"
    assert set(sc.components.keys()) == {"default/happy", "a9lim/archaic"}
    assert sc.components["default/happy"]["alpha"] == 0.5


def test_merge_into_pack_conflict(monkeypatch, tmp_path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    p = {0: torch.tensor([1.0])}
    _make_concept_with_tensors(tmp_path, "default", "happy", {"gemma": p})
    _make_concept_with_tensors(tmp_path, "a9lim", "archaic", {"gemma": p})
    merge.merge_into_pack("bard",
                          [_cs("default/happy", 0.5), _cs("a9lim/archaic", 0.5)],
                          model=None, force=False)
    with pytest.raises(merge.MergeError, match="exists"):
        merge.merge_into_pack("bard",
                              [_cs("default/happy", 0.5), _cs("a9lim/archaic", 0.5)],
                              model=None, force=False)


def test_merge_into_pack_explicit_model(monkeypatch, tmp_path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    p = {0: torch.tensor([1.0])}
    _make_concept_with_tensors(tmp_path, "default", "happy",
                                {"google__gemma-2-2b-it": p, "qwen": p})
    _make_concept_with_tensors(tmp_path, "a9lim", "archaic",
                                {"google__gemma-2-2b-it": p, "qwen": p})
    dst = merge.merge_into_pack(
        "bard",
        [_cs("default/happy", 0.5), _cs("a9lim/archaic", 0.5)],
        model="google/gemma-2-2b-it",
        force=False,
    )
    # With explicit model, only that model's tensor is written. The model
    # id is flattened via safe_model_id.
    assert (dst / "google__gemma-2-2b-it.safetensors").is_file()
    assert not (dst / "qwen.safetensors").is_file()


def test_project_away_orthogonality():
    """After projection, result should be orthogonal to b (dot product ≈ 0)."""
    # a has a component in the direction of b and a component perpendicular.
    b = {0: torch.tensor([1.0, 0.0, 0.0]), 1: torch.tensor([0.0, 1.0, 0.0])}
    # a[0] = [1, 2, 0] — has [1,0,0] component along b[0] and [0,2,0] perp.
    # a[1] = [3, 1, 5] — has [0,1,0] component along b[1] and [3,0,5] perp.
    a = {0: torch.tensor([1.0, 2.0, 0.0]), 1: torch.tensor([3.0, 1.0, 5.0])}
    result = merge.project_away(a, b)
    # Layer 0: result should be orthogonal to b[0]=[1,0,0], i.e. x-component=0
    dot0 = torch.dot(result[0].float(), b[0].float()).item()
    assert abs(dot0) < 1e-6, f"Layer 0 not orthogonal: dot={dot0}"
    # Layer 1: result should be orthogonal to b[1]=[0,1,0], i.e. y-component=0
    dot1 = torch.dot(result[1].float(), b[1].float()).item()
    assert abs(dot1) < 1e-6, f"Layer 1 not orthogonal: dot={dot1}"
    # Perpendicular components preserved
    assert torch.allclose(result[0].float(), torch.tensor([0.0, 2.0, 0.0]), atol=1e-6)
    assert torch.allclose(result[1].float(), torch.tensor([3.0, 0.0, 5.0]), atol=1e-6)


def test_project_away_near_zero_b_skipped():
    """Near-zero b direction should be skipped, returning a unchanged for that layer."""
    a = {0: torch.tensor([1.0, 2.0]), 1: torch.tensor([3.0, 4.0])}
    b = {0: torch.tensor([0.0, 0.0]), 1: torch.tensor([1.0, 0.0])}
    result = merge.project_away(a, b)
    # Layer 0: b is near-zero, a[0] copied unchanged
    assert torch.allclose(result[0], a[0])
    # Layer 1: projected normally
    dot = torch.dot(result[1].float(), b[1].float()).item()
    assert abs(dot) < 1e-6


def test_project_away_layer_in_a_not_b():
    """Layers in a but not in b are copied unchanged."""
    a = {0: torch.tensor([1.0, 2.0]), 1: torch.tensor([3.0, 4.0])}
    b = {1: torch.tensor([1.0, 0.0])}
    result = merge.project_away(a, b)
    assert torch.allclose(result[0], a[0])  # layer 0: no b, unchanged
    dot = torch.dot(result[1].float(), b[1].float()).item()
    assert abs(dot) < 1e-6


def test_linear_sum_single_component():
    """Single-component linear_sum is now valid."""
    a = {0: torch.tensor([2.0, 3.0])}
    out = merge.linear_sum([(a, 0.5)])
    assert torch.allclose(out[0], torch.tensor([1.0, 1.5]))


def test_merge_into_pack_with_projection(monkeypatch, tmp_path):
    """merge_into_pack applies projection when project_away is set."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    # a = [1, 0]: direction along x
    # b = [1, 0]: same direction — projecting b out of a yields [0, 0]
    p_a = {0: torch.tensor([1.0, 0.0])}
    p_b = {0: torch.tensor([1.0, 0.0])}
    _make_concept_with_tensors(tmp_path, "default", "a_vec", {"gemma": p_a})
    _make_concept_with_tensors(tmp_path, "default", "b_vec", {"gemma": p_b})
    dst = merge.merge_into_pack(
        "projected",
        components=[_cs("default/a_vec", 1.0, "default/b_vec")],
        model=None,
        force=False,
    )
    assert (dst / "gemma.safetensors").is_file()
    from saklas.core.vectors import load_profile as _lp
    result, _ = _lp(str(dst / "gemma.safetensors"))
    # [1,0] - proj([1,0],[1,0]) * [1,0] = [1,0] - 1*[1,0] = [0,0]
    assert torch.allclose(result[0].float(), torch.zeros(2), atol=1e-6)


def test_merge_components_stale():
    comp = {"default/happy": {"alpha": 0.5, "tensor_sha256": "old"}}
    stale = packs.merge_components_stale(comp, {"default/happy": "new"})
    assert stale == ["default/happy"]
    stale = packs.merge_components_stale(comp, {"default/happy": "old"})
    assert stale == []
    # Missing components count as stale.
    stale = packs.merge_components_stale(comp, {})
    assert stale == ["default/happy"]


def test_merge_components_status():
    comp = {
        "default/happy": {"alpha": 0.5, "tensor_sha256": "old"},
        "default/sad": {"alpha": 0.5, "tensor_sha256": "old"},
        "default/angry": {"alpha": 0.5, "tensor_sha256": "old"},
    }
    current = {"default/happy": "old", "default/sad": "new"}
    status = packs.merge_components_status(comp, current)
    assert status == {
        "default/happy": "ok",
        "default/sad": "mismatch",
        "default/angry": "missing",
    }
