import pytest
import torch

from saklas import merge, packs
from saklas.vectors import save_profile


def test_parse_components_two():
    out = merge.parse_components("default/happy:0.3,a9lim/archaic:0.4")
    assert out == [("default/happy", 0.3), ("a9lim/archaic", 0.4)]


def test_parse_components_three():
    out = merge.parse_components("a:0.1,b:0.2,c:0.3")
    assert [n for n, _ in out] == ["a", "b", "c"]
    assert [a for _, a in out] == [0.1, 0.2, 0.3]


def test_parse_components_requires_alpha():
    with pytest.raises(merge.MergeError, match="alpha"):
        merge.parse_components("a,b:0.2")


def test_parse_components_requires_two():
    with pytest.raises(merge.MergeError, match="at least two"):
        merge.parse_components("a:0.5")


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
    shared = merge.shared_models([("default/happy", 0.5), ("a9lim/archaic", 0.5)])
    assert shared == ["gemma"]


def test_shared_models_empty_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    profile = {0: torch.tensor([1.0])}
    _make_concept_with_tensors(tmp_path, "default", "happy", {"gemma": profile})
    _make_concept_with_tensors(tmp_path, "a9lim", "archaic", {"qwen": profile})
    with pytest.raises(merge.MergeError, match="no shared models"):
        merge.shared_models([("default/happy", 0.5), ("a9lim/archaic", 0.5)])


def test_merge_into_pack_single_model(monkeypatch, tmp_path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    p1 = {0: torch.tensor([1.0, 0.0])}
    p2 = {0: torch.tensor([0.0, 2.0])}
    _make_concept_with_tensors(tmp_path, "default", "happy", {"gemma": p1})
    _make_concept_with_tensors(tmp_path, "a9lim", "archaic", {"gemma": p2})
    dst = merge.merge_into_pack(
        "bard",
        components=[("default/happy", 0.5), ("a9lim/archaic", 0.25)],
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
                          [("default/happy", 0.5), ("a9lim/archaic", 0.5)],
                          model=None, force=False)
    with pytest.raises(merge.MergeError, match="exists"):
        merge.merge_into_pack("bard",
                              [("default/happy", 0.5), ("a9lim/archaic", 0.5)],
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
        [("default/happy", 0.5), ("a9lim/archaic", 0.5)],
        model="google/gemma-2-2b-it",
        force=False,
    )
    # With explicit model, only that model's tensor is written. The model
    # id is flattened via safe_model_id.
    assert (dst / "google__gemma-2-2b-it.safetensors").is_file()
    assert not (dst / "qwen.safetensors").is_file()


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
