"""vector merge — expression grammar + pack writer + projection math."""
import pytest
import torch

from saklas.io import merge, packs
from saklas.core.vectors import save_profile


# --------------------------------------------------------- expr parsing ---

def test_parse_expr_two_components(monkeypatch, tmp_path):
    """Parser rejects bare (non-namespaced) components; the happy path
    below uses a namespace-qualified expression, which shared_models
    round-trips through the parser."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    profile = {0: torch.tensor([1.0])}
    _make_concept_with_tensors(tmp_path, "default", "happy", {"gemma": profile})
    _make_concept_with_tensors(tmp_path, "a9lim", "archaic", {"gemma": profile})
    shared = merge.shared_models("0.3 default/happy + 0.4 a9lim/archaic")
    assert shared == ["gemma"]


def test_bare_component_rejected():
    """Components without a namespace prefix are rejected."""
    with pytest.raises(merge.MergeError, match="namespace"):
        merge.merge_into_pack("x", "0.5 a", model=None)


def test_trigger_rejected():
    """Merge expressions don't accept trigger annotations."""
    with pytest.raises(merge.MergeError, match="trigger"):
        merge.merge_into_pack(
            "x", "0.5 default/happy@after", model=None,
        )


def test_ortho_operator_rejected():
    """| (orthogonal) isn't meaningful at extract/merge time — require ~."""
    with pytest.raises(merge.MergeError, match="~"):
        merge.merge_into_pack(
            "x", "0.5 default/happy|default/sad", model=None,
        )


def test_empty_expression_rejected():
    with pytest.raises(merge.MergeError):
        merge.merge_into_pack("x", "", model=None)


def test_invalid_syntax_rejected():
    with pytest.raises(merge.MergeError):
        merge.merge_into_pack("x", "0.5 default/happy +", model=None)


# ------------------------------------------------------- linear_sum ---

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


def test_linear_sum_single_component():
    a = {0: torch.tensor([2.0, 3.0])}
    out = merge.linear_sum([(a, 0.5)])
    assert torch.allclose(out[0], torch.tensor([1.0, 1.5]))


# ---------------------------------------------- pack-writing end-to-end ---

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
    shared = merge.shared_models("0.5 default/happy + 0.5 a9lim/archaic")
    assert shared == ["gemma"]


def test_shared_models_empty_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    profile = {0: torch.tensor([1.0])}
    _make_concept_with_tensors(tmp_path, "default", "happy", {"gemma": profile})
    _make_concept_with_tensors(tmp_path, "a9lim", "archaic", {"qwen": profile})
    with pytest.raises(merge.MergeError, match="no shared models"):
        merge.shared_models("0.5 default/happy + 0.5 a9lim/archaic")


def test_merge_into_pack_single_model(monkeypatch, tmp_path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    p1 = {0: torch.tensor([1.0, 0.0])}
    p2 = {0: torch.tensor([0.0, 2.0])}
    _make_concept_with_tensors(tmp_path, "default", "happy", {"gemma": p1})
    _make_concept_with_tensors(tmp_path, "a9lim", "archaic", {"gemma": p2})
    dst = merge.merge_into_pack(
        "bard",
        "0.5 default/happy + 0.25 a9lim/archaic",
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
    merge.merge_into_pack(
        "bard", "0.5 default/happy + 0.5 a9lim/archaic",
        model=None, force=False,
    )
    with pytest.raises(merge.MergeError, match="exists"):
        merge.merge_into_pack(
            "bard", "0.5 default/happy + 0.5 a9lim/archaic",
            model=None, force=False,
        )


def test_merge_into_pack_explicit_model(monkeypatch, tmp_path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    p = {0: torch.tensor([1.0])}
    _make_concept_with_tensors(tmp_path, "default", "happy",
                                {"google__gemma-2-2b-it": p, "qwen": p})
    _make_concept_with_tensors(tmp_path, "a9lim", "archaic",
                                {"google__gemma-2-2b-it": p, "qwen": p})
    dst = merge.merge_into_pack(
        "bard",
        "0.5 default/happy + 0.5 a9lim/archaic",
        model="google/gemma-2-2b-it",
        force=False,
    )
    assert (dst / "google__gemma-2-2b-it.safetensors").is_file()
    assert not (dst / "qwen.safetensors").is_file()


def test_merge_into_pack_with_projection(monkeypatch, tmp_path):
    """merge_into_pack applies projection when ~ operator is used."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    # a = [1, 0]: direction along x
    # b = [1, 0]: same direction — projecting b out of a yields [0, 0]
    p_a = {0: torch.tensor([1.0, 0.0])}
    p_b = {0: torch.tensor([1.0, 0.0])}
    _make_concept_with_tensors(tmp_path, "default", "a_vec", {"gemma": p_a})
    _make_concept_with_tensors(tmp_path, "default", "b_vec", {"gemma": p_b})
    dst = merge.merge_into_pack(
        "projected",
        "1.0 default/a_vec~default/b_vec",
        model=None,
        force=False,
    )
    assert (dst / "gemma.safetensors").is_file()
    from saklas.core.vectors import load_profile as _lp
    result, _ = _lp(str(dst / "gemma.safetensors"))
    assert torch.allclose(result[0].float(), torch.zeros(2), atol=1e-6)


# ------------------------------------------------- project_away math ---

def test_project_away_orthogonality():
    b = {0: torch.tensor([1.0, 0.0, 0.0]), 1: torch.tensor([0.0, 1.0, 0.0])}
    a = {0: torch.tensor([1.0, 2.0, 0.0]), 1: torch.tensor([3.0, 1.0, 5.0])}
    result = merge.project_away(a, b)
    dot0 = torch.dot(result[0].float(), b[0].float()).item()
    assert abs(dot0) < 1e-6
    dot1 = torch.dot(result[1].float(), b[1].float()).item()
    assert abs(dot1) < 1e-6
    assert torch.allclose(result[0].float(), torch.tensor([0.0, 2.0, 0.0]), atol=1e-6)
    assert torch.allclose(result[1].float(), torch.tensor([3.0, 0.0, 5.0]), atol=1e-6)


def test_project_away_near_zero_b_skipped():
    a = {0: torch.tensor([1.0, 2.0]), 1: torch.tensor([3.0, 4.0])}
    b = {0: torch.tensor([0.0, 0.0]), 1: torch.tensor([1.0, 0.0])}
    result = merge.project_away(a, b)
    assert torch.allclose(result[0], a[0])
    dot = torch.dot(result[1].float(), b[1].float()).item()
    assert abs(dot) < 1e-6


def test_project_away_layer_in_a_not_b():
    a = {0: torch.tensor([1.0, 2.0]), 1: torch.tensor([3.0, 4.0])}
    b = {1: torch.tensor([1.0, 0.0])}
    result = merge.project_away(a, b)
    assert torch.allclose(result[0], a[0])
    dot = torch.dot(result[1].float(), b[1].float()).item()
    assert abs(dot) < 1e-6


# -------------------------------------------------- packs helpers ---

def test_merge_components_stale():
    comp = {"default/happy": {"alpha": 0.5, "tensor_sha256": "old"}}
    stale = packs.merge_components_stale(comp, {"default/happy": "new"})
    assert stale == ["default/happy"]
    stale = packs.merge_components_stale(comp, {"default/happy": "old"})
    assert stale == []
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
