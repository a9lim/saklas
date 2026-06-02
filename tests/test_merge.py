"""vector merge — expression grammar + baked-manifold writer + projection math."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from saklas.io import merge, packs
from saklas.io.manifolds import ManifoldFolder, ManifoldSidecar
from saklas.core.manifold import load_manifold
from saklas.core.vectors import folded_vector_directions, save_profile


# --------------------------------------------------------- expr parsing ---

def test_parse_expr_two_components(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
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
        merge.merge_into_manifold("x", "0.5 a", model=None)


def test_trigger_rejected():
    """Merge expressions don't accept trigger annotations."""
    with pytest.raises(merge.MergeError, match="trigger"):
        merge.merge_into_manifold(
            "x", "0.5 default/happy@after", model=None,
        )


def test_aligned_operator_rejected():
    """``~`` (component-aligned with onto) isn't meaningful at merge time —
    require ``|`` (project-away).  Pre-v2.1 merge accepted ``~`` and
    treated it as project-away, which silently inverted the meaning vs
    the unified grammar (``~`` keeps aligned, ``|`` projects away).
    """
    with pytest.raises(merge.MergeError, match=r"\|"):
        merge.merge_into_manifold(
            "x", "0.5 default/happy~default/sad", model=None,
        )


def test_empty_expression_rejected():
    with pytest.raises(merge.MergeError):
        merge.merge_into_manifold("x", "", model=None)


def test_invalid_syntax_rejected():
    with pytest.raises(merge.MergeError):
        merge.merge_into_manifold("x", "0.5 default/happy +", model=None)


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

def _make_concept_with_tensors(
    tmp_path: Path,
    ns: str,
    name: str,
    model_tensors: dict[str, Any],
) -> Path:
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


def test_shared_models_intersection(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    profile = {0: torch.tensor([1.0])}
    _make_concept_with_tensors(tmp_path, "default", "happy",
                                {"gemma": profile, "qwen": profile})
    _make_concept_with_tensors(tmp_path, "a9lim", "archaic",
                                {"gemma": profile})
    shared = merge.shared_models("0.5 default/happy + 0.5 a9lim/archaic")
    assert shared == ["gemma"]


def test_shared_models_empty_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    profile = {0: torch.tensor([1.0])}
    _make_concept_with_tensors(tmp_path, "default", "happy", {"gemma": profile})
    _make_concept_with_tensors(tmp_path, "a9lim", "archaic", {"qwen": profile})
    with pytest.raises(merge.MergeError, match="no shared models"):
        merge.shared_models("0.5 default/happy + 0.5 a9lim/archaic")


def test_merge_into_manifold_single_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    p1 = {0: torch.tensor([1.0, 0.0])}
    p2 = {0: torch.tensor([0.0, 2.0])}
    _make_concept_with_tensors(tmp_path, "default", "happy", {"gemma": p1})
    _make_concept_with_tensors(tmp_path, "a9lim", "archaic", {"gemma": p2})
    dst = merge.merge_into_manifold(
        "bard",
        "0.5 default/happy + 0.25 a9lim/archaic",
        model=None,
        force=False,
    )
    # The merge now lands a corpus-less ``fit_mode="baked"`` manifold.
    assert dst == tmp_path / "manifolds" / "local" / "bard"
    mf = ManifoldFolder.load(dst)
    assert mf.name == "bard"
    assert mf.fit_mode == "baked"
    assert mf.tags == ["merge"]
    assert mf.source == "local"
    assert (dst / "gemma.safetensors").is_file()
    sc = ManifoldSidecar.load(dst / "gemma.json")
    assert sc.method == "merge"
    assert sc.fit_mode == "baked"
    # merge always records component provenance; assert guards the Optional
    assert sc.components is not None
    assert set(sc.components.keys()) == {"default/happy", "a9lim/archaic"}
    assert sc.components["default/happy"]["alpha"] == 0.5
    # The baked tensor folds back to the linear combination:
    # 0.5·[1,0] + 0.25·[0,2] = [0.5, 0.5].
    folded = folded_vector_directions(load_manifold(dst / "gemma.safetensors"))
    assert torch.allclose(folded[0].float(), torch.tensor([0.5, 0.5]), atol=1e-5)


def test_merge_into_manifold_conflict(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    p = {0: torch.tensor([1.0])}
    _make_concept_with_tensors(tmp_path, "default", "happy", {"gemma": p})
    _make_concept_with_tensors(tmp_path, "a9lim", "archaic", {"gemma": p})
    merge.merge_into_manifold(
        "bard", "0.5 default/happy + 0.5 a9lim/archaic",
        model=None, force=False,
    )
    with pytest.raises(merge.MergeError, match="exists"):
        merge.merge_into_manifold(
            "bard", "0.5 default/happy + 0.5 a9lim/archaic",
            model=None, force=False,
        )


def test_merge_into_manifold_explicit_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    p = {0: torch.tensor([1.0])}
    _make_concept_with_tensors(tmp_path, "default", "happy",
                                {"google__gemma-2-2b-it": p, "qwen": p})
    _make_concept_with_tensors(tmp_path, "a9lim", "archaic",
                                {"google__gemma-2-2b-it": p, "qwen": p})
    dst = merge.merge_into_manifold(
        "bard",
        "0.5 default/happy + 0.5 a9lim/archaic",
        model="google/gemma-2-2b-it",
        force=False,
    )
    assert (dst / "google__gemma-2-2b-it.safetensors").is_file()
    assert not (dst / "qwen.safetensors").is_file()


def test_merge_into_manifold_with_projection(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """merge applies project-away when the ``|`` operator is used.

    v2.1 fix-up: pre-v2.1 merge accepted ``~`` for project-away,
    inverting the unified-grammar semantics (``~`` keeps aligned,
    ``|`` projects away).  Now ``|`` is the canonical spelling.
    """
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    # a = [1, 2]; b = [1, 0]: projecting b out of a removes the x-component,
    # leaving [0, 2] — orthogonal to b, the meaningful projection result.
    p_a = {0: torch.tensor([1.0, 2.0])}
    p_b = {0: torch.tensor([1.0, 0.0])}
    _make_concept_with_tensors(tmp_path, "default", "a_vec", {"gemma": p_a})
    _make_concept_with_tensors(tmp_path, "default", "b_vec", {"gemma": p_b})
    dst = merge.merge_into_manifold(
        "projected",
        "1.0 default/a_vec|default/b_vec",
        model=None,
        force=False,
    )
    assert (dst / "gemma.safetensors").is_file()
    # The baked tensor folds back to the projected direction [0, 2]:
    # orthogonal to b, magnitude preserved on the kept axis.
    folded = folded_vector_directions(load_manifold(dst / "gemma.safetensors"))
    assert torch.allclose(folded[0].float(), torch.tensor([0.0, 2.0]), atol=1e-5)
    assert abs(torch.dot(folded[0].float(), p_b[0])) < 1e-5


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
