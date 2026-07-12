"""Offline additive merge grammar and baked-manifold writer."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from saklas.io import merge
from saklas.io.manifolds import (
    ManifoldFolder, ManifoldSidecar,
    create_baked_manifold_folder, save_baked_manifold_tensor,
)
from saklas.core.manifold import load_manifold
from saklas.core.vectors import fold_directions_to_subspace, folded_vector_directions
from saklas.io.paths import encode_release_id, model_dir, safe_model_id, tensor_filename
from tests._whitener import isotropic_whitener


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
    assert shared == [safe_model_id("gemma")]


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


@pytest.mark.parametrize("operator", ["~", "|"])
def test_projection_operator_rejected(operator: str):
    """Offline bake cannot replace live Mahalanobis projection with Euclidean."""
    with pytest.raises(merge.MergeError, match="Mahalanobis"):
        merge.merge_into_manifold(
            "x", f"0.5 default/happy{operator}default/sad", model=None,
        )


@pytest.mark.parametrize(
    ("expression", "message"),
    [
        ("!default/happy", "ablation"),
        ("0.5 default/happy%label", "manifold-position"),
        ("0.5,0.2 default/happy", "one scalar"),
    ],
)
def test_dynamic_or_multicoeff_term_rejected(expression: str, message: str):
    with pytest.raises(merge.MergeError, match=message):
        merge.merge_into_manifold("x", expression, model=None)


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


def test_linear_sum_layer_union_matches_live_composition():
    a = {0: torch.tensor([1.0, 0.0]),
         1: torch.tensor([0.0, 1.0]),
         2: torch.tensor([1.0, 1.0])}
    b = {1: torch.tensor([0.0, 2.0]),
         2: torch.tensor([2.0, 2.0])}
    out = merge.linear_sum([(a, 1.0), (b, 1.0)])
    assert sorted(out.keys()) == [0, 1, 2]
    assert torch.allclose(out[0], a[0])


def test_linear_sum_disjoint_layers_uses_union():
    a = {0: torch.tensor([1.0])}
    b = {1: torch.tensor([2.0])}
    out = merge.linear_sum([(a, 1.0), (b, 1.0)])
    assert sorted(out) == [0, 1]
    assert torch.equal(out[0], a[0])
    assert torch.equal(out[1], b[1])


def test_linear_sum_strict_rejects_different_layer_coverage():
    a = {0: torch.tensor([1.0]), 1: torch.tensor([2.0])}
    b = {1: torch.tensor([3.0])}
    with pytest.raises(merge.MergeError, match="coverage differs"):
        merge.linear_sum([(a, 1.0), (b, 1.0)], strict=True)


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
    """Author a merge component as a fitted manifold (4.0).

    A merge component is resolved by folding a fitted 2-node ``pca`` manifold
    down to a single direction; the simplest fixture that round-trips a known
    profile is a corpus-less ``baked`` manifold built straight from the
    direction via :func:`fold_directions_to_subspace`.  One ``manifold.json``
    is shared across every model's per-model tensor, mirroring how
    :func:`merge.merge_into_manifold` writes multi-model output.
    """
    folder: Path | None = None
    for model_id, profile in model_tensors.items():
        _seed_neutral_cache(model_id, profile)
        means = {layer: torch.zeros_like(direction) for layer, direction in profile.items()}
        manifold = fold_directions_to_subspace(
            name, profile, means,
            whitener=isotropic_whitener(
                profile, int(next(iter(profile.values())).numel()),
            ),
            label="test",
        )
        if folder is None:
            folder, _mf = create_baked_manifold_folder(
                ns, name, "x", manifold, model_id, method="test",
                model_fingerprint=f"fp:{model_id}",
            )
        else:
            save_baked_manifold_tensor(
                folder, manifold, model_id, method="test",
                model_fingerprint=f"fp:{model_id}",
            )
    assert folder is not None
    ManifoldFolder.load(folder).write_metadata()
    return folder


def _seed_neutral_cache(model_id: str, profile: dict[int, torch.Tensor]) -> None:
    """Give offline bake the persisted metric its current contract requires."""
    import json
    from safetensors.torch import save_file
    from saklas.io.packs import hash_file

    md = model_dir(model_id)
    md.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    hashes: dict[str, str] = {}
    schema: dict[str, dict[str, Any]] = {}
    for layer, direction in profile.items():
        dim = int(direction.numel())
        g = torch.Generator().manual_seed(100 + layer)
        acts = torch.randn(64, dim, generator=g, dtype=torch.float32)
        path = md / f"neutral_activations.layer-{layer}.gen-test.safetensors"
        save_file({f"layer_{layer}": acts}, str(path))
        files[str(layer)] = path.name
        hashes[str(layer)] = hash_file(path)
        schema[str(layer)] = {"shape": list(acts.shape), "dtype": "torch.float32"}
    (md / "neutral_activations.json").write_text(json.dumps({
        "method": "neutral_activations", "format_version": 4,
        "capture_version": 1, "capture_sha256": "test-capture",
        "model_fingerprint": "test-fingerprint",
        "model_source_fingerprint": "test-source",
        "tensor_sha256": hashes, "tensor_files": files,
        "layers": list(profile), "tensor_schema": schema,
        "n_prompts": 64, "n_layers": len(profile),
    }))


def test_shared_models_intersection(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    profile = {0: torch.tensor([1.0])}
    _make_concept_with_tensors(tmp_path, "default", "happy",
                                {"gemma": profile, "qwen": profile})
    _make_concept_with_tensors(tmp_path, "a9lim", "archaic",
                                {"gemma": profile})
    shared = merge.shared_models("0.5 default/happy + 0.5 a9lim/archaic")
    assert shared == [safe_model_id("gemma")]


def test_shared_models_empty_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    profile = {0: torch.tensor([1.0])}
    _make_concept_with_tensors(tmp_path, "default", "happy", {"gemma": profile})
    _make_concept_with_tensors(tmp_path, "a9lim", "archaic", {"qwen": profile})
    with pytest.raises(merge.MergeError, match="no shared models"):
        merge.shared_models("0.5 default/happy + 0.5 a9lim/archaic")


def test_merge_resolves_each_component_model_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    profile = {0: torch.tensor([1.0])}
    _make_concept_with_tensors(
        tmp_path, "default", "happy", {"gemma": profile},
    )
    _make_concept_with_tensors(
        tmp_path, "a9lim", "archaic", {"gemma": profile},
    )
    real_resolve = merge._resolve_component
    calls: list[tuple[str, str, str]] = []

    def counted(ns: str, name: str, sid: str, variant: Any, coord: str):
        calls.append((ns, name, sid))
        return real_resolve(ns, name, sid, variant, coord)

    monkeypatch.setattr(merge, "_resolve_component", counted)
    merge.merge_into_manifold(
        "bard", "default/happy + a9lim/archaic", model=None,
    )

    assert calls == [
        ("default", "happy", safe_model_id("gemma")),
        ("a9lim", "archaic", safe_model_id("gemma")),
    ]


def test_role_variant_uses_and_validates_canonical_tensor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    folder = _make_concept_with_tensors(
        tmp_path, "default", "happy", {"gemma": {0: torch.tensor([1.0])}},
    )
    tensor_path = folder / tensor_filename("gemma")
    sidecar_path = tensor_path.with_suffix(".json")
    payload = __import__("json").loads(sidecar_path.read_text())
    payload["node_roles"] = ["pirate"] * int(payload["node_count"])
    sidecar_path.write_text(__import__("json").dumps(payload))
    ManifoldFolder.load(folder, verify_manifest=False).update_file_hashes(
        tensor_path, sidecar_path,
    )

    assert merge.shared_models("default/happy:role-pirate") == [safe_model_id("gemma")]
    with pytest.raises(merge.MergeError, match="no shared models"):
        merge.shared_models("default/happy:raw")
    dst = merge.merge_into_manifold(
        "pirate_happy", "default/happy:role-pirate", model=None,
    )
    assert (dst / tensor_filename("gemma")).is_file()


def test_transfer_variant_routes_concrete_tensor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    folder = _make_concept_with_tensors(
        tmp_path, "default", "happy", {"src": {0: torch.tensor([1.0])}},
    )
    target_tensor = folder / tensor_filename("target", transferred_from="src")
    target_sidecar = target_tensor.with_suffix(".json")
    source_tensor = folder / tensor_filename("src")
    target_tensor.write_bytes(source_tensor.read_bytes())
    payload = __import__("json").loads(source_tensor.with_suffix(".json").read_text())
    payload["method"] = "manifold_procrustes_transfer"
    payload["source_model_id"] = "src"
    payload["source_model_fingerprint"] = "fp:src"
    payload["model_fingerprint"] = "fp:target"
    target_sidecar.write_text(__import__("json").dumps(payload))
    ManifoldFolder.load(folder, verify_manifest=False).update_file_hashes(
        target_tensor, target_sidecar,
    )
    _seed_neutral_cache("target", {0: torch.tensor([1.0])})

    variant = f"from-{encode_release_id('src')}"
    assert merge.shared_models(f"default/happy:{variant}") == [safe_model_id("target")]
    dst = merge.merge_into_manifold(
        "transferred_happy", f"default/happy:{variant}", model=None,
    )
    assert (dst / tensor_filename("target")).is_file()


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
    tensor_path = dst / tensor_filename("gemma")
    assert tensor_path.is_file()
    sc = ManifoldSidecar.load(tensor_path.with_suffix(".json"))
    assert sc.method == "merge"
    assert sc.fit_mode == "baked"
    # merge always records component provenance; assert guards the Optional
    assert sc.components is not None
    assert set(sc.components.keys()) == {"0:default/happy", "1:a9lim/archaic"}
    assert sc.components["0:default/happy"]["alpha"] == 0.5
    # The baked tensor folds back to the linear combination:
    # 0.5·[1,0] + 0.25·[0,2] = [0.5, 0.5].
    folded = folded_vector_directions(load_manifold(tensor_path))
    assert folded[0][0] > 0
    assert folded[0][0] == pytest.approx(float(folded[0][1]), abs=1e-5)


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


def test_merge_retry_repairs_later_unproven_baked_pair(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    profiles = {
        "gemma": {0: torch.tensor([1.0])},
        "qwen": {0: torch.tensor([2.0])},
    }
    _make_concept_with_tensors(tmp_path, "default", "happy", profiles)
    _make_concept_with_tensors(tmp_path, "a9lim", "archaic", profiles)
    real_update = ManifoldFolder.update_file_hashes
    calls = 0

    def fail_second(self: ManifoldFolder, *paths: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected baked manifest failure")
        real_update(self, *paths)

    monkeypatch.setattr(ManifoldFolder, "update_file_hashes", fail_second)
    expression = "default/happy + a9lim/archaic"
    with pytest.raises(OSError, match="injected"):
        merge.merge_into_manifold("bard", expression, model=None)
    monkeypatch.setattr(ManifoldFolder, "update_file_hashes", real_update)

    dst = merge.merge_into_manifold("bard", expression, model=None)
    loaded = ManifoldFolder.load(dst)
    assert sorted(loaded.tensor_models()) == sorted(map(safe_model_id, ("gemma", "qwen")))


def test_merge_into_manifold_explicit_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    p = {0: torch.tensor([1.0])}
    _make_concept_with_tensors(tmp_path, "default", "happy",
                                {"google/gemma-2-2b-it": p, "qwen": p})
    _make_concept_with_tensors(tmp_path, "a9lim", "archaic",
                                {"google/gemma-2-2b-it": p, "qwen": p})
    dst = merge.merge_into_manifold(
        "bard",
        "0.5 default/happy + 0.5 a9lim/archaic",
        model="google/gemma-2-2b-it",
        force=False,
    )
    assert (dst / tensor_filename("google/gemma-2-2b-it")).is_file()
    assert not (dst / tensor_filename("qwen")).is_file()


def test_legacy_projected_bake_requires_rebake(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    folder = _make_concept_with_tensors(
        tmp_path, "local", "legacy", {"gemma": {0: torch.tensor([1.0, 2.0])}},
    )
    tensor_path = folder / tensor_filename("gemma")
    sidecar_path = tensor_path.with_suffix(".json")
    sidecar = sidecar_path.read_text()
    payload = __import__("json").loads(sidecar)
    payload["method"] = "merge"
    payload["components"] = {
        "default/a": {
            "alpha": 1.0,
            "project_away": "default/b",
            "tensor_sha256": "legacy",
        }
    }
    sidecar_path.write_text(__import__("json").dumps(payload))
    with pytest.raises(Exception, match="invalid component provenance"):
        ManifoldFolder.load(folder, verify_manifest=False)
