"""On-disk format tests for saklas.io.manifolds — CPU-only, no model."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from saklas.io.manifolds import (
    ManifoldFolder,
    ManifoldFormatError,
    ManifoldSidecar,
    hash_manifold_files,
)
from saklas.io.packs import PACK_FORMAT_VERSION


def _author_manifold(
    root: Path,
    *,
    name: str = "mood",
    cyclic: bool = True,
    labels: list[str] | None = None,
    files: dict | None = None,
) -> Path:
    """Hand-author a manifold folder; return its path."""
    if labels is None:
        labels = ["calm", "uneasy", "afraid", "frantic"]
    folder = root / name
    (folder / "nodes").mkdir(parents=True)
    for idx, label in enumerate(labels):
        statements = [f"{label} statement {i}" for i in range(3)]
        (folder / "nodes" / f"{idx:02d}_{label}.json").write_text(
            json.dumps(statements)
        )
    meta = {
        "format_version": PACK_FORMAT_VERSION,
        "name": name,
        "description": "a mood manifold",
        "cyclic": cyclic,
        "nodes": labels,
        "files": files if files is not None else {},
    }
    (folder / "manifold.json").write_text(json.dumps(meta))
    return folder


def test_load_minimal_manifold(tmp_path):
    folder = _author_manifold(tmp_path)
    mf = ManifoldFolder.load(folder)
    assert mf.name == "mood"
    assert mf.cyclic is True
    assert mf.node_labels == ["calm", "uneasy", "afraid", "frantic"]
    assert mf.description == "a mood manifold"


def test_node_groups(tmp_path):
    folder = _author_manifold(tmp_path)
    mf = ManifoldFolder.load(folder)
    groups = mf.node_groups()
    assert [label for label, _ in groups] == mf.node_labels
    label, statements = groups[0]
    assert label == "calm"
    assert statements == ["calm statement 0", "calm statement 1",
                          "calm statement 2"]


def test_missing_manifold_json_raises(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(ManifoldFormatError):
        ManifoldFolder.load(tmp_path / "empty")


def test_stale_format_version_raises(tmp_path):
    folder = _author_manifold(tmp_path)
    meta = json.loads((folder / "manifold.json").read_text())
    meta["format_version"] = 1
    (folder / "manifold.json").write_text(json.dumps(meta))
    with pytest.raises(ManifoldFormatError):
        ManifoldFolder.load(folder)


def test_too_few_nodes_raises(tmp_path):
    with pytest.raises(ManifoldFormatError):
        ManifoldFolder.load(
            _author_manifold(tmp_path, labels=["calm", "afraid"])
        )


def test_bad_node_label_raises(tmp_path):
    folder = tmp_path / "bad"
    (folder / "nodes").mkdir(parents=True)
    meta = {
        "format_version": PACK_FORMAT_VERSION,
        "name": "bad",
        "cyclic": False,
        "nodes": ["calm", "Uneasy", "afraid"],  # uppercase invalid
        "files": {},
    }
    (folder / "manifold.json").write_text(json.dumps(meta))
    with pytest.raises(ManifoldFormatError):
        ManifoldFolder.load(folder)


def test_duplicate_labels_raises(tmp_path):
    with pytest.raises(ManifoldFormatError):
        ManifoldFolder.load(
            _author_manifold(tmp_path, labels=["calm", "calm", "afraid"])
        )


def test_missing_node_file_raises(tmp_path):
    folder = _author_manifold(tmp_path)
    (folder / "nodes" / "00_calm.json").unlink()
    with pytest.raises(ManifoldFormatError):
        ManifoldFolder.load(folder)


def _add_dummy_tensor(folder: Path) -> None:
    """Write a placeholder fitted tensor + sidecar (no model needed)."""
    (folder / "stub-model.safetensors").write_bytes(b"placeholder-tensor")
    (folder / "stub-model.json").write_text(json.dumps({
        "method": "manifold_pca", "saklas_version": "0",
        "cyclic": True, "node_count": 4, "node_labels": [],
    }))


def test_integrity_check_catches_tampering(tmp_path):
    folder = _author_manifold(tmp_path)
    _add_dummy_tensor(folder)
    files = hash_manifold_files(folder)
    meta = json.loads((folder / "manifold.json").read_text())
    meta["files"] = files
    (folder / "manifold.json").write_text(json.dumps(meta))
    # Untampered: loads fine.
    ManifoldFolder.load(folder)
    # Tamper a fitted tensor.
    (folder / "stub-model.safetensors").write_bytes(b"tampered")
    with pytest.raises(ManifoldFormatError):
        ManifoldFolder.load(folder)


def test_node_corpus_edits_do_not_trip_integrity(tmp_path):
    # The node corpus is user-editable — editing it must not fail load
    # (it is the re-fit trigger, tracked via nodes_sha256).
    folder = _author_manifold(tmp_path)
    _add_dummy_tensor(folder)
    files = hash_manifold_files(folder)
    meta = json.loads((folder / "manifold.json").read_text())
    meta["files"] = files
    (folder / "manifold.json").write_text(json.dumps(meta))
    (folder / "nodes" / "00_calm.json").write_text(json.dumps(["edited"]))
    # Loads fine — node files are not in the integrity manifest.
    ManifoldFolder.load(folder)


def test_nodes_sha256_stable_and_sensitive(tmp_path):
    folder = _author_manifold(tmp_path)
    mf = ManifoldFolder.load(folder)
    h1 = mf.nodes_sha256()
    assert h1 == ManifoldFolder.load(folder).nodes_sha256()
    (folder / "nodes" / "00_calm.json").write_text(json.dumps(["changed"]))
    h2 = ManifoldFolder.load(folder).nodes_sha256()
    assert h1 != h2


def test_write_metadata_populates_files(tmp_path):
    folder = _author_manifold(tmp_path)
    _add_dummy_tensor(folder)
    mf = ManifoldFolder.load(folder)
    assert mf.files == {}
    mf.write_metadata()
    assert "stub-model.safetensors" in mf.files
    assert "stub-model.json" in mf.files
    # The node corpus is not hashed into the manifest.
    assert not any(k.startswith("nodes/") for k in mf.files)
    # Reload verifies the freshly written manifest.
    reloaded = ManifoldFolder.load(folder)
    assert reloaded.files == mf.files


def test_manifold_sidecar_load(tmp_path):
    path = tmp_path / "m.json"
    path.write_text(json.dumps({
        "method": "manifold_sae",
        "saklas_version": "3.1.0",
        "cyclic": True,
        "node_count": 4,
        "node_labels": ["a", "b", "c", "d"],
        "feature_space": "sae-gemma",
        "nodes_sha256": "deadbeef",
        "sae_release": "gemma",
    }))
    sc = ManifoldSidecar.load(path)
    assert sc.method == "manifold_sae"
    assert sc.cyclic is True
    assert sc.node_count == 4
    assert sc.feature_space == "sae-gemma"
    assert sc.nodes_sha256 == "deadbeef"
    assert sc.sae_release == "gemma"
