"""On-disk format tests for saklas.io.manifolds — CPU-only, no model."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from saklas.io.manifolds import (
    MANIFOLD_FORMAT_VERSION,
    ManifoldFolder,
    ManifoldFormatError,
    ManifoldSidecar,
    create_discover_manifold_folder,
    create_manifold_folder,
    hash_manifold_files,
    iter_manifold_folders,
    merge_discover_manifolds,
    min_nodes,
    update_manifold_folder,
)


def _box1d(periodic: bool, labels: list[str]) -> dict:
    """A 1-D box domain spec with evenly-spaced node coords."""
    k = len(labels)
    axis = (
        {"name": "t", "periodic": True, "period": 1.0}
        if periodic
        else {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}
    )
    if periodic:
        coords = [[i / k] for i in range(k)]
    else:
        coords = [[i / (k - 1) if k > 1 else 0.0] for i in range(k)]
    return {
        "domain": {"type": "box", "axes": [axis]},
        "nodes": [
            {"label": label, "coords": coords[i]}
            for i, label in enumerate(labels)
        ],
    }


def _author_manifold(
    root: Path,
    *,
    name: str = "mood",
    periodic: bool = True,
    labels: list[str] | None = None,
    files: dict | None = None,
    domain: dict | None = None,
    nodes: list[dict] | None = None,
) -> Path:
    """Hand-author a v3 manifold folder; return its path."""
    if labels is None:
        labels = ["calm", "uneasy", "afraid", "frantic"]
    folder = root / name
    (folder / "nodes").mkdir(parents=True)
    spec = _box1d(periodic, labels)
    if domain is not None:
        spec["domain"] = domain
    if nodes is not None:
        spec["nodes"] = nodes
    for idx, node in enumerate(spec["nodes"]):
        statements = [f"{node['label']} statement {i}" for i in range(3)]
        (folder / "nodes" / f"{idx:02d}_{node['label']}.json").write_text(
            json.dumps(statements)
        )
    meta = {
        "format_version": MANIFOLD_FORMAT_VERSION,
        "name": name,
        "description": "a mood manifold",
        "domain": spec["domain"],
        "nodes": spec["nodes"],
        "files": files if files is not None else {},
    }
    (folder / "manifold.json").write_text(json.dumps(meta))
    return folder


def test_min_nodes_per_dimension():
    assert min_nodes(1) == 3
    assert min_nodes(2) == 5
    assert min_nodes(3) == 7


def test_load_minimal_manifold(tmp_path):
    folder = _author_manifold(tmp_path)
    mf = ManifoldFolder.load(folder)
    assert mf.name == "mood"
    assert mf.domain["type"] == "box"
    assert mf.node_labels == ["calm", "uneasy", "afraid", "frantic"]
    assert len(mf.node_coords) == 4
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
    meta["format_version"] = 2
    (folder / "manifold.json").write_text(json.dumps(meta))
    with pytest.raises(ManifoldFormatError):
        ManifoldFolder.load(folder)


def test_domain_required(tmp_path):
    folder = _author_manifold(tmp_path)
    meta = json.loads((folder / "manifold.json").read_text())
    del meta["domain"]
    (folder / "manifold.json").write_text(json.dumps(meta))
    with pytest.raises(ManifoldFormatError):
        ManifoldFolder.load(folder)


def test_too_few_nodes_raises(tmp_path):
    with pytest.raises(ManifoldFormatError):
        ManifoldFolder.load(
            _author_manifold(tmp_path, labels=["calm", "afraid"])
        )


def test_too_few_nodes_for_2d(tmp_path):
    # A 2-D domain needs min_nodes(2) == 5; four nodes is too few.
    domain = {
        "type": "box",
        "axes": [
            {"name": "u", "periodic": False, "lo": 0.0, "hi": 1.0},
            {"name": "v", "periodic": False, "lo": 0.0, "hi": 1.0},
        ],
    }
    nodes = [
        {"label": f"n{i}", "coords": [x, y]}
        for i, (x, y) in enumerate([(0, 0), (1, 0), (0, 1), (1, 1)])
    ]
    with pytest.raises(ManifoldFormatError):
        ManifoldFolder.load(
            _author_manifold(tmp_path, domain=domain, nodes=nodes)
        )


def test_node_coords_dim_mismatch_raises(tmp_path):
    # 1-D domain but a node carrying two coordinates.
    nodes = [
        {"label": "a", "coords": [0.0, 0.0]},
        {"label": "b", "coords": [0.5]},
        {"label": "c", "coords": [1.0]},
    ]
    with pytest.raises(ManifoldFormatError):
        ManifoldFolder.load(_author_manifold(tmp_path, nodes=nodes))


def test_bad_node_label_raises(tmp_path):
    nodes = [
        {"label": "calm", "coords": [0.0]},
        {"label": "Uneasy", "coords": [0.5]},   # uppercase invalid
        {"label": "afraid", "coords": [1.0]},
    ]
    with pytest.raises(ManifoldFormatError):
        ManifoldFolder.load(_author_manifold(tmp_path, nodes=nodes))


def test_duplicate_labels_raises(tmp_path):
    nodes = [
        {"label": "calm", "coords": [0.0]},
        {"label": "calm", "coords": [0.5]},
        {"label": "afraid", "coords": [1.0]},
    ]
    with pytest.raises(ManifoldFormatError):
        ManifoldFolder.load(_author_manifold(tmp_path, nodes=nodes))


def test_poisedness_soft_warning(tmp_path):
    # Five nodes on a 2-D domain but all collinear — a soft warning, not
    # a hard error (the hard error is raised later at fit time).
    domain = {
        "type": "box",
        "axes": [
            {"name": "u", "periodic": False, "lo": 0.0, "hi": 1.0},
            {"name": "v", "periodic": False, "lo": 0.0, "hi": 1.0},
        ],
    }
    nodes = [
        {"label": f"n{i}", "coords": [t, t]}      # the diagonal line
        for i, t in enumerate([0.0, 0.25, 0.5, 0.75, 1.0])
    ]
    with pytest.warns(UserWarning, match="poised"):
        ManifoldFolder.load(
            _author_manifold(tmp_path, domain=domain, nodes=nodes)
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
        "domain": {"type": "box", "axes": [
            {"name": "t", "periodic": True, "period": 1.0}]},
        "node_count": 4, "node_labels": [],
    }))


def test_integrity_check_catches_tampering(tmp_path):
    folder = _author_manifold(tmp_path)
    _add_dummy_tensor(folder)
    files = hash_manifold_files(folder)
    meta = json.loads((folder / "manifold.json").read_text())
    meta["files"] = files
    (folder / "manifold.json").write_text(json.dumps(meta))
    ManifoldFolder.load(folder)
    (folder / "stub-model.safetensors").write_bytes(b"tampered")
    with pytest.raises(ManifoldFormatError):
        ManifoldFolder.load(folder)


def test_node_corpus_edits_do_not_trip_integrity(tmp_path):
    folder = _author_manifold(tmp_path)
    _add_dummy_tensor(folder)
    files = hash_manifold_files(folder)
    meta = json.loads((folder / "manifold.json").read_text())
    meta["files"] = files
    (folder / "manifold.json").write_text(json.dumps(meta))
    (folder / "nodes" / "00_calm.json").write_text(json.dumps(["edited"]))
    ManifoldFolder.load(folder)


def test_nodes_sha256_stable_and_sensitive(tmp_path):
    folder = _author_manifold(tmp_path)
    mf = ManifoldFolder.load(folder)
    h1 = mf.nodes_sha256()
    assert h1 == ManifoldFolder.load(folder).nodes_sha256()
    (folder / "nodes" / "00_calm.json").write_text(json.dumps(["changed"]))
    h2 = ManifoldFolder.load(folder).nodes_sha256()
    assert h1 != h2


def test_nodes_sha256_sensitive_to_coords(tmp_path):
    # Moving a node's authoring coordinate must invalidate the hash even
    # when the corpus is untouched — the fit depends on the geometry.
    folder = _author_manifold(tmp_path)
    h1 = ManifoldFolder.load(folder).nodes_sha256()
    meta = json.loads((folder / "manifold.json").read_text())
    meta["nodes"][1]["coords"] = [0.123]
    (folder / "manifold.json").write_text(json.dumps(meta))
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
    assert not any(k.startswith("nodes/") for k in mf.files)
    reloaded = ManifoldFolder.load(folder)
    assert reloaded.files == mf.files


def test_manifold_sidecar_load(tmp_path):
    path = tmp_path / "m.json"
    path.write_text(json.dumps({
        "method": "manifold_sae",
        "saklas_version": "3.1.0",
        "domain": {"type": "sphere", "dim": 2},
        "node_count": 4,
        "node_labels": ["a", "b", "c", "d"],
        "feature_space": "sae-gemma",
        "nodes_sha256": "deadbeef",
        "sae_release": "gemma",
    }))
    sc = ManifoldSidecar.load(path)
    assert sc.method == "manifold_sae"
    assert sc.domain == {"type": "sphere", "dim": 2}
    assert sc.node_count == 4
    assert sc.feature_space == "sae-gemma"
    assert sc.nodes_sha256 == "deadbeef"
    assert sc.sae_release == "gemma"


# --------------------------------------------------- authoring (create/update) ---

def _author_nodes(labels):
    """A well-spread node list for a 1-D box, statements inline."""
    out = []
    for i, label in enumerate(labels):
        out.append({
            "label": label,
            "coords": [i / (len(labels) - 1)],
            "statements": [f"{label} statement {j}" for j in range(3)],
        })
    return out


def test_create_manifold_folder_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    nodes = _author_nodes(["calm", "uneasy", "afraid"])
    folder, advisories = create_manifold_folder(
        "local", "mood", "a mood axis", domain, nodes,
    )
    assert folder.exists()
    mf = ManifoldFolder.load(folder)
    assert mf.name == "mood"
    assert mf.node_labels == ["calm", "uneasy", "afraid"]
    assert mf.files == {}
    assert dict(mf.node_groups())["calm"][0] == "calm statement 0"
    # A well-spread 1-D layout draws no poisedness advisory.
    assert advisories == []


def test_create_manifold_folder_conflict(tmp_path, monkeypatch):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    nodes = _author_nodes(["a", "b", "c"])
    create_manifold_folder("local", "dup", "", domain, nodes)
    with pytest.raises(FileExistsError):
        create_manifold_folder("local", "dup", "", domain, nodes)


def test_create_manifold_folder_too_few_nodes(tmp_path, monkeypatch):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    with pytest.raises(ManifoldFormatError):
        create_manifold_folder(
            "local", "thin", "", domain, _author_nodes(["a", "b"]),
        )


def test_create_manifold_folder_bad_coords_arity(tmp_path, monkeypatch):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    nodes = [
        {"label": "a", "coords": [0.0, 0.0], "statements": ["x"]},
        {"label": "b", "coords": [0.5], "statements": ["y"]},
        {"label": "c", "coords": [1.0], "statements": ["z"]},
    ]
    with pytest.raises(ManifoldFormatError):
        create_manifold_folder("local", "bad", "", domain, nodes)


def test_create_manifold_folder_empty_statements(tmp_path, monkeypatch):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    nodes = _author_nodes(["a", "b", "c"])
    nodes[1]["statements"] = []
    with pytest.raises(ManifoldFormatError):
        create_manifold_folder("local", "nostmt", "", domain, nodes)


def test_create_manifold_folder_bad_namespace(tmp_path, monkeypatch):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    with pytest.raises(ManifoldFormatError):
        create_manifold_folder(
            "Bad Namespace", "m", "", domain, _author_nodes(["a", "b", "c"]),
        )


def test_update_manifold_folder_statements(tmp_path, monkeypatch):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    nodes = _author_nodes(["a", "b", "c"])
    folder, _ = create_manifold_folder("local", "mood", "", domain, nodes)
    sha_before = ManifoldFolder.load(folder).nodes_sha256()

    nodes[0]["statements"].append("a fresh statement")
    update_manifold_folder(folder, description="edited", nodes=nodes)
    mf = ManifoldFolder.load(folder)
    assert mf.description == "edited"
    assert "a fresh statement" in dict(mf.node_groups())["a"]
    # Editing the corpus must invalidate the staleness key.
    assert mf.nodes_sha256() != sha_before


def test_update_manifold_folder_relabels_cleanly(tmp_path, monkeypatch):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    folder, _ = create_manifold_folder(
        "local", "mood", "", domain, _author_nodes(["a", "b", "c"]),
    )
    relabeled = _author_nodes(["x", "y", "z"])
    update_manifold_folder(folder, nodes=relabeled)
    mf = ManifoldFolder.load(folder)
    assert mf.node_labels == ["x", "y", "z"]
    # No orphaned node files from the old labels.
    assert sorted(p.name for p in (folder / "nodes").iterdir()) == [
        "00_x.json", "01_y.json", "02_z.json",
    ]


def test_malformed_json_raises_format_error(tmp_path):
    # A corrupt manifest must surface as ManifoldFormatError, not a bare
    # JSONDecodeError — the HTTP routes and iter_manifold_folders only
    # guard against the former.
    folder = _author_manifold(tmp_path)
    (folder / "manifold.json").write_text("{ not valid json")
    with pytest.raises(ManifoldFormatError):
        ManifoldFolder.load(folder)


def test_iter_manifold_folders_skips_malformed(tmp_path, monkeypatch):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    folder, _ = create_manifold_folder(
        "local", "good", "", domain, _author_nodes(["a", "b", "c"]),
    )
    bad = folder.parent / "broken"
    (bad / "nodes").mkdir(parents=True)
    (bad / "manifold.json").write_text("{ truncated")
    found = {mf.name for _ns, mf in iter_manifold_folders()}
    assert found == {"good"}


def test_iter_manifold_folders(tmp_path, monkeypatch):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    create_manifold_folder("local", "one", "", domain, _author_nodes(["a", "b", "c"]))
    create_manifold_folder("shared", "two", "", domain, _author_nodes(["a", "b", "c"]))
    found = {(ns, mf.name) for ns, mf in iter_manifold_folders()}
    assert found == {("local", "one"), ("shared", "two")}
    only_local = {(ns, mf.name) for ns, mf in iter_manifold_folders("local")}
    assert only_local == {("local", "one")}


# ============================================================ discover mode ===

def _discover_corpora(labels: list[str]) -> dict[str, list[str]]:
    """``{label: [statement, ...]}`` for the discover authoring shape."""
    return {
        label: [f"{label} statement {i}" for i in range(3)]
        for label in labels
    }


def test_create_discover_manifold_folder_round_trip(tmp_path, monkeypatch):
    """A freshly authored discover folder loads with the expected shape."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    folder = create_discover_manifold_folder(
        "local", "personas", "five personas",
        fit_mode="pca",
        node_corpora=_discover_corpora(
            ["pirate", "caveman", "assistant", "scholar", "robot"],
        ),
        hyperparams={"max_dim": 8, "var_threshold": 0.70},
    )
    mf = ManifoldFolder.load(folder)
    assert mf.name == "personas"
    assert mf.fit_mode == "pca"
    assert mf.is_discover
    assert mf.domain == {}
    assert mf.node_coords == []
    assert mf.node_labels == [
        "pirate", "caveman", "assistant", "scholar", "robot",
    ]
    assert mf.hyperparams == {"max_dim": 8, "var_threshold": 0.70}
    groups = mf.node_groups()
    assert [label for label, _ in groups] == mf.node_labels
    assert all(len(stmts) == 3 for _, stmts in groups)


def test_create_discover_manifold_rejects_unknown_fit_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    with pytest.raises(ManifoldFormatError, match="fit_mode"):
        create_discover_manifold_folder(
            "local", "bad", "",
            fit_mode="banana",
            node_corpora=_discover_corpora(["a", "b", "c"]),
        )


def test_create_discover_manifold_rejects_authored_fit_mode(tmp_path, monkeypatch):
    """``authored`` is not a valid discover-mode fit_mode."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    with pytest.raises(ManifoldFormatError, match="fit_mode"):
        create_discover_manifold_folder(
            "local", "wrong", "",
            fit_mode="authored",
            node_corpora=_discover_corpora(["a", "b", "c"]),
        )


def test_discover_manifold_rejects_coords_on_nodes(tmp_path):
    """A discover folder must not carry per-node ``coords`` — those are derived."""
    folder = tmp_path / "leaky"
    (folder / "nodes").mkdir(parents=True)
    (folder / "nodes" / "00_a.json").write_text(json.dumps(["a says"]))
    (folder / "manifold.json").write_text(json.dumps({
        "format_version": MANIFOLD_FORMAT_VERSION,
        "name": "leaky",
        "fit_mode": "pca",
        "hyperparams": {"max_dim": 4},
        "nodes": [{"label": "a", "coords": [0.5]}],
        "files": {},
    }))
    with pytest.raises(ManifoldFormatError, match="must not carry 'coords'"):
        ManifoldFolder.load(folder)


def test_discover_manifold_rejects_domain_field(tmp_path):
    """A discover folder must not carry a ``domain`` — coords are derived."""
    folder = tmp_path / "leaky2"
    (folder / "nodes").mkdir(parents=True)
    (folder / "nodes" / "00_a.json").write_text(json.dumps(["a says"]))
    (folder / "manifold.json").write_text(json.dumps({
        "format_version": MANIFOLD_FORMAT_VERSION,
        "name": "leaky2",
        "fit_mode": "spectral",
        "hyperparams": {"max_dim": 4},
        "domain": {"type": "box", "axes": [
            {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0},
        ]},
        "nodes": [{"label": "a"}],
        "files": {},
    }))
    with pytest.raises(ManifoldFormatError, match="must not carry a 'domain'"):
        ManifoldFolder.load(folder)


def test_discover_nodes_sha256_sensitive_to_hyperparams(tmp_path, monkeypatch):
    """A hyperparam edit invalidates the fit cache — the staleness key changes."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    folder = create_discover_manifold_folder(
        "local", "pca_a", "",
        fit_mode="pca",
        node_corpora=_discover_corpora(["a", "b", "c"]),
        hyperparams={"max_dim": 8, "var_threshold": 0.70},
    )
    h1 = ManifoldFolder.load(folder).nodes_sha256()
    data = json.loads((folder / "manifold.json").read_text())
    data["hyperparams"] = {"max_dim": 4, "var_threshold": 0.70}
    (folder / "manifold.json").write_text(json.dumps(data))
    h2 = ManifoldFolder.load(folder).nodes_sha256()
    assert h1 != h2, "hyperparam change must invalidate the fit cache"


def test_discover_nodes_sha256_sensitive_to_fit_mode(tmp_path, monkeypatch):
    """Switching ``pca`` ↔ ``spectral`` invalidates the fit cache."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    folder = create_discover_manifold_folder(
        "local", "mode_a", "",
        fit_mode="pca",
        node_corpora=_discover_corpora(["a", "b", "c"]),
        hyperparams={"max_dim": 8},
    )
    h_pca = ManifoldFolder.load(folder).nodes_sha256()
    data = json.loads((folder / "manifold.json").read_text())
    data["fit_mode"] = "spectral"
    (folder / "manifold.json").write_text(json.dumps(data))
    h_spec = ManifoldFolder.load(folder).nodes_sha256()
    assert h_pca != h_spec, "fit_mode change must invalidate the fit cache"


def test_discover_nodes_sha256_sensitive_to_corpus(tmp_path, monkeypatch):
    """The standing invariant: a corpus edit invalidates the fit cache."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    folder = create_discover_manifold_folder(
        "local", "corp_a", "",
        fit_mode="pca",
        node_corpora=_discover_corpora(["a", "b", "c"]),
    )
    h1 = ManifoldFolder.load(folder).nodes_sha256()
    (folder / "nodes" / "00_a.json").write_text(
        json.dumps(["a says something else"])
    )
    h2 = ManifoldFolder.load(folder).nodes_sha256()
    assert h1 != h2


def test_discover_write_metadata_round_trip(tmp_path, monkeypatch):
    """A re-written discover folder loads identically."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    folder = create_discover_manifold_folder(
        "local", "rw", "round-trip",
        fit_mode="spectral",
        node_corpora=_discover_corpora(["a", "b", "c"]),
        hyperparams={"max_dim": 6, "k_nn": 5},
    )
    mf = ManifoldFolder.load(folder)
    mf.write_metadata()
    again = ManifoldFolder.load(folder)
    assert again.fit_mode == "spectral"
    assert again.hyperparams == {"max_dim": 6, "k_nn": 5}
    assert again.node_labels == mf.node_labels
    assert again.domain == {}
    assert again.node_coords == []


def test_discover_create_drops_cross_method_hyperparams(tmp_path, monkeypatch):
    """A spectral key on a pca create gets dropped — would crash the dispatcher."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    folder = create_discover_manifold_folder(
        "local", "sanity_pca", "",
        fit_mode="pca",
        node_corpora=_discover_corpora(["a", "b", "c"]),
        hyperparams={
            "max_dim": 8,            # shared — kept
            "var_threshold": 0.70,   # pca — kept
            "k_nn": 5,               # spectral — dropped
            "bandwidth": 0.1,        # spectral — dropped
            "foo": "bar",            # unknown — dropped
        },
    )
    mf = ManifoldFolder.load(folder)
    assert mf.hyperparams == {"max_dim": 8, "var_threshold": 0.70}


def test_discover_create_drops_cross_method_hyperparams_spectral(tmp_path, monkeypatch):
    """Mirror — a pca key on a spectral create gets dropped."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    folder = create_discover_manifold_folder(
        "local", "sanity_spec", "",
        fit_mode="spectral",
        node_corpora=_discover_corpora(["a", "b", "c"]),
        hyperparams={
            "max_dim": 8,
            "k_nn": 5,
            "bandwidth": 0.1,
            "var_threshold": 0.70,   # pca — dropped
        },
    )
    mf = ManifoldFolder.load(folder)
    assert mf.hyperparams == {"max_dim": 8, "k_nn": 5, "bandwidth": 0.1}


def test_authored_fit_mode_defaults_when_field_absent(tmp_path, monkeypatch):
    """A legacy v3 manifold.json without a ``fit_mode`` field loads as authored."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    folder = _author_manifold(tmp_path)
    data = json.loads((folder / "manifold.json").read_text())
    data.pop("fit_mode", None)
    (folder / "manifold.json").write_text(json.dumps(data))
    mf = ManifoldFolder.load(folder)
    assert mf.fit_mode == "authored"
    assert not mf.is_discover


# ---------------------------------------------------------------------------
# merge_discover_manifolds
# ---------------------------------------------------------------------------


def test_merge_discover_unions_nodes(tmp_path, monkeypatch):
    """Two discover sources merge into a fresh discover folder whose
    node corpus is the union (in source order) of both inputs.
    """
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    create_discover_manifold_folder(
        "local", "src_a", "first heap",
        fit_mode="pca",
        node_corpora=_discover_corpora(["alpha", "beta"]),
        hyperparams={"max_dim": 8, "var_threshold": 0.7},
    )
    create_discover_manifold_folder(
        "local", "src_b", "second heap",
        fit_mode="pca",
        node_corpora=_discover_corpora(["gamma", "delta"]),
        hyperparams={"max_dim": 8, "var_threshold": 0.7},
    )
    target = merge_discover_manifolds(
        "local", "combined", "merged heap",
        sources=[("local", "src_a"), ("local", "src_b")],
    )
    mf = ManifoldFolder.load(target)
    assert mf.is_discover
    assert mf.fit_mode == "pca"
    assert mf.description == "merged heap"
    # Source order preserved, then per-source label order.
    assert mf.node_labels == ["alpha", "beta", "gamma", "delta"]
    # Hyperparams inherited from the first source.
    assert mf.hyperparams == {"max_dim": 8, "var_threshold": 0.7}
    # Every node carries the source corpus statements (3 each from
    # ``_discover_corpora``).
    groups = dict(mf.node_groups())
    for label in mf.node_labels:
        assert len(groups[label]) == 3


def test_merge_discover_refuses_authored_source(tmp_path, monkeypatch):
    """Authored manifolds carry user-declared geometry; merge refuses
    them because there's no shared coordinate system to reconcile.
    """
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    _author_manifold(tmp_path / "manifolds" / "local", name="authored_src")
    create_discover_manifold_folder(
        "local", "discover_src", "",
        fit_mode="pca",
        node_corpora=_discover_corpora(["a", "b", "c"]),
    )
    with pytest.raises(ManifoldFormatError, match="authored"):
        merge_discover_manifolds(
            "local", "combined", "",
            sources=[("local", "authored_src"), ("local", "discover_src")],
        )


def test_merge_discover_refuses_label_collision(tmp_path, monkeypatch):
    """Same label in two sources isn't auto-renamed — refuse so the
    user resolves the collision deliberately (renaming hides
    provenance otherwise).
    """
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    create_discover_manifold_folder(
        "local", "src_a", "",
        fit_mode="pca",
        node_corpora=_discover_corpora(["shared", "a_only"]),
    )
    create_discover_manifold_folder(
        "local", "src_b", "",
        fit_mode="pca",
        node_corpora=_discover_corpora(["shared", "b_only"]),
    )
    with pytest.raises(ManifoldFormatError, match="collision"):
        merge_discover_manifolds(
            "local", "combined", "",
            sources=[("local", "src_a"), ("local", "src_b")],
        )


def test_merge_discover_refuses_mixed_fit_modes_without_override(
    tmp_path, monkeypatch,
):
    """Sources with different fit_modes require an explicit override —
    the merged folder has one ``fit_mode``, so picking implicitly
    would silently lose information.
    """
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    create_discover_manifold_folder(
        "local", "src_pca", "",
        fit_mode="pca",
        node_corpora=_discover_corpora(["a", "b"]),
    )
    create_discover_manifold_folder(
        "local", "src_spec", "",
        fit_mode="spectral",
        node_corpora=_discover_corpora(["c", "d"]),
    )
    with pytest.raises(ManifoldFormatError, match="mixed fit_modes"):
        merge_discover_manifolds(
            "local", "combined", "",
            sources=[("local", "src_pca"), ("local", "src_spec")],
        )

    # Explicit override succeeds.
    target = merge_discover_manifolds(
        "local", "combined", "",
        sources=[("local", "src_pca"), ("local", "src_spec")],
        fit_mode="pca",
    )
    mf = ManifoldFolder.load(target)
    assert mf.fit_mode == "pca"


def test_merge_discover_refuses_missing_source(tmp_path, monkeypatch):
    """A non-existent source raises FileNotFoundError before any folder
    is written — atomic-on-failure discipline."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    create_discover_manifold_folder(
        "local", "exists", "",
        fit_mode="pca",
        node_corpora=_discover_corpora(["a", "b"]),
    )
    with pytest.raises(FileNotFoundError):
        merge_discover_manifolds(
            "local", "combined", "",
            sources=[("local", "exists"), ("local", "missing")],
        )


def test_merge_discover_force_overwrites(tmp_path, monkeypatch):
    """An existing destination raises FileExistsError without
    ``force=True``; with ``force=True`` it's rebuilt clean.
    """
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    for nm, labels in (("src_a", ["a", "b"]), ("src_b", ["c", "d"])):
        create_discover_manifold_folder(
            "local", nm, "",
            fit_mode="pca",
            node_corpora=_discover_corpora(labels),
        )
    sources = [("local", "src_a"), ("local", "src_b")]
    merge_discover_manifolds("local", "combined", "v1", sources=sources)
    with pytest.raises(FileExistsError):
        merge_discover_manifolds("local", "combined", "v2", sources=sources)
    # force=True rebuilds.
    target = merge_discover_manifolds(
        "local", "combined", "v2", sources=sources, force=True,
    )
    mf = ManifoldFolder.load(target)
    assert mf.description == "v2"


def test_merge_discover_refuses_under_two_sources(tmp_path, monkeypatch):
    """Single-source 'merge' is meaningless — refuse with a clear error."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    create_discover_manifold_folder(
        "local", "only", "",
        fit_mode="pca",
        node_corpora=_discover_corpora(["a", "b"]),
    )
    with pytest.raises(ValueError, match=">= 2"):
        merge_discover_manifolds(
            "local", "combined", "",
            sources=[("local", "only")],
        )

