"""On-disk format tests for saklas.io.manifolds — CPU-only, no model."""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pytest

import torch

from saklas.io.alignment import LayerAlignment
from saklas.io.manifolds import (
    MANIFOLD_FORMAT_VERSION,
    BakedManifoldError,
    DiscoverGenerationPlan,
    ManifoldFolder,
    ManifoldFormatError,
    ManifoldSidecar,
    append_discover_manifold_node,
    clear_manifold_tensors,
    create_baked_manifold_folder,
    create_discover_manifold_folder,
    create_manifold_folder,
    init_discover_manifold_folder,
    hash_manifold_files,
    iter_manifold_folders,
    manifold_summary,
    merge_discover_manifolds,
    min_nodes,
    plan_discover_generation,
    preflight_transfer_manifold,
    refresh_manifold,
    remove_manifold_folder,
    transfer_manifold,
    update_manifold_folder,
)


def _alignment(dense: torch.Tensor) -> LayerAlignment:
    target_dim, source_dim = dense.shape
    return LayerAlignment(dense, torch.eye(source_dim), torch.zeros(target_dim))


def _alignments(dense: dict[int, torch.Tensor]) -> dict[int, LayerAlignment]:
    return {layer: _alignment(value) for layer, value in dense.items()}


def _box1d(periodic: bool, labels: list[str]) -> dict[str, Any]:
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
    files: dict[str, Any] | None = None,
    domain: dict[str, Any] | None = None,
    nodes: list[dict[str, Any]] | None = None,
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
        "fit_mode": "authored",
        "domain": spec["domain"],
        "nodes": spec["nodes"],
        "files": files if files is not None else {},
    }
    (folder / "manifold.json").write_text(json.dumps(meta))
    return folder


def _sidecar_payload(
    *,
    name: str = "mood",
    labels: list[str] | None = None,
    domain: dict[str, Any] | None = None,
    fit_mode: str = "authored",
) -> dict[str, Any]:
    """Exact current fitted-sidecar payload for persistence unit tests."""
    node_labels = labels or ["a", "b", "c"]
    return {
        "format_version": MANIFOLD_FORMAT_VERSION,
        "name": name,
        "method": "manifold_pca",
        "saklas_version": "0",
        "domain": domain or {
            "type": "box",
            "axes": [
                {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0},
            ],
        },
        "node_count": len(node_labels),
        "node_labels": node_labels,
        "feature_space": "raw",
        "fit_mode": fit_mode,
    }


def test_min_nodes_per_dimension():
    assert min_nodes(1) == 3
    assert min_nodes(2) == 5
    assert min_nodes(3) == 7


def test_load_minimal_manifold(tmp_path: Path):
    folder = _author_manifold(tmp_path)
    mf = ManifoldFolder.load(folder)
    assert mf.name == "mood"
    assert mf.domain["type"] == "box"
    assert mf.node_labels == ["calm", "uneasy", "afraid", "frantic"]
    assert len(mf.node_coords) == 4
    assert mf.description == "a mood manifold"


def test_node_groups(tmp_path: Path):
    folder = _author_manifold(tmp_path)
    mf = ManifoldFolder.load(folder)
    groups = mf.node_groups()
    assert [label for label, _ in groups] == mf.node_labels
    label, statements = groups[0]
    assert label == "calm"
    assert statements == ["calm statement 0", "calm statement 1",
                          "calm statement 2"]


def test_missing_manifold_json_raises(tmp_path: Path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(ManifoldFormatError):
        ManifoldFolder.load(tmp_path / "empty")


def test_stale_format_version_raises(tmp_path: Path):
    folder = _author_manifold(tmp_path)
    meta = json.loads((folder / "manifold.json").read_text())
    meta["format_version"] = 2
    (folder / "manifold.json").write_text(json.dumps(meta))
    with pytest.raises(ManifoldFormatError):
        ManifoldFolder.load(folder)


def test_newer_format_version_raises(tmp_path: Path):
    # Symmetric ceiling, mirroring PackMetadata.load — a manifold authored
    # by a future saklas (format_version > local) must not load silently.
    folder = _author_manifold(tmp_path)
    meta = json.loads((folder / "manifold.json").read_text())
    meta["format_version"] = MANIFOLD_FORMAT_VERSION + 1
    (folder / "manifold.json").write_text(json.dumps(meta))
    with pytest.raises(ManifoldFormatError, match="need exactly"):
        ManifoldFolder.load(folder)


@pytest.mark.parametrize("field", ["format_version", "name", "fit_mode"])
def test_current_manifest_requires_identity_fields(tmp_path: Path, field: str) -> None:
    folder = _author_manifold(tmp_path)
    manifest_path = folder / "manifold.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.pop(field)
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ManifoldFormatError):
        ManifoldFolder.load(folder)


def test_domain_required(tmp_path: Path):
    folder = _author_manifold(tmp_path)
    meta = json.loads((folder / "manifold.json").read_text())
    del meta["domain"]
    (folder / "manifold.json").write_text(json.dumps(meta))
    with pytest.raises(ManifoldFormatError):
        ManifoldFolder.load(folder)


def test_too_few_nodes_raises(tmp_path: Path):
    with pytest.raises(ManifoldFormatError):
        ManifoldFolder.load(
            _author_manifold(tmp_path, labels=["calm", "afraid"])
        )


def test_too_few_nodes_for_2d(tmp_path: Path):
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


def test_node_coords_dim_mismatch_raises(tmp_path: Path):
    # 1-D domain but a node carrying two coordinates.
    nodes = [
        {"label": "a", "coords": [0.0, 0.0]},
        {"label": "b", "coords": [0.5]},
        {"label": "c", "coords": [1.0]},
    ]
    with pytest.raises(ManifoldFormatError):
        ManifoldFolder.load(_author_manifold(tmp_path, nodes=nodes))


def test_bad_node_label_raises(tmp_path: Path):
    nodes = [
        {"label": "calm", "coords": [0.0]},
        {"label": "Uneasy", "coords": [0.5]},   # uppercase invalid
        {"label": "afraid", "coords": [1.0]},
    ]
    with pytest.raises(ManifoldFormatError):
        ManifoldFolder.load(_author_manifold(tmp_path, nodes=nodes))


def test_duplicate_labels_raises(tmp_path: Path):
    nodes = [
        {"label": "calm", "coords": [0.0]},
        {"label": "calm", "coords": [0.5]},
        {"label": "afraid", "coords": [1.0]},
    ]
    with pytest.raises(ManifoldFormatError):
        ManifoldFolder.load(_author_manifold(tmp_path, nodes=nodes))


def test_poisedness_soft_warning(tmp_path: Path):
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


def test_missing_node_file_raises(tmp_path: Path):
    folder = _author_manifold(tmp_path)
    (folder / "nodes" / "00_calm.json").unlink()
    with pytest.raises(ManifoldFormatError):
        ManifoldFolder.load(folder)


def _add_dummy_tensor(folder: Path) -> None:
    """Write a placeholder fitted tensor + sidecar (no model needed)."""
    (folder / "stub-model.safetensors").write_bytes(b"placeholder-tensor")
    (folder / "stub-model.json").write_text(json.dumps(_sidecar_payload(
        labels=["calm", "uneasy", "afraid", "frantic"],
        domain={"type": "box", "axes": [
            {"name": "t", "periodic": True, "period": 1.0},
        ]},
    )))


def test_integrity_check_catches_tampering(tmp_path: Path):
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


def test_missing_sidecar_raises(tmp_path: Path):
    # A fitted .safetensors without its .json sidecar is a corrupt folder —
    # the same invariant ConceptFolder.load enforces for packs.
    folder = _author_manifold(tmp_path)
    _add_dummy_tensor(folder)
    (folder / "stub-model.json").unlink()
    with pytest.raises(ManifoldFormatError, match="no sidecar"):
        ManifoldFolder.load(folder)


def test_node_corpus_edits_do_not_trip_integrity(tmp_path: Path):
    folder = _author_manifold(tmp_path)
    _add_dummy_tensor(folder)
    files = hash_manifold_files(folder)
    meta = json.loads((folder / "manifold.json").read_text())
    meta["files"] = files
    (folder / "manifold.json").write_text(json.dumps(meta))
    (folder / "nodes" / "00_calm.json").write_text(json.dumps(["edited"]))
    ManifoldFolder.load(folder)


def test_nodes_sha256_stable_and_sensitive(tmp_path: Path):
    folder = _author_manifold(tmp_path)
    mf = ManifoldFolder.load(folder)
    h1 = mf.nodes_sha256()
    assert h1 == ManifoldFolder.load(folder).nodes_sha256()
    (folder / "nodes" / "00_calm.json").write_text(json.dumps(["changed"]))
    h2 = ManifoldFolder.load(folder).nodes_sha256()
    assert h1 != h2


def test_fit_parse_can_skip_historical_manifest_hashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    folder = _author_manifold(tmp_path)
    _add_dummy_tensor(folder)
    mf = ManifoldFolder.load(folder)
    mf.write_metadata()

    import saklas.io.manifold_folder as folder_module

    def _must_not_verify(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("fit-specific parse hashed the full manifest")

    monkeypatch.setattr(folder_module, "verify_integrity", _must_not_verify)
    loaded = ManifoldFolder.load(folder, verify_manifest=False)
    assert loaded.name == mf.name


def test_nodes_sha256_sensitive_to_coords(tmp_path: Path):
    # Moving a node's authoring coordinate must invalidate the hash even
    # when the corpus is untouched — the fit depends on the geometry.
    folder = _author_manifold(tmp_path)
    h1 = ManifoldFolder.load(folder).nodes_sha256()
    meta = json.loads((folder / "manifold.json").read_text())
    meta["nodes"][1]["coords"] = [0.123]
    (folder / "manifold.json").write_text(json.dumps(meta))
    h2 = ManifoldFolder.load(folder).nodes_sha256()
    assert h1 != h2


def test_write_metadata_populates_files(tmp_path: Path):
    folder = _author_manifold(tmp_path)
    _add_dummy_tensor(folder)
    mf = ManifoldFolder.load(folder)
    assert mf.files == {}
    mf.write_metadata()
    # Metadata-only rewrites preserve trusted entries; fitted writers name
    # their exact successful outputs explicitly.
    assert mf.files == {}
    mf.update_file_hashes(
        folder / "stub-model.safetensors", folder / "stub-model.json",
    )
    assert "stub-model.safetensors" in mf.files
    assert "stub-model.json" in mf.files
    assert not any(k.startswith("nodes/") for k in mf.files)
    reloaded = ManifoldFolder.load(folder)
    assert reloaded.files == mf.files


def test_update_file_hashes_only_reads_new_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    folder = _author_manifold(tmp_path)
    _add_dummy_tensor(folder)
    mf = ManifoldFolder.load(folder)
    mf.write_metadata()
    new_tensor = folder / "other-model.safetensors"
    new_sidecar = folder / "other-model.json"
    new_tensor.write_bytes(b"new tensor")
    new_sidecar.write_bytes((folder / "stub-model.json").read_bytes())

    from saklas.io import manifold_folder as folder_mod
    real_hash = folder_mod.hash_file
    hashed: list[Path] = []

    def _spy(path: Path) -> str:
        hashed.append(Path(path))
        return real_hash(path)

    monkeypatch.setattr(folder_mod, "hash_file", _spy)
    mf.update_file_hashes(new_tensor, new_sidecar)

    assert hashed == [new_tensor, new_sidecar]
    assert ManifoldFolder.load(folder).files == mf.files


def test_update_file_hashes_rejects_unreadable_future_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    folder = _author_manifold(tmp_path)
    mf = ManifoldFolder.load(folder, verify_manifest=False)
    manifest_path = folder / "manifold.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["format_version"] = MANIFOLD_FORMAT_VERSION + 1
    manifest["files"] = ["future-schema"]
    manifest_path.write_text(json.dumps(manifest))
    tensor = folder / "new-model.safetensors"
    tensor.write_bytes(b"future tensor")

    from saklas.io import manifold_folder as folder_mod

    def _unexpected_hash(_path: Path) -> str:
        pytest.fail("an unreadable manifest must be rejected before payload hashing")

    monkeypatch.setattr(folder_mod, "hash_file", _unexpected_hash)
    with pytest.raises(ManifoldFormatError, match="need exactly"):
        mf.update_file_hashes(tensor)

    assert json.loads(manifest_path.read_text()) == manifest


def test_bundle_drift_ignores_local_fit_transaction_state() -> None:
    from saklas.io.manifolds import _manifest_content_sha256

    base = {"format_version": MANIFOLD_FORMAT_VERSION, "name": "mood", "files": {}}
    local = {
        **base,
        "files": {"model.safetensors": "a" * 64},
        "artifact_id": "local-generation",
        "fit_epochs": {"model:raw": 3},
    }
    assert _manifest_content_sha256(json.dumps(base).encode()) == (
        _manifest_content_sha256(json.dumps(local).encode())
    )


def test_update_file_hashes_merges_latest_manifest_from_stale_instances(
    tmp_path: Path,
) -> None:
    """Concurrent fit snapshots cannot drop each other's fitted pair."""
    folder = _author_manifold(tmp_path)
    first = ManifoldFolder.load(folder)
    second = ManifoldFolder.load(folder)
    paths: list[Path] = []
    sidecar_payload = _sidecar_payload(
        labels=["calm", "uneasy", "afraid", "frantic"],
    )
    for stem in ("model-a", "model-b"):
        tensor = folder / f"{stem}.safetensors"
        sidecar = folder / f"{stem}.json"
        tensor.write_bytes(stem.encode())
        sidecar.write_text(json.dumps(sidecar_payload))
        paths.extend((tensor, sidecar))

    first.update_file_hashes(*paths[:2])
    second.update_file_hashes(*paths[2:])

    files = ManifoldFolder.load(folder).files
    assert set(paths_item.name for paths_item in paths) <= set(files)


def test_update_file_hashes_does_not_launder_unrelated_untracked_pair(
    tmp_path: Path,
) -> None:
    folder = _author_manifold(tmp_path)
    sidecar_payload = _sidecar_payload(
        labels=["calm", "uneasy", "afraid", "frantic"],
    )
    stale_tensor = folder / "stale.safetensors"
    stale_sidecar = folder / "stale.json"
    stale_tensor.write_bytes(b"untrusted")
    stale_sidecar.write_text(json.dumps(sidecar_payload))
    good_tensor = folder / "good.safetensors"
    good_sidecar = folder / "good.json"
    good_tensor.write_bytes(b"new")
    good_sidecar.write_text(json.dumps(sidecar_payload))

    mf = ManifoldFolder.load(folder, verify_manifest=False)
    mf.update_file_hashes(good_tensor, good_sidecar)

    files = ManifoldFolder.load(folder, verify_manifest=False).files
    assert set(files) == {"good.safetensors", "good.json"}


def test_manifold_sidecar_load(tmp_path: Path):
    path = tmp_path / "m.json"
    payload = _sidecar_payload(
        labels=["a", "b", "c", "d"],
        domain={"type": "sphere", "dim": 2},
    )
    payload.update({
        "method": "manifold_sae",
        "saklas_version": "3.1.0",
        "feature_space": "sae-gemma",
        "nodes_sha256": "deadbeef",
        "sae_release": "gemma",
    })
    path.write_text(json.dumps(payload))
    sc = ManifoldSidecar.load(path)
    assert sc.method == "manifold_sae"
    assert sc.domain == {"type": "sphere", "dim": 2}
    assert sc.node_count == 4
    assert sc.feature_space == "sae-gemma"
    assert sc.nodes_sha256 == "deadbeef"
    assert sc.sae_release == "gemma"
    # Optional diagnostics are empty when the current writer has none.
    assert sc.node_spread_per_layer == {}


def test_manifold_sidecar_node_spread_round_trips(tmp_path: Path):
    """The per-layer signal profile (``node_spread_per_layer``) round-trips."""
    path = tmp_path / "m.json"
    payload = _sidecar_payload(
        labels=["happy", "sad"],
        domain={"type": "custom", "dim": 1},
        fit_mode="pca",
    )
    payload.update({
        "method": "manifold_discover_pca",
        "saklas_version": "4.0.0",
        "node_spread_per_layer": {"5": 0.5, "12": 8.25, "20": 3.0},
    })
    path.write_text(json.dumps(payload))
    sc = ManifoldSidecar.load(path)
    assert sc.node_spread_per_layer == {"5": 0.5, "12": 8.25, "20": 3.0}
    # The peak layer is the one with the largest whitened between-node spread.
    peak = max(sc.node_spread_per_layer.items(), key=lambda kv: kv[1])
    assert peak[0] == "12"


def test_manifold_sidecar_topology_provenance_round_trips(tmp_path: Path):
    """``auto``-fit topology provenance survives the save → load round-trip.

    Regression: ``save_manifold`` filters metadata through a key whitelist, and
    ``resolved_fit_mode`` / ``topology_winner`` / ``topology_candidates`` (built
    by the extraction pipeline for an ``auto`` fit) were absent from it — so they
    were silently dropped from the on-disk sidecar even though the documented
    contract said the resolved geometry is recorded there.  The in-memory
    ``manifold.metadata`` carried them (``.update`` after save), which is why the
    extraction tests stayed green while the round-trip lost them.
    """
    import torch
    from saklas.core.manifold import (
        BoxAxis, BoxDomain, Manifold, fit_layer_subspace, load_manifold,
        save_manifold,
    )

    g = torch.Generator().manual_seed(0)
    domain = BoxDomain([BoxAxis("t", periodic=False, lo=0.0, hi=1.0)])
    coords = torch.tensor([[0.0], [0.5], [1.0]])
    embedded = domain.embed(coords)
    layers = {}
    for layer in (4, 5):
        sub, _ev_ratio = fit_layer_subspace(torch.randn(3, 6, generator=g), embedded)
        layers[layer] = sub
    man = Manifold(
        name="autotopo", domain=domain, node_labels=["a", "b", "c"],
        node_coords=coords, layers=layers,
        mahalanobis_share={4: 1.0, 5: 2.0},
    )
    candidates = [
        {"name": "flat-pca", "fit_mode": "pca", "intrinsic_dim": 2,
         "score": 12.5, "viable": True, "reason": None},
        {"name": "spectral", "fit_mode": "spectral", "intrinsic_dim": 1,
         "score": 40.0, "viable": True, "reason": None},
    ]
    out = tmp_path / "autotopo.safetensors"
    save_manifold(man, out, {
        "method": "manifold_discover_auto", "nodes_sha256": "x",
        "fit_mode": "auto", "resolved_fit_mode": "pca",
        "topology_winner": "flat-pca", "topology_candidates": candidates,
    })
    # The written sidecar JSON carries the provenance (the whitelist passes it).
    sidecar = json.loads(out.with_suffix(".json").read_text())
    assert sidecar["resolved_fit_mode"] == "pca"
    assert sidecar["topology_winner"] == "flat-pca"
    assert [c["name"] for c in sidecar["topology_candidates"]] == ["flat-pca", "spectral"]
    # And it surfaces on the loaded manifold's metadata (load reads the whole
    # sidecar back into ``metadata``).
    loaded = load_manifold(out)
    assert loaded.metadata["resolved_fit_mode"] == "pca"
    assert loaded.metadata["topology_winner"] == "flat-pca"


# --------------------------------------------------- authoring (create/update) ---

def _author_nodes(labels: list[str]) -> list[dict[str, Any]]:
    """A well-spread node list for a 1-D box, statements inline."""
    out = []
    for i, label in enumerate(labels):
        out.append({
            "label": label,
            "coords": [i / (len(labels) - 1)],
            "statements": [f"{label} statement {j}" for j in range(3)],
        })
    return out


def test_create_manifold_folder_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
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


def test_create_manifold_folder_conflict(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    nodes = _author_nodes(["a", "b", "c"])
    create_manifold_folder("local", "dup", "", domain, nodes)
    with pytest.raises(FileExistsError):
        create_manifold_folder("local", "dup", "", domain, nodes)


def test_create_manifold_folder_too_few_nodes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    with pytest.raises(ManifoldFormatError):
        create_manifold_folder(
            "local", "thin", "", domain, _author_nodes(["a", "b"]),
        )


def test_create_manifold_folder_bad_coords_arity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
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


def test_create_manifold_folder_empty_statements(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    nodes = _author_nodes(["a", "b", "c"])
    nodes[1]["statements"] = []
    with pytest.raises(ManifoldFormatError):
        create_manifold_folder("local", "nostmt", "", domain, nodes)


def test_create_manifold_folder_bad_namespace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    with pytest.raises(ManifoldFormatError):
        create_manifold_folder(
            "Bad Namespace", "m", "", domain, _author_nodes(["a", "b", "c"]),
        )


def test_update_manifold_folder_statements(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
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


def test_update_manifold_folder_relabels_cleanly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
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


def test_update_manifold_folder_does_not_hash_fitted_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    folder, _ = create_manifold_folder(
        "local", "mood", "", domain, _author_nodes(["a", "b", "c"]),
    )
    _fake_fit_tensor(folder, "test/model")
    from saklas.io import packs

    hashed: list[Path] = []

    def unexpected_hash(path: Path) -> str:
        hashed.append(Path(path))
        raise AssertionError("metadata-only authoring hashed a fitted payload")

    monkeypatch.setattr(packs, "hash_file", unexpected_hash)
    update_manifold_folder(folder, description="edited")
    assert hashed == []


def test_malformed_json_raises_format_error(tmp_path: Path):
    # A corrupt manifest must surface as ManifoldFormatError, not a bare
    # JSONDecodeError — the HTTP routes and iter_manifold_folders only
    # guard against the former.
    folder = _author_manifold(tmp_path)
    (folder / "manifold.json").write_text("{ not valid json")
    with pytest.raises(ManifoldFormatError):
        ManifoldFolder.load(folder)


@pytest.mark.parametrize("root", ["[]", "null", "1"])
def test_non_object_manifest_raises_format_error(
    tmp_path: Path, root: str,
) -> None:
    folder = _author_manifold(tmp_path)
    (folder / "manifold.json").write_text(root)

    with pytest.raises(ManifoldFormatError, match="JSON object"):
        ManifoldFolder.load(folder)


def test_non_object_manifest_rejected_by_locked_metadata_writers(
    tmp_path: Path,
) -> None:
    folder = _author_manifold(tmp_path)
    mf = ManifoldFolder.load(folder, verify_manifest=False)
    manifest_path = folder / "manifold.json"
    manifest_path.write_text("[]")
    tensor = folder / "new-model.safetensors"
    tensor.write_bytes(b"payload")

    with pytest.raises(ManifoldFormatError, match="JSON object"):
        mf.update_file_hashes(tensor)
    with pytest.raises(ManifoldFormatError, match="JSON object"):
        mf.write_metadata()


def test_write_metadata_rejects_concurrent_future_manifest(
    tmp_path: Path,
) -> None:
    folder = _author_manifold(tmp_path)
    stale = ManifoldFolder.load(folder, verify_manifest=False)
    manifest_path = folder / "manifold.json"
    future = json.loads(manifest_path.read_text())
    future["format_version"] = MANIFOLD_FORMAT_VERSION + 1
    manifest_path.write_text(json.dumps(future))

    with pytest.raises(ManifoldFormatError, match="need exactly"):
        stale.write_metadata()

    assert json.loads(manifest_path.read_text()) == future


def test_non_object_sidecar_raises_format_error(tmp_path: Path) -> None:
    path = tmp_path / "model.json"
    path.write_text("[]")

    with pytest.raises(ManifoldFormatError, match="JSON object"):
        ManifoldSidecar.load(path)


@pytest.mark.parametrize("version", [None, MANIFOLD_FORMAT_VERSION + 1])
def test_sidecar_requires_exact_current_format(
    tmp_path: Path, version: int | None,
) -> None:
    path = tmp_path / "model.json"
    payload = _sidecar_payload()
    if version is None:
        payload.pop("format_version")
    else:
        payload["format_version"] = version
    path.write_text(json.dumps(payload))

    with pytest.raises(ManifoldFormatError, match="need exactly"):
        ManifoldSidecar.load(path)


def test_sidecar_requires_current_identity_fields(tmp_path: Path) -> None:
    path = tmp_path / "model.json"
    payload = _sidecar_payload()
    payload.pop("fit_mode")
    path.write_text(json.dumps(payload))

    with pytest.raises(ManifoldFormatError, match="'fit_mode' must be str"):
        ManifoldSidecar.load(path)


def test_iter_manifold_folders_skips_malformed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
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


def test_iter_manifold_folders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    create_manifold_folder("local", "one", "", domain, _author_nodes(["a", "b", "c"]))
    create_manifold_folder("shared", "two", "", domain, _author_nodes(["a", "b", "c"]))
    found = {(ns, mf.name) for ns, mf in iter_manifold_folders()}
    assert found == {("local", "one"), ("shared", "two")}
    only_local = {(ns, mf.name) for ns, mf in iter_manifold_folders("local")}
    assert only_local == {("local", "one")}


def test_iter_manifold_folders_is_metadata_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from saklas.io import manifold_folder as manifold_folder_module

    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    folder, _ = create_manifold_folder(
        "local", "one", "", domain, _author_nodes(["a", "b", "c"]),
    )
    _add_dummy_tensor(folder)
    ManifoldFolder.load(folder, verify_manifest=False).update_file_hashes(
        folder / "stub-model.safetensors",
    )

    def _must_not_verify(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("ordinary manifold discovery verified payload hashes")

    monkeypatch.setattr(
        manifold_folder_module, "verify_integrity", _must_not_verify,
    )
    assert [(ns, mf.name) for ns, mf in iter_manifold_folders()] == [
        ("local", "one"),
    ]
    assert manifold_summary(folder)["name"] == "one"
    with pytest.raises(AssertionError, match="verified payload hashes"):
        ManifoldFolder.load(folder)


# ============================================================ discover mode ===

def _discover_corpora(labels: list[str]) -> dict[str, list[str]]:
    """``{label: [statement, ...]}`` for the discover authoring shape."""
    return {
        label: [f"{label} statement {i}" for i in range(3)]
        for label in labels
    }


def test_create_discover_manifold_folder_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
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


def test_create_discover_manifold_rejects_unknown_fit_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    with pytest.raises(ManifoldFormatError, match="fit_mode"):
        create_discover_manifold_folder(
            "local", "bad", "",
            fit_mode="banana",
            node_corpora=_discover_corpora(["a", "b", "c"]),
        )


def test_create_discover_manifold_rejects_authored_fit_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """``authored`` is not a valid discover-mode fit_mode."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    with pytest.raises(ManifoldFormatError, match="fit_mode"):
        create_discover_manifold_folder(
            "local", "wrong", "",
            fit_mode="authored",
            node_corpora=_discover_corpora(["a", "b", "c"]),
        )


def test_discover_manifold_rejects_coords_on_nodes(tmp_path: Path):
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


def test_discover_manifold_rejects_domain_field(tmp_path: Path):
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


def test_discover_nodes_sha256_sensitive_to_hyperparams(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
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


def test_discover_nodes_sha256_sensitive_to_fit_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
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


def test_discover_nodes_sha256_sensitive_to_corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
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


def test_discover_write_metadata_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
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


def test_discover_create_rejects_cross_method_hyperparams(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    with pytest.raises(ManifoldFormatError, match="k_nn"):
        create_discover_manifold_folder(
            "local", "sanity_pca", "",
            fit_mode="pca",
            node_corpora=_discover_corpora(["a", "b", "c"]),
            hyperparams={"max_dim": 8, "var_threshold": 0.70, "k_nn": 5},
        )


def test_discover_create_rejects_unknown_hyperparams(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    with pytest.raises(ManifoldFormatError, match="foo"):
        create_discover_manifold_folder(
            "local", "sanity_spec", "",
            fit_mode="spectral",
            node_corpora=_discover_corpora(["a", "b", "c"]),
            hyperparams={"max_dim": 8, "k_nn": 5, "foo": "bar"},
        )


# ---------------------------------------------------------------------------
# merge_discover_manifolds
# ---------------------------------------------------------------------------


def test_merge_discover_unions_nodes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
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


def test_merge_discover_does_not_hash_source_fitted_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    folders = [
        create_discover_manifold_folder(
            "local", name, "", fit_mode="pca",
            node_corpora=_discover_corpora(labels),
        )
        for name, labels in (
            ("src_a", ["a", "b", "c"]),
            ("src_b", ["d", "e", "f"]),
        )
    ]
    for folder in folders:
        _fake_fit_tensor(folder, "test/model")
    from saklas.io import packs

    hashed: list[Path] = []

    def unexpected_hash(path: Path) -> str:
        hashed.append(Path(path))
        raise AssertionError("metadata-only merge hashed a fitted payload")

    monkeypatch.setattr(packs, "hash_file", unexpected_hash)
    merge_discover_manifolds(
        "local", "combined", "", sources=[
            ("local", "src_a"), ("local", "src_b"),
        ],
    )
    assert hashed == []


def test_merge_discover_refuses_authored_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
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


def test_merge_discover_refuses_label_collision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
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


def test_merge_discover_refuses_missing_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
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


def test_merge_discover_force_overwrites(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
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


def test_merge_discover_refuses_under_two_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
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


# ============================================================ A1: label regex ===
#
# Node labels are stricter than the artifact NAME_REGEX — they drop ``.``
# (reserved as the bipolar-pole separator and unaddressable via ``%label``).


def test_dotted_node_label_rejected_authored_load(tmp_path: Path):
    """A dotted label is invalid even though it matches the old NAME_REGEX.

    ``deer.wolf`` passes ``^[a-z][a-z0-9._-]{0,63}$`` but the dot is the
    bipolar separator and the steering-expr lexer can't address it via
    ``%deer.wolf`` — so it must be rejected at load.
    """
    nodes = [
        {"label": "calm", "coords": [0.0]},
        {"label": "deer.wolf", "coords": [0.5]},   # dot — invalid
        {"label": "afraid", "coords": [1.0]},
    ]
    with pytest.raises(ManifoldFormatError, match="grammar-addressable"):
        ManifoldFolder.load(_author_manifold(tmp_path, nodes=nodes))


def test_dotted_node_label_rejected_create(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """``create_manifold_folder`` rejects a dotted label up front."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    nodes = _author_nodes(["a", "b", "c"])
    nodes[1]["label"] = "a.b"
    with pytest.raises(ManifoldFormatError, match="grammar-addressable"):
        create_manifold_folder("local", "dotted", "", domain, nodes)


def test_dotted_node_label_rejected_discover_create(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """``create_discover_manifold_folder`` rejects a dotted label too."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    with pytest.raises(ManifoldFormatError, match="grammar-addressable"):
        create_discover_manifold_folder(
            "local", "dotted_disc", "",
            fit_mode="pca",
            node_corpora=_discover_corpora(["alpha", "be.ta", "gamma"]),
        )


def test_underscore_and_hyphen_labels_still_valid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The stricter regex still admits ``_`` and ``-`` — only ``.`` is dropped."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    nodes = _author_nodes(["snake_case", "kebab-case", "plain"])
    folder, _ = create_manifold_folder("local", "ok_labels", "", domain, nodes)
    mf = ManifoldFolder.load(folder)
    assert mf.node_labels == ["snake_case", "kebab-case", "plain"]


# ============================================================ B3: lifecycle ===
#
# remove_manifold_folder / clear_manifold_tensors / refresh_manifold —
# the manifold analogue of pack rm / clear / refresh in cache_ops.


def _fake_fit_tensor(folder: Path, model_id: str, *, release: str | None = None) -> Path:
    """Drop a placeholder fitted tensor + sidecar for ``model_id``.

    No model needed — the lifecycle functions only glob/parse filenames.
    """
    from saklas.io.paths import sidecar_filename, tensor_filename
    ts = folder / tensor_filename(model_id, release=release)
    sc = folder / sidecar_filename(model_id, release=release)
    ts.write_bytes(b"placeholder-tensor")
    sc.write_text(json.dumps(_sidecar_payload(name=folder.name)))
    # Trust only this successful pair; metadata rewrites never scan-and-bless
    # unrelated fitted files.
    ManifoldFolder.load(folder, verify_manifest=False).update_file_hashes(ts, sc)
    return ts


def test_clear_manifold_tensors_removes_tensors_keeps_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    folder, _ = create_manifold_folder(
        "local", "mood", "", domain, _author_nodes(["a", "b", "c"]),
    )
    fitted = _fake_fit_tensor(folder, "google/gemma-3-4b-it")
    capture_sha = "c" * 64
    sidecar_path = fitted.with_suffix(".json")
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["capture_sha256"] = capture_sha
    sidecar_path.write_text(json.dumps(sidecar))
    ManifoldFolder.load(folder, verify_manifest=False).update_file_hashes(
        fitted, sidecar_path,
    )
    from saklas.io.paths import model_dir

    capture_dir = model_dir("google/gemma-3-4b-it") / "manifold_capture"
    capture_dir.mkdir(parents=True)
    capture_file = capture_dir / f"{capture_sha}.centroids.layer_0.safetensors"
    capture_file.write_bytes(b"cached")
    n = clear_manifold_tensors("local", "mood")
    assert n == 2  # tensor + sidecar
    assert not list(folder.glob("*.safetensors"))
    assert not capture_file.exists()
    # Corpus + manifest survive.
    assert (folder / "manifold.json").exists()
    assert (folder / "nodes").is_dir()
    mf = ManifoldFolder.load(folder)
    assert mf.node_labels == ["a", "b", "c"]
    # The integrity manifest no longer references the removed files.
    assert not any(k.endswith(".safetensors") for k in mf.files)


def test_shared_capture_survives_until_last_fitted_owner_is_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    folders = [
        create_manifold_folder(
            "local", name, "", domain, _author_nodes(["a", "b", "c"]),
        )[0]
        for name in ("mood-a", "mood-b")
    ]
    capture_sha = "e" * 64
    for folder in folders:
        fitted = _fake_fit_tensor(folder, "google/gemma-3-4b-it")
        sidecar_path = fitted.with_suffix(".json")
        sidecar = json.loads(sidecar_path.read_text())
        sidecar["capture_sha256"] = capture_sha
        sidecar_path.write_text(json.dumps(sidecar))
        ManifoldFolder.load(folder, verify_manifest=False).update_file_hashes(
            fitted, sidecar_path,
        )

    from saklas.io.paths import model_dir

    capture_dir = model_dir("google/gemma-3-4b-it") / "manifold_capture"
    capture_dir.mkdir(parents=True)
    capture_file = capture_dir / f"{capture_sha}.rows.layer_0.safetensors"
    capture_file.write_bytes(b"shared")

    clear_manifold_tensors("local", "mood-a")
    assert capture_file.exists()
    remove_manifold_folder("local", "mood-b")
    assert not capture_file.exists()


def test_clear_reaps_capture_when_sidecar_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    folder, _ = create_manifold_folder(
        "local", "mood", "", domain, _author_nodes(["a", "b", "c"]),
    )
    fitted = _fake_fit_tensor(folder, "model/a")
    fitted.with_suffix(".json").unlink()
    from saklas.io.paths import model_dir

    capture_sha = "1" * 64
    capture_dir = model_dir("model/a") / "manifold_capture"
    capture_dir.mkdir(parents=True)
    capture_file = capture_dir / f"{capture_sha}.rows.layer_0.safetensors"
    capture_file.write_bytes(b"orphan")

    clear_manifold_tensors("local", "mood")
    assert not capture_file.exists()


def test_clear_reaps_capture_when_both_fitted_halves_are_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    folder, _ = create_manifold_folder(
        "local", "mood", "", domain, _author_nodes(["a", "b", "c"]),
    )
    fitted = _fake_fit_tensor(folder, "model/a")
    fitted.unlink()
    fitted.with_suffix(".json").unlink()
    from saklas.io.paths import model_dir

    capture_sha = "4" * 64
    capture_dir = model_dir("model/a") / "manifold_capture"
    capture_dir.mkdir(parents=True)
    capture_file = capture_dir / f"{capture_sha}.rows.layer_0.safetensors"
    capture_file.write_bytes(b"orphan")

    assert clear_manifold_tensors("local", "mood") == 0
    assert not capture_file.exists()


def test_rm_reaps_capture_when_sidecar_is_corrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    folder, _ = create_manifold_folder(
        "local", "mood", "", domain, _author_nodes(["a", "b", "c"]),
    )
    fitted = _fake_fit_tensor(folder, "model/a")
    fitted.with_suffix(".json").write_text("{")
    from saklas.io.paths import model_dir

    capture_sha = "2" * 64
    capture_dir = model_dir("model/a") / "manifold_capture"
    capture_dir.mkdir(parents=True)
    capture_file = capture_dir / f"{capture_sha}.rows.layer_0.safetensors"
    capture_file.write_bytes(b"orphan")

    remove_manifold_folder("local", "mood")
    assert not capture_file.exists()


def test_rm_reaps_capture_when_both_fitted_halves_are_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    folder, _ = create_manifold_folder(
        "local", "mood", "", domain, _author_nodes(["a", "b", "c"]),
    )
    fitted = _fake_fit_tensor(folder, "model/a")
    fitted.unlink()
    fitted.with_suffix(".json").unlink()
    from saklas.io.paths import model_dir

    capture_sha = "5" * 64
    capture_dir = model_dir("model/a") / "manifold_capture"
    capture_dir.mkdir(parents=True)
    capture_file = capture_dir / f"{capture_sha}.rows.layer_0.safetensors"
    capture_file.write_bytes(b"orphan")

    remove_manifold_folder("local", "mood")
    assert not capture_file.exists()


def test_corrupt_owner_cleanup_preserves_capture_with_readable_shared_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    folders = [
        create_manifold_folder(
            "local", name, "", domain, _author_nodes(["a", "b", "c"]),
        )[0]
        for name in ("broken-owner", "live-owner")
    ]
    broken = _fake_fit_tensor(folders[0], "model/a")
    broken.with_suffix(".json").write_text("{")
    live = _fake_fit_tensor(folders[1], "model/a")
    capture_sha = "3" * 64
    live_sidecar = live.with_suffix(".json")
    sidecar = json.loads(live_sidecar.read_text())
    sidecar["capture_sha256"] = capture_sha
    live_sidecar.write_text(json.dumps(sidecar))
    ManifoldFolder.load(folders[1], verify_manifest=False).update_file_hashes(
        live, live_sidecar,
    )
    from saklas.io.paths import model_dir

    capture_dir = model_dir("model/a") / "manifold_capture"
    capture_dir.mkdir(parents=True)
    capture_file = capture_dir / f"{capture_sha}.rows.layer_0.safetensors"
    capture_file.write_bytes(b"shared")

    remove_manifold_folder("local", "broken-owner")
    assert capture_file.exists()


def test_clear_manifold_tensors_repairs_orphan_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    folder, _ = create_manifold_folder(
        "local", "mood", "", domain, _author_nodes(["a", "b", "c"]),
    )
    fitted = _fake_fit_tensor(folder, "google/gemma-3-4b-it")
    fitted.unlink()

    assert clear_manifold_tensors("local", "mood") == 1
    assert not fitted.with_suffix(".json").exists()
    loaded = ManifoldFolder.load(folder)
    assert not any("gemma-3-4b-it" in name for name in loaded.files)


def test_clear_manifold_tensors_repairs_orphan_tensor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    folder, _ = create_manifold_folder(
        "local", "mood", "", domain, _author_nodes(["a", "b", "c"]),
    )
    fitted = _fake_fit_tensor(folder, "google/gemma-3-4b-it")
    fitted.with_suffix(".json").unlink()

    assert clear_manifold_tensors("local", "mood") == 1
    assert not fitted.exists()
    ManifoldFolder.load(folder)


def test_clear_manifold_tensors_repairs_corrupt_fitted_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    folder, _ = create_manifold_folder(
        "local", "mood", "", domain, _author_nodes(["a", "b", "c"]),
    )
    fitted = _fake_fit_tensor(folder, "google/gemma-3-4b-it")
    fitted.write_bytes(b"same path, different fitted bytes")

    with pytest.raises(ManifoldFormatError, match="integrity"):
        ManifoldFolder.load(folder)
    assert clear_manifold_tensors("local", "mood") == 2
    ManifoldFolder.load(folder)


def test_scoped_clear_never_launders_unselected_corruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    folder, _ = create_manifold_folder(
        "local", "mood", "", domain, _author_nodes(["a", "b", "c"]),
    )
    corrupt = _fake_fit_tensor(folder, "model/a")
    _fake_fit_tensor(folder, "model/b")
    corrupt.write_bytes(b"tampered but not selected")

    assert clear_manifold_tensors("local", "mood", "model/b") == 2
    with pytest.raises(ManifoldFormatError, match="integrity"):
        ManifoldFolder.load(folder)


def test_clear_manifold_tensors_variant_filter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    folder, _ = create_manifold_folder(
        "local", "mood", "", domain, _author_nodes(["a", "b", "c"]),
    )
    _fake_fit_tensor(folder, "google/gemma-3-4b-it")
    _fake_fit_tensor(folder, "google/gemma-3-4b-it", release="gemma-scope")
    # variant="raw" only drops the canonical tensor, leaves the SAE one.
    n = clear_manifold_tensors("local", "mood", variant="raw")
    assert n == 2
    remaining = sorted(p.name for p in folder.glob("*.safetensors"))
    assert remaining == ["google__gemma-3-4b-it_sae-gemma-scope.safetensors"]


def test_clear_manifold_tensors_model_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """A model scope deletes only that model's tensors, keeping others."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    folder, _ = create_manifold_folder(
        "local", "mood", "", domain, _author_nodes(["a", "b", "c"]),
    )
    _fake_fit_tensor(folder, "google/gemma-3-4b-it")
    _fake_fit_tensor(folder, "Qwen/Qwen3-4B")
    # Scope to gemma — only its tensor + sidecar go.
    n = clear_manifold_tensors("local", "mood", "google/gemma-3-4b-it")
    assert n == 2
    remaining = sorted(p.name for p in folder.glob("*.safetensors"))
    assert remaining == ["Qwen__Qwen3-4B.safetensors"]
    # The integrity manifest still references the surviving model's tensor.
    mf = ManifoldFolder.load(folder)
    assert "Qwen__Qwen3-4B.safetensors" in mf.files


def test_clear_manifold_tensors_missing_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        clear_manifold_tensors("local", "nope")


def test_remove_manifold_folder_local(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    folder, _ = create_manifold_folder(
        "local", "mood", "", domain, _author_nodes(["a", "b", "c"]),
    )
    result = remove_manifold_folder("local", "mood")
    assert not folder.exists()
    assert result["removed"] is True
    assert result["source"] == "local"
    assert result["rematerializes_on_restart"] is False


def test_remove_manifold_folder_removes_referenced_capture_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    folder, _ = create_manifold_folder(
        "local", "mood", "", domain, _author_nodes(["a", "b", "c"]),
    )
    fitted = _fake_fit_tensor(folder, "google/gemma-3-4b-it")
    capture_sha = "d" * 64
    sidecar_path = fitted.with_suffix(".json")
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["capture_sha256"] = capture_sha
    sidecar_path.write_text(json.dumps(sidecar))
    ManifoldFolder.load(folder, verify_manifest=False).write_metadata()
    from saklas.io.paths import model_dir

    capture_dir = model_dir("google/gemma-3-4b-it") / "manifold_capture"
    capture_dir.mkdir(parents=True)
    capture_file = capture_dir / f"{capture_sha}.rows.layer_0.safetensors"
    capture_file.write_bytes(b"cached")
    remove_manifold_folder("local", "mood")
    assert not capture_file.exists()


def test_remove_manifold_folder_bundled_namespace_rematerializes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """A folder under ``default/`` reports the bundled-respawn flag."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    folder, _ = create_manifold_folder(
        "default", "shipped", "", domain, _author_nodes(["a", "b", "c"]),
    )
    result = remove_manifold_folder("default", "shipped")
    assert not folder.exists()
    assert result["rematerializes_on_restart"] is True


def test_remove_manifold_folder_missing_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        remove_manifold_folder("local", "nope")


def test_refresh_manifold_skips_local(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A local manifold has no upstream — refresh is a silent skip."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    create_manifold_folder(
        "local", "mood", "", domain, _author_nodes(["a", "b", "c"]),
    )
    assert refresh_manifold("local", "mood") == "skipped"


def test_metadata_only_lifecycle_does_not_hash_fitted_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    folder, _ = create_manifold_folder(
        "local", "mood", "", domain, _author_nodes(["a", "b", "c"]),
    )
    _fake_fit_tensor(folder, "test/model")
    from saklas.io import packs

    hashed: list[Path] = []

    def unexpected_hash(path: Path) -> str:
        hashed.append(Path(path))
        raise AssertionError("metadata-only lifecycle hashed a fitted payload")

    monkeypatch.setattr(packs, "hash_file", unexpected_hash)
    assert refresh_manifold("local", "mood") == "skipped"
    assert remove_manifold_folder("local", "mood")["removed"] is True
    assert hashed == []


def test_refresh_manifold_hf_repulls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """An ``hf://``-sourced manifold re-pulls via pull_manifold."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    folder, _ = create_manifold_folder(
        "alice", "mood", "", domain, _author_nodes(["a", "b", "c"]),
    )
    # Stamp an hf:// source the way pull_manifold would.
    mf = ManifoldFolder.load(folder)
    mf.source = "hf://alice/mood@v1"
    mf.write_metadata()

    captured: dict[str, Any] = {}
    import saklas.io.hf_manifolds as hfm

    def fake_pull(coord: str, *, target_folder: Path, force: bool, revision: str | None = None) -> Path:
        captured.update(coord=coord, target=target_folder,
                        force=force, revision=revision)
        return target_folder

    monkeypatch.setattr(hfm, "pull_manifold", fake_pull)
    assert refresh_manifold("alice", "mood") == "hf"
    assert captured["coord"] == "alice/mood"
    assert captured["revision"] == "v1"
    assert captured["target"] == folder


def test_refresh_manifold_model_scope_clears_fit_no_repull(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """A scoped refresh drops just the model's fit pair, never re-pulling.

    Mirrors ``cache_ops.refresh``'s scoped path: HF pulls are whole-repo,
    so a single-model refresh is a tensors-only delete (re-fits on next
    use), even on an ``hf://``-sourced manifold.
    """
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    folder, _ = create_manifold_folder(
        "alice", "mood", "", domain, _author_nodes(["a", "b", "c"]),
    )
    # Even on an hf:// source, a scoped refresh must NOT re-pull.
    mf = ManifoldFolder.load(folder)
    mf.source = "hf://alice/mood@v1"
    mf.write_metadata()
    _fake_fit_tensor(folder, "google/gemma-3-4b-it")
    _fake_fit_tensor(folder, "Qwen/Qwen3-4B")

    import saklas.io.hf_manifolds as hfm

    def boom(*a: object, **k: object) -> None:  # pragma: no cover - asserted not to fire
        raise AssertionError("scoped refresh must not re-pull from HF")

    monkeypatch.setattr(hfm, "pull_manifold", boom)

    assert refresh_manifold("alice", "mood", model_scope="google/gemma-3-4b-it") == "scoped"
    # Only gemma's fit pair is gone; Qwen's survives.
    remaining = sorted(p.name for p in folder.glob("*.safetensors"))
    assert remaining == ["Qwen__Qwen3-4B.safetensors"]


def test_refresh_manifold_missing_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        refresh_manifold("local", "nope")


def test_manifold_pair_lock_identity_is_external_stable_and_bounded(
    tmp_path: Path,
) -> None:
    from saklas.io.manifold_folder import manifold_pair_lock_path

    folder = tmp_path / "manifolds" / "local" / "mood"
    tensor = folder / ("model_" + "x" * 220 + ".safetensors")
    before = manifold_pair_lock_path(tensor)
    folder.mkdir(parents=True)
    folder.rmdir()
    after = manifold_pair_lock_path(tensor)

    assert before == after
    assert before.parent == folder.parent
    assert len(before.name) < 100


@pytest.mark.parametrize("operation", ["clear", "rm", "refresh"])
def test_lifecycle_mutations_wait_for_stable_pair_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    from saklas.io.manifold_folder import manifold_pair_lock

    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    namespace = "alice" if operation == "refresh" else "local"
    folder, _ = create_manifold_folder(
        namespace, "mood", "", domain, _author_nodes(["a", "b", "c"]),
    )
    fitted = _fake_fit_tensor(folder, "google/gemma-3-4b-it")
    if operation == "refresh":
        mf = ManifoldFolder.load(folder)
        mf.source = "hf://alice/mood@v1"
        mf.write_metadata()
        import saklas.io.hf_manifolds as hfm

        monkeypatch.setattr(
            hfm,
            "pull_manifold",
            lambda coord, *, target_folder, force, revision=None: target_folder,
        )

    started = threading.Event()
    done = threading.Event()
    errors: list[BaseException] = []

    def mutate() -> None:
        started.set()
        try:
            if operation == "clear":
                clear_manifold_tensors(namespace, "mood")
            elif operation == "rm":
                remove_manifold_folder(namespace, "mood")
            else:
                refresh_manifold(namespace, "mood")
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)
        finally:
            done.set()

    with manifold_pair_lock(fitted):
        worker = threading.Thread(target=mutate)
        worker.start()
        assert started.wait(1.0)
        assert not done.wait(0.1)
    worker.join(timeout=2.0)

    assert not worker.is_alive()
    assert errors == []
    assert done.is_set()


def test_force_authoring_reset_waits_for_stable_pair_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from saklas.io.manifold_folder import manifold_pair_lock

    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    folder = create_discover_manifold_folder(
        "local", "mood", "", fit_mode="pca",
        node_corpora={"old": ["old response"]},
    )
    fitted = _fake_fit_tensor(folder, "google/gemma-3-4b-it")
    started = threading.Event()
    done = threading.Event()
    errors: list[BaseException] = []

    def reset() -> None:
        started.set()
        try:
            plan_discover_generation(
                folder, "mood", "", fit_mode="pca", labels=["new"],
                force=True,
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)
        finally:
            done.set()

    with manifold_pair_lock(fitted):
        worker = threading.Thread(target=reset)
        worker.start()
        assert started.wait(1.0)
        assert not done.wait(0.1)
        assert fitted.exists()
    worker.join(timeout=2.0)

    assert not worker.is_alive()
    assert errors == []
    assert done.is_set()
    assert not fitted.exists()
    manifest = json.loads((folder / "manifold.json").read_text())
    assert manifest["nodes"] == [{"label": "new"}]


# ============================================================ B6a: transfer ===
#
# transfer_manifold applies a GIVEN per-layer Procrustes map to a fitted
# manifold's subspace, writing a _from-<safe_src> variant tensor.


def test_rectangular_affine_transfer_preserves_points_frame_and_steering(
    tmp_path: Path,
) -> None:
    import torch
    from saklas.core.manifold import (
        CustomDomain, LayerSubspace, Manifold, load_manifold, save_manifold,
        transfer_manifold_subspaces,
    )
    from saklas.io.alignment import LayerAlignment

    basis = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ])
    node_coords = torch.tensor([[0.0, 0.0], [1.0, -0.5], [-0.25, 1.5]])
    source = Manifold(
        name="rect-flat", domain=CustomDomain(2), node_labels=["a", "b", "c"],
        node_coords=node_coords,
        layers={0: LayerSubspace.affine(
            torch.tensor([0.5, -1.0, 2.0, 0.25]), basis,
            node_coords=node_coords,
        )},
    )
    dense = torch.tensor([
        [2.0, 0.2, 0.0, 0.0], [0.0, 0.5, 0.0, 0.0],
        [0.3, 0.0, 1.0, 0.0], [0.0, -0.4, 0.0, 1.0],
        [0.1, 0.2, 0.0, 0.0], [0.0, 0.3, 0.0, 0.0],
    ])
    alignment = LayerAlignment(
        dense, torch.eye(4), torch.linspace(-0.5, 0.5, 6),
    )
    target = transfer_manifold_subspaces(
        source, {0: alignment}, whitener=_target_whitener(dim=6, layers=(0,)),
        from_model="src", to_model="tgt",
    )
    for point in (torch.tensor([0.2, -0.4]), node_coords[1]):
        assert torch.allclose(
            target.manifold_point(0, point),
            alignment.apply_points(source.manifold_point(0, point)),
            atol=1e-5,
        )
    target_sub = target.layers[0]
    assert torch.allclose(
        target_sub.basis @ target_sub.basis.T, torch.eye(2), atol=1e-5,
    )
    assert target_sub.node_coords is not None
    assert torch.allclose(
        target_sub.node_coords[1] @ target_sub.basis,
        alignment.apply_vector(node_coords[1] @ basis),
        atol=1e-5,
    )
    # The explicit affine coordinate reparameterization is an on-disk contract,
    # not merely an in-memory transfer detail.
    path = tmp_path / "rect-flat.safetensors"
    save_manifold(target, path, {"method": "manifold_procrustes_transfer"})
    loaded = load_manifold(path, verify_manifest=False)
    assert torch.allclose(
        loaded.manifold_point(0, [0.2, -0.4]),
        target.manifold_point(0, [0.2, -0.4]),
        atol=1e-6,
    )


def test_rectangular_curved_transfer_preserves_world_points_and_clears_sigma() -> None:
    import torch
    from dataclasses import replace
    from saklas.core.manifold import (
        BoxAxis, BoxDomain, Manifold, fit_layer_subspace,
        fit_rbf_interpolant, transfer_manifold_subspaces,
    )
    from saklas.io.alignment import LayerAlignment

    domain = BoxDomain([BoxAxis("t", periodic=False, lo=0.0, hi=1.0)])
    coords = torch.tensor([[0.0], [0.5], [1.0]])
    embedded = domain.embed(coords)
    centroids = torch.tensor([
        [0.0, 0.0, 1.0, -1.0],
        [0.8, 0.4, 0.3, -0.2],
        [0.1, 1.2, -0.5, 0.5],
    ])
    sub, _ = fit_layer_subspace(centroids, embedded)
    assert sub.node_params is not None
    sigma_rw, sigma_pc = fit_rbf_interpolant(
        sub.node_params, torch.zeros(3, 1),
    )
    sub = replace(sub, sigma_rbf_weights=sigma_rw, sigma_poly_coeffs=sigma_pc)
    source = Manifold(
        name="rect-curve", domain=domain, node_labels=["a", "b", "c"],
        node_coords=coords, layers={0: sub},
    )
    dense = torch.tensor([
        [1.8, 0.0, 0.0, 0.0], [0.2, 0.7, 0.0, 0.0],
        [0.0, 0.1, 1.0, 0.0], [0.0, 0.0, 0.2, 1.0],
        [0.3, 0.0, 0.0, 0.0], [0.0, -0.2, 0.0, 0.0],
    ])
    alignment = LayerAlignment(dense, torch.eye(4), torch.arange(6).float() / 10)
    target = transfer_manifold_subspaces(
        source, {0: alignment}, whitener=_target_whitener(dim=6, layers=(0,)),
        from_model="src", to_model="tgt",
    )
    for position in ([0.0], [0.25], [0.5], [0.8], [1.0]):
        assert torch.allclose(
            target.manifold_point(0, position),
            alignment.apply_points(source.manifold_point(0, position)),
            atol=2e-5,
        )
    target_sub = target.layers[0]
    assert torch.allclose(
        target_sub.basis @ target_sub.basis.T,
        torch.eye(target_sub.rank), atol=1e-5,
    )
    assert not target_sub.has_sigma


def test_rectangular_transfer_rejects_collapsed_subspace_rank() -> None:
    import torch
    from saklas.core.manifold import CustomDomain, LayerSubspace, Manifold, transfer_manifold_subspaces
    from saklas.io.alignment import LayerAlignment

    source = Manifold(
        name="collapsed", domain=CustomDomain(2), node_labels=["a", "b"],
        node_coords=torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
        layers={0: LayerSubspace.affine(
            torch.zeros(3), torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            node_coords=torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
        )},
    )
    collapsed = torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    alignment = LayerAlignment(collapsed, torch.eye(3), torch.zeros(2))
    with pytest.raises(ValueError, match="rank-deficient"):
        transfer_manifold_subspaces(
            source, {0: alignment},
            whitener=_target_whitener(dim=2, layers=(0,)),
            from_model="src", to_model="tgt",
        )


def test_rectangular_transfer_rejects_oblique_rank_collapse() -> None:
    """Rank detection uses singular values, not unpivoted QR diagonals."""
    import torch
    from saklas.core.manifold import (
        CustomDomain, LayerSubspace, Manifold, transfer_manifold_subspaces,
    )
    from saklas.io.alignment import LayerAlignment

    source = Manifold(
        name="oblique-collapse", domain=CustomDomain(3),
        node_labels=["a", "b", "c"], node_coords=torch.eye(3),
        layers={0: LayerSubspace.affine(
            torch.zeros(3), torch.eye(3), node_coords=torch.eye(3),
        )},
    )
    generator = torch.Generator().manual_seed(30061)
    collapsed = (
        torch.randn(3, 2, generator=generator)
        @ torch.randn(2, 3, generator=generator)
    )
    alignment = LayerAlignment(collapsed, torch.eye(3), torch.zeros(3))

    with pytest.raises(ValueError, match="rank-deficient"):
        transfer_manifold_subspaces(
            source, {0: alignment},
            whitener=_target_whitener(dim=3, layers=(0,)),
            from_model="src", to_model="tgt",
        )


def _target_whitener(*, dim: int = 6, layers: tuple[int, ...] = (4, 5, 6)):
    """A target-model whitener over the transferred layers.

    Transfer is Mahalanobis-only now (the share re-bake is mandatory; there is
    no Euclidean rebake), so every ``transfer_manifold`` call needs a target
    whitener covering the transferred layers.
    """
    import torch
    from saklas.core.mahalanobis import LayerWhitener
    g = torch.Generator().manual_seed(99)
    acts = {L: torch.randn(120, dim, generator=g) for L in layers}
    means = {L: torch.zeros(dim) for L in layers}
    return LayerWhitener.from_neutral_activations(acts, means)


def _fit_real_manifold(folder: Path, model_id: str, *, dim: int = 6, seed: int = 0):
    """Fit a tiny real authored manifold on synthetic centroids and save it.

    Returns the path of the written canonical tensor.
    """
    import torch
    from saklas.core.manifold import (
        MANIFOLD_FIT_POLICY_VERSION, BoxAxis, BoxDomain, Manifold,
        fit_layer_subspace, save_manifold,
    )

    g = torch.Generator().manual_seed(seed)
    domain = BoxDomain([BoxAxis("t", periodic=False, lo=0.0, hi=1.0)])
    coords = torch.tensor([[0.0], [0.5], [1.0]])
    embedded = domain.embed(coords)
    layers = {}
    for layer in (4, 5, 6):
        centroids = torch.randn(3, dim, generator=g)
        sub, _ev_ratio = fit_layer_subspace(centroids, embedded)
        layers[layer] = sub
    man = Manifold(
        name=folder.name,
        domain=domain,
        node_labels=["a", "b", "c"],
        node_coords=coords,
        layers=layers,
        # A per-model whitened share, so the transfer test can assert it
        # is re-baked in target space (Σ is per-model; the source metric is
        # invalid in target space).
        mahalanobis_share={4: 1.0, 5: 2.0, 6: 3.0},
    )
    from saklas.io.paths import tensor_filename
    out = folder / tensor_filename(model_id)
    mf = ManifoldFolder.load(folder, verify_manifest=False)
    save_manifold(man, out, {
        "method": "manifold_pca", "nodes_sha256": mf.nodes_sha256(),
        "model_fingerprint": f"fp:{model_id}",
        "fit_policy_version": MANIFOLD_FIT_POLICY_VERSION,
    })
    mf.update_file_hashes(out, out.with_suffix(".json"))
    return out


def test_transfer_preflight_returns_locked_source_layer_header(
    tmp_path: Path,
) -> None:
    folder = _author_manifold(tmp_path)
    _fit_real_manifold(folder, "src/model", dim=4)

    proof = preflight_transfer_manifold(
        folder, from_model="src/model", to_model="tgt/model",
    )
    assert proof.layers == (4, 5, 6)


def test_transfer_rejects_source_generation_changed_after_preflight(
    tmp_path: Path,
) -> None:
    folder = _author_manifold(tmp_path)
    _fit_real_manifold(folder, "src/model", dim=4, seed=0)
    proof = preflight_transfer_manifold(
        folder, from_model="src/model", to_model="tgt/model",
    )
    _fit_real_manifold(folder, "src/model", dim=4, seed=1)

    with pytest.raises(ManifoldFormatError, match="changed while alignment"):
        transfer_manifold(
            folder,
            from_model="src/model",
            to_model="tgt/model",
            alignment=_alignments({layer: torch.eye(4) for layer in proof.layers}),
            source_model_fingerprint="fp:src/model",
            target_model_fingerprint="fp:tgt/model",
            whitener=_target_whitener(dim=4),
            expected_source_proof=proof,
        )

    from saklas.io.paths import tensor_filename

    assert not (
        folder / tensor_filename("tgt/model", transferred_from="src/model")
    ).exists()


def test_transfer_preflight_wraps_corrupt_manifest(tmp_path: Path) -> None:
    folder = _author_manifold(tmp_path)
    _fit_real_manifold(folder, "src/model", dim=4)
    (folder / "manifold.json").write_text("{")

    with pytest.raises(ManifoldFormatError, match="manifest is unreadable"):
        preflight_transfer_manifold(
            folder, from_model="src/model", to_model="tgt/model",
        )


@pytest.mark.parametrize("root", ["[]", "null", '"manifest"'])
def test_transfer_preflight_rejects_non_object_manifest_before_payload_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, root: str,
) -> None:
    folder = _author_manifold(tmp_path)
    _fit_real_manifold(folder, "src/model", dim=4)
    (folder / "manifold.json").write_text(root)
    monkeypatch.setattr(
        "saklas.io.packs.verify_integrity",
        lambda *_args, **_kwargs: pytest.fail(
            "non-object manifest reached payload hashing"
        ),
    )

    with pytest.raises(ManifoldFormatError, match="JSON object"):
        preflight_transfer_manifold(
            folder, from_model="src/model", to_model="tgt/model",
        )


@pytest.mark.parametrize("digest", [None, 7, "abc", "g" * 64])
def test_transfer_preflight_rejects_invalid_selected_digest_before_hashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, digest: object,
) -> None:
    folder = _author_manifold(tmp_path)
    source = _fit_real_manifold(folder, "src/model", dim=4)
    manifest_path = folder / "manifold.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][source.name] = digest
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(
        "saklas.io.packs.verify_integrity",
        lambda *_args, **_kwargs: pytest.fail(
            "malformed selected digest reached payload hashing"
        ),
    )

    with pytest.raises(ManifoldFormatError, match="invalid sha256"):
        preflight_transfer_manifold(
            folder, from_model="src/model", to_model="tgt/model",
        )


def test_transfer_preflight_rejects_future_manifest_before_payload_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    folder = _author_manifold(tmp_path)
    _fit_real_manifold(folder, "src/model", dim=4)
    manifest_path = folder / "manifold.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["format_version"] = MANIFOLD_FORMAT_VERSION + 1
    manifest_path.write_text(json.dumps(manifest))

    monkeypatch.setattr(
        "saklas.io.packs.verify_integrity",
        lambda *_args, **_kwargs: pytest.fail(
            "future manifest reached payload hashing"
        ),
    )
    with pytest.raises(ManifoldFormatError, match="need exactly"):
        preflight_transfer_manifold(
            folder, from_model="src/model", to_model="tgt/model",
        )


def test_strict_fitted_load_rejects_non_object_manifest(tmp_path: Path) -> None:
    from saklas.core.manifold import load_manifold

    folder = _author_manifold(tmp_path)
    source = _fit_real_manifold(folder, "src/model", dim=4)
    (folder / "manifold.json").write_text("[]")

    with pytest.raises(ManifoldFormatError, match="JSON object"):
        load_manifold(source)


@pytest.mark.parametrize(
    ("payload", "message"), [("{", "unreadable"), ("[]", "JSON object")],
)
def test_strict_fitted_load_normalizes_trusted_corrupt_sidecar(
    tmp_path: Path, payload: str, message: str,
) -> None:
    from saklas.core.manifold import load_manifold
    from saklas.io.packs import hash_file

    folder = _author_manifold(tmp_path)
    source = _fit_real_manifold(folder, "src/model", dim=4)
    sidecar = source.with_suffix(".json")
    sidecar.write_text(payload)
    manifest_path = folder / "manifold.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][sidecar.name] = hash_file(sidecar)
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ManifoldFormatError, match=message):
        load_manifold(source)


def test_strict_fitted_load_rejects_invalid_digest_before_hashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from saklas.core.manifold import load_manifold

    folder = _author_manifold(tmp_path)
    source = _fit_real_manifold(folder, "src/model", dim=4)
    manifest_path = folder / "manifold.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][source.name] = "invalid"
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(
        "saklas.io.packs.verify_integrity",
        lambda *_args, **_kwargs: pytest.fail(
            "invalid digest reached fitted payload hashing"
        ),
    )

    with pytest.raises(ManifoldFormatError, match="invalid sha256"):
        load_manifold(source)


def test_transfer_manifold_identity_alignment_preserves_geometry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """An identity alignment leaves the per-layer subspace unchanged and
    writes the transferred tensor at the ``_from-<safe_src>`` filename."""
    import torch
    from saklas.core.manifold import load_manifold

    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    folder, _ = create_manifold_folder(
        "local", "mood", "", domain, _author_nodes(["a", "b", "c"]),
    )
    src_model = "google/gemma-3-4b-it"
    tgt_model = "Qwen/Qwen2.5-7B-Instruct"
    src_tensor = _fit_real_manifold(folder, src_model, dim=6)
    src_man = load_manifold(src_tensor)

    # Identity map per fitted layer.
    align = _alignments({L: torch.eye(6) for L in src_man.layers})

    out = transfer_manifold(
        folder, from_model=src_model, to_model=tgt_model,
        alignment=align, transfer_quality_estimate=0.9,
        source_model_fingerprint=f"fp:{src_model}",
        target_model_fingerprint=f"fp:{tgt_model}",
        whitener=_target_whitener(),
    )
    # Filename uses the transfer variant suffix.
    from saklas.io.paths import tensor_filename
    assert out.name == tensor_filename(tgt_model, transferred_from=src_model)
    assert out.name == "Qwen__Qwen2.5-7B-Instruct_from-google__gemma-3-4b-it.safetensors"

    tgt_man = load_manifold(out)
    assert sorted(tgt_man.layers) == sorted(src_man.layers)
    for L in src_man.layers:
        assert torch.allclose(tgt_man.layers[L].mean, src_man.layers[L].mean, atol=1e-5)
        assert torch.allclose(tgt_man.layers[L].basis, src_man.layers[L].basis, atol=1e-5)
    # The source's per-model Mahalanobis share is invalid in target space, so
    # it is RE-BAKED in the target metric (mandatory now — never dropped).
    assert src_man.mahalanobis_share  # source had one
    assert set(tgt_man.mahalanobis_share.keys()) == set(src_man.layers)
    assert tgt_man.mahalanobis_share != src_man.mahalanobis_share
    # Provenance lands in the sidecar.
    with open(out.with_suffix(".json")) as f:
        sc = json.load(f)
    assert sc["method"] == "manifold_procrustes_transfer"
    assert sc["source_model_id"] == src_model
    assert sc["transfer_quality_estimate"] == pytest.approx(0.9)
    # Folder integrity manifest now covers the transferred tensor.
    mf = ManifoldFolder.load(folder)
    assert out.name in mf.files


def test_transfer_manifold_rebakes_share_in_target_space(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """A target whitener covering the transferred layers re-bakes the
    Mahalanobis share in target space (instead of dropping it) and records
    ``share_metric == "mahalanobis"``."""
    import torch
    from saklas.core.mahalanobis import LayerWhitener
    from saklas.core.manifold import load_manifold

    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    folder, _ = create_manifold_folder(
        "local", "mood", "", domain, _author_nodes(["a", "b", "c"]),
    )
    src_model = "google/gemma-3-4b-it"
    tgt_model = "Qwen/Qwen2.5-7B-Instruct"
    src_tensor = _fit_real_manifold(folder, src_model, dim=6)
    src_man = load_manifold(src_tensor)
    align = _alignments({L: torch.eye(6) for L in src_man.layers})

    # Target whitener over the fitted layers {4, 5, 6}.
    g = torch.Generator().manual_seed(99)
    acts = {L: torch.randn(120, 6, generator=g) for L in src_man.layers}
    means = {L: torch.zeros(6) for L in src_man.layers}
    w = LayerWhitener.from_neutral_activations(acts, means)

    out = transfer_manifold(
        folder, from_model=src_model, to_model=tgt_model,
        alignment=align, whitener=w,
        source_model_fingerprint=f"fp:{src_model}",
        target_model_fingerprint=f"fp:{tgt_model}",
    )
    tgt_man = load_manifold(out)

    # Share recomputed (not dropped), one per transferred layer.
    assert set(tgt_man.mahalanobis_share.keys()) == set(src_man.layers)
    for L in src_man.layers:
        assert tgt_man.mahalanobis_share[L] > 0.0
    # They are *target*-metric values, not the source's hand-set share.
    assert tgt_man.mahalanobis_share != src_man.mahalanobis_share
    with open(out.with_suffix(".json")) as f:
        sc = json.load(f)
    assert sc["share_metric"] == "mahalanobis"


def test_transfer_manifold_rotation_maps_subspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """A known rotation Q maps mean → Q@mean and basis → basis@Q^T, so the
    transferred world-space activation at a node equals Q applied to the
    source activation."""
    import torch
    from saklas.core.manifold import load_manifold

    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    folder, _ = create_manifold_folder(
        "local", "mood", "", domain, _author_nodes(["a", "b", "c"]),
    )
    src_model = "src/model"
    tgt_model = "tgt/model"
    src_tensor = _fit_real_manifold(folder, src_model, dim=6, seed=3)
    src_man = load_manifold(src_tensor)

    # Random orthogonal Q via QR.
    A = torch.randn(6, 6, generator=torch.Generator().manual_seed(7))
    Q, _ = torch.linalg.qr(A)
    align = _alignments({L: Q for L in src_man.layers})

    out = transfer_manifold(
        folder, from_model=src_model, to_model=tgt_model, alignment=align,
        source_model_fingerprint=f"fp:{src_model}",
        target_model_fingerprint=f"fp:{tgt_model}",
        whitener=_target_whitener(),
    )
    tgt_man = load_manifold(out)
    # World-space activation at node 0 should be Q @ (source activation).
    for L in src_man.layers:
        src_pt = src_man.manifold_point(L, [0.0]).to(torch.float32)
        tgt_pt = tgt_man.manifold_point(L, [0.0]).to(torch.float32)
        assert torch.allclose(tgt_pt, Q @ src_pt, atol=1e-4)


def test_transfer_manifold_drops_uncovered_layers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """Layers the alignment doesn't cover are dropped from the transfer."""
    import torch
    from saklas.core.manifold import load_manifold

    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    folder, _ = create_manifold_folder(
        "local", "mood", "", domain, _author_nodes(["a", "b", "c"]),
    )
    src_model, tgt_model = "src/m", "tgt/m"
    _fit_real_manifold(folder, src_model, dim=6)  # writes the on-disk source fit
    # Cover only layer 5 of {4, 5, 6}.
    align = _alignments({5: torch.eye(6)})
    out = transfer_manifold(
        folder, from_model=src_model, to_model=tgt_model, alignment=align,
        source_model_fingerprint=f"fp:{src_model}",
        target_model_fingerprint=f"fp:{tgt_model}",
        whitener=_target_whitener(layers=(5,)),
    )
    tgt_man = load_manifold(out)
    assert sorted(tgt_man.layers) == [5]


def test_transfer_manifold_missing_source_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    import torch
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    folder, _ = create_manifold_folder(
        "local", "mood", "", domain, _author_nodes(["a", "b", "c"]),
    )
    with pytest.raises(FileNotFoundError):
        transfer_manifold(
            folder, from_model="never/fitted", to_model="tgt/m",
            alignment=_alignments({0: torch.eye(4)}),
            source_model_fingerprint="fp:never/fitted",
            target_model_fingerprint="fp:tgt/m",
        )


def test_transfer_manifold_empty_alignment_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    folder, _ = create_manifold_folder(
        "local", "mood", "", domain, _author_nodes(["a", "b", "c"]),
    )
    _fit_real_manifold(folder, "src/m", dim=6)
    with pytest.raises(ManifoldFormatError, match="empty"):
        transfer_manifold(
            folder, from_model="src/m", to_model="tgt/m", alignment={},
            source_model_fingerprint="fp:src/m",
            target_model_fingerprint="fp:tgt/m",
        )


def test_transfer_manifold_no_overlap_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """An alignment covering no fitted layer raises rather than write empty."""
    import torch
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    folder, _ = create_manifold_folder(
        "local", "mood", "", domain, _author_nodes(["a", "b", "c"]),
    )
    _fit_real_manifold(folder, "src/m", dim=6)  # layers 4,5,6
    with pytest.raises(ManifoldFormatError, match="covered none"):
        transfer_manifold(
            folder, from_model="src/m", to_model="tgt/m",
            alignment=_alignments({0: torch.eye(6), 1: torch.eye(6)}),
            source_model_fingerprint="fp:src/m",
            target_model_fingerprint="fp:tgt/m",
        )


def test_transfer_manifold_refuses_overwrite_without_force(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    import torch
    from saklas.core.manifold import load_manifold

    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    folder, _ = create_manifold_folder(
        "local", "mood", "", domain, _author_nodes(["a", "b", "c"]),
    )
    src_tensor = _fit_real_manifold(folder, "src/m", dim=6)
    align = _alignments({L: torch.eye(6) for L in load_manifold(src_tensor).layers})
    w = _target_whitener()
    transfer_manifold(
        folder, from_model="src/m", to_model="tgt/m", alignment=align, whitener=w,
        source_model_fingerprint="fp:src/m",
        target_model_fingerprint="fp:tgt/m",
    )
    import saklas.core.manifold as manifold_mod

    real_transfer = manifold_mod.transfer_manifold_subspaces
    monkeypatch.setattr(
        manifold_mod, "transfer_manifold_subspaces",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("trusted target performed transfer compute")
        ),
    )
    with pytest.raises(FileExistsError):
        transfer_manifold(
            folder, from_model="src/m", to_model="tgt/m", alignment=align, whitener=w,
            source_model_fingerprint="fp:src/m",
            target_model_fingerprint="fp:tgt/m",
        )
    # force=True overwrites cleanly.
    monkeypatch.setattr(manifold_mod, "transfer_manifold_subspaces", real_transfer)
    transfer_manifold(
        folder, from_model="src/m", to_model="tgt/m", alignment=align,
        source_model_fingerprint="fp:src/m",
        target_model_fingerprint="fp:tgt/m",
        whitener=w, force=True,
    )


def test_transfer_retries_pair_committed_before_manifest_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unproven transferred pair is repaired without requiring force."""
    import torch
    from saklas.core.manifold import load_manifold
    from saklas.io.paths import tensor_filename

    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    folder, _ = create_manifold_folder(
        "local", "mood", "", domain, _author_nodes(["a", "b", "c"]),
    )
    src_tensor = _fit_real_manifold(folder, "src/m", dim=6)
    align = _alignments({
        layer: torch.eye(6) for layer in load_manifold(src_tensor).layers
    })
    whitener = _target_whitener()
    original = ManifoldFolder.update_file_hashes
    failed = False

    def fail_transfer_manifest_once(
        self: ManifoldFolder, *paths: Path,
    ) -> None:
        nonlocal failed
        if not failed and any("_from-" in path.name for path in paths):
            failed = True
            raise RuntimeError("injected post-pair manifest failure")
        original(self, *paths)

    monkeypatch.setattr(
        ManifoldFolder, "update_file_hashes", fail_transfer_manifest_once,
    )
    kwargs = dict(
        folder=folder, from_model="src/m", to_model="tgt/m",
        alignment=align, whitener=whitener,
        source_model_fingerprint="fp:src/m",
        target_model_fingerprint="fp:tgt/m",
    )
    with pytest.raises(RuntimeError, match="post-pair manifest"):
        transfer_manifold(**kwargs)

    target = folder / tensor_filename("tgt/m", transferred_from="src/m")
    assert target.exists()
    assert target.with_suffix(".json").exists()
    assert target.name not in json.loads(
        (folder / "manifold.json").read_text(),
    )["files"]

    retried = transfer_manifold(**kwargs)
    assert retried == target
    loaded = load_manifold(retried)
    assert loaded.metadata["source_model_id"] == "src/m"
    assert loaded.metadata["source_model_fingerprint"] == "fp:src/m"
    ManifoldFolder.load(folder)


# ============================================================ B6b: summary ===


def test_manifold_summary_authored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    folder, _ = create_manifold_folder(
        "local", "mood", "a mood axis", domain, _author_nodes(["a", "b", "c"]),
    )
    _fake_fit_tensor(folder, "google/gemma-3-4b-it")
    summ = manifold_summary(folder)
    assert summ["namespace"] == "local"
    assert summ["name"] == "mood"
    assert summ["description"] == "a mood axis"
    assert summ["source"] == "local"
    assert summ["fit_mode"] == "authored"
    assert summ["is_discover"] is False
    assert summ["domain"]["type"] == "box"
    assert summ["domain_label"] == "box(1d)"
    assert summ["intrinsic_dim"] == 1
    assert summ["min_nodes"] == 3
    assert summ["node_count"] == 3
    assert summ["node_labels"] == ["a", "b", "c"]
    assert len(summ["node_coords"]) == 3
    assert summ["node_roles"] == [None, None, None]
    assert summ["hyperparams"] == {}
    assert summ["fitted_models"] == ["google__gemma-3-4b-it"]
    assert summ["tensor_variants"]["google__gemma-3-4b-it"] == ["raw"]


def test_manifold_summary_discover_unfitted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    folder = create_discover_manifold_folder(
        "local", "personas", "five personas",
        fit_mode="pca",
        node_corpora=_discover_corpora(
            ["pirate", "caveman", "assistant", "scholar", "robot"],
        ),
        hyperparams={"max_dim": 8, "var_threshold": 0.70},
    )
    summ = manifold_summary(folder)
    assert summ["fit_mode"] == "pca"
    assert summ["is_discover"] is True
    # Discover folder carries no top-level geometry on disk.
    assert summ["domain"] == {}
    assert summ["domain_label"] == "discover-pca"
    assert summ["intrinsic_dim"] == 0
    assert summ["min_nodes"] is None
    assert summ["node_coords"] == []
    assert summ["hyperparams"] == {"max_dim": 8, "var_threshold": 0.70}
    assert summ["fitted_models"] == []


def test_manifold_summary_reports_transfer_variant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """A transferred tensor surfaces under tensor_variants as ``from-...``."""
    import torch
    from saklas.core.manifold import load_manifold

    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    folder, _ = create_manifold_folder(
        "local", "mood", "", domain, _author_nodes(["a", "b", "c"]),
    )
    src_tensor = _fit_real_manifold(folder, "src/m", dim=6)
    align = _alignments({L: torch.eye(6) for L in load_manifold(src_tensor).layers})
    transfer_manifold(
        folder, from_model="src/m", to_model="tgt/m", alignment=align,
        source_model_fingerprint="fp:src/m",
        target_model_fingerprint="fp:tgt/m",
        whitener=_target_whitener(),
    )
    summ = manifold_summary(folder)
    assert "src__m" in summ["tensor_variants"]
    assert summ["tensor_variants"]["src__m"] == ["raw"]
    assert summ["tensor_variants"]["tgt__m"] == ["from-src__m"]



# --------------------------------------------- streaming discover writers ---

def test_init_and_append_discover_streaming(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """The streaming skeleton + per-node append round-trips through load."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    labels = ["alpha", "beta", "gamma", "default"]
    folder = init_discover_manifold_folder(
        "local", "stream", "streamed discover", fit_mode="pca",
        labels=labels, hyperparams={"max_dim": 4},
    )
    # Skeleton present, no node corpus yet.
    assert (folder / "manifold.json").exists()
    assert (folder / "nodes").is_dir()
    assert list((folder / "nodes").glob("*.json")) == []

    for i, label in enumerate(labels):
        append_discover_manifold_node(
            folder, i, label, [f"{label} one", f"{label} two"],
        )

    mf = ManifoldFolder.load(folder)
    assert mf.is_discover
    assert mf.node_labels == labels
    assert dict(mf.node_groups())["beta"] == ["beta one", "beta two"]


def test_create_discover_does_not_write_scenarios(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """Current all-at-once discover authoring has no scenario provenance."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    folder = create_discover_manifold_folder(
        "local", "noscn", "d", fit_mode="pca",
        node_corpora={"a": ["x"], "b": ["y"], "default": ["z"]},
    )
    assert not (folder / "scenarios.json").exists()


def test_init_discover_rejects_duplicate_and_bad_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    with pytest.raises(ManifoldFormatError):
        init_discover_manifold_folder(
            "local", "dup", "", fit_mode="pca", labels=["a", "a"],
        )
    with pytest.raises(ManifoldFormatError):
        init_discover_manifold_folder(
            "local", "bad", "", fit_mode="pca", labels=["a", "bad.label"],
        )
    with pytest.raises(ManifoldFormatError):
        init_discover_manifold_folder(
            "local", "wrongmode", "", fit_mode="authored", labels=["a", "b"],
        )


def test_append_discover_rejects_empty_statements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    folder = init_discover_manifold_folder(
        "local", "emptychk", "", fit_mode="pca", labels=["a", "b"],
    )
    with pytest.raises(ManifoldFormatError):
        append_discover_manifold_node(folder, 0, "a", [])
    with pytest.raises(ManifoldFormatError):
        append_discover_manifold_node(folder, 0, "a", ["   "])


# ----------------------------------------- discover generation planner ---

def test_plan_fresh_creates_skeleton_all_pending(tmp_path: Path):
    folder = tmp_path / "m"
    plan = plan_discover_generation(
        folder, "m", "desc", fit_mode="pca",
        labels=["a", "b", "c"], hyperparams={"max_dim": 4},
    )
    assert isinstance(plan, DiscoverGenerationPlan)
    assert not plan.resumed
    assert plan.pending == ("a", "b", "c")
    assert plan.added == ()
    assert plan.index_of == {"a": 0, "b": 1, "c": 2}
    assert (folder / "manifold.json").exists()
    assert (folder / "nodes").is_dir()


def test_plan_force_reset_and_skeleton_share_manifest_transaction(
    tmp_path: Path,
) -> None:
    from saklas.io.manifold_folder import _locked_manifest

    folder = tmp_path / "m"
    first = plan_discover_generation(
        folder, "m", "old", fit_mode="pca", labels=["a", "b"],
    )
    append_discover_manifold_node(folder, first.index_of["a"], "a", ["old"])

    started = threading.Event()
    done = threading.Event()
    errors: list[BaseException] = []

    def reset() -> None:
        started.set()
        try:
            plan_discover_generation(
                folder, "m", "new", fit_mode="pca", labels=["c", "d"],
                force=True,
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)
        finally:
            done.set()

    with _locked_manifest(folder):
        worker = threading.Thread(target=reset)
        worker.start()
        assert started.wait(1.0)
        assert not done.wait(0.1)
    worker.join(timeout=2.0)

    assert not worker.is_alive()
    assert errors == []
    assert done.is_set()
    data = json.loads((folder / "manifold.json").read_text())
    assert data["description"] == "new"
    assert [node["label"] for node in data["nodes"]] == ["c", "d"]
    assert not (folder / "nodes" / "00_a.json").exists()


def test_plan_force_validates_before_removing_prior_folder(tmp_path: Path) -> None:
    folder = tmp_path / "m"
    plan_discover_generation(
        folder, "m", "kept", fit_mode="pca", labels=["a", "b"],
    )
    before = (folder / "manifold.json").read_bytes()

    with pytest.raises(ManifoldFormatError):
        plan_discover_generation(
            folder, "m", "bad", fit_mode="pca", labels=["not.valid", "b"],
            force=True,
        )

    assert (folder / "manifold.json").read_bytes() == before


def test_plan_force_recovers_partial_folder_without_manifest(tmp_path: Path) -> None:
    folder = tmp_path / "m"
    folder.mkdir()
    (folder / "orphan").write_text("partial")

    plan = plan_discover_generation(
        folder, "m", "fresh", fit_mode="pca", labels=["a", "b"], force=True,
    )

    assert plan.pending == ("a", "b")
    assert not (folder / "orphan").exists()
    assert (folder / "manifold.json").exists()


def test_plan_resume_reports_only_missing(tmp_path: Path):
    folder = tmp_path / "m"
    labels = ["a", "b", "c", "d"]
    plan = plan_discover_generation(folder, "m", "d", fit_mode="pca", labels=labels)
    # Simulate a kill after 2 of 4 nodes.
    for lbl in ["a", "b"]:
        append_discover_manifold_node(folder, plan.index_of[lbl], lbl, [f"{lbl} one"])
    plan2 = plan_discover_generation(folder, "m", "d", fit_mode="pca", labels=labels)
    assert plan2.resumed
    assert plan2.pending == ("c", "d")           # only the missing ones
    assert plan2.added == ()
    assert plan2.index_of == {"a": 0, "b": 1, "c": 2, "d": 3}


def test_plan_add_nodes_extends_manifest(tmp_path: Path):
    folder = tmp_path / "m"
    labels = ["a", "b", "c"]
    plan = plan_discover_generation(folder, "m", "d", fit_mode="pca", labels=labels)
    for lbl in labels:
        append_discover_manifold_node(folder, plan.index_of[lbl], lbl, [f"{lbl} x"])
    # Re-plan with a superset roster — add-nodes.
    plan2 = plan_discover_generation(
        folder, "m", "d", fit_mode="pca", labels=labels + ["d", "e"],
    )
    assert plan2.resumed
    assert plan2.added == ("d", "e")
    assert plan2.pending == ("d", "e")            # only the new labels
    mj = json.loads((folder / "manifold.json").read_text())
    assert [n["label"] for n in mj["nodes"]] == ["a", "b", "c", "d", "e"]
    for lbl in ["d", "e"]:
        append_discover_manifold_node(folder, plan2.index_of[lbl], lbl, [f"{lbl} x"])
    mf = ManifoldFolder.load(folder)
    assert mf.node_labels == ["a", "b", "c", "d", "e"]


def test_plan_partial_rejected_by_load_but_resumable(tmp_path: Path):
    """A partial fails strict load yet the lenient planner resumes it."""
    folder = tmp_path / "m"
    plan_discover_generation(folder, "m", "d", fit_mode="pca", labels=["a", "b"])
    append_discover_manifold_node(folder, 0, "a", ["x"])
    with pytest.raises(ManifoldFormatError):
        ManifoldFolder.load(folder)
    plan2 = plan_discover_generation(folder, "m", "d", fit_mode="pca", labels=["a", "b"])
    assert plan2.pending == ("b",)


def test_plan_rejects_non_discover_existing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    folder, _ = create_manifold_folder(
        "local", "authored", "d", domain, _author_nodes(["a", "b", "c"]),
    )
    with pytest.raises(ManifoldFormatError):
        plan_discover_generation(
            folder, "authored", "d", fit_mode="pca", labels=["x", "y"],
        )


def test_bundled_materialization_helpers_ignore_non_json_payloads(tmp_path: Path) -> None:
    """Package-data materialization should not mirror local metadata files."""
    from saklas.io import manifolds as M

    pkg = tmp_path / "pkg"
    (pkg / "nodes").mkdir(parents=True)
    (pkg / "manifold.json").write_text('{"name": "m"}')
    (pkg / "extra.json").write_text('{"extra": []}')
    (pkg / ".DS_Store").write_text("finder")
    (pkg / "README.txt").write_text("not package data")
    (pkg / "nodes" / "00_alpha.json").write_text('["a"]')
    (pkg / "nodes" / ".DS_Store").write_text("finder")
    (pkg / "nodes" / "notes.txt").write_text("not package data")

    target = tmp_path / "target"
    M._copy_bundled_manifold_fresh(pkg, target)

    assert (target / "manifold.json").is_file()
    assert not (target / "extra.json").exists()
    assert (target / "nodes" / "00_alpha.json").is_file()
    assert not (target / ".DS_Store").exists()
    assert not (target / "README.txt").exists()
    assert not (target / "nodes" / ".DS_Store").exists()
    assert not (target / "nodes" / "notes.txt").exists()

    (pkg / "nodes" / "01_beta.json").write_text('["b"]')
    (target / "nodes" / "stale.json").write_text('["old"]')
    (target / "nodes" / ".DS_Store").write_text("old finder")

    M._refresh_all_bundled_nodes(pkg, target)

    assert (target / "nodes" / "00_alpha.json").read_text() == '["a"]'
    assert (target / "nodes" / "01_beta.json").read_text() == '["b"]'
    assert not (target / "nodes" / "stale.json").exists()
    assert not (target / "nodes" / ".DS_Store").exists()


def test_bundled_refresh_ignores_local_fit_proofs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fit proofs in ``files`` must not read as a bundle update.

    A fit appends its tensor/sidecar hashes to ``manifold.json.files``
    (``update_file_hashes``).  The bundle-drift comparison has to ignore
    that local state — before the fix it didn't, so the next launch saw
    "manifest changed", refreshed the manifest from the bundle, and wiped
    the proofs, orphaning every fitted tensor of every bundled manifold.
    """
    from saklas.io import manifolds as M

    root = tmp_path / "bundled"
    pkg = root / "axis"
    (pkg / "nodes").mkdir(parents=True)
    bundled_manifest = {
        "format_version": M.MANIFOLD_FORMAT_VERSION,
        "name": "axis",
        "fit_mode": "pca",
        "nodes": [{"label": "pos"}, {"label": "neg"}],
    }
    (pkg / "manifold.json").write_text(json.dumps(bundled_manifest))
    (pkg / "nodes" / "00_pos.json").write_text('["a"]')
    (pkg / "nodes" / "01_neg.json").write_text('["b"]')
    monkeypatch.setattr(
        M._resources, "files",
        lambda package: root if package == "saklas.data.manifolds" else None,
    )

    default_dir = tmp_path / "default"
    M._materialize_one_bundled_manifold(default_dir, "axis")
    target = default_dir / "axis"

    # Simulate a fit: tensor + sidecar land, update_file_hashes records them.
    (target / "model.safetensors").write_bytes(b"tensor-bytes")
    (target / "model.json").write_text("{}")
    on_disk = json.loads((target / "manifold.json").read_text())
    on_disk["files"] = {
        "model.safetensors": "aa" * 32,
        "model.json": "bb" * 32,
    }
    (target / "manifold.json").write_text(json.dumps(on_disk))

    M._materialize_one_bundled_manifold(default_dir, "axis")

    refreshed = json.loads((target / "manifold.json").read_text())
    assert refreshed["files"] == on_disk["files"], "proofs were clobbered"
    assert not (target / "manifold.json.bak").exists(), (
        "a proofs-only difference must not be treated as a bundle update"
    )


def test_bundled_refresh_carries_tensor_proofs_forward(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A genuine bundle update keeps proofs for surviving fitted artifacts.

    Fitted tensors deliberately stay put across bundle updates (the
    ``nodes_sha256`` staleness check owns re-fit decisions); their
    integrity proofs must travel with them or the strict loader refuses
    the files.  Entries for since-deleted files and for bundle-shipped
    files are dropped.
    """
    from saklas.io import manifolds as M

    root = tmp_path / "bundled"
    pkg = root / "axis"
    (pkg / "nodes").mkdir(parents=True)

    def manifest(description: str) -> str:
        return json.dumps({
            "format_version": M.MANIFOLD_FORMAT_VERSION,
            "name": "axis",
            "fit_mode": "pca",
            "description": description,
            "nodes": [{"label": "pos"}, {"label": "neg"}],
        })

    (pkg / "manifold.json").write_text(manifest("v1"))
    (pkg / "nodes" / "00_pos.json").write_text('["a"]')
    (pkg / "nodes" / "01_neg.json").write_text('["b"]')
    monkeypatch.setattr(
        M._resources, "files",
        lambda package: root if package == "saklas.data.manifolds" else None,
    )

    default_dir = tmp_path / "default"
    M._materialize_one_bundled_manifold(default_dir, "axis")
    target = default_dir / "axis"

    (target / "model.safetensors").write_bytes(b"tensor-bytes")
    (target / "model.json").write_text("{}")
    on_disk = json.loads((target / "manifold.json").read_text())
    on_disk["files"] = {
        "model.safetensors": "aa" * 32,
        "model.json": "bb" * 32,
        "gone.safetensors": "cc" * 32,   # file no longer on disk
        "manifold.json": "dd" * 32,      # bundle-shipped name — never carried
    }
    (target / "manifold.json").write_text(json.dumps(on_disk))

    # Ship a real bundle update.
    (pkg / "manifold.json").write_text(manifest("v2"))
    with pytest.warns(UserWarning, match="refreshed default/axis"):
        M._materialize_one_bundled_manifold(default_dir, "axis")

    refreshed = json.loads((target / "manifold.json").read_text())
    assert refreshed["description"] == "v2"
    assert refreshed["files"] == {
        "model.safetensors": "aa" * 32,
        "model.json": "bb" * 32,
    }
    assert (target / "manifold.json.bak").exists()


def test_bundled_manifold_names_skips_incomplete_package_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A partial generation directory must not materialize as a bundled manifold."""
    from saklas.io import manifolds as M

    root = tmp_path / "bundled"
    complete = root / "complete"
    (complete / "nodes").mkdir(parents=True)
    (complete / "manifold.json").write_text(json.dumps({
        "nodes": [{"label": "alpha"}, {"label": "beta"}],
    }))
    (complete / "nodes" / "00_alpha.json").write_text('["a"]')
    (complete / "nodes" / "01_beta.json").write_text('["b"]')

    partial = root / "partial"
    (partial / "nodes").mkdir(parents=True)
    (partial / "manifold.json").write_text(json.dumps({
        "nodes": [{"label": "alpha"}, {"label": "beta"}],
    }))
    (partial / "nodes" / "00_alpha.json").write_text('["a"]')

    junk = root / "junk"
    junk.mkdir()
    (junk / "manifold.json").write_text("{not json")

    monkeypatch.setattr(
        M._resources, "files",
        lambda package: root if package == "saklas.data.manifolds" else None,
    )

    assert M.bundled_manifold_names() == ["complete"]


# =================================================== baked (corpus-less) ===
#
# A baked manifold is a pre-fitted, corpus-less artifact (merge output /
# imported control vector): geometry frozen in the tensor, no ``nodes/``
# corpus, never re-fits.


def _baked_manifold(name: str = "merged", n_layers: int = 3):
    """Build an affine R=1 Manifold (the merge/import shape) via the fold."""
    from saklas.core.vectors import fold_directions_to_subspace

    directions = {i: torch.randn(8) for i in range(n_layers)}
    return fold_directions_to_subspace(name, directions, None, label=name), directions


def test_baked_manifold_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    from saklas.core.manifold import load_manifold
    from saklas.core.vectors import folded_vector_directions

    manifold, directions = _baked_manifold("merged")
    folder, mf = create_baked_manifold_folder(
        "local", "merged", "a merged vector", manifold, "test/model",
        method="merge", model_fingerprint="fp:test/model",
    )
    assert mf.fit_mode == "baked"
    assert mf.is_discover is False
    assert mf.node_labels == ["merged"]
    assert mf.files, "files manifest should be back-filled"

    # No nodes/ corpus on disk.
    assert not (folder / "nodes").exists()

    # Reloads clean, with a stable provenance hash.
    reloaded = ManifoldFolder.load(folder)
    assert reloaded.fit_mode == "baked"
    assert reloaded.nodes_sha256() == mf.nodes_sha256()

    # The tensor folds back to the per-layer directions it was baked from.
    (tensor,) = list(folder.glob("*.safetensors"))
    folded = folded_vector_directions(load_manifold(tensor))
    assert set(folded) == set(directions)


def test_baked_publication_hashes_each_new_file_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    from saklas.io import manifold_folder as folder_module, packs

    real_hash = folder_module.hash_file
    hashed: list[Path] = []

    def count_hash(path: Path) -> str:
        hashed.append(Path(path))
        return real_hash(Path(path))

    monkeypatch.setattr(folder_module, "hash_file", count_hash)
    monkeypatch.setattr(packs, "hash_file", count_hash)
    manifold, _ = _baked_manifold("merged")

    create_baked_manifold_folder(
        "local", "merged", "", manifold, "test/model", method="merge",
        model_fingerprint="fp:test/model",
    )

    assert [path.name for path in hashed] == [
        "test__model.safetensors", "test__model.json",
    ]


def test_baked_first_publication_retry_repairs_unproven_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    manifold, _directions = _baked_manifold("merged")
    real_update = ManifoldFolder.update_file_hashes
    failed = False

    def fail_once(self: ManifoldFolder, *paths: Path) -> None:
        nonlocal failed
        if not failed:
            failed = True
            raise OSError("injected baked manifest failure")
        real_update(self, *paths)

    monkeypatch.setattr(ManifoldFolder, "update_file_hashes", fail_once)
    with pytest.raises(OSError, match="injected"):
        create_baked_manifold_folder(
            "local", "merged", "", manifold, "test/model",
            method="merge", model_fingerprint="fp:test/model",
        )

    folder, _mf = create_baked_manifold_folder(
        "local", "merged", "", manifold, "test/model",
        method="merge", model_fingerprint="fp:test/model",
    )
    loaded = ManifoldFolder.load(folder)
    assert loaded.tensor_models() == ["test__model"]


def test_baked_force_replaces_manifestless_partial_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    from saklas.io.paths import manifold_dir

    folder = manifold_dir("local", "merged")
    folder.mkdir(parents=True)
    (folder / "stale.safetensors").write_bytes(b"partial")
    (folder / "stale.json").write_text("{bad sidecar")
    manifold, _directions = _baked_manifold("merged")

    with pytest.raises(FileExistsError, match="incomplete"):
        create_baked_manifold_folder(
            "local", "merged", "", manifold, "test/model",
            method="merge", model_fingerprint="fp:test/model",
        )

    created, _mf = create_baked_manifold_folder(
        "local", "merged", "", manifold, "test/model",
        method="merge", model_fingerprint="fp:test/model", force=True,
    )
    assert not (created / "stale.safetensors").exists()
    assert ManifoldFolder.load(created).tensor_models() == ["test__model"]


def test_baked_manifold_clear_and_scoped_refresh_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    manifold, _ = _baked_manifold("merged")
    create_baked_manifold_folder(
        "local", "merged", "", manifold, "test/model", method="merge",
        model_fingerprint="fp:test/model",
    )
    # Both tensor-deleting ops refuse — a baked manifold can't re-fit, so
    # clearing its tensor would destroy the only copy of its geometry.
    with pytest.raises(BakedManifoldError, match="rm"):
        clear_manifold_tensors("local", "merged")
    with pytest.raises(BakedManifoldError):
        refresh_manifold("local", "merged", model_scope="test/model")


def test_baked_manifold_node_groups_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    manifold, _ = _baked_manifold("merged")
    _, mf = create_baked_manifold_folder(
        "local", "merged", "", manifold, "test/model", method="merge",
        model_fingerprint="fp:test/model",
    )
    with pytest.raises(ManifoldFormatError, match="no node corpus"):
        mf.node_groups()


def test_baked_manifold_rejects_coords_on_node(tmp_path: Path):
    """A baked node is label-only — coords would be ambiguous against the
    frozen tensor geometry, so the loader refuses them."""
    folder = tmp_path / "merged"
    (folder).mkdir()
    payload = {
        "format_version": MANIFOLD_FORMAT_VERSION,
        "name": "merged",
        "description": "",
        "fit_mode": "baked",
        "domain": {"type": "custom", "embed_dim": 1},
        "nodes": [{"label": "merged", "coords": [1.0]}],
        "files": {},
    }
    (folder / "manifold.json").write_text(json.dumps(payload))
    with pytest.raises(ManifoldFormatError, match="must not carry 'coords'"):
        ManifoldFolder.load(folder)


def test_baked_manifold_requires_tensor(tmp_path: Path):
    """A tensor-less baked folder is corrupt — the tensor is its only geometry."""
    folder = tmp_path / "merged"
    folder.mkdir()
    payload = {
        "format_version": MANIFOLD_FORMAT_VERSION,
        "name": "merged",
        "description": "",
        "fit_mode": "baked",
        "domain": {"type": "custom", "embed_dim": 1},
        "nodes": [{"label": "merged"}],
        "files": {},
    }
    (folder / "manifold.json").write_text(json.dumps(payload))
    with pytest.raises(ManifoldFormatError, match="no fitted tensor"):
        ManifoldFolder.load(folder)


def test_baked_manifold_requires_domain(tmp_path: Path):
    folder = tmp_path / "merged"
    folder.mkdir()
    payload = {
        "format_version": MANIFOLD_FORMAT_VERSION,
        "name": "merged",
        "description": "",
        "fit_mode": "baked",
        "nodes": [{"label": "merged"}],
        "files": {},
    }
    (folder / "manifold.json").write_text(json.dumps(payload))
    with pytest.raises(ManifoldFormatError, match="needs a 'domain'"):
        ManifoldFolder.load(folder)
