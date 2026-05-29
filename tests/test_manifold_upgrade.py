"""Tests for scripts/upgrade_manifolds.py — the pre-v3 -> v3 converter."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from saklas.io.manifolds import MANIFOLD_FORMAT_VERSION, ManifoldFolder

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "upgrade_manifolds.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("upgrade_manifolds", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _author_legacy(root: Path, *, cyclic: bool, labels: list[str],
                    with_tensor: bool = False) -> Path:
    """Hand-author a pre-v3 (1-D cyclic-spline) manifold folder."""
    folder = root / "legacy"
    (folder / "nodes").mkdir(parents=True)
    for idx, label in enumerate(labels):
        (folder / "nodes" / f"{idx:02d}_{label}.json").write_text(
            json.dumps([f"{label} {i}" for i in range(3)])
        )
    if with_tensor:
        (folder / "old-model.safetensors").write_bytes(b"stale-cubic-tensor")
        (folder / "old-model.json").write_text(json.dumps({"cyclic": cyclic}))
    (folder / "manifold.json").write_text(json.dumps({
        "format_version": 2,
        "name": "legacy",
        "description": "a legacy manifold",
        "cyclic": cyclic,
        "nodes": labels,
        "files": {},
    }))
    return folder


def test_upgrade_cyclic_to_box_domain(tmp_path: Path):
    mod = _load_script()
    folder = _author_legacy(
        tmp_path, cyclic=True, labels=["a", "b", "c", "d"],
    )
    assert mod.upgrade_manifold(folder) is True

    data = json.loads((folder / "manifold.json").read_text())
    assert data["format_version"] == MANIFOLD_FORMAT_VERSION
    assert "cyclic" not in data
    assert data["domain"]["type"] == "box"
    assert data["domain"]["axes"][0]["periodic"] is True
    # nodes are now objects with coords
    assert all("coords" in n for n in data["nodes"])
    coords = [n["coords"][0] for n in data["nodes"]]
    assert coords == [0.0, 0.25, 0.5, 0.75]   # evenly spaced, no wrap seam
    # the upgraded folder loads cleanly under the v3 reader
    ManifoldFolder.load(folder)


def test_upgrade_sequential_to_open_axis(tmp_path: Path):
    mod = _load_script()
    folder = _author_legacy(
        tmp_path, cyclic=False, labels=["a", "b", "c", "d", "e"],
    )
    mod.upgrade_manifold(folder)
    data = json.loads((folder / "manifold.json").read_text())
    assert data["domain"]["axes"][0]["periodic"] is False
    coords = [n["coords"][0] for n in data["nodes"]]
    assert coords[0] == 0.0 and coords[-1] == 1.0


def test_upgrade_removes_stale_tensors(tmp_path: Path):
    mod = _load_script()
    folder = _author_legacy(
        tmp_path, cyclic=True, labels=["a", "b", "c"], with_tensor=True,
    )
    assert (folder / "old-model.safetensors").exists()
    mod.upgrade_manifold(folder)
    # stale cubic-spline tensors are deleted — they cannot be loaded by v3
    assert not (folder / "old-model.safetensors").exists()
    assert not (folder / "old-model.json").exists()


def test_upgrade_is_idempotent(tmp_path: Path):
    mod = _load_script()
    folder = _author_legacy(
        tmp_path, cyclic=True, labels=["a", "b", "c", "d"],
    )
    assert mod.upgrade_manifold(folder) is True
    # second run: already v3, no change
    assert mod.upgrade_manifold(folder) is False
