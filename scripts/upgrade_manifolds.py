#!/usr/bin/env python3
"""Convert pre-v3 manifold folders to the arbitrary-topology v3 format.

Usage:
    python scripts/upgrade_manifolds.py <manifold_folder>
    python scripts/upgrade_manifolds.py --all

Pre-v3 manifolds were a 1-D sequence of labeled nodes with a ``cyclic``
flag and a cubic-spline fit.  v3 replaces that with a ``domain`` spec and
explicit per-node authoring ``coords``.  This converter rewrites
``manifold.json`` in place:

  1. The old ``cyclic`` flag becomes a one-axis ``box`` domain — the axis
     is periodic iff ``cyclic`` was true.
  2. Each node gets an evenly-spaced coordinate on that axis (open axis
     over ``[0, 1]``; periodic axis at ``i / K`` so no node lands on the
     wrap seam).
  3. ``nodes`` becomes a list of ``{"label", "coords"}`` objects and
     ``format_version`` is stamped to 3.
  4. Every fitted ``*.safetensors`` tensor + ``.json`` sidecar is
     **deleted** — the old tensors carry the cubic-spline representation
     (``t_knots`` / ``coords`` / ``spline_M``) and cannot be loaded by the
     v3 ``load_manifold``.  Re-fit with ``saklas vector manifold fit``.
  5. ``files`` is recomputed (empty after the tensor deletion).

Idempotent — a folder already on v3 is left untouched.  Prints one line
per folder.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

MANIFOLD_FORMAT_VERSION = 3


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def upgrade_manifold(folder: Path) -> bool:
    mj = folder / "manifold.json"
    if not mj.is_file():
        print(f"skip {folder}: no manifold.json", file=sys.stderr)
        return False
    try:
        data = json.loads(mj.read_text())
    except json.JSONDecodeError as e:
        print(f"skip {folder}: manifold.json not json ({e})", file=sys.stderr)
        return False

    if isinstance(data.get("domain"), dict) and "cyclic" not in data:
        print(f"ok       {folder}")
        return False

    nodes = data.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        print(f"skip {folder}: no 'nodes' list", file=sys.stderr)
        return False
    if nodes and isinstance(nodes[0], dict):
        # Already node-objects but still carrying a stale shape — bail
        # rather than guessing.
        print(f"skip {folder}: nodes already objects, not a pre-v3 folder",
              file=sys.stderr)
        return False

    cyclic = bool(data.get("cyclic", False))
    k = len(nodes)
    if cyclic:
        coords_seq = [[i / k] for i in range(k)]
        axis = {"name": "t", "periodic": True, "period": 1.0}
    else:
        coords_seq = [
            [i / (k - 1) if k > 1 else 0.0] for i in range(k)
        ]
        axis = {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}

    data["domain"] = {"type": "box", "axes": [axis]}
    data["nodes"] = [
        {"label": label, "coords": coords_seq[i]}
        for i, label in enumerate(nodes)
    ]
    data.pop("cyclic", None)
    data["format_version"] = MANIFOLD_FORMAT_VERSION

    # Drop stale fitted tensors — the cubic-spline representation is
    # unloadable by the v3 loader.  A re-fit is one command.
    removed = 0
    for tensor in sorted(folder.glob("*.safetensors")):
        sidecar = tensor.with_suffix(".json")
        tensor.unlink()
        if sidecar.exists():
            sidecar.unlink()
        removed += 1

    data["files"] = {
        p.name: _sha256(p)
        for p in sorted(folder.iterdir())
        if p.is_file() and p.name != "manifold.json"
    }

    mj.write_text(json.dumps(data, indent=2) + "\n")
    note = f" — {removed} stale tensor(s) removed, re-fit needed" if removed else ""
    print(f"upgraded {folder}{note}")
    return True


def _manifolds_root() -> Path:
    try:
        from saklas.io.paths import manifolds_dir
        return manifolds_dir()
    except Exception:
        import os
        home = os.environ.get("SAKLAS_HOME")
        base = Path(home) if home else Path.home() / ".saklas"
        return base / "manifolds"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folder", nargs="?", type=Path,
                    help="manifold folder with a manifold.json")
    ap.add_argument("--all", action="store_true",
                    help="walk ~/.saklas/manifolds/ and upgrade every manifold")
    args = ap.parse_args(argv)

    if args.all:
        root = _manifolds_root()
        if not root.exists():
            print(f"no manifolds dir at {root}", file=sys.stderr)
            return 1
        # Two levels: <root>/<namespace>/<name>/manifold.json
        count = 0
        for ns in sorted(p for p in root.iterdir() if p.is_dir()):
            for mdir in sorted(p for p in ns.iterdir() if p.is_dir()):
                if (mdir / "manifold.json").is_file():
                    count += 1
                    upgrade_manifold(mdir)
        if count == 0:
            print(f"no manifold folders under {root}")
        return 0

    if args.folder is None:
        ap.error("pass a manifold folder or --all")
    upgrade_manifold(args.folder)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
