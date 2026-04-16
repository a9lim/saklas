#!/usr/bin/env python3
"""Rewrite saklas pack folders to the current ``format_version``.

Usage:
    python scripts/upgrade_packs.py <pack_folder>
    python scripts/upgrade_packs.py --all

``--all`` walks ``~/.saklas/vectors/`` (or ``$SAKLAS_HOME/vectors``) and
upgrades every concept folder it finds. Per-folder invocation takes a
path to a concept directory containing a ``pack.json``.

Upgrade steps per folder:
  1. Load ``pack.json`` as a dict, stamp ``format_version`` to the
     current value.
  2. For every ``*.safetensors`` sidecar (``<stem>.json`` next to it),
     stamp ``format_version`` on the sidecar too.
  3. Recompute the ``files`` hash map from on-disk contents so tampering
     after the bump still trips integrity checks at load time.
  4. Write both files back in place.

Idempotent — running twice is safe. Prints one line per folder touched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


PACK_FORMAT_VERSION = 2


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_folder(folder: Path) -> dict[str, str]:
    return {
        p.name: _sha256(p)
        for p in sorted(folder.iterdir())
        if p.is_file() and p.name != "pack.json"
    }


def upgrade_pack(folder: Path) -> bool:
    pj = folder / "pack.json"
    if not pj.is_file():
        print(f"skip {folder}: no pack.json", file=sys.stderr)
        return False
    try:
        data = json.loads(pj.read_text())
    except json.JSONDecodeError as e:
        print(f"skip {folder}: pack.json not json ({e})", file=sys.stderr)
        return False

    changed = False
    if data.get("format_version") != PACK_FORMAT_VERSION:
        data["format_version"] = PACK_FORMAT_VERSION
        changed = True

    # Stamp every sidecar for safetensors tensors in the folder.
    for sidecar in sorted(folder.glob("*.json")):
        if sidecar.name == "pack.json":
            continue
        try:
            sc = json.loads(sidecar.read_text())
        except json.JSONDecodeError:
            continue
        if sc.get("format_version") != PACK_FORMAT_VERSION:
            sc["format_version"] = PACK_FORMAT_VERSION
            sidecar.write_text(json.dumps(sc, indent=2) + "\n")
            changed = True

    # Recompute files map so upgraded sidecars hash correctly.
    new_files = _hash_folder(folder)
    if data.get("files") != new_files:
        data["files"] = new_files
        changed = True

    if changed:
        pj.write_text(json.dumps(data, indent=2) + "\n")
        print(f"upgraded {folder}")
    else:
        print(f"ok       {folder}")
    return changed


def _vectors_root() -> Path:
    try:
        from saklas.io.paths import vectors_dir
        return vectors_dir()
    except Exception:
        import os
        home = os.environ.get("SAKLAS_HOME")
        base = Path(home) if home else Path.home() / ".saklas"
        return base / "vectors"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folder", nargs="?", type=Path,
                    help="concept folder with a pack.json")
    ap.add_argument("--all", action="store_true",
                    help="walk ~/.saklas/vectors/ and upgrade every concept")
    args = ap.parse_args(argv)

    if args.all:
        root = _vectors_root()
        if not root.exists():
            print(f"no vectors dir at {root}", file=sys.stderr)
            return 1
        # Two levels: <root>/<namespace>/<concept>/pack.json
        count = 0
        for ns in sorted(p for p in root.iterdir() if p.is_dir()):
            for concept in sorted(p for p in ns.iterdir() if p.is_dir()):
                if (concept / "pack.json").is_file():
                    upgrade_pack(concept)
                    count += 1
        print(f"scanned {count} concept folders under {root}")
        return 0

    if args.folder is None:
        ap.error("pass a folder or --all")
    upgrade_pack(args.folder.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
