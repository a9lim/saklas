#!/usr/bin/env python3
"""Migrate legacy ``vectors/`` packs to the 4.0 manifold representation.

Usage:
    python scripts/upgrade_packs.py <pack_folder>
    python scripts/upgrade_packs.py --all [-m MODEL] [--keep-source] [--restamp-only]

A steering vector *is* the ``K = 2`` case of a flat affine subspace, so the
canonical artifact for a concept is now a 2-node ``pca`` manifold under
``manifolds/``, not a baked DiM tensor under ``vectors/`` (``PACK_FORMAT_VERSION``
bumped to 3).  This script migrates an existing install:

  * **statements-bearing folder** → ported to ``manifolds/<ns>/<name>/`` via
    :func:`saklas.io.manifolds.port_legacy_vector_folder` (no tensors carried —
    they re-fit lazily / on demand), and the source ``vectors/`` folder removed
    unless ``--keep-source``.  With ``-m MODEL`` the ported manifold is fitted
    immediately so it's steer-ready.
  * **tensor-only folder** (no ``statements.json`` — HF/GGUF imports that can't
    re-fit) → ``pack.json`` + every ``*.safetensors`` sidecar re-stamped to the
    current ``format_version`` and the ``files`` hash map recomputed, so it keeps
    loading through the residual autoload path.

``--restamp-only`` skips porting and just re-stamps (the pre-4.0 behavior).
Idempotent — a folder already migrated is left alone.  Prints one line per
folder touched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

from saklas.io.packs import PACK_FORMAT_VERSION


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


def restamp_pack(folder: Path) -> bool:
    """Re-stamp ``pack.json`` + sidecars to the current ``format_version``.

    For tensor-only / general folders that can't be ported to a manifold.
    Returns ``True`` when anything changed.
    """
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

    # Stamp the sidecar of every tensor (``<stem>.safetensors`` → ``<stem>.json``).
    # Iterate tensors, not ``*.json`` — statements.json (a list) and
    # scenarios.json (a non-sidecar dict) are not sidecars and must be left
    # alone.
    for tensor in sorted(folder.glob("*.safetensors")):
        sidecar = tensor.with_suffix(".json")
        if not sidecar.exists():
            continue
        try:
            sc = json.loads(sidecar.read_text())
        except json.JSONDecodeError:
            continue
        if not isinstance(sc, dict):
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
        print(f"restamped {folder}")
    else:
        print(f"ok        {folder}")
    return changed


def port_pack(
    folder: Path, namespace: str, *, keep_source: bool,
) -> tuple[str, str] | None:
    """Port a statements-bearing legacy folder to a 2-node ``pca`` manifold.

    Returns ``(namespace, name)`` of the ported manifold (so the caller can
    fit it), or ``None`` when the folder has no statements to port.  Idempotent
    on a manifold that already exists (treated as already-ported).
    """
    from saklas.io.manifolds import port_legacy_vector_folder
    from saklas.io.paths import manifold_dir

    name = folder.name
    if not (folder / "statements.json").exists():
        return None

    already = (manifold_dir(namespace, name) / "manifold.json").exists()
    if already:
        print(f"ok        {namespace}/{name}: manifold already exists")
    else:
        try:
            port_legacy_vector_folder(folder, namespace=namespace, force=False)
        except Exception as e:
            print(f"skip {folder}: port failed ({e})", file=sys.stderr)
            return None
        print(f"ported    {namespace}/{name}  vectors/ -> manifolds/")

    if not keep_source:
        shutil.rmtree(folder)
        print(f"removed   {folder} (source migrated)")
    return (namespace, name)


def _vectors_root() -> Path:
    from saklas.io.paths import vectors_dir
    return vectors_dir()


def _fit_ported(ported: list[tuple[str, str]], model_id: str) -> None:
    """Load ``model_id`` once and fit every ported manifold so it's steer-ready."""
    if not ported:
        return
    from saklas import SaklasSession
    from saklas.io.paths import manifold_dir

    print(f"\nfitting {len(ported)} ported manifold(s) on {model_id} ...")
    with SaklasSession.from_pretrained(model_id, device="auto") as session:
        for ns, name in ported:
            folder = manifold_dir(ns, name)
            try:
                session.fit(folder)
                print(f"fitted    {ns}/{name}")
            except Exception as e:
                print(f"skip fit  {ns}/{name}: {e}", file=sys.stderr)


def migrate_folder(
    folder: Path, namespace: str, *, keep_source: bool, restamp_only: bool,
) -> tuple[str, str] | None:
    """Migrate one concept folder; returns the ported ``(ns, name)`` or ``None``."""
    if restamp_only:
        restamp_pack(folder)
        return None
    ported = port_pack(folder, namespace, keep_source=keep_source)
    if ported is not None:
        return ported
    # No statements to port — keep it loadable by re-stamping.
    restamp_pack(folder)
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folder", nargs="?", type=Path,
                    help="concept folder with a pack.json")
    ap.add_argument("--all", action="store_true",
                    help="walk ~/.saklas/vectors/ and migrate every concept")
    ap.add_argument("-m", "--model", default=None,
                    help="fit each ported manifold on this model (steer-ready)")
    ap.add_argument("--keep-source", action="store_true",
                    help="keep the legacy vectors/ folder after a successful port")
    ap.add_argument("--restamp-only", action="store_true",
                    help="never port; just re-stamp pack.json/sidecars to the "
                         "current format_version")
    args = ap.parse_args(argv)

    if args.all:
        root = _vectors_root()
        if not root.exists():
            print(f"no vectors dir at {root}", file=sys.stderr)
            return 1
        ported: list[tuple[str, str]] = []
        count = 0
        # Two levels: <root>/<namespace>/<concept>/pack.json
        for ns in sorted(p for p in root.iterdir() if p.is_dir()):
            for concept in sorted(p for p in ns.iterdir() if p.is_dir()):
                if (concept / "pack.json").is_file():
                    count += 1
                    hit = migrate_folder(
                        concept, ns.name,
                        keep_source=args.keep_source,
                        restamp_only=args.restamp_only,
                    )
                    if hit is not None:
                        ported.append(hit)
        print(f"scanned {count} concept folders under {root}")
        if args.model:
            _fit_ported(ported, args.model)
        return 0

    if args.folder is None:
        ap.error("pass a folder or --all")
    folder = args.folder.resolve()
    hit = migrate_folder(
        folder, folder.parent.name,
        keep_source=args.keep_source,
        restamp_only=args.restamp_only,
    )
    if args.model and hit is not None:
        _fit_ported([hit], args.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
