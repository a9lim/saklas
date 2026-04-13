"""One-off build script: convert saklas/datasets/*.json into the bundled
concept-folder layout under saklas/data/vectors/.

Old schema: {name, description, category, pairs}
New layout:
    saklas/data/vectors/<name>/pack.json       (metadata)
    saklas/data/vectors/<name>/statements.json (bare pairs array)

Run once. Commit the outputs. Safe to re-run (overwrites).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "saklas" / "datasets"
DST = REPO / "saklas" / "data" / "vectors"
DEFAULTS = REPO / "saklas" / "probes" / "defaults.json"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    if not SRC.is_dir():
        print(f"Source datasets dir missing: {SRC}", file=sys.stderr)
        return 1
    with open(DEFAULTS) as f:
        cat_map = json.load(f)
    name_to_cat = {name: cat for cat, names in cat_map.items() for name in names}

    DST.mkdir(parents=True, exist_ok=True)
    count = 0
    for src in sorted(SRC.glob("*.json")):
        name = src.stem
        with open(src) as f:
            data = json.load(f)
        if not isinstance(data, dict) or "pairs" not in data:
            print(f"  skip (unexpected schema): {src.name}")
            continue

        concept_dir = DST / name
        concept_dir.mkdir(parents=True, exist_ok=True)

        statements = data["pairs"]
        stmts_bytes = (json.dumps(statements, indent=2) + "\n").encode()
        (concept_dir / "statements.json").write_bytes(stmts_bytes)

        category = data.get("category") or name_to_cat.get(name, "")
        tags = [category] if category else []
        pack = {
            "name": name,
            "description": data.get("description", ""),
            "version": "1.0.0",
            "license": "AGPL-3.0-or-later",
            "tags": tags,
            "recommended_alpha": 0.5,
            "source": "bundled",
            "files": {
                "statements.json": sha256(stmts_bytes),
            },
            "signature": None,
            "signature_method": None,
        }
        (concept_dir / "pack.json").write_text(json.dumps(pack, indent=2) + "\n")
        count += 1

    print(f"Wrote {count} concept folders under {DST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
