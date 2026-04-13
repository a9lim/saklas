"""Pack format: load, validate, and write concept folders under ~/.saklas/vectors/.

A concept folder contains:
    pack.json          # human-editable metadata (required)
    statements.json    # contrastive pair list (optional)
    <model>.safetensors + <model>.json   # extracted tensor + slim sidecar (optional, 0..N)

At least one of statements.json or a tensor must be present.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


NAME_REGEX = re.compile(r"^[a-z][a-z0-9-]{0,63}$")
_REQUIRED_PACK_FIELDS = (
    "name", "description", "version", "license",
    "tags", "recommended_alpha", "source", "files",
)


class PackFormatError(ValueError):
    """Raised when a pack folder or pack.json is malformed."""


@dataclass
class PackMetadata:
    name: str
    description: str
    version: str
    license: str
    tags: list[str]
    recommended_alpha: float
    source: str
    files: dict[str, str]
    long_description: str = ""
    signature: Optional[str] = None
    signature_method: Optional[str] = None

    @classmethod
    def load(cls, folder: Path) -> "PackMetadata":
        pj = folder / "pack.json"
        if not pj.exists():
            raise PackFormatError(f"pack.json missing in {folder}")
        try:
            with open(pj) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise PackFormatError(f"pack.json parse error in {folder}: {e}") from e

        for k in _REQUIRED_PACK_FIELDS:
            if k not in data:
                raise PackFormatError(f"pack.json missing required field '{k}' in {folder}")

        name = data["name"]
        if not isinstance(name, str) or not NAME_REGEX.match(name):
            raise PackFormatError(
                f"pack.json name '{name}' invalid; must match {NAME_REGEX.pattern}"
            )

        return cls(
            name=name,
            description=data["description"],
            long_description=data.get("long_description", ""),
            version=data["version"],
            license=data["license"],
            tags=list(data["tags"]),
            recommended_alpha=float(data["recommended_alpha"]),
            source=data["source"],
            files=dict(data["files"]),
            signature=data.get("signature"),
            signature_method=data.get("signature_method"),
        )

    def to_dict(self) -> dict:
        out: dict = {
            "name": self.name,
            "description": self.description,
        }
        if self.long_description:
            out["long_description"] = self.long_description
        out.update({
            "version": self.version,
            "license": self.license,
            "tags": self.tags,
            "recommended_alpha": self.recommended_alpha,
            "source": self.source,
            "files": self.files,
            "signature": self.signature,
            "signature_method": self.signature_method,
        })
        return out

    def write(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder / "pack.json", "w") as f:
            json.dump(self.to_dict(), f, indent=2)
            f.write("\n")


@dataclass
class Sidecar:
    method: str
    scores: dict[int, float]
    saklas_version: str
    statements_sha256: Optional[str] = None
    components: Optional[dict[str, dict]] = None

    @classmethod
    def load(cls, path: Path) -> "Sidecar":
        with open(path) as f:
            data = json.load(f)
        return cls(
            method=data["method"],
            scores={int(k): float(v) for k, v in data["scores"].items()},
            saklas_version=data["saklas_version"],
            statements_sha256=data.get("statements_sha256"),
            components=data.get("components"),
        )

    def to_dict(self) -> dict:
        out: dict = {
            "method": self.method,
            "scores": {str(k): v for k, v in self.scores.items()},
            "saklas_version": self.saklas_version,
        }
        if self.statements_sha256 is not None:
            out["statements_sha256"] = self.statements_sha256
        if self.components is not None:
            out["components"] = self.components
        return out

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
            f.write("\n")
