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
from importlib import resources as _resources
from pathlib import Path
from typing import Optional, Sequence

from saklas.core.errors import SaklasError


NAME_REGEX = re.compile(r"^[a-z][a-z0-9._-]{0,63}$")
_REQUIRED_PACK_FIELDS = (
    "name", "description", "version", "license",
    "tags", "recommended_alpha", "source", "files",
)

# Current on-disk pack format version. Readers refuse anything lower; the
# number bumps any time the sidecar/pack.json shape changes in a way that
# old readers cannot safely ignore. See scripts/upgrade_packs.py.
PACK_FORMAT_VERSION = 2


class PackFormatError(ValueError, SaklasError):
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
    format_version: int = PACK_FORMAT_VERSION

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

        fmt_ver = data.get("format_version", 1)
        if not isinstance(fmt_ver, int) or fmt_ver < PACK_FORMAT_VERSION:
            raise PackFormatError(
                f"pack at {folder} has format_version={fmt_ver!r}, "
                f"need >= {PACK_FORMAT_VERSION}. "
                f"Run: python scripts/upgrade_packs.py {folder}"
            )

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
            format_version=fmt_ver,
        )

    def to_dict(self) -> dict:
        out: dict = {
            "name": self.name,
            "description": self.description,
            "format_version": self.format_version,
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
    saklas_version: str
    statements_sha256: Optional[str] = None
    components: Optional[dict[str, dict]] = None

    @classmethod
    def load(cls, path: Path) -> "Sidecar":
        with open(path) as f:
            data = json.load(f)
        return cls(
            method=data["method"],
            saklas_version=data["saklas_version"],
            statements_sha256=data.get("statements_sha256"),
            components=data.get("components"),
        )

    def to_dict(self) -> dict:
        out: dict = {
            "method": self.method,
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


def hash_folder_files(folder: Path) -> dict[str, str]:
    """Return ``{filename: sha256}`` for every file in ``folder`` except ``pack.json``.

    Shared by any code path that needs to (re)populate ``PackMetadata.files``
    after writing new files on disk. Non-recursive — concept folders are flat.
    """
    out: dict[str, str] = {}
    for entry in sorted(folder.iterdir()):
        if entry.is_file() and entry.name != "pack.json":
            out[entry.name] = hash_file(entry)
    return out


def synthesize_pack_metadata(
    *,
    name: str,
    source: str,
    pack_dir: Path,
    description: str = "",
    tags: Sequence[str] = (),
    version: str = "1.0.0",
    license: str = "unknown",
    recommended_alpha: float = 0.5,
    long_description: str = "",
) -> "PackMetadata":
    """Build a :class:`PackMetadata` with ``files`` hashed from on-disk contents.

    Both pack-less HF install synthesis and user-concept extraction converge
    on the same operation: once all real files live under ``pack_dir``, build
    a manifest from them. Callers still own any format-specific file
    fabrication (e.g. hf.py's ``method="imported"`` sidecar stubs) — this
    helper only does the hashing + construction.
    """
    return PackMetadata(
        name=name,
        description=description,
        long_description=long_description,
        version=version,
        license=license,
        tags=list(tags),
        recommended_alpha=recommended_alpha,
        source=source,
        files=hash_folder_files(pack_dir),
    )


def hash_file(path: Path) -> str:
    """Return hex sha256 of a file's contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# In-process fingerprint cache keyed by absolute file path.
# Entry: (size, mtime_ns, expected_sha256) -> last verified sha256 matched.
# Short-circuits full hashing on warm loads. First load, or any stat change,
# still runs the full sha256 before the entry is (re-)populated.
_FINGERPRINT_CACHE: dict[str, tuple[int, int, str]] = {}


def verify_integrity(folder: Path, files: dict[str, str]) -> tuple[bool, list[str]]:
    """Compare every file in `files` (path -> expected sha256) against disk.

    Returns (all_ok, list_of_bad_paths). A missing file counts as bad.

    Uses an in-process (size, mtime_ns) fingerprint cache to avoid re-hashing
    on warm loads. On first load and after any stat change, the full sha256
    still runs — the cache is purely an optimization and does not weaken the
    integrity contract.
    """
    bad: list[str] = []
    for rel, expected in files.items():
        fp = folder / rel
        if not fp.exists():
            bad.append(rel)
            continue
        key = str(fp.resolve())
        try:
            st = fp.stat()
        except OSError:
            bad.append(rel)
            continue
        fp_key = (st.st_size, st.st_mtime_ns, expected)
        cached = _FINGERPRINT_CACHE.get(key)
        if cached == fp_key:
            continue
        if hash_file(fp) != expected:
            _FINGERPRINT_CACHE.pop(key, None)
            bad.append(rel)
            continue
        _FINGERPRINT_CACHE[key] = fp_key
    return (not bad, bad)


@dataclass
class ConceptFolder:
    folder: Path
    metadata: PackMetadata
    has_statements: bool
    # safe_model_id -> Sidecar. GGUF-only entries have no sidecar (metadata
    # is embedded in the gguf file), so this dict is only populated for
    # safetensors tensors.
    _sidecars: dict[str, Sidecar] = field(default_factory=dict)
    # safe_model_id -> "safetensors" | "gguf". A safetensors file takes
    # precedence over a gguf file with the same stem (native format wins).
    _tensor_formats: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, folder: Path) -> "ConceptFolder":
        meta = PackMetadata.load(folder)

        ok, bad = verify_integrity(folder, meta.files)
        if not ok:
            raise PackFormatError(
                f"pack integrity check failed in {folder}: tampered/missing {bad}"
            )

        has_stmts = (folder / "statements.json").exists()
        safetensors = sorted(folder.glob("*.safetensors"))
        ggufs = sorted(folder.glob("*.gguf"))
        if not has_stmts and not safetensors and not ggufs:
            raise PackFormatError(
                f"concept folder must contain at least one of statements.json, "
                f"a .safetensors file, or a .gguf file: {folder}"
            )

        sidecars: dict[str, Sidecar] = {}
        formats: dict[str, str] = {}
        for t in safetensors:
            sc_path = t.with_suffix(".json")
            if not sc_path.exists():
                raise PackFormatError(
                    f"tensor {t.name} has no sidecar {sc_path.name}"
                )
            sidecars[t.stem] = Sidecar.load(sc_path)
            formats[t.stem] = "safetensors"
        for g in ggufs:
            # safetensors wins on conflict — it's the native format and
            # carries more metadata (statements hash, merge components).
            if g.stem not in formats:
                formats[g.stem] = "gguf"

        return cls(
            folder=folder,
            metadata=meta,
            has_statements=has_stmts,
            _sidecars=sidecars,
            _tensor_formats=formats,
        )

    def tensor_models(self) -> list[str]:
        return sorted(self._tensor_formats.keys())

    def tensor_format(self, safe_model_id: str) -> str:
        """Return "safetensors" or "gguf" for the tensor backing this model."""
        return self._tensor_formats[safe_model_id]

    def sidecar(self, safe_model_id: str) -> Sidecar:
        """Return the JSON sidecar for a safetensors tensor.

        Raises KeyError for GGUF-backed tensors, which carry metadata inside
        the .gguf file rather than a sibling sidecar.
        """
        return self._sidecars[safe_model_id]

    def tensor_path(self, safe_model_id: str) -> Path:
        """Return the on-disk path to the tensor file, with extension."""
        fmt = self._tensor_formats.get(safe_model_id)
        if fmt == "gguf":
            return self.folder / f"{safe_model_id}.gguf"
        # Default to safetensors (preserves pre-existing callers that
        # construct paths directly when the tensor hasn't been extracted yet).
        return self.folder / f"{safe_model_id}.safetensors"

    def statements_path(self) -> Path:
        return self.folder / "statements.json"


def is_stale(current_statements_sha: Optional[str], sidecar: Sidecar) -> bool:
    """Tensor is stale if its recorded statements hash disagrees with the current one."""
    if sidecar.statements_sha256 is None or current_statements_sha is None:
        return False
    return sidecar.statements_sha256 != current_statements_sha


def version_mismatch(sidecar: Sidecar, current: str) -> bool:
    """Major/minor mismatch between sidecar version and current saklas version."""
    def _parse(v: str) -> tuple[int, int]:
        parts = v.split(".")
        return int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
    try:
        return _parse(sidecar.saklas_version)[:2] != _parse(current)[:2]
    except (ValueError, IndexError):
        return True


def bundled_concept_names() -> list[str]:
    """List every concept shipped under saklas/data/vectors/."""
    try:
        root = _resources.files("saklas.data.vectors")
    except (ModuleNotFoundError, FileNotFoundError):
        return []
    return sorted(
        p.name for p in root.iterdir()
        if p.is_dir() and (p / "pack.json").is_file()
    )


def materialize_bundled() -> None:
    """Copy bundled package data into ~/.saklas/, leaving user files untouched.

    - neutral_statements.json -> ~/.saklas/neutral_statements.json
    - saklas/data/vectors/<concept>/ -> ~/.saklas/vectors/default/<concept>/

    On format-version drift (user's cached default/<concept>/pack.json is
    older than the shipped format), re-copy the shipped pack.json +
    statements.json in place. Extracted tensor files (<sid>.safetensors /
    .json sidecar) stay put — they're per-model and expensive to recompute.
    This keeps the upgrade path painless for existing installs.
    """
    from saklas.io.paths import saklas_home, vectors_dir, neutral_statements_path

    home = saklas_home()
    home.mkdir(parents=True, exist_ok=True)

    user_ns = neutral_statements_path()
    if not user_ns.exists():
        src = _resources.files("saklas.data").joinpath("neutral_statements.json")
        with src.open("rb") as s, open(user_ns, "wb") as d:
            d.write(s.read())

    default_dir = vectors_dir() / "default"
    default_dir.mkdir(parents=True, exist_ok=True)
    for concept in bundled_concept_names():
        target = default_dir / concept
        copy_shipped = True
        if target.exists():
            pack_json = target / "pack.json"
            if pack_json.exists():
                try:
                    with open(pack_json) as f:
                        data = json.load(f)
                    fmt = data.get("format_version")
                    # Upgrade ONLY on an explicit stale format_version.  A
                    # missing key means either a user-hand-edited file or
                    # something we don't recognize — leave it alone.
                    if not (isinstance(fmt, int) and fmt < PACK_FORMAT_VERSION):
                        copy_shipped = False
                except Exception:
                    copy_shipped = False  # corrupt; don't stomp user state
            else:
                copy_shipped = False  # no pack.json, don't create one
        if not copy_shipped:
            continue
        target.mkdir(parents=True, exist_ok=True)
        pkg_root = _resources.files("saklas.data.vectors").joinpath(concept)
        for entry in pkg_root.iterdir():
            if entry.is_file():
                with entry.open("rb") as s, open(target / entry.name, "wb") as d:
                    d.write(s.read())


def merge_components_status(
    recorded: dict[str, dict],
    current_hashes: dict[str, str],
) -> dict[str, str]:
    """Return {coord: status} where status is 'ok', 'mismatch', or 'missing'.

    'missing' = component was recorded at merge time but is no longer
    present in current_hashes (deleted from installed packs).
    'mismatch' = component's tensor_sha256 has changed since merge.
    'ok' = component still matches.
    """
    out: dict[str, str] = {}
    for coord, info in recorded.items():
        want = info.get("tensor_sha256")
        have = current_hashes.get(coord)
        if have is None:
            out[coord] = "missing"
        elif want is not None and want != have:
            out[coord] = "mismatch"
        else:
            out[coord] = "ok"
    return out


def merge_components_stale(
    recorded: dict[str, dict],
    current_hashes: dict[str, str],
) -> list[str]:
    """Return components that are 'mismatch' or 'missing' vs current hashes.

    Thin wrapper over :func:`merge_components_status` for callers that only
    need the non-ok coords.
    """
    status = merge_components_status(recorded, current_hashes)
    return [coord for coord, s in status.items() if s != "ok"]
