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
import logging
import re
from dataclasses import dataclass, field
from importlib import resources as _resources
from pathlib import Path
from typing import Any, Optional, Sequence

from saklas.core.errors import SaklasError

_log = logging.getLogger("saklas.io.packs")


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

    def user_message(self) -> tuple[int, str]:
        return (400, str(self) or self.__class__.__name__)


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
        if fmt_ver > PACK_FORMAT_VERSION:
            raise PackFormatError(
                f"pack at {folder} was created by a newer saklas "
                f"(format v{fmt_ver} > local v{PACK_FORMAT_VERSION}); "
                f"upgrade saklas, or pass `--force-legacy` to attempt to "
                f"load anyway."
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

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
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
        from saklas.io.atomic import write_json_atomic

        folder.mkdir(parents=True, exist_ok=True)
        write_json_atomic(folder / "pack.json", self.to_dict())


@dataclass
class Sidecar:
    method: str
    saklas_version: str
    statements_sha256: Optional[str] = None
    components: Optional[dict[str, dict[str, Any]]] = None

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

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "method": self.method,
            "saklas_version": self.saklas_version,
        }
        if self.statements_sha256 is not None:
            out["statements_sha256"] = self.statements_sha256
        if self.components is not None:
            out["components"] = self.components
        return out

    def write(self, path: Path) -> None:
        from saklas.io.atomic import write_json_atomic

        path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(path, self.to_dict())


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


def _canonical_json_sha256(data: bytes) -> str:
    """Return a content-stable sha256 of a JSON byte payload.

    The hash is computed over canonical JSON form (sorted keys, no
    surrounding whitespace) so cosmetic-only differences (key order,
    trailing newline, indent width) compare equal. Used by
    :func:`materialize_bundled` to detect whether a user has actually edited
    statements vs. the file just being byte-different from the shipped form.

    Falls back to a raw sha256 if the bytes don't parse as JSON — better to
    treat unparseable on-disk content as "different from bundled" than to
    silently replace it.
    """
    try:
        parsed = json.loads(data)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return hashlib.sha256(data).hexdigest()
    canonical = json.dumps(parsed, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def materialize_bundled() -> None:
    """Copy bundled package data into ~/.saklas/, leaving user files untouched.

    - neutral_statements.json -> ~/.saklas/neutral_statements.json
    - saklas/data/vectors/<concept>/ -> ~/.saklas/vectors/default/<concept>/

    On format-version drift (user's cached default/<concept>/pack.json is
    older than the shipped format), re-copy the shipped pack.json in place
    and write a ``pack.json.bak`` next to it. ``statements.json`` is only
    overwritten if its canonical-JSON hash matches the bundled copy — a
    user-edited statements file is preserved and skipped with an INFO log.
    Extracted tensor files (<sid>.safetensors / .json sidecar) stay put —
    they're per-model and expensive to recompute. This keeps the upgrade
    path painless for existing installs without silently discarding
    customization.
    """
    from saklas.io.atomic import write_bytes_atomic
    from saklas.io.paths import saklas_home, vectors_dir, neutral_statements_path

    home = saklas_home()
    home.mkdir(parents=True, exist_ok=True)

    user_ns = neutral_statements_path()
    if not user_ns.exists():
        src = _resources.files("saklas.data").joinpath("neutral_statements.json")
        write_bytes_atomic(user_ns, src.read_bytes())

    default_dir = vectors_dir() / "default"
    default_dir.mkdir(parents=True, exist_ok=True)
    for concept in bundled_concept_names():
        target = default_dir / concept
        pkg_root = _resources.files("saklas.data.vectors").joinpath(concept)

        if not target.exists():
            # Fresh install — copy every shipped file atomically.
            target.mkdir(parents=True, exist_ok=True)
            for entry in pkg_root.iterdir():
                if entry.is_file():
                    write_bytes_atomic(target / entry.name, entry.read_bytes())
            continue

        pack_json = target / "pack.json"
        if not pack_json.exists():
            # Folder exists without a pack.json — refuse to fabricate one.
            continue

        # Read the on-disk pack.json to decide whether to upgrade.
        try:
            with open(pack_json) as f:
                on_disk_pack = json.load(f)
        except Exception:
            # Corrupt; don't stomp user state.
            continue

        fmt = on_disk_pack.get("format_version")
        if not (isinstance(fmt, int) and fmt < PACK_FORMAT_VERSION):
            # No explicit stale version — leave alone.
            continue

        # Stale format_version → upgrade pack.json (always with .bak), and
        # only overwrite statements.json if the user hasn't touched it.
        bundled_pack_bytes = (pkg_root / "pack.json").read_bytes()
        on_disk_pack_bytes = pack_json.read_bytes()

        # Always preserve the previous pack.json before replacing it.
        write_bytes_atomic(pack_json.with_suffix(".json.bak"), on_disk_pack_bytes)
        write_bytes_atomic(pack_json, bundled_pack_bytes)

        on_disk_stmts = target / "statements.json"
        bundled_stmts = pkg_root / "statements.json"
        if bundled_stmts.is_file():
            bundled_stmts_bytes = bundled_stmts.read_bytes()
            if on_disk_stmts.exists():
                on_disk_hash = _canonical_json_sha256(on_disk_stmts.read_bytes())
                bundled_hash = _canonical_json_sha256(bundled_stmts_bytes)
                if on_disk_hash == bundled_hash:
                    # Bytewise / canonical-equivalent — safe to refresh.
                    write_bytes_atomic(on_disk_stmts, bundled_stmts_bytes)
                else:
                    _log.info(
                        "materialize_bundled: preserving user-edited "
                        "statements.json for default/%s "
                        "(canonical hash differs from bundled)",
                        concept,
                    )
            else:
                # No statements on disk — write the shipped copy.
                write_bytes_atomic(on_disk_stmts, bundled_stmts_bytes)

        # Copy any other shipped files that aren't pack.json or statements.json.
        for entry in pkg_root.iterdir():
            if not entry.is_file():
                continue
            if entry.name in {"pack.json", "statements.json"}:
                continue
            write_bytes_atomic(target / entry.name, entry.read_bytes())

        # One INFO log line per upgrade, after writes succeed.
        _log.info(
            "materialize_bundled: upgraded default/%s v%d -> v%d (format_version)",
            concept, fmt, PACK_FORMAT_VERSION,
        )


def merge_components_status(
    recorded: dict[str, dict[str, Any]],
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
    recorded: dict[str, dict[str, Any]],
    current_hashes: dict[str, str],
) -> list[str]:
    """Return components that are 'mismatch' or 'missing' vs current hashes.

    Thin wrapper over :func:`merge_components_status` for callers that only
    need the non-ok coords.
    """
    status = merge_components_status(recorded, current_hashes)
    return [coord for coord, s in status.items() if s != "ok"]


def enumerate_variants(folder: Path, model_id: str) -> dict[str, Path]:
    """List all on-disk tensor variants for ``(folder, model_id)``.

    Returns ``{variant_key: path}`` where ``variant_key`` is ``"raw"`` for
    the unsuffixed tensor and ``"sae-<release>"`` for SAE variants. Paths
    point at the ``.safetensors`` files; callers derive the sidecar path by
    swapping the extension.
    """
    from saklas.io.paths import safe_model_id, parse_tensor_filename

    target_model = safe_model_id(model_id)
    if not folder.is_dir():
        return {}

    out: dict[str, Path] = {}
    for p in folder.iterdir():
        if not p.is_file():
            continue
        parsed = parse_tensor_filename(p.name)
        if parsed is None:
            continue
        model, release = parsed
        if model != target_model:
            continue
        key = "raw" if release is None else f"sae-{release}"
        out[key] = p
    return out
