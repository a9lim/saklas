"""Shared artifact primitives: name validation, integrity, and profile versioning.

The 4.0 collapse retired the pack *format/distribution* surface
(``PackMetadata`` / ``ConceptFolder`` / ``Sidecar`` / ``enumerate_variants`` /
``materialize_bundled`` / HF pack distribution) — concepts are manifolds now
(:mod:`saklas.io.manifolds`).  What remains here is the cross-cutting
infrastructure several layers still share:

- ``NAME_REGEX`` — the artifact-name grammar (manifolds reuse it);
- ``hash_file`` / ``hash_folder_files`` / ``verify_integrity`` — the sha256
  integrity helpers (the neutral/layer-means/alignment caches + the manifold
  format's own integrity manifest build on these);
- ``PACK_FORMAT_VERSION`` — the current profile-cache sidecar version written by
  :func:`saklas.core.profile.save_profile`.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from saklas.core.errors import SaklasError
from saklas.io.paths import ensure_within


NAME_REGEX = re.compile(r"^[a-z][a-z0-9._-]{0,63}$")

# Current profile-cache sidecar format version.
PACK_FORMAT_VERSION = 3


class PackFormatError(ValueError, SaklasError):
    """Raised when an artifact manifest or sidecar is malformed."""

    def user_message(self) -> tuple[int, str]:
        return (400, str(self) or self.__class__.__name__)


def hash_file(path: Path) -> str:
    """Return hex sha256 of a file's contents.

    Shared by the pack and manifold integrity manifests
    (:func:`saklas.io.manifolds.hash_manifold_files` imports this).
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_folder_files(folder: Path) -> dict[str, str]:
    """Return ``{filename: sha256}`` for every file in ``folder`` except ``pack.json``.

    Non-recursive — concept folders are flat.
    """
    out: dict[str, str] = {}
    for entry in sorted(folder.iterdir()):
        if entry.is_file() and entry.name != "pack.json":
            out[entry.name] = hash_file(entry)
    return out


# In-process fingerprint cache keyed by absolute file path.
# Entry: stable file identity + expected sha256 -> last verification matched.
# Short-circuits full hashing on warm loads. First load, or any stat change,
# still runs the full sha256 before the entry is (re-)populated.
_FINGERPRINT_CACHE: dict[str, tuple[int, int, int, int, int, str]] = {}


def verify_integrity(folder: Path, files: dict[str, str]) -> tuple[bool, list[str]]:
    """Compare every file in `files` (path -> expected sha256) against disk.

    Returns (all_ok, list_of_bad_paths). A missing file counts as bad.

    Uses an in-process stat-identity fingerprint cache to avoid re-hashing
    on warm loads. On first load and after any stat change, the full sha256
    still runs — the cache is purely an optimization and does not weaken the
    integrity contract.
    """
    bad: list[str] = []
    for rel, expected in files.items():
        # A manifest entry that resolves outside ``folder`` (a ``..`` or
        # absolute ``rel`` in a downloaded manifest) is treated as a failed
        # file rather than read off-tree. ``ensure_within`` is the path-
        # traversal barrier.
        try:
            fp = ensure_within(folder, rel)
        except ValueError:
            bad.append(rel)
            continue
        if not fp.exists():
            bad.append(rel)
            continue
        key = str(fp.resolve())
        try:
            st = fp.stat()
        except OSError:
            bad.append(rel)
            continue
        fp_key = (
            st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_dev, st.st_ino,
            expected,
        )
        cached = _FINGERPRINT_CACHE.get(key)
        if cached == fp_key:
            continue
        if hash_file(fp) != expected:
            _FINGERPRINT_CACHE.pop(key, None)
            bad.append(rel)
            continue
        _FINGERPRINT_CACHE[key] = fp_key
    return (not bad, bad)
