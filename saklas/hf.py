"""Hugging Face Hub consumption wrappers for saklas pack distribution.

Pack repo convention: an HF **model** repo containing ``pack.json`` at root,
plus ``statements.json`` and/or ``<safe_model_id>.safetensors`` + ``.json``
sidecars. Packs are steering-vector artifacts tied to a base model, so the
model-hub repo type is the canonical home.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

from saklas.packs import PackFormatError, PackMetadata, verify_integrity


class HFError(RuntimeError):
    pass


_HF_SEARCH_CAP = 20


def split_revision(target: str) -> tuple[str, Optional[str]]:
    """Split an ``owner/name@revision`` target into (coord, revision).

    Revisions can be any git ref HF accepts: tag, branch, or commit SHA.
    Concept names are restricted to ``[a-z][a-z0-9-]*``, so ``@`` is
    unambiguous as a separator.
    """
    if "@" not in target:
        return target, None
    coord, _, rev = target.partition("@")
    if not rev:
        raise HFError(f"empty revision after '@' in {target!r}")
    return coord, rev


def _hf_snapshot_download(repo_id: str, **kwargs) -> str:
    """Thin indirection so tests can monkeypatch."""
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=repo_id, repo_type="model", **kwargs)


def _hf_hub_download(repo_id: str, filename: str, **kwargs) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=repo_id, filename=filename, repo_type="model", **kwargs)


def _hf_api():
    from huggingface_hub import HfApi
    return HfApi()


def _download(
    coord: str,
    allow_patterns: Optional[list[str]] = None,
    revision: Optional[str] = None,
) -> str:
    """Snapshot-download <ns>/<concept> from the HF model hub."""
    kwargs: dict = {"repo_id": coord, "allow_patterns": allow_patterns}
    if revision is not None:
        kwargs["revision"] = revision
    try:
        return _hf_snapshot_download(**kwargs)
    except Exception as e:
        label = f"{coord}@{revision}" if revision else coord
        raise HFError(f"{label}: not found ({e})") from e


def pull_pack(
    coord: str,
    target_folder: Path,
    *,
    force: bool,
    revision: Optional[str] = None,
) -> Path:
    """Download <coord> from HF and install into target_folder.

    If ``revision`` is given, pin to that git ref (tag, branch, or commit SHA)
    and record it in the installed pack's ``source`` field so refresh re-pulls
    the same revision.
    """
    tmp_dir = Path(_download(coord, revision=revision))
    if not (tmp_dir / "pack.json").is_file():
        raise HFError(f"{coord}: not a saklas pack (no pack.json at repo root)")

    try:
        meta = PackMetadata.load(tmp_dir)
    except PackFormatError as e:
        raise HFError(f"{coord}: malformed pack.json ({e})") from e

    ok, bad = verify_integrity(tmp_dir, meta.files)
    if not ok:
        raise HFError(f"{coord}: integrity check failed ({bad})")

    if target_folder.exists():
        if not force:
            raise HFError(f"{target_folder} exists; pass force=True to overwrite")
        shutil.rmtree(target_folder)
    target_folder.mkdir(parents=True, exist_ok=True)

    for entry in tmp_dir.iterdir():
        if entry.is_file() and entry.name != "pack.json":
            (target_folder / entry.name).write_bytes(entry.read_bytes())

    meta.source = f"hf://{coord}@{revision}" if revision else f"hf://{coord}"
    meta.write(target_folder)
    return target_folder


def search_packs(selector) -> list[dict]:
    """Search HF for saklas-pack-tagged model repos matching the selector.

    Returns a list of row dicts ready for display. At most _HF_SEARCH_CAP rows.
    """
    api = _hf_api()
    required_tags: list[str] = ["saklas-pack"]
    search_text: Optional[str] = None

    if selector is None:
        pass
    elif selector.kind == "name":
        search_text = selector.value
    elif selector.kind == "tag":
        required_tags.append(selector.value)
    elif selector.kind == "namespace":
        search_text = f"{selector.value}/"
    elif selector.kind == "model":
        pass  # applied post-search

    kwargs: dict = dict(filter=required_tags, limit=_HF_SEARCH_CAP)
    if search_text:
        kwargs["search"] = search_text

    try:
        results = list(api.list_models(**kwargs))
    except TypeError:
        # Older huggingface_hub uses `tags` instead of `filter`.
        kwargs.pop("filter", None)
        kwargs["tags"] = required_tags
        results = list(api.list_models(**kwargs))

    rows: list[dict] = []
    for r in results[:_HF_SEARCH_CAP]:
        coord = r.id
        if "/" in coord:
            ns, nm = coord.split("/", 1)
        else:
            ns, nm = "", coord

        # Pull whatever metadata list_models already gave us; only pay for a
        # fetch_info() call if fields we need for display are actually missing.
        raw_tags = getattr(r, "tags", None) or []
        tags = [str(t) for t in raw_tags] if isinstance(raw_tags, (list, tuple)) else []
        raw_desc = getattr(r, "description", "") or ""
        description = raw_desc if isinstance(raw_desc, str) else ""
        row = {
            "name": nm,
            "namespace": ns,
            "description": description,
            "tags": tags,
            "recommended_alpha": 0.0,
            "tensor_models": [],
        }
        need_info = (
            not description
            or not tags
            or (selector is not None and selector.kind == "model")
        )
        if need_info:
            try:
                info = fetch_info(coord)
            except Exception:
                info = {}
            if info:
                row["name"] = info.get("name", nm)
                row["namespace"] = info.get("namespace", ns)
                row["description"] = info.get("description", description)
                row["tags"] = info.get("tags", tags)
                row["recommended_alpha"] = info.get("recommended_alpha", 0.0)
                row["tensor_models"] = info.get("tensor_models", [])
        rows.append(row)

    if selector is not None and selector.kind == "model":
        safe = selector.value.replace("/", "__")
        rows = [r for r in rows if any(m.startswith(safe) for m in r.get("tensor_models", []))]

    return rows


def fetch_info(coord: str, revision: Optional[str] = None) -> dict:
    """Fetch minimal info about an HF saklas pack without downloading the whole repo."""
    label = f"{coord}@{revision}" if revision else coord
    try:
        dl_kwargs: dict = {}
        if revision is not None:
            dl_kwargs["revision"] = revision
        pj_path = _hf_hub_download(coord, "pack.json", **dl_kwargs)
        with open(pj_path) as f:
            data = json.load(f)
        api = _hf_api()
        list_kwargs: dict = {"repo_id": coord, "repo_type": "model"}
        if revision is not None:
            list_kwargs["revision"] = revision
        files = api.list_repo_files(**list_kwargs)
    except Exception as e:
        raise HFError(f"{label}: fetch_info failed ({e})") from e

    tensor_models = sorted(
        Path(f).stem for f in files
        if f.endswith(".safetensors")
    )
    ns, _, nm = coord.partition("/")
    return {
        "name": data.get("name", nm),
        "namespace": ns,
        "description": data.get("description", ""),
        "long_description": data.get("long_description", ""),
        "tags": data.get("tags", []),
        "recommended_alpha": data.get("recommended_alpha", 0.0),
        "tensor_models": tensor_models,
        "files": list(files),
    }
