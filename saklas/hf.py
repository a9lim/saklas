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

from saklas.packs import PackFormatError, PackMetadata, Sidecar, verify_integrity


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


def _render_model_card(meta: PackMetadata, sidecars: dict[str, Sidecar], coord: str) -> str:
    """Build a HF model card (YAML frontmatter + markdown body) for a pack."""
    base_models = sorted(safe.replace("__", "/") for safe in sidecars.keys())
    tags = sorted({"saklas-pack", "activation-steering", "steering-vector", *meta.tags})

    fm = ["---", "library_name: saklas", f"license: {meta.license}", "tags:"]
    fm += [f"  - {t}" for t in tags]
    if base_models:
        fm.append("base_model:")
        fm += [f"  - {bm}" for bm in base_models]
        fm.append("base_model_relation: adapter")
    fm.append("---")

    body: list[str] = [
        f"# {meta.name}",
        "",
        meta.description,
        "",
    ]
    if meta.long_description:
        body += [meta.long_description, ""]
    body += [
        f"**Recommended alpha:** `{meta.recommended_alpha}`",
        "",
        "## Install",
        "",
        "```bash",
        f"saklas install {coord}",
        "```",
        "",
    ]

    if sidecars:
        body += [
            "## Extracted tensors",
            "",
            "| base model | method | layers | mean score | saklas version |",
            "| --- | --- | --- | --- | --- |",
        ]
        for safe, sc in sorted(sidecars.items()):
            mean = sum(sc.scores.values()) / len(sc.scores) if sc.scores else 0.0
            body.append(
                f"| `{safe.replace('__', '/')}` | `{sc.method}` | {len(sc.scores)} | "
                f"{mean:.4f} | `{sc.saklas_version}` |"
            )
        body.append("")

    if meta.tags:
        body += ["## Tags", "", ", ".join(f"`{t}`" for t in meta.tags), ""]

    body += ["---", "", "Generated by `saklas push`.", ""]
    return "\n".join(fm) + "\n\n" + "\n".join(body)


def resolve_target_coord(pack_name: str, as_: Optional[str]) -> str:
    """Decide the HF coord to push to. `--as owner/name` wins; else whoami()/<pack>."""
    if as_:
        if "/" not in as_:
            raise HFError(f"--as must be '<owner>/<name>', got {as_!r}")
        return as_
    try:
        from huggingface_hub import HfApi
        who = HfApi().whoami()
    except Exception as e:
        raise HFError(
            f"could not resolve HF username ({e}); pass --as owner/name or run `hf auth login`"
        ) from e
    user = who.get("name") if isinstance(who, dict) else None
    if not user:
        raise HFError("could not resolve HF username; pass --as owner/name")
    return f"{user}/{pack_name}"


def push_pack(
    folder: Path,
    coord: str,
    *,
    private: bool = False,
    include_statements: bool = True,
    include_tensors: bool = True,
    model_scope: Optional[str] = None,
    tag_version: bool = False,
    dry_run: bool = False,
) -> tuple[str, Optional[str]]:
    """Push a concept folder to HF as a model repo.

    Stages a copy (so we can add README.md + .gitattributes + a filtered
    pack.json without mutating the source), then one atomic upload. Returns
    ``(repo_url, commit_sha)``; sha is ``None`` on dry-run.
    """
    import tempfile
    from saklas.packs import ConceptFolder
    from saklas.paths import safe_model_id as _safe_id

    cf = ConceptFolder.load(folder)  # runs integrity check
    meta = cf.metadata

    scope_expected: Optional[set[str]] = None
    if model_scope is not None:
        safe = _safe_id(model_scope)
        scope_expected = {f"{safe}.safetensors", f"{safe}.json"}

    staging = Path(tempfile.mkdtemp(prefix="saklas-push-"))
    try:
        kept_files: dict[str, str] = {}
        kept_sidecars: dict[str, Sidecar] = {}
        for rel, sha in meta.files.items():
            if rel == "statements.json":
                if not include_statements:
                    continue
            else:
                if not include_tensors:
                    continue
                if scope_expected is not None and rel not in scope_expected:
                    continue
            src = folder / rel
            if not src.exists():
                raise HFError(f"{folder}: manifest references missing file {rel!r}")
            (staging / rel).write_bytes(src.read_bytes())
            kept_files[rel] = sha
            if rel.endswith(".safetensors"):
                stem = rel[: -len(".safetensors")]
                if stem in cf._sidecars:
                    kept_sidecars[stem] = cf._sidecars[stem]

        has_tensor = any(k.endswith(".safetensors") for k in kept_files)
        has_stmts = "statements.json" in kept_files
        if not has_tensor and not has_stmts:
            raise HFError(
                "nothing to push: filters excluded every file "
                f"(include_statements={include_statements}, include_tensors={include_tensors}, "
                f"model_scope={model_scope!r})"
            )

        staged_meta = PackMetadata(
            name=meta.name,
            description=meta.description,
            long_description=meta.long_description,
            version=meta.version,
            license=meta.license,
            tags=list(meta.tags),
            recommended_alpha=meta.recommended_alpha,
            source=meta.source,
            files=kept_files,
        )
        staged_meta.write(staging)

        (staging / ".gitattributes").write_text(
            "*.safetensors filter=lfs diff=lfs merge=lfs -text\n"
        )
        (staging / "README.md").write_text(
            _render_model_card(staged_meta, kept_sidecars, coord)
        )

        repo_url = f"https://huggingface.co/{coord}"
        if dry_run:
            return (repo_url, None)

        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(repo_id=coord, repo_type="model", private=private, exist_ok=True)
        commit_msg = f"saklas push: {meta.name} v{meta.version}"
        info = api.upload_folder(
            repo_id=coord,
            repo_type="model",
            folder_path=str(staging),
            commit_message=commit_msg,
        )
        sha = getattr(info, "oid", None) or getattr(info, "commit_sha", None)

        if tag_version:
            try:
                api.create_tag(
                    repo_id=coord,
                    tag=f"v{meta.version}",
                    revision=sha,
                    repo_type="model",
                )
            except Exception as e:
                raise HFError(
                    f"uploaded but failed to tag v{meta.version}: {e}"
                ) from e

        return (repo_url, sha)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


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
