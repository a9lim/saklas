"""Implementations backing the install/refresh/clear/uninstall/list subcommands."""
from __future__ import annotations

import shutil
from importlib import resources as _resources
from pathlib import Path
from typing import Optional

from saklas.cli_selectors import (
    ResolvedConcept, Selector, invalidate as _invalidate_selector_cache, resolve,
)
from saklas.packs import PackMetadata, hash_file, verify_integrity
from saklas.paths import concept_dir, neutral_statements_path, safe_model_id, vectors_dir


class InstallConflict(RuntimeError):
    pass


class RefreshError(RuntimeError):
    pass


def _tensor_files_for(concept: ResolvedConcept, model_scope: Optional[str]) -> list[Path]:
    out: list[Path] = []
    if model_scope is not None:
        safe = safe_model_id(model_scope)
        ts = concept.folder / f"{safe}.safetensors"
        sc = concept.folder / f"{safe}.json"
        if ts.exists():
            out.append(ts)
        if sc.exists():
            out.append(sc)
        return out
    for ts in sorted(concept.folder.glob("*.safetensors")):
        out.append(ts)
        sc = ts.with_suffix(".json")
        if sc.exists():
            out.append(sc)
    return out


def _update_files_map(concept_folder: Path) -> None:
    """Recompute the pack.json `files` map after files were removed."""
    meta = PackMetadata.load(concept_folder)
    new_files: dict[str, str] = {}
    for entry in sorted(concept_folder.iterdir()):
        if entry.name == "pack.json" or not entry.is_file():
            continue
        new_files[entry.name] = hash_file(entry)
    meta.files = new_files
    meta.write(concept_folder)


def delete_tensors(selector: Selector, model_scope: Optional[str]) -> int:
    """Backs `saklas clear`. Returns the number of files deleted."""
    concepts = resolve(selector)
    deleted = 0
    for c in concepts:
        files = _tensor_files_for(c, model_scope)
        for f in files:
            f.unlink()
            deleted += 1
        if files:
            _update_files_map(c.folder)
    if deleted:
        _invalidate_selector_cache()
    return deleted


def install_folder(src: Path, namespace: str, as_: Optional[str], *, force: bool = False) -> Path:
    """Copy a concept folder into ~/.saklas/vectors/<ns>/<name>/."""
    src_meta = PackMetadata.load(src)

    if as_:
        if "/" not in as_:
            raise ValueError(f"--as must be '<namespace>/<name>', got {as_!r}")
        dst_ns, dst_name = as_.split("/", 1)
        src_meta.name = dst_name
    else:
        dst_ns = namespace
        dst_name = src_meta.name

    dst = vectors_dir() / dst_ns / dst_name
    if dst.exists() and not force:
        raise InstallConflict(f"{dst} already exists; use force=True or --as to relocate")
    if dst.exists():
        shutil.rmtree(dst)

    dst.mkdir(parents=True, exist_ok=True)
    for entry in src.iterdir():
        if entry.is_file() and entry.name != "pack.json":
            (dst / entry.name).write_bytes(entry.read_bytes())

    src_meta.write(dst)

    ok, bad = verify_integrity(dst, src_meta.files)
    if not ok:
        shutil.rmtree(dst)
        raise InstallConflict(f"integrity check failed on install: {bad}")
    _invalidate_selector_cache()
    return dst


def _refresh_bundled(target: Path, concept_name: str) -> None:
    """Copy the bundled copy of <concept_name> over <target>."""
    pkg_root = _resources.files("saklas.data.vectors").joinpath(concept_name)
    if not pkg_root.is_dir():
        raise RefreshError(f"bundled concept '{concept_name}' not in package data")
    for entry in list(target.iterdir()):
        if entry.is_file():
            entry.unlink()
    for entry in pkg_root.iterdir():
        if entry.is_file():
            with entry.open("rb") as s, open(target / entry.name, "wb") as d:
                d.write(s.read())


def refresh(selector: Selector, *, model_scope: Optional[str] = None) -> int:
    """Re-pull concepts from their `source`. Returns count refreshed.

    If model_scope is given, the refresh deletes the matching per-model tensor
    pair so it will re-extract on next use (tensors-only scoped refresh). This
    intentionally does NOT re-download from HF for just one model because HF
    pulls are whole-repo operations.
    """
    concepts = resolve(selector)
    count = 0
    for c in concepts:
        if model_scope is not None:
            # Scoped refresh: delete just that model's tensor + sidecar so it
            # re-extracts from the concept's source statements on next use.
            files = _tensor_files_for(c, model_scope)
            if files:
                for f in files:
                    f.unlink()
                _update_files_map(c.folder)
                count += 1
            continue

        src = c.metadata.source
        if src == "local":
            # Selectors like `all` or `tag:emotion` naturally sweep in local
            # concepts that have no upstream to re-pull from; skip them
            # silently rather than aborting the whole refresh. A user asking
            # for a specific local by name gets the same treatment — there's
            # nothing meaningful to do, and erroring just forces them to
            # hand-craft a selector that excludes their own work.
            continue
        if src == "bundled":
            _refresh_bundled(c.folder, c.name)
            count += 1
            continue
        if isinstance(src, str) and src.startswith("hf://"):
            from saklas.hf import pull_pack, split_revision
            coord, revision = split_revision(src[len("hf://"):])
            pull_pack(coord, target_folder=c.folder, force=True, revision=revision)
            count += 1
            continue
        raise RefreshError(f"{c.namespace}/{c.name}: unknown source {src!r}")
    if count:
        _invalidate_selector_cache()
    return count


def refresh_neutrals() -> Path:
    """Overwrite ~/.saklas/neutral_statements.json with the bundled copy.

    Returns the destination path. Stale layer means are picked up on next
    session init via the hash check in bootstrap_layer_means.
    """
    dst = neutral_statements_path()
    dst.parent.mkdir(parents=True, exist_ok=True)
    src = _resources.files("saklas.data").joinpath("neutral_statements.json")
    with src.open("rb") as s, open(dst, "wb") as d:
        d.write(s.read())
    return dst


def _strip_tensors(folder: Path) -> None:
    """Remove tensor/sidecar files and rewrite pack.json.files accordingly."""
    for entry in sorted(folder.iterdir()):
        if entry.is_file() and (
            entry.suffix == ".safetensors"
            or (entry.suffix == ".json" and entry.name not in {"pack.json", "statements.json"})
        ):
            entry.unlink()
    _update_files_map(folder)


def install(
    target: str,
    as_: Optional[str],
    *,
    force: bool = False,
    statements_only: bool = False,
) -> Path:
    """Install a pack from an HF coord or a local folder.

    target may be:
      - "<ns>/<concept>[@revision]" — HF pull
      - a local path to a folder — copy install

    If statements_only is true, any safetensors + sidecars that arrived with the
    pack are removed after install and pack.json.files is rewritten. The pack
    remains a legitimate standalone concept; tensors re-extract on demand.
    """
    p = Path(target)
    if p.exists() and p.is_dir():
        dst = install_folder(p, namespace="local", as_=as_, force=force)
        if statements_only:
            _strip_tensors(dst)
            _invalidate_selector_cache()
        return dst

    from saklas.hf import pull_pack, split_revision
    coord, revision = split_revision(target)

    if "/" not in coord:
        raise ValueError(f"install target must be '<ns>/<concept>[@revision]' or a folder path: {target!r}")

    ns, name = coord.split("/", 1)
    if as_:
        if "/" not in as_:
            raise ValueError(f"--as must be '<ns>/<name>', got {as_!r}")
        dst_ns, dst_name = as_.split("/", 1)
    else:
        dst_ns, dst_name = ns, name
    dst = vectors_dir() / dst_ns / dst_name
    result = pull_pack(coord, target_folder=dst, force=force, revision=revision)
    if statements_only:
        _strip_tensors(result)
    _invalidate_selector_cache()
    return result


def uninstall(selector: Selector, *, yes: bool = False) -> int:
    """Fully remove concept folders matching `selector`.

    Unlike `delete_tensors`, this removes statements.json and pack.json too —
    the concept folder ceases to exist. Bundled concepts will re-materialize on
    next session init; that is intentional.

    Broad selectors (`all`, bare `namespace:`) require yes=True.
    """
    if not yes and selector.kind in {"all", "namespace"}:
        raise RuntimeError(
            f"refusing to uninstall a broad selector ({selector.kind}); pass yes=True to confirm"
        )
    concepts = resolve(selector)
    count = 0
    for c in concepts:
        shutil.rmtree(c.folder)
        count += 1
    if count:
        _invalidate_selector_cache()
    return count


def push(
    selector: Selector,
    *,
    as_: Optional[str] = None,
    private: bool = False,
    model_scope: Optional[str] = None,
    statements_only: bool = False,
    no_statements: bool = False,
    tag_version: bool = False,
    dry_run: bool = False,
    force: bool = False,
) -> tuple[str, str, Optional[str]]:
    """Back `saklas push`. Returns ``(coord, repo_url, commit_sha)``."""
    from saklas import hf as hf_mod

    if statements_only and no_statements:
        raise RuntimeError("--statements-only and --no-statements are mutually exclusive")
    if statements_only and model_scope is not None:
        raise RuntimeError("--statements-only and --model are mutually exclusive")

    matches = resolve(selector)
    if not matches:
        raise RuntimeError(f"no concept matched selector {selector.value!r}")
    if len(matches) > 1:
        qualified = ", ".join(f"{c.namespace}/{c.name}" for c in matches)
        raise RuntimeError(
            f"push requires a single concept; selector matched: {qualified}"
        )
    c = matches[0]

    src = c.metadata.source or ""
    if not force and as_ is None and (src.startswith("bundled") or src.startswith("hf://")):
        raise RuntimeError(
            f"refusing to push pack with source={src!r}; "
            f"pass --as owner/name to retarget, or --force to republish in place"
        )

    # Rehash disk state so the pushed manifest matches the bytes we upload.
    _update_files_map(c.folder)

    coord = hf_mod.resolve_target_coord(c.metadata.name, as_)

    include_statements = not no_statements
    include_tensors = not statements_only

    repo_url, sha = hf_mod.push_pack(
        c.folder,
        coord,
        private=private,
        include_statements=include_statements,
        include_tensors=include_tensors,
        model_scope=None if statements_only else model_scope,
        tag_version=tag_version,
        dry_run=dry_run,
    )
    return coord, repo_url, sha


def _all_local() -> list[ResolvedConcept]:
    return resolve(Selector(kind="all", value=None))


def list_concepts(
    selector: Optional[Selector],
    *,
    hf: bool = True,
    installed_only: bool = False,
    json_output: bool = False,
    verbose: bool = False,
) -> None:
    """Print (or JSON-dump) concepts matching the selector.

    By default the HF hub is queried and results are merged with local installs.
    `installed_only=True` suppresses the HF query. `json_output=True` emits a
    machine-readable list to stdout instead of the table.
    """
    if hf and installed_only:
        hf = False

    if selector is not None and selector.kind == "name" and selector.namespace is not None:
        _print_info(selector.namespace, selector.value, hf=hf, json_output=json_output)
        return

    if selector is None:
        concepts = _all_local()
    else:
        concepts = resolve(selector)

    hf_rows: list[dict] = []
    if hf:
        from saklas.hf import HFError, search_packs
        try:
            from huggingface_hub.utils import HfHubHTTPError
        except ImportError:
            HfHubHTTPError = ()  # type: ignore[assignment,misc]
        try:
            hf_rows = search_packs(selector)
        except (HFError, HfHubHTTPError, OSError) as e:
            msg = f"hf search unavailable: {type(e).__name__}: {e}"
            if json_output:
                import json as _json
                print(_json.dumps({"error": msg, "installed": [_row_from_concept(c) for c in concepts]}))
                return
            _print_list(concepts, verbose=verbose)
            print(f"({msg})")
            return

    if json_output:
        import json as _json
        payload = [_row_from_concept(c) for c in concepts]
        installed_keys = {(r["namespace"], r["name"]) for r in payload}
        for row in hf_rows:
            if (row.get("namespace"), row.get("name")) in installed_keys:
                continue
            payload.append({
                "name": row.get("name"),
                "namespace": row.get("namespace"),
                "status": "hf",
                "recommended_alpha": row.get("recommended_alpha", 0.0),
                "tags": row.get("tags", []),
                "description": row.get("description", ""),
                "tensor_models": row.get("tensor_models", []),
            })
        print(_json.dumps(payload, indent=2))
        return

    _print_list(concepts, verbose=verbose)
    if hf_rows:
        _print_hf_rows(hf_rows, verbose=verbose)


def _row_from_concept(c: ResolvedConcept) -> dict:
    from saklas.packs import ConceptFolder
    tensor_models: list[str] = []
    status = "installed"
    error: Optional[str] = None
    try:
        cf = ConceptFolder.load(c.folder)
        tensor_models = cf.tensor_models()
    except Exception as e:
        status = "corrupt"
        error = f"{type(e).__name__}: {e}"
    row = {
        "name": c.name,
        "namespace": c.namespace,
        "status": status,
        "recommended_alpha": c.metadata.recommended_alpha,
        "tags": list(c.metadata.tags),
        "description": c.metadata.description,
        "source": c.metadata.source,
        "tensor_models": tensor_models,
    }
    if error is not None:
        row["error"] = error
    return row


def _print_list(concepts: list[ResolvedConcept], *, verbose: bool = False) -> None:
    if verbose:
        print(f"{'NAME':<24} {'NS':<12} {'STATUS':<13} {'ALPHA':<6} {'TAGS':<24} DESCRIPTION")
    else:
        print(f"{'NAME':<24} {'NS':<12} {'STATUS':<13} {'ALPHA':<6} TAGS")
    for c in concepts:
        r = _row_from_concept(c)
        tags = ",".join(r["tags"])
        tag = "[corrupt]    " if r["status"] == "corrupt" else "[installed]  "
        line = (
            f"{c.name:<24} {c.namespace:<12} {tag} "
            f"{c.metadata.recommended_alpha:<6.2f} {tags}"
        )
        if verbose:
            line = (
                f"{c.name:<24} {c.namespace:<12} {tag} "
                f"{c.metadata.recommended_alpha:<6.2f} {tags:<24} {c.metadata.description}"
            )
        print(line)
        if r.get("error"):
            print(f"  ! {r['error']}")


def _print_info(namespace: str, name: str, *, hf: bool, json_output: bool = False) -> None:
    folder = concept_dir(namespace, name)
    if folder.exists():
        from saklas.packs import ConceptFolder
        cf = ConceptFolder.load(folder)
        if json_output:
            import json as _json
            print(_json.dumps({
                "name": name, "namespace": namespace, "status": "installed",
                "description": cf.metadata.description,
                "long_description": cf.metadata.long_description,
                "tags": list(cf.metadata.tags),
                "recommended_alpha": cf.metadata.recommended_alpha,
                "source": cf.metadata.source,
                "tensor_models": cf.tensor_models(),
            }, indent=2))
            return
        print(f"{namespace}/{name} [installed]")
        print(f"  description: {cf.metadata.description}")
        if cf.metadata.long_description:
            print(f"  long:        {cf.metadata.long_description}")
        print(f"  tags:        {', '.join(cf.metadata.tags) or '(none)'}")
        print(f"  alpha:       {cf.metadata.recommended_alpha}")
        print(f"  source:      {cf.metadata.source}")
        print(f"  tensors:     {', '.join(cf.tensor_models()) or '(none)'}")
        return
    if hf:
        from saklas.hf import fetch_info
        info = fetch_info(f"{namespace}/{name}")
        if json_output:
            import json as _json
            print(_json.dumps({
                "name": name, "namespace": namespace, "status": "hf",
                "description": info.get("description", ""),
                "tags": info.get("tags", []),
                "tensor_models": info.get("tensor_models", []),
            }, indent=2))
            return
        print(f"{namespace}/{name} [hf]")
        print(f"  description: {info.get('description', '')}")
        print(f"  tags:        {', '.join(info.get('tags', []))}")
        print(f"  tensors:     {', '.join(info.get('tensor_models', []))}")
        return
    print(f"not found: {namespace}/{name}")


def _print_hf_rows(rows: list[dict], *, verbose: bool = False) -> None:
    for row in rows:
        line = (
            f"{row['name']:<24} {row['namespace']:<12} [hf]          "
            f"{row.get('recommended_alpha', 0.0):<6.2f} {','.join(row.get('tags', []))}"
        )
        if verbose:
            line = (
                f"{row['name']:<24} {row['namespace']:<12} [hf]          "
                f"{row.get('recommended_alpha', 0.0):<6.2f} "
                f"{','.join(row.get('tags', [])):<24} {row.get('description', '')}"
            )
        print(line)
