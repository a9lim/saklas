"""Implementations for -r, -x, -i (local folder), -l (local) flag handlers."""
from __future__ import annotations

import shutil
from importlib import resources as _resources
from pathlib import Path
from typing import Optional

from saklas.cli_selectors import (
    ResolvedConcept, Selector, resolve,
)
from saklas.packs import PackMetadata, hash_file, verify_integrity
from saklas.paths import concept_dir, safe_model_id, vectors_dir


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
    """Implement `-x <selector>`. Returns the number of files deleted."""
    concepts = resolve(selector)
    deleted = 0
    for c in concepts:
        files = _tensor_files_for(c, model_scope)
        for f in files:
            f.unlink()
            deleted += 1
        if files:
            _update_files_map(c.folder)
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


def refresh(selector: Selector) -> int:
    """Re-pull concepts from their `source`. Returns count refreshed."""
    concepts = resolve(selector)
    count = 0
    for c in concepts:
        src = c.metadata.source
        if src == "local":
            raise RefreshError(f"{c.namespace}/{c.name}: source=local, nothing to refresh from")
        if src == "bundled":
            _refresh_bundled(c.folder, c.name)
            count += 1
            continue
        if isinstance(src, str) and src.startswith("hf://"):
            from saklas.hf import pull_pack
            pull_pack(src[len("hf://"):], target_folder=c.folder, force=True)
            count += 1
            continue
        raise RefreshError(f"{c.namespace}/{c.name}: unknown source {src!r}")
    return count


def install(target: str, as_: Optional[str], *, force: bool = False) -> Path:
    """Unified -i entry point.

    target may be:
      - "<ns>/<concept>" — HF pull
      - a local path to a folder — copy install
    """
    p = Path(target)
    if p.exists() and p.is_dir():
        return install_folder(p, namespace="local", as_=as_, force=force)

    if "/" not in target:
        raise ValueError(f"install target must be '<ns>/<concept>' or a folder path: {target!r}")

    ns, name = target.split("/", 1)
    if as_:
        if "/" not in as_:
            raise ValueError(f"--as must be '<ns>/<name>', got {as_!r}")
        dst_ns, dst_name = as_.split("/", 1)
    else:
        dst_ns, dst_name = ns, name
    dst = vectors_dir() / dst_ns / dst_name
    from saklas.hf import pull_pack
    return pull_pack(target, target_folder=dst, force=force)


def _all_local() -> list[ResolvedConcept]:
    return resolve(Selector(kind="all", value=None))


def list_concepts(selector: Optional[Selector], hf: bool) -> None:
    """Print local (and optionally HF) concepts matching the selector."""
    if selector is not None and selector.kind == "name" and selector.namespace is not None:
        _print_info(selector.namespace, selector.value, hf=hf)
        return

    if selector is None:
        concepts = _all_local()
    else:
        concepts = resolve(selector)

    _print_list(concepts)
    if hf:
        try:
            from saklas.hf import search_packs
            hf_rows = search_packs(selector)
        except Exception as e:
            print(f"(hf search unavailable: {e})")
            return
        _print_hf_rows(hf_rows)


def _print_list(concepts: list[ResolvedConcept]) -> None:
    print(f"{'NAME':<24} {'NS':<12} {'STATUS':<13} {'ALPHA':<6} TAGS")
    for c in concepts:
        tags = ",".join(c.metadata.tags)
        print(
            f"{c.name:<24} {c.namespace:<12} [installed]   "
            f"{c.metadata.recommended_alpha:<6.2f} {tags}"
        )


def _print_info(namespace: str, name: str, hf: bool) -> None:
    folder = concept_dir(namespace, name)
    if folder.exists():
        from saklas.packs import ConceptFolder
        cf = ConceptFolder.load(folder)
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
        print(f"{namespace}/{name} [hf]")
        print(f"  description: {info.get('description', '')}")
        print(f"  tags:        {', '.join(info.get('tags', []))}")
        print(f"  tensors:     {', '.join(info.get('tensor_models', []))}")
        return
    print(f"not found: {namespace}/{name}")


def _print_hf_rows(rows: list[dict]) -> None:
    for row in rows:
        print(
            f"{row['name']:<24} {row['namespace']:<12} [hf]          "
            f"{row.get('recommended_alpha', 0.0):<6.2f} {','.join(row.get('tags', []))}"
        )
