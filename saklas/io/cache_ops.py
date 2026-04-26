"""Implementations backing the install/refresh/clear/uninstall/list subcommands.

Pure data layer: every function returns structured results
(``ConceptRow`` / ``ConceptInfo`` / ``PackListResult``) rather than
printing. The CLI surface (``saklas/cli/output.py``) handles text and
JSON rendering.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from importlib import resources as _resources
from pathlib import Path
from typing import Any, Optional

from saklas.io.selectors import (
    ResolvedConcept, Selector, invalidate as _invalidate_selector_cache, resolve,
)
from saklas.core.errors import SaklasError
from saklas.io.atomic import write_bytes_atomic
from saklas.io.packs import PackMetadata, hash_file, verify_integrity
from saklas.io.paths import concept_dir, neutral_statements_path, safe_model_id, vectors_dir


class InstallConflict(RuntimeError, SaklasError):
    def user_message(self) -> tuple[int, str]:
        return (409, str(self) or self.__class__.__name__)


class RefreshError(RuntimeError, SaklasError):
    def user_message(self) -> tuple[int, str]:
        return (500, str(self) or self.__class__.__name__)


def _variant_matches_key(key: str, filter_: str) -> bool:
    if filter_ == "all":
        return True
    if filter_ == "raw":
        return key == "raw"
    if filter_ == "sae":
        return key.startswith("sae-")
    return False


def _variant_key_from_filename(name: str) -> Optional[str]:
    """Return the variant key for a tensor/sidecar filename, or None if unparseable.

    ``<model>.safetensors`` → ``"raw"``; ``<model>_sae-<release>.safetensors`` →
    ``"sae-<release>"``; sidecars mirror their tensor partners.
    """
    from saklas.io.paths import parse_tensor_filename
    if name.endswith(".json"):
        return _variant_key_from_filename(name[:-len(".json")] + ".safetensors")
    parsed = parse_tensor_filename(name)
    if parsed is None:
        return None
    _model, release = parsed
    return "raw" if release is None else f"sae-{release}"


def _tensor_files_for(
    concept: ResolvedConcept,
    model_scope: Optional[str],
    *,
    variant: str = "all",
) -> list[Path]:
    from saklas.io.packs import enumerate_variants

    out: list[Path] = []
    if model_scope is not None:
        variants = enumerate_variants(concept.folder, model_scope)
        for key, tensor_path in variants.items():
            if not _variant_matches_key(key, variant):
                continue
            if tensor_path.exists():
                out.append(tensor_path)
            sc = tensor_path.with_suffix(".json")
            if sc.exists():
                out.append(sc)
        return out
    for ts in sorted(concept.folder.glob("*.safetensors")):
        key = _variant_key_from_filename(ts.name)
        if key is None or not _variant_matches_key(key, variant):
            continue
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


def delete_tensors(
    selector: Selector,
    model_scope: Optional[str],
    *,
    variant: str = "all",
) -> int:
    """Backs `saklas clear`. Returns the number of files deleted.

    ``variant`` filters by tensor flavor: ``"raw"`` only touches unsuffixed
    tensors, ``"sae"`` only touches ``_sae-*`` variants, ``"all"`` (default)
    touches both.
    """
    concepts = resolve(selector)
    deleted = 0
    for c in concepts:
        files = _tensor_files_for(c, model_scope, variant=variant)
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
            write_bytes_atomic(dst / entry.name, entry.read_bytes())

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
            write_bytes_atomic(target / entry.name, entry.read_bytes())


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
        if src.startswith("hf://"):
            from saklas.io.hf import pull_pack, split_revision
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
    write_bytes_atomic(dst, src.read_bytes())
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

    from saklas.io.hf import pull_pack, split_revision
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


def _resolve_model_hint(safe_id: str) -> str:
    """Derive llama.cpp's ``controlvector.model_hint`` from a safe_model_id.

    Strategy: load the base model's config via ``transformers.AutoConfig``
    (cache-first, network fallback) and return ``config.model_type``.  That's
    the same string llama.cpp's loader keys off when matching a control
    vector to a loaded model (e.g. ``"llama"``, ``"gemma2"``, ``"qwen2"``).

    Raises RuntimeError with actionable guidance if the config can't be
    resolved — callers should surface the ``--model-hint`` flag as the
    escape hatch.
    """
    hf_id = safe_id.replace("__", "/")
    try:
        from transformers import AutoConfig
    except ImportError as e:  # pragma: no cover — transformers is a hard dep
        raise RuntimeError(
            f"could not resolve model_hint for {hf_id!r}: transformers missing ({e})"
        ) from e
    try:
        cfg = AutoConfig.from_pretrained(hf_id, trust_remote_code=False)
    except Exception as e:
        raise RuntimeError(
            f"could not resolve model_hint for {hf_id!r}: {e}. "
            f"Pass --model-hint <arch> explicitly (e.g. 'llama', 'gemma2', 'qwen2')."
        ) from e
    mt = getattr(cfg, "model_type", None)
    if not mt:
        raise RuntimeError(
            f"{hf_id}: config has no model_type field; pass --model-hint explicitly"
        )
    return str(mt)


def export_gguf(
    selector: Selector,
    *,
    model_scope: Optional[str] = None,
    output: Optional[str] = None,
    model_hint: Optional[str] = None,
) -> list[Path]:
    """Export a concept's baked tensors to llama.cpp GGUF.

    ``selector`` must resolve to a single concept. ``model_scope`` restricts
    the export to one base model; without it, every model present in the pack
    is exported (one .gguf file per model).

    ``output``:
      - single-model + file path → write to exactly that path
      - single-model + directory → write to ``<dir>/<safe_model_id>.gguf``
      - multi-model → must be a directory (or None, meaning in-pack sibling)
      - None → write alongside the safetensors in the pack folder (rejected
        for bundled concepts, whose folder is restored on refresh)

    ``model_hint`` overrides the ``controlvector.model_hint`` metadata string.
    Default: derived via ``transformers.AutoConfig.model_type`` on the base
    model (cache-first, network fallback).  This is the string llama.cpp's
    control-vector loader matches against, so it must be the architecture
    name (``"llama"``, ``"gemma2"``, ``"qwen2"``) and *not* the HF coord.

    Returns the list of paths written.
    """
    from saklas.io.gguf_io import write_gguf_profile
    from saklas.core.vectors import load_profile

    concepts = resolve(selector)
    if len(concepts) != 1:
        raise RuntimeError(
            f"export_gguf requires a single concept selector; "
            f"{selector} matched {len(concepts)}"
        )
    concept = concepts[0]
    from saklas.io.packs import ConceptFolder
    cf = ConceptFolder.load(concept.folder)

    if model_scope is not None:
        sid = safe_model_id(model_scope)
        if sid not in cf.tensor_models():
            raise RuntimeError(
                f"{concept.namespace}/{concept.name}: no tensor for {model_scope}"
            )
        targets = [sid]
    else:
        # Skip any GGUFs already present — re-exporting them would be a
        # no-op at best and a round-trip loss at worst.
        targets = [
            sid for sid in cf.tensor_models()
            if cf.tensor_format(sid) == "safetensors"
        ]
        if not targets:
            raise RuntimeError(
                f"{concept.namespace}/{concept.name}: no safetensors tensors to export"
            )

    # Resolve output policy.
    out_path = Path(output) if output else None
    if out_path is not None and len(targets) > 1 and out_path.suffix == ".gguf":
        raise RuntimeError(
            "multi-model export needs a directory or no --output; "
            f"got file path {out_path}"
        )

    # Bundled concepts get their folder overwritten on refresh, so any in-place
    # write would be silently reverted.  Require an explicit --output pointing
    # outside the pack folder when exporting a bundled concept.
    if out_path is None and cf.metadata.source == "bundled":
        raise RuntimeError(
            f"{concept.namespace}/{concept.name}: bundled concept — in-place "
            f"GGUF export would be lost on next refresh. Pass --output <path> "
            f"to write outside the pack folder."
        )

    written: list[Path] = []
    for sid in targets:
        profile, _meta = load_profile(str(cf.tensor_path(sid)))
        hint = model_hint or _resolve_model_hint(sid)
        if out_path is None:
            dest = concept.folder / f"{sid}.gguf"
        elif out_path.suffix == ".gguf":
            dest = out_path
        else:
            out_path.mkdir(parents=True, exist_ok=True)
            dest = out_path / f"{sid}.gguf"
        write_gguf_profile(profile, dest, model_hint=hint)
        written.append(dest)

    # Record the new files in pack.json so integrity checks pick them up
    # on next load.  Skip if every dest lives outside the pack folder.
    if any(p.parent == concept.folder for p in written):
        _update_files_map(concept.folder)
        _invalidate_selector_cache()

    return written


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
    variant: str = "raw",
) -> tuple[str, str, Optional[str]]:
    """Back `saklas push`. Returns ``(coord, repo_url, commit_sha)``."""
    from saklas.io import hf as hf_mod

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
        variant=variant,
    )
    return coord, repo_url, sha


@dataclass
class ConceptRow:
    """Structured row describing one installed concept.

    ``status`` is ``"installed"`` for a healthy folder, ``"corrupt"`` when
    ``ConceptFolder.load`` raises (the message lands in ``error``). The
    field set is presentation-stable — CLI formatters and JSON callers
    both consume the same shape.
    """
    name: str
    namespace: str
    status: str
    recommended_alpha: float
    tags: list[str]
    description: str
    source: str
    tensor_models: list[str]
    error: Optional[str] = None


@dataclass
class HfRow:
    """Structured row describing one HF-hub-discovered concept."""
    name: str
    namespace: str
    recommended_alpha: float
    tags: list[str]
    description: str
    tensor_models: list[str]


@dataclass
class ConceptInfo:
    """Structured payload for ``pack info`` against a single ``ns/name``.

    ``status`` is ``"installed"`` when the folder is present locally,
    ``"hf"`` when only the HF hub knows about it. ``recommended_alpha``
    and ``source`` are populated for the local case only — HF-hub info
    has neither.
    """
    name: str
    namespace: str
    status: str
    description: str
    long_description: str
    tags: list[str]
    tensor_models: list[str]
    recommended_alpha: Optional[float] = None
    source: Optional[str] = None


@dataclass
class PackListResult:
    """Result of ``list_concepts`` — installed rows + optional HF rows.

    ``error`` carries an HF-search failure message when the local listing
    succeeded but the HF query did not; the CLI surface renders the
    installed rows and appends the error as a parenthetical.
    """
    installed: list[ConceptRow]
    hf_rows: list[HfRow]
    error: Optional[str] = None


def _row_from_concept(c: ResolvedConcept) -> ConceptRow:
    from saklas.io.packs import ConceptFolder
    tensor_models: list[str] = []
    status = "installed"
    error: Optional[str] = None
    try:
        cf = ConceptFolder.load(c.folder)
        tensor_models = cf.tensor_models()
    except Exception as e:
        status = "corrupt"
        error = f"{type(e).__name__}: {e}"
    return ConceptRow(
        name=c.name,
        namespace=c.namespace,
        status=status,
        recommended_alpha=c.metadata.recommended_alpha,
        tags=list(c.metadata.tags),
        description=c.metadata.description,
        source=c.metadata.source,
        tensor_models=tensor_models,
        error=error,
    )


def _hf_row_from_dict(row: dict[str, Any]) -> HfRow:
    return HfRow(
        name=row.get("name", ""),
        namespace=row.get("namespace", ""),
        recommended_alpha=row.get("recommended_alpha", 0.0),
        tags=list(row.get("tags", [])),
        description=row.get("description", ""),
        tensor_models=list(row.get("tensor_models", [])),
    )


def _all_local() -> list[ResolvedConcept]:
    return resolve(Selector(kind="all", value=None))


def list_concepts(
    selector: Optional[Selector],
    *,
    hf: bool = True,
) -> PackListResult:
    """Return concepts matching the selector as a structured result.

    ``hf=True`` queries the HF hub and merges results with local installs;
    failures land in ``PackListResult.error`` rather than raising. The
    caller (CLI / programmatic) decides how to present the result.
    """
    if selector is None:
        concepts = _all_local()
    else:
        concepts = resolve(selector)
    installed_rows = [_row_from_concept(c) for c in concepts]

    if not hf:
        return PackListResult(installed=installed_rows, hf_rows=[], error=None)

    from saklas.io.hf import HFError, search_packs
    try:
        from huggingface_hub.errors import HfHubHTTPError
        _hf_http_err: type[BaseException] = HfHubHTTPError
    except ImportError:
        _hf_http_err = type("HfHubHTTPError", (Exception,), {})
    try:
        raw_hf_rows = search_packs(selector)
    except (HFError, _hf_http_err, OSError) as e:
        return PackListResult(
            installed=installed_rows,
            hf_rows=[],
            error=f"hf search unavailable: {type(e).__name__}: {e}",
        )
    return PackListResult(
        installed=installed_rows,
        hf_rows=[_hf_row_from_dict(r) for r in raw_hf_rows],
        error=None,
    )


def pack_info(
    namespace: str,
    name: str,
    *,
    hf: bool = True,
) -> Optional[ConceptInfo]:
    """Return the info payload for ``ns/name`` as a structured result.

    Local first; falls back to HF when ``hf=True`` and the folder is
    absent. Returns ``None`` when neither source has the concept.
    """
    folder = concept_dir(namespace, name)
    if folder.exists():
        from saklas.io.packs import ConceptFolder
        cf = ConceptFolder.load(folder)
        return ConceptInfo(
            name=name,
            namespace=namespace,
            status="installed",
            description=cf.metadata.description,
            long_description=cf.metadata.long_description,
            tags=list(cf.metadata.tags),
            tensor_models=cf.tensor_models(),
            recommended_alpha=cf.metadata.recommended_alpha,
            source=cf.metadata.source,
        )
    if not hf:
        return None
    from saklas.io.hf import fetch_info
    info = fetch_info(f"{namespace}/{name}")
    if not info:
        return None
    return ConceptInfo(
        name=name,
        namespace=namespace,
        status="hf",
        description=info.get("description", ""),
        long_description="",
        tags=list(info.get("tags", [])),
        tensor_models=list(info.get("tensor_models", [])),
        recommended_alpha=None,
        source=None,
    )


def search_remote_packs(query: str) -> list[HfRow]:
    """Search HF hub for saklas-pack model repos matching ``query``.

    Returns a list of ``HfRow``; the CLI layer formats / prints them.
    Raises ``ImportError`` (HF deps missing) or any other exception
    raised by the HF backend — callers translate to user-visible text.
    """
    from saklas.io.hf import search_packs
    sel = Selector(kind="name", value=query, namespace=None) if query else None
    raw = search_packs(sel)
    return [_hf_row_from_dict(r) for r in raw]
