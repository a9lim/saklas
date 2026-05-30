"""Hugging Face Hub consumption wrappers for saklas manifold distribution.

The read-side counterpart to :mod:`saklas.io.hf`.  Manifold-folder
artifacts (``manifold.json`` + ``nodes/*.json`` + per-model fitted
``<safe_model>.safetensors``) ride the same HF *model*-repo convention
packs do — safetensors is model-hub-native and ``base_model``
frontmatter gives reverse-link discoverability.  The tagging convention
diverges: a manifold repo carries ``saklas-manifold`` (parallel to
``saklas-pack``) so the search query is unambiguous.

This module owns the pure-IO HF surface (pull + search + fetch_info).
The folder format itself lives in :mod:`saklas.io.manifolds`; the
server route layer composes them with ``manifolds_dir`` /
``ManifoldFolder.load`` for the install path.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Optional

from saklas.core.errors import SaklasError
from saklas.io.atomic import write_bytes_atomic
from saklas.io.hf import (
    HFError,
    _hf_api,
    _hf_hub_download,
    _hf_snapshot_download,
    split_revision,
)
from saklas.io.manifolds import (
    MANIFOLD_FORMAT_VERSION,
    ManifoldFolder,
    ManifoldFormatError,
)
from saklas.io.packs import NAME_REGEX

# Mirror :data:`saklas.io.hf._HF_SEARCH_CAP`.  Kept independent so the
# manifold side can diverge later (e.g. larger cap once a few canonical
# manifolds are published) without touching the pack ceiling.
_HF_SEARCH_CAP = 20


# ---------------------------------------------------------------------- pull --


def _download(
    coord: str,
    *,
    revision: Optional[str] = None,
) -> str:
    """Snapshot-download ``<owner>/<repo>`` from the HF model hub.

    Thin wrapper over :func:`saklas.io.hf._hf_snapshot_download` for
    error-shape parity with the pack pull path.  No allow-patterns
    filter: a manifold folder is small (a manifest + a ``nodes/`` corpus
    of short JSON files + at most a few safetensors), so the full
    snapshot is the simplest correct read.
    """
    kwargs: dict[str, Any] = {"repo_id": coord}
    if revision is not None:
        kwargs["revision"] = revision
    try:
        return _hf_snapshot_download(**kwargs)
    except Exception as e:
        label = f"{coord}@{revision}" if revision else coord
        raise HFError(f"{label}: not found ({e})") from e


def pull_manifold(
    coord: str,
    target_folder: Path,
    *,
    force: bool,
    revision: Optional[str] = None,
) -> Path:
    """Download ``coord`` from HF and install into ``target_folder``.

    Stage-verify-swap discipline (same shape ``pull_pack`` uses):
    the manifold folder is built under ``<target_folder>.staging/``,
    then validated by ``ManifoldFolder.load`` (which already checks
    format version + ``NAME_REGEX`` + the ``files`` integrity manifest
    when populated).  Only after a clean load does the staging dir
    atomically swap into place via ``target → .bak``, ``staging →
    target``, ``rmtree .bak``.  A crash mid-swap is recoverable from
    ``.bak``.

    Unlike packs, manifold folders carry no synthesis path — if the
    repo doesn't have a ``manifold.json`` at root we refuse rather
    than fabricate one (a manifold's geometry can't be inferred from
    a bare safetensors dump).  Repos pre-dating the manifold format
    should be re-authored, not auto-installed.
    """
    tmp_dir = Path(_download(coord, revision=revision))

    if not (tmp_dir / "manifold.json").is_file():
        raise HFError(
            f"{coord}: HF repo has no manifold.json at root — saklas "
            f"manifolds must be published with the manifest in place "
            f"(see `saklas vector manifold push`)."
        )

    source = f"hf://{coord}@{revision}" if revision else f"hf://{coord}"

    if target_folder.exists() and not force:
        raise HFError(f"{target_folder} exists; pass force=True to overwrite")

    staging = target_folder.with_name(target_folder.name + ".staging")
    backup = target_folder.with_name(target_folder.name + ".bak")

    # Crash-recovery: a previous pull that died after target → .bak but
    # before staging → target left a valid prior install in .bak only.
    # Restore it before starting a new pull so a failed new staging
    # doesn't lose the prior good install.
    if not target_folder.exists() and backup.exists():
        try:
            backup.rename(target_folder)
        except OSError:
            pass

    if staging.exists():
        shutil.rmtree(staging)
    if backup.exists():
        shutil.rmtree(backup)

    staging.mkdir(parents=True, exist_ok=True)
    try:
        _install_manifold(tmp_dir, staging, coord)
        # Validate the staged folder against the same loader the
        # session will see — catches format-version mismatches and a
        # populated-but-broken ``files`` manifest before we touch the
        # target.
        try:
            staged = ManifoldFolder.load(staging)
        except ManifoldFormatError as e:
            raise HFError(f"{coord}: staged manifold failed validation ({e})") from e
        # Record the HF coord as the manifold's source so ``refresh_manifold``
        # can re-pull from the same place (mirrors how ``pull_pack``
        # rewrites ``pack.json.source``).  ``write_metadata`` rebuilds the
        # manifest from the just-loaded folder and re-hashes ``files`` —
        # so the staged manifest stays self-consistent for the re-validate
        # below.
        staged.source = source
        staged.write_metadata(files=staged.files)
        try:
            ManifoldFolder.load(staging)
        except ManifoldFormatError as e:
            raise HFError(
                f"{coord}: staged manifold failed re-validation after "
                f"source stamp ({e})"
            ) from e
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    had_existing = target_folder.exists()
    if had_existing:
        try:
            target_folder.rename(backup)
        except OSError as e:
            shutil.rmtree(staging, ignore_errors=True)
            raise HFError(
                f"{coord}: could not move existing install aside ({e})"
            ) from e

    try:
        staging.rename(target_folder)
    except OSError as e:
        if had_existing and backup.exists() and not target_folder.exists():
            try:
                backup.rename(target_folder)
            except OSError:
                pass
        shutil.rmtree(staging, ignore_errors=True)
        raise HFError(
            f"{coord}: could not promote staging into place ({e})"
        ) from e

    if had_existing:
        shutil.rmtree(backup, ignore_errors=True)

    return target_folder


def _install_manifold(tmp_dir: Path, target_folder: Path, _coord: str) -> None:
    """Copy a manifold folder's tree from the snapshot dir to ``target_folder``.

    Preserves the directory layout the format expects:
      * ``manifold.json`` at the root
      * ``nodes/NN_<label>.json`` corpus files under ``nodes/``
      * Optional per-model fitted ``<safe_model>.safetensors`` + ``.json``
        sidecars at the root
      * Optional ``scenarios.json`` provenance file (discover-mode)

    Anything else at the snapshot root (README, .gitattributes, etc.) is
    skipped — those are HF-side artifacts, not part of the manifold
    format.
    """
    _ALLOWED_ROOT = {"manifold.json", "scenarios.json"}
    _ALLOWED_SUFFIXES = (".safetensors", ".json")

    for entry in sorted(tmp_dir.iterdir()):
        if entry.is_file():
            if (
                entry.name in _ALLOWED_ROOT
                or entry.suffix in _ALLOWED_SUFFIXES
            ):
                write_bytes_atomic(
                    target_folder / entry.name, entry.read_bytes(),
                )
        elif entry.is_dir() and entry.name == "nodes":
            (target_folder / "nodes").mkdir(parents=True, exist_ok=True)
            for child in sorted(entry.iterdir()):
                if child.is_file() and child.suffix == ".json":
                    write_bytes_atomic(
                        target_folder / "nodes" / child.name,
                        child.read_bytes(),
                    )


# ---------------------------------------------------------------------- push --
#
# HF upload, mirroring :func:`saklas.io.hf.push_pack` in shape: stage a
# filtered copy of the folder (so we can add README + .gitattributes
# without mutating the source), then one ``upload_folder``.  The
# divergences from the pack push are the ``saklas-manifold`` repo tag, the
# manifold-shaped model card, and that the corpus (``manifold.json`` +
# ``nodes/*``) is *always* included — a manifold without its node corpus
# can't be re-fit, so a tensors-only manifold push would be useless.


def _manifold_sidecar_stem_to_hf_coord(stem: str) -> Optional[str]:
    """Convert a fitted-tensor stem back to its base-model HF coord.

    Mirrors :func:`saklas.io.hf._sidecar_stem_to_hf_coord`: strips any
    variant suffix (``_sae-<release>`` / ``_from-<safe_src>``) so the
    ``base_model:`` frontmatter lists the clean base model, then flips
    ``__`` → ``/``.  Returns ``None`` for stems that don't parse.
    """
    from saklas.io.paths import parse_tensor_filename

    parsed = parse_tensor_filename(f"{stem}.safetensors")
    if parsed is None:
        return None
    safe_model, _variant = parsed
    return safe_model.replace("__", "/")


def _render_manifold_card(
    mf: ManifoldFolder, tensor_stems: list[str], coord: str,
) -> str:
    """Build a HF model card (YAML frontmatter + markdown body) for a manifold.

    Parallel to :func:`saklas.io.hf._render_model_card`, but manifold-
    shaped: the table lists the manifold's domain / node count / fit_mode
    rather than a recommended alpha.  ``base_model:`` is deduped over the
    fitted tensor stems.
    """
    base_models = sorted({
        c for stem in tensor_stems
        if (c := _manifold_sidecar_stem_to_hf_coord(stem)) is not None
    })
    tags = ["saklas-manifold", "activation-steering", "steering-manifold"]

    fm = ["---", "library_name: saklas", "tags:"]
    fm += [f"  - {t}" for t in tags]
    if base_models:
        fm.append("base_model:")
        fm += [f"  - {bm}" for bm in base_models]
        fm.append("base_model_relation: adapter")
    fm.append("---")

    from saklas.io.manifolds import domain_label

    if mf.fit_mode == "authored" and mf.domain:
        dom_lbl = domain_label(mf.domain)
    else:
        dom_lbl = f"discover-{mf.fit_mode}"

    body: list[str] = [
        f"# {mf.name}",
        "",
        mf.description,
        "",
        f"**Domain:** `{dom_lbl}`  |  **fit_mode:** `{mf.fit_mode}`  |  "
        f"**nodes:** {len(mf.node_labels)}",
        "",
        "## Install",
        "",
        "```bash",
        f"saklas vector manifold install {coord}",
        "```",
        "",
        "## Nodes",
        "",
        ", ".join(f"`{label}`" for label in mf.node_labels),
        "",
    ]
    if tensor_stems:
        body += [
            "## Fitted tensors",
            "",
            "| base model | variant |",
            "| --- | --- |",
        ]
        from saklas.io.paths import parse_tensor_filename
        for stem in sorted(tensor_stems):
            base = _manifold_sidecar_stem_to_hf_coord(stem) or stem.replace("__", "/")
            parsed = parse_tensor_filename(f"{stem}.safetensors")
            variant = "raw" if (parsed is None or parsed[1] is None) else parsed[1]
            body.append(f"| `{base}` | `{variant}` |")
        body.append("")

    body += ["---", "", "Generated by `saklas vector manifold push`.", ""]
    return "\n".join(fm) + "\n\n" + "\n".join(body)


def _manifold_variant_matches(key: str, variant: str) -> bool:
    """Variant filter for a manifold tensor, mirroring ``push_pack``'s.

    ``key`` is the parsed variant slug (``"raw"`` / ``"sae-<release>"`` /
    ``"from-<safe_src>"``); ``variant`` is one of ``"raw"`` / ``"sae"`` /
    ``"from"`` / ``"all"``.
    """
    if variant == "all":
        return True
    if variant == "raw":
        return key == "raw"
    if variant == "sae":
        return key.startswith("sae-")
    if variant == "from":
        return key.startswith("from-")
    return False


def push_manifold(
    folder: Path,
    coord: str,
    *,
    private: bool = False,
    model_scope: Optional[str] = None,
    variant: str = "raw",
    dry_run: bool = False,
) -> tuple[str, Optional[str]]:
    """Push a manifold folder to HF as a model repo.

    Mirrors :func:`saklas.io.hf.push_pack` exactly in shape — stage a
    filtered copy (adding README.md + .gitattributes without mutating the
    source), then one atomic ``upload_folder``.  Returns
    ``(repo_url, commit_sha)``; ``sha`` is ``None`` on dry-run.

    The corpus is *always* uploaded — ``manifold.json`` plus every
    ``nodes/*.json`` — because a manifold can't be re-fit without it.
    Per-model fitted ``<safe>.safetensors`` + ``.json`` sidecars are
    filtered the way ``push_pack`` filters tensors:

    - ``model_scope`` restricts to one base model (``safe_model_id``).
    - ``variant`` filters tensor flavor: ``"raw"`` (default) only
      unsuffixed, ``"sae"`` only ``_sae-*``, ``"from"`` only ``_from-*``,
      ``"all"`` every variant.  Sidecars follow their partner tensor.

    A staged manifest with no tensors is still a valid push — the
    corpus alone re-fits on the consumer side, unlike a pack where a
    tensors-and-statements-empty push is rejected.
    """
    import tempfile

    from saklas.io.atomic import write_bytes_atomic
    from saklas.io.manifolds import ManifoldFolder, hash_manifold_files
    from saklas.io.paths import parse_tensor_filename, safe_model_id as _safe_id

    mf = ManifoldFolder.load(folder)  # runs integrity check

    scope_safe: Optional[str] = None
    if model_scope is not None:
        scope_safe = _safe_id(model_scope)

    staging = Path(tempfile.mkdtemp(prefix="saklas-manifold-push-"))
    try:
        # Always stage the corpus: manifold.json + the full nodes/ tree.
        write_bytes_atomic(
            staging / "manifold.json", (folder / "manifold.json").read_bytes(),
        )
        nodes_src = folder / "nodes"
        if nodes_src.is_dir():
            for child in sorted(nodes_src.iterdir()):
                if child.is_file() and child.suffix == ".json":
                    write_bytes_atomic(
                        staging / "nodes" / child.name, child.read_bytes(),
                    )
        # Optional discover-mode provenance file.
        scen = folder / "scenarios.json"
        if scen.is_file():
            write_bytes_atomic(staging / "scenarios.json", scen.read_bytes())

        # Stage the fitted tensors that survive the model/variant filter,
        # each with its sidecar.
        kept_stems: list[str] = []
        for ts in sorted(folder.glob("*.safetensors")):
            parsed = parse_tensor_filename(ts.name)
            if parsed is None:
                continue
            file_model, var_slug = parsed
            if scope_safe is not None and file_model != scope_safe:
                continue
            vkey = "raw" if var_slug is None else var_slug
            if not _manifold_variant_matches(vkey, variant):
                continue
            write_bytes_atomic(staging / ts.name, ts.read_bytes())
            kept_stems.append(ts.stem)
            sc = ts.with_suffix(".json")
            if sc.exists():
                write_bytes_atomic(staging / sc.name, sc.read_bytes())

        # Re-hash the staged copy so the uploaded manifest matches the
        # bytes we upload (a model/variant filter changes the file set).
        # Patch the staged ``manifold.json``'s ``files`` map directly
        # rather than re-loading first: the verbatim-copied manifest
        # still references the *unfiltered* tensor set, so a
        # ``ManifoldFolder.load`` of the staging dir would fail its
        # integrity check against files the filter excluded.
        from saklas.io.atomic import write_json_atomic as _write_json_atomic
        staged_manifest = staging / "manifold.json"
        with open(staged_manifest) as f:
            staged_data = json.load(f)
        staged_data["files"] = hash_manifold_files(staging)
        _write_json_atomic(staged_manifest, staged_data)

        write_bytes_atomic(
            staging / ".gitattributes",
            b"*.safetensors filter=lfs diff=lfs merge=lfs -text\n",
        )
        write_bytes_atomic(
            staging / "README.md",
            _render_manifold_card(mf, kept_stems, coord).encode("utf-8"),
        )

        repo_url = f"https://huggingface.co/{coord}"
        if dry_run:
            return (repo_url, None)

        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(
            repo_id=coord, repo_type="model", private=private, exist_ok=True,
        )
        info = api.upload_folder(
            repo_id=coord,
            repo_type="model",
            folder_path=str(staging),
            commit_message=f"saklas manifold push: {mf.name}",
        )
        sha = getattr(info, "oid", None) or getattr(info, "commit_sha", None)
        return (repo_url, sha)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


# -------------------------------------------------------------------- search --


def search_manifolds(query: Optional[str]) -> list[dict[str, Any]]:
    """Search HF for ``saklas-manifold``-tagged model repos.

    Returns row dicts ready for display — same shape ``search_packs``
    emits so the webui can reuse its row-rendering machinery.  At most
    ``_HF_SEARCH_CAP`` rows.  ``query`` is a free-text substring; an
    empty / ``None`` query lists tagged repos by recency.
    """
    api = _hf_api()
    required_tags: list[str] = ["saklas-manifold"]
    kwargs: dict[str, Any] = dict(filter=required_tags, limit=_HF_SEARCH_CAP)
    if query:
        kwargs["search"] = query

    try:
        results = list(api.list_models(**kwargs))
    except TypeError:
        # Older huggingface_hub uses ``tags`` instead of ``filter``.
        kwargs.pop("filter", None)
        kwargs["tags"] = required_tags
        results = list(api.list_models(**kwargs))

    rows: list[dict[str, Any]] = []
    for r in results[:_HF_SEARCH_CAP]:
        coord = r.id
        if "/" in coord:
            ns, nm = coord.split("/", 1)
        else:
            ns, nm = "", coord

        raw_tags = getattr(r, "tags", None) or []
        tags = (
            [str(t) for t in raw_tags]
            if isinstance(raw_tags, (list, tuple))
            else []
        )
        raw_desc = getattr(r, "description", "") or ""
        description = raw_desc if isinstance(raw_desc, str) else ""

        # Fetch the manifest for fields list_models doesn't surface:
        # the domain label, node count, fit_mode, and any fitted-model
        # tensor stems.  Best-effort — missing info just yields empty
        # fields so the search list still renders.
        info: dict[str, Any] = {}
        if not description or not tags:
            try:
                info = fetch_manifold_info(coord)
            except Exception:
                info = {}

        row: dict[str, Any] = {
            "name": info.get("name", nm),
            "namespace": info.get("namespace", ns),
            "description": info.get("description", description),
            "tags": info.get("tags", tags),
            "node_count": info.get("node_count", 0),
            "domain_label": info.get("domain_label", "?"),
            "fit_mode": info.get("fit_mode", "authored"),
            "tensor_models": info.get("tensor_models", []),
        }
        rows.append(row)

    return rows


def fetch_manifold_info(
    coord: str, revision: Optional[str] = None,
) -> dict[str, Any]:
    """Fetch minimal info about an HF saklas-manifold repo without a full pull.

    Pulls only ``manifold.json`` plus the repo's file listing — same
    cheap probe ``hf.fetch_info`` uses for packs.  Returns a dict the
    search row renderer can consume; raises :class:`HFError` on
    transport / format failure.
    """
    label = f"{coord}@{revision}" if revision else coord
    try:
        dl_kwargs: dict[str, Any] = {}
        if revision is not None:
            dl_kwargs["revision"] = revision
        mj_path = _hf_hub_download(coord, "manifold.json", **dl_kwargs)
        with open(mj_path) as f:
            data = json.load(f)
        api = _hf_api()
        list_kwargs: dict[str, Any] = {"repo_id": coord, "repo_type": "model"}
        if revision is not None:
            list_kwargs["revision"] = revision
        files = api.list_repo_files(**list_kwargs)
    except Exception as e:
        raise HFError(f"{label}: fetch_manifold_info failed ({e})") from e

    fmt_version = data.get("format_version")
    if fmt_version is not None and fmt_version > MANIFOLD_FORMAT_VERSION:
        raise HFError(
            f"{label}: manifold format_version {fmt_version} is newer than "
            f"this saklas understands ({MANIFOLD_FORMAT_VERSION}); update saklas."
        )

    tensor_models = sorted(
        Path(f).stem for f in files
        if f.endswith(".safetensors")
    )

    ns, _, nm = coord.partition("/")
    domain = data.get("domain") or {}
    fit_mode = data.get("fit_mode", "authored")
    nodes = data.get("nodes") or []
    node_count = len(nodes) if isinstance(nodes, list) else 0

    if domain:
        kind = domain.get("type", "?")
        if kind == "box":
            n = len(domain.get("axes") or [])
        elif kind == "sphere":
            n = int(domain.get("dim", 0) or 0)
        elif kind == "custom":
            n = int(domain.get("embed_dim", 0) or 0)
        else:
            n = 0
        domain_label = f"{kind}({n}d)"
    elif fit_mode in {"pca", "spectral"}:
        domain_label = f"discover-{fit_mode}"
    else:
        domain_label = "?"

    raw_tags = data.get("tags") or []
    tags = [str(t) for t in raw_tags] if isinstance(raw_tags, (list, tuple)) else []

    return {
        "name": str(data.get("name") or nm),
        "namespace": ns,
        "description": str(data.get("description") or ""),
        "tags": tags,
        "node_count": node_count,
        "domain_label": domain_label,
        "fit_mode": fit_mode,
        "tensor_models": tensor_models,
    }


# ------------------------------------------------------------------ install --


class ManifoldInstallConflict(RuntimeError, SaklasError):
    """A folder already exists at the install target and ``force=False``."""

    def user_message(self) -> tuple[int, str]:
        return (409, str(self) or self.__class__.__name__)


def install_manifold(
    target: str,
    as_: Optional[str] = None,
    *,
    force: bool = False,
) -> Path:
    """Install a manifold from an HF coord or a local folder.

    Mirrors :func:`saklas.io.cache_ops.install` for packs — top-level
    orchestration over the HF + folder-copy primitives.  ``target`` is
    one of:
      * ``<ns>/<name>[@revision]`` — HF pull via :func:`pull_manifold`
      * a local path to a folder — copy install

    ``as_`` overrides the destination ``<dst_ns>/<dst_name>`` (must be
    fully qualified — manifold folders are always namespace-rooted).
    ``force`` overwrites an existing destination.
    """
    from saklas.io.paths import manifold_dir

    p = Path(target)
    if p.exists() and p.is_dir():
        return _install_local_manifold(p, as_=as_, force=force)

    coord, revision = split_revision(target)
    if "/" not in coord:
        raise ValueError(
            f"install target must be '<ns>/<name>[@revision]' or a folder path: "
            f"{target!r}"
        )

    _ns, name = coord.split("/", 1)
    if as_:
        if "/" not in as_:
            raise ValueError(f"as_ must be '<ns>/<name>', got {as_!r}")
        dst_ns, dst_name = as_.split("/", 1)
    else:
        dst_ns, dst_name = "local", name

    if not NAME_REGEX.match(dst_name):
        raise ValueError(
            f"install target name {dst_name!r} doesn't match NAME_REGEX "
            f"{NAME_REGEX.pattern}"
        )
    dst = manifold_dir(dst_ns, dst_name)
    if dst.exists() and not force:
        raise ManifoldInstallConflict(
            f"{dst} already exists; pass force=True or as_=<ns>/<name> to relocate"
        )
    pull_manifold(coord, target_folder=dst, force=force, revision=revision)
    return dst


def _install_local_manifold(
    src: Path, *, as_: Optional[str] = None, force: bool = False,
) -> Path:
    """Copy a local manifold folder into the cache.

    Validates the source through ``ManifoldFolder.load`` first so a
    malformed folder fails fast at the source rather than after the
    copy.  ``as_`` defaults to ``local/<src.name>``.
    """
    from saklas.io.paths import manifold_dir

    try:
        ManifoldFolder.load(src)
    except ManifoldFormatError as e:
        raise ValueError(f"{src}: source folder is not a manifold ({e})") from e

    if as_:
        if "/" not in as_:
            raise ValueError(f"as_ must be '<ns>/<name>', got {as_!r}")
        dst_ns, dst_name = as_.split("/", 1)
    else:
        dst_ns, dst_name = "local", src.name

    if not NAME_REGEX.match(dst_name):
        raise ValueError(
            f"install target name {dst_name!r} doesn't match NAME_REGEX"
        )

    dst = manifold_dir(dst_ns, dst_name)
    if dst.exists():
        if not force:
            raise ManifoldInstallConflict(
                f"{dst} already exists; pass force=True or as_=<ns>/<name>"
            )
        shutil.rmtree(dst)

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)
    return dst
