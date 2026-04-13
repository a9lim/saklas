"""Bootstrap default probe vectors from the ~/.saklas/ concept-folder layout."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from saklas.packs import (
    ConceptFolder, PackFormatError, Sidecar,
    hash_file, is_stale, materialize_bundled, version_mismatch,
)
from saklas.paths import (
    concept_dir, model_dir, neutral_statements_path, safe_model_id, vectors_dir,
)
from saklas.vectors import (
    compute_layer_means, extract_contrastive, load_contrastive_pairs,
    load_profile, save_profile,
)

log = logging.getLogger(__name__)

_LAYER_MEANS_NAME = "layer_means"


def load_defaults() -> dict[str, list[str]]:
    """Return {tag: [concept_name, ...]} for the default/ namespace.

    Triggers first-run materialization of bundled data into ~/.saklas/.
    """
    materialize_bundled()
    root = vectors_dir() / "default"
    if not root.is_dir():
        return {}
    by_tag: dict[str, list[str]] = {}
    for cdir in sorted(root.iterdir()):
        if not cdir.is_dir() or not (cdir / "pack.json").is_file():
            continue
        try:
            cf = ConceptFolder.load(cdir)
        except PackFormatError as e:
            log.warning("skipping %s: %s", cdir.name, e)
            continue
        for tag in cf.metadata.tags or ["uncategorized"]:
            by_tag.setdefault(tag, []).append(cf.metadata.name)
    return by_tag


def bootstrap_layer_means(
    model, tokenizer, layers, model_info: dict, *_unused,
) -> dict[int, torch.Tensor]:
    """Load or compute per-layer mean activations for probe centering.

    Stored at ~/.saklas/models/<safe_id>/layer_means.safetensors with a slim
    sidecar. Stale if neutral_statements.json has changed since extraction.
    """
    model_id = model_info.get("model_id", "unknown")
    md = model_dir(model_id)
    md.mkdir(parents=True, exist_ok=True)
    ts_path = md / f"{_LAYER_MEANS_NAME}.safetensors"
    sc_path = md / f"{_LAYER_MEANS_NAME}.json"

    current_ns_hash: str | None = None
    if neutral_statements_path().exists():
        current_ns_hash = hash_file(neutral_statements_path())

    if ts_path.exists() and sc_path.exists():
        try:
            sc = Sidecar.load(sc_path)
            if current_ns_hash is None or sc.statements_sha256 == current_ns_hash:
                profile, _ = load_profile(str(ts_path))
                log.debug("Loaded cached layer means")
                return {idx: vec for idx, (vec, _score) in profile.items()}
            log.info("Layer means stale (neutral_statements changed); recomputing")
        except Exception as e:
            log.warning("Corrupt layer means cache, recomputing: %s", e)

    log.info("Computing layer means (one-time per model)...")
    means = compute_layer_means(model, tokenizer, layers)
    profile = {idx: (vec, 1.0) for idx, vec in means.items()}
    save_profile(profile, str(ts_path), {
        "method": "layer_means",
        "statements_sha256": current_ns_hash or "",
    })
    return means


def bootstrap_probes(
    model, tokenizer, layers, model_info: dict, categories: list[str], *_unused,
) -> dict[str, dict[int, tuple[torch.Tensor, float]]]:
    """Load or extract probe vector profiles for the given categories."""
    from saklas import __version__ as _saklas_version

    defaults = load_defaults()
    model_id = model_info.get("model_id", "unknown")
    sid = safe_model_id(model_id)

    probes: dict[str, dict[int, tuple[torch.Tensor, float]]] = {}
    to_extract: list[tuple[str, Path, Path]] = []

    for cat in categories:
        for probe_name in defaults.get(cat, []):
            cdir = concept_dir("default", probe_name)
            ts = cdir / f"{sid}.safetensors"
            sc_path = cdir / f"{sid}.json"
            if ts.exists() and sc_path.exists():
                try:
                    profile, meta = load_profile(str(ts))
                    # load_profile already parsed the sidecar dict; build a
                    # Sidecar from it instead of re-reading from disk.
                    sc = Sidecar(
                        method=meta["method"],
                        scores={int(k): float(v) for k, v in meta.get("scores", {}).items()},
                        saklas_version=meta["saklas_version"],
                        statements_sha256=meta.get("statements_sha256"),
                        components=meta.get("components"),
                    )
                    probes[probe_name] = profile
                    stmts = cdir / "statements.json"
                    if stmts.exists():
                        current = hash_file(stmts)
                        if is_stale(current, sc):
                            log.warning(
                                "%s: statements changed since extraction; consider -r default/%s",
                                probe_name, probe_name,
                            )
                    if version_mismatch(sc, _saklas_version):
                        log.warning("%s: extracted with older saklas version", probe_name)
                    continue
                except Exception as e:
                    log.warning("Corrupt cache for %s, re-extracting: %s", probe_name, e)
            to_extract.append((probe_name, cdir, ts))

    if not to_extract:
        return probes

    log.info("Extracting %d probes...", len(to_extract))
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **kw): return x

    datasets_to_extract = []
    for name, cdir, ts in to_extract:
        stmts_path = cdir / "statements.json"
        if not stmts_path.exists():
            log.warning("statements.json missing for %s; skipping", name)
            continue
        pairs_data = load_contrastive_pairs(str(stmts_path))
        datasets_to_extract.append((name, cdir, ts, pairs_data, stmts_path))

    model_device = next(model.parameters()).device
    for name, cdir, ts, ds, stmts_path in tqdm(datasets_to_extract, desc="Extracting probes", unit="probe"):
        try:
            profile = extract_contrastive(model, tokenizer, ds["pairs"], layers=layers)
            probes[name] = profile
            save_profile(profile, str(ts), {
                "method": "contrastive_pca",
                "statements_sha256": hash_file(stmts_path),
            })
        except Exception as e:
            log.warning("Contrastive extraction failed for %s: %s", name, e)
        if model_device.type == "mps":
            torch.mps.empty_cache()
    return probes
