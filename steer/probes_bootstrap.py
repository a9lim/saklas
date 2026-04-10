"""Bootstrap default probe vectors from the probe library config."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

log = logging.getLogger(__name__)

DEFAULTS_PATH = Path(__file__).parent / "probes" / "defaults.json"


def bootstrap_probes(
    model,
    tokenizer,
    layers,
    model_info: dict,
    categories: list[str],
    cache_dir: str,
) -> dict[str, torch.Tensor]:
    """
    Load or extract probe vectors for the given categories.
    Returns dict mapping probe_name -> unit vector tensor.
    Uses batched extraction for efficiency.
    """
    from steer.vectors import extract_actadd_batched, load_vector, save_vector, get_cache_path

    defaults = _load_defaults()
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    model_id = model_info.get("model_id", "unknown")
    num_layers = model_info["num_layers"]
    # Default probe layer: penultimate
    probe_layer = num_layers - 2

    probes: dict[str, torch.Tensor] = {}
    to_extract: list[str] = []

    # Check cache first
    for cat in categories:
        cat_probes = defaults.get(cat, {})
        for probe_name, probe_cfg in cat_probes.items():
            method = probe_cfg.get("method", "actadd")
            cp = get_cache_path(cache_dir, model_id, probe_name, probe_layer, method)
            try:
                vec, _meta = load_vector(cp)
                probes[probe_name] = vec
                log.debug("Loaded cached probe: %s", probe_name)
            except FileNotFoundError:
                to_extract.append(probe_name)
            except Exception as e:
                log.warning("Corrupt cache for %s, re-extracting: %s", probe_name, e)
                to_extract.append(probe_name)

    if not to_extract:
        return probes

    # Batch extract missing probes
    log.info("Extracting %d probes...", len(to_extract))

    # Group by method
    actadd_concepts = []
    caa_concepts = []
    for probe_name in to_extract:
        cfg = _find_probe_config(defaults, probe_name)
        if cfg and cfg.get("method") == "caa":
            caa_concepts.append((probe_name, cfg))
        else:
            actadd_concepts.append((probe_name, cfg or {}))

    # Batched ActAdd extraction
    if actadd_concepts:
        concepts = [name for name, _ in actadd_concepts]
        try:
            vectors = extract_actadd_batched(
                model, tokenizer, concepts, probe_layer, layers=layers,
            )
            for name, vec in vectors.items():
                probes[name] = vec
                cp = get_cache_path(cache_dir, model_id, name, probe_layer, "actadd")
                save_vector(vec, cp, {
                    "concept": name,
                    "method": "actadd",
                    "layer_idx": probe_layer,
                    "model_id": model_id,
                    "hidden_dim": vec.shape[0],
                })
        except Exception as e:
            log.warning("Batched extraction failed, falling back to sequential: %s", e)
            from steer.vectors import extract_actadd
            for name, _ in actadd_concepts:
                try:
                    vec = extract_actadd(model, tokenizer, name, probe_layer, layers=layers)
                    probes[name] = vec
                    cp = get_cache_path(cache_dir, model_id, name, probe_layer, "actadd")
                    save_vector(vec, cp, {
                        "concept": name,
                        "method": "actadd",
                        "layer_idx": probe_layer,
                        "model_id": model_id,
                        "hidden_dim": vec.shape[0],
                    })
                except Exception as e2:
                    log.warning("Failed to extract probe %s: %s", name, e2)

    # CAA extraction for concepts with dataset files
    if caa_concepts:
        from steer.vectors import extract_caa, load_contrastive_pairs
        datasets_dir = Path(__file__).parent / "datasets"
        for name, cfg in caa_concepts:
            dataset_file = cfg.get("dataset")
            if dataset_file:
                ds_path = datasets_dir / dataset_file
                if ds_path.exists():
                    try:
                        ds = load_contrastive_pairs(str(ds_path))
                        vec = extract_caa(model, tokenizer, ds["pairs"], probe_layer, layers=layers)
                        probes[name] = vec
                        cp = get_cache_path(cache_dir, model_id, name, probe_layer, "caa")
                        save_vector(vec, cp, {
                            "concept": name,
                            "method": "caa",
                            "layer_idx": probe_layer,
                            "model_id": model_id,
                            "hidden_dim": vec.shape[0],
                            "num_pairs": len(ds["pairs"]),
                        })
                    except Exception as e:
                        log.warning("CAA extraction failed for %s: %s", name, e)

    return probes


def _load_defaults() -> dict:
    if DEFAULTS_PATH.exists():
        with open(DEFAULTS_PATH) as f:
            return json.load(f)
    return {}


def _find_probe_config(defaults: dict, probe_name: str) -> dict | None:
    for cat_probes in defaults.values():
        if probe_name in cat_probes:
            return cat_probes[probe_name]
    return None
