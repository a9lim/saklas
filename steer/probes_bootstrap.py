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
    from steer.vectors import extract_caa, load_contrastive_pairs, load_vector, save_vector, get_cache_path

    defaults = _load_defaults()
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    model_id = model_info.get("model_id", "unknown")
    num_layers = model_info["num_layers"]
    # Default probe layer: penultimate
    probe_layer = num_layers - 2

    probes: dict[str, torch.Tensor] = {}
    to_extract: list[tuple[str, str]] = []

    # Check cache first
    for cat in categories:
        cat_probes = defaults.get(cat, [])
        for probe_name in cat_probes:
            cp = get_cache_path(cache_dir, model_id, probe_name, probe_layer, "caa")
            if Path(cp).exists():
                try:
                    vec, _meta = load_vector(cp)
                    probes[probe_name] = vec
                    log.debug("Loaded cached probe: %s", probe_name)
                except Exception as e:
                    log.warning("Corrupt cache for %s, re-extracting: %s", probe_name, e)
                    to_extract.append((probe_name, cp))
            else:
                to_extract.append((probe_name, cp))

    if not to_extract:
        return probes

    log.info("Extracting %d probes...", len(to_extract))

    from tqdm import tqdm
    datasets_dir = Path(__file__).parent / "datasets"
    for name, cp in tqdm(to_extract, desc="  Extracting probes", unit="probe"):
        ds_path = datasets_dir / f"{name}.json"
        if not ds_path.exists():
            log.warning("Dataset %s.json not found for probe %s, skipping", name, name)
            continue
        try:
            ds = load_contrastive_pairs(str(ds_path))
            vec = extract_caa(model, tokenizer, ds["pairs"], probe_layer, layers=layers)
            probes[name] = vec
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


_DEFAULTS_CACHE: dict | None = None

def _load_defaults() -> dict:
    global _DEFAULTS_CACHE
    if _DEFAULTS_CACHE is None:
        if DEFAULTS_PATH.exists():
            with open(DEFAULTS_PATH) as f:
                _DEFAULTS_CACHE = json.load(f)
        else:
            _DEFAULTS_CACHE = {}
    return _DEFAULTS_CACHE
