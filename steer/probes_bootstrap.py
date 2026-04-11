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
) -> dict[str, dict[int, tuple[torch.Tensor, float]]]:
    """
    Load or extract probe vector profiles for the given categories.
    Returns dict mapping probe_name -> profile (layer_idx -> (vector, score)).
    """
    from steer.vectors import extract_contrastive, load_contrastive_pairs, load_profile, save_profile, get_cache_path

    defaults = _load_defaults()
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    model_id = model_info.get("model_id", "unknown")

    probes: dict[str, dict[int, tuple[torch.Tensor, float]]] = {}
    to_extract: list[tuple[str, str]] = []

    # Check cache first
    for cat in categories:
        cat_probes = defaults.get(cat, [])
        for probe_name in cat_probes:
            cp = get_cache_path(cache_dir, model_id, probe_name)
            if Path(cp).exists():
                try:
                    profile, _meta = load_profile(cp)
                    probes[probe_name] = profile
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

    # Load all datasets first so file I/O doesn't interleave with GPU work
    datasets_to_extract = []
    for name, cp in to_extract:
        ds_path = datasets_dir / f"{name}.json"
        if not ds_path.exists():
            log.warning("Dataset %s.json not found for probe %s, skipping", name, name)
            continue
        ds = load_contrastive_pairs(str(ds_path))
        datasets_to_extract.append((name, cp, ds))

    for name, cp, ds in tqdm(datasets_to_extract, desc="Extracting probes", unit="probe"):
        try:
            profile = extract_contrastive(model, tokenizer, ds["pairs"], layers=layers)
            probes[name] = profile
            save_profile(profile, cp, {
                "concept": name,
                "model_id": model_id,
                "num_pairs": len(ds["pairs"]),
            })
        except Exception as e:
            log.warning("Contrastive extraction failed for %s: %s", name, e)

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
