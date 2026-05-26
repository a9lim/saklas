#!/usr/bin/env python3
"""Cosine-compare archived vs re-extracted vectors for a target model.

Walks ``~/.saklas/vectors/default-archive/`` and ``~/.saklas/vectors/default/``
in parallel, loading both per-concept tensors for the requested model.
Reports magnitude-weighted cosine similarity per concept, sorted to
surface the biggest direction shifts first.

Use this after regenerating statements + re-extracting under a new
prompt regime to measure how much the extraction direction actually
moved.  Mean cosine near 1.0 means the slim prompts produced
substantively identical vectors; lower means real direction shifts.

Usage:
    python scripts/compare_archive_vs_default.py
    python scripts/compare_archive_vs_default.py --model google/gemma-4-31b-it
    python scripts/compare_archive_vs_default.py --whitener  # Mahalanobis
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from saklas.core.profile import Profile

SAKLAS_HOME = Path.home() / ".saklas"
ARCHIVE_ROOT = SAKLAS_HOME / "vectors" / "default-archive"
DEFAULT_ROOT = SAKLAS_HOME / "vectors" / "default"


def _safe_model_id(model_id: str) -> str:
    """Convert ``owner/name`` to the on-disk slug ``owner__name``."""
    return model_id.replace("/", "__")


def _emit_row(name: str, cos: float, n_layers: int, width: int) -> None:
    bar_full = max(0.0, min(1.0, cos))
    bar = "■" * int(bar_full * 30)
    print(f"  {name:<{width}}  {cos:+.4f}  ({n_layers:>2} layers)  {bar}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="google/gemma-4-31b-it",
                    help="HF model id (default: %(default)s)")
    ap.add_argument(
        "--whitener", action="store_true",
        help="use Mahalanobis cosine (loads cached LayerWhitener for the model)",
    )
    args = ap.parse_args()

    safe = _safe_model_id(args.model)
    print(f"Comparing archive vs default for model={args.model}")
    print(f"  archive: {ARCHIVE_ROOT}")
    print(f"  default: {DEFAULT_ROOT}")
    print()

    if not ARCHIVE_ROOT.exists():
        print(f"error: archive not found at {ARCHIVE_ROOT}", file=sys.stderr)
        print("hint: cp -r ~/.saklas/vectors/default ~/.saklas/vectors/default-archive",
              file=sys.stderr)
        return 2

    whitener = None
    if args.whitener:
        from saklas.core.mahalanobis import LayerWhitener
        try:
            whitener = LayerWhitener.from_cache(args.model)
            print(f"  using Mahalanobis whitener "
                  f"(coverage: {len(whitener.layers)} layers)")
        except Exception as e:
            print(f"warning: whitener load failed ({e}); falling back to Euclidean",
                  file=sys.stderr)

    results: list[tuple[str, float, int]] = []
    missing_archive: list[str] = []
    missing_default: list[str] = []

    for archive_dir in sorted(ARCHIVE_ROOT.iterdir()):
        if not archive_dir.is_dir():
            continue
        name = archive_dir.name
        archive_tensor = archive_dir / f"{safe}.safetensors"
        default_tensor = DEFAULT_ROOT / name / f"{safe}.safetensors"
        if not archive_tensor.exists():
            missing_archive.append(name)
            continue
        if not default_tensor.exists():
            missing_default.append(name)
            continue
        old = Profile.load(archive_tensor)
        new = Profile.load(default_tensor)
        shared = sorted(set(old.layers) & set(new.layers))
        if not shared:
            print(f"  [warn] {name}: no shared layers between old and new", file=sys.stderr)
            continue
        cos = old.cosine_similarity(new, whitener=whitener)
        results.append((name, float(cos), len(shared)))

    if not results:
        print("no concepts to compare. did you re-extract yet?", file=sys.stderr)
        return 1

    results.sort(key=lambda r: r[1])
    width = max(len(name) for name, _, _ in results)

    print()
    metric_label = "Mahalanobis cosine" if whitener is not None else "magnitude-weighted cosine"
    print(f"Per-concept {metric_label} (sorted, lowest first):")
    print()
    for name, cos, n_layers in results:
        _emit_row(name, cos, n_layers, width)

    print()
    mean = sum(c for _, c, _ in results) / len(results)
    n_unchanged = sum(1 for _, c, _ in results if c >= 0.95)
    n_shifted = sum(1 for _, c, _ in results if c < 0.80)
    print(
        f"Summary: {len(results)} concepts, mean cos={mean:+.4f}, "
        f"{n_unchanged} unchanged (≥0.95), {n_shifted} shifted (<0.80)"
    )

    if missing_archive:
        print(f"\n[skipped — no archive tensor for model] {', '.join(missing_archive)}",
              file=sys.stderr)
    if missing_default:
        print(f"\n[skipped — not yet re-extracted] {', '.join(missing_default)}",
              file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
