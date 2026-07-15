#!/usr/bin/env python3
"""Reproducible wall-time/RSS/work counters for Saklas fitting paths.

Examples:

  python scripts/benchmark_fitting.py jlens google/gemma-3-4b-it \
    --corpus prompts.txt --prompts 4 --layers workspace --seq-len 64
  python scripts/benchmark_fitting.py manifold google/gemma-3-4b-it \
    ~/.saklas/manifolds/local/personas --layers workspace

Each run uses a fresh temporary SAKLAS_HOME unless ``--home`` is supplied,
prints newline-delimited JSON, and measures an immediate cache/no-op repeat.
The model-forward counter is structural evidence alongside wall time; peak RSS
is the process high-water mark reported by the OS (including model weights).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

try:  # POSIX only; Windows still runs the benchmark without the RSS field.
    import resource
except ImportError:  # pragma: no cover - exercised on Windows
    resource = None  # type: ignore[assignment]


def _peak_rss_bytes() -> int | None:
    if resource is None:
        return None
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _measure(
    label: str, operation: Callable[[], Any], forwards: list[int], *,
    report_process_peak: bool = False,
) -> tuple[Any, dict[str, Any]]:
    before_calls = forwards[0]
    started = time.perf_counter()
    result = operation()
    elapsed = time.perf_counter() - started
    row = {
        "phase": label,
        "seconds": round(elapsed, 6),
        "model_forwards": forwards[0] - before_calls,
    }
    # ru_maxrss is a process-lifetime high-water mark. Report it once for the
    # force-fit run; repeating it on later phases would falsely imply a
    # phase-local cache/no-op measurement.
    if report_process_peak:
        peak = _peak_rss_bytes()
        if peak is not None:
            row["process_peak_rss_bytes"] = peak
    print(json.dumps(row, sort_keys=True), flush=True)
    return result, row


def _install_forward_counter(model: Any) -> tuple[list[int], Any]:
    calls = [0]

    def _count(_module: Any, _args: Any) -> None:
        calls[0] += 1

    return calls, model.register_forward_pre_hook(_count)


def _load_prompts(path: Path, limit: int) -> list[str]:
    prompts = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(prompts) < limit:
        raise ValueError(f"{path} has {len(prompts)} non-empty lines; need {limit}")
    return prompts[:limit]


@contextmanager
def _staged_authoring_folder(source: Path):
    """Copy authoring inputs so ``force=True`` never mutates *source*."""
    source = source.expanduser().resolve()
    if not (source / "manifold.json").is_file():
        raise ValueError(f"{source} is not a manifold folder")
    with tempfile.TemporaryDirectory(prefix="saklas-manifold-benchmark-") as root:
        staged = Path(root) / source.name
        shutil.copytree(source, staged, ignore=shutil.ignore_patterns(".locks"))
        for path in staged.glob("*.safetensors"):
            path.unlink()
        for path in staged.glob("*.gguf"):
            path.unlink()
        for path in staged.glob("*.json"):
            if path.name not in {"manifold.json", "scenarios.json"}:
                path.unlink()
        manifest = json.loads((staged / "manifold.json").read_text())
        manifest["files"] = {}
        (staged / "manifold.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        yield staged


def _run_jlens(args: argparse.Namespace) -> None:
    from saklas.core.session import SaklasSession

    prompts = _load_prompts(args.corpus, args.prompts)
    with SaklasSession.from_pretrained(
        args.model, device=args.device, quantize=args.quantize, probes=[],
    ) as session:
        counters, handle = _install_forward_counter(session._model)
        try:
            fit_kwargs = {
                "corpus_spec": f"file:{args.corpus.name}",
                "source_layers": args.layers,
                "seq_len": args.seq_len,
                "dim_batch": args.dim_batch,
                "prompt_batch": args.prompt_batch,
            }
            _measure(
                "jlens_force_fit",
                lambda: session.fit_jlens(prompts, force=True, **fit_kwargs),
                counters, report_process_peak=True,
            )
            _measure(
                "jlens_exact_cache_repeat",
                lambda: session.fit_jlens(prompts, force=False, **fit_kwargs),
                counters,
            )
        finally:
            handle.remove()


def _run_manifold(args: argparse.Namespace) -> None:
    from saklas.core.session import SaklasSession

    with _staged_authoring_folder(args.folder) as staged:
        with SaklasSession.from_pretrained(
            args.model, device=args.device, quantize=args.quantize, probes=[],
        ) as session:
            counters, handle = _install_forward_counter(session._model)
            try:
                fit_kwargs = {
                    "layers": args.layers,
                    "sae": args.sae,
                }
                _measure(
                    "manifold_force_fit",
                    lambda: session.fit(staged, force=True, **fit_kwargs),
                    counters, report_process_peak=True,
                )
                _measure(
                    "manifold_exact_cache_repeat",
                    lambda: session.fit(staged, force=False, **fit_kwargs),
                    counters,
                )
            finally:
                handle.remove()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, help="Persistent SAKLAS_HOME")
    sub = parser.add_subparsers(dest="command", required=True)

    def model_args(child: argparse.ArgumentParser) -> None:
        child.add_argument("model")
        child.add_argument("--device", default="auto")
        child.add_argument("--quantize", choices=["4bit", "8bit"])
        child.add_argument("--layers", default="workspace")

    jlens = sub.add_parser("jlens")
    model_args(jlens)
    jlens.add_argument("--corpus", type=Path, required=True)
    jlens.add_argument("--prompts", type=int, default=4)
    jlens.add_argument("--seq-len", type=int, default=64)
    jlens.add_argument("--dim-batch", type=int, default=8)
    jlens.add_argument("--prompt-batch", type=int)
    jlens.set_defaults(run=_run_jlens)

    manifold = sub.add_parser("manifold")
    model_args(manifold)
    manifold.add_argument("folder", type=Path)
    manifold.add_argument("--sae")
    manifold.set_defaults(run=_run_manifold)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.home is not None:
        args.home.mkdir(parents=True, exist_ok=True)
        os.environ["SAKLAS_HOME"] = str(args.home)
        args.run(args)
        return
    with tempfile.TemporaryDirectory(prefix="saklas-fit-benchmark-") as home:
        os.environ["SAKLAS_HOME"] = home
        args.run(args)


if __name__ == "__main__":
    main()
