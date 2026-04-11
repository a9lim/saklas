"""CLI entry point for steer."""

from __future__ import annotations

import argparse
import warnings

warnings.filterwarnings("ignore", message=".*leaked semaphore.*")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="steer",
        description="Activation steering + trait monitoring TUI for local HuggingFace models",
    )
    p.add_argument(
        "model",
        help="HuggingFace model ID or local path (e.g. google/gemma-2-9b-it)",
    )
    p.add_argument(
        "--quantize", "-q",
        choices=["4bit", "8bit"],
        default=None,
        help="Quantization mode (default: bf16/fp16)",
    )
    p.add_argument(
        "--device", "-d",
        default="auto",
        help="Device: auto (detect), cuda, mps, or cpu (default: auto)",
    )
    p.add_argument(
        "--probes", "-p",
        nargs="*",
        default=None,
        help="Probe categories to load: all, none, emotion, personality, safety, cultural, gender (default: all)",
    )
    p.add_argument(
        "--system-prompt", "-s",
        default=None,
        help="System prompt for chat",
    )
    p.add_argument(
        "--max-tokens", "-m",
        type=int,
        default=1024,
        help="Max tokens per generation (default: 1024)",
    )
    p.add_argument(
        "--cache-dir", "-c",
        default=None,
        help="Cache directory for extracted vectors (default: probes/cache/ in package)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    print(f"Loading model: {args.model}")
    if args.quantize:
        print(f"Quantization: {args.quantize}")

    all_categories = ["emotion", "personality", "safety", "cultural", "gender"]
    if args.probes is None or args.probes == ["all"]:
        probe_categories = all_categories
    elif args.probes == ["none"] or args.probes == []:
        probe_categories = []
    else:
        probe_categories = args.probes

    from steer.session import SteerSession
    session = SteerSession(
        model_id=args.model, device=args.device, quantize=args.quantize,
        probes=probe_categories, system_prompt=args.system_prompt,
        max_tokens=args.max_tokens, cache_dir=args.cache_dir,
    )

    info = session.model_info
    print(f"Architecture: {info['model_type']}")
    print(f"Layers: {info['num_layers']}, Hidden dim: {info['hidden_dim']}")
    print(f"VRAM: {info['vram_used_gb']:.1f} GB")
    print(f"Loaded {len(session.probes)} probes")

    from steer.tui.app import SteerApp
    app = SteerApp(session=session)
    app.run()
