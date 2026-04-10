"""CLI entry point for steer."""

from __future__ import annotations

import argparse


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
        "--device",
        default="auto",
        help="Device: auto (detect), cuda, mps, or cpu (default: auto)",
    )
    p.add_argument(
        "--no-compile",
        action="store_true",
        help="Skip torch.compile (use if hooks cause issues)",
    )
    p.add_argument(
        "--probes",
        nargs="*",
        default=None,
        help="Probe categories to load: emotion personality safety cultural gender (default: all)",
    )
    p.add_argument(
        "--system-prompt", "-s",
        default=None,
        help="System prompt for chat",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Max tokens per generation (default: 512)",
    )
    p.add_argument(
        "--cache-dir",
        default=None,
        help="Cache directory for extracted vectors (default: probes/cache/ in package)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None):
    args = parse_args(argv)

    # Lazy imports — don't load torch until we need it
    print(f"Loading model: {args.model}")
    if args.quantize:
        print(f"  Quantization: {args.quantize}")

    from steer.model import load_model, get_layers, get_model_info
    model, tokenizer = load_model(
        args.model,
        quantize=args.quantize,
        device=args.device,
        no_compile=args.no_compile,
    )
    info = get_model_info(model, tokenizer)
    layers = get_layers(model)

    print(f"  Architecture: {info['model_type']}")
    print(f"  Layers: {info['num_layers']}, Hidden dim: {info['hidden_dim']}")
    print(f"  VRAM: {info['vram_used_gb']:.1f} GB")

    # Bootstrap probes
    probe_categories = args.probes if args.probes else ["emotion", "personality", "safety", "cultural", "gender"]
    print(f"  Bootstrapping probes: {', '.join(probe_categories)}")

    from steer.probes_bootstrap import bootstrap_probes
    import pathlib

    cache_dir = args.cache_dir or str(
        pathlib.Path(__file__).parent / "probes" / "cache"
    )
    probes = bootstrap_probes(
        model, tokenizer, layers, info,
        categories=probe_categories,
        cache_dir=cache_dir,
    )

    if probes:
        print(f"  Loaded {len(probes)} probes")

    # Launch TUI
    from steer.tui.app import SteerApp
    app = SteerApp(
        model=model,
        tokenizer=tokenizer,
        layers=layers,
        model_info=info,
        probes=probes,
        system_prompt=args.system_prompt,
        max_tokens=args.max_tokens,
    )
    app.run()
