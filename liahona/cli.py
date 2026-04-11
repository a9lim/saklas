"""CLI entry point for liahona."""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("PYTHONWARNINGS", "ignore::UserWarning:multiprocessing.resource_tracker")


def _add_common_args(p: argparse.ArgumentParser) -> None:
    """Add arguments shared between TUI and serve subcommands."""
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
        help="Probe categories: all, none, emotion, personality, safety, cultural, gender (default: all)",
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


def _resolve_probes(raw: list[str] | None) -> list[str]:
    all_categories = ["emotion", "personality", "safety", "cultural", "gender"]
    if raw is None or raw == ["all"]:
        return all_categories
    if raw == ["none"] or raw == []:
        return []
    return raw


def _make_session(args: argparse.Namespace):
    from liahona.session import LiahonaSession
    probe_categories = _resolve_probes(args.probes)
    return LiahonaSession(
        model_id=args.model, device=args.device, quantize=args.quantize,
        probes=probe_categories, system_prompt=args.system_prompt,
        max_tokens=args.max_tokens, cache_dir=args.cache_dir,
    )


def _print_model_info(session) -> None:
    info = session.model_info
    print(f"Architecture: {info['model_type']}")
    print(f"Layers: {info['num_layers']}, Hidden dim: {info['hidden_dim']}")
    print(f"VRAM: {info['vram_used_gb']:.1f} GB")
    print(f"Loaded {len(session.probes)} probes")


_SUBCOMMANDS = {"serve"}


def _build_serve_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="steer serve", description="Start OpenAI-compatible API server")
    _add_common_args(p)
    p.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    p.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    p.add_argument(
        "--steer", action="append", default=[], metavar="NAME[:ALPHA]",
        help="Pre-load a steering vector (repeatable). e.g. --steer cheerful:0.2",
    )
    p.add_argument(
        "--cors", action="append", default=[], metavar="ORIGIN",
        help="CORS allowed origin (repeatable). Omit for no CORS.",
    )
    return p


def _build_tui_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="steer",
        description="Activation steering + trait monitoring for local HuggingFace models",
    )
    _add_common_args(p)
    return p


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    if argv is None:
        argv = sys.argv[1:]

    # Dispatch on first arg to avoid argparse subparser validation issues
    if argv and argv[0] in _SUBCOMMANDS:
        cmd = argv[0]
        if cmd == "serve":
            args = _build_serve_parser().parse_args(argv[1:])
            args.command = "serve"
        return args

    # Default: TUI mode
    args = _build_tui_parser().parse_args(argv)
    args.command = "tui"
    return args


def _run_tui(args: argparse.Namespace) -> None:
    print(f"Loading model: {args.model}")
    if args.quantize:
        print(f"Quantization: {args.quantize}")

    session = _make_session(args)
    _print_model_info(session)

    from liahona.tui.app import LiahonaApp
    app = LiahonaApp(session=session)
    app.run()


def _parse_steer_flag(raw: str) -> tuple[str, float]:
    """Parse 'name:alpha' or 'name' into (name, alpha)."""
    if ":" in raw:
        name, alpha_s = raw.rsplit(":", 1)
        return name, float(alpha_s)
    return raw, 0.0


def _run_serve(args: argparse.Namespace) -> None:
    try:
        import fastapi  # noqa: F401
        import uvicorn
    except ImportError:
        print(
            "Server dependencies not installed. Run:\n"
            "  pip install liahona[serve]",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loading model: {args.model}")
    if args.quantize:
        print(f"Quantization: {args.quantize}")

    session = _make_session(args)
    _print_model_info(session)

    # Pre-load steering vectors
    default_alphas: dict[str, float] = {}
    for spec in args.steer:
        name, alpha = _parse_steer_flag(spec)
        print(f"Extracting steering vector: {name}")
        profile = session.extract(name, on_progress=lambda m: print(f"  {m}"))
        session.steer(name, profile)
        default_alphas[name] = alpha
        print(f"  Registered '{name}' (default alpha={alpha})")

    from liahona.server import create_app
    app = create_app(session, default_alphas=default_alphas, cors_origins=args.cors or None)

    print(f"\nServing on http://{args.host}:{args.port}")
    print(f"API docs: http://{args.host}:{args.port}/docs")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    if args.command == "serve":
        _run_serve(args)
    else:
        _run_tui(args)
