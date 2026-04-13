"""CLI entry point for saklas."""

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
        "--max-tokens",
        type=int,
        default=1024,
        help="Max tokens per generation (default: 1024)",
    )



def _resolve_probes(raw: list[str] | None) -> list[str]:
    from saklas.session import PROBE_CATEGORIES
    if raw is None or raw == ["all"]:
        return list(PROBE_CATEGORIES)
    if raw == ["none"] or raw == []:
        return []
    return raw


def _make_session(args: argparse.Namespace):
    from saklas.session import SaklasSession
    probe_categories = _resolve_probes(args.probes)
    return SaklasSession(
        model_id=args.model, device=args.device, quantize=args.quantize,
        probes=probe_categories, system_prompt=args.system_prompt,
        max_tokens=args.max_tokens,
    )


def _print_model_info(session) -> None:
    info = session.model_info
    print(f"Architecture: {info['model_type']}")
    print(f"Layers: {info['num_layers']}, Hidden dim: {info['hidden_dim']}")
    print(f"VRAM: {info['vram_used_gb']:.1f} GB")
    print(f"Loaded {len(session.probes)} probes")


_SUBCOMMANDS = {"serve"}


def _build_serve_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="saklas serve", description="Start OpenAI-compatible API server")
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
    p.add_argument(
        "--api-key", default=None, metavar="KEY",
        help="Require Bearer token auth. Falls back to $SAKLAS_API_KEY. Unset = no auth.",
    )
    return p


def _build_tui_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="saklas",
        description="Activation steering + trait monitoring for local HuggingFace models",
    )
    _add_common_args(p)
    # `model` is required by _add_common_args but TUI allows it to be omitted
    # (e.g. `saklas -l`, `saklas -r default`). Relax it here.
    for action in p._actions:
        if action.dest == "model":
            action.nargs = "?"
            action.default = None
            break

    # Cache ops (composable; fall through to TUI if a model follows).
    # Each flag takes exactly one selector token; repeat for compound selectors
    # (e.g. -r tag:emotion -r model:gemma-2-2b-it combines concept + model scope).
    p.add_argument("--refresh", "-r", action="append", default=None,
                   metavar="SELECTOR",
                   help="Re-pull concept(s) from source (repeatable)")
    p.add_argument("--refresh-neutrals", "-n", action="store_true",
                   help="Overwrite ~/.saklas/neutral_statements.json with the "
                        "bundled copy (forces layer-means recompute on next run)")
    p.add_argument("--clear-tensors", "-x", action="append", default=None,
                   metavar="SELECTOR",
                   help="Delete tensors for matched concepts (repeatable; keeps statements.json)")
    p.add_argument("--install", "-i", action="append", default=None, metavar="TARGET",
                   help="Install a pack from HF coord or local folder path (repeatable)")
    p.add_argument("--merge", "-m", nargs=2, default=None,
                   metavar=("NAME", "COMPONENTS"),
                   help="Merge vectors: -m <name> ns/a:0.3,ns/b:0.4")
    p.add_argument("--as", dest="as_target", default=None, metavar="NS/NAME",
                   help="With -i or -m: relocate the installed/merged pack to a different path")
    p.add_argument("--force", action="store_true",
                   help="With -i, -m, or -r: overwrite an existing target")

    # List/info (exit-only). nargs=? so `-l` alone is distinguishable from `-l foo`
    # and a trailing positional is treated as the model.
    p.add_argument("--list", "-l", nargs="?", default=None, const="",
                   metavar="SELECTOR",
                   help="List or show info about packs; exits after printing")

    # Config file (composable).
    p.add_argument("--config", "-C", action="append", default=None, metavar="PATH",
                   help="Load setup YAML (repeatable; later overrides earlier)")
    p.add_argument("--strict", action="store_true",
                   help="With -C: fail hard on missing vectors")

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

    args = _build_tui_parser().parse_args(argv)

    # Each -r/-x/-i call appends one token. Already a flat list.
    args.delete = args.clear_tensors
    if args.merge is not None:
        args.merge_name, args.merge_components = args.merge
    else:
        args.merge_name = None
        args.merge_components = None

    has_cache_op = bool(
        args.refresh or args.refresh_neutrals or args.delete
        or args.install or args.merge_name
    )
    has_list = args.list is not None

    if has_list:
        if args.model is not None:
            _build_tui_parser().error("-l does not accept a model positional")
        args.command = "list"
        args.config_vectors = {}
        return args

    if args.config:
        from pathlib import Path as _P
        from saklas.config_file import (
            ConfigFile, compose, apply_flag_overrides, ensure_vectors_installed,
        )
        loaded = [ConfigFile.load(_P(p)) for p in args.config]
        composed = compose(loaded)
        composed = apply_flag_overrides(
            composed,
            model=args.model,
            temperature=None,
            top_p=None,
            max_tokens=args.max_tokens if args.max_tokens != 1024 else None,
            system_prompt=args.system_prompt,
        )
        args.model = composed.model or args.model
        args.temperature = composed.temperature
        args.top_p = composed.top_p
        args.orthogonalize = composed.orthogonalize
        args.thinking = composed.thinking
        args.system_prompt = composed.system_prompt or args.system_prompt
        if composed.max_tokens is not None:
            args.max_tokens = composed.max_tokens
        args.config_vectors = composed.vectors
        ensure_vectors_installed(composed, strict=args.strict)
    else:
        args.config_vectors = {}
        args.temperature = None
        args.top_p = None
        args.orthogonalize = None
        args.thinking = None

    if has_cache_op and args.model is None:
        args.command = "cache"
        return args

    if args.model is None:
        _build_tui_parser().error("the following arguments are required: model")

    args.command = "tui"
    return args


def _print_startup(args: argparse.Namespace) -> None:
    """Print common model-loading banner."""
    print(f"Loading model: {args.model}")
    if args.quantize:
        print(f"Quantization: {args.quantize}")


def _run_tui(args: argparse.Namespace) -> None:
    _print_startup(args)

    session = _make_session(args)
    _print_model_info(session)

    for coord, alpha in getattr(args, "config_vectors", {}).items():
        name = coord.split("/", 1)[-1] if "/" in coord else coord
        try:
            profile = session.extract(name)
            session.steer(coord, profile)
            print(f"  Registered '{coord}' (alpha={alpha})")
        except Exception as e:
            print(f"  Failed to register '{coord}': {e}")

    from saklas.tui.app import SaklasApp
    app = SaklasApp(session=session)
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
            "  pip install saklas[serve]",
            file=sys.stderr,
        )
        sys.exit(1)

    _print_startup(args)

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

    from saklas.server import create_app
    app = create_app(session, default_alphas=default_alphas,
                     cors_origins=args.cors or None,
                     api_key=getattr(args, "api_key", None))

    print(f"\nServing on http://{args.host}:{args.port}")
    print(f"API docs: http://{args.host}:{args.port}/docs")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


def _run_cache(args: argparse.Namespace) -> None:
    from saklas import cache_ops, merge
    from saklas.cli_selectors import parse_args as sel_parse_args

    as_target = getattr(args, "as_target", None)
    force = getattr(args, "force", False)

    if args.refresh:
        concept_sel, model_scope = sel_parse_args(args.refresh)
        n = cache_ops.refresh(concept_sel, model_scope=model_scope)
        print(f"Refreshed {n} concept(s)")

    if args.refresh_neutrals:
        dst = cache_ops.refresh_neutrals()
        print(f"Refreshed {dst}")

    if args.delete:
        concept_sel, model_scope = sel_parse_args(args.delete)
        n = cache_ops.delete_tensors(concept_sel, model_scope)
        print(f"Deleted {n} files")

    if args.install:
        for target in args.install:
            cache_ops.install(target, as_=as_target, force=force)
            print(f"Installed {target}")

    if args.merge_name is not None:
        components = merge.parse_components(args.merge_components)
        dst = merge.merge_into_pack(
            args.merge_name, components, model=None,
            force=force, strict=getattr(args, "strict", False),
        )
        print(f"Merged pack written to {dst}")


def _run_list(args: argparse.Namespace) -> None:
    from saklas import cache_ops
    from saklas.cli_selectors import parse as sel_parse

    raw = args.list
    if not raw:  # empty string or None
        cache_ops.list_concepts(selector=None, hf=True)
        return
    selector = sel_parse(raw)
    cache_ops.list_concepts(selector=selector, hf=True)


def main(argv: list[str] | None = None):
    from saklas.packs import print_migration_notice_if_needed
    print_migration_notice_if_needed()

    args = parse_args(argv)
    if args.command == "serve":
        _run_serve(args)
    elif args.command == "cache":
        _run_cache(args)
    elif args.command == "list":
        _run_list(args)
    else:
        if (args.refresh or args.refresh_neutrals or args.delete
                or args.install or args.merge_name):
            _run_cache(args)
        _run_tui(args)
