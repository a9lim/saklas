"""CLI entry point for saklas."""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("PYTHONWARNINGS", "ignore::UserWarning:multiprocessing.resource_tracker")


_SUBCOMMANDS = {"serve", "install", "refresh", "clear", "uninstall", "list", "merge"}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _add_common_args(p: argparse.ArgumentParser) -> None:
    """Model-loading args shared between the bare TUI and `serve`."""
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
        probes=probe_categories,
        system_prompt=getattr(args, "system_prompt", None),
        max_tokens=getattr(args, "max_tokens", 1024),
    )


def _print_model_info(session) -> None:
    info = session.model_info
    print(f"Architecture: {info['model_type']}")
    print(f"Layers: {info['num_layers']}, Hidden dim: {info['hidden_dim']}")
    print(f"VRAM: {info['vram_used_gb']:.1f} GB")
    print(f"Loaded {len(session.probes)} probes")


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def _build_tui_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="saklas",
        description="Activation steering + trait monitoring for local HuggingFace models",
    )
    _add_common_args(p)
    # `model` is required by _add_common_args but TUI allows it to be omitted
    # when combined with -C. Relax here.
    for action in p._actions:
        if action.dest == "model":
            action.nargs = "?"
            action.default = None
            break

    p.add_argument("--config", "-c", action="append", default=None, metavar="PATH",
                   help="Load setup YAML (repeatable; later overrides earlier)")
    p.add_argument("--strict", "-s", action="store_true",
                   help="With -c: fail hard on missing vectors")
    return p


def _build_serve_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="saklas serve", description="Start OpenAI-compatible API server")
    _add_common_args(p)
    p.add_argument("--host", "-H", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    p.add_argument("--port", "-P", type=int, default=8000, help="Bind port (default: 8000)")
    p.add_argument(
        "--steer", "-S", action="append", default=[], metavar="NAME[:ALPHA]",
        help="Pre-load a steering vector (repeatable). e.g. --steer cheerful:0.2",
    )
    p.add_argument(
        "--cors", "-C", action="append", default=[], metavar="ORIGIN",
        help="CORS allowed origin (repeatable). Omit for no CORS.",
    )
    p.add_argument(
        "--api-key", "-k", default=None, metavar="KEY",
        help="Require Bearer token auth. Falls back to $SAKLAS_API_KEY. Unset = no auth.",
    )
    return p


def _build_install_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="saklas install", description="Install a concept pack from HF or a local folder")
    p.add_argument("target", help="<ns>/<concept>[@revision] or path to a concept folder")
    p.add_argument("--statements-only", "-s", action="store_true",
                   help="Keep statements.json only; drop any bundled tensors")
    p.add_argument("--as", "-a", dest="as_target", default=None, metavar="NS/NAME",
                   help="Relocate the installed pack under a different namespace/name")
    p.add_argument("--force", "-f", action="store_true",
                   help="Overwrite an existing installation")
    return p


def _build_refresh_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="saklas refresh",
                                description="Re-pull concept(s) from their source (or `neutrals` to refresh neutral_statements.json)")
    p.add_argument("selector", help="Selector (name, tag:x, namespace:x, default, all) or the literal 'neutrals'")
    p.add_argument("--model", "-m", default=None, metavar="MODEL_ID",
                   help="Scope the refresh to one model's tensors (delete its safetensors + sidecar)")
    return p


def _build_clear_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="saklas clear",
                                description="Delete per-model tensors for matched concepts (keeps statements.json)")
    p.add_argument("selector", help="Selector (name, tag:x, namespace:x, default, all)")
    p.add_argument("--model", "-m", default=None, metavar="MODEL_ID",
                   help="Scope to one model's tensors only (default: all models)")
    p.add_argument("--yes", "-y", action="store_true",
                   help="Skip confirmation prompt on broad selectors")
    return p


def _build_uninstall_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="saklas uninstall",
                                description="Fully remove a concept folder (tensors + statements + pack.json)")
    p.add_argument("selector", help="Selector (name, tag:x, namespace:x, default, all)")
    p.add_argument("--yes", "-y", action="store_true",
                   help="Required for broad selectors (all, namespace:)")
    return p


def _build_list_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="saklas list",
                                description="List installed concepts (and HF concepts by default)")
    p.add_argument("selector", nargs="?", default=None,
                   help="Optional selector (name, tag:x, namespace:x, default, all)")
    p.add_argument("--installed", "-i", action="store_true",
                   help="Show only locally installed concepts (skip HF query)")
    p.add_argument("--json", "-j", dest="json_output", action="store_true",
                   help="Emit machine-readable JSON instead of a table")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Include descriptions in the table output")
    return p


def _build_merge_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="saklas merge",
                                description="Merge existing vectors into a new pack")
    p.add_argument("name", help="New pack name (written under local/)")
    p.add_argument("components", help="Comma-separated components: ns/a:0.3,ns/b:0.4")
    p.add_argument("--force", "-f", action="store_true",
                   help="Overwrite an existing merged pack")
    p.add_argument("--strict", "-s", action="store_true",
                   help="Fail if any component is missing")
    p.add_argument("--model", "-m", default=None, metavar="MODEL_ID",
                   help="Restrict the merge to tensors for one model")
    return p


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    if argv is None:
        argv = sys.argv[1:]

    if argv and argv[0] in _SUBCOMMANDS:
        cmd = argv[0]
        rest = argv[1:]
        if cmd == "serve":
            args = _build_serve_parser().parse_args(rest)
        elif cmd == "install":
            args = _build_install_parser().parse_args(rest)
        elif cmd == "refresh":
            args = _build_refresh_parser().parse_args(rest)
        elif cmd == "clear":
            args = _build_clear_parser().parse_args(rest)
        elif cmd == "uninstall":
            args = _build_uninstall_parser().parse_args(rest)
        elif cmd == "list":
            args = _build_list_parser().parse_args(rest)
        elif cmd == "merge":
            args = _build_merge_parser().parse_args(rest)
        else:  # pragma: no cover
            raise RuntimeError(f"unhandled subcommand {cmd}")
        args.command = cmd
        return args

    # Bare form: TUI.
    args = _build_tui_parser().parse_args(argv)
    args.command = "tui"

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
            max_tokens=None,
            system_prompt=None,
        )
        args.model = composed.model or args.model
        args.temperature = composed.temperature
        args.top_p = composed.top_p
        args.orthogonalize = composed.orthogonalize
        args.thinking = composed.thinking
        args.system_prompt = composed.system_prompt
        args.max_tokens = composed.max_tokens if composed.max_tokens is not None else 1024
        args.config_vectors = composed.vectors
        ensure_vectors_installed(composed, strict=args.strict)
    else:
        args.config_vectors = {}
        args.temperature = None
        args.top_p = None
        args.orthogonalize = None
        args.thinking = None

    if args.model is None:
        _build_tui_parser().error("the following arguments are required: model")
    return args


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def _print_startup(args: argparse.Namespace) -> None:
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


def _run_install(args: argparse.Namespace) -> None:
    from saklas import cache_ops
    cache_ops.install(
        args.target,
        as_=args.as_target,
        force=args.force,
        statements_only=args.statements_only,
    )
    suffix = " (statements only)" if args.statements_only else ""
    print(f"Installed {args.target}{suffix}")


def _run_refresh(args: argparse.Namespace) -> None:
    from saklas import cache_ops
    from saklas.cli_selectors import parse as sel_parse

    if args.selector == "neutrals":
        if args.model is not None:
            print("warning: --model has no effect with `refresh neutrals`", file=sys.stderr)
        dst = cache_ops.refresh_neutrals()
        print(f"Refreshed {dst}")
        return

    selector = sel_parse(args.selector)
    n = cache_ops.refresh(selector, model_scope=args.model)
    print(f"Refreshed {n} concept(s)")


def _run_clear(args: argparse.Namespace) -> None:
    from saklas import cache_ops
    from saklas.cli_selectors import parse as sel_parse

    selector = sel_parse(args.selector)
    if selector.kind in {"all", "namespace"} and not args.yes:
        print(
            f"refusing to clear a broad selector ({selector.kind}); pass --yes to confirm",
            file=sys.stderr,
        )
        sys.exit(2)
    n = cache_ops.delete_tensors(selector, args.model)
    print(f"Deleted {n} files")


def _run_uninstall(args: argparse.Namespace) -> None:
    from saklas import cache_ops
    from saklas.cli_selectors import parse as sel_parse

    selector = sel_parse(args.selector)
    try:
        n = cache_ops.uninstall(selector, yes=args.yes)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        sys.exit(2)
    print(f"Uninstalled {n} concept(s)")


def _run_list(args: argparse.Namespace) -> None:
    from saklas import cache_ops
    from saklas.cli_selectors import parse as sel_parse

    selector = sel_parse(args.selector) if args.selector else None
    cache_ops.list_concepts(
        selector,
        hf=not args.installed,
        installed_only=args.installed,
        json_output=args.json_output,
        verbose=args.verbose,
    )


def _run_merge(args: argparse.Namespace) -> None:
    from saklas import merge as merge_mod
    components = merge_mod.parse_components(args.components)
    dst = merge_mod.merge_into_pack(
        args.name, components, model=args.model,
        force=args.force, strict=args.strict,
    )
    print(f"Merged pack written to {dst}")


_RUNNERS = {
    "tui": _run_tui,
    "serve": _run_serve,
    "install": _run_install,
    "refresh": _run_refresh,
    "clear": _run_clear,
    "uninstall": _run_uninstall,
    "list": _run_list,
    "merge": _run_merge,
}


def main(argv: list[str] | None = None):
    from saklas.packs import print_migration_notice_if_needed
    print_migration_notice_if_needed()

    args = parse_args(argv)
    runner = _RUNNERS[args.command]
    runner(args)
