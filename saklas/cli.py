"""CLI entry point for saklas."""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("PYTHONWARNINGS", "ignore::UserWarning:multiprocessing.resource_tracker")


_SUBCOMMANDS = {"serve", "install", "refresh", "clear", "uninstall", "list", "merge", "push"}


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
        "-q", "--quantize",
        choices=["4bit", "8bit"],
        default=None,
        help="Quantization mode (default: bf16/fp16)",
    )
    p.add_argument(
        "-d", "--device",
        default="auto",
        help="Device: auto (detect), cuda, mps, or cpu (default: auto)",
    )
    p.add_argument(
        "-p", "--probes",
        nargs="*",
        default=None,
        help="Probe categories: all, none, affect, epistemic, alignment, register, social_stance, cultural (default: all)",
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

_SUBCOMMAND_DESCRIPTIONS: list[tuple[str, str]] = [
    ("serve",     "Start the OpenAI/Ollama-compatible API server"),
    ("install",   "Install a concept pack from HF or a local folder"),
    ("refresh",   "Re-pull concept(s) from their source"),
    ("clear",     "Delete per-model tensors for matched concepts"),
    ("uninstall", "Fully remove a concept folder"),
    ("list",      "List installed concepts (and HF concepts by default)"),
    ("merge",     "Merge existing vectors into a new pack"),
    ("push",      "Push a concept pack to HF as a model repo"),
]


def _tui_epilog() -> str:
    width = max(len(name) for name, _ in _SUBCOMMAND_DESCRIPTIONS)
    lines = ["subcommands:"]
    for name, desc in _SUBCOMMAND_DESCRIPTIONS:
        lines.append(f"  {name:<{width}}  {desc}")
    lines.append("")
    lines.append("Run `saklas <subcommand> -h` for subcommand options.")
    lines.append("With no subcommand, launches the TUI for the given model.")
    return "\n".join(lines)


def _build_tui_parser(model_optional: bool = False) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="saklas",
        description="Activation steering + trait monitoring for local HuggingFace models",
        epilog=_tui_epilog(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_common_args(p)
    if model_optional:
        # Allow `saklas -c foo.yaml` to supply the model via YAML instead of
        # a positional argument. Usage will show `[model]` in this case, but
        # the normal help (the one users actually see) keeps it unbracketed.
        for action in p._actions:
            if action.dest == "model":
                action.nargs = "?"
                action.default = None
                break

    p.add_argument("-c", "--config", action="append", default=None, metavar="PATH",
                   help="Load setup YAML (repeatable; later overrides earlier)")
    p.add_argument("-s", "--strict", action="store_true",
                   help="With -c: fail hard on missing vectors")
    return p


def _build_serve_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="saklas serve", description="Start OpenAI-compatible API server")
    _add_common_args(p)
    p.add_argument("-H", "--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    p.add_argument("-P", "--port", type=int, default=8000, help="Bind port (default: 8000)")
    p.add_argument(
        "-S", "--steer", action="append", default=[], metavar="NAME[:ALPHA]",
        help="Pre-load a steering vector (repeatable). e.g. --steer cheerful:0.2",
    )
    p.add_argument(
        "-C", "--cors", action="append", default=[], metavar="ORIGIN",
        help="CORS allowed origin (repeatable). Omit for no CORS.",
    )
    p.add_argument(
        "-k", "--api-key", default=None, metavar="KEY",
        help="Require Bearer token auth. Falls back to $SAKLAS_API_KEY. Unset = no auth.",
    )
    return p


def _build_install_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="saklas install", description="Install a concept pack from HF or a local folder")
    p.add_argument("target", help="<ns>/<concept>[@revision] or path to a concept folder")
    p.add_argument("-s", "--statements-only", action="store_true",
                   help="Keep statements.json only; drop any bundled tensors")
    p.add_argument("-a", "--as", dest="as_target", default=None, metavar="NS/NAME",
                   help="Relocate the installed pack under a different namespace/name")
    p.add_argument("-f", "--force", action="store_true",
                   help="Overwrite an existing installation")
    return p


def _build_refresh_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="saklas refresh",
                                description="Re-pull concept(s) from their source (or `neutrals` to refresh neutral_statements.json)")
    p.add_argument("selector", help="Selector (name, tag:x, namespace:x, default, all) or the literal 'neutrals'")
    p.add_argument("-m", "--model", default=None, metavar="MODEL_ID",
                   help="Scope the refresh to one model's tensors (delete its safetensors + sidecar)")
    return p


def _build_clear_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="saklas clear",
                                description="Delete per-model tensors for matched concepts (keeps statements.json)")
    p.add_argument("selector", help="Selector (name, tag:x, namespace:x, default, all)")
    p.add_argument("-m", "--model", default=None, metavar="MODEL_ID",
                   help="Scope to one model's tensors only (default: all models)")
    p.add_argument("-y", "--yes", action="store_true",
                   help="Skip confirmation prompt on broad selectors")
    return p


def _build_uninstall_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="saklas uninstall",
                                description="Fully remove a concept folder (tensors + statements + pack.json)")
    p.add_argument("selector", help="Selector (name, tag:x, namespace:x, default, all)")
    p.add_argument("-y", "--yes", action="store_true",
                   help="Required for broad selectors (all, namespace:)")
    return p


def _build_list_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="saklas list",
                                description="List installed concepts (and HF concepts by default)")
    p.add_argument("selector", nargs="?", default=None,
                   help="Optional selector (name, tag:x, namespace:x, default, all)")
    p.add_argument("-i", "--installed", action="store_true",
                   help="Show only locally installed concepts (skip HF query)")
    p.add_argument("-j", "--json", dest="json_output", action="store_true",
                   help="Emit machine-readable JSON instead of a table")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Include descriptions in the table output")
    return p


def _build_push_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="saklas push",
                                description="Push a concept pack to Hugging Face as a model repo")
    p.add_argument("selector", help="Single concept selector (name or ns/name)")
    p.add_argument("-a", "--as", dest="as_target", default=None, metavar="OWNER/NAME",
                   help="Target HF coord (default: <whoami>/<pack_name>)")
    p.add_argument("-p", "--private", action="store_true",
                   help="Create the repo as private")
    p.add_argument("-m", "--model", default=None, metavar="MODEL_ID",
                   help="Only push tensors for this base model")
    p.add_argument("-s", "--statements-only", action="store_true",
                   help="Push statements.json only; skip all tensors")
    p.add_argument("-n", "--no-statements", action="store_true",
                   help="Skip statements.json; push tensors only")
    p.add_argument("-t", "--tag-version", action="store_true",
                   help="Create git tag v<pack.version> on the commit")
    p.add_argument("-d", "--dry-run", action="store_true",
                   help="Stage the upload but don't contact HF")
    p.add_argument("-f", "--force", action="store_true",
                   help="Allow republishing a pack whose source is bundled/hf://")
    return p


def _build_merge_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="saklas merge",
                                description="Merge existing vectors into a new pack")
    p.add_argument("name", help="New pack name (written under local/)")
    p.add_argument("components", help="Comma-separated components: ns/a:0.3,ns/b:0.4")
    p.add_argument("-f", "--force", action="store_true",
                   help="Overwrite an existing merged pack")
    p.add_argument("-s", "--strict", action="store_true",
                   help="Fail if any component is missing")
    p.add_argument("-m", "--model", default=None, metavar="MODEL_ID",
                   help="Restrict the merge to tensors for one model")
    return p


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def _argv_has_config(argv: list[str]) -> bool:
    for a in argv:
        if a in ("-c", "--config") or a.startswith("--config="):
            return True
    return False


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    if argv is None:
        argv = sys.argv[1:]

    # Zero-arg: print friendly hint and exit cleanly.
    if not argv:
        _print_no_model_hint()
        sys.exit(0)

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
        elif cmd == "push":
            args = _build_push_parser().parse_args(rest)
        else:  # pragma: no cover
            raise RuntimeError(f"unhandled subcommand {cmd}")
        args.command = cmd
        return args

    # Bare form: TUI. Allow missing model only when -c/--config is given,
    # so the normal `saklas -h` usage line shows `model` unbracketed.
    model_optional = _argv_has_config(argv)
    args = _build_tui_parser(model_optional=model_optional).parse_args(argv)
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
        # Reached only via `saklas -c foo.yaml` whose YAML doesn't set a model.
        _print_no_model_hint()
        sys.exit(0)
    return args


def _print_no_model_hint() -> None:
    msg = (
        "saklas needs a model to run.\n"
        "\n"
        "Pass a HuggingFace repo id or local path, e.g.\n"
        "  saklas Qwen/Qwen3.5-2B\n"
        "  saklas google/gemma-4-E2B-it\n"
        "\n"
        "Browse more models at https://huggingface.co/models?pipeline_tag=text-generation\n"
        "Run `saklas --help` for all options, or `saklas list` to see installed concept packs."
    )
    print(msg)


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

    from saklas.cli_selectors import resolve_pole
    for coord, alpha in getattr(args, "config_vectors", {}).items():
        ns = coord.split("/", 1)[0] if "/" in coord else None
        raw_name = coord.split("/", 1)[-1] if "/" in coord else coord
        try:
            canonical, sign, _match = resolve_pole(raw_name, namespace=ns)
            effective_alpha = alpha * sign
            _, profile = session.extract(canonical)
            registry_key = f"{ns}/{canonical}" if ns else canonical
            session.steer(registry_key, profile)
            print(f"  Registered '{registry_key}' (alpha={effective_alpha})")
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

    from saklas.cli_selectors import resolve_pole, AmbiguousSelectorError
    default_alphas: dict[str, float] = {}
    for spec in args.steer:
        raw, alpha = _parse_steer_flag(spec)
        try:
            canonical, sign, _match = resolve_pole(raw)
        except AmbiguousSelectorError as e:
            print(f"  Failed to resolve '{raw}': {e}", file=sys.stderr)
            sys.exit(1)
        effective_alpha = alpha * sign
        print(f"Extracting steering vector: {canonical}"
              + (f" (negated from '{raw}')" if sign < 0 else ""))
        _, profile = session.extract(canonical, on_progress=lambda m: print(f"  {m}"))
        session.steer(canonical, profile)
        default_alphas[canonical] = effective_alpha
        print(f"  Registered '{canonical}' (default alpha={effective_alpha})")

    from saklas.server import create_app
    app = create_app(session, default_alphas=default_alphas,
                     cors_origins=args.cors or None,
                     api_key=getattr(args, "api_key", None))

    _warmup_session(session)

    print(f"\nServing on http://{args.host}:{args.port}")
    print(f"OpenAI-compatible:  http://{args.host}:{args.port}/v1")
    print(f"Ollama-compatible:  http://{args.host}:{args.port}/api")
    print(f"API docs:           http://{args.host}:{args.port}/docs")
    if args.port != 11434:
        print("Tip: for drop-in Ollama compatibility, run with `--port 11434`.")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


def _warmup_session(session) -> None:
    """Run a tiny stateless generation so the first real request is fast.

    Warms up lazy kernel compilation, KV cache allocation, and any JIT paths
    before uvicorn starts accepting traffic.
    """
    import time as _time
    print("Warming up generation kernels...", flush=True)
    orig_max = session.config.max_new_tokens
    try:
        session.config.max_new_tokens = 1
        start = _time.monotonic()
        session.generate("Hi", stateless=True)
        print(f"  warmed in {_time.monotonic() - start:.1f}s")
    except Exception as e:
        print(f"  warm-up skipped: {e}")
    finally:
        session.config.max_new_tokens = orig_max


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


def _run_push(args: argparse.Namespace) -> None:
    from saklas import cache_ops
    from saklas.cli_selectors import parse as sel_parse

    selector = sel_parse(args.selector)
    try:
        coord, url, sha = cache_ops.push(
            selector,
            as_=args.as_target,
            private=args.private,
            model_scope=args.model,
            statements_only=args.statements_only,
            no_statements=args.no_statements,
            tag_version=args.tag_version,
            dry_run=args.dry_run,
            force=args.force,
        )
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        sys.exit(2)

    if sha:
        print(f"Pushed {coord} -> {url} @ {sha[:12]}")
    elif args.dry_run:
        print(f"Dry-run: would push {coord} -> {url}")
    else:
        print(f"Pushed {coord} -> {url}")


_RUNNERS = {
    "tui": _run_tui,
    "serve": _run_serve,
    "install": _run_install,
    "refresh": _run_refresh,
    "clear": _run_clear,
    "uninstall": _run_uninstall,
    "list": _run_list,
    "merge": _run_merge,
    "push": _run_push,
}


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    runner = _RUNNERS[args.command]
    runner(args)
