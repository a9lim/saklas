"""Runner functions for saklas CLI subcommands."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from saklas.cli.parsers import _PACK_VERBS


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _resolve_probes(raw: list[str] | None) -> list[str]:
    from saklas.core.session import PROBE_CATEGORIES
    if raw is None or raw == ["all"]:
        return list(PROBE_CATEGORIES)
    if raw == ["none"] or raw == []:
        return []
    return raw


def _make_session(args: argparse.Namespace):
    from saklas.core.session import SaklasSession
    probe_categories = _resolve_probes(args.probes)
    return SaklasSession.from_pretrained(
        args.model, device=args.device, quantize=args.quantize,
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


def _load_effective_config(args: argparse.Namespace):
    """Compose ~/.saklas/config.yaml + any -c files and stamp args in place.

    Returns the composed ConfigFile (poles pre-resolved). Sets:
      args.config_vectors, args.temperature, args.top_p, args.thinking,
      args.system_prompt, args.max_tokens, and args.model (if YAML supplied it).
    """
    from saklas.cli.config_file import (
        ConfigFile, apply_flag_overrides, ensure_vectors_installed,
    )
    extras = [Path(p) for p in (getattr(args, "config", None) or [])]
    composed = ConfigFile.effective(extras, include_default=True)
    composed = apply_flag_overrides(
        composed,
        model=getattr(args, "model", None),
        temperature=None,
        top_p=None,
        max_tokens=None,
        system_prompt=None,
    )
    composed = composed.resolve_poles()
    if getattr(args, "model", None) is None:
        args.model = composed.model
    args.temperature = composed.temperature
    args.top_p = composed.top_p
    args.thinking = composed.thinking
    args.system_prompt = composed.system_prompt
    args.max_tokens = composed.max_tokens if composed.max_tokens is not None else 1024
    args.config_vectors = composed.vectors
    ensure_vectors_installed(composed, strict=getattr(args, "strict", False))
    return composed


def _print_startup(args: argparse.Namespace) -> None:
    print(f"Loading model: {args.model}")
    if args.quantize:
        print(f"Quantization: {args.quantize}")


def _setup_steering_vectors(
    session,
    vector_specs,
    *,
    verbose: bool = False,
) -> dict[str, float]:
    """Resolve pole aliases, extract profiles, register with session.

    vector_specs is either a dict[coord, alpha] or an iterable of
    ``(raw, alpha)`` tuples. Returns ``dict[registry_key, effective_alpha]``.
    """
    from saklas.cli.selectors import resolve_pole, AmbiguousSelectorError

    if isinstance(vector_specs, dict):
        items = [
            (coord.split("/", 1)[-1] if "/" in coord else coord,
             coord.split("/", 1)[0] if "/" in coord else None,
             coord, alpha)
            for coord, alpha in vector_specs.items()
        ]
    else:
        items = [(raw, None, raw, alpha) for raw, alpha in vector_specs]

    default_alphas: dict[str, float] = {}
    for raw_name, ns, display, alpha in items:
        try:
            canonical, sign, _match = resolve_pole(raw_name, namespace=ns)
        except AmbiguousSelectorError as e:
            if verbose:
                print(f"  Failed to resolve '{raw_name}': {e}", file=sys.stderr)
                sys.exit(1)
            print(f"  Failed to register '{display}': {e}")
            continue
        effective_alpha = alpha * sign
        try:
            if verbose:
                print(
                    f"Extracting steering vector: {canonical}"
                    + (f" (negated from '{raw_name}')" if sign < 0 else "")
                )
                _, profile = session.extract(
                    canonical, on_progress=lambda m: print(f"  {m}")
                )
            else:
                _, profile = session.extract(canonical)
        except Exception as e:
            if verbose:
                raise
            print(f"  Failed to register '{display}': {e}")
            continue
        registry_key = f"{ns}/{canonical}" if ns else canonical
        session.steer(registry_key, profile)
        default_alphas[registry_key] = effective_alpha
        print(f"  Registered '{registry_key}' (alpha={effective_alpha})"
              if not verbose else
              f"  Registered '{registry_key}' (default alpha={effective_alpha})")
    return default_alphas


def _parse_steer_flag(raw: str) -> tuple[str, float]:
    if ":" in raw:
        name, alpha_s = raw.rsplit(":", 1)
        return name, float(alpha_s)
    return raw, 0.0


def _warmup_session(session) -> None:
    """Run a tiny stateless generation so the first real request is fast."""
    import time as _time
    from saklas.core.sampling import SamplingConfig
    print("Warming up generation kernels...", flush=True)
    try:
        start = _time.monotonic()
        session.generate(
            "Hi",
            sampling=SamplingConfig(max_tokens=1),
            stateless=True,
        )
        print(f"  warmed in {_time.monotonic() - start:.1f}s")
    except Exception as e:
        print(f"  warm-up skipped: {e}")


# ---------------------------------------------------------------------------
# Top-level runners
# ---------------------------------------------------------------------------

def _run_tui(args: argparse.Namespace) -> None:
    _load_effective_config(args)
    if not args.model:
        print(
            "saklas tui: model required. Pass a HuggingFace repo id (e.g.\n"
            "  saklas tui google/gemma-2-2b-it\n"
            "or supply it via -c setup.yaml with a `model:` field.",
            file=sys.stderr,
        )
        sys.exit(2)

    _print_startup(args)
    session = _make_session(args)
    _print_model_info(session)

    _setup_steering_vectors(session, getattr(args, "config_vectors", {}) or {})

    from saklas.tui.app import SaklasApp
    app = SaklasApp(session=session)
    app.run()


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

    _load_effective_config(args)

    _print_startup(args)
    session = _make_session(args)
    _print_model_info(session)

    # Config-file vectors first, then explicit --steer flags on top.
    config_specs = getattr(args, "config_vectors", {}) or {}
    if config_specs:
        _setup_steering_vectors(session, config_specs, verbose=True)
    steer_specs = [_parse_steer_flag(spec) for spec in args.steer]
    default_alphas = _setup_steering_vectors(session, steer_specs, verbose=True)

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


# --- pack runners --------------------------------------------------------

def _run_pack(args: argparse.Namespace) -> None:
    pack_cmd = getattr(args, "pack_cmd", None)
    if pack_cmd is None:
        print("usage: saklas pack <verb> [...]")
        print()
        width = max(len(v) for v, _ in _PACK_VERBS)
        for v, desc in _PACK_VERBS:
            print(f"  {v:<{width}}  {desc}")
        print()
        print("Run `saklas pack <verb> -h` for verb-specific options.")
        sys.exit(0)
    runner = _PACK_RUNNERS[pack_cmd]
    runner(args)


def _run_install(args: argparse.Namespace) -> None:
    from saklas.io import cache_ops
    cache_ops.install(
        args.target,
        as_=args.as_target,
        force=args.force,
        statements_only=args.statements_only,
    )
    suffix = " (statements only)" if args.statements_only else ""
    print(f"Installed {args.target}{suffix}")


def _run_refresh(args: argparse.Namespace) -> None:
    from saklas.io import cache_ops
    from saklas.cli.selectors import parse as sel_parse

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
    from saklas.io import cache_ops
    from saklas.cli.selectors import parse as sel_parse

    selector = sel_parse(args.selector)
    if selector.kind in {"all", "namespace"} and not args.yes:
        print(
            f"refusing to clear a broad selector ({selector.kind}); pass --yes to confirm",
            file=sys.stderr,
        )
        sys.exit(2)
    n = cache_ops.delete_tensors(selector, args.model)
    print(f"Deleted {n} files")


def _run_rm(args: argparse.Namespace) -> None:
    from saklas.io import cache_ops
    from saklas.cli.selectors import parse as sel_parse

    selector = sel_parse(args.selector)
    try:
        n = cache_ops.uninstall(selector, yes=args.yes)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        sys.exit(2)
    print(f"Uninstalled {n} concept(s)")


def _run_ls(args: argparse.Namespace) -> None:
    from saklas.io import cache_ops
    from saklas.cli.selectors import parse as sel_parse

    selector = sel_parse(args.selector) if args.selector else None
    cache_ops.list_local_packs(
        selector,
        json_output=args.json_output,
        verbose=args.verbose,
    )


def _run_search(args: argparse.Namespace) -> None:
    from saklas.io import cache_ops
    cache_ops.search_remote_packs(
        args.query,
        json_output=args.json_output,
        verbose=args.verbose,
    )


def _run_export(args: argparse.Namespace) -> None:
    if args.format != "gguf":
        print(f"Unknown export format: {args.format}", file=sys.stderr)
        sys.exit(2)
    from saklas.io import cache_ops
    from saklas.cli.selectors import parse as sel_parse
    selector = sel_parse(args.selector)
    written = cache_ops.export_gguf(
        selector,
        model_scope=args.model,
        output=args.output,
        model_hint=args.model_hint,
    )
    for p in written:
        print(f"Wrote {p}")


def _run_merge(args: argparse.Namespace) -> None:
    from saklas.io import merge as merge_mod
    components = merge_mod.parse_components(args.components)
    dst = merge_mod.merge_into_pack(
        args.name, components, model=args.model,
        force=args.force, strict=args.strict,
    )
    print(f"Merged pack written to {dst}")


def _run_push(args: argparse.Namespace) -> None:
    from saklas.io import cache_ops
    from saklas.cli.selectors import parse as sel_parse

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


def _require_model(args: argparse.Namespace) -> None:
    if not args.model:
        print(f"{args.pack_cmd}: -m/--model is required", file=sys.stderr)
        sys.exit(2)


def _run_clone(args: argparse.Namespace) -> None:
    _require_model(args)
    from saklas.io.cloning import (
        CorpusTooShortError, CorpusTooLongError, InsufficientPairsError,
    )
    from saklas.cli.selectors import _all_concepts

    for c in _all_concepts():
        if c.name == args.name and c.namespace != "local":
            print(
                f"warning: '{args.name}' exists in namespace '{c.namespace}'; "
                f"reference this as 'local/{args.name}' to disambiguate",
                file=sys.stderr,
            )
            break

    _print_startup(args)
    session = _make_session(args)
    _print_model_info(session)

    try:
        canonical, _profile = session.clone_from_corpus(
            args.corpus_path,
            name=args.name,
            n_pairs=args.n_pairs,
            seed=args.seed,
            force=args.force,
        )
    except (CorpusTooShortError, CorpusTooLongError, InsufficientPairsError) as e:
        print(f"clone failed: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"Cloned persona -> local/{canonical}")


def _run_extract(args: argparse.Namespace) -> None:
    _require_model(args)
    from saklas.core.session import canonical_concept_name

    if len(args.concept) == 1:
        raw = args.concept[0]
        baseline = None
    elif len(args.concept) == 2:
        raw = args.concept[0]
        baseline = args.concept[1]
    else:
        print(
            "extract: expected 1 or 2 positional arguments "
            f"(got {len(args.concept)})",
            file=sys.stderr,
        )
        sys.exit(2)

    _print_startup(args)
    session = _make_session(args)
    _print_model_info(session)

    canonical = canonical_concept_name(raw, baseline)

    import pathlib
    from saklas.io.paths import safe_model_id
    from saklas.cli.selectors import _all_concepts
    candidate_folders = [c.folder for c in _all_concepts() if c.name == canonical]
    candidate_folders.append(session._local_concept_folder(canonical))
    sid = safe_model_id(session.model_id)
    candidate_paths = [
        pathlib.Path(folder) / f"{sid}.safetensors" for folder in candidate_folders
    ]
    existing = next((p for p in candidate_paths if p.exists()), None)

    if existing is not None and not args.force:
        print(f"already extracted at {existing}")
        sys.exit(0)

    if args.force:
        for p in candidate_paths:
            if p.exists():
                p.unlink()

    try:
        if baseline is not None:
            canonical, _profile = session.extract(raw, baseline=baseline)
        else:
            canonical, _profile = session.extract(raw)
    except Exception as e:
        print(f"extract failed: {e}", file=sys.stderr)
        sys.exit(1)

    final_path = next((p for p in candidate_paths if p.exists()), None)
    if final_path is None:
        final_path = (
            pathlib.Path(session._local_concept_folder(canonical)) / f"{sid}.safetensors"
        )
    print(f"extracted {canonical} -> {final_path}")


_PACK_RUNNERS = {
    "install": _run_install,
    "refresh": _run_refresh,
    "clear":   _run_clear,
    "rm":      _run_rm,
    "ls":      _run_ls,
    "search":  _run_search,
    "merge":   _run_merge,
    "push":    _run_push,
    "export":  _run_export,
    "clone":   _run_clone,
    "extract": _run_extract,
}


# --- config runners ------------------------------------------------------

def _run_config(args: argparse.Namespace) -> None:
    cmd = getattr(args, "config_cmd", None)
    if cmd == "show":
        _run_config_show(args)
    elif cmd == "validate":
        _run_config_validate(args)
    else:
        print("usage: saklas config {show,validate}")
        print()
        print("  show      Print the effective merged config")
        print("  validate  Validate a config file (exit 0 valid, 2 invalid)")
        sys.exit(0)


def _run_config_show(args: argparse.Namespace) -> None:
    from saklas import __version__
    from saklas.cli.config_file import ConfigFile, apply_flag_overrides
    extras = [Path(p) for p in (args.config or [])]
    composed = ConfigFile.effective(extras, include_default=not args.no_default)
    if args.model is not None:
        composed = apply_flag_overrides(composed, model=args.model)
    composed = composed.resolve_poles()
    header = f"# effective merged config for saklas {__version__}"
    sys.stdout.write(composed.to_yaml(header=header))


def _run_config_validate(args: argparse.Namespace) -> None:
    from saklas.cli.config_file import ConfigFile, ConfigFileError
    p = Path(args.file)
    if not p.exists():
        print(f"config validate: {p}: file not found", file=sys.stderr)
        sys.exit(2)
    try:
        cfg = ConfigFile.load(p)
        cfg = cfg.resolve_poles()
        # Dry-run: don't install, just check resolvability.
        from saklas.cli.selectors import _all_concepts
        installed = {(c.namespace, c.name) for c in _all_concepts()}
        installed_names = {c.name for c in _all_concepts()}
        missing: list[str] = []
        for coord in cfg.vectors:
            if "/" in coord:
                ns, name = coord.split("/", 1)
                if ns == "default" or (ns, name) in installed:
                    continue
                if ns == "local":
                    missing.append(coord)
                    continue
                # HF namespace — we assume install would succeed; don't probe.
                continue
            else:
                if coord in installed_names:
                    continue
                missing.append(coord)
        if missing:
            raise ConfigFileError(
                f"unresolvable vectors (not installed and no namespace to install from): {missing}"
            )
    except ConfigFileError as e:
        print(f"config validate: {p}: {e}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"config validate: {p}: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(2)
    print(f"{p}: ok")


_COMMAND_RUNNERS = {
    "tui":    _run_tui,
    "serve":  _run_serve,
    "pack":   _run_pack,
    "config": _run_config,
}
