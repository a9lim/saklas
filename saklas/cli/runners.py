"""Runner functions for saklas CLI subcommands."""

from __future__ import annotations

import argparse
import functools
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, TypeVar

from saklas.cli.parsers import _PACK_VERBS, _VECTOR_VERBS
from saklas.core.errors import SaklasError

if TYPE_CHECKING:
    from saklas.core.steering import Steering


_R = TypeVar("_R")


def _saklas_error_exit(fn: Callable[..., _R]) -> Callable[..., _R]:
    """Translate any ``SaklasError`` escaping a runner to a stderr line + exit.

    Maps the exception's HTTP-style status (from ``user_message()``) to a
    process exit code via ``min(2, code // 100)``: 4xx/5xx land on exit 2,
    nothing softer. The TUI is excluded — it owns its own surface.
    """
    @functools.wraps(fn)
    def _wrapper(*args: object, **kwargs: object) -> _R:
        try:
            return fn(*args, **kwargs)
        except SaklasError as e:
            code, msg = e.user_message()
            print(msg, file=sys.stderr)
            sys.exit(min(2, code // 100))
    return _wrapper


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
    expression: "str | None",
    *,
    verbose: bool = False,
) -> "Steering | None":
    """Extract + register every concept referenced by ``expression``.

    Walks the raw AST via :func:`referenced_selectors` so namespace
    prefixes drive extraction site selection, then returns the parsed
    :class:`Steering` with every atom pre-warmed in ``session._profiles``.
    Returns ``None`` when ``expression`` is empty / falsy.
    """
    from saklas.io.selectors import resolve_pole, AmbiguousSelectorError
    from saklas.core.steering_expr import (
        parse_expr, referenced_selectors,
    )

    if not expression:
        return None

    for ns, concept, _variant in referenced_selectors(expression):
        raw_name = concept
        display = f"{ns}/{concept}" if ns else concept
        try:
            canonical, sign, _match, _variant = resolve_pole(raw_name, namespace=ns)
        except AmbiguousSelectorError as e:
            if verbose:
                print(f"  Failed to resolve '{raw_name}': {e}", file=sys.stderr)
                sys.exit(1)
            print(f"  Failed to register '{display}': {e}")
            continue
        try:
            if verbose:
                print(
                    f"Extracting steering vector: {canonical}"
                    + (f" (negated from '{raw_name}')" if sign < 0 else "")
                )
                _, profile = session.extract(
                    canonical, on_progress=lambda m: print(f"  {m}"),
                    namespace=ns,
                )
            else:
                _, profile = session.extract(canonical, namespace=ns)
        except Exception as e:
            if verbose:
                raise
            print(f"  Failed to register '{display}': {e}")
            continue
        registry_key = canonical
        session.steer(registry_key, profile)
        print(f"  Registered '{registry_key}'"
              if not verbose else
              f"  Registered '{registry_key}'")

    return parse_expr(expression)


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

@_saklas_error_exit
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

    _setup_steering_vectors(session, getattr(args, "config_vectors", None))

    from saklas.tui.app import SaklasApp
    app = SaklasApp(session=session)
    app.run()


@_saklas_error_exit
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

    # Config-file vectors first, then any explicit --steer expression on top.
    config_expr = getattr(args, "config_vectors", None)
    if config_expr:
        _setup_steering_vectors(session, config_expr, verbose=True)
    steer_expr: str | None = args.steer
    default_steering = _setup_steering_vectors(session, steer_expr, verbose=True)
    if default_steering is None and config_expr:
        from saklas.core.steering_expr import parse_expr
        default_steering = parse_expr(config_expr)

    from saklas.server import create_app
    app = create_app(session, default_steering=default_steering,
                     cors_origins=args.cors or None,
                     api_key=getattr(args, "api_key", None),
                     web=getattr(args, "web", False))

    _warmup_session(session)

    print(f"\nServing on http://{args.host}:{args.port}")
    print(f"OpenAI-compatible:  http://{args.host}:{args.port}/v1")
    print(f"Ollama-compatible:  http://{args.host}:{args.port}/api")
    print(f"API docs:           http://{args.host}:{args.port}/docs")
    if args.port != 11434:
        print("Tip: for drop-in Ollama compatibility, run with `--port 11434`.")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


# --- pack runners --------------------------------------------------------

@_saklas_error_exit
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
    from saklas.io.selectors import parse as sel_parse

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
    from saklas.io.selectors import parse as sel_parse

    selector = sel_parse(args.selector)
    if selector.kind in {"all", "namespace"} and not args.yes:
        print(
            f"refusing to clear a broad selector ({selector.kind}); pass --yes to confirm",
            file=sys.stderr,
        )
        sys.exit(2)
    n = cache_ops.delete_tensors(selector, args.model, variant=args.variant)
    print(f"Deleted {n} files")


def _run_rm(args: argparse.Namespace) -> None:
    from saklas.io import cache_ops
    from saklas.io.selectors import parse as sel_parse

    selector = sel_parse(args.selector)
    try:
        n = cache_ops.uninstall(selector, yes=args.yes)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        sys.exit(2)
    print(f"Uninstalled {n} concept(s)")


def _run_ls(args: argparse.Namespace) -> None:
    from saklas.cli.output import render_local_pack_list
    from saklas.io.selectors import parse as sel_parse

    selector = sel_parse(args.selector) if args.selector else None
    render_local_pack_list(
        selector,
        json_output=args.json_output,
        verbose=args.verbose,
    )


def _run_search(args: argparse.Namespace) -> None:
    from saklas.cli.output import render_remote_search
    render_remote_search(
        args.query,
        json_output=args.json_output,
        verbose=args.verbose,
    )


def _run_export(args: argparse.Namespace) -> None:
    if args.format != "gguf":
        print(f"Unknown export format: {args.format}", file=sys.stderr)
        sys.exit(2)
    from saklas.io import cache_ops
    from saklas.io.selectors import parse as sel_parse
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
    dst = merge_mod.merge_into_pack(
        args.name, args.expression, model=args.model,
        force=args.force, strict=args.strict,
    )
    print(f"Merged pack written to {dst}")


def _run_push(args: argparse.Namespace) -> None:
    from saklas.io import cache_ops
    from saklas.io.selectors import parse as sel_parse

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
            variant=args.variant,
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
        cmd = getattr(args, "vector_cmd", None) or getattr(args, "pack_cmd", None) or "?"
        print(f"{cmd}: -m/--model is required", file=sys.stderr)
        sys.exit(2)


def _run_clone(args: argparse.Namespace) -> None:
    _require_model(args)
    from saklas.io.cloning import (
        CorpusTooShortError, CorpusTooLongError, InsufficientPairsError,
    )
    from saklas.io.selectors import _all_concepts

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
    from saklas.io.paths import tensor_filename
    from saklas.io.selectors import _all_concepts
    candidate_folders = [c.folder for c in _all_concepts() if c.name == canonical]
    candidate_folders.append(session._local_concept_folder(canonical))
    requested_release = getattr(args, "sae", None)
    candidate_tensor_name = tensor_filename(session.model_id, release=requested_release)
    candidate_paths = [
        pathlib.Path(folder) / candidate_tensor_name for folder in candidate_folders
    ]
    existing = next((p for p in candidate_paths if p.exists()), None)

    if existing is not None and not args.force:
        print(f"already extracted at {existing}")
        sys.exit(0)

    if args.force:
        for p in candidate_paths:
            if p.exists():
                p.unlink()

    extract_kwargs = {}
    if getattr(args, "sae", None):
        extract_kwargs["sae"] = args.sae
    if getattr(args, "sae_revision", None):
        extract_kwargs["sae_revision"] = args.sae_revision

    try:
        if baseline is not None:
            canonical, _profile = session.extract(raw, baseline=baseline, **extract_kwargs)
        else:
            canonical, _profile = session.extract(raw, **extract_kwargs)
    except Exception as e:
        print(f"extract failed: {e}", file=sys.stderr)
        sys.exit(1)

    # `canonical` may be "name:sae-<release>" — peel it for filename construction.
    if ":sae-" in canonical:
        core_name, _, rel = canonical.partition(":sae-")
        tensor_name = tensor_filename(session.model_id, release=rel)
    else:
        core_name = canonical
        tensor_name = tensor_filename(session.model_id, release=None)
    final_paths = [pathlib.Path(f) / tensor_name for f in candidate_folders]
    final_path = next((p for p in final_paths if p.exists()), None)
    if final_path is None:
        final_path = (
            pathlib.Path(session._local_concept_folder(core_name)) / tensor_name
        )
    print(f"extracted {canonical} -> {final_path}")


_PACK_RUNNERS = {
    "install": _run_install,
    "refresh": _run_refresh,
    "clear":   _run_clear,
    "rm":      _run_rm,
    "ls":      _run_ls,
    "search":  _run_search,
    "push":    _run_push,
    "export":  _run_export,
}


# --- vector runners ------------------------------------------------------


# --- config runners ------------------------------------------------------

@_saklas_error_exit
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
    header = f"# effective merged config for saklas {__version__}"
    sys.stdout.write(composed.to_yaml(header=header))


def _run_config_validate(args: argparse.Namespace) -> None:
    from saklas.cli.config_file import ConfigFile, ConfigFileError
    from saklas.core.steering_expr import referenced_selectors
    p = Path(args.file)
    if not p.exists():
        print(f"config validate: {p}: file not found", file=sys.stderr)
        sys.exit(2)
    try:
        cfg = ConfigFile.load(p)
        if cfg.vectors is None:
            print(f"{p}: ok")
            return
        # Dry-run: don't install, just check resolvability.
        from saklas.io.selectors import _all_concepts
        installed = {(c.namespace, c.name) for c in _all_concepts()}
        installed_names = {c.name for c in _all_concepts()}
        missing: list[str] = []
        for ns, concept, _variant in referenced_selectors(cfg.vectors):
            if ns is None:
                if concept in installed_names:
                    continue
                # Bare pole of an installed bipolar resolves fine too.
                slug = concept.split(".")[0] if "." in concept else concept
                if any(
                    slug in c.name.split(".")
                    for c in _all_concepts()
                    if "." in c.name
                ):
                    continue
                missing.append(concept)
                continue
            if ns == "default" or (ns, concept) in installed:
                continue
            if ns == "local":
                missing.append(f"{ns}/{concept}")
                continue
            # HF namespace — we assume install would succeed; don't probe.
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


_VARIANT_SUFFIX_RE = re.compile(r"^(raw|sae(?:-[a-z0-9._-]+)?)$")


def _split_variant_suffix(raw: str) -> tuple[str, str | None]:
    """Peel a trailing ``:<variant>`` off a selector string.

    Returns ``(name_part, variant_or_None)``. ``variant`` is ``"raw"``,
    ``"sae"``, or ``"sae-<release>"``. Non-variant colon usage
    (``tag:``, ``namespace:``, ``model:``) passes through unchanged with
    ``variant=None`` — those prefixes are caught by ``sel.parse`` later.
    """
    if ":" not in raw:
        return raw, None
    head, _, tail = raw.rpartition(":")
    if _VARIANT_SUFFIX_RE.match(tail) and head and "/" not in tail:
        # Guard against ``model:<org>/<name>`` where the ``/`` lives in
        # the right half of the final ``:``.
        return head, tail
    return raw, None


def _resolve_variant_tensor(
    folder,
    model_id: str,
    variant: str | None,
) -> "Path | None":
    """Locate the on-disk tensor for ``(folder, model, variant)``.

    ``variant`` semantics:
      - ``None`` (no suffix passed): prefer raw safetensors, fall back
        to GGUF.
      - ``"raw"``: require the raw safetensors tensor.
      - ``"sae"``: require the unique SAE variant; raise
        :class:`AmbiguousVariantError` when >1, :class:`UnknownVariantError`
        when 0.
      - ``"sae-<release>"``: require that specific release.
    """
    from saklas.core.errors import AmbiguousVariantError, UnknownVariantError
    from saklas.io.packs import enumerate_variants

    variants = enumerate_variants(folder, model_id)

    if variant is None:
        # Raw preferred, GGUF fallback.
        if "raw" in variants:
            return variants["raw"]
        from saklas.io.paths import safe_model_id as _safe
        gguf = folder / f"{_safe(model_id)}.gguf"
        return gguf if gguf.is_file() else None

    if variant == "raw":
        return variants.get("raw")

    if variant == "sae":
        sae_paths = {k: v for k, v in variants.items() if k.startswith("sae-")}
        if len(sae_paths) == 0:
            raise UnknownVariantError(
                f"no SAE variants found in {folder.name} for model {model_id} "
                f"(available: {sorted(variants) or 'none'})"
            )
        if len(sae_paths) > 1:
            raise AmbiguousVariantError(
                f"{folder.name}: multiple SAE variants for model {model_id}: "
                f"{sorted(sae_paths)}. Specify with :sae-<release>."
            )
        return next(iter(sae_paths.values()))

    # ``sae-<release>``
    path = variants.get(variant)
    if path is None:
        raise UnknownVariantError(
            f"variant '{variant}' not found in {folder.name} for model "
            f"{model_id} (available: {sorted(variants) or 'none'})"
        )
    return path


def _run_compare(args: argparse.Namespace) -> None:
    import json as _json
    from saklas.io.selectors import parse as sel_parse, resolve
    from saklas.core.errors import AmbiguousVariantError, UnknownVariantError
    from saklas.io.paths import vectors_dir
    from saklas.core.profile import Profile, ProfileError

    # Expand selectors into (name, variant) pairs. Variant travels with the
    # name through the load loop so ``foo:sae`` picks the SAE tensor.
    names: list[tuple[str, str | None]] = []
    for raw in args.concepts:
        name_part, variant = _split_variant_suffix(raw)
        try:
            sel = sel_parse(name_part)
        except Exception:
            names.append((name_part, variant))
            continue
        if sel.kind == "name":
            names.append((name_part, variant))
        else:
            # Bulk selectors (tag:/namespace:/all) expand to individual
            # names; inherit the variant suffix so `tag:emotion:sae`
            # resolves SAE tensors across the tag.
            resolved = resolve(sel)
            for c in resolved:
                names.append((f"{c.namespace}/{c.name}", variant))

    # Load profiles from disk.
    profiles: dict[str, Profile] = {}
    for name, variant in names:
        sel = sel_parse(name)
        matches = resolve(sel)
        if not matches:
            print(f"warning: '{name}' not found, skipping", file=sys.stderr)
            continue
        folder = matches[0].folder
        try:
            tensor_path = _resolve_variant_tensor(folder, args.model, variant)
        except (AmbiguousVariantError, UnknownVariantError) as e:
            print(f"warning: {e}, skipping", file=sys.stderr)
            continue
        if tensor_path is None or not tensor_path.is_file():
            print(f"warning: no tensor for '{name}' with model {args.model}, skipping",
                  file=sys.stderr)
            continue
        # Display keys carry the variant when present so compare output
        # distinguishes raw vs SAE rows.
        display = matches[0].name if variant is None else f"{matches[0].name}:{variant}"
        try:
            profiles[display] = Profile.load(tensor_path)
        except (ProfileError, Exception) as e:
            print(f"warning: failed to load '{name}': {e}", file=sys.stderr)

    if len(profiles) < 1:
        print("compare: no loadable profiles found", file=sys.stderr)
        sys.exit(1)

    ordered = list(profiles.keys())

    # 1-arg mode: rank all installed against the target.
    if len(args.concepts) == 1 and len(ordered) == 1:
        target_name = ordered[0]
        target = profiles[target_name]

        # Load all other installed profiles for this model.
        others: dict[str, Profile] = {}
        vdir = vectors_dir()
        if vdir.is_dir():
            for ns_dir in sorted(vdir.iterdir()):
                if not ns_dir.is_dir():
                    continue
                for cdir in sorted(ns_dir.iterdir()):
                    if not cdir.is_dir():
                        continue
                    if cdir.name == target_name:
                        continue
                    # Auto-scan: raw preferred, GGUF fallback. SAE-vs-all
                    # ranking requires the caller to pass the SAE selector
                    # explicitly.
                    try:
                        tp = _resolve_variant_tensor(cdir, args.model, None)
                    except (AmbiguousVariantError, UnknownVariantError):
                        continue
                    if tp is None or not tp.is_file():
                        continue
                    try:
                        others[cdir.name] = Profile.load(tp)
                    except Exception:
                        continue

        if not others:
            print(f"compare: no other profiles found for model {args.model}", file=sys.stderr)
            sys.exit(1)

        scores = {name: target.cosine_similarity(p) for name, p in others.items()}
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        if args.json_output:
            result: dict = {"target": target_name, "model": args.model,
                            "similarities": [{"name": n, "similarity": round(s, 6)}
                                              for n, s in ranked]}
            if args.verbose:
                top3 = ranked[:3]
                result["per_layer_top3"] = {
                    n: {str(k): round(v, 6)
                        for k, v in target.cosine_similarity(others[n], per_layer=True).items()}
                    for n, _ in top3
                }
            print(_json.dumps(result, indent=2))
        else:
            width = max(len(n) for n, _ in ranked)
            print(f"{target_name} vs all installed ({args.model}):")
            for name, score in ranked:
                print(f"  {name:<{width}}  {score:+.4f}")
            if args.verbose and ranked:
                print()
                print("  per-layer (top 3):")
                for name, _ in ranked[:3]:
                    per_layer = target.cosine_similarity(others[name], per_layer=True)
                    print(f"    {name}:")
                    for layer in sorted(per_layer):
                        print(f"      layer {layer:>3}: {per_layer[layer]:+.4f}")
        return

    if len(ordered) < 2:
        print("compare: need at least 2 profiles to compare", file=sys.stderr)
        sys.exit(1)

    # 2-arg mode: pairwise.
    if len(ordered) == 2:
        a_name, b_name = ordered
        a, b = profiles[a_name], profiles[b_name]
        sim = a.cosine_similarity(b)

        if args.json_output:
            result = {"a": a_name, "b": b_name, "model": args.model,
                      "similarity": round(sim, 6)}
            if args.verbose:
                result["per_layer"] = {str(k): round(v, 6)
                                       for k, v in a.cosine_similarity(b, per_layer=True).items()}
            print(_json.dumps(result, indent=2))
        else:
            print(f"{a_name} ~ {b_name}: {sim:+.4f}")
            if args.verbose:
                per_layer = a.cosine_similarity(b, per_layer=True)
                for layer in sorted(per_layer):
                    print(f"  layer {layer:>3}: {per_layer[layer]:+.4f}")
        return

    # 3+ mode: N×N matrix.
    matrix: dict[str, dict[str, float]] = {}
    for a_name in ordered:
        matrix[a_name] = {}
        for b_name in ordered:
            if a_name == b_name:
                matrix[a_name][b_name] = 1.0
            else:
                matrix[a_name][b_name] = profiles[a_name].cosine_similarity(profiles[b_name])

    if args.json_output:
        result = {"model": args.model, "concepts": ordered,
                  "matrix": {a: {b: round(v, 6) for b, v in row.items()}
                              for a, row in matrix.items()}}
        if args.verbose:
            per_layer: dict[str, dict[str, float]] = {}
            for i, a_name in enumerate(ordered):
                for b_name in ordered[i + 1:]:
                    key = f"{a_name}|{b_name}"
                    per_layer[key] = {
                        str(k): round(v, 6)
                        for k, v in profiles[a_name].cosine_similarity(
                            profiles[b_name], per_layer=True
                        ).items()
                    }
            result["per_layer"] = per_layer
        print(_json.dumps(result, indent=2))
    else:
        width = max(len(n) for n in ordered)
        header = " " * (width + 2) + "  ".join(f"{n:>{width}}" for n in ordered)
        print(header)
        for a_name in ordered:
            row = "  ".join(f"{matrix[a_name][b]:>{width}.4f}" for b in ordered)
            print(f"{a_name:<{width}}  {row}")


def _run_why(args: argparse.Namespace) -> None:
    import json as _json
    from saklas.io.selectors import parse as sel_parse, resolve
    from saklas.core.errors import AmbiguousVariantError, UnknownVariantError
    from saklas.core.profile import Profile, ProfileError

    # Peel off a ``:<variant>`` suffix before parsing as a selector.
    name_part, variant = _split_variant_suffix(args.concept)
    sel = sel_parse(name_part)
    matches = resolve(sel)
    if not matches:
        print(f"why: '{args.concept}' not found", file=sys.stderr)
        sys.exit(1)

    folder = matches[0].folder
    concept_name = matches[0].name if variant is None else f"{matches[0].name}:{variant}"

    try:
        tensor_path = _resolve_variant_tensor(folder, args.model, variant)
    except (AmbiguousVariantError, UnknownVariantError) as e:
        print(f"why: {e}", file=sys.stderr)
        sys.exit(1)
    if tensor_path is None or not tensor_path.is_file():
        print(
            f"why: no tensor for '{args.concept}' with model {args.model}",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        profile = Profile.load(tensor_path)
    except (ProfileError, Exception) as e:
        print(f"why: failed to load '{args.concept}': {e}", file=sys.stderr)
        sys.exit(1)

    layer_mags: list[tuple[int, float]] = sorted(
        ((layer, float(tensor.norm().item())) for layer, tensor in profile.items()),
        key=lambda kv: kv[0],
    )
    total_layers = len(profile)
    diagnostics = profile.diagnostics  # None when extracted before saklas 1.6

    if args.json_output:
        result: dict[str, Any] = {
            "concept": concept_name,
            "model": args.model,
            "total_layers": total_layers,
            "layers": [{"layer": l, "magnitude": round(m, 6)} for l, m in layer_mags],
        }
        if diagnostics is not None:
            result["diagnostics_by_layer"] = {
                str(layer): {k: round(float(v), 6) for k, v in metrics.items()}
                for layer, metrics in sorted(diagnostics.items())
            }
            result["diagnostics_summary"] = _summarize_diagnostics(diagnostics)
        print(_json.dumps(result, indent=2))
    else:
        _print_why_histogram(concept_name, args.model, total_layers, layer_mags)
        if diagnostics is not None:
            _print_diagnostics(diagnostics)


def _summarize_diagnostics(
    diagnostics: dict[int, dict[str, float]],
) -> dict[str, float | str]:
    """Aggregate per-layer metrics into a small summary block.

    Reports medians (robust to outlier layers) for the four metrics, plus
    a coarse ``quality`` stoplight derived from the same thresholds the
    extraction-time warning uses.  Mirrored in the JSON output so callers
    don't have to recompute it client-side.
    """
    def _median(values: list[float]) -> float:
        s = sorted(values)
        n = len(s)
        if n == 0:
            return 0.0
        mid = n // 2
        if n % 2 == 1:
            return s[mid]
        return 0.5 * (s[mid - 1] + s[mid])

    evrs = [m["evr"] for m in diagnostics.values() if "evr" in m]
    intras = [
        m["intra_pair_variance_mean"]
        for m in diagnostics.values()
        if "intra_pair_variance_mean" in m
    ]
    aligns = [
        m["inter_pair_alignment"]
        for m in diagnostics.values()
        if "inter_pair_alignment" in m
    ]
    projs = [
        m["diff_principal_projection"]
        for m in diagnostics.values()
        if "diff_principal_projection" in m
    ]

    med_evr = _median(evrs) if evrs else 0.0
    med_intra = _median(intras) if intras else 0.0
    med_align = _median(aligns) if aligns else 0.0
    med_proj = _median(projs) if projs else 0.0

    if med_evr > 0.95 and med_intra < 0.01:
        quality = "poor"
    elif med_align < 0.2:
        quality = "poor"
    elif med_align < 0.4 or med_evr < 0.2:
        quality = "shaky"
    else:
        quality = "solid"

    return {
        "median_evr": round(med_evr, 4),
        "median_intra_pair_variance": round(med_intra, 4),
        "median_inter_pair_alignment": round(med_align, 4),
        "median_diff_principal_projection": round(med_proj, 4),
        "quality": quality,
    }


def _print_diagnostics(diagnostics: dict[int, dict[str, float]]) -> None:
    """Render the diagnostics summary + per-layer table beneath the histogram."""
    summary = _summarize_diagnostics(diagnostics)
    quality = summary["quality"]
    print()
    print(f"  DIAGNOSTICS (probe quality: {quality}):")
    print(
        f"    median EVR:                 {summary['median_evr']:.3f}\n"
        f"    median intra-pair variance: {summary['median_intra_pair_variance']:.4f}\n"
        f"    median inter-pair alignment:{summary['median_inter_pair_alignment']:>7.3f}\n"
        f"    median diff→PC projection:  {summary['median_diff_principal_projection']:.3f}"
    )


def _print_why_histogram(
    concept_name: str,
    model_id: str,
    total_layers: int,
    layer_mags: list[tuple[int, float]],
) -> None:
    import shutil
    from saklas.core.histogram import HIST_BUCKETS, bucketize

    print(f"{concept_name} ({total_layers} layers, {model_id}):")
    print("  LAYERS (mean ||baked|| per bucket):")
    if not layer_mags:
        return

    term_w = shutil.get_terminal_size((80, 24)).columns
    buckets = bucketize(layer_mags, HIST_BUCKETS)
    max_norm = max(v for _, _, v in buckets) or 1.0
    label_w = max(2, len(str(max(hi for _, hi, _ in buckets))))

    def _label(lo: int, hi: int) -> str:
        return f"L{lo:0{label_w}}" if lo == hi else f"L{lo:0{label_w}}-{hi:0{label_w}}"

    label_col = max(len(_label(lo, hi)) for lo, hi, _ in buckets)
    # "    <label>  <bar>  <value>" — 4 indent + label_col + 2 + bar + 2 + 8
    value_w = 8
    bar_w = max(12, term_w - 4 - label_col - 2 - 2 - value_w)
    for lo, hi, norm in buckets:
        filled = min(int(norm / max_norm * bar_w), bar_w)
        bar = "█" * filled + "░" * (bar_w - filled)
        print(f"    {_label(lo, hi):<{label_col}}  {bar}  {norm:>{value_w}.3f}")


def _run_transfer(args: argparse.Namespace) -> None:
    """Cross-model probe transfer via Procrustes (v1.6).

    Resolves the concept folder, loads the source-model tensor, fits
    (or loads) the per-layer alignment between source and target's
    cached neutral activations, applies the transfer, and writes the
    result at the target's ``_from-<safe_src>`` variant path with a
    sidecar carrying transfer provenance.
    """
    import json as _json

    from saklas.core.profile import Profile, ProfileError
    from saklas.io.alignment import (
        AlignmentError,
        alignment_cache_path,
        alignment_quality,
        fit_alignment,
        load_alignment_map,
        load_or_compute_neutral_activations,
        save_alignment_map,
        transfer_profile,
    )
    from saklas.io.packs import hash_file
    from saklas.io.paths import safe_model_id, sidecar_filename, tensor_filename
    from saklas.io.selectors import parse as sel_parse, resolve

    sel = sel_parse(args.concept)
    matches = resolve(sel)
    if not matches:
        print(f"transfer: '{args.concept}' not found", file=sys.stderr)
        sys.exit(1)
    if len(matches) > 1:
        qualified = ", ".join(f"{c.namespace}/{c.name}" for c in matches)
        print(
            f"transfer: '{args.concept}' is ambiguous (matches {qualified}); "
            f"specify ns/name",
            file=sys.stderr,
        )
        sys.exit(1)

    folder = matches[0].folder
    src_tensor = folder / tensor_filename(args.src_model)
    if not src_tensor.is_file():
        print(
            f"transfer: source tensor not found at {src_tensor} — extract "
            f"the concept on {args.src_model} first",
            file=sys.stderr,
        )
        sys.exit(1)

    tgt_tensor = folder / tensor_filename(
        args.tgt_model, transferred_from=args.src_model,
    )
    tgt_sidecar = folder / sidecar_filename(
        args.tgt_model, transferred_from=args.src_model,
    )
    if tgt_tensor.exists() and not args.force:
        print(
            f"transfer: target already exists at {tgt_tensor}; pass -f to recompute",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load + fit / load alignment.  Heavy work — both forward passes
    # and the SVD — so we lazy-load saklas.core.session to avoid paying
    # the model-load cost when the user just runs --help.
    from saklas.core.session import SaklasSession

    try:
        src_profile = Profile.load(src_tensor)
    except ProfileError as e:
        print(f"transfer: failed to load source profile: {e}", file=sys.stderr)
        sys.exit(1)

    cached = None if args.force else load_alignment_map(args.src_model, args.tgt_model)

    if cached is None:
        # Need both models loaded to compute neutrals.  Loading two
        # large models simultaneously is non-trivial; we serialize:
        # load src, compute its neutrals, drop, load tgt, compute, drop.
        print(
            f"transfer: fitting Procrustes alignment {args.src_model} -> {args.tgt_model} "
            f"(this may load each model briefly)...",
            file=sys.stderr,
        )

        with SaklasSession.from_pretrained(args.src_model, device="auto", probes=[]) as src_sess:
            src_acts = load_or_compute_neutral_activations(
                src_sess._model, src_sess._tokenizer, src_sess._layers,
                model_id=args.src_model, force=args.force,
            )

        with SaklasSession.from_pretrained(args.tgt_model, device="auto", probes=[]) as tgt_sess:
            tgt_acts = load_or_compute_neutral_activations(
                tgt_sess._model, tgt_sess._tokenizer, tgt_sess._layers,
                model_id=args.tgt_model, force=args.force,
            )

        try:
            M = fit_alignment(src_acts, tgt_acts)
        except AlignmentError as e:
            print(f"transfer: {e}", file=sys.stderr)
            sys.exit(1)

        quality_per_layer = alignment_quality(M, src_acts, tgt_acts)
        map_path = save_alignment_map(
            M, args.src_model, args.tgt_model,
            quality_per_layer=quality_per_layer,
        )
    else:
        M, sidecar = cached
        # Replay the per-layer quality from the sidecar when present;
        # otherwise leave it None — transfer still runs.
        raw_q = sidecar.get("quality_per_layer") or {}
        quality_per_layer = {int(k): float(v) for k, v in raw_q.items()}
        map_path, _ = alignment_cache_path(args.src_model, args.tgt_model)

    median_quality: float | None = None
    if quality_per_layer:
        ordered = sorted(quality_per_layer.values())
        mid = len(ordered) // 2
        if len(ordered) % 2:
            median_quality = ordered[mid]
        else:
            median_quality = 0.5 * (ordered[mid - 1] + ordered[mid])

    transferred = transfer_profile(
        src_profile, M,
        source_model_id=args.src_model,
        transfer_quality_estimate=median_quality,
    )

    # Persist the alignment-map hash on the transferred sidecar so
    # callers can detect when an old transfer is stale against a newer
    # alignment cache.
    map_hash = hash_file(map_path) if map_path.exists() else None
    save_meta: dict[str, Any] = dict(transferred.metadata)
    if map_hash is not None:
        save_meta["alignment_map_hash"] = map_hash

    transferred.save(tgt_tensor, metadata=save_meta)

    # Refresh the pack.json files map so the new variant lands in the
    # integrity check on next load.
    from saklas.io.packs import PackMetadata, hash_folder_files
    try:
        meta = PackMetadata.load(folder)
        meta.files = hash_folder_files(folder)
        meta.write(folder)
    except Exception:
        # Pack metadata refresh is best-effort — the tensor itself is
        # written, and the next ``pack ls`` will notice the discrepancy.
        pass

    payload = {
        "concept": matches[0].name,
        "namespace": matches[0].namespace,
        "source_model": args.src_model,
        "target_model": args.tgt_model,
        "tensor": str(tgt_tensor),
        "sidecar": str(tgt_sidecar),
        "transferred_layers": sorted(M.keys()),
        "median_transfer_quality": (
            round(median_quality, 4) if median_quality is not None else None
        ),
    }
    if args.json_output:
        print(_json.dumps(payload, indent=2))
        return

    quality_str = (
        f"{median_quality:.3f}" if median_quality is not None else "n/a"
    )
    print(
        f"Transferred {matches[0].namespace}/{matches[0].name} "
        f"from {args.src_model} -> {args.tgt_model}\n"
        f"  layers:           {len(M)} shared\n"
        f"  median quality:   {quality_str} (R^2 across shared layers)\n"
        f"  tensor:           {tgt_tensor}\n"
        f"  variant suffix:   :from-{safe_model_id(args.src_model)}"
    )


_VECTOR_RUNNERS = {
    "extract":  _run_extract,
    "merge":    _run_merge,
    "clone":    _run_clone,
    "compare":  _run_compare,
    "why":      _run_why,
    "transfer": _run_transfer,
}


@_saklas_error_exit
def _run_vector(args: argparse.Namespace) -> None:
    vector_cmd = getattr(args, "vector_cmd", None)
    if vector_cmd is None:
        print("usage: saklas vector <verb> [...]")
        print()
        width = max(len(v) for v, _ in _VECTOR_VERBS)
        for v, desc in _VECTOR_VERBS:
            print(f"  {v:<{width}}  {desc}")
        print()
        print("Run `saklas vector <verb> -h` for verb-specific options.")
        sys.exit(0)
    runner = _VECTOR_RUNNERS[vector_cmd]
    runner(args)


_COMMAND_RUNNERS = {
    "tui":    _run_tui,
    "serve":  _run_serve,
    "pack":   _run_pack,
    "vector": _run_vector,
    "config": _run_config,
}
