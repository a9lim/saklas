"""Runner functions for saklas CLI subcommands."""

from __future__ import annotations

import argparse
import functools
from operator import itemgetter
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, TypeVar

from saklas.cli.parsers import (
    _EXPERIMENT_VERBS, _MANIFOLD_VERBS, _PACK_VERBS, _SUBSPACE_VERBS,
    _VECTOR_VERBS,
)
from saklas.core.errors import SaklasError
from saklas.core.stats import median_or_zero

if TYPE_CHECKING:
    from saklas.core.session import SaklasSession
    from saklas.core.steering import Steering
    from saklas.core.profile import Profile


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
    if raw in (["none"], []):
        return []
    return raw


def _target_whitener_from_neutral_cache(model_id: str) -> Any | None:
    """Build a target-model whitener from neutral activations without loading a model."""
    try:
        import torch as _torch
        from safetensors.torch import load_file

        from saklas.core.mahalanobis import LayerWhitener
        from saklas.io.paths import model_dir

        acts_path = model_dir(model_id) / "neutral_activations.safetensors"
        if not acts_path.is_file():
            return None
        raw = load_file(str(acts_path))
        acts = {
            int(k.split("_", 1)[1]): v.to(_torch.float32)
            for k, v in raw.items()
        }
        means = {layer: tensor.mean(dim=0) for layer, tensor in acts.items()}
        return LayerWhitener.from_neutral_activations(acts, means)
    except Exception:
        # Best-effort: a missing, stale, or degenerate target neutral cache
        # leaves transfers on the Euclidean fallback path.
        return None


def _make_session(args: argparse.Namespace):
    from saklas.core.session import SaklasSession
    probe_categories = _resolve_probes(args.probes)
    # ``--projection-metric`` selects the runtime ``~`` / ``|`` metric
    # (``mahalanobis`` default — closed-form LEACE; ``euclidean`` is plain
    # Gram-Schmidt); ``--no-dls`` opts out of the discriminative-layer mask.
    # Both the CLI flag and YAML are already merged onto ``args`` by
    # ``_load_effective_config``.
    projection_metric = getattr(args, "projection_metric", None) or "mahalanobis"
    dls = not bool(getattr(args, "no_dls", False))
    # ``--compile`` and ``--cuda-graphs`` opt *in* to the CUDA-side
    # perf path.  Defaults are off — compile's per-token speedup is
    # variable (~1.2–3× when it works, ~equal otherwise), the
    # ~25–50s upfront cost rarely pays off on interactive workloads,
    # and torch 2.12's inductor has known codegen bugs on newer
    # architectures (Gemma-4, Qwen3.5).  YAML ``compile: true`` /
    # ``cuda_graphs: true`` are folded onto ``args.compile`` /
    # ``args.cuda_graphs`` in ``_load_effective_config``.
    compile_enabled = bool(getattr(args, "compile", False))
    cuda_graphs_enabled = bool(getattr(args, "cuda_graphs", False))
    # Phase 1 logit pass: session-level default for top-K alternatives
    # capture.  Per-call ``SamplingConfig.return_top_k > 0`` overrides;
    # K=0 inherits this value through ``_generate_core``.  argparse
    # default ``None`` falls back to 0 here (no alts).
    return_top_k = getattr(args, "top_k_alts", None) or 0
    return SaklasSession.from_pretrained(
        args.model, device=args.device, quantize=args.quantize,
        probes=probe_categories,
        system_prompt=getattr(args, "system_prompt", None),
        max_tokens=getattr(args, "max_tokens", 1024),
        projection_metric=projection_metric,
        dls=dls,
        compile=compile_enabled,
        cuda_graphs=cuda_graphs_enabled,
        return_top_k=return_top_k,
    )


def _print_model_info(session: SaklasSession) -> None:
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
    # Projection metric on tui/serve: YAML wins when the CLI flag is unset.
    if (
        composed.projection_metric is not None
        and getattr(args, "projection_metric", None) is None
    ):
        args.projection_metric = composed.projection_metric
    # YAML ``compile: true`` folds onto ``args.compile`` (the CLI
    # opt-in).  YAML ``compile: false`` is the default, so it's a
    # no-op — but accepting it makes round-tripping symmetric with
    # other knobs.  CLI flag always wins: ``--compile`` already sets
    # ``args.compile=True``, which we leave alone.
    if composed.compile is True and not bool(getattr(args, "compile", False)):
        args.compile = True
    if (
        composed.cuda_graphs is True
        and not bool(getattr(args, "cuda_graphs", False))
    ):
        args.cuda_graphs = True
    # Phase 1 logit pass: YAML ``return_top_k:`` wins when CLI
    # ``--top-k-alts`` is unset, matching the rest of the v2.1 stack.
    if (
        composed.return_top_k is not None
        and getattr(args, "top_k_alts", None) is None
    ):
        args.top_k_alts = composed.return_top_k
    ensure_vectors_installed(composed, strict=getattr(args, "strict", False))
    return composed


def _print_startup(args: argparse.Namespace) -> None:
    print(f"Loading model: {args.model}")
    if args.quantize:
        print(f"Quantization: {args.quantize}")


def _setup_steering_vectors(
    session: SaklasSession,
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


def _warmup_session(session: SaklasSession) -> None:
    """Run a stateless generation so the first real request is fast.

    The model loader's compile probe (``_compile_with_probe``) already
    specializes the compiled artifact on a 2-token prefill + 1-token
    decode, which catches compile-time failures.  This warmup is the
    layer above: it runs a realistic prompt so dynamo's automatic
    shape promotion fires on a typical-length prefill before the user
    types anything.  Without it, the first interactive prompt would
    trigger a 0.2–0.5s recompile on each new prefill length.
    """
    import time as _time
    from saklas.core.sampling import SamplingConfig
    print("Warming up generation kernels...", flush=True)
    try:
        start = _time.monotonic()
        session.generate(
            "Please respond briefly.",
            sampling=SamplingConfig(max_tokens=32),
            stateless=True,
        )
        print(f"  warmed in {_time.monotonic() - start:.1f}s")
    except Exception as e:
        print(f"  warm-up skipped: {e}")


def _attach_default_manifold_probes(session: SaklasSession) -> None:
    """Auto-attach the bundled manifold probes on dashboard startup.

    The two bundled manifolds (``personas``, ``pad``) ship as the
    default read-side counterparts to the bundled vector probes, so the
    dashboard's probe rack opens with them already watching — the
    manifold analogue of how ``bootstrap_probes`` pre-loads the default
    vector probes at session construction.

    Only manifolds already *fitted* for the loaded model are attached:
    fitting runs a forward pass per node and would block ``serve``
    startup for minutes on a fresh model, so an unfitted bundled
    manifold is skipped with a one-line hint (fit it from the dashboard
    and it auto-loads next launch).  Selector is the fully-qualified
    ``default/<name>`` so the registered probe name matches a manual
    attach from the manifolds drawer — no duplicate rows.
    """
    from saklas.io.manifolds import (
        ManifoldFolder,
        ManifoldFormatError,
        bundled_manifold_names,
    )
    from saklas.io.paths import manifold_dir, safe_model_id

    stem = safe_model_id(session.model_id)
    for name in bundled_manifold_names():
        selector = f"default/{name}"
        try:
            mf = ManifoldFolder.load(manifold_dir("default", name))
        except (ManifoldFormatError, FileNotFoundError):
            continue
        if stem not in mf.tensor_models():
            print(
                f"  manifold probe {selector}: not fitted for this model "
                "— skipping (fit it from the dashboard to auto-load)"
            )
            continue
        try:
            session.add_manifold_probe(selector)
            print(f"  manifold probe {selector}: attached")
        except SaklasError as exc:
            print(f"  manifold probe {selector}: skipped — {exc}")


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
    _warmup_session(session)

    from saklas.tui.app import SaklasApp
    app = SaklasApp(session=session)
    app.run()


@_saklas_error_exit
def _run_serve(args: argparse.Namespace) -> None:
    try:
        import fastapi  # noqa: F401
        import uvicorn
    except ImportError:
        # fastapi + uvicorn are base dependencies since v3.x; this only
        # fires when they've been uninstalled out from under saklas.
        print(
            "Server dependencies not installed. Run:\n"
            "  pip install --upgrade saklas",
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
    # Default-on: the dashboard ships with the wheel and is the easiest
    # way for casual users to drive saklas; ``--no-web`` opts out for
    # production / proxied deployments where ``/`` already belongs to
    # something else.
    web_enabled = not getattr(args, "no_web", False)
    app = create_app(session, default_steering=default_steering,
                     cors_origins=args.cors or None,
                     api_key=getattr(args, "api_key", None),
                     web=web_enabled)

    # The dashboard's probe rack opens with the bundled manifold probes
    # already watching (the read-side default, mirroring the bundled
    # vector probes).  Gated on the dashboard being mounted — with
    # ``--no-web`` there's no rack to populate.
    if web_enabled:
        _attach_default_manifold_probes(session)

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
        # The manifold leaves (fit / discover / generate / transfer) carry the
        # leaf on ``manifold_cmd`` — surface it so the error reads "manifold
        # fit" under both the top-level ``manifold`` verb and the deprecated
        # ``vector manifold`` alias. Otherwise the leaf is the subspace verb
        # (``subspace_cmd``), the deprecated ``vector_cmd``, or a ``pack_cmd``.
        manifold_cmd = getattr(args, "manifold_cmd", None)
        if manifold_cmd:
            cmd = f"manifold {manifold_cmd}"
        else:
            cmd = (
                getattr(args, "subspace_cmd", None)
                or getattr(args, "vector_cmd", None)
                or getattr(args, "pack_cmd", None)
                or "?"
            )
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
    requested_namespace = getattr(args, "namespace", None)
    if requested_namespace is not None:
        # User pinned a destination — restrict the existence check to
        # that namespace so an unrelated match elsewhere doesn't trip
        # "already extracted".
        candidate_folders = [
            c.folder for c in _all_concepts()
            if c.name == canonical and c.namespace == requested_namespace
        ]
    else:
        candidate_folders = [c.folder for c in _all_concepts() if c.name == canonical]
    candidate_folders.append(session._local_concept_folder(
        canonical, namespace=requested_namespace or "local",
    ))
    requested_release = getattr(args, "sae", None)
    requested_role = getattr(args, "role", None)
    if requested_release and requested_role:
        print(
            "extract: --sae and --role are mutually exclusive "
            "(role substitution composes via the tensor filename but not "
            "with SAE feature space)",
            file=sys.stderr,
        )
        sys.exit(2)
    candidate_tensor_name = tensor_filename(
        session.model_id,
        release=requested_release,
        role=requested_role,
    )
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

    extract_kwargs: dict[str, Any] = {}
    if getattr(args, "sae", None):
        extract_kwargs["sae"] = args.sae
    if getattr(args, "sae_revision", None):
        extract_kwargs["sae_revision"] = args.sae_revision
    if requested_role:
        extract_kwargs["role"] = requested_role
    if requested_namespace is not None:
        extract_kwargs["namespace"] = requested_namespace
    if args.force:
        # Webui parity: ``--force`` on the CLI matches the
        # ExtractDrawer's "overwrite an existing vector" checkbox —
        # pre-deleted tensors above force a tensor-cache miss; passing
        # ``force_statements=True`` here additionally bypasses the
        # statements-cache reuse so the regenerate is end-to-end.
        extract_kwargs["force_statements"] = True

    try:
        if baseline is not None:
            canonical, _profile = session.extract(raw, baseline=baseline, **extract_kwargs)
        else:
            canonical, _profile = session.extract(raw, **extract_kwargs)
    except Exception as e:
        print(f"extract failed: {e}", file=sys.stderr)
        sys.exit(1)

    # `canonical` may carry a trailing variant suffix from the engine —
    # ``:sae-<release>`` when ``sae=`` was passed, ``:role-<slug>`` when
    # ``role=`` was.  Peel either for tensor-path construction; bare
    # ``canonical`` otherwise.  ``role`` and ``sae`` are mutually
    # exclusive at the flag layer above, so we never see both.
    if ":sae-" in canonical:
        core_name, _, rel = canonical.partition(":sae-")
        tensor_name = tensor_filename(session.model_id, release=rel)
    elif ":role-" in canonical:
        core_name, _, role_slug = canonical.partition(":role-")
        tensor_name = tensor_filename(session.model_id, role=role_slug)
    else:
        core_name = canonical
        tensor_name = tensor_filename(session.model_id, release=None)
    final_paths = [pathlib.Path(f) / tensor_name for f in candidate_folders]
    final_path = next((p for p in final_paths if p.exists()), None)
    if final_path is None:
        final_path = (
            pathlib.Path(session._local_concept_folder(
                core_name, namespace=requested_namespace or "local",
            )) / tensor_name
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


_VARIANT_SUFFIX_RE = re.compile(
    r"^(raw|sae(?:-[a-z0-9._-]+)?|"
    r"role(?:-[a-z0-9._-]+)?|from(?:-[a-z0-9._-]+)?)$"
)


def _split_variant_suffix(raw: str) -> tuple[str, str | None]:
    """Peel a trailing ``:<variant>`` off a selector string.

    Returns ``(name_part, variant_or_None)``. ``variant`` is one of the
    tensor suffixes understood by :func:`saklas.io.packs.enumerate_variants`.
    Non-variant colon usage (``tag:``, ``namespace:``, ``model:``) passes
    through unchanged with ``variant=None`` — those prefixes are caught by
    ``sel.parse`` later.
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
    folder: Path,
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
      - ``"role"`` / ``"from"``: require the unique role / transferred
        variant, with the same ambiguity behavior.
      - ``"<kind>-<id>"``: require that specific variant.
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

    if variant in {"sae", "role", "from"}:
        prefix = f"{variant}-"
        display = "SAE" if variant == "sae" else variant
        id_hint = "release" if variant == "sae" else "id"
        kind_paths = {k: v for k, v in variants.items() if k.startswith(prefix)}
        if len(kind_paths) == 0:
            raise UnknownVariantError(
                f"no {display} variants found in {folder.name} for model {model_id} "
                f"(available: {sorted(variants) or 'none'})"
            )
        if len(kind_paths) > 1:
            raise AmbiguousVariantError(
                f"{folder.name}: multiple {display} variants for model {model_id}: "
                f"{sorted(kind_paths)}. Specify with :{variant}-<{id_hint}>."
            )
        return next(iter(kind_paths.values()))

    # Specific variant key (``sae-<release>``, ``role-<slug>``,
    # ``from-<safe_src>``).
    path = variants.get(variant)
    if path is None:
        raise UnknownVariantError(
            f"variant '{variant}' not found in {folder.name} for model "
            f"{model_id} (available: {sorted(variants) or 'none'})"
        )
    return path


def _fold_manifold_to_profile(
    name: str, model_id: str, variant: str | None,
) -> "Profile | None":
    """Fold a fitted 2-node ``pca`` manifold for ``name`` into a vector Profile.

    The 4.0 fallback for the disk-side inspection verbs (``compare`` / ``why``):
    bundled and user concepts ship as 2-node manifolds, so when no ``vectors/``
    tensor resolves, a **fitted** manifold tensor is loaded and folded
    (:func:`~saklas.core.vectors.folded_vector_directions`).  Returns ``None``
    when no fitted manifold tensor exists for ``model_id`` (caller surfaces a
    "fit it first" miss) or the manifold isn't a foldable 2-node affine.  Only
    the ``raw`` / ``sae-<release>`` variants fold; role / transfer variants are
    vector-only.  Raises :class:`AmbiguousSelectorError` on a cross-namespace
    bare-name collision (mirroring vector resolution).
    """
    folded = _fold_manifold_to_profile_with_identity(name, model_id, variant)
    return folded[0] if folded is not None else None


def _fold_manifold_to_profile_with_identity(
    name: str, model_id: str, variant: str | None,
) -> "tuple[Profile, str, str] | None":
    """Fold a fitted 2-node ``pca`` manifold and return its namespace identity.

    The identity is needed by ``compare``'s rank-all mode: bundled and user
    concepts can share a bare name across namespaces, so callers must key
    scanned profiles by ``namespace/name`` rather than letting later matches
    overwrite earlier ones.
    """
    from saklas.core.profile import Profile
    from saklas.core.manifold import load_manifold
    from saklas.core.vectors import folded_vector_directions
    from saklas.io.paths import manifold_dir, manifolds_dir, tensor_filename
    from saklas.io.selectors import AmbiguousSelectorError

    if variant in (None, "raw"):
        release: str | None = None
    elif variant.startswith("sae-"):
        release = variant[len("sae-"):]
    else:
        return None

    if "/" in name:
        ns, bare = name.split("/", 1)
        search_ns = [ns]
    else:
        bare = name
        root = manifolds_dir()
        search_ns = (
            sorted(d.name for d in root.iterdir() if d.is_dir())
            if root.exists() else []
        )
    fname = tensor_filename(model_id, release=release)
    hits = [
        manifold_dir(ns, bare) / fname
        for ns in search_ns
        if (manifold_dir(ns, bare) / fname).exists()
    ]
    if not hits:
        return None
    if len(hits) > 1:
        qualified = ", ".join(f"{p.parent.parent.name}/{bare}" for p in hits)
        raise AmbiguousSelectorError(
            f"ambiguous manifold '{bare}': matches {qualified}. "
            f"Qualify it with a namespace."
        )
    try:
        manifold = load_manifold(hits[0])
        dirs = folded_vector_directions(manifold)
    except Exception:
        return None
    ns = hits[0].parent.parent.name
    return Profile(dirs), ns, bare


def _fold_all_fitted_manifolds(
    model_id: str, *, exclude_identity: tuple[str, str] | None = None,
) -> "dict[str, Profile]":
    """Fold every fitted 2-node ``pca`` manifold into ``{ns/name: Profile}``.

    Backs ``compare``'s 1-arg "rank against all installed" mode now that
    bundled concepts are manifolds.  Only manifolds with a fitted raw tensor
    for ``model_id`` that fold to a 2-node affine are included;
    ``exclude_identity`` drops the target by exact ``(namespace, name)``.
    """
    from saklas.core.profile import Profile
    from saklas.core.manifold import load_manifold
    from saklas.core.vectors import folded_vector_directions
    from saklas.io.paths import manifolds_dir, tensor_filename

    out: dict[str, Profile] = {}
    root = manifolds_dir()
    if not root.is_dir():
        return out
    fname = tensor_filename(model_id, release=None)
    for ns_dir in sorted(root.iterdir()):
        if not ns_dir.is_dir():
            continue
        for mdir in sorted(ns_dir.iterdir()):
            if not mdir.is_dir():
                continue
            identity = (ns_dir.name, mdir.name)
            if identity == exclude_identity:
                continue
            tensor = mdir / fname
            if not tensor.is_file():
                continue
            try:
                dirs = folded_vector_directions(load_manifold(tensor))
            except Exception:
                continue  # not a foldable 2-node affine (curved / rank > 1)
            out[f"{ns_dir.name}/{mdir.name}"] = Profile(dirs)
    return out


def _run_compare(args: argparse.Namespace) -> None:
    import json as _json
    from saklas.io.selectors import parse as sel_parse, resolve
    from saklas.core.errors import AmbiguousVariantError, UnknownVariantError
    from saklas.io.paths import vectors_dir
    from saklas.core.profile import Profile, ProfileError

    # Default metric is ``"mahalanobis"`` (since v2.1); ``--metric euclidean``
    # selects plain weighted cosine.
    metric = getattr(args, "metric", None) or "mahalanobis"

    # Mahalanobis path: load the per-model whitener once up front, share
    # across every ``cosine_similarity`` call below.  Failure is fatal —
    # if the user explicitly asked for the whitened metric, falling
    # silently back to Euclidean would hide the missing cache.
    whitener: "Any | None" = None
    if metric == "mahalanobis":
        from saklas.core.mahalanobis import LayerWhitener, WhitenerError

        try:
            whitener = LayerWhitener.from_cache(
                args.model,
                ridge_scale=getattr(args, "ridge_scale", 1.0),
            )
        except WhitenerError as e:
            print(f"compare: {e}", file=sys.stderr)
            sys.exit(1)

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
            names.extend((f"{c.namespace}/{c.name}", variant) for c in resolved)

    # Load profiles from disk.
    profiles: dict[str, Profile] = {}
    profile_identities: dict[str, tuple[str, str]] = {}
    for name, variant in names:
        sel = sel_parse(name)
        matches = resolve(sel)
        # Vector tensor first; fall back to folding a fitted manifold (bundled
        # & user concepts ship as 2-node manifolds).
        loaded: Profile | None = None
        identity: tuple[str, str] | None = None
        display = name if variant is None else f"{name}:{variant}"
        if matches:
            match = matches[0]
            folder = match.folder
            identity = (match.namespace, match.name)
            display_base = (
                f"{match.namespace}/{match.name}" if "/" in name else match.name
            )
            display = display_base if variant is None else f"{display_base}:{variant}"
            try:
                tensor_path = _resolve_variant_tensor(folder, args.model, variant)
            except (AmbiguousVariantError, UnknownVariantError) as e:
                print(f"warning: {e}, skipping", file=sys.stderr)
                continue
            if tensor_path is not None and tensor_path.is_file():
                try:
                    loaded = Profile.load(tensor_path)
                except (ProfileError, Exception) as e:
                    print(f"warning: failed to load '{name}': {e}", file=sys.stderr)
                    continue
        if loaded is None:
            try:
                folded = _fold_manifold_to_profile_with_identity(
                    name, args.model, variant,
                )
            except Exception as e:
                print(f"warning: {e}, skipping", file=sys.stderr)
                continue
            if folded is not None:
                loaded, ns, bare = folded
                identity = (ns, bare)
                display_base = f"{ns}/{bare}" if "/" in name else bare
                display = display_base if variant is None else f"{display_base}:{variant}"
        if loaded is None:
            print(
                f"warning: no vector tensor or fitted manifold for '{name}' "
                f"with model {args.model}, skipping",
                file=sys.stderr,
            )
            continue
        profiles[display] = loaded
        if identity is not None:
            profile_identities[display] = identity

    if len(profiles) < 1:
        print("compare: no loadable profiles found", file=sys.stderr)
        sys.exit(1)

    ordered = list(profiles.keys())

    # 1-arg mode: rank all installed against the target.
    if len(args.concepts) == 1 and len(ordered) == 1:
        target_name = ordered[0]
        target = profiles[target_name]
        target_identity = profile_identities.get(target_name)

        # Load all other installed profiles for this model. Keep the exact
        # namespace identity internally, then choose display names after the
        # full pool is known: bare for unique names, qualified for collisions.
        other_entries: dict[tuple[str, str], Profile] = {}
        vdir = vectors_dir()
        if vdir.is_dir():
            for ns_dir in sorted(vdir.iterdir()):
                if not ns_dir.is_dir():
                    continue
                for cdir in sorted(ns_dir.iterdir()):
                    if not cdir.is_dir():
                        continue
                    identity = (ns_dir.name, cdir.name)
                    if identity == target_identity:
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
                        other_entries[identity] = Profile.load(tp)
                    except Exception:
                        continue

        # Bundled & user concepts ship as manifolds — fold every fitted one
        # into the ranking pool (vector tensors above win a name collision).
        for mname, mprof in _fold_all_fitted_manifolds(
            args.model, exclude_identity=target_identity,
        ).items():
            ns, bare = mname.split("/", 1)
            other_entries.setdefault((ns, bare), mprof)

        from collections import Counter

        target_bare = (
            target_identity[1] if target_identity is not None
            else target_name.rsplit("/", 1)[-1]
        )
        bare_counts = Counter(name for _ns, name in other_entries)
        others: dict[str, Profile] = {}
        for (ns, bare), profile in sorted(other_entries.items()):
            display = (
                f"{ns}/{bare}"
                if bare_counts[bare] > 1 or bare == target_bare
                else bare
            )
            others[display] = profile

        if not others:
            print(f"compare: no other profiles found for model {args.model}", file=sys.stderr)
            sys.exit(1)

        scores: dict[str, float] = {
            name: target.cosine_similarity(p, whitener=whitener)
            for name, p in others.items()
        }
        ranked = sorted(scores.items(), key=itemgetter(1), reverse=True)

        if args.json_output:
            result: dict[str, Any] = {"target": target_name, "model": args.model,
                            "similarities": [{"name": n, "similarity": round(s, 6)}
                                              for n, s in ranked]}
            if args.verbose:
                top3 = ranked[:3]
                result["per_layer_top3"] = {
                    n: {str(k): round(v, 6)
                        for k, v in target.cosine_similarity(others[n], per_layer=True, whitener=whitener).items()}
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
                    per_layer_top3: dict[int, float] = target.cosine_similarity(others[name], per_layer=True, whitener=whitener)
                    print(f"    {name}:")
                    for layer in sorted(per_layer_top3):
                        print(f"      layer {layer:>3}: {per_layer_top3[layer]:+.4f}")
        return

    if len(ordered) < 2:
        print("compare: need at least 2 profiles to compare", file=sys.stderr)
        sys.exit(1)

    # 2-arg mode: pairwise.
    if len(ordered) == 2:
        a_name, b_name = ordered
        a, b = profiles[a_name], profiles[b_name]
        sim: float = a.cosine_similarity(b, whitener=whitener)

        if args.json_output:
            result = {"a": a_name, "b": b_name, "model": args.model,
                      "similarity": round(sim, 6)}
            if args.verbose:
                result["per_layer"] = {str(k): round(v, 6)
                                       for k, v in a.cosine_similarity(b, per_layer=True, whitener=whitener).items()}
            print(_json.dumps(result, indent=2))
        else:
            print(f"{a_name} ~ {b_name}: {sim:+.4f}")
            if args.verbose:
                per_layer_2: dict[int, float] = a.cosine_similarity(b, per_layer=True, whitener=whitener)
                for layer in sorted(per_layer_2):
                    print(f"  layer {layer:>3}: {per_layer_2[layer]:+.4f}")
        return

    # 3+ mode: N×N matrix.
    matrix: dict[str, dict[str, float]] = {}
    for a_name in ordered:
        matrix[a_name] = {}
        for b_name in ordered:
            if a_name == b_name:
                matrix[a_name][b_name] = 1.0
            else:
                matrix[a_name][b_name] = profiles[a_name].cosine_similarity(profiles[b_name], whitener=whitener)

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
                            profiles[b_name], per_layer=True, whitener=whitener,
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

    # Vector tensor first; fall back to folding a fitted 2-node manifold
    # (bundled & user concepts ship as manifolds).
    profile: "Profile | None" = None
    concept_name = name_part if variant is None else f"{name_part}:{variant}"
    if matches:
        match = matches[0]
        folder = match.folder
        concept_base = (
            f"{match.namespace}/{match.name}" if "/" in name_part else match.name
        )
        concept_name = concept_base if variant is None else f"{concept_base}:{variant}"
        try:
            tensor_path = _resolve_variant_tensor(folder, args.model, variant)
        except (AmbiguousVariantError, UnknownVariantError) as e:
            print(f"why: {e}", file=sys.stderr)
            sys.exit(1)
        if tensor_path is not None and tensor_path.is_file():
            try:
                profile = Profile.load(tensor_path)
            except (ProfileError, Exception) as e:
                print(f"why: failed to load '{args.concept}': {e}", file=sys.stderr)
                sys.exit(1)
    if profile is None:
        folded = _fold_manifold_to_profile_with_identity(
            name_part, args.model, variant,
        )
        if folded is not None:
            profile, ns, bare = folded
            concept_base = f"{ns}/{bare}" if "/" in name_part else bare
            concept_name = concept_base if variant is None else f"{concept_base}:{variant}"
    if profile is None:
        print(
            f"why: no vector tensor or fitted manifold for '{args.concept}' "
            f"with model {args.model}. If it's a manifold, fit it first: "
            f"`saklas manifold fit {name_part} -m {args.model}`.",
            file=sys.stderr,
        )
        sys.exit(1)

    layer_mags: list[tuple[int, float]] = sorted(
        ((layer, float(tensor.norm().item())) for layer, tensor in profile.items()),
        key=itemgetter(0),
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

    med_evr = median_or_zero(evrs)
    med_intra = median_or_zero(intras)
    med_align = median_or_zero(aligns)
    med_proj = median_or_zero(projs)

    if (med_evr > 0.95 and med_intra < 0.01) or med_align < 0.2:
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
        median_quality = (
            ordered[mid]
            if len(ordered) % 2
            else 0.5 * (ordered[mid - 1] + ordered[mid])
        )

    # Build the target-model whitener so the transferred share is re-baked
    # in the *target* metric (else it inherits the source model's
    # anisotropy).  Both the fit and cached-alignment paths leave the
    # target neutral-activation cache on disk (the alignment was computed
    # from it), so read it directly and center by its own per-layer mean —
    # the neutral mean *is* the probe-centering baseline, so this needs no
    # layer_means cache.  Soft ``None`` on any miss → ``transfer_profile``
    # leaves the Euclidean (source-share) bake untouched.
    target_whitener = _target_whitener_from_neutral_cache(args.tgt_model)

    transferred = transfer_profile(
        src_profile, M,
        source_model_id=args.src_model,
        transfer_quality_estimate=median_quality,
        whitener=target_whitener,
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


def _run_manifold_fit(args: argparse.Namespace) -> None:
    from saklas.io.manifolds import domain_label

    _require_model(args)
    folder = Path(args.folder)
    if not (folder / "manifold.json").exists():
        print(
            f"manifold fit: no manifold.json in {folder}", file=sys.stderr,
        )
        sys.exit(2)
    _print_startup(args)
    session = _make_session(args)
    _print_model_info(session)
    try:
        manifold = session.extract_manifold(
            folder,
            sae=getattr(args, "sae", None),
            sae_revision=getattr(args, "sae_revision", None),
            on_progress=lambda m: print(f"  {m}"),
        )
    except Exception as e:
        print(f"manifold fit failed: {e}", file=sys.stderr)
        sys.exit(1)
    print(
        f"fitted manifold '{manifold.name}' "
        f"({len(manifold.layers)} layers, {len(manifold.node_labels)} nodes, "
        f"{domain_label(manifold.domain.to_spec())}, "
        f"{manifold.feature_space})"
    )


def _iter_manifold_folders(namespace: str | None):
    """Yield ``(namespace, ManifoldFolder)`` for every installed manifold.

    Thin wrapper over :func:`saklas.io.manifolds.iter_manifold_folders` —
    the discovery walk lives in ``io`` so the server shares it without
    importing ``cli``.

    Materializes bundled manifolds (e.g. ``default/personas``) before
    the walk.  CLI verbs like ``manifold discover`` / ``show`` / ``ls``
    resolve folders *before* any session is constructed, so the session-
    startup ``materialize_bundled_manifolds`` call hasn't fired yet —
    without this hop, a user's first ``saklas manifold discover
    default/personas`` would miss the bundled folder and exit with "not
    found".  The format-version short-circuit inside the materialize
    function makes repeated calls cheap (no-op when up-to-date).  The
    server has its own session-driven materialization at startup, so
    going through ``iter_manifold_folders`` directly stays correct.
    """
    from saklas.io.manifolds import (
        iter_manifold_folders, materialize_bundled_manifolds,
    )

    materialize_bundled_manifolds()
    yield from iter_manifold_folders(namespace)


def _run_manifold_ls(args: argparse.Namespace) -> None:
    import json as _json
    from saklas.io.manifolds import domain_label

    rows = list(_iter_manifold_folders(getattr(args, "namespace", None)))
    if getattr(args, "json_output", False):
        print(_json.dumps([
            {
                "namespace": ns,
                "name": mf.name,
                "domain": mf.domain,
                "nodes": mf.node_labels,
                "fitted_models": mf.tensor_models(),
            }
            for ns, mf in rows
        ], indent=2))
        return
    if not rows:
        print("no manifolds installed under ~/.saklas/manifolds/")
        return
    verbose = getattr(args, "verbose", False)
    for ns, mf in rows:
        kind = domain_label(mf.domain)
        models = ", ".join(mf.tensor_models()) or "(unfitted)"
        print(
            f"  {ns}/{mf.name}  [{kind}, {len(mf.node_labels)} nodes]  {models}"
        )
        if verbose and mf.description:
            print(f"    {mf.description}")


def _run_manifold_show(args: argparse.Namespace) -> None:
    import json as _json
    from saklas.io.manifolds import domain_label

    name = args.name
    target_ns = None
    if "/" in name:
        target_ns, name = name.split("/", 1)
    matches = [
        (ns, mf)
        for ns, mf in _iter_manifold_folders(target_ns)
        if mf.name == name
    ]
    if not matches:
        print(f"manifold '{args.name}' not found", file=sys.stderr)
        sys.exit(1)
    if len(matches) > 1:
        qualified = ", ".join(f"{ns}/{mf.name}" for ns, mf in matches)
        print(
            f"ambiguous manifold '{name}': matches {qualified}; "
            f"qualify with a namespace",
            file=sys.stderr,
        )
        sys.exit(2)
    ns, mf = matches[0]
    fitted = []
    for stem in mf.tensor_models():
        sc = mf.sidecar(stem)
        entry: dict[str, Any] = {
            "stem": stem,
            "method": sc.method,
            "feature_space": sc.feature_space,
            "node_count": sc.node_count,
            "fit_mode": sc.fit_mode,
        }
        if sc.hyperparams:
            entry["hyperparams"] = sc.hyperparams
        if sc.diagnostics:
            entry["diagnostics"] = sc.diagnostics
        fitted.append(entry)

    # In discover mode the folder itself has no per-node coords (they are
    # derived per-model at fit time) — try to surface the derived coords
    # from the per-model tensor when a fit exists.  Cheap: one safetensors
    # header read.  Soft-fails to an empty list if the tensor isn't there.
    derived_node_coords: list[list[float]] = []
    if mf.is_discover and fitted:
        from saklas.core.manifold import load_manifold
        stem_for_session = fitted[0]["stem"]
        try:
            m = load_manifold(mf.tensor_path(stem_for_session))
            derived_node_coords = [
                [float(x) for x in row]
                for row in m.node_coords.tolist()
            ]
        except (FileNotFoundError, KeyError, ValueError):
            derived_node_coords = []

    # Per-node roles ride alongside coords in the JSON shape; ``None``
    # for a given node means "pooled under the standard assistant
    # baseline" (the legacy / non-role default).  Padded to the label
    # count so consumers can index by node.
    node_roles_padded = list(mf.node_roles) + [None] * (
        len(mf.node_labels) - len(mf.node_roles)
    )
    if getattr(args, "json_output", False):
        # Share the session-independent summary serializer with the
        # server's ``GET /saklas/v1/manifolds/{ns}/{name}`` route so both
        # surfaces emit the same keys.  For a discover folder
        # ``manifold_summary`` reports the on-disk (empty) geometry —
        # ``node_coords == []`` — since the derived per-model layout lives
        # in the fitted safetensors; the text path below still surfaces
        # the derived coords for interactive use.
        from saklas.io.manifolds import manifold_summary
        print(_json.dumps(manifold_summary(mf.folder), indent=2))
        return

    print(f"{ns}/{mf.name}")
    if mf.description:
        print(f"  {mf.description}")
    if mf.is_discover:
        print(f"  fit_mode: {mf.fit_mode} (discover — coords derived per-model)")
        if mf.hyperparams:
            hp = ", ".join(f"{k}={v}" for k, v in sorted(mf.hyperparams.items()))
            print(f"  hyperparams: {hp}")
        print("  nodes:")
        for i, label in enumerate(mf.node_labels):
            role_tail = (
                f"  [role={node_roles_padded[i]}]"
                if node_roles_padded[i] else ""
            )
            if i < len(derived_node_coords):
                coord_str = ", ".join(f"{c:.3g}" for c in derived_node_coords[i])
                print(f"    {label}  ({coord_str}){role_tail}")
            else:
                print(f"    {label}  (coords pending fit){role_tail}")
    else:
        print(f"  domain: {domain_label(mf.domain)}")
        print("  nodes:")
        for label, coords, role in zip(
            mf.node_labels, mf.node_coords, node_roles_padded,
        ):
            coord_str = ", ".join(f"{c:g}" for c in coords)
            role_tail = f"  [role={role}]" if role else ""
            print(f"    {label}  ({coord_str}){role_tail}")
    if fitted:
        print("  fitted models:")
        for f in fitted:
            tag = f"{f['method']}, {f['feature_space']}"
            if f.get("fit_mode") and f["fit_mode"] != "authored":
                tag += f", fit_mode={f['fit_mode']}"
            print(f"    {f['stem']}  ({tag})")
            diag = f.get("diagnostics") or {}
            # Per-method one-line summary.  PCA: picked_k + cumvar at k.
            # Spectral: picked_k + gap_magnitude + bandwidth + k_nn.
            if diag and f.get("fit_mode") == "pca":
                cumvar = diag.get("cumulative_variance") or []
                k = diag.get("picked_k")
                if k and 1 <= k <= len(cumvar):
                    print(
                        f"      pca: picked_k={k}, "
                        f"cumvar@k={cumvar[k - 1]:.3f} "
                        f"(threshold={diag.get('threshold', '?')})"
                    )
            elif diag and f.get("fit_mode") == "spectral":
                k = diag.get("picked_k")
                gap = diag.get("gap_magnitude")
                bw = diag.get("bandwidth")
                knn = diag.get("k_nn")
                bits = [f"picked_k={k}"]
                if gap is not None:
                    bits.append(f"gap={gap:.3g}")
                if bw is not None:
                    bits.append(f"bandwidth={bw:.3g}")
                if knn is not None:
                    bits.append(f"k_nn={knn}")
                print(f"      spectral: {', '.join(bits)}")
            if f.get("hyperparams"):
                hp = ", ".join(
                    f"{k}={v}" for k, v in sorted(f["hyperparams"].items())
                )
                print(f"      hyperparams: {hp}")
    else:
        hint = (
            "discover" if mf.is_discover else "fit"
        )
        print(f"  fitted models: (none — run `saklas manifold {hint}`)")


def _resolve_manifold_ns_name(name: str) -> tuple[str, str]:
    """Resolve a CLI-supplied ``NAME`` (or ``ns/name``) to ``(namespace, name)``.

    Mirrors :func:`_run_manifold_show`'s ambiguity handling — bare names
    resolve cross-namespace (reaching e.g. bundled ``default/`` when that's
    the only match) and raise on collision; an ``ns/name`` form pins to a
    single namespace.  Exits with a clear error on miss / ambiguity.

    This is the lifecycle analogue of the concept-selector cross-namespace
    resolution: ``clear`` / ``refresh`` / ``rm`` / ``transfer`` (and the
    folder-returning ``discover`` / ``show``) all route a bare name through
    here so it can reach any namespace, not just ``local/``.  (``generate``
    deliberately does *not* — it authors a fresh folder and defaults bare →
    ``local/`` via :func:`_split_manifold_ns_name`.)

    An explicit ``ns/name`` pins directly — it's returned verbatim without a
    filesystem walk, leaving the existence check to the io backend (which
    raises ``FileNotFoundError``).  Only a *bare* name walks the installed
    manifolds to discover its namespace, raising on collision / miss.
    """
    if "/" in name:
        ns, leaf = name.split("/", 1)
        return ns, leaf
    matches = [
        (ns, mf)
        for ns, mf in _iter_manifold_folders(None)
        if mf.name == name
    ]
    if not matches:
        print(f"manifold '{name}' not found", file=sys.stderr)
        sys.exit(1)
    if len(matches) > 1:
        qualified = ", ".join(f"{ns}/{mf.name}" for ns, mf in matches)
        print(
            f"ambiguous manifold '{name}': matches {qualified}; "
            f"qualify with a namespace",
            file=sys.stderr,
        )
        sys.exit(2)
    return matches[0][0], matches[0][1].name


def _resolve_manifold_folder(name: str) -> Path:
    """Resolve a CLI-supplied ``NAME`` (or ``ns/name``) to a folder path.

    Thin wrapper over :func:`_resolve_manifold_ns_name` for the verbs that
    want the folder directly (``discover`` / ``show``); the lifecycle verbs
    that hand a ``(namespace, name)`` pair to their io backend call the
    pair-returning form instead.
    """
    from saklas.io.paths import manifold_dir

    ns, resolved = _resolve_manifold_ns_name(name)
    return manifold_dir(ns, resolved)


def _run_manifold_discover(args: argparse.Namespace) -> None:
    from saklas.io.manifolds import domain_label

    """``saklas manifold discover`` — fit a discover-mode manifold.

    The folder must already exist (usually created by ``generate``) and
    be in discover shape.  Any CLI overrides (``--method``, ``--max-dim``,
    ``--var-threshold``, ``--k-nn``, ``--bandwidth``) are written back
    to ``manifold.json`` *before* the fit so the cache key reflects the
    actual fit inputs.  This means a subsequent fit with different
    overrides reliably misses cache; the folder's manifest is the
    single source of truth for "what hyperparams produced the cached
    tensor."
    """
    import json as _json
    from saklas.io.atomic import write_json_atomic
    from saklas.io.manifolds import (
        ManifoldFolder, ManifoldFormatError, _sanitize_hyperparams,
    )

    _require_model(args)
    folder = _resolve_manifold_folder(args.name)

    # Validate folder is discover-shape before paying for model load.
    try:
        mf = ManifoldFolder.load(folder)
    except ManifoldFormatError as e:
        print(f"manifold discover: {e}", file=sys.stderr)
        sys.exit(2)
    if not mf.is_discover:
        print(
            f"manifold discover: '{args.name}' is authored, not discover; "
            f"use `saklas manifold fit` for authored manifolds",
            file=sys.stderr,
        )
        sys.exit(2)

    # Compose the effective fit_mode + hyperparams from CLI overrides.
    new_fit_mode = args.method or mf.fit_mode
    if new_fit_mode not in {"pca", "spectral"}:
        print(
            f"manifold discover: invalid method {new_fit_mode!r}",
            file=sys.stderr,
        )
        sys.exit(2)
    new_hyperparams = dict(mf.hyperparams)
    # PCA knobs.
    if args.max_dim is not None:
        new_hyperparams["max_dim"] = int(args.max_dim)
    if args.var_threshold is not None:
        new_hyperparams["var_threshold"] = float(args.var_threshold)
    # Spectral knobs.
    if args.k_nn is not None:
        new_hyperparams["k_nn"] = int(args.k_nn)
    if args.bandwidth is not None:
        new_hyperparams["bandwidth"] = float(args.bandwidth)
    # Shared knob (applies to both PCA and spectral fits).
    if getattr(args, "max_subspace_dim", None) is not None:
        new_hyperparams["max_subspace_dim"] = int(args.max_subspace_dim)
    # Cross-method knobs that don't apply get dropped — silently
    # carrying e.g. ``var_threshold`` into a spectral fit pollutes the
    # cache key without affecting the fit, and an out-of-whitelist key
    # would crash the dispatcher at fit time.  Single source of truth
    # lives in ``io.manifolds._sanitize_hyperparams``.
    new_hyperparams = _sanitize_hyperparams(new_fit_mode, new_hyperparams)

    # Write back if anything changed.  Staged write — a crash mid-rewrite
    # would leave the folder unreadable and ``ManifoldFolder.load`` would
    # 400 on the next list call.
    if new_fit_mode != mf.fit_mode or new_hyperparams != mf.hyperparams:
        data = _json.loads((folder / "manifold.json").read_text())
        data["fit_mode"] = new_fit_mode
        data["hyperparams"] = new_hyperparams
        # Re-author the nodes list in case fit_mode changed and it
        # accidentally carries authored shape.  Discover nodes are
        # label-only.
        data["nodes"] = [{"label": label} for label in mf.node_labels]
        data.pop("domain", None)
        write_json_atomic(folder / "manifold.json", data)

    _print_startup(args)
    session = _make_session(args)
    _print_model_info(session)
    try:
        manifold = session.extract_manifold(
            folder,
            sae=getattr(args, "sae", None),
            sae_revision=getattr(args, "sae_revision", None),
            on_progress=lambda m: print(f"  {m}"),
        )
    except (ValueError, ManifoldFormatError) as e:
        print(f"manifold discover failed: {e}", file=sys.stderr)
        sys.exit(1)
    print(
        f"discovered manifold '{manifold.name}' "
        f"({len(manifold.layers)} layers, "
        f"{len(manifold.node_labels)} nodes, "
        f"{domain_label(manifold.domain.to_spec())}, "
        f"{manifold.feature_space}, "
        f"fit_mode={new_fit_mode})"
    )


def _run_manifold_generate(args: argparse.Namespace) -> None:
    """``saklas manifold generate`` — author + LLM-fill a discover folder.

    Two-step end-to-end: load the model, run
    :meth:`SaklasSession.generate_statements` to fill per-concept
    corpora with the anti-allegory K-tuple generator, then write the
    folder via :func:`create_discover_manifold_folder`.  The folder is
    ready for ``manifold discover`` to fit; the two-step split
    keeps the failure modes legible (a flaky generation run leaves
    inspectable corpora) and lets the user review the statements before
    paying for the discover fit.
    """
    from saklas.io.atomic import write_json_atomic
    from saklas.io.manifolds import (
        ManifoldFormatError,
        append_discover_manifold_node,
        plan_discover_generation,
    )
    from saklas.io.paths import manifold_dir

    _require_model(args)
    if len(args.concepts) < 2:
        print(
            "manifold generate: need >= 2 concepts (shared-scenario "
            "structure is meaningless with one)",
            file=sys.stderr,
        )
        sys.exit(2)

    namespace = "local"
    name = args.name
    if "/" in name:
        namespace, name = name.split("/", 1)

    folder = manifold_dir(namespace, name)
    # ``-f/--force`` is a clean slate; the default *resumes* — fills any
    # missing node corpora and appends concepts new to the roster (locked
    # onto the saved scenarios), so a run killed half-way picks up where
    # it left off and adding a node is a plain re-run.
    if args.force and (folder / "manifold.json").exists():
        import shutil
        shutil.rmtree(folder)

    # ``--role-per-node``: the concept slug doubles as that node's
    # assistant-role substitution at fit time (engine validates the slug;
    # an unsupported family raises later at the matching discover/fit call).
    node_roles: dict[str, str | None] | None = None
    if getattr(args, "role_per_node", False):
        node_roles = {concept: concept for concept in args.concepts}

    # Plan first — no model load needed to learn there is nothing to do.
    try:
        plan = plan_discover_generation(
            folder, name, args.description,
            fit_mode="pca", labels=list(args.concepts), node_roles=node_roles,
        )
    except ManifoldFormatError as e:
        print(f"manifold generate failed: {e}", file=sys.stderr)
        sys.exit(1)

    n_total = len(plan.index_of)
    if not plan.pending:
        print(
            f"manifold {namespace}/{name} already complete ({n_total} nodes)"
            f" — pass -f/--force to regenerate"
        )
        return
    if plan.resumed and len(plan.pending) < n_total:
        print(
            f"resuming {namespace}/{name}: "
            f"{n_total - len(plan.pending)}/{n_total} nodes present, "
            f"generating {len(plan.pending)}"
        )
    if plan.added:
        print(f"adding {len(plan.added)} new node(s): {', '.join(plan.added)}")

    _print_startup(args)
    session = _make_session(args)
    _print_model_info(session)

    print(
        f"generating {args.n_scenarios} scenarios + "
        f"{args.statements_per_concept} statements per (scenario × concept) "
        f"cell for {len(plan.pending)} concept(s)..."
    )
    captured_scenarios: list[str] = []
    try:
        session.generate_statements(
            list(plan.pending),
            scenarios=list(plan.scenarios) if plan.scenarios is not None else None,
            n_scenarios=args.n_scenarios,
            statements_per_cell=args.statements_per_concept,
            on_progress=lambda m: print(f"  {m}"),
            on_scenarios=captured_scenarios.extend,
            on_corpus=lambda label, stmts: append_discover_manifold_node(
                folder, plan.index_of[label], label, stmts,
            ),
        )
    except ValueError as e:
        print(f"manifold generate failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Record scenario provenance on the first run; a resume keeps the
    # already-locked ``scenarios.json``.  The canonical ``"scenarios"``
    # key (vector-pipeline convention) plus generator metadata.
    if plan.scenarios is None:
        write_json_atomic(
            folder / "scenarios.json",
            {
                "scenarios": captured_scenarios,
                "generator": "session.generate_statements",
                "n_scenarios": args.n_scenarios,
                "statements_per_concept": args.statements_per_concept,
                "concepts": list(plan.index_of),
                "model_id": session.model_id,
            },
        )

    print(f"wrote {namespace}/{name} ({n_total} nodes)")
    print(
        f"  -> run `saklas manifold discover {namespace}/{name}` to fit"
    )


def _split_manifold_ns_name(raw: str) -> tuple[str, str]:
    """Split a CLI-supplied manifold selector into ``(namespace, name)``.

    Manifolds are addressed by ``(namespace, name)`` directly — not the
    concept ``Selector``/``resolve`` machinery — so a bare name defaults
    to the ``local`` namespace (the manifold lifecycle backends accept
    the pair verbatim).  Mirrors the ``ns/name`` split in
    ``_run_manifold_generate``.
    """
    if "/" in raw:
        ns, name = raw.split("/", 1)
        return ns, name
    return "local", raw


def _run_manifold_install(args: argparse.Namespace) -> None:
    from saklas.io.hf_manifolds import install_manifold

    dst = install_manifold(args.target, args.as_target, force=args.force)
    print(f"Installed {args.target} -> {dst}")


def _run_manifold_search(args: argparse.Namespace) -> None:
    import json as _json
    from saklas.io.hf_manifolds import search_manifolds

    try:
        rows = search_manifolds(args.query or None)
    except ImportError as e:
        print(f"saklas manifold search unavailable: {e}", file=sys.stderr)
        return
    except Exception as e:
        print(f"hf search failed: {type(e).__name__}: {e}", file=sys.stderr)
        return
    if getattr(args, "json_output", False):
        print(_json.dumps(rows, indent=2))
        return
    if not rows:
        print("(no matches)")
        return
    for r in rows:
        line = (
            f"{r['name']:<24} {r['namespace']:<12} [hf]          "
            f"{r['domain_label']:<14} {r['node_count']:>3} nodes  "
            f"{', '.join(r['tensor_models']) or '(unfitted)'}"
        )
        if getattr(args, "verbose", False):
            line = (
                f"{r['name']:<24} {r['namespace']:<12} [hf]          "
                f"{r['domain_label']:<14} {r['node_count']:>3} nodes  "
                f"{r.get('description', '')}"
            )
        print(line)


def _run_manifold_merge(args: argparse.Namespace) -> None:
    from saklas.io.manifolds import (
        ManifoldFormatError, merge_discover_manifolds,
    )

    target_ns, target_name = _split_manifold_ns_name(args.name)
    sources = [_split_manifold_ns_name(s) for s in args.sources]
    if len(sources) < 2:
        print(
            "manifold merge: need >= 2 sources (union of one manifold is a no-op)",
            file=sys.stderr,
        )
        sys.exit(2)
    try:
        dst = merge_discover_manifolds(
            target_ns, target_name, args.description,
            sources=sources,
            fit_mode=args.method,  # parser dest unified to ``method`` (matches discover)
            force=args.force,
        )
    except (FileNotFoundError, FileExistsError, ManifoldFormatError, ValueError) as e:
        print(f"manifold merge failed: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"Merged manifold written to {dst}")
    print(
        f"  -> run `saklas manifold discover {target_ns}/{target_name}` to fit"
    )


def _run_manifold_push(args: argparse.Namespace) -> None:
    from saklas.io.hf import resolve_target_coord
    from saklas.io.hf_manifolds import push_manifold
    from saklas.io.paths import manifold_dir

    ns, name = _split_manifold_ns_name(args.selector)
    folder = manifold_dir(ns, name)
    if not (folder / "manifold.json").exists():
        print(
            f"manifold push: {ns}/{name} not found at {folder}", file=sys.stderr,
        )
        sys.exit(1)

    # The coord follows pack push's resolution: ``--as owner/name`` wins,
    # else ``<whoami>/<name>``.  ``push_manifold`` takes the resolved
    # coord directly (no internal selector machinery), so the runner owns
    # the resolution the way ``cache_ops.push`` does for packs.
    try:
        coord = resolve_target_coord(name, args.as_target)
    except Exception as e:
        print(f"manifold push: {e}", file=sys.stderr)
        sys.exit(2)

    try:
        repo_url, sha = push_manifold(
            folder, coord,
            private=args.private,
            model_scope=args.model,
            variant=args.variant,
            dry_run=args.dry_run,
        )
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        sys.exit(2)

    if sha:
        print(f"Pushed {coord} -> {repo_url} @ {sha[:12]}")
    elif args.dry_run:
        print(f"Dry-run: would push {coord} -> {repo_url}")
    else:
        print(f"Pushed {coord} -> {repo_url}")


def _run_manifold_rm(args: argparse.Namespace) -> None:
    from saklas.io.manifolds import remove_manifold_folder

    ns, name = _resolve_manifold_ns_name(args.selector)
    # Bundled (``default/``) manifolds re-materialize on next session
    # init — mirror ``pack rm``'s confirmation guard for the broad/
    # destructive case (here: a bundled folder the user likely didn't
    # mean to nuke).  ``-y`` skips it.
    if ns == "default" and not args.yes:
        print(
            f"refusing to remove bundled manifold {ns}/{name} "
            f"(re-materializes on restart); pass --yes to confirm",
            file=sys.stderr,
        )
        sys.exit(2)
    try:
        result = remove_manifold_folder(ns, name)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
    msg = f"Removed {result['namespace']}/{result['name']}"
    if result.get("rematerializes_on_restart"):
        msg += " (re-materializes on next session start)"
    print(msg)


def _run_manifold_clear(args: argparse.Namespace) -> None:
    from saklas.io.manifolds import clear_manifold_tensors

    ns, name = _resolve_manifold_ns_name(args.selector)
    # ``args.model`` is the raw model id; ``clear_manifold_tensors`` does
    # the safe-id conversion at the io boundary, exactly as the pack
    # ``cache_ops.delete_tensors`` path does (it passes ``args.model``
    # straight through to ``enumerate_variants``).
    try:
        n = clear_manifold_tensors(ns, name, args.model, variant=args.variant)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
    print(f"Deleted {n} files")


def _run_manifold_refresh(args: argparse.Namespace) -> None:
    from saklas.io.manifolds import refresh_manifold

    ns, name = _resolve_manifold_ns_name(args.selector)
    # ``args.model`` is the raw model id; ``refresh_manifold`` converts to
    # a safe id at the io boundary (via ``clear_manifold_tensors``), the
    # same convention ``cache_ops.refresh``'s scoped path uses.
    try:
        tier = refresh_manifold(ns, name, model_scope=args.model)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
    if tier == "scoped":
        print(f"Refreshed {ns}/{name} ({args.model} tensors cleared — re-fits on next use)")
    elif tier == "skipped":
        print(f"{ns}/{name}: local source, nothing to refresh")
    elif tier == "bundled":
        print(f"Refreshed {ns}/{name} (bundled — re-materialized from package data)")
    else:  # "hf"
        print(f"Refreshed {ns}/{name} (re-pulled from HF)")


def _run_manifold_transfer(args: argparse.Namespace) -> None:
    """Cross-model manifold transfer via Procrustes.

    Mirrors :func:`_run_transfer` (the vector path): resolve the manifold
    folder, fit (or load) the per-layer Procrustes alignment between the
    source and target models' cached neutral activations — the part that
    needs both models loaded — then hand the prebuilt alignment dict to
    :func:`saklas.io.manifolds.transfer_manifold`, which is pure-io and
    only *applies* the map.
    """
    import json as _json

    from saklas.io.alignment import (
        AlignmentError,
        alignment_quality,
        fit_alignment,
        load_alignment_map,
        load_or_compute_neutral_activations,
        save_alignment_map,
    )
    from saklas.io.manifolds import (
        ManifoldFormatError, transfer_manifold,
    )
    from saklas.io.paths import manifold_dir, safe_model_id, tensor_filename

    ns, name = _resolve_manifold_ns_name(args.name)
    folder = manifold_dir(ns, name)
    if not (folder / "manifold.json").exists():
        print(
            f"manifold transfer: {ns}/{name} not found at {folder}",
            file=sys.stderr,
        )
        sys.exit(1)

    src_tensor = folder / tensor_filename(args.src_model)
    if not src_tensor.exists():
        print(
            f"manifold transfer: source fit not found at {src_tensor} — fit "
            f"the manifold on {args.src_model} first",
            file=sys.stderr,
        )
        sys.exit(1)

    tgt_tensor = folder / tensor_filename(
        args.tgt_model, transferred_from=args.src_model,
    )
    if tgt_tensor.exists() and not args.force:
        print(
            f"manifold transfer: target already exists at {tgt_tensor}; "
            f"pass -f to recompute",
            file=sys.stderr,
        )
        sys.exit(1)

    # Fit / load the per-layer Procrustes alignment.  Mirrors
    # ``_run_transfer`` exactly — serialize the two model loads so we
    # never hold both in memory at once.
    from saklas.core.session import SaklasSession

    cached = None if args.force else load_alignment_map(args.src_model, args.tgt_model)
    if cached is None:
        print(
            f"manifold transfer: fitting Procrustes alignment "
            f"{args.src_model} -> {args.tgt_model} "
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
            print(f"manifold transfer: {e}", file=sys.stderr)
            sys.exit(1)
        quality_per_layer = alignment_quality(M, src_acts, tgt_acts)
        save_alignment_map(
            M, args.src_model, args.tgt_model,
            quality_per_layer=quality_per_layer,
        )
    else:
        M, sidecar = cached
        raw_q = sidecar.get("quality_per_layer") or {}
        quality_per_layer = {int(k): float(v) for k, v in raw_q.items()}

    median_quality = median_or_zero(list(quality_per_layer.values())) if quality_per_layer else None

    # Target whitener for the Mahalanobis-share re-bake (mirrors
    # ``_run_transfer``).  Read the target neutral cache directly and center by
    # its own per-layer mean — the neutral mean *is* the probe-centering
    # baseline, so no separate layer_means cache is needed.  Soft ``None`` →
    # ``transfer_manifold`` clears the share (Euclidean fallback at apply).
    target_whitener = _target_whitener_from_neutral_cache(args.tgt_model)

    try:
        out_path = transfer_manifold(
            folder,
            from_model=args.src_model,
            to_model=args.tgt_model,
            alignment=M,
            transfer_quality_estimate=median_quality,
            whitener=target_whitener,
            force=args.force,
        )
    except (FileNotFoundError, FileExistsError, ManifoldFormatError) as e:
        print(f"manifold transfer failed: {e}", file=sys.stderr)
        sys.exit(1)

    payload = {
        "namespace": ns,
        "name": name,
        "source_model": args.src_model,
        "target_model": args.tgt_model,
        "tensor": str(out_path),
        "transferred_layers": sorted(M.keys()),
        "median_transfer_quality": (
            round(median_quality, 4) if median_quality is not None else None
        ),
    }
    if args.json_output:
        print(_json.dumps(payload, indent=2))
        return

    quality_str = f"{median_quality:.3f}" if median_quality is not None else "n/a"
    print(
        f"Transferred manifold {ns}/{name} "
        f"from {args.src_model} -> {args.tgt_model}\n"
        f"  layers:           {len(M)} shared\n"
        f"  median quality:   {quality_str} (R^2 across shared layers)\n"
        f"  tensor:           {out_path}\n"
        f"  variant suffix:   :from-{safe_model_id(args.src_model)}"
    )


@_saklas_error_exit
def _run_manifold(args: argparse.Namespace) -> None:
    """Dispatch ``saklas manifold <verb>`` (and the deprecated ``vector manifold``)."""
    cmd = getattr(args, "manifold_cmd", None)
    if cmd is None:
        print("usage: saklas manifold <verb> [...]")
        print()
        width = max(len(v) for v, _ in _MANIFOLD_VERBS)
        for v, desc in _MANIFOLD_VERBS:
            print(f"  {v:<{width}}  {desc}")
        sys.exit(0)
    if cmd == "fit":
        _run_manifold_fit(args)
        return
    if cmd == "discover":
        _run_manifold_discover(args)
        return
    if cmd == "generate":
        _run_manifold_generate(args)
        return
    if cmd == "merge":
        _run_manifold_merge(args)
        return
    if cmd == "install":
        _run_manifold_install(args)
        return
    if cmd == "search":
        _run_manifold_search(args)
        return
    if cmd == "push":
        _run_manifold_push(args)
        return
    if cmd == "rm":
        _run_manifold_rm(args)
        return
    if cmd == "clear":
        _run_manifold_clear(args)
        return
    if cmd == "refresh":
        _run_manifold_refresh(args)
        return
    if cmd == "transfer":
        _run_manifold_transfer(args)
        return
    if cmd == "ls":
        _run_manifold_ls(args)
        return
    if cmd == "show":
        _run_manifold_show(args)
        return
    print(f"unknown manifold verb {cmd!r}", file=sys.stderr)
    sys.exit(2)


_SUBSPACE_RUNNERS = {
    "extract":  _run_extract,
    "merge":    _run_merge,
    "clone":    _run_clone,
    "compare":  _run_compare,
    "why":      _run_why,
    "transfer": _run_transfer,
}

# The deprecated ``vector`` alias dispatches the subspace verbs plus the
# nested ``manifold`` (which reads the same ``manifold_cmd`` dest).
_VECTOR_RUNNERS = {
    **_SUBSPACE_RUNNERS,
    "manifold": _run_manifold,
}


@_saklas_error_exit
def _run_subspace(args: argparse.Namespace) -> None:
    """Dispatch ``saklas subspace <verb>`` (the flat-artifact / vector verbs)."""
    cmd = getattr(args, "subspace_cmd", None)
    if cmd is None:
        print("usage: saklas subspace <verb> [...]")
        print()
        width = max(len(v) for v, _ in _SUBSPACE_VERBS)
        for v, desc in _SUBSPACE_VERBS:
            print(f"  {v:<{width}}  {desc}")
        print()
        print("Run `saklas subspace <verb> -h` for verb-specific options.")
        sys.exit(0)
    _SUBSPACE_RUNNERS[cmd](args)


@_saklas_error_exit
def _run_vector(args: argparse.Namespace) -> None:
    """[deprecated 4.0] alias for ``subspace`` (+ nested ``manifold``)."""
    vector_cmd = getattr(args, "vector_cmd", None)
    if vector_cmd is None:
        print("usage: saklas vector <verb> [...]   [deprecated: use `subspace`/`manifold`]")
        print()
        width = max(len(v) for v, _ in _VECTOR_VERBS)
        for v, desc in _VECTOR_VERBS:
            print(f"  {v:<{width}}  {desc}")
        print()
        print("Run `saklas vector <verb> -h` for verb-specific options.")
        sys.exit(0)
    target = "manifold" if vector_cmd == "manifold" else f"subspace {vector_cmd}"
    print(
        f"warning: `saklas vector {vector_cmd}` is deprecated (4.0) — "
        f"use `saklas {target}`.",
        file=sys.stderr,
    )
    _VECTOR_RUNNERS[vector_cmd](args)


@_saklas_error_exit
def _run_experiment(args: argparse.Namespace) -> None:
    """Dispatch ``saklas experiment <verb>``."""
    cmd = getattr(args, "experiment_cmd", None)
    if cmd is None:
        print("usage: saklas experiment <verb> [...]")
        print()
        width = max(len(v) for v, _ in _EXPERIMENT_VERBS)
        for v, desc in _EXPERIMENT_VERBS:
            print(f"  {v:<{width}}  {desc}")
        sys.exit(0)
    if cmd == "fan":
        _run_experiment_fan(args)
        return
    if cmd == "transcript":
        _run_experiment_transcript(args)
        return
    if cmd == "naturalness":
        _run_experiment_naturalness(args)
        return
    print(f"unknown experiment verb {cmd!r}", file=sys.stderr)
    sys.exit(2)


def _run_experiment_naturalness(args: argparse.Namespace) -> None:
    """Score a steered generation's behavior-manifold naturalness."""
    import json as _json

    import torch

    from saklas.core.manifold import (
        compute_node_behavior_centroid,
        compute_trajectory_distributions,
        domain_from_spec,
        fit_behavior_manifold,
        trajectory_naturalness,
    )
    from saklas.core.profile import Profile
    from saklas.core.sampling import SamplingConfig
    from saklas.core.steering_expr import ManifoldTerm, parse_expr
    from saklas.io.manifolds import ManifoldFolder

    _load_effective_config(args)
    mfolder = Path(args.manifold)
    if not (mfolder / "manifold.json").exists():
        print(
            f"experiment naturalness: no manifold.json in {mfolder}",
            file=sys.stderr,
        )
        sys.exit(2)
    mf = ManifoldFolder.load(mfolder)
    node_groups = mf.node_groups()
    domain = domain_from_spec(mf.domain)
    node_coords = torch.tensor(mf.node_coords, dtype=torch.float32)
    node_params = domain.embed(node_coords)

    _print_startup(args)
    session = _make_session(args)
    _print_model_info(session)

    # 1. Fit the behavior manifold from the node corpus — each node's
    #    mean next-token distribution, in Hellinger space.
    print(f"fitting behavior manifold ({len(node_groups)} nodes)...")
    centroids = [
        compute_node_behavior_centroid(
            session.model, session.tokenizer, session.device, statements,
        )
        for _label, statements in node_groups
    ]
    behavior = fit_behavior_manifold(torch.stack(centroids), node_params)

    sampling = SamplingConfig(max_tokens=args.max_tokens, seed=0)

    def _score(steer: str | None) -> tuple[str, float, float]:
        result = session.generate(
            args.prompt, steering=steer, sampling=sampling,
        )
        text = result.text
        traj = compute_trajectory_distributions(
            session.model, session.tokenizer, session.device, text,
        )
        per_step = trajectory_naturalness(traj, behavior, domain, node_coords)
        return text, float(per_step.mean()), float(per_step.max())

    rows: list[dict[str, Any]] = []
    _text, mean_d, max_d = _score(args.steer)
    rows.append({
        "label": "manifold", "steering": args.steer,
        "mean_bhattacharyya": mean_d, "max_bhattacharyya": max_d,
    })

    if args.compare_linear:
        steering = parse_expr(args.steer)
        mterms = [
            v for v in steering.alphas.values()
            if isinstance(v, ManifoldTerm)
        ]
        if len(mterms) != 1 or len(steering.alphas) != 1:
            print(
                "experiment naturalness: --compare-linear requires the "
                "steer expression to be a single manifold term",
                file=sys.stderr,
            )
            sys.exit(2)
        mt = mterms[0]
        session._ensure_manifold_loaded(mt.manifold)
        act_manifold = session._manifolds[mt.manifold]
        # Resolve label-form positions to coords up front — every
        # downstream call here wants a coord tuple, and the chord
        # baseline is per-coord arithmetic that can't operate on a
        # string label.
        mt_position = act_manifold.resolve_position(mt.position)
        # Linear baseline: the straight chord through activation space
        # from the manifold point at node 0 to the term's position, per
        # layer — what plain additive steering would do instead of
        # following the manifold's curvature.
        origin = act_manifold.node_coords[0]
        chord = {
            L: (
                act_manifold.manifold_point(L, mt_position)
                - act_manifold.manifold_point(L, origin)
            )
            for L in act_manifold.layer_indices
        }
        session.steer("__manifold_linear_baseline__", Profile(chord))
        _ltext, lmean, lmax = _score(
            f"{mt.coeff:g} __manifold_linear_baseline__"
        )
        pos_label = (
            mt.position if isinstance(mt.position, str)
            else ",".join(f"{c:g}" for c in mt_position)
        )
        rows.append({
            "label": "linear-chord", "steering": f"chord@{pos_label}",
            "mean_bhattacharyya": lmean, "max_bhattacharyya": lmax,
        })

    if args.json_output:
        print(_json.dumps({"prompt": args.prompt, "results": rows}, indent=2))
        return
    print()
    print("behavior-manifold naturalness  (lower = more natural)")
    for r in rows:
        print(
            f"  {r['label']:<14} mean D_B={r['mean_bhattacharyya']:.4f}  "
            f"max D_B={r['max_bhattacharyya']:.4f}"
        )
    if len(rows) == 2:
        delta = rows[1]["mean_bhattacharyya"] - rows[0]["mean_bhattacharyya"]
        verdict = (
            "manifold steering stays closer to the behavior manifold"
            if delta > 0 else
            "linear steering stays closer to the behavior manifold"
        )
        print(f"  -> {verdict} (Δmean={delta:+.4f})")


def _run_experiment_transcript(args: argparse.Namespace) -> None:
    """Dispatch ``saklas experiment transcript <verb>``."""
    cmd = getattr(args, "transcript_cmd", None)
    if cmd is None:
        print("usage: saklas experiment transcript <verb> [...]")
        print()
        print("  run  Replay a transcript on the current session")
        sys.exit(0)
    if cmd == "run":
        _run_transcript_run(args)
        return
    print(f"unknown experiment transcript verb {cmd!r}", file=sys.stderr)
    sys.exit(2)


def _parse_grid_terms(raw_terms: list[str]) -> dict[str, list[float]]:
    from saklas.tui.loom_helpers import AlphaListError, parse_alpha_list

    grid: dict[str, list[float]] = {}
    for raw in raw_terms:
        if "=" not in raw:
            print(
                f"experiment fan: grid term must be CONCEPT=ALPHAS, got {raw!r}",
                file=sys.stderr,
            )
            sys.exit(2)
        name, alpha_text = raw.split("=", 1)
        name = name.strip()
        if not name:
            print("experiment fan: grid concept cannot be empty", file=sys.stderr)
            sys.exit(2)
        try:
            alphas = parse_alpha_list(alpha_text)
        except AlphaListError as e:
            print(f"experiment fan: {name}: {e}", file=sys.stderr)
            sys.exit(2)
        if not alphas:
            print(f"experiment fan: {name}: alpha list is empty", file=sys.stderr)
            sys.exit(2)
        grid[name] = [float(a) for a in alphas]
    return grid


def _run_experiment_fan(args: argparse.Namespace) -> None:
    import json as _json

    _load_effective_config(args)
    _print_startup(args)
    session = _make_session(args)
    _print_model_info(session)

    grid = _parse_grid_terms(args.grid)
    runset = session.generate_sweep(
        args.prompt,
        grid,
        base_steering=args.base_steering,
        stateless=False,
    )
    if args.json_output:
        print(_json.dumps(runset.to_dict(), indent=2))
        return
    print(f"experiment fan: {len(runset)} run(s)")
    for idx, result in enumerate(runset):
        node_id = runset.node_ids[idx] if idx < len(runset.node_ids) else None
        row = runset.grid[idx] if idx < len(runset.grid) else {}
        row_str = ", ".join(f"{k}={v:+.3f}" for k, v in row.items())
        node_str = f" node={node_id[:8]}" if node_id else ""
        print(
            f"{idx:>3}: {row_str}{node_str} "
            f"tokens={result.token_count} finish={result.finish_reason}"
        )


def _run_transcript_run(args: argparse.Namespace) -> None:
    from saklas.core.transcript import (
        Transcript, TranscriptError,
    )

    transcript_path = Path(args.path)
    if not transcript_path.is_file():
        print(f"transcript run: {transcript_path}: file not found", file=sys.stderr)
        sys.exit(2)
    try:
        transcript = Transcript.load(transcript_path)
    except TranscriptError as e:
        print(f"transcript run: {e}", file=sys.stderr)
        sys.exit(2)

    _load_effective_config(args)
    if not args.model:
        if transcript.model_id:
            args.model = transcript.model_id
        else:
            print(
                "transcript run: model required (pass <model> or include "
                "`model_id` in the transcript)",
                file=sys.stderr,
            )
            sys.exit(2)

    _print_startup(args)
    session = _make_session(args)
    _print_model_info(session)

    # Import via ``default`` so the transcript lands as a fresh branch
    # under the synthetic root; replay walks the imported branch and
    # reports drift inline.
    try:
        leaf_id = transcript.import_into(
            session, mode="default", strict=args.strict,
        )
    except TranscriptError as e:
        print(f"transcript run: {e}", file=sys.stderr)
        sys.exit(2)

    print(f"transcript: {len(transcript.turns)} turns loaded "
          f"(leaf: {leaf_id[:8]})")
    print()
    for idx, turn in enumerate(transcript.turns):
        if turn.role != "user":
            continue
        print(f"--- replay turn {idx} ---")
        print(f"user: {turn.text[:80]}")
        # Look ahead for the assistant turn this user prompt produced.
        expected = None
        if idx + 1 < len(transcript.turns) and transcript.turns[idx + 1].role == "assistant":
            expected = transcript.turns[idx + 1]
        try:
            recipe = expected.recipe if expected is not None else None
            steering = recipe.steering if recipe is not None else None
            sampling = recipe.sampling if recipe is not None else None
            result = session.generate(
                turn.text,
                steering=steering,
                sampling=sampling,
                stateless=True,
            )
        except Exception as e:
            print(f"  replay failed: {e}")
            continue
        print(f"assistant: {result.text[:120]}")
        if expected is not None and expected.readings:
            actual = {n: r.mean for n, r in result.readings.items()}
            deltas = []
            for name, expected_v in expected.readings.items():
                actual_v = actual.get(name, 0.0)
                deltas.append((name, actual_v - expected_v, expected_v, actual_v))
            deltas.sort(key=lambda x: abs(x[1]), reverse=True)
            print("  readings drift (top 5):")
            for name, d, ev, av in deltas[:5]:
                print(f"    {name:<32}  Δ {d:+.4f}  (expected {ev:+.4f} → got {av:+.4f})")
        print()


_COMMAND_RUNNERS = {
    "tui":        _run_tui,
    "serve":      _run_serve,
    "pack":       _run_pack,
    "subspace":   _run_subspace,
    "manifold":   _run_manifold,
    "vector":     _run_vector,
    "config":     _run_config,
    "experiment": _run_experiment,
}
