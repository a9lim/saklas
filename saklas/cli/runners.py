"""Runner functions for saklas CLI subcommands."""

from __future__ import annotations

import argparse
import functools
from operator import itemgetter
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, TypeVar

from saklas.cli.parsers import (
    _EXPERIMENT_VERBS, _MANIFOLD_VERBS, _PACK_VERBS, _TEMPLATE_VERBS,
)
from saklas.core.errors import SaklasError
from saklas.core.stats import median_or_zero
from saklas.io.paths import VARIANT_SUFFIX_RE

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

def _resolve_probes(raw: list[str] | None) -> list[str] | None:
    """Resolve the ``--probes`` flag to a value for ``from_pretrained``.

    ``None`` (unset) and ``all`` both return ``None`` — the session's default-
    roster signal, which attaches the tagged concept axes *plus* every already-
    fitted bundled multi-node manifold (``personas`` / ``emotions``).  ``none``
    / ``[]`` disable probes; an explicit category list is honored verbatim (the
    tagged concepts only, no multi-node sweep).
    """
    if raw is None or raw == ["all"]:
        return None
    if raw in (["none"], []):
        return []
    return raw


def _target_whitener_from_neutral_cache(model_id: str) -> Any:
    """Build a target-model whitener from neutral activations without loading a model.

    Transfer re-bakes the share in the target Mahalanobis metric, which is
    now mandatory — there is no Euclidean rebake.  A missing, stale, or
    degenerate target neutral cache raises :class:`WhitenerError` (wrapped
    from whatever the loader raised) so the caller fails loudly with an
    actionable hint instead of silently degrading.
    """
    from saklas.core.mahalanobis import LayerWhitener, WhitenerError

    try:
        return LayerWhitener.from_cache(model_id)
    except WhitenerError:
        raise
    except Exception as e:
        raise WhitenerError(
            f"transfer requires a Mahalanobis whitener for the target model "
            f"'{model_id}', but its neutral activation cache is missing or "
            f"unusable ({e}); generate neutral activations for the TARGET "
            f"model first"
        ) from e


def _load_or_fit_transfer_alignment(
    src_model: str,
    tgt_model: str,
    *,
    force: bool,
    label: str,
) -> tuple[dict[int, Any], dict[int, float], Path]:
    """Load or fit a Procrustes alignment for vector/manifold transfer."""
    from saklas.io.alignment import (
        AlignmentError,
        alignment_cache_path,
        alignment_quality,
        fit_alignment,
        load_alignment_map,
        load_or_compute_neutral_activations,
        save_alignment_map,
    )

    cached = None if force else load_alignment_map(src_model, tgt_model)
    if cached is not None:
        M, sidecar = cached
        raw_q = sidecar.get("quality_per_layer") or {}
        quality_per_layer = {int(k): float(v) for k, v in raw_q.items()}
        map_path, _ = alignment_cache_path(src_model, tgt_model)
        return M, quality_per_layer, map_path

    # Need both models loaded to compute neutrals.  Loading two large models
    # simultaneously is non-trivial, so serialize: source, then target.
    print(
        f"{label}: fitting Procrustes alignment {src_model} -> {tgt_model} "
        f"(this may load each model briefly)...",
        file=sys.stderr,
    )
    from saklas.core.session import SaklasSession

    with SaklasSession.from_pretrained(
        src_model, device="auto", probes=[],
    ) as src_sess:
        src_acts = load_or_compute_neutral_activations(
            src_sess._model, src_sess._tokenizer, src_sess._layers,
            model_id=src_model, force=force,
        )
    with SaklasSession.from_pretrained(
        tgt_model, device="auto", probes=[],
    ) as tgt_sess:
        tgt_acts = load_or_compute_neutral_activations(
            tgt_sess._model, tgt_sess._tokenizer, tgt_sess._layers,
            model_id=tgt_model, force=force,
        )
    try:
        M = fit_alignment(src_acts, tgt_acts)
    except AlignmentError as e:
        print(f"{label}: {e}", file=sys.stderr)
        sys.exit(1)

    quality_per_layer = alignment_quality(M, src_acts, tgt_acts)
    map_path = save_alignment_map(
        M, src_model, tgt_model,
        quality_per_layer=quality_per_layer,
    )
    return M, quality_per_layer, map_path


def _make_session(args: argparse.Namespace):
    from saklas.core.session import SaklasSession
    probe_categories = _resolve_probes(args.probes)
    # ``~`` / ``|`` projection is Mahalanobis-only (closed-form LEACE);
    # ``--no-dls`` opts out of the discriminative-layer mask.  The flag and
    # YAML are already merged onto ``args`` by ``_load_effective_config``.
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
        print(f"  Registered '{registry_key}'")

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

    # The default probe roster — tagged concept axes plus every fitted bundled
    # multi-node manifold (``personas`` / ``emotions``) — is attached at session
    # construction (``_bootstrap_manifold_probes``), so the dashboard's probe
    # rack opens already watching them with no serve-side step.

    _warmup_session(session)

    print(f"\nServing on http://{args.host}:{args.port}")
    print(f"OpenAI-compatible:  http://{args.host}:{args.port}/v1")
    print(f"Ollama-compatible:  http://{args.host}:{args.port}/api")
    print(f"API docs:           http://{args.host}:{args.port}/docs")
    if args.port != 11434:
        print("Tip: for drop-in Ollama compatibility, run with `--port 11434`.")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


# --- manifold compute runners --------------------------------------------

def _run_manifold_bake(args: argparse.Namespace) -> None:
    from saklas.io import merge as merge_mod
    dst = merge_mod.merge_into_manifold(
        args.name, args.expression, model=args.model,
        force=args.force, strict=args.strict,
    )
    print(f"Merged manifold written to {dst}")


def _require_model(args: argparse.Namespace) -> None:
    if not args.model:
        # The compute leaves (extract / fit / generate / bake / transfer) carry
        # the leaf on ``manifold_cmd`` — surface it so the error reads
        # "manifold fit".
        manifold_cmd = getattr(args, "manifold_cmd", None)
        cmd = f"manifold {manifold_cmd}" if manifold_cmd else "?"
        print(f"{cmd}: -m/--model is required", file=sys.stderr)
        sys.exit(2)


def _run_manifold_extract(args: argparse.Namespace) -> None:
    _require_model(args)
    import pathlib
    from saklas.core.session import canonical_concept_name
    from saklas.io.paths import manifold_dir, tensor_filename

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

    requested_release = getattr(args, "sae", None)
    requested_role = getattr(args, "role", None)
    if requested_release and requested_role:
        print(
            "extract: --sae and --role are mutually exclusive "
            "(role substitution bakes into the manifold's node corpora; SAE "
            "reconstructs the centroids — the two don't compose)",
            file=sys.stderr,
        )
        sys.exit(2)

    _print_startup(args)
    session = _make_session(args)
    _print_model_info(session)

    # A steering vector is a 2-node ``pca`` manifold (4.0); it lives under
    # ``manifolds/<ns>/<canonical>/``.  The per-model fitted tensor (raw or
    # ``_sae-<release>``) inside it is the existence/destination marker; a
    # role-augmented fit bakes into the node corpora and writes the canonical
    # tensor name (no ``_role-`` suffix).
    ns = getattr(args, "namespace", None) or "local"
    canonical = canonical_concept_name(raw, baseline)
    tensor_name = tensor_filename(session.model_id, release=requested_release)
    tensor_path = pathlib.Path(manifold_dir(ns, canonical)) / tensor_name
    if tensor_path.exists() and not args.force:
        print(f"already extracted at {tensor_path}")
        sys.exit(0)

    extract_kwargs: dict[str, Any] = {}
    if requested_release:
        extract_kwargs["sae"] = args.sae
    if getattr(args, "sae_revision", None):
        extract_kwargs["sae_revision"] = args.sae_revision
    if requested_role:
        extract_kwargs["role"] = requested_role
    if getattr(args, "namespace", None) is not None:
        extract_kwargs["namespace"] = args.namespace
    if args.force:
        extract_kwargs["force"] = True

    try:
        if baseline is not None:
            name, _profile = session.extract(raw, baseline=baseline, **extract_kwargs)
        else:
            name, _profile = session.extract(raw, **extract_kwargs)
    except Exception as e:
        print(f"extract failed: {e}", file=sys.stderr)
        sys.exit(1)

    final_path = pathlib.Path(manifold_dir(ns, name)) / tensor_name
    print(f"extracted {name} -> {final_path}")


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


# Single source of truth lives in ``io.paths`` (owns the variant scheme).
_VARIANT_SUFFIX_RE = VARIANT_SUFFIX_RE


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
    if _VARIANT_SUFFIX_RE.match(tail) and head:
        # ``_VARIANT_SUFFIX_RE`` is anchored over ``[a-z0-9._-]`` only, so a
        # matching ``tail`` can't contain ``/`` — ``model:<org>/<name>`` falls
        # through to ``return raw, None`` because its tail fails the regex.
        return head, tail
    return raw, None


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


def _run_manifold_compare(args: argparse.Namespace) -> None:
    import json as _json
    from saklas.io.selectors import parse as sel_parse, resolve

    # Compare is Mahalanobis-only: load the per-model whitener once up
    # front and share it across every ``cosine_similarity`` call below.
    # Failure is fatal — there is no Euclidean path, so a missing neutral
    # cache surfaces directly instead of degrading silently.
    from saklas.core.mahalanobis import LayerWhitener, WhitenerError

    whitener: "Any | None"
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

    # Load profiles by folding each concept's fitted 2-node ``pca`` manifold.
    profiles: dict[str, Profile] = {}
    profile_identities: dict[str, tuple[str, str]] = {}
    for name, variant in names:
        try:
            folded = _fold_manifold_to_profile_with_identity(
                name, args.model, variant,
            )
        except Exception as e:
            print(f"warning: {e}, skipping", file=sys.stderr)
            continue
        if folded is None:
            print(
                f"warning: no fitted manifold for '{name}' with model "
                f"{args.model}, skipping",
                file=sys.stderr,
            )
            continue
        loaded, ns, bare = folded
        display_base = f"{ns}/{bare}" if "/" in name else bare
        display = display_base if variant is None else f"{display_base}:{variant}"
        profiles[display] = loaded
        profile_identities[display] = (ns, bare)

    if len(profiles) < 1:
        print("compare: no loadable profiles found", file=sys.stderr)
        sys.exit(1)

    ordered = list(profiles.keys())

    # 1-arg mode: rank all installed against the target.
    if len(args.concepts) == 1 and len(ordered) == 1:
        target_name = ordered[0]
        target = profiles[target_name]
        target_identity = profile_identities.get(target_name)

        # Every concept ships as a fitted 2-node ``pca`` manifold — fold
        # every fitted one for this model into the ranking pool.  Keep the
        # exact namespace identity internally, then choose display names after
        # the full pool is known: bare for unique names, qualified for
        # collisions.
        other_entries: dict[tuple[str, str], Profile] = {}
        for mname, mprof in _fold_all_fitted_manifolds(
            args.model, exclude_identity=target_identity,
        ).items():
            ns, bare = mname.split("/", 1)
            other_entries[(ns, bare)] = mprof

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


def _run_manifold_why(args: argparse.Namespace) -> None:
    import json as _json
    from saklas.io.selectors import AmbiguousSelectorError

    # Peel off a ``:<variant>`` suffix before resolving the manifold.
    name_part, variant = _split_variant_suffix(args.concept)

    # Every concept is a fitted 2-node ``pca`` manifold — fold it to a vector.
    profile: "Profile | None" = None
    concept_name = name_part if variant is None else f"{name_part}:{variant}"
    try:
        folded = _fold_manifold_to_profile_with_identity(
            name_part, args.model, variant,
        )
    except AmbiguousSelectorError as e:
        print(f"why: {e}", file=sys.stderr)
        sys.exit(1)
    if folded is not None:
        profile, ns, bare = folded
        concept_base = f"{ns}/{bare}" if "/" in name_part else bare
        concept_name = concept_base if variant is None else f"{concept_base}:{variant}"
    if profile is None:
        print(
            f"why: no fitted manifold for '{args.concept}' with model "
            f"{args.model}. If it's authored but unfit, fit it first: "
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


def _run_manifold_fit(args: argparse.Namespace) -> None:
    """``saklas manifold fit`` — fit an authored OR discover-mode manifold.

    The positional ``target`` is a manifold name OR a folder path.  When it
    resolves to an authored folder the fit runs as-is (no hyperparam knobs
    allowed).  When it resolves to a discover-mode folder (``pca`` /
    ``spectral``) any supplied hyperparam override (``--method`` / ``--max-dim``
    / ``--var-threshold`` / ``--k-nn`` / ``--bandwidth`` / ``--max-subspace-dim``)
    is written into ``manifold.json`` atomically *before* the fit so the cache
    key reflects the actual fit inputs.  Supplying overrides against an authored
    folder is an error (mirrors the server's 400).
    """
    import json as _json
    from saklas.io.atomic import write_json_atomic
    from saklas.io.manifolds import (
        ManifoldFolder, ManifoldFormatError, _sanitize_hyperparams,
        domain_label,
    )

    _require_model(args)

    # Resolve ``target`` to a folder.  A path that exists on disk (or that
    # carries a ``manifold.json``) is taken as an authored-folder path —
    # ``manifold fit /path/to/folder``; otherwise it is a manifold *name*
    # resolved cross-namespace the same way the lifecycle verbs resolve.
    target = args.target
    target_path = Path(target)
    if target_path.is_dir() or (target_path / "manifold.json").exists():
        folder = target_path
    else:
        folder = _resolve_manifold_folder(target)

    if not (folder / "manifold.json").exists():
        print(
            f"manifold fit: no manifold.json in {folder}", file=sys.stderr,
        )
        sys.exit(2)

    # Load the folder to learn its fit_mode before paying for a model load.
    try:
        mf = ManifoldFolder.load(folder)
    except ManifoldFormatError as e:
        print(f"manifold fit: {e}", file=sys.stderr)
        sys.exit(2)

    # Did the user supply any discover-mode hyperparam override?
    override_supplied = any(
        getattr(args, attr, None) is not None
        for attr in (
            "method", "max_dim", "min_dim", "var_threshold", "k_nn",
            "bandwidth", "max_subspace_dim",
        )
    )

    new_fit_mode = mf.fit_mode
    if override_supplied:
        if not mf.is_discover:
            # Mirror the server's 400: hyperparam overrides are discover-only.
            print(
                f"manifold fit: fit_mode/hyperparams overrides are "
                f"discover-mode only; {folder} is authored",
                file=sys.stderr,
            )
            sys.exit(2)
        new_fit_mode = args.method or mf.fit_mode
        if new_fit_mode not in {"pca", "spectral", "auto"}:
            print(
                f"manifold fit: invalid method {new_fit_mode!r}",
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
        if getattr(args, "min_dim", None) is not None:
            new_hyperparams["min_dim"] = int(args.min_dim)
        if args.k_nn is not None:
            new_hyperparams["k_nn"] = int(args.k_nn)
        if args.bandwidth is not None:
            new_hyperparams["bandwidth"] = float(args.bandwidth)
        # Spectral-only knob — dropped by ``_sanitize_hyperparams`` for a pca
        # fit (a flat fit's per-layer subspace dim is its layout dim, capped
        # by ``--max-dim``, not a separate knob).
        if getattr(args, "max_subspace_dim", None) is not None:
            new_hyperparams["max_subspace_dim"] = int(args.max_subspace_dim)
        # Curved-fit smoothing (spectral/auto): "auto" → GCV, else a float λ.
        if getattr(args, "smoothing", None) is not None:
            s = args.smoothing
            new_hyperparams["smoothing"] = s if s == "auto" else float(s)
        # Auto-only: periodic-detection persistence threshold.
        if getattr(args, "persistence_frac", None) is not None:
            new_hyperparams["persistence_frac"] = float(args.persistence_frac)
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
        manifold = session.fit(
            folder,
            sae=getattr(args, "sae", None),
            sae_revision=getattr(args, "sae_revision", None),
            on_progress=lambda m: print(f"  {m}"),
        )
    except (ValueError, ManifoldFormatError) as e:
        print(f"manifold fit failed: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"manifold fit failed: {e}", file=sys.stderr)
        sys.exit(1)
    tail = (
        f", fit_mode={new_fit_mode}" if mf.is_discover else ""
    )
    print(
        f"fitted manifold '{manifold.name}' "
        f"({len(manifold.layers)} layers, {len(manifold.node_labels)} nodes, "
        f"{domain_label(manifold.domain.to_spec())}, "
        f"{manifold.feature_space}{tail})"
    )


def _iter_manifold_folders(namespace: str | None):
    """Yield ``(namespace, ManifoldFolder)`` for every installed manifold.

    Thin wrapper over :func:`saklas.io.manifolds.iter_manifold_folders` —
    the discovery walk lives in ``io`` so the server shares it without
    importing ``cli``.

    Materializes bundled manifolds (e.g. ``default/personas``) before
    the walk.  CLI verbs like ``manifold fit`` / ``pack show`` / ``pack ls``
    resolve folders *before* any session is constructed, so the session-
    startup ``materialize_bundled_manifolds`` call hasn't fired yet —
    without this hop, a user's first ``saklas manifold fit
    default/personas`` would miss the bundled folder and exit with "not
    found".  The format-version short-circuit inside the materialize
    function makes repeated calls cheap (no-op when up-to-date).  The
    server has its own session-driven materialization at startup, so
    going through ``iter_manifold_folders`` directly stays correct.
    """
    from saklas.io.manifolds import (
        iter_manifold_folders, materialize_bundled_manifolds,
    )
    from saklas.io.templates import materialize_bundled_templates

    # Templates first — a bundled manifold may ``template_ref`` a bundled one.
    materialize_bundled_templates()
    materialize_bundled_manifolds()
    yield from iter_manifold_folders(namespace)


def _run_pack_ls(args: argparse.Namespace) -> None:
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


def _run_pack_show(args: argparse.Namespace) -> None:
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
        if sc.node_spread_per_layer:
            entry["node_spread"] = sc.node_spread_per_layer
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
            mf.node_labels, mf.node_coords, node_roles_padded, strict=True,
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
            # Concept layer profile: the whitened between-node spread per layer
            # (background-σ² units).  Report the peak layers — where this
            # concept's signal concentrates across the stack.
            spread = f.get("node_spread") or {}
            if spread:
                ranked = sorted(
                    ((int(layer), float(v)) for layer, v in spread.items()),
                    key=lambda kv: kv[1],
                    reverse=True,
                )
                peak = ", ".join(f"L{layer}={v:.2f}" for layer, v in ranked[:5])
                print(f"      signal by layer (peak): {peak}")
    else:
        print("  fitted models: (none — run `saklas manifold fit`)")


def _resolve_manifold_ns_name(name: str) -> tuple[str, str]:
    """Resolve a CLI-supplied ``NAME`` (or ``ns/name``) to ``(namespace, name)``.

    Mirrors :func:`_run_pack_show`'s ambiguity handling — bare names
    resolve cross-namespace (reaching e.g. bundled ``default/`` when that's
    the only match) and raise on collision; an ``ns/name`` form pins to a
    single namespace.  Exits with a clear error on miss / ambiguity.

    This is the lifecycle analogue of the concept-selector cross-namespace
    resolution: ``pack clear`` / ``pack refresh`` / ``pack rm`` / ``manifold
    transfer`` / ``pack export`` (and the folder-returning ``manifold fit`` by
    name / ``pack show``) all route a bare name through here so it can reach
    any namespace, not just ``local/``.  (``generate`` deliberately does *not*
    — it authors a fresh folder and defaults bare → ``local/`` via
    :func:`_split_manifold_ns_name`.)

    An explicit ``ns/name`` pins directly — it's returned verbatim without a
    filesystem walk, leaving the existence check to the io backend (which
    raises ``FileNotFoundError``).  Only a *bare* name walks the installed
    manifolds to discover its namespace, raising on collision / miss.
    """
    # Materialize bundled artifacts up front so a *qualified* reference to a
    # newly-shipped bundled manifold (``default/<name>``) resolves even on
    # an existing ``~/.saklas`` that predates it — the ``ns/name`` branch below
    # returns verbatim and never walks the installed folders, so it would
    # otherwise miss the materialize that the bare-name walk triggers.  Process-
    # scoped no-op, so this is free when already done.  Templates before
    # manifolds (a templated manifold's fit resolves its ``template_ref``).
    from saklas.io.manifolds import materialize_bundled_manifolds
    from saklas.io.templates import materialize_bundled_templates
    materialize_bundled_templates()
    materialize_bundled_manifolds()

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
    want the folder directly (``fit`` by-name, ``show``); the lifecycle verbs
    that hand a ``(namespace, name)`` pair to their io backend call the
    pair-returning form instead.
    """
    from saklas.io.paths import manifold_dir

    ns, resolved = _resolve_manifold_ns_name(name)
    return manifold_dir(ns, resolved)


def _run_manifold_generate(args: argparse.Namespace) -> None:
    """``saklas manifold generate`` -- author + LLM-fill a discover folder (A2).

    Two-step end-to-end: load the model, run :meth:`SaklasSession.generate_responses`
    to fill each node's conversational corpus (in-character responses to the
    shared baseline prompts), then leave the folder ready for ``manifold
    discover`` to fit.  The two-step split keeps failure modes legible (a flaky
    generation leaves inspectable corpora) and lets the user review before
    paying for the fit.
    """
    from saklas.io.manifolds import (
        ManifoldFormatError,
        append_discover_manifold_node,
        plan_discover_generation,
    )
    from saklas.io.paths import manifold_dir

    _require_model(args)
    if len(args.concepts) < 2:
        print(
            "manifold generate: need >= 2 concepts (a discover manifold is "
            "meaningless with one node)",
            file=sys.stderr,
        )
        sys.exit(2)

    namespace = "local"
    name = args.name
    if "/" in name:
        namespace, name = name.split("/", 1)

    folder = manifold_dir(namespace, name)
    # ``-f/--force`` is a clean slate; the default *resumes* -- fills any
    # missing node corpora and appends concepts new to the roster, so a run
    # killed half-way picks up where it left off and adding a node is a re-run.
    if args.force and (folder / "manifold.json").exists():
        import shutil
        shutil.rmtree(folder)

    # ``--role-per-node``: the concept slug doubles as that node's assistant-role
    # substitution -- a persona manifold pooled in role-baselined space (the
    # explicit role overrides the kind-derived elicitation label at both
    # generation and capture).  Otherwise the node\'s ``kind`` drives a
    # generation-only elicitation role and capture stays standard (swap-back).
    node_roles: dict[str, str | None] | None = None
    if getattr(args, "role_per_node", False):
        node_roles = {concept: concept for concept in args.concepts}
    node_kinds: dict[str, str | None] = {c: args.kind for c in args.concepts}

    # Plan first -- no model load needed to learn there is nothing to do.
    try:
        plan = plan_discover_generation(
            folder, name, args.description,
            fit_mode="pca", labels=list(args.concepts),
            node_roles=node_roles, node_kinds=node_kinds,
        )
    except ManifoldFormatError as e:
        print(f"manifold generate failed: {e}", file=sys.stderr)
        sys.exit(1)

    n_total = len(plan.index_of)
    if not plan.pending:
        print(
            f"manifold {namespace}/{name} already complete ({n_total} nodes)"
            f" -- pass -f/--force to regenerate"
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
        f"generating {args.samples_per_prompt} response(s) per baseline prompt "
        f"for {len(plan.pending)} concept(s)..."
    )
    try:
        for concept in plan.pending:
            gen_roles: dict[str, str | None] | None = (
                {concept: concept} if node_roles else None
            )
            corpora = session.generate_responses(
                [concept], [args.kind],
                roles=gen_roles,
                samples_per_prompt=args.samples_per_prompt,
                on_progress=lambda m: print(f"  {m}"),
            )
            append_discover_manifold_node(
                folder, plan.index_of[concept], concept, corpora[concept],
            )
    except (ValueError, RuntimeError) as e:
        print(f"manifold generate failed: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"wrote {namespace}/{name} ({n_total} nodes)")
    print(f"  -> run `saklas manifold fit {namespace}/{name}` to fit")


def _run_manifold_from_template(args: argparse.Namespace) -> None:
    """``saklas manifold from-template`` — author a discover manifold from a template.

    Pure-IO (no model): resolves the standalone template, expands its values x
    contexts into per-value node corpora, and writes a discover folder that
    stores the corpus + ``template_ref``. Fit it next with
    ``saklas manifold fit <name> -m MODEL``.
    """
    from saklas.io.manifolds import ManifoldFormatError, create_manifold_from_template
    from saklas.io.templates import (
        AmbiguousTemplateError, TemplateNotFoundError, resolve_template,
    )

    try:
        tmpl = resolve_template(args.template)
    except (TemplateNotFoundError, AmbiguousTemplateError) as e:
        print(f"manifold from-template: {e}", file=sys.stderr)
        sys.exit(1)
    ref = f"{tmpl.path.parent.name}/{tmpl.name}" if tmpl.path else tmpl.name
    namespace, name = _split_manifold_ns_name(args.name or tmpl.name)

    hyperparams: dict[str, object] = {}
    if args.max_dim is not None:
        hyperparams["max_dim"] = args.max_dim
    if args.var_threshold is not None:
        hyperparams["var_threshold"] = args.var_threshold

    try:
        create_manifold_from_template(
            namespace, name, args.description,
            template_ref=ref,
            fit_mode=args.fit_mode,
            hyperparams=hyperparams or None,
            force=args.force,
        )
    except FileExistsError:
        print(
            f"manifold from-template: {namespace}/{name} already exists "
            f"-- pass -f/--force to overwrite",
            file=sys.stderr,
        )
        sys.exit(1)
    except ManifoldFormatError as e:
        print(f"manifold from-template failed: {e}", file=sys.stderr)
        sys.exit(1)

    print(
        f"wrote {namespace}/{name} ({len(tmpl.values)} nodes x "
        f"{len(tmpl.contexts)} contexts, template_ref={ref}, "
        f"fit_mode={args.fit_mode})"
    )
    print(f"  -> run `saklas manifold fit {namespace}/{name} -m MODEL` to fit")


def _normalize_context_entry(entry: object) -> dict[str, object]:
    """Accept either a multi-turn ``{turns, assistant}`` context or the
    single-turn sugar ``{user, assistant}`` and return the canonical shape."""
    if not isinstance(entry, dict):
        raise ValueError("each context must be an object")
    if "turns" in entry:
        return {"turns": entry["turns"], "assistant": entry.get("assistant")}
    if "user" in entry:                      # single-turn sugar
        return {
            "turns": [{"role": "user", "content": entry["user"]}],
            "assistant": entry.get("assistant"),
        }
    raise ValueError(
        "each context needs either 'turns' (multi-turn) or 'user' "
        "(single-turn sugar), plus 'assistant'"
    )


def _run_template_create(args: argparse.Namespace) -> None:
    """``saklas template create`` — author a standalone template artifact.

    Pure-IO (no model): reads the ``--contexts`` JSON file (multi-turn contexts
    or the single-turn ``{user, assistant}`` sugar), validates the slot/value
    invariants, and writes ``~/.saklas/templates/<ns>/<name>/template.json``.
    """
    import json as _json

    from saklas.io.templates import TemplateFormatError, create_template_folder

    try:
        with open(args.contexts_file) as f:
            raw = _json.load(f)
    except (OSError, ValueError) as e:
        print(f"template create: cannot read --contexts: {e}", file=sys.stderr)
        sys.exit(1)
    if not isinstance(raw, list) or not raw:
        print(
            "template create: --contexts must be a non-empty JSON list of "
            'contexts ({"turns": [...], "assistant": ...} or {"user": ..., '
            '"assistant": ...})',
            file=sys.stderr,
        )
        sys.exit(2)
    try:
        contexts = [_normalize_context_entry(e) for e in raw]
    except ValueError as e:
        print(f"template create: {e}", file=sys.stderr)
        sys.exit(2)

    namespace, name = _split_manifold_ns_name(args.name)
    try:
        create_template_folder(
            namespace, name,
            slot=args.slot,
            values=list(args.values),
            contexts=contexts,
            description=args.description,
            force=args.force,
        )
    except FileExistsError:
        print(
            f"template create: {namespace}/{name} already exists "
            f"-- pass -f/--force to overwrite",
            file=sys.stderr,
        )
        sys.exit(1)
    except TemplateFormatError as e:
        print(f"template create failed: {e}", file=sys.stderr)
        sys.exit(1)

    print(
        f"wrote template {namespace}/{name} ({len(args.values)} values x "
        f"{len(contexts)} contexts)"
    )
    print(f"  -> `saklas template score {namespace}/{name} -m MODEL` to read the "
          f"value distribution")


def _run_template_ls(args: argparse.Namespace) -> None:
    import json as _json

    from saklas.io.templates import iter_template_folders

    rows = list(iter_template_folders())
    if getattr(args, "json", False):
        print(_json.dumps([t.summary() for t in rows], indent=2))
        return
    if not rows:
        print("no templates installed under ~/.saklas/templates/")
        return
    for t in rows:
        ns = t.path.parent.name if t.path else "?"
        print(
            f"  {ns}/{t.name}  [slot {t.slot!r}, {len(t.values)} values x "
            f"{len(t.contexts)} contexts]"
        )
        if t.description:
            print(f"    {t.description}")


def _run_template_show(args: argparse.Namespace) -> None:
    import json as _json

    from saklas.io.templates import (
        AmbiguousTemplateError, TemplateNotFoundError, resolve_template,
    )

    try:
        t = resolve_template(args.name)
    except (TemplateNotFoundError, AmbiguousTemplateError) as e:
        print(f"template show: {e}", file=sys.stderr)
        sys.exit(1)
    if getattr(args, "json", False):
        payload = t.summary()
        payload["contexts"] = [
            {"turns": list(c.turns), "assistant": c.assistant} for c in t.contexts
        ]
        print(_json.dumps(payload, indent=2))
        return
    print(f"{t.name}  (slot {t.slot!r})")
    if t.description:
        print(f"  {t.description}")
    print(f"  values ({len(t.values)}): {', '.join(t.values)}")
    print(f"  contexts ({len(t.contexts)}):")
    for i, c in enumerate(t.contexts):
        turns = " / ".join(f"{turn['role']}: {turn['content']}" for turn in c.turns)
        print(f"    [{i}] {turns}")
        print(f"        assistant: {c.assistant}")


def _run_template_rm(args: argparse.Namespace) -> None:
    from saklas.io.templates import remove_template_folder

    namespace, name = _split_manifold_ns_name(args.name)
    if not args.yes:
        print(
            f"remove template {namespace}/{name}? pass -y/--yes to confirm",
            file=sys.stderr,
        )
        sys.exit(2)
    if remove_template_folder(namespace, name):
        print(f"Removed template {namespace}/{name}")
    else:
        print(f"template rm: no template {namespace}/{name}", file=sys.stderr)
        sys.exit(1)


def _run_template_score(args: argparse.Namespace) -> None:
    """``saklas template score`` — the restricted-choice value distribution."""
    import json as _json

    from saklas.core.scoring import score_template
    from saklas.core.session import SaklasSession
    from saklas.io.templates import (
        AmbiguousTemplateError, TemplateNotFoundError, resolve_template,
    )

    try:
        tmpl = resolve_template(args.name)
    except (TemplateNotFoundError, AmbiguousTemplateError) as e:
        print(f"template score: {e}", file=sys.stderr)
        sys.exit(1)

    with SaklasSession.from_pretrained(
        args.model, device=args.device, quantize=args.quantize,
    ) as session:
        per_ctx = score_template(session, tmpl, steering=args.steer)

    if getattr(args, "json", False):
        print(_json.dumps({
            "template": tmpl.name,
            "model": args.model,
            "steering": args.steer,
            "contexts": [sc.to_dict() for sc in per_ctx],
        }, indent=2))
        return

    by = getattr(args, "by", "sum")
    steer_note = f"  (steering: {args.steer})" if args.steer else ""
    print(f"{tmpl.name} on {args.model}{steer_note}")
    for i, sc in enumerate(per_ctx):
        print(f"\n  context [{i}]:")
        for c in sc.ranked(by=by):
            prob = c.prob_sum if by == "sum" else c.prob_mean
            bar = "#" * int(round(prob * 30))
            print(
                f"    {c.label:14s} P={prob:6.3f}  "
                f"sum={c.sum_logprob:8.3f}  mean={c.mean_logprob:8.3f}  "
                f"n={c.n_tokens}  {bar}"
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


def _run_pack_install(args: argparse.Namespace) -> None:
    from saklas.io.hf_manifolds import install_manifold

    dst = install_manifold(args.target, args.as_target, force=args.force)
    print(f"Installed {args.target} -> {dst}")


def _run_pack_search(args: argparse.Namespace) -> None:
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
        f"  -> run `saklas manifold fit {target_ns}/{target_name}` to fit"
    )


def _run_pack_push(args: argparse.Namespace) -> None:
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


def _run_pack_rm(args: argparse.Namespace) -> None:
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


def _run_pack_clear(args: argparse.Namespace) -> None:
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


def _run_pack_refresh(args: argparse.Namespace) -> None:
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

    Resolve the manifold folder, fit (or load) the per-layer Procrustes
    alignment between the source and target models' cached neutral activations
    — the part that needs both models loaded — then hand the prebuilt alignment
    dict to :func:`saklas.io.manifolds.transfer_manifold`, which is pure-io and
    only *applies* the map.  Steering vectors transfer the same way (a vector is
    the K=2 case of a flat ``pca`` manifold).
    """
    import json as _json

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

    M, quality_per_layer, _ = _load_or_fit_transfer_alignment(
        args.src_model, args.tgt_model, force=args.force, label="manifold transfer",
    )

    median_quality = median_or_zero(list(quality_per_layer.values())) if quality_per_layer else None

    # Target whitener for the Mahalanobis-share re-bake.  Read the target
    # neutral cache directly and center by its own per-layer mean — the
    # neutral mean *is* the probe-centering baseline, so no separate
    # layer_means cache is needed.  Mahalanobis is mandatory: a missing /
    # unusable target cache raises ``WhitenerError`` here (no Euclidean
    # rebake), surfaced to the user as a fatal transfer error.
    from saklas.core.mahalanobis import WhitenerError

    try:
        target_whitener = _target_whitener_from_neutral_cache(args.tgt_model)
    except WhitenerError as e:
        print(f"manifold transfer failed: {e}", file=sys.stderr)
        sys.exit(1)

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


def _run_pack_export(args: argparse.Namespace) -> None:
    """Export a fitted 2-node ``pca`` manifold to an interchange format (gguf).

    Folds the manifold down to a single steering direction
    (:func:`~saklas.core.vectors.folded_vector_directions`) and writes a
    llama.cpp control-vector GGUF.
    """
    fmt = getattr(args, "format", None)
    if fmt != "gguf":
        print(f"Unknown export format: {fmt}", file=sys.stderr)
        sys.exit(2)
    from saklas.io.cache_ops import _export_gguf_manifold

    ns, name = _resolve_manifold_ns_name(args.name)
    try:
        written = _export_gguf_manifold(
            ns, name,
            model_scope=args.model,
            output=args.output,
            model_hint=args.model_hint,
        )
    except (FileNotFoundError, RuntimeError) as e:
        print(f"manifold export failed: {e}", file=sys.stderr)
        sys.exit(1)
    for p in written:
        print(f"Wrote {p}")


# The nine steering-vector / manifold *compute* verbs (model-loading +
# analysis).  ``manifold transfer`` keeps its own runner (it was already a
# manifold verb); the rest are the renamed former ``subspace`` verbs plus the
# discover→fit fold.
_MANIFOLD_RUNNERS = {
    "extract":  _run_manifold_extract,
    "generate": _run_manifold_generate,
    "from-template": _run_manifold_from_template,
    "fit":      _run_manifold_fit,
    "bake":     _run_manifold_bake,
    "merge":    _run_manifold_merge,
    "transfer": _run_manifold_transfer,
    "compare":  _run_manifold_compare,
    "why":      _run_manifold_why,
}


@_saklas_error_exit
def _run_manifold(args: argparse.Namespace) -> None:
    """Dispatch ``saklas manifold <verb>`` (the compute verbs)."""
    cmd = getattr(args, "manifold_cmd", None)
    if cmd is None:
        print("usage: saklas manifold <verb> [...]")
        print()
        width = max(len(v) for v, _ in _MANIFOLD_VERBS)
        for v, desc in _MANIFOLD_VERBS:
            print(f"  {v:<{width}}  {desc}")
        print()
        print("Run `saklas manifold <verb> -h` for verb-specific options.")
        sys.exit(0)
    runner = _MANIFOLD_RUNNERS.get(cmd)
    if runner is None:
        print(f"unknown manifold verb {cmd!r}", file=sys.stderr)
        sys.exit(2)
    runner(args)


# The nine manifold *lifecycle* verbs (pure-IO over ~/.saklas/manifolds/,
# addressed by (namespace, name); install / share / inspect / remove).
_PACK_RUNNERS = {
    "ls":       _run_pack_ls,
    "show":     _run_pack_show,
    "install":  _run_pack_install,
    "search":   _run_pack_search,
    "push":     _run_pack_push,
    "rm":       _run_pack_rm,
    "clear":    _run_pack_clear,
    "refresh":  _run_pack_refresh,
    "export":   _run_pack_export,
}


@_saklas_error_exit
def _run_pack(args: argparse.Namespace) -> None:
    """Dispatch ``saklas pack <verb>`` (the manifold lifecycle verbs)."""
    cmd = getattr(args, "pack_cmd", None)
    if cmd is None:
        print("usage: saklas pack <verb> [...]")
        print()
        width = max(len(v) for v, _ in _PACK_VERBS)
        for v, desc in _PACK_VERBS:
            print(f"  {v:<{width}}  {desc}")
        print()
        print("Run `saklas pack <verb> -h` for verb-specific options.")
        sys.exit(0)
    runner = _PACK_RUNNERS.get(cmd)
    if runner is None:
        print(f"unknown pack verb {cmd!r}", file=sys.stderr)
        sys.exit(2)
    runner(args)


_TEMPLATE_RUNNERS = {
    "create": _run_template_create,
    "ls":     _run_template_ls,
    "show":   _run_template_show,
    "score":  _run_template_score,
    "rm":     _run_template_rm,
}


@_saklas_error_exit
def _run_template(args: argparse.Namespace) -> None:
    """Dispatch ``saklas template <verb>`` (the templated-completion artifact)."""
    cmd = getattr(args, "template_cmd", None)
    if cmd is None:
        print("usage: saklas template <verb> [...]")
        print()
        width = max(len(v) for v, _ in _TEMPLATE_VERBS)
        for v, desc in _TEMPLATE_VERBS:
            print(f"  {v:<{width}}  {desc}")
        print()
        print("Run `saklas template <verb> -h` for verb-specific options.")
        sys.exit(0)
    runner = _TEMPLATE_RUNNERS.get(cmd)
    if runner is None:
        print(f"unknown template verb {cmd!r}", file=sys.stderr)
        sys.exit(2)
    runner(args)


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
    "manifold":   _run_manifold,
    "pack":       _run_pack,
    "config":     _run_config,
    "experiment": _run_experiment,
    "template":   _run_template,
}
