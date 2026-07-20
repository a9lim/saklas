"""Shared helpers for the saklas CLI runners.

Session construction, effective-config loading, steering setup, and the
manifold namespace/name resolvers used across more than one verb group.
"""

from __future__ import annotations

import argparse
import functools
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Callable, TypeVar

from saklas.core.errors import SaklasError

if TYPE_CHECKING:
    from saklas.core.session import SaklasSession
    from saklas.core.steering import Steering


_R = TypeVar("_R")


def _saklas_error_exit(fn: Callable[..., _R]) -> Callable[..., _R]:
    """Translate any ``SaklasError`` escaping a runner to a stderr line + exit.

    Maps the exception's HTTP-style status (from ``user_message()``) to a
    process exit code via ``min(2, code // 100)``: 4xx/5xx land on exit 2,
    nothing softer.
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


def _make_session(args: argparse.Namespace, *, load_probes: bool = True):
    from saklas.core.session import SaklasSession
    probe_categories = _resolve_probes(args.probes) if load_probes else []
    # ``~`` / ``|`` projection is Mahalanobis-only (closed-form LEACE);
    # ``--no-dls`` opts out of the discriminative-layer mask.  The flag and
    # YAML are already merged onto ``args`` by ``_load_effective_config``.
    dls = not bool(getattr(args, "no_dls", False))
    # ``--compile`` opts into CUDA/MPS compilation; ``--cuda-graphs`` adds
    # the CUDA-only StaticCache/graph path. YAML values are folded onto these
    # argparse fields by ``_load_effective_config``.
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
    :class:`Steering` with every atom pre-warmed in ``session.profiles``.
    Returns ``None`` when ``expression`` is empty / falsy.
    """
    from saklas.io.selectors import canonicalize_atom, AmbiguousSelectorError
    from saklas.core.steering_expr import (
        parse_expr, referenced_selectors,
    )

    if not expression:
        return None

    for ns, concept, _variant in referenced_selectors(expression):
        raw_name = concept
        display = f"{ns}/{concept}" if ns else concept
        try:
            canonical, _variant = canonicalize_atom(raw_name)
        except AmbiguousSelectorError as e:
            if verbose:
                print(f"  Failed to resolve '{raw_name}': {e}", file=sys.stderr)
                sys.exit(1)
            print(f"  Failed to register '{display}': {e}", file=sys.stderr)
            continue
        try:
            if verbose:
                print(f"Extracting steering vector: {canonical}")
                _, profile = session.extract(
                    canonical, on_progress=lambda m: print(f"  {m}"),
                    namespace=ns,
                )
            else:
                _, profile = session.extract(canonical, namespace=ns)
        except Exception as e:
            if verbose:
                raise
            print(f"  Failed to register '{display}': {e}", file=sys.stderr)
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
