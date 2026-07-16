"""``saklas serve`` runner and its live-readout startup policy helpers."""

from __future__ import annotations

import argparse
import sys
from typing import Any, Sequence

import saklas.cli.runners as _pkg
from saklas.cli.runners.shared import (
    _saklas_error_exit, _setup_steering_vectors, _warmup_session,
)


@_saklas_error_exit
def _enable_serve_live_lens_if_compatible(session: Any) -> bool:
    """Apply serve's default-live policy to an active or cached lens."""
    try:
        compatible = session.has_compatible_jlens()
        model_id = getattr(session, "model_id", None)
        if not compatible and isinstance(model_id, str) and model_id:
            from saklas.io.lens_sources import list_lens_sources

            # An older cache can predate active-source pointers. Prefer a
            # prepared local lens, then another cached provider binding, and
            # adopt the first one whose identity matches the loaded weights.
            rows = sorted(
                list_lens_sources(model_id),
                key=lambda row: (not bool(row.get("active")), row.get("kind") != "local"),
            )
            for row in rows:
                source = row.get("source")
                if not isinstance(source, str) or not source:
                    continue
                try:
                    session.select_jlens_source(source)
                    compatible = session.has_compatible_jlens()
                except Exception:  # noqa: BLE001 — try the next cached source
                    continue
                if compatible:
                    break
        if not compatible:
            return False
        layers = session.enable_live_lens()
        print(f"Live J-lens readout: on ({len(layers)} fitted layers)")
        return True
    except Exception as e:  # noqa: BLE001 — never block serve startup
        print(f"Live J-lens readout: enable failed ({e})", file=sys.stderr)
        return False


def _best_serve_sae_release(rows: Sequence[dict[str, Any]]) -> str | None:
    """Choose the strongest provider default for the dashboard runtime.

    Provider-hosted, labelled, official releases win; canonical releases beat
    broad variant sets, and a curated base release beats its ``-all`` sibling.
    The final name tie-break keeps the choice stable across registry ordering.
    Local artifacts remain explicit user choices unless one is already active.
    """
    candidates = [
        row for row in rows
        if row.get("source") == "saelens"
        and isinstance(row.get("release"), str)
        and bool(row["release"].strip())
    ]
    if not candidates:
        return None

    def rank(row: dict[str, Any]) -> tuple[bool, bool, bool, bool, str]:
        release = str(row["release"])
        repo_id = row.get("repo_id")
        official = isinstance(repo_id, str) and repo_id.startswith("google/")
        return (
            not official,
            not bool(row.get("neuronpedia")),
            "canonical" not in release,
            release.endswith("-all"),
            release,
        )

    return str(min(candidates, key=rank)["release"])


@_saklas_error_exit
def _enable_serve_live_sae_if_available(session: Any) -> bool:
    """Attach a cached or best-provider SAE and start its live view."""
    try:
        info = session.sae_info
        if info is None:
            from saklas.io.sae import list_sae_sources

            # Prepared artifacts are the cheapest and most intentional
            # default. This also repairs caches created before active.json.
            rows = sorted(
                list_sae_sources(session.model_id),
                key=lambda row: (not bool(row.get("active")), row.get("kind") != "local"),
            )
            for row in rows:
                source = row.get("source")
                if not isinstance(source, str) or not source:
                    continue
                release = source.removeprefix("saelens:")
                try:
                    info = session.load_sae(release)
                except Exception:  # noqa: BLE001 — try the next cached source
                    continue
                break
        if info is None:
            from saklas.core.sae import list_sae_releases

            release = _best_serve_sae_release(list_sae_releases(session.model_id))
            if release is None:
                return False
            info = session.load_sae(release)
        state = session.enable_live_sae(top_k=12)
        print(
            "Live SAE readout: on "
            f"({info.get('release', 'active')} at L{state.get('layer')})"
        )
        return True
    except Exception as e:  # noqa: BLE001 — never block serve startup
        print(f"Live SAE readout: enable failed ({e})", file=sys.stderr)
        return False


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

    _pkg._load_effective_config(args)
    if not args.model:
        print(
            "saklas serve: model required. Pass a HuggingFace repo id (e.g.\n"
            "  saklas serve google/gemma-2-2b-it\n"
            "or supply it via -c setup.yaml with a `model:` field.",
            file=sys.stderr,
        )
        sys.exit(2)

    _pkg._print_startup(args)
    session = _pkg._make_session(args)
    _pkg._print_model_info(session)

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

    # Live-lens-on by default: when the model has a fitted Jacobian lens,
    # serve starts with the full fitted readout streaming, so
    # the dashboard's J-LENS tab is hot on first load (the toggle still
    # turns it off per session). Serve-side policy only — the library stays
    # opt-in. The generation's logit-alternative K also controls
    # the J-lens readout width.
    _enable_serve_live_lens_if_compatible(session)

    # Web dashboard policy: attach the strongest compatible provider SAE (or
    # preserve an explicitly active local/provider source) and turn its live
    # discovery view on. API-only ``--no-web`` serves do not acquire this
    # optional provider dependency or download weights implicitly.
    if web_enabled:
        _enable_serve_live_sae_if_available(session)

    print(f"\nServing on http://{args.host}:{args.port}")
    print(f"OpenAI-compatible:  http://{args.host}:{args.port}/v1")
    print(f"Ollama-compatible:  http://{args.host}:{args.port}/api")
    print(f"API docs:           http://{args.host}:{args.port}/docs")
    if args.port != 11434:
        print("Tip: for drop-in Ollama compatibility, run with `--port 11434`.")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
