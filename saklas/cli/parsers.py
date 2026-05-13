"""Argparse builders for the saklas CLI."""

from __future__ import annotations

import argparse


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _add_common_args(p: argparse.ArgumentParser) -> None:
    """Model-loading args shared between `tui` and `serve`."""
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


def _add_config_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("-c", "--config", action="append", default=None, metavar="PATH",
                   help="Load setup YAML (repeatable; later overrides earlier)")
    p.add_argument("-s", "--strict", action="store_true",
                   help="With -c: fail hard on missing vectors")


# ---------------------------------------------------------------------------
# Top-level parsers
# ---------------------------------------------------------------------------

_PACK_VERBS: list[tuple[str, str]] = [
    ("install",   "Install a concept pack from HF or a local folder"),
    ("refresh",   "Re-pull concept(s) from their source"),
    ("clear",     "Delete per-model tensors for matched concepts"),
    ("rm",        "Fully remove a concept folder"),
    ("ls",        "List locally installed concept packs"),
    ("search",    "Search the HuggingFace hub for concept packs"),
    ("push",      "Push a concept pack to HF as a model repo"),
    ("export",    "Export a pack to an interchange format (gguf)"),
]

_VECTOR_VERBS: list[tuple[str, str]] = [
    ("extract",   "Extract a steering vector for a concept"),
    ("merge",     "Merge existing vectors into a new pack"),
    ("clone",     "Clone a persona from a text corpus"),
    ("compare",   "Cosine similarity between steering vectors"),
    ("why",       "Show which layers contribute most to a steering vector"),
    ("transfer",  "Transfer a probe from one model to another via Procrustes"),
]


def _add_injection_args(p: argparse.ArgumentParser) -> None:
    """Steering-injection options shared between ``tui`` and ``serve``.

    ``None`` defaults flow through to the YAML override layer (or
    ultimately to the v2.1 session defaults: angular + π/2).
    """
    p.add_argument(
        "--steer-mode", dest="injection_mode",
        choices=["angular", "additive"], default=None,
        help="Steering injection math.  'angular' (default) maps user α "
             "to a rotation angle; 'additive' is the legacy v1.x add+"
             "rescale path.  Unset = inherit YAML / session default.",
    )
    p.add_argument(
        "--theta-max", dest="theta_max", type=float, default=None,
        metavar="RAD",
        help="Maximum rotation angle for angular mode (radians).  Default "
             "π/2 (≈1.5708) — α=1 fully aligns the residual with the "
             "concept direction.  No effect under --steer-mode additive.",
    )
    p.add_argument(
        "--projection-metric", dest="projection_metric",
        choices=["mahalanobis", "euclidean"], default=None,
        help="Metric for runtime ``~`` / ``|`` projection in steering "
             "expressions.  'mahalanobis' (default since v2.1) uses the "
             "closed-form LEACE projector against the per-model whitener "
             "(Belrose et al. 2023) — provably erases linearly-decodable "
             "concept information along ``onto`` from ``base``.  "
             "'euclidean' is plain Gram-Schmidt (the v2.0/v2.1 behavior).  "
             "Unset = inherit YAML / session default.",
    )
    p.add_argument(
        "--no-dls", dest="no_dls", action="store_true",
        help="Disable the discriminative-layer-selection mask at "
             "extraction time.  v2.1 introduced centered DLS (Dang & "
             "Ngo 2026, Eq. 9) as the default: layers where pos- and "
             "neg-class means project to the same side of the neutral "
             "baseline along ``d̂`` are dropped — they encode concept "
             "intensity rather than concept polarity.  Pass ``--no-dls`` "
             "to keep every layer (the v2.0–v2.1 behavior, modulo the "
             "removed ``edge_drop`` heuristic).  Mutually exclusive "
             "with ``--legacy`` (which already implies ``--no-dls``).",
    )
    p.add_argument(
        "--legacy", action="store_true",
        help="v2.0 backcompat preset for steering: equivalent to "
             "``--steer-mode additive`` plus PCA extraction on first-run "
             "probe bootstrap, Euclidean ``~`` / ``|`` projection, and "
             "DLS off (instead of v2.1's DiM + Mahalanobis bake + "
             "angular + LEACE projection + DLS).  Useful for "
             "A/B-comparing the pre-v2.1 stack on the same model.  "
             "Mutually exclusive with ``--steer-mode``, "
             "``--projection-metric``, and ``--no-dls``.",
    )
    p.add_argument(
        "--no-compile", dest="no_compile", action="store_true",
        help="Disable ``torch.compile``.  v2.2+ auto-enables compile on "
             "CUDA for kernel fusion (typically 1.2–1.5× decode tok/s on "
             "small models); on MPS/CPU compile is already a no-op.  "
             "Pass this to debug architecture-specific compile breakage "
             "or when running benchmarks against the eager baseline.  "
             "YAML equivalent: ``compile: false``.",
    )
    p.add_argument(
        "--no-cuda-graphs", dest="no_cuda_graphs", action="store_true",
        help="Disable ``transformers.StaticCache`` + CUDA-graph capture "
             "(Phase B, v2.2+).  When on, generation routes through "
             "fixed-shape K/V buffers so the compile-mode "
             "``reduce-overhead`` path can capture decode CUDA graphs "
             "internally — typically an additional 1.5–2.5× decode "
             "tok/s on small models on top of plain compile.  "
             "Auto-skipped on MPS/CPU and on architectures whose "
             "StaticCache constructor fails (logged once at session "
             "init).  Pass this when debugging cache-related issues "
             "or benchmarking against the DynamicCache baseline.  "
             "YAML equivalent: ``cuda_graphs: false``.",
    )


def _build_tui_parser(parser: argparse.ArgumentParser) -> None:
    # When a model supplies -c/--config pointing at a YAML with model: set,
    # the positional can be omitted. Handled in _run_tui via composed config.
    parser.add_argument("model", nargs="?", default=None,
                        help="HuggingFace model ID or local path")
    parser.add_argument("-q", "--quantize", choices=["4bit", "8bit"], default=None,
                        help="Quantization mode (default: bf16/fp16)")
    parser.add_argument("-d", "--device", default="auto",
                        help="Device: auto (detect), cuda, mps, or cpu")
    parser.add_argument("-p", "--probes", nargs="*", default=None,
                        help="Probe categories (default: all)")
    parser.add_argument("--max-tokens", type=int, default=1024,
                        help="Default max generation tokens")
    _add_injection_args(parser)
    _add_config_args(parser)


def _build_serve_parser(parser: argparse.ArgumentParser) -> None:
    _add_common_args(parser)
    parser.add_argument("-H", "--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("-P", "--port", type=int, default=8000, help="Bind port")
    parser.add_argument("-S", "--steer", default=None, metavar="EXPR",
                        help='Default steering expression, e.g. "0.5 honest + 0.3 warm"')
    parser.add_argument("-C", "--cors", action="append", default=[], metavar="ORIGIN",
                        help="CORS allowed origin (repeatable)")
    parser.add_argument("-k", "--api-key", default=None, metavar="KEY",
                        help="Require Bearer token auth; falls back to $SAKLAS_API_KEY")
    parser.add_argument("--no-web", dest="no_web", action="store_true",
                        help="Skip the analytics dashboard mount at / "
                             "(API-only mode for production / proxied deployments)")
    _add_injection_args(parser)
    _add_config_args(parser)


# --- pack subtree --------------------------------------------------------

def _build_pack_install(p: argparse.ArgumentParser) -> None:
    p.add_argument("target", help="<ns>/<concept>[@revision] or path to a concept folder")
    p.add_argument("-s", "--statements-only", action="store_true",
                   help="Keep statements.json only; drop any bundled tensors")
    p.add_argument("-a", "--as", dest="as_target", default=None, metavar="NS/NAME",
                   help="Relocate the installed pack under a different namespace/name")
    p.add_argument("-f", "--force", action="store_true",
                   help="Overwrite an existing installation")


def _build_pack_refresh(p: argparse.ArgumentParser) -> None:
    p.add_argument("selector", help="Selector or the literal 'neutrals'")
    p.add_argument("-m", "--model", default=None, metavar="MODEL_ID",
                   help="Scope to one model's tensors")


def _build_pack_clear(p: argparse.ArgumentParser) -> None:
    p.add_argument("selector", help="Selector (name, tag:x, namespace:x, default, all)")
    p.add_argument("-m", "--model", default=None, metavar="MODEL_ID",
                   help="Scope to one model's tensors only (default: all models)")
    p.add_argument("-y", "--yes", action="store_true",
                   help="Skip confirmation prompt on broad selectors")
    p.add_argument(
        "--variant", choices=["raw", "sae", "all"], default="all",
        help="Which tensor variant(s) to delete. Default: all.",
    )


def _build_pack_rm(p: argparse.ArgumentParser) -> None:
    p.add_argument("selector", help="Selector (name, tag:x, namespace:x, default, all)")
    p.add_argument("-y", "--yes", action="store_true",
                   help="Required for broad selectors (all, namespace:)")


def _build_pack_ls(p: argparse.ArgumentParser) -> None:
    p.add_argument("selector", nargs="?", default=None,
                   help="Optional selector (name, tag:x, namespace:x, default, all)")
    p.add_argument("-j", "--json", dest="json_output", action="store_true",
                   help="Emit machine-readable JSON instead of a table")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Include descriptions in the table output")


def _build_pack_search(p: argparse.ArgumentParser) -> None:
    p.add_argument("query", nargs="?", default="",
                   help="Search text (matched against HF model ids)")
    p.add_argument("-j", "--json", dest="json_output", action="store_true",
                   help="Emit machine-readable JSON instead of a table")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Include descriptions in the table output")


def _build_pack_push(p: argparse.ArgumentParser) -> None:
    p.add_argument("selector", help="Single concept selector (name or ns/name)")
    p.add_argument("-a", "--as", dest="as_target", default=None, metavar="OWNER/NAME")
    p.add_argument("-p", "--private", action="store_true")
    p.add_argument("-m", "--model", default=None, metavar="MODEL_ID")
    p.add_argument("-s", "--statements-only", action="store_true")
    p.add_argument("-n", "--no-statements", action="store_true")
    p.add_argument("-t", "--tag-version", action="store_true")
    p.add_argument("-d", "--dry-run", action="store_true")
    p.add_argument("-f", "--force", action="store_true")
    p.add_argument(
        "--variant", choices=["raw", "sae", "all"], default="raw",
        help="Which tensor variant(s) to push. Default: raw. (SAE variants "
             "carry different provenance; opt in via --variant sae|all.)",
    )


def _build_pack_export(p: argparse.ArgumentParser) -> None:
    sub = p.add_subparsers(dest="format", required=True)
    g = sub.add_parser("gguf", help="Export baked tensors to llama.cpp GGUF")
    g.add_argument("selector", help="Single concept selector (name or ns/name)")
    g.add_argument("-m", "--model", default=None, metavar="MODEL_ID")
    g.add_argument("-o", "--output", default=None, metavar="PATH")
    g.add_argument("--model-hint", default=None, metavar="HINT")




_PACK_BUILDERS = {
    "install": _build_pack_install,
    "refresh": _build_pack_refresh,
    "clear":   _build_pack_clear,
    "rm":      _build_pack_rm,
    "ls":      _build_pack_ls,
    "search":  _build_pack_search,
    "push":    _build_pack_push,
    "export":  _build_pack_export,
}


def _build_pack_parser(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="pack_cmd", required=False, metavar="VERB")
    for verb, desc in _PACK_VERBS:
        child = sub.add_parser(verb, help=desc, description=desc)
        _PACK_BUILDERS[verb](child)


# --- vector subtree ------------------------------------------------------

def _build_vector_extract(p: argparse.ArgumentParser) -> None:
    p.add_argument("concept", nargs="+",
                   help="Either one concept (e.g. 'happy.sad') or two poles (e.g. 'happy' 'sad')")
    p.add_argument("-m", "--model", default=None, metavar="MODEL_ID")
    p.add_argument("-f", "--force", action="store_true")
    p.add_argument(
        "--method", choices=["dim", "pca"], default=None,
        help="Extraction algorithm: 'dim' (difference-of-means, default) or "
             "'pca' (legacy contrastive PCA).  DiM is the v2.1+ default per "
             "Im & Li 2025; --method pca recovers the v1.x path and writes "
             "to the legacy ``_pca`` filename suffix for side-by-side "
             "comparison.  Unset = defer to YAML ``extraction_method:`` if "
             "configured, else 'dim'.",
    )
    p.add_argument(
        "--legacy", action="store_true",
        help="v2.0 backcompat preset.  On ``vector extract`` this is "
             "equivalent to ``--method pca``; combined with ``--legacy`` "
             "on ``tui``/``serve`` (additive injection) and ``vector "
             "compare`` (Euclidean cosine), it round-trips the entire "
             "pre-v2.1 stack.  Mutually exclusive with ``--method``.",
    )
    p.add_argument(
        "--sae", default=None, metavar="RELEASE",
        help="Extract via a SAELens SAE release (requires `pip install .[sae]`). "
             "No implicit default — you must name a release.",
    )
    p.add_argument(
        "--sae-revision", dest="sae_revision", default=None, metavar="REV",
        help="Pin a specific HF revision for the SAE release",
    )
    p.set_defaults(quantize=None, device="auto", probes=None)


def _build_vector_merge(p: argparse.ArgumentParser) -> None:
    p.add_argument("name", help="New pack name (written under local/)")
    p.add_argument(
        "expression",
        help=(
            'Merge expression, e.g. "0.3 ns/a + 0.4 ns/b" or '
            '"0.5 ns/a~ns/b" for projection-removal.'
        ),
    )
    p.add_argument("-f", "--force", action="store_true")
    p.add_argument("-s", "--strict", action="store_true")
    p.add_argument("-m", "--model", default=None, metavar="MODEL_ID")


def _build_vector_clone(p: argparse.ArgumentParser) -> None:
    p.add_argument("corpus_path", help="Path to a UTF-8 text file, one utterance per line")
    p.add_argument("-N", "--name", required=True, help="Persona identifier (stored under local/<name>)")
    p.add_argument("-m", "--model", default=None, metavar="MODEL_ID")
    p.add_argument("-n", "--n-pairs", dest="n_pairs", type=int, default=90)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("-f", "--force", action="store_true")
    p.set_defaults(quantize=None, device="auto", probes=None)


def _build_vector_compare(p: argparse.ArgumentParser) -> None:
    p.add_argument("concepts", nargs="+",
                   help="One or more concept selectors (names, tag:x, namespace:x, all)")
    p.add_argument("-m", "--model", required=True, metavar="MODEL_ID",
                   help="Model id (used to locate baked tensors)")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Show per-layer breakdown (2-arg pairwise mode)")
    p.add_argument("-j", "--json", dest="json_output", action="store_true",
                   help="Emit machine-readable JSON")
    p.add_argument(
        "--metric", choices=("euclidean", "mahalanobis"), default=None,
        help=(
            "Cosine metric. 'mahalanobis' (default since v2.1) = whitened "
            "cosine ⟨u,v⟩_M = u^T Σ^{-1} v (Belrose et al. 2023), reads "
            "cached neutral activations + layer means under "
            "~/.saklas/models/<id>/ to build the per-layer whitener; "
            "falls back to Euclidean per layer when the whitener doesn't "
            "cover that layer.  'euclidean' = standard cosine (the "
            "v2.0/v2.1 behavior; selected by ``--legacy``)."
        ),
    )
    p.add_argument(
        "--ridge-scale", type=float, default=1.0, metavar="FLOAT",
        help=(
            "Ridge multiplier on the regularized covariance "
            "(λ_L = (||X_L||_F²/(N·D)) × ridge_scale). Only consulted "
            "with --metric mahalanobis; default 1.0 (mean diagonal of "
            "the un-regularized sample covariance)."
        ),
    )
    p.add_argument(
        "--legacy", action="store_true",
        help=(
            "v2.0 backcompat preset: equivalent to ``--metric euclidean``."
            "  Mutually exclusive with ``--metric``."
        ),
    )


def _build_vector_why(p: argparse.ArgumentParser) -> None:
    p.add_argument("concept", help="Concept selector (name or ns/name)")
    p.add_argument("-m", "--model", required=True, metavar="MODEL_ID",
                   help="Model id (used to locate the baked tensor)")
    p.add_argument("-j", "--json", dest="json_output", action="store_true",
                   help="Emit machine-readable JSON (full per-layer detail)")


def _build_vector_transfer(p: argparse.ArgumentParser) -> None:
    """``saklas vector transfer`` — cross-model probe alignment.

    Required:
        ``concept`` — selector resolving to a single concept folder.
        ``--from`` — HF coord of the source model (must already have a
        baked tensor for the concept under ~/.saklas/vectors/...).
        ``--to`` — HF coord of the target model (the alignment is fit
        between these two using cached neutral activations).

    Behavior: writes a transferred tensor at the target model's
    ``_from-<safe_src>`` variant path, with a sidecar carrying transfer
    provenance (``method=procrustes_transfer``, ``source_model_id``,
    ``alignment_map_hash``, ``transfer_quality_estimate``).  Reuses the
    same tensor-filename machinery as SAE variants, so subsequent
    ``saklas pack ls`` / ``saklas vector why`` see the transferred
    profile alongside any native or SAE variants.

    Cached alignment maps live at
    ``~/.saklas/models/<safe_tgt>/alignments/<safe_src>.{safetensors,json}``;
    ``--force`` recomputes even when the cache hits.
    """
    p.add_argument("concept", help="Concept selector (name or ns/name)")
    p.add_argument("--from", dest="src_model", required=True, metavar="SRC_MODEL",
                   help="Source model id (where the probe was extracted)")
    p.add_argument("--to", dest="tgt_model", required=True, metavar="TGT_MODEL",
                   help="Target model id (where the transferred probe will live)")
    p.add_argument("-f", "--force", action="store_true",
                   help="Recompute alignment + transfer even when cached")
    p.add_argument("-j", "--json", dest="json_output", action="store_true",
                   help="Emit machine-readable JSON (path + quality summary)")


_VECTOR_BUILDERS = {
    "extract":  _build_vector_extract,
    "merge":    _build_vector_merge,
    "clone":    _build_vector_clone,
    "compare":  _build_vector_compare,
    "why":      _build_vector_why,
    "transfer": _build_vector_transfer,
}


def _build_vector_parser(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="vector_cmd", required=False, metavar="VERB")
    for verb, desc in _VECTOR_VERBS:
        child = sub.add_parser(verb, help=desc, description=desc)
        _VECTOR_BUILDERS[verb](child)


# --- config subtree ------------------------------------------------------

def _build_config_parser(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="config_cmd", required=False, metavar="VERB")

    show = sub.add_parser("show", help="Print the effective merged config")
    show.add_argument("-c", "--config", action="append", default=None, metavar="PATH",
                      help="Extra YAML files to compose on top of ~/.saklas/config.yaml")
    show.add_argument("-m", "--model", default=None,
                      help="Override model field in output")
    show.add_argument("--no-default", action="store_true",
                      help="Skip loading ~/.saklas/config.yaml")

    validate = sub.add_parser("validate", help="Validate a config file (CI hook)")
    validate.add_argument("file", help="Path to YAML config file")


def _build_root_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="saklas",
        description="Activation steering + trait monitoring for local HuggingFace models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Run `saklas <verb> -h` for verb-specific options.",
    )
    sub = root.add_subparsers(dest="command", required=False, metavar="VERB")

    # Each ``help=`` here is the single source of truth — it lands in
    # the auto-generated ``positional arguments`` table on ``saklas -h``.
    # Keep it short enough to fit one line at the typical terminal
    # width; the verb's own ``-h`` carries the long-form description.
    tui = sub.add_parser(
        "tui",
        help="Launch the interactive TUI (requires <model>)",
        description="Launch the interactive TUI",
    )
    _build_tui_parser(tui)

    serve = sub.add_parser(
        "serve",
        help="Start the OpenAI + Ollama API server + analytics dashboard at /",
        description="Start the OpenAI + Ollama compatible API server",
    )
    _build_serve_parser(serve)

    pack = sub.add_parser(
        "pack",
        help="Manage concept packs (install/ls/search/push/refresh/...)",
        description="Manage concept packs",
    )
    _build_pack_parser(pack)

    vector = sub.add_parser(
        "vector",
        help="Vector operations (extract/merge/clone/compare/why/transfer)",
        description="Vector operations (extract/merge/clone/compare/why/transfer)",
    )
    _build_vector_parser(vector)

    cfg = sub.add_parser(
        "config",
        help="Inspect and validate saklas config files",
        description="Inspect/validate config",
    )
    _build_config_parser(cfg)

    return root
