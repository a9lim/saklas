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
        help="Probe categories: all, none, affect, epistemic, alignment, register, social_stance, cultural, identity (default: all)",
    )


def _add_config_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("-c", "--config", action="append", default=None, metavar="PATH",
                   help="Load setup YAML (repeatable; later overrides earlier)")
    p.add_argument("-s", "--strict", action="store_true",
                   help="With -c: fail hard on missing vectors")


def _add_logit_args(p: argparse.ArgumentParser) -> None:
    """Logit-capture options shared between ``tui`` and ``serve``.

    Phase 1 of the logit pass: ``--top-k-alts`` sets the session-level
    default for ``SamplingConfig.return_top_k`` — the number of top-K
    alternatives the engine decodes per generated token (with text). K=0
    (the default) means "logprob only", a near-free addition on top of
    the existing log_softmax in the loom path. K>0 enables the
    distributional surfaces (drilldown logits tab, inline surprise tint,
    NodeCompareDrawer logit columns); ~60 KB/turn on the wire at K=8.
    Per-call ``SamplingConfig.return_top_k > 0`` overrides; K=0 inherits.
    YAML equivalent: ``return_top_k:`` int in ``[0, 256]``.
    """
    p.add_argument(
        "--top-k-alts", dest="top_k_alts", type=int, default=None, metavar="N",
        help="Session default for top-K alternatives capture (0–256). "
             "0 (default) = chosen-token logprob only; N>0 ships top-N "
             "decoded alternatives per token for distributional surfaces. "
             "Unset = inherit YAML ``return_top_k:`` / session default.",
    )


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
    ("manifold",  "Fit and inspect spline-based steering manifolds"),
]

# Verb table for ``saklas vector manifold <verb>`` — the source of truth for
# both parser registration order and the bare-verb help block (``_run_vector_
# manifold``), mirroring how ``_VECTOR_VERBS`` / ``_PACK_VERBS`` drive theirs.
_MANIFOLD_VERBS: list[tuple[str, str]] = [
    ("fit",       "Fit an authored manifold (user-supplied coords)"),
    ("discover",  "Fit a discover-mode manifold (coords derived from activations)"),
    ("generate",  "Author a discover-mode manifold from a concept list"),
    ("ls",        "List installed manifolds"),
    ("show",      "Show a manifold's nodes and fitted models"),
    ("install",   "Install a manifold from HF or a local folder"),
    ("search",    "Search the HuggingFace hub for manifolds"),
    ("merge",     "Union discover-mode manifolds' nodes into a new manifold"),
    ("push",      "Push a manifold to HF as a model repo"),
    ("rm",        "Fully remove a manifold folder"),
    ("clear",     "Delete per-model fitted tensors for a manifold"),
    ("refresh",   "Re-pull / re-materialize a manifold from its source"),
    ("transfer",  "Transfer a manifold to another model via Procrustes"),
]

_EXPERIMENT_VERBS: list[tuple[str, str]] = [
    ("fan",         "Run an alpha grid as one experiment"),
    ("transcript",  "Replay or inspect saved transcript paths"),
    ("naturalness", "Score a steered generation's behavior-manifold naturalness"),
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
        "--compile", dest="compile", action="store_true",
        help="Enable ``torch.compile`` on CUDA.  Off by default — the "
             "compile path is intermittently broken on torch 2.12 for "
             "newer architectures (Gemma-4, Qwen3.5 hit inductor "
             "codegen bugs), and on interactive workloads the ~25–50s "
             "compile cost rarely pays off against the 1.2–3× per-token "
             "speedup it delivers when it works.  Pass this for "
             "sustained workloads (long-running serve, batch eval) "
             "where the upfront cost amortizes.  On MPS/CPU compile is "
             "already a no-op.  YAML equivalent: ``compile: true``.",
    )
    p.add_argument(
        "--cuda-graphs", dest="cuda_graphs", action="store_true",
        help="Enable ``transformers.StaticCache`` + CUDA-graph capture "
             "for additional decode speedup (1.5–2.5× on small models "
             "on top of plain ``--compile``).  Off by default for the "
             "same reason as ``--compile``.  Requires ``--compile`` to "
             "be useful — when on, ``torch.compile(mode=\"reduce-"
             "overhead\")`` captures decode CUDA graphs internally.  "
             "Auto-skipped on MPS/CPU and on architectures whose "
             "StaticCache constructor fails (logged once at session "
             "init).  YAML equivalent: ``cuda_graphs: true``.",
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
    _add_logit_args(parser)
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
    _add_logit_args(parser)
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
        "--sae", default=None, metavar="RELEASE",
        help="Extract via a SAELens SAE release (requires `pip install .[sae]`). "
             "No implicit default — you must name a release.",
    )
    p.add_argument(
        "--sae-revision", dest="sae_revision", default=None, metavar="REV",
        help="Pin a specific HF revision for the SAE release",
    )
    p.add_argument(
        "--role", default=None, metavar="SLUG",
        help="Role-augmented extraction: render pairs under a chat template "
             "whose assistant-role label is replaced by SLUG (e.g. 'pirate'). "
             "Writes the tensor under a ``_role-<slug>`` filename suffix; "
             "steer it with the matching ``:role-<slug>`` variant in any "
             "expression. Slug must match ``[a-z0-9._-]+``. Mutually "
             "exclusive with ``--sae``. Mistral-3 / talkie families don't "
             "carry a substitutable role label and raise at runtime.",
    )
    p.add_argument(
        "--namespace", default=None, metavar="NS",
        help="Destination namespace for the extracted vector folder.  "
             "Unset lands the tensor under "
             "``~/.saklas/vectors/local/<canonical>/`` — the historical "
             "landing site.  Any other value relocates to "
             "``vectors/<namespace>/<canonical>/``.  Parity with the "
             "webui ExtractDrawer's namespace control and with "
             "``vector manifold`` / `discover``'s NS slot.",
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
            "decided all-or-nothing — whitens every shared layer or, when "
            "the whitener doesn't cover them all, falls back to Euclidean "
            "for all.  'euclidean' = standard cosine (the v2.0/v2.1 "
            "behavior; selected by ``--legacy``)."
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


def _build_vector_manifold(parser: argparse.ArgumentParser) -> None:
    """``saklas vector manifold`` — fit, discover, generate, and inspect.

    Nested subparser:
      ``fit`` — fit an authored manifold (user-supplied coords).
      ``discover`` — fit a discover-mode manifold (coords derived from
        per-node activations via PCA or spectral embedding).
      ``generate`` — author a discover-mode manifold folder by asking
        the loaded model for per-concept statement corpora.
      ``ls`` / ``show`` — pure-IO discovery + inspection.

    ``fit`` / ``discover`` / ``generate`` load a model; ``ls`` / ``show``
    are pure-IO over ``~/.saklas/manifolds/``.
    """
    sub = parser.add_subparsers(dest="manifold_cmd", required=False, metavar="VERB")

    fit = sub.add_parser(
        "fit",
        help="Fit a manifold for a model from an authored corpus folder",
        description=(
            "Pool per-node centroids, fit a per-layer PCA subspace + RBF "
            "interpolant, and write the per-model manifold tensor into the "
            "folder."
        ),
    )
    fit.add_argument("folder", help="Path to an authored manifold folder")
    fit.add_argument("-m", "--model", default=None, metavar="MODEL_ID")
    fit.add_argument(
        "--sae", default=None, metavar="RELEASE",
        help="Fit in an SAELens SAE feature space (requires `.[sae]`); "
             "centroids are reconstructed through the SAE before the fit.",
    )
    fit.add_argument(
        "--sae-revision", dest="sae_revision", default=None, metavar="REV",
        help="Pin a specific HF revision for the SAE release",
    )
    fit.set_defaults(quantize=None, device="auto", probes=None)

    discover = sub.add_parser(
        "discover",
        help="Fit a discover-mode manifold (coords derived from activations)",
        description=(
            "Pool per-node centroids, derive node coordinates via PCA or "
            "spectral embedding, fit a per-layer RBF, and write the per-model "
            "manifold tensor.  Operates on an existing discover-mode manifold "
            "folder (usually authored by `saklas vector manifold generate`).  "
            "PCA is the safe default for current bundled-heap sizes (~20–48 "
            "nodes); spectral is the right choice once heaps cross ~50 nodes "
            "and start to hint at curved structure — below that the spectral "
            "gap collapses into the eigenvalue noise floor and the layout "
            "looks meaningful but isn't."
        ),
    )
    discover.add_argument("name", help="Manifold name (or ns/name)")
    discover.add_argument("-m", "--model", default=None, metavar="MODEL_ID")
    discover.add_argument(
        "--method", choices=("pca", "spectral"), default=None,
        help="Override the folder's fit_mode (default: keep folder's setting)",
    )
    discover.add_argument(
        "--max-dim", dest="max_dim", type=int, default=None, metavar="N",
        help="Cap intrinsic dimension (default: 8 for PCA, 8 for spectral)",
    )
    discover.add_argument(
        "--var-threshold", dest="var_threshold", type=float, default=None,
        metavar="T",
        help="PCA: smallest cumulative-variance prefix that crosses T "
             "is picked (default: 0.70)",
    )
    discover.add_argument(
        "--k-nn", dest="k_nn", type=int, default=None, metavar="K",
        help="Spectral: number of nearest neighbors in the k-NN graph "
             "(default: max(5, ceil(log K)))",
    )
    discover.add_argument(
        "--bandwidth", type=float, default=None, metavar="SIGMA",
        help="Spectral: heat-kernel bandwidth (default: median k-NN distance)",
    )
    discover.add_argument(
        "--max-subspace-dim", dest="max_subspace_dim", type=int, default=None,
        metavar="R",
        help="Per-layer PCA subspace dim cap (default 64). Smaller values "
             "give finer-grained steering control at large K — each axis the "
             "RBF can move along is an axis subspace_replace can displace, "
             "so fewer axes = smaller per-α effect = wider coherence regime. "
             "Recommended: set near the manifold's intrinsic dim (=picked_k) "
             "for steering use; keep at 64 for representational analysis.",
    )
    discover.add_argument(
        "--sae", default=None, metavar="RELEASE",
        help="Reconstruct centroids through an SAELens SAE before the fit "
             "(requires `.[sae]`)",
    )
    discover.add_argument(
        "--sae-revision", dest="sae_revision", default=None, metavar="REV",
    )
    discover.set_defaults(quantize=None, device="auto", probes=None)

    generate = sub.add_parser(
        "generate",
        help="Generate per-concept corpora and write a discover-mode manifold",
        description=(
            "Ask the loaded model for shared situational scenarios that have "
            "purchase for every concept on the list, then for each (scenario, "
            "concept) cell write K first-person statements as a literal "
            "instance of that concept under that scenario.  Writes a fresh "
            "discover-mode manifold folder under "
            "~/.saklas/manifolds/<ns>/<name>/ ready for `vector manifold "
            "discover` to fit.  Scenario sharing across the row is "
            "load-bearing — statement j of every concept came from the same "
            "scenario, so the per-concept centroids stay comparable."
        ),
    )
    generate.add_argument("name", help="Manifold name (use ns/name for non-local)")
    generate.add_argument(
        "--concepts", nargs="+", required=True, metavar="CONCEPT",
        help="Concept slugs to generate corpora for (>= 2)",
    )
    generate.add_argument(
        "--n-scenarios", dest="n_scenarios", type=int, default=9,
        metavar="N",
        help="Number of shared scenarios (default: 9)",
    )
    generate.add_argument(
        "--statements-per-concept", dest="statements_per_concept",
        type=int, default=5, metavar="K",
        help="Statements per (concept, scenario) cell (default: 5)",
    )
    generate.add_argument(
        "--description", default="", metavar="TEXT",
        help="Human-readable description for the manifold folder",
    )
    generate.add_argument(
        "--seed", type=int, default=None, metavar="INT",
        help="Seed the statement generation for reproducible corpora "
             "(parity with `vector clone --seed`; default: unseeded)",
    )
    generate.add_argument(
        "--role-per-node", dest="role_per_node", action="store_true",
        help=(
            "Role-augmented (persona) manifold: use each --concepts slug "
            "as that node's assistant-role substitution at fit time.  The "
            "fitted manifold lives in persona-baseline activation space "
            "and steering through it implies the nearest node's role at "
            "decode time (role-paired manifold steering).  Slugs must "
            "match [a-z0-9._-]+; family must carry a substitutable role "
            "header (Qwen / Gemma / Llama / GLM / gpt-oss — Mistral-3 / "
            "talkie raise at fit time)."
        ),
    )
    generate.add_argument(
        "-f", "--force", action="store_true",
        help="Overwrite an existing manifold folder",
    )
    generate.add_argument("-m", "--model", default=None, metavar="MODEL_ID")
    generate.set_defaults(quantize=None, device="auto", probes=None)

    ls = sub.add_parser(
        "ls", help="List installed manifolds",
        description="List local manifolds under ~/.saklas/manifolds/.",
    )
    ls.add_argument("--namespace", default=None, metavar="NS",
                    help="Restrict to a single namespace")
    ls.add_argument("-j", "--json", dest="json_output", action="store_true")
    ls.add_argument("-v", "--verbose", action="store_true",
                    help="Include descriptions in the table output")

    show = sub.add_parser(
        "show", help="Show a manifold's nodes and fitted models",
        description="Print one manifold's domain, node coordinates, and "
                    "per-model fitted tensors.",
    )
    show.add_argument("name", help="Manifold name (or ns/name)")
    show.add_argument("-j", "--json", dest="json_output", action="store_true")

    install = sub.add_parser(
        "install",
        help="Install a manifold from HF or a local folder",
        description=(
            "Pull a manifold from a HuggingFace saklas-manifold repo "
            "(`<ns>/<name>[@revision]`) or copy-install a local folder "
            "path.  The manifold analogue of `saklas pack install`."
        ),
    )
    install.add_argument(
        "target",
        help="<ns>/<name>[@revision] or path to a manifold folder",
    )
    install.add_argument(
        "-a", "--as", dest="as_target", default=None, metavar="NS/NAME",
        help="Relocate the installed manifold under a different "
             "namespace/name (must be fully qualified)",
    )
    install.add_argument(
        "-f", "--force", action="store_true",
        help="Overwrite an existing installation",
    )

    search = sub.add_parser(
        "search",
        help="Search the HuggingFace hub for manifolds",
        description=(
            "Search HF for `saklas-manifold`-tagged model repos.  The "
            "manifold analogue of `saklas pack search`."
        ),
    )
    search.add_argument(
        "query", nargs="?", default="",
        help="Search text (matched against HF model ids)",
    )
    search.add_argument(
        "-j", "--json", dest="json_output", action="store_true",
        help="Emit machine-readable JSON instead of a table",
    )
    search.add_argument(
        "-v", "--verbose", action="store_true",
        help="Include descriptions in the table output",
    )

    merge = sub.add_parser(
        "merge",
        help="Union discover-mode manifolds' nodes into a new manifold",
        description=(
            "Union the *nodes* of two or more discover-mode source "
            "manifolds into a fresh, unfitted discover folder, then run "
            "`saklas vector manifold discover <merged>` to fit it.  The "
            "manifold analogue of `saklas vector merge` — but on node "
            "corpora rather than steering directions.  Restricted to "
            "discover-mode sources (authored manifolds carry user-declared "
            "geometry that isn't mergeable without a shared coordinate "
            "system).  Label collisions across sources raise; rename one "
            "side before merging."
        ),
    )
    merge.add_argument("name", help="New manifold name (or ns/name)")
    merge.add_argument(
        "sources", nargs="+", metavar="SOURCE",
        help="Two or more discover-mode source manifolds (name or ns/name)",
    )
    merge.add_argument(
        "--description", default="", metavar="TEXT",
        help="Human-readable description for the merged manifold folder",
    )
    merge.add_argument(
        "--method", choices=("pca", "spectral"), default=None,
        help="Override the merged fit_mode (default: the sources' shared "
             "mode; required when sources disagree)",
    )
    merge.add_argument(
        "-f", "--force", action="store_true",
        help="Overwrite an existing manifold folder",
    )

    push = sub.add_parser(
        "push",
        help="Push a manifold to HF as a model repo",
        description=(
            "Push a manifold folder (corpus + fitted tensors) to HF as a "
            "`saklas-manifold`-tagged model repo.  The manifold analogue "
            "of `saklas pack push`.  The corpus is always uploaded (a "
            "manifold can't re-fit without it); per-model tensors are "
            "filtered by `-m`/`--variant`."
        ),
    )
    push.add_argument("selector", help="Single manifold (name or ns/name)")
    push.add_argument(
        "-a", "--as", dest="as_target", default=None, metavar="OWNER/NAME",
        help="Target HF coord (default: <whoami>/<name>)",
    )
    push.add_argument(
        "-m", "--model", default=None, metavar="MODEL_ID",
        help="Restrict the pushed tensors to one base model",
    )
    push.add_argument("-p", "--private", action="store_true")
    push.add_argument("-d", "--dry-run", action="store_true")
    push.add_argument(
        "--variant", choices=["raw", "sae", "all"], default="raw",
        help="Which tensor variant(s) to push. Default: raw. (SAE variants "
             "carry stronger provenance, so sharing them is opt-in.)",
    )

    rm = sub.add_parser(
        "rm",
        help="Fully remove a manifold folder",
        description=(
            "Remove a whole manifold folder.  The manifold analogue of "
            "`saklas pack rm`.  Bundled manifolds (`default/` namespace) "
            "re-materialize on next session init."
        ),
    )
    rm.add_argument("selector", help="Manifold name (or ns/name)")
    rm.add_argument(
        "-y", "--yes", action="store_true",
        help="Skip the confirmation prompt for a bundled (default/) manifold",
    )

    clear = sub.add_parser(
        "clear",
        help="Delete per-model fitted tensors for a manifold",
        description=(
            "Delete a manifold's per-model fitted tensors (they re-fit on "
            "next use) while keeping `manifold.json` + the node corpus.  "
            "The manifold analogue of `saklas pack clear`."
        ),
    )
    clear.add_argument("selector", help="Manifold name (or ns/name)")
    clear.add_argument(
        "-m", "--model", default=None, metavar="MODEL_ID",
        help="Scope to one model's tensors only (default: all models)",
    )
    clear.add_argument(
        "--variant", choices=["raw", "sae", "all"], default="all",
        help="Which tensor variant(s) to delete. Default: all.",
    )

    refresh = sub.add_parser(
        "refresh",
        help="Re-pull / re-materialize a manifold from its source",
        description=(
            "Re-pull a manifold from its source: `local` is skipped, "
            "`bundled`/`default/` re-materializes from package data, "
            "`hf://` re-pulls.  With `-m/--model`, instead does a scoped "
            "refresh — drops just that model's fitted tensor pair so it "
            "re-fits on next use, without re-pulling from the source.  The "
            "manifold analogue of `saklas pack refresh`."
        ),
    )
    refresh.add_argument("selector", help="Manifold name (or ns/name)")
    refresh.add_argument(
        "-m", "--model", default=None, metavar="MODEL_ID",
        help="Scope to one model: clear its fitted tensor pair so it re-fits "
             "on next use (does not re-pull from source). Default: all-source "
             "re-pull.",
    )

    transfer = sub.add_parser(
        "transfer",
        help="Transfer a manifold from one model to another via Procrustes",
        description=(
            "Transfer a fitted manifold from a source model to a target "
            "model by fitting a per-layer Procrustes alignment between "
            "their cached neutral activations and mapping the fitted "
            "subspace into target space.  The manifold analogue of "
            "`saklas vector transfer`.  Writes the transferred tensor at "
            "the target's `_from-<safe_src>` variant path."
        ),
    )
    transfer.add_argument("name", help="Manifold name (or ns/name)")
    transfer.add_argument(
        "--from", dest="src_model", required=True, metavar="SRC_MODEL",
        help="Source model id (where the manifold was fitted)",
    )
    transfer.add_argument(
        "--to", dest="tgt_model", required=True, metavar="TGT_MODEL",
        help="Target model id (where the transferred manifold will live)",
    )
    transfer.add_argument(
        "-f", "--force", action="store_true",
        help="Recompute alignment + transfer even when cached",
    )
    transfer.add_argument(
        "-j", "--json", dest="json_output", action="store_true",
        help="Emit machine-readable JSON (path + quality summary)",
    )


_VECTOR_BUILDERS = {
    "extract":  _build_vector_extract,
    "merge":    _build_vector_merge,
    "clone":    _build_vector_clone,
    "compare":  _build_vector_compare,
    "why":      _build_vector_why,
    "transfer": _build_vector_transfer,
    "manifold": _build_vector_manifold,
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


# --- experiment subtree --------------------------------------------------

def _build_experiment_fan(p: argparse.ArgumentParser) -> None:
    p.add_argument("model", help="HuggingFace model ID or local path")
    p.add_argument("prompt", help="Prompt to fan out")
    p.add_argument(
        "-g", "--grid",
        action="append",
        required=True,
        metavar="CONCEPT=ALPHAS",
        help=(
            "Alpha grid term, repeatable. Example: "
            "happy.sad=-0.4,0,0.4"
        ),
    )
    p.add_argument(
        "-S", "--base-steering",
        default=None,
        metavar="EXPR",
        help="Fixed steering expression composed under each grid row",
    )
    p.add_argument("-q", "--quantize", choices=["4bit", "8bit"], default=None)
    p.add_argument("-d", "--device", default="auto")
    p.add_argument("-p", "--probes", nargs="*", default=None)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("-j", "--json", dest="json_output", action="store_true")
    _add_injection_args(p)
    _add_logit_args(p)
    _add_config_args(p)


def _build_experiment_transcript_parser(parser: argparse.ArgumentParser) -> None:
    """``saklas experiment transcript`` — replay saved tree paths.

    Phase 5 ships ``run`` only; future verbs (``ls``, ``diff``) compose
    on top of the same schema.
    """
    sub = parser.add_subparsers(dest="transcript_cmd", required=False, metavar="VERB")

    run = sub.add_parser(
        "run",
        help="Replay a transcript on the current session and report readings",
        description=(
            "Load a YAML transcript, replay each user turn with the recorded "
            "recipe, and report per-turn readings against the recorded ones."
        ),
    )
    run.add_argument("path", help="Path to a saklas_transcript YAML file")
    # Override the model arg shape so transcript replay can fall back to
    # the embedded ``model_id`` header (the common case) instead of
    # forcing the user to repeat it on the command line.  When the
    # transcript also lacks ``model_id`` the runner fails with a clear
    # message; that's caught early so we don't load a model just to
    # complain about it after.
    run.add_argument(
        "model",
        nargs="?",
        default=None,
        help="HuggingFace model ID or local path (defaults to transcript's `model_id`)",
    )
    run.add_argument(
        "-q", "--quantize",
        choices=["4bit", "8bit"],
        default=None,
        help="Quantization mode (default: bf16/fp16)",
    )
    run.add_argument(
        "-d", "--device",
        default="auto",
        help="Device: auto (detect), cuda, mps, or cpu (default: auto)",
    )
    run.add_argument(
        "-p", "--probes",
        nargs="*",
        default=None,
        help="Probe categories (default: all)",
    )
    run.add_argument(
        "--max-tokens", type=int, default=256,
        help="Default max generation tokens per replay turn",
    )
    _add_injection_args(run)
    _add_config_args(run)
    # ``--strict`` reuses the ``-s/--strict`` flag added by
    # ``_add_config_args`` — same name, but here it gates "refuse on
    # probe drift" instead of "fail hard on missing vectors".  Both
    # interpretations share the spirit of the flag.


def _build_experiment_naturalness(p: argparse.ArgumentParser) -> None:
    """``saklas experiment naturalness`` — behavior-manifold naturalness eval.

    Fits a behavior-space manifold from the manifold pack's node corpus
    (each node's mean next-token distribution, mapped to Hellinger
    space), runs a steered generation, re-runs the model over the
    generated text to recover its behavioral trajectory, and reports the
    mean / max Bhattacharyya distance of that trajectory to the behavior
    manifold — low is natural, high flags off-manifold "teleportation".
    """
    p.add_argument("model", help="HuggingFace model ID or local path")
    p.add_argument("prompt", help="Prompt to generate from")
    p.add_argument(
        "--manifold", required=True, metavar="FOLDER",
        help="Manifold folder whose node corpus seeds the behavior manifold",
    )
    p.add_argument(
        "-S", "--steer", required=True, metavar="EXPR",
        help="Steering expression to evaluate (e.g. 'mood%%0.5')",
    )
    p.add_argument(
        "--compare-linear", dest="compare_linear", action="store_true",
        help="Also score a linear-chord steering baseline (the manifold "
             "term must be a single '%%' term)",
    )
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("-q", "--quantize", choices=["4bit", "8bit"], default=None)
    p.add_argument("-d", "--device", default="auto")
    p.add_argument("-p", "--probes", nargs="*", default=None)
    p.add_argument("-j", "--json", dest="json_output", action="store_true")
    _add_injection_args(p)
    _add_config_args(p)


_EXPERIMENT_BUILDERS = {
    "fan": _build_experiment_fan,
    "transcript": _build_experiment_transcript_parser,
    "naturalness": _build_experiment_naturalness,
}


def _build_experiment_parser(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="experiment_cmd", required=False, metavar="VERB")
    for verb, desc in _EXPERIMENT_VERBS:
        child = sub.add_parser(verb, help=desc, description=desc)
        _EXPERIMENT_BUILDERS[verb](child)


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

    experiment = sub.add_parser(
        "experiment",
        help="Run and replay experiment trees",
        description="Run and replay experiment trees",
    )
    _build_experiment_parser(experiment)

    return root
