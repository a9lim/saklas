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

# Verb table for ``saklas manifold <verb>`` — the *compute* surface.  A
# steering vector is the K=2 case of a flat affine subspace, so the former
# ``subspace`` verbs (extract / bake / compare / why) fold into the one
# ``manifold`` verb alongside the full steering-manifold authoring tree.
# This table is the source of truth for both parser registration order and
# the bare-verb help block.
_MANIFOLD_VERBS: list[tuple[str, str]] = [
    ("extract",   "Extract a steering vector (a 2-node flat subspace)"),
    ("generate",  "Author a discover-mode manifold from a concept list"),
    ("fit",       "Fit an authored or discover-mode manifold"),
    ("bake",      "Bake existing vectors into a new manifold"),
    ("merge",     "Union discover-mode manifolds' nodes into a new manifold"),
    ("transfer",  "Transfer a manifold to another model via Procrustes"),
    ("compare",   "Cosine similarity between steering vectors"),
    ("why",       "Show which layers contribute most to a steering vector"),
]

# Verb table for ``saklas pack <verb>`` — the *lifecycle* surface (install /
# share / inspect / remove).  These were the lifecycle leaves of the former
# ``manifold`` verb; they address an on-disk manifold by ``(namespace, name)``
# and never load a model.
_PACK_VERBS: list[tuple[str, str]] = [
    ("ls",        "List installed manifolds"),
    ("show",      "Show a manifold's nodes and fitted models"),
    ("install",   "Install a manifold from HF or a local folder"),
    ("search",    "Search the HuggingFace hub for manifolds"),
    ("push",      "Push a manifold to HF as a model repo"),
    ("rm",        "Fully remove a manifold folder"),
    ("clear",     "Delete per-model fitted tensors for a manifold"),
    ("refresh",   "Re-pull / re-materialize a manifold from its source"),
    ("export",    "Export a fitted manifold to an interchange format (gguf)"),
]

_EXPERIMENT_VERBS: list[tuple[str, str]] = [
    ("fan",         "Run an alpha grid as one experiment"),
    ("transcript",  "Replay or inspect saved transcript paths"),
    ("naturalness", "Score a steered generation's behavior-manifold naturalness"),
]


def _add_injection_args(p: argparse.ArgumentParser) -> None:
    """Steering / extraction options shared between ``tui`` and ``serve``.

    ``None`` defaults flow through to the YAML override layer (or ultimately
    to the session defaults: Mahalanobis projection + DLS on).
    """
    p.add_argument(
        "--no-dls", dest="no_dls", action="store_true",
        help="Disable the discriminative-layer-selection mask at "
             "extraction time.  v2.1 introduced centered DLS (Dang & "
             "Ngo 2026, Eq. 9) as the default: layers where pos- and "
             "neg-class means project to the same side of the neutral "
             "baseline along ``d̂`` are dropped — they encode concept "
             "intensity rather than concept polarity.  Pass ``--no-dls`` "
             "to keep every layer.",
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

# --- subspace / manifold subtree -----------------------------------------

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
        help="Destination namespace for the extracted concept manifold. "
             "Unset lands under "
             "``~/.saklas/manifolds/local/<canonical>/``. Any other "
             "value relocates to "
             "``manifolds/<namespace>/<canonical>/``. Parity with the "
             "webui ExtractDrawer's namespace control and with "
             "``manifold`` / ``discover``'s NS slot.",
    )
    p.set_defaults(quantize=None, device="auto", probes=None)


def _build_vector_merge(p: argparse.ArgumentParser) -> None:
    p.add_argument("name", help="New manifold name (written under local/)")
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
        "--ridge-scale", type=float, default=1.0, metavar="FLOAT",
        help=(
            "Ridge multiplier on the regularized covariance "
            "(λ_L = (||X_L||_F²/(N·D)) × ridge_scale); default 1.0 (mean "
            "diagonal of the un-regularized sample covariance).  Compare is "
            "Mahalanobis-only — it reads cached neutral activations + layer "
            "means under ~/.saklas/models/<id>/ to build the per-layer "
            "whitener and fails if the cache is missing."
        ),
    )


def _build_vector_why(p: argparse.ArgumentParser) -> None:
    p.add_argument("concept", help="Concept selector (name or ns/name)")
    p.add_argument("-m", "--model", required=True, metavar="MODEL_ID",
                   help="Model id (used to locate the baked tensor)")
    p.add_argument("-j", "--json", dest="json_output", action="store_true",
                   help="Emit machine-readable JSON (full per-layer detail)")


def _build_manifold_fit(fit: argparse.ArgumentParser) -> None:
    """``saklas manifold fit`` — fit an authored *or* discover-mode manifold.

    The 4.0 fold of the former ``fit`` (authored folder positional) and
    ``discover`` (name → folder + hyperparam overrides) verbs into one.
    The positional is a name-or-folder (``_run_manifold_fit`` resolves it),
    and the discover hyperparam knobs are accepted but apply only when the
    resolved folder is discover-mode (``pca`` / ``spectral``); supplying any
    of them against an authored folder is an error.
    """
    fit.add_argument(
        "target", help="manifold name or folder path",
    )
    fit.add_argument("-m", "--model", default=None, metavar="MODEL_ID")
    fit.add_argument("-f", "--force", action="store_true")
    fit.add_argument(
        "--sae", default=None, metavar="RELEASE",
        help="Fit in an SAELens SAE feature space (requires `.[sae]`); "
             "centroids are reconstructed through the SAE before the fit.",
    )
    fit.add_argument(
        "--sae-revision", dest="sae_revision", default=None, metavar="REV",
        help="Pin a specific HF revision for the SAE release",
    )
    # Discover-mode hyperparam overrides.  Written into ``manifold.json``
    # before the fit; rejected against an authored folder.
    fit.add_argument(
        "--method", choices=("pca", "spectral"), default=None,
        help="Discover-mode: override the folder's fit_mode "
             "(default: keep folder's setting)",
    )
    fit.add_argument(
        "--max-dim", dest="max_dim", type=int, default=None, metavar="N",
        help="Discover-mode: cap intrinsic dimension "
             "(default: 8 for PCA, 8 for spectral)",
    )
    fit.add_argument(
        "--min-dim", dest="min_dim", type=int, default=None, metavar="N",
        help="Discover spectral: floor the intrinsic dimension the "
             "eigenvalue-ratio cliff picks (the cliff undershoots when one "
             "mode dominates — set --min-dim == --max-dim to pin the dim "
             "exactly, e.g. PAD's 3). Ignored for --method pca.",
    )
    fit.add_argument(
        "--var-threshold", dest="var_threshold", type=float, default=None,
        metavar="T",
        help="Discover PCA: smallest cumulative-variance prefix that "
             "crosses T is picked (default: 0.70)",
    )
    fit.add_argument(
        "--k-nn", dest="k_nn", type=int, default=None, metavar="K",
        help="Discover spectral: number of nearest neighbors in the k-NN "
             "graph (default: max(5, ceil(log K)))",
    )
    fit.add_argument(
        "--bandwidth", type=float, default=None, metavar="SIGMA",
        help="Discover spectral: heat-kernel bandwidth "
             "(default: median k-NN distance)",
    )
    fit.add_argument(
        "--max-subspace-dim", dest="max_subspace_dim", type=int, default=None,
        metavar="R",
        help="Discover spectral (curved) only: per-layer RBF subspace dim "
             "cap (default 64). Smaller values give finer-grained steering "
             "control at large K — each axis the RBF can move along is an "
             "axis subspace_inject can displace, so fewer axes = smaller "
             "per-α effect = wider coherence regime. Ignored for --method pca "
             "(a flat fit's subspace dim is its layout dim — use --max-dim).",
    )
    fit.set_defaults(quantize=None, device="auto", probes=None)


def _build_manifold_generate(generate: argparse.ArgumentParser) -> None:
    generate.add_argument("name", help="Manifold name (use ns/name for non-local)")
    generate.add_argument(
        "--concepts", nargs="+", required=True, metavar="CONCEPT",
        help="Concept slugs to generate corpora for (>= 2)",
    )
    generate.add_argument(
        "--kind", choices=("abstract", "concrete"), default="abstract",
        help=(
            "Conceptual kind for every node (default: abstract).  Selects the "
            "system template + elicitation role label: abstract -> 'someone "
            "{c}', concrete -> '{art} {c}'."
        ),
    )
    generate.add_argument(
        "--samples-per-prompt", dest="samples_per_prompt",
        type=int, default=1, metavar="K",
        help="In-character responses generated per baseline prompt (default: 1)",
    )
    generate.add_argument(
        "--description", default="", metavar="TEXT",
        help="Human-readable description for the manifold folder",
    )
    generate.add_argument(
        "--seed", type=int, default=None, metavar="INT",
        help="Seed the response generation for reproducible corpora "
             "(default: unseeded)",
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


def _build_manifold_merge(merge: argparse.ArgumentParser) -> None:
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


def _build_manifold_transfer(transfer: argparse.ArgumentParser) -> None:
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


# --- pack lifecycle leaf builders ----------------------------------------

def _build_pack_ls(ls: argparse.ArgumentParser) -> None:
    ls.add_argument("--namespace", default=None, metavar="NS",
                    help="Restrict to a single namespace")
    ls.add_argument("-j", "--json", dest="json_output", action="store_true")
    ls.add_argument("-v", "--verbose", action="store_true",
                    help="Include descriptions in the table output")


def _build_pack_show(show: argparse.ArgumentParser) -> None:
    show.add_argument("name", help="Manifold name (or ns/name)")
    show.add_argument("-j", "--json", dest="json_output", action="store_true")


def _build_pack_install(install: argparse.ArgumentParser) -> None:
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


def _build_pack_search(search: argparse.ArgumentParser) -> None:
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


def _build_pack_push(push: argparse.ArgumentParser) -> None:
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


def _build_pack_rm(rm: argparse.ArgumentParser) -> None:
    rm.add_argument("selector", help="Manifold name (or ns/name)")
    rm.add_argument(
        "-y", "--yes", action="store_true",
        help="Skip the confirmation prompt for a bundled (default/) manifold",
    )


def _build_pack_clear(clear: argparse.ArgumentParser) -> None:
    clear.add_argument("selector", help="Manifold name (or ns/name)")
    clear.add_argument(
        "-m", "--model", default=None, metavar="MODEL_ID",
        help="Scope to one model's tensors only (default: all models)",
    )
    clear.add_argument(
        "--variant", choices=["raw", "sae", "all"], default="all",
        help="Which tensor variant(s) to delete. Default: all.",
    )


def _build_pack_refresh(refresh: argparse.ArgumentParser) -> None:
    refresh.add_argument("selector", help="Manifold name (or ns/name)")
    refresh.add_argument(
        "-m", "--model", default=None, metavar="MODEL_ID",
        help="Scope to one model: clear its fitted tensor pair so it re-fits "
             "on next use (does not re-pull from source). Default: all-source "
             "re-pull.",
    )


def _build_pack_export(export: argparse.ArgumentParser) -> None:
    export_fmt = export.add_subparsers(dest="format", required=True)
    gguf = export_fmt.add_parser(
        "gguf", help="Export a folded manifold to llama.cpp GGUF",
    )
    gguf.add_argument("name", help="Manifold name (or ns/name)")
    gguf.add_argument("-m", "--model", default=None, metavar="MODEL_ID")
    gguf.add_argument("-o", "--output", default=None, metavar="PATH")
    gguf.add_argument("--model-hint", default=None, metavar="HINT")


# --- manifold (compute) + pack (lifecycle) parser wiring -----------------

# Long-form ``description=`` blocks per manifold compute verb (the verb's
# ``-h`` carries these; the short ``help=`` lives in ``_MANIFOLD_VERBS``).
_MANIFOLD_DESCRIPTIONS: dict[str, str] = {
    "extract": "Extract a steering vector (a 2-node flat `pca` manifold).",
    "generate": (
        "For every concept, have the loaded model answer the shared "
        "baseline prompts *in character* (the concept rides the system "
        "prompt + a kind-derived elicitation role), writing one corpus per "
        "node.  Writes a fresh discover-mode manifold folder under "
        "~/.saklas/manifolds/<ns>/<name>/ ready for `saklas manifold fit` "
        "to fit.  The shared baseline prompts hold the topic common-mode "
        "across nodes, so the per-concept centroids stay comparable "
        "(response[i] aligns to baseline prompt[i % k])."
    ),
    "fit": (
        "Pool per-node centroids, fit a per-layer PCA subspace (+ RBF "
        "interpolant for curved manifolds), and write the per-model manifold "
        "tensor into the folder.  The positional is a manifold name OR a "
        "folder path.  Authored folders supply their own coords; discover-mode "
        "folders (fit_mode pca/spectral, usually authored by `saklas manifold "
        "generate`) derive coords per-model and accept the --method / "
        "--max-dim / --var-threshold / --k-nn / --bandwidth / "
        "--max-subspace-dim hyperparam overrides (written into manifold.json "
        "before the fit; rejected against an authored folder)."
    ),
    "bake": (
        "Bake a steering expression into a corpus-less manifold under local/ "
        "(the merge op): a linear combination / projection of existing fitted "
        "vectors, written as a fit_mode=baked manifold."
    ),
    "merge": (
        "Union the *nodes* of two or more discover-mode source manifolds into "
        "a fresh, unfitted discover folder, then run `saklas manifold fit "
        "<merged>` to fit it.  The corpus analogue of `manifold bake` — but on "
        "node corpora rather than steering directions.  Restricted to "
        "discover-mode sources (authored manifolds carry user-declared "
        "geometry that isn't mergeable without a shared coordinate system).  "
        "Label collisions across sources raise; rename one side before merging."
    ),
    "transfer": (
        "Transfer a fitted manifold from a source model to a target model by "
        "fitting a per-layer Procrustes alignment between their cached neutral "
        "activations and mapping the fitted subspace into target space.  "
        "Writes the transferred tensor at the target's `_from-<safe_src>` "
        "variant path."
    ),
    "compare": "Cosine similarity between steering vectors (Mahalanobis).",
    "why": (
        "Per-layer ‖baked‖ histogram (16 buckets) showing which layers "
        "contribute most to a steering vector."
    ),
}

_MANIFOLD_BUILDERS = {
    "extract":  _build_vector_extract,
    "generate": _build_manifold_generate,
    "fit":      _build_manifold_fit,
    "bake":     _build_vector_merge,
    "merge":    _build_manifold_merge,
    "transfer": _build_manifold_transfer,
    "compare":  _build_vector_compare,
    "why":      _build_vector_why,
}

_PACK_DESCRIPTIONS: dict[str, str] = {
    "ls": "List local manifolds under ~/.saklas/manifolds/.",
    "show": (
        "Print one manifold's domain, node coordinates, and per-model fitted "
        "tensors."
    ),
    "install": (
        "Pull a manifold from a HuggingFace saklas-manifold repo "
        "(`<ns>/<name>[@revision]`) or copy-install a local folder path."
    ),
    "search": "Search HF for `saklas-manifold`-tagged model repos.",
    "push": (
        "Push a manifold folder (corpus + fitted tensors) to HF as a "
        "`saklas-manifold`-tagged model repo.  The corpus is always uploaded "
        "(a manifold can't re-fit without it); per-model tensors are filtered "
        "by `-m`/`--variant`."
    ),
    "rm": (
        "Remove a whole manifold folder.  Bundled manifolds (`default/` "
        "namespace) re-materialize on next session init."
    ),
    "clear": (
        "Delete a manifold's per-model fitted tensors (they re-fit on next "
        "use) while keeping `manifold.json` + the node corpus."
    ),
    "refresh": (
        "Re-pull a manifold from its source: `local` is skipped, "
        "`bundled`/`default/` re-materializes from package data, `hf://` "
        "re-pulls.  With `-m/--model`, instead does a scoped refresh — drops "
        "just that model's fitted tensor pair so it re-fits on next use, "
        "without re-pulling from the source."
    ),
    "export": (
        "Fold a fitted 2-node ``pca`` manifold down to a single steering "
        "direction and write it out in an interchange format.  Only ``gguf`` "
        "(llama.cpp control-vector) is supported today; the fold requires an "
        "affine 2-node (R=1) subspace, so multi-node / curved manifolds are "
        "rejected."
    ),
}

_PACK_BUILDERS = {
    "ls":       _build_pack_ls,
    "show":     _build_pack_show,
    "install":  _build_pack_install,
    "search":   _build_pack_search,
    "push":     _build_pack_push,
    "rm":       _build_pack_rm,
    "clear":    _build_pack_clear,
    "refresh":  _build_pack_refresh,
    "export":   _build_pack_export,
}


def _build_manifold_parser(parser: argparse.ArgumentParser) -> None:
    """``saklas manifold`` — the steering-vector / manifold compute verbs."""
    sub = parser.add_subparsers(dest="manifold_cmd", required=False, metavar="VERB")
    for verb, desc in _MANIFOLD_VERBS:
        child = sub.add_parser(
            verb, help=desc,
            description=_MANIFOLD_DESCRIPTIONS.get(verb, desc),
        )
        _MANIFOLD_BUILDERS[verb](child)


def _build_pack_parser(parser: argparse.ArgumentParser) -> None:
    """``saklas pack`` — the manifold lifecycle (install/share/inspect) verbs."""
    sub = parser.add_subparsers(dest="pack_cmd", required=False, metavar="VERB")
    for verb, desc in _PACK_VERBS:
        child = sub.add_parser(
            verb, help=desc,
            description=_PACK_DESCRIPTIONS.get(verb, desc),
        )
        _PACK_BUILDERS[verb](child)


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

    manifold = sub.add_parser(
        "manifold",
        help="Manifold compute (extract/generate/fit/bake/merge/transfer/compare/why)",
        description="Steering-vector / manifold authoring, fitting, and analysis",
    )
    _build_manifold_parser(manifold)

    pack = sub.add_parser(
        "pack",
        help="Manifold lifecycle (ls/show/install/search/push/rm/clear/refresh/export)",
        description="Manifold install / share / inspect / removal lifecycle",
    )
    _build_pack_parser(pack)

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
