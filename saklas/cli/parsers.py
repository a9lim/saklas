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
]


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
    _add_config_args(parser)


def _build_serve_parser(parser: argparse.ArgumentParser) -> None:
    _add_common_args(parser)
    parser.add_argument("-H", "--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("-P", "--port", type=int, default=8000, help="Bind port")
    parser.add_argument("-S", "--steer", action="append", default=[], metavar="NAME[:ALPHA]",
                        help="Pre-load a steering vector (repeatable)")
    parser.add_argument("-C", "--cors", action="append", default=[], metavar="ORIGIN",
                        help="CORS allowed origin (repeatable)")
    parser.add_argument("-k", "--api-key", default=None, metavar="KEY",
                        help="Require Bearer token auth; falls back to $SAKLAS_API_KEY")
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
    p.set_defaults(quantize=None, device="auto", probes=None)


def _build_vector_merge(p: argparse.ArgumentParser) -> None:
    p.add_argument("name", help="New pack name (written under local/)")
    p.add_argument("components", help="Comma-separated components: ns/a:0.3,ns/b:0.4")
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


def _build_vector_why(p: argparse.ArgumentParser) -> None:
    p.add_argument("concept", help="Concept selector (name or ns/name)")
    p.add_argument("-m", "--model", required=True, metavar="MODEL_ID",
                   help="Model id (used to locate the baked tensor)")
    p.add_argument("-n", "--top-n", dest="top_n", type=int, default=5,
                   help="Number of top layers to show (default: 5)")
    p.add_argument("--all", dest="show_all", action="store_true",
                   help="Show every layer sorted descending (overrides -n)")
    p.add_argument("-j", "--json", dest="json_output", action="store_true",
                   help="Emit machine-readable JSON")


_VECTOR_BUILDERS = {
    "extract": _build_vector_extract,
    "merge":   _build_vector_merge,
    "clone":   _build_vector_clone,
    "compare": _build_vector_compare,
    "why":     _build_vector_why,
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
        epilog=(
            "top-level verbs:\n"
            "  tui      Launch the interactive TUI (requires <model>)\n"
            "  serve    Start the OpenAI + Ollama compatible API server\n"
            "  pack     Manage concept packs (install/ls/search/push/...)\n"
            "  vector   Vector operations (extract/merge/clone/compare/why)\n"
            "  config   Inspect and validate saklas config files\n"
            "\n"
            "Run `saklas <verb> -h` for verb-specific options."
        ),
    )
    sub = root.add_subparsers(dest="command", required=False, metavar="VERB")

    tui = sub.add_parser("tui", help="Launch the interactive TUI", description="Launch the interactive TUI")
    _build_tui_parser(tui)

    serve = sub.add_parser("serve", help="Start the API server", description="Start the API server")
    _build_serve_parser(serve)

    pack = sub.add_parser("pack", help="Manage concept packs", description="Manage concept packs")
    _build_pack_parser(pack)

    vector = sub.add_parser("vector", help="Vector operations",
                               description="Vector operations (extract/merge/clone/compare/why)")
    _build_vector_parser(vector)

    cfg = sub.add_parser("config", help="Inspect/validate config", description="Inspect/validate config")
    _build_config_parser(cfg)

    return root
