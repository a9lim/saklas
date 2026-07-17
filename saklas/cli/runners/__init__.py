"""Runner functions for saklas CLI subcommands.

This package splits the former single ``runners.py`` module into one file per
verb group (``serve`` / ``manifold`` / ``pack`` / ``template`` / ``config`` /
``lens`` / ``sae`` / ``experiment``) plus a ``shared`` module for the helpers
used across more than one group.  Every runner and helper is re-exported here,
so the historical import surface ``saklas.cli.runners.<name>`` (and the
monkeypatch seams the test-suite relies on) keeps working unchanged.

Session/config helpers that the test-suite swaps at the package level
(``_make_session`` / ``_print_startup`` / ``_print_model_info`` /
``_load_effective_config`` / the ``_fold_*`` folders /
``_load_or_fit_transfer_alignment``) are invoked by the submodule runners
through this package object (``import saklas.cli.runners as _pkg``), so a
``monkeypatch.setattr("saklas.cli.runners.<name>", ...)`` still takes effect.
"""

# This module is intentionally a compatibility barrel: every imported private
# name is re-exported for historical callers and monkeypatch seams.
# ruff: noqa: F401

from __future__ import annotations

from saklas.cli.runners.shared import (
    _R,
    _iter_manifold_folders,
    _load_effective_config,
    _make_session,
    _print_model_info,
    _print_startup,
    _resolve_manifold_folder,
    _resolve_manifold_ns_name,
    _resolve_probes,
    _saklas_error_exit,
    _setup_steering_vectors,
    _split_manifold_ns_name,
    _warmup_session,
)

# The transfer-alignment single-flight orchestration now lives beside the
# alignment primitives it wraps; re-exported under its historical private name
# so ``saklas.cli.runners._load_or_fit_transfer_alignment`` still resolves and
# stays monkeypatchable at the package level.
from saklas.io.alignment import (
    load_or_fit_transfer_alignment as _load_or_fit_transfer_alignment,
)

from saklas.cli.runners.serve import (
    _best_serve_sae_release,
    _enable_serve_live_lens_if_compatible,
    _enable_serve_live_sae_if_available,
    _run_serve,
)
from saklas.cli.runners.manifold import (
    _MANIFOLD_RUNNERS,
    _VARIANT_SUFFIX_RE,
    _fold_all_fitted_manifolds,
    _fold_manifold_to_profile_with_identity,
    _print_diagnostics,
    _print_why_histogram,
    _require_model,
    _run_manifold,
    _run_manifold_bake,
    _run_manifold_compare,
    _run_manifold_extract,
    _run_manifold_fit,
    _run_manifold_from_template,
    _run_manifold_generate,
    _run_manifold_merge,
    _run_manifold_transfer,
    _run_manifold_why,
    _split_variant_suffix,
)
from saklas.cli.runners.pack import (
    _PACK_RUNNERS,
    _run_pack,
    _run_pack_clear,
    _run_pack_export,
    _run_pack_install,
    _run_pack_ls,
    _run_pack_push,
    _run_pack_refresh,
    _run_pack_rm,
    _run_pack_search,
    _run_pack_show,
)
from saklas.cli.runners.template import (
    _TEMPLATE_RUNNERS,
    _normalize_context_entry,
    _run_template,
    _run_template_create,
    _run_template_ls,
    _run_template_rm,
    _run_template_score,
    _run_template_show,
)
from saklas.cli.runners.config import (
    _run_config,
    _run_config_show,
    _run_config_validate,
)
from saklas.cli.runners.lens import (
    _LENS_DOC_CHARS,
    _LENS_RUNNERS,
    _load_lens_corpus,
    _lens_fit_source_preflight_matches,
    _parse_layer_list,
    _run_lens,
    _run_lens_decompose,
    _run_lens_fetch,
    _run_lens_fit,
    _run_lens_ls,
    _run_lens_rm,
    _run_lens_show,
    _run_lens_top,
    _run_lens_use,
    _try_lens_fit_noop_preflight,
    _try_lens_fit_noop_preflight_locked,
)
from saklas.cli.runners.sae import (
    _SAE_RUNNERS,
    _load_sae_training_corpus,
    _resolve_active_sae_source,
    _run_sae,
    _run_sae_fetch,
    _run_sae_ls,
    _run_sae_rm,
    _run_sae_show,
    _run_sae_train,
    _run_sae_use,
)
from saklas.cli.runners.experiment import (
    _parse_grid_terms,
    _run_experiment,
    _run_experiment_fan,
    _run_experiment_naturalness,
    _run_experiment_transcript,
    _run_transcript_run,
)


_COMMAND_RUNNERS = {
    "serve":      _run_serve,
    "manifold":   _run_manifold,
    "pack":       _run_pack,
    "config":     _run_config,
    "experiment": _run_experiment,
    "template":   _run_template,
    "lens":       _run_lens,
    "sae":        _run_sae,
}
