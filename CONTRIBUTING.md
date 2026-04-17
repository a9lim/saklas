# Contributing to saklas

Thank you very much for wanting to contribute! I really appreciate any contribution you would like to make, whether it's a PR or bug report. New architecture support is especially welcome, it's usually just a small patch.

## Dev setup

```bash
git clone https://github.com/a9lim/saklas
cd saklas
pip install -e ".[dev]"
```

Optional extras: `[cuda]` for bitsandbytes and flash-attn, `[serve]` for the FastAPI server, `[research]` for datasets and pandas, `[sae]` for SAELens-backed SAE extraction.

## Running tests

```bash
pytest tests/ -v                    # everything
pytest tests/test_paths.py -v       # fast non-GPU tests
pytest tests/test_smoke.py -v       # GPU smoke tests (downloads gemma-3-4b-it, ~8GB)
```

Smoke tests need CUDA or Apple Silicon MPS. The non-GPU tests (`test_paths`, `test_packs`, `test_selectors`, `test_cache_ops`, `test_hf`, `test_merge`, `test_config_file`, `test_cli_flags`, `test_probes_bootstrap`, `test_results`, `test_datasource`, `test_server`) run anywhere, and are what CI exercises.

## Lint

CI runs `ruff check .` on every PR. Please run it locally first:

```bash
ruff check .
ruff check . --fix    # auto-fix what's fixable
```

Or install pre-commit to run it automatically on every commit:

```bash
pip install pre-commit
pre-commit install
```

## Adding a new model architecture

If you want to add another model's architecture, add an entry to `saklas/core/model.py:_LAYER_ACCESSORS`, keyed by the HuggingFace `model_type`. Please use a `def`, not a lambda. The accessor takes a loaded model and returns the list of transformer blocks. The rest of the code (`vectors.py`, `hooks.py`, `monitor.py`) doesn't depend on the architecture usually.

If the model is quirky (multimodal text extraction, FP8 dequantization, nonstandard tokenizer behavior), please look at how Ministral-3 is handled in `saklas/core/model.py:_load_text_from_multimodal` for a worked example.

## PRs

- Please keep them focused: one feature or fix per PR.
- If you're adding an architecture, please include a note in the PR about which model you tested against and what the extraction scores looked like.
- Please don't bump the version in your PR unless you're explicitly cutting a release; that's what triggers the PyPI publish workflow.

## Questions

Open an issue. For anything security-sensitive, please see [SECURITY.md](SECURITY.md).
