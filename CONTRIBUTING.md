# Contributing to saklas

Thanks for your interest. This is a small research tool — PRs, bug reports, and new architecture support are all welcome.

## Dev setup

```bash
git clone https://github.com/a9lim/saklas
cd saklas
pip install -e ".[dev]"
```

Optional extras: `[cuda]` for bitsandbytes + flash-attn, `[serve]` for the FastAPI server, `[research]` for datasets/pandas.

## Running tests

```bash
pytest tests/ -v                    # everything
pytest tests/test_paths.py -v       # fast non-GPU tests
pytest tests/test_smoke.py -v       # GPU smoke tests (downloads gemma-3-4b-it, ~8GB)
```

Smoke tests need CUDA or Apple Silicon MPS. Non-GPU tests (`test_paths`, `test_packs`, `test_selectors`, `test_cache_ops`, `test_hf`, `test_merge`, `test_config_file`, `test_cli_flags`, `test_probes_bootstrap`, `test_results`, `test_datasource`, `test_server`) run anywhere and are what CI exercises.

## Lint

CI runs `ruff check .` on every PR. Run it locally first:

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

One entry in `saklas/model.py:_LAYER_ACCESSORS`, keyed by HuggingFace `model_type`. Use a `def`, not a lambda. The accessor takes a loaded model and returns the list of transformer blocks. That's usually it — `vectors.py`, `hooks.py`, and `monitor.py` are architecture-agnostic.

If the model has quirks (multimodal text extraction, FP8 dequantization, non-standard tokenizer specials), look at how Ministral-3 is handled in `model.py:_load_text_from_multimodal` for a worked example.

## PRs

- Keep them focused. One feature or fix per PR.
- If you're adding an architecture, please include a note in the PR about which model you tested against and what the extraction scores looked like.
- Don't bump the version in your PR unless you're explicitly cutting a release — that's what triggers the PyPI publish workflow.

## Questions

Open an issue. For anything security-sensitive, see [SECURITY.md](SECURITY.md).
