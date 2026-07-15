# Contributing to saklas

Bug reports, focused fixes, new model support, and documentation improvements are
all welcome. The repository supports Python 3.11–3.13 and uses `pip` plus
setuptools; there is no lockfile-based development workflow.

## Dev setup

```bash
git clone https://github.com/a9lim/saklas
cd saklas
pip install -e ".[dev]"
```

Use `[hf]` for streamed dataset-backed J-lens/SAE fitting, `[gguf]` for GGUF
export, `[research]` for the analysis examples, and `[notebook]` for the Plotly
notebook helpers. SAELens, the HTTP server, the TUI, and the prebuilt dashboard
are part of the base install. `[cuda]` adds bitsandbytes and Hugging Face
`kernels`; `[cuda-experimental]` adds flash-attn. The CUDA extras are
Linux/CUDA-only.

## Running tests

```bash
pytest -q -m "not gpu"             # CI-equivalent CPU suite
pytest -q tests/                    # everything available on this machine
pytest -q tests/test_smoke.py       # real-model smoke tests
```

Tests marked `gpu` need CUDA or Apple Silicon MPS and may download model weights.
CI runs the non-GPU suite on Python 3.11, 3.12, and 3.13.

## Static checks

```bash
ruff check .
pyright
python -m build
```

`ruff check . --fix` applies safe automatic lint fixes. `pyright` covers the
package, tests, examples, and maintained scripts according to `pyproject.toml`.

## Working on the web UI

The dashboard source is a Svelte 5 and Vite app at the repo's top-level `webui/` directory. The committed `saklas/web/dist/` bundle is the source of truth that ships in the wheel.

```bash
cd webui
npm ci
npm run dev     # vite dev server on http://localhost:5173 with hot reload
npm run check   # svelte-check + theme-token validation
npm run build   # writes ../saklas/web/dist/
```

`npm run dev` proxies `/saklas`, `/v1`, and `/api` (including WebSockets) to
`http://localhost:8000`. Keep `saklas serve <model>` running in another shell.
The production bundle under `saklas/web/dist/` is committed package data, so UI
changes must include the rebuilt bundle. CI rebuilds it and fails on any diff.

## Adding a new model architecture

If you want to add another model's architecture, add an entry to `saklas/core/model.py:_LAYER_ACCESSORS`, keyed by the HuggingFace `model_type`. The accessor takes a loaded model and returns the list of transformer blocks. The rest of the code (`vectors.py`, `hooks.py`, `monitor.py`) doesn't depend on the architecture.

If the model is quirky (multimodal text extraction, FP8 dequantization,
nonstandard tokenizer behavior), inspect the existing text-only multimodal path
and architecture-specific workarounds in `saklas/core/model.py`. Include a focused
accessor test and a real-model smoke result when possible.

## PRs

- If you're adding an architecture, include the exact model ID, device, dtype or
  quantization mode, and the commands you ran. A fit or generation using a bundled
  manifold is the most useful end-to-end proof.
- Please don't bump the version in your PR unless you want a new release; the PyPI publish workflow is triggered by a version update.

## Questions

Please reach out to me or open an issue. For anything security-sensitive, also see [SECURITY.md](SECURITY.md).
