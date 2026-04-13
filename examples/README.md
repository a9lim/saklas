# saklas examples

Runnable scripts showing the Python API. Each example assumes you have a GPU (CUDA or Apple Silicon MPS) and the default extras installed:

```bash
pip install -e ".[dev]"
```

All examples default to `google/gemma-3-4b-it` — override with `--model` if you want to try another architecture.

- **[`sweep_alpha.py`](sweep_alpha.py)** — sweep a steering vector's alpha across a range and print the generations side-by-side. Useful for finding the coherent-nuanced band (~0.4–0.8) for a new concept on a new model.
- **[`ab_compare.py`](ab_compare.py)** — generate the same prompt with and without steering, then dump probe readings for both so you can see how the activation trajectory shifts.
