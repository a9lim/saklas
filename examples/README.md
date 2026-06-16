# saklas examples

These are scripts that show how the Python API works. This requires a GPU (CUDA or Apple Silicon MPS) and the dev extras installed:

```bash
pip install -e ".[dev]"
```

All examples use `google/gemma-3-4b-it` by default. Please override with `--model` if you would like to try a different architecture on them.

- **[`sweep_alpha.py`](sweep_alpha.py)**: sweep a steering vector's alpha through a range and print all the generations in sequence. It's good for finding the band where a concept is a good blend of coherent and effective on a newly added model.
- **[`ab_compare.py`](ab_compare.py)**: generate the same prompt with and without steering, and then list probe readings for both so you can see how the activations change.
- **[`representation_geometry/`](representation_geometry/)**: a longer investigation rather than a single script. It authors templated discover manifolds over structured vocabularies (countries, years) and reads the geometry of the fitted layout — showing that country names come out lexical (no map), years come out ordered, and the framing of the question reshapes the manifold. The headline result reads a model's own sense of "now" — its training-cutoff horizon — out of the activation geometry, and corroborates it against the scorer's explicit `P(current year)`. See its [README](representation_geometry/README.md).
