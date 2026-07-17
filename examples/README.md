# saklas examples

These scripts exercise the Python API against a local Hugging Face model. The
generation examples need the base package and a CUDA or Apple Silicon MPS device:

```bash
pip install -e .
```

The representation-geometry investigation also needs the plotting and analysis
dependencies in the research extra:

```bash
pip install -e ".[research]"
```

All examples use `google/gemma-3-4b-it` by default. Please override with `--model` if you would like to try a different architecture on them.

- **[`sweep_alpha.py`](sweep_alpha.py)**: extract a concept manifold, sweep its
  steering coefficient, and print deterministic generations.
- **[`ab_compare.py`](ab_compare.py)**: generate the same prompt with and without
  steering, then compare the final aggregate probe coordinates.
- **[`representation_geometry/`](representation_geometry/)**: a longer investigation rather than a single script. It authors templated discover manifolds over structured vocabularies (countries, years) and reads the geometry of the fitted layout — showing that country names come out lexical (no map), years come out ordered, and the framing of the question reshapes the manifold. The headline result reads a model's own sense of "now" — its training-cutoff horizon — out of the activation geometry, and corroborates it against the scorer's explicit `P(current year)`. See its [README](representation_geometry/README.md).
