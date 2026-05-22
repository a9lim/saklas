# saklas examples

These are scripts that show how the Python API works. This requires a GPU (CUDA or Apple Silicon MPS) and the dev extras installed:

```bash
pip install -e ".[dev]"
```

All examples use `google/gemma-3-4b-it` by default. Please override with `--model` if you would like to try a different architecture on them.

- **[`sweep_alpha.py`](sweep_alpha.py)**: sweep a steering vector's alpha through a range and print all the generations in sequence. It's good for finding the band where a concept is a good blend of coherent and effective on a newly added model.
- **[`ab_compare.py`](ab_compare.py)**: generate the same prompt with and without steering, and then list probe readings for both so you can see how the activations change.
- **[`manifold_steering.py`](manifold_steering.py)**: author a small 2-D manifold (Russell's affective circumplex as a valence x arousal disk), fit it against the model, and generate the same prompt at several points of the disk. It shows how manifold steering differs from a single linear direction, and how intensity falls out as the radius from the neutral center.
