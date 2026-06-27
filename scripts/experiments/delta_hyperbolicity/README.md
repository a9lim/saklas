# delta_hyperbolicity

Gromov δ-hyperbolicity probe for saklas concept geometry: given the pairwise
distance matrix of a set of concept centroids, how tree-like is the metric? A
tree is 0-hyperbolic; hyperbolic space has small bounded δ; Euclidean (flat) space
is not hyperbolic and its δ grows with the diameter; positive curvature (sphere,
ring) is the anti-tree. The motivating question was whether a deliberately
hierarchical concept heap (an animal taxonomy) is represented tree-like-ly enough
to justify a hyperbolic steering domain.

## Files

`delta_hyperbolicity.py` — the instrument. Gromov four-point δ, exact over all
basepoints (Fournier–Ismail–Vigneron max-min matrix form), plus an O(n⁴)
brute-force cross-check. Reports δ, diameter, and the scale-free δ_rel = δ/diam.

`validate_delta.py` — synthetic calibration. Runs the probe on spaces with known
answers and checks the ordering tree < hyperbolic < euclidean < sphere. The bands
it establishes: tree 0.00, hyperbolic ≈ 0.11, euclidean-flat ≈ 0.19–0.22,
ring/sphere ≈ 0.42–0.50.

`delta_probe_fromfit.py` — δ on already-fitted saklas manifolds, no model load.
For an affine fit `layer_N.node_coords` is the activation-space node geometry in an
orthonormal basis, so δ there equals δ on the full centroids. Cross-checks the
probe on real geometry of known shape (personas/emotions read flat ≈ 0.19–0.21;
authored rings months/daypart ≈ 0.28).

`taxonomy_heap.py` — two animal taxonomies (shallow depth-3/24-node, deep
depth-5/37-node) and their ground-truth tree-distance matrices.

`run_taxonomy_delta.py` — the experiment. Generates an embodied corpus per node,
pools fresh full-dim per-layer centroids (no flat fit in the loop, so a flat
subspace cannot pre-flatten hidden curvature), and reports δ_rel per layer in both
Euclidean and the saklas Mahalanobis metric, plus the Hewitt-Manning check (does
*squared* activation distance track tree distance — the tree-embedding signature).
Centroids cache to a taxonomy-tagged `.npz` for re-analysis without the model.

`tree_signal_test.py` — label-permutation null. δ_rel is label-free; this asks
whether the activation geometry knows *this specific* tree by testing the true
r(d², tree) against the distribution under shuffled tree labels.

## Result (gemma-4-12b)

The taxonomy is encoded as a weak, early-layer, geometrically flat tree, and depth
does not recruit curvature. The tree alignment is statistically real and replicated
across both taxonomies (pooled r(d², tree) ≈ 0.15 over early-middle layers,
permutation p ≈ 0.01–0.02, ~24/28 layers individually p < 0.05) but flat: δ_rel
median 0.141 (depth-3) and 0.164 (depth-5). The deeper, diameter-8 tree gave
curvature more room and δ_rel moved further into the flat band, not toward the
hyperbolic one. The signal lives in early-to-middle layers and collapses to noise
by the late layers, with a small flat category-resurgence at the final layer.

This falsifies the hyperbolic hypothesis for this model and elicitation: a Poincaré
steering domain would impose curvature the model does not use. It corroborates the
flat/orthogonal account of semantic concept geometry (Park-Veitch, arXiv 2406.01506)
and contrasts with the strong tree geometry found for *syntactic* parse trees
(Hewitt-Manning structural probe). The one untried lever is elicitation (concrete
embodiment vs descriptive); since the structural lever (depth) gave a clean null,
expect elicitation to move the correlation magnitude rather than the geometry.

## Reproduce

```bash
# synthetic calibration (no model)
python validate_delta.py

# δ on fitted manifolds (no model; needs a fitted personas/emotions/months)
python delta_probe_fromfit.py

# the experiment (loads the model; ~15-25 min generation on MPS)
python run_taxonomy_delta.py --taxonomy deep google/gemma-4-12b-it
python tree_signal_test.py --taxonomy deep --npz centroids_deep_google__gemma-4-12b-it.npz
```

Run from this directory (the scripts import each other by module name).
