# concept_geometry

Instruments for one question: when a model represents a structured concept family
(a taxonomy, a value circumplex, an affect space), does it preserve the family's
natural topology — a tree, a ring — or flatten it? Two experiments share the same
machinery: generate an embodied corpus per node, pool fresh full-dim per-layer
centroids (no fit in the loop, so a low-rank subspace can't pre-flatten anything),
and measure the geometry of the centroids in saklas's whitened Mahalanobis metric.

## Probes

`delta_hyperbolicity.py` — Gromov four-point δ-hyperbolicity, exact over all
basepoints (Fournier–Ismail–Vigneron max-min matrix form) with an O(n⁴) brute-force
cross-check. Reports the scale-free δ_rel = δ/diam. A tree is 0; hyperbolic space is
small; flat is moderate; sphere/ring is large.

`validate_delta.py` — synthetic calibration for the δ probe. Establishes the bands:
tree 0.00, hyperbolic 0.11, euclidean-flat 0.19–0.22, ring/sphere 0.42–0.50.

`validate_circumplex.py` — synthetic calibration for the ring probe on ring/line/
random geometries. The key check is that the eigenvalue planarity λ2/λ1 separates a
true ring (≈0.99) from a bipolar line (≈0.08) even when the line's sorted order
fakes high cyclic adjacency.

`delta_probe_fromfit.py` — δ on already-fitted saklas manifolds, no model load
(for an affine fit `layer_N.node_coords` is the activation-space node geometry in an
orthonormal basis). Cross-checks the probe on known shapes: personas/emotions read
flat (≈0.19–0.21), authored rings months/daypart ≈0.28.

## Experiment 1 — is a taxonomy a tree? (δ-hyperbolicity)

`taxonomy_heap.py` (shallow depth-3/24-node and deep depth-5/37-node animal trees),
`run_taxonomy_delta.py` (the experiment + Hewitt-Manning squared-distance check),
`tree_signal_test.py` (label-permutation null on the tree alignment).

Finding (gemma-4-12b): a weak, early-layer, geometrically flat tree. The alignment
is statistically real and replicated across both taxonomies (pooled r(d²,tree) ≈
0.15, permutation p ≈ 0.01–0.02), but flat: δ_rel median 0.141 (depth-3) and 0.164
(depth-5). Depth did not recruit curvature — the deeper, diameter-8 tree moved
δ_rel further into the flat band, not toward hyperbolic. The signal lives in
early-to-middle layers and collapses to noise by the late layers.

## Experiment 2 — is the Schwartz value circumplex a ring?

`values_heap.py` (Schwartz's 10 values in canonical circular order), `run_circumplex.py`
(whitened consensus MDS + cyclic-order recovery + planarity spectrum),
`circumplex_layers.py` (layer-resolved planarity + dominant-axis naming).

Finding (gemma-4-12b): no planar ring. The consensus spectrum has one dominant axis
(λ2/λ1 = 0.31; a ring needs ≈1), carrying 53% of the whitened spread. The cyclic
order weakly recovers Schwartz (consensus adjacency 5/10, p ≈ 0.050; best in the mid
band, layers 13–29, at 6/10) but never forms a plane. δ_rel is uninformative here —
a chordally-embedded circle reads flat — so planarity, not δ, is the discriminator.
The dominant axis is a control/striving-vs-autonomy/care contrast, not a clean
Schwartz higher-order dimension.

## The combined picture

Three structured families, three flattenings: taxonomy → weak flat tree, circumplex
→ flat dominant axis, and (already in saklas) personas/emotions → flat fans. Even
`months` auto-fits flat — its ring is authored in `manifold.json`, not discovered.
So on gemma-4-12b, structured concept geometry is held in compressed, near-flat,
dominant-axis form, and funky geometry has to be imposed by authoring rather than
discovered. This corroborates the flat/orthogonal account of concept geometry
(Park-Veitch, arXiv 2406.01506) and contrasts with the strong tree geometry found
for syntactic parse trees (Hewitt-Manning structural probe). The open lever is
cross-model: saklas fits per-model geometry, so whether another architecture
recovers a ring or a tree where gemma flattens is untested.

## Reproduce

```bash
# calibration (no model)
python validate_delta.py
python validate_circumplex.py
python delta_probe_fromfit.py        # needs a fitted personas/emotions/months

# taxonomy (loads the model; ~15-25 min generation on MPS)
python run_taxonomy_delta.py --taxonomy deep google/gemma-4-12b-it
python tree_signal_test.py --taxonomy deep --npz centroids_deep_google__gemma-4-12b-it.npz

# circumplex (loads the model; ~7 min generation)
python run_circumplex.py google/gemma-4-12b-it
python circumplex_layers.py
```

Run from this directory (the scripts import each other by module name). Centroid
`.npz` files cache to the working directory so re-analysis never re-runs the model.
