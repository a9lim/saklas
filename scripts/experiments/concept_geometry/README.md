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

`geometry_stress.py` — synthetic ground-truth stress harness for the geometry
*detector itself* (`select_topology`), no model load. Every finding above asks
whether the model flattens; this asks the prior question of whether the detector
is trustworthy when it says "flat". It runs known-topology clouds (line, blob,
plane, arc, curve, circle, ellipse, faint ring, T2/T3 torus, grid, sphere,
persona-fan) through the detector under isotropic / anisotropic / rogue-channel
whiteners and reports a confusion matrix, the flat↔curved decision boundary, the
periodic recall/precision envelope, and stability sweeps. See "Detector
validation" below.

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

## Detector validation — is `select_topology` trustworthy?

`geometry_stress.py` maps the detector's operating characteristics on synthetic
clouds of known topology, in the conditions real fits run in (anisotropic,
rogue-channel activation space whitened by the Fisher metric).

What's solid: rogue-channel **invariance is perfect** — a whitener with channels
at 200× the background gives byte-identical verdicts to an isotropic one, so the
Fisher metric divides massive activations out exactly as intended. Periodic
detection has 100% T1 recall and 0% false-positives across blob/grid/fan/arc/line;
the faint-cycle fallback recovers rings down to ~8% modulation; the read is
deterministic and stable in `persistence_frac`. Envelope edges: a torus needs ≥7
points per loop (coarser tori fill inside the `eps_max=2·eps_c` window and read
flat), and T3+ is out of practical reach (the points-per-loop needed pushes K past
the periodic regime). Neither arises in a bundled manifold.

What it found — **a flat-bias bug, now fixed.** The flat-vs-curved decision
compared the flat candidate at its PCA variance-threshold dim against the curved
candidate at its spectral eigenvalue-ratio-cliff dim — and that cliff
systematically *undershoots* (one dominant Fiedler mode picks k=1). So a curved
manifold linearly embedded in a k-plane read **flat**: the flat affine fit
reconstructs the in-plane curve near-perfectly, while the under-dimensioned curved
candidate couldn't match reconstruction it would *win* at matched dim (proved
directly — give the curved RBF the flat candidate's coords and its GCV → 0). This
is the mechanism that could have made "the model is flat" a detector artifact. The
production evidence: the bundled `personas` fit had flat dim 8 vs spectral dim 2,
`emotions` flat dim 3 vs spectral dim 1 — both decided by a comparison rigged
against curved. The fix floors the curved candidate to the flat dim
(`min_dim=k_flat` in `select_topology`, using the floor `derive_spectral_coords`
already carried for exactly this undershoot), so the two compete at matched
expressiveness; a synthetic curved-in-plane manifold now reads curved, and no
genuinely flat shape (line/blob/grid/plane/persona-fan/arc) regresses. Guarded by
`tests/test_manifold_topology.py`.

The bearing on the findings above: the δ-hyperbolicity and planarity probes are
purpose-built and don't route through `select_topology`, so the taxonomy and
circumplex conclusions stand. `personas`/`emotions` use `fit_mode=auto`, so
re-fitting them after the fix gives a fair flat-vs-curved verdict.

**Validated (gemma-4-12b, re-fit post-fix): both stay flat under the fair
comparison.** The dim-match floored the curved candidate to the flat dim and its
GCV dropped to an honest value, but flat still wins comfortably:

| manifold | flat (GCV)  | spectral before fix | spectral after fix | verdict |
|----------|-------------|---------------------|--------------------|---------|
| emotions | dim-3, 237  | dim-1, 1447 (rigged)| dim-3, 571 (fair)  | flat    |
| personas | dim-8, 251  | dim-2, 1417 (rigged)| dim-8, 756 (fair)  | flat    |

So the flatness was real, not an artifact of the spectral undershoot — flat beats
a *fairly-dimensioned* curved fit by ~2.4–3×. The "gemma flattens structured
concept geometry" account is strengthened, not overturned: the persona fan and the
PAD affect space are genuinely flat. The winning flat tensors are unchanged (the
flat-pca GCVs matched the pre-fix fits to 15 digits); only the recorded
`topology_candidates` margin is now honest. The fix matters for *future* discover
fits — a curved-in-a-flat-subspace concept on another model would now be detected
rather than mislabelled flat.

## Reproduce

```bash
# detector validation (no model) — confusion matrix, flat<->curved boundary,
# periodic envelope, stability; --quick for fewer seeds
python geometry_stress.py all --quick
python geometry_stress.py flatcurved      # the flat-bias boundary map

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
