# Auto-manifold fitting (Part 2)

Plan for the second front door to saklas's manifold steering: fitting a
manifold from a heap of statement corpora (or a flat list of installed
concept selectors) without the user authoring coordinates. The fitter
derives coordinates from the activations themselves, by PCA or by
spectral embedding. Authored manifolds — theory-driven cases like
Russell's circumplex with declared valence/arousal axes — stay
first-class. This is a peer mode, not a replacement.

Part 1 of this multi-part feature (role-augmented extraction) is being
built in parallel elsewhere; nothing in this document depends on it.

## Background

Saklas already supports manifold steering end to end. A
`ManifoldDomain` (`saklas/core/manifold.py` — `BoxDomain`,
`SphereDomain`, `CustomDomain`) defines an intrinsic-dimensional
embedding plus a distance function. The authoring flow places named
nodes at user-supplied coordinates on that domain, each node holding a
small statement corpus.
`ManifoldExtractionPipeline.fit` in `saklas/core/extraction.py`
pools each node's mean activation through `compute_node_centroid`,
embeds the authoring coords through the domain, fits a per-layer PCA
subspace + r³ polyharmonic RBF interpolant
(`fit_layer_subspace`, `fit_rbf_interpolant`), and writes the per-model
tensor under `~/.saklas/manifolds/<ns>/<name>/`. The injection hook
(`subspace_replace`) decomposes activations into in-subspace and
orthogonal residual, blends the in-subspace component toward a
manifold point, restores the original norm.

The authored mode requires the user to know — or guess — a coordinate
system. For a circumplex of affect, declaring valence and arousal as
axes is a contribution; for a heap of 48 traits with no theory in
hand, demanding coordinates is friction that drives users toward
unprincipled coordinates. Part 2 closes that gap.

## What Part 2 adds

A second fitting mode, `discover`, that takes either a list of paths to
statement-corpus folders or a flat list of installed concept selectors
and derives node coordinates from the data. Two derivation methods,
both selectable per-fit:

**PCA — the safe linear default.** Stack the per-node activation
centroids at a chosen reference layer (the DLS-style middle layer used
elsewhere, or an explicit override), run `torch.linalg.svd`, pick
intrinsic dimension `k` as the smallest prefix whose cumulative
variance crosses a 70% threshold capped at `max_dim=8`, project the
centroids onto those `k` left singular directions to get coordinates,
wrap the result in `CustomDomain(k)` with the identity embedding.
Reproduces the existing flat-subspace fit a user would get by writing
the coords themselves once they knew the answer.

**Spectral — Laplacian eigenmaps.** Recovers curved-manifold topology
that PCA flattens. Build a symmetric k-NN graph over centroids
(default `k_nn = max(5, ceil(log N))`), weight edges with a heat
kernel `w_ij = exp(-d_ij² / (2σ²))` where `σ` defaults to the median
k-NN distance, form the symmetric normalized Laplacian
`L = I − D^{-1/2} W D^{-1/2}`, eigendecompose via `torch.linalg.eigh`,
drop the trivial constant eigenvector at index 0, take the next `k`.
The eigenvalue spectrum supplies a spectral-gap diagnostic for `k`:
pick `k` as the index of the largest gap `λ_{i+1} − λ_i` for
`1 ≤ i ≤ max_dim`, or override. Coordinates are the corresponding
eigenvector entries. Stays inside saklas's no-scipy rule.

Both methods feed the same downstream machinery — `CustomDomain(k)`
with the derived coords, then `fit_layer_subspace` per layer, then
`save_manifold`. Nothing under `hook_fn` changes; the injection path is
identical to authored manifolds.

### Architectural shift: per-model coordinates

This is the load-bearing change to surface honestly. In authored
mode, coordinates are model-independent — the user declares them
once, and a Gemma manifold and a Qwen manifold share node positions
even when their activations differ. In discover mode, coordinates are
*derived from* model-specific activations, so different models will
produce different layouts for the same heap of nodes. A 48-concept
PCA fit against `gemma-3-4b-it` will not embed the concepts the same
way as the same fit against `qwen3-8b`.

Storage layout follows the constraint. For a discover-fit manifold:

- `manifold.json` carries `name`, `fit_mode`, node entries with
  `{label, corpus_path}` or `{label, selector}`, and the source-corpus
  sha256 — no `coords` field, no domain spec.
- Per-model `<safe_model_id>.safetensors` carries the derived coords
  alongside the per-layer PCA subspace and RBF coefficients. The
  sidecar `.json` gains `fit_mode: "authored" | "pca" | "spectral"`
  and, for spectral, `bandwidth`, `k_nn`, `eigenvalues`, `gap_index`.
- `domain` is reconstructed from the per-model tensor on load as
  `CustomDomain(k)` with identity embedding. Authored fits keep their
  existing layout exactly.

Cross-model coordinate alignment via Procrustes is out of scope for
v1 — a fitted discover-manifold is valid only for the model it was
fit against. A future extension would reuse the existing
`alignments/` machinery that `vector transfer` uses today
(`saklas/io/alignment.py`) to ship a coord-space rotation between
models so the same authoring point steers comparable concepts across
backends. Note the shape, do not build it yet.

## Automated scenario + statement generation

The ambitious half of Part 2. The user hands the fitter a *flat list of
concepts* with no pole structure — for example
`["pirate", "caveman", "assistant", "scholar", "robot"]` — and expects
node corpora to materialize without authoring them by hand.

Saklas's existing extraction is contrastive-pairs throughout. For
`honest.deceptive`, `session.generate_scenarios` writes nine broad
domains, then `session.generate_pairs` writes a positive (honest) +
negative (deceptive) statement per scenario. The contrastive
structure is load-bearing for difference-of-means extraction: it pins
the *axis* that differs while everything else is held constant. For a
heap of N concepts with no declared contrast, there is no pole to
pair against. The generalization is clean once you see it.

**Scenario generation, generalized from K=2 to K=N.** Instead of
writing scenarios apt for one bipolar contrast, ask the model to write
scenarios apt for *all* concepts on the list — situations broad
enough that each listed concept could plausibly have a response. For
`["pirate", "caveman", "assistant", "scholar", "robot"]`, a fit
scenario is "asked how to deal with a sudden storm" — every concept
has a take. A bad scenario is "writing a Python decorator" — only
two have a take, and the rest will collapse to defaults. The prompt
asks for scenarios with purchase for every concept on the list.

**Statement generation, generalized from pairs to N-tuples.** For
each (scenario, concept) cell, write one statement exemplifying that
concept's response under that scenario. The output is N corpora of M
statements each, where corpus `i` exemplifies concept `i` and
statement `(i, j)` is concept `i` under scenario `j`. Crucially the
*scenario `j` is shared across the row* — this is the K>2 analogue
of paired-contrast structure, and it is what makes the resulting
per-concept centroids comparable. Without scenario sharing, the
centroid differences would mix concept signal with scenario signal,
and the PCA/spectral layout would surface the larger of the two
(usually scenario).

**Anti-allegory clause carries over.** The existing
`generate_scenarios` and `generate_pairs` carry a load-bearing
anti-allegory instruction that keeps non-human axes
(`deer/wolf`, `brick/feather`) literal rather than human-allegorical
— a deer is an animal that flees, not a metaphor for timidity. The
N-concept generator must preserve this. A `caveman` is a person
living in a cave with stone tools, not a metaphor for primitive
behavior; a `robot` is a machine with sensors and actuators, not a
metaphor for cold logic. The fit will go sideways the moment the
generator starts collapsing concepts to human-allegorical readings,
because every concept then drifts toward the same "stereotyped
human" centroid.

**Implementation slot.** A new method on `SaklasSession`:

```python
def generate_concept_statements(
    self,
    concepts: list[str],
    *,
    n_scenarios: int = 9,
    statements_per_concept_per_scenario: int = 5,
    seed: int | None = None,
) -> dict[str, list[str]]:
    """Generate per-concept corpora for an amorphous concept list.

    Returns ``{concept_name: list[statement]}`` with
    ``n_scenarios * statements_per_concept_per_scenario`` statements
    per concept, sharing scenarios across concepts within each
    statement index. Anti-allegory clause applied to both the
    scenario and the statement prompt.
    """
```

The method composes two calls. First, a single
`generate_scenarios`-style prompt that takes all `concepts` and writes
`n_scenarios` situations with purchase for the whole list. Second, for
each scenario, one prompt per concept asking for
`statements_per_concept_per_scenario` literal responses by that
concept under that scenario, with the anti-allegory clause inline.
Output is shaped as `dict[concept, list[statement]]` so each value is
a drop-in corpus for one heap-fit node.

Discoverability mirrors the existing pipeline: `session.py` is where
both `generate_scenarios` and `generate_pairs` live today, and the
new method belongs as their peer.

## File-level changes

`saklas/core/manifold.py`

- Add `derive_pca_coords(centroids, *, max_dim, var_threshold)` —
  returns `(coords, diagnostics)` where diagnostics is a dataclass
  carrying per-component variance and cumulative variance. Pure
  tensor, fp32, dependency-free.
- Add `derive_spectral_coords(centroids, *, max_dim, k_nn, bandwidth)`
  — returns `(coords, diagnostics)` with the full eigenvalue spectrum,
  detected gap index, bandwidth used, and the connectivity check. The
  graph-construction helper raises `ValueError` on a disconnected
  k-NN graph (a degenerate heap with isolated centroids cannot be
  embedded by Laplacian eigenmaps; recommend the user either raise
  `k_nn` or switch to PCA).
- A small `discover_coords(centroids, method, **kwargs)` dispatcher so
  the pipeline doesn't branch.

`saklas/core/extraction.py`

- Generalize `ManifoldExtractionPipeline.fit` to accept either an
  authored folder (current path, unchanged) or a `DiscoverFitSpec`
  carrying `{nodes: list[NodeSpec], method: "pca" | "spectral",
  max_dim, ...}`. The discover path skips `domain_from_spec`, runs
  the centroid pool over each node corpus, calls `discover_coords`,
  constructs `CustomDomain(k)` with identity embedding, then proceeds
  through `fit_layer_subspace` exactly as today.
- Cache invalidation: the sidecar `nodes_sha256` already covers
  corpus drift. Add `fit_mode` + (for spectral) `bandwidth`, `k_nn`,
  `max_dim` to the cache key so a refit with different hyperparams
  doesn't quietly hit a stale tensor.

`saklas/io/manifolds.py`

- Extend `ManifoldFolder.load` to recognize the discover-mode shape
  (no `coords` per node, no `domain` field). `create_manifold_folder`
  gains a `fit_mode` parameter and a `nodes` shape that takes
  `{label, corpus_path}` or `{label, selector}` instead of
  `{label, coords}`.
- `ManifoldSidecar` gains `fit_mode`, `coords` (only for discover
  fits — authored fits keep coords in `manifold.json`), and the
  spectral diagnostics.
- `min_nodes(n)` still applies: discover fits need at least `2n+1`
  nodes for the chosen intrinsic dimension. If the spectral gap or
  PCA threshold picks a `k` for which the heap is undersized, the
  fitter raises with a message naming both `n` and the floor.

`saklas/cli/parsers.py`

- `_build_vector_manifold` gains a `discover` subverb peer to `fit`:

  ```
  saklas vector manifold discover <name> \
      --from FOLDER_OR_SELECTORS... \
      [--method pca|spectral] \
      [--max-dim N] \
      [--k-nn K] \
      [--bandwidth SIGMA] \
      [-m MODEL]
  ```

  `--from` is variadic: a list of corpus paths, a list of installed
  selectors (resolved via `saklas.io.selectors`), or a single path
  to a directory of corpora.

- A separate `generate` subverb that wraps the new
  `session.generate_concept_statements` and writes the resulting
  corpora into a manifold folder so `discover` can pick them up:

  ```
  saklas vector manifold generate <name> \
      --concepts pirate caveman assistant scholar robot \
      [--n-scenarios 9] \
      [--statements-per-concept 5] \
      [-m MODEL]
  ```

  Output is a manifold folder under `~/.saklas/manifolds/<ns>/<name>/`
  with `manifold.json` in discover shape and `nodes/NN_<label>.json`
  files holding generated statements. The user can then run
  `saklas vector manifold discover <name>` to fit it, or chain them
  via `discover --generate` as sugar.

`saklas/server/manifold_routes.py`

- `POST /saklas/v1/manifolds` accepts a discover-mode payload
  (`fit_mode: "pca" | "spectral"`, `nodes: [{label, corpus_path |
  selector}]`, no `coords`).
- `POST /saklas/v1/manifolds/{ns}/{name}/fit` accepts a
  `fit_mode` + method-specific hyperparameters body when the manifold
  was created in discover shape; ignores them when it was authored.
- `GET /saklas/v1/manifolds/{ns}/{name}` reports `fit_mode` and, when
  discover, the per-model coords and diagnostics so the webui can
  render the derived layout.
- `POST /saklas/v1/manifolds/generate` wraps
  `session.generate_concept_statements` and returns a freshly-written
  discover-shape manifold folder.

## Diagnostics surface

Both methods emit diagnostics that surface in the CLI's `vector
manifold show` output and the webui inspect panel. A user who runs a
discover fit should be able to tell whether the resulting layout was
well-supported by the data.

PCA returns:

- Per-component variance (singular values squared, normalized).
- Cumulative variance — how much of the centroid spread the first `k`
  components capture.
- Picked `k` and the threshold that picked it.

Spectral returns:

- Full eigenvalue spectrum (first `max_dim + 5` non-trivial
  eigenvalues).
- Detected gap index and gap magnitude — the spectral-gap diagnostic
  is the one knob that tells the user "the data has a clean
  k-dimensional structure" versus "no clean cut, pick a dim by hand".
- Bandwidth `σ` used (median k-NN distance unless overridden).
- k-NN connectivity result — pass / fail with the count of connected
  components if it fails.

Both diagnostics live in the per-model sidecar so they survive across
sessions, and the webui can render the spectrum as a small histogram
with the picked-`k` cut marked.

### Small-heap caveat

Spectral embedding is noisy below roughly 50 nodes — too few centroids
to estimate the k-NN graph's heat-kernel weights stably, and the
spectral gap collapses into the eigenvalue noise floor. The current
bundled set is 26 concepts and the projected post-augmentation set is
~48; both sit in the noisy regime. Recommend PCA as the default for
the current bundled and bundled-adjacent heap sizes, and surface
spectral as the right choice once heaps cross ~50 nodes and start to
hint at curved structure. Document this in the `discover` help text,
not just here — a user reaching for spectral on the 26-concept
bundled heap is going to get a layout that looks meaningful but
isn't.

## Verification

All discover-mode unit tests are CPU-only — the fit math is pure
tensor over centroids, which can be synthesized without a model.

**Circle-topology recovery (spectral, the headline test).** Generate
`N = 80` synthetic centroids in R^32 by placing points uniformly on a
2D circle and embedding the circle into a random 2-plane of R^32 plus
small isotropic noise. Run `discover_coords(..., method="spectral",
max_dim=4)`. Assert the spectral gap is at index 2 (S^1 has two
non-trivial Laplacian eigenvalues before the gap), the recovered
2-coord embedding wraps once around the origin (sort by atan2 of
recovered coords, check the original angular order is preserved up to
reflection), and PCA on the same data picks `k=2` flat (no gap), so
spectral and PCA disagree the way they should on curved input.

**Flat-subspace recovery (PCA, the inverse test).** Generate `N = 60`
synthetic centroids in R^32 with three meaningful directions and
isotropic noise everywhere else. Run `discover_coords(...,
method="pca")`. Assert the cumulative variance crosses 70% at `k=3`,
the recovered 3-coord layout is an orthogonal rotation of the
generating coords, and spectral on the same input agrees with PCA up
to rotation+reflection (the agreement-iff-flat invariant — PCA and
spectral coincide on data that genuinely lies on a flat manifold, and
diverge on curved input).

**Agreement-iff-flat (cross-method).** A composite test: on flat
synthetic input, the Procrustes distance between PCA and spectral
coords should be near zero; on circle synthetic input, it should be
large. This is the discriminating test for "did we wire the methods
up correctly" — if they agree on curved data, spectral is broken; if
they disagree on flat data, PCA picked a wrong subspace or spectral
collapsed.

**Disconnected-graph rejection.** Generate two clusters of 20 points
in R^32 with a wide gap and a small `k_nn`. Assert
`derive_spectral_coords` raises a `ValueError` naming the component
count.

**Min-nodes floor.** Spec a discover fit with `k=4` and 8 nodes;
assert it raises with a message naming `min_nodes(4) = 9`.

**Cache key invalidation.** Fit with PCA, then refit the same folder
with spectral; assert the safetensors file changes and the sidecar
records the new `fit_mode`. Refit with PCA and a different
`max_dim`; assert the tensor refits rather than hitting cache.

**Statement-generation smoke (CPU-only, mocked LM).** Mock
`session.generate_scenarios` and the per-(concept, scenario)
statement call. Assert
`generate_concept_statements(["pirate", "caveman", "assistant"])`
calls the LM with all three concept names in the scenario prompt,
shares the scenario index across the three resulting corpora, and
returns the right shape. Assert the anti-allegory instruction is
present in both the scenario and the statement prompts by string-
matching the prompt text — a regression where the clause goes
missing is exactly the failure mode this test catches.

**Discover-mode round-trip.** Author a discover-mode manifold folder
with synthetic node corpora, run the fit through
`ManifoldExtractionPipeline.fit` against a tiny mocked model handle
(no real forward pass — patch `compute_node_centroid` to return
synthetic centroids), load the resulting `Manifold` via
`load_manifold`, assert the domain is `CustomDomain(k)` with identity
embedding, and assert a `subspace_replace` call on a synthetic
activation lands on the manifold target.

**GPU smoke (gated on CUDA/MPS, opt-in).** In `test_smoke.py`, add a
discover-mode fit on a real 5-concept heap against `gemma-3-4b-it`
using `generate_concept_statements` end-to-end. Assert the fit
completes inside the existing manifold-fit time budget, the per-
layer RBF reconstructs each centroid to within the existing fit
tolerance, and a steered generation at the centroid of one node
produces output that probes for that concept score higher than at the
opposite end of the layout. This is the integration test that
catches "everything passes in isolation but the actual model pipeline
is broken" failures.

## Order of work

1. `derive_pca_coords` + `derive_spectral_coords` + unit tests on
   synthetic data. Lands first because everything else depends on
   these returning sane coords.
2. `ManifoldExtractionPipeline.fit` discover branch + sidecar
   extensions + cache-key invalidation tests.
3. `generate_concept_statements` on `SaklasSession` + mocked-LM
   tests for shape and anti-allegory preservation.
4. CLI: `vector manifold discover` and `vector manifold generate`,
   wired through `runners.py`.
5. Server routes: discover-mode payloads on `POST manifolds` and
   `POST manifolds/{ns}/{name}/fit`, plus
   `POST manifolds/generate`.
6. Webui inspect panel: render `fit_mode`, the PCA variance bars or
   the spectral spectrum, the picked-`k` cut, and the per-model
   layout. New work in `webui/src/lib/manifolds/` — peer of the
   existing authored-manifold inspector.
7. GPU smoke test added last, once the pipeline is end-to-end
   working against a synthetic mocked model.

The Procrustes cross-model alignment extension is deferred. Note it
in `saklas/io/manifolds.py` as a TODO referencing
`saklas/io/alignment.py` and `vector transfer` so the future
implementer finds the existing machinery rather than rebuilding it.
