# Manifold composition (Part 3, deferred frontier)

Plan for the open frontier of saklas's manifold steering: composing
two (or more) manifolds at the same layer. Today's engine refuses
overlap with a hard `SteeringExprError`. This document captures why
that's the right floor for now, what a composition operator would
look like, and the rough shape of the path forward. Nothing here is
implemented yet — the persona-manifold + label-position phases
(role-augmented manifolds, label-form positions, bare-name
resolution) ship without touching this constraint.

## Background

Manifold steering is destructive at the layers it covers. The
injection primitive `subspace_replace` (`saklas/core/manifold.py`)
decomposes the running activation `h = h_par + h_perp` along a
per-layer PCA basis, blends `h_par` toward the manifold's target
point by the coefficient, and keeps `h_perp` verbatim. The
in-subspace component is overwritten — not added to, not
projected through. That's the geometric move that makes manifold
steering follow learned curvature instead of cutting through
off-manifold space the way a straight chord would (Goodfire,
arXiv 2605.05115).

Composition is the question of what happens when two manifolds
both want to overwrite the in-subspace component at the same
layer. `SteeringManager.add_manifold` raises `SteeringExprError`
on overlap today, because the existing primitive has no answer:
running `subspace_replace` twice in sequence doesn't compose
either as a sum or as a chained projection. The first overwrite
discards information the second one would need; the second
overwrite discards information the first one wrote.

This is genuinely an open research frontier, not a 10-line fix.
The paper itself only steers one manifold at a time.

## Why accept the constraint for now

The user-facing impact, with the persona-manifold work in:

- Bare-name resolution (Phase C) means `0.7 pirate + 0.3 angry`
  parses as `persona%pirate + angry.calm` cleanly. The vector and
  the manifold compose fine (vector adds to `h_perp`, manifold
  overwrites `h_par`) — they're in disjoint subspaces by
  construction.
- Two manifolds — `persona%pirate + affect%angry` — raises today.
  This is the case worth fixing, but it's also the case where the
  user is asking for something the existing math can't deliver
  without a new operator.

A workable interim: when a user composes two manifolds and the
fit-layer sets overlap, the engine could *split* the layer set
(give the lower-coefficient manifold whichever layers it shares
with the higher-coefficient one, the higher takes the rest).
That's still arbitrary — there's no principled reason to think
the per-layer split preserves either manifold's intended
geometry — but it would stop the hard raise. Worth prototyping
behind a flag (`SAKLAS_MANIFOLD_COMPOSE=split`) once we have a
concrete dual-manifold use case to evaluate against.

## What a composition operator would look like

Three candidate shapes, none satisfying alone.

### Sequential (write-then-write)

Run `subspace_replace` for manifold A first, then manifold B on
the result. Order-dependent, non-commutative, and the second
operation loses A's signal entirely if the subspaces overlap.
Trivially wrong for the symmetric case (a + b == b + a should
hold for a composition operator).

### Weighted-blend in a joint subspace

Concatenate A's basis and B's basis into one combined basis at
each shared layer, re-orthogonalize, project `h` onto the joint
subspace, blend toward `α_A * target_A + α_B * target_B` (or
some other affine combination), restore norm. Symmetric,
order-independent. But the joint basis is no longer either
manifold's subspace — the curvature each manifold encodes (its
RBF interpolant's shape) is flattened to a single point under
the projection. The geometry of "follow the affect manifold
while remaining a pirate" is exactly what the user means; this
operator doesn't deliver it.

### Curvature-aware joint surface

Treat the two manifolds as defining a product manifold A × B,
fit a fresh RBF interpolant on a product grid of node pairs
(every A-node × every B-node centroid), and steer to a point on
the product. Correct geometry — the product manifold *is* "follow
A's curvature while at a position on B." But it's expensive
(K_A × K_B forward passes for the joint fit), it requires both
manifolds to have been authored against compatible corpora
(otherwise the product centroids are noise), and the discover
mode's per-model coord derivation has no obvious extension to
product spaces. Probably the right answer; far more work than
the persona-manifold + label-position work currently shipping.

## Path forward

1. Land the persona + label-form work without touching the
   constraint. Document the raise loudly so users don't trip on
   it. (This document.)
2. When a concrete dual-manifold use case lands, prototype the
   `split` interim behind a flag and evaluate it on the specific
   case. Don't ship a default-on operator that's known-wrong
   geometrically.
3. The product-manifold path is a separate document and a much
   bigger lift; defer until we have a research need for it.

## Cross-model Procrustes alignment (adjacent open item)

`saklas/io/manifolds.py::create_discover_manifold_folder` carries
a deferred TODO for cross-model alignment of discover-mode
coords. Same family of problem — the manifold's geometry is
per-model, and shipping a Procrustes rotation alongside the
per-model tensors lets an authoring coordinate on the source
map to a steering point on a target. Independent of composition
but lives in the same backlog. The machinery exists for steering
vectors (`saklas/io/alignment.py`, `vector transfer`); a peer
`transfer_manifold` reusing the same alignment cache is probably
the right shape.

## Out of scope for this document

- Persona-manifold steering (shipped — role-augmented per-node
  centroids, role-paired steering via nearest-node lookup).
- Label-form positions in the grammar (shipped — `<m>%<label>`
  parses alongside `<m>%<coords>` and round-trips).
- Bare-name resolution to manifold labels (shipped —
  `io.selectors.resolve_manifold_label` + `resolve_bare_name`
  with cross-tier ambiguity raise).
- Multi-manifold composition. (This document — open frontier.)
