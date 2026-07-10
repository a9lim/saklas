# SAE pillar — live runtime sketch

*Status: **specced, deferred** (2026-07-10). This documents the design so the
UI redesign can build the fourth pillar tab against a stable contract; no
engine code lands from this doc. Written to mirror the J-lens architecture,
which is the proven template for a per-model read/steer/gate surface.*

*Epistemic marks: SAE mechanics are standard-result; the architecture here is
synthesis (transplanting the jlens shape); every calibration claim is
speculation until a live pass.*

## Framing

The dashboard's mental model is four coequal instruments over one bench:
**subspace probes, manifold probes, SAE features, Jacobian lens** — each with
the same three verbs (observe / steer / gate), each lowering to the same
unified injection kernel for the steer verb. Saklas today has SAE support only
at fit time (`--sae <release>` reconstructs node centroids through the SAE
before the manifold fit). This sketch adds the *live* runtime: a resident SAE,
per-step feature readout, `sae/` steering atoms, feature probes and gates.

The J-lens is the template because it already solved the same problems:

| Concern            | J-lens answer                                   | SAE transplant                                     |
| ------------------ | ----------------------------------------------- | -------------------------------------------------- |
| Per-model artifact | `models/<safe>/jlens.safetensors` + sidecar     | SAE weights (HF/sae-lens) + local metadata cache    |
| Live readout       | `POST /lens/live`, WS `token.lens_readout`      | `POST /sae/live`, WS `token.sae_readout`            |
| Steer atom         | `jlens/<word>` → `W_U[v] @ J_l` direction       | `sae/<id>` → decoder row `W_dec[i]`                 |
| Probe/gate channel | readout channel, `coords = (strength,)`         | readout channel, `coords = (activation,)`           |
| Namespace          | `jlens` reserved (authoring raises)             | `sae` reserved (authoring raises)                   |
| Scoring site       | post-forward on the capture's latest slice      | same tap, one encoder matvec                        |

## Runtime

- **Loading.** `saklas sae load <release>` (CLI) / `POST
  /sessions/{id}/sae/load` (server; background job like `lens/fit`, polled).
  Resolve through the sae-lens registry (gemma-scope for the gemma family,
  llama-scope etc.); cache weights under HF hub cache as usual, metadata under
  `~/.saklas/models/<safe_model_id>/sae/<release>.json`. One release resident
  per session in v1 (multi-release is an open question below). The release
  pins a hook layer (or small set); we do NOT sweep all layers — SAE residency
  is per-layer expensive where the lens is cheap-ish (fp16 J_l per layer vs
  d×F encoder/decoder pairs; a 16k-feature SAE on a 4b model is ~2×
  d_model×16k×2 bytes per layer).
- **Session integration.** The SAE never enters the forward hooks for capture;
  the live reader consumes the capture's latest residual slice post-forward at
  the token tap, exactly like the lens reader — zero new hooks, steering
  fast-path/compile eligibility untouched.
- **Validate route.** `POST /sessions/{id}/sae/feature/validate` (`{id}` →
  `{id, label?}`) mirroring `lens/token/validate`: the UI's add forms check
  before mutating the rack.

## Read surfaces

- **Live readout.** While on (`POST /sae/live` `{enabled, top_k?}`), each
  decode step encodes the hook-layer residual once (`acts = σ(E h + b)`, one
  matvec + top-k) and the WS `token` frame carries `sae_readout:
  [feature_id, activation, label?][]` strength-descending. Session info
  reports `live_sae` for rehydration. The finalize aggregate pools the last
  content token from the capture tail ring, same as the monitor/lens roster.
- **Probes + gates.** `add_probe("sae/9143")` lands in a session SAE-probe
  registry (not the Monitor — no whitener, no direction fold; same shape as
  the lens-probe registry). Reading is ONE channel — `coords =
  (activation,)`, the feature's (post-nonlinearity) activation at the hook
  layer. Gates: `@when:sae/9143 > 3.0`. A gate forces its per-step encode
  regardless of the live toggle (a gate can't fire on aggregates); an SAE-only
  gate does not force per-token monitor scoring. Activation units are the
  SAE's own (not normalized) — apples-to-apples across tokens for one
  feature, NOT across features; the UI should normalize per-card by a running
  max or the feature's `max_act` from metadata when available.
- **Token drilldown.** A fourth drilldown tab (`sae`) mirroring the j-lens
  tab: `GET /sessions/{id}/sae/token-readout?node_id=&raw_index=` — replay
  the node's raw prefix under recipe steering, encode at the clicked
  position, return top-k features. Same exactness caveats as the lens
  variant (affine terms exact, phase/gated terms don't reproduce);
  `steered=false` for the counterfactual.

## Steer surface

- **Atoms.** `sae/<id>` is an ordinary `ns/name` atom; the direction is the
  decoder row `W_dec[i]` at the SAE's hook layer, registered lazily into the
  profile registry (a `register_sae_direction` sibling of
  `register_jlens_direction`), steering only — probes read the readout
  channel. Single-layer by construction (the one place the pillar differs
  structurally from lens atoms, which span the band). Composes into the
  merged affine subspace like any vector term.
- **Ablation.** `!sae/<id>` in v1 is directional mean-ablation of the decoder
  row through the existing kernel — consistent with every other `!` term.
  True feature *clamping* (encode → zero the feature → decode → replace) is a
  different kernel (encode-modify-decode, not subspace projection) and is
  explicitly out of scope for v1; if wanted later it arrives as a distinct
  operator, not an overload of `!`. (Note the known kernel finding: `!` is
  currently coefficient-insensitive and coherence-breaking on gemma-3-4b —
  whatever fixes that fixes it for SAE atoms too.)
- **Calibration.** Expect SAE atoms to run hot and narrow like lens atoms
  (single sharp direction, not a distributed contrast; lens sweet spot was
  α≈0.3 with shatter at α≥0.5). The whitened along-normalization should tame
  scale, but the coherent band needs a live calibration pass before the
  recommended-α hint lands in the UI. Speculation until measured.

## Labels

Feature labels come from Neuronpedia (API or bulk export) when the release is
covered: cached at `~/.saklas/models/<safe>/sae/<release>-labels.json`,
fetched lazily per feature id with offline-first behavior (no label ⇒ the id
renders bare; never block a readout on a network call). The label is display
metadata only — never part of the grammar.

## Grammar + reservation

- `sae` joins `jlens` as a reserved manifold namespace (authoring under it
  raises).
- Atom shape `sae/<id>` where `<id>` is the integer feature index. If
  multi-release residency lands, the qualified form becomes
  `sae-<release>/<id>` via the existing `ns/name` machinery — the bare `sae/`
  prefix always means "the resident release."
- Gate shape `@when:sae/<id> <op> <num>` flows through the one `ProbeGate`
  verbatim, matching the keys the SAE registry emits.

## UI (built now, runtime-gated)

The redesigned dashboard ships the SAE pillar tab immediately, gated on a
session-info `sae_loaded` capability flag the way the J-LENS tab gates on
`jlens_fitted`:

- Unloaded state: a designed empty state with a release picker (registry
  suggestions filtered by model family) driving the background load route.
- Loaded: STEER section (feature-atom cards, per-card α, add-by-id/search
  form through the validate route) + merged PROBE section (pinned feature
  cards above the live top-k discovery cards, □/■ pin glyphs, live toggle in
  the header) — the exact JLensPanel shape, gold accent.
- Card statline: `sae/9143` · label ("citrus & fruit vocabulary") · `L14`
  layer chip · activation bar + sparkline. The shared card grammar carries
  over unchanged; the layer strip degenerates to a single-layer chip.
- Manifold builder finally exposes the *existing* fit-time SAE variant
  (`--sae`) in its AdvancedSection — unrelated to the runtime but it
  completes the pillar's authoring story.

## Open questions

1. **Multi-release / multi-layer residency** — v1 pins one release; is one
   hook layer enough for the research loop, or do we want two (mid + late)?
   Memory budget on MPS is the constraint.
2. **Activation reporting** — post-ReLU (gemma-scope JumpReLU: post-threshold)
   is the natural readout; do we also want pre-activation for gates near
   threshold?
3. **Clamping kernel** — the encode-modify-decode steering mode (true feature
   clamp / feature amplification à la Anthropic's steering work) is a genuine
   second injection kernel. Defer until the subspace kernel story has fully
   settled.
4. **Which releases per tested arch** — gemma-scope covers gemma-2/3;
   coverage for qwen/llama variants to be surveyed at build time.
5. **Throughput** — one d×F matvec per decode step while live; measure
   against the 85% throughput gate before defaulting it on in `serve`.
