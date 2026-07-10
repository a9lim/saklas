# Dynamic roles — the cast model

*Status: **visual + composer unification ships in the UI redesign; engine work
sketched here, deferred** (2026-07-10). Companion to `docs/plans/sae-pillar.md`
in spirit: the redesign builds the surface so the engine work lands without
re-layout.*

## Vision (a9)

Roles become fully dynamic: "deer", "pirate", "user", "assistant" are all
swappable labels, and **both the human and the model can write for any role**.
The chat-app framing (blue human, green machine, two fixed seats) dissolves
into a loom-native one: a transcript is a *script*, turns are lines spoken by
cast members, and who typed a line is provenance, not identity.

## What already exists

Closer than it looks. Today's engine is a **two-seat theater with swappable
nameplates**:

- `role_label` is stamped per-node at generation/commit time (the SamplingStrip
  `user as` / `reply as` boxes ride the WS `sampling` block), immutable per
  turn, rendered via `roleDisplayLabel`/`roleGlyphLetter`.
- The human can author either seat: `commit_role="user"` / `"assistant"`, plus
  answer-prefill for seeding the assistant seat.
- Role-label substitution at render time (`ROLE_HEADERS`) works on
  qwen/gemma/llama/glm/gpt_oss/talkie; mistral3/ministral3 raise
  `RoleSubstitutionUnsupportedError` (positional `[INST]`, no label in the
  rendered string).
- Raw mode is the degenerate case already in hand: one flat buffer, no roles.

## The cast model

A turn is `(label, seat, provenance)`:

- **label** — arbitrary slug, what the reader sees ("deer", "pirate", "user").
- **seat** — the *structural* role the chat template renders (user-side /
  assistant-side / system). An implementation detail of templated models, not
  an identity.
- **provenance** — committed (human-typed) vs generated (model-authored).
  Research-relevant (which text did the model write?) but visually subtle.

Today label is bound to seat ("user as" labels the user seat). The vision
decouples them.

## Gaps, ordered by cost

1. **Model speaks the user seat** (small). A `generate_role: "user"` on the WS
   `generate` frame: render the prompt so it ends with a user-seat header
   (carrying its label) and decode. Alternation stays satisfied — it's still
   turn-by-turn. New node = seat `user`, provenance `generated`. Mostly prompt
   render + node bookkeeping.
2. **N-party cast (>2 speakers)** (the real project). Chat templates enforce
   two-seat alternation (gemma's template raises on non-alternating roles), so
   a genuine three-way scene can't render through `apply_chat_template`.
   Options: (a) map the cast onto alternating seats when the speaking order
   happens to alternate; (b) raw-mode-style rendering with role-marker text —
   the flat buffer generalized from one voice to N; (c) per-family header
   synthesis (hand-render `<start_of_turn>{label}` sequences), which is
   `ROLE_HEADERS`' logic promoted from substitution to construction. (c) is
   the likely winner on families with string role labels; (b) is the universal
   fallback and base-model path. Needs a support matrix like `ROLE_HEADERS`.
3. **Cast manager UI** (additive once 1–2 exist). Named cast members, per-turn
   speaker picker, loom filter/diff/transcript awareness of labels (transcript
   YAML already carries `role_label`; N-party export needs a seatless schema
   rev).

## The saklas-native frontier: cast member = (label, recipe)

Speculation, but the exciting kind: a cast member could carry a **steering
recipe** alongside its label — "pirate" speaks under `0.6 personas%pirate`,
"deer" under its own expression, swapped automatically as the speaker changes
(the per-turn analogue of what `nearest_node_role` already does for persona
manifolds at steer time, and of the recipe_override the A/B shadow path uses
per-generation). The roleplay scaffold and the steering rack meet: a scene
becomes a set of (voice, position-on-the-manifold) pairs. If this lands, the
cast manager is not a chat feature — it's a steering surface.

## What the redesign ships now (pure UI, no engine change)

- **One neutral treatment for every turn.** Color encodes nothing about role
  (under the redesign's hue ontology, roles aren't a "space", so they carry no
  hue). Identity lives in the role chip: glyph letter + label, arbitrary
  strings first-class.
- **System turns render as stage directions** — full-width, italic, dim; not a
  speaker, a note about the scene.
- **`speaking as` promoted into the composer** — the seat labels move from the
  SamplingStrip boxes to a first-class chip beside the input (same
  `user_role`/`assistant_role` client state, same wire block). This is the
  visible step toward the cast model.
- **Provenance stays subtle** — generated turns carry their existing
  mean-logprob/ppl badge; committed turns simply lack it. No color.

## Open questions

1. Does a recipe bind to the *role* (every pirate line steers alike) or the
   *turn* (a role can drift scene-to-scene)? Role-level default with per-turn
   override is the likely shape.
2. Loom semantics for seatless N-party trees — sibling sort, diff alignment,
   and the filter grammar all currently assume the two-seat rhythm somewhere.
3. Transcript schema rev for N-party (current YAML is role+role_label pairs).
4. Per-family support matrix for header synthesis; mistral-family fallback is
   raw-marker rendering or nothing.
