# Dynamic roles — the cast model

*Status: **unified composer + phases 1–3 landed** (2026-07-10–13). Phases 1–2: `core/scene.py`
(autopsy + stitcher + validation, live-validated on 11 real templates) and
the engine wiring (`scene_grammar` on the session, `build_chat_input`
through the stitcher, `gen_seat` through generate/stream/fork/regen/
readouts, loom seat param, WS `generate_seat`, base-model branch →
`render_scene_raw`). Phase 3 + the full UI slice: **cast registry**
(`LoomTree.cast` label → `CastMember(recipe, notes)`; weakest-tier
composition in `_apply_cast_defaults` — explicit kwargs, `steering: ""`
clears, regen overrides all win; rides tree save + `op="cast"` frames +
REST CRUD), **transcript v2** (`speaker:`/`thinking:` per turn + `cast:`
block, v1 shim, seat-general import), **committed thinking**
(`LoomNode.thinking_text` end-to-end with commit-time capability gate),
**per-seat stop segments** (non-assistant seats add the seat close as a
stop string when it differs from the assistant's), **freed commit seating**
under scene mode, and the webui. The original two-button generation-seat
control, selected-node prefill mode, and later swap-toggle/modifier pairing were
superseded by a visible two-part role plan over the native `submit` contract:
explicit `authored_seat` plus optional `generated_seat`, named `human` / `model`
internally but displayed through their current cast labels. The model-role picker
accepts `none` for append-only; there is no modifier shortcut. `+ thinking`, the
`cast…` manager, and loom `swap seat ⇄ branch` remain. The effective cast
is auto-derived from structural and observed turn labels; the stored map
contains configuration only. On visible web surfaces its two structural entries
take their standard names from the loaded chat template (`user` / `model` for
Gemma), leaving `human` / `model` as internal seat keys only. Session info carries
`scene_mode`/`thinking_input_supported`/`strips_history_thinking`.
Live-verified on gemma-3-4b end-to-end (cast fill + explicit clear +
user-seat continue + seat-swap + thinking 400 + all UI affordances).
Result affordances (tokens, fork, regen, comparison, export) are capability-
gated by their artifacts rather than `role == "assistant"`. Remaining:
per-turn `Recipe.overlay` UI for member-level per-turn overrides, phase 4
research (seat-swap probe diffs), TUI parity.
Companion to `docs/plans/sae-pillar.md` in spirit.*

## Vision (a9)

Roles become fully dynamic: "deer", "pirate", "human", "model" are all
swappable labels, and **both the human and the model can write for any role**.
The chat-app framing (blue human, green machine, two fixed seats) dissolves
into a loom-native one: a transcript is a *script*, turns are lines spoken by
cast members, and who typed a line is provenance, not identity.

Two ambition-setting requirements (2026-07-10):

- **Arbitrary seat sequences.** Turns may occupy seats in any order —
  a/a/u/a/u/u/a/u — not just the alternation chat templates enforce.
- **Seat-swap forks.** A conversation u/a/u/a can be forked into u/a/a/u:
  same text, different seat assignment. (Research crown jewel: a controlled
  experiment on the assistant seat's post-trained identity prior — identical
  text, one seat bit flipped, diff the persona/emotion probe trajectories.)

## The cast model

A turn is `(seat, label)` plus optional generation artifacts:

- **seat** — the *structural* Saklas role, named **human** / **model** (plus
  system as stage direction). At renderer and compatibility-protocol
  boundaries these lower to `user` / `assistant`, because those are what
  model templates know how to emit.
- **label** — arbitrary slug, what the reader (and the model) sees ("deer",
  "pirate", "user").
- **generation artifacts** — recipe, raw token ids, scored token rows, and
  logprobs. These expose only the operations they support; no separate
  provenance field participates in identity or presentation.

`LoomNode` already carries `(role=seat, role_label=label)`; the engine change
is in *rendering* and in decoupling "generated" from `role == "assistant"`.

## The scene renderer (template autopsy + stitcher)

Arbitrary seat sequences force **construction** over substitution: gemma and
mistral templates hard-raise on non-alternating roles, so `apply_chat_template`
cannot be the renderer and there is nothing to splice into. But hand-maintaining
a per-family turn grammar drifts against live templates. The resolution:

**Template autopsy** — render small sentinel-content probe conversations
through the *real* chat template once per model, locate the sentinels, and
mechanically extract the segments: prelude (BOS + default-system handling),
per-seat turn wrappers (open/label-site/close), the system block shape (real
turn vs gemma-style fold into the first turn), and the generation-prompt
appendix (including thinking-scaffold variants).

**Stitcher** — a scene renders as pure concatenation: for each turn,
`open(seat, label) + [thinking] + content + close(seat)`, plus a trailing
generation header for whichever seat speaks next. Labels are placed in headers
we construct — no occurrence-matching, so the `_splice_occurrences` collision
class (a label equal to the other seat's standard label corrupting the splice)
is structurally impossible on this path.

**Round-trip validation** — stitch a canonical alternating conversation and
byte-compare against the template's own render. Pass → the family gets scene
mode; fail → raw-marker fallback + a drift warning (same philosophy as
`RoleTemplateDriftError`). The support matrix is a runtime verdict, not a
maintained table.

**The stitcher owns all rendering** (a9 ruling). Extraction renders through
it too: the round-trip check guarantees bit-identity with `apply_chat_template`
on standard alternating conversations — exactly what extraction renders — so
no fitted manifold's baseline shifts. On validation-fail families extraction
falls back to the template directly (it never needs non-alternating
sequences). The validator is load-bearing, not advisory.

Both original gaps collapse into this one artifact: "model speaks the user
seat" = choose the trailing gen header's seat; "N-party cast" = labels are
free per turn.

## Conventions (ruled 2026-07-10)

1. **System content when turn 1 isn't user-seat.** Families with a real
   system turn render it normally; fold families (gemma) prepend into the
   first turn *whatever its seat*. Deterministic, documented, invented.
2. **Per-seat stop tokens.** The autopsy's per-seat close segments supply
   them (gpt-oss ends user turns `<|end|>` vs assistant `<|return|>`;
   gemma/qwen/llama share terminators). Generating into a seat stops on that
   seat's terminator. Asymmetric but correct.
3. **Thinking is a per-turn optional input on any seat.** The human can
   commit their own thinking block; rendered via the family think delimiters
   when they exist, error otherwise. History policy: **follow the family
   template's convention uniformly** — if the template strips thinking from
   prior turns, the stitcher strips it every turn, committed or generated
   alike (no provenance split). The composer warns before submit that a
   thinking block lasts one turn on strip families (phase-2 UI).
4. **One render authority.** The stitcher owns all rendering; the splice path
   (`apply_with_role` / `apply_with_per_turn_roles`) retires once wired.

## Fork-and-swap

`LoomTree.branch` already accepts `role=` — a seat-swap fork is a branch
whose node copies keep text and flip `role`. Downstream invalidation follows
the existing `edit` contract (cached `raw_token_ids` were produced under the
old header; the swapped branch re-renders at next generation). Missing pieces
are render support (the stitcher) and UI.

## Phasing

1. **Scene renderer** *(in progress)* — `core/scene.py`: autopsy + stitcher +
   round-trip validation + raw-marker fallback (mistral, base models).
   Arbitrary `(seat, label, thinking)` sequences, per-seat trailing gen
   headers, per-seat stop segments.
2. **Seat plumbing** — `generate_seat` on the WS generate frame (commit
   already covers both seats via `commit_role`); the consumer sweep
   decoupling "generated" from `role == "assistant"` (`regenerate` step
   logic, ppl badge, `joint_logprobs`, `jlens_token_readout`); seat-swap
   branch in the loom UI; transcript schema v2 (`speaker:` + `cast:` block,
   v1 import shim); wire `_prepare_input` through the stitcher.
3. **Cast registry + recipes** — `cast: dict[slug, CastMember]` on
   `LoomTree` (serialized in tree save + transcript); composer picker over
   cast members; per-speaker steering recipe composed at generation
   (role-level default, per-turn `Recipe.overlay` override; label agreement
   with `_active_role` warns on mismatch, `RoleBaselineMismatchWarning`
   pattern). A cast member is `(label, recipe)` — the cast manager is a
   steering surface, not a chat feature.
4. **Research** — seat-swap probe diffs, per-speaker steered scenes,
   `nearest_node_role` interplay, seat effects through the J-lens.

## Deferred / open

- Mid-scene stage directions (system turns at arbitrary positions) — v1
  allows leading system only; fold families have no mid-scene shape.
- Thinking *generation* into non-assistant seats (scaffold prefill) — v1
  renders committed thinking only.
- OpenAI-compat export of non-alternating paths (`/v1/chat/completions`
  tolerates arbitrary role sequences; the native API is the home surface).
- Loom sibling-sort/diff/filter assumptions about the two-seat rhythm —
  audited in phase 2.
