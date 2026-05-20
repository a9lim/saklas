# Webui UI/UX overhaul

Plan for a full audit and redesign of the `saklas` webui — the Svelte
workbench served by `saklas serve` and mounted from `saklas/web/`.

The brief: judge the interface against what the program *ought* to let a
person do, not against what it currently does. Two outcomes drive the
design — a steering selector a first-time user can operate without
reading source, and a visual language that reads as one piece with the
a9l.im portfolio.

This document is the audit and the plan. No code lands from it directly;
the roadmap at the end orders the implementation passes.

Decisions taken before drafting (session 2026-05-17):

- Steering selector becomes a **categorized menu with custom extraction
  inline** — browse concepts by category, custom pos/neg extraction is a
  section in the same surface rather than a separate drawer.
- Theme stays **dark-only**, tightened to match the site's dark tokens
  exactly. No light mode — a research cockpit wants one well-tuned dark
  surface, not a toggle.
- Advanced per-strip controls (trigger, variant, projection, ablate)
  **stay visible**. The fix is plain-language labels and real tooltips,
  not progressive disclosure — the audience is a power user who should
  see the whole control surface at once.

---

## The principle: ought, not is

The webui today is built for someone who already knows saklas. It
assumes you know that steering is the point of the app, that concepts
are bipolar axes, that `α` is a signed coefficient, that `B/Bf/Af` are
trigger phases, that `~`/`|` are projection operators. Every one of
those is learnable, but the interface teaches none of them — it presents
the controls and waits.

A research tool can be dense. It cannot be opaque. The difference is
whether the density is *legible*: a dense interface where every control
explains itself is a cockpit; one where it doesn't is a wall. The
overhaul keeps the density and removes the opacity.

The single sharpest instance is adding a steering vector, so that leads
the audit.

---

## Audit — where the experience breaks down

### 1. Adding steering is the worst path in the app

`SteeringRack.svelte` offers two ways to add a vector and both are bad
for a newcomer.

The **quick field** is a text input with a `<datalist>`. You type a
concept name and press Enter or click "apply vector". If you type
nothing, it silently applies `vectorOptions[0]` — whatever sorts first
alphabetically — or opens the picker. If you type a name that doesn't
resolve, `apiVectors.extract` errors and you get a raw `status: detail`
string. There is no indication of what you *can* type. The datalist only
helps if you already know a name to start typing. This is the "type a
verb and pray" path.

The **browse button** opens `VectorPickerDrawer`, which renders
`_SearchableConceptList` — a flat, alphabetical, uncategorized list of
installed packs. It is searchable, which helps if you know what you're
looking for, and useless for discovery. The 26 bundled concepts span
seven distinct domains (affect, epistemic, alignment, register, social
stance, cultural, identity) and the list flattens all of them into one
scroll. Nothing tells you `angry.calm` is an emotion axis and
`formal.casual` is a register axis.

What it ought to be: one obvious entry point — a primary **+ add
steering** button — opening a surface that *presents the catalog*
organized the way the catalog is actually organized, with custom
extraction as a section of that same surface for when the catalog
doesn't have what you want.

### 2. The bipolar axis is invisible

24 of the 26 bundled concepts are bipolar. `angry.calm` is one axis with
two poles; `α > 0` steers toward angry, `α < 0` toward calm. The
`pack.json` description says so in plain words — *"Bipolar axis: angry
(+) vs calm (-)"* — and the webui shows none of it.

`VectorStrip.svelte` renders the concept as a bare name (`angry.calm`)
next to a `−1…+1` slider. A newcomer dragging that slider has no idea
that left is calm and right is angry. The sign-to-pole mapping, the most
important fact about a steering vector, is something you can only learn
by reading `pack.json` or the AGENTS.md.

The bipolar structure should be the *visual frame* of the strip: pole
labels flanking the slider, so the control reads as
`calm ◄──●──► angry` and dragging means something. The data to do this
is already on disk and already on the wire (see §"Category data").

### 3. Cryptic controls with no plain-language anchor

Per the session decision, these controls stay on the strip. They need
to stop being cryptic.

- **Trigger pill** shows `B`, `Bf`, `Af`, `Th`, `Rs`, `Pr`, `Gn`. The
  tooltip explains, but the resting state is a code. There is room on
  the strip for the word — `both`, `before`, `after`, `thinking`,
  `response`. Show the word; keep click-to-cycle.
- **Variant chip** shows `raw` / `sae`. Fine as a label, wrong as an
  editor — clicking it opens a `window.prompt()` (see §4).
- **Projection tag** appears only once set, as `~ honest`. A newcomer
  never sees `~`/`|` explained anywhere near where they'd use it.
- **`α`** is correct notation for this audience, but the strip should
  call it *strength* in tooltips and `aria-label`s so a first read
  lands.

None of this is hiding controls. It is making the visible controls say
what they are.

### 4. `window.prompt` / `window.confirm` break the surface

`VectorStrip.pickVariant` opens a native `window.prompt`. The clipboard
fallbacks in `SteeringRack` and `VectorStrip` do too. A native prompt is
unstyled, unthemed, modal in the browser's way rather than the app's,
and visually nothing like the rest of saklas. Every one of these should
be an in-app control — a small dropdown for the variant, a toast for the
copy fallback. The projection picker already made this move (inline
modal instead of prompt); the variant chip should follow.

### 5. Twenty-six drawers behind one unlabeled menu

`Topbar.svelte`'s "tools" dropdown lists ~20 drawers in one column with
three unlabeled `<hr>` dividers. The dividers imply grouping but name no
groups. The entries mix three unrelated kinds of task:

- **vector work** — extract, merge, clone, compare, packs, load
- **analysis** — correlation matrix, layer norms, activation atlas,
  experiment lab, recipe builder
- **session & model** — system prompt, model info, session/auth, model
  health, save/load conversation, help

`InspectorPanel`'s "launchpad" 2×2 grid then duplicates four of these
(experiment lab, activation atlas, apply steering, health) as buttons —
so some tools have two entry points and most have one, with no rule a
user could infer.

Ought: the tools menu gets labeled section headers matching the three
kinds. The launchpad either becomes the *single* home for the
analysis-class tools (and they leave the tools menu) or is dropped — one
rule, not two overlapping ones.

### 6. No first-run state — the app opens empty and silent

On first load the chat is empty, the steering rack says "no active
steering vectors", the probe rack says it has no probes. Nothing on
screen says steering is the purpose of the app or how to start. A
research tool's empty state is a teaching opportunity and saklas spends
it on a gray "no active steering vectors".

Ought: the steering rack's empty state is one line of plain copy —
*"Steering shapes how the model responds. Add a concept to begin."* —
above the primary **+ add steering** button. The probe rack's empty
state names the distinction the two racks otherwise blur (see §7).

### 7. Steering rack and probe rack look identical and aren't

The two racks were deliberately built to "read as one visual system" —
same `●/○` glyphs, same row metrics, same strip shape. For a user who
knows saklas that's coherence. For a newcomer it erases the most
important distinction in the app: **steering changes the output, probes
observe it.** One is an intervention, one is a measurement.

Ought: keep the shared grammar, add one line of subtitle copy to each
rack header that states what it does. "Steering — shape the response."
"Probes — watch concepts activate." Two sentences buy the whole mental
model.

### 8. The drawer is one size for everything

`App.svelte` renders every drawer at `width: min(980px, 78%)`. The
system-prompt drawer is a single textarea. The activation atlas is a
dense heatmap. They get the same 980px. Small tools in a huge panel feel
unfinished; the drawer host should size to its content class — a narrow
panel (~480px) for forms and pickers, the wide panel for analysis views.

### 9. Theme drift from the portfolio

The webui already borrowed the "volcanic glass" palette and the
Recursive font, so it is close — but not exact, and the gaps read.

- The primary accent variable is named `--accent-cyan` and holds
  `#e11107`, which is red. `tokens.css` even comments the misnomer. Any
  future reader of the CSS is misled. Rename to `--accent`, matching the
  site's token name.
- The webui dark surfaces (`--bg`, `--bg-alt`, `--bg-elev`) happen to
  equal the site's dark `canvas`/`panelSolid`/`elevated`, but they're
  hand-copied hex, not derived from one source. They should be reconciled
  against `shared-tokens.js` so a future site palette change has an
  obvious propagation path.
- The site has motion tokens — `--ease-out: cubic-bezier(0.23,1,0.32,1)`
  and friends — and uses them everywhere. The webui has near-zero
  transitions and the few it has are `0.1s ease`. Buttons on the site
  lift `translateY(-2px)` on hover; webui buttons just swap background.
- `:focus-visible` on the site is an `--accent-glow` ring with
  `outline-offset`. The webui uses a hard `2px solid` red outline.
- The site has a polished segmented control (mode toggle with an
  animated indicator). The webui uses bare radio buttons for the
  SamplingStrip session/one-shot mode — a place the segmented control
  would slot in directly.

None of this is a rebuild. It is a reconciliation pass plus a motion
layer.

---

## Design principles for the overhaul

1. **Every control states what it is at rest.** Tooltips are the second
   line of defense, not the first. A code (`Bf`, `~`) is acceptable only
   when the word genuinely doesn't fit.
2. **The catalog is presented the way it's organized.** Concepts have
   categories and poles; the UI shows categories and poles.
3. **No native browser dialogs.** `prompt`, `confirm`, `alert` never
   appear. Every input is an in-app control in the saklas visual
   language.
4. **One entry point per task.** If a tool has a launchpad button it
   does not also have a tools-menu entry, or vice versa — never both.
5. **Empty states teach.** The first thing a newcomer sees explains the
   panel's purpose and offers the primary action.
6. **Density stays; opacity goes.** This is not a simplification. It is
   making an already-dense interface legible.
7. **The portfolio is the style guide.** Tokens, motion, and component
   shapes come from `shared-tokens.js` / `shared-base.css`, dark mode
   only.

---

## The steering selector — full redesign

This is the centerpiece. Three pieces change: the entry point, the
picker surface, and the rack strip.

### Entry point

`SteeringRack`'s footer loses the quick text field and the two-button
quick/browse split. In its place, one primary button:

```
┌────────────────────────────────────────┐
│ STEERING — shape the response           │
│                                         │
│  (strips, or empty state)               │
│                                         │
│  ┌────────────────────────────────────┐ │
│  │          + add steering            │ │  ← primary
│  └────────────────────────────────────┘ │
│                                         │
│  active steering           ✎ edit       │
│  0.30 angry.calm + 0.20 honest          │
└────────────────────────────────────────┘
```

Empty state replaces the strip list when the rack is empty:

```
  Steering shapes how the model responds.
  Add a concept to begin.

  ┌────────────────────────────────────┐
  │          + add steering            │
  └────────────────────────────────────┘
```

The quick text field is removed entirely — it was a worse version of the
picker's search box. Power users keep the expression paste-edit (the
`✎ edit` affordance), which is the real expert path and round-trips the
whole grammar.

### The picker

`+ add steering` opens a redrawn `VectorPickerDrawer` — a content-sized
panel (~480px, not the 980px default), structured as a categorized menu:

```
add steering ──────────────────────────── ✕

[ search concepts…                        ]

▾ AFFECT
   calm        ◄────●────►  angry      [add]
   sad         ◄──────●──►  happy      [add]
   unflinching ◄───●─────►  fearful    [add]

▾ EPISTEMIC
   uncertain   ◄────●────►  confident  [add]
   deceptive   ◄────●────►  honest     [add]
   grounded    ◄────●────►  hallucinat.[add]
   disinterest ◄────●────►  curious    [add]

▸ ALIGNMENT      ▸ REGISTER
▸ SOCIAL STANCE  ▸ CULTURAL
▸ IDENTITY

──────────────────────────────────────────
▾ CUSTOM EXTRACTION
   name      [ my_concept            ]
   positive  [ contrastive pos text  ]
   negative  [ contrastive neg text  ]
   method    (•) difference-of-means
             ( ) contrastive PCA
   SAE       [ release (optional)    ]
   ☑ centered DLS layer selection
   [ extract → add to rack ]
──────────────────────────────────────────
   load from disk…
```

Behavior:

- **Categories** are collapsible sections. Affect and Epistemic expand
  by default; the rest collapse to a chip row so the whole catalog fits
  without a long scroll. Section state is not persisted — it's cheap to
  re-open.
- **Each concept row** shows the bipolar axis: negative pole on the
  left, positive pole on the right, a mini non-interactive slider
  preview at the recommended-alpha resting position. `[add]` drops it
  on the rack at `recommended_alpha` and closes the drawer (matching
  today's pick-and-add).
- **Monopolar concepts** (`agentic`, `manipulative`) render with a
  single pole label and a `0…+1` preview — no left pole.
- **Search** filters across categories and keeps the section grouping;
  an empty search shows the full menu. A query with no catalog match
  pre-fills the custom-extraction `name` field instead of today's
  "extract on the fly" button — the no-match case flows naturally into
  the custom section.
- **Custom extraction** is the existing `ExtractDrawer` form, inlined as
  the last section. It calls the same `apiExtractStream` with an SSE
  progress log. `ExtractDrawer` as a standalone drawer is retired — its
  only remaining caller was the picker's footer.
- **Load from disk** stays as a footer link into `LoadDrawer` (a genuine
  separate task — a file path, not a concept).

`ProbePickerDrawer` gets the same categorized treatment from the same
shared component — a probe is the same catalog, picked to observe rather
than to steer. The shared list component (`_SearchableConceptList`)
grows category grouping and pole rendering and both drawers consume it.

### Category data — already on the wire

No protocol change is needed for grouping or poles. Confirmed against
the bundled `pack.json` files:

```json
{
  "name": "angry.calm",
  "description": "Bipolar axis: angry (+) vs calm (-). ...",
  "tags": ["affect"],
  "recommended_alpha": 0.5,
  ...
}
```

- **Category** = the category-valued tag. The seven categories are a
  fixed set (`affect`, `epistemic`, `alignment`, `register`,
  `social_stance`, `cultural`, `identity`); the picker matches `tags`
  against that set and falls back to an "other" section for packs whose
  tags miss it (user-authored / HF packs).
- **Poles** = split the canonical name on `.` (`BIPOLAR_SEP`).
  `angry.calm` → positive `angry`, negative `calm`. No dot → monopolar.
- **Resting α** = `recommended_alpha`.

`LocalPackInfo` already carries `tags` and `description` and has a
`[key: string]: unknown` passthrough. The one thing to verify: that the
server's pack-list endpoint actually surfaces `recommended_alpha` in the
row (it's in `pack.json`; confirm it reaches the response — if not, it's
a one-field add to the serializer, not a schema change). Until then the
picker can default the resting α to `0.5`.

### The rack strip, restyled

`VectorStrip` reframes the bipolar axis as the layout. Today:

```
●  angry.calm   [─────●─────]  +0.30   B   raw   ⋮  ✕
```

Ought:

```
●   calm ◄──────●──► angry    +0.30  [both][raw] ⋮ ✕
```

- Pole labels flank the slider. The concept name still lives in the
  strip (tooltip / `aria-label`, and the expression block shows the
  canonical form) but the *visible frame* is the axis.
- Negative-α drag moves toward the left pole, positive toward the
  right — now legible because the poles are labeled.
- The α readout keeps its sign-colored green/red and `tabular-nums`.
- The trigger pill shows the **word** (`both`, `after`, …), not the
  code. Click still cycles.
- The variant chip stays a chip; clicking it opens a small in-app
  dropdown (`raw` / `sae` / `sae-<release>` from the concept's available
  variants), never `window.prompt`.
- Monopolar strips show one pole label and a `0…+1` slider.
- The `⋮` menu (projection, ablate, duplicate, copy) is unchanged in
  function; the projection picker's inline modal stays.

### Keeping the expression

The canonical expression block stays — it's the receipt and the
expert-input path. It's reframed as clearly secondary: a labeled,
copyable line with an `✎ edit` affordance for paste-editing the whole
grammar. The rack strips are the primary editing surface; the expression
is for reading what you built and for pasting a recipe from elsewhere.

---

## Theme port — dark-only, exact site parity

A reconciliation pass, not a rebuild. Three pieces.

### Token reconciliation

`tokens.css` is rewritten to derive from the site's dark palette as the
single source of intent. Concretely:

- Rename `--accent-cyan` → `--accent` everywhere (it's the magma red;
  the name has misled long enough). One rename across the webui;
  `--accent-blue`, `--accent-green`, etc. stay.
- Verify each surface/text/border token against `shared-tokens.js`'s
  `dark` block and `extended` palette. They are believed equal today;
  this pins them and leaves a comment pointing at the source so a future
  site palette bump has a known propagation path.
- Add the site's motion tokens: `--ease-out`, `--ease-in-out`,
  `--ease-smooth`, plus a couple of standard durations.
- Add `--accent-glow` (`rgba(225,17,7,0.18)`) and `--accent-subtle`
  (already present) so focus rings and hover backgrounds match the site.

### Component polish

- **Focus** — swap the hard `2px solid` red outline for the site's
  `--accent-glow` ring with `outline-offset`. Global, in `global.css`.
- **Buttons** — adopt the site's hover idiom where it fits: subtle
  `translateY(-1px)` lift and a background/border transition on the
  `--ease-out` curve. Topbar actions, drawer buttons, rack actions.
- **Segmented control** — build one small shared component matching the
  site's mode toggle (animated indicator bar). First consumer: the
  SamplingStrip session/one-shot radios. Candidate second consumer: the
  ProbeRack sort mode.
- **Topbar** — the hardcoded `linear-gradient(180deg, rgba(17,24,35…))`
  becomes token-driven glass matching the site's `.glass` nav (panel
  token + subtle blur).
- **Drawer sizing** — `App.svelte` gives each drawer a size class
  (`narrow` ~480px for forms/pickers, `wide` ~980px for analysis views)
  instead of one width for all.

### Motion

Apply the easing tokens to the transitions that already exist (drawer
slide-in, hover states) and add them where the site would have them and
the webui has nothing — strip add/remove, section expand/collapse in the
new picker. Respect `prefers-reduced-motion` with the same global
override the site uses.

---

## Panel & drawer notes (ought-to)

Quick pass over the rest of the surface. Items not called out above are
structurally fine and only need the theme/motion pass.

| Surface | Ought-to note |
|---|---|
| `Topbar` | Tools menu gets labeled sections (vector / analysis / session). Glass background token-driven. |
| `InspectorPanel` | Resolve the launchpad/tools-menu duplication — launchpad becomes the single home for analysis-class tools, or is dropped. |
| `SteeringRack` | Redesigned per §"The steering selector". Header gets the one-line subtitle. |
| `VectorStrip` | Redesigned per §"The rack strip". |
| `ProbeRack` | Header gets the "watch concepts activate" subtitle. Sort control → segmented. |
| `ProbeStrip` | Structurally fine. Theme pass only. |
| `SamplingStrip` | Mode radios → segmented control. `K`/`max` numeric inputs need labels that read without the tooltip. |
| `Chat` | The highlight controls are duplicated verbatim from `ProbeRack` — pick one home for them (Chat is the better one, since highlighting is about reading the transcript) and drop the duplicate. |
| `WorkspaceRail` | Confirm it earns its 64px column — if it's one or two icons, fold into the topbar. |
| `StatusFooter` | Theme pass only. |
| Loom panels | Out of scope here — `loom.md` owns them. Theme pass only. |
| `VectorPickerDrawer` | Redesigned per §"The picker". |
| `ProbePickerDrawer` | Same categorized treatment via the shared component. |
| `ExtractDrawer` | Retired as a standalone drawer; its form inlines into the picker's custom section. |
| `MergeDrawer`, `CloneDrawer` | Stay; narrow drawer size class; theme pass. Their live-validation is good — keep it. |
| `LoadDrawer` | Stays as the "from disk" footer target. Narrow size. |
| `SystemPromptDrawer` | Narrow size class (it's one textarea). |
| `HelpDrawer` | Should be the place the grammar (`~`, `|`, `!`, `@trigger`) is taught in plain words — audit its copy against the AGENTS.md grammar. |
| `PackDrawer` | Theme pass. Its local-list pattern is the model the picker's shared component follows. |
| Analysis drawers (`Correlation`, `LayerNorms`, `ActivationAtlas`, `ExperimentLab`, `RecipeBuilder`, `Compare`, `NodeCompare`, `TokenDrilldown`) | Wide size class. Theme + motion pass. No structural change. |
| `AdvancedSamplingDrawer` | Theme pass. Confirm it isn't re-exposing controls the SamplingStrip already owns. |
| `HealthDrawer`, `ModelInfoDrawer`, `SessionAdminDrawer` | Theme pass. |
| `Save/LoadConversationDrawer`, `TranscriptDrawer` | Theme pass. |

---

## Roadmap

Ordered by ship value and by dependency. Each phase is its own PR.

**Phase 0 — token reconciliation.** Rewrite `tokens.css` against
`shared-tokens.js`; rename `--accent-cyan` → `--accent`; add motion and
glow tokens; swap the global focus ring. Pure CSS, no behavior change,
no new components. Safe to land first and everything after builds on it.
Touches: `lib/style/tokens.css`, `lib/style/global.css`, and a
mechanical rename across components.

**Phase 1 — the steering selector.** The headline. New categorized
picker (shared list component grows category + pole rendering),
restyled `VectorStrip` with pole-framed slider, custom extraction
inlined, in-app variant dropdown replacing `window.prompt`, `+ add
steering` entry point, retire `ExtractDrawer` as standalone. Touches:
`SteeringRack.svelte`, `VectorStrip.svelte`, `VectorPickerDrawer.svelte`,
`ProbePickerDrawer.svelte`, `_SearchableConceptList.svelte`,
`drawers/index.ts`, `App.svelte` (drawer wiring + size class).

**Phase 2 — empty states and rack legibility.** Steering/probe rack
empty states and header subtitles. The teaching copy. Small, isolated,
high ratio of newcomer-clarity to effort.

**Phase 3 — navigation cleanup.** Tools menu sectioning; resolve the
launchpad/tools duplication; de-duplicate the Chat/ProbeRack highlight
controls. Touches: `Topbar.svelte`, `InspectorPanel.svelte`,
`Chat.svelte`, `ProbeRack.svelte`.

**Phase 4 — drawer sizing + native-dialog purge.** Content-driven
drawer size classes; sweep for any remaining `window.prompt/confirm/
alert` and the clipboard fallbacks; replace with in-app controls and
toasts. Touches: `App.svelte`, `SteeringRack.svelte`, `VectorStrip.svelte`.

**Phase 5 — component polish sweep.** Segmented control component and
its consumers; button hover lift; topbar glass; motion on the
transitions that exist and the new ones; `prefers-reduced-motion`
override. Theme pass across every remaining drawer.

Phases 0 and 1 are the priority — they deliver the brief. 2–5 are
genuine improvements that can land at any cadence after.

---

## Protocol / server notes

The redesign is almost entirely client-side. One item to verify, not a
blocker:

- **`recommended_alpha` on the pack-list response.** It's in every
  bundled `pack.json`. Confirm `saklas serve`'s pack-list endpoint
  includes it in each `LocalPackInfo` row. If it doesn't, add it to the
  serializer — a one-field addition, not a schema change, and
  `LocalPackInfo`'s passthrough already accepts it. Until confirmed the
  picker defaults the resting slider to `0.5`.

Everything else — category from `tags`, poles from the canonical name —
is already on the wire.

---

## Open questions

- **Category source robustness.** Matching `tags` against the fixed
  seven-category set works for bundled packs. User-authored and
  HF-pulled packs may tag differently or not at all — they land in an
  "other" section. Acceptable for v1; revisit if user packs become
  common enough that "other" gets crowded.
- **WorkspaceRail's column.** Whether the 64px rail earns its width is a
  judgement call deferred to Phase 3 — it depends on how many workspaces
  the rail actually switches between, which the exploration didn't pin
  down. Decide with the panel open.
- **Picker as drawer vs modal.** The plan keeps it a (narrow) drawer for
  consistency with the rest of the tool surface. A centered modal would
  also work and reads as more "menu-like". Low stakes; drawer chosen for
  uniformity, revisit only if it feels wrong in implementation.
