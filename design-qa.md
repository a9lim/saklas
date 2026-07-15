# Design QA

- Source visual truth:
  - `/var/folders/qd/hy_hzb7975122x6czsxyl1pm0000gn/T/TemporaryItems/NSIRD_screencaptureui_Ndvvlj/Screenshot 2026-07-12 at 20.36.31.png`
  - `/var/folders/qd/hy_hzb7975122x6czsxyl1pm0000gn/T/TemporaryItems/NSIRD_screencaptureui_jDYFN5/Screenshot 2026-07-12 at 20.36.47.png`
  - `/var/folders/qd/hy_hzb7975122x6czsxyl1pm0000gn/T/TemporaryItems/NSIRD_screencaptureui_F0LNfa/Screenshot 2026-07-12 at 20.37.28.png`
  - User direction: remove the extra J-LENS prepared-source row, make the SAE and J-LENS source/custom sections isomorphic, and expose the J-LENS layer selection clearly.
- Post-fix implementation: inspected live at `http://localhost:8000` from `saklas serve google/gemma-3-4b-it`.
- Viewport: 1280 × 720, with the inspector column at its normal desktop width.
- State: J-LENS Neuronpedia source prepared and active; SAE provider available with no active source; both custom workflows idle.
- Primary interactions tested: tab switching, source-control rendering, editable J-LENS layer binding (`workspace` → `all` → `workspace`), and source/custom-row geometry.
- Console errors checked: no warnings or errors.

## Full-view comparison evidence

The prepared-artifact picker and provider picker are now one control. A prepared selection uses or reports the active source; an unprepared provider selection fetches it. Source switching remains available without the second J-LENS row.

Both pillars now render the same two groups in the same order: `PREPARED / PROVIDER`, then `CUSTOM`. The custom row uses one shared labelled-field grammar. SAE exposes `NAME`, `TOKENS`, and `LAYER`; J-LENS exposes `PROMPTS` and `LAYERS`. The J-LENS layer field remains free-form so `workspace`, `all`, and comma-separated layer ids all survive.

## Focused region comparison evidence

Browser geometry at the live desktop viewport matched exactly across tabs:

| Group | J-LENS | SAE |
|---|---:|---:|
| Prepared/provider top | 124 px | 124 px |
| Prepared/provider row | 26 px | 26 px |
| Custom top | 185 px | 185 px |
| Custom row | 39 px | 39 px |

The local action buttons share the same 26 px control baseline. Visible labels sit above every custom input, so J-LENS no longer relies on the ambiguous bare value `workspace` to communicate its layer control.

## Findings

No remaining SAE/J-LENS source-layout parity defects were observed at the tested viewport.

## Required fidelity surfaces

- Fonts and typography: source-group and field labels share one tracked UI register; identifiers and values remain mono.
- Spacing and layout rhythm: both source groups and both custom rows are emitted by the shared `InstrumentSourceSection` geometry.
- Colors and visual tokens: family accents intentionally remain blue and gold; structure and sizing are otherwise identical.
- Copy and content: the shared action changes among `fetch provider`, `use source`, and `active` according to lifecycle state. Local actions remain the genuine content difference: `fit local` versus `train local`.
- Accessibility: both selectors expose `Artifact source`; the J-LENS fields expose `J-lens corpus prompts` and `J-lens source layers`.

## Comparison history

- Iteration 1: shared section ordering landed, but the panels still supplied different provider widgets and J-LENS retained a static local-fit description.
- Iteration 2: provider picker/action moved into the shared component; SAE native datalist was removed; J-LENS prompt/layer inputs were wired to the existing route.
- Iteration 3: live comparison exposed a redundant prepared-source row and unlabeled custom controls. Prepared and provider choices were unified, custom fields received shared visible labels, and the two layouts were measured live.
- Validation: `svelte-check` passed with zero errors/warnings; the production bundle built; the rebuilt dashboard rendered cleanly against the requested model server.

## Implementation checklist

- [x] Keep one source selector instead of separate prepared/provider rows.
- [x] Preserve prepared-source switching and provider fetch behavior.
- [x] Give both pillars the same prepared/provider and custom group structure.
- [x] Give both custom workflows the same labelled-field and action alignment.
- [x] Expose J-LENS prompts and layers as visible fit controls.
- [x] Keep `workspace`, `all`, and comma-separated J-LENS layer ids editable.
- [x] Rebuild the committed production bundle.
- [x] Capture and compare both rendered panels in matching viewport geometry.

## Follow-up polish

None identified in the requested source/custom region.

final result: pass
