# Design QA

- Source visual truth:
  - `/var/folders/qd/hy_hzb7975122x6czsxyl1pm0000gn/T/TemporaryItems/NSIRD_screencaptureui_Ndvvlj/Screenshot 2026-07-12 at 20.36.31.png`
  - `/var/folders/qd/hy_hzb7975122x6czsxyl1pm0000gn/T/TemporaryItems/NSIRD_screencaptureui_jDYFN5/Screenshot 2026-07-12 at 20.36.47.png`
  - `/var/folders/qd/hy_hzb7975122x6czsxyl1pm0000gn/T/TemporaryItems/NSIRD_screencaptureui_F0LNfa/Screenshot 2026-07-12 at 20.37.28.png`
  - User direction: one shared provider control using Saklas `Select`, identical `fetch provider` copy, and editable J-LENS prompt/layer-band fields.
- Pre-fix implementation screenshots: the three source paths above.
- Post-fix implementation screenshot: unavailable; browser discovery returned no in-app or Chrome browser.
- Viewport: focused inspector-region crops at 852 x 230, 978 x 436, and 832 x 56.
- State: J-LENS provider prepared and active; SAE provider menu open; J-LENS local fit idle.
- Primary interactions tested: Svelte select/input bindings, fit payload plumbing, and focused server lifecycle tests.
- Console errors checked: blocked with browser rendering.

## Full-view comparison evidence

The three supplied implementation crops were opened at original resolution. They expose three objective parity failures: J-LENS used a static provider well while SAE used a native datalist; provider action copy differed; and J-LENS encoded its fit defaults in a static sentence rather than editable controls.

The implementation now makes `InstrumentSourceSection.svelte` own the provider `Select` and `fetch provider` button for both pillars. J-LENS and SAE can no longer choose different provider widgets or button copy. J-LENS now renders numeric prompts and free-form layer-band inputs and submits both through the existing fit API.

A rendered post-fix capture could not be produced because neither in-app Browser nor Chrome was available. Build success is not being treated as visual proof.

## Focused region comparison evidence

The supplied images are already focused on the three mismatched regions at readable scale, so smaller crops were unnecessary. Post-fix focused comparison remains blocked by browser availability.

## Findings

- [P2] Post-change pixel and interaction comparison is blocked.
  - Location: SAE and J-LENS SOURCE sections.
  - Evidence: the pre-fix mismatch is directly visible; the shared implementation and payload plumbing validate in source/tests, but no rendered post-fix capture exists.
  - Impact: exact row height, popover placement, field-width balance, and browser focus behavior cannot be visually confirmed.
  - Fix: capture the provider rows, the open SAE provider popover, and the J-LENS local row at the same inspector width once a browser is available.

## Required fidelity surfaces

- Fonts and typography: both provider controls now use the same Saklas `Select` typography; local values remain on the existing mono input style. Rendered antialiasing remains unverified.
- Spacing and layout rhythm: provider geometry and action alignment are owned by one shared grid. J-LENS local controls reuse the same input row used by SAE. Rendered wrapping remains unverified.
- Colors and visual tokens: family accents intentionally remain blue and gold; the action variant and semantic copy are identical.
- Image quality and asset fidelity: no raster imagery, logos, or custom icons appear in this region.
- Copy and content: both provider actions say `fetch provider`; J-LENS exposes `prompts` and `layers` as real fit parameters.

## Comparison history

- Iteration 1: shared section ordering landed, but the two panels still supplied different provider widgets and J-LENS retained a static local-fit description.
- User evidence: the 20:36/20:37 captures identified the native datalist, differing action copy, and non-editable J-LENS defaults.
- Iteration 2: provider picker/action moved into the shared component; SAE native datalist removed; J-LENS prompt/layer inputs wired to the existing route.
- Validation: `svelte-check` passed with zero errors/warnings; production build passed; focused J-LENS lifecycle tests passed (4 tests).
- Post-fix visual evidence: blocked because browser discovery returned no available browser.

## Implementation checklist

- [x] Use Saklas `Select` for both provider pickers.
- [x] Own the provider picker and button in one shared component.
- [x] Standardize both actions to `fetch provider`.
- [x] Replace the static J-LENS local-fit sentence with prompt and layer-band inputs.
- [x] Submit customized J-LENS values to the existing fit route.
- [x] Rebuild the committed production bundle.
- [ ] Capture and compare both rendered panels in matching states.

## Follow-up polish

None identified from source inspection beyond the blocked rendered comparison.

final result: blocked
