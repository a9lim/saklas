# Token drilldown design QA

## Scope

- Surface: the four populated token-drilldown views at `http://localhost:8000/`.
- Original redesign state: `google/gemma-3-4b-it`, generated token `" Rayleigh"`, response token 10 of 93, 1280 x 720 viewport.
- 5.1.1 harmonization state: the same model, generated token `" slate"`, response token 8 of 208, 1280 x 720 viewport.
- Visual source of truth: the live main subspace, manifold, SAE, and J-lens panels, plus the pre-redesign drilldown captures.
- Implementation truth: the production web build served by `saklas serve` after the redesign.

## Evidence

- Main-panel references: `/tmp/saklas-token-drilldown-audit-5U2AcT/01-main-subspace.png` through `04-main-lens.png`.
- Before states: `/tmp/saklas-token-drilldown-audit-5U2AcT/05-drilldown-lens-before.png` through `08-drilldown-sae-before.png`.
- Final populated states: `/tmp/saklas-token-drilldown-audit-5U2AcT/11-drilldown-geometry-after.png`, `12-drilldown-logits-after.png`, `13-drilldown-sae-after.png`, and `18-final-build.png`.
- Full retained J-lens matrix: `/tmp/saklas-token-drilldown-audit-5U2AcT/14-drilldown-lens-matrix-after.png`.
- Direct comparison input: `/tmp/saklas-token-drilldown-audit-5U2AcT/15-before-after-contact-sheet.png`.
- Focused matrix comparison: `/tmp/saklas-token-drilldown-audit-5U2AcT/17-lens-matrix-comparison.png`.

## Comparison findings and iterations

1. The previous views had four unrelated hierarchies. The harmonized views now use the same lean sequence: token and generation recipe, evidence-count tabs, compact provenance, rack-style evidence, and a method note. The redundant per-tab hero summaries and metric grids are removed.
2. Geometry now distinguishes flat subspaces from curved manifolds through the same marker and accent grammar as the main rack, while retaining coordinates, fraction, residual, membership, nearest nodes, soft assignments, depth, and layer strips.
3. The logits view retains full evidence cards and alternative branching without repeating the selected-token statistics in a separate summary surface.
4. SAE matches the main gold feature-card system and exposes normalized strength, raw activation, maxActApprox metadata, labels, source, and resident layer directly in its compact provenance row and evidence cards.
5. J-lens now leads with aggregate workspace cards and then preserves the complete layer-by-vocabulary matrix. Every retained matrix cell shows token text and probability, and the generated token is outlined.
6. The 5.1.1 follow-up removes the redundant model / loom-node / perplexity / finish facts while keeping the generation recipe visible across every tab.
7. Geometry probe readings now use the same two-column desktop grid as logits, SAE, J-lens, and pinned readings, collapsing to one column below 820 px.
8. Transient hover tooltips remain visible in some screenshots because the browser pointer is left over the control used to switch state; they are not persistent layout elements.

## Interaction and implementation verification

- All four tabs populated and switched successfully in the live app.
- Layout inspection found zero summary boxes and zero legacy context-fact boxes; the generation recipe remained present.
- Geometry rendered all 17 probe cards in two computed 466 px columns with no horizontal drawer overflow at 1280 x 720.
- Previous/next token navigation updated token id, raw index, token position, and instrument evidence, then returned to the audit token.
- Browser console: zero warnings and zero errors after the final production reload.
- `npm run check`: zero errors and zero warnings; theme validation passed.
- `npm run build`: production build passed.
- `pytest -q tests/test_instrument_routes.py`: 41 passed; one upstream Starlette/httpx deprecation warning.
- `git diff --check`: passed.

## Final result

passed
