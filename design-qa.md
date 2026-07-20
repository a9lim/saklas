# Token drilldown design QA

## Scope

- Surface: the four populated token-drilldown views at `http://localhost:8000/`.
- Original redesign state: `google/gemma-3-4b-it`, generated token `" Rayleigh"`, response token 10 of 93, 1280 x 720 viewport.
- 5.1.1 harmonization state: the same model, generated token `" slate"`, response token 8 of 208, 1280 x 720 viewport.
- Card-alignment follow-up: the same model, generated token `"anything"`, response token 15 of 23, 1280 x 720 viewport.
- Visual source of truth: the live main subspace, manifold, SAE, and J-lens panels, plus the pre-redesign drilldown captures.
- Implementation truth: the production web build served by `saklas serve` after the redesign.

## Evidence

- Main-panel references: `/tmp/saklas-token-drilldown-audit-5U2AcT/01-main-subspace.png` through `04-main-lens.png`.
- Before states: `/tmp/saklas-token-drilldown-audit-5U2AcT/05-drilldown-lens-before.png` through `08-drilldown-sae-before.png`.
- Final populated states: `/tmp/saklas-token-drilldown-audit-5U2AcT/11-drilldown-geometry-after.png`, `12-drilldown-logits-after.png`, `13-drilldown-sae-after.png`, and `18-final-build.png`.
- Full retained J-lens matrix: `/tmp/saklas-token-drilldown-audit-5U2AcT/14-drilldown-lens-matrix-after.png`.
- Direct comparison input: `/tmp/saklas-token-drilldown-audit-5U2AcT/15-before-after-contact-sheet.png`.
- Focused matrix comparison: `/tmp/saklas-token-drilldown-audit-5U2AcT/17-lens-matrix-comparison.png`.
- Header/chip reference comparison: `/tmp/saklas-token-drilldown-audit-5U2AcT/19-card-header-comparison.png`.
- Final aligned J-lens, SAE, and geometry states: `/tmp/saklas-token-drilldown-audit-5U2AcT/20-lens-aligned.png` through `22-geometry-aligned.png`.
- Canonical-value states for logits, J-lens, and SAE: `/tmp/saklas-token-drilldown-audit-5U2AcT/23-logits-canonical-value.png` through `25-sae-canonical-value.png`.

## Comparison findings and iterations

1. The previous views had four unrelated hierarchies. The harmonized views now use the same lean sequence: token and generation recipe, evidence-count tabs, compact provenance, rack-style evidence, and a method note. The redundant per-tab hero summaries and metric grids are removed.
2. Geometry now distinguishes flat subspaces from curved manifolds through the same marker and accent grammar as the main rack, while retaining coordinates, fraction, residual, membership, nearest nodes, soft assignments, depth, and layer strips.
3. The logits view retains full evidence cards and alternative branching without repeating the selected-token statistics in a separate summary surface.
4. SAE matches the main gold feature-card system and exposes normalized strength, raw activation, maxActApprox metadata, labels, source, and resident layer directly in its compact provenance row and evidence cards.
5. J-lens now leads with aggregate workspace cards and then preserves the complete layer-by-vocabulary matrix. Every retained matrix cell shows token text and probability, and the generated token is outlined.
6. The 5.1.1 follow-up removes the redundant model / loom-node / perplexity / finish facts while keeping the generation recipe visible across every tab.
7. Geometry probe readings use the same two-column desktop grid as logits, SAE, J-lens, and pinned readings, collapsing to one column below 820 px. The statline percentage and inline nearest-node distance are omitted; precise distances remain in the chips below.
8. The J-lens layer × vocabulary matrix expands to its full table height and follows the drawer's outer scroll instead of creating a fixed-height nested scroller.
9. Geometry, logits, SAE, J-lens aggregate, and pinned cards now share one statline component. Its 24 px leading slot, 12 px primary type, and common line-height put every rank/marker and identifier on the same grid. SAE raw metadata was replaced by the same compact supporting-fact chips used for geometry evidence, and the redundant `unit strength` fact was removed.
10. Logits, J-lens aggregate, and pinned-probe cards no longer repeat probability / strength as a trailing statline value; the single canonical meter value is blue for logits / lens and gold for SAE.
11. Transient hover tooltips remain visible in some screenshots because the browser pointer is left over the control used to switch state; they are not persistent layout elements.

## Interaction and implementation verification

- All four tabs populated and switched successfully in the live app.
- Layout inspection found zero summary boxes and zero legacy context-fact boxes; the generation recipe remained present.
- Geometry rendered all 17 probe cards in two computed 466 px columns with no horizontal drawer overflow at 1280 x 720.
- First-card measurements matched across all four tabs: 24 px leading slot, primary x = 39 px inside the card, 12 px primary type, 14.4 px line-height, and identical primary-text baseline.
- Geometry and SAE supporting facts rendered through the same chip component; SAE contained no `unit strength` text and geometry no longer repeated fraction in a second metadata line.
- Logit and J-lens first-card statlines contained no trailing numeric echo; their meter values computed to the lens pillar blue and SAE's to the SAE pillar gold.
- The 33-row J-lens matrix container matched its 1245 px table height with `overflow-y: visible`; only the 1795 px drawer body scrolled.
- Previous/next token navigation updated token id, raw index, token position, and instrument evidence, then returned to the audit token.
- Browser console: zero warnings and zero errors after the final production reload.
- `npm run check`: zero errors and zero warnings; theme validation passed.
- `npm run build`: production build passed.
- `pytest -q tests/test_instrument_routes.py`: 41 passed; one upstream Starlette/httpx deprecation warning.
- `git diff --check`: passed.

## Final result

passed
