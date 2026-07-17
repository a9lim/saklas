# Token drilldown design QA

## Scope

- Surface: the four populated token-drilldown views at `http://localhost:8000/`.
- Model and state: `google/gemma-3-4b-it`, generated token `" Rayleigh"`, response token 10 of 93, 1280 x 720 viewport.
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

1. The previous views had four unrelated hierarchies. The final views all use the same sequence: token and generation context, evidence-count tabs, instrument summary, four comparable metrics, provenance, rack-style evidence, and a method note.
2. Geometry now distinguishes flat subspaces from curved manifolds through the same marker and accent grammar as the main rack, while retaining coordinates, fraction, residual, membership, nearest nodes, soft assignments, depth, and layer strips.
3. The one-candidate logit state no longer collapses into a nearly empty table. It has a sampling-decision summary and a full evidence card while preserving alternative branching when more candidates exist.
4. SAE now matches the main gold feature-card system and exposes normalized strength, raw activation, maxActApprox metadata, label coverage, source, and resident layer.
5. J-lens now leads with aggregate workspace cards and then preserves the complete layer-by-vocabulary matrix. Every retained matrix cell shows token text and probability, and the generated token is outlined.
6. The first final comparison exposed truncated metric descriptions and oversized native tab tooltips. Metric details now take two lines and drilldown tab/navigation titles were shortened.
7. Transient hover tooltips remain visible in some screenshots because the browser pointer is left over the control used to switch state; they are not persistent layout elements.

## Interaction and implementation verification

- All four tabs populated and switched successfully in the live app.
- Previous/next token navigation updated token id, raw index, token position, and instrument evidence, then returned to the audit token.
- Browser console: zero warnings and zero errors after the final production reload.
- `npm run check`: zero errors and zero warnings; theme validation passed.
- `npm run build`: production build passed.
- `pytest -q tests/test_instrument_routes.py`: 41 passed; one upstream Starlette/httpx deprecation warning.
- `git diff --check`: passed.

## Final result

passed
