## What

<!-- One or two sentences on what changed -->

## Why

<!-- What problem does this solve? Please link issues with "Fixes #N" if applicable -->

## Test plan

- [ ] `ruff check .` passes
- [ ] `pyright` passes
- [ ] Non-GPU tests pass (`pytest -q -m "not gpu"`)
- [ ] GPU smoke tests pass (if touching model loading, fitting, generation, hooks, or monitors)
- [ ] `npm run check && npm run build` passes and the committed bundle is current (if touching `webui/`)
- [ ] Manually verified against: <!-- model id + device -->

## Notes

<!-- Anything reviewers should know: architectural decisions, followups, known limitations -->
