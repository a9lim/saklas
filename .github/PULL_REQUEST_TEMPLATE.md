## What

<!-- One or two sentences on what changed -->

## Why

<!-- What problem does this solve? Link issues with "Fixes #N" if applicable -->

## Test plan

- [ ] `ruff check .` passes
- [ ] Non-GPU tests pass (`pytest tests/test_paths.py tests/test_packs.py ...`)
- [ ] GPU smoke tests pass (if touching model/vector/hooks/monitor code)
- [ ] Manually verified against: <!-- model id + device -->

## Notes

<!-- Anything reviewers should know: architectural decisions, followups, known limitations -->
