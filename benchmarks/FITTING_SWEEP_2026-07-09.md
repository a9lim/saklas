# Fitting optimization sweep evidence — 2026-07-09

Comparison base: `1a39e1d` (`perf: optimize manifold and j-lens fitting`).
Candidate: the subsequent fitting sweep that ships this record. Commands used
the repository's Python 3.11 virtualenv.

Environment: Apple M5 Max, 128 GB unified memory, macOS 27.0, MPS; PyTorch
2.12.1 and Transformers 5.12.1. Process RSS is the OS process-lifetime
high-water mark, so it is reported for the force-fit process only and is not
misrepresented as phase-local cache RSS.

## Representative J-lens fit

Workload: cached `google/gemma-3-4b-it`, two 24-token-truncated English corpus
prompts, workspace-band source layers, `dim_batch=8`. The base used its default
one-prompt graph; the candidate used `prompt_batch=2`. Both runs started with a
fresh temporary `SAKLAS_HOME` and included durable final artifact writing.

| tree | wall time | model forwards | process peak RSS |
|---|---:|---:|---:|
| `1a39e1d` | 84.293 s | 2 | 4,161,568,768 B |
| candidate | 49.080 s | 1 | 4,161,404,928 B |

The candidate is 1.72x faster (41.8% lower wall time), halves model forwards,
and does not increase the process RSS high-water mark. Its immediate exact
repeat took 0.244 s and issued zero model forwards; that repeat includes
token-ID revalidation plus the v2 artifact digest check.

Reproduction command for the candidate:

```bash
python scripts/benchmark_fitting.py jlens google/gemma-3-4b-it \
  --corpus benchmarks/fixtures/jlens_prompts.txt --prompts 2 \
  --layers workspace --seq-len 24 --dim-batch 8 --prompt-batch 2
```

## Manifold capture/cache work

A 20-iteration geometry-only refit microbenchmark used the deterministic
five-node, four-layer extraction fixture. Geometry alternated while corpus,
token IDs, node partition, and loaded weights stayed fixed.

| tree | wall time | residual-capture forwards |
|---|---:|---:|
| `1a39e1d` | 0.112 s | 20 |
| candidate | 0.111 s | 0 |

The synthetic wall time is dominated by tiny CPU fit/JSON work and is not a
model-speed proxy; the useful result is the eliminated model work. Regression
tests additionally pin these work contracts:

- full capture to subset fit: zero forwards and only requested row layers read;
- partial `[1,3]` to full fit: only missing layers `[0,2]` captured;
- geometry-only refit: zero forwards via token-exact activation reuse;
- known-bad OOM batch widths are not retried;
- reusable capture hooks register once per selected layer across forwards;
- auto-curved topology reuses its Fisher bases and retained activation rows.

The real-model harness also supports forced manifold fits and exact repeats:

```bash
python scripts/benchmark_fitting.py manifold MODEL MANIFOLD_FOLDER \
  --layers workspace
```

Correctness gates for this sweep are recorded in the commit handoff, including
the full non-GPU suite and focused fallback/resume/cache-integrity tests.
