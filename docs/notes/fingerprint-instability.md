# RESOLVED: `loaded_model_fingerprint` was unstable across identical loads

*Filed 2026-07-10 during the UI-redesign Phase 3 verification. Owner: a9
(future session). Two adjacent bugs found the same day are already fixed —
see "context" below — this is the remaining open one.*

Resolved 2026-07-10. Twelve clean MPS loads reproduced the reported split
(8 produced `169717…`, 4 produced `7490c8…`). The config, trusted checkpoint
fingerprint, parameter schema, and state signature were identical. Gemma-3
registered the same four rotary buffers in two possible orders; the fingerprint
treated `named_buffers()` traversal order as identity. Parameter and buffer rows
are now sorted by name before both schema hashing and the exact-state fallback,
with a registration-order regression test.

## Symptom

The same model (`google/gemma-3-4b-it`), loaded by the same command on the
same machine, produces **different `model_fingerprint` digests across
processes**. Observed values during one afternoon:

- `7490c89ddcd481121d0d2088af91506b568e3bef74ceaa6f2b80152e75dd7d47`
- `169717bce4b24e50a5beb429a230c1e90957c15b4aa5b357b562cea7424245fe`

Five sequential `saklas manifold fit <name> -m google/gemma-3-4b-it`
invocations (separate processes) split across the two values; a subsequent
`saklas serve` boot produced one of them, a later boot the other.

## Blast radius

Every artifact that binds the loaded-weights identity:

1. **Bundled-manifold fits ping-pong re-fit at every serve boot.** The
   constructor's probe bootstrap re-fits whichever half of the roster has
   the mismatched fingerprint. The capture cache makes each re-fit take
   seconds, so this is *silent* — but it rewrites sidecars every launch and
   means fits never stabilize.
2. **The J-lens artifact reads as "not fitted" on mismatched boots.** Its
   sidecar binds the fit-time fingerprint; on the other fingerprint the
   dashboard's lens tab shows the fit button instead of the fitted lens.
   Live-observed on 4b with a valid `models/google__gemma-3-4b-it/jlens.*`
   on disk.
3. Anything else validating "exact loaded model identity" (neutral-cache
   metadata appears to have passed on both boots — worth checking whether
   it binds a different identity or just got lucky).

## Evidence in place

- Sidecars: `~/.saklas/manifolds/default/*/google__gemma-3-4b-it.json` —
  `model_fingerprint` values split across the two digests depending on
  which process wrote them.
- The fingerprint implementation: `core/model.py::loaded_model_fingerprint`
  (~line 202) — payload folds `model_id`, class path, `config.to_dict()`,
  `trusted_source`, parameter/buffer schemas, then (past the payload hash)
  distributed weight samples / state signatures.

## Diagnosis pointers

Prime suspect: an **unstable key inside `config.to_dict()`** — e.g. the
attention-implementation fallback cascade (SDPA → eager) landing
differently across processes, a dtype field, or any key carrying a runtime
object serialized via `default=str` (which stringifies memory addresses).
Second suspect: the weight-sample tier differing under MPS load order.

To pin it: dump the full fingerprint payload (pre-hash JSON) from two
fresh processes and diff — one nondeterministic key should fall out
immediately. Then decide whether the key belongs in the identity at all
(fallback-chosen attn impl arguably does NOT — same weights, same math)
or needs canonicalization.

## Context (already fixed the same day, commits on dev)

- Bundle-refresh clobbered fit proofs (`files` map) → orphaned tensors —
  fixed in `io/manifolds.py` (`_manifest_content_sha256` + proof
  carry-forward) with regression tests; a9's 20 default folders repaired.
- `_adopt_fitted_manifold` touched `self._monitor` before construction on
  the boot-time re-fit path — guarded in `core/session.py`.
- Note the interaction: the fingerprint instability *triggers* boot-time
  re-fits, which is how the `_monitor` crash surfaced. Fixing the
  fingerprint will also stop the re-fit churn that bug rode in on.
