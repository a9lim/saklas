"""The instrument contract — one shape over the three read families.

Design prose only: there are no Protocol classes here.  A ``Protocol`` that
nothing annotates against is checked by nothing, so it drifts silently from
the implementations it claims to describe, and methods get written to satisfy
it that no caller ever reaches.  The contract below is the one the session,
the composer, and the server actually exercise; the concrete families
(``geometry.py`` / ``lens.py`` / ``sae.py``) are its only definition.

An **Instrument** is the persistent, session-lifetime object for one read
family: it owns the attached-probe registry, prepares its source at the
generation boundary (``prepare`` — the disk refresh + pin decision), declares
capture demand (``plan``), and binds an immutable per-generation
**InstrumentRun**.  A capture transaction is the uniform sequence
``close_run → prepare → plan → bind`` — source refresh strictly precedes
planning because adoption may rewrite live probe specs.  Every ``prepare``
raises on a still-bound run (a stale pin short-circuits the very refresh the
step exists for), every prep carries a per-preparation ``token`` its plan
echoes, and ``bind`` refuses a plan/prep pair whose tokens or families
disagree.

An **InstrumentRun** owns the instrument-side generation-scoped state: the
immutable source/spec binding (``binding``), step stashes, and per-generation
active flags (geometry's curved warm feet stay in the Monitor engine).
``bound`` distinguishes a generation-bound run from the idle passthrough run
that backs out-of-generation reads.  Repeated ``observe`` calls for the same
``step_id`` are memoized while the run is bound — an idle run never memoizes,
since it persists indefinitely and a repeated step id with different hidden
states must not read stale.  The hot paths are wired through step identity:
the decode loop owns one ``step_id`` per forward (``len(generated_ids)``
pre-forward) and hands the SAME value to the capture sink (``step_callback``),
the gate callback (``score_callback``), and the token tap (the internal
``StepTokenCallback``).  Full-roster reads — the FULL-incremental sink, the
lens/SAE display readings — call ``prime_observation`` so one forward's gate
and payload reads share a single scoring pass; partial reads (gating scalar
subsets, lean ``coords_only`` rows, ``only=`` restrictions) NEVER prime.  The
matrix-granular gate→display reuse (band logits, encoded activations) stays
the workers' stash mechanism, step-keyed (``stash["step"] == step_id`` —
structural staleness, idempotent reuse): a reading-level memo cannot carry
logit-level partial-row reuse without a second host sync.

Division of labor the contract deliberately does NOT own:

* **Capture planning** — instruments declare demand
  (:class:`~saklas.core.instruments.types.InstrumentPlan`); the session
  planner unions demands and picks physical retention.  The
  ``INCREMENTAL -> set_tail_with_sink`` upgrade is cross-instrument
  resource sharing and stays session-side.
* **Authored-prefill orchestration** — token matching, the ``j-1``
  producer-position semantics, ordering, and loom persistence stay in the
  session; instruments only ``observe`` the hidden rows handed to them.
* **The wire** — runs return readings; serialization into the versioned
  measurement envelope and its compatibility channels is the payload layer's
  job.
"""

from __future__ import annotations

__all__: list[str] = []
