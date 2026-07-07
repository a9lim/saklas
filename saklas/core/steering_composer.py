"""SteeringComposer — the steering-resolution + stack collaborator for SaklasSession.

Extracted from :mod:`saklas.core.session` (the one cluster whose seam is clean and
whose hot-path cost is per-scope, not per-token).  Holds the LIFO steering stack
(``_stack``) and every method that runs at steering push/pop frequency: pole-alias
resolution, manifold/profile registration, the lowering of active entries into the
:class:`~saklas.core.hooks.SteeringManager`, hook (re)build, and the probe-gate
plumbing.

It reaches back into the owning session through ``self._session`` for the state it
needs (``_profiles`` / ``_manifolds`` / ``_layer_means`` / ``_monitor`` / whitener /
device / dtype / layers / events / ``_gen_lock`` / ``_gen_phase`` / capture state).
Those reads happen at push/pop frequency, NOT per token, so the extra attribute hop
is off the perf-critical path.  The ONE near-hot method is
:meth:`build_gating_score_callback`'s returned closure (per-token under an active
probe gate); it binds ``capture``/``monitor`` to locals before the inner ``def`` and
reads the session's capture state through the back-ref exactly as the unextracted
body did, so the per-token path gains no new indirection.

The session keeps a thin forwarder for every method here, because
:class:`_SteeringContext`, ``joint_logprobs.py``, and ~15 tests bind these names on
the session (monkeypatch / ``__get__`` to a stub / direct call).  The forwarders make
the extraction behavior- and API-preserving by construction.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import torch

from saklas.core.events import SteeringApplied, SteeringCleared
from saklas.core.manifold import manifold_is_affine
from saklas.core.session import (
    CaptureState,
    ConcurrentGenerationError,
    GenState,
    ManifoldNotRegisteredError,
    SteeringStackEntry,
    Trigger,
    VectorNotRegisteredError,
    _PROFILE_ABSENT,
    _affine_manifold_push,
)
from saklas.core.steering_expr import AblationTerm, ManifoldTerm

if TYPE_CHECKING:
    from saklas.core.session import SaklasSession
    from saklas.core.steering import Steering


class SteeringComposer:
    """Owns the steering stack + steering-resolution/lowering for a session.

    Instantiated once per :class:`~saklas.core.session.SaklasSession` after the
    state it depends on exists (``_steering`` manager, ``_profiles`` /
    ``_manifolds`` / ``_layer_means``, and the ``_monitor``).  The session holds the
    instance as ``_steering_composer`` and exposes ``_stack`` through a settable
    ``session._steering_stack`` property.
    """

    def __init__(self, session: "SaklasSession") -> None:
        self._session = session
        # LIFO stack of per-scope entries dicts pushed by session.steering().
        # Each entry is ``{name: (alpha, Trigger)}`` — triggers are preserved
        # through stack flattening so nested scopes with different trigger regimes
        # compose cleanly.  The flattened head (later entries overwrite earlier
        # ones) is what the steering manager installs when a generation begins.
        self._stack: list[dict[str, SteeringStackEntry]] = []

    # ---------------------------------------------------------------- profiles

    def materialize_projections(self, steering: "Steering") -> dict[str, object]:
        """Populate ``session._profiles`` with derived profiles for every
        :class:`~saklas.core.steering_expr.ProjectedTerm` in
        ``steering.alphas``.

        Ensures the ``base`` and ``onto`` profiles are loaded (invoking
        the autoload path when needed), runs
        :func:`saklas.core.vectors.project_profile` to build the derived
        tensor dict, and registers it under the synthetic key
        ``"<base><op><onto>"``.  The synthetic key matches what the parser
        used for the ``Steering.alphas`` key, so downstream pole
        resolution + hook install find the profile via the
        ``name in self._profiles`` fast path.

        Projection is Mahalanobis-only: the call site passes
        ``session.whitener`` to ``project_profile``, which uses the
        closed-form LEACE projector — provably erases linearly-decodable
        concept information along ``onto``.  ``project_profile`` requires
        the whitener to cover every projected layer (``covers_all``) and
        raises :class:`~saklas.core.mahalanobis.WhitenerError` otherwise;
        there is no Euclidean path.  A ``probes=[]`` session that hasn't
        built a neutral-activation cache yet therefore can't materialize a
        ``~``/``|`` term — regenerate the neutral cache first.

        Returns a snapshot dict ``{syn_key: prev_value_or_PROFILE_ABSENT}``
        of the synthetic-projection bindings this call clobbered, so
        the caller can restore them on scope exit.  Without this
        nested scopes that materialize the same ``a|b`` synthetic
        key would leak the inner tensor back into the outer scope's
        hooks after pop — the global ``self._profiles`` registry is
        shared across all active scopes.
        """
        from saklas.core.steering_expr import ProjectedTerm
        from saklas.core.vectors import project_profile

        whitener = self._session.whitener
        profiles = self._session._profiles

        snapshots: dict[str, object] = {}
        for syn_key, val in steering.alphas.items():
            if not isinstance(val, ProjectedTerm):
                continue
            base_tensors = self.ensure_profile_registered(
                val.base, role="projection base",
            )
            onto_tensors = self.ensure_profile_registered(
                val.onto, role="projection onto",
            )
            projected = project_profile(
                base_tensors, onto_tensors, val.operator,
                whitener=whitener,
            )
            # Snapshot prior binding *before* overwrite so the
            # context manager can restore on exit.  ``setdefault`` —
            # if the same syn_key appears twice in this Steering, only
            # the first occurrence's snapshot matters (subsequent
            # writes are this scope's own, not the outer's).
            if syn_key not in snapshots:
                if syn_key in profiles:
                    snapshots[syn_key] = profiles[syn_key]
                else:
                    snapshots[syn_key] = _PROFILE_ABSENT
            profiles[syn_key] = projected
        return snapshots

    def ensure_manifold_loaded(self, key: str) -> None:
        """Load the manifold artifact for registry key ``key`` if absent.

        ``key`` is the manifold registry key produced by the grammar:
        ``[ns/]name[:variant]``.  ``raw`` (default) selects the
        residual-stream tensor; ``sae-<release>`` selects the SAE-variant
        tensor.  A bare name (no namespace) searches every namespace
        under ``manifolds/``.  The loaded :class:`Manifold` is promoted
        onto the session device (kept fp32 — the spline math wants it).
        Raises :class:`ManifoldNotRegisteredError` on a miss.
        """
        manifolds = self._session._manifolds
        if key in manifolds:
            return
        from saklas.core.manifold import load_manifold
        from saklas.io.paths import manifold_dir, manifolds_dir, tensor_filename

        model_id = self._session.model_id

        name_part, variant = (
            key.rsplit(":", 1) if ":" in key else (key, "raw")
        )
        if "/" in name_part:
            ns, name = name_part.split("/", 1)
            search_ns = [ns]
        else:
            name = name_part
            root = manifolds_dir()
            search_ns = (
                sorted(d.name for d in root.iterdir() if d.is_dir())
                if root.exists() else []
            )
        if variant == "raw":
            release: str | None = None
        elif variant.startswith("sae-"):
            release = variant[len("sae-"):]
        else:
            raise ManifoldNotRegisteredError(
                f"manifold '{key}': unsupported variant '{variant}'"
            )
        fname = tensor_filename(model_id, release=release)

        matches = [
            (ns, manifold_dir(ns, name) / fname)
            for ns in search_ns
            if (manifold_dir(ns, name) / fname).exists()
        ]
        if not matches:
            raise ManifoldNotRegisteredError(
                f"manifold '{key}' has no fitted tensor for {model_id}; "
                f"run `saklas manifold fit` first"
            )
        if len(matches) > 1:
            # A bare name collided across namespaces — refuse rather than
            # silently pick one, mirroring concept selector resolution.
            from saklas.io.selectors import AmbiguousSelectorError
            qualified = ", ".join(f"{ns}/{name}" for ns, _ in matches)
            raise AmbiguousSelectorError(
                f"ambiguous manifold '{name}': matches {qualified}. "
                f"Qualify it with a namespace."
            )
        manifold = load_manifold(matches[0][1])
        manifolds[key] = manifold.to(
            device=self._session._device, dtype=torch.float32,
        )

    def resolve_pole_aliases(
        self, entries: dict[str, tuple[float, Trigger]],
    ) -> dict[str, tuple[float, Trigger]]:
        """Apply pole-alias resolution + sign flipping + variant routing.

        Returned keys carry the full variant-qualified name:
        ``canonical`` for raw, ``f"{canonical}:{variant}"`` otherwise. Autoload
        is variant-aware — ``honest:sae`` will look for a ``_sae-*`` tensor
        file, not the raw one.

        Namespace-qualified inputs (``alice/foo``) keep the namespace
        through resolution: the prefix is split off, the bare tail is
        canonicalized, and the prefix is re-attached to the registry key.
        This is what lets two installed packs that share a concept name
        across namespaces stay addressable via their fully-qualified form.

        Names already in ``self._profiles`` pass through verbatim — a caller
        who pre-registered under a specific key stays addressed by that key.
        """
        from saklas.io.selectors import canonicalize_atom

        profiles = self._session._profiles

        out: dict[str, tuple[float, Trigger]] = {}
        for name, (alpha, trig) in entries.items():
            if name in profiles:
                out[name] = (float(alpha), trig)
                continue
            # Split namespace prefix so re-resolution scopes to the
            # namespace the user originally typed (parser preserves it
            # in the key when supplied).  Bare names leave ``ns=None``
            # so cross-namespace collisions still raise.
            ns: str | None = None
            bare_name = name
            if "/" in name:
                ns, bare_name = name.split("/", 1)
            try:
                canonical, variant = canonicalize_atom(bare_name)
            except Exception:
                out[name] = (float(alpha), trig)
                continue
            canonical_qualified = (
                canonical if ns is None else f"{ns}/{canonical}"
            )
            registry_key = (
                canonical_qualified
                if variant == "raw"
                else f"{canonical_qualified}:{variant}"
            )
            if registry_key not in profiles:
                try:
                    # Unified resolution: fold a fitted manifold, or port a
                    # legacy vectors/ folder on first touch.  May raise
                    # Ambiguous/UnknownVariantError — keep the user's original
                    # name in ``out`` so the error surfaces at hook-install
                    # with a clear message.
                    self.ensure_profile_registered(registry_key)
                except Exception:
                    out[name] = (float(alpha), trig)
                    continue
            # Sign-flip retired with ``resolve_pole`` — a bare pole resolves
            # through the manifold-label tier as a ``%`` push, not a negated
            # vector, so the canonicalized form carries no sign.
            if registry_key in profiles:
                prev_alpha = out.get(registry_key, (0.0, trig))[0]
                out[registry_key] = (prev_alpha + float(alpha), trig)
            else:
                out[name] = (float(alpha), trig)
        return out

    def ensure_profile_registered(
        self, name: str, *, role: str = "vector",
    ) -> dict[int, torch.Tensor]:
        """Direction profile for ``name`` — registered tensor or folded manifold.

        4.0 unifies the sources a steering direction can come from, **manifold
        first** (the 6e "prefer-manifold" stance):

        1. an in-memory baked direction already in ``_profiles`` (ad-hoc
           ``extract``/``clone``/``merge`` results, ``~``/``|`` projection
           derivations) — returned verbatim;
        2. a fitted 2-node ``pca`` manifold on disk (native, or one a prior
           steer already ported) — loaded via :meth:`ensure_manifold_loaded`
           and folded by :func:`~saklas.core.vectors.folded_vector_directions`,
           then memoized in ``_profiles``;
        3. a **stale** (`< PACK_FORMAT_VERSION`) statements-bearing legacy
           ``vectors/<ns>/<name>/`` folder — ported to a 2-node manifold on
           first touch (:meth:`port_stale_legacy_vector`).  The port is
           file-only; fitting runs forward passes through the model and can't
           re-enter the generation lock from dispatch, so a freshly-ported
           manifold has no tensor yet and the call raises with the exact
           ``manifold fit`` command to run (or the bulk migration with ``-m``).

        Raises :class:`VectorNotRegisteredError` when nothing resolves.
        """
        profiles = self._session._profiles
        existing = profiles.get(name)
        if existing is not None:
            return existing
        if ":" in name:
            canonical, variant = name.rsplit(":", 1)
        else:
            canonical, variant = name, "raw"

        # (1b) Reserved J-lens namespace: ``jlens/<word>`` resolves lazily
        # through the model's fitted Jacobian lens (a per-layer ``W_U[v]@J_l``
        # direction registered into the ordinary profile registry).  Raises
        # ``LensNotFittedError`` (with the fit command) or
        # ``MultiTokenWordError`` — never falls through to extraction.
        if canonical.startswith("jlens/") and variant == "raw":
            registered = self._session.register_jlens_direction(
                canonical.split("/", 1)[1]
            )
            return profiles[registered]

        # (2) Manifold first — native or previously ported.
        folded = self.try_fold_manifold(name)
        if folded is not None:
            profiles[name] = folded
            return folded

        # (3) Stale legacy vectors/ folder → port (file-only) and nudge to fit.
        if variant == "raw":
            ported = self.port_stale_legacy_vector(canonical)
            if ported is not None:
                folded = self.try_fold_manifold(name)
                if folded is not None:
                    profiles[name] = folded
                    return folded
                ns, bare = ported
                model_id = self._session.model_id
                raise VectorNotRegisteredError(
                    f"ported legacy vector '{canonical}' to manifold "
                    f"'{ns}/{bare}' (a 2-node pca subspace), but it has no "
                    f"fitted tensor for {model_id} yet — porting is "
                    f"file-only. Fit it: "
                    f"`saklas manifold fit {ns}/{bare} -m {model_id}` "
                    f"(or `python scripts/upgrade_packs.py --all -m {model_id}` "
                    f"to migrate + fit every legacy vector at once)."
                )

        raise VectorNotRegisteredError(
            f"No vector registered for {role} '{name}'"
        )

    def try_fold_manifold(
        self, name: str,
    ) -> dict[int, torch.Tensor] | None:
        """Load a 2-node ``pca`` manifold for ``name`` and fold it to a vector.

        Returns the per-layer direction dict, or ``None`` when no fitted
        manifold resolves (so the caller falls through to porting / autoload).
        Re-raises only when a manifold *does* load but isn't a foldable 2-node
        affine subspace — that's a usage error, not a miss.
        """
        try:
            self.ensure_manifold_loaded(name)
        except Exception:
            return None
        from saklas.core.vectors import folded_vector_directions
        try:
            return folded_vector_directions(self._session._manifolds[name])
        except Exception as e:
            raise VectorNotRegisteredError(
                f"'{name}' is a manifold that does not fold to a single "
                f"steering direction (not a 2-node affine subspace): {e}"
            ) from e

    def port_stale_legacy_vector(
        self, canonical: str,
    ) -> tuple[str, str] | None:
        """Port a stale legacy ``vectors/`` folder to a 2-node ``pca`` manifold.

        4.0 6e port-on-detect.  ``canonical`` is a concept name, optionally
        ``ns/``-qualified.  Scans ``vectors/`` *directly* (not through the
        resolver, which after the v3 bump skips stale v2 folders) for a
        statements-bearing folder whose ``pack.json`` is below
        :data:`~saklas.io.packs.PACK_FORMAT_VERSION`, and ports it via
        :func:`~saklas.io.manifolds.port_legacy_vector_folder` (file-only — no
        tensors carried; they re-fit lazily).  Current-version packs are left
        for the autoload path; tensor-only packs (no ``statements.json``) can't
        re-fit and are skipped.

        Returns ``(namespace, name)`` when a folder was ported or a matching
        manifold already exists (so the caller nudges to fit), else ``None``.
        """
        import json as _json
        from saklas.io.packs import PACK_FORMAT_VERSION
        from saklas.io.paths import vectors_dir, concept_dir, manifold_dir
        from saklas.io.manifolds import port_legacy_vector_folder
        from saklas.io.selectors import invalidate as _invalidate_selectors

        if "/" in canonical:
            ns, bare = canonical.split("/", 1)
            candidates = [(ns, concept_dir(ns, bare))]
        else:
            bare = canonical
            root = vectors_dir()
            candidates = (
                [(nsd.name, nsd / bare) for nsd in sorted(root.iterdir())
                 if nsd.is_dir() and (nsd / bare).is_dir()]
                if root.exists() else []
            )

        for namespace, vfolder in candidates:
            pack_path = vfolder / "pack.json"
            if not pack_path.exists():
                continue
            if not (vfolder / "statements.json").exists():
                continue  # tensor-only — can't re-fit, leave to autoload
            try:
                fmt = _json.loads(pack_path.read_text()).get("format_version", 1)
            except (OSError, ValueError):
                fmt = 1
            if isinstance(fmt, int) and fmt >= PACK_FORMAT_VERSION:
                continue  # current — keep its tensor via autoload-fold
            if (manifold_dir(namespace, bare) / "manifold.json").exists():
                return (namespace, bare)  # already ported; nudge to fit
            try:
                port_legacy_vector_folder(vfolder, namespace=namespace, force=False)
            except Exception:
                continue
            _invalidate_selectors()
            return (namespace, bare)
        return None

    # -------------------------------------------------------- stack push / pop

    def push(
        self,
        entries: dict[str, SteeringStackEntry],
    ) -> None:
        """Push an entries dict onto the steering stack and rebuild hooks.

        If ``rebuild_hooks`` raises (e.g. an unknown vector
        name hits ``VectorNotRegisteredError``) the just-pushed entry is
        rolled back before the exception propagates, so the stack is
        never left with stale half-committed state.

        Thread-safety: acquires :attr:`_gen_lock` (blocking) for the
        rebuild phase.  In-flight generations hold the lock for their
        whole forward+sample loop, so concurrent ``session.steering()
        .__enter__()`` from a different thread waits until the active
        generation finishes rather than mutating hook tensors mid-step.
        Single-threaded users (TUI, CLI) pay an uncontended acquire;
        the server's per-session asyncio lock serializes requests
        upstream so the contention path is exercised mostly by
        ad-hoc multi-threaded callers.

        Phase guard: rejects calls from the gen worker thread when
        :attr:`_gen_phase` is ``RUNNING`` or ``FINALIZING`` (i.e. an
        ``on_token`` callback re-entered the API mid-step).  RLock would
        otherwise let the same-thread caller pass straight through —
        the lock alone protects against cross-thread races, not
        callback reentry.  Legitimate same-thread callers reach
        ``push`` only during ``PREAMBLE`` (the internal
        steering scope that ``_generate_core`` opens before flipping
        to ``RUNNING``) or ``IDLE`` (regular user code between gens).
        """
        session = self._session
        # ``_generate_core``'s internal ``steering_cm.__enter__()`` runs
        # during ``PREAMBLE`` (before the ``RUNNING`` flip), so push
        # never needs the ``_internal_steering_pop`` bypass — only the
        # exit path fires under ``RUNNING``/``FINALIZING``.  The check
        # here catches genuine callback reentry where ``on_token`` /
        # ``score_callback`` calls back into ``session.steering(...)``
        # mid-step.
        if session._gen_phase in (GenState.RUNNING, GenState.FINALIZING):
            raise ConcurrentGenerationError(
                "cannot enter session.steering() from inside a generation "
                "callback (e.g. on_token) — the steering stack mutation "
                f"would clobber hook tensors mid-step (gen_phase="
                f"{session._gen_phase.name})"
            )
        with session._gen_lock:
            self._stack.append(dict(entries))
            try:
                # Through the session forwarder so test stubs that override
                # ``session._rebuild_steering_hooks`` take effect.
                session._rebuild_steering_hooks()
            except BaseException:
                self._stack.pop()
                raise
            # Steering hooks just changed.  A cached prefix is *unsteered*
            # (``cache_prefix`` refuses to run inside a steering scope), so it
            # only stops representing the current pre-attention residual stream
            # when the new stack actually steers the prefill.  A prefill-
            # inactive scope (``@response`` / ``@generated`` / probe-gated)
            # leaves the prompt region untouched, so the unsteered prefix stays
            # valid and we keep it — this is what lets ``generate_batch`` reuse
            # one prefill across a steered batch.  Drop it only when prefill is
            # genuinely steered.
            if self.steering_active_in_prefill():
                session._invalidate_prefix_cache()
        self.emit_steering_applied()

    def pop(self) -> None:
        """Pop the top of the steering stack and rebuild hooks.

        Mirrors :meth:`push` for thread-safety and phase
        guarding: acquires :attr:`_gen_lock` so the rebuild can't fire
        mid-step in another thread's generation, and rejects same-
        thread callback reentry during ``RUNNING`` / ``FINALIZING``.
        """
        session = self._session
        if not self._stack:
            return
        # Same internal-vs-callback distinction as ``push``.
        # ``_generate_core``'s finally block sets the flag around the
        # ``steering_cm.__exit__()`` it owns, so its own scope cleanup
        # passes through; callback reentry (on_token, score_callback)
        # from inside the running loop hits the reject path.
        if (
            not session._internal_steering_pop
            and session._gen_phase in (GenState.RUNNING, GenState.FINALIZING)
        ):
            raise ConcurrentGenerationError(
                "cannot exit session.steering() from inside a generation "
                "callback — the steering stack mutation would clobber "
                f"hook tensors mid-step (gen_phase={session._gen_phase.name})"
            )
        with session._gen_lock:
            self._stack.pop()
            # Through the session forwarder so test stubs that override
            # ``session._rebuild_steering_hooks`` take effect.
            session._rebuild_steering_hooks()
            # Symmetric with ``push``: the cached prefix is unsteered,
            # so only invalidate when what remains on the stack still steers the
            # prefill.  Popping back to a prefill-inactive (or empty) stack keeps
            # the unsteered prefix valid for the next reuse.
            if self.steering_active_in_prefill():
                session._invalidate_prefix_cache()
        if not self._stack:
            session.events.emit(SteeringCleared())
        else:
            self.emit_steering_applied()

    def emit_steering_applied(self) -> None:
        """Emit SteeringApplied with alphas-only + full entries.

        ``alphas`` carries the flat ``{name: alpha}`` shape; ``entries``
        carries the full ``{name: (alpha, trigger)}`` mapping.  Ablation
        entries keyed under ``!<target>`` flatten to their ``(coeff,
        trigger)`` pair so subscribers see one uniform tuple shape.
        """
        flat = self.flatten_steering_stack()
        alphas_only: dict[str, float] = {}
        entries_out: dict[str, tuple[float, Trigger]] = {}
        for name, entry in flat.items():
            if isinstance(entry, (AblationTerm, ManifoldTerm)):
                alphas_only[name] = entry.coeff
                entries_out[name] = (entry.coeff, entry.trigger)
                continue
            alphas_only[name] = entry[0]
            entries_out[name] = entry
        self._session.events.emit(
            SteeringApplied(alphas=alphas_only, entries=entries_out)
        )

    def flatten_steering_stack(self) -> dict[str, SteeringStackEntry]:
        """Collapse the LIFO stack into a single entries dict (later wins)."""
        flat: dict[str, SteeringStackEntry] = {}
        for entry in self._stack:
            flat.update(entry)
        return flat

    def steering_needs_probe_gating(self) -> bool:
        """Return True iff any active steering trigger carries a
        :class:`~saklas.core.triggers.ProbeGate`.

        Walks the flattened steering stack head — entry tuples'
        triggers and ablation entries' triggers both inspected.
        Cheap pre-flight check that lets ``_generate_core`` decide
        whether to wire the per-step score callback at all.
        """
        flat = self.flatten_steering_stack()
        for entry in flat.values():
            if isinstance(entry, (AblationTerm, ManifoldTerm)):
                if entry.trigger.gate is not None:
                    return True
                continue
            # entry is (alpha, Trigger) — the additive / projection shape
            _alpha, trig = entry
            if trig.gate is not None:
                return True
        return False

    def gated_probe_names(self) -> set[str]:
        """Registered probe names referenced by active probe gates.

        Walks the flattened steering stack, pulls each trigger's
        :class:`~saklas.core.triggers.ProbeGate`, and maps the gate's scalar
        ``probe`` key back to the registered monitor probe NAME — the prefix
        before the first of ``[`` / ``:`` / ``@`` / ``~`` (e.g.
        ``"personas[3]"`` → ``"personas"``, ``"emotions:fraction"`` →
        ``"emotions"``, ``"confident.uncertain"`` → itself).  Only names that
        are actually attached probes are kept, so a stale / non-probe gate key
        doesn't shrink an otherwise-empty subset to "score nothing".  Used by
        :meth:`_begin_capture` to scope the per-token step sink to just the
        gated probes when gating is the sole per-token consumer (FIX #4).
        """
        attached = set(self._session._monitor.probe_names)
        out: set[str] = set()
        for entry in self.flatten_steering_stack().values():
            if isinstance(entry, (AblationTerm, ManifoldTerm)):
                trig = entry.trigger
            else:  # (alpha, Trigger)
                _alpha, trig = entry
            gate = trig.gate
            if gate is None:
                continue
            # Split off the channel suffix: the probe name is everything up to
            # the first axis/fraction/label/assignment marker.
            name = re.split(r"[\[:@~]", gate.probe, maxsplit=1)[0]
            if name in attached:
                out.add(name)
        return out

    def gated_probe_keys(self) -> set[str]:
        """Exact monitor scalar keys referenced by active probe gates."""
        attached = set(self._session._monitor.probe_names)
        out: set[str] = set()
        for entry in self.flatten_steering_stack().values():
            if isinstance(entry, (AblationTerm, ManifoldTerm)):
                trig = entry.trigger
            else:  # (alpha, Trigger)
                _alpha, trig = entry
            gate = trig.gate
            if gate is None:
                continue
            name = re.split(r"[\[:@~]", gate.probe, maxsplit=1)[0]
            if name in attached:
                out.add(gate.probe)
        return out

    def steering_active_in_prefill(self) -> bool:
        """Return True iff any active steering term fires during prompt prefill.

        A term touches the prefill residual stream iff its trigger has
        ``prompt=True`` and carries no probe gate (probe gates report
        inactive during prefill — there's no post-forward score yet; see
        :meth:`~saklas.core.triggers.Trigger.active`).  When this is False the
        prefill is numerically identical to the *unsteered* prefill, so a
        cached unsteered prefix KV (the only kind :meth:`cache_prefix` builds)
        stays valid for reuse.  This is the gate the prefix-cache consume path
        keys on, and the condition under which steering push/pop preserves the
        cache.  Mirrors :meth:`steering_needs_probe_gating`'s stack walk.
        """
        flat = self.flatten_steering_stack()
        for entry in flat.values():
            if isinstance(entry, (AblationTerm, ManifoldTerm)):
                trig = entry.trigger
            else:  # (alpha, Trigger)
                _alpha, trig = entry
            if trig.prompt and trig.gate is None:
                return True
        return False

    def steering_value_prefill_inactive(
        self, value: "str | Steering | None",
    ) -> bool:
        """Return True iff steering ``value`` would not touch the prompt prefill.

        Pre-flight form of :meth:`steering_active_in_prefill` over an
        *incoming* steering value (before any scope push) — used by
        :meth:`generate_batch` to decide whether a shared-prefix KV cache is
        reusable across the batch.  ``None`` (no steering) is trivially
        prefill-inactive; otherwise a term is prefill-active iff its trigger
        has ``prompt=True`` and no probe gate.  A malformed expression (raises
        on parse) is conservatively reported active so the caller skips
        caching and lets the normal path surface the error.
        """
        from saklas.core.steering import Steering
        from saklas.core.steering_expr import ProjectedTerm

        try:
            s = Steering.from_value(
                value, profile_names=set(getattr(self._session, "_profiles", {})),
            )
        except Exception:
            return False
        if s is None:
            return True
        default = s.trigger
        for val in s.alphas.values():
            if isinstance(val, (AblationTerm, ManifoldTerm, ProjectedTerm)):
                trig = val.trigger
            elif isinstance(val, tuple):
                trig = val[1]
            else:  # bare float — inherits the Steering default trigger
                trig = default
            if trig.prompt and trig.gate is None:
                return False
        return True

    def build_gating_score_callback(self):
        """Return a closure that scores latest captures into a
        ``dict[str, float]`` for ``generate_steered``'s ``score_callback``.

        The closure pulls ``session._capture.latest_per_layer()`` (the
        most-recent ``[D]`` slice per layer the steering hooks captured) and
        runs it through :meth:`Monitor.score_single_token` — one unified pass
        over every probe shape — flattening to gate scalars via
        :meth:`Monitor.flat_scalars`.  Returns an empty dict when the capture
        is empty (e.g. before the first forward) so probe gates report
        inactive instead of seeing stale values from a previous gen.

        Caller-side guard: only invoked when
        :meth:`steering_needs_probe_gating` is True, so the no-gate
        path stays at zero overhead.

        Hot path: the returned closure runs per token under an active gate.  It
        binds ``capture``/``monitor`` to locals before the inner ``def`` and
        reads the session's capture state (``_capture_state`` /
        ``_incremental_readings`` / ``_incremental_gate_scores``) through one
        plain ``self._session.`` hop, exactly as the unextracted body did — no
        new per-token indirection.
        """
        session = self._session
        capture = session._capture
        monitor = session._monitor

        def _score() -> dict[str, float]:
            incremental_readings = getattr(session, "_incremental_readings", [])
            incremental_gate_scores = getattr(session, "_incremental_gate_scores", [])
            # The step sink already scored this token's readings — reuse them so
            # the gate doesn't trigger a second pass.  In gating-only-subset mode
            # (FIX #4) the rows hold just the gated probes, which is exactly the
            # set the gate consults; in the full incremental mode they hold the
            # whole roster.  Both reuse the latest appended row.
            state = getattr(session, "_capture_state", None) or CaptureState()
            gating_subset = state.gating_subset
            gate_keys = state.gating_keys
            if gating_subset and incremental_gate_scores:
                return incremental_gate_scores[-1]
            if state.incremental and incremental_readings:
                scalars = monitor.flat_scalars(incremental_readings[-1])
                if gate_keys:
                    missing = gate_keys - set(scalars)
                    if missing:
                        latest = capture.latest_per_layer()
                        if latest:
                            scalars.update(
                                monitor.score_gate_scalars(
                                    latest,
                                    missing,
                                    probe_names=gating_subset if gating_subset else None,
                                )
                            )
                return scalars
            latest = capture.latest_per_layer()
            if not latest:
                return {}
            if gate_keys:
                return monitor.score_gate_scalars(
                    latest,
                    gate_keys,
                    probe_names=gating_subset if gating_subset else None,
                )
            # Flatten the coordinate readings into gate-callback scalars
            # (``name`` aliases axis 0, ``name[i]`` per axis, ``name:fraction``,
            # ``name@label`` for curved nearest).  Scope to the gated subset
            # when the per-token path is gating-only (avoids the full roster).
            agg = monitor.score_single_token(
                latest, only=gating_subset if gating_subset else None,
            )
            return monitor.flat_scalars(agg)

        return _score

    # ------------------------------------------------------------- dispatch

    def compose_steering_entries(
        self,
        entries: dict[str, SteeringStackEntry],
    ) -> None:
        """Lower the active steering entries into the ``SteeringManager`` (4.0).

        Classifies every entry and composes the one unified backend:

        - **push** terms (vectors, poles, ``~``/``|`` projections, affine-``%``
          manifolds) and **ablation** terms (``!``) are grouped by trigger;
          each group is synthesized into one merged affine subspace via
          :func:`~saklas.core.manifold.synthesize_subspace` and registered with
          :meth:`SteeringManager.add_subspace`;
        - **curved-``%``** manifold terms each get their own two-op via
          :meth:`SteeringManager.add_manifold`.

        A push fragment is ``(unit-dir rows, ‖d_L‖ coord, coeff)`` so the
        synthesizer's ``Δ = Σ coeff·(coord @ basis)`` recovers the baked
        direction; the coeff is the term's ``along`` (the strength composes into
        the merged target).  Ablation directions slide their axis to the
        neutral-anchored origin (coord 0 = mean-replacement).  The neutral
        anchor is :attr:`layer_means`; a layer with no anchor is skipped.
        """
        from saklas.core.manifold import synthesize_subspace

        session = self._session
        steering = session._steering

        steering.clear_all()
        # Raw attr (not the lazily-building ``layer_means`` property): real
        # sessions populate it at construction / first extraction, and a
        # model-less context (test stub) keeps an empty dict rather than
        # triggering a model-dependent build.  An empty anchor ⇒ the synthesizer
        # skips every layer (no steering), the same degenerate path the old
        # ablation took without ``layer_means``.
        neutral_means = session._layer_means

        # Whitened push normalization (Mahalanobis-only): hand the synthesizer the
        # session whitener so the per-layer ``along`` budget is the whitened
        # displacement ``‖Δ‖_M`` and the target is a whitened-unit direction —
        # making ``along`` a scale-stable strength knob instead of inheriting each
        # node's raw-Euclidean distance from neutral (which spans ~100× across
        # targets).  Gated on a real session (means populated): a model-less stub
        # keeps the Euclidean fallback, and the property soft-fails to ``None``
        # (covers_all is then false) so a missing whitener degrades, never raises.
        whitener = session.whitener if neutral_means else None

        # trigger -> {"push": [(basis_dirs, coord_dirs, coeff)], "ablate": [dirs]}
        grouped: dict[Trigger, dict[str, list[Any]]] = {}

        def _bucket(trigger: Trigger) -> dict[str, list[Any]]:
            return grouped.setdefault(trigger, {"push": [], "ablate": []})

        from saklas.core.vectors import fold_directions_to_subspace

        for name, entry in entries.items():
            if isinstance(entry, AblationTerm):
                ablate_prof = self.ensure_profile_registered(
                    entry.target, role="ablation target",
                )
                ablate_dirs = {
                    L: v.to(torch.float32).reshape(-1)
                    for L, v in ablate_prof.items()
                }
                _bucket(entry.trigger)["ablate"].append(ablate_dirs)
                continue
            if isinstance(entry, ManifoldTerm):
                manifold = session._manifolds.get(entry.manifold)
                if manifold is None:
                    raise ManifoldNotRegisteredError(
                        f"No manifold registered for '{entry.manifold}'"
                    )
                if manifold_is_affine(manifold):
                    # Affine ``%`` joins the merged subspace as a rank-R push
                    # toward the position's per-layer coords — a node's real
                    # coords for label form (``personas%pirate``) or the cardinal
                    # RBF layout blend for free coord form (``personas%c0,c1,…``).
                    basis_dirs, coord_dirs = _affine_manifold_push(
                        manifold, entry.position,
                    )
                    _bucket(entry.trigger)["push"].append(
                        (basis_dirs, coord_dirs, entry.along),
                    )
                else:
                    steering.add_manifold(
                        entry.manifold, manifold,
                        position=entry.position,
                        along=entry.along, onto=entry.onto,
                        trigger=entry.trigger,
                    )
                continue
            alpha, trigger = entry
            # 4.0 step 6b: every in-memory direction (ad-hoc extracts, clones,
            # merges, ``~``/``|`` projections, or a folded bundled concept)
            # lowers through the one push path — fold it to a neutral-anchored
            # one-pole-ray subspace and push label-form, exactly as an affine
            # ``%`` term does.  There is no separate baked-vector fragment.
            prof = self.ensure_profile_registered(name)
            folded = fold_directions_to_subspace(name, prof, neutral_means)
            if not folded.layers:
                continue
            basis_dirs, coord_dirs = _affine_manifold_push(
                folded, folded.node_labels[0],
            )
            _bucket(trigger)["push"].append((basis_dirs, coord_dirs, alpha))

        # One merged affine subspace per active trigger group.
        for i, (trigger, terms) in enumerate(grouped.items()):
            if not terms["push"] and not terms["ablate"]:
                continue
            synth = synthesize_subspace(
                terms["push"], terms["ablate"], neutral_means=neutral_means,
                whitener=whitener,
            )
            if not synth.layers:
                continue
            steering.add_subspace(
                f"__affine__{i}", synth, trigger=trigger,
            )

    def install_composed_steering(self) -> None:
        """Attach the currently-composed steering entries to model layers.

        Compiled fast path (MPS): a static-affine pure-push steering lowers to
        the persistent branchless offset buffers (traced into the compiled
        graph) instead of transient ctx-consulting hooks — the latter would
        force a per-gen ``torch.compile`` recompile and graph-break at every
        layer.  Gated on the offsets being available (compiled session) and on
        the capture being compile-clean too: either no probes, or this gen is
        ``_compiled_clean_eligible`` so :meth:`_begin_capture` will ride the
        persistent capture buffers instead of transient capture hooks (slice 2).
        Anything that isn't a constant add (curved ``%``, gate, phase, ``!``
        ablation) returns ``None`` from :meth:`compute_static_offsets` and falls
        through to the transient eager path.
        """
        session = self._session
        steering = session._steering
        session._steering_uses_compiled_offsets = False
        # ``getattr`` defaults keep skeleton sessions (``__new__`` test stubs that
        # bypass ``__init__``) on the transient path — they never compile.
        if (
            getattr(session, "_compiled", False)
            and session._device.type == "mps"
            and steering.has_compiled_offsets()
            and (
                not session._monitor.probe_names
                or getattr(session, "_compiled_clean_eligible", False)
            )
        ):
            offsets = steering.compute_static_offsets()
            if offsets is not None:
                steering.detach_transient_hooks()
                steering.write_compiled_offsets(offsets)
                session._steering_uses_compiled_offsets = True
                return
        # Eager / general path: zero the persistent offsets (no stale push leaks
        # into the eager model, whose layers carry the same persistent hooks)
        # and attach the transient ctx-consulting hooks.
        if steering.has_compiled_offsets():
            steering.zero_compiled_offsets()
        steering.apply_to_model(
            session._layers, session._device, session._dtype,
        )

    def rebuild_hooks(self) -> None:
        """Tear down existing hooks and install from the flattened stack head.

        Called on every push/pop.  When the stack is empty this is a clean
        ``clear_all``.  One hook installation per active layer regardless of
        nesting depth — ``compose_steering_entries`` synthesizes the merged
        affine subspace(s) + registers curved manifolds, and
        ``SteeringManager.apply_to_model`` lowers them to per-layer
        ``subspace_inject`` groups.
        """
        flat = self.flatten_steering_stack()
        if not flat:
            self._session._steering.clear_all()
            return
        self.compose_steering_entries(flat)
        self.install_composed_steering()

    def snapshot_steering_alphas(self) -> dict[str, float]:
        """Flatten the active steering stack for ``GenerationResult.vectors``.

        For a plain vector entry, ``entry[0]`` is the additive alpha (the
        strength coefficient, typically in ``[0, 1]``).

        For a :class:`~saklas.core.steering_expr.ManifoldTerm`, ``entry.coeff``
        delegates to ``entry.along`` — the *slide fraction* toward the manifold
        position (clamped ``[0, 1]`` at injection time), not an additive
        scalar.  These two kinds are semantically different (slide fraction vs
        additive alpha), but both are reported as a single ``float`` here for
        a uniform telemetry snapshot.  Callers that need to distinguish them
        should inspect the live ``Steering.alphas`` dict directly.
        """
        snap: dict[str, float] = {}
        for name, entry in self.flatten_steering_stack().items():
            if isinstance(entry, (AblationTerm, ManifoldTerm)):
                snap[name] = entry.coeff
                continue
            snap[name] = entry[0]
        return snap
