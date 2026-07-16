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

The session exposes the artifact loaders as its public registration API and keeps
narrow context/generation forwarders where the session boundary is meaningful.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch

from saklas.core.events import SteeringApplied, SteeringCleared
from saklas.core.manifold import manifold_is_affine
from saklas.core.session import (
    ConcurrentGenerationError,
    GenState,
    ManifoldNotRegisteredError,
    SteeringStackEntry,
    Trigger,
    ProfileNotRegisteredError,
    _PROFILE_ABSENT,
    _affine_manifold_push,
)
from saklas.core.steering_expr import AblationTerm, ManifoldTerm

if TYPE_CHECKING:
    from saklas.core.instruments.types import GateRef
    from saklas.core.session import SaklasSession
    from saklas.core.steering import Steering


class SteeringComposer:
    """Owns the steering stack + steering-resolution/lowering for a session.

    Instantiated once per :class:`~saklas.core.session.SaklasSession` after the
    state it depends on exists (``_steering`` manager, ``_profiles`` /
    ``_manifolds`` / ``_layer_means``, and the ``_monitor``).  The session holds the
    instance as ``_steering_composer``. The stack has one owner: this collaborator.
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
        :func:`saklas.core.capture.project_profile` to build the derived
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
        from saklas.core.capture import project_profile

        profiles = self._session._profiles

        snapshots: dict[str, object] = {}
        whitener = None
        for syn_key, val in steering.alphas.items():
            if not isinstance(val, ProjectedTerm):
                continue
            # Do not wake the lazy neutral-activation/whitener pipeline for an
            # ordinary steering expression.  Plain vector/manifold terms are
            # normalized later by ``compose_steering_entries``; only an actual
            # projection needs the whitener at materialization time.
            if whitener is None:
                whitener = self._session.whitener
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
        tensor; ``from-*`` selects a cross-model transfer; ``role-*``
        validates the canonical tensor's role baseline. A bare name searches
        under ``manifolds/``.  The loaded :class:`Manifold` is promoted
        onto the session device (kept fp32 — the spline math wants it).
        Raises :class:`ManifoldNotRegisteredError` on a miss.
        """
        manifolds = self._session._manifolds
        if key in manifolds:
            return
        from saklas.io.manifold_tensors import load_manifold
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
        transferred_from: str | None = None
        requested_role: str | None = None
        if variant == "raw":
            release: str | None = None
        elif variant.startswith("sae-"):
            release = variant[len("sae-"):]
        elif variant.startswith("from-"):
            release = None
            transferred_from = variant[len("from-"):]
        elif variant.startswith("role-"):
            release = None
            requested_role = variant[len("role-"):]
        else:
            raise ManifoldNotRegisteredError(
                f"manifold '{key}': unsupported variant '{variant}'"
            )
        fname = tensor_filename(
            model_id, release=release, transferred_from=transferred_from,
            transferred_from_is_encoded=transferred_from is not None,
        )

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
        tensor_path = matches[0][1]
        manifold = load_manifold(tensor_path)
        metadata = manifold.metadata
        from saklas.core.model import loaded_model_fingerprint
        from saklas.io.paths import encode_release_id

        live_fingerprint = loaded_model_fingerprint(
            self._session._model, model_id,
        )
        if transferred_from is not None:
            source_id = metadata.get("source_model_id")
            if (
                not isinstance(source_id, str)
                or encode_release_id(source_id) != transferred_from
            ):
                raise ManifoldNotRegisteredError(
                    f"manifold '{key}' transfer provenance does not match "
                    f"source {transferred_from!r}; recompute the transfer"
                )
        raw_roles = metadata.get("node_roles")
        roles = (
            list(raw_roles)
            if isinstance(raw_roles, list)
            else [None] * len(manifold.node_labels)
        )
        if requested_role is not None and (
            not roles or not all(role == requested_role for role in roles)
        ):
            raise ManifoldNotRegisteredError(
                f"manifold '{key}' was not fitted uniformly with role "
                f"{requested_role!r}; refit it with that role"
            )
        if variant == "raw" and any(role is not None for role in roles):
            raise ManifoldNotRegisteredError(
                f"manifold '{key}' was fitted with role baseline "
                f"{roles!r}; steer it as :role-<name> or refit raw"
            )
        if metadata.get("model_fingerprint") != live_fingerprint:
            raise ManifoldNotRegisteredError(
                f"manifold '{key}' was fitted for different loaded weights "
                f"under {model_id}; run `saklas manifold fit` again"
            )
        if metadata.get("fit_mode", "authored") != "baked":
            from saklas.core.extraction import prepare_manifold_capture_identity
            from saklas.io.manifold_folder import ManifoldFolder

            folder_mf = ManifoldFolder.load(
                tensor_path.parent, verify_manifest=False,
            )
            expected_capture = prepare_manifold_capture_identity(
                self._session, folder_mf, live_fingerprint,
            )[3]
            if metadata.get("capture_sha256") != expected_capture:
                raise ManifoldNotRegisteredError(
                    f"manifold '{key}' capture inputs changed (tokenizer, chat "
                    "template, role framing, or baseline prompts); refit it"
                )
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
            if variant in {"sae", "from", "role"}:
                import json

                from saklas.core.errors import (
                    AmbiguousVariantError, UnknownVariantError,
                )
                from saklas.io.paths import (
                    manifold_dir, manifolds_dir, parse_tensor_filename,
                    safe_model_id,
                )

                roots = (
                    [(ns, manifold_dir(ns, canonical))]
                    if ns is not None else [
                        (entry.name, entry / canonical)
                        for entry in manifolds_dir().iterdir()
                        if entry.is_dir()
                    ] if manifolds_dir().exists() else []
                )
                concrete: set[str] = set()
                safe_model = safe_model_id(self._session.model_id)
                for _candidate_ns, folder in roots:
                    if variant == "role":
                        sidecar_path = folder / f"{safe_model}.json"
                        if not sidecar_path.exists():
                            continue
                        with open(sidecar_path) as handle:
                            raw_sidecar = json.load(handle)
                        roles = raw_sidecar.get("node_roles") or [
                            None
                        ] * int(raw_sidecar.get("node_count", 0))
                        if roles and all(role == roles[0] for role in roles):
                            if roles[0] is not None:
                                concrete.add(f"role-{roles[0]}")
                        continue
                    for tensor_path in folder.glob(f"{safe_model}_*.safetensors"):
                        parsed = parse_tensor_filename(tensor_path.name)
                        if parsed is None or parsed[1] is None:
                            continue
                        parsed_variant = parsed[1]
                        if parsed_variant.startswith(f"{variant}-"):
                            concrete.add(parsed_variant)
                if len(concrete) != 1:
                    error = (
                        UnknownVariantError
                        if not concrete else AmbiguousVariantError
                    )
                    raise error(
                        f"manifold '{canonical_qualified}:{variant}' resolves "
                        f"to {sorted(concrete) or 'no fitted variants'}"
                    )
                variant = next(iter(concrete))
            registry_key = (
                canonical_qualified
                if variant == "raw"
                else f"{canonical_qualified}:{variant}"
            )
            if registry_key not in profiles:
                try:
                    # Unified resolution folds the fitted manifold. May raise
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

        Steering directions come from the current manifold-first runtime:

        1. an in-memory baked direction already in ``_profiles`` (ad-hoc
           ``extract``/``clone``/``merge`` results, ``~``/``|`` projection
           derivations) — returned verbatim;
        2. a fitted 2-node ``pca`` manifold on disk — loaded via
           :meth:`ensure_manifold_loaded`
           and folded by :func:`~saklas.core.capture.folded_directions`,
           then memoized in ``_profiles``;

        Raises :class:`ProfileNotRegisteredError` when nothing resolves.
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

        # (1c) Reserved SAE namespace: ``sae/<integer>`` is the resident
        # release's decoder row at its hook layer. It is a steering-only
        # profile; feature probes use the encoder readout channel instead.
        if canonical.startswith("sae/") and variant == "raw":
            registered = self._session.register_sae_direction(
                canonical.split("/", 1)[1]
            )
            return profiles[registered]

        # (2) Manifold-backed direction.
        folded = self.try_fold_manifold(name)
        if folded is not None:
            profiles[name] = folded
            return folded

        raise ProfileNotRegisteredError(
            f"No profile registered for {role} '{name}'"
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
        from saklas.core.capture import folded_directions
        try:
            return folded_directions(self._session._manifolds[name])
        except Exception as e:
            raise ProfileNotRegisteredError(
                f"'{name}' is a manifold that does not fold to a single "
                f"steering direction (not a 2-node affine subspace): {e}"
            ) from e

    # -------------------------------------------------------- stack push / pop

    def push(
        self,
        entries: dict[str, SteeringStackEntry],
    ) -> None:
        """Push an entries dict onto the steering stack and rebuild hooks.

        If ``rebuild_hooks`` raises (e.g. an unknown vector
        name hits ``ProfileNotRegisteredError``) the just-pushed entry is
        rolled back before the exception propagates, so the stack is
        never left with stale half-committed state.

        Thread-safety: acquires :attr:`_gen_lock` (blocking) for the
        rebuild phase.  In-flight generations hold the lock for their
        whole forward+sample loop, so concurrent ``session.steering()
        .__enter__()`` from a different thread waits until the active
        generation finishes rather than mutating hook tensors mid-step.
        Single-threaded users pay an uncontended acquire;
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
                # Rebuild through the session boundary after the stack mutation.
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
            # Rebuild through the session boundary after the stack mutation.
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
        flat = self.flatten_stack()
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

    def flatten_stack(self) -> dict[str, SteeringStackEntry]:
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
        flat = self.flatten_stack()
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

    def _gated_refs(self) -> "list[GateRef]":
        """Every active probe-gate reference, parsed once into its
        structured form (:func:`parse_gate_ref` — the ONE place the
        scalar-key discrimination lives).  Walks the flattened steering
        stack exactly like the historical per-family key walks, which this
        replaces.  Note the structured parse also fixes the old
        ``re.split("[\\[:@~]")`` truncation of variant-suffixed probe names
        (``pirate:role-x`` no longer truncates at ``:role-x``).
        """
        from saklas.core.instruments.types import parse_gate_ref

        refs = []
        for entry in self.flatten_stack().values():
            if isinstance(entry, (AblationTerm, ManifoldTerm)):
                trig = entry.trigger
            else:  # (alpha, Trigger)
                _alpha, trig = entry
            gate = trig.gate
            if gate is None:
                continue
            refs.append((gate.probe, parse_gate_ref(gate.probe)))
        return refs

    def gated_probe_names(self) -> set[str]:
        """Registered monitor probe names referenced by active probe gates.

        Only names that are actually attached probes are kept, so a stale /
        non-probe gate key doesn't shrink an otherwise-empty subset to
        "score nothing".  Used by ``_begin_capture`` to scope the per-token
        step sink to just the gated probes when gating is the sole
        per-token consumer (FIX #4).
        """
        attached = set(self._session._monitor.probe_names)
        return {
            ref.probe for _key, ref in self._gated_refs()
            if ref.probe in attached
        }

    def gated_probe_keys(self) -> set[str]:
        """Exact monitor scalar keys referenced by active probe gates."""
        attached = set(self._session._monitor.probe_names)
        return {
            key for key, ref in self._gated_refs()
            if ref.probe in attached
        }

    def gated_lens_probe_keys(self) -> set[str]:
        """Exact gate scalar keys referencing attached J-lens token probes.

        The lens sibling of :meth:`gated_probe_keys`: lens probes live in
        the session lens instrument (readout channel), not the Monitor.
        A key whose base name is an attached lens probe is also
        **channel-validated** here — the lens family produces only the
        strength axis, so ``@when:jlens/word:membership`` raises
        ``UnsupportedProbeChannelError`` at generation preflight instead of
        sitting silently inactive.
        """
        attached = set(self._session._lens_probes)
        if not attached:
            return set()
        # Historical name for the attachment check (duck-typed stubs supply
        # a plain dict); the instrument, when present, channel-validates.
        instrument = getattr(self._session, "_lens_instrument", None)
        out: set[str] = set()
        for key, ref in self._gated_refs():
            if ref.probe in attached:
                if instrument is not None:
                    instrument.validate_gate(ref)
                out.add(key)
        return out

    def gated_sae_probe_keys(self) -> set[str]:
        """Exact gate scalar keys referencing attached SAE feature probes,
        channel-validated like the lens sibling (the SAE family produces
        only the strength axis)."""
        attached = set(self._session._sae_probes)
        if not attached:
            return set()
        instrument = getattr(self._session, "_sae_instrument", None)
        out: set[str] = set()
        for key, ref in self._gated_refs():
            if ref.probe in attached:
                if instrument is not None:
                    instrument.validate_gate(ref)
                out.add(key)
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
        flat = self.flatten_stack()
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
                value, profile_names=set(self._session._profiles),
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
        # J-lens token probes referenced by active gates score on the lens
        # path (readout-channel strength), not through the Monitor —
        # detected once per generation here, merged into every return below.
        monitor_gate_keys = self.gated_probe_keys()
        lens_gate_keys = self.gated_lens_probe_keys()
        sae_gate_keys = self.gated_sae_probe_keys()
        has_lens_gates = bool(lens_gate_keys)
        has_sae_gates = bool(sae_gate_keys)
        monitor_gate_plan_cache: dict[
            tuple[frozenset[str], frozenset[str] | None],
            Any,
        ] = {}

        def _score_monitor_gate_keys(
            latest: dict[int, torch.Tensor],
            gate_keys: set[str],
            *,
            probe_names: set[str] | None = None,
        ) -> dict[str, float]:
            if not gate_keys:
                return {}
            frozen_keys = frozenset(gate_keys)
            frozen_probes = (
                frozenset(probe_names) if probe_names is not None else None
            )
            cache_key = (frozen_keys, frozen_probes)
            plan = monitor_gate_plan_cache.get(cache_key)
            if plan is None:
                plan = monitor.plan_gate_scalars(
                    set(frozen_keys),
                    probe_names=(
                        set(frozen_probes) if frozen_probes is not None else None
                    ),
                )
                monitor_gate_plan_cache[cache_key] = plan
            return (
                cast(dict[str, float], monitor.score_planned_gate_scalars(latest, plan))
                if plan else {}
            )

        def _monitor_scalars() -> dict[str, float]:
            incremental_readings = session._incremental_readings
            incremental_gate_scores = session._incremental_gate_scores
            # The step sink already scored this token's readings — reuse them so
            # the gate doesn't trigger a second pass.  In gating-only-subset mode
            # (FIX #4) the rows hold just the gated probes, which is exactly the
            # set the gate consults; in the full incremental mode they hold the
            # whole roster.  Both reuse the latest appended row.
            state = session._capture_state
            gating_subset = state.gating_subset
            gate_keys = state.gating_keys or monitor_gate_keys
            if (
                not gate_keys
                and (has_lens_gates or has_sae_gates)
            ):
                return {}
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
                                _score_monitor_gate_keys(
                                    latest,
                                    missing,
                                    probe_names=gating_subset if gating_subset else None,
                                )
                            )
                return scalars
            if not monitor.probe_names:
                return {}
            latest = capture.latest_per_layer()
            if not latest:
                return {}
            if gate_keys:
                return _score_monitor_gate_keys(
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

        def _score() -> dict[str, float]:
            out = _monitor_scalars()
            if has_lens_gates:
                # Once per forward: band lens logits → strength
                # scalars (also stashed for the display step to reuse).
                lens_scalars = session._score_lens_gate_scalars(lens_gate_keys)
                if lens_scalars:
                    out = {**out, **lens_scalars}
            if has_sae_gates:
                sae_scalars = session._score_sae_gate_scalars(sae_gate_keys)
                if sae_scalars:
                    out = {**out, **sae_scalars}
            return out

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
        neutral_means = session.layer_means

        # Whitened push normalization (Mahalanobis-only): hand the synthesizer the
        # session whitener so the per-layer ``along`` budget is the whitened
        # displacement ``‖Δ‖_M`` and the target is a whitened-unit direction —
        # making ``along`` a scale-stable strength knob instead of inheriting each
        # node's raw-Euclidean distance from neutral (which spans ~100× across
        # targets).
        whitener = session.whitener

        # trigger -> {
        #   "push": [(basis_dirs, coord_dirs, coeff)],
        #   "ablate": [(dirs, coeff)],
        # }
        #
        # Ablation coefficients must remain attached to their directions all
        # the way into synthesis.  Dropping them here silently turned every
        # partial ablation (``0.15 !x``) into a full ablation.
        grouped: dict[Trigger, dict[str, list[Any]]] = {}

        def _bucket(trigger: Trigger) -> dict[str, list[Any]]:
            return grouped.setdefault(trigger, {"push": [], "ablate": []})

        from saklas.core.capture import fold_directions_to_subspace

        for name, entry in entries.items():
            if isinstance(entry, AblationTerm):
                ablate_prof = self.ensure_profile_registered(
                    entry.target, role="ablation target",
                )
                ablate_dirs = {
                    L: v.to(torch.float32).reshape(-1)
                    for L, v in ablate_prof.items()
                }
                _bucket(entry.trigger)["ablate"].append(
                    (ablate_dirs, entry.coeff),
                )
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
            folded = fold_directions_to_subspace(
                name, prof, neutral_means, whitener=whitener,
            )
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

        Compiled accelerator fast path: a static-affine pure-push steering lowers to
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
        if (
            session._compiled
            and session._device.type in {"cuda", "mps"}
            and steering.has_compiled_offsets()
            and (
                not session._monitor.probe_names
                or session._compiled_clean_eligible
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
        flat = self.flatten_stack()
        if not flat:
            self._session._steering.clear_all()
            return
        self.compose_steering_entries(flat)
        self.install_composed_steering()

    def snapshot_steering_alphas(self) -> dict[str, float]:
        """Flatten the active steering stack for result steering receipts.

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
        for name, entry in self.flatten_stack().items():
            if isinstance(entry, (AblationTerm, ManifoldTerm)):
                snap[name] = entry.coeff
                continue
            snap[name] = entry[0]
        return snap
