"""Steering / probe extraction controller for the saklas TUI.

Holds the TUI-local steering state — per-vector alphas, enabled flags, and
manifold terms — that the :class:`~saklas.core.session.SaklasSession` does
not own, plus every slash-command handler that mutates it (``/steer``,
``/probe``, ``/extract``, ``/manifold``, ``/pairs``, ``/alpha``,
``/unsteer``, ``/unprobe``, ``/compare``, and the ``ns/`` bulk forms).

This is a plain controller composed onto :class:`~saklas.tui.app.SaklasApp`
via a back-reference (``self._app``).  The framework-dispatched surface
(slash-command registry entries, key bindings, actions) stays on the App as
one-line forwarders into this class; the App also retains the generation /
poll orchestration these handlers hang off (``_run_worker_with_queue``,
``_steer_status``, ``_refresh_left_panel``, ``_on_vector_extracted``,
``_refresh_probe_panels``, …).  The controller reaches the session, widgets,
and those App helpers through ``self._app``.
"""

from __future__ import annotations

import shlex
from typing import Any, Callable, TYPE_CHECKING

from saklas.io.selectors import AmbiguousSelectorError, canonicalize_atom
from saklas.core.errors import SaklasError
from saklas.tui.chat_panel import PendingItem
from saklas.tui.vector_panel import MAX_ALPHA
from saklas.tui.app import (
    DEFAULT_ALPHA,
    _detect_namespace_selector,
    _resolve_active_name,
    _unquote,
)

if TYPE_CHECKING:
    from saklas.tui.app import SaklasApp


class ExtractionController:
    """Owns TUI-local steering state and the handlers that mutate it."""

    def __init__(self, app: "SaklasApp") -> None:
        self._app = app
        # Local steering state — alphas and enabled flags per vector.
        # Session holds the profiles; the TUI holds the alphas.
        self._alphas: dict[str, float] = {}
        self._enabled: dict[str, bool] = {}
        # Manifold steering terms, keyed by the grammar's manifold key
        # (``<manifold>%<coords>``).  A ``ManifoldTerm`` is not a
        # :class:`Profile`, so it can't live in ``_alphas`` (typed
        # ``dict[str, float]``) — this parallel dict carries the term
        # values and is merged into the built ``Steering.alphas`` at
        # generation time (``AlphaEntry`` already admits ``ManifoldTerm``,
        # so the engine needs no change).  ``_enabled`` is shared — a
        # manifold key toggles through the same left-panel control.
        self._manifold_terms: dict[str, Any] = {}

    def _active_alphas(self) -> dict[str, Any]:
        """Build the alphas dict for generation from enabled entries.

        Merges plain scalar vectors (``_alphas``) with manifold terms
        (``_manifold_terms``); both share the ``_enabled`` flag map so a
        manifold row toggles through the same left-panel control.  The
        returned dict's values are ``float`` for vectors and
        :class:`~saklas.core.steering_expr.ManifoldTerm` for manifolds —
        ``Steering.alphas``'s ``AlphaEntry`` union admits both.
        """
        out: dict[str, Any] = {
            name: alpha for name, alpha in self._alphas.items()
            if self._enabled.get(name, True)
        }
        for key, term in self._manifold_terms.items():
            if self._enabled.get(key, True):
                out[key] = term
        return out

    def _vector_list_for_panel(self) -> list[dict[str, Any]]:
        """Build the list[dict] format the left panel expects.

        Each entry carries a ``kind`` discriminator: ``"vector"`` rows
        carry an alpha bar; ``"manifold"`` rows carry a fixed position
        and a blend coefficient instead (no scalar alpha to nudge).
        """
        result = []
        for name, alpha in self._alphas.items():
            profile = self._app._session.profiles.get(name)
            if profile is None:
                continue
            # Bind ``profile`` to a default-arg so pyright keeps the
            # narrowed-non-None type across the lambda capture (it
            # otherwise widens to include None inside the closure).
            result.append({
                "kind": "vector",
                "name": name,
                "profile": profile,
                "alpha": alpha,
                "enabled": self._enabled.get(name, True),
                "peak": max(
                    profile,
                    key=lambda k, p=profile: float(p[k].norm().item()),
                ),
                "n_active": len(profile),
            })
        for key, term in self._manifold_terms.items():
            coords_str = ",".join(f"{c:g}" for c in term.position)
            result.append({
                "kind": "manifold",
                "name": key,
                "manifold": term.manifold,
                "coords": coords_str,
                "blend": term.coeff,
                "enabled": self._enabled.get(key, True),
            })
        return result

    @staticmethod
    def _peel_role_flag(text: str) -> tuple[str, str | None]:
        """Peel a trailing ``--role <slug>`` off a slash-command argument
        string.  Returns ``(rest, role)`` — ``role`` is ``None`` when the
        flag is absent.

        Accepted forms:
            ``... --role pirate``
            ``... --role=pirate``

        Anywhere on the line is fine; first match wins, the flag and
        its value are excised, the rest is re-joined with a single
        space.  Slug validation is left to ``core.role_templates`` —
        we just route the raw string.  Multi-word slugs are not
        supported (the engine slug regex is ``[a-z0-9._-]+``).
        """
        import re

        m = re.search(r"\s*--role(?:=|\s+)(\S+)", text)
        if m is None:
            return text, None
        head = text[: m.start()].rstrip()
        tail = text[m.end():].lstrip()
        rest = (head + (" " + tail if tail else "")).strip()
        return rest, m.group(1)

    def _handle_extract(self, text: str, include_alpha: bool,
                        on_success: "Callable[..., Any]",
                        pending_type: str | None = None,
                        variant: str = "raw",
                        namespace: str | None = None) -> None:
        chat = self._app._chat_panel
        if self._app._ab_shadow_active:
            chat.add_system_message("Cannot modify vectors during A/B shadow gen.")
            return
        if pending_type is None:
            pending_type = "steer" if include_alpha else "probe"
        if self._app._is_busy:
            # Reconstruct the canonical slash-command form so pulling
            # the item back via ↑ surfaces something the user can
            # re-Enter as a slash command.  ``payload[0]`` carries the
            # raw args the dispatcher hands to the handler.
            display_text = f"/{pending_type} {text}".rstrip()
            self._app._enqueue_pending(
                PendingItem(pending_type, display_text, (text,))
            )
            return
        # Peel ``--role <slug>`` off the args before the bipolar parser
        # runs, so a multi-word pole (``a dog . a pair of cats``) doesn't
        # have to compete with the trailing flag for tokens.
        text, role = self._peel_role_flag(text)
        try:
            if include_alpha:
                concept, baseline, alpha = self._app._parse_args(text, include_alpha=True)
            else:
                concept, baseline = self._app._parse_args(text)
                alpha = None
        except (ValueError, IndexError) as e:
            chat.add_system_message(
                f"Parse error: {e}\n"
                f"Usage: /{pending_type} <pos> . <neg>"
                + (" [alpha]" if include_alpha else "")
                + " [--role <slug>]"
            )
            return

        # Canonicalize + peel any ``:variant`` suffix typed directly on the
        # concept (``honest:sae-gemma-scope...``) — that explicit form wins
        # over the ``variant`` kwarg when both are set.  The bipolar-pole
        # sign-flip is retired (a bare pole resolves through the manifold
        # tier as a ``%`` push, not a negated vector), so canonicalization is
        # pure slug + variant peel now.  Explicit bipolar form
        # (`concept - baseline`) skips it so the user's declared poles win.
        if baseline is None:
            try:
                resolved_name, explicit_variant = canonicalize_atom(concept)
                if resolved_name != concept:
                    chat.add_system_message(
                        f"  Resolved '{concept}' → '{resolved_name}'"
                    )
                concept = resolved_name
                # Explicit ``:sae-<release>`` on the concept overrides the
                # ``--sae`` preamble's variant. Lets users route a specific
                # release without the fuzzy release-detection heuristic.
                if explicit_variant != "raw":
                    variant = explicit_variant
            except AmbiguousSelectorError as e:
                chat.add_system_message(f"Error: {e.user_message()[1]}")
                return

        # Variant routing. ``--sae`` alone (variant == "sae") means "pick the
        # unique already-extracted SAE tensor for this concept on disk" —
        # session autoload handles it. To drive a fresh extraction, users
        # pass the explicit ``:sae-<release>`` suffix, which routes the
        # release through ``session.extract(sae=RELEASE)``.  Same shape
        # for role-augmented extraction via the ``:role-<slug>`` variant
        # — the variant tail wins over an explicit ``--role`` flag (the
        # variant rode through ``canonicalize_atom``'s ``explicit_variant``
        # arm and is the canonical form).
        sae_release: str | None = None
        if variant.startswith("sae-"):
            sae_release = variant[len("sae-"):]
        if variant.startswith("role-"):
            role = variant[len("role-"):]

        display = concept if len(concept) <= 20 else concept[:17] + "..."
        suffix = f" vs '{baseline}'" if baseline else ""
        variant_note = f" [{variant}]" if variant != "raw" else ""
        role_note = f" as :role-{role}" if role else ""
        chat.add_system_message(
            f"Extracting '{display}'{suffix}{variant_note}{role_note}..."
        )

        def _work() -> None:
            def _progress(msg: str) -> None:
                self._app.call_from_thread(self._app._steer_status, msg)
            # Bare ``--sae`` (variant == "sae") routes the load through
            # the unified profile resolver (manifold-fold) rather than a
            # fresh extract — it means "use the SAE variant that's already
            # on disk". Ambiguous / missing cases surface via the session
            # errors / the None-check below.
            if variant == "sae" and sae_release is None:
                from contextlib import suppress as _suppress
                autoload_key = (
                    concept if namespace is None
                    else f"{namespace}/{concept}"
                )
                key = f"{autoload_key}:sae"
                with _suppress(Exception):
                    self._app._session.ensure_profile_registered(key)
                profile_dict = self._app._session.profiles.get(key)
                if profile_dict is None:
                    raise ValueError(
                        f"no SAE variant loaded for '{autoload_key}' — "
                        f"run `saklas subspace extract {autoload_key} --sae <RELEASE>` "
                        f"first, or pick a release with "
                        f"`:sae-<release>` in the concept name."
                    )
                on_success(key, profile_dict, alpha)
                return

            extract_kwargs = {"baseline": baseline, "on_progress": _progress}
            if sae_release is not None:
                extract_kwargs["sae"] = sae_release
            if namespace is not None:
                extract_kwargs["namespace"] = namespace
            if role is not None:
                extract_kwargs["role"] = role
            # ``session.extract`` already returns the fully-qualified
            # canonical name — including the ``:sae-<release>`` suffix
            # when ``sae=`` was passed. Rebuilding it here would
            # double-suffix the key and break every downstream
            # ``/alpha`` / ``/unsteer`` / pole lookup.
            canonical, profile = self._app._session.extract(concept, **extract_kwargs)
            if namespace is not None:
                # Re-attach the namespace so the registered key matches
                # what the parser produced (so ``/alpha`` / ``/unsteer``
                # against the namespace-qualified form keep working).
                if ":" in canonical:
                    bare, suffix = canonical.rsplit(":", 1)
                    canonical = f"{namespace}/{bare}:{suffix}"
                else:
                    canonical = f"{namespace}/{canonical}"
            on_success(canonical, profile, alpha)

        self._app._run_worker_with_queue(_work)

    def _handle_steer(self, text: str) -> None:
        """Apply a steering expression — the shared grammar from
        :mod:`saklas.core.steering_expr`.

        Each plain term (``<coeff> <concept>`` with optional ``@trigger``
        and ``:variant``) updates the TUI's local alpha state. Concepts
        not yet registered are extracted + steered behind the scenes.
        Extraction-on-demand for a *new* bipolar pair (``pos . neg``
        space-delimited) lives on ``/extract``, not ``/steer``; projection
        terms are accepted and routed through session-level materialization.
        """
        from saklas.core.steering_expr import (
            AblationTerm, ManifoldTerm, ProjectedTerm, SteeringExprError,
            parse_expr,
        )

        chat = self._app._chat_panel
        text = text.strip()
        if not text:
            chat.add_system_message(
                'Usage: /steer <expression>\n'
                '  e.g. /steer 0.5 honest\n'
                '       /steer 0.3 warm@after\n'
                '       /steer 0.5 honest|sycophantic\n'
                '       /steer alice/                (bulk; default-off)\n'
                "  For new concept extraction use /extract <pos> <neg>."
            )
            return
        ns = _detect_namespace_selector(text)
        if ns is not None:
            self._handle_steer_namespace(ns)
            return
        try:
            steering = parse_expr(text)
        except SteeringExprError as e:
            chat.add_system_message(f"Steering expression error: {e.user_message()[1]}")
            return
        except SaklasError as e:
            # ``parse_expr`` calls ``resolve_bare_atom`` per term, which raises
            # ``AmbiguousSelectorError`` (a ``SelectorError(ValueError,
            # SaklasError)``) on cross-namespace bare-pole collisions —
            # not caught by the ``SteeringExprError`` arm above. Same for
            # ``AmbiguousVariantError`` from ``:sae`` resolution. Surface
            # cleanly instead of crashing the Textual worker.
            chat.add_system_message(f"Error: {e.user_message()[1]}")
            return

        # Iterate the parsed IR; for each term dispatch through the
        # existing extract pipeline to load or compute profiles, then
        # stash the effective alpha on the TUI's local state.
        for key, val in steering.alphas.items():
            if isinstance(val, ProjectedTerm):
                chat.add_system_message(
                    f"Projection terms aren't yet supported from /steer "
                    f"(got '{key}'); express them in the YAML config."
                )
                return
            if isinstance(val, AblationTerm):
                # Ablation terms (``!x``) are dispatched through a
                # separate hook path than additive steering; the TUI's
                # `/steer` surface only wires the additive side.  Reject
                # cleanly with the same pattern as `ProjectedTerm` above
                # rather than crash on ``float(AblationTerm)`` below.
                chat.add_system_message(
                    f"Ablation terms (!{key.lstrip('!')}) aren't supported "
                    f"from /steer; express them in the YAML config."
                )
                return
            if isinstance(val, ManifoldTerm):
                # Manifold steering (``manifold%coords``) routes through a
                # separate hook path — not bound to the per-vector alpha
                # sliders.  Dispatch through a worker that eager-loads and
                # validates the artifact, then stash it on
                # ``_manifold_terms`` (a ``ManifoldTerm`` is not a
                # ``Profile``, so it can't live in the scalar ``_alphas``).
                # ``continue`` rather than ``return`` so a mixed
                # expression still applies its vector siblings.
                self._dispatch_manifold_term(key, val)
                continue
            if isinstance(val, tuple):
                alpha, _trig = val
            else:
                alpha = float(val)
            alpha = max(-MAX_ALPHA, min(MAX_ALPHA, float(alpha)))
            # Peel variant suffix so the extract path sees a bare concept.
            if ":" in key:
                concept, variant = key.rsplit(":", 1)
            else:
                concept, variant = key, "raw"
            # Peel namespace prefix so ``_handle_extract`` ->
            # ``canonicalize_atom`` slugs the bare tail instead of slugging
            # ``ns/name`` into a single token.  ``_handle_extract`` will
            # use ``namespace`` as the kwarg into ``session.extract`` so
            # disk discovery stays scoped too.
            namespace: str | None = None
            if "/" in concept:
                namespace, concept = concept.split("/", 1)
            self._dispatch_steer_term(
                concept, variant, alpha, namespace=namespace,
            )

    def _dispatch_steer_term(
        self, concept: str, variant: str, alpha: float,
        *, namespace: str | None = None,
    ) -> None:
        """Route one plain steering term through the extract pipeline.

        The concept has already been canonicalized by ``parse_expr``;
        ``_handle_extract`` will re-run ``canonicalize_atom`` on the
        canonical form, which is idempotent.  ``namespace`` (when set)
        scopes the downstream ``session.extract`` call so ``alice/foo`` and
        ``bob/foo`` stay distinct end-to-end.
        """
        def _on_success(name: str, profile: Any, a: float) -> None:
            self._app._session.steer(name, profile)
            self._alphas[name] = a
            self._enabled[name] = True
            self._app.call_from_thread(self._app._on_vector_extracted, name, a, profile)
        # _handle_extract's own parser expects a "<concept> <alpha>" text;
        # embed the variant in the concept token so canonicalize_atom strips it.
        concept_with_variant = (
            concept if variant == "raw" else f"{concept}:{variant}"
        )
        text = f"{concept_with_variant} {alpha}"
        self._handle_extract(
            text, include_alpha=True, on_success=_on_success,
            variant=variant, namespace=namespace,
        )

    def _dispatch_manifold_term(self, key: str, term: Any) -> None:
        """Route one manifold term through eager artifact validation.

        ``key`` is the grammar's manifold registry key
        (``<manifold>%<coords>``); ``term`` is the parsed
        :class:`~saklas.core.steering_expr.ManifoldTerm`.  A worker
        eager-loads the artifact via ``session.ensure_manifold_loaded``
        — which raises :class:`ManifoldNotRegisteredError` when the named
        manifold has no fitted tensor for the loaded model — and then
        validates the position-tuple arity against the loaded manifold's
        domain.  On success the term lands in ``_manifold_terms`` and the
        left panel refreshes; any failure surfaces as a system message.
        """
        chat = self._app._chat_panel
        if self._app._ab_shadow_active:
            chat.add_system_message(
                "Cannot modify steering during A/B shadow gen."
            )
            return
        # Position can be a node-label string (``persona%pirate``) or a
        # coord tuple (``persona%0.3,0.8``); display each in its
        # natural form.  Arity is only meaningful in coord form — the
        # label form resolves to whatever coords the node carries.
        if isinstance(term.position, str):
            pos_str = term.position
        else:
            pos_str = ",".join(f"{c:g}" for c in term.position)
        chat.add_system_message(
            f"Loading manifold '{term.manifold}' % {pos_str}..."
        )

        def _work() -> None:
            self._app._session.ensure_manifold_loaded(term.manifold)
            manifold = self._app._session.manifolds[term.manifold]
            # ``resolve_position`` validates label-form (raises
            # UnknownManifoldLabelError on miss) and passes through
            # coord-form unchanged.  A test-double manifold without
            # ``resolve_position`` falls back to ``term.position``
            # directly — fine for coord form, raises here for
            # label form (the real engine path catches it).
            resolve = getattr(manifold, "resolve_position", None)
            if resolve is not None:
                resolved = resolve(term.position)
            elif isinstance(term.position, str):
                raise ValueError(
                    f"manifold '{term.manifold}' cannot resolve "
                    f"label {term.position!r} — manifold lacks "
                    f"resolve_position()"
                )
            else:
                resolved = term.position
            want = manifold.domain.intrinsic_dim
            got = len(resolved)
            if got != want:
                raise ValueError(
                    f"manifold '{term.manifold}' is {want}-dimensional "
                    f"but the position has {got} coordinate(s)"
                )
            self._app.call_from_thread(self._app._on_manifold_added, key, term)

        self._app._run_worker_with_queue(_work)

    def _handle_probe(self, text: str) -> None:
        ns = _detect_namespace_selector(text.strip())
        if ns is not None:
            self._handle_probe_namespace(ns)
            return

        def _on_success(name: str, _profile: Any, _alpha: Any) -> None:
            # ``_handle_extract`` already registered the folded direction in
            # ``session.profiles[name]``; ``add_probe`` resolves it from
            # there (folding the 2-node concept to a rank-1 probe) and
            # attaches it to the unified monitor.
            self._app._session.add_probe(name)
            self._app.call_from_thread(self._app._on_probe_added, name)
        self._handle_extract(text, include_alpha=False, on_success=_on_success)

    def _handle_extract_only(self, text: str) -> None:
        def _on_success(name: str, _profile: Any, _alpha: Any) -> None:
            # Pure cache-warm: no steering, no probe, no panel state.
            self._app.call_from_thread(
                self._app._steer_status, f"extracted '{name}'"
            )
        self._handle_extract(
            text, include_alpha=False, on_success=_on_success,
            pending_type="extract",
        )

    def _handle_manifold(self, text: str) -> None:
        """`/manifold fit <folder>` — fit a steering manifold.

        Manifold *authoring* (writing ``manifold.json`` + ``nodes/*.json``)
        stays out of the TUI; this command only fits an already-authored
        manifold pack folder for the loaded model.  Gated like
        ``/extract`` — refused during an A/B shadow gen, deferred behind
        an in-flight generation via the pending queue.
        """
        chat = self._app._chat_panel
        text = (text or "").strip()
        parts = text.split(maxsplit=1)
        verb = parts[0].lower() if parts else ""
        if verb != "fit":
            chat.add_system_message("Usage: /manifold fit <folder>")
            return
        folder_arg = parts[1].strip() if len(parts) > 1 else ""
        folder_arg = _unquote(folder_arg)
        if not folder_arg:
            chat.add_system_message("Usage: /manifold fit <folder>")
            return
        if self._app._ab_shadow_active:
            chat.add_system_message(
                "Cannot fit a manifold during A/B shadow gen."
            )
            return
        if self._app._is_busy:
            self._app._enqueue_pending(
                PendingItem(
                    "manifold_fit", f"/manifold fit {folder_arg}",
                    (folder_arg,),
                )
            )
            return
        self._start_manifold_fit(folder_arg)

    def _start_manifold_fit(self, folder_arg: str) -> None:
        """Run ``session.fit`` on a worker thread.

        Mirrors ``_handle_extract``'s worker structure: progress lines
        stream to the chat pane, errors surface as system messages, and
        the ``finally`` block enqueues a ``done`` sentinel so the pending
        queue keeps draining (manifold fitting runs off the gen loop).
        """
        from pathlib import Path

        folder = Path(folder_arg).expanduser()
        chat = self._app._chat_panel
        if not (folder / "manifold.json").exists():
            chat.add_system_message(
                f"/manifold fit: no manifold.json in {folder}"
            )
            self._app._ui_token_queue.put(("done", False))
            return
        chat.add_system_message(f"Fitting manifold from {folder}...")

        def _work() -> None:
            def _progress(msg: str) -> None:
                self._app.call_from_thread(self._app._steer_status, msg)
            # ``fit_rbf_interpolant`` raises a bare ``ValueError`` on
            # poisedness failure — not a ``SaklasError``; the shared
            # worker wrapper surfaces it as a plain string.
            manifold = self._app._session.fit(
                folder, on_progress=_progress,
            )
            self._app.call_from_thread(
                self._app._steer_status,
                f"fitted manifold '{manifold.name}' "
                f"({len(manifold.layers)}L, "
                f"{len(manifold.node_labels)} nodes) — "
                f"steer it with `/steer <coeff> {manifold.name}%<coords>`",
            )

        self._app._run_worker_with_queue(_work)

    def _handle_manifold_probe(self, text: str) -> None:
        """``/manifold-probe <selector>`` — attach a curved manifold probe.

        Routes through the unified ``session.add_probe`` — the selector can
        be a bundled name (``emotions``, ``personas``), an ``ns/name`` form, or an
        already-loaded manifold's registered name; the session's lazy-load
        path (``_ensure_manifold_loaded``) handles resolution.  This is the
        curved-probe front-end (the TUI mirror of the webui's "+ manifold
        probe" launcher); ``/probe`` is the flat/concept front-end.  Both
        land in the one monitor.  Refused
        during A/B shadow gen for the same reason ``/probe`` is —
        modifying the monitor set mid-shadow would interleave readings
        across the steered/unsteered split.  Deferred behind an
        in-flight gen via the pending queue (kind ``manifold_probe``).
        """
        chat = self._app._chat_panel
        selector = (text or "").strip()
        selector = _unquote(selector)
        if not selector:
            chat.add_system_message("Usage: /manifold-probe <selector>")
            return
        if self._app._ab_shadow_active:
            chat.add_system_message(
                "Cannot attach a manifold probe during A/B shadow gen."
            )
            return
        if self._app._is_busy:
            self._app._enqueue_pending(
                PendingItem(
                    "manifold_probe", f"/manifold-probe {selector}",
                    (selector,),
                )
            )
            return
        self._start_manifold_probe_attach(selector)

    def _start_manifold_probe_attach(self, selector: str) -> None:
        """Run ``session.add_probe`` on a worker thread.

        Mirrors ``_start_manifold_fit``'s worker structure — errors
        surface as system messages and a ``done`` sentinel re-enters
        the queue drain so a queued attach doesn't stall subsequent
        items.
        """
        chat = self._app._chat_panel
        chat.add_system_message(f"Attaching manifold probe '{selector}'...")

        def _work() -> None:
            name = self._app._session.add_probe(selector)
            self._app.call_from_thread(self._app._on_manifold_probe_added, name)

        self._app._run_worker_with_queue(_work)

    def _handle_manifold_probe_remove(self, text: str) -> None:
        """``/manifold-probe-remove <name>`` — detach an attached probe.

        After the monitor unification there is one probe set, so this and
        ``/unprobe`` are interchangeable; the command is kept for muscle
        memory and detaches any probe shape via ``session.remove_probe``.
        """
        chat = self._app._chat_panel
        name = (text or "").strip()
        name = _unquote(name)
        if not name:
            chat.add_system_message("Usage: /manifold-probe-remove <name>")
            return
        monitor = self._app._session.monitor
        if monitor is None or name not in monitor.probe_names:
            chat.add_system_message(f"Manifold probe '{name}' not active.")
            return
        try:
            self._app._session.remove_probe(name)
        except SaklasError as e:
            chat.add_system_message(e.user_message()[1])
            return
        except Exception as e:
            chat.add_system_message(f"{type(e).__name__}: {e}")
            return
        self._app._refresh_probe_panels()
        self._app._refresh_trait_why()
        chat.add_system_message(f"Manifold probe '{name}' removed.")

    def _handle_pairs(self, text: str) -> None:
        """`/pairs <name>` — open the custom-statement extraction modal.

        Opens :class:`~saklas.tui.pairs_modal.CustomPairsModal`, a
        multi-line editor for hand-authored ``positive | negative``
        contrastive pairs.  On submit the lines are parsed into a pairs
        list and handed straight to ``session.extract`` — bypassing
        scenario / pair generation.

        A modal cannot be a :class:`PendingItem`, so ``/pairs`` is
        refused mid-generation rather than queued (matching how other
        modal-requiring commands behave).
        """
        chat = self._app._chat_panel
        raw_args = (text or "").strip()
        raw_args, role = self._peel_role_flag(raw_args)
        name = _unquote(raw_args)
        if not name:
            chat.add_system_message("Usage: /pairs <name> [--role <slug>]")
            return
        if self._app._ab_shadow_active:
            chat.add_system_message(
                "Cannot extract a vector during A/B shadow gen."
            )
            return
        if self._app._is_busy:
            chat.add_system_message(
                "/pairs opens a modal — finish or stop the current "
                "generation first."
            )
            return

        from saklas.tui.pairs_modal import CustomPairsModal

        def _on_submit(pairs: list[tuple[str, str]] | None) -> None:
            if not pairs:
                return  # modal cancelled
            self._start_pairs_extract(name, pairs, role=role)

        self._app.push_screen(CustomPairsModal(name), _on_submit)

    def _start_pairs_extract(
        self, name: str, pairs: list[tuple[str, str]],
        *, role: str | None = None,
    ) -> None:
        """Extract a steering vector from hand-authored contrastive pairs.

        ``session.extract`` accepts a list of ``(positive, negative)``
        tuples directly as ``source``, skipping scenario / pair
        generation.  The extracted vector is steered at
        :data:`DEFAULT_ALPHA` and registered on the left panel, mirroring
        ``/steer``'s post-extraction wiring.
        """
        chat = self._app._chat_panel
        role_note = f" as :role-{role}" if role else ""
        chat.add_system_message(
            f"Extracting '{name}'{role_note} from {len(pairs)} custom pair(s)..."
        )

        def _on_success(canonical: str, profile: Any, alpha: float) -> None:
            self._app._session.steer(canonical, profile)
            self._alphas[canonical] = alpha
            self._enabled[canonical] = True
            self._app.call_from_thread(
                self._app._on_vector_extracted, canonical, alpha, profile,
            )

        def _worker() -> None:
            def _progress(msg: str) -> None:
                self._app.call_from_thread(self._app._steer_status, msg)
            try:
                # Hand-authored contrastive examples become the two pole
                # corpora of a 2-node ``pca`` manifold — positive pole vs its
                # opposite — fit directly (no scenario / pair generation).
                positive = [pos for pos, _ in pairs]
                negative = [neg for _, neg in pairs]
                extract_kwargs: dict[str, Any] = {
                    "on_progress": _progress, "namespace": "local",
                }
                if role is not None:
                    extract_kwargs["role"] = role
                canonical, profile = self._app._session.extract_vector_from_corpora(
                    name, positive, negative, **extract_kwargs,
                )
                _on_success(canonical, profile, DEFAULT_ALPHA)
            except SaklasError as e:
                self._app.call_from_thread(self._app._steer_status, e.user_message()[1])
            except (ValueError, TypeError) as e:
                self._app.call_from_thread(self._app._steer_status, str(e))
            except Exception as e:
                self._app.call_from_thread(
                    self._app._steer_status, f"{type(e).__name__}: {e}",
                )
            finally:
                self._app._ui_token_queue.put(("done", False))

        self._app.run_worker(_worker, thread=True)

    def _handle_alpha(self, arg: str) -> None:
        chat = self._app._chat_panel
        try:
            tokens = shlex.split(arg)
        except ValueError as e:
            chat.add_system_message(f"Parse error: {e}")
            return
        if len(tokens) != 2:
            chat.add_system_message("Usage: /alpha <value> <name>")
            return
        val_str, raw = tokens
        matches = _resolve_active_name(raw, self._alphas)
        if len(matches) == 0:
            chat.add_system_message(
                f"'{raw}' is not active. Use /steer to add it first."
            )
            return
        if len(matches) > 1:
            chat.add_system_message(
                f"'{raw}' is ambiguous: {', '.join(matches)}"
            )
            return
        try:
            val = float(val_str)
        except ValueError:
            chat.add_system_message(f"Invalid alpha: {val_str}")
            return
        name = matches[0]
        if name != raw:
            # Sign flip when the user typed the negative pole.
            from saklas.core.session import BIPOLAR_SEP, canonical_concept_name
            slug = canonical_concept_name(raw)
            if BIPOLAR_SEP in name:
                _pos, neg = name.split(BIPOLAR_SEP, 1)
                if slug == neg:
                    val = -val
        val = max(-MAX_ALPHA, min(MAX_ALPHA, val))
        self._alphas[name] = val
        self._app._refresh_left_panel()
        chat.add_system_message(f"Alpha for '{name}' set to {val:+.2f}")

    def _handle_unsteer(self, arg: str) -> None:
        chat = self._app._chat_panel
        raw = arg.strip()
        if not raw:
            chat.add_system_message("Usage: /unsteer <name>")
            return
        ns = _detect_namespace_selector(raw)
        if ns is not None:
            self._handle_unsteer_namespace(ns)
            return
        # Resolve against both racks — scalar vectors (``_alphas``) and
        # manifold terms (``_manifold_terms``).  A ``/steer`` with a
        # ``%`` term lands in ``_manifold_terms``; without this the
        # slash command would report a racked manifold as "not active".
        matches = _resolve_active_name(
            raw, [*self._alphas.keys(), *self._manifold_terms.keys()]
        )
        if len(matches) == 0:
            chat.add_system_message(f"'{raw}' is not active.")
            return
        if len(matches) > 1:
            chat.add_system_message(f"'{raw}' is ambiguous: {', '.join(matches)}")
            return
        name = matches[0]
        if name in self._manifold_terms:
            # Manifold rows aren't session-registered profiles — drop the
            # local term and let the next gen rebuild ``Steering``
            # without it (mirrors the panel backspace/delete path).
            self._manifold_terms.pop(name, None)
            self._enabled.pop(name, None)
            self._app._refresh_left_panel()
            chat.add_system_message(f"Removed manifold '{name}'.")
            return
        self._app._session.unsteer(name)
        self._alphas.pop(name, None)
        self._enabled.pop(name, None)
        self._app._refresh_left_panel()
        chat.add_system_message(f"Removed '{name}'.")

    def _handle_unprobe(self, arg: str) -> None:
        chat = self._app._chat_panel
        raw = arg.strip()
        if not raw:
            chat.add_system_message("Usage: /unprobe <name>")
            return
        ns = _detect_namespace_selector(raw)
        if ns is not None:
            self._handle_unprobe_namespace(ns)
            return
        monitor = self._app._session.monitor
        if not monitor:
            chat.add_system_message(f"Probe '{raw}' not active.")
            return
        matches = _resolve_active_name(raw, monitor.probe_names)
        if len(matches) == 0:
            chat.add_system_message(f"Probe '{raw}' not active.")
            return
        if len(matches) > 1:
            chat.add_system_message(f"'{raw}' is ambiguous: {', '.join(matches)}")
            return
        name = matches[0]
        self._app._session.remove_probe(name)
        self._app._refresh_probe_panels()
        if self._app._highlight_probe == name:
            self._app._highlight_probe = None
            self._app._highlighting = False
            self._app._apply_highlight_to_all()
        self._app._refresh_trait_why()
        chat.add_system_message(f"Probe '{name}' removed.")

    def _bulk_autoload_namespace(self, ns: str) -> tuple[list[str], list[str]]:
        """Autoload every concept in ``ns`` whose tensor is on disk.

        Returns ``(loaded, skipped)`` lists of namespace-qualified names.
        ``loaded`` is what landed in ``session.profiles`` (already-present
        plus freshly loaded); ``skipped`` is concepts whose ``raw`` variant
        isn't extracted for the current model. Cache-hit only — no PCA,
        no scenario gen, no network. Worker-thread safe (only touches
        ``session.profiles`` and the on-disk pack files).
        """
        from saklas.io.selectors import _all_concepts

        loaded: list[str] = []
        skipped: list[str] = []
        concepts = [c for c in _all_concepts() if c.namespace == ns]
        for c in concepts:
            key = f"{ns}/{c.name}"
            if key in self._app._session.profiles:
                loaded.append(key)
                continue
            try:
                self._app._session.ensure_profile_registered(key)
            except SaklasError:
                # Unresolved / not-yet-fit concepts surface to the user
                # below by leaving the concept in ``skipped``; the
                # detailed message would drown out the bulk summary.
                pass
            except Exception:
                pass
            if key in self._app._session.profiles:
                loaded.append(key)
            else:
                skipped.append(key)
        return loaded, skipped

    def _bulk_skip_message(self, ns: str, skipped: list[str]) -> str:
        """Two-line note for skipped concepts: list + one-line refresh hint."""
        return (
            f"Skipped {len(skipped)} not yet extracted for this model: "
            f"{', '.join(sorted(skipped))}\n"
            f"  Run `saklas pack refresh {ns} -m <model>` to extract."
        )

    def _handle_steer_namespace(self, ns: str) -> None:
        """Bulk-register every cached concept under ``ns/`` as a steering
        vector with α = ``DEFAULT_ALPHA`` and ``enabled=False`` so users
        can flip them on individually from the left panel.
        """
        chat = self._app._chat_panel
        if self._app._ab_shadow_active:
            chat.add_system_message("Cannot modify vectors during A/B shadow gen.")
            return
        if self._app._is_busy:
            arg = f"{ns}/"
            self._app._enqueue_pending(PendingItem("steer", f"/steer {arg}", (arg,)))
            return

        from saklas.io.selectors import _all_concepts
        if not [c for c in _all_concepts() if c.namespace == ns]:
            chat.add_system_message(f"No concepts installed under '{ns}/'.")
            return

        chat.add_system_message(f"Loading '{ns}/' vectors (toggled off)...")

        def _worker() -> None:
            loaded, skipped = self._bulk_autoload_namespace(ns)

            def _finish() -> None:
                from saklas.core.profile import Profile as _Profile
                for key in loaded:
                    profile = self._app._session.profiles.get(key)
                    if profile is None:
                        continue
                    # _profiles stores dict[int, Tensor]; steer() expects Profile.
                    self._app._session.steer(key, _Profile(profile))
                    self._alphas[key] = DEFAULT_ALPHA
                    self._enabled[key] = False
                self._app._refresh_left_panel()
                lines = [
                    f"Bulk steer '{ns}/': "
                    f"added {len(loaded)} vector(s) (toggled off)."
                ]
                if skipped:
                    lines.append(self._bulk_skip_message(ns, skipped))
                chat.add_system_message("\n".join(lines))

            self._app.call_from_thread(_finish)

        self._app.run_worker(_worker, thread=True)

    def _handle_probe_namespace(self, ns: str) -> None:
        """Bulk-register every cached concept under ``ns/`` as a probe.
        Highlight seeds to the last-loaded probe so the per-token overlay
        lights up immediately, matching the single-probe ``/probe`` UX.
        """
        chat = self._app._chat_panel
        if self._app._ab_shadow_active:
            chat.add_system_message("Cannot modify vectors during A/B shadow gen.")
            return
        if self._app._is_busy:
            arg = f"{ns}/"
            self._app._enqueue_pending(PendingItem("probe", f"/probe {arg}", (arg,)))
            return

        from saklas.io.selectors import _all_concepts
        if not [c for c in _all_concepts() if c.namespace == ns]:
            chat.add_system_message(f"No concepts installed under '{ns}/'.")
            return

        chat.add_system_message(f"Loading '{ns}/' probes...")

        def _worker() -> None:
            loaded, skipped = self._bulk_autoload_namespace(ns)

            def _finish() -> None:
                for key in loaded:
                    if self._app._session.profiles.get(key) is None:
                        continue
                    # The folded direction is already in ``_profiles[key]``;
                    # ``add_probe`` resolves it from there and attaches to the
                    # unified monitor.
                    self._app._session.add_probe(key)
                if loaded and self._app._session.monitor is not None:
                    self._app._refresh_probe_panels()
                    # Seed highlight to the last loaded probe — same UX as
                    # single ``/probe``: the user immediately sees one of
                    # them lit up and can navigate the trait panel to flip.
                    self._app._highlight_probe = sorted(loaded)[-1]
                    self._app._highlighting = True
                    self._app._apply_highlight_to_all()
                    self._app._refresh_trait_why()
                lines = [f"Bulk probe '{ns}/': added {len(loaded)} probe(s)."]
                if loaded:
                    lines.append("  Highlight on (⌃Y to cycle).")
                if skipped:
                    lines.append(self._bulk_skip_message(ns, skipped))
                chat.add_system_message("\n".join(lines))

            self._app.call_from_thread(_finish)

        self._app.run_worker(_worker, thread=True)

    def _handle_unsteer_namespace(self, ns: str) -> None:
        """Remove every active steering vector whose registry key sits
        under ``ns/``. Mirrors the single-vector ``/unsteer`` no-defer
        policy — modifying ``_profiles`` mid-gen doesn't disturb hooks
        already attached to the in-flight forward pass.
        """
        chat = self._app._chat_panel
        prefix = f"{ns}/"
        matches = [n for n in list(self._alphas.keys()) if n.startswith(prefix)]
        # Manifold terms share the ``ns/name`` key shape — sweep them too.
        manifold_matches = [
            n for n in list(self._manifold_terms.keys()) if n.startswith(prefix)
        ]
        if not matches and not manifold_matches:
            chat.add_system_message(f"No active vectors under '{ns}/'.")
            return
        for name in matches:
            self._app._session.unsteer(name)
            self._alphas.pop(name, None)
            self._enabled.pop(name, None)
        for name in manifold_matches:
            self._manifold_terms.pop(name, None)
            self._enabled.pop(name, None)
        self._app._refresh_left_panel()
        total = len(matches) + len(manifold_matches)
        chat.add_system_message(
            f"Removed {total} vector(s) from '{ns}/'."
        )

    def _handle_unprobe_namespace(self, ns: str) -> None:
        """Remove every active probe whose registry key sits under ``ns/``.
        Clears the highlight seed when it points into the namespace.
        """
        chat = self._app._chat_panel
        monitor = self._app._session.monitor
        if monitor is None:
            chat.add_system_message(f"No active probes under '{ns}/'.")
            return
        prefix = f"{ns}/"
        matches = [n for n in list(monitor.probe_names) if n.startswith(prefix)]
        if not matches:
            chat.add_system_message(f"No active probes under '{ns}/'.")
            return
        for name in matches:
            self._app._session.remove_probe(name)
        self._app._refresh_probe_panels()
        if self._app._highlight_probe is not None and self._app._highlight_probe.startswith(prefix):
            self._app._highlight_probe = None
            self._app._highlighting = False
            self._app._apply_highlight_to_all()
        self._app._refresh_trait_why()
        chat.add_system_message(
            f"Removed {len(matches)} probe(s) from '{ns}/'."
        )

    def _handle_compare(self, arg: str) -> None:
        chat = self._app._chat_panel
        if not arg:
            chat.add_system_message("Usage: /compare <name> [other_name]")
            return

        parts = arg.split()

        # Gather all available profiles: session profiles + monitor probes.
        # The monitor stores each probe as a ``Manifold`` now; fold it to the
        # per-layer baked direction view (``{L: δ̂_L · share_L}``) so both
        # sources expose the same cosine_similarity API.  A multi-axis /
        # curved probe has no single direction to compare and is skipped.
        from saklas.core.profile import Profile
        all_profiles: dict[str, Profile] = {}
        for name, prof in self._app._session.profiles.items():
            if isinstance(prof, Profile):
                all_profiles[name] = prof
            elif isinstance(prof, dict) and prof:
                all_profiles[name] = Profile(prof)
        monitor = self._app._session.monitor
        if monitor:
            from saklas.core.vectors import folded_vector_directions
            for name in monitor.probe_names:
                if name in all_profiles:
                    continue
                manifold = monitor.manifolds.get(name)
                if manifold is None:
                    continue
                try:
                    folded = folded_vector_directions(manifold)
                except Exception:
                    continue
                if folded:
                    all_profiles[name] = Profile(folded)

        def _resolve(raw: str) -> str | None:
            matches = _resolve_active_name(raw, all_profiles)
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                chat.add_system_message(f"'{raw}' is ambiguous: {', '.join(matches)}")
                return None
            chat.add_system_message(f"No profile found for '{raw}'")
            return None

        if len(parts) == 1:
            # 1-arg: ranked comparison against all loaded profiles.
            target_name = _resolve(parts[0])
            if target_name is None:
                return
            target = all_profiles[target_name]
            others = {n: p for n, p in all_profiles.items() if n != target_name}
            if not others:
                chat.add_system_message("No other profiles loaded to compare against.")
                return
            # Mahalanobis-only: ``cosine_similarity`` requires the session
            # whitener (the Euclidean path is gone).  Pairs whose shared
            # layers the whitener doesn't cover raise and are skipped.
            whitener = getattr(self._app._session, "whitener", None)
            scores = {}
            for name, prof in others.items():
                try:
                    scores[name] = target.cosine_similarity(prof, whitener=whitener)
                except Exception:
                    continue
            if not scores:
                chat.add_system_message("No comparable profiles (no shared layers).")
                return
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            width = max(len(n) for n, _ in ranked)
            lines = [f"{target_name} vs loaded profiles:"]
            for name, score in ranked:
                lines.append(f"  {name:<{width}}  {score:+.4f}")
            chat.add_system_message("\n".join(lines))

        elif len(parts) == 2:
            # 2-arg: pairwise.
            a_name = _resolve(parts[0])
            if a_name is None:
                return
            b_name = _resolve(parts[1])
            if b_name is None:
                return
            # Mahalanobis-only (see the 1-arg branch above).
            whitener = getattr(self._app._session, "whitener", None)
            try:
                sim = all_profiles[a_name].cosine_similarity(
                    all_profiles[b_name], whitener=whitener,
                )
            except Exception as e:
                chat.add_system_message(f"Compare failed: {e}")
                return
            chat.add_system_message(f"{a_name} ~ {b_name}: {sim:+.4f}")

        else:
            chat.add_system_message("Usage: /compare <name> [other_name]")
