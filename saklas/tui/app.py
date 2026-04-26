"""Main Textual application for saklas."""

from __future__ import annotations

import json
import math
import queue
import shlex
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import torch
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.widgets import Input
from textual.timer import Timer

from saklas import SamplingConfig, Steering
from saklas.io.selectors import AmbiguousSelectorError, resolve_pole
from saklas.core.errors import SaklasError
from saklas.core.generation import supports_thinking
from saklas.io.paths import saklas_home
from saklas.io.probes_bootstrap import load_defaults
from saklas.core.results import ResultCollector
from saklas.core.session import MIN_ELAPSED_FOR_RATE
from saklas.tui.chat_panel import ChatPanel, _AssistantMessage
from saklas.tui.vector_panel import LeftPanel, MAX_ALPHA
from saklas.tui.trait_panel import TraitPanel

DEFAULT_ALPHA = 0.5
_POLL_FPS = 15
_TOKEN_DRAIN_LIMIT = 20

_LEFT, _CHAT, _TRAIT = 0, 1, 2

_BIPOLAR_DELIM = " . "


def _unquote(s: str) -> str:
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        return s[1:-1]
    return s


def _split_bipolar(text: str) -> tuple[str, str | None]:
    """Split ``pos . neg`` on the first surrounded-by-whitespace period.

    Whitespace around the period is required, so canonical single-token
    names like ``dog.cat`` aren't split. Quotes are stripped from each
    side if the user wrapped them.
    """
    idx = text.find(_BIPOLAR_DELIM)
    if idx >= 0:
        return (
            _unquote(text[:idx].strip()),
            _unquote(text[idx + len(_BIPOLAR_DELIM):].strip()),
        )
    return _unquote(text.strip()), None


def _resolve_active_name(name: str, active) -> list[str]:
    """Resolve a user-typed name against a set of currently-active names.

    Direct hit returns a single-element list. Otherwise treats ``name``
    as a pole and scans ``active`` for any canonical entry where the
    slug appears on either side of the ``.`` separator. Returns all
    matches (caller handles 0 / 1 / many).
    """
    from saklas.core.session import BIPOLAR_SEP, canonical_concept_name

    active = list(active)
    if name in active:
        return [name]
    slug = canonical_concept_name(name)
    matches: list[str] = []
    for key in active:
        if key == slug:
            matches.append(key)
            continue
        if BIPOLAR_SEP in key:
            pos, neg = key.split(BIPOLAR_SEP, 1)
            if pos == slug or neg == slug:
                matches.append(key)
    return matches


class SaklasApp(App):
    CSS_PATH = "styles.tcss"
    ENABLE_COMMAND_PALETTE = False

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit", show=False),
        Binding("backspace", "remove_vector", "Remove", show=False),
        Binding("delete", "remove_vector", "Remove", show=False),
        Binding("ctrl+a", "ab_compare", "A/B", show=False),
        Binding("escape", "stop_generation", "Stop", show=False),
        Binding("ctrl+r", "regenerate", "Regen", show=False),
        Binding("ctrl+c", "copy_selection", "Copy", show=False),
        Binding("ctrl+t", "toggle_thinking", "Think", show=False),
        Binding("ctrl+s", "cycle_sort", "Sort", show=False),
        Binding("ctrl+y", "toggle_highlight", "Highlight", show=False),
        Binding("[", "temp_down", show=False),
        Binding("]", "temp_up", show=False),
        Binding("{", "top_p_down", show=False),
        Binding("}", "top_p_up", show=False),
    ]

    def __init__(
        self,
        session,
        **kwargs,
    ):
        super().__init__(ansi_color=True, **kwargs)
        self._session = session
        self._messages = session._history
        self._device_str = str(session._device)

        # Local steering state — alphas and enabled flags per vector.
        # Session holds the profiles; the TUI holds the alphas.
        self._alphas: dict[str, float] = {}
        self._enabled: dict[str, bool] = {}
        self._supports_thinking: bool = supports_thinking(session._tokenizer)
        self._thinking: bool = self._supports_thinking

        self._current_assistant_widget = None
        self._poll_timer: Timer | None = None
        self._last_prompt: str | None = None
        self._ab_in_progress: bool = False
        self._pending_action: tuple | None = None  # ("regenerate",) or ("submit", text)
        self._gen_active: bool = False  # main-thread-only generation guard

        self._focused_panel_idx: int = 1  # Start with chat focused

        self._highlighting: bool = False
        self._highlight_probe: str | None = None
        self._default_seed: int | None = None
        self._ui_token_queue: queue.SimpleQueue = queue.SimpleQueue()

        self._gen_start_time: float = 0.0
        self._gen_token_count: int = 0
        self._last_tok_per_sec: float = 0.0
        self._last_elapsed: float = 0.0
        # Geometric-mean perplexity accumulator — sum of ``log(ppl)`` across
        # scored steps; display is ``exp(sum/count)``.
        self._log_ppl_sum: float = 0.0
        self._ppl_count: int = 0
        self._last_gen_state: tuple = (-1, -1.0, -1.0, False, -1)
        self._assistant_messages: list[_AssistantMessage] = []

        defaults = load_defaults()
        self._probe_categories: dict[str, list[str]] = {
            cat.capitalize(): probes_list
            for cat, probes_list in defaults.items()
        }

    def _active_alphas(self) -> dict[str, float]:
        """Build alphas dict for generation from enabled vectors."""
        return {name: alpha for name, alpha in self._alphas.items()
                if self._enabled.get(name, True)}

    def _vector_list_for_panel(self) -> list[dict]:
        """Build the list[dict] format the left panel expects."""
        result = []
        for name, alpha in self._alphas.items():
            profile = self._session._profiles.get(name)
            if profile is None:
                continue
            result.append({
                "name": name,
                "profile": profile,
                "alpha": alpha,
                "enabled": self._enabled.get(name, True),
                "peak": max(profile, key=lambda k: float(profile[k].norm().item())),
                "n_active": len(profile),
            })
        return result

    def compose(self) -> ComposeResult:
        with Horizontal(id="main-area"):
            yield LeftPanel(self._session._model_info, id="left-panel")
            yield ChatPanel(id="chat-panel")
            yield TraitPanel(categories=self._probe_categories, id="trait-panel")

    def on_mount(self) -> None:
        self._left_panel = self.query_one("#left-panel", LeftPanel)
        self._chat_panel = self.query_one("#chat-panel", ChatPanel)
        self._trait_panel = self.query_one("#trait-panel", TraitPanel)
        self._panels = [self._left_panel, self._chat_panel, self._trait_panel]
        self._refresh_gen_config()

        if self._session._monitor:
            self._trait_panel.set_active_probes(set(self._session._monitor.probe_names))
            self._refresh_trait_why()

        self._poll_timer = self.set_interval(1 / _POLL_FPS, self._poll_generation)
        self._update_panel_focus()

        self._chat_panel.add_system_message(
            f"Model loaded: {self._session._model_info.get('model_id', 'unknown')}. "
            f"Type a message to chat. Use /steer and /probe commands. Tab to switch panels."
        )

    # -- Key Handling --

    def on_key(self, event) -> None:
        if isinstance(self.focused, Input):
            if event.key == "tab":
                event.prevent_default()
                event.stop()
                self.action_focus_next_panel()
            elif event.key == "shift+tab":
                event.prevent_default()
                event.stop()
                self.action_focus_prev_panel()
            return

        key = event.key
        handled = True
        if key == "tab":
            self.action_focus_next_panel()
        elif key == "shift+tab":
            self.action_focus_prev_panel()
        elif key == "down":
            self.action_nav_down()
        elif key == "up":
            self.action_nav_up()
        elif key == "left":
            self.action_nav_left()
        elif key == "right":
            self.action_nav_right()
        elif key == "enter":
            self.action_nav_enter()
        else:
            handled = False

        if handled:
            event.prevent_default()
            event.stop()

    # -- Panel Focus --

    def _update_panel_focus(self) -> None:
        for i, panel in enumerate(self._panels):
            if i == self._focused_panel_idx:
                panel.add_class("focused")
            else:
                panel.remove_class("focused")
        if self._focused_panel_idx == _CHAT:
            self.query_one("#chat-input").focus()
        else:
            self.set_focus(None)

    def action_focus_next_panel(self) -> None:
        self._focused_panel_idx = (self._focused_panel_idx + 1) % len(self._panels)
        self._update_panel_focus()

    def action_focus_prev_panel(self) -> None:
        self._focused_panel_idx = (self._focused_panel_idx - 1) % len(self._panels)
        self._update_panel_focus()

    # -- Navigation --

    def action_nav_down(self) -> None:
        if self._focused_panel_idx == _LEFT:
            self._left_panel.select_next()
        elif self._focused_panel_idx == _TRAIT:
            self._trait_panel.nav_down()
            if self._highlighting:
                self._apply_highlight_to_all()
            self._refresh_trait_why()

    def action_nav_up(self) -> None:
        if self._focused_panel_idx == _LEFT:
            self._left_panel.select_prev()
        elif self._focused_panel_idx == _TRAIT:
            self._trait_panel.nav_up()
            if self._highlighting:
                self._apply_highlight_to_all()
            self._refresh_trait_why()

    def action_nav_left(self) -> None:
        self._adjust_alpha(-0.01)

    def action_nav_right(self) -> None:
        self._adjust_alpha(0.01)

    def action_nav_enter(self) -> None:
        if self._focused_panel_idx == _LEFT:
            self.action_toggle_vector()

    # -- Chat --

    def on_chat_panel_user_submitted(self, event: ChatPanel.UserSubmitted) -> None:
        text = event.text
        if text.startswith("/"):
            self._handle_command(text)
            return
        self._last_prompt = text
        if self._gen_active:
            # Queue the message — it will be submitted once the current
            # generation finishes (see _poll_generation).
            self._pending_action = ("submit", text)
            self._session.stop()
            return
        self._start_generation(text)

    def _handle_command(self, text: str) -> None:
        chat = self._chat_panel
        parts = text.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd == "/steer":
            if not arg:
                chat.add_system_message(
                    "Usage: /steer <expression>\n"
                    "  e.g. /steer 0.5 honest\n"
                    "       /steer 0.3 warm@after\n"
                    "       /steer 0.5 honest:sae\n"
                    "  For a new bipolar extraction, use /extract <pos> <neg>."
                )
                return
            self._handle_steer(arg)
        elif cmd == "/alpha":
            self._handle_alpha(arg)
        elif cmd == "/unsteer":
            self._handle_unsteer(arg)
        elif cmd == "/unprobe":
            self._handle_unprobe(arg)
        elif cmd == "/seed":
            self._handle_seed(arg)
        elif cmd == "/save":
            self._handle_save(arg)
        elif cmd == "/load":
            self._handle_load(arg)
        elif cmd == "/export":
            self._handle_export(arg)
        elif cmd == "/regen":
            self.action_regenerate()
        elif cmd == "/model":
            self._handle_model_info()
        elif cmd == "/probe":
            if not arg:
                chat.add_system_message(
                    "Usage: /probe <concept>\n"
                    "       /probe <pos> . <neg>"
                )
                return
            self._handle_probe(arg)
        elif cmd == "/extract":
            if not arg:
                chat.add_system_message(
                    "Usage: /extract <concept>\n"
                    "       /extract <pos> . <neg>"
                )
                return
            self._handle_extract_only(arg)
        elif cmd == "/clear":
            if self._gen_active:
                self._pending_action = ("clear",)
                self._session.stop()
                return
            self._do_clear()
        elif cmd == "/rewind":
            if self._gen_active:
                self._pending_action = ("rewind",)
                self._session.stop()
                return
            self._do_rewind()
        elif cmd in ("/system", "/sys"):
            if not arg:
                chat.add_system_message(f"System prompt: {self._session.config.system_prompt or '(none)'}")
            else:
                self._session.config = replace(self._session.config, system_prompt=arg)
                chat.add_system_message("System prompt set.")
                self._refresh_gen_config()
        elif cmd == "/temp":
            if not arg:
                chat.add_system_message(f"Temperature: {self._session.config.temperature}")
            else:
                try:
                    val = max(0.0, float(arg))
                    self._session.config = replace(self._session.config, temperature=val)
                    chat.add_system_message(f"Temperature set to {val}")
                    self._refresh_gen_config()
                except ValueError:
                    chat.add_system_message("Invalid temperature value")
        elif cmd == "/top-p":
            if not arg:
                chat.add_system_message(f"Top-p: {self._session.config.top_p}")
            else:
                try:
                    val = max(0.0, min(1.0, float(arg)))
                    self._session.config = replace(self._session.config, top_p=val)
                    chat.add_system_message(f"Top-p set to {val}")
                    self._refresh_gen_config()
                except ValueError:
                    chat.add_system_message("Invalid top-p value")
        elif cmd == "/max":
            if not arg:
                chat.add_system_message(f"Max tokens: {self._session.config.max_new_tokens}")
            else:
                try:
                    val = max(1, int(arg))
                    self._session.config = replace(self._session.config, max_new_tokens=val)
                    chat.add_system_message(f"Max tokens set to {val}")
                    self._refresh_gen_config()
                except ValueError:
                    chat.add_system_message("Invalid max tokens value")
        elif cmd in ("/exit", "/quit"):
            if self._gen_active:
                self._pending_action = ("quit",)
                self._session.stop()
                return
            self.exit()
        elif cmd == "/compare":
            self._handle_compare(arg)
        elif cmd == "/help":
            chat.add_system_message(
                "Steering:\n"
                "  /steer <concept> [alpha]    — add (extract if needed)\n"
                "  /steer <pos> . <neg> [a]    — add bipolar (period delim)\n"
                "  /alpha <val> <name>         — adjust existing alpha\n"
                "  /unsteer <name>             — remove vector\n"
                "Probes:\n"
                "  /probe <concept>            — add probe (highlight on)\n"
                "  /unprobe <name>             — remove probe\n"
                "  /extract <concept>          — cache-warm only\n"
                "  /compare <a> [b]            — cosine similarity\n"
                "Session:\n"
                "  /clear, /rewind, /regen     — history ops\n"
                "  /save <name>, /load <name>  — snapshot conv + alphas\n"
                "  /export <path>              — JSONL w/ probe readings\n"
                "  /seed [n|clear]             — default sampling seed\n"
                "  /sys [prompt]               — system prompt\n"
                "  /temp, /top-p, /max         — sampling defaults\n"
                "  /model                      — model + session info\n"
                "  /exit, /help\n"
                "Keys: Tab focus · ←/→ alpha · ↑/↓ nav · Enter toggle\n"
                "Backspace remove · Ctrl+T think · Ctrl+R regen\n"
                "Ctrl+A A/B compare · Ctrl+S cycle sort · Ctrl+Y highlight\n"
                "[ ] temp · { } top-p · Esc stop · Ctrl+Q quit"
            )
        else:
            chat.add_system_message(f"Unknown command: {cmd}. Type /help for commands.")

    # -- Vector Management --

    def _on_vector_extracted(self, name: str, alpha: float,
                             profile: dict[int, torch.Tensor]) -> None:
        chat = self._chat_panel
        peak = max(profile, key=lambda k: float(profile[k].norm().item()))
        n_layers = len(profile)
        chat.add_system_message(
            f"Vector '{name}' active (α={alpha:+.1f}, {n_layers}L pk{peak})"
        )
        self._refresh_left_panel()

    @staticmethod
    def _parse_args(text: str, include_alpha: bool = False):
        """Parse /steer, /probe, /extract arguments.

        Accepted forms:
            <concept> [alpha]              single concept; canonical
                                           forms like ``dog.cat`` pass
                                           through unchanged
            <pos> . <neg> [alpha]          bipolar (period delimiter)

        Multi-word poles don't need quotes (``a dog . a pair of cats``).
        Whitespace around the period is what makes it a delimiter — so
        ``dog.cat`` stays a single canonical name.
        """
        text = text.strip()
        alpha = None

        if include_alpha:
            # Peel a trailing float alpha if present. Scan from the right
            # over any runs of trailing non-float tokens — the historical
            # grammar allowed the alpha to sit before stray junk, but in
            # practice it's always last; we accept it there specifically.
            head, _, tail = text.rpartition(" ")
            if tail:
                try:
                    alpha = float(tail)
                    text = head.rstrip()
                except ValueError:
                    pass

        concept, baseline = _split_bipolar(text)

        if include_alpha:
            alpha = (max(-MAX_ALPHA, min(MAX_ALPHA, alpha))
                     if alpha is not None else DEFAULT_ALPHA)
            return concept, baseline, alpha
        return concept, baseline

    def _handle_extract(self, text: str, include_alpha: bool, on_success,
                        pending_type: str | None = None,
                        variant: str = "raw") -> None:
        chat = self._chat_panel
        if self._ab_in_progress:
            chat.add_system_message("Cannot modify vectors during A/B comparison.")
            return
        if pending_type is None:
            pending_type = "steer" if include_alpha else "probe"
        if self._gen_active:
            self._pending_action = (pending_type, text)
            self._session.stop()
            return
        try:
            if include_alpha:
                concept, baseline, alpha = self._parse_args(text, include_alpha=True)
            else:
                concept, baseline = self._parse_args(text)
                alpha = None
        except (ValueError, IndexError) as e:
            chat.add_system_message(
                f"Parse error: {e}\n"
                f"Usage: /{pending_type} <pos> . <neg>" + (" [alpha]" if include_alpha else "")
            )
            return

        # Alias resolution: a bare pole name may refer to an installed
        # bipolar pack (e.g. `/steer wolf` when `deer.wolf` exists).
        # Sign flip is applied to the user's alpha so `/steer calm 0.5`
        # on top of `angry.calm` lands as `alphas["angry.calm"] = -0.5`.
        # ``resolve_pole`` also peels any ``:variant`` suffix typed
        # directly on the concept (``honest:sae-gemma-scope...``) — that
        # explicit form wins over the ``variant`` kwarg when both are set.
        # Explicit bipolar form (`concept - baseline`) skips resolution
        # so the user's declared poles always win.
        sign = 1
        if baseline is None:
            try:
                resolved_name, sign, _match, explicit_variant = resolve_pole(concept)
                if resolved_name != concept:
                    chat.add_system_message(
                        f"  Resolved '{concept}' → '{resolved_name}'"
                        + (" (negated)" if sign < 0 else "")
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
            if alpha is not None:
                alpha *= sign

        # Variant routing. ``--sae`` alone (variant == "sae") means "pick the
        # unique already-extracted SAE tensor for this concept on disk" —
        # session autoload handles it. To drive a fresh extraction, users
        # pass the explicit ``:sae-<release>`` suffix, which routes the
        # release through ``session.extract(sae=RELEASE)``.
        sae_release: str | None = None
        if variant.startswith("sae-"):
            sae_release = variant[len("sae-"):]

        display = concept if len(concept) <= 20 else concept[:17] + "..."
        suffix = f" vs '{baseline}'" if baseline else ""
        variant_note = f" [{variant}]" if variant != "raw" else ""
        chat.add_system_message(f"Extracting '{display}'{suffix}{variant_note}...")

        def _worker():
            def _progress(msg):
                self.call_from_thread(self._steer_status, msg)
            try:
                # Bare ``--sae`` (variant == "sae") routes the load through
                # session autoload rather than a fresh PCA extract — it
                # means "use the unique SAE variant that's already on disk".
                # Ambiguous / missing cases surface via the session errors.
                if variant == "sae" and sae_release is None:
                    self._session._try_autoload_vector(concept, variant="sae")
                    key = f"{concept}:sae"
                    profile_dict = self._session._profiles.get(key)
                    if profile_dict is None:
                        raise ValueError(
                            f"no SAE variant loaded for '{concept}' — "
                            f"run `saklas vector extract {concept} --sae <RELEASE>` "
                            f"first, or pick a release with "
                            f"`:sae-<release>` in the concept name."
                        )
                    on_success(key, profile_dict, alpha)
                    return

                extract_kwargs = {"baseline": baseline, "on_progress": _progress}
                if sae_release is not None:
                    extract_kwargs["sae"] = sae_release
                # ``session.extract`` already returns the fully-qualified
                # canonical name — including the ``:sae-<release>`` suffix
                # when ``sae=`` was passed. Rebuilding it here would
                # double-suffix the key and break every downstream
                # ``/alpha`` / ``/unsteer`` / pole lookup.
                canonical, profile = self._session.extract(concept, **extract_kwargs)
                on_success(canonical, profile, alpha)
            except SaklasError as e:
                self.call_from_thread(self._steer_status, e.user_message()[1])
            except ValueError as e:
                self.call_from_thread(self._steer_status, str(e))
            except Exception as e:
                # AmbiguousVariantError / UnknownVariantError are KeyError/ValueError
                # subclasses — either branch lands here. Surface cleanly.
                self.call_from_thread(self._steer_status, f"{type(e).__name__}: {e}")

        self.run_worker(_worker, thread=True)

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
            ProjectedTerm, SteeringExprError, parse_expr,
        )

        chat = self._chat_panel
        text = text.strip()
        if not text:
            chat.add_system_message(
                'Usage: /steer <expression>\n'
                '  e.g. /steer 0.5 honest\n'
                '       /steer 0.3 warm@after\n'
                '       /steer 0.5 honest|sycophantic\n'
                "  For new concept extraction use /extract <pos> <neg>."
            )
            return
        try:
            steering = parse_expr(text)
        except SteeringExprError as e:
            chat.add_system_message(f"Steering expression error: {e.user_message()[1]}")
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
            self._dispatch_steer_term(concept, variant, alpha)

    def _dispatch_steer_term(
        self, concept: str, variant: str, alpha: float,
    ) -> None:
        """Route one plain steering term through the extract pipeline.

        The concept has already been canonicalized and sign-flipped by
        ``parse_expr``; ``_handle_extract`` will re-run ``resolve_pole``
        on the canonical form which is idempotent (returns sign +1).
        """
        def _on_success(name, profile, a):
            self._session.steer(name, profile)
            self._alphas[name] = a
            self._enabled[name] = True
            self.call_from_thread(self._on_vector_extracted, name, a, profile)
        # _handle_extract's own parser expects a "<concept> <alpha>" text;
        # embed the variant in the concept token so resolve_pole strips it.
        concept_with_variant = (
            concept if variant == "raw" else f"{concept}:{variant}"
        )
        text = f"{concept_with_variant} {alpha}"
        self._handle_extract(
            text, include_alpha=True, on_success=_on_success, variant=variant,
        )

    def _handle_probe(self, text: str) -> None:
        def _on_success(name, profile, _alpha):
            self._session.probe(name, profile)
            self.call_from_thread(self._on_probe_added, name)
        self._handle_extract(text, include_alpha=False, on_success=_on_success)

    def _handle_extract_only(self, text: str) -> None:
        def _on_success(name, _profile, _alpha):
            # Pure cache-warm: no steering, no probe, no panel state.
            self.call_from_thread(
                self._steer_status, f"extracted '{name}'"
            )
        self._handle_extract(
            text, include_alpha=False, on_success=_on_success,
            pending_type="extract",
        )

    def _steer_status(self, msg: str) -> None:
        self._chat_panel.add_system_message(msg)

    def _on_probe_added(self, name: str) -> None:
        self._trait_panel.set_active_probes(set(self._session._monitor.probe_names))
        # Per-token highlight default-on when a probe is explicitly added
        # via /probe. Seed to this probe; Ctrl+Y toggles visually.
        self._highlight_probe = name
        self._highlighting = True
        self._apply_highlight_to_all()
        self._refresh_trait_why()
        self._steer_status(f"Probe '{name}' active. Highlight on (Ctrl+Y to toggle).")

    def _refresh_left_panel(self) -> None:
        self._left_panel.update_vectors(self._vector_list_for_panel())

    def _do_clear(self) -> None:
        self._session.clear_history()
        self._chat_panel.clear_log()
        self._assistant_messages.clear()
        self._trait_panel.update_values({}, {}, {})
        self._refresh_trait_why()
        self._chat_panel.add_system_message("Chat history cleared.")

    def _do_rewind(self) -> None:
        if not self._messages:
            self._chat_panel.add_system_message("Nothing to rewind.")
            return
        self._session.rewind()
        self._chat_panel.rewind()
        self._assistant_messages = [w for w in self._assistant_messages if w.is_mounted]
        self._chat_panel.add_system_message("Rewound to before last message.")

    def _refresh_gen_config(self) -> None:
        self._left_panel.update_gen_config(
            self._session.config.temperature,
            self._session.config.top_p,
            self._session.config.max_new_tokens,
            self._session.config.system_prompt,
            thinking=self._thinking if self._supports_thinking else None,
        )

    # -- Clipboard --

    def action_copy_selection(self) -> None:
        text = self.screen.get_selected_text()
        if text:
            self.copy_to_clipboard(text)

    # -- Generation --

    def action_stop_generation(self) -> None:
        if self._gen_active:
            self._session.stop()

    async def action_quit(self) -> None:
        if self._gen_active:
            self._session.stop()
            self._pending_action = ("quit",)
        else:
            self.exit()

    def _start_generation(self, user_text: str | None = None) -> None:
        """Kick off a generation.

        ``user_text`` is the new user message (``None`` = regeneration of
        the last turn, so we pass the existing history via input=[] style —
        actually we pop the last assistant and re-use the last user content).
        """
        self._gen_active = True

        self._gen_token_count = 0
        self._last_tok_per_sec = 0.0
        self._last_elapsed = 0.0
        self._log_ppl_sum = 0.0
        self._ppl_count = 0
        self._gen_start_time = time.monotonic()

        widget = self._chat_panel.start_assistant_message()
        self._current_assistant_widget = widget
        self._assistant_messages.append(widget)
        # Fresh widgets spawn with ``_highlight_on=False``; inherit the
        # app's current highlight state so streamed tokens render
        # highlighted from the first emit instead of requiring a Ctrl+Y
        # off/on cycle after the response completes.
        if self._highlighting:
            widget.apply_highlight(True, self._highlight_probe)

        # Snapshot alphas for this generation
        alphas = self._active_alphas()
        use_thinking = self._thinking

        # For regeneration, we re-submit the last user message. Session
        # owns history; _handle_command / action_regenerate pop the last
        # assistant turn + user turn before calling us with that text.
        if user_text is None:
            # Regenerate: take the last user message off history and
            # re-send it as input (session will re-append).
            if self._messages and self._messages[-1]["role"] == "user":
                user_text = self._messages.pop()["content"]
            else:
                self._gen_active = False
                self._chat_panel.add_system_message("Nothing to regenerate.")
                return

        sampling = SamplingConfig(
            temperature=self._session.config.temperature,
            top_p=self._session.config.top_p,
            max_tokens=self._session.config.max_new_tokens,
            seed=self._default_seed,
        )
        steering = Steering(alphas=dict(alphas), thinking=use_thinking) if alphas else None

        def _generate():
            try:
                stream = self._session.generate_stream(
                    user_text,
                    steering=steering,
                    sampling=sampling,
                    stateless=False,
                    thinking=use_thinking,
                )
                for event in stream:
                    self._ui_token_queue.put(
                        ("tok", event.text, event.thinking, event.scores,
                         event.perplexity)
                    )
                    self._gen_token_count += 1
                # Normal completion — pull per-token scores out of the
                # session and push to the widget for highlight.
                self._ui_token_queue.put(("finalize", widget))
            except BaseException as e:
                msg = e.user_message()[1] if isinstance(e, SaklasError) else str(e)
                self._ui_token_queue.put(("error", msg))
            finally:
                if self._session._device.type == "mps":
                    try:
                        torch.mps.synchronize()
                    except Exception:
                        pass
                self._ui_token_queue.put(("done",))

        self.run_worker(_generate, thread=True)

    def _poll_generation(self) -> None:
        chat = self._chat_panel
        tokens_consumed = 0
        generating = self._gen_active

        while tokens_consumed < _TOKEN_DRAIN_LIMIT:
            try:
                item = self._ui_token_queue.get_nowait()
            except queue.Empty:
                break
            kind = item[0]
            if kind == "tok":
                _, token, is_thinking, scores, perplexity = item
                widget = self._current_assistant_widget
                if widget:
                    if is_thinking:
                        widget.append_thinking_token(token)
                    else:
                        widget.ensure_thinking_collapsed()
                        widget.append_token(token)
                    if scores is not None:
                        widget.append_token_score(scores, is_thinking)
                if perplexity is not None and perplexity > 0:
                    # Geometric mean over the gen: accumulate log(ppl),
                    # display exp(mean).  Equivalent to classical sequence
                    # perplexity; one step dominated by a rare token
                    # doesn't swamp the aggregate the way an arithmetic
                    # mean would.
                    self._log_ppl_sum += math.log(perplexity)
                    self._ppl_count += 1
                tokens_consumed += 1
            elif kind == "finalize":
                # Normal end — pull per-token scores stashed by session's
                # _finalize_generation and push to the widget for highlight.
                _, widget = item
                self._finalize_widget_highlight(widget)
            elif kind == "error":
                chat.add_system_message(f"generation error: {item[1]}")
            elif kind == "done":
                if self._current_assistant_widget:
                    self._current_assistant_widget.ensure_thinking_collapsed()
                self._current_assistant_widget = None
                self._gen_active = False
                generating = False
                if self._gen_start_time > 0:
                    self._last_elapsed = time.monotonic() - self._gen_start_time
                    if self._last_elapsed > MIN_ELAPSED_FOR_RATE:
                        self._last_tok_per_sec = self._gen_token_count / self._last_elapsed
                    self._gen_start_time = 0.0

                pending = self._pending_action
                if pending is not None:
                    self._pending_action = None
                    self._dispatch_pending_action(pending)
                break

        if tokens_consumed > 0:
            chat.scroll_to_bottom()

        if generating and self._gen_start_time > 0:
            elapsed = time.monotonic() - self._gen_start_time
            tok_per_sec = self._gen_token_count / elapsed if elapsed > MIN_ELAPSED_FOR_RATE else 0.0
            self._last_tok_per_sec = tok_per_sec
            self._last_elapsed = elapsed

        new_state = (self._gen_token_count, self._last_tok_per_sec,
                     self._last_elapsed, generating, self._ppl_count)
        if new_state != self._last_gen_state:
            self._last_gen_state = new_state
            ppl_mean = (
                math.exp(self._log_ppl_sum / self._ppl_count)
                if self._ppl_count > 0 else None
            )
            chat.update_status(
                generating=generating,
                gen_tokens=self._gen_token_count,
                max_tokens=self._session.config.max_new_tokens,
                tok_per_sec=self._last_tok_per_sec,
                elapsed=self._last_elapsed,
                perplexity=ppl_mean,
            )

        if self._session._monitor and self._session._monitor.has_pending_data():
            self._session._monitor.consume_pending()
            current, previous = self._session._monitor.get_current_and_previous()
            sparklines = {name: self._session._monitor.get_sparkline(name)
                          for name in self._session._monitor.probe_names}
            self._trait_panel.update_values(
                current, previous, sparklines,
            )

    # -- Actions --

    def action_remove_vector(self) -> None:
        if self._ab_in_progress:
            return
        if self._focused_panel_idx == _TRAIT:
            self._remove_selected_probe()
            return
        lp = self._left_panel
        sel = lp.get_selected()
        if sel:
            name = sel["name"]
            self._session.unsteer(name)
            self._alphas.pop(name, None)
            self._enabled.pop(name, None)
            self._refresh_left_panel()

    def _remove_selected_probe(self) -> None:
        tp = self._trait_panel
        probe_name = tp.get_selected_probe()
        if not probe_name or not self._session._monitor:
            return
        self._session.unprobe(probe_name)
        tp.set_active_probes(set(self._session._monitor.probe_names))
        self._refresh_trait_why()

    def action_toggle_vector(self) -> None:
        if self._ab_in_progress:
            return
        lp = self._left_panel
        sel = lp.get_selected()
        if sel:
            name = sel["name"]
            self._enabled[name] = not self._enabled.get(name, True)
            self._refresh_left_panel()

    def _adjust_alpha(self, delta: float) -> None:
        if self._ab_in_progress:
            return
        lp = self._left_panel
        sel = lp.get_selected()
        if sel:
            name = sel["name"]
            self._alphas[name] = max(-MAX_ALPHA, min(MAX_ALPHA, self._alphas.get(name, 0.0) + delta))
            self._refresh_left_panel()

    def action_toggle_highlight(self) -> None:
        if self._ab_in_progress:
            return
        if not self._highlighting:
            # Prefer the stored seed, fall back to trait-panel selection.
            seed = self._highlight_probe or self._trait_panel.get_selected_probe()
            if seed is None:
                return
            self._highlight_probe = seed
            self._highlighting = True
        else:
            self._highlighting = False
        self._apply_highlight_to_all()

    def _apply_highlight_to_all(self) -> None:
        # Navigating the trait panel updates the seed live while highlight is on.
        if self._highlighting:
            nav_probe = self._trait_panel.get_selected_probe()
            if nav_probe is not None:
                self._highlight_probe = nav_probe
        probe = self._highlight_probe if self._highlighting else None
        # Prune any unmounted widgets (rewind/clear may have detached them).
        self._assistant_messages = [w for w in self._assistant_messages if w.is_mounted]
        for widget in self._assistant_messages:
            widget.apply_highlight(self._highlighting, probe)

    def _set_widget_token_data(
        self, widget: _AssistantMessage,
        response_token_strs: list[str],
        response_probe_scores: dict[str, list[float]],
        thinking_token_strs: list[str],
        thinking_probe_scores: dict[str, list[float]],
    ) -> None:
        if not widget.is_mounted:
            return
        widget.set_token_data(
            response_token_strs, response_probe_scores,
            thinking_token_strs, thinking_probe_scores,
        )
        if self._highlighting:
            widget.apply_highlight(True, self._highlight_probe)

    def _finalize_widget_highlight(self, widget: _AssistantMessage) -> None:
        """Pull per-token scores the session stashed during finalize and
        push to the widget for highlight-mode overlays.

        The session's ``per_token_scores`` are indexed in ``generated_ids``
        space (one score per forward-pass step, including delimiters and
        preamble tokens that were suppressed from the on_token stream).
        ``gen_state.emit_map`` records which ``generated_ids`` index
        corresponds to each emitted token, letting us project scores into
        the widget's token-string space without re-decoding.
        """
        per_token = self._session.last_per_token_scores
        if not per_token or not widget.is_mounted:
            return
        emit_map = self._session._gen_state.emit_map
        if not emit_map:
            return

        # Use the widget's own streamed token strings — these match exactly
        # what was rendered, avoiding batch_decode mismatches.
        response_strs = list(widget._streamed_response_tokens)
        thinking_strs = list(widget._streamed_thinking_tokens)

        # Project scores from generated_ids space to emitted-token space.
        thinking_scores: dict[str, list[float]] = {k: [] for k in per_token}
        response_scores: dict[str, list[float]] = {k: [] for k in per_token}
        think_i = 0
        resp_i = 0
        for gen_idx, is_thinking in emit_map:
            if is_thinking:
                if think_i < len(thinking_strs):
                    for k, scores in per_token.items():
                        if gen_idx < len(scores):
                            thinking_scores[k].append(scores[gen_idx])
                    think_i += 1
            else:
                if resp_i < len(response_strs):
                    for k, scores in per_token.items():
                        if gen_idx < len(scores):
                            response_scores[k].append(scores[gen_idx])
                    resp_i += 1

        widget.set_token_data(
            response_strs, response_scores,
            thinking_strs, thinking_scores,
        )
        if self._highlighting:
            widget.apply_highlight(True, self._highlight_probe)
        self._refresh_trait_why()

    # -- New slash command handlers --

    def _handle_alpha(self, arg: str) -> None:
        chat = self._chat_panel
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
        self._refresh_left_panel()
        chat.add_system_message(f"Alpha for '{name}' set to {val:+.2f}")

    def _handle_unsteer(self, arg: str) -> None:
        chat = self._chat_panel
        raw = arg.strip()
        if not raw:
            chat.add_system_message("Usage: /unsteer <name>")
            return
        matches = _resolve_active_name(raw, self._alphas)
        if len(matches) == 0:
            chat.add_system_message(f"'{raw}' is not active.")
            return
        if len(matches) > 1:
            chat.add_system_message(f"'{raw}' is ambiguous: {', '.join(matches)}")
            return
        name = matches[0]
        self._session.unsteer(name)
        self._alphas.pop(name, None)
        self._enabled.pop(name, None)
        self._refresh_left_panel()
        chat.add_system_message(f"Removed '{name}'.")

    def _handle_unprobe(self, arg: str) -> None:
        chat = self._chat_panel
        raw = arg.strip()
        if not raw:
            chat.add_system_message("Usage: /unprobe <name>")
            return
        monitor = self._session._monitor
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
        self._session.unprobe(name)
        self._trait_panel.set_active_probes(set(monitor.probe_names))
        if self._highlight_probe == name:
            self._highlight_probe = None
            self._highlighting = False
            self._apply_highlight_to_all()
        self._refresh_trait_why()
        chat.add_system_message(f"Probe '{name}' removed.")

    def _handle_seed(self, arg: str) -> None:
        chat = self._chat_panel
        arg = arg.strip()
        if not arg:
            chat.add_system_message(f"Seed: {self._default_seed}")
            return
        if arg.lower() == "clear":
            self._default_seed = None
            chat.add_system_message("Seed cleared.")
            return
        try:
            self._default_seed = int(arg)
            chat.add_system_message(f"Seed set to {self._default_seed}")
        except ValueError:
            chat.add_system_message("Invalid seed value (expected int or 'clear').")

    def _conv_dir(self) -> Path:
        d = saklas_home() / "conversations"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _handle_save(self, arg: str) -> None:
        chat = self._chat_panel
        name = arg.strip()
        if not name:
            chat.add_system_message("Usage: /save <name>")
            return
        snapshot = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_id": self._session._model_info.get("model_id", "unknown"),
            "history": list(self._session._history),
            "alphas": dict(self._alphas),
            "enabled": dict(self._enabled),
            "probes": list(self._session._monitor.probe_names) if self._session._monitor else [],
            "seed": self._default_seed,
        }
        path = self._conv_dir() / f"{name}.json"
        path.write_text(json.dumps(snapshot, indent=2))
        chat.add_system_message(f"Saved snapshot to {path}")

    def _handle_load(self, arg: str) -> None:
        chat = self._chat_panel
        name = arg.strip()
        if not name:
            chat.add_system_message("Usage: /load <name>")
            return
        path = self._conv_dir() / f"{name}.json"
        if not path.exists():
            chat.add_system_message(f"No snapshot at {path}")
            return
        try:
            snapshot = json.loads(path.read_text())
        except Exception as e:
            chat.add_system_message(f"Load error: {e}")
            return
        self._session.clear_history()
        self._chat_panel.clear_log()
        self._assistant_messages.clear()
        for msg in snapshot.get("history", []):
            self._session._history.append(msg)
            if msg["role"] == "user":
                chat.add_user_message(msg["content"]) if hasattr(chat, "add_user_message") else None
        # Restore alphas (vectors must already be installed in session)
        self._alphas = {
            k: v for k, v in snapshot.get("alphas", {}).items()
            if k in self._session._profiles
        }
        self._enabled = {k: True for k in self._alphas}
        self._enabled.update(snapshot.get("enabled", {}))
        self._default_seed = snapshot.get("seed")
        self._refresh_left_panel()
        chat.add_system_message(
            f"Loaded {name}: {len(self._session._history)} msgs, "
            f"{len(self._alphas)} alphas."
        )

    def _handle_export(self, arg: str) -> None:
        chat = self._chat_panel
        path_str = arg.strip()
        if not path_str:
            chat.add_system_message("Usage: /export <path>")
            return
        collector = ResultCollector()
        last = self._session.last_result
        if last is not None:
            collector.add(last)
        path = Path(path_str).expanduser()
        try:
            collector.to_jsonl(path)
            chat.add_system_message(f"Exported {len(collector)} result(s) to {path}")
        except Exception as e:
            chat.add_system_message(f"Export error: {e}")

    def _handle_model_info(self) -> None:
        chat = self._chat_panel
        info = self._session._model_info
        lines = [
            f"Model: {info.get('model_id', 'unknown')}",
            f"Arch: {info.get('model_type', 'unknown')}  "
            f"Device: {self._device_str}  "
            f"Layers: {len(self._session._layers)}",
            f"Thinking supported: {self._supports_thinking}  active: {self._thinking}",
            f"Active vectors: {list(self._alphas.keys()) or '(none)'}",
            f"Active probes: {list(self._session._monitor.probe_names) if self._session._monitor else '(none)'}",
            f"Seed: {self._default_seed}",
        ]
        chat.add_system_message("\n".join(lines))

    def _refresh_trait_why(self) -> None:
        """Push per-layer ||baked|| norms for the trait-panel-selected probe
        down to the panel's WHY section as a histogram in layer order.

        Per-token highlighting in the chat already surfaces which tokens
        a probe lights up on — no token list duplicated here.
        """
        probe = self._trait_panel.get_selected_probe()
        monitor = self._session._monitor
        if probe is None or monitor is None or probe not in monitor.profiles:
            self._trait_panel.update_why(None, [])
            return
        profile = monitor.profiles[probe]
        layer_norms = sorted(
            ((int(lidx), float(t.norm().item())) for lidx, t in profile.items()),
            key=lambda kv: kv[0],
        )
        self._trait_panel.update_why(probe, layer_norms)

    def _handle_compare(self, arg: str) -> None:
        chat = self._chat_panel
        if not arg:
            chat.add_system_message("Usage: /compare <name> [other_name]")
            return

        parts = arg.split()

        # Gather all available profiles: session profiles + monitor probes.
        # Monitor stores raw ``dict[int, Tensor]`` — wrap it in a Profile so
        # both sources expose the same cosine_similarity API.
        from saklas.core.profile import Profile
        all_profiles: dict[str, Profile] = {}
        for name, prof in self._session._profiles.items():
            if isinstance(prof, Profile):
                all_profiles[name] = prof
            elif isinstance(prof, dict) and prof:
                all_profiles[name] = Profile(prof)
        if self._session._monitor:
            for name, prof in self._session._monitor.profiles.items():
                if name in all_profiles:
                    continue
                if isinstance(prof, Profile):
                    all_profiles[name] = prof
                elif isinstance(prof, dict) and prof:
                    all_profiles[name] = Profile(prof)

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
            scores = {}
            for name, prof in others.items():
                try:
                    scores[name] = target.cosine_similarity(prof)
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
            try:
                sim = all_profiles[a_name].cosine_similarity(all_profiles[b_name])
            except Exception as e:
                chat.add_system_message(f"Compare failed: {e}")
                return
            chat.add_system_message(f"{a_name} ~ {b_name}: {sim:+.4f}")

        else:
            chat.add_system_message("Usage: /compare <name> [other_name]")

    def _dispatch_pending_action(self, pending: tuple) -> None:
        """Handle a queued action dispatched once the current gen finishes."""
        kind = pending[0]
        chat = self._chat_panel
        try:
            if kind == "regenerate":
                if self._messages and self._messages[-1]["role"] == "assistant":
                    self._messages.pop()
                chat.rewind_last_assistant()
                self._start_generation()
            elif kind == "submit":
                self._start_generation(pending[1])
            elif kind == "clear":
                self._do_clear()
            elif kind == "rewind":
                if self._messages and self._messages[-1]["role"] == "assistant":
                    self._messages.pop()
                chat.rewind_last_assistant()
                self._do_rewind()
            elif kind == "steer":
                self._handle_steer(pending[1])
            elif kind == "probe":
                self._handle_probe(pending[1])
            elif kind == "extract":
                self._handle_extract_only(pending[1])
            elif kind == "quit":
                self.exit()
        except KeyboardInterrupt:
            raise
        except Exception as e:
            import sys
            import traceback
            traceback.print_exc(file=sys.stderr)
            self._gen_active = False
            self._pending_action = None
            self._current_assistant_widget = None
            chat.add_system_message(f"error dispatching {kind}: {e}")

    def action_toggle_thinking(self) -> None:
        if not self._supports_thinking:
            self._chat_panel.add_system_message("This model does not support thinking mode.")
            return
        self._thinking = not self._thinking
        self._refresh_gen_config()

    def _adjust_config(self, attr: str, delta: float, lo: float, hi: float) -> None:
        if self._focused_panel_idx != _LEFT:
            return
        val = getattr(self._session.config, attr)
        new_val = round(max(lo, min(hi, val + delta)), 2)
        self._session.config = replace(self._session.config, **{attr: new_val})
        self._refresh_gen_config()

    def action_temp_down(self) -> None:
        self._adjust_config("temperature", -0.05, 0.0, float("inf"))

    def action_temp_up(self) -> None:
        self._adjust_config("temperature", 0.05, 0.0, float("inf"))

    def action_top_p_down(self) -> None:
        self._adjust_config("top_p", -0.05, 0.0, 1.0)

    def action_top_p_up(self) -> None:
        self._adjust_config("top_p", 0.05, 0.0, 1.0)

    def action_regenerate(self) -> None:
        if not self._messages:
            return
        if self._gen_active:
            # Stop the current generation; _poll_generation will pick up
            # the pending action once the worker thread finishes.
            self._pending_action = ("regenerate",)
            self._session.stop()
            return
        if self._messages[-1]["role"] == "assistant":
            self._messages.pop()
        self._start_generation()

    def action_ab_compare(self) -> None:
        chat = self._chat_panel
        if self._gen_active:
            chat.add_system_message("Cannot A/B compare while generating. Stop generation first.")
            return
        if not self._last_prompt:
            chat.add_system_message("No previous prompt to compare.")
            return
        self._ab_in_progress = True
        chat.add_system_message("A/B comparison: generating unsteered response...")

        def _ab_generate():
            try:
                # Stateless so we don't mutate history; no steering.
                result = self._session.generate(
                    self._last_prompt,
                    steering=None,
                    stateless=True,
                    thinking=self._thinking,
                )
                self.call_from_thread(self._show_ab_result, result.text)
            except Exception as e:
                err = e.user_message()[1] if isinstance(e, SaklasError) else str(e)
                self.call_from_thread(
                    lambda: (setattr(self, '_ab_in_progress', False),
                             self._chat_panel.add_system_message(f"A/B error: {err}"))
                )

        self.run_worker(_ab_generate, thread=True)

    def _show_ab_result(self, unsteered: str) -> None:
        self._ab_in_progress = False
        self._chat_panel.add_system_message(f"[Unsteered]: {unsteered}")

    def action_cycle_sort(self) -> None:
        self._trait_panel.cycle_sort()
