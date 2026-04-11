"""Main Textual application for liahona."""

from __future__ import annotations

import queue
import shlex
import time

import torch
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.widgets import Input
from textual.timer import Timer

from liahona.generation import GenerationState, build_chat_input, generate_steered
from liahona.model import _get_memory_gb
from liahona.probes_bootstrap import _load_defaults
from liahona.tui.chat_panel import ChatPanel
from liahona.tui.vector_panel import LeftPanel
from liahona.tui.trait_panel import TraitPanel

PANELS = ["left-panel", "chat-panel", "trait-panel"]


class LiahonaApp(App):
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
        Binding("ctrl+o", "toggle_ortho", "Ortho", show=False),
        Binding("ctrl+s", "cycle_sort", "Sort", show=False),
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
        self._orthogonalize: bool = False

        self._current_assistant_widget = None
        self._poll_timer: Timer | None = None
        self._last_prompt: str | None = None
        self._ab_in_progress: bool = False
        self._pending_action: tuple | None = None  # ("regenerate",) or ("submit", text)

        self._focused_panel_idx: int = 1  # Start with chat focused

        self._gen_start_time: float = 0.0
        self._gen_token_count: int = 0
        self._prompt_token_count: int = 0
        self._last_tok_per_sec: float = 0.0
        self._last_elapsed: float = 0.0
        self._cached_vram_gb: float = 0.0
        self._vram_poll_counter: int = 0
        self._last_status_args: tuple = ()

        defaults = _load_defaults()
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
        for name in self._alphas:
            if name in self._session._profiles:
                result.append({
                    "name": name,
                    "profile": self._session._profiles[name],
                    "alpha": self._alphas[name],
                    "enabled": self._enabled.get(name, True),
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
        self._left_panel.update_gen_config(
            self._session.config.temperature,
            self._session.config.top_p,
            self._session.config.max_new_tokens,
            self._session.config.system_prompt,
        )

        if self._session._monitor:
            self._trait_panel.set_active_probes(set(self._session._monitor.probe_names))

        self._poll_timer = self.set_interval(1 / 15, self._poll_generation)
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
        if self._focused_panel_idx == 1:  # chat panel
            self.query_one("#chat-input").focus()
        else:
            self.set_focus(None)

    def action_focus_next_panel(self) -> None:
        self._focused_panel_idx = (self._focused_panel_idx + 1) % len(PANELS)
        self._update_panel_focus()

    def action_focus_prev_panel(self) -> None:
        self._focused_panel_idx = (self._focused_panel_idx - 1) % len(PANELS)
        self._update_panel_focus()

    # -- Navigation --

    def action_nav_down(self) -> None:
        panel = PANELS[self._focused_panel_idx]
        if panel == "left-panel":
            self._left_panel.select_next()
        elif panel == "trait-panel":
            self._trait_panel.nav_down()

    def action_nav_up(self) -> None:
        panel = PANELS[self._focused_panel_idx]
        if panel == "left-panel":
            self._left_panel.select_prev()
        elif panel == "trait-panel":
            self._trait_panel.nav_up()

    def action_nav_left(self) -> None:
        self._adjust_alpha(-0.01)

    def action_nav_right(self) -> None:
        self._adjust_alpha(0.01)

    def action_nav_enter(self) -> None:
        panel = PANELS[self._focused_panel_idx]
        if panel == "left-panel":
            self.action_toggle_vector()

    # -- Chat --

    def on_chat_panel_user_submitted(self, event: ChatPanel.UserSubmitted) -> None:
        text = event.text
        if text.startswith("/"):
            self._handle_command(text)
            return
        self._last_prompt = text
        if self._session._gen_state.is_generating.is_set():
            # Queue the message — it will be submitted once the current
            # generation finishes (see _poll_generation).
            self._pending_action = ("submit", text)
            self._session._gen_state.request_stop()
            return
        self._messages.append({"role": "user", "content": text})
        self._start_generation()

    def _handle_command(self, text: str) -> None:
        chat = self._chat_panel
        parts = text.split(maxsplit=1)
        cmd = parts[0].lower()

        if cmd == "/steer":
            if len(parts) < 2:
                chat.add_system_message(
                    'Usage: /steer "concept" [alpha]\n'
                    '       /steer "concept" - "baseline" [alpha]'
                )
                return
            self._handle_steer(parts[1])
        elif cmd == "/probe":
            if len(parts) < 2:
                chat.add_system_message(
                    'Usage: /probe "concept"\n'
                    '       /probe "concept" - "baseline"'
                )
                return
            self._handle_probe(parts[1])
        elif cmd == "/clear":
            if self._session._gen_state.is_generating.is_set():
                self._pending_action = ("clear",)
                self._session._gen_state.request_stop()
                return
            self._session.clear_history()
            chat.clear_log()
            self._trait_panel.update_values({}, {}, {})
            chat.add_system_message("Chat history cleared.")
        elif cmd == "/rewind":
            if self._session._gen_state.is_generating.is_set():
                self._pending_action = ("rewind",)
                self._session._gen_state.request_stop()
                return
            if not self._messages:
                chat.add_system_message("Nothing to rewind.")
            else:
                self._session.rewind()
                chat.rewind()
                chat.add_system_message("Rewound to before last message.")
        elif cmd in ("/system", "/sys"):
            if len(parts) < 2:
                chat.add_system_message(f"System prompt: {self._session.config.system_prompt or '(none)'}")
            else:
                self._session.config.system_prompt = parts[1]
                chat.add_system_message("System prompt set.")
                self._refresh_gen_config()
        elif cmd == "/temp":
            if len(parts) < 2:
                chat.add_system_message(f"Temperature: {self._session.config.temperature}")
            else:
                try:
                    self._session.config.temperature = max(0.0, float(parts[1]))
                    chat.add_system_message(f"Temperature set to {self._session.config.temperature}")
                    self._refresh_gen_config()
                except ValueError:
                    chat.add_system_message("Invalid temperature value")
        elif cmd == "/top-p":
            if len(parts) < 2:
                chat.add_system_message(f"Top-p: {self._session.config.top_p}")
            else:
                try:
                    self._session.config.top_p = max(0.0, min(1.0, float(parts[1])))
                    chat.add_system_message(f"Top-p set to {self._session.config.top_p}")
                    self._refresh_gen_config()
                except ValueError:
                    chat.add_system_message("Invalid top-p value")
        elif cmd == "/max":
            if len(parts) < 2:
                chat.add_system_message(f"Max tokens: {self._session.config.max_new_tokens}")
            else:
                try:
                    self._session.config.max_new_tokens = max(1, int(parts[1]))
                    chat.add_system_message(f"Max tokens set to {self._session.config.max_new_tokens}")
                    self._refresh_gen_config()
                except ValueError:
                    chat.add_system_message("Invalid max tokens value")
        elif cmd == "/help":
            chat.add_system_message(
                'Commands: /steer "concept" [alpha], '
                '/steer "concept" - "baseline" [alpha],\n'
                '/probe "concept", '
                '/probe "concept" - "baseline",\n'
                '/clear, /rewind, /sys [prompt], '
                "/temp [val], /top-p [val], /max [n], /help\n"
                "Keys: ⇥ focus · ←/→ alpha · ↑/↓ nav · ↩ toggle\n"
                "⌫ remove · ⌃O ortho · ⌃R regen · ⌃A A/B\n"
                "[ ] temp · { } top-p · ⌃S sort · ⎋ stop · ⌃Q quit"
            )
        else:
            chat.add_system_message(f"Unknown command: {cmd}. Type /help for commands.")

    # -- Vector Management --

    def _on_vector_extracted(self, name: str, alpha: float,
                             profile: dict[int, tuple[torch.Tensor, float]]) -> None:
        chat = self._chat_panel
        peak = max(profile, key=lambda k: profile[k][1])
        n_layers = len(profile)
        chat.add_system_message(
            f"Vector '{name}' active (α={alpha:+.1f}, {n_layers}L pk{peak})"
        )
        self._refresh_left_panel()

    @staticmethod
    def _parse_args(text: str, include_alpha: bool = False):
        """Parse /steer or /probe arguments."""
        if " - " in text:
            dash_idx = text.index(" - ")
            concept = shlex.split(text[:dash_idx])[0]
            rest_tokens = shlex.split(text[dash_idx + 3:])
            baseline = rest_tokens[0] if rest_tokens else None
            trailing = [t for t in rest_tokens[1:] if not any(c.isalpha() for c in t)]
        else:
            tokens = shlex.split(text)
            concept = tokens[0]
            baseline = None
            trailing = [t for t in tokens[1:] if not any(c.isalpha() for c in t)]
        if include_alpha:
            alpha = float(trailing[0]) if trailing else 0.15
            return concept, baseline, alpha
        return concept, baseline

    def _handle_steer(self, text: str) -> None:
        chat = self._chat_panel
        if self._ab_in_progress:
            chat.add_system_message("Cannot modify vectors during A/B comparison.")
            return
        if self._session._gen_state.is_generating.is_set():
            self._pending_action = ("steer", text)
            self._session._gen_state.request_stop()
            return
        try:
            concept, baseline, alpha = self._parse_args(text, include_alpha=True)
        except (ValueError, IndexError) as e:
            chat.add_system_message(
                f"Parse error: {e}\n"
                'Usage: /steer "concept" - "baseline" [alpha]'
            )
            return

        name = concept if len(concept) <= 20 else concept[:17] + "..."
        suffix = f" vs '{baseline}'" if baseline else ""
        chat.add_system_message(f"Extracting '{name}'{suffix}...")

        def _worker():
            def _progress(msg):
                self.call_from_thread(self._steer_status, msg)
            try:
                profile = self._session.extract(concept, baseline=baseline, on_progress=_progress)
                self._session.steer(name, profile)
                self._alphas[name] = alpha
                self._enabled[name] = True
                self.call_from_thread(self._on_vector_extracted, name, alpha, profile)
            except ValueError as e:
                self.call_from_thread(self._steer_status, str(e))

        self.run_worker(_worker, thread=True)

    def _handle_probe(self, text: str) -> None:
        chat = self._chat_panel
        if self._session._gen_state.is_generating.is_set():
            self._pending_action = ("probe", text)
            self._session._gen_state.request_stop()
            return
        try:
            concept, baseline = self._parse_args(text)
        except (ValueError, IndexError) as e:
            chat.add_system_message(
                f"Parse error: {e}\n"
                'Usage: /probe "concept" - "baseline"'
            )
            return

        name = concept if len(concept) <= 20 else concept[:17] + "..."
        suffix = f" vs '{baseline}'" if baseline else ""
        chat.add_system_message(f"Extracting '{name}'{suffix}...")

        def _worker():
            def _progress(msg):
                self.call_from_thread(self._steer_status, msg)
            try:
                profile = self._session.extract(concept, baseline=baseline, on_progress=_progress)
                self._session.monitor(name, profile)
                self.call_from_thread(self._on_probe_added, name)
            except ValueError as e:
                self.call_from_thread(self._steer_status, str(e))

        self.run_worker(_worker, thread=True)

    def _steer_status(self, msg: str) -> None:
        self._chat_panel.add_system_message(msg)

    def _on_probe_added(self, name: str) -> None:
        self._trait_panel.set_active_probes(set(self._session._monitor.probe_names))
        self._steer_status(f"Probe '{name}' active.")

    def _refresh_left_panel(self) -> None:
        self._left_panel.update_vectors(
            self._vector_list_for_panel(),
            orthogonalize=self._orthogonalize,
        )

    def _refresh_gen_config(self) -> None:
        self._left_panel.update_gen_config(
            self._session.config.temperature,
            self._session.config.top_p,
            self._session.config.max_new_tokens,
            self._session.config.system_prompt,
        )

    # -- Clipboard --

    def action_copy_selection(self) -> None:
        text = self.screen.get_selected_text()
        if text:
            self.copy_to_clipboard(text)

    # -- Generation --

    def action_stop_generation(self) -> None:
        if self._session._gen_state.is_generating.is_set():
            self._session._gen_state.request_stop()

    def _start_generation(self) -> None:
        self._session._gen_state.reset()
        # Mark generating *synchronously* before spawning the worker so that
        # rapid Ctrl+R presses always see is_generating as set.  Without this,
        # there is a window between reset() (clears the flag) and the worker
        # thread's generate_steered() (sets it) where a second Ctrl+R would
        # think nothing is running and launch a concurrent worker — two threads
        # doing model forward passes simultaneously causes a segfault.
        self._session._gen_state.is_generating.set()
        if self._session._monitor:
            self._session._monitor.reset_history()

        self._gen_token_count = 0
        self._prompt_token_count = 0
        self._last_tok_per_sec = 0.0
        self._last_elapsed = 0.0
        self._gen_start_time = time.monotonic()
        self._vram_poll_counter = 0

        self._current_assistant_widget = self._chat_panel.start_assistant_message()

        # Snapshot alphas for this generation
        alphas = self._active_alphas()

        def _generate():
            # Apply steering hooks for this generation
            if alphas:
                self._session._apply_steering(alphas, orthogonalize=self._orthogonalize)

            try:
                input_ids = build_chat_input(
                    self._session._tokenizer, self._messages, self._session.config.system_prompt,
                ).to(self._session._device)
                self._prompt_token_count = input_ids.shape[-1]

                def on_token(tok: str):
                    self._session._gen_state.token_queue.put(tok)

                generated = generate_steered(
                    self._session._model, self._session._tokenizer, input_ids,
                    self._session.config, self._session._gen_state,
                    on_token=on_token,
                )

                full_text = self._session._tokenizer.decode(generated, skip_special_tokens=True)
                if full_text.strip():
                    self._messages.append({"role": "assistant", "content": full_text})
                    if self._session._monitor and self._session._monitor.probe_names:
                        self._session._monitor.measure(
                            self._session._model, self._session._tokenizer,
                            self._session._layers, full_text,
                            device=self._session._device,
                        )
            finally:
                if alphas:
                    self._session._clear_steering()
                # Flush MPS command buffers *after* all GPU work (including
                # monitor.measure) so a pending regenerate dispatched by
                # _poll_generation doesn't submit new Metal commands while
                # the monitor's forward pass is still in flight.
                if self._session._device.type == "mps":
                    torch.mps.synchronize()
                # Signal end-of-generation *after* _messages is updated so
                # pending actions (regenerate / queued submit) see the final
                # conversation state.
                self._session._gen_state.token_queue.put(None)

        self.run_worker(_generate, thread=True)

    def _poll_generation(self) -> None:
        chat = self._chat_panel
        tokens_consumed = 0
        generating = self._session._gen_state.is_generating.is_set()

        while tokens_consumed < 20:
            try:
                token = self._session._gen_state.token_queue.get_nowait()
            except queue.Empty:
                break
            if token is None:
                if self._current_assistant_widget:
                    self._current_assistant_widget.finalize()
                self._current_assistant_widget = None
                generating = False
                if self._gen_start_time > 0:
                    self._last_elapsed = time.monotonic() - self._gen_start_time
                    if self._last_elapsed > 0.1:
                        self._last_tok_per_sec = self._gen_token_count / self._last_elapsed
                    self._gen_start_time = 0.0

                # Dispatch any queued action from Ctrl+R or mid-gen submit.
                pending = self._pending_action
                if pending is not None:
                    self._pending_action = None
                    if pending[0] == "regenerate":
                        # Discard the partial response
                        if self._messages and self._messages[-1]["role"] == "assistant":
                            self._messages.pop()
                        chat.rewind_last_assistant()
                        self._start_generation()
                    elif pending[0] == "submit":
                        self._messages.append({"role": "user", "content": pending[1]})
                        self._start_generation()
                    elif pending[0] == "clear":
                        self._session.clear_history()
                        chat.clear_log()
                        self._trait_panel.update_values({}, {}, {})
                        chat.add_system_message("Chat history cleared.")
                    elif pending[0] == "rewind":
                        # Discard the partial response, then rewind
                        if self._messages and self._messages[-1]["role"] == "assistant":
                            self._messages.pop()
                        chat.rewind_last_assistant()
                        self._session.rewind()
                        chat.rewind()
                        chat.add_system_message("Rewound to before last message.")
                    elif pending[0] == "steer":
                        self._handle_steer(pending[1])
                    elif pending[0] == "probe":
                        self._handle_probe(pending[1])
                break
            if self._current_assistant_widget:
                chat.append_to_assistant(self._current_assistant_widget, token)
            self._gen_token_count += 1
            tokens_consumed += 1

        if tokens_consumed > 0:
            chat.scroll_to_bottom()

        if generating and self._gen_start_time > 0:
            elapsed = time.monotonic() - self._gen_start_time
            tok_per_sec = self._gen_token_count / elapsed if elapsed > 0.1 else 0.0
            self._last_tok_per_sec = tok_per_sec
            self._last_elapsed = elapsed

        if generating:
            self._vram_poll_counter += 1
            if self._vram_poll_counter >= 15:
                self._cached_vram_gb = _get_memory_gb(self._device_str)
                self._vram_poll_counter = 0
        elif self._vram_poll_counter != -1:
            self._cached_vram_gb = _get_memory_gb(self._device_str)
            self._vram_poll_counter = -1

        status_args = (generating, self._gen_token_count, self._session.config.max_new_tokens,
                       self._last_tok_per_sec, self._last_elapsed, self._prompt_token_count,
                       self._cached_vram_gb)
        if status_args != self._last_status_args:
            self._last_status_args = status_args
            chat.update_status(
                generating=generating,
                gen_tokens=self._gen_token_count,
                max_tokens=self._session.config.max_new_tokens,
                tok_per_sec=self._last_tok_per_sec,
                elapsed=self._last_elapsed,
                prompt_tokens=self._prompt_token_count,
                vram_gb=self._cached_vram_gb,
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
        panel = PANELS[self._focused_panel_idx]
        if panel == "trait-panel":
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
        self._session.unmonitor(probe_name)
        tp.set_active_probes(set(self._session._monitor.probe_names))

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
            self._alphas[name] = max(-0.3, min(0.3, self._alphas.get(name, 0.0) + delta))
            self._refresh_left_panel()

    def action_toggle_ortho(self) -> None:
        if self._ab_in_progress:
            return
        self._orthogonalize = not self._orthogonalize
        self._refresh_left_panel()

    def action_temp_down(self) -> None:
        if self._focused_panel_idx != 0:
            return
        self._session.config.temperature = max(0.0, round(self._session.config.temperature - 0.05, 2))
        self._refresh_gen_config()

    def action_temp_up(self) -> None:
        if self._focused_panel_idx != 0:
            return
        self._session.config.temperature = round(self._session.config.temperature + 0.05, 2)
        self._refresh_gen_config()

    def action_top_p_down(self) -> None:
        if self._focused_panel_idx != 0:
            return
        self._session.config.top_p = max(0.0, round(self._session.config.top_p - 0.05, 2))
        self._refresh_gen_config()

    def action_top_p_up(self) -> None:
        if self._focused_panel_idx != 0:
            return
        self._session.config.top_p = min(1.0, round(self._session.config.top_p + 0.05, 2))
        self._refresh_gen_config()

    def action_regenerate(self) -> None:
        if not self._messages:
            return
        if self._session._gen_state.is_generating.is_set():
            # Stop the current generation; _poll_generation will pick up
            # the pending action once the worker thread finishes.
            self._pending_action = ("regenerate",)
            self._session._gen_state.request_stop()
            return
        if self._messages[-1]["role"] == "assistant":
            self._messages.pop()
        self._start_generation()

    def action_ab_compare(self) -> None:
        chat = self._chat_panel
        if self._session._gen_state.is_generating.is_set():
            chat.add_system_message("Cannot A/B compare while generating. Stop generation first.")
            return
        if not self._last_prompt:
            chat.add_system_message("No previous prompt to compare.")
            return
        self._ab_in_progress = True
        chat.add_system_message("A/B comparison: generating unsteered response...")

        def _ab_generate():
            try:
                msgs = [{"role": "user", "content": self._last_prompt}]
                input_ids = build_chat_input(
                    self._session._tokenizer, msgs, self._session.config.system_prompt,
                ).to(self._session._device)

                # No alphas = no steering. Just generate.
                ab_state = GenerationState()
                generated = generate_steered(
                    self._session._model, self._session._tokenizer, input_ids,
                    self._session.config, ab_state,
                )
                unsteered = self._session._tokenizer.decode(generated, skip_special_tokens=True)

                self.call_from_thread(self._show_ab_result, unsteered)
            except Exception:
                self._ab_in_progress = False
                raise

        self.run_worker(_ab_generate, thread=True)

    def _show_ab_result(self, unsteered: str) -> None:
        self._ab_in_progress = False
        self._chat_panel.add_system_message(f"[Unsteered]: {unsteered}")

    def action_cycle_sort(self) -> None:
        self._trait_panel.cycle_sort()
