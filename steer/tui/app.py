"""Main Textual application for steer."""

from __future__ import annotations

import queue

import torch
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Static, Footer, Input
from textual.timer import Timer

from steer.generation import GenerationConfig, GenerationState, build_chat_input, generate_steered
from steer.hooks import SteeringManager
from steer.monitor import TraitMonitor
from steer.tui.chat_panel import ChatPanel
from steer.tui.vector_panel import VectorPanel, ControlsPanel
from steer.tui.trait_panel import TraitPanel

class SteerApp(App):
    CSS_PATH = "styles.tcss"

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+n", "new_vector", "New Vector"),
        Binding("ctrl+d", "remove_vector", "Remove Vector"),
        Binding("ctrl+p", "add_probe", "Add Probe"),
        Binding("ctrl+a", "ab_compare", "A/B Compare"),
        Binding("ctrl+r", "regenerate", "Regenerate"),
        Binding("ctrl+t", "toggle_vector", "Toggle Vector"),
        Binding("left", "alpha_down", "Alpha -", show=False),
        Binding("right", "alpha_up", "Alpha +", show=False),
        Binding("up", "layer_up", "Layer +", show=False),
        Binding("down", "layer_down", "Layer -", show=False),
        Binding("o", "toggle_ortho", "Ortho", show=False),
        Binding("s", "cycle_sort", "Sort", show=False),
    ]

    def __init__(
        self,
        model,
        tokenizer,
        layers,
        model_info: dict,
        probes: dict[str, torch.Tensor],
        system_prompt: str | None = None,
        max_tokens: int = 512,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._model = model
        self._tokenizer = tokenizer
        self._layers = layers
        self._model_info = model_info
        self._system_prompt = system_prompt
        self._max_tokens = max_tokens

        # Chat history
        self._messages: list[dict[str, str]] = []

        # Steering
        self._steering = SteeringManager()
        self._orthogonalize = False

        # Generation state
        self._gen_state = GenerationState()
        self._gen_config = GenerationConfig(
            max_new_tokens=max_tokens,
            system_prompt=system_prompt,
        )

        # Cache device/dtype to avoid repeated parameter iteration
        first_param = next(self._model.parameters())
        self._device = first_param.device
        self._dtype = first_param.dtype

        # Monitor
        monitor_layer = self._model_info["num_layers"] - 2
        self._monitor = TraitMonitor(probes, monitor_layer) if probes else None
        if self._monitor:
            self._monitor.attach(self._layers, self._device, self._dtype)

        # TUI state
        self._current_assistant_widget = None
        self._poll_timer: Timer | None = None
        self._last_prompt: str | None = None

    def compose(self) -> ComposeResult:
        info = self._model_info
        vram = info.get("vram_used_gb", 0)
        header_text = (
            f"  steer v0.1  |  {info.get('model_id', '?')}  "
            f"|  VRAM: {vram:.1f} GB  |  L:{info['num_layers']}"
        )
        yield Static(header_text, id="header")
        with Horizontal(id="main-area"):
            yield ChatPanel(id="chat-panel")
            with Vertical(id="right-column"):
                yield VectorPanel(id="vector-panel")
                yield ControlsPanel(id="controls-panel")
                yield TraitPanel(id="trait-panel")
        yield Footer()

    def on_mount(self) -> None:
        # Set up trait panel
        if self._monitor:
            trait_panel = self.query_one("#trait-panel", TraitPanel)
            trait_panel.set_active_probes(set(self._monitor.probe_names))
            # Select first probe for sparkline
            if self._monitor.probe_names:
                trait_panel.select_probe(self._monitor.probe_names[0])

        # Start poll timer for token consumption + monitor updates (~15 FPS)
        self._poll_timer = self.set_interval(1 / 15, self._poll_generation)

        # Welcome message
        chat = self.query_one("#chat-panel", ChatPanel)
        chat.add_system_message(
            f"Model loaded: {self._model_info.get('model_id', 'unknown')}. "
            f"Type a message to chat. Ctrl+N to add steering vectors."
        )

    def on_chat_panel_user_submitted(self, event: ChatPanel.UserSubmitted) -> None:
        # Check for special commands
        text = event.text
        if text.startswith("/"):
            self._handle_command(text)
            return

        self._last_prompt = text
        self._messages.append({"role": "user", "content": text})
        self._start_generation()

    def _handle_command(self, text: str) -> None:
        chat = self.query_one("#chat-panel", ChatPanel)
        parts = text.split(maxsplit=1)
        cmd = parts[0].lower()

        if cmd == "/steer":
            if len(parts) < 2:
                chat.add_system_message("Usage: /steer <concept> [alpha] [layer]")
                return
            self._add_vector_from_text(parts[1])
        elif cmd == "/probes":
            self._show_probe_info()
        elif cmd == "/clear":
            self._messages.clear()
            chat.add_system_message("Chat history cleared.")
        elif cmd == "/system":
            if len(parts) < 2:
                chat.add_system_message(f"Current system prompt: {self._system_prompt or '(none)'}")
            else:
                self._system_prompt = parts[1]
                self._gen_config.system_prompt = parts[1]
                chat.add_system_message(f"System prompt set.")
        elif cmd == "/temp":
            if len(parts) < 2:
                chat.add_system_message(f"Temperature: {self._gen_config.temperature}")
            else:
                try:
                    self._gen_config.temperature = float(parts[1])
                    chat.add_system_message(f"Temperature set to {self._gen_config.temperature}")
                except ValueError:
                    chat.add_system_message("Invalid temperature value")
        else:
            chat.add_system_message(f"Unknown command: {cmd}")

    def _add_vector_from_text(self, text: str) -> None:
        chat = self.query_one("#chat-panel", ChatPanel)
        parts = text.split()
        concept = parts[0]
        alpha = float(parts[1]) if len(parts) > 1 else 1.0
        layer_idx = int(parts[2]) if len(parts) > 2 else self._model_info["num_layers"] // 2

        chat.add_system_message(f"Extracting steering vector for '{concept}'...")

        def _extract():
            from steer.vectors import extract_actadd
            vec = extract_actadd(self._model, self._tokenizer, concept, layer_idx)
            self._steering.add_vector(concept, vec, alpha, layer_idx)
            self._steering.apply_to_model(
                self._layers, self._device, self._dtype,
                orthogonalize=self._orthogonalize,
            )
            self.call_from_thread(self._on_vector_extracted, concept, alpha, layer_idx)

        self.run_worker(_extract, thread=True)

    def _on_vector_extracted(self, concept: str, alpha: float, layer_idx: int) -> None:
        chat = self.query_one("#chat-panel", ChatPanel)
        chat.add_system_message(
            f"Vector '{concept}' active (alpha={alpha:+.1f}, layer={layer_idx})"
        )
        self._refresh_vector_panel()

    def _refresh_vector_panel(self) -> None:
        vp = self.query_one("#vector-panel", VectorPanel)
        vectors = self._steering.get_active_vectors()
        vp.update_vectors(vectors)

        cp = self.query_one("#controls-panel", ControlsPanel)
        sel = vp.get_selected()
        cp.update_for_vector(sel, self._model_info["num_layers"])

    def _start_generation(self) -> None:
        if self._gen_state.is_generating.is_set():
            self._gen_state.request_stop()
            return

        self._gen_state.reset()
        if self._monitor:
            self._monitor.reset_history()

        chat = self.query_one("#chat-panel", ChatPanel)
        self._current_assistant_widget = chat.start_assistant_message()

        def _generate():
            input_ids = build_chat_input(
                self._tokenizer, self._messages, self._gen_config.system_prompt,
            )
            input_ids = input_ids.to(self._device)

            def on_token(tok: str):
                self._gen_state.token_queue.put(tok)

            generated = generate_steered(
                self._model, self._tokenizer, input_ids,
                self._gen_config, self._gen_state,
                on_token=on_token,
            )

            # Decode full response for history
            full_text = self._tokenizer.decode(generated, skip_special_tokens=True)
            self._messages.append({"role": "assistant", "content": full_text})

        self.run_worker(_generate, thread=True)

    def _poll_generation(self) -> None:
        """Called ~15 times/sec by the timer. Drains token queue and updates monitor."""
        # Drain token queue
        chat = self.query_one("#chat-panel", ChatPanel)
        tokens_consumed = 0
        while tokens_consumed < 20:  # Cap per poll to keep TUI responsive
            try:
                token = self._gen_state.token_queue.get_nowait()
            except queue.Empty:
                break
            if token is None:
                self._current_assistant_widget = None
                break
            if self._current_assistant_widget:
                chat.append_to_assistant(self._current_assistant_widget, token)
            tokens_consumed += 1

        # Update trait monitor display
        if self._monitor and self._monitor._buf_idx > 0:
            self._monitor.flush_to_cpu()  # flush once — moves GPU buffer to CPU history
            current = self._monitor.get_current()    # reads from CPU history, no GPU sync
            previous = self._monitor.get_previous()  # reads from CPU history, no GPU sync
            sparklines = {}
            for name in self._monitor.probe_names:
                sparklines[name] = self._monitor.get_sparkline(name, width=64)

            trait_panel = self.query_one("#trait-panel", TraitPanel)
            trait_panel.update_values(current, previous, sparklines)

    def _show_probe_info(self) -> None:
        chat = self.query_one("#chat-panel", ChatPanel)
        if not self._monitor:
            chat.add_system_message("No probes loaded.")
            return
        chat.add_system_message(
            f"Active probes ({len(self._monitor.probe_names)}): "
            + ", ".join(self._monitor.probe_names)
        )

    # -- Actions --

    def action_new_vector(self) -> None:
        chat = self.query_one("#chat-panel", ChatPanel)
        chat.add_system_message(
            "Type: /steer <concept> [alpha] [layer]  (e.g. /steer happy 0.8 18)"
        )

    def action_remove_vector(self) -> None:
        vp = self.query_one("#vector-panel", VectorPanel)
        sel = vp.get_selected()
        if sel:
            self._steering.remove_vector(sel["name"])
            self._steering.apply_to_model(
                self._layers, self._device, self._dtype,
                orthogonalize=self._orthogonalize,
            )
            self._refresh_vector_panel()

    def action_toggle_vector(self) -> None:
        vp = self.query_one("#vector-panel", VectorPanel)
        sel = vp.get_selected()
        if sel:
            self._steering.toggle_vector(sel["name"])
            self._steering.apply_to_model(
                self._layers, self._device, self._dtype,
                orthogonalize=self._orthogonalize,
            )
            self._refresh_vector_panel()

    def action_alpha_up(self) -> None:
        self._adjust_alpha(0.1)

    def action_alpha_down(self) -> None:
        self._adjust_alpha(-0.1)

    def _adjust_alpha(self, delta: float) -> None:
        vp = self.query_one("#vector-panel", VectorPanel)
        sel = vp.get_selected()
        if sel:
            new_alpha = max(-3.0, min(3.0, sel["alpha"] + delta))
            self._steering.set_alpha(sel["name"], new_alpha)
            self._steering.apply_to_model(
                self._layers, self._device, self._dtype,
                orthogonalize=self._orthogonalize,
            )
            self._refresh_vector_panel()

    def action_layer_up(self) -> None:
        self._adjust_layer(1)

    def action_layer_down(self) -> None:
        self._adjust_layer(-1)

    def _adjust_layer(self, delta: int) -> None:
        vp = self.query_one("#vector-panel", VectorPanel)
        sel = vp.get_selected()
        if sel:
            new_layer = max(0, min(len(self._layers) - 1, sel["layer_idx"] + delta))
            self._steering.set_layer(sel["name"], new_layer, self._layers)
            self._steering.apply_to_model(
                self._layers, self._device, self._dtype,
                orthogonalize=self._orthogonalize,
            )
            self._refresh_vector_panel()

    def action_toggle_ortho(self) -> None:
        self._orthogonalize = not self._orthogonalize
        self._steering.apply_to_model(
            self._layers, self._device, self._dtype,
            orthogonalize=self._orthogonalize,
        )
        self._refresh_vector_panel()

    def action_regenerate(self) -> None:
        if not self._messages:
            return
        # Remove last assistant message if present
        if self._messages and self._messages[-1]["role"] == "assistant":
            self._messages.pop()
        self._start_generation()

    def action_ab_compare(self) -> None:
        chat = self.query_one("#chat-panel", ChatPanel)
        if not self._last_prompt:
            chat.add_system_message("No previous prompt to compare.")
            return
        chat.add_system_message("A/B comparison: generating unsteered response...")

        # Save current steering state, clear hooks, generate, restore
        def _ab_generate():
            # Build input from the last user message only
            msgs = [{"role": "user", "content": self._last_prompt}]
            input_ids = build_chat_input(
                self._tokenizer, msgs, self._gen_config.system_prompt,
            ).to(self._device)

            # Temporarily clear steering
            saved_vectors = self._steering.get_active_vectors()
            self._steering.clear_all()

            from steer.generation import GenerationState
            ab_state = GenerationState()
            generated = generate_steered(
                self._model, self._tokenizer, input_ids,
                self._gen_config, ab_state,
            )
            unsteered = self._tokenizer.decode(generated, skip_special_tokens=True)

            # Restore steering
            for v in saved_vectors:
                self._steering.add_vector(v["name"], v["vector"], v["alpha"], v["layer_idx"])
                if not v.get("enabled", True):
                    self._steering.toggle_vector(v["name"])
            self._steering.apply_to_model(
                self._layers, self._device, self._dtype,
                orthogonalize=self._orthogonalize,
            )

            self.call_from_thread(
                self._show_ab_result, unsteered,
            )

        self.run_worker(_ab_generate, thread=True)

    def _show_ab_result(self, unsteered: str) -> None:
        chat = self.query_one("#chat-panel", ChatPanel)
        chat.add_system_message(f"[Unsteered]: {unsteered}")

    def action_add_probe(self) -> None:
        chat = self.query_one("#chat-panel", ChatPanel)
        chat.add_system_message(
            "Type a concept to probe, e.g.: /steer curious  (it becomes both vector and probe)"
        )

    def action_cycle_sort(self) -> None:
        trait_panel = self.query_one("#trait-panel", TraitPanel)
        trait_panel.cycle_sort()
