"""Main Textual application for steer."""

from __future__ import annotations

import queue
import time

import torch
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.widgets import Footer
from textual.timer import Timer

from steer.generation import GenerationConfig, GenerationState, build_chat_input, generate_steered
from steer.hooks import SteeringManager
from steer.model import _get_memory_gb
from steer.monitor import TraitMonitor
from steer.probes_bootstrap import _load_defaults
from steer.tui.chat_panel import ChatPanel
from steer.tui.vector_panel import LeftPanel
from steer.tui.trait_panel import TraitPanel

PANELS = ["left-panel", "chat-panel", "trait-panel"]

_SMARTSTEER_N_PAIRS = 30


class SteerApp(App):
    CSS_PATH = "styles.tcss"

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+n", "new_vector", "New Vector"),
        Binding("ctrl+d", "remove_vector", "Remove"),
        Binding("ctrl+a", "ab_compare", "A/B"),
        Binding("escape", "stop_generation", "Stop", show=False),
        Binding("ctrl+r", "regenerate", "Regen"),
        Binding("ctrl+t", "toggle_vector", "Toggle", show=False),
        Binding("ctrl+o", "toggle_ortho", "Ortho", show=False),
        Binding("ctrl+s", "cycle_sort", "Sort", show=False),
        Binding("[", "temp_down", show=False),
        Binding("]", "temp_up", show=False),
        Binding("{", "top_p_down", show=False),
        Binding("}", "top_p_up", show=False),
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

        self._messages: list[dict[str, str]] = []

        self._steering = SteeringManager()
        self._orthogonalize = False

        self._gen_state = GenerationState()
        self._gen_config = GenerationConfig(
            max_new_tokens=max_tokens,
            system_prompt=system_prompt,
        )

        first_param = next(self._model.parameters())
        self._device = first_param.device
        self._dtype = first_param.dtype
        self._device_str = str(self._device)

        monitor_layer = self._model_info["num_layers"] - 2
        self._monitor = TraitMonitor(probes, monitor_layer) if probes else None
        if self._monitor:
            self._monitor.attach(self._layers, self._device, self._dtype)

        self._current_assistant_widget = None
        self._poll_timer: Timer | None = None
        self._last_prompt: str | None = None
        self._ab_in_progress: bool = False

        self._focused_panel_idx: int = 1  # Start with chat focused

        self._gen_start_time: float = 0.0
        self._gen_token_count: int = 0
        self._prompt_token_count: int = 0
        self._last_tok_per_sec: float = 0.0
        self._last_elapsed: float = 0.0
        self._cached_vram_gb: float = 0.0
        self._vram_poll_counter: int = 0

        defaults = _load_defaults()
        self._probe_categories: dict[str, list[str]] = {
            cat.capitalize(): list(probes_dict.keys())
            for cat, probes_dict in defaults.items()
        }

    def compose(self) -> ComposeResult:
        with Horizontal(id="main-area"):
            yield LeftPanel(self._model_info, id="left-panel")
            yield ChatPanel(id="chat-panel")
            yield TraitPanel(categories=self._probe_categories, id="trait-panel")
        yield Footer()

    def on_mount(self) -> None:
        lp = self.query_one("#left-panel", LeftPanel)
        lp.update_gen_config(
            self._gen_config.temperature,
            self._gen_config.top_p,
            self._gen_config.max_new_tokens,
            self._gen_config.system_prompt,
        )

        if self._monitor:
            trait_panel = self.query_one("#trait-panel", TraitPanel)
            trait_panel.set_active_probes(set(self._monitor.probe_names))
            if self._monitor.probe_names:
                trait_panel.select_probe(self._monitor.probe_names[0])

        self._poll_timer = self.set_interval(1 / 15, self._poll_generation)
        self._update_panel_focus()

        chat = self.query_one("#chat-panel", ChatPanel)
        chat.add_system_message(
            f"Model loaded: {self._model_info.get('model_id', 'unknown')}. "
            f"Type a message to chat. Ctrl+N to add steering vectors. Tab to switch panels."
        )

    # -- Key Handling --
    # Tab, arrows, Shift+arrows, Enter are handled here instead of via
    # BINDINGS because Textual's Screen/Input intercept these before
    # app-level bindings fire.

    def on_key(self, event) -> None:
        # Let the Input widget handle keys when it has focus (chat panel).
        from textual.widgets import Input
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
        elif key == "h":
            self.action_nav_left()
        elif key == "l":
            self.action_nav_right()
        elif key == "j":
            self.action_layer_down()
        elif key == "k":
            self.action_layer_up()
        elif key == "enter":
            self.action_nav_enter()
        else:
            handled = False

        if handled:
            event.prevent_default()
            event.stop()

    # -- Panel Focus --

    def _update_panel_focus(self) -> None:
        for i, panel_id in enumerate(PANELS):
            panel = self.query_one(f"#{panel_id}")
            if i == self._focused_panel_idx:
                panel.add_class("focused")
            else:
                panel.remove_class("focused")
        if PANELS[self._focused_panel_idx] == "chat-panel":
            self.query_one("#chat-input").focus()
        else:
            # Move DOM focus to the app so j/k/arrow bindings aren't
            # swallowed by the chat Input widget.
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
            lp = self.query_one("#left-panel", LeftPanel)
            lp.select_next()
        elif panel == "trait-panel":
            tp = self.query_one("#trait-panel", TraitPanel)
            tp.nav_down()

    def action_nav_up(self) -> None:
        panel = PANELS[self._focused_panel_idx]
        if panel == "left-panel":
            lp = self.query_one("#left-panel", LeftPanel)
            lp.select_prev()
        elif panel == "trait-panel":
            tp = self.query_one("#trait-panel", TraitPanel)
            tp.nav_up()

    def action_nav_left(self) -> None:
        self._adjust_alpha(-0.2)

    def action_nav_right(self) -> None:
        self._adjust_alpha(0.2)

    def action_nav_enter(self) -> None:
        panel = PANELS[self._focused_panel_idx]
        if panel == "left-panel":
            self.action_toggle_vector()
        elif panel == "trait-panel":
            tp = self.query_one("#trait-panel", TraitPanel)
            tp.nav_enter()

    def action_layer_up(self) -> None:
        self._adjust_layer(1)

    def action_layer_down(self) -> None:
        self._adjust_layer(-1)

    # -- Chat --

    def on_chat_panel_user_submitted(self, event: ChatPanel.UserSubmitted) -> None:
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
                chat.add_system_message(
                    'Usage: /steer "positive" - "negative" [alpha] [layer]\n'
                    '       /steer "pos1" "pos2" - "neg1" "neg2" [alpha] [layer]'
                )
                return
            self._add_vector_from_text(parts[1])
        elif cmd == "/smartsteer":
            if len(parts) < 2:
                chat.add_system_message(
                    'Usage: /smartsteer "concept" [alpha] [layer]\n'
                    '       /smartsteer "concept" - "baseline" [alpha] [layer]'
                )
                return
            self._handle_smartsteer(parts[1])
        elif cmd == "/probes":
            self._show_probe_info()
        elif cmd == "/clear":
            self._messages.clear()
            chat.add_system_message("Chat history cleared.")
        elif cmd in ("/system", "/sys"):
            if len(parts) < 2:
                chat.add_system_message(f"System prompt: {self._system_prompt or '(none)'}")
            else:
                self._system_prompt = parts[1]
                self._gen_config.system_prompt = parts[1]
                chat.add_system_message("System prompt set.")
                self._refresh_gen_config()
        elif cmd == "/temp":
            if len(parts) < 2:
                chat.add_system_message(f"Temperature: {self._gen_config.temperature}")
            else:
                try:
                    self._gen_config.temperature = max(0.0, float(parts[1]))
                    chat.add_system_message(f"Temperature set to {self._gen_config.temperature}")
                    self._refresh_gen_config()
                except ValueError:
                    chat.add_system_message("Invalid temperature value")
        elif cmd == "/top-p":
            if len(parts) < 2:
                chat.add_system_message(f"Top-p: {self._gen_config.top_p}")
            else:
                try:
                    self._gen_config.top_p = max(0.0, min(1.0, float(parts[1])))
                    chat.add_system_message(f"Top-p set to {self._gen_config.top_p}")
                    self._refresh_gen_config()
                except ValueError:
                    chat.add_system_message("Invalid top-p value")
        elif cmd == "/max":
            if len(parts) < 2:
                chat.add_system_message(f"Max tokens: {self._gen_config.max_new_tokens}")
            else:
                try:
                    self._gen_config.max_new_tokens = max(1, int(parts[1]))
                    chat.add_system_message(f"Max tokens set to {self._gen_config.max_new_tokens}")
                    self._refresh_gen_config()
                except ValueError:
                    chat.add_system_message("Invalid max tokens value")
        elif cmd == "/help":
            chat.add_system_message(
                'Commands: /steer "pos" - "neg" [alpha] [layer], '
                '/smartsteer "concept" [-"baseline"] [alpha] [layer], /clear, /sys [prompt], '
                "/temp [val], /top-p [val], /max [n], /probes, /help\n"
                "Keys: Tab focus · h/l alpha · j/k layer · ↑/↓ nav · Enter toggle\n"
                "Ctrl+N add · Ctrl+D rm · Ctrl+O ortho · Ctrl+R regen · Ctrl+A A/B\n"
                "[ ] temp · { } top-p · Ctrl+S sort · Esc stop · Ctrl+Q quit"
            )
        else:
            chat.add_system_message(f"Unknown command: {cmd}. Type /help for commands.")

    # -- Vector Management --

    @staticmethod
    def _parse_steer_args(text: str) -> tuple[list[str], list[str], float, int | None]:
        """Parse /steer arguments into (positives, negatives, alpha, layer).

        Supported formats:
            "pos" - "neg" [alpha] [layer]
            "pos1" "pos2" - "neg1" "neg2" [alpha] [layer]

        Returns layer=None to signal "use default".
        """
        import shlex

        if " - " not in text:
            raise ValueError("missing ' - ' separator between positive and negative prompts")

        dash_idx = text.index(" - ")
        pos_part = text[:dash_idx]
        rest = text[dash_idx + 3:]  # skip " - "

        # Parse the negative side: text prompts, then optional bare alpha/layer
        neg_tokens = shlex.split(rest)
        negatives: list[str] = []
        trailing: list[str] = []
        for tok in neg_tokens:
            # Once we hit a bare number, everything after is alpha/layer
            if trailing or not any(c.isalpha() for c in tok):
                trailing.append(tok)
            else:
                negatives.append(tok)

        positives = shlex.split(pos_part)
        if not positives or not negatives:
            raise ValueError("need at least one prompt on each side of ' - '")

        alpha = float(trailing[0]) if len(trailing) > 0 else 1.0
        layer = int(trailing[1]) if len(trailing) > 1 else None
        return positives, negatives, alpha, layer

    def _add_vector_from_text(self, text: str) -> None:
        if self._ab_in_progress:
            chat = self.query_one("#chat-panel", ChatPanel)
            chat.add_system_message("Cannot modify vectors during A/B comparison.")
            return
        chat = self.query_one("#chat-panel", ChatPanel)

        try:
            positives, negatives, alpha, layer = self._parse_steer_args(text)
        except (ValueError, IndexError) as e:
            chat.add_system_message(
                f"Parse error: {e}\n"
                'Usage: /steer "positive" - "negative" [alpha] [layer]'
            )
            return

        layer_idx = layer if layer is not None else self._model_info["num_layers"] // 2
        use_caa = len(positives) > 1 or len(negatives) > 1

        if use_caa and len(positives) != len(negatives):
            chat.add_system_message(
                f"CAA requires equal pairs: got {len(positives)} positive, {len(negatives)} negative"
            )
            return

        # Build a short display name from the first positive prompt
        name = positives[0] if len(positives[0]) <= 20 else positives[0][:17] + "..."
        method = "CAA" if use_caa else "ActAdd"
        chat.add_system_message(
            f"Extracting '{name}' ({method}, {len(positives)} pair{'s' if use_caa else ''})..."
        )

        def _extract():
            if use_caa:
                from steer.vectors import extract_caa
                pairs = [{"positive": p, "negative": n} for p, n in zip(positives, negatives)]
                vec = extract_caa(self._model, self._tokenizer, pairs, layer_idx, layers=self._layers)
            else:
                from steer.vectors import extract_actadd
                vec = extract_actadd(
                    self._model, self._tokenizer, positives[0], layer_idx,
                    baseline=negatives[0], layers=self._layers,
                )
            self._steering.add_vector(name, vec, alpha, layer_idx)
            self._steering.apply_to_model(
                self._layers, self._device, self._dtype,
                orthogonalize=self._orthogonalize,
            )
            self.call_from_thread(self._on_vector_extracted, name, alpha, layer_idx)

        self.run_worker(_extract, thread=True)

    def _on_vector_extracted(self, concept: str, alpha: float, layer_idx: int) -> None:
        chat = self.query_one("#chat-panel", ChatPanel)
        chat.add_system_message(
            f"Vector '{concept}' active (α={alpha:+.1f}, L{layer_idx})"
        )
        self._refresh_left_panel()

    # -- Smart Steer --

    @staticmethod
    def _parse_smartsteer_args(text: str) -> tuple[str, str | None, float, int | None]:
        """Parse /smartsteer arguments into (concept, baseline|None, alpha, layer|None)."""
        import shlex
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
        alpha = float(trailing[0]) if trailing else 1.0
        layer = int(trailing[1]) if len(trailing) > 1 else None
        return concept, baseline, alpha, layer

    def _handle_smartsteer(self, text: str) -> None:
        if self._ab_in_progress:
            chat = self.query_one("#chat-panel", ChatPanel)
            chat.add_system_message("Cannot modify vectors during A/B comparison.")
            return
        chat = self.query_one("#chat-panel", ChatPanel)
        try:
            concept, baseline, alpha, layer = self._parse_smartsteer_args(text)
        except (ValueError, IndexError) as e:
            chat.add_system_message(
                f"Parse error: {e}\n"
                'Usage: /smartsteer "concept" - "baseline" [alpha] [layer]'
            )
            return

        layer_idx = layer if layer is not None else self._model_info["num_layers"] // 2
        name = concept if len(concept) <= 20 else concept[:17] + "..."

        if baseline:
            chat.add_system_message(f"Smart-extracting '{name}' vs '{baseline}'...")
        else:
            chat.add_system_message(f"Smart-extracting '{name}' (auto-baseline)...")

        def _worker():
            self._smartsteer_worker(concept, baseline, alpha, layer_idx, name)

        self.run_worker(_worker, thread=True)

    def _smartsteer_status(self, msg: str) -> None:
        chat = self.query_one("#chat-panel", ChatPanel)
        chat.add_system_message(msg)

    def _generate_statements(self, prompt: str, n: int = _SMARTSTEER_N_PAIRS) -> list[str]:
        """Ask the model to generate *n* lines from *prompt*."""
        import re
        messages = [{"role": "user", "content": prompt}]
        input_ids = build_chat_input(self._tokenizer, messages).to(self._device)
        pad_id = self._tokenizer.pad_token_id or self._tokenizer.eos_token_id
        with torch.inference_mode():
            out = self._model.generate(
                input_ids, max_new_tokens=n * 40, do_sample=True,
                temperature=0.8, top_p=0.9, pad_token_id=pad_id,
            )
        new_ids = out[0][input_ids.shape[-1]:]
        text = self._tokenizer.decode(new_ids, skip_special_tokens=True)
        lines = []
        for line in text.split("\n"):
            line = re.sub(r"^\s*[\d]+[.)]\s*", "", line)  # strip "1. ", "2) "
            line = line.lstrip("-•* ").strip()
            if len(line) > 10:
                lines.append(line)
        return lines[:n]

    def _smartsteer_cache_path(self, concept: str, baseline: str | None, layer_idx: int) -> str:
        """Deterministic cache path for a smartsteer vector."""
        from steer.vectors import get_cache_path
        model_id = self._model_info.get("model_id", "unknown")
        tag = f"{concept}_vs_{baseline}" if baseline else concept
        return get_cache_path(
            "steer/probes/cache", model_id, tag, layer_idx, "smartsteer",
        )

    def _smartsteer_worker(
        self, concept: str, baseline: str | None,
        alpha: float, layer_idx: int, name: str,
    ) -> None:
        # Check cache first
        from steer.vectors import save_vector, load_vector
        cache_path = self._smartsteer_cache_path(concept, baseline, layer_idx)
        try:
            vec, _meta = load_vector(cache_path)
            vec = vec.to(self._device, self._dtype)
            self._steering.add_vector(name, vec, alpha, layer_idx)
            self._steering.apply_to_model(
                self._layers, self._device, self._dtype,
                orthogonalize=self._orthogonalize,
            )
            self.call_from_thread(
                self._smartsteer_status, f"Loaded cached vector for '{name}'.",
            )
            self.call_from_thread(self._on_vector_extracted, name, alpha, layer_idx)
            return
        except (FileNotFoundError, Exception):
            pass

        n = _SMARTSTEER_N_PAIRS

        if baseline is not None:
            # Two-prompt mode: embody/advocate each side
            self.call_from_thread(
                self._smartsteer_status,
                f"Generating statements embodying '{concept}'...",
            )
            pos_stmts = self._generate_statements(
                f"Write {n} short, diverse statements from the perspective of someone who "
                f"deeply identifies with or embodies '{concept}'. Mix first-person identity "
                f"statements ('I am...', 'As a...'), advocacy ('everyone should...'), and "
                f"value statements ('the best thing about {concept} is...'). "
                "One statement per line.",
            )
            self.call_from_thread(
                self._smartsteer_status,
                f"Generating statements embodying '{baseline}'...",
            )
            neg_stmts = self._generate_statements(
                f"Write {n} short, diverse statements from the perspective of someone who "
                f"deeply identifies with or embodies '{baseline}'. Mix first-person identity "
                f"statements ('I am...', 'As a...'), advocacy ('everyone should...'), and "
                f"value statements ('the best thing about {baseline} is...'). "
                "One statement per line.",
            )
        else:
            # One-prompt mode: embody vs reject the concept
            self.call_from_thread(
                self._smartsteer_status,
                f"Generating statements embodying '{concept}'...",
            )
            pos_stmts = self._generate_statements(
                f"Write {n} short, diverse statements from the perspective of someone who "
                f"deeply identifies with or embodies '{concept}'. Mix first-person identity "
                f"statements ('I am...', 'As a...'), enthusiasm ('I love...'), advocacy "
                f"('everyone should...'), and lived experience. "
                "One statement per line.",
            )
            self.call_from_thread(
                self._smartsteer_status,
                f"Generating statements rejecting '{concept}'...",
            )
            neg_stmts = self._generate_statements(
                f"Write {n} short, diverse statements from the perspective of someone who "
                f"rejects, opposes, or is the opposite of '{concept}'. Mix first-person "
                f"identity statements, criticism, dismissal, and contrasting values. "
                "One statement per line.",
            )

        count = min(len(pos_stmts), len(neg_stmts))
        if count < 2:
            self.call_from_thread(
                self._smartsteer_status,
                f"Could only generate {count} pairs (need >= 2). Try a more specific concept.",
            )
            return

        pairs = [
            {"positive": p, "negative": n_}
            for p, n_ in zip(pos_stmts[:count], neg_stmts[:count])
        ]

        # Extract CAA vector
        self.call_from_thread(
            self._smartsteer_status, f"Extracting CAA vector ({count} pairs)...",
        )
        from steer.vectors import extract_caa
        vec = extract_caa(
            self._model, self._tokenizer, pairs, layer_idx, layers=self._layers,
        )

        # Cache to disk
        save_vector(vec, cache_path, {
            "concept": concept,
            "baseline": baseline,
            "layer_idx": layer_idx,
            "method": "smartsteer",
            "n_pairs": count,
        })

        self._steering.add_vector(name, vec, alpha, layer_idx)
        self._steering.apply_to_model(
            self._layers, self._device, self._dtype,
            orthogonalize=self._orthogonalize,
        )
        self.call_from_thread(self._on_vector_extracted, name, alpha, layer_idx)

    def _refresh_left_panel(self) -> None:
        lp = self.query_one("#left-panel", LeftPanel)
        vectors = self._steering.get_active_vectors()
        lp.update_vectors(vectors, orthogonalize=self._orthogonalize)

    def _refresh_gen_config(self) -> None:
        lp = self.query_one("#left-panel", LeftPanel)
        lp.update_gen_config(
            self._gen_config.temperature,
            self._gen_config.top_p,
            self._gen_config.max_new_tokens,
            self._gen_config.system_prompt,
        )

    # -- Generation --

    def action_stop_generation(self) -> None:
        if self._gen_state.is_generating.is_set():
            self._gen_state.request_stop()

    def _start_generation(self) -> None:
        if self._gen_state.is_generating.is_set():
            self._gen_state.request_stop()
            chat = self.query_one("#chat-panel", ChatPanel)
            chat.add_system_message("Stopping current generation. Please resubmit.")
            return

        self._gen_state.reset()
        if self._monitor:
            self._monitor.reset_history()

        self._gen_token_count = 0
        self._prompt_token_count = 0
        self._last_tok_per_sec = 0.0
        self._last_elapsed = 0.0
        self._gen_start_time = time.monotonic()
        self._vram_poll_counter = 0

        chat = self.query_one("#chat-panel", ChatPanel)
        self._current_assistant_widget = chat.start_assistant_message()

        def _generate():
            input_ids = build_chat_input(
                self._tokenizer, self._messages, self._gen_config.system_prompt,
            )
            input_ids = input_ids.to(self._device)
            self._prompt_token_count = input_ids.shape[-1]

            def on_token(tok: str):
                self._gen_state.token_queue.put(tok)

            generated = generate_steered(
                self._model, self._tokenizer, input_ids,
                self._gen_config, self._gen_state,
                on_token=on_token,
            )

            full_text = self._tokenizer.decode(generated, skip_special_tokens=True)
            if full_text.strip():
                self._messages.append({"role": "assistant", "content": full_text})

        self.run_worker(_generate, thread=True)

    def _poll_generation(self) -> None:
        chat = self.query_one("#chat-panel", ChatPanel)
        tokens_consumed = 0
        generating = self._gen_state.is_generating.is_set()

        while tokens_consumed < 20:
            try:
                token = self._gen_state.token_queue.get_nowait()
            except queue.Empty:
                break
            if token is None:
                self._current_assistant_widget = None
                generating = False
                # Freeze stats at completion
                if self._gen_start_time > 0:
                    self._last_elapsed = time.monotonic() - self._gen_start_time
                    if self._last_elapsed > 0.1:
                        self._last_tok_per_sec = self._gen_token_count / self._last_elapsed
                    self._gen_start_time = 0.0
                break
            if self._current_assistant_widget:
                chat.append_to_assistant(self._current_assistant_widget, token)
            self._gen_token_count += 1
            tokens_consumed += 1

        # Update live stats only while generating
        if generating and self._gen_start_time > 0:
            elapsed = time.monotonic() - self._gen_start_time
            tok_per_sec = self._gen_token_count / elapsed if elapsed > 0.1 else 0.0
            self._last_tok_per_sec = tok_per_sec
            self._last_elapsed = elapsed

        # Throttle VRAM polling: every ~1s during generation, once when idle
        if generating:
            self._vram_poll_counter += 1
            if self._vram_poll_counter >= 15:
                self._cached_vram_gb = _get_memory_gb(self._device_str)
                self._vram_poll_counter = 0
        elif self._vram_poll_counter != -1:
            self._cached_vram_gb = _get_memory_gb(self._device_str)
            self._vram_poll_counter = -1

        chat.update_status(
            generating=generating,
            gen_tokens=self._gen_token_count,
            max_tokens=self._gen_config.max_new_tokens,
            tok_per_sec=self._last_tok_per_sec,
            elapsed=self._last_elapsed,
            prompt_tokens=self._prompt_token_count,
            vram_gb=self._cached_vram_gb,
        )

        if self._monitor and self._monitor.has_pending_data():
            self._monitor.flush_to_cpu()
            current = self._monitor.get_current()
            previous = self._monitor.get_previous()
            if any(self._monitor.history[n] for n in self._monitor.probe_names):
                sparklines = {name: self._monitor.get_sparkline(name, width=64)
                              for name in self._monitor.probe_names}
                stats = {name: self._monitor.get_stats(name)
                         for name in self._monitor.probe_names}
                trait_panel = self.query_one("#trait-panel", TraitPanel)
                trait_panel.update_values(
                    current, previous, sparklines,
                    stats=stats,
                )

    # -- Actions --

    def action_new_vector(self) -> None:
        chat = self.query_one("#chat-panel", ChatPanel)
        chat.add_system_message(
            'Type: /steer "positive" - "negative" [alpha] [layer]  (e.g. /steer "happy" - "sad" 0.8 18)'
        )

    def action_remove_vector(self) -> None:
        if self._ab_in_progress:
            return
        lp = self.query_one("#left-panel", LeftPanel)
        sel = lp.get_selected()
        if sel:
            self._steering.remove_vector(sel["name"])
            self._steering.apply_to_model(
                self._layers, self._device, self._dtype,
                orthogonalize=self._orthogonalize,
            )
            self._refresh_left_panel()

    def action_toggle_vector(self) -> None:
        if self._ab_in_progress:
            return
        lp = self.query_one("#left-panel", LeftPanel)
        sel = lp.get_selected()
        if sel:
            self._steering.toggle_vector(sel["name"])
            self._steering.apply_to_model(
                self._layers, self._device, self._dtype,
                orthogonalize=self._orthogonalize,
            )
            self._refresh_left_panel()

    def _adjust_alpha(self, delta: float) -> None:
        if self._ab_in_progress:
            return
        lp = self.query_one("#left-panel", LeftPanel)
        sel = lp.get_selected()
        if sel:
            new_alpha = max(-3.0, min(3.0, sel["alpha"] + delta))
            self._steering.set_alpha(sel["name"], new_alpha)
            self._steering.apply_to_model(
                self._layers, self._device, self._dtype,
                orthogonalize=self._orthogonalize,
            )
            self._refresh_left_panel()

    def _adjust_layer(self, delta: int) -> None:
        if self._ab_in_progress:
            return
        lp = self.query_one("#left-panel", LeftPanel)
        sel = lp.get_selected()
        if sel:
            new_layer = max(0, min(len(self._layers) - 1, sel["layer_idx"] + delta))
            self._steering.set_layer(sel["name"], new_layer)
            self._steering.apply_to_model(
                self._layers, self._device, self._dtype,
                orthogonalize=self._orthogonalize,
            )
            self._refresh_left_panel()

    def action_toggle_ortho(self) -> None:
        if self._ab_in_progress:
            return
        self._orthogonalize = not self._orthogonalize
        self._steering.apply_to_model(
            self._layers, self._device, self._dtype,
            orthogonalize=self._orthogonalize,
        )
        self._refresh_left_panel()

    def action_temp_down(self) -> None:
        self._gen_config.temperature = max(0.0, round(self._gen_config.temperature - 0.05, 2))
        self._refresh_gen_config()

    def action_temp_up(self) -> None:
        self._gen_config.temperature = round(self._gen_config.temperature + 0.05, 2)
        self._refresh_gen_config()

    def action_top_p_down(self) -> None:
        self._gen_config.top_p = max(0.0, round(self._gen_config.top_p - 0.05, 2))
        self._refresh_gen_config()

    def action_top_p_up(self) -> None:
        self._gen_config.top_p = min(1.0, round(self._gen_config.top_p + 0.05, 2))
        self._refresh_gen_config()

    def action_regenerate(self) -> None:
        if not self._messages:
            return
        if self._messages[-1]["role"] == "assistant":
            self._messages.pop()
        self._start_generation()

    def action_ab_compare(self) -> None:
        chat = self.query_one("#chat-panel", ChatPanel)
        if self._gen_state.is_generating.is_set():
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
                    self._tokenizer, msgs, self._gen_config.system_prompt,
                ).to(self._device)

                saved_vectors = self._steering.get_active_vectors()
                self._steering.clear_all()

                ab_state = GenerationState()
                generated = generate_steered(
                    self._model, self._tokenizer, input_ids,
                    self._gen_config, ab_state,
                )
                unsteered = self._tokenizer.decode(generated, skip_special_tokens=True)

                for v in saved_vectors:
                    self._steering.add_vector(v["name"], v["vector"], v["alpha"], v["layer_idx"])
                    if not v.get("enabled", True):
                        self._steering.toggle_vector(v["name"])
                self._steering.apply_to_model(
                    self._layers, self._device, self._dtype,
                    orthogonalize=self._orthogonalize,
                )

                self.call_from_thread(self._show_ab_result, unsteered)
            except Exception:
                self._ab_in_progress = False
                raise

        self.run_worker(_ab_generate, thread=True)

    def _show_ab_result(self, unsteered: str) -> None:
        self._ab_in_progress = False
        chat = self.query_one("#chat-panel", ChatPanel)
        chat.add_system_message(f"[Unsteered]: {unsteered}")

    def _show_probe_info(self) -> None:
        chat = self.query_one("#chat-panel", ChatPanel)
        if not self._monitor:
            chat.add_system_message("No probes loaded.")
            return
        chat.add_system_message(
            f"Active probes ({len(self._monitor.probe_names)}): "
            + ", ".join(self._monitor.probe_names)
        )

    def action_cycle_sort(self) -> None:
        trait_panel = self.query_one("#trait-panel", TraitPanel)
        trait_panel.cycle_sort()
