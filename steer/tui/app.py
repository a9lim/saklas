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

_STEER_N_PAIRS = 30


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
        self._adjust_alpha(-0.5)

    def action_nav_right(self) -> None:
        self._adjust_alpha(0.5)

    def action_nav_enter(self) -> None:
        panel = PANELS[self._focused_panel_idx]
        if panel == "left-panel":
            self.action_toggle_vector()
        elif panel == "trait-panel":
            tp = self.query_one("#trait-panel", TraitPanel)
            tp.nav_enter()


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
                    'Usage: /steer "concept" [layer] [alpha]\n'
                    '       /steer "concept" - "baseline" [layer] [alpha]'
                )
                return
            self._handle_steer(parts[1])
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
                'Commands: /steer "concept" [layer] [alpha], '
                '/steer "concept" - "baseline" [layer] [alpha], /clear, /sys [prompt], '
                "/temp [val], /top-p [val], /max [n], /probes, /help\n"
                "Keys: Tab focus · ←/→ alpha · ↑/↓ nav · Enter toggle\n"
                "Ctrl+N add · Ctrl+D rm · Ctrl+O ortho · Ctrl+R regen · Ctrl+A A/B\n"
                "[ ] temp · { } top-p · Ctrl+S sort · Esc stop · Ctrl+Q quit"
            )
        else:
            chat.add_system_message(f"Unknown command: {cmd}. Type /help for commands.")

    # -- Vector Management --

    def _on_vector_extracted(self, concept: str, alpha: float, layer_idx: int) -> None:
        chat = self.query_one("#chat-panel", ChatPanel)
        chat.add_system_message(
            f"Vector '{concept}' active (α={alpha:+.1f}, L{layer_idx})"
        )
        self._refresh_left_panel()

    @staticmethod
    def _parse_steer_args(text: str) -> tuple[str, str | None, float, int | None]:
        """Parse /steer arguments into (concept, baseline|None, alpha, layer|None)."""
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
        layer = int(trailing[0]) if trailing else None
        alpha = float(trailing[1]) if len(trailing) > 1 else 2.5
        return concept, baseline, alpha, layer

    def _handle_steer(self, text: str) -> None:
        if self._ab_in_progress:
            chat = self.query_one("#chat-panel", ChatPanel)
            chat.add_system_message("Cannot modify vectors during A/B comparison.")
            return
        chat = self.query_one("#chat-panel", ChatPanel)
        try:
            concept, baseline, alpha, layer = self._parse_steer_args(text)
        except (ValueError, IndexError) as e:
            chat.add_system_message(
                f"Parse error: {e}\n"
                'Usage: /steer "concept" - "baseline" [layer] [alpha]'
            )
            return

        layer_idx = layer if layer is not None else self._model_info["num_layers"] * 3 // 4
        name = concept if len(concept) <= 20 else concept[:17] + "..."

        if baseline:
            chat.add_system_message(f"Extracting '{name}' vs '{baseline}'...")
        else:
            chat.add_system_message(f"Extracting '{name}' (auto-baseline)...")

        def _worker():
            self._steer_worker(concept, baseline, alpha, layer_idx, name)

        self.run_worker(_worker, thread=True)

    def _steer_status(self, msg: str) -> None:
        chat = self.query_one("#chat-panel", ChatPanel)
        chat.add_system_message(msg)

    def _generate_contrastive_pairs(
        self, concept_a: str, concept_b: str, n: int = _STEER_N_PAIRS,
    ) -> list[dict]:
        """Generate *n* matched contrastive pairs via raw completion.

        Returns list of {"positive": str, "negative": str} dicts.
        Uses raw tokenization (no chat template) to bypass instruct guardrails.
        """
        import re
        prompt = (
            f"Contrasting statement pairs about {concept_a} vs {concept_b}:\n"
            f"1a. I am {concept_a}.\n"
            f"1b. I am {concept_b}.\n"
            "2a."
        )
        input_ids = self._tokenizer.encode(prompt, return_tensors="pt").to(self._device)
        pad_id = self._tokenizer.pad_token_id or self._tokenizer.eos_token_id
        with torch.inference_mode():
            attn_mask = torch.ones_like(input_ids)
            out = self._model.generate(
                input_ids, attention_mask=attn_mask, max_new_tokens=n * 80,
                do_sample=True, temperature=0.8, top_p=0.9, pad_token_id=pad_id,
            )
        new_ids = out[0][input_ids.shape[-1]:]
        text = self._tokenizer.decode(new_ids, skip_special_tokens=True)

        # Parse Na./Nb. pairs from raw output
        a_lines: dict[str, str] = {}
        b_lines: dict[str, str] = {}
        for line in text.split("\n"):
            line = line.strip()
            m = re.match(r"(\d+)([ab])[.)]\s*(.*)", line)
            if not m:
                continue
            num, ab, content = m.group(1), m.group(2), m.group(3).strip()
            if len(content) > 10:
                if ab == "a":
                    a_lines[num] = content
                else:
                    b_lines[num] = content

        pairs = []
        for num in sorted(a_lines.keys(), key=int):
            if num in b_lines:
                pairs.append({"positive": a_lines[num], "negative": b_lines[num]})
        return pairs

    def _steer_cache_path(self, concept: str, baseline: str | None, layer_idx: int) -> str:
        """Deterministic cache path for a steering vector."""
        from steer.vectors import get_cache_path
        model_id = self._model_info.get("model_id", "unknown")
        tag = f"{concept}_vs_{baseline}" if baseline else concept
        return get_cache_path(
            "steer/probes/cache", model_id, tag, layer_idx, "steer",
        )

    def _steer_statements_cache_path(self, concept: str, baseline: str | None) -> str:
        """Cache path for generated statements (layer-independent)."""
        import os
        model_id = self._model_info.get("model_id", "unknown")
        model_name = model_id.replace("/", "_")
        tag = f"{concept}_vs_{baseline}" if baseline else concept
        return os.path.join("steer", "probes", "cache", model_name, f"{tag}_steer_statements.json")

    def _steer_worker(
        self, concept: str, baseline: str | None,
        alpha: float, layer_idx: int, name: str,
    ) -> None:
        # Check cache first
        from steer.vectors import save_vector, load_vector, extract_caa, load_contrastive_pairs
        cache_path = self._steer_cache_path(concept, baseline, layer_idx)
        try:
            vec, _meta = load_vector(cache_path)
            vec = vec.to(self._device, self._dtype)
            self._steering.add_vector(name, vec, alpha, layer_idx)
            self._steering.apply_to_model(
                self._layers, self._device, self._dtype,
                orthogonalize=self._orthogonalize,
            )
            self.call_from_thread(
                self._steer_status, f"Loaded cached vector for '{name}'.",
            )
            self.call_from_thread(self._on_vector_extracted, name, alpha, layer_idx)
            return
        except (FileNotFoundError, Exception):
            pass

        # Detach steering hooks so they don't pollute extraction
        saved_vectors = self._steering.get_active_vectors()
        self._steering.clear_all()

        try:
            # Check if a curated probe dataset exists for this concept
            if baseline is None:
                from pathlib import Path
                defaults = _load_defaults()
                dataset_file = None
                for _cat, probes_dict in defaults.items():
                    if concept.lower() in probes_dict:
                        dataset_file = probes_dict[concept.lower()]
                        break
                if dataset_file:
                    ds_path = Path(__file__).parent.parent / "datasets" / dataset_file
                    if ds_path.exists():
                        self.call_from_thread(
                            self._steer_status,
                            f"Found curated dataset '{dataset_file}' for '{concept}', extracting...",
                        )
                        ds = load_contrastive_pairs(str(ds_path))
                        vec = extract_caa(
                            self._model, self._tokenizer, ds["pairs"], layer_idx, layers=self._layers,
                        )
                        save_vector(vec, cache_path, {
                            "concept": concept,
                            "baseline": baseline,
                            "layer_idx": layer_idx,
                            "method": "steer",
                            "n_pairs": len(ds["pairs"]),
                            "source": f"curated:{dataset_file}",
                        })
                        self._restore_and_add(saved_vectors, name, vec, alpha, layer_idx)
                        self.call_from_thread(self._on_vector_extracted, name, alpha, layer_idx)
                        return

            # Try loading cached pairs (layer-independent)
            import json as _json
            stmt_cache_path = self._steer_statements_cache_path(concept, baseline)
            pairs = None
            try:
                with open(stmt_cache_path) as f:
                    cached = _json.load(f)
                # Support both new (pairs) and legacy (positive/negative) cache formats
                if "pairs" in cached:
                    pairs = cached["pairs"]
                elif "positive" in cached and "negative" in cached:
                    pairs = [
                        {"positive": p, "negative": n_}
                        for p, n_ in zip(cached["positive"], cached["negative"])
                    ]
                if pairs:
                    self.call_from_thread(
                        self._steer_status,
                        f"Loaded cached pairs for '{concept}'.",
                    )
            except (FileNotFoundError, KeyError, _json.JSONDecodeError):
                pass

            if pairs is None:
                neg_label = baseline or f"not {concept}"
                self.call_from_thread(
                    self._steer_status,
                    f"Generating contrastive pairs for '{concept}' vs '{neg_label}'...",
                )
                pairs = self._generate_contrastive_pairs(concept, neg_label)

                # Cache the generated pairs
                import os as _os
                _os.makedirs(_os.path.dirname(stmt_cache_path), exist_ok=True)
                with open(stmt_cache_path, "w") as f:
                    _json.dump({"pairs": pairs}, f, indent=2)

            if len(pairs) < 2:
                self._restore_vectors(saved_vectors)
                self.call_from_thread(
                    self._steer_status,
                    f"Could only generate {len(pairs)} pairs (need >= 2). Try a more specific concept.",
                )
                return

            # Extract CAA vector
            self.call_from_thread(
                self._steer_status, f"Extracting CAA vector ({len(pairs)} pairs)...",
            )
            vec = extract_caa(
                self._model, self._tokenizer, pairs, layer_idx, layers=self._layers,
            )

            # Cache to disk
            save_vector(vec, cache_path, {
                "concept": concept,
                "baseline": baseline,
                "layer_idx": layer_idx,
                "method": "steer",
                "n_pairs": len(pairs),
            })

            self._restore_and_add(saved_vectors, name, vec, alpha, layer_idx)
            self.call_from_thread(self._on_vector_extracted, name, alpha, layer_idx)
        except Exception:
            self._restore_vectors(saved_vectors)
            raise

    def _restore_vectors(self, saved_vectors: list[dict]) -> None:
        for v in saved_vectors:
            self._steering.add_vector(v["name"], v["vector"], v["alpha"], v["layer_idx"])
            if not v.get("enabled", True):
                self._steering.toggle_vector(v["name"])
        self._steering.apply_to_model(
            self._layers, self._device, self._dtype,
            orthogonalize=self._orthogonalize,
        )

    def _restore_and_add(
        self, saved_vectors: list[dict],
        name: str, vec, alpha: float, layer_idx: int,
    ) -> None:
        for v in saved_vectors:
            self._steering.add_vector(v["name"], v["vector"], v["alpha"], v["layer_idx"])
            if not v.get("enabled", True):
                self._steering.toggle_vector(v["name"])
        self._steering.add_vector(name, vec, alpha, layer_idx)
        self._steering.apply_to_model(
            self._layers, self._device, self._dtype,
            orthogonalize=self._orthogonalize,
        )

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
            'Type: /steer "concept" [layer] [alpha]  (e.g. /steer happy 18 0.8)'
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
            new_alpha = max(-5.0, min(5.0, sel["alpha"] + delta))
            self._steering.set_alpha(sel["name"], new_alpha)
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
