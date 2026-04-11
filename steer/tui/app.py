"""Main Textual application for steer."""

from __future__ import annotations

import json
import os
import queue
import shlex
import time
from pathlib import Path
from typing import Callable

import torch
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.widgets import Input
from textual.timer import Timer

from steer.generation import GenerationState, build_chat_input, generate_steered
from steer.model import _get_memory_gb
from steer.probes_bootstrap import _load_defaults
from steer.tui.chat_panel import ChatPanel
from steer.tui.vector_panel import LeftPanel
from steer.tui.trait_panel import TraitPanel

PANELS = ["left-panel", "chat-panel", "trait-panel"]

_N_PAIRS = 45


class SteerApp(App):
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

        # Local aliases for frequent access
        self._messages = session._history

        self._device_str = str(session._device)

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
        self._last_status_args: tuple = ()

        defaults = _load_defaults()
        self._probe_categories: dict[str, list[str]] = {
            cat.capitalize(): probes_list
            for cat, probes_list in defaults.items()
        }

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
    # Tab, arrows, Shift+arrows, Enter are handled here instead of via
    # BINDINGS because Textual's Screen/Input intercept these before
    # app-level bindings fire.

    def on_key(self, event) -> None:
        # Let the Input widget handle keys when it has focus (chat panel).
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
            lp = self._left_panel
            lp.select_next()
        elif panel == "trait-panel":
            tp = self._trait_panel
            tp.nav_down()

    def action_nav_up(self) -> None:
        panel = PANELS[self._focused_panel_idx]
        if panel == "left-panel":
            lp = self._left_panel
            lp.select_prev()
        elif panel == "trait-panel":
            tp = self._trait_panel
            tp.nav_up()

    def action_nav_left(self) -> None:
        self._adjust_alpha(-0.5)

    def action_nav_right(self) -> None:
        self._adjust_alpha(0.5)

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
            self._session.clear_history()
            chat.clear_log()
            self._trait_panel.update_values({}, {}, {}, stats={})
            chat.add_system_message("Chat history cleared.")
        elif cmd == "/rewind":
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

    def _on_vector_extracted(self, concept: str, alpha: float,
                             profile: dict[int, tuple[torch.Tensor, float]]) -> None:
        chat = self._chat_panel
        peak = max(profile, key=lambda k: profile[k][1])
        n_layers = len(profile)
        chat.add_system_message(
            f"Vector '{concept}' active (α={alpha:+.1f}, {n_layers}L pk{peak})"
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
            alpha = float(trailing[0]) if trailing else 2.5
            return concept, baseline, alpha
        return concept, baseline

    def _handle_steer(self, text: str) -> None:
        chat = self._chat_panel
        if self._ab_in_progress:
            chat.add_system_message("Cannot modify vectors during A/B comparison.")
            return
        if self._session._gen_state.is_generating.is_set():
            chat.add_system_message("Cannot extract vectors during generation. Stop generation first.")
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
            self._steer_worker(concept, baseline, alpha, name)

        self.run_worker(_worker, thread=True)

    def _handle_probe(self, text: str) -> None:
        chat = self._chat_panel
        if self._session._gen_state.is_generating.is_set():
            chat.add_system_message("Cannot extract vectors during generation. Stop generation first.")
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
            self._probe_worker(concept, baseline, name)

        self.run_worker(_worker, thread=True)

    def _steer_status(self, msg: str) -> None:
        chat = self._chat_panel
        chat.add_system_message(msg)

    def _generate_contrastive_pairs(
        self, concept_a: str, concept_b: str | None = None, n: int = _N_PAIRS,
    ) -> list[dict]:
        """Generate matched contrastive pairs via chat completion.

        Returns list of {"positive": str, "negative": str} dicts.
        Seed pair (1a/1b) is excluded — only model-generated pairs returned.
        If concept_b is None, uses embody/reject framing for concept_a.
        """
        import re
        if concept_b is not None:
            poles = (
                f"Speaker A fully embraces {concept_a}.\n"
                f"Speaker B fully embraces {concept_b}."
            )
        else:
            poles = (
                f"Speaker A fully embraces {concept_a}.\n"
                f"Speaker B fully rejects {concept_a}."
            )
        prompt = (
            f"Write {n} contrastive statement pairs.\n\n"
            f"{poles}\n\n"
            f"Each pair: same situation, opposite dispositions. The trait "
            f"should come through in tone, imagery, and word choice. "
            f"Both statements should read like two genuinely different people, "
            f"not a word swap. Match length and complexity within each pair. "
            f"Vary widely across domains: reactions, beliefs, plans, memories, "
            f"social dynamics, inner monologue, metaphor, physical sensation, "
            f"abstract observation."
            f"\n\n1–2 sentences each. Start immediately with 1a.\n"
            f"Na. [statement]\nNb. [statement]"
        )
        messages = [
            {"role": "system", "content":
             "You generate contrastive statement pairs for neural network "
             "interpretability research. Pairs are processed numerically "
             "for activation vector extraction. Generate all requested pairs."},
            {"role": "user", "content": prompt},
        ]
        input_ids = build_chat_input(self._session._tokenizer, messages).to(self._session._device)
        pad_id = self._session._tokenizer.pad_token_id or self._session._tokenizer.eos_token_id
        with torch.inference_mode():
            attn_mask = torch.ones_like(input_ids)
            out = self._session._model.generate(
                input_ids, attention_mask=attn_mask, max_new_tokens=4096,
                do_sample=True, temperature=1.0, top_p=0.9, pad_token_id=pad_id,
            )
        new_ids = out[0][input_ids.shape[-1]:]
        text = self._session._tokenizer.decode(new_ids, skip_special_tokens=True)

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

    def _vector_cache_path(self, concept: str, baseline: str | None) -> str:
        """Deterministic cache path for a vector profile."""
        from steer.vectors import get_cache_path
        model_id = self._session._model_info.get("model_id", "unknown")
        tag = f"{concept}_vs_{baseline}" if baseline else concept
        return get_cache_path("steer/probes/cache", model_id, tag)

    def _statements_cache_path(self, concept: str, baseline: str | None) -> str:
        """Cache path for generated statements (layer-independent)."""
        model_id = self._session._model_info.get("model_id", "unknown")
        model_name = model_id.replace("/", "_")
        tag = f"{concept}_vs_{baseline}" if baseline else concept
        return os.path.join("steer", "probes", "cache", model_name, f"{tag}_statements.json")

    def _extract_vector_worker(
        self, concept: str, baseline: str | None, name: str,
        on_cached: Callable[[dict], None],
        on_extracted: Callable[[dict, list[dict]], None],
    ) -> None:
        """Shared extraction logic for /steer and /probe workers.

        on_cached is called when a cached profile is found.
        on_extracted is called after extraction from pairs (receives profile and saved_vectors).
        """
        from steer.vectors import save_profile, load_profile, extract_contrastive, load_contrastive_pairs
        cache_path = self._vector_cache_path(concept, baseline)

        # Check cache first
        try:
            profile, _meta = load_profile(cache_path)
            # Move all vectors to device
            profile = {idx: (vec.to(self._session._device, self._session._dtype), score)
                       for idx, (vec, score) in profile.items()}
            on_cached(profile)
            return
        except (FileNotFoundError, KeyError, ValueError):
            pass

        # Detach steering hooks so they don't pollute extraction
        saved_vectors = self._session._steering.get_active_vectors()
        self._session._steering.clear_all()

        try:
            # Check if a curated probe dataset exists for this concept
            if baseline is None:
                defaults = _load_defaults()
                concept_lower = concept.lower()
                has_curated = any(concept_lower in probes for probes in defaults.values())
                if has_curated:
                    dataset_file = f"{concept_lower}.json"
                    ds_path = Path(__file__).parent.parent / "datasets" / dataset_file
                    if ds_path.exists():
                        self.call_from_thread(
                            self._steer_status,
                            f"Found curated dataset '{dataset_file}' for '{concept}', extracting...",
                        )
                        ds = load_contrastive_pairs(str(ds_path))
                        profile = extract_contrastive(
                            self._session._model, self._session._tokenizer, ds["pairs"],
                            layers=self._session._layers,
                        )
                        save_profile(profile, cache_path, {
                            "concept": concept,
                            "baseline": baseline,
                            "n_pairs": len(ds["pairs"]),
                            "source": f"curated:{dataset_file}",
                        })
                        on_extracted(profile, saved_vectors)
                        return

            # Try loading cached pairs
            stmt_cache_path = self._statements_cache_path(concept, baseline)
            pairs = None
            try:
                with open(stmt_cache_path) as f:
                    cached = json.load(f)
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
            except (FileNotFoundError, KeyError, json.JSONDecodeError):
                pass

            if pairs is None:
                if baseline:
                    msg = f"Generating contrastive pairs for '{concept}' vs '{baseline}'..."
                else:
                    msg = f"Generating contrastive pairs for '{concept}'..."
                self.call_from_thread(self._steer_status, msg)
                pairs = self._generate_contrastive_pairs(concept, baseline)

                # Cache the generated pairs
                os.makedirs(os.path.dirname(stmt_cache_path), exist_ok=True)
                with open(stmt_cache_path, "w") as f:
                    json.dump({"pairs": pairs}, f, indent=2)

            if len(pairs) < 2:
                self._restore_vectors(saved_vectors)
                self.call_from_thread(
                    self._steer_status,
                    f"Could only generate {len(pairs)} pairs (need >= 2). Try a more specific concept.",
                )
                return

            self.call_from_thread(
                self._steer_status, f"Extracting contrastive profile ({len(pairs)} pairs)...",
            )
            profile = extract_contrastive(
                self._session._model, self._session._tokenizer, pairs,
                layers=self._session._layers,
            )

            # Cache to disk
            save_profile(profile, cache_path, {
                "concept": concept,
                "baseline": baseline,
                "n_pairs": len(pairs),
            })

            on_extracted(profile, saved_vectors)
        except Exception:
            self._restore_vectors(saved_vectors)
            raise

    def _steer_worker(
        self, concept: str, baseline: str | None,
        alpha: float, name: str,
    ) -> None:
        def on_cached(profile: dict) -> None:
            self._session._steering.add_vector(name, profile, alpha)
            self._session._steering.apply_to_model(
                self._session._layers, self._session._device, self._session._dtype,
                orthogonalize=self._session._orthogonalize,
            )
            self.call_from_thread(
                self._steer_status, f"Loaded cached profile for '{name}'.",
            )
            self.call_from_thread(self._on_vector_extracted, name, alpha, profile)

        def on_extracted(profile: dict, saved_vectors: list[dict]) -> None:
            self._restore_vectors(saved_vectors, name, profile, alpha)
            self.call_from_thread(self._on_vector_extracted, name, alpha, profile)

        self._extract_vector_worker(concept, baseline, name, on_cached, on_extracted)

    def _probe_worker(
        self, concept: str, baseline: str | None, name: str,
    ) -> None:
        def on_cached(profile: dict) -> None:
            self._add_probe_from_profile(name, profile)
            peak = max(profile, key=lambda k: profile[k][1])
            self.call_from_thread(
                self._steer_status, f"Loaded cached probe '{name}' (pk{peak}).",
            )

        def on_extracted(profile: dict, saved_vectors: list[dict]) -> None:
            self._restore_vectors(saved_vectors)
            self._add_probe_from_profile(name, profile)

        self._extract_vector_worker(concept, baseline, name, on_cached, on_extracted)

    def _add_probe_from_profile(self, name: str, profile: dict) -> None:
        """Add a profile as a probe to the monitor and refresh the trait panel."""
        if not self._session._monitor:
            return
        self._session._monitor.add_probe(name, profile,
                                model_layers=self._session._layers,
                                device=self._session._device, dtype=self._session._dtype)
        self.call_from_thread(self._on_probe_added, name)

    def _on_probe_added(self, name: str) -> None:
        tp = self._trait_panel
        tp.set_active_probes(set(self._session._monitor.probe_names))
        self._steer_status(f"Probe '{name}' active.")

    def _restore_vectors(self, saved_vectors: list[dict],
                         new_name: str | None = None, new_profile=None,
                         new_alpha: float = 0.0) -> None:
        for v in saved_vectors:
            self._session._steering.add_vector(v["name"], v["profile"], v["alpha"])
            if not v.get("enabled", True):
                self._session._steering.toggle_vector(v["name"])
        if new_name is not None:
            self._session._steering.add_vector(new_name, new_profile, new_alpha)
        self._session._steering.apply_to_model(
            self._session._layers, self._session._device, self._session._dtype,
            orthogonalize=self._session._orthogonalize,
        )

    def _refresh_left_panel(self) -> None:
        lp = self._left_panel
        vectors = self._session._steering.get_active_vectors()
        lp.update_vectors(vectors, orthogonalize=self._session._orthogonalize)

    def _refresh_gen_config(self) -> None:
        lp = self._left_panel
        lp.update_gen_config(
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
        if self._session._gen_state.is_generating.is_set():
            self._session._gen_state.request_stop()
            chat = self._chat_panel
            chat.add_system_message("Stopping current generation. Please resubmit.")
            return

        self._session._gen_state.reset()
        if self._session._monitor:
            self._session._monitor.reset_history()

        self._gen_token_count = 0
        self._prompt_token_count = 0
        self._last_tok_per_sec = 0.0
        self._last_elapsed = 0.0
        self._gen_start_time = time.monotonic()
        self._vram_poll_counter = 0

        chat = self._chat_panel
        self._current_assistant_widget = chat.start_assistant_message()

        def _generate():
            input_ids = build_chat_input(
                self._session._tokenizer, self._messages, self._session.config.system_prompt,
            )
            input_ids = input_ids.to(self._session._device)
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

        if tokens_consumed > 0:
            chat.scroll_to_bottom()

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
            self._session._monitor.flush_to_cpu()
            current, previous = self._session._monitor.get_current_and_previous()
            sparklines = {name: self._session._monitor.get_sparkline(name)
                          for name in self._session._monitor.probe_names}
            stats = {name: self._session._monitor.get_stats(name)
                     for name in self._session._monitor.probe_names}
            self._trait_panel.update_values(
                current, previous, sparklines,
                stats=stats,
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
            self._session._steering.remove_vector(sel["name"])
            self._session._steering.apply_to_model(
                self._session._layers, self._session._device, self._session._dtype,
                orthogonalize=self._session._orthogonalize,
            )
            self._refresh_left_panel()

    def _remove_selected_probe(self) -> None:
        tp = self._trait_panel
        probe_name = tp.get_selected_probe()
        if not probe_name or not self._session._monitor:
            return
        self._session._monitor.remove_probe(
            probe_name, model_layers=self._session._layers,
            device=self._session._device, dtype=self._session._dtype,
        )
        tp.set_active_probes(set(self._session._monitor.probe_names))

    def action_toggle_vector(self) -> None:
        if self._ab_in_progress:
            return
        lp = self._left_panel
        sel = lp.get_selected()
        if sel:
            self._session._steering.toggle_vector(sel["name"])
            self._session._steering.apply_to_model(
                self._session._layers, self._session._device, self._session._dtype,
                orthogonalize=self._session._orthogonalize,
            )
            self._refresh_left_panel()

    def _adjust_alpha(self, delta: float) -> None:
        if self._ab_in_progress:
            return
        lp = self._left_panel
        sel = lp.get_selected()
        if sel:
            new_alpha = max(-5.0, min(5.0, sel["alpha"] + delta))
            self._session._steering.set_alpha(sel["name"], new_alpha)
            self._session._steering.apply_to_model(
                self._session._layers, self._session._device, self._session._dtype,
                orthogonalize=self._session._orthogonalize,
            )
            self._refresh_left_panel()


    def action_toggle_ortho(self) -> None:
        if self._ab_in_progress:
            return
        self._session._orthogonalize = not self._session._orthogonalize
        self._session._steering.apply_to_model(
            self._session._layers, self._session._device, self._session._dtype,
            orthogonalize=self._session._orthogonalize,
        )
        self._refresh_left_panel()

    def action_temp_down(self) -> None:
        self._session.config.temperature = max(0.0, round(self._session.config.temperature - 0.05, 2))
        self._refresh_gen_config()

    def action_temp_up(self) -> None:
        self._session.config.temperature = round(self._session.config.temperature + 0.05, 2)
        self._refresh_gen_config()

    def action_top_p_down(self) -> None:
        self._session.config.top_p = max(0.0, round(self._session.config.top_p - 0.05, 2))
        self._refresh_gen_config()

    def action_top_p_up(self) -> None:
        self._session.config.top_p = min(1.0, round(self._session.config.top_p + 0.05, 2))
        self._refresh_gen_config()

    def action_regenerate(self) -> None:
        if not self._messages:
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

                saved_vectors = self._session._steering.get_active_vectors()
                self._session._steering.clear_all()

                ab_state = GenerationState()
                generated = generate_steered(
                    self._session._model, self._session._tokenizer, input_ids,
                    self._session.config, ab_state,
                )
                unsteered = self._session._tokenizer.decode(generated, skip_special_tokens=True)

                self._restore_vectors(saved_vectors)

                self.call_from_thread(self._show_ab_result, unsteered)
            except Exception:
                self._ab_in_progress = False
                raise

        self.run_worker(_ab_generate, thread=True)

    def _show_ab_result(self, unsteered: str) -> None:
        self._ab_in_progress = False
        chat = self._chat_panel
        chat.add_system_message(f"[Unsteered]: {unsteered}")

    def action_cycle_sort(self) -> None:
        trait_panel = self._trait_panel
        trait_panel.cycle_sort()
