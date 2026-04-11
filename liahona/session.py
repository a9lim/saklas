"""LiahonaSession — unified backend for liahona's programmatic API and TUI."""
from __future__ import annotations
import json
import os
import pathlib
import re
import threading
import time
from typing import Callable, Iterator

import torch

from liahona.datasource import DataSource
from liahona.generation import GenerationConfig, GenerationState, build_chat_input, generate_steered, supports_thinking
from liahona.hooks import SteeringManager
from liahona.model import load_model, get_layers, get_model_info
from liahona.monitor import TraitMonitor
from liahona.probes_bootstrap import bootstrap_probes, bootstrap_layer_means, _load_defaults
from liahona.results import GenerationResult, TokenEvent, ProbeReadings
from liahona.vectors import (
    extract_contrastive,
    save_profile as _save_profile,
    load_profile as _load_profile,
    load_contrastive_pairs,
    get_cache_path,
)

_N_PAIRS = 45


class LiahonaSession:
    """Unified backend for activation steering, monitoring, and generation.

    Vectors are registered via steer() and applied per-generation via the
    alphas parameter on generate()/generate_stream(). No persistent hooks
    live on the model between generations.
    """

    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        quantize: str | None = None,
        probes: list[str] | None = None,
        system_prompt: str | None = None,
        max_tokens: int = 1024,
        cache_dir: str | None = None,
    ):
        self._model, self._tokenizer = load_model(model_id, quantize=quantize, device=device)
        self._layers = get_layers(self._model)
        self._model_info = get_model_info(self._model, self._tokenizer)

        first_param = next(self._model.parameters())
        self._device = first_param.device
        self._dtype = first_param.dtype

        self._cache_dir = cache_dir or str(
            pathlib.Path(__file__).parent / "probes" / "cache"
        )

        self.config = GenerationConfig(
            max_new_tokens=max_tokens,
            system_prompt=system_prompt,
        )

        # Vector registry: name -> profile. No alphas, no hooks.
        self._profiles: dict[str, dict[int, tuple[torch.Tensor, float]]] = {}

        # Transient steering manager — used only during generation
        self._steering = SteeringManager()

        self._gen_lock = threading.Lock()
        self._gen_state = GenerationState()

        self._history: list[dict[str, str]] = []
        self._last_result: GenerationResult | None = None

        # Bootstrap probes
        all_categories = ["emotion", "personality", "safety", "cultural", "gender"]
        if probes is None:
            probe_categories = all_categories
        elif not probes:
            probe_categories = []
        else:
            probe_categories = probes

        probe_profiles: dict[str, dict] = {}
        if probe_categories:
            probe_profiles = bootstrap_probes(
                self._model, self._tokenizer, self._layers, self._model_info,
                categories=probe_categories, cache_dir=self._cache_dir,
            )

        self._layer_means: dict[int, torch.Tensor] = {}
        if probe_profiles:
            self._layer_means = bootstrap_layer_means(
                self._model, self._tokenizer, self._layers, self._model_info,
                cache_dir=self._cache_dir,
            )

        self._monitor = TraitMonitor(probe_profiles, self._layer_means) if probe_profiles else TraitMonitor({})

    # -- State queries --

    @property
    def model_info(self) -> dict:
        return dict(self._model_info)

    @property
    def vectors(self) -> dict[str, dict[int, tuple[torch.Tensor, float]]]:
        """Registered steering vector profiles: name -> profile."""
        return dict(self._profiles)

    @property
    def probes(self) -> dict[str, dict]:
        return {name: {"profile": self._monitor._raw_profiles[name]}
                for name in self._monitor.probe_names}

    @property
    def history(self) -> list[dict[str, str]]:
        return list(self._history)

    @property
    def last_result(self) -> GenerationResult | None:
        return self._last_result

    # -- Extraction --

    def _vector_cache_path(self, concept: str, baseline: str | None = None) -> str:
        model_id = self._model_info.get("model_id", "unknown")
        tag = f"{concept}_vs_{baseline}" if baseline else concept
        return get_cache_path(self._cache_dir, model_id, tag)

    def _statements_cache_path(self, concept: str, baseline: str | None = None) -> str:
        model_id = self._model_info.get("model_id", "unknown")
        model_name = model_id.replace("/", "_")
        tag = f"{concept}_vs_{baseline}" if baseline else concept
        return os.path.join(self._cache_dir, model_name, f"{tag}_statements.json")

    def generate_pairs(
        self,
        concept: str,
        baseline: str | None = None,
        n: int = _N_PAIRS,
    ) -> list[tuple[str, str]]:
        """Generate contrastive pairs using the loaded model.

        Uses the model's own generation to produce matched statement pairs
        for a concept. Returns list of (positive, negative) tuples.
        """
        if baseline is not None:
            poles = (
                f"Speaker A fully embraces {concept}.\n"
                f"Speaker B fully embraces {baseline}."
            )
        else:
            poles = (
                f"Speaker A fully embraces {concept}.\n"
                f"Speaker B fully rejects {concept}."
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
        input_ids = build_chat_input(self._tokenizer, messages).to(self._device)
        pad_id = self._tokenizer.pad_token_id or self._tokenizer.eos_token_id
        with torch.inference_mode():
            attn_mask = torch.ones_like(input_ids)
            out = self._model.generate(
                input_ids, attention_mask=attn_mask, max_new_tokens=4096,
                do_sample=True, temperature=1.0, top_p=0.9, pad_token_id=pad_id,
            )
        new_ids = out[0][input_ids.shape[-1]:]
        text = self._tokenizer.decode(new_ids, skip_special_tokens=True)

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
                pairs.append((a_lines[num], b_lines[num]))
        return pairs

    def extract(
        self,
        source,
        baseline: str | None = None,
        on_progress: Callable[[str], None] | None = None,
    ) -> dict[int, tuple[torch.Tensor, float]]:
        """Extract a steering vector profile.

        Full pipeline: cache check -> curated dataset -> statement cache ->
        generate pairs -> extract contrastive -> save to cache.

        No steering hooks are on the model between generations, so extraction
        never needs to save/restore steering state.

        Args:
            source: concept name (str), list of (positive, negative) tuples,
                    or a DataSource instance.
            baseline: optional baseline concept for contrastive extraction.
                      Only used when source is a string.
            on_progress: optional callback for progress messages.
        """
        def _progress(msg: str) -> None:
            if on_progress:
                on_progress(msg)

        # Normalize source
        if isinstance(source, str):
            concept = source
        elif isinstance(source, DataSource):
            concept = source.name
        elif isinstance(source, list):
            concept = "custom"
        else:
            raise TypeError(f"Unsupported source type: {type(source)}")

        # For DataSource or raw pairs, skip the full pipeline — just extract
        if isinstance(source, (DataSource, list)):
            if isinstance(source, list):
                ds = DataSource.from_pairs(source)
            else:
                ds = source
            cache_path = get_cache_path(
                self._cache_dir, self._model_info.get("model_id", "unknown"), ds.name,
            )
            try:
                profile, _meta = _load_profile(cache_path)
                profile = {idx: (vec.to(self._device, self._dtype), score)
                           for idx, (vec, score) in profile.items()}
                _progress(f"Loaded cached profile for '{ds.name}'.")
                return profile
            except (FileNotFoundError, KeyError, ValueError):
                pass

            _progress(f"Extracting profile ({len(ds.pairs)} pairs)...")
            pairs = [{"positive": p, "negative": n} for p, n in ds.pairs]
            profile = extract_contrastive(
                self._model, self._tokenizer, pairs, layers=self._layers,
            )
            _save_profile(profile, cache_path, {
                "concept": ds.name, "n_pairs": len(ds.pairs),
            })
            return profile

        # String source — full pipeline
        cache_path = self._vector_cache_path(concept, baseline)

        # 1. Check vector cache
        try:
            profile, _meta = _load_profile(cache_path)
            profile = {idx: (vec.to(self._device, self._dtype), score)
                       for idx, (vec, score) in profile.items()}
            _progress(f"Loaded cached profile for '{concept}'.")
            return profile
        except (FileNotFoundError, KeyError, ValueError):
            pass

        # 2. Check for curated dataset
        if baseline is None:
            defaults = _load_defaults()
            concept_lower = concept.lower()
            has_curated = any(concept_lower in probes for probes in defaults.values())
            if has_curated:
                dataset_file = f"{concept_lower}.json"
                ds_path = pathlib.Path(__file__).parent / "datasets" / dataset_file
                if ds_path.exists():
                    _progress(f"Found curated dataset '{dataset_file}', extracting...")
                    ds = load_contrastive_pairs(str(ds_path))
                    profile = extract_contrastive(
                        self._model, self._tokenizer, ds["pairs"],
                        layers=self._layers,
                    )
                    _save_profile(profile, cache_path, {
                        "concept": concept, "baseline": baseline,
                        "n_pairs": len(ds["pairs"]),
                        "source": f"curated:{dataset_file}",
                    })
                    return profile

        # 3. Check statement cache
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
                _progress(f"Loaded cached pairs for '{concept}'.")
        except (FileNotFoundError, KeyError, json.JSONDecodeError):
            pass

        # 4. Generate pairs if needed
        if pairs is None:
            suffix = f" vs '{baseline}'" if baseline else ""
            _progress(f"Generating contrastive pairs for '{concept}'{suffix}...")
            raw_pairs = self.generate_pairs(concept, baseline)
            pairs = [{"positive": p, "negative": n} for p, n in raw_pairs]

            os.makedirs(os.path.dirname(stmt_cache_path), exist_ok=True)
            with open(stmt_cache_path, "w") as f:
                json.dump({"pairs": pairs}, f, indent=2)

        if len(pairs) < 2:
            raise ValueError(
                f"Could only generate {len(pairs)} pairs (need >= 2). "
                f"Try a more specific concept."
            )

        # 5. Extract
        _progress(f"Extracting contrastive profile ({len(pairs)} pairs)...")
        profile = extract_contrastive(
            self._model, self._tokenizer, pairs, layers=self._layers,
        )

        _save_profile(profile, cache_path, {
            "concept": concept, "baseline": baseline,
            "n_pairs": len(pairs),
        })

        return profile

    def load_profile(self, path: str) -> dict[int, tuple[torch.Tensor, float]]:
        profile, _meta = _load_profile(path)
        profile = {idx: (vec.to(self._device, self._dtype), score)
                   for idx, (vec, score) in profile.items()}
        return profile

    def save_profile(self, profile: dict, path: str, metadata: dict | None = None) -> None:
        _save_profile(profile, path, metadata or {})

    # -- Steering (vector registry) --

    def steer(self, name: str, profile: dict[int, tuple[torch.Tensor, float]]) -> None:
        """Register a steering vector. Applied during generate() via alphas."""
        self._profiles[name] = profile

    def unsteer(self, name: str) -> None:
        """Remove a steering vector from the registry."""
        self._profiles.pop(name, None)

    def _apply_steering(self, alphas: dict[str, float], orthogonalize: bool = False) -> None:
        """Compose and attach steering hooks for a generation call."""
        self._steering.clear_all()
        for name, alpha in alphas.items():
            if name not in self._profiles:
                raise KeyError(f"No vector registered for '{name}'")
            self._steering.add_vector(name, self._profiles[name], alpha)
        self._steering.apply_to_model(
            self._layers, self._device, self._dtype,
            orthogonalize=orthogonalize,
        )

    def _clear_steering(self) -> None:
        """Remove all steering hooks from the model."""
        self._steering.clear_all()

    # -- Monitoring --

    def monitor(self, name: str, profile: dict | None = None) -> None:
        if profile is None:
            profile = self.extract(name)
        if not self._layer_means:
            self._layer_means = bootstrap_layer_means(
                self._model, self._tokenizer, self._layers, self._model_info,
                cache_dir=self._cache_dir,
            )
            self._monitor._layer_means = self._layer_means
        self._monitor.add_probe(name, profile)

    def unmonitor(self, name: str) -> None:
        self._monitor.remove_probe(name)

    # -- History --

    def rewind(self) -> None:
        if self._history and self._history[-1]["role"] == "assistant":
            self._history.pop()
        if self._history and self._history[-1]["role"] == "user":
            self._history.pop()

    def clear_history(self) -> None:
        self._history.clear()
        if self._monitor:
            self._monitor.reset_history()

    # -- Generation helpers --

    def _prepare_input(self, input, raw: bool = False, thinking: bool = False) -> tuple[list[dict], torch.Tensor]:
        if isinstance(input, str):
            messages = list(self._history) + [{"role": "user", "content": input}]
        elif isinstance(input, list):
            messages = list(input)
        else:
            raise TypeError(f"Unsupported input type: {type(input)}")
        if raw and isinstance(input, str):
            input_ids = self._tokenizer.encode(
                input, return_tensors="pt",
            ).to(self._device)
        else:
            input_ids = build_chat_input(
                self._tokenizer, messages, self.config.system_prompt,
                thinking=thinking,
            ).to(self._device)
        return messages, input_ids

    def _build_readings(self) -> dict[str, ProbeReadings]:
        readings: dict[str, ProbeReadings] = {}
        if not self._monitor or not self._monitor.probe_names:
            return readings
        for name in self._monitor.probe_names:
            stats = self._monitor.get_stats(name)
            count = stats["count"]
            if count == 0:
                continue
            mean = stats["sum"] / count
            variance = max(0.0, stats["sum_sq"] / count - mean ** 2)
            std = variance ** 0.5
            hist = list(self._monitor.history.get(name, []))
            if len(hist) >= 2:
                deltas = [abs(hist[i] - hist[i-1]) for i in range(1, len(hist))]
                delta_per_gen = sum(deltas) / len(deltas)
            else:
                delta_per_gen = 0.0
            readings[name] = ProbeReadings(
                per_token=hist, mean=mean, std=std,
                min=stats["min"] if stats["min"] != float("inf") else 0.0,
                max=stats["max"] if stats["max"] != float("-inf") else 0.0,
                delta_per_tok=delta_per_gen,
            )
        return readings

    # -- Generation: blocking --

    def generate(
        self,
        input,
        alphas: dict[str, float] | None = None,
        orthogonalize: bool = False,
        raw: bool = False,
        thinking: bool = False,
    ) -> GenerationResult:
        """Blocking generation.

        Args:
            input: prompt string or list of message dicts.
            alphas: steering vector alphas to apply. Keys must match
                    registered vector names. None = no steering.
            orthogonalize: Gram-Schmidt orthogonalize vectors before applying.
            raw: skip chat template, tokenize input string directly.
            thinking: enable thinking/reasoning mode for models that support it.
        """
        if not self._gen_lock.acquire(blocking=False):
            raise RuntimeError("Generation already in progress")
        try:
            return self._generate_blocking(input, alphas, orthogonalize, raw, thinking)
        finally:
            self._gen_lock.release()

    def _generate_blocking(self, input, alphas, orthogonalize, raw=False, thinking=False) -> GenerationResult:
        use_thinking = thinking and supports_thinking(self._tokenizer)
        messages, input_ids = self._prepare_input(input, raw=raw, thinking=use_thinking)
        self._gen_state.reset()
        if self._monitor:
            self._monitor.reset_history()

        vector_snapshot = dict(alphas) if alphas else {}

        if alphas:
            self._apply_steering(alphas, orthogonalize=orthogonalize)

        try:
            start = time.monotonic()
            generated_ids = generate_steered(
                self._model, self._tokenizer, input_ids,
                self.config, self._gen_state, thinking=use_thinking,
            )
            elapsed = time.monotonic() - start
        finally:
            if alphas:
                self._clear_steering()

        token_count = len(generated_ids)
        tok_per_sec = token_count / elapsed if elapsed > 0.1 else 0.0
        # Strip thinking tokens — only decode the response portion
        response_ids = generated_ids[self._gen_state.thinking_end_idx:]
        text = self._tokenizer.decode(response_ids, skip_special_tokens=True)

        if self._monitor and self._monitor.probe_names and text.strip():
            self._monitor.measure(
                self._model, self._tokenizer, self._layers, text,
                device=self._device,
            )
        readings = self._build_readings()

        result = GenerationResult(
            text=text, tokens=generated_ids, token_count=token_count,
            tok_per_sec=tok_per_sec, elapsed=elapsed,
            readings=readings, vectors=vector_snapshot,
        )
        self._last_result = result

        if isinstance(input, str):
            self._history.append({"role": "user", "content": input})
        if text.strip():
            self._history.append({"role": "assistant", "content": text})

        return result

    # -- Generation: streaming --

    def generate_stream(
        self,
        input,
        alphas: dict[str, float] | None = None,
        orthogonalize: bool = False,
        raw: bool = False,
        thinking: bool = False,
    ) -> Iterator[TokenEvent]:
        """Streaming generation. Yields TokenEvent per token.

        Args:
            input: prompt string or list of message dicts.
            alphas: steering vector alphas to apply. None = no steering.
            orthogonalize: Gram-Schmidt orthogonalize vectors before applying.
            raw: skip chat template, tokenize input string directly.
            thinking: enable thinking/reasoning mode for models that support it.
        """
        if not self._gen_lock.acquire(blocking=False):
            raise RuntimeError("Generation already in progress")
        try:
            yield from self._generate_streaming(input, alphas, orthogonalize, raw, thinking)
        finally:
            self._gen_lock.release()

    def _generate_streaming(self, input, alphas, orthogonalize, raw=False, thinking=False) -> Iterator[TokenEvent]:
        import queue as _queue

        use_thinking = thinking and supports_thinking(self._tokenizer)
        messages, input_ids = self._prepare_input(input, raw=raw, thinking=use_thinking)
        self._gen_state.reset()
        if self._monitor:
            self._monitor.reset_history()

        vector_snapshot = dict(alphas) if alphas else {}

        if alphas:
            self._apply_steering(alphas, orthogonalize=orthogonalize)

        token_events: list[TokenEvent] = []
        generated_ids: list[int] = []
        token_queue = self._gen_state.token_queue
        gen_done = threading.Event()
        gen_error: list[Exception] = []
        start = time.monotonic()

        def _worker():
            try:
                ids = generate_steered(
                    self._model, self._tokenizer, input_ids,
                    self.config, self._gen_state,
                    on_token=lambda tok, thinking: token_queue.put((tok, thinking)),
                    thinking=use_thinking,
                )
                generated_ids.extend(ids)
            except Exception as e:
                gen_error.append(e)
            finally:
                gen_done.set()

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

        idx = 0
        try:
            while True:
                try:
                    item = token_queue.get(timeout=0.05)
                except _queue.Empty:
                    if gen_done.is_set():
                        while True:
                            try:
                                item = token_queue.get_nowait()
                            except _queue.Empty:
                                break
                            if item is None:
                                break
                            tok_str, is_thinking = item
                            event = TokenEvent(
                                text=tok_str,
                                token_id=generated_ids[idx] if idx < len(generated_ids) else -1,
                                index=idx, readings=None, thinking=is_thinking,
                            )
                            token_events.append(event)
                            yield event
                            idx += 1
                        break
                    continue

                if item is None:
                    break

                tok_str, is_thinking = item
                event = TokenEvent(
                    text=tok_str,
                    token_id=generated_ids[idx] if idx < len(generated_ids) else -1,
                    index=idx, readings=None, thinking=is_thinking,
                )
                token_events.append(event)
                yield event
                idx += 1

            thread.join()
        finally:
            if alphas:
                self._clear_steering()

        if gen_error:
            raise gen_error[0]

        elapsed = time.monotonic() - start
        token_count = len(token_events)
        tok_per_sec = token_count / elapsed if elapsed > 0.1 else 0.0
        # Strip thinking tokens — only decode the response portion
        response_ids = generated_ids[self._gen_state.thinking_end_idx:]
        text = self._tokenizer.decode(response_ids, skip_special_tokens=True)

        if self._monitor and self._monitor.probe_names and text.strip():
            self._monitor.measure(
                self._model, self._tokenizer, self._layers, text,
                device=self._device,
            )
        readings = self._build_readings()

        self._last_result = GenerationResult(
            text=text, tokens=list(generated_ids), token_count=token_count,
            tok_per_sec=tok_per_sec, elapsed=elapsed,
            readings=readings, vectors=vector_snapshot,
        )

        if isinstance(input, str):
            self._history.append({"role": "user", "content": input})
        if text.strip():
            self._history.append({"role": "assistant", "content": text})

    # -- Generation control --

    def stop(self) -> None:
        self._gen_state.request_stop()

    # -- Lifecycle --

    def close(self) -> None:
        self._steering.clear_all()
        self._profiles.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
