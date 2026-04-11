"""SteerSession — unified backend for steer's programmatic API and TUI."""
from __future__ import annotations
import pathlib
import threading
import time
from typing import Iterator

import torch

from steer.datasource import DataSource
from steer.generation import GenerationConfig, GenerationState, build_chat_input, generate_steered
from steer.hooks import SteeringManager
from steer.model import load_model, get_layers, get_model_info
from steer.monitor import TraitMonitor
from steer.probes_bootstrap import bootstrap_probes, _load_defaults
from steer.results import GenerationResult, TokenEvent, ProbeReadings
from steer.vectors import (
    extract_contrastive,
    save_profile as _save_profile,
    load_profile as _load_profile,
    get_cache_path,
)


class SteerSession:
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

        self._steering = SteeringManager()
        self._orthogonalize = False
        self._steer_lock = threading.Lock()

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

        self._monitor = TraitMonitor(probe_profiles) if probe_profiles else TraitMonitor({})
        if probe_profiles:
            self._monitor.attach(self._layers, self._device, self._dtype)

    # -- State queries --

    @property
    def model_info(self) -> dict:
        return dict(self._model_info)

    @property
    def vectors(self) -> dict[str, dict]:
        return {v["name"]: v for v in self._steering.get_active_vectors()}

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

    def extract(self, source) -> dict[int, tuple[torch.Tensor, float]]:
        if isinstance(source, str):
            ds = DataSource.curated(source)
        elif isinstance(source, DataSource):
            ds = source
        elif isinstance(source, list):
            ds = DataSource.from_pairs(source)
        else:
            raise TypeError(f"Unsupported source type: {type(source)}")

        cache_path = get_cache_path(self._cache_dir, self._model_info.get("model_id", "unknown"), ds.name)
        try:
            profile, _meta = _load_profile(cache_path)
            profile = {idx: (vec.to(self._device, self._dtype), score)
                       for idx, (vec, score) in profile.items()}
            return profile
        except (FileNotFoundError, KeyError, ValueError):
            pass

        pairs = [{"positive": p, "negative": n} for p, n in ds.pairs]
        profile = extract_contrastive(
            self._model, self._tokenizer, pairs, layers=self._layers,
        )

        _save_profile(profile, cache_path, {
            "concept": ds.name,
            "n_pairs": len(ds.pairs),
        })

        return profile

    def load_profile(self, path: str) -> dict[int, tuple[torch.Tensor, float]]:
        profile, _meta = _load_profile(path)
        profile = {idx: (vec.to(self._device, self._dtype), score)
                   for idx, (vec, score) in profile.items()}
        return profile

    def save_profile(self, profile: dict, path: str, metadata: dict | None = None) -> None:
        _save_profile(profile, path, metadata or {})

    # -- Steering --

    def steer(self, name: str, profile: dict, alpha: float = 2.5) -> None:
        with self._steer_lock:
            self._steering.add_vector(name, profile, alpha)
            self._steering.apply_to_model(
                self._layers, self._device, self._dtype,
                orthogonalize=self._orthogonalize,
            )

    def set_alpha(self, name: str, alpha: float) -> None:
        with self._steer_lock:
            self._steering.set_alpha(name, alpha)
            self._steering.apply_to_model(
                self._layers, self._device, self._dtype,
                orthogonalize=self._orthogonalize,
            )

    def toggle(self, name: str) -> None:
        with self._steer_lock:
            self._steering.toggle_vector(name)
            self._steering.apply_to_model(
                self._layers, self._device, self._dtype,
                orthogonalize=self._orthogonalize,
            )

    def unsteer(self, name: str) -> None:
        with self._steer_lock:
            self._steering.remove_vector(name)
            self._steering.apply_to_model(
                self._layers, self._device, self._dtype,
                orthogonalize=self._orthogonalize,
            )

    def orthogonalize(self) -> None:
        with self._steer_lock:
            self._orthogonalize = True
            self._steering.apply_to_model(
                self._layers, self._device, self._dtype,
                orthogonalize=True,
            )

    def clear_vectors(self) -> None:
        with self._steer_lock:
            self._steering.clear_all()

    # -- Monitoring --

    def monitor(self, name: str, profile: dict | None = None) -> None:
        if profile is None:
            profile = self.extract(name)
        self._monitor.add_probe(
            name, profile,
            model_layers=self._layers,
            device=self._device, dtype=self._dtype,
        )

    def unmonitor(self, name: str) -> None:
        self._monitor.remove_probe(
            name, model_layers=self._layers,
            device=self._device, dtype=self._dtype,
        )

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

    def _prepare_input(self, input) -> tuple[list[dict], torch.Tensor]:
        """Normalize input to messages list and compute input_ids."""
        if isinstance(input, str):
            messages = list(self._history) + [{"role": "user", "content": input}]
        elif isinstance(input, list):
            messages = list(input)
        else:
            raise TypeError(f"Unsupported input type: {type(input)}")
        input_ids = build_chat_input(
            self._tokenizer, messages, self.config.system_prompt,
        ).to(self._device)
        return messages, input_ids

    def _build_readings(self) -> dict[str, ProbeReadings]:
        """Flush monitor and build ProbeReadings for all active probes."""
        readings: dict[str, ProbeReadings] = {}
        if not self._monitor or not self._monitor.probe_names:
            return readings
        self._monitor.flush_to_cpu()
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
                delta_per_tok = sum(deltas) / len(deltas)
            else:
                delta_per_tok = 0.0
            readings[name] = ProbeReadings(
                per_token=hist,
                mean=mean,
                std=std,
                min=stats["min"] if stats["min"] != float("inf") else 0.0,
                max=stats["max"] if stats["max"] != float("-inf") else 0.0,
                delta_per_tok=delta_per_tok,
            )
        return readings

    def _snapshot_vectors(self) -> dict[str, float]:
        """Snapshot active vectors and their alphas."""
        return {v["name"]: v["alpha"] for v in self._steering.get_active_vectors()
                if v.get("enabled", True)}

    # -- Generation: blocking --

    def generate(self, input, **kwargs) -> GenerationResult:
        """Blocking generation. Returns when complete."""
        if not self._gen_lock.acquire(blocking=False):
            raise RuntimeError("Generation already in progress")
        try:
            return self._generate_blocking(input)
        finally:
            self._gen_lock.release()

    def _generate_blocking(self, input) -> GenerationResult:
        messages, input_ids = self._prepare_input(input)
        self._gen_state.reset()
        if self._monitor:
            self._monitor.reset_history()

        vector_snapshot = self._snapshot_vectors()
        start = time.monotonic()

        generated_ids = generate_steered(
            self._model, self._tokenizer, input_ids,
            self.config, self._gen_state,
        )

        elapsed = time.monotonic() - start
        token_count = len(generated_ids)
        tok_per_sec = token_count / elapsed if elapsed > 0.1 else 0.0
        text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
        readings = self._build_readings()

        result = GenerationResult(
            text=text, tokens=generated_ids, token_count=token_count,
            tok_per_sec=tok_per_sec, elapsed=elapsed,
            readings=readings, vectors=vector_snapshot,
        )
        self._last_result = result

        # Append to history
        if isinstance(input, str):
            self._history.append({"role": "user", "content": input})
        if text.strip():
            self._history.append({"role": "assistant", "content": text})

        return result

    # -- Generation: streaming --

    def generate_stream(self, input, **kwargs) -> Iterator[TokenEvent]:
        """Streaming generation. Yields TokenEvent per token."""
        if not self._gen_lock.acquire(blocking=False):
            raise RuntimeError("Generation already in progress")
        try:
            yield from self._generate_streaming(input)
        finally:
            self._gen_lock.release()

    def _generate_streaming(self, input) -> Iterator[TokenEvent]:
        import queue as _queue

        messages, input_ids = self._prepare_input(input)
        self._gen_state.reset()
        if self._monitor:
            self._monitor.reset_history()

        vector_snapshot = self._snapshot_vectors()
        start = time.monotonic()
        token_events: list[TokenEvent] = []
        generated_ids: list[int] = []
        token_queue = self._gen_state.token_queue
        gen_done = threading.Event()
        gen_error: list[Exception] = []

        def _worker():
            try:
                ids = generate_steered(
                    self._model, self._tokenizer, input_ids,
                    self.config, self._gen_state,
                    on_token=lambda tok: token_queue.put(tok),
                )
                generated_ids.extend(ids)
            except Exception as e:
                gen_error.append(e)
            finally:
                gen_done.set()

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

        idx = 0
        while True:
            try:
                tok_str = token_queue.get(timeout=0.05)
            except _queue.Empty:
                if gen_done.is_set():
                    # Drain remaining
                    while True:
                        try:
                            tok_str = token_queue.get_nowait()
                        except _queue.Empty:
                            break
                        if tok_str is None:
                            break
                        readings_snap = None
                        if self._monitor and self._monitor.has_pending_data():
                            self._monitor.flush_to_cpu()
                            current, _ = self._monitor.get_current_and_previous()
                            readings_snap = dict(current) if current else None
                        event = TokenEvent(
                            text=tok_str,
                            token_id=generated_ids[idx] if idx < len(generated_ids) else -1,
                            index=idx, readings=readings_snap,
                        )
                        token_events.append(event)
                        yield event
                        idx += 1
                    break
                continue

            if tok_str is None:
                break

            readings_snap = None
            if self._monitor and self._monitor.has_pending_data():
                self._monitor.flush_to_cpu()
                current, _ = self._monitor.get_current_and_previous()
                readings_snap = dict(current) if current else None

            event = TokenEvent(
                text=tok_str,
                token_id=generated_ids[idx] if idx < len(generated_ids) else -1,
                index=idx, readings=readings_snap,
            )
            token_events.append(event)
            yield event
            idx += 1

        thread.join()

        if gen_error:
            raise gen_error[0]

        elapsed = time.monotonic() - start
        token_count = len(token_events)
        tok_per_sec = token_count / elapsed if elapsed > 0.1 else 0.0
        text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
        readings = self._build_readings()

        self._last_result = GenerationResult(
            text=text, tokens=list(generated_ids), token_count=token_count,
            tok_per_sec=tok_per_sec, elapsed=elapsed,
            readings=readings, vectors=vector_snapshot,
        )

        # Append to history
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
        if self._monitor:
            self._monitor.detach()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
