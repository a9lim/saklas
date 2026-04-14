"""SaklasSession — unified backend for saklas's programmatic API and TUI."""
from __future__ import annotations
import json
import logging
import pathlib
import queue
import re
import threading
import time
from typing import Callable, Iterator

import torch

from saklas.datasource import DataSource
from saklas.generation import GenerationConfig, GenerationState, build_chat_input, generate_steered, supports_thinking
from saklas.hooks import HiddenCapture, SteeringManager
from saklas.model import load_model, get_layers, get_model_info
from saklas.monitor import TraitMonitor
from saklas.packs import PackFormatError, PackMetadata, hash_file
from saklas.paths import concept_dir, safe_model_id
from saklas.probes_bootstrap import bootstrap_probes, bootstrap_layer_means
from saklas.results import GenerationResult, TokenEvent, ProbeReadings
from saklas.vectors import (
    extract_contrastive,
    save_profile as _save_profile,
    load_profile as _load_profile,
    load_contrastive_pairs,
)

_log = logging.getLogger(__name__)

_N_PAIRS = 45
PROBE_CATEGORIES = ["affect", "epistemic", "alignment", "register", "social_stance", "cultural"]
_BATCH_SIZE = 9
MIN_ELAPSED_FOR_RATE = 0.1

_PAIR_RE = re.compile(r"(?:\d+|N)\s*([ab])[.)]\s*(.*)", re.IGNORECASE)
_SLUG_RE = re.compile(r"[^a-z0-9]+")
BIPOLAR_SEP = "."


def _slug(s: str) -> str:
    """Normalize a single pole label to `[a-z0-9_]`.

    Collapses any non-alphanumeric run to `_`. Never produces the bipolar
    separator `.` — that is reserved for joining two slugged poles in
    `canonical_concept_name`.
    """
    return _SLUG_RE.sub("_", s.strip().lower()).strip("_")


def canonical_concept_name(concept: str, baseline: str | None = None) -> str:
    """Return the canonical on-disk name for a concept.

    Monopolar: `_slug(concept)`.
    Bipolar:   `f"{_slug(concept)}.{_slug(baseline)}"`.

    If `baseline` is None and `concept` already contains the bipolar
    separator `.`, the input is treated as a pre-composed bipolar name
    and each side is slugged independently. This makes `/steer happy.sad`
    and `/steer happy - sad` resolve to the same cache entry.
    """
    if baseline is None:
        if BIPOLAR_SEP in concept:
            pos, neg = concept.split(BIPOLAR_SEP, 1)
            return f"{_slug(pos)}{BIPOLAR_SEP}{_slug(neg)}"
        return _slug(concept)
    return f"{_slug(concept)}{BIPOLAR_SEP}{_slug(baseline)}"

_DOMAIN_SEEDS = [
    "specific facts, lore, history, or knowledge unique to this concept",
    "physical traits, mannerisms, sensory details, or observable behaviors",
    "how this concept relates to others socially — alliances, conflicts, status",
    "inner life: self-image, desires, fears, or motivations distinctive to this concept",
    "concrete routines, rituals, habitats, or environmental interactions",
]


class ConcurrentGenerationError(RuntimeError):
    """Raised when a generation call is made while another is in progress."""
    pass


class SaklasSession:
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

        if cache_dir is not None:
            _log.warning(
                "SaklasSession(cache_dir=...) is deprecated; paths now come from ~/.saklas/. "
                "Set SAKLAS_HOME env var to override."
            )

        self.config = GenerationConfig(
            max_new_tokens=max_tokens,
            system_prompt=system_prompt,
        )

        # Vector registry: name -> profile. No alphas, no hooks.
        self._profiles: dict[str, dict[int, tuple[torch.Tensor, float]]] = {}

        # Transient steering manager — used only during generation
        self._steering = SteeringManager()

        # Transient per-token hidden-state capture — attached around
        # generate_steered when probes are active so scoring happens
        # without a second forward pass.
        self._capture = HiddenCapture()

        self._gen_lock = threading.Lock()
        self._gen_state = GenerationState()

        self._history: list[dict[str, str]] = []
        self._last_result: GenerationResult | None = None
        self._last_per_token_scores: dict[str, list[float]] | None = None

        # Bootstrap probes
        probe_categories = PROBE_CATEGORIES if probes is None else probes

        probe_profiles: dict[str, dict] = {}
        if probe_categories:
            probe_profiles = bootstrap_probes(
                self._model, self._tokenizer, self._layers, self._model_info,
                probe_categories,
            )

        self._layer_means: dict[int, torch.Tensor] = {}
        if probe_profiles:
            self._layer_means = bootstrap_layer_means(
                self._model, self._tokenizer, self._layers, self._model_info,
            )

        self._monitor = TraitMonitor(probe_profiles, self._layer_means) if probe_profiles else TraitMonitor({})

    # -- State queries --

    @property
    def model_info(self) -> dict:
        return dict(self._model_info)

    @property
    def model_id(self) -> str:
        return self._model_info.get("model_id", "unknown")

    def has_vector(self, name: str) -> bool:
        return name in self._profiles

    @property
    def vectors(self) -> dict[str, dict[int, tuple[torch.Tensor, float]]]:
        """Registered steering vector profiles: name -> profile."""
        return dict(self._profiles)

    @property
    def probes(self) -> dict[str, dict]:
        profiles = self._monitor.profiles
        return {name: {"profile": profiles[name]}
                for name in self._monitor.probe_names}

    @property
    def history(self) -> list[dict[str, str]]:
        return list(self._history)

    @property
    def last_result(self) -> GenerationResult | None:
        return self._last_result

    # -- Extraction --

    def _local_concept_folder(self, canonical: str) -> pathlib.Path:
        """Return the local/<canonical>/ folder, creating pack.json if needed.

        User-extracted vectors and generated statements live under
        ~/.saklas/vectors/local/<canonical>/. A minimal pack.json with
        source=local is written on first access.
        """
        folder = concept_dir("local", canonical)
        folder.mkdir(parents=True, exist_ok=True)
        if not (folder / "pack.json").exists():
            PackMetadata(
                name=canonical,
                description=f"User-extracted: {canonical}",
                version="1.0.0",
                license="AGPL-3.0-or-later",
                tags=[],
                recommended_alpha=0.5,
                source="local",
                files={},
            ).write(folder)
        return folder

    def _vector_cache_path(self, canonical: str) -> str:
        folder = self._local_concept_folder(canonical)
        model_id = self._model_info.get("model_id", "unknown")
        return str(folder / f"{safe_model_id(model_id)}.safetensors")

    def _statements_cache_path(self, canonical: str) -> str:
        folder = self._local_concept_folder(canonical)
        return str(folder / "statements.json")

    def _update_local_pack_files(self, folder: pathlib.Path) -> None:
        """Recompute pack.json `files` map after writing new tensors/statements."""
        try:
            meta = PackMetadata.load(folder)
        except PackFormatError:
            return
        new_files: dict[str, str] = {}
        for entry in sorted(folder.iterdir()):
            if entry.is_file() and entry.name != "pack.json":
                new_files[entry.name] = hash_file(entry)
        meta.files = new_files
        meta.write(folder)

    def generate_pairs(
        self,
        concept: str,
        baseline: str | None = None,
        n: int = _N_PAIRS,
        on_progress: Callable[[str], None] | None = None,
    ) -> list[tuple[str, str]]:
        """Generate contrastive pairs in batches with diverse domain seeds.

        Splits the target count into small batches, each focused on a
        different domain (emotional reactions, social dynamics, etc.).
        Independent batches mean a parse failure in one doesn't cascade,
        and shorter generations stay higher quality.
        """
        if baseline is not None:
            poles = (
                f"Speaker A IS \"{concept}\" — everything about them "
                f"(what they know, how they look, what they do, how they "
                f"think) is shaped by being \"{concept}\".\n"
                f"Speaker B IS \"{baseline}\" in the same thorough way."
            )
        else:
            poles = (
                f"Speaker A IS \"{concept}\" — everything about them "
                f"(what they know, how they look, what they do, how they "
                f"think) is shaped by being \"{concept}\".\n"
                f"Speaker B has nothing to do with \"{concept}\" — they "
                f"are a completely unrelated person/entity with no "
                f"connection to any aspect of \"{concept}\"."
            )

        pad_id = self._tokenizer.pad_token_id or self._tokenizer.eos_token_id
        system_msg = (
            "You generate contrastive statement pairs for neural network "
            "interpretability research. Pairs are processed numerically "
            "for activation vector extraction. Generate exactly the number "
            "of pairs requested, no more, no less."
        )

        # Keep generating batches (cycling through domains) until we
        # have enough pairs or hit the attempt cap.  Small models
        # often under-generate, so we can't assume 1 batch = BATCH_SIZE
        # pairs.  The cap prevents infinite loops if the model
        # consistently fails to produce parseable output.
        max_batches = len(_DOMAIN_SEEDS) * 3
        all_pairs: list[tuple[str, str]] = []
        batch_idx = 0
        while len(all_pairs) < n and batch_idx < max_batches:
            domain = _DOMAIN_SEEDS[batch_idx % len(_DOMAIN_SEEDS)]
            batch_n = min(_BATCH_SIZE, n - len(all_pairs))

            if on_progress:
                on_progress(
                    f"Generating batch {batch_idx + 1} "
                    f"({len(all_pairs)}/{n} pairs, domain: {domain})..."
                )

            prompt = (
                f"Write exactly {batch_n} contrastive statement pairs.\n\n"
                f"{poles}\n\n"
                f"Focus on: {domain}.\n\n"
                f"Rules:\n"
                f"- Each statement MUST include concrete details specific "
                f"to \"{concept}\" — names, places, terminology, physical "
                f"descriptions, or references that ONLY apply to "
                f"\"{concept}\" and nothing else\n"
                f"- NO generic statements that could apply to any similar "
                f"concept — if you replaced \"{concept}\" with something "
                f"else and the statement still works, it's too vague\n"
                f"- Both statements should sound like genuinely different "
                f"people/entities — not a word swap or negation\n"
                f"- 1–2 sentences each\n\n"
                f"Format: number then a/b, period, then the statement. "
                f"Nothing else — no headers, no commentary.\n\n"
                f"1a. [Speaker A's statement]\n"
                f"1b. [Speaker B's statement]\n"
                f"2a. [Speaker A's statement]\n"
                f"2b. [Speaker B's statement]"
            )
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ]
            input_ids = build_chat_input(
                self._tokenizer, messages, system_prompt=None,
            ).to(self._device)

            with torch.inference_mode():
                out = self._model.generate(
                    input_ids,
                    max_new_tokens=batch_n * 150,
                    do_sample=True, temperature=1.0, top_p=0.9,
                    pad_token_id=pad_id,
                )
            new_ids = out[0][input_ids.shape[-1]:]
            text = self._tokenizer.decode(new_ids, skip_special_tokens=True)
            batch_pairs = self._parse_pairs(text)
            all_pairs.extend(batch_pairs)
            batch_idx += 1

        return all_pairs

    @staticmethod
    def _parse_pairs(text: str) -> list[tuple[str, str]]:
        """Parse contrastive pairs from generated text.

        Accepts varied formats: "1a.", "Na.", "a.", "1a)", "a)" etc.
        Pairs positionally: each 'a' entry pairs with the nearest 'b'
        (either direction), tolerating reversed, misnumbered, skipped,
        or duplicated indices.
        """
        entries: list[tuple[str, str]] = []  # ("a"|"b", content)
        for line in text.split("\n"):
            line = line.strip()
            m = _PAIR_RE.match(line)
            if not m:
                continue
            ab, content = m.group(1).lower(), m.group(2).strip()
            if len(content) > 10:
                entries.append((ab, content))
        # Pair adjacent a/b entries regardless of order
        pairs = []
        i = 0
        while i < len(entries) - 1:
            cur, nxt = entries[i], entries[i + 1]
            if cur[0] == "a" and nxt[0] == "b":
                pairs.append((cur[1], nxt[1]))
                i += 2
            elif cur[0] == "b" and nxt[0] == "a":
                pairs.append((nxt[1], cur[1]))
                i += 2
            else:
                # Two of the same in a row — skip the first, try the second
                i += 1
        return pairs

    def extract(
        self,
        source,
        baseline: str | None = None,
        on_progress: Callable[[str], None] | None = None,
    ) -> tuple[str, dict[int, tuple[torch.Tensor, float]]]:
        """Extract a steering vector profile.

        Full pipeline: cache check -> curated dataset -> statement cache ->
        generate pairs -> extract contrastive -> save to cache.

        Returns (canonical_name, profile). For bipolar extraction
        (baseline supplied), canonical_name is `f"{pos}_{neg}"` — the
        composite name used throughout storage and the vector registry.
        Callers should register the vector under the returned name.

        Args:
            source: concept name (str), list of (positive, negative) tuples,
                    or a DataSource instance.
            baseline: optional negative-pole concept for bipolar extraction.
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
            baseline = None
        elif isinstance(source, list):
            concept = "custom"
            baseline = None
        else:
            raise TypeError(f"Unsupported source type: {type(source)}")

        canonical = canonical_concept_name(concept, baseline)

        # For DataSource or raw pairs, skip the full pipeline — just extract
        if isinstance(source, (DataSource, list)):
            if isinstance(source, list):
                ds = DataSource(pairs=source)
            else:
                ds = source
            folder = self._local_concept_folder(canonical)
            cache_path = str(folder / f"{safe_model_id(self.model_id)}.safetensors")
            try:
                profile, _meta = _load_profile(cache_path)
                profile = self._promote_profile(profile)
                _progress(f"Loaded cached profile for '{canonical}'.")
                return canonical, profile
            except (FileNotFoundError, KeyError, ValueError):
                pass

            _progress(f"Extracting profile ({len(ds.pairs)} pairs)...")
            pairs = [{"positive": p, "negative": n} for p, n in ds.pairs]
            profile = extract_contrastive(
                self._model, self._tokenizer, pairs, layers=self._layers,
            )
            _save_profile(profile, cache_path, {"method": "contrastive_pca"})
            self._update_local_pack_files(folder)
            return canonical, profile

        # String source — full pipeline. Pack lookup scans all namespaces
        # (default/, hf-pulled, local/) via cli_selectors._all_concepts so
        # `/steer deer.wolf` hits an installed pack under any namespace,
        # not just default/. If no pack exists with this canonical name,
        # the concept extracts fresh under local/.
        from saklas.cli_selectors import _all_concepts
        curated_folder = None
        for c in _all_concepts():
            if c.name == canonical:
                curated_folder = c.folder
                break

        if curated_folder is not None:
            cache_path = str(curated_folder / f"{safe_model_id(self.model_id)}.safetensors")
        else:
            cache_path = self._vector_cache_path(canonical)

        # 1. Check vector cache
        try:
            profile, _meta = _load_profile(cache_path)
            profile = self._promote_profile(profile)
            _progress(f"Loaded cached profile for '{canonical}'.")
            return canonical, profile
        except (FileNotFoundError, KeyError, ValueError):
            pass

        # 2. Extract from curated statements if available
        if curated_folder is not None:
            curated_stmts = curated_folder / "statements.json"
            if curated_stmts.exists():
                _progress(f"Found curated statements for '{canonical}', extracting...")
                ds = load_contrastive_pairs(str(curated_stmts))
                profile = extract_contrastive(
                    self._model, self._tokenizer, ds["pairs"],
                    layers=self._layers,
                )
                _save_profile(profile, cache_path, {
                    "method": "contrastive_pca",
                    "statements_sha256": hash_file(curated_stmts),
                })
                self._update_local_pack_files(curated_folder)
                return canonical, profile

        # 3. Check statement cache under local/
        stmt_cache_path = self._statements_cache_path(canonical)
        local_folder = pathlib.Path(stmt_cache_path).parent
        pairs = None
        try:
            with open(stmt_cache_path) as f:
                cached = json.load(f)
            if isinstance(cached, list):
                pairs = cached
            elif isinstance(cached, dict) and "pairs" in cached:
                pairs = cached["pairs"]
            if pairs:
                _progress(f"Loaded cached pairs for '{canonical}'.")
        except (FileNotFoundError, KeyError, json.JSONDecodeError):
            pass

        # 4. Generate pairs if needed
        if pairs is None:
            suffix = f" vs '{baseline}'" if baseline else ""
            _progress(f"Generating contrastive pairs for '{concept}'{suffix}...")
            raw_pairs = self.generate_pairs(concept, baseline, on_progress=_progress)
            pairs = [{"positive": p, "negative": n} for p, n in raw_pairs]
            pathlib.Path(stmt_cache_path).parent.mkdir(parents=True, exist_ok=True)
            with open(stmt_cache_path, "w") as f:
                json.dump(pairs, f, indent=2)

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
            "method": "contrastive_pca",
            "statements_sha256": hash_file(pathlib.Path(stmt_cache_path)),
        })
        self._update_local_pack_files(local_folder)
        return canonical, profile

    def load_profile(self, path: str) -> dict[int, tuple[torch.Tensor, float]]:
        profile, _meta = _load_profile(path)
        return self._promote_profile(profile)

    def save_profile(self, profile: dict, path: str, metadata: dict | None = None) -> None:
        _save_profile(profile, path, metadata or {})

    # -- Steering (vector registry) --

    def steer(self, name: str, profile: dict[int, tuple[torch.Tensor, float]]) -> None:
        """Register a steering vector. Applied during generate() via alphas."""
        self._profiles[name] = profile

    def unsteer(self, name: str) -> None:
        """Remove a steering vector from the registry."""
        self._profiles.pop(name, None)

    def _apply_steering(self, alphas: dict[str, float]) -> None:
        """Compose and attach steering hooks for a generation call."""
        self._steering.clear_all()
        for name, alpha in alphas.items():
            if name not in self._profiles:
                raise KeyError(f"No vector registered for '{name}'")
            self._steering.add_vector(name, self._profiles[name], alpha)
        self._steering.apply_to_model(self._layers, self._device, self._dtype)

    def _clear_steering(self) -> None:
        """Remove all steering hooks from the model."""
        self._steering.clear_all()

    def _begin_capture(self) -> bool:
        """Attach hidden-state capture on probe layers. Returns True if attached."""
        if not self._monitor.probe_names:
            return False
        layer_idxs = sorted({
            idx for prof in self._monitor.profiles.values() for idx in prof
        })
        if not layer_idxs:
            return False
        self._capture.clear()
        self._capture.attach(self._layers, layer_idxs)
        return True

    def _end_capture(self) -> None:
        self._capture.detach()

    def score_captured(
        self, generated_ids: list[int], *, accumulate: bool = True,
    ) -> tuple[dict[str, float], dict[str, list[float]]]:
        """Score probes from the last hidden-state capture.

        Returns ``(aggregate_vals, per_token_scores)``. Both dicts are empty
        when the capture was never attached or the generation produced no
        tokens.
        """
        captured = self._capture.stacked()
        if not captured or not generated_ids:
            return {}, {}
        return self._monitor.score_per_token(
            captured, generated_ids, self._tokenizer, accumulate=accumulate,
        )

    def _promote_profile(self, profile: dict[int, tuple[torch.Tensor, float]]) -> dict[int, tuple[torch.Tensor, float]]:
        return {idx: (vec.to(self._device, self._dtype), score)
                for idx, (vec, score) in profile.items()}

    # -- Monitoring --

    def monitor(self, name: str, profile: dict | None = None) -> None:
        if profile is None:
            _, profile = self.extract(name)
        if not self._layer_means:
            self._layer_means = bootstrap_layer_means(
                self._model, self._tokenizer, self._layers, self._model_info,
            )
            self._monitor.layer_means = self._layer_means
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
        self._monitor.reset_history()

    # -- Generation helpers --

    def _prepare_input(
        self, input, raw: bool = False, thinking: bool = False,
        stateless: bool = False,
    ) -> torch.Tensor:
        if isinstance(input, str):
            prior = [] if stateless else list(self._history)
            messages = prior + [{"role": "user", "content": input}]
        elif isinstance(input, list):
            messages = list(input)
        else:
            raise TypeError(f"Unsupported input type: {type(input)}")
        if raw and isinstance(input, str):
            return self._tokenizer.encode(
                input, return_tensors="pt",
            ).to(self._device)
        return build_chat_input(
            self._tokenizer, messages, self.config.system_prompt,
            thinking=thinking,
        ).to(self._device)

    def build_readings(self) -> dict[str, ProbeReadings]:
        readings: dict[str, ProbeReadings] = {}
        if not self._monitor.probe_names:
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
                per_generation=hist, mean=mean, std=std,
                min=stats["min"] if stats["min"] != float("inf") else 0.0,
                max=stats["max"] if stats["max"] != float("-inf") else 0.0,
                delta_per_gen=delta_per_gen,
            )
        return readings

    def _finalize_generation(
        self, input, generated_ids: list[int], elapsed: float,
        vector_snapshot: dict[str, float], prompt_tokens: int = 0,
        stateless: bool = False,
        logprobs_list: list[tuple[int, float, list[tuple[int, float]]]] | None = None,
    ) -> GenerationResult:
        """Shared post-generation: decode, measure probes, build result, update history."""
        token_count = len(generated_ids)
        tok_per_sec = token_count / elapsed if elapsed > MIN_ELAPSED_FOR_RATE else 0.0
        response_ids = generated_ids[self._gen_state.thinking_end_idx:]
        text = self._tokenizer.decode(response_ids, skip_special_tokens=True)

        if self._monitor.probe_names and generated_ids:
            agg_vals, per_token = self.score_captured(
                generated_ids, accumulate=not stateless,
            )
            self._last_per_token_scores = per_token or None
            if stateless:
                readings = {
                    name: ProbeReadings(
                        per_generation=[v], mean=v, std=0.0,
                        min=v, max=v, delta_per_gen=0.0,
                    )
                    for name, v in agg_vals.items()
                }
            else:
                readings = self.build_readings()
        else:
            self._last_per_token_scores = None
            readings = self.build_readings()

        result = GenerationResult(
            text=text, tokens=list(generated_ids), token_count=token_count,
            tok_per_sec=tok_per_sec, elapsed=elapsed,
            readings=readings, vectors=vector_snapshot,
            prompt_tokens=prompt_tokens,
            finish_reason=self._gen_state.finish_reason,
            logprobs=logprobs_list,
        )
        self._last_result = result

        if not stateless:
            if isinstance(input, str):
                self._history.append({"role": "user", "content": input})
            if text.strip():
                self._history.append({"role": "assistant", "content": text})

        return result

    def _generation_preamble(self, input, alphas, raw, thinking, stateless=False):
        use_thinking = thinking and supports_thinking(self._tokenizer)
        input_ids = self._prepare_input(input, raw=raw, thinking=use_thinking, stateless=stateless)
        self._gen_state.reset()
        vector_snapshot = dict(alphas) if alphas else {}
        if alphas:
            self._apply_steering(alphas)
        return input_ids, use_thinking, vector_snapshot, int(input_ids.shape[1])

    # -- Generation: blocking --

    def generate(
        self,
        input,
        alphas: dict[str, float] | None = None,
        raw: bool = False,
        thinking: bool = False,
        stateless: bool = False,
        seed: int | None = None,
        stop: list[str] | None = None,
        logit_bias: dict[int, float] | None = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        logprobs: int | None = None,
    ) -> GenerationResult:
        """Blocking generation.

        Args:
            input: prompt string or list of message dicts.
            alphas: steering vector alphas to apply. Keys must match
                    registered vector names. None = no steering.
            raw: skip chat template, tokenize input string directly.
            thinking: enable thinking/reasoning mode for models that support it.
            stateless: do not mutate session history (conversation + probe
                    accumulators). Used by the OpenAI-compatible server.
            seed: RNG seed for reproducible sampling.
            stop: list of stop strings; generation terminates when any appears.
            logit_bias: per-token-id additive logit bias.
            presence_penalty: OpenAI-style presence penalty.
            frequency_penalty: OpenAI-style frequency penalty.
            logprobs: if set, capture per-token logprobs with this many
                    top alternatives (0 = chosen token only).
        """
        if not self._gen_lock.acquire(blocking=False):
            raise ConcurrentGenerationError("Generation already in progress")
        try:
            return self._generate_blocking(
                input, alphas, raw, thinking, stateless,
                seed, stop, logit_bias, presence_penalty, frequency_penalty, logprobs,
            )
        finally:
            self._gen_lock.release()

    def _generate_blocking(
        self, input, alphas, raw, thinking, stateless,
        seed, stop, logit_bias, presence_penalty, frequency_penalty, logprobs,
    ) -> GenerationResult:
        input_ids, use_thinking, vector_snapshot, prompt_tokens = self._generation_preamble(
            input, alphas, raw, thinking, stateless,
        )

        logprobs_list: list | None = [] if logprobs is not None else None

        def _capture(text, is_thinking, tid, lp, top_lp):
            if logprobs_list is not None and tid >= 0 and not is_thinking:
                logprobs_list.append((tid, lp if lp is not None else 0.0, top_lp or []))

        self._begin_capture()
        try:
            start = time.monotonic()
            generated_ids = generate_steered(
                self._model, self._tokenizer, input_ids,
                self.config, self._gen_state, thinking=use_thinking,
                on_token=_capture if logprobs is not None else None,
                seed=seed, stop=stop, logit_bias=logit_bias,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                logprobs=logprobs,
            )
            elapsed = time.monotonic() - start
        finally:
            self._end_capture()
            if alphas:
                self._clear_steering()

        return self._finalize_generation(
            input, generated_ids, elapsed, vector_snapshot,
            prompt_tokens=prompt_tokens, stateless=stateless,
            logprobs_list=logprobs_list,
        )

    # -- Generation: streaming --

    def generate_stream(
        self,
        input,
        alphas: dict[str, float] | None = None,
        raw: bool = False,
        thinking: bool = False,
        stateless: bool = False,
        seed: int | None = None,
        stop: list[str] | None = None,
        logit_bias: dict[int, float] | None = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        logprobs: int | None = None,
    ) -> Iterator[TokenEvent]:
        """Streaming generation. Yields TokenEvent per token. See generate()."""
        if not self._gen_lock.acquire(blocking=False):
            raise ConcurrentGenerationError("Generation already in progress")
        try:
            yield from self._generate_streaming(
                input, alphas, raw, thinking, stateless,
                seed, stop, logit_bias, presence_penalty, frequency_penalty, logprobs,
            )
        finally:
            self._gen_lock.release()

    def _generate_streaming(
        self, input, alphas, raw, thinking, stateless,
        seed, stop, logit_bias, presence_penalty, frequency_penalty, logprobs,
    ) -> Iterator[TokenEvent]:
        input_ids, use_thinking, vector_snapshot, prompt_tokens = self._generation_preamble(
            input, alphas, raw, thinking, stateless,
        )

        generated_ids: list[int] = []
        token_queue = self._gen_state.token_queue
        gen_error: list[Exception] = []
        logprobs_list: list | None = [] if logprobs is not None else None
        start = time.monotonic()

        self._begin_capture()

        def _worker():
            try:
                ids = generate_steered(
                    self._model, self._tokenizer, input_ids,
                    self.config, self._gen_state,
                    on_token=lambda tok, th, tid, lp, tlp: token_queue.put((tok, th, tid, lp, tlp)),
                    thinking=use_thinking,
                    seed=seed, stop=stop, logit_bias=logit_bias,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                    logprobs=logprobs,
                )
                generated_ids.extend(ids)
            except Exception as e:
                gen_error.append(e)
            finally:
                token_queue.put(None)

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

        idx = 0
        try:
            while True:
                try:
                    item = token_queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                if item is None:
                    break
                tok_str, is_thinking, token_id, lp, tlp = item
                if logprobs_list is not None and token_id >= 0 and not is_thinking:
                    logprobs_list.append((token_id, lp if lp is not None else 0.0, tlp or []))
                event = TokenEvent(
                    text=tok_str,
                    token_id=token_id,
                    index=idx, thinking=is_thinking,
                    logprob=lp, top_logprobs=tlp,
                )
                yield event
                idx += 1

            thread.join()
        finally:
            self._end_capture()
            if alphas:
                self._clear_steering()

        if gen_error:
            raise gen_error[0]

        elapsed = time.monotonic() - start
        self._finalize_generation(
            input, generated_ids, elapsed, vector_snapshot,
            prompt_tokens=prompt_tokens, stateless=stateless,
            logprobs_list=logprobs_list,
        )

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
