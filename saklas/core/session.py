"""SaklasSession — unified backend for saklas's programmatic API and TUI."""
from __future__ import annotations
import asyncio
import json
import logging
import pathlib
import queue
import re
import threading
import time
from typing import Callable, Iterator, Literal, overload

import torch

from saklas.io.datasource import DataSource
from saklas.core.errors import SaklasError
from saklas.core.events import (
    EventBus,
    GenerationFinished,
    GenerationStarted,
    ProbeScored,
    SteeringApplied,
    SteeringCleared,
    VectorExtracted,
)
from saklas.core.generation import GenerationConfig, GenerationState, build_chat_input, generate_steered, supports_thinking
from saklas.core.hooks import HiddenCapture, SteeringManager
from saklas.core.model import load_model, get_layers, get_model_info
from saklas.core.monitor import TraitMonitor
from saklas.io.packs import PackFormatError, PackMetadata, hash_file, hash_folder_files
from saklas.io.paths import concept_dir, tensor_filename
from saklas.io.probes_bootstrap import bootstrap_probes, bootstrap_layer_means
from saklas.core.profile import Profile
from saklas.core.results import GenerationResult, TokenEvent, ProbeReadings
from saklas.core.sampling import SamplingConfig
from saklas.core.steering import Steering
from saklas.core.steering_expr import AblationTerm
from saklas.core.triggers import Trigger
from saklas.core.vectors import (
    extract_contrastive,
    save_profile as _save_profile,
    load_profile as _load_profile,
    load_contrastive_pairs,
)

_log = logging.getLogger(__name__)

_N_PAIRS = 45
_N_SCENARIOS = 9           # default broad domains per concept
_N_PAIRS_PER_SCENARIO = 5  # default pairs sampled within each domain
_MAX_GEN_ATTEMPTS = 4      # retry whole generator call on short parse
PROBE_CATEGORIES = ["affect", "epistemic", "alignment", "register", "social_stance", "cultural"]
MIN_ELAPSED_FOR_RATE = 0.1

_PAIR_RE = re.compile(r"(?:\d+|N)\s*([ab])[.)]\s*(.*)", re.IGNORECASE)
_SCENARIO_LINE_RE = re.compile(r"^\s*(\d+)\s*[.\)]\s*(.+?)\s*$")

# System prompt shared by scenario and pair generators. Tightened from
# the v1 generic framing to emphasize format discipline — weaker models
# (gemma-4-e4b-it) parse first-try with this.
_GEN_SYSTEM_MSG = (
    "You generate structured output for neural network interpretability "
    "research. Your output is parsed programmatically. Emit exactly the "
    "number of items requested in exactly the format requested, nothing else."
)
_SLUG_RE = re.compile(r"[^a-z0-9]+")
BIPOLAR_SEP = "."


def _slug(s: str) -> str:
    """Normalize a single pole label to `[a-z0-9_]`.

    Collapses any non-alphanumeric run to `_`. Never produces the bipolar
    separator `.` — that is reserved for joining two slugged poles in
    `canonical_concept_name`.
    """
    return _SLUG_RE.sub("_", s.strip().lower()).strip("_")


def _humanize_concept(name: str) -> str:
    """Invert the slug `_` convention for LLM-facing prompts.

    Pack names and alphas keys use underscores (``artificial_intelligence``);
    the generator reads them better as spaces. Disk paths, canonical
    names, and progress messages keep the slug form.
    """
    return name.replace("_", " ")


def _split_composite_source(
    concept: str, baseline: str | None,
) -> tuple[str, str | None]:
    """Split a composite ``pos.neg`` slug when no explicit baseline is given.

    ``canonical_concept_name`` already performs this split for the
    storage name.  ``extract()`` needs the same split at the generator
    interface so :meth:`SaklasSession.generate_scenarios` and
    :meth:`SaklasSession.generate_pairs` route ``concept`` and
    ``baseline`` as two distinct poles — otherwise the LLM sees one
    composite blob vs "its semantic opposite" and the A/B assignment in
    the returned statements no longer matches the user's declared pole
    order.
    """
    if baseline is None and BIPOLAR_SEP in concept:
        pos, neg = concept.split(BIPOLAR_SEP, 1)
        return pos.strip(), neg.strip()
    return concept, baseline


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

class ConcurrentGenerationError(RuntimeError, SaklasError):
    """Raised when a generation call is made while another is in progress."""

    def user_message(self) -> tuple[int, str]:
        return (409, str(self) or self.__class__.__name__)


class VectorNotRegisteredError(KeyError, SaklasError):
    """Raised when a steering call references a vector not in the registry."""

    def user_message(self) -> tuple[int, str]:
        # KeyError str-formats the message as repr; reach into args
        # so the user sees the original text.
        msg = self.args[0] if self.args else self.__class__.__name__
        return (404, str(msg))


# Internal steering-stack entry shape: additive entries are
# ``(alpha, Trigger)`` tuples; ablation entries are ``AblationTerm``
# values carrying their own coeff + trigger + target.  The union flows
# through the stack, ``_flatten``, ``_push``/``_pop``, and is dispatched
# by type in ``_rebuild_steering_hooks`` and ``_apply_steering``.
SteeringStackEntry = tuple[float, Trigger] | AblationTerm


class _SteeringContext:
    """Context manager returned by SaklasSession.steering().

    Pushes an entries dict onto ``session._steering_stack`` on ``__enter__``
    and pops it on ``__exit__``.  Rebuilds hooks from the flattened stack
    head so nested scopes compose: inner entries overwrite outer entries
    for the duration of the inner scope, then the outer entry is restored.

    The stored ``_entries`` is the post-resolution entries form — each
    value is either ``(alpha, Trigger)`` for additive/projection terms or
    an :class:`~saklas.core.steering_expr.AblationTerm` for mean-replacement
    ablation.  Bare-alpha inputs to the public ``steering()`` API are
    normalized before we get here.
    """

    __slots__ = ("_session", "_entries", "_entered")

    def __init__(
        self,
        session: "SaklasSession",
        entries: dict[str, SteeringStackEntry],
    ) -> None:
        self._session = session
        self._entries = entries
        self._entered = False

    def __enter__(self) -> "_SteeringContext":
        # _push_steering rolls its own stack entry back if _rebuild_steering_hooks
        # raises (e.g. VectorNotRegisteredError).  __enter__ only flips
        # `_entered=True` AFTER a clean push so a mid-__enter__ failure leaves
        # no stale state for __exit__ to pop.
        self._session._push_steering(self._entries)
        self._entered = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._entered:
            self._session._pop_steering()
            self._entered = False


class SaklasSession:
    """Unified backend for activation steering, monitoring, and generation.

    Vectors are registered via steer() and applied per-generation via the
    alphas parameter on generate()/generate_stream(). No persistent hooks
    live on the model between generations.
    """

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        device: str = "auto",
        quantize: str | None = None,
        probes: list[str] | None = None,
        system_prompt: str | None = None,
        max_tokens: int = 1024,
    ) -> "SaklasSession":
        """Load a HF model + tokenizer and return a fully initialized session.

        This is the primary entry point for library users; it owns all the
        HF-loading heavy lifting. To wrap an already-loaded model use the
        plain ``__init__(model, tokenizer, ...)`` form.
        """
        model, tokenizer = load_model(model_id, quantize=quantize, device=device)
        return cls(
            model, tokenizer,
            probes=probes,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
        )

    def __init__(
        self,
        model,
        tokenizer,
        *,
        probes: list[str] | None = None,
        system_prompt: str | None = None,
        max_tokens: int = 1024,
    ):
        self._model = model
        self._tokenizer = tokenizer
        self._layers = get_layers(self._model)
        self._model_info = get_model_info(self._model, self._tokenizer)

        first_param = next(self._model.parameters())
        self._device = first_param.device
        self._dtype = first_param.dtype

        self.config = GenerationConfig(
            max_new_tokens=max_tokens,
            system_prompt=system_prompt,
        )

        # Vector registry: name -> profile. No alphas, no hooks.
        self._profiles: dict[str, dict[int, torch.Tensor]] = {}

        # Transient steering manager — used only during generation
        self._steering = SteeringManager()
        # LIFO stack of per-scope entries dicts pushed by session.steering().
        # Each entry is ``{name: (alpha, Trigger)}`` — triggers are
        # preserved through stack flattening so nested scopes with
        # different trigger regimes compose cleanly.  The flattened head
        # (later entries overwrite earlier ones) is what the steering
        # manager installs when a generation begins.
        self._steering_stack: list[dict[str, SteeringStackEntry]] = []

        # Synchronous event bus.  Emits on extraction, steering enter/exit,
        # probe scoring, generation start/finish.  Subscribers run on the
        # emit thread — async consumers must hop via call_soon_threadsafe
        # inside their callback.
        self.events: EventBus = EventBus()

        # Transient per-token hidden-state capture — attached around
        # generate_steered when probes are active so scoring happens
        # without a second forward pass.
        self._capture = HiddenCapture()

        self._gen_lock = threading.Lock()
        # Async-level serializer owned by the HTTP server for back-pressure.
        # Distinct from `_gen_lock` (threading, enforces single-flight at the
        # Python level): `lock` queues concurrent async requests FIFO so they
        # wait rather than 409.  Library-only callers never touch this.
        self.lock: asyncio.Lock = asyncio.Lock()
        self._gen_state = GenerationState()
        # Re-entry guard: True between preamble and finalize of any
        # generation path.  Prevents a pending-action dispatch from
        # double-attaching capture/steering hooks and leaking them.
        self._gen_active: bool = False

        self._history: list[dict[str, str]] = []
        self._last_result: GenerationResult | None = None
        self._last_per_token_scores: dict[str, list[float]] | None = None

        # Live trait SSE subscribers.  Each entry is (event_loop, asyncio.Queue).
        # The generation thread pushes tagged tuples via loop.call_soon_threadsafe;
        # SSE handlers drain the queue asynchronously.
        self._trait_queues: list[tuple] = []
        self._trait_lock = threading.Lock()

        # Ensure bundled concepts are materialized in the user cache and
        # the selector cache reflects them.  ``bootstrap_probes`` does this
        # transitively via ``load_defaults``, but is skipped entirely when
        # ``probes=[]`` — leaving freshly-added bundled concepts (e.g. via
        # ``regenerate_bundled_statements.py``) invisible to the selector
        # layer for the rest of the session.  Calling explicitly here keeps
        # the invariant intact regardless of probe-loading config; the call
        # is cheap when up-to-date (pack.json format-version short-circuit).
        from saklas.io.packs import materialize_bundled as _materialize_bundled
        from saklas.io import selectors as _selectors
        _materialize_bundled()
        _selectors.invalidate()

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

        self._monitor = TraitMonitor(probe_profiles, self._layer_means)

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
    def vectors(self) -> dict[str, Profile]:
        """Registered steering vector profiles: name -> Profile."""
        return {name: Profile(tensors) for name, tensors in self._profiles.items()}

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

    @property
    def last_per_token_scores(self) -> dict[str, list[float]] | None:
        return self._last_per_token_scores

    # -- Live trait SSE subscribers --

    @property
    def _trait_subscribers(self) -> int:
        return len(self._trait_queues)

    def register_trait_queue(self, loop, q) -> None:
        """Register an ``(event_loop, asyncio.Queue)`` pair for live trait events."""
        with self._trait_lock:
            self._trait_queues.append((loop, q))

    def unregister_trait_queue(self, loop, q) -> None:
        """Remove a previously registered trait queue."""
        with self._trait_lock:
            try:
                self._trait_queues.remove((loop, q))
            except ValueError:
                pass

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

    def _statements_cache_path(self, canonical: str) -> str:
        folder = self._local_concept_folder(canonical)
        return str(folder / "statements.json")

    def _update_local_pack_files(self, folder: pathlib.Path) -> None:
        """Recompute pack.json `files` map after writing new tensors/statements."""
        try:
            meta = PackMetadata.load(folder)
        except PackFormatError:
            return
        meta.files = hash_folder_files(folder)
        meta.write(folder)

    def _run_generator(
        self,
        system_msg: str,
        prompt: str,
        max_new_tokens: int,
    ) -> str:
        """Single-turn LLM call shared by scenario and pair generators.

        Builds a chat input from (system_msg, prompt), runs the model
        under inference_mode, decodes and returns the generated text.
        No parsing, no retry — callers drive the retry loop.
        """
        pad_id = self._tokenizer.pad_token_id or self._tokenizer.eos_token_id
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ]
        input_ids = build_chat_input(
            self._tokenizer, messages, system_prompt=None,
        ).to(self._device)
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            out = self._model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True, temperature=1.0, top_p=0.9,
                pad_token_id=pad_id,
            )
        new_ids = out[0][input_ids.shape[-1]:]
        return self._tokenizer.decode(new_ids, skip_special_tokens=True)

    def generate_scenarios(
        self,
        concept: str,
        baseline: str | None = None,
        n: int = _N_SCENARIOS,
        *,
        on_progress: Callable[[str], None] | None = None,
    ) -> list[str]:
        """Ask the generator for ``n`` broad situational domains shared across the axis.

        Each domain is 2–6 words, broad enough to contain many specific
        moments, shared across both poles so within-pair variance stays
        pure-pole while cross-pair variance is domain-diverse.

        The anti-allegory clause ("do not force human-social framing
        onto concepts that aren't about humans") is load-bearing for
        non-human concepts (deer/wolf, brick/feather) — without it
        the model defaults its pool to workplace/relationship
        situations and extracted vectors read everything as
        allegorical human-person statements.

        Pure function, no disk side effects. Callers that want
        persistence (``extract()``) write the result to scenarios.json
        themselves.
        """
        if n <= 0:
            return []
        # Slugs on the pack/alphas side use underscores; LLM prompts read
        # them as spaces ("artificial_intelligence" → "artificial
        # intelligence") so the generator treats the axis as the
        # underlying phrase rather than a literal token.
        concept_h = _humanize_concept(concept)
        baseline_h = _humanize_concept(baseline) if baseline is not None else None
        if baseline_h is not None:
            axis_phrase = f'"{concept_h}" vs "{baseline_h}"'
            poles_line = (
                f'Both "{concept_h}" and "{baseline_h}" should have natural, '
                f'distinct responses within every domain you list.'
            )
        else:
            axis_phrase = f'"{concept_h}" vs its semantic opposite'
            poles_line = (
                f'Both "{concept_h}" and its semantic opposite should have '
                f'natural, distinct responses within every domain you list.'
            )
        prompt = (
            f"For the axis {axis_phrase}, list exactly {n} broad "
            f"situational domains where the axis naturally expresses "
            f"itself.\n\n"
            f"A domain is a *category of experience*, not a specific "
            f"scenario. It should be 2 to 6 words, concrete enough to "
            f"be evocative, broad enough to contain many specific "
            f"situations. Cover the full range of contexts where the "
            f"axis lives — internal states, social or relational "
            f"contact, physical environment, routine moments, high-"
            f"stakes moments — whatever is natural to the axis itself. "
            f"Do not force human-social framing onto concepts that "
            f"aren't about humans.\n\n"
            f"{poles_line}\n\n"
            f"The {n} domains together should span the axis without "
            f"overlap. No meta-commentary, no explanations, no sub-"
            f"bullets.\n\n"
            f"Format: number, period, then the domain name. Nothing "
            f"else.\n\n"
            f"1. [domain]\n"
            f"2. [domain]\n"
            f"...\n"
            f"{n}. [domain]"
        )
        max_new_tokens = max(300, n * 40)
        best: list[str] = []
        for attempt in range(1, _MAX_GEN_ATTEMPTS + 1):
            if on_progress:
                on_progress(
                    f"Generating {n} scenarios for '{concept}' "
                    f"(attempt {attempt}/{_MAX_GEN_ATTEMPTS})..."
                )
            text = self._run_generator(_GEN_SYSTEM_MSG, prompt, max_new_tokens)
            parsed = self._parse_scenarios(text)
            if len(parsed) >= n:
                return parsed[:n]
            if len(parsed) > len(best):
                best = parsed
        return best

    @staticmethod
    def _parse_scenarios(text: str) -> list[str]:
        """Parse a numbered list of domain names from generated text.

        Accepts ``1. name``, ``1) name``, tolerates bracket markers and
        trailing punctuation, deduplicates case-insensitively.
        """
        out: list[str] = []
        seen: set[str] = set()
        for line in text.split("\n"):
            m = _SCENARIO_LINE_RE.match(line)
            if not m:
                continue
            name = m.group(2).strip().strip("[]").strip().rstrip(".,;:")
            if not name:
                continue
            key = name.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(name)
        return out

    def generate_pairs(
        self,
        concept: str,
        baseline: str | None = None,
        n: int = _N_PAIRS,
        *,
        scenarios: list[str] | None = None,
        on_progress: Callable[[str], None] | None = None,
    ) -> list[tuple[str, str]]:
        """Generate contrastive statement pairs via the open-ended pipeline.

        For each of ``len(scenarios)`` broad domains (passed via
        ``scenarios`` or generated fresh via :meth:`generate_scenarios`),
        ask the model for ``ceil(n / len(scenarios))`` first-person
        contrastive pairs drawn from concrete moments within that
        domain. Uses POV/behavior framing that generalizes across human
        and non-human concepts — a ``deer.wolf`` axis yields literal
        animal-life pairs rather than human-allegory pairs. Returns up
        to ``n`` pairs total.

        Bipolar (baseline given): Speaker A embodies ``concept``,
        Speaker B embodies ``baseline``. Monopolar (baseline None):
        Speaker B embodies the semantic opposite of ``concept``.
        """
        if scenarios is None:
            scenarios = self.generate_scenarios(
                concept, baseline, _N_SCENARIOS, on_progress=on_progress,
            )
        if not scenarios:
            return []

        pairs_per_scenario = max(1, -(-n // len(scenarios)))  # ceil div

        # See ``generate_scenarios`` — slug underscores become spaces for
        # the LLM-facing prompt only; progress messages and cache keys
        # keep the slug form.
        concept_h = _humanize_concept(concept)
        baseline_h = _humanize_concept(baseline) if baseline is not None else None
        if baseline_h is not None:
            axis_phrase = f'"{concept_h}" vs "{baseline_h}"'
            a_line = (
                f'   - Statement A: write like you ARE "{concept_h}", '
                f'facing that moment.'
            )
            b_line = (
                f'   - Statement B: write like you ARE "{baseline_h}", '
                f'facing the same moment.'
            )
            labels_ban = (
                f'Do not name the poles. Never write "I am a {concept_h}" '
                f'or "as a {baseline_h}" or any similar self-label — just '
                f'inhabit the pole directly.'
            )
        else:
            axis_phrase = f'"{concept_h}" vs its semantic opposite'
            a_line = (
                f'   - Statement A: write like you ARE "{concept_h}", '
                f'facing that moment.'
            )
            b_line = (
                f'   - Statement B: write like you ARE the semantic '
                f'opposite of "{concept_h}" — whatever that opposite '
                f'naturally is — facing the same moment.'
            )
            labels_ban = (
                f'Do not name the pole. Never write "I am a {concept_h}" '
                f'or any similar self-label — just inhabit the pole '
                f'directly.'
            )

        all_pairs: list[tuple[str, str]] = []
        for idx, scenario in enumerate(scenarios, 1):
            if len(all_pairs) >= n:
                break
            if on_progress:
                on_progress(
                    f"Generating {pairs_per_scenario} pairs for domain "
                    f"{idx}/{len(scenarios)}: {scenario}"
                )
            prompt = (
                f"Axis: {axis_phrase}.\n"
                f"Domain: {scenario}.\n\n"
                f"Write exactly {pairs_per_scenario} contrastive "
                f"statement pairs drawn from this domain.\n\n"
                f"For each pair:\n"
                f"1. Pick a specific concrete moment that naturally "
                f"lives inside the domain — a thing happening right "
                f"now, not a generality.\n"
                f"2. Write two first-person statements about that "
                f"same moment:\n"
                f"{a_line}\n"
                f"{b_line}\n\n"
                f"Write AS the pole, not ABOUT it. {labels_ban}\n\n"
                f"Both statements in a pair should have the same "
                f"overall shape — both an inner thought, or both a "
                f"description of what you do, or both something said "
                f"aloud — so the only axis of variation is "
                f"{axis_phrase}.\n\n"
                f"Each statement should be at least 12 words; longer "
                f"is fine. Natural, unhurried language. Lean into the "
                f"pole and let it speak in its natural register.\n\n"
                f"Format: number then a/b, period, then the statement. "
                f"Nothing else.\n\n"
                f"1a. [Statement A for moment 1]\n"
                f"1b. [Statement B for moment 1]\n"
                f"2a. [Statement A for moment 2]\n"
                f"2b. [Statement B for moment 2]\n"
                f"..."
            )
            max_new_tokens = max(400, pairs_per_scenario * 200)
            batch_best: list[tuple[str, str]] = []
            for _attempt in range(_MAX_GEN_ATTEMPTS):
                text = self._run_generator(
                    _GEN_SYSTEM_MSG, prompt, max_new_tokens,
                )
                parsed = self._parse_pairs(text)
                if len(parsed) >= pairs_per_scenario:
                    batch_best = parsed[:pairs_per_scenario]
                    break
                if len(parsed) > len(batch_best):
                    batch_best = parsed
            all_pairs.extend(batch_best)

        return all_pairs[:n]

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
        *,
        scenarios: list[str] | None = None,
        reuse_scenarios: bool = False,
        force_statements: bool = False,
        on_progress: Callable[[str], None] | None = None,
        sae: str | None = None,
        sae_revision: str | None = None,
        namespace: str | None = None,
    ) -> tuple[str, Profile]:
        """Extract a steering vector profile and emit VectorExtracted.

        Thin wrapper around :meth:`_extract_impl` that fires the event
        bus after the profile is available.  Cache hits and fresh
        extractions are both emitted — subscribers that only care about
        freshly-computed profiles can gate on ``event.metadata["method"]``.

        **Default behavior**: tensor cache hits short-circuit. On
        tensor miss, if ``statements.json`` exists (curated bundled
        pack or local cache), extract directly from it — statements
        are the expensive part and reuse is the sane default. On
        statements miss, run the full pipeline: generate scenarios →
        generate pairs → save both → extract tensor.

        Flags:

        - ``scenarios=[...]``: explicit scenarios input; bypasses
          scenario generation and ``scenarios.json`` cache. Written
          to disk after use. **Also bypasses the tensor cache** —
          supplying fresh scenarios means the caller wants fresh
          pairs, so any cached tensor is stale by definition.
        - ``reuse_scenarios=True``: when regenerating pairs, load
          ``scenarios.json`` from disk if present instead of
          regenerating. Default False — scenarios are cheap, so the
          full pipeline regenerates them fresh each pair-gen pass.
        - ``force_statements=True``: regenerate ``statements.json``
          from scratch. **Also bypasses the tensor cache** — same
          reasoning as ``scenarios=[...]``: if you're regenerating
          pairs, you want a tensor extracted from them, not from the
          stale ones.
        """
        canonical, profile = self._extract_impl(
            source, baseline,
            scenarios=scenarios,
            reuse_scenarios=reuse_scenarios,
            force_statements=force_statements,
            on_progress=on_progress,
            sae=sae,
            sae_revision=sae_revision,
            namespace=namespace,
        )
        try:
            meta = dict(profile.metadata) if hasattr(profile, "metadata") else {}
        except Exception:
            meta = {}
        self.events.emit(VectorExtracted(name=canonical, profile=profile, metadata=meta))
        return canonical, profile

    def _extract_impl(
        self,
        source,
        baseline: str | None = None,
        *,
        scenarios: list[str] | None = None,
        reuse_scenarios: bool = False,
        force_statements: bool = False,
        on_progress: Callable[[str], None] | None = None,
        sae: str | None = None,
        sae_revision: str | None = None,
        namespace: str | None = None,
    ) -> tuple[str, Profile]:
        """Actual extraction pipeline — see :meth:`extract` for the wrapper.

        Pipeline: tensor cache check → statements.json cache (curated
        or local) unless ``force_statements`` → otherwise generate
        scenarios + pairs → save both → extract contrastive → save
        tensor. Scenarios are persisted to
        ``local/<canonical>/scenarios.json`` alongside ``statements.json``.

        Returns (canonical_name, profile). For bipolar extraction
        (baseline supplied), canonical_name is ``f"{pos}.{neg}"`` — the
        composite name used throughout storage and the vector registry.

        Args:
            source: concept name (str), list of (positive, negative)
                    tuples, or a DataSource instance.
            baseline: optional negative-pole concept for bipolar
                      extraction. Only used when source is a string.
            scenarios: explicit scenarios list, bypassing generation
                       and the scenarios.json cache. Written to disk
                       after use. Implies fresh statement generation.
            reuse_scenarios: when regenerating pairs, reuse
                             scenarios.json from disk if present.
                             Default False — regenerate scenarios fresh.
            force_statements: ignore statements.json cache and
                              regenerate the full pipeline.
            on_progress: optional callback for progress messages.
        """
        def _progress(msg: str) -> None:
            if on_progress:
                on_progress(msg)

        # Normalize source
        if isinstance(source, str):
            concept, baseline = _split_composite_source(source, baseline)
        elif isinstance(source, DataSource):
            concept = source.name
            baseline = None
        elif isinstance(source, list):
            concept = "custom"
            baseline = None
        else:
            raise TypeError(f"Unsupported source type: {type(source)}")

        canonical = canonical_concept_name(concept, baseline)

        # Resolve the SAE backend once. No-op when ``sae is None`` —
        # the ``load_sae_backend`` import is lazy so non-SAE callers
        # don't touch the SAE layer.
        sae_backend = None
        sae_metadata: dict = {}
        if sae is not None:
            from saklas.core.sae import load_sae_backend
            sae_backend = load_sae_backend(
                sae,
                revision=sae_revision,
                model_id=self.model_id,
                device=self._device,
            )
            sae_metadata = {
                "sae_release": sae_backend.release,
                "sae_revision": sae_backend.revision,
                "sae_ids_by_layer": getattr(sae_backend, "sae_ids_by_layer", {}),
            }

        def _build_return(profile_dict: dict) -> tuple[str, Profile]:
            meta: dict = {
                "method": "pca_center_sae" if sae_backend is not None else "contrastive_pca",
            }
            meta.update(sae_metadata)
            out_name = (
                canonical
                if sae_backend is None
                else f"{canonical}:sae-{sae_backend.release}"
            )
            return out_name, Profile(profile_dict, metadata=meta)

        def _save_meta(extra: dict | None = None) -> dict:
            meta: dict = {
                "method": "pca_center_sae" if sae_backend is not None else "contrastive_pca",
            }
            if extra:
                meta.update(extra)
            meta.update(sae_metadata)
            return meta

        # For DataSource or raw pairs, skip the full pipeline — just extract
        if isinstance(source, (DataSource, list)):
            if isinstance(source, list):
                ds = DataSource(pairs=source)
            else:
                ds = source
            folder = self._local_concept_folder(canonical)
            cache_path = str(folder / tensor_filename(self.model_id, release=sae))
            try:
                profile, _meta = _load_profile(cache_path)
                profile = self._promote_profile(profile)
                _progress(f"Loaded cached profile for '{canonical}'.")
                out_name = (
                    canonical
                    if sae_backend is None
                    else f"{canonical}:sae-{sae_backend.release}"
                )
                return out_name, Profile(profile, metadata=_meta)
            except (FileNotFoundError, KeyError, ValueError):
                pass

            _progress(f"Extracting profile ({len(ds.pairs)} pairs)...")
            pairs = [{"positive": p, "negative": n} for p, n in ds.pairs]
            profile = extract_contrastive(
                self._model, self._tokenizer, pairs, layers=self._layers,
                sae=sae_backend,
            )
            _save_profile(profile, cache_path, _save_meta())
            self._update_local_pack_files(folder)
            return _build_return(profile)

        # String source — full pipeline. Pack lookup scans installed
        # namespaces, but bare names must not silently pick the first duplicate.
        # If the caller supplies namespace=, honor only that namespace.
        from saklas.io.selectors import _all_concepts, AmbiguousSelectorError
        matches = [
            c for c in _all_concepts()
            if c.name == canonical and (namespace is None or c.namespace == namespace)
        ]
        if namespace is not None and not matches and namespace != "local":
            raise FileNotFoundError(
                f"concept pack '{namespace}/{canonical}' is not installed"
            )
        if namespace is None and len(matches) > 1:
            qualified = ", ".join(f"{c.namespace}/{c.name}" for c in matches)
            raise AmbiguousSelectorError(
                f"ambiguous concept '{canonical}': matches {qualified}. "
                f"Specify with a namespace."
            )
        pack_folder = matches[0].folder if matches else None

        if pack_folder is not None:
            cache_path = str(pack_folder / tensor_filename(self.model_id, release=sae))
        else:
            cache_path = str(
                pathlib.Path(self._local_concept_folder(canonical))
                / tensor_filename(self.model_id, release=sae)
            )

        # 1. Vector cache — short-circuits unless a regen path is requested.
        #    ``force_statements=True`` or explicit ``scenarios=[...]`` both
        #    mean the caller wants fresh pairs, which definitionally
        #    invalidates any tensor trained on the old pairs — bypassing
        #    the tensor cache here is the only semantically coherent
        #    behavior for those flags. No cache hit means the full
        #    pipeline runs end-to-end and overwrites the stale tensor.
        if not force_statements and scenarios is None:
            try:
                profile, _meta = _load_profile(cache_path)
                profile = self._promote_profile(profile)
                _progress(f"Loaded cached profile for '{canonical}'.")
                out_name = (
                    canonical
                    if sae_backend is None
                    else f"{canonical}:sae-{sae_backend.release}"
                )
                return out_name, Profile(profile, metadata=_meta)
            except (FileNotFoundError, KeyError, ValueError):
                pass

        # 2. Installed-statements fast path — default reuses bundled/HF/local
        #    statements.json when present. ``force_statements=True`` skips
        #    this branch and falls through to regeneration. Passing an
        #    explicit ``scenarios=`` also skips — if the caller supplied
        #    scenarios they're clearly opting into fresh pair generation.
        if pack_folder is not None and not force_statements and scenarios is None:
            pack_stmts = pack_folder / "statements.json"
            if pack_stmts.exists():
                _progress(f"Using curated statements for '{canonical}'...")
                ds = load_contrastive_pairs(str(pack_stmts))
                profile = extract_contrastive(
                    self._model, self._tokenizer, ds["pairs"],
                    layers=self._layers,
                    sae=sae_backend,
                )
                _save_profile(profile, cache_path, _save_meta({
                    "statements_sha256": hash_file(pack_stmts),
                }))
                self._update_local_pack_files(pack_folder)
                return _build_return(profile)

        # 3. Local statements cache — default reuses if present.
        stmt_cache_path = self._statements_cache_path(canonical)
        local_folder = pathlib.Path(stmt_cache_path).parent
        pairs = None
        if not force_statements and scenarios is None:
            try:
                with open(stmt_cache_path) as f:
                    cached = json.load(f)
                if isinstance(cached, list):
                    pairs = cached
                elif isinstance(cached, dict) and "pairs" in cached:
                    pairs = cached["pairs"]
                if pairs:
                    _progress(f"Using cached pairs for '{canonical}'.")
            except (FileNotFoundError, KeyError, json.JSONDecodeError):
                pass

        # 4. Generate scenarios + pairs if needed.
        if pairs is None:
            suffix = f" vs '{baseline}'" if baseline else ""
            local_folder.mkdir(parents=True, exist_ok=True)

            # 4a. Resolve effective scenarios.
            scn_path = local_folder / "scenarios.json"
            eff_scenarios: list[str] | None = None
            if scenarios is not None:
                eff_scenarios = list(scenarios)
                _progress(
                    f"Using {len(eff_scenarios)} caller-provided scenarios."
                )
            elif reuse_scenarios and scn_path.exists():
                try:
                    with open(scn_path) as f:
                        data = json.load(f)
                    if isinstance(data, dict) and isinstance(data.get("scenarios"), list):
                        eff_scenarios = [str(s) for s in data["scenarios"]]
                    elif isinstance(data, list):
                        eff_scenarios = [str(s) for s in data]
                    if eff_scenarios:
                        _progress(
                            f"Reusing {len(eff_scenarios)} cached scenarios "
                            f"for '{canonical}'."
                        )
                except (FileNotFoundError, KeyError, json.JSONDecodeError):
                    eff_scenarios = None

            if not eff_scenarios:
                _progress(f"Generating scenarios for '{concept}'{suffix}...")
                eff_scenarios = self.generate_scenarios(
                    concept, baseline, on_progress=_progress,
                )

            if not eff_scenarios:
                raise ValueError(
                    f"Could not generate scenarios for '{concept}'. "
                    f"Try a more specific concept."
                )

            # Persist scenarios to disk (overwriting any existing file).
            with open(scn_path, "w") as f:
                json.dump({"scenarios": eff_scenarios}, f, indent=2)

            _progress(
                f"Generating contrastive pairs for '{concept}'{suffix} "
                f"across {len(eff_scenarios)} domains..."
            )
            raw_pairs = self.generate_pairs(
                concept, baseline,
                scenarios=eff_scenarios,
                on_progress=_progress,
            )
            pairs = [{"positive": p, "negative": n} for p, n in raw_pairs]
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
            sae=sae_backend,
        )
        _save_profile(profile, cache_path, _save_meta({
            "statements_sha256": hash_file(pathlib.Path(stmt_cache_path)),
        }))
        self._update_local_pack_files(local_folder)
        return _build_return(profile)

    def clone_from_corpus(
        self,
        path,
        name: str,
        *,
        n_pairs: int = 90,
        seed: int | None = None,
        batch_size: int = 5,
        force: bool = False,
    ) -> tuple[str, Profile]:
        """Extract a persona-cloning steering vector from a corpus file.

        Thin wrapper around saklas.cloning.clone_from_corpus; see that
        module for the full pipeline. Returns `(canonical_name, profile)`
        matching extract()'s return shape.
        """
        from saklas.io.cloning import clone_from_corpus as _clone
        return _clone(
            self, path, name,
            n_pairs=n_pairs, seed=seed, batch_size=batch_size, force=force,
        )

    def load_profile(self, path: str) -> Profile:
        profile, meta = _load_profile(path)
        promoted = self._promote_profile(profile)
        return Profile(promoted, metadata=meta)

    def save_profile(
        self,
        profile: Profile,
        path: str,
        metadata: dict | None = None,
    ) -> None:
        profile.save(path, metadata=metadata)

    # -- Steering (vector registry) --

    def steer(self, name: str, profile: Profile) -> None:
        """Register a steering vector. Applied during generate() via alphas.

        Internally stored as a plain dict so the steering hook's hot path
        can read tensors without attribute lookups.
        """
        self._profiles[name] = dict(profile.as_dict())

    def unsteer(self, name: str) -> None:
        """Remove a steering vector from the registry."""
        self._profiles.pop(name, None)

    def steering(
        self, value: "str | Steering",
    ) -> "_SteeringContext":
        """Context manager applying steering for the duration of a with-block.

        ``value`` is either a steering expression string (parsed through
        the shared grammar in :mod:`saklas.core.steering_expr`) or a
        pre-built :class:`Steering`.  Dict inputs are not accepted; build
        :class:`Steering` directly if you need typed construction.

        Pole aliases (``io.selectors.resolve_pole``) resolve at parse
        time; this is the canonical resolver site — CLI, server, and
        TUI all route through here.  Nesting flattens: an inner
        ``steering("0.5 angry.calm")`` overrides the outer
        ``steering("0.3 angry.calm")`` for the duration of the inner
        scope, and the outer entry is restored on ``__exit__``.  One hook
        installation per active layer regardless of nesting depth.

        Unknown vector names raise ``VectorNotRegisteredError``; genuinely
        ambiguous pole names propagate ``AmbiguousSelectorError``.
        """
        steering_obj = Steering.from_value(value)
        if steering_obj is None:
            raise TypeError(
                "session.steering() requires a non-None expression string "
                "or Steering instance"
            )
        # Materialize any ProjectedTerm entries into derived profiles
        # registered in ``self._profiles`` under the synthetic key.
        # Must run before ``normalized_entries`` because the normalized
        # form flattens ``ProjectedTerm`` into ``(coeff, trigger)`` and
        # loses the ``base`` / ``onto`` / ``operator`` fields.
        self._materialize_projections(steering_obj)
        raw_entries = steering_obj.normalized_entries()
        resolved: dict[str, SteeringStackEntry] = dict(
            self._resolve_pole_aliases(raw_entries)
        )

        # Fold in ablation entries alongside additive/projection ones.
        # ``normalized_entries`` strips ``AblationTerm`` values, so walk
        # ``steering_obj.alphas`` directly.  Keys already carry the
        # ``!<target>`` form from the parser and live in a disjoint
        # namespace from plain/projection keys, so no collision is
        # possible.  Attempt autoload for the target so a client can
        # reference an installed pack without an explicit ``extract()``;
        # genuine misses surface through ``_rebuild_steering_hooks``
        # uniformly with the additive path.
        for key, val in steering_obj.alphas.items():
            if not isinstance(val, AblationTerm):
                continue
            target = val.target
            if target not in self._profiles:
                if ":" in target:
                    canonical, variant = target.rsplit(":", 1)
                else:
                    canonical, variant = target, "raw"
                try:
                    self._try_autoload_vector(canonical, variant=variant)
                except Exception:
                    # Non-raw variant miss raises AmbiguousVariantError or
                    # UnknownVariantError; let it surface at hook-install
                    # with the shared VectorNotRegisteredError shape.
                    pass
            resolved[key] = val
        return _SteeringContext(self, resolved)

    def _materialize_projections(self, steering: Steering) -> None:
        """Populate ``self._profiles`` with derived profiles for every
        :class:`~saklas.core.steering_expr.ProjectedTerm` in
        ``steering.alphas``.

        Ensures the ``base`` and ``onto`` profiles are loaded (invoking
        the autoload path when needed), runs
        :func:`saklas.core.vectors.project_profile` to build the derived
        tensor dict, and registers it under the synthetic key
        ``"<base><op><onto>"``.  The synthetic key matches what the parser
        used for the ``Steering.alphas`` key, so downstream pole
        resolution + hook install find the profile via the
        ``name in self._profiles`` fast path.
        """
        from saklas.core.steering_expr import ProjectedTerm
        from saklas.core.vectors import project_profile

        for syn_key, val in steering.alphas.items():
            if not isinstance(val, ProjectedTerm):
                continue
            self._ensure_profile_loaded(val.base)
            self._ensure_profile_loaded(val.onto)
            base_tensors = self._profiles[val.base]
            onto_tensors = self._profiles[val.onto]
            projected = project_profile(base_tensors, onto_tensors, val.operator)
            self._profiles[syn_key] = projected

    def _ensure_profile_loaded(self, key: str) -> None:
        """Ensure ``key`` is registered in ``self._profiles``.

        ``key`` is a canonical registry key (bare for raw variants,
        ``f"{canonical}:{variant}"`` otherwise) as produced by
        :func:`saklas.core.steering_expr._resolve_atom`.  Routes to the
        existing autoload path for packs that are installed but not yet
        loaded.
        """
        if key in self._profiles:
            return
        if ":" in key:
            canonical, variant = key.rsplit(":", 1)
        else:
            canonical, variant = key, "raw"
        self._try_autoload_vector(canonical, variant=variant)
        if key not in self._profiles:
            raise VectorNotRegisteredError(
                f"projection references '{key}' which is not registered "
                f"and no pack could be autoloaded for this model"
            )

    def _resolve_pole_aliases(
        self, entries: dict[str, tuple[float, Trigger]],
    ) -> dict[str, tuple[float, Trigger]]:
        """Apply pole-alias resolution + sign flipping + variant routing.

        Returned keys carry the full variant-qualified name:
        ``canonical`` for raw, ``f"{canonical}:{variant}"`` otherwise. Autoload
        is variant-aware — ``honest:sae`` will look for a ``_sae-*`` tensor
        file, not the raw one.

        Names already in ``self._profiles`` pass through verbatim — a caller
        who pre-registered under a specific key stays addressed by that key.
        """
        from saklas.io.selectors import resolve_pole

        out: dict[str, tuple[float, Trigger]] = {}
        for name, (alpha, trig) in entries.items():
            if name in self._profiles:
                out[name] = (float(alpha), trig)
                continue
            try:
                canonical, sign, _match, variant = resolve_pole(name)
            except Exception:
                out[name] = (float(alpha), trig)
                continue
            registry_key = canonical if variant == "raw" else f"{canonical}:{variant}"
            if registry_key not in self._profiles:
                try:
                    self._try_autoload_vector(canonical, variant=variant)
                except Exception:
                    # Autoload may raise AmbiguousVariantError / UnknownVariantError.
                    # Keep the user's original name in `out` so the error surfaces
                    # at hook-install time with a clear message.
                    out[name] = (float(alpha), trig)
                    continue
            effective = float(alpha) * (1 if sign >= 0 else -1)
            if registry_key in self._profiles:
                prev_alpha = out.get(registry_key, (0.0, trig))[0]
                out[registry_key] = (prev_alpha + effective, trig)
            else:
                out[name] = (float(alpha), trig)
        return out

    def _try_autoload_vector(self, canonical: str, *, variant: str = "raw") -> None:
        """Cache-hit fast path: load an installed concept's tensor into _profiles.

        Walks installed concept packs, finds the first matching ``canonical``,
        and loads its per-model tensor. ``variant`` is the resolver's output:

        - ``"raw"`` — loads the unsuffixed tensor. Silent on miss (caller
          falls through to the normal raise path). Matches pre-Task-7 behavior.
        - ``"sae"`` — loads the unique SAE variant. Raises
          :class:`AmbiguousVariantError` when more than one is on disk,
          :class:`UnknownVariantError` when zero exist.
        - ``"sae-<release>"`` — loads that specific release.
          :class:`UnknownVariantError` when absent.

        Registered key in ``_profiles`` is ``canonical`` for raw, and
        ``f"{canonical}:{variant}"`` otherwise.
        """
        from saklas.io.selectors import _all_concepts
        from saklas.io.packs import enumerate_variants
        from saklas.core.errors import AmbiguousVariantError, UnknownVariantError
        from saklas.core.vectors import load_profile

        registry_key = canonical if variant == "raw" else f"{canonical}:{variant}"
        available: list[str] = []
        for concept in _all_concepts():
            if concept.name != canonical:
                continue
            variants = enumerate_variants(concept.folder, self.model_id)
            available.extend(variants.keys())

            if variant == "raw":
                path = variants.get("raw")
            elif variant == "sae":
                sae_paths = {k: v for k, v in variants.items() if k.startswith("sae-")}
                if len(sae_paths) == 0:
                    continue
                if len(sae_paths) > 1:
                    raise AmbiguousVariantError(
                        f"concept '{canonical}' has multiple SAE variants for "
                        f"model '{self.model_id}': {sorted(sae_paths.keys())}. "
                        f"Specify explicitly with :sae-<release>."
                    )
                path = next(iter(sae_paths.values()))
            else:
                # "sae-<release>"
                path = variants.get(variant)

            if path is None:
                continue

            try:
                profile_dict, _meta = load_profile(str(path))
            except Exception:
                continue
            self._profiles[registry_key] = self._promote_profile(profile_dict)
            return

        # Explicit non-raw variant request that didn't resolve → surface the miss.
        if variant != "raw":
            raise UnknownVariantError(
                f"variant '{variant}' not found for '{canonical}' on model "
                f"'{self.model_id}' (available: {sorted(set(available)) or 'none'})"
            )

    def _push_steering(
        self, entries: dict[str, SteeringStackEntry],
    ) -> None:
        """Push an entries dict onto the steering stack and rebuild hooks.

        If ``_rebuild_steering_hooks`` raises (e.g. an unknown vector name
        hits ``VectorNotRegisteredError``) the just-pushed entry is rolled
        back before the exception propagates, so the stack is never left
        with stale half-committed state.
        """
        self._steering_stack.append(dict(entries))
        try:
            self._rebuild_steering_hooks()
        except BaseException:
            self._steering_stack.pop()
            raise
        self._emit_steering_applied()

    def _pop_steering(self) -> None:
        """Pop the top of the steering stack and rebuild hooks."""
        if not self._steering_stack:
            return
        self._steering_stack.pop()
        self._rebuild_steering_hooks()
        if not self._steering_stack:
            self.events.emit(SteeringCleared())
        else:
            self._emit_steering_applied()

    def _emit_steering_applied(self) -> None:
        """Emit SteeringApplied with alphas-only + full entries.

        ``alphas`` carries the flat ``{name: alpha}`` shape; ``entries``
        carries the full ``{name: (alpha, trigger)}`` mapping.  Ablation
        entries keyed under ``!<target>`` flatten to their ``(coeff,
        trigger)`` pair so subscribers see one uniform tuple shape.
        """
        flat = self._flatten_steering_stack()
        alphas_only: dict[str, float] = {}
        entries_out: dict[str, tuple[float, Trigger]] = {}
        for name, entry in flat.items():
            if isinstance(entry, AblationTerm):
                alphas_only[name] = entry.coeff
                entries_out[name] = (entry.coeff, entry.trigger)
                continue
            alphas_only[name] = entry[0]
            entries_out[name] = entry
        self.events.emit(SteeringApplied(alphas=alphas_only, entries=entries_out))

    def _flatten_steering_stack(self) -> dict[str, SteeringStackEntry]:
        """Collapse the LIFO stack into a single entries dict (later wins)."""
        flat: dict[str, SteeringStackEntry] = {}
        for entry in self._steering_stack:
            flat.update(entry)
        return flat

    def _rebuild_steering_hooks(self) -> None:
        """Tear down existing hooks and install from the flattened stack head.

        Called on every push/pop.  When the stack is empty this is a clean
        ``clear_all``.  One hook installation per active layer regardless
        of nesting depth — ``SteeringManager.apply_to_model`` composes
        per-layer vectors internally and groups entries by trigger within
        each layer.  Dispatches by entry type: plain tuples route to
        :meth:`SteeringManager.add_vector`, :class:`AblationTerm` values
        route to :meth:`SteeringManager.add_ablation` using the term's
        ``target`` as the registry key.
        """
        flat = self._flatten_steering_stack()
        self._steering.clear_all()
        if not flat:
            return
        for name, entry in flat.items():
            if isinstance(entry, AblationTerm):
                target = entry.target
                if target not in self._profiles:
                    raise VectorNotRegisteredError(
                        f"No vector registered for ablation target '{target}'"
                    )
                self._steering.add_ablation(
                    target, self._profiles[target],
                    alpha=entry.coeff, trigger=entry.trigger,
                    layer_means=self._layer_means,
                )
                continue
            alpha, trigger = entry
            if name not in self._profiles:
                raise VectorNotRegisteredError(f"No vector registered for '{name}'")
            self._steering.add_vector(
                name, self._profiles[name], alpha, trigger,
            )
        self._steering.apply_to_model(self._layers, self._device, self._dtype)

    def _apply_steering(
        self, entries: dict[str, SteeringStackEntry],
    ) -> None:
        """Compose and attach steering hooks for a generation call.

        Must be called inside a ``_gen_active`` span (entry points set
        ``_gen_active=True`` before invoking this).  The check is defense
        in depth against a rogue caller re-entering outside a gen span.
        Dispatches by entry type — mirrors :meth:`_rebuild_steering_hooks`.
        """
        self._steering.clear_all()
        for name, entry in entries.items():
            if isinstance(entry, AblationTerm):
                target = entry.target
                if target not in self._profiles:
                    raise VectorNotRegisteredError(
                        f"No vector registered for ablation target '{target}'"
                    )
                self._steering.add_ablation(
                    target, self._profiles[target],
                    alpha=entry.coeff, trigger=entry.trigger,
                    layer_means=self._layer_means,
                )
                continue
            alpha, trigger = entry
            if name not in self._profiles:
                raise VectorNotRegisteredError(f"No vector registered for '{name}'")
            self._steering.add_vector(
                name, self._profiles[name], alpha, trigger,
            )
        self._steering.apply_to_model(self._layers, self._device, self._dtype)

    def _clear_steering(self) -> None:
        """Remove all steering hooks from the model."""
        self._steering.clear_all()

    def _begin_capture(self, *, widen: bool = False) -> bool:
        """Attach hidden-state capture. Returns True if attached.

        ``widen=False`` (default): cover only probe-layer union — what
        the monitor needs.  Fast path; matches v1 behavior.

        ``widen=True``: cover every model layer.  Used when the caller
        asked for ``SamplingConfig.return_hidden=True`` — the monitor
        still reads its subset, but the full dict is available on
        ``GenerationResult.hidden_states`` after the run.
        """
        if widen:
            layer_idxs = list(range(len(self._layers)))
        else:
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

    # -- Score entry points --

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

    @overload
    def score_hidden(
        self,
        hidden: dict[int, torch.Tensor],
        *,
        per_token: Literal[False] = False,
        accumulate: bool = False,
    ) -> dict[str, float]: ...
    @overload
    def score_hidden(
        self,
        hidden: dict[int, torch.Tensor],
        *,
        per_token: Literal[True],
        accumulate: bool = False,
    ) -> tuple[dict[str, float], dict[str, list[float]]]: ...
    def score_hidden(
        self,
        hidden: dict[int, torch.Tensor],
        *,
        per_token: bool = False,
        accumulate: bool = False,
    ) -> (
        dict[str, float]
        | tuple[dict[str, float], dict[str, list[float]]]
    ):
        """Score registered probes against a pre-captured hidden-state dict.

        Accepts any ``{layer_idx: Tensor}`` mapping — e.g. the
        ``GenerationResult.hidden_states`` dict from a prior
        ``generate(..., sampling=SamplingConfig(return_hidden=True))``
        call, or hidden states the caller captured externally.

        Shape rules:
        - Each value ``[D]``          → single-state aggregate.
          Returns ``dict[probe, float]``.
        - Each value ``[T, D]``       → per-token stack.
          ``per_token=False`` (default) returns the aggregate pooled from
          row ``T-1``; ``per_token=True`` returns
          ``(aggregate, per_token_scores)``.

        Mixed shapes (``[D]`` alongside ``[T, D]``) or uneven ``T`` across
        layers raise :class:`SaklasError`. Empty dict raises.

        ``accumulate`` defaults to ``False`` — ad-hoc researcher scoring
        does not pollute the monitor's running-mean history. Pass
        ``True`` to feed this call into the same stats pipeline the TUI
        reads from.
        """
        if not hidden:
            raise SaklasError("score_hidden: no layers provided")

        # Classify shapes up-front.
        shapes = [v.ndim for v in hidden.values()]
        if len(set(shapes)) > 1:
            by_ndim: dict[int, list[int]] = {}
            for layer_idx, t in hidden.items():
                by_ndim.setdefault(t.ndim, []).append(layer_idx)
            detail = ", ".join(
                f"ndim={n} at layers {ls}" for n, ls in sorted(by_ndim.items())
            )
            raise SaklasError(
                "score_hidden: mixed shapes in input; expected either all "
                f"[D] or all [T, D] across layers ({detail})",
            )
        if shapes[0] not in (1, 2):
            raise SaklasError(
                f"score_hidden: expected [D] or [T, D] tensors, got ndim={shapes[0]}",
            )

        # Dim pre-flight: each input tensor's last dim must match any
        # probe that covers that layer. Without this, a shape mismatch
        # would leak a raw torch RuntimeError at the scoring matmul,
        # violating the "all public errors are SaklasError" invariant.
        for layer_idx, t in hidden.items():
            actual_dim = t.shape[-1]
            for probe_name, profile in self._monitor.profiles.items():
                probe_vec = profile.get(layer_idx)
                if probe_vec is None:
                    continue
                expected_dim = probe_vec.shape[-1]
                if expected_dim != actual_dim:
                    raise SaklasError(
                        f"score_hidden: dim mismatch at layer {layer_idx} — "
                        f"got {actual_dim}, probe '{probe_name}' expects "
                        f"{expected_dim}",
                    )
                break  # one covering probe settles the expected dim

        if shapes[0] == 1:
            if per_token:
                # [D] input + per_token is meaningless.
                raise SaklasError(
                    "score_hidden: per_token=True requires [T, D] input; "
                    "got [D] (single state per layer)",
                )
            # Fall through to the monitor's single-state path.
            return self._monitor.measure_from_hidden(hidden, accumulate=accumulate)

        # [T, D] path — delegate to monitor.score_stack. Wrap its
        # ValueError (uneven T across layers is the only path that
        # can reach here after the shape checks above) so callers
        # catching SaklasError get a uniform exception surface.
        try:
            agg, per_tok = self._monitor.score_stack(
                hidden, accumulate=accumulate,
            )
        except ValueError as exc:
            raise SaklasError(f"score_hidden: {exc}") from exc
        return (agg, per_tok) if per_token else agg

    def _promote_profile(self, profile: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
        return {idx: vec.to(self._device, self._dtype) for idx, vec in profile.items()}

    # -- Monitoring --

    def probe(self, name: str, profile: dict | None = None) -> None:
        if profile is None:
            _, profile = self.extract(name)
        if not self._layer_means:
            self._layer_means = bootstrap_layer_means(
                self._model, self._tokenizer, self._layers, self._model_info,
            )
            self._monitor.layer_means = self._layer_means
        self._monitor.add_probe(name, profile)

    def unprobe(self, name: str) -> None:
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
        applied_steering: str | None = None,
        *,
        return_hidden: bool = False,
    ) -> GenerationResult:
        """Shared post-generation: decode, measure probes, build result, update history."""
        token_count = len(generated_ids)
        tok_per_sec = token_count / elapsed if elapsed > MIN_ELAPSED_FOR_RATE else 0.0
        response_ids = generated_ids[self._gen_state.thinking_end_idx:]
        if (
            self._gen_state.finish_reason == "stop_sequence"
            and self._gen_state.response_text is not None
        ):
            text = self._gen_state.response_text
        else:
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

        hidden_states: dict[int, torch.Tensor] | None = None
        if return_hidden and generated_ids:
            raw = self._capture.stacked()  # {layer_idx: [n_captured, D] on device}
            n = len(generated_ids)
            trimmed: dict[int, torch.Tensor] = {}
            for layer_idx, h in raw.items():
                # Same EOS off-by-one trim score_per_token applies.
                if h.shape[0] > n:
                    h = h[:n]
                elif h.shape[0] < n:
                    # Under-capture: shouldn't happen on this code path, but
                    # skip the layer rather than returning a short tensor the
                    # caller would misalign with generated_ids.
                    continue
                # `.to("cpu")` from a device tensor allocates a fresh
                # CPU tensor already; no redundant `.clone()` needed.
                trimmed[layer_idx] = h.detach().to("cpu")
            hidden_states = trimmed

        result = GenerationResult(
            text=text, tokens=list(generated_ids), token_count=token_count,
            tok_per_sec=tok_per_sec, elapsed=elapsed,
            readings=readings, vectors=vector_snapshot,
            prompt_tokens=prompt_tokens,
            finish_reason=self._gen_state.finish_reason,
            logprobs=logprobs_list,
            applied_steering=applied_steering,
            hidden_states=hidden_states,
        )
        self._last_result = result

        if readings:
            self.events.emit(
                ProbeScored(readings={name: r.mean for name, r in readings.items()}),
            )

        if not stateless:
            if isinstance(input, str):
                self._history.append({"role": "user", "content": input})
            if text.strip():
                self._history.append({"role": "assistant", "content": text})

        return result

    def _generation_preamble(self, input, raw, thinking, stateless=False):
        """Shared input prep + gen-state reset.

        Steering is NOT installed here — the caller is expected to hold a
        ``session.steering()`` scope open across the generation.
        """
        use_thinking = thinking and supports_thinking(self._tokenizer)
        input_ids = self._prepare_input(input, raw=raw, thinking=use_thinking, stateless=stateless)
        self._gen_state.reset()
        return input_ids, use_thinking, int(input_ids.shape[1])

    def _compose_gen_config(
        self, sampling: SamplingConfig | None,
    ) -> "GenerationConfig":
        """Build a per-call GenerationConfig from session defaults + sampling.

        Does NOT mutate ``self.config`` — returns a new frozen instance the
        generation worker holds for its lifetime.  ``None`` fields in
        ``sampling`` fall through to the session default.
        """
        from dataclasses import replace as _replace

        if sampling is None:
            return self.config
        overrides: dict = {}
        if sampling.temperature is not None:
            overrides["temperature"] = sampling.temperature
        if sampling.top_p is not None:
            overrides["top_p"] = sampling.top_p
        if sampling.top_k is not None:
            overrides["top_k"] = sampling.top_k
        if sampling.max_tokens is not None:
            overrides["max_new_tokens"] = sampling.max_tokens
        if not overrides:
            return self.config
        return _replace(self.config, **overrides)

    # -- Generation: core --

    def _generate_core(
        self,
        input,
        *,
        steering: "str | Steering | None" = None,
        sampling: SamplingConfig | None = None,
        stateless: bool = False,
        raw: bool = False,
        thinking: bool | None = None,
        on_token: Callable[..., None] | None = None,
    ) -> GenerationResult:
        """Shared generation implementation.

        Holds the gen lock + re-entry guard for the duration of the call,
        composes a per-call GenerationConfig, opens an internal steering
        scope (if any), runs ``generate_steered`` with capture attached,
        and finalizes the result.  ``generate`` and ``generate_stream``
        are thin wrappers around this.
        """
        if not self._gen_lock.acquire(blocking=False):
            raise ConcurrentGenerationError("Generation already in progress")
        if self._gen_active:
            self._gen_lock.release()
            raise ConcurrentGenerationError("session generation already in flight")
        self._gen_active = True

        steering_obj = Steering.from_value(steering)
        # Effective thinking: explicit kwarg wins; else Steering.thinking;
        # else auto-detect from tokenizer.
        if thinking is None:
            if steering_obj is not None and steering_obj.thinking is not None:
                use_thinking_req = steering_obj.thinking
            else:
                use_thinking_req = supports_thinking(self._tokenizer)
        else:
            use_thinking_req = thinking

        gen_config = self._compose_gen_config(sampling)
        lp_count = sampling.logprobs if sampling is not None else None
        seed = sampling.seed if sampling is not None else None
        stop_tuple = sampling.stop if sampling is not None else None
        stop_list = list(stop_tuple) if stop_tuple else None
        logit_bias = sampling.logit_bias if sampling is not None else None
        presence_penalty = sampling.presence_penalty if sampling is not None else 0.0
        frequency_penalty = sampling.frequency_penalty if sampling is not None else 0.0

        logprobs_list: list | None = [] if lp_count is not None else None
        trait_token_counter = [0]

        def _token_tap(text, is_thinking, tid, lp, top_lp, perplexity):
            if logprobs_list is not None and tid >= 0 and not is_thinking:
                logprobs_list.append((tid, lp if lp is not None else 0.0, top_lp or []))
            if on_token is not None:
                on_token(text, is_thinking, tid, lp, top_lp, perplexity)
            # Inline per-token scoring for live SSE trait subscribers.
            if self._trait_queues and self._monitor.probe_names:
                latest_hidden = {
                    layer_idx: bucket[-1]
                    for layer_idx, bucket in self._capture._per_layer.items()
                    if bucket
                }
                if latest_hidden:
                    scores = self._monitor.score_single_token(latest_hidden)
                    event = ("token", trait_token_counter[0], text, is_thinking, scores)
                    trait_token_counter[0] += 1
                    with self._trait_lock:
                        for lp_ref, q in list(self._trait_queues):
                            try:
                                lp_ref.call_soon_threadsafe(q.put_nowait, event)
                            except Exception:
                                pass

        steering_cm = None
        if steering_obj is not None and steering_obj.alphas:
            steering_cm = self.steering(steering_obj)

        def _snapshot_alphas() -> dict[str, float]:
            """Flatten the steering stack to ``{name: alpha}`` — triggers
            stripped for ``GenerationResult.vectors``.  Ablation entries
            surface under their ``!<target>`` key with the term's coeff."""
            snap: dict[str, float] = {}
            for name, entry in self._flatten_steering_stack().items():
                if isinstance(entry, AblationTerm):
                    snap[name] = entry.coeff
                    continue
                snap[name] = entry[0]
            return snap

        vector_snapshot: dict[str, float] = (
            _snapshot_alphas()
            if self._steering_stack or steering_cm is not None
            else {}
        )

        try:
            if steering_cm is not None:
                steering_cm.__enter__()
            input_ids, use_thinking, prompt_tokens = self._generation_preamble(
                input, raw, use_thinking_req, stateless=stateless,
            )
            # Refresh snapshot now that steering is pushed (first-scope case).
            vector_snapshot = _snapshot_alphas()

            want_hidden = bool(sampling and sampling.return_hidden)
            self.events.emit(GenerationStarted(input=input, stateless=stateless))
            self._begin_capture(widen=want_hidden)
            self._monitor.begin_live()
            # Reset the steering manager's TriggerContext for this generation.
            # ``generate_steered`` mutates it at lifecycle boundaries; hooks
            # read it on each forward.
            self._steering.ctx.reset()
            try:
                start = time.monotonic()
                generated_ids = generate_steered(
                    self._model, self._tokenizer, input_ids,
                    gen_config, self._gen_state, thinking=use_thinking,
                    on_token=_token_tap,
                    seed=seed, stop=stop_list, logit_bias=logit_bias,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                    logprobs=lp_count,
                    trigger_ctx=self._steering.ctx,
                )
                elapsed = time.monotonic() - start
            finally:
                self._gen_state.stop_requested.set()
                self._end_capture()
                if steering_cm is not None:
                    steering_cm.__exit__(None, None, None)
                    steering_cm = None

            applied_steering = (
                str(steering_obj) if steering_obj is not None else None
            )
            result = self._finalize_generation(
                input, generated_ids, elapsed, vector_snapshot,
                prompt_tokens=prompt_tokens, stateless=stateless,
                logprobs_list=logprobs_list,
                applied_steering=applied_steering,
                return_hidden=want_hidden,
            )
            self._monitor.end_live()
            self.events.emit(GenerationFinished(result=result))
            return result
        except BaseException:
            # If we bailed before the inner finally ran (e.g. preamble threw),
            # make sure the steering scope is popped.
            if steering_cm is not None:
                try:
                    steering_cm.__exit__(None, None, None)
                except Exception:
                    pass
            raise
        finally:
            self._monitor.end_live()
            self._gen_active = False
            self._gen_lock.release()

    # -- Generation: blocking --

    def generate(
        self,
        input,
        *,
        steering: "str | Steering | None" = None,
        sampling: SamplingConfig | None = None,
        stateless: bool = False,
        raw: bool = False,
        thinking: bool | None = None,
        on_token: Callable[..., None] | None = None,
    ) -> GenerationResult:
        """Blocking generation.

        Args:
            input: prompt string or list of message dicts.
            steering: expression string (e.g. ``"0.5 honest + 0.3 warm"``)
                or a pre-built :class:`Steering`.  Pole aliases resolve at
                parse time via ``io.selectors.resolve_pole``.  ``None`` =
                no steering.
            sampling: per-call ``SamplingConfig``.  ``None`` fields fall
                through to the session's ``GenerationConfig`` defaults.
                The session's config is never mutated by this call.
            stateless: do not mutate session history.
            raw: skip chat template, tokenize input string directly.
            thinking: per-call thinking override.  ``None`` = auto-detect
                via ``supports_thinking`` (or ``steering.thinking`` if set).
            on_token: optional callback ``(text, is_thinking, token_id,
                logprob, top_logprobs, perplexity)`` called on each emitted
                token.  ``perplexity`` is ``exp(entropy_nats)`` of the
                pre-temperature, post-steering distribution.
        """
        return self._generate_core(
            input,
            steering=steering,
            sampling=sampling,
            stateless=stateless,
            raw=raw,
            thinking=thinking,
            on_token=on_token,
        )

    # -- Generation: streaming --

    def generate_stream(
        self,
        input,
        *,
        steering: "str | Steering | None" = None,
        sampling: SamplingConfig | None = None,
        stateless: bool = False,
        raw: bool = False,
        thinking: bool | None = None,
    ) -> Iterator[TokenEvent]:
        """Streaming generation.  See :meth:`generate` for kwargs.

        Yields ``TokenEvent`` per token.  On iterator close (normal
        exhaustion, ``GeneratorExit``, or an exception raised through
        ``yield``) the worker is signaled to stop and joined, and the
        underlying ``_generate_core`` cleanup runs — probes detached,
        steering scope popped, lock released.

        ``sampling=SamplingConfig(return_hidden=True)`` works here too:
        per-token events do not carry hidden states (that would break
        the allocation-free hot path), but after iteration completes
        the populated ``session.last_result.hidden_states`` is
        available for round-tripping through :meth:`score_hidden`.
        """
        q: queue.SimpleQueue = queue.SimpleQueue()
        done = object()
        result_holder: list[GenerationResult] = []
        exc_holder: list[BaseException] = []
        idx_counter = [0]

        def _push(text, is_thinking, tid, lp, tlp, perplexity):
            scores: dict[str, float] | None = None
            if self._monitor.probe_names:
                latest_hidden = {
                    layer_idx: bucket[-1]
                    for layer_idx, bucket in self._capture._per_layer.items()
                    if bucket
                }
                if latest_hidden:
                    scores = self._monitor.score_single_token(latest_hidden)
                    self._monitor.update_live(scores)
            event = TokenEvent(
                text=text, token_id=tid, index=idx_counter[0],
                thinking=is_thinking, logprob=lp, top_logprobs=tlp,
                scores=scores, perplexity=perplexity,
            )
            idx_counter[0] += 1
            q.put(event)

        def _worker():
            try:
                result = self._generate_core(
                    input,
                    steering=steering,
                    sampling=sampling,
                    stateless=stateless,
                    raw=raw,
                    thinking=thinking,
                    on_token=_push,
                )
                result_holder.append(result)
            except BaseException as e:
                exc_holder.append(e)
            finally:
                q.put(done)

        worker = threading.Thread(target=_worker, daemon=True)
        worker.start()

        try:
            while True:
                item = q.get()
                if item is done:
                    break
                yield item
        finally:
            self._gen_state.stop_requested.set()
            worker.join(timeout=5.0)
            if exc_holder and not result_holder:
                raise exc_holder[0]

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
