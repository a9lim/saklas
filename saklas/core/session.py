"""SaklasSession — unified backend for saklas's programmatic API and TUI."""
from __future__ import annotations
import asyncio
import logging
import os
import pathlib
import queue
import re
import threading
import time
from enum import IntEnum
from typing import Any, Callable, Iterator, Literal, overload

import torch

from saklas.core.errors import SaklasError, StaleSidecarError
from saklas.core.events import (
    EventBus,
    GenerationFinished,
    GenerationStarted,
    ProbeScored,
    SteeringApplied,
    SteeringCleared,
)
from saklas.core.generation import GenerationConfig, GenerationState, build_chat_input, generate_steered, supports_thinking
from saklas.core.hooks import HiddenCapture, SteeringManager
from saklas.core.model import load_model, get_layers, get_model_info
from saklas.core.monitor import TraitMonitor
from saklas.io.packs import PackFormatError, PackMetadata, hash_folder_files
from saklas.io.paths import concept_dir
from saklas.io.probes_bootstrap import bootstrap_probes, bootstrap_layer_means
from saklas.core.profile import Profile
from saklas.core.results import GenerationResult, TokenEvent, ProbeReadings
from saklas.core.sampling import SamplingConfig
from saklas.core.steering import Steering
from saklas.core.steering_expr import AblationTerm
from saklas.core.triggers import Trigger
from saklas.core.vectors import load_profile as _load_profile

_log = logging.getLogger(__name__)

# Hybrid linear-attention models (qwen3.6-27b, lfm2, etc.) carry a
# recurrent state (``conv_states`` + ``recurrent_states``) per LA layer
# alongside the dynamic K/V cache.  ``DynamicLayer.crop`` truncates K/V
# but transformers' ``LinearAttentionLayer.crop`` is a documented no-op
# — there is no sequence dimension to truncate on a recurrent state.
# That breaks ``cache_prefix`` correctness on hybrid models: after a
# generate, the LA state has been advanced through both prefix AND
# suffix tokens; the next reuse of the cached prefix would resume from
# a polluted state, silently producing wrong outputs.
#
# Fix: on prefix install, snapshot each LA layer's ``conv_states`` /
# ``recurrent_states`` (cheap — bounded-size tensors, kernel-sized conv
# state and ``(num_heads, head_dim, head_dim)`` recurrent state).  On
# ``crop``, restore from the snapshot in-place (preserving the
# cudagraph-static address ``lazy_initialization`` set up).
#
# Patch is installed at module import; idempotent.  No-op when
# transformers doesn't expose the LA cache classes (older versions).
def _install_la_cache_patch() -> bool:
    try:
        from transformers.cache_utils import (
            LinearAttentionLayer,
            LinearAttentionAndFullAttentionLayer,
            DynamicLayer,
        )
    except ImportError:
        return False

    if getattr(LinearAttentionLayer, "_saklas_crop_patched", False):
        return False

    _orig_la_crop = LinearAttentionLayer.crop  # documented no-op upstream

    def _save_la_snapshot(layer) -> None:
        snap = {}
        if getattr(layer, "is_conv_states_initialized", False):
            snap["conv"] = layer.conv_states.detach().clone()
        if getattr(layer, "is_recurrent_states_initialized", False):
            snap["recurrent"] = layer.recurrent_states.detach().clone()
        layer._saklas_la_snapshot = snap if snap else None

    def _la_crop_with_restore(self, max_length: int):
        _orig_la_crop(self, max_length)
        snap = getattr(self, "_saklas_la_snapshot", None)
        if not snap:
            return
        # ``conv_states`` / ``recurrent_states`` were created during the
        # prefill forward, which runs inside ``torch.inference_mode()``,
        # so the underlying tensors are inference tensors.  In-place
        # ``.copy_(...)`` from outside inference_mode raises
        # ``RuntimeError: Inplace update to inference tensor outside
        # InferenceMode is not allowed``.  Wrap the restore so the
        # in-place mutation is legal regardless of caller context.
        with torch.inference_mode():
            if "conv" in snap and getattr(self, "is_conv_states_initialized", False):
                self.conv_states.copy_(snap["conv"])
            if "recurrent" in snap and getattr(self, "is_recurrent_states_initialized", False):
                self.recurrent_states.copy_(snap["recurrent"])

    def _hybrid_crop(self, max_length: int):
        DynamicLayer.crop(self, max_length)
        _la_crop_with_restore(self, max_length)

    setattr(LinearAttentionLayer, "crop", _la_crop_with_restore)
    setattr(LinearAttentionAndFullAttentionLayer, "crop", _hybrid_crop)
    setattr(LinearAttentionLayer, "_saklas_save_snapshot", _save_la_snapshot)
    setattr(LinearAttentionLayer, "_saklas_crop_patched", True)
    return True


_install_la_cache_patch()


def _snapshot_la_layers(cache) -> None:
    """Walk a Cache's layers and snapshot any linear-attention state.

    Called from :meth:`SaklasSession.cache_prefix` right after the
    prefill forward, before the cache is stored.  No-op for caches with
    no LA layers (standard transformer models).
    """
    layers = getattr(cache, "layers", None)
    if not layers:
        return
    for layer in layers:
        save = getattr(layer, "_saklas_save_snapshot", None)
        if save is not None:
            # ``_saklas_save_snapshot`` is ``setattr``-ed on the
            # ``LinearAttentionLayer`` class, so attribute lookup on an
            # instance returns a bound method — ``self`` (the layer) is
            # already passed automatically. Calling ``save(layer)`` here
            # would hand layer through *twice*, raising
            # ``TypeError: takes 1 positional argument but 2 were given``.
            save()


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

class GenState(IntEnum):
    """Lifecycle phases of a single generation call.

    Replaces the v1.x ``_gen_active: bool`` flag with a typed state so the
    five-handle teardown (lock, steering scope CM, capture, monitor live,
    threading lock) is self-documenting.

    Transitions live in :meth:`SaklasSession._generate_core`:

    - ``IDLE`` → ``PREAMBLE``: lock acquired, re-entry guard passed.
    - ``PREAMBLE`` → ``RUNNING``: capture attached, monitor ``begin_live``,
      steering :class:`TriggerContext` reset; ``generate_steered`` enters.
    - ``RUNNING`` → ``FINALIZING``: inner ``finally`` ran — capture detached,
      steering scope exited; monitor ``end_live`` / lock release pending.
    - ``FINALIZING`` → ``IDLE``: outer ``finally`` ran.

    The threading ``_gen_lock`` primitive stays alongside this enum — the
    enum makes the state field typed and self-documenting; the lock still
    enforces single-flight at the Python level.
    """

    IDLE = 0
    PREAMBLE = 1
    RUNNING = 2
    FINALIZING = 3


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


class ConcurrentExtractionError(RuntimeError, SaklasError):
    """Raised when ``session.extract`` is called while a generation is in flight.

    Mirrors :class:`ConcurrentGenerationError` — extraction runs forward
    passes through the model and would race with an active generation if
    allowed to overlap.  The gate is a one-line ``GenState`` check at the
    top of :meth:`SaklasSession.extract` (the pipeline itself is unaware
    of generation state).
    """

    def user_message(self) -> tuple[int, str]:
        return (409, str(self) or self.__class__.__name__)


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

    ``_injection_mode`` and ``_theta_max`` carry the per-call overrides
    forward through the stack so nested scopes can flip the steering
    math (or the angular cap) for the duration of the inner block.
    ``None`` means "inherit": stack walks the LIFO from top, picking the
    first non-``None`` value, falling through to the session default if
    every scope is ``None``.
    """

    __slots__ = (
        "_session", "_entries", "_entered",
        "_injection_mode", "_theta_max", "_projection_metric",
        "_synthetic_snapshots",
    )

    def __init__(
        self,
        session: "SaklasSession",
        entries: dict[str, SteeringStackEntry],
        *,
        injection_mode: str | None = None,
        theta_max: float | None = None,
        projection_metric: str | None = None,
        synthetic_snapshots: dict[str, "object"] | None = None,
    ) -> None:
        self._session = session
        self._entries = entries
        self._entered = False
        self._injection_mode = injection_mode
        self._theta_max = theta_max
        self._projection_metric = projection_metric
        # Pre-materialize snapshots of any synthetic-projection keys
        # this scope wrote to ``session._profiles`` — value is the prior
        # binding (or :data:`_PROFILE_ABSENT` when the key was unset).
        # Restored on ``__exit__`` so nested scopes that re-materialize
        # the same ``a|b`` synthetic key with a different metric don't
        # leak the inner tensor back into the outer scope's hooks.
        self._synthetic_snapshots: dict[str, object] = (
            dict(synthetic_snapshots) if synthetic_snapshots else {}
        )

    def __enter__(self) -> "_SteeringContext":
        # _push_steering rolls its own stack entry back if _rebuild_steering_hooks
        # raises (e.g. VectorNotRegisteredError).  __enter__ only flips
        # `_entered=True` AFTER a clean push so a mid-__enter__ failure leaves
        # no stale state for __exit__ to pop.
        self._session._push_steering(
            self._entries,
            injection_mode=self._injection_mode,
            theta_max=self._theta_max,
            projection_metric=self._projection_metric,
        )
        self._entered = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._entered:
            self._session._pop_steering()
            self._entered = False
        # Restore any pre-existing values for synthetic-projection
        # keys this scope clobbered.  Runs even if ``_pop_steering``
        # raised — best-effort cleanup keeps the registry consistent
        # across nested scope unwinding.  Out of __exit__'s exception
        # path on purpose: registry mutation should not swallow user
        # errors raised during the steered block.
        snapshots = self._synthetic_snapshots
        if snapshots:
            profiles = self._session._profiles
            for key, prev in snapshots.items():
                if prev is _PROFILE_ABSENT:
                    profiles.pop(key, None)
                else:
                    profiles[key] = prev  # type: ignore[assignment]
            self._synthetic_snapshots = {}


# Sentinel for ``_SteeringContext._synthetic_snapshots`` entries —
# distinguishes "key was previously absent" from "key was previously
# bound to None" (the latter shouldn't happen in practice but the
# distinction keeps restore semantics unambiguous).
_PROFILE_ABSENT = object()


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
        dtype=None,
        quantize: str | None = None,
        probes: list[str] | None = None,
        system_prompt: str | None = None,
        max_tokens: int = 1024,
        injection_mode: str = "angular",
        theta_max: float | None = None,
        extraction_method: str = "dim",
        projection_metric: str = "mahalanobis",
        dls: bool = True,
        compile: bool = True,
        compile_mode: str | None = None,
        cuda_graphs: bool = True,
    ) -> "SaklasSession":
        """Load a HF model + tokenizer and return a fully initialized session.

        This is the primary entry point for library users; it owns all the
        HF-loading heavy lifting. To wrap an already-loaded model use the
        plain ``__init__(model, tokenizer, ...)`` form.

        ``injection_mode`` selects the steering math: ``"angular"``
        (default, v2.1+) maps user α to a rotation angle; ``"additive"``
        is the legacy v1.x additive + norm-preserving path.  ``theta_max``
        sets the maximum rotation angle under angular mode (default π/2,
        i.e. α=1 fully aligns the residual with the concept direction).
        ``projection_metric`` selects the metric used when materializing
        ``~`` / ``|`` projection terms in steering expressions:
        ``"mahalanobis"`` (default since v2.1) uses the closed-form
        LEACE projector against the per-model whitener — provably erases
        linearly-decodable information along ``onto`` from ``base``;
        ``"euclidean"`` is plain Gram-Schmidt (the v2.0/v2.1 behavior).

        ``dls`` toggles the discriminative-layer-selection mask at
        extraction time (v2.1+).  When ``True`` (default), centered DLS
        per Dang & Ngo (2026) Eq. 9 drops layers where pos- and
        neg-class means project to the same side of the neutral
        baseline along ``d̂``.  Replaces the v2.0–v2.1 ``edge_drop``
        heuristic (gone in v2.1); ``--legacy`` flips this to ``False``.

        ``compile`` (default ``True``) auto-enables ``torch.compile`` on
        CUDA — kernel fusion via inductor, typically 1.2–1.5× decode
        tok/s on small models.  Auto-skipped on MPS/CPU.  Pass ``False``
        to debug architecture-specific compile issues; the angular hook
        scalars are tensor-pinned (v2.2) so a single compiled artifact
        survives α changes between generations.

        ``cuda_graphs`` (default ``True``, Phase B v2.2) auto-enables
        ``transformers.StaticCache`` + ``torch.compile(mode=
        "reduce-overhead")`` on CUDA-supported architectures.  Static
        K/V buffers across decode steps mean inductor can capture CUDA
        graphs internally — typical 1.5–2.5× decode tok/s on small
        models *on top of* the kernel-fusion win from ``compile=True``.
        Auto-skipped on MPS/CPU and on architectures whose StaticCache
        construction fails (probed at session init; the fallback reason
        is logged once).  Pass ``False`` to use DynamicCache (the v2.1
        path) — useful when debugging cache-related issues or when a
        specific architecture has subtle StaticCache quirks.

        ``compile_mode`` (default ``None`` → auto-select) overrides the
        torch.compile mode.  When None, the session picks
        ``"reduce-overhead"`` if ``cuda_graphs`` is on (paired with
        StaticCache for full graph capture) and ``"default"`` otherwise
        (kernel fusion only).  Pass an explicit value to force a
        specific mode regardless of the cuda_graphs decision.
        """
        # Load WITHOUT compile so the StaticCache probe runs against the
        # bare nn.Module (probing through the OptimizedModule wrapper
        # forwards correctly via __getattr__, but avoiding the wrapper
        # at probe time keeps the failure mode "StaticCache constructor
        # raised" rather than "compile + probe interaction").  We then
        # decide ``compile_mode`` based on the probe outcome and apply
        # ``torch.compile`` manually below.  This closes the order-of-
        # operations bug Codex flagged in v2.2 review: the previous
        # shape committed to ``"reduce-overhead"`` based on the
        # *requested* ``cuda_graphs=True`` before the probe could veto,
        # so arch-failed sessions ran DynamicCache under a graph-capture
        # compile mode (mode-and-cache mismatch).
        model, tokenizer = load_model(
            model_id,
            quantize=quantize,
            device=device,
            dtype=dtype,
            compile=False,
        )

        cg_supported = False
        cg_reason: str | None = None
        device_obj = next(model.parameters()).device
        if cuda_graphs:
            from saklas.core.cuda_graphs import is_cuda_graphs_supported
            cg_supported, cg_reason = is_cuda_graphs_supported(
                model, device_obj,
            )

        # Resolve compile_mode now that the probe outcome is known.
        # ``"reduce-overhead"`` captures CUDA graphs internally for
        # fixed-shape inference regions and only pays off paired with
        # StaticCache; ``"default"`` is the kernel-fusion-only fallback
        # that composes cleanly with DynamicCache.  An explicit
        # ``compile_mode`` arg overrides regardless of probe outcome —
        # power users (benching, debugging) can force a mismatch on
        # purpose.
        effective_compile_mode = compile_mode
        if effective_compile_mode is None:
            effective_compile_mode = (
                "reduce-overhead" if cg_supported else "default"
            )

        # Apply compile manually with the resolved mode.  Keeps the
        # device gating, dynamo availability check, and skip-on-non-cuda
        # log behavior aligned with what ``load_model`` would have
        # produced if we'd let it own the compile call.
        if compile and device_obj.type == "cuda":
            try:
                import torch._dynamo  # noqa: F401
            except ImportError:
                _log.info("torch.compile unavailable (no _dynamo); skipping")
            else:
                _log.info(
                    "Compiling model with torch.compile(mode=%r)",
                    effective_compile_mode,
                )
                model = torch.compile(
                    model, mode=effective_compile_mode, dynamic=None,
                )
        elif compile:
            _log.info(
                "compile=True but device=%s — skipping torch.compile "
                "(supported only on CUDA)",
                device_obj.type,
            )

        # ``__init__`` re-runs the probe on its own.  We could plumb the
        # result through to skip the duplicate StaticCache(max_cache_len=1)
        # construction, but that put a private-flavored kwarg on the
        # public constructor signature where direct-call users could
        # set it (Codex review flagged this).  The probe is one cache
        # allocation of one position — cost is dwarfed by the model
        # weight load that already ran.  ``warn_once`` dedupes on
        # ``id(model)`` so the user-visible logging fires exactly once
        # regardless.
        return cls(
            model, tokenizer,
            probes=probes,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            injection_mode=injection_mode,
            theta_max=theta_max,
            extraction_method=extraction_method,
            projection_metric=projection_metric,
            dls=dls,
            cuda_graphs=cuda_graphs,
        )

    def __init__(
        self,
        model,
        tokenizer,
        *,
        probes: list[str] | None = None,
        system_prompt: str | None = None,
        max_tokens: int = 1024,
        injection_mode: str = "angular",
        theta_max: float | None = None,
        extraction_method: str = "dim",
        projection_metric: str = "mahalanobis",
        dls: bool = True,
        cuda_graphs: bool = True,
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

        # Transient steering manager — used only during generation.  The
        # session-level injection_mode + θ_max are stamped onto every hook
        # at apply time; per-call ``Steering.injection_mode`` overrides
        # this default in ``_rebuild_steering_hooks``.
        from saklas.core.hooks import DEFAULT_THETA_MAX
        if injection_mode not in ("angular", "additive"):
            raise ValueError(
                f"injection_mode must be 'angular' or 'additive', "
                f"got {injection_mode!r}"
            )
        self._injection_mode: str = injection_mode
        self._theta_max: float = (
            DEFAULT_THETA_MAX if theta_max is None else float(theta_max)
        )
        if projection_metric not in ("mahalanobis", "euclidean"):
            raise ValueError(
                f"projection_metric must be 'mahalanobis' or 'euclidean', "
                f"got {projection_metric!r}"
            )
        # Default for runtime ``~`` / ``|`` projection.  Mahalanobis is
        # default since v2.1: per-layer ``project_profile`` calls receive
        # ``self.whitener`` so projection erases linearly-decodable
        # concept information along ``onto`` (closed-form LEACE per
        # Belrose et al. 2023).  ``"euclidean"`` keeps the v2.0/v2.1
        # plain Gram-Schmidt semantics — what ``--legacy`` selects.
        # When the whitener is unavailable (e.g. ``probes=[]`` session
        # with no neutral-activation cache yet), the materialize site
        # falls back to Euclidean per layer transparently.
        self._projection_metric: str = projection_metric
        self._steering = SteeringManager(
            injection_mode=self._injection_mode,  # type: ignore[arg-type]
            theta_max=self._theta_max,
        )
        # CUDA-graphs / StaticCache routing (Phase B, v2.2).  Probe
        # support once at construction so the per-generation hot path
        # only consults a boolean.  Off when (a) user opted out, (b)
        # device != cuda, or (c) the model's StaticCache constructor
        # raises (architecture-specific quirks).  The fallback reason
        # is logged once via :func:`saklas.core.cuda_graphs.warn_once`
        # which dedupes on ``id(model)``, so when ``from_pretrained``
        # already probed for its compile_mode decision and we re-probe
        # here, the user only sees one message.
        self._cuda_graphs_requested: bool = bool(cuda_graphs)
        self._cuda_graphs_active: bool = False
        if self._cuda_graphs_requested:
            from saklas.core.cuda_graphs import (
                is_cuda_graphs_supported, warn_once,
            )
            supported, reason = is_cuda_graphs_supported(
                self._model, self._device,
            )
            if supported:
                self._cuda_graphs_active = True
            elif reason is not None:
                warn_once(self._model, reason)
        # LIFO stack of per-scope entries dicts pushed by session.steering().
        # Each entry is ``{name: (alpha, Trigger)}`` — triggers are
        # preserved through stack flattening so nested scopes with
        # different trigger regimes compose cleanly.  The flattened head
        # (later entries overwrite earlier ones) is what the steering
        # manager installs when a generation begins.
        self._steering_stack: list[dict[str, SteeringStackEntry]] = []
        # Parallel LIFO of per-scope (injection_mode, theta_max,
        # projection_metric) overrides.  Each element matches its sibling
        # in ``_steering_stack``; ``None`` means "inherit".  Walked
        # top-down by ``_resolve_steering_override`` to find the active
        # value, with the session default as the floor.  Triplet shape so
        # all three knobs (steering math, rotation cap, projection
        # metric) compose under nesting.
        self._steering_override_stack: list[
            tuple[str | None, float | None, str | None]
        ] = []

        # Synchronous event bus.  Emits on extraction, steering enter/exit,
        # probe scoring, generation start/finish.  Subscribers run on the
        # emit thread — async consumers must hop via call_soon_threadsafe
        # inside their callback.
        self.events: EventBus = EventBus()

        # Transient per-token hidden-state capture — attached around
        # generate_steered when probes are active so scoring happens
        # without a second forward pass.
        self._capture = HiddenCapture()

        # Reentrant — ``_generate_core`` acquires it for the whole gen,
        # then enters an internal ``self.steering(...)`` scope which
        # routes through ``_push_steering`` → re-acquires the lock from
        # the same thread.  The single-in-flight invariant is enforced
        # by the ``_gen_phase`` state check, not by the lock owner
        # count, so RLock is correct: cross-thread re-entry blocks
        # (which is the property fix #4 wants), same-thread re-entry
        # passes (which keeps the internal steering scope from
        # deadlocking against itself).
        self._gen_lock = threading.RLock()
        # Bypass flag for the phase guard in ``_push_steering`` /
        # ``_pop_steering``.  ``_generate_core`` sets this around the
        # ``steering_cm.__exit__()`` it owns at the end of a generation
        # so the legitimate internal cleanup passes through the guard
        # that's there to catch on_token-callback reentry.  Default
        # False — user code never flips this; it's an implementation-
        # detail signal between ``_generate_core`` and the pop path.
        self._internal_steering_pop: bool = False
        # Async-level serializer owned by the HTTP server for back-pressure.
        # Distinct from `_gen_lock` (threading, enforces single-flight at the
        # Python level): `lock` queues concurrent async requests FIFO so they
        # wait rather than 409.  Library-only callers never touch this.
        self.lock: asyncio.Lock = asyncio.Lock()
        self._gen_state = GenerationState()
        # Typed lifecycle phase of the current generation (or ``IDLE`` between
        # gens).  Re-entry guard between preamble and finalize: prevents a
        # pending-action dispatch from double-attaching capture/steering
        # hooks and leaking them.  See :class:`GenState` for transitions.
        # Distinct from ``_gen_state`` (the per-call ``GenerationState``
        # holding token queue, finish_reason, etc.) — the names are close
        # because the enum field is the *session*'s view of state, while
        # ``_gen_state`` is the *generator's*.
        self._gen_phase: GenState = GenState.IDLE

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

        # Order matters: layer_means + neutral_activations + whitener must
        # exist BEFORE ``bootstrap_probes`` runs, because v2.1+ DiM
        # extraction uses the whitener for Mahalanobis-flavored share
        # allocation.  Pre-v2.1 ordering computed layer_means *after* probe
        # extraction (just for monitor centering) — the flip is the
        # extract-time dependency on the activation covariance.  When
        # ``probe_categories`` is empty there's nothing to extract, so we
        # skip the whitener build to keep ``probes=[]`` sessions cheap;
        # ad-hoc later extraction lazily builds via ``self.whitener``.
        self._layer_means: dict[int, torch.Tensor] = {}
        self._whitener: Any = None
        if probe_categories:
            self._layer_means = bootstrap_layer_means(
                self._model, self._tokenizer, self._layers, self._model_info,
            )
            self._whitener = self._build_whitener_from_cache_or_compute()

        # Stash for later session.extract calls — same default applies
        # to ad-hoc extraction unless the caller overrides per-call.
        if extraction_method not in ("dim", "pca"):
            raise ValueError(
                f"extraction_method must be 'dim' or 'pca', "
                f"got {extraction_method!r}"
            )
        self._extraction_method: str = extraction_method
        # v2.1+: DLS toggle stored on the session so ad-hoc
        # ``session.extract`` calls (via ``ExtractionPipeline``) inherit
        # it without re-passing.  ``--legacy`` sets this to False.
        self._dls: bool = bool(dls)

        probe_profiles: dict[str, dict] = {}
        if probe_categories:
            probe_profiles = bootstrap_probes(
                self._model, self._tokenizer, self._layers, self._model_info,
                probe_categories,
                method=extraction_method,
                whitener=self._whitener if extraction_method == "dim" else None,
                layer_means=self._layer_means,
                dls=self._dls,
            )

        self._monitor = TraitMonitor(probe_profiles, self._layer_means)

        # Prefix KV cache (opt-in, off by default).  Populated by
        # :meth:`cache_prefix`; consumed by :meth:`_generate_core` when the
        # incoming ``input_ids`` start with the cached prefix.  Shape:
        # ``(prefix_token_ids: torch.Tensor [seq_len] long, past_key_values,
        # prefix_len: int)``.  ``past_key_values`` is the live HF cache
        # (DynamicCache et al.) returned by the prefix-prefill forward pass
        # — generation cropping back to ``prefix_len`` after each consuming
        # call keeps it reusable.  Invalidated on any state change that
        # would alter the cached prefix's hidden-state semantics: steering
        # push/pop/steer/unsteer, probe install/remove, profile mutation.
        self._prefix_cache: tuple[torch.Tensor, object, int] | None = None

        # Concept-extraction pipeline.  Constructed once so the session
        # holds a single live instance; the pipeline takes the structural
        # dependencies it needs (model handle / pack writer / registry /
        # event bus) instead of a back-reference.  ``self`` satisfies all
        # three protocols implicitly — see :mod:`saklas.core.extraction`.
        from saklas.core.extraction import ExtractionPipeline as _Pipeline
        self._extraction = _Pipeline(self, self, self, self.events)

    # -- ModelHandle protocol surface (consumed by ExtractionPipeline) --

    @property
    def model(self) -> torch.nn.Module:
        """Live HF model.  Part of the :class:`~saklas.core.extraction.ModelHandle` protocol."""
        return self._model

    @property
    def tokenizer(self):
        """Live HF tokenizer.  Part of the :class:`~saklas.core.extraction.ModelHandle` protocol."""
        return self._tokenizer

    @property
    def device(self) -> torch.device:
        """Model device.  Part of the :class:`~saklas.core.extraction.ModelHandle` protocol."""
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """Model dtype.  Part of the :class:`~saklas.core.extraction.ModelHandle` protocol."""
        return self._dtype

    @property
    def layers(self):
        """Layer-list accessor.  Part of the :class:`~saklas.core.extraction.ModelHandle` protocol.

        Returns whatever ``get_layers`` produced — typically an
        ``nn.ModuleList``, list-like enough for the downstream
        consumers (``extract_contrastive``, hooks).
        """
        return self._layers

    # -- VectorRegistry protocol surface --
    #
    # ``__contains__`` falls through to ``self._profiles``; ``add`` writes a
    # ``Profile`` back into the registry as a plain dict so the steering
    # hook hot path reads tensors without attribute lookups.  ``has_vector``
    # already covers public membership checks; ``add`` is reserved for the
    # extraction pipeline's eventual write-back.

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        return name in self._profiles

    def add(self, name: str, profile: Profile) -> None:
        """Register a profile under ``name`` — :class:`VectorRegistry` writeback."""
        self._profiles[name] = dict(profile.as_dict())

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

    @property
    def gen_state(self) -> GenState:
        """Lifecycle phase of the current generation (``IDLE`` between gens).

        Read-only window into the session's typed re-entry guard — see
        :class:`GenState` for transitions.  Surfaces to the TUI and any
        external introspector that wants to ask "is a gen running right
        now?" without reaching past the public API.
        """
        return self._gen_phase

    @property
    def is_generating(self) -> bool:
        """``True`` whenever :attr:`gen_state` is not ``GenState.IDLE``."""
        return self._gen_phase is not GenState.IDLE

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

    # -- Neutral baseline (v2.1) --

    @property
    def layer_means(self) -> dict[int, torch.Tensor]:
        """Per-layer neutral baseline means, built lazily on first access.

        Sessions instantiated with ``probes=[]`` skip the eager
        :func:`bootstrap_layer_means` call to keep init cheap.  Callers
        that later need the means — DLS centering at extraction time,
        the Mahalanobis whitener, the trait monitor — hit this
        property, which triggers the bootstrap once and caches the
        result on ``self._layer_means``.  Disk-cached when the
        ``neutral_statements.json`` hash matches the on-disk
        ``layer_means.safetensors``; recomputes otherwise.

        Returns ``{}`` only if the bootstrap path itself fails (model
        not loaded, missing neutrals pack, etc.) — DLS / whitener
        callers fall back to no-baseline behavior in that case.

        v2.1 fix-up: previously DLS extraction read ``self._layer_means``
        directly, which left ``probes=[]`` sessions with an empty dict
        and silently disabled DLS (every layer fell through the
        "missing baseline" conservative-keep branch in
        :func:`compute_dls_mask`).  The property closes that footgun.
        """
        if not self._layer_means:
            try:
                self._layer_means = bootstrap_layer_means(
                    self._model, self._tokenizer, self._layers, self._model_info,
                )
            except Exception as exc:  # pragma: no cover — defensive
                _log.warning(
                    "session.layer_means lazy build failed: %s; "
                    "DLS and Mahalanobis paths will fall back to "
                    "no-baseline behavior", exc,
                )
        return self._layer_means

    # -- Mahalanobis whitener (v2.1) --

    @property
    def whitener(self) -> "Any":
        """Per-layer Mahalanobis whitener; built lazily on first access.

        Used by v2.1+ DiM extraction for Mahalanobis-flavored share
        allocation, by ``vector compare --metric mahalanobis``, and by
        callers that pass a whitener to ``project_profile`` for
        LEACE-style projection.  Returns a
        :class:`saklas.core.mahalanobis.LayerWhitener` or ``None`` if
        construction failed (model is mid-load, neutral activations
        couldn't be computed, etc. — we never raise here, so probe
        scoring stays alive).
        """
        if self._whitener is None:
            self._whitener = self._build_whitener_from_cache_or_compute()
        return self._whitener

    def _build_whitener_from_cache_or_compute(self) -> "Any":
        """Compute or load the per-model whitener.

        Uses ``load_or_compute_neutral_activations`` (alignment.py) for
        disk caching; combines with the in-memory ``_layer_means`` to
        instantiate the :class:`LayerWhitener`.  Soft-fails to ``None``
        on any error — extraction falls back to Euclidean scoring, and
        ``vector compare --metric mahalanobis`` already errors with a
        useful hint when ``LayerWhitener.from_cache`` can't find the
        cache.

        Lazy: only callers who actually need Mahalanobis math (DiM
        extraction at session init, on-demand ``session.whitener``
        access, or ``vector compare --metric mahalanobis``) trigger the
        forward-pass loop over neutral statements.
        """
        from saklas.core.mahalanobis import LayerWhitener
        from saklas.io.alignment import load_or_compute_neutral_activations

        if not self._layer_means:
            # Whitener requires the centering means; if they haven't been
            # built yet, build them now.  This keeps ``session.whitener``
            # working even on ``probes=[]`` sessions where the eager init
            # path was skipped.
            try:
                self._layer_means = bootstrap_layer_means(
                    self._model, self._tokenizer, self._layers, self._model_info,
                )
            except Exception as exc:  # pragma: no cover — defensive
                _log.warning("whitener: layer_means build failed: %s", exc)
                return None
        try:
            neutral_acts = load_or_compute_neutral_activations(
                self._model, self._tokenizer, self._layers,
                model_id=self._model_info.get("model_id", "unknown"),
            )
            return LayerWhitener.from_neutral_activations(
                neutral_acts, self._layer_means,
            )
        except Exception as exc:
            _log.warning(
                "whitener: build failed (%s); DiM extraction will use "
                "Euclidean scoring. Error: %s",
                type(exc).__name__, exc,
            )
            return None

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
        method: str | None = None,
        dls: bool | None = None,
    ) -> tuple[str, Profile]:
        """Extract a steering vector profile and emit ``VectorExtracted``.

        Thin delegate to :class:`saklas.core.extraction.ExtractionPipeline` —
        the pipeline owns folder probing, statement caching, scenario /
        pair generation, contrastive PCA invocation, and pack updates.
        Re-entry is gated against generation: extraction runs forward
        passes through the model and would race an active gen.

        **Default behavior**: tensor cache hits short-circuit.  On
        tensor miss, if ``statements.json`` exists (curated bundled
        pack or local cache), extract directly from it — statements
        are the expensive part and reuse is the sane default.  On
        statements miss, run the full pipeline: generate scenarios →
        generate pairs → save both → extract tensor.

        Flags:

        - ``scenarios=[...]``: explicit scenarios input; bypasses
          scenario generation and ``scenarios.json`` cache.  Written
          to disk after use.  **Also bypasses the tensor cache** —
          supplying fresh scenarios means the caller wants fresh
          pairs, so any cached tensor is stale by definition.
        - ``reuse_scenarios=True``: when regenerating pairs, load
          ``scenarios.json`` from disk if present instead of
          regenerating.  Default False — scenarios are cheap, so the
          full pipeline regenerates them fresh each pair-gen pass.
        - ``force_statements=True``: regenerate ``statements.json``
          from scratch.  **Also bypasses the tensor cache** — same
          reasoning as ``scenarios=[...]``.
        - ``method=None`` (default) inherits ``self._extraction_method``
          (set at session construction; ``"dim"`` unless ``--legacy``
          flipped it to ``"pca"``).  Explicit ``"dim"`` / ``"pca"``
          overrides per-call.
        - ``dls=None`` (default) inherits ``self._dls`` (set at session
          construction; ``True`` unless ``--legacy`` flipped it).
          Explicit ``True`` / ``False`` overrides per-call.

        Pre-v2.1 these defaults were hardcoded to ``method="dim"`` and
        ``dls=True`` regardless of session config — so ``--legacy``
        sessions calling bare ``session.extract(...)`` got the modern
        stack instead of the v2.0 one.  The ``None``-inherits-session
        shape closes that hole.
        """
        # Must hold ``_gen_lock`` to read ``_gen_phase`` race-free against
        # ``_generate_core``, which acquires the lock first then flips
        # ``_gen_phase = PREAMBLE``.  Without the lock, a concurrent
        # ``generate()`` could pass ``extract()``'s gate and then race
        # extraction over model forward passes.  The lock generalizes from
        # "serialize generations" to "serialize all model uses".
        if not self._gen_lock.acquire(blocking=False):
            raise ConcurrentExtractionError(
                "session.extract called while another model use is in flight"
            )
        try:
            if self._gen_phase is not GenState.IDLE:
                raise ConcurrentExtractionError(
                    "session.extract called while a generation is in flight"
                )
            effective_method = (
                method if method is not None else self._extraction_method
            )
            effective_dls = dls if dls is not None else self._dls
            return self._extraction.extract(
                source, baseline,
                scenarios=scenarios,
                reuse_scenarios=reuse_scenarios,
                force_statements=force_statements,
                on_progress=on_progress,
                sae=sae,
                sae_revision=sae_revision,
                namespace=namespace,
                method=effective_method,  # type: ignore[arg-type]
                dls=effective_dls,
            )
        finally:
            self._gen_lock.release()

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
        # Profile addition can change downstream steering composition
        # if a future steering scope references it; conservatively drop
        # the prefix cache so the next gen reprefills under the new
        # registry view.
        self._invalidate_prefix_cache()

    def unsteer(self, name: str) -> None:
        """Remove a steering vector from the registry."""
        self._profiles.pop(name, None)
        self._invalidate_prefix_cache()

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
        snapshots = self._materialize_projections(steering_obj)
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
        # Per-call overrides ride along with the entries.  ``None`` means
        # "inherit"; the resolver folds session defaults at hook-install.
        mode_override = getattr(steering_obj, "injection_mode", None)
        theta_override = getattr(steering_obj, "theta_max", None)
        metric_override = getattr(steering_obj, "projection_metric", None)
        if mode_override is not None and mode_override not in ("angular", "additive"):
            raise ValueError(
                f"Steering.injection_mode must be 'angular' or 'additive', "
                f"got {mode_override!r}"
            )
        if metric_override is not None and metric_override not in (
            "mahalanobis", "euclidean",
        ):
            raise ValueError(
                f"Steering.projection_metric must be 'mahalanobis' or "
                f"'euclidean', got {metric_override!r}"
            )
        return _SteeringContext(
            self, resolved,
            injection_mode=mode_override,
            theta_max=theta_override,
            projection_metric=metric_override,
            synthetic_snapshots=snapshots,
        )

    def _materialize_projections(self, steering: Steering) -> dict[str, object]:
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

        Metric selection (v2.1): the active projection metric is
        resolved via :meth:`_resolve_projection_metric`, which composes
        the per-call ``Steering.projection_metric`` override (if set)
        with any outer-scope override on
        ``_steering_override_stack`` and the session-level default.
        Under ``"mahalanobis"`` (default since v2.1) the call site
        passes ``self.whitener`` to ``project_profile``, which switches
        ``~`` / ``|`` to the closed-form LEACE projector — provably
        erases linearly-decodable concept information along ``onto``.
        Under ``"euclidean"`` we pass ``whitener=None`` and get plain
        Gram-Schmidt (the v2.0/v2.1 behavior).  When the whitener is
        unavailable for this session (no neutral-activation cache, e.g.
        a ``probes=[]`` session that hasn't extracted yet) the call
        gracefully falls back to Euclidean per-layer transparently —
        ``project_profile``'s coverage check handles per-layer misses.

        Returns a snapshot dict ``{syn_key: prev_value_or_PROFILE_ABSENT}``
        of the synthetic-projection bindings this call clobbered, so
        the caller can restore them on scope exit.  Without this
        nested scopes that materialize the same ``a|b`` synthetic
        key under a different ``projection_metric`` would leak the
        inner tensor back into the outer scope's hooks after pop —
        the global ``self._profiles`` registry is shared across all
        active scopes.
        """
        from saklas.core.steering_expr import ProjectedTerm
        from saklas.core.vectors import project_profile

        # Compute once per ``steering()`` call.  The resolver consults
        # the per-call override first, then any outer scope on the
        # override stack (this scope hasn't been pushed yet — it will
        # be on ``__enter__``), then the session default.
        metric = self._resolve_projection_metric(
            getattr(steering, "projection_metric", None),
        )
        whitener = self.whitener if metric == "mahalanobis" else None

        snapshots: dict[str, object] = {}
        for syn_key, val in steering.alphas.items():
            if not isinstance(val, ProjectedTerm):
                continue
            self._ensure_profile_loaded(val.base)
            self._ensure_profile_loaded(val.onto)
            base_tensors = self._profiles[val.base]
            onto_tensors = self._profiles[val.onto]
            projected = project_profile(
                base_tensors, onto_tensors, val.operator,
                whitener=whitener,
            )
            # Snapshot prior binding *before* overwrite so the
            # context manager can restore on exit.  ``setdefault`` —
            # if the same syn_key appears twice in this Steering, only
            # the first occurrence's snapshot matters (subsequent
            # writes are this scope's own, not the outer's).
            if syn_key not in snapshots:
                if syn_key in self._profiles:
                    snapshots[syn_key] = self._profiles[syn_key]
                else:
                    snapshots[syn_key] = _PROFILE_ABSENT
            self._profiles[syn_key] = projected
        return snapshots

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

        Namespace-qualified inputs (``alice/foo``) keep the namespace
        through resolution: the prefix is split off, fed to ``resolve_pole``
        as the ``namespace=`` kwarg so the lookup scopes to that namespace,
        and re-attached to the registry key.  This is what lets two
        installed packs that share a concept name across namespaces stay
        addressable via their fully-qualified form.

        Names already in ``self._profiles`` pass through verbatim — a caller
        who pre-registered under a specific key stays addressed by that key.
        """
        from saklas.io.selectors import resolve_pole

        out: dict[str, tuple[float, Trigger]] = {}
        for name, (alpha, trig) in entries.items():
            if name in self._profiles:
                out[name] = (float(alpha), trig)
                continue
            # Split namespace prefix so re-resolution scopes to the
            # namespace the user originally typed (parser preserves it
            # in the key when supplied).  Bare names leave ``ns=None``
            # so cross-namespace collisions still raise.
            ns: str | None = None
            bare_name = name
            if "/" in name:
                ns, bare_name = name.split("/", 1)
            try:
                canonical, sign, _match, variant = resolve_pole(
                    bare_name, namespace=ns,
                )
            except Exception:
                out[name] = (float(alpha), trig)
                continue
            canonical_qualified = (
                canonical if ns is None else f"{ns}/{canonical}"
            )
            registry_key = (
                canonical_qualified
                if variant == "raw"
                else f"{canonical_qualified}:{variant}"
            )
            if registry_key not in self._profiles:
                try:
                    self._try_autoload_vector(
                        canonical_qualified, variant=variant,
                    )
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

        ``canonical`` may be namespace-qualified (``alice/foo``), in which
        case discovery is scoped to that namespace.  Registered key in
        ``_profiles`` matches the input form: ``canonical`` for raw, and
        ``f"{canonical}:{variant}"`` otherwise — so namespace-qualified
        callers get a namespace-qualified registry key back.
        """
        from saklas.io.selectors import _all_concepts
        from saklas.io.packs import enumerate_variants
        from saklas.core.errors import AmbiguousVariantError, UnknownVariantError
        from saklas.core.vectors import load_profile

        # Split namespace prefix so concept discovery scopes to the
        # namespace the caller specified.  Bare canonicals leave
        # ``namespace=None`` and pick up the first matching concept
        # across every namespace (the historical behavior).
        ns: str | None = None
        bare_canonical = canonical
        if "/" in canonical:
            ns, bare_canonical = canonical.split("/", 1)

        registry_key = canonical if variant == "raw" else f"{canonical}:{variant}"
        available: list[str] = []
        for concept in _all_concepts():
            if ns is not None and concept.namespace != ns:
                continue
            if concept.name != bare_canonical:
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
            # Phase 2 contract: tensor must not be stale relative to the
            # concept's on-disk statements.json.  bootstrap_probes raises
            # StaleSidecarError on the same condition; autoload would
            # otherwise be a silent escape hatch for the same mismatch.
            recorded_sha = _meta.get("statements_sha256") if _meta else None
            stmts_path = concept.folder / "statements.json"
            if recorded_sha and stmts_path.exists():
                from saklas.io.packs import hash_file as _hash_file
                if (
                    _hash_file(stmts_path) != recorded_sha
                    and os.environ.get("SAKLAS_ALLOW_STALE") != "1"
                ):
                    raise StaleSidecarError(
                        f"{concept.namespace}/{bare_canonical}: statements.json has "
                        f"changed since this tensor was extracted "
                        f"(model={self.model_id}). The baked PCA no longer "
                        f"matches the on-disk pairs. Re-extract: "
                        f"`saklas pack refresh {concept.namespace}/{bare_canonical} "
                        f"-m {self.model_id}` — or set SAKLAS_ALLOW_STALE=1 "
                        f"to load the stale tensor anyway."
                    )
            self._profiles[registry_key] = self._promote_profile(profile_dict)
            return

        # Explicit non-raw variant request that didn't resolve → surface the miss.
        if variant != "raw":
            raise UnknownVariantError(
                f"variant '{variant}' not found for '{canonical}' on model "
                f"'{self.model_id}' (available: {sorted(set(available)) or 'none'})"
            )

    def _push_steering(
        self,
        entries: dict[str, SteeringStackEntry],
        *,
        injection_mode: str | None = None,
        theta_max: float | None = None,
        projection_metric: str | None = None,
    ) -> None:
        """Push an entries dict onto the steering stack and rebuild hooks.

        ``injection_mode`` / ``theta_max`` / ``projection_metric`` are
        per-scope overrides; any ``None`` field falls through to the
        next outer scope (LIFO walk) and ultimately to the session-level
        default.  ``projection_metric`` doesn't drive hook rebuild on its
        own (projection materialization happens in ``steering()`` before
        ``__enter__``); it's recorded here for symmetry with the other
        two so :meth:`_resolve_steering_override` can answer "what
        metric does the active scope want?" uniformly.

        If ``_rebuild_steering_hooks`` raises (e.g. an unknown vector
        name hits ``VectorNotRegisteredError``) the just-pushed entry is
        rolled back before the exception propagates, so the stack is
        never left with stale half-committed state.

        Thread-safety: acquires :attr:`_gen_lock` (blocking) for the
        rebuild phase.  In-flight generations hold the lock for their
        whole forward+sample loop, so concurrent ``session.steering()
        .__enter__()`` from a different thread waits until the active
        generation finishes rather than mutating hook tensors mid-step.
        Single-threaded users (TUI, CLI) pay an uncontended acquire;
        the server's per-session asyncio lock serializes requests
        upstream so the contention path is exercised mostly by
        ad-hoc multi-threaded callers.

        Phase guard: rejects calls from the gen worker thread when
        :attr:`_gen_phase` is ``RUNNING`` or ``FINALIZING`` (i.e. an
        ``on_token`` callback re-entered the API mid-step).  RLock would
        otherwise let the same-thread caller pass straight through —
        the lock alone protects against cross-thread races, not
        callback reentry.  Legitimate same-thread callers reach
        ``_push_steering`` only during ``PREAMBLE`` (the internal
        steering scope that ``_generate_core`` opens before flipping
        to ``RUNNING``) or ``IDLE`` (regular user code between gens).
        """
        # ``_generate_core``'s internal ``steering_cm.__enter__()`` runs
        # during ``PREAMBLE`` (before the ``RUNNING`` flip), so push
        # never needs the ``_internal_steering_pop`` bypass — only the
        # exit path fires under ``RUNNING``/``FINALIZING``.  The check
        # here catches genuine callback reentry where ``on_token`` /
        # ``score_callback`` calls back into ``session.steering(...)``
        # mid-step.
        if self._gen_phase in (GenState.RUNNING, GenState.FINALIZING):
            raise ConcurrentGenerationError(
                "cannot enter session.steering() from inside a generation "
                "callback (e.g. on_token) — the steering stack mutation "
                f"would clobber hook tensors mid-step (gen_phase="
                f"{self._gen_phase.name})"
            )
        with self._gen_lock:
            self._steering_stack.append(dict(entries))
            self._steering_override_stack.append(
                (injection_mode, theta_max, projection_metric),
            )
            try:
                self._rebuild_steering_hooks()
            except BaseException:
                self._steering_stack.pop()
                self._steering_override_stack.pop()
                raise
            # Steering hooks just changed; the prefix cache (built
            # under the previous regime) no longer represents the
            # current pre-attention residual stream.  Drop it.
            self._invalidate_prefix_cache()
        self._emit_steering_applied()

    def _pop_steering(self) -> None:
        """Pop the top of the steering stack and rebuild hooks.

        Mirrors :meth:`_push_steering` for thread-safety and phase
        guarding: acquires :attr:`_gen_lock` so the rebuild can't fire
        mid-step in another thread's generation, and rejects same-
        thread callback reentry during ``RUNNING`` / ``FINALIZING``.
        """
        if not self._steering_stack:
            return
        # Same internal-vs-callback distinction as ``_push_steering``.
        # ``_generate_core``'s finally block sets the flag around the
        # ``steering_cm.__exit__()`` it owns, so its own scope cleanup
        # passes through; callback reentry (on_token, score_callback)
        # from inside the running loop hits the reject path.
        if (
            not self._internal_steering_pop
            and self._gen_phase in (GenState.RUNNING, GenState.FINALIZING)
        ):
            raise ConcurrentGenerationError(
                "cannot exit session.steering() from inside a generation "
                "callback — the steering stack mutation would clobber "
                f"hook tensors mid-step (gen_phase={self._gen_phase.name})"
            )
        with self._gen_lock:
            self._steering_stack.pop()
            if self._steering_override_stack:
                self._steering_override_stack.pop()
            self._rebuild_steering_hooks()
            self._invalidate_prefix_cache()
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

    def _resolve_steering_override(
        self,
    ) -> tuple[str, float]:
        """Effective ``(injection_mode, theta_max)`` for the active scope.

        Walks the override LIFO from the top, picking the first non-None
        value for each field; falls back to the session-level default
        when no scope set it.  Symmetric across the two fields so a
        scope can override mode without setting θ_max and vice versa.
        """
        eff_mode: str | None = None
        eff_theta: float | None = None
        for mode, theta, _pm in reversed(self._steering_override_stack):
            if eff_mode is None and mode is not None:
                eff_mode = mode
            if eff_theta is None and theta is not None:
                eff_theta = theta
            if eff_mode is not None and eff_theta is not None:
                break
        return (
            eff_mode if eff_mode is not None else self._injection_mode,
            eff_theta if eff_theta is not None else self._theta_max,
        )

    def _steering_needs_probe_gating(self) -> bool:
        """Return True iff any active steering trigger carries a
        :class:`~saklas.core.triggers.ProbeGate`.

        Walks the flattened steering stack head — entry tuples'
        triggers and ablation entries' triggers both inspected.
        Cheap pre-flight check that lets ``_generate_core`` decide
        whether to wire the per-step score callback at all.
        """
        flat = self._flatten_steering_stack()
        for entry in flat.values():
            if isinstance(entry, AblationTerm):
                if entry.trigger.gate is not None:
                    return True
                continue
            # entry is (alpha, Trigger) — the additive / projection shape
            _alpha, trig = entry
            if trig.gate is not None:
                return True
        return False

    def _build_gating_score_callback(self):
        """Return a closure that scores latest captures into a
        ``dict[str, float]`` for ``generate_steered``'s ``score_callback``.

        The closure pulls ``self._capture.latest_per_layer()`` (the
        most-recent ``[D]`` slice per layer the steering hooks
        captured) and runs it through :meth:`TraitMonitor.score_single_token`.
        Returns an empty dict when the capture is empty (e.g. before
        the first forward) so probe gates report inactive instead of
        seeing stale values from a previous gen.

        Caller-side guard: only invoked when
        :meth:`_steering_needs_probe_gating` is True, so the no-gate
        path stays at zero overhead.
        """
        capture = self._capture
        monitor = self._monitor

        def _score() -> dict[str, float]:
            latest = capture.latest_per_layer()
            if not latest:
                return {}
            return monitor.score_single_token(latest)

        return _score

    def _resolve_projection_metric(
        self, override: str | None = None,
    ) -> str:
        """Effective projection metric for the about-to-materialize scope.

        Walks the override LIFO top-down for the first non-None
        ``projection_metric`` entry; ``override`` (the about-to-push
        scope's value, not yet on the stack) takes priority over the
        stack so a per-call ``Steering.projection_metric`` wins over
        any outer scope.  Falls back to the session-level default
        (``self._projection_metric``) when nothing is set.

        Used by :meth:`_materialize_projections` — by the time
        ``__enter__`` pushes the new scope onto
        ``_steering_override_stack``, projection materialization has
        already run and committed derived profiles to ``self._profiles``.
        Threading the override here keeps the v2.1 default end-to-end
        correct without re-running materialization on every scope flip.
        """
        if override is not None:
            return override
        for _mode, _theta, pm in reversed(self._steering_override_stack):
            if pm is not None:
                return pm
        return self._projection_metric

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
        eff_mode, eff_theta = self._resolve_steering_override()
        self._steering.apply_to_model(
            self._layers, self._device, self._dtype,
            injection_mode=eff_mode,  # type: ignore[arg-type]
            theta_max=eff_theta,
        )

    def _apply_steering(
        self, entries: dict[str, SteeringStackEntry],
    ) -> None:
        """Compose and attach steering hooks for a generation call.

        Must be called inside an active gen span — i.e. ``_gen_phase`` is
        ``PREAMBLE`` or ``RUNNING``.  The check is defense in depth against
        a rogue caller re-entering outside a gen span.  Dispatches by entry
        type — mirrors :meth:`_rebuild_steering_hooks`.
        """
        if self._gen_phase not in (GenState.PREAMBLE, GenState.RUNNING):
            raise ConcurrentGenerationError(
                "_apply_steering called outside a generation span"
            )
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
        # New probe → _begin_capture would attach to a different
        # layer set than was live when the prefix was prefilled.
        # Drop the cache so the next gen reprefills with the fresh
        # capture-attach layout in place. (Probes don't mutate hidden
        # states, but the safer default keeps the contract simple.)
        self._invalidate_prefix_cache()

    def unprobe(self, name: str) -> None:
        self._monitor.remove_probe(name)
        self._invalidate_prefix_cache()

    # -- History --

    def rewind(self) -> None:
        if self._history and self._history[-1]["role"] == "assistant":
            self._history.pop()
        if self._history and self._history[-1]["role"] == "user":
            self._history.pop()

    def clear_history(self) -> None:
        self._history.clear()
        self._monitor.reset_history()

    # -- Prefix KV cache --

    def cache_prefix(
        self,
        messages: "list[dict[str, str]] | torch.Tensor | None",
    ) -> int:
        """Pre-prefill an identical chat prefix so subsequent ``generate()``
        calls forward only the suffix.

        Useful for batch workloads that re-issue the same chat-template
        head (system + leading user-instruction) hundreds of times — the
        v3 emotional run motivating this method does 800 stateless
        generations with the same kaomoji instruction prefix tokens.
        Per-call savings scale with prefix_len / total_input_len.

        Accepts:
        - ``list[dict[str, str]]``: encoded via ``build_chat_input`` with
          ``add_generation_prompt=False`` (the prefix should be the
          turn(s) that PRECEDE the assistant turn — generation-prompt
          tokens for the assistant turn are part of the per-call suffix).
        - ``torch.Tensor`` of shape ``[seq_len]`` or ``[1, seq_len]``:
          stored verbatim. Use this when the natural common prefix sits
          mid-content and isn't a clean message-list boundary (e.g. a
          fixed instruction concatenated into the user message before
          the variable prompt body).
        - ``None``: clear the cache. Equivalent to ``cache_prefix()``.

        Returns the cached prefix length in tokens (0 when clearing).

        Caveats (DOCUMENTED INLINE because they're easy to step on):
        1. Only safe to call OUTSIDE any active ``session.steering()``
           scope.  The prefix is prefilled with whatever steering hooks
           are live at call time — if those differ from the steering
           regime active at consume time, the cached hidden states are
           stale.  We invalidate the cache automatically on push/pop,
           but the call itself errors if a scope is open.
        2. Hidden-state capture is suspended for the duration of the
           prefill so the cached prefix doesn't pollute later score
           buckets.  No guarantees about what the monitor sees mid-call;
           callers shouldn't rely on probe state during cache_prefix.
        3. The cached tokens MUST appear as a byte-prefix of every
           ``generate()`` call's ``input_ids``.  If the caller's
           messages drift, the cache silently MISSES (cheap — full
           prefill on miss) but never silently MIS-HITS — the
           ``input_ids[..., :prefix_len].equal(prefix_tokens)`` check
           is exact.
        """
        # Clear path.
        if messages is None:
            self._invalidate_prefix_cache()
            return 0

        if self._steering_stack:
            raise SaklasError(
                "cache_prefix called inside an active session.steering() "
                "scope; prefill must run with the neutral baseline so the "
                "cached prefix is consume-regime independent. Cache before "
                "entering any steering scope."
            )
        if self.is_generating:
            raise ConcurrentGenerationError(
                "cache_prefix called while a generation is in flight"
            )

        # Build prefix tokens.
        if isinstance(messages, torch.Tensor):
            prefix_ids = messages
            if prefix_ids.dim() == 1:
                prefix_ids = prefix_ids.unsqueeze(0)
            prefix_ids = prefix_ids.to(device=self._device, dtype=torch.long)
        else:
            # List of message dicts.  add_generation_prompt=False so we
            # don't bake the assistant-turn opener into the prefix —
            # the per-call suffix carries that, ensuring the same
            # cached prefix can serve multi-turn variants.
            prefix_ids = build_chat_input(
                self._tokenizer, list(messages),
                self.config.system_prompt,
                thinking=False,
                add_generation_prompt=False,
            ).to(self._device)

        prefix_len = int(prefix_ids.shape[1])
        if prefix_len == 0:
            self._invalidate_prefix_cache()
            return 0

        # Replace any prior cache entry; old past_key_values goes out of
        # scope and is GC'd when no longer referenced.
        self._prefix_cache = None

        # Suspend capture for the prefill so the prefix tokens don't fill
        # the per-layer buckets the next generation's scoring code reads.
        # _begin_capture/_end_capture are idempotent re no-op no-probe
        # configurations, so the bracketing here is always safe.
        self._end_capture()

        with torch.inference_mode():
            outputs = self._model(
                input_ids=prefix_ids,
                attention_mask=torch.ones_like(prefix_ids),
                use_cache=True,
            )

        past_key_values = outputs.past_key_values
        if past_key_values is None:
            # Model doesn't expose KV cache (custom modeling that ignores
            # use_cache).  Nothing to cache; drop the prefix.
            return 0

        # Snapshot any linear-attention recurrent state so the patched
        # ``crop`` can restore it on prefix reuse.  No-op for standard
        # transformer caches (no LA layers).
        _snapshot_la_layers(past_key_values)

        # Store on CPU so we can ``.equal`` against fresh device tensors
        # without round-trip cost; the cache itself stays on device.
        prefix_ids_cpu = prefix_ids[0].detach().to("cpu")
        self._prefix_cache = (prefix_ids_cpu, past_key_values, prefix_len)
        return prefix_len

    def _invalidate_prefix_cache(self) -> None:
        """Drop the prefix KV cache.

        Called on every state change that affects what a fresh prefill
        of the cached prefix would produce: steering push/pop, steer /
        unsteer, probe install / remove, profile autoload.  Cheap — just
        a reference drop; HF's cache objects are GC'd when their refcount
        falls to zero.
        """
        self._prefix_cache = None

    def _try_prefix_cache_hit(
        self, input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, object, int] | None:
        """Return (suffix_ids, past_key_values, prefix_len) on cache hit, else None.

        Cache-hit precondition: the cached prefix tokens match
        ``input_ids[0, :prefix_len]`` byte-for-byte AND the suffix is
        non-empty (a zero-length suffix has no last-token logit to
        sample from on the first iteration; we'd need a different
        codepath to handle it — for now, fall through to no-cache).
        """
        cache = self._prefix_cache
        if cache is None:
            return None
        prefix_ids_cpu, past_key_values, prefix_len = cache
        if input_ids.shape[1] <= prefix_len:
            return None
        head = input_ids[0, :prefix_len].detach().to("cpu")
        if not torch.equal(head, prefix_ids_cpu):
            return None
        suffix_ids = input_ids[:, prefix_len:].contiguous()
        return suffix_ids, past_key_values, prefix_len

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
        if self._gen_phase is not GenState.IDLE:
            self._gen_lock.release()
            raise ConcurrentGenerationError("session generation already in flight")
        self._gen_phase = GenState.PREAMBLE

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

        # Pass _token_tap into generate_steered only when at least one of its
        # branches is live: caller-supplied on_token, logprobs collection, or
        # live trait subscribers.  When all three are inactive, _token_tap
        # would be a no-op called once per generated token, AND its presence
        # forces generate_steered to compute the unconditional fp32
        # log_softmax + entropy sync per step (gate at generation.py:571).
        # Skipping it here trims that cost from the v3 stateless prefill
        # workload (800 back-to-back gens of ~16 tokens each, no logprobs,
        # no streaming, no SSE).  Stop-sequence behavior is preserved by
        # not wiring _token_tap=None when stop_list is set — the tokenizer-
        # decode + stop-match in generation.py only runs under on_token.
        _has_trait_consumer = bool(self._trait_queues and self._monitor.probe_names)
        _need_tap = (
            on_token is not None
            or logprobs_list is not None
            or _has_trait_consumer
            or stop_list is not None
        )
        _effective_tap = _token_tap if _need_tap else None

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
            try:
                # Capture attach + monitor live + ctx.reset live INSIDE the
                # inner try so a BaseException (KeyboardInterrupt, etc.)
                # between any pair of these still hits the cleanup finally.
                # ``_end_capture`` and ``end_live`` are idempotent.
                self._begin_capture(widen=want_hidden)
                self._monitor.begin_live()
                # Reset the steering manager's TriggerContext for this gen;
                # ``generate_steered`` mutates it at lifecycle boundaries.
                self._steering.ctx.reset()
                self._gen_phase = GenState.RUNNING

                # Prefix KV cache lookup. Skipped when the caller asked
                # for return_hidden — the all-layer dump expects per-token
                # captures starting from the prefix's last token forward,
                # which the cached path can't synthesize (the prefix
                # forward ran with capture suspended).  Skipped under
                # thinking too: the thinking state machine bookkeeping
                # gets twitchy when prefill spans multiple tokens of
                # already-decided content, and the v3 motivating workload
                # never sets thinking=True. Under steering scopes the
                # cache was invalidated at scope entry so this path is
                # naturally a miss; explicit guard keeps that contract
                # legible. ``cache_position_offset`` is the cached
                # prefix length — generate_steered widens its prefill
                # attention mask to cover both prefix + suffix.
                cached_pkv = None
                cache_position_offset = 0
                effective_input_ids = input_ids
                if (
                    not want_hidden
                    and not use_thinking
                    and not self._steering_stack
                    and self._prefix_cache is not None
                ):
                    hit = self._try_prefix_cache_hit(input_ids)
                    if hit is not None:
                        suffix_ids, cached_pkv, cache_position_offset = hit
                        effective_input_ids = suffix_ids

                start = time.monotonic()
                # Probe-gate score callback (v2.1): wire only when the
                # active steering stack carries at least one probe-gated
                # trigger.  ``_steering_needs_probe_gating`` is a cheap
                # walk over the flattened head; the closure references
                # ``self._capture`` and ``self._monitor`` directly so the
                # generation thread doesn't pay attribute lookups in the
                # hot path beyond what a single method call costs.  No
                # gate ⇒ ``None`` ⇒ ``generate_steered`` skips the
                # callback entirely.
                gating_callback = (
                    self._build_gating_score_callback()
                    if self._steering_needs_probe_gating()
                    else None
                )

                # StaticCache eligibility (Phase B, v2.2).  Three gates
                # have to clear before we route through the static path:
                #
                # 1. Session-level support flag — set at __init__ via
                #    ``is_cuda_graphs_supported``; off on MPS/CPU and on
                #    architectures whose StaticCache constructor failed.
                # 2. Prefix cache miss — a hit pre-prefilled a
                #    DynamicCache, and mixing cache flavors mid-generation
                #    would corrupt the K/V layout.  Future work: build
                #    the prefix cache as StaticCache when this is
                #    active so the prefix-hit path also benefits.
                # 3. Fast-path-eligible steering — slow-path hooks
                #    (probe gates, ablation, multi-trigger) read mutable
                #    ``TriggerContext`` per fire, which CUDA-graph
                #    capture under ``compile(mode="reduce-overhead")``
                #    can't track.  ``all_fast_path()`` walks the hook
                #    map; empty manager returns True (unsteered → safe).
                use_static_cache = (
                    self._cuda_graphs_active
                    and cached_pkv is None
                    and self._steering.all_fast_path()
                )
                generated_ids = generate_steered(
                    self._model, self._tokenizer, effective_input_ids,
                    gen_config, self._gen_state, thinking=use_thinking,
                    on_token=_effective_tap,
                    seed=seed, stop=stop_list, logit_bias=logit_bias,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                    logprobs=lp_count,
                    trigger_ctx=self._steering.ctx,
                    past_key_values=cached_pkv,
                    cache_position_offset=cache_position_offset,
                    score_callback=gating_callback,
                    use_static_cache=use_static_cache,
                )
                elapsed = time.monotonic() - start

                # Crop the cache back to prefix_len so the next consumer
                # gets the same bare-prefix state. HF's DynamicCache.crop
                # truncates per-layer K/V tensors in-place to a max length;
                # the appended suffix + generated tokens are dropped.
                # When the cache was a miss (cached_pkv None) the stored
                # cache wasn't touched — nothing to crop.
                if cached_pkv is not None and self._prefix_cache is not None:
                    try:
                        cached_pkv.crop(cache_position_offset)
                    except (AttributeError, TypeError):
                        # Cache type doesn't support crop (legacy tuple
                        # format, or HF API drift). Drop rather than risk
                        # serving a stale cache to the next call.
                        self._invalidate_prefix_cache()
            finally:
                self._gen_state.stop_requested.set()
                self._end_capture()
                if steering_cm is not None:
                    # Internal scope cleanup — bypass the
                    # ``_pop_steering`` phase guard since we're past
                    # the model-forward loop and the rebuild is part
                    # of the legitimate teardown sequence, not a
                    # callback mutating the stack mid-step.  Save/
                    # restore rather than unconditional clear: a
                    # KeyboardInterrupt between the assignment and
                    # the ``try:`` could leak the flag as ``True``
                    # otherwise.  Reading ``old`` first puts the read
                    # outside the try-protected window so the worst
                    # case is "we never set True", not "we leave True
                    # set" (Codex review v2 catch).
                    old_internal = self._internal_steering_pop
                    try:
                        self._internal_steering_pop = True
                        steering_cm.__exit__(None, None, None)
                    finally:
                        self._internal_steering_pop = old_internal
                    steering_cm = None
                self._gen_phase = GenState.FINALIZING

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
            # make sure the steering scope is popped.  Same internal-cleanup
            # bypass as the inner finally — phase may be PREAMBLE, RUNNING,
            # or FINALIZING depending on where we threw, and the pop is
            # always legitimate teardown here.
            if steering_cm is not None:
                # Same save/restore pattern as the inner finally —
                # robust against signal-delivery between the
                # assignment and the ``try``.
                old_internal = self._internal_steering_pop
                try:
                    self._internal_steering_pop = True
                    steering_cm.__exit__(None, None, None)
                except Exception:
                    pass
                finally:
                    self._internal_steering_pop = old_internal
            raise
        finally:
            # Defense-in-depth: even if the inner finally never ran (e.g. a
            # BaseException between the outer try entry and ``begin_capture``),
            # any hooks that did get attached must come off.  Idempotent.
            self._end_capture()
            self._monitor.end_live()
            self._gen_phase = GenState.IDLE
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

    # -- Generation: batch + sweep --

    def generate_batch(
        self,
        prompts: list,
        *,
        steering: "str | Steering | None" = None,
        sampling: SamplingConfig | None = None,
        thinking: bool | None = None,
        stateless: bool = True,
        raw: bool = False,
        on_result: Callable[[int, GenerationResult], None] | None = None,
    ) -> list[GenerationResult]:
        """Run N prompts under the same steering, return results in order.

        Wrapper-loop over the existing single-prompt generation path:
        each prompt acquires the gen-lock, runs through ``_generate_core``,
        releases.  The session's threading lock keeps concurrent
        ``generate_batch`` calls from interleaving — they queue FIFO at
        the per-call level, same as today's ``generate``.

        Args mirror ``generate``.  ``stateless`` defaults to ``True``
        (batch generation is overwhelmingly used for sweeps and evals
        where conversational history would corrupt the comparison);
        pass ``stateless=False`` if you genuinely want each prompt to
        accumulate against the running history.

        ``on_result(idx, result)`` fires after each completion — useful
        for the server's SSE sweep endpoint, which streams per-result
        events without waiting for the full batch.

        Returns:
            ``list[GenerationResult]`` aligned with ``prompts``.
        """
        if not isinstance(prompts, list) or not prompts:
            raise ValueError("generate_batch: prompts must be a non-empty list")

        results: list[GenerationResult] = []
        for idx, prompt in enumerate(prompts):
            r = self._generate_core(
                prompt,
                steering=steering,
                sampling=sampling,
                stateless=stateless,
                raw=raw,
                thinking=thinking,
            )
            results.append(r)
            if on_result is not None:
                on_result(idx, r)
        return results

    def generate_sweep(
        self,
        prompt,
        sweep: dict[str, list[float]],
        *,
        base_steering: "str | Steering | None" = None,
        sampling: SamplingConfig | None = None,
        thinking: bool | None = None,
        stateless: bool = True,
        raw: bool = False,
        on_result: Callable[[int, GenerationResult, dict[str, float]], None] | None = None,
    ) -> list[GenerationResult]:
        """Sweep a single prompt across a Cartesian product of alpha values.

        ``sweep`` maps ``concept_name → [alpha_0, alpha_1, ...]``.  The
        function generates one result per element of the product across
        every concept's alpha list.  For a single-concept sweep
        (``{"honest": [0.0, 0.3, 0.6]}``) you get three results; for
        ``{"honest": [-0.4, 0.0, 0.4], "warm": [0.0, 0.3]}`` you get
        six (3 × 2).

        Each generation runs under the steering expression
        ``base_steering + " + ".join(f"{α} {name}")``.  ``base_steering``
        defaults to ``None`` so the swept alphas are the only steering;
        pass a string to layer a fixed-alpha context underneath.

        ``on_result(idx, result, alpha_values)`` fires per completion.
        ``alpha_values`` is the ``{concept: alpha}`` dict that produced
        this row — recorded on each result's ``applied_steering`` too,
        but exposed here so SSE consumers don't have to re-parse the
        expression.

        Returns:
            ``list[GenerationResult]`` in iteration order over the
            product.  Each result's ``applied_steering`` carries the
            canonical expression string for round-trip reproduction.
        """
        if not isinstance(sweep, dict) or not sweep:
            raise ValueError("generate_sweep: sweep dict must be non-empty")
        for name, alphas in sweep.items():
            if not isinstance(alphas, (list, tuple)) or not alphas:
                raise ValueError(
                    f"generate_sweep: sweep['{name}'] must be a non-empty "
                    f"list of alpha values"
                )

        # Cartesian product across concepts.  ``itertools.product``
        # preserves the order of ``sweep.items()`` so the per-concept
        # alpha lists in the output are predictable.
        import itertools

        concept_names = list(sweep.keys())
        alpha_lists = [list(sweep[name]) for name in concept_names]

        base_str: str | None
        if base_steering is None:
            base_str = None
        elif isinstance(base_steering, str):
            base_str = base_steering
        else:
            # Pre-built Steering — render through the canonical formatter
            # so we can compose with new alpha terms via string concat.
            base_str = str(base_steering)

        results: list[GenerationResult] = []
        for idx, combo in enumerate(itertools.product(*alpha_lists)):
            alpha_values = dict(zip(concept_names, combo))
            terms = [f"{alpha} {name}" for name, alpha in alpha_values.items()]
            expr = " + ".join(terms)
            if base_str:
                expr = f"{base_str} + {expr}"

            r = self._generate_core(
                prompt,
                steering=expr,
                sampling=sampling,
                stateless=stateless,
                raw=raw,
                thinking=thinking,
            )
            results.append(r)
            if on_result is not None:
                on_result(idx, r, alpha_values)
        return results

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
