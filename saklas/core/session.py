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
from types import TracebackType
from typing import Any, Callable, Iterator, Literal, overload

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

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
from saklas.core.loom import (
    InvalidNodeOperationError,
    LoomMutated,
    LoomTree,
    Recipe,
    MutationDuringGenerationError,
    derive_seed_schedule,
)
from saklas.core.model import load_model, get_layers, get_model_info
from saklas.core.monitor import TraitMonitor
from saklas.io.packs import PackFormatError, PackMetadata, hash_folder_files
from saklas.io.paths import concept_dir
from saklas.io.probes_bootstrap import bootstrap_probes, bootstrap_layer_means
from saklas.core.profile import Profile
from saklas.core.results import GenerationResult, RunSet, TokenEvent, ProbeReadings
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

    def _save_la_snapshot(layer: Any) -> None:
        snap: dict[str, Any] = {}
        if getattr(layer, "is_conv_states_initialized", False):
            snap["conv"] = layer.conv_states.detach().clone()
        if getattr(layer, "is_recurrent_states_initialized", False):
            snap["recurrent"] = layer.recurrent_states.detach().clone()
        layer._saklas_la_snapshot = snap if snap else None

    def _la_crop_with_restore(self: Any, max_length: int) -> None:
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

    def _hybrid_crop(self: Any, max_length: int) -> None:
        DynamicLayer.crop(self, max_length)
        _la_crop_with_restore(self, max_length)

    setattr(LinearAttentionLayer, "crop", _la_crop_with_restore)
    setattr(LinearAttentionAndFullAttentionLayer, "crop", _hybrid_crop)
    setattr(LinearAttentionLayer, "_saklas_save_snapshot", _save_la_snapshot)
    setattr(LinearAttentionLayer, "_saklas_crop_patched", True)
    return True


_install_la_cache_patch()


def _snapshot_la_layers(cache: Any) -> None:
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
PROBE_CATEGORIES = [
    "affect",
    "epistemic",
    "alignment",
    "register",
    "social_stance",
    "cultural",
    "identity",
]
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
# by type in ``_compose_steering_entries``.
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

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
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
        dtype: torch.dtype | str | None = None,
        quantize: str | None = None,
        probes: list[str] | None = None,
        system_prompt: str | None = None,
        max_tokens: int = 1024,
        injection_mode: str = "angular",
        theta_max: float | None = None,
        extraction_method: str = "dim",
        projection_metric: str = "mahalanobis",
        dls: bool = True,
        compile: bool = False,
        compile_mode: str | None = None,
        cuda_graphs: bool = False,
        return_top_k: int = 0,
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

        ``compile`` (default ``False``) opts in to ``torch.compile`` on
        CUDA — kernel fusion via inductor, typically 1.2–3× decode
        tok/s on small models when it succeeds.  Off by default: torch
        2.12's inductor has known codegen bugs on newer architectures
        (Gemma-4, Qwen3.5 hit ``TypeError: Pointer argument must be
        either uint64`` in the static_cuda_launcher / regular Triton
        launcher); the loader's probe catches these but still costs
        ~25–100s upfront, which rarely amortizes on interactive
        workloads.  Pass ``True`` for sustained workloads (long-running
        serve, batch eval) where the per-token win pays back the
        compile cost.  Auto-skipped on MPS/CPU.

        ``cuda_graphs`` (default ``False``) opts in to
        ``transformers.StaticCache`` + ``torch.compile(mode=
        "reduce-overhead")`` on CUDA-supported architectures.  Static
        K/V buffers across decode steps mean inductor can capture CUDA
        graphs internally — an additional 1.5–2.5× decode tok/s on top
        of the kernel-fusion win from ``compile=True``.  Off by default
        for the same reason as ``compile``; setting this without
        ``compile=True`` has no effect.  Auto-skipped on MPS/CPU and on
        architectures whose StaticCache construction fails.

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

        # Apply compile manually with the resolved mode.  Routes through
        # ``_compile_with_probe`` so an architecture-specific inductor /
        # Triton bug surfaces at load time as a caught warning + eager
        # fallback, rather than as a segfault on the user's first token.
        if compile and device_obj.type == "cuda":
            from saklas.core.model import _compile_with_probe
            model = _compile_with_probe(
                model, tokenizer, device_obj,
                mode=effective_compile_mode,
            )
        elif compile:
            _log.info(
                "compile=True but device=%s — skipping torch.compile "
                "(supported only on CUDA)",
                device_obj.type,
            )

        # ``__init__`` consults the same probe helper, but the helper now
        # caches by the underlying module id (survives the torch.compile
        # wrapper via ``_orig_mod``), so this second call is a dictionary
        # lookup rather than another StaticCache(max_cache_len=1)
        # allocation.
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
            return_top_k=return_top_k,
        )

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        *,
        probes: list[str] | None = None,
        system_prompt: str | None = None,
        max_tokens: int = 1024,
        injection_mode: str = "angular",
        theta_max: float | None = None,
        extraction_method: str = "dim",
        projection_metric: str = "mahalanobis",
        dls: bool = True,
        cuda_graphs: bool = False,
        return_top_k: int = 0,
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
        # Phase 1 logit pass: session-level default for SamplingConfig
        # .return_top_k.  Per-call value > 0 wins; per-call K=0 (the
        # SamplingConfig default) inherits this stored value via the
        # composition in ``_generate_core``.  Clamped on entry mirroring
        # SamplingConfig.__post_init__ so out-of-range values from
        # ``--top-k-alts`` or YAML don't reach the engine slice.
        if return_top_k < 0:
            return_top_k = 0
        elif return_top_k > 256:
            return_top_k = 256
        self._default_return_top_k: int = int(return_top_k)
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

        # Conversation state lives in a :class:`LoomTree` (v2.3).  The
        # active path through the tree is what the model sees as context;
        # ``self.history`` is a derived property over ``tree.active_path``
        # for backward-compatibility with v2.2 callers that read the flat
        # list directly.  Generation routes through ``tree.add_user_turn``
        # / ``tree.begin_assistant`` / ``tree.finalize_assistant``.  The
        # tree is in-memory only — there is no automatic cross-session
        # persistence; the TUI's ``/save`` and ``/load`` are the explicit
        # save/restore path (``LoomTree.save`` / ``LoomTree.load``).
        self.tree = LoomTree(
            events=self.events,
            model_id=getattr(self._model_info, "model_id", None),
            conflict_check=self._loom_conflict_check,
        )
        self._joint_logprob_cache: dict[tuple[str, str], Any] = {}

        def _invalidate_tree_analysis_caches(event: Any) -> None:
            if isinstance(event, LoomMutated) and event.op in {
                "edit",
                "delete",
                "reset",
                "finalize_assistant",
            }:
                self._joint_logprob_cache.clear()

        self.events.subscribe(_invalidate_tree_analysis_caches)

        # Subtree root reserved by an in-flight generation (the user-parent
        # of the streaming assistant target).  None while idle; set by
        # ``_generate_core`` before token streaming begins, cleared in the
        # outermost ``finally``.  Consulted by :meth:`_loom_conflict_check`.
        self._active_gen_reservation: str | None = None
        self._last_result: GenerationResult | None = None
        self._last_per_token_scores: dict[str, list[float]] | None = None

        # Probe content-hash cache for transcript export / replay (v2.3
        # phase 5).  Keyed by probe name → sha256 hex of the baked tensor
        # bytes (concatenated layer order).  Invalidated by
        # :meth:`add_probe` / :meth:`remove_probe`; rebuilt lazily by
        # :meth:`_probe_hash`.
        self._probe_hash_cache: dict[str, str] = {}

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
        # dependencies it needs (model handle / pack writer / event bus)
        # instead of a back-reference.  ``self`` satisfies those protocols
        # implicitly — see :mod:`saklas.core.extraction`.
        from saklas.core.extraction import ExtractionPipeline as _Pipeline
        self._extraction = _Pipeline(self, self, self.events)

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

    def _compose_steering_entries(
        self,
        entries: dict[str, SteeringStackEntry],
    ) -> None:
        """Load entries into ``SteeringManager`` without installing hooks."""
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

    def _install_composed_steering(self) -> None:
        """Attach the currently-composed steering entries to model layers."""
        eff_mode, eff_theta = self._resolve_steering_override()
        self._steering.apply_to_model(
            self._layers, self._device, self._dtype,
            injection_mode=eff_mode,  # type: ignore[arg-type]
            theta_max=eff_theta,
        )

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
        if not flat:
            self._steering.clear_all()
            return
        self._compose_steering_entries(flat)
        self._install_composed_steering()

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
        # Transcript probe-hash cache (v2.3 phase 5) is keyed by name;
        # any change to the registered profiles invalidates the relevant
        # entry (cheaper to drop the whole map than diff per-probe).
        self._probe_hash_cache.pop(name, None)

    def unprobe(self, name: str) -> None:
        self._monitor.remove_probe(name)
        self._invalidate_prefix_cache()
        self._probe_hash_cache.pop(name, None)

    def _probe_hash(self, name: str) -> str | None:
        """Return sha256 hex of the baked tensor bytes for ``name``.

        Stamps :class:`saklas.core.loom.Recipe.probe_hashes` so transcript
        replay can detect probe drift between save and load (decision
        19 in ``docs/plans/loom.md``).  Cached on the session — adding
        or removing a probe invalidates the relevant cache entry.

        Returns ``None`` when the probe isn't registered.  Hashing is
        deterministic across machines: layers iterated in sorted order,
        each tensor's CPU bytes hashed (fp32 cast to keep dtype neutral
        across mixed-precision storage).
        """
        if name in self._probe_hash_cache:
            return self._probe_hash_cache[name]
        profile = self._monitor.profiles.get(name)
        if profile is None:
            return None
        import hashlib
        h = hashlib.sha256()
        for layer_idx in sorted(profile.keys()):
            tensor = profile[layer_idx]
            # ``tensor.detach().cpu().contiguous()`` keeps the hash stable
            # across device placements; fp32 cast normalizes dtype so
            # mixed-precision storage doesn't change the hex digest.
            try:
                arr = tensor.detach().to("cpu").to(torch.float32).contiguous()
                h.update(arr.numpy().tobytes())
            except Exception:
                # Synthetic probes from unit tests may not be torch
                # tensors — fall through to a stable text representation
                # so the cache still produces something deterministic.
                h.update(repr((layer_idx, tensor)).encode("utf-8"))
        digest = h.hexdigest()
        self._probe_hash_cache[name] = digest
        return digest

    def probe_hashes(self) -> dict[str, str]:
        """Return ``{probe_name: sha256_hex}`` for every registered probe."""
        out: dict[str, str] = {}
        for name in self._monitor.probe_names:
            d = self._probe_hash(name)
            if d is not None:
                out[name] = d
        return out

    # -- Cross-branch diff (v2.3 phase 5) --

    def diff_nodes(self, a_id: str, b_id: str) -> Any:
        """Return a :class:`saklas.core.loom_diff.NodeDiff` between two nodes.

        Both nodes are looked up in :attr:`tree`; the diff bundles the
        word-level text diff and the readings delta table.  ``parent_id``
        on the returned diff is the shared user-parent when both nodes
        share one (the common sibling-comparison case); ``None``
        otherwise.
        """
        from saklas.core.loom_diff import NodeDiff, readings_diff, text_diff

        a = self.tree.get(a_id)
        b = self.tree.get(b_id)
        parent_id = a.parent_id if a.parent_id == b.parent_id else None
        return NodeDiff(
            a_id=a_id,
            b_id=b_id,
            parent_id=parent_id,
            text=text_diff(a.text or "", b.text or ""),
            readings=readings_diff(
                a.aggregate_readings or {},
                b.aggregate_readings or {},
            ),
        )

    # -- Recipe-override regen (v2.3 phase 5) --

    def regen_with_modifier(
        self,
        parent_node_id: str,
        mode: "str | Recipe",
        *,
        base_recipe: "Recipe | None" = None,
        n: int = 1,
    ) -> RunSet:
        """Regenerate as a sibling of ``parent_node_id`` under a modifier.

        ``mode`` is either a built-in mode string (``"unsteered"``,
        ``"inverted"``, ``"reseed"``, ``"cool"``, ``"hot"``) or a partial
        :class:`Recipe` carrying the override fields (``"custom"`` mode —
        callers parse their own partial-recipe expressions and hand the
        resulting Recipe in directly; :meth:`Recipe.compose_modifier`
        passes Recipe instances through unchanged).  The override
        composes onto the parent node's recipe (or ``base_recipe`` if
        given): None fields fall through to the parent's setting.

        Convenience entry point: auto-regen and the manual
        ``/regen N <mode>`` flow both call this.  Returns a
        :class:`RunSet` even when ``n == 1``.
        """
        from saklas.core.loom import Recipe

        parent = self.tree.get(parent_node_id)
        # Walk up to find the assistant whose recipe we're overlaying —
        # if the caller passed a user node, use its existing assistant
        # child's recipe (regen replaces that assistant).
        anchor: "Recipe | None" = None
        if base_recipe is not None:
            anchor = base_recipe
        elif parent.role == "assistant" and parent.recipe is not None:
            anchor = parent.recipe
        else:
            # Pick the most recent assistant ancestor's recipe.
            for nid in self.tree.ancestors_of(parent_node_id):
                anc = self.tree.nodes.get(nid)
                if anc is not None and anc.role == "assistant" and anc.recipe is not None:
                    anchor = anc.recipe
                    break
        if anchor is None:
            anchor = Recipe()

        # compose_modifier handles both str ("unsteered"/"inverted"/...)
        # and Recipe (custom) — the dispatch lives on the dataclass.
        override = anchor.compose_modifier(mode)

        overlaid = anchor.overlay(override)

        # Resolve which node to anchor the regen under.  If the caller
        # passed an assistant node, regen siblings under its user-parent;
        # if a user node, siblings under it directly.
        if parent.role == "assistant":
            anchor_user_id = parent.parent_id
        else:
            anchor_user_id = parent_node_id

        # Reuse the existing user-turn text for sibling spawning.  When
        # the anchor is a user turn we feed the user-turn text to
        # ``generate`` and let ``add_user_turn``'s dedup land it on the
        # same parent.
        anchor_node = self.tree.nodes.get(anchor_user_id) if anchor_user_id else None
        if anchor_node is None or anchor_node.role != "user":
            raise InvalidNodeOperationError(
                f"regen_with_modifier: cannot anchor sibling under "
                f"{parent_node_id!r} — expected user/assistant pair, "
                f"got {parent.role}"
            )
        user_text = anchor_node.text

        sampling = overlaid.sampling
        return self.generate(
            user_text,
            steering=overlaid.steering,
            sampling=sampling,
            thinking=overlaid.thinking,
            parent_node_id=anchor_node.parent_id,
            n=n,
        )

    def fork_from_token(
        self,
        node_id: str,
        raw_index: int,
        alt_token_id: int,
        *,
        on_token: Callable[..., None] | None = None,
    ) -> GenerationResult:
        """Logit fork — regenerate ``node_id`` as a sibling with one token swapped.

        Replays the assistant node's raw decode sequence up to
        ``raw_index``, forces ``alt_token_id`` at that position, then
        samples the continuation freely.  The node's recipe (steering /
        sampling / seed / thinking) is reused verbatim, so the only thing
        that changed is the single token — a clean counterfactual.  The
        original seed is re-applied and the sampler still draws every
        forced step, so the RNG stream is bit-identical through the fork
        point; the divergence downstream is purely the model reacting to
        the swapped token.

        Lands as a sibling assistant node under the same user turn (the
        dedup path :meth:`regen_with_modifier` uses).  Raises
        :class:`InvalidNodeOperationError` when the node isn't a forkable
        assistant — wrong role, no ``raw_token_ids`` (legacy or
        transcript-loaded node), or ``raw_index`` out of range.
        """
        from dataclasses import replace as _replace

        node = self.tree.get(node_id)
        if node.role != "assistant":
            raise InvalidNodeOperationError(
                f"fork_from_token: {node_id!r} is a {node.role} node, "
                f"not a forkable assistant"
            )
        raw = node.raw_token_ids
        if not raw:
            raise InvalidNodeOperationError(
                f"fork_from_token: {node_id!r} has no raw token record "
                f"(legacy or transcript-loaded node — not forkable)"
            )
        if not 0 <= raw_index < len(raw):
            raise InvalidNodeOperationError(
                f"fork_from_token: raw_index {raw_index} out of range "
                f"[0, {len(raw)}) for {node_id!r}"
            )
        forced_prefix = [int(t) for t in raw[:raw_index]] + [int(alt_token_id)]

        user_node = (
            self.tree.nodes.get(node.parent_id) if node.parent_id else None
        )
        if user_node is None or user_node.role != "user":
            raise InvalidNodeOperationError(
                f"fork_from_token: {node_id!r} has no user parent to "
                f"anchor the forked sibling under"
            )

        recipe = node.recipe
        base_sampling = (
            recipe.sampling
            if (recipe is not None and recipe.sampling is not None)
            else SamplingConfig()
        )
        # The forced prefix burns ``len(forced_prefix)`` decode steps
        # before the continuation starts — extend the token budget by
        # that much so a deep fork keeps the original continuation
        # headroom rather than stopping just past the fork point.
        base_max = base_sampling.max_tokens or self.config.max_new_tokens
        fork_sampling = _replace(
            base_sampling, max_tokens=len(forced_prefix) + base_max,
        )

        return self._generate_core(
            user_node.text,
            steering=recipe.steering if recipe is not None else None,
            sampling=fork_sampling,
            thinking=recipe.thinking if recipe is not None else None,
            on_token=on_token,
            parent_node_id=user_node.parent_id,
            forced_prefix=forced_prefix,
        )

    def prefill_assistant(
        self,
        node_id: str,
        text: str,
        *,
        steering: "str | Steering | None" = None,
        sampling: SamplingConfig | None = None,
        on_token: Callable[..., None] | None = None,
    ) -> GenerationResult:
        """Answer-prefill — seed the assistant reply under a user node.

        Generates an assistant child of the user node ``node_id`` whose
        response *begins* with ``text``, then samples the continuation
        freely.  ``text`` is tokenized and handed to the engine as a
        ``forced_prefix`` — the same logit primitive :meth:`fork_from_token`
        rides — so the first ``len(forced_prefix)`` decode steps emit
        exactly those ids before free sampling takes over.  The result
        lands as a sibling assistant node under ``node_id`` (the dedup
        path :meth:`fork_from_token` uses: anchor at the user node's
        parent so ``add_user_turn`` re-uses the existing user turn).

        Prefill is an *answer* prefill: ``text`` is the start of the
        response, so the thinking channel is skipped entirely
        (``thinking=False``).  Unlike a fork there's no source recipe to
        inherit — ``steering`` / ``sampling`` are honored exactly as a
        normal :meth:`generate` call.  The token budget is bumped by
        ``len(forced_prefix)`` so the continuation keeps full headroom
        rather than stopping just past the seeded span.

        Raises :class:`InvalidNodeOperationError` when ``node_id`` isn't a
        user node, or when ``text`` tokenizes to an empty sequence.
        """
        from dataclasses import replace as _replace

        node = self.tree.get(node_id)
        if node.role != "user":
            raise InvalidNodeOperationError(
                f"prefill_assistant: {node_id!r} is a {node.role} node, "
                f"not a user node — prefill seeds the assistant reply "
                f"that hangs off a user turn"
            )
        forced_prefix = list(
            self._tokenizer.encode(text, add_special_tokens=False)
        )
        if not forced_prefix:
            raise InvalidNodeOperationError(
                f"prefill_assistant: {text!r} tokenized to an empty "
                f"sequence — nothing to prefill"
            )

        base_sampling = sampling if sampling is not None else SamplingConfig()
        base_max = base_sampling.max_tokens or self.config.max_new_tokens
        prefill_sampling = _replace(
            base_sampling, max_tokens=len(forced_prefix) + base_max,
        )

        return self._generate_core(
            node.text,
            steering=steering,
            sampling=prefill_sampling,
            thinking=False,
            on_token=on_token,
            parent_node_id=node.parent_id,
            forced_prefix=forced_prefix,
        )

    # -- Authored turns (no-generation commits) --

    def append_user_turn(
        self,
        parent_node_id: str | None,
        text: str,
    ) -> str:
        """Land a user turn under ``parent_node_id`` without generating.

        The Ctrl+Enter "commit" on the chat surfaces: the typed text
        lands as a user-role child and the active node advances, but no
        decode runs.  The follow-up move is usually to type a prefill
        and hit Enter (which then routes through
        :meth:`prefill_assistant`) — together these let a user assemble
        a turn pair manually.

        Refuses anchoring under a user-role parent
        (:meth:`_check_user_send_target`) — that would corrupt the
        v2 chat-message flatten by producing user-under-user.  Dedup is
        on: a same-text user sibling under ``parent_node_id`` is
        returned without growing the tree, matching the regen workflow.
        Returns the new (or deduped) user node id.

        ``text`` must be non-empty (whitespace-only is honored, but a
        completely empty string raises) — empty commits should be
        no-op'd at the surface, not sent over the wire.
        """
        if text == "":
            raise InvalidNodeOperationError(
                "append_user_turn: text must be non-empty"
            )
        self._check_user_send_target(parent_node_id)
        return self.tree.add_user_turn(text, parent_id=parent_node_id)

    def append_assistant_turn(
        self,
        user_node_id: str,
        text: str,
    ) -> str:
        """Land a user-authored assistant turn under ``user_node_id``.

        The Ctrl+Enter "commit" on a user node: the typed text becomes
        the whole assistant turn — no sampling, no steering, no
        thinking.  The result is a sibling assistant under
        ``user_node_id`` whose ``text`` is exactly the typed text, with
        ``raw_token_ids`` populated by ``tokenizer.encode(text,
        add_special_tokens=False)`` so the node remains forkable.
        ``recipe`` stays ``None`` — that's the implicit "no model run
        produced this" marker, the same shape transcript-loaded nodes
        already carry.

        Raises :class:`InvalidNodeOperationError` when ``user_node_id``
        isn't a user node, when ``text`` is empty, or when it tokenizes
        to an empty sequence.  Returns the new assistant node id; the
        loom's active node advances to it.
        """
        if text == "":
            raise InvalidNodeOperationError(
                "append_assistant_turn: text must be non-empty"
            )
        node = self.tree.get(user_node_id)
        if node.role != "user":
            raise InvalidNodeOperationError(
                f"append_assistant_turn: {user_node_id!r} is a "
                f"{node.role} node, not a user node — an authored "
                f"assistant turn hangs off a user turn"
            )
        raw_token_ids = list(
            self._tokenizer.encode(text, add_special_tokens=False)
        )
        if not raw_token_ids:
            raise InvalidNodeOperationError(
                f"append_assistant_turn: {text!r} tokenized to an "
                f"empty sequence — nothing to commit"
            )

        new_id = self.tree.begin_assistant(user_node_id, recipe=None)
        # Drop the empty token blobs ``begin_assistant`` seeded — an
        # authored turn has no per-token scores; ``None`` matches the
        # transcript-loaded shape so renderers/saves treat it the same.
        try:
            authored = self.tree.nodes[new_id]
            authored.tokens = None
            authored.thinking_tokens = None
        except KeyError:  # pragma: no cover — begin_assistant just added it
            pass
        self.tree.finalize_assistant(
            new_id,
            text=text,
            applied_steering=None,
            finish_reason="stop",
            raw_token_ids=raw_token_ids,
        )
        return new_id

    # -- History / loom tree --

    def _check_user_send_target(self, parent_node_id: str | None) -> None:
        """D15 — refuse sending a new user turn from a user-role node.

        The plan's send-semantics table (``docs/plans/loom.md`` §"Active-
        node send semantics") rejects sending a fresh user turn when the
        resolved parent is itself a user node: the user node is already
        waiting for an assistant.  Allowing it would corrupt the tree
        shape (user-under-user) and break the v2 chat-message flatten.

        Resolved parent = ``parent_node_id`` when passed, else the active
        node.  Internal regen paths pass ``parent_node_id=<grandparent>``
        explicitly so add_user_turn's dedup re-uses the existing user
        sibling rather than triggering this reject.

        Raises :class:`InvalidNodeOperationError` (HTTP 400) on violation.
        """
        target_parent_id = (
            parent_node_id if parent_node_id is not None
            else self.tree.active_node_id
        )
        parent_node = (
            self.tree.nodes.get(target_parent_id)
            if target_parent_id is not None else None
        )
        if parent_node is not None and parent_node.role == "user":
            raise InvalidNodeOperationError(
                f"cannot send a new user turn from a user node "
                f"({target_parent_id}): the active turn is already "
                f"waiting for an assistant.  Use /regen to redo the "
                f"assistant, or navigate away first."
            )

    def _loom_conflict_check(self, node_id: str, op: str) -> None:
        """Tree-mutation conflict checker — see :mod:`saklas.core.loom`.

        The :class:`LoomTree` calls this at the entry of every mutator;
        we raise :class:`MutationDuringGenerationError` (HTTP 409) when
        the requested op conflicts with an in-flight generation's
        subtree reservation.

        Rules (per ``docs/plans/loom.md`` phase 1):

        - Decoration ops (``star``, ``note``) and ``branch`` never raise.
        - ``add_user_turn`` / ``begin_assistant`` / ``finalize_assistant``
          run from inside the gen path itself; never raise here.
        - ``edit``, ``delete_subtree``, and ``reset`` raise when the
          target intersects ``self._active_gen_reservation``.

        ``self._active_gen_reservation`` is the user-parent of the
        streaming assistant node; the subtree rooted at that node is
        reserved.
        """
        reservation = self._active_gen_reservation
        if reservation is None:
            return  # idle — every op is free
        if op in ("add_user_turn", "begin_assistant", "finalize_assistant",
                  "branch", "star", "note", "navigate"):
            return
        # ``op`` is "edit", "delete_subtree", "reset", or some future
        # mutator: refuse when ``node_id`` is the reservation root, a
        # descendant of it, or — for "reset" — anything at all.
        if op == "reset":
            raise MutationDuringGenerationError(
                "cannot reset tree while a generation is in flight"
            )
        if node_id == reservation or self.tree.is_ancestor_of(reservation, node_id) \
                or self.tree.is_ancestor_of(node_id, reservation):
            raise MutationDuringGenerationError(
                f"cannot {op} on a node inside an in-flight generation's "
                f"reservation (reservation root: {reservation})"
            )

    @property
    def history(self) -> list[dict[str, str]]:
        """Compat shim — chat messages along the active path.

        Replaces v2.2's ``self._history`` flat list with a derived view
        over :attr:`tree.active_path`.  Callers that mutated ``_history``
        directly need to migrate; readers (`session.history`) work
        unchanged.
        """
        return self.tree.messages_for()

    def rewind(self) -> None:
        """Walk the active node back one user→assistant pair.

        Non-destructive under v2.3 loom: the rewound pair stays in the
        tree as a dead branch, navigable back via the sidebar / loom
        screen.  ``clear_history`` is the destructive verb.
        """
        self.tree.rewind()
        # Monitor history is kept aligned with the active path's
        # finalize stream; rewinding the path means dropping the trailing
        # readings so live trait scoring continues from a coherent state.
        # See ``clear_history`` for the wipe-all variant.
        self._monitor.reset_history()

    def clear_history(self) -> None:
        """Reset the tree to a fresh root.

        Destructive — drops every branch.  Matches v2.2 user expectation
        of ``/clear`` meaning wipe.  Use :meth:`rewind` for the
        non-destructive step-back.
        """
        self.tree.reset()
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
        parent_node_id: str | None = None,
    ) -> torch.Tensor:
        if isinstance(input, str):
            if stateless:
                prior: list[dict[str, str]] = []
            else:
                # Walk the path to ``parent_node_id`` (or the active node).
                # Loom: the model sees the conversation along whatever path
                # the user is currently focused on, not a single flat log.
                prior = self.tree.messages_for(parent_node_id)
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
        logprobs_list: list[tuple[int, float, list[Any]]] | None = None,
        applied_steering: str | None = None,
        *,
        return_hidden: bool = False,
        assistant_node_id: str | None = None,
        mean_logprob: float | None = None,
        mean_surprise: float | None = None,
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

        # Finalize the in-flight assistant node in the tree (loom v2.3).
        # The user node was added by the gen preamble; the assistant node
        # was created via ``begin_assistant`` and accumulated tokens along
        # the way.  Stateless gens skip the entire tree mutation path —
        # ``assistant_node_id`` is None on that branch.
        # ``mean_logprob`` / ``mean_surprise`` come pre-computed from
        # ``_generate_core`` (the only function with scope on the
        # ``_token_tap`` accumulator); both are ``None`` when no logprob
        # capture was live, so legacy paths land cleanly.
        if not stateless and assistant_node_id is not None:
            self._stamp_raw_indices(assistant_node_id)
            self.tree.finalize_assistant(
                assistant_node_id,
                text=text,
                aggregate_readings={n: r.mean for n, r in readings.items()},
                applied_steering=applied_steering,
                finish_reason=self._gen_state.finish_reason,
                mean_logprob=mean_logprob,
                mean_surprise=mean_surprise,
                raw_token_ids=generated_ids,
            )

        return result

    def _stamp_raw_indices(self, node_id: str) -> None:
        """Stamp each emitted token row with its ``generated_ids`` index.

        ``state.emit_map`` records ``(raw_index, is_thinking)`` per emitted
        token in emission order; the node's ``tokens`` / ``thinking_tokens``
        rows are in that same order, split by the thinking flag.  The
        stamped ``raw_index`` is the join key a logit fork uses to slice
        ``raw_token_ids`` at the clicked token — emitted-token coordinates
        don't line up with raw decode steps once delimiters (suppressed)
        and partial-UTF-8 runs (merged) enter the picture.
        """
        node = self.tree.nodes.get(node_id)
        if node is None:
            return
        resp_rows = node.tokens or []
        think_rows = node.thinking_tokens or []
        r_i = t_i = 0
        for raw_index, is_thinking in self._gen_state.emit_map:
            if is_thinking:
                if t_i < len(think_rows):
                    think_rows[t_i]["raw_index"] = raw_index
                    t_i += 1
            elif r_i < len(resp_rows):
                resp_rows[r_i]["raw_index"] = raw_index
                r_i += 1

    def _generation_preamble(self, input, raw, thinking, stateless=False,
                             parent_node_id: str | None = None):
        """Shared input prep + gen-state reset.

        Steering is NOT installed here — the caller is expected to hold a
        ``session.steering()`` scope open across the generation.
        ``parent_node_id`` selects which loom-tree path the input is
        anchored against (default: the active path).
        """
        use_thinking = thinking and supports_thinking(self._tokenizer)
        input_ids = self._prepare_input(
            input, raw=raw, thinking=use_thinking, stateless=stateless,
            parent_node_id=parent_node_id,
        )
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

    def _resolve_recipe_override(
        self,
        recipe_override: "Recipe | str | None",
        *,
        parent_node_id: str | None,
        steering: "str | Steering | None",
        sampling: SamplingConfig | None,
        thinking: bool | None,
    ) -> tuple["str | Steering | None", SamplingConfig | None, bool | None]:
        """Apply a recipe override to the per-call kwargs.

        ``recipe_override`` is either a :class:`Recipe` partial (None
        fields fall through) or a built-in mode string forwarded to
        :meth:`Recipe.compose_modifier`.  The override composes onto the
        parent node's recipe (or an empty Recipe when no parent has
        one) and the resulting fields *replace* the explicit kwargs
        when set — explicit kwargs only win where the override is None.
        Returns ``(steering, sampling, thinking)`` tuple ready to feed
        the gen path.
        """
        if recipe_override is None:
            return steering, sampling, thinking

        # Resolve the anchor recipe — the parent assistant's recipe
        # when present, else an empty Recipe.  Walk ancestors so a
        # user-anchored regen still finds a recipe to overlay onto.
        anchor: Recipe | None = None
        if parent_node_id is not None:
            parent = self.tree.nodes.get(parent_node_id)
            if parent is not None:
                if parent.role == "assistant" and parent.recipe is not None:
                    anchor = parent.recipe
                else:
                    for nid in self.tree.ancestors_of(parent_node_id):
                        anc = self.tree.nodes.get(nid)
                        if anc is not None and anc.role == "assistant" and anc.recipe is not None:
                            anchor = anc.recipe
                            break
        if anchor is None:
            anchor = Recipe()

        # compose_modifier handles both str modes and Recipe (custom)
        # — Recipe instances pass through unchanged on that path.
        override = anchor.compose_modifier(recipe_override)
        overlaid = anchor.overlay(override)

        # Override fields *win* over the caller's explicit kwargs when
        # set.  The auto-regen UI expects "configure once via the
        # override, ignore the caller's per-turn defaults" semantics.
        new_steering = overlaid.steering if overlaid.steering is not None else steering
        new_sampling = overlaid.sampling if overlaid.sampling is not None else sampling
        new_thinking = overlaid.thinking if overlaid.thinking is not None else thinking
        # Seed lives on the SamplingConfig — fold into new_sampling
        # when the override specifies one and the caller's sampling
        # doesn't already pin a seed.
        if overlaid.seed is not None:
            from dataclasses import replace as _replace
            base = new_sampling if isinstance(new_sampling, SamplingConfig) else SamplingConfig()
            new_sampling = _replace(base, seed=overlaid.seed)
        return new_steering, new_sampling, new_thinking

    def _prepare_generation_call(
        self,
        steering: "str | Steering | None",
        sampling: SamplingConfig | None,
        thinking: bool | None,
    ) -> tuple[
        Steering | None,
        bool,
        GenerationConfig,
        int | None,
        int | None,
        list[str] | None,
        dict[int, float] | None,
        float,
        float,
        list | None,
    ]:
        """Normalize per-call generation controls before model work."""
        steering_obj = Steering.from_value(steering)
        if thinking is None:
            if steering_obj is not None and steering_obj.thinking is not None:
                use_thinking_req = steering_obj.thinking
            else:
                use_thinking_req = supports_thinking(self._tokenizer)
        else:
            use_thinking_req = thinking

        gen_config = self._compose_gen_config(sampling)
        raw_lp = sampling.logprobs if sampling is not None else None
        raw_top_k = sampling.return_top_k if sampling is not None else 0
        if raw_top_k == 0:
            raw_top_k = self._default_return_top_k
        if raw_top_k > 0:
            lp_count: int | None = (
                max(raw_top_k, raw_lp) if raw_lp is not None else raw_top_k
            )
        else:
            lp_count = raw_lp

        seed = sampling.seed if sampling is not None else None
        stop_tuple = sampling.stop if sampling is not None else None
        stop_list = list(stop_tuple) if stop_tuple else None
        logit_bias = sampling.logit_bias if sampling is not None else None
        presence_penalty = sampling.presence_penalty if sampling is not None else 0.0
        frequency_penalty = (
            sampling.frequency_penalty if sampling is not None else 0.0
        )
        logprobs_list: list | None = [] if raw_lp is not None else None
        return (
            steering_obj,
            use_thinking_req,
            gen_config,
            lp_count,
            seed,
            stop_list,
            logit_bias,
            presence_penalty,
            frequency_penalty,
            logprobs_list,
        )

    def _snapshot_steering_alphas(self) -> dict[str, float]:
        """Flatten the active steering stack for ``GenerationResult.vectors``."""
        snap: dict[str, float] = {}
        for name, entry in self._flatten_steering_stack().items():
            if isinstance(entry, AblationTerm):
                snap[name] = entry.coeff
                continue
            snap[name] = entry[0]
        return snap

    def _start_loom_assistant(
        self,
        input,
        *,
        stateless: bool,
        parent_node_id: str | None,
        sampling: SamplingConfig | None,
        steering_obj: Steering | None,
        use_thinking_req: bool,
    ) -> str | None:
        """Create the loom user/assistant nodes for a stateful generation."""
        if stateless:
            return None

        if isinstance(input, str):
            self._check_user_send_target(parent_node_id)
            user_node_id = self.tree.add_user_turn(input, parent_id=parent_node_id)
        else:
            user_node_id = parent_node_id or self.tree.active_node_id

        self._active_gen_reservation = user_node_id
        seed_val = sampling.seed if sampling is not None else None
        recipe = Recipe(
            steering=str(steering_obj) if steering_obj is not None else None,
            sampling=sampling,
            thinking=use_thinking_req,
            seed=seed_val,
            probes=list(self._monitor.probe_names),
        )
        recipe = recipe._fill_probe_hashes(self)
        return self.tree.begin_assistant(user_node_id, recipe=recipe)

    def _run_generation_loop(
        self,
        input_ids: torch.Tensor,
        gen_config: GenerationConfig,
        *,
        use_thinking: bool,
        want_hidden: bool,
        effective_tap: Callable[..., None] | None,
        seed: int | None,
        stop_list: list[str] | None,
        logit_bias: dict[int, float] | None,
        presence_penalty: float,
        frequency_penalty: float,
        lp_count: int | None,
        forced_prefix: list[int] | None = None,
    ) -> tuple[list[int], float]:
        """Run the decode loop once capture and steering are installed."""
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
        gating_callback = (
            self._build_gating_score_callback()
            if self._steering_needs_probe_gating()
            else None
        )
        use_static_cache = (
            self._cuda_graphs_active
            and cached_pkv is None
            and self._steering.all_fast_path()
        )
        generated_ids = generate_steered(
            self._model, self._tokenizer, effective_input_ids,
            gen_config, self._gen_state, thinking=use_thinking,
            on_token=effective_tap,
            seed=seed, stop=stop_list, logit_bias=logit_bias,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            logprobs=lp_count,
            trigger_ctx=self._steering.ctx,
            past_key_values=cached_pkv,
            cache_position_offset=cache_position_offset,
            score_callback=gating_callback,
            use_static_cache=use_static_cache,
            forced_prefix=forced_prefix,
        )
        elapsed = time.monotonic() - start

        if cached_pkv is not None and self._prefix_cache is not None:
            try:
                cached_pkv.crop(cache_position_offset)
            except (AttributeError, TypeError):
                self._invalidate_prefix_cache()
        return generated_ids, elapsed

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
        parent_node_id: str | None = None,
        recipe_override: "Recipe | str | None" = None,
        forced_prefix: list[int] | None = None,
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

        # v2.3 phase 5: apply recipe override (auto-regen / manual mode)
        # before constructing the Steering object so the overlay wins
        # over the per-call kwargs.
        if recipe_override is not None:
            steering, sampling, thinking = self._resolve_recipe_override(
                recipe_override,
                parent_node_id=parent_node_id,
                steering=steering,
                sampling=sampling,
                thinking=thinking,
            )

        (
            steering_obj,
            use_thinking_req,
            gen_config,
            lp_count,
            seed,
            stop_list,
            logit_bias,
            presence_penalty,
            frequency_penalty,
            logprobs_list,
        ) = self._prepare_generation_call(steering, sampling, thinking)
        # ``mean_logprob_accum`` averages chosen-token logprobs over the
        # non-thinking response span — surfaced on ``LoomNode.mean_logprob``
        # at finalize-assistant time so the loom sidebar can sort siblings
        # by surprise.  Populates whenever the engine captures chosen
        # logprob (any ``on_token`` consumer with ``lp_count is not None``
        # — i.e. the loom path or an explicit logprobs request).
        mean_logprob_sum: float = 0.0
        mean_logprob_count: int = 0
        trait_token_counter = [0]

        def _token_tap(text, is_thinking, tid, lp, top_alts, perplexity):
            nonlocal mean_logprob_sum, mean_logprob_count
            if logprobs_list is not None and tid >= 0 and not is_thinking:
                logprobs_list.append((tid, lp if lp is not None else 0.0, top_alts or []))
            if lp is not None and tid >= 0 and not is_thinking:
                mean_logprob_sum += lp
                mean_logprob_count += 1
            if assistant_node_id is not None and tid is not None:
                token_row: dict[str, Any] = {
                    "token_id": int(tid),
                    "text": text,
                    "logprob": float(lp) if lp is not None else None,
                    "perplexity": (
                        float(perplexity) if perplexity is not None else None
                    ),
                }
                if top_alts:
                    token_row["top_alts"] = [
                        {
                            "id": int(a.id),
                            "text": a.text,
                            "logprob": float(a.logprob),
                        }
                        for a in top_alts
                    ]
                # Per-token probe scoring lives here so the persisted node
                # row carries the same ``probes`` + ``per_layer_scores`` the
                # live WS ``token`` event ships.  Without this, the webui
                # rehydrates a tree across a page refresh with tokens but
                # no scores — highlight tint and the token-drilldown heatmap
                # silently go blank for historical turns.  The WS handler
                # reads these back off the row rather than recomputing.
                if self._monitor.probe_names:
                    latest_hidden_for_persist = {
                        layer_idx: bucket[-1]
                        for layer_idx, bucket in self._capture._per_layer.items()
                        if bucket
                    }
                    if latest_hidden_for_persist:
                        agg = self._monitor.score_single_token(
                            latest_hidden_for_persist,
                        )
                        if agg:
                            token_row["probes"] = {
                                p: round(float(v), 6) for p, v in agg.items()
                            }
                        per_layer = self._monitor.score_single_token_per_layer(
                            latest_hidden_for_persist,
                        )
                        if per_layer:
                            token_row["per_layer_scores"] = {
                                str(layer): {
                                    p: round(float(v), 6) for p, v in metrics.items()
                                }
                                for layer, metrics in per_layer.items()
                            }
                self.tree.append_token(
                    assistant_node_id,
                    token_row,
                    thinking=bool(is_thinking),
                )
            if on_token is not None:
                on_token(text, is_thinking, tid, lp, top_alts, perplexity)
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
        # The tap also writes per-token ``probes`` / ``per_layer_scores``
        # onto the loom row when probes are loaded and the gen is loom-
        # attached — required so a webui refresh can rehydrate highlight
        # tints and the token-drilldown heatmap from the server tree.
        _persists_probe_row = bool(not stateless and self._monitor.probe_names)
        _need_tap = (
            on_token is not None
            or logprobs_list is not None
            or _has_trait_consumer
            or _persists_probe_row
            or stop_list is not None
        )
        _effective_tap = _token_tap if _need_tap else None

        steering_cm = None
        if steering_obj is not None and steering_obj.alphas:
            steering_cm = self.steering(steering_obj)

        vector_snapshot: dict[str, float] = (
            self._snapshot_steering_alphas()
            if self._steering_stack or steering_cm is not None
            else {}
        )

        assistant_node_id = self._start_loom_assistant(
            input,
            stateless=stateless,
            parent_node_id=parent_node_id,
            sampling=sampling,
            steering_obj=steering_obj,
            use_thinking_req=use_thinking_req,
        )

        try:
            if steering_cm is not None:
                steering_cm.__enter__()
            input_ids, use_thinking, prompt_tokens = self._generation_preamble(
                input, raw, use_thinking_req, stateless=stateless,
                parent_node_id=parent_node_id,
            )
            # Refresh snapshot now that steering is pushed (first-scope case).
            vector_snapshot = self._snapshot_steering_alphas()

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

                generated_ids, elapsed = self._run_generation_loop(
                    input_ids,
                    gen_config,
                    use_thinking=use_thinking,
                    want_hidden=want_hidden,
                    effective_tap=_effective_tap,
                    seed=seed,
                    stop_list=stop_list,
                    logit_bias=logit_bias,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                    lp_count=lp_count,
                    forced_prefix=forced_prefix,
                )
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
            # Phase 1 logit pass: convert the in-loop logprob accumulator
            # into per-turn rollups before handing finalize the slot.
            # ``mean_logprob_count == 0`` covers both "no captures because
            # gen was empty" and "no captures because no on_token consumer
            # was wired" — both produce ``None`` so the wire/tree carry a
            # clean back-compat shape.
            _mean_logprob_out: float | None = None
            _mean_surprise_out: float | None = None
            if mean_logprob_count > 0:
                _mean_logprob_out = mean_logprob_sum / mean_logprob_count
                _mean_surprise_out = -_mean_logprob_out
            result = self._finalize_generation(
                input, generated_ids, elapsed, vector_snapshot,
                prompt_tokens=prompt_tokens, stateless=stateless,
                logprobs_list=logprobs_list,
                applied_steering=applied_steering,
                return_hidden=want_hidden,
                assistant_node_id=assistant_node_id,
                mean_logprob=_mean_logprob_out,
                mean_surprise=_mean_surprise_out,
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
            # Release the loom-tree reservation in the same scope as the
            # gen-lock release.  Even if finalize raised, mutators (edit /
            # delete on this subtree) need to be free again now that the
            # streaming target is no longer live.
            self._active_gen_reservation = None
            self._gen_phase = GenState.IDLE
            self._gen_lock.release()

    def _generate_runset(
        self,
        input,
        *,
        steering: "str | Steering | None" = None,
        sampling: SamplingConfig | None = None,
        stateless: bool = False,
        raw: bool = False,
        thinking: bool | None = None,
        on_token: Callable[..., None] | None = None,
        parent_node_id: str | None = None,
        n: int = 1,
        recipe_override: "Recipe | str | None" = None,
    ) -> RunSet:
        """Run one or more sibling generations and return a ``RunSet``."""
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        if n == 1:
            result = self._generate_core(
                input,
                steering=steering,
                sampling=sampling,
                stateless=stateless,
                raw=raw,
                thinking=thinking,
                on_token=on_token,
                parent_node_id=parent_node_id,
                recipe_override=recipe_override,
            )
            node_id = self.tree.active_node_id if not stateless else None
            return RunSet([result], node_ids=[node_id], kind="generation")

        # N-way regen: derive per-sibling seeds from the supplied base seed
        # (or a fresh entropy-derived one). Each iteration runs
        # ``_generate_core`` independently; ``add_user_turn`` dedups so all
        # siblings share the same user-parent.
        base_seed = sampling.seed if sampling is not None else None
        schedule = derive_seed_schedule(base_seed, n)
        results: list[GenerationResult] = []
        node_ids: list[str | None] = []
        for seed_i in schedule:
            from dataclasses import replace as _replace
            si = sampling if sampling is not None else SamplingConfig()
            si = _replace(si, seed=seed_i)
            r = self._generate_core(
                input,
                steering=steering,
                sampling=si,
                stateless=stateless,
                raw=raw,
                thinking=thinking,
                on_token=on_token,
                parent_node_id=parent_node_id,
                recipe_override=recipe_override,
            )
            results.append(r)
            node_ids.append(self.tree.active_node_id if not stateless else None)
            # External stop requested mid-batch: cancel the remainder.
            # Sibling boundaries are the only valid stop points.
            if self._gen_state.stop_requested.is_set():
                break
        return RunSet(results, node_ids=node_ids, kind="fan")

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
        parent_node_id: str | None = None,
        n: int = 1,
        recipe_override: "Recipe | str | None" = None,
    ) -> RunSet:
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
                logprob, top_alts, perplexity)`` called on each emitted
                token.  ``top_alts`` is ``list[TokenAlt]`` (decoded
                ``(id, text, logprob)`` triples) when ``sampling.logprobs > 0``
                or ``sampling.return_top_k > 0``; ``None`` otherwise.
                ``perplexity`` is ``exp(entropy_nats)`` of the
                sampler distribution after temperature, top-k, and
                top-p renormalization.
            parent_node_id: loom-tree node id to anchor the new turn
                under.  ``None`` = active node (today's behavior).
            n: fan-out count.  ``n=1`` (default) returns a one-result
                :class:`RunSet`; ``n>1`` runs the same prompt
                ``n`` times under deterministically-derived per-sibling
                seeds (see :func:`~saklas.core.loom.derive_seed_schedule`)
                and returns a multi-result ``RunSet`` in sibling order.

        Returns:
            :class:`RunSet` in every case.  It is list-like for fan-out
            and exposes ``.first`` for the common single-result case.
        """
        return self._generate_runset(
            input,
            steering=steering,
            sampling=sampling,
            stateless=stateless,
            raw=raw,
            thinking=thinking,
            on_token=on_token,
            parent_node_id=parent_node_id,
            n=n,
            recipe_override=recipe_override,
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
        parent_node_id: str | None = None,
        recipe_override: "Recipe | str | None" = None,
        live_scores: bool = True,
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

        ``live_scores=False`` skips inline per-token probe scoring for this
        stream. Final aggregate/per-token readings are still computed during
        generation finalization from the captured hidden states.
        """
        q: queue.SimpleQueue = queue.SimpleQueue()
        done = object()
        result_holder: list[GenerationResult] = []
        exc_holder: list[BaseException] = []
        idx_counter = [0]

        def _push(text, is_thinking, tid, lp, top_alts, perplexity):
            scores: dict[str, float] | None = None
            if live_scores and self._monitor.probe_names:
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
                thinking=is_thinking, logprob=lp, top_alts=top_alts,
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
                    parent_node_id=parent_node_id,
                    recipe_override=recipe_override,
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
    ) -> RunSet:
        """Run N prompts under the same steering, return a ``RunSet``.

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

        ``on_result(idx, result)`` fires after each completion for local
        progress hooks.

        Returns:
            ``RunSet`` aligned with ``prompts``.  ``runset.grid`` records
            ``{"prompt_index": i}`` for each row.
        """
        if not isinstance(prompts, list) or not prompts:
            raise ValueError("generate_batch: prompts must be a non-empty list")

        results: list[GenerationResult] = []
        node_ids: list[str | None] = []
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
            node_ids.append(self.tree.active_node_id if not stateless else None)
            if on_result is not None:
                on_result(idx, r)
        return RunSet(
            results,
            node_ids=node_ids,
            grid=[{"prompt_index": i} for i in range(len(results))],
            kind="batch",
        )

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
        parent_node_id: str | None = None,
    ) -> RunSet:
        """Fan a single prompt across a Cartesian product of alpha values.

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
            ``RunSet`` in product order.  ``runset.grid[i]`` is the
            alpha dict for row ``i``; ``runset.node_ids[i]`` is the
            assistant node id when ``stateless=False``.
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
        total = 1
        for values in alpha_lists:
            total *= len(values)
        base_seed = sampling.seed if sampling is not None else None
        seed_schedule = derive_seed_schedule(base_seed, total)

        base_str: str | None
        if base_steering is None:
            base_str = None
        elif isinstance(base_steering, str):
            base_str = base_steering
        else:
            # Pre-built Steering — render through the canonical formatter
            # so we can compose with new alpha terms via string concat.
            base_str = str(base_steering)

        # v2.3 loom: anchor every sibling under a shared user turn so
        # the surfaces render the sweep as siblings under a common
        # parent rather than a flat result list.  Stateless sweeps
        # skip the tree mutation entirely (matches the v2.2 contract);
        # stateful sweeps land siblings under ``parent_node_id`` (or
        # the active node when None) and dedup on identical user text.
        anchor_user_id: str | None = None
        if not stateless and isinstance(prompt, str):
            anchor_user_id = self.tree.add_user_turn(
                prompt, parent_id=parent_node_id,
            )
            # The anchor parent for ``_generate_core`` is the user
            # node's *parent* — generate's ``add_user_turn`` will dedup
            # against the user we just spawned, so every sibling gets
            # attached under it without spawning duplicate user turns.
            gen_parent_id = self.tree.nodes[anchor_user_id].parent_id
        else:
            gen_parent_id = parent_node_id

        results: list[GenerationResult] = []
        sibling_node_ids: list[str | None] = []
        grid_rows: list[dict[str, float]] = []
        for idx, combo in enumerate(itertools.product(*alpha_lists)):
            alpha_values = dict(zip(concept_names, combo))
            alpha_values = {k: float(v) for k, v in alpha_values.items()}
            terms = [f"{alpha} {name}" for name, alpha in alpha_values.items()]
            expr = " + ".join(terms)
            if base_str:
                expr = f"{base_str} + {expr}"

            from dataclasses import replace as _replace
            si = sampling if sampling is not None else SamplingConfig()
            si = _replace(si, seed=seed_schedule[idx])

            r = self._generate_core(
                prompt,
                steering=expr,
                sampling=si,
                stateless=stateless,
                raw=raw,
                thinking=thinking,
                parent_node_id=gen_parent_id,
            )
            results.append(r)
            # The sibling's assistant node id is the active node after
            # ``_generate_core`` returns — ``finalize_assistant`` leaves
            # it active so the path-walker view stays coherent.
            sib_id = self.tree.active_node_id if not stateless else None
            sibling_node_ids.append(sib_id)
            grid_rows.append(alpha_values)
            if on_result is not None:
                on_result(idx, r, alpha_values)

        return RunSet(
            results,
            node_ids=sibling_node_ids,
            grid=grid_rows,
            kind="fan",
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
