"""SaklasSession — unified backend for saklas's programmatic and server APIs."""
from __future__ import annotations
import asyncio
import json
import logging
import queue
import threading
import time
from contextlib import contextmanager, nullcontext, suppress
from dataclasses import dataclass
from enum import Enum, IntEnum
from types import TracebackType
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Callable, Iterator, Literal, Protocol, cast, overload

import torch
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    StoppingCriteria,
    StoppingCriteriaList,
)

from saklas.core.errors import SaklasError
from saklas.core.events import (
    EventBus,
    GenerationFinished,
    GenerationStarted,
    ManifoldExtracted,
    ProbeScored,
)
from saklas.core.generation import (
    GenerationConfig,
    GenerationState,
    _get_eos_ids,
    build_chat_input,
    detect_base_model,
    generate_steered,
    supports_thinking,
)
from saklas.core.hooks import HiddenCapture, SteeringManager
from saklas.core.naming import BIPOLAR_SEP, _slug, canonical_concept_name
from saklas.io import selectors as _selectors
from saklas.core.token_callback import (
    StepTokenCallback,
    TokenCallback,
    TokenConsumer,
    TokenConsumerOptions,
    consumer_options,
)
from saklas.core.scene import (
    SceneGrammarError,
    TurnGrammar,
    extract_turn_grammar,
    validate_turn_grammar,
)
from saklas.core.loom import (
    CastMember,
    InvalidNodeOperationError,
    LoomMutated,
    LoomTree,
    LoomTreeError,
    MutationDuringGenerationError,
    Recipe,
    derive_seed_schedule,
)
from saklas.core.model import (
    get_layers,
    get_model_info,
    load_model,
    loaded_model_fingerprint,
    workspace_layer_indices,
)
from saklas.core.instruments.types import (
    AGG_TAIL_DEPTH,
    ReadRequest,
)
from saklas.core.monitor import Monitor
from saklas.io.probes_bootstrap import bootstrap_layer_means
from saklas.core.profile import Profile, load_profile as _load_profile
from saklas.core.results import (
    GenerationResult,
    ProbeReading,
    RunSet,
    TokenAlt,
    TokenEvent,
)
from saklas.core.sampling import SamplingConfig
from saklas.core.steering import Steering
from saklas.core.steering_expr import AblationTerm, ManifoldTerm, ProjectedTerm
from saklas.core.manifold import Manifold

if TYPE_CHECKING:
    from saklas.core.instruments.geometry import GeometryInstrument
    from saklas.core.instruments.lens import LensInstrument
    from saklas.core.instruments.sae import SaeInstrument
    from saklas.core.scoring import ChoiceScores
    from saklas.core.steering_composer import SteeringComposer
    from saklas.io.templates import TemplateFolder
from saklas.core.triggers import Trigger

_log = logging.getLogger(__name__)

_DEFAULT_JLENS_TOP_K = 8


class GenerationStream(Protocol):
    """Current streaming iterator contract returned by :meth:`generate_stream`."""

    def __iter__(self) -> "GenerationStream": ...

    def __next__(self) -> TokenEvent: ...

    def close(self) -> None: ...

    @property
    def result(self) -> GenerationResult | None: ...

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
        layer._saklas_la_snapshot = snap or None

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

    linear_attention_layer: Any = LinearAttentionLayer
    hybrid_attention_layer: Any = LinearAttentionAndFullAttentionLayer
    linear_attention_layer.crop = _la_crop_with_restore
    hybrid_attention_layer.crop = _hybrid_crop
    linear_attention_layer._saklas_save_snapshot = _save_la_snapshot
    linear_attention_layer._saklas_crop_patched = True
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


_RESPONSE_MAX_TOKENS = 256  # per in-character response (4.0 / A2 elicitation)
# Chunk size for batched corpus generation (``_run_generator_batch``, the
# generation half of ``generate_responses`` / ``generate_neutral_responses``).
# One left-padded ``model.generate`` over this many baseline prompts replaces
# that many sequential batch-1 decodes — the dominant cost of extraction's
# corpus-generation phase (each decode is up to ``_RESPONSE_MAX_TOKENS`` long).
_CORPUS_GEN_BATCH = 16
# Tail-ring depth for aggregate-only capture (probes attached, no per-token
# consumer): how many trailing hidden slices to retain so finalize can pool the
# last *content* token after walking back past trailing special tokens (EOS,
# end-of-turn). 8 comfortably covers any chat template's trailing-marker run.
# Canonical value lives in the instrument-plan vocabulary so families can
# declare the ring depth they demand.
_AGG_TAIL_DEPTH = AGG_TAIL_DEPTH

# Floor on the shared token prefix ``generate_batch`` will KV-cache.  Below
# this the saved prefill is in the noise and the double-render + cache-management
# overhead isn't worth it; the prefix must also leave a non-empty suffix per row
# (``_try_prefix_cache_hit`` requires it), so the cached length is capped at
# ``min_row_len - 1``.
_PREFIX_CACHE_MIN_TOKENS = 16


@dataclass(slots=True)
class _PrefixCacheEntry:
    prefix_ids_cpu: torch.Tensor
    past_key_values: object
    prefix_len: int
    static: bool = False
    max_cache_len: int | None = None


@dataclass(slots=True)
class _AuthoredPromptTarget:
    """One visible authored node channel awaiting its first prompt capture."""

    node_id: str
    thinking: bool
    token_ids: tuple[int, ...]
    token_texts: tuple[str, ...]


@dataclass(slots=True)
class _AuthoredPromptMatch:
    """A target located in the full rendered prompt token sequence."""

    target: _AuthoredPromptTarget
    token_positions: tuple[int, ...]
    capture_columns: tuple[int | None, ...] = ()


@dataclass(slots=True)
class _AuthoredPromptCapture:
    """Per-generation policy for loom-authored prefill measurements."""

    targets: list[_AuthoredPromptTarget]
    monitor_active: bool
    lens_active: bool
    sae_active: bool
    lens_top_k: int
    persist_per_layer_scores: bool
    steering: str | None
    search_end: int | None = None


@dataclass(slots=True)
class _SerialGenerationJob:
    input: Any
    steering: Any = None
    sampling: SamplingConfig | None = None
    raw: bool = False
    thinking: bool | None = None
    on_token: TokenCallback | None = None
    parent_node_id: str | None = None
    recipe_override: Recipe | str | None = None
    grid: dict[str, Any] | None = None
    gen_seat: str = "assistant"


class _SessionStopCriteria(StoppingCriteria):
    """Bridge ``SaklasSession.stop()`` into ``transformers.generate``."""

    def __init__(self, state: GenerationState) -> None:
        self._state = state

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor | None,
        **kwargs: Any,
    ) -> Any:
        del scores, kwargs
        stop = 1 if self._state.stop_requested.is_set() else 0
        return input_ids.new_full((int(input_ids.shape[0]),), stop).bool()


def _run_serial_generation_jobs(
    session: Any,
    jobs: list[_SerialGenerationJob],
    *,
    stateless: bool,
    kind: str,
    on_result: Callable[[int, GenerationResult, dict[str, Any]], None] | None = None,
    stop_between_jobs: bool = False,
) -> RunSet:
    """Run generation jobs through the shared serial executor.

    This is still a one-row-at-a-time runner; centralizing it gives the
    eventual true batched decode path one place to branch from, instead of
    keeping separate loops in fan, batch, and sweep surfaces.
    """
    results: list[GenerationResult] = []
    node_ids: list[str | None] = []
    grid_rows: list[dict[str, Any]] = []
    for idx, job in enumerate(jobs):
        result = session._generate_core(
            job.input,
            steering=job.steering,
            sampling=job.sampling,
            stateless=stateless,
            raw=job.raw,
            thinking=job.thinking,
            on_token=job.on_token,
            parent_node_id=job.parent_node_id,
            recipe_override=job.recipe_override,
            gen_seat=job.gen_seat,
        )
        row = dict(job.grid or {})
        results.append(result)
        node_ids.append(session.tree.active_node_id if not stateless else None)
        grid_rows.append(row)
        if on_result is not None:
            on_result(idx, result, row)
        if stop_between_jobs and session._gen_state.stop_requested.is_set():
            break
    return RunSet(results, node_ids=node_ids, grid=grid_rows, kind=kind)


PROBE_CATEGORIES = [
    "epistemic",
    "alignment",
    "register",
    "cultural",
]
MIN_ELAPSED_FOR_RATE = 0.1


def _humanize_concept(name: str) -> str:
    """Invert the slug `_` convention for LLM-facing prompts.

    Pack names and alphas keys use underscores (``artificial_intelligence``);
    the generator reads them better as spaces. Disk paths, canonical
    names, and progress messages keep the slug form.
    """
    return name.replace("_", " ")


# Conversational (4.0 / A2) elicitation templates keyed by node ``kind``.  The
# concept goes in the system prompt; {c} is the humanized concept, {art} its
# a/an article.  ``abstract`` traits read as "someone {c}", ``concrete``
# entities as "{art} {c}".
_KIND_TEMPLATES = {
    "abstract": "You are someone {c}. Respond exactly as someone {c} would.",
    "concrete": "You are {art} {c}. Respond exactly as {art} {c} would.",
}


def _article(word: str) -> str:
    """Naive a/an for the concrete-entity template (an alien, a deer)."""
    return "an" if word[:1].lower() in "aeiou" else "a"


def _system_for(
    concept_h: str, kind: str | None, custom: str | None = None,
) -> str:
    """A2 system prompt for a humanized concept under its kind (default abstract).

    ``abstract`` reads as "someone {c}", ``concrete`` as "{art} {c}", and
    ``custom`` uses the caller-supplied ``custom`` template — a free-form
    elicitation frame (``{c}``/``{art}`` placeholders) for a concept that is
    neither a trait nor an entity (a month embodying its season, say), so the
    generation framing is no longer limited to the two fixed templates.
    """
    if (kind or "abstract") == "custom":
        if not custom:
            raise ValueError(
                "_system_for: kind='custom' requires a custom system template"
            )
        return custom.format(c=concept_h, art=_article(concept_h))
    template = _KIND_TEMPLATES.get(kind or "abstract", _KIND_TEMPLATES["abstract"])
    return template.format(c=concept_h, art=_article(concept_h))


def _role_for(slug: str, kind: str | None) -> str | None:
    """A2 elicitation role label (the swapped assistant header) for a node.

    Abstract traits get a ``someone_{slug}`` speaker ("someone happy"); concrete
    entities are the bare slug ("pirate").  The underscore de-slugs to a space
    at render (:func:`saklas.core.role_templates._render_label`).  ``custom``
    returns ``None`` — no role swap; the custom system prompt carries the
    framing and the corpus is pooled in standard-assistant space (the
    persona-stays-generation-only pattern).
    """
    k = kind or "abstract"
    if k == "custom":
        return None
    return f"someone_{slug}" if k == "abstract" else slug


def _split_composite_source(
    concept: str, baseline: str | None,
) -> tuple[str, str | None]:
    """Split a composite ``pos.neg`` slug when no explicit baseline is given.

    ``canonical_concept_name`` already performs this split for the
    storage name.  ``extract()`` needs the same split at the generator
    interface so :meth:`SaklasSession.generate_responses` elicits
    ``concept`` and ``baseline`` as two distinct poles — otherwise the LLM
    sees one composite blob and the pole assignment no longer matches the
    user's declared order.
    """
    if baseline is None and BIPOLAR_SEP in concept:
        pos, neg = concept.split(BIPOLAR_SEP, 1)
        return pos.strip(), neg.strip()
    return concept, baseline


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


class CaptureMode(Enum):
    """How the per-gen hidden-state capture scores its probes (legal-by-construction).

    Replaces the five correlated capture booleans (``_capture_incremental`` /
    ``_capture_aggregate_only`` / ``_capture_lean`` / ``_capture_gating_subset``)
    that a hand-kept if/elif chain used to keep consistent — illegal combinations
    (e.g. incremental *and* aggregate-only) were representable.  One mode is the
    single source of truth; :meth:`SaklasSession._begin_capture` picks it and every
    ``_score_*`` dispatch keys off it.  The modes trade only *when/how* scoring
    runs (per token vs once) and what memory the capture keeps (length-1 buffer vs
    tail ring vs full stack); every read is a full per-probe ``ProbeReading`` (see
    ``hooks.py`` ``HiddenCapture``).

    - ``INCREMENTAL`` — a full-reading live consumer wants per-token readings:
      score each token live (full roster) into ``_incremental_readings``.
    - ``LEAN_INCREMENTAL`` (FIX F2) — the only per-token consumers read axis-0
      coords (trait stream / loom probe row): score each token ``coords_only`` and
      re-score the full aggregate once at finalize from a bounded tail ring.
    - ``AGGREGATE_ONLY`` — probes attached but nothing consumes a per-token
      reading (e.g. stateless server gen): NO per-token scoring, pool the last
      content token once at finalize from a bounded tail ring.
    - ``GATING_SUBSET`` (FIX #4) — per-token scoring is needed *only* to feed probe
      gates: score just the gated subset per token (into ``_incremental_gate_scores``)
      while a tail ring lets finalize pool the FULL roster once.
    - ``FULL`` — full-retention append (``return_hidden`` widen, or any
      non-incremental read), or the degenerate no-probe / capture-disabled state:
      distinct clones per step so ``stacked()`` builds the full ``[T, D]``.
    """

    INCREMENTAL = "incremental"
    LEAN_INCREMENTAL = "lean_incremental"
    AGGREGATE_ONLY = "aggregate_only"
    GATING_SUBSET = "gating_subset"
    FULL = "full"


@dataclass
class CaptureState:
    """Per-generation capture configuration — the legal-by-construction state.

    Carries the one :class:`CaptureMode` plus the orthogonal ``persistent`` flag
    (capture rides the always-on compile-clean buffers rather than transient
    per-gen hooks) and, for gated generations, the gated probe names / exact
    scalar keys.  Set wholesale in :meth:`SaklasSession._begin_capture`; read by
    the ``_score_*`` dispatch, the gating callback, and the streaming tap.  The
    convenience predicates name the modes so the read sites stay legible.
    """

    mode: "CaptureMode" = CaptureMode.FULL
    persistent: bool = False
    gating_subset: "set[str] | None" = None
    gating_keys: "set[str] | None" = None
    final_probe_aggregate: bool = True

    @property
    def incremental(self) -> bool:
        return self.mode is CaptureMode.INCREMENTAL

    @property
    def lean(self) -> bool:
        return self.mode is CaptureMode.LEAN_INCREMENTAL

    @property
    def aggregate_only(self) -> bool:
        # GATING_SUBSET also finalizes off the tail ring (its per-token rows feed
        # only the gate), so it shares the aggregate-only full-roster pool when
        # the caller still wants a final full probe aggregate.
        return (
            self.mode is CaptureMode.AGGREGATE_ONLY
            or (
                self.mode is CaptureMode.GATING_SUBSET
                and self.final_probe_aggregate
            )
        )


class ConcurrentGenerationError(RuntimeError, SaklasError):
    """Raised when a generation call is made while another is in progress."""

    def user_message(self) -> tuple[int, str]:
        return (409, str(self) or self.__class__.__name__)


class ProfileNotRegisteredError(KeyError, SaklasError):
    """Raised when a steering call references a profile not in the registry."""

    def user_message(self) -> tuple[int, str]:
        # KeyError str-formats the message as repr; reach into args
        # so the user sees the original text.
        msg = self.args[0] if self.args else self.__class__.__name__
        return (404, str(msg))


class ManifoldNotRegisteredError(KeyError, SaklasError):
    """Raised when manifold steering references an unknown / unfitted manifold."""

    def user_message(self) -> tuple[int, str]:
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
# through the stack, ``flatten_stack``, ``push``/``pop``, and is
# dispatched by type in ``SteeringComposer.compose_steering_entries``.
SteeringStackEntry = tuple[float, Trigger] | AblationTerm | ManifoldTerm


def _affine_manifold_push(
    manifold: "Manifold", position: "tuple[float, ...] | str",
) -> "tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]":
    """An affine ``%`` term → a rank-R push fragment per layer.

    Both position forms are supported and **equivalent at the nodes**:

    - **Label form** (``position`` a ``str``): the node's per-layer real coords
      (``LayerSubspace.node_coords[idx]``) are the push target on that layer's
      basis — a direct table lookup, no interpolation.
    - **Coord form** (``position`` a coord tuple): the free authoring
      coordinate is mapped into each layer's reduced frame by cardinal RBF
      interpolation over the node layout (:func:`rbf_cardinal_weights` on the
      shared ``Manifold.node_coords``), so the per-layer target is the layout
      blend ``node_coords_L.T @ w``.  The weights are solved once in authoring
      space (layer-agnostic) and are exact at the nodes (``w = e_idx`` at node
      ``idx``), so placing a coord-form term at a node's coords reproduces the
      label-form push toward that node, and interpolates the per-layer target
      between nodes off-node — the flat-subspace analogue of a curved ``%``
      following its RBF surface instead of a straight chord.

    The affine fit's domain is the identity carrier (authoring == embedded), so
    the weights are built directly in authoring-coordinate space.  Arity
    (coord form) is validated against the domain's intrinsic dim, matching the
    curved :meth:`SteeringManager.add_manifold` path; a non-poised layout (no
    interpolant) raises ``SteeringExprError`` advising the user to steer by
    node label instead.
    """
    from saklas.core.steering_expr import SteeringExprError
    from saklas.core.manifold import rbf_cardinal_weights

    weights: "torch.Tensor | None" = None
    idx = 0
    if isinstance(position, str):
        idx = manifold.nearest_node_index(position)
    else:
        n = manifold.domain.intrinsic_dim
        pos = tuple(float(c) for c in position)
        if len(pos) != n:
            from saklas.core.errors import ManifoldArityError
            raise ManifoldArityError(
                f"manifold {manifold.name!r} has a {n}-dimensional domain but "
                f"the steering position has {len(pos)} coordinate(s)"
            )
        query = torch.tensor(pos, dtype=torch.float32)
        try:
            weights = rbf_cardinal_weights(manifold.node_coords, query)
        except ValueError as exc:
            raise SteeringExprError(
                f"manifold {manifold.name!r}: cannot place free coord-form "
                f"position {pos!r} — {exc}; steer by node label instead "
                f"(e.g. '{manifold.name}%<label>')"
            ) from exc

    basis_dirs: dict[int, torch.Tensor] = {}
    coord_dirs: dict[int, torch.Tensor] = {}
    for L, sub in manifold.layers.items():
        if sub.node_coords is None:
            raise SteeringExprError(
                f"affine manifold {manifold.name!r} layer {L} carries no "
                f"per-layer node_coords; re-fit to steer it by node label"
            )
        basis_dirs[L] = sub.basis
        if weights is None:
            coord_dirs[L] = sub.node_coords[idx]
        else:
            w = weights.to(device=sub.node_coords.device, dtype=sub.node_coords.dtype)
            coord_dirs[L] = sub.node_coords.transpose(0, 1) @ w   # (R,)
    return basis_dirs, coord_dirs


class _SteeringContext:
    """Context manager returned by SaklasSession.steering().

    Pushes an entries dict onto the session's steering composer on ``__enter__``
    and pops it on ``__exit__``.  Rebuilds hooks from the flattened stack
    head so nested scopes compose: inner entries overwrite outer entries
    for the duration of the inner scope, then the outer entry is restored.

    The stored ``_entries`` is the post-resolution entries form — each
    value is either ``(alpha, Trigger)`` for additive/projection terms or
    an :class:`~saklas.core.steering_expr.AblationTerm` for mean-replacement
    ablation.  Bare-alpha inputs to the public ``steering()`` API are
    normalized before we get here.

    ``~``/``|`` projection terms always materialize in the Mahalanobis
    metric (closed-form LEACE against the session whitener) — there is no
    per-call metric override.
    """

    __slots__ = (
        "_active_role",
        "_entered",
        "_entries",
        "_prev_active_role",
        "_session",
        "_synthetic_snapshots",
    )

    def __init__(
        self,
        session: "SaklasSession",
        entries: dict[str, SteeringStackEntry],
        *,
        synthetic_snapshots: dict[str, "object"] | None = None,
        active_role: str | None = None,
    ) -> None:
        self._session = session
        self._entries = entries
        self._entered = False
        # Pre-materialize snapshots of any synthetic-projection keys
        # this scope wrote to ``session._profiles`` — value is the prior
        # binding (or :data:`_PROFILE_ABSENT` when the key was unset).
        # Restored on ``__exit__`` so nested scopes that re-materialize
        # the same ``a|b`` synthetic key don't leak the inner tensor back
        # into the outer scope's hooks.
        self._synthetic_snapshots: dict[str, object] = (
            dict(synthetic_snapshots) if synthetic_snapshots else {}
        )
        # Active role (role-extraction Phase 7): the role label every
        # role-augmented term in this scope shares.  Restored on exit so
        # nested scopes save/restore inner-wins.
        self._active_role = active_role
        self._prev_active_role: str | None = None

    def __enter__(self) -> "_SteeringContext":
        # _push_steering rolls its own stack entry back if _rebuild_steering_hooks
        # raises (e.g. ProfileNotRegisteredError).  __enter__ only flips
        # `_entered=True` AFTER a clean push so a mid-__enter__ failure leaves
        # no stale state for __exit__ to pop.
        self._session._push_steering(self._entries)
        # Save / overwrite the session-level active_role.  Inner scopes
        # override outer; the outer is restored on ``__exit__``.  An
        # inner scope with ``active_role=None`` inherits — leave the
        # outer's value in place.
        self._prev_active_role = getattr(self._session, "_active_role", None)
        if self._active_role is not None:
            self._session._active_role = self._active_role
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
        # Restore the prior active_role.  Done unconditionally — even
        # if pop_steering raised — so the session never leaks a stale
        # role binding into the next request.
        if self._active_role is not None:
            self._session._active_role = self._prev_active_role
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
                    profiles[key] = prev  # pyright: ignore[reportArgumentType]  # sentinel restore: prev is dict[int, Tensor] at runtime
            self._synthetic_snapshots = {}


# Sentinel for ``_SteeringContext._synthetic_snapshots`` entries —
# distinguishes "key was previously absent" from "key was previously
# bound to None" (the latter shouldn't happen in practice but the
# distinction keeps restore semantics unambiguous).
_PROFILE_ABSENT = object()


def _jlens_sidecar_identity(sidecar: "Mapping[str, Any] | None") -> tuple[Any, ...] | None:
    """Stable identity binding a resident fp32 lens to its disk artifact."""
    if sidecar is None:
        return None
    return (
        sidecar.get("tensor_sha256"),
        sidecar.get("corpus_sha256"),
        sidecar.get("corpus_hash_kind"),
        sidecar.get("model_fingerprint"),
        int(sidecar.get("n_prompts", 0)),
        int(sidecar.get("d_model", 0)),
        tuple(int(layer) for layer in sidecar.get("source_layers", [])),
        int(sidecar.get("seq_len", 0)),
        sidecar.get("estimator_policy"),
        json.dumps(sidecar.get("_source"), sort_keys=True, default=str),
    )


def _jlens_matches_loaded_model(
    sidecar: "Mapping[str, Any] | None", model: Any, model_id: str,
) -> bool:
    """Validate local exact-weight or external immutable-checkpoint identity."""
    if sidecar is None:
        return False
    fitted_fingerprint = sidecar.get("model_fingerprint")
    if isinstance(fitted_fingerprint, str) and fitted_fingerprint:
        return fitted_fingerprint == loaded_model_fingerprint(model, model_id)
    source = sidecar.get("_source")
    if not isinstance(source, Mapping) or source.get("kind") != "huggingface":
        return False
    if str(source.get("model_id", "")).casefold() != model_id.casefold():
        return False
    base = getattr(model, "_orig_mod", model)
    config = getattr(base, "config", getattr(model, "config", None))
    live_commit = (
        getattr(config, "_commit_hash", None)
        or getattr(getattr(config, "text_config", None), "_commit_hash", None)
    )
    if not isinstance(live_commit, str) or live_commit != source.get("model_revision"):
        return False
    text_config = getattr(config, "text_config", config)
    hidden = getattr(text_config, "hidden_size", getattr(text_config, "n_embd", None))
    n_layers = getattr(
        text_config, "num_hidden_layers", getattr(text_config, "n_layer", None),
    )
    layers = sidecar.get("source_layers", [])
    return bool(
        isinstance(hidden, int)
        and not isinstance(hidden, bool)
        and int(sidecar.get("d_model", -1)) == hidden
        and isinstance(n_layers, int)
        and not isinstance(n_layers, bool)
        and isinstance(layers, list)
        and layers
        and all(isinstance(layer, int) and 0 <= layer < n_layers for layer in layers)
    )


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

        ``~`` / ``|`` projection terms always materialize through the
        closed-form LEACE projector against the per-model whitener (it
        provably erases linearly-decodable information along ``onto`` from
        ``base``); there is no Euclidean path and no metric knob.

        ``dls`` toggles the discriminative-layer-selection mask at
        extraction time (v2.1+).  When ``True`` (default), centered DLS
        per Dang & Ngo (2026) Eq. 9 drops layers where pos- and
        neg-class means project to the same side of the neutral
        baseline along ``d̂``.  Replaces the v2.0–v2.1 ``edge_drop``
        heuristic (gone in v2.1); pass ``dls=False`` (CLI ``--no-dls``)
        to opt out.

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
        _cg_reason: str | None = None
        device_obj = next(model.parameters()).device
        if cuda_graphs:
            from saklas.core.cuda_graphs import is_cuda_graphs_supported
            cg_supported, _cg_reason = is_cuda_graphs_supported(
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
        #
        # MPS is now eligible (not just CUDA): inductor's MPS backend fuses the
        # per-layer kernels and amortizes dispatch, which is the dominant decode
        # cost on Metal — measured ~1.7x on gemma-3-4b paired with StaticCache.
        # CUDA-graph capture (``reduce-overhead``) stays CUDA-only, so on MPS
        # ``effective_compile_mode`` is ``"default"`` (``cg_supported`` is False
        # there) — fusion without graph capture, which composes with the
        # transient steering hooks and StaticCache the decode loop installs.
        offset_buffers: dict[int, Any] = {}
        offset_handles: list[Any] = []
        capture_buffers: dict[int, Any] = {}
        capture_handles: list[Any] = []
        if compile and device_obj.type in ("cuda", "mps"):
            from saklas.core.hooks import (
                install_persistent_capture_hooks,
                install_persistent_offset_hooks,
            )
            from saklas.core.model import _compile_with_probe, get_layers
            # Attach the persistent branchless steering hooks BEFORE
            # compile so they ride inside the captured graph (post-compile hook
            # changes are not retraced — ``skip_nnmodule_hook_guards``).  The
            # offset hooks (``add_(offset)``) carry static-affine steering and
            # update their buffers in place per gen, so the compiled fast path
            # sees a stable hook topology and never recompiles. Persistent
            # capture hooks (``copy_`` the last-token slice) are installed only
            # when the construction-time probe roster is non-empty/default; an
            # explicit ``probes=[]`` session takes a no-capture compiled mode and
            # later ad-hoc probes fall back to transient capture. ``hidden_size``
            # falls back to the text sub-config for multimodal wrappers
            # (Gemma-3/4).
            hidden_size = getattr(model.config, "hidden_size", None)
            if hidden_size is None and hasattr(model.config, "get_text_config"):
                hidden_size = model.config.get_text_config().hidden_size
            if hidden_size is not None:
                model_dtype = next(model.parameters()).dtype
                layers_ml = get_layers(model)
                offset_buffers, offset_handles = install_persistent_offset_hooks(
                    layers_ml, int(hidden_size), device_obj, model_dtype,
                )
                if probes is None or bool(probes):
                    capture_buffers, capture_handles = (
                        install_persistent_capture_hooks(
                            layers_ml, int(hidden_size), device_obj, model_dtype,
                        )
                    )
            model = _compile_with_probe(
                model, tokenizer, device_obj,
                mode=effective_compile_mode,
            )
        elif compile:
            _log.info(
                "compile=True but device=%s — skipping torch.compile "
                "(supported only on CUDA / MPS)",
                device_obj.type,
            )

        # ``__init__`` consults the same probe helper, but the helper now
        # caches by the underlying module id (survives the torch.compile
        # wrapper via ``_orig_mod``), so this second call is a dictionary
        # lookup rather than another StaticCache(max_cache_len=1)
        # allocation.
        session = cls(
            model,  # pyright: ignore[reportArgumentType]  # may be torch.compile OptimizedModule wrapping PreTrainedModel
            tokenizer,
            probes=probes,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            dls=dls,
            cuda_graphs=cuda_graphs,
            return_top_k=return_top_k,
        )
        # Adopt the persistent offset + capture hooks iff compile actually stuck;
        # if it fell back to eager, drop them (their add_(0) / copy_ would be pure
        # overhead with no compiled graph to ride).
        if offset_handles or capture_handles:
            if session._compiled:
                session._steering.adopt_compiled_offsets(
                    offset_buffers, offset_handles,
                )
                session._capture_buffers = capture_buffers
                session._capture_handles = capture_handles
            else:
                for h in offset_handles:
                    h.remove()
                for h in capture_handles:
                    h.remove()
        return session

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        *,
        probes: list[str] | None = None,
        system_prompt: str | None = None,
        max_tokens: int = 1024,
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
        # Manifold registry: registry key -> loaded Manifold artifact,
        # populated lazily by ``ensure_manifold_loaded`` on scope entry.
        self._manifolds: dict[str, Manifold] = {}

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
        self._steering = SteeringManager()
        # CUDA-graphs / StaticCache routing (Phase B, v2.2).  Probe
        # support once at construction so the per-generation hot path
        # only consults a boolean.  Off when (a) user opted out, (b)
        # device != cuda, or (c) the model's StaticCache constructor
        # raises (architecture-specific quirks).  The fallback reason
        # is logged once via :func:`saklas.core.cuda_graphs.warn_once`
        # which dedupes on ``id(model)``, so when ``from_pretrained``
        # already probed for its compile_mode decision and we re-probe
        # here, the user only sees one message.
        cuda_graphs_requested = bool(cuda_graphs)
        self._cuda_graphs_active: bool = False
        if cuda_graphs_requested:
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
        # ``torch.compile`` left an ``_orig_mod`` wrapper iff it stuck (the
        # probe in ``from_pretrained`` falls back to the eager model on failure).
        # The compiled fast path wants StaticCache so inductor traces one fixed
        # decode shape instead of re-specializing as a DynamicCache grows;
        # ``_static_cache_active`` gates that, device-agnostically (CUDA *or*
        # MPS), distinct from the CUDA-only ``_cuda_graphs_active`` graph-capture
        # signal.  Either one (or a future StaticCache-only opt-in) enables the
        # preallocated K/V path for fast-path-eligible steering.
        self._compiled: bool = hasattr(self._model, "_orig_mod")
        # CUDA graph trees are thread-local in PyTorch. ``from_pretrained``
        # warms them on this construction thread; synchronous ``generate`` can
        # replay them here, while ``generate_stream`` must route its worker to
        # eager execution. Non-graph compiled modes are cross-thread safe.
        self._compile_owner_thread: int = threading.get_ident()
        self._static_cache_active: bool = False
        if self._compiled or self._cuda_graphs_active:
            from saklas.core.cuda_graphs import (
                is_static_cache_supported, warn_once,
            )
            sc_supported, sc_reason = is_static_cache_supported(
                self._model, self._device,
            )
            if sc_supported:
                self._static_cache_active = True
            elif sc_reason is not None:
                warn_once(self._model, sc_reason)
        # The LIFO steering stack and push/pop-frequency steering methods live
        # on the eagerly initialized ``SteeringComposer`` collaborator.

        # Set by ``SteeringComposer.install_composed_steering`` when the current
        # steering lowers to the persistent compile-clean offset buffers
        # (static-affine push, compiled CUDA/MPS session) instead of transient hooks —
        # routes the decode loop to the compiled module + StaticCache.
        self._steering_uses_compiled_offsets: bool = False

        # Persistent compile-clean *capture* buffers + hook handles, adopted from
        # ``from_pretrained`` when compile stuck (slice 2).  The always-on hooks
        # ``copy_`` each layer's last-token slice into ``_capture_buffers[L]``
        # every forward (fused into the compiled graph); a probed gen on the
        # compiled path reads them post-forward via ``HiddenCapture.ingest_persistent``
        # instead of registering transient capture hooks that would graph-break.
        # Empty unless a compiled CUDA/MPS session adopted them.
        self._capture_buffers: dict[int, torch.Tensor] = {}
        self._capture_handles: list[Any] = []
        # Per-gen flags: ``_compiled_clean_eligible`` (set early in
        # ``_generate_core``) means this gen *can* take the compiled clean path —
        # compiled CUDA/MPS, static cache, capture buffers present, not return_hidden —
        # provided steering also lowers to offsets.  ``_capture_state.persistent``
        # (set in ``_begin_capture``) means the active capture rides the persistent
        # buffers, so the decode loop wires ``ingest_persistent`` as its step
        # callback and the routing keeps the compiled module.
        self._compiled_clean_eligible: bool = False

        # One session-resident StaticCache for ordinary (non-prefix-cache)
        # generations.  CUDA graphs guard on the K/V tensors' identities, so a
        # fresh cache per request forces a full Dynamo recompile even when every
        # shape is unchanged.  Grow this cache only when needed, then reset and
        # reuse it in place across serialized session generations.
        self._generation_static_cache: object | None = None
        self._generation_static_cache_len: int = 0

        # Active assistant-role label for the current ``session.steering()``
        # scope — populated when every role-tagged term in the resolved
        # expression agrees on a role.  ``None`` means "use the family's
        # standard assistant label", the standard zero-overhead path.
        # Push/save/restore is handled by ``_SteeringContext`` so nested
        # scopes inner-wins for the duration of the inner block.  The
        # generation surface reads this when assembling the chat-template
        # input so the assistant turn opens with ``<role>`` instead of
        # ``assistant``.
        self._active_role: str | None = None

        # Scene grammar (the cast model's stitcher): lazily autopsied +
        # round-trip-validated against the live chat template on first
        # access.  ``None`` after resolution = scene mode unavailable
        # (base model, label-free family, or validation failure) — every
        # render falls back to the standard chat-template paths.
        self._scene_grammar: TurnGrammar | None = None
        self._scene_grammar_resolved: bool = False

        # Synchronous event bus.  Emits on extraction, steering enter/exit,
        # probe scoring, generation start/finish.  Subscribers run on the
        # emit thread — async consumers must hop via call_soon_threadsafe
        # inside their callback.
        self.events: EventBus = EventBus()

        # Transient per-token hidden-state capture — attached around
        # generate_steered when probes are active so scoring happens
        # without a second forward pass.
        self._capture = HiddenCapture()
        # Per-gen capture configuration — one :class:`CaptureMode` plus the
        # orthogonal ``persistent`` flag (and the gated subset / keys for
        # ``GATING_SUBSET``).  Replaces the former five correlated booleans
        # (``_capture_incremental`` / ``_capture_aggregate_only`` /
        # ``_capture_lean`` / ``_capture_gating_subset`` / ``_capture_persistent``)
        # that a hand-kept if/elif chain had to keep consistent; the enum makes
        # illegal combinations unrepresentable.  Set wholesale in
        # ``_begin_capture`` and reset on teardown; the ``_score_*`` dispatch keys
        # off ``mode``.  Each mode and its memory/scoring trade-off is documented
        # on :class:`CaptureMode`.
        self._capture_state: CaptureState = CaptureState()
        # Per-token reading buffers the modes append into (the *data*, distinct
        # from the *config* above).  ``_incremental_readings`` holds the full /
        # lean per-token ``ProbeReading`` rows (INCREMENTAL / LEAN_INCREMENTAL);
        # ``_incremental_gate_scores`` holds the gated-subset scalar dicts
        # (GATING_SUBSET).  ``_finalize_generation`` builds (aggregate, per_token)
        # from these instead of rescoring captured hidden states.  Reset at each
        # ``_begin_capture`` and on teardown.
        self._incremental_readings: list[dict[str, ProbeReading]] = []
        self._incremental_gate_scores: list[dict[str, float]] = []

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
        # ``_gen_lock`` doubles as the process-wide **exclusive-GPU** lock:
        # PyTorch's MPS backend serializes every device op through one global,
        # NOT-thread-safe command buffer, so two threads committing it at once
        # aborts the process ("commit an already committed command buffer").
        # Every model op (fit / extract / generate) already holds it for its
        # duration; read-side GPU work (the analytics CPU snapshot below,
        # ``add_probe``) acquires it non-blocking and defers rather than race.
        # Read-side analytics (correlation / pairwise) snapshot each steering
        # direction to CPU once and serve every poll from this cache, so the
        # hot polling path never issues an MPS->CPU copy on a threadpool
        # thread (which would race a concurrent model op and abort).  See
        # ``analytics_profile``; cleared on any registry/manifold change.
        self._analytics_cpu_cache: dict[str, "Profile"] = {}
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

        # Conversation state lives in a :class:`LoomTree`. The active path
        # through the tree is what the model sees as context. Generation routes
        # through ``tree.add_user_turn``
        # / ``tree.begin_assistant`` / ``tree.finalize_assistant``.  The
        # tree is in-memory only — there is no automatic cross-session
        # persistence. ``LoomTree.save`` / ``LoomTree.load`` are the explicit
        # save/restore path.
        self.tree = LoomTree(
            events=self.events,
            model_id=self._model_info["model_id"],
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
        self._last_token_probe_payload: dict[str, Any] | None = None
        # The merged per-family :class:`ProbeReading` dict for the latest token
        # (the union of all three families' slots).  The measurement envelope
        # on ``_last_token_probe_payload`` carries the serialized readings; this
        # keeps the live objects so ``generate_stream`` can populate
        # ``TokenEvent.probe_readings`` (the compat vendor-extension / traits
        # channel) without reconstructing them from the envelope.
        self._last_token_probe_readings: dict[str, "ProbeReading"] | None = None

        # Probe content-hash cache for transcript export / replay (v2.3
        # phase 5).  Keyed by probe name → sha256 hex of the baked tensor
        # bytes (concatenated layer order).  Invalidated by
        # :meth:`add_probe` / :meth:`remove_probe`; rebuilt lazily by
        # :meth:`_probe_hash`.
        self._probe_hash_cache: dict[str, str] = {}

        # Live trait SSE subscribers.  Each entry is (event_loop, asyncio.Queue).
        # The generation thread pushes tagged tuples via loop.call_soon_threadsafe;
        # SSE handlers drain the queue asynchronously.
        self._trait_queues: list[tuple[Any, ...]] = []
        self._trait_lock = threading.Lock()

        # Ensure bundled concepts are materialized in the user cache and
        # the selector cache reflects them.  ``_bootstrap_manifold_probes``
        # does this transitively via ``load_default_manifolds``, but is
        # skipped entirely when
        # ``probes=[]`` — leaving freshly-added bundled concepts (e.g. via
        # updating package-data bundled manifolds) invisible to the selector
        # layer for the rest of the session.  Calling explicitly here keeps
        # the invariant intact regardless of probe-loading config; the call
        # is cheap when up-to-date (format-version short-circuit).  Bundled
        # concepts and manifolds (e.g. ``happy.sad``, ``personas``) all
        # materialize in the same pre-invalidate window so the bare-name
        # resolver picks up every bundled node label.
        from saklas.io.manifolds import (
            materialize_bundled_manifolds as _materialize_bundled_manifolds,
        )
        from saklas.io.templates import (
            materialize_bundled_templates as _materialize_bundled_templates,
        )
        # Templates first: a bundled manifold may ``template_ref`` a bundled
        # ``default/<name>`` template, and its fit resolves that ref.
        _materialize_bundled_templates()
        _materialize_bundled_manifolds()
        _selectors.invalidate()

        # Bootstrap probes
        probe_categories = PROBE_CATEGORIES if probes is None else probes

        # Order matters: layer_means + neutral_activations + whitener must
        # exist BEFORE ``_bootstrap_manifold_probes`` runs, because the
        # extraction pipeline uses the whitener for Mahalanobis-flavored
        # share allocation — the probe fold has an extract-time dependency
        # on the activation covariance, so the centering means and whitener
        # are built first.  When
        # ``probe_categories`` is empty there's nothing to extract, so we
        # skip the whitener build to keep ``probes=[]`` sessions cheap;
        # ad-hoc later extraction lazily builds via ``self.whitener``.
        self._layer_means: dict[int, torch.Tensor] = {}
        self._whitener: Any = None
        self._jlens: Any = None  # lazy per-model Jacobian lens (io/lens.py)
        self._jlens_identity: tuple[Any, ...] | None = None
        self._jlens_device_cache: dict[tuple[int, str, tuple[int, ...]], torch.Tensor] = {}
        self._jlens_readout_module_cache: (
            tuple[torch.Tensor, torch.nn.Module] | None
        ) = None
        self._jlens_depths_cache: dict[tuple[int, ...], list[float]] = {}
        self._jlens_depth_tensor_cache: dict[
            tuple[str, tuple[int, ...]], torch.Tensor
        ] = {}
        self._readout_long_tensor_cache: dict[
            tuple[str, tuple[int, ...]], torch.Tensor
        ] = {}
        self._jlens_decode_cache: dict[int, str] = {}
        # The J-lens read family lives on its instrument (probe registry,
        # live-readout runtime, per-forward stash, per-generation
        # disk-identity pin); the session re-exposes the state fields under
        # their historical private names via delegating properties, and the
        # instrument itself is created lazily on first touch (the
        # ``_lens_instrument`` property) so narrow test stubs that skip
        # ``__init__`` self-heal.  Touch it here so the real construction
        # path is eager and deterministic.
        _ = self._lens_instrument
        # Live sparse-autoencoder runtime. One source + one hook layer is
        # resident in v1; provider weights stay in SAELens/HF caches while
        # Saklas-trained weights live under SAKLAS_HOME. Readout probes are
        # session-local siblings of the J-lens probe registry.
        self._sae_backend: Any = None
        self._sae_layer: int | None = None
        self._sae_width: int | None = None
        # Per-feature Neuronpedia metadata: ``{str(id): {label, max_act}}``.
        # ``max_act`` (maxActApprox — the corpus-max activation) normalizes the
        # readout channel to a 0..1 strength; entries are lazily fetched.
        self._sae_feature_meta: dict[str, dict[str, Any]] = {}
        # The SAE read family's probe/live state lives on its instrument
        # (lazy property, like the lens family); backend residency
        # (`_sae_backend`/`_sae_layer`/`_sae_width`) and the Neuronpedia
        # metadata cache stay session-side (shared with steering atoms and
        # the offline token replay).
        _ = self._sae_instrument
        # CAA live toggle: when False, per-token monitor scoring is disabled
        # for UI/trait/loom consumers (aggregate-only capture); probe gates
        # still force the per-token subset they need.
        self._live_probe_scores: bool = True
        if probe_categories:
            self._whitener = self._build_whitener_from_cache_or_compute()

        # DLS toggle stored on the session so ad-hoc ``session.extract``
        # calls (via ``ExtractionPipeline``) inherit it without re-passing.
        self._dls: bool = bool(dls)

        # Steering-resolution collaborator is required while bootstrapping
        # manifold probes: the loader is the single artifact registration
        # authority. Construction itself only captures the session reference;
        # push/pop paths that consult the monitor run after initialization.
        from saklas.core.steering_composer import SteeringComposer
        self._steering_composer: SteeringComposer = SteeringComposer(self)

        probe_manifolds: dict[str, "Manifold"] = {}
        if probe_categories:
            # Every concept lives as a 2-node ``pca`` manifold (4.0): fit-or-
            # load each tagged manifold and hand the flat ``Manifold`` to the
            # Monitor, which reads it as a coordinate (the rank-1 case of the
            # subspace readout).  No folded-direction probe path anymore.
            # ``probes is None`` is the default-roster signal: beyond the
            # tagged concept axes, also attach every already-fitted bundled
            # multi-node manifold (``personas`` / ``emotions``).  An explicit
            # category list (``probes=[...]``) is honored exactly — no sweep.
            probe_manifolds = self._bootstrap_manifold_probes(
                probe_categories, include_fitted_defaults=probes is None,
            )

        # One unified Monitor for every probe shape — flat concept axes
        # (rank-1), flat discover fits (rank-R, e.g. ``personas``), and curved
        # manifolds (``emotions``).  There is no batched-affine fast path: every
        # probe — flat or curved — is read per token through one whitened
        # per-layer geometry pass (the research-tool priority is full per-token
        # info — nearest, coords, residual, per-layer — over throughput).
        # Per-token score callbacks emit one flat-scalar dict into
        # ``TriggerContext.probe_scores`` so ``@when:`` gates fire on any probe
        # without grammar changes.
        self._monitor = Monitor(
            probe_manifolds, self._layer_means, whitener=self._whitener,
            n_layers=len(self._layers),
        )

        # Prefix KV cache (opt-in, off by default).  Populated by
        # :meth:`cache_prefix`; consumed by :meth:`_generate_core` when the
        # incoming ``input_ids`` start with the cached prefix.  Shape:
        # ``_PrefixCacheEntry(prefix_token_ids_cpu, past_key_values,
        # prefix_len, static, max_cache_len)``.  ``past_key_values`` is the
        # live HF cache returned/mutated by the prefix-prefill forward pass.
        # Dynamic entries crop back to ``prefix_len`` after reuse; StaticCache
        # entries additionally carry capacity so oversized generations miss
        # safely instead of writing past the preallocated buffers.  Invalidated
        # on any state change that would alter the cached prefix's hidden-state
        # semantics: steering push/pop/steer/unsteer, probe install/remove,
        # profile mutation.
        self._prefix_cache: _PrefixCacheEntry | None = None

        # An explicit ``sae use`` selection persists across sessions. Provider
        # payloads are expected to be present in their own cache already;
        # failure is non-fatal so an offline cache eviction cannot prevent the
        # model itself from starting.
        from saklas.io.sae import load_active_sae_source, load_sae_metadata

        active_sae = load_active_sae_source(self.model_id)
        if active_sae is not None:
            release = (
                f"local:{active_sae['name']}"
                if active_sae["kind"] == "local"
                else active_sae["name"]
            )
            # Provider bindings remember the last explicitly selected hook
            # layer. Restore that exact measurement surface rather than
            # silently returning to the release's automatic default.
            metadata = (
                load_sae_metadata(self.model_id, release)
                if active_sae["kind"] == "saelens"
                else None
            )
            layer = metadata.get("layer") if metadata is not None else None
            try:
                self.load_sae(release, layer=layer)
            except Exception as exc:
                _log.warning(
                    "could not restore active SAE %s for %s: %s",
                    release, self.model_id, exc,
                )

    # -- ModelHandle protocol surface (consumed by ManifoldExtractionPipeline) --

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
        consumers (the extraction pipeline and hooks).
        """
        return self._layers

    # -- State queries --

    @property
    def model_info(self) -> dict[str, Any]:
        return dict(self._model_info)

    @property
    def model_id(self) -> str:
        return self._model_info.get("model_id", "unknown")

    def has_profile(self, name: str) -> bool:
        return name in self._profiles

    def analytics_names(self) -> list[str]:
        """Names available to read-side analytics: registered steering
        vectors plus attached probes (a vector wins a same-named probe,
        matching the correlation pool's dedup).  Pure metadata — no GPU.

        A multi-node / curved probe (a rank-``R`` fit like ``personas`` or
        ``emotions``) is **excluded** — the direction-cosine analytics fold every
        name to a single steering direction, which only a folded affine
        ``R = 1`` manifold has.  Listing one would only render an all-null
        row in the correlation matrix (or a misleading miss in ``pairwise``)
        and, before the guard below, made the fold raise a 500."""
        from saklas.core.capture import is_foldable_vector_manifold

        names = set(self._profiles)  # registered vectors are always R=1
        # One locked roster snapshot — this serves un-locked analytics
        # polling, where a raw per-name ``manifolds`` rebuild tears under
        # a concurrent detach.
        for pname, manifold in self._geometry_instrument.manifolds().items():
            if pname in names:
                continue  # a registered vector wins the same-named probe
            if is_foldable_vector_manifold(manifold):
                names.add(pname)
        return sorted(names)

    def _live_direction_tensors(self, name: str) -> "dict[int, torch.Tensor] | None":
        """Live per-layer direction tensors for *name* (may be device-
        resident — callers must hold ``_gen_lock``).  A registered steering
        vector wins over a same-named probe; otherwise the folded view of an
        attached probe's manifold.

        Returns ``None`` for a multi-node / curved probe — it has no single
        direction to fold, so the direction analytics skip it rather than
        crashing on ``folded_directions``."""
        prof = self._profiles.get(name)
        if prof is not None:
            return dict(prof)
        manifold = self._monitor.manifolds.get(name)
        if manifold is not None:
            from saklas.core.capture import (
                folded_directions,
                is_foldable_vector_manifold,
            )
            if is_foldable_vector_manifold(manifold):
                return folded_directions(manifold)
        return None

    def analytics_profile(self, name: str) -> "Profile | None":
        """CPU-resident snapshot of a steering vector / probe direction for
        read-side analytics (correlation, pairwise), cached.

        PyTorch's MPS backend serializes every device op through one global,
        non-thread-safe command buffer.  A read endpoint evacuating an
        MPS-resident direction to CPU on its own threadpool thread races a
        concurrent model op's command-buffer commit and aborts the process
        (``IOGPUMetalCommandBuffer validate: commit an already committed
        command buffer``).  This snapshots each direction to CPU **once,
        under ``_gen_lock``** (the exclusive-GPU lock every fit / extract /
        generation holds), then serves every later poll from the CPU cache —
        so the hot polling path never touches the GPU.

        Returns ``None`` when *name* isn't registered, or when a model op
        currently holds the GPU and no snapshot is cached yet; the caller
        renders that cell as null and a later poll builds it once the GPU is
        free.
        """
        cached = self._analytics_cpu_cache.get(name)
        if cached is not None:
            return cached
        # The source fold + device->host copy is GPU work: run it under the
        # exclusive-GPU lock so it can't overlap a model op.  Non-blocking —
        # defer to a later poll rather than stall or race while one runs.
        if not self._gen_lock.acquire(blocking=False):
            return None
        try:
            live = self._live_direction_tensors(name)
            if live is None:
                return None
            snap = Profile({
                int(L): t.detach().float().cpu() for L, t in live.items()
            })
        finally:
            self._gen_lock.release()
        self._analytics_cpu_cache[name] = snap
        return snap

    def _invalidate_analytics_cache(self) -> None:
        """Drop the read-side analytics CPU snapshots (rebuilt lazily on the
        next poll).  Called on any change to the steering-vector / probe
        registry or the fitted manifolds those directions read from."""
        self._analytics_cpu_cache.clear()

    @property
    def probes(self) -> dict[str, dict[str, Any]]:
        # All three families via the instruments' LOCKED snapshots: this
        # property serves the un-locked session-info route, and a raw
        # registry iteration tears under a concurrent detach (round-7 for
        # lens/SAE; the geometry state_lock closed the Monitor sibling).
        out: dict[str, dict[str, Any]] = {
            name: {"manifold": m}
            for name, m in self._geometry_instrument.manifolds().items()
        }
        out.update(
            {name: {"lens": spec}
             for name, spec in self._lens_instrument.specs().items()}
        )
        out.update(
            {name: {"sae": spec}
             for name, spec in self._sae_instrument.specs().items()}
        )
        return out

    @property
    def last_result(self) -> GenerationResult | None:
        return self._last_result

    @property
    def last_per_token_scores(self) -> dict[str, list[float]] | None:
        return self._last_per_token_scores

    @property
    def monitor(self) -> Monitor:
        """The unified read-side :class:`Monitor` (probe roster + scoring).

        Frontends read ``monitor.probe_names`` / ``monitor.manifolds`` to
        enumerate the attached probes without reaching past the public API.
        """
        return self._monitor

    @property
    def manifolds(self) -> dict[str, Manifold]:
        """Loaded-manifold registry: registry key (``[ns/]name[:variant]``) ->
        :class:`Manifold`.  The live dict — mutating it mutates the session
        registry (the CLI/server prune fitted entries through it)."""
        return self._manifolds

    @property
    def profiles(self) -> dict[str, dict[int, torch.Tensor]]:
        """In-memory baked steering-direction registry: name -> per-layer
        tensors.  The canonical steering-profile registry — the live dict,
        so mutating it mutates the registry; wrap an entry in
        :class:`Profile` for the read-only public view."""
        return self._profiles

    @property
    def generation_state(self) -> GenerationState:
        """The current per-call :class:`GenerationState` (finish reason, emit
        map, thinking spans, stop event).

        Distinct from :attr:`gen_state`, which reports the coarse lifecycle
        *phase* (``IDLE``/``RUNNING``/…); this is the live mutable streaming
        state the server reads while a generation is in flight.
        """
        return self._gen_state

    @property
    def token_probe_payload(self) -> dict[str, Any]:
        """Latest token-tap payload for live frontend serialization.

        The token tap is the sole producer. Frontends consume this current
        step snapshot instead of reconstructing it from persisted loom rows.
        """
        return self._last_token_probe_payload or {}

    @property
    def gen_lock(self) -> "threading.RLock":
        """The exclusive-GPU re-entry lock.  Callers that need to probe
        whether a generation is in flight without blocking use
        ``gen_lock.acquire(blocking=False)`` / ``release()``."""
        return self._gen_lock

    @property
    def joint_logprob_cache(self) -> dict[tuple[str, str], Any]:
        """Cross-branch joint-logprob cache (loom NodeCompareDrawer).

        Read/write — assigning a fresh cache object is supported (the native
        API swaps it to bound memory)."""
        return self._joint_logprob_cache

    @joint_logprob_cache.setter
    def joint_logprob_cache(self, cache: dict[tuple[str, str], Any]) -> None:
        self._joint_logprob_cache = cache

    @property
    def loom_conflict_check(self) -> "Callable[[str, str], None]":
        """The bound tree-mutation conflict checker, for handing to a freshly
        loaded :class:`LoomTree` via ``set_conflict_check`` — raises
        :class:`MutationDuringGenerationError` on a conflicting op."""
        return self._loom_conflict_check

    def ensure_manifold_loaded(self, key: str) -> None:
        """Load the manifold artifact for registry key ``key`` if absent.

        ``key`` is the grammar's ``[ns/]name[:variant]`` registry key.
        Raises :class:`ManifoldNotRegisteredError` on a miss."""
        self._steering_composer.ensure_manifold_loaded(key)

    def ensure_profile_registered(
        self, name: str, *, role: str = "vector",
    ) -> dict[int, torch.Tensor]:
        """Resolve and register a steering direction for ``name``, returning
        its per-layer tensors.

        Uses the manifold-first resolution chain: in-memory bake, then a fitted
        2-node ``pca`` manifold."""
        return self._steering_composer.ensure_profile_registered(name, role=role)

    @property
    def gen_state(self) -> GenState:
        """Lifecycle phase of the current generation (``IDLE`` between gens).

        Read-only window into the session's typed re-entry guard — see
        :class:`GenState` for transitions. Surfaces to any external
        introspector that wants to ask "is a gen running right
        now?" without reaching past the public API.
        """
        return self._gen_phase

    @property
    def is_generating(self) -> bool:
        """``True`` whenever :attr:`gen_state` is not ``GenState.IDLE``."""
        return self._gen_phase is not GenState.IDLE

    @property
    def is_base_model(self) -> bool:
        """``True`` when the loaded model has no chat template (a base model).

        Frontends branch on this to switch to a raw-completion UI — no
        roles, no chat bubbles, the whole buffer is a prefill.
        """
        return detect_base_model(self._tokenizer)

    # -- Live trait SSE subscribers --

    def register_trait_queue(self, loop: Any, q: Any) -> None:
        """Register an ``(event_loop, asyncio.Queue)`` pair for live trait events."""
        with self._trait_lock:
            self._trait_queues.append((loop, q))

    def unregister_trait_queue(self, loop: Any, q: Any) -> None:
        """Remove a previously registered trait queue."""
        with self._trait_lock, suppress(ValueError):
            self._trait_queues.remove((loop, q))

    # -- Neutral baseline (v2.1) --

    @property
    def layer_means(self) -> dict[int, torch.Tensor]:
        """Per-layer neutral baseline means, built lazily on first access.

        Sessions instantiated with ``probes=[]`` skip the eager
        :func:`bootstrap_layer_means` call to keep init cheap.  Callers
        that later need the means — DLS centering at extraction time,
        the Mahalanobis whitener, the trait monitor — hit this
        property, which triggers the bootstrap once and caches the
        result on ``self._layer_means``. The means are derived from the single
        fp32 ``neutral_activations`` cache, whose loaded-model identity, exact
        rendered token rows, layer schema, and payload digest must all match;
        there is no separate layer-means artifact.

        Returns ``{}`` only if the bootstrap path itself fails (model
        not loaded, missing neutrals pack, etc.) — DLS / whitener
        callers fall back to no-baseline behavior in that case.

        v2.1 fix-up: previously DLS extraction read ``self._layer_means``
        directly, which left ``probes=[]`` sessions with an empty dict
        and silently disabled DLS (every layer fell through the
        "missing baseline" conservative-keep branch in
        :func:`compute_dls_axes`).  The property closes that footgun.
        """
        if not self._layer_means:
            with self._model_exclusive(
                "layer_means requested while another model use is in flight",
                phase_msg="layer_means requested while generation is in flight",
            ):
                SaklasSession._assert_unsteered_artifact_operation(self)
                # Double-check after acquiring: another caller may have built
                # the neutral artifact while this thread waited.
                if not self._layer_means:
                    try:
                        means = bootstrap_layer_means(
                            self._model, self._tokenizer, self._layers,
                            self._model_info,
                        )
                    except Exception as exc:
                        raise SaklasError(
                            "failed to build neutral layer means"
                        ) from exc
                    if not means:
                        raise SaklasError("neutral layer-mean bootstrap returned empty")
                    self._layer_means = means
        return self._layer_means

    # -- Mahalanobis whitener (v2.1) --

    @property
    def whitener(self) -> "Any":
        """Per-layer Mahalanobis whitener; built lazily on first access.

        Used by v2.1+ DiM extraction for Mahalanobis-flavored share
        allocation, by ``manifold compare``, and by
        callers that pass a whitener to ``project_profile`` for
        LEACE-style projection.  Returns a
        :class:`saklas.core.mahalanobis.LayerWhitener` or ``None`` if
        construction failed (model is mid-load, neutral activations
        couldn't be computed, etc. — we never raise here, so probe
        scoring stays alive).
        """
        if self._whitener is None:
            with self._model_exclusive(
                "whitener requested while another model use is in flight",
                phase_msg="whitener requested while generation is in flight",
            ):
                SaklasSession._assert_unsteered_artifact_operation(self)
                self._install_whitener_if_missing()
        return self._whitener

    def _install_whitener_if_missing(self) -> None:
        """Build and publish the lazy whitener while model use is exclusive.

        The public :attr:`whitener` property acquires that exclusivity itself.
        Generation already owns ``_gen_lock`` before it enters PREAMBLE, so its
        steering preflight calls this helper directly: re-entering the public
        property there would correctly see a non-idle generation phase and
        reject the request even though the current generation is the lock owner.
        """
        if self._whitener is not None:
            return
        self._whitener = self._build_whitener_from_cache_or_compute()
        # Keep the trait monitor's read metric in lock-step with the session
        # whitener when it is built lazily.  ``set_whitener`` rebuilds every
        # attached probe's whitened factors in place — a roster mutation, so
        # it lands under the geometry-state boundary like attach/detach.
        monitor = getattr(self, "_monitor", None)
        if monitor is not None and self._whitener is not None:
            with self._geometry_instrument.state_lock:
                monitor.set_whitener(self._whitener)

    def _build_whitener_from_cache_or_compute(self) -> "Any":
        """Compute or load the per-model whitener.

        Uses ``load_or_compute_neutral_activations`` (alignment.py) for
        disk caching; derives missing centering means from that same loaded
        tensor set, then instantiates the :class:`LayerWhitener`.  The shared
        load matters on a cold model: probe bootstrap no longer reads the
        neutral cache once for means and immediately again for covariance.
        Soft-fails to ``None``
        on any error — but the engine is Mahalanobis-only now, so a
        ``None`` whitener makes the activation-space consumers (fit,
        ``~``/``|`` projection, probe scoring, ``manifold compare``) raise
        ``WhitenerError`` with a regenerate-neutrals hint rather than
        degrade to Euclidean.

        Lazy: only callers who actually need Mahalanobis math (DiM
        extraction at session init, on-demand ``session.whitener``
        access, or ``manifold compare``) trigger the
        forward-pass loop over neutral statements.
        """
        from saklas.core.mahalanobis import LayerWhitener
        from saklas.io.alignment import load_or_compute_neutral_activations

        try:
            neutral_acts = load_or_compute_neutral_activations(
                self._model, self._tokenizer, self._layers,
                model_id=self._model_info.get("model_id", "unknown"),
            )
            if not self._layer_means:
                self._layer_means = {
                    idx: activations.mean(dim=0)
                    for idx, activations in neutral_acts.items()
                }
            return LayerWhitener.from_neutral_activations(
                neutral_acts, self._layer_means,
            )
        except Exception as exc:
            _log.warning(
                "whitener: build failed (%s); Mahalanobis-only fitting, "
                "projection, and probe reads are unavailable. Error: %s",
                type(exc).__name__, exc,
            )
            return None

    # -- Jacobian lens (verbalizable-workspace readout) --

    @property
    def jlens(self) -> "Any":
        """The model's fitted Jacobian lens, or ``None`` when not fitted.

        Loaded lazily from the active local/external source. Local fits use
        immutable Saklas-owned shards; external bindings read provider-owned
        cache bytes. Fit or fetch one with ``saklas lens fit|fetch``. Returns a
        :class:`saklas.core.jlens.JacobianLens`.

        The refresh/adopt/evict transaction runs under the lens-state
        lock (``LensInstrument.state_lock``) — the same boundary
        ``prepare``'s snapshot holds — so an un-locked read (this getter
        serves the session-info route) cannot tear a generation
        boundary's snapshot mid-``prepare``.
        """
        with self._lens_instrument.state_lock:
            if self._generation_jlens_active:
                return self._generation_jlens

            from saklas.io.lens import load_lens, load_lens_sidecar

            sidecar = load_lens_sidecar(self.model_id)
            if sidecar is None:
                if self._jlens is not None:
                    SaklasSession._evict_resident_jlens(self)
                return None
            if not _jlens_matches_loaded_model(
                sidecar, self._model, self.model_id,
            ):
                if self._jlens is not None:
                    SaklasSession._evict_resident_jlens(self)
                _log.warning(
                    "ignoring stale Jacobian lens for %s: loaded-model "
                    "identity is missing or changed; fetch or re-fit the "
                    "lens",
                    self.model_id,
                )
                return None
            disk_identity = _jlens_sidecar_identity(sidecar)
            if (
                self._jlens is not None
                and self._jlens_identity == disk_identity
            ):
                return self._jlens
            loaded = load_lens(self.model_id)
            if loaded is None:
                SaklasSession._evict_resident_jlens(self)
                return None
            if not _jlens_matches_loaded_model(
                loaded[1], self._model, self.model_id,
            ):
                SaklasSession._evict_resident_jlens(self)
                _log.warning(
                    "ignoring concurrently replaced Jacobian lens for %s: "
                    "loaded payload does not match the live model",
                    self.model_id,
                )
                return None
            SaklasSession._adopt_fitted_jlens(
                self, loaded[0], sidecar=loaded[1],
            )
            return self._jlens

    def has_compatible_jlens(self) -> bool:
        """Whether lens metadata matches the currently loaded weights.

        Runs un-locked on the session-info route, so its adopt/evict
        side effects serialize on the lens-state lock like the getter's.
        """
        with self._lens_instrument.state_lock:
            if self._generation_jlens_active:
                return self._generation_jlens is not None

            from saklas.io.lens import load_lens, load_lens_sidecar

            sidecar = load_lens_sidecar(self.model_id)
            compatible = _jlens_matches_loaded_model(
                sidecar, self._model, self.model_id,
            )
            if not compatible:
                if self._jlens is not None:
                    SaklasSession._evict_resident_jlens(self)
                return False
            assert sidecar is not None
            if (
                self._jlens is not None
                and self._jlens_identity != _jlens_sidecar_identity(sidecar)
            ):
                loaded = load_lens(self.model_id)
                if loaded is None:
                    SaklasSession._evict_resident_jlens(self)
                    return False
                if not _jlens_matches_loaded_model(
                    loaded[1], self._model, self.model_id,
                ):
                    SaklasSession._evict_resident_jlens(self)
                    return False
                SaklasSession._adopt_fitted_jlens(
                    self, loaded[0], sidecar=loaded[1],
                )
            return True

    def _require_jlens(self) -> "Any":
        from saklas.core.jlens import LensNotFittedError

        lens = self.jlens
        if lens is None:
            raise LensNotFittedError(
                f"no Jacobian lens available for {self.model_id} — run "
                f"`saklas lens fetch {self.model_id}` for a supported official "
                f"lens, or `saklas lens fit {self.model_id}`"
            )
        return lens

    def _evict_resident_jlens(self) -> None:
        """Drop a removed/replaced disk lens and every derived resident view."""
        with self._lens_instrument.state_lock:
            self._jlens = None
            self._jlens_identity = None
            self._jlens_device_cache = {}
            self._jlens_readout_module_cache = None
            self._jlens_depths_cache = {}
            self._jlens_depth_tensor_cache = {}
            self._readout_long_tensor_cache = {}
            self._jlens_decode_cache = {}
            self._lens_step_stash = None
            self._last_lens_step_readings = None
            for key in list(self._profiles):
                if key.startswith("jlens/"):
                    self._profiles.pop(key, None)
            for name, spec in self._lens_probes.items():
                spec["layers"] = []
                self._probe_hash_cache.pop(name, None)
            self._live_lens = None
            self._invalidate_prefix_cache()
            self._invalidate_analytics_cache()

    def select_jlens_source(self, source: str) -> None:
        """Select a prepared local/external lens and evict derived live state.

        Source preparation is deliberately separate: callers fit or fetch
        first, then use this exact source identifier to switch the session.
        """
        from saklas.io.lens_sources import use_lens_source

        use_lens_source(self.model_id, source)
        self._evict_resident_jlens()

    def _adopt_fitted_jlens(
        self, lens: "Any", *, sidecar: "Mapping[str, Any] | None" = None,
    ) -> "Any":
        """Replace a fitted lens and rebuild every derived live consumer.

        One lens-state transaction: the registry rewrite and the
        live-state rebuild land atomically under the lens-state lock, so
        a concurrent ``prepare`` snapshot sees either the old or the new
        world, never a mix.
        """
        with self._lens_instrument.state_lock:
            previous_live = self._live_lens
            previous_layers = (
                list(previous_live["layers"])
                if previous_live is not None else []
            )
            previous_used_all_layers = bool(
                previous_live is not None
                and previous_live.get("uses_all_layers", False)
            )

            self._jlens = lens
            self._jlens_identity = _jlens_sidecar_identity(sidecar)
            self._jlens_device_cache = {}
            self._jlens_readout_module_cache = None
            self._jlens_depths_cache = {}
            self._jlens_depth_tensor_cache = {}
            self._readout_long_tensor_cache = {}
            self._jlens_decode_cache = {}
            self._lens_step_stash = None
            self._last_lens_step_readings = None
            for key in list(self._profiles):
                if key.startswith("jlens/"):
                    self._profiles.pop(key, None)

            readout_layers = [int(layer) for layer in lens.source_layers]
            for name, spec in self._lens_probes.items():
                spec["layers"] = list(readout_layers)
                self._probe_hash_cache.pop(name, None)
            if self._lens_probes and readout_layers:
                self._jlens_transport_stack(
                    lens, sorted(readout_layers), self._device,
                )

            if previous_live is not None:
                valid = [
                    layer for layer in previous_layers
                    if layer in lens.jacobians
                ]
                self.enable_live_lens(
                    layers=(
                        None if previous_used_all_layers else (valid or None)
                    ),
                )
            else:
                self._live_lens = None

            self._invalidate_prefix_cache()
            self._invalidate_analytics_cache()
            return lens

    def fit_jlens(
        self,
        prompts: "Sequence[str]",
        *,
        corpus_spec: str = "custom",
        source_layers: "Sequence[int] | str | None" = None,
        dim_batch: int | None = None,
        prompt_batch: int | None = None,
        seq_len: int | None = None,
        force: bool = False,
        checkpoint_every: int | None = None,
        on_progress: Callable[[str], None] | None = None,
        cancel_event: "Any | None" = None,
    ) -> "Any":
        """Fit/resume one per-model J-lens under its cross-process transaction."""
        from saklas.io.lens import cleanup_lens_artifacts, lens_fit_lock

        with lens_fit_lock(self.model_id):
            cleanup_lens_artifacts(self.model_id)
            if cancel_event is not None and cancel_event.is_set():
                from saklas.core.jlens import JacobianLensCancelled

                raise JacobianLensCancelled(
                    "Jacobian-lens fit cancelled before start",
                )
            return SaklasSession._fit_jlens_transaction(
                self,
                prompts,
                corpus_spec=corpus_spec,
                source_layers=source_layers,
                dim_batch=dim_batch,
                prompt_batch=prompt_batch,
                seq_len=seq_len,
                force=force,
                checkpoint_every=checkpoint_every,
                on_progress=on_progress,
                cancel_event=cancel_event,
            )

    def _fit_jlens_transaction(
        self,
        prompts: "Sequence[str]",
        *,
        corpus_spec: str = "custom",
        source_layers: "Sequence[int] | str | None" = None,
        dim_batch: int | None = None,
        prompt_batch: int | None = None,
        seq_len: int | None = None,
        force: bool = False,
        checkpoint_every: int | None = None,
        on_progress: Callable[[str], None] | None = None,
        cancel_event: "Any | None" = None,
    ) -> "Any":
        """Fit (or resume fitting) this model's Jacobian lens and persist it.

        Prompts too short for the estimator (≤ ``SKIP_FIRST_POSITIONS + 1``
        tokens) are dropped up front so the saved ``n_prompts`` counts
        consumed prompts exactly — that is what makes resume slicing sound.
        Resume-by-default: when a saved lens matches the filtered token-id
        corpus (or an exact prefix of it), the loaded-model fingerprint, and
        sequence policy, only the remainder is fitted and merged in;
        ``force=True`` restarts from zero. Checkpoints every
        ``DEFAULT_CHECKPOINT_EVERY`` prompts, so an interrupted fit resumes from
        the last checkpoint. ``prompt_batch`` controls consecutive ragged
        prompts per autograd graph independently of the output ``dim_batch``.

        Gated against generation: the fit runs forward *and backward* passes
        through the model (the only backward passes in saklas).
        """
        import hashlib

        from saklas.core.jlens import (
            DEFAULT_CHECKPOINT_EVERY,
            DEFAULT_DIM_BATCH,
            DEFAULT_SEQ_LEN,
            SKIP_FIRST_POSITIONS,
            JacobianLens,
            JacobianLensCancelled,
            JacobianLensError,
            fit_jacobian_lens,
        )
        from saklas.io.lens import (
            LENS_FORMAT_VERSION,
            load_local_lens as load_lens,
            load_local_lens_sidecar as load_lens_sidecar,
            save_lens,
        )
        from saklas.io.lens import (
            _load_lens_verified,
            load_lens_checkpoint,
            load_lens_checkpoint_sidecar,
            promote_lens_checkpoint,
            remove_lens_checkpoint,
            remove_subsumed_lens_checkpoint,
            save_lens_checkpoint_accumulator,
        )

        dim_batch = dim_batch or DEFAULT_DIM_BATCH
        seq_len = seq_len or DEFAULT_SEQ_LEN
        checkpoint_every = checkpoint_every or DEFAULT_CHECKPOINT_EVERY
        if dim_batch <= 0:
            raise ValueError("dim_batch must be > 0")
        if prompt_batch is not None and prompt_batch <= 0:
            raise ValueError("prompt_batch must be > 0")
        if seq_len <= 0:
            raise ValueError("seq_len must be > 0")
        if checkpoint_every <= 0:
            raise ValueError("checkpoint_every must be > 0")
        prompt_list = list(prompts)
        raw_corpus_sha = hashlib.sha256(repr(prompt_list).encode("utf-8")).hexdigest()
        raw_prompt_count = len(prompt_list)
        if cancel_event is not None and cancel_event.is_set():
            raise JacobianLensCancelled("Jacobian-lens fit cancelled before start")

        with self._model_exclusive(
            "session.fit_jlens called while another model use is in flight",
            phase_msg="session.fit_jlens called while a generation is in flight",
        ):
            SaklasSession._assert_unsteered_artifact_operation(self)
            fit_model = getattr(self._model, "_orig_mod", self._model)
            fit_layers = list(get_layers(fit_model))
            model_fingerprint = loaded_model_fingerprint(
                fit_model, self.model_id,
            )
            model_source_fp = getattr(
                fit_model, "_saklas_source_fingerprint", None,
            )
            expected_sources = self._resolve_jlens_source_layers(
                source_layers, n_layers=len(fit_layers),
            )
            expected_set = set(expected_sources)
            if force:
                remove_lens_checkpoint(self.model_id)

            usable: list[str] = []
            consumed_ids: list[list[int]] = []
            for prompt in prompt_list:
                try:
                    ids = self._tokenizer(
                        prompt, return_tensors="pt", truncation=True,
                        max_length=seq_len,
                    )["input_ids"]
                except TypeError:
                    # Minimal/test tokenizers may not expose HF truncation kwargs.
                    ids = self._tokenizer(prompt, return_tensors="pt")[
                        "input_ids"
                    ][:, :seq_len]
                if ids.shape[1] > SKIP_FIRST_POSITIONS + 1:
                    usable.append(prompt)
                    consumed_ids.append([int(tok) for tok in ids[0].tolist()])
            if len(usable) < raw_prompt_count and on_progress is not None:
                on_progress(
                    f"dropped {raw_prompt_count - len(usable)} too-short prompts "
                    f"({len(usable)} usable)"
                )
            if not usable:
                raise JacobianLensError(
                    "no usable prompts: the Jacobian estimator needs prompts "
                    f"longer than {SKIP_FIRST_POSITIONS + 1} tokens"
                )
            corpus_sha = hashlib.sha256(
                repr(consumed_ids).encode("utf-8")
            ).hexdigest()
            all_consumed_ids = list(consumed_ids)

            def _token_rows_sha(rows: "Sequence[Sequence[int]]") -> str:
                return hashlib.sha256(repr(list(rows)).encode("utf-8")).hexdigest()

            def _checkpoint_progress(sidecar: "Mapping[str, Any] | None") -> int:
                if sidecar is None:
                    return -1
                try:
                    return int(sidecar.get("base_n_prompts", 0)) + int(
                        sidecar.get("n_prompts", -1)
                    )
                except (TypeError, ValueError):
                    return -1

            corpus_hash_kind = "token_ids_v1"
            base: Any = None
            checkpoint_sidecar = None
            checkpoint_meta_matches = False
            checkpoint_base_n = -1
            checkpoint_n = -1
            resident_evicted_early = False
            pre_evicted_live: dict[str, Any] | None = None
            durable_fallback_after_checkpoint = False
            existing_reuse_proof: Any = None
            if not force:
                checkpoint_sidecar = load_lens_checkpoint_sidecar(self.model_id)
                checkpoint_progress = _checkpoint_progress(checkpoint_sidecar)
                checkpoint_prefix_matches = bool(
                    checkpoint_sidecar is not None
                    and 0 < checkpoint_progress <= len(all_consumed_ids)
                    and checkpoint_sidecar.get("consumed_prefix_sha256")
                    == _token_rows_sha(all_consumed_ids[:checkpoint_progress])
                )
                checkpoint_meta_matches = bool(
                    checkpoint_sidecar is not None
                    and (
                        checkpoint_sidecar.get("corpus_sha256") == corpus_sha
                        or checkpoint_prefix_matches
                    )
                    and checkpoint_sidecar.get("corpus_hash_kind")
                    == corpus_hash_kind
                    and checkpoint_progress > 0
                    and checkpoint_sidecar.get("seq_len") == seq_len
                    and checkpoint_sidecar.get("model_fingerprint")
                    == model_fingerprint
                    and [
                        int(l)
                        for l in checkpoint_sidecar.get("source_layers", [])
                    ] == expected_sources
                )
                if checkpoint_meta_matches:
                    assert checkpoint_sidecar is not None
                    checkpoint_base_n = int(
                        checkpoint_sidecar.get("base_n_prompts", -1)
                    )
                    checkpoint_n = int(checkpoint_sidecar.get("n_prompts", -1))
                sidecar = load_lens_sidecar(self.model_id)
                saved_n = 0
                if (
                    sidecar is not None
                    and sidecar.get("corpus_hash_kind") == corpus_hash_kind
                    and sidecar.get("seq_len") == seq_len
                    and sidecar.get("model_fingerprint") == model_fingerprint
                ):
                    saved_n = int(sidecar.get("n_prompts", 0))
                    saved_sha = sidecar.get("corpus_sha256")
                    corpus_matches = saved_sha == corpus_sha
                    prefix_matches = (
                        0 < saved_n <= len(consumed_ids)
                        and saved_sha == _token_rows_sha(consumed_ids[:saved_n])
                    )
                    if corpus_matches or prefix_matches:
                        durable_sources = {
                            int(layer) for layer in sidecar.get("source_layers", [])
                        }
                        if (
                            saved_n < len(usable)
                            and bool(durable_sources - expected_set)
                        ):
                            # v4 has one corpus/progress identity shared by every
                            # layer. Publishing only the requested extension would
                            # silently discard the other durable layers; carrying
                            # them forward would falsely claim they were averaged
                            # over the longer corpus. Keep the proven superset
                            # untouched unless replacement is explicit.
                            raise JacobianLensError(
                                "cannot extend only a subset of an existing "
                                f"J-lens: durable layers {sorted(durable_sources)} "
                                f"include unrequested layers "
                                f"{sorted(durable_sources - expected_set)}. "
                                "Request the full durable layer set to extend "
                                "its corpus, or pass force=True to explicitly "
                                "replace it with the requested subset."
                            )
                        prefer_checkpoint = bool(
                            checkpoint_meta_matches
                            and checkpoint_base_n == 0
                            and checkpoint_n > saved_n
                        )
                        existing_payload_verified = False
                        resident = self._jlens
                        resident_layers = (
                            set(resident.source_layers)
                            if resident is not None else set()
                        )
                        if prefer_checkpoint:
                            durable_fallback_after_checkpoint = True
                            if resident is not None:
                                pre_evicted_live = (
                                    {
                                        "layers": list(self._live_lens["layers"]),
                                        "uses_all_layers": bool(
                                            self._live_lens.get(
                                                "uses_all_layers", False,
                                            )
                                        ),
                                    }
                                    if self._live_lens is not None else None
                                )
                                SaklasSession._evict_resident_jlens(self)
                                resident = None
                                resident_evicted_early = True
                            existing = None
                        elif (
                            resident is not None
                            and resident.n_prompts == saved_n
                            # A resident stamped with the durable pointer must
                            # represent that whole artifact, not merely the
                            # selected return view from an earlier subset call.
                            and resident_layers
                            == {int(layer) for layer in sidecar["source_layers"]}
                            and resident_layers >= expected_set
                            and self._jlens_identity
                            == _jlens_sidecar_identity(sidecar)
                        ):
                            existing = (resident, sidecar)
                        else:
                            # A fresh same-corpus subset no-op needs only the
                            # requested shards. Any path that will publish a
                            # union or resume an accumulator still loads the
                            # complete durable roster so preservation is exact.
                            selective_layers = (
                                expected_set
                                if (
                                    saved_n >= len(usable)
                                    and durable_sources >= expected_set
                                    and int(sidecar.get("format_version", 0))
                                    == LENS_FORMAT_VERSION
                                )
                                else None
                            )
                            verified_existing = _load_lens_verified(
                                self.model_id,
                                requested_layers=selective_layers,
                            )
                            if verified_existing is None:
                                existing = None
                                existing_reuse_proof = None
                            else:
                                existing = verified_existing[:2]
                                existing_reuse_proof = verified_existing[2]
                            existing_payload_verified = bool(
                                existing is not None
                                and selective_layers is None
                            )
                    else:
                        existing = None
                        existing_payload_verified = False
                else:
                    existing = None
                    existing_payload_verified = False
                if existing is not None:
                    lens, sidecar = existing
                    existing_set = set(lens.source_layers)
                    if lens.n_prompts >= len(usable):
                        if existing_set >= expected_set:
                            if on_progress is not None:
                                on_progress(
                                    f"lens already fitted on {lens.n_prompts} "
                                    "prompts — nothing to do"
                                )
                            # A kill after durable publication but before the
                            # checkpoint unlink leaves a second full shard set.
                            # Reap it only when the final artifact proves it is
                            # not farther ahead or semantically different.
                            remove_subsumed_lens_checkpoint(
                                self.model_id,
                                verified_final_sidecar=(
                                    sidecar if existing_payload_verified else None
                                ),
                            )
                            selected = lens.select_layers(expected_sources)
                            durable_set = {
                                int(layer)
                                for layer in sidecar.get("source_layers", [])
                            }
                            if existing_set == durable_set:
                                SaklasSession._adopt_fitted_jlens(
                                    self, lens, sidecar=sidecar,
                                )
                            else:
                                # A selective no-op is a return view, not the
                                # complete artifact named by this sidecar. Do
                                # not stamp it with the durable identity or a
                                # later ``session.jlens`` access would mistake
                                # the narrow view for the full disk union.
                                SaklasSession._evict_resident_jlens(self)
                            return selected
                        missing_sources = sorted(expected_set - existing_set)
                        if on_progress is not None:
                            on_progress(
                                "reusing existing fitted layers; fitting missing "
                                f"J-lens layers {missing_sources}"
                            )
                        missing_base = None
                        topup_sidecar = load_lens_checkpoint_sidecar(self.model_id)
                        topup_progress = _checkpoint_progress(topup_sidecar)
                        topup_corpus_matches = bool(
                            topup_sidecar is not None
                            and (
                                topup_sidecar.get("corpus_sha256") == corpus_sha
                                or (
                                    0 < topup_progress <= len(all_consumed_ids)
                                    and topup_sidecar.get(
                                        "consumed_prefix_sha256"
                                    ) == _token_rows_sha(
                                        all_consumed_ids[:topup_progress]
                                    )
                                )
                            )
                        )
                        topup_meta_matches = bool(
                            topup_sidecar is not None
                            and topup_corpus_matches
                            and topup_sidecar.get("corpus_hash_kind")
                            == corpus_hash_kind
                            and topup_progress > 0
                            and topup_sidecar.get("seq_len") == seq_len
                            and topup_sidecar.get("model_fingerprint")
                            == model_fingerprint
                            and topup_sidecar.get("base_n_prompts") == 0
                            and [int(l) for l in topup_sidecar.get("source_layers", [])]
                            == missing_sources
                        )
                        topup_ckpt = (
                            load_lens_checkpoint(self.model_id)
                            if topup_meta_matches else None
                        )
                        if topup_ckpt is not None:
                            partial, topup_sidecar = topup_ckpt
                            if partial.source_layers == missing_sources:
                                missing_base = partial
                                if on_progress is not None:
                                    on_progress(
                                        "resuming missing-layer checkpoint at "
                                        f"{partial.n_prompts} prompts"
                                    )
                        topup_offset = (
                            missing_base.n_prompts
                            if missing_base is not None else 0
                        )
                        topup_prompts = usable[topup_offset:]
                        topup_ids = consumed_ids[topup_offset:]

                        def _save_topup_accumulator(
                            sums: "Mapping[int, torch.Tensor]",
                            completed: int,
                            d_model: int,
                        ) -> None:
                            save_lens_checkpoint_accumulator(
                                sums, completed, d_model, self.model_id,
                                # ``fit_jacobian_lens`` owns/mutates the prefix
                                # as part of this raw accumulator.
                                base=None,
                                corpus_spec=corpus_spec,
                                corpus_sha256=corpus_sha,
                                corpus_hash_kind=corpus_hash_kind,
                                seq_len=seq_len,
                                dim_batch=dim_batch,
                                skip_first=SKIP_FIRST_POSITIONS,
                                raw_corpus_sha256=raw_corpus_sha,
                                raw_prompt_count=raw_prompt_count,
                                usable_prompt_count=len(consumed_ids),
                                model_layer_count=len(fit_layers),
                                model_fingerprint=model_fingerprint,
                                model_source_fingerprint=model_source_fp,
                                consumed_prefix_sha256=_token_rows_sha(
                                    all_consumed_ids[:completed]
                                ),
                            )

                        if topup_prompts:
                            missing_tail = fit_jacobian_lens(
                                fit_model, self._tokenizer, topup_prompts,
                                fit_layers, source_layers=missing_sources,
                                dim_batch=dim_batch, max_seq_len=seq_len,
                                prompt_batch=prompt_batch,
                                checkpoint_accumulator_cb=(
                                    _save_topup_accumulator
                                ),
                                checkpoint_every=checkpoint_every,
                                on_progress=on_progress,
                                progress_base=topup_offset,
                                input_id_rows=topup_ids,
                                cancel_event=cancel_event,
                                initial_lens=missing_base,
                            )
                            missing = missing_tail
                        else:
                            assert missing_base is not None
                            missing = missing_base
                        merged = JacobianLens.union_layers([lens, missing])
                        save_lens(
                            merged, self.model_id,
                            corpus_spec=corpus_spec, corpus_sha256=corpus_sha,
                            corpus_hash_kind=corpus_hash_kind,
                            seq_len=seq_len, dim_batch=dim_batch,
                            skip_first=SKIP_FIRST_POSITIONS,
                            durable=True,
                            raw_corpus_sha256=raw_corpus_sha,
                            raw_prompt_count=raw_prompt_count,
                            usable_prompt_count=len(consumed_ids),
                            model_layer_count=len(fit_layers),
                            model_fingerprint=model_fingerprint,
                            model_source_fingerprint=model_source_fp,
                            reuse_layers=existing_set,
                            _verified_reuse_proof=existing_reuse_proof,
                        )
                        remove_lens_checkpoint(self.model_id)
                        selected = merged.select_layers(expected_sources)
                        SaklasSession._adopt_fitted_jlens(
                            self, merged,
                            sidecar=load_lens_sidecar(self.model_id),
                        )
                        return selected
                    if existing_set >= expected_set:
                        base = lens.select_layers(expected_sources)
                    else:
                        base = None
                load_checkpoint_payload = bool(
                    checkpoint_meta_matches
                    and (
                        (
                            checkpoint_base_n == 0
                            and (base is None or checkpoint_n > base.n_prompts)
                        )
                        or (
                            checkpoint_base_n > 0
                            and base is not None
                            and checkpoint_base_n == base.n_prompts
                        )
                    )
                )
                if (
                    load_checkpoint_payload
                    and checkpoint_base_n == 0
                    and self._jlens is not None
                    and not resident_evicted_early
                ):
                    pre_evicted_live = (
                        {
                            "layers": list(self._live_lens["layers"]),
                            "uses_all_layers": bool(
                                self._live_lens.get("uses_all_layers", False)
                            ),
                        }
                        if self._live_lens is not None else None
                    )
                    SaklasSession._evict_resident_jlens(self)
                    resident_evicted_early = True
                ckpt = (
                    load_lens_checkpoint(self.model_id)
                    if load_checkpoint_payload else None
                )
                if (
                    ckpt is None
                    and load_checkpoint_payload
                    and durable_fallback_after_checkpoint
                ):
                    # Metadata/header preflight intentionally avoids paging the
                    # durable prefix, but the preferred checkpoint can still
                    # fail its full digest/finite validation. Load the durable
                    # artifact only after that failed payload is gone instead
                    # of silently restarting from prompt zero.
                    fallback = load_lens(self.model_id)
                    if fallback is not None:
                        fallback_lens, fallback_sidecar = fallback
                        if set(fallback_lens.source_layers) >= expected_set:
                            lens = fallback_lens
                            sidecar = fallback_sidecar
                            base = fallback_lens.select_layers(expected_sources)
                if ckpt is not None:
                    partial, ckpt_sidecar = ckpt
                    if checkpoint_meta_matches:
                        ckpt_base_n = int(ckpt_sidecar.get("base_n_prompts", -1))
                        if ckpt_base_n == 0 and (
                            base is None or partial.n_prompts >= base.n_prompts
                        ):
                            # New checkpoints are self-contained.  Prefer the
                            # furthest durable accumulator even when an older
                            # full artifact is also present; otherwise a second
                            # interruption would silently rewind to that artifact.
                            base = partial
                            if on_progress is not None:
                                on_progress(
                                    "resuming from checkpoint at "
                                    f"{base.n_prompts} prompts"
                                )
                        elif base is not None and ckpt_base_n == base.n_prompts:
                            base = JacobianLens.merge_into(
                                [base, partial], target=-1,
                            )
                            if on_progress is not None:
                                on_progress(
                                    "resuming from checkpoint at "
                                    f"{base.n_prompts} prompts"
                                )
                if base is not None:
                    base_n_prompts = base.n_prompts
                    usable = usable[base_n_prompts:]
                    consumed_ids = consumed_ids[base_n_prompts:]
                    if on_progress is not None:
                        on_progress(
                            f"resuming from {base_n_prompts} prompts "
                            f"({len(usable)} remaining)"
                        )

            # `select_layers` deliberately aliases selected tensors. Drop the
            # source containers before resume so an all-layer resident/disk
            # lens does not keep every unrequested fp32 matrix alive beside the
            # subset accumulator. The selected `base` owns the only references
            # the estimator needs.
            existing = None
            verified_existing = None
            fallback = None
            ckpt = None
            partial = None
            lens = None

            prompt_base = base.n_prompts if base is not None else 0

            def _save_full(lens: "Any") -> "Any":
                save_lens(
                    lens, self.model_id,
                    corpus_spec=corpus_spec, corpus_sha256=corpus_sha,
                    corpus_hash_kind=corpus_hash_kind,
                    seq_len=seq_len, dim_batch=dim_batch,
                    skip_first=SKIP_FIRST_POSITIONS,
                    durable=True,
                    raw_corpus_sha256=raw_corpus_sha,
                    raw_prompt_count=raw_prompt_count,
                    usable_prompt_count=len(consumed_ids) + prompt_base,
                    model_layer_count=len(fit_layers),
                    model_fingerprint=model_fingerprint,
                    model_source_fingerprint=model_source_fp,
                )
                remove_lens_checkpoint(self.model_id)
                return lens

            checkpoint_written = False
            checkpoint_proof: Any = None

            def _save_checkpoint_accumulator(
                sums: "Mapping[int, torch.Tensor]",
                completed: int,
                d_model: int,
            ) -> None:
                nonlocal checkpoint_written, checkpoint_proof
                proof_out: list[Any] = []
                save_lens_checkpoint_accumulator(
                    sums, completed, d_model, self.model_id,
                    # The prefix is already folded into ``sums`` by the
                    # single-accumulator resume path.
                    base=None,
                    corpus_spec=corpus_spec,
                    corpus_sha256=corpus_sha,
                    corpus_hash_kind=corpus_hash_kind,
                    seq_len=seq_len,
                    dim_batch=dim_batch,
                    skip_first=SKIP_FIRST_POSITIONS,
                    raw_corpus_sha256=raw_corpus_sha,
                    raw_prompt_count=raw_prompt_count,
                    usable_prompt_count=len(consumed_ids) + prompt_base,
                    model_layer_count=len(fit_layers),
                    model_fingerprint=model_fingerprint,
                    model_source_fingerprint=model_source_fp,
                    consumed_prefix_sha256=_token_rows_sha(
                        all_consumed_ids[:completed]
                    ),
                    _proof_out=proof_out,
                )
                checkpoint_written = True
                checkpoint_proof = proof_out[0] if proof_out else None

            if base is not None and not usable:
                try:
                    merged = _save_full(base)
                    # The live-state restore primes the locked adoption's
                    # rebuild; both must land as one lens-state
                    # transaction (round-5: an un-locked restore write
                    # between them could be observed torn).
                    with self._lens_instrument.state_lock:
                        if pre_evicted_live is not None:
                            self._live_lens = pre_evicted_live
                        return SaklasSession._adopt_fitted_jlens(
                            self,
                            merged,
                            sidecar=load_lens_sidecar(self.model_id),
                        )
                except BaseException:
                    if resident_evicted_early:
                        restored = load_lens(self.model_id)
                        if restored is not None:
                            with self._lens_instrument.state_lock:
                                if pre_evicted_live is not None:
                                    self._live_lens = pre_evicted_live
                                SaklasSession._adopt_fitted_jlens(
                                    self, restored[0], sidecar=restored[1],
                                )
                    raise

            resident = self._jlens
            had_resident = resident is not None or resident_evicted_early
            resume_live = (
                pre_evicted_live
                if resident_evicted_early else
                {
                    "layers": list(self._live_lens["layers"]),
                    "uses_all_layers": bool(
                        self._live_lens.get("uses_all_layers", False)
                    ),
                }
                if self._live_lens is not None else None
            )
            if resident is not None:
                # The estimator takes ownership and converts averages to sums.
                # Drop any resident reference for the duration. It may alias
                # the prefix or be a stale externally-replaced lens; retaining
                # either (and especially its device-stack cache) would defeat
                # the one-full-matrix-set resume bound. Fresh force refits get
                # the same bounded behavior. Preserve only the tiny live config
                # so adoption can rebuild it from the replacement lens.
                SaklasSession._evict_resident_jlens(self)
                resident = None
            try:
                merged = fit_jacobian_lens(
                    fit_model, self._tokenizer, usable, fit_layers,
                    source_layers=expected_sources, dim_batch=dim_batch,
                    prompt_batch=prompt_batch,
                    max_seq_len=seq_len,
                    checkpoint_accumulator_cb=_save_checkpoint_accumulator,
                    checkpoint_every=checkpoint_every,
                    on_progress=on_progress,
                    progress_base=prompt_base,
                    input_id_rows=consumed_ids,
                    cancel_event=cancel_event,
                    initial_lens=base,
                )
                if checkpoint_written and promote_lens_checkpoint(
                    self.model_id,
                    n_prompts=merged.n_prompts,
                    source_layers=merged.source_layers,
                    corpus_sha256=corpus_sha,
                    corpus_hash_kind=corpus_hash_kind,
                    seq_len=seq_len,
                    d_model=merged.d_model,
                    model_fingerprint=model_fingerprint,
                    _verified_proof=checkpoint_proof,
                ):
                    sidecar = load_lens_sidecar(self.model_id)
                else:
                    merged = _save_full(merged)
                    sidecar = load_lens_sidecar(self.model_id)
                with self._lens_instrument.state_lock:
                    if resume_live is not None:
                        self._live_lens = resume_live
                    return SaklasSession._adopt_fitted_jlens(
                        self, merged, sidecar=sidecar,
                    )
            except BaseException:
                if had_resident:
                    restored = load_lens(self.model_id)
                    if restored is not None:
                        restored_lens, restored_sidecar = restored
                        with self._lens_instrument.state_lock:
                            if resume_live is not None:
                                self._live_lens = resume_live
                            SaklasSession._adopt_fitted_jlens(
                                self, restored_lens, sidecar=restored_sidecar,
                            )
                raise

    def _resolve_jlens_layers(
        self,
        lens: "Any",
        layers: "Sequence[int] | str | None",
        *,
        sample_count: int = 9,
    ) -> list[int]:
        if layers is None or layers == "all":
            return list(lens.source_layers)
        if isinstance(layers, str):
            mode = layers.lower()
            if mode in {"band", "workspace"}:
                return self._jlens_workspace_band(lens)
            if mode == "sample":
                source_layers = list(lens.source_layers)
                if len(source_layers) <= sample_count:
                    return source_layers
                step = (len(source_layers) - 1) / max(sample_count - 1, 1)
                return sorted({source_layers[round(i * step)] for i in range(sample_count)})
            raise ValueError(f"unknown J-lens layer mode {layers!r}")
        return [int(layer) for layer in layers]

    def _resolve_jlens_source_layers(
        self,
        source_layers: "Sequence[int] | str | None",
        *,
        n_layers: int,
    ) -> list[int]:
        final_idx = n_layers - 1
        if source_layers is None or source_layers == "all":
            return list(range(final_idx))
        if isinstance(source_layers, str):
            mode = source_layers.lower()
            if mode in {"workspace", "band"}:
                return workspace_layer_indices(range(final_idx), n_layers)
            if mode == "sample":
                all_sources = list(range(final_idx))
                if len(all_sources) <= 9:
                    return all_sources
                step = (len(all_sources) - 1) / 8
                return sorted({all_sources[round(i * step)] for i in range(9)})
            raise ValueError(f"unknown J-lens source-layer mode {source_layers!r}")
        return sorted(set(int(layer) for layer in source_layers))

    def _jlens_transport_stack(
        self, lens: "Any", layers: list[int], device: torch.device,
    ) -> torch.Tensor:
        cache = cast(
            "dict[tuple[int, str, tuple[int, ...]], torch.Tensor]",
            self._jlens_device_cache,
        )
        key = (id(lens), str(device), tuple(layers))
        cached = cache.get(key)
        if cached is not None:
            return cached
        stack = torch.stack([
            lens.jacobians[layer].to(device=device, dtype=torch.float32)
            for layer in layers
        ])
        cache[key] = stack
        return stack

    def _jlens_readout_modules(self) -> tuple[torch.Tensor, torch.nn.Module]:
        """Final unembedding + norm modules for J-lens readout logits."""
        cached = self._jlens_readout_module_cache
        if cached is not None:
            return cached
        from saklas.core.model import get_final_norm, get_unembedding

        modules = (get_unembedding(self._model), get_final_norm(self._model))
        self._jlens_readout_module_cache = modules
        return modules

    def _jlens_decode_id(self, token_id: int) -> str:
        """Cache-backed single-token decode shared by every lens readout."""
        decode_cache = cast(
            "dict[int, str]", self._jlens_decode_cache,
        )
        tok = decode_cache.get(token_id)
        if tok is None:
            tok = str(self._tokenizer.decode([token_id]))
            decode_cache[token_id] = tok
        return tok

    def _jlens_depths(self, layers: "Sequence[int]") -> list[float]:
        """Normalized layer depths (``layer / (n_layers − 1)``, 0 = first
        block, 1 = last) — the depth axis of the aggregate readout's
        center-of-mass statistic."""
        layer_tuple = tuple(int(layer) for layer in layers)
        cache = self._jlens_depths_cache
        cached = cache.get(layer_tuple)
        if cached is not None:
            return cached
        denom = max(len(self._layers) - 1, 1)
        depths = [layer / denom for layer in layer_tuple]
        cache[layer_tuple] = depths
        return depths

    def _jlens_depth_tensor(
        self,
        layers: "Sequence[int]",
        device: torch.device,
    ) -> torch.Tensor:
        """Device depth column for a fixed layer set, cached across decode steps."""
        cache = self._jlens_depth_tensor_cache
        layer_tuple = tuple(int(layer) for layer in layers)
        key = (str(device), layer_tuple)
        cached = cache.get(key)
        if cached is not None:
            return cached
        depths = self._jlens_depths(layer_tuple)
        tensor = torch.tensor(
            depths, dtype=torch.float32, device=device,
        ).reshape(len(depths), 1)
        cache[key] = tensor
        return tensor

    def _readout_long_tensor(
        self,
        values: "Sequence[int]",
        device: torch.device,
    ) -> torch.Tensor:
        """Device long selector tensor cached by value tuple and device."""
        cache = self._readout_long_tensor_cache
        value_tuple = tuple(int(value) for value in values)
        key = (str(device), value_tuple)
        cached = cache.get(key)
        if cached is not None:
            return cached
        tensor = torch.tensor(value_tuple, dtype=torch.long, device=device)
        cache[key] = tensor
        return tensor

    @staticmethod
    def _select_tensor_rows(
        tensor: torch.Tensor,
        rows: "Sequence[int]",
    ) -> torch.Tensor:
        """Select rows without copying when the request is identity/contiguous."""
        row_list = [int(row) for row in rows]
        if not row_list:
            return tensor[:0]
        if row_list == list(range(int(tensor.shape[0]))):
            return tensor
        start = row_list[0]
        if row_list == list(range(start, start + len(row_list))):
            return tensor[start:start + len(row_list)]
        return tensor.index_select(
            0,
            torch.tensor(row_list, device=tensor.device, dtype=torch.long),
        )

    def _jlens_logits_rows(
        self,
        lens: "Any",
        rows: list[tuple[int, torch.Tensor]],
    ) -> torch.Tensor:
        """Full-vocab lens logits ``[n_rows, vocab]`` for a batch of
        ``(layer, hidden_row)`` pairs — the shared front half of the
        per-layer top-k and the layer-aggregated readout (one bmm + one
        unembed matvec serves both)."""
        unembed, final_norm = self._jlens_readout_modules()
        device = unembed.device
        unique_layers = sorted({layer for layer, _ in rows})
        J_unique = self._jlens_transport_stack(lens, unique_layers, device)
        layer_to_row = {layer: idx for idx, layer in enumerate(unique_layers)}
        J_rows = SaklasSession._select_tensor_rows(
            J_unique,
            [layer_to_row[layer] for layer, _ in rows],
        )
        H = torch.stack([
            hidden.detach().to(torch.float32) for _, hidden in rows
        ]).to(device)
        transported = torch.bmm(J_rows, H.unsqueeze(-1)).squeeze(-1)
        normed = final_norm(transported)
        return normed.to(unembed.dtype) @ unembed.T

    def _jlens_aggregate_rows(
        self,
        logits: torch.Tensor | None,
        layers: "Sequence[int]",
        *,
        top_k: int,
        probabilities: torch.Tensor | None = None,
    ) -> list[tuple[str, float, float, float]]:
        """Layer-aggregate per-layer lens logits into the decoded chip list
        ``[(token, strength, com, spread), ...]`` (see
        :func:`saklas.core.jlens.aggregate_readout` for the statistics)."""
        from saklas.core.jlens import (
            aggregate_readout,
            aggregate_readout_from_probabilities,
        )

        if probabilities is None:
            if logits is None:
                raise ValueError("logits are required when probabilities are absent")
            depth_tensor = self._jlens_depth_tensor(layers, logits.device)
            rows = aggregate_readout(
                logits.float(),
                self._jlens_depths(layers),
                top_k=top_k,
                depth_tensor=depth_tensor,
            )
        else:
            depth_tensor = self._jlens_depth_tensor(layers, probabilities.device)
            rows = aggregate_readout_from_probabilities(
                probabilities,
                self._jlens_depths(layers),
                top_k=top_k,
                depth_tensor=depth_tensor,
            )
        return [
            (self._jlens_decode_id(token_id), strength, com, spread)
            for token_id, strength, com, spread in rows
        ]

    def _jlens_topk_rows(
        self,
        lens: "Any",
        rows: list[tuple[int, torch.Tensor]],
        *,
        top_k: int,
        logits: torch.Tensor | None = None,
    ) -> list[list[tuple[str, float, int]]]:
        from saklas.core.jlens import topk_logprobs

        if not rows:
            return []
        if logits is None:
            logits = self._jlens_logits_rows(lens, rows)
        vals, idxs = topk_logprobs(logits, top_k)
        all_vals = vals.cpu()
        all_idxs = idxs.cpu()
        out: list[list[tuple[str, float, int]]] = []
        for vrow, irow in zip(all_vals, all_idxs):
            row: list[tuple[str, float, int]] = []
            for value, idx in zip(vrow, irow):
                token_id = int(idx)
                row.append((self._jlens_decode_id(token_id), float(value), token_id))
            out.append(row)
        return out

    @overload
    def jlens_readout(
        self,
        prompt: str,
        *,
        layers: "Sequence[int] | str | None" = None,
        positions: "Sequence[int] | None" = None,
        top_k: int = 10,
        aggregate: Literal[False] = False,
    ) -> dict[int, list[list[tuple[str, float]]]]: ...

    @overload
    def jlens_readout(
        self,
        prompt: str,
        *,
        layers: "Sequence[int] | str | None" = None,
        positions: "Sequence[int] | None" = None,
        top_k: int = 10,
        aggregate: Literal[True],
    ) -> tuple[
        dict[int, list[list[tuple[str, float]]]],
        list[list[tuple[str, float, float, float]]],
    ]: ...

    def jlens_readout(
        self,
        prompt: str,
        *,
        layers: "Sequence[int] | str | None" = None,
        positions: "Sequence[int] | None" = None,
        top_k: int = 10,
        aggregate: bool = False,
    ) -> (
        dict[int, list[list[tuple[str, float]]]]
        | tuple[
            dict[int, list[list[tuple[str, float]]]],
            list[list[tuple[str, float, float, float]]],
        ]
    ):
        """Jacobian-lens readout on a raw prompt: the top-``top_k`` vocabulary
        tokens per (layer, position), with log-probabilities.

        ``layers`` defaults to every fitted layer; ``positions`` defaults to
        the final position only (pass explicit indices, negative ok, for
        more). Returns ``{layer: [per-position [(token, logprob), ...]]}``.
        One vocab-sized matvec per (layer, position row) — full-position
        sweeps over all layers are an offline-analysis cost, not a decode
        cost.

        ``aggregate=True`` additionally layer-aggregates each position's
        readout (per-layer softmax → mean-probability strength +
        probability-mass-weighted depth center of mass; see
        :func:`saklas.core.jlens.aggregate_readout`) and returns the pair
        ``(per_layer, aggregate)`` where ``aggregate`` is one
        ``[(token, strength, com, spread), ...]`` list per position, from
        the same logits (no extra forward or matvec). The aggregate uses every
        requested fitted layer: the model-specific depth profile remains visible
        in ``com``/``spread`` rather than being pre-filtered by a fixed band.
        """
        from saklas.core.capture import _capture_all_hidden_states

        lens = self._require_jlens()
        req = self._resolve_jlens_layers(lens, layers)
        missing = [l for l in req if l not in lens.jacobians]
        if missing:
            raise ValueError(
                f"layers {missing} not in the fitted lens "
                f"(fitted: {lens.source_layers[0]}..{lens.source_layers[-1]})"
            )
        with self._model_exclusive(
            "session.jlens_readout called while another model use is in flight",
            phase_msg="session.jlens_readout called while a generation is in flight",
        ):
            ids = self._tokenizer(prompt, return_tensors="pt")["input_ids"].to(
                self._device
            )
            n_pos = ids.shape[1]
            pos = (
                [n_pos - 1]
                if positions is None
                else [p if p >= 0 else n_pos + p for p in positions]
            )
            hidden = _capture_all_hidden_states(
                self._model, self._layers, ids, layer_indices=req,
                pool_index=(n_pos - 1 if positions is None else None),
            )
            row_refs: list[tuple[int, int, torch.Tensor]] = []
            for layer in req:
                if positions is None:
                    row_refs.append((layer, 0, hidden[layer]))
                else:
                    h = hidden[layer][0, pos, :]
                    for pos_idx, row in enumerate(h):
                        row_refs.append((layer, pos_idx, row))
            pair_rows = [(layer, row) for layer, _, row in row_refs]
            logits = self._jlens_logits_rows(lens, pair_rows)
            decoded = self._jlens_topk_rows(
                lens, pair_rows, top_k=top_k, logits=logits,
            )
            out: dict[int, list[list[tuple[str, float]]]] = {}
            for layer in req:
                out[layer] = [[] for _ in pos]
            for (layer, pos_idx, _), row in zip(row_refs, decoded):
                out[layer][pos_idx] = [(tok, lp) for tok, lp, _ in row]
            if not aggregate:
                return out
            agg: list[list[tuple[str, float, float, float]]] = []
            for pos_idx in range(len(pos)):
                sel = [
                    i for i, (layer, p, _) in enumerate(row_refs)
                    if p == pos_idx
                ]
                agg.append(self._jlens_aggregate_rows(
                    logits[sel],
                    [row_refs[i][0] for i in sel],
                    top_k=top_k,
                ))
            return out, agg

    def jlens_token_readout(
        self,
        node_id: str,
        raw_index: int,
        *,
        layers: "Sequence[int] | str | None" = None,
        top_k: int = 8,
        apply_steering: bool = True,
        raw: bool = False,
    ) -> dict[str, Any]:
        """Jacobian-lens readout at one decode step of a loom node.

        Rebuilds the exact token stream that *produced* the clicked token —
        the node's rendered prompt (the same render :meth:`fork_from_token`
        replays: loom path + stamped role labels + the recipe's thinking
        toggle) plus ``raw_token_ids[:raw_index]`` — runs one capture
        forward, and reads ``softmax(W_U · norm(J_l h))`` at the final
        position per layer.  The row is the workspace counterpart to the
        token's sampled logits: what each layer's residual was disposed to
        make the model say at the step that chose this token.

        ``apply_steering`` (default on) replays under the node's recipe
        steering, so a steered generation reads its *steered* workspace.
        For an always-active affine term (the dominant case) the single
        forward reproduces the original decode injections exactly — the
        slide is position-independent; phase-split (``@response`` /
        ``@thinking``) and probe-gated terms don't reproduce (a bare
        forward has no decode phases or per-step scores), so their terms
        fire as the trigger context's rest state decides — same fidelity
        class as the fork replay.  ``raw`` selects the flat (base-model /
        raw-buffer) render — the raw flag isn't stamped on the node, so
        the caller supplies it.

        Returns ``{node_id, raw_index, token_id, token_text, steering,
        readout: {layer: [(token, logprob, id), ...]},
        aggregate: [(token, strength, com, spread), ...]}`` — ``aggregate``
        is the layer-aggregated view of the same logits (per-layer softmax
        → mean-probability strength + probability-mass-weighted depth
        center of mass; :func:`saklas.core.jlens.aggregate_readout`).
        Raises :class:`~saklas.core.jlens.LensNotFittedError` with no
        fitted lens, :class:`UnknownNodeError` /
        :class:`InvalidNodeOperationError` on a bad target (mirrors
        :meth:`fork_from_token`), ``ValueError`` on an unfitted layer.
        """
        from saklas.core.capture import _capture_all_hidden_states

        lens = self._require_jlens()
        req = self._resolve_jlens_layers(lens, layers)
        missing = [l for l in req if l not in lens.jacobians]
        if missing:
            raise ValueError(
                f"layers {missing} not in the fitted lens "
                f"(fitted: {lens.source_layers[0]}..{lens.source_layers[-1]})"
            )
        node = self.tree.get(node_id)
        if node.role not in ("assistant", "user"):
            raise InvalidNodeOperationError(
                f"jlens_token_readout: {node_id!r} is a {node.role} node — "
                f"only a turn with a decode record can be read out"
            )
        raw_ids = node.raw_token_ids
        if not raw_ids:
            raise InvalidNodeOperationError(
                f"jlens_token_readout: {node_id!r} has no raw token record "
                f"(transcript-loaded node)"
            )
        if not 0 <= raw_index < len(raw_ids):
            raise InvalidNodeOperationError(
                f"jlens_token_readout: raw_index {raw_index} out of range "
                f"[0, {len(raw_ids)}) for {node_id!r}"
            )

        recipe = node.recipe
        steering_expr = (
            recipe.steering if (apply_steering and recipe is not None) else None
        )

        # The steering scope opens OUTSIDE the exclusive-GPU guard:
        # ``SteeringComposer.push``/``pop`` acquire ``_gen_lock`` blocking
        # (non-reentrant), so nesting the scope inside ``_model_exclusive``
        # would self-deadlock — the same ordering ``score_choices`` uses.
        scope = (
            self.steering(steering_expr) if steering_expr else nullcontext()
        )
        with scope:
            thinking_req = recipe.thinking if recipe is not None else None
            use_thinking = (
                supports_thinking(self._tokenizer)
                if thinking_req is None
                else bool(thinking_req) and supports_thinking(self._tokenizer)
            )
            if raw:
                prompt_ids = self._prepare_input(
                    "", raw=True, parent_node_id=node.parent_id,
                )
            else:
                # Continue-mode rebuild: the history walk to
                # ``node.parent_id`` already carries every prior turn with
                # its stamped label (the parent included), so the render is
                # byte-identical to the one that produced the node — and
                # seat-general, no user-parent assumption (a cast-model
                # node may hang under any turn).
                prompt_ids = self._prepare_input(
                    None,
                    thinking=use_thinking,
                    parent_node_id=node.parent_id,
                    user_role=(
                        node.role_label if node.role == "user" else None
                    ),
                    assistant_role=(
                        node.role_label if node.role == "assistant" else None
                    ),
                    gen_seat=node.role,
                )
            if raw_index > 0:
                prefix = torch.tensor(
                    [[int(t) for t in raw_ids[:raw_index]]],
                    dtype=prompt_ids.dtype, device=prompt_ids.device,
                )
                ids = torch.cat([prompt_ids, prefix], dim=1)
            else:
                ids = prompt_ids
            with self._model_exclusive(
                "session.jlens_token_readout called while another model use "
                "is in flight",
                phase_msg="session.jlens_token_readout called while a "
                "generation is in flight",
            ):
                hidden = _capture_all_hidden_states(
                    self._model, self._layers, ids, layer_indices=req,
                    pool_index=ids.shape[1] - 1,
                )
                pair_rows = [(layer, hidden[layer]) for layer in req]
                logits = self._jlens_logits_rows(lens, pair_rows)
                decoded = self._jlens_topk_rows(
                    lens, pair_rows, top_k=top_k, logits=logits,
                )
                readout = {
                    layer: row for layer, row in zip(req, decoded)
                }
                agg = self._jlens_aggregate_rows(
                    logits,
                    req,
                    top_k=top_k,
                )
        return {
            "node_id": node_id,
            "raw_index": int(raw_index),
            "token_id": int(raw_ids[raw_index]),
            "token_text": str(self._tokenizer.decode([int(raw_ids[raw_index])])),
            "steering": steering_expr,
            "readout": readout,
            "aggregate": agg,
        }

    def register_jlens_direction(self, word: str) -> str:
        """Register the J-lens direction for a single-token word as a profile.

        The direction (``W_U[v] @ J_l`` per layer) lands in the ordinary
        profile registry under ``jlens/<word>``, so it steers and ablates
        exactly like an extracted vector. Idempotent. This is the resolver
        behind the lazy ``jlens/`` steering branch — *probes* no longer
        fold this direction: a ``jlens/<word>`` probe reads the readout
        channel (per-layer softmax probability) via the session
        lens-probe registry instead.

        Uses every fitted lens layer. Informative depth ranges vary by model;
        callers that want a narrower intervention can fit or load a lens with
        an explicit source-layer subset.
        """
        from saklas.core.jlens import resolve_word_token
        from saklas.core.model import get_unembedding

        name = f"jlens/{word}"
        if name in self._profiles:
            return name
        lens = self._require_jlens()
        token_id = resolve_word_token(self._tokenizer, word)
        fitted_layers = set(int(layer) for layer in lens.source_layers)
        directions = lens.token_direction(
            token_id, get_unembedding(self._model), layers=sorted(fitted_layers),
        )
        # ``token_direction`` returns CPU tensors; land the profile on the
        # session device so the probe fold (which follows the directions'
        # device) builds device-consistent whitened factors — CPU factors
        # crash the per-probe read paths (``_subspace_coords_for``) against
        # on-device activations.
        self._profiles[name] = {
            l: d.to(self._device)
            for l, d in directions.items()
            if l in fitted_layers
        }
        self._invalidate_prefix_cache()
        self._invalidate_analytics_cache()
        return name

    # -- J-lens instrument state (delegating views) --
    # The lens read family lives on ``self._lens_instrument``; these
    # properties re-expose its state under the historical private names so
    # capture planning, the steering composer, the token tap, the wire
    # layers, and tests keep one source of truth during the instrument
    # migration (the plan/run split retargets the call sites later).

    @property
    def _lens_instrument(self) -> "LensInstrument":
        """The J-lens instrument, created lazily on first touch.

        Lazy so a narrow test stub that subclasses the session but skips
        ``__init__`` (or assigns lens state in ``__new__``) self-heals: the
        first state read/write constructs the instrument, whose ``__init__``
        only stores the session reference.
        """
        inst = self.__dict__.get("_lens_instrument")
        if inst is None:
            from saklas.core.instruments.lens import LensInstrument

            inst = LensInstrument(self)
            self.__dict__["_lens_instrument"] = inst
        return inst

    @property
    def _lens_probes(self) -> dict[str, dict[str, Any]]:
        return self._lens_instrument.probes

    @_lens_probes.setter
    def _lens_probes(self, value: dict[str, dict[str, Any]]) -> None:
        self._lens_instrument.probes = value

    @property
    def _live_lens(self) -> dict[str, Any] | None:
        return self._lens_instrument.live

    @_live_lens.setter
    def _live_lens(self, value: "dict[str, Any] | None") -> None:
        self._lens_instrument.live = value

    @property
    def _lens_step_stash(self) -> dict[str, Any] | None:
        return self._lens_instrument.step_stash

    @_lens_step_stash.setter
    def _lens_step_stash(self, value: "dict[str, Any] | None") -> None:
        self._lens_instrument.step_stash = value

    @property
    def _last_lens_step_readings(self) -> "dict[str, ProbeReading] | None":
        return self._lens_instrument.last_step_readings

    @_last_lens_step_readings.setter
    def _last_lens_step_readings(
        self, value: "dict[str, ProbeReading] | None",
    ) -> None:
        self._lens_instrument.last_step_readings = value

    @property
    def _live_lens_active_for_generation(self) -> bool:
        return self._lens_instrument.active_for_generation

    @_live_lens_active_for_generation.setter
    def _live_lens_active_for_generation(self, value: bool) -> None:
        self._lens_instrument.active_for_generation = bool(value)

    @property
    def _generation_jlens(self) -> Any:
        return self._lens_instrument.generation_lens

    @_generation_jlens.setter
    def _generation_jlens(self, value: Any) -> None:
        self._lens_instrument.generation_lens = value

    @property
    def _generation_jlens_active(self) -> bool:
        return self._lens_instrument.generation_lens_active

    @_generation_jlens_active.setter
    def _generation_jlens_active(self, value: bool) -> None:
        self._lens_instrument.generation_lens_active = bool(value)

    # -- SAE instrument state (delegating views) --

    @property
    def _sae_instrument(self) -> "SaeInstrument":
        """The SAE instrument, created lazily on first touch (stub-safe,
        like ``_lens_instrument``)."""
        inst = self.__dict__.get("_sae_instrument")
        if inst is None:
            from saklas.core.instruments.sae import SaeInstrument

            inst = SaeInstrument(self)
            self.__dict__["_sae_instrument"] = inst
        return inst

    @property
    def _sae_probes(self) -> dict[str, dict[str, Any]]:
        return self._sae_instrument.probes

    @_sae_probes.setter
    def _sae_probes(self, value: dict[str, dict[str, Any]]) -> None:
        self._sae_instrument.probes = value

    @property
    def _live_sae(self) -> dict[str, Any] | None:
        return self._sae_instrument.live

    @_live_sae.setter
    def _live_sae(self, value: "dict[str, Any] | None") -> None:
        self._sae_instrument.live = value

    @property
    def _sae_step_stash(self) -> dict[str, Any] | None:
        return self._sae_instrument.step_stash

    @_sae_step_stash.setter
    def _sae_step_stash(self, value: "dict[str, Any] | None") -> None:
        self._sae_instrument.step_stash = value

    @property
    def _last_sae_step_readings(self) -> "dict[str, ProbeReading] | None":
        return self._sae_instrument.last_step_readings

    @_last_sae_step_readings.setter
    def _last_sae_step_readings(
        self, value: "dict[str, ProbeReading] | None",
    ) -> None:
        self._sae_instrument.last_step_readings = value

    @property
    def _live_sae_active_for_generation(self) -> bool:
        return self._sae_instrument.active_for_generation

    @_live_sae_active_for_generation.setter
    def _live_sae_active_for_generation(self, value: bool) -> None:
        self._sae_instrument.active_for_generation = bool(value)

    @property
    def _geometry_instrument(self) -> "GeometryInstrument":
        """The geometry instrument (thin Monitor adapter), created lazily
        on first touch like its lens/SAE siblings."""
        inst = self.__dict__.get("_geometry_instrument")
        if inst is None:
            from saklas.core.instruments.geometry import GeometryInstrument

            inst = GeometryInstrument(self)
            self.__dict__["_geometry_instrument"] = inst
        return inst

    @property
    def instruments(self) -> dict[str, Any]:
        """The three read-family instruments keyed by family name —
        ``geometry`` (the Monitor adapter), ``lens``, ``sae``.  The
        uniform registry behind the probe-hash roster, gate preflight,
        and the server's instrument enumeration."""
        return {
            "geometry": self._geometry_instrument,
            "lens": self._lens_instrument,
            "sae": self._sae_instrument,
        }

    def _close_instrument_runs(self) -> None:
        """Close every family's per-generation run (idempotent teardown).

        Every capture transaction that binds runs must reach this in its
        ``finally`` — the ordinary generation, the batch fast path, and
        the joint-logprob replay.  A bound run leaking past its
        transaction pins a stale lens (suppressing disk refresh between
        generations) and freezes SAE units past their generation."""
        self._geometry_instrument.close_run()
        self._lens_instrument.close_run()
        self._sae_instrument.close_run()

    @property
    def lens(self) -> "LensInstrument":
        """The J-lens instrument — the typed public face of the lens read
        family (``session.instruments["lens"]``): probe attach/detach,
        live-readout enable/disable, per-step and aggregate scoring.
        Artifact lifecycle (``fit_jlens`` / ``select_jlens_source`` /
        ``jlens_readout`` / token replay) stays on the session — it is
        source management, not measurement."""
        return self._lens_instrument

    @property
    def sae(self) -> "SaeInstrument":
        """The SAE instrument — the typed public face of the SAE read
        family (``session.instruments["sae"]``).  Backend lifecycle
        (``train_sae`` / ``load_sae`` / ``unload_sae`` / ``sae_info``)
        stays on the session."""
        return self._sae_instrument

    def enable_live_lens(
        self,
        *,
        layers: "Sequence[int] | None" = None,
    ) -> list[int]:
        """Stream the J-lens readout live during generation.

        Delegates to :meth:`LensInstrument.enable_live` — see
        ``core/instruments/lens.py`` for the contract (device residency,
        layer defaulting, no forward hooks / compile eligibility untouched).
        Returns the resolved layer list.
        """
        return self._lens_instrument.enable_live(layers=layers)

    def _active_jlens_source_label(self) -> str | None:
        """Return the active J-lens source in the public source syntax."""
        from saklas.io.lens_sources import load_active_lens_source

        active = load_active_lens_source(self.model_id)
        if active is None:
            return None
        if active["kind"] == "local":
            return f"local:{active['name']}"
        return str(active["name"])

    def disable_live_lens(self) -> None:
        """Stop streaming the live lens readout and free the device J_l copies."""
        self._lens_instrument.disable_live()

    @property
    def live_lens_layers(self) -> list[int] | None:
        """The live lens readout's layer list, or ``None`` when it's off."""
        return self._lens_instrument.live_layers

    def _jlens_workspace_band(self, lens: "Any") -> list[int]:
        """Explicit legacy ``workspace`` mode: fitted layers in the 40–90%
        depth band, with a shallow-model fallback. J-lens defaults do not use
        this model-agnostic heuristic."""
        n = len(self._layers)
        return workspace_layer_indices(list(lens.source_layers), n)

    def _live_lens_readout_step(
        self, *, top_k: int = 8, step_id: int = -1,
    ) -> (
        tuple[
            dict[int, list[tuple[str, float]]],
            list[tuple[str, float, float, float]],
            dict[int, list[int]],
        ]
        | None
    ):
        """One decode step's live lens readout (token-tap, post-forward).

        Delegates to :meth:`LensInstrument.live_readout_step` — stash reuse
        with the gate callback (step-keyed: rows are reused iff the stash
        came from this same forward), per-layer softmax probabilities,
        aggregate chips, one packed host transfer.
        """
        return self._lens_instrument.live_readout_step(
            top_k=top_k, step_id=step_id,
        )

    # -- Sparse-autoencoder runtime -------------------------------------

    def train_sae(
        self,
        name: str,
        documents: "Sequence[str]",
        *,
        layer: int,
        corpus_spec: str = "user",
        tokens: int = 1_000_000,
        seq_len: int = 128,
        batch_size: int = 8,
        d_sae: int | None = None,
        expansion: int = 8,
        learning_rate: float = 3e-4,
        l1_coefficient: float = 1e-3,
        dead_feature_threshold: float = 1e-6,
        seed: int = 0,
        force: bool = False,
        on_progress: "Callable[[str], None] | None" = None,
        cancel_event: "threading.Event | None" = None,
    ) -> dict[str, Any]:
        """Train and persist a Saklas-owned residual-post SAE."""
        import hashlib

        from saklas.core.sae_training import train_residual_sae
        from saklas.io.sae_artifacts import (
            local_sae_release,
            normalize_local_sae_name,
            save_local_sae,
        )

        local_name = normalize_local_sae_name(name)
        docs = list(documents)
        if not docs:
            raise ValueError("SAE training needs at least one corpus document")
        corpus_sha = hashlib.sha256(repr(docs).encode("utf-8")).hexdigest()
        with self._model_exclusive(
            "session.train_sae called while another model use is in flight",
            phase_msg="session.train_sae called while a generation is in flight",
        ):
            SaklasSession._assert_unsteered_artifact_operation(self)
            fit_model = getattr(self._model, "_orig_mod", self._model)
            tensors, metrics = train_residual_sae(
                fit_model,
                self._tokenizer,
                list(get_layers(fit_model)),
                docs,
                layer=layer,
                tokens=tokens,
                seq_len=seq_len,
                batch_size=batch_size,
                d_sae=d_sae,
                expansion=expansion,
                learning_rate=learning_rate,
                l1_coefficient=l1_coefficient,
                dead_feature_threshold=dead_feature_threshold,
                seed=seed,
                on_progress=on_progress,
                cancel_event=cancel_event,
            )
            manifest = save_local_sae(
                self.model_id,
                local_name,
                tensors,
                model_fingerprint=loaded_model_fingerprint(
                    fit_model, self.model_id,
                ),
                model_source_fingerprint=getattr(
                    fit_model, "_saklas_source_fingerprint", None,
                ),
                layer=layer,
                corpus_spec=corpus_spec,
                corpus_sha256=corpus_sha,
                tokens_trained=int(metrics["tokens_trained"]),
                seq_len=seq_len,
                batch_size=batch_size,
                learning_rate=learning_rate,
                l1_coefficient=l1_coefficient,
                dead_feature_threshold=dead_feature_threshold,
                force=force,
            )
        runtime = self.load_sae(local_sae_release(local_name), layer=layer)
        return {
            "source": local_sae_release(local_name),
            "artifact": str(manifest),
            "metrics": metrics,
            "runtime": runtime,
        }

    def load_sae(self, release: str, *, layer: int | None = None) -> dict[str, Any]:
        """Load one local or SAELens source as the session's resident SAE.

        Provider registry resolution stays eager while the selected layer's
        weights are made resident here; local artifacts resolve directly from
        Saklas storage. The deterministic default is the covered layer nearest
        65% model depth, preferring the workspace band.
        """
        from saklas.core.sae import (
            LocalSaeBackend,
            load_sae_backend,
            select_runtime_layer,
            validate_residual_width,
        )
        from saklas.io.sae import (
            load_sae_feature_meta,
            save_sae_metadata,
            set_active_sae_source,
        )

        release = release.strip()
        if not release:
            raise ValueError("SAE release must not be empty")
        with self._model_exclusive(
            "load_sae called while another model operation is in flight; retry shortly"
        ):
            backend = load_sae_backend(
                release, model_id=self.model_id, device=self._device,
                dtype=self._dtype, warn_on_multiple=False,
            )
            if (
                isinstance(backend, LocalSaeBackend)
                and backend.model_fingerprint is not None
                and backend.model_fingerprint
                != loaded_model_fingerprint(self._model, self.model_id)
            ):
                raise ValueError(
                    f"local SAE {release!r} was trained against different "
                    "loaded model weights"
                )
            covered = frozenset(
                idx for idx in backend.layers if 0 <= idx < len(self._layers)
            )
            selected = select_runtime_layer(covered, len(self._layers), layer)
            width = int(backend.feature_count(selected))  # materialize weights
            if width <= 0:
                raise ValueError(f"SAE release {release!r} has no features")
            validate_residual_width(
                backend, selected, int(self._model_info["hidden_dim"]),
            )
            feature_meta = (
                {} if isinstance(backend, LocalSaeBackend)
                else load_sae_feature_meta(self.model_id, release)
            )
            # Publish/validate the source binding before replacing the
            # session's resident runtime.  A metadata failure must leave the
            # previous source intact; the old order reported "load failed"
            # while ``sae_loaded`` was nevertheless true for the half-adopted
            # provider backend.
            if isinstance(backend, LocalSaeBackend):
                set_active_sae_source(
                    self.model_id, "local", release.removeprefix("local:"),
                )
            else:
                save_sae_metadata(self.model_id, release, {
                    "layer": selected,
                    "width": width,
                    "revision": backend.revision,
                    "fingerprint": backend.fingerprint,
                    "sae_id": backend.sae_ids_by_layer.get(str(selected)),
                    "repo_id": backend.repo_id,
                    "neuronpedia_id": backend.neuronpedia_ids_by_layer.get(
                        str(selected)
                    ),
                })
            self._sae_backend = backend
            self._sae_layer = selected
            self._sae_width = width
            self._sae_feature_meta = feature_meta
            self._live_sae = None
            self._sae_step_stash = None
            self._last_sae_step_readings = None
            # Feature ids belong to the resident release; changing it evicts
            # stale directions and pinned probes rather than silently reusing ids.
            for name in [key for key in self._profiles if key.startswith("sae/")]:
                del self._profiles[name]
            with self._sae_instrument.state_lock:
                for name in list(self._sae_probes):
                    self._probe_hash_cache.pop(name, None)
                self._sae_probes.clear()
        self._invalidate_prefix_cache()
        self._invalidate_analytics_cache()
        return self.sae_info or {}

    def unload_sae(self) -> None:
        """Release the resident SAE and every session-local feature handle."""
        with self._model_exclusive(
            "unload_sae called while another model operation is in flight; retry shortly"
        ):
            self._sae_backend = None
            self._sae_layer = None
            self._sae_width = None
            self._sae_feature_meta = {}
            self._live_sae = None
            self._sae_step_stash = None
            self._last_sae_step_readings = None
            for name in [key for key in self._profiles if key.startswith("sae/")]:
                del self._profiles[name]
            with self._sae_instrument.state_lock:
                for name in list(self._sae_probes):
                    self._probe_hash_cache.pop(name, None)
                self._sae_probes.clear()
        self._invalidate_prefix_cache()
        self._invalidate_analytics_cache()

    @property
    def sae_info(self) -> dict[str, Any] | None:
        backend = self._sae_backend
        layer = self._sae_layer
        width = self._sae_width
        if backend is None or layer is None or width is None:
            return None
        return {
            "release": backend.release,
            "revision": backend.revision,
            "fingerprint": backend.fingerprint,
            "layer": int(layer),
            "width": int(width),
            "sae_id": backend.sae_ids_by_layer.get(str(layer)),
            "repo_id": backend.repo_id,
            "neuronpedia_id": backend.neuronpedia_ids_by_layer.get(str(layer)),
        }

    def _require_sae(self) -> tuple[Any, int, int]:
        from saklas.core.errors import SaeNotLoadedError

        backend = self._sae_backend
        layer = self._sae_layer
        width = self._sae_width
        if backend is None or layer is None or width is None:
            raise SaeNotLoadedError(
                "no SAE loaded for this session — select one with "
                "`saklas sae use MODEL SOURCE` "
                "or POST /saklas/v1/sessions/default/sae/load"
            )
        return backend, int(layer), int(width)

    def validate_sae_feature(self, feature_id: int | str) -> dict[str, Any]:
        from saklas.core.errors import SaeFeatureError

        try:
            idx = int(feature_id)
        except (TypeError, ValueError) as exc:
            raise SaeFeatureError(f"SAE feature id must be an integer: {feature_id!r}") from exc
        _backend, layer, width = self._require_sae()
        if not 0 <= idx < width:
            raise SaeFeatureError(
                f"SAE feature {idx} out of range [0, {width}) for layer {layer}"
            )
        meta = self._sae_feature_meta.get(str(idx))
        if meta is None or (meta.get("max_act") is None and not meta.get("checked")):
            meta = self._fetch_sae_feature_meta(idx) or meta or {}
        return {
            "id": idx,
            "label": meta.get("label"),
            "layer": layer,
            "max_act": meta.get("max_act"),
        }

    def _sae_label(self, feature_id: int) -> str | None:
        entry = self._sae_feature_meta.get(str(feature_id))
        label = entry.get("label") if entry else None
        return label if isinstance(label, str) and label else None

    def _sae_max_act(self, feature_id: int) -> float | None:
        """The feature's Neuronpedia ``maxActApprox`` — the strength unit.

        ``None`` when metadata is missing (offline / not on Neuronpedia), in
        which case every readout surface for the feature stays raw.
        """
        entry = self._sae_feature_meta.get(str(feature_id))
        max_act = entry.get("max_act") if entry else None
        if isinstance(max_act, (int, float)) and float(max_act) > 0:
            return float(max_act)
        return None

    def _fetch_neuronpedia_feature(self, feature_id: int) -> dict[str, Any] | None:
        """One raw Neuronpedia feature fetch → ``{label, max_act, checked}``.

        ``None`` on any transport failure (retry next time); a successful
        response always yields an entry — ``checked`` marks "we asked", so a
        feature with no Neuronpedia data isn't re-fetched on every validate.
        """
        info = self.sae_info or {}
        neuronpedia_id = info.get("neuronpedia_id")
        if not isinstance(neuronpedia_id, str) or "/" not in neuronpedia_id:
            return None
        import json
        from urllib.parse import quote
        from huggingface_hub import get_session

        model, source = neuronpedia_id.split("/", 1)
        url = (
            "https://www.neuronpedia.org/api/feature/"
            f"{quote(model, safe='')}/{quote(source, safe='')}/{feature_id}"
        )
        try:
            response = get_session().get(
                url,
                timeout=2.0,
                headers={"User-Agent": "saklas-sae-meta/1"},
            )
            response.raise_for_status()
            payload = json.loads(response.content)
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        label = None
        for row in payload.get("explanations", []) or []:
            if not isinstance(row, dict):
                continue
            description = row.get("description")
            if isinstance(description, str) and description.strip():
                label = description.strip()
                break
        max_act = payload.get("maxActApprox")
        if not (isinstance(max_act, (int, float)) and float(max_act) > 0):
            max_act = None
        return {
            "label": label,
            "max_act": float(max_act) if max_act else None,
            "checked": True,
        }

    def _fetch_sae_feature_meta(self, feature_id: int) -> dict[str, Any] | None:
        """Best-effort lazy Neuronpedia metadata lookup for an explicit id.

        Fetches the display label and ``maxActApprox`` together. Live/top-k
        readout never calls this method; discovery remains entirely local and
        renders raw activations until metadata is cached (the dashboard
        backfills via :meth:`fetch_sae_feature_meta` between generations).
        """
        entry = self._fetch_neuronpedia_feature(feature_id)
        if entry is None:
            return None
        from saklas.io.sae import save_sae_feature_meta

        self._sae_feature_meta[str(feature_id)] = entry
        backend, _layer, _width = self._require_sae()
        save_sae_feature_meta(
            self.model_id, backend.release, self._sae_feature_meta,
        )
        return entry

    def fetch_sae_feature_meta(
        self, feature_ids: "Sequence[int]",
    ) -> dict[str, dict[str, Any]]:
        """Fetch-and-cache Neuronpedia metadata for a set of feature ids.

        The dashboard's discovery backfill: called between generations with
        the ids the live top-k surfaced, never from the decode loop. Cached
        ids skip the network; misses fetch on a small thread pool with the
        single-id path's timeout. Returns ``{str(id): {label, max_act}}`` for
        every requested id with metadata afterwards (out-of-range ids are
        silently dropped — the top-k can't produce one, so there is nothing
        to report).
        """
        backend, _layer, width = self._require_sae()
        seen: set[int] = set()
        wanted: list[int] = []
        for raw in feature_ids:
            idx = int(raw)
            if not 0 <= idx < width or idx in seen:
                continue
            seen.add(idx)
            entry = self._sae_feature_meta.get(str(idx))
            if entry is None or (
                entry.get("max_act") is None and not entry.get("checked")
            ):
                wanted.append(idx)
        if wanted:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=min(8, len(wanted))) as pool:
                rows = list(pool.map(self._fetch_neuronpedia_feature, wanted))
            fetched = {
                str(idx): entry
                for idx, entry in zip(wanted, rows)
                if entry is not None
            }
            if fetched:
                from saklas.io.sae import save_sae_feature_meta

                self._sae_feature_meta.update(fetched)
                save_sae_feature_meta(
                    self.model_id, backend.release, self._sae_feature_meta,
                )
                self._refresh_sae_probe_meta(fetched)
        return {
            str(idx): {
                "label": entry.get("label"),
                "max_act": entry.get("max_act"),
            }
            for idx in sorted(seen)
            if (entry := self._sae_feature_meta.get(str(idx))) is not None
        }

    def _refresh_sae_probe_meta(self, fetched: dict[str, dict[str, Any]]) -> None:
        """Reflect newly fetched metadata onto attached feature probes.

        A pinned probe's spec carries label/max_act for the probe listing, and
        ``max_act`` is part of the readout-channel identity (it sets the
        strength unit), so the probe-hash cache entry is invalidated too.
        """
        with self._sae_instrument.state_lock:
            for name, spec in list(self._sae_probes.items()):
                entry = fetched.get(str(spec.get("feature_id")))
                if entry is None:
                    continue
                # Whole-dict replacement (one atomic store): idle readers
                # hold per-call spec snapshots or shared references — a
                # field-level mutation could hand them a half-updated
                # unit (round-6).
                self._sae_probes[name] = {
                    **spec,
                    "label": entry.get("label"),
                    "max_act": entry.get("max_act"),
                }
                self._probe_hash_cache.pop(name, None)

    def register_sae_direction(self, feature_id: int | str) -> str:
        """Register ``W_dec[id]`` at the resident hook layer as a profile."""
        validated = self.validate_sae_feature(feature_id)
        idx = int(validated["id"])
        name = f"sae/{idx}"
        if name in self._profiles:
            return name
        backend, layer, _width = self._require_sae()
        direction = backend.feature_direction(layer, idx)
        self._profiles[name] = {
            layer: direction.detach().to(self._device, self._dtype)
        }
        self._invalidate_prefix_cache()
        self._invalidate_analytics_cache()
        return name

    def enable_live_sae(self, *, top_k: int = 8) -> dict[str, Any]:
        """Enable the one-matvec live feature readout at the resident layer
        (delegates to :meth:`SaeInstrument.enable_live`)."""
        return self._sae_instrument.enable_live(top_k=top_k)

    def disable_live_sae(self) -> None:
        self._sae_instrument.disable_live()

    @property
    def live_sae(self) -> bool:
        return self._sae_instrument.is_live

    def _encode_sae_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        backend, layer, width = self._require_sae()
        row = hidden
        if row.ndim == 1:
            row = row.unsqueeze(0)
        acts = backend.encode_layer(layer, row)
        if acts.ndim == 2 and acts.shape[0] == 1:
            acts = acts[0]
        acts = acts.reshape(-1)
        if int(acts.numel()) != width:
            raise ValueError(
                f"SAE encoder returned {acts.numel()} features; expected {width}"
            )
        # SAE reads are diagnostics, probes, and steering metadata; none is a
        # training surface.  Some third-party backends return grad-tracked
        # activations even during an otherwise inference-only session, which
        # retains an unnecessary graph and warns when probe values are scalarized.
        return acts.detach()

    def _score_sae_probes(
        self, hidden: dict[int, torch.Tensor] | None = None,
        *, activations: torch.Tensor | None = None,
        only: "set[str] | None" = None,
        raw_by_fid: Mapping[int, float] | None = None,
    ) -> dict[str, "ProbeReading"]:
        """Score attached SAE probes (delegates to
        :meth:`SaeInstrument.score_probes`)."""
        return self._sae_instrument.score_probes(
            hidden, activations=activations, only=only, raw_by_fid=raw_by_fid,
        )

    def _sae_probe_values(
        self,
        activations: torch.Tensor,
        *,
        only: "set[str] | None" = None,
        raw_by_fid: Mapping[int, float] | None = None,
    ) -> list[tuple[str, int, float, float]]:
        """Pinned SAE probe values (delegates to
        :meth:`SaeInstrument.probe_values`)."""
        return self._sae_instrument.probe_values(
            activations, only=only, raw_by_fid=raw_by_fid,
        )

    def _score_sae_gate_scalars(
        self, gate_keys: "set[str] | None" = None, *, step_id: int = -1,
    ) -> dict[str, float]:
        """Per-forward SAE gate scalars — through the run protocol face
        (:meth:`SaeRun.gate_scalars` → the instrument worker; emits only
        the real strength channel — the historical fake
        ``:fraction``/``:membership`` constants are gone).  ``step_id``
        keys the encode stash so the display step reuses this forward's
        activations."""
        return self._sae_instrument.current_run.gate_scalars(
            step_id, None, gate_keys,
        )

    def _live_sae_readout_step(
        self, *, step_id: int = -1,
    ) -> list[tuple[int, float, str | None, float | None]] | None:
        """One decode step's live SAE top-k (delegates to
        :meth:`SaeInstrument.live_readout_step`; step-keyed stash reuse)."""
        return self._sae_instrument.live_readout_step(step_id=step_id)

    def _score_sae_probes_aggregate(
        self,
        generated_ids: list[int],
        *,
        pooled: dict[int, torch.Tensor] | None = None,
    ) -> dict[str, "ProbeReading"]:
        """End-of-gen SAE-probe aggregate (delegates to
        :meth:`SaeInstrument.score_aggregate`)."""
        return self._sae_instrument.score_aggregate(
            generated_ids, pooled=pooled,
        )

    def sae_token_readout(
        self,
        node_id: str,
        raw_index: int,
        *,
        top_k: int = 8,
        apply_steering: bool = True,
        raw: bool = False,
    ) -> dict[str, Any]:
        """SAE feature readout at the forward that produced a loom token."""
        from saklas.core.capture import _capture_all_hidden_states

        _backend, layer, width = self._require_sae()
        if not 1 <= top_k <= min(width, 100):
            raise ValueError(f"top_k must be in [1, {min(width, 100)}]")
        node = self.tree.get(node_id)
        if node.role not in ("assistant", "user"):
            raise InvalidNodeOperationError(
                f"sae_token_readout: {node_id!r} is a {node.role} node — "
                f"only a turn with a decode record can be read out"
            )
        raw_ids = node.raw_token_ids
        if not raw_ids:
            raise InvalidNodeOperationError(
                f"sae_token_readout: {node_id!r} has no raw token record"
            )
        if not 0 <= raw_index < len(raw_ids):
            raise InvalidNodeOperationError(
                f"sae_token_readout: raw_index {raw_index} out of range "
                f"[0, {len(raw_ids)})"
            )
        recipe = node.recipe
        steering_expr = (
            recipe.steering if (apply_steering and recipe is not None) else None
        )
        scope = self.steering(steering_expr) if steering_expr else nullcontext()
        with scope:
            thinking_req = recipe.thinking if recipe is not None else None
            use_thinking = (
                supports_thinking(self._tokenizer)
                if thinking_req is None
                else bool(thinking_req) and supports_thinking(self._tokenizer)
            )
            if raw:
                prompt_ids = self._prepare_input(
                    "", raw=True, parent_node_id=node.parent_id,
                )
            else:
                # Continue-mode rebuild — seat-general, mirrors
                # ``jlens_token_readout``.
                prompt_ids = self._prepare_input(
                    None,
                    thinking=use_thinking,
                    parent_node_id=node.parent_id,
                    user_role=(
                        node.role_label if node.role == "user" else None
                    ),
                    assistant_role=(
                        node.role_label if node.role == "assistant" else None
                    ),
                    gen_seat=node.role,
                )
            if raw_index > 0:
                prefix = torch.tensor(
                    [[int(token) for token in raw_ids[:raw_index]]],
                    dtype=prompt_ids.dtype,
                    device=prompt_ids.device,
                )
                ids = torch.cat([prompt_ids, prefix], dim=1)
            else:
                ids = prompt_ids
            with self._model_exclusive(
                "session.sae_token_readout called while another model use is in flight",
                phase_msg="session.sae_token_readout called while a generation is in flight",
            ):
                hidden = _capture_all_hidden_states(
                    self._model, self._layers, ids,
                    layer_indices=[layer], pool_index=ids.shape[1] - 1,
                )
                acts = self._encode_sae_hidden(hidden[layer])
                values, indices = torch.topk(acts, k=top_k)
                features = [
                    {
                        "id": int(idx),
                        "activation": float(value),
                        "label": self._sae_label(int(idx)),
                        "max_act": self._sae_max_act(int(idx)),
                    }
                    for value, idx in zip(values.cpu(), indices.cpu())
                ]
        return {
            "node_id": node_id,
            "raw_index": int(raw_index),
            "token_id": int(raw_ids[raw_index]),
            "token_text": str(self._tokenizer.decode([int(raw_ids[raw_index])])),
            "steering": steering_expr,
            "layer": layer,
            "features": features,
        }

    def geometry_token_readout(
        self,
        node_id: str,
        raw_index: int,
        *,
        apply_steering: bool = True,
        raw: bool = False,
    ) -> dict[str, Any]:
        """Whitened subspace-probe readings at the forward that produced a
        loom token — the geometry family's token replay.

        Rebuilds the exact token stream that produced the clicked token
        (the same render :meth:`fork_from_token` and the lens/SAE replays
        use), runs one capture forward at the attached probes' layer
        union, and scores the full Monitor roster at the final position
        via :meth:`Monitor.score_single_token` — so the reading shape
        (coords / fraction / nearest / residual / assignment / membership
        + per-layer traces) matches the live per-token read at that
        position.  Post-hoc by construction: it reads generations that ran
        aggregate-only (live probe scores off) and probes attached after
        the fact, and ``apply_steering=False`` reads the unsteered
        counterfactual.  Same steering-fidelity class as the fork replay:
        an always-active affine term reproduces the original decode
        injections exactly (the slide is position-independent);
        phase-split and probe-gated terms fire as the trigger context's
        rest state decides.  Curved probes score with a cold foot solve
        (warm-start is the sequential live path's optimization; a stale
        foot from a prior generation must not seed a one-shot replay).

        Returns ``{node_id, raw_index, token_id, token_text, steering,
        readings: {probe_name: ProbeReading}}``.  Raises ``ValueError``
        with no geometry probe attached, :class:`UnknownNodeError` /
        :class:`InvalidNodeOperationError` on a bad target (mirrors
        :meth:`fork_from_token`), and ``WhitenerError`` when the whitener
        doesn't cover the probed layers.
        """
        from saklas.core.capture import _capture_all_hidden_states

        if not self._monitor.probe_names:
            raise ValueError(
                "geometry_token_readout: no geometry probes attached — "
                "attach one with session.add_probe(selector) first"
            )
        node = self.tree.get(node_id)
        if node.role not in ("assistant", "user"):
            raise InvalidNodeOperationError(
                f"geometry_token_readout: {node_id!r} is a {node.role} "
                f"node — only a turn with a decode record can be read out"
            )
        raw_ids = node.raw_token_ids
        if not raw_ids:
            raise InvalidNodeOperationError(
                f"geometry_token_readout: {node_id!r} has no raw token "
                f"record (transcript-loaded node)"
            )
        if not 0 <= raw_index < len(raw_ids):
            raise InvalidNodeOperationError(
                f"geometry_token_readout: raw_index {raw_index} out of "
                f"range [0, {len(raw_ids)}) for {node_id!r}"
            )
        recipe = node.recipe
        steering_expr = (
            recipe.steering if (apply_steering and recipe is not None) else None
        )
        # The steering scope opens OUTSIDE the exclusive-GPU guard — the
        # composer's push/pop take ``_gen_lock`` blocking, so nesting it
        # inside ``_model_exclusive`` would self-deadlock (same ordering
        # as the lens/SAE replays and ``score_choices``).
        scope = self.steering(steering_expr) if steering_expr else nullcontext()
        with scope:
            thinking_req = recipe.thinking if recipe is not None else None
            use_thinking = (
                supports_thinking(self._tokenizer)
                if thinking_req is None
                else bool(thinking_req) and supports_thinking(self._tokenizer)
            )
            if raw:
                prompt_ids = self._prepare_input(
                    "", raw=True, parent_node_id=node.parent_id,
                )
            else:
                # Continue-mode rebuild — seat-general, mirrors
                # ``jlens_token_readout``.
                prompt_ids = self._prepare_input(
                    None,
                    thinking=use_thinking,
                    parent_node_id=node.parent_id,
                    user_role=(
                        node.role_label if node.role == "user" else None
                    ),
                    assistant_role=(
                        node.role_label if node.role == "assistant" else None
                    ),
                    gen_seat=node.role,
                )
            if raw_index > 0:
                prefix = torch.tensor(
                    [[int(token) for token in raw_ids[:raw_index]]],
                    dtype=prompt_ids.dtype,
                    device=prompt_ids.device,
                )
                ids = torch.cat([prompt_ids, prefix], dim=1)
            else:
                ids = prompt_ids
            with self._model_exclusive(
                "session.geometry_token_readout called while another model "
                "use is in flight",
                phase_msg="session.geometry_token_readout called while a "
                "generation is in flight",
            ):
                # Repeat the roster check under the exclusive section: a
                # detach winning the gap since the entry check would
                # otherwise surface as a capture-layer implementation error
                # instead of the intended caller-facing message.
                if not self._monitor.probe_names:
                    raise ValueError(
                        "geometry_token_readout: no geometry probes "
                        "attached — attach one with "
                        "session.add_probe(selector) first"
                    )
                layer_idxs = sorted(self._monitor.probe_layers())
                hidden = _capture_all_hidden_states(
                    self._model, self._layers, ids,
                    layer_indices=layer_idxs, pool_index=ids.shape[1] - 1,
                )
                self._monitor.enable_curved_warm(False)
                readings = self._monitor.score_single_token(hidden)
        return {
            "node_id": node_id,
            "raw_index": int(raw_index),
            "token_id": int(raw_ids[raw_index]),
            "token_text": str(self._tokenizer.decode([int(raw_ids[raw_index])])),
            "steering": steering_expr,
            "readings": readings,
        }

    def jspace_decompose(
        self,
        selector: str,
        *,
        k: int = 16,
        layers: "Sequence[int] | str | None" = None,
    ) -> dict[int, tuple[float, list[tuple[str, float]]]]:
        """Split a direction into its J-space component, per layer.

        ``selector`` resolves through the ordinary steering resolver
        (registered profile → fitted 2-node manifold → …), so any steerable
        vector decomposes. Returns ``{layer: (share, [(token, coeff), ...])}``
        — ``share`` is the fraction of the direction's variance carried by
        its best sparse nonnegative combination of ``k`` J-lens vectors
        (the paper's measure of how *verbalizable* the direction is; ~6–15%
        is typical for concept vectors, most of whose variance lies outside
        the workspace).
        """
        from saklas.core.jlens import sparse_nonneg_decompose
        from saklas.core.model import get_unembedding

        lens = self._require_jlens()
        directions = self.ensure_profile_registered(selector)
        unembed = get_unembedding(self._model)
        req = [
            l for l in self._resolve_jlens_layers(lens, layers)
            if l in directions and l in lens.jacobians
        ]
        if not req:
            raise ValueError(
                f"no overlap between {selector!r}'s layers and the fitted "
                f"lens layers"
            )
        out: dict[int, tuple[float, list[tuple[str, float]]]] = {}
        for layer in req:
            dec = sparse_nonneg_decompose(
                directions[layer], lens.jacobians[layer], unembed,
                layer=layer, k=k, atom_norms=lens.atom_norms(layer, unembed),
            )
            out[layer] = (
                dec.share,
                [
                    (str(self._tokenizer.decode([tok])), coeff)
                    for tok, coeff in dec.tokens
                ],
            )
        return out

    # -- Extraction --

    def _run_generator(
        self,
        system_msg: str,
        prompt: str,
        max_new_tokens: int,
        *,
        role: str | None = None,
    ) -> str:
        """Single-turn LLM call shared by scenario and pair generators.

        Builds a chat input from (system_msg, prompt), runs the model
        under inference_mode, decodes and returns the generated text.
        No parsing, no retry — callers drive the retry loop.

        An empty ``system_msg`` (``""``) drops the system role entirely
        — the chat template sees only the user turn.  Useful for
        measuring whether the system framing is biasing statement
        register: with the persona-priming gone, the model writes from
        its default assistant identity rather than the "interpretability
        research output generator" framing the constant has historically
        carried.

        ``role`` (optional): substitute a custom assistant-role label so
        the generation prompt opens with ``<role>`` instead of
        ``assistant``.  Routed through :func:`build_chat_input`.  Mirrors
        the role-augmented extraction path so the scenario / pair
        generators inhabit the same persona the corpus will be pooled
        from.  ``role=None`` is the zero-overhead default.
        """
        pad_id = self._tokenizer.pad_token_id or self._tokenizer.eos_token_id
        messages: list[dict[str, str]] = []
        if system_msg:
            messages.append({"role": "system", "content": system_msg})
        messages.append({"role": "user", "content": prompt})
        model_type_for_role: str | None = None
        if role is not None:
            model_cfg = getattr(self._model, "config", None)
            text_cfg = getattr(model_cfg, "text_config", None) if model_cfg is not None else None
            model_type_for_role = (
                getattr(text_cfg, "model_type", None)
                if text_cfg is not None
                else None
            )
            if model_type_for_role is None and model_cfg is not None:
                model_type_for_role = getattr(model_cfg, "model_type", None)
        input_ids = build_chat_input(
            self._tokenizer, messages, system_prompt=None,
            thinking=False,
            gen_role=role, model_type=model_type_for_role,
            scene=self.scene_grammar,
        ).to(self._device)
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            out = self._model.generate(  # pyright: ignore[reportCallIssue]  # transformers stubs don't expose generate on PreTrainedModel directly
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True, temperature=1.0, top_p=0.9,
                pad_token_id=pad_id,
            )
        new_ids = out[0][input_ids.shape[-1]:]
        decoded = self._tokenizer.decode(new_ids, skip_special_tokens=True)
        return decoded if isinstance(decoded, str) else decoded[0]

    def _run_generator_batch(
        self,
        system_msg: str,
        prompts: list[str],
        max_new_tokens: int,
        *,
        role: str | None = None,
    ) -> list[str]:
        """Batched sibling of :meth:`_run_generator`.

        Renders each prompt as a ``[system?, user]`` chat, **left-pads** the
        batch to a common length (decoder-only generation aligns continuations
        on the right, so every row's prompt must end at the same column), and
        runs ONE ``model.generate`` over the whole batch.  Returns the decoded
        continuations in prompt order.  Sampling is per-row independent
        (``do_sample``), with the same ``temperature=1.0`` / ``top_p=0.9`` the
        single-prompt generator uses, so a batched corpus is distributionally
        the same as the sequential one — just far fewer ``generate`` calls.

        The seam ``generate_responses`` / ``generate_neutral_responses`` call;
        ``_run_generator`` is kept for the single-shot ``ModelHandle`` protocol
        and any caller that wants one response.
        """
        if not prompts:
            return []
        # ``cast`` (not ``int()``): the tokenizer stub types pad/eos as a loose
        # union (``int | list[int] | str | …``) though the id is an int; casting
        # avoids a spurious ``int(list)`` type error without a runtime no-op.
        pad_id = cast(
            int,
            self._tokenizer.pad_token_id
            or self._tokenizer.eos_token_id
            or 0,
        )
        model_type_for_role: str | None = None
        if role is not None:
            model_cfg = getattr(self._model, "config", None)
            text_cfg = getattr(model_cfg, "text_config", None) if model_cfg is not None else None
            model_type_for_role = (
                getattr(text_cfg, "model_type", None)
                if text_cfg is not None
                else None
            )
            if model_type_for_role is None and model_cfg is not None:
                model_type_for_role = getattr(model_cfg, "model_type", None)

        # Render each prompt to ids (build_chat_input returns CPU tensors).
        seqs: list[torch.Tensor] = []
        for prompt in prompts:
            messages: list[dict[str, str]] = []
            if system_msg:
                messages.append({"role": "system", "content": system_msg})
            messages.append({"role": "user", "content": prompt})
            ids = build_chat_input(
                self._tokenizer, messages, system_prompt=None,
                thinking=False,
                gen_role=role, model_type=model_type_for_role,
                scene=self.scene_grammar,
            )
            seqs.append(ids[0])

        lengths = [int(s.shape[0]) for s in seqs]
        max_len = max(lengths)
        batch = len(seqs)
        # Left-pad: each row's prompt occupies the rightmost ``lengths[i]``
        # columns, so the continuation begins at ``max_len`` for every row and
        # ``out[:, max_len:]`` is the batch's generated tail.
        input_ids = torch.full(
            (batch, max_len), pad_id, dtype=seqs[0].dtype,
        )
        attn = torch.zeros((batch, max_len), dtype=torch.long)
        for i, seq in enumerate(seqs):
            input_ids[i, max_len - lengths[i]:] = seq
            attn[i, max_len - lengths[i]:] = 1
        input_ids = input_ids.to(self._device)
        attn = attn.to(self._device)

        with torch.inference_mode():
            out = self._model.generate(  # pyright: ignore[reportCallIssue]  # transformers stubs don't expose generate on PreTrainedModel directly
                input_ids,
                attention_mask=attn,
                max_new_tokens=max_new_tokens,
                do_sample=True, temperature=1.0, top_p=0.9,
                pad_token_id=pad_id,
            )
        new_ids = out[:, max_len:]
        decoded = self._tokenizer.batch_decode(new_ids, skip_special_tokens=True)
        return list(decoded)

    def generate_responses(
        self,
        concepts: list[str],
        kinds: list[str | None],
        *,
        roles: dict[str, str | None] | None = None,
        custom_system: str | None = None,
        samples_per_prompt: int = 1,
        max_new_tokens: int = _RESPONSE_MAX_TOKENS,
        on_progress: Callable[[str], None] | None = None,
    ) -> dict[str, list[str]]:
        """Generate each concept\'s conversational corpus (4.0 / A2 elicitation).

        For every ``(concept, kind)`` the model answers the shared baseline
        prompts *in character* -- the concept rides the system prompt
        (:func:`_system_for`, led by the shared :data:`~saklas.core.capture._LENGTH_DIRECTIVE`
        so responses stay one short paragraph) and the swapped assistant-role
        label (:func:`_role_for`, overridable per concept via ``roles``).  The
        length directive is common-mode with the neutral corpus and with capture
        (it leads every system prompt), so it cancels at extraction.  Responses
        are emitted samples-outer / prompts-inner so ``response[i]`` aligns to
        ``prompt[i % k]`` -- the alignment :func:`compute_node_centroid` and the
        node corpus files assume.

        ``custom_system`` is the system template used for any concept whose
        ``kind`` is ``"custom"`` (``{c}`` = the humanized concept) -- the
        free-form elicitation frame, e.g. ``"You are the month of {c}; speak as
        that month."``.  ``custom`` kinds do **not** role-swap (role is ``None``,
        the framing rides the system prompt and the corpus pools in
        standard-assistant space), so ``custom`` works on every family including
        those without a substitutable role header.

        Returns ``{concept: [response, ...]}`` with
        ``len == max(1, samples_per_prompt) * len(baseline_prompts)`` per
        concept.  ``abstract``/``concrete`` role-swap (a family without role
        support raises ``RoleSubstitutionUnsupportedError`` at generation);
        ``custom`` is the system-only exception.
        """
        from saklas.core.capture import _LENGTH_DIRECTIVE, _load_baseline_prompts

        if any((k or "abstract") == "custom" for k in kinds) and not custom_system:
            raise ValueError(
                "generate_responses: kind='custom' requires custom_system "
                "(a system template with a {c} placeholder)"
            )
        prompts = _load_baseline_prompts()
        roles = roles or {}
        reps = max(1, samples_per_prompt)
        total = reps * len(prompts)
        out: dict[str, list[str]] = {}
        for concept, kind in zip(concepts, kinds, strict=True):
            concept_h = _humanize_concept(concept)
            # The length directive leads the persona system prompt (and is the
            # whole system for the neutral baseline + capture), so it is shared
            # common-mode that cancels at extraction.
            system = f"{_LENGTH_DIRECTIVE} {_system_for(concept_h, kind, custom_system)}"
            gen_role = roles.get(concept) or _role_for(_slug(concept), kind)
            responses: list[str] = []
            for _ in range(reps):
                # Prompts-inner, batched in chunks: one model.generate per chunk
                # rather than per prompt.  Order is preserved within the rep, so
                # ``response[i] -> prompt[i % k]`` still holds.
                for start in range(0, len(prompts), _CORPUS_GEN_BATCH):
                    chunk = prompts[start:start + _CORPUS_GEN_BATCH]
                    if on_progress:
                        on_progress(
                            f"Generating {concept!r} responses "
                            f"{len(responses) + 1}-{len(responses) + len(chunk)}"
                            f"/{total}..."
                        )
                    texts = self._run_generator_batch(
                        system, chunk, max_new_tokens, role=gen_role,
                    )
                    responses.extend(text.strip() for text in texts)
            out[concept] = responses
        return out

    def generate_neutral_responses(
        self,
        *,
        samples_per_prompt: int = 1,
        max_new_tokens: int = _RESPONSE_MAX_TOKENS,
        on_progress: Callable[[str], None] | None = None,
    ) -> list[str]:
        """Generate the neutral baseline corpus -- organic, unconditioned responses.

        The neutral counterpart to :meth:`generate_responses`: the model answers
        the shared baseline prompts under the shared
        :data:`~saklas.core.capture._LENGTH_DIRECTIVE` as its *only* system prompt
        (no persona) with the standard assistant label, so the corpus is the
        model\'s own voice -- just brief.  The directive is the only framing it
        shares with the node corpora (same on every node system + every capture),
        so it cancels at extraction while leaving neutral the default-voice
        reference the contrast subtracts against.  Same prompts-cycled order
        (``response[i] -> prompt[i % k]``).  This is what ``neutral_statements.json``
        holds under 4.0 -- regenerate it with this and the per-model
        ``layer_means`` / neutral-activation caches recompute conversationally.
        """
        from saklas.core.capture import _LENGTH_DIRECTIVE, _load_baseline_prompts

        prompts = _load_baseline_prompts()
        reps = max(1, samples_per_prompt)
        total = reps * len(prompts)
        responses: list[str] = []
        for _ in range(reps):
            for start in range(0, len(prompts), _CORPUS_GEN_BATCH):
                chunk = prompts[start:start + _CORPUS_GEN_BATCH]
                if on_progress:
                    on_progress(
                        f"Generating neutral responses "
                        f"{len(responses) + 1}-{len(responses) + len(chunk)}"
                        f"/{total}..."
                    )
                texts = self._run_generator_batch(
                    _LENGTH_DIRECTIVE, chunk, max_new_tokens, role=None,
                )
                responses.extend(text.strip() for text in texts)
        return responses

    @contextmanager
    def _model_exclusive(
        self,
        busy_msg: str,
        *,
        phase_msg: str | None = None,
        exc: type[Exception] = ConcurrentExtractionError,
    ) -> Iterator[None]:
        """Hold the exclusive-GPU ``_gen_lock`` for a model op.

        Acquires non-blocking — a cross-thread model op in flight raises
        ``exc(busy_msg)`` rather than races PyTorch's single global MPS
        command buffer (which would abort the process).  When ``phase_msg``
        is given, also guards ``_gen_phase is IDLE`` (raises ``exc(phase_msg)``
        on a generation in flight); ``add_probe`` passes ``None`` to skip the
        phase guard.  Releases in ``finally``.
        """
        if not self._gen_lock.acquire(blocking=False):
            raise exc(busy_msg)
        try:
            if phase_msg is not None and self._gen_phase is not GenState.IDLE:
                raise exc(phase_msg)
            yield
        finally:
            self._gen_lock.release()

    def _assert_unsteered_artifact_operation(self) -> None:
        """Refuse persisted neutral/fit work under active steering hooks."""
        if self._steering_composer._stack:
            raise ConcurrentExtractionError(
                "artifact-producing model operations cannot run inside an "
                "active steering scope; exit session.steering() first"
            )

    def extract(
        self,
        concept: str,
        baseline: str | None = None,
        *,
        kind: str = "abstract",
        force: bool = False,
        on_progress: Callable[[str], None] | None = None,
        sae: str | None = None,
        sae_revision: str | None = None,
        namespace: str | None = None,
        role: str | None = None,
    ) -> tuple[str, Profile]:
        """Author + fit a steering vector via conversational (4.0 / A2) extraction.

        A steering vector is a flat ``pca`` manifold.  A **bipolar** axis
        (``concept`` + ``baseline``) authors a 2-node manifold — node 0 the
        positive pole, node 1 the negative.  A **monopolar** concept
        (``baseline=None``) authors a genuinely **1-node** manifold; the fit
        recognizes the single-node ``pca`` shape and folds ``concept − ν``
        (ν = the model's neutral activation mean, ``layer_means``) into a
        1-node neutral-anchored ray — neutral is the implicit negative pole,
        sourced per-model at fit (never a stored corpus).  Both resolve as
        steerable vectors: ``0.5 <concept>`` steers toward the concept pole.

        Each node's corpus is generated by :meth:`generate_responses`: the model
        answers the shared baseline prompts *in character* (concept in the system
        prompt + a kind-derived elicitation role), and extraction pools the
        swapped-back ``[user, assistant]`` pairs in standard-assistant space.

        Returns ``(canonical_name, Profile)`` — the folded per-layer direction
        view of the fitted manifold (the in-memory steering-vector shape), with
        the manifold the single on-disk artifact.  Emits ``ManifoldExtracted``
        (with its ``profile`` field carrying the folded view).

        Flags:

        - ``kind=`` (``"abstract"`` | ``"concrete"``): selects each node's system
          template + elicitation role label (``someone {c}`` vs ``{c}``).
          Applies to both poles of a bipolar axis.
        - ``force=True``: regenerate the corpora + re-author the folder
          (otherwise an existing manifold's corpus is reused and the fit
          cache-hits when a tensor for this model is already present).
        - ``role=<slug>``: persona-baselined extraction — the explicit role
          overrides the kind-derived label at *both* generation and capture
          (the centroid lives in role-baselined space; stored as the node
          ``role``).  Omit for the standard swap-back baseline.
        - ``sae=<release>``: fit in SAE feature space (a ``_sae-<release>``
          tensor variant beside the raw one).

        Gated against generation: fitting runs forward passes through the
        model and would race an active gen.
        """
        with self._model_exclusive(
            "session.extract called while another model use is in flight",
            phase_msg="session.extract called while a generation is in flight",
        ):
            SaklasSession._assert_unsteered_artifact_operation(self)
            from saklas.io.paths import manifold_dir

            ns = namespace or "local"
            pos_raw, neg_raw = _split_composite_source(concept, baseline)
            name = canonical_concept_name(concept, baseline)
            pos_label = _slug(pos_raw)
            if neg_raw is None:
                # Monopolar = a single concept node.  The folder is genuinely
                # one node; the engine recognizes a 1-node ``pca`` fit and folds
                # ``concept − ν`` (ν = the model's neutral activation mean,
                # ``layer_means``) into a 1-node neutral-anchored ray
                # (`ManifoldExtractionPipeline`).  ν is the implicit negative
                # pole, sourced per-model at fit — never a stored corpus — so
                # ``0.5 <concept>`` steers neutral → concept.
                gen_concepts = [pos_raw]
                labels = [pos_label]
                desc = f"Monopolar axis: {pos_raw} (+) vs neutral baseline (-)."
            else:
                gen_concepts = [pos_raw, neg_raw]
                labels = [pos_label, _slug(neg_raw)]
                desc = f"Bipolar axis: {pos_raw} (+) vs {neg_raw} (-)."

            folder = manifold_dir(ns, name)
            manifest = folder / "manifold.json"
            if manifest.exists() and not force:
                from saklas.io.manifold_folder import ManifoldFolder

                existing = ManifoldFolder.load(
                    folder, verify_manifest=False,
                )._roles_padded()
                if not existing or not all(value == role for value in existing):
                    raise ValueError(
                        f"manifold {ns}/{name} already carries role baseline "
                        f"{existing!r}, not uniformly {role!r}; pass force=True "
                        "to regenerate its corpus for the requested baseline"
                    )
            node_corpora: dict[str, list[str]] | None = None
            if force or not manifest.exists():
                gen_roles: dict[str, str | None] | None = (
                    {c: role for c in gen_concepts} if role else None
                )
                corpora = self.generate_responses(
                    gen_concepts, [kind] * len(gen_concepts),
                    roles=gen_roles, on_progress=on_progress,
                )
                node_corpora = {
                    label: corpora[c]
                    for label, c in zip(labels, gen_concepts, strict=True)
                }
            node_kinds: dict[str, str | None] = {
                label: kind for label in labels
            }
            return self._author_and_fit_2node(
                ns, name, desc, node_corpora=node_corpora, node_kinds=node_kinds,
                role=role, force=force, sae=sae, sae_revision=sae_revision,
                on_progress=on_progress,
            )

    def extract_from_corpora(
        self,
        name: str,
        positive: list[str],
        negative: list[str],
        *,
        kind: str = "abstract",
        namespace: str | None = None,
        role: str | None = None,
        sae: str | None = None,
        sae_revision: str | None = None,
        on_progress: Callable[[str], None] | None = None,
        force: bool = False,
        description: str = "",
    ) -> tuple[str, Profile]:
        """Author + fit a steering vector from two ready-made pole corpora.

        The corpus-in sibling of :meth:`extract` — used by persona cloning and
        hand-authored HTTP/library paths, which already hold the positive and
        negative corpora and so skip generation. Authors a 2-node ``pca``
        manifold (``positive`` → pole node, ``negative`` → its opposite) and
        fits it; returns ``(canonical_name, Profile)`` like :meth:`extract`.

        Under 4.0 the corpora are pooled conversationally — each entry is treated
        as a response to the shared baseline prompts (``response[i] -> prompt[i
        % k]``), so each corpus length must be a multiple of the baseline prompt
        set.  ``kind`` is recorded per node (provenance); ``role`` opts into a
        persona-baselined fit as in :meth:`extract`.
        """
        with self._model_exclusive(
            "session.extract_from_corpora called while another "
            "model use is in flight",
            phase_msg="session.extract_from_corpora called while a "
            "generation is in flight",
        ):
            SaklasSession._assert_unsteered_artifact_operation(self)
            ns = namespace or "local"
            pos_raw, neg_raw = _split_composite_source(name, None)
            canonical = canonical_concept_name(name)
            pos_label = _slug(pos_raw)
            neg_label = _slug(neg_raw) if neg_raw is not None else f"{pos_label}_neg"
            return self._author_and_fit_2node(
                ns, canonical, description,
                node_corpora={pos_label: positive, neg_label: negative},
                node_kinds={pos_label: kind, neg_label: kind},
                role=role, force=force, sae=sae, sae_revision=sae_revision,
                on_progress=on_progress,
            )

    def _author_and_fit_2node(
        self,
        ns: str,
        name: str,
        description: str,
        *,
        node_corpora: dict[str, list[str]] | None,
        node_kinds: dict[str, str | None],
        role: str | None,
        force: bool,
        sae: str | None,
        sae_revision: str | None,
        on_progress: Callable[[str], None] | None,
    ) -> tuple[str, Profile]:
        """Author a 2-node ``pca`` folder (when stale) + fit it — the shared tail
        of :meth:`extract` / :meth:`extract_from_corpora`.

        Both callers differ only in their corpus source (generate vs given); by
        the time they reach here they hold ``node_corpora`` (label → corpus) and
        ``node_kinds`` (label → kind).  This rmtrees + re-authors the folder when
        ``force`` or the manifest is missing (``node_corpora`` is consulted only
        then, so a cache-hit caller may pass ``None``), then delegates to
        :meth:`_fit_concept_manifold`.  Runs with ``_gen_lock`` already held by
        the caller.
        """
        from saklas.io.manifolds import create_discover_manifold_folder
        from saklas.io.manifold_folder import ManifoldFolder
        from saklas.io.paths import manifold_dir

        if sae is not None and role is not None:
            raise ValueError(
                "SAE-backed and role-baselined extraction are mutually exclusive"
            )

        folder = manifold_dir(ns, name)
        manifest = folder / "manifold.json"
        from saklas.io.manifold_folder import _locked_manifest

        with _locked_manifest(folder):
            if manifest.exists() and not force:
                existing = ManifoldFolder.load(
                    folder, verify_manifest=False,
                )._roles_padded()
                if not existing or not all(value == role for value in existing):
                    raise ValueError(
                        f"manifold {ns}/{name} already carries role baseline "
                        f"{existing!r}, not uniformly {role!r}; pass "
                        "force=True to replace its corpus"
                    )
            if force or not manifest.exists():
                assert node_corpora is not None
                node_roles: dict[str, str | None] | None = (
                    {label: role for label in node_corpora} if role else None
                )
                if folder.exists():
                    from saklas.io.manifold_folder import reset_manifold_folder

                    reset_manifold_folder(folder)
                create_discover_manifold_folder(
                    ns, name, description, fit_mode="pca",
                    node_corpora=node_corpora,
                    hyperparams={"max_dim": 1, "var_threshold": 0.7},
                    node_roles=node_roles, node_kinds=node_kinds,
                )
        return self._fit_concept_manifold(
            name, folder, sae=sae, sae_revision=sae_revision,
            role=role, on_progress=on_progress,
        )

    def _fit_concept_manifold(
        self,
        name: str,
        folder: Any,
        *,
        sae: str | None,
        sae_revision: str | None,
        role: str | None,
        on_progress: Callable[[str], None] | None,
    ) -> tuple[str, Profile]:
        """Fit a 2-node pca manifold (lock held) and return its folded Profile.

        Shared tail of :meth:`extract` / :meth:`extract_from_corpora`:
        runs :class:`ManifoldExtractionPipeline` directly (the public
        :meth:`fit` re-acquires ``_gen_lock``, which the callers
        already hold), folds the fitted manifold to a per-layer direction
        :class:`Profile`, and emits a single ``ManifoldExtracted`` carrying that
        folded profile (the pipeline's own emission is suppressed).

        The returned name carries the variant tail (``:sae-<release>`` /
        ``:role-<slug>``) so the caller steers the right per-model tensor; the
        on-disk manifold folder stays the bare canonical name.
        """
        from saklas.core.extraction import ManifoldExtractionPipeline
        from saklas.core.capture import folded_directions

        pipe = ManifoldExtractionPipeline(self, self.events)
        # Suppress the pipeline's own ``ManifoldExtracted`` so the vector path
        # fires exactly one event: the enriched emit below carries the folded
        # ``Profile`` and the variant-tailed name.
        manifold = pipe.fit(
            folder, sae=sae, sae_revision=sae_revision, on_progress=on_progress,
            emit_event=False,
        )
        self._adopt_fitted_manifold(folder, manifold)
        if sae:
            ret_name = f"{name}:sae-{sae}"
        elif role:
            ret_name = f"{name}:role-{role}"
        else:
            ret_name = name
        profile = Profile(
            folded_directions(manifold),
            metadata={
                "method": "manifold_pca",
                "name": ret_name,
                "share_metric": manifold.metadata.get("share_metric"),
            },
        )
        self.events.emit(ManifoldExtracted(
            name=ret_name, manifold=manifold,
            metadata=dict(manifold.metadata), profile=profile,
        ))
        if ret_name in self._profiles:
            self._profiles[ret_name] = self._promote_profile(profile.as_dict())
        return ret_name, profile

    def _adopt_fitted_manifold(self, folder: Any, manifold: Manifold) -> None:
        """Refresh every in-memory consumer of a newly fitted artifact.

        A fit replaces an on-disk tensor under an existing selector. Loaded
        manifolds, folded profiles, attached monitor factors, and cached prefix
        K/V must therefore move together; keeping any one of them would mix old
        geometry with the new fit in the same session.
        """
        from pathlib import Path
        from saklas.core.capture import (
            folded_directions,
            is_foldable_vector_manifold,
        )

        folder_path = Path(folder)
        qualified = f"{folder_path.parent.name}/{folder_path.name}"
        artifact_names = {folder_path.name, qualified}

        def _matches_key(key: str) -> bool:
            head, variant = (
                key.rsplit(":", 1) if ":" in key else (key, "raw")
            )
            if head not in artifact_names:
                return False
            if variant.startswith("sae-"):
                feature = variant
            elif variant == "raw" or variant.startswith("role-"):
                feature = "raw"
            else:
                # Cross-model transfers and other derived variants are
                # independent artifacts, not aliases of this fitted tensor.
                return False
            return feature == manifold.feature_space

        old_objects: set[int] = set()
        matching_keys: list[str] = []
        for key, loaded in list(self._manifolds.items()):
            if _matches_key(key):
                matching_keys.append(key)
                old_objects.add(id(loaded))

        promoted = None
        if matching_keys:
            promoted = manifold.to(device=self._device, dtype=torch.float32)
            for key in matching_keys:
                self._manifolds[key] = promoted

        # Rebuild attach-time whitened factors for probes backed by an evicted
        # object. Preserve public probe name and nearest-node roster size.
        # ``_monitor`` doesn't exist yet when a fit runs from the constructor's
        # probe bootstrap (``_bootstrap_manifold_probes`` re-fits a stale
        # tagged axis in-session before the Monitor is built) — there are no
        # attached probes to refresh at that point; the fresh manifold reaches
        # the Monitor through its constructor roster instead.
        monitor = getattr(self, "_monitor", None)
        if monitor is not None:
            # Geometry-state boundary: this walk mutates the Monitor roster
            # outside the instrument's attach/detach, so the remove+add
            # pair must land atomically against the un-locked coherent
            # readers (session.probes, specs, plan).  The fit path already
            # holds ``_gen_lock`` (outer) — state_lock is a leaf.
            with self._geometry_instrument.state_lock:
                for probe_name, attached in monitor.attached_probes().items():
                    if id(attached.manifold) not in old_objects:
                        continue
                    assert promoted is not None
                    top_n = attached.top_n
                    monitor.remove_probe(probe_name)
                    monitor.add_probe(probe_name, promoted, top_n=top_n)
                    self._probe_hash_cache.pop(probe_name, None)

        matching_profiles = [key for key in self._profiles if _matches_key(key)]
        if matching_profiles and is_foldable_vector_manifold(manifold):
            folded = self._promote_profile(folded_directions(manifold))
            for key in matching_profiles:
                self._profiles[key] = dict(folded)
        else:
            for key in matching_profiles:
                self._profiles.pop(key, None)

        self._invalidate_prefix_cache()
        self._invalidate_analytics_cache()

    def _evict_failed_manifold_override(
        self, folder: Any, *, sae: str | None,
    ) -> None:
        """Drop consumers whose on-disk manifest changed before a failed fit."""
        from pathlib import Path

        folder_path = Path(folder)
        artifact_names = {
            folder_path.name,
            f"{folder_path.parent.name}/{folder_path.name}",
        }
        target_feature = f"sae-{sae}" if sae is not None else "raw"

        def _matches_key(key: str) -> bool:
            head, variant = (
                key.rsplit(":", 1) if ":" in key else (key, "raw")
            )
            if head not in artifact_names:
                return False
            feature = (
                variant if variant.startswith("sae-")
                else "raw" if variant == "raw" or variant.startswith("role-")
                else None
            )
            return feature == target_feature

        old_objects = {
            id(manifold)
            for key, manifold in self._manifolds.items()
            if _matches_key(key)
        }
        for key in [key for key in self._manifolds if _matches_key(key)]:
            self._manifolds.pop(key, None)
        for key in [key for key in self._profiles if _matches_key(key)]:
            self._profiles.pop(key, None)

        monitor = getattr(self, "_monitor", None)
        if monitor is not None:
            # Geometry-state boundary — same contract as the promotion
            # walk in ``_adopt_fitted_manifold``.
            with self._geometry_instrument.state_lock:
                for probe_name, attached in list(
                    monitor.attached_probes().items()
                ):
                    if id(attached.manifold) in old_objects:
                        monitor.remove_probe(probe_name)
                        self._probe_hash_cache.pop(probe_name, None)

        self._invalidate_prefix_cache()
        self._invalidate_analytics_cache()

    def fit(
        self,
        folder: Any,
        *,
        sae: str | None = None,
        sae_revision: str | None = None,
        layers: "Sequence[int] | str | None" = None,
        fit_mode: str | None = None,
        hyperparams: Mapping[str, object] | None = None,
        force: bool = False,
        on_progress: Callable[[str], None] | None = None,
    ) -> Manifold:
        """Fit a steering manifold from a manifold folder (authored or discover).

        Thin delegate to :class:`ManifoldExtractionPipeline` — that
        pipeline owns corpus loading, per-node centroid pooling, the
        per-layer PCA + spline fit (dispatching on the folder's ``fit_mode``),
        and the cache short-circuit.  The Python mirror of CLI ``manifold fit``.
        Gated against generation like :meth:`extract`: manifold fitting runs
        forward passes through the model.  ``force=True`` bypasses the per-model
        tensor cache and re-pools/re-fits unconditionally (CLI ``-f/--force``).
        ``layers`` optionally restricts fitting to explicit transformer indices
        or the 40–90% ``"workspace"`` band.
        ``fit_mode`` and ``hyperparams`` are discover-folder overrides applied
        under the same manifest lock as cache-key derivation and publication.
        """
        with self._model_exclusive(
            "session.fit called while another model use is in flight",
            phase_msg="session.fit called while a generation is in flight",
        ):
            SaklasSession._assert_unsteered_artifact_operation(self)
            manifest_before: bytes | None = None
            if fit_mode is not None or hyperparams is not None:
                from pathlib import Path

                try:
                    manifest_before = (Path(folder) / "manifold.json").read_bytes()
                except OSError:
                    pass
            try:
                from saklas.core.extraction import ManifoldExtractionPipeline
                pipe = ManifoldExtractionPipeline(self, self.events)
                manifold = pipe.fit(
                    folder, sae=sae, sae_revision=sae_revision,
                    layer_indices=layers, fit_mode=fit_mode,
                    hyperparams=hyperparams, force=force,
                    on_progress=on_progress,
                )
                self._adopt_fitted_manifold(folder, manifold)
                return manifold
            except BaseException:
                if manifest_before is not None:
                    from pathlib import Path

                    try:
                        manifest_changed = (
                            (Path(folder) / "manifold.json").read_bytes()
                            != manifest_before
                        )
                    except OSError:
                        manifest_changed = False
                    if manifest_changed:
                        self._evict_failed_manifold_override(folder, sae=sae)
                raise
            finally:
                # A re-fit changes the folded directions any probe reads from.
                self._invalidate_analytics_cache()

    def bake(
        self,
        name: str,
        expression: str,
        *,
        force: bool = True,
        strict: bool = False,
    ) -> tuple[str, Profile]:
        """Bake additive steering terms into a corpus-less manifold.

        The Python mirror of CLI ``manifold bake``.  Wraps
        :func:`saklas.io.bake.merge_into_manifold`, model-scoped to this
        session's loaded model — the merge lands a corpus-less
        ``fit_mode="baked"`` manifold under ``local/<name>/`` — then folds the
        fitted tensor back to a steering :class:`Profile` and registers it
        (:meth:`steer`) so it is immediately steerable.  Returns
        ``(name, Profile)``, the same shape :meth:`extract` returns.

        Dynamic terms and Mahalanobis projection operators are rejected because
        the offline merge path does not carry an identity-matched whitener.
        """
        from saklas.io.bake import MergeError, merge_into_manifold
        from saklas.io.paths import tensor_filename
        from saklas.io.manifold_tensors import load_manifold
        from saklas.core.model import loaded_model_fingerprint
        from saklas.core.capture import folded_directions

        live_fingerprint = loaded_model_fingerprint(self._model, self.model_id)
        dst_folder = merge_into_manifold(
            name, expression, self.model_id, force=force, strict=strict,
            expected_model_fingerprint=live_fingerprint,
        )
        tensor_path = dst_folder / tensor_filename(self.model_id)
        if not tensor_path.is_file():
            raise MergeError(
                f"bake produced no tensor for {self.model_id} at {tensor_path}"
            )
        manifold = load_manifold(str(tensor_path))
        profile = Profile(folded_directions(manifold))
        self.steer(name, profile)
        return name, profile

    def load_profile(self, path: str) -> Profile:
        profile, meta = _load_profile(path)
        promoted = self._promote_profile(profile)
        return Profile(promoted, metadata=meta)

    def save_profile(
        self,
        profile: Profile,
        path: str,
        metadata: dict[str, Any] | None = None,
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
        self._invalidate_analytics_cache()

    def unsteer(self, name: str) -> None:
        """Remove a steering vector from the registry."""
        self._profiles.pop(name, None)
        self._invalidate_prefix_cache()
        self._invalidate_analytics_cache()

    def steering(
        self, value: "str | Steering",
    ) -> "_SteeringContext":
        """Context manager applying steering for the duration of a with-block.

        ``value`` is either a steering expression string (parsed through
        the shared grammar in :mod:`saklas.core.steering_expr`) or a
        pre-built :class:`Steering`.  Dict inputs are not accepted; build
        :class:`Steering` directly if you need typed construction.

        Pole aliases (``io.selectors.resolve_bare_atom``) resolve at parse
        time; this is the canonical resolver site — CLI and server route through
        here. Nesting flattens: an inner
        ``steering("0.5 angry.calm")`` overrides the outer
        ``steering("0.3 angry.calm")`` for the duration of the inner
        scope, and the outer entry is restored on ``__exit__``.  One hook
        installation per active layer regardless of nesting depth.

        Unknown profile names raise ``ProfileNotRegisteredError``; genuinely
        ambiguous pole names propagate ``AmbiguousSelectorError``.
        """
        steering_obj = Steering.from_value(
            value, profile_names=set(self._profiles),
        )
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
        composer = self._steering_composer
        snapshots = composer.materialize_projections(steering_obj)
        raw_entries = steering_obj.normalized_entries()
        resolved: dict[str, SteeringStackEntry] = dict(
            composer.resolve_pole_aliases(raw_entries)
        )

        # Role-augmented extraction (role-extraction Phase 7):
        # collect the set of ``role-<id>`` variants across every resolved
        # term so the generation surface can substitute the assistant-
        # role label.  Unanimity is the only coherent regime — mixing
        # multiple roles in one scope has no defined semantics.  A plain
        # term mixed with a role-tagged term is allowed but warns: the
        # plain term's baseline was ``assistant``, not ``<role>``.
        role_variants: set[str] = set()
        any_plain_term = False
        for key in resolved:
            if ":" not in key:
                any_plain_term = True
                continue
            variant = key.rsplit(":", 1)[1]
            if variant.startswith("role-"):
                role_variants.add(variant[len("role-"):])
            else:
                # Non-role variant (sae, sae-<rel>) — treat as plain for
                # baseline-mismatch tracking.  Its baseline is also the
                # standard assistant role, distinct from the role
                # substitution path.
                any_plain_term = True
        if len(role_variants) > 1:
            from saklas.core.steering_expr import SteeringExprError
            raise SteeringExprError(
                f"conflicting roles in expression: {sorted(role_variants)}; "
                f"all role-augmented terms must agree on role"
            )
        active_role: str | None = next(iter(role_variants), None)
        if active_role is not None and any_plain_term:
            import warnings as _warnings
            from saklas.core.errors import RoleBaselineMismatchWarning
            _warnings.warn(
                f"steering scope mixes plain terms with role-augmented "
                f"terms (role={active_role!r}); the plain term's "
                f"extraction baseline was the standard assistant role, "
                f"not {active_role!r}",
                RoleBaselineMismatchWarning,
                stacklevel=2,
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
                # Resolve via the unified fitted-manifold path. A miss or
                # non-raw variant error surfaces at hook-install with the
                # shared ProfileNotRegisteredError shape.
                with suppress(Exception):
                    self.ensure_profile_registered(target, role="ablation")
            resolved[key] = val
        # Fold in manifold terms.  Like ablation, ``normalized_entries``
        # strips ``ManifoldTerm`` values, so walk ``alphas`` directly.
        # Keys carry the ``<manifold>%<pos>`` form — disjoint from
        # plain/projection/ablation keys.  The manifold artifact is
        # loaded into ``self._manifolds`` here so a miss surfaces
        # eagerly (``ManifoldNotRegisteredError``) rather than at hook
        # install.
        manifold_terms: list[ManifoldTerm] = []
        for key, val in steering_obj.alphas.items():
            if not isinstance(val, ManifoldTerm):
                continue
            self.ensure_manifold_loaded(val.manifold)
            resolved[key] = val
            manifold_terms.append(val)

        # Role-paired manifold steering (Phase A.4): each manifold term
        # implies a role via its nearest-node lookup.  Aggregate across
        # all manifold terms by highest absolute coefficient (Phase A.4
        # disagreement policy: soft-warn + highest-coeff wins).  An
        # explicit ``:role-<X>`` term wins over any manifold-implied
        # role — explicit user intent dominates implicit lookup — with
        # a soft warn if they disagreed.  ``None``-roles (nodes opting
        # out of substitution) abstain from the aggregate; they don't
        # vote for the standard baseline against an explicit role.
        manifold_role: str | None = None
        manifold_role_coeff = -1.0
        manifold_role_disagreement = False
        for term in manifold_terms:
            manifold = self._manifolds.get(term.manifold)
            if manifold is None:
                # Should not happen — ``ensure_manifold_loaded`` either
                # populates ``self._manifolds`` or raises.  Defensive.
                continue
            try:
                implied = manifold.nearest_node_role(term.position)
            except ValueError:
                implied = None
            if implied is None:
                continue
            abs_c = abs(term.coeff)
            if manifold_role is None:
                manifold_role = implied
                manifold_role_coeff = abs_c
            elif implied == manifold_role:
                # Same role — pick the larger coefficient as tiebreaker
                # so the warning copy below carries the dominant term.
                manifold_role_coeff = max(manifold_role_coeff, abs_c)
            else:
                manifold_role_disagreement = True
                if abs_c > manifold_role_coeff:
                    manifold_role = implied
                    manifold_role_coeff = abs_c
        if manifold_role_disagreement:
            import warnings as _warnings
            from saklas.core.errors import RoleBaselineMismatchWarning
            _warnings.warn(
                f"steering scope has manifold terms implying distinct "
                f"per-node roles; highest-coefficient term wins "
                f"(role={manifold_role!r}, |coeff|={manifold_role_coeff:.3f}). "
                f"Compose with a single manifold term or align the active "
                f"nodes if you want a different role.",
                RoleBaselineMismatchWarning,
                stacklevel=2,
            )

        # Resolve the final active role across both tiers:
        # explicit ``:role-<X>`` (already aggregated as ``active_role``
        # above) wins over any manifold-implied role.  Soft-warn when
        # the two tiers disagreed — the user typed an explicit role and
        # also has a manifold term pulling toward a different persona.
        if active_role is None:
            active_role = manifold_role
        elif manifold_role is not None and manifold_role != active_role:
            import warnings as _warnings
            from saklas.core.errors import RoleBaselineMismatchWarning
            _warnings.warn(
                f"steering scope has an explicit role={active_role!r} "
                f"and a manifold term implying role={manifold_role!r}; "
                f"explicit role wins, manifold-implied role ignored. "
                f"Use the same role on both, or drop the explicit "
                f":role-{active_role} variant to let the manifold drive.",
                RoleBaselineMismatchWarning,
                stacklevel=2,
            )
        return _SteeringContext(
            self, resolved,
            synthetic_snapshots=snapshots,
            active_role=active_role,
        )

    def _bootstrap_manifold_probes(
        self, categories: list[str], *, include_fitted_defaults: bool = False,
    ) -> dict[str, "Manifold"]:
        """Source the default probe roster as fitted bundled manifolds.

        One pass, two tiers:

        - **Tagged concept axes** — for each bundled 2-node ``pca`` manifold
          tagged in a requested category (roster from
          :func:`~saklas.io.probes_bootstrap.load_default_manifolds`) fit-or-
          load the per-model subspace (eager, same cost band as the legacy DiM
          extraction — both run forward passes, both disk-cache) and hand the
          flat :class:`Manifold` to the :class:`Monitor`, which reads it as a
          coordinate (the rank-1 case of the subspace readout).  Registered
          under the **bare** name (``confident.uncertain``) so the gate grammar
          and trait panel key off it.
        - **Fitted multi-node defaults** (``include_fitted_defaults``, the
          ``probes=None`` default roster) — sweep every bundled ``default/``
          manifold and additionally attach any that is *already fitted* for the
          loaded model and not already attached (``personas`` / ``emotions``).
          Attach-only: fitting a 107-node manifold runs a forward pass per node
          and would block startup for minutes, so an unfitted one is skipped
          with a one-line log (fit it and it auto-loads next launch).
          Registered under the qualified ``default/<name>`` selector so a manual
          attach from the manifolds drawer matches — no duplicate rows.  This
          folds the former serve-only ``_attach_default_manifold_probes`` into
          the construction-time pass so every frontend gets the same roster.

        A fit/load failure for one manifold is logged and skipped, never fatal
        to session construction.
        """
        from saklas.io.probes_bootstrap import load_default_manifolds
        from saklas.io.paths import manifold_dir, safe_model_id

        defaults = load_default_manifolds()
        probes: dict[str, "Manifold"] = {}
        seen: set[str] = set()
        for cat in categories:
            for name in defaults.get(cat, []):
                if name in seen:
                    continue
                seen.add(name)
                key = f"default/{name}"
                try:
                    try:
                        self.ensure_manifold_loaded(key)
                    except ManifoldNotRegisteredError:
                        self.fit(manifold_dir("default", name))
                        self.ensure_manifold_loaded(key)
                    probes[name] = self._manifolds[key]
                except Exception as e:
                    _log.warning("manifold probe '%s' failed to fit/load: %s", name, e)

        if not include_fitted_defaults:
            return probes

        # Attach-only sweep: every fitted bundled multi-node manifold that the
        # tagged tier didn't already pick up (personas / emotions carry no
        # category tag, so they fall through to here).
        from saklas.io.manifolds import (
            ManifoldFolder,
            ManifoldFormatError,
            bundled_manifold_names,
        )

        stem = safe_model_id(self.model_id)
        for name in bundled_manifold_names():
            key = f"default/{name}"
            if name in probes or key in probes:
                continue
            try:
                # Attach preflight needs fitted stems only.  The selected
                # model pair is strictly verified by ensure_manifold_loaded.
                folder = ManifoldFolder.load(
                    manifold_dir("default", name), verify_manifest=False,
                )
            except (ManifoldFormatError, FileNotFoundError):
                continue
            if stem not in folder.tensor_models():
                _log.info(
                    "manifold probe '%s' not fitted for %s — skipping "
                    "(fit it to auto-attach next launch)",
                    key, self.model_id,
                )
                continue
            try:
                self.ensure_manifold_loaded(key)
                probes[key] = self._manifolds[key]
            except Exception as e:
                _log.warning("manifold probe '%s' failed to load: %s", key, e)
        return probes

    def _push_steering(
        self,
        entries: dict[str, SteeringStackEntry],
    ) -> None:
        """Push an entries dict onto the steering stack and rebuild hooks.

        Thin forwarder to :meth:`SteeringComposer.push`; ``_SteeringContext``
        calls it on the session, so it stays as a session method.
        """
        self._steering_composer.push(entries)

    def _pop_steering(self) -> None:
        """Pop the top of the steering stack and rebuild hooks.

        Thin forwarder to :meth:`SteeringComposer.pop`; ``_SteeringContext``
        calls it on the session, so it stays as a session method.
        """
        self._steering_composer.pop()

    def _steering_needs_probe_gating(self) -> bool:
        """Return True iff any active steering trigger carries a
        :class:`~saklas.core.triggers.ProbeGate`.

        Delegates the generation query to :class:`SteeringComposer`.
        """
        return self._steering_composer.steering_needs_probe_gating()

    def _steering_active_in_prefill(self) -> bool:
        """Return True iff any active steering term fires during prompt prefill.

        Thin forwarder to :meth:`SteeringComposer.steering_active_in_prefill`.
        """
        return self._steering_composer.steering_active_in_prefill()

    def _steering_value_prefill_inactive(
        self, value: "str | Steering | None",
    ) -> bool:
        """Return True iff steering ``value`` would not touch the prompt prefill.

        Thin forwarder to :meth:`SteeringComposer.steering_value_prefill_inactive`.
        """
        return self._steering_composer.steering_value_prefill_inactive(value)

    def _exit_internal_steering(self, steering_cm: Any, *, swallow: bool) -> None:
        """Pop ``_generate_core``'s internally-entered steering scope.

        Bypasses the ``_pop_steering`` phase guard (we're past the
        model-forward loop and the rebuild is legitimate teardown, not a
        callback mutating the stack mid-step) by raising the
        ``_internal_steering_pop`` flag around the ``__exit__``.

        Exception-safety-critical (Codex review v2): ``old_internal`` is
        read *before* the ``try`` so the worst case under a signal between
        the read and the assignment is "we never set True", not "we leave
        True set" — the flag never leaks across the gen-lock boundary.

        ``swallow`` mirrors the two call sites: the inner ``finally`` lets
        a teardown ``Exception`` propagate (``swallow=False``); the outer
        ``except BaseException`` path is already re-raising the original
        failure and must not let a teardown ``Exception`` mask it
        (``swallow=True``).  A ``BaseException`` (KeyboardInterrupt /
        SystemExit) from ``__exit__`` always propagates in both — only the
        ``finally`` restore runs.
        """
        old_internal = self._internal_steering_pop
        try:
            self._internal_steering_pop = True
            steering_cm.__exit__(None, None, None)
        except Exception:
            if not swallow:
                raise
        finally:
            self._internal_steering_pop = old_internal

    def _build_gating_score_callback(self):
        """Return a closure that scores latest captures into a
        ``dict[str, float]`` for ``generate_steered``'s ``score_callback``.

        Delegates gate callback construction to :class:`SteeringComposer`.
        """
        return self._steering_composer.build_gating_score_callback()

    def _rebuild_steering_hooks(self) -> None:
        """Tear down existing hooks and install from the flattened stack head.

        Thin forwarder used by the composer after stack mutations.
        """
        self._steering_composer.rebuild_hooks()

    def _begin_capture(
        self, *, widen: bool = False, need_per_token: bool = True,
        gating_only_probes: set[str] | None = None,
        gating_probe_keys: set[str] | None = None,
        lens_gating_probe_keys: set[str] | None = None,
        sae_gating_probe_keys: set[str] | None = None,
        lean_per_token: bool = False,
        final_probe_aggregate: bool = True,
        live_lens_active: bool = True,
        live_sae_active: bool = True,
        capture_prompt: bool = False,
    ) -> bool:
        """Attach hidden-state capture. Returns True if attached.

        The capture layer set is **plan-driven**: each instrument family
        prepares its source snapshot (``Instrument.prepare(ReadRequest)``),
        declares its demand via ``Instrument.plan(prep) ->
        InstrumentPlan``, and this planner unions the declared layers.
        Retention (incremental vs tail ring vs full stack) remains the
        planner's decision below — cross-instrument resource sharing the
        plans deliberately do not decide (``protocol.py``).

        ``widen=False`` (default): cover the union of vector-probe
        layers and manifold-probe layers — what both monitors need.
        Fast path; matches v1 behavior when only vector probes are
        attached.

        ``widen=True``: cover every model layer.  Used when the caller
        asked for ``SamplingConfig.return_hidden=True`` — the monitor still
        reads its probe subset, but the full dict is available on
        ``GenerationResult.hidden_states`` after the run.

        ``need_per_token`` (default True): whether anything consumes a
        per-token reading this gen — a probe gate, a loom token row, a trait
        stream, or a live-scores client.  When False (probes attached but only
        the end-of-gen aggregate is wanted, e.g. stateless server scoring) the
        capture runs in **aggregate-only** mode: a bounded tail ring, NO
        per-token scoring (T scorings + T host syncs → 1 at finalize via
        :meth:`_score_aggregate_only`).

        ``gating_only_probes`` (FIX #4): when per-token scoring is needed *only*
        to feed probe gates (no UI / trait / loom / persist consumer), this is
        the set of gated probe names.  The incremental step sink then scores
        just that subset every token (the gate consumes only those scalars),
        while the big-K roster's per-token nearest-distance work is skipped.
        If ``final_probe_aggregate`` is true, the end-of-gen aggregate still
        covers the **full** roster: the tail ring is also kept, so
        :meth:`_finalize_generation` pools the last content token once and scores
        every probe (the gated subset live-scored, the rest one-shot).  When the
        caller explicitly disables final probe readings, capture narrows to the
        gated probe layers and keeps no full-roster tail. ``None`` (or empty)
        keeps the full per-token scoring.
        """
        # ---- Per-family capture demand (Instrument.plan) -------------------
        # The planner hands each family what the session knows about this
        # generation's read demand; each family answers with its declared
        # ``InstrumentPlan``.  The union of declared layers replaces the
        # former inline per-family union branches.  RETENTION (incremental
        # vs tail ring vs full stack) stays session logic below — the
        # ``INCREMENTAL -> set_tail_with_sink`` upgrade is cross-instrument
        # resource sharing, which plans deliberately do not decide
        # (``protocol.py`` division of labor).
        per_token_full_consumer = bool(
            need_per_token and gating_only_probes is None
        )
        # The uniform capture transaction (``protocol.py``): close →
        # prepare → plan → bind.  The defensive close first (the
        # generation finallys are the ordinary teardown) — a stale pin
        # would short-circuit the lens disk refresh inside ``prepare``,
        # and a stale binding would keep serving frozen specs to idle
        # reads.  ``prepare`` is the source-boundary step: the lens
        # family reads the disk-refreshing ``jlens`` getter under pin
        # demand there and snapshots specs + live config against that
        # identity; ``plan``/``bind`` consume the prep only, so an
        # interleaved adoption (the un-locked ``has_compatible_jlens``)
        # cannot desynchronize them (``LensInstrument.prepare`` carries
        # the ordering + snapshot rationale).
        self._close_instrument_runs()
        geometry_prep = self._geometry_instrument.prepare(ReadRequest(
            gate_keys=frozenset(gating_probe_keys or ()),
            per_token_consumers=per_token_full_consumer,
            final_aggregate=final_probe_aggregate,
            return_hidden=widen,
        ))
        lens_prep = self._lens_instrument.prepare(ReadRequest(
            gate_keys=frozenset(lens_gating_probe_keys or ()),
            live=live_lens_active,
            per_token_consumers=per_token_full_consumer,
            final_aggregate=final_probe_aggregate,
            return_hidden=widen,
        ))
        sae_prep = self._sae_instrument.prepare(ReadRequest(
            gate_keys=frozenset(sae_gating_probe_keys or ()),
            live=live_sae_active,
            per_token_consumers=per_token_full_consumer,
            final_aggregate=final_probe_aggregate,
            return_hidden=widen,
        ))
        geometry_plan = self._geometry_instrument.plan(geometry_prep)
        lens_plan = self._lens_instrument.plan(lens_prep)
        sae_plan = self._sae_instrument.plan(sae_prep)
        has_lens_gate = bool(lens_plan.gate_keys)
        has_sae_gate = bool(sae_plan.gate_keys)

        # Bind each family's per-generation run (``Instrument.bind``): probe
        # specs freeze into the immutable ``InstrumentBinding`` (a concurrent
        # SAE metadata backfill can no longer change what this generation
        # measures), stashes and step memos start fresh, and the lens run
        # carries the pin its prep took — every token, gate, and final
        # aggregate below consumes the same resident lens even if another
        # process switches ``jlens.json`` mid-decode.
        self._geometry_instrument.bind(geometry_plan, geometry_prep)
        self._lens_instrument.bind(lens_plan, lens_prep)
        self._sae_instrument.bind(sae_plan, sae_prep)
        if widen:
            layer_idxs = list(range(len(self._layers)))
        else:
            union: set[int] = set(
                geometry_plan.latest_layers
                | lens_plan.latest_layers | lens_plan.tail_layers
                | sae_plan.latest_layers | sae_plan.tail_layers
            )
            if not union:
                # No probes ⇒ the degenerate FULL (capture-disabled) state.
                self._capture_state = CaptureState(
                    gating_keys=set(gating_probe_keys or ())
                    if gating_probe_keys else None,
                    final_probe_aggregate=final_probe_aggregate,
                )
                self._incremental_readings = []
                return False
            layer_idxs = sorted(union)
        # Persistent compile-clean capture (slice 2): when this gen is eligible
        # for the compiled clean path AND steering lowered to offsets (or is
        # unsteered), capture rides the always-on persistent buffers — the decode
        # loop ``copy_``s each slice in inside the compiled graph, and
        # ``ingest_persistent`` (wired as the step callback) does the per-token
        # accumulation + scoring post-forward.  No transient ``register_forward_hook``
        # is installed, so the routing keeps the compiled module + StaticCache.
        # Anything that forced the eager path (curved / gated / phased steering,
        # ``return_hidden``) falls back to transient capture hooks.
        use_persistent = bool(
            not widen
            and not capture_prompt
            # Persistent capture buffers are installed for every layer before
            # compile; layer_idxs selects the subset this generation consumes.
            # Live lens layers can therefore ride the same compile-clean path
            # instead of forcing transient hooks.
            and self._compiled_clean_eligible
            and self._capture_buffers
            and (
                self._steering_uses_compiled_offsets
                or self._steering.all_fast_path()
            )
        )
        self._capture.clear()
        if use_persistent:
            self._capture.attach_persistent(layer_idxs, self._capture_buffers)
        else:
            self._capture.attach(self._layers, layer_idxs)
        # Default this gen to FULL (append / full-retention); the mode branches
        # below replace it.  ``persistent`` is orthogonal to the mode — set here
        # from the compiled-clean routing decision and carried through unchanged.
        self._capture_state = CaptureState(
            persistent=use_persistent,
            gating_keys=set(gating_probe_keys or ()) if gating_probe_keys else None,
            final_probe_aggregate=final_probe_aggregate,
        )
        self._incremental_readings = []
        self._incremental_gate_scores = []
        # Gating-only-subset path: per-token scoring is needed solely to feed
        # probe gates, so the step sink scores only the gated probes (the gate
        # consumes only those scalars), skipping the big-K roster's per-token
        # nearest-distance work.  A bounded tail ring is kept alongside the sink
        # so finalize still pools the FULL roster once via the aggregate path —
        # the gated subset's per-token rows are NOT the source of the final
        # readings, the one-shot full aggregate is, so no probe drops.
        gating_subset = (
            gating_only_probes
            if (gating_only_probes and need_per_token and not widen
                and self._monitor.probe_names)
            else None
        )
        if not widen and self._monitor.probe_names and need_per_token and gating_subset:
            subset = set(gating_subset)
            gate_keys = set(gating_probe_keys or ())
            gate_plan = self._monitor.plan_gate_scalars(
                gate_keys, probe_names=subset,
            )

            def _score_step_subset(
                step_id: int, latest: dict[int, torch.Tensor],
            ) -> None:
                # Score ONLY the exact gate scalar keys each token (the gate's
                # sole consumer); append even an empty dict to keep row indices
                # aligned with decode forwards.  The full roster is pooled once at
                # finalize from the tail ring (``_score_aggregate_only``).
                # NEVER primes the observe memo — these rows are a scalar
                # subset, not full readings (the completeness trap).
                del step_id
                self._incremental_gate_scores.append(
                    self._monitor.score_planned_gate_scalars(latest, gate_plan)
                    if latest and gate_plan else {}
                )

            if final_probe_aggregate:
                # Deep tail ring (full-roster finalize via ``tail_slice_at``) PLUS a
                # per-token step sink (gated subset, for the gate) — armed together by
                # ``set_tail_with_sink`` (neither single setter gives both).
                self._capture.set_tail_with_sink(_AGG_TAIL_DEPTH, _score_step_subset)
            else:
                # Pure control/gating call: no final aggregate requested, so a
                # length-1 incremental buffer is enough and capture has already been
                # narrowed to the gated probes' layer union.
                self._capture.set_incremental(_score_step_subset)
            self._capture_state.mode = CaptureMode.GATING_SUBSET
            self._capture_state.gating_subset = subset
            self._capture_state.gating_keys = gate_keys
            self._capture_state.final_probe_aggregate = final_probe_aggregate
            # The subset is scored one token per step in order, so curved gated
            # probes can warm-start their foot; cold-start the feet first.
            self._monitor.reset_curved_feet()
            self._monitor.enable_curved_warm(True)
        elif (
            not widen and self._monitor.probe_names
            and need_per_token and lean_per_token
        ):
            # Lean-incremental (FIX F2): the only per-token consumers read just
            # the axis-0 coord (the trait stream / loom probe row) — no nearest /
            # assignment / per-layer trace, no probe gate.  Score each token
            # ``coords_only`` (skips the big-K nearest norm + assignment softmax +
            # per-layer host reconstruction), store the lean rows for the per-token
            # stream, and keep a bounded tail ring so finalize re-scores the FULL
            # aggregate once via ``_score_lean_incremental``.
            def _score_step_lean(
                step_id: int, latest: dict[int, torch.Tensor],
            ) -> None:
                # NEVER primes the observe memo — ``coords_only`` readings
                # omit nearest / assignment / per-layer traces, so exposing
                # them as the full ``observe()`` result would be the second
                # completeness trap beyond the gating subset.
                del step_id
                self._incremental_readings.append(
                    self._monitor.score_single_token(latest, coords_only=True)
                    if latest else {}
                )

            # Deep tail ring (full-roster aggregate at finalize) PLUS the lean
            # per-token step sink — same dual-arming as the gating-subset path.
            self._capture.set_tail_with_sink(_AGG_TAIL_DEPTH, _score_step_lean)
            self._capture_state.mode = CaptureMode.LEAN_INCREMENTAL
            # Lean rows are scored one token per step in order, so curved probes
            # can warm-start their foot for the per-token coord stream.
            self._monitor.reset_curved_feet()
            self._monitor.enable_curved_warm(True)
        elif not widen and self._monitor.probe_names and need_per_token:
            geometry_run = self._geometry_instrument.current_run

            def _score_step(
                step_id: int, latest: dict[int, torch.Tensor],
            ) -> None:
                # Score once while the just-produced hidden slice is still the
                # only retained device payload.  Append even an empty dict so
                # row indices remain aligned with decode forwards; finalization
                # trims any terminal EOS-only overcapture to generated_ids.
                readings = (
                    self._monitor.score_single_token(latest) if latest else {}
                )
                self._incremental_readings.append(readings)
                # FULL-incremental rows ARE the complete per-probe readings,
                # so prime the geometry run's step-keyed observe memo — the
                # gate callback and the token tap consult ``observe(step)``
                # for this forward and hit it instead of rescoring.  The
                # gating-subset and lean sinks never prime (partial rows).
                if readings:
                    geometry_run.prime_observation(step_id, readings)

            if (
                lens_plan.final_aggregate or has_lens_gate
                or sae_plan.final_aggregate or has_sae_gate
            ):
                # Lens probes pool their finalize aggregate from the tail
                # ring; ``set_incremental``'s length-1 buffers can't walk back
                # past trailing specials, so arm the ring alongside the sink.
                # When final probe aggregates are disabled, pinned J-lens/SAE
                # probes still need the latest slice for gates/live consumers,
                # but do not need the 8-deep EOS walk-back ring.  The ring's
                # layer span is the readout families' declared finalize-
                # pooling demand (empty when final readings are off).
                readout_tail_layers = set(
                    lens_plan.tail_layers | sae_plan.tail_layers
                )
                self._capture.set_tail_with_sink(
                    _AGG_TAIL_DEPTH if final_probe_aggregate else 1,
                    _score_step,
                    tail_layers=readout_tail_layers or None,
                )
            else:
                self._capture.set_incremental(_score_step)
            self._capture_state.mode = CaptureMode.INCREMENTAL
            # The step sink scores one token per decode step in order, so curved
            # probes can warm-start their nearest-point foot from the previous
            # token. Cold-start the feet for this generation first.
            self._monitor.reset_curved_feet()
            self._monitor.enable_curved_warm(True)
        elif not widen and self._monitor.probe_names:
            # Aggregate-only: probes attached but no per-token consumer. Keep a
            # bounded tail ring, score nothing per token; finalize pools the
            # last content token once. Curved warm-start is a sequential-live
            # optimization, so leave it off for the one-shot aggregate read.
            self._capture.set_aggregate_tail(_AGG_TAIL_DEPTH)
            self._capture_state.mode = CaptureMode.AGGREGATE_ONLY
            self._monitor.enable_curved_warm(False)
        elif not widen and (
            lens_plan.latest_layers or lens_plan.tail_layers
            or sae_plan.latest_layers or sae_plan.tail_layers
        ):
            # Lens/SAE-only capture (live readout on and/or readout probes
            # pinned, no monitor probes): a bounded tail ring keeps the
            # reader's latest slice fresh without FULL retention growing over
            # the generation. No step sink — live display runs at the token tap,
            # gates score from ``latest_per_layer`` in the gate callback, and
            # final aggregates pool from the ring only when requested. If final
            # readings are disabled, keep a length-1 latest buffer instead of
            # cloning/popping the 8-deep EOS walk-back tail every token.
            self._capture.set_aggregate_tail(
                _AGG_TAIL_DEPTH if final_probe_aggregate else 1
            )
            self._monitor.enable_curved_warm(False)
        else:
            # FULL retention (the ``CaptureState`` default): return_hidden full
            # stack, or no per-token consumer on a non-tail read.  Scored out of
            # order / one-shot, so keep curved probes on the cold foot solve for
            # reproducibility.
            self._monitor.enable_curved_warm(False)
        return True

    def _end_capture(self) -> None:
        self._capture.detach()

    # -- Score entry points --

    def score_choices(
        self,
        messages: list[dict[str, str]],
        choices: list[str],
        *,
        assistant_prefix: str = "",
        labels: list[str] | None = None,
        steering: "str | Steering | None" = None,
        system_prompt: str | None = None,
    ) -> "ChoiceScores":
        """Restricted-choice logprob distribution over ``choices``.

        Scores each candidate completion against the raw model distribution
        given ``messages`` (+ optional ``assistant_prefix`` before the slot) and
        returns the set with per-candidate ``sum``/``mean`` logprobs and their
        softmax probabilities. ``steering=`` runs the scoring forward under a
        steering expression — the distributional before/after read. See
        :func:`saklas.core.scoring.score_choices`.
        """
        from saklas.core.scoring import score_choices as _score_choices

        return _score_choices(
            self, messages, choices,
            assistant_prefix=assistant_prefix, labels=labels,
            steering=steering, system_prompt=system_prompt,
        )

    def score_template(
        self,
        template: "str | TemplateFolder",
        *,
        steering: "str | Steering | None" = None,
        system_prompt: str | None = None,
    ) -> list["ChoiceScores"]:
        """Score a template's values against each of its contexts.

        ``template`` is a :class:`~saklas.io.templates.TemplateFolder` or a
        selector string (``<name>`` / ``<ns>/<name>``). Returns one
        :class:`~saklas.core.scoring.ChoiceScores` per context.
        """
        from saklas.core.scoring import score_template as _score_template
        from saklas.io.templates import TemplateFolder, resolve_template

        tmpl = (
            template if isinstance(template, TemplateFolder)
            else resolve_template(template)
        )
        return _score_template(
            self, tmpl, steering=steering, system_prompt=system_prompt,
        )

    def score_captured(
        self, generated_ids: list[int], *, accumulate: bool = True,
    ) -> tuple[dict[str, "ProbeReading"], dict[str, list[float]]]:
        """Score probes from the last hidden-state capture.

        Returns ``(aggregate_readings, per_token_scores)`` — the aggregate is a
        per-probe :class:`ProbeReading` (coordinate reading), and
        ``per_token_scores`` is the per-probe axis-0 coordinate stream. Both
        dicts are empty when the capture was never attached or the generation
        produced no tokens.
        """
        captured = self._capture.stacked()
        if not captured or not generated_ids:
            return {}, {}
        return self._monitor.score_per_token(
            captured, generated_ids, self._tokenizer, accumulate=accumulate,
        )

    def _aggregate_forward_index(self, generated_ids: list[int]) -> int | None:
        """Forward index to use for the end-of-generation aggregate read."""
        idx = getattr(self._gen_state, "response_aggregate_index", None)
        if idx is not None and 0 <= int(idx) < len(generated_ids):
            return int(idx)
        if (
            self._gen_state.finish_reason == "stop_sequence"
            and getattr(self._gen_state, "response_text", None) is not None
        ):
            return None
        from saklas.core.capture import last_content_index
        return last_content_index(generated_ids, self._tokenizer)

    def _empty_readings(self, names: list[str]) -> dict[str, "ProbeReading"]:
        """A zero ``ProbeReading`` per probe — the no-tokens / missing-pool
        fallback shared by the incremental and aggregate-only score paths."""
        return {
            name: ProbeReading(fraction=0.0, nearest=[], coords=())
            for name in names
        }

    def _extract_coord_stream(
        self, rows: list[dict[str, "ProbeReading"]], n: int, names: list[str],
    ) -> dict[str, list[float]]:
        """Per-probe axis-0 coord stream from per-token reading rows.

        Reads ``reading.coords[0]`` (0.0 when a row is empty) into a length-``n``
        per-probe list — the loop shared by the (full) incremental and the lean
        incremental score paths.
        """
        per_token: dict[str, list[float]] = {name: [0.0] * n for name in names}
        for i, readings in enumerate(rows):
            for name, reading in readings.items():
                if name in per_token:
                    per_token[name][i] = (
                        reading.coords[0] if reading.coords else 0.0
                    )
        return per_token

    def _score_incremental(
        self, generated_ids: list[int], *, accumulate: bool = True,
    ) -> tuple[dict[str, "ProbeReading"], dict[str, list[float]]]:
        """Aggregate readings + per-token coord stream from live-scored rows."""
        names = list(self._monitor.probe_names)
        n = len(generated_ids)
        if n == 0 or not self._incremental_readings:
            return self._empty_readings(names), {name: [] for name in names}

        rows = self._incremental_readings[:n]
        per_token = self._extract_coord_stream(rows, n, names)

        empty_agg = self._empty_readings(names)
        agg_idx = self._aggregate_forward_index(generated_ids)
        agg_row = rows[agg_idx] if agg_idx is not None and agg_idx < len(rows) else {}
        agg_vals = {
            name: agg_row.get(name, empty_agg[name]) for name in names
        }
        if accumulate and agg_vals:
            self._monitor.accumulate_readings(agg_vals)
        return agg_vals, per_token

    def _score_aggregate_only(
        self,
        generated_ids: list[int],
        *,
        accumulate: bool = True,
        pooled: dict[int, torch.Tensor] | None = None,
    ) -> dict[str, "ProbeReading"]:
        """Score *only* the end-of-gen aggregate from the bounded tail ring.

        The aggregate-only capture path (``CaptureMode.AGGREGATE_ONLY``, shared by
        ``GATING_SUBSET`` whose per-token rows feed only the gate) scored nothing
        of the full roster per token; here we pool the last content token's slice
        from the
        tail ring and run one :meth:`Monitor.score_aggregate`.  ``generated_ids``
        token ``k`` was produced by decode forward ``k``, so the last content
        token's forward index is ``last_content_index(generated_ids)`` — which
        :meth:`HiddenCapture.tail_slice_at` maps into the ring.
        """
        names = list(self._monitor.probe_names)
        empty = self._empty_readings(names)
        if not generated_ids:
            return empty
        if pooled is None:
            agg_fwd = self._aggregate_forward_index(generated_ids)
            if agg_fwd is None:
                return empty
            pooled = self._capture.tail_slice_at(agg_fwd)
        if not pooled:
            return empty
        # One-shot pooled read over the full roster — keep curved probes on the
        # cold foot solve so the aggregate is reproducible (it pools the last
        # *content* token, which may differ from the live loop's last forward,
        # so a warm foot seeded during gating-only subset scoring would not be
        # the right warm start here).  Routed through the geometry run's
        # ``observe_aggregate`` (the run protocol face — the last finalize
        # read that bypassed its run); the cold-foot reset above stays
        # session-side.
        self._monitor.enable_curved_warm(False)
        agg_vals = self._geometry_instrument.current_run.observe_aggregate(
            pooled,
        )
        # Fill any probe the pool missed (e.g. a layer absent from the ring).
        agg_vals = {name: agg_vals.get(name, empty[name]) for name in names}
        if accumulate and agg_vals:
            self._monitor.accumulate_readings(agg_vals)
        return agg_vals

    def _score_lean_incremental(
        self,
        generated_ids: list[int],
        *,
        accumulate: bool = True,
        pooled: dict[int, torch.Tensor] | None = None,
    ) -> tuple[dict[str, "ProbeReading"], dict[str, list[float]]]:
        """Lean per-token coord stream + full aggregate from the tail ring (FIX F2).

        The lean-incremental capture path scored each token ``coords_only`` (just
        the cross-layer axis-0 coord / fraction, no nearest / assignment /
        per-layer trace), so the per-token coordinate stream comes straight from
        those stored rows.  The end-of-gen aggregate, which DOES carry the full
        reading, is re-scored once from the bounded tail ring via
        :meth:`_score_aggregate_only` — so no field is lost despite the lean
        per-token rows.
        """
        names = list(self._monitor.probe_names)
        n = len(generated_ids)
        rows = self._incremental_readings[:n]
        per_token = self._extract_coord_stream(rows, n, names)
        agg_vals = self._score_aggregate_only(
            generated_ids, accumulate=accumulate, pooled=pooled,
        )
        return agg_vals, per_token

    @overload
    def score_hidden(
        self,
        hidden: dict[int, torch.Tensor],
        *,
        per_token: Literal[False] = False,
        accumulate: bool = False,
    ) -> dict[str, "ProbeReading"]: ...
    @overload
    def score_hidden(
        self,
        hidden: dict[int, torch.Tensor],
        *,
        per_token: Literal[True],
        accumulate: bool = False,
    ) -> tuple[dict[str, "ProbeReading"], dict[str, list[float]]]: ...
    def score_hidden(
        self,
        hidden: dict[int, torch.Tensor],
        *,
        per_token: bool = False,
        accumulate: bool = False,
    ) -> (
        dict[str, "ProbeReading"]
        | tuple[dict[str, "ProbeReading"], dict[str, list[float]]]
    ):
        """Score registered probes against a pre-captured hidden-state dict.

        Accepts any ``{layer_idx: Tensor}`` mapping — e.g. the
        ``GenerationResult.hidden_states`` dict from a prior
        ``generate(..., sampling=SamplingConfig(return_hidden=True))``
        call, or hidden states the caller captured externally.

        Shape rules:
        - Each value ``[D]``          → single-state aggregate.
          Returns ``dict[probe, ProbeReading]``.
        - Each value ``[T, D]``       → per-token stack.
          ``per_token=False`` (default) returns the aggregate pooled from
          row ``T-1``; ``per_token=True`` returns
          ``(aggregate, per_token_scores)`` where ``per_token_scores`` is
          the per-probe axis-0 coordinate stream.

        Mixed shapes (``[D]`` alongside ``[T, D]``) or uneven ``T`` across
        layers raise :class:`SaklasError`. Empty dict raises.

        ``accumulate`` defaults to ``False`` — ad-hoc researcher scoring
        does not pollute the monitor's history. Pass ``True`` to feed this call
        into the same cross-generation stats pipeline as generation finalization.
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
        probe_manifolds = self._geometry_instrument.manifolds()
        for layer_idx, t in hidden.items():
            actual_dim = t.shape[-1]
            for probe_name, manifold in probe_manifolds.items():
                sub = manifold.layers.get(layer_idx)
                if sub is None:
                    continue
                expected_dim = sub.mean.shape[-1]
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

    def add_probe(
        self,
        selector: str,
        *,
        as_name: str | None = None,
        top_n: int = 3,
    ) -> str:
        """Attach a read-side probe — any shape — and return its name.

        One attach for vector and manifold probes alike: a vector probe is the
        rank-1 case of the unified subspace readout.  ``selector`` rides the
        same ``[ns/]name[:variant]`` shape the steering grammar uses for ``%``
        operands, so a probe shares the lazy-load cache with a manifold steered
        through ``<selector>%<position>``.  ``as_name`` overrides the registered
        name (default ``selector``); ``top_n`` sets the nearest-node list
        length.  Resolution order is in :meth:`_resolve_probe_manifold`.
        """
        # Reserved J-lens namespace: a ``jlens/<word>`` probe is NOT a linear
        # probe — it reads the readout channel (per-layer softmax probability
        # of the token under ``softmax(W_U · norm(J_l h))``, mean-banded),
        # the paper-native "how disposed is the model to say this
        # word" quantity, not a whitened coordinate along ``W_U[v] @ J_l``.
        # Routed to the session lens-probe registry; the Monitor never sees it
        # (no whitener requirement — the softmax is the calibration).
        if selector.startswith("jlens/"):
            return self._add_lens_probe(selector, as_name=as_name)
        if selector.startswith("sae/"):
            return self._add_sae_probe(selector, as_name=as_name)
        # Geometry (Monitor) probes attach through their instrument like the
        # other families; the whitener build + manifold resolution + monitor
        # registration run under the exclusive-GPU section inside it.
        name = self._geometry_instrument.attach(
            selector, as_name=as_name, top_n=top_n,
        )
        # New probe → _begin_capture attaches a different layer set than was
        # live when the prefix was prefilled; drop the cache so the next gen
        # re-prefills with the fresh capture-attach layout in place.
        self._invalidate_prefix_cache()
        # Transcript probe-hash cache is keyed by name; any change to the
        # registered probes invalidates the relevant entry.
        self._probe_hash_cache.pop(name, None)
        self._invalidate_analytics_cache()
        return name

    def _add_lens_probe(self, selector: str, *, as_name: str | None) -> str:
        """Attach a ``jlens/<word>`` token probe (readout channel).

        Delegates the validation + registry write to
        :meth:`LensInstrument.attach`; the session owns the cache
        invalidation at this boundary (the capture layer union changed, and
        the probe hash / analytics caches key on the roster).
        """
        name = self._lens_instrument.attach(selector, as_name=as_name)
        self._invalidate_prefix_cache()
        self._probe_hash_cache.pop(name, None)
        self._invalidate_analytics_cache()
        return name

    def _add_sae_probe(self, selector: str, *, as_name: str | None) -> str:
        """Attach a resident SAE feature as a one-channel readout probe
        (delegates to :meth:`SaeInstrument.attach`; the session owns the
        cache invalidation at this boundary)."""
        name = self._sae_instrument.attach(selector, as_name=as_name)
        self._invalidate_prefix_cache()
        self._probe_hash_cache.pop(name, None)
        self._invalidate_analytics_cache()
        return name

    def remove_probe(self, name: str) -> None:
        """Detach a previously-attached probe (any shape).

        Lens/SAE detaches are safe mid-generation — a bound run's frozen
        binding keeps measuring the bind-time roster, so the removal
        applies at the next generation boundary.  A geometry detach
        mutates the live Monitor roster in-flight scoring walks, so it
        routes through the instrument's exclusive section and rejects
        while a generation is running (retry when idle).  The lens
        removal is the instrument's atomic ``try_detach`` — a bare
        membership check + direct registry delete bypassed the
        lens-state lock and could land inside a ``prepare`` snapshot or
        an adoption (round-5).
        """
        if self._lens_instrument.try_detach(name):
            pass
        elif self._sae_instrument.try_detach(name):
            pass
        else:
            self._geometry_instrument.detach(name)
        self._invalidate_prefix_cache()
        self._probe_hash_cache.pop(name, None)
        self._invalidate_analytics_cache()

    @property
    def lens_probe_names(self) -> list[str]:
        """Names of the attached J-lens token probes (readout channel)."""
        return self._lens_instrument.names

    @property
    def lens_probe_specs(self) -> dict[str, dict[str, Any]]:
        """Snapshot of attached J-lens probe specifications."""
        return self._lens_instrument.specs()

    def _lens_probe_layers(self, names: set[str] | None = None) -> set[int]:
        """Union of the attached lens probes' fitted layers."""
        return self._lens_instrument.probe_layers(names)

    @property
    def sae_probe_names(self) -> list[str]:
        return self._sae_instrument.names

    @property
    def sae_probe_specs(self) -> dict[str, dict[str, Any]]:
        """Snapshot of attached SAE probe specifications."""
        return self._sae_instrument.specs()

    def _score_lens_probes(
        self,
        hidden: dict[int, torch.Tensor],
        *,
        logits: torch.Tensor | None = None,
        probabilities: torch.Tensor | None = None,
        layers: "Sequence[int] | None" = None,
        only: "set[str] | None" = None,
    ) -> dict[str, "ProbeReading"]:
        """Score attached lens probes (delegates to
        :meth:`LensInstrument.score_probes`)."""
        return self._lens_instrument.score_probes(
            hidden,
            logits=logits,
            probabilities=probabilities,
            layers=layers,
            only=only,
        )

    def _score_lens_gate_scalars(
        self, gate_keys: "set[str] | None" = None, *, step_id: int = -1,
    ) -> dict[str, float]:
        """Per-forward lens gate scalars — through the run protocol face
        (:meth:`LensRun.gate_scalars`, which reaches the instrument worker;
        ``step_id`` keys the band-logit stash so the display step reuses
        this forward's rows)."""
        return self._lens_instrument.current_run.gate_scalars(
            step_id, None, gate_keys,
        )

    def _score_lens_probes_aggregate(
        self,
        generated_ids: list[int],
        *,
        pooled: dict[int, torch.Tensor] | None = None,
    ) -> dict[str, "ProbeReading"]:
        """End-of-gen lens-probe aggregate (delegates to
        :meth:`LensInstrument.score_aggregate`)."""
        return self._lens_instrument.score_aggregate(
            generated_ids, pooled=pooled,
        )

    @property
    def live_probe_scores(self) -> bool:
        """Whether per-token monitor scoring feeds live consumers (the CAA
        live toggle).  When False, generations run aggregate-only capture —
        probes still report the end-of-gen aggregate, but no per-token
        stream, loom token rows, or trait events are produced.  Probe gates
        are unaffected: a gate forces the per-token subset it needs."""
        return self._live_probe_scores

    def set_live_probe_scores(self, enabled: bool) -> bool:
        """Toggle live per-token monitor scoring (see
        :attr:`live_probe_scores`).  Returns the new state."""
        self._live_probe_scores = bool(enabled)
        return self._live_probe_scores

    def _resolve_probe_manifold(self, selector: str) -> "Manifold":
        """Resolve a probe selector to a loaded :class:`Manifold`.

        Mirrors the steering resolver, in order: (1) an in-memory baked
        ``Profile`` already in ``_profiles`` (ad-hoc ``extract`` / ``merge`` /
        projection results) → folded into a 1-node neutral-anchored ray;
        (2) a fitted manifold on disk (``[ns/]name[:variant]``) → loaded
        directly, so a 2-node ``pca`` reads rank-1 and a discover / curved fit
        reads rank-R; (3) a bare concept with neither → extracted, then folded.

        ``jlens/<word>`` selectors never reach this resolver — ``add_probe``
        routes them to the lens-probe registry (the readout channel), so the
        reserved namespace can't fall through to ``extract()`` and author a
        nonsense manifold named "jlens/<word>".
        """
        profile = self._profiles.get(selector)
        if profile is not None:
            return self._fold_profile_probe(selector, profile)
        try:
            self.ensure_manifold_loaded(selector)
            return self._manifolds[selector]
        except ManifoldNotRegisteredError:
            pass
        _, profile = self.extract(selector)
        return self._fold_profile_probe(selector, profile)

    def _fold_profile_probe(self, name: str, profile: Any) -> "Manifold":
        """Fold a baked per-layer ``Profile`` into a 1-node neutral-anchored ray.

        A ``Profile`` is a per-layer baked direction; the monitor reads probes
        as flat manifolds (the rank-1 coordinate case), so fold it via the
        same primitive ``extract`` uses for a monopolar concept.
        """
        if not self._layer_means:
            _ = self.layer_means
            self._monitor.layer_means = self._layer_means
        from saklas.core.capture import fold_directions_to_subspace
        return fold_directions_to_subspace(
            name, dict(profile), self._layer_means, whitener=self.whitener,
        )

    def _probe_hash(self, name: str) -> str | None:
        """Return sha256 hex of the baked tensor bytes for ``name``.

        Stamps :class:`saklas.core.loom.Recipe.probe_hashes` so transcript
        replay can detect probe drift between save and load. Cached on the
        session — adding
        or removing a probe invalidates the relevant cache entry.

        Returns ``None`` when the probe isn't registered.  Hashing is
        deterministic across machines: layers iterated in sorted order,
        each tensor's CPU bytes hashed (fp32 cast to keep dtype neutral
        across mixed-precision storage).
        """
        if name in self._probe_hash_cache:
            return self._probe_hash_cache[name]
        lens_digest = self._lens_instrument.probe_hash(name)
        if lens_digest is not None:
            # A lens probe has no baked tensor — the instrument hashes the
            # readout-channel identity (model, token, band, channel version)
            # so transcript drift detection works across a semantics change.
            self._probe_hash_cache[name] = lens_digest
            return lens_digest
        sae_digest = self._sae_instrument.probe_hash(name)
        if sae_digest is not None:
            # The instrument hashes the readout-channel identity; ``max_act``
            # is part of it (it sets the unit), so drift detection catches a
            # unit change.
            self._probe_hash_cache[name] = sae_digest
            return sae_digest
        geometry_digest = self._geometry_instrument.probe_hash(name)
        if geometry_digest is None:
            return None
        self._probe_hash_cache[name] = geometry_digest
        return geometry_digest

    def probe_hashes(self) -> dict[str, str]:
        """Return ``{probe_name: sha256_hex}`` for every registered probe.

        Roster names come from the instruments' LOCKED snapshots (raw
        registry expansion tears under a concurrent detach; round-7);
        the per-name hash lookup already tolerates a probe vanishing
        between enumeration and hashing.
        """
        out: dict[str, str] = {}
        for name in (
            *self._geometry_instrument.names,
            *self._lens_instrument.names,
            *self._sae_instrument.names,
        ):
            d = self._probe_hash(name)
            if d is not None:
                out[name] = d
        return out

    # -- Cross-branch diff (v2.3 phase 5) --

    def diff_nodes(self, a_id: str, b_id: str) -> Any:
        """Return a :class:`saklas.core.loom_diff.NodeDiff` between two nodes.

        Both nodes are looked up in :attr:`tree`; the diff bundles the
        word-level text diff and the readings delta table.  ``parent_id``
        on the returned diff is the shared parent when both nodes
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

    def _resolve_anchor_recipe(
        self,
        parent_node_id: str | None,
        *,
        base_recipe: "Recipe | None" = None,
    ) -> "Recipe":
        """Resolve the recipe a regen-modifier override composes onto.

        Precedence: an explicit ``base_recipe`` wins; else the parent's
        recipe when the parent is an assistant carrying one; else the
        nearest assistant *ancestor* with a recipe (so a user-anchored
        regen still finds one to overlay); else an empty :class:`Recipe`.
        ``parent_node_id`` may be ``None`` (no parent context) — only the
        empty-Recipe fallback applies then.
        """
        from saklas.core.loom import Recipe

        if base_recipe is not None:
            return base_recipe
        anchor: "Recipe | None" = None
        if parent_node_id is not None:
            parent = self.tree.nodes.get(parent_node_id)
            if parent is not None:
                # A recipe marks a *generated* node whatever its seat (the
                # cast model: provenance, not role).
                if parent.recipe is not None:
                    anchor = parent.recipe
                else:
                    for nid in self.tree.ancestors_of(parent_node_id):
                        anc = self.tree.nodes.get(nid)
                        if anc is not None and anc.recipe is not None:
                            anchor = anc.recipe
                            break
        return anchor if anchor is not None else Recipe()

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
        parent = self.tree.get(parent_node_id)
        # Resolve the assistant recipe we're overlaying — if the caller
        # passed a user node, walk to the nearest assistant ancestor's
        # recipe (regen replaces that assistant); see
        # ``_resolve_anchor_recipe``.
        anchor = self._resolve_anchor_recipe(
            parent_node_id, base_recipe=base_recipe,
        )

        # compose_modifier handles both str ("unsteered"/"inverted"/...)
        # and Recipe (custom) — the dispatch lives on the dataclass.
        override = anchor.compose_modifier(mode)

        overlaid = anchor.overlay(override)

        # Resolve where the regen anchors (continue-mode: no text resend,
        # the history walk carries every turn).  A generated node regens
        # as a sibling under its parent in the same seat.  A committed turn
        # receives a fresh reply in the opposite seat.  Recipe presence is
        # the capability boundary; seat never stands in for provenance.
        if parent.recipe is not None:
            anchor_parent = parent.parent_id
            seat = parent.role
        else:
            anchor_parent = parent_node_id
            seat = "user" if parent.role == "assistant" else "assistant"
        if anchor_parent is None:
            raise InvalidNodeOperationError(
                f"regen_with_modifier: cannot anchor sibling under "
                f"{parent_node_id!r} — the node has no parent"
            )

        sampling = overlaid.sampling
        return self.generate(
            None,
            steering=overlaid.steering,
            sampling=sampling,
            thinking=overlaid.thinking,
            parent_node_id=anchor_parent,
            n=n,
            gen_seat=seat,
        )

    def fork_from_token(
        self,
        node_id: str,
        raw_index: int,
        alt_token_id: int,
        *,
        on_token: TokenCallback | None = None,
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

        Lands as a sibling under the same parent turn, occupying the same
        seat as the source node (a user-seat generated node forks into
        user-seat siblings — the cast model).  Raises
        :class:`InvalidNodeOperationError` when the node has no
        ``raw_token_ids`` capability (legacy or transcript-loaded node), or
        when ``raw_index`` is out of range.
        """
        from dataclasses import replace as _replace

        node = self.tree.get(node_id)
        raw = node.raw_token_ids
        if not raw:
            raise InvalidNodeOperationError(
                f"fork_from_token: {node_id!r} has no raw token record "
                f"(transcript-loaded node — not forkable)"
            )
        if not 0 <= raw_index < len(raw):
            raise InvalidNodeOperationError(
                f"fork_from_token: raw_index {raw_index} out of range "
                f"[0, {len(raw)}) for {node_id!r}"
            )
        if node.parent_id is None:
            raise InvalidNodeOperationError(
                f"fork_from_token: {node_id!r} has no parent to anchor "
                f"the forked sibling under"
            )
        forced_prefix = [int(t) for t in raw[:raw_index]] + [int(alt_token_id)]

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
        # The sibling inherits the source node's label on its own seat —
        # the tree's stamped label wins over whatever the recipe's
        # sampling boxes held.
        if node.role == "user":
            fork_sampling = _replace(fork_sampling, user_role=node.role_label)
        else:
            fork_sampling = _replace(
                fork_sampling, assistant_role=node.role_label,
            )

        return self._generate_core(
            None,
            steering=recipe.steering if recipe is not None else None,
            sampling=fork_sampling,
            thinking=recipe.thinking if recipe is not None else None,
            on_token=on_token,
            parent_node_id=node.parent_id,
            forced_prefix=forced_prefix,
            gen_seat=node.role,
        )

    def prefill_assistant(
        self,
        node_id: str,
        text: str,
        *,
        steering: "str | Steering | None" = None,
        sampling: SamplingConfig | None = None,
        on_token: TokenCallback | None = None,
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

    def _check_thinking_commit(self, thinking: str | None) -> None:
        """Refuse a committed thinking block the renderer can't carry.

        Committed thinking renders through the scene stitcher's family
        think delimiters — a family without them (gemma), a channel-
        thinking family (gpt-oss), or a template-fallback family (GLM)
        would raise at the *next* render, long after the commit.  Fail
        at commit time instead, with the same error the renderer uses.
        """
        if thinking is None:
            return
        from saklas.core.scene import SceneThinkingUnsupportedError

        grammar = self.scene_grammar
        if grammar is None or grammar.think_open is None:
            raise SceneThinkingUnsupportedError(
                "this model family has no scene-mode think delimiters — "
                "a committed thinking block could not be rendered"
            )

    def append_user_turn(
        self,
        parent_node_id: str | None,
        text: str,
        *,
        allow_any_parent: bool = False,
        role_label: str | None = None,
        thinking: str | None = None,
    ) -> str:
        """Append a user-seat span without generating.

        If the selected parent already has the user structural role and the
        same effective ``role_label``, ``text`` is concatenated exactly onto
        that node. Otherwise a new user-role child is created. Legacy chat
        templates reject the latter user-under-user shape; scene renderers
        accept it.

        ``allow_any_parent`` skips the user-under-user guard: in flat
        (base-model) mode the role tag is authorship provenance, not a
        turn-taking constraint, so an authored span may hang under a
        node of any role.

        ``text`` must be non-empty (whitespace-only is honored, but a
        completely empty string raises) — empty appends should be
        no-op'd at the surface, not sent over the wire.
        """
        if text == "":
            raise InvalidNodeOperationError(
                "append_user_turn: text must be non-empty"
            )
        if thinking is not None:
            self._check_thinking_commit(thinking)
        parent_id = parent_node_id or self.tree.active_node_id
        parent = self.tree.get(parent_id)
        if parent.role == "user" and parent.role_label == role_label:
            combined = parent.text + text
            raw_token_ids = list(
                self._tokenizer.encode(combined, add_special_tokens=False)
            )
            return self.tree.append_authored(
                parent_id,
                text,
                thinking_text=thinking,
                raw_token_ids=raw_token_ids,
            )

        # A different role label is a genuinely new turn.  Scene mode lifts
        # the legacy user-under-user guard for that shape.
        if not allow_any_parent and self.scene_grammar is None:
            self._check_user_send_target(parent_node_id)
        return self.tree.add_user_turn(
            text, parent_id=parent_node_id, role_label=role_label,
            thinking_text=thinking)

    def append_turn(
        self,
        parent_node_id: str | None,
        text: str,
        *,
        role: Literal["user", "assistant"],
        raw: bool = False,
        role_label: str | None = None,
        thinking: str | None = None,
    ) -> str:
        """Append one authored span in a structural chat-template role.

        This is the role-neutral append primitive. The specialized
        :meth:`append_user_turn` and :meth:`append_assistant_turn` methods
        retain their role-specific validation and concatenation behavior.
        """
        if role == "user":
            return self.append_user_turn(
                parent_node_id,
                text,
                allow_any_parent=raw,
                role_label=role_label,
                thinking=thinking,
            )
        if role != "assistant":
            raise ValueError(
                f"role must be 'user' or 'assistant', got {role!r}"
            )
        parent = parent_node_id or self.tree.active_node_id
        return self.append_assistant_turn(
            parent,
            text,
            role_label=role_label,
            thinking=thinking,
        )

    def append_assistant_turn(
        self,
        user_node_id: str,
        text: str,
        *,
        role_label: str | None = None,
        thinking: str | None = None,
    ) -> str:
        """Append an assistant-role span without generating.

        If ``user_node_id`` is already an assistant structural role under the
        same effective ``role_label``, ``text`` is concatenated exactly onto
        that node. Otherwise an authored assistant child is created with no
        sampling or steering. Authored messages keep ``recipe=None`` and carry
        tokenized full text so they remain forkable.

        Raises :class:`InvalidNodeOperationError` when ``user_node_id``
        isn't a user node on a legacy-render family (scene mode frees
        the seating — the parameter name is historic), when ``text`` is
        empty, or when it tokenizes to an empty sequence.  Returns the
        new assistant node id; the loom's active node advances to it.
        """
        if text == "":
            raise InvalidNodeOperationError(
                "append_assistant_turn: text must be non-empty"
            )
        if thinking is not None:
            self._check_thinking_commit(thinking)
        node = self.tree.get(user_node_id)
        if node.role == "assistant" and node.role_label == role_label:
            combined = node.text + text
            raw_token_ids = list(
                self._tokenizer.encode(combined, add_special_tokens=False)
            )
            if not raw_token_ids:
                raise InvalidNodeOperationError(
                    f"append_assistant_turn: {combined!r} tokenized to an "
                    f"empty sequence — nothing to append"
                )
            return self.tree.append_authored(
                user_node_id,
                text,
                thinking_text=thinking,
                raw_token_ids=raw_token_ids,
            )
        # Scene mode lifts the user-parent requirement: the stitcher
        # renders arbitrary seat sequences, so an authored assistant
        # turn may hang under any node (a/a, assistant-first shapes).
        # Legacy-render families keep the strict guard — their
        # templates can't render the freed shapes.
        if node.role != "user" and self.scene_grammar is None:
            raise InvalidNodeOperationError(
                f"append_assistant_turn: {user_node_id!r} is a "
                f"{node.role} node, not a user node — an authored "
                f"assistant turn hangs off a user turn (free seating "
                f"requires scene mode)"
            )
        raw_token_ids = list(
            self._tokenizer.encode(text, add_special_tokens=False)
        )
        if not raw_token_ids:
            raise InvalidNodeOperationError(
                f"append_assistant_turn: {text!r} tokenized to an "
                f"empty sequence — nothing to append"
            )

        new_id = self.tree.begin_assistant(
            user_node_id, recipe=None, role_label=role_label)
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
            thinking_text=thinking,
        )
        return new_id

    # -- Cast roster (phase 3) --

    def set_cast_member(
        self,
        label: str,
        *,
        steering: str | None = None,
        sampling: SamplingConfig | None = None,
        thinking: bool | None = None,
        seed: int | None = None,
        notes: str = "",
    ) -> "CastMember":
        """Create or replace the cast member under ``label``.

        Convenience over :meth:`LoomTree.set_cast_member` that builds the
        member's :class:`Recipe` from parts and validates the steering
        expression *now* (``parse_expr``) so a typo surfaces at authoring
        time, not on the member's first generation.  A member whose
        fields are all unset is a bare named label — legal, it just
        contributes no defaults.
        """
        if steering:
            from saklas.core.steering_expr import parse_expr
            parse_expr(steering)  # raises SteeringExprError on bad syntax
        recipe: Recipe | None = None
        if (
            steering is not None
            or sampling is not None
            or thinking is not None
            or seed is not None
        ):
            recipe = Recipe(
                steering=steering, sampling=sampling,
                thinking=thinking, seed=seed,
            )
        member = CastMember(recipe=recipe, notes=notes)
        self.tree.set_cast_member(label, member)
        return member

    def remove_cast_member(self, label: str) -> None:
        """Drop the cast member under ``label`` (no-op when absent)."""
        self.tree.remove_cast_member(label)

    @property
    def cast(self) -> "dict[str, CastMember]":
        """Read view of the tree's cast roster (label → member)."""
        return dict(self.tree.cast)

    # -- History / loom tree --

    def _check_user_send_target(self, parent_node_id: str | None) -> None:
        """Refuse sending a new user turn from a user-role node.

        A fresh user turn is invalid when the resolved parent is itself a
        user node: that node is already
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

        Rules:

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

    def restore_tree(self, data: dict[str, Any]) -> LoomTree:
        """Atomically replace the live conversation with a validated tree.

        The caller must hold the server's async session lock (or otherwise
        guarantee that no generation is in flight).  Imported files are
        model-bound: recipes and token ids are not portable across models,
        so a mismatch is rejected instead of producing a subtly corrupt
        continuation.  Runtime session identity is retained while every
        conversation-owned field (branches, recipes, cast, tokens, notes)
        comes from the file.
        """
        restored = LoomTree.from_dict(
            data,
            events=self.events,
            conflict_check=self._loom_conflict_check,
        )
        if restored.model_id not in (None, self.model_id):
            raise LoomTreeError(
                f"tree model {restored.model_id!r} does not match loaded model "
                f"{self.model_id!r}"
            )

        old_tree = self.tree
        old_ids = set(old_tree.nodes)
        restored.model_id = self.model_id
        restored.session_id = old_tree.session_id
        # A restore is a mutation of this live session even when the file was
        # saved at a lower revision.  Preserve monotonic wire revisions so
        # connected clients can apply the replacement without a special
        # rollback rule.
        restored.rev = max(old_tree.rev, restored.rev) + 1
        self.tree = restored
        self._joint_logprob_cache.clear()
        self._monitor.reset_history()

        new_ids = set(restored.nodes)
        # LoomTree deliberately accepts the session EventBus through a
        # cycle-breaking duck-typed boundary; emit through that same typed
        # adapter rather than widening the core lifecycle Event union here.
        restored._emit(LoomMutated(  # pyright: ignore[reportPrivateUsage]
            op="restore",
            rev=restored.rev,
            added=tuple(nid for nid in restored.nodes if nid not in old_ids),
            removed=tuple(nid for nid in old_tree.nodes if nid not in new_ids),
            updated=tuple(nid for nid in restored.nodes if nid in old_ids),
            active_node_id=restored.active_node_id,
        ))
        return restored

    # -- Prefix KV cache --

    def cache_prefix(
        self,
        messages: "list[dict[str, str]] | torch.Tensor | None",
        *,
        max_new_tokens: int | None = None,
        prefer_static: bool = False,
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

        ``prefer_static=True`` builds a StaticCache-backed prefix entry when the
        session supports StaticCache. ``max_new_tokens`` sizes the decode
        headroom; callers that pass a larger generation budget later will miss
        this static entry and fall back to the regular full-prefill path. When
        StaticCache construction or prefill fails, the method falls back to the
        ordinary DynamicCache prefix entry.

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

        if self._steering_composer._stack:
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
            prefix_ids = cast(torch.Tensor, messages)
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
                scene=self.scene_grammar,
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

        use_static = bool(prefer_static and self._static_cache_active)
        static_max_cache_len: int | None = None
        past_key_values: object | None = None
        cache_position: torch.Tensor | None = None
        if use_static:
            try:
                from saklas.core.cuda_graphs import make_static_cache

                model_dtype = next(self._model.parameters()).dtype
                headroom = int(
                    max_new_tokens
                    if max_new_tokens is not None
                    else self.config.max_new_tokens
                )
                static_max_cache_len = prefix_len + max(headroom, 1)
                past_key_values = make_static_cache(
                    self._model,
                    max_cache_len=static_max_cache_len,
                    device=self._device,
                    dtype=model_dtype,
                )
                cache_position = torch.arange(
                    prefix_len, device=self._device, dtype=torch.long,
                )
            except Exception as exc:
                _log.debug(
                    "cache_prefix: StaticCache prefix skipped (%s)", exc,
                )
                use_static = False
                past_key_values = None
                cache_position = None
                static_max_cache_len = None

        try:
            with torch.inference_mode():
                if use_static and past_key_values is not None and cache_position is not None:
                    outputs = self._model(
                        input_ids=prefix_ids,
                        attention_mask=torch.ones_like(prefix_ids),
                        past_key_values=past_key_values,
                        use_cache=True,
                        cache_position=cache_position,
                    )
                else:
                    outputs = self._model(
                        input_ids=prefix_ids,
                        attention_mask=torch.ones_like(prefix_ids),
                        use_cache=True,
                    )
        except Exception:
            if not use_static:
                raise
            _log.debug(
                "cache_prefix: StaticCache prefill failed; retrying DynamicCache",
                exc_info=True,
            )
            use_static = False
            past_key_values = None
            cache_position = None
            static_max_cache_len = None
            with torch.inference_mode():
                outputs = self._model(
                    input_ids=prefix_ids,
                    attention_mask=torch.ones_like(prefix_ids),
                    use_cache=True,
                )

        if use_static and past_key_values is not None:
            # StaticCache mutates in place. Some modeling files return the same
            # object, others may return ``None``; keep the preallocated cache in
            # both cases.
            returned_pkv = outputs.past_key_values
            past_key_values = (
                returned_pkv if returned_pkv is not None else past_key_values
            )
        else:
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
        self._prefix_cache = _PrefixCacheEntry(
            prefix_ids_cpu=prefix_ids_cpu,
            past_key_values=past_key_values,
            prefix_len=prefix_len,
            static=use_static,
            max_cache_len=static_max_cache_len,
        )
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
        self,
        input_ids: torch.Tensor,
        *,
        static_eligible: bool = False,
        required_max_new_tokens: int | None = None,
    ) -> tuple[torch.Tensor, object, int, bool] | None:
        """Return (suffix_ids, past_key_values, prefix_len, static) on hit.

        Cache-hit precondition: the cached prefix tokens match
        ``input_ids[0, :prefix_len]`` byte-for-byte AND the suffix is
        non-empty (a zero-length suffix has no last-token logit to
        sample from on the first iteration; we'd need a different
        codepath to handle it — for now, fall through to no-cache).

        StaticCache entries additionally require the current generation to be
        StaticCache-eligible and to fit inside the entry's preallocated
        ``max_cache_len``. Oversized static entries miss safely; the caller can
        then do a regular full-prefill with a freshly sized StaticCache.
        """
        entry = self._prefix_cache
        if entry is None:
            return None
        # Tolerate stale test sentinels or pre-dataclass tuples by treating
        # anything unexpected as a miss. Real cache entries are always
        # ``_PrefixCacheEntry``.
        if not isinstance(entry, _PrefixCacheEntry):  # pyright: ignore[reportUnnecessaryIsInstance]  # stale/test sentinels can violate annotation
            return None
        prefix_ids_cpu = entry.prefix_ids_cpu
        prefix_len = entry.prefix_len
        if input_ids.shape[1] <= prefix_len:
            return None
        head = input_ids[0, :prefix_len].detach().to("cpu")
        if not torch.equal(head, prefix_ids_cpu):
            return None
        suffix_ids = input_ids[:, prefix_len:].contiguous()
        if entry.static:
            if not static_eligible:
                return None
            if entry.max_cache_len is not None:
                required = (
                    prefix_len
                    + int(suffix_ids.shape[1])
                    + max(int(required_max_new_tokens or 0), 1)
                )
                if required > entry.max_cache_len:
                    return None
        return suffix_ids, entry.past_key_values, prefix_len, entry.static

    # -- Generation helpers --

    def _resolved_model_type(self) -> str | None:
        """``model_type`` with the multimodal ``text_config`` unwrap."""
        model_cfg = getattr(self._model, "config", None)
        text_cfg = (
            getattr(model_cfg, "text_config", None)
            if model_cfg is not None else None
        )
        model_type = (
            getattr(text_cfg, "model_type", None)
            if text_cfg is not None else None
        )
        if model_type is None and model_cfg is not None:
            model_type = getattr(model_cfg, "model_type", None)
        return model_type

    @property
    def scene_grammar(self) -> TurnGrammar | None:
        """The validated turn grammar for the loaded chat template, or None.

        Lazily runs the template autopsy + byte-exact round-trip validation
        (``core/scene.py``) once per session.  A passing grammar makes the
        stitcher the render authority — arbitrary seat sequences, per-turn
        cast labels, generation prompts on either seat — with renders
        bit-identical to ``apply_chat_template`` on standard alternating
        conversations.  ``None`` = scene mode unavailable (base model,
        label-free family like mistral, or a template shape that defeated
        the autopsy); rendering falls back to the chat-template paths and a
        one-time warning names the reason.
        """
        if not self._scene_grammar_resolved:
            self._scene_grammar_resolved = True
            self._scene_grammar = None
            template = getattr(self._tokenizer, "chat_template", None)
            if template:
                think = (
                    ("<think>", "</think>") if "<think>" in template else None
                )
                try:
                    grammar = extract_turn_grammar(
                        self._tokenizer,
                        self._resolved_model_type() or "",
                        think_delimiters=think,
                    )
                    validate_turn_grammar(grammar, self._tokenizer)
                    self._scene_grammar = grammar
                except SceneGrammarError as e:
                    _log.warning(
                        "scene mode unavailable for this model (%s); "
                        "rendering falls back to the chat template", e,
                    )
        return self._scene_grammar

    def _prepare_input(
        self, input: Any, raw: bool = False, thinking: bool = False,
        stateless: bool = False,
        parent_node_id: str | None = None,
        user_role: str | None = None,
        assistant_role: str | None = None,
        to_device: bool = True,
        gen_seat: str = "assistant",
        add_generation_prompt: bool = True,
    ) -> torch.Tensor:
        if raw and (isinstance(input, str) or input is None):
            # Flat (base-model / completion) path: no chat template, no
            # role markers.  The model sees the active-path text verbatim
            # — every node along the loom path concatenated — plus this
            # call's own ``input``.  ``stateless`` skips the tree walk so
            # the buffer is purely ``input``.  ``input=None`` is a bare
            # continuation (no new span).
            prefix = "" if stateless else self.tree.flat_text(parent_node_id)
            encoded = self._tokenizer.encode(
                prefix + (input or ""), return_tensors="pt",
            )
            ids = cast(torch.Tensor, encoded)  # return_tensors="pt" gives Tensor, not list[int]
            return ids.to(self._device) if to_device else ids
        if isinstance(input, str) or input is None:
            if stateless:
                prior: list[dict[str, Any]] = []
            else:
                # Walk the path to ``parent_node_id`` (or the active node).
                # Loom: the model sees the conversation along whatever path
                # the user is currently focused on, not a single flat log.
                # ``with_labels`` carries each prior turn's stamped
                # ``role_label`` so the render is faithful per-turn — earlier
                # turns render with the roles they were *sent* with.
                prior = self.tree.messages_for(parent_node_id, with_labels=True)
            # The new user turn carries this send's user-role label.
            # ``input=None`` is a continue: no new turn, the generation
            # prompt opens directly after the existing history (the a/a and
            # u-continue shapes of the cast model).
            if input is None:
                messages = prior
            else:
                messages = prior + [
                    {"role": "user", "content": input, "label": user_role}
                ]
        elif isinstance(input, list):
            messages = list(input)
        else:
            raise TypeError(f"Unsupported input type: {type(input)}")
        # Generation-prompt label.  Assistant seat: a role-augmented steering
        # scope (``_active_role``, transient, set by ``_SteeringContext`` for
        # ``:role-<slug>`` vectors / persona manifolds) wins so steer
        # baseline matches extract baseline; otherwise this send's
        # ``assistant_role`` box drives the to-be-generated turn.  User
        # seat: the ``user_role`` box labels the turn being generated (the
        # elicitation-baseline contract is an assistant-seat construct, so
        # ``_active_role`` doesn't apply there).  Prior turns' labels ride
        # on the messages themselves (above).
        steer_role = self._active_role
        if gen_seat == "assistant":
            gen_role = steer_role if steer_role is not None else assistant_role
            if (
                add_generation_prompt
                and steer_role is not None
                and assistant_role is not None
                and steer_role != assistant_role
            ):
                # The steering scope's role baseline wins over the send's
                # label (extract baseline = steer baseline), but the caller
                # asked for a different header — say so rather than
                # silently rendering a turn they didn't label.
                import warnings as _warnings
                from saklas.core.errors import RoleBaselineMismatchWarning
                _warnings.warn(
                    f"steering scope implies role {steer_role!r} but this "
                    f"send labels the generated turn {assistant_role!r}; "
                    f"the steering role wins for the generation header",
                    RoleBaselineMismatchWarning,
                    stacklevel=2,
                )
        else:
            gen_role = user_role
        model_type_for_role: str | None = None
        any_label = gen_role is not None or any(
            isinstance(m, dict) and m.get("label") for m in messages
        )
        if any_label:
            model_type_for_role = self._resolved_model_type()
        ids = build_chat_input(
            self._tokenizer, messages, self.config.system_prompt,
            thinking=thinking,
            add_generation_prompt=add_generation_prompt,
            gen_role=gen_role,
            model_type=model_type_for_role,
            scene=self.scene_grammar,
            gen_seat=gen_seat,
        )
        return ids.to(self._device) if to_device else ids

    def _finalize_generation(
        self, generated_ids: list[int], elapsed: float,
        vector_snapshot: dict[str, float], prompt_tokens: int = 0,
        stateless: bool = False,
        logprobs_list: list[tuple[int, float, list[Any]]] | None = None,
        applied_steering: str | None = None,
        *,
        return_hidden: bool = False,
        return_probe_readings: bool = True,
        assistant_node_id: str | None = None,
        mean_logprob: float | None = None,
        mean_surprise: float | None = None,
    ) -> GenerationResult:
        """Finalize result state through the generation finalizer collaborator."""
        from saklas.core.generation_finalizer import finalize_generation

        return finalize_generation(
            self, generated_ids, elapsed, vector_snapshot,
            prompt_tokens=prompt_tokens,
            stateless=stateless,
            logprobs_list=logprobs_list,
            applied_steering=applied_steering,
            return_hidden=return_hidden,
            return_probe_readings=return_probe_readings,
            assistant_node_id=assistant_node_id,
            mean_logprob=mean_logprob,
            mean_surprise=mean_surprise,
            min_elapsed_for_rate=MIN_ELAPSED_FOR_RATE,
        )

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

    def _generation_preamble(self, input: Any, raw: bool, thinking: bool, stateless: bool = False,
                             parent_node_id: str | None = None,
                             user_role: str | None = None,
                             assistant_role: str | None = None,
                             gen_seat: str = "assistant"):
        """Shared input prep + gen-state reset.

        Steering is NOT installed here — the caller is expected to hold a
        ``session.steering()`` scope open across the generation.
        ``parent_node_id`` selects which loom-tree path the input is
        anchored against (default: the active path).  ``user_role`` /
        ``assistant_role`` are this send's per-message role labels (the
        roleplay scaffold): the new user turn renders under ``user_role``
        and the generation prompt under ``assistant_role`` (a role-bearing
        steering scope overrides the latter inside ``_prepare_input``).
        """
        use_thinking = thinking and supports_thinking(self._tokenizer)
        input_ids = self._prepare_input(
            input, raw=raw, thinking=use_thinking, stateless=stateless,
            parent_node_id=parent_node_id,
            user_role=user_role, assistant_role=assistant_role,
            gen_seat=gen_seat,
        )
        self._gen_state.reset()
        return input_ids, use_thinking, int(input_ids.shape[1])

    def _tokenize_authored_prompt_text(
        self,
        node_id: str,
        text: str,
        *,
        thinking: bool,
    ) -> _AuthoredPromptTarget | None:
        """Tokenize visible authored text without changing its rendered bytes.

        Fast-tokenizer offsets let the loom keep the user's exact whitespace
        while assigning each span to the model token that consumed it.  A slow
        tokenizer falls back to one-token decodes only when they concatenate
        byte-for-byte to the original text; otherwise the channel is left plain
        instead of replacing authored copy with a normalized reconstruction.
        """
        token_ids = tuple(int(tid) for tid in self._tokenizer.encode(
            text, add_special_tokens=False,
        ))
        if not token_ids:
            return None

        pieces: tuple[str, ...] | None = None
        try:
            encoded = self._tokenizer(
                text,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            encoded_ids = encoded["input_ids"]
            offsets = encoded["offset_mapping"]
            if isinstance(encoded_ids, torch.Tensor):
                encoded_ids = encoded_ids.detach().to("cpu").tolist()
            if encoded_ids and isinstance(encoded_ids[0], list):
                encoded_ids = encoded_ids[0]
            if isinstance(offsets, torch.Tensor):
                offsets = offsets.detach().to("cpu").tolist()
            if offsets and isinstance(offsets[0], list) and len(offsets) == 1:
                offsets = offsets[0]
            if tuple(int(tid) for tid in encoded_ids) == token_ids:
                cursor = 0
                spans: list[str] = []
                valid = len(offsets) == len(token_ids)
                for raw_start, raw_end in offsets:
                    start, end = int(raw_start), int(raw_end)
                    if start < 0 or end < start or end > len(text) or end < cursor:
                        valid = False
                        break
                    # Attach gaps (usually leading spaces omitted by a word
                    # token's formal offset) to the following visible token.
                    spans.append(text[cursor:end])
                    cursor = end
                if valid and spans:
                    spans[-1] += text[cursor:]
                    if "".join(spans) == text:
                        pieces = tuple(spans)
        except (KeyError, TypeError, ValueError, NotImplementedError):
            pass

        if pieces is None:
            decoded: list[str] = []
            try:
                for tid in token_ids:
                    try:
                        piece = self._tokenizer.decode(
                            [tid],
                            skip_special_tokens=False,
                            clean_up_tokenization_spaces=False,
                        )
                    except TypeError:
                        piece = self._tokenizer.decode(
                            [tid], skip_special_tokens=False,
                        )
                    decoded.append(str(piece))
            except (AttributeError, TypeError, ValueError):
                return None
            if "".join(decoded) != text:
                return None
            pieces = tuple(decoded)

        return _AuthoredPromptTarget(
            node_id=node_id,
            thinking=thinking,
            token_ids=token_ids,
            token_texts=pieces,
        )

    def _pending_authored_prompt_targets(
        self,
        assistant_node_id: str | None,
        *,
        raw: bool,
    ) -> list[_AuthoredPromptTarget]:
        """Uncaptured authored channels on the prompt path, in render order."""
        if assistant_node_id is None:
            return []
        generated = self.tree.nodes.get(assistant_node_id)
        if generated is None or generated.parent_id is None:
            return []
        targets: list[_AuthoredPromptTarget] = []
        for node in self.tree.path_to(generated.parent_id):
            if node.id == self.tree.root_id or node.recipe is not None:
                continue
            # Raw/flat rendering ignores thinking_text; chat rendering places
            # authored thinking before the visible response content.
            if not raw and node.thinking_text and node.thinking_tokens is None:
                target = self._tokenize_authored_prompt_text(
                    node.id, node.thinking_text, thinking=True,
                )
                if target is not None:
                    targets.append(target)
            if node.text and node.tokens is None:
                target = self._tokenize_authored_prompt_text(
                    node.id, node.text, thinking=False,
                )
                if target is not None:
                    targets.append(target)
        return targets

    @staticmethod
    def _rfind_token_subsequence(
        haystack: list[int],
        needle: tuple[int, ...],
        end: int,
    ) -> int:
        """Rightmost exact token subsequence ending at or before ``end``."""
        width = len(needle)
        for start in range(min(end, len(haystack)) - width, -1, -1):
            if tuple(haystack[start:start + width]) == needle:
                return start
        return -1

    def _match_authored_prompt_targets(
        self,
        targets: list[_AuthoredPromptTarget],
        input_ids: torch.Tensor,
        *,
        cache_position_offset: int,
        search_end: int | None = None,
    ) -> tuple[list[_AuthoredPromptMatch], list[int]]:
        """Locate visible text in the full prompt and map producer rows.

        Matching walks backward because the generation prompt follows the last
        visible message and duplicate authored lines are common during rerolls.
        Positions are first resolved in the full prompt, then translated into
        the suffix-local coordinates of a KV-prefix hit.  The first suffix token
        has no producer row in the new forward and is therefore left unscored.
        """
        full_ids = [
            int(tid) for tid in input_ids[0].detach().to("cpu").tolist()
        ]
        limit = min(
            len(full_ids),
            int(search_end) if search_end is not None else len(full_ids),
        )
        reversed_matches: list[_AuthoredPromptMatch] = []
        for target in reversed(targets):
            start = self._rfind_token_subsequence(
                full_ids, target.token_ids, limit,
            )
            # Gemma-family templates trim message content before tokenizing.
            # Preserve the loom's verbatim authored whitespace, but if the
            # exact standalone token sequence is absent, retry the stripped
            # content and attach the trimmed edge bytes to its first/last
            # visible token span. Whitespace the template removed has no model
            # position of its own, so this is the only faithful display mapping.
            if start < 0:
                original_text = "".join(target.token_texts)
                stripped_text = original_text.strip()
                if stripped_text and stripped_text != original_text:
                    stripped_target = self._tokenize_authored_prompt_text(
                        target.node_id,
                        stripped_text,
                        thinking=target.thinking,
                    )
                    if stripped_target is not None:
                        pieces = list(stripped_target.token_texts)
                        left = original_text[:len(original_text) - len(
                            original_text.lstrip()
                        )]
                        right = original_text[len(original_text.rstrip()):]
                        pieces[0] = left + pieces[0]
                        pieces[-1] += right
                        stripped_target.token_texts = tuple(pieces)
                        stripped_start = self._rfind_token_subsequence(
                            full_ids, stripped_target.token_ids, limit,
                        )
                        if stripped_start >= 0:
                            target = stripped_target
                            start = stripped_start
            if start < 0:
                _log.debug(
                    "authored prompt capture skipped node %s: content tokens "
                    "not found verbatim in rendered prompt",
                    target.node_id,
                )
                continue
            positions = tuple(range(start, start + len(target.token_ids)))
            reversed_matches.append(_AuthoredPromptMatch(target, positions))
            limit = start
        matches = list(reversed(reversed_matches))

        effective_producers = sorted({
            token_position - 1 - cache_position_offset
            for match in matches
            for token_position in match.token_positions
            if token_position - 1 >= cache_position_offset
        })
        producer_to_column = {
            producer: column
            for column, producer in enumerate(effective_producers)
        }
        for match in matches:
            match.capture_columns = tuple(
                producer_to_column.get(
                    token_position - 1 - cache_position_offset
                )
                if token_position - 1 >= cache_position_offset
                else None
                for token_position in match.token_positions
            )
        return matches, effective_producers

    def _authored_lens_capture(
        self,
        hidden: dict[int, torch.Tensor],
        *,
        top_k: int,
    ) -> tuple[
        dict[int, list[tuple[str, float]]],
        list[tuple[str, float, float, float]],
        dict[int, list[int]],
        dict[str, ProbeReading],
    ] | None:
        """Live J-LENS payload for one retained authored producer row
        (delegates to :meth:`LensInstrument.authored_capture`; the
        token-matching orchestration stays session-side)."""
        return self._lens_instrument.authored_capture(hidden, top_k=top_k)

    def _authored_sae_capture(
        self,
        hidden: dict[int, torch.Tensor],
    ) -> tuple[
        list[tuple[int, float, str | None, float | None]],
        dict[str, ProbeReading],
    ] | None:
        """Live SAE payload for one retained authored producer row
        (delegates to :meth:`SaeInstrument.authored_capture`)."""
        return self._sae_instrument.authored_capture(hidden)

    def _persist_authored_prompt_captures(
        self,
        matches: list[_AuthoredPromptMatch],
        *,
        monitor_active: bool,
        lens_active: bool,
        sae_active: bool,
        lens_top_k: int,
        persist_per_layer_scores: bool,
        steering: str | None,
    ) -> None:
        """Score selected prefill rows and attach them to authored loom nodes."""
        prompt_rows = self._capture.prompt_stacked()
        if not matches or not prompt_rows:
            return
        from saklas.core.token_payloads import TokenProbePayload

        output_rows: list[list[dict[str, Any]]] = [
            [
                {
                    "token_id": int(tid),
                    "text": text,
                    "logprob": None,
                    "perplexity": None,
                }
                for tid, text in zip(
                    match.target.token_ids,
                    match.target.token_texts,
                    strict=True,
                )
            ]
            for match in matches
        ]
        ordered: list[tuple[int, int, int, int]] = []
        for match_index, match in enumerate(matches):
            for token_index, (token_position, column) in enumerate(zip(
                match.token_positions, match.capture_columns, strict=True,
            )):
                if column is not None:
                    ordered.append((token_position, match_index, token_index, column))

        populated_matches: set[int] = set()
        for _position, match_index, token_index, column in sorted(ordered):
            hidden = {
                layer: rows[column]
                for layer, rows in prompt_rows.items()
                if column < int(rows.shape[0])
            }
            if not hidden:
                continue
            payload = TokenProbePayload()
            if monitor_active:
                payload.merge_readings(
                    self._monitor.score_single_token(hidden),
                    family="geometry",
                    per_layer=persist_per_layer_scores,
                )

            lens_payload = lens_aggregate = lens_token_ids = None
            if lens_active:
                lens_capture = self._authored_lens_capture(
                    hidden, top_k=lens_top_k,
                )
                if lens_capture is not None:
                    (
                        lens_payload,
                        lens_aggregate,
                        lens_token_ids,
                        lens_readings,
                    ) = lens_capture
                    payload.merge_readings(
                        lens_readings,
                        family="lens",
                        per_layer=persist_per_layer_scores,
                    )

            sae_payload = None
            if sae_active:
                sae_capture = self._authored_sae_capture(hidden)
                if sae_capture is not None:
                    sae_payload, sae_readings = sae_capture
                    payload.merge_readings(
                        sae_readings,
                        family="sae",
                        per_layer=persist_per_layer_scores,
                    )

            shaped = payload.to_token_payload(
                lens=lens_payload,
                lens_aggregate=lens_aggregate,
                lens_token_ids=lens_token_ids,
                lens_source=(
                    self._live_lens.get("source")
                    if self._live_lens is not None else None
                ),
                sae=sae_payload,
                sae_source=(
                    self._live_sae.get("source")
                    if self._live_sae is not None else None
                ),
                sae_layer=(
                    int(self._live_sae["layer"])
                    if self._live_sae is not None else None
                ),
                steering=steering,
            )
            measurements = shaped.get("measurements")
            if not measurements:
                continue
            row = output_rows[match_index][token_index]
            if payload.scores:
                row["probes"] = {
                    name: round(float(value), 6)
                    for name, value in payload.scores.items()
                }
            if payload.per_layer_scores:
                row["per_layer_scores"] = payload.per_layer_scores
            row["measurements"] = measurements
            populated_matches.add(match_index)

        for match_index in sorted(populated_matches):
            match = matches[match_index]
            self.tree.set_authored_token_scores(
                match.target.node_id,
                output_rows[match_index],
                thinking=match.target.thinking,
            )

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
        overrides: dict[str, Any] = {}
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

    def _seat_stop_augmentation(
        self,
        stop_list: list[str] | None,
        *,
        gen_seat: str,
        raw: bool,
    ) -> list[str] | None:
        """Add the gen seat's close segment as a stop string when needed.

        Convention 2 (the cast model): generating into a seat stops on
        that seat's terminator.  On every validated family the two seats
        share a terminator, which the engine's EOS-id union already
        catches — so the assistant path (and any shared-close family)
        adds nothing and keeps its zero-cost no-stop-list fast path.
        Only a non-assistant-seat gen on a family whose seat closes
        *differ* (gpt-oss: user ``<|end|>`` vs assistant ``<|return|>``)
        gains a stop string, guaranteeing the stop even where the
        seat's terminator escapes the EOS union.  Raw / flat mode has
        no seat closes.
        """
        if raw or gen_seat == "assistant":
            return stop_list
        grammar = self.scene_grammar
        if grammar is None:
            return stop_list
        close = grammar.seat(gen_seat).close.strip()
        if not close or close == grammar.assistant.close.strip():
            return stop_list
        out = list(stop_list) if stop_list else []
        if close not in out:
            out.append(close)
        return out

    def _apply_cast_defaults(
        self,
        steering: "str | Steering | None",
        sampling: SamplingConfig | None,
        thinking: bool | None,
        *,
        raw: bool,
        gen_seat: str,
    ) -> tuple["str | Steering | None", SamplingConfig | None, bool | None]:
        """Fill unset per-call kwargs from the gen label's cast recipe.

        The cast roster (phase 3) is the *weakest* tier: the member's
        recipe fills only fields the call left unset (``steering=None``
        means "unset" here; pass ``""`` for an explicit unsteered
        override), sampling merges field-wise with the call's
        non-default fields winning, and the member's seed applies only
        when the call's sampling doesn't pin one.  The label must ride
        the call's sampling box (``user_role`` for a user-seat gen,
        ``assistant_role`` otherwise) — a bare continuation of a cast
        turn doesn't re-trigger the recipe.  Raw / flat mode renders no
        labels, so the roster doesn't apply there.
        """
        cast_label: str | None = None
        if not raw:
            explicit_label = None
            if sampling is not None:
                explicit_label = (
                    sampling.user_role if gen_seat == "user"
                    else sampling.assistant_role
                )
            cast_label = explicit_label or gen_seat
        member = self.tree.cast.get(cast_label) if cast_label else None
        if member is None or member.recipe is None:
            return steering, sampling, thinking
        base = member.recipe
        if steering is None and base.steering is not None:
            steering = base.steering
        if thinking is None and base.thinking is not None:
            thinking = base.thinking
        if base.sampling is not None:
            sampling = base.sampling.merged_with(sampling)
        if base.seed is not None and (
            sampling is None or sampling.seed is None
        ):
            from dataclasses import replace as _replace
            sampling = _replace(
                sampling or SamplingConfig(), seed=base.seed,
            )
        return steering, sampling, thinking

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

        # Resolve the anchor recipe — the parent assistant's recipe when
        # present, else the nearest assistant ancestor's, else an empty
        # Recipe (see ``_resolve_anchor_recipe``).
        anchor = self._resolve_anchor_recipe(parent_node_id)

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

    def _effective_return_top_k(self, sampling: SamplingConfig | None) -> int:
        """Resolve the generation's logit-alternative width.

        Per-call ``return_top_k`` wins when positive; zero inherits the
        session default. The same resolved width drives both the ordinary
        logit alternatives and the live J-lens readout.
        """
        top_k = sampling.return_top_k if sampling is not None else 0
        return int(top_k or self._default_return_top_k)

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
        list[Any] | None,
    ]:
        """Normalize per-call generation controls before model work."""
        steering_obj = Steering.from_value(
            steering, profile_names=set(self._profiles),
        )
        if thinking is None:
            config_thinking = getattr(self.config, "thinking", None)
            if steering_obj is not None and steering_obj.thinking is not None:
                use_thinking_req = bool(steering_obj.thinking)
            elif config_thinking is not None:
                use_thinking_req = bool(config_thinking)
            else:
                use_thinking_req = supports_thinking(self._tokenizer)
        else:
            use_thinking_req = bool(thinking)

        gen_config = self._compose_gen_config(sampling)
        raw_lp = sampling.logprobs if sampling is not None else None
        raw_top_k = self._effective_return_top_k(sampling)
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
        logprobs_list: list[Any] | None = [] if raw_lp is not None else None
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
        """Flatten the active steering stack for result receipts."""
        return self._steering_composer.snapshot_steering_alphas()

    def _start_loom_assistant(
        self,
        input: Any,
        *,
        stateless: bool,
        raw: bool,
        parent_node_id: str | None,
        sampling: SamplingConfig | None,
        steering_obj: Steering | None,
        use_thinking_req: bool,
        gen_seat: str = "assistant",
        continuation_node_id: str | None = None,
    ) -> str | None:
        """Create the loom nodes for a stateful generation.

        ``gen_seat`` is the seat the generated node occupies (the cast
        model: "generated" is provenance, not a seat).  An explicit
        non-assistant seat implies scene mode, where seating is free — the
        user-under-user guard is skipped. ``continuation_node_id`` reuses an
        existing same-role message whose text the caller will replay as the
        forced decode prefix.
        """
        if stateless:
            return None

        # Per-message role labels (roleplay scaffold) ride this send's
        # SamplingConfig and are stamped onto the nodes at creation — the
        # turn keeps its role regardless of later box changes.  Raw / flat
        # mode has no chat template, so role labels don't apply there.
        user_role = sampling.user_role if sampling is not None and not raw else None
        assistant_role = (
            sampling.assistant_role if sampling is not None and not raw else None
        )

        if isinstance(input, str) and raw:
            # Flat (base-model) path: no user/assistant turn pairing.  A
            # non-empty ``input`` is a typed span recorded as its own
            # node; an empty ``input`` is a bare continuation that just
            # extends the active leaf.  ``_check_user_send_target`` is
            # skipped — in flat mode the role tag is authorship
            # provenance, not a turn-taking constraint, so a span may
            # hang under a node of any role.
            if input != "":
                user_node_id = self.tree.add_user_turn(
                    input, parent_id=parent_node_id)
            else:
                user_node_id = parent_node_id or self.tree.active_node_id
        elif isinstance(input, str):
            if gen_seat == "assistant":
                self._check_user_send_target(parent_node_id)
            user_node_id = self.tree.add_user_turn(
                input, parent_id=parent_node_id, role_label=user_role)
        else:
            # ``input=None`` (generate — no new authored turn) or a raw
            # messages list: anchor directly on the parent.
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
        # The generated node's label follows its seat (user-seat gens are
        # labeled by the ``user_role`` box, assistant-seat by ``assistant_role``).
        gen_label = user_role if gen_seat == "user" else assistant_role
        if continuation_node_id is not None:
            return self.tree.begin_continuation(
                continuation_node_id, recipe=recipe,
            )
        return self.tree.begin_assistant(
            user_node_id, recipe=recipe, role_label=gen_label, seat=gen_seat)

    def _compiled_graph_thread_safe(self) -> bool:
        """Whether this thread owns PyTorch's warmed CUDA graph tree."""
        if not getattr(self, "_cuda_graphs_active", False):
            return True
        return (
            threading.get_ident()
            == getattr(self, "_compile_owner_thread", threading.get_ident())
        )

    def _acquire_generation_static_cache(
        self,
        model: PreTrainedModel,
        *,
        max_cache_len: int,
    ) -> object:
        """Return a reset, identity-stable StaticCache with enough capacity."""
        cached = self._generation_static_cache
        if cached is not None and self._generation_static_cache_len >= max_cache_len:
            reset = getattr(cached, "reset", None)
            if callable(reset):
                reset()
                return cached
            self._generation_static_cache = None
            self._generation_static_cache_len = 0

        from saklas.core.cuda_graphs import make_static_cache

        model_dtype = next(model.parameters()).dtype
        cache = make_static_cache(
            model,
            max_cache_len=max_cache_len,
            device=self._device,
            dtype=model_dtype,
        )
        self._generation_static_cache = cache
        self._generation_static_cache_len = max_cache_len
        return cache

    def _run_generation_loop(
        self,
        input_ids: torch.Tensor,
        gen_config: GenerationConfig,
        *,
        use_thinking: bool,
        want_hidden: bool,
        effective_tap: "StepTokenCallback | None",
        seed: int | None,
        stop_list: list[str] | None,
        logit_bias: dict[int, float] | None,
        presence_penalty: float,
        frequency_penalty: float,
        lp_count: int | None,
        forced_prefix: list[int] | None = None,
        want_perplexity: bool = True,
        cache_token_text: bool = True,
        authored_capture: _AuthoredPromptCapture | None = None,
    ) -> tuple[list[int], float]:
        """Run the decode loop once capture and steering are installed."""
        cached_pkv = None
        cache_position_offset = 0
        cached_static = False
        effective_input_ids = input_ids
        static_cache_eligible = (
            self._static_cache_active
            and self._compiled_graph_thread_safe()
            and (
                self._steering.all_fast_path()
                or self._steering.static_steerable()
            )
        )
        # The cached prefix KV is unsteered.  It's safe to reuse whenever the
        # active steering doesn't touch the prefill region — ``@response`` /
        # ``@generated`` / probe-gated steering leaves the prompt KV identical
        # to unsteered, so a steered (response-phase) batch can still share one
        # prefill.  ``want_hidden`` (needs every prefix hidden state) and
        # thinking (different decode path) still force a full re-prefill.
        if (
            not want_hidden
            and not use_thinking
            and not self._steering_active_in_prefill()
            and self._prefix_cache is not None
        ):
            hit = self._try_prefix_cache_hit(
                input_ids,
                static_eligible=static_cache_eligible,
                required_max_new_tokens=gen_config.max_new_tokens,
            )
            if hit is not None:
                suffix_ids, cached_pkv, cache_position_offset, cached_static = hit
                effective_input_ids = suffix_ids

        authored_matches: list[_AuthoredPromptMatch] = []
        if authored_capture is not None:
            authored_matches, prompt_positions = self._match_authored_prompt_targets(
                authored_capture.targets,
                input_ids,
                cache_position_offset=cache_position_offset,
                search_end=authored_capture.search_end,
            )
            self._capture.set_prompt_positions(prompt_positions)

        base_step_callback = (
            self._capture.ingest_persistent
            if self._capture_state.persistent
            else self._capture.fire_step_sink
        )
        prompt_capture_pending = bool(authored_matches)

        def _step_callback(step_id: int) -> None:
            nonlocal prompt_capture_pending
            # The first callback follows the full/suffix prefill.  Score the
            # visible authored positions in prompt order before the ordinary
            # final-prompt sink advances curved monitor feet to the state that
            # produces generated token 0.
            if prompt_capture_pending and authored_capture is not None:
                prompt_capture_pending = False
                self._persist_authored_prompt_captures(
                    authored_matches,
                    monitor_active=authored_capture.monitor_active,
                    lens_active=authored_capture.lens_active,
                    sae_active=authored_capture.sae_active,
                    lens_top_k=authored_capture.lens_top_k,
                    persist_per_layer_scores=(
                        authored_capture.persist_per_layer_scores
                    ),
                    steering=authored_capture.steering,
                )
            base_step_callback(step_id)

        start = time.monotonic()
        composer = self._steering_composer
        gating_callback = (
            composer.build_gating_score_callback()
            if composer.steering_needs_probe_gating()
            else None
        )
        # StaticCache (+ compile fusion on MPS, + CUDA-graph capture on CUDA) is
        # eligible when generation is unsteered (``all_fast_path``) OR every
        # steered layer is the static single-affine fast path
        # (``static_steerable`` — Trigger.BOTH affine slide, no ctx, no foot, no
        # gate).  StaticCache never bypasses the forward hooks, so the steering
        # still applies; only the KV buffers are preallocated and the decode
        # shape stays fixed (what lets the compiled graph stay specialized).
        # ``_static_cache_active`` is device-agnostic (set when compile stuck or
        # CUDA graphs are on).  Curved / gated / phased steering keeps the eager
        # DynamicCache path.
        use_static_cache = static_cache_eligible and (
            cached_pkv is None or cached_static
        )
        # Compiled-model routing (CUDA/MPS). The graph was traced with ONLY the
        # persistent branchless offset + capture hooks present (attached
        # pre-compile), so the compiled module is correct exactly when the live
        # hook topology matches that: no *transient* per-token capture hooks
        # (``self._capture.is_transient()`` false — true for the persistent-capture
        # and no-probe paths) AND either unsteered or a static-affine steering that
        # lowered to the persistent offset buffers (``_steering_uses_compiled_offsets``
        # — the push rides the offset, no transient hook).  A probed gen on the
        # persistent-capture path (slice 2) satisfies both, so it now keeps the
        # compiled graph.  Everything else (transient ctx-consulting steering
        # hooks, transient capture hooks for curved/gated/return_hidden) graph-breaks
        # / recompiles, so route it to the eager original (``_orig_mod``) +
        # DynamicCache — where the same persistent hooks still apply any offsets
        # and capture, so correctness holds and ``compile=True`` never regresses
        # the hooked path. On CUDA this boundary also avoids Dynamo retracing on
        # every per-generation hook topology.
        gen_model = self._model
        steering_uses_compiled_offsets = bool(
            self._steering_uses_compiled_offsets
        )
        if self._compiled and self._device.type in {"cuda", "mps"}:
            compiled_clean = not self._capture.is_transient() and (
                steering_uses_compiled_offsets
                or self._steering.all_fast_path()
            )
            if compiled_clean and self._compiled_graph_thread_safe():
                use_static_cache = self._static_cache_active and (
                    cached_pkv is None or cached_static
                )
            else:
                gen_model = getattr(self._model, "_orig_mod", self._model)
                use_static_cache = False
        generation_pkv = cached_pkv
        if use_static_cache and generation_pkv is None:
            # Reserve at least the session default decode window and bucket the
            # total length.  Most interactive requests then keep one cache
            # allocation and one set of Dynamo guards even when their per-call
            # max_tokens differ.  The session lock serializes access, so reset +
            # reuse cannot race another generation.
            reserve_tokens = max(
                int(gen_config.max_new_tokens),
                int(self.config.max_new_tokens),
                1,
            )
            required_len = (
                cache_position_offset
                + int(effective_input_ids.shape[1])
                + reserve_tokens
            )
            cache_len = ((required_len + 255) // 256) * 256
            try:
                generation_pkv = self._acquire_generation_static_cache(
                    gen_model,
                    max_cache_len=cache_len,
                )
            except Exception as exc:
                _log.warning(
                    "Reusable StaticCache allocation failed (%s: %s); "
                    "falling back to DynamicCache",
                    type(exc).__name__,
                    exc,
                )
                use_static_cache = False
        generated_ids = generate_steered(
            gen_model, self._tokenizer, effective_input_ids,
            gen_config, self._gen_state, thinking=use_thinking,
            on_token=effective_tap,
            seed=seed, stop=stop_list, logit_bias=logit_bias,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            logprobs=lp_count,
            trigger_ctx=self._steering.ctx,
            past_key_values=generation_pkv,
            cache_position_offset=cache_position_offset,
            score_callback=gating_callback,
            # Per-token probe scoring fires post-forward (FIX F1), not inside the
            # capture hook.  On the persistent-capture path the step callback is
            # ``ingest_persistent`` (accumulate from the persistent buffers +
            # fire the step sink); otherwise ``fire_step_sink`` (the transient
            # hooks already accumulated in-forward).  Both no-op when no per-token
            # sink is installed (aggregate-only / full-retention / no-probe).
            step_callback=_step_callback,
            use_static_cache=use_static_cache,
            forced_prefix=forced_prefix,
            steering_active=bool(
                self._steering.hooks or steering_uses_compiled_offsets
            ),
            want_perplexity=want_perplexity,
            cache_token_text=cache_token_text,
        )
        elapsed = time.monotonic() - start

        if cached_pkv is not None and self._prefix_cache is not None:
            try:
                cached_pkv.crop(cache_position_offset)  # pyright: ignore[reportAttributeAccessIssue]  # HF cache object; crop() is present at runtime
            except (AttributeError, TypeError):
                self._invalidate_prefix_cache()
        return generated_ids, elapsed

    def _generate_core(
        self,
        input: Any,
        *,
        steering: "str | Steering | None" = None,
        sampling: SamplingConfig | None = None,
        stateless: bool = False,
        raw: bool = False,
        thinking: bool | None = None,
        on_token: TokenCallback | None = None,
        parent_node_id: str | None = None,
        recipe_override: "Recipe | str | None" = None,
        forced_prefix: list[int] | None = None,
        gen_seat: str = "assistant",
        append_same_role: bool = False,
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
        steering_cm = None
        try:

            # Cast roster (phase 3): the gen label's standing recipe is
            # the weakest tier; the regen override below still composes
            # on top of the result.
            steering, sampling, thinking = self._apply_cast_defaults(
                steering, sampling, thinking, raw=raw, gen_seat=gen_seat,
            )

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

            # Ordinary one-shot continuation is message-oriented: if the
            # selected leaf already occupies the same structural seat under
            # the same effective role label, regenerate that node in place
            # with its current text as the forced prefix.  Specialist fan,
            # fork, and prefill paths leave this flag off and retain their
            # explicit sibling semantics.
            continuation_node_id: str | None = None
            continuation_prefix: list[int] | None = None
            continuation_parent_id: str | None = None
            if (
                append_same_role
                and not stateless
                and forced_prefix is None
                and (input is None or (raw and input == ""))
            ):
                candidate_id = parent_node_id or self.tree.active_node_id
                candidate = self.tree.get(candidate_id)
                candidate_seat = (
                    "user" if gen_seat == "user" else "assistant"
                )
                generated_label = None
                if sampling is not None and not raw:
                    generated_label = (
                        sampling.user_role
                        if gen_seat == "user"
                        else sampling.assistant_role
                    )
                if (
                    candidate.role == candidate_seat
                    and candidate.role_label == generated_label
                ):
                    continuation_node_id = candidate.id
                    continuation_parent_id = candidate.parent_id
                    continuation_prefix = list(
                        self._tokenizer.encode(
                            candidate.text, add_special_tokens=False,
                        )
                    )
                    if continuation_prefix:
                        from dataclasses import replace as _replace
                        base_sampling = (
                            sampling if sampling is not None else SamplingConfig()
                        )
                        base_max = (
                            base_sampling.max_tokens
                            or self.config.max_new_tokens
                        )
                        sampling = _replace(
                            base_sampling,
                            max_tokens=len(continuation_prefix) + base_max,
                        )
                    # Same-message continuation is an answer/body prefill,
                    # not a request to open a fresh thinking channel.
                    thinking = False

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
            jlens_top_k = (
                self._effective_return_top_k(sampling)
                or _DEFAULT_JLENS_TOP_K
            )
            # Steering synthesis is Mahalanobis-normalized and therefore needs
            # the session whitener.  On a cold ``probes=[]`` session it may not
            # exist yet.  Prime it here while this generation already owns the
            # model lock but before the steering scope is entered; asking the
            # public lazy property from inside PREAMBLE would trip its deliberate
            # generation-phase guard (the first-use J-LENS steering regression).
            if (
                steering_obj is not None
                and steering_obj.alphas
                and self._whitener is None
            ):
                self._install_whitener_if_missing()
            stop_list = self._seat_stop_augmentation(
                stop_list, gen_seat=gen_seat, raw=raw,
            )
            return_probe_readings = bool(
                sampling is None or sampling.return_probe_readings
            )
            # ``mean_logprob_accum`` averages chosen-token logprobs over the
            # non-thinking response span — surfaced on ``LoomNode.mean_logprob``
            # at finalize-assistant time so the loom sidebar can sort siblings
            # by surprise.  Populates whenever the engine captures chosen
            # logprob (any ``on_token`` consumer with ``lp_count is not None``
            # — i.e. the loom path or an explicit logprobs request).
            mean_logprob_sum: float = 0.0
            mean_logprob_count: int = 0
            trait_token_counter = [0]
            # CAA live toggle: when off, every per-token monitor consumer is
            # masked at the source — generations run aggregate-only capture
            # (probes still report the end-of-gen aggregate) and only probe
            # gates can force a per-token subset.
            _live_scores_on = self._live_probe_scores
            _callback_options = consumer_options(on_token)
            _wants_live_token_scores = bool(
                _live_scores_on
                and on_token is not None
                and _callback_options.live_scores
            )
            _persists_layer_scores = bool(
                _live_scores_on
                and (
                    (sampling is not None and sampling.persist_per_layer_scores)
                    or (
                        on_token is not None and _callback_options.per_layer_scores
                    )
                )
            )
            # Probe-inspector live point + fading trail: stamp each token's probe
            # reading with per-layer whitened subspace coords.  Gates the monitor
            # post-pass (set here, reset in the teardown ``finally``) and forces
            # per-token scoring so the per-token reading actually exists to carry it.
            _persists_subspace_coords = bool(
                _live_scores_on
                and sampling is not None
                and sampling.persist_subspace_coords
            )
            self._monitor.set_subspace_coords(_persists_subspace_coords)
            from saklas.core.token_payloads import (
                TokenProbePayload,
                build_token_probe_payload,
            )

            _has_monitor_probes = bool(self._monitor.probe_names)

            def _token_tap(
                text: str, is_thinking: bool, tid: int | None, lp: float | None,
                top_alts: list[TokenAlt] | None, perplexity: float | None = None,
                step_id: int = -1,
            ) -> None:
                # ``step_id`` is the decode loop's forward index (the internal
                # StepTokenCallback contract) — the SAME value this forward's
                # gate callback received, so the lens/SAE display steps below
                # can reuse the gate's step-keyed stash rows and the geometry
                # payload can hit the run's observe memo.  User callbacks keep
                # the public six-argument TokenCallback shape (invoked below).
                nonlocal mean_logprob_sum, mean_logprob_count
                self._last_token_probe_payload = None
                self._last_token_probe_readings = None
                if logprobs_list is not None and tid is not None and tid >= 0 and not is_thinking:
                    logprobs_list.append((tid, lp if lp is not None else 0.0, top_alts or []))
                if lp is not None and tid is not None and tid >= 0 and not is_thinking:
                    mean_logprob_sum += lp
                    mean_logprob_count += 1
                needs_scores = bool(
                    _live_scores_on
                    and _has_monitor_probes
                    and (
                        assistant_node_id is not None
                        or _has_trait_consumer
                        or _wants_live_token_scores
                        or _persists_subspace_coords
                    )
                )
                payload = (
                    build_token_probe_payload(
                        monitor=self._monitor,
                        capture=self._capture,
                        capture_state=self._capture_state,
                        incremental_readings=self._incremental_readings,
                        geometry_run=self._geometry_instrument.current_run,
                        step_id=step_id,
                        needs_scores=needs_scores,
                        persists_layer_scores=_persists_layer_scores,
                        assistant_node_id=assistant_node_id,
                    )
                    if needs_scores
                    else None
                )
                # Live J-lens readout (None when off): the step's top-k
                # lens tokens per selected layer + the layer-aggregated
                # chip list.  ``step_id`` lets the display reuse the gate
                # callback's stash rows iff they came from THIS forward
                # (step identity replaced the old freshness handshake).
                lens_step = (
                    self._live_lens_readout_step(
                        top_k=jlens_top_k, step_id=step_id,
                    )
                    if _has_lens_consumer
                    else None
                )
                sae_step = (
                    self._live_sae_readout_step(step_id=step_id)
                    if _has_sae_consumer
                    else None
                )
                # Pinned lens probes tick on the same step (readings extracted
                # from the display logits inside the readout step) — merge them
                # into every populated probe channel so the loom row, trait
                # stream, and WS frames carry them uniformly.
                if lens_step is not None and self._last_lens_step_readings:
                    if payload is None:
                        payload = TokenProbePayload()
                    payload.merge_readings(
                        self._last_lens_step_readings,
                        family="lens",
                        per_layer=(
                            assistant_node_id is not None
                            and _persists_layer_scores
                        ),
                    )
                if sae_step is not None and self._last_sae_step_readings:
                    if payload is None:
                        payload = TokenProbePayload()
                    payload.merge_readings(
                        self._last_sae_step_readings,
                        family="sae",
                        per_layer=(
                            assistant_node_id is not None
                            and _persists_layer_scores
                        ),
                    )
                scores = payload.scores if payload is not None else None
                per_layer_payload = (
                    payload.per_layer_scores if payload is not None else None
                )
                lens_payload = lens_step[0] if lens_step is not None else None
                lens_aggregate_payload = (
                    lens_step[1] if lens_step is not None else None
                )
                lens_token_ids = (
                    lens_step[2]
                    if lens_step is not None and len(lens_step) > 2
                    else None
                )
                if (
                    lens_payload
                    or lens_aggregate_payload
                    or sae_step
                    or (
                        payload is not None
                        and (
                            payload.scores
                            or payload.per_layer_scores
                            or payload.all_readings
                        )
                    )
                ):
                    if payload is None:
                        payload = TokenProbePayload()
                    self._last_token_probe_readings = payload.all_readings or None
                    recipe_steering = None
                    if assistant_node_id is not None:
                        node = self.tree.nodes.get(assistant_node_id)
                        if node is not None and node.recipe is not None:
                            recipe_steering = node.recipe.steering
                    self._last_token_probe_payload = payload.to_token_payload(
                        lens=lens_payload,
                        lens_aggregate=lens_aggregate_payload,
                        lens_token_ids=lens_token_ids,
                        lens_source=(
                            self._live_lens.get("source")
                            if self._live_lens is not None
                            else None
                        ),
                        sae=sae_step,
                        sae_source=(
                            self._live_sae.get("source")
                            if self._live_sae is not None
                            else None
                        ),
                        sae_layer=(
                            int(self._live_sae["layer"])
                            if self._live_sae is not None
                            else None
                        ),
                        steering=recipe_steering,
                    )
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
                    if scores:
                        token_row["probes"] = {
                            p: round(float(v), 6) for p, v in scores.items()
                        }
                    if per_layer_payload:
                        token_row["per_layer_scores"] = per_layer_payload
                    measurements = (
                        self._last_token_probe_payload.get("measurements")
                        if self._last_token_probe_payload is not None
                        else None
                    )
                    if measurements:
                        token_row["measurements"] = measurements
                    self.tree.append_token(
                        assistant_node_id,
                        token_row,
                        thinking=bool(is_thinking),
                    )
                if on_token is not None:
                    on_token(text, is_thinking, tid, lp, top_alts, perplexity)
                # Inline per-token scoring for live SSE trait subscribers.
                if self._trait_queues and _has_monitor_probes and scores:
                    event = ("token", trait_token_counter[0], text, is_thinking, scores)
                    trait_token_counter[0] += 1
                    with self._trait_lock:
                        for lp_ref, q in list(self._trait_queues):
                            with suppress(Exception):
                                lp_ref.call_soon_threadsafe(q.put_nowait, event)

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
            _has_trait_consumer = bool(
                _live_scores_on
                and self._trait_queues
                and _has_monitor_probes
            )
            # The tap also writes per-token ``probes`` / ``per_layer_scores``
            # onto the loom row when probes are loaded and the gen is loom-
            # attached — required so a webui refresh can rehydrate highlight
            # tints and the token-drilldown heatmap from the server tree.
            _persists_probe_row = bool(
                _live_scores_on
                and return_probe_readings
                and not stateless
                and _has_monitor_probes
            )
            _need_tap = (
                on_token is not None
                or logprobs_list is not None
                or _has_trait_consumer
                or _persists_probe_row
                or stop_list is not None
            )
            _has_lens_consumer = bool(
                self._live_lens is not None
                and on_token is not None
                and _callback_options.lens_readout
            )
            _has_sae_consumer = bool(
                self._live_sae is not None
                and on_token is not None
                and _callback_options.sae_readout
            )
            _effective_tap = _token_tap if _need_tap else None
            _tap_has_text_consumer = bool(
                on_token is not None
                or (logprobs_list is not None and (lp_count or 0) > 0)
                or _has_trait_consumer
                or _persists_probe_row
            )

            # Compiled-clean eligibility for this gen, decided BEFORE the steering
            # context enters (``steering_cm.__enter__`` runs
            # ``SteeringComposer.install_composed_steering``, which consults this
            # to decide whether a probed gen may still lower steering to the
            # persistent offset buffers — slice 2). Eligible when the compiled
            # CUDA/MPS graph + static cache are live, the persistent capture
            # buffers were adopted, and the caller didn't ask for the full per-step
            # hidden stack (``return_hidden`` keeps the transient full-retention
            # capture).  Capture then rides the persistent buffers; steering rides the
            # offsets — both compile-clean.
            self._compiled_clean_eligible = bool(
                self._compiled
                and self._device.type in {"cuda", "mps"}
                and self._static_cache_active
                and self._capture_buffers
                and not (sampling and sampling.return_hidden)
            )

            if steering_obj is not None and steering_obj.alphas:
                steering_cm = self.steering(steering_obj)

            # Snapshot the chat-history anchor BEFORE _start_loom_assistant
            # mutates the tree.  Otherwise the new (empty) assistant node it
            # creates becomes the active leaf, and ``messages_for(None)`` later
            # walks INCLUDING that empty assistant — producing a trailing
            # empty-content assistant message AND a duplicated trailing user
            # turn after ``_prepare_input`` re-appends ``input``.  Strict chat
            # templates (Mistral 3) reject the empty assistant outright with
            # ``Assistant message must have a string or a list of chunks ...``;
            # permissive templates render the duplicate user silently and
            # degrade prompt quality.
            chat_history_anchor = (
                continuation_parent_id
                if continuation_node_id is not None
                else parent_node_id
            )
            if (
                not stateless
                and (isinstance(input, str) or input is None)
                and chat_history_anchor is None
            ):
                chat_history_anchor = self.tree.active_node_id

            assistant_node_id = self._start_loom_assistant(
                input,
                stateless=stateless,
                raw=raw,
                parent_node_id=parent_node_id,
                sampling=sampling,
                steering_obj=steering_obj,
                use_thinking_req=use_thinking_req,
                gen_seat=gen_seat,
                continuation_node_id=continuation_node_id,
            )

            if steering_cm is not None:
                steering_cm.__enter__()
            input_ids, use_thinking, prompt_tokens = self._generation_preamble(
                input, raw, use_thinking_req, stateless=stateless,
                parent_node_id=chat_history_anchor,
                user_role=sampling.user_role if sampling is not None else None,
                assistant_role=(
                    sampling.assistant_role if sampling is not None else None
                ),
                gen_seat=gen_seat,
            )
            # Refresh snapshot now that steering is pushed (first-scope case).
            vector_snapshot: dict[str, float] = self._snapshot_steering_alphas()

            want_hidden = bool(sampling and sampling.return_hidden)
            # Per-token scoring is only needed when something consumes a
            # per-token reading: a probe gate, a loom token row, an SSE trait
            # stream, a live-scores client, or a per-layer-heatmap persist.
            # Otherwise (probes attached but only the aggregate wanted, e.g. a
            # stateless server gen) the capture skips per-token scoring entirely
            # and pools the aggregate once at finalize.
            composer = self._steering_composer
            _needs_gating = composer.steering_needs_probe_gating()
            # The five UI / trait / loom / persist consumers each need a
            # per-token reading for the FULL roster.  When NONE of them is live
            # but a probe gate is, gating is the SOLE per-token consumer (FIX
            # #4): the step sink can score just the gated probes per token and
            # leave the big-K roster to the one-shot full aggregate at finalize.
            _per_token_full_consumer = bool(
                (_live_scores_on and assistant_node_id is not None)
                or _has_trait_consumer
                or _wants_live_token_scores
                or _persists_layer_scores
                or _persists_probe_row
                or _persists_subspace_coords
            )
            gating_probe_keys: set[str] | None = (
                composer.gated_probe_keys()
                if _needs_gating
                else None
            )
            lens_gating_probe_keys: set[str] | None = (
                composer.gated_lens_probe_keys()
                if _needs_gating
                else None
            )
            sae_gating_probe_keys: set[str] | None = (
                composer.gated_sae_probe_keys()
                if _needs_gating
                else None
            )
            # J-lens token probes gate on the lens path (the gating score
            # callback computes their scalars from the latest capture slices),
            # so a lens-only gate doesn't force per-token MONITOR scoring —
            # ``need_per_token`` keys on monitor-attached gate keys only.
            _needs_monitor_gating = bool(gating_probe_keys)
            need_per_token = bool(
                _needs_monitor_gating or _per_token_full_consumer
            )
            gating_only_probes: set[str] | None = (
                composer.gated_probe_names()
                if (_needs_monitor_gating and not _per_token_full_consumer)
                else None
            )
            # Lean-incremental (FIX F2): the live consumers present read ONLY the
            # axis-0 coord (the SSE trait stream, the loom probe row, the loom
            # node text/aggregate) — none of the consumers that genuinely need a
            # richer per-token reading is live, and there is no probe gate.  Then
            # the per-token scoring drops the nearest / assignment / per-layer
            # work (the full aggregate is re-scored once from the tail ring at
            # finalize), which is the common loom monitoring path.
            _full_reading_consumer = bool(
                _wants_live_token_scores       # full ProbeReading in TokenEvent
                or _persists_layer_scores      # per-layer heatmap row
                or _persists_subspace_coords   # inspector whitened coords
            )
            lean_per_token = bool(
                need_per_token
                and not want_hidden
                and not _needs_monitor_gating
                and not _full_reading_consumer
                and _has_monitor_probes
            )
            self._live_lens_active_for_generation = _has_lens_consumer
            self._live_sae_active_for_generation = _has_sae_consumer
            authored_targets = self._pending_authored_prompt_targets(
                assistant_node_id, raw=raw,
            )
            authored_channels_active = bool(
                (_live_scores_on and _has_monitor_probes)
                or _has_lens_consumer
                or _has_sae_consumer
            )
            authored_search_end: int | None = None
            if authored_targets and authored_channels_active and not raw:
                content_only_ids = self._prepare_input(
                    input,
                    raw=False,
                    thinking=use_thinking,
                    stateless=stateless,
                    parent_node_id=chat_history_anchor,
                    user_role=(
                        sampling.user_role if sampling is not None else None
                    ),
                    assistant_role=(
                        sampling.assistant_role if sampling is not None else None
                    ),
                    to_device=False,
                    gen_seat=gen_seat,
                    add_generation_prompt=False,
                )
                authored_search_end = int(content_only_ids.shape[1])
            authored_recipe_steering = None
            if assistant_node_id is not None:
                authored_recipe_node = self.tree.nodes.get(assistant_node_id)
                if (
                    authored_recipe_node is not None
                    and authored_recipe_node.recipe is not None
                ):
                    authored_recipe_steering = (
                        authored_recipe_node.recipe.steering
                    )
            authored_capture = (
                _AuthoredPromptCapture(
                    targets=authored_targets,
                    monitor_active=bool(_live_scores_on and _has_monitor_probes),
                    lens_active=_has_lens_consumer,
                    sae_active=_has_sae_consumer,
                    lens_top_k=jlens_top_k,
                    persist_per_layer_scores=_persists_layer_scores,
                    steering=authored_recipe_steering,
                    search_end=authored_search_end,
                )
                if authored_targets and authored_channels_active
                else None
            )
            self.events.emit(GenerationStarted(input=input, stateless=stateless))
            try:
                # Capture attach + monitor live + ctx.reset live INSIDE the
                # inner try so a BaseException (KeyboardInterrupt, etc.)
                # between any pair of these still hits the cleanup finally.
                # ``_end_capture`` and ``end_live`` are idempotent.
                self._begin_capture(
                    widen=want_hidden, need_per_token=need_per_token,
                    gating_only_probes=gating_only_probes,
                    gating_probe_keys=gating_probe_keys,
                    lens_gating_probe_keys=lens_gating_probe_keys,
                    sae_gating_probe_keys=sae_gating_probe_keys,
                    lean_per_token=lean_per_token,
                    final_probe_aggregate=return_probe_readings,
                    live_lens_active=_has_lens_consumer,
                    live_sae_active=_has_sae_consumer,
                    capture_prompt=authored_capture is not None,
                )
                # Reset the steering manager's TriggerContext for this gen;
                # ``generate_steered`` mutates it at lifecycle boundaries.
                self._steering.ctx.reset()
                # Cold-start every manifold foot-follower so this gen re-seeds
                # at the origin instead of inheriting the prior run's foot.
                self._steering.reset_manifold_feet()
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
                    forced_prefix=(
                        continuation_prefix
                        if continuation_node_id is not None
                        else forced_prefix
                    ),
                    # Per-token perplexity costs one host sync/token.  Compute
                    # it only when something surfaces it: a loom-attached gen
                    # persists it on the token row, an interactive (non-
                    # stateless) gen may show it, or a caller opts its on_token
                    # in.  Stateless server streaming (OpenAI/Ollama deltas, the
                    # native WS tap) never reads per-token ppl, so it skips the
                    # sync.  Callers can force it via ``on_token
                    # requests perplexity through ``TokenConsumerOptions``.
                    want_perplexity=(
                        not stateless
                        or assistant_node_id is not None
                        or (
                            on_token is not None and _callback_options.perplexity
                        )
                    ),
                    cache_token_text=_tap_has_text_consumer,
                    authored_capture=authored_capture,
                )
            finally:
                self._gen_state.stop_requested.set()
                self._end_capture()
                if steering_cm is not None:
                    # Internal scope cleanup (see ``_exit_internal_steering``);
                    # ``swallow=False`` so a teardown failure surfaces here.
                    self._exit_internal_steering(steering_cm, swallow=False)
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
            # current empty/no-capture shape.
            _mean_logprob_out: float | None = None
            _mean_surprise_out: float | None = None
            if mean_logprob_count > 0:
                _mean_logprob_out = mean_logprob_sum / mean_logprob_count
                _mean_surprise_out = -_mean_logprob_out
            result = self._finalize_generation(
                generated_ids, elapsed, vector_snapshot,
                prompt_tokens=prompt_tokens, stateless=stateless,
                logprobs_list=logprobs_list,
                applied_steering=applied_steering,
                return_hidden=want_hidden,
                return_probe_readings=return_probe_readings,
                assistant_node_id=assistant_node_id,
                mean_logprob=_mean_logprob_out,
                mean_surprise=_mean_surprise_out,
            )
            self.events.emit(GenerationFinished(result=result))
            return result
        except BaseException:
            # If we bailed before the inner finally ran (e.g. preamble threw),
            # make sure the steering scope is popped.  Same internal-cleanup
            # bypass as the inner finally — phase may be PREAMBLE, RUNNING,
            # or FINALIZING depending on where we threw, and the pop is
            # always legitimate teardown here.
            if steering_cm is not None:
                # ``swallow=True``: we're already re-raising the original
                # failure below, so a teardown ``Exception`` must not mask
                # it (see ``_exit_internal_steering``).
                self._exit_internal_steering(steering_cm, swallow=True)
            raise
        finally:
            try:
                try:
                    # Defense-in-depth: even if the inner finally never ran
                    # (e.g. a BaseException between the outer try entry and
                    # ``begin_capture``), any hooks that did get attached
                    # must come off.  Idempotent.
                    self._end_capture()
                    # Probe-inspector subspace-coords post-pass is
                    # per-generation; clear it so it never leaks into a
                    # later gen that didn't opt in.
                    self._monitor.set_subspace_coords(False)
                    # Release the loom-tree reservation in the same scope as
                    # the gen-lock release.  Even if finalize raised,
                    # mutators (edit / delete on this subtree) need to be
                    # free again now that the streaming target is no longer
                    # live.
                    self._active_gen_reservation = None
                    self._last_token_probe_payload = None
                    self._last_token_probe_readings = None
                finally:
                    # Run closure and capture-state resets must survive a
                    # teardown failure above (a raising hook detach must not
                    # leave a bound run pinning a stale lens).
                    self._close_instrument_runs()
                    # Reset capture state to the default (FULL,
                    # non-persistent) so the next gen starts clean (finalize
                    # has already consumed the rows by now).
                    # Belt-and-suspenders: ``_begin_capture`` resets it at
                    # gen start too.
                    self._capture_state = CaptureState()
                    self._compiled_clean_eligible = False
                    self._incremental_readings = []
                    self._incremental_gate_scores = []
                    # Zero the persistent compile-clean steering offsets so a
                    # static-affine push can't leak into a later generation
                    # that takes the eager / unsteered path without
                    # re-running ``_install_composed_steering`` (unsteered
                    # gens have no steering scope to reset them).
                    self._steering_uses_compiled_offsets = False
                    if self._steering.has_compiled_offsets():
                        self._steering.zero_compiled_offsets()
            finally:
                # Unconditional: an earlier teardown exception must not
                # leave the session phase-wedged or the gen lock held.
                self._gen_phase = GenState.IDLE
                self._gen_lock.release()

    def _generate_runset(
        self,
        input: Any,
        *,
        steering: "str | Steering | None" = None,
        sampling: SamplingConfig | None = None,
        stateless: bool = False,
        raw: bool = False,
        thinking: bool | None = None,
        on_token: TokenCallback | None = None,
        parent_node_id: str | None = None,
        n: int = 1,
        recipe_override: "Recipe | str | None" = None,
        gen_seat: str = "assistant",
        append_same_role: bool = True,
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
                gen_seat=gen_seat,
                append_same_role=append_same_role,
            )
            node_id = self.tree.active_node_id if not stateless else None
            return RunSet([result], node_ids=[node_id], kind="generation")

        # The batched fast fan renders through its own prefix path — route
        # non-assistant seats through the serial runner instead.
        fast_fan = None if gen_seat != "assistant" else self._generate_fan_fast(
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
        if fast_fan is not None:
            return fast_fan

        # N-way regen: derive per-sibling seeds from the supplied base seed
        # (or a fresh entropy-derived one). Each iteration runs
        # ``_generate_core`` independently; ``add_user_turn`` dedups so all
        # siblings share the same user-parent.
        base_seed = sampling.seed if sampling is not None else None
        schedule = derive_seed_schedule(base_seed, n)
        jobs: list[_SerialGenerationJob] = []
        for seed_i in schedule:
            from dataclasses import replace as _replace
            si = sampling if sampling is not None else SamplingConfig()
            si = _replace(si, seed=seed_i)
            jobs.append(_SerialGenerationJob(
                input=input,
                steering=steering,
                sampling=si,
                raw=raw,
                thinking=thinking,
                on_token=on_token,
                parent_node_id=parent_node_id,
                recipe_override=recipe_override,
                gen_seat=gen_seat,
            ))
        return _run_serial_generation_jobs(
            self,
            jobs,
            stateless=stateless,
            kind="fan",
            stop_between_jobs=True,
        )

    # -- Generation: blocking --

    def _generate_fan_fast(
        self,
        input: Any,
        *,
        steering: "str | Steering | None",
        sampling: SamplingConfig | None,
        stateless: bool,
        raw: bool,
        thinking: bool | None,
        on_token: TokenCallback | None,
        parent_node_id: str | None,
        n: int,
        recipe_override: "Recipe | str | None",
    ) -> RunSet | None:
        """Batch deterministic fan-out without changing seed semantics."""
        del parent_node_id
        if (
            n <= 1
            or not stateless
            or on_token is not None
            or recipe_override is not None
        ):
            return None
        if self._compose_gen_config(sampling).temperature > 0:
            return None
        fast = self._generate_batch_fast(
            [input for _ in range(n)],
            steering=steering,
            sampling=sampling,
            thinking=thinking,
            stateless=stateless,
            raw=raw,
            on_result=None,
        )
        if fast is None:
            return None
        return RunSet(
            fast,
            node_ids=[None] * len(fast),
            grid=[{} for _ in fast],
            kind="fan",
        )

    def generate(
        self,
        input: Any,
        *,
        steering: "str | Steering | None" = None,
        sampling: SamplingConfig | None = None,
        stateless: bool = False,
        raw: bool = False,
        thinking: bool | None = None,
        on_token: TokenCallback | None = None,
        parent_node_id: str | None = None,
        n: int = 1,
        recipe_override: "Recipe | str | None" = None,
        gen_seat: str = "assistant",
        append_same_role: bool = True,
    ) -> RunSet:
        """Blocking generation.

        Args:
            input: prompt string or list of message dicts.
            steering: expression string (e.g. ``"0.5 honest + 0.3 warm"``)
                or a pre-built :class:`Steering`.  Pole aliases resolve at
                parse time via ``io.selectors.resolve_bare_atom``.  ``None`` =
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
            gen_seat: which seat the generated turn occupies (the cast
                model).  ``"assistant"`` (default) is the classic flow;
                ``"user"`` renders the generation prompt as a user-seat
                header (labeled by ``sampling.user_role``) and the new
                node lands with ``role="user"`` + a stamped recipe —
                generated is provenance, not a seat.  A non-assistant
                seat needs a validated scene grammar
                (:attr:`scene_grammar`). ``input=None`` skips an authored
                span. For a one-shot generation, ``append_same_role=True``
                reuses a matching leaf and treats its existing text as a
                forced prefix; fan-out retains sibling branches.
            append_same_role: coalesce a one-shot bare generation into a leaf
                with the same structural role and role label. Disable for
                specialist operations whose contract requires a new sibling.

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
            gen_seat=gen_seat,
            append_same_role=append_same_role,
        )

    # -- Generation: streaming --

    def generate_stream(
        self,
        input: Any,
        *,
        steering: "str | Steering | None" = None,
        sampling: SamplingConfig | None = None,
        stateless: bool = False,
        raw: bool = False,
        thinking: bool | None = None,
        parent_node_id: str | None = None,
        recipe_override: "Recipe | str | None" = None,
        live_scores: bool = True,
        live_readouts: bool = True,
        gen_seat: str = "assistant",
        append_same_role: bool = True,
    ) -> GenerationStream:
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

        ``live_readouts=False`` skips live J-LENS/SAE token readouts for
        stream formats that never serialize those richer cards.
        """
        q: queue.SimpleQueue[Any] = queue.SimpleQueue()
        done = object()
        result_holder: list[GenerationResult] = []
        exc_holder: list[BaseException] = []
        idx_counter = [0]

        def _push(
            text: str, is_thinking: bool, tid: int | None, lp: float | None,
            top_alts: list[TokenAlt] | None, perplexity: float | None = None,
        ) -> None:
            payload = self._last_token_probe_payload or {}
            probe_readings: dict[str, "ProbeReading"] | None = None
            # Monitor probes AND pinned lens/SAE probes (readout channels) all
            # land in the merged per-family readings — a lens-only roster still
            # carries per-token readings while the live lens is on.  This is the
            # ``TokenEvent.probe_readings`` compat channel (vendor extension /
            # traits SSE), kept as live :class:`ProbeReading` objects; the
            # serialized envelope rides ``measurements``.
            if live_scores and (
                self._monitor.probe_names
                or self._lens_probes
                or self._sae_probes
            ):
                raw_readings = self._last_token_probe_readings
                if isinstance(raw_readings, dict) and raw_readings:
                    probe_readings = raw_readings
            event = TokenEvent(
                text=text, token_id=tid if tid is not None else -1, index=idx_counter[0],
                thinking=is_thinking, logprob=lp, top_alts=top_alts,
                probe_readings=probe_readings, perplexity=perplexity,
                measurements=payload.get("measurements"),
            )
            idx_counter[0] += 1
            q.put(event)
        consumer = TokenConsumer(
            _push,
            TokenConsumerOptions(
                live_scores=bool(live_scores),
                lens_readout=bool(live_readouts),
                sae_readout=bool(live_readouts),
            ),
        )

        def _worker():
            try:
                result = self._generate_core(
                    input,
                    steering=steering,
                    sampling=sampling,
                    stateless=stateless,
                    raw=raw,
                    thinking=thinking,
                    on_token=consumer,
                    parent_node_id=parent_node_id,
                    recipe_override=recipe_override,
                    gen_seat=gen_seat,
                    append_same_role=append_same_role,
                )
                result_holder.append(result)
            except BaseException as e:
                exc_holder.append(e)
            finally:
                q.put(done)

        worker = threading.Thread(target=_worker, daemon=True)
        worker.start()

        def _events() -> Iterator[TokenEvent]:
            try:
                while True:
                    item = q.get()
                    if item is done:
                        break
                    yield item
            finally:
                self._gen_state.stop_requested.set()
                worker.join()
                if exc_holder and not result_holder:
                    raise exc_holder[0]

        class _GenerationStream:
            def __init__(self, iterator: Iterator[TokenEvent]) -> None:
                self._iterator = iterator

            def __iter__(self) -> "_GenerationStream":
                return self

            def __next__(self) -> TokenEvent:
                return next(self._iterator)

            def close(self) -> None:
                close = getattr(self._iterator, "close", None)
                if callable(close):
                    close()

            @property
            def result(self) -> GenerationResult | None:
                return result_holder[0] if result_holder else None

        return _GenerationStream(_events())

    # -- Generation: batch + sweep --

    def _generate_batch_fast(
        self,
        prompts: list[Any],
        *,
        steering: "str | Steering | None",
        sampling: SamplingConfig | None,
        thinking: bool | None,
        stateless: bool,
        raw: bool,
        on_result: Callable[[int, GenerationResult], None] | None,
    ) -> RunSet | None:
        """Run a compatible stateless batch through one HF ``generate`` call.

        ``None`` means "fall back to the serial Saklas decode loop."  The fast
        path is intentionally narrow: no per-token consumers/logprobs/stop
        strings, no stateful loom writes, and only always-active steering.  When
        probes are attached it uses a batched aggregate-only capture ring and
        scores one final reading per row after decode; per-token probe gates and
        streams stay on the serial Saklas loop.
        """
        if len(prompts) < 2 or not stateless:
            return None
        if self._batch_fast_sampling_blocked(sampling):
            return None
        if self._trait_queues:
            return None
        return_probe_readings = bool(
            sampling is None or sampling.return_probe_readings
        )
        has_sae_probes = bool(return_probe_readings and self._sae_probes)
        if has_sae_probes and self._sae_layer is None:
            return None

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
        if (
            use_thinking_req
            or gen_config.max_new_tokens < 1
            or lp_count is not None
            or (seed is not None and gen_config.temperature > 0)
            or stop_list is not None
            or logit_bias is not None
            or presence_penalty != 0.0
            or frequency_penalty != 0.0
            or logprobs_list is not None
            or not self._batch_fast_steering_is_always_on(steering_obj)
        ):
            return None

        model = self._batch_generate_model()
        if model is None:
            return None

        return self._run_generate_batch_fast(
            prompts,
            model=model,
            steering_obj=steering_obj,
            sampling=sampling,
            gen_config=gen_config,
            stateless=stateless,
            raw=raw,
            on_result=on_result,
        )

    def _finalize_batch_probe_result(
        self,
        *,
        generated_ids: list[int],
        elapsed: float,
        vector_snapshot: dict[str, float],
        prompt_tokens: int,
        finish_reason: str,
        applied_steering: str | None,
        probe_readings: dict[str, ProbeReading],
    ) -> GenerationResult:
        """Build a stateless batch result with pre-scored probe aggregates."""
        token_count = len(generated_ids)
        tok_per_sec = (
            token_count / elapsed if elapsed > MIN_ELAPSED_FOR_RATE else 0.0
        )
        decoded = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
        text = decoded if isinstance(decoded, str) else decoded[0]
        result = GenerationResult(
            text=text,
            tokens=list(generated_ids),
            token_count=token_count,
            tok_per_sec=tok_per_sec,
            elapsed=elapsed,
            steering_alphas=vector_snapshot,
            prompt_tokens=prompt_tokens,
            finish_reason=finish_reason,
            logprobs=None,
            applied_steering=applied_steering,
            hidden_states=None,
            probe_readings=probe_readings,
        )
        self._last_result = result
        self._last_per_token_scores = None
        if probe_readings:
            scalar_readings = {
                name: (reading.coords[0] if reading.coords else 0.0)
                for name, reading in probe_readings.items()
            }
            self.events.emit(ProbeScored(readings=scalar_readings))
        return result

    def _batch_probe_aggregate_for_row(
        self,
        row_index: int,
        generated_ids: list[int],
        probe_names: list[str],
        pooled: dict[int, torch.Tensor] | None = None,
    ) -> dict[str, ProbeReading]:
        """Score the final aggregate probe reading for one batched row."""
        empty = self._empty_readings(probe_names)
        if not generated_ids:
            return empty
        if pooled is None:
            pooled = self._batch_pooled_aggregate_for_row(row_index, generated_ids)
        if not pooled:
            return empty
        # Routed through the geometry run like the serial finalize
        # (``_score_aggregate_only``); the cold-foot reset stays here.
        self._monitor.enable_curved_warm(False)
        agg_vals = self._geometry_instrument.current_run.observe_aggregate(
            pooled,
        )
        return {name: agg_vals.get(name, empty[name]) for name in probe_names}

    def _batch_pooled_aggregate_for_row(
        self,
        row_index: int,
        generated_ids: list[int],
    ) -> dict[int, torch.Tensor]:
        """Shared final-token pooled slice for one batched generation row."""
        if not generated_ids:
            return {}
        from saklas.core.capture import last_content_index
        agg_fwd = last_content_index(generated_ids, self._tokenizer)
        return self._capture.batch_tail_slice_at(row_index, agg_fwd)

    def _batch_readout_probe_aggregate_for_row(
        self,
        pooled: dict[int, torch.Tensor],
    ) -> dict[str, ProbeReading]:
        """Pinned J-lens/SAE readout probe aggregates from a batched tail slice.

        The forwarders resolve through each family's bound run state
        (frozen spec snapshot + pinned lens), so every row of the batch
        measures against the same bind-time binding — the guards consult
        the binding too, so an ``on_result`` callback detaching a probe
        cannot change later rows' roster."""
        if not pooled:
            return {}
        out: dict[str, ProbeReading] = {}
        if self._lens_instrument._measurement_specs():
            out.update(
                self._lens_instrument.current_run.observe_aggregate(pooled)
            )
        if self._sae_instrument._measurement_specs():
            out.update(
                self._sae_instrument.current_run.observe_aggregate(pooled)
            )
        return out

    @staticmethod
    def _batch_fast_sampling_blocked(sampling: SamplingConfig | None) -> bool:
        """True iff per-row custom-loop sampling features are requested."""
        if sampling is None:
            return False
        return bool(
            sampling.stop is not None
            or sampling.logit_bias is not None
            or sampling.presence_penalty != 0.0
            or sampling.frequency_penalty != 0.0
            or sampling.logprobs is not None
            or sampling.return_hidden
            or sampling.return_top_k != 0
            or sampling.persist_per_layer_scores
            or sampling.persist_subspace_coords
        )

    @staticmethod
    def _batch_fast_steering_is_always_on(
        steering_obj: Steering | None,
    ) -> bool:
        """HF ``generate`` can batch only trigger-free/``Trigger.BOTH`` terms."""
        if steering_obj is None:
            return True
        for entry in steering_obj.alphas.values():
            if isinstance(entry, (ProjectedTerm, AblationTerm, ManifoldTerm)):
                trigger = entry.trigger
            elif isinstance(entry, tuple):
                trigger = entry[1]
            else:
                trigger = steering_obj.trigger
            if trigger is not Trigger.BOTH:
                return False
        return True

    def _batch_generate_model(self) -> Any | None:
        """Model object exposing ``generate``; unwrap compiled modules if needed."""
        model = self._model
        if callable(getattr(model, "generate", None)):
            return model
        orig = getattr(model, "_orig_mod", None)
        if orig is not None and callable(getattr(orig, "generate", None)):
            return orig
        return None

    def _batch_pad_token_id(self) -> int:
        """Pad id for left-padded generation batches."""
        for attr in ("pad_token_id", "eos_token_id"):
            value = getattr(self._tokenizer, attr, None)
            if value is None:
                continue
            if isinstance(value, (list, tuple)):
                if value:
                    return int(value[0])
            else:
                return int(value)
        return 0

    def _prepare_batch_input_ids(
        self,
        prompts: list[Any],
        *,
        sampling: SamplingConfig | None,
        raw: bool,
        pad_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        """Render and left-pad a stateless prompt batch."""
        rendered: list[torch.Tensor] = []
        user_role = sampling.user_role if sampling is not None else None
        assistant_role = sampling.assistant_role if sampling is not None else None
        for prompt in prompts:
            ids = self._prepare_input(
                prompt,
                raw=raw,
                thinking=False,
                stateless=True,
                user_role=user_role,
                assistant_role=assistant_role,
                to_device=False,
            )
            if ids.ndim != 2 or ids.shape[0] != 1:
                raise RuntimeError(
                    "generate_batch fast path expected rendered prompt shape [1, T]"
                )
            rendered.append(ids[0].detach().to(device="cpu", dtype=torch.long))

        lengths = [int(ids.shape[0]) for ids in rendered]
        max_len = max(lengths)
        batch = torch.full(
            (len(rendered), max_len), pad_id, dtype=torch.long,
        )
        attention = torch.zeros_like(batch)
        for row, ids in enumerate(rendered):
            length = lengths[row]
            batch[row, max_len - length:] = ids
            attention[row, max_len - length:] = 1
        return (
            batch.to(self._device),
            attention.to(self._device),
            lengths,
        )

    @staticmethod
    def _batch_generated_tokens(
        row: torch.Tensor,
        *,
        eos_ids: set[int],
        pad_id: int,
        max_new_tokens: int,
    ) -> tuple[list[int], str]:
        """Trim HF batch output into Saklas' generated-token convention."""
        raw_ids = [int(tok) for tok in row.detach().to("cpu").tolist()]
        tokens: list[int] = []
        finish_reason = "length"
        for token_id in raw_ids:
            if token_id in eos_ids:
                finish_reason = "stop"
                break
            tokens.append(token_id)
        else:
            stripped_pad = False
            while tokens and tokens[-1] == pad_id:
                tokens.pop()
                stripped_pad = True
            if stripped_pad or len(tokens) < max_new_tokens:
                finish_reason = "stop"
        return tokens, finish_reason

    def _run_generate_batch_fast(
        self,
        prompts: list[Any],
        *,
        model: Any,
        steering_obj: Steering | None,
        sampling: SamplingConfig | None,
        gen_config: GenerationConfig,
        stateless: bool,
        raw: bool,
        on_result: Callable[[int, GenerationResult], None] | None,
    ) -> RunSet:
        """Implementation of the compatible batched ``model.generate`` path."""
        if not self._gen_lock.acquire(blocking=False):
            raise ConcurrentGenerationError("Generation already in progress")
        if self._gen_phase is not GenState.IDLE:
            self._gen_lock.release()
            raise ConcurrentGenerationError("session generation already in flight")

        self._gen_phase = GenState.PREAMBLE
        steering_cm = None
        failed = False
        try:
            if steering_obj is not None and steering_obj.alphas:
                steering_cm = self.steering(steering_obj)
                steering_cm.__enter__()
            vector_snapshot: dict[str, float] = (
                self._snapshot_steering_alphas()
                if self._steering_composer._stack or steering_cm is not None
                else {}
            )
            pad_id = self._batch_pad_token_id()
            input_ids, attention_mask, prompt_lengths = self._prepare_batch_input_ids(
                prompts,
                sampling=sampling,
                raw=raw,
                pad_id=pad_id,
            )
            return_probe_readings = bool(
                sampling is None or sampling.return_probe_readings
            )
            probe_names = (
                list(self._monitor.probe_names)
                if return_probe_readings else []
            )
            has_lens_probes = bool(return_probe_readings and self._lens_probes)
            has_sae_probes = bool(return_probe_readings and self._sae_probes)
            # Batched generation bypasses ``_begin_capture`` but still owns
            # one generation transaction, and runs the same uniform
            # sequence: close → prepare → plan → bind.  The lens prepare
            # takes the disk refresh + pin and snapshots specs against
            # that identity BEFORE the plans/bindings consume them (its
            # pin-demand formula reduces to "probes attached and a final
            # aggregate wanted" on a batch request); one pinned lens then
            # serves every row aggregate rather than reopening all shards
            # per batch item, and the bindings freeze probe specs against
            # concurrent mutation for the whole batch.
            self._close_instrument_runs()
            batch_request = ReadRequest(
                final_aggregate=return_probe_readings,
                batch=True,
            )
            geometry_prep = self._geometry_instrument.prepare(batch_request)
            lens_prep = self._lens_instrument.prepare(batch_request)
            sae_prep = self._sae_instrument.prepare(batch_request)
            geometry_plan = self._geometry_instrument.plan(geometry_prep)
            lens_plan = self._lens_instrument.plan(lens_prep)
            sae_plan = self._sae_instrument.plan(sae_prep)
            self._geometry_instrument.bind(geometry_plan, geometry_prep)
            self._lens_instrument.bind(lens_plan, lens_prep)
            self._sae_instrument.bind(sae_plan, sae_prep)
            capture_probe_aggregates = bool(
                probe_names or has_lens_probes or has_sae_probes
            )
            if capture_probe_aggregates:
                layer_set: set[int] = set()
                for batch_plan in (geometry_plan, lens_plan, sae_plan):
                    layer_set |= batch_plan.latest_layers
                    layer_set |= batch_plan.tail_layers
                layer_idxs = sorted(layer_set)
                self._capture.clear()
                if layer_idxs:
                    self._capture.attach_batch_tail(
                        self._layers,
                        layer_idxs,
                        depth=_AGG_TAIL_DEPTH,
                    )
                self._capture_state = CaptureState(mode=CaptureMode.AGGREGATE_ONLY)
                if probe_names:
                    self._monitor.enable_curved_warm(False)
            self._gen_state.reset()
            self._steering.ctx.reset()
            self._steering.reset_manifold_feet()
            self._gen_phase = GenState.RUNNING
            for prompt in prompts:
                self.events.emit(GenerationStarted(input=prompt, stateless=stateless))

            generate_kwargs: dict[str, Any] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "max_new_tokens": int(gen_config.max_new_tokens),
                "pad_token_id": pad_id,
                "stopping_criteria": StoppingCriteriaList([
                    _SessionStopCriteria(self._gen_state),
                ]),
            }
            eos_token_id = getattr(self._tokenizer, "eos_token_id", None)
            if eos_token_id is not None:
                generate_kwargs["eos_token_id"] = eos_token_id
            do_sample = gen_config.temperature > 0
            generate_kwargs["do_sample"] = do_sample
            if do_sample:
                generate_kwargs["temperature"] = float(gen_config.temperature)
                generate_kwargs["top_p"] = float(gen_config.top_p)
                if gen_config.top_k is not None:
                    generate_kwargs["top_k"] = int(gen_config.top_k)

            start = time.monotonic()
            with torch.inference_mode():
                generated = model.generate(**generate_kwargs)
            elapsed = time.monotonic() - start
            if capture_probe_aggregates:
                self._end_capture()
            sequences = getattr(generated, "sequences", generated)
            if not isinstance(sequences, torch.Tensor) or sequences.ndim != 2:
                raise RuntimeError(
                    "generate_batch fast path expected model.generate to return "
                    "a [batch, sequence] tensor"
                )

            eos_ids = _get_eos_ids(model, self._tokenizer)
            max_prompt_len = int(input_ids.shape[1])
            self._gen_phase = GenState.FINALIZING
            results: list[GenerationResult] = []
            node_ids: list[str | None] = []
            grid_rows: list[dict[str, Any]] = []
            applied_steering = (
                str(steering_obj) if steering_obj is not None else None
            )
            for idx, prompt in enumerate(prompts):
                generated_ids, finish_reason = self._batch_generated_tokens(
                    sequences[idx, max_prompt_len:],
                    eos_ids=eos_ids,
                    pad_id=pad_id,
                    max_new_tokens=int(gen_config.max_new_tokens),
                )
                self._gen_state.thinking_end_idx = 0
                self._gen_state.emit_map = []
                self._gen_state.response_text = None
                self._gen_state.response_aggregate_index = None
                self._gen_state.finish_reason = finish_reason
                if capture_probe_aggregates:
                    pooled = self._batch_pooled_aggregate_for_row(idx, generated_ids)
                    probe_readings: dict[str, ProbeReading] = (
                        self._batch_probe_aggregate_for_row(
                            idx,
                            generated_ids,
                            probe_names,
                            pooled=pooled,
                        )
                        if probe_names
                        else {}
                    )
                    probe_readings.update(
                        self._batch_readout_probe_aggregate_for_row(pooled)
                    )
                    result = self._finalize_batch_probe_result(
                        generated_ids=generated_ids,
                        elapsed=elapsed,
                        vector_snapshot=vector_snapshot,
                        prompt_tokens=prompt_lengths[idx],
                        finish_reason=finish_reason,
                        applied_steering=applied_steering,
                        probe_readings=probe_readings,
                    )
                else:
                    result = self._finalize_generation(
                        generated_ids,
                        elapsed,
                        vector_snapshot,
                        prompt_tokens=prompt_lengths[idx],
                        stateless=stateless,
                        logprobs_list=None,
                        applied_steering=applied_steering,
                        return_hidden=False,
                        return_probe_readings=return_probe_readings,
                        assistant_node_id=None,
                    )
                results.append(result)
                node_ids.append(None)
                row = {"prompt_index": idx}
                grid_rows.append(row)
                self.events.emit(GenerationFinished(result=result))
                if on_result is not None:
                    on_result(idx, result)
            self._gen_state.stop_requested.set()
            total_tokens = sum(r.token_count for r in results)
            batch_tok_per_sec = (
                total_tokens / elapsed if elapsed > MIN_ELAPSED_FOR_RATE else 0.0
            )
            return RunSet(
                results,
                node_ids=node_ids,
                grid=grid_rows,
                kind="batch",
                metrics={
                    "batch_elapsed": elapsed,
                    "batch_token_count": total_tokens,
                    "batch_tok_per_sec": batch_tok_per_sec,
                },
            )
        except BaseException:
            failed = True
            raise
        finally:
            try:
                try:
                    if steering_cm is not None:
                        self._exit_internal_steering(steering_cm, swallow=failed)
                finally:
                    try:
                        self._end_capture()
                        self._active_gen_reservation = None
                        self._last_token_probe_payload = None
                        self._last_token_probe_readings = None
                    finally:
                        # Run closure and state resets must survive a
                        # teardown failure above (a raising hook detach must
                        # not leave a bound run pinning a stale lens).
                        self._close_instrument_runs()
                        self._capture_state = CaptureState()
                        self._compiled_clean_eligible = False
                        self._incremental_readings = []
                        self._incremental_gate_scores = []
                        self._steering_uses_compiled_offsets = False
                        if self._steering.has_compiled_offsets():
                            self._steering.zero_compiled_offsets()
            finally:
                # Unconditional: an earlier teardown exception must not
                # leave the session phase-wedged or the gen lock held.
                self._gen_phase = GenState.IDLE
                self._gen_lock.release()

    def generate_batch(
        self,
        prompts: list[Any],
        *,
        steering: "str | Steering | None" = None,
        sampling: SamplingConfig | None = None,
        thinking: bool | None = None,
        stateless: bool = True,
        raw: bool = False,
        on_result: Callable[[int, GenerationResult], None] | None = None,
    ) -> RunSet:
        """Run N prompts under the same steering, return a ``RunSet``.

        Compatible stateless batches run through one ``transformers.generate``
        call, including aggregate-only probe reads via batched tail capture.
        Anything that needs Saklas' row-shaped custom loop (stateful history,
        live lens, hidden capture returns, logprobs, stop strings, seeded
        per-row sampling, penalties, or phased/gated triggers) falls back to the
        serial single-prompt path.  The session's threading lock keeps concurrent
        ``generate_batch`` calls from interleaving — they queue FIFO at the
        per-call level, same as today's ``generate``.

        Args mirror ``generate``.  ``stateless`` defaults to ``True``
        (batch generation is overwhelmingly used for sweeps and evals
        where conversational history would corrupt the comparison);
        pass ``stateless=False`` if you genuinely want each prompt to
        accumulate against the running history.

        **Prefix KV caching.**  When the rows share a token prefix (a common
        system prompt, few-shot preamble, or shared instruction) and the
        steering doesn't touch the prefill region (``None`` or ``@response`` /
        ``@generated`` / probe-gated), that shared prefix is prefilled once and
        its KV reused across every row — the per-call saving scales with
        ``shared_prefix_len / total_input_len``.  Skipped for ``stateless=False``
        (per-row history breaks the shared prefix), ``return_hidden`` (needs
        every prefix hidden state), thinking, or prefill-active steering (the
        prompt KV would differ per row).  Correctness is unaffected either way —
        a cache hit produces the identical token stream.

        ``on_result(idx, result)`` fires after each completion for local
        progress hooks.

        Returns:
            ``RunSet`` aligned with ``prompts``.  ``runset.grid`` records
            ``{"prompt_index": i}`` for each row.
        """
        prompts_in: Any = prompts
        if not isinstance(prompts_in, list) or not prompts_in:
            raise ValueError("generate_batch: prompts must be a non-empty list")

        fast = self._generate_batch_fast(
            prompts,
            steering=steering,
            sampling=sampling,
            thinking=thinking,
            stateless=stateless,
            raw=raw,
            on_result=on_result,
        )
        if fast is not None:
            return fast

        prefix_cached = self._maybe_cache_batch_prefix(
            prompts, steering=steering, sampling=sampling,
            thinking=thinking, stateless=stateless, raw=raw,
        )
        try:
            jobs = [
                _SerialGenerationJob(
                    input=prompt,
                    steering=steering,
                    sampling=sampling,
                    raw=raw,
                    thinking=thinking,
                    grid={"prompt_index": idx},
                )
                for idx, prompt in enumerate(prompts)
            ]
            runset = _run_serial_generation_jobs(
                self,
                jobs,
                stateless=stateless,
                kind="batch",
                on_result=(
                    (lambda idx, result, _row: on_result(idx, result))
                    if on_result is not None
                    else None
                ),
            )
        finally:
            if prefix_cached:
                self.cache_prefix(None)
        return runset

    def _maybe_cache_batch_prefix(
        self,
        prompts: list[Any],
        *,
        steering: "str | Steering | None",
        sampling: SamplingConfig | None,
        thinking: bool | None,
        stateless: bool,
        raw: bool,
    ) -> bool:
        """Prefill + cache the batch's shared token prefix when reuse is valid.

        Returns ``True`` iff a prefix was cached (the caller clears it after the
        batch).  Eligibility (all must hold): ``stateless`` (per-row history
        would break the shared prefix), the steering is prefill-inactive
        (``_steering_value_prefill_inactive`` — else the prompt KV differs per
        row and the consume gate would refuse the hit anyway), no
        ``return_hidden`` and no thinking (both force a full re-prefill), at
        least two rows, and a shared prefix of at least
        :data:`_PREFIX_CACHE_MIN_TOKENS`. Ordinary ineligibility returns False;
        lifecycle, rendering, and initialization failures propagate.
        """
        if len(prompts) < 2 or not stateless:
            return False
        if thinking:
            return False
        if sampling is not None and getattr(sampling, "return_hidden", False):
            return False
        if not self._steering_value_prefill_inactive(steering):
            return False
        common = self._batch_common_prefix_ids(prompts, raw=raw)
        if common is None:
            return False
        max_new_tokens = (
            sampling.max_tokens
            if sampling is not None and sampling.max_tokens is not None
            else self.config.max_new_tokens
        )
        return (
            self.cache_prefix(
                common,
                max_new_tokens=max_new_tokens,
                prefer_static=bool(
                    self._static_cache_active and steering is None
                ),
            )
            >= _PREFIX_CACHE_MIN_TOKENS
        )

    def _batch_common_prefix_ids(
        self, prompts: list[Any], *, raw: bool,
    ) -> "torch.Tensor | None":
        """Longest shared leading token run across the batch's rendered inputs.

        Renders each prompt to ``input_ids`` in stateless mode (the same render
        ``_generate_core`` will reproduce, so the cached tokens are a true
        byte-prefix of every row) and returns the shared head as a 1-D tensor,
        or ``None`` when the shared run is shorter than
        :data:`_PREFIX_CACHE_MIN_TOKENS`.  The run is capped at ``min_row_len -
        1`` so every row keeps a non-empty suffix to sample from.
        """
        rendered: list[torch.Tensor] = []
        for prompt in prompts:
            ids = self._prepare_input(
                prompt, raw=raw, stateless=True, to_device=False,
            )
            # Prefix discovery is scalar-heavy Python work. Keep the token
            # comparison on CPU so MPS/CUDA do not pay one device sync per
            # ``int(t[i])`` while walking the common prefix.
            rendered.append(ids[0].detach().to("cpu"))
        min_len = min(int(t.shape[0]) for t in rendered)
        if min_len <= _PREFIX_CACHE_MIN_TOKENS:
            return None
        ref = rendered[0]
        common = 0
        for i in range(min_len):
            tok = int(ref[i])
            if all(int(t[i]) == tok for t in rendered):
                common += 1
            else:
                break
        common = min(common, min_len - 1)  # leave a >=1-token suffix per row
        if common < _PREFIX_CACHE_MIN_TOKENS:
            return None
        return ref[:common].clone()

    def _generate_sweep_fast(
        self,
        jobs: list[_SerialGenerationJob],
        *,
        stateless: bool,
        on_result: Callable[[int, GenerationResult, dict[str, Any]], None] | None,
    ) -> RunSet | None:
        """Fast path for exact degenerate sweeps.

        Different alpha rows intentionally install different hook coefficients,
        so they must stay on the serial path until the hook layer can carry
        per-row steering.  When every row resolves to the same steering
        expression, however, a stateless greedy sweep is equivalent to a batch
        of repeated prompts under one steering scope; route that through the
        shared ``generate_batch`` fast path.
        """
        if len(jobs) < 2 or not stateless:
            return None
        first = jobs[0]
        gen_config = self._compose_gen_config(first.sampling)
        if gen_config.temperature > 0:
            return None
        for job in jobs[1:]:
            if (
                job.input != first.input
                or job.steering != first.steering
                or job.raw != first.raw
                or job.thinking != first.thinking
                or job.recipe_override is not None
                or self._compose_gen_config(job.sampling).temperature > 0
            ):
                return None

        def _batch_on_result(
            idx: int,
            result: GenerationResult,
        ) -> None:
            if on_result is not None:
                on_result(idx, result, jobs[idx].grid or {})

        fast = self._generate_batch_fast(
            [job.input for job in jobs],
            steering=first.steering,
            sampling=first.sampling,
            thinking=first.thinking,
            stateless=True,
            raw=first.raw,
            on_result=_batch_on_result if on_result is not None else None,
        )
        if fast is None:
            return None
        return RunSet(
            list(fast),
            node_ids=[None] * len(fast),
            grid=[job.grid or {} for job in jobs],
            kind="fan",
        )

    def generate_sweep(
        self,
        prompt: Any,
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
        sweep_in: Any = sweep
        if not isinstance(sweep_in, dict) or not sweep_in:
            raise ValueError("generate_sweep: sweep dict must be non-empty")
        for name, alphas in sweep.items():
            alphas_in: Any = alphas
            if not isinstance(alphas_in, (list, tuple)) or not alphas_in:
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

        jobs: list[_SerialGenerationJob] = []
        for idx, combo in enumerate(itertools.product(*alpha_lists)):
            alpha_values = dict(zip(concept_names, combo, strict=True))
            alpha_values = {k: float(v) for k, v in alpha_values.items()}
            terms = [f"{alpha} {name}" for name, alpha in alpha_values.items()]
            expr = " + ".join(terms)
            if base_str:
                expr = f"{base_str} + {expr}"

            from dataclasses import replace as _replace
            si = sampling if sampling is not None else SamplingConfig()
            si = _replace(si, seed=seed_schedule[idx])

            jobs.append(_SerialGenerationJob(
                input=prompt,
                steering=expr,
                sampling=si,
                raw=raw,
                thinking=thinking,
                parent_node_id=gen_parent_id,
                grid=alpha_values,
            ))

        fast_sweep = self._generate_sweep_fast(
            jobs,
            stateless=stateless,
            on_result=(
                (lambda idx, result, row: on_result(
                    idx, result, cast(dict[str, float], row),
                ))
                if on_result is not None
                else None
            ),
        )
        if fast_sweep is not None:
            return fast_sweep

        return _run_serial_generation_jobs(
            self,
            jobs,
            stateless=stateless,
            kind="fan",
            on_result=(
                (lambda idx, result, row: on_result(
                    idx, result, cast(dict[str, float], row),
                ))
                if on_result is not None
                else None
            ),
        )

    # -- Generation control --

    def stop(self) -> None:
        self._gen_state.request_stop()

    # -- Lifecycle --

    def close(self) -> None:
        self._steering.clear_all()
        self._profiles.clear()
        self._manifolds.clear()

    def __enter__(self) -> "SaklasSession":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
    def __iter__(self) -> "GenerationStream": ...

    def __next__(self) -> TokenEvent: ...
