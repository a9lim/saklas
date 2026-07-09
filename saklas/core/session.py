"""SaklasSession — unified backend for saklas's programmatic API and TUI."""
from __future__ import annotations
import asyncio
import logging
import queue
import re
import threading
import time
from contextlib import contextmanager, nullcontext, suppress
from dataclasses import dataclass
from enum import Enum, IntEnum
from types import TracebackType
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Callable, Iterator, Literal, cast, overload

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
    ProbeScored,
    VectorExtracted,
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
from saklas.core.loom import (
    InvalidNodeOperationError,
    LoomMutated,
    LoomTree,
    Recipe,
    MutationDuringGenerationError,
    derive_seed_schedule,
)
from saklas.core.model import load_model, get_layers, get_model_info
from saklas.core.monitor import Monitor
from saklas.io.probes_bootstrap import bootstrap_layer_means
from saklas.core.profile import Profile, load_profile as _load_profile
from saklas.core.results import (
    GenerationResult,
    ProbeReading,
    ProbeReadings,
    RunSet,
    TokenEvent,
)
from saklas.core.sampling import SamplingConfig
from saklas.core.steering import Steering
from saklas.core.steering_expr import AblationTerm, ManifoldTerm, ProjectedTerm
from saklas.core.manifold import Manifold, manifold_is_affine

if TYPE_CHECKING:
    from saklas.core.scoring import ChoiceScores
    from saklas.core.steering_composer import SteeringComposer
    from saklas.io.templates import TemplateFolder
from saklas.core.triggers import Trigger

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
_AGG_TAIL_DEPTH = 8

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
class _SerialGenerationJob:
    input: Any
    steering: Any = None
    sampling: SamplingConfig | None = None
    raw: bool = False
    thinking: bool | None = None
    on_token: Callable[..., None] | None = None
    parent_node_id: str | None = None
    recipe_override: Any = None
    grid: dict[str, Any] | None = None


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


class VectorNotRegisteredError(KeyError, SaklasError):
    """Raised when a steering call references a vector not in the registry."""

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
# through the stack, ``flatten_steering_stack``, ``push``/``pop``, and is
# dispatched by type in ``SteeringComposer.compose_steering_entries``.
SteeringStackEntry = tuple[float, Trigger] | AblationTerm | ManifoldTerm


# Back-compat alias: the function was moved to core/manifold.py as the public
# ``manifold_is_affine``.  Tests that monkeypatch ``session._manifold_is_affine``
# and internal callers that reference it by its old name continue to resolve here.
_manifold_is_affine = manifold_is_affine


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

    Pushes an entries dict onto ``session._steering_stack`` on ``__enter__``
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
        # raises (e.g. VectorNotRegisteredError).  __enter__ only flips
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
        # populated lazily by ``_ensure_manifold_loaded`` on scope entry.
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
        # The LIFO steering stack + every push/pop-frequency steering method now
        # live on the ``SteeringComposer`` collaborator (instantiated near the end
        # of ``__init__``, once ``_monitor`` exists).  ``self._steering_stack`` is a
        # settable property over ``_steering_composer._stack`` — see the property
        # below; the composer is lazily created on first access so ``__new__`` test
        # stubs that bypass ``__init__`` still resolve it.

        # Set by ``SteeringComposer.install_composed_steering`` when the current
        # steering lowers to the persistent compile-clean offset buffers
        # (static-affine push, compiled MPS session) instead of transient hooks —
        # routes the decode loop to the compiled module + StaticCache.
        self._steering_uses_compiled_offsets: bool = False

        # Persistent compile-clean *capture* buffers + hook handles, adopted from
        # ``from_pretrained`` when compile stuck (slice 2).  The always-on hooks
        # ``copy_`` each layer's last-token slice into ``_capture_buffers[L]``
        # every forward (fused into the compiled graph); a probed gen on the
        # compiled path reads them post-forward via ``HiddenCapture.ingest_persistent``
        # instead of registering transient capture hooks that would graph-break.
        # Empty unless a compiled MPS session adopted them.
        self._capture_buffers: dict[int, torch.Tensor] = {}
        self._capture_handles: list[Any] = []
        # Per-gen flags: ``_compiled_clean_eligible`` (set early in
        # ``_generate_core``) means this gen *can* take the compiled clean path —
        # compiled MPS, static cache, capture buffers present, not return_hidden —
        # provided steering also lowers to offsets.  ``_capture_state.persistent``
        # (set in ``_begin_capture``) means the active capture rides the persistent
        # buffers, so the decode loop wires ``ingest_persistent`` as its step
        # callback and the routing keeps the compiled module.
        self._compiled_clean_eligible: bool = False

        # Active assistant-role label for the current ``session.steering()``
        # scope — populated when every role-tagged term in the resolved
        # expression agrees on a role.  ``None`` means "use the family's
        # standard assistant label", the legacy zero-overhead path.
        # Push/save/restore is handled by ``_SteeringContext`` so nested
        # scopes inner-wins for the duration of the inner block.  The
        # generation surface reads this when assembling the chat-template
        # input so the assistant turn opens with ``<role>`` instead of
        # ``assistant``.
        self._active_role: str | None = None

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
        self._last_token_probe_payload: dict[str, Any] | None = None

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
        from saklas.io import selectors as _selectors
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
        self._jlens_device_cache: dict[tuple[int, str, tuple[int, ...]], torch.Tensor] = {}
        self._jlens_decode_cache: dict[int, str] = {}
        # Live workspace readout (enable_live_lens): device-resident J_l
        # subset + settings, or None when off.
        self._live_lens: dict[str, Any] | None = None
        # Pinned J-lens token probes: name -> {word, token_id, layers}.  NOT
        # monitor probes — they read the lens readout channel (per-layer
        # softmax salience/probability), not a whitened subspace coordinate,
        # and are scored on the post-forward lens path (`_score_lens_probes`).
        self._lens_probes: dict[str, dict[str, Any]] = {}
        # Per-forward stash: the gate callback computes band logits first
        # (score_callback runs before the token tap), the display step reuses
        # them when the layer sets match.  Reset per generation.
        self._lens_step_stash: dict[str, Any] | None = None
        self._last_lens_step_readings: dict[str, "ProbeReading"] | None = None
        # CAA live toggle: when False, per-token monitor scoring is disabled
        # for UI/trait/loom consumers (aggregate-only capture); probe gates
        # still force the per-token subset they need.
        self._live_probe_scores: bool = True
        if probe_categories:
            self._layer_means = bootstrap_layer_means(
                self._model, self._tokenizer, self._layers, self._model_info,
            )
            self._whitener = self._build_whitener_from_cache_or_compute()

        # DLS toggle stored on the session so ad-hoc ``session.extract``
        # calls (via ``ExtractionPipeline``) inherit it without re-passing.
        self._dls: bool = bool(dls)

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

        # Steering-resolution + stack collaborator (extracted from the session).
        # Instantiated last among the steering state because its push/pop methods
        # read ``_monitor.probe_names``; lazily importable so the module-load graph
        # stays acyclic (composer imports session-level symbols at import time).
        from saklas.core.steering_composer import SteeringComposer
        self._steering_composer: SteeringComposer = SteeringComposer(self)

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

    def has_vector(self, name: str) -> bool:
        return name in self._profiles

    @property
    def vectors(self) -> dict[str, Profile]:
        """Registered steering vector profiles: name -> :class:`Profile`.

        Returns a snapshot dict — each entry is a fresh :class:`Profile`
        wrapping the backing tensors.  The raw tensor dict lives on
        :attr:`profiles` (internal / mutable accessor); use this property
        for read-only public access and prefer ``profiles`` when you need
        in-place mutation of the registry.

        Note: the name ``vectors`` pre-dates the manifold unification;
        ``profiles`` is the internal canonical name for the same registry.
        """
        return {name: Profile(tensors) for name, tensors in self._profiles.items()}

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
        from saklas.core.vectors import is_foldable_vector_manifold

        names = set(self._profiles)  # registered vectors are always R=1
        try:
            for pname in self._monitor.probe_names:
                if pname in names:
                    continue  # a registered vector wins the same-named probe
                manifold = self._monitor.manifolds.get(pname)
                if manifold is not None and is_foldable_vector_manifold(manifold):
                    names.add(pname)
        except Exception:
            pass
        return sorted(names)

    def _live_direction_tensors(self, name: str) -> "dict[int, torch.Tensor] | None":
        """Live per-layer direction tensors for *name* (may be device-
        resident — callers must hold ``_gen_lock``).  A registered steering
        vector wins over a same-named probe; otherwise the folded view of an
        attached probe's manifold.

        Returns ``None`` for a multi-node / curved probe — it has no single
        direction to fold, so the direction analytics skip it rather than
        crashing on ``folded_vector_directions``."""
        prof = self._profiles.get(name)
        if prof is not None:
            return dict(prof)
        manifold = self._monitor.manifolds.get(name)
        if manifold is not None:
            from saklas.core.vectors import (
                folded_vector_directions,
                is_foldable_vector_manifold,
            )
            if is_foldable_vector_manifold(manifold):
                return folded_vector_directions(manifold)
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
        out: dict[str, dict[str, Any]] = {
            name: {"manifold": m}
            for name, m in self._monitor.manifolds.items()
        }
        # Pinned J-lens token probes (readout channel, not monitor probes).
        out.update(
            {name: {"lens": dict(spec)}
             for name, spec in self._lens_probes.items()}
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
        tensors.  The raw backing of :attr:`vectors` (which wraps each entry
        in a :class:`Profile`); the live dict, mutating it mutates the
        registry."""
        return self._profiles

    def _get_steering_composer(self) -> "SteeringComposer":
        """The steering collaborator, created lazily if absent.

        Real sessions set ``_steering_composer`` in ``__init__``; ``__new__`` test
        stubs that bypass it (and custom-``__init__`` stubs) get one built on first
        touch so the ``_steering_stack`` property and the steering forwarders below
        resolve without the caller wiring the collaborator by hand.
        """
        composer = self.__dict__.get("_steering_composer")
        if composer is None:
            from saklas.core.steering_composer import SteeringComposer
            composer = SteeringComposer(self)
            self._steering_composer = composer
        return composer

    @property
    def _steering_stack(self) -> list[dict[str, SteeringStackEntry]]:
        """The LIFO steering stack — owned by the :class:`SteeringComposer`.

        A read/write view over ``_steering_composer._stack``.  Settable so the
        existing test stubs (and any caller) can assign a fresh stack; the setter
        routes the assignment into the composer.
        """
        return self._get_steering_composer()._stack

    @_steering_stack.setter
    def _steering_stack(
        self, value: list[dict[str, SteeringStackEntry]],
    ) -> None:
        self._get_steering_composer()._stack = value

    @property
    def model_metadata(self) -> Any:
        """The structured model-info object (id, arch, layer count, …).

        Distinct from :attr:`model_info`, which returns a plain ``dict`` copy;
        this is the live object the loader produced, consumed by panels that
        want the typed fields.
        """
        return self._model_info

    @property
    def generation_state(self) -> GenerationState:
        """The current per-call :class:`GenerationState` (finish reason, emit
        map, thinking spans, stop event).

        Distinct from :attr:`gen_state`, which reports the coarse lifecycle
        *phase* (``IDLE``/``RUNNING``/…); this is the live mutable streaming
        state the server/TUI read while a generation is in flight.
        """
        return self._gen_state

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

        Public wrapper over the lazy loader — ``key`` is the grammar's
        ``[ns/]name[:variant]`` registry key.  See the private for the full
        contract (raises :class:`ManifoldNotRegisteredError` on a miss)."""
        self._ensure_manifold_loaded(key)

    def ensure_profile_registered(
        self, name: str, *, role: str = "vector",
    ) -> dict[int, torch.Tensor]:
        """Resolve and register a steering direction for ``name``, returning
        its per-layer tensors.

        Public wrapper over the manifold-first resolution chain (in-memory
        bake → fitted 2-node ``pca`` manifold → ported legacy vector)."""
        return self._ensure_profile_registered(name, role=role)

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
        :func:`compute_dls_axes`).  The property closes that footgun.
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
        allocation, by ``manifold compare``, and by
        callers that pass a whitener to ``project_profile`` for
        LEACE-style projection.  Returns a
        :class:`saklas.core.mahalanobis.LayerWhitener` or ``None`` if
        construction failed (model is mid-load, neutral activations
        couldn't be computed, etc. — we never raise here, so probe
        scoring stays alive).
        """
        if self._whitener is None:
            self._whitener = self._build_whitener_from_cache_or_compute()
            # Keep the trait monitor's read metric in lock-step with the
            # session whitener: when it's built lazily (e.g. a ``probes=[]``
            # session that later extracts), push it into the monitor so
            # probe reads switch to the Mahalanobis cosine.  ``set_whitener``
            # is a no-op when the identity is unchanged.  Guarded with
            # ``getattr`` because the property can be touched mid-init
            # before ``_monitor`` is assigned.
            monitor = getattr(self, "_monitor", None)
            if monitor is not None and self._whitener is not None:
                monitor.set_whitener(self._whitener)
        return self._whitener

    def _build_whitener_from_cache_or_compute(self) -> "Any":
        """Compute or load the per-model whitener.

        Uses ``load_or_compute_neutral_activations`` (alignment.py) for
        disk caching; combines with the in-memory ``_layer_means`` to
        instantiate the :class:`LayerWhitener`.  Soft-fails to ``None``
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

    # -- Jacobian lens (verbalizable-workspace readout) --

    @property
    def jlens(self) -> "Any":
        """The model's fitted Jacobian lens, or ``None`` when not fitted.

        Loaded lazily from the per-model artifact
        (``models/<safe_id>/jlens.safetensors``); fit one with
        :meth:`fit_jlens` or ``saklas lens fit``. Returns a
        :class:`saklas.core.jlens.JacobianLens`.
        """
        if self._jlens is None:
            from saklas.io.lens import load_lens

            loaded = load_lens(self.model_id)
            if loaded is not None:
                self._jlens = loaded[0]
                self._jlens_device_cache = {}
        return self._jlens

    def _require_jlens(self) -> "Any":
        from saklas.core.jlens import LensNotFittedError

        lens = self.jlens
        if lens is None:
            raise LensNotFittedError(
                f"no Jacobian lens fitted for {self.model_id} — run "
                f"`saklas lens fit {self.model_id}` first"
            )
        return lens

    def fit_jlens(
        self,
        prompts: "Sequence[str]",
        *,
        corpus_spec: str = "custom",
        source_layers: "Sequence[int] | str | None" = None,
        dim_batch: int | None = None,
        seq_len: int | None = None,
        force: bool = False,
        checkpoint_every: int | None = None,
        on_progress: Callable[[str], None] | None = None,
    ) -> "Any":
        """Fit (or resume fitting) this model's Jacobian lens and persist it.

        Prompts too short for the estimator (≤ ``SKIP_FIRST_POSITIONS + 1``
        tokens) are dropped up front so the saved ``n_prompts`` counts
        consumed prompts exactly — that is what makes resume slicing sound.
        Resume-by-default: when a saved lens matches this corpus (sha256 of
        the filtered prompts) and covers fewer prompts than requested, only
        the remainder is fitted and merged in; ``force=True`` restarts from
        zero. Checkpoints every ``DEFAULT_CHECKPOINT_EVERY`` prompts, so an
        interrupted fit resumes from the last checkpoint.

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
            JacobianLensError,
            fit_jacobian_lens,
        )
        from saklas.io.lens import load_lens, load_lens_sidecar, save_lens
        from saklas.io.lens import (
            load_lens_checkpoint,
            remove_lens_checkpoint,
            save_lens_checkpoint,
        )

        dim_batch = dim_batch or DEFAULT_DIM_BATCH
        seq_len = seq_len or DEFAULT_SEQ_LEN
        checkpoint_every = checkpoint_every or DEFAULT_CHECKPOINT_EVERY
        prompt_list = list(prompts)
        raw_corpus_sha = hashlib.sha256(repr(prompt_list).encode("utf-8")).hexdigest()
        raw_prompt_count = len(prompt_list)

        with self._model_exclusive(
            "session.fit_jlens called while another model use is in flight",
            phase_msg="session.fit_jlens called while a generation is in flight",
        ):
            fit_model = getattr(self._model, "_orig_mod", self._model)
            fit_layers = list(get_layers(fit_model))
            expected_sources = self._resolve_jlens_source_layers(
                source_layers, n_layers=len(fit_layers),
            )
            expected_set = set(expected_sources)
            if force:
                remove_lens_checkpoint(self.model_id)
            else:
                sidecar = load_lens_sidecar(self.model_id)
                if (
                    sidecar is not None
                    and sidecar.get("raw_corpus_sha256") == raw_corpus_sha
                    and sidecar.get("raw_prompt_count") == raw_prompt_count
                    and sidecar.get("seq_len") == seq_len
                    and set(int(l) for l in sidecar.get("source_layers", []))
                    >= expected_set
                ):
                    usable_count = int(sidecar.get("usable_prompt_count", -1))
                    existing = load_lens(self.model_id)
                    if existing is not None:
                        lens, _ = existing
                        if usable_count >= 0 and lens.n_prompts >= usable_count:
                            if on_progress is not None:
                                on_progress(
                                    f"lens already fitted on {lens.n_prompts} "
                                    "prompts — nothing to do"
                                )
                            selected = lens.select_layers(expected_sources)
                            self._jlens = selected
                            self._jlens_device_cache = {}
                            return selected

            usable: list[str] = []
            consumed_ids: list[list[int]] = []
            for prompt in prompt_list:
                ids = self._tokenizer(prompt, return_tensors="pt")["input_ids"][
                    :, :seq_len
                ]
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
            corpus_hash_kind = "token_ids_v1"
            base: Any = None
            if not force:
                sidecar = load_lens_sidecar(self.model_id)
                if (
                    sidecar is not None
                    and sidecar.get("corpus_sha256") == corpus_sha
                    and sidecar.get("corpus_hash_kind") == corpus_hash_kind
                    and sidecar.get("seq_len") == seq_len
                ):
                    existing = load_lens(self.model_id)
                else:
                    existing = None
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
                            selected = lens.select_layers(expected_sources)
                            self._jlens = selected
                            self._jlens_device_cache = {}
                            return selected
                        missing_sources = sorted(expected_set - existing_set)
                        if on_progress is not None:
                            on_progress(
                                "reusing existing fitted layers; fitting missing "
                                f"J-lens layers {missing_sources}"
                            )
                        missing = fit_jacobian_lens(
                            fit_model, self._tokenizer, usable, fit_layers,
                            source_layers=missing_sources, dim_batch=dim_batch,
                            max_seq_len=seq_len, on_progress=on_progress,
                            input_id_rows=consumed_ids,
                        )
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
                        )
                        remove_lens_checkpoint(self.model_id)
                        selected = merged.select_layers(expected_sources)
                        self._jlens = selected
                        self._jlens_device_cache = {}
                        return selected
                    if existing_set >= expected_set:
                        base = lens.select_layers(expected_sources)
                    else:
                        base = None
                ckpt = load_lens_checkpoint(self.model_id)
                if ckpt is not None:
                    partial, ckpt_sidecar = ckpt
                    ckpt_matches = (
                        ckpt_sidecar.get("corpus_sha256") == corpus_sha
                        and ckpt_sidecar.get("corpus_hash_kind") == corpus_hash_kind
                        and ckpt_sidecar.get("seq_len") == seq_len
                        and [int(l) for l in ckpt_sidecar.get("source_layers", [])]
                        == expected_sources
                    )
                    if ckpt_matches:
                        ckpt_base_n = int(ckpt_sidecar.get("base_n_prompts", -1))
                        if base is not None and ckpt_base_n == base.n_prompts:
                            base = JacobianLens.merge([base, partial])
                            if on_progress is not None:
                                on_progress(
                                    "resuming from checkpoint at "
                                    f"{base.n_prompts} prompts"
                                )
                        elif base is None and ckpt_base_n == 0:
                            base = partial
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
                )
                remove_lens_checkpoint(self.model_id)
                return lens

            def _save_checkpoint(partial: "Any") -> None:
                save_lens_checkpoint(
                    partial, self.model_id,
                    base_n_prompts=prompt_base,
                    corpus_spec=corpus_spec,
                    corpus_sha256=corpus_sha,
                    corpus_hash_kind=corpus_hash_kind,
                    seq_len=seq_len,
                    dim_batch=dim_batch,
                    skip_first=SKIP_FIRST_POSITIONS,
                    raw_corpus_sha256=raw_corpus_sha,
                    raw_prompt_count=raw_prompt_count,
                    usable_prompt_count=len(consumed_ids) + prompt_base,
                )

            if base is not None and not usable:
                merged = _save_full(base)
                self._jlens = merged
                self._jlens_device_cache = {}
                return merged

            fitted = fit_jacobian_lens(
                fit_model, self._tokenizer, usable, fit_layers,
                source_layers=expected_sources, dim_batch=dim_batch,
                max_seq_len=seq_len,
                checkpoint_cb=_save_checkpoint,
                checkpoint_every=checkpoint_every,
                on_progress=on_progress,
                input_id_rows=consumed_ids,
            )
            merged = (
                JacobianLens.merge([base, fitted]) if base is not None else fitted
            )
            merged = _save_full(merged)
            self._jlens = merged
            self._jlens_device_cache = {}
            return merged

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
                band = [
                    l for l in range(final_idx)
                    if 0.40 <= l / max(n_layers - 1, 1) <= 0.90
                ]
                return band or list(range(final_idx))
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
            "dict[tuple[int, str, tuple[int, ...]], torch.Tensor] | None",
            getattr(self, "_jlens_device_cache", None),
        )
        if cache is None:
            cache = {}
            self._jlens_device_cache = cache
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

    def _jlens_decode_id(self, token_id: int) -> str:
        """Cache-backed single-token decode shared by every lens readout."""
        decode_cache = cast(
            "dict[int, str] | None", getattr(self, "_jlens_decode_cache", None),
        )
        if decode_cache is None:
            decode_cache = {}
            self._jlens_decode_cache = decode_cache
        tok = decode_cache.get(token_id)
        if tok is None:
            tok = str(self._tokenizer.decode([token_id]))
            decode_cache[token_id] = tok
        return tok

    def _jlens_depths(self, layers: "Sequence[int]") -> list[float]:
        """Normalized layer depths (``layer / (n_layers − 1)``, 0 = first
        block, 1 = last) — the depth axis of the aggregate readout's
        center-of-mass statistic."""
        denom = max(len(self._layers) - 1, 1)
        return [layer / denom for layer in layers]

    def _jlens_logits_rows(
        self,
        lens: "Any",
        rows: list[tuple[int, torch.Tensor]],
    ) -> torch.Tensor:
        """Full-vocab lens logits ``[n_rows, vocab]`` for a batch of
        ``(layer, hidden_row)`` pairs — the shared front half of the
        per-layer top-k and the layer-aggregated readout (one bmm + one
        unembed matvec serves both)."""
        from saklas.core.model import get_final_norm, get_unembedding

        unembed = get_unembedding(self._model)
        device = unembed.device
        unique_layers = sorted({layer for layer, _ in rows})
        J_unique = self._jlens_transport_stack(lens, unique_layers, device)
        layer_to_row = {layer: idx for idx, layer in enumerate(unique_layers)}
        J_rows = J_unique.index_select(
            0,
            torch.tensor(
                [layer_to_row[layer] for layer, _ in rows],
                device=device,
                dtype=torch.long,
            ),
        )
        H = torch.stack([
            hidden.detach().to(torch.float32) for _, hidden in rows
        ]).to(device)
        transported = torch.bmm(J_rows, H.unsqueeze(-1)).squeeze(-1)
        normed = get_final_norm(self._model)(transported)
        return normed.to(unembed.dtype) @ unembed.T

    def _jlens_aggregate_rows(
        self,
        logits: torch.Tensor,
        layers: "Sequence[int]",
        *,
        top_k: int,
    ) -> list[tuple[str, float, float, float]]:
        """Layer-aggregate per-layer lens logits into the decoded chip list
        ``[(token, strength, com, spread), ...]`` (see
        :func:`saklas.core.jlens.aggregate_readout` for the statistics)."""
        from saklas.core.jlens import aggregate_readout

        rows = aggregate_readout(
            logits.float(), self._jlens_depths(layers), top_k=top_k,
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
        salience-weighted depth center of mass; see
        :func:`saklas.core.jlens.aggregate_readout`) and returns the pair
        ``(per_layer, aggregate)`` where ``aggregate`` is one
        ``[(token, strength, com, spread), ...]`` list per position, from
        the same logits (no extra forward or matvec).  The aggregate uses
        the **workspace-band subset** of the requested layers (past 90%
        depth the lens converges on the sampled next-token distribution and
        the early third is noise — the same band policy ``jlens/`` steering
        hard-codes), falling back to every requested layer when none are in
        band; the per-layer matrix always covers the full request.
        """
        from saklas.core.vectors import _capture_all_hidden_states

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
            band = set(self._jlens_workspace_band(lens))
            agg_layers = [l for l in req if l in band] or list(req)
            keep = set(agg_layers)
            agg: list[list[tuple[str, float, float, float]]] = []
            for pos_idx in range(len(pos)):
                sel = [
                    i for i, (layer, p, _) in enumerate(row_refs)
                    if p == pos_idx and layer in keep
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
        workspace_band, readout: {layer: [(token, logprob, id), ...]},
        aggregate: [(token, strength, com, spread), ...]}`` — ``aggregate``
        is the layer-aggregated view of the same logits (per-layer softmax
        → mean-probability strength + salience-weighted depth center of
        mass; :func:`saklas.core.jlens.aggregate_readout`).
        Raises :class:`~saklas.core.jlens.LensNotFittedError` with no
        fitted lens, :class:`UnknownNodeError` /
        :class:`InvalidNodeOperationError` on a bad target (mirrors
        :meth:`fork_from_token`), ``ValueError`` on an unfitted layer.
        """
        from saklas.core.vectors import _capture_all_hidden_states

        lens = self._require_jlens()
        req = self._resolve_jlens_layers(lens, layers)
        missing = [l for l in req if l not in lens.jacobians]
        if missing:
            raise ValueError(
                f"layers {missing} not in the fitted lens "
                f"(fitted: {lens.source_layers[0]}..{lens.source_layers[-1]})"
            )
        node = self.tree.get(node_id)
        if node.role != "assistant":
            raise InvalidNodeOperationError(
                f"jlens_token_readout: {node_id!r} is a {node.role} node, "
                f"not an assistant node with a decode record"
            )
        raw_ids = node.raw_token_ids
        if not raw_ids:
            raise InvalidNodeOperationError(
                f"jlens_token_readout: {node_id!r} has no raw token record "
                f"(legacy or transcript-loaded node)"
            )
        if not 0 <= raw_index < len(raw_ids):
            raise InvalidNodeOperationError(
                f"jlens_token_readout: raw_index {raw_index} out of range "
                f"[0, {len(raw_ids)}) for {node_id!r}"
            )
        user_node = (
            self.tree.nodes.get(node.parent_id) if node.parent_id else None
        )
        if not raw and (user_node is None or user_node.role != "user"):
            raise InvalidNodeOperationError(
                f"jlens_token_readout: {node_id!r} has no user parent to "
                f"rebuild the prompt from (raw-mode node? pass raw=true)"
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
                assert user_node is not None  # narrowed above
                prompt_ids = self._prepare_input(
                    user_node.text,
                    thinking=use_thinking,
                    parent_node_id=user_node.parent_id,
                    user_role=user_node.role_label,
                    assistant_role=node.role_label,
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
                # Same band policy as jlens_readout: aggregate over the
                # workspace-band subset (the route's default request IS the
                # band, so this is a no-op there), matrix over the full
                # request.
                band = set(self._jlens_workspace_band(lens))
                agg_idx = [i for i, l in enumerate(req) if l in band]
                if not agg_idx:
                    agg_idx = list(range(len(req)))
                agg = self._jlens_aggregate_rows(
                    logits[agg_idx],
                    [req[i] for i in agg_idx],
                    top_k=top_k,
                )
        return {
            "node_id": node_id,
            "raw_index": int(raw_index),
            "token_id": int(raw_ids[raw_index]),
            "token_text": str(self._tokenizer.decode([int(raw_ids[raw_index])])),
            "steering": steering_expr,
            "workspace_band": self._jlens_workspace_band(lens),
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
        channel (per-layer softmax salience/probability) via the session
        lens-probe registry instead.

        Restricted to the **workspace band** (40–90% depth): in the motor
        regime a lens direction converges on the raw unembedding row, so
        pushing there is direct token-forcing — live-verified on gemma-3-4b
        to shatter into token loops at every α — and the early third of the
        lens is noise. The paper's own injection/ablation experiments act on
        the workspace layers only.
        """
        from saklas.core.jlens import resolve_word_token
        from saklas.core.model import get_unembedding

        name = f"jlens/{word}"
        if name in self._profiles:
            return name
        lens = self._require_jlens()
        token_id = resolve_word_token(self._tokenizer, word)
        band = set(self._jlens_workspace_band(lens))
        directions = lens.token_direction(
            token_id, get_unembedding(self._model), layers=sorted(band),
        )
        # ``token_direction`` returns CPU tensors; land the profile on the
        # session device so the probe fold (which follows the directions'
        # device) builds device-consistent whitened factors — CPU factors
        # crash the per-probe read paths (``_subspace_coords_for``) against
        # on-device activations.
        self._profiles[name] = {
            l: d.to(self._device) for l, d in directions.items() if l in band
        }
        self._invalidate_prefix_cache()
        self._invalidate_analytics_cache()
        return name

    def enable_live_lens(
        self,
        *,
        layers: "Sequence[int] | None" = None,
        top_k: int = 5,
    ) -> list[int]:
        """Stream the J-lens readout live during generation.

        Every decode step, the top-``top_k`` lens tokens at each selected
        layer ride ``TokenEvent.lens_readout`` (and the TUI's ``/lens``
        panel). ``layers`` defaults to **every** fitted layer in the 40–90%
        depth band (the paper's workspace range) — the per-step cost is one
        d×d matvec + one vocab matvec + an on-device top-k per layer, cheap
        enough that the full band beats a 5-layer subsample (which
        under-sampled the depth CoM). The selected layers' ``J_l`` move
        device-resident here, once — a full-band residency is
        ``n_band · d_model² · 4`` bytes (~450 MB on a 4B, ~1.5 GB on a
        12B); pass an explicit ``layers`` subset to trade coverage for
        memory.

        Attaches **no forward hooks** (the reader consumes the capture's
        existing latest-slice buffers post-forward), so steering fast-path /
        compile eligibility is untouched; the one concession is that capture
        itself routes through transient hooks while the live lens is on
        (the persistent compiled-clean capture buffers only cover probe
        layers). Returns the resolved layer list.
        """
        from saklas.core.model import get_final_norm, get_unembedding

        lens = self._require_jlens()
        if layers is None:
            layers = sorted(self._jlens_workspace_band(lens))
        else:
            layers = sorted(set(int(l) for l in layers))
            missing = [l for l in layers if l not in lens.jacobians]
            if missing:
                raise ValueError(
                    f"layers {missing} not in the fitted lens "
                    f"(fitted: {lens.source_layers[0]}..{lens.source_layers[-1]})"
                )
        device = self._device
        layer_list = list(layers)
        if layer_list:
            j_stack = self._jlens_transport_stack(lens, layer_list, device)
        else:
            sample = next(iter(lens.jacobians.values()))
            j_stack = torch.empty(
                (0, *sample.shape), device=device, dtype=torch.float32,
            )
        band = set(self._jlens_workspace_band(lens))
        self._live_lens = {
            "layers": layer_list,
            "top_k": int(top_k),
            "J_stack": j_stack,
            "layer_rows": {l: i for i, l in enumerate(layer_list)},
            "unembed": get_unembedding(self._model),
            "norm": get_final_norm(self._model),
            # Aggregate over the workspace-band subset of the live layers
            # (falling back to all of them) — the same band policy every
            # other aggregate surface applies.
            "agg_layers": (
                frozenset(l for l in layer_list if l in band)
                or frozenset(layer_list)
            ),
        }
        return list(layers)

    def disable_live_lens(self) -> None:
        """Stop streaming the live lens readout and free the device J_l copies."""
        self._live_lens = None

    @property
    def live_lens_layers(self) -> list[int] | None:
        """The live lens readout's layer list, or ``None`` when it's off."""
        if self._live_lens is None:
            return None
        return list(self._live_lens["layers"])

    def _jlens_workspace_band(self, lens: "Any") -> list[int]:
        """Fitted lens layers in the 40–90% depth band — the paper's
        workspace range. Falls back to every fitted layer for models too
        shallow to have a band (the CPU-test toys)."""
        n = len(self._layers)
        band = [
            l for l in lens.source_layers if 0.40 <= l / max(n - 1, 1) <= 0.90
        ]
        return band or list(lens.source_layers)

    def _live_lens_readout_step(
        self,
    ) -> (
        tuple[
            dict[int, list[tuple[str, float]]],
            list[tuple[str, float, float, float]],
        ]
        | None
    ):
        """One decode step's lens readout from the capture's latest slices.

        Runs post-forward at the token tap (never inside a hook). On-device
        matvecs + top-k; a handful of small host transfers per step; token
        decoding memoized across steps. Returns ``(per_layer, aggregate)``
        — the top-k tokens per selected layer, each scored by its
        **per-layer softmax probability** (the same strength unit every
        other lens surface reports — apples-to-apples across layers, unlike
        a raw logit), plus the layer-aggregated chip list ``[(token,
        strength, com, spread), ...]`` over the workspace-band subset of
        the live layers.
        """
        state = self._live_lens
        if state is None or not getattr(self, "_live_lens_active_for_generation", True):
            return None
        buckets = self._capture.per_layer_buckets()
        unembed = state["unembed"]
        layers_present: list[int] = []
        hidden_rows: list[torch.Tensor] = []
        transport_rows: list[int] = []
        layer_rows: dict[int, int] = state["layer_rows"]
        for layer in state["layers"]:
            bucket = buckets.get(layer)
            if not bucket:
                continue
            layers_present.append(layer)
            hidden_rows.append(bucket[-1].to(torch.float32))
            transport_rows.append(layer_rows[layer])
        if not layers_present:
            return None
        # ``getattr`` — test stubs bind these methods without instance state.
        stash = getattr(self, "_lens_step_stash", None)
        if (
            stash is not None
            and stash.get("fresh")
            and stash.get("layers") == tuple(layers_present)
        ):
            # The gate callback already computed this forward's band logits
            # (score_callback runs before the token tap) — reuse them.
            stash["fresh"] = False
            logits = stash["logits"]
        else:
            J_stack: torch.Tensor = state["J_stack"]
            rows = torch.tensor(
                transport_rows, device=J_stack.device, dtype=torch.long,
            )
            J = J_stack.index_select(0, rows)
            H = torch.stack(hidden_rows, dim=0).to(J.device)
            transported = torch.bmm(J, H.unsqueeze(-1)).squeeze(-1)
            normed = state["norm"](transported)
            logits = normed.to(unembed.dtype) @ unembed.T
        # Pinned lens probes ride the same logits — per-step readout-channel
        # readings for the payload merge (zero extra matvecs).
        if getattr(self, "_lens_probes", None):
            self._last_lens_step_readings = self._score_lens_probes(
                {}, logits=logits, layers=list(layers_present),
            )
        else:
            self._last_lens_step_readings = None
        # Display scores are per-layer softmax probabilities — the one
        # strength unit every lens surface reports (softmax is monotone, so
        # the top-k selection is unchanged from the raw-logit ranking).
        vals, idxs = logits.float().softmax(dim=-1).topk(
            state["top_k"], dim=-1,
        )
        # one batched host transfer for the per-layer block
        all_vals = vals.cpu()
        all_idxs = idxs.cpu()
        out: dict[int, list[tuple[str, float]]] = {}
        for row, layer in enumerate(layers_present):
            pairs: list[tuple[str, float]] = []
            for v, i in zip(all_vals[row], all_idxs[row]):
                pairs.append((self._jlens_decode_id(int(i)), float(v)))
            out[layer] = pairs
        agg_keep: frozenset[int] = state["agg_layers"]
        agg_sel = [
            row for row, layer in enumerate(layers_present) if layer in agg_keep
        ] or list(range(len(layers_present)))
        agg = self._jlens_aggregate_rows(
            logits[agg_sel],
            [layers_present[row] for row in agg_sel],
            top_k=state["top_k"],
        )
        return out, agg

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
        directions = self._ensure_profile_registered(selector)
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
        (:func:`_system_for`, led by the shared :data:`~saklas.core.vectors._LENGTH_DIRECTIVE`
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
        from saklas.core.vectors import _LENGTH_DIRECTIVE, _load_baseline_prompts

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
        :data:`~saklas.core.vectors._LENGTH_DIRECTIVE` as its *only* system prompt
        (no persona) with the standard assistant label, so the corpus is the
        model\'s own voice -- just brief.  The directive is the only framing it
        shares with the node corpora (same on every node system + every capture),
        so it cancels at extraction while leaving neutral the default-voice
        reference the contrast subtracts against.  Same prompts-cycled order
        (``response[i] -> prompt[i % k]``).  This is what ``neutral_statements.json``
        holds under 4.0 -- regenerate it with this and the per-model
        ``layer_means`` / neutral-activation caches recompute conversationally.
        """
        from saklas.core.vectors import _LENGTH_DIRECTIVE, _load_baseline_prompts

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
        the manifold the single on-disk artifact.  Emits ``VectorExtracted``.

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

    def extract_vector_from_corpora(
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
        the hand-authored TUI/HTTP paths, which already hold the positive and
        negative corpora and so skip generation.  Authors a 2-node ``pca``
        manifold (``positive`` → pole node, ``negative`` → its opposite) and
        fits it; returns ``(canonical_name, Profile)`` like :meth:`extract`.

        Under 4.0 the corpora are pooled conversationally — each entry is treated
        as a response to the shared baseline prompts (``response[i] -> prompt[i
        % k]``), so each corpus length must be a multiple of the baseline prompt
        set.  ``kind`` is recorded per node (provenance); ``role`` opts into a
        persona-baselined fit as in :meth:`extract`.
        """
        with self._model_exclusive(
            "session.extract_vector_from_corpora called while another "
            "model use is in flight",
            phase_msg="session.extract_vector_from_corpora called while a "
            "generation is in flight",
        ):
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
        of :meth:`extract` / :meth:`extract_vector_from_corpora`.

        Both callers differ only in their corpus source (generate vs given); by
        the time they reach here they hold ``node_corpora`` (label → corpus) and
        ``node_kinds`` (label → kind).  This rmtrees + re-authors the folder when
        ``force`` or the manifest is missing (``node_corpora`` is consulted only
        then, so a cache-hit caller may pass ``None``), then delegates to
        :meth:`_fit_vector_manifold`.  Runs with ``_gen_lock`` already held by
        the caller.
        """
        from saklas.io.manifolds import create_discover_manifold_folder
        from saklas.io.paths import manifold_dir

        folder = manifold_dir(ns, name)
        manifest = folder / "manifold.json"
        if force or not manifest.exists():
            assert node_corpora is not None
            node_roles: dict[str, str | None] | None = (
                {label: role for label in node_corpora} if role else None
            )
            if manifest.exists():
                import shutil
                shutil.rmtree(folder)
            create_discover_manifold_folder(
                ns, name, description, fit_mode="pca",
                node_corpora=node_corpora,
                hyperparams={"max_dim": 1, "var_threshold": 0.7},
                node_roles=node_roles, node_kinds=node_kinds,
            )
        return self._fit_vector_manifold(
            name, folder, sae=sae, sae_revision=sae_revision,
            role=role, on_progress=on_progress,
        )

    def _fit_vector_manifold(
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

        Shared tail of :meth:`extract` / :meth:`extract_vector_from_corpora`:
        runs :class:`ManifoldExtractionPipeline` directly (the public
        :meth:`fit` re-acquires ``_gen_lock``, which the callers
        already hold), folds the fitted manifold to a per-layer direction
        :class:`Profile`, and emits ``VectorExtracted``.

        The returned name carries the variant tail (``:sae-<release>`` /
        ``:role-<slug>``) so the caller steers the right per-model tensor; the
        on-disk manifold folder stays the bare canonical name.
        """
        from saklas.core.extraction import ManifoldExtractionPipeline
        from saklas.core.vectors import folded_vector_directions

        pipe = ManifoldExtractionPipeline(self, self.events)
        manifold = pipe.fit(
            folder, sae=sae, sae_revision=sae_revision, on_progress=on_progress,
        )
        # The newly-fit manifold changes folded directions a probe may read.
        self._invalidate_analytics_cache()
        if sae:
            ret_name = f"{name}:sae-{sae}"
        elif role:
            ret_name = f"{name}:role-{role}"
        else:
            ret_name = name
        profile = Profile(
            folded_vector_directions(manifold),
            metadata={
                "method": "manifold_pca",
                "name": ret_name,
                "share_metric": manifold.metadata.get("share_metric"),
            },
        )
        self.events.emit(VectorExtracted(
            name=ret_name, profile=profile, metadata=dict(profile.metadata),
        ))
        return ret_name, profile

    def fit(
        self,
        folder: Any,
        *,
        sae: str | None = None,
        sae_revision: str | None = None,
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
        """
        with self._model_exclusive(
            "session.fit called while another model use is in flight",
            phase_msg="session.fit called while a generation is in flight",
        ):
            try:
                from saklas.core.extraction import ManifoldExtractionPipeline
                pipe = ManifoldExtractionPipeline(self, self.events)
                return pipe.fit(
                    folder, sae=sae, sae_revision=sae_revision,
                    force=force, on_progress=on_progress,
                )
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
        """Bake a steering expression into a corpus-less manifold (the merge op).

        The Python mirror of CLI ``manifold bake``.  Wraps
        :func:`saklas.io.merge.merge_into_manifold`, model-scoped to this
        session's loaded model — the merge lands a corpus-less
        ``fit_mode="baked"`` manifold under ``local/<name>/`` — then folds the
        fitted tensor back to a steering :class:`Profile` and registers it
        (:meth:`steer`) so it is immediately steerable.  Returns
        ``(name, Profile)``, the same shape :meth:`extract` returns.
        """
        from saklas.io.merge import MergeError, merge_into_manifold
        from saklas.io.paths import tensor_filename
        from saklas.core.manifold import load_manifold
        from saklas.core.vectors import folded_vector_directions

        dst_folder = merge_into_manifold(
            name, expression, self.model_id, force=force, strict=strict,
        )
        tensor_path = dst_folder / tensor_filename(self.model_id)
        if not tensor_path.is_file():
            raise MergeError(
                f"bake produced no tensor for {self.model_id} at {tensor_path}"
            )
        manifold = load_manifold(str(tensor_path))
        profile = Profile(folded_vector_directions(manifold))
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
        time; this is the canonical resolver site — CLI, server, and
        TUI all route through here.  Nesting flattens: an inner
        ``steering("0.5 angry.calm")`` overrides the outer
        ``steering("0.3 angry.calm")`` for the duration of the inner
        scope, and the outer entry is restored on ``__exit__``.  One hook
        installation per active layer regardless of nesting depth.

        Unknown vector names raise ``VectorNotRegisteredError``; genuinely
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
        composer = self._get_steering_composer()
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
                # Resolve via the unified path (fold a fitted manifold, or
                # port a legacy vectors/ folder on first touch).  A miss /
                # non-raw variant error surfaces at hook-install with the
                # shared VectorNotRegisteredError shape.
                with suppress(Exception):
                    self._ensure_profile_registered(target, role="ablation")
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
            self._ensure_manifold_loaded(val.manifold)
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
                # Should not happen — ``_ensure_manifold_loaded`` either
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

    def _ensure_manifold_loaded(self, key: str) -> None:
        """Load the manifold artifact for registry key ``key`` if absent.

        Compatibility wrapper over
        :meth:`SteeringComposer.ensure_manifold_loaded`; production steering
        code talks to the composer directly.
        """
        self._get_steering_composer().ensure_manifold_loaded(key)

    def _ensure_profile_registered(
        self, name: str, *, role: str = "vector",
    ) -> dict[int, torch.Tensor]:
        """Direction profile for ``name`` — registered tensor or folded manifold.

        Compatibility wrapper over
        :meth:`SteeringComposer.ensure_profile_registered`; public callers
        should prefer :meth:`ensure_profile_registered`.
        """
        return self._get_steering_composer().ensure_profile_registered(name, role=role)

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
                        self._get_steering_composer().ensure_manifold_loaded(key)
                    except ManifoldNotRegisteredError:
                        self.fit(manifold_dir("default", name))
                        self._get_steering_composer().ensure_manifold_loaded(key)
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
                folder = ManifoldFolder.load(manifold_dir("default", name))
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
                self._get_steering_composer().ensure_manifold_loaded(key)
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
        self._get_steering_composer().push(entries)

    def _pop_steering(self) -> None:
        """Pop the top of the steering stack and rebuild hooks.

        Thin forwarder to :meth:`SteeringComposer.pop`; ``_SteeringContext``
        calls it on the session, so it stays as a session method.
        """
        self._get_steering_composer().pop()

    def _steering_needs_probe_gating(self) -> bool:
        """Return True iff any active steering trigger carries a
        :class:`~saklas.core.triggers.ProbeGate`.

        Compatibility wrapper over
        :meth:`SteeringComposer.steering_needs_probe_gating`; production
        generation code talks to the composer directly.
        """
        return self._get_steering_composer().steering_needs_probe_gating()

    def _steering_active_in_prefill(self) -> bool:
        """Return True iff any active steering term fires during prompt prefill.

        Thin forwarder to :meth:`SteeringComposer.steering_active_in_prefill`.
        """
        return self._get_steering_composer().steering_active_in_prefill()

    def _steering_value_prefill_inactive(
        self, value: "str | Steering | None",
    ) -> bool:
        """Return True iff steering ``value`` would not touch the prompt prefill.

        Thin forwarder to :meth:`SteeringComposer.steering_value_prefill_inactive`.
        """
        return self._get_steering_composer().steering_value_prefill_inactive(value)

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

        Compatibility wrapper over
        :meth:`SteeringComposer.build_gating_score_callback`; production
        generation code talks to the composer directly.
        """
        return self._get_steering_composer().build_gating_score_callback()

    def _rebuild_steering_hooks(self) -> None:
        """Tear down existing hooks and install from the flattened stack head.

        Thin forwarder to :meth:`SteeringComposer.rebuild_hooks`; kept on the
        session because test stubs override it (and the composer's push/pop reach
        it through the session back-ref).
        """
        self._get_steering_composer().rebuild_hooks()

    def _begin_capture(
        self, *, widen: bool = False, need_per_token: bool = True,
        gating_only_probes: set[str] | None = None,
        gating_probe_keys: set[str] | None = None,
        lean_per_token: bool = False,
        final_probe_aggregate: bool = True,
        live_lens_active: bool = True,
    ) -> bool:
        """Attach hidden-state capture. Returns True if attached.

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
        # ``getattr`` like ``_compiled_clean_eligible`` below — spec'd mock
        # stubs carry class attributes only, not instance state.
        live_lens = getattr(self, "_live_lens", None) if live_lens_active else None
        lens_probes = getattr(self, "_lens_probes", None) or {}
        # Per-generation reset of the per-forward lens stash + display row.
        self._lens_step_stash = None
        self._last_lens_step_readings = None
        if widen:
            layer_idxs = list(range(len(self._layers)))
        else:
            gate_only_no_final = bool(
                gating_only_probes
                and need_per_token
                and not final_probe_aggregate
                and self._monitor.probe_names
            )
            gate_probe_names = (
                set(gating_only_probes)
                if gate_only_no_final and gating_only_probes is not None
                else None
            )
            union: set[int] = self._monitor.probe_layers(gate_probe_names)
            if live_lens is not None:
                # The live workspace readout consumes the same latest-slice
                # buffers the monitor does — its layers join the capture set.
                union = union | set(live_lens["layers"])
            if lens_probes:
                # Pinned lens probes read their band layers: per-forward for
                # gates (live or not), and the tail ring pools the finalize
                # aggregate from the same slices.
                union = union | self._lens_probe_layers()
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
            # Persistent capture buffers are installed for every layer before
            # compile; layer_idxs selects the subset this generation consumes.
            # Live lens layers can therefore ride the same compile-clean path
            # instead of forcing transient hooks.
            and getattr(self, "_compiled_clean_eligible", False)
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

            def _score_step_subset(latest: dict[int, torch.Tensor]) -> None:
                # Score ONLY the exact gate scalar keys each token (the gate's
                # sole consumer); append even an empty dict to keep row indices
                # aligned with decode forwards.  The full roster is pooled once at
                # finalize from the tail ring (``_score_aggregate_only``).
                self._incremental_gate_scores.append(
                    self._monitor.score_gate_scalars(
                        latest, gate_keys, probe_names=subset,
                    )
                    if latest and gate_keys else {}
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
            def _score_step_lean(latest: dict[int, torch.Tensor]) -> None:
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
            def _score_step(latest: dict[int, torch.Tensor]) -> None:
                # Score once while the just-produced hidden slice is still the
                # only retained device payload.  Append even an empty dict so
                # row indices remain aligned with decode forwards; finalization
                # trims any terminal EOS-only overcapture to generated_ids.
                self._incremental_readings.append(
                    self._monitor.score_single_token(latest) if latest else {}
                )

            if lens_probes:
                # Lens probes pool their finalize aggregate from the tail
                # ring; ``set_incremental``'s length-1 buffers can't walk back
                # past trailing specials, so arm the ring alongside the sink.
                self._capture.set_tail_with_sink(_AGG_TAIL_DEPTH, _score_step)
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
        elif not widen and (live_lens is not None or lens_probes):
            # Lens-only capture (live lens on and/or lens probes pinned, no
            # monitor probes): a bounded tail ring keeps the reader's latest
            # slice fresh without FULL retention growing over the generation.
            # No step sink — the lens display reader runs at the token tap,
            # lens gates score from ``latest_per_layer`` in the gate
            # callback, and the lens finalize aggregate pools from the ring.
            self._capture.set_aggregate_tail(_AGG_TAIL_DEPTH)
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
        from saklas.core.vectors import last_content_index
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
        self, generated_ids: list[int], *, accumulate: bool = True,
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
        # the right warm start here).
        self._monitor.enable_curved_warm(False)
        agg_vals = self._monitor.score_aggregate(pooled)
        # Fill any probe the pool missed (e.g. a layer absent from the ring).
        agg_vals = {name: agg_vals.get(name, empty[name]) for name in names}
        if accumulate and agg_vals:
            self._monitor.accumulate_readings(agg_vals)
        return agg_vals

    def _score_lean_incremental(
        self, generated_ids: list[int], *, accumulate: bool = True,
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
            generated_ids, accumulate=accumulate,
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
            for probe_name, manifold in self._monitor.manifolds.items():
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
        # probe — it reads the readout channel (per-layer softmax salience +
        # probability of the token under ``softmax(W_U · norm(J_l h))``, mean-
        # banded), the paper-native "how disposed is the model to say this
        # word" quantity, not a whitened coordinate along ``W_U[v] @ J_l``.
        # Routed to the session lens-probe registry; the Monitor never sees it
        # (no whitener requirement — the softmax is the calibration).
        if selector.startswith("jlens/"):
            return self._add_lens_probe(selector, as_name=as_name)
        # Manifold reads are Mahalanobis-only: force the lazy whitener build
        # (and its push into the monitor) before attaching, or ``add_probe`` →
        # ``_build_whitened_factors`` raises on a missing covering whitener.
        _ = self.whitener
        name = as_name if as_name is not None else selector
        # Probe attach loads the manifold onto the model device and builds
        # device-resident whitened factors — GPU work that must not run
        # concurrently with a fit / extract / generation on PyTorch's single
        # global MPS command buffer (which would abort the process).  Hold the
        # exclusive-GPU ``_gen_lock`` non-blocking: a cross-thread model op in
        # flight refuses rather than races; same-thread reentry passes (RLock).
        with self._model_exclusive(
            "add_probe called while another model operation is in "
            "flight; retry shortly"
        ):
            manifold = self._resolve_probe_manifold(selector)
            self._monitor.add_probe(name, manifold, top_n=top_n)
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
        """Attach a ``jlens/<word>`` token probe to the lens-probe registry.

        Validates the lens artifact + single-token word (the same
        ``resolve_word_token`` contract steering atoms use), records the
        workspace-band layer set, and pre-warms the device transport stack so
        the first decode step doesn't hitch on the J_l transfer.  The probe
        reads ONE channel — ``coords = (strength,)``, the mean band
        probability ``mean_l p_l(v)`` (the workspace card's ``strength``;
        objective and apples-to-apples across tokens and layers) — so
        ``@when:jlens/<word> > x`` gates strength.
        """
        from saklas.core.jlens import resolve_word_token

        word = selector.split("/", 1)[1]
        if not word:
            raise ValueError("empty jlens probe word")
        name = as_name if as_name is not None else selector
        with self._model_exclusive(
            "add_probe called while another model operation is in "
            "flight; retry shortly"
        ):
            lens = self._require_jlens()
            token_id = resolve_word_token(self._tokenizer, word)
            band = [int(l) for l in self._jlens_workspace_band(lens)]
            self._jlens_transport_stack(lens, sorted(band), self._device)
            self._lens_probes[name] = {
                "word": word, "token_id": int(token_id), "layers": band,
            }
        # Same invalidation set as a monitor attach: the capture layer union
        # changed (band layers join it), and the probe hash / analytics
        # caches key on the roster.
        self._invalidate_prefix_cache()
        self._probe_hash_cache.pop(name, None)
        self._invalidate_analytics_cache()
        return name

    def remove_probe(self, name: str) -> None:
        """Detach a previously-attached probe (any shape)."""
        # ``getattr`` — spec'd mock stubs carry class attributes only.
        lens_probes = getattr(self, "_lens_probes", None)
        if lens_probes and name in lens_probes:
            del lens_probes[name]
        else:
            self._monitor.remove_probe(name)
        self._invalidate_prefix_cache()
        self._probe_hash_cache.pop(name, None)
        self._invalidate_analytics_cache()

    @property
    def lens_probe_names(self) -> list[str]:
        """Names of the attached J-lens token probes (readout channel)."""
        return list(self._lens_probes)

    def _lens_probe_layers(self) -> set[int]:
        """Union of the attached lens probes' band layers."""
        out: set[int] = set()
        for spec in self._lens_probes.values():
            out.update(spec["layers"])
        return out

    def _score_lens_probes(
        self,
        hidden: dict[int, torch.Tensor],
        *,
        logits: torch.Tensor | None = None,
        layers: "Sequence[int] | None" = None,
    ) -> dict[str, "ProbeReading"]:
        """Score every attached lens probe from hidden slices (or reuse
        precomputed lens ``logits`` rows aligned with ``layers``).

        Returns ``{name: ProbeReading}`` with ``coords = (strength,)`` —
        the ONE readout channel, mean band probability —
        ``coords_per_layer[l] = (p_l,)``, and the depth CoM — the
        readout-channel synthesis of the unified reading shape (geometry
        fields defaulted: ``fraction`` / ``residual`` 0, ``nearest`` /
        ``assignment`` empty, ``membership`` 1.0).  Empty when no probe
        layer is available.
        """
        from saklas.core.jlens import token_readout_stats

        if not self._lens_probes:
            return {}
        lens = self.jlens
        if lens is None:
            return {}
        band = self._lens_probe_layers()
        if logits is None:
            layers = sorted(l for l in band if l in hidden)
            if not layers:
                return {}
            logits = self._jlens_logits_rows(
                lens, [(l, hidden[l]) for l in layers],
            )
        else:
            assert layers is not None
            # Restrict precomputed rows (e.g. a custom live-lens layer set)
            # to the probes' band.
            keep = [i for i, l in enumerate(layers) if l in band]
            if not keep:
                return {}
            if len(keep) != len(layers):
                logits = logits[keep]
            layers = [layers[i] for i in keep]
        names = list(self._lens_probes)
        stats = token_readout_stats(
            logits.float(),
            self._jlens_depths(layers),
            [self._lens_probes[n]["token_id"] for n in names],
        )
        out: dict[str, ProbeReading] = {}
        for name, (strength, com, spread, per_layer) in zip(names, stats):
            out[name] = ProbeReading(
                fraction=0.0,
                nearest=[],
                coords=(strength,),
                residual=0.0,
                coords_per_layer={
                    l: (p_l,) for l, p_l in zip(layers, per_layer)
                },
                depth_com=(com,),
                depth_spread=(spread,),
            )
        return out

    def _score_lens_gate_scalars(self) -> dict[str, float]:
        """Per-forward lens-probe gate scalars from the latest capture slices.

        Called from the gating score callback (once per decode forward, before
        the token tap).  Computes the band lens logits, stashes them for the
        display step to reuse (``_lens_step_stash``), and flattens the
        synthesized readings through :meth:`Monitor.flat_scalars` so the gate
        key space is uniform (``jlens/<word>`` = strength, the mean band
        probability).  Empty when nothing is capturable yet.
        """
        if not self._lens_probes:
            return {}
        lens = self.jlens
        if lens is None:
            return {}
        latest = self._capture.latest_per_layer()
        if not latest:
            return {}
        band = self._lens_probe_layers()
        layers = sorted(l for l in band if l in latest)
        if not layers:
            return {}
        logits = self._jlens_logits_rows(
            lens, [(l, latest[l]) for l in layers],
        )
        self._lens_step_stash = {
            "layers": tuple(layers), "logits": logits, "fresh": True,
        }
        readings = self._score_lens_probes({}, logits=logits, layers=layers)
        return Monitor.flat_scalars(readings)

    def _score_lens_probes_aggregate(
        self, generated_ids: list[int],
    ) -> dict[str, "ProbeReading"]:
        """End-of-gen lens-probe aggregate pooled at the last content token.

        Mirrors :meth:`_score_aggregate_only`: one readout at the pooled
        slice from the capture tail ring (or the FULL-mode stack), so the
        aggregate semantics match the monitor probes' exactly.
        """
        if not self._lens_probes or not generated_ids:
            return {}
        agg_fwd = self._aggregate_forward_index(generated_ids)
        if agg_fwd is None:
            return {}
        pooled = self._capture.tail_slice_at(agg_fwd)
        if not pooled:
            stacked = self._capture.stacked()
            pooled = {
                l: t[agg_fwd]
                for l, t in stacked.items()
                if t.shape[0] > agg_fwd
            }
        if not pooled:
            return {}
        return self._score_lens_probes(pooled)

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
            self._ensure_manifold_loaded(selector)
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
            self._layer_means = bootstrap_layer_means(
                self._model, self._tokenizer, self._layers, self._model_info,
            )
            self._monitor.layer_means = self._layer_means
        from saklas.core.vectors import fold_directions_to_subspace
        return fold_directions_to_subspace(
            name, dict(profile), self._layer_means, whitener=self.whitener,
        )

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
        lens_spec = self._lens_probes.get(name)
        if lens_spec is not None:
            # A lens probe has no baked tensor — hash the readout-channel
            # identity (model, token, band, channel version) so transcript
            # drift detection still works across a semantics change (v2:
            # single strength axis; v1 carried a salience axis).
            import hashlib
            digest = hashlib.sha256(
                repr(
                    (
                        "jlens-readout-v2", self.model_id, lens_spec["word"],
                        lens_spec["token_id"], tuple(lens_spec["layers"]),
                    )
                ).encode("utf-8")
            ).hexdigest()
            self._probe_hash_cache[name] = digest
            return digest
        manifold = self._monitor.manifolds.get(name)
        if manifold is None:
            return None
        # Hash the probe's baked direction view (the folded ``{L: δ̂_L ·
        # share_L}``) for continuity with the pre-coords scalar monitor:
        # the per-layer baked tensor is what the transcript-drift check
        # compared, and it's stable across the manifold round-trip.
        from saklas.core.vectors import folded_vector_directions
        import hashlib
        h = hashlib.sha256()
        try:
            # 2-node R=1 concept probe: hash the folded baked-direction view,
            # for continuity with the pre-coords scalar monitor's drift check.
            profile = folded_vector_directions(manifold)
            per_layer = {L: [profile[L]] for L in profile}
        except ValueError:
            # Multi-node / curved probe (e.g. ``personas``): no R=1 fold exists.
            # Hash the per-layer subspace geometry directly — mean + basis (+ the
            # real node_coords when stamped) — a deterministic digest for any
            # manifold shape, so attaching a multi-node probe is reproducible too.
            per_layer = {}
            for layer_idx, sub in manifold.layers.items():
                tensors = [sub.mean, sub.basis]
                if sub.node_coords is not None:
                    tensors.append(sub.node_coords)
                per_layer[layer_idx] = tensors
        for layer_idx in sorted(per_layer.keys()):
            for tensor in per_layer[layer_idx]:
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
        for name in (*self._monitor.probe_names, *self._lens_probes):
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
                if parent.role == "assistant" and parent.recipe is not None:
                    anchor = parent.recipe
                else:
                    for nid in self.tree.ancestors_of(parent_node_id):
                        anc = self.tree.nodes.get(nid)
                        if anc is not None and anc.role == "assistant" and anc.recipe is not None:
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

        # Resolve which node to anchor the regen under.  If the caller
        # passed an assistant node, regen siblings under its user-parent;
        # if a user node, siblings under it directly.
        anchor_user_id = parent.parent_id if parent.role == "assistant" else parent_node_id

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
        *,
        allow_any_parent: bool = False,
        role_label: str | None = None,
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

        ``allow_any_parent`` skips the user-under-user guard: in flat
        (base-model) mode the role tag is authorship provenance, not a
        turn-taking constraint, so an authored span may hang under a
        node of any role.

        ``text`` must be non-empty (whitespace-only is honored, but a
        completely empty string raises) — empty commits should be
        no-op'd at the surface, not sent over the wire.
        """
        if text == "":
            raise InvalidNodeOperationError(
                "append_user_turn: text must be non-empty"
            )
        if not allow_any_parent:
            self._check_user_send_target(parent_node_id)
        return self.tree.add_user_turn(
            text, parent_id=parent_node_id, role_label=role_label)

    def append_assistant_turn(
        self,
        user_node_id: str,
        text: str,
        *,
        role_label: str | None = None,
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

        use_static = bool(prefer_static and getattr(self, "_static_cache_active", False))
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
        if not isinstance(entry, _PrefixCacheEntry):
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

    def _prepare_input(
        self, input: Any, raw: bool = False, thinking: bool = False,
        stateless: bool = False,
        parent_node_id: str | None = None,
        user_role: str | None = None,
        assistant_role: str | None = None,
        to_device: bool = True,
    ) -> torch.Tensor:
        if raw and isinstance(input, str):
            # Flat (base-model / completion) path: no chat template, no
            # role markers.  The model sees the active-path text verbatim
            # — every node along the loom path concatenated — plus this
            # call's own ``input``.  ``stateless`` skips the tree walk so
            # the buffer is purely ``input``.
            prefix = "" if stateless else self.tree.flat_text(parent_node_id)
            encoded = self._tokenizer.encode(
                prefix + input, return_tensors="pt",
            )
            ids = cast(torch.Tensor, encoded)  # return_tensors="pt" gives Tensor, not list[int]
            return ids.to(self._device) if to_device else ids
        if isinstance(input, str):
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
            messages = prior + [
                {"role": "user", "content": input, "label": user_role}
            ]
        elif isinstance(input, list):
            messages = list(input)
        else:
            raise TypeError(f"Unsupported input type: {type(input)}")
        # Generation-prompt assistant label: a role-augmented steering scope
        # (``_active_role``, transient, set by ``_SteeringContext`` for
        # ``:role-<slug>`` vectors / persona manifolds) wins so steer
        # baseline matches extract baseline; otherwise this send's
        # ``assistant_role`` box drives the to-be-generated turn.  Prior
        # turns' labels ride on the messages themselves (above).
        steer_role = getattr(self, "_active_role", None)
        gen_role = steer_role if steer_role is not None else assistant_role
        model_type_for_role: str | None = None
        any_label = gen_role is not None or any(
            isinstance(m, dict) and m.get("label") for m in messages
        )
        if any_label:
            model_cfg = getattr(self._model, "config", None)
            text_cfg = (
                getattr(model_cfg, "text_config", None)
                if model_cfg is not None else None
            )
            model_type_for_role = (
                getattr(text_cfg, "model_type", None)
                if text_cfg is not None
                else None
            )
            if model_type_for_role is None and model_cfg is not None:
                model_type_for_role = getattr(model_cfg, "model_type", None)
        ids = build_chat_input(
            self._tokenizer, messages, self.config.system_prompt,
            thinking=thinking,
            gen_role=gen_role,
            model_type=model_type_for_role,
        )
        return ids.to(self._device) if to_device else ids

    def build_readings(self) -> dict[str, ProbeReadings]:
        """Per-probe cross-generation :class:`ProbeReadings`, per coordinate axis."""
        readings: dict[str, ProbeReadings] = {}
        if not self._monitor.probe_names:
            return readings
        for name in self._monitor.probe_names:
            stats = self._monitor.get_stats(name)
            count = stats["count"]
            if count == 0:
                continue
            # Per-axis accumulators (axis 0 falls back to the scalar stats
            # for a degenerate probe with no axis record).
            axes = self._monitor.axis_stats(name) or [stats]
            R = len(axes)
            hist = [tuple(float(c) for c in coords)
                    for coords in self._monitor.history.get(name, [])]
            # An empty reading is accumulated as the rank-1 ``(0.0,)`` fallback
            # (monitor ``_apply_accumulate``), while ``axes`` grows to the max
            # rank a probe ever saw — so a probe that yields both an empty and a
            # full reading leaves ``hist`` ragged.  Pad short rows to ``R`` (zeros
            # match the axis-0 empty fallback) so the per-axis delta never
            # over-indexes and ``per_generation`` stays uniform-width.
            if any(len(row) < R for row in hist):
                hist = [row + (0.0,) * (R - len(row)) if len(row) < R else row
                        for row in hist]
            means = tuple(a["sum"] / count for a in axes)
            stds = tuple(
                max(0.0, a["sum_sq"] / count - (a["sum"] / count) ** 2) ** 0.5
                for a in axes
            )
            mins = tuple(
                a["min"] if a["min"] != float("inf") else 0.0 for a in axes
            )
            maxs = tuple(
                a["max"] if a["max"] != float("-inf") else 0.0 for a in axes
            )
            if len(hist) >= 2:
                delta = tuple(
                    sum(abs(hist[i][k] - hist[i - 1][k])
                        for i in range(1, len(hist))) / (len(hist) - 1)
                    for k in range(R)
                )
            else:
                delta = tuple(0.0 for _ in range(R))
            readings[name] = ProbeReadings(
                per_generation=hist, mean=means, std=stds,
                min=mins, max=maxs, delta_per_gen=delta,
            )
        return readings

    def _finalize_generation(
        self, input: Any, generated_ids: list[int], elapsed: float,
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
            self, input, generated_ids, elapsed, vector_snapshot,
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
                             assistant_role: str | None = None):
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
        """Flatten the active steering stack for ``GenerationResult.vectors``."""
        return self._get_steering_composer().snapshot_steering_alphas()

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
    ) -> str | None:
        """Create the loom user/assistant nodes for a stateful generation."""
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
            self._check_user_send_target(parent_node_id)
            user_node_id = self.tree.add_user_turn(
                input, parent_id=parent_node_id, role_label=user_role)
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
        return self.tree.begin_assistant(
            user_node_id, recipe=recipe, role_label=assistant_role)

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
        want_perplexity: bool = True,
        cache_token_text: bool = True,
    ) -> tuple[list[int], float]:
        """Run the decode loop once capture and steering are installed."""
        cached_pkv = None
        cache_position_offset = 0
        cached_static = False
        effective_input_ids = input_ids
        static_cache_eligible = (
            self._static_cache_active
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

        start = time.monotonic()
        composer = self._get_steering_composer()
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
        # Compiled-model routing (MPS).  The graph was traced with ONLY the
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
        # the hooked path.  CUDA keeps its existing graph-capture path untouched.
        gen_model = self._model
        steering_uses_compiled_offsets = bool(
            getattr(self, "_steering_uses_compiled_offsets", False)
        )
        if self._compiled and self._device.type == "mps":
            compiled_clean = not self._capture.is_transient() and (
                steering_uses_compiled_offsets
                or self._steering.all_fast_path()
            )
            if compiled_clean:
                use_static_cache = self._static_cache_active and (
                    cached_pkv is None or cached_static
                )
            else:
                gen_model = getattr(self._model, "_orig_mod", self._model)
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
            past_key_values=cached_pkv,
            cache_position_offset=cache_position_offset,
            score_callback=gating_callback,
            # Per-token probe scoring fires post-forward (FIX F1), not inside the
            # capture hook.  On the persistent-capture path the step callback is
            # ``ingest_persistent`` (accumulate from the persistent buffers +
            # fire the step sink); otherwise ``fire_step_sink`` (the transient
            # hooks already accumulated in-forward).  Both no-op when no per-token
            # sink is installed (aggregate-only / full-retention / no-probe).
            step_callback=(
                self._capture.ingest_persistent
                if self._capture_state.persistent
                else self._capture.fire_step_sink
            ),
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
        steering_cm = None
        try:

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
            _live_scores_on = getattr(self, "_live_probe_scores", True)
            _wants_live_token_scores = bool(
                _live_scores_on
                and on_token is not None
                and getattr(on_token, "_saklas_wants_live_scores", False)
            )
            _persists_layer_scores = bool(
                _live_scores_on
                and (
                    (sampling is not None and sampling.persist_per_layer_scores)
                    or (
                        on_token is not None
                        and getattr(
                            on_token, "_saklas_wants_per_layer_scores", False,
                        )
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
            from saklas.core.token_payloads import build_token_probe_payload

            def _token_tap(text: str, is_thinking: bool, tid: int | None, lp: float | None, top_alts: Any, perplexity: float | None) -> None:
                nonlocal mean_logprob_sum, mean_logprob_count
                self._last_token_probe_payload = None
                if logprobs_list is not None and tid is not None and tid >= 0 and not is_thinking:
                    logprobs_list.append((tid, lp if lp is not None else 0.0, top_alts or []))
                if lp is not None and tid is not None and tid >= 0 and not is_thinking:
                    mean_logprob_sum += lp
                    mean_logprob_count += 1
                needs_scores = bool(
                    _live_scores_on
                    and self._monitor.probe_names
                    and (
                        assistant_node_id is not None
                        or _has_trait_consumer
                        or _wants_live_token_scores
                        or _persists_subspace_coords
                    )
                )
                payload = build_token_probe_payload(
                    monitor=self._monitor,
                    capture=self._capture,
                    capture_state=self._capture_state,
                    incremental_readings=self._incremental_readings,
                    needs_scores=needs_scores,
                    wants_live_token_scores=_wants_live_token_scores,
                    persists_layer_scores=_persists_layer_scores,
                    assistant_node_id=assistant_node_id,
                )
                # Live workspace readout (None when off): the step's top-k
                # lens tokens per selected layer + the layer-aggregated
                # chip list.
                lens_step = self._live_lens_readout_step()
                # Pinned lens probes tick on the same step (readings extracted
                # from the display logits inside the readout step) — merge them
                # into every populated probe channel so the loom row, trait
                # stream, and WS frames carry them uniformly.
                if lens_step is not None and self._last_lens_step_readings:
                    payload.merge_readings(
                        self._last_lens_step_readings,
                        per_layer=(
                            assistant_node_id is not None
                            and _persists_layer_scores
                        ),
                        live=_wants_live_token_scores,
                    )
                scores = payload.scores
                per_layer_payload = payload.per_layer_scores
                self._last_token_probe_payload = payload.to_token_payload(
                    lens=lens_step[0] if lens_step is not None else None,
                    lens_aggregate=(
                        lens_step[1] if lens_step is not None else None
                    ),
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
                    self.tree.append_token(
                        assistant_node_id,
                        token_row,
                        thinking=bool(is_thinking),
                    )
                if on_token is not None:
                    on_token(text, is_thinking, tid, lp, top_alts, perplexity)
                # Inline per-token scoring for live SSE trait subscribers.
                if self._trait_queues and self._monitor.probe_names and scores:
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
                and self._monitor.probe_names
            )
            # The tap also writes per-token ``probes`` / ``per_layer_scores``
            # onto the loom row when probes are loaded and the gen is loom-
            # attached — required so a webui refresh can rehydrate highlight
            # tints and the token-drilldown heatmap from the server tree.
            _persists_probe_row = bool(
                _live_scores_on
                and return_probe_readings
                and not stateless
                and self._monitor.probe_names
            )
            _need_tap = (
                on_token is not None
                or logprobs_list is not None
                or _has_trait_consumer
                or _persists_probe_row
                or stop_list is not None
            )
            _has_lens_consumer = bool(
                getattr(self, "_live_lens", None) is not None
                and on_token is not None
                and getattr(on_token, "_saklas_wants_lens_readout", False)
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
            # MPS graph + static cache are live, the persistent capture
            # buffers were adopted, and the caller didn't ask for the full per-step
            # hidden stack (``return_hidden`` keeps the transient full-retention
            # capture).  Capture then rides the persistent buffers; steering rides the
            # offsets — both compile-clean.
            self._compiled_clean_eligible = bool(
                getattr(self, "_compiled", False)
                and self._device.type == "mps"
                and self._static_cache_active
                and self._capture_buffers
                and not (sampling and sampling.return_hidden)
            )

            if steering_obj is not None and steering_obj.alphas:
                steering_cm = self.steering(steering_obj)

            vector_snapshot: dict[str, float] = (
                self._snapshot_steering_alphas()
                if self._steering_stack or steering_cm is not None
                else {}
            )

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
            chat_history_anchor = parent_node_id
            if (
                not stateless
                and isinstance(input, str)
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
            )
            # Refresh snapshot now that steering is pushed (first-scope case).
            vector_snapshot = self._snapshot_steering_alphas()

            want_hidden = bool(sampling and sampling.return_hidden)
            # Per-token scoring is only needed when something consumes a
            # per-token reading: a probe gate, a loom token row, an SSE trait
            # stream, a live-scores client, or a per-layer-heatmap persist.
            # Otherwise (probes attached but only the aggregate wanted, e.g. a
            # stateless server gen) the capture skips per-token scoring entirely
            # and pools the aggregate once at finalize.
            composer = self._get_steering_composer()
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
            # finalize), which is the common TUI/loom monitoring path.
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
                and self._monitor.probe_names
            )
            self._live_lens_active_for_generation = _has_lens_consumer
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
                    lean_per_token=lean_per_token,
                    final_probe_aggregate=return_probe_readings,
                    live_lens_active=_has_lens_consumer,
                )
                self._monitor.begin_live()
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
                    forced_prefix=forced_prefix,
                    # Per-token perplexity costs one host sync/token.  Compute
                    # it only when something surfaces it: a loom-attached gen
                    # persists it on the token row, an interactive (non-
                    # stateless) gen may show it, or a caller opts its on_token
                    # in.  Stateless server streaming (OpenAI/Ollama deltas, the
                    # native WS tap) never reads per-token ppl, so it skips the
                    # sync.  Callers can force it via ``on_token
                    # ._saklas_wants_perplexity = True``.
                    want_perplexity=(
                        not stateless
                        or assistant_node_id is not None
                        or (
                            on_token is not None
                            and getattr(
                                on_token, "_saklas_wants_perplexity", False,
                            )
                        )
                    ),
                    cache_token_text=_tap_has_text_consumer,
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
                return_probe_readings=return_probe_readings,
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
                # ``swallow=True``: we're already re-raising the original
                # failure below, so a teardown ``Exception`` must not mask
                # it (see ``_exit_internal_steering``).
                self._exit_internal_steering(steering_cm, swallow=True)
            raise
        finally:
            # Defense-in-depth: even if the inner finally never ran (e.g. a
            # BaseException between the outer try entry and ``begin_capture``),
            # any hooks that did get attached must come off.  Idempotent.
            self._end_capture()
            self._monitor.end_live()
            # Probe-inspector subspace-coords post-pass is per-generation; clear
            # it so it never leaks into a later gen that didn't opt in.
            self._monitor.set_subspace_coords(False)
            # Release the loom-tree reservation in the same scope as the
            # gen-lock release.  Even if finalize raised, mutators (edit /
            # delete on this subtree) need to be free again now that the
            # streaming target is no longer live.
            self._active_gen_reservation = None
            self._last_token_probe_payload = None
            self._live_lens_active_for_generation = True
            # Reset capture state to the default (FULL, non-persistent) so the
            # next gen starts clean (finalize has already consumed the rows by
            # now). Belt-and-suspenders: ``_begin_capture`` resets it at gen start
            # too.
            self._capture_state = CaptureState()
            self._compiled_clean_eligible = False
            self._incremental_readings = []
            self._incremental_gate_scores = []
            # Zero the persistent compile-clean steering offsets so a static-
            # affine push can't leak into a later generation that takes the
            # eager / unsteered path without re-running ``_install_composed_
            # steering`` (unsteered gens have no steering scope to reset them).
            self._steering_uses_compiled_offsets = False
            if self._steering.has_compiled_offsets():
                self._steering.zero_compiled_offsets()
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

        fast_fan = self._generate_fan_fast(
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
        on_token: Callable[..., None] | None,
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
        q: queue.SimpleQueue[Any] = queue.SimpleQueue()
        done = object()
        result_holder: list[GenerationResult] = []
        exc_holder: list[BaseException] = []
        idx_counter = [0]

        def _push(text: str, is_thinking: bool, tid: int | None, lp: float | None, top_alts: Any, perplexity: float | None) -> None:
            payload = self._last_token_probe_payload or {}
            probe_readings: dict[str, "ProbeReading"] | None = None
            # Monitor probes AND pinned lens probes (readout channel) both
            # land in the payload's merged ``readings`` — a lens-only roster
            # still carries per-token readings while the live lens is on.
            if live_scores and (
                self._monitor.probe_names or getattr(self, "_lens_probes", None)
            ):
                raw_readings = payload.get("readings")
                if isinstance(raw_readings, dict) and raw_readings:
                    probe_readings = raw_readings
                    # ``update_live`` folds coordinate axis 0 of each reading
                    # into the running mean; ``TokenEvent.probe_readings`` carries
                    # the readings dict verbatim (the live-stream consumer reads
                    # ``fraction`` / ``nearest`` / ``coords`` off each).
                    # ``TokenEvent.scores`` is a back-compat property alias for
                    # ``probe_readings`` — no second field to populate.
                    self._monitor.update_live(probe_readings)
            event = TokenEvent(
                text=text, token_id=tid if tid is not None else -1, index=idx_counter[0],
                thinking=is_thinking, logprob=lp, top_alts=top_alts,
                probe_readings=probe_readings, perplexity=perplexity,
                lens_readout=payload.get("lens"),
                lens_aggregate=payload.get("lens_aggregate"),
            )
            idx_counter[0] += 1
            q.put(event)
        _push_flags: Any = _push
        _push_flags._saklas_wants_live_scores = bool(live_scores)
        _push_flags._saklas_wants_lens_readout = True

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
        if not self._batch_fast_runtime_available():
            return None
        if self._batch_fast_sampling_blocked(sampling):
            return None
        if getattr(self, "_trait_queues", None):
            return None
        has_probes = bool(getattr(getattr(self, "_monitor", None), "probe_names", []))
        if has_probes and not (
            hasattr(self, "_layers")
            and hasattr(self, "_capture")
            and callable(getattr(self._monitor, "probe_layers", None))
        ):
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

    @staticmethod
    def _stateless_readings_from_probe_aggregate(
        agg_vals: dict[str, ProbeReading],
    ) -> dict[str, ProbeReadings]:
        """Stateless ``GenerationResult.readings`` from one aggregate row."""
        readings: dict[str, ProbeReadings] = {}
        for name, reading in agg_vals.items():
            coords = reading.coords or (0.0,)
            zeros = tuple(0.0 for _ in coords)
            readings[name] = ProbeReadings(
                per_generation=[coords],
                mean=coords,
                std=zeros,
                min=coords,
                max=coords,
                delta_per_gen=zeros,
            )
        return readings

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
        readings = self._stateless_readings_from_probe_aggregate(probe_readings)
        result = GenerationResult(
            text=text,
            tokens=list(generated_ids),
            token_count=token_count,
            tok_per_sec=tok_per_sec,
            elapsed=elapsed,
            readings=readings,
            vectors=vector_snapshot,
            prompt_tokens=prompt_tokens,
            finish_reason=finish_reason,
            logprobs=None,
            applied_steering=applied_steering,
            hidden_states=None,
            probe_readings=probe_readings,
        )
        self._last_result = result
        self._last_per_token_scores = None
        if readings:
            scalar_readings = {
                name: (reading.mean[0] if reading.mean else 0.0)
                for name, reading in readings.items()
            }
            self.events.emit(ProbeScored(readings=scalar_readings))
        return result

    def _batch_probe_aggregate_for_row(
        self,
        row_index: int,
        generated_ids: list[int],
        probe_names: list[str],
    ) -> dict[str, ProbeReading]:
        """Score the final aggregate probe reading for one batched row."""
        empty = self._empty_readings(probe_names)
        if not generated_ids:
            return empty
        from saklas.core.vectors import last_content_index
        agg_fwd = last_content_index(generated_ids, self._tokenizer)
        pooled = self._capture.batch_tail_slice_at(row_index, agg_fwd)
        if not pooled:
            return empty
        self._monitor.enable_curved_warm(False)
        agg_vals = self._monitor.score_aggregate(pooled)
        return {name: agg_vals.get(name, empty[name]) for name in probe_names}

    def _batch_fast_runtime_available(self) -> bool:
        """Return True when a real session has the pieces the fast path uses."""
        required = (
            "_model",
            "_tokenizer",
            "_device",
            "_gen_lock",
            "_gen_phase",
            "_gen_state",
            "_monitor",
            "_profiles",
            "_steering",
            "_default_return_top_k",
            "config",
            "events",
        )
        return all(hasattr(self, name) for name in required)

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
                if self._steering_stack or steering_cm is not None
                else {}
            )
            pad_id = self._batch_pad_token_id()
            input_ids, attention_mask, prompt_lengths = self._prepare_batch_input_ids(
                prompts,
                sampling=sampling,
                raw=raw,
                pad_id=pad_id,
            )
            probe_names = list(getattr(self._monitor, "probe_names", []))
            if probe_names:
                layer_idxs = sorted(self._monitor.probe_layers())
                self._capture.clear()
                if layer_idxs:
                    self._capture.attach_batch_tail(
                        self._layers,
                        layer_idxs,
                        depth=_AGG_TAIL_DEPTH,
                    )
                self._capture_state = CaptureState(mode=CaptureMode.AGGREGATE_ONLY)
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
            if probe_names:
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
                if probe_names:
                    probe_readings = self._batch_probe_aggregate_for_row(
                        idx,
                        generated_ids,
                        probe_names,
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
                        prompt,
                        generated_ids,
                        elapsed,
                        vector_snapshot,
                        prompt_tokens=prompt_lengths[idx],
                        stateless=stateless,
                        logprobs_list=None,
                        applied_steering=applied_steering,
                        return_hidden=False,
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
                if steering_cm is not None:
                    self._exit_internal_steering(steering_cm, swallow=failed)
            finally:
                if hasattr(self, "_capture"):
                    self._end_capture()
                self._active_gen_reservation = None
                self._last_token_probe_payload = None
                self._capture_state = CaptureState()
                self._compiled_clean_eligible = False
                self._incremental_readings = []
                self._incremental_gate_scores = []
                self._steering_uses_compiled_offsets = False
                if self._steering.has_compiled_offsets():
                    self._steering.zero_compiled_offsets()
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
        :data:`_PREFIX_CACHE_MIN_TOKENS`.  Best-effort: any failure to render or
        cache falls back to the uncached path without disturbing the batch.
        """
        if len(prompts) < 2 or not stateless:
            return False
        if thinking:
            return False
        if sampling is not None and getattr(sampling, "return_hidden", False):
            return False
        if not self._steering_value_prefill_inactive(steering):
            return False
        try:
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
                        getattr(self, "_static_cache_active", False)
                        and steering is None
                    ),
                )
                >= _PREFIX_CACHE_MIN_TOKENS
            )
        except Exception as exc:
            # Prefix caching is a pure optimization: a cache_prefix guard
            # (active scope / in-flight gen), a render failure, or an
            # under-initialized session must fall back to the normal per-row
            # prefill, never break the batch.  ``cache_prefix`` clears its own
            # state before setting it, so no half-built cache can leak.
            _log.debug("generate_batch: skipping prefix cache (%s)", exc)
            return False

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
        if not self._batch_fast_runtime_available():
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

        fast_sweep_fn = getattr(self, "_generate_sweep_fast", None)
        if callable(fast_sweep_fn):
            fast_sweep = fast_sweep_fn(
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
                return cast(RunSet, fast_sweep)

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
