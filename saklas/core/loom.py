"""LoomTree — engine-side tree of conversation nodes.

Replaces the v2.2 flat ``session._history: list[dict]`` with a tree where
nodes are conversation turns and children are alternative continuations.
The active path is what the model sees as context for the next gen; the
rest of the tree is preserved as dead branches for navigation.

Architectural note: this module owns the data model only.  Generation
integration, gen-lock concurrency, persistence to ``~/.saklas/sessions/``,
and HTTP/WS event delivery live in ``saklas/core/session.py`` and the
server layer.  Tree mutators raise :class:`MutationDuringGenerationError`
when a conflict is detected — the session is responsible for calling
:meth:`LoomTree._assert_no_conflict` *via the session's own conflict
checker* before invoking them, because the gen-lock state lives on the
session, not the tree.

The five primitives (edit, branch, navigate, delete_subtree, plus
``regenerate`` exposed via ``session.generate(parent_node_id=..., n=...)``)
are all that exist at the engine level.  Surface-level verbs (slash
commands, keyboard shortcuts, context menus) compose from these.

Per-node token blobs (``tokens`` and ``thinking_tokens``) are held in
memory during streaming.  ``LoomTree.to_dict`` omits them by default so
routine wire payloads stay small; ``LoomTree.save`` writes them to a
compressed sidecar next to the main tree JSON so explicit save/load keeps
the analysis surface intact.  The tree is otherwise in-memory only —
there is no automatic cross-session persistence.
"""

from __future__ import annotations

import gzip
import json
import os
import secrets
import tempfile
import threading
import time
from dataclasses import dataclass, field, fields
from typing import Any, Callable, Iterator, Literal

from saklas.core.errors import SaklasError
from saklas.core.sampling import SamplingConfig

# ---------------------------------------------------------------------------
# ulid — tiny inline implementation
# ---------------------------------------------------------------------------
# Crockford base32; sortable by timestamp prefix.  No external dep — ~30
# lines is cheaper than pulling python-ulid for the only thing we need
# from it.  Format: 10 chars (48-bit ms timestamp) + 16 chars (80-bit
# randomness) = 26 chars total.  Lexicographic order matches chronological
# order on the timestamp prefix; ties broken by the random tail.

_ULID_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def _ulid() -> str:
    """Return a fresh 26-char ULID."""
    ts_ms = int(time.time() * 1000) & ((1 << 48) - 1)
    rand = secrets.randbits(80)
    n = (ts_ms << 80) | rand
    out = []
    for _ in range(26):
        out.append(_ULID_ALPHABET[n & 0x1F])
        n >>= 5
    return "".join(reversed(out))


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class LoomTreeError(SaklasError):
    """Base class for tree-mutation errors."""

    def user_message(self) -> tuple[int, str]:
        return (400, str(self) or self.__class__.__name__)


class UnknownNodeError(KeyError, LoomTreeError):
    """Raised when a node id is not in the tree."""

    def user_message(self) -> tuple[int, str]:
        msg = self.args[0] if self.args else self.__class__.__name__
        return (404, str(msg))


class InvalidNodeOperationError(ValueError, LoomTreeError):
    """Raised when an op is semantically invalid (e.g. deleting an ancestor of active)."""

    def user_message(self) -> tuple[int, str]:
        return (400, str(self) or self.__class__.__name__)


class MutationDuringGenerationError(RuntimeError, LoomTreeError):
    """Raised when a tree mutation conflicts with an in-flight generation.

    The ``_gen_lock`` holder owns the subtree rooted at the user-parent of
    its target.  Decoration ops and branches are always free; edits and
    deletes on that reservation refuse with this error (HTTP layer maps
    to 409).
    """

    def user_message(self) -> tuple[int, str]:
        return (409, str(self) or self.__class__.__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


# Token-score dict shape mirrors what ``generate`` emits via the on_token
# callback today.  Kept as ``dict[str, Any]`` rather than a typed alias
# because the surfaces also tack on per-probe scores and other extras.
TokenScoreDict = dict


@dataclass
class Recipe:
    """Reproducibility receipt for an assistant node.

    Captures everything needed to replay the same generation: steering
    expression, sampling parameters, thinking mode, RNG seed, and the
    probe set live at gen time (with per-probe content hashes so transcript
    replay can detect probe drift).
    """

    steering: str | None = None
    sampling: SamplingConfig | None = None
    thinking: bool | None = None
    seed: int | None = None
    # Names of probes active during the gen, plus their content hashes
    # (sha256 of baked tensor bytes).  ``probe_hashes`` is empty when
    # the session has no probes, or when the surfaces couldn't compute
    # hashes (e.g. for synthetic test probes).
    probes: list[str] = field(default_factory=list)
    probe_hashes: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        sampling = None
        if self.sampling is not None:
            sampling = {}
            for f in fields(SamplingConfig):
                k = f.name
                v = getattr(self.sampling, k)
                if k == "stop" and v is not None:
                    v = list(v)
                elif k == "logit_bias" and v is not None:
                    v = {str(int(tid)): float(bias) for tid, bias in v.items()}
                sampling[k] = v
        return {
            "steering": self.steering,
            "sampling": sampling,
            "thinking": self.thinking,
            "seed": self.seed,
            "probes": list(self.probes),
            "probe_hashes": dict(self.probe_hashes),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Recipe":
        sampling = None
        s = data.get("sampling")
        if s is not None:
            s = dict(s)
            if s.get("logit_bias") is not None:
                s["logit_bias"] = {
                    int(tid): float(bias)
                    for tid, bias in dict(s["logit_bias"]).items()
                }
            sampling = SamplingConfig(**{k: v for k, v in s.items() if v is not None})
        return cls(
            steering=data.get("steering"),
            sampling=sampling,
            thinking=data.get("thinking"),
            seed=data.get("seed"),
            probes=list(data.get("probes", [])),
            probe_hashes=dict(data.get("probe_hashes", {})),
        )

    # ------------------------------------------------------------------
    # Overlay / modifier helpers (v2.3 phase 5)
    # ------------------------------------------------------------------

    def overlay(self, override: "Recipe | None") -> "Recipe":
        """Return a new Recipe with ``override``'s non-None fields applied.

        ``None`` fields on ``override`` fall through to ``self``.  The
        recipe-override mechanism for auto-regen and manual modified
        regen is a thin overlay over this: built-in modes
        (``unsteered``/``inverted``/``reseed``/``cool``/``hot``) produce
        a partial Recipe; ``Recipe.overlay`` composes it onto whatever
        the parent's recipe was.

        ``probes`` / ``probe_hashes`` are non-overrideable here — they
        track the registered probe set at gen time, not a user choice.
        The session re-stamps them on the regen.
        """
        if override is None:
            return self
        return Recipe(
            steering=(
                override.steering if override.steering is not None else self.steering
            ),
            sampling=(
                override.sampling if override.sampling is not None else self.sampling
            ),
            thinking=(
                override.thinking if override.thinking is not None else self.thinking
            ),
            seed=override.seed if override.seed is not None else self.seed,
            probes=list(self.probes),
            probe_hashes=dict(self.probe_hashes),
        )

    def invert_steering(self) -> "Recipe":
        """Return a Recipe with every steering term's α sign flipped.

        Used by the ``inverted`` auto-regen mode.  The grammar carries
        triggers / projections / ablations through; only the numeric
        coefficient changes.  When this recipe has no steering, returns
        an empty-steering recipe so the caller can compose it onto the
        parent (``Recipe.overlay``) and see the no-op.
        """
        from saklas.core.steering_expr import (
            AblationTerm, ProjectedTerm, parse_expr, format_expr,
        )
        from saklas.core.steering import Steering

        if not self.steering:
            return Recipe(steering="")
        parsed = parse_expr(self.steering)
        flipped: dict[str, Any] = {}
        for name, val in parsed.alphas.items():
            if isinstance(val, ProjectedTerm):
                flipped[name] = ProjectedTerm(
                    coeff=-val.coeff,
                    trigger=val.trigger,
                    operator=val.operator,
                    base=val.base,
                    onto=val.onto,
                )
                continue
            if isinstance(val, AblationTerm):
                flipped[name] = AblationTerm(
                    coeff=-val.coeff,
                    trigger=val.trigger,
                    target=val.target,
                )
                continue
            if isinstance(val, tuple):
                flipped[name] = (-float(val[0]), val[1])
                continue
            flipped[name] = -float(val)
        new = Steering(
            alphas=flipped,
            thinking=parsed.thinking,
            trigger=parsed.trigger,
            injection_mode=parsed.injection_mode,
            theta_max=parsed.theta_max,
            projection_metric=parsed.projection_metric,
        )
        return Recipe(steering=format_expr(new))

    def compose_modifier(self, mode: "str | Recipe") -> "Recipe":
        """Return a partial Recipe for the named auto-regen mode.

        Recognized string modes:

        - ``"unsteered"``: steering wiped (empty expression).
        - ``"inverted"``: flips every term's α sign — see
          :meth:`invert_steering`.
        - ``"reseed"``: fresh entropy seed; everything else inherits.
        - ``"cool"``: temperature 0.3; everything else inherits.
        - ``"hot"``: temperature 1.2; everything else inherits.

        A :class:`Recipe` instance passes through unchanged — that's the
        ``custom`` path: callers (e.g. the TUI's ``/auto-regen custom:
        <expr>``) parse the user's partial-recipe expression themselves
        and hand the resulting Recipe in directly.  Unknown string modes
        raise ``ValueError``.
        """
        if isinstance(mode, Recipe):
            return mode
        if mode == "unsteered":
            return Recipe(steering="")
        if mode == "inverted":
            return self.invert_steering()
        if mode == "reseed":
            return Recipe(seed=secrets.randbits(31))
        if mode == "cool":
            return Recipe(sampling=SamplingConfig(temperature=0.3))
        if mode == "hot":
            return Recipe(sampling=SamplingConfig(temperature=1.2))
        raise ValueError(
            f"unknown recipe-override mode {mode!r}; valid: unsteered, "
            f"inverted, reseed, cool, hot, or pass a Recipe directly"
        )

    def _fill_probe_hashes(self, session: Any) -> "Recipe":
        """Return a copy of this Recipe with ``probe_hashes`` populated.

        Looks up each registered probe's sha256 via
        ``session._probe_hash(name)`` (cached on the session).  Drops
        names the session doesn't carry — transcripts always reflect
        the session's view at recipe stamp time.  Phase 5 wires this
        into the gen path so :attr:`probe_hashes` rides every assistant
        node's recipe automatically.
        """
        out: dict[str, str] = {}
        for name in self.probes:
            digest = session._probe_hash(name)
            if digest is not None:
                out[name] = digest
        return Recipe(
            steering=self.steering,
            sampling=self.sampling,
            thinking=self.thinking,
            seed=self.seed,
            probes=list(self.probes),
            probe_hashes=out,
        )


Role = Literal["user", "assistant", "system"]


@dataclass
class LoomNode:
    """A single node in the loom tree.

    Token lists (``tokens``, ``thinking_tokens``) are owned here for live
    streaming but are persisted in side files by the session store; the
    tree's ``to_dict()`` omits them so the main tree file stays small.

    ``edit_count`` and ``edited_at`` flag assistant nodes whose text has
    been mutated in place since the model emitted it — downstream
    consumers (transcript replay, comparison views) use this to know
    that the text isn't pristine from the model.
    """

    id: str
    parent_id: str | None
    role: Role
    text: str = ""
    tokens: list[TokenScoreDict] | None = None
    thinking_tokens: list[TokenScoreDict] | None = None
    recipe: Recipe | None = None
    aggregate_readings: dict[str, float] = field(default_factory=dict)
    applied_steering: str | None = None
    finish_reason: str | None = None
    starred: bool = False
    notes: str = ""
    created_at: float = field(default_factory=time.time)
    edited_at: float | None = None
    edit_count: int = 0
    # Mean chosen-token logprob over the non-thinking response span,
    # computed in :meth:`SaklasSession._generate_core` and stamped at
    # :meth:`LoomTree.finalize_assistant` time. ``None`` for legacy
    # nodes replayed from pre-logit-pass transcripts. ``mean_surprise``
    # caches ``-mean_logprob`` so the sidebar's "sort by surprise" mode
    # is a keyboard sort, not a recompute. Both fields surface through
    # ``to_dict`` / ``from_dict`` for tree persistence + WS bridge.
    mean_logprob: float | None = None
    mean_surprise: float | None = None
    # Raw decode-step token ids for an assistant node — the exact
    # sequence the engine sampled, *including* the structural delimiters
    # (`<think>` / `</think>` / channel markers) that ``tokens`` /
    # ``thinking_tokens`` suppress, and with partial-UTF-8 bytes
    # unmerged.  Stamped at :meth:`LoomTree.finalize_assistant` from the
    # engine's ``generated_ids``.  This is the forceable prefix a logit
    # fork replays; ``None`` for legacy / transcript-loaded nodes, where
    # the fork affordance falls back to disabled.  Persisted in the token
    # sidecar alongside ``tokens`` (bulky, per-decode-step granularity).
    raw_token_ids: list[int] | None = None

    def to_dict(self, *, include_tokens: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "parent_id": self.parent_id,
            "role": self.role,
            "text": self.text,
            "aggregate_readings": dict(self.aggregate_readings),
            "applied_steering": self.applied_steering,
            "finish_reason": self.finish_reason,
            "starred": self.starred,
            "notes": self.notes,
            "created_at": self.created_at,
            "edited_at": self.edited_at,
            "edit_count": self.edit_count,
            "mean_logprob": self.mean_logprob,
            "mean_surprise": self.mean_surprise,
        }
        if self.recipe is not None:
            out["recipe"] = self.recipe.to_dict()
        if include_tokens:
            out["tokens"] = self.tokens
            out["thinking_tokens"] = self.thinking_tokens
            out["raw_token_ids"] = self.raw_token_ids
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LoomNode":
        recipe = None
        if data.get("recipe") is not None:
            recipe = Recipe.from_dict(data["recipe"])
        return cls(
            id=data["id"],
            parent_id=data.get("parent_id"),
            role=data["role"],
            text=data.get("text", ""),
            tokens=data.get("tokens"),
            thinking_tokens=data.get("thinking_tokens"),
            recipe=recipe,
            aggregate_readings=dict(data.get("aggregate_readings", {})),
            applied_steering=data.get("applied_steering"),
            finish_reason=data.get("finish_reason"),
            starred=bool(data.get("starred", False)),
            notes=data.get("notes", ""),
            created_at=float(data.get("created_at", time.time())),
            edited_at=data.get("edited_at"),
            edit_count=int(data.get("edit_count", 0)),
            mean_logprob=data.get("mean_logprob"),
            mean_surprise=data.get("mean_surprise"),
            raw_token_ids=data.get("raw_token_ids"),
        )


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoomMutated:
    """Tree-mutation event for ``session.events``.

    ``op`` is one of ``"edit"``, ``"branch"``, ``"navigate"``, ``"delete"``,
    ``"star"``, ``"note"``, ``"reset"``, ``"add_user"``, ``"begin_assistant"``,
    ``"finalize_assistant"``.

    Delta payload fields carry ids only at the engine level — the
    server WS layer (:mod:`saklas.server.saklas_api`'s
    ``WS /saklas/v1/sessions/{id}/stream``) enriches each id into full
    ``LoomNodeJSON`` payloads via :func:`_node_json` before forwarding
    to clients, so the wire-level ``tree_mutated`` event matches the
    shape described in ``docs/plans/loom.md`` phase 2.  In-process
    subscribers (TUI, library callers) that already hold the tree
    look the ids up themselves; the network hop is the only place that
    needs the inlined node data.

    Field semantics:

    - ``added``: newly-created node ids
    - ``removed``: dropped node ids (delete_subtree)
    - ``updated``: ids of nodes whose fields changed in place (edit/star/note)
    - ``active_node_id``: present when the active node moves
    - ``rev``: monotonic tree revision after the mutation
    """

    op: str
    rev: int
    added: tuple[str, ...] = ()
    removed: tuple[str, ...] = ()
    updated: tuple[str, ...] = ()
    active_node_id: str | None = None


# ---------------------------------------------------------------------------
# Seed schedule
# ---------------------------------------------------------------------------


def _mix_seed(base: int, i: int) -> int:
    """Deterministically derive a 31-bit seed from ``(base, i)``.

    Uses blake2b with an 8-byte digest as the avalanche function;
    cross-machine deterministic and well-mixed for small ``i`` (which
    a naive FNV-1a over little-endian bytes is not — the high bytes
    of small integers are zero, so consecutive ``i`` values differ in
    only one input byte and the FNV-1a output is nearly linear).
    """
    import hashlib
    import struct

    payload = struct.pack("<qq", int(base) & 0x7FFFFFFFFFFFFFFF, int(i))
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    val = struct.unpack("<Q", digest)[0]
    return val & 0x7FFFFFFF


def derive_seed_schedule(base_seed: int | None, n: int) -> list[int]:
    """Return ``n`` deterministic seeds derived from ``base_seed``.

    When ``base_seed`` is None, resolves an entropy-derived base (and the
    caller persists the resolved base in the first sibling's Recipe so
    the run remains reproducible after the fact).  ``n=1`` returns the
    base seed verbatim so single regen with no schedule looks unchanged.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if base_seed is None:
        # ``secrets.randbits`` gives us a 31-bit seed in a torch-friendly range.
        base_seed = secrets.randbits(31)
    if n == 1:
        return [int(base_seed) & 0x7FFFFFFF]
    return [_mix_seed(int(base_seed), i) for i in range(n)]


# ---------------------------------------------------------------------------
# LoomTree
# ---------------------------------------------------------------------------


# Format version for ``tree.json``.  Bumps follow the pack-format pattern
# in ``saklas/io/packs.py``: loader raises on a future version it doesn't
# understand, and writes always emit the current version.
TREE_FORMAT_VERSION = 1


# Hook signature used by the session to enforce gen-lock conflicts.  The
# tree calls this before any mutator; the session implementation maps
# the (node_id, op) pair to either a no-op (free op) or a raise
# (MutationDuringGenerationError).  Default is a no-op for unit-test
# usage of the tree without a session.
ConflictChecker = Callable[[str, str], None]


def _noop_conflict_check(node_id: str, op: str) -> None:
    del node_id, op  # signature is the protocol; no work to do here
    return None


class LoomTree:
    """Mutation-safe tree of conversation nodes.

    Owned by :class:`saklas.core.session.SaklasSession`; callers go through
    session methods so locking + event emission happen in one place.  The
    tree's own methods are thread-safe under an internal ``RLock`` so the
    session can call them from arbitrary threads (gen worker, server
    handler, TUI poller).

    Operations that mutate bump :attr:`rev` and emit a :class:`LoomMutated`
    event through the session's :class:`EventBus`.  Surfaces track the
    last-seen ``rev`` and full-refetch the tree if they detect a gap.
    """

    def __init__(
        self,
        *,
        events: Any | None = None,
        model_id: str | None = None,
        session_id: str | None = None,
        name: str | None = None,
        conflict_check: ConflictChecker | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._events = events  # EventBus-like; emit(LoomMutated) when set
        self.model_id: str | None = model_id
        self.session_id: str | None = session_id
        self.name: str | None = name
        self._conflict_check: ConflictChecker = conflict_check or _noop_conflict_check

        self.nodes: dict[str, LoomNode] = {}
        self.children_of: dict[str, list[str]] = {}
        self.rev: int = 0

        # Synthetic root: role="system", text empty, no parent.  First user
        # turn is its child.  This keeps the tree structure uniform (every
        # real node has a parent) without burning a special-case branch
        # in the active-path walker.
        root = LoomNode(id=_ulid(), parent_id=None, role="system")
        self.nodes[root.id] = root
        self.children_of[root.id] = []
        self.root_id: str = root.id
        self.active_node_id: str = root.id

    # ------------------------------------------------------------------
    # Conflict-check wiring
    # ------------------------------------------------------------------

    def set_conflict_check(self, fn: ConflictChecker | None) -> None:
        """Install a conflict-check hook.

        The session sets this in its ``__init__`` so all mutator paths
        consult ``session._loom_conflict_check`` before proceeding.
        """
        self._conflict_check = fn or _noop_conflict_check

    def attach_events(self, events: Any | None) -> None:
        """Wire (or unwire) the EventBus that mutations emit on."""
        self._events = events

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, node_id: str) -> LoomNode:
        try:
            return self.nodes[node_id]
        except KeyError:
            raise UnknownNodeError(node_id) from None

    def has(self, node_id: str) -> bool:
        return node_id in self.nodes

    def children(self, node_id: str) -> list[LoomNode]:
        with self._lock:
            return [self.nodes[c] for c in self.children_of.get(node_id, [])]

    def child_ids(self, node_id: str) -> list[str]:
        with self._lock:
            return list(self.children_of.get(node_id, []))

    def descendants(self, node_id: str) -> Iterator[LoomNode]:
        """Depth-first iteration over the descendants of ``node_id``."""
        with self._lock:
            stack = list(self.children_of.get(node_id, []))
        # We snapshot the child list under lock, then walk it lock-free.
        # Concurrent mutations of the subtree mid-iteration aren't supported
        # — callers wanting that need to snapshot first.
        while stack:
            cur = stack.pop()
            node = self.nodes.get(cur)
            if node is None:
                continue
            yield node
            stack.extend(self.children_of.get(cur, []))

    def path_to(self, node_id: str) -> list[LoomNode]:
        """Return the path from the root to ``node_id``, inclusive."""
        with self._lock:
            if node_id not in self.nodes:
                raise UnknownNodeError(node_id)
            chain: list[LoomNode] = []
            cur: str | None = node_id
            while cur is not None:
                chain.append(self.nodes[cur])
                cur = self.nodes[cur].parent_id
            chain.reverse()
            return chain

    def active_path(self) -> list[LoomNode]:
        """Path from the root to :attr:`active_node_id`."""
        return self.path_to(self.active_node_id)

    def messages_for(
        self,
        leaf_id: str | None = None,
        *,
        include_system: bool = False,
    ) -> list[dict[str, str]]:
        """Return the active-path (or path to ``leaf_id``) as chat messages.

        Returns the v2 ``[{"role": ..., "content": ...}, ...]`` shape that
        the rest of the engine + servers consume.  Skips the synthetic
        root by default (its empty text isn't a real system prompt) —
        callers wanting an explicit system prompt prepend it themselves.
        """
        target = leaf_id if leaf_id is not None else self.active_node_id
        path = self.path_to(target)
        out: list[dict[str, str]] = []
        for node in path:
            if node.id == self.root_id and not include_system:
                continue
            out.append({"role": node.role, "content": node.text})
        return out

    def ancestors_of(self, node_id: str) -> Iterator[str]:
        """Yield ``node_id``'s ancestor ids (parent first, root last)."""
        cur = self.nodes[node_id].parent_id
        while cur is not None:
            yield cur
            cur = self.nodes[cur].parent_id

    def is_ancestor_of(self, ancestor_id: str, descendant_id: str) -> bool:
        return ancestor_id in set(self.ancestors_of(descendant_id))

    # ------------------------------------------------------------------
    # Internal mutator scaffolding
    # ------------------------------------------------------------------

    def _emit(self, event: LoomMutated) -> None:
        if self._events is not None:
            try:
                self._events.emit(event)
            except Exception:
                # Event delivery must not break a mutation.  ``EventBus``
                # itself swallows subscriber errors; this guard catches
                # anything from a non-EventBus shim.
                pass

    def _add_child(self, parent_id: str, node: LoomNode) -> None:
        self.nodes[node.id] = node
        self.children_of.setdefault(node.id, [])
        self.children_of.setdefault(parent_id, []).append(node.id)

    # ------------------------------------------------------------------
    # Streaming/generation entry points
    # ------------------------------------------------------------------

    def add_user_turn(
        self,
        text: str,
        parent_id: str | None = None,
        *,
        dedup_existing: bool = True,
    ) -> str:
        """Add a user turn under ``parent_id`` (default: the active node).

        If ``dedup_existing`` is set and ``parent_id`` already has a user-turn
        child with the exact same text, returns that existing child's id
        without growing the tree.  This spares users a redundant tree level
        for the regen workflow where the user re-sends the same prompt.
        """
        with self._lock:
            parent = parent_id if parent_id is not None else self.active_node_id
            if parent not in self.nodes:
                raise UnknownNodeError(parent)
            self._conflict_check(parent, "add_user_turn")
            if dedup_existing:
                for cid in self.children_of.get(parent, []):
                    sib = self.nodes[cid]
                    if sib.role == "user" and sib.text == text:
                        self.active_node_id = sib.id
                        self.rev += 1
                        self._emit(LoomMutated(
                            op="navigate", rev=self.rev,
                            active_node_id=self.active_node_id,
                        ))
                        return sib.id
            node = LoomNode(id=_ulid(), parent_id=parent, role="user", text=text)
            self._add_child(parent, node)
            self.active_node_id = node.id
            self.rev += 1
            self._emit(LoomMutated(
                op="add_user", rev=self.rev,
                added=(node.id,), active_node_id=node.id,
            ))
            return node.id

    def begin_assistant(
        self,
        parent_id: str,
        recipe: Recipe | None = None,
    ) -> str:
        """Create an empty assistant node under ``parent_id``.

        Returns the new node id.  The session calls this in the gen
        preamble; subsequent ``append_token`` / ``finalize_assistant``
        calls populate the node as the generation streams.
        """
        with self._lock:
            if parent_id not in self.nodes:
                raise UnknownNodeError(parent_id)
            self._conflict_check(parent_id, "begin_assistant")
            node = LoomNode(
                id=_ulid(),
                parent_id=parent_id,
                role="assistant",
                recipe=recipe,
                tokens=[],
                thinking_tokens=[],
            )
            self._add_child(parent_id, node)
            self.active_node_id = node.id
            self.rev += 1
            self._emit(LoomMutated(
                op="begin_assistant", rev=self.rev,
                added=(node.id,), active_node_id=node.id,
            ))
            return node.id

    def append_token(
        self,
        node_id: str,
        score: TokenScoreDict,
        *,
        thinking: bool = False,
    ) -> None:
        """Append a streaming token-score blob to ``node_id``.

        Doesn't bump ``rev`` or emit events — token streaming runs in the
        hot loop and we don't want to flood the WS with per-token tree-
        mutated events.  Token deltas ride the existing ``token`` stream
        with a ``node_id`` tag (added in phase 2).
        """
        with self._lock:
            node = self.nodes.get(node_id)
            if node is None:
                raise UnknownNodeError(node_id)
            if thinking:
                if node.thinking_tokens is None:
                    node.thinking_tokens = []
                node.thinking_tokens.append(score)
            else:
                if node.tokens is None:
                    node.tokens = []
                node.tokens.append(score)

    def finalize_assistant(
        self,
        node_id: str,
        *,
        text: str,
        aggregate_readings: dict[str, float] | None = None,
        applied_steering: str | None = None,
        finish_reason: str | None = None,
        mean_logprob: float | None = None,
        mean_surprise: float | None = None,
        raw_token_ids: list[int] | None = None,
    ) -> None:
        """Mark an in-flight assistant node as complete.

        ``mean_logprob`` / ``mean_surprise`` are the per-turn rollups
        computed in :meth:`SaklasSession._generate_core` from the engine's
        chosen-token logprob stream (response span only — thinking tokens
        are excluded by construction).  ``None`` when logprob capture
        wasn't live (no on_token consumer + no logprobs request), which
        also covers replay-from-legacy-transcripts.

        ``raw_token_ids`` is the engine's exact ``generated_ids`` decode
        sequence (delimiters included) — the forceable prefix a logit
        fork replays.  ``None`` leaves the node's prior value untouched.
        """
        with self._lock:
            node = self.nodes.get(node_id)
            if node is None:
                raise UnknownNodeError(node_id)
            node.text = text
            if aggregate_readings is not None:
                node.aggregate_readings = dict(aggregate_readings)
            node.applied_steering = applied_steering
            node.finish_reason = finish_reason
            node.mean_logprob = mean_logprob
            node.mean_surprise = mean_surprise
            if raw_token_ids is not None:
                node.raw_token_ids = list(raw_token_ids)
            self.rev += 1
            self._emit(LoomMutated(
                op="finalize_assistant", rev=self.rev,
                updated=(node.id,),
            ))

    # ------------------------------------------------------------------
    # Core primitives — edit, branch, navigate, delete_subtree
    # ------------------------------------------------------------------

    def edit(self, node_id: str, text: str) -> None:
        """In-place text replacement.

        No new node, no tree-shape change.  Bumps ``edit_count`` and
        sets ``edited_at`` so downstream consumers can flag the node
        as no-longer-pristine.  Refused when ``node_id`` is in the
        reservation of an in-flight generation.
        """
        with self._lock:
            node = self.nodes.get(node_id)
            if node is None:
                raise UnknownNodeError(node_id)
            if node_id == self.root_id:
                raise InvalidNodeOperationError("cannot edit the root node")
            self._conflict_check(node_id, "edit")
            node.text = text
            node.edit_count += 1
            node.edited_at = time.time()
            self.rev += 1
            self._emit(LoomMutated(
                op="edit", rev=self.rev, updated=(node_id,),
            ))

    def branch(
        self,
        node_id: str,
        text: str,
        *,
        role: Role | None = None,
        make_active: bool = True,
    ) -> str:
        """Create a new sibling of ``node_id`` with the given text.

        Always-sibling: the original is preserved.  Empty text is the
        "branch from blank" UI flavor; pre-fill text is "fork-and-edit".
        Defaults to the same role as the sibling; pass ``role=`` to
        override.  Returns the new node id.  Allowed during in-flight
        generation (creating a sibling doesn't disturb the streaming
        target).
        """
        with self._lock:
            sibling = self.nodes.get(node_id)
            if sibling is None:
                raise UnknownNodeError(node_id)
            if sibling.parent_id is None:
                raise InvalidNodeOperationError(
                    "cannot branch from the root — branch off its children instead"
                )
            new = LoomNode(
                id=_ulid(),
                parent_id=sibling.parent_id,
                role=role if role is not None else sibling.role,
                text=text,
            )
            self._add_child(sibling.parent_id, new)
            if make_active:
                self.active_node_id = new.id
            self.rev += 1
            self._emit(LoomMutated(
                op="branch", rev=self.rev,
                added=(new.id,),
                active_node_id=self.active_node_id if make_active else None,
            ))
            return new.id

    def navigate(self, node_id: str) -> None:
        """Re-point :attr:`active_node_id` to ``node_id``.

        Always free relative to in-flight generation; the gen continues
        attached to its original target invisibly.  Emits a ``navigate``
        mutation event so surfaces re-render.
        """
        with self._lock:
            if node_id not in self.nodes:
                raise UnknownNodeError(node_id)
            if node_id == self.active_node_id:
                return
            self.active_node_id = node_id
            self.rev += 1
            self._emit(LoomMutated(
                op="navigate", rev=self.rev,
                active_node_id=node_id,
            ))

    def delete_subtree(self, node_id: str) -> int:
        """Drop the subtree rooted at ``node_id``.

        Refuses to delete an ancestor of the active node (forces
        navigate-away first) and refuses (via the conflict checker)
        when the subtree intersects an in-flight generation's
        reservation.  Returns the count of nodes removed.
        """
        with self._lock:
            if node_id == self.root_id:
                raise InvalidNodeOperationError("cannot delete the root")
            if node_id not in self.nodes:
                raise UnknownNodeError(node_id)
            if self.is_ancestor_of(node_id, self.active_node_id) or node_id == self.active_node_id:
                raise InvalidNodeOperationError(
                    "cannot delete a subtree containing the active node — navigate away first"
                )
            self._conflict_check(node_id, "delete_subtree")

            # Collect every node in the subtree (DFS) plus the root.
            to_remove: list[str] = [node_id]
            stack = list(self.children_of.get(node_id, []))
            while stack:
                cur = stack.pop()
                to_remove.append(cur)
                stack.extend(self.children_of.get(cur, []))

            parent_id = self.nodes[node_id].parent_id
            if parent_id is not None:
                self.children_of[parent_id] = [
                    c for c in self.children_of.get(parent_id, []) if c != node_id
                ]
            for nid in to_remove:
                self.nodes.pop(nid, None)
                self.children_of.pop(nid, None)

            self.rev += 1
            self._emit(LoomMutated(
                op="delete", rev=self.rev,
                removed=tuple(to_remove),
            ))
            return len(to_remove)

    # ------------------------------------------------------------------
    # Decoration ops — always free
    # ------------------------------------------------------------------

    def star(self, node_id: str, on: bool = True) -> None:
        with self._lock:
            node = self.nodes.get(node_id)
            if node is None:
                raise UnknownNodeError(node_id)
            if node.starred == on:
                return
            node.starred = on
            self.rev += 1
            self._emit(LoomMutated(
                op="star", rev=self.rev, updated=(node_id,),
            ))

    def annotate(self, node_id: str, notes: str) -> None:
        with self._lock:
            node = self.nodes.get(node_id)
            if node is None:
                raise UnknownNodeError(node_id)
            node.notes = notes
            self.rev += 1
            self._emit(LoomMutated(
                op="note", rev=self.rev, updated=(node_id,),
            ))

    # ------------------------------------------------------------------
    # Engine-level workflows: clear (reset), rewind
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Drop the entire tree.  Refused under in-flight generation."""
        with self._lock:
            self._conflict_check(self.root_id, "reset")
            removed = tuple(nid for nid in self.nodes if nid != self.root_id)
            # Wipe everything; rebuild a fresh root.
            self.nodes.clear()
            self.children_of.clear()
            root = LoomNode(id=_ulid(), parent_id=None, role="system")
            self.nodes[root.id] = root
            self.children_of[root.id] = []
            self.root_id = root.id
            self.active_node_id = root.id
            self.rev += 1
            self._emit(LoomMutated(
                op="reset", rev=self.rev,
                removed=removed,
                added=(root.id,),
                active_node_id=root.id,
            ))

    def rewind(self) -> None:
        """Navigate the active node one user→assistant pair up the path.

        Non-destructive: the rewound pair stays in the tree as a dead
        branch, navigable back to.  No-op when the active node is the
        root or its direct child.
        """
        with self._lock:
            path = self.active_path()
            if len(path) < 2:
                return  # nothing meaningful to rewind from
            # Walk back: if active is assistant, go up two (assistant -> user -> parent).
            # If active is user, go up one (user -> parent).  Land on root in
            # the worst case.
            anchor = path[-1]
            steps = 2 if anchor.role == "assistant" else 1
            target_idx = max(0, len(path) - 1 - steps)
            target = path[target_idx]
            if target.id == self.active_node_id:
                return
            self.active_node_id = target.id
            self.rev += 1
            self._emit(LoomMutated(
                op="navigate", rev=self.rev,
                active_node_id=target.id,
            ))

    # ------------------------------------------------------------------
    # Predicate ops (engine surface for phase 5's UI)
    # ------------------------------------------------------------------

    def filter(self, pred: Callable[[LoomNode], bool]) -> set[str]:
        """Return the set of node ids whose nodes satisfy ``pred``."""
        with self._lock:
            return {nid for nid, node in self.nodes.items() if pred(node)}

    def filter_by_expr(
        self,
        text: str,
        *,
        per_token_scores: Any = None,
    ) -> set[str]:
        """Apply a filter-grammar expression to every node.

        Thin wrapper over :func:`saklas.core.tree_filter.filter_tree` —
        the grammar (``agg:``/``any:``/``last:``) is documented in
        :mod:`saklas.core.tree_filter`.  ``per_token_scores`` is an
        optional ``{node_id: {probe: [scores]}}`` mapping needed for
        ``any:`` / ``last:`` clauses; absent entries fail those clauses
        per the documented missing-probe semantics.
        """
        from saklas.core.tree_filter import filter_tree
        return filter_tree(self, text, per_token_scores=per_token_scores)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_dict(self, *, include_tokens: bool = False) -> dict[str, Any]:
        with self._lock:
            # ``saklas_version`` rides alongside ``tree_format`` so future
            # migrations can branch on the originating build even when the
            # schema number hasn't moved — same pattern packs use.  Imported
            # lazily so a circular at module-load time stays impossible.
            from saklas import __version__ as _saklas_version
            return {
                "tree_format": TREE_FORMAT_VERSION,
                "saklas_version": _saklas_version,
                "model_id": self.model_id,
                "session_id": self.session_id,
                "name": self.name,
                "rev": self.rev,
                "root_id": self.root_id,
                "active_node_id": self.active_node_id,
                "nodes": [
                    self.nodes[nid].to_dict(include_tokens=include_tokens)
                    for nid in self.nodes
                ],
                "children_of": {k: list(v) for k, v in self.children_of.items()},
            }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        events: Any | None = None,
        conflict_check: ConflictChecker | None = None,
    ) -> "LoomTree":
        version = data.get("tree_format", 1)
        if version != TREE_FORMAT_VERSION:
            raise LoomTreeError(
                f"unsupported tree_format {version!r} "
                f"(this build supports {TREE_FORMAT_VERSION})"
            )
        tree = cls.__new__(cls)
        tree._lock = threading.RLock()
        tree._events = events
        tree._conflict_check = conflict_check or _noop_conflict_check
        tree.model_id = data.get("model_id")
        tree.session_id = data.get("session_id")
        tree.name = data.get("name")
        tree.rev = int(data.get("rev", 0))
        tree.nodes = {}
        for raw in data.get("nodes", []):
            node = LoomNode.from_dict(raw)
            tree.nodes[node.id] = node
        tree.children_of = {
            k: list(v) for k, v in data.get("children_of", {}).items()
        }
        # Ensure every node has a children_of entry (empty for leaves).
        for nid in tree.nodes:
            tree.children_of.setdefault(nid, [])
        tree.root_id = data["root_id"]
        tree.active_node_id = data.get("active_node_id", tree.root_id)
        return tree

    def save(self, path: Any) -> None:
        """Atomic write of the tree to ``path`` as JSON.

        The main tree file omits token blobs; response/thinking token rows
        are written to a gzip sidecar named ``<stem>.tokens.json.gz``.  This
        keeps routine tree payloads small while explicit save/load preserves
        logprob/top-alt/probe drilldown data.
        """
        from pathlib import Path
        from saklas.io.atomic import write_json_atomic

        out_path = Path(path)
        sidecar = out_path.with_name(f"{out_path.stem}.tokens.json.gz")
        with self._lock:
            data = self.to_dict(include_tokens=False)
            token_nodes: dict[str, dict[str, Any]] = {}
            for node in self.nodes.values():
                if node.tokens or node.thinking_tokens or node.raw_token_ids:
                    token_nodes[node.id] = {
                        "tokens": node.tokens or [],
                        "thinking_tokens": node.thinking_tokens or [],
                        "raw_token_ids": node.raw_token_ids or [],
                    }
            token_payload = None
            if token_nodes:
                data["token_sidecar"] = sidecar.name
                token_payload = {
                    "token_sidecar_format": 1,
                    "nodes": token_nodes,
                }

        if token_payload is None:
            write_json_atomic(out_path, data)
            try:
                sidecar.unlink()
            except FileNotFoundError:
                pass
            return

        sidecar.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{sidecar.name}.",
            suffix=".tmp",
            dir=str(sidecar.parent),
        )
        os.close(fd)
        tmp = Path(tmp_name)
        try:
            with gzip.open(tmp, "wt", encoding="utf-8") as f:
                json.dump(token_payload, f, ensure_ascii=False, separators=(",", ":"))
            os.replace(tmp, sidecar)
        finally:
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass
        write_json_atomic(out_path, data)

    @classmethod
    def load(cls, path: Any, *, events: Any | None = None) -> "LoomTree":
        from pathlib import Path

        in_path = Path(path)
        with open(in_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        sidecar_name = data.get("token_sidecar")
        if sidecar_name:
            sidecar = in_path.parent / str(sidecar_name)
            try:
                with gzip.open(sidecar, "rt", encoding="utf-8") as f:
                    token_payload = json.load(f)
            except FileNotFoundError as e:
                raise LoomTreeError(
                    f"token sidecar declared but missing: {sidecar}"
                ) from e
            if token_payload.get("token_sidecar_format") != 1:
                raise LoomTreeError(
                    f"unsupported token_sidecar_format "
                    f"{token_payload.get('token_sidecar_format')!r}"
                )
            by_id = dict(token_payload.get("nodes", {}))
            for raw in data.get("nodes", []):
                node_tokens = by_id.get(raw.get("id"))
                if not isinstance(node_tokens, dict):
                    continue
                raw["tokens"] = list(node_tokens.get("tokens") or [])
                raw["thinking_tokens"] = list(
                    node_tokens.get("thinking_tokens") or []
                )
                rti = node_tokens.get("raw_token_ids")
                if rti:
                    raw["raw_token_ids"] = list(rti)
        return cls.from_dict(data, events=events)


__all__ = [
    "LoomNode",
    "LoomTree",
    "Recipe",
    "LoomMutated",
    "LoomTreeError",
    "UnknownNodeError",
    "InvalidNodeOperationError",
    "MutationDuringGenerationError",
    "TREE_FORMAT_VERSION",
    "derive_seed_schedule",
]
