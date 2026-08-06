"""Native WebSocket token+probe co-stream (``WS /saklas/v1/sessions/{id}/stream``).

Bidirectional token + per-token probe co-stream — the killer feature of
the native tree API.  The connection runs a single perpetual reader, a
loom-mutation forwarder, and a per-generate-turn worker; see the
``_ws_handle_generate`` docstring for the concurrency design.
"""

# pyright: reportUnusedFunction=false

from __future__ import annotations

import asyncio
import logging
import uuid
from collections import deque
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Literal, cast

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from saklas.core.errors import SaklasError
from saklas.core.loom import LoomMutated
from saklas.core.results import GenerationResult, TokenAlt
from saklas.core.sampling import SamplingConfig
from saklas.core.session import SaklasSession
from saklas.core.token_callback import TokenConsumer, TokenConsumerOptions
from saklas.core.steering import Steering
from saklas.server.app import acquire_session_lock, ws_auth_ok
from saklas.server.native_common import SINGLE_SESSION_ID
from saklas.server.request_helpers import merge_steering, parse_request_steering
from saklas.server.tree_models import cast_json, node_json
from saklas.server.ws_events import build_token_event
from saklas.server.ws_models import (
    WSGenerateMessage,
    WSSubmitMessage,
    build_input,
    build_sampling_config,
    result_to_json,
)

_logger = logging.getLogger(__name__)

JSONValue = None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]
JSONObject = dict[str, JSONValue]


@dataclass(frozen=True)
class _Stop:
    pass


@dataclass(frozen=True)
class _Disconnect:
    pass


@dataclass(frozen=True)
class _ReaderFailure:
    message: str
    code: str


@dataclass(frozen=True)
class _InvalidInbound:
    message: str


_Inbound = (
    WSGenerateMessage | WSSubmitMessage | _Stop | _Disconnect
    | _ReaderFailure | _InvalidInbound
)


@dataclass(frozen=True)
class _TokenFrame:
    payload: JSONObject


@dataclass(frozen=True)
class _TokenDone:
    pass


_TokenQueueItem = _TokenFrame | _TokenDone


def _error_frame(
    message: str,
    *,
    code: str = "ValueError",
    status: int = 400,
    node_id: str | None = None,
    sibling_index: int = 0,
) -> JSONObject:
    """Build the one ``{type: "error", …}`` frame shape this stream emits.

    Every error frame carries the same five keys, so a client never has to
    branch on which of them happens to be present.  ``node_id`` is filled only
    once a generation has produced its assistant node.
    """
    return {
        "type": "error",
        "message": message,
        "code": code,
        "status": status,
        "node_id": node_id,
        "sibling_index": sibling_index,
    }


def _validation_message(exc: ValidationError) -> str:
    """Render a pydantic rejection as one line, the native tree's convention.

    Each error reads ``"<field path>: <message>"``, the shape
    ``app._on_validation_error`` gives native REST bodies; a model-level rule
    has no path, so its message stands alone (``"a generate message cannot be
    both a fork and a prefill"``) — ``PydanticCustomError`` in ``ws_models``
    is what keeps that verbatim rather than ``"Value error, …"``.

    Every error is reported, not just the first: ``input`` is a union, so its
    real failure (a bad key inside a message object) is the *second* branch
    error and a first-only render would name the wrong problem.
    """
    parts: list[str] = []
    for error in exc.errors():
        message = str(error.get("msg", "invalid request"))
        loc = ".".join(str(part) for part in error.get("loc", ()))
        parts.append(f"{loc}: {message}" if loc else message)
    return "; ".join(parts) or str(exc)


class _WSRequestError(Exception):
    """A request-shape problem to report as an error frame, not a close.

    Raised by the pre-dispatch validators so the field-consistency rules read
    as straight-line checks instead of a ``send_json``/``return`` pair each.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str = "ValueError",
        status: int = 400,
        node_id: str | None = None,
        sibling_index: int = 0,
    ) -> None:
        super().__init__(message)
        self.frame = _error_frame(
            message, code=code, status=status,
            node_id=node_id, sibling_index=sibling_index,
        )


@dataclass(frozen=True)
class _AuthoredTurn:
    """The turn a ``submit`` commits before its generation runs."""

    text: str
    role: Literal["user", "assistant"]
    thinking: str | None


def register_ws_stream(app: FastAPI) -> None:
    """Mount the bidirectional WebSocket token+probe co-stream."""
    session = app.state.session

    @app.websocket("/saklas/v1/sessions/{session_id}/stream")
    async def session_stream(websocket: WebSocket, session_id: str):
        if not ws_auth_ok(websocket):
            await websocket.close(code=1008, reason="unauthorized")
            return
        if session_id != SINGLE_SESSION_ID:
            await websocket.accept()
            await websocket.close(code=1008, reason="session not found")
            return
        await websocket.accept()

        # Single perpetual reader.  ``websocket.receive_json()`` is bound
        # to a per-connection ``recv_in_progress`` flag in the underlying
        # ``websockets`` library; cancelling a pending receive doesn't
        # clear the flag immediately, so any handler that called
        # ``receive_json()`` while another concurrent (even just-cancelled)
        # caller was pending tripped a "cannot call recv while another
        # coroutine is already waiting" RuntimeError.  Routing every
        # incoming frame through one queue lets both the outer dispatch
        # loop and the in-flight generation share the read side without
        # ever overlapping calls into the WS.
        incoming: asyncio.Queue[_Inbound] = asyncio.Queue()

        async def _reader():
            try:
                while True:
                    raw = await websocket.receive_json()
                    if not isinstance(raw, dict):
                        await incoming.put(_InvalidInbound("message must be an object"))
                    elif raw.get("type") in ("generate", "submit"):
                        try:
                            message = (
                                WSSubmitMessage(**raw)
                                if raw.get("type") == "submit"
                                else WSGenerateMessage(**raw)
                            )
                            await incoming.put(message)
                        except ValidationError as exc:
                            await incoming.put(
                                _InvalidInbound(_validation_message(exc))
                            )
                    elif raw.get("type") == "stop":
                        await incoming.put(_Stop())
                    else:
                        await incoming.put(_InvalidInbound(f"unknown message type: {raw.get('type')!r}"))
            except WebSocketDisconnect:
                await incoming.put(_Disconnect())
            except Exception as e:
                # Surface any other read-side failure into the queue so
                # the dispatcher can close cleanly instead of leaking.
                await incoming.put(_ReaderFailure(str(e), type(e).__name__))

        reader_task = asyncio.create_task(_reader())

        # Loom: subscribe to ``LoomMutated`` for the connection's
        # lifetime and forward exact ``tree_mutated`` frames. Held
        # in a queue + forwarder task so the EventBus callback (which
        # runs on the gen thread) never touches the WS directly.
        loop = asyncio.get_running_loop()
        tree_event_queue: asyncio.Queue[JSONObject] = asyncio.Queue()
        # ``websocket.send_json`` is not safe for concurrent callers —
        # starlette serializes per-call but two tasks can interleave
        # bytes on the wire and corrupt the frame sequence.  This lock
        # is the single send-side serializer the connection uses; both
        # the generate-handler and the tree-forwarder acquire it before
        # every send.
        ws_send_lock = asyncio.Lock()

        async def _send_json(payload: JSONObject) -> None:
            async with ws_send_lock:
                await websocket.send_json(payload)

        def _queue_tree_event(payload: JSONObject) -> None:
            with suppress(Exception):
                loop.call_soon_threadsafe(tree_event_queue.put_nowait, payload)

        def _on_loom_event(event: object) -> None:
            if not isinstance(event, LoomMutated):
                return
            added_nodes = [node_json(session, nid) for nid in event.added]
            mutated_payload = cast(JSONObject, {
                "type": "tree_mutated",
                "op": event.op,
                "rev": event.rev,
                "added": added_nodes,
                "removed": list(event.removed),
                "updated": [
                    node_json(session, nid)
                    for nid in event.updated
                    if session.tree.has(nid)
                ],
                "active_node_id": event.active_node_id,
            })
            # The roster is derived from the full tree's observed labels as
            # well as explicit configuration.  Inline the small effective
            # roster on every mutation so adds, deletes, and restores reconcile
            # identity without a refetch or provenance inference client-side.
            mutated_payload["cast"] = cast(JSONValue, cast_json(session))
            _queue_tree_event(mutated_payload)
        loom_unsub = session.events.subscribe(_on_loom_event)

        async def _tree_forwarder():
            """Forward tree-mutated events as WS frames.

            Runs as a dedicated task for the connection's lifetime so
            tree mutations from any source (this WS, a REST route on a
            different connection, the gen loop) reach the client without
            interleaving with the per-turn token loop.
            """
            try:
                while True:
                    payload = await tree_event_queue.get()
                    try:
                        await _send_json(payload)
                    except Exception:
                        return
                    if payload.get("op") == "finalize_assistant":
                        updated = payload.get("updated")
                        if isinstance(updated, list):
                            for node in updated:
                                if not isinstance(node, dict):
                                    continue
                                node_id = node.get("id")
                                if isinstance(node_id, str):
                                    tree_forwarded_finalized.add(node_id)
                    tree_forwarded_event.set()
            except asyncio.CancelledError:
                return

        tree_forwarded_finalized: set[str] = set()
        tree_forwarded_event = asyncio.Event()
        forwarder_task = asyncio.create_task(_tree_forwarder())

        async def _wait_for_tree_finalization(node_id: str) -> None:
            """Do not let ``done`` overtake its authoritative tree delta.

            The mutation forwarder and generation dispatcher are separate
            tasks sharing a send lock.  Serialization alone does not impose
            ordering: the dispatcher could acquire the lock first even though
            the worker had already queued its ``finalize`` event, briefly
            leaving clients with a completed but empty assistant node.
            """
            while node_id not in tree_forwarded_finalized:
                tree_forwarded_event.clear()
                if node_id in tree_forwarded_finalized:
                    return
                event_wait = asyncio.create_task(tree_forwarded_event.wait())
                finished, pending = await asyncio.wait(
                    {event_wait, forwarder_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    if task is not forwarder_task:
                        task.cancel()
                        with suppress(asyncio.CancelledError):
                            await task
                if (
                    forwarder_task in finished
                    and node_id not in tree_forwarded_finalized
                ):
                    raise RuntimeError(
                        "tree mutation stream ended before generation finalization"
                    )

        deferred_incoming: deque[_Inbound] = deque()

        async def _cancel_and_wait(task: asyncio.Task[Any]) -> None:
            task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await task

        def _stop_session_safely() -> None:
            with suppress(Exception):
                session.stop()

        try:
            while True:
                msg = (
                    deferred_incoming.popleft()
                    if deferred_incoming
                    else await incoming.get()
                )
                if isinstance(msg, _Disconnect):
                    raise WebSocketDisconnect(code=1000)
                if isinstance(msg, _ReaderFailure):
                    raise RuntimeError(msg.message)
                if isinstance(msg, (WSGenerateMessage, WSSubmitMessage)):
                    await _ws_handle_generate(
                        session, msg, app.state.default_steering, incoming,
                        deferred_incoming, _send_json, _wait_for_tree_finalization,
                    )
                elif isinstance(msg, _Stop):
                    # Idle-state stop: nothing in flight.
                    continue
                else:
                    await _send_json(_error_frame(
                        msg.message, code="ValidationError", status=400,
                    ))
        except WebSocketDisconnect:
            # Ensure any stray generation is signaled.
            _stop_session_safely()
            return
        except Exception as e:
            try:
                await _send_json(_error_frame(
                    str(e), code=type(e).__name__, status=500,
                ))
            finally:
                with suppress(Exception):
                    await websocket.close(code=1011)
        finally:
            # Drop the loom subscription before tearing down the reader
            # so the EventBus stops dispatching into a queue nobody
            # reads.
            with suppress(Exception):
                loom_unsub()
            await _cancel_and_wait(forwarder_task)
            # Reader holds the only ``receive_json()`` call on the WS.
            # Cancel + await so the cancellation propagates fully before
            # the connection tears down.
            await _cancel_and_wait(reader_task)


def _normalize_submit(
    submit: WSSubmitMessage,
) -> tuple[WSGenerateMessage, _AuthoredTurn | None, bool]:
    """Lower a ``submit`` frame onto the specialist ``generate`` schema.

    ``submit`` is the ONE authored-turn contract on this wire: an explicit
    authored role plus an optional generated role.  Append-only submissions
    take the no-decode commit path; append+generate keeps the authored turn
    for one atomic commit inside the generation worker and generates from
    ``input=None`` once that turn has landed.

    Returns ``(generate_message, authored_turn, commit_only)``.  When
    ``commit_only`` the authored turn is the whole request and the returned
    generate message carries only the commit's context (parent, sampling
    labels, raw); otherwise the authored turn — if any — is committed inside
    the generation worker.  Raises :class:`_WSRequestError` on any
    field-consistency violation.
    """
    if submit.text is None:
        if submit.authored_role is not None:
            raise _WSRequestError("authored_role requires non-empty text")
        if submit.authored_thinking is not None:
            raise _WSRequestError("authored_thinking requires non-empty text")
        authored: _AuthoredTurn | None = None
    else:
        if submit.text == "":
            raise _WSRequestError("submit text must be non-empty when present")
        if submit.authored_role is None:
            raise _WSRequestError("text requires authored_role")
        authored = _AuthoredTurn(
            text=submit.text,
            role=submit.authored_role,
            thinking=submit.authored_thinking,
        )
    if submit.generated_role is None and authored is None:
        raise _WSRequestError("submit requires text or generated_role")

    if submit.generated_role is None:
        assert authored is not None  # guarded by the check above
        return (
            WSGenerateMessage(
                type="generate",
                parent_node_id=submit.parent_node_id,
                sampling=submit.sampling,
                raw=submit.raw,
            ),
            authored,
            True,
        )
    return (
        WSGenerateMessage(
            type="generate",
            input=None,
            steering=submit.steering,
            sampling=submit.sampling,
            thinking=submit.thinking,
            stateless=False,
            raw=submit.raw,
            parent_node_id=submit.parent_node_id,
            n=submit.n,
            recipe_override=submit.recipe_override,
            generate_seat=submit.generated_role,
        ),
        authored,
        False,
    )


async def _ws_handle_commit(
    session: SaklasSession,
    msg: WSGenerateMessage,
    authored: _AuthoredTurn,
    send_json: Callable[[JSONObject], Awaitable[None]],
) -> None:
    """Land one authored turn with no decode.

    Short-circuits the n-way fan-out and the streaming worker entirely: one
    tree mutation, one ``started`` (``node_id`` null — the node id is only
    known after the append), one ``done`` carrying the new node.  No token
    frames.  Seating is free: ``session.append_turn`` is the role-neutral
    primitive, so an assistant-authored opening turn needs no parent.
    """
    parent_node_id = msg.parent_node_id
    generation_id = uuid.uuid4().hex[:12]
    await send_json({
        "type": "started",
        "generation_id": generation_id,
        "node_id": None,
        "sibling_index": 0,
        "sibling_count": 1,
    })
    # Per-message role labels ride the commit's sampling block too
    # (roleplay scaffold).  Raw / flat commits carry no chat-template
    # role, so labels are suppressed there.
    commit_label = None
    if msg.sampling is not None:
        commit_label = (
            msg.sampling.user_role
            if authored.role == "user"
            else msg.sampling.assistant_role
        ) or None
    async with acquire_session_lock(session) as acquired:
        if not acquired:
            await send_json(_error_frame(
                "session locked — try again when the current generation finishes",
                code="SessionLocked", status=503,
            ))
            return
        try:
            new_id = await asyncio.to_thread(
                session.append_turn,
                parent_node_id,
                authored.text,
                role=authored.role,
                raw=msg.raw,
                role_label=None if msg.raw else commit_label,
                thinking=None if msg.raw else (authored.thinking or None),
            )
        except SaklasError as e:
            status, message = e.user_message()
            await send_json(_error_frame(
                message, code=type(e).__name__, status=status,
            ))
            return
    await send_json({
        "type": "done",
        "result": {
            "role": authored.role,
            "text": authored.text,
            "node_id": new_id,
            "finish_reason": "stop",
            "mean_logprob": None,
            "mean_surprise": None,
        },
        "node_id": new_id,
        "sibling_index": 0,
        "sibling_count": 1,
    })


async def _ws_handle_generate(
    session: SaklasSession,
    msg: WSGenerateMessage | WSSubmitMessage,
    default_steering: "Steering | None",
    incoming: asyncio.Queue[_Inbound],
    deferred_incoming: "deque[_Inbound]",
    send_json: Callable[[JSONObject], Awaitable[None]],
    wait_for_tree_finalization: Callable[[str], Awaitable[None]],
) -> None:
    """Validate one inbound turn and dispatch it to its mode handler.

    ``generate``'s own field-consistency rules already ran in the schema
    (``WSGenerateMessage`` model validators, rejected by the reader), so the
    only checks left here are ``submit``'s cross-field rules, which have no
    schema home: they decide *which* generate frame to lower onto
    (:func:`_normalize_submit`).  Then the steering expression is composed
    over the server default and the turn goes to exactly one of two handlers:
    :func:`_ws_handle_commit` (no decode) or :func:`_ws_stream_generation`
    (the token fan-out).  Every rejection here is an error frame on a
    connection that stays open — a 400-grade user mistake must not close the
    socket.
    """
    authored: _AuthoredTurn | None = None
    commit_only = False
    if isinstance(msg, WSSubmitMessage):
        try:
            msg, authored, commit_only = _normalize_submit(msg)
        except _WSRequestError as e:
            await send_json(e.frame)
            return

    if commit_only:
        assert authored is not None  # set together by ``_normalize_submit``
        await _ws_handle_commit(session, msg, authored, send_json)
        return

    sampling = build_sampling_config(msg.sampling)
    try:
        req_steering, explicit_clear = parse_request_steering(msg.steering)
        thinking_override: bool | None = None
        if req_steering is not None and req_steering.thinking is not None:
            thinking_override = req_steering.thinking
        steering = merge_steering(
            req_steering, default_steering, explicit_clear, thinking_override,
        )
    except SaklasError as e:
        # ``parse_request_steering`` -> ``parse_expr`` -> ``resolve_bare_atom`` can
        # raise ``SteeringExprError`` / ``AmbiguousSelectorError`` /
        # ``AmbiguousVariantError`` on malformed or colliding input.
        # FastAPI's ``@app.exception_handler(SaklasError)`` doesn't apply
        # to WebSocket routes, so without this guard the exception falls
        # through to the outer reader loop's ``except Exception`` which
        # closes the socket with code 1011. A 400-grade user mistake
        # shouldn't kill the connection — send the error frame and let
        # the client try again on the same WS.
        status, message = e.user_message()
        await send_json(_error_frame(
            message, code=type(e).__name__, status=status,
        ))
        return

    await _ws_stream_generation(
        session, msg, steering, sampling, authored,
        incoming, deferred_incoming, send_json, wait_for_tree_finalization,
    )


async def _ws_stream_generation(
    session: SaklasSession,
    msg: WSGenerateMessage,
    steering: "Steering | None",
    sampling: SamplingConfig | None,
    authored: _AuthoredTurn | None,
    incoming: asyncio.Queue[_Inbound],
    deferred_incoming: "deque[_Inbound]",
    send_json: Callable[[JSONObject], Awaitable[None]],
    wait_for_tree_finalization: Callable[[str], Awaitable[None]],
) -> None:
    """Run one generate turn and stream token/done/error events.

    Concurrency design: the synchronous ``session.generate_stream`` is
    driven from a worker thread via ``asyncio.to_thread``.  Its
    ``on_token`` callback is invoked on the worker thread; it bridges
    into the asyncio loop by calling
    ``loop.call_soon_threadsafe(queue.put_nowait, event)``.  The main
    coroutine races two tasks: one pulls ``TokenEvent``s from a local
    queue and forwards them as ``{type: "token", ...}`` frames; the
    other pulls client frames from the shared ``incoming`` queue
    (populated by the connection's single reader task) so an in-flight
    ``{type: "stop"}`` can call ``session.stop()`` without blocking on
    the token loop.

    ``asyncio.wait(..., FIRST_COMPLETED)`` is used in a loop: whenever
    the incoming task returns a stop frame we signal the session and
    keep draining tokens until the worker joins; whenever the queue
    delivers a sentinel we finish.  The WS stays open across generate
    turns — a client can submit ``{type: "generate", ...}`` again after
    ``done``, and the perpetual reader keeps feeding the shared queue
    between turns so we never have two ``receive_json()`` calls in
    flight.

    **Loom**: ``parent_node_id`` attaches the assistant node
    under a specific tree node; ``n>1`` fans out N siblings serially.
    Each sibling produces its own ``started`` / token-stream / ``done``
    triplet, all tagged with the assistant node id.  ``tree_mutated``
    events ride the connection-level subscription
    in ``session_stream``; this handler only emits the per-sibling
    ``started`` / ``token`` / ``done`` frames.  ``authored`` (from a
    ``submit`` that also generates) is committed once inside the worker,
    under the same session lock as the whole fan.
    """
    loop = asyncio.get_running_loop()
    n = msg.n
    parent_node_id = msg.parent_node_id
    submitted_parent_holder: list[str] = []

    # Per-sibling seed schedule: when n>1, derive deterministic per-
    # sibling seeds from the request seed (or fresh entropy).  Single
    # streams (n=1) use the user's seed verbatim.
    from saklas.core.loom import derive_seed_schedule
    base_seed = sampling.seed if sampling is not None else None
    seeds: list[int | None]
    seeds = [base_seed] if n == 1 else list(derive_seed_schedule(base_seed, n))

    def _stop_session_safely() -> None:
        with suppress(Exception):
            session.stop()

    # Acquire the session lock for the full N-way batch lifetime so
    # concurrent WS clients serialize FIFO instead of overlapping.
    # ``session.generate_stream`` itself uses the threading ``_gen_lock``
    # to gate the actual generation, but the async-level lock is what
    # queues HTTP/WS endpoints fairly.  Bounded to SESSION_LOCK_TIMEOUT_SECONDS
    # (300 s) so a long-running generation doesn't pin the lock forever;
    # a timeout surfaces as a WS error frame (no HTTP in the WS context).
    async with acquire_session_lock(session) as acquired:
        if not acquired:
            await send_json(_error_frame(
                "session locked — try again when the current generation finishes",
                code="SessionLocked", status=503,
            ))
            return
        for sibling_idx, seed_i in enumerate(seeds):
            generation_id = uuid.uuid4().hex[:12]
            tree_rev_before_generation = int(session.tree.rev)

            # Per-sibling sampling override carrying the derived seed.
            if n == 1 and seed_i is None:
                per_sibling_sampling = sampling
            else:
                from dataclasses import replace as _dc_replace
                base_sc = sampling if sampling is not None else SamplingConfig()
                per_sibling_sampling = _dc_replace(base_sc, seed=seed_i)

            token_queue: asyncio.Queue[_TokenQueueItem] = asyncio.Queue()
            # The tree assigns the assistant node id at ``begin_assistant``
            # time inside ``_generate_core``; we don't know it before the
            # gen starts.  The on_token callback reads the live active
            # node off the tree (which is set to the streaming assistant
            # node for the lifetime of the gen).
            current_node_holder: list[str | None] = [None]

            def _on_token(
                text: str,
                is_thinking: bool,
                tid: int | None,
                lp: float | None,
                top_alts: list[TokenAlt] | None,
                perplexity: float | None = None,
                _node_holder: list[str | None] = current_node_holder,
                _token_queue: asyncio.Queue[_TokenQueueItem] = token_queue,
            ) -> None:
                event = build_token_event(
                    session,
                    _node_holder,
                    text=text,
                    is_thinking=is_thinking,
                    tid=tid,
                    lp=lp,
                    top_alts=top_alts,
                    perplexity=perplexity,
                )
                loop.call_soon_threadsafe(
                    _token_queue.put_nowait, _TokenFrame(cast(JSONObject, event))
                )
            consumer = TokenConsumer(
                _on_token,
                TokenConsumerOptions(
                    live_scores=True,
                    per_layer_scores=True,
                    lens_readout=True,
                    sae_readout=True,
                    perplexity=True,
                ),
            )
            # Live J-lens workspace readout: computed only when the session's
            # live lens is enabled (POST .../lens/live) AND the tap consumer
            # declares interest through ``TokenConsumerOptions`` (the same
            # typed capability object used by ``generate_stream``), so an enabled lens
            # streams per-step top-k on the ``token`` frame's ``lens_readout``.

            result_holder: list[GenerationResult] = []
            error_holder: list[BaseException] = []

            # Recipe-override (phase 5): accept either a mode string or a
            # partial-recipe expression.  We pass it through ``generate``
            # so the engine resolves the overlay against the parent's
            # recipe; ``session.regen_with_modifier`` is the matching
            # higher-level wrapper but the WS path already has the
            # required context.
            recipe_override = msg.recipe_override

            def _worker(
                _sampling: SamplingConfig | None = per_sibling_sampling,
                _on_token: TokenConsumer = consumer,
                _result_holder: list[GenerationResult] = result_holder,
                _error_holder: list[BaseException] = error_holder,
                _token_queue: asyncio.Queue[_TokenQueueItem] = token_queue,
                _recipe_override: str | None = recipe_override,
            ) -> None:
                try:
                    effective_parent = parent_node_id
                    if authored is not None:
                        if not submitted_parent_holder:
                            commit_label = None
                            if not msg.raw and msg.sampling is not None:
                                commit_label = (
                                    msg.sampling.user_role
                                    if authored.role == "user"
                                    else msg.sampling.assistant_role
                                ) or None
                            committed_id = session.append_turn(
                                parent_node_id,
                                authored.text,
                                role=authored.role,
                                raw=msg.raw,
                                role_label=commit_label,
                                thinking=authored.thinking,
                            )
                            submitted_parent_holder.append(committed_id)
                        effective_parent = submitted_parent_holder[0]
                    if msg.fork_node_id is not None:
                        # Fork: recipe / sampling / parent all come from
                        # the source node inside ``fork_from_token``; the
                        # WS-level steering/sampling/n fields are ignored.
                        result = session.fork_from_token(
                            msg.fork_node_id,
                            int(msg.fork_raw_index),  # pyright: ignore[reportArgumentType]  # guarded non-None by is_fork check above; int() accepts int|None only at runtime with None already excluded
                            int(msg.fork_alt_token_id),  # pyright: ignore[reportArgumentType]  # guarded non-None by is_fork check above; int() accepts int|None only at runtime with None already excluded
                            on_token=_on_token,
                        )
                    elif msg.prefill_node_id is not None:
                        # Prefill: anchor / parent come from the user node
                        # inside ``prefill_assistant``; ``input`` is
                        # ignored.  ``steering`` / ``sampling`` ride through
                        # like a normal generate; ``thinking`` is forced
                        # off (the prefill is an answer, not a thought).
                        result = session.prefill_assistant(
                            msg.prefill_node_id,
                            str(msg.prefill_text),
                            steering=steering,
                            sampling=_sampling,
                            on_token=_on_token,
                        )
                    else:
                        gen_kwargs: dict[str, Any] = {
                            "steering": steering,
                            "sampling": _sampling,
                            "stateless": msg.stateless,
                            "raw": msg.raw,
                            "thinking": msg.thinking,
                            "on_token": _on_token,
                            "parent_node_id": effective_parent,
                        }
                        if _recipe_override is not None:
                            gen_kwargs["recipe_override"] = _recipe_override
                        if msg.generate_seat is not None:
                            gen_kwargs["gen_seat"] = msg.generate_seat
                        # Fan-out is intentionally sibling-oriented. A normal
                        # one-shot call takes the engine's coalescing default;
                        # only the exceptional fan case needs an override.
                        if n > 1:
                            gen_kwargs["append_same_role"] = False
                        result = session.generate(
                            build_input(msg.input), **gen_kwargs,
                        ).first
                    _result_holder.append(result)
                except BaseException as e:
                    _error_holder.append(e)
                finally:
                    loop.call_soon_threadsafe(_token_queue.put_nowait, _TokenDone())

            await send_json({
                "type": "started",
                "generation_id": generation_id,
                # ``node_id`` is filled in lazily by the first token
                # event (the assistant node is created inside
                # ``_generate_core``); ``started`` includes the request-
                # level context the client needs to allocate state.
                "node_id": None,
                "sibling_index": sibling_idx,
                "sibling_count": n,
            })

            worker_task = asyncio.create_task(asyncio.to_thread(_worker))

            # Race two queue reads — token frames from the worker and
            # client frames from the connection's perpetual reader.
            # Neither side ever calls ``websocket.receive_json()``
            # directly, so the underlying ``recv_in_progress`` flag is
            # owned by the reader task alone for the connection's
            # lifetime.
            done = False
            stop_signaled = False
            token_get = asyncio.create_task(token_queue.get())
            client_get = asyncio.create_task(incoming.get())
            try:
                while not done:
                    finished, _pending = await asyncio.wait(
                        {token_get, client_get}, return_when=asyncio.FIRST_COMPLETED,
                    )
                    if client_get in finished:
                        incoming_msg = client_get.result()
                        # Disconnect / reader-error lifecycle messages:
                        # signal the worker to wind down; let the outer
                        # loop propagate the disconnect on the next
                        # iteration.
                        if isinstance(incoming_msg, _Stop):
                            _stop_session_safely()
                            stop_signaled = True
                        elif isinstance(incoming_msg, (_Disconnect, _ReaderFailure)):
                            _stop_session_safely()
                            stop_signaled = True
                            deferred_incoming.append(incoming_msg)
                        else:
                            # Out-of-band generate/invalid frame: defer until done.
                            deferred_incoming.append(incoming_msg)
                        client_get = asyncio.create_task(incoming.get())
                    if token_get in finished:
                        item = token_get.result()
                        if isinstance(item, _TokenDone):
                            done = True
                        else:
                            await send_json(item.payload)
                            token_get = asyncio.create_task(token_queue.get())
            finally:
                # Keep each queue read alive until it resolves.  Cancelling and
                # recreating ``incoming.get()`` after every token can race a
                # just-delivered stop frame: Queue.get has already removed the
                # item, but task cancellation wins before the dispatcher sees
                # the result.  At turn end, preserve any frame that completed
                # just after the final token wait for the outer dispatcher.
                if not client_get.done():
                    client_get.cancel()
                with suppress(asyncio.CancelledError):
                    await client_get
                if not client_get.cancelled():
                    deferred_incoming.append(client_get.result())
                if not token_get.done():
                    token_get.cancel()
                with suppress(asyncio.CancelledError):
                    await token_get
                # Drain any residual events the worker pushed between
                # sentinel and join — should be none because the
                # sentinel is last, but cheap insurance.
                await worker_task

            if error_holder and not result_holder:
                exc = error_holder[0]
                # The model worker runs in a thread, so by the time control
                # returns here there is no active exception for
                # ``logger.exception`` to capture.  Preserve its original
                # traceback explicitly: native WebSocket failures used to be
                # surfaced only to the browser, leaving operators with no
                # actionable server-side evidence for backend/device bugs.
                _logger.error(
                    "native WebSocket generation failed",
                    exc_info=(type(exc), exc, exc.__traceback__),
                )
                if isinstance(exc, SaklasError):
                    status, message = exc.user_message()
                else:
                    status = 500
                    message = "Generation failed. Check the server log for details."
                await send_json(_error_frame(
                    message,
                    code=type(exc).__name__,
                    status=status,
                    node_id=current_node_holder[0],
                    sibling_index=sibling_idx,
                ))
                # On error inside a sibling, abort the remaining fan-out
                # rather than continuing with stale state.
                return

            if not result_holder:
                raise RuntimeError("generation completed without a result")
            result = result_holder[0]
            result_json = result_to_json(result)
            # The engine deliberately uses the OpenAI-compatible ``stop``
            # finish reason for EOS, stop sequences, and an external stop
            # request.  This native WebSocket knows which case happened:
            # surface a distinct UI-only reason so the dashboard does not
            # present a user-cancelled 659/1024-token turn as ordinary
            # completion.  Protocol adapters and the stored loom recipe keep
            # the engine's canonical reason unchanged.
            if stop_signaled and result_json.get("finish_reason") == "stop":
                result_json["finish_reason"] = "cancelled"
            # The settled per-probe aggregate rides the ``done`` event in the
            # 5.x measurement envelope and nowhere else — the same clean break
            # the ``token`` frame already made.  Geometry / lens / SAE readings
            # are split by family inside ``instruments``; a client merges them
            # exactly as it merges the token frame's.  The flat pre-5.x
            # ``probe_readings`` block is gone from this frame (it survives on
            # the OpenAI / Ollama ``x-saklas-probe-readings`` extension, which
            # is a real external contract).  Result-parameterized so each n>1
            # sibling reports its own result.
            # Built once by the engine at finalize (``GenerationResult.
            # measurements``), so the server does not re-split readings by
            # family — and the lens/SAE channels keep their native
            # ``ScalarReading`` shape instead of a reprojection.
            if result is not None and result.measurements:
                result_json["measurements"] = result.measurements
            # Phase 1 logit pass: stamp the per-turn logprob rollup on the
            # ``done`` event so subscribers (loom sidebar's sort-by-surprise,
            # webui chat-header summary) don't need to re-fetch the node.
            # Source of truth is the finalized loom node, populated by
            # :meth:`LoomTree.finalize_assistant` upstream of this branch.
            # Stateless gens / pre-logit-pass replays land with ``None``
            # which the wire layer passes through transparently.
            mean_logprob_out: float | None = None
            mean_surprise_out: float | None = None
            finalized_node_id = current_node_holder[0]
            if finalized_node_id is not None:
                node = session.tree.get(finalized_node_id)
                mean_logprob_out = node.mean_logprob
                mean_surprise_out = node.mean_surprise
            result_json["mean_logprob"] = mean_logprob_out
            result_json["mean_surprise"] = mean_surprise_out
            if (
                finalized_node_id is not None
                and int(session.tree.rev) > tree_rev_before_generation
            ):
                await wait_for_tree_finalization(finalized_node_id)
            await send_json({
                "type": "done",
                "result": result_json,
                "node_id": current_node_holder[0],
                "sibling_index": sibling_idx,
                "sibling_count": n,
            })

            # Mid-batch stop honors the batch plan: cancel the currently
            # streaming sibling and skip every remaining queued sibling.
            if stop_signaled:
                break
