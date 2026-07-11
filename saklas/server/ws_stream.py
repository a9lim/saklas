"""Native WebSocket token+probe co-stream (``WS /saklas/v1/sessions/{id}/stream``).

Bidirectional token + per-token probe co-stream — the killer feature of
the native tree API.  The connection runs a single perpetual reader, a
loom-mutation forwarder, and a per-generate-turn worker; see the
``_ws_handle_generate`` docstring for the concurrency design.
"""

# pyright: reportUnusedFunction=false

from __future__ import annotations

import asyncio
import uuid
from collections import deque
from contextlib import suppress
from typing import Any, Awaitable, Callable

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from saklas.core.errors import SaklasError
from saklas.core.loom import LoomMutated
from saklas.core.results import GenerationResult, RunSet
from saklas.core.sampling import SamplingConfig
from saklas.core.session import SaklasSession
from saklas.core.steering import Steering
from saklas.server.app import acquire_session_lock, ws_auth_ok
from saklas.server.native_common import SINGLE_SESSION_ID
from saklas.server.request_helpers import merge_steering, parse_request_steering
from saklas.server.streaming import probe_reading_aggregate
from saklas.server.tree_models import node_json
from saklas.server.ws_events import build_token_event
from saklas.server.ws_models import (
    WSGenerateMessage,
    build_sampling,
    per_token_probes,
    result_to_json,
)


def register_ws_stream(app: FastAPI) -> None:
    """Mount the bidirectional WebSocket token+probe co-stream."""
    session = app.state.session

    @app.websocket("/saklas/v1/sessions/{session_id}/stream")
    async def session_stream(websocket: WebSocket, session_id: str):
        # NOTE: only ``session_id == "default"`` is actually reachable
        # here — HF model ids contain '/' and the FastAPI path parameter
        # is not declared ``{session_id:path}``, so the model-id branch
        # is an HTTP-route convenience only.  Kept as a no-op guard.
        if not ws_auth_ok(websocket):
            await websocket.close(code=1008, reason="unauthorized")
            return
        if session_id not in (SINGLE_SESSION_ID, session.model_id):
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
        incoming: asyncio.Queue[Any] = asyncio.Queue()
        _DISCONNECT = object()

        async def _reader():
            try:
                while True:
                    msg = await websocket.receive_json()
                    await incoming.put(msg)
            except WebSocketDisconnect:
                await incoming.put(_DISCONNECT)
            except Exception as e:
                # Surface any other read-side failure into the queue so
                # the dispatcher can close cleanly instead of leaking.
                await incoming.put({"_reader_error": str(e), "_type": type(e).__name__})

        reader_task = asyncio.create_task(_reader())

        # Loom: subscribe to ``LoomMutated`` for the connection's
        # lifetime and forward as ``tree_mutated`` frames.  Also tag
        # ``begin_assistant`` events into ``node_created`` so the client
        # can pre-allocate render slots before token frames arrive.  Held
        # in a queue + forwarder task so the EventBus callback (which
        # runs on the gen thread) never touches the WS directly.
        loop = asyncio.get_running_loop()
        tree_event_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        # ``websocket.send_json`` is not safe for concurrent callers —
        # starlette serializes per-call but two tasks can interleave
        # bytes on the wire and corrupt the frame sequence.  This lock
        # is the single send-side serializer the connection uses; both
        # the generate-handler and the tree-forwarder acquire it before
        # every send.
        ws_send_lock = asyncio.Lock()

        async def _send_json(payload: Any) -> None:
            async with ws_send_lock:
                await websocket.send_json(payload)

        def _queue_tree_event(payload: dict[str, Any]) -> None:
            with suppress(Exception):
                loop.call_soon_threadsafe(tree_event_queue.put_nowait, payload)

        def _on_loom_event(event: object) -> None:
            if not isinstance(event, LoomMutated):
                return
            try:
                tree = session.tree
                added_nodes = [
                    node_json(session, nid)
                    for nid in event.added
                    if tree.has(nid)
                ]
            except Exception:
                added_nodes = []
            mutated_payload: dict[str, Any] = {
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
            }
            _queue_tree_event(mutated_payload)
            # ``begin_assistant`` and ``branch`` both materialize a new
            # node — surface a separate ``node_created`` event with the
            # parent + role so the client can allocate a render slot
            # without waiting for the assistant text to start streaming.
            if event.op in ("begin_assistant", "branch", "add_user"):
                for nid in event.added:
                    try:
                        node = session.tree.get(nid)
                    except Exception:
                        continue
                    node_payload = {
                        "type": "node_created",
                        "node_id": nid,
                        "parent_id": node.parent_id,
                        "role": node.role,
                        "rev": event.rev,
                    }
                    _queue_tree_event(node_payload)

        loom_unsub = session.events.subscribe(_on_loom_event)

        async def _tree_forwarder():
            """Forward tree-mutated / node-created events as WS frames.

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
            except asyncio.CancelledError:
                return

        forwarder_task = asyncio.create_task(_tree_forwarder())

        deferred_incoming: deque[Any] = deque()

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
                if msg is _DISCONNECT:
                    raise WebSocketDisconnect(code=1000)
                if isinstance(msg, dict) and "_reader_error" in msg:
                    raise RuntimeError(msg["_reader_error"])

                mtype = msg.get("type") if isinstance(msg, dict) else None
                if mtype == "generate":
                    try:
                        parsed = WSGenerateMessage(**msg)
                    except Exception as e:
                        await _send_json({
                            "type": "error",
                            "message": f"invalid generate message: {e}",
                            "code": "ValidationError",
                        })
                        continue
                    await _ws_handle_generate(
                        session, parsed, app.state.default_steering, incoming,
                        deferred_incoming, _send_json,
                    )
                elif mtype == "stop":
                    # Idle-state stop: nothing in flight.
                    continue
                else:
                    await _send_json({
                        "type": "error",
                        "message": f"unknown message type: {mtype!r}",
                        "code": "UnknownMessageType",
                    })
        except WebSocketDisconnect:
            # Ensure any stray generation is signaled.
            _stop_session_safely()
            return
        except Exception as e:
            try:
                await _send_json({
                    "type": "error",
                    "message": str(e),
                    "code": type(e).__name__,
                })
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


async def _ws_handle_generate(
    session: SaklasSession,
    msg: WSGenerateMessage,
    default_steering: "Steering | None",
    incoming: asyncio.Queue[Any],
    deferred_incoming: "deque[Any]",
    send_json: Callable[[Any], Awaitable[None]],
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

    **Loom (v2.3)**: ``parent_node_id`` attaches the assistant node
    under a specific tree node; ``n>1`` fans out N siblings serially
    (per decision 7 in the plan — N-way gen is serial in v1).  Each
    sibling produces its own ``started`` / token-stream / ``done``
    triplet, all tagged with the assistant node id.  ``tree_mutated``
    and ``node_created`` events ride the connection-level subscription
    in ``session_stream``; this handler only emits the per-sibling
    ``started`` / ``token`` / ``done`` frames.
    """
    loop = asyncio.get_running_loop()

    sampling = build_sampling(msg.sampling)
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
        await send_json({
            "type": "error",
            "message": message,
            "code": type(e).__name__,
            "status": status,
        })
        return

    n = msg.n
    if n < 1:
        await send_json({
            "type": "error",
            "message": f"n must be >= 1, got {n}",
            "code": "ValueError",
            "status": 400,
        })
        return

    parent_node_id = msg.parent_node_id

    # Logit fork: when ``fork_node_id`` is set the worker calls
    # ``session.fork_from_token`` instead of ``session.generate``.  All
    # three fork fields must be present together.
    is_fork = msg.fork_node_id is not None
    if is_fork and (msg.fork_raw_index is None or msg.fork_alt_token_id is None):
        await send_json({
            "type": "error",
            "message": (
                "fork requires fork_node_id, fork_raw_index, and "
                "fork_alt_token_id together"
            ),
            "code": "ValueError",
            "status": 400,
        })
        return

    # Answer-prefill: when ``prefill_node_id`` is set the worker calls
    # ``session.prefill_assistant`` instead of ``session.generate``.  It
    # needs ``prefill_text`` alongside it, and can't co-exist with a fork.
    is_prefill = msg.prefill_node_id is not None
    if is_prefill and (msg.prefill_text is None or msg.prefill_text == ""):
        await send_json({
            "type": "error",
            "message": "prefill requires prefill_node_id and prefill_text together",
            "code": "ValueError",
            "status": 400,
        })
        return
    if is_prefill and is_fork:
        await send_json({
            "type": "error",
            "message": "a generate message cannot be both a fork and a prefill",
            "code": "ValueError",
            "status": 400,
        })
        return

    # Commit (Ctrl+Enter on either surface): land a turn under
    # ``parent_node_id`` without running a decode.  Short-circuits the
    # n-way fan-out / streaming worker entirely — one mutation, one
    # ``done`` event, no token frames.  Mutually exclusive with prefill
    # and fork (rejected above by symmetry).
    is_commit = msg.commit_role is not None
    if is_commit:
        if msg.commit_text is None or msg.commit_text == "":
            await send_json({
                "type": "error",
                "message": "commit requires commit_role and commit_text together",
                "code": "ValueError",
                "status": 400,
            })
            return
        if msg.commit_role not in ("user", "assistant"):
            await send_json({
                "type": "error",
                "message": (
                    f"commit_role must be 'user' or 'assistant', "
                    f"got {msg.commit_role!r}"
                ),
                "code": "ValueError",
                "status": 400,
            })
            return
        if is_fork or is_prefill:
            await send_json({
                "type": "error",
                "message": (
                    "a generate message cannot mix commit with fork or prefill"
                ),
                "code": "ValueError",
                "status": 400,
            })
            return
        if msg.commit_role == "assistant" and parent_node_id is None:
            await send_json({
                "type": "error",
                "message": (
                    "commit_role='assistant' requires parent_node_id "
                    "(the user node the authored turn hangs off)"
                ),
                "code": "ValueError",
                "status": 400,
            })
            return

        generation_id = uuid.uuid4().hex[:12]
        commit_text = str(msg.commit_text)
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
        commit_user_role = (
            msg.sampling.user_role if msg.sampling is not None else None
        ) or None
        commit_asst_role = (
            msg.sampling.assistant_role if msg.sampling is not None else None
        ) or None
        async with acquire_session_lock(session) as acquired:
            if not acquired:
                await send_json({
                    "type": "error",
                    "message": "session locked — try again when the current generation finishes",
                    "code": "SessionLocked",
                    "status": 503,
                    "node_id": None,
                    "sibling_index": 0,
                })
                return
            try:
                if msg.commit_role == "user":
                    # ``raw`` flags a flat (base-model) commit — the
                    # authored span may hang under a node of any role,
                    # so the user-under-user guard is lifted.
                    new_id = await asyncio.to_thread(
                        session.append_user_turn,
                        parent_node_id,
                        commit_text,
                        allow_any_parent=msg.raw,
                        role_label=None if msg.raw else commit_user_role,
                    )
                else:
                    # ``parent_node_id`` is non-None here (validated above
                    # for the assistant role); narrow for the type-checker.
                    assert parent_node_id is not None
                    new_id = await asyncio.to_thread(
                        session.append_assistant_turn,
                        parent_node_id,
                        commit_text,
                        role_label=None if msg.raw else commit_asst_role,
                    )
            except SaklasError as e:
                status, message = e.user_message()
                await send_json({
                    "type": "error",
                    "message": message,
                    "code": type(e).__name__,
                    "status": status,
                    "node_id": None,
                    "sibling_index": 0,
                })
                return
        await send_json({
            "type": "done",
            "result": {
                "kind": "commit",
                "role": msg.commit_role,
                "text": commit_text,
                "node_id": new_id,
                "finish_reason": "stop",
                "per_token_probes": [],
                "mean_logprob": None,
                "mean_surprise": None,
            },
            "node_id": new_id,
            "sibling_index": 0,
            "sibling_count": 1,
        })
        return

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
            await send_json({
                "type": "error",
                "message": "session locked — try again when the current generation finishes",
                "code": "SessionLocked",
                "status": 503,
                "node_id": None,
                "sibling_index": 0,
            })
            return
        for sibling_idx, seed_i in enumerate(seeds):
            generation_id = uuid.uuid4().hex[:12]

            # Per-sibling sampling override carrying the derived seed.
            if n == 1 and seed_i is None:
                per_sibling_sampling = sampling
            else:
                from dataclasses import replace as _dc_replace
                base_sc = sampling if sampling is not None else SamplingConfig()
                per_sibling_sampling = _dc_replace(base_sc, seed=seed_i)

            token_queue: asyncio.Queue[Any] = asyncio.Queue()
            _SENTINEL = object()
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
                top_alts: list[Any] | None,
                perplexity: float | None = None,
                _node_holder: list[str | None] = current_node_holder,
                _token_queue: asyncio.Queue[Any] = token_queue,
            ) -> None:
                event = build_token_event(
                    session,
                    _node_holder,
                    text=text,
                    is_thinking=is_thinking,
                    tid=tid,
                    lp=lp,
                    top_alts=top_alts,
                )
                loop.call_soon_threadsafe(_token_queue.put_nowait, event)
            _on_token_flags: Any = _on_token
            _on_token_flags._saklas_wants_live_scores = True
            _on_token_flags._saklas_wants_per_layer_scores = True
            # Live J-lens workspace readout: computed only when the session's
            # live lens is enabled (POST .../lens/live) AND the tap consumer
            # declares interest — this stamp is the declaration (mirrors
            # ``generate_stream``'s ``_push_flags``), so an enabled lens
            # streams per-step top-k on the ``token`` frame's ``lens_readout``.
            _on_token_flags._saklas_wants_lens_readout = True
            _on_token_flags._saklas_wants_sae_readout = True

            result_holder: list[GenerationResult | RunSet] = []
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
                _on_token: Callable[..., Any] = _on_token,
                _result_holder: list[GenerationResult | RunSet] = result_holder,
                _error_holder: list[BaseException] = error_holder,
                _token_queue: asyncio.Queue[Any] = token_queue,
                _sentinel: object = _SENTINEL,
                _recipe_override: Any = recipe_override,
            ) -> None:
                try:
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
                            "parent_node_id": parent_node_id,
                        }
                        if _recipe_override is not None:
                            gen_kwargs["recipe_override"] = _recipe_override
                        if msg.generate_seat is not None:
                            gen_kwargs["gen_seat"] = msg.generate_seat
                        result = session.generate(msg.input, **gen_kwargs)
                    _result_holder.append(result)
                except BaseException as e:
                    _error_holder.append(e)
                finally:
                    loop.call_soon_threadsafe(_token_queue.put_nowait, _sentinel)

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
            try:
                while not done:
                    token_get = asyncio.create_task(token_queue.get())
                    client_get = asyncio.create_task(incoming.get())
                    finished, pending = await asyncio.wait(
                        {token_get, client_get}, return_when=asyncio.FIRST_COMPLETED,
                    )
                    if client_get in finished:
                        incoming_msg = client_get.result()
                        # ``_DISCONNECT`` / reader-error sentinels:
                        # signal the worker to wind down; let the outer
                        # loop propagate the disconnect on the next
                        # iteration.
                        if isinstance(incoming_msg, dict):
                            if incoming_msg.get("type") == "stop":
                                _stop_session_safely()
                                stop_signaled = True
                            elif "_reader_error" in incoming_msg:
                                _stop_session_safely()
                                stop_signaled = True
                                # Defer so the outer dispatch loop
                                # surfaces the error after we wind down.
                                deferred_incoming.append(incoming_msg)
                            else:
                                # Out-of-band frame during a generation —
                                # defer so the outer loop sees it
                                # after this turn finishes.  Most likely
                                # an early ``{type: "generate"}`` from a
                                # client that didn't wait for ``done``.
                                #
                                # Do not put it back on ``incoming`` here:
                                # the next loop iteration would consume it
                                # immediately, cancel token_queue.get(), and
                                # spin until the worker happened to have a
                                # token already queued.
                                deferred_incoming.append(incoming_msg)
                        else:
                            # Disconnect sentinel from the reader.
                            _stop_session_safely()
                            stop_signaled = True
                            deferred_incoming.append(incoming_msg)
                    if token_get in finished:
                        item = token_get.result()
                        if item is _SENTINEL:
                            done = True
                        else:
                            await send_json(item)
                    for task in pending:
                        task.cancel()
                    if pending:
                        await asyncio.gather(*pending, return_exceptions=True)
            finally:
                # Drain any residual events the worker pushed between
                # sentinel and join — should be none because the
                # sentinel is last, but cheap insurance.
                await worker_task

            if error_holder and not result_holder:
                exc = error_holder[0]
                await send_json({
                    "type": "error",
                    "message": str(exc),
                    "code": type(exc).__name__,
                    "node_id": current_node_holder[0],
                    "sibling_index": sibling_idx,
                })
                # On error inside a sibling, abort the remaining fan-out
                # rather than continuing with stale state.
                return

            result = result_holder[0] if result_holder else None
            result_json = result_to_json(result)
            if result is not None:
                result_json["per_token_probes"] = per_token_probes(
                    session, getattr(result, "token_count", 0) or 0,
                )
                # Per-attached-manifold-probe aggregate readings ride on
                # the ``done`` event so a WS client picks up the
                # geometric channel alongside the existing vector-probe
                # ``per_token_probes`` block.  Empty dict when no
                # manifold probe is attached.  Shared with the SSE / NDJSON
                # finalization via ``probe_reading_aggregate`` (result-
                # parameterized so each n>1 sibling scores its own result).
                mf_readings = probe_reading_aggregate(session, result)
                if mf_readings:
                    result_json["probe_readings"] = mf_readings
            else:
                result_json["per_token_probes"] = []
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
                try:
                    node = session.tree.nodes.get(finalized_node_id)
                    if node is not None:
                        mean_logprob_out = node.mean_logprob
                        mean_surprise_out = node.mean_surprise
                except Exception:
                    # Defensive: tree access during shutdown / mocked
                    # session edge cases. Default-None values keep the
                    # wire payload well-formed.
                    pass
            result_json["mean_logprob"] = mean_logprob_out
            result_json["mean_surprise"] = mean_surprise_out
            await send_json({
                "type": "done",
                "result": result_json,
                "node_id": current_node_holder[0],
                "sibling_index": sibling_idx,
                "sibling_count": n,
            })

            # Mid-batch stop honors the plan's decision (#7 / phase 1
            # spec): "stop_requested cancels the currently-streaming
            # sibling. Remaining queued siblings are skipped, not
            # started."
            if stop_signaled:
                break
