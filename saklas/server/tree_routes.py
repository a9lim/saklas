"""Native loom-tree route group (``/saklas/v1/sessions/{id}/tree/*``)."""

# pyright: reportUnusedFunction=false

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response

from saklas.server.app import acquire_session_lock
from saklas.server.native_common import resolve_session_id
from saklas.server.tree_models import (
    CastMemberRequest,
    JointLogprobsRequest,
    TreeBranchRequest,
    TreeDiffRequest,
    TreeEditRequest,
    TreeNavigateRequest,
    TreeNoteRequest,
    TreeStarRequest,
    TreeTranscriptLoadRequest,
    TreeTranscriptRequest,
    active_path_json,
    node_json,
    tree_to_json,
)


def register_tree_routes(app: FastAPI) -> None:
    """Mount the loom ``tree/*`` routes (v2.3 phase 2)."""
    session = app.state.session

    @app.get("/saklas/v1/sessions/{session_id}/tree")
    def get_tree(session_id: str):
        """Full tree as JSON.

        Same shape :meth:`LoomTree.to_dict` produces. Surfaces hydrate
        their state from this on bootstrap and reconcile via the WS
        ``tree_mutated`` delta stream after.
        """
        resolve_session_id(session_id)
        return tree_to_json(session)

    @app.get("/saklas/v1/sessions/{session_id}/tree/active")
    def get_tree_active(session_id: str):
        """Active path: chat messages + parallel node-id list.

        Cheaper than the full tree for surfaces that only need the
        currently-rendered conversation. The node-id list is parallel to
        ``messages`` so a click on message ``i`` maps to ``node_ids[i]``.
        """
        resolve_session_id(session_id)
        return active_path_json(session)

    @app.post("/saklas/v1/sessions/{session_id}/tree/navigate")
    async def tree_navigate(session_id: str, req: TreeNavigateRequest):
        """Re-point the active node.

        Free relative to in-flight generation (per the concurrency
        invariant in the plan): the gen continues attached to its
        original target, the user simply sees a different active path.
        """
        resolve_session_id(session_id)
        session.tree.navigate(req.node_id)
        return active_path_json(session)

    @app.post("/saklas/v1/sessions/{session_id}/tree/edit")
    async def tree_edit(session_id: str, req: TreeEditRequest):
        """In-place text replacement.

        409 when the node is in the reservation of an in-flight
        generation (mapped via ``SaklasError.user_message``); 404 on
        unknown id; 400 on root-edit or other invalid ops.
        """
        resolve_session_id(session_id)
        session.tree.edit(req.node_id, req.text)
        return node_json(session, req.node_id)

    @app.post("/saklas/v1/sessions/{session_id}/tree/branch")
    async def tree_branch(session_id: str, req: TreeBranchRequest):
        """Always-sibling — create a new node next to ``node_id``.

        Allowed during in-flight generation; the new sibling sits on the
        same user-parent as the gen target without disturbing it.
        Returns ``{node_id, node, active_path}`` so the caller can place
        the new node and (if it became active) re-render the chat
        without a follow-up fetch.
        """
        resolve_session_id(session_id)
        new_id = session.tree.branch(
            req.node_id, req.text, role=req.role,
        )
        return {
            "node_id": new_id,
            "node": node_json(session, new_id),
            "active_path": active_path_json(session),
        }

    @app.delete("/saklas/v1/sessions/{session_id}/tree/{node_id}")
    async def tree_delete(session_id: str, node_id: str):
        """Subtree delete.

        400 for the root delete; 409 when the subtree intersects an
        in-flight generation's reservation; 404 on unknown id.  When
        the active node sits inside the deleted subtree, the engine
        re-seats the active pointer on the surviving parent and emits
        the new ``active_node_id`` on the mutation event.  Returns
        ``{removed: <count>}``.
        """
        resolve_session_id(session_id)
        removed = session.tree.delete_subtree(node_id)
        return {"removed": removed}

    @app.post("/saklas/v1/sessions/{session_id}/tree/star")
    async def tree_star(session_id: str, req: TreeStarRequest):
        """Toggle a node's ``starred`` flag.

        Decoration-only; never raises a concurrency conflict.
        """
        resolve_session_id(session_id)
        session.tree.star(req.node_id, req.on)
        return node_json(session, req.node_id)

    @app.post("/saklas/v1/sessions/{session_id}/tree/note")
    async def tree_note(session_id: str, req: TreeNoteRequest):
        """Set a node's free-text ``notes`` annotation.

        Decoration-only; never raises a concurrency conflict.
        """
        resolve_session_id(session_id)
        session.tree.annotate(req.node_id, req.text)
        return node_json(session, req.node_id)

    @app.get("/saklas/v1/sessions/{session_id}/tree/cast")
    def tree_cast(session_id: str):
        """The tree's cast roster: label → member (recipe + notes).

        Also rides the full-tree GET (``cast`` key when non-empty) and
        the ``op="cast"`` ``tree_mutated`` frame — this endpoint is the
        cheap standalone read.
        """
        resolve_session_id(session_id)
        return {
            "cast": {
                label: member.to_dict()
                for label, member in session.tree.cast.items()
            }
        }

    @app.put("/saklas/v1/sessions/{session_id}/tree/cast/{label}")
    async def tree_cast_put(session_id: str, label: str, req: CastMemberRequest):
        """Create or replace the cast member under ``label``.

        Validates the label as a role slug and the steering expression's
        syntax up front (400 via ``SaklasError.user_message`` on either).
        Decoration-tier — never a concurrency conflict.
        """
        resolve_session_id(session_id)
        member = session.set_cast_member(
            label,
            steering=req.steering,
            thinking=req.thinking,
            seed=req.seed,
            notes=req.notes,
        )
        return {"label": label, "member": member.to_dict()}

    @app.delete(
        "/saklas/v1/sessions/{session_id}/tree/cast/{label}", status_code=204,
    )
    async def tree_cast_delete(session_id: str, label: str):
        """Drop the cast member under ``label`` (no-op when absent)."""
        resolve_session_id(session_id)
        session.remove_cast_member(label)
        return None

    @app.post("/saklas/v1/sessions/{session_id}/tree/reset", status_code=204)
    async def tree_reset(session_id: str):
        """Drop the entire tree and rebuild a fresh root.

        Equivalent to ``session.clear_history()``; 409 when a generation
        is in flight (per the concurrency invariant — ``reset`` cannot
        race the gen path because the gen path owns the streaming target
        in the tree itself).
        """
        resolve_session_id(session_id)
        async with acquire_session_lock(session) as acquired:
            if not acquired:
                raise HTTPException(503, "session locked")
            session.clear_history()
        return Response(status_code=204)

    @app.post("/saklas/v1/sessions/{session_id}/tree/transcript")
    def tree_transcript(session_id: str, req: TreeTranscriptRequest):
        """Render the path ending at ``node_id`` (or active) as transcript YAML.

        Phase 5 producer: uses :meth:`Transcript.from_path` so probe
        sha256 hashes are real and the YAML round-trips through
        :meth:`Transcript.from_yaml` cleanly.  Returns
        ``{"yaml": "<text>", "node_id": "<leaf-of-rendered-path>"}``.
        """
        from saklas.core.transcript import Transcript

        resolve_session_id(session_id)
        leaf = req.node_id if req.node_id is not None else session.tree.active_node_id
        # Validate the id before touching the renderer so the 404 lands
        # cleanly through the existing ``SaklasError`` handler.
        session.tree.get(leaf)
        transcript = Transcript.from_path(leaf, session)
        return {"yaml": transcript.to_yaml(), "node_id": leaf}

    @app.post("/saklas/v1/sessions/{session_id}/tree/transcript/load")
    async def tree_transcript_load(
        session_id: str, req: TreeTranscriptLoadRequest,
    ):
        """Import a transcript YAML into the live session tree (phase 5).

        Wraps :meth:`Transcript.from_yaml` + :meth:`Transcript.import_into`.
        Modes are ``"default"`` / ``"here"`` / ``"merge"``; ``strict``
        refuses on probe-hash drift.  Returns
        ``{"leaf_id": "<id>", "rev": <int>, "guards": [...]}``.

        Guards (model mismatch, system-prompt mismatch, probe drift) are
        also stamped on the imported branch's root node as ``notes`` so
        the surfaces can show a banner there.  Returning them in the body
        too saves the client one fetch.
        """
        from saklas.core.transcript import (
            Transcript,
            TranscriptError,
            TranscriptFormatError,
        )

        resolve_session_id(session_id)
        mode = req.mode
        try:
            transcript = Transcript.from_yaml(req.yaml)
        except TranscriptFormatError as e:
            raise HTTPException(400, f"invalid transcript: {e}") from e
        import warnings

        captured: list[str] = []

        def _on_warning(
            message: Warning | str,
            category: type[Warning],
            filename: str,
            lineno: int,
            file: Any = None,
            line: str | None = None,
        ) -> None:
            captured.append(str(message))

        async with acquire_session_lock(session) as acquired:
            if not acquired:
                raise HTTPException(503, "session locked")
            with warnings.catch_warnings():
                warnings.showwarning = _on_warning
                try:
                    leaf_id = await asyncio.to_thread(
                        transcript.import_into,
                        session,
                        mode=mode,
                        strict=req.strict,
                    )
                except TranscriptError as e:
                    raise HTTPException(400, str(e)) from e
        return {
            "leaf_id": leaf_id,
            "rev": session.tree.rev,
            "guards": captured,
        }

    @app.get("/saklas/v1/sessions/{session_id}/tree/edge_label")
    def tree_edge_label(session_id: str, parent_id: str, child_id: str):
        """Steering-delta label for the parent → child edge (phase 5).

        Returns ``{"label": "<text>"}`` — empty string when the two
        recipes are identical.  Both nodes must exist; the label is
        computed from the canonical ``applied_steering`` strings on the
        parent's and child's recipes (parent's may be ``None`` when it's
        a user turn, in which case the delta is "from-nothing").
        """
        from saklas.core.loom_diff import steering_delta

        resolve_session_id(session_id)
        parent = session.tree.get(parent_id)
        child = session.tree.get(child_id)
        parent_expr = parent.applied_steering
        if parent_expr is None and parent.recipe is not None:
            parent_expr = parent.recipe.steering
        child_expr = child.applied_steering
        if child_expr is None and child.recipe is not None:
            child_expr = child.recipe.steering
        return {"label": steering_delta(parent_expr, child_expr)}

    @app.get("/saklas/v1/sessions/{session_id}/tree/filter")
    def tree_filter(session_id: str, expr: str = ""):
        """Apply a filter-grammar expression and return matching node ids.

        Grammar in :mod:`saklas.core.tree_filter` — comma-AND'd
        ``agg:|any:|last:<probe> <op> <threshold>`` clauses.  Empty
        ``expr`` returns every node id (clears the filter).  Bad
        expressions land as 400 via :class:`FilterParseError`.
        """
        from saklas.core.tree_filter import FilterParseError

        resolve_session_id(session_id)
        text = (expr or "").strip()
        if not text:
            return {"expr": "", "matching_node_ids": []}
        try:
            matches = session.tree.filter_by_expr(text)
        except FilterParseError as e:
            raise HTTPException(400, str(e)) from e
        return {"expr": text, "matching_node_ids": sorted(matches)}

    @app.post("/saklas/v1/sessions/{session_id}/tree/diff")
    def tree_diff(session_id: str, req: TreeDiffRequest):
        """Cross-branch diff between two assistant nodes (phase 5).

        Returns a JSON view of :class:`NodeDiff` (text spans + readings
        deltas) augmented with the parent-recipe steering delta and any
        per-token deltas available from the session's
        ``last_per_token_scores`` — the per-token table is only present
        for the most-recently-generated assistant so callers shouldn't
        rely on it.
        """
        from saklas.core.loom_diff import per_token_diff, steering_delta

        resolve_session_id(session_id)
        diff = session.diff_nodes(req.a_id, req.b_id)
        a_node = session.tree.get(req.a_id)
        b_node = session.tree.get(req.b_id)

        # Steering-delta against the shared parent's expression — only
        # meaningful for sibling diffs (parent_id present).
        parent_expr: str | None = None
        if diff.parent_id is not None:
            parent = session.tree.nodes.get(diff.parent_id)
            if parent is not None:
                parent_expr = parent.applied_steering
                if parent_expr is None and parent.recipe is not None:
                    parent_expr = parent.recipe.steering

        a_expr = a_node.applied_steering or (
            a_node.recipe.steering if a_node.recipe else None
        )
        b_expr = b_node.applied_steering or (
            b_node.recipe.steering if b_node.recipe else None
        )

        # Per-token diff: only when both nodes carry token sequences.
        # Tokens may be absent on serialized-only nodes (loaded transcripts).
        a_tok_strs: list[str] = []
        if a_node.tokens:
            a_tok_strs = [t.get("text", "") for t in a_node.tokens]
        b_tok_strs: list[str] = []
        if b_node.tokens:
            b_tok_strs = [t.get("text", "") for t in b_node.tokens]
        per_token_spans: list[dict[str, Any]] = []
        if a_tok_strs and b_tok_strs:
            spans = per_token_diff(a_tok_strs, b_tok_strs)
            per_token_spans.extend(
                {
                    "a_index": sp.a_index,
                    "b_index": sp.b_index,
                    "a_text": sp.a_text,
                    "b_text": sp.b_text,
                    "aligned": sp.aligned,
                    "reading_deltas": [
                        {
                            "name": rd.name,
                            "delta": round(float(rd.delta), 6),
                            "a_value": round(float(rd.a_value), 6),
                            "b_value": round(float(rd.b_value), 6),
                        }
                        for rd in sp.reading_deltas
                    ],
                }
                for sp in spans
            )

        return {
            "a_id": diff.a_id,
            "b_id": diff.b_id,
            "parent_id": diff.parent_id,
            "a_text": a_node.text,
            "b_text": b_node.text,
            "a_applied_steering": a_expr,
            "b_applied_steering": b_expr,
            "parent_applied_steering": parent_expr,
            "steering_delta": steering_delta(a_expr, b_expr),
            "parent_to_a_delta": (
                steering_delta(parent_expr, a_expr)
                if parent_expr is not None or a_expr is not None
                else ""
            ),
            "parent_to_b_delta": (
                steering_delta(parent_expr, b_expr)
                if parent_expr is not None or b_expr is not None
                else ""
            ),
            "text": [
                {"state": sp.state, "text": sp.text}
                for sp in diff.text
            ],
            "readings": [
                {
                    "name": rd.name,
                    "delta": round(float(rd.delta), 6),
                    "a_value": round(float(rd.a_value), 6),
                    "b_value": round(float(rd.b_value), 6),
                }
                for rd in diff.readings
            ],
            "per_token": per_token_spans,
        }

    @app.post("/saklas/v1/sessions/{session_id}/tree/joint_logprobs")
    async def tree_joint_logprobs(session_id: str, req: JointLogprobsRequest):
        """Cross-evaluation between two sibling assistant nodes.

        Logit-pass Phase 5 of ``docs/plans/logit-pass.md``.  Force-replays
        each branch under the node's stamped recipe, steering hooks, probe
        gates, penalties, logit bias, and sampler transform, then returns
        per-aligned-position records carrying both branches' chosen-token
        logprobs *and* the cross-branch evaluation (what each side would
        have given the other's chosen token at the same byte-aligned
        position).

        Cache shape:
        * Stored on ``session.joint_logprob_cache: dict[tuple[str,
          str], JointLogprobs]`` keyed by sorted ``(a_id, b_id)`` so
          the symmetric pair shares an entry.
        * Invalidated by tree edits/deletes/finalize events in
          ``SaklasSession``; navigate/star/note leave it intact.

        Held under ``acquire_session_lock`` because the forward passes
        compete for the same model with any concurrent generation;
        request queues FIFO at the lock rather than 409ing.
        """
        from saklas.core.joint_logprobs import (
            compute_joint_logprobs,
            _cache_key,
            reorient_for_request,
        )

        resolve_session_id(session_id)
        if req.a_id == req.b_id:
            raise HTTPException(400, "a_id and b_id must differ")
        if req.a_id not in session.tree.nodes:
            raise HTTPException(404, f"unknown node id: {req.a_id}")
        if req.b_id not in session.tree.nodes:
            raise HTTPException(404, f"unknown node id: {req.b_id}")

        cache = session.joint_logprob_cache

        key = _cache_key(req.a_id, req.b_id)
        hit = cache.get(key)
        if hit is None:
            async with acquire_session_lock(session):
                # Double-check under lock — another request may have
                # populated the cache while we waited.
                hit = cache.get(key)
                if hit is None:
                    hit = await asyncio.to_thread(
                        compute_joint_logprobs, session, req.a_id, req.b_id,
                    )
                    cache[key] = hit
        return reorient_for_request(hit, req.a_id, req.b_id).to_dict()
