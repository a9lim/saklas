"""Tests for the loom-tree HTTP / WS surface (phase 2).

The session under test is a thin wrapper around a real :class:`LoomTree`
plus a generation stub — no model, no GPU.  This exercises:

- REST routes under ``/saklas/v1/sessions/{id}/tree`` (CRUD, transcript)
- Concurrency conflict 409 mapping when ``_active_gen_reservation`` is set
- WS ``parent_node_id`` + ``n>1`` fan-out (siblings created, started/done
  tagged with ``node_id`` and complete ``tree_mutated`` events
  fire)

Heavy generation paths are stubbed so the assertions stay focused on the
tree wiring and protocol shapes phase 2 introduced.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from saklas.core.events import EventBus
from saklas.core.loom import LoomTree, Recipe
from saklas.core.results import GenerationResult, RunSet

# ---------------------------------------------------------------------------
# Test session factory
# ---------------------------------------------------------------------------


class _StubSession:
    """Minimal SaklasSession-shaped object backing the routes.

    Wraps a real :class:`LoomTree` so the routes operate on actual tree
    semantics (rev bumps, event emission, conflict-check on mutators).
    ``generate`` is the only generation entry point: it walks the
    sibling loop, calls ``tree.begin_assistant`` + ``tree.append_token``
    + ``tree.finalize_assistant`` per sibling, and emits tokens through
    the supplied ``on_token`` callback.
    """

    def __init__(self) -> None:
        self.model_id = "test/model"
        self.model_info = {
            "model_type": "gemma2",
            "num_layers": 4,
            "hidden_dim": 16,
            "device": "cpu",
            "dtype": "torch.float32",
        }
        self._device = "cpu"
        self._dtype = "torch.float32"
        self._created_ts = 1_700_000_000

        self.events = EventBus()
        self.tree = LoomTree(
            events=self.events,
            model_id=self.model_id,
            conflict_check=self._loom_conflict_check,
        )
        self._active_gen_reservation: str | None = None

        cfg = MagicMock()
        cfg.temperature = 1.0
        cfg.top_p = 0.9
        cfg.top_k = None
        cfg.max_new_tokens = 64
        cfg.system_prompt = "You are a stub."
        self.config = cfg

        self.vectors: dict[str, Any] = {}
        self.probes: dict[str, Any] = {}

        monitor = MagicMock()
        monitor.probe_names = []
        monitor.profiles = {}
        self.monitor = monitor
        self._monitor = monitor
        self._joint_logprob_cache: dict[Any, Any] = {}
        self.lens_probe_names: list[str] = []
        self.sae_probe_names: list[str] = []
        self.token_probe_payload: dict[str, Any] = {}
        self._tokenizer = MagicMock()
        self._tokenizer.encode.side_effect = lambda text, **_: [
            3000 + i for i, _ch in enumerate(str(text))
        ]
        self._layers = []
        capture = MagicMock()
        capture._per_layer = {}
        self._capture = capture
        self._last_per_token_scores = None
        self._last_result = None
        self.last_per_token_scores = None
        self.last_result = None

        gen_state = MagicMock()
        gen_state.finish_reason = "stop"
        self.generation_state = gen_state

        self.lock = asyncio.Lock()

        # Trait queue infrastructure (used by SSE traits/stream endpoint).
        self._trait_queues = []
        self._trait_lock = threading.Lock()

        self._next_token_stream: list[str] = ["hi"]
        self._fail_next: bool = False
        self._block_until_stop = False
        self._stop_event = threading.Event()

    def clear_history(self) -> None:
        self.tree.reset()

    def restore_tree(self, data: dict[str, Any]):
        from saklas.core.session import SaklasSession
        return SaklasSession.restore_tree(self, data)  # type: ignore[arg-type]

    def rewind(self) -> None:
        self.tree.rewind()

    def build_readings(self):  # pragma: no cover - unused by these tests
        return {}

    def probe_hashes(self) -> dict[str, str]:
        return {}

    def register_trait_queue(self, loop: Any, q: Any) -> None:
        with self._trait_lock:
            self._trait_queues.append((loop, q))

    def unregister_trait_queue(self, loop: Any, q: Any) -> None:
        with self._trait_lock:
            try:
                self._trait_queues.remove((loop, q))
            except ValueError:
                pass

    def stop(self) -> None:
        self._stop_event.set()

    # ----- loom conflict check (mirrors SaklasSession._loom_conflict_check)
    def _loom_conflict_check(self, node_id: str, op: str) -> None:
        from saklas.core.loom import MutationDuringGenerationError
        reservation = self._active_gen_reservation
        if reservation is None:
            return
        if op in (
            "add_user_turn", "begin_assistant", "finalize_assistant",
            "branch", "star", "note", "navigate",
        ):
            return
        if op == "reset":
            raise MutationDuringGenerationError(
                "cannot reset tree while a generation is in flight"
            )
        if (
            node_id == reservation
            or self.tree.is_ancestor_of(reservation, node_id)
            or self.tree.is_ancestor_of(node_id, reservation)
        ):
            raise MutationDuringGenerationError(
                f"cannot {op} on a node inside an in-flight generation's "
                f"reservation (reservation root: {reservation})"
            )

    # ----- generation entry point --------------------------------------
    def generate(self, input: Any, *, steering: Any = None, sampling: Any = None,
                 stateless: bool = False, raw: bool = False, thinking: Any = None,
                 on_token: Any = None, parent_node_id: Any = None, n: int = 1,
                 gen_seat: str = "assistant", recipe_override: Any = None,
                 append_same_role: bool = True):
        """Stub generate.

        Routes through the tree the same way SaklasSession does for
        phase-2's WS plumbing to see the right LoomMutated events fire.
        Each sibling emits one synthetic token, finalizes, and produces
        a :class:`GenerationResult`.
        """
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        from saklas.core.loom import derive_seed_schedule
        base_seed = sampling.seed if sampling is not None else None
        schedule = derive_seed_schedule(base_seed, n) if n > 1 else [base_seed]

        results = []
        for sibling_idx, seed_i in enumerate(schedule):
            # Text input retains the compatibility generate contract (a user
            # commit); ``None`` is the native bare continuation used by
            # seat-neutral submit after it commits explicitly.
            if input is None:
                anchor_id = parent_node_id or self.tree.active_node_id
            else:
                anchor_id = self.tree.add_user_turn(
                    str(input), parent_id=parent_node_id,
                )
            self._active_gen_reservation = anchor_id
            role_label = None
            if sampling is not None and not raw:
                role_label = (
                    sampling.user_role
                    if gen_seat == "user"
                    else sampling.assistant_role
                )
            anchor = self.tree.get(anchor_id)
            can_continue = bool(
                append_same_role
                and n == 1
                and input is None
                and anchor.role == gen_seat
                and anchor.role_label == role_label
            )
            prefix = anchor.text if can_continue else ""
            if can_continue:
                assistant_id = self.tree.begin_continuation(
                    anchor_id, Recipe(),
                )
            else:
                assistant_id = self.tree.begin_assistant(
                    anchor_id, recipe=Recipe(), role_label=role_label,
                    seat=gen_seat,
                )
            try:
                token_text = f"tok{sibling_idx}"
                if on_token is not None:
                    on_token(token_text, False, 1000 + sibling_idx, None, None)
                self.tree.append_token(
                    assistant_id, {"token_id": 1000 + sibling_idx, "text": token_text},
                )
                if self._block_until_stop:
                    assert self._stop_event.wait(timeout=5.0)
                full_text = prefix + token_text
                result = GenerationResult(
                    text=full_text, tokens=[1000 + sibling_idx],
                    token_count=1, tok_per_sec=10.0, elapsed=0.1,
                    finish_reason="stop",
                )
                self.tree.finalize_assistant(
                    assistant_id,
                    text=full_text,
                    aggregate_readings={},
                    applied_steering=None,
                    finish_reason="stop",
                )
                results.append(result)
                self._last_result = result
                self.last_result = result
            finally:
                self._active_gen_reservation = None
        return RunSet(results)

    # ----- answer-prefill entry point ----------------------------------
    def prefill_assistant(self, node_id: Any, text: Any, *, steering: Any = None,
                          sampling: Any = None, on_token: Any = None):
        """Stub answer-prefill.

        Mirrors ``SaklasSession.prefill_assistant``'s tree shape: anchor
        at the user node's parent so ``add_user_turn`` dedup re-uses the
        existing user turn, then land an assistant child whose text opens
        with ``text`` (the seeded prefix) followed by a synthetic
        continuation token.
        """
        from saklas.core.loom import InvalidNodeOperationError
        node = self.tree.get(node_id)
        if node.role != "user":
            raise InvalidNodeOperationError(
                f"prefill_assistant: {node_id!r} is a {node.role} node"
            )
        user_id = self.tree.add_user_turn(node.text, parent_id=node.parent_id)
        self._active_gen_reservation = user_id
        assistant_id = self.tree.begin_assistant(user_id)
        try:
            if on_token is not None:
                on_token(text, False, 2000, None, None)
                on_token("-cont", False, 2001, None, None)
            self.tree.append_token(
                assistant_id, {"token_id": 2000, "text": text},
            )
            self.tree.append_token(
                assistant_id, {"token_id": 2001, "text": "-cont"},
            )
            full = text + "-cont"
            result = GenerationResult(
                text=full, tokens=[2000, 2001], token_count=2,
                tok_per_sec=10.0, elapsed=0.1, finish_reason="stop",
            )
            self.tree.finalize_assistant(
                assistant_id, text=full, aggregate_readings={},
                applied_steering=None, finish_reason="stop",
            )
            self._last_result = result
            self.last_result = result
        finally:
            self._active_gen_reservation = None
        return result

    # ----- commit entry points (Ctrl+Enter on either surface) ----------
    def append_user_turn(
        self, parent_node_id: Any, text: Any, *, allow_any_parent: bool = False, role_label: Any = None, thinking: Any = None
    ):
        """Stub commit-user.

        Mirrors ``SaklasSession.append_user_turn``: refuses anchoring
        under a user-role parent (the same D15 rule the normal-send
        path enforces) unless ``allow_any_parent`` is set (flat /
        base-model commit), otherwise wraps ``LoomTree.add_user_turn``
        so dedup + active-node advancement match the real session.
        """
        from saklas.core.loom import InvalidNodeOperationError
        if text == "":
            raise InvalidNodeOperationError(
                "append_user_turn: text must be non-empty"
            )
        resolved_parent = (
            parent_node_id
            if parent_node_id is not None
            else self.tree.active_node_id
        )
        parent = (
            self.tree.nodes.get(resolved_parent)
            if resolved_parent is not None
            else None
        )
        if (
            parent is not None
            and parent.role == "user"
            and parent.role_label == role_label
        ):
            raw_token_ids = self._tokenizer.encode(
                parent.text + text, add_special_tokens=False,
            )
            return self.tree.append_authored(
                parent.id, text, thinking_text=thinking,
                raw_token_ids=raw_token_ids,
            )
        if not allow_any_parent and parent is not None and parent.role == "user":
            raise InvalidNodeOperationError(
                f"cannot send a new user turn from a user node "
                f"({resolved_parent}): the active turn is already "
                f"waiting for an assistant."
            )
        return self.tree.add_user_turn(
            text, parent_id=parent_node_id, role_label=role_label,
            thinking_text=thinking)

    def append_assistant_turn(self, user_node_id: Any, text: Any, *, role_label: Any = None, thinking: Any = None):
        """Stub commit-assistant.

        Mirrors ``SaklasSession.append_assistant_turn``: refuses non-user
        parents, lands a finalized assistant sibling under the user node
        with ``text`` as the whole turn and a synthetic ``raw_token_ids``
        derived from the (mocked) tokenizer.
        """
        from saklas.core.loom import InvalidNodeOperationError
        if text == "":
            raise InvalidNodeOperationError(
                "append_assistant_turn: text must be non-empty"
            )
        node = self.tree.get(user_node_id)
        if node.role == "assistant" and node.role_label == role_label:
            raw_token_ids = self._tokenizer.encode(
                node.text + text, add_special_tokens=False,
            )
            return self.tree.append_authored(
                node.id, text, thinking_text=thinking,
                raw_token_ids=raw_token_ids,
            )
        if node.role != "user" and getattr(self, "scene_grammar", None) is None:
            raise InvalidNodeOperationError(
                f"append_assistant_turn: {user_node_id!r} is a "
                f"{node.role} node, not a user node"
            )
        # Stub tokenization: one synthetic id per word so tests can
        # assert ``raw_token_ids`` was populated without a real model.
        raw_token_ids = [3000 + i for i, _ in enumerate(text.split())]
        if not raw_token_ids:
            raise InvalidNodeOperationError(
                f"append_assistant_turn: {text!r} tokenized to an "
                f"empty sequence"
            )
        new_id = self.tree.begin_assistant(
            user_node_id, recipe=None, role_label=role_label)
        authored = self.tree.nodes[new_id]
        authored.tokens = None
        authored.thinking_tokens = None
        self.tree.finalize_assistant(
            new_id,
            text=text,
            applied_steering=None,
            finish_reason="stop",
            raw_token_ids=raw_token_ids,
            thinking_text=thinking,
        )
        return new_id

    def append_turn(
        self, parent_node_id: Any, text: Any, *, role: Any,
        raw: bool = False, role_label: Any = None, thinking: Any = None,
    ):
        from saklas.core.session import SaklasSession
        return SaklasSession.append_turn(
            self,  # pyright: ignore[reportArgumentType]
            parent_node_id, text, role=role, raw=raw,
            role_label=role_label, thinking=thinking,
        )

    # Cast roster passthroughs — the real session methods only touch
    # ``self.tree``, so borrow them wholesale (same trick the loom
    # tests use for the commit methods).
    def set_cast_member(self, label: Any, **kwargs: Any):
        from saklas.core.session import SaklasSession
        return SaklasSession.set_cast_member(self, label, **kwargs)  # type: ignore[arg-type]

    def remove_cast_member(self, label: Any) -> None:
        self.tree.remove_cast_member(label)


@pytest.fixture
def session_and_client():
    from typing import cast
    from saklas.core.session import SaklasSession
    from saklas.server import create_app
    session = _StubSession()
    app = create_app(cast(SaklasSession, session), default_steering=None)
    return session, TestClient(app)


# ---------------------------------------------------------------------------
# REST: tree GET shape
# ---------------------------------------------------------------------------


class TestTreeGet:
    def test_root_only(self, session_and_client: Any):
        session, client = session_and_client
        resp = client.get("/saklas/v1/sessions/default/tree")
        assert resp.status_code == 200
        data = resp.json()
        from saklas.core.loom import TREE_FORMAT_VERSION

        assert data["tree_format"] == TREE_FORMAT_VERSION
        assert data["root_id"] == session.tree.root_id
        assert data["active_node_id"] == session.tree.root_id
        assert len(data["nodes"]) == 1
        # The synthetic root carries no recipe / text
        root_node = data["nodes"][0]
        assert root_node["role"] == "system"
        assert root_node["text"] == ""
        assert root_node["id"] == session.tree.root_id

    def test_full_tree_restore_round_trip(self, session_and_client: Any):
        session, client = session_and_client
        user_id = session.tree.add_user_turn("saved branch")
        assistant_id = session.tree.begin_assistant(user_id)
        session.tree.finalize_assistant(assistant_id, text="saved reply")
        saved = client.get("/saklas/v1/sessions/default/tree").json()
        saved_rev = saved["rev"]

        assert client.post("/saklas/v1/sessions/default/tree/reset").status_code == 204
        restored = client.put(
            "/saklas/v1/sessions/default/tree",
            json={"tree": saved},
        )
        assert restored.status_code == 200
        body = restored.json()
        assert body["nodes"] == 3
        assert body["active_node_id"] == assistant_id
        assert body["rev"] > saved_rev
        tree = client.get("/saklas/v1/sessions/default/tree").json()
        assert [node["text"] for node in tree["nodes"]] == [
            "", "saved branch", "saved reply",
        ]

    def test_full_tree_restore_rejects_model_mismatch(self, session_and_client: Any):
        _, client = session_and_client
        saved = client.get("/saklas/v1/sessions/default/tree").json()
        saved["model_id"] = "other/model"
        restored = client.put(
            "/saklas/v1/sessions/default/tree",
            json={"tree": saved},
        )
        assert restored.status_code == 400
        assert "does not match loaded model" in str(restored.json())

    def test_full_tree_restore_emits_snapshot_barrier(self, session_and_client: Any):
        session, client = session_and_client
        saved = client.get("/saklas/v1/sessions/default/tree").json()
        saved["name"] = "restored elsewhere"

        with client.websocket_connect("/saklas/v1/sessions/default/stream") as ws:
            response = client.put(
                "/saklas/v1/sessions/default/tree",
                json={"tree": saved},
            )
            assert response.status_code == 200
            while True:
                frame = ws.receive_json()
                if frame.get("type") == "tree_mutated":
                    break

        assert frame["op"] == "restore"
        assert frame["rev"] == session.tree.rev
        assert frame["active_node_id"] == session.tree.active_node_id
        assert session.tree.name == "restored elsewhere"

    def test_matches_to_dict(self, session_and_client: Any):
        session, client = session_and_client
        # Add a user turn so structure has more than just root.
        uid = session.tree.add_user_turn("hello")
        # Tree GET ships ``include_tokens=True`` (webui rehydration
        # requirement); compare against that same shape.
        expected = session.tree.to_dict(include_tokens=True)
        resp = client.get("/saklas/v1/sessions/default/tree")
        assert resp.status_code == 200
        data = resp.json()
        # Same node count, ids and rev as the underlying to_dict
        assert data["rev"] == expected["rev"]
        ids = {n["id"] for n in data["nodes"]}
        assert uid in ids
        assert data["children_of"][session.tree.root_id] == [uid]

    def test_ships_per_token_rows(self, session_and_client: Any):
        """Tree GET serializes ``tokens`` / ``thinking_tokens`` so the
        webui can rehydrate inline highlight tints + token-drilldown
        click targets after a force-refresh.  Regression guard against
        flipping ``include_tokens`` back to ``False`` (which would
        silently break highlight on historical turns)."""
        session, client = session_and_client
        u1 = session.tree.add_user_turn("hi")
        a1 = session.tree.begin_assistant(u1)
        # Two response tokens carrying the same per-token shape the
        # engine's ``_token_tap`` stamps: token_id + text + logprob +
        # the probe / per-layer blobs the highlight tint depends on.
        session.tree.append_token(a1, {
            "token_id": 100,
            "text": "he",
            "logprob": -1.5,
            "probes": {"calm": 0.42},
            "per_layer_scores": {"5": {"calm": 0.38}, "10": {"calm": 0.45}},
            "measurements": {
                "version": 1,
                "scope": "token",
                "provenance": "captured",
                "instruments": {
                    "lens": {
                        "readout": {
                            "layers": [{
                                "layer": 5,
                                "tokens": [{
                                    "token": "hello", "id": 100, "logprob": -0.1,
                                }],
                            }],
                            "aggregate": [],
                        },
                        "binding": {"source": "local:default", "steering": None},
                    },
                },
            },
        })
        session.tree.append_token(a1, {
            "token_id": 101,
            "text": "llo",
            "logprob": -0.9,
            "probes": {"calm": 0.39},
        })
        session.tree.finalize_assistant(
            a1, text="hello", aggregate_readings={"calm": 0.40},
        )

        resp = client.get("/saklas/v1/sessions/default/tree")
        assert resp.status_code == 200
        data = resp.json()
        assistant_node = next(n for n in data["nodes"] if n["id"] == a1)
        assert assistant_node["tokens"] is not None
        assert len(assistant_node["tokens"]) == 2
        first = assistant_node["tokens"][0]
        assert first["text"] == "he"
        assert first["token_id"] == 100
        assert first["logprob"] == -1.5
        assert first["probes"] == {"calm": 0.42}
        assert first["per_layer_scores"] == {
            "5": {"calm": 0.38}, "10": {"calm": 0.45},
        }
        lens_readout = (
            first["measurements"]["instruments"]["lens"]["readout"]
        )
        assert lens_readout["layers"][0]["tokens"][0] == {
            "token": "hello", "id": 100, "logprob": -0.1,
        }

    def test_active_path_shape(self, session_and_client: Any):
        session, client = session_and_client
        u1 = session.tree.add_user_turn("greet")
        a1 = session.tree.begin_assistant(u1)
        session.tree.finalize_assistant(a1, text="hi back")

        resp = client.get("/saklas/v1/sessions/default/tree/active")
        assert resp.status_code == 200
        data = resp.json()
        assert data["active_node_id"] == a1
        assert data["messages"] == [
            {"role": "user", "content": "greet"},
            {"role": "assistant", "content": "hi back"},
        ]
        assert data["node_ids"] == [u1, a1]


# ---------------------------------------------------------------------------
# REST: tree mutations
# ---------------------------------------------------------------------------


class TestTreeNavigate:
    def test_navigate_updates_active(self, session_and_client: Any):
        session, client = session_and_client
        u1 = session.tree.add_user_turn("hi")
        u2 = session.tree.add_user_turn("hi-again", parent_id=session.tree.root_id)
        # u2 should now be active; navigate back to u1
        assert session.tree.active_node_id == u2
        resp = client.post(
            "/saklas/v1/sessions/default/tree/navigate",
            json={"node_id": u1},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["active_node_id"] == u1
        assert session.tree.active_node_id == u1

    def test_navigate_unknown_node_404(self, session_and_client: Any):
        _, client = session_and_client
        resp = client.post(
            "/saklas/v1/sessions/default/tree/navigate",
            json={"node_id": "DOES_NOT_EXIST"},
        )
        assert resp.status_code == 404


class TestTreeEdit:
    def test_edit_in_place(self, session_and_client: Any):
        session, client = session_and_client
        u1 = session.tree.add_user_turn("typo")
        resp = client.post(
            "/saklas/v1/sessions/default/tree/edit",
            json={"node_id": u1, "text": "fixed"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == u1
        assert data["text"] == "fixed"
        assert data["edit_count"] == 1
        # Underlying tree was mutated
        assert session.tree.get(u1).text == "fixed"

    def test_edit_409_during_reservation(self, session_and_client: Any):
        session, client = session_and_client
        u1 = session.tree.add_user_turn("hi")
        # Simulate an in-flight gen reserving u1's subtree.
        session._active_gen_reservation = u1
        resp = client.post(
            "/saklas/v1/sessions/default/tree/edit",
            json={"node_id": u1, "text": "no edit during gen"},
        )
        assert resp.status_code == 409
        # Tree text unchanged
        assert session.tree.get(u1).text == "hi"

    def test_edit_root_400(self, session_and_client: Any):
        session, client = session_and_client
        resp = client.post(
            "/saklas/v1/sessions/default/tree/edit",
            json={"node_id": session.tree.root_id, "text": "nope"},
        )
        assert resp.status_code == 400

    def test_edit_unknown_node_404(self, session_and_client: Any):
        _, client = session_and_client
        resp = client.post(
            "/saklas/v1/sessions/default/tree/edit",
            json={"node_id": "GHOST", "text": "x"},
        )
        assert resp.status_code == 404


class TestTreeBranch:
    def test_branch_succeeds(self, session_and_client: Any):
        session, client = session_and_client
        u1 = session.tree.add_user_turn("hello")
        resp = client.post(
            "/saklas/v1/sessions/default/tree/branch",
            json={"node_id": u1, "text": "hello world"},
        )
        assert resp.status_code == 200
        data = resp.json()
        new_id = data["node_id"]
        assert new_id != u1
        assert data["node"]["text"] == "hello world"
        # Both siblings live under root
        siblings = session.tree.children_of[session.tree.root_id]
        assert u1 in siblings and new_id in siblings

    def test_branch_allowed_during_reservation(self, session_and_client: Any):
        """Branches don't touch the streaming target — must succeed even
        while a gen reservation is held."""
        session, client = session_and_client
        u1 = session.tree.add_user_turn("hi")
        session._active_gen_reservation = u1
        resp = client.post(
            "/saklas/v1/sessions/default/tree/branch",
            json={"node_id": u1, "text": "alternative"},
        )
        assert resp.status_code == 200

    def test_branch_root_400(self, session_and_client: Any):
        session, client = session_and_client
        resp = client.post(
            "/saklas/v1/sessions/default/tree/branch",
            json={"node_id": session.tree.root_id, "text": "x"},
        )
        assert resp.status_code == 400


class TestTreeDelete:
    def test_delete_disjoint_subtree(self, session_and_client: Any):
        session, client = session_and_client
        u1 = session.tree.add_user_turn("a")
        u2 = session.tree.add_user_turn("b", parent_id=session.tree.root_id)
        # Active is u2; deleting u1's subtree (disjoint) is fine.
        resp = client.delete(f"/saklas/v1/sessions/default/tree/{u1}")
        assert resp.status_code == 200
        assert resp.json()["removed"] == 1
        assert not session.tree.has(u1)
        # Active node untouched
        assert session.tree.active_node_id == u2

    def test_delete_containing_active_repoints_active(self, session_and_client: Any):
        # Deleting a subtree that contains the active node used to 400.
        # The engine now repoints the active pointer to the surviving
        # parent (root → fresh start) and the route returns 200.
        session, client = session_and_client
        u1 = session.tree.add_user_turn("a")
        # active is u1 itself — deleting it removes the active node.
        resp = client.delete(f"/saklas/v1/sessions/default/tree/{u1}")
        assert resp.status_code == 200
        assert resp.json()["removed"] == 1
        assert not session.tree.has(u1)
        assert session.tree.active_node_id == session.tree.root_id

    def test_delete_409_during_reservation(self, session_and_client: Any):
        session, client = session_and_client
        u1 = session.tree.add_user_turn("a")
        u2 = session.tree.add_user_turn("b", parent_id=session.tree.root_id)
        session.tree.navigate(u2)  # so deleting u1 is otherwise allowed
        session._active_gen_reservation = u1
        resp = client.delete(f"/saklas/v1/sessions/default/tree/{u1}")
        assert resp.status_code == 409


class TestTreeStarNote:
    def test_star_round_trip(self, session_and_client: Any):
        session, client = session_and_client
        u1 = session.tree.add_user_turn("hi")
        resp = client.post(
            "/saklas/v1/sessions/default/tree/star",
            json={"node_id": u1, "on": True},
        )
        assert resp.status_code == 200
        assert resp.json()["starred"] is True
        # Confirm via the full-tree GET
        tree_resp = client.get("/saklas/v1/sessions/default/tree")
        match = [n for n in tree_resp.json()["nodes"] if n["id"] == u1]
        assert match and match[0]["starred"] is True

    def test_note_round_trip(self, session_and_client: Any):
        session, client = session_and_client
        u1 = session.tree.add_user_turn("hi")
        resp = client.post(
            "/saklas/v1/sessions/default/tree/note",
            json={"node_id": u1, "text": "this is the seed prompt"},
        )
        assert resp.status_code == 200
        assert resp.json()["notes"] == "this is the seed prompt"
        # Confirm via single-node fetch through full tree GET
        tree_resp = client.get("/saklas/v1/sessions/default/tree")
        match = [n for n in tree_resp.json()["nodes"] if n["id"] == u1]
        assert match and match[0]["notes"] == "this is the seed prompt"


class TestTreeReset:
    def test_reset_drops_branches(self, session_and_client: Any):
        session, client = session_and_client
        old_root = session.tree.root_id
        session.tree.add_user_turn("a")
        session.tree.add_user_turn("b", parent_id=old_root)
        resp = client.post("/saklas/v1/sessions/default/tree/reset")
        assert resp.status_code == 204
        # New root, no children
        assert session.tree.root_id != old_root
        assert session.tree.active_node_id == session.tree.root_id
        assert session.tree.children_of[session.tree.root_id] == []

    def test_reset_409_during_reservation(self, session_and_client: Any):
        session, client = session_and_client
        u1 = session.tree.add_user_turn("x")
        session._active_gen_reservation = u1
        resp = client.post("/saklas/v1/sessions/default/tree/reset")
        assert resp.status_code == 409


class TestTranscript:
    def test_transcript_yaml_shape(self, session_and_client: Any):
        session, client = session_and_client
        u1 = session.tree.add_user_turn("hello")
        a1 = session.tree.begin_assistant(u1)
        session.tree.finalize_assistant(a1, text="hi there")

        resp = client.post(
            "/saklas/v1/sessions/default/tree/transcript",
            json={"node_id": a1},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["node_id"] == a1
        text = body["yaml"]
        # to_yaml uses pyyaml's safe_dump; safe-scalar strings like
        # ``test/model`` and ``hello`` come back unquoted (valid YAML).  We
        # check substrings rather than exact quoting form.
        assert "saklas_transcript: 2" in text
        assert "model_id:" in text and "test/model" in text
        # Probes block exists (empty in this stub)
        assert "probes:" in text
        # Two turns: user + assistant
        assert "role: user" in text
        assert "role: assistant" in text
        assert "hello" in text
        assert "hi there" in text
        # Round-trip via the YAML loader to confirm structural shape.
        import yaml
        parsed = yaml.safe_load(text)
        assert parsed["saklas_transcript"] == 2
        assert parsed["model_id"] == "test/model"
        assert len(parsed["turns"]) == 2
        assert parsed["turns"][0]["role"] == "user"
        assert parsed["turns"][1]["role"] == "assistant"

    def test_transcript_unknown_node_404(self, session_and_client: Any):
        _, client = session_and_client
        resp = client.post(
            "/saklas/v1/sessions/default/tree/transcript",
            json={"node_id": "MISSING"},
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# WS: parent_node_id + n-way fan-out
# ---------------------------------------------------------------------------


class TestWebSocketLoom:
    def test_user_stop_is_distinct_from_natural_completion(self, session_and_client: Any):
        """The native stream labels an explicit user cancellation distinctly.

        The engine and stored loom node retain the protocol-compatible
        ``stop`` reason; only the dashboard-facing done frame says
        ``cancelled``.
        """
        session, client = session_and_client
        session._block_until_stop = True

        with client.websocket_connect("/saklas/v1/sessions/default/stream") as ws:
            ws.send_json({"type": "generate", "input": "keep going"})
            # The connection's perpetual reader queues this while the
            # dispatcher starts the worker, matching a fast stop-button click.
            ws.send_json({"type": "stop"})
            while True:
                msg = ws.receive_json()
                if msg["type"] == "error":
                    pytest.fail(msg["message"])
                if msg["type"] == "done":
                    done = msg
                    break

        assert done["result"]["finish_reason"] == "cancelled"
        assert session.tree.get(done["node_id"]).finish_reason == "stop"

    def test_generate_with_parent_node_id(self, session_and_client: Any):
        """Result attaches under the supplied parent_node_id."""
        session, client = session_and_client
        # Build a tree with two user-turn options.
        u1 = session.tree.add_user_turn("path A")
        u2 = session.tree.add_user_turn("path B", parent_id=session.tree.root_id)
        # Navigate to u1 so default-parent would land under u1.
        session.tree.navigate(u1)
        rev_before = session.tree.rev

        with client.websocket_connect("/saklas/v1/sessions/default/stream") as ws:
            ws.send_json({
                "type": "generate",
                "input": "explore",
                "parent_node_id": u2,
            })
            # Collect events until done.
            done = None
            seen_tree_mut = False
            seen_finalize = False
            seen_node_id_on_token = None
            while True:
                msg = ws.receive_json()
                t = msg["type"]
                if t == "started":
                    assert msg["sibling_count"] == 1
                elif t == "tree_mutated":
                    seen_tree_mut = True
                    if msg.get("op") == "finalize_assistant":
                        seen_finalize = True
                        assert any(
                            node.get("text") == "tok0"
                            and node.get("finish_reason") == "stop"
                            for node in msg["updated"]
                        )
                    assert msg["rev"] >= rev_before
                elif t == "token":
                    seen_node_id_on_token = msg.get("node_id")
                elif t == "done":
                    done = msg
                    break

        assert done is not None
        assert seen_tree_mut, "tree_mutated event should fire when tree mutates"
        assert seen_finalize, "finalized node must arrive before the done frame"
        assert seen_node_id_on_token is not None, "token frames should be tagged with node_id"
        # New user turn attached under u2 (not u1, which would have been
        # the active node before the request).  The assistant attaches
        # under that new user turn.
        u2_children = session.tree.children_of[u2]
        assert len(u2_children) == 1, "exactly one user-turn child of the parent"
        new_user_id = u2_children[0]
        assert session.tree.get(new_user_id).role == "user"
        assistant_children = session.tree.children_of[new_user_id]
        assert len(assistant_children) == 1
        assistant_id = assistant_children[0]
        assert session.tree.get(assistant_id).role == "assistant"
        # The token-tag matches the assistant node id.
        assert seen_node_id_on_token == assistant_id
        # No assistant landed under u1 (the request bypassed the active node).
        assert session.tree.children_of[u1] == []

    def test_generate_n2_creates_two_siblings(self, session_and_client: Any):
        """n=2 produces two assistant siblings under the same user-parent.

        Pinned to ``parent_node_id=root`` so the engine's add_user_turn
        dedup matches on iter 1 (same parent + same text); without an
        explicit parent the active node walks to the assistant after
        iter 0 and the second iteration would attach under it instead.
        """
        session, client = session_and_client
        root_id = session.tree.root_id
        with client.websocket_connect("/saklas/v1/sessions/default/stream") as ws:
            ws.send_json({
                "type": "generate",
                "input": "twice",
                "n": 2,
                "parent_node_id": root_id,
            })
            done_events = []
            seen_sibling_indices = set()
            while True:
                msg = ws.receive_json()
                t = msg["type"]
                if t == "started":
                    seen_sibling_indices.add(msg["sibling_index"])
                    assert msg["sibling_count"] == 2
                elif t == "done":
                    done_events.append(msg)
                    if len(done_events) == 2:
                        break

        assert len(done_events) == 2
        assert seen_sibling_indices == {0, 1}
        # Two assistant siblings under the same user node.
        # The user node was created via add_user_turn (dedup keeps a single one).
        user_children = session.tree.children_of[session.tree.root_id]
        assert len(user_children) == 1, "siblings should share one user parent"
        user_id = user_children[0]
        assistant_siblings = session.tree.children_of[user_id]
        assistant_ids = [
            nid for nid in assistant_siblings
            if session.tree.get(nid).role == "assistant"
        ]
        assert len(assistant_ids) == 2
        # done frames carry distinct node_ids matching the two assistants.
        done_node_ids = {ev["node_id"] for ev in done_events}
        assert done_node_ids == set(assistant_ids)


# ---------------------------------------------------------------------------
# WS: answer-prefill
# ---------------------------------------------------------------------------


class TestPrefill:
    def test_missing_text_400(self, session_and_client: Any):
        """prefill_node_id without prefill_text is rejected before dispatch."""
        session, client = session_and_client
        uid = session.tree.add_user_turn("seed me")
        with client.websocket_connect("/saklas/v1/sessions/default/stream") as ws:
            ws.send_json({"type": "generate", "prefill_node_id": uid})
            msg = ws.receive_json()
        assert msg["type"] == "error"
        assert msg["status"] == 400
        assert "prefill" in msg["message"].lower()

    def test_empty_text_400(self, session_and_client: Any):
        """An empty prefill_text is treated as 'no text' — same 400."""
        session, client = session_and_client
        uid = session.tree.add_user_turn("seed me")
        with client.websocket_connect("/saklas/v1/sessions/default/stream") as ws:
            ws.send_json({
                "type": "generate",
                "prefill_node_id": uid,
                "prefill_text": "",
            })
            msg = ws.receive_json()
        assert msg["type"] == "error"
        assert msg["status"] == 400

    def test_conflicts_with_fork_400(self, session_and_client: Any):
        """A message can't be both a fork and a prefill."""
        session, client = session_and_client
        uid = session.tree.add_user_turn("seed me")
        with client.websocket_connect("/saklas/v1/sessions/default/stream") as ws:
            ws.send_json({
                "type": "generate",
                "prefill_node_id": uid,
                "prefill_text": "Sure",
                "fork_node_id": "whatever",
                "fork_raw_index": 0,
                "fork_alt_token_id": 1,
            })
            msg = ws.receive_json()
        assert msg["type"] == "error"
        assert msg["status"] == 400

    def test_creates_seeded_sibling(self, session_and_client: Any):
        """A valid prefill lands an assistant child opening with the text."""
        session, client = session_and_client
        uid = session.tree.add_user_turn("what's the weather?")
        rev_before = session.tree.rev

        with client.websocket_connect("/saklas/v1/sessions/default/stream") as ws:
            ws.send_json({
                "type": "generate",
                "prefill_node_id": uid,
                "prefill_text": "It is sunny",
            })
            done = None
            seen_tree_mut = False
            while True:
                msg = ws.receive_json()
                t = msg["type"]
                if t == "tree_mutated":
                    seen_tree_mut = True
                    assert msg["rev"] >= rev_before
                elif t == "done":
                    done = msg
                    break

        assert done is not None
        assert seen_tree_mut
        # The assistant landed under the prefill target user node.
        assistant_children = [
            nid for nid in session.tree.children_of[uid]
            if session.tree.get(nid).role == "assistant"
        ]
        assert len(assistant_children) == 1
        assistant = session.tree.get(assistant_children[0])
        assert assistant.text.startswith("It is sunny")


# ---------------------------------------------------------------------------
# WS: commit (Ctrl+Enter — no-generation send)
# ---------------------------------------------------------------------------


class TestCommit:
    def test_missing_text_400(self, session_and_client: Any):
        """commit_role without commit_text is rejected before dispatch."""
        _session, client = session_and_client
        with client.websocket_connect("/saklas/v1/sessions/default/stream") as ws:
            ws.send_json({"type": "generate", "commit_role": "user"})
            msg = ws.receive_json()
        assert msg["type"] == "error"
        assert msg["status"] == 400
        assert "commit" in msg["message"].lower()

    def test_invalid_role_400(self, session_and_client: Any):
        _session, client = session_and_client
        with client.websocket_connect("/saklas/v1/sessions/default/stream") as ws:
            ws.send_json({
                "type": "generate",
                "commit_role": "system",
                "commit_text": "anything",
            })
            msg = ws.receive_json()
        # Pydantic rejects the Literal-mismatch at parse time before the
        # handler runs, so the error frame source can be either layer —
        # both surface 400.
        assert msg["type"] == "error"

    def test_assistant_commit_requires_parent_400(self, session_and_client: Any):
        """``commit_role='assistant'`` without parent_node_id is rejected —
        the authored turn has to hang off a known user node."""
        _session, client = session_and_client
        with client.websocket_connect("/saklas/v1/sessions/default/stream") as ws:
            ws.send_json({
                "type": "generate",
                "commit_role": "assistant",
                "commit_text": "the answer",
            })
            msg = ws.receive_json()
        assert msg["type"] == "error"
        assert msg["status"] == 400
        assert "parent_node_id" in msg["message"]

    def test_conflicts_with_prefill_400(self, session_and_client: Any):
        """A message can't be both a commit and a prefill."""
        session, client = session_and_client
        uid = session.tree.add_user_turn("seed me")
        with client.websocket_connect("/saklas/v1/sessions/default/stream") as ws:
            ws.send_json({
                "type": "generate",
                "commit_role": "user",
                "commit_text": "manual",
                "prefill_node_id": uid,
                "prefill_text": "It is",
            })
            msg = ws.receive_json()
        assert msg["type"] == "error"
        assert msg["status"] == 400

    def test_commit_user_lands_user_child(self, session_and_client: Any):
        """Ctrl+Enter on a non-user active node — server creates a user
        child under the active node and acks with the new node id."""
        session, client = session_and_client
        root = session.tree.root_id
        rev_before = session.tree.rev

        started = None
        done = None
        with client.websocket_connect("/saklas/v1/sessions/default/stream") as ws:
            ws.send_json({
                "type": "generate",
                "commit_role": "user",
                "commit_text": "manual user input",
            })
            while True:
                msg = ws.receive_json()
                t = msg["type"]
                if t == "started":
                    started = msg
                elif t == "done":
                    done = msg
                    break

        assert started is not None
        assert started["node_id"] is None
        assert done is not None
        assert done["result"]["kind"] == "commit"
        assert done["result"]["role"] == "user"
        assert done["result"]["text"] == "manual user input"
        assert done["node_id"] == done["result"]["node_id"]

        # The new user node sits under the root (active node at request
        # time) and the tree advanced its rev.
        new_id = done["node_id"]
        assert session.tree.get(new_id).role == "user"
        assert session.tree.get(new_id).text == "manual user input"
        assert session.tree.nodes[new_id].parent_id == root
        assert session.tree.rev > rev_before


class TestSubmit:
    def test_text_requires_authored_role(self, session_and_client: Any):
        _session, client = session_and_client
        with client.websocket_connect("/saklas/v1/sessions/default/stream") as ws:
            ws.send_json({
                "type": "submit",
                "text": "hello",
                "generated_role": "assistant",
            })
            msg = ws.receive_json()
        assert msg["type"] == "error"
        assert msg["status"] == 400

    def test_swapped_commit_only_can_start_with_assistant(
        self, session_and_client: Any,
    ):
        session, client = session_and_client
        # Role swapping is exposed only for scene-capable renderers, where an
        # assistant-authored first turn is a legal structural shape.
        session.scene_grammar = MagicMock()
        with client.websocket_connect("/saklas/v1/sessions/default/stream") as ws:
            ws.send_json({
                "type": "submit",
                "text": "opening as the assistant",
                "authored_role": "assistant",
            })
            while True:
                msg = ws.receive_json()
                if msg["type"] == "done":
                    break
                assert msg["type"] != "error", msg
        assert msg["result"]["role"] == "assistant"
        assert msg["result"]["kind"] == "append"
        path = session.tree.active_path()[1:]
        assert [node.role for node in path] == ["assistant"]
        assert path[0].recipe is None

    def test_unswapped_commits_user_then_generates_assistant(
        self, session_and_client: Any,
    ):
        session, client = session_and_client
        with client.websocket_connect("/saklas/v1/sessions/default/stream") as ws:
            ws.send_json({
                "type": "submit",
                "text": "hello",
                "authored_role": "user",
                "generated_role": "assistant",
            })
            while True:
                msg = ws.receive_json()
                if msg["type"] == "done":
                    break
                assert msg["type"] != "error", msg
        path = session.tree.active_path()[1:]
        assert [node.role for node in path] == ["user", "assistant"]
        assert path[0].text == "hello"

    def test_swapped_commits_assistant_then_generates_user(
        self, session_and_client: Any,
    ):
        session, client = session_and_client
        user_id = session.tree.add_user_turn("question")
        with client.websocket_connect("/saklas/v1/sessions/default/stream") as ws:
            ws.send_json({
                "type": "submit",
                "text": "authored answer",
                "authored_role": "assistant",
                "generated_role": "user",
                "parent_node_id": user_id,
            })
            while True:
                msg = ws.receive_json()
                if msg["type"] == "done":
                    break
                assert msg["type"] != "error", msg
        path = session.tree.active_path()[1:]
        assert [node.role for node in path] == ["user", "assistant", "user"]
        assert path[-2].text == "authored answer"

    def test_empty_with_no_modifier_semantics_continues_generated_role(
        self, session_and_client: Any,
    ):
        session, client = session_and_client
        with client.websocket_connect("/saklas/v1/sessions/default/stream") as ws:
            ws.send_json({"type": "submit", "generated_role": "assistant"})
            while True:
                msg = ws.receive_json()
                if msg["type"] == "done":
                    break
                assert msg["type"] != "error", msg
        path = session.tree.active_path()[1:]
        assert [node.role for node in path] == ["assistant"]

    def test_append_reuses_matching_authored_message(
        self, session_and_client: Any,
    ):
        session, client = session_and_client
        with client.websocket_connect("/saklas/v1/sessions/default/stream") as ws:
            ws.send_json({
                "type": "submit", "text": "one",
                "authored_role": "user",
            })
            while (first := ws.receive_json())["type"] != "done":
                assert first["type"] != "error", first
            ws.send_json({
                "type": "submit", "text": " two",
                "authored_role": "user",
            })
            while (second := ws.receive_json())["type"] != "done":
                assert second["type"] != "error", second
        assert second["node_id"] == first["node_id"]
        path = session.tree.active_path()[1:]
        assert len(path) == 1
        assert path[0].text == "one two"

    def test_generate_reuses_matching_role_as_prefill(
        self, session_and_client: Any,
    ):
        session, client = session_and_client
        session.scene_grammar = MagicMock()
        with client.websocket_connect("/saklas/v1/sessions/default/stream") as ws:
            ws.send_json({
                "type": "submit", "text": "Once",
                "authored_role": "assistant",
            })
            while (appended := ws.receive_json())["type"] != "done":
                assert appended["type"] != "error", appended
            ws.send_json({"type": "submit", "generated_role": "assistant"})
            while (generated := ws.receive_json())["type"] != "done":
                assert generated["type"] != "error", generated
        assert generated["node_id"] == appended["node_id"]
        path = session.tree.active_path()[1:]
        assert len(path) == 1
        assert path[0].role == "assistant"
        assert path[0].text == "Oncetok0"


class TestCommitContinued:
    def test_commit_user_appends_under_user_node(self, session_and_client: Any):
        """The compatibility commit adapter shares same-role append semantics."""
        session, client = session_and_client
        uid = session.tree.add_user_turn("first")  # active = uid
        with client.websocket_connect("/saklas/v1/sessions/default/stream") as ws:
            ws.send_json({
                "type": "generate",
                "commit_role": "user",
                "commit_text": "second",
                "parent_node_id": uid,
            })
            while True:
                msg = ws.receive_json()
                if msg["type"] in ("error", "done"):
                    break
        assert msg["type"] == "done"
        assert msg["node_id"] == uid
        assert session.tree.nodes[uid].text == "firstsecond"

    def test_commit_user_raw_allows_user_parent(self, session_and_client: Any):
        """A flat (base-model) commit carries ``raw: true`` — the
        user-under-user guard is lifted, so an authored span lands
        under a node of any role."""
        session, client = session_and_client
        uid = session.tree.add_user_turn("first")  # active = uid, role user
        done = None
        with client.websocket_connect("/saklas/v1/sessions/default/stream") as ws:
            ws.send_json({
                "type": "generate",
                "commit_role": "user",
                "commit_text": "second",
                "parent_node_id": uid,
                "raw": True,
            })
            while True:
                msg = ws.receive_json()
                if msg["type"] == "done":
                    done = msg
                    break
                assert msg["type"] != "error", msg
        assert done is not None
        new_id = done["result"]["node_id"]
        assert new_id == uid
        assert session.tree.nodes[new_id].text == "firstsecond"

    def test_commit_assistant_lands_authored_sibling(self, session_and_client: Any):
        """Ctrl+Enter on a user active node lands a sibling assistant
        whose text *is* the typed text — no decode, no continuation."""
        session, client = session_and_client
        uid = session.tree.add_user_turn("the question?")
        rev_before = session.tree.rev

        done = None
        with client.websocket_connect("/saklas/v1/sessions/default/stream") as ws:
            ws.send_json({
                "type": "generate",
                "commit_role": "assistant",
                "commit_text": "the canned answer",
                "parent_node_id": uid,
            })
            while True:
                msg = ws.receive_json()
                if msg["type"] == "done":
                    done = msg
                    break

        assert done is not None
        assert done["result"]["role"] == "assistant"
        new_id = done["node_id"]
        assistant = session.tree.get(new_id)
        assert assistant.role == "assistant"
        assert assistant.text == "the canned answer"
        assert assistant.recipe is None  # implicit authored marker
        assert assistant.raw_token_ids  # tokenized by the stub
        assert session.tree.nodes[new_id].parent_id == uid
        assert session.tree.rev > rev_before


class TestCast:
    def test_cast_crud_round_trip(self, session_and_client: Any) -> None:
        session, client = session_and_client
        # PUT creates.
        resp = client.put(
            "/saklas/v1/sessions/default/tree/cast/deer",
            json={"steering": "0.5 formal.casual", "notes": "skittish"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["label"] == "deer"
        assert body["member"]["recipe"]["steering"] == "0.5 formal.casual"
        # GET reads the roster.
        resp = client.get("/saklas/v1/sessions/default/tree/cast")
        assert resp.status_code == 200
        assert "deer" in resp.json()["cast"]
        # The full-tree GET carries the roster too.
        resp = client.get("/saklas/v1/sessions/default/tree")
        assert resp.json()["cast"]["deer"]["notes"] == "skittish"
        # DELETE removes; absent delete is still 204.
        assert client.delete(
            "/saklas/v1/sessions/default/tree/cast/deer"
        ).status_code == 204
        assert "deer" not in session.tree.cast
        assert client.delete(
            "/saklas/v1/sessions/default/tree/cast/deer"
        ).status_code == 204

    def test_cast_put_validates(self, session_and_client: Any) -> None:
        _session, client = session_and_client
        # Bad label (uppercase/space) -> 400 via SaklasError mapping.
        resp = client.put(
            "/saklas/v1/sessions/default/tree/cast/Not%20A%20Slug",
            json={},
        )
        assert resp.status_code == 400
        # Bad steering expression -> 400 at authoring time.
        resp = client.put(
            "/saklas/v1/sessions/default/tree/cast/deer",
            json={"steering": "0.5 !!nope!!"},
        )
        assert resp.status_code == 400

    def test_cast_mutation_emits_ws_frame_with_roster(
        self, session_and_client: Any,
    ) -> None:
        session, client = session_and_client
        u1 = session.tree.add_user_turn("hello")
        del u1
        with client.websocket_connect("/saklas/v1/sessions/default/stream") as ws:
            resp = client.put(
                "/saklas/v1/sessions/default/tree/cast/deer",
                json={"steering": "0.5 formal.casual"},
            )
            assert resp.status_code == 200
            frame = None
            for _ in range(10):
                msg = ws.receive_json()
                if msg.get("type") == "tree_mutated" and msg.get("op") == "cast":
                    frame = msg
                    break
            assert frame is not None, "op=cast tree_mutated frame should arrive"
            assert frame["cast"]["deer"]["recipe"]["steering"] == "0.5 formal.casual"
            assert frame["added"] == [] and frame["updated"] == []
