"""Tests for the native /saklas/v1/* API (no GPU)."""

import asyncio
import json
import threading
import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from saklas.core.results import GenerationResult


def _mock_session():
    session = MagicMock()
    session.model_id = "test/model"
    session.model_info = {
        "model_type": "gemma2",
        "num_layers": 26,
        "hidden_dim": 2304,
        "device": "cpu",
        "dtype": "torch.bfloat16",
    }
    session._device = "cpu"
    session._dtype = "torch.bfloat16"
    session._created_ts = 1_700_000_000

    session.config = MagicMock()
    session.config.temperature = 1.0
    session.config.top_p = 0.9
    session.config.top_k = None
    session.config.max_new_tokens = 1024
    session.config.system_prompt = None

    session.vectors = {}
    session.probes = {}
    session.history = []

    monitor = MagicMock()
    monitor.probe_names = []
    monitor.profiles = {}
    session._monitor = monitor
    session._tokenizer = MagicMock()
    session._layers = []
    session._last_per_token_scores = None
    session._last_result = None
    session.last_per_token_scores = None
    session.last_result = None

    gen_state = MagicMock()
    gen_state.finish_reason = "stop"
    session._gen_state = gen_state

    session.build_readings.return_value = {}
    session.lock = asyncio.Lock()

    # Trait queue infrastructure (used by SSE traits/stream endpoint).
    session._trait_queues = []
    session._trait_lock = threading.Lock()
    session._trait_subscribers = property(lambda self: len(self._trait_queues))

    def _register_trait_queue(loop, q):
        with session._trait_lock:
            session._trait_queues.append((loop, q))
    session.register_trait_queue = _register_trait_queue

    def _unregister_trait_queue(loop, q):
        with session._trait_lock:
            try:
                session._trait_queues.remove((loop, q))
            except ValueError:
                pass
    session.unregister_trait_queue = _unregister_trait_queue

    # EventBus mock with subscribe/unsubscribe support.
    _event_subscribers = []

    def _subscribe(cb):
        _event_subscribers.append(cb)
        def _unsub():
            try:
                _event_subscribers.remove(cb)
            except ValueError:
                pass
        return _unsub

    def _emit(event):
        for cb in list(_event_subscribers):
            try:
                cb(event)
            except Exception:
                pass

    events = MagicMock()
    events.subscribe = _subscribe
    events.emit = _emit
    session.events = events
    session._event_subscribers = _event_subscribers

    return session


@pytest.fixture
def session_and_client():
    from saklas.server import create_app
    session = _mock_session()
    app = create_app(session, default_alphas={})
    return session, TestClient(app)


# ---- sessions collection -------------------------------------------------


class TestSessions:
    def test_list(self, session_and_client):
        _, client = session_and_client
        with patch("saklas.server.saklas_api.supports_thinking", return_value=False):
            resp = client.get("/saklas/v1/sessions")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["sessions"]) == 1
        s = data["sessions"][0]
        assert s["id"] == "default"
        assert s["model_id"] == "test/model"
        assert "config" in s
        assert s["config"]["temperature"] == 1.0

    def test_create_idempotent(self, session_and_client):
        _, client = session_and_client
        with patch("saklas.server.saklas_api.supports_thinking", return_value=False):
            resp = client.post("/saklas/v1/sessions", json={})
        assert resp.status_code == 200
        assert resp.json()["id"] == "default"

    def test_create_model_mismatch_logs_warning(self, session_and_client, caplog):
        _, client = session_and_client
        with patch("saklas.server.saklas_api.supports_thinking", return_value=False):
            resp = client.post("/saklas/v1/sessions", json={"model": "other/model"})
        assert resp.status_code == 200
        assert resp.json()["model_id"] == "test/model"

    def test_get_by_default(self, session_and_client):
        _, client = session_and_client
        with patch("saklas.server.saklas_api.supports_thinking", return_value=False):
            resp = client.get("/saklas/v1/sessions/default")
        assert resp.status_code == 200

    def test_get_not_found(self, session_and_client):
        _, client = session_and_client
        resp = client.get("/saklas/v1/sessions/other")
        assert resp.status_code == 404

    def test_delete_is_noop(self, session_and_client):
        _, client = session_and_client
        resp = client.delete("/saklas/v1/sessions/default")
        assert resp.status_code == 204

    def test_patch_updates_config(self, session_and_client):
        session, client = session_and_client
        with patch("saklas.server.saklas_api.supports_thinking", return_value=False):
            resp = client.patch(
                "/saklas/v1/sessions/default",
                json={"temperature": 0.3, "system_prompt": "Be brief."},
            )
        assert resp.status_code == 200
        assert session.config.temperature == 0.3
        assert session.config.system_prompt == "Be brief."

    def test_clear(self, session_and_client):
        session, client = session_and_client
        resp = client.post("/saklas/v1/sessions/default/clear")
        assert resp.status_code == 204
        session.clear_history.assert_called_once()

    def test_rewind_empty(self, session_and_client):
        session, client = session_and_client
        session.history = []
        resp = client.post("/saklas/v1/sessions/default/rewind")
        assert resp.status_code == 400


# ---- vectors -------------------------------------------------------------


class TestVectors:
    def test_list_empty(self, session_and_client):
        _, client = session_and_client
        resp = client.get("/saklas/v1/sessions/default/vectors")
        assert resp.status_code == 200
        assert resp.json()["vectors"] == []

    def test_get_not_found(self, session_and_client):
        _, client = session_and_client
        resp = client.get("/saklas/v1/sessions/default/vectors/missing")
        assert resp.status_code == 404

    def test_delete_not_found(self, session_and_client):
        _, client = session_and_client
        resp = client.delete("/saklas/v1/sessions/default/vectors/missing")
        assert resp.status_code == 404


# ---- probes --------------------------------------------------------------


class TestProbes:
    def test_list_empty(self, session_and_client):
        _, client = session_and_client
        resp = client.get("/saklas/v1/sessions/default/probes")
        assert resp.status_code == 200
        assert resp.json()["probes"] == []

    def test_defaults(self, session_and_client):
        _, client = session_and_client
        with patch(
            "saklas.server.saklas_api.load_defaults",
            return_value={"emotion": ["happiness"]},
        ):
            resp = client.get("/saklas/v1/sessions/default/probes/defaults")
        assert resp.status_code == 200
        assert "emotion" in resp.json()["defaults"]

    def test_activate(self, session_and_client):
        session, client = session_and_client
        resp = client.post("/saklas/v1/sessions/default/probes/happy")
        assert resp.status_code == 204
        session.probe.assert_called_once_with("happy")

    def test_deactivate_not_found(self, session_and_client):
        _, client = session_and_client
        resp = client.delete("/saklas/v1/sessions/default/probes/missing")
        assert resp.status_code == 404

    def test_score_probe_oneshot(self, session_and_client):
        session, client = session_and_client
        session._monitor.probe_names = ["happy"]
        session._monitor.measure.return_value = {"happy": 0.42}
        resp = client.post(
            "/saklas/v1/sessions/default/probe",
            json={"text": "hello world"},
        )
        assert resp.status_code == 200
        assert resp.json()["readings"]["happy"] == pytest.approx(0.42)


# ---- extract -------------------------------------------------------------


class TestExtract:
    def test_extract_json(self, session_and_client):
        import torch
        session, client = session_and_client
        profile = {0: torch.zeros(4), 1: torch.ones(4)}
        session.extract.return_value = ("angry.calm", profile)
        resp = client.post(
            "/saklas/v1/sessions/default/extract",
            json={"name": "angry.calm", "source": "angry", "register": False},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["canonical"] == "angry.calm"
        assert data["profile"]["layers"] == [0, 1]


# ---- WebSocket token+probe co-stream ------------------------------------


class TestWebSocket:
    def _attach_generate(self, session, tokens):
        """Install a fake ``session.generate`` that drives ``on_token``."""
        def _gen(input, *, steering=None, sampling=None, stateless=False,
                 raw=False, thinking=None, on_token=None):
            for i, tok in enumerate(tokens):
                on_token(tok, False, 1000 + i, None, None)
                time.sleep(0.001)
            result = GenerationResult(
                text="".join(tokens), tokens=list(range(1000, 1000 + len(tokens))),
                token_count=len(tokens), tok_per_sec=50.0, elapsed=0.05,
                finish_reason="stop",
            )
            session._last_result = result
            session.last_result = result
            per_token = {
                "happy": [0.1 * (i + 1) for i in range(len(tokens))],
            }
            session._last_per_token_scores = per_token
            session.last_per_token_scores = per_token
            return result

        session.generate.side_effect = _gen

    def test_generate_happy_path(self, session_and_client):
        session, client = session_and_client
        self._attach_generate(session, ["Hello", " ", "world"])

        with client.websocket_connect("/saklas/v1/sessions/default/stream") as ws:
            ws.send_json({"type": "generate", "input": "hi"})
            msg = ws.receive_json()
            assert msg["type"] == "started"
            tokens = []
            while True:
                msg = ws.receive_json()
                if msg["type"] == "token":
                    tokens.append(msg["text"])
                elif msg["type"] == "done":
                    done = msg
                    break
            assert tokens == ["Hello", " ", "world"]
            assert done["result"]["finish_reason"] == "stop"
            ptp = done["result"]["per_token_probes"]
            assert len(ptp) == 3
            assert ptp[0]["probes"]["happy"] == pytest.approx(0.1)

    def test_unknown_message_type(self, session_and_client):
        _, client = session_and_client
        with client.websocket_connect("/saklas/v1/sessions/default/stream") as ws:
            ws.send_json({"type": "frobnicate"})
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "unknown message type" in msg["message"]

    def test_session_mismatch_closes(self, session_and_client):
        _, client = session_and_client
        with pytest.raises(Exception):
            with client.websocket_connect("/saklas/v1/sessions/other/stream") as ws:
                ws.receive_json()

    def test_ws_requires_bearer_when_api_key_set(self):
        from saklas.server import create_app
        session = _mock_session()
        app = create_app(session, default_alphas={}, api_key="s3cret")
        client = TestClient(app)
        # No Authorization header -> close(1008) before accept.
        with pytest.raises(Exception):
            with client.websocket_connect("/saklas/v1/sessions/default/stream") as ws:
                ws.receive_json()
        # Wrong token -> same.
        with pytest.raises(Exception):
            with client.websocket_connect(
                "/saklas/v1/sessions/default/stream",
                headers={"Authorization": "Bearer wrong"},
            ) as ws:
                ws.receive_json()
        # Correct token -> handshake succeeds.
        with client.websocket_connect(
            "/saklas/v1/sessions/default/stream",
            headers={"Authorization": "Bearer s3cret"},
        ) as ws:
            ws.send_json({"type": "frobnicate"})
            msg = ws.receive_json()
            assert msg["type"] == "error"


# ---- Live traits SSE stream -----------------------------------------------


class TestTraitsStream:
    def test_session_not_found_404(self, session_and_client):
        _, client = session_and_client
        resp = client.get("/saklas/v1/sessions/nonexistent/traits/stream")
        assert resp.status_code == 404

    def test_auth_required(self):
        """With api_key set, the SSE endpoint requires Bearer auth."""
        from saklas.server import create_app
        session = _mock_session()
        app = create_app(session, default_alphas={}, api_key="s3cret")
        client = TestClient(app)
        resp = client.get("/saklas/v1/sessions/default/traits/stream")
        assert resp.status_code == 401

    def test_register_unregister_trait_queue(self, session_and_client):
        """Trait queue registration/unregistration works correctly."""
        session, _ = session_and_client
        loop = asyncio.new_event_loop()
        q = asyncio.Queue()
        assert len(session._trait_queues) == 0
        session.register_trait_queue(loop, q)
        assert len(session._trait_queues) == 1
        session.unregister_trait_queue(loop, q)
        assert len(session._trait_queues) == 0
        # Double unregister is a no-op.
        session.unregister_trait_queue(loop, q)
        assert len(session._trait_queues) == 0
        loop.close()

    def test_trait_queue_receives_events_via_loop(self):
        """Events pushed via loop.call_soon_threadsafe arrive on the queue."""
        loop = asyncio.new_event_loop()
        q = asyncio.Queue()

        async def _run():
            loop.call_soon_threadsafe(
                q.put_nowait,
                ("token", 0, "Hi", False, {"happy": 0.5}),
            )
            item = await asyncio.wait_for(q.get(), timeout=1.0)
            assert item[0] == "token"
            assert item[1] == 0
            assert item[2] == "Hi"
            assert item[4]["happy"] == 0.5

        loop.run_until_complete(_run())
        loop.close()


# ---- score_single_token (monitor) ----------------------------------------


class TestScoreSingleToken:
    def test_returns_scores_without_accumulation(self):
        import torch
        from saklas.core.monitor import TraitMonitor

        dim = 16
        probe_vec = torch.randn(dim)
        profiles = {"test_probe": {0: probe_vec}}
        means = {0: torch.zeros(dim)}
        monitor = TraitMonitor(profiles, means)

        hidden = {0: torch.randn(dim)}
        scores = monitor.score_single_token(hidden)

        assert "test_probe" in scores
        assert isinstance(scores["test_probe"], float)
        # History should NOT have been updated.
        assert len(monitor.history["test_probe"]) == 0
        assert monitor._stats["test_probe"]["count"] == 0

    def test_consistent_with_measure_from_hidden(self):
        import torch
        from saklas.core.monitor import TraitMonitor

        dim = 16
        probe_vec = torch.randn(dim)
        profiles = {"p1": {0: probe_vec, 1: torch.randn(dim)}}
        means = {0: torch.zeros(dim), 1: torch.zeros(dim)}
        monitor = TraitMonitor(profiles, means)

        hidden = {0: torch.randn(dim), 1: torch.randn(dim)}
        single = monitor.score_single_token(hidden)
        no_acc = monitor.measure_from_hidden(hidden, accumulate=False)

        assert single["p1"] == pytest.approx(no_acc["p1"])
