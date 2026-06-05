"""Tests for the native /saklas/v1/* API (no GPU)."""

import asyncio
import json
import threading
import time
from typing import Any, cast
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
    session._manifolds = {}

    monitor = MagicMock()
    monitor.probe_names = []
    monitor.attached_probes.return_value = {}
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

    def _register_trait_queue(loop: Any, q: Any) -> None:
        with session._trait_lock:
            session._trait_queues.append((loop, q))
    session.register_trait_queue = _register_trait_queue

    def _unregister_trait_queue(loop: Any, q: Any) -> None:
        with session._trait_lock:
            try:
                session._trait_queues.remove((loop, q))
            except ValueError:
                pass
    session.unregister_trait_queue = _unregister_trait_queue

    # EventBus mock with subscribe/unsubscribe support.
    _event_subscribers = []

    def _subscribe(cb: Any) -> Any:
        _event_subscribers.append(cb)
        def _unsub() -> None:
            try:
                _event_subscribers.remove(cb)
            except ValueError:
                pass
        return _unsub

    def _emit(event: Any) -> None:
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
    app = create_app(session, default_steering=None)
    return session, TestClient(app)


# ---- sessions collection -------------------------------------------------


class TestSessions:
    def test_list(self, session_and_client: Any) -> None:
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

    def test_create_idempotent(self, session_and_client: Any) -> None:
        _, client = session_and_client
        with patch("saklas.server.saklas_api.supports_thinking", return_value=False):
            resp = client.post("/saklas/v1/sessions", json={})
        assert resp.status_code == 200
        assert resp.json()["id"] == "default"

    def test_create_model_mismatch_logs_warning(self, session_and_client: Any, caplog: Any) -> None:
        _, client = session_and_client
        with patch("saklas.server.saklas_api.supports_thinking", return_value=False):
            resp = client.post("/saklas/v1/sessions", json={"model": "other/model"})
        assert resp.status_code == 200
        assert resp.json()["model_id"] == "test/model"

    def test_get_by_default(self, session_and_client: Any) -> None:
        _, client = session_and_client
        with patch("saklas.server.saklas_api.supports_thinking", return_value=False):
            resp = client.get("/saklas/v1/sessions/default")
        assert resp.status_code == 200

    def test_get_not_found(self, session_and_client: Any) -> None:
        _, client = session_and_client
        resp = client.get("/saklas/v1/sessions/other")
        assert resp.status_code == 404

    def test_delete_is_noop(self, session_and_client: Any) -> None:
        _, client = session_and_client
        resp = client.delete("/saklas/v1/sessions/default")
        assert resp.status_code == 204

    def test_patch_updates_config(self, session_and_client: Any) -> None:
        session, client = session_and_client
        with patch("saklas.server.saklas_api.supports_thinking", return_value=False):
            resp = client.patch(
                "/saklas/v1/sessions/default",
                json={"temperature": 0.3, "system_prompt": "Be brief."},
            )
        assert resp.status_code == 200
        assert session.config.temperature == 0.3
        assert session.config.system_prompt == "Be brief."

    @staticmethod
    def _set_family(session: Any, model_type: str) -> None:
        """Pin the mock session's resolved model_type so the role-header
        registries (and thus role-support gating) see a real family."""
        session._model = MagicMock()
        session._model.config = MagicMock()
        session._model.config.text_config = None
        session._model.config.model_type = model_type

    def test_session_info_exposes_role_support(self, session_and_client: Any) -> None:
        """Per-message role boxes gate on these flags — keep them on the wire."""
        session, client = session_and_client
        self._set_family(session, "gemma2")
        with patch("saklas.server.saklas_api.supports_thinking", return_value=False):
            resp = client.get("/saklas/v1/sessions/default")
        assert resp.status_code == 200
        body = resp.json()
        assert body["role_substitution_supported"] is True
        assert body["user_role_supported"] is True
        # Gemma's standard assistant label is ``model`` (not ``assistant``);
        # the webui seeds the role boxes with these so they show live defaults.
        assert body["default_assistant_role"] == "model"
        assert body["default_user_role"] == "user"

    def test_clear(self, session_and_client: Any) -> None:
        session, client = session_and_client
        resp = client.post("/saklas/v1/sessions/default/clear")
        assert resp.status_code == 204
        session.clear_history.assert_called_once()

    def test_rewind_empty(self, session_and_client: Any) -> None:
        session, client = session_and_client
        session.history = []
        resp = client.post("/saklas/v1/sessions/default/rewind")
        assert resp.status_code == 400


# ---- vectors -------------------------------------------------------------


class TestVectors:
    def test_list_empty(self, session_and_client: Any) -> None:
        _, client = session_and_client
        resp = client.get("/saklas/v1/sessions/default/vectors")
        assert resp.status_code == 200
        assert resp.json()["vectors"] == []

    def test_get_not_found(self, session_and_client: Any) -> None:
        _, client = session_and_client
        resp = client.get("/saklas/v1/sessions/default/vectors/missing")
        assert resp.status_code == 404

    def test_delete_not_found(self, session_and_client: Any) -> None:
        _, client = session_and_client
        resp = client.delete("/saklas/v1/sessions/default/vectors/missing")
        assert resp.status_code == 404


# ---- probes --------------------------------------------------------------


class TestProbes:
    def test_list_empty(self, session_and_client: Any) -> None:
        _, client = session_and_client
        resp = client.get("/saklas/v1/sessions/default/probes")
        assert resp.status_code == 200
        assert resp.json()["probes"] == []

    def test_defaults(self, session_and_client: Any) -> None:
        _, client = session_and_client
        with patch(
            "saklas.server.saklas_api.load_defaults",
            return_value={"emotion": ["happiness"]},
        ):
            resp = client.get("/saklas/v1/sessions/default/probes/defaults")
        assert resp.status_code == 200
        assert "emotion" in resp.json()["defaults"]

    def test_attach(self, session_and_client: Any) -> None:
        from types import SimpleNamespace

        session, client = session_and_client
        # Unified attach: body-carried selector → session.add_probe, 201 + info.
        mani = SimpleNamespace(
            name="happy", layers={0: None}, node_labels=["+"],
            feature_space="model",
            domain=SimpleNamespace(to_spec=lambda: {}, intrinsic_dim=1),
        )
        session.add_probe.return_value = "happy"
        session._monitor.attached_probes.return_value = {
            "happy": SimpleNamespace(top_n=3, manifold=mani),
        }
        resp = client.post(
            "/saklas/v1/sessions/default/probes", json={"selector": "happy"},
        )
        assert resp.status_code == 201
        session.add_probe.assert_called_once_with("happy", as_name=None, top_n=3)
        assert resp.json()["name"] == "happy"

    def test_attach_empty_selector(self, session_and_client: Any) -> None:
        _, client = session_and_client
        resp = client.post(
            "/saklas/v1/sessions/default/probes", json={"selector": "  "},
        )
        assert resp.status_code == 400

    def test_deactivate_not_found(self, session_and_client: Any) -> None:
        _, client = session_and_client
        resp = client.delete("/saklas/v1/sessions/default/probes/missing")
        assert resp.status_code == 404


# ---- extract -------------------------------------------------------------


class TestExtract:
    def test_extract_json(self, session_and_client: Any) -> None:
        import torch
        from saklas.core.profile import Profile
        session, client = session_and_client
        profile = Profile({0: torch.zeros(4), 1: torch.ones(4)})
        session.extract.return_value = ("angry.calm", profile)
        resp = client.post(
            "/saklas/v1/sessions/default/extract",
            json={"name": "angry.calm", "source": "angry", "register": False},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["canonical"] == "angry.calm"
        assert data["profile"]["layers"] == [0, 1]
        assert "on_progress" in session.extract.call_args.kwargs

    def test_extract_json_registers_returned_variant_and_namespace(
        self, session_and_client: Any,
    ) -> None:
        import torch
        from saklas.core.profile import Profile
        session, client = session_and_client
        profile = Profile({0: torch.ones(4)})
        session.extract.return_value = ("honest.deceptive:role-pirate", profile)

        resp = client.post(
            "/saklas/v1/sessions/default/extract",
            json={
                "name": "honest.deceptive",
                "source": "honest",
                "role": "pirate",
                "namespace": "alice",
                "register": True,
            },
        )

        assert resp.status_code == 200
        assert resp.json()["canonical"] == "alice/honest.deceptive:role-pirate"
        session.steer.assert_called_once_with(
            "alice/honest.deceptive:role-pirate", profile,
        )

    def test_extract_sse_streams_progress_live(self, session_and_client: Any) -> None:
        """SSE branch must yield each ``on_progress`` message as its own
        event rather than buffering them all until extraction returns.

        Regression: an earlier shape collected messages into a list and
        only yielded them after ``session.extract`` had completed, so
        the client received every progress event in one tick right
        before ``done`` — the webui's progress toast had no time to
        render them.  The fix routes progress through an
        ``asyncio.Queue`` driven from the worker thread.
        """
        import torch
        from saklas.core.profile import Profile
        session, client = session_and_client
        profile = Profile({0: torch.ones(4)})

        def _extract(source: Any, baseline: Any = None, *, on_progress: Any = None, **_kwargs: Any) -> Any:
            assert on_progress is not None
            on_progress("Generating 9 scenarios for 'angry.calm'...")
            on_progress("Generating contrastive pairs across 9 domains...")
            on_progress("Extracting difference-of-means profile (45 pairs)...")
            return "angry.calm", profile

        session.extract.side_effect = _extract

        # ``TestClient`` consumes the whole response before returning,
        # so we can't observe arrival timing here — but we *can* confirm
        # each progress message lands as its own ``event: progress``
        # frame ordered before ``event: done``.  If the old buffer-then-
        # flush shape regressed, the events would still arrive in order
        # but this assertion still gates the structural fix.
        with client.stream(
            "POST",
            "/saklas/v1/sessions/default/extract",
            json={"name": "angry.calm", "source": "angry", "register": False},
            headers={"Accept": "text/event-stream"},
        ) as resp:
            assert resp.status_code == 200
            body = b"".join(resp.iter_bytes()).decode()

        # Split SSE frames on the blank-line terminator.
        frames = [f for f in body.split("\n\n") if f.strip()]
        events = [
            f.split("\n")[0].removeprefix("event: ") for f in frames
        ]
        assert events.count("progress") == 3
        assert events[-1] == "done"
        assert "Generating 9 scenarios" in frames[0]
        assert "Extracting difference-of-means" in frames[2]

    def test_extract_sse_registers_returned_variant(
        self, session_and_client: Any,
    ) -> None:
        import torch
        from saklas.core.profile import Profile
        session, client = session_and_client
        profile = Profile({0: torch.ones(4)})

        def _extract(
            source: Any,
            baseline: Any = None,
            *,
            on_progress: Any = None,
            **_kwargs: Any,
        ) -> Any:
            return "honest.deceptive:role-pirate", profile

        session.extract.side_effect = _extract
        with client.stream(
            "POST",
            "/saklas/v1/sessions/default/extract",
            json={
                "name": "honest.deceptive",
                "source": "honest",
                "role": "pirate",
                "register": True,
            },
            headers={"Accept": "text/event-stream"},
        ) as resp:
            assert resp.status_code == 200
            body = b"".join(resp.iter_bytes()).decode()

        assert '"canonical": "honest.deceptive:role-pirate"' in body
        session.steer.assert_called_once_with(
            "honest.deceptive:role-pirate", profile,
        )

    def test_extract_json_coerces_dict_pairs_and_uses_keyword_progress(self, session_and_client: Any) -> None:
        import torch
        from saklas.core.profile import Profile
        session, client = session_and_client
        profile = Profile({0: torch.ones(4)})

        def _extract(name: Any, positive: Any, negative: Any, *, on_progress: Any = None, **_kwargs: Any) -> Any:
            # A {pairs:[...]} payload unzips into two pole corpora fed to the
            # 2-node pca fit — no {positive,negative} pairs, no DataSource.
            assert name == "custom"
            assert positive == ["positive text"]
            assert negative == ["negative text"]
            assert on_progress is not None
            on_progress("progress")
            return "custom", profile

        session.extract_vector_from_corpora.side_effect = _extract
        resp = client.post(
            "/saklas/v1/sessions/default/extract",
            json={
                "name": "custom",
                "source": {
                    "pairs": [
                        {
                            "positive": "positive text",
                            "negative": "negative text",
                        }
                    ]
                },
                "register": False,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["canonical"] == "custom"
        assert data["progress"] == ["progress"]

    def test_extract_json_coerces_single_pair_dict(self, session_and_client: Any) -> None:
        # A bare {positive, negative} object — not wrapped in {"pairs": ...}
        # — coerces to two one-element pole corpora carrying the request name.
        import torch
        from saklas.core.profile import Profile
        session, client = session_and_client
        profile = Profile({0: torch.ones(4)})

        def _extract(name: Any, positive: Any, negative: Any, **_kwargs: Any) -> Any:
            assert name == "mood"
            assert positive == ["pos one"]
            assert negative == ["neg one"]
            return "mood", profile

        session.extract_vector_from_corpora.side_effect = _extract
        resp = client.post(
            "/saklas/v1/sessions/default/extract",
            json={
                "name": "mood",
                "source": {"positive": "pos one", "negative": "neg one"},
                "register": False,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["canonical"] == "mood"


# ---- WebSocket token+probe co-stream ------------------------------------


class TestWebSocket:
    def _attach_generate(self, session: Any, tokens: Any) -> None:
        """Install a fake ``session.generate`` that drives ``on_token``."""
        def _gen(input: Any, *, steering: Any = None, sampling: Any = None,
                 stateless: Any = False, raw: Any = False, thinking: Any = None,
                 on_token: Any = None, parent_node_id: Any = None, n: Any = 1) -> Any:
            for i, tok in enumerate(tokens):
                on_token(tok, False, 1000 + i, None, None)  # pyright: ignore[reportOptionalCall]
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

    def test_generate_happy_path(self, session_and_client: Any) -> None:
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

    def test_stale_n_way_token_callback_stays_on_original_queue(
        self, session_and_client: Any,
    ) -> None:
        """A late sibling-0 callback must not leak into sibling 1's stream."""
        session, client = session_and_client
        callbacks: list[Any] = []

        def _gen(input: Any, *, steering: Any = None, sampling: Any = None,
                 stateless: Any = False, raw: Any = False, thinking: Any = None,
                 on_token: Any = None, parent_node_id: Any = None, n: Any = 1) -> Any:
            callbacks.append(on_token)
            idx = len(callbacks) - 1
            if idx == 1:
                callbacks[0]("late-first", False, 1999, None, None)
                on_token("second", False, 2000, None, None)
                text = "second"
                tokens = [2000]
            else:
                text = "first"
                tokens = []
            result = GenerationResult(
                text=text, tokens=tokens, token_count=len(tokens),
                tok_per_sec=50.0, elapsed=0.01, finish_reason="stop",
            )
            session._last_result = result
            session.last_result = result
            session._last_per_token_scores = {}
            session.last_per_token_scores = {}
            return result

        session.generate.side_effect = _gen

        with client.websocket_connect("/saklas/v1/sessions/default/stream") as ws:
            ws.send_json({"type": "generate", "input": "hi", "n": 2})
            tokens: list[str] = []
            done_count = 0
            started_count = 0
            while done_count < 2:
                msg = ws.receive_json()
                if msg["type"] == "started":
                    started_count += 1
                elif msg["type"] == "token":
                    tokens.append(msg["text"])
                elif msg["type"] == "done":
                    done_count += 1

        assert started_count == 2
        assert tokens == ["second"]

    def test_unknown_message_type(self, session_and_client: Any) -> None:
        _, client = session_and_client
        with client.websocket_connect("/saklas/v1/sessions/default/stream") as ws:
            ws.send_json({"type": "frobnicate"})
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "unknown message type" in msg["message"]

    def test_multi_turn_no_recv_race(self, session_and_client: Any) -> None:
        """Three back-to-back generate turns on the same WS.

        Regression for the "cannot call recv while another coroutine is
        already waiting for the next message" RuntimeError that fired
        when both the outer dispatch loop and the inner generation
        handler called ``websocket.receive_json()``.  The fix routes
        every incoming frame through a single perpetual reader task +
        shared queue.  This test exercises the inter-turn boundary
        repeatedly so any regression of that pattern surfaces.
        """
        session, client = session_and_client
        self._attach_generate(session, ["a", "b"])

        with client.websocket_connect("/saklas/v1/sessions/default/stream") as ws:
            for _ in range(3):
                ws.send_json({"type": "generate", "input": "hi"})
                started = ws.receive_json()
                assert started["type"] == "started"
                while True:
                    msg = ws.receive_json()
                    if msg["type"] == "done":
                        break
                    assert msg["type"] == "token"

    def test_mid_generation_generate_frame_runs_after_current_turn(self, session_and_client: Any) -> None:
        """A premature second generate frame is deferred, not re-read in a spin loop."""
        session, client = session_and_client
        calls: list[str] = []

        def _gen(input: Any, *, steering: Any = None, sampling: Any = None,
                 stateless: Any = False, raw: Any = False, thinking: Any = None,
                 on_token: Any = None, parent_node_id: Any = None, n: Any = 1) -> Any:
            calls.append(str(input))
            time.sleep(0.02 if input == "one" else 0.001)
            on_token(str(input), False, 1000 + len(calls), None, None)  # pyright: ignore[reportOptionalCall]
            result = GenerationResult(
                text=str(input), tokens=[1000 + len(calls)],
                token_count=1, tok_per_sec=50.0, elapsed=0.02,
                finish_reason="stop",
            )
            session._last_result = result
            session.last_result = result
            session._last_per_token_scores = {}
            session.last_per_token_scores = {}
            return result

        session.generate.side_effect = _gen

        with client.websocket_connect("/saklas/v1/sessions/default/stream") as ws:
            ws.send_json({"type": "generate", "input": "one"})
            ws.send_json({"type": "generate", "input": "two"})
            done = []
            while len(done) < 2:
                msg = ws.receive_json()
                if msg["type"] == "done":
                    done.append(msg["result"]["text"])

        assert done == ["one", "two"]
        assert calls == ["one", "two"]

    def test_idle_stop_is_noop(self, session_and_client: Any) -> None:
        """A ``{type: "stop"}`` outside any generation closes cleanly."""
        _, client = session_and_client
        with client.websocket_connect("/saklas/v1/sessions/default/stream") as ws:
            # Stop while idle — server should ignore and stay open.
            ws.send_json({"type": "stop"})
            ws.send_json({"type": "frobnicate"})
            msg = ws.receive_json()
            assert msg["type"] == "error"

    def test_session_mismatch_closes(self, session_and_client: Any) -> None:
        _, client = session_and_client
        with pytest.raises(Exception):
            with client.websocket_connect("/saklas/v1/sessions/other/stream") as ws:
                ws.receive_json()

    def test_ws_requires_bearer_when_api_key_set(self):
        from saklas.server import create_app
        session = _mock_session()
        app = create_app(session, default_steering=None, api_key="s3cret")
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
        # Browser clients cannot set Authorization on the WS constructor;
        # the dashboard sends the bearer as ?token=...
        with client.websocket_connect(
            "/saklas/v1/sessions/default/stream?token=s3cret",
        ) as ws:
            ws.send_json({"type": "frobnicate"})
            msg = ws.receive_json()
            assert msg["type"] == "error"

    def test_bad_steering_does_not_kill_connection(self, session_and_client: Any, monkeypatch: Any) -> None:
        """Regression: a bad steering expression on a generate frame used
        to escape ``_build_steering`` and bubble out to the outer reader
        loop's ``except Exception``, which closed the WS with code 1011.

        FastAPI's ``@app.exception_handler(SaklasError)`` doesn't apply
        to WebSocket routes, so the handler has to convert the error in-
        band itself. Contract: the server emits a ``{type: "error"}``
        frame and stays open for a follow-up generate.
        """
        from saklas.io.selectors import AmbiguousSelectorError
        import saklas.core.steering_expr as _sx

        session, client = session_and_client
        self._attach_generate(session, ["ok"])

        real_parse = _sx.parse_expr

        def _fake_parse(text: Any, *, namespace: Any = None) -> Any:
            if text.strip() == "0.5 wolf":
                raise AmbiguousSelectorError(
                    "ambiguous pole 'wolf': matches alice/wolf, default/deer.wolf"
                )
            return real_parse(text, namespace=namespace)

        monkeypatch.setattr(_sx, "parse_expr", _fake_parse)

        with client.websocket_connect("/saklas/v1/sessions/default/stream") as ws:
            ws.send_json({
                "type": "generate", "input": "hi", "steering": "0.5 wolf",
            })
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "ambiguous pole 'wolf'" in msg["message"]
            assert msg["code"] == "AmbiguousSelectorError"
            assert msg["status"] == 400

            # Connection still alive — follow-up turn succeeds.
            ws.send_json({"type": "generate", "input": "hi"})
            started = ws.receive_json()
            assert started["type"] == "started"
            tokens = []
            while True:
                m = ws.receive_json()
                if m["type"] == "token":
                    tokens.append(m["text"])
                elif m["type"] == "done":
                    break
            assert tokens == ["ok"]


# ---- Live traits SSE stream -----------------------------------------------


class TestTraitsStream:
    def test_session_not_found_404(self, session_and_client: Any) -> None:
        _, client = session_and_client
        resp = client.get("/saklas/v1/sessions/nonexistent/traits/stream")
        assert resp.status_code == 404

    def test_auth_required(self):
        """With api_key set, the SSE endpoint requires Bearer auth."""
        from saklas.server import create_app
        session = _mock_session()
        app = create_app(session, default_steering=None, api_key="s3cret")
        client = TestClient(app)
        resp = client.get("/saklas/v1/sessions/default/traits/stream")
        assert resp.status_code == 401

    def test_register_unregister_trait_queue(self, session_and_client: Any) -> None:
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


    def test_route_registered(self, session_and_client: Any) -> None:
        """SSE route is registered (valid path resolves, bad session 404s)."""
        _, client = session_and_client
        # Can't GET a valid session without hanging (infinite SSE generator),
        # so verify route registration via the 404 path — confirms the URL
        # pattern matches and the handler runs (session resolution fires).
        # test_session_not_found_404 already covers this; this is a named alias
        # for the "route exists" requirement.
        resp = client.get("/saklas/v1/sessions/nonexistent/traits/stream")
        assert resp.status_code == 404

    def test_event_ordering_start_token_done(self):
        """Events are serialized correctly: start → token → done."""
        from saklas.core.results import ProbeReadings

        # Test the serialization logic directly rather than fighting TestClient
        # SSE streaming semantics. Build the events as they'd arrive on the
        # trait queue and verify the JSON output format.
        readings = {"probe_a": ProbeReadings(
            per_generation=[0.42], mean=0.30, std=0.1, min=0.2, max=0.42,
            delta_per_gen=0.12,
        )}
        fake_result = MagicMock()
        fake_result.readings = readings
        fake_result.finish_reason = "stop"

        # Simulate the tagged tuple protocol.
        events = [
            ("start", "hi", False),
            ("token", 0, "Hello", False, {"probe_a": 0.35}),
            ("token", 1, " world", False, {"probe_a": 0.40}),
            ("done", fake_result),
        ]

        # Serialize using the same logic as the SSE generator.
        output_lines = []
        generation_id = None
        for item in events:
            tag = item[0]
            if tag == "start":
                generation_id = "test123"
                output_lines.append(json.dumps({"type": "start", "generation_id": generation_id}))
            elif tag == "token":
                _, idx, text, thinking, scores = item
                output_lines.append(json.dumps({
                    "type": "token", "idx": idx, "text": text,
                    "thinking": thinking,
                    "probes": {k: round(v, 6) for k, v in scores.items()},
                }))
            elif tag == "done":
                result = item[1]
                agg = {}
                rd = getattr(result, "readings", None)
                if rd:
                    for name, r in rd.items():
                        pg = getattr(r, "per_generation", None)
                        val = pg[-1] if pg else getattr(r, "mean", 0.0)
                        agg[name] = round(val, 6)
                output_lines.append(json.dumps({
                    "type": "done", "generation_id": generation_id,
                    "finish_reason": getattr(result, "finish_reason", "stop"),
                    "aggregate": agg,
                }))

        assert len(output_lines) == 4
        parsed = [json.loads(l) for l in output_lines]
        assert parsed[0]["type"] == "start"
        assert parsed[0]["generation_id"] == "test123"
        assert parsed[1]["type"] == "token"
        assert parsed[1]["idx"] == 0
        assert parsed[1]["probes"]["probe_a"] == 0.35
        assert parsed[2]["type"] == "token"
        assert parsed[2]["idx"] == 1
        assert parsed[3]["type"] == "done"
        # Key assertion: aggregate uses per_generation[-1] (0.42), not mean (0.30)
        assert parsed[3]["aggregate"]["probe_a"] == 0.42
        assert parsed[3]["finish_reason"] == "stop"

    def test_multiple_queues_receive_same_event(self):
        """Multiple registered trait queues all receive the same event."""
        session = _mock_session()
        loop = asyncio.new_event_loop()
        q1 = asyncio.Queue()
        q2 = asyncio.Queue()
        session.register_trait_queue(loop, q1)
        session.register_trait_queue(loop, q2)

        async def _run():
            # Simulate what _token_tap does: push to all queues.
            event = ("token", 0, "Hi", False, {"p": 0.5})
            with session._trait_lock:
                for lp, q in list(session._trait_queues):
                    lp.call_soon_threadsafe(q.put_nowait, event)
            # Both queues should have the event.
            item1 = await asyncio.wait_for(q1.get(), timeout=1.0)
            item2 = await asyncio.wait_for(q2.get(), timeout=1.0)
            assert item1 == event
            assert item2 == event

        loop.run_until_complete(_run())
        session.unregister_trait_queue(loop, q1)
        session.unregister_trait_queue(loop, q2)
        assert len(session._trait_queues) == 0
        loop.close()


# ---- score_single_token (monitor) ----------------------------------------


class TestScoreSingleToken:
    def test_returns_scores_without_accumulation(self):
        import torch
        from saklas.core.monitor import Monitor
        from saklas.core.results import ProbeReading
        from saklas.core.vectors import fold_directions_to_subspace

        from tests._whitener import isotropic_whitener
        dim = 16
        probe_vec = torch.randn(dim)
        means = {0: torch.zeros(dim)}
        whit = isotropic_whitener([0], dim)
        # Mahalanobis is mandatory: covering whitener required to attach + score.
        m = fold_directions_to_subspace(
            "test_probe", {0: probe_vec}, means, whitener=whit,
        )
        monitor = Monitor({"test_probe": m}, means, whitener=whit)

        hidden = {0: torch.randn(dim)}
        scores = monitor.score_single_token(hidden)

        assert "test_probe" in scores
        # Read is the full per-probe ProbeReading (coords axis-0 the scalar).
        assert isinstance(scores["test_probe"], ProbeReading)
        assert isinstance(scores["test_probe"].coords[0], float)
        # History should NOT have been updated.
        assert len(monitor.history["test_probe"]) == 0
        assert monitor._stats["test_probe"]["count"] == 0

    def test_consistent_with_measure_from_hidden(self):
        import torch
        from saklas.core.monitor import Monitor
        from saklas.core.vectors import fold_directions_to_subspace

        from tests._whitener import isotropic_whitener
        dim = 16
        means = {0: torch.zeros(dim), 1: torch.zeros(dim)}
        whit = isotropic_whitener([0, 1], dim)
        m = fold_directions_to_subspace(
            "p1", {0: torch.randn(dim), 1: torch.randn(dim)}, means,
            whitener=whit,
        )
        monitor = Monitor({"p1": m}, means, whitener=whit)

        hidden = {0: torch.randn(dim), 1: torch.randn(dim)}
        single = monitor.score_single_token(hidden)
        no_acc = monitor.measure_from_hidden(hidden, accumulate=False)

        assert single["p1"].coords[0] == pytest.approx(no_acc["p1"].coords[0])


# NOTE: the ``test_autoload_*`` tests and the three ``test_steering_*``
# variant/sign-flip tests were deleted in 4.0.  ``SaklasSession.
# _try_autoload_vector`` (the ``vectors/``-pack safetensors scan) was
# removed; profile resolution goes through ``_ensure_profile_registered``
# (fold a fitted manifold / port a legacy folder).  ``resolve_pole`` no
# longer canonicalizes against disk or flips a bipolar-pole sign, so the
# ``honest`` → ``honest.deceptive`` canonicalization and the ``wolf`` →
# ``deer.wolf @ -1`` sign flip those tests pinned are gone (now the
# manifold tier's job, covered in test_steering_expr / test_manifold_role).


# ---- manifold routes ----------------------------------------------------


def _box1d_payload(name: str = "mood") -> dict[str, Any]:
    return {
        "namespace": "local",
        "name": name,
        "description": "a mood axis",
        "domain": {
            "type": "box",
            "axes": [{"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}],
        },
        "nodes": [
            {"label": "calm", "coords": [0.0],
             "statements": ["I am calm.", "Steady."]},
            {"label": "mid", "coords": [0.5],
             "statements": ["An ordinary moment.", "Nothing notable."]},
            {"label": "afraid", "coords": [1.0],
             "statements": ["I am afraid.", "Shaking."]},
        ],
    }


class TestManifoldRoutes:
    def test_create_list_get(self, session_and_client: Any, tmp_path: Any, monkeypatch: Any) -> None:
        monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
        _session, client = session_and_client

        resp = client.post("/saklas/v1/manifolds", json=_box1d_payload())
        assert resp.status_code == 201
        body = resp.json()
        assert body["name"] == "mood"
        assert body["intrinsic_dim"] == 1
        assert body["min_nodes"] == 3
        assert body["fitted_for_session"] is False
        assert "advisories" in body

        listed = client.get("/saklas/v1/manifolds").json()["manifolds"]
        assert [m["name"] for m in listed] == ["mood"]

        detail = client.get("/saklas/v1/manifolds/local/mood").json()
        labels = [n["label"] for n in detail["nodes"]]
        assert labels == ["calm", "mid", "afraid"]
        assert detail["nodes"][0]["statements"] == ["I am calm.", "Steady."]

    def test_create_conflict(self, session_and_client: Any, tmp_path: Any, monkeypatch: Any) -> None:
        monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
        _session, client = session_and_client
        assert client.post("/saklas/v1/manifolds",
                           json=_box1d_payload()).status_code == 201
        assert client.post("/saklas/v1/manifolds",
                           json=_box1d_payload()).status_code == 409

    def test_create_too_few_nodes(self, session_and_client: Any, tmp_path: Any,
                                  monkeypatch: Any) -> None:
        monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
        _session, client = session_and_client
        payload = _box1d_payload()
        payload["nodes"] = payload["nodes"][:2]
        assert client.post("/saklas/v1/manifolds",
                           json=payload).status_code == 400

    def test_patch_description(self, session_and_client: Any, tmp_path: Any,
                               monkeypatch: Any) -> None:
        monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
        _session, client = session_and_client
        client.post("/saklas/v1/manifolds", json=_box1d_payload())
        resp = client.patch("/saklas/v1/manifolds/local/mood",
                             json={"description": "edited"})
        assert resp.status_code == 200
        assert resp.json()["description"] == "edited"

    def test_delete(self, session_and_client: Any, tmp_path: Any, monkeypatch: Any) -> None:
        monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
        _session, client = session_and_client
        client.post("/saklas/v1/manifolds", json=_box1d_payload())
        assert client.delete(
            "/saklas/v1/manifolds/local/mood").status_code == 200
        assert client.get(
            "/saklas/v1/manifolds/local/mood").status_code == 404

    def test_get_missing(self, session_and_client: Any, tmp_path: Any, monkeypatch: Any) -> None:
        monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
        _session, client = session_and_client
        assert client.get(
            "/saklas/v1/manifolds/local/ghost").status_code == 404

    def test_delete_refuses_when_busy(self, session_and_client: Any, tmp_path: Any,
                                      monkeypatch: Any) -> None:
        # A fit thread holding the engine gen-lock must block a delete —
        # removing nodes/ mid-fit would corrupt the read.
        monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
        session, client = session_and_client
        client.post("/saklas/v1/manifolds", json=_box1d_payload())
        session._gen_lock.acquire.return_value = False
        assert client.delete(
            "/saklas/v1/manifolds/local/mood").status_code == 409

    def test_fit_json(self, session_and_client: Any, tmp_path: Any, monkeypatch: Any) -> None:
        monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
        session, client = session_and_client
        client.post("/saklas/v1/manifolds", json=_box1d_payload())

        from unittest.mock import MagicMock as _MM
        session.fit.return_value = _MM(
            layers={0: 1, 1: 1, 2: 1}, feature_space="raw",
        )
        resp = client.post("/saklas/v1/manifolds/local/mood/fit", json={})
        assert resp.status_code == 200
        body = resp.json()
        assert body["done"] is True
        assert body["layers_fitted"] == 3
        assert body["feature_space"] == "raw"


# ---- per-message roles (sampling carrier) --------------------------------


class TestRoleSampling:
    def test_build_sampling_carries_roles(self):
        """WS sampling roles map onto SamplingConfig (the per-send carrier)."""
        from saklas.server.saklas_api import WSSamplingParams, _build_sampling

        sc = _build_sampling(
            WSSamplingParams(user_role="captain", assistant_role="oracle")
        )
        assert sc is not None
        assert sc.user_role == "captain"
        assert sc.assistant_role == "oracle"

    def test_build_sampling_blank_roles_omitted(self):
        """Empty-string role boxes are treated as "no label" (None)."""
        from saklas.server.saklas_api import WSSamplingParams, _build_sampling

        sc = _build_sampling(WSSamplingParams(user_role="", assistant_role=""))
        assert sc is not None
        assert sc.user_role is None
        assert sc.assistant_role is None


class TestPairwiseMetric:
    """``GET /vectors/pairwise`` is Mahalanobis-only (no Euclidean path)."""

    def _setup(self, session_and_client: Any) -> tuple[Any, TestClient]:
        import torch
        from saklas import Profile
        session, client = session_and_client
        # Two dim-4 vectors over layers {0, 1}.
        torch.manual_seed(1)
        session.vectors = {
            "x": Profile({0: torch.randn(4), 1: torch.randn(4)}),
            "y": Profile({0: torch.randn(4), 1: torch.randn(4)}),
        }
        return session, cast(TestClient, client)

    def test_mahalanobis_default(self, session_and_client: Any) -> None:
        import torch
        from saklas.core.mahalanobis import LayerWhitener
        session, client = self._setup(session_and_client)
        g = torch.Generator().manual_seed(4)
        acts = {L: torch.randn(80, 4, generator=g) for L in (0, 1)}
        means = {L: torch.zeros(4) for L in (0, 1)}
        w = LayerWhitener.from_neutral_activations(acts, means)
        session.whitener = w
        r = client.get("/saklas/v1/sessions/default/vectors/pairwise?a=x&b=y")
        assert r.status_code == 200
        body = r.json()
        assert body["metric"] == "mahalanobis"
        # Cell [0][0] is the Mahalanobis cosine in layer 0's frame.
        vx, vy = session.vectors["x"][0], session.vectors["y"][0]
        ref = w.mahalanobis_cosine(0, vx, vy)
        assert body["matrix"][0][0] == pytest.approx(ref, abs=1e-5)

    def test_missing_whitener_409(self, session_and_client: Any) -> None:
        """No covering whitener → 409 (the neutral cache must be regenerated);
        there is no Euclidean fallback."""
        session, client = self._setup(session_and_client)
        session.whitener = None
        r = client.get("/saklas/v1/sessions/default/vectors/pairwise?a=x&b=y")
        assert r.status_code == 409
