"""Tests for the OpenAI-compatible API server (no GPU required)."""

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from liahona.results import GenerationResult, TokenEvent
from liahona.session import ConcurrentGenerationError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _mock_session():
    """Create a mock LiahonaSession with realistic attributes."""
    session = MagicMock()
    session.model_id = "test/model"
    session.model_info = {
        "model_type": "gemma2",
        "num_layers": 26,
        "hidden_dim": 2304,
        "vram_used_gb": 5.2,
    }

    session.config = MagicMock()
    session.config.temperature = 1.0
    session.config.top_p = 0.9
    session.config.max_new_tokens = 1024
    session.config.system_prompt = None

    session.vectors = {}
    session.probes = {}
    session.history = []

    session.build_readings.return_value = {}
    return session


@pytest.fixture
def client():
    from liahona.server import create_app
    session = _mock_session()
    app = create_app(session, default_alphas={"test_vec": 0.1})
    return TestClient(app)


@pytest.fixture
def session_and_client():
    from liahona.server import create_app
    session = _mock_session()
    app = create_app(session, default_alphas={})
    return session, TestClient(app)


# ---------------------------------------------------------------------------
# Model endpoints
# ---------------------------------------------------------------------------

class TestModels:
    def test_list_models(self, client):
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["id"] == "test/model"
        assert data["data"][0]["owned_by"] == "local"

    def test_get_model(self, client):
        resp = client.get("/v1/models/test/model")
        assert resp.status_code == 200
        assert resp.json()["id"] == "test/model"

    def test_get_model_not_found(self, client):
        resp = client.get("/v1/models/other/model")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Chat completions
# ---------------------------------------------------------------------------

class TestChatCompletions:
    def test_non_streaming(self, session_and_client):
        session, client = session_and_client
        result = GenerationResult(
            text="Hello there!", tokens=[1, 2, 3], token_count=3,
            tok_per_sec=10.0, elapsed=0.3,
        )
        session.generate.return_value = result

        resp = client.post("/v1/chat/completions", json={
            "model": "test/model",
            "messages": [{"role": "user", "content": "Hi"}],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"]["content"] == "Hello there!"
        assert data["choices"][0]["finish_reason"] == "stop"
        assert data["usage"]["completion_tokens"] == 3

        session.generate.assert_called_once()
        call_args = session.generate.call_args
        messages = call_args[0][0]
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hi"

    def test_with_steer_params(self, session_and_client):
        session, client = session_and_client
        session.generate.return_value = GenerationResult(
            text="Ok", tokens=[1], token_count=1,
            tok_per_sec=5.0, elapsed=0.2,
        )
        resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "test"}],
            "steer": {"alphas": {"vec1": 0.3}, "orthogonalize": True},
        })
        assert resp.status_code == 200
        call_kwargs = session.generate.call_args[1]
        assert call_kwargs["alphas"] == {"vec1": 0.3}
        assert call_kwargs["orthogonalize"] is True

    def test_streaming(self, session_and_client):
        session, client = session_and_client

        def _mock_stream(*args, **kwargs):
            yield TokenEvent(text="Hello", token_id=1, index=0)
            yield TokenEvent(text=" world", token_id=2, index=1)

        session.generate_stream.return_value = _mock_stream()

        resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        })
        assert resp.status_code == 200

        lines = [l for l in resp.text.strip().split("\n") if l.startswith("data: ")]
        assert len(lines) >= 3  # 2 content chunks + final + [DONE]

        # First chunk has content
        chunk0 = json.loads(lines[0].removeprefix("data: "))
        assert chunk0["choices"][0]["delta"]["content"] == "Hello"

        # Last data line before [DONE] has finish_reason
        done_idx = next(i for i, l in enumerate(lines) if l == "data: [DONE]")
        final = json.loads(lines[done_idx - 1].removeprefix("data: "))
        assert final["choices"][0]["finish_reason"] == "stop"

    def test_conflict_on_concurrent_generation(self, session_and_client):
        session, client = session_and_client
        session.generate.side_effect = ConcurrentGenerationError("Generation already in progress")

        resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Hi"}],
        })
        assert resp.status_code == 409

    def test_gen_config_override(self, session_and_client):
        session, client = session_and_client
        session.generate.return_value = GenerationResult(
            text="x", tokens=[1], token_count=1,
            tok_per_sec=5.0, elapsed=0.1,
        )
        resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "test"}],
            "temperature": 0.5,
            "top_p": 0.8,
            "max_tokens": 256,
        })
        assert resp.status_code == 200
        # Config should be restored after generation
        assert session.config.temperature == 1.0
        assert session.config.top_p == 0.9
        assert session.config.max_new_tokens == 1024


# ---------------------------------------------------------------------------
# Text completions
# ---------------------------------------------------------------------------

class TestCompletions:
    def test_non_streaming(self, session_and_client):
        session, client = session_and_client
        session.generate.return_value = GenerationResult(
            text="42", tokens=[1], token_count=1,
            tok_per_sec=5.0, elapsed=0.2,
        )
        resp = client.post("/v1/completions", json={
            "prompt": "The answer is",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "text_completion"
        assert data["choices"][0]["text"] == "42"

        call_kwargs = session.generate.call_args[1]
        assert call_kwargs["raw"] is True


# ---------------------------------------------------------------------------
# Vector management
# ---------------------------------------------------------------------------

class TestVectors:
    def test_list_empty(self, session_and_client):
        session, client = session_and_client
        resp = client.get("/v1/liahona/vectors")
        assert resp.status_code == 200
        assert resp.json()["vectors"] == {}

    def test_delete_not_found(self, session_and_client):
        session, client = session_and_client
        resp = client.delete("/v1/liahona/vectors/nonexistent")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Probe management
# ---------------------------------------------------------------------------

class TestProbes:
    def test_list_empty(self, session_and_client):
        session, client = session_and_client
        resp = client.get("/v1/liahona/probes")
        assert resp.status_code == 200
        assert resp.json()["probes"] == {}

    def test_list_defaults(self, session_and_client):
        session, client = session_and_client
        with patch("liahona.server.load_defaults", return_value={"emotion": ["happiness"]}):
            resp = client.get("/v1/liahona/probes/defaults")
        assert resp.status_code == 200
        assert "emotion" in resp.json()["defaults"]

    def test_activate(self, session_and_client):
        session, client = session_and_client
        resp = client.post("/v1/liahona/probes/test_probe", json={})
        assert resp.status_code == 200
        session.monitor.assert_called_once_with("test_probe", None)

    def test_deactivate_not_found(self, session_and_client):
        session, client = session_and_client
        resp = client.delete("/v1/liahona/probes/nonexistent")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

class TestSession:
    def test_get_session(self, session_and_client):
        session, client = session_and_client
        resp = client.get("/v1/liahona/session")
        assert resp.status_code == 200
        data = resp.json()
        assert data["model"] == "test/model"
        assert data["config"]["temperature"] == 1.0

    def test_patch_session(self, session_and_client):
        session, client = session_and_client
        resp = client.patch("/v1/liahona/session", json={
            "temperature": 0.5,
            "system_prompt": "Be concise.",
        })
        assert resp.status_code == 200
        assert session.config.temperature == 0.5
        assert session.config.system_prompt == "Be concise."

    def test_clear(self, session_and_client):
        session, client = session_and_client
        resp = client.post("/v1/liahona/session/clear")
        assert resp.status_code == 204
        session.clear_history.assert_called_once()

    def test_rewind(self, session_and_client):
        session, client = session_and_client
        session.history = [{"role": "user", "content": "hi"}]
        resp = client.post("/v1/liahona/session/rewind")
        assert resp.status_code == 204
        session.rewind.assert_called_once()

    def test_rewind_empty(self, session_and_client):
        session, client = session_and_client
        session.history = []
        resp = client.post("/v1/liahona/session/rewind")
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# CLI arg parsing
# ---------------------------------------------------------------------------

class TestCLIParsing:
    def test_tui_default(self):
        from liahona.cli import parse_args
        args = parse_args(["google/gemma-2-2b-it"])
        assert args.command == "tui"
        assert args.model == "google/gemma-2-2b-it"

    def test_serve_subcommand(self):
        from liahona.cli import parse_args
        args = parse_args(["serve", "google/gemma-2-2b-it", "--port", "9000"])
        assert args.command == "serve"
        assert args.model == "google/gemma-2-2b-it"
        assert args.port == 9000

    def test_serve_steer_flag(self):
        from liahona.cli import parse_args, _parse_steer_flag
        assert _parse_steer_flag("cheerful:0.2") == ("cheerful", 0.2)
        assert _parse_steer_flag("cheerful") == ("cheerful", 0.0)

    def test_serve_cors(self):
        from liahona.cli import parse_args
        args = parse_args(["serve", "m", "--cors", "http://localhost:3000", "--cors", "*"])
        assert args.cors == ["http://localhost:3000", "*"]
