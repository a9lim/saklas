"""Tests for the OpenAI-compatible API server (no GPU required)."""

import asyncio
import json
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from saklas.core.results import GenerationResult, TokenEvent
from saklas.core.session import ConcurrentGenerationError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _mock_session():
    """Create a mock SaklasSession with realistic attributes."""
    session = MagicMock()
    session.model_id = "test/model"
    session.model_info = {
        "model_type": "gemma2",
        "num_layers": 26,
        "hidden_dim": 2304,
        "vram_used_gb": 5.2,
        "param_count": 2_614_000_000,
        "dtype": "torch.bfloat16",
    }

    session.config = MagicMock()
    session.config.temperature = 1.0
    session.config.top_p = 0.9
    session.config.max_new_tokens = 1024
    session.config.system_prompt = None

    session.vectors = {}
    session.probes = {}
    session.history = []

    # Gen state carries the real finish_reason after each generation.
    gen_state = MagicMock()
    gen_state.finish_reason = "stop"
    session._gen_state = gen_state
    session._last_result = None
    session._tokenizer = MagicMock()
    session._tokenizer.decode.side_effect = lambda ids: f"<{ids[0]}>" if ids else ""

    session.build_readings.return_value = {}
    # Real asyncio.Lock so `async with session.lock:` works under the
    # FastAPI test client's event loop.
    session.lock = asyncio.Lock()
    return session


@pytest.fixture
def client():
    from saklas.server import create_app
    from saklas.core.steering import Steering
    session = _mock_session()
    app = create_app(session, default_steering=Steering(alphas={"test_vec": 0.1}))
    return TestClient(app)


@pytest.fixture
def session_and_client():
    from saklas.server import create_app
    session = _mock_session()
    app = create_app(session, default_steering=None)
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

    def test_with_steering_string(self, session_and_client):
        session, client = session_and_client
        session.generate.return_value = GenerationResult(
            text="Ok", tokens=[1], token_count=1,
            tok_per_sec=5.0, elapsed=0.2,
        )
        resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "test"}],
            "steering": "0.3 vec1",
        })
        assert resp.status_code == 200
        call_kwargs = session.generate.call_args[1]
        steering = call_kwargs["steering"]
        assert steering is not None
        assert dict(steering.alphas) == {"vec1": 0.3}
        assert "orthogonalize" not in call_kwargs

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
        assert len(lines) >= 4  # role + 2 content chunks + final + [DONE]

        # First chunk is the role delta (OpenAI convention)
        chunk0 = json.loads(lines[0].removeprefix("data: "))
        assert chunk0["choices"][0]["delta"] == {"role": "assistant"}
        chunk1 = json.loads(lines[1].removeprefix("data: "))
        assert chunk1["choices"][0]["delta"]["content"] == "Hello"

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

    def test_sampling_overrides_ride_on_sampling_config(self, session_and_client):
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
        # session.config is never mutated — overrides ride on SamplingConfig.
        sc = session.generate.call_args[1]["sampling"]
        assert sc.temperature == 0.5
        assert sc.top_p == 0.8
        assert sc.max_tokens == 256
        # Session defaults untouched.
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
# CLI arg parsing
# ---------------------------------------------------------------------------

class TestCLIParsing:
    def test_tui_default(self):
        from saklas.cli import parse_args
        args = parse_args(["tui", "google/gemma-2-2b-it"])
        assert args.command == "tui"
        assert args.model == "google/gemma-2-2b-it"

    def test_serve_subcommand(self):
        from saklas.cli import parse_args
        args = parse_args(["serve", "google/gemma-2-2b-it", "--port", "9000"])
        assert args.command == "serve"
        assert args.model == "google/gemma-2-2b-it"
        assert args.port == 9000

    def test_serve_steer_flag(self):
        from saklas.cli import parse_args
        args = parse_args([
            "serve", "m", "--steer", "0.2 cheerful + 0.3 warm",
        ])
        assert args.steer == "0.2 cheerful + 0.3 warm"

    def test_serve_cors(self):
        from saklas.cli import parse_args
        args = parse_args(["serve", "m", "--cors", "http://localhost:3000", "--cors", "*"])
        assert args.cors == ["http://localhost:3000", "*"]

    # Cache-op coverage lives in tests/test_cache_ops.py (delete_tensors
    # across concept/tag/model selectors) and tests/test_cli_flags.py
    # (the -r/-x/-i/-l/-m cache-ops flag grammar).


# ---------------------------------------------------------------------------
# Ollama-compatible /api/* routes
# ---------------------------------------------------------------------------

class TestOllamaApi:
    def test_version(self, client):
        resp = client.get("/api/version")
        assert resp.status_code == 200
        data = resp.json()
        assert "version" in data
        assert data["version"].startswith("saklas-")

    def test_tags_lists_loaded_model(self, client):
        resp = client.get("/api/tags")
        assert resp.status_code == 200
        models = resp.json()["models"]
        assert len(models) >= 1
        names = [m["name"] for m in models]
        assert "test/model" in names
        first = models[0]
        for key in ("name", "model", "modified_at", "size", "digest", "details"):
            assert key in first
        assert first["digest"].startswith("sha256:")
        assert first["details"]["format"] == "safetensors"
        assert first["details"]["family"] == "gemma2"
        assert first["details"]["parameter_size"] == "2.6B"
        assert first["details"]["quantization_level"] == "BF16"

    def test_tags_advertises_aliases_for_known_model(self):
        from saklas.server import create_app
        session = _mock_session()
        session.model_id = "google/gemma-2-2b-it"
        app = create_app(session)
        c = TestClient(app)
        names = [m["name"] for m in c.get("/api/tags").json()["models"]]
        assert "google/gemma-2-2b-it" in names
        assert "gemma2:2b" in names

    def test_ps(self, client):
        resp = client.get("/api/ps")
        assert resp.status_code == 200
        entries = resp.json()["models"]
        assert len(entries) >= 1
        assert "expires_at" in entries[0]
        assert "size_vram" in entries[0]

    def test_show(self, client):
        resp = client.post("/api/show", json={"model": "test/model"})
        assert resp.status_code == 200
        data = resp.json()
        assert "modelfile" in data
        assert "details" in data
        assert "model_info" in data
        assert data["details"]["family"] == "gemma2"
        assert data["model_info"]["general.architecture"] == "gemma2"
        assert data["model_info"]["gemma2.block_count"] == 26
        assert data["model_info"]["saklas.loaded_model"] == "test/model"

    def test_chat_non_streaming(self, session_and_client):
        session, client = session_and_client
        session.generate.return_value = GenerationResult(
            text="Hello there!", tokens=[1, 2, 3], token_count=3, prompt_tokens=2,
            tok_per_sec=10.0, elapsed=0.3,
        )
        resp = client.post("/api/chat", json={
            "model": "test/model",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": False,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["done"] is True
        assert data["done_reason"] == "stop"
        assert data["message"]["role"] == "assistant"
        assert data["message"]["content"] == "Hello there!"
        assert data["model"] == "test/model"
        assert data["eval_count"] == 3
        assert data["prompt_eval_count"] == 2
        assert data["total_duration"] > 0
        # Session should have been called with the translated messages.
        messages = session.generate.call_args[0][0]
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hi"

    def test_chat_with_system_field(self, session_and_client):
        session, client = session_and_client
        session.generate.return_value = GenerationResult(
            text="ok", tokens=[1], token_count=1, prompt_tokens=5,
            tok_per_sec=5.0, elapsed=0.1,
        )
        resp = client.post("/api/chat", json={
            "messages": [{"role": "user", "content": "Hi"}],
            "system": "You are a pirate.",
            "stream": False,
        })
        assert resp.status_code == 200
        msgs = session.generate.call_args[0][0]
        assert msgs[0] == {"role": "system", "content": "You are a pirate."}
        assert msgs[1]["role"] == "user"

    def test_chat_options_passthrough(self, session_and_client):
        session, client = session_and_client
        session.generate.return_value = GenerationResult(
            text="ok", tokens=[1], token_count=1, prompt_tokens=1,
            tok_per_sec=5.0, elapsed=0.1,
        )
        resp = client.post("/api/chat", json={
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": False,
            "options": {
                "temperature": 0.2, "top_p": 0.7, "seed": 42,
                "num_predict": 64, "stop": ["\n\n"],
                "steer": "0.3 vec1",
            },
        })
        assert resp.status_code == 200
        kw = session.generate.call_args[1]
        sc = kw["sampling"]
        assert sc.seed == 42
        assert sc.stop == ("\n\n",)
        assert sc.temperature == 0.2
        assert sc.top_p == 0.7
        assert sc.max_tokens == 64
        steering = kw["steering"]
        assert steering is not None
        assert dict(steering.alphas) == {"vec1": 0.3}
        # Session defaults untouched — sampling overrides ride on SamplingConfig.
        assert session.config.temperature == 1.0
        assert session.config.top_p == 0.9
        assert session.config.max_new_tokens == 1024

    def test_chat_repeat_penalty_maps_to_presence_penalty(self, session_and_client):
        # Ollama's repeat_penalty divides positive logits by the penalty,
        # which is equivalent to subtracting ln(penalty) from the logit.
        # That matches presence_penalty semantics (subtract a constant per
        # seen token, count-independent), not frequency_penalty (count-weighted).
        import math

        session, client = session_and_client
        session.generate.return_value = GenerationResult(
            text="ok", tokens=[1], token_count=1, prompt_tokens=1,
            tok_per_sec=5.0, elapsed=0.1,
        )
        resp = client.post("/api/chat", json={
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": False,
            "options": {"repeat_penalty": 1.3},
        })
        assert resp.status_code == 200
        kw = session.generate.call_args[1]
        sc = kw["sampling"]
        assert abs(sc.presence_penalty - math.log(1.3)) < 1e-6
        assert sc.frequency_penalty == 0.0

    def test_chat_streaming(self, session_and_client):
        session, client = session_and_client

        def _mock_stream(*args, **kwargs):
            yield TokenEvent(text="Hello", token_id=1, index=0)
            yield TokenEvent(text=" world", token_id=2, index=1)

        session.generate_stream.side_effect = _mock_stream
        session._last_result = GenerationResult(
            text="Hello world", tokens=[1, 2], token_count=2, prompt_tokens=3,
            tok_per_sec=5.0, elapsed=0.4,
        )

        resp = client.post("/api/chat", json={
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        })
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/x-ndjson")
        lines = [l for l in resp.text.strip().split("\n") if l]
        assert len(lines) >= 3  # 2 content + 1 final
        chunks = [json.loads(l) for l in lines]
        # Intermediate chunks carry content tokens and done=False.
        assert chunks[0]["done"] is False
        assert chunks[0]["message"]["content"] == "Hello"
        assert chunks[1]["message"]["content"] == " world"
        # Final chunk has done=True with duration stats.
        final = chunks[-1]
        assert final["done"] is True
        assert final["done_reason"] == "stop"
        assert final["eval_count"] == 2
        assert final["prompt_eval_count"] == 3
        assert final["message"]["content"] == ""

    def test_generate_non_streaming(self, session_and_client):
        session, client = session_and_client
        session.generate.return_value = GenerationResult(
            text="42", tokens=[1], token_count=1, prompt_tokens=1,
            tok_per_sec=5.0, elapsed=0.1,
        )
        resp = client.post("/api/generate", json={
            "model": "test/model",
            "prompt": "What is 6 times 7?",
            "stream": False,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["response"] == "42"
        assert data["done"] is True
        # saklas intentionally omits `context` since it can't round-trip
        # Ollama's tokenized continuation state honestly.
        assert "context" not in data
        # Matching Ollama: /api/generate applies the chat template by default;
        # callers must set "raw": true to bypass it.
        assert session.generate.call_args[1]["raw"] is False

    def test_generate_raw_mode(self, session_and_client):
        session, client = session_and_client
        session.generate.return_value = GenerationResult(
            text="x", tokens=[1], token_count=1, prompt_tokens=1,
            tok_per_sec=5.0, elapsed=0.1,
        )
        resp = client.post("/api/generate", json={
            "prompt": "raw prompt",
            "stream": False,
            "raw": True,
        })
        assert resp.status_code == 200
        assert session.generate.call_args[1]["raw"] is True

    def test_generate_with_system_uses_chat_template(self, session_and_client):
        session, client = session_and_client
        session.generate.return_value = GenerationResult(
            text="arrr", tokens=[1], token_count=1, prompt_tokens=2,
            tok_per_sec=5.0, elapsed=0.1,
        )
        resp = client.post("/api/generate", json={
            "prompt": "Hello",
            "system": "You are a pirate.",
            "stream": False,
        })
        assert resp.status_code == 200
        # With system, we switch off raw mode and build a message list.
        assert session.generate.call_args[1]["raw"] is False
        msgs = session.generate.call_args[0][0]
        assert msgs[0]["role"] == "system"

    def test_pull_known_model_is_success(self, client):
        resp = client.post("/api/pull", json={"model": "test/model"})
        assert resp.status_code == 200
        lines = [l for l in resp.text.strip().split("\n") if l]
        last = json.loads(lines[-1])
        assert last["status"] == "success"

    def test_pull_unknown_model_404(self, client):
        resp = client.post("/api/pull", json={"model": "nope:latest"})
        assert resp.status_code == 404

    def test_embeddings_not_implemented(self, client):
        resp = client.post("/api/embeddings", json={"model": "test/model", "prompt": "hi"})
        assert resp.status_code == 501

    def test_ollama_routes_respect_api_key(self):
        from saklas.server import create_app
        session = _mock_session()
        app = create_app(session, api_key="secret")
        c = TestClient(app)
        # No auth -> 401
        assert c.get("/api/tags").status_code == 401
        # Wrong scheme -> 401
        assert c.get("/api/tags", headers={"Authorization": "Basic secret"}).status_code == 401
        # Correct key -> 200
        resp = c.get("/api/tags", headers={"Authorization": "Bearer secret"})
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Cluster 4: LangChain compat, native steering field, session.lock back-pressure
# ---------------------------------------------------------------------------

class TestLangChainCompat:
    def test_empty_tools_accepted(self, session_and_client):
        session, client = session_and_client
        session.generate.return_value = GenerationResult(
            text="hi", tokens=[1], token_count=1, tok_per_sec=1.0, elapsed=0.1,
        )
        resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [],
            "tool_choice": "none",
            "response_format": {"type": "text"},
        })
        assert resp.status_code == 200

    def test_non_empty_tools_rejected(self, session_and_client):
        session, client = session_and_client
        resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "x", "parameters": {}}}],
        })
        assert resp.status_code == 400
        assert "tool" in resp.json()["error"]["message"].lower()

    def test_required_tool_choice_rejected(self, session_and_client):
        session, client = session_and_client
        resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hi"}],
            "tool_choice": "required",
        })
        assert resp.status_code == 400

    def test_response_format_text_accepted(self, session_and_client):
        session, client = session_and_client
        session.generate.return_value = GenerationResult(
            text="hi", tokens=[1], token_count=1, tok_per_sec=1.0, elapsed=0.1,
        )
        resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {"type": "text"},
        })
        assert resp.status_code == 200

    def test_response_format_json_object_rejected(self, session_and_client):
        session, client = session_and_client
        resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {"type": "json_object"},
        })
        assert resp.status_code == 400

    def test_response_format_json_schema_rejected(self, session_and_client):
        session, client = session_and_client
        resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "x"}},
        })
        assert resp.status_code == 400


class TestNativeSteeringField:
    def test_top_level_steering_expression(self, session_and_client):
        session, client = session_and_client
        session.generate.return_value = GenerationResult(
            text="ok", tokens=[1], token_count=1, tok_per_sec=1.0, elapsed=0.1,
        )
        resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hi"}],
            "steering": "0.5 angry.calm",
        })
        assert resp.status_code == 200
        kw = session.generate.call_args[1]
        assert kw["steering"] is not None
        assert kw["steering"].alphas == {"angry.calm": 0.5}

    def test_steering_projection_term(self, session_and_client):
        session, client = session_and_client
        session.generate.return_value = GenerationResult(
            text="ok", tokens=[1], token_count=1, tok_per_sec=1.0, elapsed=0.1,
        )
        resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hi"}],
            "steering": "-0.4 wolf",
        })
        assert resp.status_code == 200
        kw = session.generate.call_args[1]
        # Parser resolves ``wolf`` → identity with sign +1 when nothing is
        # installed; coefficient carries through as -0.4.
        assert kw["steering"].alphas == {"wolf": -0.4}

    def test_steering_merges_with_server_defaults(self):
        from saklas.server import create_app
        from saklas.core.steering import Steering
        session = _mock_session()
        session.generate.return_value = GenerationResult(
            text="ok", tokens=[1], token_count=1, tok_per_sec=1.0, elapsed=0.1,
        )
        app = create_app(session, default_steering=Steering(alphas={"base": 0.2}))
        c = TestClient(app)
        resp = c.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hi"}],
            "steering": "0.7 override",
        })
        assert resp.status_code == 200
        kw = session.generate.call_args[1]
        assert kw["steering"].alphas == {"base": 0.2, "override": 0.7}

    def test_steering_request_overrides_default(self):
        from saklas.server import create_app
        from saklas.core.steering import Steering
        session = _mock_session()
        session.generate.return_value = GenerationResult(
            text="ok", tokens=[1], token_count=1, tok_per_sec=1.0, elapsed=0.1,
        )
        # Use a key outside the bundled-probe vocabulary so resolve_pole
        # doesn't reroute "myvec" to a bipolar canonical.
        app = create_app(session, default_steering=Steering(alphas={"myvec": 0.1}))
        c = TestClient(app)
        resp = c.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hi"}],
            "steering": "0.7 myvec",
        })
        assert resp.status_code == 200
        kw = session.generate.call_args[1]
        assert kw["steering"].alphas == {"myvec": 0.7}

    def test_thinking_field_default_is_none_auto(self, session_and_client):
        session, client = session_and_client
        session.generate.return_value = GenerationResult(
            text="ok", tokens=[1], token_count=1, tok_per_sec=1.0, elapsed=0.1,
        )
        resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert resp.status_code == 200
        kw = session.generate.call_args[1]
        assert kw["thinking"] is None

    def test_thinking_explicit_false(self, session_and_client):
        session, client = session_and_client
        session.generate.return_value = GenerationResult(
            text="ok", tokens=[1], token_count=1, tok_per_sec=1.0, elapsed=0.1,
        )
        resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hi"}],
            "thinking": False,
        })
        assert resp.status_code == 200
        kw = session.generate.call_args[1]
        assert kw["thinking"] is False


class TestSessionLockBackpressure:
    def test_acquire_session_lock_queues_fifo(self):
        """Two async waiters on ``session.lock`` run in order, not in parallel.

        The real FastAPI path parks each request's ``async with session.lock``
        on the same asyncio.Lock; this test drives ``acquire_session_lock``
        directly under one event loop to prove queuing works without
        introducing cross-loop lock state (which TestClient's per-request
        thread+loop model cannot exercise honestly).
        """
        import asyncio as _asyncio
        from saklas.server import acquire_session_lock

        session = _mock_session()
        order: list[str] = []

        async def _waiter(tag: str, hold: float) -> None:
            async with acquire_session_lock(session) as acquired:
                assert acquired
                order.append(f"{tag}:enter")
                await _asyncio.sleep(hold)
                order.append(f"{tag}:exit")

        async def _driver():
            t1 = _asyncio.create_task(_waiter("a", 0.05))
            # Yield so t1 enters first.
            await _asyncio.sleep(0)
            t2 = _asyncio.create_task(_waiter("b", 0.0))
            await _asyncio.gather(t1, t2)

        _asyncio.run(_driver())
        # Exact interleave proves b waited for a's exit before entering.
        assert order == ["a:enter", "a:exit", "b:enter", "b:exit"]

    def test_no_app_state_gen_lock(self):
        """``app.state.gen_lock`` is gone; all serialization is on session.lock."""
        from saklas.server import create_app
        app = create_app(_mock_session())
        assert not hasattr(app.state, "gen_lock")
