"""Tests for the OpenAI-compatible API server (no GPU required)."""

import asyncio
import json
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from saklas.core.results import GenerationResult, RunSet, TokenEvent
from tests._generation_stream import TestGenerationStream
from saklas.core.session import ConcurrentGenerationError, ProfileNotRegisteredError


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
        "device": "cpu",
        "dtype": "torch.bfloat16",
    }
    session.model = MagicMock()
    session.model.config.model_type = "gemma2"
    session.layers = [MagicMock() for _ in range(26)]

    session.config = MagicMock()
    session.config.temperature = 1.0
    session.config.top_p = 0.9
    session.config.top_k = None
    session.config.max_new_tokens = 1024
    session.config.system_prompt = None
    session.config.thinking = None

    session.profiles = {}
    session.probes = {}
    session.tree = MagicMock()
    session.tree.messages_for.return_value = []
    session.tree.active_node_id = "test-assistant"
    session.tree.get.return_value.mean_logprob = None
    session.tree.get.return_value.mean_surprise = None
    session.is_base_model = False
    session.has_compatible_jlens.return_value = False
    session.live_lens_layers = None
    session.sae_info = None
    session.live_sae = False
    session.live_probe_scores = True
    session.scene_grammar = None
    session.joint_logprob_cache = {}
    session.lens_probe_names = []
    session.sae_probe_names = []
    session.lens_probe_specs = {}
    session.sae_probe_specs = {}
    session.token_probe_payload = {}

    # Gen state carries the real finish_reason after each generation.
    gen_state = MagicMock()
    gen_state.finish_reason = "stop"
    gen_state.emit_map = []
    session.generation_state = gen_state
    session.last_result = None
    session.tokenizer = MagicMock()
    session.tokenizer.decode.side_effect = lambda ids: f"<{ids[0]}>" if ids else ""

    session.build_readings.return_value = {}
    # Real asyncio.Lock so `async with session.lock:` works under the
    # FastAPI test client's event loop.
    session.lock = asyncio.Lock()
    return session


def _single_run(**kwargs: Any) -> RunSet:
    return RunSet([GenerationResult(**kwargs)])


@pytest.fixture
def client():
    from saklas.server import create_app
    from saklas.core.steering import Steering
    session = _mock_session()
    app = create_app(session, default_steering=Steering(alphas={"test_vec": 0.1}))
    # Keep one portal/event loop alive for the whole test.  Routes intentionally
    # launch background tasks whose lifetime extends beyond the response that
    # created them; a bare TestClient may tear its per-request portal down first.
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def session_and_client():
    from saklas.server import create_app
    session = _mock_session()
    app = create_app(session, default_steering=None)
    with TestClient(app) as test_client:
        yield session, test_client


# ---------------------------------------------------------------------------
# Model endpoints
# ---------------------------------------------------------------------------

class TestModels:
    def test_list_models(self, client: Any) -> None:
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["id"] == "test/model"
        assert data["data"][0]["owned_by"] == "local"

    def test_get_model(self, client: Any) -> None:
        resp = client.get("/v1/models/test/model")
        assert resp.status_code == 200
        assert resp.json()["id"] == "test/model"

    def test_get_model_not_found(self, client: Any) -> None:
        resp = client.get("/v1/models/other/model")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Chat completions
# ---------------------------------------------------------------------------

class TestChatCompletions:
    def test_non_streaming(self, session_and_client: Any) -> None:
        session, client = session_and_client
        result = GenerationResult(
            text="Hello there!", tokens=[1, 2, 3], token_count=3,
            tok_per_sec=10.0, elapsed=0.3,
        )
        session.generate.return_value = RunSet([result])

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

    def test_with_steering_string(self, session_and_client: Any) -> None:
        session, client = session_and_client
        session.generate.return_value = _single_run(
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

    def test_streaming(self, session_and_client: Any) -> None:
        session, client = session_and_client

        result = GenerationResult(
            text="Hello world", tokens=[1, 2], token_count=2,
            tok_per_sec=5.0, elapsed=0.4, finish_reason="stop",
        )
        session.generate_stream.return_value = TestGenerationStream([
            TokenEvent(text="Hello", token_id=1, index=0),
            TokenEvent(text=" world", token_id=2, index=1),
        ], result)

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

    def test_streaming_saklas_error_is_sent_in_band(self, session_and_client: Any) -> None:
        session, client = session_and_client

        session.generate_stream.return_value = TestGenerationStream(
            [], None,
            error=ProfileNotRegisteredError("No profile registered for 'missing'"),
        )

        resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
            "steering": "0.5 missing",
        })
        assert resp.status_code == 200

        lines = [l for l in resp.text.strip().split("\n") if l.startswith("data: ")]
        chunks = [
            json.loads(l.removeprefix("data: "))
            for l in lines
            if l != "data: [DONE]"
        ]
        err = next(c["error"] for c in chunks if "error" in c)
        assert err["code"] == 404
        assert err["type"] == "invalid_request_error"
        assert "No profile registered for 'missing'" in err["message"]

    def test_conflict_on_concurrent_generation(self, session_and_client: Any) -> None:
        session, client = session_and_client
        session.generate.side_effect = ConcurrentGenerationError("Generation already in progress")

        resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Hi"}],
        })
        assert resp.status_code == 409

    def test_sampling_overrides_ride_on_sampling_config(self, session_and_client: Any) -> None:
        session, client = session_and_client
        session.generate.return_value = _single_run(
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
    def test_non_streaming(self, session_and_client: Any) -> None:
        session, client = session_and_client
        session.generate.return_value = _single_run(
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

    def test_streaming_disables_unserialized_live_readouts(
        self,
        session_and_client: Any,
    ) -> None:
        session, client = session_and_client
        session.last_result = GenerationResult(
            text="42", tokens=[1], token_count=1,
            tok_per_sec=5.0, elapsed=0.2,
        )
        session.generate_stream.return_value = TestGenerationStream([
            TokenEvent(text="42", token_id=1, index=0),
        ], session.last_result)

        resp = client.post("/v1/completions", json={
            "prompt": "The answer is",
            "stream": True,
        })

        assert resp.status_code == 200
        call_kwargs = session.generate_stream.call_args.kwargs
        assert call_kwargs["live_scores"] is False
        assert call_kwargs["live_readouts"] is False


# ---------------------------------------------------------------------------
# CLI arg parsing
# ---------------------------------------------------------------------------

class TestCLIParsing:
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

    def test_serve_no_web_flag_default_off(self):
        from saklas.cli import parse_args
        # Dashboard is on by default; ``args.no_web`` defaults to False
        # so create_app will receive ``web=True`` from the runner.
        args = parse_args(["serve", "m"])
        assert args.no_web is False

    def test_serve_no_web_flag_opt_out(self):
        from saklas.cli import parse_args
        args = parse_args(["serve", "m", "--no-web"])
        assert args.no_web is True

    # Cache-op coverage lives in tests/test_cache_ops.py (delete_tensors
    # across concept/tag/model selectors) and tests/test_cli_flags.py
    # (the -r/-x/-i/-l/-m cache-ops flag grammar).


# ---------------------------------------------------------------------------
# Ollama-compatible /api/* routes
# ---------------------------------------------------------------------------

class TestOllamaApi:
    def test_version(self, client: Any) -> None:
        resp = client.get("/api/version")
        assert resp.status_code == 200
        data = resp.json()
        assert "version" in data
        assert data["version"].startswith("saklas-")

    def test_tags_lists_loaded_model(self, client: Any) -> None:
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

    def test_ps(self, client: Any) -> None:
        resp = client.get("/api/ps")
        assert resp.status_code == 200
        entries = resp.json()["models"]
        assert len(entries) >= 1
        assert "expires_at" in entries[0]
        assert "size_vram" in entries[0]

    def test_show(self, client: Any) -> None:
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

    def test_chat_non_streaming(self, session_and_client: Any) -> None:
        session, client = session_and_client
        session.generate.return_value = _single_run(
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

    def test_chat_non_streaming_done_reason_comes_from_result(self, session_and_client: Any) -> None:
        session, client = session_and_client
        session.generation_state.finish_reason = "length"
        session.generate.return_value = _single_run(
            text="Hello there!", tokens=[1, 2, 3], token_count=3, prompt_tokens=2,
            tok_per_sec=10.0, elapsed=0.3, finish_reason="stop",
        )
        resp = client.post("/api/chat", json={
            "model": "test/model",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": False,
        })
        assert resp.status_code == 200
        assert resp.json()["done_reason"] == "stop"

    def test_chat_with_system_field(self, session_and_client: Any) -> None:
        session, client = session_and_client
        session.generate.return_value = _single_run(
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

    def test_chat_options_passthrough(self, session_and_client: Any) -> None:
        session, client = session_and_client
        session.generate.return_value = _single_run(
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

    def test_chat_malformed_ollama_option_returns_400(self, session_and_client: Any) -> None:
        session, client = session_and_client
        resp = client.post("/api/chat", json={
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": False,
            "options": {"presence_penalty": {"bad": "type"}},
        })
        assert resp.status_code == 400
        assert "presence_penalty" in resp.json()["error"]
        session.generate.assert_not_called()

    def test_chat_malformed_steer_type_returns_400(self, session_and_client: Any) -> None:
        session, client = session_and_client
        resp = client.post("/api/chat", json={
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": False,
            "options": {"steer": 3},
        })
        assert resp.status_code == 400
        assert "steer" in resp.json()["error"]
        session.generate.assert_not_called()

    def test_chat_repeat_penalty_maps_to_presence_penalty(self, session_and_client: Any) -> None:
        # Ollama's repeat_penalty divides positive logits by the penalty,
        # which is equivalent to subtracting ln(penalty) from the logit.
        # That matches presence_penalty semantics (subtract a constant per
        # seen token, count-independent), not frequency_penalty (count-weighted).
        import math

        session, client = session_and_client
        session.generate.return_value = _single_run(
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

    def test_chat_streaming(self, session_and_client: Any) -> None:
        session, client = session_and_client

        result = GenerationResult(
            text="Hello world", tokens=[1, 2], token_count=2, prompt_tokens=3,
            tok_per_sec=5.0, elapsed=0.4,
        )
        session.generate_stream.return_value = TestGenerationStream([
            TokenEvent(text="Hello", token_id=1, index=0),
            TokenEvent(text=" world", token_id=2, index=1),
        ], result)

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

    def test_chat_streaming_materialization_error_is_ndjson(self, session_and_client: Any) -> None:
        session, client = session_and_client

        session.generate_stream.return_value = TestGenerationStream(
            [], None,
            error=ProfileNotRegisteredError("No profile registered for 'missing'"),
        )

        resp = client.post("/api/chat", json={
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
            "options": {"steer": "0.5 missing"},
        })
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/x-ndjson")
        lines = [json.loads(l) for l in resp.text.strip().split("\n") if l]
        # An in-band error frame, then a terminating ``done`` frame so
        # ollama-python / ChatOllama don't stall waiting for stream end.
        assert lines == [
            {
                "model": "test/model",
                "created_at": lines[0]["created_at"],
                "error": "No profile registered for 'missing'",
            },
            {
                "model": "test/model",
                "created_at": lines[1]["created_at"],
                "done": True,
                "done_reason": "error",
            },
        ]

    def test_generate_non_streaming(self, session_and_client: Any) -> None:
        session, client = session_and_client
        session.generate.return_value = _single_run(
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

    def test_generate_raw_mode(self, session_and_client: Any) -> None:
        session, client = session_and_client
        session.generate.return_value = _single_run(
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

    def test_generate_with_system_uses_chat_template(self, session_and_client: Any) -> None:
        session, client = session_and_client
        session.generate.return_value = _single_run(
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

    def test_pull_known_model_is_success(self, client: Any) -> None:
        resp = client.post("/api/pull", json={"model": "test/model"})
        assert resp.status_code == 200
        lines = [l for l in resp.text.strip().split("\n") if l]
        last = json.loads(lines[-1])
        assert last["status"] == "success"

    def test_pull_unknown_model_404(self, client: Any) -> None:
        resp = client.post("/api/pull", json={"model": "nope:latest"})
        assert resp.status_code == 404

    def test_embeddings_not_implemented(self, client: Any) -> None:
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

    def test_chat_streaming_bad_steer_returns_400_not_mid_stream_disconnect(
        self, session_and_client: Any, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Regression: ``options.steer`` parsing used to live inside the
        NDJSON streaming generator. By the time ``parse_expr`` raised,
        ``StreamingResponse`` had already flushed 200 OK headers, so the
        FastAPI ``SaklasError`` handler couldn't rewrite the response —
        the client saw a TCP cutoff mid-stream with no body.

        Fix: option resolution is hoisted to the route handler, so a bad
        steering expression now surfaces as the canonical Ollama-shape
        ``{"error": "..."}`` 400.
        """
        from saklas.io.selectors import AmbiguousSelectorError
        import saklas.core.steering_expr as _sx

        session, client = session_and_client

        real_parse = _sx.parse_expr

        def _fake_parse(text: str, *, namespace: str | None = None) -> Any:
            if text.strip() == "0.5 wolf":
                raise AmbiguousSelectorError(
                    "ambiguous pole 'wolf': matches alice/wolf, default/deer.wolf"
                )
            return real_parse(text, namespace=namespace)

        monkeypatch.setattr(_sx, "parse_expr", _fake_parse)

        resp = client.post("/api/chat", json={
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
            "options": {"steer": "0.5 wolf"},
        })
        # Clean 400 with Ollama error shape — not a 200 OK that cuts off.
        assert resp.status_code == 400
        assert resp.headers["content-type"].startswith("application/json")
        body = resp.json()
        assert "error" in body
        assert "ambiguous pole 'wolf'" in body["error"]
        # ``session.generate_stream`` was never called — the stream never
        # started, so there's nothing to clean up.
        session.generate_stream.assert_not_called()

    def test_chat_non_streaming_bad_steer_returns_400(
        self, session_and_client: Any, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Belt-and-suspenders: non-streaming Ollama already routed
        ``SaklasError`` through the FastAPI handler. Pin the contract."""
        from saklas.io.selectors import AmbiguousSelectorError
        import saklas.core.steering_expr as _sx

        session, client = session_and_client

        def _fake_parse(text: str, *, namespace: str | None = None) -> Any:
            raise AmbiguousSelectorError("ambiguous pole 'wolf': matches a/wolf, b/wolf")
        monkeypatch.setattr(_sx, "parse_expr", _fake_parse)

        resp = client.post("/api/chat", json={
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": False,
            "options": {"steer": "0.5 wolf"},
        })
        assert resp.status_code == 400
        assert resp.json()["error"].startswith("ambiguous pole 'wolf'")
        session.generate.assert_not_called()


# ---------------------------------------------------------------------------
# Cluster 4: LangChain compat, native steering field, session.lock back-pressure
# ---------------------------------------------------------------------------

class TestLangChainCompat:
    def test_empty_tools_accepted(self, session_and_client: Any) -> None:
        session, client = session_and_client
        session.generate.return_value = _single_run(
            text="hi", tokens=[1], token_count=1, tok_per_sec=1.0, elapsed=0.1,
        )
        resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [],
            "tool_choice": "none",
            "response_format": {"type": "text"},
        })
        assert resp.status_code == 200

    def test_non_empty_tools_rejected(self, session_and_client: Any) -> None:
        session, client = session_and_client
        resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "x", "parameters": {}}}],
        })
        assert resp.status_code == 400
        assert "tool" in resp.json()["error"]["message"].lower()

    def test_required_tool_choice_rejected(self, session_and_client: Any) -> None:
        session, client = session_and_client
        resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hi"}],
            "tool_choice": "required",
        })
        assert resp.status_code == 400

    def test_response_format_text_accepted(self, session_and_client: Any) -> None:
        session, client = session_and_client
        session.generate.return_value = _single_run(
            text="hi", tokens=[1], token_count=1, tok_per_sec=1.0, elapsed=0.1,
        )
        resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {"type": "text"},
        })
        assert resp.status_code == 200

    def test_response_format_json_object_rejected(self, session_and_client: Any) -> None:
        session, client = session_and_client
        resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {"type": "json_object"},
        })
        assert resp.status_code == 400

    def test_response_format_json_schema_rejected(self, session_and_client: Any) -> None:
        session, client = session_and_client
        resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "x"}},
        })
        assert resp.status_code == 400


class TestNativeSteeringField:
    def test_top_level_steering_expression(self, session_and_client: Any) -> None:
        session, client = session_and_client
        session.generate.return_value = _single_run(
            text="ok", tokens=[1], token_count=1, tok_per_sec=1.0, elapsed=0.1,
        )
        # ``myvec.baseline`` is a synthetic name outside the bundled-probe
        # vocabulary, so it parses to a plain vector term (a bundled name
        # like ``angry.calm`` now routes to a 2-node pca ManifoldTerm when
        # the bundled manifolds are materialized in the active home).
        resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hi"}],
            "steering": "0.5 myvec.baseline",
        })
        assert resp.status_code == 200
        kw = session.generate.call_args[1]
        assert kw["steering"] is not None
        assert kw["steering"].alphas == {"myvec.baseline": 0.5}

    def test_steering_projection_term(self, session_and_client: Any) -> None:
        session, client = session_and_client
        session.generate.return_value = _single_run(
            text="ok", tokens=[1], token_count=1, tok_per_sec=1.0, elapsed=0.1,
        )
        resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hi"}],
            "steering": "-0.4 zzfakevec",
        })
        assert resp.status_code == 200
        kw = session.generate.call_args[1]
        # A bare name matching no installed manifold/concept resolves to a plain
        # vector term (identity, sign +1); the coefficient carries through as
        # -0.4.  (Must NOT collide with a bundled node — ``wolf`` is a real
        # ``personas`` node now, so it would resolve to ``personas%wolf``.)
        assert kw["steering"].alphas == {"zzfakevec": -0.4}

    def test_steering_merges_with_server_defaults(self):
        from saklas.server import create_app
        from saklas.core.steering import Steering
        session = _mock_session()
        session.generate.return_value = _single_run(
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
        session.generate.return_value = _single_run(
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

    def test_empty_steering_clears_server_default(self):
        from saklas.server import create_app
        from saklas.core.steering import Steering
        session = _mock_session()
        session.generate.return_value = _single_run(
            text="ok", tokens=[1], token_count=1, tok_per_sec=1.0, elapsed=0.1,
        )
        app = create_app(session, default_steering=Steering(alphas={"base": 0.2}))
        c = TestClient(app)
        resp = c.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hi"}],
            "steering": "",
        })
        assert resp.status_code == 200
        kw = session.generate.call_args[1]
        # An explicit clear arrives as an *empty* Steering, not None —
        # None means "unset" engine-side, and the cast roster would fill
        # unset steering with the gen label's standing recipe.
        assert kw["steering"] is not None
        assert kw["steering"].alphas == {}

    def test_thinking_field_default_is_none_auto(self, session_and_client: Any) -> None:
        session, client = session_and_client
        session.generate.return_value = _single_run(
            text="ok", tokens=[1], token_count=1, tok_per_sec=1.0, elapsed=0.1,
        )
        resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert resp.status_code == 200
        kw = session.generate.call_args[1]
        assert kw["thinking"] is None

    def test_thinking_explicit_false(self, session_and_client: Any) -> None:
        session, client = session_and_client
        session.generate.return_value = _single_run(
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


# ---------------------------------------------------------------------------
# Session-info state reads for the instrument families
# ---------------------------------------------------------------------------


class TestSessionInfoInstrumentFields:
    """The routes that toggle these moved to /instruments; the *state reads*
    stay on session info (session_routes is unchanged)."""

    def test_jlens_fitted(self, session_and_client: Any) -> None:
        session, client = session_and_client
        resp = client.get("/saklas/v1/sessions/default")
        assert resp.status_code == 200
        assert resp.json()["jlens_fitted"] is False
        session.has_compatible_jlens.return_value = True
        got = client.get("/saklas/v1/sessions/default").json()
        assert got["jlens_fitted"] is True

    def test_live_lens_layers(self, session_and_client: Any) -> None:
        session, client = session_and_client
        resp = client.get("/saklas/v1/sessions/default")
        assert resp.status_code == 200
        assert resp.json()["live_lens_layers"] is None
        session.live_lens_layers = [10, 14, 18]
        got = client.get("/saklas/v1/sessions/default").json()
        assert got["live_lens_layers"] == [10, 14, 18]

    def test_live_probe_scores(self, session_and_client: Any) -> None:
        session, client = session_and_client
        resp = client.get("/saklas/v1/sessions/default")
        assert resp.status_code == 200
        assert resp.json()["live_probe_scores"] is True
        session.live_probe_scores = False
        got = client.get("/saklas/v1/sessions/default").json()
        assert got["live_probe_scores"] is False

    def test_sae_runtime(self, session_and_client: Any) -> None:
        session, client = session_and_client
        session.sae_info = {"release": "scope", "layer": 14, "width": 16_384}
        session.live_sae = True
        resp = client.get("/saklas/v1/sessions/default")
        assert resp.status_code == 200
        assert resp.json()["sae_loaded"] is True
        assert resp.json()["sae_info"]["layer"] == 14
        assert resp.json()["live_sae"] is True


class TestWSTokenEventLens:
    """``build_token_event`` forwards the token tap's 5.x ``measurements``
    envelope verbatim onto the frame and carries no legacy per-token aliases."""

    @staticmethod
    def _event(payload: Any) -> dict[str, Any]:
        from types import SimpleNamespace

        from saklas.server.ws_events import build_token_event

        session = SimpleNamespace(
            token_probe_payload=payload or {},
            tree=SimpleNamespace(active_node_id="node-1"),
            generation_state=SimpleNamespace(emit_map=[]),
        )
        return build_token_event(
            cast(Any, session),
            [None],
            text=" x",
            is_thinking=False,
            tid=5,
            lp=-0.1,
            top_alts=None,
        )

    def test_measurements_ride_token_frame_verbatim(self) -> None:
        """The envelope forwards by identity — the token tap is the single
        authority; event shaping never rebuilds or reshapes channels."""
        measurements = {
            "version": 1,
            "scope": "token",
            "provenance": "captured",
            "scores": {"calm": 0.424243},
            "per_layer_scores": {"5": {"calm": 0.38}},
            "instruments": {
                "geometry": {"readings": {"calm": {"coords": [0.424243]}}},
                "lens": {
                    "readout": {
                        "layers": [{
                            "layer": 12,
                            "tokens": [
                                {"token": " a", "id": 7, "logprob": -0.25},
                            ],
                        }],
                        "aggregate": [
                            {"token": " a", "strength": 0.41,
                             "com": 0.31, "spread": 0.05},
                        ],
                    },
                    "binding": {"source": "local:default", "steering": None},
                },
                "sae": {
                    "readout": {
                        "features": [
                            {"id": 362, "activation": 3216.0,
                             "label": "code blocks", "max_act": 4000.0},
                        ],
                    },
                    "binding": {
                        "source": "saelens:x", "steering": None, "layer": 12,
                    },
                },
            },
        }
        event = self._event({"measurements": measurements})
        assert event["measurements"] is measurements
        # The six legacy per-token aliases are gone from the frame.
        for dead in (
            "scores", "per_layer_scores", "probe_readings",
            "lens_readout", "lens_aggregate", "sae_readout", "captured",
        ):
            assert dead not in event

    def test_perplexity_rides_token_frame(self) -> None:
        from types import SimpleNamespace

        from saklas.server.ws_events import build_token_event

        session = SimpleNamespace(
            token_probe_payload={},
            tree=SimpleNamespace(active_node_id="node-1"),
            generation_state=SimpleNamespace(emit_map=[]),
        )
        event = build_token_event(
            cast(Any, session),
            [None],
            text=" x",
            is_thinking=False,
            tid=5,
            lp=-0.1,
            top_alts=None,
            perplexity=2.75,
        )

        assert event["perplexity"] == 2.75

    def test_measurements_not_reconstructed_from_tree_rows(self) -> None:
        from types import SimpleNamespace

        from saklas.server.ws_events import build_token_event

        class ExplodingTree:
            @property
            def active_node_id(self) -> str:
                raise AssertionError("payload path should not read active node")

            @property
            def nodes(self) -> Any:
                raise AssertionError("payload path should not read tree rows")

        session = SimpleNamespace(
            token_probe_payload={},
            tree=ExplodingTree(),
            generation_state=SimpleNamespace(emit_map=[]),
        )

        event = build_token_event(
            cast(Any, session),
            ["node-1"],
            text=" x",
            is_thinking=False,
            tid=5,
            lp=None,
            top_alts=None,
        )

        assert "measurements" not in event

    def test_absent_when_no_measurements(self) -> None:
        event = self._event({"measurements": None})
        assert "measurements" not in event

    def test_absent_when_no_payload(self) -> None:
        event = self._event(None)
        assert "measurements" not in event


class TestLensProbeRoutes:
    """Lens (readout-channel) probes on the unified ``/probes`` routes."""

    _SPEC = {"word": "fake", "token_id": 42, "layers": [12, 14, 18]}

    def test_list_includes_lens_probes(self, session_and_client: Any) -> None:
        session, client = session_and_client
        session.monitor.attached_probes.return_value = {}
        session.lens_probe_specs = {"jlens/fake": dict(self._SPEC)}
        resp = client.get("/saklas/v1/sessions/default/probes")
        assert resp.status_code == 200
        (row,) = resp.json()["probes"]
        assert row["name"] == "jlens/fake"
        assert row["lens"] is True
        assert row["word"] == "fake"
        assert row["token_id"] == 42
        assert row["layers"] == [12, 14, 18]
        assert row["feature_space"] == "readout"
        assert row["intrinsic_dim"] == 1  # the one strength axis
        assert row["node_coords"] is None

    def test_attach_returns_lens_info(self, session_and_client: Any) -> None:
        session, client = session_and_client

        def _attach(selector: str, **_kw: Any) -> str:
            session.lens_probe_specs = {
                selector: dict(TestLensProbeRoutes._SPEC),
            }
            return selector

        session.add_probe.side_effect = _attach
        resp = client.post(
            "/saklas/v1/sessions/default/probes",
            json={"selector": "jlens/fake"},
        )
        assert resp.status_code == 201
        assert resp.json()["lens"] is True
        assert resp.json()["name"] == "jlens/fake"

    def test_attach_lens_not_fitted_404(self, session_and_client: Any) -> None:
        from saklas.core.jlens import LensNotFittedError

        session, client = session_and_client
        session.add_probe.side_effect = LensNotFittedError(
            "no lens fitted — run `saklas lens fit test/model`"
        )
        resp = client.post(
            "/saklas/v1/sessions/default/probes",
            json={"selector": "jlens/fake"},
        )
        assert resp.status_code == 404
        assert "lens fit" in resp.json()["detail"]

    def test_attach_multi_token_word_400(self, session_and_client: Any) -> None:
        from saklas.core.jlens import MultiTokenWordError

        session, client = session_and_client
        session.add_probe.side_effect = MultiTokenWordError(
            "'antidisestablishment' is not a single token"
        )
        resp = client.post(
            "/saklas/v1/sessions/default/probes",
            json={"selector": "jlens/antidisestablishment"},
        )
        assert resp.status_code == 400

    def test_detach_lens_probe(self, session_and_client: Any) -> None:
        session, client = session_and_client
        session.monitor.probe_names = []
        session.lens_probe_specs = {"jlens/fake": dict(self._SPEC)}
        resp = client.delete("/saklas/v1/sessions/default/probes/jlens%2Ffake")
        assert resp.status_code == 204
        session.remove_probe.assert_called_once_with("jlens/fake")

    def test_detach_unknown_404(self, session_and_client: Any) -> None:
        session, client = session_and_client
        session.monitor.probe_names = []
        session.lens_probe_specs = {}
        resp = client.delete("/saklas/v1/sessions/default/probes/jlens%2Ffake")
        assert resp.status_code == 404
