"""Tests for SteerSession programmatic API.
Requires CUDA and downloads google/gemma-2-2b-it (~5GB) on first run.
"""
from __future__ import annotations
import pytest
import torch
from steer.results import GenerationResult, TokenEvent

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)

MODEL_ID = "google/gemma-2-2b-it"

@pytest.fixture(scope="module")
def session():
    from steer.session import SteerSession
    s = SteerSession(MODEL_ID, device="cuda", probes=["emotion"])
    yield s
    s.close()

class TestConstruction:
    def test_model_info(self, session):
        info = session.model_info
        assert info["model_type"] == "gemma2"
        assert info["hidden_dim"] > 0
        assert info["num_layers"] > 0

    def test_config_defaults(self, session):
        assert session.config.temperature == 1.0
        assert session.config.top_p == 0.9
        assert session.config.max_new_tokens == 1024

    def test_probes_loaded(self, session):
        assert len(session.probes) > 0

    def test_history_starts_empty(self, session):
        assert session.history == []

    def test_vectors_starts_empty(self, session):
        assert session.vectors == {}

    def test_last_result_starts_none(self, session):
        assert session.last_result is None

class TestSteering:
    def test_extract_and_steer(self, session):
        profile = session.extract([("I am happy", "I am sad")])
        assert isinstance(profile, dict)
        assert all(isinstance(k, int) for k in profile)
        session.steer("happy", profile, alpha=1.5)
        assert "happy" in session.vectors
        assert session.vectors["happy"]["alpha"] == 1.5
        assert session.vectors["happy"]["enabled"] is True

    def test_set_alpha(self, session):
        session.set_alpha("happy", 2.0)
        assert session.vectors["happy"]["alpha"] == 2.0

    def test_toggle(self, session):
        session.toggle("happy")
        assert session.vectors["happy"]["enabled"] is False
        session.toggle("happy")
        assert session.vectors["happy"]["enabled"] is True

    def test_unsteer(self, session):
        session.unsteer("happy")
        assert "happy" not in session.vectors

    def test_extract_curated(self, session):
        profile = session.extract("calm")
        assert isinstance(profile, dict)
        assert len(profile) > 0

    def test_extract_datasource(self, session):
        from steer.datasource import DataSource
        ds = DataSource.from_pairs([("formal", "casual")])
        profile = session.extract(ds)
        assert isinstance(profile, dict)

class TestMonitoring:
    def test_monitor_and_unmonitor(self, session):
        profile = session.extract([("I am honest", "I am deceptive")])
        session.monitor("test_probe", profile)
        assert "test_probe" in session.probes
        session.unmonitor("test_probe")
        assert "test_probe" not in session.probes

class TestLifecycle:
    def test_context_manager(self):
        from steer.session import SteerSession
        with SteerSession(MODEL_ID, device="cuda", probes=[]) as s:
            assert s.model_info["model_type"] == "gemma2"

class TestGeneration:
    def test_generate_blocking_string(self, session):
        result = session.generate("Say hello in one word.")
        assert isinstance(result, GenerationResult)
        assert len(result.text) > 0
        assert result.token_count > 0
        assert result.tok_per_sec > 0
        assert result.elapsed > 0

    def test_generate_blocking_messages(self, session):
        result = session.generate([
            {"role": "user", "content": "Say hello in one word."},
        ])
        assert isinstance(result, GenerationResult)
        assert len(result.text) > 0

    def test_generate_appends_to_history(self, session):
        session.clear_history()
        session.generate("Say hi.")
        assert len(session.history) == 2
        assert session.history[0]["role"] == "user"
        assert session.history[1]["role"] == "assistant"

    def test_generate_with_probes(self, session):
        session.clear_history()
        result = session.generate("Tell me something exciting!")
        if session.probes:
            assert isinstance(result.readings, dict)

    def test_generate_snapshots_vectors(self, session):
        profile = session.extract([("formal", "casual")])
        session.steer("formal", profile, alpha=1.0)
        result = session.generate("Hello.")
        assert "formal" in result.vectors
        assert result.vectors["formal"] == 1.0
        session.unsteer("formal")

    def test_last_result(self, session):
        session.clear_history()
        result = session.generate("Hello.")
        assert session.last_result is result

class TestStreamingGeneration:
    def test_generate_stream(self, session):
        session.clear_history()
        tokens = []
        for event in session.generate_stream("Say hello."):
            assert isinstance(event, TokenEvent)
            tokens.append(event)
        assert len(tokens) > 0
        assert all(isinstance(t.text, str) for t in tokens)
        assert session.last_result is not None
        assert session.last_result.token_count == len(tokens)
