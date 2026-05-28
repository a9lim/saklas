"""Tests for Phase 3a manifold-probe server surface (no GPU required).

Covers the native ``/saklas/v1/manifold-probes`` route family plus the
``x-saklas-manifold-readings`` extension surfaced on OpenAI chat /
completions and Ollama chat / generate responses (streaming and
non-streaming).  All exercises mock the session — the goal is to pin
the wire shape, not re-run engine integration.
"""

from __future__ import annotations

import asyncio
import json
import threading
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from saklas.core.results import (
    GenerationResult,
    ManifoldAggregate,
    ManifoldTokenReading,
    TokenEvent,
)


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _mock_domain(kind: str = "box", dim: int = 2) -> MagicMock:
    domain = MagicMock()
    domain.intrinsic_dim = dim
    if kind == "box":
        domain.to_spec.return_value = {
            "type": "box",
            "axes": [
                {"name": f"axis_{i}", "periodic": False, "lo": 0.0, "hi": 1.0}
                for i in range(dim)
            ],
        }
    elif kind == "sphere":
        domain.to_spec.return_value = {"type": "sphere", "dim": dim}
    else:
        domain.to_spec.return_value = {"type": "custom", "embed_dim": dim}
    return domain


def _mock_manifold(
    name: str = "circumplex",
    *,
    domain_kind: str = "box",
    intrinsic_dim: int = 2,
    node_labels: list[str] | None = None,
    layers: list[int] | None = None,
) -> MagicMock:
    manifold = MagicMock()
    manifold.name = name
    manifold.domain = _mock_domain(domain_kind, intrinsic_dim)
    manifold.node_labels = node_labels or ["happy", "sad", "calm", "angry"]
    manifold.layers = {idx: MagicMock() for idx in (layers or [4, 8, 12])}
    manifold.feature_space = "raw"
    return manifold


def _mock_probe(name: str, manifold: MagicMock, top_n: int = 3) -> MagicMock:
    probe = MagicMock()
    probe.name = name
    probe.manifold = manifold
    probe.top_n = top_n
    return probe


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
    monitor.profiles = {}
    session._monitor = monitor

    manifold_monitor = MagicMock()
    manifold_monitor.probe_names = []
    manifold_monitor.attached_probes.return_value = {}
    session._manifold_monitor = manifold_monitor
    session.manifold_monitor = manifold_monitor

    session._tokenizer = MagicMock()
    session._tokenizer.decode.side_effect = lambda ids: f"<{ids[0]}>" if ids else ""
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

    session._trait_queues = []
    session._trait_lock = threading.Lock()
    return session


@pytest.fixture
def session_and_client():
    from saklas.server import create_app
    session = _mock_session()
    app = create_app(session, default_steering=None)
    return session, TestClient(app)


# ---------------------------------------------------------------------------
# /saklas/v1/manifold-probes route family
# ---------------------------------------------------------------------------


class TestManifoldProbeRoutes:
    def test_list_empty(self, session_and_client):
        _, client = session_and_client
        resp = client.get("/saklas/v1/manifold-probes")
        assert resp.status_code == 200
        assert resp.json() == {"probes": []}

    def test_post_attach_and_list_round_trip(self, session_and_client):
        session, client = session_and_client
        manifold = _mock_manifold()
        probe = _mock_probe("circumplex", manifold, top_n=3)

        def _add(selector, *, as_name=None, top_n=3):
            registered = as_name or selector
            session._manifold_monitor.probe_names = [registered]
            session._manifold_monitor.attached_probes.return_value = {
                registered: probe,
            }
            return registered

        session.add_manifold_probe.side_effect = _add

        resp = client.post(
            "/saklas/v1/manifold-probes",
            json={"selector": "default/circumplex"},
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["name"] == "default/circumplex"
        assert body["manifold"] == "circumplex"
        assert body["top_n"] == 3
        assert body["layers"] == [4, 8, 12]
        assert body["node_labels"] == ["happy", "sad", "calm", "angry"]
        assert body["node_count"] == 4
        assert body["intrinsic_dim"] == 2
        assert body["domain"]["type"] == "box"
        assert body["feature_space"] == "raw"

        # List route surfaces the attached probe.
        resp = client.get("/saklas/v1/manifold-probes")
        assert resp.status_code == 200
        rows = resp.json()["probes"]
        assert len(rows) == 1
        assert rows[0]["name"] == "default/circumplex"

    def test_post_with_name_and_top_n(self, session_and_client):
        session, client = session_and_client
        manifold = _mock_manifold("personas", domain_kind="custom", intrinsic_dim=8)
        probe = _mock_probe("my-probe", manifold, top_n=5)

        def _add(selector, *, as_name=None, top_n=3):
            registered = as_name or selector
            session._manifold_monitor.probe_names = [registered]
            session._manifold_monitor.attached_probes.return_value = {
                registered: probe,
            }
            return registered

        session.add_manifold_probe.side_effect = _add

        resp = client.post(
            "/saklas/v1/manifold-probes",
            json={
                "selector": "default/personas",
                "name": "my-probe",
                "top_n": 5,
            },
        )
        assert resp.status_code == 201
        assert resp.json()["name"] == "my-probe"
        assert resp.json()["top_n"] == 5
        # Underlying session was called with the requested overrides.
        session.add_manifold_probe.assert_called_once_with(
            "default/personas", as_name="my-probe", top_n=5,
        )

    def test_post_rejects_empty_selector(self, session_and_client):
        _, client = session_and_client
        resp = client.post("/saklas/v1/manifold-probes", json={"selector": ""})
        assert resp.status_code == 400

    def test_post_404_on_unknown_selector(self, session_and_client):
        session, client = session_and_client
        session.add_manifold_probe.side_effect = FileNotFoundError(
            "manifold 'default/missing' not installed",
        )
        resp = client.post(
            "/saklas/v1/manifold-probes",
            json={"selector": "default/missing"},
        )
        assert resp.status_code == 404

    def test_delete_round_trip(self, session_and_client):
        session, client = session_and_client
        session._manifold_monitor.probe_names = ["circumplex"]

        resp = client.delete("/saklas/v1/manifold-probes/circumplex")
        assert resp.status_code == 204
        session.remove_manifold_probe.assert_called_once_with("circumplex")

    def test_delete_404_on_unknown_name(self, session_and_client):
        session, client = session_and_client
        session._manifold_monitor.probe_names = []
        resp = client.delete("/saklas/v1/manifold-probes/missing")
        assert resp.status_code == 404
        session.remove_manifold_probe.assert_not_called()


# ---------------------------------------------------------------------------
# OpenAI extension: x-saklas-manifold-readings
# ---------------------------------------------------------------------------


def _attach_aggregate(session, *, name: str = "circumplex") -> None:
    manifold = _mock_manifold(name)
    probe = _mock_probe(name, manifold)
    session._manifold_monitor.probe_names = [name]
    session._manifold_monitor.attached_probes.return_value = {name: probe}


def _populate_last_result(session, *, name: str = "circumplex") -> None:
    aggregate = ManifoldAggregate(
        fraction_mean=0.42,
        fraction_per_layer={4: 0.4, 8: 0.5},
        nearest=[("happy", 0.13), ("calm", 0.22)],
        coords=(0.61, 0.42),
        coords_per_layer={4: (0.6, 0.4), 8: (0.62, 0.45)},
        residual_mean=0.07,
        residual_per_layer={4: 0.05, 8: 0.09},
    )
    result = GenerationResult(
        text="hi", tokens=[1], token_count=1, prompt_tokens=1,
        tok_per_sec=5.0, elapsed=0.1,
        manifold_readings={name: aggregate},
    )
    session._last_result = result
    return result


class TestOpenAIManifoldExtension:
    def test_chat_completion_carries_extension_on_choice(self, session_and_client):
        session, client = session_and_client
        _attach_aggregate(session)
        result = _populate_last_result(session)
        session.generate.return_value = result

        resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Hi"}],
        })
        assert resp.status_code == 200
        choice = resp.json()["choices"][0]
        ext = choice.get("x-saklas-manifold-readings")
        assert ext is not None
        assert "circumplex" in ext
        assert ext["circumplex"]["fraction_mean"] == pytest.approx(0.42)
        assert ext["circumplex"]["coords"] == [pytest.approx(0.61), pytest.approx(0.42)]
        assert ext["circumplex"]["nearest"] == [["happy", 0.13], ["calm", 0.22]]

    def test_chat_completion_absent_when_no_probes(self, session_and_client):
        session, client = session_and_client
        session.generate.return_value = GenerationResult(
            text="hi", tokens=[1], token_count=1,
            tok_per_sec=5.0, elapsed=0.1,
        )
        resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Hi"}],
        })
        assert resp.status_code == 200
        choice = resp.json()["choices"][0]
        assert "x-saklas-manifold-readings" not in choice

    def test_text_completion_carries_extension_on_choice(self, session_and_client):
        session, client = session_and_client
        _attach_aggregate(session)
        result = _populate_last_result(session)
        session.generate.return_value = result

        resp = client.post("/v1/completions", json={"prompt": "x"})
        assert resp.status_code == 200
        ext = resp.json()["choices"][0].get("x-saklas-manifold-readings")
        assert ext is not None
        assert ext["circumplex"]["fraction_mean"] == pytest.approx(0.42)

    def test_chat_stream_surface_per_token_and_aggregate(self, session_and_client):
        session, client = session_and_client
        _attach_aggregate(session)
        result = _populate_last_result(session)
        session._last_result = result

        token_reading = ManifoldTokenReading(
            fraction=0.51,
            nearest=[("happy", 0.18), ("calm", 0.31)],
        )

        def _mock_stream(*args, **kwargs):
            yield TokenEvent(
                text="Hi", token_id=1, index=0,
                manifold_readings={"circumplex": token_reading},
            )
            yield TokenEvent(
                text=" there", token_id=2, index=1,
                manifold_readings={"circumplex": token_reading},
            )

        session.generate_stream.return_value = _mock_stream()

        resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        })
        assert resp.status_code == 200
        lines = [l for l in resp.text.strip().split("\n") if l.startswith("data: ")]
        chunks = [
            json.loads(l.removeprefix("data: "))
            for l in lines if l != "data: [DONE]"
        ]
        token_chunks = [c for c in chunks if c["choices"][0].get("delta", {}).get("content")]
        assert token_chunks
        for c in token_chunks:
            ext = c["choices"][0].get("x-saklas-manifold-readings")
            assert ext is not None
            assert ext["circumplex"]["fraction"] == pytest.approx(0.51)
            assert ext["circumplex"]["nearest"] == [["happy", 0.18], ["calm", 0.31]]

        # Final chunk carries the aggregate.
        final = next(
            c for c in chunks
            if c["choices"][0].get("finish_reason") == "stop"
        )
        agg = final["choices"][0].get("x-saklas-manifold-readings")
        assert agg is not None
        assert agg["circumplex"]["fraction_mean"] == pytest.approx(0.42)


# ---------------------------------------------------------------------------
# Ollama extension: x-saklas-manifold-readings (top-level)
# ---------------------------------------------------------------------------


class TestOllamaManifoldExtension:
    def test_chat_non_streaming_carries_extension(self, session_and_client):
        session, client = session_and_client
        _attach_aggregate(session)
        result = _populate_last_result(session)
        session.generate.return_value = result

        resp = client.post("/api/chat", json={
            "model": "test/model",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": False,
        })
        assert resp.status_code == 200
        body = resp.json()
        ext = body.get("x-saklas-manifold-readings")
        assert ext is not None
        assert ext["circumplex"]["fraction_mean"] == pytest.approx(0.42)

    def test_chat_non_streaming_absent_when_no_probes(self, session_and_client):
        session, client = session_and_client
        session.generate.return_value = GenerationResult(
            text="hi", tokens=[1], token_count=1, prompt_tokens=1,
            tok_per_sec=5.0, elapsed=0.1,
        )
        resp = client.post("/api/chat", json={
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": False,
        })
        assert resp.status_code == 200
        assert "x-saklas-manifold-readings" not in resp.json()

    def test_generate_non_streaming_carries_extension(self, session_and_client):
        session, client = session_and_client
        _attach_aggregate(session)
        result = _populate_last_result(session)
        session.generate.return_value = result

        resp = client.post("/api/generate", json={
            "prompt": "x",
            "stream": False,
        })
        assert resp.status_code == 200
        ext = resp.json().get("x-saklas-manifold-readings")
        assert ext is not None
        assert ext["circumplex"]["fraction_mean"] == pytest.approx(0.42)

    def test_chat_streaming_surface_per_token_and_aggregate(self, session_and_client):
        session, client = session_and_client
        _attach_aggregate(session)
        result = _populate_last_result(session)
        session._last_result = result

        token_reading = ManifoldTokenReading(
            fraction=0.33,
            nearest=[("happy", 0.2)],
        )

        def _mock_stream(*args, **kwargs):
            yield TokenEvent(
                text="Hi", token_id=1, index=0,
                manifold_readings={"circumplex": token_reading},
            )

        session.generate_stream.side_effect = _mock_stream

        resp = client.post("/api/chat", json={
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        })
        assert resp.status_code == 200
        chunks = [json.loads(l) for l in resp.text.strip().split("\n") if l]
        # First chunk: per-token reading rides under the extension key.
        first = chunks[0]
        assert first["done"] is False
        ext = first.get("x-saklas-manifold-readings")
        assert ext is not None
        assert ext["circumplex"]["fraction"] == pytest.approx(0.33)
        # Final chunk: aggregate.
        final = chunks[-1]
        assert final["done"] is True
        agg = final.get("x-saklas-manifold-readings")
        assert agg is not None
        assert agg["circumplex"]["fraction_mean"] == pytest.approx(0.42)
