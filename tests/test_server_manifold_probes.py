"""Tests for Phase 3a manifold-probe server surface (no GPU required).

Covers the native ``/saklas/v1/manifold-probes`` route family plus the
``x-saklas-manifold-readings`` extension surfaced on OpenAI chat /
completions and Ollama chat / generate responses (streaming and
non-streaming), plus per-token ``manifold_readings`` on the native WS
``/saklas/v1/sessions/{id}/stream`` ``token`` frame.  All exercises
mock the session — the goal is to pin the wire shape, not re-run
engine integration.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Any, Callable
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
    def test_list_empty(self, session_and_client: Any) -> None:
        _, client = session_and_client
        resp = client.get("/saklas/v1/manifold-probes")
        assert resp.status_code == 200
        assert resp.json() == {"probes": []}

    def test_post_attach_and_list_round_trip(self, session_and_client: Any) -> None:
        session, client = session_and_client
        manifold = _mock_manifold()
        probe = _mock_probe("circumplex", manifold, top_n=3)

        def _add(selector: str, *, as_name: str | None = None, top_n: int = 3) -> str:
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

    def test_post_with_name_and_top_n(self, session_and_client: Any) -> None:
        session, client = session_and_client
        manifold = _mock_manifold("personas", domain_kind="custom", intrinsic_dim=8)
        probe = _mock_probe("my-probe", manifold, top_n=5)

        def _add(selector: str, *, as_name: str | None = None, top_n: int = 3) -> str:
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

    def test_post_rejects_empty_selector(self, session_and_client: Any) -> None:
        _, client = session_and_client
        resp = client.post("/saklas/v1/manifold-probes", json={"selector": ""})
        assert resp.status_code == 400

    def test_post_404_on_unknown_selector(self, session_and_client: Any) -> None:
        session, client = session_and_client
        session.add_manifold_probe.side_effect = FileNotFoundError(
            "manifold 'default/missing' not installed",
        )
        resp = client.post(
            "/saklas/v1/manifold-probes",
            json={"selector": "default/missing"},
        )
        assert resp.status_code == 404

    def test_delete_round_trip(self, session_and_client: Any) -> None:
        session, client = session_and_client
        session._manifold_monitor.probe_names = ["circumplex"]

        resp = client.delete("/saklas/v1/manifold-probes/circumplex")
        assert resp.status_code == 204
        session.remove_manifold_probe.assert_called_once_with("circumplex")

    def test_delete_404_on_unknown_name(self, session_and_client: Any) -> None:
        session, client = session_and_client
        session._manifold_monitor.probe_names = []
        resp = client.delete("/saklas/v1/manifold-probes/missing")
        assert resp.status_code == 404
        session.remove_manifold_probe.assert_not_called()

    def test_search_route_returns_results(self, session_and_client: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        """``GET /saklas/v1/manifolds/search`` proxies to
        ``hf_manifolds.search_manifolds`` and returns the row shape the
        webui's HF picker expects.  Mocks the HF call so the test runs
        offline.
        """
        _, client = session_and_client
        rows = [
            {
                "name": "personas",
                "namespace": "a9lim",
                "description": "100 persona archetypes",
                "tags": ["saklas-manifold"],
                "node_count": 100,
                "domain_label": "discover-pca",
                "fit_mode": "pca",
                "tensor_models": ["google__gemma-3-4b-it"],
            },
        ]
        monkeypatch.setattr(
            "saklas.server.manifold_routes.search_manifolds",
            lambda _q: rows,
        )
        resp = client.get("/saklas/v1/manifolds/search?q=persona")
        assert resp.status_code == 200
        body = resp.json()
        assert body["results"] == rows

    def test_search_route_503_when_hf_missing(
        self, session_and_client: Any, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``huggingface_hub`` missing surfaces as 503 (not 500) so the
        webui can render a friendly install hint."""
        _, client = session_and_client

        def _raise(_q: str) -> None:
            raise ImportError("huggingface_hub")

        monkeypatch.setattr(
            "saklas.server.manifold_routes.search_manifolds", _raise,
        )
        resp = client.get("/saklas/v1/manifolds/search?q=anything")
        assert resp.status_code == 503

    def test_merge_route_round_trip(self, session_and_client: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        """``POST /saklas/v1/manifolds/merge`` orchestrates the discover-
        merge under the session lock and returns the same manifold-detail
        JSON ``GET /saklas/v1/manifolds/{ns}/{name}`` ships."""
        from pathlib import Path
        from unittest.mock import MagicMock

        session, client = session_and_client

        captured: dict[str, Any] = {}

        def _merge(*args: Any, **kwargs: Any) -> Any:
            captured["args"] = args
            captured["kwargs"] = kwargs
            merged_dir = MagicMock(spec=Path)
            merged_dir.parent.name = "local"
            merged_dir.name = "combined"
            return merged_dir

        monkeypatch.setattr(
            "saklas.server.manifold_routes.merge_discover_manifolds",
            _merge,
        )
        fake_folder = MagicMock()
        fake_folder.name = "combined"
        monkeypatch.setattr(
            "saklas.server.manifold_routes._find_manifold",
            lambda ns, name: (ns, fake_folder),
        )
        monkeypatch.setattr(
            "saklas.server.manifold_routes._manifold_json",
            lambda ns, mf, sess, *, full=False: {
                "namespace": ns, "name": mf.name, "fit_mode": "pca",
            },
        )

        resp = client.post(
            "/saklas/v1/manifolds/merge",
            json={
                "name": "combined",
                "description": "fold heap",
                "sources": [
                    {"namespace": "local", "name": "a"},
                    {"namespace": "local", "name": "b"},
                ],
                "fit_mode": "pca",
            },
        )
        assert resp.status_code == 201
        assert resp.json() == {
            "namespace": "local",
            "name": "combined",
            "fit_mode": "pca",
        }
        # Inner call shape: target identity, source tuples, fit_mode
        # ride through unchanged.
        assert captured["args"] == ("local", "combined", "fold heap")
        assert captured["kwargs"]["sources"] == [
            ("local", "a"), ("local", "b"),
        ]
        assert captured["kwargs"]["fit_mode"] == "pca"

    def test_merge_route_rejects_one_source(self, session_and_client: Any) -> None:
        """Single-source merge fails fast at the route layer."""
        _, client = session_and_client
        resp = client.post(
            "/saklas/v1/manifolds/merge",
            json={
                "name": "combined",
                "sources": [{"namespace": "local", "name": "only"}],
            },
        )
        assert resp.status_code == 400

    def test_install_route_round_trip(self, session_and_client: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        """``POST /saklas/v1/manifolds/install`` orchestrates the HF
        pull under the session lock and returns the same detail JSON
        ``GET /saklas/v1/manifolds/{ns}/{name}`` ships.
        """
        from pathlib import Path
        from unittest.mock import MagicMock

        session, client = session_and_client
        installed_dir = MagicMock(spec=Path)
        installed_dir.parent.name = "local"
        installed_dir.name = "personas"
        installed_dir.parent.parent = Path("/tmp/.saklas/manifolds")

        monkeypatch.setattr(
            "saklas.server.manifold_routes.install_manifold",
            lambda *_args, **_kwargs: installed_dir,
        )
        # _find_manifold + _manifold_json read the loaded folder back
        # via ManifoldFolder.load — short-circuit both so we don't need
        # an on-disk fixture.
        fake_folder = MagicMock()
        fake_folder.name = "personas"
        fake_folder.description = "100 personas"
        fake_folder.domain = {"type": "custom", "embed_dim": 8}
        fake_folder.node_labels = ["hacker", "caveman"]
        fake_folder.node_coords = []
        fake_folder.node_roles = [None, None]
        fake_folder.fit_mode = "pca"
        fake_folder.hyperparams = {}
        fake_folder.tensor_models.return_value = []
        fake_folder.nodes_sha256.return_value = "x"
        fake_folder.is_discover = True
        fake_folder.node_groups.return_value = {}
        # Make `_find_manifold` return our fake folder.  It does
        # ManifoldFolder.load(folder) where folder is via
        # manifold_dir(ns, name); the simpler route is patching the
        # _find_manifold helper itself.
        monkeypatch.setattr(
            "saklas.server.manifold_routes._find_manifold",
            lambda ns, name: (ns, fake_folder),
        )
        monkeypatch.setattr(
            "saklas.server.manifold_routes._manifold_json",
            lambda ns, mf, sess, *, full=False: {
                "namespace": ns, "name": mf.name, "fitted": [],
            },
        )

        resp = client.post(
            "/saklas/v1/manifolds/install",
            json={"target": "a9lim/personas"},
        )
        assert resp.status_code == 201
        assert resp.json() == {
            "namespace": "local",
            "name": "personas",
            "fitted": [],
        }

    def test_delete_qualified_namespaced_name(self, session_and_client: Any) -> None:
        """Probes attached by qualified selector (``default/personas``)
        carry a registered name containing ``/`` — the DELETE route
        must accept that via ``{name:path}`` or the webui ✕ button 404s.
        """
        session, client = session_and_client
        session._manifold_monitor.probe_names = ["default/personas"]

        # Both raw-slash and percent-encoded forms must reach the
        # handler with the slash intact.  encodeURIComponent in the
        # webui sends the percent-encoded form.
        resp = client.delete("/saklas/v1/manifold-probes/default/personas")
        assert resp.status_code == 204
        session.remove_manifold_probe.assert_called_once_with("default/personas")

        session.remove_manifold_probe.reset_mock()
        session._manifold_monitor.probe_names = ["default/personas"]
        resp = client.delete("/saklas/v1/manifold-probes/default%2Fpersonas")
        assert resp.status_code == 204
        session.remove_manifold_probe.assert_called_once_with("default/personas")


# ---------------------------------------------------------------------------
# OpenAI extension: x-saklas-manifold-readings
# ---------------------------------------------------------------------------


def _attach_aggregate(session: Any, *, name: str = "circumplex") -> None:
    manifold = _mock_manifold(name)
    probe = _mock_probe(name, manifold)
    session._manifold_monitor.probe_names = [name]
    session._manifold_monitor.attached_probes.return_value = {name: probe}


def _populate_last_result(session: Any, *, name: str = "circumplex") -> GenerationResult:
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
    def test_chat_completion_carries_extension_on_choice(self, session_and_client: Any) -> None:
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

    def test_chat_completion_absent_when_no_probes(self, session_and_client: Any) -> None:
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

    def test_text_completion_carries_extension_on_choice(self, session_and_client: Any) -> None:
        session, client = session_and_client
        _attach_aggregate(session)
        result = _populate_last_result(session)
        session.generate.return_value = result

        resp = client.post("/v1/completions", json={"prompt": "x"})
        assert resp.status_code == 200
        ext = resp.json()["choices"][0].get("x-saklas-manifold-readings")
        assert ext is not None
        assert ext["circumplex"]["fraction_mean"] == pytest.approx(0.42)

    def test_chat_stream_surface_per_token_and_aggregate(self, session_and_client: Any) -> None:
        session, client = session_and_client
        _attach_aggregate(session)
        result = _populate_last_result(session)
        session._last_result = result

        token_reading = ManifoldTokenReading(
            fraction=0.51,
            nearest=[("happy", 0.18), ("calm", 0.31)],
        )

        def _mock_stream(*args: Any, **kwargs: Any) -> Any:
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
    def test_chat_non_streaming_carries_extension(self, session_and_client: Any) -> None:
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

    def test_chat_non_streaming_absent_when_no_probes(self, session_and_client: Any) -> None:
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

    def test_generate_non_streaming_carries_extension(self, session_and_client: Any) -> None:
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

    def test_chat_streaming_surface_per_token_and_aggregate(self, session_and_client: Any) -> None:
        session, client = session_and_client
        _attach_aggregate(session)
        result = _populate_last_result(session)
        session._last_result = result

        token_reading = ManifoldTokenReading(
            fraction=0.33,
            nearest=[("happy", 0.2)],
        )

        def _mock_stream(*args: Any, **kwargs: Any) -> Any:
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


# ---------------------------------------------------------------------------
# Native WebSocket: per-token manifold_readings on token frames
# ---------------------------------------------------------------------------


class TestWebSocketManifoldReadings:
    """The native ``/saklas/v1/sessions/{id}/stream`` WS must carry
    ``manifold_readings`` on every ``token`` frame when manifold probes
    are attached, and the final ``done`` event still carries the
    aggregate.  Vector probes go through the persisted-loom-row path
    (see ``test_saklas_api.TestWebSocket``); manifold readings have no
    such persistence yet, so the server scores directly off
    ``session._capture._per_layer`` inside the WS ``_on_token``
    callback.

    Regression for the Phase 3c bug where the webui mini-map cursor
    stayed stuck at ``awaiting first token...`` because the WS ``token``
    frame omitted the field entirely until ``done``.
    """

    def _attach_manifold(self, session: Any, *, name: str = "circumplex") -> Any:
        manifold = _mock_manifold(name)
        probe = _mock_probe(name, manifold)
        session._manifold_monitor.probe_names = [name]
        session._manifold_monitor.attached_probes.return_value = {name: probe}

        # Per-token reading the monitor returns each time the WS
        # callback hits the inline-score branch.
        reading = ManifoldTokenReading(
            fraction=0.51,
            nearest=[("happy", 0.18), ("calm", 0.31)],
        )

        def _score(hidden: Any) -> dict[str, Any]:
            return {name: reading}

        session._manifold_monitor.score_single_token.side_effect = _score
        return reading

    def _wire_capture(self, session: Any) -> None:
        """Wire ``session._capture._per_layer`` so the inline manifold
        scoring branch sees non-empty per-layer captures and therefore
        actually runs.  The values themselves don't matter — the mocked
        ``score_single_token`` ignores its input."""
        capture = MagicMock()
        capture._per_layer = {4: ["sentinel"], 8: ["sentinel"]}
        session._capture = capture

    def _attach_generate(self, session: Any, tokens: Any, aggregate: Any = None) -> None:
        """Install a fake ``session.generate`` that drives ``on_token``.

        Mirrors the pattern in ``test_saklas_api.TestWebSocket``.  When
        ``aggregate`` is provided it is stashed on the returned result's
        ``manifold_readings`` so the ``done`` event surfaces the
        aggregate alongside the per-token frames.
        """
        def _gen(
            input: Any, *, steering: Any = None, sampling: Any = None, stateless: bool = False,
            raw: bool = False, thinking: Any = None, on_token: Callable[..., Any] | None = None,
            parent_node_id: Any = None, n: int = 1, recipe_override: Any = None,
        ) -> GenerationResult:
            for i, tok in enumerate(tokens):
                if on_token is not None:
                    on_token(tok, False, 1000 + i, None, None)
                time.sleep(0.001)
            result = GenerationResult(
                text="".join(tokens),
                tokens=list(range(1000, 1000 + len(tokens))),
                token_count=len(tokens), tok_per_sec=50.0, elapsed=0.05,
                finish_reason="stop",
                manifold_readings=(
                    {"circumplex": aggregate} if aggregate else {}
                ),
            )
            session._last_result = result
            session.last_result = result
            session._last_per_token_scores = None
            session.last_per_token_scores = None
            return result

        session.generate.side_effect = _gen

    def test_token_frame_carries_manifold_readings(self, session_and_client: Any) -> None:
        session, client = session_and_client
        self._attach_manifold(session)
        self._wire_capture(session)
        self._attach_generate(session, ["Hello", " ", "world"])

        with client.websocket_connect(
            "/saklas/v1/sessions/default/stream",
        ) as ws:
            ws.send_json({"type": "generate", "input": "hi"})
            started = ws.receive_json()
            assert started["type"] == "started"
            token_frames = []
            done = None
            while True:
                msg = ws.receive_json()
                if msg["type"] == "token":
                    token_frames.append(msg)
                elif msg["type"] == "done":
                    done = msg
                    break

        assert len(token_frames) == 3
        for frame in token_frames:
            mf = frame.get("manifold_readings")
            assert mf is not None, (
                "per-token manifold_readings missing on WS token frame — "
                "regression of the Phase 3c stall bug"
            )
            assert "circumplex" in mf
            assert mf["circumplex"]["fraction"] == pytest.approx(0.51)
            assert mf["circumplex"]["nearest"] == [
                ["happy", 0.18], ["calm", 0.31],
            ]
        assert done is not None
        # The done event carries the aggregate only when the result
        # actually has one — this happy-path test omits the aggregate
        # to isolate the per-token wire.  The full round-trip is
        # covered by ``test_done_frame_carries_manifold_aggregate``
        # below.

    def test_done_frame_carries_manifold_aggregate(self, session_and_client: Any) -> None:
        session, client = session_and_client
        self._attach_manifold(session)
        self._wire_capture(session)
        aggregate = ManifoldAggregate(
            fraction_mean=0.42,
            fraction_per_layer={4: 0.4, 8: 0.5},
            nearest=[("happy", 0.13), ("calm", 0.22)],
            coords=(0.61, 0.42),
            coords_per_layer={4: (0.6, 0.4), 8: (0.62, 0.45)},
            residual_mean=0.07,
            residual_per_layer={4: 0.05, 8: 0.09},
        )
        self._attach_generate(session, ["Hello"], aggregate=aggregate)

        with client.websocket_connect(
            "/saklas/v1/sessions/default/stream",
        ) as ws:
            ws.send_json({"type": "generate", "input": "hi"})
            assert ws.receive_json()["type"] == "started"
            while True:
                msg = ws.receive_json()
                if msg["type"] == "done":
                    done = msg
                    break

        agg_blob = done["result"].get("manifold_readings")
        assert agg_blob is not None
        assert agg_blob["circumplex"]["fraction_mean"] == pytest.approx(0.42)
        assert agg_blob["circumplex"]["coords"] == [
            pytest.approx(0.61), pytest.approx(0.42),
        ]

    def test_token_frame_omits_field_when_no_manifold_probes(
        self, session_and_client: Any,
    ) -> None:
        """No manifold probes attached → ``manifold_readings`` omitted
        from every token frame.  Legacy clients see the unchanged shape.
        """
        session, client = session_and_client
        # Probe list stays empty; capture wiring left default so the
        # inline branch never triggers.
        session._manifold_monitor.probe_names = []
        self._attach_generate(session, ["Hi"])

        with client.websocket_connect(
            "/saklas/v1/sessions/default/stream",
        ) as ws:
            ws.send_json({"type": "generate", "input": "hi"})
            assert ws.receive_json()["type"] == "started"
            saw_token = False
            while True:
                msg = ws.receive_json()
                if msg["type"] == "token":
                    saw_token = True
                    assert "manifold_readings" not in msg
                elif msg["type"] == "done":
                    break
        assert saw_token


# ---------------------------------------------------------------------------
# Shared serializer (manifold_summary) + delete-helper reuse
# ---------------------------------------------------------------------------
#
# These exercise the *real* ``_manifold_json`` against an on-disk authored
# manifold (the route tests above monkeypatch ``_manifold_json``, so they
# don't cover the refactor).  Pin two contracts:
#  (a) the keys the GET route shares with ``io.manifolds.manifold_summary``
#      carry identical names + values — both consume the same serializer,
#      so CLI ``manifold show -j`` and the server stay in lockstep;
#  (b) the DELETE route routes its removal through the shared
#      ``remove_manifold_folder`` helper (single source of truth with the
#      CLI ``manifold rm``), and its response carries that helper's
#      richer shape.


def _author_manifold_on_disk(
    home: Any,
    *,
    namespace: str = "local",
    name: str = "mood",
    labels: list[str] | None = None,
) -> Any:
    """Hand-author a v4 authored 1-D box manifold under ``$SAKLAS_HOME``.

    Mirrors the minimal fixture ``test_manifolds_io`` uses, but writes
    into the live ``manifolds/<ns>/<name>/`` tree so the server routes
    (which resolve through ``manifold_dir``) can find it.  Returns the
    folder path.
    """
    import json as _json
    from pathlib import Path

    from saklas.io.manifolds import MANIFOLD_FORMAT_VERSION

    labels = labels or ["calm", "uneasy", "afraid", "frantic"]
    k = len(labels)
    folder = Path(home) / "manifolds" / namespace / name
    (folder / "nodes").mkdir(parents=True)
    coords = [[i / (k - 1)] for i in range(k)]
    nodes = [{"label": lbl, "coords": coords[i]} for i, lbl in enumerate(labels)]
    for idx, node in enumerate(nodes):
        statements = [f"{node['label']} statement {i}" for i in range(3)]
        (folder / "nodes" / f"{idx:02d}_{node['label']}.json").write_text(
            _json.dumps(statements)
        )
    meta = {
        "format_version": MANIFOLD_FORMAT_VERSION,
        "name": name,
        "description": "a mood manifold",
        "domain": {
            "type": "box",
            "axes": [{"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}],
        },
        "nodes": nodes,
        "files": {},
    }
    (folder / "manifold.json").write_text(_json.dumps(meta))
    return folder


class TestManifoldSharedSerializer:
    """``GET /manifolds`` shares ``manifold_summary``'s session-independent keys."""

    @pytest.fixture
    def home_and_client(self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch):
        # Point $SAKLAS_HOME at a tmp tree so manifold_dir resolves into
        # an isolated, empty manifolds root (the mock session never runs
        # bundled materialization).
        monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
        from saklas.server import create_app

        session = _mock_session()
        app = create_app(session, default_steering=None)
        return tmp_path, session, TestClient(app)

    def test_get_one_shares_summary_keys(self, home_and_client: Any) -> None:
        """Every key the GET-one route shares with ``manifold_summary``
        carries an identical value — the refactor's core guarantee."""
        from saklas.io.manifolds import manifold_summary

        home, _session, client = home_and_client
        folder = _author_manifold_on_disk(home, namespace="local", name="mood")

        summary = manifold_summary(folder)

        resp = client.get("/saklas/v1/manifolds/local/mood")
        assert resp.status_code == 200
        body = resp.json()

        # All shared (session-independent) keys must match byte-for-byte.
        for key, value in summary.items():
            assert key in body, f"GET route dropped shared key {key!r}"
            assert body[key] == value, (
                f"shared key {key!r} diverged: "
                f"summary={value!r} route={body[key]!r}"
            )

        # Session-only extras layered on top — present, not in the summary.
        assert "fitted_for_session" in body
        assert "stale" in body
        assert body["fitted_for_session"] is False  # mock model has no tensor
        assert body["stale"] is False
        # ``full`` detail blocks the list route omits.
        assert "nodes" in body
        assert [n["label"] for n in body["nodes"]] == summary["node_labels"]

    def test_list_shares_summary_keys(self, home_and_client: Any) -> None:
        """The list route builds the same shared keys from the summary."""
        from saklas.io.manifolds import manifold_summary

        home, _session, client = home_and_client
        folder = _author_manifold_on_disk(home, namespace="local", name="mood")
        summary = manifold_summary(folder)

        resp = client.get("/saklas/v1/manifolds")
        assert resp.status_code == 200
        rows = resp.json()["manifolds"]
        assert len(rows) == 1
        row = rows[0]
        for key, value in summary.items():
            assert row[key] == value, f"list row diverged on {key!r}"
        # List route is light — no per-node ``nodes`` / per-tensor ``fitted``.
        assert "nodes" not in row
        assert "fitted" not in row

    def test_delete_routes_through_shared_helper(
        self, home_and_client: Any,
    ) -> None:
        """DELETE removes the folder via ``remove_manifold_folder`` and
        returns that helper's richer ``{namespace, name, source, removed,
        rematerializes_on_restart}`` shape (superset of the historical
        ``{namespace, name, removed}``)."""
        home, _session, client = home_and_client
        folder = _author_manifold_on_disk(home, namespace="local", name="doomed")
        assert (folder / "manifold.json").exists()

        resp = client.delete("/saklas/v1/manifolds/local/doomed")
        assert resp.status_code == 200
        body = resp.json()
        assert body["namespace"] == "local"
        assert body["name"] == "doomed"
        assert body["removed"] is True
        # Shared-helper extras — a local manifold does not respawn.
        assert body["source"] == "local"
        assert body["rematerializes_on_restart"] is False
        # Folder is actually gone.
        assert not folder.exists()

    def test_delete_bundled_flags_respawn(self, home_and_client: Any) -> None:
        """A ``default/``-namespace folder flips ``rematerializes_on_restart``
        through the shared helper — same signal the pack DELETE route emits."""
        home, _session, client = home_and_client
        folder = _author_manifold_on_disk(
            home, namespace="default", name="circumplex",
        )

        resp = client.delete("/saklas/v1/manifolds/default/circumplex")
        assert resp.status_code == 200
        body = resp.json()
        assert body["removed"] is True
        assert body["rematerializes_on_restart"] is True
        assert not folder.exists()

    def test_delete_missing_404(self, home_and_client: Any) -> None:
        """DELETE on a folder that was never authored → 404 (pre-lock check)."""
        _home, _session, client = home_and_client
        resp = client.delete("/saklas/v1/manifolds/local/ghost")
        assert resp.status_code == 404
