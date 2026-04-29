"""Tests for POST /saklas/v1/sessions/{id}/vectors/clone."""

# pyright: reportUnusedVariable=false

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import pytest
import torch
from fastapi.testclient import TestClient

from saklas.core.profile import Profile
from saklas.core.session import SaklasSession


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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
    session.lock = asyncio.Lock()

    session._trait_queues = []
    session._trait_lock = threading.Lock()
    session.register_trait_queue = lambda *_a, **_kw: None
    session.unregister_trait_queue = lambda *_a, **_kw: None
    session.events = MagicMock()
    session.events.subscribe = lambda cb: (lambda: None)
    return session


@pytest.fixture
def session_and_client():
    from saklas.server import create_app
    session = _mock_session()
    app = create_app(session, default_steering=None)
    return session, TestClient(app)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCloneVector:
    def test_happy_path_json(self, session_and_client: tuple[SaklasSession, TestClient], tmp_path: Path):
        session, client = session_and_client
        corpus = tmp_path / "corpus.txt"
        corpus.write_text("line one\nline two\n")

        cloned_profile = Profile({0: torch.zeros(4), 3: torch.ones(4)})

        def _clone(
            path: str | Path,
            name: str,
            *,
            n_pairs: int = 90,
            seed: int | None = None,
        ) -> tuple[str, Profile]:
            assert str(path) == str(corpus)
            assert name == "tone"
            assert n_pairs == 16
            assert seed == 42
            return ("tone", cloned_profile)

        cast(MagicMock, session.clone_from_corpus).side_effect = _clone

        resp = client.post(
            "/saklas/v1/sessions/default/vectors/clone",
            json={
                "name": "tone",
                "corpus_path": str(corpus),
                "n_pairs": 16,
                "seed": 42,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["canonical"] == "tone"
        assert data["profile"]["layers"] == [0, 3]
        # Auto-register on the session.
        cast(MagicMock, session.steer).assert_called_once_with("tone", cloned_profile)

    def test_missing_corpus_404_json(self, session_and_client: tuple[SaklasSession, TestClient]):
        session, client = session_and_client
        cast(MagicMock, session.clone_from_corpus).side_effect = FileNotFoundError(
            "corpus file not found: /no/such/path"
        )
        resp = client.post(
            "/saklas/v1/sessions/default/vectors/clone",
            json={"name": "x", "corpus_path": "/no/such/path"},
        )
        assert resp.status_code == 404
        # FastAPI's default HTTPException shape: {"detail": "..."}.
        assert "no/such/path" in resp.json()["detail"]

    def test_session_not_found_404(self, session_and_client: tuple[SaklasSession, TestClient]):
        _, client = session_and_client
        resp = client.post(
            "/saklas/v1/sessions/other/vectors/clone",
            json={"name": "x", "corpus_path": "/tmp/x"},
        )
        assert resp.status_code == 404

    def test_sse_progress_branch_done_event(self, session_and_client: tuple[SaklasSession, TestClient], tmp_path: Path):
        session, client = session_and_client
        corpus = tmp_path / "corpus.txt"
        corpus.write_text("a\nb\n")
        cloned_profile = Profile({0: torch.zeros(4)})
        cast(MagicMock, session.clone_from_corpus).return_value = ("tone", cloned_profile)

        # ``stream=True`` keeps the response open so we can read SSE frames.
        with client.stream(
            "POST",
            "/saklas/v1/sessions/default/vectors/clone",
            json={"name": "tone", "corpus_path": str(corpus)},
            headers={"Accept": "text/event-stream"},
        ) as resp:
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/event-stream")
            body = b"".join(resp.iter_bytes())

        text = body.decode("utf-8")
        # SSE stream: at minimum a ``done`` event with the registered profile.
        assert "event: done" in text
        # Pull the JSON payload out of the ``data:`` line.
        data_line = next(
            line for line in text.splitlines() if line.startswith("data: ")
        )
        payload = json.loads(data_line[len("data: "):])
        assert payload["done"] is True
        assert payload["canonical"] == "tone"
        assert payload["profile"]["layers"] == [0]
        # Auto-register fires on the SSE branch too.
        cast(MagicMock, session.steer).assert_called_once_with("tone", cloned_profile)

    def test_sse_error_event_on_missing_corpus(self, session_and_client: tuple[SaklasSession, TestClient]):
        session, client = session_and_client
        cast(MagicMock, session.clone_from_corpus).side_effect = FileNotFoundError(
            "corpus file not found"
        )
        with client.stream(
            "POST",
            "/saklas/v1/sessions/default/vectors/clone",
            json={"name": "x", "corpus_path": "/no/such"},
            headers={"Accept": "text/event-stream"},
        ) as resp:
            assert resp.status_code == 200  # SSE always 200; payload carries error
            body = b"".join(resp.iter_bytes()).decode("utf-8")
        assert "event: error" in body
        assert "FileNotFoundError" in body
        # No registration on error.
        cast(MagicMock, session.steer).assert_not_called()
