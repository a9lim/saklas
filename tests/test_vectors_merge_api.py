"""Tests for POST /saklas/v1/sessions/{id}/vectors/merge."""

# pyright: reportUnusedVariable=false

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

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


class TestMergeVector:
    def test_happy_path(self, session_and_client: tuple[SaklasSession, TestClient], tmp_path: Path):
        session, client = session_and_client
        # Build a fake merge tensor on disk so the route's ``load_profile``
        # call lands on a real file via the mocked merge_into_pack output.
        merged_folder = tmp_path / "local" / "noble"
        merged_folder.mkdir(parents=True)
        # Fake tensor file that the route expects to find.
        from saklas.io.paths import tensor_filename
        tensor_path = merged_folder / tensor_filename(session.model_id)
        tensor_path.touch()

        merged_profile = Profile({0: torch.zeros(4), 5: torch.ones(4)})

        def _load_profile(path: str | Path) -> Profile:
            assert Path(path) == tensor_path
            return merged_profile

        cast(MagicMock, session.load_profile).side_effect = _load_profile

        with patch(
            "saklas.io.merge.merge_into_pack",
            return_value=merged_folder,
        ) as m:
            resp = client.post(
                "/saklas/v1/sessions/default/vectors/merge",
                json={
                    "name": "noble",
                    "expression": "0.3 default/honest + 0.4 default/warm",
                },
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "noble"
        assert data["layers"] == [0, 5]
        # Confirm we forwarded model + expression + force=True.
        kwargs = m.call_args.kwargs
        args = m.call_args.args
        assert args[0] == "noble"
        assert args[1] == "0.3 default/honest + 0.4 default/warm"
        assert args[2] == "test/model"
        assert kwargs["force"] is True
        # Auto-register on the session.
        steer_mock = cast(MagicMock, session.steer)
        steer_mock.assert_called_once_with("noble", merged_profile)

    def test_invalid_expression_400(self, session_and_client: tuple[SaklasSession, TestClient]):
        from saklas.io.merge import MergeError
        _, client = session_and_client
        with patch(
            "saklas.io.merge.merge_into_pack",
            side_effect=MergeError("merge requires at least one component"),
        ):
            resp = client.post(
                "/saklas/v1/sessions/default/vectors/merge",
                json={"name": "x", "expression": ""},
            )
        # MergeError is a SaklasError → 400 via the global handler.
        assert resp.status_code == 400

    def test_missing_component_400(self, session_and_client: tuple[SaklasSession, TestClient]):
        from saklas.io.merge import MergeError
        _, client = session_and_client
        with patch(
            "saklas.io.merge.merge_into_pack",
            side_effect=MergeError("component default/missing not installed"),
        ):
            resp = client.post(
                "/saklas/v1/sessions/default/vectors/merge",
                json={
                    "name": "x",
                    "expression": "0.3 default/missing",
                },
            )
        assert resp.status_code == 400

    def test_session_not_found_404(self, session_and_client: tuple[SaklasSession, TestClient]):
        _, client = session_and_client
        resp = client.post(
            "/saklas/v1/sessions/other/vectors/merge",
            json={"name": "x", "expression": "0.3 default/a"},
        )
        assert resp.status_code == 404

    def test_no_tensor_for_model_500(self, session_and_client: tuple[SaklasSession, TestClient], tmp_path: Path):
        """If merge_into_pack returns a folder with no tensor for our model.

        Defensive 500 — merge_into_pack should always produce one when given
        a model arg, but if it doesn't (e.g. silent skip), we surface it.
        """
        session, client = session_and_client
        empty_folder = tmp_path / "local" / "x"
        empty_folder.mkdir(parents=True)

        with patch(
            "saklas.io.merge.merge_into_pack",
            return_value=empty_folder,
        ):
            resp = client.post(
                "/saklas/v1/sessions/default/vectors/merge",
                json={"name": "x", "expression": "0.3 default/a"},
            )
        assert resp.status_code == 500
