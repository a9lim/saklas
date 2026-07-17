"""Tests for POST /saklas/v1/sessions/{id}/profiles/bake."""

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

    session.profiles = {}
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
    def test_happy_path(self, session_and_client: tuple[SaklasSession, TestClient], tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        session, client = session_and_client
        # The merge now lands a corpus-less ``fit_mode="baked"`` manifold; the
        # route loads + folds it back to a steering Profile.  Build a real one
        # on disk so the route's ``load_manifold`` + fold has a valid target
        # via the mocked ``merge_into_manifold`` output.
        monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
        from saklas.core.capture import (
            fold_directions_to_subspace, folded_directions,
        )
        from saklas.io.manifold_tensors import load_manifold
        from saklas.core.model import loaded_model_fingerprint
        from saklas.io.manifolds import create_baked_manifold_folder
        from saklas.io.paths import tensor_filename

        dirs = {0: torch.tensor([1.0, 0.0, 0.0, 0.0]), 5: torch.tensor([0.0, 2.0, 0.0, 0.0])}
        from tests._whitener import isotropic_whitener
        means = {layer: torch.zeros_like(direction) for layer, direction in dirs.items()}
        manifold = fold_directions_to_subspace(
            "noble", dirs, means,
            whitener=isotropic_whitener(dirs, 4), label="merged",
        )
        from saklas.io.packs import hash_file

        honest_source = tmp_path / "honest-source.safetensors"
        warm_source = tmp_path / "warm-source.safetensors"
        honest_source.write_bytes(b"manifest-proven honest source")
        warm_source.write_bytes(b"manifest-proven warm source")
        merged_folder, _ = create_baked_manifold_folder(
            "local", "noble", "merged", manifold, session.model_id, method="merge",
            components={
                "0": {
                    "selector": "default/honest",
                    "alpha": 0.3,
                    "tensor_sha256": hash_file(honest_source),
                },
                "1": {
                    "selector": "default/warm",
                    "alpha": 0.4,
                    "tensor_sha256": hash_file(warm_source),
                },
            },
            model_fingerprint=loaded_model_fingerprint(
                session._model, session.model_id,
            ),
        )
        tensor_path = merged_folder / tensor_filename(session.model_id)
        expected = folded_directions(load_manifold(tensor_path))

        with patch(
            "saklas.io.bake.merge_into_manifold",
            return_value=merged_folder,
        ) as m:
            resp = client.post(
                "/saklas/v1/sessions/default/profiles/bake",
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
        # Auto-register on the session with the folded merged direction.
        steer_mock = cast(MagicMock, session.steer)
        steer_mock.assert_called_once()
        call_name, call_profile = steer_mock.call_args.args
        assert call_name == "noble"
        assert call_profile.layers == [0, 5]
        assert torch.allclose(call_profile[0].float(), expected[0].float(), atol=1e-5)
        assert torch.allclose(call_profile[5].float(), expected[5].float(), atol=1e-5)

    def test_invalid_expression_400(self, session_and_client: tuple[SaklasSession, TestClient]):
        from saklas.io.bake import MergeError
        _, client = session_and_client
        with patch(
            "saklas.io.bake.merge_into_manifold",
            side_effect=MergeError("merge requires at least one component"),
        ):
            resp = client.post(
                "/saklas/v1/sessions/default/profiles/bake",
                json={"name": "x", "expression": ""},
            )
        # MergeError is a SaklasError → 400 via the global handler.
        assert resp.status_code == 400

    def test_missing_component_400(self, session_and_client: tuple[SaklasSession, TestClient]):
        from saklas.io.bake import MergeError
        _, client = session_and_client
        with patch(
            "saklas.io.bake.merge_into_manifold",
            side_effect=MergeError("component default/missing not installed"),
        ):
            resp = client.post(
                "/saklas/v1/sessions/default/profiles/bake",
                json={
                    "name": "x",
                    "expression": "0.3 default/missing",
                },
            )
        assert resp.status_code == 400

    def test_session_not_found_404(self, session_and_client: tuple[SaklasSession, TestClient]):
        _, client = session_and_client
        resp = client.post(
            "/saklas/v1/sessions/other/profiles/bake",
            json={"name": "x", "expression": "0.3 default/a"},
        )
        assert resp.status_code == 404

    def test_no_tensor_for_model_500(self, session_and_client: tuple[SaklasSession, TestClient], tmp_path: Path):
        """If merge_into_manifold returns a folder with no tensor for our model.

        Defensive 500 — merge_into_manifold should always produce one when given
        a model arg, but if it doesn't (e.g. silent skip), we surface it.
        """
        session, client = session_and_client
        empty_folder = tmp_path / "local" / "x"
        empty_folder.mkdir(parents=True)

        with patch(
            "saklas.io.bake.merge_into_manifold",
            return_value=empty_folder,
        ):
            resp = client.post(
                "/saklas/v1/sessions/default/profiles/bake",
                json={"name": "x", "expression": "0.3 default/a"},
            )
        assert resp.status_code == 500
