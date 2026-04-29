"""Tests for GET /saklas/v1/sessions/{id}/vectors/{name}/diagnostics."""

# pyright: reportUnusedVariable=false

from __future__ import annotations

import asyncio
import threading
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from saklas.core.histogram import HIST_BUCKETS
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


def _make_profile(num_layers: int, dim: int = 8, with_diag: bool = False) -> Profile:
    """Build a Profile with monotonically increasing magnitudes per layer.

    Magnitudes go 1, 2, 3, ... so bucketize's mean values are predictable
    and test-stable.
    """
    tensors: dict[int, torch.Tensor] = {}
    for i in range(num_layers):
        # Vector with norm == (i + 1) — each entry sqrt((i+1)^2 / dim).
        scale = float(i + 1) / (dim ** 0.5)
        tensors[i] = torch.full((dim,), scale)
    metadata: dict[str, Any] = {"method": "pca_center"}
    if with_diag:
        metadata["diagnostics"] = {
            i: {
                "evr": 0.4 + 0.01 * i,
                "intra_pair_variance_mean": 0.1 + 0.005 * i,
                "intra_pair_variance_std": 0.02,
                "inter_pair_alignment": 0.6,
                "diff_principal_projection": 0.7,
            }
            for i in range(num_layers)
        }
    return Profile(tensors, metadata=metadata)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestVectorDiagnostics:
    def test_happy_path_no_diagnostics(self, session_and_client: tuple[SaklasSession, TestClient], monkeypatch: MonkeyPatch):
        """Profile with no diagnostics carries layers + histogram only."""
        session, client = session_and_client
        profile = _make_profile(40, dim=8, with_diag=False)
        monkeypatch.setattr(session, "vectors", {"honest": profile}, raising=False)

        resp = client.get(
            "/saklas/v1/sessions/default/vectors/honest/diagnostics",
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "honest"
        assert data["model"] == "test/model"
        assert data["total_layers"] == 40

        hist = data["histogram"]
        assert hist["buckets"] == HIST_BUCKETS
        # 40 layers → 16 buckets, all non-empty since 40 > 16.
        assert len(hist["data"]) == HIST_BUCKETS
        # Buckets carry ascending lo / hi indices in layer order.
        prev_hi = -1
        for bucket in hist["data"]:
            assert bucket["lo"] <= bucket["hi"]
            assert bucket["lo"] > prev_hi
            assert "mean_norm" in bucket
            prev_hi = bucket["hi"]

        assert len(data["layers"]) == 40
        assert data["layers"][0]["layer"] == 0
        # Magnitude ordering matches our construction (layer i has norm i+1).
        assert data["layers"][0]["magnitude"] == pytest.approx(1.0, rel=1e-4)
        assert data["layers"][-1]["magnitude"] == pytest.approx(40.0, rel=1e-4)
        assert "diagnostics_by_layer" not in data
        assert "diagnostics_summary" not in data

    def test_few_layers_buckets_collapse(self, session_and_client: tuple[SaklasSession, TestClient], monkeypatch: MonkeyPatch):
        """With layers < HIST_BUCKETS, each layer becomes its own bucket."""
        session, client = session_and_client
        profile = _make_profile(8, dim=4, with_diag=False)
        monkeypatch.setattr(session, "vectors", {"x": profile}, raising=False)

        resp = client.get(
            "/saklas/v1/sessions/default/vectors/x/diagnostics",
        )
        assert resp.status_code == 200
        data = resp.json()
        # 8 layers < 16 → 8 buckets, lo == hi everywhere.
        assert len(data["histogram"]["data"]) == 8
        for b in data["histogram"]["data"]:
            assert b["lo"] == b["hi"]

    def test_with_diagnostics(self, session_and_client: tuple[SaklasSession, TestClient], monkeypatch: MonkeyPatch):
        session, client = session_and_client
        profile = _make_profile(20, dim=8, with_diag=True)
        monkeypatch.setattr(session, "vectors", {"warm": profile}, raising=False)

        resp = client.get(
            "/saklas/v1/sessions/default/vectors/warm/diagnostics",
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "diagnostics_by_layer" in data
        assert "diagnostics_summary" in data
        # Per-layer keys are stringified ints (JSON friendly), every layer present.
        assert set(data["diagnostics_by_layer"].keys()) == {str(i) for i in range(20)}
        # Summary has medians + a coarse stoplight.
        s = data["diagnostics_summary"]
        for k in (
            "median_evr",
            "median_intra_pair_variance",
            "median_inter_pair_alignment",
            "median_diff_principal_projection",
            "quality",
        ):
            assert k in s
        assert s["quality"] in {"poor", "shaky", "solid"}

    def test_unknown_vector_404(self, session_and_client: tuple[SaklasSession, TestClient], monkeypatch: MonkeyPatch):
        session, client = session_and_client
        monkeypatch.setattr(session, "vectors", {}, raising=False)
        resp = client.get(
            "/saklas/v1/sessions/default/vectors/missing/diagnostics",
        )
        assert resp.status_code == 404

    def test_session_not_found_404(self, session_and_client: tuple[SaklasSession, TestClient]):
        _, client = session_and_client
        resp = client.get(
            "/saklas/v1/sessions/other/vectors/x/diagnostics",
        )
        assert resp.status_code == 404
