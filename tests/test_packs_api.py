"""Tests for the native /saklas/v1/packs/* routes (no GPU)."""

# pyright: reportUnusedVariable=false

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from saklas.core.session import SaklasSession
from saklas.io.cache_ops import ConceptRow, HfRow, PackListResult


# ---------------------------------------------------------------------------
# Fixtures (mirrors test_saklas_api.py's mock surface)
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
    session.events.subscribe = lambda _cb: (lambda: None)
    return session


@pytest.fixture
def session_and_client():
    from saklas.server import create_app
    session = _mock_session()
    app = create_app(session, default_steering=None)
    return session, TestClient(app)


# ---------------------------------------------------------------------------
# GET /saklas/v1/packs
# ---------------------------------------------------------------------------


class TestListPacks:
    def test_empty_local(self, session_and_client: tuple[SaklasSession, TestClient]):
        _, client = session_and_client
        with patch(
            "saklas.io.cache_ops.list_concepts",
            return_value=PackListResult(installed=[], hf_rows=[], error=None),
        ):
            resp = client.get("/saklas/v1/packs")
        assert resp.status_code == 200
        assert resp.json() == {"packs": []}

    def test_lists_installed_only(self, session_and_client: tuple[SaklasSession, TestClient]):
        _, client = session_and_client
        rows = [
            ConceptRow(
                name="happy.sad", namespace="default", status="installed",
                recommended_alpha=0.5, tags=["affect"], description="Happy vs sad",
                source="bundled", tensor_models=["test__model"],
            ),
            ConceptRow(
                name="custom", namespace="local", status="installed",
                recommended_alpha=0.4, tags=["custom"], description="",
                source="local", tensor_models=[],
            ),
        ]
        with patch(
            "saklas.io.cache_ops.list_concepts",
            return_value=PackListResult(installed=rows, hf_rows=[], error=None),
        ) as m:
            resp = client.get("/saklas/v1/packs")
        # ``hf=False`` is the load-bearing flag — confirms no HF query fires.
        assert m.call_args.kwargs["hf"] is False
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["packs"]) == 2
        assert data["packs"][0]["name"] == "happy.sad"
        assert data["packs"][0]["namespace"] == "default"
        assert data["packs"][0]["status"] == "installed"
        assert data["packs"][0]["tags"] == ["affect"]
        assert data["packs"][1]["namespace"] == "local"

    def test_corrupt_pack_carries_error(self, session_and_client: tuple[SaklasSession, TestClient]):
        _, client = session_and_client
        rows = [
            ConceptRow(
                name="broken", namespace="default", status="corrupt",
                recommended_alpha=0.5, tags=[], description="",
                source="bundled", tensor_models=[],
                error="PackFormatError: stale v1",
            ),
        ]
        with patch(
            "saklas.io.cache_ops.list_concepts",
            return_value=PackListResult(installed=rows, hf_rows=[], error=None),
        ):
            resp = client.get("/saklas/v1/packs")
        assert resp.status_code == 200
        assert resp.json()["packs"][0]["status"] == "corrupt"
        assert "PackFormatError" in resp.json()["packs"][0]["error"]


# ---------------------------------------------------------------------------
# GET /saklas/v1/packs/search
# ---------------------------------------------------------------------------


class TestSearchPacks:
    def test_search_returns_rows(self, session_and_client: tuple[SaklasSession, TestClient]):
        _, client = session_and_client
        rows = [
            HfRow(
                name="lying.honest", namespace="someone",
                recommended_alpha=0.5, tags=["epistemic"],
                description="Lying vs honest probe", tensor_models=["llama__model"],
            ),
            HfRow(
                name="extra", namespace="other",
                recommended_alpha=0.3, tags=[],
                description="", tensor_models=[],
            ),
        ]
        with patch(
            "saklas.io.cache_ops.search_remote_packs",
            return_value=rows,
        ) as m:
            resp = client.get("/saklas/v1/packs/search", params={"q": "lying"})
        assert m.call_args.args[0] == "lying"
        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "lying"
        assert len(data["results"]) == 2
        assert data["results"][0]["name"] == "lying.honest"
        assert data["results"][0]["namespace"] == "someone"
        assert data["results"][0]["tags"] == ["epistemic"]

    def test_search_respects_limit(self, session_and_client: tuple[SaklasSession, TestClient]):
        _, client = session_and_client
        rows = [
            HfRow(name=f"r{i}", namespace="ns", recommended_alpha=0.5,
                  tags=[], description="", tensor_models=[])
            for i in range(20)
        ]
        with patch(
            "saklas.io.cache_ops.search_remote_packs",
            return_value=rows,
        ):
            resp = client.get(
                "/saklas/v1/packs/search", params={"q": "x", "limit": 5},
            )
        assert resp.status_code == 200
        assert len(resp.json()["results"]) == 5

    def test_search_empty_query_passes_through(self, session_and_client: tuple[SaklasSession, TestClient]):
        _, client = session_and_client
        with patch(
            "saklas.io.cache_ops.search_remote_packs",
            return_value=[],
        ) as m:
            resp = client.get("/saklas/v1/packs/search")
        # Default ``q=""`` — passes through to the underlying search.
        assert m.call_args.args[0] == ""
        assert resp.status_code == 200
        assert resp.json()["results"] == []

    def test_search_missing_hf_dep_503(self, session_and_client: tuple[SaklasSession, TestClient]):
        _, client = session_and_client
        with patch(
            "saklas.io.cache_ops.search_remote_packs",
            side_effect=ImportError("no module named huggingface_hub"),
        ):
            resp = client.get("/saklas/v1/packs/search", params={"q": "x"})
        assert resp.status_code == 503

    def test_search_transport_error_502(self, session_and_client: tuple[SaklasSession, TestClient]):
        _, client = session_and_client
        with patch(
            "saklas.io.cache_ops.search_remote_packs",
            side_effect=RuntimeError("HF flaky"),
        ):
            resp = client.get("/saklas/v1/packs/search", params={"q": "x"})
        assert resp.status_code == 502


# ---------------------------------------------------------------------------
# POST /saklas/v1/packs
# ---------------------------------------------------------------------------


class TestInstallPack:
    def test_install_success(self, session_and_client: tuple[SaklasSession, TestClient], tmp_path: Path):
        _, client = session_and_client
        installed_at = tmp_path / "vectors" / "ns" / "name"

        with patch(
            "saklas.io.cache_ops.install",
            return_value=installed_at,
        ) as m:
            resp = client.post(
                "/saklas/v1/packs",
                json={"target": "ns/concept"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["target"] == "ns/concept"
        assert data["installed_at"] == str(installed_at)
        assert data["statements_only"] is False
        # Positional args: ``target, as_``, kwargs: ``force, statements_only``.
        assert m.call_args.args[0] == "ns/concept"
        assert m.call_args.args[1] is None
        assert m.call_args.kwargs == {"force": False, "statements_only": False}

    def test_install_with_force_and_statements_only(self, session_and_client: tuple[SaklasSession, TestClient], tmp_path: Path):
        _, client = session_and_client
        installed_at = tmp_path / "vectors" / "local" / "x"

        with patch(
            "saklas.io.cache_ops.install",
            return_value=installed_at,
        ) as m:
            resp = client.post(
                "/saklas/v1/packs",
                json={
                    "target": "ns/concept",
                    "force": True,
                    "statements_only": True,
                },
            )
        assert resp.status_code == 200
        assert resp.json()["statements_only"] is True
        assert m.call_args.kwargs == {"force": True, "statements_only": True}

    def test_install_with_as_alias(self, session_and_client: tuple[SaklasSession, TestClient], tmp_path: Path):
        _, client = session_and_client
        with patch(
            "saklas.io.cache_ops.install",
            return_value=tmp_path,
        ) as m:
            resp = client.post(
                "/saklas/v1/packs",
                json={"target": "ns/concept", "as": "local/renamed"},
            )
        assert resp.status_code == 200
        # ``as`` is the wire field name; ``as_`` the Python attribute.
        assert m.call_args.args[1] == "local/renamed"

    def test_install_missing_target_404(self, session_and_client: tuple[SaklasSession, TestClient]):
        _, client = session_and_client
        with patch(
            "saklas.io.cache_ops.install",
            side_effect=FileNotFoundError("no such repo: ns/missing"),
        ):
            resp = client.post(
                "/saklas/v1/packs",
                json={"target": "ns/missing"},
            )
        assert resp.status_code == 404

    def test_install_conflict_409(self, session_and_client: tuple[SaklasSession, TestClient]):
        from saklas.io.cache_ops import InstallConflict
        _, client = session_and_client
        with patch(
            "saklas.io.cache_ops.install",
            side_effect=InstallConflict("destination already exists"),
        ):
            resp = client.post(
                "/saklas/v1/packs",
                json={"target": "ns/concept"},
            )
        # InstallConflict goes through the SaklasError handler (409 by user_message).
        assert resp.status_code == 409

    def test_install_value_error_400(self, session_and_client: tuple[SaklasSession, TestClient]):
        _, client = session_and_client
        with patch(
            "saklas.io.cache_ops.install",
            side_effect=ValueError("install target must be 'ns/name'"),
        ):
            resp = client.post(
                "/saklas/v1/packs",
                json={"target": "garbage"},
            )
        assert resp.status_code == 400
