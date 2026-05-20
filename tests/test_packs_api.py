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

    def test_lists_installed_carries_has_tensor_for_session_model(
        self, session_and_client: tuple[SaklasSession, TestClient],
    ):
        """``has_tensor`` is the session-relative flag the unified webui
        vectors drawer splits on — ``True`` when the pack has a baked
        tensor for the loaded model (``<safe_model_id>.safetensors``).
        ``session.model_id = "test/model"`` → safe id ``test__model``.
        """
        _, client = session_and_client
        rows = [
            ConceptRow(
                name="has_one", namespace="default", status="installed",
                recommended_alpha=0.5, tags=[], description="",
                source="bundled", tensor_models=["test__model", "other__id"],
            ),
            ConceptRow(
                name="other_model_only", namespace="default", status="installed",
                recommended_alpha=0.5, tags=[], description="",
                source="bundled", tensor_models=["other__id"],
            ),
            ConceptRow(
                name="empty", namespace="local", status="installed",
                recommended_alpha=0.4, tags=[], description="",
                source="local", tensor_models=[],
            ),
        ]
        with patch(
            "saklas.io.cache_ops.list_concepts",
            return_value=PackListResult(installed=rows, hf_rows=[], error=None),
        ):
            resp = client.get("/saklas/v1/packs")
        data = resp.json()
        by_name = {p["name"]: p for p in data["packs"]}
        assert by_name["has_one"]["has_tensor"] is True
        assert by_name["other_model_only"]["has_tensor"] is False
        assert by_name["empty"]["has_tensor"] is False

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


# ---------------------------------------------------------------------------
# DELETE /saklas/v1/packs/{namespace}/{name}
# ---------------------------------------------------------------------------


class TestDeletePack:
    def _patch_list(self, rows: list[ConceptRow]):
        return patch(
            "saklas.io.cache_ops.list_concepts",
            return_value=PackListResult(installed=rows, hf_rows=[], error=None),
        )

    def test_delete_local_pack_removes_folder(
        self, session_and_client: tuple[SaklasSession, TestClient],
    ):
        session, client = session_and_client
        rows = [
            ConceptRow(
                name="custom", namespace="local", status="installed",
                recommended_alpha=0.5, tags=[], description="",
                source="local", tensor_models=["test__model"],
            ),
        ]
        with self._patch_list(rows), patch(
            "saklas.io.cache_ops.uninstall", return_value=1,
        ) as m_un:
            resp = client.delete("/saklas/v1/packs/local/custom")
        assert resp.status_code == 200
        body = resp.json()
        assert body == {
            "namespace": "local",
            "name": "custom",
            "source": "local",
            "removed": 1,
            "rematerializes_on_restart": False,
        }
        # ``yes=True`` is load-bearing — fully-qualified selector still
        # gets the explicit confirm flag from the route.
        assert m_un.call_args.kwargs == {"yes": True}

    def test_delete_bundled_carries_rematerialize_hint(
        self, session_and_client: tuple[SaklasSession, TestClient],
    ):
        _, client = session_and_client
        rows = [
            ConceptRow(
                name="happy.sad", namespace="default", status="installed",
                recommended_alpha=0.5, tags=[], description="",
                source="bundled", tensor_models=[],
            ),
        ]
        with self._patch_list(rows), patch(
            "saklas.io.cache_ops.uninstall", return_value=1,
        ):
            resp = client.delete("/saklas/v1/packs/default/happy.sad")
        assert resp.status_code == 200
        assert resp.json()["rematerializes_on_restart"] is True

    def test_delete_missing_404(
        self, session_and_client: tuple[SaklasSession, TestClient],
    ):
        _, client = session_and_client
        with self._patch_list([]):
            resp = client.delete("/saklas/v1/packs/local/nonexistent")
        assert resp.status_code == 404

    def test_delete_unregisters_from_session(
        self, session_and_client: tuple[SaklasSession, TestClient],
    ):
        """If the concept is currently registered (steering rack), the
        delete route detaches it first so the engine doesn't keep a
        stale pointer."""
        from typing import cast, Any
        session, client = session_and_client
        # Cast through ``Any`` — the fixture is typed as the real
        # SaklasSession for editor help, but the underlying object is
        # a MagicMock, so attribute writes / mock-call assertions need
        # the type relaxed.
        mock_session = cast(Any, session)
        mock_session.vectors = {"custom": object()}
        rows = [
            ConceptRow(
                name="custom", namespace="local", status="installed",
                recommended_alpha=0.5, tags=[], description="",
                source="local", tensor_models=["test__model"],
            ),
        ]
        with self._patch_list(rows), patch(
            "saklas.io.cache_ops.uninstall", return_value=1,
        ):
            resp = client.delete("/saklas/v1/packs/local/custom")
        assert resp.status_code == 200
        # ``unsteer`` was called with the canonical name.
        mock_session.unsteer.assert_any_call("custom")
