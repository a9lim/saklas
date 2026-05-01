"""Web UI mount + protocol additions for the analytics dashboard.

Covers:
* ``saklas.web.register_web_routes`` mounts the SPA bundle at ``/`` with
  a fallback to index.html for client-side routes.
* GET /saklas/v1/sessions/{id}/correlation returns the right matrix shape.
* GET /saklas/v1/sessions/{id}/vectors/{name} carries per_layer_norms.
* The WS token event surfaces per_layer_scores when probes are loaded.

CPU-only.  No npm build runs here — the committed dist/ bundle is the
artifact under test.
"""
from __future__ import annotations

import asyncio
import threading
from unittest.mock import MagicMock

import pytest
import torch
from fastapi.testclient import TestClient

from saklas.core.profile import Profile


# ---------------------------------------------------------------------------
# Fixture: minimal session that exposes the surfaces the web routes touch.
# ---------------------------------------------------------------------------


def _profile_from_layers(layers: dict[int, list[float]]) -> Profile:
    return Profile(
        {layer: torch.tensor(values, dtype=torch.float32) for layer, values in layers.items()},
        metadata={"method": "contrastive_pca"},
    )


def _mock_session_with_vectors(vectors: dict[str, Profile]):
    session = MagicMock()
    session.model_id = "test/model"
    session.model_info = {"model_type": "gemma2", "num_layers": 4, "hidden_dim": 16}
    session._device = "cpu"
    session._dtype = "torch.bfloat16"
    session._created_ts = 1_700_000_000

    session.config = MagicMock()
    session.config.temperature = 1.0
    session.config.top_p = 0.9
    session.config.top_k = None
    session.config.max_new_tokens = 64
    session.config.system_prompt = None

    session.vectors = vectors
    session.probes = {}
    session.history = []
    session._monitor = MagicMock()
    session._monitor.probe_names = []
    session._tokenizer = MagicMock()
    session._layers = []
    session._gen_state = MagicMock()
    session._gen_state.finish_reason = "stop"
    session.lock = asyncio.Lock()

    session._trait_queues = []
    session._trait_lock = threading.Lock()
    session.register_trait_queue = lambda loop, q: session._trait_queues.append((loop, q))
    session.unregister_trait_queue = lambda loop, q: None

    session.events = MagicMock()
    session.events.subscribe = lambda cb: (lambda: None)
    session.events.emit = lambda event: None
    return session


@pytest.fixture
def web_client():
    from saklas.server import create_app

    vectors = {
        "honest": _profile_from_layers({
            0: [1.0, 0.0, 0.0, 0.0],
            5: [0.0, 1.0, 0.0, 0.0],
        }),
        "warm": _profile_from_layers({
            0: [0.5, 0.5, 0.0, 0.0],
            5: [0.0, 0.5, 0.5, 0.0],
        }),
    }
    session = _mock_session_with_vectors(vectors)
    app = create_app(session, default_steering=None, web=True)
    return session, TestClient(app)


@pytest.fixture
def api_only_client():
    """Same session but no dashboard mount; mirrors ``--no-web`` mode
    on the CLI and the library default (``create_app(..., web=False)``)
    so embedded API surfaces don't pick up the dashboard."""
    from saklas.server import create_app

    vectors = {
        "honest": _profile_from_layers({0: [1.0, 0.0]}),
        "warm": _profile_from_layers({0: [0.5, 0.5]}),
    }
    session = _mock_session_with_vectors(vectors)
    app = create_app(session, default_steering=None, web=False)
    return session, TestClient(app)


# ---------------------------------------------------------------------------
# Static-files mount.
# ---------------------------------------------------------------------------


def _index_asset_paths(html: bytes) -> list[str]:
    """Extract /assets/* paths the index.html references.

    Vite emits hashed filenames by default; the source-tree config pins
    saklas.js but lets the CSS chunk name itself.  Pulling references
    out of the html keeps the tests robust across bundler changes
    without sacrificing real coverage of the asset-mount path.
    """
    import re

    return [
        m.decode("utf-8")
        for m in re.findall(rb'(?:src|href)="(/assets/[^"]+)"', html)
    ]


class TestWebMount:
    def test_root_serves_spa_shell(self, web_client) -> None:
        _session, client = web_client
        r = client.get("/")
        assert r.status_code == 200
        # SPA shell: doctype + an #app mount point + at least one /assets/*
        # reference (script or stylesheet — the bundler owns the names).
        assert b"<!DOCTYPE html>" in r.content
        assert b'id="app"' in r.content
        assets = _index_asset_paths(r.content)
        assert len(assets) >= 1, "index.html should reference at least one /assets/* file"

    def test_assets_referenced_by_index_are_servable(self, web_client) -> None:
        _session, client = web_client
        index = client.get("/").content
        for path in _index_asset_paths(index):
            r = client.get(path)
            assert r.status_code == 200, f"missing asset: {path}"
            # Bundles are never empty in practice; even the smallest
            # Vite chunk weighs in at hundreds of bytes.
            assert len(r.content) > 0

    def test_unknown_route_falls_back_to_index(self, web_client) -> None:
        _session, client = web_client
        # SPA fallback: /lab is a client-side route the SPA owns; the
        # server returns index.html so the SPA can take over routing.
        r = client.get("/lab")
        assert r.status_code == 200
        assert b'id="app"' in r.content

    def test_no_web_does_not_mount_root(self, api_only_client) -> None:
        _session, client = api_only_client
        # ``--no-web`` (CLI) / ``web=False`` (library): GET / shouldn't
        # return the dashboard.  Detect by absence of the SPA's mount
        # point.
        r = client.get("/")
        assert b'id="app"' not in r.content

    def test_path_traversal_falls_back_to_index(self, web_client) -> None:
        # The SPA fallback ``/{full_path:path}`` accepts attacker-
        # controlled input.  ``..`` segments and absolute-style paths
        # must not escape the dist directory; the resolver clamps to
        # ``index.html`` when the candidate would resolve outside.
        _session, client = web_client
        for evil in (
            "../../../etc/passwd",
            "..%2f..%2fetc%2fpasswd",
            "assets/../../../etc/passwd",
        ):
            r = client.get(f"/{evil}")
            assert r.status_code == 200, f"unexpected status for {evil!r}"
            # Either the SPA shell or a real /assets/* asset; never the
            # contents of /etc/passwd.
            assert b"root:" not in r.content
            assert b"/bin/" not in r.content


# ---------------------------------------------------------------------------
# Protocol additions: correlation + per_layer_norms.
# ---------------------------------------------------------------------------


class TestCorrelationEndpoint:
    def test_default_returns_all_loaded_vectors(self, web_client) -> None:
        _session, client = web_client
        r = client.get("/saklas/v1/sessions/default/correlation")
        assert r.status_code == 200
        data = r.json()
        assert sorted(data["names"]) == ["honest", "warm"]
        assert data["matrix"]["honest"]["honest"] == pytest.approx(1.0)
        # Symmetric off-diagonal.
        assert data["matrix"]["honest"]["warm"] == pytest.approx(data["matrix"]["warm"]["honest"])

    def test_names_filter_restricts_matrix(self, web_client) -> None:
        _session, client = web_client
        r = client.get("/saklas/v1/sessions/default/correlation?names=honest")
        assert r.status_code == 200
        data = r.json()
        assert data["names"] == ["honest"]
        assert list(data["matrix"]["honest"].keys()) == ["honest"]

    def test_unknown_name_returns_404(self, web_client) -> None:
        _session, client = web_client
        r = client.get("/saklas/v1/sessions/default/correlation?names=missing,honest")
        assert r.status_code == 404

    def test_layers_shared_records_pair_overlap(self, web_client) -> None:
        _session, client = web_client
        r = client.get("/saklas/v1/sessions/default/correlation")
        data = r.json()
        # honest + warm both have layers {0, 5}, so shared = 2.
        key = "honest__warm"
        assert key in data["layers_shared"]
        assert data["layers_shared"][key] == 2


class TestVectorPerLayerNorms:
    def test_get_vector_returns_per_layer_norms(self, web_client) -> None:
        _session, client = web_client
        r = client.get("/saklas/v1/sessions/default/vectors/honest")
        assert r.status_code == 200
        data = r.json()
        assert "per_layer_norms" in data
        norms = data["per_layer_norms"]
        # Layers from the fixture profile: {0, 5}.
        assert set(norms.keys()) == {"0", "5"}
        # Layer 0 is the unit vector along axis 0; norm = 1.0.
        assert norms["0"] == pytest.approx(1.0)
        assert norms["5"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# monitor.score_single_token_per_layer — un-aggregated heatmap source.
# ---------------------------------------------------------------------------


class TestScoreSingleTokenPerLayer:
    def test_returns_per_layer_per_probe_dict(self) -> None:
        # Build a real TraitMonitor, register two probes that share
        # layer 0, score against a hidden state.  No torch model needed.
        from saklas.core.monitor import TraitMonitor

        probes = {
            "honest": {0: torch.tensor([1.0, 0.0, 0.0, 0.0])},
            "warm":   {0: torch.tensor([0.0, 1.0, 0.0, 0.0])},
        }
        monitor = TraitMonitor(probes, layer_means={0: torch.zeros(4)})

        hidden = {0: torch.tensor([1.0, 0.0, 0.0, 0.0])}
        result = monitor.score_single_token_per_layer(hidden)

        assert 0 in result
        assert set(result[0].keys()) == {"honest", "warm"}
        # Hidden state aligns perfectly with the honest direction; warm
        # is orthogonal.
        assert result[0]["honest"] == pytest.approx(1.0, abs=1e-4)
        assert result[0]["warm"] == pytest.approx(0.0, abs=1e-4)

    def test_empty_input_returns_empty(self) -> None:
        from saklas.core.monitor import TraitMonitor

        monitor = TraitMonitor(
            {"honest": {0: torch.tensor([1.0, 0.0, 0.0, 0.0])}},
            layer_means={0: torch.zeros(4)},
        )
        assert monitor.score_single_token_per_layer({}) == {}

    def test_layers_outside_probe_cache_omitted(self) -> None:
        from saklas.core.monitor import TraitMonitor

        monitor = TraitMonitor(
            {"honest": {0: torch.tensor([1.0, 0.0, 0.0, 0.0])}},
            layer_means={0: torch.zeros(4)},
        )
        # Hidden state at layer 1, but honest only covers layer 0.
        hidden = {1: torch.tensor([1.0, 0.0, 0.0, 0.0])}
        result = monitor.score_single_token_per_layer(hidden)
        # Layer 1 has no probe coverage; output omits it entirely.
        assert result == {}


# ---------------------------------------------------------------------------
# saklas.web.dist_path / register_web_routes wiring.
# ---------------------------------------------------------------------------


class TestRegisterWebRoutes:
    def test_dist_path_resolves_to_real_directory(self) -> None:
        from saklas.web import dist_path

        d = dist_path()
        assert d.is_dir()
        assert (d / "index.html").is_file()
        assert (d / "assets").is_dir()

    def test_register_against_empty_dist_raises_clear_error(self, tmp_path, monkeypatch) -> None:
        from fastapi import FastAPI

        from saklas.web import routes as web_routes

        # Point dist_path at an empty temp directory and verify the
        # error message names the build command.
        monkeypatch.setattr(web_routes, "dist_path", lambda: tmp_path)
        with pytest.raises(web_routes.WebUINotBuilt, match="npm run build"):
            web_routes.register_web_routes(FastAPI())
