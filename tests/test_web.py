"""Web UI mount + protocol additions for the analytics dashboard.

Covers:
* ``saklas.web.register_web_routes`` mounts the SPA bundle at ``/`` with
  a fallback to index.html for client-side routes.
* GET /saklas/v1/sessions/{id}/correlation returns the right matrix shape.
* The WS token event surfaces per_layer_scores when probes are loaded.

CPU-only.  No npm build runs here — the committed dist/ bundle is the
artifact under test.
"""
from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any
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


class _IdentityWhitener:
    def __init__(self) -> None:
        self.apply_calls: list[int] = []

    def covers_all(self, _layers: list[int]) -> bool:
        return True

    def apply_inv(self, layer: int, vec: torch.Tensor) -> torch.Tensor:
        self.apply_calls.append(int(layer))
        return vec.float().cpu()


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
    session.config.thinking = None

    session.vectors = vectors
    session.probes = {}
    session.history = []
    session.monitor = MagicMock()
    session.monitor.probe_names = []
    # Read-side analytics now flow through a CPU-snapshot cache so the polled
    # correlation/pairwise endpoints never touch the GPU.  Mirror that shape
    # against the mock's (already-CPU) vector Profiles.
    session.analytics_names = lambda: sorted(vectors.keys())
    session.analytics_profile = lambda name: vectors.get(name)
    session._tokenizer = MagicMock()
    session._layers = []
    session._gen_state = MagicMock()
    session._gen_state.finish_reason = "stop"
    session.whitener = _IdentityWhitener()
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
    def test_root_serves_spa_shell(self, web_client: Any) -> None:
        _session, client = web_client
        r = client.get("/")
        assert r.status_code == 200
        # SPA shell: doctype + an #app mount point + at least one /assets/*
        # reference (script or stylesheet — the bundler owns the names).
        assert b"<!DOCTYPE html>" in r.content
        assert b'id="app"' in r.content
        assets = _index_asset_paths(r.content)
        assert len(assets) >= 1, "index.html should reference at least one /assets/* file"

    def test_assets_referenced_by_index_are_servable(self, web_client: Any) -> None:
        _session, client = web_client
        index = client.get("/").content
        for path in _index_asset_paths(index):
            r = client.get(path)
            assert r.status_code == 200, f"missing asset: {path}"
            # Bundles are never empty in practice; even the smallest
            # Vite chunk weighs in at hundreds of bytes.
            assert len(r.content) > 0

    def test_unknown_route_falls_back_to_index(self, web_client: Any) -> None:
        _session, client = web_client
        # SPA fallback: /lab is a client-side route the SPA owns; the
        # server returns index.html so the SPA can take over routing.
        r = client.get("/lab")
        assert r.status_code == 200
        assert b'id="app"' in r.content

    def test_no_web_does_not_mount_root(self, api_only_client: Any) -> None:
        _session, client = api_only_client
        # ``--no-web`` (CLI) / ``web=False`` (library): GET / shouldn't
        # return the dashboard.  Detect by absence of the SPA's mount
        # point.
        r = client.get("/")
        assert b'id="app"' not in r.content

    def test_path_traversal_falls_back_to_index(self, web_client: Any) -> None:
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
# Protocol additions: correlation.
# ---------------------------------------------------------------------------


class TestCorrelationEndpoint:
    def test_default_returns_all_loaded_vectors(self, web_client: Any) -> None:
        _session, client = web_client
        r = client.get("/saklas/v1/sessions/default/correlation")
        assert r.status_code == 200
        data = r.json()
        assert sorted(data["names"]) == ["honest", "warm"]
        assert data["matrix"]["honest"]["honest"] == pytest.approx(1.0)
        # Symmetric off-diagonal.
        assert data["matrix"]["honest"]["warm"] == pytest.approx(data["matrix"]["warm"]["honest"])

    def test_names_filter_restricts_matrix(self, web_client: Any) -> None:
        _session, client = web_client
        r = client.get("/saklas/v1/sessions/default/correlation?names=honest")
        assert r.status_code == 200
        data = r.json()
        assert data["names"] == ["honest"]
        assert list(data["matrix"]["honest"].keys()) == ["honest"]

    def test_unknown_name_returns_404(self, web_client: Any) -> None:
        _session, client = web_client
        r = client.get("/saklas/v1/sessions/default/correlation?names=missing,honest")
        assert r.status_code == 404

    def test_layers_shared_records_pair_overlap(self, web_client: Any) -> None:
        _session, client = web_client
        r = client.get("/saklas/v1/sessions/default/correlation")
        data = r.json()
        # honest + warm both have layers {0, 5}, so shared = 2.
        key = "honest__warm"
        assert key in data["layers_shared"]
        assert data["layers_shared"][key] == 2

    def test_whitens_each_profile_layer_once_per_request(self, web_client: Any) -> None:
        session, client = web_client
        r = client.get("/saklas/v1/sessions/default/correlation")
        assert r.status_code == 200
        # 2 names × 2 layers.  The endpoint should reuse these factors across
        # the upper-triangle matrix instead of reapplying the whitener per pair.
        assert sorted(session.whitener.apply_calls) == [0, 0, 5, 5]


class TestRemovedDashboardSurfaces:
    def test_vector_profile_omits_layer_norm_payload(self, web_client: Any) -> None:
        _session, client = web_client
        r = client.get("/saklas/v1/sessions/default/vectors/honest")
        assert r.status_code == 200
        data = r.json()
        assert set(data) == {"name", "layers", "metadata"}

    def test_removed_native_and_styleguide_routes_are_not_spa_fallbacks(
        self, web_client: Any,
    ) -> None:
        _session, client = web_client
        for path in (
            "/saklas/v1/sessions/default/vectors/honest/diagnostics",
            "/saklas/v1/sessions/default/experiments/fan",
            "/styleguide",
            "/styleguide/anything",
        ):
            assert client.get(path).status_code == 404


# ---------------------------------------------------------------------------
# monitor.score_single_token_per_layer — un-aggregated heatmap source.
# ---------------------------------------------------------------------------


class TestScoreSingleTokenPerLayer:
    def test_returns_per_layer_per_probe_dict(self) -> None:
        # Build a real TraitMonitor, register two probes that share
        # layer 0, score against a hidden state.  No torch model needed.
        from saklas.core.monitor import Monitor
        from saklas.core.capture import fold_directions_to_subspace

        from tests._whitener import isotropic_whitener
        means = {0: torch.zeros(4)}
        whit = isotropic_whitener([0], 4)
        # Mahalanobis-only: an isotropic whitener reproduces the Euclidean
        # coordinate for these axis-aligned probes (diagonal Σ⁻¹ ⇒ axis 0 ⊥
        # axis 1).  Each direction folds to a 1-node ray (coord 1.0 at the
        # pole) — the session's vector-probe path.
        probes = {
            "honest": fold_directions_to_subspace(
                "honest", {0: torch.tensor([1.0, 0.0, 0.0, 0.0])},
                means, whitener=whit,
            ),
            "warm": fold_directions_to_subspace(
                "warm", {0: torch.tensor([0.0, 1.0, 0.0, 0.0])},
                means, whitener=whit,
            ),
        }
        monitor = Monitor(probes, layer_means=means, whitener=whit)

        hidden = {0: torch.tensor([1.0, 0.0, 0.0, 0.0])}
        result = monitor.score_single_token_per_layer(hidden)

        assert 0 in result
        assert set(result[0].keys()) == {"honest", "warm"}
        # Hidden aligns with the honest direction (coord ≈ 1 at the pole);
        # warm is orthogonal (coord ≈ 0 under isotropic Σ).
        assert result[0]["honest"] == pytest.approx(1.0, abs=0.1)
        assert result[0]["warm"] == pytest.approx(0.0, abs=0.1)

    def test_empty_input_returns_empty(self) -> None:
        from saklas.core.monitor import Monitor
        from saklas.core.capture import fold_directions_to_subspace
        from tests._whitener import isotropic_whitener

        means = {0: torch.zeros(4)}
        whit = isotropic_whitener([0], 4)
        m = fold_directions_to_subspace(
            "honest", {0: torch.tensor([1.0, 0.0, 0.0, 0.0])}, means,
            whitener=whit,
        )
        monitor = Monitor({"honest": m}, layer_means=means, whitener=whit)
        assert monitor.score_single_token_per_layer({}) == {}

    def test_layers_outside_probe_cache_omitted(self) -> None:
        from saklas.core.monitor import Monitor
        from saklas.core.capture import fold_directions_to_subspace
        from tests._whitener import isotropic_whitener

        means = {0: torch.zeros(4)}
        whit = isotropic_whitener([0], 4)
        m = fold_directions_to_subspace(
            "honest", {0: torch.tensor([1.0, 0.0, 0.0, 0.0])}, means,
            whitener=whit,
        )
        monitor = Monitor({"honest": m}, layer_means=means, whitener=whit)
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

    def test_register_against_empty_dist_raises_clear_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from fastapi import FastAPI

        from saklas.web import routes as web_routes

        # Point dist_path at an empty temp directory and verify the
        # error message names the build command.
        monkeypatch.setattr(web_routes, "dist_path", lambda: tmp_path)
        with pytest.raises(web_routes.WebUINotBuilt, match="npm run build"):
            web_routes.register_web_routes(FastAPI())
