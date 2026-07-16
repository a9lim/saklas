"""Route tests for the unified ``/instruments`` family (no GPU required)."""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

_SID = "default"
_BASE = f"/saklas/v1/sessions/{_SID}/instruments"


def _mock_session() -> Any:
    session = MagicMock()
    session.model_id = "test/model"
    session.model_info = {
        "model_type": "gemma2", "num_layers": 26, "hidden_dim": 2304,
        "device": "cpu", "dtype": "torch.bfloat16",
    }
    session.model = MagicMock()
    session.model.config.model_type = "gemma2"
    session.layers = [MagicMock() for _ in range(26)]
    session.config = MagicMock()
    session.config.system_prompt = None
    session.config.thinking = None
    session.tokenizer = MagicMock()

    # instrument state reads
    session.monitor.probe_names = []
    session.lens_probe_names = []
    session.sae_probe_names = []
    session.live_probe_scores = True
    session.live_lens_layers = None
    session.live_sae = False
    session._live_sae = None
    session.sae_info = None

    session.lock = asyncio.Lock()
    return session


@pytest.fixture
def session_and_client():
    from saklas.server import create_app
    session = _mock_session()
    app = create_app(session, default_steering=None)
    with TestClient(app) as client:
        yield session, client


# ---------------------------------------------------------------------------
# GET /instruments — family enumeration
# ---------------------------------------------------------------------------

class TestListing:
    def test_enumerates_three_families(
        self, session_and_client: Any, monkeypatch: Any,
    ) -> None:
        session, client = session_and_client
        monkeypatch.setattr(
            "saklas.io.lens_sources.list_lens_sources",
            lambda _m: [{"source": "local:default", "active": True}],
        )
        session.lens_probe_names = ["jlens/fake"]
        session.sae_info = {"release": "scope", "layer": 14, "width": 16_384}
        resp = client.get(_BASE)
        assert resp.status_code == 200
        fams = {f["family"]: f for f in resp.json()["instruments"]}
        assert set(fams) == {"geometry", "lens", "sae"}

        geo = fams["geometry"]
        assert geo["live"] == {"enabled": True}
        assert geo["source"] is None
        assert geo["capabilities"] == {
            "sources": False, "preparations": [],
            "token_readout": True, "source_switch": False,
        }

        lens = fams["lens"]
        assert lens["live"] == {"enabled": False, "layers": None}
        assert lens["source"] == "local:default"
        assert lens["probes"] == ["jlens/fake"]
        assert lens["capabilities"] == {
            "sources": True, "preparations": ["fetch", "fit"],
            "token_readout": True, "source_switch": True,
        }

        sae = fams["sae"]
        assert sae["live"] == {"enabled": False, "layer": None, "top_k": None}
        assert sae["source"] == "saelens:scope"
        assert sae["capabilities"] == {
            "sources": True, "preparations": ["load", "train"],
            "token_readout": True, "source_switch": False,
        }

    def test_live_states_reflect_runtime(self, session_and_client: Any) -> None:
        session, client = session_and_client
        session.live_probe_scores = False
        session.live_lens_layers = [10, 14]
        session.live_sae = True
        session._live_sae = {"layer": 14, "top_k": 12, "source": "saelens:x"}
        fams = {f["family"]: f for f in client.get(_BASE).json()["instruments"]}
        assert fams["geometry"]["live"] == {"enabled": False}
        assert fams["lens"]["live"] == {"enabled": True, "layers": [10, 14]}
        assert fams["sae"]["live"] == {
            "enabled": True, "layer": 14, "top_k": 12,
        }


# ---------------------------------------------------------------------------
# POST /instruments/{family}/live — uniform live toggle
# ---------------------------------------------------------------------------

class TestLiveToggle:
    def test_geometry_toggle(self, session_and_client: Any) -> None:
        session, client = session_and_client
        session.set_live_probe_scores.return_value = False
        resp = client.post(f"{_BASE}/geometry/live", json={"enabled": False})
        assert resp.status_code == 200
        assert resp.json() == {"enabled": False}
        session.set_live_probe_scores.assert_called_once_with(False)

    def test_geometry_rejects_layers(self, session_and_client: Any) -> None:
        _session, client = session_and_client
        resp = client.post(
            f"{_BASE}/geometry/live", json={"enabled": True, "layers": [1, 2]},
        )
        assert resp.status_code == 400

    def test_geometry_rejects_top_k(self, session_and_client: Any) -> None:
        _session, client = session_and_client
        resp = client.post(
            f"{_BASE}/geometry/live", json={"enabled": True, "top_k": 5},
        )
        assert resp.status_code == 400

    def test_lens_enable_returns_layers(self, session_and_client: Any) -> None:
        session, client = session_and_client
        session.enable_live_lens.return_value = [10, 14, 18]
        resp = client.post(f"{_BASE}/lens/live", json={"enabled": True})
        assert resp.status_code == 200
        assert resp.json() == {"enabled": True, "layers": [10, 14, 18]}
        assert session.enable_live_lens.call_args.kwargs["layers"] is None

    def test_lens_disable(self, session_and_client: Any) -> None:
        session, client = session_and_client
        resp = client.post(f"{_BASE}/lens/live", json={"enabled": False})
        assert resp.status_code == 200
        assert resp.json() == {"enabled": False, "layers": None}
        session.disable_live_lens.assert_called_once()

    def test_lens_not_fitted_404(self, session_and_client: Any) -> None:
        from saklas.core.jlens import LensNotFittedError
        session, client = session_and_client
        session.enable_live_lens.side_effect = LensNotFittedError("no lens")
        resp = client.post(f"{_BASE}/lens/live", json={"enabled": True})
        assert resp.status_code == 404

    def test_lens_rejects_top_k(self, session_and_client: Any) -> None:
        _session, client = session_and_client
        resp = client.post(
            f"{_BASE}/lens/live", json={"enabled": True, "top_k": 4},
        )
        assert resp.status_code == 400

    def test_sae_enable(self, session_and_client: Any) -> None:
        session, client = session_and_client
        session.enable_live_sae.return_value = {"layer": 14, "top_k": 12}
        resp = client.post(
            f"{_BASE}/sae/live", json={"enabled": True, "top_k": 12},
        )
        assert resp.status_code == 200
        assert resp.json() == {"enabled": True, "layer": 14, "top_k": 12}
        session.enable_live_sae.assert_called_once_with(top_k=12)

    def test_sae_disable(self, session_and_client: Any) -> None:
        session, client = session_and_client
        resp = client.post(f"{_BASE}/sae/live", json={"enabled": False})
        assert resp.status_code == 200
        assert resp.json() == {"enabled": False, "layer": None, "top_k": None}
        session.disable_live_sae.assert_called_once()

    def test_sae_rejects_layers(self, session_and_client: Any) -> None:
        _session, client = session_and_client
        resp = client.post(
            f"{_BASE}/sae/live", json={"enabled": True, "layers": [1]},
        )
        assert resp.status_code == 400

    def test_unknown_family_404(self, session_and_client: Any) -> None:
        _session, client = session_and_client
        resp = client.post(f"{_BASE}/nope/live", json={"enabled": True})
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /instruments/{family}/sources
# ---------------------------------------------------------------------------

class TestSources:
    def test_geometry_404(self, session_and_client: Any) -> None:
        _session, client = session_and_client
        resp = client.get(f"{_BASE}/geometry/sources")
        assert resp.status_code == 404
        assert "source lifecycle" in resp.json()["detail"]

    def test_lens_lists_prepared(
        self, session_and_client: Any, monkeypatch: Any,
    ) -> None:
        _session, client = session_and_client
        monkeypatch.setattr(
            "saklas.io.lens_sources.list_lens_sources",
            lambda _m: [{
                "source": "local:default", "kind": "local", "name": "default",
                "active": True, "path": "/tmp/x.json",
            }],
        )
        resp = client.get(f"{_BASE}/lens/sources")
        assert resp.status_code == 200
        rows = resp.json()["sources"]
        assert rows[0]["source"] == "local:default"
        assert "path" not in rows[0]  # path stripped

    def test_sae_merges_prepared_and_releases(
        self, session_and_client: Any, monkeypatch: Any,
    ) -> None:
        _session, client = session_and_client
        monkeypatch.setattr(
            "saklas.io.sae.list_sae_sources",
            lambda _m: [{
                "source": "local:mine", "kind": "local", "name": "mine",
                "active": True, "path": "/tmp/m.json", "layer": 14,
                "features": 4096,
            }],
        )
        monkeypatch.setattr(
            "saklas.core.sae.list_sae_releases",
            lambda _m: [{"release": "scope", "layer": 14}],
        )
        resp = client.get(f"{_BASE}/sae/sources")
        assert resp.status_code == 200
        body = resp.json()
        assert body["sources"][0]["source"] == "local:mine"
        assert "path" not in body["sources"][0]
        assert body["releases"][0]["release"] == "scope"


# ---------------------------------------------------------------------------
# PUT /instruments/{family}/source
# ---------------------------------------------------------------------------

class TestSourceSwitch:
    def test_lens_switch(self, session_and_client: Any) -> None:
        session, client = session_and_client
        session.enable_live_lens.return_value = [10, 11]
        resp = client.put(
            f"{_BASE}/lens/source", json={"source": "neuronpedia"},
        )
        assert resp.status_code == 200
        assert resp.json() == {"source": "neuronpedia", "live_layers": [10, 11]}
        session.disable_live_lens.assert_called_once()
        session.select_jlens_source.assert_called_once_with("neuronpedia")

    def test_sae_409_points_at_preparations(
        self, session_and_client: Any,
    ) -> None:
        _session, client = session_and_client
        resp = client.put(f"{_BASE}/sae/source", json={"source": "saelens:x"})
        assert resp.status_code == 409
        assert "preparation" in resp.json()["detail"]

    def test_geometry_404(self, session_and_client: Any) -> None:
        _session, client = session_and_client
        resp = client.put(f"{_BASE}/geometry/source", json={"source": "x"})
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST/GET/DELETE /instruments/{family}/preparations
# ---------------------------------------------------------------------------

class TestPreparations:
    def test_geometry_404(self, session_and_client: Any) -> None:
        _session, client = session_and_client
        assert client.get(f"{_BASE}/geometry/preparations").status_code == 404
        assert client.post(
            f"{_BASE}/geometry/preparations", json={"operation": "fit"},
        ).status_code == 404
        assert client.delete(f"{_BASE}/geometry/preparations").status_code == 404

    def test_idle_status_before_any_op(self, session_and_client: Any) -> None:
        _session, client = session_and_client
        body = client.get(f"{_BASE}/lens/preparations").json()
        assert body["state"] == "idle"
        assert body["operation"] is None

    def test_unknown_operation_400(self, session_and_client: Any) -> None:
        _session, client = session_and_client
        resp = client.post(
            f"{_BASE}/lens/preparations", json={"operation": "load"},
        )
        assert resp.status_code == 400  # 'load' is an sae op, not lens

    def test_lens_fit_status_mapping_and_mutual_exclusion(
        self, session_and_client: Any, monkeypatch: Any,
    ) -> None:
        session, client = session_and_client
        started = threading.Event()
        release = threading.Event()

        def _blocking_fit(*_a: Any, **_k: Any) -> None:
            started.set()
            assert release.wait(timeout=2.0)

        session.fit_jlens.side_effect = _blocking_fit
        monkeypatch.setattr(
            "saklas.io.lens.stream_default_lens_corpus",
            lambda _n, *, cancel_event=None: (["a long enough prompt."], "spec"),
        )
        resp = client.post(
            f"{_BASE}/lens/preparations",
            json={"operation": "fit", "prompts": 1, "layers": "workspace"},
        )
        assert resp.status_code == 202
        body = resp.json()
        assert body["operation"] == "fit"
        assert body["state"] == "running"
        assert body["progress"] == {
            "current": 0, "total": 1, "unit": "prompts",
        }
        assert body["cancellable"] is True

        assert started.wait(timeout=2.0)
        try:
            # fetch shares the mutual-exclusion group → 409 while fit runs
            second = client.post(
                f"{_BASE}/lens/preparations", json={"operation": "fetch"},
            )
            assert second.status_code == 409
            # DELETE cancels the running fit
            cancel = client.delete(f"{_BASE}/lens/preparations")
            assert cancel.status_code == 200
            assert cancel.json()["message"] == "cancelling…"
        finally:
            release.set()

    def test_lens_delete_without_running_fit_409(
        self, session_and_client: Any,
    ) -> None:
        _session, client = session_and_client
        resp = client.delete(f"{_BASE}/lens/preparations")
        assert resp.status_code == 409

    def test_sae_train_status_mapping(
        self, session_and_client: Any, monkeypatch: Any,
    ) -> None:
        session, client = session_and_client
        session.train_sae.return_value = {
            "runtime": {}, "metrics": {"tokens_trained": 100}, "source": "local:s",
        }
        monkeypatch.setattr(
            "saklas.io.lens.stream_default_lens_corpus",
            lambda _n, **_k: (["p"], "spec"),
        )
        resp = client.post(
            f"{_BASE}/sae/preparations",
            json={"operation": "train", "name": "mine", "tokens": 1000},
        )
        assert resp.status_code == 202
        body = resp.json()
        assert body["operation"] == "train"
        assert body["progress"]["unit"] == "tokens"
        assert body["cancellable"] is True
        # let the worker finish
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            st = client.get(f"{_BASE}/sae/preparations").json()
            if st["state"] != "running":
                break
            time.sleep(0.01)

    def test_sae_load_no_denominator_progress_null(
        self, session_and_client: Any,
    ) -> None:
        session, client = session_and_client
        session.load_sae.return_value = {"layer": 14, "width": 4096}
        resp = client.post(
            f"{_BASE}/sae/preparations",
            json={"operation": "load", "release": "saelens:scope"},
        )
        assert resp.status_code == 202
        assert resp.json()["operation"] == "load"
        assert resp.json()["progress"] is None
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            st = client.get(f"{_BASE}/sae/preparations").json()
            if st["state"] != "running":
                break
            time.sleep(0.01)
        session.load_sae.assert_called_once_with("scope", layer=None)

    def test_sae_delete_unloads_when_no_train(
        self, session_and_client: Any,
    ) -> None:
        session, client = session_and_client
        resp = client.delete(f"{_BASE}/sae/preparations")
        assert resp.status_code == 200
        session.unload_sae.assert_called_once()


# ---------------------------------------------------------------------------
# GET /instruments/{family}/token-readout — measurements replay envelope
# ---------------------------------------------------------------------------

class TestTokenReadout:
    _LENS_OUT = {
        "node_id": "n1", "raw_index": 3, "token_id": 42, "token_text": " magic",
        "steering": "0.3 formal.casual",
        "readout": {
            18: [(" b", -0.5, 7), (" c", -1.2, 9)],
            12: [(" a", -0.25, 5), (" d", -2.0, 3)],
        },
        "aggregate": [(" a", 0.41, 0.31, 0.05), (" b", 0.2, 0.8, 0.1)],
    }

    @staticmethod
    def _geometry_out(steering: "str | None") -> dict[str, Any]:
        from saklas.core.results import ProbeReading

        return {
            "node_id": "n1", "raw_index": 3, "token_id": 42,
            "token_text": " magic", "steering": steering,
            "readings": {
                "formal.casual": ProbeReading(
                    fraction=0.4,
                    nearest=[("formal", 1.2)],
                    coords=(0.7,),
                    residual=0.0,
                ),
            },
        }

    def test_geometry_replay_envelope(self, session_and_client: Any) -> None:
        session, client = session_and_client
        session.geometry_token_readout.return_value = self._geometry_out(
            "0.3 formal.casual",
        )
        resp = client.get(
            f"{_BASE}/geometry/token-readout",
            params={"node_id": "n1", "raw_index": 3},
        )
        assert resp.status_code == 200
        m = resp.json()["measurements"]
        assert m["scope"] == "replay"
        assert m["provenance"] == "replayed"
        geo = m["instruments"]["geometry"]
        assert geo["binding"] == {
            "source": None, "steering": "0.3 formal.casual",
        }
        reading = geo["readings"]["formal.casual"]
        assert reading["coords"] == [0.7]
        assert m["scores"]["formal.casual"] == 0.7
        kwargs = session.geometry_token_readout.call_args.kwargs
        assert kwargs["apply_steering"] is True
        assert kwargs["raw"] is False

    def test_geometry_unsteered_nulls_binding_steering(
        self, session_and_client: Any,
    ) -> None:
        session, client = session_and_client
        session.geometry_token_readout.return_value = self._geometry_out(None)
        resp = client.get(
            f"{_BASE}/geometry/token-readout",
            params={"node_id": "n1", "raw_index": 3, "steered": "false"},
        )
        assert resp.status_code == 200
        geo = resp.json()["measurements"]["instruments"]["geometry"]
        assert geo["binding"] == {"source": None, "steering": None}
        call = session.geometry_token_readout.call_args
        assert call.kwargs["apply_steering"] is False

    def test_geometry_no_probes_400(self, session_and_client: Any) -> None:
        session, client = session_and_client
        session.geometry_token_readout.side_effect = ValueError(
            "no geometry probes attached",
        )
        resp = client.get(
            f"{_BASE}/geometry/token-readout",
            params={"node_id": "n1", "raw_index": 0},
        )
        assert resp.status_code == 400

    def test_geometry_unknown_node_404(self, session_and_client: Any) -> None:
        from saklas.core.loom import UnknownNodeError

        session, client = session_and_client
        session.geometry_token_readout.side_effect = UnknownNodeError("gone")
        resp = client.get(
            f"{_BASE}/geometry/token-readout",
            params={"node_id": "nope", "raw_index": 0},
        )
        assert resp.status_code == 404

    def test_lens_replay_envelope(
        self, session_and_client: Any, monkeypatch: Any,
    ) -> None:
        session, client = session_and_client
        monkeypatch.setattr(
            "saklas.io.lens_sources.list_lens_sources",
            lambda _m: [{"source": "local:default", "active": True}],
        )
        session.jlens_token_readout.return_value = dict(self._LENS_OUT)
        resp = client.get(
            f"{_BASE}/lens/token-readout",
            params={"node_id": "n1", "raw_index": 3, "top_k": 2},
        )
        assert resp.status_code == 200
        m = resp.json()["measurements"]
        assert m["scope"] == "replay"
        assert m["provenance"] == "replayed"
        lens = m["instruments"]["lens"]
        assert lens["binding"]["source"] == "local:default"
        assert lens["binding"]["steering"] == "0.3 formal.casual"
        assert [row["layer"] for row in lens["readout"]["layers"]] == [12, 18]
        assert lens["readout"]["aggregate"][0]["token"] == " a"
        assert session.jlens_token_readout.call_args.kwargs["top_k"] == 2

    def test_lens_unsteered_nulls_binding_steering(
        self, session_and_client: Any,
    ) -> None:
        session, client = session_and_client
        session.jlens_token_readout.return_value = dict(self._LENS_OUT)
        resp = client.get(
            f"{_BASE}/lens/token-readout",
            params={"node_id": "n1", "raw_index": 3, "steered": "false"},
        )
        assert resp.status_code == 200
        binding = resp.json()["measurements"]["instruments"]["lens"]["binding"]
        assert binding["steering"] is None
        assert session.jlens_token_readout.call_args.kwargs["apply_steering"] is False

    def test_lens_not_fitted_404(self, session_and_client: Any) -> None:
        from saklas.core.jlens import LensNotFittedError
        session, client = session_and_client
        session.jlens_token_readout.side_effect = LensNotFittedError("nope")
        resp = client.get(
            f"{_BASE}/lens/token-readout",
            params={"node_id": "n1", "raw_index": 0},
        )
        assert resp.status_code == 404

    def test_sae_replay_envelope(self, session_and_client: Any) -> None:
        session, client = session_and_client
        session.sae_info = {"release": "scope", "layer": 14, "width": 4096}
        session.sae_token_readout.return_value = {
            "node_id": "n1", "raw_index": 2, "token_id": 7, "token_text": "x",
            "steering": None, "layer": 14,
            "features": [{
                "id": 42, "activation": 3.5, "label": "fruit", "max_act": 121.1,
            }],
        }
        resp = client.get(
            f"{_BASE}/sae/token-readout",
            params={"node_id": "n1", "raw_index": 2},
        )
        assert resp.status_code == 200
        sae = resp.json()["measurements"]["instruments"]["sae"]
        assert sae["binding"]["source"] == "saelens:scope"
        assert sae["binding"]["layer"] == 14
        assert sae["readout"]["features"][0]["id"] == 42


# ---------------------------------------------------------------------------
# Family extras
# ---------------------------------------------------------------------------

class TestExtras:
    def test_lens_token_validate(self, session_and_client: Any) -> None:
        session, client = session_and_client
        session.tokenizer.encode.return_value = [42]
        session.tokenizer.decode.return_value = " magic"
        resp = client.post(
            f"{_BASE}/lens/token/validate", json={"word": "  magic  "},
        )
        assert resp.status_code == 200
        assert resp.json() == {"word": "magic", "token_id": 42}

    def test_lens_token_validate_multi_token_400(
        self, session_and_client: Any,
    ) -> None:
        session, client = session_and_client
        session.tokenizer.encode.return_value = [3, 4]
        session.tokenizer.decode.side_effect = lambda ids: {
            3: "anti", 4: "dis",
        }[ids[0]]
        resp = client.post(
            f"{_BASE}/lens/token/validate", json={"word": "antidis"},
        )
        assert resp.status_code == 400

    def test_sae_feature_validate(self, session_and_client: Any) -> None:
        session, client = session_and_client
        session.validate_sae_feature.return_value = {
            "id": 42, "label": "fruit", "layer": 14,
        }
        resp = client.post(
            f"{_BASE}/sae/features/validate", json={"id": 42},
        )
        assert resp.status_code == 200
        assert resp.json() == {"id": 42, "label": "fruit", "layer": 14}
        session.validate_sae_feature.assert_called_once_with(42)

    def test_sae_features_metadata(self, session_and_client: Any) -> None:
        session, client = session_and_client
        session.fetch_sae_feature_meta.return_value = {
            "42": {"label": "fruit", "max_act": 121.1},
        }
        resp = client.post(
            f"{_BASE}/sae/features/metadata", json={"ids": [42, 42, 7]},
        )
        assert resp.status_code == 200
        assert resp.json() == {
            "features": {"42": {"label": "fruit", "max_act": 121.1}},
        }
        session.fetch_sae_feature_meta.assert_called_once_with([42, 42, 7])

    def test_sae_features_metadata_oversized_400(
        self, session_and_client: Any,
    ) -> None:
        _session, client = session_and_client
        resp = client.post(
            f"{_BASE}/sae/features/metadata", json={"ids": list(range(65))},
        )
        assert resp.status_code == 400
