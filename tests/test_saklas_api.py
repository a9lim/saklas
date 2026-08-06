"""Tests for the native /saklas/v1/* API (no GPU)."""

import asyncio
import json
import time
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from saklas.core.errors import SaklasError
from saklas.core.generation import GenerationConfig
from saklas.core.results import GenerationResult, RunSet
from saklas.server.ws_models import WSSamplingParams, build_sampling_config


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
    session.model = MagicMock()
    session.model.config.model_type = "gemma2"
    session._device = "cpu"
    session._dtype = "torch.bfloat16"
    session._created_ts = 1_700_000_000

    session.config = GenerationConfig()

    session.profiles = {}
    session.probes = {}
    session.tree = MagicMock()
    session.tree.messages_for.return_value = []
    session.tree.active_node_id = "test-assistant"
    session.tree.get.return_value.mean_logprob = None
    session.tree.get.return_value.mean_surprise = None
    session.manifolds = {}
    session.is_base_model = False
    session.has_compatible_jlens.return_value = False
    session.live_lens_layers = None
    session.sae_info = None
    session.live_sae = False
    session.live_probe_scores = True
    session.scene_grammar = None
    session.joint_logprob_cache = {}
    session.lens_probe_names = []
    session.sae_probe_names = []
    session.token_probe_payload = {}

    monitor = MagicMock()
    monitor.probe_names = []
    monitor.attached_probes.return_value = {}
    session.monitor = monitor
    session.tokenizer = MagicMock()
    session._layers = []
    session.last_per_token_scores = None
    session.last_result = None
    session.last_per_token_scores = None
    session.last_result = None

    gen_state = MagicMock()
    gen_state.finish_reason = "stop"
    gen_state.emit_map = []
    session.generation_state = gen_state

    session.build_readings.return_value = {}
    session.lock = asyncio.Lock()

    # EventBus mock with subscribe/unsubscribe support.
    _event_subscribers = []

    def _subscribe(cb: Any) -> Any:
        _event_subscribers.append(cb)
        def _unsub() -> None:
            try:
                _event_subscribers.remove(cb)
            except ValueError:
                pass
        return _unsub

    def _emit(event: Any) -> None:
        for cb in list(_event_subscribers):
            try:
                cb(event)
            except Exception:
                pass

    events = MagicMock()
    events.subscribe = _subscribe
    events.emit = _emit
    session.events = events
    session._event_subscribers = _event_subscribers

    return session


@pytest.fixture
def session_and_client():
    from saklas.server import create_app
    session = _mock_session()
    app = create_app(session, default_steering=None)
    return session, TestClient(app)


# ---- sessions collection -------------------------------------------------


class TestSessions:
    def test_list(self, session_and_client: Any) -> None:
        _, client = session_and_client
        with patch("saklas.server.session_models.supports_thinking", return_value=False):
            resp = client.get("/saklas/v1/sessions")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["sessions"]) == 1
        s = data["sessions"][0]
        assert s["id"] == "default"
        assert s["model_id"] == "test/model"
        assert "config" in s
        assert s["config"]["temperature"] == 1.0

    def test_create_idempotent(self, session_and_client: Any) -> None:
        _, client = session_and_client
        with patch("saklas.server.session_models.supports_thinking", return_value=False):
            resp = client.post("/saklas/v1/sessions", json={})
        assert resp.status_code == 200
        assert resp.json()["id"] == "default"

    def test_create_rejects_unknown_fields(self, session_and_client: Any) -> None:
        """A native body-validation failure speaks the native envelope.

        Every ``/saklas/v1/*`` failure — typed ``SaklasError``, bare
        ``HTTPException``, body validation — renders as ``{"detail": "<str>"}``
        so a client never has to guess which of three shapes it got.
        """
        _, client = session_and_client
        resp = client.post("/saklas/v1/sessions", json={"legacy_id": "old"})
        assert resp.status_code == 400
        assert resp.json() == {
            "detail": "legacy_id: Extra inputs are not permitted",
        }

    def test_create_model_mismatch_logs_warning(self, session_and_client: Any, caplog: Any) -> None:
        _, client = session_and_client
        with patch("saklas.server.session_models.supports_thinking", return_value=False):
            resp = client.post("/saklas/v1/sessions", json={"model": "other/model"})
        assert resp.status_code == 200
        assert resp.json()["model_id"] == "test/model"

    def test_get_by_default(self, session_and_client: Any) -> None:
        _, client = session_and_client
        with patch("saklas.server.session_models.supports_thinking", return_value=False):
            resp = client.get("/saklas/v1/sessions/default")
        assert resp.status_code == 200
        data = resp.json()
        assert "aliases" not in data

    def test_safe_model_id_is_not_a_session_alias(
        self, session_and_client: Any,
    ) -> None:
        _, client = session_and_client
        with patch("saklas.server.session_models.supports_thinking", return_value=False):
            resp = client.get("/saklas/v1/sessions/test__model")
        assert resp.status_code == 404

    def test_get_not_found(self, session_and_client: Any) -> None:
        _, client = session_and_client
        resp = client.get("/saklas/v1/sessions/other")
        assert resp.status_code == 404

    def test_delete_is_noop(self, session_and_client: Any) -> None:
        _, client = session_and_client
        resp = client.delete("/saklas/v1/sessions/default")
        assert resp.status_code == 204

    def test_patch_updates_config(self, session_and_client: Any) -> None:
        session, client = session_and_client
        with patch("saklas.server.session_models.supports_thinking", return_value=False):
            resp = client.patch(
                "/saklas/v1/sessions/default",
                json={
                    "temperature": 0.3,
                    "system_prompt": "Be brief.",
                    "thinking": False,
                },
            )
        assert resp.status_code == 200
        assert session.config.temperature == 0.3
        assert session.config.system_prompt == "Be brief."
        assert session.config.thinking is False
        assert resp.json()["config"]["thinking"] is False

    def test_patch_top_k_distinguishes_omitted_from_explicit_null(
        self, session_and_client: Any,
    ) -> None:
        """The dashboard's blank top-k field must really disable the cutoff."""
        from dataclasses import replace

        session, client = session_and_client
        session.config = replace(session.config, top_k=40)
        with patch("saklas.server.session_models.supports_thinking", return_value=False):
            preserved = client.patch(
                "/saklas/v1/sessions/default", json={"temperature": 0.4},
            )
            cleared = client.patch(
                "/saklas/v1/sessions/default", json={"top_k": None},
            )
        assert preserved.status_code == 200
        assert preserved.json()["config"]["top_k"] == 40
        assert cleared.status_code == 200
        assert cleared.json()["config"]["top_k"] is None
        assert session.config.top_k is None

    def test_validate_steering_parses_and_dry_installs(
        self, session_and_client: Any,
    ) -> None:
        session, client = session_and_client
        resp = client.post(
            "/saklas/v1/sessions/default/steering/validate",
            json={"expression": "0.2 default/honest.deceptive@response"},
        )
        assert resp.status_code == 200
        assert resp.json() == {
            "valid": True,
            "expression": "0.2 default/honest.deceptive%honest@response",
            "error": None,
        }
        session.steering.assert_called_once()

    def test_validate_steering_returns_expected_errors_in_band(
        self, session_and_client: Any,
    ) -> None:
        session, client = session_and_client
        session.steering.side_effect = SaklasError(
            "No vector registered for ablation target 'definitely_missing'",
        )
        resp = client.post(
            "/saklas/v1/sessions/default/steering/validate",
            json={"expression": "0.5 !definitely_missing"},
        )
        assert resp.status_code == 200
        assert resp.json() == {
            "valid": False,
            "expression": "0.5 !definitely_missing",
            "error": "No vector registered for ablation target 'definitely_missing'",
        }

    def test_validate_steering_keeps_empty_as_explicit_unsteered(
        self, session_and_client: Any,
    ) -> None:
        session, client = session_and_client
        resp = client.post(
            "/saklas/v1/sessions/default/steering/validate",
            json={"expression": "   "},
        )
        assert resp.status_code == 200
        assert resp.json() == {"valid": True, "expression": "", "error": None}
        session.steering.assert_not_called()

    @staticmethod
    def _set_family(session: Any, model_type: str) -> None:
        """Pin the mock session's resolved model_type so the role-header
        registries (and thus role-support gating) see a real family."""
        session.model = MagicMock()
        session.model.config = MagicMock()
        session.model.config.text_config = None
        session.model.config.model_type = model_type

    def test_session_info_exposes_role_support(self, session_and_client: Any) -> None:
        """Per-message role boxes gate on these flags — keep them on the wire."""
        session, client = session_and_client
        self._set_family(session, "gemma2")
        with patch("saklas.server.session_models.supports_thinking", return_value=False):
            resp = client.get("/saklas/v1/sessions/default")
        assert resp.status_code == 200
        body = resp.json()
        assert body["role_substitution_supported"] is True
        assert body["user_role_supported"] is True
        # Gemma's standard assistant label is ``model`` (not ``assistant``);
        # the webui seeds the role boxes with these so they show live defaults.
        assert body["default_assistant_role"] == "model"
        assert body["default_user_role"] == "user"

    def test_session_info_exposes_scene_capabilities(
        self, session_and_client: Any,
    ) -> None:
        """The composer's seat toggle / thinking box / one-turn warning
        gate on these three flags — keep them on the wire.  A stub
        session (no real scene grammar) reads as scene mode off."""
        session, client = session_and_client
        with patch("saklas.server.session_models.supports_thinking", return_value=False):
            resp = client.get("/saklas/v1/sessions/default")
        assert resp.status_code == 200
        body = resp.json()
        assert body["scene_mode"] is False
        assert body["thinking_input_supported"] is False
        assert body["strips_history_thinking"] is False

        # A think-capable strip-family grammar flips all three.
        from saklas.core.scene import SeatWrapper, TurnGrammar

        session.scene_grammar = TurnGrammar(
            model_type="qwen3",
            prelude="",
            user=SeatWrapper("<t>", "\n", "<end>", "user"),
            assistant=SeatWrapper("<t>", "\n", "<end>", "assistant"),
            system=None,
            system_fold_sep=None,
            gen_extra="",
            think_open="<think>",
            think_close="</think>",
            strips_history_thinking=True,
        )
        with patch("saklas.server.session_models.supports_thinking", return_value=False):
            resp = client.get("/saklas/v1/sessions/default")
        body = resp.json()
        assert body["scene_mode"] is True
        assert body["thinking_input_supported"] is True
        assert body["strips_history_thinking"] is True

    def test_clear(self, session_and_client: Any) -> None:
        session, client = session_and_client
        resp = client.post("/saklas/v1/sessions/default/clear")
        assert resp.status_code == 204
        session.clear_history.assert_called_once()

    def test_rewind_empty(self, session_and_client: Any) -> None:
        session, client = session_and_client
        session.tree.messages_for.return_value = []
        resp = client.post("/saklas/v1/sessions/default/rewind")
        assert resp.status_code == 400


# ---- vectors -------------------------------------------------------------


class TestProfiles:
    def test_list_empty(self, session_and_client: Any) -> None:
        _, client = session_and_client
        resp = client.get("/saklas/v1/sessions/default/profiles")
        assert resp.status_code == 200
        assert resp.json()["profiles"] == []

    def test_get_not_found(self, session_and_client: Any) -> None:
        _, client = session_and_client
        resp = client.get("/saklas/v1/sessions/default/profiles/missing")
        assert resp.status_code == 404

    def test_delete_not_found(self, session_and_client: Any) -> None:
        _, client = session_and_client
        resp = client.delete("/saklas/v1/sessions/default/profiles/missing")
        assert resp.status_code == 404


# ---- probes --------------------------------------------------------------


class TestProbes:
    def test_list_empty(self, session_and_client: Any) -> None:
        _, client = session_and_client
        resp = client.get("/saklas/v1/sessions/default/probes")
        assert resp.status_code == 200
        assert resp.json()["probes"] == []

    def test_defaults(self, session_and_client: Any) -> None:
        _, client = session_and_client
        with patch(
            "saklas.server.probe_routes.load_default_manifolds",
            return_value={"emotion": ["happiness"]},
        ):
            resp = client.get("/saklas/v1/sessions/default/probes/defaults")
        assert resp.status_code == 200
        assert "emotion" in resp.json()["defaults"]

    def test_attach(self, session_and_client: Any) -> None:
        from types import SimpleNamespace
        import torch

        from saklas.core.capture import fold_directions_to_subspace
        from tests._whitener import isotropic_whitener

        session, client = session_and_client
        # Unified attach: body-carried selector → session.add_probe, 201 + info.
        means = {0: torch.zeros(2)}
        mani = fold_directions_to_subspace(
            "happy", {0: torch.tensor([1.0, 0.0])}, means,
            whitener=isotropic_whitener(means, 2),
        )
        session.add_probe.return_value = "happy"
        session.monitor.attached_probes.return_value = {
            "happy": SimpleNamespace(top_n=3, manifold=mani),
        }
        resp = client.post(
            "/saklas/v1/sessions/default/probes", json={"selector": "happy"},
        )
        assert resp.status_code == 201
        session.add_probe.assert_called_once_with("happy", as_name=None, top_n=3)
        assert resp.json()["name"] == "happy"

    def test_attach_empty_selector(self, session_and_client: Any) -> None:
        _, client = session_and_client
        resp = client.post(
            "/saklas/v1/sessions/default/probes", json={"selector": "  "},
        )
        assert resp.status_code == 400

    def test_deactivate_not_found(self, session_and_client: Any) -> None:
        _, client = session_and_client
        resp = client.delete("/saklas/v1/sessions/default/probes/missing")
        assert resp.status_code == 404


# ---- extract -------------------------------------------------------------


class TestExtract:
    def test_extract_json(self, session_and_client: Any) -> None:
        import torch
        from saklas.core.profile import Profile
        session, client = session_and_client
        profile = Profile({0: torch.zeros(4), 1: torch.ones(4)})
        session.extract.return_value = ("angry.calm", profile)
        resp = client.post(
            "/saklas/v1/sessions/default/extract",
            json={"concept": "angry", "baseline": "calm"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["canonical"] == "angry.calm"
        assert data["profile"]["layers"] == [0, 1]
        assert "on_progress" in session.extract.call_args.kwargs
        session.extract.assert_called_once()
        assert session.extract.call_args.args == ("angry", "calm")
        session.steer.assert_called_once_with("angry.calm", profile)

    def test_extract_defaults_kind_to_abstract(self, session_and_client: Any) -> None:
        """The elicitation framing is explicit on the wire, defaulted here."""
        import torch
        from saklas.core.profile import Profile
        session, client = session_and_client
        session.extract.return_value = ("angry.calm", Profile({0: torch.zeros(4)}))
        resp = client.post(
            "/saklas/v1/sessions/default/extract",
            json={"concept": "angry", "baseline": "calm"},
        )
        assert resp.status_code == 200
        assert session.extract.call_args.kwargs["kind"] == "abstract"
        assert session.extract.call_args.kwargs["custom_system"] is None

    def test_extract_threads_kind_and_custom_system(
        self, session_and_client: Any,
    ) -> None:
        """``kind`` / ``custom_system`` reach ``session.extract`` verbatim."""
        import torch
        from saklas.core.profile import Profile
        session, client = session_and_client
        session.extract.return_value = ("january.july", Profile({0: torch.zeros(4)}))
        resp = client.post(
            "/saklas/v1/sessions/default/extract",
            json={
                "concept": "january", "baseline": "july",
                "kind": "custom",
                "custom_system": "You are the month of {c}.",
            },
        )
        assert resp.status_code == 200
        assert session.extract.call_args.kwargs["kind"] == "custom"
        assert (
            session.extract.call_args.kwargs["custom_system"]
            == "You are the month of {c}."
        )

    def test_extract_rejects_custom_kind_without_system(
        self, session_and_client: Any,
    ) -> None:
        """Same rejection ``POST /manifolds/generate`` applies — one contract."""
        session, client = session_and_client
        resp = client.post(
            "/saklas/v1/sessions/default/extract",
            json={"concept": "january", "baseline": "july", "kind": "custom"},
        )
        assert resp.status_code == 400
        assert "custom_system" in resp.text
        session.extract.assert_not_called()

    def test_extract_rejects_unknown_kind(self, session_and_client: Any) -> None:
        """``kind`` is a closed set at the pydantic layer."""
        session, client = session_and_client
        resp = client.post(
            "/saklas/v1/sessions/default/extract",
            json={"concept": "a", "baseline": "b", "kind": "nope"},
        )
        assert resp.status_code == 400
        session.extract.assert_not_called()

    def test_extract_json_still_reports_progress_lines(
        self, session_and_client: Any,
    ) -> None:
        """The JSON branch returns what the SSE branch streams as frames."""
        import torch
        from saklas.core.profile import Profile
        session, client = session_and_client
        profile = Profile({0: torch.zeros(4)})

        def _extract(*_args: Any, on_progress: Any = None, **_kwargs: Any) -> Any:
            if on_progress is not None:
                on_progress("capturing")
                on_progress("fitting")
            return ("angry.calm", profile)

        session.extract.side_effect = _extract
        resp = client.post(
            "/saklas/v1/sessions/default/extract",
            json={"concept": "angry", "baseline": "calm"},
        )
        assert resp.status_code == 200
        assert resp.json()["progress"] == ["capturing", "fitting"]

    def test_extract_json_maps_a_value_error_to_400(
        self, session_and_client: Any,
    ) -> None:
        """Extract shares manifold ``fit``'s typed error policy.

        Both drive the one extraction pipeline; the JSON branch used to map
        nothing, so an authoring-grade ``ValueError`` surfaced as a 500.
        """
        session, client = session_and_client
        session.extract.side_effect = ValueError("nodes are not poised")
        resp = client.post(
            "/saklas/v1/sessions/default/extract",
            json={"concept": "angry", "baseline": "calm"},
        )
        assert resp.status_code == 400
        assert "poised" in resp.text

    def test_extract_json_maps_a_concurrent_run_to_409(
        self, session_and_client: Any,
    ) -> None:
        from saklas.core.session import ConcurrentExtractionError

        session, client = session_and_client
        session.extract.side_effect = ConcurrentExtractionError("busy")
        resp = client.post(
            "/saklas/v1/sessions/default/extract",
            json={"concept": "angry", "baseline": "calm"},
        )
        assert resp.status_code == 409

    def test_extract_sse_surfaces_a_typed_safe_message(
        self, session_and_client: Any,
    ) -> None:
        """A typed failure keeps its message instead of the generic frame."""
        session, client = session_and_client
        session.extract.side_effect = ValueError("poisedness check failed")
        with client.stream(
            "POST",
            "/saklas/v1/sessions/default/extract",
            json={"concept": "angry", "baseline": "calm"},
            headers={"accept": "text/event-stream"},
        ) as resp:
            text = b"".join(resp.iter_bytes()).decode("utf-8")
        assert "event: error" in text
        assert "poisedness check failed" in text
        assert '"code": "PoisednessError"' in text

    def test_extract_json_registers_returned_variant_and_namespace(
        self, session_and_client: Any,
    ) -> None:
        import torch
        from saklas.core.profile import Profile
        session, client = session_and_client
        profile = Profile({0: torch.ones(4)})
        session.extract.return_value = ("honest.deceptive:role-pirate", profile)

        resp = client.post(
            "/saklas/v1/sessions/default/extract",
            json={
                "concept": "honest",
                "baseline": "deceptive",
                "role": "pirate",
                "namespace": "alice",
            },
        )

        assert resp.status_code == 200
        assert resp.json()["canonical"] == "alice/honest.deceptive:role-pirate"
        session.steer.assert_called_once_with(
            "alice/honest.deceptive:role-pirate", profile,
        )

    def test_extract_sse_streams_progress_live(self, session_and_client: Any) -> None:
        """SSE branch must yield each ``on_progress`` message as its own
        event rather than buffering them all until extraction returns.

        Regression: an earlier shape collected messages into a list and
        only yielded them after ``session.extract`` had completed, so
        the client received every progress event in one tick right
        before ``done`` — the webui's progress toast had no time to
        render them.  The fix routes progress through an
        ``asyncio.Queue`` driven from the worker thread.
        """
        import torch
        from saklas.core.profile import Profile
        session, client = session_and_client
        profile = Profile({0: torch.ones(4)})

        def _extract(source: Any, baseline: Any = None, *, on_progress: Any = None, **_kwargs: Any) -> Any:
            assert on_progress is not None
            on_progress("Generating 9 scenarios for 'angry.calm'...")
            on_progress("Generating contrastive pairs across 9 domains...")
            on_progress("Extracting difference-of-means profile (45 pairs)...")
            return "angry.calm", profile

        session.extract.side_effect = _extract

        # ``TestClient`` consumes the whole response before returning,
        # so we can't observe arrival timing here — but we *can* confirm
        # each progress message lands as its own ``event: progress``
        # frame ordered before ``event: done``.  If the old buffer-then-
        # flush shape regressed, the events would still arrive in order
        # but this assertion still gates the structural fix.
        with client.stream(
            "POST",
            "/saklas/v1/sessions/default/extract",
            json={"concept": "angry", "baseline": "calm"},
            headers={"Accept": "text/event-stream"},
        ) as resp:
            assert resp.status_code == 200
            body = b"".join(resp.iter_bytes()).decode()

        # Split SSE frames on the blank-line terminator.
        frames = [f for f in body.split("\n\n") if f.strip()]
        events = [
            f.split("\n")[0].removeprefix("event: ") for f in frames
        ]
        assert events.count("progress") == 3
        assert events[-1] == "done"
        assert "Generating 9 scenarios" in frames[0]
        assert "Extracting difference-of-means" in frames[2]

    def test_extract_sse_registers_returned_variant(
        self, session_and_client: Any,
    ) -> None:
        import torch
        from saklas.core.profile import Profile
        session, client = session_and_client
        profile = Profile({0: torch.ones(4)})

        def _extract(
            source: Any,
            baseline: Any = None,
            *,
            on_progress: Any = None,
            **_kwargs: Any,
        ) -> Any:
            return "honest.deceptive:role-pirate", profile

        session.extract.side_effect = _extract
        with client.stream(
            "POST",
            "/saklas/v1/sessions/default/extract",
            json={
                "concept": "honest",
                "baseline": "deceptive",
                "role": "pirate",
            },
            headers={"Accept": "text/event-stream"},
        ) as resp:
            assert resp.status_code == 200
            body = b"".join(resp.iter_bytes()).decode()

        assert '"canonical": "honest.deceptive:role-pirate"' in body
        session.steer.assert_called_once_with(
            "honest.deceptive:role-pirate", profile,
        )

    def test_extract_rejects_legacy_polymorphic_shape(
        self, session_and_client: Any,
    ) -> None:
        session, client = session_and_client
        resp = client.post(
            "/saklas/v1/sessions/default/extract",
            json={
                "concept": "custom",
                "source": {"positive": ["yes"], "negative": ["no"]},
                "register": False,
            },
        )
        assert resp.status_code == 400
        session.extract.assert_not_called()

    def test_raw_profile_load_route_is_removed(
        self, session_and_client: Any,
    ) -> None:
        session, client = session_and_client
        resp = client.post(
            "/saklas/v1/sessions/default/profiles",
            json={"name": "custom", "source_path": "/tmp/profile.pt"},
        )
        assert resp.status_code == 405
        session.load_profile.assert_not_called()


# ---- WebSocket token+probe co-stream ------------------------------------


def test_ws_sampling_can_disable_final_probe_readings() -> None:
    sc = build_sampling_config(WSSamplingParams(return_probe_readings=False))
    assert sc is not None
    assert sc.return_probe_readings is False


class TestWebSocket:
    def _attach_generate(self, session: Any, tokens: Any) -> None:
        """Install a fake ``session.generate`` that drives ``on_token``."""
        def _gen(input: Any, *, steering: Any = None, sampling: Any = None,
                 stateless: Any = False, raw: Any = False, thinking: Any = None,
                 on_token: Any = None, parent_node_id: Any = None, n: Any = 1,
                 append_same_role: Any = True) -> Any:
            for i, tok in enumerate(tokens):
                on_token(tok, False, 1000 + i, None, None)  # pyright: ignore[reportOptionalCall]
                time.sleep(0.001)
            result = GenerationResult(
                text="".join(tokens), tokens=list(range(1000, 1000 + len(tokens))),
                token_count=len(tokens), tok_per_sec=50.0, elapsed=0.05,
                finish_reason="stop",
            )
            session.last_result = result
            session.last_result = result
            per_token = {
                "happy": [0.1 * (i + 1) for i in range(len(tokens))],
            }
            session.last_per_token_scores = per_token
            session.last_per_token_scores = per_token
            return RunSet([result])

        session.generate.side_effect = _gen

    def test_generate_happy_path(self, session_and_client: Any) -> None:
        session, client = session_and_client
        self._attach_generate(session, ["Hello", " ", "world"])

        with client.websocket_connect("/saklas/v1/sessions/default/stream") as ws:
            ws.send_json({"type": "generate", "input": "hi"})
            msg = ws.receive_json()
            assert msg["type"] == "started"
            tokens = []
            while True:
                msg = ws.receive_json()
                if msg["type"] == "token":
                    tokens.append(msg["text"])
                elif msg["type"] == "done":
                    done = msg
                    break
            assert tokens == ["Hello", " ", "world"]
            assert done["result"]["finish_reason"] == "stop"
            assert "per_token_probes" not in done["result"]

    def test_generate_worker_typed_error_uses_safe_message(
        self, session_and_client: Any,
    ) -> None:
        session, client = session_and_client
        session.generate.side_effect = SaklasError("safe generation detail")

        with client.websocket_connect("/saklas/v1/sessions/default/stream") as ws:
            ws.send_json({"type": "generate", "input": "hi"})
            assert ws.receive_json()["type"] == "started"
            error = ws.receive_json()

        assert error["type"] == "error"
        assert error["message"] == "safe generation detail"
        assert error["status"] == 500

    def test_generate_worker_untyped_error_is_scrubbed(
        self, session_and_client: Any,
    ) -> None:
        session, client = session_and_client
        session.generate.side_effect = RuntimeError("private backend detail")

        with client.websocket_connect("/saklas/v1/sessions/default/stream") as ws:
            ws.send_json({"type": "generate", "input": "hi"})
            assert ws.receive_json()["type"] == "started"
            error = ws.receive_json()

        assert error["type"] == "error"
        assert error["message"] == (
            "Generation failed. Check the server log for details."
        )
        assert "private backend detail" not in error["message"]
        assert error["status"] == 500

    def test_generate_rejects_unknown_fields(self, session_and_client: Any) -> None:
        _, client = session_and_client
        with client.websocket_connect("/saklas/v1/sessions/default/stream") as ws:
            ws.send_json({"type": "generate", "input": "hi", "legacy": True})
            msg = ws.receive_json()
        assert msg["type"] == "error"
        assert msg["code"] == "ValidationError"
        assert "Extra inputs are not permitted" in msg["message"]

    def test_generate_rejects_unknown_nested_message_fields(
        self, session_and_client: Any,
    ) -> None:
        _, client = session_and_client
        with client.websocket_connect("/saklas/v1/sessions/default/stream") as ws:
            ws.send_json({
                "type": "generate",
                "input": [{"role": "user", "content": "hi", "name": "old"}],
            })
            msg = ws.receive_json()
        assert msg["type"] == "error"
        assert msg["code"] == "ValidationError"
        assert "Extra inputs are not permitted" in msg["message"]

    def test_generate_accepts_labelled_input_messages(
        self, session_and_client: Any,
    ) -> None:
        """The dashboard's auto-regen shadow replays a conversation as an
        explicit message list carrying each turn's cast ``label``.  The wire
        message is the same shape a loom-derived message dict is, so the
        label reaches ``_prepare_input`` verbatim — re-rendering the shadow
        under default labels would be a different prompt than the one being
        shadowed."""
        session, client = session_and_client
        self._attach_generate(session, ["ok"])
        with client.websocket_connect("/saklas/v1/sessions/default/stream") as ws:
            ws.send_json({
                "type": "generate",
                "input": [
                    {"role": "user", "content": "hi", "label": "narrator"},
                    {"role": "assistant", "content": "hello", "label": "deer"},
                    {"role": "user", "content": "again", "label": None},
                ],
                "steering": "",
                "stateless": True,
            })
            while (msg := ws.receive_json())["type"] != "done":
                assert msg["type"] != "error", msg

        sent = session.generate.call_args.args[0]
        assert sent == [
            {"role": "user", "content": "hi", "label": "narrator"},
            {"role": "assistant", "content": "hello", "label": "deer"},
            {"role": "user", "content": "again", "label": None},
        ]

    def test_stale_n_way_token_callback_stays_on_original_queue(
        self, session_and_client: Any,
    ) -> None:
        """A late sibling-0 callback must not leak into sibling 1's stream."""
        session, client = session_and_client
        callbacks: list[Any] = []

        def _gen(input: Any, *, steering: Any = None, sampling: Any = None,
                 stateless: Any = False, raw: Any = False, thinking: Any = None,
                 on_token: Any = None, parent_node_id: Any = None, n: Any = 1,
                 append_same_role: Any = True) -> Any:
            callbacks.append(on_token)
            idx = len(callbacks) - 1
            if idx == 1:
                callbacks[0]("late-first", False, 1999, None, None)
                on_token("second", False, 2000, None, None)
                text = "second"
                tokens = [2000]
            else:
                text = "first"
                tokens = []
            result = GenerationResult(
                text=text, tokens=tokens, token_count=len(tokens),
                tok_per_sec=50.0, elapsed=0.01, finish_reason="stop",
            )
            session.last_result = result
            session.last_result = result
            session.last_per_token_scores = {}
            session.last_per_token_scores = {}
            return RunSet([result])

        session.generate.side_effect = _gen

        with client.websocket_connect("/saklas/v1/sessions/default/stream") as ws:
            ws.send_json({"type": "generate", "input": "hi", "n": 2})
            tokens: list[str] = []
            done_count = 0
            started_count = 0
            while done_count < 2:
                msg = ws.receive_json()
                if msg["type"] == "started":
                    started_count += 1
                elif msg["type"] == "token":
                    tokens.append(msg["text"])
                elif msg["type"] == "done":
                    done_count += 1

        assert started_count == 2
        assert tokens == ["second"]

    def test_unknown_message_type(self, session_and_client: Any) -> None:
        _, client = session_and_client
        with client.websocket_connect("/saklas/v1/sessions/default/stream") as ws:
            ws.send_json({"type": "frobnicate"})
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "unknown message type" in msg["message"]

    def test_generate_rejects_nonpositive_n(self, session_and_client: Any) -> None:
        """``n`` is a declared schema bound now (``Field(ge=1)``), so the
        rejection is a ``ValidationError`` naming the field rather than a
        hand-rolled handler check."""
        session, client = session_and_client
        with client.websocket_connect("/saklas/v1/sessions/default/stream") as ws:
            ws.send_json({"type": "generate", "input": "hi", "n": 0})
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert msg["status"] == 400
            assert msg["code"] == "ValidationError"
            assert msg["message"].startswith("n: ")
        session.generate.assert_not_called()

    def test_submit_rejects_nonpositive_n(self, session_and_client: Any) -> None:
        """``submit`` carries the same bound — ``_normalize_submit`` forwards
        ``n`` into a ``WSGenerateMessage`` construction that sits outside the
        handler's error-frame guard, so the reader has to catch it first."""
        session, client = session_and_client
        with client.websocket_connect("/saklas/v1/sessions/default/stream") as ws:
            ws.send_json({
                "type": "submit", "text": "hi",
                "authored_role": "user", "generated_role": "assistant",
                "n": 0,
            })
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert msg["status"] == 400
            assert msg["code"] == "ValidationError"
            assert msg["message"].startswith("n: ")
        session.generate.assert_not_called()

    def test_multi_turn_no_recv_race(self, session_and_client: Any) -> None:
        """Three back-to-back generate turns on the same WS.

        Regression for the "cannot call recv while another coroutine is
        already waiting for the next message" RuntimeError that fired
        when both the outer dispatch loop and the inner generation
        handler called ``websocket.receive_json()``.  The fix routes
        every incoming frame through a single perpetual reader task +
        shared queue.  This test exercises the inter-turn boundary
        repeatedly so any regression of that pattern surfaces.
        """
        session, client = session_and_client
        self._attach_generate(session, ["a", "b"])

        with client.websocket_connect("/saklas/v1/sessions/default/stream") as ws:
            for _ in range(3):
                ws.send_json({"type": "generate", "input": "hi"})
                started = ws.receive_json()
                assert started["type"] == "started"
                while True:
                    msg = ws.receive_json()
                    if msg["type"] == "done":
                        break
                    assert msg["type"] == "token"

    def test_mid_generation_generate_frame_runs_after_current_turn(self, session_and_client: Any) -> None:
        """A premature second generate frame is deferred, not re-read in a spin loop."""
        session, client = session_and_client
        calls: list[str] = []

        def _gen(input: Any, *, steering: Any = None, sampling: Any = None,
                 stateless: Any = False, raw: Any = False, thinking: Any = None,
                 on_token: Any = None, parent_node_id: Any = None, n: Any = 1,
                 append_same_role: Any = True) -> Any:
            calls.append(str(input))
            time.sleep(0.02 if input == "one" else 0.001)
            on_token(str(input), False, 1000 + len(calls), None, None)  # pyright: ignore[reportOptionalCall]
            result = GenerationResult(
                text=str(input), tokens=[1000 + len(calls)],
                token_count=1, tok_per_sec=50.0, elapsed=0.02,
                finish_reason="stop",
            )
            session.last_result = result
            session.last_result = result
            session.last_per_token_scores = {}
            session.last_per_token_scores = {}
            return RunSet([result])

        session.generate.side_effect = _gen

        with client.websocket_connect("/saklas/v1/sessions/default/stream") as ws:
            ws.send_json({"type": "generate", "input": "one"})
            ws.send_json({"type": "generate", "input": "two"})
            done = []
            while len(done) < 2:
                msg = ws.receive_json()
                if msg["type"] == "done":
                    done.append(msg["result"]["text"])

        assert done == ["one", "two"]
        assert calls == ["one", "two"]

    def test_idle_stop_is_noop(self, session_and_client: Any) -> None:
        """A ``{type: "stop"}`` outside any generation closes cleanly."""
        _, client = session_and_client
        with client.websocket_connect("/saklas/v1/sessions/default/stream") as ws:
            # Stop while idle — server should ignore and stay open.
            ws.send_json({"type": "stop"})
            ws.send_json({"type": "frobnicate"})
            msg = ws.receive_json()
            assert msg["type"] == "error"

    def test_session_mismatch_closes(self, session_and_client: Any) -> None:
        _, client = session_and_client
        with pytest.raises(Exception):
            with client.websocket_connect("/saklas/v1/sessions/other/stream") as ws:
                ws.receive_json()

    def test_ws_requires_bearer_when_api_key_set(self):
        from saklas.server import create_app
        session = _mock_session()
        app = create_app(session, default_steering=None, api_key="s3cret")
        client = TestClient(app)
        # No Authorization header -> close(1008) before accept.
        with pytest.raises(Exception):
            with client.websocket_connect("/saklas/v1/sessions/default/stream") as ws:
                ws.receive_json()
        # Wrong token -> same.
        with pytest.raises(Exception):
            with client.websocket_connect(
                "/saklas/v1/sessions/default/stream",
                headers={"Authorization": "Bearer wrong"},
            ) as ws:
                ws.receive_json()
        # Correct token -> handshake succeeds.
        with client.websocket_connect(
            "/saklas/v1/sessions/default/stream",
            headers={"Authorization": "Bearer s3cret"},
        ) as ws:
            ws.send_json({"type": "frobnicate"})
            msg = ws.receive_json()
            assert msg["type"] == "error"
        # Browser clients cannot set Authorization on the WS constructor;
        # the dashboard sends the bearer as ?token=...
        with client.websocket_connect(
            "/saklas/v1/sessions/default/stream?token=s3cret",
        ) as ws:
            ws.send_json({"type": "frobnicate"})
            msg = ws.receive_json()
            assert msg["type"] == "error"

    def test_bad_steering_does_not_kill_connection(self, session_and_client: Any, monkeypatch: Any) -> None:
        """Regression: a bad steering expression on a generate frame used
        to escape request-steering parsing and bubble out to the outer
        reader loop's ``except Exception``, which closed the WS with code 1011.

        FastAPI's ``@app.exception_handler(SaklasError)`` doesn't apply
        to WebSocket routes, so the handler has to convert the error in-
        band itself. Contract: the server emits a ``{type: "error"}``
        frame and stays open for a follow-up generate.
        """
        from saklas.io.selectors import AmbiguousSelectorError
        import saklas.core.steering_expr as _sx

        session, client = session_and_client
        self._attach_generate(session, ["ok"])

        real_parse = _sx.parse_expr

        def _fake_parse(text: Any, *, namespace: Any = None) -> Any:
            if text.strip() == "0.5 wolf":
                raise AmbiguousSelectorError(
                    "ambiguous pole 'wolf': matches alice/wolf, default/deer.wolf"
                )
            return real_parse(text, namespace=namespace)

        monkeypatch.setattr(_sx, "parse_expr", _fake_parse)

        with client.websocket_connect("/saklas/v1/sessions/default/stream") as ws:
            ws.send_json({
                "type": "generate", "input": "hi", "steering": "0.5 wolf",
            })
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "ambiguous pole 'wolf'" in msg["message"]
            assert msg["code"] == "AmbiguousSelectorError"
            assert msg["status"] == 400

            # Connection still alive — follow-up turn succeeds.
            ws.send_json({"type": "generate", "input": "hi"})
            started = ws.receive_json()
            assert started["type"] == "started"
            tokens = []
            while True:
                m = ws.receive_json()
                if m["type"] == "token":
                    tokens.append(m["text"])
                elif m["type"] == "done":
                    break
            assert tokens == ["ok"]


# ---- Deleted surfaces ------------------------------------------------------


class TestDeletedTraitsStream:
    """The traits SSE stream is gone — the WS ``measurements`` envelope and the
    OpenAI/Ollama ``x-saklas-probe-readings`` extension are the live per-token
    delivery channels."""

    def test_traits_stream_route_is_not_registered(
        self, session_and_client: Any,
    ) -> None:
        _, client = session_and_client
        resp = client.get("/saklas/v1/sessions/default/traits/stream")
        assert resp.status_code == 404

    def test_session_has_no_trait_queue_plumbing(self) -> None:
        from saklas.core.session import SaklasSession

        for attr in (
            "register_trait_queue", "unregister_trait_queue", "_trait_queues",
        ):
            assert not hasattr(SaklasSession, attr)


# ---- score_single_token (monitor) ----------------------------------------


class TestScoreSingleToken:
    def test_returns_scores_without_accumulation(self):
        import torch
        from saklas.core.monitor import Monitor
        from saklas.core.results import ProbeReading
        from saklas.core.capture import fold_directions_to_subspace

        from tests._whitener import isotropic_whitener
        dim = 16
        probe_vec = torch.randn(dim)
        means = {0: torch.zeros(dim)}
        whit = isotropic_whitener([0], dim)
        # Mahalanobis is mandatory: covering whitener required to attach + score.
        m = fold_directions_to_subspace(
            "test_probe", {0: probe_vec}, means, whitener=whit,
        )
        monitor = Monitor({"test_probe": m}, whitener=whit)

        hidden = {0: torch.randn(dim)}
        scores = monitor.score_single_token(hidden)

        assert "test_probe" in scores
        # Read is the full per-probe ProbeReading (coords axis-0 the scalar).
        assert isinstance(scores["test_probe"], ProbeReading)
        assert isinstance(scores["test_probe"].coords[0], float)
        # History should NOT have been updated.
        assert len(monitor.history["test_probe"]) == 0

    def test_consistent_with_measure_from_hidden(self):
        import torch
        from saklas.core.monitor import Monitor
        from saklas.core.capture import fold_directions_to_subspace

        from tests._whitener import isotropic_whitener
        dim = 16
        means = {0: torch.zeros(dim), 1: torch.zeros(dim)}
        whit = isotropic_whitener([0, 1], dim)
        m = fold_directions_to_subspace(
            "p1", {0: torch.randn(dim), 1: torch.randn(dim)}, means,
            whitener=whit,
        )
        monitor = Monitor({"p1": m}, whitener=whit)

        hidden = {0: torch.randn(dim), 1: torch.randn(dim)}
        single = monitor.score_single_token(hidden)
        no_acc = monitor.measure_from_hidden(hidden, accumulate=False)

        assert single["p1"].coords[0] == pytest.approx(no_acc["p1"].coords[0])

# ---- manifold routes ----------------------------------------------------


def _box1d_payload(name: str = "mood") -> dict[str, Any]:
    return {
        "namespace": "local",
        "name": name,
        "description": "a mood axis",
        "domain": {
            "type": "box",
            "axes": [{"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}],
        },
        "nodes": [
            {"label": "calm", "coords": [0.0],
             "statements": ["I am calm.", "Steady."]},
            {"label": "mid", "coords": [0.5],
             "statements": ["An ordinary moment.", "Nothing notable."]},
            {"label": "afraid", "coords": [1.0],
             "statements": ["I am afraid.", "Shaking."]},
        ],
    }


class TestManifoldRoutes:
    def test_create_list_get(self, session_and_client: Any, tmp_path: Any, monkeypatch: Any) -> None:
        monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
        _session, client = session_and_client

        resp = client.post("/saklas/v1/manifolds", json=_box1d_payload())
        assert resp.status_code == 201
        body = resp.json()
        assert body["name"] == "mood"
        assert body["intrinsic_dim"] == 1
        assert body["min_nodes"] == 3
        assert body["fitted_for_session"] is False
        assert "advisories" in body

        listed = client.get("/saklas/v1/manifolds").json()["manifolds"]
        assert [m["name"] for m in listed] == ["mood"]
        # An authored folder carries a concrete geometry on the wire so the
        # rack family split can route it without a fit.
        assert listed[0]["resolved_fit_mode"] == "authored"

        detail = client.get("/saklas/v1/manifolds/local/mood").json()
        labels = [n["label"] for n in detail["nodes"]]
        assert labels == ["calm", "mid", "afraid"]
        assert detail["nodes"][0]["statements"] == ["I am calm.", "Steady."]

    def test_create_conflict(self, session_and_client: Any, tmp_path: Any, monkeypatch: Any) -> None:
        monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
        _session, client = session_and_client
        assert client.post("/saklas/v1/manifolds",
                           json=_box1d_payload()).status_code == 201
        assert client.post("/saklas/v1/manifolds",
                           json=_box1d_payload()).status_code == 409

    def test_create_manifold_from_template(
        self, session_and_client: Any, tmp_path: Any, monkeypatch: Any,
    ) -> None:
        monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
        _session, client = session_and_client
        template_payload = {
            "name": "weekday",
            "slot": "[DAY]",
            "values": ["Monday", "Tuesday", "Wednesday"],
            "contexts": [
                {
                    "turns": [{"role": "user", "content": "what day is it?"}],
                    "assistant": "today is [DAY]",
                },
                {
                    "turns": [{"role": "user", "content": "which day?"}],
                    "assistant": "it's [DAY]",
                },
            ],
        }
        assert client.post(
            "/saklas/v1/templates", json=template_payload,
        ).status_code == 201
        payload = {
            "name": "calendar",
            "fit_mode": "auto",
            "template_ref": "local/weekday",
        }
        resp = client.post("/saklas/v1/manifolds/from-template", json=payload)
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["name"] == "calendar"
        assert body["fit_mode"] == "auto"
        manifest = json.loads(
            (tmp_path / "manifolds/local/calendar/manifold.json").read_text()
        )
        assert manifest["template_ref"] == "local/weekday"
        assert body["node_labels"] == ["monday", "tuesday", "wednesday"]
        # Unfitted ``auto`` folder: geometry unresolved on the wire (null), so
        # the rack client shows it in *both* family drawers.  Regression: an
        # auto manifold used to match neither subspace nor manifold family and
        # vanished from every drawer.
        assert body["resolved_fit_mode"] is None
        weekday_row = next(
            m for m in client.get("/saklas/v1/manifolds").json()["manifolds"]
            if m["name"] == "calendar"
        )
        assert weekday_row["resolved_fit_mode"] is None
        assert weekday_row["domain_label"] == "discover-auto"

        detail = client.get("/saklas/v1/manifolds/local/calendar").json()
        monday = next(n for n in detail["nodes"] if n["label"] == "monday")
        assert monday["statements"] == ["today is Monday", "it's Monday"]

        conflict = client.post("/saklas/v1/manifolds/from-template", json=payload)
        assert conflict.status_code == 409
        forced = client.post(
            "/saklas/v1/manifolds/from-template",
            json={**payload, "force": True},
        )
        assert forced.status_code == 201, forced.text

    def test_create_manifold_from_template_missing_template(
        self, session_and_client: Any, tmp_path: Any, monkeypatch: Any,
    ) -> None:
        monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
        _session, client = session_and_client
        resp = client.post("/saklas/v1/manifolds/from-template", json={
            "name": "bad",
            "template_ref": "local/missing",
        })
        assert resp.status_code == 404

    def test_create_manifold_from_template_ambiguous_template(
        self, session_and_client: Any, tmp_path: Any, monkeypatch: Any,
    ) -> None:
        monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
        _session, client = session_and_client
        template = {
            "name": "weekday",
            "slot": "[DAY]",
            "values": ["Monday", "Tuesday"],
            "contexts": [{
                "turns": [{"role": "user", "content": "which day?"}],
                "assistant": "today is [DAY]",
            }],
        }
        assert client.post(
            "/saklas/v1/templates", json={**template, "namespace": "local"},
        ).status_code == 201
        assert client.post(
            "/saklas/v1/templates", json={**template, "namespace": "other"},
        ).status_code == 201
        resp = client.post("/saklas/v1/manifolds/from-template", json={
            "name": "calendar",
            "template_ref": "weekday",
        })
        assert resp.status_code == 409

    def test_create_too_few_nodes(self, session_and_client: Any, tmp_path: Any,
                                  monkeypatch: Any) -> None:
        monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
        _session, client = session_and_client
        payload = _box1d_payload()
        payload["nodes"] = payload["nodes"][:2]
        assert client.post("/saklas/v1/manifolds",
                           json=payload).status_code == 400

    def test_patch_description(self, session_and_client: Any, tmp_path: Any,
                               monkeypatch: Any) -> None:
        monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
        _session, client = session_and_client
        client.post("/saklas/v1/manifolds", json=_box1d_payload())
        resp = client.patch("/saklas/v1/manifolds/local/mood",
                             json={"description": "edited"})
        assert resp.status_code == 200
        assert resp.json()["description"] == "edited"

    def test_delete(self, session_and_client: Any, tmp_path: Any, monkeypatch: Any) -> None:
        monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
        _session, client = session_and_client
        client.post("/saklas/v1/manifolds", json=_box1d_payload())
        assert client.delete(
            "/saklas/v1/manifolds/local/mood").status_code == 200
        assert client.get(
            "/saklas/v1/manifolds/local/mood").status_code == 404

    def test_get_missing(self, session_and_client: Any, tmp_path: Any, monkeypatch: Any) -> None:
        monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
        _session, client = session_and_client
        assert client.get(
            "/saklas/v1/manifolds/local/ghost").status_code == 404

    def test_authored_node_label_resolves_without_restart(
        self, session_and_client: Any, tmp_path: Any, monkeypatch: Any,
    ) -> None:
        """A dashboard-authored manifold must steer in the same serve process.

        ``saklas serve`` is long-lived, so a resolver index warmed before the
        POST would otherwise hide the new node labels until restart — the
        one-command-one-process CLI shape hides this entirely.
        """
        monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
        _session, client = session_and_client
        from saklas.io import selectors

        selectors.invalidate()
        assert selectors.resolve_manifold_label("afraid") is None  # warms it

        assert client.post(
            "/saklas/v1/manifolds", json=_box1d_payload(),
        ).status_code == 201

        hit = selectors.resolve_manifold_label("afraid")
        assert hit is not None and hit.manifold_key == "local/mood"

        assert client.delete("/saklas/v1/manifolds/local/mood").status_code == 200
        assert selectors.resolve_manifold_label("afraid") is None

    def test_delete_refuses_when_busy(self, session_and_client: Any, tmp_path: Any,
                                      monkeypatch: Any) -> None:
        # A fit thread holding the engine gen-lock must block a delete —
        # removing nodes/ mid-fit would corrupt the read.
        monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
        session, client = session_and_client
        client.post("/saklas/v1/manifolds", json=_box1d_payload())
        session.gen_lock.acquire.return_value = False
        assert client.delete(
            "/saklas/v1/manifolds/local/mood").status_code == 409

    def test_fit_json(self, session_and_client: Any, tmp_path: Any, monkeypatch: Any) -> None:
        monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
        session, client = session_and_client
        client.post("/saklas/v1/manifolds", json=_box1d_payload())

        from unittest.mock import MagicMock as _MM
        session.fit.return_value = _MM(
            layers={0: 1, 1: 1, 2: 1}, feature_space="raw",
        )
        resp = client.post(
            "/saklas/v1/manifolds/local/mood/fit", json={"layers": [1, 2]},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["done"] is True
        assert body["layers_fitted"] == 3
        assert body["feature_space"] == "raw"
        assert session.fit.call_args.kwargs["layers"] == [1, 2]

    def test_fit_authoring_conflict_is_a_retryable_409(
        self, session_and_client: Any, tmp_path: Any, monkeypatch: Any,
    ) -> None:
        """Re-authoring under an in-flight fit is a conflict, not a 500.

        ``ManifoldAuthoringChangedError`` is the same conflict
        ``ConcurrentExtractionError`` models, reached from the other side.
        """
        from saklas.core.extraction import ManifoldAuthoringChangedError

        monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
        session, client = session_and_client
        client.post("/saklas/v1/manifolds", json=_box1d_payload())
        session.fit.side_effect = ManifoldAuthoringChangedError(
            "manifold authoring changed during fit"
        )
        resp = client.post("/saklas/v1/manifolds/local/mood/fit", json={})
        assert resp.status_code == 409

    def test_fit_sse_authoring_conflict_is_a_typed_frame(
        self, session_and_client: Any, tmp_path: Any, monkeypatch: Any,
    ) -> None:
        from saklas.core.extraction import ManifoldAuthoringChangedError

        monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
        session, client = session_and_client
        client.post("/saklas/v1/manifolds", json=_box1d_payload())
        session.fit.side_effect = ManifoldAuthoringChangedError(
            "manifold authoring changed during fit"
        )
        with client.stream(
            "POST",
            "/saklas/v1/manifolds/local/mood/fit",
            json={},
            headers={"accept": "text/event-stream"},
        ) as resp:
            text = b"".join(resp.iter_bytes()).decode("utf-8")
        assert "event: error" in text
        assert '"code": "Conflict"' in text
        assert "authoring changed during fit" in text


# ---- templates (standalone artifact + scorer) ----------------------------

_TMPL_PAYLOAD = {
    "namespace": "local",
    "name": "weekday",
    "slot": "[DAY]",
    "values": ["Monday", "Tuesday", "Wednesday"],
    "contexts": [
        {"turns": [{"role": "user", "content": "what day is it?"}],
         "assistant": "today is [DAY]"},
        {"turns": [{"role": "user", "content": "hi"},
                   {"role": "assistant", "content": "hello!"},
                   {"role": "user", "content": "remind me the day?"}],
         "assistant": "it's [DAY]"},
    ],
}


class TestTemplateRoutes:
    def test_create_list_get_delete(
        self, session_and_client: Any, tmp_path: Any, monkeypatch: Any,
    ) -> None:
        monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
        _session, client = session_and_client
        resp = client.post("/saklas/v1/templates", json=_TMPL_PAYLOAD)
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["name"] == "weekday"
        assert body["labels"] == ["monday", "tuesday", "wednesday"]
        assert body["n_contexts"] == 2

        listing = client.get("/saklas/v1/templates").json()["templates"]
        assert any(t["name"] == "weekday" for t in listing)

        detail = client.get("/saklas/v1/templates/local/weekday").json()
        assert detail["slot"] == "[DAY]"
        assert len(detail["contexts"]) == 2
        assert detail["contexts"][1]["turns"][-1]["content"] == "remind me the day?"

        assert client.delete("/saklas/v1/templates/local/weekday").status_code == 200
        assert client.get("/saklas/v1/templates/local/weekday").status_code == 404

    def test_create_slot_in_history_rejected(
        self, session_and_client: Any, tmp_path: Any, monkeypatch: Any,
    ) -> None:
        monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
        _session, client = session_and_client
        bad = {
            "name": "bad", "slot": "[DAY]", "values": ["Monday", "Tuesday"],
            "contexts": [{"turns": [{"role": "user", "content": "is it [DAY]?"}],
                          "assistant": "yes [DAY]"}],
        }
        assert client.post("/saklas/v1/templates", json=bad).status_code == 400

    def test_score_route_wires_session(
        self, session_and_client: Any, tmp_path: Any, monkeypatch: Any,
    ) -> None:
        monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
        session, client = session_and_client
        client.post("/saklas/v1/templates", json=_TMPL_PAYLOAD)

        from saklas.core.scoring import ChoiceScore, ChoiceScores
        fake = [ChoiceScores(choices=(
            ChoiceScore("Monday", "monday", (1,), 1, -1.0, -1.0, 0.7, 0.7),
            ChoiceScore("Tuesday", "tuesday", (2,), 1, -2.0, -2.0, 0.3, 0.3),
        ), steering="0.5 a.b")]
        session.score_template.return_value = fake

        resp = client.post(
            "/saklas/v1/templates/local/weekday/score",
            json={"steering": "0.5 a.b"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["template"] == "weekday"
        assert body["steering"] == "0.5 a.b"
        assert body["contexts"][0]["choices"][0]["label"] == "monday"
        assert body["contexts"][0]["choices"][0]["prob_sum"] == 0.7

    def test_score_missing_template_404(
        self, session_and_client: Any, tmp_path: Any, monkeypatch: Any,
    ) -> None:
        monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
        _session, client = session_and_client
        assert client.post(
            "/saklas/v1/templates/local/ghost/score", json={},
        ).status_code == 404


# ---- per-message roles (sampling carrier) --------------------------------


class TestRoleSampling:
    def test_build_sampling_carries_roles(self):
        """WS sampling roles map onto SamplingConfig (the per-send carrier)."""
        from saklas.server.ws_models import WSSamplingParams, build_sampling_config

        sc = build_sampling_config(
            WSSamplingParams(user_role="captain", assistant_role="oracle")
        )
        assert sc is not None
        assert sc.user_role == "captain"
        assert sc.assistant_role == "oracle"

    def test_build_sampling_blank_roles_omitted(self):
        """Empty-string role boxes are treated as "no label" (None)."""
        from saklas.server.ws_models import WSSamplingParams, build_sampling_config

        sc = build_sampling_config(WSSamplingParams(user_role="", assistant_role=""))
        assert sc is not None
        assert sc.user_role is None
        assert sc.assistant_role is None


class TestPairwiseMetric:
    """``GET /profiles/pairwise`` is Mahalanobis-only (no Euclidean path)."""

    def _setup(self, session_and_client: Any) -> tuple[Any, TestClient]:
        import torch
        from saklas import Profile
        session, client = session_and_client
        # Two dim-4 vectors over layers {0, 1}.
        torch.manual_seed(1)
        session.profiles = {
            "x": Profile({0: torch.randn(4), 1: torch.randn(4)}),
            "y": Profile({0: torch.randn(4), 1: torch.randn(4)}),
        }
        return session, cast(TestClient, client)

    def test_mahalanobis_default(self, session_and_client: Any) -> None:
        import torch
        from saklas.core.mahalanobis import LayerWhitener
        session, client = self._setup(session_and_client)
        g = torch.Generator().manual_seed(4)
        acts = {L: torch.randn(80, 4, generator=g) for L in (0, 1)}
        means = {L: torch.zeros(4) for L in (0, 1)}
        w = LayerWhitener.from_neutral_activations(acts, means)
        session.whitener = w
        r = client.get("/saklas/v1/sessions/default/profiles/pairwise?a=x&b=y")
        assert r.status_code == 200
        body = r.json()
        assert body["metric"] == "mahalanobis"
        for i, la in enumerate(body["layers_a"]):
            for j, lb in enumerate(body["layers_b"]):
                # Each cell is whitened in the row layer's frame.
                vx, vy = session.profiles["x"][la], session.profiles["y"][lb]
                ref = w.mahalanobis_cosine(la, vx, vy)
                assert body["matrix"][i][j] == pytest.approx(ref, abs=1e-5)

    def test_missing_whitener_409(self, session_and_client: Any) -> None:
        """No covering whitener → 409 (the neutral cache must be regenerated);
        there is no Euclidean fallback."""
        session, client = self._setup(session_and_client)
        session.whitener = None
        r = client.get("/saklas/v1/sessions/default/profiles/pairwise?a=x&b=y")
        assert r.status_code == 409


class TestAnalyticsMultiNodeProbe:
    """A multi-node / curved probe (the ``personas`` rank-R fan) has no single
    steering direction, so the direction-cosine analytics must *exclude* it —
    not 500 on ``folded_directions`` (regression: a rank-8 probe attached
    while the web UI polled ``/correlation`` aborted the request)."""

    def _wire(self, session_and_client: Any) -> tuple[Any, TestClient]:
        import threading

        import torch

        from saklas.core.mahalanobis import LayerWhitener
        from saklas.core.manifold import (
            CustomDomain, LayerSubspace, Manifold,
        )
        from saklas.core.session import SaklasSession
        from saklas.core.capture import fold_directions_to_subspace

        session, client = session_and_client
        torch.manual_seed(3)

        # One registered steering vector + one foldable (R=1) vector probe +
        # one rank-3 multi-node probe (the shape that used to crash), all over
        # layers {0, 1} in dim 4.
        vx = {0: torch.randn(4), 1: torch.randn(4)}
        g = torch.Generator().manual_seed(11)
        acts = {L: torch.randn(80, 4, generator=g) for L in (0, 1)}
        means = {L: torch.zeros(4) for L in (0, 1)}
        whitener = LayerWhitener.from_neutral_activations(acts, means)
        vp = fold_directions_to_subspace(
            "vp", {0: torch.randn(4), 1: torch.randn(4)}, means,
            whitener=whitener,
        )
        K, R, D = 4, 3, 4
        basis, _ = torch.linalg.qr(torch.randn(D, R))
        basis = basis.T.contiguous()
        fan = Manifold(
            name="fan",
            domain=CustomDomain(R),
            node_labels=[f"n{i}" for i in range(K)],
            node_coords=torch.randn(K, R),
            layers={
                L: LayerSubspace.affine(
                    torch.zeros(D), basis, node_coords=torch.randn(K, R),
                )
                for L in (0, 1)
            },
        )

        # These stubs feed the *engine's own* analytics methods, bound real
        # below — ``analytics_names`` / ``_live_direction_tensors`` read the
        # session privates (``self._profiles`` / ``self._monitor``) directly, so
        # the stub seeds the private backing, not the public alias.  (The public
        # ``monitor`` / ``profiles`` properties just return these same objects.)
        session._profiles = {"vx": vx}
        session._monitor.probe_names = ["vp", "fan"]
        session._monitor.manifolds = {"vp": vp, "fan": fan}
        session.gen_lock = threading.Lock()
        session._analytics_cpu_cache = {}
        # ``analytics_names`` reads the roster through the geometry
        # instrument's LOCKED ``manifolds()`` snapshot now — a bare MagicMock
        # instrument silently yields an empty roster, so wire the real one
        # (it reads the same ``session._monitor.manifolds`` stub above).
        from saklas.core.instruments.geometry import GeometryInstrument

        session._geometry_instrument = GeometryInstrument(session)
        # Bind the real analytics methods onto the mock so the endpoint
        # exercises the production fold path, not a vacuous MagicMock.
        session.analytics_names = lambda: SaklasSession.analytics_names(session)
        session._live_direction_tensors = (
            lambda n: SaklasSession._live_direction_tensors(session, n)
        )
        session.analytics_profile = (
            lambda n: SaklasSession.analytics_profile(session, n)
        )

        session.whitener = whitener
        return session, cast(TestClient, client)

    def test_analytics_names_excludes_multinode(self, session_and_client: Any) -> None:
        session, _ = self._wire(session_and_client)
        # The fan is dropped; the vector and the R=1 probe survive.
        assert session.analytics_names() == ["vp", "vx"]
        assert session._live_direction_tensors("fan") is None

    def test_correlation_skips_multinode_no_500(self, session_and_client: Any) -> None:
        _, client = self._wire(session_and_client)
        r = client.get("/saklas/v1/sessions/default/correlation")
        assert r.status_code == 200
        body = r.json()
        assert "fan" not in body["names"]
        assert body["names"] == ["vp", "vx"]
        # The surviving pair has a real (non-null) cosine cell.
        assert body["matrix"]["vp"]["vx"] is not None

    def test_correlation_explicit_multinode_404(self, session_and_client: Any) -> None:
        _, client = self._wire(session_and_client)
        r = client.get("/saklas/v1/sessions/default/correlation?names=fan,vx")
        assert r.status_code == 404

    def test_pairwise_multinode_404(self, session_and_client: Any) -> None:
        _, client = self._wire(session_and_client)
        r = client.get("/saklas/v1/sessions/default/profiles/pairwise?a=fan&b=vx")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# Native error envelope
# ---------------------------------------------------------------------------


class TestNativeErrorEnvelope:
    """Every ``/saklas/v1/*`` failure renders as ``{"detail": "<string>"}``.

    The three shapes a native route can fail in — a typed ``SaklasError``, a
    bare ``HTTPException`` (with a string, dict, or list detail), and a body
    validation error — used to emit three different envelopes, so the client
    had to probe for whichever one it got.
    """

    def test_bare_http_exception_detail_is_a_string(
        self, session_and_client: Any,
    ) -> None:
        _, client = session_and_client
        resp = client.get("/saklas/v1/sessions/nope")
        assert resp.status_code == 404
        body = resp.json()
        assert set(body) == {"detail"}
        assert isinstance(body["detail"], str)

    def test_typed_saklas_error_uses_the_native_envelope(
        self, session_and_client: Any,
    ) -> None:
        from saklas.core.errors import SaklasError

        class _Nope(RuntimeError, SaklasError):
            def user_message(self) -> tuple[int, str]:
                return (422, "geometry instrument is unavailable")

        session, client = session_and_client
        session.set_live_probe_scores.side_effect = _Nope()
        resp = client.post(
            "/saklas/v1/sessions/default/instruments/geometry/live",
            json={"enabled": True},
        )
        assert resp.status_code == 422
        assert resp.json() == {"detail": "geometry instrument is unavailable"}

    def test_dict_detail_is_flattened(self) -> None:
        """The auth dependency's OpenAI-shaped 401 detail flattens too."""
        from saklas.server import create_app

        session = _mock_session()
        client = TestClient(create_app(session, api_key="s3cret"))
        resp = client.get("/saklas/v1/sessions")
        assert resp.status_code == 401
        assert resp.json() == {"detail": "Invalid API key"}

    def test_openai_routes_keep_their_own_envelope(self) -> None:
        from saklas.server import create_app

        session = _mock_session()
        client = TestClient(create_app(session, api_key="s3cret"))
        resp = client.get("/v1/models")
        assert resp.status_code == 401
        assert set(resp.json()["detail"]) == {"message", "type", "param", "code"}

    def test_protocol_error_renders_each_envelope(self) -> None:
        import json as _json

        from saklas.server.app import _protocol_error

        native = _protocol_error("/saklas/v1/sessions", 409, "busy")
        assert _json.loads(bytes(native.body)) == {"detail": "busy"}

        ollama = _protocol_error("/api/chat", 400, "busy")
        assert _json.loads(bytes(ollama.body)) == {"error": "busy"}

        openai = _protocol_error("/v1/chat/completions", 409, "busy")
        assert _json.loads(bytes(openai.body)) == {
            "error": {
                "message": "busy", "type": "conflict",
                "param": None, "code": 409,
            },
        }
