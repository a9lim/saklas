"""Slash command dispatch + `_generate` worker contract tests.

These tests mock out the session and exercise ``SaklasApp`` without mounting
a Textual app — we instantiate via ``object.__new__`` and manually initialize
just the state the dispatchers touch. TUI rendering is out of scope.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock


import saklas
from saklas.tui.app import SaklasApp


def _make_app():
    """Instantiate SaklasApp without its Textual __init__.

    The slash-command dispatch + _generate worker only need the attribute
    bag — not a live Textual tree — so we build one by hand.
    """
    app = object.__new__(SaklasApp)
    session = MagicMock()
    session._history = []
    session._profiles = {}
    session._model_info = {"model_id": "mock/mock", "model_type": "mock"}
    session._device = SimpleNamespace(type="cpu")
    session._layers = [0, 1, 2]
    session._monitor = MagicMock()
    session._monitor.probe_names = []
    session._monitor.profiles = {}
    session._last_result = None
    session._last_per_token_scores = None
    session._tokenizer = MagicMock()
    session.config = SimpleNamespace(
        temperature=0.7, top_p=0.9, max_new_tokens=128,
        system_prompt=None,
    )

    app._session = session
    app._messages = session._history
    app._device_str = "cpu"
    app._alphas = {}
    app._enabled = {}
    app._supports_thinking = False
    app._thinking = False
    app._current_assistant_widget = None
    app._poll_timer = None
    app._last_prompt = None
    app._ab_in_progress = False
    app._pending_action = None
    app._gen_active = False
    app._focused_panel_idx = 1
    app._highlighting = False
    app._highlight_probe = None
    app._default_seed = None
    import queue
    app._ui_token_queue = queue.SimpleQueue()
    app._gen_start_time = 0.0
    app._gen_token_count = 0
    app._prompt_token_count = 0
    app._last_tok_per_sec = 0.0
    app._last_elapsed = 0.0
    app._cached_vram_gb = 0.0
    app._vram_poll_counter = 0
    app._last_gen_state = (-1, -1.0, -1.0, -1.0, False)
    app._assistant_messages = []

    # Mock chat panel — only capture system messages.
    chat = MagicMock()
    chat.messages = []
    chat.add_system_message = lambda msg: chat.messages.append(msg)
    app._chat_panel = chat

    trait = MagicMock()
    trait.get_selected_probe = MagicMock(return_value=None)
    app._trait_panel = trait

    left = MagicMock()
    app._left_panel = left

    return app


def _msgs(app):
    return "\n".join(app._chat_panel.messages)


# ---- Task B: /steer triad dispatch ----


def test_alpha_rejects_unregistered():
    app = _make_app()
    app._handle_command("/alpha nonexistent 0.5")
    assert "not active" in _msgs(app)


def test_alpha_adjusts_existing():
    app = _make_app()
    app._alphas["angry.calm"] = 0.3
    app._refresh_left_panel = MagicMock()
    app._handle_command("/alpha angry.calm 0.7")
    assert app._alphas["angry.calm"] == 0.7
    assert "set to" in _msgs(app)


def test_alpha_invalid_value():
    app = _make_app()
    app._alphas["foo"] = 0.1
    app._refresh_left_panel = MagicMock()
    app._handle_command("/alpha foo notanumber")
    assert "Invalid alpha" in _msgs(app)


def test_alpha_usage_on_missing_args():
    app = _make_app()
    app._handle_command("/alpha foo")
    assert "Usage: /alpha" in _msgs(app)


def test_unsteer_removes():
    app = _make_app()
    app._alphas["foo"] = 0.5
    app._enabled["foo"] = True
    app._refresh_left_panel = MagicMock()
    app._handle_command("/unsteer foo")
    assert "foo" not in app._alphas
    app._session.unsteer.assert_called_with("foo")


def test_unsteer_rejects_missing():
    app = _make_app()
    app._handle_command("/unsteer ghost")
    assert "not active" in _msgs(app)


# ---- Task C: new slash commands ----


def test_seed_set_clear_show():
    app = _make_app()
    app._handle_command("/seed 42")
    assert app._default_seed == 42
    app._chat_panel.messages.clear()
    app._handle_command("/seed")
    assert "42" in _msgs(app)
    app._handle_command("/seed clear")
    assert app._default_seed is None


def test_seed_invalid():
    app = _make_app()
    app._handle_command("/seed notanint")
    assert "Invalid seed" in _msgs(app)


def test_unprobe_missing():
    app = _make_app()
    app._handle_command("/unprobe ghost")
    assert "not active" in _msgs(app)


def test_unprobe_removes():
    app = _make_app()
    app._session._monitor.probe_names = ["happy.sad"]
    app._trait_panel.set_active_probes = MagicMock()
    app._apply_highlight_to_all = MagicMock()

    def _unprobe(name):
        app._session._monitor.probe_names = []
    app._session.unprobe.side_effect = _unprobe

    app._highlight_probe = "happy.sad"
    app._highlighting = True
    app._handle_command("/unprobe happy.sad")
    app._session.unprobe.assert_called_with("happy.sad")
    # Highlight seed cleared when its probe was removed.
    assert app._highlight_probe is None
    assert app._highlighting is False


def test_model_info():
    app = _make_app()
    app._handle_command("/model")
    msg = _msgs(app)
    assert "mock/mock" in msg
    assert "Active vectors" in msg


def test_save_load_roundtrip(tmp_path, monkeypatch):
    # Redirect saklas_home() → tmp_path via env var.
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    app = _make_app()
    app._session._history.extend([
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ])
    app._alphas["foo"] = 0.5
    app._session._profiles["foo"] = {0: None}  # so load restores it
    app._enabled["foo"] = True
    app._default_seed = 7
    app._handle_command("/save convtest")

    saved = (tmp_path / "conversations" / "convtest.json")
    assert saved.exists()

    # Wipe state and load.
    app2 = _make_app()
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    app2._session._profiles["foo"] = {0: None}
    app2._refresh_left_panel = MagicMock()

    # chat_panel mocks used by _do_clear.
    app2._chat_panel.clear_log = MagicMock()
    app2._session.clear_history = MagicMock(
        side_effect=lambda: app2._session._history.clear()
    )

    app2._handle_command("/load convtest")
    assert app2._alphas.get("foo") == 0.5
    assert app2._default_seed == 7


def test_help_mentions_new_bindings():
    app = _make_app()
    app._handle_command("/help")
    msg = _msgs(app)
    assert "Ctrl+A" in msg
    assert "Ctrl+S" in msg
    assert "/alpha" in msg
    assert "/unsteer" in msg
    assert "/save" in msg
    assert "/load" in msg
    assert "/why" in msg
    assert "/seed" in msg
    assert "/regen" in msg
    assert "/export" in msg
    assert "/model" in msg


# ---- Task A: _generate worker uses new API ----


def test_generate_worker_uses_generate_stream(monkeypatch):
    app = _make_app()
    app._session._device = SimpleNamespace(type="cpu")
    # Stub generate_stream to capture kwargs and yield one event.
    captured = {}

    class _Event:
        text = "hi"
        thinking = False
        token_id = 1
        logprob = None
        top_logprobs = None
        index = 0

    def _fake_stream(input, **kwargs):
        captured["input"] = input
        captured["kwargs"] = kwargs
        yield _Event()
    app._session.generate_stream = _fake_stream

    # Mock the chat panel widget machinery.
    widget = MagicMock()
    app._chat_panel.start_assistant_message = MagicMock(return_value=widget)

    # Track worker dispatch — run inline.
    def _run_worker(fn, thread=True):
        fn()
    app.run_worker = _run_worker

    app._start_generation("hello world")

    assert captured["input"] == "hello world"
    kwargs = captured["kwargs"]
    assert "sampling" in kwargs
    assert "steering" in kwargs
    assert "thinking" in kwargs
    assert isinstance(kwargs["sampling"], saklas.SamplingConfig)
    # No steering registered → None.
    assert kwargs["steering"] is None


def test_generate_worker_passes_steering_when_alphas_active():
    app = _make_app()
    app._alphas["foo"] = 0.5
    app._enabled["foo"] = True
    captured = {}

    def _fake_stream(input, **kwargs):
        captured["kwargs"] = kwargs
        return iter([])
    app._session.generate_stream = _fake_stream
    app._chat_panel.start_assistant_message = MagicMock(return_value=MagicMock())

    def _run_worker(fn, thread=True):
        fn()
    app.run_worker = _run_worker

    app._start_generation("hello")
    steering = captured["kwargs"]["steering"]
    assert isinstance(steering, saklas.Steering)
    assert steering.alphas == {"foo": 0.5}


# ---- Task D: /probe seeds highlight ----


def test_probe_seeds_highlight():
    app = _make_app()
    # Simulate probe-added callback (what _handle_probe ends up calling).
    app._session._monitor.probe_names = ["happy.sad"]
    app._trait_panel.set_active_probes = MagicMock()
    app._assistant_messages = []
    app._on_probe_added("happy.sad")
    assert app._highlight_probe == "happy.sad"
    assert app._highlighting is True
