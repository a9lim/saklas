"""Slash command dispatch + `_generate` worker contract tests.

These tests mock out the session and exercise ``SaklasApp`` without mounting
a Textual app — we instantiate via ``object.__new__`` and manually initialize
just the state the dispatchers touch. TUI rendering is out of scope.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

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
    app._last_tok_per_sec = 0.0
    app._last_elapsed = 0.0
    app._log_ppl_sum = 0.0
    app._ppl_count = 0
    app._last_gen_state = (-1, -1.0, -1.0, False, -1)
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


# ---- /compare command ----

def test_compare_pairwise():
    import torch
    from saklas.core.profile import Profile

    app = _make_app()
    t = {0: torch.randn(8), 1: torch.randn(8)}
    app._session._profiles = {
        "angry.calm": Profile(t),
        "happy.sad": Profile({k: v.clone() for k, v in t.items()}),
    }
    app._session._monitor.profiles = {}
    app._handle_command("/compare angry.calm happy.sad")
    msg = _msgs(app)
    assert "angry.calm" in msg and "happy.sad" in msg


def test_compare_ranked():
    import torch
    from saklas.core.profile import Profile

    app = _make_app()
    base = {0: torch.randn(8), 1: torch.randn(8)}
    app._session._profiles = {
        "angry.calm": Profile(base),
        "happy.sad": Profile({k: torch.randn(8) for k in base}),
        "formal.casual": Profile({k: torch.randn(8) for k in base}),
    }
    app._session._monitor.profiles = {
        "angry.calm": Profile(base),
        "happy.sad": Profile({k: torch.randn(8) for k in base}),
        "formal.casual": Profile({k: torch.randn(8) for k in base}),
    }
    app._handle_command("/compare angry.calm")
    msg = _msgs(app)
    assert "angry.calm" in msg


def test_compare_unknown_name():
    app = _make_app()
    app._session._profiles = {}
    app._session._monitor.profiles = {}
    app._handle_command("/compare ghost")
    msg = _msgs(app)
    assert "not found" in msg.lower() or "no profile" in msg.lower()


def test_compare_no_args():
    app = _make_app()
    app._handle_command("/compare")
    msg = _msgs(app)
    assert "Usage" in msg or "usage" in msg


# ---- _parse_args: period delim, multi-word poles, hyphen-in-name ----


def test_parse_single_concept_no_alpha():
    concept, baseline = SaklasApp._parse_args("happy")
    assert concept == "happy"
    assert baseline is None


def test_parse_canonical_dotted_stays_whole():
    # `dog.cat` (no surrounding spaces on the dot) is a single canonical
    # name, not a split.
    concept, baseline = SaklasApp._parse_args("dog.cat")
    assert concept == "dog.cat"
    assert baseline is None


def test_parse_period_delim_splits():
    concept, baseline = SaklasApp._parse_args("dog . cat")
    assert concept == "dog"
    assert baseline == "cat"


def test_parse_dash_no_longer_splits():
    # `-` is allowed inside NAME_REGEX, so `dog - cat` is treated as a
    # single (invalid-but-unsplit) concept. Downstream validation rejects
    # the spaces; the parser does not split on the hyphen.
    concept, baseline = SaklasApp._parse_args("dog - cat")
    assert concept == "dog - cat"
    assert baseline is None


def test_parse_multiword_unquoted_period():
    concept, baseline = SaklasApp._parse_args("a dog . a pair of cats")
    assert concept == "a dog"
    assert baseline == "a pair of cats"


def test_parse_quoted_poles_still_accepted():
    concept, baseline = SaklasApp._parse_args('"a dog" . "a pair of cats"')
    assert concept == "a dog"
    assert baseline == "a pair of cats"


def test_parse_with_alpha_single():
    concept, baseline, alpha = SaklasApp._parse_args("happy 0.3", include_alpha=True)
    assert concept == "happy"
    assert baseline is None
    assert alpha == 0.3


def test_parse_with_alpha_bipolar_period():
    concept, baseline, alpha = SaklasApp._parse_args(
        "happy . sad 0.4", include_alpha=True
    )
    assert concept == "happy"
    assert baseline == "sad"
    assert alpha == 0.4


def test_parse_with_alpha_multiword_period():
    concept, baseline, alpha = SaklasApp._parse_args(
        "a dog . a pair of cats 0.25", include_alpha=True
    )
    assert concept == "a dog"
    assert baseline == "a pair of cats"
    assert alpha == 0.25


def test_parse_default_alpha_when_missing():
    from saklas.tui.app import DEFAULT_ALPHA
    concept, baseline, alpha = SaklasApp._parse_args("happy", include_alpha=True)
    assert alpha == DEFAULT_ALPHA


def test_parse_alpha_clamped_to_max():
    from saklas.tui.vector_panel import MAX_ALPHA
    _, _, alpha = SaklasApp._parse_args("happy 99", include_alpha=True)
    assert alpha == MAX_ALPHA
    _, _, alpha = SaklasApp._parse_args("happy -99", include_alpha=True)
    assert alpha == -MAX_ALPHA


# ---- /steer routes through the shared expression grammar ----


def test_steer_expression_parses_sae_variant(monkeypatch, tmp_path):
    """``/steer 0.3 myvec:sae`` parses through the shared grammar; the
    variant is preserved on the alphas key."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    from saklas.cli import selectors as _sel
    _sel.invalidate()
    from saklas.core.steering_expr import parse_expr
    s = parse_expr("0.3 myvec:sae")
    assert s.alphas == {"myvec:sae": pytest.approx(0.3)}


def test_steer_expression_hyphenated_concept(monkeypatch, tmp_path):
    """Dash-joined identifiers parse as a single concept name; the
    resolver's slug step collapses ``-`` to ``_`` so the final key uses
    underscores."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    from saklas.cli import selectors as _sel
    _sel.invalidate()
    from saklas.core.steering_expr import parse_expr
    s = parse_expr("0.3 high-context")
    assert list(s.alphas.keys()) == ["high_context"]


def test_steer_expression_release_suffix(monkeypatch, tmp_path):
    """Explicit release rides on the ``:sae-<release>`` suffix."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    from saklas.cli import selectors as _sel
    _sel.invalidate()
    from saklas.core.steering_expr import parse_expr
    s = parse_expr("0.3 myvec:sae-gemma-scope-2b-pt-res-canonical")
    assert "myvec:sae-gemma-scope-2b-pt-res-canonical" in s.alphas


def test_handle_extract_trusts_canonical_from_session(monkeypatch, tmp_path):
    """Regression: ``session.extract(sae=RELEASE)`` already returns a
    canonical with the ``:sae-<release>`` suffix. The TUI worker must NOT
    re-append ``:{variant}`` — doing so produces ``foo:sae-R:sae-R`` and
    breaks every subsequent ``/alpha`` / ``/unsteer`` / pole lookup.

    Contract: ``session.extract`` owns the final name. The TUI passes it
    through unchanged.
    """
    import torch
    from saklas.core.profile import Profile
    from saklas.cli import selectors as _sel
    # Isolate from user's real pack tree so pole alias resolution is a
    # no-op on the fabricated name.
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    _sel.invalidate()

    app = _make_app()

    # Mock session.extract to return the session-side canonical (suffixed).
    def _fake_extract(concept, **kwargs):
        assert kwargs.get("sae") == "gemma-scope-2b-pt-res-canonical"
        canonical = f"{concept}:sae-gemma-scope-2b-pt-res-canonical"
        return canonical, Profile({0: torch.zeros(4)})
    app._session.extract = _fake_extract

    # Run worker inline so we can capture the final registered name.
    def _run_worker(fn, thread=True):
        fn()
    app.run_worker = _run_worker

    captured: dict = {}

    def _on_success(name, profile, alpha):
        captured["name"] = name

    app._handle_extract(
        "honest 0.3", include_alpha=True, on_success=_on_success,
        variant="sae-gemma-scope-2b-pt-res-canonical",
    )

    assert "name" in captured, f"worker never called on_success: {_msgs(app)!r}"
    # The canonical is correct; no double-suffix, no bare unsuffixed name.
    assert captured["name"] == "honest:sae-gemma-scope-2b-pt-res-canonical"
    assert ":sae-" in captured["name"]
    assert captured["name"].count(":sae-") == 1


def test_handle_extract_raw_variant_passes_canonical_through(monkeypatch, tmp_path):
    """Raw (no SAE) path: ``session.extract`` returns the bare canonical."""
    import torch
    from saklas.core.profile import Profile
    from saklas.cli import selectors as _sel
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    _sel.invalidate()

    app = _make_app()

    def _fake_extract(concept, **kwargs):
        assert "sae" not in kwargs
        return concept, Profile({0: torch.zeros(4)})
    app._session.extract = _fake_extract

    def _run_worker(fn, thread=True):
        fn()
    app.run_worker = _run_worker

    captured: dict = {}

    def _on_success(name, profile, alpha):
        captured["name"] = name

    app._handle_extract(
        "honest 0.3", include_alpha=True, on_success=_on_success, variant="raw",
    )
    assert captured["name"] == "honest"


def test_handle_extract_explicit_sae_suffix_in_concept(monkeypatch, tmp_path):
    """Option C: typing ``concept:sae-<release>`` routes the release to
    ``session.extract(sae=release)`` even without the ``--sae`` preamble.
    """
    import torch
    from saklas.core.profile import Profile
    from saklas.cli import selectors as _sel
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    _sel.invalidate()

    app = _make_app()

    def _fake_extract(concept, **kwargs):
        # The ``:sae-<release>`` suffix must get peeled before the
        # concept reaches session.extract — release rides on ``sae=``.
        assert concept == "honest"
        assert kwargs["sae"] == "my-release"
        return f"{concept}:sae-my-release", Profile({0: torch.zeros(4)})
    app._session.extract = _fake_extract

    def _run_worker(fn, thread=True):
        fn()
    app.run_worker = _run_worker

    captured: dict = {}

    def _on_success(name, profile, alpha):
        captured["name"] = name

    # variant defaults to "raw" — the suffix inside the concept flips it.
    app._handle_extract(
        "honest:sae-my-release 0.3", include_alpha=True,
        on_success=_on_success, variant="raw",
    )
    assert captured["name"] == "honest:sae-my-release"


def test_handle_extract_bare_sae_uses_autoload(monkeypatch, tmp_path):
    """Option C ``--sae <concept>``: no fresh extract — session autoload
    picks the unique SAE tensor already on disk.
    """
    import torch
    from saklas.cli import selectors as _sel
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    _sel.invalidate()

    app = _make_app()

    # session.extract must NOT be called for bare --sae.
    def _fail_extract(*a, **kw):
        raise AssertionError("session.extract must not run for bare --sae")
    app._session.extract = _fail_extract

    # _try_autoload_vector populates _profiles[<concept>:sae].
    def _autoload(canonical, *, variant):
        assert canonical == "honest"
        assert variant == "sae"
        app._session._profiles["honest:sae"] = {0: torch.zeros(4)}
    app._session._try_autoload_vector = _autoload

    def _run_worker(fn, thread=True):
        fn()
    app.run_worker = _run_worker

    captured: dict = {}

    def _on_success(name, profile, alpha):
        captured["name"] = name

    app._handle_extract(
        "honest 0.3", include_alpha=True,
        on_success=_on_success, variant="sae",
    )
    assert captured["name"] == "honest:sae"
