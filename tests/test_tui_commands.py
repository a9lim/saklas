"""Slash command dispatch + `_generate` worker contract tests.

These tests mock out the session and exercise ``SaklasApp`` without mounting
a Textual app — we instantiate via ``object.__new__`` and manually initialize
just the state the dispatchers touch. TUI rendering is out of scope.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

import saklas
from saklas.tui.app import SaklasApp


def _make_app():
    """Instantiate SaklasApp without its Textual __init__.

    The slash-command dispatch + _generate worker only need the attribute
    bag — not a live Textual tree — so we build one by hand.
    """
    app: Any = object.__new__(SaklasApp)
    session: Any = MagicMock()
    # v2.3 loom: conversation lives in ``session.tree`` (LoomTree).
    # We install a real LoomTree so the regen/rewind path's
    # navigate/edit calls work; ``session.history`` is the derived
    # view the TUI's ``_messages`` property reads.
    from saklas import LoomTree as _LoomTree
    session.tree = _LoomTree()
    session.history = []
    session._profiles = {}
    session._model_info = {"model_id": "mock/mock", "model_type": "mock"}
    session._device = SimpleNamespace(type="cpu")
    session._layers = [0, 1, 2]
    session._monitor = MagicMock()
    session._monitor.probe_names = []
    session._monitor.profiles = {}
    # Manifold monitor — peer to ``_monitor`` per Phase 1.  Tests that
    # exercise manifold-probe slash commands populate ``probe_names`` /
    # ``attached_probes`` directly on the mock; the default empty state
    # makes the trait-panel manifold section a no-op renderer.
    session._manifold_monitor = MagicMock()
    session._manifold_monitor.probe_names = []
    session._manifold_monitor.attached_probes = MagicMock(return_value={})
    session._manifold_monitor.has_pending_per_token = MagicMock(return_value=False)
    session.manifold_monitor = session._manifold_monitor
    session._last_result = None
    session._last_per_token_scores = None
    session._tokenizer = MagicMock()
    session.config = SimpleNamespace(
        temperature=0.7, top_p=0.9, max_new_tokens=128,
        system_prompt=None,
    )
    # Explicit False — MagicMock attribute access otherwise returns a
    # MagicMock (truthy), which would flip every "is a gen running?"
    # check in slash dispatch into the pending-action defer branch.
    session.is_generating = False
    session.gen_state = saklas.GenState.IDLE

    app._session = session
    # ``app._messages`` is now a property derived from ``session.history``
    # under v2.3 loom; the v2.2 shared-list assignment is no longer needed.
    app._device_str = "cpu"
    app._alphas = {}
    app._enabled = {}
    app._manifold_terms = {}
    app._supports_thinking = False
    app._is_base_model = False
    app._render_mode = "chat"
    app._thinking = False
    app._current_assistant_widget = None
    app._poll_timer = None
    app._last_prompt = None
    app._ab_mode = False
    app._ab_shadow_active = False
    app._ab_shadow_row = None
    app._row_for_widget = {}
    # Pending-queue replaces the legacy single-slot ``_pending_action``.
    # ``_pulled_slot`` tracks an in-progress ↑-pull-and-edit; default
    # None means "no slot pulled."  Tests that need a populated queue
    # mutate ``_pending_queue`` directly.
    app._pending_queue = []
    app._pulled_slot = None
    app._ui_gen_active = False
    app._focused_panel_idx = 1
    app._highlighting = False
    app._highlight_probe = None
    app._default_seed = None
    import queue
    app._ui_token_queue = queue.SimpleQueue()
    # Input-history ring (↑/↓ recall in the chat input).
    app._input_history = []
    app._history_index = None
    app._history_stash = ""
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
    # ``_repaint_chat_from_active_path`` (loom navigation / ``/load``)
    # unpacks the ``(row, widget)`` tuple ``add_finalized_assistant``
    # returns — give the default mock a real tuple so the repaint path
    # doesn't trip over MagicMock's empty ``__iter__``.
    chat.add_finalized_assistant = MagicMock(
        return_value=(MagicMock(), MagicMock()),
    )
    app._chat_panel = chat

    trait = MagicMock()
    trait.get_selected_probe = MagicMock(return_value=None)
    app._trait_panel = trait

    left = MagicMock()
    app._left_panel = left

    return app


def _msgs(app: Any) -> str:
    return "\n".join(app._chat_panel.messages)


# ---- Task B: /steer triad dispatch ----


def test_alpha_rejects_unregistered():
    # Value-first: ``/alpha 0.5 nonexistent`` matches the expression
    # grammar (``0.5 honest``) instead of flipping noun/number order.
    app = _make_app()
    app._handle_command("/alpha 0.5 nonexistent")
    assert "not active" in _msgs(app)


def test_alpha_adjusts_existing():
    app = _make_app()
    app._alphas["angry.calm"] = 0.3
    app._refresh_left_panel = MagicMock()
    app._handle_command("/alpha 0.7 angry.calm")
    assert app._alphas["angry.calm"] == 0.7
    assert "set to" in _msgs(app)


def test_alpha_invalid_value():
    app = _make_app()
    app._alphas["foo"] = 0.1
    app._refresh_left_panel = MagicMock()
    app._handle_command("/alpha notanumber foo")
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


def test_unsteer_removes_manifold_term():
    """``/unsteer <manifold>`` resolves against ``_manifold_terms`` —
    a ``%`` term racked via ``/steer`` must be removable by name from
    the slash command, not only the panel backspace path."""
    from saklas.core.steering_expr import ManifoldTerm
    from saklas.core.triggers import Trigger

    app = _make_app()
    term = ManifoldTerm(
        coeff=0.7, trigger=Trigger.BOTH, manifold="circumplex",
        position=(0.3, 0.8),
    )
    app._manifold_terms = {"circumplex%0.3,0.8": term}
    app._enabled = {"circumplex%0.3,0.8": True}
    app._refresh_left_panel = MagicMock()

    app._handle_command("/unsteer circumplex%0.3,0.8")

    assert "circumplex%0.3,0.8" not in app._manifold_terms
    assert "circumplex%0.3,0.8" not in app._enabled
    # Manifold terms aren't session-registered profiles — the session's
    # ``unsteer`` must not be touched for them.
    app._session.unsteer.assert_not_called()
    assert "Removed manifold" in _msgs(app)


def test_unsteer_namespace_sweeps_manifold_terms():
    """``/unsteer ns/`` sweeps both scalar vectors and manifold terms
    whose keys sit under the namespace."""
    from saklas.core.steering_expr import ManifoldTerm
    from saklas.core.triggers import Trigger

    app = _make_app()
    app._alphas = {"alice/foo": 0.5}
    term = ManifoldTerm(
        coeff=0.6, trigger=Trigger.BOTH, manifold="alice/circ",
        position=(0.1,),
    )
    app._manifold_terms = {"alice/circ%0.1": term}
    app._enabled = {"alice/foo": True, "alice/circ%0.1": True}
    app._refresh_left_panel = MagicMock()

    app._handle_command("/unsteer alice/")

    assert app._alphas == {}
    assert app._manifold_terms == {}
    app._session.unsteer.assert_called_once_with("alice/foo")
    # The count in the report folds in the manifold term.
    assert "Removed 2 vector(s)" in _msgs(app)


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

    def _unprobe(name: Any) -> None:
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


def test_save_load_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """/save serializes the full loom tree; /load swaps it back in."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    app = _make_app()
    # Seed a small tree: user "hi" → assistant "hello".
    uid = app._session.tree.add_user_turn("hi")
    aid = app._session.tree.begin_assistant(uid)
    app._session.tree.finalize_assistant(aid, text="hello")

    app._handle_command("/save convtest")
    saved = tmp_path / "conversations" / "convtest.json"
    assert saved.exists()
    assert "saved tree" in _msgs(app)

    # Fresh app — load the saved tree back.
    app2 = _make_app()
    app2._handle_command("/load convtest")
    assert "loaded tree" in _msgs(app2)

    # The loaded tree carries the saved nodes (full tree, every branch).
    texts = {n.text for n in app2._session.tree.nodes.values()}
    assert "hi" in texts
    assert "hello" in texts


def test_load_missing_file_reports():
    """/load on a name with no saved file reports cleanly."""
    app = _make_app()
    app._handle_command("/load does-not-exist")
    assert "no saved tree" in _msgs(app)


def test_help_mentions_new_bindings():
    app = _make_app()
    app._handle_command("/help")
    msg = _msgs(app)
    assert "⌃A" in msg
    assert "⌃S" in msg
    assert "/alpha" in msg
    assert "/unsteer" in msg
    assert "/save" in msg
    assert "/load" in msg
    assert "/seed" in msg
    assert "/regen" in msg
    assert "/export" in msg
    assert "/model" in msg


# ---- Task A: _generate worker uses new API ----


def test_generate_worker_uses_generate_stream(monkeypatch: pytest.MonkeyPatch):
    app = _make_app()
    app._session._device = SimpleNamespace(type="cpu")
    # Stub generate_stream to capture kwargs and yield one event.
    captured = {}

    class _Event:
        text = "hi"
        thinking = False
        token_id = 1
        logprob = None
        # Phase 1 logit pass: renamed ``top_logprobs`` → ``top_alts``
        # (now carries decoded ``TokenAlt`` triples instead of id/lp
        # pairs).  Stub keeps it None — this test exercises a code path
        # that doesn't consume alts.
        top_alts = None
        index = 0
        scores = None
        perplexity = None

    def _fake_stream(input: Any, **kwargs: Any) -> Any:
        captured["input"] = input
        captured["kwargs"] = kwargs
        yield _Event()
    app._session.generate_stream = _fake_stream

    # Mock the chat panel widget machinery.
    widget = MagicMock()
    app._chat_panel.start_assistant_message = MagicMock(return_value=(MagicMock(), widget))

    # Track worker dispatch — run inline.
    def _run_worker(fn: Any, thread: bool = True) -> None:
        fn()
    app.run_worker = _run_worker

    app._start_generation("hello world")

    assert captured["input"] == "hello world"
    kwargs = captured["kwargs"]
    assert "sampling" in kwargs
    assert "steering" in kwargs
    assert "thinking" in kwargs
    assert kwargs["live_scores"] is False
    assert isinstance(kwargs["sampling"], saklas.SamplingConfig)
    # No steering registered → None.
    assert kwargs["steering"] is None


def test_generate_worker_enables_live_scores_for_probe_highlight():
    app = _make_app()
    app._session._device = SimpleNamespace(type="cpu")
    app._session._monitor.probe_names = ["happy.sad"]
    app._highlighting = True
    app._highlight_probe = "happy.sad"
    captured = {}

    def _fake_stream(input: Any, **kwargs: Any) -> Any:
        captured["kwargs"] = kwargs
        return iter([])

    app._session.generate_stream = _fake_stream
    app._chat_panel.start_assistant_message = MagicMock(
        return_value=(MagicMock(), MagicMock()),
    )
    app.run_worker = lambda fn, thread=True: fn()

    app._start_generation("hello")

    assert captured["kwargs"]["live_scores"] is True


def test_start_generation_inherits_highlight_state():
    """Fresh assistant widgets spawn with ``_highlight_on=False``; the
    app must push its current highlight state onto the widget at
    generation start so streamed tokens render highlighted from the
    first emit (regression: required a Ctrl+Y mode-cycle round trip
    post-gen).
    """
    app = _make_app()
    app._session._device = SimpleNamespace(type="cpu")
    app._highlighting = True
    app._highlight_probe = "honest.deceptive"

    widget = MagicMock()
    app._chat_panel.start_assistant_message = MagicMock(return_value=(MagicMock(), widget))
    app._session.generate_stream = MagicMock(return_value=iter([]))

    def _run_worker(fn: Any, thread: bool = True) -> None:
        fn()
    app.run_worker = _run_worker

    app._start_generation("hello")
    widget.apply_highlight.assert_called_with(True, "honest.deceptive")


def test_start_generation_skips_highlight_when_off():
    app = _make_app()
    app._session._device = SimpleNamespace(type="cpu")
    app._highlighting = False

    widget = MagicMock()
    app._chat_panel.start_assistant_message = MagicMock(return_value=(MagicMock(), widget))
    app._session.generate_stream = MagicMock(return_value=iter([]))

    def _run_worker(fn: Any, thread: bool = True) -> None:
        fn()
    app.run_worker = _run_worker

    app._start_generation("hello")
    widget.apply_highlight.assert_not_called()


def test_generate_worker_passes_steering_when_alphas_active():
    app = _make_app()
    app._alphas["foo"] = 0.5
    app._enabled["foo"] = True
    captured = {}

    def _fake_stream(input: Any, **kwargs: Any) -> Any:
        captured["kwargs"] = kwargs
        return iter([])
    app._session.generate_stream = _fake_stream
    app._chat_panel.start_assistant_message = MagicMock(return_value=(MagicMock(), MagicMock()))

    def _run_worker(fn: Any, thread: bool = True) -> None:
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


def test_steer_expression_parses_sae_variant(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """``/steer 0.3 myvec:sae`` parses through the shared grammar; the
    variant is preserved on the alphas key."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    from saklas.io import selectors as _sel
    _sel.invalidate()
    from saklas.core.steering_expr import parse_expr
    s = parse_expr("0.3 myvec:sae")
    assert s.alphas == {"myvec:sae": pytest.approx(0.3)}


def test_steer_expression_hyphenated_concept(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Dash-joined identifiers parse as a single concept name; the
    resolver's slug step collapses ``-`` to ``_`` so the final key uses
    underscores."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    from saklas.io import selectors as _sel
    _sel.invalidate()
    from saklas.core.steering_expr import parse_expr
    s = parse_expr("0.3 high-context")
    assert list(s.alphas.keys()) == ["high_context"]


def test_steer_expression_release_suffix(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Explicit release rides on the ``:sae-<release>`` suffix."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    from saklas.io import selectors as _sel
    _sel.invalidate()
    from saklas.core.steering_expr import parse_expr
    s = parse_expr("0.3 myvec:sae-gemma-scope-2b-pt-res-canonical")
    assert "myvec:sae-gemma-scope-2b-pt-res-canonical" in s.alphas


def test_handle_extract_trusts_canonical_from_session(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Regression: ``session.extract(sae=RELEASE)`` already returns a
    canonical with the ``:sae-<release>`` suffix. The TUI worker must NOT
    re-append ``:{variant}`` — doing so produces ``foo:sae-R:sae-R`` and
    breaks every subsequent ``/alpha`` / ``/unsteer`` / pole lookup.

    Contract: ``session.extract`` owns the final name. The TUI passes it
    through unchanged.
    """
    import torch
    from saklas.core.profile import Profile
    from saklas.io import selectors as _sel
    # Isolate from user's real pack tree so pole alias resolution is a
    # no-op on the fabricated name.
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    _sel.invalidate()

    app = _make_app()

    # Mock session.extract to return the session-side canonical (suffixed).
    def _fake_extract(concept: Any, **kwargs: Any) -> Any:
        assert kwargs.get("sae") == "gemma-scope-2b-pt-res-canonical"
        canonical = f"{concept}:sae-gemma-scope-2b-pt-res-canonical"
        return canonical, Profile({0: torch.zeros(4)})
    app._session.extract = _fake_extract

    # Run worker inline so we can capture the final registered name.
    def _run_worker(fn: Any, thread: bool = True) -> None:
        fn()
    app.run_worker = _run_worker

    captured: dict[str, Any] = {}

    def _on_success(name: Any, profile: Any, alpha: Any) -> None:
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


def test_handle_extract_raw_variant_passes_canonical_through(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Raw (no SAE) path: ``session.extract`` returns the bare canonical."""
    import torch
    from saklas.core.profile import Profile
    from saklas.io import selectors as _sel
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    _sel.invalidate()

    app = _make_app()

    def _fake_extract(concept: Any, **kwargs: Any) -> Any:
        assert "sae" not in kwargs
        return concept, Profile({0: torch.zeros(4)})
    app._session.extract = _fake_extract

    def _run_worker(fn: Any, thread: bool = True) -> None:
        fn()
    app.run_worker = _run_worker

    captured: dict[str, Any] = {}

    def _on_success(name: Any, profile: Any, alpha: Any) -> None:
        captured["name"] = name

    app._handle_extract(
        "honest 0.3", include_alpha=True, on_success=_on_success, variant="raw",
    )
    assert captured["name"] == "honest"


def test_handle_extract_explicit_sae_suffix_in_concept(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Option C: typing ``concept:sae-<release>`` routes the release to
    ``session.extract(sae=release)`` even without the ``--sae`` preamble.
    """
    import torch
    from saklas.core.profile import Profile
    from saklas.io import selectors as _sel
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    _sel.invalidate()

    app = _make_app()

    def _fake_extract(concept: Any, **kwargs: Any) -> Any:
        # The ``:sae-<release>`` suffix must get peeled before the
        # concept reaches session.extract — release rides on ``sae=``.
        assert concept == "honest"
        assert kwargs["sae"] == "my-release"
        return f"{concept}:sae-my-release", Profile({0: torch.zeros(4)})
    app._session.extract = _fake_extract

    def _run_worker(fn: Any, thread: bool = True) -> None:
        fn()
    app.run_worker = _run_worker

    captured: dict[str, Any] = {}

    def _on_success(name: Any, profile: Any, alpha: Any) -> None:
        captured["name"] = name

    # variant defaults to "raw" — the suffix inside the concept flips it.
    app._handle_extract(
        "honest:sae-my-release 0.3", include_alpha=True,
        on_success=_on_success, variant="raw",
    )
    assert captured["name"] == "honest:sae-my-release"


def test_handle_extract_bare_sae_uses_autoload(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Option C ``--sae <concept>``: no fresh extract — session autoload
    picks the unique SAE tensor already on disk.
    """
    import torch
    from saklas.io import selectors as _sel
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    _sel.invalidate()

    app = _make_app()

    # session.extract must NOT be called for bare --sae.
    def _fail_extract(*a: Any, **kw: Any) -> Any:
        raise AssertionError("session.extract must not run for bare --sae")
    app._session.extract = _fail_extract

    # _try_autoload_vector populates _profiles[<concept>:sae].
    def _autoload(canonical: Any, *, variant: Any) -> None:
        assert canonical == "honest"
        assert variant == "sae"
        app._session._profiles["honest:sae"] = {0: torch.zeros(4)}
    app._session._try_autoload_vector = _autoload

    def _run_worker(fn: Any, thread: bool = True) -> None:
        fn()
    app.run_worker = _run_worker

    captured: dict[str, Any] = {}

    def _on_success(name: Any, profile: Any, alpha: Any) -> None:
        captured["name"] = name

    app._handle_extract(
        "honest 0.3", include_alpha=True,
        on_success=_on_success, variant="sae",
    )
    assert captured["name"] == "honest:sae"


# ---- /steer error-path regression tests ----


def test_handle_steer_ambiguous_pole_does_not_crash(monkeypatch: pytest.MonkeyPatch):
    """Regression: ``/steer 0.5 <bare colliding name>`` used to escape
    ``except SteeringExprError`` because ``AmbiguousSelectorError`` is
    a ``SelectorError(ValueError, SaklasError)`` rather than a
    ``SteeringExprError``. The exception bubbled out of the slash-command
    handler, killed the Textual worker, and landed in ``crash.log``.

    Contract: ambiguous bare poles surface as a system message in the
    chat pane and ``_handle_steer`` returns cleanly. State (alphas,
    enabled, session steer calls) stays untouched.
    """
    from saklas.io.selectors import AmbiguousSelectorError
    import saklas.io.selectors as _sel

    app = _make_app()

    def _ambiguous(*_args: Any, **_kwargs: Any) -> Any:
        raise AmbiguousSelectorError(
            "ambiguous pole 'wolf': matches alice/wolf, default/deer.wolf"
        )
    # ``parse_expr`` imports ``resolve_pole`` lazily inside ``_resolve_atom``,
    # so monkeypatching the module attribute reaches the parser.
    monkeypatch.setattr(_sel, "resolve_pole", _ambiguous)

    app._handle_command("/steer 0.5 wolf")

    msgs = _msgs(app)
    assert "ambiguous pole 'wolf'" in msgs
    assert "alice/wolf" in msgs and "default/deer.wolf" in msgs
    # User-facing disambiguation hint comes from ``user_message()``.
    assert "namespace/name" in msgs
    # Slash command bailed before any state mutation.
    assert app._alphas == {}
    assert app._enabled == {}
    app._session.steer.assert_not_called()


def test_handle_steer_expression_error_still_caught(monkeypatch: pytest.MonkeyPatch):
    """Negative control: keep the original ``SteeringExprError`` arm
    working — bad grammar still emits the ``Steering expression error``
    prefix, not the generic ``Error`` one introduced by the new arm."""
    app = _make_app()
    # Empty expression after the slash command's own usage check passes.
    app._handle_command("/steer @@@nonsense")
    msgs = _msgs(app)
    assert "Steering expression error" in msgs
    assert app._alphas == {}


# ---- Namespace-bulk selector + handlers ----


def test_detect_namespace_selector_recognizes_trailing_slash():
    from saklas.tui.app import _detect_namespace_selector

    assert _detect_namespace_selector("alice/") == "alice"
    assert _detect_namespace_selector("  alice/  ") == "alice"
    assert _detect_namespace_selector("default/") == "default"


def test_detect_namespace_selector_rejects_non_bulk_forms():
    from saklas.tui.app import _detect_namespace_selector

    # Per-concept forms must NOT match — bulk would silently shadow them.
    assert _detect_namespace_selector("alice/foo") is None
    assert _detect_namespace_selector("0.5 alice/") is None
    assert _detect_namespace_selector("alice/foo/") is None
    assert _detect_namespace_selector("/") is None
    assert _detect_namespace_selector("") is None
    # Invalid namespace name (uppercase, leading digit) — same regex used
    # everywhere else for namespace strings.
    assert _detect_namespace_selector("Alice/") is None
    assert _detect_namespace_selector("9live/") is None


def _stub_concepts(monkeypatch: pytest.MonkeyPatch, concepts: Any) -> None:
    """Patch ``_all_concepts`` to return a synthetic list of namespaced
    folders. Each entry is a SimpleNamespace with ``namespace`` and
    ``name`` attributes — the only fields the bulk handlers read.
    """
    import saklas.io.selectors as _sel

    fakes = [SimpleNamespace(namespace=ns, name=n) for ns, n in concepts]
    monkeypatch.setattr(_sel, "_all_concepts", lambda: fakes)


def _drain_workers(app: Any) -> None:
    """Synchronously execute the most recent ``run_worker`` call.

    The TUI worker normally runs on a Textual thread; in tests we just
    inline it. ``call_from_thread`` is patched to call directly so the
    finish callback runs in the same thread.
    """
    for call in app.run_worker.call_args_list:
        # ``run_worker(_worker, thread=True)`` — first positional is the
        # callable. ``call_from_thread`` runs the finish closure inline.
        target = call.args[0] if call.args else call.kwargs.get("worker")
        if target is not None:
            target()
    app.run_worker.reset_mock()


def test_handle_steer_namespace_bulk_loads_cached_and_warns_on_skip(monkeypatch: pytest.MonkeyPatch):
    app = _make_app()
    # Two cached, one missing on disk for this model.
    _stub_concepts(monkeypatch, [
        ("alice", "honest.deceptive"),
        ("alice", "warm.clinical"),
        ("alice", "needs_extract"),
    ])

    cached_keys = {"alice/honest.deceptive", "alice/warm.clinical"}

    import torch

    def _autoload(canonical: Any, *, variant: Any = "raw") -> None:
        # Simulate cache hit for the two pre-baked tensors; miss for the third.
        # A real tensor (not object()) so the bulk handler's Profile(...) wrap
        # — which re-validates the cached dict before re-registering — passes.
        if canonical in cached_keys:
            app._session._profiles[canonical] = {0: torch.zeros(2)}
    app._session._try_autoload_vector = _autoload
    app.run_worker = MagicMock()
    app.call_from_thread = lambda fn, *a, **kw: fn(*a, **kw)
    app._refresh_left_panel = MagicMock()

    app._handle_command("/steer alice/")
    _drain_workers(app)

    assert app._alphas == {
        "alice/honest.deceptive": pytest.approx(0.5),
        "alice/warm.clinical": pytest.approx(0.5),
    }
    # Default-off — user toggles in left panel.
    assert app._enabled == {
        "alice/honest.deceptive": False,
        "alice/warm.clinical": False,
    }
    msgs = _msgs(app)
    assert "Bulk steer 'alice/'" in msgs
    assert "added 2 vector(s)" in msgs
    assert "toggled off" in msgs
    assert "Skipped 1" in msgs
    assert "alice/needs_extract" in msgs
    assert "saklas pack refresh alice -m" in msgs
    # Each loaded vector got registered with the session.
    assert app._session.steer.call_count == 2


def test_handle_steer_namespace_empty_namespace_short_circuits(monkeypatch: pytest.MonkeyPatch):
    app = _make_app()
    _stub_concepts(monkeypatch, [("default", "foo")])  # different ns
    app.run_worker = MagicMock()

    app._handle_command("/steer alice/")

    assert "No concepts installed under 'alice/'" in _msgs(app)
    app.run_worker.assert_not_called()
    assert app._alphas == {}


def test_handle_probe_namespace_bulk_loads_and_seeds_highlight(monkeypatch: pytest.MonkeyPatch):
    app = _make_app()
    _stub_concepts(monkeypatch, [
        ("alice", "calm.angry"),
        ("alice", "happy.sad"),
    ])
    app._session._profiles["alice/calm.angry"] = {0: object()}
    app._session._profiles["alice/happy.sad"] = {0: object()}
    app.run_worker = MagicMock()
    app.call_from_thread = lambda fn, *a, **kw: fn(*a, **kw)
    app._apply_highlight_to_all = MagicMock()
    app._refresh_trait_why = MagicMock()

    app._handle_command("/probe alice/")
    _drain_workers(app)

    assert app._session.probe.call_count == 2
    assert app._highlighting is True
    # Seeded to the lexicographically last loaded probe — deterministic
    # so tests don't flake on dict iteration order.
    assert app._highlight_probe == "alice/happy.sad"
    msgs = _msgs(app)
    assert "Bulk probe 'alice/'" in msgs
    assert "added 2 probe(s)" in msgs
    assert "⌃Y" in msgs


def test_handle_unsteer_namespace_removes_only_matching_prefix():
    app = _make_app()
    # Mixed registry across two namespaces — only ``alice/`` should die.
    app._alphas = {
        "alice/foo": 0.5,
        "alice/bar": 0.3,
        "default/baz": 0.4,
    }
    app._enabled = {k: True for k in app._alphas}
    app._refresh_left_panel = MagicMock()

    app._handle_command("/unsteer alice/")

    assert set(app._alphas.keys()) == {"default/baz"}
    assert set(app._enabled.keys()) == {"default/baz"}
    assert app._session.unsteer.call_count == 2
    assert "Removed 2 vector(s) from 'alice/'" in _msgs(app)


def test_handle_unsteer_namespace_empty_match_reports_clean():
    app = _make_app()
    app._alphas = {"default/baz": 0.4}
    app._enabled = {"default/baz": True}

    app._handle_command("/unsteer alice/")

    assert "No active vectors under 'alice/'" in _msgs(app)
    app._session.unsteer.assert_not_called()
    assert app._alphas == {"default/baz": 0.4}


def test_handle_unprobe_namespace_clears_highlight_when_seed_is_in_namespace():
    app = _make_app()
    app._session._monitor.probe_names = ["alice/calm", "alice/happy", "default/keep"]
    app._highlight_probe = "alice/happy"
    app._highlighting = True
    app._apply_highlight_to_all = MagicMock()
    app._refresh_trait_why = MagicMock()

    # Mutate probe_names as ``unprobe`` would so ``set_active_probes``
    # afterward observes the trimmed set.
    def _unprobe(name: Any) -> None:
        app._session._monitor.probe_names = [
            p for p in app._session._monitor.probe_names if p != name
        ]
    app._session.unprobe.side_effect = _unprobe

    app._handle_command("/unprobe alice/")

    assert app._session.unprobe.call_count == 2
    assert app._session._monitor.probe_names == ["default/keep"]
    # Highlight seed sat inside the namespace — gets dropped.
    assert app._highlight_probe is None
    assert app._highlighting is False
    assert "Removed 2 probe(s) from 'alice/'" in _msgs(app)


def test_handle_unprobe_namespace_keeps_highlight_when_seed_outside_namespace():
    app = _make_app()
    app._session._monitor.probe_names = ["alice/x", "default/keep"]
    app._highlight_probe = "default/keep"
    app._highlighting = True
    app._apply_highlight_to_all = MagicMock()
    app._refresh_trait_why = MagicMock()

    def _unprobe(name: Any) -> None:
        app._session._monitor.probe_names = [
            p for p in app._session._monitor.probe_names if p != name
        ]
    app._session.unprobe.side_effect = _unprobe

    app._handle_command("/unprobe alice/")

    # Seed wasn't in the removed namespace, so highlight state is preserved.
    assert app._highlight_probe == "default/keep"
    assert app._highlighting is True


# ---- Shift+arrow alpha step ----


# ---- Input history (↑/↓ recall) ----


class _FakeDocument:
    """Minimal stand-in for ``textual.document.Document`` — enough of
    the TextArea-side API for the input-history helpers to land cursor
    placement and read line counts.  Lines are split on ``\\n`` so an
    empty buffer reads as a single empty line (matches Textual)."""

    def __init__(self, text: str = "") -> None:
        self._lines = text.split("\n") if text else [""]

    @property
    def line_count(self) -> int:
        return len(self._lines)

    def get_line(self, row: int) -> str:
        return self._lines[row]

    def _set(self, text: str) -> None:
        self._lines = text.split("\n") if text else [""]


class _FakeInput:
    """Stand-in for the :class:`ChatInput` (TextArea subclass) exposing
    only the recall-helper surface: ``text`` (full buffer), ``load_text``
    (replace), ``cursor_location`` (``(row, col)``), and ``document``
    (line count + per-line access).  Avoids mounting a Textual app for
    unit-level coverage of ``_history_navigate`` + ``_set_input_text``.
    """

    def __init__(self, value: str = "") -> None:
        self._document = _FakeDocument(value)
        last_row = self._document.line_count - 1
        last_col = len(self._document.get_line(last_row))
        self.cursor_location: tuple[int, int] = (last_row, last_col)
        # Mirror the ChatInput flag that the pending-queue pull/restore path
        # reads and sets to allow empty-string submits for pulled slots.
        self.allow_empty_submit: bool = False

    @property
    def text(self) -> str:
        return "\n".join(self._document._lines)

    @property
    def document(self) -> _FakeDocument:
        return self._document

    def load_text(self, text: str) -> None:
        self._document._set(text)


def _wire_fake_input(app: Any, value: str = "") -> _FakeInput:
    fake = _FakeInput(value)
    app.query_one = MagicMock(return_value=fake)
    return fake


def test_push_input_history_dedupes_and_caps():
    from saklas.tui.app import _INPUT_HISTORY_MAX

    app = _make_app()

    app._push_input_history("hello")
    app._push_input_history("hello")  # exact repeat collapses
    app._push_input_history("/steer 0.5 angry")
    app._push_input_history("/steer 0.5 angry")  # exact repeat collapses
    app._push_input_history("hello")  # ping-pong: re-records

    assert app._input_history == ["hello", "/steer 0.5 angry", "hello"]

    # Empty / whitespace-only input is a no-op.
    app._push_input_history("")
    app._push_input_history("   ")
    assert app._input_history == ["hello", "/steer 0.5 angry", "hello"]

    # Cap: overflow drops oldest, keeps newest.
    overflow = [f"line{i}" for i in range(_INPUT_HISTORY_MAX + 50)]
    for line in overflow:
        app._push_input_history(line)
    assert len(app._input_history) == _INPUT_HISTORY_MAX
    assert app._input_history[-1] == overflow[-1]
    assert app._input_history[0] == overflow[-_INPUT_HISTORY_MAX]


def test_history_navigate_up_walks_back_and_stashes_draft():
    app = _make_app()
    app._input_history = ["one", "two", "three"]
    inp = _wire_fake_input(app, value="draft-in-progress")

    # First ↑: stash draft, jump to newest entry.
    app._history_navigate(-1)
    assert inp.text == "three"
    assert inp.cursor_location == (0, len("three"))
    assert app._history_index == 2
    assert app._history_stash == "draft-in-progress"

    app._history_navigate(-1)
    assert inp.text == "two"
    assert app._history_index == 1

    app._history_navigate(-1)
    assert inp.text == "one"
    assert app._history_index == 0

    # Past the oldest pins to entry 0 — bash semantics, no wrap.
    app._history_navigate(-1)
    assert inp.text == "one"
    assert app._history_index == 0


def test_history_navigate_down_restores_stash_at_bottom():
    app = _make_app()
    app._input_history = ["alpha", "beta"]
    inp = _wire_fake_input(app, value="my draft")

    # Walk up twice then back down twice — should hit the stash.
    app._history_navigate(-1)  # → "beta"
    app._history_navigate(-1)  # → "alpha"
    assert inp.text == "alpha"

    app._history_navigate(+1)  # → "beta"
    assert inp.text == "beta"
    assert app._history_index == 1

    app._history_navigate(+1)  # → restore stash, clear index
    assert inp.text == "my draft"
    assert app._history_index is None
    assert app._history_stash == ""


def test_history_navigate_down_at_live_slot_is_noop():
    app = _make_app()
    app._input_history = ["something"]
    inp = _wire_fake_input(app, value="fresh")

    app._history_navigate(+1)
    # No recall in flight — ↓ leaves the input alone.
    assert inp.text == "fresh"
    assert app._history_index is None


def test_history_navigate_empty_history_is_noop():
    app = _make_app()
    inp = _wire_fake_input(app, value="x")

    app._history_navigate(-1)
    app._history_navigate(+1)
    assert inp.text == "x"
    assert app._history_index is None


def test_user_submit_appends_to_history():
    """End-to-end: messages flowing through ``UserSubmitted`` land in
    the recall ring regardless of whether they're slash commands.
    Downstream dispatch (generation worker / slash registry) is mocked
    so the test stays at the unit level."""
    from saklas.tui.chat_panel import ChatPanel

    app = _make_app()
    # Block both branches so the recall-push side-effect is the only
    # observable behavior left to assert on.
    app._start_generation = MagicMock()
    app._handle_command = MagicMock()

    app.on_chat_panel_user_submitted(ChatPanel.UserSubmitted("hello world"))
    app.on_chat_panel_user_submitted(ChatPanel.UserSubmitted("/steer 0.5 angry"))
    app.on_chat_panel_user_submitted(ChatPanel.UserSubmitted("/steer 0.5 angry"))  # dedupe

    assert app._input_history == ["hello world", "/steer 0.5 angry"]
    # Slash commands still routed through the dispatcher; chat messages
    # still kicked off generation.  The history push doesn't replace
    # either downstream path.
    app._start_generation.assert_called_once_with("hello world")
    assert app._handle_command.call_count == 2


# ---------------------------------------------------------------------------
# Pending queue + ↑/↓ pull-and-edit
# ---------------------------------------------------------------------------


def test_history_navigate_walks_pending_then_history():
    """``↑`` walks the queue (most-recent first) before falling into
    committed input history.  Pending positions land on
    ``_pulled_slot``; history positions land on ``_history_index``."""
    from saklas.tui.chat_panel import PendingItem

    app = _make_app()
    app._input_history = ["older"]
    app._pending_queue = [
        PendingItem("submit", "first queued"),
        PendingItem("submit", "second queued"),
    ]
    inp = _wire_fake_input(app, value="composing")

    # First ↑ — most-recent pending.
    app._history_navigate(-1)
    assert inp.text == "second queued"
    assert app._pulled_slot == 1
    assert app._history_index is None
    assert app._history_stash == "composing"
    assert inp.allow_empty_submit is True

    # Second ↑ — earlier pending.
    app._history_navigate(-1)
    assert inp.text == "first queued"
    assert app._pulled_slot == 0
    assert inp.allow_empty_submit is True

    # Third ↑ — falls into history.
    app._history_navigate(-1)
    assert inp.text == "older"
    assert app._pulled_slot is None
    assert app._history_index == 0
    assert inp.allow_empty_submit is False

    # Fourth ↑ — clamps at the oldest history entry.
    app._history_navigate(-1)
    assert inp.text == "older"


def test_history_navigate_down_returns_through_pending_to_live():
    """``↓`` walks back through pending and restores the stash at live."""
    from saklas.tui.chat_panel import PendingItem

    app = _make_app()
    app._pending_queue = [
        PendingItem("submit", "alpha"),
        PendingItem("submit", "beta"),
    ]
    inp = _wire_fake_input(app, value="composing")

    app._history_navigate(-1)  # → "beta" (slot 1)
    app._history_navigate(-1)  # → "alpha" (slot 0)
    assert app._pulled_slot == 0

    app._history_navigate(+1)  # → "beta" (slot 1)
    assert inp.text == "beta"
    assert app._pulled_slot == 1

    app._history_navigate(+1)  # → restore stash
    assert inp.text == "composing"
    assert app._pulled_slot is None
    assert app._history_index is None
    assert inp.allow_empty_submit is False


def test_pulled_pending_resubmit_replaces_slot_in_place():
    """``Enter`` after editing a pulled slot replaces *that* slot rather
    than appending to the queue tail — slot-preserving edit."""
    from saklas.tui.chat_panel import ChatPanel, PendingItem

    app = _make_app()
    app._session.is_generating = True  # busy so submit enqueues
    app._pending_queue = [
        PendingItem("submit", "a"),
        PendingItem("submit", "b"),
        PendingItem("submit", "c"),
    ]
    _wire_fake_input(app, value="")
    # Simulate the user having pulled slot 1 ("b") via ↑↑.
    app._pulled_slot = 1

    app.on_chat_panel_user_submitted(ChatPanel.UserSubmitted("B prime"))

    # Slot 1 replaced; order preserved.
    assert [p.text for p in app._pending_queue] == ["a", "B prime", "c"]
    # Pull state cleared.
    assert app._pulled_slot is None


def test_pulled_pending_empty_enter_removes_slot():
    """Empty ``Enter`` while a slot is pulled removes that slot —
    keyboard equivalent of the GUI's per-bubble ``×``."""
    from saklas.tui.chat_panel import ChatPanel, PendingItem

    app = _make_app()
    app._pending_queue = [
        PendingItem("submit", "keep me"),
        PendingItem("submit", "cancel me"),
    ]
    _wire_fake_input(app, value="")
    app._pulled_slot = 1

    app.on_chat_panel_user_submitted(ChatPanel.UserSubmitted(""))

    assert [p.text for p in app._pending_queue] == ["keep me"]
    assert app._pulled_slot is None


def test_pulled_pending_esc_cancels_pull_without_removing():
    """``Esc`` while pulled cancels the *edit* — the slot stays in the
    queue, the input restores its pre-pull stash."""
    from saklas.tui.chat_panel import PendingItem

    app = _make_app()
    app._pending_queue = [PendingItem("submit", "queued")]
    inp = _wire_fake_input(app, value="composing")

    app._history_navigate(-1)  # pull slot 0
    assert app._pulled_slot == 0
    assert inp.text == "queued"

    app.action_stop_generation()  # no gen running → cancel pull
    assert app._pulled_slot is None
    assert inp.text == "composing"
    assert app._pending_queue == [PendingItem("submit", "queued")]
    assert inp.allow_empty_submit is False


def test_drain_next_pending_decrements_pulled_slot():
    """When the queue head drains during a pull, the pulled-slot index
    shifts so the user keeps tracking the same item."""
    from saklas.tui.chat_panel import PendingItem

    app = _make_app()
    app._pending_queue = [
        PendingItem("submit", "head"),
        PendingItem("submit", "middle"),
        PendingItem("submit", "tail"),
    ]
    _wire_fake_input(app, value="")
    app._pulled_slot = 2  # user is editing "tail"
    # Block the dispatch so we only see the slot accounting.
    app._dispatch_pending_action = MagicMock()

    app._drain_next_pending()
    assert [p.text for p in app._pending_queue] == ["middle", "tail"]
    assert app._pulled_slot == 1  # still on "tail" — index slid down


def test_drain_next_pending_cancels_pull_when_head_was_pulled():
    """When the user pulled slot 0, the drain pops that very item —
    cancel the pull so the stale ``_pulled_slot`` doesn't outlive the
    queue mutation."""
    from saklas.tui.chat_panel import PendingItem

    app = _make_app()
    app._pending_queue = [
        PendingItem("submit", "about to fire"),
        PendingItem("submit", "next up"),
    ]
    inp = _wire_fake_input(app, value="draft")
    app._history_stash = "draft"
    app._pulled_slot = 0
    app._dispatch_pending_action = MagicMock()

    app._drain_next_pending()
    assert [p.text for p in app._pending_queue] == ["next up"]
    assert app._pulled_slot is None
    # Pull cancelled — input restored from stash.
    assert inp.text == "draft"


def test_pending_strip_markup_round_trips_through_rich():
    """Direct check that ``PendingStrip`` builds well-formed Rich
    markup for every pending kind, including the pulled-slot
    highlight and item text containing brackets / newlines.  Catches
    a v2.x regression where ``[[`` was used as a literal-bracket
    escape and tripped ``MarkupError: auto closing tag ('[/]') has
    nothing to close`` when the strip first re-rendered."""
    from rich.console import Console
    from rich.text import Text
    from saklas.tui.chat_panel import PendingItem, PendingStrip
    import io

    # Side-step Textual's mount lifecycle by calling the markup
    # builder via Static.update with a captured update target.
    strip = object.__new__(PendingStrip)
    captured: list[str] = []
    strip.update = lambda s: captured.append(s)  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]  # Textual Static.update stub; lambda captures VisualType → str
    strip.add_class = lambda _c: None  # pyright: ignore[reportAttributeAccessIssue]  # Textual Widget.add_class stub
    strip.remove_class = lambda _c: None  # pyright: ignore[reportAttributeAccessIssue]  # Textual Widget.remove_class stub
    strip._queue = []

    items = [
        PendingItem("submit", "what do you think?"),
        PendingItem("clear", "/clear"),
        PendingItem("steer", "/steer 0.5 angry"),
        PendingItem("submit", "with [brackets] and \\backslashes"),
        PendingItem("submit", "multi\nline\nmessage"),
    ]
    for slot in [None, 0, 2, len(items) - 1]:
        PendingStrip.update_queue(strip, items, pulled_slot=slot)
        # Parsing through Text.from_markup raises MarkupError on bad
        # markup — the assertion is "no raise."
        Console(file=io.StringIO(), force_terminal=True).print(
            Text.from_markup(captured[-1])
        )


def test_slash_command_during_gen_enqueues_canonical_text():
    """Mid-gen ``/clear`` enqueues a :class:`PendingItem` carrying the
    full slash text so the user can pull and edit it via ↑.  The
    in-flight gen is not stopped — queue model preserves tokens."""
    from saklas.tui.chat_panel import PendingItem

    app = _make_app()
    app._session.is_generating = True
    app._session.stop = MagicMock()

    app._handle_command("/clear")

    assert app._pending_queue == [PendingItem("clear", "/clear")]
    app._session.stop.assert_not_called()


def test_shift_arrow_uses_coarse_alpha_step():
    """Holding shift with ←/→ nudges alpha by 0.1 instead of 0.01."""
    from saklas.tui.app import _ALPHA_STEP_FINE, _ALPHA_STEP_COARSE

    # Sanity: the constants encode the documented fine/coarse split.
    assert _ALPHA_STEP_FINE == pytest.approx(0.01)
    assert _ALPHA_STEP_COARSE == pytest.approx(0.1)

    app = _make_app()
    app._refresh_left_panel = MagicMock()
    app._left_panel.get_selected = MagicMock(return_value={"name": "honest"})
    app._alphas = {"honest": 0.0}

    # Plain right arrow path: 0.01 step.
    app.action_nav_right()
    assert app._alphas["honest"] == pytest.approx(_ALPHA_STEP_FINE)

    # Shift+right path: 0.1 step (10× coarser).
    app._adjust_alpha(_ALPHA_STEP_COARSE)
    assert app._alphas["honest"] == pytest.approx(0.11)

    # Shift+left undoes the shift+right exactly.
    app._adjust_alpha(-_ALPHA_STEP_COARSE)
    assert app._alphas["honest"] == pytest.approx(_ALPHA_STEP_FINE)


# ---- Logit-pass: highlight mode cycle (Ctrl+Y) ----


def test_highlight_cycle_off_to_probe_to_surprise():
    """Ctrl+Y walks {off → probe → surprise → off} with a probe loaded.

    The cycle defers to the trait-panel selection for the ``probe``
    slot so navigating the right rack still drives WHICH probe lights
    up — Ctrl+Y only switches between "off / a probe / surprise".
    """
    from saklas.tui.chat_panel import SURPRISE_PROBE

    app = _make_app()
    # Pretend a probe is loaded and trait-panel-selected.
    app._trait_panel.get_selected_probe = MagicMock(return_value="angry.calm")
    app._apply_highlight_to_all = MagicMock()

    # Start at off.
    assert app._highlighting is False

    # off → probe.  The cycle no longer emits a chat message — mode is
    # surfaced by the persistent HL line in the left panel instead — so
    # the assertions track ``_highlighting`` / ``_highlight_probe``.
    app.action_cycle_highlight_mode()
    assert app._highlighting is True
    assert app._highlight_probe == "angry.calm"

    # probe → surprise
    app.action_cycle_highlight_mode()
    assert app._highlighting is True
    assert app._highlight_probe == SURPRISE_PROBE

    # surprise → off
    app.action_cycle_highlight_mode()
    assert app._highlighting is False


def test_highlight_cycle_backward_walks_reverse():
    """Ctrl+Shift+Y walks the cycle backward from any state."""
    from saklas.tui.chat_panel import SURPRISE_PROBE

    app = _make_app()
    app._trait_panel.get_selected_probe = MagicMock(return_value="warm.clinical")
    app._apply_highlight_to_all = MagicMock()

    # off → surprise (backward)
    app.action_cycle_highlight_mode_back()
    assert app._highlight_probe == SURPRISE_PROBE
    assert app._highlighting is True

    # surprise → probe (backward)
    app.action_cycle_highlight_mode_back()
    assert app._highlight_probe == "warm.clinical"

    # probe → off (backward)
    app.action_cycle_highlight_mode_back()
    assert app._highlighting is False


def test_left_panel_highlight_line_renders():
    """``LeftPanel.update_highlight`` puts an ``HL`` line in the
    GENERATION block: ``off`` dimmed, probe names verbatim, long names
    truncated to the 15-char preview budget."""
    from saklas.tui.vector_panel import LeftPanel

    panel: Any = LeftPanel(model_info={})
    captured: list[str] = []
    panel._gen_config_widget = SimpleNamespace(update=captured.append)

    panel.update_highlight("off")
    assert "HL" in captured[-1] and "off" in captured[-1]

    panel.update_highlight("angry.calm")
    assert "angry.calm" in captured[-1]

    panel.update_highlight("surprise")
    assert "surprise" in captured[-1]

    # Long probe names truncate like the Sys-prompt preview.
    panel.update_highlight("high_context.low_context")
    assert "high_context.lo..." in captured[-1]
    assert "high_context.low_context" not in captured[-1]


def test_highlight_cycle_skips_probe_when_none_selectable():
    """With no probes loaded, ``probe`` slot is skipped so the cycle
    collapses to {off ↔ surprise} rather than getting stuck."""
    from saklas.tui.chat_panel import SURPRISE_PROBE

    app = _make_app()
    # No trait-panel selection and no stored seed — probe slot has
    # nothing to anchor to.
    app._trait_panel.get_selected_probe = MagicMock(return_value=None)
    app._apply_highlight_to_all = MagicMock()
    app._highlight_probe = None

    # off → (probe-skip) → surprise
    app.action_cycle_highlight_mode()
    assert app._highlight_probe == SURPRISE_PROBE
    assert app._highlighting is True

    # surprise → off
    app.action_cycle_highlight_mode()
    assert app._highlighting is False

    # Backward direction also skips cleanly.
    app.action_cycle_highlight_mode_back()
    assert app._highlight_probe == SURPRISE_PROBE


def test_apply_highlight_to_all_preserves_surprise_sentinel():
    """Trait-panel arrow keys must not clobber the SURPRISE_PROBE
    sentinel back to a probe — that latent bug from the Phase 3 pass
    would have flipped surprise mode off the moment the user moved
    in the right rack."""
    from saklas.tui.chat_panel import SURPRISE_PROBE

    app = _make_app()
    # Trait panel has a different probe selected — without the
    # surprise-guard this would clobber the sentinel.
    app._trait_panel.get_selected_probe = MagicMock(return_value="angry.calm")
    app._highlight_probe = SURPRISE_PROBE
    app._highlighting = True
    app._assistant_messages = []  # no widgets to apply to

    app._apply_highlight_to_all()

    # Sentinel survived; trait-panel selection was ignored under
    # surprise mode.
    assert app._highlight_probe == SURPRISE_PROBE


# ---------------------------------------------------------------------------
# Queue role-awareness — predicted active-node role flips the input mode
# ---------------------------------------------------------------------------


def test_predicted_on_user_node_falls_through_to_live_when_queue_empty():
    """No queued items → predicted equals live."""
    from saklas.tui.chat_panel import PendingItem

    app = _make_app()
    # Live active is the synthetic root (system) — not a user node.
    assert app._prefill_target_node_id() is None
    assert app._predicted_on_user_node() is False

    # A queued ``/steer`` doesn't shift the role, so prediction still
    # mirrors live.
    app._pending_queue = [PendingItem("steer", "/steer 0.5 angry", ("0.5 angry",))]
    assert app._predicted_on_user_node() is False


def test_predicted_on_user_node_reflects_queued_commit_user():
    """A queued ``commit_user`` predicts the next submission as prefill."""
    from saklas.tui.chat_panel import PendingItem

    app = _make_app()
    # Live active is root — not a user node.
    assert app._predicted_on_user_node() is False

    # Queue a commit_user — the next item should land in prefill mode.
    app._pending_queue = [PendingItem("commit_user", "hi")]
    assert app._predicted_on_user_node() is True

    # Add a commit_assistant on top — final role flips back to assistant.
    app._pending_queue.append(PendingItem("commit_assistant", "hello", ("uid",)))
    assert app._predicted_on_user_node() is False


def test_predicted_walks_past_no_change_kinds():
    """Items with no role mapping (``/steer``, ``/probe``) are walked past
    so a queued ``commit_user`` followed by ``/steer`` still predicts
    user mode."""
    from saklas.tui.chat_panel import PendingItem

    app = _make_app()
    app._pending_queue = [
        PendingItem("commit_user", "hi"),
        PendingItem("steer", "/steer 0.5 angry", ("0.5 angry",)),
        PendingItem("probe", "/probe calm", ("calm",)),
    ]
    assert app._predicted_on_user_node() is True


def test_enqueue_pending_refreshes_input_mode():
    """Enqueueing a role-shifting item updates the placeholder
    immediately — no need to wait for the queue to drain."""
    from saklas.tui.chat_panel import PendingItem

    app = _make_app()
    set_prefill_mode = MagicMock()
    app._chat_panel.set_prefill_mode = set_prefill_mode

    app._enqueue_pending(PendingItem("commit_user", "hi"))
    # Last call reflects the queue-aware mode.
    set_prefill_mode.assert_called_with(True)

    app._enqueue_pending(PendingItem("commit_assistant", "hello", ("uid",)))
    set_prefill_mode.assert_called_with(False)


def test_remove_pending_slot_refreshes_input_mode():
    """Cancelling the queued role-shifter restores live mode."""
    from saklas.tui.chat_panel import PendingItem

    app = _make_app()
    app._pending_queue = [PendingItem("commit_user", "hi")]
    set_prefill_mode = MagicMock()
    app._chat_panel.set_prefill_mode = set_prefill_mode

    app._remove_pending_slot(0)
    # After removal, prediction falls back to live (root, not user).
    set_prefill_mode.assert_called_with(False)


# ---------------------------------------------------------------------------
# Pending-queue drain chaining — sync kinds chain inline, async kinds break
# ---------------------------------------------------------------------------


def test_drain_next_pending_chains_through_sync_kinds():
    """A run of sync slash kinds (/clear /steer /probe /rewind) drains
    all in one call — the old single-item drain would have left them
    stuck waiting for a ``done`` that never arrives."""
    from saklas.tui.chat_panel import PendingItem

    app = _make_app()
    app._pending_queue = [
        PendingItem("steer", "/steer 0.5 angry", ("0.5 angry",)),
        PendingItem("probe", "/probe calm", ("calm",)),
        PendingItem("clear", "/clear"),
    ]
    _wire_fake_input(app, value="")
    dispatched: list[str] = []
    app._dispatch_pending_action = MagicMock(
        side_effect=lambda item: dispatched.append(item.kind),
    )

    app._drain_next_pending()
    # All three drained in one call — chain-inline behavior.
    assert dispatched == ["steer", "probe", "clear"]
    assert app._pending_queue == []


def test_drain_next_pending_breaks_at_first_async_kind():
    """Drain chains through sync kinds but breaks at the first kind
    that runs a worker / kicks a gen — that one's own ``done`` will
    advance the queue, so chaining past it would race the worker."""
    from saklas.tui.chat_panel import PendingItem

    app = _make_app()
    app._pending_queue = [
        PendingItem("clear", "/clear"),                # sync — chain
        PendingItem("commit_user", "hi"),              # async — break here
        PendingItem("submit", "next"),                 # stays queued
    ]
    _wire_fake_input(app, value="")
    dispatched: list[str] = []
    app._dispatch_pending_action = MagicMock(
        side_effect=lambda item: dispatched.append(item.kind),
    )

    app._drain_next_pending()
    # Only the first two ran; submit waits for commit_user's done.
    assert dispatched == ["clear", "commit_user"]
    assert [p.kind for p in app._pending_queue] == ["submit"]


def test_drain_next_pending_handles_pure_async_chain():
    """When the head is async, drain pops exactly one — matches the
    pre-existing single-item semantics every gen-bearing kind
    depends on."""
    from saklas.tui.chat_panel import PendingItem

    app = _make_app()
    app._pending_queue = [
        PendingItem("submit", "first"),
        PendingItem("submit", "second"),
    ]
    _wire_fake_input(app, value="")
    app._dispatch_pending_action = MagicMock()

    app._drain_next_pending()
    assert [p.text for p in app._pending_queue] == ["second"]


# ---------------------------------------------------------------------------
# /manifold fit dispatch
# ---------------------------------------------------------------------------


def test_manifold_command_rejects_unknown_subverb():
    """`/manifold` only knows the ``fit`` subverb."""
    app = _make_app()
    app._handle_command("/manifold show foo")
    assert "Usage: /manifold fit" in _msgs(app)


def test_manifold_fit_missing_folder_reports(tmp_path: Path) -> None:
    """`/manifold fit` on a folder with no manifold.json reports cleanly
    without firing a worker."""
    app = _make_app()
    app.run_worker = MagicMock()
    app._handle_command(f"/manifold fit {tmp_path}")
    assert "no manifold.json" in _msgs(app)
    app.run_worker.assert_not_called()


def test_manifold_fit_runs_session_extract_manifold(tmp_path: Path) -> None:
    """`/manifold fit <folder>` with a manifold.json present routes
    through ``session.extract_manifold`` on a worker."""
    (tmp_path / "manifold.json").write_text("{}")
    app = _make_app()

    fitted = SimpleNamespace(
        name="circumplex", layers=[0, 1, 2], node_labels=["a", "b", "c"],
    )
    captured: dict[str, Any] = {}

    def _fake_extract_manifold(folder: Any, **kwargs: Any) -> Any:
        captured["folder"] = folder
        return fitted
    app._session.extract_manifold = _fake_extract_manifold
    app.run_worker = lambda fn, thread=True: fn()
    app.call_from_thread = lambda fn, *a, **kw: fn(*a, **kw)

    app._handle_command(f"/manifold fit {tmp_path}")

    assert "folder" in captured
    assert str(captured["folder"]) == str(tmp_path)
    assert "fitted manifold 'circumplex'" in _msgs(app)


def test_manifold_fit_mid_gen_enqueues_pending():
    """`/manifold fit` while a generation is in flight queues a
    ``manifold_fit`` PendingItem rather than running immediately."""
    from saklas.tui.chat_panel import PendingItem

    app = _make_app()
    app._session.is_generating = True
    app._handle_command("/manifold fit /tmp/myfold")
    assert app._pending_queue == [
        PendingItem("manifold_fit", "/manifold fit /tmp/myfold", ("/tmp/myfold",))
    ]


# ---------------------------------------------------------------------------
# /steer manifold terms
# ---------------------------------------------------------------------------


def test_steer_manifold_term_validates_and_registers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """A ``%`` manifold term in ``/steer`` routes through eager artifact
    validation and lands on ``_manifold_terms`` — not the scalar
    ``_alphas`` dict."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    from saklas.io import selectors as _sel
    _sel.invalidate()

    app = _make_app()
    app._refresh_left_panel = MagicMock()

    # Stub the session manifold-load + registry so validation passes.
    domain = SimpleNamespace(intrinsic_dim=2)
    manifold = SimpleNamespace(domain=domain)

    def _ensure(key: Any) -> None:
        app._session._manifolds[key] = manifold
    app._session._ensure_manifold_loaded = _ensure
    app._session._manifolds = {}
    app.run_worker = lambda fn, thread=True: fn()
    app.call_from_thread = lambda fn, *a, **kw: fn(*a, **kw)

    app._handle_command("/steer 0.7 circumplex%0.3,0.8")

    assert "circumplex%0.3,0.8" in app._manifold_terms
    term = app._manifold_terms["circumplex%0.3,0.8"]
    assert term.position == (0.3, 0.8)
    assert app._alphas == {}  # not a scalar vector
    assert app._enabled.get("circumplex%0.3,0.8") is True


def test_steer_manifold_term_arity_mismatch_reports(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """A position with the wrong coordinate count is rejected against the
    loaded manifold's domain."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    from saklas.io import selectors as _sel
    _sel.invalidate()

    app = _make_app()
    # Domain is 2-D but the position has 3 coords.
    domain = SimpleNamespace(intrinsic_dim=2)
    manifold = SimpleNamespace(domain=domain)

    def _ensure(key: Any) -> None:
        app._session._manifolds[key] = manifold
    app._session._ensure_manifold_loaded = _ensure
    app._session._manifolds = {}
    app.run_worker = lambda fn, thread=True: fn()
    app.call_from_thread = lambda fn, *a, **kw: fn(*a, **kw)

    app._handle_command("/steer 0.5 circumplex%0.1,0.2,0.3")

    assert "2-dimensional" in _msgs(app)
    assert app._manifold_terms == {}


def test_steer_manifold_unregistered_reports(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """An unfitted manifold surfaces ``ManifoldNotRegisteredError`` as a
    system message rather than crashing the worker."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    from saklas.io import selectors as _sel
    _sel.invalidate()
    from saklas.core.session import ManifoldNotRegisteredError

    app = _make_app()
    app._session._manifolds = {}

    def _ensure(key: Any) -> None:
        raise ManifoldNotRegisteredError(
            f"manifold '{key}' has no fitted tensor"
        )
    app._session._ensure_manifold_loaded = _ensure
    app.run_worker = lambda fn, thread=True: fn()
    app.call_from_thread = lambda fn, *a, **kw: fn(*a, **kw)

    app._handle_command("/steer 0.5 ghost%0.5")

    assert "no fitted tensor" in _msgs(app)
    assert app._manifold_terms == {}


def test_steer_mixed_expression_applies_vector_siblings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """A mixed expression (vector + manifold) still applies its scalar
    vector term — the manifold term uses ``continue``, not ``return``."""
    import torch
    from saklas.core.profile import Profile
    from saklas.io import selectors as _sel
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    _sel.invalidate()

    app = _make_app()
    app._refresh_left_panel = MagicMock()

    def _fake_extract(concept: Any, **kwargs: Any) -> Any:
        return concept, Profile({0: torch.zeros(4)})
    app._session.extract = _fake_extract

    domain = SimpleNamespace(intrinsic_dim=1)
    manifold = SimpleNamespace(domain=domain)

    def _ensure(key: Any) -> None:
        app._session._manifolds[key] = manifold
    app._session._ensure_manifold_loaded = _ensure
    app._session._manifolds = {}
    app.run_worker = lambda fn, thread=True: fn()
    app.call_from_thread = lambda fn, *a, **kw: fn(*a, **kw)

    app._handle_command("/steer 0.5 honest + 0.7 circumplex%0.4")

    # The vector sibling landed despite the manifold term in the same
    # expression.
    assert "honest" in app._alphas
    assert "circumplex%0.4" in app._manifold_terms


def test_active_alphas_merges_manifold_terms(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """``_active_alphas`` merges scalar vectors and manifold terms,
    honoring the shared ``_enabled`` flag."""
    from saklas.core.steering_expr import ManifoldTerm
    from saklas.core.triggers import Trigger

    app = _make_app()
    app._alphas = {"honest": 0.5, "warm": 0.3}
    term = ManifoldTerm(
        coeff=0.7, trigger=Trigger.BOTH, manifold="circumplex",
        position=(0.3, 0.8),
    )
    app._manifold_terms = {"circumplex%0.3,0.8": term}
    app._enabled = {
        "honest": True, "warm": False, "circumplex%0.3,0.8": True,
    }

    merged = app._active_alphas()
    assert merged == {"honest": 0.5, "circumplex%0.3,0.8": term}

    # Disabling the manifold drops it from the merged dict.
    app._enabled["circumplex%0.3,0.8"] = False
    assert "circumplex%0.3,0.8" not in app._active_alphas()


# ---------------------------------------------------------------------------
# /pairs custom-statement extraction
# ---------------------------------------------------------------------------


def test_pairs_command_requires_name():
    app = _make_app()
    app._handle_command("/pairs")
    assert "Usage: /pairs" in _msgs(app)


def test_pairs_command_refused_mid_gen():
    """`/pairs` opens a modal — it cannot be queued, so it's refused
    while a generation is in flight rather than enqueued."""
    app = _make_app()
    app._session.is_generating = True
    app._handle_command("/pairs mood")
    assert "modal" in _msgs(app)
    assert app._pending_queue == []


def test_pairs_command_pushes_modal():
    """`/pairs <name>` pushes a ``CustomPairsModal``."""
    from saklas.tui.pairs_modal import CustomPairsModal

    app = _make_app()
    pushed = {}
    app.push_screen = lambda screen, callback=None: pushed.update(
        screen=screen, callback=callback,
    )
    app._handle_command("/pairs mood")
    assert isinstance(pushed["screen"], CustomPairsModal)


def test_pairs_modal_parses_pair_lines():
    """The modal's line parser splits on ``|`` and rejects malformed
    lines."""
    from saklas.tui.pairs_modal import parse_pair_lines

    pairs, errors = parse_pair_lines(
        "i am happy | i am sad\n"
        "\n"
        "calm words | angry words\n"
    )
    assert pairs == [
        ("i am happy", "i am sad"),
        ("calm words", "angry words"),
    ]
    assert errors == []

    # Malformed: no delimiter, empty side, two delimiters.
    pairs, errors = parse_pair_lines(
        "no delimiter here\n"
        "good | \n"
        "a | b | c\n"
    )
    assert pairs == []
    assert len(errors) == 3
    assert "line 1" in errors[0]


# ---------------------------------------------------------------------------
# /manifold-probe attach / remove + trait panel rendering
# ---------------------------------------------------------------------------


def test_manifold_probe_requires_selector():
    app = _make_app()
    app._handle_command("/manifold-probe")
    assert "Usage: /manifold-probe" in _msgs(app)


def test_manifold_probe_remove_requires_name():
    app = _make_app()
    app._handle_command("/manifold-probe-remove")
    assert "Usage: /manifold-probe-remove" in _msgs(app)


def test_manifold_probe_attach_routes_through_session(monkeypatch: pytest.MonkeyPatch):
    """``/manifold-probe <selector>`` calls ``session.add_manifold_probe``
    on a worker; on success the trait panel is refreshed with the
    attached probe's manifold artifact."""
    app = _make_app()
    captured = {}

    def _fake_add(selector: Any, **kwargs: Any) -> Any:
        captured["selector"] = selector
        # Pretend the session registered the probe under the selector
        # name and populated the monitor.
        manifold = SimpleNamespace(
            domain=SimpleNamespace(intrinsic_dim=1),
            node_labels=["a"], node_coords=None,
        )
        app._session._manifold_monitor.probe_names = [selector]
        app._session._manifold_monitor.attached_probes = MagicMock(
            return_value={selector: SimpleNamespace(manifold=manifold)},
        )
        return selector
    app._session.add_manifold_probe = _fake_add
    app._trait_panel.set_active_manifold_probes = MagicMock()
    app.run_worker = lambda fn, thread=True: fn()
    app.call_from_thread = lambda fn, *a, **kw: fn(*a, **kw)

    app._handle_command("/manifold-probe circumplex")

    assert captured["selector"] == "circumplex"
    app._trait_panel.set_active_manifold_probes.assert_called()
    pushed = app._trait_panel.set_active_manifold_probes.call_args.args[0]
    assert "circumplex" in pushed
    assert "Manifold probe 'circumplex' active" in _msgs(app)


def test_manifold_probe_remove_routes_through_session():
    app = _make_app()
    app._session._manifold_monitor.probe_names = ["circumplex"]
    app._session._manifold_monitor.attached_probes = MagicMock(return_value={})

    def _fake_remove(name: Any) -> None:
        app._session._manifold_monitor.probe_names = []
    app._session.remove_manifold_probe = _fake_remove
    app._trait_panel.set_active_manifold_probes = MagicMock()

    app._handle_command("/manifold-probe-remove circumplex")

    assert app._session._manifold_monitor.probe_names == []
    app._trait_panel.set_active_manifold_probes.assert_called_with({})
    assert "removed" in _msgs(app).lower()


def test_manifold_probe_remove_missing_reports():
    app = _make_app()
    app._session._manifold_monitor.probe_names = []
    app._handle_command("/manifold-probe-remove ghost")
    assert "not active" in _msgs(app)


def test_manifold_probe_mid_gen_enqueues_pending():
    from saklas.tui.chat_panel import PendingItem

    app = _make_app()
    app._session.is_generating = True
    app._handle_command("/manifold-probe circumplex")
    assert app._pending_queue == [
        PendingItem(
            "manifold_probe", "/manifold-probe circumplex", ("circumplex",),
        ),
    ]


def test_manifold_probe_during_ab_shadow_is_refused():
    app = _make_app()
    app._ab_shadow_active = True
    app._handle_command("/manifold-probe circumplex")
    assert "A/B shadow" in _msgs(app)
    assert app._pending_queue == []


def test_pull_manifold_aggregates_pushes_from_last_result():
    """``_finalize_widget_highlight`` calls ``_pull_manifold_aggregates``,
    which reads ``session.last_result.manifold_readings`` and pushes the
    aggregate map to the trait panel."""
    from saklas.core.results import GenerationResult, ManifoldAggregate

    app = _make_app()
    agg = ManifoldAggregate(
        fraction_mean=0.4,
        fraction_per_layer={0: 0.4},
        nearest=[("happy", 0.1), ("calm", 0.2)],
        coords=(0.3, 0.5),
        coords_per_layer={0: (0.3, 0.5)},
        residual_mean=0.05,
        residual_per_layer={0: 0.05},
    )
    result = GenerationResult(
        text="x", tokens=[], token_count=0, tok_per_sec=0.0, elapsed=0.0,
        manifold_readings={"circumplex": agg},
    )
    app._session.last_result = result
    # Pretend a probe is attached so the early-out guard lets us through.
    app._session._manifold_monitor.probe_names = ["circumplex"]
    app._trait_panel.update_manifold_readings = MagicMock()

    app._pull_manifold_aggregates()

    app._trait_panel.update_manifold_readings.assert_called_once_with(
        aggregates={"circumplex": agg},
    )


def test_trait_panel_renders_manifold_section_empty_state():
    """An attached probe with no readings renders the bar at zero and
    the nearest-list as empty without crashing."""
    from saklas.tui.trait_panel import TraitPanel

    panel: Any = object.__new__(TraitPanel)
    panel._categories = {}
    panel._current_values = {}
    panel._previous_values = {}
    panel._sparklines = {}
    panel._active_probes = set()
    panel._sort_mode = "name"
    panel._nav_items = []
    panel._nav_idx = 0
    panel._cached_render_text = ""
    panel._manifold_probes = {}
    panel._manifold_readings = {}
    panel._manifold_aggregates = {}
    panel._cached_manifold_text = ""
    # Capture writes to the static targets.
    header_writes: list[str] = []
    content_writes: list[str] = []
    panel._manifold_header = SimpleNamespace(update=header_writes.append)
    panel._manifold_content = SimpleNamespace(update=content_writes.append)

    # Empty registry — section is hidden.
    panel._render_manifold_probes()
    assert header_writes[-1] == ""
    assert content_writes[-1] == ""

    # Register a 1-D BoxDomain manifold without readings.
    manifold = SimpleNamespace(
        domain=SimpleNamespace(intrinsic_dim=1),
        node_labels=["a"], node_coords=None,
    )
    panel._manifold_probes = {"toy": manifold}
    panel._render_manifold_probes()
    # Header appears, content has the probe name and a zero-filled bar.
    assert "MANIFOLD PROBES" in header_writes[-1]
    assert "toy" in content_writes[-1]
    assert "0.00" in content_writes[-1]


def test_trait_panel_renders_manifold_minimap_for_2d_box():
    """A 2-D BoxDomain manifold draws an ASCII mini-map.  The coord dot
    from the aggregate lands on a row that contains the ``●`` marker."""
    import torch
    from saklas.core.results import ManifoldAggregate
    from saklas.tui.trait_panel import TraitPanel

    panel: Any = object.__new__(TraitPanel)
    panel._categories = {}
    panel._current_values = {}
    panel._previous_values = {}
    panel._sparklines = {}
    panel._active_probes = set()
    panel._sort_mode = "name"
    panel._nav_items = []
    panel._nav_idx = 0
    panel._cached_render_text = ""
    panel._manifold_probes = {}
    panel._manifold_readings = {}
    panel._manifold_aggregates = {}
    panel._cached_manifold_text = ""
    header_writes: list[str] = []
    content_writes: list[str] = []
    panel._manifold_header = SimpleNamespace(update=header_writes.append)
    panel._manifold_content = SimpleNamespace(update=content_writes.append)

    # Russell-style 2-D box: valence x arousal, each in [-1, 1].
    ax = SimpleNamespace(periodic=False, period=1.0, lo=-1.0, hi=1.0)
    domain = SimpleNamespace(intrinsic_dim=2, axes=(ax, ax))
    # Five nodes: corners + origin.  Use a real tensor so .tolist() works.
    coords = torch.tensor([
        [-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0], [0.0, 0.0],
    ])
    manifold = SimpleNamespace(
        domain=domain, node_labels=["a", "b", "c", "d", "e"],
        node_coords=coords,
    )
    panel._manifold_probes = {"circ": manifold}
    panel._manifold_aggregates = {"circ": ManifoldAggregate(
        fraction_mean=0.5,
        fraction_per_layer={0: 0.5},
        nearest=[("e", 0.1)],
        coords=(0.0, 0.0),  # origin — center of map
        coords_per_layer={0: (0.0, 0.0)},
        residual_mean=0.0,
        residual_per_layer={0: 0.0},
    )}

    panel._render_manifold_probes()
    rendered = content_writes[-1]
    # Mini-map markers present.
    assert "●" in rendered  # coord dot
    assert "·" in rendered  # at least one node marker
    # Bar reading shows the fraction.
    assert "0.50" in rendered


def test_trait_panel_skips_minimap_for_higher_dim():
    """An 8-D CustomDomain (like ``personas``) renders the bar + labels
    but no mini-map."""
    from saklas.core.results import ManifoldTokenReading
    from saklas.tui.trait_panel import TraitPanel

    panel: Any = object.__new__(TraitPanel)
    panel._categories = {}
    panel._current_values = {}
    panel._previous_values = {}
    panel._sparklines = {}
    panel._active_probes = set()
    panel._sort_mode = "name"
    panel._nav_items = []
    panel._nav_idx = 0
    panel._cached_render_text = ""
    panel._manifold_probes = {}
    panel._manifold_readings = {}
    panel._manifold_aggregates = {}
    panel._cached_manifold_text = ""
    content_writes: list[str] = []
    panel._manifold_header = SimpleNamespace(update=lambda _t: None)
    panel._manifold_content = SimpleNamespace(update=content_writes.append)

    domain = SimpleNamespace(intrinsic_dim=8)  # no axes attribute
    manifold = SimpleNamespace(
        domain=domain, node_labels=["hacker", "pirate"], node_coords=None,
    )
    panel._manifold_probes = {"personas": manifold}
    panel._manifold_readings = {"personas": ManifoldTokenReading(
        fraction=0.3, nearest=[("hacker", 0.1)],
    )}
    panel._render_manifold_probes()
    rendered = content_writes[-1]
    assert "personas" in rendered
    assert "hacker" in rendered
    # No mini-map markers (markers only appear in the 2-D code path).
    assert "●" not in rendered
    assert "·" not in rendered


def test_help_lists_manifold_probe_commands():
    app = _make_app()
    app._handle_command("/help")
    msg = _msgs(app)
    assert "/manifold-probe" in msg
    assert "/manifold-probe-remove" in msg


def test_pairs_extract_routes_through_session_extract():
    """A submitted pair list extracts via ``session.extract`` wrapped in
    a ``DataSource`` carrying the user-supplied name."""
    import torch
    from saklas.core.profile import Profile
    from saklas.io.datasource import DataSource

    app = _make_app()
    app._refresh_left_panel = MagicMock()
    captured: dict[str, Any] = {}

    def _fake_extract(source: Any, **kwargs: Any) -> Any:
        captured["source"] = source
        captured["kwargs"] = kwargs
        return "mood", Profile({0: torch.zeros(4)})
    app._session.extract = _fake_extract
    app.run_worker = lambda fn, thread=True: fn()
    app.call_from_thread = lambda fn, *a, **kw: fn(*a, **kw)

    app._start_pairs_extract(
        "mood", [("happy", "sad"), ("calm", "angry")],
    )

    src = captured["source"]
    assert isinstance(src, DataSource)
    assert src.name == "mood"
    assert src.pairs == [("happy", "sad"), ("calm", "angry")]
    assert "mood" in app._alphas
