"""Tests for ``SaklasError.user_message()`` centralization (Phase 4).

Each subclass returns an HTTP-style ``(status, msg)`` tuple; the three
user-facing surfaces (server, CLI, TUI) consume the value to translate
exceptions consistently.  Tests pin the contract so subclasses can't
silently lose their override.
"""

from __future__ import annotations

import pytest

from saklas.cli.config_file import ConfigFileError
from saklas.cli.selectors import AmbiguousSelectorError, SelectorError
from saklas.core.errors import (
    AmbiguousVariantError,
    SaeBackendImportError,
    SaeCoverageError,
    SaeModelMismatchError,
    SaeReleaseNotFoundError,
    SaklasError,
    StaleSidecarError,
    UnknownVariantError,
)
from saklas.core.profile import ProfileError
from saklas.core.session import (
    ConcurrentGenerationError,
    VectorNotRegisteredError,
)
from saklas.core.steering_expr import SteeringExprError
from saklas.io.cache_ops import InstallConflict, RefreshError
from saklas.io.cloning import (
    CorpusTooLongError,
    CorpusTooShortError,
    InsufficientPairsError,
)
from saklas.io.gguf_io import GGUFNotInstalled
from saklas.io.hf import HFError
from saklas.io.merge import MergeError
from saklas.io.packs import PackFormatError


def test_base_default_status_and_message():
    """Plain ``SaklasError`` returns ``(500, str(self))``."""
    assert SaklasError("x").user_message() == (500, "x")


def test_base_empty_falls_back_to_class_name():
    """Empty args fall back to the class name so the user sees something."""
    assert SaklasError().user_message() == (500, "SaklasError")


# (subclass, expected_status_code) — one entry per overriding class.
# Bumping the status is a deliberate contract change; any drift here is
# user-visible.
_OVERRIDES: list[tuple[type[SaklasError], int]] = [
    # core/errors.py
    (SaeBackendImportError, 400),
    (SaeReleaseNotFoundError, 400),
    (SaeModelMismatchError, 400),
    (SaeCoverageError, 400),
    (AmbiguousVariantError, 400),
    (UnknownVariantError, 404),
    (StaleSidecarError, 409),
    # core/session.py
    (ConcurrentGenerationError, 409),
    (VectorNotRegisteredError, 404),
    # core/profile.py
    (ProfileError, 400),
    # core/steering_expr.py
    (SteeringExprError, 400),
    # cli/selectors.py
    (SelectorError, 400),
    (AmbiguousSelectorError, 400),
    # cli/config_file.py
    (ConfigFileError, 400),
    # io/packs.py
    (PackFormatError, 400),
    # io/hf.py
    (HFError, 502),
    # io/cache_ops.py
    (InstallConflict, 409),
    (RefreshError, 500),
    # io/cloning.py
    (CorpusTooShortError, 400),
    (CorpusTooLongError, 400),
    (InsufficientPairsError, 422),
    # io/gguf_io.py
    (GGUFNotInstalled, 400),
    # io/merge.py
    (MergeError, 400),
]


@pytest.mark.parametrize("cls,expected_status", _OVERRIDES)
def test_subclass_status_codes(cls: type[SaklasError], expected_status: int):
    """Each overriding subclass returns its declared status code."""
    code, _msg = cls("test message").user_message()
    assert code == expected_status, (
        f"{cls.__name__} returned status {code}, expected {expected_status}"
    )


@pytest.mark.parametrize("cls,_status", _OVERRIDES)
def test_subclass_message_round_trips(cls: type[SaklasError], _status: int):
    """The message string is non-empty and contains the original payload.

    KeyError-derived subclasses (``UnknownVariantError``,
    ``VectorNotRegisteredError``) reach into ``args[0]`` to avoid
    KeyError's repr-quoted ``str()`` shape; we just check the original
    payload is in the surfaced message.
    """
    payload = "specific test payload"
    _code, msg = cls(payload).user_message()
    assert payload in msg, f"{cls.__name__} dropped the message payload"


def test_ambiguous_selector_carries_disambiguation_tip():
    """``AmbiguousSelectorError`` appends the disambiguation hint."""
    _code, msg = AmbiguousSelectorError("'wolf' matches deer/wolf and zoo/wolf").user_message()
    assert "namespace/name" in msg


def test_ambiguous_selector_skips_redundant_tip():
    """When the message already names the fix, don't double-suffix it."""
    err = AmbiguousSelectorError("ambiguous; disambiguate with namespace/name (e.g. deer/wolf)")
    _code, msg = err.user_message()
    # Should not contain the canned tip twice.
    assert msg.count("namespace/name") == 1


def test_unknown_variant_strips_keyerror_quotes():
    """``KeyError`` would render as ``"'foo'"``; the override uses ``args[0]``."""
    _code, msg = UnknownVariantError("missing variant").user_message()
    assert msg == "missing variant"


def test_concurrent_generation_carries_message():
    _code, msg = ConcurrentGenerationError("already running").user_message()
    assert msg == "already running"


def test_steering_expr_error_carries_col_in_message():
    """``SteeringExprError`` formats column info into ``str(self)``;
    ``user_message`` shouldn't drop it."""
    err = SteeringExprError("unexpected token", col=12)
    _code, msg = err.user_message()
    assert "col 12" in msg


# ---------------------------------------------------------------------------
# Server integration smoke — ensure the handler routes correctly.
# ---------------------------------------------------------------------------


def test_server_routes_user_message_status_codes():
    """``server/app.py:_on_saklas_error`` honors ``user_message()`` status."""
    import asyncio
    from unittest.mock import MagicMock

    from fastapi.testclient import TestClient

    from saklas.core.results import GenerationResult
    from saklas.server import create_app

    def _mock_session():
        s = MagicMock()
        s.model_id = "test/model"
        s.model_info = {
            "model_type": "gemma2",
            "num_layers": 26,
            "hidden_dim": 2304,
            "vram_used_gb": 5.2,
            "param_count": 2_614_000_000,
            "dtype": "torch.bfloat16",
        }
        s.config = MagicMock()
        s.config.temperature = 1.0
        s.config.top_p = 0.9
        s.config.max_new_tokens = 1024
        s.config.system_prompt = None
        s.vectors = {}
        s.probes = {}
        s.history = []
        gs = MagicMock()
        gs.finish_reason = "stop"
        s._gen_state = gs
        s._last_result = None
        s._tokenizer = MagicMock()
        s._tokenizer.decode.side_effect = lambda ids: f"<{ids[0]}>" if ids else ""
        s.build_readings.return_value = {}
        s.lock = asyncio.Lock()
        return s

    # 409: ConcurrentGenerationError already pinned by test_server.py;
    # we add coverage for one more family per status to confirm
    # the handler doesn't regress when the table is removed.
    cases: list[tuple[Exception, int]] = [
        (ConcurrentGenerationError("busy"), 409),
        (AmbiguousSelectorError("'x' matches a/x and b/x"), 400),
        (HFError("network down"), 502),
        (InstallConflict("already exists"), 409),
        (PackFormatError("malformed"), 400),
    ]

    for exc, expected_status in cases:
        session = _mock_session()
        session.generate.side_effect = exc
        app = create_app(session)
        client = TestClient(app)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test/model",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 4,
            },
        )
        assert resp.status_code == expected_status, (
            f"{type(exc).__name__} → got {resp.status_code}, expected {expected_status}"
        )
