"""Shared mock-session factory for server-layer CPU tests.

Four server test files used to each define a near-identical local
``_mock_session()`` factory.  This module provides ONE canonical factory that
the files import, so wiring changes land in one place.

Only the genuinely shared wiring lives here.  Per-file extras stay local:
- ``test_saklas_api.py``:  real trait-queue logic, real EventBus subscribe/emit
  callbacks, ``session.monitor`` (public property), ``session.manifolds``.  These
  are integral to the WebSocket + SSE tests and can't be naively lifted out
  without coupling this module to those tests.  ``test_saklas_api.py`` imports
  ``make_mock_session`` and layers its extras on top.
- ``test_web.py``:  ``_mock_session_with_vectors`` has a completely different
  shape (takes a ``vectors`` dict, exposes analytics helpers, wires ``whitener``)
  and is not a variant of the common factory — it stays local.

Usage::

    from tests._fakes import make_mock_session

    session = make_mock_session()
    # add per-test extras...
"""
from __future__ import annotations

import asyncio
import threading
from typing import Any
from unittest.mock import MagicMock


def make_mock_session(**overrides: Any) -> Any:
    """Return a ``MagicMock`` pre-wired to look like a ``SaklasSession``.

    The minimal shared wiring covers what every server route touches:
    - model identity / info
    - per-session config scalars
    - empty ``vectors`` / ``probes`` / ``history``
    - a real ``asyncio.Lock`` so ``async with session.lock`` works under
      ``TestClient``'s event loop
    - ``session.build_readings.return_value = {}``

    Callers may pass keyword overrides to patch specific attributes after
    construction; they take effect as ``setattr(session, k, v)``.
    """
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

    session.profiles = {}
    session.probes = {}
    session.history = []

    session.lock = asyncio.Lock()
    session.build_readings.return_value = {}

    # Trait queue infrastructure — minimal stubs (real logic stays in
    # test_saklas_api.py which overrides these with the actual callbacks).
    session._trait_queues = []
    session._trait_lock = threading.Lock()
    session.register_trait_queue = lambda *_a, **_kw: None
    session.unregister_trait_queue = lambda *_a, **_kw: None

    # Minimal events stub so routes that call ``events.subscribe`` don't error.
    session.events = MagicMock()
    session.events.subscribe = lambda cb: (lambda: None)
    session.events.emit = lambda event: None

    for k, v in overrides.items():
        setattr(session, k, v)

    return session
