"""Shared mock-session factory for the native-API CPU tests.

``test_saklas_api.py``, ``test_server_manifold_probes.py`` and
``test_profiles_bake_api.py`` each opened their local ``_mock_session()``
with the same block — model identity, the device/dtype/created-at scalars,
the default sampling config, empty ``profiles``/``probes``, and a real
``asyncio.Lock``.  That block lives here once; each file calls
:func:`make_mock_session` and layers its own wiring on top.

The remaining sibling factories deliberately stay local — their shapes are
not variants of this one:

- ``test_server.py``:  a seven-key ``model_info`` (``vram_used_gb`` /
  ``param_count``), a ``config`` carrying ``thinking``, a 26-entry
  ``layers`` list, and none of the identity scalars.
- ``test_instrument_routes.py``:  real ``GeometryInstrument`` /
  ``LensInstrument`` / ``SaeInstrument`` objects over a faked source
  lifecycle, and no probe/profile collections at all.
- ``test_user_message.py``:  a nested single-test helper wired on the
  private attribute names (``_gen_state`` / ``_last_result`` /
  ``_tokenizer``).
- ``test_web.py``:  ``_mock_session_with_vectors`` has a different signature
  (takes a ``vectors`` dict), exposes analytics helpers, and wires
  ``whitener``.

Usage::

    from tests._fakes import make_mock_session

    session = make_mock_session()             # or (config=GenerationConfig())
    # add per-file extras...
"""
from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock


def make_mock_session(**overrides: Any) -> Any:
    """Return a ``MagicMock`` pre-wired to look like a ``SaklasSession``.

    The shared wiring covers what the native routes touch on every one of
    its three consumers:

    - model identity / info
    - the ``_device`` / ``_dtype`` / ``_created_ts`` scalars
    - the default sampling config
    - empty ``profiles`` / ``probes``
    - a real ``asyncio.Lock`` so ``async with session.lock`` works under
      ``TestClient``'s event loop

    Keyword overrides are applied last as ``setattr(session, k, v)``, so a
    caller wanting a real ``GenerationConfig`` passes ``config=...`` rather
    than reassigning after the call.
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

    session.lock = asyncio.Lock()

    for k, v in overrides.items():
        setattr(session, k, v)

    return session
