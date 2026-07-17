"""CPU tests for ``session.geometry_token_readout`` (stub session).

The geometry family's token replay: same prompt-rebuild + one-capture-forward
orchestration as the lens/SAE replays (whose stub machinery this reuses from
``test_jlens_session``), scored through the Monitor roster.  The Monitor is
stubbed — its scoring math has its own suite; these tests pin the replay
*orchestration*: validation, render/prefix/pool position, steering-scope
ordering, cold-foot forcing, and the returned record.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from saklas.core.loom import (
    InvalidNodeOperationError,
    Recipe,
    UnknownNodeError,
)
from saklas.core.results import ProbeReading
from saklas.core.session import SaklasSession
from tests.test_jlens_session import (
    _PROMPT_RENDER,
    _TreeStubSession,
    _tree_with_assistant,
)


class _StubMonitor:
    """Records what the replay hands the Monitor; returns a fixed reading."""

    def __init__(self, probe_names: list[str], layers: set[int]) -> None:
        self.probe_names = probe_names
        self._layers = set(layers)
        self.scored: list[dict[int, torch.Tensor]] = []
        self.warm_calls: list[bool] = []

    def probe_layers(self) -> set[int]:
        return set(self._layers)

    def enable_curved_warm(self, flag: bool) -> None:
        self.warm_calls.append(bool(flag))

    def score_single_token(
        self, hidden: dict[int, torch.Tensor],
    ) -> dict[str, ProbeReading]:
        self.scored.append(hidden)
        return {
            name: ProbeReading(
                fraction=0.4,
                nearest=[("formal", 1.2)],
                coords=(0.7,),
                residual=0.0,
            )
            for name in self.probe_names
        }


class _GeometryStubSession(_TreeStubSession):
    geometry_token_readout = SaklasSession.geometry_token_readout

    def __init__(self) -> None:
        super().__init__()
        self._monitor = _StubMonitor(["formal.casual"], {0, 1})


def test_geometry_token_readout_shape_and_position() -> None:
    s = _GeometryStubSession()
    raw_ids = s._tokenizer.encode("abcdefg")
    node_id = _tree_with_assistant(s, raw_ids)

    seen: list[tuple[int, int | None, tuple[int, ...]]] = []
    import saklas.core.capture as _capture_mod

    real_capture = _capture_mod._capture_all_hidden_states

    def _spy(model: Any, layers: Any, ids: torch.Tensor, **kw: Any) -> Any:
        pool = kw.get("pool_index")
        seen.append((
            int(ids.shape[1]),
            int(pool) if pool is not None else None,
            tuple(kw.get("layer_indices") or ()),
        ))
        return real_capture(model, layers, ids, **kw)

    _capture_mod._capture_all_hidden_states = _spy
    try:
        out = s.geometry_token_readout(node_id, 3)
    finally:
        _capture_mod._capture_all_hidden_states = real_capture

    prompt_len = len(s._tokenizer.encode(_PROMPT_RENDER))
    # Readout position: the forward that PRODUCED the clicked token —
    # prompt + raw[:3], pooled at the final position, never including the
    # clicked token itself; capture narrowed to the roster's layer union.
    assert seen == [(prompt_len + 3, prompt_len + 2, (0, 1))]
    assert out["node_id"] == node_id
    assert out["raw_index"] == 3
    assert out["token_id"] == raw_ids[3]
    assert out["token_text"] == s._tokenizer.decode([raw_ids[3]])
    assert out["steering"] is None
    assert set(out["readings"]) == {"formal.casual"}
    assert out["readings"]["formal.casual"].coords == (0.7,)
    # The Monitor scored exactly the captured layer union, and the replay
    # forced the cold foot solve (a stale warm foot from a prior
    # generation must not seed a one-shot read).
    assert set(s._monitor.scored[0]) == {0, 1}
    assert s._monitor.warm_calls == [False]
    # Continue-mode rebuild: no text resend — the history walk to the
    # parent carries every prior turn.
    call = s.prepare_calls[0]
    assert call["input"] is None
    assert call["parent_node_id"] == s.tree.get(node_id).parent_id
    assert call["gen_seat"] == "assistant"


def test_geometry_token_readout_steering_scope() -> None:
    s = _GeometryStubSession()
    recipe = Recipe(steering="0.3 formal.casual")
    node_id = _tree_with_assistant(
        s, s._tokenizer.encode("abcd"), recipe,
    )

    out = s.geometry_token_readout(node_id, 2)
    assert out["steering"] == "0.3 formal.casual"
    assert s.steering_scopes == ["0.3 formal.casual"]

    # The unsteered counterfactual: no scope opened, steering reported null.
    out = s.geometry_token_readout(node_id, 2, apply_steering=False)
    assert out["steering"] is None
    assert s.steering_scopes == ["0.3 formal.casual"]


def test_geometry_token_readout_raw_mode_render() -> None:
    s = _GeometryStubSession()
    node_id = _tree_with_assistant(s, s._tokenizer.encode("abcd"))
    s.geometry_token_readout(node_id, 1, raw=True)
    call = s.prepare_calls[0]
    assert call["raw"] is True
    assert call["input"] == ""
    assert call["parent_node_id"] == s.tree.get(node_id).parent_id


def test_geometry_token_readout_errors() -> None:
    s = _GeometryStubSession()
    node_id = _tree_with_assistant(s, s._tokenizer.encode("abc"))

    # Unknown node.
    with pytest.raises(UnknownNodeError):
        s.geometry_token_readout("no-such-node", 0)
    # Index out of range.
    with pytest.raises(InvalidNodeOperationError, match="out of range"):
        s.geometry_token_readout(node_id, 99)
    # A system/root node has no decode record.
    assert s.tree.root_id is not None
    with pytest.raises(InvalidNodeOperationError):
        s.geometry_token_readout(s.tree.root_id, 0)
    # A finalized turn without raw ids (transcript import).
    user_id = s.tree.get(node_id).parent_id
    assert user_id is not None
    bare = s.tree.begin_assistant(user_id)
    s.tree.finalize_assistant(bare, text="no raw record", finish_reason="stop")
    with pytest.raises(InvalidNodeOperationError, match="raw token record"):
        s.geometry_token_readout(bare, 0)
    # An empty roster is a clear caller error, not an empty reading.
    s._monitor = _StubMonitor([], set())
    with pytest.raises(ValueError, match="no geometry probes attached"):
        s.geometry_token_readout(node_id, 0)


def test_geometry_token_readout_detach_race_keeps_caller_error() -> None:
    """sol's round-3 P3: a detach winning the gap between the entry roster
    check and the exclusive section must still surface the caller-facing
    "no geometry probes attached" error — not a capture-layer
    implementation leak.  The roster check is repeated under
    ``_model_exclusive``; this session's exclusive guard performs the
    detach to pin the interleaving deterministically."""
    from contextlib import contextmanager

    s = _GeometryStubSession()
    node_id = _tree_with_assistant(s, s._tokenizer.encode("abc"))

    @contextmanager
    def _detaching_exclusive(msg: str, **_kw: Any):
        del msg
        s._monitor = _StubMonitor([], set())  # the detach lands here
        yield

    s._model_exclusive = _detaching_exclusive
    with pytest.raises(ValueError, match="no geometry probes attached"):
        s.geometry_token_readout(node_id, 0)
