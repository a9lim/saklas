"""CPU tests for the session-level Jacobian-lens API (stub session).

The real ``SaklasSession`` methods are class-bound onto a light stub (the
established ``__new__``-stub pattern) so ``fit_jlens`` / ``jlens_readout`` /
``register_jlens_direction`` run against the toy model with no HF load.
"""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
import torch

from saklas.core.jlens import LensNotFittedError, MultiTokenWordError
from saklas.core.loom import (
    InvalidNodeOperationError,
    LoomTree,
    Recipe,
    UnknownNodeError,
)
from saklas.core.session import SaklasSession
from saklas.io.lens import load_lens, save_lens
from tests._jlens_toys import CharTokenizer, frozen_toy

_MODEL_ID = "toy/jlens-model"


class _StubSession:
    jlens = SaklasSession.jlens
    _require_jlens = SaklasSession._require_jlens
    fit_jlens = SaklasSession.fit_jlens
    jlens_readout = SaklasSession.jlens_readout
    register_jlens_direction = SaklasSession.register_jlens_direction
    enable_live_lens = SaklasSession.enable_live_lens
    disable_live_lens = SaklasSession.disable_live_lens
    _live_lens_readout_step = SaklasSession._live_lens_readout_step
    _jlens_workspace_band = SaklasSession._jlens_workspace_band

    def __init__(self) -> None:
        model = frozen_toy(n_layers=3)
        self._model = model
        self._tokenizer = CharTokenizer()
        self._layers = model.model.layers
        self._device = torch.device("cpu")
        self._profiles: dict[str, Any] = {}
        self._jlens: Any = None
        self._live_lens: Any = None
        self._capture: Any = None
        self.model_id = _MODEL_ID

    @contextmanager
    def _model_exclusive(self, msg: str, *, phase_msg: str | None = None):
        del msg, phase_msg
        yield

    def _invalidate_prefix_cache(self) -> None:
        pass

    def _invalidate_analytics_cache(self) -> None:
        pass


_PROMPTS = [
    "a first prompt that is long enough..",
    "the second prompt, also long enough.",
    "and a third one to round out corpus.",
    "plus a fourth for the resume checks.",
]


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))


def test_fit_jlens_persists_and_property_loads() -> None:
    s = _StubSession()
    fitted = s.fit_jlens(_PROMPTS, corpus_spec="test")
    assert fitted.n_prompts == len(_PROMPTS)

    on_disk = load_lens(_MODEL_ID)
    assert on_disk is not None
    lens, sidecar = on_disk
    assert lens.n_prompts == len(_PROMPTS)
    assert sidecar["corpus_spec"] == "test"

    fresh = _StubSession()  # new stub: property must lazy-load from disk
    assert fresh.jlens is not None
    assert fresh.jlens.n_prompts == len(_PROMPTS)


def test_fit_jlens_already_done_short_circuits() -> None:
    s = _StubSession()
    first = s.fit_jlens(_PROMPTS)
    messages: list[str] = []
    again = s.fit_jlens(_PROMPTS, on_progress=messages.append)
    assert any("nothing to do" in m for m in messages)
    # `again` reloads from the fp16 on-disk artifact — half-precision tolerance
    for layer in first.source_layers:
        assert torch.allclose(
            first.jacobians[layer], again.jacobians[layer], atol=2e-3,
        )


def test_fit_jlens_resumes_from_partial_and_matches_full_fit() -> None:
    s = _StubSession()
    full = s.fit_jlens(_PROMPTS, force=True)

    # Simulate an interrupted fit: a checkpoint covering the first 2 prompts.
    partial = _StubSession()
    head = partial.fit_jlens(_PROMPTS[:2], force=True)
    corpus_sha = hashlib.sha256("\n\x00".join(_PROMPTS).encode("utf-8")).hexdigest()
    save_lens(
        head, _MODEL_ID,
        corpus_spec="test", corpus_sha256=corpus_sha,
        seq_len=128, dim_batch=8, skip_first=16,
    )

    resumed_session = _StubSession()
    messages: list[str] = []
    resumed = resumed_session.fit_jlens(_PROMPTS, on_progress=messages.append)
    assert any("resuming from 2 prompts" in m for m in messages)
    assert resumed.n_prompts == len(_PROMPTS)
    # the resume base round-trips through the fp16 artifact — half-precision
    # tolerance against the pure-fp32 from-scratch fit
    for layer in full.source_layers:
        assert torch.allclose(
            resumed.jacobians[layer], full.jacobians[layer], atol=2e-3,
        ), f"layer {layer}: resumed fit diverges from the from-scratch fit"


def test_fit_jlens_drops_short_prompts() -> None:
    s = _StubSession()
    messages: list[str] = []
    fitted = s.fit_jlens(["tiny", *_PROMPTS], on_progress=messages.append)
    assert fitted.n_prompts == len(_PROMPTS)
    assert any("dropped 1 too-short prompts" in m for m in messages)


def test_jlens_readout_shape_and_default_position() -> None:
    s = _StubSession()
    s.fit_jlens(_PROMPTS)
    out = s.jlens_readout("a prompt that is long enough.", top_k=3)
    assert set(out) == {0, 1}  # 3-layer toy: sources are 0 and 1
    for rows in out.values():
        assert len(rows) == 1  # default: final position only
        assert len(rows[0]) == 3
        token, logprob = rows[0][0]
        assert isinstance(token, str) and logprob <= 0.0


def test_jlens_readout_requires_fitted_lens() -> None:
    s = _StubSession()
    with pytest.raises(LensNotFittedError, match="saklas lens fit"):
        s.jlens_readout("a prompt that is long enough.")


def test_jlens_readout_rejects_unfitted_layer() -> None:
    s = _StubSession()
    s.fit_jlens(_PROMPTS)
    with pytest.raises(ValueError, match="not in the fitted lens"):
        s.jlens_readout("a prompt that is long enough.", layers=[9])


def test_register_jlens_direction_registers_profile() -> None:
    s = _StubSession()
    lens = s.fit_jlens(_PROMPTS)
    name = s.register_jlens_direction("g")  # 'g' round-trips in the toy vocab
    assert name == "jlens/g"
    dirs = s._profiles[name]
    # restricted to the workspace band — for the 3-layer toy that's layer 1
    # (layer 0 sits at 0% depth, outside the 40–90% band)
    assert set(dirs) == {1}
    expected = lens.token_direction(
        s._tokenizer.encode("g")[0], s._model.lm_head.weight,
    )
    for layer, vec in dirs.items():
        assert torch.allclose(vec, expected[layer])
    # idempotent
    assert s.register_jlens_direction("g") == name


def test_register_jlens_direction_multi_token_raises() -> None:
    s = _StubSession()
    s.fit_jlens(_PROMPTS)
    with pytest.raises(MultiTokenWordError):
        s.register_jlens_direction("gg")


# ------------------------------------------------------------- live lens ----


class _FakeCapture:
    """Minimal HiddenCapture stand-in: per_layer_buckets() -> latest slices."""

    def __init__(self, slices: dict[int, torch.Tensor]) -> None:
        self._buckets = {l: [t] for l, t in slices.items()}

    def per_layer_buckets(self) -> dict[int, list[torch.Tensor]]:
        return self._buckets


def test_enable_live_lens_defaults_and_disable() -> None:
    s = _StubSession()
    s.fit_jlens(_PROMPTS)
    layers = s.enable_live_lens()
    # 3-layer toy: fitted sources are [0, 1]; the 40-90% band keeps layer 1
    assert layers and all(l in (0, 1) for l in layers)
    assert s._live_lens is not None
    assert s._live_lens["layers"] == layers
    s.disable_live_lens()
    assert s._live_lens is None


def test_enable_live_lens_rejects_unfitted_layer() -> None:
    s = _StubSession()
    s.fit_jlens(_PROMPTS)
    with pytest.raises(ValueError, match="not in the fitted lens"):
        s.enable_live_lens(layers=[7])


def test_enable_live_lens_requires_lens() -> None:
    s = _StubSession()
    with pytest.raises(LensNotFittedError):
        s.enable_live_lens()


def test_enable_live_lens_registers_no_forward_hooks() -> None:
    """The live lens must not touch the model: no hooks, no wrapping — the
    reader consumes existing capture buffers (compile/fast-path safety)."""
    s = _StubSession()
    s.fit_jlens(_PROMPTS)
    before = [
        (len(block._forward_hooks), len(block._forward_pre_hooks))
        for block in s._layers
    ]
    s.enable_live_lens()
    after = [
        (len(block._forward_hooks), len(block._forward_pre_hooks))
        for block in s._layers
    ]
    assert before == after


def test_live_lens_readout_step_reads_latest_slices() -> None:
    s = _StubSession()
    s.fit_jlens(_PROMPTS)
    s.enable_live_lens(layers=[0, 1], top_k=3)
    # The per-step reader should use the pre-stacked transport cache, not the
    # per-layer dict.  Replacing the dict entries would have blown up the old
    # per-token ``state["J"][layer].to(...)`` path.
    class Bomb:
        def to(self, *_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("live lens readout should use J_stack")

    assert s._live_lens is not None
    s._live_lens["J"] = {0: Bomb(), 1: Bomb()}
    gen = torch.Generator().manual_seed(11)
    s._capture = _FakeCapture({
        0: torch.randn(6, generator=gen),
        1: torch.randn(6, generator=gen),
    })

    out = SaklasSession._live_lens_readout_step(s)  # type: ignore[arg-type]
    assert out is not None and set(out) == {0, 1}
    for row in out.values():
        assert len(row) == 3
        assert all(isinstance(tok, str) for tok, _ in row)
    # scores are descending
    scores = [sc for _, sc in out[0]]
    assert scores == sorted(scores, reverse=True)


def test_live_lens_readout_step_none_when_off() -> None:
    s = _StubSession()
    assert SaklasSession._live_lens_readout_step(s) is None  # type: ignore[arg-type]


# --------------------------------------------------- token readout (loom) ----


_PROMPT_RENDER = "the prompt render, chat shaped."


class _TreeStubSession(_StubSession):
    """Stub with a real loom tree + recorded prompt render / steering scopes."""

    jlens_token_readout = SaklasSession.jlens_token_readout

    def __init__(self) -> None:
        super().__init__()
        self.tree = LoomTree(model_id=_MODEL_ID)
        self.prepare_calls: list[dict[str, Any]] = []
        self.steering_scopes: list[Any] = []

    def _prepare_input(
        self,
        input: Any,
        raw: bool = False,
        thinking: bool = False,
        stateless: bool = False,
        parent_node_id: str | None = None,
        user_role: str | None = None,
        assistant_role: str | None = None,
        to_device: bool = True,
    ) -> torch.Tensor:
        self.prepare_calls.append({
            "input": input, "raw": raw, "thinking": thinking,
            "parent_node_id": parent_node_id,
            "user_role": user_role, "assistant_role": assistant_role,
        })
        return torch.tensor(
            [self._tokenizer.encode(_PROMPT_RENDER)], dtype=torch.long,
        )

    @contextmanager
    def steering(self, value: Any):
        self.steering_scopes.append(value)
        yield


def _tree_with_assistant(
    s: _TreeStubSession,
    raw_ids: list[int] | None,
    recipe: Recipe | None = None,
) -> str:
    user_id = s.tree.add_user_turn("a user turn")
    node_id = s.tree.begin_assistant(user_id, recipe=recipe)
    s.tree.finalize_assistant(
        node_id, text="an assistant turn", finish_reason="stop",
        raw_token_ids=raw_ids,
    )
    return node_id


def test_jlens_token_readout_shape_and_position() -> None:
    s = _TreeStubSession()
    s.fit_jlens(_PROMPTS)
    raw_ids = s._tokenizer.encode("abcdefg")
    node_id = _tree_with_assistant(s, raw_ids)

    seen_lens: list[int] = []
    import saklas.core.vectors as _vectors

    real_capture = _vectors._capture_all_hidden_states

    def _spy(model: Any, layers: Any, ids: torch.Tensor, **kw: Any) -> Any:
        seen_lens.append(int(ids.shape[1]))
        return real_capture(model, layers, ids, **kw)

    _vectors._capture_all_hidden_states = _spy
    try:
        out = s.jlens_token_readout(node_id, 3, top_k=4)
    finally:
        _vectors._capture_all_hidden_states = real_capture

    prompt_len = len(s._tokenizer.encode(_PROMPT_RENDER))
    # readout position: the forward that PRODUCED the clicked token —
    # prompt + raw[:3], never including the clicked token itself.
    assert seen_lens == [prompt_len + 3]
    assert out["node_id"] == node_id
    assert out["raw_index"] == 3
    assert out["token_id"] == raw_ids[3]
    assert out["token_text"] == s._tokenizer.decode([raw_ids[3]])
    assert out["steering"] is None
    assert out["workspace_band"] == [1]  # 3-layer toy: 40-90% band keeps L1
    assert set(out["readout"]) == {0, 1}  # fitted sources of the 3-layer toy
    for rows in out["readout"].values():
        assert len(rows) == 4
        tok, lp, tid = rows[0]
        assert isinstance(tok, str) and lp <= 0.0 and isinstance(tid, int)
    # user_role/assistant_role of the replayed render come off the nodes
    assert s.prepare_calls[0]["input"] == "a user turn"
    assert s.prepare_calls[0]["raw"] is False


def test_jlens_token_readout_index_zero_reads_prompt_only() -> None:
    s = _TreeStubSession()
    s.fit_jlens(_PROMPTS)
    node_id = _tree_with_assistant(s, s._tokenizer.encode("abc"))

    seen_lens: list[int] = []
    import saklas.core.vectors as _vectors

    real_capture = _vectors._capture_all_hidden_states

    def _spy(model: Any, layers: Any, ids: torch.Tensor, **kw: Any) -> Any:
        seen_lens.append(int(ids.shape[1]))
        return real_capture(model, layers, ids, **kw)

    _vectors._capture_all_hidden_states = _spy
    try:
        s.jlens_token_readout(node_id, 0, top_k=2)
    finally:
        _vectors._capture_all_hidden_states = real_capture
    assert seen_lens == [len(s._tokenizer.encode(_PROMPT_RENDER))]


def test_jlens_token_readout_steering_scope() -> None:
    s = _TreeStubSession()
    s.fit_jlens(_PROMPTS)
    recipe = Recipe(steering="0.3 formal.casual", thinking=False)
    node_id = _tree_with_assistant(s, s._tokenizer.encode("abcd"), recipe)

    out = s.jlens_token_readout(node_id, 2, top_k=2)
    assert s.steering_scopes == ["0.3 formal.casual"]
    assert out["steering"] == "0.3 formal.casual"

    s.steering_scopes.clear()
    out = s.jlens_token_readout(node_id, 2, top_k=2, apply_steering=False)
    assert s.steering_scopes == []
    assert out["steering"] is None


def test_jlens_token_readout_raw_mode_render() -> None:
    s = _TreeStubSession()
    s.fit_jlens(_PROMPTS)
    node_id = _tree_with_assistant(s, s._tokenizer.encode("abcd"))

    s.jlens_token_readout(node_id, 1, top_k=2, raw=True)
    call = s.prepare_calls[0]
    assert call["raw"] is True and call["input"] == ""
    # raw render anchors at the assistant node's parent (the flat prefix)
    assert call["parent_node_id"] == s.tree.get(node_id).parent_id


def test_jlens_token_readout_errors() -> None:
    s = _TreeStubSession()
    node_id = _tree_with_assistant(s, s._tokenizer.encode("abc"))

    with pytest.raises(LensNotFittedError):
        s.jlens_token_readout(node_id, 0)

    s.fit_jlens(_PROMPTS)
    user_id = s.tree.get(node_id).parent_id
    assert user_id is not None
    with pytest.raises(UnknownNodeError):
        s.jlens_token_readout("nope", 0)
    with pytest.raises(InvalidNodeOperationError, match="not an assistant"):
        s.jlens_token_readout(user_id, 0)
    with pytest.raises(InvalidNodeOperationError, match="out of range"):
        s.jlens_token_readout(node_id, 3)
    with pytest.raises(InvalidNodeOperationError, match="out of range"):
        s.jlens_token_readout(node_id, -1)
    with pytest.raises(ValueError, match="not in the fitted lens"):
        s.jlens_token_readout(node_id, 0, layers=[9])

    bare = s.tree.begin_assistant(user_id)
    s.tree.finalize_assistant(bare, text="no raw record", finish_reason="stop")
    with pytest.raises(InvalidNodeOperationError, match="no raw token record"):
        s.jlens_token_readout(bare, 0)
