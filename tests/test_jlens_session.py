"""CPU tests for the session-level Jacobian-lens API (stub session).

The real ``SaklasSession`` methods are class-bound onto a light stub (the
established ``__new__``-stub pattern) so ``fit_jlens`` / ``jlens_readout`` /
``register_jlens_direction`` run against the toy model with no HF load.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from saklas.core.jlens import LensNotFittedError, MultiTokenWordError
from saklas.core.model import loaded_model_fingerprint, model_source_fingerprint
from saklas.core.loom import (
    InvalidNodeOperationError,
    LoomTree,
    Recipe,
    UnknownNodeError,
)
from saklas.core.session import SaklasSession
from saklas.io.lens import (
    lens_checkpoint_paths,
    lens_paths,
    load_lens,
    load_lens_checkpoint,
    save_lens_checkpoint,
)
from tests._jlens_toys import CharTokenizer, frozen_toy

_MODEL_ID = "toy/jlens-model"


class _StubSession:
    jlens = SaklasSession.jlens
    _require_jlens = SaklasSession._require_jlens
    fit_jlens = SaklasSession.fit_jlens
    jlens_readout = SaklasSession.jlens_readout
    _resolve_jlens_layers = SaklasSession._resolve_jlens_layers
    _resolve_jlens_source_layers = SaklasSession._resolve_jlens_source_layers
    _jlens_transport_stack = SaklasSession._jlens_transport_stack
    _jlens_topk_rows = SaklasSession._jlens_topk_rows
    _jlens_logits_rows = SaklasSession._jlens_logits_rows
    _jlens_aggregate_rows = SaklasSession._jlens_aggregate_rows
    _jlens_decode_id = SaklasSession._jlens_decode_id
    _jlens_depths = SaklasSession._jlens_depths
    register_jlens_direction = SaklasSession.register_jlens_direction
    enable_live_lens = SaklasSession.enable_live_lens
    disable_live_lens = SaklasSession.disable_live_lens
    _live_lens_readout_step = SaklasSession._live_lens_readout_step
    _jlens_workspace_band = SaklasSession._jlens_workspace_band
    _add_lens_probe = SaklasSession._add_lens_probe
    _lens_probe_layers = SaklasSession._lens_probe_layers
    _score_lens_probes = SaklasSession._score_lens_probes
    _score_lens_gate_scalars = SaklasSession._score_lens_gate_scalars

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
        self._lens_probes: dict[str, Any] = {}
        self._probe_hash_cache: dict[str, str] = {}
        self._lens_step_stash: Any = None
        self._last_lens_step_readings: Any = None
        self._monitor: Any = None
        self.model_id = _MODEL_ID

    @contextmanager
    def _model_exclusive(self, msg: str, *, phase_msg: str | None = None):
        del msg, phase_msg
        yield

    def _invalidate_prefix_cache(self) -> None:
        pass

    def _invalidate_analytics_cache(self) -> None:
        pass


class _CountingTokenizer(CharTokenizer):
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, text: str, return_tensors: str = "pt") -> dict[str, torch.Tensor]:
        self.calls += 1
        return super().__call__(text, return_tensors=return_tensors)


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


def test_refit_rebuilds_live_lens_probes_and_evicts_directions() -> None:
    s = _StubSession()
    s.fit_jlens(_PROMPTS, source_layers=[0])
    s.enable_live_lens(layers=[0], top_k=7)
    old_stack = s._live_lens["J_stack"]
    s._profiles["jlens/a"] = {0: torch.ones(4)}
    s._lens_probes["jlens/a"] = {
        "word": "a", "token_id": 1, "layers": [0],
    }

    fitted = s.fit_jlens(_PROMPTS, source_layers=[1], force=True)

    assert fitted.source_layers == [1]
    assert s._live_lens["layers"] == [1]
    assert s._live_lens["top_k"] == 7
    assert s._live_lens["J_stack"] is not old_stack
    assert s._lens_probes["jlens/a"]["layers"] == [1]
    assert "jlens/a" not in s._profiles


def test_jlens_property_rejects_legacy_sidecar_without_weight_identity() -> None:
    fitted = _StubSession()
    fitted.fit_jlens(_PROMPTS)
    _, sidecar_path = lens_paths(_MODEL_ID)
    sidecar = json.loads(sidecar_path.read_text())
    sidecar.pop("model_fingerprint")
    sidecar_path.write_text(json.dumps(sidecar))
    assert _StubSession().jlens is None


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


def test_fit_jlens_noop_revalidates_token_ids_before_loading_tensor() -> None:
    s = _StubSession()
    s._tokenizer = _CountingTokenizer()
    s.fit_jlens(_PROMPTS)
    s._tokenizer.calls = 0

    again = s.fit_jlens(_PROMPTS, source_layers=[1])

    assert again.source_layers == [1]
    assert s._tokenizer.calls == len(_PROMPTS)


def test_fit_jlens_changed_tokenizer_invalidates_cache() -> None:
    s = _StubSession()
    s.fit_jlens(_PROMPTS)
    original = load_lens(_MODEL_ID)
    assert original is not None

    class _ShiftedTokenizer(CharTokenizer):
        def __call__(
            self, text: str, return_tensors: str = "pt",
        ) -> dict[str, torch.Tensor]:
            result = super().__call__(text, return_tensors=return_tensors)
            result["input_ids"] = (result["input_ids"] + 1) % 13
            return result

    s._tokenizer = _ShiftedTokenizer()
    s._jlens = None
    s.fit_jlens(_PROMPTS)
    changed = load_lens(_MODEL_ID)
    assert changed is not None
    assert changed[1]["corpus_sha256"] != original[1]["corpus_sha256"]


def test_fit_jlens_resumes_from_partial_and_matches_full_fit() -> None:
    s = _StubSession()
    full = s.fit_jlens(_PROMPTS, force=True)

    # A normal smaller-corpus artifact carries the hash of that actual prefix.
    # Extending 2 -> N must recognize it without the old test-only trick of
    # stamping the prefix tensor with the future full-corpus hash.
    partial = _StubSession()
    partial.fit_jlens(_PROMPTS[:2], force=True)

    resumed_session = _StubSession()
    messages: list[str] = []
    resumed = resumed_session.fit_jlens(_PROMPTS, on_progress=messages.append)
    assert any("resuming from 2 prompts" in m for m in messages)
    assert any("prompt 4/4" in m for m in messages)
    assert resumed.n_prompts == len(_PROMPTS)
    # the resume base round-trips through the fp16 artifact — half-precision
    # tolerance against the pure-fp32 from-scratch fit
    for layer in full.source_layers:
        assert torch.allclose(
            resumed.jacobians[layer], full.jacobians[layer], atol=2e-3,
        ), f"layer {layer}: resumed fit diverges from the from-scratch fit"


def test_fit_jlens_changed_prefix_restarts_from_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import saklas.core.jlens as jlens_module

    _StubSession().fit_jlens(_PROMPTS[:2], force=True)
    changed = ["a changed first prompt that is long enough", *_PROMPTS[1:]]
    real_fit = jlens_module.fit_jacobian_lens
    fitted_widths: list[int] = []

    def counted_fit(*args: Any, **kwargs: Any) -> Any:
        fitted_widths.append(len(args[2]))
        return real_fit(*args, **kwargs)

    monkeypatch.setattr(jlens_module, "fit_jacobian_lens", counted_fit)
    result = _StubSession().fit_jlens(changed)
    assert result.n_prompts == len(changed)
    assert fitted_widths == [len(changed)]


def test_fit_jlens_loaded_weight_change_invalidates_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import saklas.core.jlens as jlens_module

    first = _StubSession()
    first.fit_jlens(_PROMPTS)
    changed = _StubSession()
    changed_model: Any = changed._model
    with torch.no_grad():
        changed_model.model.layers[0].w1.data.reshape(-1)[1] += 0.125
    real_fit = jlens_module.fit_jacobian_lens
    fitted_widths: list[int] = []

    def counted_fit(*args: Any, **kwargs: Any) -> Any:
        fitted_widths.append(len(args[2]))
        return real_fit(*args, **kwargs)

    monkeypatch.setattr(jlens_module, "fit_jacobian_lens", counted_fit)
    changed.fit_jlens(_PROMPTS)
    assert fitted_widths == [len(_PROMPTS)]


def test_loaded_model_fingerprint_hashes_buffers_and_original_dtype_bits() -> None:
    class _Stateful(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.tensor([1.0], dtype=torch.float64),
            )
            self.register_buffer("scale", torch.tensor([2], dtype=torch.int64))
            self.config = SimpleNamespace(
                model_type="toy", _commit_hash="same-claimed-commit",
                _name_or_path="toy",
            )

    base = _Stateful()
    weight_changed = _Stateful()
    weight_changed.weight.data[0] = torch.nextafter(
        weight_changed.weight.data[0], torch.tensor(2.0, dtype=torch.float64),
    )
    buffer_changed = _Stateful()
    buffer_scale: Any = buffer_changed.scale
    buffer_scale[0] = 3
    fp = loaded_model_fingerprint(base, "toy")
    assert loaded_model_fingerprint(weight_changed, "toy") != fp
    assert loaded_model_fingerprint(buffer_changed, "toy") != fp


def test_loaded_model_fingerprint_memoizes_until_sanctioned_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = torch.nn.Linear(8, 8, bias=False)
    real_to = torch.Tensor.to
    transfers = 0

    def counted_to(self: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        nonlocal transfers
        if kwargs.get("device") == "cpu":
            transfers += 1
        return real_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", counted_to)
    first = loaded_model_fingerprint(model, "toy")
    after_first = transfers
    assert loaded_model_fingerprint(model, "toy") == first
    assert transfers == after_first
    with torch.no_grad():
        model.weight.reshape(-1)[3].add_(0.5)
    assert loaded_model_fingerprint(model, "toy") != first
    assert transfers > after_first


def test_loaded_model_fingerprint_explicitly_invalidates_data_writes() -> None:
    from saklas.core.model import invalidate_loaded_model_fingerprint

    model = torch.nn.Linear(10, 10, bias=False)
    first = loaded_model_fingerprint(model, "toy")
    model.weight.data.reshape(-1)[7].add_(10)
    invalidate_loaded_model_fingerprint(model)
    assert loaded_model_fingerprint(model, "toy") != first


def test_local_source_fingerprint_includes_remote_code_and_tokenizer_files(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "local-model"
    model_dir.mkdir()
    (model_dir / "model.safetensors").write_bytes(b"weights")
    (model_dir / "config.json").write_text("{}")
    code = model_dir / "modeling_custom.py"
    vocab = model_dir / "vocab.json"
    code.write_text("VALUE = 1\n")
    vocab.write_text('{"a": 0}')
    config = SimpleNamespace(
        model_type="custom", _commit_hash=None,
        _name_or_path=str(model_dir),
    )
    first = model_source_fingerprint(
        str(model_dir), config=config, device="cpu",
        parameter_dtype=torch.float32,
    )
    assert first is not None
    code.write_text("VALUE = 2\n")
    second = model_source_fingerprint(
        str(model_dir), config=config, device="cpu",
        parameter_dtype=torch.float32,
    )
    assert second != first
    vocab.write_text('{"b": 0}')
    third = model_source_fingerprint(
        str(model_dir), config=config, device="cpu",
        parameter_dtype=torch.float32,
    )
    assert third != second
    arbitrary_resource = model_dir / "bpe.codes"
    arbitrary_resource.write_text("old rules")
    fourth = model_source_fingerprint(
        str(model_dir), config=config, device="cpu",
        parameter_dtype=torch.float32,
    )
    assert fourth != third


def test_local_source_fingerprint_detects_same_size_rewrite_with_restored_mtime(
    tmp_path: Path,
) -> None:
    import os

    model_dir = tmp_path / "local-model"
    model_dir.mkdir()
    weights = model_dir / "model.safetensors"
    weights.write_bytes(b"weights-a")
    config = SimpleNamespace(
        model_type="custom", _commit_hash=None, _name_or_path=str(model_dir),
    )
    first = model_source_fingerprint(
        str(model_dir), config=config, device="cpu",
        parameter_dtype=torch.float32,
    )
    original = weights.stat()
    weights.write_bytes(b"weights-b")
    os.utime(weights, ns=(original.st_atime_ns, original.st_mtime_ns))

    second = model_source_fingerprint(
        str(model_dir), config=config, device="cpu",
        parameter_dtype=torch.float32,
    )
    assert second != first


def test_jlens_property_rejects_changed_loaded_weights() -> None:
    _StubSession().fit_jlens(_PROMPTS)
    changed = _StubSession()
    changed_model: Any = changed._model
    with torch.no_grad():
        changed_model.model.layers[0].w1.data.reshape(-1)[1] += 0.125
    assert changed.jlens is None


def test_fit_jlens_resumes_from_checkpoint_without_full_artifact() -> None:
    full = _StubSession().fit_jlens(_PROMPTS, force=True)
    head_session = _StubSession()
    head = head_session.fit_jlens(_PROMPTS[:2], force=True)
    for path in lens_paths(_MODEL_ID):
        path.unlink()
    consumed = [
        [int(tok) for tok in head_session._tokenizer(
            p, return_tensors="pt",
        )["input_ids"][0].tolist()]
        for p in _PROMPTS
    ]
    corpus_sha = hashlib.sha256(repr(consumed).encode("utf-8")).hexdigest()
    save_lens_checkpoint(
        head, _MODEL_ID,
        base_n_prompts=0,
        corpus_spec="test",
        corpus_sha256=corpus_sha,
        corpus_hash_kind="token_ids_v1",
        seq_len=128,
        dim_batch=8,
        skip_first=16,
        model_fingerprint=loaded_model_fingerprint(
            head_session._model, _MODEL_ID,
        ),
    )

    messages: list[str] = []
    resumed = _StubSession().fit_jlens(_PROMPTS, on_progress=messages.append)

    assert any("resuming from checkpoint at 2 prompts" in m for m in messages)
    assert resumed.n_prompts == len(_PROMPTS)
    for layer in full.source_layers:
        assert torch.allclose(
            resumed.jacobians[layer], full.jacobians[layer], atol=2e-3,
        )
    assert load_lens(_MODEL_ID) is not None
    assert not any(path.exists() for path in lens_checkpoint_paths(_MODEL_ID))


def test_fit_jlens_checkpoint_survives_two_interruptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import saklas.core.jlens as jlens_mod

    full = _StubSession().fit_jlens(_PROMPTS, force=True)
    head_session = _StubSession()
    head = head_session.fit_jlens(_PROMPTS[:2], force=True)
    consumed = [
        [int(tok) for tok in head_session._tokenizer(
            prompt, return_tensors="pt",
        )["input_ids"][0].tolist()]
        for prompt in _PROMPTS
    ]
    corpus_sha = hashlib.sha256(repr(consumed).encode("utf-8")).hexdigest()
    save_lens_checkpoint(
        head, _MODEL_ID,
        base_n_prompts=0,
        corpus_spec="test",
        corpus_sha256=corpus_sha,
        corpus_hash_kind="token_ids_v1",
        seq_len=128,
        dim_batch=8,
        skip_first=16,
        model_fingerprint=loaded_model_fingerprint(
            head_session._model, _MODEL_ID,
        ),
    )

    real_fit = jlens_mod.fit_jacobian_lens

    def _interrupt_after_one(*args: Any, **kwargs: Any) -> Any:
        prompts = list(args[2])
        args = (*args[:2], prompts[:1], *args[3:])
        kwargs["input_id_rows"] = kwargs["input_id_rows"][:1]
        kwargs["checkpoint_every"] = 1
        real_fit(*args, **kwargs)
        raise RuntimeError("simulated second interruption")

    monkeypatch.setattr(jlens_mod, "fit_jacobian_lens", _interrupt_after_one)
    with pytest.raises(RuntimeError, match="second interruption"):
        _StubSession().fit_jlens(_PROMPTS)

    checkpoint = load_lens_checkpoint(_MODEL_ID)
    assert checkpoint is not None
    assert checkpoint[0].n_prompts == 3
    assert checkpoint[1]["base_n_prompts"] == 0

    monkeypatch.setattr(jlens_mod, "fit_jacobian_lens", real_fit)
    messages: list[str] = []
    resumed = _StubSession().fit_jlens(_PROMPTS, on_progress=messages.append)
    assert any("checkpoint at 3 prompts" in message for message in messages)
    for layer in full.source_layers:
        assert torch.allclose(
            resumed.jacobians[layer], full.jacobians[layer], atol=2e-3,
        )


def test_fit_jlens_missing_layer_topup_resumes_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import saklas.core.jlens as jlens_mod

    full = _StubSession().fit_jlens(
        _PROMPTS, force=True, source_layers=[0, 1],
    )
    _StubSession().fit_jlens(_PROMPTS, force=True, source_layers=[0])
    real_fit = jlens_mod.fit_jacobian_lens

    def _interrupt_topup(*args: Any, **kwargs: Any) -> Any:
        assert kwargs.get("source_layers") == [1]
        prompts = list(args[2])
        args = (*args[:2], prompts[:2], *args[3:])
        kwargs["input_id_rows"] = kwargs["input_id_rows"][:2]
        kwargs["checkpoint_every"] = 1
        real_fit(*args, **kwargs)
        raise RuntimeError("simulated topup interruption")

    monkeypatch.setattr(jlens_mod, "fit_jacobian_lens", _interrupt_topup)
    with pytest.raises(RuntimeError, match="topup interruption"):
        _StubSession().fit_jlens(_PROMPTS, source_layers=[0, 1])

    checkpoint = load_lens_checkpoint(_MODEL_ID)
    assert checkpoint is not None
    assert checkpoint[0].source_layers == [1]
    assert checkpoint[0].n_prompts == 2

    monkeypatch.setattr(jlens_mod, "fit_jacobian_lens", real_fit)
    messages: list[str] = []
    resumed = _StubSession().fit_jlens(
        _PROMPTS, source_layers=[0, 1], on_progress=messages.append,
    )
    assert any("missing-layer checkpoint at 2" in message for message in messages)
    for layer in full.source_layers:
        assert torch.allclose(
            resumed.jacobians[layer], full.jacobians[layer], atol=2e-3,
        )


def test_fit_jlens_drops_short_prompts() -> None:
    s = _StubSession()
    messages: list[str] = []
    fitted = s.fit_jlens(["tiny", *_PROMPTS], on_progress=messages.append)
    assert fitted.n_prompts == len(_PROMPTS)
    assert any("dropped 1 too-short prompts" in m for m in messages)


def test_jlens_readout_shape_and_default_position() -> None:
    s = _StubSession()
    s.fit_jlens(_PROMPTS)
    seen_pool: list[int | None] = []
    import saklas.core.vectors as _vectors

    real_capture = _vectors._capture_all_hidden_states

    def _spy(model: Any, layers: Any, ids: torch.Tensor, **kw: Any) -> Any:
        pool = kw.get("pool_index")
        seen_pool.append(int(pool) if pool is not None else None)
        return real_capture(model, layers, ids, **kw)

    _vectors._capture_all_hidden_states = _spy
    try:
        out = s.jlens_readout("a prompt that is long enough.", top_k=3)
    finally:
        _vectors._capture_all_hidden_states = real_capture
    assert seen_pool == [len(s._tokenizer.encode("a prompt that is long enough.")) - 1]
    assert set(out) == {0, 1}  # 3-layer toy: sources are 0 and 1
    for rows in out.values():
        assert len(rows) == 1  # default: final position only
        assert len(rows[0]) == 3
        token, logprob = rows[0][0]
        assert isinstance(token, str) and logprob <= 0.0


def test_jlens_readout_aggregate_rides_same_logits() -> None:
    s = _StubSession()
    s.fit_jlens(_PROMPTS)
    result = s.jlens_readout(
        "a prompt that is long enough.", top_k=3, aggregate=True,
    )
    out, agg = result
    assert set(out) == {0, 1}
    # default position only → one aggregate list
    assert len(agg) == 1
    rows = agg[0]
    assert len(rows) == 3
    strengths = [r[1] for r in rows]
    assert strengths == sorted(strengths, reverse=True)
    for tok, strength, com, spread in rows:
        assert isinstance(tok, str)
        assert 0.0 <= strength <= 1.0
        assert 0.0 <= com <= 1.0
        assert spread >= 0.0
    # 3-layer toy: the 40-90% band keeps only L1, so the aggregate is the
    # single-layer degenerate case — com pinned at L1's normalized depth,
    # spread 0.
    depth_l1 = 1 / (3 - 1)
    for _, _, com, spread in rows:
        assert com == pytest.approx(depth_l1, abs=1e-6)
        assert spread == pytest.approx(0.0, abs=1e-6)


def test_jlens_readout_aggregate_multi_position() -> None:
    s = _StubSession()
    s.fit_jlens(_PROMPTS)
    result = s.jlens_readout(
        "a prompt that is long enough.", positions=[-2, -1], top_k=2,
        aggregate=True,
    )
    out, agg = result
    assert all(len(rows) == 2 for rows in out.values())
    assert len(agg) == 2
    assert all(len(rows) == 2 for rows in agg)


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
    seen_layers: list[list[int] | None] = []
    real_token_direction = lens.token_direction

    def _spy_token_direction(
        token_id: int,
        unembed: torch.Tensor,
        *,
        layers: list[int] | None = None,
    ) -> dict[int, torch.Tensor]:
        seen_layers.append(layers)
        return real_token_direction(token_id, unembed, layers=layers)

    lens.token_direction = _spy_token_direction  # type: ignore[method-assign]
    name = s.register_jlens_direction("g")  # 'g' round-trips in the toy vocab
    assert name == "jlens/g"
    assert seen_layers == [[1]]
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
    assert "J" not in s._live_lens
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

    step = SaklasSession._live_lens_readout_step(s)  # type: ignore[arg-type]
    assert step is not None
    out, agg = step
    assert set(out) == {0, 1}
    for row in out.values():
        assert len(row) == 3
        assert all(isinstance(tok, str) for tok, _ in row)
    # display scores are per-layer softmax probabilities, descending — the
    # one strength unit every lens surface reports
    scores = [sc for _, sc in out[0]]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= sc <= 1.0 for sc in scores)
    # the aggregate chip list rides the same step: top_k rows of
    # (token, strength, com, spread) with strength descending in [0, 1]
    # and com/spread valid normalized depths
    assert len(agg) == 3
    strengths = [srow[1] for srow in agg]
    assert strengths == sorted(strengths, reverse=True)
    for tok, strength, com, spread in agg:
        assert isinstance(tok, str)
        assert 0.0 <= strength <= 1.0
        assert 0.0 <= com <= 1.0
        assert spread >= 0.0


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

    seen_lens: list[tuple[int, int | None]] = []
    import saklas.core.vectors as _vectors

    real_capture = _vectors._capture_all_hidden_states

    def _spy(model: Any, layers: Any, ids: torch.Tensor, **kw: Any) -> Any:
        pool = kw.get("pool_index")
        seen_lens.append((int(ids.shape[1]), int(pool) if pool is not None else None))
        return real_capture(model, layers, ids, **kw)

    _vectors._capture_all_hidden_states = _spy
    try:
        out = s.jlens_token_readout(node_id, 3, top_k=4)
    finally:
        _vectors._capture_all_hidden_states = real_capture

    prompt_len = len(s._tokenizer.encode(_PROMPT_RENDER))
    # readout position: the forward that PRODUCED the clicked token —
    # prompt + raw[:3], never including the clicked token itself.
    assert seen_lens == [(prompt_len + 3, prompt_len + 2)]
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
    # the aggregate block rides the same logits, band-restricted (L1 only
    # in the 3-layer toy → single-layer degenerate com/spread)
    assert len(out["aggregate"]) == 4
    for tok, strength, com, spread in out["aggregate"]:
        assert isinstance(tok, str)
        assert 0.0 <= strength <= 1.0
        assert com == pytest.approx(1 / (3 - 1), abs=1e-6)
        assert spread == pytest.approx(0.0, abs=1e-6)
    # user_role/assistant_role of the replayed render come off the nodes
    assert s.prepare_calls[0]["input"] == "a user turn"
    assert s.prepare_calls[0]["raw"] is False


def test_jlens_token_readout_index_zero_reads_prompt_only() -> None:
    s = _TreeStubSession()
    s.fit_jlens(_PROMPTS)
    node_id = _tree_with_assistant(s, s._tokenizer.encode("abc"))

    seen_lens: list[tuple[int, int | None]] = []
    import saklas.core.vectors as _vectors

    real_capture = _vectors._capture_all_hidden_states

    def _spy(model: Any, layers: Any, ids: torch.Tensor, **kw: Any) -> Any:
        pool = kw.get("pool_index")
        seen_lens.append((int(ids.shape[1]), int(pool) if pool is not None else None))
        return real_capture(model, layers, ids, **kw)

    _vectors._capture_all_hidden_states = _spy
    try:
        s.jlens_token_readout(node_id, 0, top_k=2)
    finally:
        _vectors._capture_all_hidden_states = real_capture
    prompt_len = len(s._tokenizer.encode(_PROMPT_RENDER))
    assert seen_lens == [(prompt_len, prompt_len - 1)]


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
