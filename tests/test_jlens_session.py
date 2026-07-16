"""CPU tests for the session-level Jacobian-lens API (stub session).

The real ``SaklasSession`` methods are class-bound onto a light stub (the
established ``__new__``-stub pattern) so ``fit_jlens`` / ``jlens_readout`` /
``register_jlens_direction`` run against the toy model with no HF load.
"""

from __future__ import annotations

import hashlib
import json
import threading
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from saklas.core.jlens import (
    JacobianLens,
    JacobianLensError,
    LensNotFittedError,
    MultiTokenWordError,
)
from saklas.core.model import loaded_model_fingerprint, model_source_fingerprint
from saklas.core.loom import (
    InvalidNodeOperationError,
    LoomTree,
    Recipe,
    UnknownNodeError,
)
from saklas.core.session import SaklasSession
from saklas.core.steering_composer import SteeringComposer
from saklas.io.lens import (
    lens_checkpoint_paths,
    lens_paths,
    load_lens,
    load_lens_checkpoint,
    remove_lens,
    save_lens_checkpoint_accumulator,
)
from tests._jlens_toys import CharTokenizer, frozen_toy

_MODEL_ID = "toy/jlens-model"


def _save_checkpoint(
    partial: JacobianLens, model_id: str, *, base_n_prompts: int, **kwargs: Any,
) -> Path:
    """Publish a self-contained current checkpoint from estimator sums."""
    assert base_n_prompts == 0
    sums = {
        layer: matrix * partial.n_prompts
        for layer, matrix in partial.jacobians.items()
    }
    kwargs.setdefault(
        "consumed_prefix_sha256",
        hashlib.sha256(json.dumps({
            "corpus_sha256": kwargs.get("corpus_sha256"),
            "n_prompts": partial.n_prompts,
        }, sort_keys=True).encode()).hexdigest(),
    )
    return save_lens_checkpoint_accumulator(
        sums, partial.n_prompts, partial.d_model, model_id,
        base=None, **kwargs,
    )


class _StubSession:
    jlens = SaklasSession.jlens
    has_compatible_jlens = SaklasSession.has_compatible_jlens
    _require_jlens = SaklasSession._require_jlens
    fit_jlens = SaklasSession.fit_jlens
    jlens_readout = SaklasSession.jlens_readout
    _resolve_jlens_layers = SaklasSession._resolve_jlens_layers
    _resolve_jlens_source_layers = SaklasSession._resolve_jlens_source_layers
    _jlens_transport_stack = SaklasSession._jlens_transport_stack
    _jlens_readout_modules = SaklasSession._jlens_readout_modules
    _jlens_topk_rows = SaklasSession._jlens_topk_rows
    _jlens_logits_rows = SaklasSession._jlens_logits_rows
    _jlens_aggregate_rows = SaklasSession._jlens_aggregate_rows
    _jlens_decode_id = SaklasSession._jlens_decode_id
    _jlens_depths = SaklasSession._jlens_depths
    _jlens_depth_tensor = SaklasSession._jlens_depth_tensor
    _readout_long_tensor = SaklasSession._readout_long_tensor
    register_jlens_direction = SaklasSession.register_jlens_direction
    enable_live_lens = SaklasSession.enable_live_lens
    disable_live_lens = SaklasSession.disable_live_lens
    _live_lens_readout_step = SaklasSession._live_lens_readout_step
    _jlens_workspace_band = SaklasSession._jlens_workspace_band
    _add_lens_probe = SaklasSession._add_lens_probe
    _lens_probe_layers = SaklasSession._lens_probe_layers
    _score_lens_probes = SaklasSession._score_lens_probes
    _score_lens_gate_scalars = SaklasSession._score_lens_gate_scalars
    _effective_return_top_k = SaklasSession._effective_return_top_k
    _active_jlens_source_label = SaklasSession._active_jlens_source_label
    _select_tensor_rows = staticmethod(SaklasSession._select_tensor_rows)
    # Lens state lives on the LensInstrument (constructed in __init__);
    # borrow the session's delegating property objects so the plain
    # attribute assignments below route into the instrument exactly as the
    # real session's do.
    _lens_probes = SaklasSession._lens_probes
    _live_lens = SaklasSession._live_lens
    _lens_step_stash = SaklasSession._lens_step_stash
    _last_lens_step_readings = SaklasSession._last_lens_step_readings
    _live_lens_active_for_generation = (
        SaklasSession._live_lens_active_for_generation
    )
    _generation_jlens = SaklasSession._generation_jlens
    _generation_jlens_active = SaklasSession._generation_jlens_active
    _close_instrument_runs = SaklasSession._close_instrument_runs

    def __init__(self, *, n_layers: int = 3) -> None:
        from saklas.core.instruments.geometry import GeometryInstrument
        from saklas.core.instruments.lens import LensInstrument
        from saklas.core.instruments.sae import SaeInstrument

        self._lens_instrument = LensInstrument(self)  # type: ignore[arg-type]
        # ``_begin_capture`` consumes every family's ``plan()`` demand, so
        # the stub carries all three real instruments like the session.
        self._geometry_instrument = GeometryInstrument(self)  # type: ignore[arg-type]
        self._sae_instrument = SaeInstrument(self)  # type: ignore[arg-type]
        model = frozen_toy(n_layers=n_layers)
        self._model = model
        self._tokenizer = CharTokenizer()
        self._layers = model.model.layers
        self._device = torch.device("cpu")
        self._profiles: dict[str, Any] = {}
        self._jlens: Any = None
        self._jlens_identity: Any = None
        self._generation_jlens = None
        self._generation_jlens_active = False
        self._live_lens = None
        self._capture: Any = None
        self._capture_buffers: dict[int, torch.Tensor] = {}
        self._compiled_clean_eligible = False
        self._steering_uses_compiled_offsets = False
        self._steering: Any = None
        self._lens_probes = {}
        self._live_lens_active_for_generation = True
        self._live_sae: Any = None
        self._sae_probes: dict[str, Any] = {}
        self._probe_hash_cache: dict[str, str] = {}
        self._lens_step_stash = None
        self._last_lens_step_readings = None
        self._jlens_readout_module_cache: Any = None
        self._jlens_device_cache: dict[Any, Any] = {}
        self._jlens_depths_cache: dict[Any, list[float]] = {}
        self._jlens_depth_tensor_cache: dict[Any, torch.Tensor] = {}
        self._readout_long_tensor_cache: dict[Any, torch.Tensor] = {}
        self._monitor: Any = None
        self.model_id = _MODEL_ID
        self._steering_composer = SteeringComposer(self)  # type: ignore[arg-type]

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


def test_terminal_checkpoint_is_promoted_without_second_tensor_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import saklas.io.lens as lens_io
    from saklas.io import packs

    real_save = lens_io._save_fp32_square_safetensors_atomic
    writes = 0
    hashes = 0

    def _counting_save(*args: Any, **kwargs: Any) -> Any:
        nonlocal writes
        writes += 1
        return real_save(*args, **kwargs)

    monkeypatch.setattr(
        lens_io, "_save_fp32_square_safetensors_atomic", _counting_save,
    )
    real_hash = packs.hash_file

    def _counting_hash(path: Path) -> str:
        nonlocal hashes
        hashes += 1
        return real_hash(path)

    monkeypatch.setattr(packs, "hash_file", _counting_hash)
    fitted = _StubSession().fit_jlens(
        _PROMPTS[:2], force=True, checkpoint_every=2,
    )

    assert fitted.n_prompts == 2
    # One checkpoint shard per fitted layer; promotion only switches the
    # durable sidecar pointer and performs no second serialization pass.
    assert writes == len(fitted.source_layers)
    assert hashes == 0
    assert load_lens(_MODEL_ID) is not None
    assert load_lens_checkpoint(_MODEL_ID) is None


def test_refit_rebuilds_live_lens_probes_and_evicts_directions() -> None:
    s = _StubSession()
    s.fit_jlens(_PROMPTS, source_layers=[0])
    s.enable_live_lens(layers=[0])
    old_stack = s._live_lens["J_stack"]
    s._profiles["jlens/a"] = {0: torch.ones(4)}
    s._lens_probes["jlens/a"] = {
        "word": "a", "token_id": 1, "layers": [0],
    }

    fitted = s.fit_jlens(_PROMPTS, source_layers=[1], force=True)

    assert fitted.source_layers == [1]
    assert s._live_lens["layers"] == [1]
    assert s._live_lens["uses_all_layers"] is True
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


def test_fit_jlens_already_done_short_circuits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s = _StubSession()
    first = s.fit_jlens(_PROMPTS)
    messages: list[str] = []
    import saklas.io.lens as lens_io

    monkeypatch.setattr(
        lens_io, "load_lens",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("same-session no-op should reuse the resident lens")
        ),
    )
    again = s.fit_jlens(_PROMPTS, on_progress=messages.append)
    assert any("nothing to do" in m for m in messages)
    for layer in first.source_layers:
        assert torch.equal(first.jacobians[layer], again.jacobians[layer])


def test_subset_noop_keeps_full_durable_lens_resident() -> None:
    s = _StubSession()
    full = s.fit_jlens(_PROMPTS, source_layers=[0, 1], force=True)

    selected = s.fit_jlens(_PROMPTS, source_layers=[1])

    assert selected.source_layers == [1]
    assert full.source_layers == [0, 1]
    assert s._jlens.source_layers == [0, 1]
    assert s.jlens.source_layers == [0, 1]


def test_fresh_subset_noop_reads_only_requested_shard_and_preserves_disk_union(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import saklas.io.lens as lens_io

    _StubSession().fit_jlens(_PROMPTS, source_layers=[0, 1], force=True)
    payload_reads: list[str] = []
    real_safe_open = lens_io.safe_open

    class _TrackingSafeOpen:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._opened = real_safe_open(*args, **kwargs)
            self._tensors: Any = None

        def __enter__(self) -> "_TrackingSafeOpen":
            self._tensors = self._opened.__enter__()
            return self

        def __exit__(self, *args: Any) -> Any:
            return self._opened.__exit__(*args)

        def keys(self) -> Any:
            return self._tensors.keys()

        def get_slice(self, key: str) -> Any:
            return self._tensors.get_slice(key)

        def get_tensor(self, key: str) -> Any:
            payload_reads.append(key)
            return self._tensors.get_tensor(key)

    monkeypatch.setattr(lens_io, "safe_open", _TrackingSafeOpen)
    session = _StubSession()
    selected = session.fit_jlens(_PROMPTS, source_layers=[1])

    assert payload_reads == ["layer_1"]
    assert selected.source_layers == [1]
    monkeypatch.setattr(lens_io, "safe_open", real_safe_open)
    durable = load_lens(_MODEL_ID)
    assert durable is not None
    assert durable[0].source_layers == [0, 1]
    assert session.jlens.source_layers == [0, 1]


def test_fresh_partial_extension_rejects_before_payload_and_preserves_union(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gc
    import weakref

    import saklas.io.lens as lens_io

    writer = _StubSession()
    writer.fit_jlens(_PROMPTS[:2], source_layers=[0, 1], force=True)
    resident_ref = weakref.ref(writer._jlens)
    del writer
    gc.collect()
    assert resident_ref() is None

    payload_reads: list[str] = []
    real_load = lens_io._load_lens_verified

    def record_load(*args: Any, **kwargs: Any) -> Any:
        payload_reads.append("load")
        return real_load(*args, **kwargs)

    monkeypatch.setattr(lens_io, "_load_lens_verified", record_load)
    with pytest.raises(JacobianLensError, match="full durable layer set"):
        _StubSession().fit_jlens(_PROMPTS, source_layers=[0])

    assert payload_reads == []
    monkeypatch.setattr(lens_io, "_load_lens_verified", real_load)
    durable = load_lens(_MODEL_ID)
    assert durable is not None
    assert durable[0].source_layers == [0, 1]
    assert durable[0].n_prompts == 2


def test_overlapping_partial_extension_requires_explicit_replacement() -> None:
    _StubSession(n_layers=4).fit_jlens(
        _PROMPTS[:2], source_layers=[0, 1], force=True,
    )

    with pytest.raises(JacobianLensError, match="full durable layer set"):
        _StubSession(n_layers=4).fit_jlens(
            _PROMPTS, source_layers=[1, 2],
        )

    durable = load_lens(_MODEL_ID)
    assert durable is not None
    assert durable[0].source_layers == [0, 1]
    assert durable[0].n_prompts == 2

    replaced = _StubSession(n_layers=4).fit_jlens(
        _PROMPTS, source_layers=[1, 2], force=True,
    )
    assert replaced.source_layers == [1, 2]
    durable = load_lens(_MODEL_ID)
    assert durable is not None
    assert durable[0].source_layers == [1, 2]
    assert durable[0].n_prompts == len(_PROMPTS)


def test_exact_noop_reaps_checkpoint_left_after_final_publication() -> None:
    s = _StubSession()
    full = s.fit_jlens(_PROMPTS, force=True)
    loaded = load_lens(_MODEL_ID)
    assert loaded is not None
    sidecar = loaded[1]
    _save_checkpoint(
        full, _MODEL_ID, base_n_prompts=0,
        corpus_spec=str(sidecar["corpus_spec"]),
        corpus_sha256=str(sidecar["corpus_sha256"]),
        corpus_hash_kind=str(sidecar["corpus_hash_kind"]),
        seq_len=int(sidecar["seq_len"]),
        dim_batch=int(sidecar["dim_batch"]),
        skip_first=int(sidecar["skip_first_positions"]),
        model_fingerprint=str(sidecar["model_fingerprint"]),
    )
    assert load_lens_checkpoint(_MODEL_ID) is not None

    selected = s.fit_jlens(_PROMPTS)

    assert selected.source_layers == full.source_layers
    assert load_lens_checkpoint(_MODEL_ID) is None


def test_resident_noop_does_not_reap_checkpoint_when_final_payload_corrupt() -> None:
    s = _StubSession()
    full = s.fit_jlens(_PROMPTS, force=True)
    loaded = load_lens(_MODEL_ID)
    assert loaded is not None
    sidecar = loaded[1]
    _save_checkpoint(
        full, _MODEL_ID, base_n_prompts=0,
        corpus_spec=str(sidecar["corpus_spec"]),
        corpus_sha256=str(sidecar["corpus_sha256"]),
        corpus_hash_kind=str(sidecar["corpus_hash_kind"]),
        seq_len=int(sidecar["seq_len"]),
        dim_batch=int(sidecar["dim_batch"]),
        skip_first=int(sidecar["skip_first_positions"]),
        model_fingerprint=str(sidecar["model_fingerprint"]),
    )
    final_tensor, _ = lens_paths(_MODEL_ID)
    payload = bytearray(final_tensor.read_bytes())
    payload[-1] ^= 1
    final_tensor.write_bytes(payload)

    # The resident lens makes the fit a no-op, but it is not proof that the
    # current on-disk payload is sound enough to discard the recovery point.
    selected = s.fit_jlens(_PROMPTS)

    assert selected.source_layers == full.source_layers
    assert load_lens_checkpoint(_MODEL_ID) is not None


def test_overlapping_topup_keeps_preserved_extra_layer_resident() -> None:
    s = _StubSession(n_layers=4)
    s.fit_jlens(_PROMPTS, source_layers=[0, 1], force=True)

    selected = s.fit_jlens(_PROMPTS, source_layers=[1, 2])

    assert selected.source_layers == [1, 2]
    assert s._jlens.source_layers == [0, 1, 2]
    assert s.jlens.source_layers == [0, 1, 2]


def test_resident_lens_is_not_reused_after_external_artifact_replacement() -> None:
    corpus_b = [f"replacement corpus prompt {i} with enough content" for i in range(4)]
    session_a = _StubSession()
    resident_a = session_a.fit_jlens(_PROMPTS, force=True)
    disk_b = _StubSession().fit_jlens(corpus_b, force=True)
    assert any(
        not torch.equal(resident_a.jacobians[layer], disk_b.jacobians[layer])
        for layer in resident_a.source_layers
    )

    refreshed = session_a.fit_jlens(corpus_b)

    for layer in disk_b.source_layers:
        assert torch.allclose(
            refreshed.jacobians[layer], disk_b.jacobians[layer], atol=2e-3,
        )


def test_jlens_property_refreshes_and_evicts_after_external_lifecycle() -> None:
    corpus_b = [f"replacement property prompt {i} with enough content" for i in range(4)]
    session_a = _StubSession()
    resident_a = session_a.fit_jlens(_PROMPTS, force=True)
    session_a._profiles["jlens/example"] = {0: torch.ones(6)}
    session_a._lens_probes["jlens/example"] = {
        "word": "example", "token_id": 1, "layers": [0, 1],
    }
    disk_b = _StubSession().fit_jlens(corpus_b, force=True)

    assert session_a.has_compatible_jlens()
    refreshed = session_a._jlens

    assert refreshed is not None
    assert any(
        not torch.equal(resident_a.jacobians[layer], refreshed.jacobians[layer])
        for layer in refreshed.source_layers
    )
    for layer in disk_b.source_layers:
        assert torch.allclose(
            refreshed.jacobians[layer], disk_b.jacobians[layer], atol=2e-3,
        )

    assert remove_lens(_MODEL_ID)
    assert not session_a.has_compatible_jlens()
    assert session_a.jlens is None
    assert "jlens/example" not in session_a._profiles
    assert session_a._lens_probes["jlens/example"]["layers"] == []
    assert session_a._live_lens is None


def test_generation_boundary_refreshes_external_lens_once() -> None:
    """A generation snapshots the current pointer before fixing capture layers."""
    session_a = _StubSession()
    resident_a = session_a.fit_jlens(_PROMPTS, force=True)
    session_a._lens_probes["jlens/example"] = {
        "word": "example", "token_id": 1,
        "layers": list(resident_a.source_layers),
    }

    corpus_b = [
        f"replacement boundary prompt {i} with enough content" for i in range(4)
    ]
    disk_b = _StubSession().fit_jlens(corpus_b, force=True)

    class _Capture:
        def clear(self) -> None:
            pass

        def attach(self, _layers: Any, _indices: list[int]) -> None:
            pass

        def set_aggregate_tail(self, _depth: int) -> None:
            pass

    session_a._capture = _Capture()
    session_a._capture_buffers = {}
    session_a._compiled_clean_eligible = False
    session_a._steering_uses_compiled_offsets = False
    session_a._steering = SimpleNamespace(all_fast_path=lambda: True)
    session_a._monitor = SimpleNamespace(
        probe_names=[],
        enable_curved_warm=lambda _enabled: None,
    )
    session_a._live_sae = None
    session_a._sae_probes = {}

    SaklasSession._begin_capture(
        cast(Any, session_a), final_probe_aggregate=True,
    )

    snap = session_a._generation_jlens
    assert snap is not None and snap is session_a._jlens
    assert session_a._generation_jlens_active is True
    assert any(
        not torch.equal(resident_a.jacobians[layer], snap.jacobians[layer])
        for layer in snap.source_layers
    )
    for layer in disk_b.source_layers:
        assert torch.allclose(
            snap.jacobians[layer], disk_b.jacobians[layer], atol=2e-3,
        )

    # An external deletion is likewise observed before the next generation,
    # and the validated missing state is pinned (no per-token retry loop).
    assert remove_lens(_MODEL_ID)
    SaklasSession._begin_capture(
        cast(Any, session_a), final_probe_aggregate=True,
    )
    assert session_a._generation_jlens_active is True
    assert session_a._generation_jlens is None
    assert session_a._jlens is None
    assert session_a._lens_probes["jlens/example"]["layers"] == []


def test_external_lens_replacement_plans_and_freezes_refreshed_layers() -> None:
    """The refresh-then-plan boundary (sol's slice-B finding 2): an external
    replacement lens on a DISJOINT layer set must be adopted before the
    capture plan and the frozen binding are taken — pairing the new lens
    with stale layers KeyErrors in the transport stack."""
    session_a = _StubSession()
    session_a.fit_jlens(_PROMPTS, source_layers=[0], force=True)
    session_a._lens_probes["jlens/example"] = {
        "word": "example", "token_id": 1, "layers": [0],
    }

    # Another process replaces the artifact with a disjoint-layer fit.
    corpus_b = [
        f"replacement disjoint prompt {i} with enough content" for i in range(4)
    ]
    disk_b = _StubSession().fit_jlens(corpus_b, source_layers=[1], force=True)
    assert list(disk_b.source_layers) == [1]

    attached_layers: list[int] = []

    class _Capture:
        def clear(self) -> None:
            pass

        def attach(self, _layers: Any, indices: list[int]) -> None:
            attached_layers.extend(indices)

        def set_aggregate_tail(self, _depth: int) -> None:
            pass

    session_a._capture = _Capture()
    session_a._capture_buffers = {}
    session_a._compiled_clean_eligible = False
    session_a._steering_uses_compiled_offsets = False
    session_a._steering = SimpleNamespace(all_fast_path=lambda: True)
    session_a._monitor = SimpleNamespace(
        probe_names=[],
        enable_curved_warm=lambda _enabled: None,
    )
    session_a._live_sae = None
    session_a._sae_probes = {}

    ok = SaklasSession._begin_capture(
        cast(Any, session_a), final_probe_aggregate=True,
    )

    assert ok is True
    # The adoption rewrote the live registry BEFORE planning/freezing:
    # the capture plan, the frozen binding, and the pinned lens all agree
    # on the replacement's layer set.
    assert session_a._lens_probes["jlens/example"]["layers"] == [1]
    assert attached_layers == [1]
    run = session_a._lens_instrument.current_run
    assert run.pinned is True
    assert list(run.binding.specs["jlens/example"]["layers"]) == [1]
    assert list(run.lens.source_layers) == [1]
    # And the pinned lens can actually score the frozen band — the stale
    # pairing raised KeyError(0) out of the transport stack.
    hidden_dim = int(session_a._model.lm_head.weight.shape[1])
    hidden = {1: torch.randn(hidden_dim)}
    readings = session_a._lens_instrument.score_probes(hidden)
    assert "jlens/example" in readings


def test_prepare_is_the_refresh_site_and_supplies_the_pin() -> None:
    """The refresh/pin protocol step: a bare ``prepare()`` — no session
    special-casing around it — adopts an external replacement lens
    (rewriting the live probe layer lists before any plan is taken) and
    returns the ``LensPrep`` snapshot that ``plan``/``bind`` consume."""
    from saklas.core.instruments.types import LensPrep, ReadRequest

    session_a = _StubSession()
    session_a.fit_jlens(_PROMPTS, source_layers=[0], force=True)
    session_a._lens_probes["jlens/example"] = {
        "word": "example", "token_id": 1, "layers": [0],
    }
    assert session_a.jlens is not None  # resident before the swap

    corpus_b = [
        f"replacement disjoint prompt {i} with enough content" for i in range(4)
    ]
    _StubSession().fit_jlens(corpus_b, source_layers=[1], force=True)

    inst = session_a._lens_instrument
    prep = inst.prepare(ReadRequest(final_aggregate=True))
    assert isinstance(prep, LensPrep)
    assert prep.pinned is True
    assert prep.lens is not None and list(prep.lens.source_layers) == [1]
    # The adoption rewrote the live registry inside prepare, and the prep's
    # spec snapshot carries the layers derived from the prepared identity.
    assert session_a._lens_probes["jlens/example"]["layers"] == [1]
    assert list(prep.specs["jlens/example"]["layers"]) == [1]
    run = inst.bind(inst.plan(prep), prep)
    assert run.pinned is True and run.lens is prep.lens
    assert list(run.binding.specs["jlens/example"]["layers"]) == [1]
    session_a._close_instrument_runs()


def test_interleaved_adoption_cannot_desync_plan_from_pin() -> None:
    """sol's round-3 P1: prepare pins lens A; before plan/bind, the
    un-locked ``has_compatible_jlens`` (the session-info route runs it
    without the generation lock) observes a replacement lens B and adopts
    it, rewriting the live probe layer lists.  Because plan and bind
    consume the prep's snapshot — never the live registry — the run still
    measures A's layers with A pinned; the old registry reread paired A
    with B's layers and KeyErrored in the transport stack."""
    from saklas.core.instruments.types import ReadRequest

    session_a = _StubSession()
    session_a.fit_jlens(_PROMPTS, source_layers=[0], force=True)
    session_a._lens_probes["jlens/example"] = {
        "word": "example", "token_id": 1, "layers": [0],
    }
    inst = session_a._lens_instrument
    prep = inst.prepare(ReadRequest(final_aggregate=True))
    assert prep.lens is not None and list(prep.lens.source_layers) == [0]

    # Another process swaps the artifact; the un-locked path adopts it
    # INSIDE the prepare->bind window.
    corpus_b = [
        f"replacement disjoint prompt {i} with enough content" for i in range(4)
    ]
    _StubSession().fit_jlens(corpus_b, source_layers=[1], force=True)
    assert session_a.has_compatible_jlens() is True  # adopts B
    assert session_a._lens_probes["jlens/example"]["layers"] == [1]

    plan = inst.plan(prep)
    run = inst.bind(plan, prep)
    # Plan layers, frozen binding, and the pinned lens all agree on A.
    assert set(plan.latest_layers) == {0}
    assert list(run.binding.specs["jlens/example"]["layers"]) == [0]
    assert run.lens is prep.lens
    hidden_dim = int(session_a._model.lm_head.weight.shape[1])
    readings = inst.score_probes({0: torch.randn(hidden_dim)})
    assert "jlens/example" in readings
    session_a._close_instrument_runs()


def test_bound_run_reads_prepare_time_live_state() -> None:
    """The live-readout runtime dict is part of the prep snapshot: an
    interleaved adoption rebuilds the instrument-level ``live`` against
    the NEW lens, and a bound run must keep reading the state that
    matches its pin.  Idle runs pass through to the live config."""
    from saklas.core.instruments.types import ReadRequest

    session = _StubSession()
    inst = session._lens_instrument
    state_a = {"layers": [0]}
    inst.live = state_a
    prep = inst.prepare(ReadRequest(final_aggregate=False, live=True))
    assert prep.live_state is state_a

    inst.live = {"layers": [1]}  # the adoption rebuild lands in the window
    inst.bind(inst.plan(prep), prep)
    assert inst._measurement_live() is state_a
    session._close_instrument_runs()
    assert inst._measurement_live() is inst.live
    inst.live = None


def test_lens_state_lock_serializes_getter_against_snapshot() -> None:
    """The one lens-state boundary (sol's round-4 P1): the getter's
    refresh/adopt/evict transaction and ``prepare``'s snapshot hold the
    same reentrant lock, so an un-locked read (the session-info route)
    cannot land inside a generation boundary's snapshot — it blocks
    until the snapshot completes."""
    session = _StubSession()
    session.fit_jlens(_PROMPTS, force=True)
    inst = session._lens_instrument

    entered = threading.Event()
    seen: list[Any] = []

    def _getter_read() -> None:
        entered.set()
        seen.append(session.jlens)

    with inst.state_lock:
        reader = threading.Thread(target=_getter_read)
        reader.start()
        assert entered.wait(timeout=5.0)
        reader.join(timeout=0.2)
        assert reader.is_alive()  # blocked on the lens-state lock
    reader.join(timeout=5.0)
    assert not reader.is_alive()
    assert seen and seen[0] is not None


def test_prepare_pin_demand_formula() -> None:
    """Pin demand = a live readout, or attached probes with a final
    aggregate or a lens gate — the one formula both generation
    boundaries (``_begin_capture`` and the batch preamble) reduce to."""
    from saklas.core.instruments.types import ReadRequest

    session = _StubSession()
    session.fit_jlens(_PROMPTS, force=True)
    inst = session._lens_instrument

    # No probes, no live consumer -> unpinned, no disk read taken.
    prep = inst.prepare(ReadRequest(final_aggregate=True))
    assert prep.pinned is False and prep.lens is None

    # Probes + final aggregate (also the batch shape) -> pinned.
    session._lens_probes["jlens/example"] = {
        "word": "example", "token_id": 1, "layers": [0],
    }
    prep = inst.prepare(ReadRequest(final_aggregate=True, batch=True))
    assert prep.pinned is True and prep.lens is not None

    # Final readings disabled: dormant probes do not pin — a lens gate does.
    prep = inst.prepare(ReadRequest(final_aggregate=False))
    assert prep.pinned is False
    prep = inst.prepare(ReadRequest(
        final_aggregate=False,
        gate_keys=frozenset({"jlens/example"}),
    ))
    assert prep.pinned is True

    # A live-readout consumer pins with an empty probe registry.
    del session._lens_probes["jlens/example"]
    inst.live = {"layers": [0]}
    prep = inst.prepare(ReadRequest(final_aggregate=False, live=True))
    assert prep.pinned is True and prep.request.live is True
    inst.live = None


def test_prepare_on_a_bound_run_raises() -> None:
    """The jlens getter short-circuits on a bound run's pin flag, so a
    prepare taken without closing the prior run would silently skip the
    refresh — every family rejects it instead."""
    from saklas.core.instruments.types import ReadRequest

    session = _StubSession()
    session._monitor = SimpleNamespace(probe_names=[])
    request = ReadRequest(final_aggregate=False)
    instruments = (
        session._lens_instrument,
        session._sae_instrument,
        session._geometry_instrument,
    )
    for inst in instruments:
        prep = inst.prepare(request)
        inst.bind(inst.plan(prep), prep)
        with pytest.raises(RuntimeError, match="bound run"):
            inst.prepare(request)
    session._close_instrument_runs()
    for inst in instruments:
        inst.prepare(request)  # idle runs again — every prepare passes


def test_bind_and_plan_reject_foreign_or_missing_preps() -> None:
    """The transaction contract is enforced, not advisory: a bare
    ``bind(plan)`` is a TypeError, a wrong-family prep (including a
    wrong-family ``LensPrep``) is rejected by every family, and plans are
    validated for family provenance too."""
    from saklas.core.instruments.types import (
        InstrumentPlan,
        InstrumentPrep,
        LensPrep,
        ReadRequest,
    )

    session = _StubSession()
    session._monitor = SimpleNamespace(probe_names=[])
    request = ReadRequest(final_aggregate=False)

    lens = session._lens_instrument
    lens_prep = lens.prepare(request)
    with pytest.raises(TypeError):
        lens.bind(InstrumentPlan(family="lens"))  # type: ignore[call-arg]  # bare bind: prep required
    with pytest.raises(TypeError, match="LensPrep"):
        lens.bind(InstrumentPlan(family="lens"), InstrumentPrep(family="lens"))
    with pytest.raises(TypeError, match="LensPrep"):
        lens.bind(
            InstrumentPlan(family="lens"), LensPrep(family="sae"),
        )
    with pytest.raises(TypeError, match="LensPrep"):
        lens.plan(InstrumentPrep(family="lens"))
    with pytest.raises(ValueError, match="plan family"):
        lens.bind(InstrumentPlan(family="sae"), lens_prep)
    assert lens.current_run.bound is False

    sae = session._sae_instrument
    sae_prep = sae.prepare(request)
    with pytest.raises(TypeError):
        sae.bind(InstrumentPlan(family="sae"))  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="family"):
        sae.bind(InstrumentPlan(family="sae"), InstrumentPrep(family="lens"))
    with pytest.raises(TypeError, match="family"):
        sae.plan(InstrumentPrep(family="geometry"))
    with pytest.raises(ValueError, match="plan family"):
        sae.bind(InstrumentPlan(family="lens"), sae_prep)
    assert sae.current_run.bound is False

    geometry = session._geometry_instrument
    geometry_prep = geometry.prepare(request)
    with pytest.raises(TypeError):
        geometry.bind(InstrumentPlan(family="geometry"))  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="family"):
        geometry.bind(
            InstrumentPlan(family="geometry"), InstrumentPrep(family="sae"),
        )
    with pytest.raises(TypeError, match="family"):
        geometry.plan(InstrumentPrep(family="lens"))
    with pytest.raises(ValueError, match="plan family"):
        geometry.bind(InstrumentPlan(family="lens"), geometry_prep)
    assert geometry.current_run.bound is False


def test_bind_rejects_same_family_plan_prep_crossing() -> None:
    """sol's round-4 P2: a plan derived from prep A must not bind with
    prep B of the same family — the session's capture union would retain
    A's layers while the run measures B.  The per-preparation token the
    plan echoes is compared at bind, in every family (a hand-built plan,
    which carries no token, is rejected the same way)."""
    from saklas.core.instruments.types import InstrumentPlan, ReadRequest

    session = _StubSession()
    session._monitor = SimpleNamespace(probe_names=[])
    request = ReadRequest(final_aggregate=False)
    for inst in (
        session._lens_instrument,
        session._sae_instrument,
        session._geometry_instrument,
    ):
        prep_a = inst.prepare(request)
        prep_b = inst.prepare(request)
        plan_a = inst.plan(prep_a)
        assert plan_a.prep_token == prep_a.token != prep_b.token
        with pytest.raises(ValueError, match="prep_token"):
            inst.bind(plan_a, prep_b)
        with pytest.raises(ValueError, match="prep_token"):
            inst.bind(InstrumentPlan(family=inst.family), prep_a)
        assert inst.current_run.bound is False
        inst.bind(plan_a, prep_a)  # the matched pair binds
        session._close_instrument_runs()


def test_prepare_rejects_a_layerless_pin_demanded_lens() -> None:
    """A pin-demanded lens without ``source_layers`` is structurally
    broken (layers align captures, specs, and Jacobians) — it fails at
    the prepare boundary, not as a mid-generation KeyError."""
    from saklas.core.instruments.types import ReadRequest

    class _BrokenLensStub(_StubSession):
        _broken_lens: Any

        @property
        def jlens(self) -> Any:
            return self._broken_lens

    session = _BrokenLensStub()
    session._broken_lens = object()  # no source_layers
    inst = session._lens_instrument
    inst.live = {"layers": [0]}
    with pytest.raises(RuntimeError, match="source_layers"):
        inst.prepare(ReadRequest(final_aggregate=False, live=True))


def test_generation_lens_snapshot_avoids_per_token_disk_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Probe and gate scoring share the boundary snapshot without shard opens."""
    import saklas.io.lens as lens_io

    session = _StubSession()
    lens = session.fit_jlens(_PROMPTS, force=True)
    layers = list(lens.source_layers)
    session._lens_probes["jlens/a"] = {
        "word": "a", "token_id": 1, "layers": layers,
    }
    session._generation_jlens = lens
    session._generation_jlens_active = True
    session._live_lens_active_for_generation = False

    monkeypatch.setattr(
        lens_io,
        "load_lens_sidecar",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("per-token lens scoring reopened the disk pointer")
        ),
    )
    assert session.jlens is lens
    assert session.has_compatible_jlens()

    vocab = int(session._model.lm_head.weight.shape[0])
    probabilities = torch.softmax(torch.randn(len(layers), vocab), dim=-1)
    for _ in range(3):
        readings = session._score_lens_probes(
            {}, probabilities=probabilities, layers=layers,
        )
        assert "jlens/a" in readings

    hidden = {
        layer: torch.randn(lens.d_model)
        for layer in layers
    }
    session._capture = SimpleNamespace(latest_per_layer=lambda: hidden)
    for _ in range(3):
        assert "jlens/a" in session._score_lens_gate_scalars({"jlens/a"})


def test_fit_jlens_serializes_complete_cross_session_transaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered: list[str] = []
    first_entered = threading.Event()
    release_first = threading.Event()

    def _transaction(self: _StubSession, prompts: object, **kwargs: object) -> str:
        del prompts, kwargs
        entered.append(self.model_id)
        if len(entered) == 1:
            first_entered.set()
            assert release_first.wait(timeout=1.0)
        return "done"

    monkeypatch.setattr(SaklasSession, "_fit_jlens_transaction", _transaction)
    first = _StubSession()
    second = _StubSession()
    results: list[str] = []
    thread_a = threading.Thread(target=lambda: results.append(first.fit_jlens(_PROMPTS)))
    thread_b = threading.Thread(target=lambda: results.append(second.fit_jlens(_PROMPTS)))
    thread_a.start()
    assert first_entered.wait(timeout=1.0)
    thread_b.start()
    assert len(entered) == 1
    release_first.set()
    thread_a.join(timeout=1.0)
    thread_b.join(timeout=1.0)
    assert len(entered) == 2
    assert results == ["done", "done"]


def test_fresh_fit_does_not_promote_stale_incompatible_checkpoint() -> None:
    prompts = _PROMPTS[:2]
    tokenizer = CharTokenizer()
    consumed = [
        [int(tok) for tok in tokenizer(
            prompt, return_tensors="pt",
        )["input_ids"][0].tolist()]
        for prompt in prompts
    ]
    corpus_sha = hashlib.sha256(repr(consumed).encode("utf-8")).hexdigest()
    stale = JacobianLens(
        {layer: torch.full((6, 6), 99.0) for layer in (0, 1)},
        n_prompts=2, d_model=6,
    )
    _save_checkpoint(
        stale, _MODEL_ID,
        base_n_prompts=0,
        corpus_spec="stale",
        corpus_sha256=corpus_sha,
        corpus_hash_kind="token_ids_v1",
        seq_len=128,
        dim_batch=8,
        skip_first=16,
        model_fingerprint="WRONG",
    )

    fitted = _StubSession().fit_jlens(prompts, checkpoint_every=25)
    durable = load_lens(_MODEL_ID)

    assert durable is not None
    assert durable[1]["model_fingerprint"] != "WRONG"
    for layer in fitted.source_layers:
        assert not torch.equal(durable[0].jacobians[layer], stale.jacobians[layer])


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
    # The resume base round-trips losslessly through the fp32 artifact. Allow
    # only ordinary accumulation-order noise against the from-scratch fit.
    for layer in full.source_layers:
        assert torch.allclose(
            resumed.jacobians[layer], full.jacobians[layer], atol=1e-6,
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


def test_loaded_model_fingerprint_ignores_registration_order() -> None:
    class _SameState(torch.nn.Module):
        def __init__(self, buffer_names: tuple[str, ...]) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.0]))
            for name in buffer_names:
                self.register_buffer(name, torch.tensor([ord(name[0])]))
            self.config = SimpleNamespace(model_type="toy", _name_or_path="toy")

    forward = _SameState(("alpha", "zeta"))
    reversed_order = _SameState(("zeta", "alpha"))

    assert [name for name, _ in forward.named_buffers()] != [
        name for name, _ in reversed_order.named_buffers()
    ]
    assert loaded_model_fingerprint(forward, "toy") == loaded_model_fingerprint(
        reversed_order, "toy",
    )


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


def test_jlens_property_rechecks_loaded_pointer_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import saklas.io.lens as lens_io

    session = _StubSession()
    live_fp = loaded_model_fingerprint(session._model, _MODEL_ID)
    lens = JacobianLens({0: torch.eye(6)}, n_prompts=1, d_model=6)
    compatible = {
        "model_fingerprint": live_fp, "source_layers": [0],
        "n_prompts": 1, "tensor_sha256": "a" * 64,
    }
    replaced = {**compatible, "model_fingerprint": "different-weights"}
    monkeypatch.setattr(lens_io, "load_lens_sidecar", lambda _model: compatible)
    monkeypatch.setattr(lens_io, "load_lens", lambda _model: (lens, replaced))

    assert session.jlens is None


def test_has_compatible_jlens_rechecks_loaded_pointer_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import saklas.io.lens as lens_io

    session = _StubSession()
    live_fp = loaded_model_fingerprint(session._model, _MODEL_ID)
    lens = JacobianLens({0: torch.eye(6)}, n_prompts=1, d_model=6)
    compatible = {
        "model_fingerprint": live_fp, "source_layers": [0],
        "n_prompts": 1, "tensor_sha256": "a" * 64,
    }
    replaced = {**compatible, "model_fingerprint": "different-weights"}
    session._jlens = lens
    session._jlens_identity = ("old",)
    monkeypatch.setattr(lens_io, "load_lens_sidecar", lambda _model: compatible)
    monkeypatch.setattr(lens_io, "load_lens", lambda _model: (lens, replaced))

    assert not session.has_compatible_jlens()


def test_fit_jlens_extends_real_prefix_checkpoint_without_full_artifact() -> None:
    full = _StubSession().fit_jlens(_PROMPTS, force=True)
    head_session = _StubSession()
    head = head_session.fit_jlens(_PROMPTS[:2], force=True)
    for path in lens_paths(_MODEL_ID):
        path.unlink()
    consumed_prefix = [
        [int(tok) for tok in head_session._tokenizer(
            p, return_tensors="pt",
        )["input_ids"][0].tolist()]
        for p in _PROMPTS[:2]
    ]
    prefix_sha = hashlib.sha256(
        repr(consumed_prefix).encode("utf-8")
    ).hexdigest()
    _save_checkpoint(
        head, _MODEL_ID,
        base_n_prompts=0,
        corpus_spec="test",
        # This is the real identity an interrupted two-prompt request owns,
        # not the future four-prompt corpus hash.
        corpus_sha256=prefix_sha,
        corpus_hash_kind="token_ids_v1",
        seq_len=128,
        dim_batch=8,
        skip_first=16,
        model_fingerprint=loaded_model_fingerprint(
            head_session._model, _MODEL_ID,
        ),
        consumed_prefix_sha256=prefix_sha,
    )

    messages: list[str] = []
    resumed = head_session.fit_jlens(_PROMPTS, on_progress=messages.append)

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
    import saklas.io.lens as lens_io

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
    _save_checkpoint(
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
    monkeypatch.setattr(
        lens_io, "load_lens",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError(
                "farther self-contained checkpoint should win before "
                "durable payload materialization"
            )
        ),
    )
    messages: list[str] = []
    resumed = _StubSession().fit_jlens(_PROMPTS, on_progress=messages.append)
    assert any("checkpoint at 3 prompts" in message for message in messages)
    for layer in full.source_layers:
        assert torch.allclose(
            resumed.jacobians[layer], full.jacobians[layer], atol=2e-3,
        )


def test_corrupt_farther_checkpoint_falls_back_to_durable_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import saklas.core.jlens as jlens_mod

    session = _StubSession()
    durable = session.fit_jlens(_PROMPTS[:2], force=True)
    consumed = [
        [int(tok) for tok in session._tokenizer(
            prompt, return_tensors="pt",
        )["input_ids"][0].tolist()]
        for prompt in _PROMPTS
    ]
    corpus_sha = hashlib.sha256(repr(consumed).encode("utf-8")).hexdigest()
    farther = JacobianLens(
        {layer: tensor.clone() for layer, tensor in durable.jacobians.items()},
        n_prompts=3, d_model=durable.d_model,
    )
    _save_checkpoint(
        farther, _MODEL_ID, base_n_prompts=0,
        corpus_spec="test", corpus_sha256=corpus_sha,
        corpus_hash_kind="token_ids_v1", seq_len=128, dim_batch=8,
        skip_first=16,
        model_fingerprint=loaded_model_fingerprint(
            session._model, _MODEL_ID,
        ),
    )
    checkpoint_tensor, _checkpoint_sidecar = lens_checkpoint_paths(_MODEL_ID)
    payload = bytearray(checkpoint_tensor.read_bytes())
    payload[-1] ^= 1
    checkpoint_tensor.write_bytes(payload)

    real_fit = jlens_mod.fit_jacobian_lens
    initial_counts: list[int | None] = []

    def capture_initial(*args: Any, **kwargs: Any) -> Any:
        initial = kwargs.get("initial_lens")
        initial_counts.append(initial.n_prompts if initial is not None else None)
        return real_fit(*args, **kwargs)

    monkeypatch.setattr(jlens_mod, "fit_jacobian_lens", capture_initial)
    result = session.fit_jlens(_PROMPTS)

    assert initial_counts == [2]
    assert result.n_prompts == len(_PROMPTS)


def test_matching_checkpoint_evicts_incompatible_resident_before_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gc
    import weakref

    import saklas.io.lens as lens_io

    session = _StubSession()
    session.fit_jlens(_PROMPTS[:2], force=True)
    resident = session._jlens
    assert resident is not None
    resident_ref = weakref.ref(resident)
    checkpoint = JacobianLens(
        {layer: tensor.clone() for layer, tensor in resident.jacobians.items()},
        n_prompts=2, d_model=resident.d_model,
    )
    del resident
    changed = ["a changed first prompt that is long enough", *_PROMPTS[1:]]
    consumed = [
        [int(tok) for tok in session._tokenizer(
            prompt, return_tensors="pt",
        )["input_ids"][0].tolist()]
        for prompt in changed
    ]
    _save_checkpoint(
        checkpoint, _MODEL_ID, base_n_prompts=0,
        corpus_spec="test",
        corpus_sha256=hashlib.sha256(repr(consumed).encode("utf-8")).hexdigest(),
        corpus_hash_kind="token_ids_v1", seq_len=128, dim_batch=8,
        skip_first=16,
        model_fingerprint=loaded_model_fingerprint(
            session._model, _MODEL_ID,
        ),
    )
    real_load = lens_io.load_lens_checkpoint
    resident_gone_at_load: list[bool] = []

    def observe_load(*args: Any, **kwargs: Any) -> Any:
        gc.collect()
        resident_gone_at_load.append(resident_ref() is None)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(lens_io, "load_lens_checkpoint", observe_load)
    result = session.fit_jlens(changed)

    assert resident_gone_at_load == [True]
    assert result.n_prompts == len(changed)


def test_resident_prefix_is_reloaded_after_resume_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import saklas.core.jlens as jlens_mod

    session = _StubSession()
    session.fit_jlens(_PROMPTS[:2], force=True)
    durable_pair = load_lens(_MODEL_ID)
    assert durable_pair is not None
    durable = durable_pair[0]
    expected = {
        layer: tensor.clone() for layer, tensor in durable.jacobians.items()
    }
    real_fit = jlens_mod.fit_jacobian_lens

    def fail_after_one(*args: Any, **kwargs: Any) -> Any:
        prompts = list(args[2])
        args = (*args[:2], prompts[:1], *args[3:])
        kwargs["input_id_rows"] = kwargs["input_id_rows"][:1]
        real_fit(*args, **kwargs)
        raise RuntimeError("injected resident resume failure")

    monkeypatch.setattr(jlens_mod, "fit_jacobian_lens", fail_after_one)
    with pytest.raises(RuntimeError, match="resident resume"):
        session.fit_jlens(_PROMPTS)

    assert session._jlens is not None
    assert session._jlens.n_prompts == 2
    for layer, tensor in expected.items():
        assert torch.allclose(session._jlens.jacobians[layer], tensor)


def test_subset_resume_releases_unrequested_resident_matrices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gc
    import weakref

    import saklas.core.jlens as jlens_mod

    session = _StubSession()
    session.fit_jlens(_PROMPTS[:2], force=True, source_layers=[0, 1])
    assert session._jlens is not None
    session._jlens_transport_stack(session._jlens, [0, 1], torch.device("cpu"))
    assert session._jlens_device_cache
    old_live_stack = torch.zeros(64)
    live_stack_ref = weakref.ref(old_live_stack)
    session._live_lens = {
        "layers": [0], "top_k": 3, "J_stack": old_live_stack,
    }
    del old_live_stack
    resident_ref = weakref.ref(session._jlens)
    real_fit = jlens_mod.fit_jacobian_lens
    released: list[bool] = []

    def observe_release(*args: Any, **kwargs: Any) -> Any:
        gc.collect()
        released.append(
            resident_ref() is None
            and live_stack_ref() is None
            and not session._jlens_device_cache
        )
        return real_fit(*args, **kwargs)

    monkeypatch.setattr(jlens_mod, "fit_jacobian_lens", observe_release)
    resumed = session.fit_jlens(_PROMPTS, source_layers=[0], force=True)

    assert released == [True]
    assert resumed.source_layers == [0]


def test_finalization_failure_rebuilds_evicted_resident(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import saklas.io.lens as lens_io

    session = _StubSession()
    session.fit_jlens(_PROMPTS[:2], force=True)
    durable_pair = load_lens(_MODEL_ID)
    assert durable_pair is not None
    expected = durable_pair[0]
    assert session._jlens is not None
    session._jlens_transport_stack(
        session._jlens, session._jlens.source_layers, torch.device("cpu"),
    )

    def fail_promotion(*_args: Any, **_kwargs: Any) -> bool:
        raise OSError("injected promotion failure")

    monkeypatch.setattr(lens_io, "promote_lens_checkpoint", fail_promotion)
    with pytest.raises(OSError, match="promotion failure"):
        session.fit_jlens(_PROMPTS, checkpoint_every=1)

    assert session._jlens is not None
    assert session._jlens.n_prompts == expected.n_prompts
    assert session._jlens_device_cache == {}
    for layer in expected.source_layers:
        assert torch.allclose(
            session._jlens.jacobians[layer], expected.jacobians[layer],
        )


def test_fit_jlens_missing_layer_topup_resumes_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import saklas.core.jlens as jlens_mod
    import saklas.io.lens as lens_io
    from saklas.io import packs

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
    real_save = lens_io.save_lens
    reused: list[set[int]] = []
    hashes = 0
    real_hash = packs.hash_file

    def _count_hash(path: Path) -> str:
        nonlocal hashes
        hashes += 1
        return real_hash(path)

    monkeypatch.setattr(packs, "hash_file", _count_hash)

    def _capture_reuse(*args: Any, **kwargs: Any) -> Any:
        reused.append(set(kwargs.get("reuse_layers") or ()))
        return real_save(*args, **kwargs)

    monkeypatch.setattr(lens_io, "save_lens", _capture_reuse)
    messages: list[str] = []
    resumed = _StubSession().fit_jlens(
        _PROMPTS, source_layers=[0, 1], on_progress=messages.append,
    )
    assert any("missing-layer checkpoint at 2" in message for message in messages)
    assert reused == [{0}]
    assert hashes == 0
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
    import saklas.core.capture as _vectors

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
    # Both fitted source layers contribute. Softmax assigns positive mass to
    # every vocabulary item at each layer, so every aggregate row has a
    # non-degenerate depth distribution over L0 and L1.
    for _, _, com, spread in rows:
        assert 0.0 < com < 1 / (3 - 1)
        assert spread > 0.0


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
    assert seen_layers == [[0, 1]]
    dirs = s._profiles[name]
    assert set(dirs) == {0, 1}
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

    def latest_per_layer(self) -> dict[int, torch.Tensor]:
        return {layer: rows[-1] for layer, rows in self._buckets.items()}


def test_enable_live_lens_defaults_and_disable() -> None:
    s = _StubSession()
    s.fit_jlens(_PROMPTS)
    layers = s.enable_live_lens()
    assert layers == [0, 1]
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
    s.enable_live_lens(layers=[0, 1])
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

    step = s._live_lens_readout_step(top_k=3)
    assert step is not None
    out, agg, token_ids = step
    assert set(out) == {0, 1}
    assert set(token_ids) == {0, 1}
    for row in out.values():
        assert len(row) == 3
        assert all(isinstance(tok, str) for tok, _ in row)
    assert all(len(row) == 3 for row in token_ids.values())
    assert all(isinstance(token_id, int) for row in token_ids.values() for token_id in row)
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


def test_live_lens_readout_step_avoids_float64_for_mps_compatibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_to = torch.Tensor.to

    def reject_float64(
        self: torch.Tensor, *args: Any, **kwargs: Any,
    ) -> torch.Tensor:
        dtype = kwargs.get("dtype")
        if args and isinstance(args[0], torch.dtype):
            dtype = args[0]
        if dtype is torch.float64:
            raise AssertionError("device-side float64 is unsupported on MPS")
        return real_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", reject_float64)
    s = _StubSession()
    s.fit_jlens(_PROMPTS)
    s.enable_live_lens(layers=[0, 1])
    s._capture = _FakeCapture({0: torch.randn(6), 1: torch.randn(6)})

    step = s._live_lens_readout_step(top_k=3)
    assert step is not None


def test_jlens_row_selector_avoids_copy_for_identity_and_contiguous_rows() -> None:
    tensor = torch.arange(24, dtype=torch.float32).reshape(4, 6)

    identity = SaklasSession._select_tensor_rows(tensor, [0, 1, 2, 3])
    contiguous = SaklasSession._select_tensor_rows(tensor, [1, 2])
    gathered = SaklasSession._select_tensor_rows(tensor, [0, 2])

    assert identity is tensor
    assert contiguous.tolist() == tensor[1:3].tolist()
    assert (
        contiguous.untyped_storage().data_ptr()
        == tensor.untyped_storage().data_ptr()
    )
    assert gathered.tolist() == tensor[[0, 2]].tolist()
    assert (
        gathered.untyped_storage().data_ptr()
        != tensor.untyped_storage().data_ptr()
    )


def test_live_lens_step_normalizes_once_across_all_consumers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import saklas.core.jlens as jlens_module

    s = _StubSession()
    s.fit_jlens(_PROMPTS)
    s.enable_live_lens(layers=[1])
    s._add_lens_probe("jlens/g", as_name=None)
    s._capture = _FakeCapture({1: torch.randn(6)})
    calls = 0
    stat_calls = 0
    original = jlens_module.readout_probabilities
    original_stats = jlens_module.token_readout_stats_from_probabilities

    def counting_probabilities(logits: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return original(logits)

    def counting_stats(*args: Any, **kwargs: Any) -> Any:
        nonlocal stat_calls
        stat_calls += 1
        return original_stats(*args, **kwargs)

    monkeypatch.setattr(
        jlens_module, "readout_probabilities", counting_probabilities,
    )
    monkeypatch.setattr(
        jlens_module,
        "token_readout_stats_from_probabilities",
        counting_stats,
    )

    # A gated pinned probe calibrates before the token tap and stashes the
    # matrix. Cards, pinned readings, and aggregate must all reuse it.
    scalars = s._score_lens_gate_scalars()
    assert scalars
    assert s._live_lens_readout_step(top_k=3) is not None
    assert calls == 1
    assert stat_calls == 1
    assert s._last_lens_step_readings is not None
    assert s._last_lens_step_readings["jlens/g"].coords[0] == pytest.approx(
        scalars["jlens/g"],
    )


def test_live_lens_readout_reuses_depth_and_token_selector_tensors() -> None:
    s = _StubSession()
    s.fit_jlens(_PROMPTS)
    s.enable_live_lens(layers=[0, 1])
    s._add_lens_probe("jlens/g", as_name=None)
    s._capture = _FakeCapture({
        0: torch.randn(6, generator=torch.Generator().manual_seed(20)),
        1: torch.randn(6, generator=torch.Generator().manual_seed(21)),
    })

    assert s._score_lens_gate_scalars({"jlens/g"})
    first_depth_ids = {
        key: id(value) for key, value in s._jlens_depth_tensor_cache.items()
    }
    first_selector_ids = {
        key: id(value) for key, value in s._readout_long_tensor_cache.items()
    }
    assert first_depth_ids
    assert first_selector_ids

    assert s._score_lens_gate_scalars({"jlens/g"})

    assert {
        key: id(value) for key, value in s._jlens_depth_tensor_cache.items()
    } == first_depth_ids
    assert {
        key: id(value) for key, value in s._readout_long_tensor_cache.items()
    } == first_selector_ids


def test_live_lens_exact_stash_reuse_skips_hidden_cast() -> None:
    class BombHidden:
        def to(self, *_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("exact stash reuse should not cast hidden rows")

    s = _StubSession()
    s.fit_jlens(_PROMPTS)
    s.enable_live_lens(layers=[1])
    s._capture = _FakeCapture({1: cast(Any, BombHidden())})
    assert s._live_lens is not None
    vocab = int(s._model.lm_head.weight.shape[0])
    logits = torch.randn(1, vocab, generator=torch.Generator().manual_seed(7))
    import saklas.core.jlens as jlens_module

    probabilities = jlens_module.readout_probabilities(logits)
    s._lens_step_stash = {
        "fresh": True,
        "layers": (1,),
        "logits": logits,
        "probabilities": probabilities,
    }

    step = s._live_lens_readout_step(top_k=3)

    assert step is not None
    assert s._lens_step_stash["fresh"] is False


def test_live_lens_reuses_gated_subset_rows_for_wider_display() -> None:
    s = _StubSession()
    s.fit_jlens(_PROMPTS)
    s.enable_live_lens(layers=[0, 1])
    s._add_lens_probe("jlens/g", as_name=None)
    s._lens_probes["jlens/g"]["layers"] = [1]
    s._capture = _FakeCapture({
        0: torch.randn(6, generator=torch.Generator().manual_seed(0)),
        1: torch.randn(6, generator=torch.Generator().manual_seed(1)),
    })

    assert s._score_lens_gate_scalars({"jlens/g"})
    assert s._lens_step_stash is not None
    assert s._lens_step_stash["layers"] == (1,)
    # Poison the live transport row for the gated layer after the gate callback.
    # The display still covers both live layers, but layer 1 should ride the
    # cached gate logits/probabilities instead of recomputing from this row.
    assert s._live_lens is not None
    row = s._live_lens["layer_rows"][1]
    s._live_lens["J_stack"][row].fill_(float("nan"))

    step = s._live_lens_readout_step(top_k=3)

    assert step is not None
    per_layer, aggregate, token_ids = step
    assert set(per_layer) == {0, 1}
    assert set(token_ids) == {0, 1}
    assert all(torch.isfinite(torch.tensor(score)) for _tok, score in per_layer[1])
    assert all(torch.isfinite(torch.tensor(row[1])) for row in aggregate)


def test_live_lens_readout_step_none_when_off() -> None:
    s = _StubSession()
    assert s._live_lens_readout_step() is None


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
        gen_seat: str = "assistant",
    ) -> torch.Tensor:
        self.prepare_calls.append({
            "input": input, "raw": raw, "thinking": thinking,
            "parent_node_id": parent_node_id,
            "user_role": user_role, "assistant_role": assistant_role,
            "gen_seat": gen_seat,
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
    import saklas.core.capture as _vectors

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
    assert set(out["readout"]) == {0, 1}  # fitted sources of the 3-layer toy
    for rows in out["readout"].values():
        assert len(rows) == 4
        tok, lp, tid = rows[0]
        assert isinstance(tok, str) and lp <= 0.0 and isinstance(tid, int)
    # The aggregate block rides the same logits across both fitted layers.
    assert len(out["aggregate"]) == 4
    for tok, strength, com, spread in out["aggregate"]:
        assert isinstance(tok, str)
        assert 0.0 <= strength <= 1.0
        assert 0.0 < com < 1 / (3 - 1)
        assert spread > 0.0
    # Continue-mode rebuild (cast model): no text resend — the history walk
    # to the node's parent carries the user turn; the gen header opens the
    # node's own seat.
    assert s.prepare_calls[0]["input"] is None
    assert s.prepare_calls[0]["raw"] is False
    assert s.prepare_calls[0]["gen_seat"] == "assistant"
    node = s.tree.get(node_id)
    assert s.prepare_calls[0]["parent_node_id"] == node.parent_id


def test_jlens_token_readout_index_zero_reads_prompt_only() -> None:
    s = _TreeStubSession()
    s.fit_jlens(_PROMPTS)
    node_id = _tree_with_assistant(s, s._tokenizer.encode("abc"))

    seen_lens: list[tuple[int, int | None]] = []
    import saklas.core.capture as _vectors

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
    # A committed user turn is a valid *shape* under the cast model (user-
    # seat generated nodes are forkable/readable) — but this one has no
    # decode record, so it fails on that instead of its role.
    with pytest.raises(InvalidNodeOperationError, match="no raw token record"):
        s.jlens_token_readout(user_id, 0)
    # The system root stays out of bounds — not a turn at all.
    assert s.tree.root_id is not None
    with pytest.raises(InvalidNodeOperationError, match="only a turn"):
        s.jlens_token_readout(s.tree.root_id, 0)
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
