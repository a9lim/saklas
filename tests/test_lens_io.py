"""CPU round-trip tests for the per-model Jacobian-lens artifact."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from saklas.core.jlens import JacobianLens
from saklas.io.lens import (
    LENS_FORMAT_VERSION,
    lens_paths,
    load_lens,
    load_lens_sidecar,
    remove_lens,
    save_lens,
)

_MODEL = "test-org/tiny-model"
_D = 8


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))


def _lens(n_layers: int = 3, n_prompts: int = 7) -> JacobianLens:
    gen = torch.Generator().manual_seed(0)
    return JacobianLens(
        {l: torch.randn(_D, _D, generator=gen) for l in range(n_layers)},
        n_prompts=n_prompts,
        d_model=_D,
    )


def _save(lens: JacobianLens) -> None:
    save_lens(
        lens, _MODEL,
        corpus_spec="test-corpus", corpus_sha256="abc123",
        seq_len=128, dim_batch=8, skip_first=16,
    )


def test_paths_layout() -> None:
    ts, sc = lens_paths(_MODEL)
    assert ts.suffix == ".safetensors" and sc.suffix == ".json"
    assert "test-org__tiny-model" in str(ts)


def test_save_load_round_trip() -> None:
    lens = _lens()
    _save(lens)

    loaded = load_lens(_MODEL)
    assert loaded is not None
    got, sidecar = loaded
    assert got.n_prompts == 7
    assert got.d_model == _D
    assert got.source_layers == [0, 1, 2]
    # fp16 storage: round-trip within half-precision tolerance, promoted fp32
    for layer in got.source_layers:
        assert got.jacobians[layer].dtype == torch.float32
        assert torch.allclose(got.jacobians[layer], lens.jacobians[layer], atol=2e-3)
    assert sidecar["format_version"] == LENS_FORMAT_VERSION
    assert sidecar["corpus_spec"] == "test-corpus"
    assert sidecar["corpus_sha256"] == "abc123"
    assert sidecar["corpus_hash_kind"] == "text_v1"
    assert sidecar["skip_first_positions"] == 16


def test_load_lens_sidecar_does_not_read_tensors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _save(_lens())

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("metadata-only load should not read safetensors")

    monkeypatch.setattr("saklas.io.lens.load_file", _boom)
    sidecar = load_lens_sidecar(_MODEL)
    assert sidecar is not None
    assert sidecar["source_layers"] == [0, 1, 2]
    assert sidecar["d_model"] == _D


def test_save_lens_preserves_existing_tensor_on_failed_replace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _lens(n_prompts=3)
    _save(first)
    ts_path, _ = lens_paths(_MODEL)
    before = ts_path.read_bytes()

    def _fail_save(_tensors: object, path: str) -> None:
        Path(path).write_bytes(b"partial")
        raise RuntimeError("simulated save failure")

    monkeypatch.setattr("saklas.io.lens.save_file", _fail_save)
    with pytest.raises(RuntimeError, match="simulated"):
        _save(_lens(n_prompts=9))
    assert ts_path.read_bytes() == before
    assert not ts_path.with_suffix(ts_path.suffix + ".tmp").exists()

    loaded = load_lens(_MODEL)
    assert loaded is not None
    assert loaded[0].n_prompts == 3


def test_load_missing_returns_none() -> None:
    assert load_lens(_MODEL) is None
    assert load_lens_sidecar(_MODEL) is None


def test_load_wrong_format_version_returns_none() -> None:
    _save(_lens())
    _, sc_path = lens_paths(_MODEL)
    sidecar = json.loads(sc_path.read_text())
    sidecar["format_version"] = LENS_FORMAT_VERSION + 1
    sc_path.write_text(json.dumps(sidecar))
    assert load_lens(_MODEL) is None


def test_load_non_finite_returns_none() -> None:
    lens = _lens()
    lens.jacobians[1][0, 0] = float("inf")
    _save(lens)
    assert load_lens(_MODEL) is None


def test_load_wrong_tensor_shape_returns_none() -> None:
    _save(_lens())
    ts_path, _sc_path = lens_paths(_MODEL)
    save_file({"layer_0": torch.randn(_D, _D + 1)}, str(ts_path))
    assert load_lens(_MODEL) is None


def test_load_sidecar_layer_mismatch_returns_none() -> None:
    _save(_lens())
    _ts_path, sc_path = lens_paths(_MODEL)
    sidecar = json.loads(sc_path.read_text())
    sidecar["source_layers"] = [0, 1, 9]
    sc_path.write_text(json.dumps(sidecar))
    assert load_lens(_MODEL) is None


def test_load_corrupt_sidecar_returns_none() -> None:
    _save(_lens())
    _, sc_path = lens_paths(_MODEL)
    sc_path.write_text("{not json")
    assert load_lens(_MODEL) is None


def test_remove_lens() -> None:
    _save(_lens())
    assert remove_lens(_MODEL) is True
    assert load_lens(_MODEL) is None
    assert remove_lens(_MODEL) is False
