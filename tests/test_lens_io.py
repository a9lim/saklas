"""CPU round-trip tests for the per-model Jacobian-lens artifact."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from saklas.core.jlens import JacobianLens
from saklas.io.lens import (
    LENS_FORMAT_VERSION,
    lens_paths,
    load_lens,
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
    assert sidecar["skip_first_positions"] == 16


def test_load_missing_returns_none() -> None:
    assert load_lens(_MODEL) is None


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
