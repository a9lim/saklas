"""CPU round-trip tests for the per-model Jacobian-lens artifact."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from saklas.core.jlens import JacobianLens
from saklas.io.lens import (
    LENS_FORMAT_VERSION,
    lens_artifact_size,
    lens_checkpoint_paths,
    lens_fit_lock,
    lens_paths,
    load_lens,
    load_lens_checkpoint,
    load_lens_sidecar,
    remove_lens,
    remove_subsumed_lens_checkpoint,
    promote_lens_checkpoint,
    save_lens,
    save_lens_checkpoint,
    save_lens_checkpoint_accumulator,
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
    assert set(sidecar["tensor_files"]) == {"0", "1", "2"}
    assert set(sidecar["tensor_sha256"]) == {"0", "1", "2"}
    assert all(
        ".gen-" in filename and f"layer-{layer}" in filename
        for layer, filename in sidecar["tensor_files"].items()
    )
    assert lens_paths(_MODEL)[0].name == sidecar["tensor_files"]["0"]


def test_load_migrates_legacy_fixed_tensor_path() -> None:
    import saklas.io.lens as lens_io

    lens = _lens()
    _save(lens)
    generation, sc_path = lens_paths(_MODEL)
    legacy = generation.parent / "jlens.safetensors"
    layers = lens.source_layers

    def _rows(layer: int, start: int, end: int) -> torch.Tensor:
        return lens.jacobians[layer][start:end].to(torch.float16).contiguous()

    digest = lens_io._save_fp16_square_safetensors_atomic(
        legacy, layers, _D, _rows,
    )
    sidecar = json.loads(sc_path.read_text())
    sidecar["format_version"] = 3
    sidecar.pop("tensor_files")
    sidecar["tensor_sha256"] = digest
    sc_path.write_text(json.dumps(sidecar))

    loaded = load_lens(_MODEL)

    assert loaded is not None
    assert lens_paths(_MODEL)[0] == legacy


def test_missing_layer_topup_reuses_immutable_existing_shards() -> None:
    initial = _lens(n_layers=2)
    _save(initial)
    _, sc_path = lens_paths(_MODEL)
    before = json.loads(sc_path.read_text())
    before_stats = {
        layer: (sc_path.parent / filename).stat()
        for layer, filename in before["tensor_files"].items()
    }
    extra = _lens(n_layers=3).jacobians[2]
    merged = JacobianLens(
        {**initial.jacobians, 2: extra},
        n_prompts=initial.n_prompts, d_model=_D,
    )

    save_lens(
        merged, _MODEL,
        corpus_spec="test-corpus", corpus_sha256="abc123",
        seq_len=128, dim_batch=8, skip_first=16,
        reuse_layers={0, 1},
    )

    after = json.loads(sc_path.read_text())
    assert after["tensor_files"]["0"] == before["tensor_files"]["0"]
    assert after["tensor_files"]["1"] == before["tensor_files"]["1"]
    assert after["tensor_files"]["2"] not in set(before["tensor_files"].values())
    for layer in ("0", "1"):
        stat = (sc_path.parent / after["tensor_files"][layer]).stat()
        old = before_stats[layer]
        assert (stat.st_ino, stat.st_mtime_ns, stat.st_size) == (
            old.st_ino, old.st_mtime_ns, old.st_size,
        )
    loaded = load_lens(_MODEL)
    assert loaded is not None and loaded[0].source_layers == [0, 1, 2]


def test_shard_reuse_is_refused_when_fit_identity_changes() -> None:
    lens = _lens(n_layers=2)
    _save(lens)
    _, sc_path = lens_paths(_MODEL)
    before = json.loads(sc_path.read_text())["tensor_files"]

    save_lens(
        lens, _MODEL,
        corpus_spec="other-corpus", corpus_sha256="different",
        seq_len=128, dim_batch=8, skip_first=16,
        reuse_layers={0, 1},
    )

    after = json.loads(sc_path.read_text())["tensor_files"]
    assert set(after.values()).isdisjoint(before.values())


def test_artifact_size_refreshes_a_stale_sidecar_snapshot() -> None:
    _save(_lens(n_layers=2))
    stale = load_lens_sidecar(_MODEL)
    assert stale is not None
    _save(_lens(n_layers=3, n_prompts=9))
    current = load_lens_sidecar(_MODEL)
    assert current is not None
    expected = sum(
        (lens_paths(_MODEL)[1].parent / filename).stat().st_size
        for filename in current["tensor_files"].values()
    )

    assert lens_artifact_size(_MODEL, stale) == expected


def test_corrupt_reusable_shard_is_rewritten_while_valid_shard_is_kept() -> None:
    initial = _lens(n_layers=2)
    _save(initial)
    _, sc_path = lens_paths(_MODEL)
    before = json.loads(sc_path.read_text())["tensor_files"]
    corrupt = sc_path.parent / before["0"]
    payload = bytearray(corrupt.read_bytes())
    payload[-1] ^= 1
    corrupt.write_bytes(payload)
    merged = JacobianLens(
        {**initial.jacobians, 2: torch.eye(_D)},
        n_prompts=initial.n_prompts, d_model=_D,
    )

    save_lens(
        merged, _MODEL,
        corpus_spec="test-corpus", corpus_sha256="abc123",
        seq_len=128, dim_batch=8, skip_first=16,
        reuse_layers={0, 1},
    )

    after = json.loads(sc_path.read_text())["tensor_files"]
    assert after["0"] != before["0"]
    assert after["1"] == before["1"]
    loaded = load_lens(_MODEL)
    assert loaded is not None and loaded[0].source_layers == [0, 1, 2]


def test_save_load_checkpoint_round_trip() -> None:
    partial = _lens(n_layers=2, n_prompts=5)
    save_lens_checkpoint(
        partial, _MODEL,
        base_n_prompts=7,
        corpus_spec="test-corpus",
        corpus_sha256="abc123",
        seq_len=128,
        dim_batch=8,
        skip_first=16,
        raw_corpus_sha256="raw456",
        raw_prompt_count=13,
        usable_prompt_count=12,
    )

    loaded = load_lens_checkpoint(_MODEL)
    assert loaded is not None
    got, sidecar = loaded
    assert got.n_prompts == 5
    assert got.source_layers == [0, 1]
    assert sidecar["checkpoint"] is True
    assert sidecar["base_n_prompts"] == 7
    assert sidecar["partial_n_prompts"] == 5
    assert sidecar["raw_corpus_sha256"] == "raw456"
    assert sidecar["raw_prompt_count"] == 13
    assert sidecar["usable_prompt_count"] == 12


def test_checkpoint_accumulator_is_self_contained_and_merges_prefix() -> None:
    base = _lens(n_layers=2, n_prompts=3)
    tail = _lens(n_layers=2, n_prompts=2)
    sums = {layer: J * tail.n_prompts for layer, J in tail.jacobians.items()}

    save_lens_checkpoint_accumulator(
        sums, tail.n_prompts, _D, _MODEL,
        base=base,
        corpus_spec="test-corpus",
        corpus_sha256="abc123",
        seq_len=128,
        dim_batch=8,
        skip_first=16,
        model_layer_count=3,
    )

    loaded = load_lens_checkpoint(_MODEL)
    assert loaded is not None
    got, sidecar = loaded
    expected = JacobianLens.merge([base, tail])
    assert got.n_prompts == expected.n_prompts == 5
    assert sidecar["base_n_prompts"] == 0
    assert sidecar["partial_n_prompts"] == 5
    assert sidecar["model_layer_count"] == 3
    for layer in got.source_layers:
        assert torch.allclose(
            got.jacobians[layer], expected.jacobians[layer], atol=2e-3,
        )


def test_load_lens_sidecar_validates_tensor_header_without_materializing() -> None:
    _save(_lens())
    sidecar = load_lens_sidecar(_MODEL)
    assert sidecar is not None
    assert sidecar["source_layers"] == [0, 1, 2]
    assert sidecar["d_model"] == _D

    ts_path, _ = lens_paths(_MODEL)
    ts_path.write_bytes(ts_path.read_bytes()[:32])
    assert load_lens_sidecar(_MODEL) is None


def test_save_lens_preserves_existing_tensor_on_failed_replace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _lens(n_prompts=3)
    _save(first)
    ts_path, _ = lens_paths(_MODEL)
    before = ts_path.read_bytes()

    def _fail_save(_fd: int) -> None:
        raise RuntimeError("simulated save failure")

    monkeypatch.setattr("saklas.io.lens.os.fsync", _fail_save)
    with pytest.raises(RuntimeError, match="simulated"):
        _save(_lens(n_prompts=9))
    assert ts_path.read_bytes() == before
    assert not ts_path.with_suffix(ts_path.suffix + ".tmp").exists()

    loaded = load_lens(_MODEL)
    assert loaded is not None
    assert loaded[0].n_prompts == 3


def test_checkpoint_promotion_sidecar_failure_preserves_both_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import saklas.io.lens as lens_io

    final = _lens(n_layers=2, n_prompts=3)
    save_lens(
        final, _MODEL, corpus_spec="old", corpus_sha256="old-sha",
        seq_len=128, dim_batch=8, skip_first=16,
        model_fingerprint="weights",
    )
    checkpoint = _lens(n_layers=2, n_prompts=5)
    save_lens_checkpoint(
        checkpoint, _MODEL, base_n_prompts=0,
        corpus_spec="new", corpus_sha256="new-sha",
        corpus_hash_kind="token_ids_v1", seq_len=128, dim_batch=8,
        skip_first=16, model_fingerprint="weights",
    )
    _final_tensor, final_sc = lens_paths(_MODEL)
    real_write = lens_io.write_json_atomic

    def _fail_final_pointer(path: Path, payload: object) -> None:
        if path == final_sc:
            raise OSError("simulated final pointer failure")
        real_write(path, payload)

    monkeypatch.setattr(lens_io, "write_json_atomic", _fail_final_pointer)
    with pytest.raises(OSError, match="pointer failure"):
        promote_lens_checkpoint(
            _MODEL, n_prompts=5, source_layers=[0, 1],
            corpus_sha256="new-sha", corpus_hash_kind="token_ids_v1",
            seq_len=128, d_model=_D, model_fingerprint="weights",
        )

    still_final = load_lens(_MODEL)
    still_checkpoint = load_lens_checkpoint(_MODEL)
    assert still_final is not None and still_final[0].n_prompts == 3
    assert still_checkpoint is not None and still_checkpoint[0].n_prompts == 5


def test_checkpoint_promotion_rejects_corrupt_shard_and_keeps_pointers() -> None:
    save_lens(
        _lens(n_layers=2, n_prompts=3), _MODEL,
        corpus_spec="old", corpus_sha256="old-sha",
        seq_len=128, dim_batch=8, skip_first=16,
        model_fingerprint="weights",
    )
    save_lens_checkpoint(
        _lens(n_layers=2, n_prompts=5), _MODEL, base_n_prompts=0,
        corpus_spec="new", corpus_sha256="new-sha",
        corpus_hash_kind="token_ids_v1", seq_len=128, dim_batch=8,
        skip_first=16, model_fingerprint="weights",
    )
    checkpoint_ts, checkpoint_sc = lens_checkpoint_paths(_MODEL)
    payload = bytearray(checkpoint_ts.read_bytes())
    payload[-1] ^= 1
    checkpoint_ts.write_bytes(payload)

    assert not promote_lens_checkpoint(
        _MODEL, n_prompts=5, source_layers=[0, 1],
        corpus_sha256="new-sha", corpus_hash_kind="token_ids_v1",
        seq_len=128, d_model=_D, model_fingerprint="weights",
    )
    final = load_lens(_MODEL)
    assert final is not None and final[0].n_prompts == 3
    assert checkpoint_sc.exists() and checkpoint_ts.exists()


def test_checkpoint_promotion_rejects_legacy_v3_without_destroying_it() -> None:
    import saklas.io.lens as lens_io

    checkpoint = _lens(n_layers=2, n_prompts=5)
    save_lens_checkpoint(
        checkpoint, _MODEL, base_n_prompts=0,
        corpus_spec="new", corpus_sha256="new-sha",
        corpus_hash_kind="token_ids_v1", seq_len=128, dim_batch=8,
        skip_first=16, model_fingerprint="weights",
    )
    generation, checkpoint_sc = lens_checkpoint_paths(_MODEL)
    legacy = generation.parent / "jlens.partial.safetensors"

    def _rows(layer: int, start: int, end: int) -> torch.Tensor:
        return checkpoint.jacobians[layer][start:end].to(
            torch.float16,
        ).contiguous()

    digest = lens_io._save_fp16_square_safetensors_atomic(
        legacy, checkpoint.source_layers, _D, _rows,
    )
    sidecar = json.loads(checkpoint_sc.read_text())
    sidecar["format_version"] = 3
    sidecar.pop("tensor_files")
    sidecar["tensor_sha256"] = digest
    checkpoint_sc.write_text(json.dumps(sidecar))

    assert not promote_lens_checkpoint(
        _MODEL, n_prompts=5, source_layers=[0, 1],
        corpus_sha256="new-sha", corpus_hash_kind="token_ids_v1",
        seq_len=128, d_model=_D, model_fingerprint="weights",
    )
    loaded = load_lens_checkpoint(_MODEL)
    assert loaded is not None and loaded[0].n_prompts == 5
    assert legacy.exists() and checkpoint_sc.exists()


def test_subsumed_checkpoint_recovery_handles_base_progress_and_subset_layers() -> None:
    final = _lens(n_layers=3, n_prompts=7)
    save_lens(
        final, _MODEL, corpus_spec="same", corpus_sha256="same-sha",
        corpus_hash_kind="token_ids_v1", seq_len=128, dim_batch=8,
        skip_first=16, model_fingerprint="weights",
    )
    # Legacy-compatible progress semantics: 3 durable base prompts plus a
    # 2-prompt partial means this checkpoint reaches prompt 5, not prompt 2.
    checkpoint = JacobianLens(
        {1: torch.eye(_D)}, n_prompts=2, d_model=_D,
    )
    save_lens_checkpoint(
        checkpoint, _MODEL, base_n_prompts=3,
        corpus_spec="same", corpus_sha256="same-sha",
        corpus_hash_kind="token_ids_v1", seq_len=128, dim_batch=8,
        skip_first=16, model_fingerprint="weights",
    )

    assert remove_subsumed_lens_checkpoint(_MODEL)
    assert load_lens_checkpoint(_MODEL) is None
    loaded = load_lens(_MODEL)
    assert loaded is not None and loaded[0].source_layers == [0, 1, 2]


def test_subsumed_checkpoint_recovery_preserves_farther_progress() -> None:
    save_lens(
        _lens(n_layers=2, n_prompts=3), _MODEL,
        corpus_spec="same", corpus_sha256="same-sha",
        corpus_hash_kind="token_ids_v1", seq_len=128, dim_batch=8,
        skip_first=16, model_fingerprint="weights",
    )
    save_lens_checkpoint(
        _lens(n_layers=2, n_prompts=2), _MODEL, base_n_prompts=3,
        corpus_spec="same", corpus_sha256="same-sha",
        corpus_hash_kind="token_ids_v1", seq_len=128, dim_batch=8,
        skip_first=16, model_fingerprint="weights",
    )

    assert not remove_subsumed_lens_checkpoint(_MODEL)
    assert load_lens_checkpoint(_MODEL) is not None


def test_subsumed_checkpoint_recovery_preserves_other_corpus() -> None:
    save_lens(
        _lens(n_layers=2, n_prompts=7), _MODEL,
        corpus_spec="final", corpus_sha256="final-sha",
        corpus_hash_kind="token_ids_v1", seq_len=128, dim_batch=8,
        skip_first=16, model_fingerprint="weights",
    )
    save_lens_checkpoint(
        _lens(n_layers=1, n_prompts=2), _MODEL, base_n_prompts=0,
        corpus_spec="checkpoint", corpus_sha256="other-sha",
        corpus_hash_kind="token_ids_v1", seq_len=128, dim_batch=8,
        skip_first=16, model_fingerprint="weights",
    )

    assert not remove_subsumed_lens_checkpoint(_MODEL)
    assert load_lens_checkpoint(_MODEL) is not None


def test_subsumed_checkpoint_recovery_preserves_recovery_point_if_final_corrupt() -> None:
    save_lens(
        _lens(n_layers=2, n_prompts=7), _MODEL,
        corpus_spec="same", corpus_sha256="same-sha",
        corpus_hash_kind="token_ids_v1", seq_len=128, dim_batch=8,
        skip_first=16, model_fingerprint="weights",
    )
    save_lens_checkpoint(
        _lens(n_layers=1, n_prompts=2), _MODEL, base_n_prompts=0,
        corpus_spec="same", corpus_sha256="same-sha",
        corpus_hash_kind="token_ids_v1", seq_len=128, dim_batch=8,
        skip_first=16, model_fingerprint="weights",
    )
    final_tensor, _ = lens_paths(_MODEL)
    payload = bytearray(final_tensor.read_bytes())
    payload[-1] ^= 1
    final_tensor.write_bytes(payload)

    assert not remove_subsumed_lens_checkpoint(_MODEL)
    assert load_lens_checkpoint(_MODEL) is not None


def test_remove_waits_for_fit_transaction() -> None:
    _save(_lens())
    started = threading.Event()
    finished = threading.Event()

    def _remove() -> None:
        started.set()
        remove_lens(_MODEL)
        finished.set()

    with lens_fit_lock(_MODEL):
        worker = threading.Thread(target=_remove)
        worker.start()
        assert started.wait(timeout=1.0)
        assert not finished.wait(timeout=0.05)
        assert load_lens(_MODEL) is not None
    worker.join(timeout=1.0)
    assert finished.is_set()
    assert load_lens(_MODEL) is None


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


def test_load_changed_estimator_policy_returns_none() -> None:
    _save(_lens())
    _, sc_path = lens_paths(_MODEL)
    sidecar = json.loads(sc_path.read_text())
    sidecar["estimator_policy"]["skip_first_positions"] += 1
    sc_path.write_text(json.dumps(sidecar))
    assert load_lens_sidecar(_MODEL) is None
    assert load_lens(_MODEL) is None


def test_load_non_finite_returns_none() -> None:
    lens = _lens()
    lens.jacobians[1][0, 0] = float("inf")
    _save(lens)
    assert load_lens(_MODEL) is None


def test_load_rejects_same_shape_finite_payload_corruption() -> None:
    _save(_lens())
    ts_path, _ = lens_paths(_MODEL)
    payload = bytearray(ts_path.read_bytes())
    payload[-1] ^= 1
    ts_path.write_bytes(payload)
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
    save_lens_checkpoint(
        _lens(n_layers=1), _MODEL,
        base_n_prompts=0,
        corpus_spec="test-corpus",
        corpus_sha256="abc123",
        seq_len=128,
        dim_batch=8,
        skip_first=16,
    )
    ckpt_ts, ckpt_sc = lens_checkpoint_paths(_MODEL)
    assert ckpt_ts.exists() and ckpt_sc.exists()
    assert remove_lens(_MODEL) is True
    assert load_lens(_MODEL) is None
    assert load_lens_checkpoint(_MODEL) is None
    assert remove_lens(_MODEL) is False


def test_remove_unpublishes_before_best_effort_tensor_gc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _save(_lens())
    real_unlink = Path.unlink

    def _fail_tensor_unlink(path: Path, missing_ok: bool = False) -> None:
        if path.suffix == ".safetensors" and ".gen-" in path.name:
            raise OSError("simulated tensor GC failure")
        real_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", _fail_tensor_unlink)

    assert remove_lens(_MODEL)
    assert load_lens(_MODEL) is None
    assert load_lens_checkpoint(_MODEL) is None
