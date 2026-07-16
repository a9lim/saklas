"""Neutral-activation cache is stored fp32 (no precision seam).

Every saklas safetensor artifact is stored fp32 (cast at the writer); the
per-model neutral-activation cache was the last exception (bf16) and is now
fp32 too.  These tests pin three things: the on-disk store is fp32, the
compute path (cache miss) and the cache-hit path return bit-identical tensors
(the seam fix — the whitener covariance is bit-reproducible across the cache
boundary), and a stale bf16 cache is invalidated and recomputed to fp32.

CPU-only: ``compute_neutral_activations`` is monkeypatched to a deterministic
fp32 dict so no real model loads.  ``$SAKLAS_HOME`` is pointed at a tmp dir so
the real cache is never touched.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from safetensors.torch import load_file, save_file

from saklas.core.mahalanobis import LayerWhitener, WhitenerError
from saklas.io.alignment import (
    _neutral_acts_paths,
    load_validated_neutral_cache,
    load_or_compute_neutral_activations,
    load_or_compute_neutral_activations_with_metadata,
    validate_neutral_cache_metadata,
)
from saklas.io.paths import model_dir

MODEL_ID = "test-org/test-model"


class _Tokenizer:
    chat_template = None
    all_special_ids: list[int] = []
    added_tokens_encoder: dict[str, int] = {}
    bos_token_id = 1
    eos_token_id = 2

    def __call__(
        self, text: str, *, return_tensors: str, add_special_tokens: bool = False,
    ) -> dict[str, torch.Tensor]:
        del return_tensors, add_special_tokens
        return {"input_ids": torch.tensor([[3 + ord(char) % 31 for char in text]])}


TOKENIZER = _Tokenizer()
MODEL = torch.nn.Module()
cast(Any, MODEL)._saklas_source_fingerprint = "1" * 64


def _deterministic_acts() -> dict[int, torch.Tensor]:
    """Three layers of [N, D] fp32, fixed RNG so equality is testable."""
    g = torch.Generator().manual_seed(1234)
    return {
        idx: torch.randn(6, 8, generator=g, dtype=torch.float32)
        for idx in (0, 1, 2)
    }


def _install_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point $SAKLAS_HOME at a tmp dir so the real cache is never touched."""
    home = tmp_path / "saklas_home"
    home.mkdir()
    monkeypatch.setenv("SAKLAS_HOME", str(home))
    return home


def _patch_compute(
    monkeypatch: pytest.MonkeyPatch, acts: dict[int, torch.Tensor],
) -> dict[str, int]:
    """Monkeypatch the compute entry point to a fixed fp32 dict + count calls."""
    calls = {"n": 0}

    def _fake_compute(
        model: object, tokenizer: object, layers: object, **kwargs: object,
    ) -> dict[int, torch.Tensor]:
        del model, tokenizer, layers, kwargs
        calls["n"] += 1
        return {idx: t.clone() for idx, t in acts.items()}

    # Patched on the source module so alignment's deferred import resolves it.
    import saklas.core.capture as vectors

    monkeypatch.setattr(vectors, "compute_neutral_activations", _fake_compute)
    return calls


def _compute() -> dict[int, torch.Tensor]:
    return load_or_compute_neutral_activations(
        model=MODEL, tokenizer=TOKENIZER, layers=[0, 1, 2], model_id=MODEL_ID
    )


def test_store_is_fp32(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_home(tmp_path, monkeypatch)
    acts = _deterministic_acts()
    _patch_compute(monkeypatch, acts)

    _compute()

    anchor, pointer = _neutral_acts_paths(MODEL_ID)
    sidecar = json.loads(pointer.read_text())
    assert not anchor.exists(), "v3 must not publish a destructive fixed monolith"
    for layer, filename in sidecar["tensor_files"].items():
        on_disk = load_file(str(anchor.parent / filename))
        tensor = on_disk[f"layer_{layer}"]
        assert tensor.dtype == torch.float32


def test_metadata_preflight_does_not_materialize_tensor_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_home(tmp_path, monkeypatch)
    _patch_compute(monkeypatch, _deterministic_acts())
    _compute()

    import saklas.io.alignment as alignment

    monkeypatch.setattr(
        alignment, "load_safetensors",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("metadata preflight materialized tensors")
        ),
    )
    sidecar = validate_neutral_cache_metadata(MODEL_ID)
    assert sidecar["n_prompts"] == 6


def test_compute_and_cache_paths_bit_identical(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_home(tmp_path, monkeypatch)
    acts = _deterministic_acts()
    calls = _patch_compute(monkeypatch, acts)

    # First call: cache miss -> compute path returns the fresh fp32 dict.
    out_compute = _compute()
    assert calls["n"] == 1

    # Second call: cache hit -> load path. No recompute.
    out_cache = _compute()
    assert calls["n"] == 1, "cache hit should not recompute"

    assert out_compute.keys() == out_cache.keys()
    for layer in out_compute:
        assert torch.equal(out_compute[layer], out_cache[layer]), (
            f"layer {layer} differs across the cache boundary (precision seam)"
        )


def test_metadata_returning_load_reuses_the_single_payload_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dependent builders get identity without rehashing a validated cache."""
    _install_home(tmp_path, monkeypatch)
    _patch_compute(monkeypatch, _deterministic_acts())
    _compute()

    import saklas.io.alignment as alignment

    real_hash = alignment.hash_file
    hashed: list[Path] = []

    def count_hash(path: Path) -> str:
        hashed.append(Path(path))
        return real_hash(path)

    monkeypatch.setattr(alignment, "hash_file", count_hash)
    acts, sidecar = load_or_compute_neutral_activations_with_metadata(
        model=MODEL, tokenizer=TOKENIZER, layers=[0, 1, 2], model_id=MODEL_ID,
    )

    assert set(acts) == {0, 1, 2}
    anchor, _ = _neutral_acts_paths(MODEL_ID)
    assert sidecar["tensor_sha256"] == {
        layer: real_hash(anchor.parent / filename)
        for layer, filename in sidecar["tensor_files"].items()
    }
    assert hashed == []


def test_concurrent_cold_neutral_cache_is_single_flight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_home(tmp_path, monkeypatch)
    calls = _patch_compute(monkeypatch, _deterministic_acts())
    barrier = threading.Barrier(2)
    results: list[dict[int, torch.Tensor]] = []
    errors: list[BaseException] = []

    def run() -> None:
        try:
            barrier.wait()
            results.append(_compute())
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5.0)

    assert all(not thread.is_alive() for thread in threads)
    assert errors == []
    assert len(results) == 2
    assert calls["n"] == 1


def test_alignment_fit_lock_serializes_same_direction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from saklas.io.alignment import alignment_fit_lock

    _install_home(tmp_path, monkeypatch)
    entered = threading.Event()
    done = threading.Event()

    def acquire() -> None:
        entered.set()
        with alignment_fit_lock("source/model", "target/model"):
            done.set()

    with alignment_fit_lock("source/model", "target/model"):
        worker = threading.Thread(target=acquire)
        worker.start()
        assert entered.wait(1.0)
        assert not done.wait(0.1)
    worker.join(timeout=2.0)

    assert not worker.is_alive()
    assert done.is_set()


def test_stale_bf16_cache_invalidated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_home(tmp_path, monkeypatch)
    acts = _deterministic_acts()
    calls = _patch_compute(monkeypatch, acts)

    # Hand-write a stale bf16 cache + matching sidecar (the pre-decision
    # shape).  The sidecar hash is left blank so staleness is decided purely
    # by the on-disk dtype, not the statements-hash check.
    ts_path, sc_path = _neutral_acts_paths(MODEL_ID)
    model_dir(MODEL_ID).mkdir(parents=True, exist_ok=True)
    bf16 = {
        f"layer_{idx}": t.contiguous().to(torch.bfloat16).cpu()
        for idx, t in acts.items()
    }
    save_file(bf16, str(ts_path))
    sc_path.write_text(json.dumps({
        "method": "neutral_activations",
        "statements_sha256": "",
        "n_prompts": next(iter(acts.values())).shape[0],
        "n_layers": len(acts),
    }))
    # Sanity: the on-disk store is genuinely bf16 before the call.
    assert all(t.dtype == torch.bfloat16 for t in load_file(str(ts_path)).values())

    # The bf16 dtype must be treated as stale and recomputed as fp32.
    out = _compute()
    assert calls["n"] == 1, "stale bf16 cache should trigger a recompute"

    # In-memory result is fp32 ...
    for layer, t in out.items():
        assert t.dtype == torch.float32, f"layer {layer} returned {t.dtype}"
    # ... and the fixed stale monolith was replaced by fp32 v3 shards.
    assert not ts_path.exists()
    sidecar = json.loads(sc_path.read_text())
    for layer, filename in sidecar["tensor_files"].items():
        tensor = load_file(str(ts_path.parent / filename))[f"layer_{layer}"]
        assert tensor.dtype == torch.float32


def test_sidecar_written(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The sidecar rides alongside the fp32 store (cache-validity contract)."""
    _install_home(tmp_path, monkeypatch)
    acts = _deterministic_acts()
    _patch_compute(monkeypatch, acts)

    _compute()

    _, sc_path = _neutral_acts_paths(MODEL_ID)
    assert sc_path.exists()
    sc = json.loads(sc_path.read_text())
    assert sc["method"] == "neutral_activations"
    assert sc["n_layers"] == 3


def test_from_cache_rejects_stale_bf16(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``LayerWhitener.from_cache`` refuses a stale bf16 neutral cache.

    ``from_cache`` (the offline ``manifold compare`` + transfer-rebake path)
    can't recompute — no model is loaded — so it must fail loud rather than
    build a bf16-sourced whitener, which would reopen the precision seam the
    fp32 store closes.  The rejection is decided by the neutral-activation
    dtype; there is no separate ``layer_means`` cache to pair it with anymore
    (the centering mean is derived from the neutral activations).
    """
    _install_home(tmp_path, monkeypatch)
    acts = _deterministic_acts()
    _patch_compute(monkeypatch, acts)
    _compute()
    md = model_dir(MODEL_ID)

    pointer = json.loads((md / "neutral_activations.json").read_text())
    shard = md / pointer["tensor_files"]["0"]
    save_file({"layer_0": acts[0].to(torch.bfloat16)}, str(shard))

    with pytest.raises(WhitenerError, match="corrupt|legacy"):
        LayerWhitener.from_cache(MODEL_ID)


def test_selective_load_reads_only_requested_neutral_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_home(tmp_path, monkeypatch)
    _patch_compute(monkeypatch, _deterministic_acts())
    _compute()
    real_read = Path.read_bytes
    read_names: list[str] = []

    def counted_read(path: Path) -> bytes:
        read_names.append(path.name)
        return real_read(path)

    monkeypatch.setattr(Path, "read_bytes", counted_read)
    rows, _ = load_validated_neutral_cache(MODEL_ID, requested_layers=[1, 99])
    assert set(rows) == {1}
    assert len(read_names) == 1 and ".layer-1." in read_names[0]


def test_failed_neutral_pointer_publication_preserves_prior_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_home(tmp_path, monkeypatch)
    acts = _deterministic_acts()
    _patch_compute(monkeypatch, acts)
    _compute()
    anchor, pointer = _neutral_acts_paths(MODEL_ID)
    prior_pointer = pointer.read_bytes()
    prior_files = set(json.loads(prior_pointer)["tensor_files"].values())

    import saklas.io.alignment as alignment

    monkeypatch.setattr(
        alignment, "write_json_atomic",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("pointer failure")),
    )
    with pytest.raises(OSError, match="pointer failure"):
        load_or_compute_neutral_activations_with_metadata(
            MODEL, TOKENIZER, [0, 1, 2], model_id=MODEL_ID, force=True,
        )
    assert pointer.read_bytes() == prior_pointer
    assert {
        path.name for path in anchor.parent.glob(
            f"{anchor.stem}.layer-*.gen-*.safetensors"
        )
    } == prior_files


def test_neutral_exception_after_pointer_replace_preserves_new_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_home(tmp_path, monkeypatch)
    _patch_compute(monkeypatch, _deterministic_acts())
    import saklas.io.alignment as alignment

    _compute()
    anchor, pointer = _neutral_acts_paths(MODEL_ID)
    prior_files = set(json.loads(pointer.read_text())["tensor_files"].values())
    real_write = alignment.write_json_atomic

    def write_then_fail(path: Path, payload: object) -> None:
        real_write(path, payload)
        raise OSError("after neutral pointer replace")

    monkeypatch.setattr(alignment, "write_json_atomic", write_then_fail)
    with pytest.raises(OSError, match="after neutral pointer"):
        load_or_compute_neutral_activations_with_metadata(
            MODEL, TOKENIZER, [0, 1, 2], model_id=MODEL_ID, force=True,
        )

    monkeypatch.setattr(alignment, "write_json_atomic", real_write)
    rows, sidecar = load_validated_neutral_cache(MODEL_ID)
    assert set(rows) == {0, 1, 2}
    assert json.loads(pointer.read_text()) == sidecar
    assert all(
        (anchor.parent / name).is_file()
        for name in sidecar["tensor_files"].values()
    )
    assert not any((anchor.parent / name).exists() for name in prior_files)


def test_neutral_directory_barriers_bracket_pointer_and_precede_gc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_home(tmp_path, monkeypatch)
    _patch_compute(monkeypatch, _deterministic_acts())
    import saklas.io.alignment as alignment

    events: list[str] = []
    real_write = alignment.write_json_atomic
    real_cleanup = alignment._cleanup_neutral_generations

    def record_write(path: Path, payload: object) -> None:
        events.append("pointer")
        real_write(path, payload)

    def record_cleanup(anchor: Path, sidecar: object) -> None:
        events.append("gc")
        real_cleanup(anchor, sidecar)  # type: ignore[arg-type]

    monkeypatch.setattr(alignment, "write_json_atomic", record_write)
    monkeypatch.setattr(
        alignment, "fsync_directory", lambda _path: events.append("barrier"),
    )
    monkeypatch.setattr(alignment, "_cleanup_neutral_generations", record_cleanup)
    _compute()
    assert events == ["barrier", "pointer", "barrier", "gc", "barrier"]


def test_from_cache_builds_without_layer_means(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``from_cache`` derives centering means from the neutral cache alone.

    The neutral mean *is* the probe-centering baseline (same corpus, same
    pooling), so the offline whitener loader reads only
    ``neutral_activations.safetensors`` — there is no ``layer_means`` cache.
    """
    _install_home(tmp_path, monkeypatch)
    acts = _deterministic_acts()
    _patch_compute(monkeypatch, acts)
    _compute()

    got = LayerWhitener.from_cache(MODEL_ID)
    expected = LayerWhitener.from_neutral_activations(
        acts,
        {i: t.mean(dim=0) for i, t in acts.items()},
    )

    assert got.layers == set(acts)
    v = torch.randn(8, generator=torch.Generator().manual_seed(55))
    for layer in acts:
        assert torch.allclose(got.apply_inv(layer, v), expected.apply_inv(layer, v))
