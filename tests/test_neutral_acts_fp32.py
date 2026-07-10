"""Neutral-activation cache is stored fp32 (no precision seam).

Every saklas safetensor artifact is stored fp32 (cast at the writer); the
per-model neutral-activation cache was the last exception (bf16) and is now
fp32 too.  These tests pin three things: the on-disk store is fp32, the
compute path (cache miss) and the cache-hit path return bit-identical tensors
(the seam fix — the whitener covariance is bit-reproducible across the cache
boundary), and a pre-existing bf16 cache is invalidated and recomputed to fp32.

CPU-only: ``compute_neutral_activations`` is monkeypatched to a deterministic
fp32 dict so no real model loads.  ``$SAKLAS_HOME`` is pointed at a tmp dir so
the real cache is never touched.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from saklas.core.mahalanobis import LayerWhitener, WhitenerError
from saklas.io.alignment import (
    _neutral_acts_paths,
    load_or_compute_neutral_activations,
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
    import saklas.core.vectors as vectors

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

    ts_path, _ = _neutral_acts_paths(MODEL_ID)
    on_disk = load_file(str(ts_path))
    assert on_disk, "expected neutral-activation tensors on disk"
    for k, t in on_disk.items():
        assert t.dtype == torch.float32, f"{k} stored {t.dtype}, expected fp32"


def test_metadata_preflight_does_not_materialize_tensor_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_home(tmp_path, monkeypatch)
    _patch_compute(monkeypatch, _deterministic_acts())
    _compute()

    import saklas.io.alignment as alignment

    monkeypatch.setattr(
        alignment, "load_file",
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


def test_legacy_bf16_cache_invalidated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_home(tmp_path, monkeypatch)
    acts = _deterministic_acts()
    calls = _patch_compute(monkeypatch, acts)

    # Hand-write a legacy bf16 cache + matching sidecar (the pre-decision
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
    assert calls["n"] == 1, "legacy bf16 cache should trigger a recompute"

    # In-memory result is fp32 ...
    for layer, t in out.items():
        assert t.dtype == torch.float32, f"layer {layer} returned {t.dtype}"
    # ... and the on-disk store was rewritten fp32.
    on_disk = load_file(str(ts_path))
    for k, t in on_disk.items():
        assert t.dtype == torch.float32, f"{k} rewritten as {t.dtype}, expected fp32"


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


def test_from_cache_rejects_legacy_bf16(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``LayerWhitener.from_cache`` refuses a legacy bf16 neutral cache.

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

    save_file(
        {f"layer_{i}": t.contiguous().to(torch.bfloat16).cpu() for i, t in acts.items()},
        str(md / "neutral_activations.safetensors"),
    )

    with pytest.raises(WhitenerError, match="corrupt|legacy"):
        LayerWhitener.from_cache(MODEL_ID)


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
