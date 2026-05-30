"""Equivalence harness for the incremental monitored-capture path.

The VRAM optimization scores each generated token on-device as it is
produced (``TraitMonitor.score_single_token_tensor`` per forward) and
keeps only the per-token ``[P]`` score rows, instead of holding the full
``[T, layers, D]`` capture stack until ``_finalize_generation``. This is
purely an internal-mechanics change: the aggregate vals and per-token
dict it builds must be BIT-IDENTICAL (within fp tolerance) to the batched
``TraitMonitor.score_per_token`` path.

These CPU tests are the required substitute for the GPU-only in-flight
capture coverage (test_session / test_smoke). Two layers:

1. Monitor + finalize-aggregation equivalence: build a synthetic capture
   stack, run it through ``score_per_token`` (batched) and through the
   incremental row-by-row path that reproduces ``_finalize_generation``'s
   aggregation, and assert they agree — including the EOS off-by-one
   overshoot case proving the ``[:n]`` trim matches.
2. ``HiddenCapture`` incremental-mode unit test: attach to a tiny
   ``nn.ModuleList`` in incremental mode, run forwards, assert the step
   sink fires exactly once per forward and ``latest_per_layer`` returns
   the latest slice (so the streaming ``bucket[-1]`` tap still works).
"""
from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.nn as nn

from saklas.core.hooks import HiddenCapture
from saklas.core.monitor import TraitMonitor
from saklas.core.vectors import last_content_index, special_token_ids


# --------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------- #


class _FakeTokenizer:
    """Minimal tokenizer surface ``score_per_token`` consumes."""

    def __init__(self, special_ids: list[int]) -> None:
        self.all_special_ids = list(special_ids)


class _FakeTokenizerAdded:
    """Tokenizer whose chat markers live in ``added_tokens_encoder`` only.

    Mirrors talkie-style tokenizers that don't promote ``<|end|>`` /
    ``<|assistant|>`` to ``all_special_ids`` — the canonical
    ``last_content_index`` walkback must skip these too.
    """

    def __init__(
        self, special_ids: list[int], added: dict[str, int],
    ) -> None:
        self.all_special_ids = list(special_ids)
        self.added_tokens_encoder = dict(added)


# --------------------------------------------------------------------- #
# last_content_index — the one canonical "last non-special token" walkback
# --------------------------------------------------------------------- #


def test_last_content_index_skips_trailing_all_special() -> None:
    tok = _FakeTokenizer([100, 101])
    # ...content, then two structural markers.
    assert last_content_index([5, 6, 7, 100, 101], tok) == 2


def test_last_content_index_no_specials_is_final_row() -> None:
    tok = _FakeTokenizer([])
    assert last_content_index([5, 6, 7], tok) == 2


def test_last_content_index_skips_added_tokens_encoder() -> None:
    # The trailing 999 is NOT in all_special_ids, only added_tokens_encoder
    # — the weaker all_special_ids-only walkback would wrongly land on it.
    tok = _FakeTokenizerAdded([100], {"<|end|>": 999})
    assert special_token_ids(tok) == {100, 999}
    assert last_content_index([5, 6, 999], tok) == 1


def test_last_content_index_all_special_floors_at_zero() -> None:
    tok = _FakeTokenizer([100])
    # Every position is special — floor at 0 rather than going negative.
    assert last_content_index([100, 100, 100], tok) == 0


def test_last_content_index_empty_is_zero() -> None:
    tok = _FakeTokenizer([100])
    assert last_content_index([], tok) == 0


def _build_monitor(
    *, n_probes: int = 3, layers: tuple[int, ...] = (1, 3, 5), dim: int = 8,
    seed: int = 0,
) -> TraitMonitor:
    """A monitor with a few random fp32 probes over a few layers + means."""
    g = torch.Generator().manual_seed(seed)
    profiles: dict[str, dict[int, torch.Tensor]] = {}
    for p in range(n_probes):
        prof: dict[int, torch.Tensor] = {}
        for li in layers:
            # Random direction scaled by a random magnitude so the per-layer
            # ||baked|| weight varies across layers/probes (exercises W).
            vec = torch.randn(dim, generator=g, dtype=torch.float32)
            scale = 0.3 + torch.rand(1, generator=g).item()
            prof[li] = vec * scale
        profiles[f"probe_{p}"] = prof
    layer_means = {
        li: torch.randn(dim, generator=g, dtype=torch.float32) * 0.1
        for li in layers
    }
    return TraitMonitor(profiles, layer_means=layer_means)


def _synthetic_capture(
    layers: tuple[int, ...], *, T: int, dim: int, seed: int = 1,
) -> dict[int, torch.Tensor]:
    """``{layer: [T, D]}`` random fp32 capture stack."""
    g = torch.Generator().manual_seed(seed)
    return {
        li: torch.randn(T, dim, generator=g, dtype=torch.float32)
        for li in layers
    }


def _incremental_finalize(
    monitor: TraitMonitor,
    captured: dict[int, torch.Tensor],
    generated_ids: list[int],
    tokenizer: Any,
    *,
    accumulate: bool,
) -> tuple[dict[str, float], dict[str, list[float]]]:
    """Reproduce session._score_incremental from per-token slices.

    Feeds each token's per-layer slice through
    ``score_single_token_tensor`` (read-only, on-device), stacks the rows,
    and runs the EXACT aggregation ``SaklasSession._finalize_generation``
    does on the incremental path. Kept in lockstep with session.py.
    """
    n = len(generated_ids)
    # Build the rows exactly as the step_sink does: one per forward, fed
    # the latest per-layer slice. The capture overshoot (F rows for n
    # tokens) models the EOS off-by-one: the forward fires once more than
    # generated_ids has entries.
    any_h = next(iter(captured.values()))
    n_forwards = any_h.shape[0]
    rows: list[torch.Tensor] = []
    for t in range(n_forwards):
        latest = {li: captured[li][t] for li in captured}
        rows.append(monitor.score_single_token_tensor(latest))

    if not rows or n == 0:
        empty_agg = {name: 0.0 for name in monitor._raw_profiles}
        return empty_agg, {name: [] for name in monitor._raw_profiles}

    probe_keys = monitor._cache_probe_keys
    n_probes = len(probe_keys)
    stacked = torch.stack(rows)  # [F, P]
    if stacked.shape[0] > n:
        stacked = stacked[:n]
    result = stacked.cpu().tolist()

    per_token: dict[str, list[float]] = {
        name: [] for name in monitor._raw_profiles
    }
    for i, name in enumerate(probe_keys):
        per_token[name] = [row[i] for row in result]
    for name in monitor._raw_profiles:
        if not per_token[name]:
            per_token[name] = [0.0] * n

    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    agg_idx = n - 1
    while agg_idx > 0 and int(generated_ids[agg_idx]) in special_ids:
        agg_idx -= 1
    if n_probes and 0 <= agg_idx < len(result):
        agg_row = result[agg_idx]
        agg_vals = {name: agg_row[i] for i, name in enumerate(probe_keys)}
    else:
        agg_vals = {}
    for name in monitor._raw_profiles:
        agg_vals.setdefault(name, 0.0)

    if accumulate:
        monitor._apply_accumulate(agg_vals)
        monitor._pending_per_token = True
    return agg_vals, per_token


def _assert_equiv(
    agg_b: dict[str, float], pt_b: dict[str, list[float]],
    agg_i: dict[str, float], pt_i: dict[str, list[float]],
    *, tol: float = 1e-5,
) -> None:
    assert agg_b.keys() == agg_i.keys()
    for name in agg_b:
        assert agg_i[name] == pytest.approx(agg_b[name], abs=tol), (
            f"aggregate mismatch on {name}: {agg_i[name]} vs {agg_b[name]}"
        )
    assert pt_b.keys() == pt_i.keys()
    for name in pt_b:
        assert len(pt_i[name]) == len(pt_b[name]), (
            f"per-token length mismatch on {name}: "
            f"{len(pt_i[name])} vs {len(pt_b[name])}"
        )
        for k, (a, b) in enumerate(zip(pt_i[name], pt_b[name])):
            assert a == pytest.approx(b, abs=tol), (
                f"per-token mismatch on {name}[{k}]: {a} vs {b}"
            )


# --------------------------------------------------------------------- #
# Equivalence tests: batched score_per_token vs incremental finalize
# --------------------------------------------------------------------- #


def test_incremental_matches_batched_exact_length():
    layers = (1, 3, 5)
    dim = 8
    T = 6
    monitor_b = _build_monitor(layers=layers, dim=dim)
    monitor_i = _build_monitor(layers=layers, dim=dim)  # identical seeds
    captured = _synthetic_capture(layers, T=T, dim=dim)
    generated_ids = [10, 11, 12, 13, 14, 15]
    tok = _FakeTokenizer(special_ids=[0, 1, 2])

    agg_b, pt_b = monitor_b.score_per_token(
        captured, generated_ids, tok, accumulate=False,
    )
    agg_i, pt_i = _incremental_finalize(
        monitor_i, captured, generated_ids, tok, accumulate=False,
    )
    _assert_equiv(agg_b, pt_b, agg_i, pt_i)
    # Both per-token tracks cover all n tokens.
    for name in pt_b:
        assert len(pt_b[name]) == T
        assert len(pt_i[name]) == T


def test_incremental_matches_batched_eos_overshoot():
    """Capture has T = n + 1 rows (EOS off-by-one); the [:n] trim aligns."""
    layers = (0, 2, 4)
    dim = 8
    T = 7  # one extra capture row beyond the 6 generated ids
    monitor_b = _build_monitor(layers=layers, dim=dim, seed=3)
    monitor_i = _build_monitor(layers=layers, dim=dim, seed=3)
    captured = _synthetic_capture(layers, T=T, dim=dim, seed=9)
    generated_ids = [20, 21, 22, 23, 24, 25]  # n = 6, T = 7
    tok = _FakeTokenizer(special_ids=[99])

    agg_b, pt_b = monitor_b.score_per_token(
        captured, generated_ids, tok, accumulate=False,
    )
    agg_i, pt_i = _incremental_finalize(
        monitor_i, captured, generated_ids, tok, accumulate=False,
    )
    _assert_equiv(agg_b, pt_b, agg_i, pt_i)
    for name in pt_b:
        assert len(pt_b[name]) == len(generated_ids)
        assert len(pt_i[name]) == len(generated_ids)


def test_incremental_matches_batched_with_trailing_special_agg():
    """Aggregate walks back past trailing special tokens — both paths agree."""
    layers = (1, 2)
    dim = 8
    T = 5
    monitor_b = _build_monitor(layers=layers, dim=dim, seed=7)
    monitor_i = _build_monitor(layers=layers, dim=dim, seed=7)
    captured = _synthetic_capture(layers, T=T, dim=dim, seed=4)
    # Last two ids are special → agg pools row 2.
    generated_ids = [30, 31, 32, 88, 89]
    tok = _FakeTokenizer(special_ids=[88, 89])

    agg_b, pt_b = monitor_b.score_per_token(
        captured, generated_ids, tok, accumulate=False,
    )
    agg_i, pt_i = _incremental_finalize(
        monitor_i, captured, generated_ids, tok, accumulate=False,
    )
    _assert_equiv(agg_b, pt_b, agg_i, pt_i)


def test_incremental_accumulate_matches_history_and_flags():
    """accumulate=True: history, stats, and pending flags match batched."""
    layers = (1, 3)
    dim = 8
    T = 4
    monitor_b = _build_monitor(layers=layers, dim=dim, seed=11)
    monitor_i = _build_monitor(layers=layers, dim=dim, seed=11)
    captured = _synthetic_capture(layers, T=T, dim=dim, seed=12)
    generated_ids = [40, 41, 42, 43]
    tok = _FakeTokenizer(special_ids=[])

    agg_b, pt_b = monitor_b.score_per_token(
        captured, generated_ids, tok, accumulate=True,
    )
    agg_i, pt_i = _incremental_finalize(
        monitor_i, captured, generated_ids, tok, accumulate=True,
    )
    _assert_equiv(agg_b, pt_b, agg_i, pt_i)

    # History / stats accumulation must match exactly.
    for name in monitor_b._raw_profiles:
        hb = list(monitor_b.history[name])
        hi = list(monitor_i.history[name])
        assert len(hb) == len(hi) == 1
        assert hi[-1] == pytest.approx(hb[-1], abs=1e-5)
        sb, si = monitor_b._stats[name], monitor_i._stats[name]
        assert si["count"] == sb["count"] == 1
        assert si["sum"] == pytest.approx(sb["sum"], abs=1e-5)
        assert si["sum_sq"] == pytest.approx(sb["sum_sq"], abs=1e-5)
        assert si["min"] == pytest.approx(sb["min"], abs=1e-5)
        assert si["max"] == pytest.approx(sb["max"], abs=1e-5)

    # Both pending flags flip identically.
    assert monitor_i._pending_aggregate == monitor_b._pending_aggregate is True
    assert monitor_i._pending_per_token == monitor_b._pending_per_token is True


def test_incremental_stateless_leaves_history_untouched():
    """accumulate=False: no history / stats / flag mutation, like batched."""
    layers = (1, 3)
    dim = 8
    monitor = _build_monitor(layers=layers, dim=dim, seed=13)
    captured = _synthetic_capture(layers, T=4, dim=dim, seed=14)
    generated_ids = [50, 51, 52, 53]
    tok = _FakeTokenizer(special_ids=[])

    _incremental_finalize(
        monitor, captured, generated_ids, tok, accumulate=False,
    )
    for name in monitor._raw_profiles:
        assert list(monitor.history[name]) == []
        assert monitor._stats[name]["count"] == 0
    assert monitor._pending_aggregate is False
    assert monitor._pending_per_token is False


def test_score_single_token_tensor_no_sync_and_shape():
    """The per-token tensor scorer returns [P] on the input device, no sync."""
    layers = (1, 3, 5)
    dim = 8
    monitor = _build_monitor(layers=layers, dim=dim, seed=15)
    captured = _synthetic_capture(layers, T=2, dim=dim, seed=16)
    latest = {li: captured[li][0] for li in layers}
    row = monitor.score_single_token_tensor(latest)
    assert isinstance(row, torch.Tensor)
    assert row.shape == (len(monitor._raw_profiles),)
    assert row.dtype == torch.float32
    # The on-device row must equal the dict-form aggregate from
    # score_single_token (which goes through the same num/den math).
    agg = monitor.score_single_token(latest)
    keys = monitor._cache_probe_keys
    for i, name in enumerate(keys):
        assert float(row[i]) == pytest.approx(agg[name], abs=1e-6)


def test_score_single_token_tensor_empty_and_no_probes():
    """Empty hidden / no probes → zeros [P] on a sensible device."""
    layers = (1, 3)
    monitor = _build_monitor(layers=layers, dim=8, seed=17)
    # Empty hidden — P probes, all-zero row.
    row = monitor.score_single_token_tensor({})
    assert row.shape == (len(monitor._raw_profiles),)
    assert torch.all(row == 0.0)

    # No probes at all → [0] tensor.
    empty_monitor = TraitMonitor({}, layer_means=None)
    row2 = empty_monitor.score_single_token_tensor(
        {1: torch.randn(8)},
    )
    assert row2.shape == (0,)


# --------------------------------------------------------------------- #
# HiddenCapture incremental-mode unit test
# --------------------------------------------------------------------- #


class _IdentityLayer(nn.Module):
    """Returns a tuple ``(hidden,)`` like a transformer decoder layer.

    Adds a per-layer bias so each layer's captured slice is distinct and
    we can verify the step sink sees this step's value at every layer.
    """

    def __init__(self, bias: float) -> None:
        super().__init__()
        self.bias = bias

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        return (x + self.bias,)


def test_hidden_capture_incremental_sink_fires_once_per_forward():
    dim = 4
    n_layers = 3
    layers = nn.ModuleList(_IdentityLayer(float(i)) for i in range(n_layers))
    layer_indices = list(range(n_layers))

    cap = HiddenCapture()
    cap.attach(layers, layer_indices)

    fire_log: list[dict[int, torch.Tensor]] = []

    def sink(latest: dict[int, torch.Tensor]) -> None:
        # Snapshot the latest dict (clone slices so later forwards don't
        # alias what we recorded).
        fire_log.append({li: t.clone() for li, t in latest.items()})

    cap.set_incremental(sink)
    assert cap._incremental is True
    assert cap._max_layer == n_layers - 1

    # Run two "forwards": feed the [1, 1, D] shape the real hook slices
    # via output[0][0, -1, :]. Each layer transforms the previous output.
    def run_forward(seq: torch.Tensor) -> None:
        x = seq
        for layer in layers:
            (x,) = layer(x)

    seq0 = torch.zeros(1, 1, dim)
    run_forward(seq0)
    assert len(fire_log) == 1, "sink must fire exactly once per forward"

    seq1 = torch.ones(1, 1, dim) * 10.0
    run_forward(seq1)
    assert len(fire_log) == 2, "second forward fires the sink once more"

    # Each fire must carry every hooked layer's slice for THAT step.
    for fired in fire_log:
        assert set(fired.keys()) == set(layer_indices)

    # Buckets stay length-1 in incremental mode (O(layers·D) memory).
    for li in layer_indices:
        assert len(cap._per_layer[li]) == 1

    # latest_per_layer returns the latest slice; bucket[-1] (the tap) too.
    latest = cap.latest_per_layer()
    assert set(latest.keys()) == set(layer_indices)
    for li in layer_indices:
        # After forward 2 (input all-10s), layer li output = 10 + cumulative
        # bias 0+1+...+li. Verify the latest matches the second forward.
        expected = 10.0 + sum(range(li + 1))
        assert torch.allclose(latest[li], torch.full((dim,), expected))
        assert torch.allclose(cap._per_layer[li][-1], latest[li])


def test_hidden_capture_non_incremental_appends_full_stack():
    """Default (non-incremental) mode is byte-identical to the append path."""
    dim = 4
    n_layers = 2
    layers = nn.ModuleList(_IdentityLayer(float(i)) for i in range(n_layers))
    cap = HiddenCapture()
    cap.attach(layers, list(range(n_layers)))
    assert cap._incremental is False

    def run_forward(seq: torch.Tensor) -> None:
        x = seq
        for layer in layers:
            (x,) = layer(x)

    run_forward(torch.zeros(1, 1, dim))
    run_forward(torch.ones(1, 1, dim))
    run_forward(torch.ones(1, 1, dim) * 2)

    stacked = cap.stacked()
    for li in range(n_layers):
        assert stacked[li].shape == (3, dim)  # full [T, D] retained


def test_hidden_capture_attach_resets_incremental_state():
    """attach() must clear incremental state set by a prior gen."""
    layers = nn.ModuleList(_IdentityLayer(0.0) for _ in range(2))
    cap = HiddenCapture()
    cap.attach(layers, [0, 1])
    cap.set_incremental(lambda latest: None)
    assert cap._incremental is True

    # Re-attach (next gen) — incremental flags reset to the append default.
    cap.attach(layers, [0, 1])
    assert cap._incremental is False
    assert cap._step_sink is None
    assert cap._max_layer is None


def test_hidden_capture_clear_resets_incremental_state():
    layers = nn.ModuleList(_IdentityLayer(0.0) for _ in range(2))
    cap = HiddenCapture()
    cap.attach(layers, [0, 1])
    cap.set_incremental(lambda latest: None)
    cap.detach()
    cap.clear()
    assert cap._per_layer == {}
    assert cap._incremental is False
    assert cap._step_sink is None
    assert cap._max_layer is None
