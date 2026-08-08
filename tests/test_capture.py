"""CPU-only hidden-capture regressions for extraction primitives."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

from saklas.core.capture import (
    _capture_all_hidden_states,
    _load_json_with_override,
    _ReusablePooledCapture,
)


class _AddLayer(torch.nn.Module):
    def __init__(self, offset: float) -> None:
        super().__init__()
        self.offset = offset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.offset


class _ToyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([_AddLayer(1.0), _AddLayer(2.0)])
        self.head_calls = 0

    def forward(self, *, input_ids: torch.Tensor, use_cache: bool = False) -> Any:
        del use_cache
        h = input_ids.to(torch.float32).unsqueeze(-1).expand(-1, -1, 3)
        for layer in self.layers:
            h = layer(h)
        self.head_calls += 1
        return SimpleNamespace(last_hidden_state=h)


def test_capture_all_hidden_states_can_pool_inside_hook() -> None:
    model = _ToyModel()
    ids = torch.tensor([[10, 11, 12, 13]])

    full = _capture_all_hidden_states(model, model.layers, ids)
    pooled = _capture_all_hidden_states(
        model, model.layers, ids, pool_index=2,
    )

    assert full[0].shape == (1, 4, 3)
    assert pooled[0].shape == (3,)
    assert torch.allclose(pooled[0], full[0][0, 2].float())
    assert torch.allclose(pooled[1], full[1][0, 2].float())
    assert model.head_calls == 0


def test_capture_all_hidden_states_can_filter_layers() -> None:
    model = _ToyModel()
    ids = torch.tensor([[10, 11, 12, 13]])

    pooled = _capture_all_hidden_states(
        model, model.layers, ids, pool_index=2, layer_indices=[1],
    )

    assert set(pooled) == {1}
    assert pooled[1].shape == (3,)
    assert model.head_calls == 0


def test_reusable_capture_registers_hooks_once_for_many_forwards(
    monkeypatch: Any,
) -> None:
    model = _ToyModel()
    registrations = 0
    original = torch.nn.Module.register_forward_hook

    def counted_register(module: torch.nn.Module, *args: Any, **kwargs: Any):
        nonlocal registrations
        registrations += 1
        return original(module, *args, **kwargs)

    monkeypatch.setattr(torch.nn.Module, "register_forward_hook", counted_register)
    ids = torch.tensor([[10, 11, 12], [20, 21, 22]])
    pool = torch.tensor([1, 2])
    with _ReusablePooledCapture(model, model.layers, [0, 1]) as capture:
        first = _capture_all_hidden_states(
            model, model.layers, ids, pool_index=pool,
            layer_indices=[0, 1], capture_context=capture,
        )
        second = _capture_all_hidden_states(
            model, model.layers, ids + 1, pool_index=pool,
            layer_indices=[0, 1], capture_context=capture,
        )
        assert registrations == 2
        assert set(first) == {0, 1}
        assert torch.allclose(second[1], first[1] + 1)
    assert all(not layer._forward_hooks for layer in model.layers)


def test_one_off_per_row_pool_matches_the_reusable_context() -> None:
    """The one-off per-row capture and the fit-wide reusable context run the
    same hook, so they must agree exactly (and both leave the model clean)."""
    model = _ToyModel()
    ids = torch.tensor([[10, 11, 12], [20, 21, 22]])
    pool = torch.tensor([1, 2])

    one_off = _capture_all_hidden_states(
        model, model.layers, ids, pool_index=pool, layer_indices=[0, 1],
    )
    with _ReusablePooledCapture(model, model.layers, [0, 1]) as capture:
        reused = _capture_all_hidden_states(
            model, model.layers, ids, pool_index=pool,
            layer_indices=[0, 1], capture_context=capture,
        )

    assert set(one_off) == {0, 1}
    for layer in (0, 1):
        assert one_off[layer].shape == (2, 3)
        assert torch.allclose(one_off[layer], reused[layer])
    assert all(not layer._forward_hooks for layer in model.layers)
    assert model.head_calls == 0


def test_one_off_per_row_pool_clamps_positions_past_the_sequence() -> None:
    model = _ToyModel()
    ids = torch.tensor([[10, 11, 12], [20, 21, 22]])

    clamped = _capture_all_hidden_states(
        model, model.layers, ids, pool_index=torch.tensor([1, 99]),
        layer_indices=[1],
    )
    last = _capture_all_hidden_states(
        model, model.layers, ids, pool_index=torch.tensor([1, 2]),
        layer_indices=[1],
    )

    assert torch.allclose(clamped[1], last[1])


def test_json_override_prefers_the_user_file_over_the_bundled_resource(
    tmp_path: Any,
) -> None:
    override = tmp_path / "baseline_prompts.json"
    bundled = _load_json_with_override(override, "baseline_prompts.json")
    assert len(bundled) == 48

    override.write_text('["only this"]')
    assert _load_json_with_override(override, "baseline_prompts.json") == [
        "only this",
    ]
