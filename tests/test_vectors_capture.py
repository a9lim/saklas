"""CPU-only hidden-capture regressions for extraction primitives."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

from saklas.core.vectors import _capture_all_hidden_states


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

    def forward(self, *, input_ids: torch.Tensor, use_cache: bool = False) -> Any:
        del use_cache
        h = input_ids.to(torch.float32).unsqueeze(-1).expand(-1, -1, 3)
        for layer in self.layers:
            h = layer(h)
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


def test_capture_all_hidden_states_can_filter_layers() -> None:
    model = _ToyModel()
    ids = torch.tensor([[10, 11, 12, 13]])

    pooled = _capture_all_hidden_states(
        model, model.layers, ids, pool_index=2, layer_indices=[1],
    )

    assert set(pooled) == {1}
    assert pooled[1].shape == (3,)
