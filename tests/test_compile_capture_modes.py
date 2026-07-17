"""Compile-time persistent hook mode selection."""

from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any

import pytest
import torch


class _FakeModel:
    def __init__(self, device: str = "mps") -> None:
        self.config = SimpleNamespace(hidden_size=4)
        self._param = SimpleNamespace(
            device=torch.device(device),
            dtype=torch.float32,
        )

    def parameters(self):
        yield self._param


class _FakeCompiled(_FakeModel):
    def __init__(self, base: _FakeModel) -> None:
        super().__init__()
        self._orig_mod = base
        self.config = base.config
        self._param = base._param


@pytest.mark.parametrize("device", ["mps", "cuda"])
def test_compiled_explicit_no_probes_skips_persistent_capture_hooks(
    device: str,
    monkeypatch: Any, tmp_path: Any,
) -> None:
    """``probes=[]`` should take the no-capture compiled accelerator mode."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))

    from saklas.core import cuda_graphs
    from saklas.core import model as model_mod
    from saklas.core import session as session_mod
    from saklas.core.session import SaklasSession
    from saklas.core.steering_composer import SteeringComposer

    calls: dict[str, int] = {"offset": 0, "capture": 0}
    base_model = _FakeModel(device)
    fake_tokenizer = SimpleNamespace()
    fake_layers = [object(), object()]

    def _offset(*args: Any, **kwargs: Any) -> tuple[dict[int, torch.Tensor], list[Any]]:
        calls["offset"] += 1
        return {}, []

    def _capture(*args: Any, **kwargs: Any) -> tuple[dict[int, torch.Tensor], list[Any]]:
        calls["capture"] += 1
        return {}, []

    monkeypatch.setattr(
        session_mod,
        "load_model",
        lambda *args, **kwargs: (base_model, fake_tokenizer),
    )
    monkeypatch.setattr(session_mod, "get_layers", lambda _model: fake_layers)
    monkeypatch.setattr(
        session_mod,
        "get_model_info",
        lambda _model, _tok: {
            "model_id": "fake/model",
            "device": device,
            "dtype": "float32",
        },
    )
    monkeypatch.setattr(model_mod, "get_layers", lambda _model: fake_layers)
    monkeypatch.setattr(
        model_mod,
        "_compile_with_probe",
        lambda model, tokenizer, device, mode=None: _FakeCompiled(model),
    )
    monkeypatch.setattr(
        cuda_graphs,
        "is_static_cache_supported",
        lambda model, device: (True, None),
    )
    monkeypatch.setattr(
        "saklas.core.hooks.install_persistent_offset_hooks",
        _offset,
    )
    monkeypatch.setattr(
        "saklas.core.hooks.install_persistent_capture_hooks",
        _capture,
    )

    session = SaklasSession.from_pretrained(
        "fake/model", device=device, compile=True, probes=[],
    )

    assert isinstance(session._steering_composer, SteeringComposer)
    assert calls == {"offset": 1, "capture": 0}
    assert session._capture_buffers == {}


def test_cuda_graph_compile_owner_is_thread_local() -> None:
    from saklas.core.session import SaklasSession

    session = SaklasSession.__new__(SaklasSession)
    session._cuda_graphs_active = True
    session._compile_owner_thread = threading.get_ident()
    assert session._compiled_graph_thread_safe() is True

    observed: list[bool] = []
    worker = threading.Thread(
        target=lambda: observed.append(session._compiled_graph_thread_safe())
    )
    worker.start()
    worker.join()

    assert observed == [False]
