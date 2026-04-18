"""CPU-only session resolution regressions."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from saklas.core.events import EventBus
from saklas.core.session import SaklasSession


def _write_tensor_pack(tmp_path, namespace: str, name: str, model_id: str, value: float):
    from saklas.core.vectors import save_profile
    from saklas.io.packs import PackMetadata, hash_folder_files
    from saklas.io.paths import tensor_filename

    folder = tmp_path / "vectors" / namespace / name
    folder.mkdir(parents=True)
    save_profile(
        {0: torch.full((4,), value)},
        str(folder / tensor_filename(model_id)),
        {"method": "test"},
    )
    meta = PackMetadata(
        name=name,
        description="test",
        version="1.0.0",
        license="MIT",
        tags=[],
        recommended_alpha=0.5,
        source="local",
        files=hash_folder_files(folder),
    )
    meta.write(folder)
    return folder


def _stub_session(model_id: str) -> SaklasSession:
    session = SaklasSession.__new__(SaklasSession)
    session._model_info = {"model_id": model_id}
    session._device = torch.device("cpu")
    session._dtype = torch.float32
    session.events = EventBus()
    session._model = SimpleNamespace()
    session._tokenizer = SimpleNamespace()
    session._layers = []
    return session


def test_extract_honors_namespace_when_pack_names_collide(monkeypatch, tmp_path):
    from saklas.cli.selectors import invalidate

    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    invalidate()
    model_id = "fake/model"
    _write_tensor_pack(tmp_path, "default", "shared", model_id, 1.0)
    _write_tensor_pack(tmp_path, "bob", "shared", model_id, 2.0)

    name, profile = _stub_session(model_id).extract("shared", namespace="bob")

    assert name == "shared"
    assert torch.allclose(profile[0], torch.full((4,), 2.0))


def test_extract_bare_duplicate_pack_name_raises(monkeypatch, tmp_path):
    from saklas.cli.selectors import AmbiguousSelectorError, invalidate

    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    invalidate()
    model_id = "fake/model"
    _write_tensor_pack(tmp_path, "default", "shared", model_id, 1.0)
    _write_tensor_pack(tmp_path, "bob", "shared", model_id, 2.0)

    with pytest.raises(AmbiguousSelectorError):
        _stub_session(model_id).extract("shared")
