"""CPU-only session resolution regressions."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from saklas.core.events import EventBus
from saklas.core.session import SaklasSession


def _write_tensor_pack(tmp_path: Path, namespace: str, name: str, model_id: str, value: float):
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
    from saklas.core.extraction import ExtractionPipeline
    from saklas.core.session import GenState

    session = SaklasSession.__new__(SaklasSession)
    s: Any = session  # cast to Any for stub attribute assignments bypassing typed slots
    s._model_info = {"model_id": model_id}
    s._device = torch.device("cpu")
    s._dtype = torch.float32
    s.events = EventBus()
    s._model = SimpleNamespace()
    s._tokenizer = SimpleNamespace()
    s._layers = []
    s._profiles = {}
    s._gen_phase = GenState.IDLE
    import threading
    s._gen_lock = threading.Lock()
    # ``session.extract`` consults ``_dls`` / ``_layer_means`` for
    # defaults; without them on the stub it crashes with AttributeError
    # before delegating to the pipeline.
    s._dls = True
    s._layer_means = {}
    # Pipeline normally constructed in __init__; stub skips that, wire one in.
    s._extraction = ExtractionPipeline(session, session, session.events)
    return session


def test_extract_honors_namespace_when_pack_names_collide(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from saklas.io.selectors import invalidate

    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    invalidate()
    model_id = "fake/model"
    _write_tensor_pack(tmp_path, "default", "shared", model_id, 1.0)
    _write_tensor_pack(tmp_path, "bob", "shared", model_id, 2.0)

    name, profile = _stub_session(model_id).extract("shared", namespace="bob")

    assert name == "shared"
    assert torch.allclose(profile[0], torch.full((4,), 2.0))


def test_extract_bare_duplicate_pack_name_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from saklas.io.selectors import AmbiguousSelectorError, invalidate

    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    invalidate()
    model_id = "fake/model"
    _write_tensor_pack(tmp_path, "default", "shared", model_id, 1.0)
    _write_tensor_pack(tmp_path, "bob", "shared", model_id, 2.0)

    with pytest.raises(AmbiguousSelectorError):
        _stub_session(model_id).extract("shared")
