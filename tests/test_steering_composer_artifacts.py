"""Cold runtime loading for integrity- and identity-bound manifold artifacts."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from saklas.core.model import loaded_model_fingerprint
from saklas.core.steering_composer import SteeringComposer
from saklas.core.vectors import fold_directions_to_subspace
from saklas.io.manifold_folder import ManifoldFormatError
from saklas.io.manifolds import create_baked_manifold_folder, transfer_manifold
from saklas.io.paths import encode_release_id
from tests._whitener import synthetic_means, synthetic_whitener


def _runtime(model_id: str, model: torch.nn.Module) -> Any:
    return SimpleNamespace(
        model_id=model_id, _model=model, _device=torch.device("cpu"),
        _manifolds={}, _profiles={},
    )


def _fold(name: str, directions: dict[int, torch.Tensor], *, label: str) -> Any:
    layers = sorted(directions)
    dim = int(next(iter(directions.values())).numel())
    means = synthetic_means(layers, dim)
    whitener = synthetic_whitener(layers, dim, means=means)
    return fold_directions_to_subspace(
        name, directions, means, whitener=whitener, label=label,
    )


def test_baked_manifold_cold_loads_with_proven_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    model = torch.nn.Linear(4, 4, bias=False)
    model_id = "test/model"
    fingerprint = loaded_model_fingerprint(model, model_id)
    manifold = _fold(
        "merged", {0: torch.tensor([1.0, 0.0, 0.0, 0.0])}, label="merged",
    )
    create_baked_manifold_folder(
        "local", "merged", "", manifold, model_id, method="folded_vector",
        model_fingerprint=fingerprint,
    )

    runtime = _runtime(model_id, model)
    SteeringComposer(runtime).ensure_manifold_loaded("local/merged")
    assert "local/merged" in runtime._manifolds


def test_cold_load_rejects_finite_tensor_corruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    model = torch.nn.Linear(4, 4, bias=False)
    model_id = "test/model"
    fingerprint = loaded_model_fingerprint(model, model_id)
    manifold = _fold("merged", {0: torch.ones(4)}, label="merged")
    folder, _ = create_baked_manifold_folder(
        "local", "merged", "", manifold, model_id, method="folded_vector",
        model_fingerprint=fingerprint,
    )
    tensor = next(folder.glob("*.safetensors"))
    payload = bytearray(tensor.read_bytes())
    payload[-1] ^= 1
    tensor.write_bytes(payload)

    with pytest.raises(ManifoldFormatError, match="integrity"):
        SteeringComposer(_runtime(model_id, model)).ensure_manifold_loaded(
            "local/merged",
        )


def test_cold_load_rejects_untracked_fitted_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    model = torch.nn.Linear(4, 4, bias=False)
    model_id = "test/model"
    fingerprint = loaded_model_fingerprint(model, model_id)
    manifold = _fold("merged", {0: torch.ones(4)}, label="merged")
    folder, _ = create_baked_manifold_folder(
        "local", "merged", "", manifold, model_id, method="folded_vector",
        model_fingerprint=fingerprint,
    )
    manifest_path = folder / "manifold.json"
    import json

    manifest = json.loads(manifest_path.read_text())
    manifest["files"] = {}
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ManifoldFormatError, match="no proof"):
        SteeringComposer(_runtime(model_id, model)).ensure_manifold_loaded(
            "local/merged",
        )


def test_transfer_variant_cold_loads_for_target_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    source_id = "src/model"
    target_id = "tgt/model"
    source_fp = "source-fingerprint"
    target_model = torch.nn.Linear(4, 4, bias=False)
    target_fp = loaded_model_fingerprint(target_model, target_id)
    manifold = _fold(
        "merged", {0: torch.tensor([1.0, 0.0, 0.0, 0.0])}, label="merged",
    )
    folder, _ = create_baked_manifold_folder(
        "local", "merged", "", manifold, source_id, method="folded_vector",
        model_fingerprint=source_fp,
    )
    from saklas.core.mahalanobis import LayerWhitener
    from saklas.io.alignment import LayerAlignment

    acts = {0: torch.randn(32, 4)}
    whitener = LayerWhitener.from_neutral_activations(
        acts, {0: acts[0].mean(dim=0)},
    )
    transfer_manifold(
        folder, from_model=source_id, to_model=target_id,
        alignment={0: LayerAlignment(torch.eye(4), torch.eye(4), torch.zeros(4))},
        whitener=whitener,
        target_layer_means={0: acts[0].mean(dim=0)},
        source_model_fingerprint=source_fp,
        target_model_fingerprint=target_fp,
    )

    key = f"local/merged:from-{encode_release_id(source_id)}"
    runtime = _runtime(target_id, target_model)
    SteeringComposer(runtime).ensure_manifold_loaded(key)
    assert key in runtime._manifolds
