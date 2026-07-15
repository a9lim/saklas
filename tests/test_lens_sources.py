from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from saklas.io.atomic import write_json_atomic


@pytest.fixture(autouse=True)
def _home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path / "saklas"))


def test_external_lens_stays_in_provider_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from saklas.io import lens_sources as sources
    from saklas.io.lens import load_lens, load_lens_sidecar

    provider = tmp_path / "hf-cache"
    provider.mkdir()
    config = provider / "config.yaml"
    config.write_text("hf_model_name: org/model\n")
    checkpoint = provider / "lens.pt"
    torch.save(
        {
            "J": {0: torch.eye(4, dtype=torch.float16)},
            "n_prompts": 12,
            "source_layers": [0],
            "d_model": 4,
        },
        checkpoint,
    )
    import hashlib

    binding = sources.ExternalLensBinding(
        name="neuronpedia",
        model_id="org/model",
        model_revision="model-commit",
        repo_id="neuronpedia/jacobian-lens",
        repo_revision="lens-commit",
        checkpoint="model/lens.pt",
        config_file="model/config.yaml",
        config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
        corpus="Salesforce/wikitext:wikitext-103-raw-v1:train",
        n_prompts=12,
        seq_len=128,
        dim_batch=8,
        d_model=4,
        source_layers=(0,),
    )
    write_json_atomic(
        sources.lens_binding_path("org/model", "neuronpedia"),
        binding.to_json(),
    )
    sources.set_active_lens_source("org/model", "huggingface", "neuronpedia")

    def download(_repo: str, filename: str, **_kwargs: object) -> str:
        return str(config if filename.endswith("config.yaml") else checkpoint)

    monkeypatch.setattr(sources, "_hf_hub_download", download)
    sidecar = load_lens_sidecar("org/model")
    assert sidecar is not None and sidecar["_source"]["provider"] == "neuronpedia"
    loaded = load_lens("org/model")
    assert loaded is not None
    lens, _ = loaded
    assert lens.source_layers == [0]
    assert lens.jacobians[0].dtype == torch.float32
    assert not list((tmp_path / "saklas").rglob("*.safetensors"))
    assert checkpoint.exists()
    rows = sources.list_lens_sources("org/model")
    assert rows[0]["source"] == "neuronpedia"
    assert rows[0]["repo_id"] == "neuronpedia/jacobian-lens"
    assert sources.remove_external_lens_binding("org/model")
    assert checkpoint.exists()  # provider cache ownership is preserved


def test_local_lens_layout_and_source_selection() -> None:
    from saklas.core.jlens import JacobianLens
    from saklas.io.lens import lens_paths, save_lens
    from saklas.io.lens_sources import list_lens_sources, load_active_lens_source

    save_lens(
        JacobianLens({0: torch.eye(3)}, n_prompts=2, d_model=3),
        "org/model",
        corpus_spec="test",
        corpus_sha256="a" * 64,
        seq_len=8,
        dim_batch=1,
        skip_first=0,
    )
    _tensor, manifest = lens_paths("org/model")
    assert manifest.parts[-4:] == ("jlens", "local", "default", "manifest.json")
    assert load_active_lens_source("org/model") == {
        "format_version": 1,
        "model_id": "org/model",
        "kind": "local",
        "name": "default",
    }
    rows = list_lens_sources("org/model")
    assert len(rows) == 1 and rows[0]["source"] == "local:default" and rows[0]["active"]


def test_lens_registry_requires_an_active_source() -> None:
    from saklas.core.jlens import JacobianLens
    from saklas.io.lens import load_lens, save_lens
    from saklas.io.lens_sources import lens_active_path

    save_lens(
        JacobianLens({0: torch.eye(3)}, n_prompts=2, d_model=3),
        "org/model",
        corpus_spec="test",
        corpus_sha256="a" * 64,
        seq_len=8,
        dim_batch=1,
        skip_first=0,
    )
    lens_active_path("org/model").unlink()
    assert load_lens("org/model") is None


def test_external_lens_identity_binds_model_commit_and_shape() -> None:
    from saklas.core.session import _jlens_matches_loaded_model

    model = SimpleNamespace(
        config=SimpleNamespace(
            _commit_hash="model-commit",
            hidden_size=4,
            num_hidden_layers=3,
        )
    )
    sidecar = {
        "model_fingerprint": None,
        "d_model": 4,
        "source_layers": [0, 1],
        "_source": {
            "kind": "huggingface",
            "model_id": "org/model",
            "model_revision": "model-commit",
        },
    }
    assert _jlens_matches_loaded_model(sidecar, model, "org/model")
    sidecar["_source"]["model_revision"] = "changed"
    assert not _jlens_matches_loaded_model(sidecar, model, "org/model")
