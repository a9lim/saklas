"""Source registry for local and externally managed Jacobian lenses.

Saklas owns lenses it fits and stores those payloads below ``SAKLAS_HOME``.
External publishers retain ownership of their payloads: a Hugging Face lens
stays in the Hub cache and Saklas writes only a small, commit-pinned binding
plus the per-model active-source selection.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import torch
import yaml

from saklas.core.jlens import JacobianLens
from saklas.io.atomic import artifact_lock, write_json_atomic
from saklas.io.paths import ensure_within, model_dir

LENS_SOURCE_FORMAT_VERSION = 1
NEURONPEDIA_REPO = "neuronpedia/jacobian-lens"
NEURONPEDIA_BINDING = "neuronpedia"
_LOCAL_NAME_RE = re.compile(r"^[a-z][a-z0-9._-]{0,63}$")


def lens_root(model_id: str) -> Path:
    return model_dir(model_id) / "jlens"


def local_lens_dir(model_id: str, name: str = "default") -> Path:
    if _LOCAL_NAME_RE.fullmatch(name) is None:
        raise ValueError(f"invalid local J-lens name {name!r}")
    return ensure_within(lens_root(model_id) / "local", name)


def lens_bindings_dir(model_id: str) -> Path:
    return lens_root(model_id) / "bindings"


def lens_binding_path(model_id: str, name: str) -> Path:
    if _LOCAL_NAME_RE.fullmatch(name) is None:
        raise ValueError(f"invalid J-lens binding name {name!r}")
    return ensure_within(lens_bindings_dir(model_id), f"{name}.json")


def lens_active_path(model_id: str) -> Path:
    return lens_root(model_id) / "active.json"


def _active_payload(model_id: str, kind: str, name: str) -> dict[str, Any]:
    return {
        "format_version": LENS_SOURCE_FORMAT_VERSION,
        "model_id": model_id,
        "kind": kind,
        "name": name,
    }


def set_active_lens_source(model_id: str, kind: str, name: str) -> Path:
    if kind not in {"local", "huggingface"}:
        raise ValueError(f"unknown J-lens source kind {kind!r}")
    if _LOCAL_NAME_RE.fullmatch(name) is None:
        raise ValueError(f"invalid J-lens source name {name!r}")
    if kind == "local":
        if not (local_lens_dir(model_id, name) / "manifest.json").exists():
            raise FileNotFoundError(f"local J-lens {name!r} is not fitted")
    elif not lens_binding_path(model_id, name).exists():
        raise FileNotFoundError(f"external J-lens binding {name!r} is not fetched")
    path = lens_active_path(model_id)
    with artifact_lock(path):
        write_json_atomic(path, _active_payload(model_id, kind, name))
    return path


def set_active_local_lens(model_id: str, name: str = "default") -> Path:
    """Make a local lens active after its manifest has been published."""
    return set_active_lens_source(model_id, "local", name)


def load_active_lens_source(model_id: str) -> dict[str, Any] | None:
    path = lens_active_path(model_id)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if (
        not isinstance(payload, dict)
        or set(payload) != {"format_version", "model_id", "kind", "name"}
        or payload.get("format_version") != LENS_SOURCE_FORMAT_VERSION
        or payload.get("model_id") != model_id
        or payload.get("kind") not in {"local", "huggingface"}
        or not isinstance(payload.get("name"), str)
        or _LOCAL_NAME_RE.fullmatch(payload["name"]) is None
    ):
        return None
    return payload


def _model_commit(config: Any) -> str | None:
    commit = getattr(config, "_commit_hash", None) or getattr(
        getattr(config, "text_config", None), "_commit_hash", None
    )
    return str(commit) if isinstance(commit, str) and commit else None


def _config_dimensions(config: Any) -> tuple[int | None, int | None]:
    text = getattr(config, "text_config", config)
    hidden = getattr(text, "hidden_size", getattr(text, "n_embd", None))
    layers = getattr(
        text,
        "num_hidden_layers",
        getattr(text, "n_layer", None),
    )
    return (
        int(hidden) if isinstance(hidden, int) and not isinstance(hidden, bool) else None,
        int(layers) if isinstance(layers, int) and not isinstance(layers, bool) else None,
    )


def _snapshot_download(*args: Any, **kwargs: Any) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(*args, **kwargs)


def _hf_hub_download(*args: Any, **kwargs: Any) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(*args, **kwargs)


def _hf_api() -> Any:
    from huggingface_hub import HfApi

    return HfApi()


@dataclass(frozen=True)
class ExternalLensBinding:
    name: str
    model_id: str
    model_revision: str
    repo_id: str
    repo_revision: str
    checkpoint: str
    config_file: str
    config_sha256: str
    corpus: str
    n_prompts: int
    seq_len: int
    dim_batch: int
    d_model: int
    source_layers: tuple[int, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "format_version": LENS_SOURCE_FORMAT_VERSION,
            "kind": "huggingface",
            "provider": "neuronpedia",
            "name": self.name,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "repo_id": self.repo_id,
            "repo_revision": self.repo_revision,
            "checkpoint": self.checkpoint,
            "config_file": self.config_file,
            "config_sha256": self.config_sha256,
            "corpus": self.corpus,
            "n_prompts": self.n_prompts,
            "seq_len": self.seq_len,
            "dim_batch": self.dim_batch,
            "d_model": self.d_model,
            "source_layers": list(self.source_layers),
        }


_BINDING_FIELDS = {
    "format_version",
    "kind",
    "provider",
    "name",
    "model_id",
    "model_revision",
    "repo_id",
    "repo_revision",
    "checkpoint",
    "config_file",
    "config_sha256",
    "corpus",
    "n_prompts",
    "seq_len",
    "dim_batch",
    "d_model",
    "source_layers",
}


def _parse_binding(payload: Any, model_id: str, name: str) -> ExternalLensBinding:
    if not isinstance(payload, dict) or set(payload) != _BINDING_FIELDS:
        raise ValueError("external J-lens binding has an invalid schema")
    layers = payload.get("source_layers")
    if (
        payload.get("format_version") != LENS_SOURCE_FORMAT_VERSION
        or payload.get("kind") != "huggingface"
        or payload.get("provider") != "neuronpedia"
        or payload.get("name") != name
        or payload.get("model_id") != model_id
        or not isinstance(layers, list)
        or not layers
        or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in layers)
        or layers != sorted(set(layers))
    ):
        raise ValueError("external J-lens binding has invalid identity metadata")
    for key in (
        "model_revision",
        "repo_id",
        "repo_revision",
        "checkpoint",
        "config_file",
        "config_sha256",
        "corpus",
    ):
        if not isinstance(payload.get(key), str) or not payload[key]:
            raise ValueError(f"external J-lens binding has invalid {key}")
    for key in ("n_prompts", "seq_len", "dim_batch", "d_model"):
        if isinstance(payload.get(key), bool) or not isinstance(payload.get(key), int) or payload[key] <= 0:
            raise ValueError(f"external J-lens binding has invalid {key}")
    if len(payload["config_sha256"]) != 64:
        raise ValueError("external J-lens binding has invalid config digest")
    return ExternalLensBinding(
        name=name,
        model_id=model_id,
        model_revision=payload["model_revision"],
        repo_id=payload["repo_id"],
        repo_revision=payload["repo_revision"],
        checkpoint=payload["checkpoint"],
        config_file=payload["config_file"],
        config_sha256=payload["config_sha256"],
        corpus=payload["corpus"],
        n_prompts=payload["n_prompts"],
        seq_len=payload["seq_len"],
        dim_batch=payload["dim_batch"],
        d_model=payload["d_model"],
        source_layers=tuple(layers),
    )


def load_external_lens_binding(
    model_id: str,
    name: str = NEURONPEDIA_BINDING,
) -> ExternalLensBinding | None:
    path = lens_binding_path(model_id, name)
    if not path.exists():
        return None
    try:
        return _parse_binding(json.loads(path.read_text()), model_id, name)
    except (OSError, ValueError, TypeError, KeyError):
        return None


def _match_official_config(
    root: Path,
    model_id: str,
    dataset: str,
) -> tuple[Path, dict[str, Any], bytes]:
    configs = sorted(root.glob(f"*/jlens/{dataset}/config.yaml"))
    matches: list[tuple[Path, dict[str, Any], bytes]] = []
    for path in configs:
        raw = path.read_bytes()
        payload = yaml.safe_load(raw)
        if (
            isinstance(payload, dict)
            and isinstance(payload.get("hf_model_name"), str)
            and payload["hf_model_name"].casefold() == model_id.casefold()
        ):
            matches.append((path, payload, raw))
    if not matches:
        raise FileNotFoundError(f"Neuronpedia has no official Jacobian lens for {model_id!r}")
    if len(matches) != 1:
        raise ValueError(f"multiple Neuronpedia lens configs match {model_id!r}")
    return matches[0]


def _select_checkpoint(files: list[str], config_file: str) -> str:
    parent = str(Path(config_file).parent)
    candidates = sorted(
        filename
        for filename in files
        if str(Path(filename).parent) == parent
        and Path(filename).suffix == ".pt"
        and "jacobian_lens" in Path(filename).name
    )
    if not candidates:
        raise FileNotFoundError(f"no Jacobian-lens checkpoint beside {config_file}")
    n1000 = [path for path in candidates if "_n1000.pt" in path]
    plain = [path for path in candidates if path.endswith("_jacobian_lens.pt")]
    return (n1000 or plain or candidates)[0]


def _load_external_checkpoint(path: str) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {
        "J",
        "n_prompts",
        "source_layers",
        "d_model",
    }:
        raise ValueError("external Jacobian-lens checkpoint has an invalid schema")
    return payload


def _validated_external_lens(
    payload: dict[str, Any],
    *,
    expected_d_model: int | None = None,
    expected_layers: int | None = None,
) -> JacobianLens:
    jacobians = payload.get("J")
    layers = payload.get("source_layers")
    d_model = payload.get("d_model")
    n_prompts = payload.get("n_prompts")
    if (
        not isinstance(jacobians, dict)
        or not isinstance(layers, list)
        or not layers
        or layers != sorted(set(layers))
        or set(jacobians) != set(layers)
        or isinstance(d_model, bool)
        or not isinstance(d_model, int)
        or d_model <= 0
        or isinstance(n_prompts, bool)
        or not isinstance(n_prompts, int)
        or n_prompts <= 0
        or (expected_d_model is not None and d_model != expected_d_model)
        or (
            expected_layers is not None
            and any(not isinstance(layer, int) or not 0 <= layer < expected_layers for layer in layers)
        )
    ):
        raise ValueError("external Jacobian-lens checkpoint metadata is incompatible")
    clean: dict[int, torch.Tensor] = {}
    for layer in layers:
        value = jacobians[layer]
        if (
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != (d_model, d_model)
            or not value.dtype.is_floating_point
            or not torch.isfinite(value).all()
        ):
            raise ValueError(f"external Jacobian-lens layer {layer} is invalid")
        clean[int(layer)] = value
    return JacobianLens(clean, n_prompts=n_prompts, d_model=d_model)


def fetch_neuronpedia_lens(
    model_id: str,
    *,
    repo_id: str = NEURONPEDIA_REPO,
    revision: str = "main",
    dataset: str = "Salesforce-wikitext",
    force: bool = False,
) -> ExternalLensBinding:
    """Fetch a supported official lens into the HF cache and bind it.

    No external payload is copied into ``SAKLAS_HOME``. The binding pins both
    the publisher repository and base-model repository to immutable commits.
    """
    from transformers import AutoConfig

    api = _hf_api()
    repo_info = api.model_info(repo_id, revision=revision)
    repo_revision = getattr(repo_info, "sha", None)
    if not isinstance(repo_revision, str) or not repo_revision:
        raise ValueError(f"could not resolve immutable revision for {repo_id}")
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    model_revision = _model_commit(config)
    if model_revision is None:
        raise ValueError(f"could not resolve immutable Hugging Face model revision for {model_id}")
    d_model, n_layers = _config_dimensions(config)
    if d_model is None or n_layers is None:
        raise ValueError(f"could not resolve model dimensions for {model_id}")

    root = Path(
        _snapshot_download(
            repo_id,
            revision=repo_revision,
            allow_patterns=[f"*/jlens/{dataset}/config.yaml"],
        )
    )
    config_path, fit_config, config_raw = _match_official_config(
        root,
        model_id,
        dataset,
    )
    config_file = config_path.relative_to(root).as_posix()
    files = list(api.list_repo_files(repo_id, revision=repo_revision))
    checkpoint = _select_checkpoint(files, config_file)
    checkpoint_path = _hf_hub_download(
        repo_id,
        checkpoint,
        revision=repo_revision,
    )
    lens = _validated_external_lens(
        _load_external_checkpoint(checkpoint_path),
        expected_d_model=d_model,
        expected_layers=n_layers,
    )
    config_model = fit_config.get("hf_model_name")
    if not isinstance(config_model, str) or config_model.casefold() != model_id.casefold():
        raise ValueError("Neuronpedia config model does not match requested model")
    fit_raw = fit_config.get("fit")
    fit: Mapping[str, Any] = fit_raw if isinstance(fit_raw, Mapping) else {}
    dataset_raw = fit_config.get("dataset")
    dataset_cfg: Mapping[str, Any] = dataset_raw if isinstance(dataset_raw, Mapping) else {}
    corpus = ":".join(str(dataset_cfg.get(key, "")) for key in ("name", "config", "split"))
    binding = ExternalLensBinding(
        name=NEURONPEDIA_BINDING,
        model_id=model_id,
        model_revision=model_revision,
        repo_id=repo_id,
        repo_revision=repo_revision,
        checkpoint=checkpoint,
        config_file=config_file,
        config_sha256=hashlib.sha256(config_raw).hexdigest(),
        corpus=corpus,
        n_prompts=lens.n_prompts,
        seq_len=int(fit.get("max_seq_len", 128)),
        dim_batch=int(fit.get("dim_batch", 1)),
        d_model=lens.d_model,
        source_layers=tuple(lens.source_layers),
    )
    path = lens_binding_path(model_id, binding.name)
    with artifact_lock(path):
        if path.exists() and not force:
            current = load_external_lens_binding(model_id, binding.name)
            if current == binding:
                set_active_lens_source(model_id, "huggingface", binding.name)
                return binding
        write_json_atomic(path, binding.to_json())
    set_active_lens_source(model_id, "huggingface", binding.name)
    return binding


def load_external_lens(
    model_id: str,
    name: str = NEURONPEDIA_BINDING,
) -> tuple[JacobianLens, dict[str, Any]] | None:
    """Load one pinned external lens from the provider cache, offline-only."""
    binding = load_external_lens_binding(model_id, name)
    if binding is None:
        return None
    try:
        config_path = _hf_hub_download(
            binding.repo_id,
            binding.config_file,
            revision=binding.repo_revision,
            local_files_only=True,
        )
        raw = Path(config_path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != binding.config_sha256:
            raise ValueError("cached Neuronpedia config digest changed")
        checkpoint_path = _hf_hub_download(
            binding.repo_id,
            binding.checkpoint,
            revision=binding.repo_revision,
            local_files_only=True,
        )
        lens = _validated_external_lens(
            _load_external_checkpoint(checkpoint_path),
            expected_d_model=binding.d_model,
        )
        if lens.n_prompts != binding.n_prompts or tuple(lens.source_layers) != binding.source_layers:
            raise ValueError("cached external lens does not match its binding")
    except (OSError, ValueError, RuntimeError, TypeError):
        return None
    return lens, external_lens_sidecar(binding)


def external_lens_sidecar(binding: ExternalLensBinding) -> dict[str, Any]:
    """Return canonical runtime metadata without loading provider weights."""
    return {
        "format_version": LENS_SOURCE_FORMAT_VERSION,
        "method": "anthropic_jlens",
        "n_prompts": binding.n_prompts,
        "d_model": binding.d_model,
        "source_layers": list(binding.source_layers),
        "dtype": "external",
        "corpus_spec": binding.corpus,
        "corpus_sha256": binding.config_sha256,
        "seq_len": binding.seq_len,
        "dim_batch": binding.dim_batch,
        "skip_first_positions": 0,
        "model_fingerprint": None,
        "model_source_fingerprint": binding.model_revision,
        "_source": {
            "kind": "huggingface",
            "name": binding.name,
            "provider": "neuronpedia",
            "repo_id": binding.repo_id,
            "repo_revision": binding.repo_revision,
            "checkpoint": binding.checkpoint,
            "model_id": binding.model_id,
            "model_revision": binding.model_revision,
        },
    }


def load_external_lens_sidecar(
    model_id: str,
    name: str = NEURONPEDIA_BINDING,
) -> dict[str, Any] | None:
    binding = load_external_lens_binding(model_id, name)
    return None if binding is None else external_lens_sidecar(binding)


def list_lens_sources(model_id: str) -> list[dict[str, Any]]:
    active = load_active_lens_source(model_id)
    rows: list[dict[str, Any]] = []
    local_manifest = local_lens_dir(model_id) / "manifest.json"
    if local_manifest.exists():
        rows.append(
            {
                "source": "local:default",
                "kind": "local",
                "name": "default",
                "active": active == _active_payload(model_id, "local", "default"),
                "path": str(local_manifest),
            }
        )
    for path in sorted(lens_bindings_dir(model_id).glob("*.json")):
        binding = load_external_lens_binding(model_id, path.stem)
        if binding is None:
            continue
        rows.append(
            {
                "source": binding.name,
                "kind": "huggingface",
                "name": binding.name,
                "provider": "neuronpedia",
                "repo_id": binding.repo_id,
                "repo_revision": binding.repo_revision,
                "checkpoint": binding.checkpoint,
                "active": active == _active_payload(model_id, "huggingface", binding.name),
                "path": str(path),
            }
        )
    return rows


def use_lens_source(model_id: str, source: str) -> Path:
    source = source.strip()
    if source.startswith("local:"):
        return set_active_lens_source(model_id, "local", source[6:])
    if source == "neuronpedia":
        return set_active_lens_source(
            model_id,
            "huggingface",
            NEURONPEDIA_BINDING,
        )
    raise ValueError("J-lens source must be local:NAME or neuronpedia")


def remove_external_lens_binding(
    model_id: str,
    name: str = NEURONPEDIA_BINDING,
) -> bool:
    """Forget an external binding without touching its provider cache."""
    path = lens_binding_path(model_id, name)
    active_path = lens_active_path(model_id)
    with artifact_lock(path), artifact_lock(active_path):
        active = load_active_lens_source(model_id)
        removed = path.exists()
        path.unlink(missing_ok=True)
        if active is not None and active["kind"] == "huggingface" and active["name"] == name:
            active_path.unlink(missing_ok=True)
        return removed
