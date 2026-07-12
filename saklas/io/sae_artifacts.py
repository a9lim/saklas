"""Saklas-owned sparse-autoencoder artifacts.

Provider-owned SAEs stay in SAELens/Hugging Face caches. This module owns only
SAEs trained by Saklas itself under ``models/<safe>/sae/local/<name>``.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping

from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch

from saklas.io.atomic import artifact_lock, fsync_directory, write_json_atomic
from saklas.io.paths import ensure_within, model_dir
from saklas.io.sae import load_active_sae_source, sae_active_path, set_active_sae_source

LOCAL_SAE_FORMAT_VERSION = 1
_LOCAL_NAME_RE = re.compile(r"^[a-z][a-z0-9._-]{0,63}$")
_MANIFEST_FIELDS = {
    "format_version",
    "kind",
    "name",
    "release",
    "model_id",
    "model_fingerprint",
    "model_source_fingerprint",
    "layer",
    "d_model",
    "d_sae",
    "activation",
    "tensor_file",
    "tensor_sha256",
    "corpus_spec",
    "corpus_sha256",
    "tokens_trained",
    "seq_len",
    "batch_size",
    "learning_rate",
    "l1_coefficient",
    "dead_feature_threshold",
}


def normalize_local_sae_name(value: str) -> str:
    value = value.strip()
    if value.startswith("local:"):
        value = value[6:]
    if _LOCAL_NAME_RE.fullmatch(value) is None:
        raise ValueError("local SAE name must match [a-z][a-z0-9._-]{0,63}")
    return value


def local_sae_release(name: str) -> str:
    return f"local:{normalize_local_sae_name(name)}"


def local_sae_root(model_id: str) -> Path:
    return model_dir(model_id) / "sae" / "local"


def local_sae_dir(model_id: str, name: str) -> Path:
    return ensure_within(local_sae_root(model_id), normalize_local_sae_name(name))


def local_sae_manifest_path(model_id: str, name: str) -> Path:
    return local_sae_dir(model_id, name) / "manifest.json"


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_manifest(
    payload: Any,
    model_id: str,
    name: str,
) -> dict[str, Any]:
    name = normalize_local_sae_name(name)
    if not isinstance(payload, dict) or set(payload) != _MANIFEST_FIELDS:
        raise ValueError("local SAE manifest has an invalid schema")
    if (
        payload.get("format_version") != LOCAL_SAE_FORMAT_VERSION
        or payload.get("kind") != "local"
        or payload.get("name") != name
        or payload.get("release") != local_sae_release(name)
        or payload.get("model_id") != model_id
        or payload.get("activation") != "relu"
    ):
        raise ValueError("local SAE manifest has invalid identity metadata")
    for key in ("model_fingerprint", "model_source_fingerprint", "corpus_spec", "corpus_sha256"):
        value = payload.get(key)
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError(f"local SAE manifest has invalid {key}")
    for key in (
        "layer",
        "d_model",
        "d_sae",
        "tokens_trained",
        "seq_len",
        "batch_size",
    ):
        value = payload.get(key)
        minimum = 0 if key == "layer" else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"local SAE manifest has invalid {key}")
    for key in ("learning_rate", "l1_coefficient", "dead_feature_threshold"):
        value = payload.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not torch.isfinite(torch.tensor(float(value))).item()
            or float(value) < 0
        ):
            raise ValueError(f"local SAE manifest has invalid {key}")
    tensor_file = payload.get("tensor_file")
    if (
        not isinstance(tensor_file, str)
        or Path(tensor_file).name != tensor_file
        or not tensor_file.endswith(".safetensors")
    ):
        raise ValueError("local SAE manifest has invalid tensor_file")
    digest = payload.get("tensor_sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError("local SAE manifest has invalid tensor digest")
    return payload


def load_local_sae_manifest(model_id: str, name: str) -> dict[str, Any] | None:
    path = local_sae_manifest_path(model_id, name)
    if not path.exists():
        return None
    try:
        payload = _validate_manifest(json.loads(path.read_text()), model_id, name)
        tensor_path = path.parent / payload["tensor_file"]
        if not tensor_path.exists() or _hash_file(tensor_path) != payload["tensor_sha256"]:
            return None
        expected = {
            "W_enc": (payload["d_model"], payload["d_sae"]),
            "W_dec": (payload["d_sae"], payload["d_model"]),
            "b_enc": (payload["d_sae"],),
            "b_dec": (payload["d_model"],),
        }
        with safe_open(str(tensor_path), framework="pt", device="cpu") as tensors:
            if set(tensors.keys()) != set(expected):
                return None
            for key, shape in expected.items():
                view = tensors.get_slice(key)
                if tuple(view.get_shape()) != shape or view.get_dtype() != "F32":
                    return None
        return payload
    except (OSError, ValueError, TypeError, KeyError):
        return None


def load_local_sae_tensors(
    model_id: str,
    name: str,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]] | None:
    manifest = load_local_sae_manifest(model_id, name)
    if manifest is None:
        return None
    tensors = load_file(str(local_sae_dir(model_id, name) / manifest["tensor_file"]))
    if any(not torch.isfinite(value).all() for value in tensors.values()):
        return None
    return tensors, manifest


def save_local_sae(
    model_id: str,
    name: str,
    tensors: Mapping[str, torch.Tensor],
    *,
    model_fingerprint: str | None,
    model_source_fingerprint: str | None,
    layer: int,
    corpus_spec: str,
    corpus_sha256: str,
    tokens_trained: int,
    seq_len: int,
    batch_size: int,
    learning_rate: float,
    l1_coefficient: float,
    dead_feature_threshold: float,
    force: bool = False,
) -> Path:
    name = normalize_local_sae_name(name)
    required = {"W_enc", "W_dec", "b_enc", "b_dec"}
    if set(tensors) != required:
        raise ValueError(f"local SAE tensors must be exactly {sorted(required)}")
    w_enc = tensors["W_enc"]
    w_dec = tensors["W_dec"]
    if w_enc.ndim != 2 or w_dec.ndim != 2 or tuple(w_dec.shape) != tuple(reversed(w_enc.shape)):
        raise ValueError("local SAE encoder/decoder shapes do not agree")
    d_model, d_sae = (int(w_enc.shape[0]), int(w_enc.shape[1]))
    expected = {
        "W_enc": (d_model, d_sae),
        "W_dec": (d_sae, d_model),
        "b_enc": (d_sae,),
        "b_dec": (d_model,),
    }
    clean: dict[str, torch.Tensor] = {}
    for key, shape in expected.items():
        value = tensors[key].detach().to(device="cpu", dtype=torch.float32).contiguous()
        if tuple(value.shape) != shape or not torch.isfinite(value).all():
            raise ValueError(f"local SAE tensor {key} is invalid")
        clean[key] = value

    root = local_sae_dir(model_id, name)
    manifest_path = root / "manifest.json"
    with artifact_lock(manifest_path):
        if manifest_path.exists() and not force:
            raise FileExistsError(f"local SAE {name!r} already exists; pass --force to replace it")
        root.mkdir(parents=True, exist_ok=True)
        generation = f"layer-{int(layer)}.gen-{os.urandom(12).hex()}.safetensors"
        tensor_path = root / generation
        tmp = tensor_path.with_suffix(tensor_path.suffix + ".tmp")
        save_file(clean, str(tmp))
        with open(tmp, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp, tensor_path)
        digest = _hash_file(tensor_path)
        payload = {
            "format_version": LOCAL_SAE_FORMAT_VERSION,
            "kind": "local",
            "name": name,
            "release": local_sae_release(name),
            "model_id": model_id,
            "model_fingerprint": model_fingerprint,
            "model_source_fingerprint": model_source_fingerprint,
            "layer": int(layer),
            "d_model": d_model,
            "d_sae": d_sae,
            "activation": "relu",
            "tensor_file": generation,
            "tensor_sha256": digest,
            "corpus_spec": corpus_spec,
            "corpus_sha256": corpus_sha256,
            "tokens_trained": int(tokens_trained),
            "seq_len": int(seq_len),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "l1_coefficient": float(l1_coefficient),
            "dead_feature_threshold": float(dead_feature_threshold),
        }
        _validate_manifest(payload, model_id, name)
        write_json_atomic(manifest_path, payload)
        for old in root.glob("*.safetensors"):
            if old != tensor_path:
                old.unlink(missing_ok=True)
        fsync_directory(root)
    set_active_sae_source(model_id, "local", name)
    return manifest_path


def remove_local_sae(model_id: str, name: str) -> bool:
    name = normalize_local_sae_name(name)
    root = local_sae_dir(model_id, name)
    manifest = root / "manifest.json"
    with artifact_lock(manifest):
        removed = root.exists()
        if root.exists():
            for path in sorted(root.glob("*")):
                if path.is_file():
                    path.unlink(missing_ok=True)
            try:
                root.rmdir()
            except OSError:
                pass
        active = load_active_sae_source(model_id)
        if active == {
            "format_version": 1,
            "model_id": model_id,
            "kind": "local",
            "name": name,
        }:
            sae_active_path(model_id).unlink(missing_ok=True)
        return removed


def list_local_saes(model_id: str) -> list[dict[str, Any]]:
    active = load_active_sae_source(model_id)
    rows: list[dict[str, Any]] = []
    for root in sorted(local_sae_root(model_id).glob("*")):
        if not root.is_dir():
            continue
        manifest = load_local_sae_manifest(model_id, root.name)
        if manifest is None:
            continue
        rows.append(
            {
                "source": f"local:{root.name}",
                "kind": "local",
                "name": root.name,
                "active": bool(active is not None and active["kind"] == "local" and active["name"] == root.name),
                "path": str(root / "manifest.json"),
                "layer": manifest["layer"],
                "features": manifest["d_sae"],
            }
        )
    return rows
