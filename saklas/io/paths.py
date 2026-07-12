"""Filesystem path helpers for the ~/.saklas/ tree.

All paths resolve through saklas_home(), which honors the SAKLAS_HOME
environment variable for testing and non-default installs.
"""
from __future__ import annotations

import base64
import os
import re
from pathlib import Path

# Variant filename conventions:
#   raw (DiM)           -> ``<safe_model_id>.safetensors``
#   SAE-DiM             -> ``<safe_model_id>_sae-<release>.safetensors``
#   transferred (v1.6)  -> ``<safe_model_id>_from-<safe_src>.safetensors``
#
# The literals ``_sae-`` and ``_from-`` are the *kind*
# separators. Model ids and right-hand-side slugs can legally contain those
# literals, so filename components escape them before concatenation and the
# parser splits at the earliest unescaped kind separator.
# Difference-of-means is the only extraction method (4.0), so the raw
# tensor carries no method suffix.  Kind suffixes are mutually exclusive
# — a tensor is at most one of {SAE, transferred}.
_VARIANT_SEP_SAE = "_sae-"
_VARIANT_SEP_FROM = "_from-"
_VARIANT_SEPARATORS: tuple[tuple[str, str], ...] = (
    (_VARIANT_SEP_SAE, "sae"),
    (_VARIANT_SEP_FROM, "from"),
)
_UNSAFE_VARIANT_CHARS = re.compile(r"[^a-z0-9._-]+")


def _encode_tensor_component(value: str) -> str:
    """Escape reserved filename separators inside one model/variant slug."""
    value = value.replace("%", "%25")
    for separator, _kind in _VARIANT_SEPARATORS:
        value = value.replace(separator, f"%5F{separator[1:]}")
    return value


def _decode_tensor_component(value: str) -> str:
    """Reverse :func:`_encode_tensor_component`."""
    for separator, _kind in _VARIANT_SEPARATORS:
        value = value.replace(f"%5F{separator[1:]}", separator)
    return value.replace("%25", "%")

# The trailing ``:<variant>`` scheme understood across selectors / runners.
# One source of truth for the variant suffix grammar (this module owns the
# tensor-filename variant scheme); ``io.selectors`` and ``cli.runners`` import it.
VARIANT_SUFFIX_RE = re.compile(
    r"^(raw|sae(?:-[a-z0-9._-]+)?|role(?:-[a-z0-9._-]+)?|from(?:-[a-z0-9._-]+)?)$"
)

def saklas_home() -> Path:
    """Return the root ~/.saklas/ directory. Honors $SAKLAS_HOME override."""
    override = os.environ.get("SAKLAS_HOME")
    if override:
        return Path(override)
    return Path.home() / ".saklas"


def models_dir() -> Path:
    return saklas_home() / "models"


def manifolds_dir() -> Path:
    """Root of the manifold-steering artifact tree.

    A manifold is a labeled-node artifact placed on an n-dimensional domain.
    """
    return saklas_home() / "manifolds"


def templates_dir() -> Path:
    """Root of the templated-completion artifact tree.

    A template is a first-class artifact (peer to a manifold): a slot, a set
    of candidate values, and one or more multi-turn contexts whose final
    assistant turn carries the slot. Two consumers read it — the manifold fit
    (pools the slot-filled assistant centroids into nodes) and the completion
    scorer (the restricted-choice logprob distribution over the values).
    """
    return saklas_home() / "templates"


def ensure_within(root: Path, *parts: str) -> Path:
    """Join ``parts`` onto ``root`` and verify the result stays inside ``root``.

    Path-traversal barrier for user-supplied path components — namespaces,
    concept / manifold names, model ids, and manifest-relative filenames all
    reach the filesystem layer from HTTP request bodies, CLI args, and
    downloaded ``pack.json`` / ``manifold.json`` manifests. A ``..`` segment
    or an absolute component must not be allowed to escape the ``~/.saklas/``
    subtree.

    Normalizes the joined path (collapsing ``..`` segments) and rejects a
    result that does not stay under ``root`` — the normalize-then-prefix-check
    idiom. Returns the normalized in-bounds path; raises :class:`ValueError`
    otherwise. ``NAME_REGEX`` / ``_LABEL_REGEX`` already reject traversal
    syntax upstream, but this is the defense-in-depth barrier at the boundary
    where the string actually becomes a path.
    """
    root_norm = os.path.normpath(os.fspath(root))
    candidate = os.path.normpath(os.path.join(root_norm, *parts))
    if candidate != root_norm and not candidate.startswith(root_norm + os.sep):
        raise ValueError(f"unsafe path component {parts!r} escapes {root_norm!r}")
    return Path(candidate)


def manifold_dir(namespace: str, name: str) -> Path:
    return ensure_within(manifolds_dir(), namespace, name)


def neutral_statements_path() -> Path:
    return saklas_home() / "neutral_statements.json"


def baseline_prompts_path() -> Path:
    """User-override path for the shared A2 baseline user prompts.

    Conversational extraction generates each node's corpus as responses to
    this fixed prompt set; a node corpus aligns ``response[i] -> prompt[i % k]``.
    Falls back to the bundled ``saklas/data/baseline_prompts.json`` when absent.
    """
    return saklas_home() / "baseline_prompts.json"


def safe_model_id(model_id: str) -> str:
    """Encode any HF or local model id as one reversible filename component."""
    if not model_id:
        raise ValueError("model id must not be empty")
    payload = base64.urlsafe_b64encode(model_id.encode("utf-8")).decode("ascii")
    return "_z" + payload.rstrip("=")


def unsafe_model_id(safe_id: str) -> str:
    """Reverse :func:`safe_model_id`, rejecting malformed escape sequences."""
    if not safe_id.startswith("_z"):
        raise ValueError(f"invalid safe model id {safe_id!r}")
    encoded = safe_id[2:]
    encoded += "=" * (-len(encoded) % 4)
    try:
        model_id = base64.b64decode(
            encoded, altchars=b"-_", validate=True,
        ).decode("utf-8")
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError(f"invalid encoded safe model id {safe_id!r}") from exc
    if not model_id or safe_model_id(model_id) != safe_id:
        raise ValueError(f"invalid encoded safe model id {safe_id!r}")
    return model_id


def model_dir(model_id: str) -> Path:
    return ensure_within(models_dir(), safe_model_id(model_id))


def safe_sae_suffix(release: str | None) -> str:
    """Filename suffix for an SAE variant.  ``None``/``""`` = raw (no suffix)."""
    if not release:
        return ""
    slug = _encode_tensor_component(
        _UNSAFE_VARIANT_CHARS.sub("_", release.lower()),
    )
    return f"{_VARIANT_SEP_SAE}{slug}"


def safe_from_suffix(source_safe_id: str | None) -> str:
    """Filename suffix for a transferred-profile variant.

    Input is a *safe model id* (already passed through :func:`safe_model_id`)
    so the slug is byte-stable across operating systems.  Returns the
    empty string for ``None`` / empty (no transfer = raw).
    """
    if not source_safe_id:
        return ""
    slug = _encode_tensor_component(
        _UNSAFE_VARIANT_CHARS.sub("_", source_safe_id.lower()),
    )
    return f"{_VARIANT_SEP_FROM}{slug}"


def tensor_filename(
    model_id: str,
    *,
    release: str | None = None,
    transferred_from: str | None = None,
    model_id_is_safe: bool = False,
    transferred_from_is_safe: bool = False,
) -> str:
    """Construct the canonical tensor filename.

    At most one of ``release`` and ``transferred_from`` may be set — composed
    kind variants are not supported. ``transferred_from`` accepts an HF/local model id by
    default. Internal callers holding parsed safe ids must set
    ``transferred_from_is_safe=True``; the target equivalent is
    ``model_id_is_safe=True``.

    The raw tensor at the canonical path is difference-of-means (the only
    extraction method as of 4.0), so it carries no method suffix.

    """
    if release and transferred_from:
        raise ValueError(
            "tensor_filename: release and transferred_from are mutually exclusive"
        )
    target_safe = model_id if model_id_is_safe else safe_model_id(model_id)
    target = _encode_tensor_component(target_safe)
    if release:
        return f"{target}{safe_sae_suffix(release)}.safetensors"
    if transferred_from:
        src = (
            transferred_from
            if transferred_from_is_safe else safe_model_id(transferred_from)
        )
        return f"{target}{safe_from_suffix(src)}.safetensors"
    return f"{target}.safetensors"


def sidecar_filename(
    model_id: str,
    *,
    release: str | None = None,
    transferred_from: str | None = None,
) -> str:
    """Sidecar JSON partner for a tensor filename."""
    if release and transferred_from:
        raise ValueError(
            "sidecar_filename: release and transferred_from are mutually exclusive"
        )
    target = _encode_tensor_component(safe_model_id(model_id))
    if release:
        return f"{target}{safe_sae_suffix(release)}.json"
    if transferred_from:
        src = safe_model_id(transferred_from)
        return f"{target}{safe_from_suffix(src)}.json"
    return f"{target}.json"


def parse_tensor_filename(
    filename: str,
) -> tuple[str, str | None] | None:
    """Reverse of :func:`tensor_filename`. Returns ``(safe_model_id, variant)``.

    ``variant`` is one of:
      * ``None`` — raw DiM tensor (no separator).
      * ``"sae-<release>"`` — SAE-DiM variant.
      * ``"from-<safe_src>"`` — transferred-from variant.
    The variant string carries its kind tag so callers can dispatch
    without re-parsing.  Returns ``None`` for filenames that aren't
    ``.safetensors``.
    """
    if not filename.endswith(".safetensors"):
        return None
    stem = filename[: -len(".safetensors")]
    matches = [
        (stem.find(separator), separator, kind)
        for separator, kind in _VARIANT_SEPARATORS
        if separator in stem
    ]
    if matches:
        index, separator, kind = min(matches, key=lambda item: item[0])
        model = _decode_tensor_component(stem[:index])
        value = _decode_tensor_component(stem[index + len(separator):])
        return model, f"{kind}-{value}"
    return _decode_tensor_component(stem), None
