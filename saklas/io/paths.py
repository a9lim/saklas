"""Filesystem path helpers for the ~/.saklas/ tree.

All paths resolve through saklas_home(), which honors the SAKLAS_HOME
environment variable for testing and non-default installs.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

# Variant filename conventions:
#   raw (DiM, default)  -> ``<safe_model_id>.safetensors``
#   raw (PCA, legacy)   -> ``<safe_model_id>_pca.safetensors``
#   SAE-DiM             -> ``<safe_model_id>_sae-<release>.safetensors``
#   SAE-PCA (legacy)    -> ``<safe_model_id>_sae-<release>_pca.safetensors``
#   transferred (v1.6)  -> ``<safe_model_id>_from-<safe_src>.safetensors``
#   role                -> ``<safe_model_id>_role-<name>.safetensors``
#   role-PCA (legacy)   -> ``<safe_model_id>_role-<name>_pca.safetensors``
#
# The literals ``_sae-``, ``_from-``, and ``_role-`` are the *kind*
# separators — no HF model id slug contains any of them, and the right-
# hand-side slugs (release strings, safe-model-ids, role names) follow
# the same ``[a-z0-9._-]`` discipline so the parse is unambiguous.
# ``_pca`` is a *method* suffix: applied last, stripped first on parse,
# never composed with ``_from-`` (transfer preserves the source method).
# Default method is DiM (no suffix); ``_pca`` opts into the legacy
# contrastive-PCA tensors that coexist with the canonical DiM file.
# Kind suffixes are mutually exclusive in v1 — a tensor is at most one
# of {SAE, transferred, role}.
_VARIANT_SEP_SAE = "_sae-"
_VARIANT_SEP_FROM = "_from-"
_VARIANT_SEP_ROLE = "_role-"
_METHOD_SUFFIX_PCA = "_pca"
_VARIANT_SEPARATORS: tuple[tuple[str, str], ...] = (
    (_VARIANT_SEP_SAE, "sae"),
    (_VARIANT_SEP_FROM, "from"),
    (_VARIANT_SEP_ROLE, "role"),
)
# Back-compat: the old single-separator alias many callers already
# imported.  Kept identical to the SAE form because that's what every
# external caller meant when they reached for it pre-1.6.
_VARIANT_SEP = _VARIANT_SEP_SAE
_UNSAFE_VARIANT_CHARS = re.compile(r"[^a-z0-9._-]+")

# Recognised extraction methods.  ``"dim"`` is the canonical default
# (difference-of-means, Im & Li 2025); ``"pca"`` is the legacy
# contrastive-PCA path retained behind the ``--method pca`` flag and
# the ``:pca`` selector variant.
_KNOWN_METHODS: frozenset[str] = frozenset({"dim", "pca"})


def _method_suffix(method: str) -> str:
    """Filename suffix for an extraction method.  ``"dim"`` (default) returns
    the empty string; ``"pca"`` returns ``_pca``.
    """
    if method == "dim":
        return ""
    if method == "pca":
        return _METHOD_SUFFIX_PCA
    raise ValueError(
        f"unknown extraction method {method!r} (expected one of "
        f"{sorted(_KNOWN_METHODS)})"
    )


def saklas_home() -> Path:
    """Return the root ~/.saklas/ directory. Honors $SAKLAS_HOME override."""
    override = os.environ.get("SAKLAS_HOME")
    if override:
        return Path(override)
    return Path.home() / ".saklas"


def vectors_dir() -> Path:
    return saklas_home() / "vectors"


def models_dir() -> Path:
    return saklas_home() / "models"


def manifolds_dir() -> Path:
    """Root of the manifold-steering artifact tree.

    Parallel to :func:`vectors_dir` — a manifold is its own artifact kind
    (labeled nodes placed on an n-dimensional domain), not a concept
    folder, so it lives under its own root.
    """
    return saklas_home() / "manifolds"


def manifold_dir(namespace: str, name: str) -> Path:
    return manifolds_dir() / namespace / name


def neutral_statements_path() -> Path:
    return saklas_home() / "neutral_statements.json"


def safe_model_id(model_id: str) -> str:
    """Flatten an HF-style model ID for filesystem use: '/' -> '__'."""
    return model_id.replace("/", "__")


def concept_dir(namespace: str, concept: str) -> Path:
    return vectors_dir() / namespace / concept


def model_dir(model_id: str) -> Path:
    return models_dir() / safe_model_id(model_id)


def safe_variant_suffix(release: str | None) -> str:
    """Render the SAE filename suffix.  ``None``/``""`` = raw (no suffix).

    Kept for back-compat with callers that pre-date the v1.6 transfer
    variant.  New code should prefer :func:`safe_sae_suffix` for SAE or
    :func:`safe_from_suffix` for transferred profiles.
    """
    if not release:
        return ""
    slug = _UNSAFE_VARIANT_CHARS.sub("_", release.lower())
    return f"{_VARIANT_SEP_SAE}{slug}"


def safe_sae_suffix(release: str | None) -> str:
    """Filename suffix for an SAE variant.  ``None``/``""`` = raw."""
    return safe_variant_suffix(release)


def safe_from_suffix(source_safe_id: str | None) -> str:
    """Filename suffix for a transferred-profile variant.

    Input is a *safe model id* (already passed through :func:`safe_model_id`)
    so the slug is byte-stable across operating systems.  Returns the
    empty string for ``None`` / empty (no transfer = raw).
    """
    if not source_safe_id:
        return ""
    slug = _UNSAFE_VARIANT_CHARS.sub("_", source_safe_id.lower())
    return f"{_VARIANT_SEP_FROM}{slug}"


def safe_role_suffix(role_name: str | None) -> str:
    """Filename suffix for a role variant.  ``None``/``""`` = raw (no suffix).

    The role name is slugified with the same ``[a-z0-9._-]`` discipline
    as :func:`safe_sae_suffix` — lowercased, unsafe characters collapsed
    to ``_`` — so the parse is unambiguous against neighbouring kind
    separators.
    """
    if not role_name:
        return ""
    slug = _UNSAFE_VARIANT_CHARS.sub("_", role_name.lower())
    return f"{_VARIANT_SEP_ROLE}{slug}"


def tensor_filename(
    model_id: str,
    *,
    release: str | None = None,
    transferred_from: str | None = None,
    method: str = "dim",
    role: str | None = None,
) -> str:
    """Construct the canonical tensor filename.

    At most one of ``release``, ``transferred_from``, and ``role`` may be
    set — composed kind variants (SAE-on-transferred, role-on-SAE, etc.)
    are not supported in v1.  ``transferred_from`` accepts either an HF
    model id (``"google/gemma-3-4b-it"``) or its safe form
    (``"google__gemma-3-4b-it"``); both flatten to the same slug.

    ``method`` controls the trailing method suffix:

      * ``"dim"`` (default, v2.1+) — no suffix.  Tensors at the canonical
        path were extracted via difference-of-means.
      * ``"pca"`` — legacy contrastive-PCA tensors land at the same path
        with a ``_pca`` suffix appended after any kind suffix.
        ``transferred_from`` rejects ``method="pca"`` because transfers
        preserve their source method (the source's sidecar carries the
        provenance string).  ``role`` composes with ``method="pca"``.

    ``role`` (optional) names a role/persona variant; the filename gets
    a ``_role-<safe_role>`` suffix, slugified like the SAE release slug.
    """
    if sum(bool(x) for x in (release, transferred_from, role)) > 1:
        raise ValueError(
            "tensor_filename: release, transferred_from, and role are "
            "mutually exclusive"
        )
    if transferred_from and method != "dim":
        raise ValueError(
            "tensor_filename: transferred_from preserves source method; "
            "explicit method= is not supported"
        )
    suffix = _method_suffix(method)
    if release:
        return (
            f"{safe_model_id(model_id)}{safe_sae_suffix(release)}{suffix}.safetensors"
        )
    if transferred_from:
        # Accept either form; ``safe_model_id`` is idempotent on
        # already-safe ids (no '/' to replace), so callers can pass
        # whichever they have.
        src = safe_model_id(transferred_from)
        return f"{safe_model_id(model_id)}{safe_from_suffix(src)}.safetensors"
    if role:
        return (
            f"{safe_model_id(model_id)}{safe_role_suffix(role)}{suffix}.safetensors"
        )
    return f"{safe_model_id(model_id)}{suffix}.safetensors"


def sidecar_filename(
    model_id: str,
    *,
    release: str | None = None,
    transferred_from: str | None = None,
    method: str = "dim",
    role: str | None = None,
) -> str:
    """Sidecar JSON partner for a tensor filename."""
    if sum(bool(x) for x in (release, transferred_from, role)) > 1:
        raise ValueError(
            "sidecar_filename: release, transferred_from, and role are "
            "mutually exclusive"
        )
    if transferred_from and method != "dim":
        raise ValueError(
            "sidecar_filename: transferred_from preserves source method; "
            "explicit method= is not supported"
        )
    suffix = _method_suffix(method)
    if release:
        return f"{safe_model_id(model_id)}{safe_sae_suffix(release)}{suffix}.json"
    if transferred_from:
        src = safe_model_id(transferred_from)
        return f"{safe_model_id(model_id)}{safe_from_suffix(src)}.json"
    if role:
        return f"{safe_model_id(model_id)}{safe_role_suffix(role)}{suffix}.json"
    return f"{safe_model_id(model_id)}{suffix}.json"


def parse_tensor_filename(
    filename: str,
) -> tuple[str, str | None] | None:
    """Reverse of :func:`tensor_filename`. Returns ``(safe_model_id, variant)``.

    ``variant`` is one of:
      * ``None`` — raw DiM tensor (no separator, no method suffix).
      * ``"pca"`` — raw PCA tensor (legacy method suffix only).
      * ``"sae-<release>"`` — SAE-DiM variant.
      * ``"sae-<release>-pca"`` — SAE-PCA variant (legacy).
      * ``"from-<safe_src>"`` — transferred-from variant (method-agnostic;
        transfers preserve source method).
      * ``"role-<name>"`` — role variant (DiM).
      * ``"role-<name>-pca"`` — role variant with legacy PCA method.

    The variant string carries its kind / method tags so callers can
    dispatch without re-parsing.  Returns ``None`` for filenames that
    aren't ``.safetensors``.
    """
    if not filename.endswith(".safetensors"):
        return None
    stem = filename[: -len(".safetensors")]
    # Method suffix is parsed first (right-to-left): ``_pca`` is applied
    # last on construction, so we strip it before looking for the kind
    # separator.  ``_pca`` cannot legally appear inside any kind slug —
    # ``_UNSAFE_VARIANT_CHARS`` keeps slugs to ``[a-z0-9._-]`` and the
    # leading ``_`` rules out collision.
    is_pca = stem.endswith(_METHOD_SUFFIX_PCA)
    if is_pca:
        stem = stem[: -len(_METHOD_SUFFIX_PCA)]
    for sep, kind in _VARIANT_SEPARATORS:
        if sep in stem:
            model, value = stem.split(sep, 1)
            if kind == "from" and is_pca:
                # Transferred-from never carries an explicit method —
                # if a stray ``_pca`` ends up in the filename, treat it
                # as part of the source slug to keep the round-trip
                # idempotent.  No production path produces this.
                value = f"{value}{_METHOD_SUFFIX_PCA}"
            tag = f"{kind}-{value}"
            return model, f"{tag}-pca" if is_pca and kind != "from" else tag
    return stem, ("pca" if is_pca else None)
