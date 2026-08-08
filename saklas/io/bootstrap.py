"""The one entry point that materializes every bundled artifact kind.

Saklas ships two artifact kinds inside the wheel — templates under
``saklas/data/templates/`` and manifolds under ``saklas/data/manifolds/`` —
which copy-on-miss into ``~/.saklas/`` the first time a process needs them.
Both materializers are process-scoped per ``SAKLAS_HOME``, and the **order
between them is load-bearing**: a bundled manifold may ``template_ref`` a
bundled ``default/<name>`` template, and its fit resolves that ref (a hard
``TemplateNotFoundError`` otherwise, with ``nodes_sha256`` degrading to the
bare ref string).

That invariant used to be re-derived by every bootstrap site from a copied
comment. :func:`materialize_bundled_artifacts` owns it instead, so a new
bootstrap site cannot get the order wrong.

Also home to the canonical-JSON drift hash both materializers use to decide
whether the shipped bundle has moved under an on-disk copy.
"""
from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json


def canonical_payload_sha256(
    data: bytes, *, strip_keys: Sequence[str] = (),
) -> str:
    """Bundle-drift sha256 of a JSON manifest payload.

    Hashes the canonical-JSON form (sorted keys, no whitespace) so
    cosmetic-only differences — key order, indent, trailing newline — compare
    equal.  ``strip_keys`` drops top-level keys that carry *local* transaction
    state rather than bundle content: a manifold's ``files`` accumulates
    per-model fit proofs locally, so hashing it would misread every fit as a
    bundle update and the refresh would clobber the proofs, orphaning the
    tensors (the strict loader refuses a tensor with no proof).

    Falls back to a raw sha256 when the bytes don't parse as JSON, so
    unparseable on-disk content reads as "user edited" rather than getting
    silently overwritten.
    """
    try:
        parsed = json.loads(data)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return hashlib.sha256(data).hexdigest()
    if strip_keys and isinstance(parsed, dict):
        for key in strip_keys:
            parsed.pop(key, None)
    return hashlib.sha256(
        json.dumps(parsed, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def materialize_bundled_artifacts() -> None:
    """Materialize bundled templates, then bundled manifolds.

    The single bootstrap call every entry point should make.  Templates go
    first because a bundled manifold may reference one (see the module note).
    ``materialize_bundled_manifolds`` drops the selector index itself, so a
    bare reference to a just-dropped manifold's node label resolves in the
    same process.

    Both materializers are per-home process-scoped no-ops after their first
    call, so repeated bootstrap is cheap.
    """
    from saklas.io.manifolds import materialize_bundled_manifolds
    from saklas.io.templates import materialize_bundled_templates

    materialize_bundled_templates()
    materialize_bundled_manifolds()
