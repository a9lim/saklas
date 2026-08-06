"""The per-model active-source selection shared by the lens and SAE families.

Both families are *source registries*: Saklas owns what it fits or trains and
stores it under ``SAKLAS_HOME``, external publishers keep their payloads in
their own caches and Saklas stores only a pinned binding, and one
``<family_root>/active.json`` names which of them the runtime should use.

That selection file is the same artifact in both families — a
``{format_version, model_id, kind, name}`` payload, written atomically under
the artifact lock, read back only when every field validates. This module owns
it once, parameterized by the family's format version, kind vocabulary, name
grammar, and existence precondition, so the two lifecycles cannot drift on
locking or validation discipline the way they did when each rolled its own.

What stays per-family: the payload *schemas* of the bindings the selection
points at (a J-lens binding vs an SAE runtime-metadata sidecar), their format
versions, and the loaders that turn a selection into a resident artifact.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
import json
from pathlib import Path
from typing import Any

from saklas.io.atomic import artifact_lock, write_json_atomic

_PAYLOAD_FIELDS = frozenset({"format_version", "model_id", "kind", "name"})


class ActiveSourceRegistry:
    """The ``active.json`` selection for one artifact family.

    ``label`` names the family in error messages.  ``kinds`` is the closed
    vocabulary a selection may carry.  ``validate_name(kind, name)`` raises on
    a name the family cannot address — a saklas-owned artifact validates
    against the artifact-name grammar, while an external provider's release id
    is provider-shaped and only has to be non-empty.  ``exists(root, kind,
    name)`` is the precondition: the selection must never point at a source
    that is not on disk, so a runtime reading it never has to second-guess it.
    """

    def __init__(
        self,
        *,
        label: str,
        format_version: int,
        kinds: Iterable[str],
        validate_name: Callable[[str, str], None],
        exists: Callable[[Path, str, str], bool] | None = None,
    ) -> None:
        self.label = label
        self.format_version = format_version
        self.kinds = frozenset(kinds)
        self._validate_name = validate_name
        self._exists = exists

    # -- paths + payload ---------------------------------------------------

    def path(self, root: Path) -> Path:
        """The selection file under a family's per-model runtime directory."""
        return root / "active.json"

    def payload(self, model_id: str, kind: str, name: str) -> dict[str, Any]:
        """The exact on-disk selection payload — also the equality probe.

        Comparing a loaded selection against this is how a lifecycle call
        decides whether the source it is removing is the active one, without
        restating the schema at every such site.
        """
        return {
            "format_version": self.format_version,
            "model_id": model_id,
            "kind": kind,
            "name": name,
        }

    # -- read / write ------------------------------------------------------

    def read(self, root: Path, model_id: str) -> dict[str, Any] | None:
        """Return the validated selection, or ``None``.

        Any deviation — unreadable file, non-current format version, foreign
        model id, unknown kind, ungrammatical name — reads as "no selection"
        rather than raising: a stale or hand-edited selection must not stop a
        model from starting.
        """
        path = self.path(root)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            return None
        if (
            not isinstance(data, dict)
            or set(data) != _PAYLOAD_FIELDS
            or data.get("format_version") != self.format_version
            or data.get("model_id") != model_id
            or data.get("kind") not in self.kinds
            or not isinstance(data.get("name"), str)
            or not data["name"]
        ):
            return None
        try:
            self._validate_name(data["kind"], data["name"])
        except ValueError:
            return None
        return data

    def write(self, root: Path, model_id: str, kind: str, name: str) -> Path:
        """Select a source, validating kind, name, and on-disk existence."""
        if kind not in self.kinds:
            raise ValueError(f"unknown {self.label} source kind {kind!r}")
        self._validate_name(kind, name)
        if self._exists is not None and not self._exists(root, kind, name):
            raise FileNotFoundError(
                f"{self.label} source {kind}:{name} is not available"
            )
        path = self.path(root)
        with artifact_lock(path):
            write_json_atomic(path, self.payload(model_id, kind, name))
        return path

    def clear_if_active(
        self, root: Path, model_id: str, kind: str, name: str,
    ) -> bool:
        """Unpublish the selection when it names exactly this source.

        Called while a lifecycle operation removes a source, so the selection
        can never outlive what it points at.  Returns whether it cleared.
        """
        path = self.path(root)
        with artifact_lock(path):
            if self.read(root, model_id) != self.payload(model_id, kind, name):
                return False
            path.unlink(missing_ok=True)
            return True
