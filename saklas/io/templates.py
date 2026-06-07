"""Standalone templated-completion artifact — ``~/.saklas/templates/<ns>/<name>/``.

A *template* is a first-class artifact (peer to a manifold): a ``slot`` token, a
set of candidate ``values``, and one or more multi-turn ``contexts`` whose final
assistant turn carries the slot. Two consumers read the same artifact:

* **The completion scorer** (``core/scoring.py``) — for each context, the
  restricted-choice logprob distribution over the values: ``P("one second" |
  context)`` vs ``P("one week" | context)``, the model's belief about the slot.
* **The manifold fit** — pools each value's slot-filled assistant centroid into a
  discover node (a manifold ``template_ref``-erences the template), so the same
  template yields both a steering surface and a logit read.

On disk it is a single self-contained ``template.json`` (the node corpora are
derived deterministically from ``slot × values × contexts``, so there is nothing
to hash beyond the JSON itself):

```json
{
  "format_version": 1,
  "name": "weekday",
  "slot": "[DAY]",
  "values": ["Monday", "Tuesday", ...],
  "contexts": [
    {"turns": [{"role": "user", "content": "hi"},
               {"role": "assistant", "content": "hello!"},
               {"role": "user", "content": "what day is it?"}],
     "assistant": "today is [DAY]"}
  ]
}
```

**Invariant** (carried from the embedded-block predecessor, generalized to
multi-turn): the slot appears exactly once in each context's final ``assistant``
string and *never* in a history turn — history is shared common-mode across the
values, so the slot lives only where the value is read.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from saklas.core.errors import SaklasError
from saklas.io.atomic import write_json_atomic
from saklas.io.packs import NAME_REGEX
from saklas.io.paths import ensure_within, templates_dir

TEMPLATE_FORMAT_VERSION = 1

# Node label discipline — mirrors ``io.manifolds._LABEL_REGEX`` (a templated
# manifold's nodes are this template's slugged values, so the slug must be a
# valid node label).  Defined here, not imported, so the migration's
# ``manifolds -> templates`` import direction stays acyclic.
_LABEL_REGEX = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_VALID_ROLES = ("system", "user", "assistant")


class TemplateFormatError(ValueError, SaklasError):
    """A template artifact violates the on-disk format or an invariant."""


def _slug_value(value: str) -> str:
    """Slug a template value to a node label (mirrors ``io.manifolds._slug_value``).

    ``[^a-z0-9]+ -> _`` over the lowercased value, trimmed of edge underscores —
    ``"Monday" -> "monday"``, ``"New York" -> "new_york"`` — so the label a user
    steers (``weekday%monday``) matches the value typed.
    """
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _canonical_json(obj: Any) -> bytes:
    """Stable serialization for the staleness hash (sorted keys, compact)."""
    import json

    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


@dataclass(frozen=True)
class TemplateContext:
    """One multi-turn context: a conversation prefix + the slotted final turn.

    ``turns`` is the ordered history (``[{role, content}]``, last role ``user``);
    ``assistant`` is the final assistant turn that carries the slot exactly once.
    A single-turn template is the degenerate ``turns=[{user: ...}]`` case.
    """

    turns: tuple[dict[str, str], ...]
    assistant: str

    def messages(self) -> list[dict[str, str]]:
        """The history turns as a fresh mutable message list (scorer prefix)."""
        return [dict(t) for t in self.turns]

    def split_slot(self, slot: str) -> tuple[str, str]:
        """``(before, after)`` the slot in the assistant turn.

        The scorer conditions on ``before`` (the assistant text up to the slot)
        and scores the value tokens; ``after`` is to the right of the slot and so
        is irrelevant to a left-to-right read of the slot itself.
        """
        before, _, after = self.assistant.partition(slot)
        return before, after

    def fill(self, slot: str, value: str) -> str:
        """The full slotted assistant turn (manifold node corpus entry)."""
        return self.assistant.replace(slot, value)


@dataclass
class TemplateFolder:
    """In-memory view of a ``template.json`` artifact."""

    name: str
    slot: str
    values: tuple[str, ...]
    contexts: tuple[TemplateContext, ...]
    description: str = ""
    source: str = "local"
    tags: tuple[str, ...] = ()
    path: Path | None = None

    # -- derived views ----------------------------------------------------

    def node_labels(self) -> list[str]:
        """Slugged value labels, in value order (the manifold's node labels)."""
        return [_slug_value(v) for v in self.values]

    def node_corpora(self) -> dict[str, list[str]]:
        """``{label: [slotted assistant turn for each context]}`` for the fit.

        Corpus order tracks context order, so a manifold's capture aligns
        ``corpus[i]`` with ``contexts[i]`` (the multi-turn elicitation prefix) —
        the multi-turn generalization of the A2 ``response[i] -> prompt[i]``
        alignment.
        """
        return {
            _slug_value(value): [ctx.fill(self.slot, value) for ctx in self.contexts]
            for value in self.values
        }

    def score_inputs(self) -> list[dict[str, Any]]:
        """Per-context scorer inputs.

        Each entry: ``{messages, assistant_prefix, choices, labels, suffix}`` —
        ``messages`` is the context history, ``assistant_prefix`` the assistant
        text before the slot, ``choices`` the raw values (filled into the slot),
        ``labels`` their slugs. The scorer reports a distribution per entry
        (per-context, matching the artifact's design).
        """
        labels = self.node_labels()
        out: list[dict[str, Any]] = []
        for ctx in self.contexts:
            before, after = ctx.split_slot(self.slot)
            out.append({
                "messages": ctx.messages(),
                "assistant_prefix": before,
                "suffix": after,
                "choices": list(self.values),
                "labels": labels,
            })
        return out

    # -- persistence ------------------------------------------------------

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format_version": TEMPLATE_FORMAT_VERSION,
            "name": self.name,
            "slot": self.slot,
            "values": list(self.values),
            "contexts": [
                {"turns": [dict(t) for t in ctx.turns], "assistant": ctx.assistant}
                for ctx in self.contexts
            ],
        }
        if self.description:
            payload["description"] = self.description
        if self.source and self.source != "local":
            payload["source"] = self.source
        if self.tags:
            payload["tags"] = list(self.tags)
        return payload

    def sha256(self) -> str:
        """Content hash over ``slot × values × contexts`` (the staleness key).

        A templated manifold folds this into its ``nodes_sha256`` so a template
        edit (a value added, a context reworded) re-fits. Excludes display-only
        fields (``description``/``tags``) — they don't change the geometry.
        """
        core = {
            "format_version": TEMPLATE_FORMAT_VERSION,
            "slot": self.slot,
            "values": list(self.values),
            "contexts": [
                {"turns": [dict(t) for t in ctx.turns], "assistant": ctx.assistant}
                for ctx in self.contexts
            ],
        }
        return hashlib.sha256(_canonical_json(core)).hexdigest()

    def write(self, path: Path | None = None) -> Path:
        """Atomically write ``template.json``. Returns the folder path."""
        folder = path if path is not None else self.path
        if folder is None:
            raise ValueError("TemplateFolder.write: no path set")
        folder.mkdir(parents=True, exist_ok=True)
        write_json_atomic(folder / "template.json", self.to_payload())
        self.path = folder
        return folder

    @classmethod
    def load(cls, folder: Path) -> "TemplateFolder":
        """Load + validate a ``template.json`` from ``folder``."""
        import json

        manifest = folder / "template.json"
        if not manifest.exists():
            raise TemplateFormatError(f"no template.json under {folder}")
        with open(manifest) as f:
            data = json.load(f)
        return cls.from_payload(data, path=folder)

    @classmethod
    def from_payload(
        cls, data: Any, *, path: Path | None = None,
    ) -> "TemplateFolder":
        if not isinstance(data, dict):
            raise TemplateFormatError("template.json must be a JSON object")
        fmt = data.get("format_version")
        if fmt is not None and fmt > TEMPLATE_FORMAT_VERSION:
            raise TemplateFormatError(
                f"template format_version {fmt} is newer than this build "
                f"supports ({TEMPLATE_FORMAT_VERSION})"
            )
        name = data.get("name")
        if not isinstance(name, str) or not NAME_REGEX.match(name):
            raise TemplateFormatError(
                f"template 'name' {name!r} must match {NAME_REGEX.pattern}"
            )
        slot, values, contexts = _validate_body(name, data)
        tags = data.get("tags") or ()
        if tags and not all(isinstance(t, str) for t in tags):
            raise TemplateFormatError(f"template {name!r} 'tags' must be strings")
        return cls(
            name=name,
            slot=slot,
            values=tuple(values),
            contexts=tuple(contexts),
            description=str(data.get("description", "") or ""),
            source=str(data.get("source", "local") or "local"),
            tags=tuple(tags),
            path=path,
        )

    def summary(self) -> dict[str, Any]:
        """Session-independent serializer (CLI ``show`` / HTTP list)."""
        return {
            "name": self.name,
            "slot": self.slot,
            "n_values": len(self.values),
            "n_contexts": len(self.contexts),
            "values": list(self.values),
            "labels": self.node_labels(),
            "description": self.description,
            "source": self.source,
            "tags": list(self.tags),
        }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_body(
    name: str, data: dict[str, Any],
) -> tuple[str, list[str], list[TemplateContext]]:
    slot = data.get("slot")
    if not isinstance(slot, str) or not slot:
        raise TemplateFormatError(
            f"template {name!r} 'slot' must be a non-empty string"
        )

    values = data.get("values")
    if (
        not isinstance(values, list)
        or len(values) < 2
        or not all(isinstance(v, str) and v.strip() for v in values)
    ):
        raise TemplateFormatError(
            f"template {name!r} 'values' must be a list of >= 2 non-blank "
            f"strings (one choice per value)"
        )
    seen_labels: set[str] = set()
    for value in values:
        label = _slug_value(value)
        if not _LABEL_REGEX.match(label):
            raise TemplateFormatError(
                f"template {name!r} value {value!r} slugs to {label!r}, which "
                f"is not a valid node label (must match {_LABEL_REGEX.pattern})"
            )
        if label in seen_labels:
            raise TemplateFormatError(
                f"template {name!r} value {value!r} (label {label!r}) collides "
                f"with another value's label"
            )
        seen_labels.add(label)

    raw_contexts = data.get("contexts")
    if not isinstance(raw_contexts, list) or not raw_contexts:
        raise TemplateFormatError(
            f"template {name!r} 'contexts' must be a non-empty list"
        )
    contexts: list[TemplateContext] = []
    for i, ctx in enumerate(raw_contexts):
        contexts.append(_validate_context(name, i, ctx, slot))
    return slot, [str(v) for v in values], contexts


def _validate_context(
    name: str, i: int, ctx: Any, slot: str,
) -> TemplateContext:
    if not isinstance(ctx, dict):
        raise TemplateFormatError(
            f"template {name!r} context {i} must be an object with "
            f"'turns' and 'assistant'"
        )
    raw_turns = ctx.get("turns")
    if not isinstance(raw_turns, list) or not raw_turns:
        raise TemplateFormatError(
            f"template {name!r} context {i} 'turns' must be a non-empty list "
            f"of {{role, content}} objects"
        )
    turns: list[dict[str, str]] = []
    for j, turn in enumerate(raw_turns):
        if not isinstance(turn, dict):
            raise TemplateFormatError(
                f"template {name!r} context {i} turn {j} must be a "
                f"{{role, content}} object"
            )
        role = turn.get("role")
        content = turn.get("content")
        if role not in _VALID_ROLES:
            raise TemplateFormatError(
                f"template {name!r} context {i} turn {j} 'role' must be one of "
                f"{_VALID_ROLES}, got {role!r}"
            )
        if not isinstance(content, str) or not content.strip():
            raise TemplateFormatError(
                f"template {name!r} context {i} turn {j} 'content' must be a "
                f"non-blank string"
            )
        if slot in content:
            raise TemplateFormatError(
                f"template {name!r} context {i} turn {j} 'content' must not "
                f"contain the slot {slot!r}: history turns are shared "
                f"common-mode across values, so the slot lives only in the "
                f"final assistant turn"
            )
        turns.append({"role": role, "content": content})
    if turns[-1]["role"] != "user":
        raise TemplateFormatError(
            f"template {name!r} context {i} last history turn must be 'user' "
            f"(the slotted assistant turn follows it), got "
            f"{turns[-1]['role']!r}"
        )

    assistant = ctx.get("assistant")
    if not isinstance(assistant, str) or not assistant.strip():
        raise TemplateFormatError(
            f"template {name!r} context {i} 'assistant' must be a non-blank "
            f"string carrying the slot"
        )
    count = assistant.count(slot)
    if count != 1:
        raise TemplateFormatError(
            f"template {name!r} context {i} 'assistant' must contain the slot "
            f"{slot!r} exactly once (found {count}) — the value is read at a "
            f"single position"
        )
    return TemplateContext(turns=tuple(turns), assistant=assistant)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def template_dir(namespace: str, name: str) -> Path:
    return ensure_within(templates_dir(), namespace, name)


def create_template_folder(
    namespace: str,
    name: str,
    *,
    slot: str,
    values: list[str],
    contexts: list[dict[str, Any]],
    description: str = "",
    tags: list[str] | None = None,
    force: bool = False,
) -> TemplateFolder:
    """Author a fresh template artifact on disk.

    ``contexts`` is the authoring shape ``[{turns: [{role, content}],
    assistant: "...slot..."}]``. Validates the whole body before writing, so a
    malformed template leaves no half-built folder. Raises
    :class:`TemplateFormatError` on validation failure and
    :class:`FileExistsError` when a template already lives at the path (unless
    ``force``).
    """
    if not NAME_REGEX.match(namespace):
        raise TemplateFormatError(
            f"template namespace {namespace!r} invalid; must match "
            f"{NAME_REGEX.pattern}"
        )
    folder = TemplateFolder.from_payload(
        {
            "format_version": TEMPLATE_FORMAT_VERSION,
            "name": name,
            "slot": slot,
            "values": values,
            "contexts": contexts,
            "description": description,
            "tags": tags or [],
        },
    )
    path = template_dir(namespace, name)
    if (path / "template.json").exists() and not force:
        raise FileExistsError(
            f"template already exists at {path}; pass force=True to overwrite"
        )
    folder.write(path)
    return folder


def load_template(namespace: str, name: str) -> TemplateFolder:
    return TemplateFolder.load(template_dir(namespace, name))


def iter_template_folders() -> Iterator[TemplateFolder]:
    """Yield every loadable template under ``templates_dir()`` (skips broken)."""
    root = templates_dir()
    if not root.exists():
        return
    for ns_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for tdir in sorted(p for p in ns_dir.iterdir() if p.is_dir()):
            if not (tdir / "template.json").exists():
                continue
            try:
                yield TemplateFolder.load(tdir)
            except (TemplateFormatError, ValueError, OSError):
                continue


class AmbiguousTemplateError(ValueError, SaklasError):
    """A bare template name resolves to more than one namespace."""


class TemplateNotFoundError(KeyError, SaklasError):
    """No template matches the requested selector."""


def resolve_template(selector: str) -> TemplateFolder:
    """Resolve ``<name>`` or ``<ns>/<name>`` to a loaded :class:`TemplateFolder`.

    A bare name searches every namespace and raises
    :class:`AmbiguousTemplateError` on a cross-namespace collision (mirroring the
    manifold selector grammar). Raises :class:`TemplateNotFoundError` on a miss.
    """
    if "/" in selector:
        ns, _, nm = selector.partition("/")
        path = template_dir(ns, nm)
        if not (path / "template.json").exists():
            raise TemplateNotFoundError(f"no template {selector!r}")
        return TemplateFolder.load(path)
    matches = [t for t in iter_template_folders() if t.name == selector]
    if not matches:
        raise TemplateNotFoundError(f"no template named {selector!r}")
    if len(matches) > 1:
        where = sorted(str(t.path.parent.name) for t in matches if t.path)
        raise AmbiguousTemplateError(
            f"template {selector!r} is ambiguous across namespaces {where}; "
            f"qualify it as <ns>/{selector}"
        )
    return matches[0]


def remove_template_folder(namespace: str, name: str) -> bool:
    """Delete a template folder. Returns True if it existed."""
    import shutil

    path = template_dir(namespace, name)
    if not path.exists():
        return False
    shutil.rmtree(path)
    return True
