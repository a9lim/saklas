"""Selector grammar used by -r, -x, -l.

Kinds:
    name      : single concept; optionally scoped by namespace (Selector.namespace)
    tag       : concepts whose pack.json.tags contains this value
    namespace : all concepts under this namespace
    model     : resource scope (restrict operation to tensors for this model)
    all       : everything

Special alias: "default" -> namespace/default.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from saklas.packs import NAME_REGEX, PackFormatError, PackMetadata
from saklas.paths import vectors_dir


class SelectorError(ValueError):
    """Raised when a selector string cannot be parsed."""


class AmbiguousSelectorError(SelectorError):
    """Raised when a bare name matches multiple namespaces."""


@dataclass
class Selector:
    kind: str          # "name" | "tag" | "namespace" | "model" | "all"
    value: Optional[str]
    namespace: Optional[str] = None  # only meaningful when kind == "name"


@dataclass
class ResolvedConcept:
    namespace: str
    name: str
    folder: Path
    metadata: PackMetadata


_VALID_PREFIXES = {"tag", "namespace", "model"}


def parse(raw: str) -> Selector:
    if raw == "all":
        return Selector(kind="all", value=None)
    if raw == "default":
        return Selector(kind="namespace", value="default")

    if ":" in raw:
        prefix, rest = raw.split(":", 1)
        if prefix not in _VALID_PREFIXES:
            raise SelectorError(f"unknown selector prefix '{prefix}' in '{raw}'")
        if not rest:
            raise SelectorError(f"empty value after '{prefix}:' in '{raw}'")
        return Selector(kind=prefix, value=rest)

    if "/" in raw:
        ns, name = raw.split("/", 1)
        if not NAME_REGEX.match(name):
            raise SelectorError(f"invalid concept name '{name}' in '{raw}'")
        if not NAME_REGEX.match(ns):
            raise SelectorError(f"invalid namespace '{ns}' in '{raw}'")
        return Selector(kind="name", value=name, namespace=ns)

    if not NAME_REGEX.match(raw):
        raise SelectorError(f"invalid concept name '{raw}'")
    return Selector(kind="name", value=raw)


def _all_concepts() -> list[ResolvedConcept]:
    root = vectors_dir()
    if not root.is_dir():
        return []
    out: list[ResolvedConcept] = []
    for ns_dir in sorted(root.iterdir()):
        if not ns_dir.is_dir():
            continue
        for cdir in sorted(ns_dir.iterdir()):
            if not cdir.is_dir() or not (cdir / "pack.json").is_file():
                continue
            try:
                meta = PackMetadata.load(cdir)
            except PackFormatError:
                continue
            out.append(ResolvedConcept(
                namespace=ns_dir.name, name=cdir.name, folder=cdir, metadata=meta,
            ))
    return out


def resolve(selector: Selector) -> list[ResolvedConcept]:
    concepts = _all_concepts()

    if selector.kind == "all":
        return concepts

    if selector.kind == "namespace":
        return [c for c in concepts if c.namespace == selector.value]

    if selector.kind == "tag":
        return [c for c in concepts if selector.value in c.metadata.tags]

    if selector.kind == "model":
        safe = selector.value.replace("/", "__")
        return [
            c for c in concepts
            if (c.folder / f"{safe}.safetensors").is_file()
        ]

    if selector.kind == "name":
        matches = [
            c for c in concepts
            if c.name == selector.value
            and (selector.namespace is None or c.namespace == selector.namespace)
        ]
        if len(matches) > 1 and selector.namespace is None:
            qualified = ", ".join(f"{c.namespace}/{c.name}" for c in matches)
            raise AmbiguousSelectorError(
                f"ambiguous concept '{selector.value}': matches {qualified}. "
                f"Specify with a namespace."
            )
        return matches

    raise SelectorError(f"unknown selector kind: {selector.kind}")


def parse_args(tokens: list[str]) -> tuple[Selector, Optional[str]]:
    """Parse a list of selector tokens into (concept selector, optional model scope).

    Rules:
      - at most one concept selector (name|tag|namespace|all)
      - at most one model: scope
    """
    concept: Optional[Selector] = None
    model: Optional[str] = None

    for tok in tokens:
        s = parse(tok)
        if s.kind == "model":
            if model is not None:
                raise SelectorError("only one model: scope allowed per invocation")
            model = s.value
        else:
            if concept is not None:
                raise SelectorError("only one concept selector allowed per invocation")
            concept = s

    if concept is None:
        concept = Selector(kind="all", value=None)
    return concept, model
