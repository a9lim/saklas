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


# Module-level cache keyed by vectors root path. Walking the tree hits the
# filesystem for every concept folder — compound selectors like `-r tag:x -x
# model:y` resolve multiple times per invocation, so we memoize here and rely
# on cache_ops to call `invalidate()` after any mutation.
_concepts_cache: dict[Path, list[ResolvedConcept]] = {}


def invalidate() -> None:
    """Drop the cached concept walk. Call after install/delete/refresh."""
    _concepts_cache.clear()


def _all_concepts() -> list[ResolvedConcept]:
    root = vectors_dir()
    cached = _concepts_cache.get(root)
    if cached is not None:
        return cached
    if not root.is_dir():
        _concepts_cache[root] = []
        return _concepts_cache[root]
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
    _concepts_cache[root] = out
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


def resolve_pole(raw: str, namespace: Optional[str] = None) -> tuple[str, int, Optional["ResolvedConcept"]]:
    """Resolve a user-typed concept reference to ``(canonical, sign, match)``.

    Alias resolution for bipolar packs: if the user types a single-pole
    name that appears on either side of an installed bipolar concept,
    return the full composite with a sign of +1 (positive pole) or -1
    (negative pole). Callers multiply the user-supplied alpha by ``sign``
    before storing it.

    Examples (assuming ``default/angry.calm`` is installed):
      ``resolve_pole("angry")`` -> ``("angry.calm", +1, <resolved>)``
      ``resolve_pole("calm")``  -> ``("angry.calm", -1, <resolved>)``
      ``resolve_pole("angry.calm")`` -> ``("angry.calm", +1, <resolved>)``

    Not-installed names fall through as fresh monopolar concepts with
    sign +1 and ``match=None`` so the caller can still feed them into
    the extraction pipeline.

    Raises:
        AmbiguousSelectorError: when multiple installed concepts match
            the input under different canonical names (e.g. both
            ``alice/angry`` and ``default/angry.calm`` exist and the
            caller didn't supply a namespace). Also raised for
            intra-namespace collisions like ``default/happy.sad`` +
            ``default/happy.calm``.
    """
    # Lazy import to avoid a cycle: session.py imports cli_selectors for
    # the broadened extract() lookup.
    from saklas.session import BIPOLAR_SEP, canonical_concept_name

    slug = canonical_concept_name(raw)
    scope = [c for c in _all_concepts()
             if namespace is None or c.namespace == namespace]

    matches: list[tuple[str, int, ResolvedConcept]] = []
    for c in scope:
        if c.name == slug:
            matches.append((c.name, +1, c))
            continue
        if BIPOLAR_SEP in c.name:
            pos, neg = c.name.split(BIPOLAR_SEP, 1)
            if pos == slug:
                matches.append((c.name, +1, c))
            elif neg == slug:
                matches.append((c.name, -1, c))

    if not matches:
        return slug, +1, None

    # Ambiguous if the matches don't collapse to a single (canonical, sign)
    # or span multiple namespaces when none was specified — both raise the
    # same error class as resolve() does for plain selectors.
    canonicals = {(m[0], m[1]) for m in matches}
    namespaces = {m[2].namespace for m in matches}
    if len(canonicals) > 1 or (namespace is None and len(namespaces) > 1):
        qualified = ", ".join(
            f"{m[2].namespace}/{m[0]}{' (negated)' if m[1] < 0 else ''}"
            for m in matches
        )
        raise AmbiguousSelectorError(
            f"ambiguous pole '{raw}': matches {qualified}. "
            f"Specify the full composite or a namespace."
        )

    return matches[0]


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
