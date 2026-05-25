"""Selector grammar used by pack and vector subcommands.

Kinds:
    name      : single concept; optionally scoped by namespace (Selector.namespace)
    tag       : concepts whose pack.json.tags contains this value
    namespace : all concepts under this namespace
    model     : resource scope (restrict operation to tensors for this model)
    all       : everything

Special alias: "default" -> namespace/default.

This module lives under ``saklas.io`` because both ``io.cache_ops`` and
``core.session`` need the grammar; importing from ``cli`` would invert the
layer dependency. CLI runners parse argv with :func:`parse` and pass the
resulting :class:`Selector` instances down.
"""
from __future__ import annotations

import re as _re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from saklas.core.errors import SaklasError
from saklas.io.packs import NAME_REGEX, PackFormatError, PackMetadata
from saklas.io.paths import vectors_dir

_VARIANT_REGEX = _re.compile(r"^(raw|pca|sae(?:-[a-z0-9._-]+)?|role(?:-[a-z0-9._-]+)?)$")


class SelectorError(ValueError, SaklasError):
    """Raised when a selector string cannot be parsed."""

    def user_message(self) -> tuple[int, str]:
        return (400, str(self) or self.__class__.__name__)


class AmbiguousSelectorError(SelectorError):
    """Raised when a bare name matches multiple namespaces."""

    def user_message(self) -> tuple[int, str]:
        msg = str(self) or self.__class__.__name__
        # Suffix the canonical disambiguation tip — keeps the original
        # message (which already lists the colliding namespaces) but
        # nudges the user toward the fix.
        if "namespace/name" not in msg:
            msg = f"{msg} (disambiguate with namespace/name)"
        return (400, msg)


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
        if prefix in _VALID_PREFIXES:
            if not rest:
                raise SelectorError(f"empty value after '{prefix}:' in '{raw}'")
            return Selector(kind=prefix, value=rest)
        # Otherwise it's a name-with-variant; validate the variant and strip it.
        if not _VARIANT_REGEX.match(rest):
            raise SelectorError(f"unknown variant '{rest}' in '{raw}'")
        # Fall through to name validation with the variant stripped.
        raw = prefix

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
        # ``model:X`` matches any concept with an installed tensor for X,
        # regardless of variant — raw or SAE.
        from saklas.io.packs import enumerate_variants
        return [
            c for c in concepts
            if enumerate_variants(c.folder, selector.value)
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


def resolve_pole(
    raw: str, namespace: Optional[str] = None,
) -> tuple[str, int, Optional["ResolvedConcept"], str]:
    """Resolve a user-typed concept reference to ``(canonical, sign, match, variant)``.

    Grammar: ``<name_part>[":"<variant>]`` where ``<variant>`` is ``raw``
    (the default when no suffix is present), ``sae`` (unique SAE variant),
    ``sae-<release>``, ``role``, or ``role-<id>``. The name part feeds the
    existing pole-alias pipeline; variant is passed through unchanged for
    callers to propagate to autoload / registry-key selection.

    Alias resolution for bipolar packs: if the user types a single-pole
    name that appears on either side of an installed bipolar concept,
    return the full composite with a sign of +1 (positive pole) or -1
    (negative pole). Callers multiply the user-supplied alpha by ``sign``
    before storing it.

    Examples (assuming ``default/angry.calm`` is installed):
      ``resolve_pole("angry")`` -> ``("angry.calm", +1, <resolved>, "raw")``
      ``resolve_pole("calm")``  -> ``("angry.calm", -1, <resolved>, "raw")``
      ``resolve_pole("angry.calm")`` -> ``("angry.calm", +1, <resolved>, "raw")``
      ``resolve_pole("angry:sae")``  -> ``("angry.calm", +1, <resolved>, "sae")``

    Not-installed names fall through as fresh monopolar concepts with
    sign +1 and ``match=None`` so the caller can still feed them into
    the extraction pipeline.

    Raises:
        SelectorError: when the ``:variant`` suffix doesn't match
            ``_VARIANT_REGEX`` (``raw`` | ``pca`` | ``sae`` |
            ``sae-<release>`` | ``role`` | ``role-<id>``).
        AmbiguousSelectorError: when multiple installed concepts match
            the input under different canonical names (e.g. both
            ``alice/angry`` and ``default/angry.calm`` exist and the
            caller didn't supply a namespace). Also raised for
            intra-namespace collisions like ``default/happy.sad`` +
            ``default/happy.calm``.
    """
    # Variant suffix strips first — pole alias logic is variant-agnostic.
    variant = "raw"
    if ":" in raw:
        name_part, maybe_variant = raw.rsplit(":", 1)
        if _VARIANT_REGEX.match(maybe_variant):
            variant = maybe_variant
            raw = name_part
        else:
            raise SelectorError(f"unknown variant '{maybe_variant}' in '{raw}'")

    # Lazy import to avoid a cycle: session.py imports this module for
    # the broadened extract() lookup.
    from saklas.core.session import BIPOLAR_SEP, canonical_concept_name

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
        return slug, +1, None, variant

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

    c_name, c_sign, c_resolved = matches[0]
    return c_name, c_sign, c_resolved, variant


@dataclass(frozen=True)
class ResolvedManifoldLabel:
    """A unique match for :func:`resolve_manifold_label`.

    Carries both the manifold's namespace-qualified name (the form a
    grammar ``%`` term consumes) and the bare label so the synthesized
    term reads naturally:  ``<manifold_key>%<label>``.
    """

    namespace: str
    manifold_name: str
    label: str

    @property
    def manifold_key(self) -> str:
        """``<ns>/<name>`` — the form the grammar uses to reference the manifold."""
        return f"{self.namespace}/{self.manifold_name}"


def resolve_manifold_label(
    label: str, *, namespace: Optional[str] = None,
) -> Optional[ResolvedManifoldLabel]:
    """Resolve a bare label to (namespace, manifold, label) — or ``None``.

    Walks every installed manifold (or every manifold inside
    ``namespace`` when set) for one whose ``node_labels`` contains
    ``label``.  Returns a :class:`ResolvedManifoldLabel` on a single
    match, ``None`` when nothing matches (the caller falls through to
    other resolution tiers — e.g. a fresh concept), and raises
    :class:`AmbiguousSelectorError` when multiple manifolds carry a
    node by the same label.  This is the manifold analogue of
    :func:`resolve_pole`'s bipolar-pole alias; together they let
    ``persona`` (the manifold node) and ``angry`` (the bipolar pole)
    resolve through the same bare-name surface.

    The lookup is a folder scan, not a fitted-tensor check — labels
    exist on the artifact regardless of whether the manifold has been
    fitted for the loaded model.  A persona manifold authored on
    disk but never fitted still surfaces here.
    """
    # Lazy import — selectors.py lives under io/ and shouldn't import
    # at module-load time from a peer that may import it back.
    from saklas.io.manifolds import (
        ManifoldFormatError, iter_manifold_folders,
    )

    matches: list[ResolvedManifoldLabel] = []
    try:
        manifolds = list(iter_manifold_folders(namespace))
    except ManifoldFormatError:
        # A single malformed folder shouldn't poison the whole resolve;
        # ``iter_manifold_folders`` already skips them, but defensive.
        return None
    for ns, mf in manifolds:
        if label in mf.node_labels:
            matches.append(ResolvedManifoldLabel(
                namespace=ns,
                manifold_name=mf.name,
                label=label,
            ))
    if not matches:
        return None
    if len(matches) > 1:
        qualified = ", ".join(
            f"{m.manifold_key} (label '{m.label}')" for m in matches
        )
        raise AmbiguousSelectorError(
            f"ambiguous manifold label '{label}': matches {qualified}. "
            f"Specify the manifold explicitly as <ns>/<name>%{label}."
        )
    return matches[0]


def resolve_bare_name(
    raw: str, *, namespace: Optional[str] = None,
) -> tuple[
    Optional[tuple[str, int, Optional["ResolvedConcept"], str]],
    Optional[ResolvedManifoldLabel],
]:
    """Unified bare-name resolver across the bipolar-pole and manifold-label tiers.

    Returns ``(pole_hit, manifold_hit)`` — exactly one is non-``None``
    on a successful resolution, both are ``None`` when no installed
    artifact owns ``raw``, and an :class:`AmbiguousSelectorError` is
    raised when both tiers claim ``raw`` (cross-tier collision —
    e.g. ``pirate`` is both a bipolar pole and a manifold node).

    Resolution order: pole resolution runs first because it's the
    historical surface and the disambiguation message it produces is
    already a known shape.  Manifold-label resolution is tried only
    when pole resolution doesn't hit an installed concept.  This
    means a bare ``pirate`` with a ``civilian.pirate`` pack *and* a
    ``persona`` manifold with a ``pirate`` node raises rather than
    silently picking one tier — same shape as cross-namespace pole
    ambiguity.

    The caller routes the hit downstream: a ``pole_hit`` synthesizes
    a plain vector term; a ``manifold_hit`` synthesizes a
    ``<manifold>%<label>`` :class:`~saklas.core.steering_expr.ManifoldTerm`.
    """
    # ``resolve_pole`` returns ``match=None`` on a miss but doesn't
    # raise — peel its tuple to detect whether anything actually hit.
    pole_tuple = resolve_pole(raw, namespace=namespace)
    pole_hit = pole_tuple if pole_tuple[2] is not None else None

    manifold_hit = resolve_manifold_label(raw, namespace=namespace)

    if pole_hit is not None and manifold_hit is not None:
        canonical, sign, match, variant = pole_hit
        pole_label = (
            f"{match.namespace}/{canonical}"
            + (" (negated)" if sign < 0 else "")
            + (f":{variant}" if variant != "raw" else "")
        )
        raise AmbiguousSelectorError(
            f"ambiguous bare name '{raw}': matches both a vector "
            f"pole ({pole_label}) and a manifold node "
            f"({manifold_hit.manifold_key}%{manifold_hit.label}). "
            f"Qualify the form: '<canonical>' for the vector or "
            f"'<manifold>%<label>' for the manifold node."
        )
    return pole_hit, manifold_hit


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
