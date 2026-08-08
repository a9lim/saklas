"""Selector grammar used by manifold and subspace subcommands.

Kinds:
    name      : single concept; optionally scoped by namespace (Selector.namespace)
    tag       : concepts whose manifold.json tags contain this value
    namespace : all concepts under this namespace
    model     : resource scope (restrict operation to tensors for this model)
    all       : everything

Special alias: "default" -> namespace/default.

This module lives under ``saklas.io`` because both the io lifecycle layer and
``core.session`` need the grammar; importing from ``cli`` would invert the
layer dependency. CLI runners parse argv with :func:`parse` and pass the
resulting :class:`Selector` instances down.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from saklas.core.errors import SaklasError
from saklas.core.naming import canonical_concept_name
from saklas.io.integrity import NAME_REGEX
from saklas.io.paths import VARIANT_SUFFIX_RE, manifolds_dir

# Single source of truth lives in ``io.paths`` (owns the variant scheme).
_VARIANT_REGEX = VARIANT_SUFFIX_RE


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
    """An installed concept — a fitted/authored manifold (4.0).

    ``folder`` is the manifold folder (``~/.saklas/manifolds/<ns>/<name>/``);
    ``tags`` carries the manifold's category tags for ``tag:`` selectors.
    """
    namespace: str
    name: str
    folder: Path
    tags: list[str]


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


@dataclass(frozen=True)
class _ManifoldIndex:
    """The three selector views, all projected from a single tree walk.

    ``concepts`` is every installed manifold as a :class:`ResolvedConcept`;
    ``labels`` is the flattened ``(namespace, manifold, node label)`` index;
    ``names`` is the 2-node ``pca`` subset addressable by manifold *name*.
    """

    concepts: list[ResolvedConcept]
    labels: list["ResolvedManifoldLabel"]
    names: list["ResolvedManifoldName"]


# One memoized walk keyed by the manifolds root. Walking the tree opens and
# parses every ``manifold.json`` under ``manifolds_dir()``, and a single
# compound steering expression hits all three views — so the walk runs once and
# the views are built from it in memory rather than each re-walking. Cleared by
# ``invalidate()``, which every io mutator calls for itself.
_index_cache: dict[Path, _ManifoldIndex] = {}


def _manifold_index() -> _ManifoldIndex:
    """Walk ``manifolds_dir()`` once and project the three selector views."""
    root = manifolds_dir()
    cached = _index_cache.get(root)
    if cached is not None:
        return cached
    # Lazy import — selectors.py lives under io/ and shouldn't import at
    # module-load time from a peer that may import it back.
    from saklas.io.manifolds import ManifoldFormatError, iter_manifold_folders

    try:
        folders = list(iter_manifold_folders())
    except ManifoldFormatError:
        # A single malformed folder shouldn't poison the whole resolve;
        # ``iter_manifold_folders`` already skips them, but defensive.
        folders = []

    concepts: list[ResolvedConcept] = []
    labels: list[ResolvedManifoldLabel] = []
    names: list[ResolvedManifoldName] = []
    for ns, mf in folders:
        concepts.append(ResolvedConcept(
            namespace=ns, name=mf.name, folder=mf.folder,
            tags=list(mf.tags or []),
        ))
        labels.extend(
            ResolvedManifoldLabel(namespace=ns, manifold_name=mf.name, label=label)
            for label in mf.node_labels
        )
        # Only a 2-node ``pca`` fit reads as a vector composite — its name maps
        # to node 0. Multi-node / curved / authored manifolds are addressed by
        # ``<name>%<label>`` only and never surface in the name view.
        if mf.fit_mode == "pca" and len(mf.node_labels) == 2:
            names.append(ResolvedManifoldName(
                namespace=ns, manifold_name=mf.name, pole_label=mf.node_labels[0],
            ))

    index = _ManifoldIndex(concepts=concepts, labels=labels, names=names)
    _index_cache[root] = index
    return index


def invalidate() -> None:
    """Drop the memoized manifold walk.

    Every io mutator that can change the installed roster calls this itself
    (authoring, lifecycle, install/pull, bundled materialization), so callers
    carry no invalidation duty.
    """
    _index_cache.clear()


def all_concepts() -> list[ResolvedConcept]:
    """Every installed concept — i.e. every installed manifold (4.0).

    Concepts and steering manifolds are the same artifact now, so this reads
    the memoized ``manifolds_dir()`` walk (:func:`_manifold_index`, which goes
    through :func:`~saklas.io.manifolds.iter_manifold_folders` and already
    skips malformed folders).
    """
    return _manifold_index().concepts


def resolve(selector: Selector) -> list[ResolvedConcept]:
    concepts = all_concepts()

    if selector.kind == "all":
        return concepts

    if selector.kind == "namespace":
        return [c for c in concepts if c.namespace == selector.value]

    if selector.kind == "tag":
        return [c for c in concepts if selector.value in c.tags]

    if selector.kind == "model":
        # ``model:X`` matches any concept (manifold) with a fitted tensor for X,
        # regardless of variant — raw or SAE.
        # parse() guarantees selector.value is non-None for kind="model".
        assert selector.value is not None
        from saklas.io.paths import parse_tensor_filename, safe_model_id
        sid = safe_model_id(selector.value)

        def _has_tensor(folder: Path) -> bool:
            for p in folder.glob("*.safetensors"):
                parsed = parse_tensor_filename(p.name)
                if parsed is not None and parsed[0] == sid:
                    return True
            return False

        return [c for c in concepts if _has_tensor(c.folder)]

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


def canonicalize_atom(raw: str) -> tuple[str, str]:
    """Peel a ``:variant`` suffix and canonicalize a concept reference.

    Grammar: ``<name_part>[":"<variant>]`` where ``<variant>`` is ``raw``
    (the default when no suffix is present), ``sae`` (unique SAE variant),
    ``sae-<release>``, ``role``, ``role-<id>``, or ``from``/``from-<src>``.
    Returns ``(canonical, variant)`` — the canonicalized slug and the parsed
    variant.  This is the surviving behavior of the retired ``resolve_pole``:
    bipolar-pole alias resolution (``wolf`` → ``deer.wolf @ -0.5``) moved
    entirely to the **manifold** tier (a bipolar concept is a 2-node ``pca``
    manifold), so a bare pole resolves through the label tier of
    :func:`resolve_bare_atom` as an affine ``%`` push and there is no distinct
    pole artifact left to scan / sign-flip — only the slug + variant peel
    remains.

    Raises:
        SelectorError: when the ``:variant`` suffix doesn't match
            ``_VARIANT_REGEX``.
    """
    variant = "raw"
    if ":" in raw:
        name_part, maybe_variant = raw.rsplit(":", 1)
        if _VARIANT_REGEX.match(maybe_variant):
            variant = maybe_variant
            raw = name_part
        else:
            raise SelectorError(f"unknown variant '{maybe_variant}' in '{raw}'")

    slug = canonical_concept_name(raw)
    return slug, variant


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

    Tier 1 of :func:`resolve_bare_atom`.  Scans the memoized manifold-label
    index (every installed manifold, or every manifold inside ``namespace``
    when set) for one whose ``node_labels`` contains ``label``.  Returns a
    :class:`ResolvedManifoldLabel` on a single
    match, ``None`` when nothing matches (the caller falls through to
    other resolution tiers — e.g. a fresh concept), and raises
    :class:`AmbiguousSelectorError` when multiple manifolds carry a
    node by the same label.  The bare-pole/label collapse (a bipolar
    concept *is* a 2-node manifold) lets ``persona`` (the manifold node)
    and ``angry`` (the bipolar pole) resolve through this one surface; a hit
    routes to a ``<manifold>%<label>``
    :class:`~saklas.core.steering_expr.ManifoldTerm`.

    The index is the all-namespace walk, narrowed in memory — bare-label
    resolution dominates and a namespace-scoped query is just a filter.
    The lookup is a folder scan, not a fitted-tensor check — labels
    exist on the artifact regardless of whether the manifold has been
    fitted for the loaded model.  A persona manifold authored on
    disk but never fitted still surfaces here.
    """
    index = _manifold_index().labels
    matches = [
        m for m in index
        if m.label == label
        and (namespace is None or m.namespace == namespace)
    ]
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


@dataclass(frozen=True)
class ResolvedManifoldName:
    """A unique match for :func:`resolve_manifold_name`.

    A 2-node ``pca`` manifold reads as a steering vector: its bare *name*
    (the composite ``happy.sad``) resolves to ``pole_label`` = node 0, the
    ``orient_to=0`` (+) pole, so ``0.5 happy.sad`` steers toward ``happy``
    exactly as the bipolar vector did pre-4.0.
    """

    namespace: str
    manifold_name: str
    pole_label: str

    @property
    def manifold_key(self) -> str:
        """``<ns>/<name>`` — the form the grammar uses to reference the manifold."""
        return f"{self.namespace}/{self.manifold_name}"


def resolve_manifold_name(
    name: str, *, namespace: Optional[str] = None,
) -> Optional[ResolvedManifoldName]:
    """Resolve a 2-node ``pca`` manifold *name* to its node-0 (+) pole.

    The vector-composite read path: ``happy.sad`` (the manifold name, which
    contains a ``.`` and so skips the bare-label tier) resolves to the
    ``happy`` pole.  Returns ``None`` on a miss (the caller falls through to
    other tiers), and raises :class:`AmbiguousSelectorError` when the same
    name lives in two namespaces and none was specified.
    """
    index = _manifold_index().names
    matches = [
        m for m in index
        if m.manifold_name == name
        and (namespace is None or m.namespace == namespace)
    ]
    if not matches:
        return None
    if len(matches) > 1:
        qualified = ", ".join(m.manifold_key for m in matches)
        raise AmbiguousSelectorError(
            f"ambiguous manifold name '{name}': matches {qualified}. "
            f"Specify the namespace as <ns>/{name}."
        )
    return matches[0]


@dataclass(frozen=True)
class ResolvedBareAtom:
    """The outcome of :func:`resolve_bare_atom` — the bare-atom tier ladder.

    Exactly one of ``manifold`` / ``manifold_name`` / ``pole`` is set,
    discriminated by ``kind``:

    - ``"label"``: a manifold node-label hit — ``manifold`` is the
      :class:`ResolvedManifoldLabel` (the caller synthesizes a label-form
      ``<manifold>%<label>`` ``ManifoldTerm``).
    - ``"name"``: a 2-node ``pca`` manifold *name* hit — ``manifold_name`` is
      the :class:`ResolvedManifoldName` (the caller synthesizes a ``%``
      push to node 0, the ``orient_to=0`` + pole).
    - ``"pole"``: neither manifold tier matched — ``pole`` is the
      ``(canonical_slug, variant)`` pair from :func:`canonicalize_atom`, a
      plain vector concept.
    """

    kind: str  # "label" | "name" | "pole"
    manifold: Optional[ResolvedManifoldLabel] = None
    manifold_name: Optional[ResolvedManifoldName] = None
    pole: Optional[tuple[str, str]] = None


def resolve_bare_atom(
    concept: str,
    *,
    namespace: Optional[str] = None,
    typed_namespace: Optional[str] = None,
    variant: str = "raw",
) -> ResolvedBareAtom:
    """Resolve a bare atom through the whole bare-atom tier ladder.

    This is the **single owner** of the ordering + arbitration the steering
    grammar used to hand-sequence (`core.steering_expr`): a concept *is* a
    manifold now, so a bare reference resolves, in order:

    1. **manifold-label tier** (:func:`resolve_manifold_label`) — when the atom is a
       plain bare slug (``variant == "raw"``, no user-typed namespace, no
       bipolar ``.``), a hit on a manifold's node label routes to a label-form
       ``%`` push.  Cross-*manifold* label collisions raise
       :class:`AmbiguousSelectorError` here.
    2. **composite-name tier** (:func:`resolve_manifold_name`) — when
       ``variant == "raw"``, a name that *is* a 2-node ``pca`` manifold
       (``formal.casual`` — the ``.`` skips tier 1) routes to its node-0 (+)
       pole.  Scoped by the user-typed namespace when present, else
       ``namespace``.
    3. **pole canonicalization** (:func:`canonicalize_atom`) — neither manifold
       tier matched, so the atom is a plain vector concept: peel the
       ``:variant`` suffix + canonicalize the slug.  Every bipolar pole is
       itself a node label, so a bare pole has already resolved through tier 1
       — this tier carries the genuine fresh-vector / variant / dotted-name
       references.

    ``namespace`` is the default (expression-level) namespace scope;
    ``typed_namespace`` is the namespace the user explicitly typed on the atom
    (``None`` for a bare reference).  A non-``AmbiguousSelectorError`` failure
    from a manifold tier falls through to the pole canonicalization, where a
    genuine error re-surfaces with the canonical message.

    Raises:
        AmbiguousSelectorError: a cross-namespace / cross-manifold collision in
            either manifold tier.
        SelectorError: an unknown ``:variant`` suffix (from tier 3).
    """
    # Tier 1 — manifold-label.  Gated to a plain bare slug: a typed namespace,
    # a non-``raw`` variant, or a bipolar ``.`` all skip it (those forms are
    # addressed explicitly and must not alias to an arbitrary node label).
    if (
        variant == "raw"
        and typed_namespace is None
        and "." not in concept
    ):
        try:
            label_hit = resolve_manifold_label(concept, namespace=namespace)
        except AmbiguousSelectorError:
            raise
        except Exception:
            label_hit = None
        if label_hit is not None:
            return ResolvedBareAtom(kind="label", manifold=label_hit)

    # Tier 2 — composite-name (the 2-node ``pca`` vector-composite read path).
    # ``formal.casual`` skips tier 1 on the ``.`` and lands here.
    if variant == "raw":
        ns_scope = typed_namespace or namespace
        try:
            name_hit = resolve_manifold_name(concept, namespace=ns_scope)
        except AmbiguousSelectorError:
            raise
        except Exception:
            name_hit = None
        if name_hit is not None:
            return ResolvedBareAtom(kind="name", manifold_name=name_hit)

    # Tier 3 — plain vector concept: peel + canonicalize.
    raw = concept if variant == "raw" else f"{concept}:{variant}"
    return ResolvedBareAtom(kind="pole", pole=canonicalize_atom(raw))


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
