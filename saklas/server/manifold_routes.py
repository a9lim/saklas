"""Native ``/saklas/v1/manifolds/*`` routes — manifold steering artifacts.

A manifold is a top-level artifact (labeled nodes on a domain), not
session-scoped, so these routes live beside the pack routes rather than
under ``/sessions/{id}``.  The exception is ``fit``, which needs the
loaded model — it runs ``session.fit`` under the session
lock and streams progress like ``/extract`` does.

Authoring (create / update) writes ``manifold.json`` + ``nodes/*.json``
through :mod:`saklas.io.manifolds`; steering a fitted manifold needs no
route — a ``%`` term in any steering expression already loads the
artifact lazily on scope entry.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Literal, cast

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from saklas.core.manifold import domain_from_spec, manifold_is_affine
from saklas.core.session import (
    ConcurrentExtractionError,
    SaklasSession,
)
from saklas.io.hf_manifolds import (
    HFError as ManifoldHFError,
    ManifoldInstallConflict,
    install_manifold,
    search_manifolds,
)
from saklas.io.manifolds import (
    ManifoldFolder,
    ManifoldFormatError,
    append_discover_manifold_node,
    create_discover_manifold_folder,
    create_manifold_folder,
    create_manifold_from_template,
    domain_label,
    iter_manifold_folders,
    manifold_summary,
    merge_discover_manifolds,
    min_nodes,
    plan_discover_generation,
    remove_manifold_folder,
    update_manifold_folder,
)
from saklas.io.paths import manifold_dir
from saklas.server.app import acquire_session_lock
from saklas.server.sse import ProgressCallback, progress_sse_response

log = logging.getLogger("saklas.api")


# ----------------------------------------------------------- request models ---

class BoxAxisSpec(BaseModel):
    name: str = "axis"
    periodic: bool = False
    period: float = 1.0
    lo: float = 0.0
    hi: float = 1.0


class DomainSpec(BaseModel):
    """A manifold domain — box or sphere only (custom is JSON-authored)."""

    type: Literal["box", "sphere"]
    axes: list[BoxAxisSpec] | None = None
    dim: int | None = None


class NodeSpec(BaseModel):
    label: str
    coords: list[float]
    statements: list[str]
    # Optional per-node assistant-role substitution.  ``None`` (default)
    # = standard assistant baseline.  When set, this node's centroid is
    # pooled with the chat-template's assistant-role label replaced by
    # this slug.  Engine validates the slug shape (``[a-z0-9._-]+``)
    # and the family's support at fit time.
    role: str | None = None


class DiscoverNodeSpec(BaseModel):
    """A node in a discover-mode authoring payload — label + statements only.

    No ``coords``: coords are derived per-model at fit time from the
    pooled centroids.  ``role`` (optional) carries the per-node
    assistant-role substitution; see :class:`NodeSpec`.
    """

    label: str
    statements: list[str]
    role: str | None = None


class CreateManifoldRequest(BaseModel):
    namespace: str = "local"
    name: str
    description: str = ""
    domain: DomainSpec
    nodes: list[NodeSpec]


class CreateDiscoverManifoldRequest(BaseModel):
    """Author a discover-mode manifold from supplied per-concept corpora.

    The user provides labeled statement corpora; the fit derives node
    coordinates per-model via PCA or spectral embedding when the
    matching ``POST .../fit`` runs.
    """

    namespace: str = "local"
    name: str
    description: str = ""
    fit_mode: Literal["pca", "spectral", "auto"] = "pca"
    nodes: list[DiscoverNodeSpec]
    hyperparams: dict[str, Any] = {}


class TemplatePairSpec(BaseModel):
    """One ``{user, assistant}`` chat-turn template. The slot lives in the
    assistant turn (read off its last content token); the user turn is shared
    common-mode across nodes (no slot)."""

    user: str
    assistant: str


class CreateTemplatedManifoldRequest(BaseModel):
    """Author a templated discover manifold from a slot + values + pair set.

    The server writes a standalone template, expands ``slot`` across ``values``
    into per-value node corpora, and writes a discover folder carrying a
    ``template_ref`` back to that source template. The right tool for categories
    one references rather than embodies (days, months, colours, directions).
    Pair with ``POST .../fit`` (``fit_mode`` auto suits cyclic categories).
    """

    namespace: str = "local"
    name: str
    description: str = ""
    fit_mode: Literal["pca", "spectral", "auto"] = "auto"
    slot: str
    values: list[str]
    pairs: list[TemplatePairSpec]
    hyperparams: dict[str, Any] = {}
    force: bool = False


class GenerateManifoldRequest(BaseModel):
    """LLM-author a discover-mode manifold from a flat concept list.

    Wraps :meth:`SaklasSession.generate_responses` — produces one
    conversational corpus per concept (in-character responses to the shared
    baseline prompts, selected by ``kind``).  No coords supplied; the fit
    derives them per-model.
    """

    namespace: str = "local"
    name: str
    description: str = ""
    concepts: list[str]
    kind: Literal["abstract", "concrete", "custom"] = "abstract"
    # Required when ``kind == "custom"`` — the free-form elicitation system
    # prompt (``{c}`` = the concept), no role swap (pools in standard space).
    custom_system: str | None = None
    samples_per_prompt: int = 1
    fit_mode: Literal["pca", "spectral", "auto"] = "pca"
    hyperparams: dict[str, Any] = {}
    force: bool = False
    # ``role_per_node=True`` (the GUI "use slug as per-node role"
    # checkbox) means each ``concepts[i]`` slug doubles as that node's
    # assistant-role substitution at fit time — producing a persona
    # manifold where the geometry captures persona-relative structure
    # in role-baselined activation space.
    role_per_node: bool = False


class UpdateManifoldRequest(BaseModel):
    description: str | None = None
    nodes: list[NodeSpec] | None = None


class FitManifoldRequest(BaseModel):
    """Body for ``POST /manifolds/{ns}/{name}/fit``.

    For authored manifolds only ``sae`` is honored.
    For discover-mode manifolds ``fit_mode`` and ``hyperparams`` can
    override the folder's stored values; when provided, the folder
    manifest is rewritten *before* the fit so the cache key reflects
    the actual fit inputs.
    """

    sae: str | None = None
    layers: list[int] | Literal["workspace", "all"] | None = None
    force: bool = False
    # Discover-mode override fields — ignored when the folder is authored.
    fit_mode: Literal["pca", "spectral", "auto"] | None = None
    hyperparams: dict[str, Any] | None = None


class InstallManifoldRequest(BaseModel):
    """Body for ``POST /manifolds/install``.

    ``target`` is an HF coord (``owner/name[@revision]``) or a local folder path;
    ``as_`` re-namespaces the destination; ``force`` overwrites an existing
    folder at the destination.
    """

    target: str
    as_: str | None = Field(default=None, alias="as")
    force: bool = False

    model_config = {"populate_by_name": True}


class MergeManifoldSource(BaseModel):
    """One source folder in a manifold merge — fully qualified ``ns/name``."""

    namespace: str
    name: str


class MergeManifoldRequest(BaseModel):
    """Body for ``POST /manifolds/merge``.

    Restricted to discover-mode (autofitted) sources by design — see
    :func:`saklas.io.manifolds.merge_discover_manifolds`.  Pools the
    sources' node corpora + roles into one heap, writes a fresh
    discover folder; the next ``POST .../fit`` derives coords from the
    combined heap.
    """

    namespace: str = "local"
    name: str
    description: str = ""
    sources: list[MergeManifoldSource]
    # Defaults to the sources' shared fit_mode if they agree, else
    # required.  Server-side reconciliation lives in
    # ``merge_discover_manifolds``.
    fit_mode: Literal["pca", "spectral", "auto"] | None = None
    hyperparams: dict[str, Any] | None = None
    force: bool = False


def _intrinsic_dim(spec: dict[str, Any]) -> int:
    try:
        return domain_from_spec(spec).intrinsic_dim
    except (ValueError, KeyError):
        return 0


def _resolve_intrinsic_dim(mf: ManifoldFolder, session_stem: str) -> tuple[int, dict[str, Any]]:
    """Resolve the manifold's intrinsic dimension + effective domain spec.

    Authored folders carry the domain inline on ``manifold.json``; the
    answer is just ``_intrinsic_dim(mf.domain)`` and ``mf.domain`` itself.

    Discover folders carry an empty top-level ``domain`` — the real
    ``CustomDomain(picked_k)`` is materialized at fit time and lives on
    the per-model sidecar.  When a fit exists for the loaded model we
    read the sidecar's ``domain`` so the wire reports the actual dim
    (otherwise the webui's manifold strip renders zero controls on an
    otherwise-fitted persona manifold).

    Returns ``(intrinsic_dim, domain_spec)``.  When no fit is available
    on a discover folder we still report 0 / ``{}`` — the strip's
    "unfitted" warning covers that state.
    """
    n = _intrinsic_dim(mf.domain)
    if n > 0:
        return n, mf.domain
    if mf.is_discover and session_stem in mf.tensor_models():
        try:
            sc_domain = mf.sidecar(session_stem).domain
            return _intrinsic_dim(sc_domain), sc_domain
        except (KeyError, ManifoldFormatError):
            pass
    return 0, mf.domain


def _manifold_json(
    namespace: str,
    mf: ManifoldFolder,
    session: SaklasSession,
    *,
    full: bool,
) -> dict[str, Any]:
    """Serialize a :class:`ManifoldFolder` for the wire.

    The session-independent fields come straight from
    :func:`saklas.io.manifolds.manifold_summary` so this route and the
    CLI ``manifold show -j`` emit byte-identical values for every
    key they share (``namespace`` / ``name`` / ``description`` /
    ``source`` / ``fit_mode`` / ``is_discover`` / ``node_count`` /
    ``node_labels`` / ``node_roles`` / ``hyperparams`` / ``fitted_models``
    / ``tensor_variants``).  The server then layers the *session-aware*
    extras on top: ``fitted_for_session`` / ``stale``, plus — for a
    discover folder fitted on the loaded model — the materialized
    per-model geometry (``domain`` / ``domain_label`` / ``intrinsic_dim``
    / ``min_nodes`` / ``node_coords``) that ``manifold_summary`` leaves
    empty by design (it can't read the per-model safetensors without a
    session).

    ``full`` adds per-node statements and per-tensor fit detail — the
    list route omits both to stay light.
    """
    folder = manifold_dir(namespace, mf.name)
    out: dict[str, Any] = manifold_summary(folder)

    from saklas.io.paths import tensor_filename

    session_stem = tensor_filename(session.model_id).removesuffix(".safetensors")
    fitted_models = out["fitted_models"]
    fitted_for_session = session_stem in fitted_models
    n, effective_domain = _resolve_intrinsic_dim(mf, session_stem)

    stale = False
    if fitted_for_session:
        try:
            from saklas.core.model import loaded_model_fingerprint

            sidecar = mf.sidecar(session_stem)
            stale = (
                sidecar.nodes_sha256 != mf.nodes_sha256()
                or sidecar.model_fingerprint
                != loaded_model_fingerprint(session.model, session.model_id)
            )
        except (KeyError, ManifoldFormatError, OSError):
            stale = True

    # Roles are index-aligned with ``node_labels``; ``manifold_summary``
    # already padded them via ``_roles_padded`` so a consumer can ``zip``
    # against ``node_labels`` without a length check.  Reuse below.
    node_roles_padded = out["node_roles"]

    # For fitted discover folders the derived per-model coords live in
    # the safetensors, not on the folder.  Load them once and share
    # between the list-level ``node_coords`` field and the detail-level
    # ``nodes`` block — the list shape needs them so the manifold rack
    # strip's snap-to-node action can sync the position sliders to the
    # picked node's actual coords (otherwise label-form selections show
    # zeros on every axis).  Cheap (one safetensors header read).
    derived_coords: list[list[float]] = []
    resolved_affine: bool | None = None
    if fitted_for_session and mf.is_discover:
        from saklas.core.manifold import load_manifold
        try:
            m = load_manifold(mf.tensor_path(session_stem))
            derived_coords = [
                [float(x) for x in row]
                for row in m.node_coords.tolist()
            ]
            resolved_affine = manifold_is_affine(m)
        except (FileNotFoundError, KeyError, ValueError):
            derived_coords = []

    # Session-aware geometry override.  ``manifold_summary`` reports the
    # empty/discover form for a discover folder (no session, no per-model
    # read); the server lifts the materialized ``CustomDomain(picked_k)``
    # spec + derived coords from the per-model sidecar/tensor when a fit
    # exists so the rack strip can build N sliders and snap to a node.
    # Authored folders already have the right values from the summary, so
    # only touch these on the discover path.
    if mf.is_discover:
        out["domain"] = effective_domain
        out["domain_label"] = (
            domain_label(effective_domain)
            if effective_domain else domain_label(mf.domain)
        )
        out["intrinsic_dim"] = n
        out["min_nodes"] = min_nodes(n) if n > 0 else None
        out["node_coords"] = derived_coords

    # Resolved flat/curved geometry for the rack family split.  An
    # ``auto`` discover folder's family is per-model and only known
    # post-fit, so the client can't route it to the subspace (flat) vs
    # manifold (curved) drawer off ``fit_mode`` alone.  Surface the
    # resolved mode: ``"pca"`` (flat) / ``"spectral"`` (curved) for a
    # fitted discover folder, the concrete ``fit_mode`` for an authored /
    # baked folder, and ``None`` when an ``auto`` folder isn't fitted for
    # this model yet (the client shows an unresolved auto manifold in
    # both drawers until a fit pins the geometry).
    if mf.is_discover:
        out["resolved_fit_mode"] = (
            None if resolved_affine is None
            else ("pca" if resolved_affine else "spectral")
        )
    else:
        out["resolved_fit_mode"] = mf.fit_mode

    # Session-only extras layered on top of the shared summary.
    out["fitted_for_session"] = fitted_for_session
    out["stale"] = stale

    if full:
        groups = dict(mf.node_groups())
        if mf.fit_mode == "authored":
            out["nodes"] = [
                {
                    "label": label,
                    "coords": list(coords),
                    "statements": list(groups.get(label, [])),
                    "role": role,
                }
                for label, coords, role in zip(
                    mf.node_labels, mf.node_coords, node_roles_padded,
                    strict=True,
                )
            ]
        else:
            # Discover: nodes carry the derived per-model coords loaded
            # above (or ``None`` per node when no fit exists yet).
            out["nodes"] = [
                {
                    "label": label,
                    "coords": (
                        derived_coords[i] if i < len(derived_coords) else None
                    ),
                    "statements": list(groups.get(label, [])),
                    "role": node_roles_padded[i],
                }
                for i, label in enumerate(mf.node_labels)
            ]
        fitted: list[dict[str, Any]] = []
        for stem in fitted_models:
            try:
                sc = mf.sidecar(stem)
            except KeyError:
                continue
            entry: dict[str, Any] = {
                "stem": stem,
                "method": sc.method,
                "feature_space": sc.feature_space,
                "node_count": sc.node_count,
                "nodes_sha256": sc.nodes_sha256,
                "fit_mode": sc.fit_mode,
            }
            if sc.hyperparams:
                entry["hyperparams"] = sc.hyperparams
            if sc.diagnostics:
                entry["diagnostics"] = sc.diagnostics
            fitted.append(entry)
        out["fitted"] = fitted

    return out


def _find_manifold(
    namespace: str, name: str,
) -> tuple[str, ManifoldFolder]:
    """Locate one manifold by namespace + name, or 404."""
    folder = manifold_dir(namespace, name)
    if not (folder / "manifold.json").exists():
        raise HTTPException(404, f"manifold {namespace}/{name} not found")
    try:
        return namespace, ManifoldFolder.load(folder, verify_manifest=False)
    except ManifoldFormatError as e:
        raise HTTPException(
            400, f"manifold {namespace}/{name} is malformed: {e}",
        ) from e


def _refuse_if_busy(session: SaklasSession) -> None:
    """Raise 409 when the engine gen-lock is held.

    ``session.lock`` (the asyncio HTTP serializer) orders manifold
    mutations against each other and the JSON fit path, but an SSE fit
    whose request was cancelled leaves its worker thread — and the
    ``gen_lock`` it holds — alive past the cancel.  A non-blocking probe
    of ``gen_lock`` refuses a folder mutation while that thread runs.
    """
    if not session.gen_lock.acquire(blocking=False):
        raise HTTPException(
            409, "a model operation is in flight; retry shortly",
        )
    session.gen_lock.release()


def _evict_manifold(session: SaklasSession, namespace: str, name: str) -> None:
    """Drop any cached in-memory ``Manifold`` for this artifact.

    The grammar key is ``[ns/]name[:variant]``; a bare name resolves
    cross-namespace, so both forms can be cached.  Pop every match so a
    delete / re-fit does not leave a stale tensor live.
    """
    prefixes = (name, f"{namespace}/{name}")
    for key in list(session.manifolds):
        head = key.rsplit(":", 1)[0] if ":" in key else key
        if head in prefixes:
            session.manifolds.pop(key, None)


# ------------------------------------------------------------------- routes ---

def register_manifold_routes(app: FastAPI) -> None:
    """Mount the ``/saklas/v1/manifolds/*`` tree onto ``app``."""

    session: SaklasSession = app.state.session

    @app.get("/saklas/v1/manifolds")
    def list_manifolds():
        """List every installed manifold with per-session fit status."""
        return {
            "manifolds": [
                _manifold_json(ns, mf, session, full=False)
                for ns, mf in iter_manifold_folders()
            ],
        }

    @app.get("/saklas/v1/manifolds/{namespace}/{name}")
    def get_manifold(namespace: str, name: str):
        """One manifold: domain, nodes with statements, per-tensor fit detail."""
        ns, mf = _find_manifold(namespace, name)
        return _manifold_json(ns, mf, session, full=True)

    @app.post("/saklas/v1/manifolds", status_code=201)
    def create_manifold(req: CreateManifoldRequest):
        """Author a fresh authored manifold artifact on disk.

        Writes ``manifold.json`` + the ``nodes/`` corpus.  Returns the
        manifold detail plus ``advisories`` — soft poisedness / flat-axis
        warnings so the UI can flag a deficient node layout before a fit
        is paid for.  For discover-mode authoring (no per-node coords),
        ``POST /saklas/v1/manifolds/discover`` is the entry point.
        """
        domain_spec = req.domain.model_dump(exclude_none=True)
        nodes = [n.model_dump() for n in req.nodes]
        try:
            folder, advisories = create_manifold_folder(
                req.namespace, req.name, req.description, domain_spec, nodes,
            )
        except FileExistsError as e:
            raise HTTPException(409, str(e)) from e
        except ManifoldFormatError as e:
            raise HTTPException(400, str(e)) from e
        mf = ManifoldFolder.load(folder)
        body = _manifold_json(req.namespace, mf, session, full=True)
        body["advisories"] = advisories
        return body

    @app.post("/saklas/v1/manifolds/discover", status_code=201)
    def create_discover_manifold(req: CreateDiscoverManifoldRequest):
        """Author a fresh discover-mode manifold from supplied per-concept corpora.

        Nodes carry ``{label, statements}`` only — coords are derived
        per-model at fit time.  Pair with ``POST /saklas/v1/manifolds/
        {ns}/{name}/fit`` to run the discovery + fit.
        """
        node_corpora = {n.label: list(n.statements) for n in req.nodes}
        # Roles ride per-node; ``None`` (default) on a NodeSpec maps to
        # the standard assistant baseline.  An all-``None`` dict is
        # equivalent to "no roles" and is byte-identical to the
        # pre-A-phase manifest.
        node_roles_map = {n.label: n.role for n in req.nodes}
        try:
            folder = create_discover_manifold_folder(
                req.namespace, req.name, req.description,
                fit_mode=req.fit_mode,
                node_corpora=node_corpora,
                hyperparams=req.hyperparams,
                node_roles=node_roles_map,
            )
        except FileExistsError as e:
            raise HTTPException(409, str(e)) from e
        except ManifoldFormatError as e:
            raise HTTPException(400, str(e)) from e
        mf = ManifoldFolder.load(folder)
        return _manifold_json(req.namespace, mf, session, full=True)

    @app.post("/saklas/v1/manifolds/templated", status_code=201)
    def create_templated_manifold(req: CreateTemplatedManifoldRequest):
        """Author a templated manifold from a slot + values + pair set.

        Bridge over the standalone-template artifact: writes a
        ``~/.saklas/templates/<ns>/<name>/`` template (single-turn contexts from
        the ``pairs``) then a discover manifold that ``template_ref``-erences it.
        The template is the authoring source of truth; the manifold derives its
        node corpora from it. Pair with ``POST .../fit`` to run discovery + fit.
        (Multi-turn contexts + the completion scorer ride the dedicated template
        routes / ``saklas template`` CLI.)
        """
        from saklas.io.templates import TemplateFormatError, create_template_folder

        contexts = [
            {"turns": [{"role": "user", "content": p.user}], "assistant": p.assistant}
            for p in req.pairs
        ]
        try:
            create_template_folder(
                req.namespace, req.name,
                slot=req.slot, values=list(req.values), contexts=contexts,
                description=req.description, force=req.force,
            )
            folder = create_manifold_from_template(
                req.namespace, req.name, req.description,
                template_ref=f"{req.namespace}/{req.name}",
                fit_mode=req.fit_mode,
                hyperparams=req.hyperparams or None,
                force=req.force,
            )
        except FileExistsError as e:
            raise HTTPException(409, str(e)) from e
        except (ManifoldFormatError, TemplateFormatError) as e:
            raise HTTPException(400, str(e)) from e
        mf = ManifoldFolder.load(folder)
        return _manifold_json(req.namespace, mf, session, full=True)

    @app.get("/saklas/v1/manifolds/search")
    def search_remote_manifolds(q: str = "", limit: int = 20):
        """Search HF for ``saklas-manifold``-tagged repos matching ``q``.

        Delegates to :func:`saklas.io.hf_manifolds.search_manifolds` and returns
        the manifold-specific ``domain_label`` / ``node_count`` / ``fit_mode``
        fields a frontend needs to render a search result without an extra
        round-trip. Missing ``huggingface_hub`` → 503, HF transport error → 502.
        """
        try:
            rows = search_manifolds(q or None)
        except ImportError as e:
            raise HTTPException(503, f"huggingface_hub not installed: {e}") from e
        except ManifoldHFError as e:
            raise HTTPException(502, str(e)) from e
        # ``limit`` defaults to the search cap; callers can request
        # fewer rows for narrow pickers but the server still respects
        # the cap when ``limit`` is omitted or oversized.
        if limit and limit > 0:
            rows = rows[: int(limit)]
        # Echo ``query`` for parity with ``GET /saklas/v1/packs/search``.
        return {"query": q, "results": rows}

    @app.post("/saklas/v1/manifolds/merge", status_code=201)
    async def merge_manifold(req: MergeManifoldRequest):
        """Union N discover-mode manifolds' nodes into a fresh discover folder.

        The manifold-side counterpart to ``POST /saklas/v1/vectors/bake``,
        but operating on *node corpora* rather than steering directions:
        pools the sources' nodes into one heap and writes an unfitted
        discover folder.  The next ``POST .../fit`` derives coords from
        the combined heap.

        Restricted to discover-mode sources by design — authored manifolds
        carry user-declared geometry and aren't mergeable without a
        shared coordinate system.  Label collisions across sources are
        refused with a clear error (rename in source folders first).

        Held under the session lock so a parallel fit on one of the
        sources can't race the corpus read; ``_refuse_if_busy`` guards
        against an in-flight engine operation holding the gen-lock.
        """
        if len(req.sources) < 2:
            raise HTTPException(
                400,
                f"manifold merge: need >= 2 sources, got {len(req.sources)}",
            )
        source_tuples = [(s.namespace, s.name) for s in req.sources]
        async with acquire_session_lock(session) as acquired:
            if not acquired:
                raise HTTPException(503, "session locked")
            _refuse_if_busy(session)
            try:
                folder = await asyncio.to_thread(
                    merge_discover_manifolds,
                    req.namespace,
                    req.name,
                    req.description,
                    sources=source_tuples,
                    fit_mode=req.fit_mode,
                    hyperparams=req.hyperparams,
                    force=req.force,
                )
            except FileExistsError as e:
                raise HTTPException(409, str(e)) from e
            except FileNotFoundError as e:
                raise HTTPException(404, str(e)) from e
            except ManifoldFormatError as e:
                raise HTTPException(400, str(e)) from e
            except ValueError as e:
                raise HTTPException(400, str(e)) from e
            # Evict any cached in-memory ``Manifold`` for the merged
            # target — paranoia in case the user is merging-over an
            # existing folder with the same name (force=True path).
            _evict_manifold(session, req.namespace, req.name)
        ns, mf = _find_manifold(folder.parent.name, folder.name)
        return _manifold_json(ns, mf, session, full=True)

    @app.post("/saklas/v1/manifolds/install", status_code=201)
    async def install_remote_manifold(req: InstallManifoldRequest):
        """Install a manifold from an HF coord or local folder.

        Mirrors ``POST /saklas/v1/packs`` for parity with the vector
        pack side.  ``target`` is an HF coord (``owner/name[@revision]``)
        or a local folder path; ``as_`` overrides the destination
        namespace+name (must be fully qualified); ``force`` overwrites
        an existing folder.  Held under the session lock so a parallel
        delete / fit can't race the swap-into-place.  Returns the same
        manifold-detail JSON shape ``GET /saklas/v1/manifolds/{ns}/{name}``
        ships.
        """
        async with acquire_session_lock(session) as acquired:
            if not acquired:
                raise HTTPException(503, "session locked")
            _refuse_if_busy(session)
            try:
                folder = await asyncio.to_thread(
                    install_manifold,
                    req.target,
                    req.as_,
                    force=req.force,
                )
            except ManifoldInstallConflict as e:
                raise HTTPException(409, str(e)) from e
            except FileNotFoundError as e:
                raise HTTPException(404, str(e)) from e
            except ValueError as e:
                raise HTTPException(400, str(e)) from e
            except ImportError as e:
                raise HTTPException(503, f"huggingface_hub not installed: {e}") from e
            except ManifoldHFError as e:
                raise HTTPException(502, str(e)) from e

        # The just-installed folder lives at ``manifolds/<ns>/<name>/`` —
        # derive namespace/name from the resolved path so the response
        # carries the destination identity even when ``as_`` re-routed it.
        dst_namespace = folder.parent.name
        dst_name = folder.name
        ns, mf = _find_manifold(dst_namespace, dst_name)
        return _manifold_json(ns, mf, session, full=True)

    @app.post("/saklas/v1/manifolds/generate", status_code=201)
    async def generate_manifold(req: GenerateManifoldRequest, request: Request):
        """LLM-author a discover-mode manifold from a flat concept list (A2).

        Runs :meth:`SaklasSession.generate_responses` under the session lock —
        each concept answers the shared baseline prompts in character (concept
        in the system prompt + a kind-derived elicitation role), one corpus per
        node.  SSE progress when ``Accept: text/event-stream``, JSON otherwise.
        Writes a fresh discover-mode manifold folder; pair with ``POST .../fit``
        to derive coords + fit.
        """
        if len(req.concepts) < 2:
            raise HTTPException(
                400,
                "manifold generate: need >= 2 concepts "
                "(a discover manifold is meaningless with one node)",
            )
        folder = manifold_dir(req.namespace, req.name)

        # ``role_per_node`` mirrors the CLI ``--role-per-node`` flag — each
        # concept slug doubles as that node's assistant-role substitution (a
        # persona manifold, pooled in role-baselined space).  Otherwise the
        # node's ``kind`` drives a generation-only elicitation role and capture
        # stays standard (swap-back).
        if req.kind == "custom" and not req.custom_system:
            raise HTTPException(
                400, "kind='custom' requires custom_system (a system template "
                "with a {c} placeholder)",
            )
        node_roles_map: dict[str, str | None] | None = None
        if req.role_per_node:
            node_roles_map = {c: c for c in req.concepts}
        node_kinds_map: dict[str, str | None] = {c: req.kind for c in req.concepts}

        def _gen(on_progress: Callable[[str], None]) -> dict[str, Any]:
            # ``force`` is a clean slate; the default *resumes* — fill missing
            # nodes + append any concepts new to the roster.
            try:
                plan = plan_discover_generation(
                    folder, req.name, req.description,
                    fit_mode=req.fit_mode, labels=list(req.concepts),
                    hyperparams=req.hyperparams, node_roles=node_roles_map,
                    node_kinds=node_kinds_map,
                    force=req.force,
                )
            except ManifoldFormatError as e:
                raise HTTPException(400, str(e)) from e
            for concept in plan.pending:
                gen_roles: dict[str, str | None] | None = (
                    {concept: concept} if node_roles_map else None
                )
                corpora = session.generate_responses(
                    [concept], [req.kind],
                    roles=gen_roles,
                    custom_system=req.custom_system,
                    samples_per_prompt=req.samples_per_prompt,
                    on_progress=on_progress,
                )
                append_discover_manifold_node(
                    folder, plan.index_of[concept], concept, corpora[concept],
                )
            _evict_manifold(session, req.namespace, req.name)
            ns, mf = _find_manifold(req.namespace, req.name)
            body = _manifold_json(ns, mf, session, full=True)
            body["done"] = True
            return body

        accept = request.headers.get("accept", "application/json")
        if "text/event-stream" in accept:
            async def _job(on_progress: ProgressCallback) -> dict[str, Any]:
                return await asyncio.to_thread(_gen, on_progress)

            def _format_error(e: Exception) -> dict[str, Any] | None:
                if isinstance(e, HTTPException):
                    http_error = cast(HTTPException, e)
                    # The generate job only raises HTTPException to wrap a
                    # ManifoldFormatError, whose detail embeds the on-disk
                    # manifold path.  Log the detail server-side and surface a
                    # path-free frame (SSE info-disclosure discipline — see
                    # server/AGENTS.md).
                    log.warning("manifold generate: format error: %s", http_error.detail)
                    return {
                        "message": "manifold has an unsupported on-disk format",
                        "code": "ManifoldFormatError",
                    }
                if isinstance(e, ValueError):
                    return {"message": str(e), "code": "ValueError"}
                return None

            return progress_sse_response(
                session.lock,
                _job,
                error_message="generate failed",
                log_message="manifold generate crashed",
                error_formatter=_format_error,
                logger=log,
            )

        async with acquire_session_lock(session) as acquired:
            if not acquired:
                raise HTTPException(503, "session locked")
            try:
                return await asyncio.to_thread(_gen, lambda _msg: None)
            except ValueError as e:
                raise HTTPException(400, str(e)) from e

    @app.patch("/saklas/v1/manifolds/{namespace}/{name}")
    async def update_manifold(
        namespace: str, name: str, req: UpdateManifoldRequest,
    ):
        """Re-author a manifold's description and/or node corpus.

        Held under the session lock so a node-corpus rewrite cannot race
        an in-flight fit reading the same ``nodes/`` directory.  Existing
        fitted tensors are kept but become stale.
        """
        folder = manifold_dir(namespace, name)
        if not (folder / "manifold.json").exists():
            raise HTTPException(404, f"manifold {namespace}/{name} not found")
        nodes = (
            [n.model_dump() for n in req.nodes]
            if req.nodes is not None else None
        )
        # Bounded via ``acquire_session_lock`` (300 s → 503) so a
        # long-running fit doesn't pin the lock for this PATCH indefinitely.
        # The ``_gen_lock`` probe additionally refuses while a fit thread is
        # in flight (an SSE fit whose request was cancelled leaves the worker
        # thread — and the lock — alive past the cancel) so a node-corpus
        # rewrite can't race a fit reading ``nodes/``.
        async with acquire_session_lock(session) as acquired:
            if not acquired:
                raise HTTPException(503, "session locked")
            _refuse_if_busy(session)
            try:
                await asyncio.to_thread(
                    update_manifold_folder,
                    folder,
                    description=req.description,
                    nodes=nodes,
                )
            except ManifoldFormatError as e:
                raise HTTPException(400, str(e)) from e
            _evict_manifold(session, namespace, name)
        ns, mf = _find_manifold(namespace, name)
        return _manifold_json(ns, mf, session, full=True)

    @app.delete("/saklas/v1/manifolds/{namespace}/{name}")
    async def delete_manifold(namespace: str, name: str):
        """Remove a manifold folder.

        Delegates the actual removal to
        :func:`saklas.io.manifolds.remove_manifold_folder` — the single
        source of truth shared with the CLI ``manifold rm`` — so
        bundled-respawn semantics stay in one place.  The response
        carries that helper's ``{namespace, name, source, removed,
        rematerializes_on_restart}`` (a superset of the historical
        ``{namespace, name, removed}``; ``source == "bundled"`` /
        ``default``-namespace flip ``rematerializes_on_restart`` so the
        client can pick a friendlier toast, matching the pack DELETE
        route).

        Held under ``session.lock`` so it serializes against PATCH and
        the JSON fit path; refuses (409) when a fit thread still holds
        the engine gen-lock — deleting ``nodes/`` mid-fit would corrupt
        the read.
        """
        folder = manifold_dir(namespace, name)
        if not (folder / "manifold.json").exists():
            raise HTTPException(404, f"manifold {namespace}/{name} not found")
        async with acquire_session_lock(session) as acquired:
            if not acquired:
                raise HTTPException(503, "session locked")
            _refuse_if_busy(session)
            _evict_manifold(session, namespace, name)
            try:
                return await asyncio.to_thread(
                    remove_manifold_folder, namespace, name,
                )
            except FileNotFoundError as e:
                # Lost a race with another delete between the pre-lock
                # existence check and acquiring the lock.
                raise HTTPException(404, str(e)) from e

    @app.post("/saklas/v1/manifolds/{namespace}/{name}/fit")
    async def fit_manifold(
        namespace: str, name: str, req: FitManifoldRequest, request: Request,
    ):
        """Fit the manifold for the loaded model.

        Runs :meth:`SaklasSession.fit` under the session
        lock.  SSE progress when ``Accept: text/event-stream``, JSON
        otherwise.  Poisedness failures (a bare ``ValueError`` from the
        RBF solve, not a ``SaklasError``) are caught explicitly and
        surfaced as a clean error frame.
        """
        folder = manifold_dir(namespace, name)
        if not (folder / "manifold.json").exists():
            raise HTTPException(404, f"manifold {namespace}/{name} not found")

        def _fit(on_progress: Callable[[str], None]) -> dict[str, Any]:
            manifold = session.fit(
                folder, sae=req.sae,
                layers=req.layers,
                fit_mode=req.fit_mode,
                hyperparams=req.hyperparams,
                force=req.force,
                on_progress=on_progress,
            )
            _evict_manifold(session, namespace, name)
            ns, mf = _find_manifold(namespace, name)
            body = _manifold_json(ns, mf, session, full=True)
            body["done"] = True
            body["layers_fitted"] = len(manifold.layers)
            body["feature_space"] = manifold.feature_space
            return body

        accept = request.headers.get("accept", "application/json")
        if "text/event-stream" in accept:
            async def _job(on_progress: ProgressCallback) -> dict[str, Any]:
                return await asyncio.to_thread(_fit, on_progress)

            def _format_error(e: Exception) -> dict[str, Any] | None:
                if isinstance(e, ConcurrentExtractionError):
                    return {"message": str(e), "code": "Conflict"}
                if isinstance(e, (ValueError, ManifoldFormatError)):
                    return {
                        "message": str(e),
                        "code": (
                            "PoisednessError"
                            if "poisedness" in str(e).lower()
                            else type(e).__name__
                        ),
                    }
                return None

            return progress_sse_response(
                session.lock,
                _job,
                error_message="fit failed",
                log_message="manifold fit crashed",
                error_formatter=_format_error,
                logger=log,
            )

        async with acquire_session_lock(session) as acquired:
            if not acquired:
                raise HTTPException(503, "session locked")
            try:
                return await asyncio.to_thread(_fit, lambda _msg: None)
            except ConcurrentExtractionError as e:
                raise HTTPException(409, str(e)) from e
            except (ValueError, ManifoldFormatError) as e:
                raise HTTPException(400, str(e)) from e
