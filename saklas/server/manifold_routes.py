"""Native ``/saklas/v1/manifolds/*`` routes — manifold steering artifacts.

A manifold is a top-level artifact (labeled nodes on a domain), not
session-scoped, so these routes live beside the pack routes rather than
under ``/sessions/{id}``.  The exception is ``fit``, which needs the
loaded model — it runs ``session.extract_manifold`` under the session
lock and streams progress like ``/extract`` does.

Authoring (create / update) writes ``manifold.json`` + ``nodes/*.json``
through :mod:`saklas.io.manifolds`; steering a fitted manifold needs no
route — a ``%`` term in any steering expression already loads the
artifact lazily on scope entry.
"""
from __future__ import annotations

import asyncio
import json
import logging
import shutil
from typing import Any, Callable, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from saklas.core.manifold import domain_from_spec
from saklas.core.session import ConcurrentExtractionError, SaklasSession
from saklas.io.atomic import write_json_atomic
from saklas.io.manifolds import (
    ManifoldFolder,
    ManifoldFormatError,
    _sanitize_hyperparams,
    create_discover_manifold_folder,
    create_manifold_folder,
    iter_manifold_folders,
    min_nodes,
    update_manifold_folder,
)
from saklas.io.paths import manifold_dir, safe_model_id

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
    fit_mode: Literal["pca", "spectral"] = "pca"
    nodes: list[DiscoverNodeSpec]
    hyperparams: dict[str, Any] = {}


class GenerateManifoldRequest(BaseModel):
    """LLM-author a discover-mode manifold from a flat concept list.

    Wraps :meth:`SaklasSession.generate_concept_statements` — produces
    one statement corpus per concept by asking the loaded model for
    shared scenarios, then per-cell statements.  No coords supplied;
    the fit derives them per-model.
    """

    namespace: str = "local"
    name: str
    description: str = ""
    concepts: list[str]
    n_scenarios: int = 9
    statements_per_concept: int = 5
    fit_mode: Literal["pca", "spectral"] = "pca"
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

    For authored manifolds only ``sae`` / ``sae_revision`` are honored.
    For discover-mode manifolds ``fit_mode`` and ``hyperparams`` can
    override the folder's stored values; when provided, the folder
    manifest is rewritten *before* the fit so the cache key reflects
    the actual fit inputs.
    """

    sae: str | None = None
    sae_revision: str | None = None
    # Discover-mode override fields — ignored when the folder is authored.
    fit_mode: Literal["pca", "spectral"] | None = None
    hyperparams: dict[str, Any] | None = None


# ------------------------------------------------------------------ helpers ---

def _domain_label(spec: dict[str, Any]) -> str:
    """Short ``type(Nd)`` label for a manifold domain spec dict."""
    kind = spec.get("type", "?")
    if kind == "box":
        n = len(spec.get("axes", []))
    elif kind == "sphere":
        n = int(spec.get("dim", 0))
    elif kind == "custom":
        n = int(spec.get("embed_dim", 0))
    else:
        n = 0
    return f"{kind}({n}d)"


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

    ``full`` adds per-node statements and per-tensor fit detail — the
    list route omits both to stay light.
    """
    session_stem = safe_model_id(session.model_id)
    fitted_models = mf.tensor_models()
    fitted_for_session = session_stem in fitted_models
    n, effective_domain = _resolve_intrinsic_dim(mf, session_stem)

    stale = False
    if fitted_for_session:
        try:
            stale = mf.sidecar(session_stem).nodes_sha256 != mf.nodes_sha256()
        except (KeyError, ManifoldFormatError, OSError):
            stale = False

    # Per-node roles are part of the manifest shape — surface alongside
    # labels/coords.  Padded to ``node_count`` with ``None`` so a
    # consumer can ``zip`` against ``node_labels`` without checking the
    # length first.  All-``None`` (the legacy / non-role default)
    # serializes identically to today.
    node_roles_padded = list(mf.node_roles) + [None] * (
        len(mf.node_labels) - len(mf.node_roles)
    )

    # For fitted discover folders the derived per-model coords live in
    # the safetensors, not on the folder.  Load them once and share
    # between the list-level ``node_coords`` field and the detail-level
    # ``nodes`` block — the list shape needs them so the manifold rack
    # strip's snap-to-node action can sync the position sliders to the
    # picked node's actual coords (otherwise label-form selections show
    # zeros on every axis).  Cheap (one safetensors header read).
    derived_coords: list[list[float]] = []
    if fitted_for_session and mf.is_discover:
        from saklas.core.manifold import load_manifold
        try:
            m = load_manifold(mf.tensor_path(session_stem))
            derived_coords = [
                [float(x) for x in row]
                for row in m.node_coords.tolist()
            ]
        except (FileNotFoundError, KeyError, ValueError):
            derived_coords = []

    if mf.is_discover:
        node_coords_wire = derived_coords
    else:
        node_coords_wire = [list(c) for c in mf.node_coords]

    out: dict[str, Any] = {
        "namespace": namespace,
        "name": mf.name,
        "description": mf.description,
        # ``domain`` is the effective spec the frontend needs to render
        # controls — for an unfitted discover folder this stays the
        # empty ``{}`` from ``manifold.json``; for a fitted one we
        # surface the materialized ``CustomDomain(picked_k)`` spec so
        # the rack strip can build N sliders.
        "domain": effective_domain,
        "domain_label": _domain_label(effective_domain) if effective_domain else _domain_label(mf.domain),
        "intrinsic_dim": n,
        "min_nodes": min_nodes(n) if n > 0 else None,
        "node_count": len(mf.node_labels),
        "node_labels": list(mf.node_labels),
        "node_coords": node_coords_wire,
        "node_roles": node_roles_padded,
        "fit_mode": mf.fit_mode,
        "hyperparams": dict(mf.hyperparams),
        "fitted_models": fitted_models,
        "fitted_for_session": fitted_for_session,
        "stale": stale,
    }

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
        return namespace, ManifoldFolder.load(folder)
    except ManifoldFormatError as e:
        raise HTTPException(400, f"manifold {namespace}/{name} is malformed: {e}")


def _refuse_if_busy(session: SaklasSession) -> None:
    """Raise 409 when the engine gen-lock is held.

    ``session.lock`` (the asyncio HTTP serializer) orders manifold
    mutations against each other and the JSON fit path, but an SSE fit
    whose request was cancelled leaves its worker thread — and the
    ``_gen_lock`` it holds — alive past the cancel.  A non-blocking probe
    of ``_gen_lock`` refuses a folder mutation while that thread runs.
    """
    if not session._gen_lock.acquire(blocking=False):
        raise HTTPException(
            409, "a model operation is in flight; retry shortly",
        )
    session._gen_lock.release()


def _evict_manifold(session: SaklasSession, namespace: str, name: str) -> None:
    """Drop any cached in-memory ``Manifold`` for this artifact.

    The grammar key is ``[ns/]name[:variant]``; a bare name resolves
    cross-namespace, so both forms can be cached.  Pop every match so a
    delete / re-fit does not leave a stale tensor live.
    """
    prefixes = (name, f"{namespace}/{name}")
    for key in list(session._manifolds):
        head = key.rsplit(":", 1)[0] if ":" in key else key
        if head in prefixes:
            session._manifolds.pop(key, None)


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
            raise HTTPException(409, str(e))
        except ManifoldFormatError as e:
            raise HTTPException(400, str(e))
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
            raise HTTPException(409, str(e))
        except ManifoldFormatError as e:
            raise HTTPException(400, str(e))
        mf = ManifoldFolder.load(folder)
        return _manifold_json(req.namespace, mf, session, full=True)

    @app.post("/saklas/v1/manifolds/generate", status_code=201)
    async def generate_manifold(req: GenerateManifoldRequest, request: Request):
        """LLM-author a discover-mode manifold from a flat concept list.

        Runs :meth:`SaklasSession.generate_statements` under the
        session lock — the unified K-tuple generator producing one
        statement corpus per concept, scenarios shared across the
        row.  SSE progress when ``Accept: text/event-stream``, JSON
        otherwise.  Writes a fresh discover-mode manifold folder;
        pair with ``POST .../fit`` to derive coords + fit.
        """
        if len(req.concepts) < 2:
            raise HTTPException(
                400,
                "manifold generate: need >= 2 concepts "
                "(shared-scenario structure is meaningless with one)",
            )
        folder = manifold_dir(req.namespace, req.name)
        if (folder / "manifold.json").exists():
            if not req.force:
                raise HTTPException(
                    409,
                    f"manifold {req.namespace}/{req.name} already exists "
                    f"(pass force=true to overwrite)",
                )

        # ``role_per_node`` mirrors the CLI ``--role-per-node`` flag —
        # each concept slug doubles as that node's assistant-role
        # substitution.  An unsupported family raises at fit time
        # (the folder is model-agnostic; we don't pay the family check
        # at generate time).
        node_roles_map: dict[str, str | None] | None = None
        if req.role_per_node:
            node_roles_map = {c: c for c in req.concepts}

        def _gen(on_progress: Callable[[str], None]) -> dict[str, Any]:
            corpora = session.generate_statements(
                list(req.concepts),
                n_scenarios=req.n_scenarios,
                statements_per_cell=req.statements_per_concept,
                on_progress=on_progress,
            )
            if folder.exists():
                shutil.rmtree(folder)
            try:
                out_folder = create_discover_manifold_folder(
                    req.namespace, req.name, req.description,
                    fit_mode=req.fit_mode,
                    node_corpora=corpora,
                    hyperparams=req.hyperparams,
                    node_roles=node_roles_map,
                )
            except (FileExistsError, ManifoldFormatError) as e:
                raise HTTPException(400, str(e))
            write_json_atomic(
                out_folder / "scenarios.json",
                {
                    "generator": "session.generate_statements",
                    "n_scenarios": req.n_scenarios,
                    "statements_per_concept": req.statements_per_concept,
                    "concepts": list(req.concepts),
                    "model_id": session.model_id,
                },
            )
            _evict_manifold(session, req.namespace, req.name)
            ns, mf = _find_manifold(req.namespace, req.name)
            body = _manifold_json(ns, mf, session, full=True)
            body["done"] = True
            return body

        accept = request.headers.get("accept", "application/json")
        if "text/event-stream" in accept:
            async def _sse():
                loop = asyncio.get_running_loop()
                queue: asyncio.Queue[tuple[Any, ...]] = asyncio.Queue()

                def _on_progress(msg: str) -> None:
                    loop.call_soon_threadsafe(
                        queue.put_nowait, ("progress", msg),
                    )

                async with session.lock:
                    async def _run() -> None:
                        try:
                            body = await asyncio.to_thread(_gen, _on_progress)
                            queue.put_nowait(("done", body))
                        except HTTPException as e:
                            queue.put_nowait((
                                "error",
                                {"message": e.detail, "code": "HTTPException"},
                            ))
                        except ValueError as e:
                            queue.put_nowait((
                                "error",
                                {"message": str(e), "code": "ValueError"},
                            ))
                        except Exception as e:  # noqa: BLE001
                            log.exception("manifold generate crashed")
                            queue.put_nowait((
                                "error",
                                {
                                    "message": str(e) or "generate failed",
                                    "code": type(e).__name__,
                                },
                            ))

                    task = asyncio.create_task(_run())
                    try:
                        while True:
                            kind, payload = await queue.get()
                            if kind == "progress":
                                yield (
                                    f"event: progress\n"
                                    f"data: {json.dumps({'message': payload})}\n\n"
                                )
                            elif kind == "done":
                                yield (
                                    f"event: done\n"
                                    f"data: {json.dumps(payload)}\n\n"
                                )
                                break
                            elif kind == "error":
                                yield (
                                    f"event: error\n"
                                    f"data: {json.dumps(payload)}\n\n"
                                )
                                break
                    finally:
                        if not task.done():
                            task.cancel()
                            try:
                                await task
                            except BaseException:  # noqa: BLE001
                                pass

            return StreamingResponse(_sse(), media_type="text/event-stream")

        async with session.lock:
            try:
                return await asyncio.to_thread(_gen, lambda _msg: None)
            except ValueError as e:
                raise HTTPException(400, str(e))

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
        # ``session.lock`` serializes against another PATCH and the JSON
        # fit path; the ``_gen_lock`` probe additionally refuses while a
        # fit thread is in flight (an SSE fit whose request was cancelled
        # leaves the worker thread — and the lock — alive past the cancel)
        # so a node-corpus rewrite can't race a fit reading ``nodes/``.
        async with session.lock:
            _refuse_if_busy(session)
            try:
                await asyncio.to_thread(
                    update_manifold_folder,
                    folder,
                    description=req.description,
                    nodes=nodes,
                )
            except ManifoldFormatError as e:
                raise HTTPException(400, str(e))
            _evict_manifold(session, namespace, name)
        ns, mf = _find_manifold(namespace, name)
        return _manifold_json(ns, mf, session, full=True)

    @app.delete("/saklas/v1/manifolds/{namespace}/{name}")
    async def delete_manifold(namespace: str, name: str):
        """Remove a manifold folder.

        Held under ``session.lock`` so it serializes against PATCH and
        the JSON fit path; refuses (409) when a fit thread still holds
        the engine gen-lock — deleting ``nodes/`` mid-fit would corrupt
        the read.
        """
        folder = manifold_dir(namespace, name)
        if not (folder / "manifold.json").exists():
            raise HTTPException(404, f"manifold {namespace}/{name} not found")
        async with session.lock:
            _refuse_if_busy(session)
            _evict_manifold(session, namespace, name)
            await asyncio.to_thread(shutil.rmtree, folder)
        return {"namespace": namespace, "name": name, "removed": True}

    @app.post("/saklas/v1/manifolds/{namespace}/{name}/fit")
    async def fit_manifold(
        namespace: str, name: str, req: FitManifoldRequest, request: Request,
    ):
        """Fit the manifold for the loaded model.

        Runs :meth:`SaklasSession.extract_manifold` under the session
        lock.  SSE progress when ``Accept: text/event-stream``, JSON
        otherwise.  Poisedness failures (a bare ``ValueError`` from the
        RBF solve, not a ``SaklasError``) are caught explicitly and
        surfaced as a clean error frame.
        """
        folder = manifold_dir(namespace, name)
        if not (folder / "manifold.json").exists():
            raise HTTPException(404, f"manifold {namespace}/{name} not found")

        # Discover-mode hyperparam overrides: write to the folder
        # manifest *before* the fit so the cache key reflects the
        # actual fit inputs.  Authored folders ignore these fields
        # (FitManifoldRequest accepts them, the discriminator below
        # gates the rewrite).
        if req.fit_mode is not None or req.hyperparams is not None:
            try:
                pre_mf = ManifoldFolder.load(folder)
            except ManifoldFormatError as e:
                raise HTTPException(400, str(e))
            if not pre_mf.is_discover and (
                req.fit_mode is not None or req.hyperparams is not None
            ):
                raise HTTPException(
                    400,
                    f"fit_mode/hyperparams overrides are discover-mode "
                    f"only; {namespace}/{name} is authored",
                )
            new_fit_mode = req.fit_mode or pre_mf.fit_mode
            new_hp = dict(pre_mf.hyperparams)
            if req.hyperparams is not None:
                new_hp.update(req.hyperparams)
            # Method-incompatible knobs get dropped at the IO boundary.
            new_hp = _sanitize_hyperparams(new_fit_mode, new_hp)
            data = json.loads((folder / "manifold.json").read_text())
            data["fit_mode"] = new_fit_mode
            data["hyperparams"] = new_hp
            data["nodes"] = [{"label": label} for label in pre_mf.node_labels]
            data.pop("domain", None)
            # Staged write — a crash mid-rewrite would corrupt the
            # manifest and 400 every subsequent route call.  Same
            # discipline ``io.manifolds`` uses for every other manifest
            # write; this override path was the lone outlier.
            write_json_atomic(folder / "manifold.json", data)

        def _fit(on_progress: Callable[[str], None]) -> dict[str, Any]:
            manifold = session.extract_manifold(
                folder, sae=req.sae, sae_revision=req.sae_revision,
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
            async def _sse():
                loop = asyncio.get_running_loop()
                queue: asyncio.Queue[tuple[Any, ...]] = asyncio.Queue()

                def _on_progress(msg: str) -> None:
                    loop.call_soon_threadsafe(
                        queue.put_nowait, ("progress", msg),
                    )

                async with session.lock:
                    async def _run() -> None:
                        try:
                            body = await asyncio.to_thread(_fit, _on_progress)
                            await asyncio.sleep(0)
                            queue.put_nowait(("done", body))
                        except ConcurrentExtractionError as e:
                            queue.put_nowait((
                                "error",
                                {"message": str(e), "code": "Conflict"},
                            ))
                        except (ValueError, ManifoldFormatError) as e:
                            queue.put_nowait((
                                "error",
                                {
                                    "message": str(e),
                                    "code": (
                                        "PoisednessError"
                                        if "poisedness" in str(e).lower()
                                        else type(e).__name__
                                    ),
                                },
                            ))
                        except Exception as e:  # noqa: BLE001
                            log.exception("manifold fit crashed")
                            queue.put_nowait((
                                "error",
                                {
                                    "message": str(e) or "fit failed",
                                    "code": type(e).__name__,
                                },
                            ))

                    task = asyncio.create_task(_run())
                    try:
                        while True:
                            item = await queue.get()
                            kind = item[0]
                            if kind == "progress":
                                yield (
                                    f"event: progress\n"
                                    f"data: {json.dumps({'message': item[1]})}\n\n"
                                )
                            elif kind == "done":
                                yield (
                                    f"event: done\n"
                                    f"data: {json.dumps(item[1])}\n\n"
                                )
                                break
                            elif kind == "error":
                                yield (
                                    f"event: error\n"
                                    f"data: {json.dumps(item[1])}\n\n"
                                )
                                break
                    finally:
                        if not task.done():
                            task.cancel()
                            try:
                                await task
                            except BaseException:  # noqa: BLE001
                                pass

            return StreamingResponse(_sse(), media_type="text/event-stream")

        async with session.lock:
            try:
                return await asyncio.to_thread(_fit, lambda _msg: None)
            except ConcurrentExtractionError as e:
                raise HTTPException(409, str(e))
            except (ValueError, ManifoldFormatError) as e:
                raise HTTPException(400, str(e))
