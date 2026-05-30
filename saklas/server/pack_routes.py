"""Native ``/saklas/v1/packs`` route group."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from saklas.server.manifold_routes import _refuse_if_busy


class InstallPackRequest(BaseModel):
    target: str
    as_: str | None = Field(default=None, alias="as")
    force: bool = False
    statements_only: bool = False

    model_config = {"populate_by_name": True}


def _pack_row(r: Any, sid: str) -> dict[str, Any]:
    """Serialize one installed-pack listing row for the wire.

    Shared by ``GET /packs`` (list) and ``GET /packs/{ns}/{name}``
    (single-pack detail) so both surfaces emit identical per-pack
    shapes.  ``sid`` is the loaded model's safe-id slug, used to derive
    the session-relative ``has_tensor`` flag.
    """
    return {
        "name": r.name,
        "namespace": r.namespace,
        "status": r.status,
        "recommended_alpha": r.recommended_alpha,
        "tags": list(r.tags),
        "description": r.description,
        "source": r.source,
        "tensor_models": list(r.tensor_models),
        "has_tensor": sid in r.tensor_models,
        **({"error": r.error} if r.error else {}),
    }


def register_pack_routes(app: FastAPI) -> None:
    """Mount pack listing/search/install/delete routes."""
    session = app.state.session

    @app.get("/saklas/v1/packs")
    def list_packs():
        from saklas.io.cache_ops import list_concepts as _list_concepts
        from saklas.io.paths import safe_model_id as _safe_id

        result = _list_concepts(None, hf=False)
        sid = _safe_id(session.model_id)
        return {"packs": [_pack_row(r, sid) for r in result.installed]}

    @app.get("/saklas/v1/packs/search")
    def search_packs(q: str = "", limit: int = 50):
        from saklas.io.cache_ops import search_remote_packs as _search

        try:
            rows = _search(q)
        except ImportError as e:
            raise HTTPException(503, f"hf search unavailable: {e}")
        except Exception as e:
            raise HTTPException(502, f"hf search failed: {type(e).__name__}: {e}")
        if limit and limit > 0:
            rows = rows[:limit]
        return {
            "query": q,
            "results": [
                {
                    "name": r.name,
                    "namespace": r.namespace,
                    "recommended_alpha": r.recommended_alpha,
                    "tags": list(r.tags),
                    "description": r.description,
                    "tensor_models": list(r.tensor_models),
                }
                for r in rows
            ],
        }

    @app.get("/saklas/v1/packs/{namespace}/{name}")
    def get_pack(namespace: str, name: str):
        """One installed pack's detail — the same per-pack shape ``GET
        /packs`` list entries carry.  Parallels ``GET /saklas/v1/
        manifolds/{ns}/{name}``.  404 when not installed.
        """
        from saklas.io.cache_ops import list_concepts as _list_concepts
        from saklas.io.paths import safe_model_id as _safe_id

        rows = _list_concepts(None, hf=False).installed
        match = next(
            (r for r in rows if r.namespace == namespace and r.name == name),
            None,
        )
        if match is None:
            raise HTTPException(404, f"pack '{namespace}/{name}' not installed")
        return _pack_row(match, _safe_id(session.model_id))

    @app.post("/saklas/v1/packs")
    async def install_pack(req: InstallPackRequest):
        from saklas.io.cache_ops import InstallConflict, install as _install
        from saklas.io.cache_ops import list_concepts as _list_concepts
        from saklas.io.paths import safe_model_id as _safe_id

        try:
            dst = await asyncio.to_thread(
                _install,
                req.target,
                req.as_,
                force=req.force,
                statements_only=req.statements_only,
            )
        except FileNotFoundError as e:
            raise HTTPException(404, f"pack not found: {e}")
        except InstallConflict as e:
            raise HTTPException(409, str(e))
        except ValueError as e:
            raise HTTPException(400, str(e))
        # Non-breaking superset: keep the historical receipt keys
        # (``target`` / ``installed_at`` / ``statements_only``) and layer
        # the installed-pack detail fields on top — paralleling the
        # manifold install route, which returns full detail.  Derive the
        # destination identity from the resolved folder so an ``as``
        # re-route is reflected.
        dst_namespace = dst.parent.name
        dst_name = dst.name
        rows = _list_concepts(None, hf=False).installed
        match = next(
            (
                r for r in rows
                if r.namespace == dst_namespace and r.name == dst_name
            ),
            None,
        )
        receipt = {
            "target": req.target,
            "installed_at": str(dst),
            "statements_only": req.statements_only,
        }
        if match is not None:
            return {**_pack_row(match, _safe_id(session.model_id)), **receipt}
        return receipt

    @app.delete("/saklas/v1/packs/{namespace}/{name}")
    async def delete_pack(namespace: str, name: str):
        from saklas.io.cache_ops import list_concepts as _list_concepts
        from saklas.io.cache_ops import uninstall as _uninstall
        from saklas.io.selectors import parse as _parse_selector

        try:
            selector = _parse_selector(f"{namespace}/{name}")
        except ValueError as e:
            raise HTTPException(400, str(e))

        rows = _list_concepts(None, hf=False).installed
        match = next(
            (r for r in rows if r.namespace == namespace and r.name == name),
            None,
        )
        if match is None:
            raise HTTPException(404, f"pack '{namespace}/{name}' not installed")
        source = match.source

        async with session.lock:
            # Refuse (409) while an in-flight extract holds the engine
            # gen-lock — parity with the manifold DELETE route, so a pack
            # removal can't race a concurrent extraction on its tensor.
            _refuse_if_busy(session)
            await asyncio.to_thread(
                lambda: (
                    session.unsteer(name) if name in session.vectors else None,
                ),
            )
            qualified = f"{namespace}/{name}"
            if qualified in session.vectors:
                await asyncio.to_thread(session.unsteer, qualified)
            try:
                count = await asyncio.to_thread(
                    _uninstall, selector, yes=True,
                )
            except RuntimeError as e:
                raise HTTPException(400, str(e))

        if count == 0:
            raise HTTPException(404, f"pack '{namespace}/{name}' not installed")
        return {
            "namespace": namespace,
            "name": name,
            "source": source,
            "removed": count,
            "rematerializes_on_restart": source == "bundled",
        }
