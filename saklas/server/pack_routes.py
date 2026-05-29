"""Native ``/saklas/v1/packs`` route group."""

from __future__ import annotations

import asyncio

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


class InstallPackRequest(BaseModel):
    target: str
    as_: str | None = Field(default=None, alias="as")
    force: bool = False
    statements_only: bool = False

    model_config = {"populate_by_name": True}


def register_pack_routes(app: FastAPI) -> None:
    """Mount pack listing/search/install/delete routes."""
    session = app.state.session

    @app.get("/saklas/v1/packs")
    def list_packs():
        from saklas.io.cache_ops import list_concepts as _list_concepts
        from saklas.io.paths import safe_model_id as _safe_id

        result = _list_concepts(None, hf=False)
        sid = _safe_id(session.model_id)
        return {
            "packs": [
                {
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
                for r in result.installed
            ],
        }

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

    @app.post("/saklas/v1/packs")
    async def install_pack(req: InstallPackRequest):
        from saklas.io.cache_ops import InstallConflict, install as _install

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
        return {
            "target": req.target,
            "installed_at": str(dst),
            "statements_only": req.statements_only,
        }

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
