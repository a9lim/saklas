"""Native ``/saklas/v1/manifold-probes`` routes."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel


class ManifoldProbeRequest(BaseModel):
    """Body for ``POST /saklas/v1/manifold-probes``."""

    selector: str
    name: str | None = None
    top_n: int | None = None


def _manifold_probe_info(name: str, probe: Any) -> dict[str, Any]:
    """Serialize one attached manifold probe to JSON for the wire."""
    manifold = probe.manifold
    try:
        domain_spec = manifold.domain.to_spec()
    except Exception:
        domain_spec = {}
    try:
        intrinsic_dim = int(manifold.domain.intrinsic_dim)
    except Exception:
        intrinsic_dim = 0
    return {
        "name": name,
        "manifold": manifold.name,
        "top_n": int(probe.top_n),
        "layers": sorted(manifold.layers.keys()),
        "node_labels": list(manifold.node_labels),
        "node_count": len(manifold.node_labels),
        "domain": domain_spec,
        "intrinsic_dim": intrinsic_dim,
        "feature_space": manifold.feature_space,
    }


def register_manifold_probe_routes(app: FastAPI) -> None:
    """Mount the read-side manifold probe route group."""
    session = app.state.session

    @app.get("/saklas/v1/manifold-probes")
    def list_manifold_probes():
        try:
            attached = session.manifold_monitor.attached_probes()
        except Exception:
            attached = {}
        return {
            "probes": [
                _manifold_probe_info(name, probe)
                for name, probe in attached.items()
            ],
        }

    @app.post("/saklas/v1/manifold-probes", status_code=201)
    def add_manifold_probe(req: ManifoldProbeRequest):
        if not req.selector or not req.selector.strip():
            raise HTTPException(400, "selector must not be empty")
        top_n = req.top_n if req.top_n and req.top_n > 0 else 3
        try:
            registered_name = session.add_manifold_probe(
                req.selector, as_name=req.name, top_n=top_n,
            )
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))
        except ValueError as e:
            raise HTTPException(400, str(e))
        attached = session.manifold_monitor.attached_probes()
        probe = attached.get(registered_name)
        if probe is None:
            raise HTTPException(
                500,
                f"manifold probe '{registered_name}' attach did not register",
            )
        return _manifold_probe_info(registered_name, probe)

    @app.delete(
        "/saklas/v1/manifold-probes/{name:path}",
        status_code=204,
    )
    def remove_manifold_probe(name: str):
        try:
            attached_names = session.manifold_monitor.probe_names
        except Exception:
            attached_names = []
        if name not in attached_names:
            raise HTTPException(404, f"manifold probe '{name}' not attached")
        session.remove_manifold_probe(name)
        return JSONResponse(status_code=204, content=None)
