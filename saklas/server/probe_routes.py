"""Native probe route group — the read-side counterpart to manifold steering.

One unified collection under ``/saklas/v1/sessions/{id}/probes``.  Every probe
is a :class:`~saklas.core.manifold.Manifold` — a 2-node concept axis is the
rank-1 case, a discover / curved fit the rank-R case — attached on the session's
single :class:`~saklas.core.monitor.Monitor` via ``add_probe`` / ``remove_probe``.
The pre-4.0 split (vector probes by name under ``/probes``; manifold probes by
selector under ``/manifold-probes``) collapsed with the monitor unification.

One-shot text scoring (``POST .../probe`` / ``.../manifold-probe``) was removed in
4.0: scoring out of generation context required re-rendering arbitrary text in a
non-conversational regime, which the conversational (A2) capture model retires.
Live per-token scoring during generation rides the traits SSE stream and the
WS / OpenAI / Ollama reading extensions.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from saklas.core.session import _manifold_is_affine
from saklas.server import saklas_api as _api
from saklas.server.saklas_api import _resolve_session_id


class ProbeRequest(BaseModel):
    """Body for ``POST /saklas/v1/sessions/{id}/probes``."""

    selector: str
    name: str | None = None
    top_n: int | None = None


def _probe_info(name: str, probe: Any) -> dict[str, Any]:
    """Serialize one attached probe (any rank) to JSON for the wire."""
    manifold = probe.manifold
    try:
        domain_spec = manifold.domain.to_spec()
    except Exception:
        domain_spec = {}
    try:
        intrinsic_dim = int(manifold.domain.intrinsic_dim)
    except Exception:
        intrinsic_dim = 0
    try:
        nc = manifold.node_coords
        node_coords = nc.tolist() if nc is not None else None
    except Exception:
        node_coords = None
    try:
        is_affine = _manifold_is_affine(manifold)
    except Exception:
        is_affine = False
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
        # Flat (affine) probes are the subspace family — a 2-node concept axis
        # through the rank-8 personas fan; curved fits are the manifold family.
        # The client classifies subspace-vs-manifold off this single flag.
        "is_affine": is_affine,
        # Per-node authoring/display layout (K, n), aligned with node_labels;
        # backs the client mini-map node dots + per-token trajectory lookup.
        # None on an unfitted discover manifold (no per-model layout yet).
        "node_coords": node_coords,
    }


def register_probe_routes(app: FastAPI) -> None:
    """Mount the unified probe listing + attach / detach routes."""
    session = app.state.session

    @app.get("/saklas/v1/sessions/{session_id}/probes")
    def list_probes(session_id: str):
        _resolve_session_id(session, session_id)
        try:
            attached = session._monitor.attached_probes()
        except Exception:
            attached = {}
        return {
            "probes": [
                _probe_info(name, probe) for name, probe in attached.items()
            ],
        }

    @app.get("/saklas/v1/sessions/{session_id}/probes/defaults")
    def list_default_probes(session_id: str):
        _resolve_session_id(session, session_id)
        return {"defaults": _api.load_defaults()}

    @app.get("/saklas/v1/sessions/{session_id}/probes/{name:path}/geometry")
    def probe_geometry(session_id: str, name: str):
        """Static geometry for the dashboard probe-inspector plot.

        Per-layer node centroids + neutral + (rank>=3) a top-3 PCA rotation +
        the curve/surface overlay for a curved fit — all in the whitened frame
        the reads use, so the per-token live point (the reading's
        ``subspace_coords_per_layer``) overlays directly.  ``defaults`` is
        registered before this greedy ``{name:path}`` route so it still resolves.
        """
        _resolve_session_id(session, session_id)
        try:
            return session._monitor.probe_geometry(name)
        except KeyError as e:
            raise HTTPException(404, f"probe '{name}' not attached") from e

    @app.post("/saklas/v1/sessions/{session_id}/probes", status_code=201)
    def add_probe(session_id: str, req: ProbeRequest):
        _resolve_session_id(session, session_id)
        if not req.selector or not req.selector.strip():
            raise HTTPException(400, "selector must not be empty")
        top_n = req.top_n if req.top_n and req.top_n > 0 else 3
        try:
            registered = session.add_probe(
                req.selector, as_name=req.name, top_n=top_n,
            )
        except FileNotFoundError as e:
            raise HTTPException(404, str(e)) from e
        except (KeyError, ValueError) as e:
            raise HTTPException(400, str(e)) from e
        attached = session._monitor.attached_probes()
        probe = attached.get(registered)
        if probe is None:
            raise HTTPException(
                500, f"probe '{registered}' attach did not register",
            )
        return _probe_info(registered, probe)

    @app.delete(
        "/saklas/v1/sessions/{session_id}/probes/{name:path}", status_code=204,
    )
    def remove_probe(session_id: str, name: str):
        _resolve_session_id(session, session_id)
        if name not in session._monitor.probe_names:
            raise HTTPException(404, f"probe '{name}' not attached")
        session.remove_probe(name)
        return Response(status_code=204)
