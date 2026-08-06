"""Native probe route group — the read-side counterpart to manifold steering.

One unified collection under ``/saklas/v1/sessions/{id}/probes`` covering every
probe shape.  A Monitor probe is a :class:`~saklas.core.manifold.Manifold` — a
2-node concept axis is the rank-1 case, a discover / curved fit the rank-R case
— attached via ``add_probe`` / ``remove_probe``; a ``jlens/<word>`` or
``sae/<id>`` selector lands on the session's readout-channel registries instead
and is listed alongside.

Live per-token scoring during generation rides the WS ``measurements`` envelope
and the OpenAI / Ollama reading extensions, not a route here.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response

from saklas.core.errors import SaklasError
from saklas.core.manifold import manifold_is_affine
from saklas.core.monitor_attach import AttachedManifoldProbe
from saklas.core.session import SaklasSession
from saklas.io.probes_bootstrap import load_default_manifolds
from saklas.server.native_common import NativeRequest, resolve_session_id
from saklas.server.response_models import (
    GeometryProbeInfo,
    LensProbeInfo,
    ProbeDefaultsResponse,
    ProbeGeometryResponse,
    ProbeInfo,
    ProbeListResponse,
    SaeProbeInfo,
)


class ProbeRequest(NativeRequest):
    """Body for ``POST /saklas/v1/sessions/{id}/probes``."""

    selector: str
    name: str | None = None
    top_n: int | None = None


def _probe_info(name: str, probe: AttachedManifoldProbe) -> GeometryProbeInfo:
    """Serialize one attached geometry probe (any rank) to JSON for the wire.

    Rows are discriminated by an explicit ``family`` key and carry only what
    their family can actually produce.  A client keys off ``family``, not off
    an out-of-band ``lens: true`` flag against an otherwise geometry-shaped
    row.
    """
    manifold = probe.manifold
    domain_spec = manifold.domain.to_spec()
    intrinsic_dim = int(manifold.domain.intrinsic_dim)
    nc = manifold.node_coords
    node_coords = nc.tolist() if nc is not None else None
    is_affine = manifold_is_affine(manifold)
    return {
        "family": "geometry",
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


def _lens_probe_info(name: str, spec: dict[str, Any]) -> LensProbeInfo:
    """Serialize one pinned J-lens token probe (readout channel) to JSON.

    There is no subspace behind a readout probe, so the row carries no
    geometry fields: the invented ``manifold: "jlens"`` (no such manifold
    exists), ``top_n: 0``, ``domain: {}``, ``is_affine``, ``node_coords``,
    ``node_count`` and the ``node_labels: [word]`` stand-in are gone.  What
    remains is real: the fitted layer set, the one strength axis, and the
    word / vocabulary id the channel reads.
    """
    return {
        "family": "lens",
        "name": name,
        "layers": sorted(int(l) for l in spec.get("layers", ())),
        "intrinsic_dim": 1,
        "feature_space": "readout",
        "word": spec.get("word", ""),
        "token_id": spec.get("token_id"),
    }


def _lens_probe_specs(session: SaklasSession) -> dict[str, dict[str, Any]]:
    """Snapshot the session's pinned lens-probe registry."""
    return session.lens.specs()


def _sae_probe_info(name: str, spec: dict[str, Any]) -> SaeProbeInfo:
    """Serialize one pinned SAE feature probe (encoder readout channel).

    Same discipline as the lens row: no invented ``manifold``/``domain``/
    ``node_*`` geometry.  A feature probe reads one encoder channel at one
    resident layer.
    """
    return {
        "family": "sae",
        "name": name,
        "layers": [int(spec.get("layer", 0))],
        "intrinsic_dim": 1,
        "feature_space": "sae-readout",
        "feature_id": int(spec.get("feature_id", -1)),
        "label": spec.get("label"),
        # The strength unit — the reading is ``activation / max_act`` when
        # set, raw activation when null (no Neuronpedia metadata).
        "max_act": spec.get("max_act"),
    }


def _sae_probe_specs(session: SaklasSession) -> dict[str, dict[str, Any]]:
    return session.sae.specs()


def register_probe_routes(app: FastAPI) -> None:
    """Mount the unified probe listing + defaults + geometry + attach / detach."""
    session = app.state.session

    @app.get("/saklas/v1/sessions/{session_id}/probes")
    def list_probes(session_id: str) -> ProbeListResponse:
        resolve_session_id(session_id)
        attached = session.monitor.attached_probes()
        rows: list[ProbeInfo] = [
            _probe_info(name, probe) for name, probe in attached.items()
        ]
        rows.extend(
            _lens_probe_info(name, spec)
            for name, spec in _lens_probe_specs(session).items()
        )
        rows.extend(
            _sae_probe_info(name, spec)
            for name, spec in _sae_probe_specs(session).items()
        )
        return {"probes": rows}

    @app.get("/saklas/v1/sessions/{session_id}/probes/defaults")
    def list_default_probes(session_id: str) -> ProbeDefaultsResponse:
        resolve_session_id(session_id)
        return {"defaults": load_default_manifolds()}

    @app.get("/saklas/v1/sessions/{session_id}/probes/{name:path}/geometry")
    def probe_geometry(session_id: str, name: str) -> ProbeGeometryResponse:
        """Static geometry for the dashboard probe-inspector plot.

        Per-layer node centroids + neutral + (rank>=3) a top-3 PCA rotation +
        the curve/surface overlay for a curved fit — all in the whitened frame
        the reads use, so the per-token live point (the reading's
        ``subspace_coords_per_layer``) overlays directly.  ``defaults`` is
        registered before this greedy ``{name:path}`` route so it still resolves.
        """
        resolve_session_id(session_id)
        try:
            return session.monitor.probe_geometry(name)
        except KeyError as e:
            raise HTTPException(404, f"probe '{name}' not attached") from e

    @app.post("/saklas/v1/sessions/{session_id}/probes", status_code=201)
    def add_probe(session_id: str, req: ProbeRequest) -> ProbeInfo:
        resolve_session_id(session_id)
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
        except SaklasError as e:
            # e.g. LensNotFittedError from a jlens/<word> probe → 404 with
            # the fit command.
            status, text = e.user_message()
            raise HTTPException(status, text) from e
        lens_specs = _lens_probe_specs(session)
        if registered in lens_specs:
            return _lens_probe_info(registered, lens_specs[registered])
        sae_specs = _sae_probe_specs(session)
        if registered in sae_specs:
            return _sae_probe_info(registered, sae_specs[registered])
        attached = session.monitor.attached_probes()
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
        resolve_session_id(session_id)
        if (
            name not in session.monitor.probe_names
            and name not in _lens_probe_specs(session)
            and name not in _sae_probe_specs(session)
        ):
            raise HTTPException(404, f"probe '{name}' not attached")
        session.remove_probe(name)
        return Response(status_code=204)
