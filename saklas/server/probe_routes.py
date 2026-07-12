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

from saklas.core.errors import SaklasError
from saklas.core.manifold import manifold_is_affine
from saklas.io.probes_bootstrap import load_default_manifolds
from saklas.server.native_common import NativeRequest, resolve_session_id


class ProbeRequest(NativeRequest):
    """Body for ``POST /saklas/v1/sessions/{id}/probes``."""

    selector: str
    name: str | None = None
    top_n: int | None = None


class LiveProbesRequest(NativeRequest):
    """Body for ``POST /saklas/v1/sessions/{id}/probes/live``."""

    enabled: bool


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
        is_affine = manifold_is_affine(manifold)
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


def _lens_probe_info(name: str, spec: dict[str, Any]) -> dict[str, Any]:
    """Serialize one pinned J-lens token probe (readout channel) to JSON.

    Shape-compatible with :func:`_probe_info` (the client keys off the
    ``lens`` discriminator): the ONE coordinate axis is ``strength`` (mean
    band probability, [0, 1]), per-layer traces are ``(p_l,)`` over the
    workspace band — there is no subspace geometry behind a readout probe.
    """
    return {
        "name": name,
        "manifold": "jlens",
        "top_n": 0,
        "layers": sorted(int(l) for l in spec.get("layers", ())),
        "node_labels": [spec.get("word", "")],
        "node_count": 1,
        "domain": {},
        "intrinsic_dim": 1,
        "feature_space": "readout",
        "is_affine": False,
        "node_coords": None,
        "lens": True,
        "word": spec.get("word", ""),
        "token_id": spec.get("token_id"),
    }


def _lens_probe_specs(session: Any) -> dict[str, dict[str, Any]]:
    """The session's pinned lens-probe registry, defensively coerced."""
    specs = getattr(session, "_lens_probes", None)
    return dict(specs) if isinstance(specs, dict) else {}


def _sae_probe_info(name: str, spec: dict[str, Any]) -> dict[str, Any]:
    """Serialize one pinned SAE feature probe (encoder readout channel)."""
    feature_id = int(spec.get("feature_id", -1))
    label = spec.get("label")
    return {
        "name": name,
        "manifold": "sae",
        "top_n": 0,
        "layers": [int(spec.get("layer", 0))],
        "node_labels": [str(label) if label else str(feature_id)],
        "node_count": 1,
        "domain": {},
        "intrinsic_dim": 1,
        "feature_space": "sae-readout",
        "is_affine": False,
        "node_coords": None,
        "sae": True,
        "feature_id": feature_id,
        "label": label,
        # The strength unit — coords are ``activation / max_act`` when set,
        # raw activation when null (no Neuronpedia metadata).
        "max_act": spec.get("max_act"),
    }


def _sae_probe_specs(session: Any) -> dict[str, dict[str, Any]]:
    specs = getattr(session, "_sae_probes", None)
    return dict(specs) if isinstance(specs, dict) else {}


def register_probe_routes(app: FastAPI) -> None:
    """Mount the unified probe listing + attach / detach + live-toggle routes."""
    session = app.state.session

    @app.get("/saklas/v1/sessions/{session_id}/probes")
    def list_probes(session_id: str):
        resolve_session_id(session_id)
        try:
            attached = session.monitor.attached_probes()
        except Exception:
            attached = {}
        rows = [_probe_info(name, probe) for name, probe in attached.items()]
        rows.extend(
            _lens_probe_info(name, spec)
            for name, spec in _lens_probe_specs(session).items()
        )
        rows.extend(
            _sae_probe_info(name, spec)
            for name, spec in _sae_probe_specs(session).items()
        )
        return {"probes": rows}

    @app.post("/saklas/v1/sessions/{session_id}/probes/live")
    async def live_probes_toggle(session_id: str, body: LiveProbesRequest):
        """Toggle live per-token monitor scoring (the CAA live toggle).

        When off, generations run aggregate-only capture: probes still
        report the end-of-gen aggregate, but no per-token stream / loom
        token rows / trait events are produced.  Probe gates are
        unaffected — a gate forces the per-token subset it needs.  Waits
        on the session lock so it never races an in-flight stream.
        """
        from saklas.server.app import acquire_session_lock

        resolve_session_id(session_id)
        async with acquire_session_lock(session) as acquired:
            if not acquired:
                raise HTTPException(503, "session locked")
            enabled = bool(session.set_live_probe_scores(body.enabled))
        return {"enabled": enabled}

    @app.get("/saklas/v1/sessions/{session_id}/probes/defaults")
    def list_default_probes(session_id: str):
        resolve_session_id(session_id)
        return {"defaults": load_default_manifolds()}

    @app.get("/saklas/v1/sessions/{session_id}/probes/{name:path}/geometry")
    def probe_geometry(session_id: str, name: str):
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
    def add_probe(session_id: str, req: ProbeRequest):
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
