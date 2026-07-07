"""Native Jacobian-lens route group — the workspace readout at a loom token.

One read route under ``/saklas/v1/sessions/{id}/lens``: the dashboard's
token-drilldown ``j-lens`` tab asks for the per-layer workspace readout at a
clicked token (``session.jlens_token_readout`` — rebuild the node's prompt
render + raw decode prefix, one capture forward under the node's recipe
steering, ``softmax(W_U · norm(J_l h))`` top-k per fitted layer).  Lens
*fitting* stays CLI-only (``saklas lens fit`` — backward passes, minutes of
wall clock); discovery rides ``jlens_fitted`` on the session info payload
(a path-existence check, never the ~GB lazy artifact load).
"""

# pyright: reportUnusedFunction=false

from __future__ import annotations

import asyncio

from fastapi import FastAPI, HTTPException

from saklas.core.errors import SaklasError
from saklas.core.jlens import LensNotFittedError
from saklas.core.loom import InvalidNodeOperationError, UnknownNodeError
from saklas.server.app import acquire_session_lock
from saklas.server.native_common import resolve_session_id


def _parse_layers(layers: str | None) -> list[int] | str | None:
    """``"3,7,11"`` → ``[3, 7, 11]``; named modes pass through."""
    if layers is None or not layers.strip():
        return None
    lowered = layers.strip().lower()
    if lowered in {"workspace", "band", "sample", "all"}:
        return lowered
    try:
        return [int(part) for part in layers.split(",") if part.strip()]
    except ValueError as e:
        raise HTTPException(
            400,
            f"malformed layers list: {layers!r} "
            "(want csv, workspace, sample, or all)",
        ) from e


def register_lens_routes(app: FastAPI) -> None:
    """Mount the Jacobian-lens read routes."""
    session = app.state.session

    @app.get("/saklas/v1/sessions/{session_id}/lens/token-readout")
    async def lens_token_readout(
        session_id: str,
        node_id: str,
        raw_index: int,
        top_k: int = 8,
        steered: bool = True,
        raw: bool = False,
        layers: str | None = None,
    ):
        """Workspace readout at one decode step of a loom node.

        ``steered`` (default on) replays under the node's recipe steering —
        exact for always-active affine terms, the dominant case; pass
        ``steered=false`` for the unsteered counterfactual read of the same
        token stream.  ``raw`` selects the flat (base-model / raw-buffer)
        render; the client supplies it because raw-ness isn't stamped on
        the node.  ``layers`` restricts the readout (csv or workspace/sample/all);
        default is the fitted workspace band.
        """
        resolve_session_id(session, session_id)
        req_layers = _parse_layers(layers) or "workspace"
        if not 1 <= top_k <= 50:
            raise HTTPException(400, "top_k must be in [1, 50]")
        async with acquire_session_lock(session) as acquired:
            if not acquired:
                raise HTTPException(503, "session locked")
            try:
                out = await asyncio.to_thread(
                    session.jlens_token_readout,
                    node_id,
                    raw_index,
                    layers=req_layers,
                    top_k=top_k,
                    apply_steering=steered,
                    raw=raw,
                )
            except (LensNotFittedError, UnknownNodeError) as e:
                raise HTTPException(404, str(e)) from e
            except InvalidNodeOperationError as e:
                raise HTTPException(400, str(e)) from e
            except ValueError as e:
                raise HTTPException(400, str(e)) from e
            except SaklasError as e:
                # Steering-expression resolution / whitener prerequisites /
                # busy-model races — the family carries its own status.
                status, text = e.user_message()
                raise HTTPException(status, text) from e
        band = set(out["workspace_band"])
        return {
            "node_id": out["node_id"],
            "raw_index": out["raw_index"],
            "token_id": out["token_id"],
            "token_text": out["token_text"],
            "steering": out["steering"],
            "layers": [
                {
                    "layer": layer,
                    "in_band": layer in band,
                    "tokens": [
                        {"token": tok, "id": tid, "logprob": round(lp, 4)}
                        for tok, lp, tid in rows
                    ],
                }
                for layer, rows in sorted(out["readout"].items())
            ],
        }
