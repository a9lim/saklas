"""Native experiment route group."""

from __future__ import annotations

import asyncio

from fastapi import FastAPI, HTTPException

from saklas.server.app import acquire_session_lock
from saklas.server.experiment_models import ExperimentFanRequest
from saklas.server.native_common import resolve_session_id
from saklas.server.ws_models import build_sampling


def register_experiment_routes(app: FastAPI) -> None:
    """Mount experiment execution routes."""
    session = app.state.session

    @app.post("/saklas/v1/sessions/{session_id}/experiments/fan")
    async def run_experiment_fan(session_id: str, req: ExperimentFanRequest):
        """Run an alpha grid as loom siblings and return a RunSet summary."""
        resolve_session_id(session_id)

        if not req.grid:
            raise HTTPException(400, "grid must be non-empty")
        for name, alphas in req.grid.items():
            if not alphas:
                raise HTTPException(400, f"grid['{name}'] must be non-empty")
        sampling_cfg = build_sampling(req.sampling)

        async with acquire_session_lock(session) as acquired:
            if not acquired:
                raise HTTPException(503, "session locked")
            runset = await asyncio.to_thread(
                session.generate_sweep,
                req.prompt,
                req.grid,
                base_steering=req.base_steering,
                sampling=sampling_cfg,
                thinking=req.thinking,
                stateless=False,
                raw=req.raw,
            )
        rows = []
        for idx, result in enumerate(runset):
            readings_summary: dict[str, float] = {}
            for probe_name, reading in (
                getattr(result, "readings", {}) or {}
            ).items():
                # ``per_generation`` samples and ``mean`` are per-axis
                # coordinate tuples now; the scalar summary reads axis 0.
                per_generation = getattr(reading, "per_generation", None)
                sample = (
                    per_generation[-1]
                    if per_generation
                    else getattr(reading, "mean", None)
                )
                value = sample[0] if sample else 0.0
                readings_summary[probe_name] = round(float(value), 6)
            rows.append({
                "idx": idx,
                "alpha_values": runset.grid[idx] if idx < len(runset.grid) else {},
                "node_id": (
                    runset.node_ids[idx] if idx < len(runset.node_ids) else None
                ),
                "result": {
                    "text": result.text,
                    "token_count": result.token_count,
                    "tok_per_sec": result.tok_per_sec,
                    "elapsed": result.elapsed,
                    "finish_reason": result.finish_reason,
                    "applied_steering": result.applied_steering,
                    "readings": readings_summary,
                },
            })
        return {
            "kind": runset.kind,
            "total": len(runset),
            "node_ids": runset.node_ids,
            "rows": rows,
        }
