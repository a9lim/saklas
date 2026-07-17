"""Native profiles route group (``/saklas/v1/sessions/{id}/profiles/*``).

Steering-profile lifecycle under the session: list / profile JSON / delete /
extract / bake, plus the
cross-layer whitened ``pairwise`` matrix and the N×N ``correlation``
matrix across loaded profiles and probes.
"""

# pyright: reportUnusedFunction=false

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response

from saklas.core.profile import Profile
from saklas.server.app import acquire_session_lock
from saklas.server.native_common import resolve_session_id
from saklas.server.profile_models import (
    BakeProfileRequest,
    ExtractRequest,
    extract_registry_name,
    profile_to_json,
)
from saklas.server.sse import ProgressCallback, progress_sse_response


def register_profile_routes(app: FastAPI) -> None:
    """Mount the ``/sessions/{id}/profiles/*`` + ``/correlation`` routes."""
    session = app.state.session

    @app.get("/saklas/v1/sessions/{session_id}/profiles")
    def list_profiles(session_id: str):
        resolve_session_id(session_id)
        return {
            "profiles": [
                profile_to_json(name, Profile(tensors))
                for name, tensors in sorted(session.profiles.items())
            ],
        }

    @app.get("/saklas/v1/sessions/{session_id}/profiles/pairwise")
    def pairwise_compare(session_id: str, a: str, b: str):
        """Cross-layer whitened cosine matrix between two named profiles / probes.

        Query: ``?a=<name>&b=<name>``.  Each cell ``matrix[i][j]`` is the
        Mahalanobis cosine between vector ``a``'s layer ``layers_a[i]`` and
        vector ``b``'s layer ``layers_b[j]``.  Output:

            {
              "a": "honest",
              "b": "warm",
              "metric": "mahalanobis",
              "layers_a": [0, 5, ...],
              "layers_b": [0, 5, ...],
              "matrix": [[1.0, 0.41, ...], [0.13, 0.92, ...], ...],
              "model": "google/gemma-3-4b-it",
            }

        Pool unions ``session.profiles`` and ``monitor.probe_names`` (same
        as :func:`correlation_matrix`) so probes that were never
        registered as steering profiles still resolve.  Near-zero layer
        norms land as ``None`` so the client can render them as empty
        cells.  The matrix is the structural signal the webui
        pairwise-compare heatmap reads, distinct from the aggregate
        scalar :meth:`Profile.cosine_similarity` returns.

        **Metric.**  Mahalanobis-only: each cell is whitened in the
        per-model :class:`LayerWhitener` metric, downweighting alignment
        that is merely shared high-variance base-rate structure.
        Mahalanobis cosine is a *single-layer* (single-``Σ``) operation,
        but this matrix is **cross-layer** (``layers_a × layers_b``), so
        each cell is whitened in vector ``a``'s row-layer frame
        (``⟨v_a, v_b⟩_M / (‖v_a‖_M ‖v_b‖_M)`` under ``Σ_{La}^{-1}``):
        exact on the layer-aligned diagonal (``La == Lb``), an A-frame
        read off it.  There is no Euclidean path: a missing whitener, or
        one that doesn't cover every row-layer of ``a``, is a 409 (the
        neutral activation cache must be regenerated).

        Registered *before* ``GET /profiles/{name}`` so the literal path
        wins the routing match — Starlette matches in registration order
        and ``pairwise`` would otherwise be swallowed by ``{name}``.
        """
        resolve_session_id(session_id)

        # CPU snapshots only (cached, built once under the exclusive-GPU lock)
        # so this endpoint never issues an MPS->CPU copy on its threadpool
        # thread that could race a concurrent model op on PyTorch's single
        # global, non-thread-safe command buffer (which aborts the process).
        prof_a = session.analytics_profile(a)
        prof_b = session.analytics_profile(b)
        unavailable = [n for n, p in ((a, prof_a), (b, prof_b)) if p is None]
        if unavailable:
            known = set(session.analytics_names())
            missing = [n for n in unavailable if n not in known]
            if missing:
                raise HTTPException(404, f"names not loaded: {missing}")
            # Registered, but a model op holds the GPU and no snapshot is
            # cached yet — retryable rather than a hard miss.
            raise HTTPException(
                409,
                "a model operation is in flight; the comparison snapshot is "
                "not ready yet — retry shortly",
            )
        assert prof_a is not None and prof_b is not None  # guarded above
        layers_a = sorted(prof_a.keys())
        layers_b = sorted(prof_b.keys())

        # Precompute fp32 vectors.  Both sides are forced to CPU so a
        # cross-device pair (e.g. an actively-steered vector hooked on MPS
        # vs. a disk-loaded peer on CPU) computes cleanly rather than
        # raising on the dot — hidden_dim × layer-count is small enough
        # that the device round-trip is free relative to the request
        # budget.
        import torch as _torch
        vecs_a: list["_torch.Tensor"] = []
        for L in layers_a:
            v = prof_a[L].float().cpu()
            vecs_a.append(v)
        vecs_b: list["_torch.Tensor"] = []
        for L in layers_b:
            v = prof_b[L].float().cpu()
            vecs_b.append(v)

        # Resolve the whitener (Mahalanobis-only).  ``session.whitener`` is
        # a lazy property (builds from the neutral-activation cache on first
        # access).  It must cover every row-layer of ``a`` (each row is
        # framed in its row-layer's covariance) — there is no Euclidean
        # fallback, so a missing / non-covering whitener is a 409.
        whitener = session.whitener
        if whitener is None or not whitener.covers_all(layers_a):
            raise HTTPException(
                409,
                "pairwise compare requires a Mahalanobis whitener covering "
                f"every row-layer {layers_a} of '{a}'; regenerate the neutral "
                "activation cache for this model (the Euclidean path is gone)",
            )

        matrix: list[list[float | None]] = []
        for la, va in zip(layers_a, vecs_a, strict=True):
            row: list[float | None] = [None] * len(vecs_b)
            si_va = whitener.apply_inv(la, va).float()
            aa = max(
                float(_torch.dot(va.reshape(-1), si_va.reshape(-1)).item()),
                0.0,
            )
            if aa < 1e-12:
                matrix.append(row)
                continue

            # Batch every compatible B-vector in this row through one
            # Woodbury application.  Different hidden dims are a
            # model-mismatch bug; keep those cells empty so a partially
            # misconfigured pool still renders the cells that line up.
            b_indices: list[int] = []
            b_rows: list["_torch.Tensor"] = []
            for j, vb in enumerate(vecs_b):
                if va.shape == vb.shape:
                    b_indices.append(j)
                    b_rows.append(vb)
            if not b_rows:
                matrix.append(row)
                continue

            block = _torch.stack(b_rows)
            si_block = whitener.apply_inv(la, block).float()
            bb = (block.reshape(len(b_rows), -1) * si_block.reshape(
                len(b_rows), -1,
            )).sum(dim=1).clamp_min(0.0)
            uv = (block.reshape(len(b_rows), -1) * si_va.reshape(1, -1)).sum(dim=1)
            denom = (bb * aa).sqrt()
            valid = denom >= 1e-12
            cosines = uv / denom.clamp_min(1e-12)
            for j, ok, cos in zip(b_indices, valid.tolist(), cosines.tolist(), strict=True):
                if ok:
                    row[j] = round(float(cos), 6)
            matrix.append(row)

        return {
            "a": a,
            "b": b,
            "metric": "mahalanobis",
            "layers_a": layers_a,
            "layers_b": layers_b,
            "matrix": matrix,
            "model": session.model_id,
        }

    @app.get("/saklas/v1/sessions/{session_id}/profiles/{name}")
    def get_profile(session_id: str, name: str):
        resolve_session_id(session_id)
        profiles = session.profiles
        if name not in profiles:
            raise HTTPException(404, f"profile '{name}' not found")
        return profile_to_json(name, Profile(profiles[name]))

    @app.get("/saklas/v1/sessions/{session_id}/correlation")
    def correlation_matrix(session_id: str, names: str | None = None):
        """N×N magnitude-weighted cosine matrix across loaded vectors and probes.

        Query: ``?names=a,b,c`` restricts the matrix to a subset; default
        is every steering vector AND every active probe currently
        registered in the session, deduplicated by name (a registered
        steering vector wins over a same-named probe — they share the
        underlying tensor).  Output:

            {
              "names": ["a", "b", ...],
              "matrix": {"a": {"a": 1.0, "b": 0.42, ...}, ...},
              "layers_shared": {"a__b": 36, ...}
            }

        Used by the web UI's correlation overlay — heavy compute lives
        server-side so the client doesn't have to ship full per-layer
        tensors over the wire.
        """
        resolve_session_id(session_id)

        # Read-side pool of **CPU snapshots**, one per direction (a steering
        # vector wins a same-named probe, matching the historical dedup).
        # ``session.analytics_profile`` does the device->host copy once, under
        # the exclusive-GPU lock, and caches it — so this polled endpoint
        # never issues an MPS->CPU copy on its threadpool thread that could
        # race a concurrent model op on PyTorch's single global, non-thread-
        # safe command buffer (which aborts the whole process).
        available = session.analytics_names()
        if names is not None and names.strip():
            requested = [n.strip() for n in names.split(",") if n.strip()]
            missing = [n for n in requested if n not in available]
            if missing:
                raise HTTPException(404, f"names not loaded: {missing}")
            ordered = requested
        else:
            ordered = available
        # A name whose CPU snapshot isn't ready (a model op currently holds
        # the GPU) is left out of the pool; its cells render null until a
        # later poll builds it once the GPU is free.
        pool: dict[str, "Profile"] = {}
        for name in ordered:
            snap = session.analytics_profile(name)
            if snap is not None:
                pool[name] = snap

        # Mahalanobis-only: ``cosine_similarity`` requires a whitener
        # covering each pair's shared layers.  Resolve it once; a missing
        # whitener is a 409 (regenerate the neutral cache).  A pair the
        # whitener doesn't fully cover still raises inside the loop and
        # lands as ``None`` for that cell.
        whitener = session.whitener
        if whitener is None:
            raise HTTPException(
                409,
                "correlation requires a Mahalanobis whitener; regenerate the "
                "neutral activation cache for this model (the Euclidean path "
                "is gone)",
            )

        # Request-scope Mahalanobis cache.  The old upper-triangle loop called
        # ``Profile.cosine_similarity`` per pair, which reapplied ``Σ⁻¹`` to
        # the same profile-layer O(N) times as the dashboard roster grew.
        # Cache ``(v, Σ⁻¹v, vᵀΣ⁻¹v)`` once per available name/layer and assemble
        # the symmetric matrix from those factors.
        import math as _math
        import torch as _torch
        white_cache: dict[str, dict[int, tuple[Any, Any, float]]] = {}
        for name, prof in pool.items():
            entries: dict[int, tuple[Any, Any, float]] = {}
            for layer, tensor in prof.items():
                vec = tensor.float().cpu().reshape(-1)
                si_vec = whitener.apply_inv(layer, vec).float().reshape(-1)
                norm_sq = max(float(_torch.dot(vec, si_vec).item()), 0.0)
                entries[int(layer)] = (vec, si_vec, norm_sq)
            white_cache[name] = entries

        matrix: dict[str, dict[str, float | None]] = {a: {} for a in ordered}
        layers_shared: dict[str, int] = {}
        for i, a in enumerate(ordered):
            for j, b in enumerate(ordered):
                if j < i:
                    matrix[a][b] = matrix[b][a]
                    continue
                if i == j:
                    matrix[a][b] = 1.0
                    continue
                if a not in pool or b not in pool:
                    # Snapshot not ready (GPU busy) — null cell this poll.
                    matrix[a][b] = None
                    continue
                shared = sorted(
                    set(pool[a].keys()) & set(pool[b].keys())
                )
                # Pair key sorted alphabetically so a__b == b__a in the lookup.
                key = "__".join(sorted([a, b]))
                layers_shared[key] = len(shared)
                if not shared or not whitener.covers_all(shared):
                    matrix[a][b] = None
                    continue
                entries_a = white_cache.get(a, {})
                entries_b = white_cache.get(b, {})
                if any(L not in entries_a or L not in entries_b for L in shared):
                    matrix[a][b] = None
                    continue
                num = 0.0
                den = 0.0
                for layer in shared:
                    vec_a, _si_a, aa = entries_a[layer]
                    _vec_b, si_b, bb = entries_b[layer]
                    if aa < 1e-12 or bb < 1e-12:
                        continue
                    num += float(_torch.dot(vec_a, si_b).item())
                    den += _math.sqrt(aa * bb)
                matrix[a][b] = (
                    round(num / den, 6) if den >= 1e-12 else None
                )
        return {
            "names": ordered,
            "matrix": matrix,
            "layers_shared": layers_shared,
        }

    @app.delete("/saklas/v1/sessions/{session_id}/profiles/{name}", status_code=204)
    def delete_profile(session_id: str, name: str):
        resolve_session_id(session_id)
        if name not in session.profiles:
            raise HTTPException(404, f"profile '{name}' not found")
        session.unsteer(name)
        # Drop the profile from the default steering (if present) so the
        # next request doesn't autoload it back under a stale alpha.
        ds = app.state.default_steering
        if ds is not None and name in ds.alphas:
            from dataclasses import replace as _replace
            new_alphas = {k: v for k, v in ds.alphas.items() if k != name}
            app.state.default_steering = (
                _replace(ds, alphas=new_alphas) if new_alphas else None
            )
        return Response(status_code=204)

    @app.post("/saklas/v1/sessions/{session_id}/extract")
    async def extract_profile(session_id: str, req: ExtractRequest, request: Request):
        resolve_session_id(session_id)

        def _run(on_progress: ProgressCallback) -> tuple[str, Any]:
            return session.extract(
                req.concept, req.baseline,
                on_progress=on_progress,
                sae=req.sae,
                role=req.role, namespace=req.namespace, force=req.force,
            )

        accept = request.headers.get("accept", "application/json")
        if "text/event-stream" in accept:
            async def _job(on_progress: ProgressCallback) -> dict[str, Any]:
                canonical, profile = await asyncio.to_thread(_run, on_progress)
                registry_name = extract_registry_name(canonical, req.namespace)
                session.steer(registry_name, profile)
                return {
                    "done": True,
                    "profile": profile_to_json(registry_name, profile),
                    "canonical": registry_name,
                }

            return progress_sse_response(
                session.lock,
                _job,
                error_message="extract failed",
                log_message=f"extract failed for session={session_id}",
            )

        progress_msgs: list[str] = []
        async with acquire_session_lock(session) as acquired:
            if not acquired:
                raise HTTPException(503, "session locked")
            canonical, profile = await asyncio.to_thread(_run, progress_msgs.append)
            registry_name = extract_registry_name(canonical, req.namespace)
            session.steer(registry_name, profile)
        return {
            "canonical": registry_name,
            "profile": profile_to_json(registry_name, profile),
            "progress": progress_msgs,
        }

    @app.post("/saklas/v1/sessions/{session_id}/profiles/bake")
    async def bake_profile(session_id: str, req: BakeProfileRequest):
        """Merge an expression of installed directions into a baked manifold.

        Wraps :func:`saklas.io.bake.merge_into_manifold` (model-scoped to
        the session's loaded model) — the merge lands a corpus-less
        ``fit_mode="baked"`` manifold — then folds the fitted tensor back to a
        steering Profile and registers it so it's immediately steerable.
        Returns the same profile-JSON shape ``GET /profiles/{name}`` produces.
        """
        from saklas.io.bake import merge_into_manifold, MergeError
        from saklas.io.paths import tensor_filename
        from saklas.io.manifold_tensors import load_manifold
        from saklas.core.capture import folded_directions
        from saklas.server.manifold_routes import _refuse_if_busy
        resolve_session_id(session_id)

        async with acquire_session_lock(session) as acquired:
            if not acquired:
                raise HTTPException(503, "session locked")
            # Refuse (409) while an in-flight extract holds the engine
            # gen-lock — parity with the manifold mutating routes, so a
            # merge can't race a concurrent extraction.
            _refuse_if_busy(session)
            try:
                dst_folder = await asyncio.to_thread(
                    merge_into_manifold,
                    req.name,
                    req.expression,
                    session.model_id,
                    force=True,  # session-driven merges always overwrite
                    strict=False,
                )
            except MergeError:
                # Re-raised through the SaklasError handler (400).
                raise
            tensor_path = dst_folder / tensor_filename(session.model_id)
            if not tensor_path.is_file():
                raise HTTPException(
                    500,
                    f"merge produced no tensor for {session.model_id} at {tensor_path}",
                )

            def _load_folded() -> Profile:
                manifold = load_manifold(str(tensor_path))
                return Profile(folded_directions(manifold))

            profile = await asyncio.to_thread(_load_folded)
            session.steer(req.name, profile)
        return profile_to_json(req.name, profile)
