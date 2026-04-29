"""Mount the pre-built Svelte+Vite bundle from ``saklas/web/dist/``.

The bundle is a single-page app — every unknown route falls back to
``index.html`` so client-side routing works.  Asset hashing is handled
by Vite; aggressive caching of ``/assets/*`` is safe because filenames
are content-hashed.
"""
from __future__ import annotations

from importlib import resources as _resources
from pathlib import Path

from fastapi import FastAPI


class WebUINotBuilt(RuntimeError):
    """Raised when ``saklas serve --web`` runs against an empty dist dir.

    The wheel ships a pre-built bundle; this only fires in source-tree
    installs that haven't run ``cd webui && npm run build`` yet.
    """


def dist_path() -> Path:
    """Return the on-disk path to the bundled web UI assets.

    Resolved through ``importlib.resources`` so it works for both
    editable and wheel installs.  The traversable returned is always a
    ``Path`` because saklas itself ships as a regular filesystem
    package (not a zip).
    """
    files = _resources.files("saklas.web")
    dist = Path(str(files / "dist"))
    return dist


def register_web_routes(app: FastAPI) -> None:
    """Mount the SPA bundle at ``/`` with an SPA-fallback to index.html.

    Idempotent — calling it twice on the same app is a no-op (the second
    mount would shadow the first).  Caller (server.create_app) is
    expected to gate this on a ``--web`` opt-in flag so production API
    deployments don't pay the static-files mount cost.
    """
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    dist = dist_path()
    if not dist.exists() or not (dist / "index.html").is_file():
        raise WebUINotBuilt(
            f"web UI bundle not found at {dist}. "
            f"Build it with: cd webui && npm ci && npm run build"
        )

    # Mount /assets/* directly on StaticFiles; fall back to index.html
    # for any other path so client-side routing in the SPA works.
    assets_dir = dist / "assets"
    if assets_dir.is_dir():
        app.mount(
            "/assets",
            StaticFiles(directory=str(assets_dir)),
            name="saklas-web-assets",
        )

    @app.get("/", include_in_schema=False)
    async def _root() -> FileResponse:
        return FileResponse(str(dist / "index.html"))

    # Catch-all for any other top-level route the SPA needs to handle —
    # e.g. /chat, /lab, /vectors/<name>.  Won't shadow any earlier
    # /api/*, /v1/*, /saklas/v1/* routes because FastAPI evaluates in
    # registration order; --web opt-in mounts this last.
    @app.get("/{full_path:path}", include_in_schema=False)
    async def _spa_fallback(full_path: str) -> FileResponse:
        # Direct file under dist/ wins (favicon, manifest, etc).  Anything
        # else is an SPA route; serve index.html.
        candidate = dist / full_path
        if candidate.is_file():
            return FileResponse(str(candidate))
        return FileResponse(str(dist / "index.html"))
