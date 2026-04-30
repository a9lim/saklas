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
    """Raised when ``saklas serve`` runs against an empty dist dir.

    The wheel ships a pre-built bundle; this only fires in source-tree
    installs that haven't run ``cd webui && npm run build`` yet.  Pass
    ``--no-web`` to skip the dashboard mount when the bundle is missing
    on purpose.
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
    mount would shadow the first).  CLI default-on (``saklas serve``);
    ``--no-web`` opts out for production / proxied deployments.
    Library callers using ``create_app`` directly default-off.
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
    # registration order; create_app mounts this last.
    dist_root = dist.resolve()
    index_html = dist_root / "index.html"

    # Build an allowlist of top-level dist files at mount time
    # (favicon.ico, manifest.json, robots.txt, …).  ``full_path`` from
    # the request is then only ever used as a dict key — never as a
    # path component — so ``..`` traversal and absolute-path injection
    # are structurally impossible.  Nested paths fall through to the
    # SPA shell; hashed bundle assets are already served by the
    # StaticFiles mount on ``/assets``.
    top_level_files: dict[str, Path] = {
        p.name: p
        for p in dist_root.iterdir()
        if p.is_file() and p.name != "index.html"
    }

    @app.get("/{full_path:path}", include_in_schema=False)
    async def _spa_fallback(full_path: str) -> FileResponse:
        direct = top_level_files.get(full_path)
        if direct is not None:
            return FileResponse(str(direct))
        return FileResponse(str(index_html))
