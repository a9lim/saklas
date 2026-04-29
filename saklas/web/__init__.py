"""Static web UI bundle and FastAPI mount helpers.

The Svelte+Vite source lives at the repo's ``webui/`` directory; the
pre-built bundle lands at ``saklas/web/dist/`` and ships in the wheel
via ``[tool.setuptools.package-data]``.  ``register_web_routes(app)``
mounts ``StaticFiles`` against that bundle and adds an SPA fallback.

Two paths to install / run:

* ``pip install saklas[web]`` brings in the same FastAPI + uvicorn the
  ``[serve]`` extra carries (the static-file mount needs nothing more
  than what the API server already requires).  Then
  ``saklas serve --web <model>`` exposes the dashboard at ``/``.
* From source: ``cd webui && npm ci && npm run build`` regenerates the
  bundle into ``saklas/web/dist/``.  CI verifies the committed bundle
  matches the source tree.
"""
from __future__ import annotations

from saklas.web.routes import dist_path, register_web_routes

__all__ = ["dist_path", "register_web_routes"]
