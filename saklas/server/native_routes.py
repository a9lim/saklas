"""Registrar for the native ``/saklas/v1/*`` route tree."""

from __future__ import annotations

from fastapi import FastAPI


def register_saklas_routes(app: FastAPI) -> None:
    """Mount the native ``/saklas/v1/*`` tree onto ``app``."""

    from saklas.server.manifold_routes import register_manifold_routes

    register_manifold_routes(app)

    from saklas.server.template_routes import register_template_routes

    register_template_routes(app)

    from saklas.server.session_routes import register_session_routes

    register_session_routes(app)

    from saklas.server.tree_routes import register_tree_routes

    register_tree_routes(app)

    from saklas.server.vector_routes import register_vector_routes

    register_vector_routes(app)

    from saklas.server.probe_routes import register_probe_routes

    register_probe_routes(app)

    from saklas.server.traits_routes import register_traits_routes

    register_traits_routes(app)

    from saklas.server.lens_routes import register_lens_routes

    register_lens_routes(app)

    from saklas.server.sae_routes import register_sae_routes

    register_sae_routes(app)

    from saklas.server.ws_stream import register_ws_stream

    register_ws_stream(app)
