"""Native saklas HTTP namespace registrar.

Route-specific schemas and serializers live beside their route groups. This
module mounts the native ``/saklas/v1/*`` tree and re-exports the old names for
callers that still import from ``saklas.server.saklas_api``.
"""

from __future__ import annotations

from fastapi import FastAPI

from saklas.core.generation import supports_thinking, thinking_is_optional
from saklas.io.probes_bootstrap import load_default_manifolds as load_defaults
from saklas.server.experiment_models import ExperimentFanRequest
from saklas.server.native_common import (
    SINGLE_SESSION_ID,
    _SINGLE_SESSION_ID,
    _session_aliases,
    _resolve_session_id,
    resolve_session_id,
    session_aliases,
)
from saklas.server.session_models import (
    CreateSessionRequest,
    PatchSessionRequest,
    _default_role_labels,
    _device_dtype,
    _role_support,
    _session_config_dict,
    _session_info,
    _session_model_type,
    default_role_labels,
    device_dtype,
    role_support,
    session_config_dict,
    session_info,
    session_model_type,
)
from saklas.server.tree_models import (
    JointLogprobsRequest,
    TreeBranchRequest,
    TreeDiffRequest,
    TreeEditRequest,
    TreeNavigateRequest,
    TreeNoteRequest,
    TreeStarRequest,
    TreeTranscriptLoadRequest,
    TreeTranscriptRequest,
    _active_path_json,
    _node_json,
    _tree_to_json,
    active_path_json,
    node_json,
    tree_to_json,
)
from saklas.server.vector_models import (
    BakeVectorRequest,
    ExtractRequest,
    LoadVectorRequest,
    _coerce_corpora,
    _extract_registry_name,
    _probe_profile_tensors,
    _profile_to_json,
    coerce_corpora,
    extract_registry_name,
    probe_profile_tensors,
    profile_to_json,
)
from saklas.server.ws_models import (
    WSGenerateMessage,
    WSSamplingParams,
    _build_sampling,
    _per_token_probes,
    _result_to_json,
    build_sampling,
    per_token_probes,
    result_to_json,
)

__all__ = [
    "BakeVectorRequest",
    "CreateSessionRequest",
    "ExperimentFanRequest",
    "ExtractRequest",
    "JointLogprobsRequest",
    "LoadVectorRequest",
    "PatchSessionRequest",
    "SINGLE_SESSION_ID",
    "TreeBranchRequest",
    "TreeDiffRequest",
    "TreeEditRequest",
    "TreeNavigateRequest",
    "TreeNoteRequest",
    "TreeStarRequest",
    "TreeTranscriptLoadRequest",
    "TreeTranscriptRequest",
    "WSGenerateMessage",
    "WSSamplingParams",
    "_SINGLE_SESSION_ID",
    "_active_path_json",
    "_build_sampling",
    "_coerce_corpora",
    "_default_role_labels",
    "_device_dtype",
    "_extract_registry_name",
    "_node_json",
    "_per_token_probes",
    "_probe_profile_tensors",
    "_profile_to_json",
    "_resolve_session_id",
    "_result_to_json",
    "_role_support",
    "_session_aliases",
    "_session_config_dict",
    "_session_info",
    "_session_model_type",
    "_tree_to_json",
    "active_path_json",
    "build_sampling",
    "coerce_corpora",
    "default_role_labels",
    "device_dtype",
    "extract_registry_name",
    "load_defaults",
    "node_json",
    "per_token_probes",
    "probe_profile_tensors",
    "profile_to_json",
    "register_saklas_routes",
    "resolve_session_id",
    "result_to_json",
    "role_support",
    "session_aliases",
    "session_config_dict",
    "session_info",
    "session_model_type",
    "supports_thinking",
    "thinking_is_optional",
    "tree_to_json",
]


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

    from saklas.server.experiment_routes import register_experiment_routes

    register_experiment_routes(app)

    from saklas.server.traits_routes import register_traits_routes

    register_traits_routes(app)

    from saklas.server.lens_routes import register_lens_routes

    register_lens_routes(app)

    from saklas.server.ws_stream import register_ws_stream

    register_ws_stream(app)
