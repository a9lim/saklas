"""Native saklas HTTP namespace (``/saklas/v1/*``).

This is the saklas-native resource-tree API, distinct from the OpenAI
(``/v1/*``) and Ollama (``/api/*``) compat shims.  Shape is designed
multi-session — URL-paths carry ``{session_id}`` — but the current impl
is single-session.  The one session has id ``"default"``; both that
literal and the loaded model id resolve to it, everything else 404s.

Killer feature: ``WS /saklas/v1/sessions/{id}/stream`` bidirectional
token + probe co-stream.  Per-token probe readings are pushed inline on
each token event (``ws_events.build_token_event``); the ``done`` event
also carries the full ``per_token_probes`` array, assembled from
``session.last_per_token_scores``.

Old ``/v1/saklas/*`` routes were removed in the same commit that
introduced this file — no aliases.
"""

# pyright: reportUnusedFunction=false

from __future__ import annotations

import time
from operator import itemgetter
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from saklas.core.generation import supports_thinking, thinking_is_optional
from saklas.io.probes_bootstrap import load_default_manifolds as load_defaults  # noqa: F401
from saklas.core.profile import Profile
from saklas.core.results import GenerationResult, RunSet
from saklas.core.sampling import SamplingConfig
from saklas.core.session import SaklasSession
from saklas.core.steering import Steering


_SINGLE_SESSION_ID = "default"


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------


class CreateSessionRequest(BaseModel):
    model: str | None = None
    device: str | None = None
    dtype: str | None = None


class PatchSessionRequest(BaseModel):
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    system_prompt: str | None = None
    thinking: bool | None = None


class ExtractRequest(BaseModel):
    name: str
    source: Any = None
    baseline: str | None = None
    sae: str | None = None
    sae_revision: str | None = None
    # Role-augmented extraction (engine: ``core/role_templates``).  When
    # set, the chat template's assistant-role label is replaced by this
    # slug at extract time, and the same substitution is auto-applied at
    # steer time so the extract baseline matches the steer baseline.
    # Tensor lands under a ``_role-<slug>`` filename suffix and the
    # returned ``canonical`` carries a ``:role-<slug>`` variant tail.
    # Slug must match ``[a-z0-9._-]+``; family must carry a
    # substitutable role header (Qwen / Gemma / Llama / GLM / gpt-oss).
    # Mutually exclusive with ``sae`` at the engine layer.
    role: str | None = None
    # Destination namespace for the extracted concept manifold. ``None``
    # (default) lands under ``~/.saklas/manifolds/local/<canonical>/``.
    # Any other namespace value relocates the folder to
    # ``~/.saklas/manifolds/<namespace>/<canonical>/``; parity with the
    # manifold builder's namespace control.
    namespace: str | None = None
    # Force a fresh extraction even if a cached tensor / node corpus exists at
    # the destination. Default ``False`` keeps the cache-hit short-circuit
    # (instant, no work). Parity with the manifold builder's ``force``
    # overwrite control.
    force: bool = False
    auto_register: bool = Field(True, alias="register")

    model_config = {"populate_by_name": True}


class LoadVectorRequest(BaseModel):
    name: str
    source_path: str


class WSSamplingParams(BaseModel):
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    seed: int | None = None
    stop: list[str] | None = None
    logit_bias: dict[int, float] | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    # Phase 1 logit pass: webui "show alts" toggle wires through this
    # field. 0 (default) inherits the session-level
    # ``return_top_k`` set at startup via ``--top-k-alts`` / YAML;
    # K > 0 overrides per-request, so a webui session can flip alts
    # capture on/off without re-loading the model.  Clamped at the
    # SamplingConfig layer to ``[0, 256]``; pydantic accepts the int
    # and forwards as-is.
    return_top_k: int = 0
    # Native-dashboard opt-in for the heavier layer×probe heatmap payload.
    # Regular API clients can leave this false and still get aggregate
    # per-token probe scores.
    persist_per_layer_scores: bool = False
    # Native-dashboard opt-in for the per-layer whitened subspace coords on each
    # token's probe reading — the probe-inspector live point + fading trail.  Set
    # true only while that inspector is open; forces per-token incremental scoring.
    persist_subspace_coords: bool = False
    # Per-message role-substitution labels (roleplay scaffold).  Ride each
    # generate like ``seed``; stamped onto the produced loom nodes (user
    # turn ← ``user_role``, generated assistant turn ← ``assistant_role``)
    # and rendered per-turn.  ``None`` / empty leaves the standard label.
    user_role: str | None = None
    assistant_role: str | None = None


class WSGenerateMessage(BaseModel):
    type: str
    input: Any = None
    # Steering expression string (shared grammar); pole aliases resolve
    # inside session.steering().
    steering: str | None = None
    sampling: WSSamplingParams | None = None
    thinking: bool | None = None
    stateless: bool = True
    raw: bool = False
    # Loom (v2.3): attach the generated assistant node under a specific
    # tree node, and fan out ``n`` siblings on the same user-parent.
    # ``parent_node_id=None`` falls through to the active node; ``n=1``
    # preserves the v2.2 single-stream protocol.
    parent_node_id: str | None = None
    n: int = 1
    # Loom phase 5: optional recipe-override modifier.  Either a built-in
    # mode string (``"unsteered"``/``"inverted"``/``"reseed"``/``"cool"``/
    # ``"hot"``) or a free-form partial recipe expression (``"seed=42,
    # temperature=1.5"``).  Resolved through ``session.regen_with_modifier``
    # when set; ignored when None.
    recipe_override: Any = None
    # Logit fork (v2.3): regenerate an existing assistant node as a
    # sibling with one token swapped.  When ``fork_node_id`` is set the
    # handler ignores ``input`` / ``steering`` / ``sampling`` / ``n`` —
    # ``session.fork_from_token`` reuses the node's stamped recipe and
    # replays its raw decode sequence up to ``fork_raw_index``, forcing
    # ``fork_alt_token_id`` there before sampling the continuation.
    fork_node_id: str | None = None
    fork_raw_index: int | None = None
    fork_alt_token_id: int | None = None
    # Answer-prefill (v2.3): seed an assistant reply under a user node.
    # When ``prefill_node_id`` is set the handler ignores ``input`` and
    # the ``fork_*`` fields — ``session.prefill_assistant`` tokenizes
    # ``prefill_text`` into a forced decode prefix, lands the result as a
    # sibling assistant under the user node, and forces ``thinking=False``
    # (the prefilled text is the start of the *answer*).  ``steering`` /
    # ``sampling`` / ``n`` are honored as on a normal generate.
    prefill_node_id: str | None = None
    prefill_text: str | None = None
    # Commit (Ctrl+Enter on either surface): land a turn under
    # ``parent_node_id`` without running a decode.  ``commit_role="user"``
    # routes to ``session.append_user_turn`` (active node must not be
    # user); ``commit_role="assistant"`` routes to
    # ``session.append_assistant_turn`` (parent must be a user node, the
    # text becomes the whole turn).  Mutually exclusive with prefill and
    # fork; ``input`` / ``steering`` / ``sampling`` / ``thinking`` / ``n``
    # are ignored.  Both fields must travel together.
    commit_role: Literal["user", "assistant"] | None = None
    commit_text: str | None = None


# --- Loom tree request bodies (phase 2) --------------------------------

class TreeNavigateRequest(BaseModel):
    node_id: str


class TreeEditRequest(BaseModel):
    node_id: str
    text: str


class TreeBranchRequest(BaseModel):
    node_id: str
    text: str = ""
    role: str | None = None


class TreeStarRequest(BaseModel):
    node_id: str
    on: bool = True


class TreeNoteRequest(BaseModel):
    node_id: str
    text: str


class TreeTranscriptRequest(BaseModel):
    node_id: str | None = None


class TreeTranscriptLoadRequest(BaseModel):
    """Body for ``POST /saklas/v1/sessions/{id}/tree/transcript/load`` (phase 5).

    ``yaml`` is the full transcript YAML produced by the export route.
    ``mode`` chooses the attach point: ``"default"`` attaches as a fresh
    branch off root, ``"here"`` attaches at the active node, ``"merge"``
    walks for the deepest user-turn match and attaches the divergent
    tail there.  ``strict`` refuses the load on any probe-hash drift.
    """

    yaml: str
    mode: str = "default"
    strict: bool = False


class TreeDiffRequest(BaseModel):
    """Body for ``POST /saklas/v1/sessions/{id}/tree/diff`` (phase 5)."""

    a_id: str
    b_id: str


class JointLogprobsRequest(BaseModel):
    """Body for ``POST /saklas/v1/sessions/{id}/tree/joint_logprobs``
    (logit-pass Phase 5 of ``docs/plans/logit-pass.md``).

    Lazy / on-demand cross-evaluation between two sibling assistant
    nodes — fired only when ``NodeCompareDrawer`` asks for it.  Results
    cache on the session for the session lifetime, keyed by sorted
    ``(a_id, b_id)`` so ``(A, B)`` and ``(B, A)`` requests share an
    entry; the response is re-oriented to match the request's
    a/b ordering before serialization.
    """

    a_id: str
    b_id: str


class BakeVectorRequest(BaseModel):
    """Body for ``POST /saklas/v1/sessions/{id}/vectors/bake``.

    ``expression`` is a merge expression in the shared steering grammar
    (``"0.3 default/honest + 0.4 default/warm"``); ``name`` becomes the
    new merged manifold's local name.  Reuses
    :func:`saklas.io.merge.merge_into_manifold`.
    """
    name: str
    expression: str


class ExperimentFanRequest(BaseModel):
    """Body for ``POST /saklas/v1/sessions/{id}/experiments/fan``.

    ``grid`` maps concept name to alpha values.  The Cartesian product
    becomes sibling assistant nodes under one shared user turn.
    ``base_steering`` (optional) is a steering expression string
    composed underneath each grid term so callers can hold a
    fixed-alpha context while sweeping another concept.
    """
    prompt: Any
    grid: dict[str, list[float]]
    base_steering: str | None = None
    sampling: WSSamplingParams | None = None
    thinking: bool | None = None
    raw: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_session_id(session: SaklasSession, session_id: str) -> None:
    """Raise 404 if ``session_id`` doesn't map to the single session."""
    if session_id == _SINGLE_SESSION_ID:
        return
    if session_id == session.model_id:
        return
    raise HTTPException(
        status_code=404,
        detail=f"session '{session_id}' not found",
    )


def _session_config_dict(session: SaklasSession) -> dict[str, Any]:
    cfg = session.config
    return {
        "temperature": getattr(cfg, "temperature", None),
        "top_p": getattr(cfg, "top_p", None),
        "top_k": getattr(cfg, "top_k", None),
        "max_tokens": getattr(cfg, "max_new_tokens", None),
        "system_prompt": getattr(cfg, "system_prompt", None),
    }


def _session_model_type(session: SaklasSession) -> str | None:
    """Resolve the loaded model's ``model_type`` (unwrapping multimodal
    ``text_config``) — the key both role-header registries are indexed by."""
    model_cfg = getattr(getattr(session, "model", None), "config", None)
    if model_cfg is None:
        return None
    text_cfg = getattr(model_cfg, "text_config", None)
    mt = getattr(text_cfg, "model_type", None) if text_cfg is not None else None
    return mt or getattr(model_cfg, "model_type", None)


def _role_support(session: SaklasSession) -> tuple[bool, bool]:
    """``(assistant_supported, user_supported)`` for the loaded family —
    a non-``None`` entry in the respective role-header registry."""
    from saklas.core.role_templates import ROLE_HEADERS, USER_ROLE_HEADERS

    mt = _session_model_type(session)
    if mt is None:
        return (False, False)
    return (
        ROLE_HEADERS.get(mt) is not None,
        USER_ROLE_HEADERS.get(mt) is not None,
    )


def _default_role_labels(session: SaklasSession) -> tuple[str | None, str | None]:
    """``(assistant_label, user_label)`` — the family's *standard* role
    strings (e.g. Gemma ``model`` / ``user``, ChatML ``assistant`` / ``user``),
    or ``None`` when the family can't substitute that side.  The webui seeds
    the role boxes with these so they show the live defaults rather than an
    empty placeholder."""
    from saklas.core.role_templates import ROLE_HEADERS, USER_ROLE_HEADERS

    mt = _session_model_type(session)
    if mt is None:
        return (None, None)
    asst = ROLE_HEADERS.get(mt)
    usr = USER_ROLE_HEADERS.get(mt)
    return (
        asst.label if asst is not None else None,
        usr.label if usr is not None else None,
    )


def _device_dtype(session: SaklasSession) -> tuple[str, str]:
    info = session.model_info or {}
    device = str(info.get("device", getattr(session, "device", "")))
    dtype = str(info.get("dtype", getattr(session, "_dtype", "")))
    return device, dtype


def _session_info(
    session: SaklasSession, default_steering: "Steering | None",
) -> dict[str, Any]:
    device, dtype = _device_dtype(session)
    try:
        thinks = bool(supports_thinking(session.tokenizer))
        thinks_optional = bool(thinking_is_optional(session.tokenizer))
    except Exception:
        thinks = False
        thinks_optional = False
    try:
        is_base = bool(session.is_base_model)
    except Exception:
        is_base = False
    created = getattr(session, "_created_ts", None) or int(time.time())
    default_expr = str(default_steering) if default_steering is not None else None
    try:
        assistant_role_ok, user_role_ok = _role_support(session)
        default_assistant_role, default_user_role = _default_role_labels(session)
    except Exception:
        assistant_role_ok = user_role_ok = False
        default_assistant_role = default_user_role = None
    # Path-existence check only — the lens artifact is fitted fp16 matrices
    # (hundreds of MB); the lazy ``session.jlens`` load must never ride a
    # session-info poll.
    try:
        from saklas.io.lens import lens_paths

        jlens_fitted = lens_paths(session.model_id)[0].exists()
    except Exception:
        jlens_fitted = False
    return {
        "id": _SINGLE_SESSION_ID,
        "model_id": session.model_id,
        "device": device,
        "dtype": dtype,
        "created": created,
        "config": _session_config_dict(session),
        "vectors": sorted(session.vectors.keys()),
        "probes": sorted(session.probes.keys()),
        "history_length": len(session.history) if hasattr(session, "history") else 0,
        "supports_thinking": thinks,
        "thinking_is_optional": thinks_optional,
        "is_base_model": is_base,
        "jlens_fitted": jlens_fitted,
        "default_steering": default_expr,
        "role_substitution_supported": assistant_role_ok,
        "user_role_supported": user_role_ok,
        "default_assistant_role": default_assistant_role,
        "default_user_role": default_user_role,
    }


def _profile_to_json(name: str, profile: Profile) -> dict[str, Any]:
    layer_norms = [(idx, float(vec.norm().item())) for idx, vec in profile.items()]
    top = sorted(layer_norms, key=itemgetter(1), reverse=True)[:5]
    # Full per-layer ||baked|| keyed by layer index — stringified for
    # JSON-key compatibility, mirroring how diagnostics_by_layer round-trips.
    # The web UI's LayerNorms panel consumes this directly.
    per_layer_norms = {str(idx): round(mag, 6) for idx, mag in sorted(layer_norms)}
    return {
        "name": name,
        "layers": profile.layers,
        "top_layers": [{"layer": idx, "magnitude": round(m, 4)} for idx, m in top],
        "per_layer_norms": per_layer_norms,
        "metadata": profile.metadata,
    }


def _extract_registry_name(canonical: str, namespace: str | None) -> str:
    """Return the steerable key for a freshly extracted vector.

    ``session.extract`` returns the concept-local canonical name with any
    tensor variant suffix, while a non-default destination namespace lives
    outside that name.  Reattach the namespace at the registry/API boundary so
    immediate steering uses the same key that disk lookup will resolve later.
    """
    if namespace is None:
        return canonical
    if ":" in canonical:
        bare, suffix = canonical.rsplit(":", 1)
        return f"{namespace}/{bare}:{suffix}"
    return f"{namespace}/{canonical}"


def _probe_profile_tensors(
    session: SaklasSession, name: str,
) -> dict[int, Any] | None:
    """Folded per-layer direction view of a vector probe, or ``None``.

    The monitor now holds each vector probe as a flat 2-node
    :class:`~saklas.core.manifold.Manifold` (``session.monitor.manifolds``);
    the legacy ``monitor.profiles`` baked-``dict[int, Tensor]`` accessor is
    gone.  This folds the named probe's manifold back to the same
    ``{L: δ̂_L · share_L}`` baked-direction view callers used to read off
    ``profiles`` (whitened-cosine matrices, the diagnostics histogram), so
    the wire shape those routes emit is unchanged.
    """
    manifold = session.monitor.manifolds.get(name)
    if manifold is None:
        return None
    from saklas.core.vectors import (
        folded_vector_directions,
        is_foldable_vector_manifold,
    )

    # A multi-node / curved probe has no single baked direction to fold —
    # the diagnostics histogram only means anything for an R=1 vector.
    if not is_foldable_vector_manifold(manifold):
        return None
    return folded_vector_directions(manifold)


def _build_sampling(body: WSSamplingParams | None) -> SamplingConfig | None:
    if body is None:
        return None
    stop = tuple(body.stop) if body.stop else None
    return SamplingConfig(
        temperature=body.temperature,
        top_p=body.top_p,
        top_k=body.top_k,
        max_tokens=body.max_tokens,
        seed=body.seed,
        stop=stop,
        logit_bias=body.logit_bias,
        presence_penalty=body.presence_penalty or 0.0,
        frequency_penalty=body.frequency_penalty or 0.0,
        return_top_k=body.return_top_k,
        user_role=(body.user_role or None),
        assistant_role=(body.assistant_role or None),
        persist_per_layer_scores=bool(body.persist_per_layer_scores),
        persist_subspace_coords=bool(body.persist_subspace_coords),
    )


def _coerce_corpora(source: Any) -> Any:
    """Normalize a JSON extract source into a concept name or two pole corpora.

    A concept-name string passes through unchanged.  A
    ``{positive: [...], negative: [...]}`` object (two pole corpora), a
    ``{pairs: [{positive, negative}, ...]}`` object, or a bare single
    ``{positive, negative}`` object are all turned into a
    ``(positive_corpus, negative_corpus)`` tuple — the two node corpora of the
    2-node ``pca`` manifold the steering vector is fit as.  No
    ``{positive, negative}`` pairs are retained: hand-authored contrastive
    examples become two independent corpora at the boundary.
    """
    if not isinstance(source, dict):
        return source

    # Two-corpora form: ``{positive: [...], negative: [...]}``.
    if (
        isinstance(source.get("positive"), list)
        and isinstance(source.get("negative"), list)
    ):
        return (
            [str(s) for s in source["positive"]],
            [str(s) for s in source["negative"]],
        )

    # Pairs / single-pair forms — unzip into the two corpora.
    raw_pairs: list[Any]
    if "pairs" in source:
        raw_pairs = list(source["pairs"])
    elif "positive" in source and "negative" in source:
        raw_pairs = [source]
    else:
        return source

    positive: list[str] = []
    negative: list[str] = []
    for idx, pair in enumerate(raw_pairs):
        if isinstance(pair, dict):
            if "positive" not in pair or "negative" not in pair:
                raise HTTPException(
                    400,
                    f"pairs[{idx}] must contain 'positive' and 'negative'",
                )
            positive.append(str(pair["positive"]))
            negative.append(str(pair["negative"]))
        elif isinstance(pair, (list, tuple)) and len(pair) == 2:
            positive.append(str(pair[0]))
            negative.append(str(pair[1]))
        else:
            raise HTTPException(
                400,
                f"pairs[{idx}] must be a [positive, negative] pair",
            )
    return positive, negative


def _result_to_json(result: GenerationResult | RunSet | None) -> dict[str, Any]:
    if result is None:
        return {}
    prompt_tokens = getattr(result, "prompt_tokens", 0) or 0
    completion = getattr(result, "token_count", 0) or 0
    return {
        "text": getattr(result, "text", ""),
        "tokens": completion,
        "finish_reason": getattr(result, "finish_reason", "stop"),
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion,
            "total_tokens": prompt_tokens + completion,
        },
    }


def _tree_to_json(session: SaklasSession) -> dict[str, Any]:
    """Serialize the session's loom tree to JSON.

    Thin wrapper over :meth:`LoomTree.to_dict` with ``include_tokens=True``
    so a webui force-refresh can rehydrate per-token spans (highlight
    tints, click-to-drilldown, surprise mode, fork affordance) from the
    server tree.  Per-token blobs ride the per-node ``tokens`` /
    ``thinking_tokens`` arrays — bulky on long conversations but the
    only way the inline highlight survives a tab reload (the client's
    in-memory ``tokenScoreCache`` is wiped, and the on-disk save sidecar
    is not in the wire path).
    """
    return session.tree.to_dict(include_tokens=True)


def _active_path_json(session: SaklasSession) -> dict[str, Any]:
    """Active path as a chat-message list paired with node ids.

    Returns ``{"active_node_id", "rev", "messages": [...], "node_ids": [...]}``
    where ``messages`` is the v2 chat-message shape (skipping the synthetic
    root) and ``node_ids`` is the parallel list of loom-tree node ids in
    the same order. Surfaces that need both the chat-render and the
    tree-navigation can read them off one fetch.
    """
    tree = session.tree
    path = tree.active_path()
    messages: list[dict[str, str]] = []
    node_ids: list[str] = []
    for node in path:
        if node.id == tree.root_id:
            continue
        messages.append({"role": node.role, "content": node.text})
        node_ids.append(node.id)
    return {
        "active_node_id": tree.active_node_id,
        "rev": tree.rev,
        "messages": messages,
        "node_ids": node_ids,
    }


def _node_json(session: SaklasSession, node_id: str) -> dict[str, Any]:
    """Serialize a single node to JSON, including its child-id list.

    The child-id list isn't on ``LoomNode`` itself (the tree owns the
    structure map), but surfaces routinely want it alongside the node
    payload — so we attach it here.  ``include_tokens=True`` mirrors
    :func:`_tree_to_json` so ``tree_mutated`` deltas carry the same
    per-token shape the initial tree GET ships.
    """
    node = session.tree.get(node_id)
    out = node.to_dict(include_tokens=True)
    out["children"] = list(session.tree.children_of.get(node_id, []))
    return out


def _per_token_probes(session: SaklasSession, n_tokens: int) -> list[dict[str, Any]]:
    scores = session.last_per_token_scores
    if not scores:
        return []
    return [
        {
            "token_idx": i,
            "probes": {name: float(vals[i]) for name, vals in scores.items() if i < len(vals)},
        }
        for i in range(n_tokens)
    ]


# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------


def register_saklas_routes(app: FastAPI) -> None:
    """Mount the native ``/saklas/v1/*`` tree onto ``app``.

    ``session`` and ``default_steering`` are pulled off ``app.state`` so the
    signature matches ``register_ollama_routes`` and ``create_app`` doesn't
    need to thread them.
    """

    # ----- manifolds (top-level, own resource tree) ----------------------

    from saklas.server.manifold_routes import register_manifold_routes
    register_manifold_routes(app)

    # ----- templates (templated-completion artifact + scorer) ------------

    from saklas.server.template_routes import register_template_routes
    register_template_routes(app)

    # ----- sessions collection -------------------------------------------

    from saklas.server.session_routes import register_session_routes
    register_session_routes(app)

    # ----- loom tree (v2.3 phase 2) --------------------------------------

    from saklas.server.tree_routes import register_tree_routes
    register_tree_routes(app)

    # ----- vectors -------------------------------------------------------

    from saklas.server.vector_routes import register_vector_routes
    register_vector_routes(app)

    # ----- probes / experiments / live traits ----------------------------

    from saklas.server.probe_routes import register_probe_routes
    register_probe_routes(app)

    from saklas.server.experiment_routes import register_experiment_routes
    register_experiment_routes(app)

    from saklas.server.traits_routes import register_traits_routes
    register_traits_routes(app)

    from saklas.server.lens_routes import register_lens_routes
    register_lens_routes(app)

    # ----- WebSocket token+probe co-stream -------------------------------

    from saklas.server.ws_stream import register_ws_stream
    register_ws_stream(app)
