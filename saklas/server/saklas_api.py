"""Native saklas HTTP namespace (``/saklas/v1/*``).

This is the saklas-native resource-tree API, distinct from the OpenAI
(``/v1/*``) and Ollama (``/api/*``) compat shims.  Shape is designed
multi-session — URL-paths carry ``{session_id}`` — but the current impl
is single-session.  The one session has id ``"default"``; both that
literal and the loaded model id resolve to it, everything else 404s.

Killer feature: ``WS /saklas/v1/sessions/{id}/stream`` bidirectional
token + probe co-stream.  Per-token probe readings can't currently be
pushed inline from the session hot path (they're computed once the run
finalizes, via ``score_captured``).  So the WS protocol ships plain
token events during the run and a single ``per_token_probes`` array in
the ``done`` event, assembled from ``session._last_per_token_scores``.
Future clusters can upgrade to inline streaming without changing the
wire format meaningfully.

Old ``/v1/saklas/*`` routes were removed in the same commit that
introduced this file — no aliases.
"""

# pyright: reportUnusedFunction=false

from __future__ import annotations

from saklas.server.app import acquire_session_lock, ws_auth_ok

import asyncio
import time
import uuid
from collections import deque
from contextlib import suppress
from operator import itemgetter
from typing import Any, Awaitable, Callable, Literal

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from pydantic import BaseModel, Field

from saklas.core.errors import SaklasError
from saklas.core.generation import supports_thinking, thinking_is_optional
from saklas.io.probes_bootstrap import load_default_manifolds as load_defaults  # noqa: F401
from saklas.core.loom import LoomMutated
from saklas.core.profile import Profile
from saklas.core.results import GenerationResult, RunSet
from saklas.core.sampling import SamplingConfig
from saklas.core.session import SaklasSession
from saklas.core.steering import Steering
from saklas.server.sse import ProgressCallback, progress_sse_response
from saklas.server.ws_events import build_token_event


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
    model_cfg = getattr(getattr(session, "_model", None), "config", None)
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
    device = str(info.get("device", getattr(session, "_device", "")))
    dtype = str(info.get("dtype", getattr(session, "_dtype", "")))
    return device, dtype


def _session_info(
    session: SaklasSession, default_steering: "Steering | None",
) -> dict[str, Any]:
    device, dtype = _device_dtype(session)
    try:
        thinks = bool(supports_thinking(session._tokenizer))
        thinks_optional = bool(thinking_is_optional(session._tokenizer))
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
    :class:`~saklas.core.manifold.Manifold` (``session._monitor.manifolds``);
    the legacy ``monitor.profiles`` baked-``dict[int, Tensor]`` accessor is
    gone.  This folds the named probe's manifold back to the same
    ``{L: δ̂_L · share_L}`` baked-direction view callers used to read off
    ``profiles`` (whitened-cosine matrices, the diagnostics histogram), so
    the wire shape those routes emit is unchanged.
    """
    manifold = session._monitor.manifolds.get(name)
    if manifold is None:
        return None
    from saklas.core.vectors import folded_vector_directions

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
    )


def _build_steering(
    raw: str | None, default_steering: "Steering | None",
) -> "Steering | None":
    """Compose a request expression string over the server default Steering.

    ``None`` inherits the server default.  An explicit empty string is a
    request for no steering, used by the web UI's unsteered shadow runs.
    Non-empty request keys override the default at the key level.
    """
    from saklas.core.steering_expr import parse_expr

    req: "Steering | None" = None
    explicit_clear = raw is not None and not raw.strip()
    if raw is not None and raw.strip():
        req = parse_expr(raw)

    thinking: bool | None = None
    if req is not None and req.thinking is not None:
        thinking = req.thinking

    merged: dict[str, Any] = {}
    if default_steering is not None and not explicit_clear:
        merged.update(default_steering.alphas)
    if req is not None:
        merged.update(req.alphas)

    if not merged and thinking is None:
        return None
    return Steering(alphas=merged, thinking=thinking)


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
    n = min(n_tokens, *(len(v) for v in scores.values())) if scores else 0
    return [
        {
            "token_idx": i,
            "probes": {name: float(vals[i]) for name, vals in scores.items() if i < len(vals)},
        }
        for i in range(n)
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

    session: SaklasSession = app.state.session

    # ----- manifolds (top-level, own resource tree) ----------------------

    from saklas.server.manifold_routes import register_manifold_routes
    register_manifold_routes(app)

    # ----- sessions collection -------------------------------------------

    from saklas.server.session_routes import register_session_routes
    register_session_routes(app)

    # ----- loom tree (v2.3 phase 2) --------------------------------------

    @app.get("/saklas/v1/sessions/{session_id}/tree")
    def get_tree(session_id: str):
        """Full tree as JSON.

        Same shape :meth:`LoomTree.to_dict` produces. Surfaces hydrate
        their state from this on bootstrap and reconcile via the WS
        ``tree_mutated`` delta stream after.
        """
        _resolve_session_id(session, session_id)
        return _tree_to_json(session)

    @app.get("/saklas/v1/sessions/{session_id}/tree/active")
    def get_tree_active(session_id: str):
        """Active path: chat messages + parallel node-id list.

        Cheaper than the full tree for surfaces that only need the
        currently-rendered conversation. The node-id list is parallel to
        ``messages`` so a click on message ``i`` maps to ``node_ids[i]``.
        """
        _resolve_session_id(session, session_id)
        return _active_path_json(session)

    @app.post("/saklas/v1/sessions/{session_id}/tree/navigate")
    async def tree_navigate(session_id: str, req: TreeNavigateRequest):
        """Re-point the active node.

        Free relative to in-flight generation (per the concurrency
        invariant in the plan): the gen continues attached to its
        original target, the user simply sees a different active path.
        """
        _resolve_session_id(session, session_id)
        session.tree.navigate(req.node_id)
        return _active_path_json(session)

    @app.post("/saklas/v1/sessions/{session_id}/tree/edit")
    async def tree_edit(session_id: str, req: TreeEditRequest):
        """In-place text replacement.

        409 when the node is in the reservation of an in-flight
        generation (mapped via ``SaklasError.user_message``); 404 on
        unknown id; 400 on root-edit or other invalid ops.
        """
        _resolve_session_id(session, session_id)
        session.tree.edit(req.node_id, req.text)
        return _node_json(session, req.node_id)

    @app.post("/saklas/v1/sessions/{session_id}/tree/branch")
    async def tree_branch(session_id: str, req: TreeBranchRequest):
        """Always-sibling — create a new node next to ``node_id``.

        Allowed during in-flight generation; the new sibling sits on the
        same user-parent as the gen target without disturbing it.
        Returns ``{node_id, node, active_path}`` so the caller can place
        the new node and (if it became active) re-render the chat
        without a follow-up fetch.
        """
        _resolve_session_id(session, session_id)
        # Cast role through the Literal-narrowing layer the tree owns.
        role_arg = req.role
        new_id = session.tree.branch(
            req.node_id, req.text, role=role_arg,  # pyright: ignore[reportArgumentType]  # req.role is str|None; branch() expects Role|None (Literal narrowing)
        )
        return {
            "node_id": new_id,
            "node": _node_json(session, new_id),
            "active_path": _active_path_json(session),
        }

    @app.delete("/saklas/v1/sessions/{session_id}/tree/{node_id}")
    async def tree_delete(session_id: str, node_id: str):
        """Subtree delete.

        400 for the root delete; 409 when the subtree intersects an
        in-flight generation's reservation; 404 on unknown id.  When
        the active node sits inside the deleted subtree, the engine
        re-seats the active pointer on the surviving parent and emits
        the new ``active_node_id`` on the mutation event.  Returns
        ``{removed: <count>}``.
        """
        _resolve_session_id(session, session_id)
        removed = session.tree.delete_subtree(node_id)
        return {"removed": removed}

    @app.post("/saklas/v1/sessions/{session_id}/tree/star")
    async def tree_star(session_id: str, req: TreeStarRequest):
        """Toggle a node's ``starred`` flag.

        Decoration-only; never raises a concurrency conflict.
        """
        _resolve_session_id(session, session_id)
        session.tree.star(req.node_id, req.on)
        return _node_json(session, req.node_id)

    @app.post("/saklas/v1/sessions/{session_id}/tree/note")
    async def tree_note(session_id: str, req: TreeNoteRequest):
        """Set a node's free-text ``notes`` annotation.

        Decoration-only; never raises a concurrency conflict.
        """
        _resolve_session_id(session, session_id)
        session.tree.annotate(req.node_id, req.text)
        return _node_json(session, req.node_id)

    @app.post("/saklas/v1/sessions/{session_id}/tree/reset", status_code=204)
    async def tree_reset(session_id: str):
        """Drop the entire tree and rebuild a fresh root.

        Equivalent to ``session.clear_history()``; 409 when a generation
        is in flight (per the concurrency invariant — ``reset`` cannot
        race the gen path because the gen path owns the streaming target
        in the tree itself).
        """
        _resolve_session_id(session, session_id)
        async with session.lock:
            session.clear_history()
        return Response(status_code=204)

    @app.post("/saklas/v1/sessions/{session_id}/tree/transcript")
    def tree_transcript(session_id: str, req: TreeTranscriptRequest):
        """Render the path ending at ``node_id`` (or active) as transcript YAML.

        Phase 5 producer: uses :meth:`Transcript.from_path` so probe
        sha256 hashes are real and the YAML round-trips through
        :meth:`Transcript.from_yaml` cleanly.  Returns
        ``{"yaml": "<text>", "node_id": "<leaf-of-rendered-path>"}``.
        """
        from saklas.core.transcript import Transcript

        _resolve_session_id(session, session_id)
        leaf = req.node_id if req.node_id is not None else session.tree.active_node_id
        # Validate the id before touching the renderer so the 404 lands
        # cleanly through the existing ``SaklasError`` handler.
        session.tree.get(leaf)
        transcript = Transcript.from_path(leaf, session)
        return {"yaml": transcript.to_yaml(), "node_id": leaf}

    @app.post("/saklas/v1/sessions/{session_id}/tree/transcript/load")
    async def tree_transcript_load(
        session_id: str, req: TreeTranscriptLoadRequest,
    ):
        """Import a transcript YAML into the live session tree (phase 5).

        Wraps :meth:`Transcript.from_yaml` + :meth:`Transcript.import_into`.
        Modes are ``"default"`` / ``"here"`` / ``"merge"``; ``strict``
        refuses on probe-hash drift.  Returns
        ``{"leaf_id": "<id>", "rev": <int>, "guards": [...]}``.

        Guards (model mismatch, system-prompt mismatch, probe drift) are
        also stamped on the imported branch's root node as ``notes`` so
        the surfaces can show a banner there.  Returning them in the body
        too saves the client one fetch.
        """
        from saklas.core.transcript import (
            Transcript,
            TranscriptError,
            TranscriptFormatError,
        )

        _resolve_session_id(session, session_id)
        mode = req.mode or "default"
        if mode not in ("default", "here", "merge"):
            raise HTTPException(
                400, f"unknown import mode {mode!r}; valid: default, here, merge",
            )
        try:
            transcript = Transcript.from_yaml(req.yaml)
        except TranscriptFormatError as e:
            raise HTTPException(400, f"invalid transcript: {e}") from e
        import warnings

        captured: list[str] = []

        def _on_warning(
            message: Warning | str,
            category: type[Warning],
            filename: str,
            lineno: int,
            file: Any = None,
            line: str | None = None,
        ) -> None:
            captured.append(str(message))

        async with session.lock:
            with warnings.catch_warnings():
                warnings.showwarning = _on_warning
                try:
                    leaf_id = await asyncio.to_thread(
                        transcript.import_into,
                        session,
                        mode=mode,
                        strict=req.strict,
                    )
                except TranscriptError as e:
                    raise HTTPException(400, str(e)) from e
        return {
            "leaf_id": leaf_id,
            "rev": session.tree.rev,
            "guards": captured,
        }

    @app.get("/saklas/v1/sessions/{session_id}/tree/edge_label")
    def tree_edge_label(session_id: str, parent_id: str, child_id: str):
        """Steering-delta label for the parent → child edge (phase 5).

        Returns ``{"label": "<text>"}`` — empty string when the two
        recipes are identical.  Both nodes must exist; the label is
        computed from the canonical ``applied_steering`` strings on the
        parent's and child's recipes (parent's may be ``None`` when it's
        a user turn, in which case the delta is "from-nothing").
        """
        from saklas.core.loom_diff import steering_delta

        _resolve_session_id(session, session_id)
        parent = session.tree.get(parent_id)
        child = session.tree.get(child_id)
        parent_expr = parent.applied_steering
        if parent_expr is None and parent.recipe is not None:
            parent_expr = parent.recipe.steering
        child_expr = child.applied_steering
        if child_expr is None and child.recipe is not None:
            child_expr = child.recipe.steering
        return {"label": steering_delta(parent_expr, child_expr)}

    @app.get("/saklas/v1/sessions/{session_id}/tree/filter")
    def tree_filter(session_id: str, expr: str = ""):
        """Apply a filter-grammar expression and return matching node ids.

        Grammar in :mod:`saklas.core.tree_filter` — comma-AND'd
        ``agg:|any:|last:<probe> <op> <threshold>`` clauses.  Empty
        ``expr`` returns every node id (clears the filter).  Bad
        expressions land as 400 via :class:`FilterParseError`.
        """
        from saklas.core.tree_filter import FilterParseError

        _resolve_session_id(session, session_id)
        text = (expr or "").strip()
        if not text:
            return {"expr": "", "matching_node_ids": []}
        try:
            matches = session.tree.filter_by_expr(text)
        except FilterParseError as e:
            raise HTTPException(400, str(e)) from e
        return {"expr": text, "matching_node_ids": sorted(matches)}

    @app.post("/saklas/v1/sessions/{session_id}/tree/diff")
    def tree_diff(session_id: str, req: TreeDiffRequest):
        """Cross-branch diff between two assistant nodes (phase 5).

        Returns a JSON view of :class:`NodeDiff` (text spans + readings
        deltas) augmented with the parent-recipe steering delta and any
        per-token deltas available from the session's
        ``last_per_token_scores`` — the per-token table is only present
        for the most-recently-generated assistant so callers shouldn't
        rely on it.
        """
        from saklas.core.loom_diff import per_token_diff, steering_delta

        _resolve_session_id(session, session_id)
        diff = session.diff_nodes(req.a_id, req.b_id)
        a_node = session.tree.get(req.a_id)
        b_node = session.tree.get(req.b_id)

        # Steering-delta against the shared parent's expression — only
        # meaningful for sibling diffs (parent_id present).
        parent_expr: str | None = None
        if diff.parent_id is not None:
            parent = session.tree.nodes.get(diff.parent_id)
            if parent is not None:
                parent_expr = parent.applied_steering
                if parent_expr is None and parent.recipe is not None:
                    parent_expr = parent.recipe.steering

        a_expr = a_node.applied_steering or (
            a_node.recipe.steering if a_node.recipe else None
        )
        b_expr = b_node.applied_steering or (
            b_node.recipe.steering if b_node.recipe else None
        )

        # Per-token diff: only when both nodes carry token sequences.
        # Tokens may be absent on serialized-only nodes (loaded transcripts).
        a_tok_strs: list[str] = []
        if a_node.tokens:
            a_tok_strs = [t.get("text", "") for t in a_node.tokens]
        b_tok_strs: list[str] = []
        if b_node.tokens:
            b_tok_strs = [t.get("text", "") for t in b_node.tokens]
        per_token_spans: list[dict[str, Any]] = []
        if a_tok_strs and b_tok_strs:
            spans = per_token_diff(a_tok_strs, b_tok_strs)
            per_token_spans.extend(
                {
                    "a_index": sp.a_index,
                    "b_index": sp.b_index,
                    "a_text": sp.a_text,
                    "b_text": sp.b_text,
                    "aligned": sp.aligned,
                    "reading_deltas": [
                        {
                            "name": rd.name,
                            "delta": round(float(rd.delta), 6),
                            "a_value": round(float(rd.a_value), 6),
                            "b_value": round(float(rd.b_value), 6),
                        }
                        for rd in sp.reading_deltas
                    ],
                }
                for sp in spans
            )

        return {
            "a_id": diff.a_id,
            "b_id": diff.b_id,
            "parent_id": diff.parent_id,
            "a_text": a_node.text,
            "b_text": b_node.text,
            "a_applied_steering": a_expr,
            "b_applied_steering": b_expr,
            "parent_applied_steering": parent_expr,
            "steering_delta": steering_delta(a_expr, b_expr),
            "parent_to_a_delta": (
                steering_delta(parent_expr, a_expr)
                if parent_expr is not None or a_expr is not None
                else ""
            ),
            "parent_to_b_delta": (
                steering_delta(parent_expr, b_expr)
                if parent_expr is not None or b_expr is not None
                else ""
            ),
            "text": [
                {"state": sp.state, "text": sp.text}
                for sp in diff.text
            ],
            "readings": [
                {
                    "name": rd.name,
                    "delta": round(float(rd.delta), 6),
                    "a_value": round(float(rd.a_value), 6),
                    "b_value": round(float(rd.b_value), 6),
                }
                for rd in diff.readings
            ],
            "per_token": per_token_spans,
        }

    @app.post("/saklas/v1/sessions/{session_id}/tree/joint_logprobs")
    async def tree_joint_logprobs(session_id: str, req: JointLogprobsRequest):
        """Cross-evaluation between two sibling assistant nodes.

        Logit-pass Phase 5 of ``docs/plans/logit-pass.md``.  Force-replays
        each branch under the node's stamped recipe, steering hooks, probe
        gates, penalties, logit bias, and sampler transform, then returns
        per-aligned-position records carrying both branches' chosen-token
        logprobs *and* the cross-branch evaluation (what each side would
        have given the other's chosen token at the same byte-aligned
        position).

        Cache shape:
        * Stored on ``session._joint_logprob_cache: dict[tuple[str,
          str], JointLogprobs]`` keyed by sorted ``(a_id, b_id)`` so
          the symmetric pair shares an entry.
        * Invalidated by tree edits/deletes/finalize events in
          ``SaklasSession``; navigate/star/note leave it intact.

        Held under ``acquire_session_lock`` because the forward passes
        compete for the same model with any concurrent generation;
        request queues FIFO at the lock rather than 409ing.
        """
        from saklas.core.joint_logprobs import (
            compute_joint_logprobs,
            _cache_key,
            reorient_for_request,
        )

        _resolve_session_id(session, session_id)
        if req.a_id == req.b_id:
            raise HTTPException(400, "a_id and b_id must differ")
        if req.a_id not in session.tree.nodes:
            raise HTTPException(404, f"unknown node id: {req.a_id}")
        if req.b_id not in session.tree.nodes:
            raise HTTPException(404, f"unknown node id: {req.b_id}")

        # New sessions create this cache in SaklasSession; keep the lazy
        # fallback for older test doubles and external session shims.
        cache_obj: Any = getattr(session, "_joint_logprob_cache", None)
        if cache_obj is None:
            cache_obj = {}
            session._joint_logprob_cache = cache_obj
        cache: dict[tuple[str, str], Any] = cache_obj

        key = _cache_key(req.a_id, req.b_id)
        hit = cache.get(key)
        if hit is None:
            async with acquire_session_lock(session):
                # Double-check under lock — another request may have
                # populated the cache while we waited.
                hit = cache.get(key)
                if hit is None:
                    hit = await asyncio.to_thread(
                        compute_joint_logprobs, session, req.a_id, req.b_id,
                    )
                    cache[key] = hit
        return reorient_for_request(hit, req.a_id, req.b_id).to_dict()

    # ----- vectors -------------------------------------------------------

    @app.get("/saklas/v1/sessions/{session_id}/vectors")
    def list_vectors(session_id: str):
        _resolve_session_id(session, session_id)
        return {
            "vectors": [
                _profile_to_json(name, profile)
                for name, profile in sorted(session.vectors.items())
            ],
        }

    @app.get("/saklas/v1/sessions/{session_id}/vectors/pairwise")
    def pairwise_compare(session_id: str, a: str, b: str):
        """Cross-layer whitened cosine matrix between two named vectors / probes.

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

        Pool unions ``session.vectors`` and ``monitor.probe_names`` (same
        as :func:`correlation_matrix`) so probes that were never
        registered as steering vectors still resolve.  Near-zero layer
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

        Registered *before* ``GET /vectors/{name}`` so the literal path
        wins the routing match — Starlette matches in registration order
        and ``pairwise`` would otherwise be swallowed by ``{name}``.
        """
        from saklas import Profile

        _resolve_session_id(session, session_id)

        pool: dict[str, "Profile"] = dict(session.vectors)
        try:
            for probe_name in session._monitor.probe_names:
                if probe_name in pool:
                    continue
                tensors = _probe_profile_tensors(session, probe_name)
                if tensors is None:
                    continue
                pool[probe_name] = Profile(tensors)
        except Exception:
            pass

        missing = [n for n in (a, b) if n not in pool]
        if missing:
            raise HTTPException(404, f"names not loaded: {missing}")

        prof_a, prof_b = pool[a], pool[b]
        layers_a = sorted(prof_a.keys())
        layers_b = sorted(prof_b.keys())

        # Precompute fp32 vectors + norms so the inner loop is a single
        # dot per cell.  ``None`` for near-zero norms — propagates to the
        # cell so the client can render an empty / dimmed square instead
        # of NaN or a meaningless cosine.  Both sides are forced to CPU
        # so a cross-device pair (e.g. an actively-steered vector hooked
        # on MPS vs. a disk-loaded peer on CPU) computes cleanly rather
        # than raising on the dot — hidden_dim × layer-count is small
        # enough that the device round-trip is free relative to the
        # request budget.
        import torch as _torch
        vecs_a: list[tuple["_torch.Tensor", float]] = []
        for L in layers_a:
            v = prof_a[L].float().cpu()
            n = float(v.norm().item())
            vecs_a.append((v, n))
        vecs_b: list[tuple["_torch.Tensor", float]] = []
        for L in layers_b:
            v = prof_b[L].float().cpu()
            n = float(v.norm().item())
            vecs_b.append((v, n))

        # Resolve the whitener (Mahalanobis-only).  ``session.whitener`` is
        # a lazy property (builds from the neutral-activation cache on first
        # access).  It must cover every row-layer of ``a`` (each row is
        # framed in its row-layer's covariance) — there is no Euclidean
        # fallback, so a missing / non-covering whitener is a 409.
        whitener = getattr(session, "whitener", None)
        if whitener is None or not whitener.covers_all(layers_a):
            raise HTTPException(
                409,
                "pairwise compare requires a Mahalanobis whitener covering "
                f"every row-layer {layers_a} of '{a}'; regenerate the neutral "
                "activation cache for this model (the Euclidean path is gone)",
            )

        matrix: list[list[float | None]] = []
        for la, (va, na) in zip(layers_a, vecs_a, strict=True):
            row: list[float | None] = []
            for vb, nb in vecs_b:
                if na < 1e-12 or nb < 1e-12:
                    row.append(None)
                    continue
                # Different hidden dims would be a model-mismatch bug —
                # surface as None rather than raising so a misconfigured
                # pool still renders the cells that *do* line up.
                if va.shape != vb.shape:
                    row.append(None)
                    continue
                cos = whitener.mahalanobis_cosine(la, va, vb)
                row.append(round(cos, 6))
            matrix.append(row)

        return {
            "a": a,
            "b": b,
            "metric": "mahalanobis",
            "layers_a": layers_a,
            "layers_b": layers_b,
            "matrix": matrix,
            "model": getattr(session, "model_id", None),
        }

    @app.get("/saklas/v1/sessions/{session_id}/vectors/{name}")
    def get_vector(session_id: str, name: str):
        _resolve_session_id(session, session_id)
        vectors = session.vectors
        if name not in vectors:
            raise HTTPException(404, f"vector '{name}' not found")
        return _profile_to_json(name, vectors[name])

    @app.post("/saklas/v1/sessions/{session_id}/vectors")
    def load_vector(session_id: str, req: LoadVectorRequest):
        _resolve_session_id(session, session_id)
        try:
            profile = session.load_profile(req.source_path)
        except FileNotFoundError as e:
            raise HTTPException(400, f"file not found: {req.source_path}") from e
        session.steer(req.name, profile)
        return _profile_to_json(req.name, profile)

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
        from saklas import Profile

        _resolve_session_id(session, session_id)

        # Build a unified pool of {name: Profile} covering both registries.
        # Steering vectors come first (so they win on collision); probe
        # tensors are wrapped into Profile so the same cosine_similarity
        # call works for either source.
        pool: dict[str, "Profile"] = dict(session.vectors)
        try:
            for probe_name in session._monitor.probe_names:
                if probe_name in pool:
                    continue
                tensors = _probe_profile_tensors(session, probe_name)
                if tensors is None:
                    continue
                pool[probe_name] = Profile(tensors)
        except Exception:
            # Monitor not available — fall back to vectors-only pool.
            pass

        if names is not None and names.strip():
            requested = [n.strip() for n in names.split(",") if n.strip()]
            missing = [n for n in requested if n not in pool]
            if missing:
                raise HTTPException(404, f"names not loaded: {missing}")
            ordered = requested
        else:
            ordered = sorted(pool.keys())

        # Mahalanobis-only: ``cosine_similarity`` requires a whitener
        # covering each pair's shared layers.  Resolve it once; a missing
        # whitener is a 409 (regenerate the neutral cache).  A pair the
        # whitener doesn't fully cover still raises inside the loop and
        # lands as ``None`` for that cell.
        whitener = getattr(session, "whitener", None)
        if whitener is None:
            raise HTTPException(
                409,
                "correlation requires a Mahalanobis whitener; regenerate the "
                "neutral activation cache for this model (the Euclidean path "
                "is gone)",
            )

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
                try:
                    cos = pool[a].cosine_similarity(pool[b], whitener=whitener)
                    matrix[a][b] = round(float(cos), 6)
                except Exception:
                    matrix[a][b] = None
                shared = sorted(
                    set(pool[a].keys()) & set(pool[b].keys())
                )
                # Pair key sorted alphabetically so a__b == b__a in the lookup.
                key = "__".join(sorted([a, b]))
                layers_shared[key] = len(shared)
        return {
            "names": ordered,
            "matrix": matrix,
            "layers_shared": layers_shared,
        }

    @app.delete("/saklas/v1/sessions/{session_id}/vectors/{name}", status_code=204)
    def delete_vector(session_id: str, name: str):
        _resolve_session_id(session, session_id)
        if name not in session.vectors:
            raise HTTPException(404, f"vector '{name}' not found")
        session.unsteer(name)
        # Drop the vector from the default steering (if present) so the
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
    async def extract_vector(session_id: str, req: ExtractRequest, request: Request):
        _resolve_session_id(session, session_id)
        coerced: Any = _coerce_corpora(
            req.source if req.source is not None else req.name
        )

        def _run(on_progress: ProgressCallback) -> tuple[str, Any]:
            # Two pole corpora -> author + fit directly; a concept name ->
            # generate the corpora first.  Both land a 2-node ``pca`` manifold.
            if isinstance(coerced, tuple):
                positive, negative = coerced
                return session.extract_vector_from_corpora(
                    req.name, positive, negative,
                    on_progress=on_progress,
                    sae=req.sae, sae_revision=req.sae_revision,
                    role=req.role, namespace=req.namespace, force=req.force,
                )
            return session.extract(
                coerced, req.baseline,
                on_progress=on_progress,
                sae=req.sae, sae_revision=req.sae_revision,
                role=req.role, namespace=req.namespace, force=req.force,
            )

        accept = request.headers.get("accept", "application/json")
        if "text/event-stream" in accept:
            async def _job(on_progress: ProgressCallback) -> dict[str, Any]:
                canonical, profile = await asyncio.to_thread(_run, on_progress)
                registry_name = _extract_registry_name(canonical, req.namespace)
                if req.auto_register:
                    session.steer(registry_name, profile)
                return {
                    "done": True,
                    "profile": _profile_to_json(registry_name, profile),
                    "canonical": registry_name,
                }

            return progress_sse_response(
                session.lock,
                _job,
                error_message="extract failed",
                log_message=f"extract failed for session={session_id}",
            )

        progress_msgs: list[str] = []
        async with session.lock:
            canonical, profile = await asyncio.to_thread(_run, progress_msgs.append)
            registry_name = _extract_registry_name(canonical, req.namespace)
            if req.auto_register:
                session.steer(registry_name, profile)
        return {
            "canonical": registry_name,
            "profile": _profile_to_json(registry_name, profile),
            "progress": progress_msgs,
        }

    @app.post("/saklas/v1/sessions/{session_id}/vectors/bake")
    async def bake_vector(session_id: str, req: BakeVectorRequest):
        """Merge an expression of installed directions into a baked manifold.

        Wraps :func:`saklas.io.merge.merge_into_manifold` (model-scoped to
        the session's loaded model) — the merge lands a corpus-less
        ``fit_mode="baked"`` manifold — then folds the fitted tensor back to a
        steering Profile and registers it so it's immediately steerable.
        Returns the same profile-JSON shape ``GET /vectors/{name}`` produces.
        """
        from saklas.io.merge import merge_into_manifold, MergeError
        from saklas.io.paths import tensor_filename
        from saklas.core.manifold import load_manifold
        from saklas.core.vectors import folded_vector_directions
        from saklas.server.manifold_routes import _refuse_if_busy
        _resolve_session_id(session, session_id)

        async with session.lock:
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
                return Profile(folded_vector_directions(manifold))

            profile = await asyncio.to_thread(_load_folded)
            session.steer(req.name, profile)
        return _profile_to_json(req.name, profile)

    @app.get("/saklas/v1/sessions/{session_id}/vectors/{name}/diagnostics")
    def vector_diagnostics(session_id: str, name: str):
        """Per-layer ``||baked||`` histogram + diagnostics for a registered vector.

        Mirrors what ``saklas manifold why <concept> -m MODEL --json`` produces:
        a 16-bucket layer-magnitude histogram plus the ``diagnostics_by_layer``
        / ``diagnostics_summary`` block when the profile carries them.
        Drives the WHY-histogram strip in the web UI's probe rack.
        """
        from saklas.cli.runners import _summarize_diagnostics
        from saklas.core.histogram import HIST_BUCKETS, bucketize

        _resolve_session_id(session, session_id)
        # Probes and steering vectors share the per-layer ``dict[int,
        # Tensor]`` direction shape but live in different registries —
        # session.vectors holds steering profiles; a vector probe lives as a
        # flat manifold in ``session._monitor`` and folds back to the same
        # baked-direction view via ``_probe_profile_tensors``.  The
        # diagnostics endpoint serves either; the layer-norms drawer overlay
        # in the web UI hits this for every selected name (vector or probe).
        profile = session.vectors.get(name)
        if profile is None:
            try:
                folded = _probe_profile_tensors(session, name)
                profile = Profile(folded) if folded is not None else None
            except Exception:
                profile = None
        if profile is None:
            raise HTTPException(404, f"vector or probe '{name}' not found")

        layer_mags: list[tuple[int, float]] = sorted(
            ((layer, float(vec.norm().item())) for layer, vec in profile.items()),
            key=itemgetter(0),
        )
        buckets = bucketize(layer_mags, HIST_BUCKETS)
        # Buckets: ``(lo_layer, hi_layer, mean_norm)`` triples — same shape the
        # CLI ``manifold why`` text path renders, JSON-friendly here.
        bucket_payload = [
            {"lo": lo, "hi": hi, "mean_norm": round(mag, 6)}
            for lo, hi, mag in buckets
        ]

        # ``diagnostics`` is a Profile attribute; a probe folded from its
        # manifold carries no ``diagnostics`` block.  ``getattr`` covers both.
        diagnostics = getattr(profile, "diagnostics", None)
        # ``total_layers`` is the *model's* layer count, not the profile's
        # — the layer-norms drawer in the web UI fills layers absent from
        # the profile with zero so the user can read the DLS pattern.
        # Using ``len(profile)`` here used to lie when DLS dropped layers
        # (the drawer would stop at the profile's deepest layer instead
        # of the model's true depth).
        model_layers = int(session.model_info.get("num_layers") or len(profile))
        payload: dict[str, Any] = {
            "name": name,
            "model": session.model_id,
            "total_layers": model_layers,
            "histogram": {
                "buckets": HIST_BUCKETS,
                "data": bucket_payload,
            },
            "layers": [
                {"layer": layer, "magnitude": round(mag, 6)}
                for layer, mag in layer_mags
            ],
        }
        if diagnostics is not None:
            payload["diagnostics_by_layer"] = {
                str(layer): {k: round(float(v), 6) for k, v in metrics.items()}
                for layer, metrics in sorted(diagnostics.items())
            }
            payload["diagnostics_summary"] = _summarize_diagnostics(diagnostics)
        return payload

    # ----- probes / experiments / live traits ----------------------------

    from saklas.server.probe_routes import register_probe_routes
    register_probe_routes(app)

    from saklas.server.experiment_routes import register_experiment_routes
    register_experiment_routes(app)

    from saklas.server.traits_routes import register_traits_routes
    register_traits_routes(app)

    # ----- WebSocket token+probe co-stream -------------------------------

    @app.websocket("/saklas/v1/sessions/{session_id}/stream")
    async def session_stream(websocket: WebSocket, session_id: str):
        # NOTE: only ``session_id == "default"`` is actually reachable
        # here — HF model ids contain '/' and the FastAPI path parameter
        # is not declared ``{session_id:path}``, so the model-id branch
        # is an HTTP-route convenience only.  Kept as a no-op guard.
        if not ws_auth_ok(websocket):
            await websocket.close(code=1008, reason="unauthorized")
            return
        if session_id not in (_SINGLE_SESSION_ID, session.model_id):
            await websocket.accept()
            await websocket.close(code=1008, reason="session not found")
            return
        await websocket.accept()

        # Single perpetual reader.  ``websocket.receive_json()`` is bound
        # to a per-connection ``recv_in_progress`` flag in the underlying
        # ``websockets`` library; cancelling a pending receive doesn't
        # clear the flag immediately, so any handler that called
        # ``receive_json()`` while another concurrent (even just-cancelled)
        # caller was pending tripped a "cannot call recv while another
        # coroutine is already waiting" RuntimeError.  Routing every
        # incoming frame through one queue lets both the outer dispatch
        # loop and the in-flight generation share the read side without
        # ever overlapping calls into the WS.
        incoming: asyncio.Queue[Any] = asyncio.Queue()
        _DISCONNECT = object()

        async def _reader():
            try:
                while True:
                    msg = await websocket.receive_json()
                    await incoming.put(msg)
            except WebSocketDisconnect:
                await incoming.put(_DISCONNECT)
            except Exception as e:
                # Surface any other read-side failure into the queue so
                # the dispatcher can close cleanly instead of leaking.
                await incoming.put({"_reader_error": str(e), "_type": type(e).__name__})

        reader_task = asyncio.create_task(_reader())

        # Loom: subscribe to ``LoomMutated`` for the connection's
        # lifetime and forward as ``tree_mutated`` frames.  Also tag
        # ``begin_assistant`` events into ``node_created`` so the client
        # can pre-allocate render slots before token frames arrive.  Held
        # in a queue + forwarder task so the EventBus callback (which
        # runs on the gen thread) never touches the WS directly.
        loop = asyncio.get_running_loop()
        tree_event_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        # ``websocket.send_json`` is not safe for concurrent callers —
        # starlette serializes per-call but two tasks can interleave
        # bytes on the wire and corrupt the frame sequence.  This lock
        # is the single send-side serializer the connection uses; both
        # the generate-handler and the tree-forwarder acquire it before
        # every send.
        ws_send_lock = asyncio.Lock()

        async def _send_json(payload: Any) -> None:
            async with ws_send_lock:
                await websocket.send_json(payload)

        def _queue_tree_event(payload: dict[str, Any]) -> None:
            with suppress(Exception):
                loop.call_soon_threadsafe(tree_event_queue.put_nowait, payload)

        def _on_loom_event(event: object) -> None:
            if not isinstance(event, LoomMutated):
                return
            try:
                tree = session.tree
                added_nodes = [
                    _node_json(session, nid)
                    for nid in event.added
                    if tree.has(nid)
                ]
            except Exception:
                added_nodes = []
            mutated_payload: dict[str, Any] = {
                "type": "tree_mutated",
                "op": event.op,
                "rev": event.rev,
                "added": added_nodes,
                "removed": list(event.removed),
                "updated": [
                    _node_json(session, nid)
                    for nid in event.updated
                    if session.tree.has(nid)
                ],
                "active_node_id": event.active_node_id,
            }
            _queue_tree_event(mutated_payload)
            # ``begin_assistant`` and ``branch`` both materialize a new
            # node — surface a separate ``node_created`` event with the
            # parent + role so the client can allocate a render slot
            # without waiting for the assistant text to start streaming.
            if event.op in ("begin_assistant", "branch", "add_user"):
                for nid in event.added:
                    try:
                        node = session.tree.get(nid)
                    except Exception:
                        continue
                    node_payload = {
                        "type": "node_created",
                        "node_id": nid,
                        "parent_id": node.parent_id,
                        "role": node.role,
                        "rev": event.rev,
                    }
                    _queue_tree_event(node_payload)

        loom_unsub = session.events.subscribe(_on_loom_event)

        async def _tree_forwarder():
            """Forward tree-mutated / node-created events as WS frames.

            Runs as a dedicated task for the connection's lifetime so
            tree mutations from any source (this WS, a REST route on a
            different connection, the gen loop) reach the client without
            interleaving with the per-turn token loop.
            """
            try:
                while True:
                    payload = await tree_event_queue.get()
                    try:
                        await _send_json(payload)
                    except Exception:
                        return
            except asyncio.CancelledError:
                return

        forwarder_task = asyncio.create_task(_tree_forwarder())

        deferred_incoming: deque[Any] = deque()

        async def _cancel_and_wait(task: asyncio.Task[Any]) -> None:
            task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await task

        def _stop_session_safely() -> None:
            with suppress(Exception):
                session.stop()

        try:
            while True:
                msg = (
                    deferred_incoming.popleft()
                    if deferred_incoming
                    else await incoming.get()
                )
                if msg is _DISCONNECT:
                    raise WebSocketDisconnect(code=1000)
                if isinstance(msg, dict) and "_reader_error" in msg:
                    raise RuntimeError(msg["_reader_error"])

                mtype = msg.get("type") if isinstance(msg, dict) else None
                if mtype == "generate":
                    try:
                        parsed = WSGenerateMessage(**msg)
                    except Exception as e:
                        await _send_json({
                            "type": "error",
                            "message": f"invalid generate message: {e}",
                            "code": "ValidationError",
                        })
                        continue
                    await _ws_handle_generate(
                        session, parsed, app.state.default_steering, incoming,
                        deferred_incoming, _send_json,
                    )
                elif mtype == "stop":
                    # Idle-state stop: nothing in flight.
                    continue
                else:
                    await _send_json({
                        "type": "error",
                        "message": f"unknown message type: {mtype!r}",
                        "code": "UnknownMessageType",
                    })
        except WebSocketDisconnect:
            # Ensure any stray generation is signaled.
            _stop_session_safely()
            return
        except Exception as e:
            try:
                await _send_json({
                    "type": "error",
                    "message": str(e),
                    "code": type(e).__name__,
                })
            finally:
                with suppress(Exception):
                    await websocket.close(code=1011)
        finally:
            # Drop the loom subscription before tearing down the reader
            # so the EventBus stops dispatching into a queue nobody
            # reads.
            with suppress(Exception):
                loom_unsub()
            await _cancel_and_wait(forwarder_task)
            # Reader holds the only ``receive_json()`` call on the WS.
            # Cancel + await so the cancellation propagates fully before
            # the connection tears down.
            await _cancel_and_wait(reader_task)


async def _ws_handle_generate(
    session: SaklasSession,
    msg: WSGenerateMessage,
    default_steering: "Steering | None",
    incoming: asyncio.Queue[Any],
    deferred_incoming: "deque[Any]",
    send_json: Callable[[Any], Awaitable[None]],
) -> None:
    """Run one generate turn and stream token/done/error events.

    Concurrency design: the synchronous ``session.generate_stream`` is
    driven from a worker thread via ``asyncio.to_thread``.  Its
    ``on_token`` callback is invoked on the worker thread; it bridges
    into the asyncio loop by calling
    ``loop.call_soon_threadsafe(queue.put_nowait, event)``.  The main
    coroutine races two tasks: one pulls ``TokenEvent``s from a local
    queue and forwards them as ``{type: "token", ...}`` frames; the
    other pulls client frames from the shared ``incoming`` queue
    (populated by the connection's single reader task) so an in-flight
    ``{type: "stop"}`` can call ``session.stop()`` without blocking on
    the token loop.

    ``asyncio.wait(..., FIRST_COMPLETED)`` is used in a loop: whenever
    the incoming task returns a stop frame we signal the session and
    keep draining tokens until the worker joins; whenever the queue
    delivers a sentinel we finish.  The WS stays open across generate
    turns — a client can submit ``{type: "generate", ...}`` again after
    ``done``, and the perpetual reader keeps feeding the shared queue
    between turns so we never have two ``receive_json()`` calls in
    flight.

    **Loom (v2.3)**: ``parent_node_id`` attaches the assistant node
    under a specific tree node; ``n>1`` fans out N siblings serially
    (per decision 7 in the plan — N-way gen is serial in v1).  Each
    sibling produces its own ``started`` / token-stream / ``done``
    triplet, all tagged with the assistant node id.  ``tree_mutated``
    and ``node_created`` events ride the connection-level subscription
    in ``session_stream``; this handler only emits the per-sibling
    ``started`` / ``token`` / ``done`` frames.
    """
    loop = asyncio.get_running_loop()

    sampling = _build_sampling(msg.sampling)
    try:
        steering = _build_steering(msg.steering, default_steering)
    except SaklasError as e:
        # ``_build_steering`` -> ``parse_expr`` -> ``resolve_pole`` can
        # raise ``SteeringExprError`` / ``AmbiguousSelectorError`` /
        # ``AmbiguousVariantError`` on malformed or colliding input.
        # FastAPI's ``@app.exception_handler(SaklasError)`` doesn't apply
        # to WebSocket routes, so without this guard the exception falls
        # through to the outer reader loop's ``except Exception`` which
        # closes the socket with code 1011. A 400-grade user mistake
        # shouldn't kill the connection — send the error frame and let
        # the client try again on the same WS.
        status, message = e.user_message()
        await send_json({
            "type": "error",
            "message": message,
            "code": type(e).__name__,
            "status": status,
        })
        return

    n = msg.n if msg.n and msg.n > 0 else 1
    if n < 1:
        await send_json({
            "type": "error",
            "message": f"n must be >= 1, got {n}",
            "code": "ValueError",
            "status": 400,
        })
        return

    parent_node_id = msg.parent_node_id

    # Logit fork: when ``fork_node_id`` is set the worker calls
    # ``session.fork_from_token`` instead of ``session.generate``.  All
    # three fork fields must be present together.
    is_fork = msg.fork_node_id is not None
    if is_fork and (msg.fork_raw_index is None or msg.fork_alt_token_id is None):
        await send_json({
            "type": "error",
            "message": (
                "fork requires fork_node_id, fork_raw_index, and "
                "fork_alt_token_id together"
            ),
            "code": "ValueError",
            "status": 400,
        })
        return

    # Answer-prefill: when ``prefill_node_id`` is set the worker calls
    # ``session.prefill_assistant`` instead of ``session.generate``.  It
    # needs ``prefill_text`` alongside it, and can't co-exist with a fork.
    is_prefill = msg.prefill_node_id is not None
    if is_prefill and (msg.prefill_text is None or msg.prefill_text == ""):
        await send_json({
            "type": "error",
            "message": "prefill requires prefill_node_id and prefill_text together",
            "code": "ValueError",
            "status": 400,
        })
        return
    if is_prefill and is_fork:
        await send_json({
            "type": "error",
            "message": "a generate message cannot be both a fork and a prefill",
            "code": "ValueError",
            "status": 400,
        })
        return

    # Commit (Ctrl+Enter on either surface): land a turn under
    # ``parent_node_id`` without running a decode.  Short-circuits the
    # n-way fan-out / streaming worker entirely — one mutation, one
    # ``done`` event, no token frames.  Mutually exclusive with prefill
    # and fork (rejected above by symmetry).
    is_commit = msg.commit_role is not None
    if is_commit:
        if msg.commit_text is None or msg.commit_text == "":
            await send_json({
                "type": "error",
                "message": "commit requires commit_role and commit_text together",
                "code": "ValueError",
                "status": 400,
            })
            return
        if msg.commit_role not in ("user", "assistant"):
            await send_json({
                "type": "error",
                "message": (
                    f"commit_role must be 'user' or 'assistant', "
                    f"got {msg.commit_role!r}"
                ),
                "code": "ValueError",
                "status": 400,
            })
            return
        if is_fork or is_prefill:
            await send_json({
                "type": "error",
                "message": (
                    "a generate message cannot mix commit with fork or prefill"
                ),
                "code": "ValueError",
                "status": 400,
            })
            return
        if msg.commit_role == "assistant" and parent_node_id is None:
            await send_json({
                "type": "error",
                "message": (
                    "commit_role='assistant' requires parent_node_id "
                    "(the user node the authored turn hangs off)"
                ),
                "code": "ValueError",
                "status": 400,
            })
            return

        generation_id = uuid.uuid4().hex[:12]
        commit_text = str(msg.commit_text)
        await send_json({
            "type": "started",
            "generation_id": generation_id,
            "node_id": None,
            "sibling_index": 0,
            "sibling_count": 1,
        })
        # Per-message role labels ride the commit's sampling block too
        # (roleplay scaffold).  Raw / flat commits carry no chat-template
        # role, so labels are suppressed there.
        commit_user_role = (
            msg.sampling.user_role if msg.sampling is not None else None
        ) or None
        commit_asst_role = (
            msg.sampling.assistant_role if msg.sampling is not None else None
        ) or None
        async with session.lock:
            try:
                if msg.commit_role == "user":
                    # ``raw`` flags a flat (base-model) commit — the
                    # authored span may hang under a node of any role,
                    # so the user-under-user guard is lifted.
                    new_id = await asyncio.to_thread(
                        session.append_user_turn,
                        parent_node_id,
                        commit_text,
                        allow_any_parent=msg.raw,
                        role_label=None if msg.raw else commit_user_role,
                    )
                else:
                    # ``parent_node_id`` is non-None here (validated above
                    # for the assistant role); narrow for the type-checker.
                    assert parent_node_id is not None
                    new_id = await asyncio.to_thread(
                        session.append_assistant_turn,
                        parent_node_id,
                        commit_text,
                        role_label=None if msg.raw else commit_asst_role,
                    )
            except SaklasError as e:
                status, message = e.user_message()
                await send_json({
                    "type": "error",
                    "message": message,
                    "code": type(e).__name__,
                    "status": status,
                    "node_id": None,
                    "sibling_index": 0,
                })
                return
        await send_json({
            "type": "done",
            "result": {
                "kind": "commit",
                "role": msg.commit_role,
                "text": commit_text,
                "node_id": new_id,
                "finish_reason": "stop",
                "per_token_probes": [],
                "mean_logprob": None,
                "mean_surprise": None,
            },
            "node_id": new_id,
            "sibling_index": 0,
            "sibling_count": 1,
        })
        return

    # Per-sibling seed schedule: when n>1, derive deterministic per-
    # sibling seeds from the request seed (or fresh entropy).  Single
    # streams (n=1) use the user's seed verbatim.
    from saklas.core.loom import derive_seed_schedule
    base_seed = sampling.seed if sampling is not None else None
    seeds: list[int | None]
    seeds = [base_seed] if n == 1 else list(derive_seed_schedule(base_seed, n))

    def _stop_session_safely() -> None:
        with suppress(Exception):
            session.stop()

    # Acquire the session lock for the full N-way batch lifetime so
    # concurrent WS clients serialize FIFO instead of overlapping.
    # ``session.generate_stream`` itself uses the threading ``_gen_lock``
    # to gate the actual generation, but the async-level lock is what
    # queues HTTP/WS endpoints fairly.
    async with session.lock:
        for sibling_idx, seed_i in enumerate(seeds):
            generation_id = uuid.uuid4().hex[:12]

            # Per-sibling sampling override carrying the derived seed.
            if n == 1 and seed_i is None:
                per_sibling_sampling = sampling
            else:
                from dataclasses import replace as _dc_replace
                base_sc = sampling if sampling is not None else SamplingConfig()
                per_sibling_sampling = _dc_replace(base_sc, seed=seed_i)

            token_queue: asyncio.Queue[Any] = asyncio.Queue()
            _SENTINEL = object()
            # The tree assigns the assistant node id at ``begin_assistant``
            # time inside ``_generate_core``; we don't know it before the
            # gen starts.  The on_token callback reads the live active
            # node off the tree (which is set to the streaming assistant
            # node for the lifetime of the gen).
            current_node_holder: list[str | None] = [None]

            def _on_token(
                text: str,
                is_thinking: bool,
                tid: int | None,
                lp: float | None,
                top_alts: list[Any] | None,
                perplexity: float | None = None,
                _node_holder: list[str | None] = current_node_holder,
                _token_queue: asyncio.Queue[Any] = token_queue,
            ) -> None:
                event = build_token_event(
                    session,
                    _node_holder,
                    text=text,
                    is_thinking=is_thinking,
                    tid=tid,
                    lp=lp,
                    top_alts=top_alts,
                )
                loop.call_soon_threadsafe(_token_queue.put_nowait, event)
            _on_token_flags: Any = _on_token
            _on_token_flags._saklas_wants_live_scores = True
            _on_token_flags._saklas_wants_per_layer_scores = True

            result_holder: list[GenerationResult | RunSet] = []
            error_holder: list[BaseException] = []

            # Recipe-override (phase 5): accept either a mode string or a
            # partial-recipe expression.  We pass it through ``generate``
            # so the engine resolves the overlay against the parent's
            # recipe; ``session.regen_with_modifier`` is the matching
            # higher-level wrapper but the WS path already has the
            # required context.
            recipe_override = msg.recipe_override

            def _worker(
                _sampling: SamplingConfig | None = per_sibling_sampling,
                _on_token: Callable[..., Any] = _on_token,
                _result_holder: list[GenerationResult | RunSet] = result_holder,
                _error_holder: list[BaseException] = error_holder,
                _token_queue: asyncio.Queue[Any] = token_queue,
                _sentinel: object = _SENTINEL,
                _recipe_override: Any = recipe_override,
            ) -> None:
                try:
                    if msg.fork_node_id is not None:
                        # Fork: recipe / sampling / parent all come from
                        # the source node inside ``fork_from_token``; the
                        # WS-level steering/sampling/n fields are ignored.
                        result = session.fork_from_token(
                            msg.fork_node_id,
                            int(msg.fork_raw_index),  # pyright: ignore[reportArgumentType]  # guarded non-None by is_fork check above; int() accepts int|None only at runtime with None already excluded
                            int(msg.fork_alt_token_id),  # pyright: ignore[reportArgumentType]  # guarded non-None by is_fork check above; int() accepts int|None only at runtime with None already excluded
                            on_token=_on_token,
                        )
                    elif msg.prefill_node_id is not None:
                        # Prefill: anchor / parent come from the user node
                        # inside ``prefill_assistant``; ``input`` is
                        # ignored.  ``steering`` / ``sampling`` ride through
                        # like a normal generate; ``thinking`` is forced
                        # off (the prefill is an answer, not a thought).
                        result = session.prefill_assistant(
                            msg.prefill_node_id,
                            str(msg.prefill_text),
                            steering=steering,
                            sampling=_sampling,
                            on_token=_on_token,
                        )
                    else:
                        gen_kwargs: dict[str, Any] = {
                            "steering": steering,
                            "sampling": _sampling,
                            "stateless": msg.stateless,
                            "raw": msg.raw,
                            "thinking": msg.thinking,
                            "on_token": _on_token,
                            "parent_node_id": parent_node_id,
                        }
                        if _recipe_override is not None:
                            gen_kwargs["recipe_override"] = _recipe_override
                        result = session.generate(msg.input, **gen_kwargs)
                    _result_holder.append(result)
                except BaseException as e:
                    _error_holder.append(e)
                finally:
                    loop.call_soon_threadsafe(_token_queue.put_nowait, _sentinel)

            await send_json({
                "type": "started",
                "generation_id": generation_id,
                # ``node_id`` is filled in lazily by the first token
                # event (the assistant node is created inside
                # ``_generate_core``); ``started`` includes the request-
                # level context the client needs to allocate state.
                "node_id": None,
                "sibling_index": sibling_idx,
                "sibling_count": n,
            })

            worker_task = asyncio.create_task(asyncio.to_thread(_worker))

            # Race two queue reads — token frames from the worker and
            # client frames from the connection's perpetual reader.
            # Neither side ever calls ``websocket.receive_json()``
            # directly, so the underlying ``recv_in_progress`` flag is
            # owned by the reader task alone for the connection's
            # lifetime.
            done = False
            stop_signaled = False
            try:
                while not done:
                    token_get = asyncio.create_task(token_queue.get())
                    client_get = asyncio.create_task(incoming.get())
                    finished, pending = await asyncio.wait(
                        {token_get, client_get}, return_when=asyncio.FIRST_COMPLETED,
                    )
                    if client_get in finished:
                        incoming_msg = client_get.result()
                        # ``_DISCONNECT`` / reader-error sentinels:
                        # signal the worker to wind down; let the outer
                        # loop propagate the disconnect on the next
                        # iteration.
                        if isinstance(incoming_msg, dict):
                            if incoming_msg.get("type") == "stop":
                                _stop_session_safely()
                                stop_signaled = True
                            elif "_reader_error" in incoming_msg:
                                _stop_session_safely()
                                stop_signaled = True
                                # Defer so the outer dispatch loop
                                # surfaces the error after we wind down.
                                deferred_incoming.append(incoming_msg)
                            else:
                                # Out-of-band frame during a generation —
                                # defer so the outer loop sees it
                                # after this turn finishes.  Most likely
                                # an early ``{type: "generate"}`` from a
                                # client that didn't wait for ``done``.
                                #
                                # Do not put it back on ``incoming`` here:
                                # the next loop iteration would consume it
                                # immediately, cancel token_queue.get(), and
                                # spin until the worker happened to have a
                                # token already queued.
                                deferred_incoming.append(incoming_msg)
                        else:
                            # Disconnect sentinel from the reader.
                            _stop_session_safely()
                            stop_signaled = True
                            deferred_incoming.append(incoming_msg)
                    if token_get in finished:
                        item = token_get.result()
                        if item is _SENTINEL:
                            done = True
                        else:
                            await send_json(item)
                    for task in pending:
                        task.cancel()
                    if pending:
                        await asyncio.gather(*pending, return_exceptions=True)
            finally:
                # Drain any residual events the worker pushed between
                # sentinel and join — should be none because the
                # sentinel is last, but cheap insurance.
                await worker_task

            if error_holder and not result_holder:
                exc = error_holder[0]
                await send_json({
                    "type": "error",
                    "message": str(exc),
                    "code": type(exc).__name__,
                    "node_id": current_node_holder[0],
                    "sibling_index": sibling_idx,
                })
                # On error inside a sibling, abort the remaining fan-out
                # rather than continuing with stale state.
                return

            result = result_holder[0] if result_holder else None
            result_json = _result_to_json(result)
            if result is not None:
                result_json["per_token_probes"] = _per_token_probes(
                    session, getattr(result, "token_count", 0) or 0,
                )
                # Per-attached-manifold-probe aggregate readings ride on
                # the ``done`` event so a WS client picks up the
                # geometric channel alongside the existing vector-probe
                # ``per_token_probes`` block.  Empty dict when no
                # manifold probe is attached.
                mf_readings = getattr(result, "probe_readings", None) or {}
                if mf_readings:
                    result_json["probe_readings"] = {
                        k: v.to_dict() for k, v in mf_readings.items()
                    }
            else:
                result_json["per_token_probes"] = []
            # Phase 1 logit pass: stamp the per-turn logprob rollup on the
            # ``done`` event so subscribers (loom sidebar's sort-by-surprise,
            # webui chat-header summary) don't need to re-fetch the node.
            # Source of truth is the finalized loom node, populated by
            # :meth:`LoomTree.finalize_assistant` upstream of this branch.
            # Stateless gens / pre-logit-pass replays land with ``None``
            # which the wire layer passes through transparently.
            mean_logprob_out: float | None = None
            mean_surprise_out: float | None = None
            finalized_node_id = current_node_holder[0]
            if finalized_node_id is not None:
                try:
                    node = session.tree.nodes.get(finalized_node_id)
                    if node is not None:
                        mean_logprob_out = node.mean_logprob
                        mean_surprise_out = node.mean_surprise
                except Exception:
                    # Defensive: tree access during shutdown / mocked
                    # session edge cases. Default-None values keep the
                    # wire payload well-formed.
                    pass
            result_json["mean_logprob"] = mean_logprob_out
            result_json["mean_surprise"] = mean_surprise_out
            await send_json({
                "type": "done",
                "result": result_json,
                "node_id": current_node_holder[0],
                "sibling_index": sibling_idx,
                "sibling_count": n,
            })

            # Mid-batch stop honors the plan's decision (#7 / phase 1
            # spec): "stop_requested cancels the currently-streaming
            # sibling. Remaining queued siblings are skipped, not
            # started."
            if stop_signaled:
                break
