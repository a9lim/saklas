"""Response schemas for the non-streaming native ``/saklas/v1/*`` routes.

Fifty-one strict *request* models and zero response models is how the wire
contract ended up maintained by hand in TypeScript.  This module is the
Python-side declaration of what each native route actually returns; the
route functions annotate their return type with these ``TypedDict``s, so
FastAPI emits a real OpenAPI response schema and
``scripts/generate_webui_types.py`` renders the dashboard's REST types from
it.  Wire drift is then a failed CI diff rather than a mis-rendered panel.

Two conventions carry the whole file:

- **Names match the dashboard.**  Each TypedDict is named exactly as the
  TypeScript interface the generator emits, so a schema change lands in the
  webui under the name consumers already import.  The one family that keeps
  its engine names is :mod:`saklas.core.measurements` — the generator
  carries a small explicit rename table for it rather than renaming the
  engine's own vocabulary.
- **Open where the producer is not this module.**  ``_Open`` marks a shape
  whose payload comes from an engine or io serializer that may legitimately
  carry more than the wire contract names (opaque token rows, per-fit
  diagnostics, HF search rows).  Pydantic response validation drops keys a
  closed model does not declare, and silently truncating a producer's
  payload is a worse failure than an under-specified type — so those shapes
  declare ``extra="allow"`` and the generator renders an index signature.

Streaming surfaces are deliberately absent: the SSE progress frames and the
WebSocket frame vocabulary are not OpenAPI-describable and stay hand-written
in ``webui/src/lib/types.ts``.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import ConfigDict
from typing_extensions import NotRequired, TypedDict

#: Response-model config for shapes whose payload is produced outside this
#: module.  Extra keys are preserved rather than silently dropped by the
#: response validator; the generator turns it into a TS index signature.
_Open = ConfigDict(extra="allow")


# ---------------------------------------------------------------- sessions --

class SamplingFields(TypedDict):
    """``session_info.config`` — the session's default sampling knobs."""

    temperature: float | None
    top_p: float | None
    top_k: int | None
    max_tokens: int | None
    system_prompt: str | None
    thinking: bool | None


class InstrumentCapabilities(TypedDict):
    """What a read family supports, declared by the server rather than guessed."""

    sources: bool
    preparations: list[str]
    token_readout: bool
    source_switch: bool


class GeometryLiveState(TypedDict):
    """Geometry live state — an all-or-nothing per-token scoring switch."""

    enabled: bool


class LensLiveState(TypedDict):
    """Lens live state — the resolved fitted-layer set."""

    enabled: bool
    layers: list[int] | None


class SaeLiveState(TypedDict):
    """SAE live state — the resident layer + source."""

    enabled: bool
    layer: int | None
    source: str | None


#: Geometry's key set is a prefix of the other two, and a TypedDict ignores
#: keys it does not declare — so a smart-union match on the *narrowest*
#: member would silently strip ``layers`` / ``layer`` / ``source`` off the
#: wire.  Widest first is what makes the union exact: each family's payload
#: fails every member above its own on a missing required key.  The
#: dashboard narrows the same way (``"layers" in live``).
InstrumentLiveState = SaeLiveState | LensLiveState | GeometryLiveState


class InstrumentFamilyBlock(TypedDict):
    """One read family's state.

    The SAME block is listed by ``GET .../instruments`` and embedded in
    ``session_info.instruments`` — one representation of instrument state.
    """

    family: str
    live: InstrumentLiveState
    source: str | None
    probes: list[str]
    capabilities: InstrumentCapabilities


class SessionInfo(TypedDict):
    """``session_info()`` — the single session's full descriptor."""

    id: str
    model_id: str
    device: str
    dtype: str
    created: float
    config: SamplingFields
    profiles: list[str]
    probes: list[str]
    history_length: int
    supports_thinking: bool
    thinking_is_optional: bool
    is_base_model: bool
    jlens_fitted: bool
    instruments: list[InstrumentFamilyBlock]
    default_steering: str | None
    role_substitution_supported: bool
    user_role_supported: bool
    default_assistant_role: str | None
    default_user_role: str | None
    scene_mode: bool
    thinking_input_supported: bool
    strips_history_thinking: bool


class SessionListResponse(TypedDict):
    sessions: list[SessionInfo]


class SteeringValidationResponse(TypedDict):
    """``POST .../steering/validate`` — a form result, not a failed request."""

    valid: bool
    expression: str
    error: str | None


# -------------------------------------------------------------------- loom --

class RecipeSamplingJSON(TypedDict):
    """A stamped recipe's sampling block.

    ``Recipe.to_dict`` walks ``fields(SamplingConfig)``, so every field is
    present — the two container fields are normalized on the way out
    (``stop`` to a list, ``logit_bias`` to string-keyed floats).
    """

    temperature: float | None
    top_p: float | None
    top_k: int | None
    max_tokens: int | None
    seed: int | None
    stop: list[str] | None
    logit_bias: dict[str, float] | None
    presence_penalty: float
    frequency_penalty: float
    logprobs: int | None
    return_hidden: bool
    return_top_k: int
    user_role: str | None
    assistant_role: str | None
    persist_per_layer_scores: bool
    persist_subspace_coords: bool
    return_probe_readings: bool


class RecipeJSON(TypedDict):
    """``Recipe.to_dict()`` — the per-node reproducibility receipt."""

    steering: str | None
    sampling: RecipeSamplingJSON | None
    thinking: bool | None
    seed: int | None
    probes: list[str]
    probe_hashes: dict[str, str]


class CastMemberJSON(TypedDict):
    """``CastMember.to_dict()`` plus the server-derived ``origin``."""

    recipe: RecipeJSON | None
    notes: str
    origin: NotRequired[str]


class LoomNodeJSON(TypedDict):
    """``LoomNode.to_dict(include_tokens=True)``.

    ``tokens`` / ``thinking_tokens`` stay opaque here: a token row is an
    open, feature-dependent capture record (the ``measurements`` envelope,
    logit alternatives, per-layer heatmap channels), and running thousands
    of them through response validation on every tree fetch would be both
    slow and a rejection risk for restored trees.  The generator maps them
    onto the hand-written ``LoomTokenRowJSON``.
    """

    id: str
    parent_id: str | None
    role: Literal["user", "assistant", "system"]
    text: str
    role_label: str | None
    thinking_text: str | None
    aggregate_readings: dict[str, float]
    applied_steering: str | None
    finish_reason: str | None
    starred: bool
    notes: str
    created_at: float
    edited_at: float | None
    edit_count: int
    mean_logprob: float | None
    mean_surprise: float | None
    recipe: RecipeJSON | None
    tokens: list[dict[str, Any]] | None
    thinking_tokens: list[dict[str, Any]] | None
    raw_token_ids: list[int] | None


class LoomNodeDetailJSON(LoomNodeJSON):
    """A single node plus its ordered child ids (the mutation responses)."""

    children: list[str]


class LoomTreeJSON(TypedDict):
    """``LoomTree.to_dict(include_tokens=True)`` + the effective cast roster."""

    tree_format: int
    saklas_version: str
    model_id: str | None
    session_id: str | None
    name: str | None
    rev: int
    root_id: str
    active_node_id: str
    nodes: list[LoomNodeJSON]
    children_of: dict[str, list[str]]
    cast: dict[str, CastMemberJSON]


class TreeRestoreResponse(TypedDict):
    rev: int
    root_id: str
    active_node_id: str
    nodes: int


class ChatMessageJSON(TypedDict):
    role: Literal["user", "assistant", "system"]
    content: str


class ActivePathJSON(TypedDict):
    """Active path: chat messages + a parallel node-id list."""

    active_node_id: str
    rev: int
    messages: list[ChatMessageJSON]
    node_ids: list[str]


class TreeBranchResponse(TypedDict):
    node_id: str
    node: LoomNodeDetailJSON
    active_path: ActivePathJSON


class TreeDeleteResponse(TypedDict):
    removed: int


class CastRosterResponse(TypedDict):
    cast: dict[str, CastMemberJSON]


class CastMemberResponse(TypedDict):
    label: str
    member: CastMemberJSON


class TranscriptResponse(TypedDict):
    yaml: str
    node_id: str


class TranscriptLoadResponseJSON(TypedDict):
    leaf_id: str
    rev: int
    guards: list[str]


class EdgeLabelResponse(TypedDict):
    """``GET .../tree/edge-label`` — empty string when the recipes match."""

    label: str


class FilterMatchesJSON(TypedDict):
    expr: str
    matching_node_ids: list[str]


class DiffTextSpanJSON(TypedDict):
    state: str
    text: str


class DiffReadingDeltaJSON(TypedDict):
    name: str
    delta: float
    a_value: float
    b_value: float


class DiffTokenSpanJSON(TypedDict):
    a_index: int
    b_index: int
    a_text: str
    b_text: str
    aligned: bool
    reading_deltas: list[DiffReadingDeltaJSON]


class NodeDiffJSON(TypedDict):
    a_id: str
    b_id: str
    parent_id: str | None
    a_text: str
    b_text: str
    a_applied_steering: str | None
    b_applied_steering: str | None
    parent_applied_steering: str | None
    steering_delta: str
    parent_to_a_delta: str
    parent_to_b_delta: str
    text: list[DiffTextSpanJSON]
    readings: list[DiffReadingDeltaJSON]
    per_token: list[DiffTokenSpanJSON]


class JointLogprobRowJSON(TypedDict):
    a_index: int
    b_index: int
    a_text: str
    b_text: str
    aligned: bool
    lp_a_in_a: float | None
    lp_b_in_b: float | None
    lp_a_in_b: float | None
    lp_b_in_a: float | None
    rank_changed: bool
    approx_kl: float | None


class JointLogprobsJSON(TypedDict):
    a_id: str
    b_id: str
    parent_id: str | None
    rows: list[JointLogprobRowJSON]
    n_rank1_changed: int


# ------------------------------------------------------------------ probes --

#: A manifold's domain spec rides the wire verbatim from the io layer: a
#: ``type``-tagged union (``box`` carries ``axes``, ``sphere`` carries
#: ``dim``, ``custom`` is the JSON-authored escape hatch) plus the empty
#: ``{}`` of an unfitted discover folder.  A TypedDict cannot express a
#: tagged union, and flattening it to optional fields would cost the
#: dashboard its ``domain.type === "box"`` narrowing — so the Python side
#: stays open and the generator maps it onto the hand-written
#: ``ManifoldDomain`` union in ``types.ts``.
ManifoldDomainSpec = dict[str, Any]


class GeometryProbeInfo(TypedDict):
    """One attached geometry (Monitor subspace) probe, any rank."""

    family: Literal["geometry"]
    name: str
    manifold: str
    top_n: int
    layers: list[int]
    node_labels: list[str]
    node_count: int
    domain: ManifoldDomainSpec
    intrinsic_dim: int
    feature_space: str
    is_affine: bool
    node_coords: list[list[float]] | None


class LensProbeInfo(TypedDict):
    """One pinned J-lens token probe (the readout channel)."""

    family: Literal["lens"]
    name: str
    layers: list[int]
    intrinsic_dim: int
    feature_space: str
    word: str
    token_id: int | None


class SaeProbeInfo(TypedDict):
    """One pinned resident SAE feature probe (the encoder readout channel)."""

    family: Literal["sae"]
    name: str
    layers: list[int]
    intrinsic_dim: int
    feature_space: str
    feature_id: int
    label: str | None
    max_act: float | None


ProbeInfo = GeometryProbeInfo | LensProbeInfo | SaeProbeInfo


class ProbeListResponse(TypedDict):
    probes: list[ProbeInfo]


class ProbeDefaultsResponse(TypedDict):
    """``GET .../probes/defaults`` — ``{tag: [manifold name]}``."""

    defaults: dict[str, list[str]]


class ProbeOverlay(TypedDict):
    """A curved fit's curve/surface overlay, sampled into the whitened frame."""

    kind: str
    points: list[list[float]]
    grid_shape: NotRequired[list[int]]


class ProbeLayerGeometry(TypedDict):
    """One fitted layer's geometry for the probe-inspector plot."""

    layer: int
    rank: int
    intrinsic_dim: int
    is_affine: bool
    node_white: list[list[float]]
    neutral_white: list[float]
    pca_rotation: list[list[float]] | None
    explained_variance_pcs: list[float] | None
    mahalanobis_share: float
    overlay: ProbeOverlay | None


class ProbeGeometryResponse(TypedDict):
    name: str
    manifold: str
    intrinsic_dim: int
    is_affine: bool
    node_labels: list[str]
    rank_uniform: bool
    layers: dict[str, ProbeLayerGeometry]


# ---------------------------------------------------------------- profiles --

class VectorInfo(TypedDict):
    """``profile_to_json`` — a registered steering profile's identity."""

    name: str
    layers: list[int]
    metadata: dict[str, Any]


class ProfileListResponse(TypedDict):
    profiles: list[VectorInfo]


class PairwiseCompareResponse(TypedDict):
    """Cross-layer whitened cosine matrix between two named profiles/probes."""

    a: str
    b: str
    metric: str
    layers_a: list[int]
    layers_b: list[int]
    matrix: list[list[float | None]]
    model: str | None


class CorrelationData(TypedDict):
    names: list[str]
    matrix: dict[str, dict[str, float | None]]
    layers_shared: dict[str, int]


class ExtractResponse(TypedDict):
    """``POST .../extract`` — the JSON (non-SSE) branch."""

    done: bool
    profile: VectorInfo
    canonical: str
    progress: NotRequired[list[str]]


# --------------------------------------------------------------- manifolds --

class ManifoldNodeDetail(TypedDict):
    """One node in the detail (``full=True``) manifold shape.

    ``coords`` is the authored layout for an authored folder, the derived
    per-model layout for a fitted discover folder, and ``null`` for a
    discover folder with no fit for the loaded model.
    """

    label: str
    coords: list[float] | None
    statements: list[str]
    role: str | None


class ManifoldFitInfo(TypedDict):
    """Per-fitted-tensor record in the manifold detail shape."""

    __pydantic_config__ = _Open  # type: ignore[misc]

    stem: str
    method: str
    feature_space: str
    node_count: int
    # A sidecar written before the corpus hash existed carries no
    # ``nodes_sha256``; the field is genuinely nullable on the wire.
    nodes_sha256: str | None
    fit_mode: str
    hyperparams: NotRequired[dict[str, Any]]
    diagnostics: NotRequired[dict[str, Any]]


class ManifoldInfo(TypedDict):
    """``_manifold_json`` — ``manifold_summary`` plus the session-aware extras.

    ``nodes`` / ``fitted`` are detail-only (``full=True``); ``advisories``
    rides the authoring response; ``done`` / ``layers_fitted`` /
    ``feature_space`` ride the generate + fit JSON branches.
    """

    namespace: str
    name: str
    description: str
    source: str
    tags: list[str]
    template_ref: str | None
    fit_mode: str
    is_discover: bool
    domain: ManifoldDomainSpec
    domain_label: str
    intrinsic_dim: int
    min_nodes: int | None
    node_count: int
    node_labels: list[str]
    node_coords: list[list[float]]
    node_roles: list[str | None]
    node_kinds: list[str | None]
    hyperparams: dict[str, Any]
    fitted_models: list[str]
    tensor_variants: dict[str, list[str]]
    fitted_for_session: bool
    stale: bool
    resolved_fit_mode: str | None
    nodes: NotRequired[list[ManifoldNodeDetail]]
    fitted: NotRequired[list[ManifoldFitInfo]]
    advisories: NotRequired[list[str]]
    done: NotRequired[bool]
    layers_fitted: NotRequired[int]
    feature_space: NotRequired[str]
    progress: NotRequired[list[str]]


class ManifoldListResponse(TypedDict):
    manifolds: list[ManifoldInfo]


class RemoteManifoldInfo(TypedDict):
    """One Hugging Face search row, as the picker renders it."""

    __pydantic_config__ = _Open  # type: ignore[misc]

    name: str
    namespace: str
    description: str
    tags: list[str]
    node_count: int
    domain_label: str
    fit_mode: str
    tensor_models: list[str]


class ManifoldSearchResponse(TypedDict):
    query: str
    results: list[RemoteManifoldInfo]


class ManifoldDeleteResponse(TypedDict):
    namespace: str
    name: str
    source: str
    removed: bool
    rematerializes_on_restart: bool


# ---------------------------------------------------------------- templates --

class TemplateTurnJSON(TypedDict):
    role: str
    content: str


class TemplateContextJSON(TypedDict):
    turns: list[TemplateTurnJSON]
    assistant: str


class TemplateSummary(TypedDict):
    """A template list row."""

    __pydantic_config__ = _Open  # type: ignore[misc]

    namespace: str
    name: str
    slot: str
    n_values: int
    n_contexts: int
    values: list[str]
    labels: list[str]
    description: str
    tags: list[str]


class TemplateDetail(TemplateSummary):
    """A template summary plus the full contexts."""

    contexts: list[TemplateContextJSON]


class TemplateListResponse(TypedDict):
    templates: list[TemplateSummary]


class TemplateDeleteResponse(TypedDict):
    namespace: str
    name: str
    removed: bool


class ChoiceScore(TypedDict):
    """One candidate's score within a context's restricted-choice set."""

    text: str
    label: str
    n_tokens: int
    sum_logprob: float
    mean_logprob: float
    prob_sum: float
    prob_mean: float


class ChoiceScores(TypedDict):
    steering: str | None
    choices: list[ChoiceScore]


class ScoreTemplateResponse(TypedDict):
    template: str
    namespace: str
    steering: str | None
    contexts: list[ChoiceScores]


# -------------------------------------------------------------- instruments --

class InstrumentsResponse(TypedDict):
    instruments: list[InstrumentFamilyBlock]


class InstrumentSourceJSON(TypedDict):
    """One usable artifact source (lens binding / SAE source row)."""

    __pydantic_config__ = _Open  # type: ignore[misc]

    source: str
    kind: NotRequired[str]
    name: NotRequired[str]
    active: NotRequired[bool]
    provider: NotRequired[str]
    repo_id: NotRequired[str]
    repo_revision: NotRequired[str]
    checkpoint: NotRequired[str]
    layer: NotRequired[int]
    features: NotRequired[int]


class SaeReleaseJSON(TypedDict):
    """One provider release candidate from the SAELens registry.

    Distinct from :class:`InstrumentSourceJSON`: a release is a *candidate*
    the dashboard can still fetch, keyed by ``release`` rather than by the
    ``source`` string a prepared row is addressed with.
    """

    release: str
    model: str | None
    layers: list[int]
    repo_id: str | None
    neuronpedia: bool
    source: Literal["local", "saelens"]


class SourcesResponse(TypedDict):
    sources: list[InstrumentSourceJSON]
    releases: NotRequired[list[SaeReleaseJSON]]


class SourceSwitchResponse(TypedDict):
    source: str
    live_layers: list[int]


class PreparationProgress(TypedDict):
    current: int | None
    total: int | None
    unit: str


class SaeRuntimeInfo(TypedDict):
    """Resident SAE identity once a fetch/train preparation lands."""

    __pydantic_config__ = _Open  # type: ignore[misc]

    release: NotRequired[str]
    revision: NotRequired[str | None]
    fingerprint: NotRequired[str | None]
    layer: NotRequired[int]
    width: NotRequired[int]
    sae_id: NotRequired[str | None]
    repo_id: NotRequired[str | None]
    neuronpedia_id: NotRequired[str | None]


class PreparationStatusJSON(TypedDict):
    """One shape over lens ``fetch``/``fit`` and sae ``fetch``/``train``."""

    state: str
    operation: str | None
    progress: PreparationProgress | None
    message: str | None
    error: str | None
    started_at: float | None
    finished_at: float | None
    cancellable: bool
    live_layers: NotRequired[list[int] | None]
    source: NotRequired[str | None]
    release: NotRequired[str | None]
    name: NotRequired[str | None]
    info: NotRequired[SaeRuntimeInfo | None]


class LensTokenValidationJSON(TypedDict):
    word: str
    token_id: int


class SaeFeatureValidationJSON(TypedDict):
    """``POST .../instruments/sae/features/validate`` — a read-only range +
    metadata check against the resident SAE."""

    id: int
    label: str | None
    layer: int
    max_act: float | None


class SaeFeatureMetaEntry(TypedDict):
    label: str | None
    max_act: float | None


class SaeFeatureMetaResponse(TypedDict):
    """``POST .../instruments/sae/features/metadata`` — the discovery backfill."""

    features: dict[str, SaeFeatureMetaEntry]
