// Shared types for the saklas webui.  Every panel/drawer/store imports
// from here so renames stay one-shot.  Mirrors the JSON shapes in
// saklas/server route models and the steering grammar in
// saklas/core/steering_expr.py.

// ---------------------------------------------------------- triggers --

/** Per-term trigger keyword from the steering-expression grammar.
 *
 * Wire/UI form mirrors ``_TRIGGER_PRESETS`` in steering_expr.py.  ``BOTH`` is
 * the default; the canonical render of each preset goes back through the
 * formatter (``before``/``after``/``both``/``thinking``/``response``).
 * ``prompt`` and ``generated`` are accepted as aliases on parse but
 * normalize to ``before`` and ``response`` respectively at format time. */
export type Trigger =
  | "BOTH"
  | "BEFORE" // == prompt
  | "AFTER" // after-thinking
  | "THINKING"
  | "RESPONSE" // == generated
  | "PROMPT" // alias of BEFORE
  | "GENERATED"; // alias of RESPONSE

/** Tensor variant suffix from the shared steering grammar. */
export type Variant =
  | "raw"
  | "sae"
  | `sae-${string}`
  | "role"
  | `role-${string}`
  | "from"
  | `from-${string}`;

// ----------------------------------------------------- session info --

export interface SamplingFields {
  temperature: number | null;
  top_p: number | null;
  top_k: number | null;
  max_tokens: number | null;
  system_prompt: string | null;
  thinking: boolean | null;
}

export interface SessionInfo {
  id: string;
  model_id: string;
  device: string;
  dtype: string;
  created: number;
  config: SamplingFields;
  profiles: string[];
  probes: string[];
  history_length: number;
  supports_thinking: boolean;
  /** True iff the user can actually turn thinking off — i.e. the chat
   *  template has an ``enable_thinking`` switch.  Forced-thinking
   *  families (gpt-oss / Mistral-3 Reasoning / Qwen3-Thinking) ship
   *  ``supports_thinking=true`` but ``thinking_is_optional=false`` so
   *  the UI can lock the toggle and explain why pressing it is a
   *  no-op. */
  thinking_is_optional: boolean;
  default_steering: string | null;
  /** True iff the loaded model has no chat template. */
  is_base_model: boolean;
  /** True iff a Jacobian lens artifact is fitted for the loaded model
   *  (a server-side path check, not a load).  Gates the token
   *  drilldown's j-lens tab. */
  jlens_fitted: boolean;
  /** Live workspace-readout state (``POST .../instruments/lens/live``): the
   *  resolved layer list while the live lens is enabled, ``null`` while off.
   *  Rehydrates the WORKSPACE panel toggle across page reloads. */
  live_lens_layers: number[] | null;
  /** Resident SAE runtime capability and identity. */
  sae_loaded: boolean;
  sae_info: {
    release: string;
    revision?: string | null;
    fingerprint?: string | null;
    layer: number;
    width: number;
    sae_id?: string | null;
    repo_id?: string | null;
    neuronpedia_id?: string | null;
  } | null;
  /** True while per-token SAE discovery readout is enabled. */
  live_sae: boolean;
  /** CAA live toggle state (``POST .../instruments/geometry/live``): whether
   *  per-token monitor scoring feeds live consumers.  Off ⇒ probes report only
   *  the end-of-gen aggregate (gates still force what they need). */
  live_probe_scores: boolean;
  /** True iff the loaded model family supports assistant-role
   *  substitution (Qwen / Gemma / Llama / GLM / gpt-oss yes; Mistral /
   *  talkie no). Drives whether the roles control is enabled. */
  role_substitution_supported: boolean;
  /** True iff the family supports *user*-role substitution. */
  user_role_supported: boolean;
  /** The family's *standard* assistant-role label (e.g. Gemma ``model``,
   *  ChatML ``assistant``), or ``null`` when the family can't
   *  substitute the assistant side.  Seeds the assistant-role box so it
   *  shows the live default; a box value equal to this is treated as "no
   *  override" on send. */
  default_assistant_role: string | null;
  /** The family's *standard* user-role label (``user`` everywhere today),
   *  or ``null`` when unsupported.  Seeds the user-role box. */
  default_user_role: string | null;
  /** True iff the session's validated scene grammar is active (the cast
   *  model's stitcher renders — arbitrary seat sequences, seat toggle,
   *  free commit seating). */
  scene_mode: boolean;
  /** True iff a committed thinking block can be rendered (scene mode +
   *  family think delimiters).  Gates the composer's thinking box. */
  thinking_input_supported: boolean;
  /** True when the family template strips history thinking — a committed
   *  thinking block lasts one turn.  Drives the composer warning. */
  strips_history_thinking: boolean;
}

// -------------------------------------------------- jacobian lens --

/** ``POST /sessions/{id}/instruments/lens/token/validate`` — a read-only
 *  check that a J-lens word resolves to exactly one vocabulary token. */
export interface LensTokenValidationJSON {
  word: string;
  token_id: number;
}

/** One vocabulary entry of a per-layer J-lens readout row. */
export interface LensReadoutTokenJSON {
  token: string;
  id: number;
  /** ``log softmax(W_U · norm(J_l h))`` at this token — exp() for the
   *  within-row probability. */
  logprob: number;
}

/** One layer row of the J-lens readout matrix. */
export interface LensReadoutLayerJSON {
  layer: number;
  tokens: LensReadoutTokenJSON[];
}

/** One token of the layer-aggregated J-lens readout: per-layer softmax
 *  → mean probability (``strength``, 0..1) + the probability-mass-
 *  weighted depth center of mass (``com``, 0 = first block, 1 = last) and
 *  its std (``spread``). */
export interface LensAggregateTokenJSON {
  token: string;
  strength: number;
  com: number;
  spread: number;
}

/** The J-lens readout at one decode step of a loom node (the forward that
 *  produced the clicked token) — the drilldown's render shape.  Built from
 *  the ``GET .../instruments/lens/token-readout`` measurements-replay
 *  envelope (``instruments.lens.readout`` + ``binding``). */
export interface LensTokenReadoutJSON {
  node_id: string;
  raw_index: number;
  /** The clicked token — for highlighting its appearances in the matrix. */
  token_id: number;
  token_text: string;
  /** The steering expression the replay ran under, or ``null`` for an
   *  unsteered read (no recipe steering, or ``steered=false``). */
  steering: string | null;
  /** Layer-aggregated view of the same logits across all requested layers,
   *  strength-descending.  Empty from a pre-aggregate server. */
  aggregate?: LensAggregateTokenJSON[];
  layers: LensReadoutLayerJSON[];
}

/** The instrument-preparation operation a POST launches / a GET/DELETE
 *  reports. */
export type PreparationOp = "fetch" | "fit" | "load" | "train";

/** Unified status of a background instrument preparation
 *  (``POST/GET/DELETE .../instruments/{family}/preparations``).  One shape
 *  over lens ``fetch``/``fit`` and sae ``load``/``train`` — ``state`` is the
 *  discriminator, ``progress.unit`` carries ``"prompts"`` vs ``"tokens"`` for
 *  the label, and the op-specific extras (``live_layers`` / ``release`` /
 *  ``name`` / ``info``) ride alongside.  ``state === "done"`` (finished, no
 *  error) means the preparation landed — refresh session info. */
export interface PreparationStatusJSON {
  state: "idle" | "running" | "done" | "error";
  operation: PreparationOp | null;
  progress: { current: number; total: number; unit: string } | null;
  message: string | null;
  error: string | null;
  started_at: number | null;
  finished_at: number | null;
  cancellable: boolean;
  /** fetch/fit: layers the post-preparation auto-enable turned live. */
  live_layers?: number[] | null;
  /** fetch: the artifact source being fetched. */
  source?: string | null;
  /** load: the resident release. */
  release?: string | null;
  /** train: the local SAE name. */
  name?: string | null;
  /** load/train: resident SAE identity once the preparation lands. */
  info?: SessionInfo["sae_info"];
}

/** One usable artifact source. The ``source`` string is deliberately the
 * same identifier accepted by the sibling source-switch action. */
export interface InstrumentSourceJSON {
  source: string;
  kind: "local" | "huggingface" | "saelens";
  name: string;
  active: boolean;
  path?: string;
  provider?: string;
  repo_id?: string;
  repo_revision?: string;
  checkpoint?: string;
  layer?: number;
  features?: number;
}

// ------------------------------------------------ sparse autoencoder --

export interface SaeFeatureJSON {
  id: number;
  activation: number;
  label?: string | null;
  /** Cached Neuronpedia ``maxActApprox`` — the strength unit.  Render
   *  ``activation / max_act`` as the normalized 0..1 strength; ``null``
   *  until the metadata backfill lands (then bars fall back to the
   *  panel-shared raw scale). */
  max_act?: number | null;
}

/** ``POST .../instruments/sae/features/metadata`` — the discovery backfill.
 *  Fetches and caches Neuronpedia metadata for up to 64 feature ids; ids
 *  without metadata after the fetch are absent from the response. */
export interface SaeFeatureMetaResponse {
  features: Record<string, { label: string | null; max_act: number | null }>;
}

export interface SaeTokenReadoutJSON {
  node_id: string;
  raw_index: number;
  token_id: number;
  token_text: string;
  steering: string | null;
  layer: number;
  features: SaeFeatureJSON[];
}

// --------------------------------- 5.x measurement envelope (token wire) --
//
// The single JSON-safe read-side record — the same object on the WS ``token``
// frame, the loom token row (``measurements`` key), and the token-readout
// replay endpoint (wrapped in ``{measurements}``).  It replaces the former
// ``captured`` record and the six top-level per-token aliases (``scores`` /
// ``per_layer_scores`` / ``probe_readings`` / ``lens_readout`` /
// ``lens_aggregate`` / ``sae_readout``); see saklas.core.measurements.

export type TokenReadoutProvenance = "captured" | "replayed";

/** What a family was measuring: source identity + recipe steering (+ resident
 *  layer for sae), so historical rows stay interpretable after a source
 *  switch. */
export interface MeasurementBindingJSON {
  source: string | null;
  steering: string | null;
  /** sae only — the resident hook layer. */
  layer?: number | null;
}

/** Geometry (Monitor subspace) family — attached-probe readings only. */
export interface GeometryInstrumentJSON {
  readings: Record<string, ProbeReadingJSON>;
}

/** The J-lens native discovery readout: per-layer top-k matrix +
 *  layer-aggregated chip list. */
export interface LensReadoutBlockJSON {
  layers: LensReadoutLayerJSON[];
  aggregate: LensAggregateTokenJSON[];
}

/** J-lens family — attached ``jlens/<word>`` probe ``readings`` plus the
 *  native ``readout`` discovery surface, with a ``binding``. */
export interface LensInstrumentJSON {
  binding: MeasurementBindingJSON;
  readings?: Record<string, ProbeReadingJSON>;
  readout?: LensReadoutBlockJSON;
}

/** The SAE native discovery readout: per-step top-k feature activations. */
export interface SaeReadoutBlockJSON {
  features: SaeFeatureJSON[];
}

/** SAE family — attached ``sae/<id>`` probe ``readings`` plus the native
 *  ``readout`` discovery surface, with a ``binding``. */
export interface SaeInstrumentJSON {
  binding: MeasurementBindingJSON;
  readings?: Record<string, ProbeReadingJSON>;
  readout?: SaeReadoutBlockJSON;
}

export interface MeasurementInstrumentsJSON {
  geometry?: GeometryInstrumentJSON;
  lens?: LensInstrumentJSON;
  sae?: SaeInstrumentJSON;
}

/** The one measurement envelope — versioned, scoped, per-family
 *  ``instruments`` plus the flat cross-family ``scores`` /
 *  ``per_layer_scores`` views (their consumers key probes across families by
 *  name — transcript tinting, the loom heatmap). */
export interface MeasurementsEnvelopeJSON {
  version: number;
  scope: "token" | "aggregate" | "replay";
  provenance: TokenReadoutProvenance;
  /** Flat cross-family axis-0 view (highlight tinting). */
  scores?: Record<string, number>;
  /** Optional per-layer × per-probe heatmap view. */
  per_layer_scores?: Record<string, Record<string, number>>;
  instruments: MeasurementInstrumentsJSON;
}

// ----------------------------------------------------- manifolds --

/** One axis of a Box manifold domain.  ``periodic`` axes wrap (the
 *  authoring coordinate is taken mod ``period``); open axes clamp to
 *  ``[lo, hi]``. */
export interface AxisSpec {
  name: string;
  periodic: boolean;
  /** Period of a periodic axis — server canonicalizes; for an open
   *  axis this is typically ``hi - lo``. */
  period: number;
  lo: number;
  hi: number;
}

/** Geometry of a steering manifold's authoring domain.  ``box`` carries
 *  per-axis specs (1D/2D/3D in the webui builder); ``sphere`` is S^dim
 *  with a chordal metric; ``custom`` is the JSON-authored escape hatch
 *  the webui shows read-only. */
export type ManifoldDomain =
  | { type: "box"; axes: AxisSpec[] }
  | { type: "sphere"; dim: number }
  | { type: "custom"; [key: string]: unknown };

/** One node of a manifold — a label, its authoring coordinates (one
 *  per intrinsic dimension), and the statement corpus pooled into its
 *  centroid. */
export interface ManifoldNodeSpec {
  label: string;
  coords: number[];
  statements: string[];
  /** Optional per-node assistant-role substitution.  When set, this
   *  node's centroid is pooled with the chat-template's assistant-role
   *  label replaced by this slug — the persona-manifold building block
   *  that lets one fitted manifold span multiple personas in
   *  role-baselined activation space.  Slug must match
   *  ``[a-z0-9._-]+``.  Omit / leave empty for the standard assistant
   *  baseline (the legacy default).  Family-unsupported (Mistral-3 /
   *  talkie) raises at fit time. */
  role?: string | null;
}

/** PCA discover-fit diagnostics block surfaced in the inspector.
 *  Wire-shape mirror of ``saklas.core.manifold.PcaDiagnostics``.  Tensor
 *  fields are flattened to plain number[]'s server-side; everything
 *  else is a primitive. */
export interface ManifoldPcaDiagnostics {
  per_component_variance: number[];
  cumulative_variance: number[];
  picked_k: number;
  threshold: number;
}

/** Spectral (Laplacian-eigenmaps) discover-fit diagnostics block.
 *  Wire-shape mirror of ``saklas.core.manifold.SpectralDiagnostics``. */
export interface ManifoldSpectralDiagnostics {
  eigenvalues: number[];
  picked_k: number;
  gap_index: number;
  gap_magnitude: number;
  bandwidth: number;
  k_nn: number;
  component_count: number;
}

/** Per-model fit record returned in the manifold detail shape.
 *
 *  ``fit_mode`` ``"authored"`` means the user supplied per-node coords;
 *  ``"pca"`` / ``"spectral"`` mean the coords were derived per-model at
 *  fit time and live in the safetensors payload, with ``diagnostics``
 *  giving the variance bars or spectrum the inspector renders. */
export interface ManifoldFitInfo {
  stem: string;
  method: string;
  feature_space: string;
  node_count: number;
  nodes_sha256: string;
  /** Discriminator: ``authored`` for hand-placed coords, ``pca`` /
   *  ``spectral`` for coords derived from per-node activations, ``baked``
   *  for a corpus-less precomputed direction. */
  fit_mode: "authored" | "pca" | "spectral" | "auto" | "baked";
  /** Discover-mode only.  ``max_dim``/``var_threshold`` for PCA;
   *  ``max_dim``/``k_nn``/``bandwidth``/``reference_layer`` for spectral. */
  hyperparams?: Record<string, number | string>;
  /** Discover-mode only.  Method-tagged via ``picked_k`` + presence of
   *  ``per_component_variance`` (PCA) vs ``eigenvalues`` (spectral). */
  diagnostics?: ManifoldPcaDiagnostics | ManifoldSpectralDiagnostics;
}

/** Manifold list/detail row.  The list route omits ``nodes``'
 *  statements and ``fitted``; the detail route includes both. */
export interface ManifoldInfo {
  namespace: string;
  name: string;
  description: string;
  domain: ManifoldDomain;
  domain_label: string;
  intrinsic_dim: number;
  min_nodes: number;
  node_count: number;
  node_labels: string[];
  node_coords: number[][];
  /** Per-node assistant-role substitution recorded on the manifold,
   *  aligned with ``node_labels``.  ``null`` for a given node means
   *  "pooled under the standard assistant baseline".  An all-``null`` array marks a non-role
   *  manifold; any non-``null`` entry marks a persona / role-paired
   *  manifold. */
  node_roles: (string | null)[];
  fitted_models: string[];
  /** True iff a tensor for the loaded session model is present. */
  fitted_for_session: boolean;
  /** True iff a fitted tensor's ``nodes_sha256`` no longer matches the
   *  current node geometry — the fit is stale and should be re-run. */
  stale: boolean;
  /** Discriminator: ``authored`` for hand-placed coords, ``pca`` /
   *  ``spectral`` for coords derived per-model from activations, ``auto``
   *  for a discover folder whose flat-vs-curved geometry is resolved
   *  per-model at fit time, ``baked`` for a corpus-less precomputed
   *  direction. */
  fit_mode: "authored" | "pca" | "spectral" | "auto" | "baked";
  /** The geometry an ``auto`` folder resolved to for the loaded model —
   *  ``"pca"`` (flat) / ``"spectral"`` (curved) once fitted, ``null`` when
   *  not yet fitted (geometry unknown → show in both rack drawers).  For a
   *  non-``auto`` folder this mirrors ``fit_mode``. */
  resolved_fit_mode: "pca" | "spectral" | "authored" | "baked" | null;
  /** True for ``pca`` / ``spectral`` (coords derived per-model), false
   *  for ``authored``. */
  is_discover: boolean;
  /** Category-valued tags off ``manifold.json`` (e.g. ``register`` /
   *  ``cultural``).  Drives the category grouping in the shared RackDrawer.
   *  Current list and detail routes always emit it. */
  tags: string[];
  /** Resting steering coefficient hint for a concept axis.  Read by
   *  ``recommendedAlpha`` (defaults to 0.5 when absent).  Not currently
   *  emitted by the list serializer — provenance for a future field. */
  recommended_alpha?: number;
  /** Discover-mode only: the knobs the fit (will) use.  Empty / absent
   *  on authored folders.  PCA accepts ``max_dim`` / ``var_threshold``;
   *  spectral accepts ``max_dim`` / ``k_nn`` / ``bandwidth``. */
  hyperparams: Record<string, number | string>;
  /** Detail-only: full node specs with statement corpora.  In discover
   *  mode each node's ``coords`` is either the derived per-model layout
   *  (when a fit exists) or ``null`` (pending fit). */
  nodes?: (ManifoldNodeSpec | { label: string; coords: number[] | null; statements: string[] })[];
  /** Detail-only: per-tensor fit records. */
  fitted?: ManifoldFitInfo[];
}

export interface ManifoldListResponse {
  manifolds: ManifoldInfo[];
}

/** One HF search-row carrying enough metadata to render a result row
 *  without an extra round-trip.  Mirrors the pack-side ``RemotePackInfo``
 *  but trades pack-specific fields (``recommended_alpha``,
 *  ``tensor_models``-via-pack-format) for the manifold-specific ones
 *  the picker needs (``domain_label``, ``node_count``, ``fit_mode``). */
export interface RemoteManifoldInfo {
  /** Concept slug on HF (``<name>`` half of ``<ns>/<name>``). */
  name: string;
  /** HF owner (``<ns>`` half of ``<ns>/<name>``). */
  namespace: string;
  description: string;
  tags: string[];
  node_count: number;
  /** Short ``type(Nd)`` label — ``box(2d)``, ``sphere(3d)``,
   *  ``discover-pca``, etc. */
  domain_label: string;
  /** ``"authored"``/``"pca"``/``"spectral"`` — the folder's fit-mode
   *  discriminator. */
  fit_mode: string;
  /** Safe-model-id stems with a fitted tensor in the HF repo.  Same
   *  shape ``RemotePackInfo.tensor_models`` carries. */
  tensor_models: string[];
}

/** Body for POST /saklas/v1/manifolds/install. */
export interface InstallManifoldRequest {
  /** HF coord (``owner/repo[@revision]``) or local folder path. */
  target: string;
  /** Override the install destination (``<ns>/<name>``).  Wire field is
   *  ``as_`` since ``as`` is a Python keyword — matches the route
   *  body model. */
  as_?: string;
  force?: boolean;
}

/** One source folder in a manifold merge — fully qualified ``ns/name``. */
export interface MergeManifoldSource {
  namespace: string;
  name: string;
}

/** Body for POST /saklas/v1/manifolds/merge.
 *
 *  Restricted to discover-mode (autofitted) sources by design — the
 *  server unions their node corpora into one heap and writes a fresh
 *  unfitted discover folder.  Run ``apiManifoldFitStream`` against the
 *  merged folder to derive coords from the combined heap.
 */
export interface MergeManifoldRequest {
  /** Destination namespace (defaults to ``"local"`` server-side). */
  namespace?: string;
  /** Destination manifold name. */
  name: string;
  description?: string;
  /** ≥ 2 discover-mode source folders. */
  sources: MergeManifoldSource[];
  /** Override the merged folder's fit_mode.  Required when sources
   *  disagree; defaults to the shared mode otherwise. */
  fit_mode?: "pca" | "spectral";
  hyperparams?: Record<string, unknown>;
  force?: boolean;
}

/** Body for POST /saklas/v1/manifolds. */
export interface CreateManifoldRequest {
  namespace?: string;
  name: string;
  description: string;
  domain: ManifoldDomain;
  nodes: ManifoldNodeSpec[];
}

/** Body for PATCH /saklas/v1/manifolds/{ns}/{name}. */
export interface UpdateManifoldRequest {
  description?: string;
  nodes?: ManifoldNodeSpec[];
}

/** One node of a discover-mode manifold — label + statements only.
 *  Coords are derived per-model at fit time, so the authoring shape
 *  carries no ``coords`` field. */
export interface DiscoverManifoldNodeSpec {
  label: string;
  statements: string[];
  /** Optional per-node assistant-role substitution; see
   *  :class:`ManifoldNodeSpec` for semantics. */
  role?: string | null;
}

/** Body for POST /saklas/v1/manifolds/discover.
 *
 *  The user supplies labeled statement corpora; the matching ``fit``
 *  call derives node coordinates per-model via PCA, spectral embedding,
 *  or ``auto`` (let ``select_topology`` pick the geometry per-model). */
export interface CreateDiscoverManifoldRequest {
  namespace?: string;
  name: string;
  description?: string;
  fit_mode: "pca" | "spectral" | "auto";
  nodes: DiscoverManifoldNodeSpec[];
  hyperparams?: Record<string, number | string>;
}

/** Body for POST /saklas/v1/manifolds/from-template. */
export interface CreateManifoldFromTemplateRequest {
  namespace?: string;
  name: string;
  description?: string;
  fit_mode: "pca" | "spectral" | "auto";
  template_ref: string;
  hyperparams?: Record<string, number | string>;
  force?: boolean;
}

// ---- standalone templated-completion artifact (/saklas/v1/templates) ----

/** One turn in a template context's multi-turn history. */
export interface TemplateTurn {
  role: "system" | "user" | "assistant";
  content: string;
}

/** A multi-turn context: history turns + the slotted final assistant turn.
 *  The slot appears exactly once in ``assistant`` and never in a history turn. */
export interface TemplateContextSpec {
  turns: TemplateTurn[];
  assistant: string;
}

/** Body for POST /saklas/v1/templates — author a standalone template. */
export interface CreateTemplateRequest {
  namespace?: string;
  name: string;
  slot: string;
  values: string[];
  contexts: TemplateContextSpec[];
  description?: string;
  tags?: string[];
  force?: boolean;
}

/** A template list-row / summary. */
export interface TemplateSummary {
  namespace: string;
  name: string;
  slot: string;
  n_values: number;
  n_contexts: number;
  values: string[];
  labels: string[];
  description: string;
  tags: string[];
}

/** A template detail (summary + the full contexts). */
export interface TemplateDetail extends TemplateSummary {
  contexts: TemplateContextSpec[];
}

/** One candidate's score within a context's distribution. */
export interface ChoiceScore {
  text: string;
  label: string;
  n_tokens: number;
  sum_logprob: number;
  mean_logprob: number;
  prob_sum: number;
  prob_mean: number;
}

/** One context's restricted-choice distribution. */
export interface ChoiceScores {
  steering: string | null;
  choices: ChoiceScore[];
}

/** Response from POST /saklas/v1/templates/{ns}/{name}/score. */
export interface ScoreTemplateResponse {
  template: string;
  namespace: string;
  steering: string | null;
  contexts: ChoiceScores[];
}

/** Body for POST /saklas/v1/manifolds/generate.
 *
 *  LLM-author a discover-mode manifold from a flat concept list: the
 *  server runs ``SaklasSession.generate_responses`` (A2 conversational
 *  extraction — each concept answers the shared baseline prompts in
 *  character, one corpus per node) and writes a fresh discover folder
 *  ready for ``POST .../fit``. */
export interface GenerateManifoldRequest {
  namespace?: string;
  name: string;
  description?: string;
  concepts: string[];
  /** Per-concept system-prompt framing: ``abstract`` → "someone {c}",
   *  ``concrete`` → "{article} {c}", ``custom`` → ``custom_system``.
   *  Default abstract. */
  kind?: "abstract" | "concrete" | "custom";
  /** Required for ``kind: "custom"``; ``{c}`` is replaced by each concept. */
  custom_system?: string;
  /** In-character responses generated per shared baseline prompt. */
  samples_per_prompt?: number;
  fit_mode?: "pca" | "spectral" | "auto";
  hyperparams?: Record<string, number | string>;
  force?: boolean;
  /** Persona-manifold opt-in: each ``concepts[i]`` slug doubles as
   *  that node's assistant-role substitution at fit time, producing a
   *  role-paired manifold.  Steering through it implies the nearest
   *  node's role at decode time (the manifold lives in
   *  role-baselined activation space). */
  role_per_node?: boolean;
}

/** Body for POST /saklas/v1/manifolds/{ns}/{name}/fit.
 *
 *  Authored folders consume ``sae`` plus layer/force controls; discover
 *  folders additionally accept ``fit_mode`` / ``hyperparams`` overrides
 *  that get persisted into the folder before the fit runs so the cache
 *  key reflects the actual inputs. */
export interface FitManifoldRequest {
  sae?: string | null;
  layers?: number[] | "workspace" | "all" | null;
  force?: boolean;
  fit_mode?: "pca" | "spectral" | "auto" | null;
  hyperparams?: Record<string, number | string> | null;
}

// ----------------------------------------------------- vectors --

export interface VectorInfo {
  name: string;
  layers: number[];
  metadata: Record<string, unknown>;
}

export interface ProfileListResponse {
  profiles: VectorInfo[];
}

export interface ExtractRequest {
  /** Concept represented by the positive node (or sole monopolar node). */
  concept: string;
  baseline?: string | null;
  sae?: string | null;
  /** Role-augmented extraction: replace the assistant-role label in
   * the chat template with this slug at extract time (e.g. "pirate").
   * The same substitution rides at steer time so the extract baseline
   * matches the steer baseline.  The canonical tensor records the uniform
   * role and is addressed via the matching ``:role-<slug>`` alias; the
   * reserved ``_role-*`` filename is not written. Slug must match
   * ``[a-z0-9._-]+``;
   * mutually exclusive with ``sae``. */
  role?: string | null;
  /** Destination namespace for the extracted 1/2-node manifold folder.
   *  ``null`` / unset lands under
   *  ``~/.saklas/manifolds/local/<canonical>/``; another value selects
   *  ``manifolds/<namespace>/<canonical>/``. */
  namespace?: string | null;
  /** Regenerate/re-author the manifold corpus and refit even when a valid
   *  fitted tensor exists. Default false keeps the exact cache hit. */
  force?: boolean;
}

export interface ExtractResponse {
  canonical: string;
  profile: VectorInfo;
  progress: string[];
}


// ----------------------------------------------------- probes --

/** One attached probe — any rank.  The unified read-side row the server's
 *  ``probe_routes._probe_info`` emits (the pre-4.0 split of vector probes vs
 *  manifold probes collapsed onto one ``/probes`` collection).  ``is_affine``
 *  is the flat-vs-curved discriminator the client classifies on: flat probes
 *  (a 2-node concept axis through the rank-8 personas fan) are the *subspace*
 *  family, curved fits the *manifold* family. */
export interface ProbeInfo {
  /** Registered probe name (defaults to the selector at attach time). */
  name: string;
  /** Underlying manifold display name (``ns/name`` or bare). */
  manifold: string;
  /** Per-token nearest-node list length. */
  top_n: number;
  /** Sorted ascending list of layer indices the probe reads from. */
  layers: number[];
  /** Authoring node labels.  Aligned with ``node_coords`` when fitted. */
  node_labels: string[];
  node_count: number;
  /** Manifold domain spec — same shape as ``ManifoldInfo.domain``.  ``{}``
   *  for an unfitted discover manifold (``intrinsic_dim = 0`` then). */
  domain: ManifoldDomain | Record<string, never>;
  intrinsic_dim: number;
  /** ``"raw"`` for plain activation space, ``"sae-<release>"`` for SAE. */
  feature_space: string;
  /** Flat (affine) ⇒ subspace family; curved ⇒ manifold family. */
  is_affine: boolean;
  /** Per-node authoring/display layout (K, n), aligned with ``node_labels``.
   *  Backs the mini-map node dots + per-token trajectory lookup.  ``null``
   *  on an unfitted discover manifold (no per-model layout yet). */
  node_coords?: number[][] | null;
  /** True for a pinned J-lens token probe (the READOUT channel — the one
   *  coordinate axis is ``strength`` in [0,1], the mean fitted-layer probability;
   *  per-layer traces are ``(p_l,)`` over all fitted layers; no subspace
   *  geometry behind it). */
  lens?: boolean;
  /** The lens probe's word (``jlens/<word>``). */
  word?: string;
  /** The lens probe's resolved single-token vocabulary id. */
  token_id?: number | null;
  /** True for a pinned resident SAE feature probe. */
  sae?: boolean;
  feature_id?: number | null;
  label?: string | null;
  /** SAE probes only — the strength unit.  Coords (and so sparklines /
   *  gate scalars) are ``activation / max_act`` when set, raw activation
   *  when null (no Neuronpedia metadata). */
  max_act?: number | null;
}

export interface ProbeListResponse {
  probes: ProbeInfo[];
}

/** Body for ``POST /saklas/v1/sessions/{id}/probes`` — attach any probe
 *  shape by selector (the same ``[ns/]name[:variant]`` the ``%`` steering
 *  term consumes). */
export interface ProbeRequest {
  selector: string;
  name?: string;
  top_n?: number;
}

// ------------------------------------------------- probe readings --

/** One probe's reading — the single wire shape for *both* the per-token
 *  stream and the end-of-gen aggregate (the aggregate is the reading pooled
 *  at the last-content token).  Mirrors
 *  ``saklas.core.results.ProbeReading.to_dict()``.  ``coords`` is the
 *  domain-frame position (signed pole-normalized axis-0 at rank-1);
 *  ``residual`` is ``0`` for a flat (subspace) fit and the normalized
 *  off-surface distance for a curved (manifold) fit. ``assignment`` is the
 *  soft node posterior and ``membership`` the learned-tube density. Per-layer
 *  maps are string-keyed by layer index. */
export interface ProbeReadingJSON {
  fraction: number;
  nearest: [string, number][];
  coords: number[];
  residual: number;
  fraction_per_layer: Record<string, number>;
  coords_per_layer: Record<string, number[]>;
  residual_per_layer: Record<string, number>;
  assignment?: [string, number][];
  membership?: number;
  /** Per-axis depth center of mass (+ std) of the per-layer coordinate
   *  trace — where in the layer stack the probe reads, in normalized
   *  depth (0 = first block, 1 = last).  Mass per layer is
   *  ``share_weight_L · |coord_L|``.  Aligned with ``coords``; empty when
   *  the reading carries no per-layer trace (lean per-token modes) or the
   *  server predates the field. */
  depth_com?: number[];
  depth_spread?: number[];
  /** Per-layer whitened subspace coords (the live point + trail for the
   *  probe-inspector geometry plot).  Keyed by layer-index string -> that
   *  layer's ``(R,)`` whitened coords, in the same frame as the geometry
   *  endpoint's ``node_white``.  Present only when the generate request set
   *  ``persist_subspace_coords`` (the inspector being open); absent otherwise. */
  subspace_coords_per_layer?: Record<string, number[]>;
}

// ----------------------------------------------------- probe geometry --

/** One fitted layer's geometry for the probe-inspector plot.  All coords
 *  are in the **whitened (Mahalanobis) frame** — distances are Mahalanobis
 *  distances and the cloud is de-rogued.  ``rank`` (subspace dimension)
 *  drives the plot branch: 1 -> line, 2 -> 2D scatter, 3+ -> 3D PCA scatter.
 *  ``intrinsic_dim`` drives the overlay: 1 -> curve, 2 -> surface, else none. */
export interface ProbeLayerGeometry {
  layer: number;
  rank: number;
  intrinsic_dim: number;
  is_affine: boolean;
  /** (K, R) node centroids in whitened coords, aligned with node_labels. */
  node_white: number[][];
  /** (R,) neutral anchor in whitened coords (origin for a flat fit). */
  neutral_white: number[];
  /** (R, 3) projection onto the top-3 PCs of the node cloud; null for rank<3. */
  pca_rotation: number[][] | null;
  /** Variance share of the top-3 PCs; null for rank<3. */
  explained_variance_pcs: number[] | null;
  /** Per-layer Mahalanobis share — the steering budget; also the read weight. */
  mahalanobis_share: number;
  /** Curved-fit manifold overlay sampled into the whitened frame, or null. */
  overlay: ProbeOverlay | null;
}

export interface ProbeOverlay {
  kind: "curve" | "surface";
  /** Sampled points (S, R) for a curve; (nu*nv, R) row-major for a surface. */
  points: number[][];
  /** [nu, nv] mesh dims; present for ``kind === "surface"`` only. */
  grid_shape?: [number, number];
}

export interface ProbeGeometryResponse {
  name: string;
  manifold: string;
  intrinsic_dim: number;
  is_affine: boolean;
  node_labels: string[];
  /** False when a flat-DLS fit kept a different rank per layer. */
  rank_uniform: boolean;
  /** Keyed by layer-index string. */
  layers: Record<string, ProbeLayerGeometry>;
}

export interface ProbeDefaultsResponse {
  defaults: string[];
}

// ----------------------------------------------------- correlation --

export interface CorrelationData {
  names: string[];
  matrix: Record<string, Record<string, number | null>>;
  layers_shared: Record<string, number>;
}

// --------------------------------------------------- pairwise compare --

/** Cross-layer cosine matrix between two named vectors / probes.  Each
 *  ``matrix[i][j]`` is the raw cosine between ``a``'s layer
 *  ``layers_a[i]`` and ``b``'s layer ``layers_b[j]``.  Near-zero norms
 *  and shape mismatches land as ``null`` so the client can render
 *  empty / dimmed cells. */
export interface PairwiseCompareResponse {
  a: string;
  b: string;
  layers_a: number[];
  layers_b: number[];
  matrix: (number | null)[][];
  model: string | null;
}

// ----------------------------------------------------- traits SSE --

export type TraitsEvent =
  | { type: "start"; generation_id: string }
  | {
      type: "token";
      idx: number;
      text: string;
      thinking: boolean;
      probes: Record<string, number>;
    }
  | {
      type: "done";
      generation_id: string | null;
      finish_reason: string;
      aggregate: Record<string, number>;
    };

// ----------------------------------------------------- WS protocol --

export interface WSSampling {
  temperature?: number | null;
  top_p?: number | null;
  top_k?: number | null;
  max_tokens?: number | null;
  seed?: number | null;
  stop?: string[] | null;
  logit_bias?: Record<string, number> | null;
  presence_penalty?: number;
  frequency_penalty?: number;
  /** Logit-pass: opt in to top-K alternatives + chosen-token logprob on
   *  the WS ``token`` event.  Server-side clamped to ``[0, 256]``.  Zero
   *  (or absent) means logprob-only — chosen-token logprob still flows
   *  when any on_token consumer is live, just no top alternatives.
   *  Default 0 keeps the wire shape unchanged for opt-out users. */
  return_top_k?: number | null;
  /** Skip final aggregate probe scoring when only gate control is needed. */
  return_probe_readings?: boolean | null;
  /** Native dashboard requests this so streamed token rows can rehydrate
   *  the token-drilldown layer heatmap after a refresh. */
  persist_per_layer_scores?: boolean | null;
  /** Native dashboard requests per-layer whitened subspace coords on each
   *  token's probe reading (the probe-inspector live point + fading trail).
   *  Set true only while that inspector is open; forces per-token scoring. */
  persist_subspace_coords?: boolean | null;
  /** Per-message role-substitution labels (roleplay scaffold).  Ride each
   *  generate / commit like ``seed``; stamped onto the produced loom nodes
   *  and rendered per-turn.  null/empty = standard role label. */
  user_role?: string | null;
  assistant_role?: string | null;
}

export interface WSGenerateRequest {
  type: "generate";
  input?: string | unknown;
  steering?: string | null;
  sampling?: WSSampling | null;
  thinking?: boolean | null;
  stateless?: boolean;
  raw?: boolean;
  /** Loom: attach result as a child of this node.  ``null``/absent =
   *  active node.  Lets phase-3 regen target a specific user-parent. */
  parent_node_id?: string | null;
  /** Loom: spawn ``n`` sibling assistant nodes (deterministic seed schedule
   *  per Decision 20).  Default 1 server-side. */
  n?: number;
  /** Loom: partial Recipe overlaid on the parent's — phase-5 fan-out /
   *  auto-regen.  Accepted as a mode string (``"unsteered"`` etc) or a
   *  partial-recipe expression string.  Engine resolves the overlay. */
  recipe_override?: string | Record<string, unknown> | null;
  /** Logit fork: regenerate an existing assistant node as a sibling with
   *  one token swapped.  When ``fork_node_id`` is set the server ignores
   *  ``input`` / ``steering`` / ``sampling`` / ``n`` and reuses the
   *  node's stamped recipe; the three fields must travel together. */
  fork_node_id?: string | null;
  fork_raw_index?: number | null;
  fork_alt_token_id?: number | null;
  /** Answer-prefill: seed an assistant reply under a user node.  When
   *  ``prefill_node_id`` is set the server ignores ``input`` and the
   *  ``fork_*`` fields, tokenizes ``prefill_text`` into a forced decode
   *  prefix, and lands the result as a sibling assistant under the user
   *  node (``thinking`` forced off — the text is the start of the
   *  answer).  ``steering`` / ``sampling`` / ``n`` ride through. */
  prefill_node_id?: string | null;
  prefill_text?: string | null;
  /** Commit (Ctrl+Enter on either surface): land a turn under
   *  ``parent_node_id`` without running a decode.  ``commit_role="user"``
   *  routes to ``session.append_user_turn`` (active node must not be a
   *  user node); ``commit_role="assistant"`` routes to
   *  ``session.append_assistant_turn`` (``parent_node_id`` must be the
   *  user node the authored turn hangs off).  Mutually exclusive with
   *  prefill and fork; ``input`` / ``steering`` / ``sampling`` /
   *  ``thinking`` / ``n`` are ignored.  Both fields must travel
   *  together. */
  commit_role?: "user" | "assistant" | null;
  commit_text?: string | null;
  /** Optional committed thinking block riding a commit (any seat) —
   *  stored on the node's ``thinking_text`` and rendered through the
   *  family think delimiters (400 when the family can't carry it). */
  commit_thinking?: string | null;
  /** Cast model: which seat the generated turn occupies.  ``"user"``
   *  renders the generation prompt as a user-seat header (labeled by
   *  ``sampling.user_role``) and lands the node with ``role="user"`` +
   *  a stamped recipe.  Absent/null = assistant.  Needs scene mode. */
  generate_seat?: "user" | "assistant" | null;
}

export type ChatRole = "user" | "assistant";

/** Native composer submission. The authored and generated structural roles
 * are independent; omit ``generated_role`` for an append-only action. */
export interface WSSubmitRequest {
  type: "submit";
  text?: string | null;
  authored_role?: ChatRole | null;
  generated_role?: ChatRole | null;
  steering?: string | null;
  sampling?: WSSampling | null;
  thinking?: boolean | null;
  authored_thinking?: string | null;
  raw?: boolean;
  parent_node_id?: string | null;
  n?: number;
  recipe_override?: string | null;
}

export interface WSStopRequest {
  type: "stop";
}

export type WSClientMessage = WSGenerateRequest | WSSubmitRequest | WSStopRequest;

export interface WSStartedEvent {
  type: "started";
  generation_id: string;
  node_id: string | null;
  sibling_index: number;
  sibling_count: number;
}

/** Logit-pass (v2.3): one alternative the model considered at this
 *  position.  Wire-shape mirror of ``saklas.core.results.TokenAlt``.
 *  ``logprob`` is the post-sampler natural-log probability under the
 *  post-temperature / post-top-p / post-top-k distribution sampling
 *  actually drew from. */
export interface TokenAltJSON {
  id: number;
  text: string;
  logprob: number;
}

export interface WSTokenEvent {
  type: "token";
  text: string;
  thinking: boolean;
  token_id: number | null;
  /** Logit-pass: chosen-token logprob under the post-sampler distribution.
   *  Populated whenever the engine's log_softmax ran (any ``on_token``
   *  consumer or an explicit ``logprobs``/``return_top_k`` request).
   *  Absent on current uncaptured events. */
  logprob?: number | null;
  /** Per-token perplexity under the sampled distribution.  The native WS
   *  explicitly opts into this channel so the workbench status and exported
   *  turn provenance are backed by the engine rather than reconstructed. */
  perplexity?: number | null;
  /** Logit-pass: top-K alternatives sorted by descending logprob.  Length
   *  matches ``SamplingConfig.return_top_k`` when populated, else absent.
   *  The chosen token may or may not appear in this list depending on
   *  K. */
  top_alts?: TokenAltJSON[] | null;
  /** Logit-pass: raw decode-step index — the join key a logit fork slices
   *  ``raw_token_ids`` on.  Rides the ``token`` event directly; absent on
   *  current uncaptured events. */
  raw_index?: number | null;
  /** Loom: node id this token belongs to.  Routes the token to the right
   * sibling render during n-way regen.  Optional. */
  node_id: string | null;
  /** The 5.x measurement envelope — the single read-side record carrying the
   *  per-family ``instruments`` (geometry / lens / sae ``readings`` +
   *  ``readout``) plus the flat ``scores`` / ``per_layer_scores`` views.  The
   *  identical object is persisted on the loom token row.  Omitted when
   *  nothing was measured, so clients read it defensively. */
  measurements?: MeasurementsEnvelopeJSON;
}

export interface WSDoneResult {
  text: string;
  tokens: number;
  finish_reason: string;
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
  /** Logit-pass: per-turn mean chosen-token logprob over the assistant
   *  response span (thinking tokens excluded by construction).  Null when
   *  logprob capture wasn't live (replay / no on_token consumer). */
  mean_logprob?: number | null;
  mean_surprise?: number | null;
  /** End-of-generation per-attached-probe aggregate — the same
   *  ``ProbeReadingJSON`` shape as the per-token stream (the aggregate is the
   *  reading pooled at the last-content token).  Keys are probe names.
   *  Omitted entirely when no probe is attached — read defensively. */
  probe_readings?: Record<string, ProbeReadingJSON>;
}

export interface WSDoneEvent {
  type: "done";
  result: WSDoneResult;
  /** Loom: node id this gen finalised. */
  node_id: string | null;
  sibling_index: number;
  sibling_count: number;
}

export interface WSErrorEvent {
  type: "error";
  message: string;
  code?: string;
  node_id?: string;
}

// ----------------------------------------------------- loom (v2.3) --

/** Wire-shape mirror of saklas.core.loom.LoomNode.  Optional fields are
 * absent on the wire when null/empty server-side to keep payloads slim. */
/** One token-row inside a node's ``tokens`` / ``thinking_tokens`` array.
 *  Server-side token rows have required identity/score fields plus
 *  feature-dependent optional capture channels; the
 *  fields below are the ones :meth:`session._token_tap` stamps and the
 *  ones the webui knows how to consume.  All optional because the engine
 *  legitimately omits some on certain paths (e.g. ``top_alts`` only when
 *  ``return_top_k > 0``; ``probes`` / ``per_layer_scores`` only when the
 *  monitor has probes loaded; ``raw_index`` is stamped at finalize and
 *  absent for transcript-imported nodes). */
export interface LoomTokenRowJSON {
  token_id: number;
  text: string;
  logprob: number | null;
  perplexity: number | null;
  top_alts?: { id: number; text: string; logprob: number }[];
  raw_index?: number | null;
  /** Per-token magnitude-weighted aggregate probe score
   *  (``score_single_token``), persisted at append time.  Drives the
   *  highlight tint when the user rehydrates a tree across page refresh. */
  probes?: Record<string, number>;
  /** Per-layer × per-probe heatmap (``score_single_token_per_layer``),
   *  keyed by stringified layer index.  Drives the token-drilldown
   *  drawer's heatmap on rehydrated turns. */
  per_layer_scores?: Record<string, Record<string, number>>;
  /** The 5.x measurement envelope captured by the original generation. This
   * survives tree rehydration and explicit loom save/load without replaying
   * the model. Replaces the pre-5.x ``captured`` record. */
  measurements?: MeasurementsEnvelopeJSON;
}

export interface LoomNodeJSON {
  id: string;
  parent_id: string | null;
  role: "user" | "assistant" | "system";
  text: string;
  /** Per-turn role-substitution label (roleplay scaffold) — the custom
   *  role this turn was *sent* with (e.g. "captain" / "pirate"), or null
   *  for the standard role.  Drives the bubble heading + loom glyph.
   *  Null means the standard role. */
  role_label: string | null;
  /** The turn's verbatim thinking block — committed by the author, or the
   *  decoded thinking channel of a generated node (stamped at finalize).
   *  Strip families re-render it for one turn only. Null means no block. */
  thinking_text: string | null;
  /** Generated nodes only, irrespective of structural role. Mirrors Recipe. */
  recipe: {
    steering: string | null;
    sampling: WSSampling | null;
    thinking: boolean | null;
    seed: number | null;
    probes: string[];
    probe_hashes: Record<string, string>;
  } | null;
  aggregate_readings: Record<string, number>;
  applied_steering: string | null;
  finish_reason: string | null;
  starred: boolean;
  notes: string;
  created_at: number;
  edited_at: number | null;
  edit_count: number;
  /** Logit-pass: mean chosen-token logprob over the response span when
   *  logprob capture was live; absent on current uncaptured nodes.  Drives
   *  the loom sidebar's surprise edge-weighting and the
   *  ``sort:surprise`` / ``sort:confidence`` filter grammar. */
  mean_logprob: number | null;
  mean_surprise: number | null;
  /** Per-token response-span rows captured during streaming.  Present
   *  when the server serializes the tree with ``include_tokens=True``
   *  (the webui tree GET path).  Absent on transcript-imported
   *  nodes that never streamed under the v2.4 token-row schema. */
  tokens: LoomTokenRowJSON[] | null;
  /** Per-token thinking-span rows.  Same shape as ``tokens``; populated
   *  only when the engine emitted thinking content for the node. */
  thinking_tokens: LoomTokenRowJSON[] | null;
  /** Raw decode-step ids the engine sampled, including suppressed
   *  delimiters and unmerged partial-UTF-8 bytes.  The forceable prefix
   *  a logit fork replays from; ``null`` on transcript-imported
   *  nodes, in which case the fork affordance falls back to disabled. */
  raw_token_ids: number[] | null;
}

/** Full tree dump returned by GET /sessions/{id}/tree.
 *
 *  Server's ``LoomTree.to_dict`` serializes ``nodes`` as a list (flat,
 *  preserves insertion order) and ``children_of`` as a parent→ordered
 *  child-id map.  Clients pivot the node list into a dict keyed by id
 *  for the in-memory cache. */
export interface LoomTreeJSON {
  tree_format: number;
  saklas_version: string;
  root_id: string;
  active_node_id: string;
  rev: number;
  nodes: LoomNodeJSON[];
  /** parent_id → ordered list of child ids. */
  children_of: Record<string, string[]>;
  /** Optional model identifier the tree was generated against. */
  model_id: string | null;
  session_id: string | null;
  name: string | null;
  /** Cast roster (phase 3): label → member.  Absent when empty. */
  cast: Record<string, CastMemberJSON>;
}

/** One cast-roster member — a named label plus its standing recipe
 *  fragment (the weakest steering tier at generation).  Mirrors
 *  ``saklas.core.loom.CastMember``. */
export interface CastMemberJSON {
  recipe?: {
    steering?: string | null;
    thinking?: boolean | null;
    seed?: number | null;
  } | null;
  notes?: string;
  origin?: "structural" | "observed" | "configured";
}

/** Phase-5 cross-branch diff response (server side: NodeDiff +
 *  per_token spans + steering-delta labels).  Returned by
 *  ``POST /sessions/{id}/tree/diff``. */
export interface DiffTextSpanJSON {
  state: "equal" | "insert" | "delete";
  text: string;
}

export interface DiffReadingDeltaJSON {
  name: string;
  delta: number;
  a_value: number;
  b_value: number;
}

export interface DiffTokenSpanJSON {
  a_index: number;
  b_index: number;
  a_text: string;
  b_text: string;
  aligned: boolean;
  reading_deltas: DiffReadingDeltaJSON[];
}

export interface NodeDiffJSON {
  a_id: string;
  b_id: string;
  parent_id: string | null;
  a_text: string;
  b_text: string;
  a_applied_steering: string | null;
  b_applied_steering: string | null;
  parent_applied_steering: string | null;
  steering_delta: string;
  parent_to_a_delta: string;
  parent_to_b_delta: string;
  text: DiffTextSpanJSON[];
  readings: DiffReadingDeltaJSON[];
  per_token: DiffTokenSpanJSON[];
}

/** Phase-5 filter route response. */
export interface FilterMatchesJSON {
  expr: string;
  matching_node_ids: string[];
}

/** Logit-pass Phase 5 — one aligned-position row in the joint-logprobs
 *  response.  Mirrors ``saklas.core.joint_logprobs.JointLogprobRow``.
 *
 *  ``lp_*_in_*`` are post-temperature, post-sampler natural-log
 *  probabilities (matches the engine's chosen-token logprob shape).
 *  Cross fields and ``approx_kl`` are populated only on byte-aligned
 *  rows — divergent positions leave them ``null`` because the cross
 *  probability is ambiguous on non-aligned positions. */
export interface JointLogprobRowJSON {
  a_index: number;
  b_index: number;
  a_text: string;
  b_text: string;
  aligned: boolean;
  lp_a_in_a: number | null;
  lp_b_in_b: number | null;
  lp_a_in_b: number | null;
  lp_b_in_a: number | null;
  rank_changed: boolean;
  approx_kl: number | null;
}

/** Logit-pass Phase 5 — joint-logprobs response.  ``rows`` covers the
 *  full byte-walk; ``n_rank1_changed`` is a summary stat of how many
 *  aligned rows flipped argmax across the two branches. */
export interface JointLogprobsJSON {
  a_id: string;
  b_id: string;
  parent_id: string | null;
  rows: JointLogprobRowJSON[];
  n_rank1_changed: number;
}

/** Phase-5 transcript-load route response. */
export interface TranscriptLoadResponseJSON {
  leaf_id: string;
  rev: number;
  guards: string[];
}

/** Per-op delta sent on every tree mutation.  Clients apply in-place
 * keyed by ``rev`` continuity; full re-fetch on gap.
 *
 * Note: phase-2 server sends ``updated`` as full LoomNodeJSON entries
 * (the plan's "partial fields" shape simplifies to "send the node again"
 * because LoomMutated doesn't track which fields changed).  Clients merge
 * by replacing the node entry wholesale. */
export interface WSTreeMutatedEvent {
  type: "tree_mutated";
  op:
    | "edit"
    | "branch"
    | "navigate"
    | "delete"
    | "star"
    | "note"
    | "reset"
    | "regenerate"
    | "begin_assistant"
    | "add_user"
    | "finalize"
    | "cast"
    | string;
  added?: LoomNodeJSON[];
  removed?: string[];
  updated?: LoomNodeJSON[];
  active_node_id?: string | null;
  rev: number;
  /** ``op="cast"`` only: the full roster inlined (label → member) so
   *  clients reconcile without a refetch. */
  cast?: Record<string, CastMemberJSON>;
}

/** Fired at the start of each branch in an n-way generate so the client
 * can allocate render slots before token events arrive. */
export type WSServerMessage =
  | WSStartedEvent
  | WSTokenEvent
  | WSDoneEvent
  | WSErrorEvent
  | WSTreeMutatedEvent;

// ----------------------------------------------------- chat / UI --

/** Per-token score row for chat highlighting.  ``perToken`` is the
 * canonical projected score from ``last_per_token_scores``; ``live`` is
 * the inline streamed value (overwritten on finalize). */
export interface TokenScore {
  text: string;
  thinking: boolean;
  /** Whichever score we know for the currently-selected highlight probe.
   * Filled at render time, not persisted. */
  score?: number;
  /** Full per-probe scores once available. */
  probes?: Record<string, number>;
  /** Full per-axis domain-frame coordinates per probe, captured live from the
   *  ``probe_readings`` wire channel.  Backs per-PC token highlighting (the
   *  ``personas[3]`` axis targets) — axis 0 already lives in ``probes``, so
   *  this is populated only for multi-axis (rank-R) probes.  It survives
   *  in-session navigation by reference in ``tokenScoreCache`` but is absent
   *  after a transcript / localStorage reload. */
  coordsByProbe?: Record<string, number[]>;
  /** Token-id from the WS event when available — useful for debugging. */
  tokenId?: number | null;
  /** Per-layer × per-probe heatmap data captured during streaming.
   * Drives the click-token drilldown drawer. */
  perLayerScores?: Record<string, Record<string, number>>;
  /** Logit-pass: chosen-token post-sampler logprob. Absent on imported
   *  turns and when no consumer requested log-softmax capture. Drives the inline ``surprise`` highlight
   *  mode and the token drilldown's logits tab. */
  logprob?: number | null;
  /** Logit-pass: top-K alternatives captured at this position (descending
   *  by logprob).  Absent when ``return_top_k == 0`` or replayed. */
  topAlts?: TokenAltJSON[] | null;
  /** Raw decode-step index of this token in the backing node's
   *  ``raw_token_ids`` — the join key a logit fork slices on.  Absent on
   *  transcript-imported nodes (engine pre-dates raw-id capture),
   *  in which case the token can't be forked. */
  rawIndex?: number | null;
  /** Loom-owned 5.x measurement envelope from the original decode step. */
  measurements?: MeasurementsEnvelopeJSON;
}

export interface ChatTurn {
  role: "user" | "assistant" | "system";
  text: string;
  /** Per-turn role-substitution label (roleplay scaffold) carried from the
   *  backing loom node — drives the bubble heading.  null/undefined =
   *  standard role label. */
  roleLabel?: string | null;
  /** Loom node backing this turn, when the server tree is active. */
  nodeId?: string | null;
  /** Whether a model run authored this turn. Used for generation artifacts
   *  and analysis only; never to name, style, or gate rerolling the role. */
  generated?: boolean;
  /** True iff any thinking content was emitted. */
  thinking?: boolean;
  /** Visible response tokens with score data. */
  tokens?: TokenScore[];
  /** Thinking-only tokens with score data (rendered inside the
   * <Collapsible> equivalent). */
  thinkingTokens?: TokenScore[];
  /** A/B-mode pair: the unsteered same-seat shadow turn. */
  abPair?: ChatTurn;
  /** Steering expression applied — round-trips through parseExpression. */
  appliedSteering?: string | null;
  /** Aggregate probe readings for the turn (mean per probe). */
  aggregateReadings?: Record<string, number>;
  /** Generation timing summary, populated at done. */
  finishReason?: string;
  tokensSoFar?: number;
  maxTokens?: number;
  tokPerSec?: number;
  elapsedSec?: number;
  perplexity?: number;
  /** Logit-pass: per-turn mean chosen-token logprob (response span only,
   *  thinking excluded).  Populated from the WS ``done`` event; absent for
   *  current uncaptured turns. */
  meanLogprob?: number | null;
}

// ----------------------------------------------------- steer rack --
//
// One unified steering term, addressed as a position on a fitted geometry —
// a steering vector is the K=2 flat case of a manifold, so there is no longer
// a separate "vector" shape.  ``mode`` is the geometry family, the
// discriminator the card branches on and the serializer reads:
//
//   ``subspace`` — a flat affine fit (a 2-node bipolar axis through the
//                  rank-8 ``personas`` fan).  Every subspace term shares one
//                  rack-level ``subspaceAlong`` master (the merged affine
//                  subspace has a single slide), so the card carries NO
//                  per-card along knob — only a position (snap-to-node /
//                  XYPad).  Serializes ``<subspaceAlong> name[:variant]%pos``.
//   ``manifold`` — a curved fit (e.g. ``emotions``).  Each curved term is its own
//                  injection, so it keeps a per-card ``along`` + ``onto``.
//                  Serializes ``<along[,onto]> name[:variant]%pos``.
//
// ``mode`` is set at add time (``RackDrawer`` picks the adder off the
// catalog's ``fit_mode``: pca/baked → subspace, spectral/authored →
// manifold) and at parse time (a curved ``%`` or an ``onto`` coeff → manifold;
// else subspace).  The pre-4.1 ``~``/``|`` projection and ``!`` ablation are
// no longer authorable in the rack (a ``%`` term can't carry them); a pasted
// expression using them parses with a one-time warning and the operator
// dropped.  ``:variant`` survives — it rides the atom (``name:sae%pos``).

/** Subspace (flat affine) steering term — a position on a flat fit.  The
 *  magnitude is the rack-level ``subspaceAlong`` master (shared across every
 *  subspace term — the merged affine subspace slides once), so this entry
 *  carries no per-card coefficient; relative weight between subspace terms is
 *  expressed by how far each position sits from neutral. */
export interface SubspaceSteerEntry {
  mode: "subspace";
  /** Authoring coordinates, one per intrinsic dimension.  Rank-1 (a 2-node
   *  concept) is a single signed coord on the bipolar axis. */
  coords: number[];
  /** Node-label form (``name%label``); ``null`` = free coords (drag).  A
   *  fresh 2-node concept defaults to its positive pole's label. */
  label: string | null;
  /** Tensor variant — rides the atom (``name:sae%pos``).  Not authorable via
   *  the card today (kept for round-trip of pasted/legacy expressions). */
  variant: Variant;
  trigger: Trigger;
  /** When false, the term is excluded from serialization (visual but
   * not active). */
  enabled: boolean;
}

/** Manifold (curved) steering term — a placement on a curved fit with its own
 *  per-card ``along`` + ``onto`` (each curved term is its own injection). */
export interface ManifoldSteerEntry {
  mode: "manifold";
  /** ``along`` blend fraction in [0, 1] — how far to slide the in-subspace
   *  foot toward the position.  Serializes as the first value of the ``%``
   *  coefficient slot. */
  blend: number;
  /** ``onto`` collapse fraction in [0, 1] — pulls the off-surface in-subspace
   *  residual onto the surface.  ``0`` = off.  Serializes as the second value
   *  of the coefficient slot (``along,onto``) only when > 0. */
  onto: number;
  /** Authoring coordinates, one per intrinsic dimension. */
  coords: number[];
  /** Node-label form (``name%label``); ``null`` = free coords (drag). */
  label: string | null;
  /** Tensor variant — rides the atom (``name:sae%pos``). */
  variant: Variant;
  trigger: Trigger;
  enabled: boolean;
}

/** J-lens token steering term — pushes along the lens direction
 *  ``W_U[v] @ J_l`` over all fitted layers (``α jlens/<word>``). The rack
 *  key is the full ``jlens/<word>`` atom.  Per-chip ``alpha`` (not the
 *  shared ``subspaceAlong`` master): lens atoms run hotter than concept
 *  vectors — α≈0.3 is the coherent sweet spot, α≥0.5 over-steers into
 *  repetition — so each token needs its own dial. */
export interface JLensSteerEntry {
  mode: "jlens";
  /** Push coefficient (the plain-atom α slot). */
  alpha: number;
  trigger: Trigger;
  enabled: boolean;
}

/** Resident SAE decoder-row steering term (``α sae/<id>``). */
export interface SaeSteerEntry {
  mode: "sae";
  alpha: number;
  trigger: Trigger;
  enabled: boolean;
}

/** A racked steering term — subspace (flat), manifold (curved), or a
 *  J-lens token atom. */
export type SteerEntry =
  | SubspaceSteerEntry
  | ManifoldSteerEntry
  | JLensSteerEntry
  | SaeSteerEntry;

// ----------------------------------------------------- extract pairs --

/** One contrastive statement pair for custom-statement vector
 *  extraction.  Mirrors the server's ``{positive, negative}`` shape. */
export interface StatementPair {
  positive: string;
  negative: string;
}

// ----------------------------------------------------- probe rack --

export type ProbeSortMode = "name" | "value" | "change";

export interface ProbeRackEntry {
  /** Server-side row — metadata, domain, node layout, and the ``is_affine``
   *  flat-vs-curved flag that selects the subspace vs manifold card. */
  info: ProbeInfo;
  /** Last N values of the primary scalar for the sparkline — ring-buffer-ish,
   *  capped client-side.  Primary scalar is the signed axis-0 ``coords[0]``
   *  for a subspace (flat) probe, the ``fraction`` for a manifold (curved). */
  sparkline: number[];
  current: number;
  previous: number;
  /** Most recent token's per-layer readings for *this* probe.  Layer-key
   * strings keep the wire shape; the card sorts numerically.  For a subspace
   * probe this is axis-0 ``coords_per_layer``; for a manifold, ``fraction_per_layer``. */
  perLayer: Record<string, number>;
  /** Latest full per-token reading (coords / fraction / nearest / residual +
   *  per-layer traces).  Null until the first ``token`` event lands. */
  reading: ProbeReadingJSON | null;
  /** End-of-gen aggregate the ``done`` event lands — the settled reading.
   *  Null between gens; set on ``done``, cleared on the next ``started``. */
  aggregate: ProbeReadingJSON | null;
  /** Scalar aggregate restored from the selected saved Loom node.  The tree
   * keeps this portable summary but not the full per-layer reading; cards use
   * it instead of presenting a false zero after reload/navigation. */
  savedAggregate: number | null;
  /** Most-recent per-token nearest list (ascending distance).  Drives the
   *  inline nearest readout + mini-map hover; empty until the first token. */
  nearest: [string, number][];
  /** Inferred per-token coord trajectory for 2-D box mini-map rendering —
   *  each token's ``nearest[0]`` looked up in ``info.node_coords``.  Empty
   *  for non-2-D / sphere / custom probes and unfitted-discover (no coords). */
  trajectory: number[][];
  /** Ring buffer (last ~64 tokens) of per-layer whitened subspace coords for
   *  the probe-inspector geometry plot's live point + fading trail.  Each entry
   *  is one token's ``subspace_coords_per_layer`` (layer-key -> (R,) coords), so
   *  the inspector can reproject for any scrubbed layer at render time.  Only
   *  populated while the inspector is open (the ``persist_subspace_coords``
   *  generate flag); cleared on each generation ``started``. */
  subspaceTrail: SubspaceTrailSample[];
}

/** One token's per-layer whitened subspace coords for the inspector trail. */
export interface SubspaceTrailSample {
  perLayer: Record<string, number[]>;
}

// ----------------------------------------------------- gen status --

export interface PerplexityAccumulator {
  /** Sum of ln(ppl) across scored steps — geometric mean assembled
   * lazily via ``geometricMeanPpl``. */
  logSum: number;
  count: number;
  mean: number | null;
}

export interface GenStatus {
  active: boolean;
  tokensSoFar: number;
  maxTokens: number;
  /** Wall-clock start (``performance.now()`` ms). */
  startedAt: number | null;
  tokPerSec: number;
  ppl: PerplexityAccumulator;
  finishReason: string | null;
}

// ----------------------------------------------------- pending actions --

/** Actions queued during in-flight generation.
 *
 * The queue drains one item per WS ``done`` event in arrival order —
 * each ``apply`` either kicks off another gen (``awaitsGen=true``,
 * the next drain waits for that gen's own ``done``) or completes
 * instantly (``awaitsGen=false``, the next drain fires immediately).
 *
 * ``text`` is the user-facing string for the chat-side pending
 * bubble and the ↑-pull-and-edit re-issue path; ``rebuild`` is a
 * factory the input recall path calls to re-encode a pulled-and-
 * edited item with the same kind/role/target unchanged.  Both are
 * ``null`` for non-editable items (instant mutations like
 * ``clearChat`` / ``regen``) — those render as ghosted action chips
 * and can't be pulled, only cancelled with the ``×``.
 */
export interface PendingAction {
  id: string;
  label: string;
  text: string | null;
  apply: () => void | Promise<void>;
  awaitsGen: boolean;
  rebuild: ((newText: string) => PendingAction) | null;
  createdAt: number;
  /** Predicted active-node role after this action drains.  Drives the
   *  input box's role-aware placeholder + send-button label: a queued
   *  ``commit_user`` (this field = ``true``) flips the next message into
   *  prefill / commit-assistant mode even though the live active node
   *  hasn't moved yet.  ``null`` for actions that don't change the
   *  active node (rack mutations, sampling tweaks).  ``false`` for
   *  actions that land an assistant or root active node (send, prefill,
   *  commit_assistant, regen, /clear). */
  endsOnUserNode?: boolean | null;
  /** Coalesce tag for fold-into-tail batching.  When a fresh action
   *  carries the same ``coalesceKey`` as the *current queue tail*, its
   *  ``apply`` is chained onto that tail item instead of appending a
   *  new slot — so a slider drag (dozens of intermediate steering
   *  values) collapses to a single queued bubble carrying the net
   *  effect.  Only set on instant rack/steering mutations; ``undefined``
   *  for sends, commits, and one-shot mutations, which never coalesce. */
  coalesceKey?: string;
}

// ----------------------------------------------------- drawers --

export type DrawerName =
  /** Shared rack browser, subspace (flat) family — every flat affine
   *  fit (``fit_mode`` pca / baked): 2-node concept axes plus higher-rank
   *  flats like ``personas``.  White ``--accent``.  Split Fitted /
   *  Unfitted, per-row steer / probe / re-fit / delete, with a
   *  "+ build manifold" launcher (flat authoring folds into the manifold
   *  builder's pca path).  ``RackDrawer`` with ``family: "subspace"``.
   *  Opened from both rack "+ add" buttons. */
  | "subspace"
  /** Shared rack browser, manifold (curved) family — curved fits only
   *  (``fit_mode`` spectral / authored), e.g. ``emotions``.  Purple
   *  ``--pillar-manifold``.  Same layout as the subspace half, with a
   *  "+ build manifold" launcher.  ``RackDrawer`` with
   *  ``family: "manifold"``. */
  | "manifolds"
  /** Manifold authoring form — domain step + node editor.  Reached
   *  from the "+ build manifold" button inside ``manifolds``. */
  | "manifold_builder"
  /** Discover-mode node-union merge.  Unions the node corpora of two or
   *  more discover-mode manifolds into a fresh discover folder; restricted
   *  to discover sources by design. Reached from the command palette. */
  | "manifold_merge"
  /** Local manifold catalog plus HF search/install for
   *  ``saklas-manifold``-tagged repositories. */
  | "manifold_pack"
  | "save_conversation"
  | "load_conversation"
  | "compare"
  | "system_prompt"
  | "token_drilldown"
  | "correlation"
  /** Per-probe inspector — subsumes the layer-norms view for probes and
   *  adds a rank-aware whitened geometry plot (line / 2D scatter / 3D PCA
   *  scatter) with a layer scrubber and a fading live trajectory trail.
   *  Opened from a probe card's ⓘ button.  ``params: { name }``. */
  | "probe_inspector"
  | "advanced_sampling"
  | "health"
  | "session_admin"
  | "export"
  | "help"
  /** Cross-branch diff drawer — phase 5.  ``params`` carries the
   * selected node ids (1 user node → compare its children, 2+
   * assistant nodes → compare those). */
  | "node_compare"
  /** Transcript export/import drawer — phase 5. */
  | "transcript"
  /** Templated-completion lab — author standalone templates (slot + values
   *  + multi-turn contexts) and score the restricted-choice value
   *  distribution (steering-aware before/after). Reached from the workspace
   *  rail's "manifolds → templates…" entry. */
  | "template_lab"
  /** Cast manager (phase 3) — the tree's roster of named labels with
   *  standing steering recipes.  Reached from the composer cast row's
   *  "cast…" launcher.  A steering surface, not a chat feature. */
  | "cast";

export interface DrawerState {
  open: DrawerName | null;
  /** Per-drawer params — typed loosely because each drawer owns its own
   * shape (e.g. token drilldown carries the click-target token row). */
  params: unknown;
}
