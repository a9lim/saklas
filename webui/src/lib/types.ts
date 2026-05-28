// Shared types for the saklas webui.  Every panel/drawer/store imports
// from here so renames stay one-shot.  Mirrors the JSON shapes in
// saklas/server/saklas_api.py and the steering grammar in
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
  | "pca"
  | "sae"
  | `sae-${string}`
  | "role"
  | `role-${string}`;

// ----------------------------------------------------- session info --

export interface SamplingFields {
  temperature: number | null;
  top_p: number | null;
  top_k: number | null;
  max_tokens: number | null;
  system_prompt: string | null;
}

export interface SessionInfo {
  id: string;
  model_id: string;
  device: string;
  dtype: string;
  created: number;
  config: SamplingFields;
  vectors: string[];
  probes: string[];
  history_length: number;
  supports_thinking: boolean;
  /** True iff the user can actually turn thinking off — i.e. the chat
   *  template has an ``enable_thinking`` switch.  Forced-thinking
   *  families (gpt-oss / Mistral-3 Reasoning / Qwen3-Thinking) ship
   *  ``supports_thinking=true`` but ``thinking_is_optional=false`` so
   *  the UI can lock the toggle and explain why pressing it is a
   *  no-op.  Older servers may omit this field; clients should treat
   *  ``undefined`` as ``true`` (the historical default — most thinking
   *  models were toggleable). */
  thinking_is_optional?: boolean;
  default_steering: string | null;
  /** Non-canonical: optional architecture string surfaced for the
   * yellow-banner warning when ``model_type`` isn't in
   * ``_TESTED_ARCHS``.  Server may or may not populate this; clients
   * should tolerate ``undefined``. */
  architecture?: string;
  /** True iff the loaded model has no chat template — generation runs
   *  as flat completion (no roles, no bubbles).  Older servers omit
   *  this; clients treat ``undefined`` as ``false`` (chat model). */
  is_base_model?: boolean;
  /** True iff the loaded model family supports assistant-role
   *  substitution (Qwen / Gemma / Llama / GLM / gpt-oss yes; Mistral /
   *  talkie no). Drives whether the roles control is enabled. Older
   *  servers omit this; treat ``undefined`` as ``false``. */
  role_substitution_supported?: boolean;
  /** True iff the family supports *user*-role substitution. Same family
   *  set as the assistant side today. Treat ``undefined`` as ``false``. */
  user_role_supported?: boolean;
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
  nodes_sha256?: string;
  /** Discriminator: ``authored`` for hand-placed coords, ``pca`` /
   *  ``spectral`` for coords derived from per-node activations.  Older
   *  servers may omit this; treat ``undefined`` as ``"authored"``. */
  fit_mode?: "authored" | "pca" | "spectral";
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
   *  "pooled under the standard assistant baseline" (the legacy
   *  default).  An all-``null`` array (or absent) marks a non-role
   *  manifold; any non-``null`` entry marks a persona / role-paired
   *  manifold. */
  node_roles?: (string | null)[];
  fitted_models: string[];
  /** True iff a tensor for the loaded session model is present. */
  fitted_for_session: boolean;
  /** True iff a fitted tensor's ``nodes_sha256`` no longer matches the
   *  current node geometry — the fit is stale and should be re-run. */
  stale: boolean;
  /** Discriminator: ``authored`` for hand-placed coords, ``pca`` /
   *  ``spectral`` for coords derived per-model from activations.  Older
   *  servers may omit this; treat ``undefined`` as ``"authored"``. */
  fit_mode?: "authored" | "pca" | "spectral";
  /** Discover-mode only: the knobs the fit (will) use.  Empty / absent
   *  on authored folders.  PCA accepts ``max_dim`` / ``var_threshold``;
   *  spectral accepts ``max_dim`` / ``k_nn`` / ``bandwidth``. */
  hyperparams?: Record<string, number | string>;
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
 *  call derives node coordinates per-model via PCA or spectral
 *  embedding. */
export interface CreateDiscoverManifoldRequest {
  namespace?: string;
  name: string;
  description?: string;
  fit_mode: "pca" | "spectral";
  nodes: DiscoverManifoldNodeSpec[];
  hyperparams?: Record<string, number | string>;
}

/** Body for POST /saklas/v1/manifolds/generate.
 *
 *  LLM-author a discover-mode manifold from a flat concept list: the
 *  server runs ``SaklasSession.generate_concept_statements`` (shared
 *  scenarios + per-concept statements) and writes a fresh discover
 *  folder ready for ``POST .../fit``. */
export interface GenerateManifoldRequest {
  namespace?: string;
  name: string;
  description?: string;
  concepts: string[];
  n_scenarios?: number;
  statements_per_concept?: number;
  fit_mode?: "pca" | "spectral";
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
 *  Authored folders only consume ``sae`` / ``sae_revision``; discover
 *  folders additionally accept ``fit_mode`` / ``hyperparams`` overrides
 *  that get persisted into the folder before the fit runs so the cache
 *  key reflects the actual inputs. */
export interface FitManifoldRequest {
  sae?: string | null;
  sae_revision?: string | null;
  fit_mode?: "pca" | "spectral" | null;
  hyperparams?: Record<string, number | string> | null;
}

// ----------------------------------------------------- vectors --

export interface VectorTopLayer {
  layer: number;
  magnitude: number;
}

export interface VectorInfo {
  name: string;
  layers: number[];
  top_layers: VectorTopLayer[];
  per_layer_norms: Record<string, number>;
  metadata: Record<string, unknown>;
}

export interface VectorListResponse {
  vectors: VectorInfo[];
}

export interface ExtractRequest {
  name: string;
  /** Either a string (concept name like "happy.sad"), a {pos, neg} pair,
   * or a {pairs: [{positive, negative}, ...]} bundle. */
  source?: unknown;
  baseline?: string | null;
  method?: "dim" | "pca" | null;
  dls?: boolean | null;
  sae?: string | null;
  sae_revision?: string | null;
  /** Role-augmented extraction: replace the assistant-role label in
   * the chat template with this slug at extract time (e.g. "pirate").
   * The same substitution rides at steer time so the extract baseline
   * matches the steer baseline.  The tensor lands under a
   * ``_role-<slug>`` filename suffix and is steerable via the matching
   * ``:role-<slug>`` variant.  Slug must match ``[a-z0-9._-]+``;
   * mutually exclusive with ``sae``. */
  role?: string | null;
  /** Destination namespace for the extracted vector folder.  ``null`` /
   *  unset lands the vector under ``~/.saklas/vectors/local/<canonical>/``
   *  — the historical behavior.  Any other value relocates the folder
   *  to ``~/.saklas/vectors/<namespace>/<canonical>/``.  Parity with
   *  the manifold builder's namespace control. */
  namespace?: string | null;
  /** Force a fresh extraction even if a cached tensor / statements
   *  file exists at the destination.  Wires to the engine's
   *  ``force_statements`` flag.  Default ``false`` keeps the cache-hit
   *  short-circuit.  Parity with the manifold builder's overwrite
   *  control. */
  force?: boolean;
  register?: boolean;
}

export interface ExtractResponse {
  canonical: string;
  profile: VectorInfo;
  progress: string[];
}

/** Body for POST /sessions/{id}/vectors/merge — registered output is a
 * derived profile keyed by ``name``. */
export interface MergeVectorRequest {
  name: string;
  expression: string;
}

export type MergeVectorResponse = VectorInfo;

/** Body for POST /sessions/{id}/vectors/clone — wraps the clone CLI. */
export interface CloneVectorRequest {
  name: string;
  corpus_path: string;
  n_pairs?: number;
  seed?: number;
  baseline?: string | null;
}

export interface CloneVectorResponse {
  canonical: string;
  profile: VectorInfo;
  progress: string[];
}

/** Output of GET /sessions/{id}/vectors/{name}/diagnostics — per-layer
 * ``||baked||`` magnitudes + bucket histogram + (optional) probe-quality
 * diagnostics from ``saklas vector why``.  Resolves either steering
 * vectors or active probes — the server falls back to monitor profiles
 * on miss. */
export interface VectorDiagnosticsResponse {
  name: string;
  model: string;
  total_layers: number;
  /** Bucket histogram for the WHY view.  ``buckets`` is the bucket count
   * (HIST_BUCKETS, 16 by default); ``data`` is the per-bucket entries. */
  histogram: {
    buckets: number;
    data: { lo: number; hi: number; mean_norm: number }[];
  };
  /** Full per-layer ``||baked||`` magnitudes — one entry per retained
   * model layer, sorted ascending.  Drives the layer-norms overlay. */
  layers: { layer: number; magnitude: number }[];
  /** Probe-quality diagnostics when the profile carries them (v1.6+). */
  diagnostics_by_layer?: Record<string, Record<string, number>>;
  diagnostics_summary?: {
    evr: number | null;
    intra_pair_variance_mean: number | null;
    inter_pair_alignment: number | null;
    diff_principal_projection: number | null;
    stoplight: "solid" | "shaky" | "poor" | "unknown";
  };
}

// ----------------------------------------------------- probes --

export interface ProbeInfo {
  name: string;
  active: boolean;
  layers: number[];
}

export interface ProbeListResponse {
  probes: ProbeInfo[];
}

// ------------------------------------------------- manifold probes --

/** Row returned by the manifold-probes list / create routes.  Carries
 *  enough manifold geometry for the client to render the node layout
 *  (labels + intrinsic dim + domain spec) without re-fetching the source
 *  artifact.  Mirrors ``saklas.core.monitor.ManifoldProbeInfo``. */
export interface ManifoldProbeInfo {
  /** Registered probe name (defaults to the selector when ``name`` was
   *  omitted at attach time). */
  name: string;
  /** Underlying manifold display name (``ns/name`` or bare). */
  manifold: string;
  /** Per-token nearest-node list length. */
  top_n: number;
  /** Sorted ascending list of layer indices the probe reads from. */
  layers: number[];
  /** Authoring node labels.  Aligned with the manifold's ``node_coords``
   *  when the geometry is authored / fitted. */
  node_labels: string[];
  node_count: number;
  /** Manifold domain spec — same shape as ``ManifoldInfo.domain``.  May
   *  be ``{}`` for an unfitted discover manifold (the server reports
   *  ``intrinsic_dim = 0`` then). */
  domain: ManifoldDomain | Record<string, never>;
  intrinsic_dim: number;
  /** ``"raw"`` for plain activation space, ``"sae-<release>"`` for an
   *  SAE-reconstructed manifold. */
  feature_space: string;
  /** Per-node authoring coordinates the server attached to the probe row.
   *  Aligned with ``node_labels``; populated when the manifold carries a
   *  per-model layout (authored or fitted-discover).  Empty / absent on
   *  unfitted discover probes. */
  node_coords?: number[][];
}

export interface ManifoldProbeListResponse {
  probes: ManifoldProbeInfo[];
}

export interface AttachManifoldProbeRequest {
  selector: string;
  name?: string;
  top_n?: number;
}

/** Per-token streaming reading for one manifold probe.  Mirrors
 *  ``ManifoldTokenReading.to_dict()``.  ``fraction`` is the
 *  EV-weighted subspace-fraction across layers; ``nearest`` is the
 *  top-N nearest-node list as ``(label, distance)`` pairs sorted
 *  ascending. */
export interface ManifoldTokenReadingJSON {
  fraction: number;
  nearest: [string, number][];
}

/** End-of-generation aggregate reading for one manifold probe.  Mirrors
 *  ``ManifoldAggregate.to_dict()``.  ``coords`` is the EV-weighted mean
 *  authoring-space position recovered by ``invert_parameterization``;
 *  ``coords_per_layer`` exposes the same per-layer trace. */
export interface ManifoldAggregateJSON {
  fraction_mean: number;
  fraction_per_layer: Record<string, number>;
  nearest: [string, number][];
  coords: number[];
  coords_per_layer: Record<string, number[]>;
  residual_mean: number;
  residual_per_layer: Record<string, number>;
}

export interface ProbeDefaultsResponse {
  defaults: string[];
}

export interface ScoreProbeRequest {
  text: string;
  probes?: string[] | null;
}

export interface ScoreProbeResponse {
  readings: Record<string, number>;
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

// ----------------------------------------------------- packs --

export interface LocalPackInfo {
  name: string;
  namespace: string;
  source: "bundled" | "local" | string;
  description?: string;
  tags?: string[];
  layers?: number[];
  has_tensor?: boolean;
  has_sae?: boolean;
  variants?: string[];
  /** Loose passthrough for fields the server adds later. */
  [key: string]: unknown;
}

export interface PackListResponse {
  packs: LocalPackInfo[];
}

export interface RemotePackInfo {
  repo_id: string;
  description?: string;
  downloads?: number;
  likes?: number;
  tags?: string[];
  last_modified?: string;
  /** Loose passthrough — HF rows have many optional fields. */
  [key: string]: unknown;
}

export interface PackSearchResponse {
  query: string;
  results: RemotePackInfo[];
}

export interface InstallPackRequest {
  /** HF coord (``owner/repo``) or local folder path. */
  target: string;
  /** Override the install namespace (``-a NS/N`` in the CLI).  Wire field
   * is ``as`` (Python keyword in code, plain key in JSON). */
  as?: string;
  force?: boolean;
  /** Statements-only install (skip per-model tensor pull). */
  statements_only?: boolean;
}

export interface InstallPackResponse {
  target: string;
  installed_at: string;
  statements_only: boolean;
}

export interface DeletePackResponse {
  namespace: string;
  name: string;
  /** ``"bundled"`` / ``"local"`` / ``"hf://..."``.  Drives the
   *  toast wording — bundled concepts re-materialize on restart. */
  source: string;
  removed: number;
  /** Bundled concepts respawn on next session init. */
  rematerializes_on_restart: boolean;
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
}

export interface WSStopRequest {
  type: "stop";
}

export type WSClientMessage = WSGenerateRequest | WSStopRequest;

export interface WSStartedEvent {
  type: "started";
  generation_id: string;
  /** Loom: node id receiving this gen's tokens.  Optional for backward
   * compat with the pre-phase-2 single-path server. */
  node_id?: string | null;
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
  /** Magnitude-weighted aggregate probe score per probe, populated only
   * when probes are loaded.  Same value the TUI tints live tokens with;
   * the webui's inline highlight reads this so live highlighting matches
   * the post-generation pass. */
  scores?: Record<string, number>;
  /** Per-layer × per-probe map populated only when probes are loaded.
   * Keys: layer-index strings → probe names → cosine-sim score.  Drives
   * the token drilldown heatmap, not the inline tint. */
  per_layer_scores?: Record<string, Record<string, number>>;
  /** Logit-pass: chosen-token logprob under the post-sampler distribution.
   *  Populated whenever the engine's log_softmax ran (any ``on_token``
   *  consumer or an explicit ``logprobs``/``return_top_k`` request).
   *  Absent on legacy / replayed events. */
  logprob?: number | null;
  /** Logit-pass: top-K alternatives sorted by descending logprob.  Length
   *  matches ``SamplingConfig.return_top_k`` when populated, else absent.
   *  The chosen token may or may not appear in this list depending on
   *  K. */
  top_alts?: TokenAltJSON[] | null;
  /** Logit-pass: raw decode-step index — the join key a logit fork slices
   *  ``raw_token_ids`` on.  Rides the ``token`` event directly; absent on
   *  legacy / replayed events. */
  raw_index?: number | null;
  /** Loom: node id this token belongs to.  Routes the token to the right
   * sibling render during n-way regen.  Optional. */
  node_id?: string | null;
  /** Per-attached-probe manifold reading for this token.  Keys are probe
   *  names; values carry ``fraction`` (EV-weighted subspace fraction) and
   *  ``nearest`` (top-N node-label distances).  Field is omitted entirely
   *  when no manifold probe is attached, so legacy clients see no change. */
  manifold_readings?: Record<string, ManifoldTokenReadingJSON>;
}

export interface WSDoneResultPerToken {
  token_idx: number;
  probes: Record<string, number>;
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
  per_token_probes: WSDoneResultPerToken[];
  /** Logit-pass: per-turn mean chosen-token logprob over the assistant
   *  response span (thinking tokens excluded by construction).  Null when
   *  logprob capture wasn't live (replay / no on_token consumer). */
  mean_logprob?: number | null;
  /** End-of-generation per-attached-probe aggregate.  Keys are probe
   *  names; values carry ``fraction_mean``, ``coords``, ``nearest`` and
   *  the per-layer traces.  Field is omitted entirely when no manifold
   *  probe is attached — read defensively. */
  manifold_readings?: Record<string, ManifoldAggregateJSON>;
}

export interface WSDoneEvent {
  type: "done";
  result: WSDoneResult;
  /** Loom: node id this gen finalised. */
  node_id?: string | null;
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
 *  Server-side ``TokenScoreDict`` is loose (``dict[str, Any]``); the
 *  fields below are the ones :meth:`session._token_tap` stamps and the
 *  ones the webui knows how to consume.  All optional because the engine
 *  legitimately omits some on certain paths (e.g. ``top_alts`` only when
 *  ``return_top_k > 0``; ``probes`` / ``per_layer_scores`` only when the
 *  monitor has probes loaded; ``raw_index`` is stamped at finalize and
 *  absent for legacy / transcript-loaded nodes). */
export interface LoomTokenRowJSON {
  token_id?: number;
  text?: string;
  logprob?: number | null;
  perplexity?: number | null;
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
}

export interface LoomNodeJSON {
  id: string;
  parent_id: string | null;
  role: "user" | "assistant" | "system";
  text: string;
  /** Per-turn role-substitution label (roleplay scaffold) — the custom
   *  role this turn was *sent* with (e.g. "captain" / "pirate"), or null
   *  for the standard role.  Drives the bubble heading + loom glyph.
   *  Older servers omit this; treat undefined as null. */
  role_label?: string | null;
  /** Assistant nodes only.  Mirrors saklas.core.loom.Recipe. */
  recipe?: {
    steering?: string | null;
    sampling?: WSSampling | null;
    thinking?: boolean | null;
    seed?: number | null;
    probes?: string[];
    probe_hashes?: Record<string, string>;
  } | null;
  aggregate_readings?: Record<string, number>;
  applied_steering?: string | null;
  finish_reason?: string | null;
  starred?: boolean;
  notes?: string;
  created_at?: number;
  edited_at?: number | null;
  edit_count?: number;
  /** Logit-pass: mean chosen-token logprob over the response span when
   *  logprob capture was live; absent on legacy / replayed nodes.  Drives
   *  the loom sidebar's surprise edge-weighting and the
   *  ``sort:surprise`` / ``sort:confidence`` filter grammar. */
  mean_logprob?: number | null;
  /** Per-token response-span rows captured during streaming.  Present
   *  when the server serializes the tree with ``include_tokens=True``
   *  (the webui tree GET path).  Absent on legacy / transcript-loaded
   *  nodes that never streamed under the v2.4 token-row schema. */
  tokens?: LoomTokenRowJSON[] | null;
  /** Per-token thinking-span rows.  Same shape as ``tokens``; populated
   *  only when the engine emitted thinking content for the node. */
  thinking_tokens?: LoomTokenRowJSON[] | null;
  /** Raw decode-step ids the engine sampled, including suppressed
   *  delimiters and unmerged partial-UTF-8 bytes.  The forceable prefix
   *  a logit fork replays from; ``null`` on legacy / transcript-loaded
   *  nodes, in which case the fork affordance falls back to disabled. */
  raw_token_ids?: number[] | null;
}

/** Full tree dump returned by GET /sessions/{id}/tree.
 *
 *  Server's ``LoomTree.to_dict`` serializes ``nodes`` as a list (flat,
 *  preserves insertion order) and ``children_of`` as a parent→ordered
 *  child-id map.  Clients pivot the node list into a dict keyed by id
 *  for the in-memory cache. */
export interface LoomTreeJSON {
  tree_format?: number;
  root_id: string;
  active_node_id: string;
  rev: number;
  nodes: LoomNodeJSON[];
  /** parent_id → ordered list of child ids. */
  children_of: Record<string, string[]>;
  /** Optional model identifier the tree was generated against. */
  model_id?: string | null;
  session_id?: string | null;
  name?: string | null;
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

// ----------------------------------------------------- experiments --

export interface ExperimentFanRequest {
  prompt: unknown;
  /** concept name -> alpha grid */
  grid: Record<string, number[]>;
  base_steering?: string | null;
  sampling?: WSSampling | null;
  thinking?: boolean | null;
  raw?: boolean;
}

export interface ExperimentFanRow {
  idx: number;
  alpha_values: Record<string, number>;
  node_id: string | null;
  result: {
    text: string;
    token_count: number;
    tok_per_sec: number;
    elapsed: number;
    finish_reason: string;
    applied_steering: string | null;
    readings: Record<string, number>;
  };
}

export interface ExperimentFanResponse {
  kind: "fan" | string;
  total: number;
  node_ids: Array<string | null>;
  rows: ExperimentFanRow[];
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
    | string;
  added?: LoomNodeJSON[];
  removed?: string[];
  updated?: LoomNodeJSON[];
  active_node_id?: string | null;
  rev: number;
}

/** Fired at the start of each branch in an n-way generate so the client
 * can allocate render slots before token events arrive. */
export interface WSNodeCreatedEvent {
  type: "node_created";
  node_id: string;
  parent_id: string | null;
  role: "user" | "assistant" | "system";
  rev: number;
}

export type WSServerMessage =
  | WSStartedEvent
  | WSTokenEvent
  | WSDoneEvent
  | WSErrorEvent
  | WSTreeMutatedEvent
  | WSNodeCreatedEvent;

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
  /** Token-id from the WS event when available — useful for debugging. */
  tokenId?: number | null;
  /** Per-layer × per-probe heatmap data captured during streaming.
   * Drives the click-token drilldown drawer. */
  perLayerScores?: Record<string, Record<string, number>>;
  /** Logit-pass: chosen-token post-sampler logprob.  Absent on legacy /
   *  replayed turns when ``return_top_k`` wasn't enabled and the engine
   *  didn't run log_softmax.  Drives the inline ``surprise`` highlight
   *  mode and the token drilldown's logits tab. */
  logprob?: number | null;
  /** Logit-pass: top-K alternatives captured at this position (descending
   *  by logprob).  Absent when ``return_top_k == 0`` or replayed. */
  topAlts?: TokenAltJSON[] | null;
  /** Raw decode-step index of this token in the backing node's
   *  ``raw_token_ids`` — the join key a logit fork slices on.  Absent on
   *  legacy / transcript-loaded nodes (engine pre-dates raw-id capture),
   *  in which case the token can't be forked. */
  rawIndex?: number | null;
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
  /** True iff any thinking content was emitted. */
  thinking?: boolean;
  /** Visible response tokens with score data. */
  tokens?: TokenScore[];
  /** Thinking-only tokens with score data (rendered inside the
   * <Collapsible> equivalent). */
  thinkingTokens?: TokenScore[];
  /** A/B-mode pair: the unsteered shadow turn, rendered side-by-side
   * when present.  Always role: "assistant". */
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
   *  legacy / replayed turns. */
  meanLogprob?: number | null;
}

// ----------------------------------------------------- vector rack --

export interface ProjectionSpec {
  op: "~" | "|";
  target: string;
}

export interface VectorRackEntry {
  /** Slider value in [-1, +1].  Sign is the user's typed sign — ``serialize``
   * preserves it as the term coefficient. */
  alpha: number;
  trigger: Trigger;
  variant: Variant;
  /** Optional projection — keep (``~``) or remove (``|``) the shared
   * component with another concept. */
  projection: ProjectionSpec | null;
  /** When true, term is rendered as ``!name``; bare ``!`` defaults to
   * coeff=1.0 (fully replace).  Cannot compose with projection. */
  ablate: boolean;
  /** When false, the term is excluded from serialization (visual but
   * not active). */
  enabled: boolean;
}

// ----------------------------------------------------- manifold rack --

/** One racked manifold — a steering term placing generation at a point
 *  of a fitted manifold.  ``coords`` is one authoring coordinate per
 *  intrinsic dimension; ``blend`` is the soft subspace-replace fraction
 *  in ``[0, 1]``. */
export interface ManifoldRackEntry {
  /** Blend fraction in [0, 1] — the term coefficient. */
  blend: number;
  /** Authoring coordinates, one per intrinsic dimension. */
  coords: number[];
  /** Optional node-label form of the position: when set, the term
   *  serializes as ``<name>%<label>`` (Phase B label-form) and the
   *  coords are this node's authoring coords mirrored for the XYPad
   *  display.  ``null`` = the position was authored coord-wise (drag
   *  on the XYPad), serialize as the comma-joined coord list.  Pulling
   *  on the XYPad clears the label; picking from the snap-to-node
   *  dropdown sets it. */
  label?: string | null;
  trigger: Trigger;
  enabled: boolean;
}

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
  /** Last N values for the sparkline — ring-buffer-ish, capped client-side. */
  sparkline: number[];
  current: number;
  previous: number;
  /** Most recent token's per-layer readings for *this* probe.  Layer-key
   * strings keep the wire shape; ProbeStrip sorts numerically.  Empty
   * until the first ``token`` event with ``per_layer_scores`` lands. */
  perLayer: Record<string, number>;
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
  /** Unified vector management drawer (replaces the legacy
   *  ``vector_picker`` + ``probe_picker`` pair).  Two sections split
   *  on the server-supplied ``has_tensor`` flag: extracted rows get
   *  steer/probe/delete toggles, statements-only rows get
   *  extract/delete.  Opened from both rack "+ add" buttons. */
  | "vectors"
  /** Custom-vector extraction form — reached from the
   *  "+ custom vector" button at the top of ``vectors``.  Submitting
   *  closes back to the vectors drawer so the new row appears
   *  reactively. */
  | "extract"
  /** Manifold browser — split Fitted / Unfitted, per-row steer / fit /
   *  delete, with a "+ build manifold" launcher. */
  | "manifolds"
  /** Manifold authoring form — domain step + node editor.  Reached
   *  from the "+ build manifold" button inside ``manifolds``. */
  | "manifold_builder"
  /** Manifold-side counterpart to ``MergeDrawer``.  Unions the node
   *  corpora of two or more discover-mode manifolds into a fresh
   *  discover folder; restricted to discover sources by design.
   *  Reached from the workspace rail's "manifolds → merge…" entry,
   *  parallel to "vectors → merge vector…". */
  | "manifold_merge"
  /** Manifold-side counterpart to ``PackDrawer``.  Two tabs: local
   *  catalog, plus HF search/install for ``saklas-manifold``-tagged
   *  repos.  Reached from the workspace rail's "manifolds → packs…"
   *  entry, parallel to "vectors → packs…". */
  | "manifold_pack"
  | "save_conversation"
  | "load_conversation"
  | "compare"
  | "pack"
  | "merge"
  | "clone"
  | "system_prompt"
  | "token_drilldown"
  | "correlation"
  | "layer_norms"
  | "experiment_lab"
  | "activation_atlas"
  | "recipe_builder"
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
  | "transcript";

export interface DrawerState {
  open: DrawerName | null;
  /** Per-drawer params — typed loosely because each drawer owns its own
   * shape (e.g. token drilldown carries the click-target token row). */
  params: unknown;
}
