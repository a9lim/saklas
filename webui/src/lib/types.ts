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

/** SAE variant suffix — ``raw`` (default), ``sae`` (unique), ``sae-<release>``. */
export type Variant = "raw" | "sae" | `sae-${string}`;

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
  default_steering: string | null;
  /** Non-canonical: optional architecture string surfaced for the
   * yellow-banner warning when ``model_type`` isn't in
   * ``_TESTED_ARCHS``.  Server may or may not populate this; clients
   * should tolerate ``undefined``. */
  architecture?: string;
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
  register?: boolean;
}

export interface ExtractResponse {
  canonical: string;
  profile: VectorInfo;
  progress: string[];
}

export interface LoadVectorRequest {
  name: string;
  source_path: string;
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
}

// ----------------------------------------------------- drawers --

export type DrawerName =
  | "load"
  | "vector_picker"
  | "probe_picker"
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
