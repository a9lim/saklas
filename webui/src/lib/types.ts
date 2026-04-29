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

export interface MergeVectorResponse {
  canonical: string;
  profile: VectorInfo;
}

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

/** Output of GET /sessions/{id}/vectors/{name}/diagnostics — 16-bucket
 * histogram + summary metrics from `saklas vector why`. */
export interface VectorDiagnosticsResponse {
  name: string;
  /** Layer histogram entries.  ``label`` is ``LXX`` or ``LXX-YY`` zero-padded;
   * ``value`` is the mean ``||baked||`` for the bucket. */
  histogram: { label: string; value: number; lo: number; hi: number }[];
  /** Per-layer ``||baked||`` keyed by string layer index. */
  per_layer_norms: Record<string, number>;
  /** Median diagnostics across retained layers (when extracted with v1.6+). */
  diagnostics_summary: {
    evr: number | null;
    intra_pair_variance_mean: number | null;
    inter_pair_alignment: number | null;
    diff_principal_projection: number | null;
    /** "solid" | "shaky" | "poor" — derived stoplight. */
    stoplight: "solid" | "shaky" | "poor" | "unknown";
  };
  diagnostics_by_layer: Record<string, Record<string, number>>;
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
  installed: string[];
  progress: string[];
}

// ----------------------------------------------------- sweep --

export interface SweepRequest {
  prompt: unknown;
  /** ``{concept_name: [alpha, ...], ...}``.  Cartesian product across
   * concepts becomes one generation per row. */
  sweep: Record<string, number[]>;
  base_steering?: string | null;
  sampling?: WSSampling | null;
  thinking?: boolean | null;
  stateless?: boolean;
  raw?: boolean;
}

export interface SweepRowReadings {
  [probe: string]: number;
}

export interface SweepRowResult {
  text: string;
  token_count: number;
  tok_per_sec: number;
  elapsed: number;
  finish_reason: string;
  applied_steering: string | null;
  readings: SweepRowReadings;
}

export type SweepEvent =
  | { type: "started"; sweep_id: string; total: number }
  | {
      type: "result";
      idx: number;
      alpha_values: Record<string, number>;
      result: SweepRowResult;
    }
  | {
      type: "done";
      sweep_id: string;
      summary: {
        completed: number;
        total: number;
        total_tokens: number;
        tok_per_sec: number;
        elapsed: number;
      };
    }
  | { type: "error"; message: string };

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
  presence_penalty?: number;
  frequency_penalty?: number;
}

export interface WSGenerateRequest {
  type: "generate";
  input: string | unknown;
  steering?: string | null;
  sampling?: WSSampling | null;
  thinking?: boolean | null;
  stateless?: boolean;
  raw?: boolean;
}

export interface WSStopRequest {
  type: "stop";
}

export type WSClientMessage = WSGenerateRequest | WSStopRequest;

export interface WSStartedEvent {
  type: "started";
  generation_id: string;
}

export interface WSTokenEvent {
  type: "token";
  text: string;
  thinking: boolean;
  token_id: number | null;
  /** Per-layer × per-probe map populated only when probes are loaded.
   * Keys: layer-index strings → probe names → cosine-sim score. */
  per_layer_scores?: Record<string, Record<string, number>>;
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
}

export interface WSDoneEvent {
  type: "done";
  result: WSDoneResult;
}

export interface WSErrorEvent {
  type: "error";
  message: string;
  code?: string;
}

export type WSServerMessage =
  | WSStartedEvent
  | WSTokenEvent
  | WSDoneEvent
  | WSErrorEvent;

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
}

export interface ChatTurn {
  role: "user" | "assistant" | "system";
  text: string;
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

/** Actions queued during in-flight generation.  ``apply`` is the closure
 * the store invokes once the WS ``done`` event arrives (or immediately
 * if the user hits "apply now").  ``label`` shows in the topbar pending
 * badge for traceability. */
export interface PendingAction {
  id: string;
  label: string;
  apply: () => void | Promise<void>;
  createdAt: number;
}

// ----------------------------------------------------- drawers --

export type DrawerName =
  | "extract"
  | "load"
  | "vector_picker"
  | "probe_picker"
  | "save_conversation"
  | "load_conversation"
  | "compare"
  | "sweep"
  | "pack"
  | "merge"
  | "clone"
  | "system_prompt"
  | "model_info"
  | "token_drilldown"
  | "export"
  | "help";

export interface DrawerState {
  open: DrawerName | null;
  /** Per-drawer params — typed loosely because each drawer owns its own
   * shape (e.g. token drilldown carries the click-target token row). */
  params: unknown;
}
