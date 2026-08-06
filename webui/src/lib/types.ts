// Shared types for the saklas webui.  Every panel/drawer/store imports
// from here so renames stay one-shot — this module is THE import surface,
// including for the generated half.
//
// Two halves:
//
//   * `./types.gen` — the native REST response shapes, GENERATED from the
//     FastAPI app's OpenAPI schema by `scripts/generate_webui_types.py`.
//     Re-exported below, so `import type { SessionInfo } from "./types"`
//     keeps resolving.  Never edit that file; edit the server's
//     `response_models.py` and regenerate.
//   * this file — the shapes OpenAPI cannot describe: the WebSocket frame
//     vocabulary, request bodies, and the dashboard's own client-local
//     state types.  It also composes a few unions over generated members
//     (`ProbeInfo`, `InstrumentLiveState`), which is why the generated
//     names are imported as well as re-exported.

export * from "./types.gen";

import type {
  CastMemberJSON,
  GeometryProbeInfo,
  GeometryLiveState,
  LensAggregateTokenJSON,
  LensLiveState,
  LensProbeInfo,
  LensReadoutLayerJSON,
  LoomNodeJSON,
  MeasurementsEnvelopeJSON,
  ProbeReadingJSON,
  SaeFeatureJSON,
  SaeLiveState,
  SaeProbeInfo,
  ScalarReadingJSON,
} from "./types.gen";

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

// -------------------------------------------------- jacobian lens --

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
export type PreparationOp = "fetch" | "fit" | "train";

export type InstrumentFamily = "geometry" | "lens" | "sae";

/** Live-readout state, discriminated by family: geometry is an
 *  all-or-nothing switch, the lens resolves a layer list, the SAE reports
 *  its resident layer + source.  The three members are generated; the union
 *  over them is composed here (OpenAPI inlines an `anyOf` at the property
 *  rather than naming it). */
export type InstrumentLiveState =
  | GeometryLiveState
  | LensLiveState
  | SaeLiveState;

// ------------------------------------------------ sparse autoencoder --

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

/** Either family's reading shape, as it arrives inside the envelope. */
export type AnyReadingJSON = ProbeReadingJSON | ScalarReadingJSON;

/** True for the single-axis families' native reading. */
export function isScalarReading(
  reading: AnyReadingJSON,
): reading is ScalarReadingJSON {
  return "value" in reading;
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
 *  the webui shows read-only.
 *
 *  Hand-written rather than generated: it is a ``type``-tagged union, and
 *  the server-side declaration is a plain open mapping (a TypedDict cannot
 *  express the tag).  The generated response types reference this one
 *  through the generator's property-override table. */
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

export interface ExtractRequest {
  /** Concept represented by the positive node (or sole monopolar node). */
  concept: string;
  baseline?: string | null;
  /** Elicitation framing for both poles — the same knob
   *  ``POST /manifolds/generate`` carries. ``abstract`` -> "someone {c}",
   *  ``concrete`` -> "{art} {c}", ``custom`` -> ``custom_system`` (no role
   *  swap, works on every model family). */
  kind?: "abstract" | "concrete" | "custom";
  /** System template for ``kind: "custom"`` ({c} = the concept). Required
   *  when ``kind`` is ``custom``; rejected-with-400 otherwise. */
  custom_system?: string | null;
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

// ----------------------------------------------------- probes --

/** One attached probe row, discriminated by ``family`` — the unified
 *  read-side collection under ``/probes`` (the pre-4.0 split of vector vs
 *  manifold probes collapsed onto one route). */
export type ProbeInfo = GeometryProbeInfo | LensProbeInfo | SaeProbeInfo;

/** Body for ``POST /saklas/v1/sessions/{id}/probes`` — attach any probe
 *  shape by selector (the same ``[ns/]name[:variant]`` the ``%`` steering
 *  term consumes). */
export interface ProbeRequest {
  selector: string;
  name?: string;
  top_n?: number;
}

// ------------------------------------------------- probe readings --

// ----------------------------------------------------- probe geometry --

// ----------------------------------------------------- correlation --

// --------------------------------------------------- pairwise compare --

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

/** One message of an explicit conversation replay on the ``generate``
 *  frame's ``input`` list. */
export interface WSInputMessage {
  role: ChatRole;
  content: string;
  label?: string | null;
}

export interface WSGenerateRequest {
  type: "generate";
  /** ``null`` is a continue — no committed turn, the model speaks next
   *  from ``parent_node_id`` (or the active leaf).  A messages list is the
   *  explicit-conversation replay the unsteered shadow uses. */
  input?: WSInputMessage[] | null;
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
  /** End-of-generation measurement envelope, ``scope: "aggregate"`` — the
   *  same shape the ``token`` frame carries, pooled at the last-content
   *  token.  The per-attached-probe readings live under
   *  ``instruments.{geometry,lens,sae}.readings`` and merge by name exactly
   *  as the token path merges them; there is no flat ``probe_readings``
   *  alias on this frame.  Omitted when no probe is attached — read
   *  defensively. */
  measurements?: MeasurementsEnvelopeJSON;
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
  /** Per-layer × per-probe heatmap — each reading's ``coords_per_layer``
   *  axis 0, flattened server-side by ``token_payloads._per_layer_axis0``
   *  and keyed by stringified layer index.  Drives the token-drilldown
   *  drawer's heatmap on rehydrated turns. */
  per_layer_scores?: Record<string, Record<string, number>>;
  /** The 5.x measurement envelope captured by the original generation. This
   * survives tree rehydration and explicit loom save/load without replaying
   * the model. Replaces the pre-5.x ``captured`` record. */
  measurements?: MeasurementsEnvelopeJSON;
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

/** Per-token score row for chat highlighting — the text + whichever
 * probe scores are known for the token, filled at render time. */
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

/** The two single-direction *atom* families — a J-lens token direction
 *  and an SAE decoder row.  Both rack as ``α <prefix><id>`` with one
 *  per-card coefficient and no geometry, so they share a card and a
 *  mutator set; only the key prefix, the accent hue, and the marker
 *  glyph differ. */
export type AtomMode = "jlens" | "sae";

export type AtomSteerEntry = JLensSteerEntry | SaeSteerEntry;

/** A racked steering term — subspace (flat), manifold (curved), or a
 *  single-direction atom. */
export type SteerEntry =
  | SubspaceSteerEntry
  | ManifoldSteerEntry
  | JLensSteerEntry
  | SaeSteerEntry;

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
  reading: AnyReadingJSON | null;
  /** End-of-gen aggregate the ``done`` event lands — the settled reading.
   *  Null between gens; set on ``done``, cleared on the next ``started``. */
  aggregate: AnyReadingJSON | null;
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
