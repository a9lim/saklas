// Typed REST + WS + SSE client for the native /saklas/v1/* API.
//
// Single source of truth for HTTP shapes; types live in types.ts so panels
// and drawers can `import type` without dragging the fetch helpers along.
//
// Bearer auth: read once at module load from <meta name="api-key">; if the
// page injects a key after load, call ``setApiKey()`` to refresh it.

import type {
  CloneVectorRequest,
  CloneVectorResponse,
  CorrelationData,
  ExtractRequest,
  ExtractResponse,
  InstallPackRequest,
  InstallPackResponse,
  LoadVectorRequest,
  MergeVectorRequest,
  MergeVectorResponse,
  PackListResponse,
  PackSearchResponse,
  ProbeDefaultsResponse,
  ProbeListResponse,
  ScoreProbeRequest,
  ScoreProbeResponse,
  SessionInfo,
  SweepEvent,
  SweepRequest,
  TraitsEvent,
  VectorDiagnosticsResponse,
  VectorInfo,
  VectorListResponse,
  WSClientMessage,
  WSServerMessage,
} from "./types";

// Re-export the wire-shape types other modules consume.  Doing this here
// rather than via barrel-export keeps `import { ApiError, getSession }`
// consumers one-stop without a separate type import line.
export type {
  CloneVectorRequest,
  CloneVectorResponse,
  CorrelationData,
  ExtractRequest,
  ExtractResponse,
  InstallPackRequest,
  InstallPackResponse,
  LoadVectorRequest,
  MergeVectorRequest,
  MergeVectorResponse,
  PackListResponse,
  PackSearchResponse,
  ProbeDefaultsResponse,
  ProbeListResponse,
  ScoreProbeRequest,
  ScoreProbeResponse,
  SessionInfo,
  SweepEvent,
  SweepRequest,
  TraitsEvent,
  VectorDiagnosticsResponse,
  VectorInfo,
  VectorListResponse,
  WSClientMessage,
  WSServerMessage,
} from "./types";

// Aliased name for legacy compat with the v1.6 ``WsMessage`` type — the
// old Chat panel imports it; keeping the alias means we don't need to
// touch panels in this phase.
export type WsMessage = WSServerMessage;

// --------------------------------------------------------- session id --

/** The native API is multi-session-shaped but the current impl is single-session.
 * The id ``"default"`` always resolves; the loaded model id also resolves but
 * isn't reachable via the WS path-param (HF model ids contain ``/`` which
 * the WS route doesn't declare as a ``:path`` param). */
const SESSION = "default";

/** Legacy v1.6 export — old panels build URLs as ``${API}/vectors``.  Kept
 * unchanged so this rewrite doesn't ripple into the panel layer. */
export const API = `/saklas/v1/sessions/${SESSION}`;

const SESSION_BASE = (id: string = SESSION) => `/saklas/v1/sessions/${id}`;
const PACKS_BASE = "/saklas/v1/packs";

// --------------------------------------------------------- auth --

let _apiKey: string | null = readApiKeyFromMeta();

function readApiKeyFromMeta(): string | null {
  if (typeof document === "undefined") return null;
  const tag = document.querySelector<HTMLMetaElement>('meta[name="api-key"]');
  const v = tag?.content?.trim();
  return v ? v : null;
}

/** Override the in-memory API key.  Pass ``null`` to clear. */
export function setApiKey(key: string | null): void {
  _apiKey = key && key.trim() ? key.trim() : null;
}

export function getApiKey(): string | null {
  return _apiKey;
}

function authHeaders(extra: HeadersInit = {}): HeadersInit {
  const h: Record<string, string> = {};
  if (_apiKey) h["Authorization"] = `Bearer ${_apiKey}`;
  // Merge extra last so the caller can override (e.g. drop Authorization
  // for an open endpoint).
  if (extra instanceof Headers) {
    extra.forEach((v, k) => (h[k] = v));
  } else if (Array.isArray(extra)) {
    for (const [k, v] of extra) h[k] = v;
  } else {
    Object.assign(h, extra);
  }
  return h;
}

// --------------------------------------------------------- error type --

/** Wraps a non-2xx HTTP response with structured detail.  Always thrown
 * from the typed helpers below — call sites can ``catch (e) { if (e
 * instanceof ApiError) ... }`` to extract status / parsed body / raw text. */
export class ApiError extends Error {
  readonly status: number;
  readonly path: string;
  readonly body: unknown;
  readonly rawBody: string;

  constructor(status: number, path: string, rawBody: string, parsed: unknown) {
    super(
      `${path}: ${status} ${
        parsed && typeof parsed === "object" && "detail" in (parsed as object)
          ? (parsed as { detail: unknown }).detail
          : rawBody.slice(0, 200)
      }`,
    );
    this.name = "ApiError";
    this.status = status;
    this.path = path;
    this.rawBody = rawBody;
    this.body = parsed;
  }
}

// --------------------------------------------------------- core fetch --

async function parseBody(r: Response): Promise<{ text: string; json: unknown }> {
  const text = await r.text();
  if (!text) return { text, json: null };
  try {
    return { text, json: JSON.parse(text) };
  } catch {
    return { text, json: null };
  }
}

async function request<T>(
  path: string,
  init: RequestInit & { acceptJson?: boolean } = {},
): Promise<T> {
  const { acceptJson = true, headers, ...rest } = init;
  const merged = authHeaders(headers ?? {});
  if (acceptJson) {
    (merged as Record<string, string>)["Accept"] = "application/json";
  }
  const r = await fetch(path, { ...rest, headers: merged });
  if (!r.ok) {
    const { text, json } = await parseBody(r);
    throw new ApiError(r.status, path, text, json);
  }
  if (r.status === 204) return undefined as T;
  if (acceptJson) {
    const { json, text } = await parseBody(r);
    if (json === null) {
      // Some 200 responses have empty bodies — narrow to T at the boundary.
      return text as unknown as T;
    }
    return json as T;
  }
  return (await r.text()) as unknown as T;
}

/** Legacy export — preserved for the v1.6 stores.ts file.  Throws ``Error``
 * (not ``ApiError``) for backwards-compat. */
export async function getJson<T>(path: string): Promise<T> {
  const r = await fetch(path, { headers: authHeaders() });
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return (await r.json()) as T;
}

function jsonBody(body: unknown): RequestInit {
  return {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
}

// =========================================================== sessions ==

export const apiSessions = {
  list(): Promise<{ sessions: SessionInfo[] }> {
    return request("/saklas/v1/sessions");
  },
  /** POST /sessions — single-session impl; calling with a different
   * model logs a server warning but returns the existing session. */
  create(body: { model?: string; device?: string; dtype?: string } = {}): Promise<
    SessionInfo
  > {
    return request("/saklas/v1/sessions", jsonBody(body));
  },
  get(id: string = SESSION): Promise<SessionInfo> {
    return request(SESSION_BASE(id));
  },
  patch(
    body: Partial<{
      temperature: number;
      top_p: number;
      top_k: number;
      max_tokens: number;
      system_prompt: string;
      thinking: boolean;
    }>,
    id: string = SESSION,
  ): Promise<SessionInfo> {
    return request(SESSION_BASE(id), { ...jsonBody(body), method: "PATCH" });
  },
  delete(id: string = SESSION): Promise<void> {
    return request<void>(SESSION_BASE(id), { method: "DELETE" });
  },
  clear(id: string = SESSION): Promise<void> {
    return request<void>(`${SESSION_BASE(id)}/clear`, { method: "POST" });
  },
  rewind(id: string = SESSION): Promise<void> {
    return request<void>(`${SESSION_BASE(id)}/rewind`, { method: "POST" });
  },
};

// ============================================================ vectors ==

export const apiVectors = {
  list(id: string = SESSION): Promise<VectorListResponse> {
    return request(`${SESSION_BASE(id)}/vectors`);
  },
  get(name: string, id: string = SESSION): Promise<VectorInfo> {
    return request(`${SESSION_BASE(id)}/vectors/${encodeURIComponent(name)}`);
  },
  load(req: LoadVectorRequest, id: string = SESSION): Promise<VectorInfo> {
    return request(`${SESSION_BASE(id)}/vectors`, jsonBody(req));
  },
  delete(name: string, id: string = SESSION): Promise<void> {
    return request<void>(
      `${SESSION_BASE(id)}/vectors/${encodeURIComponent(name)}`,
      { method: "DELETE" },
    );
  },
  /** Synchronous extract.  For SSE progress pass ``onProgress`` to
   * ``apiExtractStream`` instead. */
  extract(req: ExtractRequest, id: string = SESSION): Promise<ExtractResponse> {
    return request(`${SESSION_BASE(id)}/extract`, jsonBody(req));
  },
  merge(
    req: MergeVectorRequest,
    id: string = SESSION,
  ): Promise<MergeVectorResponse> {
    return request(`${SESSION_BASE(id)}/vectors/merge`, jsonBody(req));
  },
  /** Synchronous clone.  Use ``apiCloneStream`` for SSE progress. */
  clone(
    req: CloneVectorRequest,
    id: string = SESSION,
  ): Promise<CloneVectorResponse> {
    return request(`${SESSION_BASE(id)}/vectors/clone`, jsonBody(req));
  },
  diagnostics(
    name: string,
    id: string = SESSION,
  ): Promise<VectorDiagnosticsResponse> {
    return request(
      `${SESSION_BASE(id)}/vectors/${encodeURIComponent(name)}/diagnostics`,
    );
  },
  correlation(
    names?: string[] | null,
    id: string = SESSION,
  ): Promise<CorrelationData> {
    const q = names && names.length ? `?names=${encodeURIComponent(names.join(","))}` : "";
    return request(`${SESSION_BASE(id)}/correlation${q}`);
  },
};

// ============================================================= probes ==

export const apiProbes = {
  list(id: string = SESSION): Promise<ProbeListResponse> {
    return request(`${SESSION_BASE(id)}/probes`);
  },
  defaults(id: string = SESSION): Promise<ProbeDefaultsResponse> {
    return request(`${SESSION_BASE(id)}/probes/defaults`);
  },
  activate(name: string, id: string = SESSION): Promise<void> {
    return request<void>(
      `${SESSION_BASE(id)}/probes/${encodeURIComponent(name)}`,
      { method: "POST" },
    );
  },
  deactivate(name: string, id: string = SESSION): Promise<void> {
    return request<void>(
      `${SESSION_BASE(id)}/probes/${encodeURIComponent(name)}`,
      { method: "DELETE" },
    );
  },
  /** One-shot text scoring; no generation involved. */
  score(req: ScoreProbeRequest, id: string = SESSION): Promise<ScoreProbeResponse> {
    return request(`${SESSION_BASE(id)}/probe`, jsonBody(req));
  },
};

// ============================================================== packs ==

/** Pack endpoints are top-level (not under /sessions/{id}/) — server-side
 * pack management is independent of any active session.  Search lives at
 * /saklas/v1/packs/search; ``list`` and ``install`` share the collection
 * URL with method-based dispatch. */
export const apiPacks = {
  list(): Promise<PackListResponse> {
    return request(PACKS_BASE);
  },
  search(query: string, limit?: number): Promise<PackSearchResponse> {
    const q = new URLSearchParams({ q: query });
    if (limit !== undefined) q.set("limit", String(limit));
    return request(`${PACKS_BASE}/search?${q.toString()}`);
  },
  install(req: InstallPackRequest): Promise<InstallPackResponse> {
    return request(PACKS_BASE, jsonBody(req));
  },
};

// ============================================================== sweep ==

export const apiSweep = {
  /** Returns the SSE Response — call ``consumeSse`` on ``response.body``
   * to iterate events.  Sweep is always SSE because the JSON branch
   * doesn't exist server-side. */
  async start(req: SweepRequest, id: string = SESSION): Promise<Response> {
    const r = await fetch(`${SESSION_BASE(id)}/sweep`, {
      method: "POST",
      headers: authHeaders({
        "Content-Type": "application/json",
        Accept: "text/event-stream",
      }),
      body: JSON.stringify(req),
    });
    if (!r.ok) {
      const { text, json } = await parseBody(r);
      throw new ApiError(r.status, `${SESSION_BASE(id)}/sweep`, text, json);
    }
    return r;
  },
};

// ============================================================ extract ==

/** Streaming extract — pass ``onEvent`` to receive each SSE frame as it
 * arrives.  Resolves with the final ``done`` payload (or rejects on the
 * server's ``error`` frame). */
export async function apiExtractStream(
  req: ExtractRequest,
  onEvent: (ev: { event: string; data: unknown }) => void,
  id: string = SESSION,
): Promise<{ canonical: string; profile: VectorInfo }> {
  const r = await fetch(`${SESSION_BASE(id)}/extract`, {
    method: "POST",
    headers: authHeaders({
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    }),
    body: JSON.stringify(req),
  });
  if (!r.ok) {
    const { text, json } = await parseBody(r);
    throw new ApiError(r.status, `${SESSION_BASE(id)}/extract`, text, json);
  }
  if (!r.body) throw new Error("extract: server returned no SSE body");
  let final: { canonical: string; profile: VectorInfo } | null = null;
  let lastError: string | null = null;
  for await (const evt of consumeSse(r.body)) {
    onEvent(evt);
    if (evt.event === "done" && evt.data && typeof evt.data === "object") {
      const d = evt.data as { canonical?: string; profile?: VectorInfo };
      if (d.canonical && d.profile) {
        final = { canonical: d.canonical, profile: d.profile };
      }
    } else if (evt.event === "error") {
      const d = evt.data as { message?: string };
      lastError = d?.message ?? "extract failed";
    }
  }
  if (lastError) throw new Error(lastError);
  if (!final) throw new Error("extract: stream ended without done event");
  return final;
}

// ============================================================== clone ==

/** Streaming clone, mirrors ``apiExtractStream``.  ``done`` data carries
 * ``canonical`` + ``profile``; only the ``done`` and ``error`` events fire
 * (the underlying clone path has no progress callback). */
export async function apiCloneStream(
  req: CloneVectorRequest,
  onEvent: (ev: { event: string; data: unknown }) => void,
  id: string = SESSION,
): Promise<{ canonical: string; profile: VectorInfo }> {
  const path = `${SESSION_BASE(id)}/vectors/clone`;
  const r = await fetch(path, {
    method: "POST",
    headers: authHeaders({
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    }),
    body: JSON.stringify(req),
  });
  if (!r.ok) {
    const { text, json } = await parseBody(r);
    throw new ApiError(r.status, path, text, json);
  }
  if (!r.body) throw new Error("clone: server returned no SSE body");
  let final: { canonical: string; profile: VectorInfo } | null = null;
  let lastError: string | null = null;
  for await (const evt of consumeSse(r.body)) {
    onEvent(evt);
    if (evt.event === "done" && evt.data && typeof evt.data === "object") {
      const d = evt.data as { canonical?: string; profile?: VectorInfo };
      if (d.canonical && d.profile) {
        final = { canonical: d.canonical, profile: d.profile };
      }
    } else if (evt.event === "error") {
      lastError =
        (evt.data as { message?: string } | null)?.message ?? "clone failed";
    }
  }
  if (lastError) throw new Error(lastError);
  if (!final) throw new Error("clone: stream ended without done event");
  return final;
}

// =========================================================== traits ====

/** Open the live traits SSE stream.  Returns the underlying ``Response``
 * so the caller owns lifecycle (cancel via ``response.body.cancel()``). */
export async function apiTraitsStream(id: string = SESSION): Promise<Response> {
  const r = await fetch(`${SESSION_BASE(id)}/traits/stream`, {
    headers: authHeaders({ Accept: "text/event-stream" }),
  });
  if (!r.ok) {
    const { text, json } = await parseBody(r);
    throw new ApiError(r.status, `${SESSION_BASE(id)}/traits/stream`, text, json);
  }
  return r;
}

// ============================================================ SSE util =

export interface SseEvent {
  /** SSE ``event:`` field — defaults to ``"message"`` per the spec when
   * absent (which is also the wire format the sweep / traits endpoints use). */
  event: string;
  /** Parsed JSON if the data line was JSON, otherwise the raw string. */
  data: unknown;
  /** Optional event id from ``id:`` line.  None of the saklas SSE endpoints
   * set it today, but exposing the field future-proofs. */
  id?: string;
}

/** Async generator over a Server-Sent Events stream.  Accepts either a
 * ``ReadableStream<Uint8Array>`` (from ``Response.body``) or a fully-built
 * ``Response`` — the latter is the more ergonomic call site.  Yields one
 * event per blank-line-delimited frame; tolerates missing ``event:`` lines
 * by defaulting to ``"message"``; tries ``JSON.parse(data)`` and falls
 * back to the raw string on failure (the sweep stream uses bare ``data:``
 * frames with JSON inside, the extract stream uses named events). */
export async function* consumeSse(
  source: Response | ReadableStream<Uint8Array>,
): AsyncGenerator<SseEvent, void, void> {
  const stream =
    source instanceof Response
      ? source.body
      : (source as ReadableStream<Uint8Array>);
  if (!stream) throw new Error("consumeSse: no readable stream");
  const reader = stream.getReader();
  const decoder = new TextDecoder("utf-8");
  let buf = "";
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let idx: number;
      // SSE frames are separated by a blank line — accept either ``\n\n``
      // or ``\r\n\r\n``.  Loop because one network read can deliver many.
      while (true) {
        const a = buf.indexOf("\n\n");
        const b = buf.indexOf("\r\n\r\n");
        if (a === -1 && b === -1) break;
        if (a !== -1 && (b === -1 || a < b)) {
          idx = a;
          const frame = buf.slice(0, idx);
          buf = buf.slice(idx + 2);
          const ev = parseSseFrame(frame);
          if (ev) yield ev;
        } else {
          idx = b;
          const frame = buf.slice(0, idx);
          buf = buf.slice(idx + 4);
          const ev = parseSseFrame(frame);
          if (ev) yield ev;
        }
      }
    }
    // Flush any trailing frame without separator.
    if (buf.trim()) {
      const ev = parseSseFrame(buf);
      if (ev) yield ev;
    }
  } finally {
    try {
      reader.releaseLock();
    } catch {
      /* ignore */
    }
  }
}

function parseSseFrame(frame: string): SseEvent | null {
  // Comments — ``: heartbeat`` lines — are not events.  Strip and bail
  // when the entire frame is comment-only.
  let event = "message";
  let id: string | undefined;
  const dataLines: string[] = [];
  let sawAny = false;
  for (const line of frame.split(/\r?\n/)) {
    if (!line) continue;
    if (line.startsWith(":")) continue;
    sawAny = true;
    const colon = line.indexOf(":");
    if (colon === -1) {
      // Bare field (e.g. ``data``) per spec — treat value as empty.
      if (line === "data") dataLines.push("");
      continue;
    }
    const field = line.slice(0, colon);
    // Spec: a single space after the colon is part of the format, not
    // the value — strip exactly one if present.
    let value = line.slice(colon + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    if (field === "event") event = value || "message";
    else if (field === "data") dataLines.push(value);
    else if (field === "id") id = value;
    // ``retry`` and unknown fields are ignored, per spec.
  }
  if (!sawAny) return null;
  const raw = dataLines.join("\n");
  let data: unknown = raw;
  if (raw) {
    try {
      data = JSON.parse(raw);
    } catch {
      // Leave as string.
    }
  }
  return { event, data, id };
}

// ================================================================ WS ===

/** Open a WebSocket to the per-session token+probe co-stream.  Auth tokens
 * can't ride a custom header on the browser WS API; the server's
 * ``ws_auth_ok`` accepts a query-string ``?token=...`` fallback for
 * Origin checks but the bearer is enforced on the HTTP side too — for
 * now we send the token as a query param when present.  When the API key
 * is unset the connection is open. */
export function connectWs(id: string = SESSION): WebSocket {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  let url = `${proto}://${location.host}${SESSION_BASE(id)}/stream`;
  if (_apiKey) {
    // Token-as-query-param is the standard fallback for browser WS auth
    // since the constructor can't set Authorization.  Server-side
    // middleware must accept it via ``ws_auth_ok``.
    url += `?token=${encodeURIComponent(_apiKey)}`;
  }
  return new WebSocket(url);
}

// =================================================== legacy compat ====

/** Legacy v1.6 export — kept so existing panels keep working through the
 * Phase 2/3 transition.  Newer code should use ``apiSessions.get()``. */
export interface LegacySessionInfo {
  id: string;
  model_id: string;
  device: string;
  dtype: string;
  vectors: string[];
  probes: string[];
}
