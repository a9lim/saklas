// Minimal REST + WS clients for the native /saklas/v1/* API.

const SESSION = "default";
export const API = `/saklas/v1/sessions/${SESSION}`;

export async function getJson<T>(path: string): Promise<T> {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return (await r.json()) as T;
}

export interface SessionInfo {
  id: string;
  model_id: string;
  device: string;
  dtype: string;
  vectors: string[];
  probes: string[];
}

export interface VectorInfo {
  name: string;
  layers: number[];
  per_layer_norms: Record<string, number>;
  metadata: Record<string, unknown>;
}

export interface CorrelationData {
  names: string[];
  matrix: Record<string, Record<string, number | null>>;
  layers_shared: Record<string, number>;
}

export type WsMessage =
  | { type: "started"; generation_id: string }
  | {
      type: "token";
      text: string;
      thinking: boolean;
      token_id: number | null;
      per_layer_scores?: Record<string, Record<string, number>>;
    }
  | { type: "done"; result: unknown }
  | { type: "error"; message: string; code?: string };

export function connectWs(): WebSocket {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  return new WebSocket(`${proto}://${location.host}/saklas/v1/sessions/${SESSION}/stream`);
}
