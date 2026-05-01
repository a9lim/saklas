// Reactive stores backing the dashboard panels.

import { writable } from "svelte/store";
import { API, getJson } from "./api";
import type { CorrelationData, SessionInfo, VectorInfo } from "./api";

export const sessionInfo = writable<SessionInfo | null>(null);
export const vectors = writable<string[]>([]);
export const selectedVector = writable<string | null>(null);
export const layerNormsByVector = writable<Record<string, VectorInfo>>({});
export const correlation = writable<CorrelationData | null>(null);

// Inspector data: per-token columns of (text, thinking, scores).
export interface InspectorToken {
  text: string;
  thinking: boolean;
  scores: Record<string, Record<string, number>>;
}
export const inspectorTokens = writable<InspectorToken[]>([]);

export async function refreshSession(): Promise<void> {
  try {
    const info = await getJson<SessionInfo>(API);
    sessionInfo.set(info);
    vectors.set(info.vectors);
  } catch {
    sessionInfo.set(null);
  }
}

export async function refreshVector(name: string): Promise<void> {
  const info = await getJson<VectorInfo>(`${API}/vectors/${encodeURIComponent(name)}`);
  layerNormsByVector.update((cur) => ({ ...cur, [name]: info }));
}

export async function refreshCorrelation(): Promise<void> {
  try {
    const data = await getJson<CorrelationData>(`${API}/correlation`);
    correlation.set(data);
  } catch {
    correlation.set(null);
  }
}
