// Conversation-walking cursor for the token drilldown.
//
// The drilldown treats the whole conversation as ONE walkable token
// sequence: each turn contributes up to two segments (thinking, then
// response), and stepping past a segment boundary rolls into the next
// segment / turn instead of stopping.  That makes chat mode navigate
// exactly like the raw-buffer view of the same tree — a flat stream —
// while the cursor still knows which (turn, segment, index) it sits on
// so every read surface keys correctly.
//
// Pure helpers over ChatTurn[] — no runes, no store reads.  The drawer
// owns the reactive cursor state and calls these to move it.

import type { ChatTurn, TokenScore } from "../../lib/types";

export type SegmentKind = "thinking" | "response";

/** A position in the conversation token stream. */
export interface TokenCursor {
  turnIdx: number;
  seg: SegmentKind;
  tokenIdx: number;
}

/** One inspectable segment of the walk (a turn's thinking or response
 *  token list, when non-empty). */
export interface SegmentRef {
  turnIdx: number;
  seg: SegmentKind;
  length: number;
}

/** The token list backing one segment of a turn. */
export function segmentTokens(
  turn: ChatTurn | null | undefined,
  seg: SegmentKind,
): TokenScore[] {
  if (!turn) return [];
  return (seg === "thinking" ? turn.thinkingTokens : turn.tokens) ?? [];
}

/** Flatten the conversation into its ordered inspectable segments. */
export function segmentsOf(turns: ChatTurn[]): SegmentRef[] {
  const out: SegmentRef[] = [];
  turns.forEach((turn, turnIdx) => {
    const think = turn.thinkingTokens?.length ?? 0;
    if (think > 0) out.push({ turnIdx, seg: "thinking", length: think });
    const resp = turn.tokens?.length ?? 0;
    if (resp > 0) out.push({ turnIdx, seg: "response", length: resp });
  });
  return out;
}

export function segmentIndex(segs: SegmentRef[], cursor: TokenCursor): number {
  return segs.findIndex(
    (s) => s.turnIdx === cursor.turnIdx && s.seg === cursor.seg,
  );
}

/** Step the cursor ±1 token, rolling across segment and turn boundaries.
 *  Returns null at the very start / end of the conversation (or when the
 *  cursor's segment no longer exists). */
export function stepCursor(
  segs: SegmentRef[],
  cursor: TokenCursor,
  delta: 1 | -1,
): TokenCursor | null {
  const i = segmentIndex(segs, cursor);
  if (i < 0) return null;
  const next = cursor.tokenIdx + delta;
  if (next >= 0 && next < segs[i].length) {
    return { ...cursor, tokenIdx: next };
  }
  const j = i + delta;
  if (j < 0 || j >= segs.length) return null;
  const s = segs[j];
  return {
    turnIdx: s.turnIdx,
    seg: s.seg,
    tokenIdx: delta > 0 ? 0 : s.length - 1,
  };
}

/** Jump to the first token of the previous / next turn that carries any
 *  token rows.  Null when no such turn exists in that direction. */
export function jumpTurn(
  segs: SegmentRef[],
  cursor: TokenCursor,
  delta: 1 | -1,
): TokenCursor | null {
  const target =
    delta > 0
      ? segs.find((s) => s.turnIdx > cursor.turnIdx)
      : [...segs].reverse().find((s) => s.turnIdx < cursor.turnIdx);
  if (!target) return null;
  // Land on the turn's FIRST segment even when scanning backwards.
  const first = segs.find((s) => s.turnIdx === target.turnIdx)!;
  return { turnIdx: first.turnIdx, seg: first.seg, tokenIdx: 0 };
}

/** Clamp a cursor onto the current segment list — token lists grow while
 *  streaming and shrink on branch flips / turn deletion.  Falls to the
 *  turn's other segment, then the nearest following turn; null only when
 *  the conversation has no tokens at all. */
export function clampCursor(
  segs: SegmentRef[],
  cursor: TokenCursor,
): TokenCursor | null {
  if (segs.length === 0) return null;
  const i = segmentIndex(segs, cursor);
  if (i >= 0) {
    return cursor.tokenIdx < segs[i].length
      ? cursor
      : { ...cursor, tokenIdx: segs[i].length - 1 };
  }
  const sameTurn = segs.find((s) => s.turnIdx === cursor.turnIdx);
  if (sameTurn) {
    return {
      turnIdx: sameTurn.turnIdx,
      seg: sameTurn.seg,
      tokenIdx: Math.min(cursor.tokenIdx, sameTurn.length - 1),
    };
  }
  const fallback =
    segs.find((s) => s.turnIdx > cursor.turnIdx) ?? segs[segs.length - 1];
  return { turnIdx: fallback.turnIdx, seg: fallback.seg, tokenIdx: 0 };
}
