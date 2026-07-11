<script lang="ts">
  // A single-node chip in the loom sidebar.  Role glyph, first ~40 chars of
  // text, active-path bolded, dead branches dimmed, plus steering-delta edge
  // labels, probe-aggregate ring decoration, star/note glyphs, and pin
  // affordances — the sidebar feeds all of these live (LoomSidebar's
  // ringFor / weightBadgeFor / steerLabelFor).

  import type { LoomNodeJSON } from "../../lib/types";
  import { roleGlyphLetter } from "../../lib/stores.svelte";

  interface Props {
    node: LoomNodeJSON;
    /** Active path membership — bold + ring + accent. */
    onActivePath: boolean;
    /** Current focused node for sidebar keyboard nav (j/k/h/l). */
    focused: boolean;
    /** Dead branch (not on active path) — render at reduced opacity. */
    dead: boolean;
    /** In-flight target — pulse the node so the user sees streaming. */
    streaming: boolean;
    /** Click handler — navigate to this node. */
    onclick?: (ev: MouseEvent) => void;
    /** Right-click handler — open the context menu. */
    oncontextmenu?: (ev: MouseEvent) => void;
    /** Optional probe ring fill in [-1, 1] (null when no probe aggregate). */
    ring?: number | null;
    /** Logit-pass: per-turn ``mean_logprob`` to render as a numeric
     *  badge.  Null (capture wasn't live) suppresses the badge. */
    weightBadge?: number | null;
    /** Steering-delta label for the edge into this node (e.g.
     *  ``0.45 angry.calm``).  Rendered as a trailing chip — it used to
     *  be an absolutely-positioned label on the edge column that
     *  overlapped this node's text.  Null suppresses it. */
    steerLabel?: string | null;
  }

  let {
    node,
    onActivePath,
    focused,
    dead,
    streaming,
    onclick,
    oncontextmenu,
    ring = null,
    weightBadge = null,
    steerLabel = null,
  }: Props = $props();

  const PREVIEW_CHARS = 40;

  // Glyph honors the node's per-turn role label (e.g. ``captain`` → ``C``);
  // default roles reduce to ``U`` / ``A`` / ``S``.
  function roleGlyph(node: LoomNodeJSON): string {
    return roleGlyphLetter(node.role, node.role_label);
  }

  const preview = $derived.by(() => {
    const t = (node.text ?? "").replace(/\s+/g, " ").trim();
    if (!t) return node.role === "system" && !node.parent_id ? "root" : "(empty)";
    return t.length > PREVIEW_CHARS ? t.slice(0, PREVIEW_CHARS) + "…" : t;
  });

  // Ring color: caller passes ``ring`` in [-1,1] — negative red, positive
  // green; null when there's no probe aggregate to show.
  const ringColor = $derived(
    ring === null ? null : ring >= 0 ? "var(--accent-green)" : "var(--accent-red)",
  );
</script>

<div
  class="node"
  class:active={onActivePath}
  class:focused
  class:dead
  class:streaming
  class:starred={node.starred}
  class:user={node.role === "user"}
  class:assistant={node.role === "assistant"}
  class:system={node.role === "system"}
  role="treeitem"
  aria-selected={onActivePath}
  tabindex={focused ? 0 : -1}
  data-node-id={node.id}
  {onclick}
  {oncontextmenu}
  onkeydown={(ev) => {
    if (ev.key === "Enter") {
      ev.preventDefault();
      onclick?.(ev as unknown as MouseEvent);
    }
  }}
>
  <span class="glyph" aria-hidden="true">{roleGlyph(node)}</span>
  {#if ringColor}
    <span class="ring" style="border-color: {ringColor}" aria-hidden="true"></span>
  {/if}
  {#if node.starred}
    <span class="star" title="starred" aria-hidden="true">★</span>
  {/if}
  <span class="preview">{preview}</span>
  {#if steerLabel}
    <span class="steer" title="steering delta from parent">{steerLabel}</span>
  {/if}
  {#if weightBadge != null}
    <span class="weight" title="mean chosen-token logprob (response span)">
      {weightBadge.toFixed(2)}
    </span>
  {/if}
  {#if node.notes}
    <span class="note-mark" title={node.notes} aria-label="has note">●</span>
  {/if}
</div>

<style>
  .node {
    display: grid;
    /* Columns: glyph · ring? · star? · preview (1fr) · steer? · weight? ·
     * note? .  Optional cells get ``auto`` slots; absent items collapse
     * to zero width. */
    grid-template-columns: auto auto auto 1fr auto auto auto;
    align-items: center;
    gap: var(--space-3);
    padding: var(--space-1) var(--space-3);
    border-radius: var(--radius);
    cursor: pointer;
    font: inherit;
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    line-height: 1.35;
    color: var(--fg-strong);
    background: transparent;
    transition: background var(--dur-fast) var(--ease-out);
    min-width: 0;
    user-select: none;
  }
  .node:hover {
    background: var(--bg-elev);
  }
  .node.focused {
    background: var(--accent-subtle);
    outline: 1px solid var(--accent-glow);
    outline-offset: -1px;
  }
  .node.active {
    font-weight: var(--weight-medium);
    color: var(--fg);
    background: var(--glass);
    border-radius: var(--radius-sm);
  }
  .node.dead {
    opacity: 0.3;
  }
  .node.dead:hover {
    opacity: 0.6;
  }
  .node.streaming {
    background: color-mix(in srgb, var(--live) 8%, transparent);
    box-shadow: var(--glow-live);
  }
  /* No stripes at all (cast model: roles carry no hue — identity is the
     glyph letter alone; the active path reads from the glass fill +
     weight, and the streaming node from its live glow). */
  .glyph {
    font-weight: var(--weight-bold);
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.09);
    color: var(--fg);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 9px;
    flex: none;
    text-transform: uppercase;
  }
  .preview {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    min-width: 0;
  }
  .star {
    color: var(--fg-dim);
    font-size: var(--text-xs);
  }
  /* Steering-delta chip — trailing, truncated so a long delta can't
   * blow out the row or collide with the preview text.  This is where
   * the edge label actually renders (LoomEdge only fetches/caches it),
   * so it wears the "edge label chip" look: a borderless glass-strong
   * pill, mono 2xs. */
  .steer {
    color: var(--fg-dim);
    font-family: var(--font-mono);
    font-size: var(--text-2xs);
    font-variant-numeric: tabular-nums;
    padding: 1px 6px;
    border-radius: var(--radius-sm);
    background: var(--glass-strong);
    max-width: 11ch;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .note-mark {
    color: var(--accent-purple);
    font-size: var(--text-2xs);
    line-height: 1;
  }
  /* Phase-5 hook: a thin colored ring around the role glyph keyed off
   * the highlight-probe's per-node aggregate reading.  Renders only
   * when ``ring`` prop is non-null. */
  .ring {
    width: 0.9em;
    height: 0.9em;
    border-radius: 50%;
    border: 1px solid transparent;
    display: inline-block;
  }
  /* Logit-pass: numeric ``mean_logprob`` badge.  Tabular-nums so the
     digits line up across siblings even when sort:surprise reorders
     them; subdued color so the badge reads as metadata, not content.
     Like .steer, no own background — inherits the row's highlight. */
  .weight {
    color: var(--fg-muted);
    font-size: var(--text-xs);
    font-variant-numeric: tabular-nums;
    padding: 0 var(--space-2);
  }
</style>
