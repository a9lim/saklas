<script lang="ts">
  // Connector between a parent node and a child node.  Phase 5: when
  // ``parentId`` and ``childId`` are provided, fetch the canonical
  // steering-delta label lazily from the server (cached in the store
  // per `parent|child` key) and render it inline.

  import { fetchEdgeLabel, loomTree } from "../../lib/stores.svelte";

  interface Props {
    /** Active-path membership — bold the line so the user can trace
     *  the chosen branch downward. */
    active?: boolean;
    /** Dead branch — drop opacity in lockstep with the child. */
    dead?: boolean;
    /** Steering-delta label endpoints — when both are set the
     *  component fetches the label from the server lazily and
     *  caches the result. */
    parentId?: string | null;
    childId?: string | null;
    /** Manual label override — phase-3 callers can pass a static
     *  string; phase-5 fetches lazily when this is unset. */
    label?: string | null;
    /** Logit-pass: child's per-turn ``mean_logprob``.  Null when capture
     *  wasn't live; the edge then renders flat. */
    weight?: number | null;
  }

  let {
    active = false,
    dead = false,
    parentId = null,
    childId = null,
    label = null,
    weight = null,
  }: Props = $props();

  /** Logit-pass: map ``mean_logprob`` (≤ 0) to a [0, 1] surprise
   *  intensity for edge stroke-width / opacity scaling.
   *
   *  surprise = 1 - exp(logprob) = 1 - probability
   *           # logprob → 0 ⇒ 0; logprob → -∞ ⇒ 1
   *
   *  Returns ``null`` when the input is missing — caller then renders
   *  flat (the unweighted shape). */
  const intensity = $derived.by<number | null>(() => {
    if (weight == null || !Number.isFinite(weight) || weight > 0) return null;
    return 1 - Math.exp(weight);
  });

  /** Scale the line's CSS variables.  Width grows from 1px → 3px,
   *  opacity from 0.35 → 1.  Active edges still get the green-accent
   *  override below from the class; weighting just modulates magnitude. */
  const lineStyle = $derived.by<string>(() => {
    if (intensity === null) return "";
    const width = 1 + 2 * intensity;
    const opacity = 0.35 + 0.65 * intensity;
    return `width: ${width.toFixed(2)}px; opacity: ${opacity.toFixed(3)};`;
  });

  // Lazy fetch on mount + whenever the tree revision changes (cache
  // gets invalidated, so we need to re-request).  Skip when the
  // caller already passed an explicit ``label`` or when either
  // endpoint is missing.
  //
  // The fetched steering-delta label is *rendered by LoomNode* (as a
  // trailing chip in the node row) — not here.  An absolutely-positioned
  // label on this 1ch-wide edge column overlapped the node text.  This
  // component still owns the fetch so the cache stays populated; the
  // sidebar reads ``edgeLabelCache`` and forwards the label to the node.
  $effect(() => {
    void loomTree.rev;
    if (label !== null) return;
    if (!parentId || !childId) return;
    // Fetch for every edge: the steering-delta is a property of the
    // edge into this child, not of the sibling set.  A steered node
    // with no siblings still has a delta to show — the old ``≥2
    // children`` guard hid the chip until a sibling was branched off.
    fetchEdgeLabel(parentId, childId);
  });
</script>

<div
  class="edge"
  class:active
  class:dead
  class:weighted={intensity !== null}
  aria-hidden="true"
>
  <span class="line" style={lineStyle}></span>
</div>

<style>
  .edge {
    position: relative;
    width: 1ch;
    flex: 0 0 auto;
    display: flex;
    align-items: stretch;
    justify-content: center;
    pointer-events: none;
  }
  .line {
    width: 1px;
    background: var(--border);
    height: 100%;
  }
  .edge.active .line {
    background: var(--accent-green);
    width: 2px;
  }
  .edge.dead {
    opacity: 0.3;
  }
</style>
