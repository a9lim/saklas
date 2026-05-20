<script lang="ts">
  // Cross-branch diff drawer — phase 5.  Renders a side-by-side
  // word-level diff between two (or more) assistant nodes, plus the
  // sorted readings-delta table and the per-token alignment for the
  // primary pair.
  //
  // Drawer params: ``{node_ids: string[], parent_id?: string}``.
  // Surfaces:
  //   * 2 ids — single side-by-side comparison.
  //   * 3+ ids — N-column rendering, one column per id, diffs computed
  //     against the first id as anchor.

  import { onMount } from "svelte";
  import { apiTree, ApiError } from "../lib/api";
  import {
    clearNodeSelection,
    closeDrawer,
    drawerState,
    loomTree,
  } from "../lib/stores.svelte";
  import type {
    DiffReadingDeltaJSON,
    DiffTextSpanJSON,
    DiffTokenSpanJSON,
    JointLogprobsJSON,
    JointLogprobRowJSON,
    NodeDiffJSON,
  } from "../lib/types";

  // --------------------------------------------------------- props ---

  interface Params {
    node_ids?: string[];
    parent_id?: string | null;
  }

  let { params }: { params: unknown } = $props();
  const ids = $derived.by<string[]>(() => {
    const p = (params ?? drawerState.params ?? {}) as Params;
    return Array.isArray(p.node_ids) ? p.node_ids.filter(Boolean) : [];
  });
  const anchorId = $derived(ids[0] ?? null);

  // --------------------------------------------------------- diffs ---

  // For each non-anchor id, fetch the diff against the anchor.  When
  // there are N>=2 ids, we fetch N-1 diffs.  Single fetch when N=2
  // (the common case).  Re-runs whenever ``ids`` changes.

  type DiffOrError =
    | { kind: "ok"; diff: NodeDiffJSON }
    | { kind: "err"; message: string };

  /** Logit-pass Phase 5: joint-logprobs cache state per (anchor, other)
   *  pair.  Loads lazily after the main diff lands so the drawer feels
   *  responsive — the forward passes for cross-evaluation are slower
   *  than the text diff. */
  type JointOrError =
    | { kind: "loading" }
    | { kind: "ok"; data: JointLogprobsJSON }
    | { kind: "err"; message: string };

  let diffs: DiffOrError[] = $state([]);
  let joints: JointOrError[] = $state([]);
  let loading = $state(false);
  let hoveredAnchorIdx: number | null = $state(null);

  /** Layout helper — ``"unified"`` stacks the diff vertically for
   * narrow screens; ``"side-by-side"`` is the default. */
  let layout: "side-by-side" | "unified" = $state("side-by-side");
  let sortBy: "magnitude" | "name" = $state("magnitude");

  async function fetchAll(): Promise<void> {
    const list = ids;
    if (list.length < 2 || !anchorId) {
      diffs = [];
      joints = [];
      return;
    }
    loading = true;
    const out: DiffOrError[] = [];
    try {
      for (let i = 1; i < list.length; i++) {
        try {
          const r = await apiTree.diff(anchorId, list[i]);
          out.push({ kind: "ok", diff: r });
        } catch (e) {
          if (e instanceof ApiError) {
            const detail =
              e.body && typeof e.body === "object" && "detail" in (e.body as object)
                ? String((e.body as { detail: unknown }).detail)
                : e.message;
            out.push({ kind: "err", message: `${e.status}: ${detail}` });
          } else {
            out.push({
              kind: "err",
              message: e instanceof Error ? e.message : String(e),
            });
          }
        }
      }
      diffs = out;
    } finally {
      loading = false;
    }
    // Logit-pass Phase 5: fire the cross-evaluation forward passes in
    // parallel after the text diff is in place.  Loading state per
    // pair so the drawer can show "(crunching cross logprobs…)"
    // independently of the diff rendering.
    await fetchJoints();
  }

  /** Lazy fetch of joint logprobs for each (anchor, other) pair.
   *  Hits the session-lifetime cache on the server side; second open
   *  of the same pair is a memory lookup. */
  async function fetchJoints(): Promise<void> {
    const list = ids;
    if (list.length < 2 || !anchorId) {
      joints = [];
      return;
    }
    joints = list.slice(1).map(() => ({ kind: "loading" }) as JointOrError);
    const out: JointOrError[] = [];
    for (let i = 1; i < list.length; i++) {
      try {
        const r = await apiTree.jointLogprobs(anchorId, list[i]);
        out.push({ kind: "ok", data: r });
      } catch (e) {
        if (e instanceof ApiError) {
          const detail =
            e.body && typeof e.body === "object" && "detail" in (e.body as object)
              ? String((e.body as { detail: unknown }).detail)
              : e.message;
          out.push({ kind: "err", message: `${e.status}: ${detail}` });
        } else {
          out.push({
            kind: "err",
            message: e instanceof Error ? e.message : String(e),
          });
        }
      }
      // Update incrementally so each pair lands as soon as the server
      // returns — useful for N-way comparisons.
      joints = [...out, ...joints.slice(out.length)];
    }
    joints = out;
  }

  // Refetch whenever the id list changes.
  $effect(() => {
    void ids;
    void fetchAll();
  });

  onMount(() => {
    // Clear the multi-select once the drawer's data is in flight so the
    // sidebar's selection-bar disappears (user-feel: "ok, I've moved
    // them into the drawer").
    return () => {
      clearNodeSelection();
    };
  });

  // ----------------------------------------------- node previews ---

  function nodePreview(id: string): string {
    const n = loomTree.nodes.get(id);
    const t = (n?.text ?? "").replace(/\s+/g, " ").trim();
    if (!t) return "(empty)";
    return t.length > 80 ? t.slice(0, 80) + "…" : t;
  }

  // ---------------------------------------- readings sort + top-N --

  function sortedReadings(rs: DiffReadingDeltaJSON[]): DiffReadingDeltaJSON[] {
    const arr = [...rs];
    if (sortBy === "magnitude") {
      arr.sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta));
    } else {
      arr.sort((a, b) => a.name.localeCompare(b.name));
    }
    return arr;
  }

  function topKSet(rs: DiffReadingDeltaJSON[], k = 5): Set<string> {
    const sorted = [...rs].sort(
      (a, b) => Math.abs(b.delta) - Math.abs(a.delta),
    );
    return new Set(sorted.slice(0, k).map((r) => r.name));
  }

  // ------------------------------------- per-token hover alignment --

  // Decision (option a): render the per-pane token spans directly from
  // ``diff.per_token`` rather than re-splitting ``a_text`` / ``b_text``
  // on whitespace.  The server's :func:`per_token_diff` walks the real
  // model-tokenizer byte offsets, so its ``a_index`` / ``b_index`` keys
  // only line up with the spans it emitted — never with a client-side
  // whitespace split.  Iterating ``per_token`` gives us correct
  // tokenization for free, plus the hover-cross-highlight is exact
  // because each rendered span carries its own counterpart index.
  //
  // ``per_token`` may be empty (loaded transcripts that didn't persist
  // token sequences, or pre-v2.3 nodes); in that case we fall back to
  // the whitespace-split renderer so the diff stays visible — alignment
  // tooltips just won't fire on the fallback path.

  /** Anchor-side tokens (a_index >= 0) for one diff's per-token array.
   *  Returns ``[]`` when the diff carries no per-token data. */
  function panelTokensA(spans: DiffTokenSpanJSON[]): DiffTokenSpanJSON[] {
    return spans.filter((sp) => sp.a_index >= 0);
  }
  /** Other-side tokens (b_index >= 0). */
  function panelTokensB(spans: DiffTokenSpanJSON[]): DiffTokenSpanJSON[] {
    return spans.filter((sp) => sp.b_index >= 0);
  }

  function formatReadingDelta(r: DiffReadingDeltaJSON): string {
    const sign = r.delta >= 0 ? "+" : "";
    return `${sign}${r.delta.toFixed(3)}`;
  }

  function deltaColor(delta: number): string {
    if (delta === 0) return "var(--fg-muted)";
    return delta >= 0 ? "var(--accent-green)" : "var(--accent-red)";
  }

  function spanColor(state: DiffTextSpanJSON["state"]): string {
    if (state === "insert") return "rgba(126, 231, 135, 0.18)";
    if (state === "delete") return "rgba(248, 81, 73, 0.18)";
    return "transparent";
  }

  function tokensFor(text: string): string[] {
    // Fallback splitter for the per-pane render when ``per_token`` is
    // empty (transcript-loaded nodes without token sequences).  Word-
    // split has no alignment metadata, so cross-pane hover stays off
    // on this path.
    return text.split(/(\s+)/);
  }

  function spanTooltip(sp: DiffTokenSpanJSON): string {
    if (!sp.reading_deltas || sp.reading_deltas.length === 0) return "";
    return sp.reading_deltas
      .slice(0, 3)
      .map((r) => `${r.name} ${formatReadingDelta(r)}`)
      .join(" · ");
  }

  // ----------------------------------------- joint logprobs (Phase 5) --

  /** Format a logprob.  Null / non-finite render as ``—``. */
  function fmtLp(v: number | null | undefined): string {
    if (v == null || !Number.isFinite(v)) return "—";
    return v.toFixed(2);
  }

  /** Format a delta logprob (always signed) — useful when reading
   *  "did B give this token less / more weight than A did?". */
  function fmtDeltaLp(a: number | null, b: number | null): string {
    if (a == null || b == null || !Number.isFinite(a) || !Number.isFinite(b)) {
      return "—";
    }
    const d = a - b;
    const sign = d >= 0 ? "+" : "";
    return `${sign}${d.toFixed(2)}`;
  }

  /** Format the approx-KL value — small numbers get more precision so
   *  the leading digits remain readable. */
  function fmtKl(v: number | null): string {
    if (v == null || !Number.isFinite(v)) return "—";
    if (Math.abs(v) < 0.01) return v.toExponential(1);
    return v.toFixed(2);
  }

  /** Aligned-only filter — divergent rows have null cross-eval and are
   *  surfaced separately in the divergent-text view above.  Keeps the
   *  logprob table focused on the byte-aligned positions where the
   *  cross comparison is well-defined. */
  function alignedRows(rows: JointLogprobRowJSON[]): JointLogprobRowJSON[] {
    return rows.filter((r) => r.aligned);
  }

  // ------------------------------ siblings summary (Phase 6) ------------
  //
  // Per-sibling rollup for the loom "compare children" surface — answers
  // "did my α=0.5 shift the distribution at the contentious tokens, or
  // just the argmax?" without making the researcher scan the per-token
  // table.  Baseline = anchor (left-most node); one row per non-anchor
  // sibling shows ``mean_logprob`` (from the node), ``rank-1 %
  // unchanged`` (= 1 − n_rank1_changed / n_aligned), and the mean
  // ``approx_kl`` across aligned positions.  The anchor row carries
  // ``—`` for the comparison columns (it's not compared to itself).

  interface SiblingSummaryRow {
    nodeId: string;
    isAnchor: boolean;
    label: string;
    preview: string;
    meanLogprob: number | null;
    /** Fraction of aligned positions where the argmax did NOT change.
     *  Null for the anchor row. */
    rank1Unchanged: number | null;
    /** Mean approx-KL across aligned positions (the joint table's
     *  ``approx_kl`` column).  Null for the anchor row. */
    klMean: number | null;
    /** Joint-table loading state per row, so we can show a "(crunching)"
     *  shape next to the values instead of a blank row. */
    jointReady: boolean;
  }

  /** Build the summary table from currently-loaded state.  Pulls
   *  ``mean_logprob`` from the local tree cache and rolls up rank-1 /
   *  KL stats from each joint-logprobs response.  Re-runs reactively
   *  whenever ``ids`` or ``joints`` updates. */
  const siblingSummary = $derived.by<SiblingSummaryRow[]>(() => {
    if (ids.length < 2) return [];
    const out: SiblingSummaryRow[] = [];
    for (let i = 0; i < ids.length; i++) {
      const id = ids[i];
      const node = loomTree.nodes.get(id);
      const isAnchor = i === 0;
      const baseRow: SiblingSummaryRow = {
        nodeId: id,
        isAnchor,
        label: isAnchor
          ? "A"
          : ids.length === 2
            ? "B"
            : `B${i}`,
        preview: nodePreview(id),
        meanLogprob:
          node && typeof node.mean_logprob === "number"
            ? node.mean_logprob
            : null,
        rank1Unchanged: null,
        klMean: null,
        jointReady: isAnchor,
      };
      if (!isAnchor) {
        // Anchor takes index 0; ``joints`` is indexed by non-anchor
        // position so ``joints[i - 1]`` matches this row.
        const j = joints[i - 1];
        if (j) {
          if (j.kind === "ok") {
            const aligned = alignedRows(j.data.rows);
            if (aligned.length > 0) {
              baseRow.rank1Unchanged =
                1 - j.data.n_rank1_changed / aligned.length;
              let sum = 0;
              let n = 0;
              for (const r of aligned) {
                if (r.approx_kl != null && Number.isFinite(r.approx_kl)) {
                  sum += r.approx_kl;
                  n += 1;
                }
              }
              baseRow.klMean = n > 0 ? sum / n : null;
            }
            baseRow.jointReady = true;
          } else if (j.kind === "err") {
            baseRow.jointReady = true;
          }
        }
      }
      out.push(baseRow);
    }
    return out;
  });

  function fmtPct(v: number | null): string {
    if (v == null || !Number.isFinite(v)) return "—";
    return `${(v * 100).toFixed(0)}%`;
  }

  function fmtKlMean(v: number | null): string {
    if (v == null || !Number.isFinite(v)) return "—";
    if (Math.abs(v) < 0.01) return v.toExponential(1);
    return v.toFixed(3);
  }
</script>

<section class="drawer-shell" aria-label="Cross-branch diff drawer">
  <header class="header">
    <span class="title">compare branches</span>
    <div class="header-controls">
      <label class="header-ctl">
        <span>layout</span>
        <select bind:value={layout} aria-label="Layout">
          <option value="side-by-side">side-by-side</option>
          <option value="unified">unified</option>
        </select>
      </label>
      <label class="header-ctl">
        <span>sort by</span>
        <select bind:value={sortBy} aria-label="Sort readings by">
          <option value="magnitude">|Δ| desc</option>
          <option value="name">name</option>
        </select>
      </label>
    </div>
    <button type="button" class="close" aria-label="Close" onclick={closeDrawer}
      >✕</button>
  </header>

  <div class="body">
    {#if ids.length < 2}
      <p class="empty">
        Pick at least two assistant nodes via the sidebar's "select for
        compare" right-click action, or right-click a user node with
        ≥2 assistant children and pick "compare children".
      </p>
    {:else if loading && diffs.length === 0}
      <p class="empty">computing diff…</p>
    {:else}
      <!-- Column headers: anchor + each diff target. -->
      <div class="columns" class:unified={layout === "unified"}>
        <div class="col anchor-col">
          <header class="col-header">
            <code class="col-id">{anchorId?.slice(0, 12) ?? ""}</code>
            <span class="col-tag">anchor (A)</span>
          </header>
          <p class="col-preview">{nodePreview(anchorId ?? "")}</p>
        </div>
        {#each ids.slice(1) as otherId, otherIdx (otherId)}
          <div class="col">
            <header class="col-header">
              <code class="col-id">{otherId.slice(0, 12)}</code>
              <span class="col-tag">B{ids.length > 2 ? otherIdx + 1 : ""}</span>
            </header>
            <p class="col-preview">{nodePreview(otherId)}</p>
          </div>
        {/each}
      </div>

      <!-- Logit-pass Phase 6: per-sibling rollup summary.  Renders for
           both N=2 and N>2 modes — the table is more useful at N>2, but
           it is still a clean
           at-a-glance read for a single pair.  KL / rank-1 columns
           populate lazily as the joint-logprobs fetches complete. -->
      {#if siblingSummary.length > 0}
        <section class="siblings-summary">
          <header class="ss-header">
            <span class="ss-label">siblings · distributional rollup</span>
            <span class="ss-foot">vs. {siblingSummary[0]?.label ?? "A"} (baseline)</span>
          </header>
          <table class="ss-table">
            <thead>
              <tr>
                <th class="ss-tag">tag</th>
                <th class="ss-preview">preview</th>
                <th class="ss-num" title="mean chosen-token logprob over the response span">
                  mean lp
                </th>
                <th class="ss-num" title="fraction of aligned positions where argmax did not change">
                  rk1 unchanged
                </th>
                <th class="ss-num" title="mean top-K-truncated KL(baseline ∥ this) across aligned positions">
                  mean ≈KL
                </th>
              </tr>
            </thead>
            <tbody>
              {#each siblingSummary as row (row.nodeId)}
                <tr class:anchor={row.isAnchor}>
                  <td class="ss-tag">{row.label}</td>
                  <td class="ss-preview">{row.preview}</td>
                  <td class="ss-num">{fmtLp(row.meanLogprob)}</td>
                  <td class="ss-num">
                    {row.isAnchor
                      ? "—"
                      : row.jointReady
                        ? fmtPct(row.rank1Unchanged)
                        : "…"}
                  </td>
                  <td class="ss-num">
                    {row.isAnchor
                      ? "—"
                      : row.jointReady
                        ? fmtKlMean(row.klMean)
                        : "…"}
                  </td>
                </tr>
              {/each}
            </tbody>
          </table>
        </section>
      {/if}

      {#each diffs as result, diffIdx (diffIdx)}
        {#if result.kind === "err"}
          <p class="error" role="alert">{result.message}</p>
        {:else}
          {@const d = result.diff}
          {@const topDeltas = topKSet(d.readings, 5)}
          {@const sortedRs = sortedReadings(d.readings)}
          {@const joint = joints[diffIdx]}

          <section class="diff-block">
            <header class="diff-header">
              <span class="diff-title">
                {ids.length === 2 ? "A vs B" : `A vs B${diffIdx + 1}`}
              </span>
              {#if d.parent_applied_steering !== null || d.steering_delta}
                <code class="recipe-delta" title="steering delta A → B">
                  Δ steering: {d.steering_delta || "(none)"}
                </code>
              {/if}
            </header>

            <!-- Text diff: side-by-side or unified. -->
            {#if layout === "side-by-side"}
              {@const tokensA = panelTokensA(d.per_token)}
              {@const tokensB = panelTokensB(d.per_token)}
              <div class="text-grid">
                <div class="text-pane">
                  <span class="pane-label">A</span>
                  <div class="text-body">
                    {#if tokensA.length > 0}
                      {#each tokensA as sp (`a-${sp.a_index}`)}
                        <!-- svelte-ignore a11y_no_static_element_interactions -->
                        <span
                          class="tok"
                          class:highlight-anchor={hoveredAnchorIdx === sp.a_index}
                          title={spanTooltip(sp)}
                          onmouseenter={() => (hoveredAnchorIdx = sp.a_index)}
                          onmouseleave={() => (hoveredAnchorIdx = null)}
                        >{sp.a_text}</span>
                      {/each}
                    {:else}
                      {#each tokensFor(d.a_text) as part, idx (idx)}
                        <span class="tok">{part}</span>
                      {/each}
                    {/if}
                  </div>
                </div>
                <div class="text-pane">
                  <span class="pane-label">B{ids.length > 2 ? diffIdx + 1 : ""}</span>
                  <div class="text-body">
                    {#if tokensB.length > 0}
                      {#each tokensB as sp (`b-${sp.b_index}`)}
                        {@const matched =
                          sp.a_index >= 0 &&
                          hoveredAnchorIdx !== null &&
                          sp.a_index === hoveredAnchorIdx}
                        <span
                          class="tok"
                          class:highlight-target={matched}
                          title={spanTooltip(sp)}
                        >{sp.b_text}</span>
                      {/each}
                    {:else}
                      {#each tokensFor(d.b_text) as part, idx (idx)}
                        <span class="tok">{part}</span>
                      {/each}
                    {/if}
                  </div>
                </div>
              </div>
            {:else}
              <div class="unified-body">
                {#each d.text as span, idx (idx)}
                  <span
                    class="tok-span"
                    class:span-equal={span.state === "equal"}
                    class:span-insert={span.state === "insert"}
                    class:span-delete={span.state === "delete"}
                    style={`background-color: ${spanColor(span.state)}`}
                  >
                    {#if span.state === "insert"}<span class="span-sign">+</span
                      >{:else if span.state === "delete"}<span class="span-sign"
                        >−</span
                      >{/if}{span.text}{" "}
                  </span>
                {/each}
              </div>
            {/if}

            <!-- Logit-pass Phase 5: per-aligned-token cross-evaluation
                 table.  Lazy fetch in parallel with the diff; loading /
                 error states surface inline.  ``joint`` is declared at
                 the top of the {:else} block (Svelte requires @const at
                 the start of a parent block). -->
            {#if joint}
              {#if joint.kind === "loading"}
                <p class="dim small">
                  computing cross-branch logprobs…
                </p>
              {:else if joint.kind === "err"}
                <p class="error small" role="alert">
                  joint logprobs unavailable: {joint.message}
                </p>
              {:else}
                {@const aligned = alignedRows(joint.data.rows)}
                {#if aligned.length > 0}
                  <div class="joint-table">
                    <header class="joint-header">
                      <span class="joint-label">cross-branch logprobs</span>
                      <span class="joint-summary">
                        rank-1 changed at <strong>
                          {joint.data.n_rank1_changed}
                        </strong> of {aligned.length} aligned positions
                      </span>
                    </header>
                    <table class="lp-table">
                      <thead>
                        <tr>
                          <th class="lp-pos">pos</th>
                          <th class="lp-tok">token</th>
                          <th class="lp-num">lp(A)</th>
                          <th class="lp-num">lp(B)</th>
                          <th class="lp-num" title="lp(A's token under B's distribution) − lp(A's own)">
                            Δ lp(A)
                          </th>
                          <th class="lp-num" title="approx KL(A ∥ B), top-K truncated">
                            ≈KL
                          </th>
                          <th class="lp-flag" title="argmax differs">rk1Δ</th>
                        </tr>
                      </thead>
                      <tbody>
                        {#each aligned as row (`${row.a_index}-${row.b_index}`)}
                          <tr class:rank-flip={row.rank_changed}>
                            <td class="lp-pos">{row.a_index}</td>
                            <td class="lp-tok">
                              <code>{JSON.stringify(row.a_text)}</code>
                            </td>
                            <td class="lp-num">{fmtLp(row.lp_a_in_a)}</td>
                            <td class="lp-num">{fmtLp(row.lp_b_in_b)}</td>
                            <td class="lp-num">
                              {fmtDeltaLp(row.lp_a_in_b, row.lp_a_in_a)}
                            </td>
                            <td class="lp-num">{fmtKl(row.approx_kl)}</td>
                            <td class="lp-flag">{row.rank_changed ? "●" : ""}</td>
                          </tr>
                        {/each}
                      </tbody>
                    </table>
                    <p class="joint-foot">
                      lp: chosen-token logprob.  Δ lp(A): how B would have
                      scored A's chosen token, minus what A actually gave
                      it (negative = B disagreed).  ≈KL: top-{32}-truncated
                      KL(A ∥ B), approximate signal, not measurement.
                    </p>
                  </div>
                {:else}
                  <p class="dim small">
                    no byte-aligned assistant tokens between these
                    branches, so cross-evaluation has nothing to score.
                  </p>
                {/if}
              {/if}
            {/if}

            <!-- Readings delta table. -->
            {#if sortedRs.length > 0}
              <div class="readings-table">
                <span class="readings-label">readings Δ (B − A)</span>
                <div class="readings-grid">
                  {#each sortedRs as r (r.name)}
                    {@const top = topDeltas.has(r.name)}
                    <div class="reading-row" class:top-delta={top}>
                      <span class="r-name">{r.name}</span>
                      <span class="r-vals">
                        <span class="r-side">{r.a_value.toFixed(3)}</span>
                        <span class="r-arrow">→</span>
                        <span class="r-side">{r.b_value.toFixed(3)}</span>
                      </span>
                      <div class="r-bar-track">
                        <div
                          class="r-bar-fill"
                          style={`width: ${Math.min(100, Math.abs(r.delta) * 100)}%; background: ${deltaColor(r.delta)}`}
                        ></div>
                      </div>
                      <span class="r-delta" style={`color: ${deltaColor(r.delta)}`}>
                        {formatReadingDelta(r)}
                      </span>
                    </div>
                  {/each}
                </div>
              </div>
            {:else}
              <p class="dim small">
                no readings recorded; either node has empty
                ``aggregate_readings``.
              </p>
            {/if}
          </section>
        {/if}
      {/each}
    {/if}
  </div>

  <footer class="footer">
    <button type="button" class="btn" onclick={closeDrawer}>close</button>
  </footer>
</section>

<style>
  .drawer-shell {
    display: flex;
    flex-direction: column;
    height: 100%;
    min-height: 0;
    color: var(--fg);
    font-family: var(--font-mono);
    font-size: var(--text);
  }
  .header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-4);
    padding: var(--space-5) var(--space-6);
    border-bottom: 1px solid var(--border);
  }
  .title {
    color: var(--accent-blue);
    text-transform: lowercase;
    letter-spacing: 0;
    flex: 0 0 auto;
  }
  .header-controls {
    display: flex;
    gap: var(--space-4);
    flex: 1 1 auto;
    justify-content: center;
    color: var(--fg-muted);
    font-size: var(--text-sm);
  }
  .header-ctl {
    display: inline-flex;
    align-items: center;
    gap: var(--space-2);
  }
  .header-ctl select {
    background: var(--bg-alt);
    color: var(--fg-strong);
    border: 1px solid var(--border);
    padding: var(--space-1) var(--space-2);
    font: inherit;
    font-family: var(--font-mono);
    font-size: var(--text-xs);
  }
  .close {
    background: transparent;
    border: 0;
    color: var(--fg-dim);
    cursor: pointer;
    padding: var(--space-2) var(--space-2);
    font-size: var(--text);
    line-height: 1;
  }
  .close:hover {
    color: var(--accent-red);
  }

  .body {
    flex: 1 1 auto;
    overflow-y: auto;
    padding: var(--space-6);
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
    min-height: 0;
  }
  .empty {
    color: var(--fg-muted);
    font-size: var(--text-sm);
    margin: 0;
  }
  .error {
    color: var(--accent-red);
    font-size: var(--text-sm);
    margin: 0;
  }

  .columns {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: var(--space-4);
  }
  .col {
    background: var(--bg-deep);
    border: 1px solid var(--border);
    padding: var(--space-3) var(--space-4);
  }
  .col-header {
    display: flex;
    justify-content: space-between;
    color: var(--fg-muted);
    font-size: var(--text-xs);
    margin-bottom: var(--space-2);
  }
  .col-id {
    color: var(--accent-yellow);
  }
  .col-tag {
    color: var(--accent-blue);
  }
  .col-preview {
    margin: 0;
    color: var(--fg);
    font-size: var(--text-sm);
    line-height: 1.4;
    overflow: hidden;
    text-overflow: ellipsis;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    line-clamp: 3;
    -webkit-box-orient: vertical;
  }

  .diff-block {
    background: var(--bg-deep);
    border: 1px solid var(--border);
    padding: var(--space-4) var(--space-4);
    display: flex;
    flex-direction: column;
    gap: var(--space-4);
  }
  .diff-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-4);
    flex-wrap: wrap;
  }
  .diff-title {
    color: var(--accent-blue);
    font-size: var(--text-sm);
  }
  .recipe-delta {
    color: var(--accent-yellow);
    font-size: var(--text-xs);
    background: var(--bg-elev);
    padding: var(--space-1) var(--space-2);
    border-radius: var(--radius);
  }

  .text-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: var(--space-3);
  }
  .columns.unified,
  .columns.unified > .col {
    grid-template-columns: 1fr;
  }
  .text-pane {
    background: var(--bg-alt);
    border: 1px solid var(--border);
    padding: var(--space-2) var(--space-3);
    min-height: 5em;
    max-height: 30em;
    overflow-y: auto;
  }
  .pane-label {
    color: var(--accent-blue);
    font-size: var(--text-xs);
    text-transform: lowercase;
    letter-spacing: 0;
    display: block;
    margin-bottom: var(--space-1);
  }
  .text-body {
    color: var(--fg);
    line-height: 1.5;
    white-space: pre-wrap;
    word-break: break-word;
    font-size: var(--text-sm);
  }
  .tok {
    cursor: default;
    transition: background-color var(--dur-fast) var(--ease-out);
  }
  .tok:hover {
    background: rgba(72, 138, 203, 0.1);
  }
  .tok.highlight-anchor {
    background: rgba(72, 138, 203, 0.22);
  }
  .tok.highlight-target {
    background: rgba(210, 153, 34, 0.28);
  }

  .unified-body {
    background: var(--bg-alt);
    border: 1px solid var(--border);
    padding: var(--space-3) var(--space-4);
    line-height: 1.5;
    font-size: var(--text-sm);
    word-break: break-word;
    white-space: normal;
  }
  .tok-span {
    padding: 0 0.05em;
  }
  .span-sign {
    margin-right: var(--space-1);
    font-weight: var(--weight-medium);
  }
  .span-insert .span-sign {
    color: var(--accent-green);
  }
  .span-delete .span-sign {
    color: var(--accent-red);
  }
  .span-equal {
    color: var(--fg-strong);
  }

  /* Logit-pass Phase 6: per-sibling distributional rollup. */
  .siblings-summary {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    background: var(--bg-alt);
    border: 1px solid var(--border);
    padding: var(--space-3) var(--space-4);
    margin: 0;
  }
  .ss-header {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: var(--space-4);
  }
  .ss-label {
    color: var(--accent-blue);
    text-transform: lowercase;
    letter-spacing: 0;
    font-size: var(--text-xs);
  }
  .ss-foot {
    color: var(--fg-muted);
    font-size: var(--text-xs);
  }
  .ss-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-size: var(--text-xs);
    font-variant-numeric: tabular-nums;
  }
  .ss-table th,
  .ss-table td {
    padding: var(--space-2) var(--space-3);
    border-bottom: 1px solid var(--border);
    text-align: left;
  }
  .ss-table thead th {
    color: var(--fg-muted);
    text-transform: uppercase;
    letter-spacing: 0;
    font-weight: var(--weight-normal);
  }
  .ss-table td.ss-num,
  .ss-table th.ss-num {
    text-align: right;
    white-space: nowrap;
    width: 6em;
  }
  .ss-table td.ss-tag,
  .ss-table th.ss-tag {
    width: 3em;
    color: var(--fg-dim);
  }
  .ss-table td.ss-preview {
    color: var(--fg);
    max-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .ss-table tr.anchor td {
    background: var(--accent-subtle);
  }
  .ss-table tr.anchor td.ss-tag {
    color: var(--accent-blue);
    font-weight: var(--weight-medium);
  }

  /* Logit-pass Phase 5: cross-evaluation table. */
  .joint-table {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    margin: var(--space-2) 0 var(--space-3) 0;
  }
  .joint-header {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: var(--space-4);
  }
  .joint-label {
    color: var(--fg-muted);
    font-size: var(--text-xs);
    text-transform: lowercase;
    letter-spacing: 0;
  }
  .joint-summary {
    color: var(--fg-dim);
    font-size: var(--text-xs);
  }
  .joint-summary strong {
    color: var(--fg-strong);
  }
  .lp-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-variant-numeric: tabular-nums;
    font-size: var(--text-xs);
    background: var(--bg-alt);
    border: 1px solid var(--border);
  }
  .lp-table th,
  .lp-table td {
    padding: var(--space-1) var(--space-3);
    border-bottom: 1px solid var(--border);
    text-align: left;
  }
  .lp-table thead th {
    color: var(--fg-muted);
    font-weight: var(--weight-normal);
    text-transform: uppercase;
    letter-spacing: 0;
    background: var(--bg-deep);
  }
  .lp-table td.lp-num,
  .lp-table th.lp-num {
    text-align: right;
    width: 4.5em;
  }
  .lp-table td.lp-pos,
  .lp-table th.lp-pos {
    text-align: right;
    width: 2.5em;
    color: var(--fg-dim);
  }
  .lp-table td.lp-flag,
  .lp-table th.lp-flag {
    text-align: center;
    width: 2em;
  }
  .lp-table td.lp-tok code {
    color: var(--fg-strong);
    background: transparent;
    word-break: break-all;
  }
  /* Rank-1 flips read as the highest-signal rows — give them a soft
     yellow tint, mirroring the readings table's top-delta affordance. */
  .lp-table tr.rank-flip td {
    background: rgba(240, 200, 88, 0.10);
  }
  .lp-table tr.rank-flip .lp-flag {
    color: var(--accent-yellow);
  }
  .joint-foot {
    color: var(--fg-muted);
    font-size: var(--text-xs);
    line-height: 1.4;
    margin: 0;
  }

  .readings-table {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }
  .readings-label {
    color: var(--fg-muted);
    font-size: var(--text-xs);
    text-transform: lowercase;
    letter-spacing: 0;
  }
  .readings-grid {
    display: flex;
    flex-direction: column;
    gap: var(--space-1);
  }
  .reading-row {
    display: grid;
    grid-template-columns: minmax(8em, 12em) minmax(6em, 9em) 1fr 4em;
    align-items: center;
    gap: var(--space-2);
    font-size: var(--text-sm);
  }
  .reading-row.top-delta .r-name {
    color: var(--accent-yellow);
    font-weight: var(--weight-medium);
  }
  .r-name {
    color: var(--fg-strong);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .r-vals {
    color: var(--fg-muted);
    font-size: var(--text-xs);
    display: inline-flex;
    gap: var(--space-2);
    align-items: center;
  }
  .r-arrow {
    color: var(--fg-subtle);
  }
  .r-bar-track {
    background: var(--bg-elev);
    height: 6px;
    border-radius: var(--radius);
    position: relative;
    overflow: hidden;
  }
  .r-bar-fill {
    height: 100%;
  }
  .r-delta {
    text-align: right;
    font-variant-numeric: tabular-nums;
  }
  .dim {
    color: var(--fg-muted);
  }
  .small {
    font-size: var(--text-sm);
  }

  .footer {
    display: flex;
    justify-content: flex-end;
    gap: var(--space-3);
    padding: var(--space-5) var(--space-6);
    border-top: 1px solid var(--border);
  }
  .btn {
    background: var(--bg-alt);
    color: var(--fg-strong);
    border: 1px solid var(--border);
    padding: var(--space-2) var(--space-5);
    font: inherit;
    font-family: var(--font-mono);
    cursor: pointer;
  }
  .btn:hover {
    background: var(--bg-elev);
    border-color: var(--fg-muted);
  }
</style>
