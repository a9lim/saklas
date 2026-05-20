<script lang="ts">
  // Help drawer — keyboard shortcut reference + steering grammar cheat
  // sheet.  Pure-static content; closes via the header X or any backdrop
  // click handled at the App level.

  import { closeDrawer } from "../lib/stores.svelte";

  let _drawerProps: { params?: unknown } = $props();
  $effect(() => {
    void _drawerProps.params;
  });

  // Pre-derive the platform-appropriate modifier label so the shortcut
  // hints don't lie on Linux or Windows.  ``navigator`` may be undefined
  // in non-browser test environments — fall back to ``Cmd`` to match the
  // Mac-first development stance documented in CLAUDE.md.
  const modKey =
    typeof navigator !== "undefined" &&
    /Mac|iPhone|iPad|iPod/.test(navigator.platform)
      ? "Cmd"
      : "Ctrl";
</script>

<section class="drawer-shell" aria-label="Help drawer">
  <header class="header">
    <span class="title">help</span>
    <button type="button" class="close" aria-label="Close" onclick={closeDrawer}
      >✕</button
    >
  </header>

  <div class="body">
    <section class="block">
      <h3>keyboard shortcuts</h3>
      <table class="kb">
        <tbody>
          <tr>
            <td><kbd>Esc</kbd></td>
            <td>stop in-flight generation; close drawer when idle</td>
          </tr>
          <tr>
            <td><kbd>Enter</kbd></td>
            <td>send message</td>
          </tr>
          <tr>
            <td><kbd>Shift</kbd> + <kbd>Enter</kbd></td>
            <td>newline in chat input</td>
          </tr>
          <tr>
            <td><kbd>{modKey}</kbd> + <kbd>Enter</kbd></td>
            <td>send message (alternate)</td>
          </tr>
          <tr>
            <td>
              <kbd>{modKey}</kbd> + <kbd>Shift</kbd> + <kbd>R</kbd>
            </td>
            <td>regenerate last response (reserved, not yet wired)</td>
          </tr>
          <tr>
            <td>click token</td>
            <td>open per-layer × per-probe drilldown for that token</td>
          </tr>
        </tbody>
      </table>
    </section>

    <section class="block">
      <h3>loom tree</h3>
      <p class="prose">
        the single-letter keys fire while the loom sidebar is focused;
        the <kbd>{modKey}</kbd>-combos act on the active node from
        anywhere.
      </p>
      <table class="kb">
        <tbody>
          <tr>
            <td><kbd>j</kbd> / <kbd>k</kbd></td>
            <td>focus the previous / next sibling</td>
          </tr>
          <tr>
            <td><kbd>h</kbd> / <kbd>l</kbd></td>
            <td>focus the parent / first child</td>
          </tr>
          <tr>
            <td><kbd>Enter</kbd></td>
            <td>activate the focused node</td>
          </tr>
          <tr>
            <td><kbd>s</kbd></td>
            <td>star / unstar the focused node</td>
          </tr>
          <tr>
            <td><kbd>n</kbd></td>
            <td>add or edit a note</td>
          </tr>
          <tr>
            <td><kbd>/</kbd></td>
            <td>search node text</td>
          </tr>
          <tr>
            <td><kbd>{modKey}</kbd> + <kbd>R</kbd></td>
            <td>regenerate the active node</td>
          </tr>
          <tr>
            <td><kbd>{modKey}</kbd> + <kbd>E</kbd></td>
            <td>edit the active node's text</td>
          </tr>
          <tr>
            <td><kbd>{modKey}</kbd> + <kbd>B</kbd></td>
            <td>branch a new sibling</td>
          </tr>
          <tr>
            <td><kbd>{modKey}</kbd> + <kbd>N</kbd></td>
            <td>navigate by node-id prefix</td>
          </tr>
          <tr>
            <td><kbd>{modKey}</kbd> + <kbd>D</kbd></td>
            <td>delete the active node's subtree</td>
          </tr>
        </tbody>
      </table>
    </section>

    <section class="block">
      <h3>steering grammar</h3>
      <p class="prose">
        the same grammar speaks Python, YAML, the chat input, and this UI's
        rack.  Server-side parser lives in
        <code>saklas.core.steering_expr</code>.
      </p>
      <pre class="grammar">{`expr     := term (("+" | "-") term)*
term     := [coeff "*"?] ["!"] selector ["@" trigger]
selector := atom (("~" | "|") atom)?
atom     := [ns "/"] NAME ["." NAME] [":" variant]
trigger  := before | after | both | thinking | response
            | prompt   (alias of before)
            | generated (alias of response)
variant  := raw | sae | sae-<release>
`}</pre>

      <p class="prose">examples</p>
      <pre class="grammar">{`0.3 honest                   # plain additive, default coeff = 0.5
0.4 warm@after               # active only after </think>
-0.5 wolf                    # bare pole resolves to deer.wolf @ -0.5
0.6 honest:sae               # pull from the SAE-feature-space tensor
0.5 honest|sycophantic       # remove shared component with sycophantic
0.5 honest~confident         # keep the shared component, drop the rest
!sycophantic                 # mean-ablate (coeff = 1.0 fully replaces)
0.3 a + 0.5 b@thinking - 0.2 c|d   # compose
`}</pre>

      <table class="kb">
        <tbody>
          <tr>
            <td><code>+</code> / <code>-</code></td>
            <td>add / subtract terms</td>
          </tr>
          <tr>
            <td><code>*</code></td>
            <td>attach explicit coefficient (optional)</td>
          </tr>
          <tr>
            <td><code>!</code></td>
            <td>mean-ablate; does not compose with projection</td>
          </tr>
          <tr>
            <td><code>~</code></td>
            <td>project onto direction (keep shared component)</td>
          </tr>
          <tr>
            <td><code>|</code></td>
            <td>project orthogonal (remove shared component)</td>
          </tr>
          <tr>
            <td><code>@trigger</code></td>
            <td>per-term trigger override</td>
          </tr>
          <tr>
            <td><code>:variant</code></td>
            <td>route to SAE tensor (raw is default)</td>
          </tr>
        </tbody>
      </table>
    </section>

    <section class="block">
      <h3>tips</h3>
      <ul class="prose-list">
        <li>
          bare pole names (<code>wolf</code>) resolve cross-namespace and
          flip sign automatically; <code>ns/concept</code> disambiguates.
        </li>
        <li>
          dotted bipolar names (<code>happy.sad</code>) are first-class;
          <code>.</code> and <code>_</code> are the only allowed
          punctuation in concept identifiers.
        </li>
        <li>
          mid-generation rack edits queue as pending actions and apply on
          the next <code>done</code> event (or instantly with apply-now).
        </li>
      </ul>
    </section>
  </div>

  <footer class="footer">
    <button type="button" class="btn primary" onclick={closeDrawer}>
      close
    </button>
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
    padding: var(--space-6);
    border-bottom: 1px solid var(--border);
  }
  .title {
    color: var(--accent-blue);
    text-transform: lowercase;
    letter-spacing: 0;
  }
  .close {
    background: transparent;
    border: 0;
    color: var(--fg-dim);
    cursor: pointer;
    padding: var(--space-2) var(--space-3);
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
    gap: var(--space-5);
    min-height: 0;
  }
  .block h3 {
    margin: 0 0 var(--space-3);
    color: var(--accent-green);
    font-size: var(--text);
    text-transform: lowercase;
    letter-spacing: 0;
  }
  .kb {
    border-collapse: collapse;
    width: 100%;
    color: var(--fg-strong);
    font-size: var(--text-sm);
  }
  .kb td {
    padding: var(--space-2) var(--space-3);
    vertical-align: top;
    border-bottom: 1px solid var(--border);
  }
  .kb td:first-child {
    color: var(--fg-dim);
    white-space: nowrap;
    width: 9em;
  }
  kbd {
    background: var(--bg-elev);
    border: 1px solid var(--border);
    color: var(--fg-strong);
    padding: 0 var(--space-2);
    border-radius: var(--radius);
    font-family: inherit;
    font-size: var(--text-xs);
  }
  code {
    color: var(--accent-blue);
  }
  .grammar {
    background: var(--bg-deep);
    border: 1px solid var(--border);
    padding: var(--space-3) var(--space-4);
    margin: var(--space-3) 0;
    color: var(--fg-strong);
    font-size: var(--text-sm);
    line-height: 1.4;
    overflow-x: auto;
    white-space: pre;
  }
  .prose {
    margin: var(--space-2) 0;
    color: var(--fg-dim);
    font-size: var(--text-sm);
    line-height: 1.4;
  }
  .prose-list {
    margin: var(--space-2) 0 0;
    padding-left: 1.2em;
    color: var(--fg-dim);
    font-size: var(--text-sm);
    line-height: 1.5;
  }
  .footer {
    display: flex;
    justify-content: flex-end;
    gap: var(--space-3);
    padding: var(--space-6);
    border-top: 1px solid var(--border);
  }
  .btn {
    background: var(--bg-alt);
    color: var(--fg-strong);
    border: 1px solid var(--border);
    padding: var(--space-3) var(--space-5);
    font: inherit;
    font-family: var(--font-mono);
    cursor: pointer;
  }
  .btn.primary {
    background: var(--accent);
    color: var(--text-on-accent);
    border-color: var(--accent);
  }
  .btn.primary:hover {
    background: var(--accent-light);
    border-color: var(--accent-light);
  }
</style>
