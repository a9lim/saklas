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
      <h3>shortcuts</h3>
      <table class="kb">
        <tbody>
          <tr>
            <td><kbd>Esc</kbd></td>
            <td>stop / close</td>
          </tr>
          <tr>
            <td><kbd>Enter</kbd></td>
            <td>send</td>
          </tr>
          <tr>
            <td><kbd>Shift</kbd> + <kbd>Enter</kbd></td>
            <td>newline</td>
          </tr>
          <tr>
            <td><kbd>{modKey}</kbd> + <kbd>Enter</kbd></td>
            <td>commit</td>
          </tr>
          <tr>
            <td>click token</td>
            <td>token details</td>
          </tr>
        </tbody>
      </table>
    </section>

    <section class="block">
      <h3>loom tree</h3>
      <table class="kb">
        <tbody>
          <tr>
            <td><kbd>j</kbd> / <kbd>k</kbd></td>
            <td>previous / next sibling</td>
          </tr>
          <tr>
            <td><kbd>h</kbd> / <kbd>l</kbd></td>
            <td>parent / first child</td>
          </tr>
          <tr>
            <td><kbd>Enter</kbd></td>
            <td>activate</td>
          </tr>
          <tr>
            <td><kbd>s</kbd></td>
            <td>star</td>
          </tr>
          <tr>
            <td><kbd>n</kbd></td>
            <td>note</td>
          </tr>
          <tr>
            <td><kbd>/</kbd></td>
            <td>search</td>
          </tr>
          <tr>
            <td><kbd>{modKey}</kbd> + <kbd>R</kbd></td>
            <td>regenerate</td>
          </tr>
          <tr>
            <td><kbd>{modKey}</kbd> + <kbd>E</kbd></td>
            <td>edit</td>
          </tr>
          <tr>
            <td><kbd>{modKey}</kbd> + <kbd>B</kbd></td>
            <td>branch</td>
          </tr>
          <tr>
            <td><kbd>{modKey}</kbd> + <kbd>N</kbd></td>
            <td>navigate</td>
          </tr>
          <tr>
            <td><kbd>{modKey}</kbd> + <kbd>D</kbd></td>
            <td>delete subtree</td>
          </tr>
        </tbody>
      </table>
    </section>

    <section class="block">
      <h3>steering grammar</h3>
      <pre class="grammar">{`expr      := term (("+" | "-") term)*
term      := [coeff "*"?] ["!"] selector ["@" trigger]
selector  := atom (("~" | "|") atom | "%" position)?
position  := signed_num ("," signed_num)* | label
atom      := [ns "/"] NAME ["." NAME] [":" variant]
trigger   := before | after | both | thinking | response
             | prompt | generated | when:<probe><op><num>
variant   := raw | sae | sae-<release>
             | role | role-<name> | from | from-<source>
`}</pre>

      <h3>examples</h3>
      <pre class="grammar">{`0.3 honest                   # plain additive, default coeff = 0.5
0.4 warm@after               # active only after </think>
-0.5 wolf                    # bare pole resolves to deer.wolf @ -0.5
0.6 honest:sae               # pull from the SAE-feature-space tensor
0.6 honest:role-pirate       # role-augmented tensor
0.6 honest:from-gemma_4_31b  # transferred tensor
0.5 honest|sycophantic       # remove shared component with sycophantic
0.5 honest~confident         # keep the shared component, drop the rest
0.5 personas%pirate          # steer to a manifold node label
0.4 circumplex%0.6,0.2@response # steer to manifold coordinates
!sycophantic                 # mean-ablate (coeff = 1.0 fully replaces)
0.3 a + 0.5 b@thinking - 0.2 c|d   # compose
`}</pre>

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
    font-family: var(--font-ui);
    font-size: var(--text);
  }
  .header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: var(--space-5) var(--space-6);
  }
  .title {
    color: var(--accent);
    text-transform: lowercase;
    letter-spacing: 0;
    font-size: var(--text-md);
    font-weight: var(--weight-medium);
  }
  .close {
    background: var(--glass);
    color: var(--fg-muted);
    border: 1px solid transparent;
    border-radius: 50%;
    width: 26px;
    height: 26px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font: inherit;
    font-size: var(--text-md);
    line-height: 1;
    cursor: pointer;
    flex: none;
    transition:
      color var(--dur-fast) var(--ease-out),
      background var(--dur-fast) var(--ease-out);
  }
  .close:hover {
    color: var(--fg);
    background: var(--glass-strong);
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
  }
  .kb td:first-child {
    color: var(--fg-dim);
    white-space: nowrap;
    width: 9em;
  }
  kbd {
    background: var(--bg-elev);
    color: var(--fg-strong);
    padding: 0 var(--space-2);
    border-radius: var(--radius);
    font-family: inherit;
    font-size: var(--text-xs);
  }
  .grammar {
    background: var(--bg-deep);
    padding: var(--space-3) var(--space-4);
    margin: var(--space-3) 0;
    color: var(--fg-strong);
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    line-height: 1.4;
    overflow-x: auto;
    white-space: pre;
  }
  .footer {
    display: flex;
    justify-content: flex-end;
    gap: var(--space-3);
    padding: var(--space-3) var(--space-6);
    color: var(--fg-muted);
  }
  .btn {
    background: var(--glass);
    color: var(--fg-strong);
    border: 1px solid transparent;
    padding: var(--space-3) var(--space-5);
    font: inherit;
    font-family: var(--font-mono);
    cursor: pointer;
  }
  .btn.primary {
    background: var(--accent);
    color: var(--text-on-accent);
    border-color: transparent;
  }
  .btn.primary:hover {
    background: var(--accent-light);
    border-color: transparent;
  }
</style>
