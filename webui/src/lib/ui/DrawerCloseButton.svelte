<script lang="ts">
  interface Props {
    onclick: (ev: MouseEvent) => void;
    label?: string;
  }

  let { onclick, label = "Close drawer" }: Props = $props();
</script>

<button type="button" class="drawer-close" aria-label={label} {onclick}>✕</button>

<style>
  .drawer-close {
    display: inline-grid;
    place-items: center;
    flex: none;
    width: var(--control-compact);
    height: var(--control-compact);
    padding: 0;
    border: 1px solid transparent;
    border-radius: var(--radius-pill);
    background: var(--glass);
    color: var(--fg-muted);
    font-size: var(--text-sm);
    line-height: 1;
    transition:
      color var(--dur-fast) var(--ease-out),
      background var(--dur-fast) var(--ease-out),
      transform var(--dur-fast) var(--ease-out);
  }

  .drawer-close:hover {
    background: var(--glass-strong);
    color: var(--fg);
  }

  .drawer-close:active {
    transform: scale(0.94);
  }
</style>
