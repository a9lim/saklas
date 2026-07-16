<script lang="ts">
  // Saklas editable combobox — arbitrary role labels plus a themed roster
  // picker. Unlike a native datalist, the popup stays inside the webui's
  // surface, type, focus, and keyboard system.
  import { onMount, tick } from "svelte";

  interface Option { value: string; label: string; }
  interface Props {
    value: string;
    options: Option[];
    placeholder?: string;
    disabled?: boolean;
    invalid?: boolean;
    ariaLabel?: string;
    title?: string;
    spellcheck?: boolean;
  }

  let {
    value = $bindable(), options, placeholder = "", disabled = false,
    invalid = false, ariaLabel, title, spellcheck = false,
  }: Props = $props();

  let open = $state(false);
  let filtering = $state(false);
  let highlight = $state(-1);
  let input: HTMLInputElement | null = $state(null);
  let listbox: HTMLUListElement | null = $state(null);
  let popoverStyle = $state("");
  const uid = $props.id();

  const visible = $derived.by(() => {
    if (!filtering || !value.trim()) return options;
    const needle = value.trim().toLowerCase();
    return options.filter((opt) => opt.label.toLowerCase().includes(needle));
  });

  async function openPopover(filter = false): Promise<void> {
    if (disabled) return;
    filtering = filter;
    open = true;
    highlight = visible.findIndex((opt) => opt.value === value);
    if (highlight < 0 && visible.length > 0) highlight = 0;
    await tick();
    try { listbox?.showPopover(); } catch { /* fixed fallback */ }
    placeListbox();
  }

  function closePopover(): void {
    if (!open) return;
    try { listbox?.hidePopover(); } catch { /* already closed */ }
    open = false;
    filtering = false;
  }

  async function commit(index: number): Promise<void> {
    const opt = visible[index];
    if (!opt) return;
    value = opt.value;
    closePopover();
    await tick();
    input?.focus();
    input?.setSelectionRange(value.length, value.length);
  }

  function move(dir: 1 | -1): void {
    if (visible.length === 0) return;
    highlight = (highlight + dir + visible.length) % visible.length;
    const row = listbox?.children[highlight] as HTMLElement | undefined;
    row?.scrollIntoView({ block: "nearest" });
  }

  function onKeydown(ev: KeyboardEvent): void {
    if (ev.key === "ArrowDown" || ev.key === "ArrowUp") {
      ev.preventDefault();
      if (!open) void openPopover(false);
      else move(ev.key === "ArrowDown" ? 1 : -1);
    } else if (ev.key === "Enter" && open && highlight >= 0) {
      ev.preventDefault();
      void commit(highlight);
    } else if (ev.key === "Escape" && open) {
      ev.preventDefault();
      closePopover();
    } else if (ev.key === "Tab") {
      closePopover();
    }
  }

  function placeListbox(): void {
    if (!input || !listbox) return;
    const r = input.parentElement?.getBoundingClientRect()
      ?? input.getBoundingClientRect();
    const gutter = 8, gap = 2;
    const desired = Math.min(240, Math.max(32, listbox.scrollHeight));
    const below = window.innerHeight - r.bottom - gutter - gap;
    const above = r.top - gutter - gap;
    const upward = below < Math.min(desired, 160) && above > below;
    const maxHeight = Math.max(32, Math.min(240, upward ? above : below));
    const height = Math.min(desired, maxHeight);
    const width = Math.min(r.width, window.innerWidth - gutter * 2);
    const left = Math.max(gutter, Math.min(r.left, window.innerWidth - width - gutter));
    const top = upward
      ? Math.max(gutter, r.top - height - gap)
      : Math.min(window.innerHeight - height - gutter, r.bottom + gap);
    popoverStyle = `left:${left}px;top:${top}px;width:${width}px;max-height:${maxHeight}px`;
  }

  function onDocumentPointer(ev: PointerEvent): void {
    if (!open) return;
    const target = ev.target as Node;
    if (input?.parentElement?.contains(target) || listbox?.contains(target)) return;
    closePopover();
  }
  function reposition(): void { if (open) placeListbox(); }

  onMount(() => {
    document.addEventListener("pointerdown", onDocumentPointer, true);
    window.addEventListener("resize", reposition);
    window.addEventListener("scroll", reposition, true);
    return () => {
      document.removeEventListener("pointerdown", onDocumentPointer, true);
      window.removeEventListener("resize", reposition);
      window.removeEventListener("scroll", reposition, true);
    };
  });
</script>

<div class="sk-combobox" class:is-open={open} class:is-invalid={invalid}>
  <input
    bind:this={input} bind:value {placeholder} {disabled} {title} {spellcheck}
    role="combobox" aria-label={ariaLabel} aria-autocomplete="list"
    aria-expanded={open} aria-controls={open ? `${uid}-listbox` : undefined}
    aria-activedescendant={open && highlight >= 0 ? `${uid}-option-${highlight}` : undefined}
    aria-invalid={invalid}
    oninput={() => void openPopover(true)} onkeydown={onKeydown}
  />
  <button
    type="button" class="caret" aria-label={`Choose ${ariaLabel ?? "value"}`}
    tabindex="-1" {disabled}
    onclick={() => (open ? closePopover() : void openPopover(false))}
  >▾</button>

  {#if open}
    <ul bind:this={listbox} id={`${uid}-listbox`} class="popover"
      popover="manual" style={popoverStyle} role="listbox">
      {#each visible as opt, i (opt.value)}
        <li id={`${uid}-option-${i}`} role="option"
          class:highlight={i === highlight} class:active={opt.value === value}
          aria-selected={opt.value === value}
          tabindex="-1"
          onpointerenter={() => (highlight = i)}
          onpointerdown={(ev) => ev.preventDefault()}
          onkeydown={(ev) => {
            if (ev.key === "Enter" || ev.key === " ") void commit(i);
          }}
          onclick={() => void commit(i)}>{opt.label}</li>
      {:else}
        <li class="empty" role="option" aria-selected="false" aria-disabled="true">
          new role
        </li>
      {/each}
    </ul>
  {/if}
</div>

<style>
  .sk-combobox { position: relative; display: flex; align-items: center; min-width: 0;
    width: 100%; border-radius: var(--radius); background: var(--input-well);
    transition: background var(--dur-fast) var(--ease-out); }
  .sk-combobox:hover, .sk-combobox.is-open { background: var(--surface-hi); }
  .sk-combobox:focus-within { outline: 2px solid var(--focus-ring); outline-offset: 1px; }
  .sk-combobox.is-invalid { outline: 1px solid var(--accent-red); }
  input { width: 100%; min-width: 0; padding: var(--space-2) 26px var(--space-2) var(--space-3);
    border: 0; outline: 0; background: transparent; color: var(--fg); font: inherit;
    font-family: var(--font-mono); font-size: var(--text-sm); }
  input:disabled, input:disabled + .caret { opacity: .5; cursor: not-allowed; }
  .caret { position: absolute; right: 0; display: grid; place-items: center; width: var(--control-compact);
    height: 100%; padding: 0; border: 0; background: transparent; color: var(--fg-muted);
    cursor: pointer; transition: color var(--dur-fast) var(--ease-out),
      transform var(--dur-fast) var(--ease-out); }
  .is-open .caret { color: var(--accent); transform: rotate(180deg); }
  .popover { position: fixed; inset: auto; z-index: var(--z-modal); box-sizing: border-box;
    overflow-y: auto; margin: 0; padding: var(--space-1) 0; list-style: none;
    border: 1px solid var(--glass-line); border-radius: var(--radius); background: var(--surface-hi);
    box-shadow: var(--shadow-overlay); }
  li { min-height: var(--control-target); padding: var(--space-2) var(--space-3); overflow: hidden;
    color: var(--fg-strong); font-size: var(--text-sm); line-height: 1.3;
    text-overflow: ellipsis; white-space: nowrap; cursor: pointer; }
  li.highlight { background: var(--bg-hover); color: var(--fg); }
  li.active { background: var(--accent-subtle); }
  li.active.highlight { background: var(--accent-strong); }
  li.empty { color: var(--fg-muted); cursor: default; }
</style>
