// Command-palette state — a tiny slice so both the App shell (⌘K) and
// the workspace rail's search button can drive the one palette.

export const paletteState: { open: boolean } = $state({ open: false });

let opener: HTMLElement | null = null;

export function openPalette(): void {
  if (
    !paletteState.open &&
    typeof document !== "undefined" &&
    document.activeElement instanceof HTMLElement
  ) {
    opener = document.activeElement;
  }
  paletteState.open = true;
}

export function closePalette(): void {
  const restore = opener;
  opener = null;
  paletteState.open = false;
  queueMicrotask(() => {
    if (restore?.isConnected) restore.focus();
  });
}

export function togglePalette(): void {
  if (paletteState.open) closePalette();
  else openPalette();
}
