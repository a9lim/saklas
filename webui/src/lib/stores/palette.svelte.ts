// Command-palette state shared by the app shell and launcher hint.

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
