// Command-palette state — a tiny slice so both the App shell (⌘K) and
// the workspace rail's search button can drive the one palette.

export const paletteState: { open: boolean } = $state({ open: false });

export function openPalette(): void {
  paletteState.open = true;
}

export function closePalette(): void {
  paletteState.open = false;
}

export function togglePalette(): void {
  paletteState.open = !paletteState.open;
}
