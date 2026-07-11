import type { DrawerName, DrawerState } from "../types";

export const drawerState: DrawerState = $state({
  open: null,
  params: null,
});

let opener: HTMLElement | null = null;

export function openDrawer(name: DrawerName, params: unknown = null): void {
  if (
    drawerState.open === null &&
    typeof document !== "undefined" &&
    document.activeElement instanceof HTMLElement
  ) {
    opener = document.activeElement;
  }
  drawerState.open = name;
  drawerState.params = params;
}

export function closeDrawer(): void {
  const restore = opener;
  opener = null;
  drawerState.open = null;
  drawerState.params = null;
  queueMicrotask(() => {
    if (restore?.isConnected) restore.focus();
  });
}
