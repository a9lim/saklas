import type { DrawerName, DrawerState } from "../types";

export const drawerState: DrawerState = $state({
  open: null,
  params: null,
});

export function openDrawer(name: DrawerName, params: unknown = null): void {
  drawerState.open = name;
  drawerState.params = params;
}

export function closeDrawer(): void {
  drawerState.open = null;
  drawerState.params = null;
}
