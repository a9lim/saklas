// Drawer barrel — re-exports every drawer component under a short name
// so the orchestrator (App.svelte / DrawerHost) can import the whole
// suite as one module:
//
//   import * as Drawers from "./drawers";
//   <Drawers.Extract params={drawerState.params} />
//
// Names match the DrawerName union in lib/types.ts (modulo the trivial
// snake_case → PascalCase mapping):
//
//   "extract"            → Extract
//   "load"               → Load
//   "save_conversation"  → SaveConversation
//   "load_conversation"  → LoadConversation
//   "compare"            → Compare
//   "system_prompt"      → SystemPrompt
//   "model_info"         → ModelInfo
//   "help"               → Help
//
// The export drawer doesn't sit on the DrawerName union yet (the plan
// notes drawer expansion as part of phase 9+) but is included here so
// the wiring in App.svelte can adopt it without a second pass.

export { default as Extract } from "./ExtractDrawer.svelte";
export { default as Load } from "./LoadDrawer.svelte";
export { default as VectorPicker } from "./VectorPickerDrawer.svelte";
export { default as ProbePicker } from "./ProbePickerDrawer.svelte";
export { default as SaveConversation } from "./SaveConversationDrawer.svelte";
export { default as LoadConversation } from "./LoadConversationDrawer.svelte";
export { default as Compare } from "./CompareDrawer.svelte";
export { default as SystemPrompt } from "./SystemPromptDrawer.svelte";
export { default as ModelInfo } from "./ModelInfoDrawer.svelte";
export { default as Help } from "./HelpDrawer.svelte";
export { default as Export } from "./ExportDrawer.svelte";
export { default as Correlation } from "./CorrelationDrawer.svelte";
export { default as LayerNorms } from "./LayerNormsDrawer.svelte";
