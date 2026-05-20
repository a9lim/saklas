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
//   "load"               → Load
//   "save_conversation"  → SaveConversation
//   "load_conversation"  → LoadConversation
//   "compare"            → Compare
//   "system_prompt"      → SystemPrompt
//   "help"               → Help
//
// Standalone custom extraction was retired in the webui overhaul — the
// extract form now inlines into the steering picker (VectorPickerDrawer).

export { default as Load } from "./LoadDrawer.svelte";
export { default as VectorPicker } from "./VectorPickerDrawer.svelte";
export { default as ProbePicker } from "./ProbePickerDrawer.svelte";
export { default as SaveConversation } from "./SaveConversationDrawer.svelte";
export { default as LoadConversation } from "./LoadConversationDrawer.svelte";
export { default as Compare } from "./CompareDrawer.svelte";
export { default as SystemPrompt } from "./SystemPromptDrawer.svelte";
export { default as Help } from "./HelpDrawer.svelte";
export { default as Export } from "./ExportDrawer.svelte";
export { default as Correlation } from "./CorrelationDrawer.svelte";
export { default as LayerNorms } from "./LayerNormsDrawer.svelte";
export { default as ExperimentLab } from "./ExperimentLabDrawer.svelte";
export { default as ActivationAtlas } from "./ActivationAtlasDrawer.svelte";
export { default as RecipeBuilder } from "./RecipeBuilderDrawer.svelte";
export { default as AdvancedSampling } from "./AdvancedSamplingDrawer.svelte";
export { default as Health } from "./HealthDrawer.svelte";
export { default as SessionAdmin } from "./SessionAdminDrawer.svelte";
// Phase 5 drawers — cross-branch diff and transcript IO.
export { default as NodeCompare } from "./NodeCompareDrawer.svelte";
export { default as Transcript } from "./TranscriptDrawer.svelte";
