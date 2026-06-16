// Drawer barrel — re-exports every drawer component under a short name
// so the orchestrator (App.svelte / DrawerHost) can import the whole
// suite as one module:
//
//   import * as Drawers from "./drawers";
//   <Drawers.RackDrawer params={drawerState.params} />
//
// Names match the DrawerName union in lib/types.ts (modulo the trivial
// snake_case → PascalCase mapping):
//
//   "save_conversation"  → SaveConversation
//   "load_conversation"  → LoadConversation
//   "compare"            → Compare
//   "system_prompt"      → SystemPrompt
//   "help"               → Help
//
// ``RackDrawer`` is the shared rack browser — one component reskinned by
// geometry family.  ``family: "subspace"`` (flat pca / baked fits, white
// accent) and ``family: "manifold"`` (curved spectral / authored fits,
// purple accent) are mirror images, differing only by accent, label, and
// catalog filter.  Both rack "+ add" buttons open it; the "+ build
// manifold" launcher inside it routes to ``ManifoldBuilderDrawer`` for
// both families (a flat fit is just a pca manifold).

export { default as RackDrawer } from "./RackDrawer.svelte";
export { default as ManifoldBuilder } from "./ManifoldBuilderDrawer.svelte";
export { default as ManifoldMerge } from "./ManifoldMergeDrawer.svelte";
export { default as ManifoldPack } from "./ManifoldPacksDrawer.svelte";
export { default as SaveConversation } from "./SaveConversationDrawer.svelte";
export { default as LoadConversation } from "./LoadConversationDrawer.svelte";
export { default as Compare } from "./CompareDrawer.svelte";
export { default as SystemPrompt } from "./SystemPromptDrawer.svelte";
export { default as Help } from "./HelpDrawer.svelte";
export { default as Export } from "./ExportDrawer.svelte";
export { default as Correlation } from "./CorrelationDrawer.svelte";
export { default as LayerNorms } from "./LayerNormsDrawer.svelte";
export { default as ProbeInspector } from "./ProbeInspectorDrawer.svelte";
export { default as ExperimentLab } from "./ExperimentLabDrawer.svelte";
export { default as ActivationAtlas } from "./ActivationAtlasDrawer.svelte";
export { default as RecipeBuilder } from "./RecipeBuilderDrawer.svelte";
export { default as AdvancedSampling } from "./AdvancedSamplingDrawer.svelte";
export { default as Health } from "./HealthDrawer.svelte";
export { default as SessionAdmin } from "./SessionAdminDrawer.svelte";
// Phase 5 drawers — cross-branch diff and transcript IO.
export { default as NodeCompare } from "./NodeCompareDrawer.svelte";
export { default as Transcript } from "./TranscriptDrawer.svelte";
// Templated-completion lab — author templates + score the value distribution.
export { default as TemplateLab } from "./TemplateLabDrawer.svelte";
