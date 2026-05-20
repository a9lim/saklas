# core/

Engine layer: model loading, vector extraction, steering hooks, trait monitoring, session orchestration, the generation loop, and the loom conversation tree.

## model.py

HF causal LM loading. `_LAYER_ACCESSORS` maps `model_type` → layer-list accessor (`def`s, not lambdas). `_TESTED_ARCHS` frozenset gates a one-time `UserWarning` (tracked in `_warned`) when `model_type` isn't known-working. Cascading fallbacks on attention impl (SDPA → eager), dtype, and device. `_load_text_from_multimodal` extracts text-only sub-models (Ministral-as-Mistral3), strips `language_model.` prefixes, dequantizes FP8 inline. `patch_torch_for_mps()` installs two process-global MPS workarounds lazily, MPS-only: `torch.histc` casts integer inputs to float for MoE routing, and `torch.ldexp` round-trips MXFP4 dequant through CPU while honoring `out=`.

## vectors.py

The low-level extraction primitives. One forward pass per prompt; `_capture_all_hidden_states` hooks every layer in a single pass. `_encode_and_capture_all` pools from the **last content token**, walking back past both `tokenizer.all_special_ids` *and* `tokenizer.added_tokens_encoder` to skip trailing chat-template markers. The `added_tokens_encoder` arm matters for tokenizers that don't promote chat-boundary tokens to `all_special_ids` (talkie's `<|user|>`/`<|end|>`/`<|assistant|>`) — without it, extraction pools at the structural turn marker where outlier channels dominate and bakes wildly oversized probe magnitudes. Contrastive diffs are cast to **fp32** before differencing (fp16 loses precision on close vectors).

Two extractors share scaffolding. `_capture_diffs_for_pairs` runs the contrastive forward-pass loop and returns diffs plus per-layer pos/neg running means and SAE layer set; `_share_bake_and_warn` applies the DLS keep set, share-bakes retained layers, and emits the diagnostics warning. Both extractors close over these helpers; only the per-layer direction differs.

- `extract_difference_of_means` — per-layer direction = `mean(diffs)` in fp32. Score = `||direction||_M / ref_norm` when a `whitener` is passed (Mahalanobis bake), `||direction||_2 / ref_norm` otherwise. No SVD.
- `extract_contrastive` — first principal component per layer (one batched `torch.linalg.svd` across all layers), oriented by majority vote on `diff @ direction`. Score = explained-variance ratio (or `diff_norm / activation_norm` for single-pair). Whitener is ignored (EVR is metric-invariant).

Both accept `sae=<SaeBackend>` to run extraction in SAE feature space (encode pos/neg stacks, compute the direction, decode back). Both return `(profile, diagnostics)` — `dict[int, Tensor]` over retained layers plus per-layer probe-quality metrics on the same keys.

Share-baking: each direction is scaled to its layer's mean activation norm, then multiplied by `score_i / sum(scores)` over the DLS-retained subset. The angular hook reads `||baked_L||` back out as the layer share at apply time; the additive hook absorbs share into magnitude.

`compute_dls_mask(mu_pos, mu_neg, directions, layer_means)` — discriminative layer selection (Selective Steering, Dang & Ngo 2026 Eq. 9). A layer is retained iff `(μ_pos − μ_neutral)·d̂` and `(μ_neg − μ_neutral)·d̂` have opposite signs; same-side layers encode concept *intensity*, not *polarity*. `dls=False` or `layer_means=None` skips the mask; if every layer fails, the helper warns and keeps all rather than emptying the profile.

`_compute_layer_diagnostics` returns five floats per layer: `evr`, `intra_pair_variance_mean`/`_std`, `inter_pair_alignment` (mean off-diagonal `|cos|` of unit diffs), `diff_principal_projection`. They persist through `save_profile` → sidecar `diagnostics_by_layer` → `Profile.metadata["diagnostics"]` → `Profile.diagnostics`. `_emit_diagnostics_warning` fires once per extract when medians cross `_DIAG_DEGENERATE_EVR=0.95` + `_DIAG_DEGENERATE_INTRA_VAR=0.01` or `_DIAG_LOW_ALIGNMENT=0.2`; never raises. Surfaced via `saklas vector why`.

`compute_layer_means` averages neutral prompts for probe centering; `compute_neutral_activations` caches the per-prompt stack the whitener needs. `save_profile` / `load_profile` handle the safetensors+sidecar round-trip; `load_profile` dispatches on extension. `_template_overhead_cache` is keyed by `id(tokenizer)` — safe only while one tokenizer lives the session lifetime.

`project_profile(base, onto, operator, *, whitener=None)` — per-layer projection used by `session._materialize_projections`. `operator="~"` keeps the component aligned with `onto`, `"|"` keeps the orthogonal component. `whitener=None` → plain Gram-Schmidt; with a whitener → the closed-form LEACE projector for a single direction (Belrose et al. 2023). Layers absent from the whitener fall back to Euclidean per layer.

## extraction.py

`ExtractionPipeline` — the custom-concept orchestration lifted out of `session.py`. Owns the full flow: tensor cache → curated/local `statements.json` (reused by default) → generate scenarios → save → generate pairs → save → extract → save tensor. Dependencies are passed structurally via two runtime-checkable protocols, `ModelHandle` and `PackWriter`; `SaklasSession` satisfies both, so construction reads `ExtractionPipeline(self, self, self.events)`. `DEFAULT_EXTRACTION_METHOD = "dim"`. `_extractor_for` dispatches through `globals()` so module-scope test monkeypatches reach it; `_method_label` produces the sidecar `method` string. Cache-hit short-circuits defer `layer_means`/`whitener` resolution (both can trigger a lazy neutral-activation build) until after the tensor-cache check via `_resolve_extract_kwargs`. `extract()` emits `VectorExtracted`.

## sae.py

SAE backend abstraction. `SaeBackend` is a runtime-checkable `Protocol`: `encode_layer`/`decode_layer`, `release`, `revision`, `layers: frozenset[int]`. `MockSaeBackend` is an identity-by-default dataclass for CPU tests with optional per-layer fn overrides. `SaeLensBackend` is the concrete adapter; `load_sae_backend(release, *, revision, model_id, device, dtype)` queries SAELens's release registry, validates base-model compatibility, resolves per-layer sae_ids (`_canonical_layer_map` picks the narrowest-width SAE per layer and warns on multiple candidates), and records `sae_ids_by_layer` for the sidecar. `sae_lens` imports are gated inside `load_sae_backend` so the module imports without the `[sae]` extra. Missing dep → `SaeBackendImportError`; unknown release → `SaeReleaseNotFoundError` with `difflib` near-match suggestions. Model-name matching (`_model_names_match`) is lenient — SAELens short names vs full HF ids.

## mahalanobis.py

`LayerWhitener` holds per-layer centered neutral activations `X_L ∈ ℝ^(N, D)` and the small Woodbury inverse `K_L = (NλI + X Xᵀ)⁻¹`; `apply_inv(layer, v)` computes `Σ_reg⁻¹ v = (1/λ)(v − Xᵀ K X v)` in O(ND) without materializing D×D. Ridge `λ_L = (||X_L||_F² / (N·D)) · ridge_scale` (`DEFAULT_RIDGE_SCALE = 1.0`). Built lazily from `~/.saklas/models/<id>/{layer_means,neutral_activations}.safetensors` via `from_neutral_activations` (in-memory) or `from_cache(model_id)` (disk-only). Primitives: `mahalanobis_cosine`, `leace_project`, `apply_inv`. `SaklasSession.whitener` is a lazy property; `bootstrap_probes` builds it eagerly so default extraction is end-to-end Mahalanobis. `Profile.cosine_similarity(..., whitener=)` and `project_profile(..., whitener=)` fall back to Euclidean when `whitener=None`.

## hooks.py

`SteeringHook` carries the per-layer composed state and routes each forward through one of two injection paths. `injection_mode` (`"angular"` default, or `"additive"`) is stamped at construction and can be re-stamped on `recompose`, so flipping the session mode between calls re-warms the cache without rebuilding hooks.

`recompose(additive_entries, ablation_entries, device, dtype, ctx, *, injection_mode=None, theta_max=None)` groups `(tensor, effective_alpha, trigger)` triples by `Trigger` value. Per additive group it builds `composed = Σ α_i × baked_i` and `theta_strength = ||Σ α_i × (baked_i/||baked_i||)||`. The fast path applies when a single `Trigger.BOTH` group has no ablation; under angular, `_refresh_angular_cache` populates Python-scalar `_d_hat`/`_theta`/`_cos_t`/`_sin_t` so the hot path runs zero `.item()` calls.

- **Angular hot path**: per-position Givens rotation (`_angular_inplace`). Decompose `h = h_∥ + h_⊥`, rotate the unit vector toward `d̂`, restore the original per-position norm. Norm-preserving by construction — no rescale. Near-aligned positions (`||d_perp|| < 1e-6`) get a `torch.where` no-op fallback to dodge shrinkage.
- **Additive hot path**: `vector_norm(fp32) → add_(composed) → vector_norm(fp32).clamp_(1e-6) → mul_(ratio)` — two norms, one in-place add, one rescale.

Slow path: ablation groups fire first (`h' = h − α(h·d̂ − μ·d̂)d̂` per active direction), then active additive groups are summed and rotated/added once. Multi-direction ablation at one layer is naive-parallel — exact for orthogonal targets, over-ablates correlated ones (compose `!a + !b|a` for a clean subspace). A pre-check short-circuits when no group fires (e.g. `AFTER_THINKING` during prefill).

`SteeringManager` groups vectors by layer and dispatches per mode at `apply_to_model(...)`. Additive: `effective_alpha = user_alpha × _STEER_GAIN`; share is automatic via `||baked_L||`. Angular: pre-computes `share_L = ||baked_L|| / Σ ||baked||` and passes `effective_alpha = user_alpha × share_L` per layer, so cumulative `Σ_L θ_L = |α| × θ_max` regardless of layer count. Without this share-weighting the per-layer rotations stack to `N × |α| × θ_max` and crash coherence — keep it. `_STEER_GAIN = 2.0` multiplies **only** under additive mode. `DEFAULT_THETA_MAX = π/2`. `add_vector` and `add_ablation` parallel each other; ablation entries whose profile layers are missing from `layer_means` are silently skipped.

`HiddenCapture` — session/TUI companion: `attach`/`detach`/`stacked`/`latest_per_layer`. Each capture is `detach().clone()` of `output[0, -1, :]` (device-local). The session sets capture width: `_begin_capture(widen=False)` attaches to the probe-layer union; `widen=True` (driven by `SamplingConfig.return_hidden`) attaches to every layer and pays one device→host transfer per layer at finalize.

## monitor.py

`TraitMonitor` scores probes against per-layer hidden states. `_normalize_hidden` mean-centers with layer means and L2-normalizes; per-layer probe weight is `||baked||` (= `share × ref_norm`), recovered from tensor magnitude. `_ensure_cache` builds a per-layer stacked cache `{layer: (V[P,D], W[P])}` — one matmul per layer scores every probe in one launch, one `.cpu().tolist()` at the end, zero `.item()` calls in the hot path.

Entry points: `score_per_token` (primary; returns `(aggregate, per_token)`, updates history when `accumulate=True`, trims the EOS capture off-by-one so `capture[i] ↔ generated_ids[i]`); `measure_from_hidden` (pre-aggregated test path); `measure` (runs a forward pass for arbitrary text); `score_single_token` / `score_single_token_per_layer` (inline per-token, used by the SSE stream and the probe-gate callback); `score_stack` (per-token over a pre-captured `[T,D]` stack, no `generated_ids`/`tokenizer` dependency — backs `session.score_hidden`, `accumulate` defaults `False`).

Live running mean for streaming: `begin_live`/`update_live`/`end_live`; `get_current_and_previous` prefers live values so the TUI trait panel ticks during streaming. `add_probe`/`remove_probe` invalidate the stacked cache wholesale.

## triggers.py

`Trigger` (frozen dataclass) carries phase flags plus an optional `gate: ProbeGate | None`. `Trigger.active(ctx)` consults the phase flags and, when a gate is set, `ctx.probe_scores[gate.probe]` against `score <op> threshold`. `Trigger.first(n)` / `Trigger.after(n)` / `Trigger.when(probe, op, threshold)` are the factories; `when` builds the canonical `prompt=False, gate=ProbeGate(...)` shape. `TriggerContext.probe_scores` is filled by the per-step score callback during decoding and cleared on `reset()`. Gated triggers report inactive during prefill (no reading yet) and for missing probes (no raise). `ProbeGate` is frozen/hashable so identical gates compose under equality in `recompose`'s per-trigger grouping.

## hooks ↔ generation: cuda_graphs.py

CUDA-graphs / `StaticCache` support. When `cuda_graphs=True` + device is CUDA + a StaticCache-compatible architecture + fast-path-eligible steering, generation routes through `transformers.StaticCache` so kernel shapes stay fixed and `torch.compile` can capture graphs. `is_cuda_graphs_supported(model, device)` probes viability and caches `(supported, reason)` keyed by underlying module id (through `torch.compile`'s `_orig_mod`), device, and dtype — so a pre-compile probe and a post-compile call hit the same entry. `make_static_cache` builds a cache sized to `prompt_len + max_new_tokens + offset`. `warn_once` logs the fallback reason once per model. Slow-path steering (probe gates, multi-trigger, ablation under CTX mutation) bypasses StaticCache — the eligibility check lives at the steering layer.

## generation.py

Token-by-token decode + KV cache, wrapped in `torch.inference_mode()`. Models that return no `past_key_values` during prefill (custom modeling that ignores the cache kwarg, e.g. talkie) flip `no_cache_mode` and re-feed the full accumulated input each step — O(N²), one-time warning, off the throughput-invariant path; a preallocated buffer + advancing view avoids per-step `torch.cat`.

`GenerationConfig` (frozen) holds session-level defaults (`max_new_tokens`, `temperature`, `top_p`, `top_k`, `system_prompt`); per-call `SamplingConfig` overrides are composed into a local copy and never mutate `session.config`. Top-p is via `torch.topk`, not full-vocab sort; `top_k` (default `None` → 1024 cap) is a hard candidate-pool cap applied **before** top-p (llama.cpp/Ollama order). In-place ops throughout; MPS sync at end-of-loop prevents Metal command-buffer reuse crashes.

`generate_steered(...)` accepts `seed`, `stop`, `logit_bias`, `presence_penalty`/`frequency_penalty` (sparse device-side unique-id/count buffers, applied to raw logits before temperature), `logprobs` (0 = chosen-only, >0 = top-k), `score_callback`, and `forced_prefix`. `score_callback` enables probe-gated triggers: when set, it's called once per forward (after the model returns, before the next hook fire) and its dict is written to `trigger_ctx.probe_scores`. The session wires it only when the active steering carries a gated `Trigger`; otherwise the kwarg stays `None` (one `is None` check per step).

`forced_prefix` is a list of token ids forced for the first `len(forced_prefix)` decode steps instead of the sampled token. The `multinomial` draw still runs every step, so re-seeding keeps the RNG stream bit-identical through the fork point. `session.fork_from_token(node_id, raw_index, alt_token_id)` slices a source assistant node's `raw_token_ids` and lands a sibling under the same recipe; `session.prefill_assistant(node_id, text, *, steering, sampling, on_token)` tokenizes `text` (`add_special_tokens=False`) into the prefix and lands a sibling assistant under a *user* node, always `thinking=False`. Both bump the token budget by the prefix length. Legacy/transcript-loaded nodes carry no `raw_token_ids` and aren't forkable.

`session.append_user_turn(parent_node_id, text)` / `session.append_assistant_turn(user_node_id, text)` are the no-generation "commit" entry points (Ctrl+Enter on either surface). Neither runs a decode — they land a turn directly on the tree and advance the active node. `append_user_turn` wraps `tree.add_user_turn` (same dedup, refuses anchoring under a user-role parent via `_check_user_send_target`); `append_assistant_turn` tokenizes `text` into `raw_token_ids` so the authored turn stays forkable, lands the node with `recipe=None` (the implicit "no model run produced this" marker, same shape transcript-loaded nodes carry), and clears `tokens`/`thinking_tokens` since authored turns have no per-token scores. Both raise `InvalidNodeOperationError` on empty text or shape violations.

`supports_thinking` delegates to `_detect_think_delimiters`, which round-trips an assistant message through the chat template and returns `(start_id, end_id, response_start_id, starts_in_thinking)` — no hardcoded delimiter strings, works across Qwen/Gemma/gpt-oss. `ThinkingState` enum (`IDLE → PREAMBLE → THINKING → RESPONSE_PREAMBLE → RESPONSE → DONE`) is set at every transition and forced to `DONE` in `finally`. `GenerationState` holds `stop_requested` (Event), `token_queue` (SimpleQueue), `thinking_end_idx`, `thinking_state`, `finish_reason` (`"stop"` default, flips to `"length"`/`"stop_sequence"`), `emit_map` (maps emitted tokens back to `generated_ids` space), and `response_text` (authoritative final non-thinking text, since stop sequences can trim a partial token).

`�` partial-UTF-8 tokens are marked `None` in the token table and buffered until complete. `_get_token_table` uses chunked (8192) `batch_decode` with per-chunk fallback to per-id `decode()`. `_tok_key` is the shared cache-key helper for `_eos_cache` / `_token_table_cache` / `_think_delim_cache`. The `None` done sentinel is **not** emitted by `generate_steered` — the session worker / TUI closure puts it on the queue in `finally`.

## session.py

`SaklasSession` owns the model, profile registry (`_profiles`), monitor, `SteeringManager`, `HiddenCapture`, generation defaults (`session.config`), the loom tree, and a synchronous `events: EventBus`.

Construction: `from_pretrained(model_id, *, device, dtype, quantize, probes, injection_mode="angular", theta_max=None, ...)` does HF load + probe bootstrap + layer-mean compute; `__init__(model, tokenizer, ...)` accepts a pre-loaded model. `theta_max=None` → `DEFAULT_THETA_MAX`. Both paths unconditionally call `materialize_bundled()` + `selectors.invalidate()` early — `bootstrap_probes` does this transitively but is skipped when `probes=[]`, which would otherwise leave fresh bundled concepts invisible to `_all_concepts()`.

`generate` / `generate_stream` are keyword-only and accept `steering: str | Steering | None` (expression string or pre-built `Steering`; dicts rejected), `sampling: SamplingConfig`, `thinking=None` (auto-detect via `supports_thinking`), plus loom args (`parent_node_id`, `n`, `recipe_override`). Both return `RunSet`; `.first` is the underlying `GenerationResult` and common attributes delegate to it. `generate_stream(live_scores=False)` suppresses inline per-token scoring but still computes final per-token scores.

`session.steering(value)` is a context manager: coerces via `Steering.from_value`, materializes `ProjectedTerm` entries into derived profiles, and pushes onto a LIFO stack. Per-call `injection_mode`/`theta_max`/`projection_metric` ride a parallel `_steering_override_stack` (triplets); `_resolve_projection_metric` / the resolver walks it top-down for inner-wins. `_resolve_pole_aliases` keeps a cache-hit auto-load fast path (`_try_autoload_vector`) so HTTP clients can steer bundled probes without pre-registration. Default projection metric is Mahalanobis — `_materialize_projections` passes `session.whitener` to `project_profile`; whitener-coverage misses fall back to Gram-Schmidt transparently.

No persistent steering hooks. `generate`/`generate_stream` wrap `_generate_core`, which owns the `_gen_lock` re-entry guard, `_gen_phase` lifecycle (`IDLE`/`PREAMBLE`/`RUNNING`/`FINALIZING`, read via `session.gen_state`), the steering context, `_begin_capture`/`_end_capture`, `_finalize_generation`, and deterministic teardown. Helpers split the mechanics: `_prepare_generation_call`, `_start_loom_assistant`, `_snapshot_steering_alphas`, `_run_generation_loop`. `generate_stream` spawns a worker enqueueing `TokenEvent`s into a `SimpleQueue`; iterator `close()` triggers the same teardown.

Monitor scoring is in-flight: `HiddenCapture` is attached on the probe-layer union, `_finalize_generation` → `monitor.score_per_token`, no second forward pass. Hidden-state round-trip: `SamplingConfig(return_hidden=True)` widens capture to every layer and lands `GenerationResult.hidden_states` (`dict[int, Tensor]`, CPU); `session.score_hidden(hidden, *, per_token, accumulate)` re-scores any compatible dict.

`session.lock` is an `asyncio.Lock` (server-owned FIFO serializer), distinct from the threading `_gen_lock`. `generate_batch` / `generate_sweep` return `RunSet` (list-like, plus `node_ids`/`grid`/`.to_collector()`/`.to_dataframe()`); `generate_sweep` builds the Cartesian product of `{concept: [alphas]}` and lands the runs as loom siblings under one user turn. Events from the hot path: `GenerationStarted`, `SteeringApplied`, `SteeringCleared`, `ProbeScored`, `GenerationFinished`, plus `VectorExtracted`; server/TUI subscribers must hop via `loop.call_soon_threadsafe`.

The custom-concept extraction pipeline lives in `extraction.py`; the session delegates to `ExtractionPipeline`. `_N_PAIRS = 45`, `_N_PAIRS_PER_SCENARIO = 5`.

## loom.py

`LoomTree` — the engine-side conversation tree. Nodes are turns, children are alternative continuations; the active path is the model's context. Owns the data model only — generation integration, gen-lock concurrency, persistence, and event delivery live in `session.py`/the server. Node ids are 26-char ULIDs (inline `_ulid`, sortable by timestamp prefix). Five primitives: edit, branch, navigate, delete_subtree, plus `regenerate` (via `generate(parent_node_id=...)`). `Recipe` is the per-node reproducibility receipt (steering expression, sampling, thinking, seed, probe set + per-probe sha256); `Recipe.overlay`/`invert_steering`/`compose_modifier` back the auto-regen modes (`unsteered`/`inverted`/`reseed`/`cool`/`hot`/custom). Per-node token blobs are in memory during streaming; `to_dict` omits them, `save` writes them to a gzip sidecar. Mutators raise `MutationDuringGenerationError` (HTTP 409) when conflicting with an in-flight gen; `UnknownNodeError` (404) and `InvalidNodeOperationError` (400) cover the rest.

## tree_filter.py

Filter grammar for tree pruning — deliberately distinct from the steering `@when:` grammar (that gates per-step readings; this gates per-node aggregates). Clauses are `<agg>:<probe> <op> <threshold>`; `agg_op` is `agg` (default, `LoomNode.aggregate_readings`), `any` (max/min over per-token scores), or `last`; multi-clause is AND. `parse_filter` → `FilterClause`; `FilterClause.evaluate(node, *, per_token_scores=)`; `filter_tree(tree, text, ...)` is the free-function form behind `LoomTree.filter_by_expr`. Missing probe → clause is `False`. `FilterParseError` on any parse problem.

## loom_diff.py

Cross-branch diff primitives. `text_diff` — word-level diff over whitespace-split tokens via `difflib.SequenceMatcher`, returns aligned `DiffSpan`s. `readings_diff` — per-probe `Δ = b − a` table sorted by `abs(delta)`, missing-side readings default `0.0` with `a_value`/`b_value` preserved. `per_token_diff` — byte-offset alignment between two token sequences, stops where bytes diverge. `steering_delta` — compact `"+0.2 calm"` edge label by walking both expressions through the shared grammar.

## transcript.py

Transcript export/import for loom paths. A `Transcript` is a saved path through the tree (system prompt + user turns + each assistant turn's `Recipe` + final readings), serialized to YAML via `to_yaml`/`from_yaml`. `SAKLAS_TRANSCRIPT_VERSION = 1`. Three import modes: **default** (new top-level branch off root), **here** (child of active node), **merge** (attach the non-matching tail at the deepest matching user-turn prefix). Guards: model mismatch refuses `--merge` (`TranscriptModelMismatch`); system-prompt mismatch warns + banners; missing probes warn; probe hash drift warns, or raises `TranscriptProbeDriftError` under `--strict`.

## joint_logprobs.py

Cross-branch joint-logprob computation for the loom NodeCompareDrawer. Given two sibling assistant nodes, force-replays each branch under its stamped recipe and assembles per-aligned-position rows carrying both branches' chosen-token logprobs *and* the cross-branch evaluation. `_compute_rows` is the pure-tensor inner loop — alignment via `per_token_diff` + sampler-logprob lookup at `prefix_len + i − 1`. `compute_joint_logprobs(session, a_id, b_id)` is the IO wrapper — reconstructs the prefix (shared prefix via `_shared_prefix_len`, not re-tokenization), force-replays through the model, then calls the inner loop. `approx_kl` is top-K-truncated `KL(P_A‖P_B)` — approximate signal, the tail is unobserved. Results cache lazily on `session._joint_logprob_cache` keyed by sorted `(a_id, b_id)`; the route layer flips a/b labelling via `reorient_for_request`. Held under the session lock — the forward passes compete with generation.

## results.py

`GenerationResult`, `RunSet`, `TokenAlt`, `TokenEvent`, `ProbeReadings`, `ResultCollector`. `RunSet` is the standard multi-run shape: list-like, plus `node_ids`, `grid`, `.first`, `.to_collector()`, `.to_dataframe()`. `TokenAlt` is a frozen `(id, text, logprob)` triple — engine decodes text at capture time so consumers don't re-tokenize. `TokenEvent` carries `thinking`, `logprob`, `top_alts`, `finish_reason`, `scores` (per-probe cosine sims, populated by `generate_stream` only when probes are active and `live_scores=True`), `perplexity`. `GenerationResult` carries `prompt_tokens`, `finish_reason`, optional `logprobs`. `applied_steering` holds the canonical steering expression that produced the result (round-trips through `parse_expr`). `ResultCollector` handles batch export (JSONL/CSV/DataFrame). `to_dict()` omits `hidden_states` (tensors don't serialize — persist with `torch.save`).

## histogram.py

Shared per-layer magnitude-histogram helper. `HIST_BUCKETS = 16`; `bucketize(norms, buckets)` collapses sorted per-layer norms into evenly-sized groups (one bucket per layer when there are fewer layers than buckets). Used by the TUI WHY footer and CLI `vector why`.

## sampling.py / steering.py / steering_expr.py / events.py / errors.py / profile.py

`SamplingConfig` — per-call frozen config with `merged_with`. `Steering` — frozen; `from_value` accepts `str | Steering | None` (strings route through the shared parser, dicts rejected); optional `injection_mode`, `theta_max`, `projection_metric` (all `None` = inherit session default). `events.py` — synchronous `EventBus` + 6 event dataclasses. `errors.py` — `SaklasError` base; every saklas exception multi-inherits through it while keeping its stdlib MRO, so `except SaklasError` catches the family and `except ValueError`/`RuntimeError` at existing sites still works. `user_message()` returns an `(http_status, text)` pair. `profile.py` — `Profile` wraps `dict[int, Tensor]` with `.layers`/`.weight_at`/`.save`/`.load`/`.to_gguf`/`.merged`/`.merged_with`/`.promoted_to`/`.cosine_similarity`/`.projected_away`; `cosine_similarity(other, *, per_layer=False, whitener=None)` is magnitude-weighted; empty layer intersection raises `ProfileError`.

`steering_expr.py` hosts the unified grammar every input surface shares. `parse_expr(text, *, namespace=None)` → `Steering`; `format_expr(steering)` renders it back (round-trips); `referenced_selectors(text)` yields `(ns, concept, variant)` triples for install-time checks. Projection terms land in `Steering.alphas` as `ProjectedTerm(coeff, trigger, operator, base, onto)`, materialized into derived profiles by the session before the hook layer sees them. `!atom` produces an `AblationTerm` (default coeff 1.0 — fully replace with the neutral mean); it doesn't compose with `~`/`|`. Ablation terms are dispatched separately — `normalized_entries()` skips them and they route to `SteeringManager.add_ablation`.
