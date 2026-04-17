# core/

Engine layer: model loading, contrastive PCA extraction, steering hooks, trait monitoring, session orchestration, generation loop.

## model.py

HF causal LM loading. `_LAYER_ACCESSORS` maps `model_type` → layer-list accessor (`def`, not lambdas). `_TESTED_ARCHS` frozenset gates a one-time `UserWarning` (via `_warned: set[str]`) when `model_type` isn't known-working. Cascading fallbacks on attention impl (SDPA → eager), dtype, device. `_load_text_from_multimodal` extracts text-only sub-models (Ministral-as-Mistral3), strips `language_model.` prefixes, dequantizes FP8 inline. Patches `torch.histc` for MPS MoE routing.

## vectors.py

One forward pass per prompt; `_capture_all_hidden_states` hooks every layer in a single pass. `_encode_and_capture_all` pools from the **last content token**, walking back past `tokenizer.all_special_ids` to skip trailing chat-template markers (Llama `<|eot_id|>`, Gemma `<end_of_turn>`, Qwen `<|im_end|>`). Contrastive diffs cast to **fp32** before differencing (fp16 loses precision on close vectors, makes SVD degenerate).

- **Multi-pair**: batched SVD across all layers (one `torch.linalg.svd` on stacked `(n_layers, N, dim)`), scored by explained-variance ratio.
- **Single-pair**: scored by `diff_norm / activation_norm`.
- **Shares baked at extraction**: each direction scaled to mean activation norm of its layer, then multiplied by `score_i / sum(scores)`; hook-side math collapses to flat `user_alpha * _STEER_GAIN * sum(baked)` (all-zero fallback to uniform).

Returns a **profile**: `dict[int, Tensor]` covering every layer — no scores dict. `compute_layer_means` averages 90 neutral prompts for centering. `load_profile` dispatches on extension. `_template_overhead_cache` is keyed by `id(tokenizer)` — safe only while one tokenizer lives the session lifetime.

**SAE branch (optional).** `extract_contrastive(sae=<SaeBackend>)` filters layers to `sae.layers`, captures per-pair pos/neg stacks alongside diffs (only on covered layers — non-covered layers pay nothing extra), encodes both stacks through `SaeBackend.encode_layer`, runs `pca_center` SVD on the mean-centered feature-space stack, orients by majority vote on `(F_pos - F_neg) @ v_feat`, decodes the first PC back through `SaeBackend.decode_layer`, then hands off to the existing normalize + share-bake step. Branches surgical — the default (sae=None) path is bit-identical to v1.x. Raises `SaeCoverageError` when the SAE covers zero of the model's layers.

## sae.py

SAE backend abstraction. `SaeBackend` is a minimal runtime-checkable `Protocol` — `encode_layer(idx, h) -> features`, `decode_layer(idx, feat) -> model_space_vec`, `release`, `revision`, `layers: frozenset[int]`. `MockSaeBackend` is an identity-by-default dataclass for CPU-only tests with optional per-layer `encode_fn`/`decode_fn` overrides.

`SaeLensBackend` is the concrete adapter. `load_sae_backend(release, *, revision, model_id, device, dtype)` queries SAELens's pretrained-release registry, validates base-model compatibility (`SaeModelMismatchError`), resolves per-layer sae_ids via `_canonical_layer_map` (bucket by hook_layer, pick lexicographically smallest — narrowest width — per layer, warn when multiple candidates exist), loads each SAE module, and returns a fully-populated `SaeLensBackend`. `sae_ids_by_layer` gets recorded in the pack sidecar for reproducibility.

`sae_lens` imports are gated inside `load_sae_backend` so installations without the `[sae]` extra can still import the module (for Protocol type hints or the mock). Missing dep raises `SaeBackendImportError` with the install hint; unknown release raises `SaeReleaseNotFoundError` with near-match suggestions via `difflib` (falls back to listing the first 10 available releases when the fuzzy matcher returns nothing).

Model-name matching is lenient (`_model_names_match`): SAELens's `cfg.model_name` is often a short name (`gpt2-small`), saklas passes full HF ids (`openai-community/gpt2`); we split on `/`, lowercase, accept either-contains-the-other.

## hooks.py

`SteeringHook` adds a pre-composed vector to hidden states and **unconditionally rescales each position back to its pre-injection norm** (in-place, `torch.linalg.vector_norm(dtype=float32)` to avoid fp16 sum-of-squares overflow at hidden_dim ≥ 2048). No flag, no escape hatch.

**Trigger grouping (v1.5)**: `recompose` takes `(tensor, alpha, trigger)` triples and groups entries by `Trigger` value. Fast path (single group with `Trigger.BOTH`) collapses to one composed tensor and skips the per-step `.active(ctx)` check, bit-identical to the v1.4 hook. Slow path stores `composed_groups: list[(Trigger, Tensor)]`; `hook_fn` consults each group's trigger against a shared `TriggerContext` and adds only active ones. The norm-preservation rescale wraps the conditional sum, and a pre-check short-circuits the fp32 norm round-trip when no group fires (e.g. `AFTER_THINKING` during prefill).

`SteeringManager` groups vectors by layer, sums co-layer directions into **one pre-composed hook per active layer**. `add_vector(name, profile, alpha, trigger=Trigger.BOTH)`. **Flat-scalar apply**: `effective_injection = user_alpha * _STEER_GAIN * sum(baked)` — no per-layer score lookup, no `sum(scores)` division. Preserves layer-count invariance and score-magnitude invariance; both moved from apply-time to extract-time. Manager owns the per-generation `TriggerContext`; `generate_steered` mutates its `is_prefill` / `thinking` / `gen_step` fields before each forward pass.

`_STEER_GAIN = 3.5` calibrated on gemma-4-31b-it. Coherent band α ≈ 0.2–0.6, cliff α ≈ 0.75. Cliff transfers within ±0.1α across architectures; smaller/MoE/safety-trained models may need proportionally higher α.

`HiddenCapture` — session + TUI companion: `attach(layers, layer_indices)` / `detach()` / `stacked()`. Each capture is `detach().clone()` of `output[0, -1, :]` (device-local, no sync); k-th capture = "state that produced generated token k".

## monitor.py

`TraitMonitor` scores probes against per-layer hidden states via `_score_probes` (shared `_normalize_hidden` mean-centers with layer means, L2-normalizes, magnitude-weighted cosine sim). **Per-layer weight = `||baked||`** (= `share_L * ref_norm_L`) — recovered from tensor magnitude itself.

`_ensure_cache` builds a **per-layer stacked cache** `{layer_idx: (V[P,D], W[P])}` where `V` holds unit-normed probe directions (zero rows for missing layers) and `W[p]` = `||baked_p_L||` (0 when missing). One matmul per layer scores every probe against a hidden state in one kernel launch; one `.cpu().tolist()` at the end — the hot path has **zero `.item()` calls** regardless of probe count.

Entry points:
- `score_per_token(captured, generated_ids, tokenizer, *, accumulate)` — primary. Returns `(aggregate_vals, per_token_scores)`, updates history when `accumulate=True`. Aggregate pools from last non-special token.
- **EOS off-by-one trim**: when generation ends on EOS, the final model forward fires (capture +1) but the EOS token is not appended to `generated_ids`. `score_per_token` trims trailing extras to align `capture[i] ↔ generated_ids[i]` (previously zeroed every per-token score on EOS-terminated gens).
- `measure_from_hidden(hidden_per_layer)` — test-path pre-aggregated entry.
- `measure(model, tokenizer, layers, text)` — runs a forward pass; convenience for scoring arbitrary text outside a generation run.
- `score_single_token(hidden_per_layer)` — inline per-token entry, used by SSE trait stream **and** `generate_stream._push` for live scoring on every emit.

**Live running mean** for streaming: `begin_live()` / `update_live(scores)` / `end_live()` maintain a per-probe running mean across the current gen; `get_current_and_previous` prefers live values when present so the TUI trait-panel ticks during streaming. `_pending_aggregate` / `_pending_per_token` split so the TUI poll sees aggregate readiness independently.

`add_probe`/`remove_probe` invalidate the stacked cache wholesale (rebuilt lazily). `probe_names`, `profiles`, `layer_means` are properties (no defensive copies).

## session.py

`SaklasSession` owns model, profile registry (`_profiles`), monitor, `SteeringManager`, `HiddenCapture`, generation defaults (`session.config`, frozen `GenerationConfig`), conversation history, and a synchronous `events: EventBus`.

**Construction**: `SaklasSession.from_pretrained(model_id, *, device, dtype, quantize, probes, ...)` does HF load + probe bootstrap + layer-mean compute. Plain `__init__(model, tokenizer, *, probes, ...)` accepts a pre-loaded `PreTrainedModel` for multi-session-on-one-model scenarios.

**Public API on `generate` / `generate_stream`** is keyword-only:
```python
session.generate(input, *, steering=None, sampling=None, stateless=False, raw=False, thinking=None, on_token=None)
```
`steering` accepts a `Steering` or bare `dict[str, float]`; `sampling` is a `SamplingConfig`; `thinking=None` auto-detects via `supports_thinking`; `on_token` is a public callback.

**`session.steering(alphas)` context manager** is the canonical pole-resolution site: pushes onto a LIFO stack, resolves pole aliases via `cli.selectors.resolve_pole`, rebuilds hooks from the flattened head; nesting flattens with inner-wins semantics; emits `SteeringApplied` / `SteeringCleared`. `_resolve_pole_aliases` has a **cache-hit auto-load fast path** (`_try_autoload_vector`) — if a concept name isn't in `_profiles` but has an installed pack with an already-extracted per-model tensor, the tensor is loaded inline so HTTP clients can steer bundled probes without pre-registration. No PCA, no network.

**No persistent steering hooks.** `generate` and `generate_stream` are thin wrappers around `_generate_core`, which owns:
- `_gen_lock` threading re-entry guard (flips `_gen_active`)
- the steering context
- `_begin_capture` / `_end_capture`
- `_finalize_generation`
- deterministic teardown: `stop_requested.set()` → worker join → `_end_capture()` → `_clear_steering()`

`generate_stream` spawns a worker running `_generate_core` with an internal `on_token` that enqueues `TokenEvent`s into a local `SimpleQueue`; iterator `close()` / `GeneratorExit` triggers the same teardown.

**Monitor scoring is in-flight**: `_generate_core` attaches `HiddenCapture` on the union of probe layers, `_finalize_generation` → `score_captured` → `monitor.score_per_token`, no second forward pass. Events emitted from the hot path: `GenerationStarted`, `SteeringApplied`, `SteeringCleared`, `ProbeScored`, `GenerationFinished`, plus `VectorExtracted` from `extract()`. Server/TUI subscribers must hop via `loop.call_soon_threadsafe` for an event-loop context.

**`session.lock`** is an `asyncio.Lock` distinct from the threading `_gen_lock`; it's the server-owned async-level serializer that queues concurrent HTTP requests FIFO, not a generation-reentry guard.

`last_per_token_scores` and `last_result` are public properties. **Regime note**: scoring uses actual generation-regime hidden states with full prompt context; `layer_means` are computed wrap-assistant — slightly off-regime, recompute via `saklas pack refresh neutrals` if bias appears.

`_generation_preamble` dedupes lock/steering/preamble setup.

**Full custom-concept pipeline**: tensor cache → curated or local `statements.json` (reused by default) → generate scenarios → save → generate pairs → save → contrastive PCA → save tensor. Curated concepts save under `default/<c>/`; user concepts under `local/<c>/`. Scenario and statement caches are model-independent. `_update_local_pack_files` refreshes `pack.json.files` via `hash_folder_files` from `io/packs.py`. `_N_PAIRS = 45`. Extract lookup scans `saklas.cli.selectors._all_concepts()` across every namespace.

## sampling.py / steering.py / events.py / errors.py / profile.py

Per-call `SamplingConfig` (frozen with `merged_with`), `Steering` (frozen, `from_value` coerces dict/None), synchronous `EventBus` + 6 event dataclasses, `SaklasError` base (all others multi-inherit through it preserving stdlib MRO), and the `Profile` class wrapping `dict[int, Tensor]` with `.layers` / `.weight_at` / `.save` / `.load` / `.to_gguf` / `.merged` / `.merged_with` / `.promoted_to` / `.cosine_similarity` / `.projected_away`. `cosine_similarity(other, *, per_layer=False)` computes magnitude-weighted cosine over shared layers (aggregate `float`) or raw per-layer cosines (`dict[int, float]`). `projected_away(other)` removes other's direction per-layer via orthogonal projection (fp32, near-zero guard). Empty intersection raises `ProfileError`.

## generation.py

Token-by-token + KV cache, wrapped in `torch.inference_mode()`. `GenerationConfig` is `@dataclass(frozen=True)` holding session-level defaults (`max_new_tokens`, `temperature`, `top_p`, `top_k`, `system_prompt`); callers rebind session defaults via `dataclasses.replace(session.config, ...)` rather than mutating.

**Per-call sampling overrides never touch `session.config`** — `_generate_core` composes a local `GenerationConfig` from `session.config` merged with the call's `SamplingConfig` via `_compose_gen_config(sampling)`, and hands that to `generate_steered`. In-flight gens are immune to concurrent session-default rebinds.

Top-p via `torch.topk`, not full-vocab sort; `GenerationConfig.top_k` optional (default `None` = 1024-cap) is a hard cap on candidate pool applied **before** top-p (matches llama.cpp/Ollama order). In-place ops throughout. MPS sync at end-of-loop prevents Metal command-buffer reuse crashes.

`generate_steered` accepts `seed`, `stop` (list), `logit_bias` (pre-built `(idx, val)` tensors), `presence_penalty`/`frequency_penalty` (per-token `completion_counts` dict, applied to raw logits before temperature), `logprobs` (0 = chosen-only, >0 = top-k via `log_softmax(logits.float())` before sampling; gated off hot path when `None`).

`state.finish_reason` defaults to `length`, flips to `stop` on EOS/external-stop or `stop_sequence` on stop-string match (runs against accumulated non-thinking response text per emitted token; trims current emit to pre-stop prefix). **`None` done sentinel is NOT emitted by `generate_steered`** — the session worker or TUI closure puts it on the queue in `finally`.

`supports_thinking` delegates to `_detect_think_delimiters` which renders a round-trip assistant message through the chat template and returns `(start_id, end_id, response_start_id, starts_in_thinking)` — works across families (Qwen `</think>`, Gemma `<channel|>`, gpt-oss `<|channel|>…<|message|>`) without hardcoded delimiter strings.

**Explicit `ThinkingState` enum** on `GenerationState` (`IDLE → PREAMBLE → THINKING → RESPONSE_PREAMBLE → RESPONSE → DONE`), set at every transition and forced to `DONE` in `finally`; additive — `thinking_end_idx` remains the authoritative position marker.

`\ufffd` partial-UTF-8 tokens marked `None` in the token table and buffered until a complete token, then flushed via `tokenizer.decode(pending_ids)`. `_get_token_table` uses **chunked (8192) `batch_decode`** with per-chunk try/except fallback to per-id `decode()` — fast first-call on 150k+ vocabs. `_tok_key` shared cache-key helper for `_eos_cache`, `_token_table_cache`, `_think_delim_cache`.

`GenerationState`: `stop_requested` (Event), `token_queue` (SimpleQueue), `thinking_end_idx`, `thinking_state`, `finish_reason`, `emit_map` (list of `(generated_ids_index, is_thinking)` per `on_token` call — maps emitted tokens back to the `generated_ids` space for TUI highlight score projection).

## results.py

`GenerationResult`, `TokenEvent`, `ProbeReadings` dataclasses with `to_dict()`. `TokenEvent`: `thinking`, `logprob`, `top_logprobs`, `finish_reason`, `scores: dict[str, float] | None` (per-probe cosine sims against the latest captured hidden state, populated by `generate_stream` only when probes are active — drives live highlighting / WHY top-tokens / live trait readings without waiting for finalize). `GenerationResult`: `prompt_tokens`, `finish_reason` (`stop`/`length`/`stop_sequence`), optional `logprobs`. `ResultCollector` for batch export (dicts/JSONL/CSV/DataFrame).
