"""Token-by-token generation loop with KV cache, steering hooks, and monitor integration."""

import enum
import queue
import logging
import threading
from enum import IntEnum
from typing import Callable

import torch

from saklas.core.triggers import TriggerContext


class _ThinkState(IntEnum):
    IDLE = 0
    PREAMBLE = 1
    THINKING = 2
    RESPONSE_PREAMBLE = 3


class ThinkingState(enum.Enum):
    """Explicit lifecycle signal for the thinking state machine.

    Parallel to the internal ``_ThinkState`` IntEnum — this one is the
    public, stable signal exposed on ``GenerationState.thinking_state``
    so consumers (session, TUI, tests) can ask "are we currently
    generating thinking content?" without inferring it from
    ``thinking_end_idx``.
    """

    IDLE = "idle"
    PREAMBLE = "preamble"
    THINKING = "thinking"
    RESPONSE_PREAMBLE = "response_preamble"
    RESPONSE = "response"
    DONE = "done"

log = logging.getLogger(__name__)

def _tok_key(tokenizer) -> tuple[str, int]:
    return (getattr(tokenizer, 'name_or_path', ''), tokenizer.vocab_size)


_eos_cache: dict[tuple[str, int], set[int]] = {}


def _get_eos_ids(model, tokenizer) -> set[int]:
    """Return cached set of all EOS token IDs for model+tokenizer."""
    tok_key = _tok_key(tokenizer)
    cached = _eos_cache.get(tok_key)
    if cached is not None:
        return cached
    eos_ids: set[int] = set()
    if hasattr(model, "generation_config") and model.generation_config.eos_token_id is not None:
        eid = model.generation_config.eos_token_id
        if isinstance(eid, int):
            eos_ids.add(eid)
        else:
            eos_ids.update(eid)
    if tokenizer.eos_token_id is not None:
        eos_ids.add(tokenizer.eos_token_id)
    # Pick up end-of-turn tokens that some models (Gemma 4, etc.) add as
    # special tokens but don't list in generation_config.eos_token_id.
    _EOT_NAMES = {"<end_of_turn>", "<|endoftext|>", "<|end|>", "<|eot_id|>",
                  "<turn|>", "<|im_end|>"}
    added = getattr(tokenizer, "added_tokens_encoder", {})
    for tok_str, tok_id in added.items():
        if tok_str in _EOT_NAMES:
            eos_ids.add(tok_id)
    _eos_cache[tok_key] = eos_ids
    return eos_ids


_token_table_cache: dict[tuple[str, int], list[str | None]] = {}


def _get_token_table(tokenizer, vocab_size: int) -> list[str | None]:
    """Return cached token-id-to-string lookup table.

    Replaces per-token ``convert_ids_to_tokens`` calls with a single
    list index.  Built once per tokenizer, amortized across generations.
    Entries are ``None`` for tokens that decode to partial UTF-8 sequences
    (replacement char U+FFFD) — these must be buffered and decoded together
    with subsequent tokens (e.g. multi-token emoji).
    """
    tok_key = _tok_key(tokenizer)
    cached = _token_table_cache.get(tok_key)
    if cached is not None:
        return cached
    # batch_decode is orders of magnitude faster than per-id decode()
    # for large vocabs (150k+ tokens in modern models) — Rust-side loop
    # instead of a Python round-trip per entry.  Chunked so that a single
    # pathological token doesn't force the entire vocab onto the slow path.
    _CHUNK = 8192
    table: list[str | None] = [''] * vocab_size
    for start in range(0, vocab_size, _CHUNK):
        end = min(start + _CHUNK, vocab_size)
        ids = [[i] for i in range(start, end)]
        try:
            decoded = tokenizer.batch_decode(ids)
        except Exception:
            decoded = None
        if decoded is not None and len(decoded) == (end - start):
            for i, s in enumerate(decoded):
                table[start + i] = s if '\ufffd' not in s else None
        else:
            for i in range(start, end):
                try:
                    s = tokenizer.decode([i])
                    table[i] = s if '\ufffd' not in s else None
                except Exception:
                    table[i] = ''
    _token_table_cache[tok_key] = table
    return table


_think_delim_cache: dict[tuple[str, int], tuple[int | None, int | None, int | None, bool]] = {}
_none_result: tuple[int | None, int | None, int | None, bool] = (None, None, None, False)


def _detect_channel_delimiters(
    tokenizer,
) -> tuple[int | None, int | None, int | None, bool] | None:
    """Detect channel-based thinking for models that always use channels.

    Models like gpt-oss generate ``<|channel|>analysis<|message|>`` for
    thinking and ``<|channel|>response<|message|>`` for the reply without
    an ``enable_thinking`` template parameter.  Returns the delimiter
    tuple if both ``<|channel|>`` and ``<|message|>`` are added tokens,
    ``None`` otherwise.
    """
    added = getattr(tokenizer, "added_tokens_encoder", {})
    channel_id = added.get("<|channel|>")
    message_id = added.get("<|message|>")
    if channel_id is None or message_id is None:
        return None
    # Check whether the generation prompt already opens a channel
    # (model starts in thinking) or the model must emit it explicitly.
    try:
        gen_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "hi"}],
            add_generation_prompt=True, tokenize=False,
        )
    except Exception:
        gen_prompt = ""
    starts_in = "<|channel|>" in gen_prompt
    log.debug(
        "channel-based delimiters: channel=%d message=%d"
        " starts_in_thinking=%s",
        channel_id, message_id, starts_in,
    )
    return (
        None if starts_in else channel_id,
        channel_id,
        message_id,
        starts_in,
    )


def _detect_think_delimiters(
    tokenizer,
) -> tuple[int | None, int | None, int | None, bool]:
    """Detect thinking start/end delimiter tokens from the chat template.

    Returns ``(start_id, end_id, response_start_id, starts_in_thinking)``
    where:

    * **start_id** — token that opens a thinking section, or ``None`` if
      the generation prompt itself puts us in thinking mode (e.g. Qwen
      appends ``<think>`` to the prompt).
    * **end_id** — token that closes a thinking section.
    * **response_start_id** — token that marks the start of actual response
      content after the thinking section ends, or ``None`` if the response
      begins immediately after ``end_id``.  Used by channel-based formats
      (e.g. gpt-oss ``<|channel|>…<|message|>``) where multiple tokens
      separate thinking from response content.
    * **starts_in_thinking** — ``True`` when the first generated token is
      already thinking content (Qwen-style).  ``False`` when the model
      must explicitly open a thinking channel (Gemma-style).

    Detection works by rendering a round-trip assistant message through
    the tokenizer's own chat template and inspecting the delimiters that
    bracket the known thinking content.
    """
    tok_key = _tok_key(tokenizer)
    cached = _think_delim_cache.get(tok_key)
    if cached is not None:
        return cached

    template = getattr(tokenizer, "chat_template", None) or ""
    if "enable_thinking" not in template:
        # Check for channel-based thinking (e.g. gpt-oss) where the
        # model always generates <|channel|>analysis<|message|> without
        # an enable_thinking template parameter.
        result = _detect_channel_delimiters(tokenizer)
        _think_delim_cache[tok_key] = result if result else _none_result
        return _think_delim_cache[tok_key]

    think_marker = "XTHINKCONTENTX"
    response_marker = "XRESPONSECONTENTX"
    _dummy_tc = [{"function": {"name": "x", "arguments": {}}}]

    # Different model families represent thinking differently in assistant
    # messages.  Gemma requires reasoning/reasoning_content + tool_calls,
    # Qwen embeds <think>...</think> in content.  Try each schema until
    # one produces both markers in the rendered output.
    attempts = [
        {"role": "assistant", "reasoning_content": think_marker, "content": response_marker,
         "tool_calls": _dummy_tc},
        {"role": "assistant", "reasoning": think_marker, "content": response_marker,
         "tool_calls": _dummy_tc},
        {"role": "assistant", "thought": think_marker, "content": response_marker,
         "tool_calls": _dummy_tc},
        {"role": "assistant", "thought": think_marker, "content": response_marker},
        {"role": "assistant", "reasoning_content": think_marker, "content": response_marker},
        {"role": "assistant", "content": f"<think>\n{think_marker}\n</think>\n{response_marker}"},
    ]

    added = getattr(tokenizer, "added_tokens_encoder", {})

    for asst_msg in attempts:
        try:
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": "hi"}, asst_msg],
                tokenize=False, enable_thinking=True,
            )
        except Exception:
            continue

        ti = rendered.find(think_marker)
        ri = rendered.find(response_marker)
        if ti < 0 or ri <= ti:
            continue

        # --- end delimiter: first special token between the two markers ---
        # response_start_id is only relevant for channel-based models
        # (detected via _detect_channel_delimiters).  For enable_thinking
        # models the response follows the end delimiter directly.
        between = rendered[ti + len(think_marker):ri]
        end_pos, end_tok, end_id = len(between), None, None
        for tok_str, tok_id in added.items():
            pos = between.find(tok_str)
            if 0 <= pos < end_pos:
                end_pos, end_tok, end_id = pos, tok_str, tok_id
        if end_id is None:
            continue
        rs_id = None

        # --- start delimiter: closest special token before think_marker ---
        start_pos, start_tok, start_id = -1, None, None
        for tok_str, tok_id in added.items():
            pos = rendered.rfind(tok_str, 0, ti)
            if pos > start_pos:
                start_pos, start_tok, start_id = pos, tok_str, tok_id

        # If the start token already appears in the generation prompt the
        # model starts in thinking mode from the first generated token
        # (Qwen-style).  Otherwise the model must emit the start token
        # explicitly (Gemma-style) and may skip thinking entirely.
        starts_in_thinking = False
        if start_id is not None and start_tok is not None:
            try:
                gen_prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": "hi"}],
                    add_generation_prompt=True, tokenize=False,
                    enable_thinking=True,
                )
                if start_tok in gen_prompt:
                    starts_in_thinking = True
                    start_id = None  # nothing to detect at runtime
            except Exception:
                pass

        result = (start_id, end_id, rs_id, starts_in_thinking)
        log.debug(
            "thinking delimiters: start=%r end=%r response_start=%r"
            " starts_in_thinking=%s",
            start_tok if start_id is not None else "(prompt)",
            end_tok, None, starts_in_thinking,
        )
        _think_delim_cache[tok_key] = result
        return result

    log.warning("thinking supported but could not detect delimiters")
    _think_delim_cache[tok_key] = _none_result
    return _none_result


def supports_thinking(tokenizer) -> bool:
    """Check if the tokenizer's chat template supports thinking mode."""
    return _detect_think_delimiters(tokenizer) != _none_result


from dataclasses import dataclass  # noqa: E402


@dataclass(frozen=True)
class GenerationConfig:
    """Immutable sampling + system-prompt configuration.

    Frozen so an in-flight generation holding a local reference is
    immune to subsequent rebinds.  Callers that need to change a field
    rebind the attribute via ``dataclasses.replace``:

        session.config = replace(session.config, temperature=0.8)
    """

    max_new_tokens: int = 1024
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int | None = None
    system_prompt: str | None = None


class GenerationState:
    """Shared mutable state for controlling generation from the TUI."""

    def __init__(self):
        self.stop_requested = threading.Event()
        self.token_queue: queue.SimpleQueue = queue.SimpleQueue()
        self.thinking_end_idx: int = 0
        self.finish_reason: str = "stop"
        self.thinking_state: ThinkingState = ThinkingState.IDLE
        # For each on_token emission, the index in generated_ids of the
        # token that triggered it (last buffered ID for multi-byte emits),
        # plus whether the emit was thinking.  Used by the TUI to map
        # per-token probe scores (which are in generated_ids space) back
        # to the emitted token stream.
        self.emit_map: list[tuple[int, bool]] = []

    def request_stop(self):
        self.stop_requested.set()

    def reset(self):
        self.stop_requested.clear()
        self.token_queue = queue.SimpleQueue()
        self.thinking_end_idx = 0
        self.finish_reason = "stop"
        self.thinking_state = ThinkingState.IDLE
        self.emit_map = []


def build_chat_input(
    tokenizer,
    messages: list[dict[str, str]],
    system_prompt: str | None = None,
    thinking: bool = False,
) -> torch.Tensor:
    chat = []
    if system_prompt:
        chat.append({"role": "system", "content": system_prompt})
    chat.extend(messages)
    if getattr(tokenizer, "chat_template", None) is not None:
        kwargs: dict = {}
        if "enable_thinking" in (getattr(tokenizer, "chat_template", "") or ""):
            kwargs["enable_thinking"] = thinking
        result = tokenizer.apply_chat_template(
            chat, add_generation_prompt=True, return_tensors="pt", **kwargs,
        )
        # Some tokenizers return a BatchEncoding dict instead of a raw tensor
        if isinstance(result, torch.Tensor):
            return result
        return result["input_ids"]
    # Base model without chat template — add minimal role markers
    text = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in chat) + "\nAssistant:"
    return tokenizer(text, return_tensors="pt")["input_ids"]


def generate_steered(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    config: GenerationConfig,
    state: GenerationState,
    on_token: Callable[..., None] | None = None,
    thinking: bool = False,
    seed: int | None = None,
    stop: list[str] | None = None,
    logit_bias: dict[int, float] | None = None,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    logprobs: int | None = None,
    trigger_ctx: TriggerContext | None = None,
) -> list[int]:
    """
    Runs in a worker thread (not the async event loop).

    *on_token(text, is_thinking, token_id, logprob, top_logprobs)* is called
    for each emitted token.  For multi-token UTF-8 sequences (buffered
    partials), *token_id* is ``-1`` and logprob is None.

    ``logprobs`` is None (disabled) or the number of top logprobs to include
    per token (0 = only the chosen token's logprob).  ``stop`` is a list of
    strings that terminate generation when any appears in the completion
    text.  ``seed`` seeds the RNG for deterministic sampling.

    Sets ``state.finish_reason`` on exit: "stop" (EOS/external), "length"
    (max tokens), "stop_sequence" (stop string matched).

    Returns list of generated token IDs.
    """
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    device = input_ids.device
    eos_ids = _get_eos_ids(model, tokenizer)
    past_key_values = None
    current_input = input_ids
    generated_ids: list[int] = []
    _cfg = getattr(model.config, "text_config", model.config)
    _vocab = _cfg.vocab_size
    # top_k caps the candidate pool before top-p.  When unset, use 1024 as a
    # performance ceiling (nucleus sampling is insensitive beyond that).  When
    # set, honour the user's value as a hard cap — matches llama.cpp/Ollama
    # semantics where top_k is applied before top_p.
    _user_top_k = config.top_k if (config.top_k and config.top_k > 0) else 1024
    topk_k = min(_user_top_k, _vocab)
    token_table = _get_token_table(tokenizer, _vocab) if on_token else None
    seq_len = input_ids.shape[1]
    attn_mask_buf = torch.ones(1, seq_len, device=device, dtype=torch.long)
    prefill = True

    # Penalty / bias / stop / logprobs setup
    use_penalties = presence_penalty != 0.0 or frequency_penalty != 0.0
    completion_counts: dict[int, int] = {}
    bias_idx: torch.Tensor | None = None
    bias_val: torch.Tensor | None = None
    if logit_bias:
        bias_idx = torch.tensor(list(logit_bias.keys()), dtype=torch.long, device=device)
        bias_val = torch.tensor(list(logit_bias.values()), dtype=torch.float32, device=device)
    stop_list = list(stop) if stop else None
    completion_text = ""  # accumulated non-thinking emitted text, for stop matching
    state.finish_reason = "length"  # default: loop exhausted

    # Thinking state tracking
    if thinking:
        think_start_id, think_end_id, response_start_id, starts_in_thinking = (
            _detect_think_delimiters(tokenizer)
        )
    else:
        think_start_id = think_end_id = response_start_id = None
        starts_in_thinking = False
    # Hoisted: true iff channel-based format (gpt-oss) where EOS acts as
    # a channel separator inside thinking/preamble rather than terminating.
    has_response_start = response_start_id is not None
    tstate = (
        _ThinkState.THINKING
        if (starts_in_thinking and think_end_id is not None)
        else _ThinkState.IDLE
    )
    # Mirror the internal tstate onto the public ThinkingState enum.
    # Outside the thinking machine (no thinking requested, or no delimiters
    # detected) we go straight to RESPONSE.
    if thinking and think_end_id is not None:
        state.thinking_state = (
            ThinkingState.THINKING
            if starts_in_thinking
            else ThinkingState.IDLE
        )
    else:
        state.thinking_state = ThinkingState.RESPONSE

    # Buffer for multi-token characters (emoji, rare Unicode).
    # Tokens whose table entry is None represent partial UTF-8 byte sequences;
    # they accumulate here until a complete-token follows, at which point the
    # buffer is decoded as a group and flushed.
    pending_ids: list[int] = []
    pending_thinking: bool = False

    try:
        with torch.inference_mode():
            for _ in range(config.max_new_tokens):
                if state.stop_requested.is_set():
                    state.finish_reason = "stop"
                    break

                # Update the shared TriggerContext read by steering hooks.
                # ``prefill`` is the per-iter flag cleared after the first
                # model call; ``tstate`` is the thinking-state machine from
                # the previous iteration's bookkeeping; ``gen_step`` is the
                # raw token position the upcoming forward will produce.
                # Three attribute writes per step — below the noise floor
                # of the forward pass that follows.
                if trigger_ctx is not None:
                    trigger_ctx.is_prefill = prefill
                    trigger_ctx.thinking = (tstate == _ThinkState.THINKING)
                    trigger_ctx.gen_step = len(generated_ids)

                outputs = model(
                    input_ids=current_input,
                    attention_mask=attn_mask_buf if prefill else None,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                prefill = False

                past_key_values = outputs.past_key_values
                logits = outputs.logits[:, -1, :]
                # Steering can push hidden states past fp16 range, cascading
                # to inf/NaN logits.  nan_to_num clears NaN/inf first;
                # clamp then bounds any remaining finite outliers.
                logits.nan_to_num_(nan=0.0, posinf=100.0, neginf=-100.0)
                logits.clamp_(-100.0, 100.0)

                # Presence + frequency penalty (applied to raw logits,
                # before temperature, per OpenAI semantics).
                if use_penalties and completion_counts:
                    ids = list(completion_counts.keys())
                    counts = list(completion_counts.values())
                    idx_t = torch.tensor(ids, dtype=torch.long, device=device)
                    cnt_t = torch.tensor(counts, dtype=logits.dtype, device=device)
                    logits[0, idx_t] -= frequency_penalty * cnt_t + presence_penalty

                if bias_idx is not None:
                    logits[0, bias_idx] += bias_val.to(logits.dtype)

                # Compute logprobs of the pre-sampling distribution if requested.
                # TODO(perf): logprobs hot path forces per-token .item()/.tolist()
                # CPU syncs below. Batching to a single post-loop transfer would
                # win a few % on logprobs-enabled runs, but on_token streams
                # logprobs live in each emitted chunk (chat completions chunks,
                # TUI), so deferring would either break streaming semantics or
                # require parallel index bookkeeping. Leaving per-token sync
                # until we have a non-streaming caller that actually cares.
                chosen_logprob: float | None = None
                top_lp_pairs: list[tuple[int, float]] | None = None
                if logprobs is not None:
                    lp = torch.log_softmax(logits.float(), dim=-1)
                else:
                    lp = None

                if config.temperature <= 0:
                    # Greedy
                    next_token = logits.argmax(dim=-1, keepdim=True)
                else:
                    # Temperature + top-p (nucleus) sampling
                    logits.div_(config.temperature)
                    top_logits, top_idx = logits.topk(topk_k, dim=-1, sorted=True)
                    probs = top_logits.softmax(dim=-1)
                    cumprobs = probs.cumsum(dim=-1)
                    mask = (cumprobs - probs) >= config.top_p
                    probs[mask] = 0.0
                    probs[:, :1].clamp_(min=1e-8)
                    probs.div_(probs.sum(dim=-1, keepdim=True))

                    token_idx = torch.multinomial(probs, 1)
                    next_token = top_idx.gather(-1, token_idx)

                token_id = next_token.item()

                if lp is not None:
                    chosen_logprob = lp[0, token_id].item()
                    if logprobs > 0:
                        tlv, tli = lp[0].topk(min(logprobs, _vocab))
                        top_lp_pairs = [(int(i), float(v)) for i, v in zip(tli.tolist(), tlv.tolist())]

                if token_id in eos_ids:
                    # Channel-based models (gpt-oss) use EOS tokens as
                    # channel separators.  For these models only, skip
                    # EOS while inside thinking/preamble and transition
                    # to response preamble.  For enable_thinking models
                    # (Gemma, Qwen) EOS always terminates generation.
                    if not has_response_start or tstate == _ThinkState.IDLE:
                        state.finish_reason = "stop"
                        break
                    generated_ids.append(token_id)
                    current_input = next_token
                    if tstate == _ThinkState.THINKING:
                        if on_token and pending_ids:
                            on_token(tokenizer.decode(pending_ids),
                                     pending_thinking, -1, None, None)
                            pending_ids.clear()
                        tstate = _ThinkState.RESPONSE_PREAMBLE
                        state.thinking_state = ThinkingState.RESPONSE_PREAMBLE
                    elif tstate == _ThinkState.PREAMBLE:
                        if on_token and pending_ids:
                            on_token(tokenizer.decode(pending_ids),
                                     pending_thinking, -1, None, None)
                            pending_ids.clear()
                        tstate = _ThinkState.IDLE
                        state.thinking_end_idx = len(generated_ids)
                        state.thinking_state = ThinkingState.RESPONSE
                    continue

                # Advance KV cache state (common to all non-EOS paths)
                generated_ids.append(token_id)
                current_input = next_token

                # Handle thinking start delimiter (Gemma-style: model
                # explicitly opens a thinking channel)
                if (think_start_id is not None
                        and token_id == think_start_id
                        and tstate == _ThinkState.IDLE):
                    tstate = _ThinkState.PREAMBLE
                    state.thinking_state = ThinkingState.PREAMBLE
                    continue  # suppress start delimiter

                if tstate == _ThinkState.PREAMBLE:
                    if token_id == think_end_id:
                        tstate = _ThinkState.IDLE
                        state.thinking_end_idx = len(generated_ids)
                        state.thinking_state = ThinkingState.RESPONSE
                    elif (response_start_id is not None
                          and token_id == response_start_id):
                        tstate = _ThinkState.THINKING
                        state.thinking_state = ThinkingState.THINKING
                    else:
                        tok_text = tokenizer.decode([token_id])
                        if '\n' in tok_text:
                            tstate = _ThinkState.THINKING
                            state.thinking_state = ThinkingState.THINKING
                    continue  # suppress preamble

                # Handle end-of-thinking delimiter
                if tstate == _ThinkState.THINKING and token_id == think_end_id:
                    if on_token and pending_ids:
                        on_token(tokenizer.decode(pending_ids), pending_thinking, -1, None, None)
                        pending_ids.clear()
                    if response_start_id is not None:
                        tstate = _ThinkState.RESPONSE_PREAMBLE
                        state.thinking_state = ThinkingState.RESPONSE_PREAMBLE
                    else:
                        tstate = _ThinkState.IDLE
                        state.thinking_end_idx = len(generated_ids)
                        state.thinking_state = ThinkingState.RESPONSE
                    continue

                if tstate == _ThinkState.RESPONSE_PREAMBLE:
                    if token_id == response_start_id:
                        tstate = _ThinkState.IDLE
                        state.thinking_end_idx = len(generated_ids)
                        state.thinking_state = ThinkingState.RESPONSE
                    continue

                # Penalty bookkeeping: count all emitted completion tokens
                # (thinking and response alike, matching OpenAI's treatment
                # of the full completion sequence).
                if use_penalties:
                    completion_counts[token_id] = completion_counts.get(token_id, 0) + 1

                if on_token:
                    tok_str = token_table[token_id] if token_id < _vocab else ''
                    emit_text: str | None = None
                    emit_id = token_id
                    emit_thinking = tstate == _ThinkState.THINKING
                    if tok_str is None:
                        # Partial UTF-8 byte sequence — buffer until complete
                        if not pending_ids:
                            pending_thinking = emit_thinking
                        pending_ids.append(token_id)
                    elif pending_ids:
                        pending_ids.append(token_id)
                        emit_text = tokenizer.decode(pending_ids)
                        emit_id = -1
                        emit_thinking = pending_thinking
                        pending_ids.clear()
                    else:
                        emit_text = tok_str

                    if emit_text is not None:
                        # Stop-sequence check (response text only).
                        if stop_list and not emit_thinking:
                            prev_len = len(completion_text)
                            new_text = completion_text + emit_text
                            hit_idx = -1
                            for s in stop_list:
                                i = new_text.find(s, max(0, prev_len - len(s) + 1))
                                if i >= 0 and (hit_idx < 0 or i < hit_idx):
                                    hit_idx = i
                            if hit_idx >= 0:
                                # Trim emit to the pre-stop portion only.
                                trimmed = new_text[:hit_idx][prev_len:]
                                if trimmed:
                                    state.emit_map.append((len(generated_ids) - 1, emit_thinking))
                                    on_token(trimmed, emit_thinking, emit_id,
                                             chosen_logprob, top_lp_pairs)
                                state.finish_reason = "stop_sequence"
                                break
                            completion_text = new_text
                        state.emit_map.append((len(generated_ids) - 1, emit_thinking))
                        on_token(emit_text, emit_thinking, emit_id,
                                 chosen_logprob, top_lp_pairs)

        # Flush any remaining buffered partial tokens
        if on_token and pending_ids:
            state.emit_map.append((len(generated_ids) - 1, pending_thinking))
            on_token(tokenizer.decode(pending_ids), pending_thinking, -1, None, None)

    finally:
        state.thinking_state = ThinkingState.DONE
        # Flush MPS command buffers before signalling completion — without
        # this, a rapid regenerate can submit new work while Metal is still
        # processing the previous generation's command buffers, triggering
        # "commit an already committed command buffer".
        if device.type == "mps":
            torch.mps.synchronize()

    return generated_ids
