"""Token-by-token generation loop with KV cache, steering hooks, and monitor integration."""

from __future__ import annotations

import queue
import logging
import threading
from typing import Callable

import torch

log = logging.getLogger(__name__)

_eos_cache: dict[tuple[str, int], set[int]] = {}


def _get_eos_ids(model, tokenizer) -> set[int]:
    """Return cached set of all EOS token IDs for model+tokenizer."""
    tok_key = (getattr(tokenizer, 'name_or_path', ''), tokenizer.vocab_size)
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
    tok_key = (getattr(tokenizer, 'name_or_path', ''), vocab_size)
    cached = _token_table_cache.get(tok_key)
    if cached is not None:
        return cached
    table: list[str | None] = [''] * vocab_size
    for i in range(vocab_size):
        try:
            s = tokenizer.decode([i])
            table[i] = s if '\ufffd' not in s else None
        except Exception:
            table[i] = ''
    _token_table_cache[tok_key] = table
    return table


_think_delim_cache: dict[tuple[str, int], tuple[int | None, int | None, int | None, bool]] = {}


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
    tok_key = (getattr(tokenizer, 'name_or_path', ''), tokenizer.vocab_size)
    cached = _think_delim_cache.get(tok_key)
    if cached is not None:
        return cached

    _none_result: tuple[int | None, int | None, int | None, bool] = (None, None, None, False)
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
    template = getattr(tokenizer, "chat_template", None) or ""
    if "enable_thinking" in template:
        return True
    # Channel-based thinking (e.g. gpt-oss) — always active
    added = getattr(tokenizer, "added_tokens_encoder", {})
    return "<|channel|>" in added and "<|message|>" in added


class GenerationConfig:
    __slots__ = (
        "max_new_tokens", "temperature", "top_p", "system_prompt",
    )

    def __init__(
        self,
        max_new_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: float = 0.9,
        system_prompt: str | None = None,
    ):
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.system_prompt = system_prompt


class GenerationState:
    """Shared mutable state for controlling generation from the TUI."""

    def __init__(self):
        self.stop_requested = threading.Event()
        self.is_generating = threading.Event()
        self.token_queue: queue.SimpleQueue[tuple[str, bool] | None] = queue.SimpleQueue()
        self.thinking_end_idx: int = 0

    def request_stop(self):
        self.stop_requested.set()

    def reset(self):
        self.stop_requested.clear()
        self.is_generating.clear()
        self.token_queue = queue.SimpleQueue()
        self.thinking_end_idx = 0


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
    on_token: Callable[[str, bool], None] | None = None,
    thinking: bool = False,
) -> list[int]:
    """
    Runs in a worker thread (not the async event loop).
    Returns list of generated token IDs.
    """
    state.is_generating.set()
    device = input_ids.device
    eos_ids = _get_eos_ids(model, tokenizer)
    past_key_values = None
    current_input = input_ids
    generated_ids: list[int] = []
    _cfg = getattr(model.config, "text_config", model.config)
    _vocab = _cfg.vocab_size
    topk_k = min(1024, _vocab)
    token_table = _get_token_table(tokenizer, _vocab) if on_token else None
    seq_len = input_ids.shape[1]
    attn_mask_buf = torch.ones(1, seq_len, device=device, dtype=torch.long)
    prefill = True

    # Thinking state tracking
    if thinking:
        think_start_id, think_end_id, response_start_id, starts_in_thinking = (
            _detect_think_delimiters(tokenizer)
        )
    else:
        think_start_id = think_end_id = response_start_id = None
        starts_in_thinking = False
    in_thinking = starts_in_thinking and think_end_id is not None
    in_preamble = False  # suppressing start-of-thinking tokens (e.g. "thought\n")
    in_response_preamble = False  # suppressing post-thinking channel labels

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
                    break

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
                # to inf/NaN logits. Clamp bounds the range for stable softmax;
                # nan_to_num replaces any remaining NaN→0.
                logits.clamp_(-100.0, 100.0)
                torch.nan_to_num(logits, nan=0.0, out=logits)

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

                if token_id in eos_ids:
                    # Channel-based models (gpt-oss) use EOS tokens as
                    # channel separators.  For these models only, skip
                    # EOS while inside thinking/preamble and transition
                    # to response preamble.  For enable_thinking models
                    # (Gemma, Qwen) EOS always terminates generation.
                    if response_start_id is None or not (
                        in_thinking or in_preamble or in_response_preamble
                    ):
                        break
                    # EOS inside a thinking/preamble phase — advance KV
                    # state, transition out of thinking, and keep generating.
                    generated_ids.append(token_id)
                    current_input = next_token
                    if in_thinking:
                        # EOS ends the thinking channel — enter response
                        # preamble so inter-channel tokens (turn markers
                        # like <|start|>assistant) are suppressed.
                        in_thinking = False
                        if on_token and pending_ids:
                            on_token(tokenizer.decode(pending_ids),
                                     pending_thinking)
                            pending_ids.clear()
                        in_response_preamble = True
                    elif in_preamble:
                        in_preamble = False
                        if on_token and pending_ids:
                            on_token(tokenizer.decode(pending_ids),
                                     pending_thinking)
                            pending_ids.clear()
                        state.thinking_end_idx = len(generated_ids)
                    continue

                # Advance KV cache state (common to all non-EOS paths)
                generated_ids.append(token_id)
                current_input = next_token

                # Handle thinking start delimiter (Gemma-style: model
                # explicitly opens a thinking channel)
                if (think_start_id is not None
                        and token_id == think_start_id
                        and not in_thinking and not in_preamble):
                    in_preamble = True
                    continue  # suppress start delimiter

                # Suppress preamble tokens between the start delimiter and
                # the first newline or content marker (e.g. Gemma's
                # "thought\n" channel label, gpt-oss's "analysis<|message|>")
                if in_preamble:
                    if token_id == think_end_id:
                        # Empty thinking section — end delimiter hit
                        # during preamble
                        in_preamble = False
                        state.thinking_end_idx = len(generated_ids)
                    elif (response_start_id is not None
                          and token_id == response_start_id):
                        # Channel-based format: content marker ends preamble
                        in_preamble = False
                        in_thinking = True
                    else:
                        tok_text = tokenizer.decode([token_id])
                        if '\n' in tok_text:
                            in_preamble = False
                            in_thinking = True
                    continue  # suppress preamble

                # Handle end-of-thinking delimiter
                if in_thinking and token_id == think_end_id:
                    in_thinking = False
                    # Flush any buffered partial tokens before the delimiter
                    if on_token and pending_ids:
                        on_token(tokenizer.decode(pending_ids), pending_thinking)
                        pending_ids.clear()
                    if response_start_id is not None:
                        # Channel-based format (e.g. gpt-oss): suppress
                        # response channel preamble until content marker
                        in_response_preamble = True
                    else:
                        state.thinking_end_idx = len(generated_ids)
                    continue

                # Suppress response preamble tokens between end-of-thinking
                # and start-of-response (channel-based formats)
                if in_response_preamble:
                    if token_id == response_start_id:
                        in_response_preamble = False
                        state.thinking_end_idx = len(generated_ids)
                    continue

                if on_token:
                    tok_str = token_table[token_id] if token_id < _vocab else ''
                    if tok_str is None:
                        # Partial UTF-8 byte sequence — buffer until complete
                        if not pending_ids:
                            pending_thinking = in_thinking
                        pending_ids.append(token_id)
                    elif pending_ids:
                        pending_ids.append(token_id)
                        on_token(tokenizer.decode(pending_ids), pending_thinking)
                        pending_ids.clear()
                    else:
                        on_token(tok_str, in_thinking)

        # Flush any remaining buffered partial tokens
        if on_token and pending_ids:
            on_token(tokenizer.decode(pending_ids), pending_thinking)

    finally:
        # Flush MPS command buffers before signalling completion — without
        # this, a rapid regenerate can submit new work while Metal is still
        # processing the previous generation's command buffers, triggering
        # "commit an already committed command buffer".
        if device.type == "mps":
            torch.mps.synchronize()
        state.is_generating.clear()

    return generated_ids
