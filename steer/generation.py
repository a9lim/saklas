"""Token-by-token generation loop with KV cache, steering hooks, and monitor integration."""

from __future__ import annotations

import queue
import logging
import threading
from typing import Callable

import torch

log = logging.getLogger(__name__)

_eos_cache: tuple[int, set[int]] | None = None


def _get_eos_ids(model, tokenizer) -> set[int]:
    """Return cached set of all EOS token IDs for model+tokenizer."""
    global _eos_cache
    tok_id = id(tokenizer)
    if _eos_cache is not None and _eos_cache[0] == tok_id:
        return _eos_cache[1]
    eos_ids: set[int] = set()
    if hasattr(model, "generation_config") and model.generation_config.eos_token_id is not None:
        eid = model.generation_config.eos_token_id
        if isinstance(eid, int):
            eos_ids.add(eid)
        else:
            eos_ids.update(eid)
    if tokenizer.eos_token_id is not None:
        eos_ids.add(tokenizer.eos_token_id)
    _eos_cache = (tok_id, eos_ids)
    return eos_ids


class GenerationConfig:
    __slots__ = (
        "max_new_tokens", "temperature", "top_p", "system_prompt",
    )

    def __init__(
        self,
        max_new_tokens: int = 512,
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
        self.token_queue: queue.SimpleQueue[str | None] = queue.SimpleQueue()

    def request_stop(self):
        self.stop_requested.set()

    def reset(self):
        self.stop_requested.clear()
        self.is_generating.clear()
        # Drain leftover tokens
        while not self.token_queue.empty():
            try:
                self.token_queue.get_nowait()
            except queue.Empty:
                break


def build_chat_input(
    tokenizer,
    messages: list[dict[str, str]],
    system_prompt: str | None = None,
) -> torch.Tensor:
    chat = []
    if system_prompt:
        chat.append({"role": "system", "content": system_prompt})
    chat.extend(messages)
    if getattr(tokenizer, "chat_template", None) is not None:
        result = tokenizer.apply_chat_template(
            chat, add_generation_prompt=True, return_tensors="pt",
        )
        # Some tokenizers return a BatchEncoding dict instead of a raw tensor
        if isinstance(result, torch.Tensor):
            return result
        return result["input_ids"]
    # Base model without chat template — concatenate raw text
    text = "".join(m["content"] for m in chat)
    return tokenizer(text, return_tensors="pt")["input_ids"]


def generate_steered(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    config: GenerationConfig,
    state: GenerationState,
    on_token: Callable[[str], None] | None = None,
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
    seq_len = input_ids.shape[1]
    attn_mask_buf = torch.ones(1, seq_len + config.max_new_tokens, device=device, dtype=torch.long)
    prefill = True

    try:
        with torch.inference_mode():
            for _ in range(config.max_new_tokens):
                if state.stop_requested.is_set():
                    break

                outputs = model(
                    input_ids=current_input,
                    attention_mask=attn_mask_buf[:, :seq_len] if prefill else None,
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
                generated_ids.append(token_id)
                current_input = next_token
                seq_len += 1

                if on_token:
                    token_str = tokenizer.decode([token_id], skip_special_tokens=True)
                    on_token(token_str)

                if token_id in eos_ids:
                    break

    finally:
        state.is_generating.clear()
        # Signal end of generation
        if on_token:
            state.token_queue.put(None)

    return generated_ids
