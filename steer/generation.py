"""Token-by-token generation loop with KV cache, steering hooks, and monitor integration."""

from __future__ import annotations

import queue
import logging
import threading
from typing import Callable

import torch

log = logging.getLogger(__name__)


class GenerationConfig:
    __slots__ = (
        "max_new_tokens", "temperature", "top_p", "system_prompt",
    )

    def __init__(
        self,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
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
    # Build set of all EOS token IDs — model.generation_config often has
    # additional stop tokens (e.g. <turn|> for Gemma chat models).
    eos_ids: set[int] = set()
    if hasattr(model, "generation_config") and model.generation_config.eos_token_id is not None:
        eid = model.generation_config.eos_token_id
        if isinstance(eid, int):
            eos_ids.add(eid)
        else:
            eos_ids.update(eid)
    if tokenizer.eos_token_id is not None:
        eos_ids.add(tokenizer.eos_token_id)
    past_key_values = None
    current_input = input_ids
    generated_ids: list[int] = []

    try:
        with torch.inference_mode():
            for _ in range(config.max_new_tokens):
                if state.stop_requested.is_set():
                    break

                outputs = model(
                    input_ids=current_input,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

                past_key_values = outputs.past_key_values
                logits = outputs.logits[:, -1, :]
                # Steering can push hidden states past fp16 range, cascading
                # to inf/NaN logits. nan_to_num replaces NaN→0 and inf→max,
                # then clamp bounds the range for stable softmax.
                torch.nan_to_num(logits, nan=0.0, posinf=100.0, neginf=-100.0, out=logits)
                logits.clamp_(-100.0, 100.0)

                # Temperature
                if config.temperature > 0:
                    logits.div_(config.temperature)
                else:
                    # Greedy
                    next_token = logits.argmax(dim=-1, keepdim=True)
                    token_id = next_token.item()
                    generated_ids.append(token_id)
                    current_input = next_token
                    if on_token:
                        on_token(tokenizer.decode([token_id], skip_special_tokens=True))
                    if token_id in eos_ids:
                        break
                    continue

                # Top-p (nucleus) sampling — use topk to avoid sorting full vocab
                k = min(max(1000, logits.shape[-1] // 32), logits.shape[-1])
                top_logits, top_idx = logits.topk(k, dim=-1, sorted=True)
                probs = top_logits.softmax(dim=-1)
                cumprobs = probs.cumsum(dim=-1)
                mask = (cumprobs - probs) >= config.top_p
                probs[mask] = 0.0
                total = probs.sum(dim=-1, keepdim=True)
                if total.item() == 0.0:
                    probs[0, 0] = 1.0  # fallback to highest-prob token
                    total = probs.sum(dim=-1, keepdim=True)
                probs.div_(total)

                token_idx = torch.multinomial(probs, 1)
                next_token = top_idx.gather(-1, token_idx)

                token_id = next_token.item()
                generated_ids.append(token_id)
                current_input = next_token

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
