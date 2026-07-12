"""Native residual-post sparse-autoencoder training for unsupported models."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import hashlib
import math
from typing import Any

import torch


class _SaeCaptureComplete(RuntimeError):
    """Internal short-circuit after the selected residual layer is captured."""


def _token_rows(tokenizer: Any, documents: Sequence[str], seq_len: int) -> list[list[int]]:
    rows: list[list[int]] = []
    for document in documents:
        encoded = tokenizer(
            document,
            add_special_tokens=True,
            truncation=True,
            max_length=seq_len,
        )
        raw = encoded["input_ids"]
        if isinstance(raw, torch.Tensor):
            raw = raw.reshape(-1).tolist()
        ids = [int(token) for token in raw]
        if len(ids) >= 2:
            rows.append(ids)
    if not rows:
        raise ValueError("SAE training corpus has no documents with at least two tokens")
    return rows


def _padded_batch(
    rows: Sequence[list[int]],
    pad_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(row) for row in rows)
    ids = torch.full((len(rows), width), int(pad_id), dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.long)
    for idx, row in enumerate(rows):
        ids[idx, : len(row)] = torch.tensor(row, dtype=torch.long)
        mask[idx, : len(row)] = 1
    return ids.to(device), mask.to(device)


def _capture_layer_tokens(
    model: torch.nn.Module,
    layer: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    captured: list[torch.Tensor] = []

    def hook(_module: torch.nn.Module, _args: tuple[Any, ...], output: Any) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
            raise ValueError("SAE hook layer did not return a [batch, seq, hidden] tensor")
        captured.append(hidden.detach())
        raise _SaeCaptureComplete

    handle = layer.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            try:
                model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
            except _SaeCaptureComplete:
                pass
    finally:
        handle.remove()
    if len(captured) != 1:
        raise ValueError("SAE hook layer was not captured exactly once")
    hidden = captured[0]
    valid = attention_mask.to(device=hidden.device, dtype=torch.bool)
    return hidden[valid].to(dtype=torch.float32)


def train_residual_sae(
    model: torch.nn.Module,
    tokenizer: Any,
    layers: Sequence[torch.nn.Module],
    documents: Sequence[str],
    *,
    layer: int,
    tokens: int,
    seq_len: int = 128,
    batch_size: int = 8,
    d_sae: int | None = None,
    expansion: int = 8,
    learning_rate: float = 3e-4,
    l1_coefficient: float = 1e-3,
    dead_feature_threshold: float = 1e-6,
    seed: int = 0,
    on_progress: Callable[[str], None] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Train a one-layer ReLU SAE over transformer block outputs.

    Model forwards run under inference mode. Only the four SAE parameters have
    gradients, and decoder rows are renormalized after every optimizer step so
    the L1 objective cannot be evaded through encoder/decoder rescaling.
    """
    if not 0 <= layer < len(layers):
        raise ValueError(f"SAE layer {layer} is outside model layers 0..{len(layers) - 1}")
    if tokens <= 0 or seq_len <= 0 or batch_size <= 0 or expansion <= 0:
        raise ValueError("tokens, seq_len, batch_size, and expansion must be positive")
    if learning_rate <= 0 or l1_coefficient < 0 or dead_feature_threshold < 0:
        raise ValueError("invalid SAE optimizer/sparsity parameters")
    rows = _token_rows(tokenizer, documents, seq_len)
    first_parameter = next(model.parameters(), None)
    if first_parameter is None:
        raise ValueError("cannot train an SAE for a parameterless model")
    device = first_parameter.device
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(pad_id, (list, tuple)):
        pad_id = pad_id[0] if pad_id else None
    if pad_id is None:
        pad_id = 0

    first_ids, first_mask = _padded_batch(rows[:batch_size], int(pad_id), device)
    initial = _capture_layer_tokens(
        model,
        layers[layer],
        first_ids,
        first_mask,
    )
    d_model = int(initial.shape[-1])
    width = int(d_sae if d_sae is not None else d_model * expansion)
    if width <= 0:
        raise ValueError("SAE feature width must be positive")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    decoder_init = torch.randn(width, d_model, generator=generator)
    decoder_init = decoder_init / decoder_init.norm(dim=1, keepdim=True).clamp_min(1e-8)
    w_dec = torch.nn.Parameter(decoder_init.to(device))
    w_enc = torch.nn.Parameter(decoder_init.T.contiguous().to(device))
    b_enc = torch.nn.Parameter(torch.zeros(width, device=device))
    b_dec = torch.nn.Parameter(initial.mean(dim=0).to(device))
    optimizer = torch.optim.Adam((w_enc, w_dec, b_enc, b_dec), lr=learning_rate)

    feature_fires = torch.zeros(width, dtype=torch.int64)
    token_count = 0
    step = 0
    epoch = 0
    cursor = 0
    losses: list[float] = []
    mse_values: list[float] = []
    l1_values: list[float] = []
    while token_count < tokens:
        if cursor >= len(rows):
            cursor = 0
            epoch += 1
        batch_rows = rows[cursor : cursor + batch_size]
        cursor += len(batch_rows)
        ids, mask = _padded_batch(batch_rows, int(pad_id), device)
        activations = _capture_layer_tokens(model, layers[layer], ids, mask)
        remaining = tokens - token_count
        if activations.shape[0] > remaining:
            activations = activations[:remaining]

        features = torch.relu((activations - b_dec) @ w_enc + b_enc)
        reconstruction = features @ w_dec + b_dec
        mse = (reconstruction - activations).square().mean()
        sparsity = features.abs().mean()
        loss = mse + float(l1_coefficient) * sparsity
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            w_dec.div_(w_dec.norm(dim=1, keepdim=True).clamp_min(1e-8))
            feature_fires.add_((features.detach().cpu() > dead_feature_threshold).sum(dim=0))
        count = int(activations.shape[0])
        token_count += count
        step += 1
        losses.append(float(loss.detach().cpu()))
        mse_values.append(float(mse.detach().cpu()))
        l1_values.append(float(sparsity.detach().cpu()))
        if on_progress is not None and (step == 1 or token_count >= tokens or step % 25 == 0):
            on_progress(
                f"trained {token_count:,}/{tokens:,} tokens; mse={mse_values[-1]:.5g}, mean|f|={l1_values[-1]:.5g}"
            )

    tensors = {
        "W_enc": w_enc.detach().cpu(),
        "W_dec": w_dec.detach().cpu(),
        "b_enc": b_enc.detach().cpu(),
        "b_dec": b_dec.detach().cpu(),
    }
    metrics = {
        "tokens_trained": token_count,
        "steps": step,
        "epochs": epoch + 1,
        "d_model": d_model,
        "d_sae": width,
        "mean_loss": math.fsum(losses) / len(losses),
        "mean_mse": math.fsum(mse_values) / len(mse_values),
        "mean_feature_activation": math.fsum(l1_values) / len(l1_values),
        "dead_features": int((feature_fires == 0).sum()),
        "corpus_token_rows_sha256": hashlib.sha256(json_bytes := repr(rows).encode("utf-8")).hexdigest(),
        "corpus_token_rows_bytes": len(json_bytes),
    }
    return tensors, metrics
