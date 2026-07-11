"""Generation result finalization for :class:`SaklasSession`."""

from __future__ import annotations

from typing import Any

import torch

from saklas.core.events import ProbeScored
from saklas.core.results import GenerationResult, ProbeReading, ProbeReadings


def finalize_generation(
    session: Any,
    input: Any,
    generated_ids: list[int],
    elapsed: float,
    vector_snapshot: dict[str, float],
    prompt_tokens: int = 0,
    stateless: bool = False,
    logprobs_list: list[tuple[int, float, list[Any]]] | None = None,
    applied_steering: str | None = None,
    *,
    return_hidden: bool = False,
    return_probe_readings: bool = True,
    assistant_node_id: str | None = None,
    mean_logprob: float | None = None,
    mean_surprise: float | None = None,
    min_elapsed_for_rate: float = 1e-9,
) -> GenerationResult:
    """Decode, score probes, update session side effects, and build a result."""
    del input  # kept in the public wrapper signature for compatibility

    token_count = len(generated_ids)
    tok_per_sec = token_count / elapsed if elapsed > min_elapsed_for_rate else 0.0
    response_ids = generated_ids[session._gen_state.thinking_end_idx:]
    if (
        session._gen_state.finish_reason == "stop_sequence"
        and session._gen_state.response_text is not None
    ):
        text: str = session._gen_state.response_text
    else:
        decoded = session._tokenizer.decode(response_ids, skip_special_tokens=True)
        text = decoded if isinstance(decoded, str) else decoded[0]

    capture_mode = session._capture_state.mode
    capture_mode_name = getattr(capture_mode, "name", "")
    captured_stack: dict[int, torch.Tensor] = {}
    if (
        generated_ids
        and (
            return_hidden
            or (
                return_probe_readings
                and session._monitor.probe_names
                and capture_mode_name == "FULL"
            )
        )
    ):
        captured_stack = session._capture.stacked()

    agg_vals: dict[str, ProbeReading] = {}
    if return_probe_readings and session._monitor.probe_names and generated_ids:
        if capture_mode_name == "INCREMENTAL":
            agg_vals, per_token = session._score_incremental(
                generated_ids, accumulate=not stateless,
            )
        elif capture_mode_name == "LEAN_INCREMENTAL":
            agg_vals, per_token = session._score_lean_incremental(
                generated_ids, accumulate=not stateless,
            )
        elif session._capture_state.aggregate_only:
            agg_vals = session._score_aggregate_only(
                generated_ids, accumulate=not stateless,
            )
            per_token = {}
        else:
            if captured_stack:
                aggregate_index = session._aggregate_forward_index(generated_ids)
                agg_vals, per_token = session._monitor.score_per_token(
                    captured_stack, generated_ids, session._tokenizer,
                    accumulate=not stateless,
                    aggregate_index=(
                        -1 if aggregate_index is None else aggregate_index
                    ),
                )
            else:
                agg_vals, per_token = {}, {}
        session._last_per_token_scores = per_token or None
        if stateless:
            readings = {}
            for name, reading in agg_vals.items():
                coords = reading.coords or (0.0,)
                zeros = tuple(0.0 for _ in coords)
                readings[name] = ProbeReadings(
                    per_generation=[coords], mean=coords, std=zeros,
                    min=coords, max=coords, delta_per_gen=zeros,
                )
        else:
            readings = session.build_readings()
    else:
        session._last_per_token_scores = None
        readings = {} if stateless else session.build_readings()

    hidden_states: dict[int, torch.Tensor] | None = None
    if return_hidden and generated_ids and captured_stack:
        n = len(generated_ids)
        trimmed: dict[int, torch.Tensor] = {}
        for layer_idx, hidden in captured_stack.items():
            if hidden.shape[0] > n:
                hidden = hidden[:n]
            elif hidden.shape[0] < n:
                continue
            trimmed[layer_idx] = hidden.detach().to("cpu")
        hidden_states = trimmed

    manifold_aggregates: dict[str, Any] = {}
    if return_probe_readings and session._monitor.probe_names and generated_ids:
        manifold_aggregates = dict(agg_vals)
    # Pinned J-lens token probes (readout channel — not monitor probes):
    # one band readout pooled at the last content token, same aggregate
    # semantics as the monitor roster.
    if (
        return_probe_readings
        and generated_ids
        and getattr(session, "_lens_probes", None)
    ):
        manifold_aggregates.update(
            session._score_lens_probes_aggregate(generated_ids)
        )
    if (
        return_probe_readings
        and generated_ids
        and getattr(session, "_sae_probes", None)
    ):
        manifold_aggregates.update(
            session._score_sae_probes_aggregate(generated_ids)
        )

    result = GenerationResult(
        text=text, tokens=list(generated_ids), token_count=token_count,
        tok_per_sec=tok_per_sec, elapsed=elapsed,
        readings=readings, vectors=vector_snapshot,
        prompt_tokens=prompt_tokens,
        finish_reason=session._gen_state.finish_reason,
        logprobs=logprobs_list,
        applied_steering=applied_steering,
        hidden_states=hidden_states,
        probe_readings=manifold_aggregates,
    )
    session._last_result = result

    if readings:
        scalar_readings = {
            name: (reading.mean[0] if reading.mean else 0.0)
            for name, reading in readings.items()
        }
        session.events.emit(ProbeScored(readings=scalar_readings))

    if not stateless and assistant_node_id is not None:
        session._stamp_raw_indices(assistant_node_id)
        # Decoded thinking-channel text, joined from the streamed token
        # rows — stamped so history re-renders can carry the block
        # through the family's think delimiters (the stitcher applies
        # the family's history policy; strip families render it only
        # while the turn is last).  Stamped ONLY when the scene grammar
        # can actually re-render it (``think_open`` set): on a family
        # whose thinking isn't delimiter-shaped (gpt-oss channels) or
        # that fell back to template rendering, a stamped block would
        # make every later render of this path raise.
        thinking_text: str | None = None
        grammar = getattr(session, "scene_grammar", None)
        if grammar is not None and grammar.think_open is not None:
            node = session.tree.nodes.get(assistant_node_id)
            if node is not None and node.thinking_tokens:
                joined = "".join(
                    str(t.get("text", "")) for t in node.thinking_tokens
                )
                thinking_text = joined or None
        session.tree.finalize_assistant(
            assistant_node_id,
            text=text,
            aggregate_readings={
                name: (reading.mean[0] if reading.mean else 0.0)
                for name, reading in readings.items()
            },
            applied_steering=applied_steering,
            finish_reason=session._gen_state.finish_reason,
            mean_logprob=mean_logprob,
            mean_surprise=mean_surprise,
            raw_token_ids=generated_ids,
            thinking_text=thinking_text,
        )

    return result
