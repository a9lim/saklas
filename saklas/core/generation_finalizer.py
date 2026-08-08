"""Generation result finalization for :class:`SaklasSession`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from saklas.core.capture import CaptureMode
from saklas.core.events import ProbeScored
from saklas.core.instruments.types import (
    ScalarReading,
    as_probe_reading,
    reading_axis0,
)
from saklas.core.measurements import build_measurements
from saklas.core.results import GenerationResult, ProbeReading

if TYPE_CHECKING:
    from saklas.core.session import SaklasSession


def finalize_generation(
    session: "SaklasSession",
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
    captured_stack: dict[int, torch.Tensor] = {}
    if (
        generated_ids
        and (
            return_hidden
            or (
                return_probe_readings
                and session._monitor.probe_names
                and capture_mode is CaptureMode.FULL
            )
        )
    ):
        captured_stack = session._capture.stacked()

    aggregate_index_computed = False
    aggregate_index: int | None = None
    aggregate_tail_computed = False
    aggregate_tail: dict[int, torch.Tensor] = {}
    aggregate_stack_computed = False
    aggregate_stack: dict[int, torch.Tensor] = {}

    def _aggregate_index() -> int | None:
        nonlocal aggregate_index_computed, aggregate_index
        if not aggregate_index_computed:
            aggregate_index = session._aggregate_forward_index(generated_ids)
            aggregate_index_computed = True
        return aggregate_index

    def _aggregate_tail_slice() -> dict[int, torch.Tensor]:
        nonlocal aggregate_tail_computed, aggregate_tail
        if aggregate_tail_computed:
            return aggregate_tail
        aggregate_tail_computed = True
        agg_fwd = _aggregate_index()
        aggregate_tail = (
            session._capture.tail_slice_at(agg_fwd)
            if agg_fwd is not None else {}
        )
        return aggregate_tail

    def _aggregate_pooled_slice() -> dict[int, torch.Tensor]:
        nonlocal aggregate_stack_computed, aggregate_stack
        tail = _aggregate_tail_slice()
        if tail:
            return tail
        agg_fwd = _aggregate_index()
        if agg_fwd is None:
            return {}
        if not aggregate_stack_computed:
            stack = captured_stack or session._capture.stacked()
            aggregate_stack = {
                layer: rows[agg_fwd]
                for layer, rows in stack.items()
                if rows.shape[0] > agg_fwd
            }
            aggregate_stack_computed = True
        return aggregate_stack

    agg_vals: dict[str, ProbeReading] = {}
    if return_probe_readings and session._monitor.probe_names and generated_ids:
        if capture_mode is CaptureMode.INCREMENTAL:
            agg_vals, per_token = session._score_incremental(
                generated_ids, accumulate=not stateless,
            )
        elif capture_mode is CaptureMode.LEAN_INCREMENTAL:
            agg_vals, per_token = session._score_lean_incremental(
                generated_ids,
                accumulate=not stateless,
                pooled=_aggregate_tail_slice(),
            )
        elif session._capture_state.aggregate_only:
            agg_vals = session._score_aggregate_only(
                generated_ids,
                accumulate=not stateless,
                pooled=_aggregate_tail_slice(),
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
    else:
        session._last_per_token_scores = None

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

    geometry_aggregates: dict[str, ProbeReading] = {}
    if return_probe_readings and session._monitor.probe_names and generated_ids:
        geometry_aggregates = dict(agg_vals)
    # Pinned J-lens token probes (readout channel — not monitor probes):
    # one band readout pooled at the last content token, same aggregate
    # semantics as the monitor roster.  Their native ``ScalarReading``s stay
    # native all the way into the envelope; the cross-family
    # ``probe_readings`` dict below is the one projection.
    lens_aggregates: dict[str, ScalarReading] = {}
    if (
        return_probe_readings
        and generated_ids
        and session.lens.names
    ):
        lens_aggregates = session._score_lens_probes_aggregate(
            generated_ids, pooled=_aggregate_pooled_slice(),
        )
    sae_aggregates: dict[str, ScalarReading] = {}
    if (
        return_probe_readings
        and generated_ids
        and session.sae.names
    ):
        sae_aggregates = session._score_sae_probes_aggregate(
            generated_ids, pooled=_aggregate_pooled_slice(),
        )

    manifold_aggregates: dict[str, Any] = {
        **geometry_aggregates, **lens_aggregates, **sae_aggregates,
    }
    live_lens = session.lens.live
    live_sae = session.sae.live
    measurement_envelope = build_measurements(
        scope="aggregate",
        geometry_readings=geometry_aggregates or None,
        lens_readings=lens_aggregates or None,
        sae_readings=sae_aggregates or None,
        lens_source=(
            live_lens.get("source")
            if lens_aggregates and isinstance(live_lens, dict) else None
        ),
        sae_source=(
            live_sae.get("source")
            if sae_aggregates and isinstance(live_sae, dict) else None
        ),
        sae_layer=(
            live_sae.get("layer")
            if sae_aggregates and isinstance(live_sae, dict) else None
        ),
        steering=applied_steering,
    )
    # ``Measurements`` is a TypedDict describing the wire shape, while the
    # established ``GenerationResult`` compatibility field is a plain mapping.
    # Materialize the latter at this boundary without changing the envelope.
    measurements = (
        dict(measurement_envelope)
        if measurement_envelope is not None else None
    )

    result = GenerationResult(
        text=text, tokens=list(generated_ids), token_count=token_count,
        tok_per_sec=tok_per_sec, elapsed=elapsed,
        steering_alphas=vector_snapshot,
        prompt_tokens=prompt_tokens,
        finish_reason=session._gen_state.finish_reason,
        logprobs=logprobs_list,
        applied_steering=applied_steering,
        hidden_states=hidden_states,
        # The compatibility dict: one reading type across families, for the
        # vendor extension and cross-family callers.
        probe_readings={
            name: as_probe_reading(reading)
            for name, reading in manifold_aggregates.items()
        },
        measurements=measurements,
    )
    session._last_result = result

    if manifold_aggregates:
        session.events.emit(ProbeScored(readings={
            name: reading_axis0(reading)
            for name, reading in manifold_aggregates.items()
        }))

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
        grammar = session.scene_grammar
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
                name: reading_axis0(reading)
                for name, reading in manifold_aggregates.items()
            },
            applied_steering=applied_steering,
            finish_reason=session._gen_state.finish_reason,
            mean_logprob=mean_logprob,
            mean_surprise=mean_surprise,
            raw_token_ids=generated_ids,
            thinking_text=thinking_text,
        )

    return result
