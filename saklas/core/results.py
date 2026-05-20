from __future__ import annotations
import csv
import json
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class TokenAlt:
    """One alternative the model considered at a given position.

    Captured at decode time when ``SamplingConfig.return_top_k > 0`` (or
    when OpenAI ``logprobs`` is set to a positive int).  ``logprob`` is the
    post-sampler, post-temperature natural-log probability under the same
    distribution the chosen token was drawn from — that's the calibrated
    quantity for "how surprising was this token to the configured sampler."
    """
    id: int
    text: str
    logprob: float


@dataclass
class ProbeReadings:
    """Probe monitor readings across a generation run."""
    per_generation: list[float]
    mean: float
    std: float
    min: float
    max: float
    delta_per_gen: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GenerationResult:
    """Result of a generation call."""
    text: str
    tokens: list[int]
    token_count: int
    tok_per_sec: float
    elapsed: float
    readings: dict[str, ProbeReadings] = field(default_factory=dict)
    vectors: dict[str, float] = field(default_factory=dict)
    prompt_tokens: int = 0
    finish_reason: str = "stop"
    # Per-completion-token ``(token_id, logprob, top_alts)`` — populated
    # only when chosen-logprob capture is live (any ``on_token`` consumer
    # or an explicit logprobs request). ``top_alts`` is a list of
    # :class:`TokenAlt` carrying ``(id, text, logprob)`` triples; empty
    # list when no top-K alternatives were requested (``return_top_k == 0``
    # and OpenAI ``logprobs`` was set to 0). Inner-tuple shape replaces the
    # legacy ``list[tuple[int, float]]`` pair shape; renderers that consume
    # this field read ``alt.text`` directly rather than re-decoding.
    logprobs: list[tuple[int, float, list[TokenAlt]]] | None = None
    # Steering expression applied to this generation, stringified via
    # :func:`saklas.core.steering_expr.format_expr` for round-trip
    # reproduction.  ``None`` when no steering was active.  Receipts /
    # ``saklas replay`` land on this single field.
    applied_steering: str | None = None
    # Per-generated-token, per-layer residual-stream captures. ``None``
    # when SamplingConfig.return_hidden was False (default).  When set,
    # keyed by absolute layer index; each value is a ``[T, D]`` CPU
    # tensor where ``T == len(tokens)`` and ``D == hidden_size``. Dtype
    # matches the model's working dtype (fp16/bf16/fp32). ``to_dict``
    # deliberately omits this field — tensors don't serialize cleanly to
    # the JSON path; persist explicitly with ``torch.save`` if needed.
    hidden_states: dict[int, torch.Tensor] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "tokens": list(self.tokens),
            "token_count": self.token_count,
            "tok_per_sec": self.tok_per_sec,
            "elapsed": self.elapsed,
            "readings": {k: v.to_dict() for k, v in self.readings.items()},
            "vectors": dict(self.vectors),
            "prompt_tokens": self.prompt_tokens,
            "finish_reason": self.finish_reason,
            "applied_steering": self.applied_steering,
        }


class RunSet(list[GenerationResult]):
    """Ordered set of generation results plus experiment metadata.

    ``RunSet`` is intentionally list-like: existing batch/sweep callers
    can still iterate, index, and take ``len(...)``.  Single-run callers
    get one stable shape too, while ``.first`` and the small ``__getattr__``
    compatibility shim keep common ``session.generate(...).text`` code
    readable during the transition.
    """

    def __init__(
        self,
        results: Iterable[GenerationResult] = (),
        *,
        node_ids: Iterable[str | None] | None = None,
        grid: Iterable[dict[str, Any]] | None = None,
        kind: str = "generation",
    ) -> None:
        super().__init__(results)
        self.node_ids: list[str | None] = (
            list(node_ids) if node_ids is not None else [None] * len(self)
        )
        if len(self.node_ids) < len(self):
            self.node_ids.extend([None] * (len(self) - len(self.node_ids)))
        self.grid: list[dict[str, Any]] = (
            [dict(row) for row in grid] if grid is not None else [{} for _ in self]
        )
        if len(self.grid) < len(self):
            self.grid.extend({} for _ in range(len(self) - len(self.grid)))
        self.kind = kind

    @property
    def results(self) -> list[GenerationResult]:
        return list(self)

    @property
    def first(self) -> GenerationResult:
        if not self:
            raise IndexError("RunSet is empty")
        return self[0]

    @property
    def node_id(self) -> str | None:
        return self.node_ids[0] if self.node_ids else None

    def __getattr__(self, name: str) -> Any:
        """Delegate common single-run attribute access to ``.first``.

        This keeps old ``session.generate(...).text`` snippets working
        while the public shape settles on an always-run-set return.
        """
        try:
            first = self.first
        except IndexError as e:
            raise AttributeError(name) from e
        return getattr(first, name)

    def to_collector(self) -> "ResultCollector":
        collector = ResultCollector()
        for idx, result in enumerate(self):
            tags: dict[str, Any] = {"run_idx": idx}
            if idx < len(self.node_ids) and self.node_ids[idx] is not None:
                tags["node_id"] = self.node_ids[idx]
            if idx < len(self.grid):
                tags.update(self.grid[idx])
            collector.add(result, **tags)
        return collector

    def to_dataframe(self) -> Any:
        return self.to_collector().to_dataframe()

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "node_ids": list(self.node_ids),
            "grid": [dict(row) for row in self.grid],
            "results": [result.to_dict() for result in self],
        }


@dataclass
class TokenEvent:
    """Single token yielded during streaming generation."""
    text: str
    token_id: int
    index: int
    thinking: bool = False
    logprob: float | None = None
    # Top-K alternatives the model considered at this position. ``None``
    # when no top-K was requested (``return_top_k == 0`` and OpenAI
    # ``logprobs`` was None or 0). Replaces the legacy
    # ``top_logprobs: list[tuple[int, float]]`` pair shape — entries now
    # carry decoded text so consumers don't re-tokenize.
    top_alts: list[TokenAlt] | None = None
    finish_reason: str | None = None
    # Per-probe cosine similarities computed inline against the latest
    # captured hidden state. Populated by ``generate_stream`` only when
    # the session has active probes; otherwise None.
    scores: dict[str, float] | None = None
    # Perplexity of the configured sampler distribution after temperature,
    # top-k, and top-p renormalization — ``exp`` of Shannon entropy in nats.
    # Bounded above by sampler support size; a confident prediction
    # approaches 1. Consumers take ``log`` to recover entropy-nats.
    perplexity: float | None = None


class ResultCollector:
    """Accumulates GenerationResults with tags for batch export."""

    def __init__(self) -> None:
        self._rows: list[dict[str, Any]] = []

    @property
    def results(self) -> list[dict[str, Any]]:
        return list(self._rows)

    def add(self, result: GenerationResult, **tags: Any) -> None:
        row: dict[str, Any] = {
            "text": result.text,
            "token_count": result.token_count,
            "tok_per_sec": result.tok_per_sec,
            "elapsed": result.elapsed,
        }
        for probe_name, readings in result.readings.items():
            row[f"probe_{probe_name}_mean"] = readings.mean
            row[f"probe_{probe_name}_std"] = readings.std
            row[f"probe_{probe_name}_min"] = readings.min
            row[f"probe_{probe_name}_max"] = readings.max
            row[f"probe_{probe_name}_delta"] = readings.delta_per_gen
        for vec_name, alpha in result.vectors.items():
            row[f"vector_{vec_name}_alpha"] = alpha
        row.update(tags)
        self._rows.append(row)

    def to_jsonl(self, path: str) -> None:
        with open(path, "w") as f:
            for row in self._rows:
                f.write(json.dumps(row) + "\n")

    def to_csv(self, path: str) -> None:
        if not self._rows:
            return
        all_keys = list(dict.fromkeys(k for row in self._rows for k in row))
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(self._rows)

    def to_dataframe(self) -> Any:
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for to_dataframe(). "
                "Install with: pip install saklas[research]"
            )
        return pd.DataFrame(self._rows)
