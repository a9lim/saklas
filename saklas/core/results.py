from __future__ import annotations
import csv
import json
from dataclasses import asdict, dataclass, field
from typing import Any


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
    # Per-completion-token (token_id, logprob, top_logprobs) — populated
    # only when logprobs were requested. top_logprobs is list[(id, logprob)].
    logprobs: list[tuple[int, float, list[tuple[int, float]]]] | None = None
    # Steering expression applied to this generation, stringified via
    # :func:`saklas.core.steering_expr.format_expr` for round-trip
    # reproduction.  ``None`` when no steering was active.  Receipts /
    # ``saklas replay`` land on this single field.
    applied_steering: str | None = None

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


@dataclass
class TokenEvent:
    """Single token yielded during streaming generation."""
    text: str
    token_id: int
    index: int
    thinking: bool = False
    logprob: float | None = None
    top_logprobs: list[tuple[int, float]] | None = None
    finish_reason: str | None = None
    # Per-probe cosine similarities computed inline against the latest
    # captured hidden state. Populated by ``generate_stream`` only when
    # the session has active probes; otherwise None.
    scores: dict[str, float] | None = None
    # Perplexity of the pre-temperature, post-steering next-token
    # distribution — ``exp`` of full-vocab Shannon entropy in nats.
    # Bounded above by ``vocab_size``; a confident prediction approaches
    # 1. Consumers take ``log`` to recover entropy-nats for averaging.
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
