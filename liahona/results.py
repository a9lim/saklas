from __future__ import annotations
import csv
import json
from dataclasses import dataclass, field


@dataclass
class ProbeReadings:
    """Probe monitor readings across a generation run."""
    per_token: list[float]
    mean: float
    std: float
    min: float
    max: float
    delta_per_tok: float

    def to_dict(self) -> dict:
        return {
            "per_token": list(self.per_token),
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "delta_per_tok": self.delta_per_tok,
        }


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

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "tokens": list(self.tokens),
            "token_count": self.token_count,
            "tok_per_sec": self.tok_per_sec,
            "elapsed": self.elapsed,
            "readings": {k: v.to_dict() for k, v in self.readings.items()},
            "vectors": dict(self.vectors),
        }


@dataclass
class TokenEvent:
    """Single token yielded during streaming generation."""
    text: str
    token_id: int
    index: int
    readings: dict[str, float] | None
    thinking: bool = False


class ResultCollector:
    """Accumulates GenerationResults with tags for batch export."""

    def __init__(self):
        self._rows: list[dict] = []

    @property
    def results(self) -> list[dict]:
        return list(self._rows)

    def add(self, result: GenerationResult, **tags) -> None:
        row = {
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
            row[f"probe_{probe_name}_delta"] = readings.delta_per_tok
        for vec_name, alpha in result.vectors.items():
            row[f"vector_{vec_name}_alpha"] = alpha
        row.update(tags)
        self._rows.append(row)

    def to_dicts(self) -> list[dict]:
        return list(self._rows)

    def to_jsonl(self, path: str) -> None:
        with open(path, "w") as f:
            for row in self._rows:
                f.write(json.dumps(row) + "\n")

    def to_csv(self, path: str) -> None:
        if not self._rows:
            return
        all_keys: list[str] = []
        seen: set[str] = set()
        for row in self._rows:
            for k in row:
                if k not in seen:
                    all_keys.append(k)
                    seen.add(k)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(self._rows)

    def to_dataframe(self):
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for to_dataframe(). "
                "Install with: pip install steer[pandas]"
            )
        return pd.DataFrame(self._rows)
