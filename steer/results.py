from __future__ import annotations
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
