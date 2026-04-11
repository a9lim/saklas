"""Tests for structured output dataclasses."""
from steer.results import ProbeReadings, GenerationResult, TokenEvent


class TestProbeReadings:
    def test_from_per_token_data(self):
        readings = ProbeReadings(
            per_token=[0.1, 0.3, 0.5, 0.4],
            mean=0.325, std=0.1479, min=0.1, max=0.5, delta_per_tok=0.1,
        )
        assert readings.mean == 0.325
        assert readings.min == 0.1
        assert len(readings.per_token) == 4

    def test_to_dict_returns_plain_types(self):
        readings = ProbeReadings(
            per_token=[0.1, 0.2], mean=0.15, std=0.05, min=0.1, max=0.2, delta_per_tok=0.1,
        )
        d = readings.to_dict()
        assert isinstance(d, dict)
        assert isinstance(d["per_token"], list)
        assert isinstance(d["mean"], float)


class TestGenerationResult:
    def test_to_dict_no_probes(self):
        result = GenerationResult(
            text="Hello world", tokens=[1, 2, 3], token_count=3,
            tok_per_sec=10.0, elapsed=0.3, readings={}, vectors={"happy": 1.5},
        )
        d = result.to_dict()
        assert d["text"] == "Hello world"
        assert d["tokens"] == [1, 2, 3]
        assert d["readings"] == {}
        assert d["vectors"] == {"happy": 1.5}

    def test_to_dict_with_probes(self):
        readings = ProbeReadings(
            per_token=[0.5], mean=0.5, std=0.0, min=0.5, max=0.5, delta_per_tok=0.0,
        )
        result = GenerationResult(
            text="Hi", tokens=[1], token_count=1, tok_per_sec=5.0, elapsed=0.2,
            readings={"honest": readings}, vectors={},
        )
        d = result.to_dict()
        assert "honest" in d["readings"]
        assert isinstance(d["readings"]["honest"], dict)


class TestTokenEvent:
    def test_fields(self):
        event = TokenEvent(text="hello", token_id=42, index=0, readings={"honest": 0.5})
        assert event.text == "hello"
        assert event.token_id == 42
        assert event.readings["honest"] == 0.5

    def test_no_readings(self):
        event = TokenEvent(text="hi", token_id=1, index=0, readings=None)
        assert event.readings is None
