"""Tests for structured output dataclasses."""
import json
import csv
import tempfile
from pathlib import Path
from liahona.results import ProbeReadings, GenerationResult, TokenEvent, ResultCollector


class TestPublicAPI:
    def test_imports(self):
        from liahona import LiahonaSession, DataSource, ResultCollector
        from liahona import GenerationResult, TokenEvent, ProbeReadings
        assert LiahonaSession is not None
        assert DataSource is not None
        assert ResultCollector is not None
        assert GenerationResult is not None
        assert TokenEvent is not None
        assert ProbeReadings is not None


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
        event = TokenEvent(text="hello", token_id=42, index=0)
        assert event.text == "hello"
        assert event.token_id == 42
        assert event.index == 0

    def test_thinking_flag(self):
        event = TokenEvent(text="hi", token_id=1, index=0, thinking=True)
        assert event.thinking is True


class TestResultCollector:
    def _make_result(self, text="Hello", alpha=1.0):
        return GenerationResult(
            text=text, tokens=[1, 2], token_count=2,
            tok_per_sec=10.0, elapsed=0.2, readings={}, vectors={"happy": alpha},
        )

    def test_add_and_to_dicts(self):
        collector = ResultCollector()
        collector.add(self._make_result(), prompt="hi", concept="happy")
        dicts = collector.results
        assert len(dicts) == 1
        assert dicts[0]["text"] == "Hello"
        assert dicts[0]["prompt"] == "hi"
        assert dicts[0]["vector_happy_alpha"] == 1.0

    def test_probe_readings_flattened(self):
        readings = ProbeReadings(
            per_token=[0.5], mean=0.5, std=0.0, min=0.5, max=0.5, delta_per_tok=0.0,
        )
        result = GenerationResult(
            text="Hi", tokens=[1], token_count=1, tok_per_sec=5.0, elapsed=0.2,
            readings={"honest": readings}, vectors={},
        )
        collector = ResultCollector()
        collector.add(result)
        d = collector.results[0]
        assert d["probe_honest_mean"] == 0.5
        assert d["probe_honest_std"] == 0.0
        assert d["probe_honest_min"] == 0.5
        assert d["probe_honest_max"] == 0.5

    def test_to_jsonl(self):
        collector = ResultCollector()
        collector.add(self._make_result())
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        collector.to_jsonl(path)
        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["text"] == "Hello"
        Path(path).unlink()

    def test_to_csv(self):
        collector = ResultCollector()
        collector.add(self._make_result(alpha=1.0), concept="happy")
        collector.add(self._make_result(text="World", alpha=2.0), concept="happy")
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        collector.to_csv(path)
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2
        assert rows[0]["text"] == "Hello"
        assert rows[1]["text"] == "World"
        Path(path).unlink()

    def test_to_dataframe_without_pandas(self):
        collector = ResultCollector()
        collector.add(self._make_result())
        try:
            import pandas
            df = collector.to_dataframe()
            assert len(df) == 1
        except ImportError:
            import pytest
            with pytest.raises(ImportError):
                collector.to_dataframe()

    def test_results_property(self):
        collector = ResultCollector()
        collector.add(self._make_result(), run=1)
        collector.add(self._make_result(), run=2)
        assert len(collector.results) == 2
