"""Tests for structured output dataclasses and monitor scoring."""
import json
import csv
import tempfile
from pathlib import Path

import torch
from saklas.core.results import ProbeReadings, GenerationResult, TokenEvent, ResultCollector


class TestPublicAPI:
    def test_imports(self):
        from saklas import SaklasSession, DataSource, ResultCollector
        from saklas import GenerationResult, TokenEvent, ProbeReadings
        assert SaklasSession is not None
        assert DataSource is not None
        assert ResultCollector is not None
        assert GenerationResult is not None
        assert TokenEvent is not None
        assert ProbeReadings is not None


class TestProbeReadings:
    def test_from_per_generation_data(self):
        readings = ProbeReadings(
            per_generation=[0.1, 0.3, 0.5, 0.4],
            mean=0.325, std=0.1479, min=0.1, max=0.5, delta_per_gen=0.1,
        )
        assert readings.mean == 0.325
        assert readings.min == 0.1
        assert len(readings.per_generation) == 4

    def test_to_dict_returns_plain_types(self):
        readings = ProbeReadings(
            per_generation=[0.1, 0.2], mean=0.15, std=0.05, min=0.1, max=0.2, delta_per_gen=0.1,
        )
        d = readings.to_dict()
        assert isinstance(d, dict)
        assert isinstance(d["per_generation"], list)
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
            per_generation=[0.5], mean=0.5, std=0.0, min=0.5, max=0.5, delta_per_gen=0.0,
        )
        result = GenerationResult(
            text="Hi", tokens=[1], token_count=1, tok_per_sec=5.0, elapsed=0.2,
            readings={"honest": readings}, vectors={},
        )
        d = result.to_dict()
        assert "honest" in d["readings"]
        assert isinstance(d["readings"]["honest"], dict)

    def test_applied_steering_default_none(self):
        """Default value is ``None`` — no steering was active."""
        result = GenerationResult(
            text="Hi", tokens=[1], token_count=1, tok_per_sec=5.0, elapsed=0.2,
        )
        assert result.applied_steering is None
        assert result.to_dict()["applied_steering"] is None

    def test_applied_steering_round_trip_expression(self, monkeypatch, tmp_path):
        """Stored expression round-trips through ``parse_expr``."""
        monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
        from saklas.cli import selectors as _sel
        _sel.invalidate()
        from saklas.core.steering_expr import parse_expr
        result = GenerationResult(
            text="Hi", tokens=[1], token_count=1, tok_per_sec=5.0, elapsed=0.2,
            applied_steering="0.5 myvec + 0.3 othervec@after",
        )
        assert result.applied_steering == "0.5 myvec + 0.3 othervec@after"
        reparsed = parse_expr(result.applied_steering)
        assert "myvec" in reparsed.alphas
        assert "othervec" in reparsed.alphas


class TestTokenEvent:
    def test_fields(self):
        event = TokenEvent(text="hello", token_id=42, index=0)
        assert event.text == "hello"
        assert event.token_id == 42
        assert event.index == 0

    def test_thinking_flag(self):
        event = TokenEvent(text="hi", token_id=1, index=0, thinking=True)
        assert event.thinking is True

    def test_optional_fields_default_none(self):
        """All extension fields default to ``None`` / unset — old callers unaffected."""
        event = TokenEvent(text="x", token_id=0, index=0)
        assert event.logprob is None
        assert event.top_logprobs is None
        assert event.finish_reason is None
        assert event.scores is None
        assert event.perplexity is None

    def test_logprobs_carried_through(self):
        """Populated logprob fields land on the dataclass as given."""
        event = TokenEvent(
            text="x", token_id=0, index=0,
            logprob=-1.5, top_logprobs=[(0, -0.2), (1, -2.3)],
        )
        assert event.logprob == -1.5
        assert event.top_logprobs == [(0, -0.2), (1, -2.3)]

    def test_scores_and_perplexity(self):
        event = TokenEvent(
            text="x", token_id=0, index=0,
            scores={"happy": 0.4, "sad": -0.1}, perplexity=12.7,
        )
        assert event.scores == {"happy": 0.4, "sad": -0.1}
        assert event.perplexity == 12.7

    def test_finish_reason_set(self):
        event = TokenEvent(text="", token_id=0, index=5, finish_reason="stop")
        assert event.finish_reason == "stop"


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
            per_generation=[0.5], mean=0.5, std=0.0, min=0.5, max=0.5, delta_per_gen=0.0,
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
            import pandas  # noqa: F401
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

    def test_to_csv_empty_is_noop(self, tmp_path):
        """Empty collector returns early — no file written, no crash."""
        collector = ResultCollector()
        path = tmp_path / "out.csv"
        collector.to_csv(str(path))
        assert not path.exists()

    def test_results_property_returns_copy(self):
        """``.results`` returns a copy — mutation doesn't bleed back."""
        collector = ResultCollector()
        collector.add(self._make_result())
        rows = collector.results
        rows.append({"injected": True})
        assert len(collector.results) == 1


class TestTraitMonitorScoring:
    """Tests for TraitMonitor probe scoring — runs anywhere (no GPU)."""

    @staticmethod
    def _make_monitor():
        from saklas.core.monitor import TraitMonitor
        dim = 16
        # Create a probe vector pointing in a known direction
        probe_vec = torch.zeros(dim)
        probe_vec[0] = 1.0  # unit vector along dim 0
        probe_profile = {0: probe_vec}
        layer_means = {0: torch.zeros(dim)}
        return TraitMonitor({"test_probe": probe_profile}, layer_means)

    def test_measure_from_hidden_matches_manual(self):
        monitor = self._make_monitor()
        # Hidden state pointing in the same direction as probe
        h = torch.zeros(16)
        h[0] = 5.0
        monitor.measure_from_hidden({0: h})
        assert len(monitor.history["test_probe"]) == 1
        # Cosine similarity of aligned vectors = 1.0
        assert abs(monitor.history["test_probe"][0] - 1.0) < 1e-5

    def test_measure_from_hidden_orthogonal(self):
        monitor = self._make_monitor()
        # Hidden state orthogonal to probe — cosine = 0
        h = torch.zeros(16)
        h[1] = 5.0
        monitor.measure_from_hidden({0: h})
        assert abs(monitor.history["test_probe"][0]) < 1e-5

    def test_history_accumulates_across_calls(self):
        monitor = self._make_monitor()
        h1 = torch.zeros(16)
        h1[0] = 1.0
        h2 = torch.zeros(16)
        h2[1] = 1.0
        monitor.measure_from_hidden({0: h1})
        monitor.measure_from_hidden({0: h2})
        assert len(monitor.history["test_probe"]) == 2
        assert monitor.history["test_probe"][0] > monitor.history["test_probe"][1]

    def test_stats_accumulate(self):
        monitor = self._make_monitor()
        for _ in range(3):
            h = torch.randn(16)
            monitor.measure_from_hidden({0: h})
        stats = monitor.get_stats("test_probe")
        assert stats["count"] == 3
        assert stats["min"] <= stats["max"]

    def test_sparkline_grows_with_history(self):
        monitor = self._make_monitor()
        for _ in range(4):
            h = torch.randn(16)
            monitor.measure_from_hidden({0: h})
        sparkline = monitor.get_sparkline("test_probe")
        assert len(sparkline) == 4
