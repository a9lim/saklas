"""Tests for DataSource multi-format contrastive pair normalizer."""
import json
import csv
import tempfile
from pathlib import Path
from steer.datasource import DataSource

class TestFromPairs:
    def test_basic(self):
        ds = DataSource.from_pairs([("hello", "goodbye")])
        assert ds.pairs == [("hello", "goodbye")]
        assert ds.name == "custom"

    def test_custom_name(self):
        ds = DataSource.from_pairs([("a", "b")], name="test")
        assert ds.name == "test"

class TestCurated:
    def test_loads_happy(self):
        ds = DataSource.curated("happy")
        assert len(ds.pairs) > 0
        assert ds.name == "happy"
        for pos, neg in ds.pairs:
            assert isinstance(pos, str)
            assert isinstance(neg, str)

    def test_missing_raises(self):
        import pytest
        with pytest.raises(FileNotFoundError):
            DataSource.curated("nonexistent_concept_xyz")

class TestJson:
    def test_loads_steer_format(self):
        data = {
            "name": "test", "description": "test concept", "category": "test",
            "pairs": [
                {"positive": "I am happy", "negative": "I am sad"},
                {"positive": "Joy", "negative": "Sorrow"},
            ],
        }
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump(data, f)
            path = f.name
        ds = DataSource.json(path)
        assert len(ds.pairs) == 2
        assert ds.pairs[0] == ("I am happy", "I am sad")
        assert ds.name == "test"
        Path(path).unlink()

class TestCsv:
    def test_default_columns(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False, newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["positive", "negative"])
            writer.writerow(["happy", "sad"])
            writer.writerow(["joyful", "gloomy"])
            path = f.name
        ds = DataSource.csv(path)
        assert len(ds.pairs) == 2
        assert ds.pairs[0] == ("happy", "sad")
        Path(path).unlink()

    def test_custom_columns(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False, newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["good", "bad"])
            writer.writerow(["nice", "mean"])
            path = f.name
        ds = DataSource.csv(path, positive_col="good", negative_col="bad")
        assert ds.pairs[0] == ("nice", "mean")
        Path(path).unlink()

    def test_name_inferred_from_filename(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False, newline="",
                                         prefix="empathy_") as f:
            writer = csv.writer(f)
            writer.writerow(["positive", "negative"])
            writer.writerow(["a", "b"])
            path = f.name
        ds = DataSource.csv(path)
        assert ds.name == Path(path).stem
        Path(path).unlink()

class TestHuggingFace:
    def test_import_error_without_datasets(self):
        try:
            import datasets
        except ImportError:
            import pytest
            with pytest.raises(ImportError, match="datasets"):
                DataSource.huggingface("some/dataset")
