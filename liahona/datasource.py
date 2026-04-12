"""Multi-format contrastive pair normalizer."""
from __future__ import annotations
import csv
import json
from pathlib import Path


class DataSource:
    """Normalizes contrastive pairs from multiple input formats."""

    def __init__(self, pairs: list[tuple[str, str]], name: str = "custom",
                 description: str | None = None):
        self.pairs = pairs
        self.name = name
        self.description = description

    @classmethod
    def from_pairs(cls, pairs: list[tuple[str, str]], name: str = "custom",
                   description: str | None = None) -> DataSource:
        return cls(pairs=list(pairs), name=name, description=description)

    @classmethod
    def curated(cls, concept: str) -> DataSource:
        datasets_dir = Path(__file__).parent / "datasets"
        ds_path = datasets_dir / f"{concept.lower()}.json"
        if not ds_path.exists():
            raise FileNotFoundError(
                f"No curated dataset for '{concept}'. "
                f"Available: {', '.join(p.stem for p in sorted(datasets_dir.glob('*.json')))}"
            )
        with open(ds_path) as f:
            data = json.load(f)
        pairs = [(p["positive"], p["negative"]) for p in data["pairs"]]
        return cls(pairs=pairs, name=data.get("name", concept), description=data.get("description"))

    @classmethod
    def json(cls, path: str, name: str | None = None) -> DataSource:
        with open(path) as f:
            data = json.load(f)
        pairs = [(p["positive"], p["negative"]) for p in data["pairs"]]
        return cls(pairs=pairs, name=name or data.get("name", Path(path).stem),
                   description=data.get("description"))

    @classmethod
    def csv(cls, path: str, positive_col: str = "positive",
            negative_col: str = "negative", name: str | None = None) -> DataSource:
        pairs = []
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pairs.append((row[positive_col], row[negative_col]))
        return cls(pairs=pairs, name=name or Path(path).stem)

    @classmethod
    def huggingface(cls, dataset_id: str, positive_col: str = "positive",
                    negative_col: str = "negative", split: str = "train",
                    name: str | None = None) -> DataSource:
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "The 'datasets' package is required for DataSource.huggingface(). "
                "Install with: pip install liahona-ai[research]"
            )
        ds = load_dataset(dataset_id, split=split)
        pairs = [(row[positive_col], row[negative_col]) for row in ds]
        return cls(pairs=pairs, name=name or dataset_id.split("/")[-1])
