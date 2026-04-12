"""Multi-format contrastive pair normalizer."""
from __future__ import annotations
import csv
import json
from pathlib import Path


class DataSource:
    """Normalizes contrastive pairs from multiple input formats."""

    def __init__(self, pairs: list[tuple[str, str]], name: str = "custom"):
        self.pairs = pairs
        self.name = name

    @classmethod
    def _from_json_file(cls, path, name_override: str | None = None) -> DataSource:
        with open(path) as f:
            data = json.load(f)
        pairs = [(p["positive"], p["negative"]) for p in data["pairs"]]
        name = name_override or data.get("name", Path(path).stem)
        return cls(pairs=pairs, name=name)

    @classmethod
    def curated(cls, concept: str) -> DataSource:
        datasets_dir = Path(__file__).parent / "datasets"
        ds_path = datasets_dir / f"{concept.lower()}.json"
        if not ds_path.exists():
            raise FileNotFoundError(
                f"No curated dataset for '{concept}'. "
                f"Available: {', '.join(p.stem for p in sorted(datasets_dir.glob('*.json')))}"
            )
        return cls._from_json_file(ds_path, name_override=concept)

    @classmethod
    def json(cls, path: str, name: str | None = None) -> DataSource:
        return cls._from_json_file(path, name_override=name)

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
