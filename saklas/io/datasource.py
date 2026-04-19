"""Multi-format contrastive pair normalizer."""
from __future__ import annotations
import csv
import json
from pathlib import Path
from typing import Any, cast


class DataSource:
    """Normalizes contrastive pairs from multiple input formats."""

    def __init__(self, pairs: list[tuple[str, str]], name: str = "custom") -> None:
        self.pairs = pairs
        self.name = name

    @classmethod
    def _from_json_file(
        cls, path: str | Path, name_override: str | None = None,
    ) -> DataSource:
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            pairs = [(p["positive"], p["negative"]) for p in data]
            name = name_override or Path(path).stem
        else:
            pairs = [(p["positive"], p["negative"]) for p in data["pairs"]]
            name = name_override or data.get("name", Path(path).stem)
        return cls(pairs=pairs, name=name)

    @classmethod
    def curated(cls, concept: str) -> DataSource:
        """Load the bundled 'default/<concept>' statements.json.

        Triggers first-run materialization of bundled data into ~/.saklas/
        and reads from there so users can edit the statements freely.
        """
        from saklas.io.paths import concept_dir

        name = concept.lower()
        folder = concept_dir("default", name)
        ds_path = folder / "statements.json"

        # Short-circuit: if the concept is already materialized in the user
        # cache, skip materialize_bundled() and the directory walk entirely.
        if ds_path.exists():
            return cls._from_json_file(ds_path, name_override=concept)

        from saklas.io.packs import materialize_bundled
        materialize_bundled()
        if not ds_path.exists():
            default_root = folder.parent
            available = sorted(
                p.name for p in default_root.iterdir() if p.is_dir()
            ) if default_root.exists() else []
            raise FileNotFoundError(
                f"No curated dataset for '{concept}'. "
                f"Available: {', '.join(available)}"
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
                "Install with: pip install saklas[research]"
            )
        ds = load_dataset(dataset_id, split=split)
        pairs = [
            (cast(Any, row)[positive_col], cast(Any, row)[negative_col])
            for row in ds
        ]
        return cls(pairs=pairs, name=name or dataset_id.split("/")[-1])
