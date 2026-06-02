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
        """Load the bundled 'default/<concept>' contrastive pairs.

        Triggers first-run materialization of bundled data into ~/.saklas/
        and reads from there so users can edit the statements freely.  A
        steering vector now ships as a 2-node ``pca`` manifold (4.0); when no
        ``vectors/`` ``statements.json`` exists the pairs are reconstructed
        from the manifold's two node corpora (node 0 = positive pole, node 1 =
        negative).  The pairing is arbitrary — DiM only needs the two
        centroids — so the re-fit vector is identical either way.
        """
        from saklas.io.paths import concept_dir

        name = concept.lower()
        folder = concept_dir("default", name)
        ds_path = folder / "statements.json"

        # Short-circuit: already materialized as an editable vector corpus.
        if ds_path.exists():
            return cls._from_json_file(ds_path, name_override=concept)

        # Manifold corpus already materialized?
        manifold_ds = cls._from_bundled_manifold(name, concept)
        if manifold_ds is not None:
            return manifold_ds

        from saklas.io.packs import materialize_bundled
        from saklas.io.manifolds import materialize_bundled_manifolds
        materialize_bundled()
        materialize_bundled_manifolds()
        if ds_path.exists():
            return cls._from_json_file(ds_path, name_override=concept)
        manifold_ds = cls._from_bundled_manifold(name, concept)
        if manifold_ds is not None:
            return manifold_ds

        default_root = folder.parent
        available = sorted(
            p.name for p in default_root.iterdir() if p.is_dir()
        ) if default_root.exists() else []
        raise FileNotFoundError(
            f"No curated dataset for '{concept}'. "
            f"Available: {', '.join(available)}"
        )

    @classmethod
    def _from_bundled_manifold(
        cls, name: str, concept: str,
    ) -> DataSource | None:
        """Reconstruct contrastive pairs from a 2-node ``pca`` manifold.

        Returns ``None`` when no such manifold is installed (the caller falls
        through to materialization / the not-found error).
        """
        from saklas.io.paths import manifold_dir

        mdir = manifold_dir("default", name)
        if not (mdir / "manifold.json").exists():
            return None
        from saklas.io.manifolds import ManifoldFolder, ManifoldFormatError
        try:
            mf = ManifoldFolder.load(mdir)
        except ManifoldFormatError:
            return None
        if mf.fit_mode != "pca" or len(mf.node_labels) != 2:
            return None
        groups = mf.node_groups()  # [(label, [statements]), ...] in node order
        pos = groups[0][1]
        neg = groups[1][1]
        pairs = [(p, n) for p, n in zip(pos, neg)]
        if not pairs:
            return None
        return cls(pairs=pairs, name=concept)

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
