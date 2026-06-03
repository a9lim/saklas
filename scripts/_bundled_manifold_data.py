"""Small loaders for bundled concept manifolds used by experiment scripts."""
from __future__ import annotations

import json
from importlib import resources
from importlib.resources.abc import Traversable
from typing import Any


def _read_json(path: Traversable) -> Any:
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return json.loads(path.read_text())


def _bundled_manifold_folder(concept: str) -> Traversable:
    folder = resources.files("saklas.data").joinpath("manifolds").joinpath(concept)
    if not folder.is_dir():
        raise FileNotFoundError(f"bundled manifold not found: {concept}")
    return folder


def load_bundled_manifold_scenarios(concept: str) -> list[str]:
    """Return the optional shared scenario provenance for a bundled manifold."""
    folder = _bundled_manifold_folder(concept)
    data = _read_json(folder.joinpath("scenarios.json"))
    scenarios = data.get("scenarios") if isinstance(data, dict) else data
    if (
        not isinstance(scenarios, list)
        or not all(isinstance(s, str) for s in scenarios)
    ):
        raise ValueError(f"{concept}: scenarios.json must contain a string list")
    return list(scenarios)


def _read_node_corpus(folder: Traversable, concept: str, index: int) -> list[str]:
    prefix = f"{index:02d}_"
    nodes_dir = folder.joinpath("nodes")
    matches = sorted(
        (
            path for path in nodes_dir.iterdir()
            if path.name.startswith(prefix) and path.name.endswith(".json")
        ),
        key=lambda path: path.name,
    )
    if len(matches) != 1:
        raise ValueError(
            f"{concept}: expected exactly one node corpus with prefix {prefix!r}"
        )
    corpus = _read_json(matches[0])
    if not isinstance(corpus, list) or not all(isinstance(s, str) for s in corpus):
        raise ValueError(f"{concept}: {matches[0].name} must be a string list")
    return list(corpus)


def load_bipolar_manifold_pairs(concept: str) -> tuple[list[str], list[dict[str, str]]]:
    """Return ``(scenarios, [{positive, negative}, ...])`` for a 2-node manifold."""
    folder = _bundled_manifold_folder(concept)
    spec = _read_json(folder.joinpath("manifold.json"))
    nodes = spec.get("nodes") if isinstance(spec, dict) else None
    if not isinstance(nodes, list) or len(nodes) != 2:
        raise ValueError(f"{concept}: expected a 2-node bundled manifold")

    scenarios = load_bundled_manifold_scenarios(concept)
    positive = _read_node_corpus(folder, concept, 0)
    negative = _read_node_corpus(folder, concept, 1)
    if len(positive) != len(negative):
        raise ValueError(
            f"{concept}: positive/negative node corpora have different lengths"
        )
    if scenarios and len(positive) % len(scenarios) != 0:
        raise ValueError(
            f"{concept}: node corpus length is not divisible by scenario count"
        )

    pairs = [
        {"positive": pos, "negative": neg}
        for pos, neg in zip(positive, negative, strict=True)
    ]
    return scenarios, pairs
