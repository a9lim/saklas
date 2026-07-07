"""Hierarchical concept heaps + their ground-truth tree metrics.

Two taxonomies, identical elicitation, so DEPTH is the only variable:
  - "shallow": 24-node animal tree, depth 3, diameter 5 (the first run)
  - "deep":    37-node animal tree, depth 5, diameter 8 (deep mammal spine)

Hyperbolic advantage grows with depth+branching, so if curvature is ever
recruited, the deep tree is where it should show up.
"""

from __future__ import annotations

import numpy as np

# ---- shallow (depth 3) ------------------------------------------------------
PARENT_SHALLOW = {
    "animal": None,
    "mammal": "animal", "bird": "animal", "reptile": "animal", "fish": "animal",
    "carnivore": "mammal", "primate": "mammal", "cetacean": "mammal",
    "dog": "carnivore", "cat": "carnivore", "wolf": "carnivore",
    "monkey": "primate", "gorilla": "primate",
    "whale": "cetacean", "dolphin": "cetacean",
    "eagle": "bird", "sparrow": "bird", "penguin": "bird",
    "snake": "reptile", "lizard": "reptile", "turtle": "reptile",
    "shark": "fish", "salmon": "fish", "tuna": "fish",
}

# ---- deep (depth 5) ---------------------------------------------------------
PARENT_DEEP = {
    "animal": None,
    # L1
    "vertebrate": "animal", "invertebrate": "animal",
    # L2
    "mammal": "vertebrate", "bird": "vertebrate", "reptile": "vertebrate",
    "insect": "invertebrate", "mollusk": "invertebrate",
    # L3
    "carnivore": "mammal", "primate": "mammal",
    "raptor": "bird", "waterbird": "bird",
    "snake": "reptile", "lizard": "reptile",
    "beetle": "insect", "ant": "insect", "bee": "insect",
    "octopus": "mollusk", "snail": "mollusk",
    # L4
    "canine": "carnivore", "feline": "carnivore",
    "ape": "primate", "monkey": "primate",
    "eagle": "raptor", "owl": "raptor",
    "duck": "waterbird", "penguin": "waterbird",
    # L5 (leaves)
    "dog": "canine", "wolf": "canine", "fox": "canine",
    "cat": "feline", "lion": "feline", "tiger": "feline",
    "gorilla": "ape", "chimpanzee": "ape",
    "baboon": "monkey", "macaque": "monkey",
}

TAXONOMIES = {"shallow": PARENT_SHALLOW, "deep": PARENT_DEEP}

# back-compat: bare names default to the shallow tree (first-run artifacts)
PARENT = PARENT_SHALLOW
NODES = list(PARENT_SHALLOW.keys())


def _tree_distance_matrix(nodes, parent):
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    INF = 10**9
    D = np.full((n, n), INF)
    np.fill_diagonal(D, 0)
    for c, p in parent.items():
        if p is not None:
            i, j = idx[c], idx[p]
            D[i, j] = D[j, i] = 1
    for k in range(n):  # Floyd-Warshall
        D = np.minimum(D, D[:, k][:, None] + D[k, :][None, :])
    return D.astype(float)


def tree_distance_matrix(nodes=None, parent=None):
    nodes = nodes if nodes is not None else NODES
    parent = parent if parent is not None else PARENT
    return _tree_distance_matrix(nodes, parent)


def get_taxonomy(name):
    """Return (nodes, tree_distance_matrix, parent) for a named taxonomy."""
    parent = TAXONOMIES[name]
    nodes = list(parent.keys())
    return nodes, _tree_distance_matrix(nodes, parent), parent


def depth_of(node, parent=PARENT):
    d = 0
    while parent[node] is not None:
        node = parent[node]
        d += 1
    return d


if __name__ == "__main__":
    from delta_hyperbolicity import gromov_delta
    for name in ("shallow", "deep"):
        nodes, D, parent = get_taxonomy(name)
        depth = max(depth_of(n, parent) for n in nodes)
        r = gromov_delta(D)
        print(f"{name:8s}: {len(nodes):2d} nodes | depth {depth} | "
              f"diam {int(D.max())} | tree δ_rel {r['delta_rel']:.4f} (must be 0)")
