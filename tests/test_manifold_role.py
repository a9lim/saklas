"""Tests for the persona-manifold + label-form position work.

Covers the new surfaces introduced by Phases A/B/C of the
manifold-composition next-steps plan:

- Per-node ``role`` field in ``manifold.json`` round-trips through
  ``ManifoldFolder.load`` / ``.write_metadata`` and feeds the
  ``nodes_sha256`` cache key.
- ``Manifold.resolve_position`` accepts label-form payloads and raises
  ``UnknownManifoldLabelError`` on unknown labels.
- ``Manifold.nearest_node_role`` returns the role at the closest node
  (or ``None`` when no role is recorded).
- The grammar (``saklas.core.steering_expr``) parses both
  ``persona%0.3,0.8`` and ``persona%pirate`` and round-trips both forms.
- ``io.selectors.resolve_manifold_label`` and ``resolve_bare_name``
  resolve a bare name to a manifold-label hit and raise on cross-tier
  collisions.

CPU-only; no model load.
"""
from __future__ import annotations

import json
from pathlib import Path
import pytest
import torch

from saklas.core.manifold import (
    CustomDomain,
    Manifold,
    UnknownManifoldLabelError,
)
from saklas.core.steering_expr import (
    ManifoldTerm,
    format_expr,
    parse_expr,
)


# -- IO round-trip ----------------------------------------------------------

def _author_role_folder(root: Path, *, with_roles: bool = True) -> Path:
    """Hand-author a discover-mode persona manifold under ``root``."""
    from saklas.io.manifolds import MANIFOLD_FORMAT_VERSION

    folder = root / "persona"
    (folder / "nodes").mkdir(parents=True)
    labels = ["pirate", "cowboy", "professor"]
    for idx, label in enumerate(labels):
        statements = [f"as a {label} i would: {i}" for i in range(3)]
        (folder / "nodes" / f"{idx:02d}_{label}.json").write_text(
            json.dumps(statements)
        )
    nodes = [{"label": label} for label in labels]
    if with_roles:
        for entry, label in zip(nodes, labels):
            entry["role"] = label
    meta = {
        "format_version": MANIFOLD_FORMAT_VERSION,
        "name": "persona",
        "description": "personas as nodes",
        "fit_mode": "pca",
        "hyperparams": {"max_dim": 4, "var_threshold": 0.7},
        "nodes": nodes,
        "files": {},
    }
    (folder / "manifold.json").write_text(json.dumps(meta))
    return folder


def test_per_node_role_round_trip(tmp_path: Path):
    from saklas.io.manifolds import ManifoldFolder

    folder = _author_role_folder(tmp_path)
    mf = ManifoldFolder.load(folder)
    assert mf.node_roles == ["pirate", "cowboy", "professor"]
    # Rewrite and re-read — the role field should survive the round trip.
    mf.write_metadata()
    mf2 = ManifoldFolder.load(folder)
    assert mf2.node_roles == ["pirate", "cowboy", "professor"]


def test_legacy_folder_without_roles_loads_all_none(tmp_path: Path):
    """Loading a manifold whose nodes carry no ``role`` field yields an
    all-``None`` ``node_roles`` list of the right length.  This is
    semantically "no roles" — :meth:`ManifoldFolder.nodes_sha256` and
    :meth:`Manifold.nearest_node_role` both treat an all-``None`` list
    as the legacy / non-role path, byte-identical to the
    pre-Phase-A behavior."""
    from saklas.io.manifolds import ManifoldFolder

    folder = _author_role_folder(tmp_path, with_roles=False)
    mf = ManifoldFolder.load(folder)
    assert mf.node_roles == [None, None, None]
    assert not any(r is not None for r in mf.node_roles)


def test_role_field_invalidates_nodes_sha256(tmp_path: Path):
    from saklas.io.manifolds import ManifoldFolder

    folder_legacy = _author_role_folder(tmp_path / "a", with_roles=False)
    folder_role = _author_role_folder(tmp_path / "b", with_roles=True)
    legacy = ManifoldFolder.load(folder_legacy)
    role = ManifoldFolder.load(folder_role)
    # Same labels + same corpora + same hyperparams; only roles differ.
    assert legacy.nodes_sha256() != role.nodes_sha256()


def test_invalid_role_slug_rejected(tmp_path: Path):
    from saklas.io.manifolds import ManifoldFolder, ManifoldFormatError

    folder = tmp_path / "persona"
    (folder / "nodes").mkdir(parents=True)
    labels = ["pirate", "cowboy", "professor"]
    for idx, label in enumerate(labels):
        (folder / "nodes" / f"{idx:02d}_{label}.json").write_text(
            json.dumps(["x"])
        )
    from saklas.io.manifolds import MANIFOLD_FORMAT_VERSION

    nodes = [{"label": label} for label in labels]
    # Uppercase fails the slug regex.
    nodes[0]["role"] = "Pirate"
    (folder / "manifold.json").write_text(
        json.dumps({
            "format_version": MANIFOLD_FORMAT_VERSION,
            "name": "persona",
            "description": "",
            "fit_mode": "pca",
            "hyperparams": {},
            "nodes": nodes,
            "files": {},
        })
    )
    with pytest.raises(ManifoldFormatError, match="role 'Pirate' invalid"):
        ManifoldFolder.load(folder)


def test_create_discover_folder_with_roles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from saklas.io.manifolds import (
        ManifoldFolder, create_discover_manifold_folder,
    )

    # Redirect SAKLAS_HOME so the folder lands inside tmp_path.
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    create_discover_manifold_folder(
        "local", "persona", "test",
        fit_mode="pca",
        node_corpora={
            "pirate": ["arr"],
            "cowboy": ["howdy"],
            "professor": ["ahem"],
        },
        node_roles={"pirate": "pirate", "cowboy": "cowboy"},
    )
    folder = tmp_path / "manifolds" / "local" / "persona"
    mf = ManifoldFolder.load(folder)
    assert mf.node_roles == ["pirate", "cowboy", None]


# -- nearest-node + resolve_position ---------------------------------------

def _persona_manifold() -> Manifold:
    return Manifold(
        name="persona",
        domain=CustomDomain(2),
        node_labels=["pirate", "cowboy", "professor"],
        node_coords=torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]),
        layers={},
        node_roles=["pirate", "cowboy", None],
    )


def test_resolve_position_passes_through_coords():
    mf = _persona_manifold()
    assert mf.resolve_position((0.3, 0.7)) == (0.3, 0.7)
    assert mf.resolve_position([0.5, 0.5]) == (0.5, 0.5)


def test_resolve_position_label_form():
    mf = _persona_manifold()
    assert mf.resolve_position("pirate") == (0.0, 0.0)
    assert mf.resolve_position("cowboy") == (1.0, 0.0)
    assert mf.resolve_position("professor") == (0.0, 1.0)


def test_resolve_position_unknown_label_raises():
    mf = _persona_manifold()
    with pytest.raises(UnknownManifoldLabelError, match="ghost"):
        mf.resolve_position("ghost")


def test_nearest_node_role_at_coord():
    mf = _persona_manifold()
    assert mf.nearest_node_role((0.1, 0.1)) == "pirate"
    assert mf.nearest_node_role((0.9, 0.0)) == "cowboy"
    # Node 2 opts out of role substitution.
    assert mf.nearest_node_role((0.0, 0.9)) is None


def test_nearest_node_role_label_short_circuits():
    mf = _persona_manifold()
    assert mf.nearest_node_role("pirate") == "pirate"
    assert mf.nearest_node_role("professor") is None


def test_nearest_node_role_legacy_empty():
    """A legacy fitted manifold without ``node_roles`` returns None."""
    mf = Manifold(
        name="legacy",
        domain=CustomDomain(2),
        node_labels=["a", "b", "c"],
        node_coords=torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        layers={},
        # node_roles omitted; defaults to empty list.
    )
    assert mf.nearest_node_role((0.5, 0.5)) is None


# -- grammar (parser + formatter round-trip) -------------------------------

def test_grammar_coord_form_round_trip():
    s = parse_expr("0.7 persona%0.3,0.8")
    terms = list(s.alphas.values())
    assert len(terms) == 1
    term = terms[0]
    assert isinstance(term, ManifoldTerm)
    assert term.manifold == "persona"
    assert term.position == (0.3, 0.8)
    assert term.coeff == pytest.approx(0.7)
    assert format_expr(s) == "0.7 persona%0.3,0.8"


def test_grammar_label_form_round_trip():
    s = parse_expr("0.5 persona%pirate")
    term = next(iter(s.alphas.values()))
    assert isinstance(term, ManifoldTerm)
    assert term.position == "pirate"
    assert format_expr(s) == "0.5 persona%pirate"


def test_grammar_mixed_forms_compose():
    s = parse_expr("0.7 persona%pirate + 0.3 affect%0.1,0.2@response")
    terms = list(s.alphas.values())
    assert len(terms) == 2
    by_name = {t.manifold: t for t in terms if isinstance(t, ManifoldTerm)}
    assert by_name["persona"].position == "pirate"
    assert by_name["affect"].position == (0.1, 0.2)


def test_grammar_label_form_rejects_projection():
    from saklas.core.steering_expr import SteeringExprError

    with pytest.raises(SteeringExprError, match="does not compose"):
        parse_expr("0.5 persona%pirate~angry.calm")


# -- selector resolution + cross-tier ambiguity ----------------------------

def test_resolve_manifold_label_unique_match(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from saklas.io.manifolds import create_discover_manifold_folder
    from saklas.io.selectors import resolve_manifold_label

    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    create_discover_manifold_folder(
        "local", "persona", "test",
        fit_mode="pca",
        node_corpora={
            "pirate": ["arr"],
            "cowboy": ["howdy"],
            "professor": ["ahem"],
        },
    )
    hit = resolve_manifold_label("pirate")
    assert hit is not None
    assert hit.namespace == "local"
    assert hit.manifold_name == "persona"
    assert hit.label == "pirate"
    assert hit.manifold_key == "local/persona"


def test_resolve_manifold_label_miss_returns_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from saklas.io.selectors import resolve_manifold_label

    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    assert resolve_manifold_label("pirate") is None


def test_resolve_manifold_label_ambiguous_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from saklas.io.manifolds import create_discover_manifold_folder
    from saklas.io.selectors import (
        AmbiguousSelectorError,
        resolve_manifold_label,
    )

    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    create_discover_manifold_folder(
        "alice", "personas_a", "test",
        fit_mode="pca",
        node_corpora={
            "pirate": ["arr"],
            "knight": ["honor"],
            "priest": ["amen"],
        },
    )
    create_discover_manifold_folder(
        "bob", "personas_b", "test",
        fit_mode="pca",
        node_corpora={
            "pirate": ["plunder"],
            "wizard": ["abracadabra"],
            "thief": ["sneak"],
        },
    )
    with pytest.raises(AmbiguousSelectorError, match="ambiguous manifold label"):
        resolve_manifold_label("pirate")


def test_bare_name_resolves_to_manifold_term(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from saklas.io.manifolds import create_discover_manifold_folder

    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    create_discover_manifold_folder(
        "local", "persona", "test",
        fit_mode="pca",
        node_corpora={
            "pirate": ["arr"],
            "cowboy": ["howdy"],
            "professor": ["ahem"],
        },
    )
    s = parse_expr("0.7 pirate")
    term = next(iter(s.alphas.values()))
    assert isinstance(term, ManifoldTerm)
    assert term.manifold == "local/persona"
    assert term.position == "pirate"


def test_parse_bare_name_cross_tier_collision_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    from saklas.io.manifolds import create_discover_manifold_folder
    from saklas.io.packs import PackMetadata, hash_folder_files
    from saklas.io.selectors import AmbiguousSelectorError, invalidate

    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))

    vector_folder = tmp_path / "vectors" / "default" / "civilian.pirate"
    vector_folder.mkdir(parents=True)
    (vector_folder / "statements.json").write_text(json.dumps([]))
    PackMetadata(
        name="civilian.pirate",
        description="test",
        version="1.0.0",
        license="MIT",
        tags=[],
        recommended_alpha=0.5,
        source="local",
        files=hash_folder_files(vector_folder),
    ).write(vector_folder)

    create_discover_manifold_folder(
        "local", "persona", "test",
        fit_mode="pca",
        node_corpora={
            "pirate": ["arr"],
            "cowboy": ["howdy"],
            "professor": ["ahem"],
        },
    )
    invalidate()

    with pytest.raises(AmbiguousSelectorError, match="matches both"):
        parse_expr("0.7 pirate")


def test_namespace_qualified_bare_name_still_resolves(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from saklas.io.manifolds import create_discover_manifold_folder

    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    create_discover_manifold_folder(
        "local", "persona", "test",
        fit_mode="pca",
        node_corpora={
            "pirate": ["arr"],
            "cowboy": ["howdy"],
            "professor": ["ahem"],
        },
    )
    # An explicit ``%`` form continues to take the fast path — the
    # bare-name resolver only fires on plain non-namespace-qualified
    # terms.  Round-trip preserves the explicit form.
    s = parse_expr("0.5 local/persona%pirate")
    term = next(iter(s.alphas.values()))
    assert isinstance(term, ManifoldTerm)
    assert term.manifold == "local/persona"
    assert term.position == "pirate"
