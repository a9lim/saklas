"""``io.shards`` — the immutable per-layer generation primitive.

The neutral cache, the alignment cache, and the Jacobian lens all publish
per-layer payloads behind one atomic JSON pointer. These tests pin the shared
primitive directly, and assert each family still routes through it so the three
crash-recovery paths cannot drift apart again.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from saklas.io import shards


def _anchor(tmp_path: Path) -> Path:
    return tmp_path / "artifact.safetensors"


def test_generation_path_is_layer_tagged_and_never_repeats(tmp_path: Path) -> None:
    anchor = _anchor(tmp_path)
    first = shards.generation_path(anchor, 3)
    second = shards.generation_path(anchor, 3)
    assert first.parent == anchor.parent
    assert first.name.startswith("artifact.layer-3.gen-")
    assert first.suffix == ".safetensors"
    assert first != second, "a fresh generation must never reuse a payload name"


def test_shard_paths_resolves_declared_generations(tmp_path: Path) -> None:
    anchor = _anchor(tmp_path)
    sidecar = {"tensor_files": {"0": "artifact.layer-0.gen-aa.safetensors",
                                "5": "artifact.layer-5.gen-bb.safetensors"}}
    out = shards.shard_paths(anchor, sidecar, [0, 5], label="x")
    assert out == {
        0: tmp_path / "artifact.layer-0.gen-aa.safetensors",
        5: tmp_path / "artifact.layer-5.gen-bb.safetensors",
    }


def test_shard_paths_requires_exact_keys_by_default(tmp_path: Path) -> None:
    sidecar = {"tensor_files": {"0": "a.safetensors", "1": "b.safetensors"}}
    with pytest.raises(ValueError, match="tensor shard keys"):
        shards.shard_paths(_anchor(tmp_path), sidecar, [0], label="x")
    # A scoped read of a wider pointer is explicitly allowed.
    assert set(
        shards.shard_paths(
            _anchor(tmp_path), sidecar, [0], label="x", require_exact_keys=False,
        )
    ) == {0}


def test_shard_paths_rejects_traversal_and_missing_map(tmp_path: Path) -> None:
    anchor = _anchor(tmp_path)
    with pytest.raises(ValueError, match="no tensor shard map"):
        shards.shard_paths(anchor, {}, [0], label="x")
    for bad in ("../escape.safetensors", "/abs/path.safetensors", ""):
        with pytest.raises(ValueError, match="invalid tensor shard for layer 0"):
            shards.shard_paths(
                anchor, {"tensor_files": {"0": bad}}, [0], label="x",
            )


def test_representative_shard_path_picks_lowest_layer(tmp_path: Path) -> None:
    anchor = _anchor(tmp_path)
    sidecar = {"tensor_files": {"10": "hi.safetensors", "2": "lo.safetensors"}}
    assert shards.representative_shard_path(
        anchor, sidecar, label="x",
    ) == tmp_path / "lo.safetensors"
    with pytest.raises(ValueError, match="no tensor shard map"):
        shards.representative_shard_path(anchor, {"tensor_files": {}}, label="x")


def test_json_pointer_matches_only_on_exact_payload(tmp_path: Path) -> None:
    pointer = tmp_path / "manifest.json"
    payload = {"format_version": 6, "tensor_files": {"0": "a.safetensors"}}
    assert not shards.json_pointer_matches(pointer, payload)
    pointer.write_text(json.dumps(payload))
    assert shards.json_pointer_matches(pointer, payload)
    pointer.write_text(json.dumps({**payload, "format_version": 5}))
    assert not shards.json_pointer_matches(pointer, payload)
    pointer.write_text("{ not json")
    assert not shards.json_pointer_matches(pointer, payload)


def test_cleanup_keeps_referenced_generations_only(tmp_path: Path) -> None:
    anchor = _anchor(tmp_path)
    keep = tmp_path / "artifact.layer-0.gen-keep.safetensors"
    drop = tmp_path / "artifact.layer-0.gen-drop.safetensors"
    stale_tmp = tmp_path / "artifact.layer-1.gen-old.safetensors.tmp"
    unrelated = tmp_path / "other.layer-0.gen-x.safetensors"
    for path in (keep, drop, stale_tmp, unrelated, anchor):
        path.write_bytes(b"x")

    shards.cleanup_generations(
        anchor, {"tensor_files": {"0": keep.name}}, label="x",
    )

    assert keep.exists()
    assert unrelated.exists(), "another artifact's generations are off-limits"
    assert not drop.exists()
    assert not stale_tmp.exists()
    assert not anchor.exists(), "the pre-sharded anchor is itself collectable"


def test_fit_lock_is_a_sibling_of_the_anchor(tmp_path: Path) -> None:
    anchor = _anchor(tmp_path)
    locks = tmp_path / ".locks"
    with shards.fit_lock(anchor):
        # The fit lock must not be the anchor's own pair lock: short pair-lock
        # transactions run inside the long fit.
        assert (locks / "artifact.fit.lock").exists()
        assert not (locks / "artifact.safetensors.lock").exists()


def test_families_route_through_the_shared_primitive() -> None:
    """The three families must not carry private copies of this machinery."""
    from saklas.io import alignment, lens

    assert alignment.generation_path is shards.generation_path
    assert alignment._json_pointer_matches is shards.json_pointer_matches
    assert lens._new_layer_generation is shards.generation_path
    assert lens._json_pointer_matches is shards.json_pointer_matches
