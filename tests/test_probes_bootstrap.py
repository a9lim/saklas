import pytest

from saklas import packs, probes_bootstrap


def _mk_concept(home, ns, name, tags):
    d = home / "vectors" / ns / name
    d.mkdir(parents=True)
    (d / "statements.json").write_text("[]")
    packs.PackMetadata(
        name=name, description="x", version="1.0.0", license="MIT",
        tags=tags, recommended_alpha=0.5, source="bundled",
        files={"statements.json": packs.hash_file(d / "statements.json")},
    ).write(d)


def test_load_defaults_groups_by_tag(monkeypatch, tmp_path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    # Materialization will also run and copy the bundled 28; pre-create the
    # ones we're asserting on so they win the tag placement.
    _mk_concept(tmp_path, "default", "zz-custom-a", ["custom-tag"])
    _mk_concept(tmp_path, "default", "zz-custom-b", ["custom-tag"])
    d = probes_bootstrap.load_defaults()
    assert sorted(d["custom-tag"]) == ["zz-custom-a", "zz-custom-b"]
    # Bundled happy.sad should have materialized under the affect tag.
    # Skip until the bundled pack is regenerated to the new 21-probe layout
    # via scripts/regenerate_bundled_statements.py --purge.
    if "affect" not in d:
        pytest.skip("bundled pack pending regeneration to new 21-probe layout")
    assert "happy.sad" in d["affect"]
