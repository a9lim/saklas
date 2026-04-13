import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from saklas import hf, packs


def _fake_repo(tmp_path, name="happy"):
    repo = tmp_path / "downloaded" / name
    repo.mkdir(parents=True)
    (repo / "statements.json").write_text("[]")
    meta = packs.PackMetadata(
        name=name, description="x", version="1.0.0", license="MIT",
        tags=["test"], recommended_alpha=0.5,
        source="hf://user/happy", files={},
    )
    meta.files = {"statements.json": packs.hash_file(repo / "statements.json")}
    meta.write(repo)
    return repo


def test_snapshot_download_dataset_first(tmp_path, monkeypatch):
    fake = _fake_repo(tmp_path)
    calls = []

    def fake_dl(repo_id, repo_type, **kwargs):
        calls.append((repo_id, repo_type))
        return str(fake)

    monkeypatch.setattr(hf, "_hf_snapshot_download", fake_dl)
    path = hf._download("user/happy")
    assert calls[0] == ("user/happy", "dataset")
    assert Path(path).is_dir()


def test_snapshot_download_falls_back_to_model(tmp_path, monkeypatch):
    fake = _fake_repo(tmp_path)
    calls = []

    def fake_dl(repo_id, repo_type, **kwargs):
        calls.append((repo_id, repo_type))
        if repo_type == "dataset":
            raise RuntimeError("not a dataset")
        return str(fake)

    monkeypatch.setattr(hf, "_hf_snapshot_download", fake_dl)
    hf._download("user/happy")
    assert [c[1] for c in calls] == ["dataset", "model"]


def test_snapshot_download_both_fail_raises(monkeypatch):
    def fake_dl(repo_id, repo_type, **kwargs):
        raise RuntimeError("no such repo")
    monkeypatch.setattr(hf, "_hf_snapshot_download", fake_dl)
    with pytest.raises(hf.HFError, match="not found"):
        hf._download("user/nope")


def test_pull_pack_installs_to_target(tmp_path, monkeypatch):
    fake = _fake_repo(tmp_path)
    monkeypatch.setattr(hf, "_hf_snapshot_download", lambda **kw: str(fake))
    target = tmp_path / "installed" / "happy"
    hf.pull_pack("user/happy", target_folder=target, force=False)
    assert (target / "pack.json").is_file()
    assert (target / "statements.json").is_file()
    m = packs.PackMetadata.load(target)
    assert m.source == "hf://user/happy"


def test_pull_pack_without_pack_json_errors(tmp_path, monkeypatch):
    bad = tmp_path / "downloaded" / "nope"
    bad.mkdir(parents=True)
    (bad / "random.txt").write_text("x")
    monkeypatch.setattr(hf, "_hf_snapshot_download", lambda **kw: str(bad))
    target = tmp_path / "installed" / "nope"
    with pytest.raises(hf.HFError, match="not a saklas pack"):
        hf.pull_pack("user/nope", target_folder=target, force=False)


def test_search_packs_bare_query(monkeypatch):
    fake_results = [
        MagicMock(id="user/happy", tags=["saklas-pack", "emotion"]),
        MagicMock(id="other/angry", tags=["saklas-pack", "emotion"]),
    ]
    api = MagicMock()
    api.list_datasets.return_value = fake_results
    monkeypatch.setattr(hf, "_hf_api", lambda: api)

    monkeypatch.setattr(hf, "fetch_info", lambda coord: {
        "name": coord.split("/")[-1],
        "namespace": coord.split("/")[0],
        "description": "x",
        "tags": ["emotion"],
        "recommended_alpha": 0.5,
    })

    from saklas.cli_selectors import parse as sparse
    rows = hf.search_packs(sparse("calm"))
    assert len(rows) == 2
    api.list_datasets.assert_called_once()
    kwargs = api.list_datasets.call_args.kwargs
    filter_arg = kwargs.get("filter") or kwargs.get("tags") or []
    assert "saklas-pack" in filter_arg


def test_search_packs_tag_filter(monkeypatch):
    api = MagicMock()
    api.list_datasets.return_value = []
    monkeypatch.setattr(hf, "_hf_api", lambda: api)
    monkeypatch.setattr(hf, "fetch_info", lambda coord: {})

    from saklas.cli_selectors import parse as sparse
    hf.search_packs(sparse("tag:emotion"))
    kwargs = api.list_datasets.call_args.kwargs
    filter_arg = kwargs.get("filter") or kwargs.get("tags") or []
    assert "emotion" in filter_arg


def test_fetch_info_reads_pack_json(tmp_path, monkeypatch):
    pj = tmp_path / "pack.json"
    pj.write_text(json.dumps({
        "name": "happy",
        "description": "x",
        "version": "1.0.0",
        "license": "MIT",
        "tags": ["emotion"],
        "recommended_alpha": 0.4,
        "source": "bundled",
        "files": {},
    }))

    def fake_download(repo_id, filename, repo_type, **kwargs):
        assert filename == "pack.json"
        return str(pj)

    monkeypatch.setattr(hf, "_hf_hub_download", fake_download)

    api = MagicMock()
    api.list_repo_files.return_value = ["pack.json", "statements.json",
                                         "google__gemma-2-2b-it.safetensors"]
    monkeypatch.setattr(hf, "_hf_api", lambda: api)

    info = hf.fetch_info("user/happy")
    assert info["name"] == "happy"
    assert info["namespace"] == "user"
    assert info["description"] == "x"
    assert info["tags"] == ["emotion"]
    assert "google__gemma-2-2b-it" in info["tensor_models"]
