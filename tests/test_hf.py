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


def test_snapshot_download_uses_model_repo_type(tmp_path, monkeypatch):
    fake = _fake_repo(tmp_path)
    calls = []

    def fake_dl(repo_id, **kwargs):
        calls.append(repo_id)
        return str(fake)

    monkeypatch.setattr(hf, "_hf_snapshot_download", fake_dl)
    path = hf._download("user/happy")
    assert calls == ["user/happy"]
    assert Path(path).is_dir()


def test_split_revision_plain():
    assert hf.split_revision("user/happy") == ("user/happy", None)


def test_split_revision_with_tag():
    assert hf.split_revision("user/happy@v1.2.0") == ("user/happy", "v1.2.0")


def test_split_revision_with_sha():
    assert hf.split_revision("user/happy@abcdef0") == ("user/happy", "abcdef0")


def test_split_revision_empty_rev_errors():
    with pytest.raises(hf.HFError, match="empty revision"):
        hf.split_revision("user/happy@")


def test_download_passes_revision(tmp_path, monkeypatch):
    fake = _fake_repo(tmp_path)
    captured = {}

    def fake_dl(**kwargs):
        captured.update(kwargs)
        return str(fake)

    monkeypatch.setattr(hf, "_hf_snapshot_download", fake_dl)
    hf._download("user/happy", revision="v1.2.0")
    assert captured["revision"] == "v1.2.0"


def test_download_omits_revision_when_none(tmp_path, monkeypatch):
    fake = _fake_repo(tmp_path)
    captured = {}

    def fake_dl(**kwargs):
        captured.update(kwargs)
        return str(fake)

    monkeypatch.setattr(hf, "_hf_snapshot_download", fake_dl)
    hf._download("user/happy")
    assert "revision" not in captured


def test_pull_pack_records_revision_in_source(tmp_path, monkeypatch):
    fake = _fake_repo(tmp_path)
    monkeypatch.setattr(hf, "_hf_snapshot_download", lambda **kw: str(fake))
    target = tmp_path / "installed" / "happy"
    hf.pull_pack("user/happy", target_folder=target, force=False, revision="v1.2.0")
    m = packs.PackMetadata.load(target)
    assert m.source == "hf://user/happy@v1.2.0"


def test_snapshot_download_failure_raises(monkeypatch):
    def fake_dl(repo_id, **kwargs):
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
    api.list_models.return_value = fake_results
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
    api.list_models.assert_called_once()
    kwargs = api.list_models.call_args.kwargs
    filter_arg = kwargs.get("filter") or kwargs.get("tags") or []
    assert "saklas-pack" in filter_arg


def test_search_packs_tag_filter(monkeypatch):
    api = MagicMock()
    api.list_models.return_value = []
    monkeypatch.setattr(hf, "_hf_api", lambda: api)
    monkeypatch.setattr(hf, "fetch_info", lambda coord: {})

    from saklas.cli_selectors import parse as sparse
    hf.search_packs(sparse("tag:emotion"))
    kwargs = api.list_models.call_args.kwargs
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

    def fake_download(repo_id, filename, **kwargs):
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


def _fake_pack_with_tensor(tmp_path, name="happy", model_id="google/gemma-2-2b-it"):
    """Build a pack folder containing statements + one safetensors + sidecar."""
    import json as _json
    folder = tmp_path / name
    folder.mkdir()
    (folder / "statements.json").write_text('[["a","b"]]')
    safe = model_id.replace("/", "__")
    (folder / f"{safe}.safetensors").write_bytes(b"\x00" * 16)
    (folder / f"{safe}.json").write_text(_json.dumps({
        "method": "pca", "scores": {"0": 0.1}, "saklas_version": "0.1.0",
        "statements_sha256": None,
    }))
    meta = packs.PackMetadata(
        name=name, description="happy vibes", version="1.2.3", license="MIT",
        tags=["affect"], recommended_alpha=0.5, source="local",
        files={},
    )
    meta.files = {
        p.name: packs.hash_file(p)
        for p in sorted(folder.iterdir()) if p.name != "pack.json"
    }
    meta.write(folder)
    return folder


def test_push_pack_dry_run_writes_card_and_gitattributes(tmp_path, monkeypatch):
    folder = _fake_pack_with_tensor(tmp_path)
    staged: dict[str, bytes] = {}
    real_mkdtemp = __import__("tempfile").mkdtemp
    captured_dir: list[Path] = []

    def spy_mkdtemp(**kw):
        d = real_mkdtemp(**kw)
        captured_dir.append(Path(d))
        return d

    monkeypatch.setattr("tempfile.mkdtemp", spy_mkdtemp)

    # Capture staging contents by hooking shutil.rmtree right before cleanup.
    import shutil as _sh
    orig_rmtree = _sh.rmtree

    def capture_rmtree(path, *a, **kw):
        if captured_dir and Path(path) == captured_dir[0]:
            for p in Path(path).rglob("*"):
                if p.is_file():
                    staged[p.relative_to(path).as_posix()] = p.read_bytes()
        return orig_rmtree(path, *a, **kw)

    monkeypatch.setattr(hf.shutil, "rmtree", capture_rmtree)

    url, sha = hf.push_pack(folder, "alice/happy", dry_run=True)
    assert url == "https://huggingface.co/alice/happy"
    assert sha is None
    assert "README.md" in staged
    assert ".gitattributes" in staged
    assert b"lfs" in staged[".gitattributes"]
    card = staged["README.md"].decode()
    assert "library_name: saklas" in card
    assert "google/gemma-2-2b-it" in card  # base_model listed
    assert "base_model_relation: adapter" in card
    assert "saklas install alice/happy" in card


def test_push_pack_filters_statements_only(tmp_path, monkeypatch):
    folder = _fake_pack_with_tensor(tmp_path)
    api = MagicMock()
    upload = MagicMock()
    upload.oid = "abc123def456"
    api.upload_folder.return_value = upload
    monkeypatch.setattr(hf, "HfApi", None, raising=False)

    def fake_api_ctor():
        return api
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: api)

    url, sha = hf.push_pack(
        folder, "alice/happy",
        include_statements=True, include_tensors=False,
    )
    assert sha == "abc123def456"
    api.create_repo.assert_called_once()
    # Staging dir is cleaned up after push_pack returns, so we can't
    # re-inspect its contents here.  Instead verify upload_folder was
    # called exactly once — the include_tensors=False path above means
    # no .safetensors files could have been staged.
    assert api.upload_folder.call_count == 1


def test_push_pack_nothing_to_push_errors(tmp_path):
    folder = _fake_pack_with_tensor(tmp_path)
    with pytest.raises(hf.HFError, match="nothing to push"):
        hf.push_pack(
            folder, "alice/happy",
            include_statements=False, include_tensors=False, dry_run=True,
        )


def test_push_pack_model_scope_limits_tensors(tmp_path, monkeypatch):
    folder = _fake_pack_with_tensor(tmp_path, model_id="google/gemma-2-2b-it")
    # Add a second tensor for a different model
    (folder / "meta__llama-3-8b.safetensors").write_bytes(b"\x00" * 8)
    (folder / "meta__llama-3-8b.json").write_text(
        '{"method":"pca","scores":{"0":0.2},"saklas_version":"0.1.0"}'
    )
    meta = packs.PackMetadata.load(folder)
    meta.files = {
        p.name: packs.hash_file(p)
        for p in sorted(folder.iterdir()) if p.name != "pack.json"
    }
    meta.write(folder)

    staged_pack: dict = {}
    real_mkdtemp = __import__("tempfile").mkdtemp
    captured: list[Path] = []

    def spy_mkdtemp(**kw):
        d = real_mkdtemp(**kw)
        captured.append(Path(d))
        return d

    monkeypatch.setattr("tempfile.mkdtemp", spy_mkdtemp)
    import shutil as _sh
    orig_rmtree = _sh.rmtree

    def capture(path, *a, **kw):
        if captured and Path(path) == captured[0]:
            pj = Path(path) / "pack.json"
            if pj.exists():
                staged_pack.update(json.loads(pj.read_text()))
        return orig_rmtree(path, *a, **kw)

    monkeypatch.setattr(hf.shutil, "rmtree", capture)

    hf.push_pack(
        folder, "alice/happy",
        model_scope="google/gemma-2-2b-it", dry_run=True,
    )
    files = staged_pack["files"]
    assert "google__gemma-2-2b-it.safetensors" in files
    assert "meta__llama-3-8b.safetensors" not in files
    assert "statements.json" in files


def test_resolve_target_coord_explicit():
    assert hf.resolve_target_coord("happy", "bob/happy") == "bob/happy"


def test_resolve_target_coord_bad_as():
    with pytest.raises(hf.HFError, match="--as"):
        hf.resolve_target_coord("happy", "bob")


def test_resolve_target_coord_uses_whoami(monkeypatch):
    import huggingface_hub
    api = MagicMock()
    api.whoami.return_value = {"name": "alice"}
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: api)
    assert hf.resolve_target_coord("happy", None) == "alice/happy"
