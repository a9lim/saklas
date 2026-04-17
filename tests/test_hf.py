import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from saklas.io import hf, packs


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


def test_pull_pack_without_pack_json_and_no_tensors_errors(tmp_path, monkeypatch):
    bad = tmp_path / "downloaded" / "nope"
    bad.mkdir(parents=True)
    (bad / "random.txt").write_text("x")
    monkeypatch.setattr(hf, "_hf_snapshot_download", lambda **kw: str(bad))
    target = tmp_path / "installed" / "nope"
    with pytest.raises(hf.HFError, match="no .safetensors or .gguf"):
        hf.pull_pack("user/nope", target_folder=target, force=False)


def test_pull_pack_synthesizes_pack_json_from_raw_safetensors(tmp_path, monkeypatch):
    """Repo without a pack.json but with bare .safetensors should install cleanly."""
    import torch
    from saklas.core.vectors import save_profile

    raw = tmp_path / "downloaded" / "happy_raw"
    raw.mkdir(parents=True)
    profile = {0: torch.randn(8), 1: torch.randn(8)}
    save_profile(profile, str(raw / "google__gemma-2-2b-it.safetensors"),
                 {"method": "contrastive_pca"})
    monkeypatch.setattr(hf, "_hf_snapshot_download", lambda **kw: str(raw))

    target = tmp_path / "installed" / "alice_happy_raw"
    hf.pull_pack("alice/happy-raw", target_folder=target, force=False)

    # Synthesized pack is a fully legitimate concept folder.
    cf = packs.ConceptFolder.load(target)
    assert cf.metadata.source == "hf://alice/happy-raw"
    assert cf.tensor_models() == ["google__gemma-2-2b-it"]
    assert cf.tensor_format("google__gemma-2-2b-it") == "safetensors"
    # Name slug comes from the repo name (dashes allowed, underscore OK).
    assert cf.metadata.name == "happy-raw"


def test_pull_pack_synthesizes_pack_json_from_raw_gguf(tmp_path, monkeypatch):
    """Frictionless path: an HF GGUF-only repo installs without any saklas metadata."""
    pytest.importorskip("gguf")
    import torch
    from saklas.io.gguf_io import write_gguf_profile

    raw = tmp_path / "downloaded" / "angry_gguf"
    raw.mkdir(parents=True)
    profile = {0: torch.randn(8), 1: torch.randn(8), 14: torch.randn(8)}
    write_gguf_profile(profile, raw / "llama3.1-8b.gguf", model_hint="llama")
    monkeypatch.setattr(hf, "_hf_snapshot_download", lambda **kw: str(raw))

    target = tmp_path / "installed" / "jukofyork_creative"
    hf.pull_pack("jukofyork/creative", target_folder=target, force=False)

    cf = packs.ConceptFolder.load(target)
    assert cf.metadata.source == "hf://jukofyork/creative"
    assert cf.tensor_models() == ["llama3.1-8b"]
    assert cf.tensor_format("llama3.1-8b") == "gguf"


def test_pull_pack_synthesizes_sidecars_for_raw_safetensors(tmp_path, monkeypatch):
    """A raw repo with .safetensors but no sidecars should still install cleanly."""
    from safetensors.torch import save_file
    import torch

    raw = tmp_path / "downloaded" / "raw_no_sidecar"
    raw.mkdir(parents=True)
    # Bypass save_profile so no sidecar is written.
    save_file({"layer_0": torch.randn(4)}, str(raw / "llama.safetensors"))
    monkeypatch.setattr(hf, "_hf_snapshot_download", lambda **kw: str(raw))

    target = tmp_path / "installed" / "alice_raw"
    hf.pull_pack("alice/raw-no-sidecar", target_folder=target, force=False)
    cf = packs.ConceptFolder.load(target)
    assert cf.tensor_models() == ["llama"]
    sc = cf.sidecar("llama")
    assert sc.method == "imported"


def test_pull_pack_synthesized_name_slugs_invalid_chars(tmp_path, monkeypatch):
    import torch
    from saklas.core.vectors import save_profile

    raw = tmp_path / "downloaded" / "Weird_Name"
    raw.mkdir(parents=True)
    save_profile({0: torch.randn(4)}, str(raw / "foo.safetensors"),
                 {"method": "contrastive_pca"})
    monkeypatch.setattr(hf, "_hf_snapshot_download", lambda **kw: str(raw))

    target = tmp_path / "installed" / "alice_Weird_Name"
    hf.pull_pack("alice/Weird_Name!!", target_folder=target, force=False)
    cf = packs.ConceptFolder.load(target)
    # NAME_REGEX lowercase + only [a-z0-9._-].
    assert packs.NAME_REGEX.match(cf.metadata.name)


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

    from saklas.cli.selectors import parse as sparse
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

    from saklas.cli.selectors import parse as sparse
    hf.search_packs(sparse("tag:emotion"))
    kwargs = api.list_models.call_args.kwargs
    filter_arg = kwargs.get("filter") or kwargs.get("tags") or []
    assert "emotion" in filter_arg


def test_fetch_info_reads_pack_json(tmp_path, monkeypatch):
    pj = tmp_path / "pack.json"
    pj.write_text(json.dumps({
        "name": "happy",
        "description": "x",
        "format_version": 2,
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
        "method": "pca", "saklas_version": "0.1.0",
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
    assert "saklas pack install alice/happy" in card


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
        '{"method":"pca","saklas_version":"0.1.0"}'
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


def test_sidecar_stem_to_hf_coord_strips_sae_suffix():
    """Regression: ``_sidecar_stem_to_hf_coord`` must strip the
    ``_sae-<release>`` suffix before translating ``__`` -> ``/``. A naive
    ``replace("__", "/")`` on the SAE-variant stem produces a non-existent
    HF repo and poisons ``base_model:`` frontmatter.
    """
    from saklas.io import hf as _hf

    assert _hf._sidecar_stem_to_hf_coord(
        "google__gemma-3-4b-it"
    ) == "google/gemma-3-4b-it"
    assert _hf._sidecar_stem_to_hf_coord(
        "google__gemma-3-4b-it_sae-gemma-scope-2b-pt-res-canonical"
    ) == "google/gemma-3-4b-it"


def test_render_model_card_base_model_dedupes_sae_variants(tmp_path):
    """A pack with both raw + SAE tensors for the same base model should
    list that base model exactly once under ``base_model:`` — not twice,
    and never as ``google/gemma-3-4b-it_sae-<release>`` (broken coord).
    """
    from saklas.io import hf as _hf
    from saklas.io.packs import Sidecar

    meta = packs.PackMetadata(
        name="honesty", description="x", version="1.0.0", license="MIT",
        tags=[], recommended_alpha=0.5, source="local", files={},
    )
    # Two sidecars keyed by the full file stem — one raw, one SAE.
    raw_stem = "google__gemma-3-4b-it"
    sae_stem = "google__gemma-3-4b-it_sae-gemma-scope-2b-pt-res-canonical"
    sidecars: dict[str, Sidecar] = {
        raw_stem: Sidecar(method="contrastive_pca", saklas_version="2.0.0"),
        sae_stem: Sidecar(method="pca_center_sae", saklas_version="2.0.0"),
    }

    card = _hf._render_model_card(meta, sidecars, "alice/honesty")

    # Base model listed exactly once.
    assert card.count("- google/gemma-3-4b-it\n") == 1
    # The poisoned coord must not appear anywhere.
    assert "google/gemma-3-4b-it_sae-" not in card
    assert "gemma-3-4b-it_sae-gemma-scope" not in card
    # Tensors table row should also use the clean base coord.
    assert "| `google/gemma-3-4b-it` | `contrastive_pca` |" in card
    assert "| `google/gemma-3-4b-it` | `pca_center_sae` |" in card


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
