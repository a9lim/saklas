from typing import Any
import json
import shutil
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from saklas.io import hf


def _write_fitted_manifold(
    folder: Path, model_id: str, *, release: str | None = None,
) -> Path:
    import torch

    from saklas.core.manifold import (
        MANIFOLD_FIT_POLICY_VERSION, save_manifold,
    )
    from saklas.core.vectors import fold_directions_to_subspace
    from saklas.io.manifolds import ManifoldFolder
    from saklas.io.paths import tensor_filename

    manifold = fold_directions_to_subspace(
        folder.name, {0: torch.tensor([1.0, 0.0])}, None, label="test",
        feature_space="raw" if release is None else f"sae-{release}",
    )
    mf = ManifoldFolder.load(folder, verify_manifest=False)
    path = folder / tensor_filename(model_id, release=release)
    metadata: dict[str, Any] = {
        "method": "manifold_pca",
        "nodes_sha256": mf.nodes_sha256(),
        "model_fingerprint": f"fp:{model_id}",
        "fit_policy_version": MANIFOLD_FIT_POLICY_VERSION,
    }
    if release is not None:
        metadata.update(
            sae_release=release,
            sae_revision="test",
            sae_fingerprint=f"sae:{release}",
        )
    save_manifold(manifold, path, metadata)
    mf.update_file_hashes(path, path.with_suffix(".json"))
    return path


def test_split_revision_plain():
    assert hf.split_revision("user/happy") == ("user/happy", None)


def test_split_revision_with_tag():
    assert hf.split_revision("user/happy@v1.2.0") == ("user/happy", "v1.2.0")


def test_split_revision_with_sha():
    assert hf.split_revision("user/happy@abcdef0") == ("user/happy", "abcdef0")


def test_split_revision_empty_rev_errors():
    with pytest.raises(hf.HFError, match="empty revision"):
        hf.split_revision("user/happy@")


def test_resolve_target_coord_explicit():
    assert hf.resolve_target_coord("happy", "bob/happy") == "bob/happy"


def test_resolve_target_coord_bad_as():
    with pytest.raises(hf.HFError, match="--as"):
        hf.resolve_target_coord("happy", "bob")


def test_resolve_target_coord_uses_whoami(monkeypatch: pytest.MonkeyPatch):
    import huggingface_hub
    api = MagicMock()
    api.whoami.return_value = {"name": "alice"}
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: api)
    assert hf.resolve_target_coord("happy", None) == "alice/happy"


# ============================================================ push_manifold ===
#
# B1: HF upload for manifolds — parallel to push_pack, but tagged
# ``saklas-manifold`` and always including the node corpus.


def _author_fake_manifold(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, name: str = "mood"):
    """Author a tiny authored manifold + one fake fitted tensor under SAKLAS_HOME."""
    from saklas.io.manifolds import create_manifold_folder

    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    nodes = [
        {"label": label, "coords": [i / 2.0],
         "statements": [f"{label} says {j}" for j in range(3)]}
        for i, label in enumerate(["calm", "uneasy", "afraid"])
    ]
    folder, _ = create_manifold_folder("local", name, "a mood axis", domain, nodes)

    _write_fitted_manifold(folder, "google/gemma-2-2b-it")
    return folder


def test_pull_manifold_records_revision_in_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from saklas.io import hf_manifolds as hfm
    from saklas.io.manifolds import ManifoldFolder

    fake = _author_fake_manifold(tmp_path, monkeypatch)
    monkeypatch.setattr(hfm, "_hf_snapshot_download", lambda **kw: str(fake))
    target = tmp_path / "installed" / "mood"

    hfm.pull_manifold(
        "alice/mood", target_folder=target, force=False, revision="v1.2.0",
    )

    assert ManifoldFolder.load(target).source == "hf://alice/mood@v1.2.0"


def test_force_pull_waits_for_existing_pair_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from saklas.io import hf_manifolds as hfm
    from saklas.io.manifold_folder import manifold_pair_lock

    source = _author_fake_manifold(tmp_path, monkeypatch)
    monkeypatch.setattr(hfm, "_hf_snapshot_download", lambda **kw: str(source))
    target = tmp_path / "installed" / "mood"
    shutil.copytree(source, target)
    fitted = next(target.glob("*.safetensors"))
    started = threading.Event()
    done = threading.Event()
    errors: list[BaseException] = []

    def pull() -> None:
        started.set()
        try:
            hfm.pull_manifold("alice/mood", target, force=True)
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)
        finally:
            done.set()

    with manifold_pair_lock(fitted):
        worker = threading.Thread(target=pull)
        worker.start()
        assert started.wait(1.0)
        assert not done.wait(0.1)
    worker.join(timeout=2.0)

    assert not worker.is_alive()
    assert errors == []
    assert done.is_set()


def test_pull_recovers_backup_before_nonforce_conflict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from saklas.io import hf_manifolds as hfm

    source = _author_fake_manifold(tmp_path, monkeypatch)
    target = tmp_path / "installed" / "mood"
    backup = target.with_name("mood.bak")
    shutil.copytree(source, backup)
    monkeypatch.setattr(
        hfm, "_hf_snapshot_download",
        lambda **_kw: (_ for _ in ()).throw(
            AssertionError("existing recovered target should fail before download")
        ),
    )

    with pytest.raises(hfm.HFError, match="exists"):
        hfm.pull_manifold("alice/mood", target, force=False)

    assert target.exists()
    assert not backup.exists()
    assert next(target.glob("*.safetensors")).exists()


def test_force_pull_recovered_backup_waits_for_logical_pair_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from saklas.io import hf_manifolds as hfm
    from saklas.io.manifold_folder import manifold_pair_lock

    source = _author_fake_manifold(tmp_path, monkeypatch)
    monkeypatch.setattr(hfm, "_hf_snapshot_download", lambda **_kw: str(source))
    target = tmp_path / "installed" / "mood"
    backup = target.with_name("mood.bak")
    shutil.copytree(source, backup)
    logical_tensor = target / next(backup.glob("*.safetensors")).name
    started = threading.Event()
    done = threading.Event()

    def pull() -> None:
        started.set()
        hfm.pull_manifold("alice/mood", target, force=True)
        done.set()

    with manifold_pair_lock(logical_tensor):
        worker = threading.Thread(target=pull)
        worker.start()
        assert started.wait(1.0)
        assert not done.wait(0.1)
    worker.join(timeout=2.0)

    assert not worker.is_alive()
    assert done.is_set()
    assert not backup.exists()


def _capture_push_staging(monkeypatch: pytest.MonkeyPatch):
    """Hook tempfile.mkdtemp + shutil.rmtree to snapshot the push staging dir."""
    from saklas.io import hf_manifolds as hfm

    captured_dir: list[Path] = []
    staged: dict[str, bytes] = {}
    real_mkdtemp = __import__("tempfile").mkdtemp

    def spy_mkdtemp(**kw: Any):
        d = real_mkdtemp(**kw)
        captured_dir.append(Path(d))
        return d

    monkeypatch.setattr("tempfile.mkdtemp", spy_mkdtemp)
    import shutil as _sh
    orig_rmtree = _sh.rmtree

    def capture(path: Any, *a: Any, **kw: Any):
        if captured_dir and Path(path) == captured_dir[0]:
            for p in Path(path).rglob("*"):
                if p.is_file():
                    staged[p.relative_to(path).as_posix()] = p.read_bytes()
        return orig_rmtree(path, *a, **kw)

    monkeypatch.setattr(hfm.shutil, "rmtree", capture)
    return staged


def test_push_manifold_dry_run_stages_corpus_and_card(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from saklas.io import hf_manifolds as hfm
    folder = _author_fake_manifold(tmp_path, monkeypatch)
    staged = _capture_push_staging(monkeypatch)

    url, sha = hfm.push_manifold(folder, "alice/mood", dry_run=True)
    assert url == "https://huggingface.co/alice/mood"
    assert sha is None
    # Corpus always staged.
    assert "manifold.json" in staged
    assert "nodes/00_calm.json" in staged
    assert "nodes/01_uneasy.json" in staged
    # Fitted tensor + sidecar staged.
    assert "google__gemma-2-2b-it.safetensors" in staged
    assert "google__gemma-2-2b-it.json" in staged
    # README + gitattributes written.
    assert ".gitattributes" in staged
    card = staged["README.md"].decode()
    assert "library_name: saklas" in card
    assert "saklas-manifold" in card
    assert "google/gemma-2-2b-it" in card          # base_model frontmatter
    assert "base_model_relation: adapter" in card
    assert "`calm`" in card                          # node labels in body


def test_push_manifold_uploads_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from saklas.io import hf_manifolds as hfm
    folder = _author_fake_manifold(tmp_path, monkeypatch)

    api = MagicMock()
    upload = MagicMock()
    upload.oid = "deadbeefcafe"
    api.upload_folder.return_value = upload
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: api)

    url, sha = hfm.push_manifold(folder, "alice/mood")
    assert sha == "deadbeefcafe"
    api.create_repo.assert_called_once()
    # Tag is saklas-manifold, repo_type model.
    _, kwargs = api.create_repo.call_args
    assert kwargs["repo_type"] == "model"
    assert api.upload_folder.call_count == 1
    _, up_kwargs = api.upload_folder.call_args
    assert up_kwargs["repo_type"] == "model"


def test_push_manifold_model_scope_and_variant_filter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from saklas.io import hf_manifolds as hfm
    from saklas.io.manifolds import ManifoldFolder

    folder = _author_fake_manifold(tmp_path, monkeypatch)
    # Add a second model's tensor + an SAE variant for the first model.
    for model_id, release in (
        ("meta/llama-3-8b", None),
        ("google/gemma-2-2b-it", "gemma-scope"),
    ):
        _write_fitted_manifold(folder, model_id, release=release)
    ManifoldFolder.load(folder)

    staged = _capture_push_staging(monkeypatch)
    # model_scope = gemma, variant = raw → only the canonical gemma tensor.
    hfm.push_manifold(
        folder, "alice/mood",
        model_scope="google/gemma-2-2b-it", variant="raw", dry_run=True,
    )
    tensors = sorted(k for k in staged if k.endswith(".safetensors"))
    assert tensors == ["google__gemma-2-2b-it.safetensors"]
    # Corpus still present despite the tensor filter.
    assert "manifold.json" in staged
    assert "nodes/00_calm.json" in staged
    # Staged manifest re-hashed to match the filtered file set.
    staged_manifest = json.loads(staged["manifold.json"])
    assert "google__gemma-2-2b-it.safetensors" in staged_manifest["files"]
    assert "meta__llama-3-8b.safetensors" not in staged_manifest["files"]
    assert "google__gemma-2-2b-it_sae-gemma-scope.safetensors" not in staged_manifest["files"]


def test_push_manifold_corpus_only_when_unfitted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """An unfitted manifold (no tensors) still pushes — corpus alone re-fits."""
    from saklas.io import hf_manifolds as hfm
    from saklas.io.manifolds import create_manifold_folder

    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    nodes = [
        {"label": label, "coords": [i / 2.0], "statements": [f"{label} s"]}
        for i, label in enumerate(["a", "b", "c"])
    ]
    folder, _ = create_manifold_folder("local", "bare", "", domain, nodes)
    staged = _capture_push_staging(monkeypatch)
    url, sha = hfm.push_manifold(folder, "alice/bare", dry_run=True)
    assert sha is None
    assert "manifold.json" in staged
    assert "nodes/00_a.json" in staged
    assert not any(k.endswith(".safetensors") for k in staged)


def test_push_manifold_rejects_fitted_tensor_after_corpus_edit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from saklas.io import hf_manifolds as hfm
    from saklas.io.manifolds import ManifoldFormatError

    folder = _author_fake_manifold(tmp_path, monkeypatch)
    node = folder / "nodes" / "00_calm.json"
    node.write_text(json.dumps(["changed source corpus"]))

    with pytest.raises(ManifoldFormatError, match="stale"):
        hfm.push_manifold(folder, "alice/mood", dry_run=True)


# ========================================================== legacy-pack port ===
#
# 4.0 back-compat: `manifold install` ports a published/local legacy
# saklas-pack (statements-bearing, no manifold.json) to a 2-node `pca`
# manifold. A bare control-vector repo (no statements, no manifest) is refused.


def test_install_manifold_ports_local_legacy_pack(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path / "home"))
    from saklas.io.hf_manifolds import install_manifold
    from saklas.io.manifolds import ManifoldFolder
    from saklas.io.paths import manifold_dir

    # A legacy v2 vector folder: statements.json of {positive, negative} pairs.
    legacy = tmp_path / "legacy" / "happy.sad"
    legacy.mkdir(parents=True)
    (legacy / "statements.json").write_text(json.dumps([
        {"positive": "what a wonderful day", "negative": "everything is bleak"},
        {"positive": "i feel great", "negative": "i feel hopeless"},
    ]))

    dst = install_manifold(str(legacy))
    assert dst == manifold_dir("local", "happy.sad")
    mf = ManifoldFolder.load(dst)
    assert mf.fit_mode == "pca"
    assert sorted(mf.node_labels) == ["happy", "sad"]


def test_install_manifold_refuses_bare_control_vector_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path / "home"))
    from saklas.io.hf_manifolds import install_manifold

    # Only a safetensors dump — no manifold.json, no statements.json. The
    # geometry/authoring can't be recovered, so it's refused (re-author it).
    bare = tmp_path / "bare" / "control"
    bare.mkdir(parents=True)
    (bare / "model.safetensors").write_bytes(b"not a real tensor")
    with pytest.raises(ValueError, match="statements.json"):
        install_manifold(str(bare))
