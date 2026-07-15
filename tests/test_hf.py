from typing import Any
import hashlib
import json
import shutil
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from saklas.io import hf


def _write_fitted_manifold(
    folder: Path,
    model_id: str,
    *,
    release: str | None = None,
    direction: tuple[float, float] = (1.0, 0.0),
    publish_manifest: bool = True,
) -> Path:
    import torch

    from saklas.core.manifold import (
        MANIFOLD_FIT_POLICY_VERSION, save_manifold,
    )
    from saklas.core.vectors import fold_directions_to_subspace
    from saklas.io.manifolds import ManifoldFolder
    from saklas.io.paths import tensor_filename
    from tests._whitener import isotropic_whitener

    means = {0: torch.zeros(len(direction))}
    manifold = fold_directions_to_subspace(
        folder.name, {0: torch.tensor(direction)}, means, label="test",
        feature_space="raw" if release is None else f"sae-{release}",
        whitener=isotropic_whitener(means, len(direction)),
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
            sae_ids_by_layer={"0": f"{release}:layer-0"},
        )
    save_manifold(manifold, path, metadata)
    if publish_manifest:
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
    from saklas.io.paths import sidecar_filename, tensor_filename
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
    assert tensor_filename("google/gemma-2-2b-it") in staged
    assert sidecar_filename("google/gemma-2-2b-it") in staged
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
    from saklas.io.paths import tensor_filename

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
    assert tensors == [tensor_filename("google/gemma-2-2b-it")]
    # Corpus still present despite the tensor filter.
    assert "manifold.json" in staged
    assert "nodes/00_calm.json" in staged
    # Staged manifest re-hashed to match the filtered file set.
    staged_manifest = json.loads(staged["manifold.json"])
    assert tensor_filename("google/gemma-2-2b-it") in staged_manifest["files"]
    assert tensor_filename("meta/llama-3-8b") not in staged_manifest["files"]
    assert tensor_filename("google/gemma-2-2b-it", release="gemma-scope") not in staged_manifest["files"]


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


def test_push_manifold_stages_one_locked_source_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A concurrent fitted-pair replacement cannot split the push snapshot."""
    from saklas.io import hf_manifolds as hfm

    folder = _author_fake_manifold(tmp_path, monkeypatch)
    tensor = next(folder.glob("*.safetensors"))
    old_tensor = tensor.read_bytes()
    staged = _capture_push_staging(monkeypatch)
    real_write = hfm.write_bytes_atomic
    manifest_copied = threading.Event()
    continue_snapshot = threading.Event()

    def pause_after_manifest(path: Path, data: bytes) -> None:
        real_write(path, data)
        path = Path(path)
        if (
            path.name == "manifold.json"
            and path.parent.name.startswith("saklas-manifold-push-")
        ):
            manifest_copied.set()
            assert continue_snapshot.wait(2.0)

    monkeypatch.setattr(hfm, "write_bytes_atomic", pause_after_manifest)
    push_errors: list[BaseException] = []
    mutation_errors: list[BaseException] = []
    push_done = threading.Event()
    mutation_done = threading.Event()

    def push() -> None:
        try:
            hfm.push_manifold(folder, "alice/mood", dry_run=True)
        except BaseException as exc:  # pragma: no cover - asserted below
            push_errors.append(exc)
        finally:
            push_done.set()

    def replace_pair() -> None:
        try:
            _write_fitted_manifold(
                folder,
                "google/gemma-2-2b-it",
                direction=(0.0, 1.0),
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            mutation_errors.append(exc)
        finally:
            mutation_done.set()

    push_worker = threading.Thread(target=push)
    push_worker.start()
    assert manifest_copied.wait(1.0)

    mutation_worker = threading.Thread(target=replace_pair)
    mutation_worker.start()
    # The push owns the logical pair lock from validation through the tensor
    # copy, so replacement cannot enter the old manifest/tensor split window.
    assert not mutation_done.wait(0.1)

    continue_snapshot.set()
    push_worker.join(timeout=2.0)
    mutation_worker.join(timeout=2.0)

    assert not push_worker.is_alive()
    assert not mutation_worker.is_alive()
    assert push_done.is_set()
    assert mutation_done.is_set()
    assert push_errors == []
    assert mutation_errors == []
    assert staged[tensor.name] == old_tensor
    assert tensor.read_bytes() != old_tensor
    staged_manifest = json.loads(staged["manifold.json"])
    assert staged_manifest["files"][tensor.name] == hashlib.sha256(
        old_tensor,
    ).hexdigest()


def test_push_manifold_freezes_candidates_before_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A lower-level new pair born after the lock scan is not re-globbed."""
    from saklas.io import hf_manifolds as hfm

    folder = _author_fake_manifold(tmp_path, monkeypatch)
    staged = _capture_push_staging(monkeypatch)
    real_write = hfm.write_bytes_atomic
    injected: list[Path] = []

    def inject_after_manifest(path: Path, data: bytes) -> None:
        real_write(path, data)
        path = Path(path)
        if (
            not injected
            and path.name == "manifold.json"
            and path.parent.name.startswith("saklas-manifold-push-")
        ):
            injected.append(_write_fitted_manifold(
                folder, "new/model", publish_manifest=False,
            ))

    monkeypatch.setattr(hfm, "write_bytes_atomic", inject_after_manifest)
    hfm.push_manifold(folder, "alice/mood", dry_run=True)

    new_tensor = injected[0]
    assert new_tensor.is_file()
    assert new_tensor.name not in staged
    assert new_tensor.with_suffix(".json").name not in staged
    staged_manifest = json.loads(staged["manifold.json"])
    assert new_tensor.name not in staged_manifest["files"]
    assert new_tensor.with_suffix(".json").name not in staged_manifest["files"]


def test_install_manifold_requires_current_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path / "home"))
    from saklas.io.hf_manifolds import install_manifold

    source = tmp_path / "source" / "happy.sad"
    source.mkdir(parents=True)
    (source / "statements.json").write_text(json.dumps([
        {"positive": "what a wonderful day", "negative": "everything is bleak"},
    ]))
    with pytest.raises(ValueError, match="not a manifold"):
        install_manifold(str(source))


def test_install_manifold_refuses_bare_control_vector_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path / "home"))
    from saklas.io.hf_manifolds import install_manifold

    # Only a safetensors dump — no manifold.json, no statements.json. The
    # geometry/authoring can't be recovered, so it's refused (re-author it).
    bare = tmp_path / "bare" / "control"
    bare.mkdir(parents=True)
    (bare / "model.safetensors").write_bytes(b"not a real tensor")
    with pytest.raises(ValueError, match="not a manifold"):
        install_manifold(str(bare))


def test_force_local_install_onto_itself_is_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path / "home"))
    from saklas.io.hf_manifolds import install_manifold
    from saklas.io.manifolds import ManifoldFolder

    folder = _author_fake_manifold(tmp_path / "home", monkeypatch)
    manifest_before = (folder / "manifold.json").read_bytes()

    assert install_manifold(str(folder), force=True) == folder
    assert (folder / "manifold.json").read_bytes() == manifest_before
    ManifoldFolder.load(folder)


def test_local_as_install_rewrites_identity_and_resolves_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("SAKLAS_HOME", str(home))
    from saklas.io import hf_manifolds as hfm
    from saklas.core.manifold import load_manifold
    from saklas.io.manifolds import ManifoldFolder
    from saklas.io.paths import manifold_dir, tensor_filename
    from saklas.io.selectors import invalidate, parse, resolve

    source = _author_fake_manifold(home, monkeypatch, name="source")
    installed = hfm.install_manifold(
        str(source), as_="renamed/target", force=False,
    )

    assert installed == manifold_dir("renamed", "target")
    assert ManifoldFolder.load(installed).name == "target"
    fitted = load_manifold(installed / tensor_filename("google/gemma-2-2b-it"))
    assert fitted.name == "target"
    assert fitted.metadata["name"] == "target"
    invalidate()
    matches = resolve(parse("renamed/target"))
    assert [(m.namespace, m.name, m.folder) for m in matches] == [
        ("renamed", "target", installed),
    ]


def test_hf_as_install_rewrites_identity_and_resolves_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("SAKLAS_HOME", str(home))
    from saklas.io import hf_manifolds as hfm
    from saklas.core.manifold import load_manifold
    from saklas.io.manifolds import ManifoldFolder
    from saklas.io.paths import manifold_dir, tensor_filename
    from saklas.io.selectors import invalidate, parse, resolve

    source = _author_fake_manifold(home, monkeypatch, name="source")
    monkeypatch.setattr(
        hfm, "_hf_snapshot_download", lambda **_kw: str(source),
    )
    installed = hfm.install_manifold(
        "alice/source", as_="renamed/target", force=False,
    )

    assert installed == manifold_dir("renamed", "target")
    mf = ManifoldFolder.load(installed)
    assert mf.name == "target"
    assert mf.source == "hf://alice/source"
    fitted = load_manifold(installed / tensor_filename("google/gemma-2-2b-it"))
    assert fitted.name == "target"
    assert fitted.metadata["name"] == "target"
    invalidate()
    matches = resolve(parse("renamed/target"))
    assert [(m.namespace, m.name, m.folder) for m in matches] == [
        ("renamed", "target", installed),
    ]


@pytest.mark.parametrize(
    ("source_kind", "remove"),
    [
        ("local", "sidecar"),
        ("local", "both"),
        ("hf", "tensor"),
        ("hf", "both"),
    ],
)
def test_install_rejects_unmanifested_fitted_pair_during_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_kind: str,
    remove: str,
) -> None:
    """Install-as never blesses a fitted half absent from source proofs."""
    home = tmp_path / "home"
    monkeypatch.setenv("SAKLAS_HOME", str(home))
    from saklas.io import hf_manifolds as hfm
    from saklas.io.paths import manifold_dir

    source = _author_fake_manifold(home, monkeypatch, name="source")
    tensor = next(source.glob("*.safetensors"))
    sidecar = tensor.with_suffix(".json")
    manifest_path = source / "manifold.json"
    manifest = json.loads(manifest_path.read_text())
    if remove in {"tensor", "both"}:
        manifest["files"].pop(tensor.name)
    if remove in {"sidecar", "both"}:
        manifest["files"].pop(sidecar.name)
    manifest_path.write_text(json.dumps(manifest))

    target = manifold_dir("renamed", "target")
    if source_kind == "hf":
        monkeypatch.setattr(
            hfm, "_hf_snapshot_download", lambda **_kw: str(source),
        )
        with pytest.raises(hfm.HFError, match="untrusted fitted pair"):
            hfm.install_manifold(
                "alice/source", as_="renamed/target", force=False,
            )
    else:
        with pytest.raises(
            hfm.ManifoldInstallConflict, match="untrusted fitted pair",
        ):
            hfm.install_manifold(
                str(source), as_="renamed/target", force=False,
            )
    assert not target.exists()


def test_force_local_install_copy_failure_preserves_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("SAKLAS_HOME", str(home))
    from saklas.io import hf_manifolds as hfm
    from saklas.io.manifolds import ManifoldFolder

    source = _author_fake_manifold(home, monkeypatch, name="source")
    target = _author_fake_manifold(home, monkeypatch, name="target")
    manifest_before = (target / "manifold.json").read_bytes()
    real_copytree = hfm.shutil.copytree

    def fail_source_copy(src: Any, dst: Any, *args: Any, **kwargs: Any) -> Any:
        if Path(src) == source:
            raise OSError("injected local copy failure")
        return real_copytree(src, dst, *args, **kwargs)

    monkeypatch.setattr(hfm.shutil, "copytree", fail_source_copy)
    with pytest.raises(OSError, match="local copy failure"):
        hfm.install_manifold(str(source), as_="local/target", force=True)

    assert (target / "manifold.json").read_bytes() == manifest_before
    ManifoldFolder.load(target)
    assert not target.with_name("target.staging").exists()


def test_local_install_recovers_backup_before_nonforce_conflict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("SAKLAS_HOME", str(home))
    from saklas.io import hf_manifolds as hfm
    from saklas.io.manifolds import ManifoldFolder

    source = _author_fake_manifold(home, monkeypatch, name="source")
    target = _author_fake_manifold(home, monkeypatch, name="target")
    backup = target.with_name("target.bak")
    target.rename(backup)

    with pytest.raises(hfm.ManifoldInstallConflict, match="exists"):
        hfm.install_manifold(str(source), as_="local/target", force=False)

    assert target.exists()
    assert not backup.exists()
    assert ManifoldFolder.load(target).name == "target"


@pytest.mark.parametrize("suffix", [".staging", ".bak"])
def test_force_local_install_snapshots_reserved_sibling_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, suffix: str,
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("SAKLAS_HOME", str(home))
    from saklas.io import hf_manifolds as hfm
    from saklas.io.manifolds import ManifoldFolder
    from saklas.io.paths import manifold_dir

    source = _author_fake_manifold(home, monkeypatch, name="source")
    target = manifold_dir("local", "target")
    reserved_source = target.with_name(target.name + suffix)
    source.rename(reserved_source)

    installed = hfm.install_manifold(
        str(reserved_source), as_="local/target", force=True,
    )

    assert installed == target
    ManifoldFolder.load(target)
