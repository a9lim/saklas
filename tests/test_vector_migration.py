"""4.0 step 6e — legacy ``vectors/`` → 2-node ``pca`` manifold migration.

Three layers, all CPU-only (no model):

1. ``port_legacy_vector_folder`` (io) — the shared port primitive: a legacy
   concept folder's ``statements.json`` + ``scenarios.json`` reconstruct the
   equivalent 2-node manifold authoring; no tensors carried.
2. ``SaklasSession._port_stale_legacy_vector`` / ``_ensure_profile_registered``
   — port-on-detect in the live steer path (prefer-manifold), gated on
   staleness, with the actionable "fit it" raise (fitting can't re-enter the
   gen lock from dispatch).
3. ``scripts/upgrade_packs.py`` — the bulk migrator: port statements-bearing
   folders, re-stamp tensor-only ones to the current ``format_version``.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Generator

import pytest

from saklas.io import selectors as _sel
from saklas.io.manifolds import port_legacy_vector_folder
from saklas.io.packs import PACK_FORMAT_VERSION
from saklas.io.paths import concept_dir, manifold_dir, vectors_dir


@pytest.fixture(autouse=True)
def _isolated_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> Generator[None, None, None]:
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    _sel.invalidate()
    yield
    _sel.invalidate()


def _make_legacy_vector(
    ns: str,
    name: str,
    *,
    version: int = PACK_FORMAT_VERSION - 1,
    tags: list[str] | None = None,
    with_statements: bool = True,
    with_scenarios: bool = True,
    n_pairs: int = 6,
) -> Path:
    """Author a legacy ``vectors/<ns>/<name>/`` folder on disk."""
    folder = concept_dir(ns, name)
    folder.mkdir(parents=True, exist_ok=True)
    pack = {
        "name": name,
        "description": f"legacy {name}",
        "version": "1.0.0",
        "license": "unknown",
        "tags": tags or [],
        "recommended_alpha": 0.5,
        "source": "local",
        "files": {},
        "format_version": version,
    }
    (folder / "pack.json").write_text(json.dumps(pack))
    if with_statements:
        pairs = [
            {"positive": f"pos statement {i}", "negative": f"neg statement {i}"}
            for i in range(n_pairs)
        ]
        (folder / "statements.json").write_text(json.dumps(pairs))
    if with_scenarios:
        # Real vector extraction writes the dict form.
        (folder / "scenarios.json").write_text(
            json.dumps({"scenarios": ["scenario a", "scenario b"]})
        )
    return folder


# --------------------------------------------------------------------------
# 1. port_legacy_vector_folder (io primitive)
# --------------------------------------------------------------------------

class TestPortLegacyVectorFolder:
    def test_bipolar_splits_to_two_nodes(self) -> None:
        _make_legacy_vector("local", "happy.sad", tags=["affect"])
        target, mf = port_legacy_vector_folder(
            concept_dir("local", "happy.sad"), namespace="local"
        )

        assert target == manifold_dir("local", "happy.sad")
        assert (target / "manifold.json").exists()
        assert mf.fit_mode == "pca"
        assert mf.node_labels == ["happy", "sad"]
        # Hyperparams pin the rank-1 (K=2) affine fit.
        assert mf.hyperparams.get("max_dim") == 1
        # Tags carried from pack.json.
        assert mf.tags == ["affect"]

        groups = dict(mf.node_groups())
        assert groups["happy"] == [f"pos statement {i}" for i in range(6)]
        assert groups["sad"] == [f"neg statement {i}" for i in range(6)]

    def test_monopolar_gets_neg_node(self) -> None:
        _make_legacy_vector("local", "agentic")
        _target, mf = port_legacy_vector_folder(
            concept_dir("local", "agentic"), namespace="local"
        )
        assert mf.node_labels == ["agentic", "agentic_neg"]

    def test_no_tensors_carried(self) -> None:
        _make_legacy_vector("local", "happy.sad")
        target, _ = port_legacy_vector_folder(
            concept_dir("local", "happy.sad"), namespace="local"
        )
        # A ported manifold is unfitted — re-fits lazily / on demand.
        assert list(target.glob("*.safetensors")) == []

    def test_scenarios_ported_from_dict_form(self) -> None:
        # Regression: real vectors write ``{"scenarios": [...]}``; the port
        # must read the dict form, not only a bare list.
        _make_legacy_vector("local", "happy.sad", with_scenarios=True)
        target, _ = port_legacy_vector_folder(
            concept_dir("local", "happy.sad"), namespace="local"
        )
        scn = json.loads((target / "scenarios.json").read_text())
        assert scn["scenarios"] == ["scenario a", "scenario b"]

    def test_missing_statements_raises(self) -> None:
        _make_legacy_vector("local", "tensoronly", with_statements=False)
        with pytest.raises(FileNotFoundError):
            port_legacy_vector_folder(
                concept_dir("local", "tensoronly"), namespace="local"
            )

    def test_existing_manifold_without_force_raises(self) -> None:
        _make_legacy_vector("local", "happy.sad")
        port_legacy_vector_folder(
            concept_dir("local", "happy.sad"), namespace="local"
        )
        with pytest.raises(FileExistsError):
            port_legacy_vector_folder(
                concept_dir("local", "happy.sad"), namespace="local", force=False
            )

    def test_force_reports(self) -> None:
        _make_legacy_vector("local", "happy.sad")
        port_legacy_vector_folder(
            concept_dir("local", "happy.sad"), namespace="local"
        )
        # force=True re-ports cleanly.
        _target, mf = port_legacy_vector_folder(
            concept_dir("local", "happy.sad"), namespace="local", force=True
        )
        assert mf.node_labels == ["happy", "sad"]


# --------------------------------------------------------------------------
# 2. session port-on-detect (prefer-manifold)
# --------------------------------------------------------------------------

def _bare_session(model_id: str = "test/model"):
    """A SaklasSession with just the attributes the resolve path touches."""
    from saklas.core.session import SaklasSession

    sess = SaklasSession.__new__(SaklasSession)
    sess._profiles = {}
    sess._manifolds = {}
    sess._model_info = {"model_id": model_id}  # model_id is a read-only property
    return sess


class TestPortOnDetect:
    def test_stale_statements_folder_is_ported(self) -> None:
        _make_legacy_vector("local", "happy.sad", version=PACK_FORMAT_VERSION - 1)
        sess = _bare_session()
        hit = sess._port_stale_legacy_vector("local/happy.sad")
        assert hit == ("local", "happy.sad")
        assert (manifold_dir("local", "happy.sad") / "manifold.json").exists()

    def test_bare_name_scans_namespaces(self) -> None:
        _make_legacy_vector("alice", "happy.sad", version=PACK_FORMAT_VERSION - 1)
        sess = _bare_session()
        hit = sess._port_stale_legacy_vector("happy.sad")
        assert hit == ("alice", "happy.sad")

    def test_current_version_folder_not_ported(self) -> None:
        # A current-version pack keeps its tensor via autoload — don't port it.
        _make_legacy_vector("local", "fresh", version=PACK_FORMAT_VERSION)
        sess = _bare_session()
        assert sess._port_stale_legacy_vector("local/fresh") is None
        assert not (manifold_dir("local", "fresh") / "manifold.json").exists()

    def test_tensor_only_folder_not_ported(self) -> None:
        # No statements → can't re-fit → left for the autoload residue.
        _make_legacy_vector(
            "local", "tensoronly",
            version=PACK_FORMAT_VERSION - 1, with_statements=False,
        )
        sess = _bare_session()
        assert sess._port_stale_legacy_vector("local/tensoronly") is None

    def test_already_ported_returns_hit_without_reporting(self) -> None:
        _make_legacy_vector("local", "happy.sad", version=PACK_FORMAT_VERSION - 1)
        sess = _bare_session()
        sess._port_stale_legacy_vector("local/happy.sad")
        # Second call: manifold exists → returns the hit, no re-port crash.
        hit = sess._port_stale_legacy_vector("local/happy.sad")
        assert hit == ("local", "happy.sad")

    def test_ensure_profile_registered_ports_then_nudges_to_fit(self) -> None:
        from saklas.core.session import VectorNotRegisteredError

        _make_legacy_vector("local", "happy.sad", version=PACK_FORMAT_VERSION - 1)
        sess = _bare_session()
        with pytest.raises(VectorNotRegisteredError) as exc:
            sess._ensure_profile_registered("local/happy.sad")
        msg = str(exc.value)
        # The artifact was ported (file-only) ...
        assert (manifold_dir("local", "happy.sad") / "manifold.json").exists()
        # ... and the raise carries the exact fit command + model id.
        assert "saklas manifold fit local/happy.sad" in msg
        assert "test/model" in msg


# --------------------------------------------------------------------------
# 3. scripts/upgrade_packs.py (bulk migrator)
# --------------------------------------------------------------------------

def _load_migrator():
    repo = Path(__file__).resolve().parent.parent
    path = repo / "scripts" / "upgrade_packs.py"
    spec = importlib.util.spec_from_file_location("saklas_upgrade_packs", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestMigrationScript:
    def test_ports_statements_folder_and_removes_source(self) -> None:
        _make_legacy_vector("local", "happy.sad", version=PACK_FORMAT_VERSION - 1)
        mig = _load_migrator()
        hit = mig.migrate_folder(
            concept_dir("local", "happy.sad"), "local",
            keep_source=False, restamp_only=False,
        )
        assert hit == ("local", "happy.sad")
        assert (manifold_dir("local", "happy.sad") / "manifold.json").exists()
        # Source removed by default.
        assert not concept_dir("local", "happy.sad").exists()

    def test_keep_source_retains_legacy_folder(self) -> None:
        _make_legacy_vector("local", "happy.sad", version=PACK_FORMAT_VERSION - 1)
        mig = _load_migrator()
        mig.migrate_folder(
            concept_dir("local", "happy.sad"), "local",
            keep_source=True, restamp_only=False,
        )
        assert concept_dir("local", "happy.sad").exists()
        assert (manifold_dir("local", "happy.sad") / "manifold.json").exists()

    def test_tensor_only_folder_is_restamped(self) -> None:
        folder = _make_legacy_vector(
            "local", "tensoronly",
            version=PACK_FORMAT_VERSION - 1, with_statements=False,
        )
        # Author a fake tensor + stale sidecar so the re-stamp has work to do.
        (folder / "model.safetensors").write_bytes(b"\x00\x01")
        (folder / "model.json").write_text(
            json.dumps({"format_version": PACK_FORMAT_VERSION - 1, "method": "x"})
        )
        mig = _load_migrator()
        hit = mig.migrate_folder(
            folder, "local", keep_source=False, restamp_only=False,
        )
        assert hit is None  # not ported
        assert folder.exists()
        pack = json.loads((folder / "pack.json").read_text())
        assert pack["format_version"] == PACK_FORMAT_VERSION
        sidecar = json.loads((folder / "model.json").read_text())
        assert sidecar["format_version"] == PACK_FORMAT_VERSION
        # files map recomputed over the non-pack.json files.
        assert "model.safetensors" in pack["files"]

    def test_restamp_only_skips_porting(self) -> None:
        _make_legacy_vector("local", "happy.sad", version=PACK_FORMAT_VERSION - 1)
        mig = _load_migrator()
        hit = mig.migrate_folder(
            concept_dir("local", "happy.sad"), "local",
            keep_source=False, restamp_only=True,
        )
        assert hit is None
        # No manifold; the vectors folder is re-stamped instead.
        assert not (manifold_dir("local", "happy.sad") / "manifold.json").exists()
        pack = json.loads((concept_dir("local", "happy.sad") / "pack.json").read_text())
        assert pack["format_version"] == PACK_FORMAT_VERSION

    def test_idempotent_with_keep_source(self) -> None:
        _make_legacy_vector("local", "happy.sad", version=PACK_FORMAT_VERSION - 1)
        mig = _load_migrator()
        first = mig.migrate_folder(
            concept_dir("local", "happy.sad"), "local",
            keep_source=True, restamp_only=False,
        )
        second = mig.migrate_folder(
            concept_dir("local", "happy.sad"), "local",
            keep_source=True, restamp_only=False,
        )
        assert first == second == ("local", "happy.sad")

    def test_main_all_walks_vectors_root(self) -> None:
        _make_legacy_vector("local", "happy.sad", version=PACK_FORMAT_VERSION - 1)
        _make_legacy_vector("local", "angry.calm", version=PACK_FORMAT_VERSION - 1)
        assert vectors_dir().exists()
        mig = _load_migrator()
        rc = mig.main(["--all"])
        assert rc == 0
        assert (manifold_dir("local", "happy.sad") / "manifold.json").exists()
        assert (manifold_dir("local", "angry.calm") / "manifold.json").exists()
