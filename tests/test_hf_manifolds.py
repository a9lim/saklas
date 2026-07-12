"""CPU tests for ``saklas.io.hf_manifolds`` — the HF distribution surface.

Network-free: every huggingface_hub indirection (``_hf_snapshot_download`` /
``_hf_hub_download`` / ``_hf_api`` / ``HfApi``) is monkeypatched, so these run in
CI with no hub access and no model download.  The existing ``tests/test_hf.py``
already covers the happy push/pull/legacy-port paths; this file fills the gaps:

- ``search_manifolds`` — the tag-filtered list, the ``filter``→``tags`` fallback
  for older huggingface_hub, the per-row enrichment via ``fetch_manifold_info``,
  the ``_HF_SEARCH_CAP`` ceiling, and namespace/name splitting;
- ``fetch_manifold_info`` — the cheap manifest+listing probe, domain-label
  derivation (box / sphere / custom / discover), tensor-stem listing, and the
  format-version-too-new + transport-failure error mapping to ``HFError``;
- ``pull_manifold`` error wrapping — the ``_download`` failure → ``HFError`` and
  the no-``manifold.json`` rejection;
- ``install_manifold`` path/coord parsing + conflict handling — bad ``as_``,
  missing-slash coord, ``NAME_REGEX`` rejection, and the
  ``ManifoldInstallConflict`` (409) on an existing destination.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from saklas.io import hf_manifolds as hfm
from saklas.io.hf import HFError
from saklas.io.manifolds import MANIFOLD_FORMAT_VERSION


# --------------------------------------------------------------------------- #
# helpers: a fake HfApi + a manifest-on-disk authoring shortcut
# --------------------------------------------------------------------------- #


class _FakeModel:
    """A ``list_models`` row stub — only the attributes the row builder reads."""

    def __init__(self, mid: str, *, tags: list[str] | None = None, desc: str = ""):
        self.id = mid
        self.tags = tags
        self.description = desc


class _FakeApi:
    """Records ``list_models`` kwargs and serves canned rows / file listings.

    ``list_models_raises_on_filter`` models an older huggingface_hub that rejects
    the ``filter=`` kwarg with ``TypeError`` (the search path's fallback to
    ``tags=``).
    """

    def __init__(
        self,
        models: list[_FakeModel] | None = None,
        files: list[str] | None = None,
        *,
        list_models_raises_on_filter: bool = False,
    ) -> None:
        self._models = models or []
        self._files = files or []
        self._raise_on_filter = list_models_raises_on_filter
        self.list_models_calls: list[dict[str, Any]] = []
        self.list_repo_files_calls: list[dict[str, Any]] = []

    def list_models(self, **kwargs: Any) -> list[_FakeModel]:
        self.list_models_calls.append(kwargs)
        if self._raise_on_filter and "filter" in kwargs:
            raise TypeError("unexpected keyword argument 'filter'")
        return list(self._models)

    def list_repo_files(self, **kwargs: Any) -> list[str]:
        self.list_repo_files_calls.append(kwargs)
        return list(self._files)


def _write_manifest(folder: Path, payload: dict[str, Any]) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / "manifold.json"
    p.write_text(json.dumps(payload))
    return p


# =========================================================== fetch_manifold_info ===


def test_fetch_manifold_info_box_domain(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A box-domain authored manifold reports ``box(1d)`` + node count + tensors."""
    from saklas.io.paths import safe_model_id, tensor_filename

    manifest = _write_manifest(tmp_path / "repo", {
        "format_version": 6,
        "name": "months",
        "description": "month ring",
        "fit_mode": "authored",
        "domain": {"type": "box", "axes": [
            {"name": "t", "periodic": True, "lo": 0.0, "hi": 12.0}]},
        "nodes": [{"label": "january"}, {"label": "february"}],
        "tags": ["temporal"],
    })
    api = _FakeApi(files=["manifold.json", tensor_filename("google/gemma-2-2b-it"),
                          "nodes/00_january.json"])
    monkeypatch.setattr(hfm, "_hf_hub_download", lambda coord, fn, **kw: str(manifest))
    monkeypatch.setattr(hfm, "_hf_api", lambda: api)

    info = hfm.fetch_manifold_info("alice/months")
    assert info["name"] == "months"
    assert info["namespace"] == "alice"
    assert info["description"] == "month ring"
    assert info["node_count"] == 2
    assert info["fit_mode"] == "authored"
    assert info["domain_label"] == "box(1d)"
    assert info["tags"] == ["temporal"]
    # Only the .safetensors file becomes a tensor-stem entry.
    assert info["tensor_models"] == [safe_model_id("google/gemma-2-2b-it")]


def test_fetch_manifold_info_sphere_and_custom_and_discover(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """domain_label derivation across sphere / custom / discover shapes."""
    api = _FakeApi(files=["manifold.json"])
    monkeypatch.setattr(hfm, "_hf_api", lambda: api)

    cases = [
        ({"fit_mode": "authored",
          "domain": {"type": "sphere", "dim": 2}}, "sphere(2d)"),
        ({"fit_mode": "authored",
          "domain": {"type": "custom", "embed_dim": 5}}, "custom(5d)"),
        ({"fit_mode": "auto"}, "discover-auto"),
        ({"fit_mode": "pca"}, "discover-pca"),
    ]
    for i, (extra, expect) in enumerate(cases):
        payload = {"format_version": 6, "name": f"m{i}", "nodes": []}
        payload.update(extra)
        manifest = _write_manifest(tmp_path / f"repo{i}", payload)
        monkeypatch.setattr(hfm, "_hf_hub_download", lambda c, fn, _m=manifest, **kw: str(_m))
        info = hfm.fetch_manifold_info(f"alice/m{i}")
        assert info["domain_label"] == expect, (extra, info["domain_label"])


def test_fetch_manifold_info_rejects_newer_format(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    manifest = _write_manifest(tmp_path / "repo", {
        "format_version": MANIFOLD_FORMAT_VERSION + 1,
        "name": "future", "fit_mode": "pca", "nodes": [],
    })
    api = _FakeApi(files=["manifold.json"])
    monkeypatch.setattr(hfm, "_hf_hub_download", lambda c, fn, **kw: str(manifest))
    monkeypatch.setattr(hfm, "_hf_api", lambda: api)

    with pytest.raises(HFError, match="must be exactly"):
        hfm.fetch_manifold_info("alice/future")


@pytest.mark.parametrize("field", ["format_version", "name", "fit_mode"])
def test_fetch_manifold_info_requires_current_identity_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str,
) -> None:
    payload = {
        "format_version": MANIFOLD_FORMAT_VERSION,
        "name": "current",
        "fit_mode": "pca",
        "nodes": [],
    }
    payload.pop(field)
    manifest = _write_manifest(tmp_path / "repo", payload)
    api = _FakeApi(files=["manifold.json"])
    monkeypatch.setattr(hfm, "_hf_hub_download", lambda c, fn, **kw: str(manifest))
    monkeypatch.setattr(hfm, "_hf_api", lambda: api)

    with pytest.raises(HFError):
        hfm.fetch_manifold_info("alice/current")


def test_fetch_manifold_info_wraps_transport_failure(monkeypatch: pytest.MonkeyPatch):
    def boom(coord: str, fn: str, **kw: Any) -> str:
        raise RuntimeError("404 not found")

    monkeypatch.setattr(hfm, "_hf_hub_download", boom)
    with pytest.raises(HFError, match="fetch_manifold_info failed"):
        hfm.fetch_manifold_info("alice/missing")


def test_fetch_manifold_info_threads_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """A revision flows into both the download and the file-listing calls."""
    manifest = _write_manifest(tmp_path / "repo", {
        "format_version": 6, "name": "m", "fit_mode": "pca", "nodes": [],
    })
    api = _FakeApi(files=["manifold.json"])
    dl_kwargs: list[dict[str, Any]] = []

    def dl(coord: str, fn: str, **kw: Any) -> str:
        dl_kwargs.append(kw)
        return str(manifest)

    monkeypatch.setattr(hfm, "_hf_hub_download", dl)
    monkeypatch.setattr(hfm, "_hf_api", lambda: api)

    hfm.fetch_manifold_info("alice/m", revision="v2")
    assert dl_kwargs == [{"revision": "v2"}]
    assert api.list_repo_files_calls[0]["revision"] == "v2"


# =============================================================== search_manifolds ===


def test_search_manifolds_filters_by_tag_and_caps(monkeypatch: pytest.MonkeyPatch):
    """The query rides ``search=``, the tag rides ``filter=``, rows respect the cap."""
    # More rows than the cap so the [:_HF_SEARCH_CAP] slice bites.
    models = [
        _FakeModel(f"alice/m{i}", tags=["saklas-manifold"], desc=f"d{i}")
        for i in range(hfm._HF_SEARCH_CAP + 5)
    ]
    api = _FakeApi(models=models)
    monkeypatch.setattr(hfm, "_hf_api", lambda: api)

    rows = hfm.search_manifolds("mood")
    assert len(rows) == hfm._HF_SEARCH_CAP
    call = api.list_models_calls[0]
    assert call["filter"] == ["saklas-manifold"]
    assert call["search"] == "mood"
    assert call["limit"] == hfm._HF_SEARCH_CAP
    # description + tags present means no per-row fetch_manifold_info probe.
    first = rows[0]
    assert first["namespace"] == "alice"
    assert first["name"] == "m0"
    assert first["tags"] == ["saklas-manifold"]
    assert first["description"] == "d0"


def test_search_manifolds_empty_query_omits_search(monkeypatch: pytest.MonkeyPatch):
    api = _FakeApi(models=[_FakeModel("a/m", tags=["saklas-manifold"], desc="d")])
    monkeypatch.setattr(hfm, "_hf_api", lambda: api)
    hfm.search_manifolds(None)
    assert "search" not in api.list_models_calls[0]
    assert api.list_models_calls[0]["filter"] == ["saklas-manifold"]


def test_search_manifolds_falls_back_to_tags_kwarg(monkeypatch: pytest.MonkeyPatch):
    """Older huggingface_hub: ``filter=`` raises TypeError → retry with ``tags=``."""
    api = _FakeApi(
        models=[_FakeModel("a/m", tags=["saklas-manifold"], desc="d")],
        list_models_raises_on_filter=True,
    )
    monkeypatch.setattr(hfm, "_hf_api", lambda: api)
    rows = hfm.search_manifolds("q")
    assert len(rows) == 1
    # Two calls: the filter= attempt (raised) and the tags= retry.
    assert len(api.list_models_calls) == 2
    assert "filter" not in api.list_models_calls[1]
    assert api.list_models_calls[1]["tags"] == ["saklas-manifold"]


def test_search_manifolds_enriches_row_when_fields_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """A row missing description/tags triggers a fetch_manifold_info enrichment."""
    from saklas.io.paths import safe_model_id, tensor_filename

    api = _FakeApi(
        models=[_FakeModel("alice/months", tags=None, desc="")],
        files=["manifold.json", tensor_filename("google/gemma-2-2b-it")],
    )
    monkeypatch.setattr(hfm, "_hf_api", lambda: api)

    manifest = _write_manifest(tmp_path / "repo", {
        "format_version": 6, "name": "months", "description": "the year",
        "fit_mode": "authored",
        "domain": {"type": "box", "axes": [
            {"name": "t", "periodic": True, "lo": 0.0, "hi": 12.0}]},
        "nodes": [{"label": "january"}], "tags": ["temporal"],
    })
    monkeypatch.setattr(hfm, "_hf_hub_download", lambda c, fn, **kw: str(manifest))

    rows = hfm.search_manifolds(None)
    assert len(rows) == 1
    row = rows[0]
    # Enriched fields came from the manifest, not the (empty) list_models row.
    assert row["description"] == "the year"
    assert row["domain_label"] == "box(1d)"
    assert row["node_count"] == 1
    assert row["tensor_models"] == [safe_model_id("google/gemma-2-2b-it")]
    assert row["tags"] == ["temporal"]


def test_search_manifolds_enrichment_failure_is_best_effort(
    monkeypatch: pytest.MonkeyPatch,
):
    """If the per-row probe raises, the row still renders with defaults."""
    api = _FakeApi(models=[_FakeModel("alice/m", tags=None, desc="")])
    monkeypatch.setattr(hfm, "_hf_api", lambda: api)

    def boom(coord: str, rev: Any = None) -> dict[str, Any]:
        raise HFError("probe down")

    monkeypatch.setattr(hfm, "fetch_manifold_info", boom)
    rows = hfm.search_manifolds(None)
    assert len(rows) == 1
    row = rows[0]
    assert row["name"] == "m"
    assert row["namespace"] == "alice"
    assert row["node_count"] == 0           # default when enrichment fails
    assert row["domain_label"] == "?"
    assert row["fit_mode"] == "authored"


def test_search_manifolds_splits_namespaceless_id(monkeypatch: pytest.MonkeyPatch):
    """A bare (slash-free) repo id yields an empty namespace."""
    api = _FakeApi(models=[_FakeModel("solo", tags=["saklas-manifold"], desc="d")])
    monkeypatch.setattr(hfm, "_hf_api", lambda: api)
    rows = hfm.search_manifolds(None)
    assert rows[0]["namespace"] == ""
    assert rows[0]["name"] == "solo"


# ============================================================ pull_manifold errors ===


def test_pull_manifold_download_failure_wraps_hferror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """A snapshot-download failure maps to an ``HFError`` carrying the coord."""
    def boom(**kwargs: Any) -> str:
        raise RuntimeError("connection refused")

    monkeypatch.setattr(hfm, "_hf_snapshot_download", boom)
    with pytest.raises(HFError, match="alice/mood@v1: not found"):
        hfm.pull_manifold(
            "alice/mood", target_folder=tmp_path / "dst", force=False, revision="v1",
        )


def test_pull_manifold_no_manifest_no_statements_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """A repo with neither manifold.json nor statements.json is refused."""
    snap = tmp_path / "snapshot"
    snap.mkdir()
    (snap / "model.safetensors").write_bytes(b"bare tensor")
    monkeypatch.setattr(hfm, "_hf_snapshot_download", lambda **kw: str(snap))

    with pytest.raises(HFError, match="no manifold.json at root"):
        hfm.pull_manifold(
            "alice/bare", target_folder=tmp_path / "dst", force=False,
        )


# ========================================================== install_manifold parsing ===


def test_install_manifold_coord_requires_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path / "home"))
    with pytest.raises(ValueError, match="must be '<ns>/<name>"):
        hfm.install_manifold("noslash")


def test_install_manifold_bad_as_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path / "home"))
    with pytest.raises(ValueError, match="as_ must be"):
        hfm.install_manifold("alice/mood", as_="noslash")


def test_install_manifold_bad_name_regex_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path / "home"))
    # Uppercase violates NAME_REGEX (^[a-z]...).
    with pytest.raises(ValueError, match="NAME_REGEX"):
        hfm.install_manifold("alice/Mood")


def test_install_manifold_conflict_raises_409(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """An existing destination without ``force`` raises ManifoldInstallConflict."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path / "home"))
    from saklas.io.paths import manifold_dir

    dst = manifold_dir("local", "mood")
    dst.mkdir(parents=True)
    # _hf_snapshot_download must not even be reached — the conflict fires first.
    monkeypatch.setattr(
        hfm, "_hf_snapshot_download",
        lambda **kw: pytest.fail("download should not run on conflict"),
    )
    with pytest.raises(hfm.ManifoldInstallConflict) as ei:
        hfm.install_manifold("alice/mood")
    # 409 user_message contract.
    status, msg = ei.value.user_message()
    assert status == 409
    assert "already exists" in msg


def test_install_manifold_conflict_relocate_via_as(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """``as_`` retargets the destination namespace/name, dodging the conflict."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path / "home"))
    from saklas.io.paths import manifold_dir

    # Occupy local/mood, then install to alice/mood instead.
    manifold_dir("local", "mood").mkdir(parents=True)

    snap = tmp_path / "snapshot"
    snap.mkdir()
    # Minimal valid pull: a manifold.json the staged loader accepts.  We stub
    # pull_manifold itself to keep this focused on the as_ retarget routing.
    pulled: dict[str, Any] = {}

    def fake_pull(coord: str, target_folder: Path, *, force: bool, revision: Any) -> Path:
        pulled["coord"] = coord
        pulled["target"] = target_folder
        target_folder.mkdir(parents=True, exist_ok=True)
        return target_folder

    monkeypatch.setattr(hfm, "pull_manifold", fake_pull)
    dst = hfm.install_manifold("someone/mood", as_="alice/mood")
    assert dst == manifold_dir("alice", "mood")
    assert pulled["coord"] == "someone/mood"
    assert pulled["target"] == manifold_dir("alice", "mood")


def test_install_manifold_default_namespace_is_local(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """Without ``as_`` the destination namespace defaults to ``local``."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path / "home"))
    from saklas.io.paths import manifold_dir

    seen: dict[str, Any] = {}

    def fake_pull(coord: str, target_folder: Path, *, force: bool, revision: Any) -> Path:
        seen["target"] = target_folder
        target_folder.mkdir(parents=True, exist_ok=True)
        return target_folder

    monkeypatch.setattr(hfm, "pull_manifold", fake_pull)
    dst = hfm.install_manifold("alice/mood")
    assert dst == manifold_dir("local", "mood")
    assert seen["target"] == manifold_dir("local", "mood")


def test_install_manifold_revision_threads_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """``ns/name@rev`` splits the revision and forwards it to pull_manifold."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path / "home"))
    seen: dict[str, Any] = {}

    def fake_pull(coord: str, target_folder: Path, *, force: bool, revision: Any) -> Path:
        seen["coord"] = coord
        seen["revision"] = revision
        target_folder.mkdir(parents=True, exist_ok=True)
        return target_folder

    monkeypatch.setattr(hfm, "pull_manifold", fake_pull)
    hfm.install_manifold("alice/mood@v3.1")
    assert seen["coord"] == "alice/mood"
    assert seen["revision"] == "v3.1"


# ================================================ install_manifold local-folder routing ===


def test_install_manifold_local_folder_routes_to_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """An existing local directory routes to the folder-copy install path."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path / "home"))
    src = tmp_path / "srcdir"
    src.mkdir()

    routed: dict[str, Any] = {}

    def fake_local(p: Path, *, as_: Any = None, force: bool = False) -> Path:
        routed["src"] = p
        routed["as_"] = as_
        return tmp_path / "home" / "installed"

    monkeypatch.setattr(hfm, "_install_local_manifold", fake_local)
    hfm.install_manifold(str(src), as_="local/x")
    assert routed["src"] == src
    assert routed["as_"] == "local/x"


def test_install_local_manifold_bad_as_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """``_install_local_manifold`` rejects a slash-free ``as_`` before copying."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path / "home"))
    from saklas.io.manifolds import create_manifold_folder

    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    nodes = [
        {"label": label, "coords": [i / 2.0], "statements": [f"{label} s"]}
        for i, label in enumerate(["a", "b", "c"])
    ]
    folder, _ = create_manifold_folder("local", "src", "", domain, nodes)
    with pytest.raises(ValueError, match="as_ must be"):
        hfm._install_local_manifold(folder, as_="noslash")
