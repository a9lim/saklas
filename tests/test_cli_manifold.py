"""CLI wiring tests for the ``saklas manifold`` lifecycle subverbs.

Parser shape + runner dispatch for the parity verbs added alongside the
fit/discover/generate/ls/show block: install / search / merge / push /
rm / clear / refresh / transfer, plus the ``ls -v`` and ``show -j``
changes.  The io-layer backends are mocked the way ``test_cli_flags``
mocks ``cache_ops`` / ``hf`` — these tests exercise the CLI plumbing
(arg parsing, runner→backend call shape, output idioms), not the
backends themselves (those live in ``test_manifolds_io`` /
``test_hf``).
"""
from __future__ import annotations

import json as _json
import threading
from pathlib import Path
from typing import Any

import pytest

from saklas import cli


def _materialize_bundles_for_cli_test(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    monkeypatch.setattr("saklas.io.manifolds._materialized_this_process", False)
    from saklas.io import selectors
    from saklas.io.manifolds import materialize_bundled_manifolds
    selectors.invalidate()
    materialize_bundled_manifolds()
    selectors.invalidate()


# ---------------------------------------------------------------------------
# Parser shape — each new subverb lands on the Namespace correctly
# ---------------------------------------------------------------------------

def test_parse_manifold_install():
    args = cli.parse_args(["pack", "install", "alice/circumplex"])
    assert args.command == "pack"
    assert args.pack_cmd == "install"
    assert args.target == "alice/circumplex"
    assert args.as_target is None
    assert args.force is False


def test_parse_manifold_install_flags():
    args = cli.parse_args([
        "pack", "install", "alice/circumplex",
        "-a", "local/mood", "-f",
    ])
    assert args.as_target == "local/mood"
    assert args.force is True


def test_parse_manifold_search():
    args = cli.parse_args(["pack", "search", "mood", "-j", "-v"])
    assert args.pack_cmd == "search"
    assert args.query == "mood"
    assert args.json_output is True
    assert args.verbose is True


def test_parse_manifold_search_empty_query():
    args = cli.parse_args(["pack", "search"])
    assert args.pack_cmd == "search"
    assert args.query == ""


def test_parse_manifold_merge():
    args = cli.parse_args([
        "manifold", "merge", "combined", "a", "local/b", "c",
        "--method", "spectral", "-f",
    ])
    assert args.manifold_cmd == "merge"
    assert args.name == "combined"
    assert args.sources == ["a", "local/b", "c"]
    assert args.method == "spectral"
    assert args.force is True


def test_parse_manifold_merge_needs_at_least_one_source():
    # argparse requires >=1 positional source; the runner enforces >=2.
    with pytest.raises(SystemExit):
        cli.parse_args(["manifold", "merge", "combined"])


def test_parse_manifold_generate_seed():
    # ``--seed`` is parity with ``vector clone --seed`` (default unseeded).
    args = cli.parse_args([
        "manifold", "generate", "mood",
        "--concepts", "happy", "sad", "--seed", "42",
    ])
    assert args.manifold_cmd == "generate"
    assert args.seed == 42
    default = cli.parse_args([
        "manifold", "generate", "mood", "--concepts", "happy", "sad",
    ])
    assert default.seed is None


def test_require_model_reports_manifold_leaf_verb(capsys: pytest.CaptureFixture[str]):
    """The -m error names the manifold leaf verb (``manifold fit``), not just ``manifold``."""
    with pytest.raises(SystemExit) as ex:
        cli.main(["manifold", "fit", "/tmp/some_folder"])
    assert ex.value.code == 2
    assert "manifold fit: -m/--model is required" in capsys.readouterr().err


def test_parse_manifold_push():
    args = cli.parse_args([
        "pack", "push", "local/circumplex",
        "-a", "alice/circumplex", "-m", "google/gemma-3-4b-it",
        "--variant", "sae", "-p", "-d",
    ])
    assert args.pack_cmd == "push"
    assert args.selector == "local/circumplex"
    assert args.as_target == "alice/circumplex"
    assert args.model == "google/gemma-3-4b-it"
    assert args.variant == "sae"
    assert args.private is True
    assert args.dry_run is True


def test_parse_manifold_push_variant_default_raw():
    args = cli.parse_args(["pack", "push", "circumplex"])
    # Aligned with `pack push` — SAE variants are opt-in.
    assert args.variant == "raw"
    assert args.private is False
    assert args.dry_run is False


def test_parse_manifold_push_variant_rejects_unknown():
    with pytest.raises(SystemExit):
        cli.parse_args([
            "pack", "push", "circumplex", "--variant", "weird",
        ])


def test_parse_manifold_rm():
    args = cli.parse_args(["pack", "rm", "local/mood", "-y"])
    assert args.pack_cmd == "rm"
    assert args.selector == "local/mood"
    assert args.yes is True


def test_parse_manifold_clear():
    args = cli.parse_args([
        "pack", "clear", "circumplex",
        "-m", "foo/bar", "--variant", "raw",
    ])
    assert args.pack_cmd == "clear"
    assert args.selector == "circumplex"
    assert args.model == "foo/bar"
    assert args.variant == "raw"


def test_parse_manifold_clear_variant_default_all():
    args = cli.parse_args(["pack", "clear", "circumplex"])
    assert args.variant == "all"


def test_parse_manifold_refresh():
    args = cli.parse_args(["pack", "refresh", "alice/circumplex"])
    assert args.pack_cmd == "refresh"
    assert args.selector == "alice/circumplex"
    assert args.model is None


def test_parse_manifold_from_template():
    args = cli.parse_args([
        "manifold", "from-template", "weekday",
        "--name", "wd", "--fit-mode", "auto", "--max-dim", "3", "-f",
    ])
    assert args.manifold_cmd == "from-template"
    assert args.template == "weekday"
    assert args.name == "wd"
    assert args.fit_mode == "auto"
    assert args.max_dim == 3
    assert args.force is True


def test_parse_manifold_from_template_fit_mode_default_auto():
    args = cli.parse_args(["manifold", "from-template", "weekday"])
    assert args.fit_mode == "auto"
    assert args.name is None
    assert args.max_dim is None


def test_run_manifold_from_template_writes_folder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    """End-to-end: template create -> manifold from-template derives the folder."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    ctx_file = tmp_path / "ctx.json"
    ctx_file.write_text(_json.dumps([
        {"user": "what day is it?", "assistant": "today is [DAY]"},
        {"user": "which day?", "assistant": "it's [DAY]"},
    ]))
    cli.main([
        "template", "create", "weekday",
        "--slot", "[DAY]",
        "--values", "Monday", "Tuesday", "Wednesday",
        "--contexts", str(ctx_file),
    ])
    cli.main(["manifold", "from-template", "weekday", "--fit-mode", "auto"])
    out = capsys.readouterr().out
    assert "3 nodes x 2 contexts" in out

    from saklas.io.manifolds import ManifoldFolder
    from saklas.io.paths import manifold_dir
    mf = ManifoldFolder.load(manifold_dir("local", "weekday"))
    assert mf.fit_mode == "auto"
    assert mf.node_labels == ["monday", "tuesday", "wednesday"]
    assert mf.template_ref == "local/weekday"
    assert dict(mf.node_groups())["monday"] == ["today is Monday", "it's Monday"]


def test_run_manifold_from_template_missing_template_exits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    with pytest.raises(SystemExit):
        cli.main(["manifold", "from-template", "nonexistent"])


def test_parse_manifold_transfer():
    args = cli.parse_args([
        "manifold", "transfer", "circumplex",
        "--from", "google/gemma-3-4b-it", "--to", "Qwen/Qwen3-4B", "-f", "-j",
    ])
    assert args.manifold_cmd == "transfer"
    assert args.name == "circumplex"
    assert args.src_model == "google/gemma-3-4b-it"
    assert args.tgt_model == "Qwen/Qwen3-4B"
    assert args.force is True
    assert args.json_output is True


def test_parse_manifold_transfer_missing_from_errors():
    with pytest.raises(SystemExit):
        cli.parse_args([
            "manifold", "transfer", "circumplex",
            "--to", "Qwen/Qwen3-4B",
        ])


def test_parse_manifold_ls_verbose():
    args = cli.parse_args(["pack", "ls", "-v"])
    assert args.pack_cmd == "ls"
    assert args.verbose is True


def test_parse_pack_no_verb_lists_lifecycle_subverbs(capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit) as ex:
        cli.main(["pack"])
    assert ex.value.code == 0
    out = capsys.readouterr().out
    for verb in ("install", "search", "push", "rm", "clear",
                 "refresh", "ls", "show", "export"):
        assert verb in out


def test_parse_manifold_no_verb_lists_compute_subverbs(capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit) as ex:
        cli.main(["manifold"])
    assert ex.value.code == 0
    out = capsys.readouterr().out
    for verb in ("extract", "generate", "fit", "bake", "merge",
                 "transfer", "compare", "why"):
        assert verb in out


# ---------------------------------------------------------------------------
# Runner dispatch — each runner calls its backend with the mapped args
# ---------------------------------------------------------------------------

def test_run_manifold_install_calls_backend(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    calls: list[tuple[Any, ...]] = []

    def fake_install(target: str, as_: Any = None, *, force: bool = False) -> Path:
        calls.append((target, as_, force))
        return Path("/home/.saklas/manifolds/local/circumplex")

    monkeypatch.setattr("saklas.io.hf_manifolds.install_manifold", fake_install)
    cli.main(["pack", "install", "alice/circumplex", "-a", "local/mood", "-f"])
    assert calls == [("alice/circumplex", "local/mood", True)]
    out = capsys.readouterr().out
    assert "Installed alice/circumplex" in out


def test_run_manifold_search_calls_backend(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    seen: list[Any] = []

    def fake_search(query: Any) -> list[dict[str, Any]]:
        seen.append(query)
        return [{
            "name": "circumplex", "namespace": "alice", "description": "moods",
            "tags": [], "node_count": 9, "domain_label": "box(2d)",
            "fit_mode": "authored", "tensor_models": ["gemma"],
        }]

    monkeypatch.setattr("saklas.io.hf_manifolds.search_manifolds", fake_search)
    cli.main(["pack", "search", "mood"])
    assert seen == ["mood"]
    out = capsys.readouterr().out
    assert "circumplex" in out
    assert "box(2d)" in out


def test_run_manifold_search_empty_query_passes_none(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    seen: list[Any] = []

    def fake_search(query: Any) -> list[dict[str, Any]]:
        seen.append(query)
        return []

    monkeypatch.setattr("saklas.io.hf_manifolds.search_manifolds", fake_search)
    cli.main(["pack", "search"])
    # Empty CLI default coerces to None for the backend (list by recency).
    assert seen == [None]
    out = capsys.readouterr().out
    assert "(no matches)" in out


def test_run_manifold_search_json(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    rows = [{
        "name": "circumplex", "namespace": "alice", "description": "moods",
        "tags": [], "node_count": 9, "domain_label": "box(2d)",
        "fit_mode": "authored", "tensor_models": [],
    }]
    monkeypatch.setattr("saklas.io.hf_manifolds.search_manifolds", lambda q: rows)
    cli.main(["pack", "search", "mood", "-j"])
    out = capsys.readouterr().out
    data = _json.loads(out)
    assert data[0]["name"] == "circumplex"
    assert data[0]["domain_label"] == "box(2d)"


def test_run_manifold_merge_calls_backend(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    calls: list[dict[str, Any]] = []

    def fake_merge(target_ns: str, target_name: str, desc: str, *, sources: Any,
                   fit_mode: Any = None, force: bool = False) -> Path:
        calls.append({
            "ns": target_ns, "name": target_name, "desc": desc,
            "sources": sources, "fit_mode": fit_mode, "force": force,
        })
        return Path("/home/.saklas/manifolds/local/combined")

    monkeypatch.setattr("saklas.io.manifolds.merge_discover_manifolds", fake_merge)
    cli.main([
        "manifold", "merge", "local/combined", "a", "alice/b",
        "--method", "spectral", "-f",
    ])
    assert len(calls) == 1
    c = calls[0]
    assert c["ns"] == "local" and c["name"] == "combined"
    assert c["sources"] == [("local", "a"), ("alice", "b")]
    assert c["fit_mode"] == "spectral"
    assert c["force"] is True
    out = capsys.readouterr().out
    assert "Merged manifold written to" in out
    assert "fit local/combined" in out


def test_run_manifold_merge_one_source_errors(monkeypatch: pytest.MonkeyPatch):
    # The parser allows >=1 positional source; the runner refuses <2.
    monkeypatch.setattr(
        "saklas.io.manifolds.merge_discover_manifolds",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not call backend")),
    )
    with pytest.raises(SystemExit) as ex:
        cli.main(["manifold", "merge", "combined", "only-one"])
    assert ex.value.code == 2


def test_run_manifold_push_calls_backend(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    from saklas.io.paths import manifold_dir
    folder = manifold_dir("local", "circumplex")
    folder.mkdir(parents=True)
    (folder / "manifold.json").write_text("{}")

    calls: list[dict[str, Any]] = []

    def fake_push(folder_arg: Path, coord: str, *, private: bool = False,
                  model_scope: Any = None, variant: str = "all",
                  dry_run: bool = False) -> tuple[str, Any]:
        calls.append({
            "folder": folder_arg, "coord": coord, "private": private,
            "model_scope": model_scope, "variant": variant, "dry_run": dry_run,
        })
        return (f"https://huggingface.co/{coord}", "abcdef123456")

    monkeypatch.setattr("saklas.io.hf_manifolds.push_manifold", fake_push)
    # -a override means resolve_target_coord doesn't need whoami().
    cli.main([
        "pack", "push", "local/circumplex",
        "-a", "alice/circumplex", "-m", "google/gemma-3-4b-it",
        "--variant", "sae", "-p",
    ])
    assert len(calls) == 1
    c = calls[0]
    assert c["coord"] == "alice/circumplex"
    assert c["model_scope"] == "google/gemma-3-4b-it"
    assert c["variant"] == "sae"
    assert c["private"] is True
    assert c["dry_run"] is False
    out = capsys.readouterr().out
    assert "Pushed alice/circumplex" in out
    assert "abcdef123456" in out


def test_run_manifold_push_dry_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    from saklas.io.paths import manifold_dir
    folder = manifold_dir("local", "circumplex")
    folder.mkdir(parents=True)
    (folder / "manifold.json").write_text("{}")

    monkeypatch.setattr(
        "saklas.io.hf_manifolds.push_manifold",
        lambda f, coord, **k: (f"https://huggingface.co/{coord}", None),
    )
    cli.main([
        "pack", "push", "local/circumplex",
        "-a", "alice/circumplex", "-d",
    ])
    out = capsys.readouterr().out
    assert "Dry-run: would push alice/circumplex" in out


def test_run_manifold_push_missing_folder_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    monkeypatch.setattr(
        "saklas.io.hf_manifolds.push_manifold",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not call backend")),
    )
    with pytest.raises(SystemExit) as ex:
        cli.main(["pack", "push", "local/nope", "-a", "x/y"])
    assert ex.value.code == 1


def test_run_manifold_rm_calls_backend(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    calls: list[tuple[str, str]] = []

    def fake_rm(ns: str, name: str) -> dict[str, Any]:
        calls.append((ns, name))
        return {"namespace": ns, "name": name, "source": "local",
                "removed": True, "rematerializes_on_restart": False}

    monkeypatch.setattr("saklas.io.manifolds.remove_manifold_folder", fake_rm)
    cli.main(["pack", "rm", "local/mood"])
    assert calls == [("local", "mood")]
    out = capsys.readouterr().out
    assert "Removed local/mood" in out


def test_run_manifold_rm_bundled_refuses_without_yes(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "saklas.io.manifolds.remove_manifold_folder",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not call backend")),
    )
    with pytest.raises(SystemExit) as ex:
        cli.main(["pack", "rm", "default/personas"])
    assert ex.value.code == 2


def test_run_manifold_rm_bundled_with_yes(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    def fake_rm(ns: str, name: str) -> dict[str, Any]:
        return {"namespace": ns, "name": name, "source": "bundled",
                "removed": True, "rematerializes_on_restart": True}

    monkeypatch.setattr("saklas.io.manifolds.remove_manifold_folder", fake_rm)
    cli.main(["pack", "rm", "default/personas", "-y"])
    out = capsys.readouterr().out
    assert "Removed default/personas" in out
    assert "re-materializes" in out


def test_run_manifold_rm_missing_errors(monkeypatch: pytest.MonkeyPatch):
    def fake_rm(ns: str, name: str) -> dict[str, Any]:
        raise FileNotFoundError("manifold local/nope not found")

    monkeypatch.setattr("saklas.io.manifolds.remove_manifold_folder", fake_rm)
    with pytest.raises(SystemExit) as ex:
        cli.main(["pack", "rm", "local/nope"])
    assert ex.value.code == 1


def test_run_manifold_clear_calls_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    calls: list[dict[str, Any]] = []

    def fake_clear(ns: str, name: str, model_scope: Any = None, *, variant: str = "all") -> int:
        calls.append({"ns": ns, "name": name, "model_scope": model_scope, "variant": variant})
        return 3

    monkeypatch.setattr("saklas.io.manifolds.clear_manifold_tensors", fake_clear)
    _materialize_bundles_for_cli_test(monkeypatch, tmp_path)
    # Bare ``personas`` resolves cross-namespace to the bundled ``default/personas``
    # (the only installed match) — the lifecycle verbs no longer hard-default
    # a bare name to ``local/``.
    cli.main(["pack", "clear", "personas", "--variant", "raw"])
    assert calls == [
        {"ns": "default", "name": "personas", "model_scope": None, "variant": "raw"},
    ]
    out = capsys.readouterr().out
    assert "Deleted 3 files" in out


def test_run_manifold_clear_passes_model_scope(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    """``-m`` is now real — it threads through as ``model_scope`` (no warning)."""
    calls: list[dict[str, Any]] = []

    def fake_clear(ns: str, name: str, model_scope: Any = None, *, variant: str = "all") -> int:
        calls.append({"ns": ns, "name": name, "model_scope": model_scope, "variant": variant})
        return 1

    monkeypatch.setattr("saklas.io.manifolds.clear_manifold_tensors", fake_clear)
    _materialize_bundles_for_cli_test(monkeypatch, tmp_path)
    # Bare name resolves cross-namespace to bundled ``default/personas``.
    cli.main(["pack", "clear", "personas", "-m", "foo/bar"])
    assert calls == [
        {"ns": "default", "name": "personas", "model_scope": "foo/bar", "variant": "all"},
    ]
    captured = capsys.readouterr()
    assert "Deleted 1 files" in captured.out
    assert "ignored" not in captured.err


def test_run_manifold_refresh_tiers(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    for tier, fragment in (("skipped", "nothing to refresh"),
                           ("bundled", "bundled"),
                           ("hf", "re-pulled from HF")):
        monkeypatch.setattr(
            "saklas.io.manifolds.refresh_manifold",
            lambda ns, name, *, model_scope=None, t=tier: t,
        )
        cli.main(["pack", "refresh", "alice/circumplex"])
        out = capsys.readouterr().out
        assert fragment in out


def test_run_manifold_refresh_passes_model_scope(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    """``-m`` does a real scoped refresh now — threads ``model_scope``, no warning."""
    calls: list[dict[str, Any]] = []

    def fake_refresh(ns: str, name: str, *, model_scope: Any = None) -> str:
        calls.append({"ns": ns, "name": name, "model_scope": model_scope})
        return "scoped"

    monkeypatch.setattr("saklas.io.manifolds.refresh_manifold", fake_refresh)
    _materialize_bundles_for_cli_test(monkeypatch, tmp_path)
    # Bare name resolves cross-namespace to bundled ``default/personas``.
    cli.main(["pack", "refresh", "personas", "-m", "foo/bar"])
    assert calls == [{"ns": "default", "name": "personas", "model_scope": "foo/bar"}]
    captured = capsys.readouterr()
    assert "foo/bar" in captured.out and "re-fits on next use" in captured.out
    assert "no effect" not in captured.err


def test_run_manifold_refresh_missing_errors(monkeypatch: pytest.MonkeyPatch):
    def fake_refresh(ns: str, name: str, *, model_scope: Any = None) -> str:
        raise FileNotFoundError("not installed")

    monkeypatch.setattr("saklas.io.manifolds.refresh_manifold", fake_refresh)
    with pytest.raises(SystemExit) as ex:
        cli.main(["pack", "refresh", "local/nope"])
    assert ex.value.code == 1


def test_cold_alignment_reuses_loaded_target_neutrals_for_whitener(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Cold transfer neither revalidates identities nor reloads target rows."""
    import torch

    import saklas.core.model as model_mod
    import saklas.core.session as session_mod
    import saklas.io.alignment as alignment_mod
    from saklas.cli import runners

    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))

    class _SessionContext:
        def __init__(self, model_id: str) -> None:
            self.model_id = model_id
            self.model = object()
            self.tokenizer = object()
            self.layers = [object()]

        def __enter__(self) -> "_SessionContext":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(
        session_mod.SaklasSession,
        "from_pretrained",
        staticmethod(lambda model_id, **_kwargs: _SessionContext(model_id)),
    )
    loaded: list[str] = []
    src_acts = {0: torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 1.0]])}
    tgt_acts = {0: torch.tensor([[2.0, 1.0], [1.0, 3.0], [4.0, 2.0]])}

    def load_with_metadata(
        _model: object, _tokenizer: object, _layers: object, *,
        model_id: str, force: bool,
    ) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
        assert force is False
        loaded.append(model_id)
        acts = src_acts if model_id == "src/model" else tgt_acts
        return acts, {
            "format_version": 3,
            "capture_version": 1,
            "model_fingerprint": f"fp:{model_id}",
            "model_source_fingerprint": f"source:{model_id}",
            "capture_sha256": model_id,
            "tensor_sha256": {"0": "0" * 64},
            "tensor_files": {"0": "neutral.layer-0.safetensors"},
            "layers": [0],
            "tensor_schema": {"0": {"shape": [3, 2], "dtype": "torch.float32"}},
            "n_prompts": 3,
        }

    monkeypatch.setattr(
        alignment_mod,
        "_load_or_compute_neutral_activations_with_metadata_locked",
        load_with_metadata,
    )
    monkeypatch.setattr(
        alignment_mod, "validate_neutral_cache_metadata",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            FileNotFoundError("cold cache")
        ),
    )
    monkeypatch.setattr(model_mod, "model_source_fingerprint", lambda *_args: None)
    monkeypatch.setattr(
        alignment_mod, "fit_alignment",
        lambda _src, _tgt, **_kwargs: {0: torch.eye(2)},
    )
    monkeypatch.setattr(
        alignment_mod, "alignment_quality",
        lambda _maps, _src, _tgt: {0: 1.0},
    )
    monkeypatch.setattr(
        alignment_mod, "save_alignment_map",
        lambda *_args, **_kwargs: tmp_path / "alignment.safetensors",
    )
    maps, quality, _path, _src_id, _tgt_id, whitener = (
        runners._load_or_fit_transfer_alignment(
            "src/model", "tgt/model", force=True, label="test transfer",
        )
    )

    assert loaded == ["src/model", "tgt/model"]
    assert quality == {0: 1.0}
    assert torch.equal(maps[0], torch.eye(2))
    assert whitener.layers == {0}


def test_cold_narrow_alignment_releases_full_seed_rosters_before_fit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    import gc
    import weakref

    import torch

    import saklas.core.model as model_mod
    import saklas.core.session as session_mod
    import saklas.io.alignment as alignment_mod
    from saklas.cli import runners

    class _SessionContext:
        def __init__(self, model_id: str) -> None:
            self.model_id = model_id
            self.model = object()
            self.tokenizer = object()
            self.layers = [object() for _ in range(10)]

        def __enter__(self) -> "_SessionContext":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(
        session_mod.SaklasSession,
        "from_pretrained",
        staticmethod(lambda model_id, **_kwargs: _SessionContext(model_id)),
    )
    monkeypatch.setattr(model_mod, "model_source_fingerprint", lambda *_args: None)
    monkeypatch.setattr(
        alignment_mod, "validate_neutral_cache_metadata",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            FileNotFoundError("cold cache")
        ),
    )

    unrequested_refs: list[weakref.ReferenceType[torch.Tensor]] = []

    def capture(
        _model: object, _tokenizer: object, _layers: object, *,
        model_id: str, force: bool,
    ) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
        assert force is False
        rows = {
            layer: torch.tensor(
                [[1.0 + layer, 0.0], [0.0, 1.0], [2.0, 1.0]],
            )
            for layer in range(10)
        }
        unrequested_refs.append(weakref.ref(rows[9]))
        return rows, {
            "format_version": 3,
            "capture_version": 1,
            "model_fingerprint": f"fp:{model_id}",
            "model_source_fingerprint": f"source:{model_id}",
            "capture_sha256": model_id,
            "tensor_sha256": {str(layer): "0" * 64 for layer in range(10)},
            "tensor_files": {
                str(layer): f"neutral.layer-{layer}.safetensors"
                for layer in range(10)
            },
            "layers": list(range(10)),
            "tensor_schema": {
                str(layer): {
                    "shape": [3, 2], "dtype": "torch.float32",
                }
                for layer in range(10)
            },
            "n_prompts": 3,
        }

    monkeypatch.setattr(
        alignment_mod,
        "_load_or_compute_neutral_activations_with_metadata_locked",
        capture,
    )
    monkeypatch.setattr(alignment_mod, "load_alignment_map", lambda *_a, **_k: None)

    def fit(
        src: dict[int, torch.Tensor], tgt: dict[int, torch.Tensor], **kwargs: Any,
    ) -> dict[int, torch.Tensor]:
        gc.collect()
        assert len(unrequested_refs) == 2
        assert all(ref() is None for ref in unrequested_refs)
        assert set(src) == set(tgt) == {5}
        assert kwargs["requested_layers"] == {5}
        return {5: torch.eye(2)}

    monkeypatch.setattr(alignment_mod, "fit_alignment", fit)
    monkeypatch.setattr(
        alignment_mod, "alignment_quality", lambda *_args, **_kwargs: {5: 1.0},
    )
    monkeypatch.setattr(
        alignment_mod, "save_alignment_map",
        lambda *_args, **_kwargs: tmp_path / "alignment.safetensors",
    )

    result = runners._load_or_fit_transfer_alignment(
        "src/model", "tgt/model", force=True, label="test transfer",
        requested_layers=[5],
    )

    assert set(result[0]) == {5}
    assert result[5].layers == {5}


def test_cached_alignment_keeps_model_free_offline_whitener_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A cached repeat loads target rows once and never loads model weights."""
    import torch

    import saklas.core.model as model_mod
    import saklas.core.session as session_mod
    import saklas.io.alignment as alignment_mod
    from saklas.cli import runners

    def sidecar(model_id: str) -> dict[str, Any]:
        return {
            "format_version": 3,
            "capture_version": 1,
            "model_fingerprint": f"fp:{model_id}",
            "model_source_fingerprint": f"source:{model_id}",
            "capture_sha256": model_id,
            "tensor_sha256": {"0": "0" * 64},
            "tensor_files": {"0": "neutral.layer-0.safetensors"},
            "layers": [0],
            "tensor_schema": {"0": {"shape": [3, 2], "dtype": "torch.float32"}},
            "n_prompts": 3,
        }

    metadata_calls: list[str] = []
    payload_calls: list[str] = []
    target_acts = {0: torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])}

    def metadata_only(model_id: str, **_kwargs: Any) -> dict[str, Any]:
        metadata_calls.append(model_id)
        return sidecar(model_id)

    def load_validated(
        model_id: str, **_kwargs: Any,
    ) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
        payload_calls.append(model_id)
        assert model_id == "tgt/model"
        return target_acts, sidecar(model_id)

    monkeypatch.setattr(
        alignment_mod, "validate_neutral_cache_metadata", metadata_only,
    )
    monkeypatch.setattr(
        alignment_mod, "load_validated_neutral_cache", load_validated,
    )
    monkeypatch.setattr(
        model_mod, "model_source_fingerprint",
        lambda model_id: f"source:{model_id}",
    )
    monkeypatch.setattr(
        alignment_mod, "load_alignment_map",
        lambda *_args, **_kwargs: (
            {0: torch.eye(2)}, {"quality_per_layer": {"0": 0.75}},
        ),
    )
    monkeypatch.setattr(
        alignment_mod, "alignment_cache_path",
        lambda *_args: (
            tmp_path / "alignment.safetensors", tmp_path / "alignment.json",
        ),
    )
    monkeypatch.setattr(
        session_mod.SaklasSession,
        "from_pretrained",
        staticmethod(lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("exact cached repeat loaded a model")
        )),
    )
    result = runners._load_or_fit_transfer_alignment(
        "src/model", "tgt/model", force=False, label="test transfer",
    )

    assert result[1] == {0: 0.75}
    assert result[5].layers == {0}
    assert metadata_calls == ["src/model", "tgt/model", "tgt/model"]
    assert payload_calls == ["tgt/model"]


def test_missing_cached_alignment_fits_offline_from_proven_neutral_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A map miss reuses both neutral caches and never loads either model."""
    import torch

    import saklas.core.model as model_mod
    import saklas.core.session as session_mod
    import saklas.io.alignment as alignment_mod
    from saklas.cli import runners

    src_acts = {0: torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 1.0]])}
    tgt_acts = {0: torch.tensor([[2.0, 1.0], [1.0, 3.0], [4.0, 2.0]])}

    def sidecar(model_id: str) -> dict[str, Any]:
        return {
            "format_version": 3,
            "capture_version": 1,
            "model_fingerprint": f"fp:{model_id}",
            "model_source_fingerprint": f"source:{model_id}",
            "capture_sha256": model_id,
            "tensor_sha256": {"0": "0" * 64},
            "tensor_files": {"0": "neutral.layer-0.safetensors"},
            "layers": [0],
            "tensor_schema": {
                "0": {"shape": [3, 2], "dtype": "torch.float32"},
            },
            "n_prompts": 3,
        }

    metadata_calls: list[str] = []
    payload_calls: list[str] = []

    def metadata_only(model_id: str, **_kwargs: Any) -> dict[str, Any]:
        metadata_calls.append(model_id)
        return sidecar(model_id)

    def load_validated(
        model_id: str, **_kwargs: Any,
    ) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
        payload_calls.append(model_id)
        return (
            src_acts if model_id == "src/model" else tgt_acts,
            sidecar(model_id),
        )

    monkeypatch.setattr(
        alignment_mod, "validate_neutral_cache_metadata", metadata_only,
    )
    monkeypatch.setattr(
        alignment_mod, "load_validated_neutral_cache", load_validated,
    )
    monkeypatch.setattr(
        model_mod, "model_source_fingerprint",
        lambda model_id: f"source:{model_id}",
    )
    monkeypatch.setattr(
        alignment_mod, "load_alignment_map", lambda *_args, **_kwargs: None,
    )
    fitted: list[tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]] = []

    def fit_offline(
        src: dict[int, torch.Tensor], tgt: dict[int, torch.Tensor], **_kwargs: Any,
    ) -> dict[int, torch.Tensor]:
        fitted.append((src, tgt))
        return {0: torch.eye(2)}

    monkeypatch.setattr(alignment_mod, "fit_alignment", fit_offline)
    monkeypatch.setattr(
        alignment_mod, "alignment_quality",
        lambda _maps, _src, _tgt: {0: 0.9},
    )
    saved: list[dict[str, Any]] = []

    def save_offline(*_args: Any, **kwargs: Any) -> Path:
        saved.append(kwargs)
        return tmp_path / "alignment.safetensors"

    monkeypatch.setattr(alignment_mod, "save_alignment_map", save_offline)
    monkeypatch.setattr(
        session_mod.SaklasSession,
        "from_pretrained",
        staticmethod(lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("offline alignment fit loaded model weights")
        )),
    )

    result = runners._load_or_fit_transfer_alignment(
        "src/model", "tgt/model", force=False, label="test transfer",
    )

    assert metadata_calls == [
        "src/model", "tgt/model", "tgt/model", "src/model",
    ]
    assert payload_calls == ["tgt/model", "src/model"]
    assert len(fitted) == 1
    assert fitted[0][0] is src_acts
    assert set(fitted[0][1]) == {0}
    assert fitted[0][1][0] is tgt_acts[0]
    assert len(saved) == 1
    assert saved[0]["source_identity"] == result[3]
    assert saved[0]["target_identity"] == result[4]
    assert result[1] == {0: 0.9}
    assert result[5].layers == {0}


def test_force_refits_only_requested_layer_without_loading_models(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    import torch

    import saklas.core.model as model_mod
    import saklas.core.session as session_mod
    import saklas.io.alignment as alignment_mod
    from saklas.cli import runners

    def sidecar(model_id: str) -> dict[str, Any]:
        return {
            "format_version": 3, "capture_version": 1,
            "model_fingerprint": f"fp:{model_id}",
            "model_source_fingerprint": f"source:{model_id}",
            "capture_sha256": model_id,
            "tensor_sha256": {str(layer): "0" * 64 for layer in range(10)},
            "tensor_files": {
                str(layer): f"neutral.layer-{layer}.safetensors"
                for layer in range(10)
            },
            "layers": list(range(10)),
            "tensor_schema": {
                str(layer): {"shape": [3, 2], "dtype": "torch.float32"}
                for layer in range(10)
            },
            "n_prompts": 3,
        }

    rows = {
        model_id: {
            layer: torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
            for layer in range(10)
        }
        for model_id in ("src/model", "tgt/model")
    }
    payload_calls: list[tuple[str, set[int] | None]] = []

    def load_rows(
        model_id: str, *, requested_layers: object = None,
    ) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
        requested = (
            None if requested_layers is None
            else {int(layer) for layer in requested_layers}  # type: ignore[union-attr]
        )
        payload_calls.append((model_id, requested))
        selected = set(rows[model_id]) if requested is None else requested
        return ({layer: rows[model_id][layer] for layer in selected}, sidecar(model_id))

    monkeypatch.setattr(
        alignment_mod, "validate_neutral_cache_metadata",
        lambda model_id, **_kwargs: sidecar(model_id),
    )
    monkeypatch.setattr(alignment_mod, "load_validated_neutral_cache", load_rows)
    monkeypatch.setattr(
        model_mod, "model_source_fingerprint",
        lambda model_id: f"source:{model_id}",
    )
    requested_map_layers: list[set[int]] = []

    def load_map(*_args: Any, **kwargs: Any) -> tuple[dict[int, Any], dict[str, Any]]:
        requested_map_layers.append(set(kwargs["requested_layers"]))
        return ({}, {"quality_per_layer": {"0": 0.5}})

    monkeypatch.setattr(alignment_mod, "load_alignment_map", load_map)
    fitted: list[tuple[set[int], dict[str, Any]]] = []

    def fit(src: dict[int, Any], _tgt: dict[int, Any], **kwargs: Any) -> dict[int, Any]:
        fitted.append((set(src), kwargs))
        return {5: torch.eye(2)}

    monkeypatch.setattr(alignment_mod, "fit_alignment", fit)
    monkeypatch.setattr(
        alignment_mod, "alignment_quality", lambda *_args, **_kwargs: {5: 0.9},
    )
    saved: list[dict[str, Any]] = []
    monkeypatch.setattr(
        alignment_mod, "save_alignment_map",
        lambda *_args, **kwargs: saved.append(kwargs) or tmp_path / "alignment",
    )
    monkeypatch.setattr(
        session_mod.SaklasSession, "from_pretrained",
        staticmethod(lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("force refit recaptured neutral activations")
        )),
    )

    result = runners._load_or_fit_transfer_alignment(
        "src/model", "tgt/model", force=True, label="test transfer",
        requested_layers=[5],
    )
    assert requested_map_layers == [set()]
    assert payload_calls == [("tgt/model", {5}), ("src/model", {5})]
    assert fitted[0][0] == {5}
    assert fitted[0][1]["requested_layers"] == {5}
    assert fitted[0][1]["available_shared_layers"] == set(range(10))
    assert saved[0]["extend"] is True
    assert set(result[0]) == {5} and result[5].layers == {5}


def test_target_neutral_generation_race_replans_cached_alignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    import saklas.core.model as model_mod
    import saklas.core.session as session_mod
    import saklas.io.alignment as alignment_mod
    from saklas.cli import runners

    target_generation = "old"

    def sidecar(model_id: str) -> dict[str, Any]:
        generation = target_generation if model_id == "tgt/model" else "stable"
        return {
            "format_version": 3, "capture_version": 1,
            "model_fingerprint": f"fp:{model_id}:{generation}",
            "model_source_fingerprint": f"source:{model_id}",
            "capture_sha256": generation,
            "tensor_sha256": {"0": "0" * 64},
            "tensor_files": {"0": "neutral.layer-0.safetensors"},
            "layers": [0],
            "tensor_schema": {"0": {"shape": [3, 2], "dtype": "torch.float32"}},
            "n_prompts": 3,
        }

    rows = {0: torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])}

    def load_rows(
        model_id: str, *, requested_layers: object = None,
    ) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
        nonlocal target_generation
        del requested_layers
        if model_id == "tgt/model" and target_generation == "old":
            target_generation = "new"
            raise ValueError("target cache changed during payload load")
        return rows, sidecar(model_id)

    monkeypatch.setattr(
        alignment_mod, "validate_neutral_cache_metadata",
        lambda model_id, **_kwargs: sidecar(model_id),
    )
    monkeypatch.setattr(alignment_mod, "load_validated_neutral_cache", load_rows)
    monkeypatch.setattr(
        model_mod, "model_source_fingerprint",
        lambda model_id: f"source:{model_id}",
    )
    map_generations: list[str] = []

    def load_map(*_args: Any, **kwargs: Any):
        generation = kwargs["target_identity"]["capture_sha256"]
        map_generations.append(generation)
        factor = 1.0 if generation == "old" else 2.0
        return (
            {0: factor * torch.eye(2)},
            {"quality_per_layer": {"0": 0.8, "9": -5.0}},
        )

    monkeypatch.setattr(alignment_mod, "load_alignment_map", load_map)
    monkeypatch.setattr(
        session_mod.SaklasSession, "from_pretrained",
        staticmethod(lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("identity replan loaded a model")
        )),
    )

    result = runners._load_or_fit_transfer_alignment(
        "src/model", "tgt/model", force=False, label="test",
        requested_layers=[0],
    )

    assert map_generations == ["old", "new"]
    assert torch.equal(result[0][0].to_dense(), 2.0 * torch.eye(2))
    assert result[1] == {0: 0.8}
    assert result[4]["capture_sha256"] == "new"


def test_concurrent_distinct_alignments_single_flight_shared_model_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    import torch

    import saklas.core.model as model_mod
    import saklas.core.session as session_mod
    import saklas.io.alignment as alignment_mod
    from saklas.cli import runners

    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    cached: dict[str, tuple[dict[int, torch.Tensor], dict[str, Any]]] = {}
    loads: dict[str, int] = {}

    def sidecar(model_id: str) -> dict[str, Any]:
        return {
            "format_version": 3, "capture_version": 1,
            "model_fingerprint": f"fp:{model_id}",
            "model_source_fingerprint": f"source:{model_id}",
            "capture_sha256": model_id,
            "tensor_sha256": {str(layer): "0" * 64 for layer in range(10)},
            "tensor_files": {
                str(layer): f"neutral.layer-{layer}.safetensors"
                for layer in range(10)
            },
            "layers": list(range(10)), "tensor_schema": {}, "n_prompts": 3,
        }

    def metadata(model_id: str, **_kwargs: Any) -> dict[str, Any]:
        if model_id not in cached:
            raise FileNotFoundError(model_id)
        return cached[model_id][1]

    def load_rows(
        model_id: str, *, requested_layers: object = None,
    ) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
        rows, sc = cached[model_id]
        if requested_layers is None:
            return rows, sc
        wanted = {int(layer) for layer in requested_layers}  # type: ignore[union-attr]
        return {layer: rows[layer] for layer in wanted if layer in rows}, sc

    class Session:
        def __init__(self, model_id: str) -> None:
            loads[model_id] = loads.get(model_id, 0) + 1
            self.model_id = model_id
            self.model = object()
            self.tokenizer = object()
            self.layers = [object()]

        def __enter__(self) -> "Session":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    def capture(
        _model: object, _tokenizer: object, _layers: object, *,
        model_id: str, force: bool,
    ) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
        assert force is False
        rows = {0: torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])}
        cached[model_id] = (rows, sidecar(model_id))
        return cached[model_id]

    monkeypatch.setattr(alignment_mod, "validate_neutral_cache_metadata", metadata)
    monkeypatch.setattr(alignment_mod, "load_validated_neutral_cache", load_rows)
    monkeypatch.setattr(
        alignment_mod, "_load_or_compute_neutral_activations_with_metadata_locked",
        capture,
    )
    monkeypatch.setattr(
        model_mod, "model_source_fingerprint",
        lambda model_id: f"source:{model_id}",
    )
    monkeypatch.setattr(
        session_mod.SaklasSession, "from_pretrained",
        staticmethod(lambda model_id, **_kwargs: Session(model_id)),
    )
    monkeypatch.setattr(alignment_mod, "load_alignment_map", lambda *_a, **_k: None)
    monkeypatch.setattr(
        alignment_mod, "fit_alignment",
        lambda *_args, **_kwargs: {0: torch.eye(2)},
    )
    monkeypatch.setattr(
        alignment_mod, "alignment_quality", lambda *_args: {0: 1.0},
    )
    monkeypatch.setattr(
        alignment_mod, "save_alignment_map",
        lambda *_args, **_kwargs: tmp_path / "alignment",
    )
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def run(target: str) -> None:
        try:
            barrier.wait()
            runners._load_or_fit_transfer_alignment(
                "shared/source", target, force=False, label="test",
                requested_layers=[0],
            )
        except BaseException as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=run, args=(target,))
        for target in ("target/a", "target/b")
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    assert all(not thread.is_alive() for thread in threads)
    assert errors == []
    assert loads == {"shared/source": 1, "target/a": 1, "target/b": 1}


def test_cross_process_distinct_alignments_single_flight_shared_model_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The filesystem neutral lock precedes model construction across commands."""
    import multiprocessing
    import time
    import torch

    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("cross-process monkeypatch inheritance requires fork")
    ctx = multiprocessing.get_context("fork")

    import saklas.core.model as model_mod
    import saklas.core.session as session_mod
    import saklas.io.alignment as alignment_mod
    from saklas.cli import runners

    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    source_marker = tmp_path / "source-neutral-ready"
    load_count = ctx.Value("i", 0)
    start = ctx.Barrier(2)
    errors = ctx.Queue()
    rows = {0: torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])}

    def sidecar(model_id: str) -> dict[str, Any]:
        return {
            "format_version": 3, "capture_version": 1,
            "model_fingerprint": f"fp:{model_id}",
            "model_source_fingerprint": f"source:{model_id}",
            "capture_sha256": model_id,
            "tensor_sha256": {"0": "0" * 64},
            "tensor_files": {"0": "neutral.layer-0.safetensors"},
            "layers": [0],
            "tensor_schema": {"0": {"shape": [3, 2], "dtype": "torch.float32"}},
            "n_prompts": 3,
        }

    def metadata(model_id: str, **_kwargs: Any) -> dict[str, Any]:
        if model_id == "shared/source" and not source_marker.exists():
            raise FileNotFoundError(model_id)
        return sidecar(model_id)

    def load_rows(
        model_id: str, *, requested_layers: object = None,
    ) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
        del requested_layers
        if model_id == "shared/source" and not source_marker.exists():
            raise FileNotFoundError(model_id)
        return rows, sidecar(model_id)

    class Session:
        def __init__(self, model_id: str) -> None:
            assert model_id == "shared/source"
            with load_count.get_lock():
                load_count.value += 1
            self.model = object()
            self.tokenizer = object()
            self.layers = [object()]

        def __enter__(self) -> "Session":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    def capture(
        _model: object, _tokenizer: object, _layers: object, *,
        model_id: str, force: bool,
    ) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
        assert model_id == "shared/source" and force is False
        time.sleep(0.15)
        source_marker.write_text("ready")
        return rows, sidecar(model_id)

    monkeypatch.setattr(alignment_mod, "validate_neutral_cache_metadata", metadata)
    monkeypatch.setattr(alignment_mod, "load_validated_neutral_cache", load_rows)
    monkeypatch.setattr(
        alignment_mod, "_load_or_compute_neutral_activations_with_metadata_locked",
        capture,
    )
    monkeypatch.setattr(
        model_mod, "model_source_fingerprint",
        lambda model_id: f"source:{model_id}",
    )
    monkeypatch.setattr(
        session_mod.SaklasSession, "from_pretrained",
        staticmethod(lambda model_id, **_kwargs: Session(model_id)),
    )
    monkeypatch.setattr(
        alignment_mod, "load_alignment_map",
        lambda *_args, **_kwargs: (
            {0: torch.eye(2)}, {"quality_per_layer": {"0": 1.0}},
        ),
    )

    def worker(target: str) -> None:
        try:
            start.wait()
            runners._load_or_fit_transfer_alignment(
                "shared/source", target, force=False, label="test",
                requested_layers=[0],
            )
        except BaseException as exc:
            errors.put(repr(exc))

    processes = [
        ctx.Process(target=worker, args=(target,))
        for target in ("target/a", "target/b")
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=10)

    assert all(not process.is_alive() for process in processes)
    assert [process.exitcode for process in processes] == [0, 0]
    assert errors.empty()
    assert load_count.value == 1


def test_run_manifold_transfer_calls_backend(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    from saklas.io.paths import manifold_dir, safe_model_id
    folder = manifold_dir("local", "circumplex")
    folder.mkdir(parents=True)
    (folder / "manifold.json").write_text("{}")
    src_model = "google/gemma-3-4b-it"
    (folder / f"{safe_model_id(src_model)}.safetensors").write_bytes(b"x")
    from saklas.io.manifolds import TransferSourceProof

    source_proof = TransferSourceProof(
        tensor_name="source.safetensors",
        tensor_sha256="a" * 64,
        sidecar_sha256="b" * 64,
        layers=(14, 15),
    )
    monkeypatch.setattr(
        "saklas.io.manifolds.preflight_transfer_manifold",
        lambda *_args, **_kwargs: source_proof,
    )

    import torch
    from saklas.io.alignment import LayerAlignment

    fake_M = {
        layer: LayerAlignment(torch.eye(4), torch.eye(4), torch.zeros(4))
        for layer in (14, 15)
    }
    target_whitener = object()
    alignment_calls: list[dict[str, Any]] = []

    def fake_alignment(*_args: Any, **kwargs: Any):
        alignment_calls.append(kwargs)
        return (
            fake_M, {14: 0.9, 15: 0.8}, tmp_path / "alignment.safetensors",
            {"model_fingerprint": "src-fp"},
            {"model_fingerprint": "tgt-fp"},
            target_whitener,
        )

    monkeypatch.setattr(
        "saklas.cli.runners._load_or_fit_transfer_alignment", fake_alignment,
    )

    calls: list[dict[str, Any]] = []

    def fake_transfer(folder_arg: Path, *, from_model: str, to_model: str,
                      alignment: Any, transfer_quality_estimate: Any = None,
                      source_model_fingerprint: Any = None,
                      target_model_fingerprint: Any = None,
                      whitener: Any = None, layer_means: Any = None,
                      force: bool = False,
                      expected_source_proof: Any = None) -> Path:
        calls.append({
            "folder": folder_arg, "from": from_model, "to": to_model,
            "layers": sorted(alignment.keys()), "quality": transfer_quality_estimate,
            "force": force, "whitener": whitener,
            "source_proof": expected_source_proof,
        })
        return folder_arg / "Qwen__Qwen3-4B_from-google__gemma-3-4b-it.safetensors"

    monkeypatch.setattr("saklas.io.manifolds.transfer_manifold", fake_transfer)
    cli.main([
        "manifold", "transfer", "local/circumplex",
        "--from", src_model, "--to", "Qwen/Qwen3-4B",
    ])
    assert len(calls) == 1
    c = calls[0]
    assert c["from"] == src_model
    assert c["to"] == "Qwen/Qwen3-4B"
    assert c["layers"] == [14, 15]
    assert c["whitener"] is target_whitener
    assert alignment_calls[0]["requested_layers"] == [14, 15]
    assert c["source_proof"] is source_proof
    # Median of {0.9, 0.8} = 0.85.
    assert abs(c["quality"] - 0.85) < 1e-9
    out = capsys.readouterr().out
    assert "Transferred manifold local/circumplex" in out
    assert "2 shared" in out


def test_run_manifold_transfer_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    from saklas.io.paths import manifold_dir, safe_model_id
    folder = manifold_dir("local", "circumplex")
    folder.mkdir(parents=True)
    (folder / "manifold.json").write_text("{}")
    src_model = "src/model"
    (folder / f"{safe_model_id(src_model)}.safetensors").write_bytes(b"x")
    monkeypatch.setattr(
        "saklas.io.manifolds.preflight_transfer_manifold",
        lambda *_args, **_kwargs: None,
    )

    import torch
    target_whitener = object()
    monkeypatch.setattr(
        "saklas.cli.runners._load_or_fit_transfer_alignment",
        lambda *args, **kwargs: (
            {10: torch.eye(2)}, {10: 0.5}, tmp_path / "alignment.safetensors",
            {"model_fingerprint": "src-fp"},
            {"model_fingerprint": "tgt-fp"},
            target_whitener,
        ),
    )
    monkeypatch.setattr(
        "saklas.io.manifolds.transfer_manifold",
        lambda f, **k: f / "out.safetensors",
    )
    cli.main([
        "manifold", "transfer", "local/circumplex",
        "--from", src_model, "--to", "tgt/model", "-j",
    ])
    out = capsys.readouterr().out
    data = _json.loads(out)
    assert data["source_model"] == src_model
    assert data["target_model"] == "tgt/model"
    assert data["transferred_layers"] == [10]
    assert abs(data["median_transfer_quality"] - 0.5) < 1e-9


def test_run_manifold_transfer_retries_unproven_target(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A post-pair manifest failure must not make the CLI demand ``-f``."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    from saklas.io.paths import manifold_dir, safe_model_id, tensor_filename

    folder = manifold_dir("local", "circumplex")
    folder.mkdir(parents=True)
    (folder / "manifold.json").write_text("{}")
    src_model = "src/model"
    tgt_model = "tgt/model"
    (folder / f"{safe_model_id(src_model)}.safetensors").write_bytes(b"source")
    monkeypatch.setattr(
        "saklas.io.manifolds.preflight_transfer_manifold",
        lambda *_args, **_kwargs: None,
    )

    import torch

    target_whitener = object()
    monkeypatch.setattr(
        "saklas.cli.runners._load_or_fit_transfer_alignment",
        lambda *args, **kwargs: (
            {0: torch.eye(2)}, {0: 0.5}, tmp_path / "alignment.safetensors",
            {"model_fingerprint": "src-fp"},
            {"model_fingerprint": "tgt-fp"},
            target_whitener,
        ),
    )
    target = folder / tensor_filename(tgt_model, transferred_from=src_model)
    calls = 0

    def fail_once(folder_arg: Path, **_kwargs: Any) -> Path:
        nonlocal calls
        calls += 1
        if calls == 1:
            # Model the durable pair / missing manifest-proof crash window.
            target.write_bytes(b"unproven")
            target.with_suffix(".json").write_text("{}")
            raise RuntimeError("injected post-pair manifest failure")
        return target

    monkeypatch.setattr("saklas.io.manifolds.transfer_manifold", fail_once)
    argv = [
        "manifold", "transfer", "local/circumplex",
        "--from", src_model, "--to", tgt_model,
    ]
    with pytest.raises(RuntimeError, match="post-pair manifest"):
        cli.main(argv)
    cli.main(argv)

    assert calls == 2


def test_run_manifold_transfer_trusted_target_skips_alignment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    from saklas.io.paths import manifold_dir

    folder = manifold_dir("local", "circumplex")
    folder.mkdir(parents=True)
    (folder / "manifold.json").write_text("{}")
    monkeypatch.setattr(
        "saklas.io.manifolds.preflight_transfer_manifold",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            FileExistsError("trusted transferred target exists")
        ),
    )
    monkeypatch.setattr(
        "saklas.cli.runners._load_or_fit_transfer_alignment",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("trusted target performed alignment work")
        ),
    )

    with pytest.raises(SystemExit) as exc:
        cli.main([
            "manifold", "transfer", "local/circumplex",
            "--from", "src/model", "--to", "tgt/model",
        ])

    assert exc.value.code == 1


def test_run_manifold_transfer_missing_source_fit_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    from saklas.io.paths import manifold_dir
    folder = manifold_dir("local", "circumplex")
    folder.mkdir(parents=True)
    (folder / "manifold.json").write_text("{}")
    # No source safetensors → runner exits before any alignment work.
    with pytest.raises(SystemExit) as ex:
        cli.main([
            "manifold", "transfer", "local/circumplex",
            "--from", "src/model", "--to", "tgt/model",
        ])
    assert ex.value.code == 1


# ---------------------------------------------------------------------------
# Cross-namespace bare-name lifecycle resolution (clear/refresh/rm/transfer)
# ---------------------------------------------------------------------------

def test_lifecycle_bare_name_not_found_exits(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """A bare name with no installed match exits 1, no backend call."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    monkeypatch.setattr(
        "saklas.io.manifolds.clear_manifold_tensors",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not call backend")),
    )
    with pytest.raises(SystemExit) as ex:
        cli.main(["pack", "clear", "nope_not_installed"])
    assert ex.value.code == 1


def test_lifecycle_bare_name_resolves_local(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    """A bare name uniquely installed under ``local/`` resolves there."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    _author_circumplex_lite(tmp_path)  # writes local/moodlite

    calls: list[tuple[str, str]] = []

    def fake_clear(ns: str, name: str, model_scope: Any = None, *, variant: str = "all") -> int:
        calls.append((ns, name))
        return 0

    monkeypatch.setattr("saklas.io.manifolds.clear_manifold_tensors", fake_clear)
    cli.main(["pack", "clear", "moodlite"])
    assert calls == [("local", "moodlite")]


def test_lifecycle_bare_name_ambiguous_exits(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """A bare name installed in two namespaces raises an ambiguity error (exit 2)."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    from saklas.io.manifolds import create_manifold_folder
    domain_spec = {"type": "box", "axes": [{"min": -1.0, "max": 1.0, "periodic": False}]}
    nodes = [
        {"label": "low", "coords": [-1.0], "statements": ["s."]},
        {"label": "mid", "coords": [0.0], "statements": ["s."]},
        {"label": "high", "coords": [1.0], "statements": ["s."]},
    ]
    for ns in ("local", "alice"):
        create_manifold_folder(ns, "dup", "d", domain_spec, nodes)

    monkeypatch.setattr(
        "saklas.io.manifolds.clear_manifold_tensors",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not call backend")),
    )
    with pytest.raises(SystemExit) as ex:
        cli.main(["pack", "clear", "dup"])
    assert ex.value.code == 2


def test_lifecycle_explicit_ns_pins_without_walk(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """An explicit ``ns/name`` pins verbatim — no existence pre-check, backend runs."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))  # nothing installed
    calls: list[tuple[str, str]] = []

    def fake_clear(ns: str, name: str, model_scope: Any = None, *, variant: str = "all") -> int:
        calls.append((ns, name))
        return 0

    monkeypatch.setattr("saklas.io.manifolds.clear_manifold_tensors", fake_clear)
    cli.main(["pack", "clear", "alice/ghost"])
    assert calls == [("alice", "ghost")]


# ---------------------------------------------------------------------------
# ls -v / show -j changes
# ---------------------------------------------------------------------------

def _author_circumplex_lite(home: Path) -> Path:
    """Write a minimal authored 1-D box manifold folder under SAKLAS_HOME.

    Two nodes on an open axis (min_nodes(1) = 3 isn't enforced until fit,
    and ``ManifoldFolder.load`` enforces it for authored folders — so use
    enough nodes to pass load).  Just enough to drive ls / show.
    """
    from saklas.io.manifolds import create_manifold_folder
    domain_spec = {"type": "box", "axes": [{"min": -1.0, "max": 1.0, "periodic": False}]}
    nodes = [
        {"label": "low", "coords": [-1.0], "statements": ["a statement here."]},
        {"label": "mid", "coords": [0.0], "statements": ["a statement here."]},
        {"label": "high", "coords": [1.0], "statements": ["a statement here."]},
    ]
    folder, _adv = create_manifold_folder(
        "local", "moodlite", "a tiny test mood manifold", domain_spec, nodes,
    )
    return folder


def test_run_manifold_ls_verbose_shows_description(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    _author_circumplex_lite(tmp_path)
    from saklas.io import selectors
    selectors.invalidate()
    cli.main(["pack", "ls", "--namespace", "local", "-v"])
    out = capsys.readouterr().out
    assert "local/moodlite" in out
    assert "a tiny test mood manifold" in out


def test_run_manifold_ls_non_verbose_hides_description(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    _author_circumplex_lite(tmp_path)
    from saklas.io import selectors
    selectors.invalidate()
    cli.main(["pack", "ls", "--namespace", "local"])
    out = capsys.readouterr().out
    assert "local/moodlite" in out
    assert "a tiny test mood manifold" not in out


def test_run_manifold_show_json_uses_summary_keys(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    _author_circumplex_lite(tmp_path)
    from saklas.io import selectors
    selectors.invalidate()
    cli.main(["pack", "show", "local/moodlite", "-j"])
    out = capsys.readouterr().out
    data = _json.loads(out)
    # manifold_summary's contract keys (shared with the server route).
    for key in ("namespace", "name", "description", "source", "fit_mode",
                "is_discover", "domain", "domain_label", "intrinsic_dim",
                "min_nodes", "node_count", "node_labels", "node_coords",
                "node_roles", "hyperparams", "fitted_models", "tensor_variants"):
        assert key in data, f"missing summary key {key!r}"
    assert data["namespace"] == "local"
    assert data["name"] == "moodlite"
    assert data["node_count"] == 3
    assert data["node_labels"] == ["low", "mid", "high"]


def test_run_manifold_show_json_matches_summary_helper(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    """CLI ``show -j`` output is byte-equivalent to ``manifold_summary``."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    folder = _author_circumplex_lite(tmp_path)
    from saklas.io import selectors
    selectors.invalidate()
    from saklas.io.manifolds import manifold_summary
    expected = manifold_summary(folder)
    cli.main(["pack", "show", "local/moodlite", "-j"])
    out = capsys.readouterr().out
    assert _json.loads(out) == expected
