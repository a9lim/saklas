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
from pathlib import Path
from typing import Any

import pytest

from saklas import cli


# ---------------------------------------------------------------------------
# Parser shape — each new subverb lands on the Namespace correctly
# ---------------------------------------------------------------------------

def test_parse_manifold_install():
    args = cli.parse_args(["manifold", "install", "alice/circumplex"])
    assert args.command == "manifold"
    assert args.manifold_cmd == "install"
    assert args.target == "alice/circumplex"
    assert args.as_target is None
    assert args.force is False


def test_parse_manifold_install_flags():
    args = cli.parse_args([
        "manifold", "install", "alice/circumplex",
        "-a", "local/mood", "-f",
    ])
    assert args.as_target == "local/mood"
    assert args.force is True


def test_parse_manifold_search():
    args = cli.parse_args(["manifold", "search", "mood", "-j", "-v"])
    assert args.manifold_cmd == "search"
    assert args.query == "mood"
    assert args.json_output is True
    assert args.verbose is True


def test_parse_manifold_search_empty_query():
    args = cli.parse_args(["manifold", "search"])
    assert args.manifold_cmd == "search"
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
        "manifold", "push", "local/circumplex",
        "-a", "alice/circumplex", "-m", "google/gemma-3-4b-it",
        "--variant", "sae", "-p", "-d",
    ])
    assert args.manifold_cmd == "push"
    assert args.selector == "local/circumplex"
    assert args.as_target == "alice/circumplex"
    assert args.model == "google/gemma-3-4b-it"
    assert args.variant == "sae"
    assert args.private is True
    assert args.dry_run is True


def test_parse_manifold_push_variant_default_raw():
    args = cli.parse_args(["manifold", "push", "circumplex"])
    # Aligned with `pack push` — SAE variants are opt-in.
    assert args.variant == "raw"
    assert args.private is False
    assert args.dry_run is False


def test_parse_manifold_push_variant_rejects_unknown():
    with pytest.raises(SystemExit):
        cli.parse_args([
            "manifold", "push", "circumplex", "--variant", "weird",
        ])


def test_parse_manifold_rm():
    args = cli.parse_args(["manifold", "rm", "local/mood", "-y"])
    assert args.manifold_cmd == "rm"
    assert args.selector == "local/mood"
    assert args.yes is True


def test_parse_manifold_clear():
    args = cli.parse_args([
        "manifold", "clear", "circumplex",
        "-m", "foo/bar", "--variant", "raw",
    ])
    assert args.manifold_cmd == "clear"
    assert args.selector == "circumplex"
    assert args.model == "foo/bar"
    assert args.variant == "raw"


def test_parse_manifold_clear_variant_default_all():
    args = cli.parse_args(["manifold", "clear", "circumplex"])
    assert args.variant == "all"


def test_parse_manifold_refresh():
    args = cli.parse_args(["manifold", "refresh", "alice/circumplex"])
    assert args.manifold_cmd == "refresh"
    assert args.selector == "alice/circumplex"
    assert args.model is None


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
    args = cli.parse_args(["manifold", "ls", "-v"])
    assert args.manifold_cmd == "ls"
    assert args.verbose is True


def test_parse_manifold_no_verb_lists_new_subverbs(capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit) as ex:
        cli.main(["manifold"])
    assert ex.value.code == 0
    out = capsys.readouterr().out
    for verb in ("install", "search", "merge", "push", "rm", "clear",
                 "refresh", "transfer"):
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
    cli.main(["manifold", "install", "alice/circumplex", "-a", "local/mood", "-f"])
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
    cli.main(["manifold", "search", "mood"])
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
    cli.main(["manifold", "search"])
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
    cli.main(["manifold", "search", "mood", "-j"])
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
    assert "discover local/combined" in out


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
        "manifold", "push", "local/circumplex",
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
        "manifold", "push", "local/circumplex",
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
        cli.main(["manifold", "push", "local/nope", "-a", "x/y"])
    assert ex.value.code == 1


def test_run_manifold_rm_calls_backend(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    calls: list[tuple[str, str]] = []

    def fake_rm(ns: str, name: str) -> dict[str, Any]:
        calls.append((ns, name))
        return {"namespace": ns, "name": name, "source": "local",
                "removed": True, "rematerializes_on_restart": False}

    monkeypatch.setattr("saklas.io.manifolds.remove_manifold_folder", fake_rm)
    cli.main(["manifold", "rm", "local/mood"])
    assert calls == [("local", "mood")]
    out = capsys.readouterr().out
    assert "Removed local/mood" in out


def test_run_manifold_rm_bundled_refuses_without_yes(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "saklas.io.manifolds.remove_manifold_folder",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not call backend")),
    )
    with pytest.raises(SystemExit) as ex:
        cli.main(["manifold", "rm", "default/personas"])
    assert ex.value.code == 2


def test_run_manifold_rm_bundled_with_yes(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    def fake_rm(ns: str, name: str) -> dict[str, Any]:
        return {"namespace": ns, "name": name, "source": "bundled",
                "removed": True, "rematerializes_on_restart": True}

    monkeypatch.setattr("saklas.io.manifolds.remove_manifold_folder", fake_rm)
    cli.main(["manifold", "rm", "default/personas", "-y"])
    out = capsys.readouterr().out
    assert "Removed default/personas" in out
    assert "re-materializes" in out


def test_run_manifold_rm_missing_errors(monkeypatch: pytest.MonkeyPatch):
    def fake_rm(ns: str, name: str) -> dict[str, Any]:
        raise FileNotFoundError("manifold local/nope not found")

    monkeypatch.setattr("saklas.io.manifolds.remove_manifold_folder", fake_rm)
    with pytest.raises(SystemExit) as ex:
        cli.main(["manifold", "rm", "local/nope"])
    assert ex.value.code == 1


def test_run_manifold_clear_calls_backend(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    calls: list[dict[str, Any]] = []

    def fake_clear(ns: str, name: str, model_scope: Any = None, *, variant: str = "all") -> int:
        calls.append({"ns": ns, "name": name, "model_scope": model_scope, "variant": variant})
        return 3

    monkeypatch.setattr("saklas.io.manifolds.clear_manifold_tensors", fake_clear)
    # Bare ``pad`` resolves cross-namespace to the bundled ``default/pad``
    # (the only installed match) — the lifecycle verbs no longer hard-default
    # a bare name to ``local/``.
    cli.main(["manifold", "clear", "pad", "--variant", "raw"])
    assert calls == [
        {"ns": "default", "name": "pad", "model_scope": None, "variant": "raw"},
    ]
    out = capsys.readouterr().out
    assert "Deleted 3 files" in out


def test_run_manifold_clear_passes_model_scope(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    """``-m`` is now real — it threads through as ``model_scope`` (no warning)."""
    calls: list[dict[str, Any]] = []

    def fake_clear(ns: str, name: str, model_scope: Any = None, *, variant: str = "all") -> int:
        calls.append({"ns": ns, "name": name, "model_scope": model_scope, "variant": variant})
        return 1

    monkeypatch.setattr("saklas.io.manifolds.clear_manifold_tensors", fake_clear)
    # Bare name resolves cross-namespace to bundled ``default/pad``.
    cli.main(["manifold", "clear", "pad", "-m", "foo/bar"])
    assert calls == [
        {"ns": "default", "name": "pad", "model_scope": "foo/bar", "variant": "all"},
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
        cli.main(["manifold", "refresh", "alice/circumplex"])
        out = capsys.readouterr().out
        assert fragment in out


def test_run_manifold_refresh_passes_model_scope(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    """``-m`` does a real scoped refresh now — threads ``model_scope``, no warning."""
    calls: list[dict[str, Any]] = []

    def fake_refresh(ns: str, name: str, *, model_scope: Any = None) -> str:
        calls.append({"ns": ns, "name": name, "model_scope": model_scope})
        return "scoped"

    monkeypatch.setattr("saklas.io.manifolds.refresh_manifold", fake_refresh)
    # Bare name resolves cross-namespace to bundled ``default/pad``.
    cli.main(["manifold", "refresh", "pad", "-m", "foo/bar"])
    assert calls == [{"ns": "default", "name": "pad", "model_scope": "foo/bar"}]
    captured = capsys.readouterr()
    assert "foo/bar" in captured.out and "re-fits on next use" in captured.out
    assert "no effect" not in captured.err


def test_run_manifold_refresh_missing_errors(monkeypatch: pytest.MonkeyPatch):
    def fake_refresh(ns: str, name: str, *, model_scope: Any = None) -> str:
        raise FileNotFoundError("not installed")

    monkeypatch.setattr("saklas.io.manifolds.refresh_manifold", fake_refresh)
    with pytest.raises(SystemExit) as ex:
        cli.main(["manifold", "refresh", "local/nope"])
    assert ex.value.code == 1


def test_run_manifold_transfer_calls_backend(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    from saklas.io.paths import manifold_dir, safe_model_id
    folder = manifold_dir("local", "circumplex")
    folder.mkdir(parents=True)
    (folder / "manifold.json").write_text("{}")
    src_model = "google/gemma-3-4b-it"
    (folder / f"{safe_model_id(src_model)}.safetensors").write_bytes(b"x")

    # Alignment cache hit so no model load happens — mirror _run_transfer's
    # cached branch.  ``load_alignment_map`` returns ``(M, sidecar)``.
    import torch
    fake_M = {14: torch.eye(4), 15: torch.eye(4)}
    monkeypatch.setattr(
        "saklas.io.alignment.load_alignment_map",
        lambda src, tgt: (fake_M, {"quality_per_layer": {"14": 0.9, "15": 0.8}}),
    )

    calls: list[dict[str, Any]] = []

    def fake_transfer(folder_arg: Path, *, from_model: str, to_model: str,
                      alignment: Any, transfer_quality_estimate: Any = None,
                      whitener: Any = None, layer_means: Any = None,
                      force: bool = False) -> Path:
        calls.append({
            "folder": folder_arg, "from": from_model, "to": to_model,
            "layers": sorted(alignment.keys()), "quality": transfer_quality_estimate,
            "force": force,
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

    import torch
    monkeypatch.setattr(
        "saklas.io.alignment.load_alignment_map",
        lambda src, tgt: ({10: torch.eye(2)}, {"quality_per_layer": {"10": 0.5}}),
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
        cli.main(["manifold", "clear", "nope_not_installed"])
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
    cli.main(["manifold", "clear", "moodlite"])
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
        cli.main(["manifold", "clear", "dup"])
    assert ex.value.code == 2


def test_lifecycle_explicit_ns_pins_without_walk(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """An explicit ``ns/name`` pins verbatim — no existence pre-check, backend runs."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))  # nothing installed
    calls: list[tuple[str, str]] = []

    def fake_clear(ns: str, name: str, model_scope: Any = None, *, variant: str = "all") -> int:
        calls.append((ns, name))
        return 0

    monkeypatch.setattr("saklas.io.manifolds.clear_manifold_tensors", fake_clear)
    cli.main(["manifold", "clear", "alice/ghost"])
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
    cli.main(["manifold", "ls", "--namespace", "local", "-v"])
    out = capsys.readouterr().out
    assert "local/moodlite" in out
    assert "a tiny test mood manifold" in out


def test_run_manifold_ls_non_verbose_hides_description(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    _author_circumplex_lite(tmp_path)
    from saklas.io import selectors
    selectors.invalidate()
    cli.main(["manifold", "ls", "--namespace", "local"])
    out = capsys.readouterr().out
    assert "local/moodlite" in out
    assert "a tiny test mood manifold" not in out


def test_run_manifold_show_json_uses_summary_keys(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    _author_circumplex_lite(tmp_path)
    from saklas.io import selectors
    selectors.invalidate()
    cli.main(["manifold", "show", "local/moodlite", "-j"])
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
    cli.main(["manifold", "show", "local/moodlite", "-j"])
    out = capsys.readouterr().out
    assert _json.loads(out) == expected
