"""The six-verb root CLI.

``manifold`` is the steering-vector / manifold *compute* surface
(extract/generate/fit/bake/merge/transfer/compare/why); ``pack`` is the
manifold *lifecycle* surface (ls/show/install/search/push/rm/clear/refresh/
export).  The former ``subspace`` verb and the deprecated ``vector`` alias are
gone — the flat-artifact verbs folded into ``manifold``.  These tests exercise
the parser shape + dispatch wiring, not the backends.
"""
from __future__ import annotations

from pathlib import Path
from typing import Generator

import pytest

from saklas import cli
from saklas.cli.runners import _COMMAND_RUNNERS


@pytest.fixture(autouse=True)
def _isolated_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> Generator[None, None, None]:
    from saklas.io import selectors as _sel
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    _sel.invalidate()
    yield
    _sel.invalidate()


# ---------------------------------------------------------------------------
# Root verb set
# ---------------------------------------------------------------------------

def test_seven_top_level_verbs() -> None:
    assert set(_COMMAND_RUNNERS) == {
        "tui", "serve", "manifold", "pack", "config", "experiment", "template",
    }


def test_template_verb_parses() -> None:
    args = cli.parse_args(["template", "score", "weekday", "-m", "m/x"])
    assert args.command == "template"
    assert args.template_cmd == "score"
    assert args.name == "weekday"
    assert args.model == "m/x"


def test_manifold_from_template_parses() -> None:
    args = cli.parse_args(["manifold", "from-template", "weekday", "--name", "wd"])
    assert args.command == "manifold"
    assert args.manifold_cmd == "from-template"
    assert args.template == "weekday"
    assert args.name == "wd"


def test_bare_template_prints_help_exit_0(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["template"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "saklas template <verb>" in out


def test_root_help_lists_manifold_and_pack(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        cli.parse_args(["--help"])
    out = capsys.readouterr().out
    assert "manifold" in out
    assert "pack" in out


# ---------------------------------------------------------------------------
# manifold — the compute verbs (extract/bake/compare/why folded in)
# ---------------------------------------------------------------------------

def test_manifold_extract_parses() -> None:
    args = cli.parse_args(["manifold", "extract", "happy", "sad"])
    assert args.command == "manifold"
    assert args.manifold_cmd == "extract"
    assert args.concept == ["happy", "sad"]


def test_manifold_bake_parses() -> None:
    args = cli.parse_args(["manifold", "bake", "bard", "0.3 a + 0.4 b"])
    assert args.command == "manifold"
    assert args.manifold_cmd == "bake"
    assert args.name == "bard"
    assert args.expression == "0.3 a + 0.4 b"


def test_manifold_compare_parses() -> None:
    args = cli.parse_args(["manifold", "compare", "happy.sad", "-m", "m/x"])
    assert args.command == "manifold"
    assert args.manifold_cmd == "compare"
    assert args.model == "m/x"


def test_manifold_why_parses() -> None:
    args = cli.parse_args(["manifold", "why", "happy.sad", "-m", "m/x"])
    assert args.manifold_cmd == "why"


def test_manifold_fit_parses() -> None:
    args = cli.parse_args(["manifold", "fit", "/tmp/folder", "-m", "m/x"])
    assert args.command == "manifold"
    assert args.manifold_cmd == "fit"
    assert args.target == "/tmp/folder"


def test_manifold_fit_discover_hyperparams_parse() -> None:
    # discover folded into fit — the hyperparam flags ride the one verb.
    args = cli.parse_args([
        "manifold", "fit", "mood", "--method", "spectral", "-m", "m/x",
    ])
    assert args.manifold_cmd == "fit"
    assert args.method == "spectral"
    assert args.target == "mood"


def test_manifold_transfer_parses() -> None:
    args = cli.parse_args([
        "manifold", "transfer", "circumplex",
        "--from", "a/b", "--to", "c/d",
    ])
    assert args.manifold_cmd == "transfer"
    assert args.name == "circumplex"


def test_manifold_has_no_lifecycle_subverb() -> None:
    # ``ls`` is a pack verb now — not nested under manifold.
    with pytest.raises(SystemExit):
        cli.parse_args(["manifold", "ls"])


def test_manifold_has_no_discover_verb() -> None:
    # discover folded into fit.
    with pytest.raises(SystemExit):
        cli.parse_args(["manifold", "discover", "mood"])


def test_bare_manifold_prints_help_exit_0(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["manifold"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "saklas manifold <verb>" in out


# ---------------------------------------------------------------------------
# pack — the lifecycle verbs (moved off manifold)
# ---------------------------------------------------------------------------

def test_pack_ls_parses() -> None:
    args = cli.parse_args(["pack", "ls"])
    assert args.command == "pack"
    assert args.pack_cmd == "ls"


def test_pack_export_gguf_parses() -> None:
    args = cli.parse_args(["pack", "export", "gguf", "happy.sad", "-m", "m/x"])
    assert args.command == "pack"
    assert args.pack_cmd == "export"
    assert args.format == "gguf"
    assert args.name == "happy.sad"


def test_pack_has_no_compute_subverb() -> None:
    # ``extract`` is a manifold verb now — not nested under pack.
    with pytest.raises(SystemExit):
        cli.parse_args(["pack", "extract", "happy", "sad"])


def test_bare_pack_prints_help_exit_0(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["pack"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "saklas pack <verb>" in out


# ---------------------------------------------------------------------------
# subspace / vector — removed (subspace folded into manifold)
# ---------------------------------------------------------------------------

def test_subspace_verb_removed() -> None:
    with pytest.raises(SystemExit):
        cli.parse_args(["subspace", "extract", "happy", "sad"])


def test_vector_alias_removed() -> None:
    with pytest.raises(SystemExit):
        cli.parse_args(["vector", "extract", "happy", "sad"])
