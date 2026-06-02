"""4.0 step 6c — the eight-verb root CLI.

``subspace`` (flat / vector ops) and ``manifold`` (curved + flat manifold
authoring) are promoted to top-level verbs; ``vector`` becomes a deprecated
alias that parses the old tree identically and dispatches to the new runners
with a one-time deprecation notice.  These tests exercise the parser shape +
dispatch wiring, not the backends.
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

def test_eight_top_level_verbs() -> None:
    assert set(_COMMAND_RUNNERS) == {
        "tui", "serve", "pack", "subspace", "manifold", "vector",
        "config", "experiment",
    }


def test_root_help_lists_subspace_and_manifold(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        cli.parse_args(["--help"])
    out = capsys.readouterr().out
    assert "subspace" in out
    assert "manifold" in out


# ---------------------------------------------------------------------------
# subspace — the flat-artifact verbs parse on the top-level verb
# ---------------------------------------------------------------------------

def test_subspace_extract_parses() -> None:
    args = cli.parse_args(["subspace", "extract", "happy", "sad"])
    assert args.command == "subspace"
    assert args.subspace_cmd == "extract"
    assert args.concept == ["happy", "sad"]


def test_subspace_compare_parses() -> None:
    args = cli.parse_args(["subspace", "compare", "happy.sad", "-m", "m/x"])
    assert args.command == "subspace"
    assert args.subspace_cmd == "compare"
    assert args.model == "m/x"


def test_subspace_why_parses() -> None:
    args = cli.parse_args(["subspace", "why", "happy.sad", "-m", "m/x"])
    assert args.subspace_cmd == "why"


def test_subspace_has_no_manifold_subverb() -> None:
    # ``manifold`` is its own top-level verb now — not nested under subspace.
    with pytest.raises(SystemExit):
        cli.parse_args(["subspace", "manifold", "ls"])


def test_bare_subspace_prints_help_exit_0(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["subspace"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "saklas subspace <verb>" in out


# ---------------------------------------------------------------------------
# manifold — promoted to top-level, identical subparser shape
# ---------------------------------------------------------------------------

def test_manifold_ls_parses() -> None:
    args = cli.parse_args(["manifold", "ls"])
    assert args.command == "manifold"
    assert args.manifold_cmd == "ls"


def test_manifold_fit_parses() -> None:
    args = cli.parse_args(["manifold", "fit", "/tmp/folder", "-m", "m/x"])
    assert args.command == "manifold"
    assert args.manifold_cmd == "fit"


def test_manifold_discover_flags_parse() -> None:
    args = cli.parse_args([
        "manifold", "discover", "mood", "--method", "spectral", "-m", "m/x",
    ])
    assert args.manifold_cmd == "discover"
    assert args.method == "spectral"


def test_bare_manifold_prints_help_exit_0(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["manifold"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "saklas manifold <verb>" in out


# ---------------------------------------------------------------------------
# vector — deprecated alias keeps parsing the old tree
# ---------------------------------------------------------------------------

def test_vector_alias_still_parses() -> None:
    args = cli.parse_args(["vector", "extract", "happy", "sad"])
    assert args.command == "vector"
    assert args.vector_cmd == "extract"


def test_vector_manifold_alias_still_parses() -> None:
    args = cli.parse_args(["vector", "manifold", "ls"])
    assert args.command == "vector"
    assert args.vector_cmd == "manifold"
    assert args.manifold_cmd == "ls"


def test_vector_dispatch_emits_deprecation(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # ``manifold ls`` is pure-IO; under the empty isolated home it lists
    # nothing and exits cleanly, so we can assert on the warning.
    cli.main(["vector", "manifold", "ls"])
    err = capsys.readouterr().err
    assert "deprecated" in err
    assert "saklas manifold" in err


def test_vector_subspace_verb_dispatch_emits_deprecation(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # ``compare`` with no model resolves nothing under the empty home; the
    # deprecation note must mention the subspace target.
    with pytest.raises(SystemExit):
        cli.main(["vector", "compare", "nonexistent", "-m", "m/x"])
    err = capsys.readouterr().err
    assert "deprecated" in err
    assert "saklas subspace compare" in err
