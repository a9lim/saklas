"""The nine-verb root CLI.

``manifold`` is the steering-vector / manifold *compute* surface
(extract/generate/from-template/fit/bake/merge/transfer/compare/why); ``pack`` is
the manifold *lifecycle* surface (ls/show/install/search/push/rm/clear/refresh/
export); ``template`` owns the standalone templated-completion artifact; ``lens``
owns the per-model Jacobian-lens artifact
(fit/fetch/ls/show/use/top/decompose/rm); ``sae`` exposes the parallel
train/fetch/ls/show/use/rm lifecycle.  The
former ``subspace`` verb and the deprecated ``vector`` alias are gone — the
flat-artifact verbs folded into ``manifold``.  These tests exercise the parser
shape + dispatch wiring, not the backends.
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

def test_nine_top_level_verbs() -> None:
    assert set(_COMMAND_RUNNERS) == {
        "tui", "serve", "manifold", "pack", "config", "experiment", "template",
        "lens", "sae",
    }


def test_sae_fetch_parses_with_harmonized_model_first_shape() -> None:
    args = cli.parse_args([
        "sae", "fetch", "m/x", "saelens:gemma-scope", "--layer", "14", "-j",
    ])
    assert args.command == "sae"
    assert args.sae_cmd == "fetch"
    assert args.model == "m/x"
    assert args.source == "saelens:gemma-scope"
    assert args.layer == 14
    assert args.json_output is True


def test_lens_and_sae_lifecycle_shapes_are_parallel() -> None:
    lens = cli.parse_args(["lens", "use", "m/x", "neuronpedia"])
    sae = cli.parse_args(["sae", "use", "m/x", "local:mine"])
    assert (lens.model, lens.source) == ("m/x", "neuronpedia")
    assert (sae.model, sae.source) == ("m/x", "local:mine")

    lens_ls = cli.parse_args(["lens", "ls", "m/x", "-j"])
    sae_ls = cli.parse_args(["sae", "ls", "m/x", "-j"])
    assert lens_ls.model == sae_ls.model == "m/x"
    assert lens_ls.json_output and sae_ls.json_output


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


def test_manifold_fit_layers_parse() -> None:
    args = cli.parse_args([
        "manifold", "fit", "mood", "-m", "m/x", "--layers", "4,8,12",
    ])
    assert args.layers == "4,8,12"


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


# ---------------------------------------------------------------------------
# lens
# ---------------------------------------------------------------------------

def test_lens_fit_parses() -> None:
    args = cli.parse_args([
        "lens", "fit", "m/x", "--prompts", "50", "--dim-batch", "32",
        "--prompt-batch", "4", "-f",
    ])
    assert args.command == "lens"
    assert args.lens_cmd == "fit"
    assert args.model == "m/x"
    assert args.prompts == 50
    assert args.dim_batch == 32
    assert args.prompt_batch == 4
    assert args.force is True


def test_lens_top_parses() -> None:
    args = cli.parse_args([
        "lens", "top", "m/x", "some prompt", "-k", "5",
        "--layers", "12,24", "--position", "-1",
    ])
    assert args.command == "lens"
    assert args.lens_cmd == "top"
    assert args.model == "m/x"
    assert args.prompt == "some prompt"
    assert args.top_k == 5
    assert args.layers == "12,24"
    assert args.position == [-1]


def test_lens_decompose_parses() -> None:
    args = cli.parse_args([
        "lens", "decompose", "confident.uncertain", "-m", "m/x", "-j",
    ])
    assert args.command == "lens"
    assert args.lens_cmd == "decompose"
    assert args.selector == "confident.uncertain"
    assert args.model == "m/x"
    assert args.json_output is True


def test_lens_show_and_rm_parse() -> None:
    show = cli.parse_args(["lens", "show", "m/x", "-j"])
    assert (show.lens_cmd, show.model, show.json_output) == ("show", "m/x", True)
    rm = cli.parse_args(["lens", "rm", "m/x", "-y"])
    assert (rm.lens_cmd, rm.model, rm.yes) == ("rm", "m/x", True)


def test_bare_lens_prints_help_exit_0(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["lens"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "saklas lens <verb>" in out
