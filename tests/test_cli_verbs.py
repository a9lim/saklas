"""The eight-verb root CLI.

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

import json
from pathlib import Path
from typing import Any, Generator

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
        "serve", "manifold", "pack", "config", "experiment", "template",
        "lens", "sae",
    }


def test_sae_fetch_parses_with_harmonized_model_first_shape() -> None:
    args = cli.parse_args([
        "sae", "fetch", "m/x", "saelens:gemma-scope", "--layer", "14",
        "--revision", "abc123", "-j",
    ])
    assert args.command == "sae"
    assert args.sae_cmd == "fetch"
    assert args.model == "m/x"
    assert args.source == "saelens:gemma-scope"
    assert args.layer == 14
    assert args.revision == "abc123"
    assert args.json_output is True


@pytest.mark.parametrize("flag", ["-d", "-q"])
def test_sae_fetch_carries_no_model_loading_flags(flag: str) -> None:
    """``sae fetch`` is pure IO, like ``lens fetch``: no device / quantize.

    Those flags existed only because the runner opened a full session to
    write a pointer file; it validates against the published config now.
    """
    with pytest.raises(SystemExit) as exc:
        cli.parse_args(["sae", "fetch", "m/x", "saelens:rel", flag, "cpu"])
    assert exc.value.code == 2


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


def test_manifold_extract_kind_defaults_to_abstract() -> None:
    args = cli.parse_args(["manifold", "extract", "happy", "sad"])
    assert args.kind == "abstract"
    assert args.custom_system is None


def test_manifold_extract_kind_and_system_parse() -> None:
    """``--kind`` / ``--system`` reach the 2-node path like ``generate``."""
    args = cli.parse_args([
        "manifold", "extract", "january", "july",
        "--kind", "custom", "--system", "You are the month of {c}.",
    ])
    assert args.kind == "custom"
    assert args.custom_system == "You are the month of {c}."


def test_manifold_extract_rejects_unknown_kind() -> None:
    with pytest.raises(SystemExit):
        cli.parse_args(["manifold", "extract", "happy", "sad", "--kind", "nope"])


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


# ---------------------------------------------------------------------------
# Bare-verb-group menus — one shared helper, so every group reads identically
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("argv", [
    ["manifold"], ["pack"], ["template"], ["lens"], ["sae"],
    ["experiment"], ["experiment", "transcript"],
])
def test_bare_verb_group_menu_carries_help_hint(
    argv: list[str], capsys: pytest.CaptureFixture[str],
) -> None:
    group = " ".join(argv)
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert out.startswith(f"usage: saklas {group} <verb> [...]\n")
    assert f"Run `saklas {group} <verb> -h` for verb-specific options." in out


# ---------------------------------------------------------------------------
# Short-flag hygiene
# ---------------------------------------------------------------------------

def test_sae_show_takes_json_but_rm_does_not() -> None:
    """``sae rm`` never emitted JSON; the flag is gone rather than inert
    (``lens rm``, its sibling, has no ``-j`` either)."""
    show = cli.parse_args(["sae", "show", "m/x", "-j"])
    assert show.json_output is True
    with pytest.raises(SystemExit) as exc:
        cli.parse_args(["sae", "rm", "m/x", "local:mine", "-j"])
    assert exc.value.code == 2


def _patch_sae_fetch_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    seen: dict[str, Any] | None = None,
) -> None:
    """Stub the provider backend + config read behind ``sae fetch``."""
    import saklas.cli.runners.sae as sae_runner

    class _Backend:
        release = "rel"
        revision = "deadbeef"
        fingerprint = "f" * 64
        layers = frozenset({12, 14, 16})
        sae_ids_by_layer = {"14": "layer_14"}
        repo_id = "provider/saes"
        neuronpedia_ids_by_layer = {"14": "np-14"}

        def feature_count(self, _idx: int) -> int:
            return 16384

        def feature_direction(self, _idx: int, _fid: int) -> Any:
            import torch
            return torch.zeros(2304)

    def _load(release: str, **kwargs: Any) -> Any:
        if seen is not None:
            seen.update({"release": release, **kwargs})
        return _Backend()

    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    monkeypatch.setattr(
        sae_runner, "_model_shape_from_config", lambda _m: (26, 2304),
    )
    monkeypatch.setattr("saklas.core.sae.load_sae_backend", _load)


def test_sae_fetch_announces_before_the_provider_download(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Mirrors ``lens fetch``: a saklas-authored status line before the
    network call, suppressed under ``-j`` so JSON stays parseable."""
    _patch_sae_fetch_backend(monkeypatch, tmp_path)

    cli.main(["sae", "fetch", "m/x", "saelens:rel"])
    out = capsys.readouterr().out
    assert "Fetching saelens:rel into Hugging Face cache..." in out

    cli.main(["sae", "fetch", "m/x", "saelens:rel", "-j"])
    out = capsys.readouterr().out
    assert "Fetching saelens:rel" not in out
    assert json.loads(out)["source"] == "saelens:rel"


def test_sae_fetch_writes_the_binding_without_loading_the_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The pointer file lands from provider metadata alone.

    The runner used to open a full ``SaklasSession.from_pretrained`` — tens
    of GB of weights — to write a few hundred bytes.  ``from_pretrained`` is
    stubbed to fail here, so reaching it is the test failure.
    """
    import saklas.core.session as session_mod
    from saklas.io.sae import load_active_sae, load_sae_metadata

    seen: dict[str, Any] = {}
    _patch_sae_fetch_backend(monkeypatch, tmp_path, seen=seen)

    def _explode(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("sae fetch must not load the base model")

    monkeypatch.setattr(
        session_mod.SaklasSession, "from_pretrained", staticmethod(_explode),
    )

    cli.main([
        "sae", "fetch", "m/x", "saelens:rel", "--layer", "14",
        "--revision", "abc123",
    ])
    out = capsys.readouterr().out
    assert "L14, 16384 features" in out

    # The provider revision is forwarded to the loader, which is where the
    # "honored only when the installed SAELens exposes revision=" policy lives.
    assert seen["revision"] == "abc123"
    assert seen["model_id"] == "m/x"

    binding = load_sae_metadata("m/x", "rel")
    assert binding is not None
    assert binding["layer"] == 14
    assert binding["width"] == 16384
    assert binding["repo_id"] == "provider/saes"
    assert binding["neuronpedia_id"] == "np-14"
    # ...and it is the active source, as the printed line claims.
    active = load_active_sae("m/x")
    assert active is not None and active[0] == "rel"


def test_sae_fetch_rejects_a_layer_the_release_does_not_cover(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Coverage is checked against the config's layer count, not a session."""
    _patch_sae_fetch_backend(monkeypatch, tmp_path)
    with pytest.raises(SystemExit) as exc:
        cli.main(["sae", "fetch", "m/x", "saelens:rel", "--layer", "3"])
    assert exc.value.code == 2
    assert "does not cover layer 3" in capsys.readouterr().err


def test_experiment_fan_steer_flag_and_hidden_alias(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``-S EXPR`` reads identically on every verb that takes one; the
    historical ``--base-steering`` spelling still parses, hidden from help."""
    args = cli.parse_args([
        "experiment", "fan", "m/x", "p", "-g", "a=0,1", "-S", "0.5 warm",
    ])
    assert args.base_steering == "0.5 warm"
    alias = cli.parse_args([
        "experiment", "fan", "m/x", "p", "-g", "a=0,1",
        "--base-steering", "0.5 warm",
    ])
    assert alias.base_steering == "0.5 warm"
    with pytest.raises(SystemExit):
        cli.parse_args(["experiment", "fan", "-h"])
    out = capsys.readouterr().out
    assert "--steer" in out
    assert "--base-steering" not in out
