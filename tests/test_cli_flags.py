from __future__ import annotations

from pathlib import Path
from typing import Any
from types import SimpleNamespace

import pytest

from saklas import cli
from saklas.cli import runners as cli_runners


@pytest.mark.parametrize(
    "argv",
    [
        ["tui", "model", "--top-k-alts", "-1"],
        ["serve", "model", "--top-k-alts", "257"],
        ["tui", "model", "--max-tokens", "0"],
        ["serve", "model", "--port", "70000"],
        ["manifold", "fit", "m", "--max-dim", "0"],
        ["manifold", "fit", "m", "--k-nn", "-2"],
        ["sae", "load", "release", "-m", "model", "--layer", "-1"],
    ],
)
def test_numeric_flags_reject_out_of_range_values(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        cli.parse_args(argv)


# ---------------------------------------------------------------------------
# parse_args — top-level subcommand dispatch
# ---------------------------------------------------------------------------

def test_parse_zero_args_prints_help_and_exits_zero(capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit) as ex:
        cli.parse_args([])
    assert ex.value.code == 0
    out = capsys.readouterr().out
    assert "tui" in out and "serve" in out and "pack" in out and "manifold" in out and "config" in out


def test_parse_bare_unknown_model_id_errors():
    # No more argv[0] peek: bare `saklas some/model-id` is an invalid verb.
    with pytest.raises(SystemExit):
        cli.parse_args(["google/gemma-2-2b-it"])


def test_parse_tui_subcommand():
    args = cli.parse_args(["tui", "google/gemma-2-2b-it"])
    assert args.command == "tui"
    assert args.model == "google/gemma-2-2b-it"


def test_parse_tui_with_config_only():
    # tui model positional is optional — YAML may supply it via -c.
    args = cli.parse_args(["tui", "-c", "/nowhere.yaml"])
    assert args.command == "tui"
    assert args.model is None
    assert args.config == ["/nowhere.yaml"]


# ---------------------------------------------------------------------------
# manifold compute subtree (the flat-artifact / vector verbs folded in)
# ---------------------------------------------------------------------------

def test_parse_vector_merge():
    args = cli.parse_args([
        "manifold","bake", "bard",
        "0.3 default/happy + 0.4 a9lim/archaic",
    ])
    assert args.command == "manifold"
    assert args.manifold_cmd == "bake"
    assert args.name == "bard"
    assert args.expression == "0.3 default/happy + 0.4 a9lim/archaic"


def test_parse_vector_extract_one_positional():
    args = cli.parse_args(["manifold","extract", "happy.sad"])
    assert args.manifold_cmd == "extract"
    assert args.concept == ["happy.sad"]
    assert args.model is None
    assert args.force is False


def test_parse_vector_extract_two_positionals():
    args = cli.parse_args(["manifold","extract", "happy", "sad"])
    assert args.concept == ["happy", "sad"]


def test_parse_vector_extract_all_flags():
    args = cli.parse_args(["manifold","extract", "happy.sad", "-m", "foo/bar", "-f"])
    assert args.concept == ["happy.sad"]
    assert args.model == "foo/bar"
    assert args.force is True


def test_parse_pack_export_gguf():
    args = cli.parse_args(["pack", "export", "gguf", "happy.sad", "-m", "foo/bar"])
    assert args.command == "pack"
    assert args.pack_cmd == "export"
    assert args.format == "gguf"
    assert args.name == "happy.sad"
    assert args.model == "foo/bar"


def test_lens_layers_empty_string_errors(capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit) as ex:
        cli_runners._parse_layer_list("")
    assert ex.value.code == 2
    assert "at least one source layer" in capsys.readouterr().err


def test_lens_layers_named_modes_parse() -> None:
    assert cli_runners._parse_layer_list("workspace") == "workspace"
    assert cli_runners._parse_layer_list("band") == "band"
    assert cli_runners._parse_layer_list("all") == "all"


# ---------------------------------------------------------------------------
# config subtree
# ---------------------------------------------------------------------------

def test_parse_config_show():
    args = cli.parse_args(["config", "show"])
    assert args.command == "config"
    assert args.config_cmd == "show"


def test_parse_config_validate():
    args = cli.parse_args(["config", "validate", "/tmp/setup.yaml"])
    assert args.config_cmd == "validate"
    assert args.file == "/tmp/setup.yaml"


def test_config_show_runs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    cli.main(["config", "show", "--no-default"])
    out = capsys.readouterr().out
    assert "saklas" in out  # header


def test_config_show_with_extra(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    p = tmp_path / "x.yaml"
    p.write_text("model: google/gemma-2-2b-it\ntemperature: 0.7\n")
    cli.main(["config", "show", "--no-default", "-c", str(p)])
    out = capsys.readouterr().out
    assert "google/gemma-2-2b-it" in out
    assert "temperature" in out


def test_config_validate_ok(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    p = tmp_path / "x.yaml"
    p.write_text("model: google/gemma-2-2b-it\n")
    cli.main(["config", "validate", str(p)])
    out = capsys.readouterr().out
    assert "ok" in out


def test_config_validate_missing_file(tmp_path: Path):
    with pytest.raises(SystemExit) as ex:
        cli.main(["config", "validate", str(tmp_path / "nope.yaml")])
    assert ex.value.code == 2


def test_config_validate_local_vector_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    p = tmp_path / "x.yaml"
    p.write_text("vectors:\n  local/nope: 0.5\n")
    with pytest.raises(SystemExit) as ex:
        cli.main(["config", "validate", str(p)])
    assert ex.value.code == 2


# ---------------------------------------------------------------------------
# --compile / YAML compile: opt-in (defaults off — torch 2.12 inductor
# bugs on newer architectures and limited per-token benefit on
# interactive workloads make compile a deliberate choice, not a default)
# ---------------------------------------------------------------------------

def test_tui_compile_flag_parses():
    """``--compile`` is the opt-in for CUDA torch.compile.  Off by default."""
    args = cli.parse_args(["tui", "google/gemma-2-2b-it"])
    assert getattr(args, "compile", False) is False
    args = cli.parse_args(["tui", "google/gemma-2-2b-it", "--compile"])
    assert args.compile is True


def test_serve_compile_flag_parses():
    args = cli.parse_args(["serve", "google/gemma-2-2b-it"])
    assert getattr(args, "compile", False) is False
    args = cli.parse_args(["serve", "google/gemma-2-2b-it", "--compile"])
    assert args.compile is True


def test_yaml_compile_true_folds_onto_args(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """YAML ``compile: true`` should set ``args.compile=True`` when the
    CLI didn't already pass ``--compile``."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    p = tmp_path / "on.yaml"
    p.write_text("model: google/gemma-2-2b-it\ncompile: true\n")
    args = cli.parse_args(["tui", "-c", str(p)])
    # Pre-effective: argparse default is False.
    assert getattr(args, "compile", False) is False
    cli_runners._load_effective_config(args)
    # YAML opt-in folded on.
    assert args.compile is True


def test_yaml_compile_false_is_noop(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """``compile: false`` matches the default — accepting it in YAML
    keeps round-trip symmetry but doesn't flip ``args.compile``."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    p = tmp_path / "off.yaml"
    p.write_text("model: google/gemma-2-2b-it\ncompile: false\n")
    args = cli.parse_args(["tui", "-c", str(p)])
    cli_runners._load_effective_config(args)
    assert getattr(args, "compile", False) is False


def test_cli_compile_overrides_yaml_compile_false(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """CLI flag wins over YAML — passing ``--compile`` even with
    ``compile: false`` in YAML must keep the opt-in."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    p = tmp_path / "off.yaml"
    p.write_text("model: google/gemma-2-2b-it\ncompile: false\n")
    args = cli.parse_args(["tui", "-c", str(p), "--compile"])
    cli_runners._load_effective_config(args)
    assert args.compile is True


def test_yaml_compile_invalid_type_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Reject non-boolean ``compile:`` values rather than coercing —
    ``compile: "true"`` (a string) would otherwise pass through as
    truthy and silently turn compile on."""
    from saklas.cli.config_file import ConfigFile, ConfigFileError
    p = tmp_path / "bad.yaml"
    p.write_text("compile: \"false\"\n")
    with pytest.raises(ConfigFileError, match="compile must be a boolean"):
        ConfigFile.load(p)


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def test_run_extract_corrupt_tensor_does_not_short_circuit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))

    from saklas.io.paths import manifold_dir, tensor_filename
    model_id = "fake/model"
    # A steering vector is a 2-node pca manifold (4.0); extract lands it under
    # ``manifolds/<ns>/<canonical>/``.  A present per-model tensor is the
    # A bare corrupt tensor must not be treated as a validated cache hit.
    folder = manifold_dir("local", "happy.sad")
    folder.mkdir(parents=True, exist_ok=True)
    tensor = folder / tensor_filename(model_id)
    tensor.write_bytes(b"")

    called = {"extract": 0}

    class FakeSession:
        def __init__(self, **kw: Any) -> None:
            self.model_id = model_id
            self.model_info = {"model_type": "fake", "num_layers": 1,
                               "hidden_dim": 8, "vram_used_gb": 0.0}
            self.probes = {}

        def extract(self, *a: Any, **kw: Any) -> Any:
            called["extract"] += 1
            return "happy.sad", object()

    monkeypatch.setattr(
        cli_runners, "_make_session", lambda args, **_kwargs: FakeSession(),
    )
    monkeypatch.setattr(cli_runners, "_print_model_info", lambda s: None)
    monkeypatch.setattr(cli_runners, "_print_startup", lambda args: None)

    cli.main(["manifold", "extract", "happy.sad", "-m", model_id])
    assert called["extract"] == 1
    out = capsys.readouterr().out
    assert "extracted happy.sad" in out


@pytest.mark.parametrize(
    "argv",
    [
        ["manifold", "extract", "happy", "-m", "fake/model",
         "--sae-revision", "rev"],
        ["manifold", "fit", "missing", "-m", "fake/model",
         "--sae-revision", "rev"],
    ],
)
def test_sae_revision_flag_is_not_advertised(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.parse_args(argv)
    assert exc.value.code == 2


def test_fit_smoothing_override_is_delegated_to_fit_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from saklas.core.manifold import CustomDomain
    from saklas.io.manifold_authoring import create_discover_manifold_folder

    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    folder = create_discover_manifold_folder(
        "local", "roles", "", fit_mode="auto",
        node_corpora={"pirate": ["arrr"], "scholar": ["indeed"]},
        node_roles={"pirate": "pirate", "scholar": "scholar"},
        node_kinds={"pirate": "concrete", "scholar": "abstract"},
    )
    fake_manifold = SimpleNamespace(
        name="roles", layers={0: object()}, node_labels=["pirate", "scholar"],
        domain=CustomDomain(1), feature_space="raw",
        metadata={"fit_mode": "auto"},
    )
    fit_kwargs: dict[str, Any] = {}

    class _FakeSession:
        model_info = {}

        def fit(self, *_args: Any, **kwargs: Any) -> Any:
            fit_kwargs.update(kwargs)
            return fake_manifold

    monkeypatch.setattr(
        cli_runners, "_make_session", lambda *_args, **_kwargs: _FakeSession(),
    )
    monkeypatch.setattr(cli_runners, "_print_startup", lambda _args: None)
    monkeypatch.setattr(cli_runners, "_print_model_info", lambda _session: None)
    args = cli.parse_args([
        "manifold", "fit", str(folder), "-m", "fake/model",
        "--smoothing", "0.25",
    ])
    cli_runners._run_manifold_fit(args)
    assert fit_kwargs["fit_mode"] is None
    assert fit_kwargs["hyperparams"] == {"smoothing": 0.25}


def test_serve_stale_lens_gate_uses_weight_compatibility() -> None:
    class _Session:
        enabled = False

        def has_compatible_jlens(self) -> bool:
            return False

        def enable_live_lens(self, **_kwargs: Any) -> None:
            self.enabled = True

    session = _Session()
    assert not cli_runners._enable_serve_live_lens_if_compatible(session)
    assert not session.enabled


def test_serve_selects_cached_lens_when_active_pointer_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Session:
        model_id = "org/model"
        selected: str | None = None
        live_enabled = False

        def has_compatible_jlens(self) -> bool:
            return self.selected == "local:default"

        def select_jlens_source(self, source: str) -> None:
            self.selected = source

        def enable_live_lens(self) -> list[int]:
            self.live_enabled = True
            return [4, 5]

    monkeypatch.setattr(
        "saklas.io.lens_sources.list_lens_sources",
        lambda _model: [{
            "source": "local:default", "kind": "local", "active": False,
        }],
    )
    session = _Session()
    assert cli_runners._enable_serve_live_lens_if_compatible(session)
    assert session.selected == "local:default"
    assert session.live_enabled


def test_best_serve_sae_release_prefers_official_canonical_provider() -> None:
    rows = [
        {
            "release": "third-party-canonical", "source": "saelens",
            "repo_id": "someone/sae", "neuronpedia": True,
        },
        {
            "release": "gemma-scope-res-all", "source": "saelens",
            "repo_id": "google/gemma-scope", "neuronpedia": True,
        },
        {
            "release": "gemma-scope-res-canonical", "source": "saelens",
            "repo_id": "google/gemma-scope", "neuronpedia": True,
        },
    ]
    assert cli_runners._best_serve_sae_release(rows) == (
        "gemma-scope-res-canonical"
    )


def test_serve_attaches_best_sae_and_enables_live(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Session:
        model_id = "google/gemma-3-4b-it"
        sae_info = None
        loaded: str | None = None
        live_top_k: int | None = None

        def load_sae(self, release: str) -> dict[str, Any]:
            self.loaded = release
            return {"release": release, "layer": 22, "width": 16_384}

        def enable_live_sae(self, *, top_k: int) -> dict[str, int]:
            self.live_top_k = top_k
            return {"layer": 22, "top_k": top_k}

    monkeypatch.setattr(
        "saklas.io.sae.list_sae_sources",
        lambda _model: [],
    )
    monkeypatch.setattr(
        "saklas.core.sae.list_sae_releases",
        lambda _model: [{
            "release": "gemma-scope-2-4b-it-res",
            "source": "saelens",
            "repo_id": "google/gemma-scope-2-4b-it",
            "neuronpedia": True,
        }],
    )
    session = _Session()
    assert cli_runners._enable_serve_live_sae_if_available(session)
    assert session.loaded == "gemma-scope-2-4b-it-res"
    assert session.live_top_k == 12


def test_serve_prefers_cached_sae_over_registry_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Session:
        model_id = "google/gemma-3-4b-it"
        sae_info = None
        loaded: str | None = None

        def load_sae(self, release: str) -> dict[str, Any]:
            self.loaded = release
            return {"release": release, "layer": 22, "width": 16_384}

        def enable_live_sae(self, *, top_k: int) -> dict[str, int]:
            return {"layer": 22, "top_k": top_k}

    monkeypatch.setattr(
        "saklas.io.sae.list_sae_sources",
        lambda _model: [{
            "source": "local:cached", "kind": "local", "active": False,
        }],
    )
    monkeypatch.setattr(
        "saklas.core.sae.list_sae_releases",
        lambda _model: (_ for _ in ()).throw(
            AssertionError("registry should not be consulted")
        ),
    )
    session = _Session()
    assert cli_runners._enable_serve_live_sae_if_available(session)
    assert session.loaded == "local:cached"


def test_run_tui_registers_config_vectors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    # 4.0: a concept is a 2-node ``pca`` manifold — author ``default/happy.sad``
    # so the config reference resolves (and ``ensure_vectors_installed`` sees it
    # as already installed).
    from saklas.io.manifolds import create_discover_manifold_folder
    create_discover_manifold_folder(
        "default", "happy.sad", "x", fit_mode="pca",
        node_corpora={"happy": ["a statement."], "sad": ["b statement."]},
        hyperparams={"max_dim": 1},
    )
    from saklas.io import selectors as _sel
    _sel.invalidate()

    p = tmp_path / "setup.yaml"
    p.write_text('model: fake-model\nvectors: "0.4 default/happy.sad"\n')

    registered = {}
    extract_calls = []

    class FakeSession:
        def __init__(self, **kw: Any) -> None:
            self.kw = kw
            self.model_info = {"model_type": "fake", "num_layers": 1,
                               "hidden_dim": 8, "vram_used_gb": 0.0}
            self.probes = {}

        def extract(self, name: str, **kw: Any) -> Any:
            extract_calls.append((name, kw))
            return name, "PROFILE"

        def steer(self, name: str, profile: Any, alpha: Any = None) -> None:
            registered[name] = (profile, alpha)

    class FakeApp:
        def __init__(self, session: Any) -> None:
            self.session = session

        def run(self):
            pass

    monkeypatch.setattr(cli_runners, "_make_session", lambda args: FakeSession())
    monkeypatch.setattr(cli_runners, "_print_model_info", lambda s: None)

    import saklas.tui.app as _tui_app
    monkeypatch.setattr(_tui_app, "SaklasApp", FakeApp)

    cli.main(["tui", "-c", str(p)])
    assert "happy.sad" in registered
    assert extract_calls[-1] == ("happy.sad", {"namespace": "default"})


def test_parse_vector_compare_two_args():
    args = cli.parse_args(["manifold","compare", "angry.calm", "happy.sad", "-m", "foo/bar"])
    assert args.command == "manifold"
    assert args.manifold_cmd == "compare"
    assert args.concepts == ["angry.calm", "happy.sad"]
    assert args.model == "foo/bar"


def test_parse_vector_compare_one_arg():
    args = cli.parse_args(["manifold","compare", "angry.calm", "-m", "foo/bar"])
    assert args.concepts == ["angry.calm"]


def test_parse_vector_compare_three_plus_args():
    args = cli.parse_args(["manifold","compare", "angry.calm", "happy.sad", "formal.casual", "-m", "foo/bar"])
    assert args.concepts == ["angry.calm", "happy.sad", "formal.casual"]


def test_parse_vector_compare_selector_arg():
    args = cli.parse_args(["manifold","compare", "tag:affect", "-m", "foo/bar"])
    assert args.concepts == ["tag:affect"]


def test_parse_vector_compare_verbose_and_json():
    args = cli.parse_args(["manifold","compare", "a", "b", "-m", "x", "-v", "-j"])
    assert args.verbose is True
    assert args.json_output is True


def test_parse_vector_compare_missing_model_errors():
    with pytest.raises(SystemExit):
        cli.parse_args(["manifold","compare", "angry.calm"])


def test_manifold_pack_appear_in_help_subspace_vector_gone(capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit):
        cli.parse_args([])
    out = capsys.readouterr().out
    assert "manifold" in out
    assert "pack" in out
    # The verb harmonization retired the `subspace` verb and the `vector`
    # alias — neither parses as a top-level command anymore.
    for gone in (["subspace", "extract", "x"], ["vector", "extract", "x"]):
        with pytest.raises(SystemExit):
            cli.parse_args(gone)


# ---------------------------------------------------------------------------
# _run_compare — verbose modes (1-arg ranked, N×N matrix)
# ---------------------------------------------------------------------------

def _patch_fold_helpers(
    monkeypatch: pytest.MonkeyPatch,
    profiles_by_display: dict[str, Any],
) -> None:
    """Stub the two manifold-fold helpers ``_run_compare`` / ``_run_why`` call.

    4.0: ``manifold compare`` / ``why`` fold a fitted 2-node ``pca`` manifold
    to a ``Profile`` via ``runners._fold_manifold_to_profile_with_identity``
    (1-arg lookup) and ``runners._fold_all_fitted_manifolds`` (rank-all pool).
    Rather than author + fit a real manifold per concept, we stub both helpers
    to serve the mock profiles keyed by display name (``"<ns>/<name>"`` or the
    bare name, with an optional ``:variant`` suffix).

    ``profiles_by_display`` maps the *display* name the runner asks for
    (e.g. ``"angry.calm"``, ``"angry.calm:sae-my-release"``,
    ``"default/happy.sad"``) to a mock profile.  All concepts are authored
    under ``default/``, so the returned identity is ``("default", bare)``.
    """
    import saklas.cli.runners as runners

    def _fold(name: str, model_id: str, variant: str | None):
        bare = name.split("/", 1)[1] if "/" in name else name
        key = bare if variant is None else f"{bare}:{variant}"
        prof = profiles_by_display.get(key)
        if prof is None:
            return None
        return prof, "default", bare

    def _fold_all(
        model_id: str,
        *,
        exclude_identity: tuple[str, str] | None = None,
    ):
        out: dict[str, Any] = {}
        for key, prof in profiles_by_display.items():
            if ":" in key:
                continue  # only raw concepts join the rank-all pool
            if exclude_identity is not None and exclude_identity == ("default", key):
                continue
            out[f"default/{key}"] = prof
        return out

    monkeypatch.setattr(
        runners, "_fold_manifold_to_profile_with_identity", _fold,
    )
    monkeypatch.setattr(runners, "_fold_all_fitted_manifolds", _fold_all)


def _setup_compare_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Set SAKLAS_HOME and return the manifolds_dir path.

    ``manifold compare`` is Mahalanobis-only now: ``_run_compare`` loads the
    per-model whitener via ``LayerWhitener.from_cache`` up front and fails if
    it's missing.  These tests mock ``Profile.cosine_similarity`` (which
    ignores the whitener), so we patch ``from_cache`` to return a sentinel
    rather than seed a real neutral cache.
    """
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    from saklas.io import selectors
    selectors.invalidate()
    from saklas.core import mahalanobis as _maha
    monkeypatch.setattr(
        _maha.LayerWhitener, "from_cache",
        classmethod(lambda cls, model_id, **kw: object()),
    )
    return tmp_path / "manifolds"


def _mock_profile(agg_fn: Any, pl_fn: Any) -> Any:
    """Return a simple duck-typed mock (not a Profile subclass) with cosine_similarity."""
    class MockProfile:
        def cosine_similarity(self, other: Any, *, per_layer: bool = False, whitener: Any = None) -> Any:
            # ``whitener`` is accepted (forwarded by ``_run_compare`` after
            # the v2.1 ``--metric mahalanobis`` wiring) but ignored here —
            # the mocks return a fixed scalar that doesn't depend on the
            # metric.  Tests that need to exercise the Mahalanobis path
            # use ``test_mahalanobis.py`` against real Profile + tensors.
            return pl_fn(other) if per_layer else agg_fn(other)
    return MockProfile()


def test_run_compare_one_arg_verbose_text(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    """1-arg + -v text mode: prints per-layer breakdown for top 3."""
    _setup_compare_env(monkeypatch, tmp_path)
    model_id = "fake/model"

    happy_profile = _mock_profile(lambda o: None, lambda o: None)
    warm_profile = _mock_profile(lambda o: None, lambda o: None)

    def target_agg(other: Any) -> float:
        if other is happy_profile:
            return 0.3421
        if other is warm_profile:
            return 0.1893
        return 0.0

    def target_pl(other: Any) -> dict[int, float]:
        if other is happy_profile:
            return {14: 0.5122, 15: 0.4891}
        if other is warm_profile:
            return {14: 0.1010, 15: 0.0987}
        return {}

    target_profile = _mock_profile(target_agg, target_pl)

    _patch_fold_helpers(monkeypatch, {
        "angry.calm": target_profile,
        "happy.sad": happy_profile,
        "warm.clinical": warm_profile,
    })

    # Compare is Mahalanobis-only; the whitener load is patched in
    # ``_setup_compare_env`` and ``Profile.cosine_similarity`` is mocked.
    cli.main(["manifold","compare", "angry.calm", "-m", model_id, "-v"])
    out = capsys.readouterr().out
    assert "angry.calm vs all installed" in out
    assert "happy.sad" in out
    assert "per-layer (top 3)" in out
    assert "0.5122" in out


def test_run_compare_one_arg_verbose_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    """1-arg + -v -j JSON mode: output includes per_layer_top3 key."""
    import json as _json
    _setup_compare_env(monkeypatch, tmp_path)
    model_id = "fake/model"

    happy_profile = _mock_profile(lambda o: None, lambda o: None)
    target_profile = _mock_profile(
        lambda o: 0.3421,
        lambda o: {14: 0.5122, 15: 0.4891},
    )

    _patch_fold_helpers(monkeypatch, {
        "angry.calm": target_profile,
        "happy.sad": happy_profile,
    })

    cli.main(["manifold","compare", "angry.calm", "-m", model_id, "-v", "-j"])
    out = capsys.readouterr().out
    data = _json.loads(out)
    assert data["target"] == "angry.calm"
    assert "per_layer_top3" in data
    assert "happy.sad" in data["per_layer_top3"]
    pl = data["per_layer_top3"]["happy.sad"]
    assert "14" in pl
    assert abs(pl["14"] - 0.5122) < 1e-5


def test_run_compare_matrix_verbose_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    """3-arg N×N + -v -j JSON mode: output includes per_layer dict keyed by 'a|b'."""
    import json as _json
    _setup_compare_env(monkeypatch, tmp_path)
    model_id = "fake/model"

    concepts = ["angry.calm", "happy.sad", "warm.clinical"]
    per_layer_vals = {14: 0.3456, 15: 0.2345}
    _patch_fold_helpers(monkeypatch, {
        c: _mock_profile(lambda o: 0.25, lambda o: per_layer_vals)
        for c in concepts
    })

    cli.main(["manifold","compare"] + concepts + ["-m", model_id, "-v", "-j"])
    out = capsys.readouterr().out
    data = _json.loads(out)
    assert "per_layer" in data
    assert "angry.calm|happy.sad" in data["per_layer"]
    assert "angry.calm|warm.clinical" in data["per_layer"]
    assert "happy.sad|warm.clinical" in data["per_layer"]
    assert "angry.calm|angry.calm" not in data["per_layer"]
    pl = data["per_layer"]["angry.calm|happy.sad"]
    assert "14" in pl
    assert abs(pl["14"] - 0.3456) < 1e-5


def test_run_compare_matrix_verbose_text_unchanged(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    """3-arg N×N + -v text mode: no per-layer section (text matrix is dense enough)."""
    _setup_compare_env(monkeypatch, tmp_path)
    model_id = "fake/model"

    concepts = ["angry.calm", "happy.sad", "warm.clinical"]
    _patch_fold_helpers(monkeypatch, {
        c: _mock_profile(lambda o: 0.25, lambda o: {14: 0.1})
        for c in concepts
    })

    cli.main(["manifold","compare"] + concepts + ["-m", model_id, "-v"])
    out = capsys.readouterr().out
    assert "per-layer" not in out
    assert "per_layer" not in out
    assert "angry.calm" in out


# ---------------------------------------------------------------------------
# _run_why — layer introspection
# ---------------------------------------------------------------------------

def _setup_why_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Set SAKLAS_HOME and return the manifolds_dir path."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    from saklas.io import selectors
    selectors.invalidate()
    return tmp_path / "manifolds"


def _mock_why_profile(layer_mags: dict[int, float], diagnostics: dict[str, Any] | None = None) -> Any:
    """Return a duck-typed mock profile for _run_why.

    Carries the four surfaces ``_run_why`` reads: ``items()`` (per-layer
    magnitudes), ``__len__`` (total layers), ``diagnostics``, and
    ``has_diagnostics``.  Diagnostics default to ``None`` to mirror the
    pre-1.6 sidecar shape.
    """
    import torch

    class MockProfile:
        def items(self):
            return {layer: torch.full((1,), mag) for layer, mag in layer_mags.items()}.items()

        def __len__(self):
            return len(layer_mags)

        @property
        def diagnostics(self):
            return diagnostics

        @property
        def has_diagnostics(self):
            return diagnostics is not None and bool(diagnostics)

    return MockProfile()


def test_parse_vector_why_basic():
    args = cli.parse_args(["manifold","why", "angry.calm", "-m", "foo/bar"])
    assert args.command == "manifold"
    assert args.manifold_cmd == "why"
    assert args.concept == "angry.calm"
    assert args.model == "foo/bar"
    assert args.json_output is False


def test_parse_vector_why_json():
    args = cli.parse_args(["manifold","why", "angry.calm", "-m", "foo/bar", "-j"])
    assert args.json_output is True


def test_parse_vector_why_removed_flags_rejected():
    # ``--all`` and ``-n`` were removed with the histogram overhaul.
    for argv in (
        ["manifold","why", "angry.calm", "-m", "foo/bar", "--all"],
        ["manifold","why", "angry.calm", "-m", "foo/bar", "-n", "10"],
    ):
        with pytest.raises(SystemExit):
            cli.parse_args(argv)


def test_parse_vector_why_missing_model_errors():
    with pytest.raises(SystemExit):
        cli.parse_args(["manifold","why", "angry.calm"])


def test_run_why_text_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    """Text output renders a per-bucket histogram covering every layer."""
    _setup_why_env(monkeypatch, tmp_path)
    model_id = "fake/model"

    layer_mags = {14: 0.847, 15: 0.812, 13: 0.793, 12: 0.641, 11: 0.589, 10: 0.400}
    profile = _mock_why_profile(layer_mags)
    _patch_fold_helpers(monkeypatch, {"angry.calm": profile})

    cli.main(["manifold","why", "angry.calm", "-m", model_id])
    out = capsys.readouterr().out
    assert "angry.calm" in out
    assert "6 layers" in out
    assert "LAYERS" in out
    # With 6 layers ≤ 24 buckets, each layer is its own bucket — every layer
    # label must appear.
    for layer in layer_mags:
        assert f"L{layer}" in out
    # Bar glyph present.
    assert "█" in out


def test_run_why_text_buckets_large_profile(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    """With >24 layers, buckets collapse into layer ranges."""
    _setup_why_env(monkeypatch, tmp_path)
    model_id = "fake/model"

    layer_mags = {i: float(i + 1) * 0.01 for i in range(62)}
    profile = _mock_why_profile(layer_mags)
    _patch_fold_helpers(monkeypatch, {"angry.calm": profile})

    cli.main(["manifold","why", "angry.calm", "-m", model_id])
    out = capsys.readouterr().out
    assert "62 layers" in out
    # Range label form used when buckets span more than one layer.
    assert "-" in out
    # Exactly HIST_BUCKETS histogram rows — count bar glyphs on left edge.
    from saklas.core.histogram import HIST_BUCKETS
    bar_lines = [ln for ln in out.splitlines() if "█" in ln or "░" in ln]
    assert len(bar_lines) == HIST_BUCKETS


def test_run_why_json_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    """JSON output is full-fidelity, in layer order."""
    import json as _json
    _setup_why_env(monkeypatch, tmp_path)
    model_id = "fake/model"

    layer_mags = {14: 0.847, 15: 0.812, 13: 0.793}
    profile = _mock_why_profile(layer_mags)
    _patch_fold_helpers(monkeypatch, {"angry.calm": profile})

    cli.main(["manifold","why", "angry.calm", "-m", model_id, "-j"])
    out = capsys.readouterr().out
    data = _json.loads(out)
    assert data["concept"] == "angry.calm"
    assert data["model"] == model_id
    assert data["total_layers"] == 3
    assert [e["layer"] for e in data["layers"]] == [13, 14, 15]
    assert abs(data["layers"][1]["magnitude"] - 0.847) < 1e-4


def test_run_why_concept_not_found(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    """Missing concept exits with code 1."""
    _setup_why_env(monkeypatch, tmp_path)
    _patch_fold_helpers(monkeypatch, {})  # nothing folds → miss
    with pytest.raises(SystemExit) as exc:
        cli.main(["manifold","why", "nonexistent_concept_xyz", "-m", "foo/bar"])
    assert exc.value.code == 1


# ---------------------------------------------------------------------------
# SAE variant resolution in manifold compare / manifold why
# ---------------------------------------------------------------------------


def test_split_variant_suffix_parses_sae_variants():
    from saklas.cli.runners import _split_variant_suffix
    assert _split_variant_suffix("honest") == ("honest", None)
    assert _split_variant_suffix("honest:raw") == ("honest", "raw")
    assert _split_variant_suffix("honest:sae") == ("honest", "sae")
    assert _split_variant_suffix("honest:sae-my-release") == (
        "honest", "sae-my-release",
    )
    assert _split_variant_suffix("honest:role-pirate") == (
        "honest", "role-pirate",
    )
    assert _split_variant_suffix("honest:from-google__gemma-4-31b-it") == (
        "honest", "from-google__gemma-4-31b-it",
    )
    assert _split_variant_suffix("deer.wolf:sae") == ("deer.wolf", "sae")
    # Prefix selectors pass through untouched — their value carries the colon
    # but _split_variant_suffix only peels a trailing variant token.
    assert _split_variant_suffix("tag:emotion") == ("tag:emotion", None)
    assert _split_variant_suffix("model:google/gemma-3-4b-it") == (
        "model:google/gemma-3-4b-it", None,
    )


# NOTE: ``test_resolve_variant_tensor_accepts_full_variant_family`` was deleted
# in 4.0 — ``runners._resolve_variant_tensor`` was removed (compare/why fold
# fitted manifolds now, no ``vectors/`` ``Profile.load`` tensor pick).
# ``test_run_why_sae_ambiguous_errors`` was also deleted: the "multiple SAE
# variants" error came from the removed ``enumerate_variants`` tensor scan; the
# fold helper now simply returns ``None`` for a bare ``:sae`` (→ "no fitted
# manifold").


def test_run_why_accepts_sae_suffix(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    """``manifold why foo:sae-<rel>`` folds the SAE-variant manifold tensor."""
    _setup_why_env(monkeypatch, tmp_path)
    model_id = "fake/model"

    raw_profile = _mock_why_profile({0: 0.1, 1: 0.1})
    sae_profile = _mock_why_profile({14: 0.9, 15: 0.8})
    _patch_fold_helpers(monkeypatch, {
        "angry.calm": raw_profile,
        "angry.calm:sae-my-release": sae_profile,
    })

    cli.main(["manifold","why", "angry.calm:sae-my-release", "-m", model_id])
    out = capsys.readouterr().out
    # Suffix propagates into the display name.
    assert "angry.calm:sae-my-release" in out
    # The SAE profile magnitudes appear, not the raw ones.
    assert "L14" in out
    assert "0.900" in out
    assert "L0" not in out


def test_run_compare_accepts_sae_suffix(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    """Each concept in ``manifold compare`` parses its own :variant suffix."""
    _setup_compare_env(monkeypatch, tmp_path)
    model_id = "fake/model"

    a_raw_profile = _mock_profile(lambda o: 0.1, lambda o: {0: 0.1})
    a_sae_profile = _mock_profile(lambda o: 0.9, lambda o: {14: 0.9})
    b_profile = _mock_profile(lambda o: 0.5, lambda o: {0: 0.5})
    _patch_fold_helpers(monkeypatch, {
        "angry.calm": a_raw_profile,
        "angry.calm:sae-my-release": a_sae_profile,
        "happy.sad": b_profile,
    })

    cli.main([
        "manifold","compare",
        "angry.calm:sae-my-release", "happy.sad",
        "-m", model_id,
    ])
    out = capsys.readouterr().out
    # Variant suffix carries into display keys.
    assert "angry.calm:sae-my-release" in out
    assert "happy.sad" in out


def test_config_bare_pole_resolves_canonical(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """4.0: a config-YAML bare pole resolves through the MANIFOLD tier.

    Pre-4.0 a ``deer.wolf`` ``vectors/`` pack made bare ``wolf`` resolve to
    the signed vector ``deer.wolf @ -0.5``.  Now a bipolar concept is a 2-node
    ``pca`` manifold (nodes ``deer``/``wolf``), so ``0.5 wolf`` resolves to a
    label-form ``ManifoldTerm`` at the ``wolf`` node (``default/deer.wolf%wolf``).
    """
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    from saklas.io.manifolds import create_discover_manifold_folder
    create_discover_manifold_folder(
        "default", "deer.wolf", "x", fit_mode="pca",
        node_corpora={"deer": ["a statement."], "wolf": ["b statement."]},
        hyperparams={"max_dim": 1},
    )
    from saklas.io.selectors import invalidate
    invalidate()

    from saklas.core.steering_expr import ManifoldTerm, parse_expr
    steering = parse_expr("0.5 wolf")
    assert "default/deer.wolf%wolf" in steering.alphas
    term = steering.alphas["default/deer.wolf%wolf"]
    assert isinstance(term, ManifoldTerm)
    assert term.manifold == "default/deer.wolf"
    assert term.position == "wolf"
    assert term.along == 0.5


# ---------------------------------------------------------------------------
# vector extract --sae
# ---------------------------------------------------------------------------

def test_vector_extract_parses_sae_flag():
    """--sae RELEASE is captured on the Namespace as `sae`."""
    args = cli.parse_args([
        "manifold","extract", "honest.deceptive",
        "-m", "google/gemma-2-2b-it",
        "--sae", "gemma-scope-2b-pt-res-canonical",
    ])
    assert args.sae == "gemma-scope-2b-pt-res-canonical"
    assert not hasattr(args, "sae_revision")


def test_vector_extract_sae_revision_is_rejected():
    with pytest.raises(SystemExit):
        cli.parse_args([
            "manifold", "extract", "honest.deceptive",
            "-m", "google/gemma-2-2b-it", "--sae", "release-x",
            "--sae-revision", "v1.0",
        ])


def test_vector_extract_no_sae_defaults_to_none():
    args = cli.parse_args([
        "manifold","extract", "honest.deceptive", "-m", "model",
    ])
    assert args.sae is None
    assert not hasattr(args, "sae_revision")


def test_vector_extract_sae_requires_value():
    """--sae must be followed by a release name; it's not a boolean switch."""
    with pytest.raises(SystemExit):
        cli.parse_args([
            "manifold","extract", "honest.deceptive", "-m", "m", "--sae",
        ])
