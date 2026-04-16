import pytest

from saklas import cli
from saklas.cli import runners as cli_runners


# ---------------------------------------------------------------------------
# parse_args — top-level subcommand dispatch
# ---------------------------------------------------------------------------

def test_parse_zero_args_prints_help_and_exits_zero(capsys):
    with pytest.raises(SystemExit) as ex:
        cli.parse_args([])
    assert ex.value.code == 0
    out = capsys.readouterr().out
    assert "tui" in out and "serve" in out and "pack" in out and "config" in out


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
# pack subtree
# ---------------------------------------------------------------------------

def test_parse_pack_no_verb_exits_zero(capsys):
    # pack with no verb prints help and exits 0.
    with pytest.raises(SystemExit) as ex:
        cli.main(["pack"])
    assert ex.value.code == 0


def test_parse_pack_install():
    args = cli.parse_args(["pack", "install", "a9lim/happy"])
    assert args.command == "pack"
    assert args.pack_cmd == "install"
    assert args.target == "a9lim/happy"
    assert args.statements_only is False


def test_parse_pack_install_flags():
    args = cli.parse_args(["pack", "install", "a9lim/happy", "-s", "-a", "local/cheer", "-f"])
    assert args.statements_only is True
    assert args.as_target == "local/cheer"
    assert args.force is True


def test_parse_pack_refresh():
    args = cli.parse_args(["pack", "refresh", "happy"])
    assert args.pack_cmd == "refresh"
    assert args.selector == "happy"
    assert args.model is None


def test_parse_pack_refresh_neutrals():
    args = cli.parse_args(["pack", "refresh", "neutrals"])
    assert args.pack_cmd == "refresh"
    assert args.selector == "neutrals"


def test_parse_pack_clear():
    args = cli.parse_args(["pack", "clear", "tag:emotion", "-m", "gemma-2-2b-it"])
    assert args.pack_cmd == "clear"
    assert args.selector == "tag:emotion"
    assert args.model == "gemma-2-2b-it"


def test_parse_pack_rm():
    args = cli.parse_args(["pack", "rm", "happy"])
    assert args.pack_cmd == "rm"
    assert args.selector == "happy"
    assert args.yes is False


def test_parse_pack_ls_empty():
    args = cli.parse_args(["pack", "ls"])
    assert args.pack_cmd == "ls"
    assert args.selector is None


def test_parse_pack_ls_with_selector():
    args = cli.parse_args(["pack", "ls", "tag:emotion", "-j"])
    assert args.pack_cmd == "ls"
    assert args.selector == "tag:emotion"
    assert args.json_output is True


def test_parse_pack_search():
    args = cli.parse_args(["pack", "search", "emotion"])
    assert args.pack_cmd == "search"
    assert args.query == "emotion"


def test_parse_pack_merge():
    args = cli.parse_args(["pack", "merge", "bard", "default/happy:0.3,a9lim/archaic:0.4"])
    assert args.pack_cmd == "merge"
    assert args.name == "bard"
    assert args.components == "default/happy:0.3,a9lim/archaic:0.4"


def test_parse_pack_clone_required_args():
    args = cli.parse_args(["pack", "clone", "/tmp/corpus.txt", "--name", "alice"])
    assert args.pack_cmd == "clone"
    assert args.corpus_path == "/tmp/corpus.txt"
    assert args.name == "alice"
    assert args.n_pairs == 90
    assert args.seed is None
    assert args.force is False
    assert args.model is None


def test_parse_pack_clone_missing_name_errors():
    with pytest.raises(SystemExit):
        cli.parse_args(["pack", "clone", "/tmp/corpus.txt"])


def test_parse_pack_extract_one_positional():
    args = cli.parse_args(["pack", "extract", "happy.sad"])
    assert args.pack_cmd == "extract"
    assert args.concept == ["happy.sad"]
    assert args.model is None
    assert args.force is False


def test_parse_pack_extract_two_positionals():
    args = cli.parse_args(["pack", "extract", "happy", "sad"])
    assert args.concept == ["happy", "sad"]


def test_parse_pack_extract_all_flags():
    args = cli.parse_args(["pack", "extract", "happy.sad", "-m", "foo/bar", "-f"])
    assert args.concept == ["happy.sad"]
    assert args.model == "foo/bar"
    assert args.force is True


def test_parse_pack_export_gguf():
    args = cli.parse_args(["pack", "export", "gguf", "happy.sad", "-m", "foo/bar"])
    assert args.pack_cmd == "export"
    assert args.format == "gguf"
    assert args.selector == "happy.sad"
    assert args.model == "foo/bar"


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


def test_config_show_runs(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    cli.main(["config", "show", "--no-default"])
    out = capsys.readouterr().out
    assert "saklas" in out  # header


def test_config_show_with_extra(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    p = tmp_path / "x.yaml"
    p.write_text("model: google/gemma-2-2b-it\ntemperature: 0.7\n")
    cli.main(["config", "show", "--no-default", "-c", str(p)])
    out = capsys.readouterr().out
    assert "google/gemma-2-2b-it" in out
    assert "temperature" in out


def test_config_validate_ok(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    p = tmp_path / "x.yaml"
    p.write_text("model: google/gemma-2-2b-it\n")
    cli.main(["config", "validate", str(p)])
    out = capsys.readouterr().out
    assert "ok" in out


def test_config_validate_missing_file(tmp_path):
    with pytest.raises(SystemExit) as ex:
        cli.main(["config", "validate", str(tmp_path / "nope.yaml")])
    assert ex.value.code == 2


def test_config_validate_local_vector_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    p = tmp_path / "x.yaml"
    p.write_text("vectors:\n  local/nope: 0.5\n")
    with pytest.raises(SystemExit) as ex:
        cli.main(["config", "validate", str(p)])
    assert ex.value.code == 2


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def test_run_refresh_bundled(monkeypatch, tmp_path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    from saklas.io import packs
    packs.materialize_bundled()
    target = tmp_path / "vectors" / "default" / "angry.calm" / "statements.json"
    if not target.exists():
        pytest.skip("angry.calm statements.json not yet regenerated")
    target.write_text("[]")
    cli.main(["pack", "refresh", "default"])
    assert target.read_text() != "[]"


def test_run_refresh_neutrals(monkeypatch, tmp_path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    (tmp_path / "neutral_statements.json").write_text("[]")
    cli.main(["pack", "refresh", "neutrals"])
    content = (tmp_path / "neutral_statements.json").read_text()
    assert content != "[]"


def test_run_install_folder(monkeypatch, tmp_path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path / "home"))
    from saklas.io import packs
    src = tmp_path / "src" / "happy"
    src.mkdir(parents=True)
    (src / "statements.json").write_text("[]")
    packs.PackMetadata(
        name="happy", description="x", version="1.0.0", license="MIT",
        tags=[], recommended_alpha=0.5, source="local",
        files={"statements.json": packs.hash_file(src / "statements.json")},
    ).write(src)
    cli.main(["pack", "install", str(src)])
    assert (tmp_path / "home" / "vectors" / "local" / "happy" / "pack.json").is_file()


def test_run_ls_empty(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))

    def boom(_sel):
        raise AssertionError("HF query must not run with pack ls")

    monkeypatch.setattr("saklas.io.hf.search_packs", boom)
    cli.main(["pack", "ls"])
    out = capsys.readouterr().out
    assert "NAME" in out


def test_run_search_calls_hf(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    calls = []

    def fake(sel):
        calls.append(sel)
        return [{"name": "happy", "namespace": "alice", "tags": [], "recommended_alpha": 0.5}]

    monkeypatch.setattr("saklas.io.hf.search_packs", fake)
    cli.main(["pack", "search", "happy"])
    assert calls and calls[0].value == "happy"
    out = capsys.readouterr().out
    assert "happy" in out


def test_run_rm_refuses_broad(monkeypatch, tmp_path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    from saklas.io import packs
    packs.materialize_bundled()
    with pytest.raises(SystemExit):
        cli.main(["pack", "rm", "all"])


def test_run_rm_specific(monkeypatch, tmp_path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    from saklas.io import packs
    packs.materialize_bundled()
    cli.main(["pack", "rm", "happy.sad"])
    assert not (tmp_path / "vectors" / "default" / "happy.sad").exists()


def test_run_extract_cache_hit_prints_already_extracted(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    from saklas.io import packs
    packs.materialize_bundled()

    from saklas.io.paths import vectors_dir, safe_model_id
    model_id = "fake/model"
    folder = vectors_dir() / "default" / "happy.sad"
    tensor = folder / f"{safe_model_id(model_id)}.safetensors"
    tensor.write_bytes(b"")

    class FakeSession:
        def __init__(self, **kw):
            self.model_id = model_id
            self.model_info = {"model_type": "fake", "num_layers": 1,
                               "hidden_dim": 8, "vram_used_gb": 0.0}
            self.probes = {}

        def _local_concept_folder(self, canonical):
            return vectors_dir() / "local" / canonical

        def extract(self, *a, **kw):
            raise AssertionError("extract() must not be called on cache hit")

    monkeypatch.setattr(cli_runners, "_make_session", lambda args: FakeSession())
    monkeypatch.setattr(cli_runners, "_print_model_info", lambda s: None)
    monkeypatch.setattr(cli_runners, "_print_startup", lambda args: None)

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["pack", "extract", "happy.sad", "-m", model_id])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "already extracted at" in out
    assert str(tensor) in out


def test_run_tui_registers_config_vectors(monkeypatch, tmp_path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    from saklas.io import packs
    packs.materialize_bundled()

    p = tmp_path / "setup.yaml"
    p.write_text("model: fake-model\nvectors:\n  default/happy.sad: 0.4\n")

    registered = {}

    class FakeSession:
        def __init__(self, **kw):
            self.kw = kw
            self.model_info = {"model_type": "fake", "num_layers": 1,
                               "hidden_dim": 8, "vram_used_gb": 0.0}
            self.probes = {}

        def extract(self, name, **kw):
            return name, "PROFILE"

        def steer(self, name, profile, alpha=None):
            registered[name] = (profile, alpha)

    class FakeApp:
        def __init__(self, session):
            self.session = session

        def run(self):
            pass

    monkeypatch.setattr(cli_runners, "_make_session", lambda args: FakeSession())
    monkeypatch.setattr(cli_runners, "_print_model_info", lambda s: None)

    import saklas.tui.app as _tui_app
    monkeypatch.setattr(_tui_app, "SaklasApp", FakeApp)

    cli.main(["tui", "-c", str(p)])
    assert "default/happy.sad" in registered


def test_config_bare_pole_resolves_canonical(monkeypatch, tmp_path):
    """Config YAML with bare pole 'wolf' should resolve to 'deer.wolf' with -1 sign."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    from saklas.io import packs
    packs.materialize_bundled()

    # Create a fake deer.wolf pack under local so resolve_pole sees it.
    d = tmp_path / "vectors" / "local" / "deer.wolf"
    d.mkdir(parents=True)
    (d / "statements.json").write_text("[]")
    packs.PackMetadata(
        name="deer.wolf", description="x", version="1.0.0", license="MIT",
        tags=[], recommended_alpha=0.5, source="local",
        files={"statements.json": packs.hash_file(d / "statements.json")},
    ).write(d)
    # Invalidate selector cache so the new pack is visible.
    from saklas.cli.selectors import invalidate
    invalidate()

    from saklas.cli.config_file import ConfigFile
    c = ConfigFile(vectors={"wolf": 0.5})
    resolved = c.resolve_poles()
    assert "deer.wolf" in resolved.vectors
    assert resolved.vectors["deer.wolf"] == -0.5
