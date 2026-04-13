import pytest

from saklas import cli


# ---------------------------------------------------------------------------
# parse_args — subcommand dispatch
# ---------------------------------------------------------------------------

def test_parse_bare_tui():
    args = cli.parse_args(["google/gemma-2-2b-it"])
    assert args.command == "tui"
    assert args.model == "google/gemma-2-2b-it"


def test_parse_refresh():
    args = cli.parse_args(["refresh", "happy"])
    assert args.command == "refresh"
    assert args.selector == "happy"
    assert args.model is None


def test_parse_refresh_with_model_scope():
    args = cli.parse_args(["refresh", "tag:emotion", "-m", "gemma-2-2b-it"])
    assert args.command == "refresh"
    assert args.selector == "tag:emotion"
    assert args.model == "gemma-2-2b-it"


def test_parse_refresh_neutrals():
    args = cli.parse_args(["refresh", "neutrals"])
    assert args.command == "refresh"
    assert args.selector == "neutrals"


def test_parse_clear():
    args = cli.parse_args(["clear", "tag:emotion", "-m", "gemma-2-2b-it"])
    assert args.command == "clear"
    assert args.selector == "tag:emotion"
    assert args.model == "gemma-2-2b-it"


def test_parse_install():
    args = cli.parse_args(["install", "a9lim/happy"])
    assert args.command == "install"
    assert args.target == "a9lim/happy"
    assert args.statements_only is False


def test_parse_install_statements_only():
    args = cli.parse_args(["install", "a9lim/happy", "-s"])
    assert args.command == "install"
    assert args.statements_only is True


def test_parse_install_as_force():
    args = cli.parse_args(["install", "a9lim/happy", "-a", "local/cheer", "-f"])
    assert args.as_target == "local/cheer"
    assert args.force is True


def test_parse_uninstall():
    args = cli.parse_args(["uninstall", "happy"])
    assert args.command == "uninstall"
    assert args.selector == "happy"
    assert args.yes is False


def test_parse_list_empty():
    args = cli.parse_args(["list"])
    assert args.command == "list"
    assert args.selector is None
    assert args.installed is False


def test_parse_list_installed_only():
    args = cli.parse_args(["list", "-i"])
    assert args.installed is True


def test_parse_list_with_selector_and_json():
    args = cli.parse_args(["list", "tag:emotion", "-j"])
    assert args.selector == "tag:emotion"
    assert args.json_output is True


def test_parse_merge():
    args = cli.parse_args(["merge", "bard", "default/happy:0.3,a9lim/archaic:0.4"])
    assert args.command == "merge"
    assert args.name == "bard"
    assert args.components == "default/happy:0.3,a9lim/archaic:0.4"


def test_parse_config_flag(monkeypatch, tmp_path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    p = tmp_path / "setup.yaml"
    p.write_text("model: google/gemma-2-2b-it\n")
    args = cli.parse_args(["-c", str(p)])
    assert args.command == "tui"
    assert args.config == [str(p)]
    assert args.model == "google/gemma-2-2b-it"


def test_parse_tui_requires_model():
    with pytest.raises(SystemExit):
        cli.parse_args([])


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def test_run_refresh_bundled(monkeypatch, tmp_path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    from saklas import packs
    packs.materialize_bundled()
    target = tmp_path / "vectors" / "default" / "angry.calm" / "statements.json"
    if not target.exists():
        import pytest
        pytest.skip("angry.calm statements.json not yet regenerated")
    target.write_text("[]")
    cli.main(["refresh", "default"])
    assert target.read_text() != "[]"


def test_run_refresh_neutrals(monkeypatch, tmp_path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    (tmp_path / "neutral_statements.json").write_text("[]")
    cli.main(["refresh", "neutrals"])
    content = (tmp_path / "neutral_statements.json").read_text()
    assert content != "[]"


def test_run_install_folder(monkeypatch, tmp_path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path / "home"))
    from saklas import packs
    src = tmp_path / "src" / "happy"
    src.mkdir(parents=True)
    (src / "statements.json").write_text("[]")
    packs.PackMetadata(
        name="happy", description="x", version="1.0.0", license="MIT",
        tags=[], recommended_alpha=0.5, source="local",
        files={"statements.json": packs.hash_file(src / "statements.json")},
    ).write(src)
    cli.main(["install", str(src)])
    assert (tmp_path / "home" / "vectors" / "local" / "happy" / "pack.json").is_file()


def test_run_list_empty(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    monkeypatch.setattr("saklas.hf.search_packs", lambda selector: [])
    cli.main(["list"])
    out = capsys.readouterr().out
    assert "NAME" in out


def test_run_list_installed_only_skips_hf(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))

    def boom(_sel):
        raise AssertionError("HF query must not run with --installed")

    monkeypatch.setattr("saklas.hf.search_packs", boom)
    cli.main(["list", "-i"])
    capsys.readouterr()  # drain


def test_run_uninstall_refuses_broad(monkeypatch, tmp_path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    from saklas import packs
    packs.materialize_bundled()
    with pytest.raises(SystemExit):
        cli.main(["uninstall", "all"])


def test_run_uninstall_with_yes(monkeypatch, tmp_path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    from saklas import packs
    packs.materialize_bundled()
    cli.main(["uninstall", "happy"])
    assert not (tmp_path / "vectors" / "default" / "happy").exists()


def test_run_tui_registers_config_vectors(monkeypatch, tmp_path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    from saklas import packs
    packs.materialize_bundled()

    p = tmp_path / "setup.yaml"
    p.write_text("model: fake-model\nvectors:\n  default/happy: 0.4\n")

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

    def fake_make_session(args):
        return FakeSession()

    class FakeApp:
        def __init__(self, session):
            self.session = session

        def run(self):
            pass

    monkeypatch.setattr(cli, "_make_session", fake_make_session)
    monkeypatch.setattr(cli, "_print_model_info", lambda s: None)

    import saklas.tui.app as _tui_app
    monkeypatch.setattr(_tui_app, "SaklasApp", FakeApp)

    cli.main(["-c", str(p)])
    assert "default/happy" in registered
