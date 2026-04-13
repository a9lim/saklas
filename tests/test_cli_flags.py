import pytest

from saklas import cli


def test_parse_refresh_flag():
    args = cli.parse_args(["-r", "happy"])
    assert args.command == "cache"
    assert args.refresh == ["happy"]


def test_parse_delete_tensors_flag():
    # Compound selectors use repeated -x flags (action="append" single-token
    # per invocation, to keep trailing model positionals unambiguous).
    args = cli.parse_args(["-x", "tag:emotion", "-x", "model:gemma-2-2b-it"])
    assert args.command == "cache"
    assert args.delete == ["tag:emotion", "model:gemma-2-2b-it"]


def test_parse_install_flag():
    args = cli.parse_args(["-i", "a9lim/happy"])
    assert args.command == "cache"
    assert args.install == ["a9lim/happy"]


def test_parse_list_flag_empty():
    args = cli.parse_args(["-l"])
    assert args.command == "list"
    assert args.list == ""


def test_parse_list_flag_with_selector():
    args = cli.parse_args(["-l", "tag:emotion"])
    assert args.command == "list"
    assert args.list == "tag:emotion"


def test_parse_merge_flag():
    args = cli.parse_args(["-m", "bard", "default/happy:0.3,a9lim/archaic:0.4"])
    assert args.command == "cache"
    assert args.merge_name == "bard"
    assert args.merge_components == "default/happy:0.3,a9lim/archaic:0.4"


def test_parse_config_flag(monkeypatch, tmp_path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    p = tmp_path / "setup.yaml"
    p.write_text("model: google/gemma-2-2b-it\n")
    args = cli.parse_args(["-C", str(p)])
    assert args.config == [str(p)]
    assert args.model == "google/gemma-2-2b-it"


def test_parse_flag_plus_model_falls_through():
    args = cli.parse_args(["-r", "default", "google/gemma-2-2b-it"])
    assert args.command == "tui"
    assert args.refresh == ["default"]
    assert args.model == "google/gemma-2-2b-it"


def test_parse_list_rejects_trailing_model():
    with pytest.raises(SystemExit):
        cli.parse_args(["-l", "happy", "google/gemma-2-2b-it"])


def test_parse_requires_action_or_model():
    with pytest.raises(SystemExit):
        cli.parse_args([])


def test_run_cache_refresh_bundled(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    from saklas import packs
    packs.materialize_bundled()
    (tmp_path / "vectors" / "default" / "happy" / "statements.json").write_text("[]")
    cli.main(["-r", "default"])
    content = (tmp_path / "vectors" / "default" / "happy" / "statements.json").read_text()
    assert content != "[]"


def test_run_cache_install_folder(monkeypatch, tmp_path):
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
    cli.main(["-i", str(src)])
    assert (tmp_path / "home" / "vectors" / "local" / "happy" / "pack.json").is_file()


def test_run_list_empty(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    # Force list to skip HF (no network in CI) by patching search_packs to return nothing
    monkeypatch.setattr("saklas.hf.search_packs", lambda selector: [])
    cli.main(["-l"])
    out = capsys.readouterr().out
    assert "NAME" in out


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
            return "PROFILE"

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

    cli.main(["-C", str(p)])
    assert "default/happy" in registered
