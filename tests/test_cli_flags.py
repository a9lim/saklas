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
    assert "tui" in out and "serve" in out and "pack" in out and "vector" in out and "config" in out


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


def test_parse_pack_merge_is_gone():
    """merge moved to vector subtree."""
    with pytest.raises(SystemExit):
        cli.parse_args(["pack", "merge", "bard", "default/happy:0.3"])


# ---------------------------------------------------------------------------
# vector subtree
# ---------------------------------------------------------------------------

def test_parse_vector_no_verb_exits_zero(capsys):
    with pytest.raises(SystemExit) as ex:
        cli.main(["vector"])
    assert ex.value.code == 0


def test_parse_vector_merge():
    args = cli.parse_args([
        "vector", "merge", "bard",
        "0.3 default/happy + 0.4 a9lim/archaic",
    ])
    assert args.command == "vector"
    assert args.vector_cmd == "merge"
    assert args.name == "bard"
    assert args.expression == "0.3 default/happy + 0.4 a9lim/archaic"


def test_parse_vector_clone_required_args():
    args = cli.parse_args(["vector", "clone", "/tmp/corpus.txt", "--name", "alice"])
    assert args.vector_cmd == "clone"
    assert args.corpus_path == "/tmp/corpus.txt"
    assert args.name == "alice"
    assert args.n_pairs == 90
    assert args.seed is None
    assert args.force is False
    assert args.model is None


def test_parse_vector_clone_missing_name_errors():
    with pytest.raises(SystemExit):
        cli.parse_args(["vector", "clone", "/tmp/corpus.txt"])


def test_parse_vector_extract_one_positional():
    args = cli.parse_args(["vector", "extract", "happy.sad"])
    assert args.vector_cmd == "extract"
    assert args.concept == ["happy.sad"]
    assert args.model is None
    assert args.force is False


def test_parse_vector_extract_two_positionals():
    args = cli.parse_args(["vector", "extract", "happy", "sad"])
    assert args.concept == ["happy", "sad"]


def test_parse_vector_extract_all_flags():
    args = cli.parse_args(["vector", "extract", "happy.sad", "-m", "foo/bar", "-f"])
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
        cli.main(["vector", "extract", "happy.sad", "-m", model_id])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "already extracted at" in out
    assert str(tensor) in out


def test_run_tui_registers_config_vectors(monkeypatch, tmp_path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    from saklas.io import packs
    packs.materialize_bundled()

    p = tmp_path / "setup.yaml"
    p.write_text('model: fake-model\nvectors: "0.4 default/happy.sad"\n')

    registered = {}
    extract_calls = []

    class FakeSession:
        def __init__(self, **kw):
            self.kw = kw
            self.model_info = {"model_type": "fake", "num_layers": 1,
                               "hidden_dim": 8, "vram_used_gb": 0.0}
            self.probes = {}

        def extract(self, name, **kw):
            extract_calls.append((name, kw))
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
    assert "happy.sad" in registered
    assert extract_calls[-1] == ("happy.sad", {"namespace": "default"})


def test_parse_vector_compare_two_args():
    args = cli.parse_args(["vector", "compare", "angry.calm", "happy.sad", "-m", "foo/bar"])
    assert args.command == "vector"
    assert args.vector_cmd == "compare"
    assert args.concepts == ["angry.calm", "happy.sad"]
    assert args.model == "foo/bar"


def test_parse_vector_compare_one_arg():
    args = cli.parse_args(["vector", "compare", "angry.calm", "-m", "foo/bar"])
    assert args.concepts == ["angry.calm"]


def test_parse_vector_compare_three_plus_args():
    args = cli.parse_args(["vector", "compare", "angry.calm", "happy.sad", "formal.casual", "-m", "foo/bar"])
    assert args.concepts == ["angry.calm", "happy.sad", "formal.casual"]


def test_parse_vector_compare_selector_arg():
    args = cli.parse_args(["vector", "compare", "tag:affect", "-m", "foo/bar"])
    assert args.concepts == ["tag:affect"]


def test_parse_vector_compare_verbose_and_json():
    args = cli.parse_args(["vector", "compare", "a", "b", "-m", "x", "-v", "-j"])
    assert args.verbose is True
    assert args.json_output is True


def test_parse_vector_compare_missing_model_errors():
    with pytest.raises(SystemExit):
        cli.parse_args(["vector", "compare", "angry.calm"])


def test_vector_appears_in_help(capsys):
    with pytest.raises(SystemExit):
        cli.parse_args([])
    out = capsys.readouterr().out
    assert "vector" in out


# ---------------------------------------------------------------------------
# _run_compare — verbose modes (1-arg ranked, N×N matrix)
# ---------------------------------------------------------------------------

def _setup_compare_env(monkeypatch, tmp_path):
    """Set SAKLAS_HOME, materialize bundled, and return the vectors_dir path."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    from saklas.io import packs
    packs.materialize_bundled()
    from saklas.cli import selectors
    selectors.invalidate()
    return tmp_path / "vectors"


def _mock_profile(agg_fn, pl_fn):
    """Return a simple duck-typed mock (not a Profile subclass) with cosine_similarity."""
    class MockProfile:
        def cosine_similarity(self, other, *, per_layer=False):
            return pl_fn(other) if per_layer else agg_fn(other)
    return MockProfile()


def test_run_compare_one_arg_verbose_text(monkeypatch, tmp_path, capsys):
    """1-arg + -v text mode: prints per-layer breakdown for top 3."""
    vdir = _setup_compare_env(monkeypatch, tmp_path)
    model_id = "fake/model"
    from saklas.io.paths import safe_model_id
    sid = safe_model_id(model_id)

    target_dir = vdir / "default" / "angry.calm"
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / f"{sid}.safetensors").write_bytes(b"x")

    happy_dir = vdir / "default" / "happy.sad"
    happy_dir.mkdir(parents=True, exist_ok=True)
    (happy_dir / f"{sid}.safetensors").write_bytes(b"x")

    warm_dir = vdir / "default" / "warm.clinical"
    warm_dir.mkdir(parents=True, exist_ok=True)
    (warm_dir / f"{sid}.safetensors").write_bytes(b"x")

    happy_profile = _mock_profile(lambda o: None, lambda o: None)
    warm_profile = _mock_profile(lambda o: None, lambda o: None)

    def target_agg(other):
        if other is happy_profile:
            return 0.3421
        if other is warm_profile:
            return 0.1893
        return 0.0

    def target_pl(other):
        if other is happy_profile:
            return {14: 0.5122, 15: 0.4891}
        if other is warm_profile:
            return {14: 0.1010, 15: 0.0987}
        return {}

    target_profile = _mock_profile(target_agg, target_pl)

    profiles_by_path = {
        str(target_dir / f"{sid}.safetensors"): target_profile,
        str(happy_dir / f"{sid}.safetensors"): happy_profile,
        str(warm_dir / f"{sid}.safetensors"): warm_profile,
    }

    from saklas.core.profile import Profile

    monkeypatch.setattr(Profile, "load", staticmethod(lambda p: profiles_by_path[str(p)]))
    from saklas.cli.selectors import invalidate
    invalidate()

    cli.main(["vector", "compare", "angry.calm", "-m", model_id, "-v"])
    out = capsys.readouterr().out
    assert "angry.calm vs all installed" in out
    assert "happy.sad" in out
    assert "per-layer (top 3)" in out
    assert "0.5122" in out


def test_run_compare_one_arg_verbose_json(monkeypatch, tmp_path, capsys):
    """1-arg + -v -j JSON mode: output includes per_layer_top3 key."""
    import json as _json
    vdir = _setup_compare_env(monkeypatch, tmp_path)
    model_id = "fake/model"
    from saklas.io.paths import safe_model_id
    sid = safe_model_id(model_id)

    target_dir = vdir / "default" / "angry.calm"
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / f"{sid}.safetensors").write_bytes(b"x")

    happy_dir = vdir / "default" / "happy.sad"
    happy_dir.mkdir(parents=True, exist_ok=True)
    (happy_dir / f"{sid}.safetensors").write_bytes(b"x")

    happy_profile = _mock_profile(lambda o: None, lambda o: None)
    target_profile = _mock_profile(
        lambda o: 0.3421,
        lambda o: {14: 0.5122, 15: 0.4891},
    )

    profiles_by_path = {
        str(target_dir / f"{sid}.safetensors"): target_profile,
        str(happy_dir / f"{sid}.safetensors"): happy_profile,
    }

    from saklas.core.profile import Profile

    monkeypatch.setattr(Profile, "load", staticmethod(lambda p: profiles_by_path[str(p)]))
    from saklas.cli.selectors import invalidate
    invalidate()

    cli.main(["vector", "compare", "angry.calm", "-m", model_id, "-v", "-j"])
    out = capsys.readouterr().out
    data = _json.loads(out)
    assert data["target"] == "angry.calm"
    assert "per_layer_top3" in data
    assert "happy.sad" in data["per_layer_top3"]
    pl = data["per_layer_top3"]["happy.sad"]
    assert "14" in pl
    assert abs(pl["14"] - 0.5122) < 1e-5


def test_run_compare_matrix_verbose_json(monkeypatch, tmp_path, capsys):
    """3-arg N×N + -v -j JSON mode: output includes per_layer dict keyed by 'a|b'."""
    import json as _json
    vdir = _setup_compare_env(monkeypatch, tmp_path)
    model_id = "fake/model"
    from saklas.io.paths import safe_model_id
    sid = safe_model_id(model_id)

    concepts = ["angry.calm", "happy.sad", "warm.clinical"]
    dirs = {}
    for c in concepts:
        d = vdir / "default" / c
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{sid}.safetensors").write_bytes(b"x")
        dirs[c] = d

    from saklas.core.profile import Profile

    per_layer_vals = {14: 0.3456, 15: 0.2345}
    profiles_by_path: dict = {}
    for c in concepts:
        fp = _mock_profile(lambda o: 0.25, lambda o: per_layer_vals)
        profiles_by_path[str(dirs[c] / f"{sid}.safetensors")] = fp

    monkeypatch.setattr(Profile, "load", staticmethod(lambda p: profiles_by_path[str(p)]))
    from saklas.cli.selectors import invalidate
    invalidate()

    cli.main(["vector", "compare"] + concepts + ["-m", model_id, "-v", "-j"])
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


def test_run_compare_matrix_verbose_text_unchanged(monkeypatch, tmp_path, capsys):
    """3-arg N×N + -v text mode: no per-layer section (text matrix is dense enough)."""
    vdir = _setup_compare_env(monkeypatch, tmp_path)
    model_id = "fake/model"
    from saklas.io.paths import safe_model_id
    sid = safe_model_id(model_id)

    concepts = ["angry.calm", "happy.sad", "warm.clinical"]
    dirs = {}
    for c in concepts:
        d = vdir / "default" / c
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{sid}.safetensors").write_bytes(b"x")
        dirs[c] = d

    from saklas.core.profile import Profile
    profiles_by_path: dict = {}
    for c in concepts:
        fp = _mock_profile(lambda o: 0.25, lambda o: {14: 0.1})
        profiles_by_path[str(dirs[c] / f"{sid}.safetensors")] = fp

    monkeypatch.setattr(Profile, "load", staticmethod(lambda p: profiles_by_path[str(p)]))
    from saklas.cli.selectors import invalidate
    invalidate()

    cli.main(["vector", "compare"] + concepts + ["-m", model_id, "-v"])
    out = capsys.readouterr().out
    assert "per-layer" not in out
    assert "per_layer" not in out
    assert "angry.calm" in out


# ---------------------------------------------------------------------------
# _run_why — layer introspection
# ---------------------------------------------------------------------------

def _setup_why_env(monkeypatch, tmp_path):
    """Set SAKLAS_HOME, materialize bundled, return vectors_dir."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    from saklas.io import packs
    packs.materialize_bundled()
    from saklas.cli import selectors
    selectors.invalidate()
    return tmp_path / "vectors"


def _mock_why_profile(layer_mags: dict):
    """Return a duck-typed mock profile for _run_why (only needs .items() and len)."""
    import torch

    class MockProfile:
        def items(self):
            return {layer: torch.full((1,), mag) for layer, mag in layer_mags.items()}.items()

        def __len__(self):
            return len(layer_mags)

    return MockProfile()


def test_parse_vector_why_basic():
    args = cli.parse_args(["vector", "why", "angry.calm", "-m", "foo/bar"])
    assert args.command == "vector"
    assert args.vector_cmd == "why"
    assert args.concept == "angry.calm"
    assert args.model == "foo/bar"
    assert args.top_n == 5
    assert args.show_all is False
    assert args.json_output is False


def test_parse_vector_why_all_and_json():
    args = cli.parse_args(["vector", "why", "angry.calm", "-m", "foo/bar", "--all", "-j"])
    assert args.show_all is True
    assert args.json_output is True


def test_parse_vector_why_top_n():
    args = cli.parse_args(["vector", "why", "angry.calm", "-m", "foo/bar", "-n", "10"])
    assert args.top_n == 10


def test_parse_vector_why_missing_model_errors():
    with pytest.raises(SystemExit):
        cli.parse_args(["vector", "why", "angry.calm"])


def test_run_why_text_output(monkeypatch, tmp_path, capsys):
    """Basic text output: shows top N layers sorted by magnitude."""
    vdir = _setup_why_env(monkeypatch, tmp_path)
    model_id = "fake/model"
    from saklas.io.paths import safe_model_id
    sid = safe_model_id(model_id)

    target_dir = vdir / "default" / "angry.calm"
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / f"{sid}.safetensors").write_bytes(b"x")

    layer_mags = {14: 0.847, 15: 0.812, 13: 0.793, 12: 0.641, 11: 0.589, 10: 0.400}
    profile = _mock_why_profile(layer_mags)

    from saklas.core.profile import Profile
    monkeypatch.setattr(Profile, "load", staticmethod(lambda p: profile))
    from saklas.cli.selectors import invalidate
    invalidate()

    cli.main(["vector", "why", "angry.calm", "-m", model_id])
    out = capsys.readouterr().out
    assert "angry.calm" in out
    assert "6 layers" in out
    assert "top layers" in out
    assert "L14" in out
    assert "0.847" in out
    # Only top 5, layer 10 should not appear
    assert "L10" not in out


def test_run_why_text_all(monkeypatch, tmp_path, capsys):
    """--all shows every layer without 'top layers' prefix."""
    vdir = _setup_why_env(monkeypatch, tmp_path)
    model_id = "fake/model"
    from saklas.io.paths import safe_model_id
    sid = safe_model_id(model_id)

    target_dir = vdir / "default" / "angry.calm"
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / f"{sid}.safetensors").write_bytes(b"x")

    layer_mags = {14: 0.847, 10: 0.400}
    profile = _mock_why_profile(layer_mags)

    from saklas.core.profile import Profile
    monkeypatch.setattr(Profile, "load", staticmethod(lambda p: profile))
    from saklas.cli.selectors import invalidate
    invalidate()

    cli.main(["vector", "why", "angry.calm", "-m", model_id, "--all"])
    out = capsys.readouterr().out
    assert "top layers" not in out
    assert "layers (by ||baked||)" in out
    assert "L14" in out
    assert "L10" in out


def test_run_why_json_output(monkeypatch, tmp_path, capsys):
    """JSON output has expected keys and values."""
    import json as _json
    vdir = _setup_why_env(monkeypatch, tmp_path)
    model_id = "fake/model"
    from saklas.io.paths import safe_model_id
    sid = safe_model_id(model_id)

    target_dir = vdir / "default" / "angry.calm"
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / f"{sid}.safetensors").write_bytes(b"x")

    layer_mags = {14: 0.847, 15: 0.812, 13: 0.793}
    profile = _mock_why_profile(layer_mags)

    from saklas.core.profile import Profile
    monkeypatch.setattr(Profile, "load", staticmethod(lambda p: profile))
    from saklas.cli.selectors import invalidate
    invalidate()

    cli.main(["vector", "why", "angry.calm", "-m", model_id, "-j"])
    out = capsys.readouterr().out
    data = _json.loads(out)
    assert data["concept"] == "angry.calm"
    assert data["model"] == model_id
    assert data["total_layers"] == 3
    assert isinstance(data["layers"], list)
    assert len(data["layers"]) == 3  # default top_n=5 but only 3 layers
    # First entry should be layer 14 (highest magnitude)
    assert data["layers"][0]["layer"] == 14
    assert abs(data["layers"][0]["magnitude"] - 0.847) < 1e-4


def test_run_why_concept_not_found(monkeypatch, tmp_path, capsys):
    """Missing concept exits with code 1."""
    _setup_why_env(monkeypatch, tmp_path)
    with pytest.raises(SystemExit) as exc:
        cli.main(["vector", "why", "nonexistent_concept_xyz", "-m", "foo/bar"])
    assert exc.value.code == 1


# ---------------------------------------------------------------------------
# SAE variant resolution in vector compare / vector why
# ---------------------------------------------------------------------------


def test_split_variant_suffix_parses_sae_variants():
    from saklas.cli.runners import _split_variant_suffix
    assert _split_variant_suffix("honest") == ("honest", None)
    assert _split_variant_suffix("honest:raw") == ("honest", "raw")
    assert _split_variant_suffix("honest:sae") == ("honest", "sae")
    assert _split_variant_suffix("honest:sae-my-release") == (
        "honest", "sae-my-release",
    )
    assert _split_variant_suffix("deer.wolf:sae") == ("deer.wolf", "sae")
    # Prefix selectors pass through untouched — their value carries the colon
    # but _split_variant_suffix only peels a trailing variant token.
    assert _split_variant_suffix("tag:emotion") == ("tag:emotion", None)
    assert _split_variant_suffix("model:google/gemma-3-4b-it") == (
        "model:google/gemma-3-4b-it", None,
    )


def test_run_why_accepts_sae_suffix(monkeypatch, tmp_path, capsys):
    """``vector why foo:sae`` must load the SAE tensor, not the raw one."""
    vdir = _setup_why_env(monkeypatch, tmp_path)
    model_id = "fake/model"
    from saklas.io.paths import safe_model_id, tensor_filename
    sid = safe_model_id(model_id)

    target_dir = vdir / "default" / "angry.calm"
    target_dir.mkdir(parents=True, exist_ok=True)
    # Two tensors for the same model: raw and SAE. The caller picks via suffix.
    raw_path = target_dir / f"{sid}.safetensors"
    sae_path = target_dir / tensor_filename(model_id, release="my-release")
    raw_path.write_bytes(b"x")
    sae_path.write_bytes(b"x")

    raw_profile = _mock_why_profile({0: 0.1, 1: 0.1})
    sae_profile = _mock_why_profile({14: 0.9, 15: 0.8})

    by_path = {str(raw_path): raw_profile, str(sae_path): sae_profile}

    from saklas.core.profile import Profile
    monkeypatch.setattr(Profile, "load", staticmethod(lambda p: by_path[str(p)]))
    from saklas.cli.selectors import invalidate
    invalidate()

    cli.main(["vector", "why", "angry.calm:sae-my-release", "-m", model_id])
    out = capsys.readouterr().out
    # Suffix propagates into the display name.
    assert "angry.calm:sae-my-release" in out
    # The SAE profile magnitudes appear, not the raw ones.
    assert "L14" in out
    assert "0.900" in out
    assert "L0" not in out


def test_run_why_sae_ambiguous_errors(monkeypatch, tmp_path, capsys):
    """``:sae`` with multiple installed SAE variants must error out."""
    vdir = _setup_why_env(monkeypatch, tmp_path)
    model_id = "fake/model"
    from saklas.io.paths import tensor_filename

    target_dir = vdir / "default" / "angry.calm"
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / tensor_filename(model_id, release="release-a")).write_bytes(b"x")
    (target_dir / tensor_filename(model_id, release="release-b")).write_bytes(b"x")

    from saklas.cli.selectors import invalidate
    invalidate()

    with pytest.raises(SystemExit) as exc:
        cli.main(["vector", "why", "angry.calm:sae", "-m", model_id])
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "multiple SAE variants" in err


def test_run_compare_accepts_sae_suffix(monkeypatch, tmp_path, capsys):
    """Each concept in ``vector compare`` parses its own :variant suffix."""
    vdir = _setup_compare_env(monkeypatch, tmp_path)
    model_id = "fake/model"
    from saklas.io.paths import safe_model_id, tensor_filename
    sid = safe_model_id(model_id)

    a_dir = vdir / "default" / "angry.calm"
    b_dir = vdir / "default" / "happy.sad"
    a_dir.mkdir(parents=True, exist_ok=True)
    b_dir.mkdir(parents=True, exist_ok=True)
    # `angry.calm` has a raw tensor and an SAE tensor; `happy.sad` has only raw.
    a_raw = a_dir / f"{sid}.safetensors"
    a_sae = a_dir / tensor_filename(model_id, release="my-release")
    b_raw = b_dir / f"{sid}.safetensors"
    for p in (a_raw, a_sae, b_raw):
        p.write_bytes(b"x")

    a_raw_profile = _mock_profile(lambda o: 0.1, lambda o: {0: 0.1})
    a_sae_profile = _mock_profile(lambda o: 0.9, lambda o: {14: 0.9})
    b_profile = _mock_profile(lambda o: 0.5, lambda o: {0: 0.5})

    by_path = {
        str(a_raw): a_raw_profile,
        str(a_sae): a_sae_profile,
        str(b_raw): b_profile,
    }

    from saklas.core.profile import Profile
    monkeypatch.setattr(Profile, "load", staticmethod(lambda p: by_path[str(p)]))
    from saklas.cli.selectors import invalidate
    invalidate()

    cli.main([
        "vector", "compare",
        "angry.calm:sae-my-release", "happy.sad",
        "-m", model_id,
    ])
    out = capsys.readouterr().out
    # Variant suffix carries into display keys.
    assert "angry.calm:sae-my-release" in out
    assert "happy.sad" in out


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

    # The parser runs pole resolution on the expression: bare ``wolf``
    # resolves to ``deer.wolf`` with sign -1, so the 0.5 coefficient
    # becomes -0.5 in the resulting Steering.alphas.
    from saklas.core.steering_expr import parse_expr
    steering = parse_expr("0.5 wolf")
    assert "deer.wolf" in steering.alphas
    assert steering.alphas["deer.wolf"] == -0.5


# ---------------------------------------------------------------------------
# vector extract --sae / --sae-revision
# ---------------------------------------------------------------------------

def test_vector_extract_parses_sae_flag():
    """--sae RELEASE is captured on the Namespace as `sae`."""
    args = cli.parse_args([
        "vector", "extract", "honest.deceptive",
        "-m", "google/gemma-2-2b-it",
        "--sae", "gemma-scope-2b-pt-res-canonical",
    ])
    assert args.sae == "gemma-scope-2b-pt-res-canonical"
    assert args.sae_revision is None


def test_vector_extract_sae_revision():
    args = cli.parse_args([
        "vector", "extract", "honest.deceptive",
        "-m", "google/gemma-2-2b-it",
        "--sae", "release-x",
        "--sae-revision", "v1.0",
    ])
    assert args.sae == "release-x"
    assert args.sae_revision == "v1.0"


def test_vector_extract_no_sae_defaults_to_none():
    args = cli.parse_args([
        "vector", "extract", "honest.deceptive", "-m", "model",
    ])
    assert args.sae is None
    assert args.sae_revision is None


def test_vector_extract_sae_requires_value():
    """--sae must be followed by a release name; it's not a boolean switch."""
    with pytest.raises(SystemExit):
        cli.parse_args([
            "vector", "extract", "honest.deceptive", "-m", "m", "--sae",
        ])


# ---------------------------------------------------------------------------
# pack push / pack clear --variant
# ---------------------------------------------------------------------------

def test_pack_push_variant_flag_default_raw():
    """--variant default is 'raw' for push — users opt into sharing SAE variants."""
    args = cli.parse_args([
        "pack", "push", "honest.deceptive", "-m", "model",
    ])
    assert args.variant == "raw"


def test_pack_push_variant_all():
    args = cli.parse_args([
        "pack", "push", "honest.deceptive", "-m", "model",
        "--variant", "all",
    ])
    assert args.variant == "all"


def test_pack_push_variant_sae():
    args = cli.parse_args([
        "pack", "push", "honest.deceptive", "-m", "model",
        "--variant", "sae",
    ])
    assert args.variant == "sae"


def test_pack_clear_variant_flag_default_all():
    """--variant default is 'all' for clear — stale extractions should be fully gone."""
    args = cli.parse_args([
        "pack", "clear", "honest.deceptive",
    ])
    assert args.variant == "all"


def test_pack_push_variant_rejects_unknown():
    with pytest.raises(SystemExit):
        cli.parse_args([
            "pack", "push", "honest.deceptive", "-m", "model",
            "--variant", "weird",
        ])
