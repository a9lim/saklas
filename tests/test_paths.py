from saklas.io import paths


def test_default_home(monkeypatch, tmp_path):
    monkeypatch.delenv("SAKLAS_HOME", raising=False)
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    assert paths.saklas_home() == tmp_path / ".saklas"


def test_env_override(monkeypatch, tmp_path):
    custom = tmp_path / "custom_root"
    monkeypatch.setenv("SAKLAS_HOME", str(custom))
    assert paths.saklas_home() == custom


def test_subdirs(monkeypatch, tmp_path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    assert paths.vectors_dir() == tmp_path / "vectors"
    assert paths.models_dir() == tmp_path / "models"
    assert paths.neutral_statements_path() == tmp_path / "neutral_statements.json"


def test_concept_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    assert paths.concept_dir("default", "happy") == tmp_path / "vectors" / "default" / "happy"


def test_model_dir_flattens_slashes(monkeypatch, tmp_path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    assert paths.model_dir("google/gemma-2-2b-it") == tmp_path / "models" / "google__gemma-2-2b-it"


def test_safe_model_id():
    assert paths.safe_model_id("google/gemma-2-2b-it") == "google__gemma-2-2b-it"
    assert paths.safe_model_id("Qwen/Qwen2.5-7B-Instruct") == "Qwen__Qwen2.5-7B-Instruct"
    assert paths.safe_model_id("local-model") == "local-model"
