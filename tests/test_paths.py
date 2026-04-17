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


def test_safe_variant_suffix_raw():
    assert paths.safe_variant_suffix(None) == ""
    assert paths.safe_variant_suffix("") == ""


def test_safe_variant_suffix_release():
    assert paths.safe_variant_suffix("gemma-scope-2b-pt-res-canonical") == "_sae-gemma-scope-2b-pt-res-canonical"


def test_safe_variant_suffix_slugs_unsafe_chars():
    # Slashes and upper-case get slugged to underscores / lowered.
    assert paths.safe_variant_suffix("Org/Repo") == "_sae-org_repo"


def test_tensor_filename_roundtrip_raw():
    name = paths.tensor_filename("google/gemma-2-2b-it", release=None)
    assert name == "google__gemma-2-2b-it.safetensors"
    parsed = paths.parse_tensor_filename(name)
    assert parsed == ("google__gemma-2-2b-it", None)


def test_tensor_filename_roundtrip_sae():
    name = paths.tensor_filename("google/gemma-2-2b-it", release="gemma-scope-2b-pt-res-canonical")
    assert name == "google__gemma-2-2b-it_sae-gemma-scope-2b-pt-res-canonical.safetensors"
    parsed = paths.parse_tensor_filename(name)
    assert parsed == ("google__gemma-2-2b-it", "gemma-scope-2b-pt-res-canonical")


def test_parse_tensor_filename_rejects_non_safetensors():
    assert paths.parse_tensor_filename("model.json") is None
    assert paths.parse_tensor_filename("model.gguf") is None


def test_sidecar_filename_partners_tensor():
    assert paths.sidecar_filename("google/gemma-2-2b-it", release=None) == "google__gemma-2-2b-it.json"
    assert paths.sidecar_filename(
        "google/gemma-2-2b-it", release="gemma-scope-2b-pt-res-canonical",
    ) == "google__gemma-2-2b-it_sae-gemma-scope-2b-pt-res-canonical.json"
