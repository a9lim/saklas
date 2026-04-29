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
    # parse_tensor_filename returns the kind-prefixed variant slug so
    # callers can dispatch on prefix without re-detecting the kind.
    assert parsed == ("google__gemma-2-2b-it", "sae-gemma-scope-2b-pt-res-canonical")


def test_tensor_filename_roundtrip_from_transferred(monkeypatch):
    # Transferred profiles ride the same variant-suffix machinery as SAE.
    name = paths.tensor_filename(
        "Qwen/Qwen2.5-7B-Instruct",
        transferred_from="google/gemma-3-4b-it",
    )
    assert name == "Qwen__Qwen2.5-7B-Instruct_from-google__gemma-3-4b-it.safetensors"
    parsed = paths.parse_tensor_filename(name)
    assert parsed == ("Qwen__Qwen2.5-7B-Instruct", "from-google__gemma-3-4b-it")


def test_tensor_filename_rejects_double_variant():
    import pytest as _pytest

    with _pytest.raises(ValueError, match="mutually exclusive"):
        paths.tensor_filename(
            "Qwen/Qwen2.5-7B-Instruct",
            release="gemma-scope-2b-pt-res-canonical",
            transferred_from="google/gemma-3-4b-it",
        )


def test_parse_tensor_filename_rejects_non_safetensors():
    assert paths.parse_tensor_filename("model.json") is None
    assert paths.parse_tensor_filename("model.gguf") is None


def test_sidecar_filename_partners_tensor():
    assert paths.sidecar_filename("google/gemma-2-2b-it", release=None) == "google__gemma-2-2b-it.json"
    assert paths.sidecar_filename(
        "google/gemma-2-2b-it", release="gemma-scope-2b-pt-res-canonical",
    ) == "google__gemma-2-2b-it_sae-gemma-scope-2b-pt-res-canonical.json"
    assert paths.sidecar_filename(
        "Qwen/Qwen2.5-7B-Instruct",
        transferred_from="google/gemma-3-4b-it",
    ) == "Qwen__Qwen2.5-7B-Instruct_from-google__gemma-3-4b-it.json"


def test_safe_from_suffix_slugs_unsafe_chars():
    # Source ids are usually already safe (since the caller passes the
    # safe form), but the helper is defensively idempotent on slashes.
    assert paths.safe_from_suffix(None) == ""
    assert paths.safe_from_suffix("") == ""
    assert paths.safe_from_suffix("google__gemma-3-4b-it") == "_from-google__gemma-3-4b-it"
