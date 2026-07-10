from pathlib import Path

import pytest

from saklas.io import paths


def test_default_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("SAKLAS_HOME", raising=False)
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    assert paths.saklas_home() == tmp_path / ".saklas"


def test_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    custom = tmp_path / "custom_root"
    monkeypatch.setenv("SAKLAS_HOME", str(custom))
    assert paths.saklas_home() == custom


def test_subdirs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    assert paths.vectors_dir() == tmp_path / "vectors"
    assert paths.models_dir() == tmp_path / "models"
    assert paths.neutral_statements_path() == tmp_path / "neutral_statements.json"


def test_concept_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    assert paths.concept_dir("default", "happy") == tmp_path / "vectors" / "default" / "happy"


def test_model_dir_flattens_slashes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    assert paths.model_dir("google/gemma-2-2b-it") == tmp_path / "models" / "google__gemma-2-2b-it"


def test_safe_model_id():
    assert paths.safe_model_id("google/gemma-2-2b-it") == "google__gemma-2-2b-it"
    assert paths.safe_model_id("Qwen/Qwen2.5-7B-Instruct") == "Qwen__Qwen2.5-7B-Instruct"
    assert paths.safe_model_id("local-model") == "local-model"
    # Single underscores keep the released cache spelling.
    assert paths.safe_model_id("org/model_name") == "org__model_name"


@pytest.mark.parametrize(
    "model_id",
    [
        "org/model_name",
        "org__model_name",
        r"C:\\models\\local model",
        "local/%model",
        "组织/模型",
        "_z-prefixed/model",
    ],
)
def test_safe_model_id_round_trips_every_supported_id(model_id: str) -> None:
    assert paths.unsafe_model_id(paths.safe_model_id(model_id)) == model_id


def test_safe_sae_suffix_raw():
    assert paths.safe_sae_suffix(None) == ""
    assert paths.safe_sae_suffix("") == ""


def test_safe_sae_suffix_release():
    assert paths.safe_sae_suffix("gemma-scope-2b-pt-res-canonical") == "_sae-gemma-scope-2b-pt-res-canonical"


def test_safe_sae_suffix_slugs_unsafe_chars():
    # Slashes and upper-case get slugged to underscores / lowered.
    assert paths.safe_sae_suffix("Org/Repo") == "_sae-org_repo"


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


def test_tensor_filename_roundtrip_from_transferred(monkeypatch: pytest.MonkeyPatch) -> None:
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


@pytest.mark.parametrize("reserved", ["_sae-", "_from-", "_role-"])
def test_tensor_filename_round_trips_reserved_separator_in_model_id(
    reserved: str,
) -> None:
    model_id = f"org/model{reserved}name"
    filename = paths.tensor_filename(model_id)
    assert paths.parse_tensor_filename(filename) == (
        paths.safe_model_id(model_id), None,
    )


def test_transfer_filename_round_trips_reserved_separators_in_both_ids() -> None:
    target = "org/target_sae-name"
    source = "org/source_from-name_role-x"
    filename = paths.tensor_filename(target, transferred_from=source)
    assert paths.parse_tensor_filename(filename) == (
        paths.safe_model_id(target),
        f"from-{paths.safe_model_id(source).lower()}",
    )


def test_safe_model_id_is_bijective_across_slash_and_literal_double_underscore() -> None:
    assert paths.safe_model_id("org/model") != paths.safe_model_id("org__model")


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


def test_tensor_filename_roundtrip_role():
    name = paths.tensor_filename("google/gemma-2-2b-it", role="pirate")
    assert name == "google__gemma-2-2b-it_role-pirate.safetensors"
    parsed = paths.parse_tensor_filename(name)
    assert parsed == ("google__gemma-2-2b-it", "role-pirate")


def test_tensor_filename_role_excludes_sae():
    import pytest as _pytest

    with _pytest.raises(ValueError, match="mutually exclusive"):
        paths.tensor_filename(
            "google/gemma-2-2b-it",
            role="pirate",
            release="gemma-scope-2b-pt-res-canonical",
        )


def test_tensor_filename_role_excludes_transferred():
    import pytest as _pytest

    with _pytest.raises(ValueError, match="mutually exclusive"):
        paths.tensor_filename(
            "Qwen/Qwen2.5-7B-Instruct",
            role="pirate",
            transferred_from="google/gemma-3-4b-it",
        )


def test_safe_role_suffix_slug_normalization():
    # Uppercase + spaces + slashes get lowercased and slugified per
    # _UNSAFE_VARIANT_CHARS — the same discipline as the SAE suffix.
    assert paths.safe_role_suffix(None) == ""
    assert paths.safe_role_suffix("") == ""
    assert paths.safe_role_suffix("Pirate") == "_role-pirate"
    assert paths.safe_role_suffix("Salty Pirate") == "_role-salty_pirate"
    assert paths.safe_role_suffix("Org/Persona") == "_role-org_persona"


def test_sidecar_filename_role_partners_tensor():
    assert paths.sidecar_filename("google/gemma-2-2b-it", role="pirate") == (
        "google__gemma-2-2b-it_role-pirate.json"
    )
