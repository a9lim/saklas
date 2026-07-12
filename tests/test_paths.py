from pathlib import Path

import pytest

from saklas.io import paths

GOOGLE_2B = "_zZ29vZ2xlL2dlbW1hLTItMmItaXQ"
GOOGLE_3_4B = "_zZ29vZ2xlL2dlbW1hLTMtNGItaXQ"
QWEN_2_5 = "_zUXdlbi9Rd2VuMi41LTdCLUluc3RydWN0"


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
    assert paths.models_dir() == tmp_path / "models"
    assert paths.neutral_statements_path() == tmp_path / "neutral_statements.json"


def test_model_dir_flattens_slashes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    assert paths.model_dir("google/gemma-2-2b-it") == tmp_path / "models" / GOOGLE_2B


def test_safe_model_id():
    assert paths.safe_model_id("google/gemma-2-2b-it") == GOOGLE_2B
    assert paths.safe_model_id("Qwen/Qwen2.5-7B-Instruct") == QWEN_2_5
    assert paths.safe_model_id("local-model") == "_zbG9jYWwtbW9kZWw"
    assert paths.safe_model_id("org/model_name") == "_zb3JnL21vZGVsX25hbWU"
    assert all(paths.safe_model_id(value).startswith("_z") for value in (
        "org/model", "local-model", "org__model", "组织/模型",
    ))


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


@pytest.mark.parametrize("value", ["", "plain-model", "_z", "_zYQ==", "_z***"])
def test_model_id_codec_rejects_non_current_or_noncanonical_values(value: str) -> None:
    with pytest.raises(ValueError):
        if value:
            paths.unsafe_model_id(value)
        else:
            paths.safe_model_id(value)


def test_safe_sae_suffix_raw():
    assert paths.safe_sae_suffix(None) == ""
    assert paths.safe_sae_suffix("") == ""


def test_safe_sae_suffix_release():
    encoded = paths.encode_release_id("gemma-scope-2b-pt-res-canonical")
    assert paths.safe_sae_suffix("gemma-scope-2b-pt-res-canonical") == f"_sae-{encoded}"
    assert paths.decode_release_id(encoded) == "gemma-scope-2b-pt-res-canonical"


def test_safe_sae_suffix_slugs_unsafe_chars():
    # Distinct release identities remain distinct and reversible.
    assert paths.safe_sae_suffix("Org/Repo") != paths.safe_sae_suffix("org_repo")
    assert paths.decode_release_id(paths.encode_release_id("Org/Repo")) == "Org/Repo"


def test_tensor_filename_roundtrip_raw():
    name = paths.tensor_filename("google/gemma-2-2b-it", release=None)
    assert name == f"{GOOGLE_2B}.safetensors"
    parsed = paths.parse_tensor_filename(name)
    assert parsed == (GOOGLE_2B, None)


def test_tensor_filename_roundtrip_sae():
    name = paths.tensor_filename("google/gemma-2-2b-it", release="gemma-scope-2b-pt-res-canonical")
    release = paths.encode_release_id("gemma-scope-2b-pt-res-canonical")
    assert name == f"{GOOGLE_2B}_sae-{release}.safetensors"
    parsed = paths.parse_tensor_filename(name)
    # parse_tensor_filename returns the kind-prefixed variant slug so
    # callers can dispatch on prefix without re-detecting the kind.
    assert parsed == (GOOGLE_2B, f"sae-{release}")


def test_tensor_filename_roundtrip_from_transferred(monkeypatch: pytest.MonkeyPatch) -> None:
    # Transferred profiles ride the same variant-suffix machinery as SAE.
    name = paths.tensor_filename(
        "Qwen/Qwen2.5-7B-Instruct",
        transferred_from="google/gemma-3-4b-it",
    )
    source = paths.encode_release_id("google/gemma-3-4b-it")
    assert name == f"{QWEN_2_5}_from-{source}.safetensors"
    parsed = paths.parse_tensor_filename(name)
    assert parsed == (QWEN_2_5, f"from-{source}")


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
        f"from-{paths.encode_release_id(source)}",
    )


def test_safe_model_id_is_bijective_across_slash_and_literal_double_underscore() -> None:
    assert paths.safe_model_id("org/model") != paths.safe_model_id("org__model")


def test_sidecar_filename_partners_tensor():
    assert paths.sidecar_filename("google/gemma-2-2b-it", release=None) == f"{GOOGLE_2B}.json"
    assert paths.sidecar_filename(
        "google/gemma-2-2b-it", release="gemma-scope-2b-pt-res-canonical",
    ) == f"{GOOGLE_2B}_sae-{paths.encode_release_id('gemma-scope-2b-pt-res-canonical')}.json"
    assert paths.sidecar_filename(
        "Qwen/Qwen2.5-7B-Instruct",
        transferred_from="google/gemma-3-4b-it",
    ) == f"{QWEN_2_5}_from-{paths.encode_release_id('google/gemma-3-4b-it')}.json"


def test_safe_from_suffix_slugs_unsafe_chars():
    # Source ids use the same reversible lowercase identity codec as releases.
    assert paths.safe_from_suffix(None) == ""
    assert paths.safe_from_suffix("") == ""
    assert paths.safe_from_suffix("google/gemma-3-4b-it") == (
        f"_from-{paths.encode_release_id('google/gemma-3-4b-it')}"
    )
    assert paths.safe_from_suffix("Org/Model") != paths.safe_from_suffix("org/model")


def test_role_like_text_is_part_of_raw_model_id() -> None:
    name = paths.tensor_filename("org/model_role-pirate")
    assert paths.parse_tensor_filename(name) == (
        paths.safe_model_id("org/model_role-pirate"), None,
    )
