"""The weight-free ``manifold fit`` / ``manifold extract`` no-op preflight.

Every test here answers one question: can an exact refit be *proven*
unnecessary without instantiating the transformer, and does the proof collapse
the instant any fit input moves?  The failure mode being defended against is
asymmetric — a false "unproven" costs a model load, a false "proven" silently
serves a stale artifact — so the refusal cases outnumber the acceptance case.

CPU only: the fit that publishes the artifact runs through the same synthetic
capture stub the rest of the extraction suite uses, and the preflight's model
touchpoints (source fingerprint, neutral cache, config shape, tokenizer) are
stubbed at their module boundaries.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import saklas.cli.runners as cli_runners
from saklas.core import capture as V
from saklas.core import model as model_mod
from saklas.core.events import EventBus
from saklas.core.extraction import ManifoldExtractionPipeline
from saklas.core.model import loaded_model_fingerprint
from saklas.io import alignment as alignment_mod
from saklas.io.manifold_authoring import create_discover_manifold_folder
from saklas.io.manifold_folder import ManifoldFolder, ManifoldSidecar
from saklas.io.manifolds import preflight_manifold_fit_noop
from saklas.io.paths import tensor_filename
from tests._whitener import synthetic_means, synthetic_whitener
from tests.test_manifold_extraction import (
    _CaptureTokenizer,
    _stub_encoder_batch,
)

_MODEL_ID = "stub/model"
_SOURCE_FP = "5c" * 32
_DIM = 8
_N_LAYERS = 4
_CORPORA = {
    "calm": ["calm statement 0", "calm statement 1"],
    "frantic": ["frantic statement 0", "frantic statement 1"],
}


class _ProvableHandle:
    """A ``ModelHandle`` whose weights carry a provable checkpoint source.

    ``loaded_model_fingerprint`` folds ``_saklas_source_fingerprint`` in as
    ``trusted_source``, and the fit stamps that same attribute onto the sidecar
    — which is exactly the pair the preflight later re-establishes off-model.
    """

    def __init__(self) -> None:
        self.model_id = _MODEL_ID
        self.model: torch.nn.Module = torch.nn.Linear(1, 1)
        self.model._saklas_source_fingerprint = _SOURCE_FP  # type: ignore[assignment]
        self.tokenizer: Any = _CaptureTokenizer()
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.layers: Any = [object()] * _N_LAYERS
        self.layer_means = synthetic_means(range(_N_LAYERS), _DIM)
        self.whitener = synthetic_whitener(
            range(_N_LAYERS), _DIM, means=self.layer_means,
        )

    def _run_generator(
        self, system_msg: str, prompt: str, max_new_tokens: int,
    ) -> str:
        raise NotImplementedError("stub: not called in CPU preflight tests")

    def generate_responses(self, *_args: Any, **_kwargs: Any) -> Any:
        raise NotImplementedError("stub: not called in CPU preflight tests")


@pytest.fixture
def fitted(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    """A published fit plus every stub the preflight's model touchpoints need."""
    torch.manual_seed(0)
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(V, "_encode_and_capture_all_batch", _stub_encoder_batch)
    monkeypatch.setattr(V, "_load_baseline_prompts", lambda: ["baseline prompt"])

    folder = create_discover_manifold_folder(
        "local", "mood", "", fit_mode="pca", node_corpora=dict(_CORPORA),
    )
    handle = _ProvableHandle()
    loaded_fp = loaded_model_fingerprint(handle.model, _MODEL_ID)
    ManifoldExtractionPipeline(handle, EventBus()).fit(folder)

    # The four model touchpoints, each stubbed where the preflight imports it.
    monkeypatch.setattr(
        model_mod, "model_source_fingerprint",
        lambda *_a, **_k: _SOURCE_FP,
    )
    monkeypatch.setattr(
        model_mod, "config_model_shape", lambda _m: (_N_LAYERS, _DIM),
    )
    monkeypatch.setattr(
        alignment_mod, "validate_neutral_cache_metadata",
        lambda _model_id, **_k: {
            "model_source_fingerprint": _SOURCE_FP,
            "model_fingerprint": loaded_fp,
        },
    )
    import transformers

    monkeypatch.setattr(
        transformers.AutoConfig, "from_pretrained",
        classmethod(lambda _cls, *_a, **_k: SimpleNamespace(model_type=None)),
    )
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained",
        classmethod(lambda _cls, *_a, **_k: _CaptureTokenizer()),
    )
    return SimpleNamespace(
        folder=folder, handle=handle, loaded_fp=loaded_fp,
        tensor=folder / tensor_filename(_MODEL_ID),
    )


def _preflight(fitted: Any, **kwargs: Any) -> Any:
    return preflight_manifold_fit_noop(
        fitted.folder, model_id=_MODEL_ID, **kwargs,
    )


def _rewrite_sidecar(fitted: Any, **fields: Any) -> None:
    """Patch the fitted sidecar and re-prove it in the folder manifest.

    Editing the sidecar without refreshing the manifest hash would fail the
    pair proof for the *wrong* reason, so every field-level refusal test goes
    through here and isolates the field it is actually about.
    """
    sidecar_path = fitted.tensor.with_suffix(".json")
    payload = json.loads(sidecar_path.read_text())
    payload.update(fields)
    sidecar_path.write_text(json.dumps(payload))
    ManifoldFolder.load(fitted.folder, verify_manifest=False).update_file_hashes(
        fitted.tensor, sidecar_path,
    )


# ---- the acceptance case --------------------------------------------------


def test_preflight_proves_an_exact_refit_is_a_no_op(fitted: Any) -> None:
    proof = _preflight(fitted)
    assert proof is not None
    assert proof.model_id == _MODEL_ID
    assert proof.manifold_name == "mood"
    assert proof.tensor_path == fitted.tensor
    assert list(proof.node_labels) == list(_CORPORA)
    assert list(proof.fitted_layers) == list(range(_N_LAYERS))


def test_preflight_loads_no_model_weights(
    fitted: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every weight-loading entry point is poisoned for the duration."""
    def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("preflight loaded model weights")

    import transformers

    monkeypatch.setattr(model_mod, "load_model", _boom, raising=False)
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM, "from_pretrained",
        classmethod(lambda *_a, **_k: _boom()),
    )
    assert _preflight(fitted) is not None


# ---- refusals: the fit inputs ---------------------------------------------


def test_preflight_refuses_after_a_baseline_prompt_edit(
    fitted: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact failure a partial proof would hide.

    The corpus files are byte-identical and the model is unchanged; only the
    shared elicitation prompt moved, which changes every captured residual.
    """
    monkeypatch.setattr(V, "_load_baseline_prompts", lambda: ["a new prompt"])
    assert _preflight(fitted) is None


def test_preflight_refuses_after_a_corpus_edit(fitted: Any) -> None:
    node = fitted.folder / "nodes" / "00_calm.json"
    node.write_text(json.dumps(["calm statement 0", "calm statement EDITED"]))
    assert _preflight(fitted) is None


def test_preflight_refuses_on_a_layer_set_change(fitted: Any) -> None:
    assert _preflight(fitted, layer_indices=[0, 1]) is None
    assert _preflight(fitted, layer_indices="workspace") is None
    # The same roster spelled differently is still the same roster.
    assert _preflight(fitted, layer_indices="all") is not None
    assert _preflight(fitted, layer_indices=list(range(_N_LAYERS))) is not None


def test_preflight_refuses_an_old_format_sidecar(fitted: Any) -> None:
    from saklas.io.manifold_folder import MANIFOLD_FORMAT_VERSION

    _rewrite_sidecar(fitted, format_version=MANIFOLD_FORMAT_VERSION - 1)
    assert _preflight(fitted) is None


# ---- refusals: the model identity -----------------------------------------


def test_preflight_refuses_when_the_checkpoint_source_is_unprovable(
    fitted: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        model_mod, "model_source_fingerprint", lambda *_a, **_k: None,
    )
    assert _preflight(fitted) is None


def test_preflight_refuses_when_the_source_no_longer_matches(
    fitted: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        model_mod, "model_source_fingerprint", lambda *_a, **_k: "ab" * 32,
    )
    assert _preflight(fitted) is None


def test_preflight_refuses_when_the_neutral_bridge_names_other_weights(
    fitted: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bridge is what supplies the loaded fingerprint; it must be exact."""
    monkeypatch.setattr(
        alignment_mod, "validate_neutral_cache_metadata",
        lambda _model_id, **_k: {
            "model_source_fingerprint": _SOURCE_FP,
            "model_fingerprint": "different-loaded-weights",
        },
    )
    assert _preflight(fitted) is None


def test_preflight_refuses_without_a_neutral_cache(
    fitted: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _missing(_model_id: str, **_kwargs: Any) -> Any:
        raise FileNotFoundError("no neutral cache")

    monkeypatch.setattr(
        alignment_mod, "validate_neutral_cache_metadata", _missing,
    )
    assert _preflight(fitted) is None


def test_preflight_refuses_when_the_capture_schema_moved(fitted: Any) -> None:
    _rewrite_sidecar(fitted, capture_version=99)
    assert _preflight(fitted) is None


def test_preflight_refuses_a_fit_with_no_provable_capture_identity(
    fitted: Any,
) -> None:
    """A fit whose handle had no callable tokenizer stamps null digests."""
    _rewrite_sidecar(
        fitted, capture_sha256=None, capture_render_sha256=None,
        baseline_prompts_sha256=None,
    )
    assert _preflight(fitted) is None


def test_preflight_refuses_a_fit_from_an_unprovable_checkpoint(
    fitted: Any,
) -> None:
    _rewrite_sidecar(fitted, model_source_fingerprint=None)
    assert _preflight(fitted) is None


# ---- refusals: request shapes that are not provable ------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sae": "some/release"},
        {"fit_mode": "spectral"},
        {"hyperparams": {"max_dim": 2}},
    ],
    ids=["sae", "fit-mode-override", "hyperparam-override"],
)
def test_preflight_declines_unprovable_requests(
    fitted: Any, kwargs: dict[str, Any],
) -> None:
    assert _preflight(fitted, **kwargs) is None


def test_preflight_refuses_a_missing_or_unproven_pair(fitted: Any) -> None:
    fitted.tensor.unlink()
    assert _preflight(fitted) is None


def test_preflight_refuses_a_tampered_tensor(fitted: Any) -> None:
    fitted.tensor.write_bytes(fitted.tensor.read_bytes() + b"\x00")
    assert _preflight(fitted) is None


# ---- the staleness contract: old sidecar = cache miss, never an error ------


def test_old_format_sidecar_reads_as_a_cache_miss_not_an_error(
    fitted: Any,
) -> None:
    """A format bump expires one fit, not the whole artifact.

    The folder must still load — its labels, corpus, and geometry are
    untouched — and simply report no fit for that model, which is what makes
    the next fit a plain overwrite rather than a repair.
    """
    from saklas.io.manifold_folder import MANIFOLD_FORMAT_VERSION

    _rewrite_sidecar(fitted, format_version=MANIFOLD_FORMAT_VERSION - 1)

    mf = ManifoldFolder.load(fitted.folder, verify_manifest=False)
    assert mf.node_labels == list(_CORPORA)
    assert mf.tensor_models() == []


def test_stale_fit_is_overwritten_by_the_next_fit(
    fitted: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from saklas.io.manifold_folder import MANIFOLD_FORMAT_VERSION

    _rewrite_sidecar(fitted, format_version=MANIFOLD_FORMAT_VERSION - 1)
    monkeypatch.setattr(V, "_encode_and_capture_all_batch", _stub_encoder_batch)

    manifold = ManifoldExtractionPipeline(fitted.handle, EventBus()).fit(
        fitted.folder,
    )
    assert manifold.node_labels == list(_CORPORA)
    sidecar = ManifoldSidecar.load(fitted.tensor.with_suffix(".json"))
    assert sidecar.capture_render_sha256 is not None
    assert _preflight(fitted) is not None


def test_stale_fit_resolves_as_a_miss_not_a_codec_error(
    fitted: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Steering / probe resolution must say "refit", not raise a format error.

    ``_bootstrap_manifold_probes`` retries a `ManifoldNotRegisteredError` by
    fitting; a raw `ManifoldFormatError` escapes that retry and silently drops
    the probe, which is how a format bump could empty a session's whole roster.
    """
    from saklas import ManifoldNotRegisteredError
    from saklas.core.steering_composer import SteeringComposer
    from saklas.io.manifold_folder import MANIFOLD_FORMAT_VERSION

    _rewrite_sidecar(fitted, format_version=MANIFOLD_FORMAT_VERSION - 1)

    session = SimpleNamespace(
        _manifolds={}, model_id=_MODEL_ID, _model=fitted.handle.model,
    )
    composer = SteeringComposer.__new__(SteeringComposer)
    composer._session = session  # type: ignore[attr-defined]
    with pytest.raises(ManifoldNotRegisteredError, match="no fitted tensor"):
        composer.ensure_manifold_loaded("local/mood")


def test_corrupt_sidecar_still_raises(fitted: Any) -> None:
    """Staleness is forgiven; corruption is not."""
    from saklas.io.manifold_folder import ManifoldFormatError

    sidecar_path = fitted.tensor.with_suffix(".json")
    payload = json.loads(sidecar_path.read_text())
    payload["node_labels"] = ["only-one"]
    sidecar_path.write_text(json.dumps(payload))
    with pytest.raises(ManifoldFormatError):
        ManifoldFolder.load(fitted.folder, verify_manifest=False)


# ---- CLI adoption ----------------------------------------------------------


def _poison_session(monkeypatch: pytest.MonkeyPatch, calls: list[str]) -> None:
    def _make(*_args: Any, **_kwargs: Any) -> Any:
        calls.append("make_session")
        raise AssertionError("CLI loaded the model on a proven no-op")

    monkeypatch.setattr(cli_runners, "_make_session", _make)
    monkeypatch.setattr(cli_runners, "_print_startup", lambda _args: None)
    monkeypatch.setattr(cli_runners, "_print_model_info", lambda _session: None)


def test_cli_fit_reports_the_hit_without_constructing_a_session(
    fitted: Any, monkeypatch: pytest.MonkeyPatch, capsys: Any,
) -> None:
    import saklas.cli as cli

    calls: list[str] = []
    _poison_session(monkeypatch, calls)
    cli_runners._run_manifold_fit(cli.parse_args([
        "manifold", "fit", str(fitted.folder), "-m", _MODEL_ID,
    ]))
    out = capsys.readouterr().out
    assert "nothing to do" in out
    assert "mood" in out
    assert calls == []


def test_cli_fit_force_still_loads_the_model(
    fitted: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import saklas.cli as cli

    calls: list[str] = []

    def _make(*_args: Any, **_kwargs: Any) -> Any:
        calls.append("make_session")
        raise RuntimeError("stop here")

    monkeypatch.setattr(cli_runners, "_make_session", _make)
    monkeypatch.setattr(cli_runners, "_print_startup", lambda _args: None)
    with pytest.raises(RuntimeError, match="stop here"):
        cli_runners._run_manifold_fit(cli.parse_args([
            "manifold", "fit", str(fitted.folder), "-m", _MODEL_ID, "-f",
        ]))
    assert calls == ["make_session"]


def test_cli_fit_falls_through_when_a_component_is_unproven(
    fitted: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import saklas.cli as cli

    monkeypatch.setattr(
        model_mod, "model_source_fingerprint", lambda *_a, **_k: None,
    )
    calls: list[str] = []

    def _make(*_args: Any, **_kwargs: Any) -> Any:
        calls.append("make_session")
        raise RuntimeError("stop here")

    monkeypatch.setattr(cli_runners, "_make_session", _make)
    monkeypatch.setattr(cli_runners, "_print_startup", lambda _args: None)
    with pytest.raises(RuntimeError, match="stop here"):
        cli_runners._run_manifold_fit(cli.parse_args([
            "manifold", "fit", str(fitted.folder), "-m", _MODEL_ID,
        ]))
    assert calls == ["make_session"]


def test_cli_extract_reports_the_hit_without_constructing_a_session(
    fitted: Any, monkeypatch: pytest.MonkeyPatch, capsys: Any,
) -> None:
    """``extract calm frantic`` canonicalizes onto the same folder."""
    import saklas.cli as cli

    calls: list[str] = []
    _poison_session(monkeypatch, calls)
    # The fixture's folder is local/mood; extract addresses local/calm.frantic,
    # so republish the same artifact under the name extract would author.
    target = fitted.folder.parent / "calm.frantic"
    import shutil

    shutil.copytree(fitted.folder, target)
    payload = json.loads((target / "manifold.json").read_text())
    payload["name"] = "calm.frantic"
    (target / "manifold.json").write_text(json.dumps(payload))
    sidecar_path = target / fitted.tensor.with_suffix(".json").name
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["name"] = "calm.frantic"
    sidecar_path.write_text(json.dumps(sidecar))
    ManifoldFolder.load(target, verify_manifest=False).update_file_hashes(
        target / fitted.tensor.name, sidecar_path,
    )

    cli_runners._run_manifold_extract(cli.parse_args([
        "manifold", "extract", "calm", "frantic", "-m", _MODEL_ID,
    ]))
    out = capsys.readouterr().out
    assert "nothing to do" in out
    assert "calm.frantic" in out
    assert calls == []


def test_cli_extract_role_mismatch_falls_through_to_the_error_path(
    fitted: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A role request against a role-free corpus is an error, not a hit."""
    import saklas.cli as cli

    calls: list[str] = []

    def _make(*_args: Any, **_kwargs: Any) -> Any:
        calls.append("make_session")
        raise RuntimeError("stop here")

    monkeypatch.setattr(cli_runners, "_make_session", _make)
    monkeypatch.setattr(cli_runners, "_print_startup", lambda _args: None)
    with pytest.raises(RuntimeError, match="stop here"):
        cli_runners._run_manifold_extract(cli.parse_args([
            "manifold", "extract", "calm", "frantic", "-m", _MODEL_ID,
            "--role", "pirate",
        ]))
    assert calls == ["make_session"]
