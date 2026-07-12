"""ManifoldExtractionPipeline tests — CPU only, synthetic encoder.

Mirrors the stub-encoder pattern in :mod:`tests.test_dim_extraction`:
monkeypatch :func:`saklas.core.vectors._encode_and_capture_all` so no
real model is needed.
"""
from __future__ import annotations

import hashlib
import gc
import json
import threading
import weakref
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from saklas.core import vectors as V
from saklas.core.events import EventBus, ManifoldExtracted
from saklas.core.extraction import ManifoldExtractionPipeline
from saklas.core.sae import MockSaeBackend
from saklas.io.manifolds import MANIFOLD_FORMAT_VERSION, ManifoldFolder
from tests._whitener import synthetic_means, synthetic_whitener

_LABELS = ["calm", "uneasy", "afraid", "frantic", "numb"]
_DIM = 8
_N_LAYERS = 4


class _CaptureTokenizer:
    chat_template = None
    pad_token_id = 0
    eos_token_id = 0
    bos_token_id = 0
    all_special_ids: list[int] = []
    added_tokens_encoder: dict[str, int] = {}

    def __call__(
        self, text: str, *, return_tensors: str = "pt",
        add_special_tokens: bool = False,
    ) -> dict[str, torch.Tensor]:
        del return_tensors, add_special_tokens
        ids = [1 + (ord(char) % 97) for char in text] or [1]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}


def _stub_encoder(
    model: Any, tokenizer: Any, prompt: str, response: str, layers: Any,
    device: Any, **_kwargs: Any,
) -> dict[int, torch.Tensor]:
    """Synthetic per-layer activations, deterministic per node label.

    Conversational signature: ignores the baseline ``prompt`` and keys off the
    ``response`` (the node corpus entry, ``"<label> statement i"``)."""
    label = response.split()[0]
    seed = int(hashlib.sha256(label.encode("utf-8")).hexdigest()[:8], 16)
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(_DIM, generator=g)
    out: dict[int, torch.Tensor] = {}
    for layer in range(len(layers)):
        v = base.clone()
        v[2] = 0.5 * layer            # layer-varying offset
        out[layer] = v + 0.01 * torch.randn(_DIM)
    return out


def _stub_encoder_batch(
    model: Any, tokenizer: Any, prompts: Any, responses: Any, layers: Any,
    device: Any, **kwargs: Any,
) -> dict[int, torch.Tensor]:
    """Batched seam matching ``vectors._encode_and_capture_all_batch``.

    Stacks the per-row :func:`_stub_encoder` over the chunk (same call order, so
    the deterministic-per-label activations and RNG consumption are identical to
    the old per-row capture).  Returns ``{layer: (B, D)}``."""
    rows = [
        _stub_encoder(model, tokenizer, p, r, layers, device, **kwargs)
        for p, r in zip(prompts, responses)
    ]
    return {
        idx: torch.stack([row[idx] for row in rows])
        for idx in range(len(layers))
    }


class _Handle:
    """Minimal ModelHandle stub.

    Satisfies the ``ModelHandle`` protocol used by
    ``ManifoldExtractionPipeline``.  The generator methods are never
    called in CPU-only manifold-extraction tests; they exist only to
    complete the structural protocol.
    """

    def __init__(self) -> None:
        self.model_id = "stub-model"
        # Use a real nn.Module so the protocol's ``model: nn.Module`` is met.
        self.model: torch.nn.Module = torch.nn.Linear(1, 1)
        self.tokenizer: Any = _CaptureTokenizer()
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.layers: Any = [object()] * _N_LAYERS
        # Activation-space fits require a covering whitener (no Euclidean
        # fallback post-4.0); synthesize one + matching neutral means on CPU.
        self.layer_means = synthetic_means(range(_N_LAYERS), _DIM)
        self.whitener = synthetic_whitener(
            range(_N_LAYERS), _DIM, means=self.layer_means,
        )

    def _run_generator(
        self, system_msg: str, prompt: str, max_new_tokens: int,
    ) -> str:
        raise NotImplementedError("stub: not called in CPU manifold tests")

    def generate_responses(
        self,
        concepts: list[str],
        kinds: list[str | None],
        *,
        roles: dict[str, str | None] | None = None,
        custom_system: str | None = None,
        samples_per_prompt: int = 1,
        max_new_tokens: int = 256,
        on_progress: Any = None,
    ) -> dict[str, list[str]]:
        raise NotImplementedError("stub: not called in CPU manifold tests")


def _box1d_domain(periodic: bool, k: int) -> dict[str, Any]:
    axis = (
        {"name": "t", "periodic": True, "period": 1.0}
        if periodic
        else {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}
    )
    return {"type": "box", "axes": [axis]}


def _author_manifold(
    root: Path,
    *,
    periodic: bool = True,
    labels: list[str] | None = None,
    domain: dict[str, Any] | None = None,
    coords: list[list[float]] | None = None,
) -> Path:
    labels = labels or _LABELS
    folder = root / "mood"
    (folder / "nodes").mkdir(parents=True)
    for idx, label in enumerate(labels):
        (folder / "nodes" / f"{idx:02d}_{label}.json").write_text(
            json.dumps([f"{label} statement {i}" for i in range(3)])
        )
    k = len(labels)
    if domain is None:
        domain = _box1d_domain(periodic, k)
    if coords is None:
        if periodic:
            coords = [[i / k] for i in range(k)]
        else:
            coords = [[i / (k - 1)] for i in range(k)]
    (folder / "manifold.json").write_text(json.dumps({
        "format_version": MANIFOLD_FORMAT_VERSION,
        "name": "mood",
        "description": "moods",
        "fit_mode": "authored",
        "domain": domain,
        "nodes": [
            {"label": label, "coords": coords[i]}
            for i, label in enumerate(labels)
        ],
        "files": {},
    }))
    return folder


@pytest.fixture(autouse=True)
def _stub(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    torch.manual_seed(0)
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path / "saklas-home"))
    monkeypatch.setattr(V, "_encode_and_capture_all_batch", _stub_encoder_batch)
    # Single baseline prompt so any node corpus length is a multiple of k=1
    # (the conversational alignment invariant); the stub ignores the prompt.
    monkeypatch.setattr(V, "_load_baseline_prompts", lambda: ["baseline prompt"])


def test_fit_produces_manifold(tmp_path: Path) -> None:
    folder = _author_manifold(tmp_path)
    pipe = ManifoldExtractionPipeline(_Handle(), EventBus())
    manifold = pipe.fit(folder)

    assert manifold.name == "mood"
    assert manifold.domain.intrinsic_dim == 1
    assert manifold.node_labels == _LABELS
    assert sorted(manifold.layers) == list(range(_N_LAYERS))
    assert manifold.feature_space == "raw"


def test_templated_fit_rematerializes_value_and_assistant_edits() -> None:
    from saklas.io.manifolds import create_manifold_from_template
    from saklas.io.templates import create_template_folder

    create_template_folder(
        "local", "weekday", slot="[DAY]",
        values=["Monday", "Tuesday", "Sunday"],
        contexts=[{
            "turns": [{"role": "user", "content": "which day?"}],
            "assistant": "[DAY] speaks",
        }],
    )
    folder = create_manifold_from_template(
        "local", "weekday-fit", "", template_ref="local/weekday",
        fit_mode="pca",
    )
    pipe = ManifoldExtractionPipeline(_Handle(), EventBus())
    pipe.fit(folder)

    create_template_folder(
        "local", "weekday", slot="[DAY]",
        values=["Monday", "Wednesday", "Sunday"],
        contexts=[{
            "turns": [{"role": "user", "content": "which day?"}],
            "assistant": "[DAY] answers now",
        }],
        force=True,
    )
    fitted = pipe.fit(folder)
    mf = ManifoldFolder.load(folder, verify_manifest=False)

    assert fitted.node_labels == ["monday", "wednesday", "sunday"]
    assert dict(mf.node_groups())["wednesday"] == ["Wednesday answers now"]


def test_template_mutation_between_resolve_and_hash_cannot_publish_stale_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from saklas.core.extraction import ManifoldAuthoringChangedError
    from saklas.io.manifolds import create_manifold_from_template
    from saklas.io.templates import TemplateFolder, create_template_folder

    create_template_folder(
        "local", "racy-template", slot="[X]", values=["A", "B", "C"],
        contexts=[{
            "turns": [{"role": "user", "content": "choose"}],
            "assistant": "[X] old",
        }],
    )
    folder = create_manifold_from_template(
        "local", "racy-fit", "", template_ref="local/racy-template",
        fit_mode="pca",
    )
    real_sha = TemplateFolder.sha256
    mutated = False

    def _mutating_sha(self: TemplateFolder) -> str:
        nonlocal mutated
        digest = real_sha(self)
        if not mutated:
            mutated = True
            create_template_folder(
                "local", "racy-template", slot="[X]",
                values=["A", "B", "C"],
                contexts=[{
                    "turns": [{"role": "user", "content": "choose"}],
                    "assistant": "[X] new",
                }],
                force=True,
            )
        return digest

    monkeypatch.setattr(TemplateFolder, "sha256", _mutating_sha)
    with pytest.raises(ManifoldAuthoringChangedError):
        ManifoldExtractionPipeline(_Handle(), EventBus()).fit(folder)


def test_fit_can_restrict_transformer_layers(tmp_path: Path) -> None:
    folder = _author_manifold(tmp_path)
    pipe = ManifoldExtractionPipeline(_Handle(), EventBus())
    manifold = pipe.fit(folder, layer_indices=[1, 3])
    assert sorted(manifold.layers) == [1, 3]

    from saklas.io.manifolds import ManifoldSidecar
    from saklas.io.paths import tensor_filename

    sidecar = ManifoldSidecar.load(
        (folder / tensor_filename("stub-model")).with_suffix(".json")
    )
    assert sidecar.fitted_layers == [1, 3]


def test_fit_workspace_layers_match_canonical_band(tmp_path: Path) -> None:
    folder = _author_manifold(tmp_path)
    manifold = ManifoldExtractionPipeline(_Handle(), EventBus()).fit(
        folder, layer_indices="workspace",
    )
    # Canonical predicate: 0.40 <= L/(n_layers-1) <= 0.90.
    assert sorted(manifold.layers) == [2]


def test_fit_layer_scope_invalidates_incompatible_tensor_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    folder = _author_manifold(tmp_path)
    pipe = ManifoldExtractionPipeline(_Handle(), EventBus())
    pipe.fit(folder, layer_indices=[1, 3])
    captured_scopes: list[list[int]] = []

    def _counting(*args: Any, **kwargs: Any) -> Any:
        captured_scopes.append(list(kwargs["layer_indices"]))
        return _stub_encoder_batch(*args, **kwargs)

    monkeypatch.setattr(V, "_encode_and_capture_all_batch", _counting)
    manifold = pipe.fit(folder)
    assert sorted(manifold.layers) == list(range(_N_LAYERS))
    assert captured_scopes == [[0, 2]]


def test_full_capture_cache_serves_subset_without_forward(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    folder = _author_manifold(tmp_path)
    pipe = ManifoldExtractionPipeline(_Handle(), EventBus())
    pipe.fit(folder)

    import saklas.core.extraction as extraction_module
    from saklas.core.manifold import ActivationRowStore

    loaded_scopes: list[list[int] | None] = []
    loaded_centroid_shards: list[str] = []
    real_load = ActivationRowStore.load_shards.__func__
    real_load_file = extraction_module.load_file

    def _tracked_load(
        cls: type[ActivationRowStore], paths: dict[int, Path], node_sizes: Any,
    ) -> ActivationRowStore:
        loaded_scopes.append(sorted(paths))
        return real_load(cls, paths, node_sizes)

    def _tracked_load_file(path: str, *args: Any, **kwargs: Any) -> Any:
        loaded_centroid_shards.append(Path(path).name)
        return real_load_file(path, *args, **kwargs)

    monkeypatch.setattr(
        ActivationRowStore, "load_shards", classmethod(_tracked_load),
    )
    monkeypatch.setattr(extraction_module, "load_file", _tracked_load_file)

    def _must_not_capture(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("full capture cache did not serve a layer subset")

    monkeypatch.setattr(V, "_encode_and_capture_all_batch", _must_not_capture)
    subset = pipe.fit(folder, layer_indices=[1, 3])
    assert sorted(subset.layers) == [1, 3]
    assert loaded_scopes == [[1], [3]]
    assert sorted(
        int(name.split("layer_", 1)[1].split(".", 1)[0])
        for name in loaded_centroid_shards
    ) == [1, 3]


def test_shared_capture_stem_lock_serializes_independent_folders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The per-model capture is one transaction across manifold folders."""
    from saklas.io import atomic

    folder_a = _author_manifold(tmp_path / "a")
    folder_b = _author_manifold(tmp_path / "b")
    handle = _Handle()
    entered_capture = threading.Event()
    release_capture = threading.Event()
    second_waiting = threading.Event()
    second_acquired = threading.Event()
    calls = 0
    calls_guard = threading.Lock()

    def _blocking_encoder(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        with calls_guard:
            calls += 1
            call_number = calls
        if call_number == 1:
            entered_capture.set()
            assert release_capture.wait(timeout=5)
        return _stub_encoder_batch(*args, **kwargs)

    real_artifact_lock = atomic.artifact_lock

    @contextmanager
    def _tracked_lock(path: Path):
        is_second_capture = (
            threading.current_thread().name == "capture-second"
            and path.parent.name == "manifold_capture"
        )
        if is_second_capture:
            second_waiting.set()
        with real_artifact_lock(path):
            if is_second_capture:
                second_acquired.set()
            yield

    monkeypatch.setattr(V, "_encode_and_capture_all_batch", _blocking_encoder)
    monkeypatch.setattr(atomic, "artifact_lock", _tracked_lock)
    errors: list[BaseException] = []

    def _run(folder: Path) -> None:
        try:
            ManifoldExtractionPipeline(handle, EventBus()).fit(
                folder, layer_indices=[0],
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    first = threading.Thread(target=_run, args=(folder_a,), name="capture-first")
    second = threading.Thread(target=_run, args=(folder_b,), name="capture-second")
    first.start()
    assert entered_capture.wait(timeout=5)
    second.start()
    assert second_waiting.wait(timeout=5)
    assert not second_acquired.is_set()
    release_capture.set()
    first.join(timeout=10)
    second.join(timeout=10)

    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    assert second_acquired.is_set()
    assert calls == 1


def test_cold_fit_prepares_token_identity_once_across_capture_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import saklas.core.extraction as extraction_module

    folder = _author_manifold(tmp_path)
    real_prepare = extraction_module.prepare_manifold_capture_identity
    calls = 0

    def _counting_prepare(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return real_prepare(*args, **kwargs)

    monkeypatch.setattr(
        extraction_module, "prepare_manifold_capture_identity",
        _counting_prepare,
    )
    ManifoldExtractionPipeline(_Handle(), EventBus()).fit(
        folder, layer_indices=[0],
    )

    assert calls == 1


def test_fit_overrides_share_manifest_transaction_without_lost_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Conflicting override requests serialize through cache-key derivation."""
    from saklas.io.manifold_authoring import create_discover_manifold_folder

    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path / "home"))
    folder = create_discover_manifold_folder(
        "local", "locked-overrides", "", fit_mode="auto",
        node_corpora={
            "alpha": ["a"], "beta": ["b"], "gamma": ["c"],
            "delta": ["d"], "epsilon": ["e"],
        },
        node_roles={"alpha": "captain"},
        node_kinds={"alpha": "concrete"},
    )
    first_entered = threading.Event()
    release_first = threading.Event()
    second_attempted = threading.Event()
    second_entered = threading.Event()
    observations: list[tuple[str, dict[str, object]]] = []
    errors: list[BaseException] = []

    def _observe_locked(
        _self: ManifoldExtractionPipeline, target: Path, **_kwargs: Any,
    ) -> object:
        mf = ManifoldFolder.load(target, verify_manifest=False)
        observations.append((mf.fit_mode, dict(mf.hyperparams)))
        if len(observations) == 1:
            first_entered.set()
            assert release_first.wait(timeout=5)
        else:
            second_entered.set()
        return object()

    monkeypatch.setattr(
        ManifoldExtractionPipeline, "_fit_locked", _observe_locked,
    )
    pipe = ManifoldExtractionPipeline(_Handle(), EventBus())

    def _run_first() -> None:
        try:
            pipe.fit(folder, fit_mode="spectral", hyperparams={"k_nn": 3})
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    def _run_second() -> None:
        second_attempted.set()
        try:
            pipe.fit(folder, fit_mode="pca", hyperparams={"max_dim": 2})
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    first = threading.Thread(target=_run_first)
    second = threading.Thread(target=_run_second)
    first.start()
    assert first_entered.wait(timeout=5)
    second.start()
    assert second_attempted.wait(timeout=5)
    assert not second_entered.wait(timeout=0.1)
    release_first.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert not first.is_alive() and not second.is_alive()
    assert errors == []
    assert observations == [
        ("spectral", {"k_nn": 3}),
        ("pca", {"max_dim": 2}),
    ]
    final = ManifoldFolder.load(folder, verify_manifest=False)
    assert final.fit_mode == "pca"
    assert final.hyperparams == {"max_dim": 2}
    assert final.node_roles[0] == "captain"
    assert final.node_kinds[0] == "concrete"


def test_distinct_model_targets_compute_concurrently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    folder = _author_manifold(tmp_path)
    barrier = threading.Barrier(2)
    entered: list[str] = []
    errors: list[BaseException] = []

    def _overlap(
        self: ManifoldExtractionPipeline, _folder: Path, **_kwargs: Any,
    ) -> object:
        entered.append(self._handle.model_id)  # pyright: ignore[reportPrivateUsage]
        barrier.wait(timeout=5)
        return object()

    monkeypatch.setattr(ManifoldExtractionPipeline, "_fit_locked", _overlap)
    handles = [_Handle(), _Handle()]
    handles[1].model_id = "other-model"

    def _run(handle: _Handle) -> None:
        try:
            ManifoldExtractionPipeline(handle, EventBus()).fit(folder)
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    threads = [threading.Thread(target=_run, args=(handle,)) for handle in handles]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    assert errors == []
    assert sorted(entered) == ["other-model", "stub-model"]


def test_target_fit_lock_name_is_bounded_for_long_valid_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    folder = _author_manifold(tmp_path / ("f" * 64))
    handle = _Handle()
    handle.model_id = "org/" + ("model" * 30)
    monkeypatch.setattr(
        ManifoldExtractionPipeline, "_fit_locked",
        lambda *_args, **_kwargs: object(),
    )
    ManifoldExtractionPipeline(handle, EventBus()).fit(folder)


def test_reader_stays_available_and_authoring_cas_rejects_stale_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import saklas.core.manifold as manifold_module
    from saklas.core.extraction import ManifoldAuthoringChangedError
    from saklas.core.manifold import load_manifold
    from saklas.io.paths import tensor_filename

    folder = _author_manifold(tmp_path)
    pipe = ManifoldExtractionPipeline(_Handle(), EventBus())
    pipe.fit(folder)
    target = folder / tensor_filename("stub-model")
    entered = threading.Event()
    release = threading.Event()
    real_fit = manifold_module.fit_layer_subspace
    first = True

    def _blocking_fit(*args: Any, **kwargs: Any) -> Any:
        nonlocal first
        if first:
            first = False
            entered.set()
            assert release.wait(timeout=5)
        return real_fit(*args, **kwargs)

    monkeypatch.setattr(manifold_module, "fit_layer_subspace", _blocking_fit)
    errors: list[BaseException] = []

    def _run() -> None:
        try:
            pipe.fit(folder, force=True)
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    thread = threading.Thread(target=_run)
    thread.start()
    assert entered.wait(timeout=5)
    assert load_manifold(target).name == "mood"
    node_path = folder / "nodes" / "00_calm.json"
    node_path.write_text(json.dumps(["calm changed"] * 3))
    release.set()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], ManifoldAuthoringChangedError)


def test_interrupted_pair_publish_is_retryable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import saklas.core.manifold as manifold_module

    folder = _author_manifold(tmp_path)
    pipe = ManifoldExtractionPipeline(_Handle(), EventBus())
    real_replace = manifold_module._replace_manifold_file
    failed = False

    def _fail_tensor_once(source: Path, target: Path) -> None:
        nonlocal failed
        if target.suffix == ".safetensors" and not failed:
            failed = True
            raise OSError("injected tensor commit failure")
        real_replace(source, target)

    monkeypatch.setattr(
        manifold_module, "_replace_manifold_file", _fail_tensor_once,
    )
    with pytest.raises(OSError, match="injected"):
        pipe.fit(folder)
    assert not (folder / "stub-model.safetensors").exists()
    assert (folder / "stub-model.json").exists()

    monkeypatch.setattr(
        manifold_module, "_replace_manifold_file", real_replace,
    )
    fitted = pipe.fit(folder)
    assert fitted.name == "mood"
    assert (folder / "stub-model.safetensors").exists()


def test_interrupted_existing_pair_replacement_self_repairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import saklas.core.manifold as manifold_module
    from saklas.core.manifold import load_manifold
    from saklas.io.paths import tensor_filename

    folder = _author_manifold(tmp_path)
    pipe = ManifoldExtractionPipeline(_Handle(), EventBus())
    pipe.fit(folder)
    target = folder / tensor_filename("stub-model")
    real_replace = manifold_module._replace_manifold_file
    failed = False

    def _fail_between_pair_replacements(source: Path, dest: Path) -> None:
        nonlocal failed
        if dest == target and not failed:
            failed = True
            raise OSError("injected replacement split")
        real_replace(source, dest)

    monkeypatch.setattr(
        manifold_module, "_replace_manifold_file", _fail_between_pair_replacements,
    )
    with pytest.raises(OSError, match="replacement split"):
        pipe.fit(folder, force=True)
    monkeypatch.setattr(manifold_module, "_replace_manifold_file", real_replace)
    pipe.fit(folder)
    assert load_manifold(target).name == "mood"
    ManifoldFolder.load(folder)


def test_pair_committed_before_manifest_failure_self_repairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from saklas.core.manifold import load_manifold
    from saklas.io.paths import tensor_filename

    folder = _author_manifold(tmp_path)
    pipe = ManifoldExtractionPipeline(_Handle(), EventBus())
    pipe.fit(folder)
    target = folder / tensor_filename("stub-model")
    real_update = ManifoldFolder.update_file_hashes
    failed = False

    def _fail_manifest_once(self: ManifoldFolder, *paths: Path) -> None:
        nonlocal failed
        if not failed:
            failed = True
            raise OSError("injected manifest commit failure")
        real_update(self, *paths)

    monkeypatch.setattr(ManifoldFolder, "update_file_hashes", _fail_manifest_once)
    with pytest.raises(OSError, match="manifest commit"):
        pipe.fit(folder, force=True)
    monkeypatch.setattr(ManifoldFolder, "update_file_hashes", real_update)
    pipe.fit(folder)
    assert load_manifold(target).name == "mood"
    ManifoldFolder.load(folder)


def test_clear_invalidates_only_selected_inflight_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from saklas.core.extraction import (
        ManifoldAuthoringChangedError,
        _authoring_snapshot_locked,
        _publish_fit_if_current,
    )
    from saklas.io import manifold_lifecycle
    from saklas.io.manifold_folder import _locked_manifest
    from saklas.io.paths import tensor_filename

    folder = _author_manifold(tmp_path)
    handle_a = _Handle()
    fit_a = ManifoldExtractionPipeline(handle_a, EventBus()).fit(folder)
    handle_b = _Handle()
    handle_b.model_id = "other-model"
    fit_b = ManifoldExtractionPipeline(handle_b, EventBus()).fit(folder)
    target_a = folder / tensor_filename(handle_a.model_id)
    target_b = folder / tensor_filename(handle_b.model_id)
    with _locked_manifest(folder):
        _m, _n, revision_a, _g, _b = _authoring_snapshot_locked(
            folder, target_a,
        )
        _m, _n, revision_b, _g, _b = _authoring_snapshot_locked(
            folder, target_b,
        )

    monkeypatch.setattr(manifold_lifecycle, "manifold_dir", lambda *_: folder)
    manifold_lifecycle.clear_manifold_tensors(
        "local", "mood", handle_a.model_id, variant="raw",
    )
    with pytest.raises(ManifoldAuthoringChangedError):
        _publish_fit_if_current(
            folder, expected_revision=revision_a, manifold=fit_a,
            tensor_path=target_a, metadata=dict(fit_a.metadata),
        )
    _publish_fit_if_current(
        folder, expected_revision=revision_b, manifold=fit_b,
        tensor_path=target_b, metadata=dict(fit_b.metadata),
    )
    assert not target_a.exists()
    assert target_b.exists()


def test_removed_and_recreated_folder_rejects_stale_publication(
    tmp_path: Path,
) -> None:
    from saklas.core.extraction import (
        ManifoldAuthoringChangedError,
        _authoring_snapshot_locked,
        _publish_fit_if_current,
    )
    from saklas.io.manifold_folder import _locked_manifest
    from saklas.io.paths import tensor_filename

    folder = _author_manifold(tmp_path)
    fitted = ManifoldExtractionPipeline(_Handle(), EventBus()).fit(folder)
    target = folder / tensor_filename("stub-model")
    with _locked_manifest(folder):
        _m, _n, revision, _g, _b = _authoring_snapshot_locked(folder, target)
    import shutil

    shutil.rmtree(folder)
    replacement = _author_manifold(tmp_path)
    assert replacement == folder
    with pytest.raises(ManifoldAuthoringChangedError):
        _publish_fit_if_current(
            folder, expected_revision=revision, manifold=fitted,
            tensor_path=target, metadata=dict(fitted.metadata),
        )
    assert not target.exists()


def test_disjoint_layer_top_up_preserves_existing_row_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replacing the row safetensors carries forward unselected layers."""
    from safetensors import safe_open
    from saklas.io.paths import model_dir

    folder_a = _author_manifold(tmp_path / "a")
    folder_b = _author_manifold(tmp_path / "b")
    handle = _Handle()
    ManifoldExtractionPipeline(handle, EventBus()).fit(
        folder_a, layer_indices=[0],
    )
    from saklas.core import extraction as extraction_module

    capture_dir = model_dir("stub-model") / "manifold_capture"
    layer_zero_row, = capture_dir.glob("*.rows.layer_0.safetensors")
    layer_zero_before = layer_zero_row.stat()
    scopes: list[list[int]] = []
    persisted: list[str] = []
    real_save = extraction_module._save_safetensors_atomic
    real_row_save = extraction_module._save_row_safetensors_atomic

    def _counting(*args: Any, **kwargs: Any) -> Any:
        scopes.append(list(kwargs["layer_indices"]))
        return _stub_encoder_batch(*args, **kwargs)

    def _tracked_save(tensors: Any, path: Path, **kwargs: Any) -> Any:
        persisted.append(Path(path).name)
        return real_save(tensors, path, **kwargs)

    def _tracked_row_save(key: str, tensor: torch.Tensor, path: Path) -> str:
        persisted.append(Path(path).name)
        return real_row_save(key, tensor, path)

    monkeypatch.setattr(V, "_encode_and_capture_all_batch", _counting)
    monkeypatch.setattr(extraction_module, "_save_safetensors_atomic", _tracked_save)
    monkeypatch.setattr(
        extraction_module, "_save_row_safetensors_atomic", _tracked_row_save,
    )
    ManifoldExtractionPipeline(handle, EventBus()).fit(
        folder_b, layer_indices=[1],
    )

    meta_path, = capture_dir.glob("*.json")
    meta = json.loads(meta_path.read_text())
    assert scopes == [[1]]
    assert meta["centroid_layers"] == [0, 1]
    assert meta["row_layers"] == [0, 1]
    assert sorted(persisted) == sorted([
        next(capture_dir.glob("*.centroids.layer_1.safetensors")).name,
        next(capture_dir.glob("*.rows.layer_1.safetensors")).name,
    ])
    layer_zero_after = layer_zero_row.stat()
    assert (layer_zero_after.st_ino, layer_zero_after.st_mtime_ns) == (
        layer_zero_before.st_ino, layer_zero_before.st_mtime_ns,
    )
    row_paths = sorted(capture_dir.glob("*.rows.layer_*.safetensors"))
    assert len(row_paths) == 2
    seen_keys: set[str] = set()
    for row_path in row_paths:
        with safe_open(str(row_path), framework="pt", device="cpu") as tensors:
            keys = set(tensors.keys())
            assert len(keys) == 1
            seen_keys.update(keys)
    assert seen_keys == {"layer_0", "layer_1"}


def test_capture_generation_pointer_failure_recovers_without_recapture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A complete crash-left top-up is adopted over the prior good pointer."""
    from saklas.core import extraction as extraction_module
    from saklas.io import atomic
    from saklas.io.paths import model_dir

    folder_a = _author_manifold(tmp_path / "a")
    folder_b = _author_manifold(tmp_path / "b")
    handle = _Handle()
    ManifoldExtractionPipeline(handle, EventBus()).fit(
        folder_a, layer_indices=[0],
    )
    capture_dir = model_dir("stub-model") / "manifold_capture"
    pointer, = [
        path for path in capture_dir.glob("*.json")
        if ".pending-" not in path.name
    ]
    prior_pointer = pointer.read_bytes()
    prior_payload = json.loads(prior_pointer)
    prior_layer_zero = prior_payload["centroid_files"]["0"]

    real_write_json = atomic.write_json_atomic
    real_publish = extraction_module._publish_fit_if_current
    pointer_failed = False

    def fail_pointer(path: Path, payload: Any, **kwargs: Any) -> None:
        nonlocal pointer_failed
        if Path(path) == pointer and not pointer_failed:
            pointer_failed = True
            raise OSError("injected capture pointer failure")
        real_write_json(path, payload, **kwargs)

    def simulate_crash(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("simulated crash before fitted publication")

    scopes: list[list[int]] = []

    def count_capture(*args: Any, **kwargs: Any) -> Any:
        scopes.append(list(kwargs["layer_indices"]))
        return _stub_encoder_batch(*args, **kwargs)

    monkeypatch.setattr(atomic, "write_json_atomic", fail_pointer)
    monkeypatch.setattr(extraction_module, "_publish_fit_if_current", simulate_crash)
    monkeypatch.setattr(V, "_encode_and_capture_all_batch", count_capture)
    with pytest.raises(RuntimeError, match="simulated crash"):
        ManifoldExtractionPipeline(handle, EventBus()).fit(
            folder_b, layer_indices=[1],
        )

    assert scopes == [[1]]
    assert pointer.read_bytes() == prior_pointer
    assert (capture_dir / prior_layer_zero).is_file()
    pending = list(capture_dir.glob("*.pending-*.json"))
    assert len(pending) == 1
    assert list(capture_dir.glob("*.gen-*.centroids.layer_1.safetensors"))

    monkeypatch.setattr(atomic, "write_json_atomic", real_write_json)
    monkeypatch.setattr(extraction_module, "_publish_fit_if_current", real_publish)
    monkeypatch.setattr(
        V, "_encode_and_capture_all_batch",
        lambda *_args, **_kwargs: pytest.fail(
            "complete orphan generation was not adopted"
        ),
    )
    fitted = ManifoldExtractionPipeline(handle, EventBus()).fit(
        folder_b, layer_indices=[1],
    )

    assert sorted(fitted.layers) == [1]
    recovered = json.loads(pointer.read_text())
    assert recovered["centroid_layers"] == [0, 1]
    assert recovered["row_layers"] == [0, 1]
    assert recovered["centroid_files"]["0"] == prior_layer_zero
    assert not list(capture_dir.glob("*.pending-*.json"))


def test_capture_post_pointer_exception_is_cleaned_on_fitted_cache_hit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A live pointer plus old-parent journal is recognized as committed."""
    from saklas.io import atomic
    from saklas.io.paths import model_dir

    folder = _author_manifold(tmp_path)
    handle = _Handle()
    ManifoldExtractionPipeline(handle, EventBus()).fit(
        folder, layer_indices=[0],
    )
    capture_dir = model_dir("stub-model") / "manifold_capture"
    pointer, = [
        path for path in capture_dir.glob("*.json")
        if ".pending-" not in path.name
    ]
    prior = json.loads(pointer.read_text())
    prior_files = {
        *prior["centroid_files"].values(), *prior["row_files"].values(),
    }
    real_write = atomic.write_json_atomic
    failed = False

    def write_then_fail(path: Path, payload: Any, **kwargs: Any) -> None:
        nonlocal failed
        real_write(path, payload, **kwargs)
        if Path(path) == pointer and not failed:
            failed = True
            raise OSError("after capture pointer replace")

    monkeypatch.setattr(atomic, "write_json_atomic", write_then_fail)
    fitted = ManifoldExtractionPipeline(handle, EventBus()).fit(
        folder, layer_indices=[0], force=True,
    )
    assert sorted(fitted.layers) == [0]
    assert list(capture_dir.glob("*.pending-*.json"))
    assert any((capture_dir / name).exists() for name in prior_files)

    monkeypatch.setattr(atomic, "write_json_atomic", real_write)
    monkeypatch.setattr(
        V, "_encode_and_capture_all_batch",
        lambda *_args, **_kwargs: pytest.fail("fitted cache hit recaptured"),
    )
    cached = ManifoldExtractionPipeline(handle, EventBus()).fit(
        folder, layer_indices=[0],
    )

    assert sorted(cached.layers) == [0]
    assert not list(capture_dir.glob("*.pending-*.json"))
    assert not any((capture_dir / name).exists() for name in prior_files)


def test_capture_generation_fsyncs_before_pointer_and_gc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Payload entries and recovery journal are durable before pointer GC."""
    from saklas.io import atomic
    from saklas.io.paths import model_dir

    folder = _author_manifold(tmp_path)
    capture_dir = model_dir("stub-model") / "manifold_capture"
    events: list[str] = []
    real_write_json = atomic.write_json_atomic
    real_fsync_directory = atomic.fsync_directory

    def track_write(path: Path, payload: Any, **kwargs: Any) -> None:
        resolved = Path(path)
        if resolved.parent == capture_dir:
            events.append(
                "write-pending" if ".pending-" in resolved.name else "write-pointer"
            )
        real_write_json(resolved, payload, **kwargs)

    def track_fsync(path: Path) -> None:
        if Path(path) == capture_dir:
            events.append("fsync-dir")
        real_fsync_directory(path)

    monkeypatch.setattr(atomic, "write_json_atomic", track_write)
    monkeypatch.setattr(atomic, "fsync_directory", track_fsync)
    ManifoldExtractionPipeline(_Handle(), EventBus()).fit(folder)

    assert events[:5] == [
        "fsync-dir", "write-pending", "fsync-dir", "write-pointer", "fsync-dir",
    ]
    assert events[-1] == "fsync-dir"  # generation GC durability barrier


def test_capture_shard_fsync_precedes_atomic_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from saklas.core import extraction as extraction_module

    events: list[str] = []
    real_fsync = extraction_module.os.fsync
    real_replace = extraction_module.os.replace

    def track_fsync(fd: int) -> None:
        events.append("fsync-file")
        real_fsync(fd)

    def track_replace(src: Any, dst: Any) -> None:
        events.append("replace")
        real_replace(src, dst)

    monkeypatch.setattr(extraction_module.os, "fsync", track_fsync)
    monkeypatch.setattr(extraction_module.os, "replace", track_replace)
    extraction_module._save_safetensors_atomic(
        {"layer_0": torch.ones(2, 3)}, tmp_path / "shard.safetensors",
    )

    assert events == ["fsync-file", "replace"]


def test_concurrent_deferred_row_topups_merge_latest_pointer(
    tmp_path: Path,
) -> None:
    from saklas.core.extraction import _publish_deferred_row_shards
    from saklas.core.manifold import ActivationRowStore

    stem = "c" * 64
    meta_path = tmp_path / f"{stem}.json"
    meta_path.write_text(json.dumps({
        "format_version": 4,
        "capture_sha256": stem,
        "node_sizes": [2],
        "centroid_layers": [0, 1],
        "row_layers": [],
        "centroid_files": {},
        "row_files": {},
        "centroid_shapes": {},
        "row_shapes": {},
        "row_dtypes": {},
        "row_tensor_sha256": {},
        "files": {},
    }))
    stores: list[ActivationRowStore] = []
    for layer in (0, 1):
        store = ActivationRowStore([2])
        store.write(
            layer, torch.arange(2),
            torch.full((2, 3), float(layer + 1)),
        )
        stores.append(store)
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def _topup(store: ActivationRowStore) -> None:
        try:
            barrier.wait(timeout=5)
            _publish_deferred_row_shards(
                tmp_path, stem, meta_path, store, [2],
                set(), {}, {}, {}, {},
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    threads = [threading.Thread(target=_topup, args=(store,)) for store in stores]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    for store in stores:
        store.close()

    assert errors == []
    payload = json.loads(meta_path.read_text())
    assert payload["row_layers"] == [0, 1]
    assert set(payload["row_files"]) == {"0", "1"}
    assert all((tmp_path / name).exists() for name in payload["row_files"].values())


def test_v3_monolith_is_replaced_safely_by_v4_layer_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from safetensors.torch import load_file, save_file
    from saklas.io.packs import hash_file
    from saklas.io.paths import model_dir

    folder = _author_manifold(tmp_path)
    pipe = ManifoldExtractionPipeline(_Handle(), EventBus())
    pipe.fit(folder)
    capture_dir = model_dir("stub-model") / "manifold_capture"
    meta_path, = capture_dir.glob("*.json")
    meta = json.loads(meta_path.read_text())
    stem = meta_path.stem

    centroid_tensors: dict[str, torch.Tensor] = {}
    row_tensors: dict[str, torch.Tensor] = {}
    for idx in meta["centroid_layers"]:
        shard = capture_dir / meta["centroid_files"][str(idx)]
        centroid_tensors.update(load_file(str(shard), device="cpu"))
    for idx in meta["row_layers"]:
        shard = capture_dir / meta["row_files"][str(idx)]
        row_tensors.update(load_file(str(shard), device="cpu"))
    centroid_monolith = capture_dir / f"{stem}.centroids.safetensors"
    row_monolith = capture_dir / f"{stem}.rows.safetensors"
    save_file(centroid_tensors, str(centroid_monolith))
    save_file(row_tensors, str(row_monolith))
    for shard in capture_dir.glob(f"{stem}.*.layer_*.safetensors"):
        shard.unlink()
    meta["format_version"] = 3
    meta.pop("centroid_files")
    meta.pop("row_files")
    meta["files"] = {centroid_monolith.name: hash_file(centroid_monolith)}
    meta_path.write_text(json.dumps(meta))

    # Miss the fitted tensor while preserving the exact token capture identity.
    manifest = json.loads((folder / "manifold.json").read_text())
    manifest["nodes"][0]["coords"] = [0.125]
    (folder / "manifold.json").write_text(json.dumps(manifest))
    scopes: list[list[int]] = []

    def _counting(*args: Any, **kwargs: Any) -> Any:
        scopes.append(list(kwargs["layer_indices"]))
        return _stub_encoder_batch(*args, **kwargs)

    monkeypatch.setattr(V, "_encode_and_capture_all_batch", _counting)
    pipe.fit(folder, layer_indices=[1])

    migrated = json.loads(meta_path.read_text())
    assert scopes == [[1]]
    assert migrated["format_version"] == 4
    assert migrated["centroid_layers"] == [1]
    assert migrated["row_layers"] == [1]
    assert not centroid_monolith.exists()
    assert not row_monolith.exists()


def test_row_cache_uses_layer_digests_without_container_rehash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from saklas.io import packs
    from saklas.io.paths import model_dir

    real_hash_file = packs.hash_file
    hashed: list[Path] = []

    def _tracked_hash(path: Path) -> str:
        resolved = Path(path)
        hashed.append(resolved)
        if ".rows.layer_" in resolved.name:
            raise AssertionError("row shard was redundantly whole-file hashed")
        return real_hash_file(resolved)

    monkeypatch.setattr(packs, "hash_file", _tracked_hash)
    folder = _author_manifold(tmp_path)
    pipe = ManifoldExtractionPipeline(_Handle(), EventBus())
    pipe.fit(folder)

    # Force a geometry refit with identical token rows, exercising the cache
    # read path as well as publication.
    manifest = json.loads((folder / "manifold.json").read_text())
    manifest["nodes"][0]["coords"] = [0.125]
    (folder / "manifold.json").write_text(json.dumps(manifest))
    pipe.fit(folder)

    capture_dir = model_dir("stub-model") / "manifold_capture"
    meta_path, = capture_dir.glob("*.json")
    meta = json.loads(meta_path.read_text())
    assert meta["format_version"] == 4
    assert set(meta["row_tensor_sha256"]) == {"0", "1", "2", "3"}
    assert set(meta["files"]) == {
        path.name for path in capture_dir.glob("*.centroids.layer_*.safetensors")
    }
    assert any(".centroids.layer_" in path.name for path in hashed)
    assert not any(".rows.layer_" in path.name for path in hashed)


def test_cold_row_publication_traverses_each_payload_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from saklas.core import extraction as extraction_module
    from saklas.io.paths import model_dir

    real_chunks = extraction_module._iter_tensor_byte_chunks
    traversals: list[int] = []

    def _tracked(tensor: torch.Tensor, **kwargs: Any) -> Any:
        traversals.append(id(tensor))
        yield from real_chunks(tensor, **kwargs)

    monkeypatch.setattr(extraction_module, "_iter_tensor_byte_chunks", _tracked)
    monkeypatch.setattr(
        extraction_module, "_tensor_sha256",
        lambda _tensor: pytest.fail(
            "row publication must not start a second tensor traversal"
        ),
    )
    folder = _author_manifold(tmp_path)
    ManifoldExtractionPipeline(_Handle(), EventBus()).fit(folder)

    capture_dir = model_dir("stub-model") / "manifold_capture"
    row_shards = list(capture_dir.glob("*.rows.layer_*.safetensors"))
    assert len(row_shards) == _N_LAYERS
    assert len(traversals) == len(row_shards)


def test_deferred_row_publication_traverses_each_payload_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from safetensors.torch import load_file
    from saklas.core import extraction as extraction_module
    from saklas.core.manifold import ActivationRowStore

    stem = "d" * 64
    meta_path = tmp_path / f"{stem}.json"
    meta_path.write_text(json.dumps({
        "format_version": 4,
        "capture_sha256": stem,
        "node_sizes": [2],
        "centroid_layers": [0],
        "row_layers": [],
        "centroid_files": {},
        "row_files": {},
        "centroid_shapes": {},
        "row_shapes": {},
        "row_dtypes": {},
        "row_tensor_sha256": {},
        "files": {},
    }))
    expected = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    store = ActivationRowStore([2])
    store.write(0, torch.arange(2), expected)
    real_chunks = extraction_module._iter_tensor_byte_chunks
    traversals = 0

    def _tracked(tensor: torch.Tensor, **kwargs: Any) -> Any:
        nonlocal traversals
        traversals += 1
        yield from real_chunks(tensor, **kwargs)

    monkeypatch.setattr(extraction_module, "_iter_tensor_byte_chunks", _tracked)
    real_tensor_sha = extraction_module._tensor_sha256
    monkeypatch.setattr(
        extraction_module, "_tensor_sha256",
        lambda _tensor: pytest.fail(
            "row publication must not start a second tensor traversal"
        ),
    )
    try:
        extraction_module._publish_deferred_row_shards(
            tmp_path, stem, meta_path, store, [2], set(), {}, {}, {}, {},
        )
    finally:
        store.close()
    monkeypatch.setattr(extraction_module, "_tensor_sha256", real_tensor_sha)

    payload = json.loads(meta_path.read_text())
    row_path = tmp_path / payload["row_files"]["0"]
    assert traversals == 1
    assert real_tensor_sha(expected) == (
        payload["row_tensor_sha256"]["0"]
    )
    assert torch.equal(load_file(str(row_path))["layer_0"], expected)


def test_selected_row_digest_tamper_recaptures_that_layer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from safetensors.torch import load_file, save_file
    from saklas.io.paths import model_dir

    folder = _author_manifold(tmp_path)
    pipe = ManifoldExtractionPipeline(_Handle(), EventBus())
    pipe.fit(folder)
    capture_dir = model_dir("stub-model") / "manifold_capture"
    row_path, = capture_dir.glob("*.rows.layer_1.safetensors")
    tensors = load_file(str(row_path), device="cpu")
    tensors["layer_1"] = tensors["layer_1"].clone()
    tensors["layer_1"][0, 0] += 1.0
    save_file(tensors, str(row_path))
    manifest = json.loads((folder / "manifold.json").read_text())
    manifest["nodes"][0]["coords"] = [0.125]
    (folder / "manifold.json").write_text(json.dumps(manifest))
    scopes: list[list[int]] = []

    def _counting(*args: Any, **kwargs: Any) -> Any:
        scopes.append(list(kwargs["layer_indices"]))
        return _stub_encoder_batch(*args, **kwargs)

    monkeypatch.setattr(V, "_encode_and_capture_all_batch", _counting)
    pipe.fit(folder)
    assert scopes == [[1]]


def test_baseline_prompt_change_invalidates_final_tensor_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    folder = _author_manifold(tmp_path)
    pipe = ManifoldExtractionPipeline(_Handle(), EventBus())
    pipe.fit(folder)
    monkeypatch.setattr(V, "_load_baseline_prompts", lambda: ["changed baseline"])
    calls = 0

    def _counting(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return _stub_encoder_batch(*args, **kwargs)

    monkeypatch.setattr(V, "_encode_and_capture_all_batch", _counting)
    pipe.fit(folder)
    assert calls > 0


def test_node_label_rename_invalidates_fitted_artifact(tmp_path: Path) -> None:
    folder = _author_manifold(tmp_path)
    pipe = ManifoldExtractionPipeline(_Handle(), EventBus())
    first = pipe.fit(folder)
    data = json.loads((folder / "manifold.json").read_text())
    data["nodes"][0]["label"] = "serene"
    (folder / "manifold.json").write_text(json.dumps(data))
    (folder / "nodes" / "00_calm.json").rename(
        folder / "nodes" / "00_serene.json",
    )
    renamed = pipe.fit(folder)
    assert first.node_labels[0] == "calm"
    assert renamed.node_labels[0] == "serene"


def test_capture_does_not_retry_a_known_bad_batch_width(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels = [f"node{i}" for i in range(20)]
    folder = _author_manifold(tmp_path, labels=labels)
    widths: list[int] = []

    def _bounded(*args: Any, **kwargs: Any) -> Any:
        responses = args[3]
        widths.append(len(responses))
        if len(responses) >= 16:
            raise RuntimeError("synthetic out of memory")
        return _stub_encoder_batch(*args, **kwargs)

    monkeypatch.setattr(V, "_encode_and_capture_all_batch", _bounded)
    ManifoldExtractionPipeline(_Handle(), EventBus()).fit(folder)
    assert widths.count(16) == 1


def test_fit_returns_node_roles_without_reload(tmp_path: Path) -> None:
    from types import SimpleNamespace
    from saklas.core.manifold import load_manifold
    from saklas.io.paths import tensor_filename

    folder = _author_manifold(tmp_path)
    meta = json.loads((folder / "manifold.json").read_text())
    for node in meta["nodes"]:
        node["role"] = node["label"]
    (folder / "manifold.json").write_text(json.dumps(meta))

    handle = _Handle()
    handle.model.config = SimpleNamespace(model_type="gemma2")  # type: ignore[attr-defined]
    manifold = ManifoldExtractionPipeline(handle, EventBus()).fit(folder)

    assert manifold.node_roles == _LABELS
    assert manifold.nearest_node_role(_LABELS[0]) == _LABELS[0]

    loaded = load_manifold(folder / tensor_filename(handle.model_id))
    assert loaded.node_roles == _LABELS


def test_fit_writes_tensor_and_manifest(tmp_path: Path) -> None:
    folder = _author_manifold(tmp_path)
    ManifoldExtractionPipeline(_Handle(), EventBus()).fit(folder)
    assert (folder / "stub-model.safetensors").exists()
    assert (folder / "stub-model.json").exists()
    mf = ManifoldFolder.load(folder)
    assert "stub-model.safetensors" in mf.files
    assert "stub-model.json" in mf.files


def test_fit_emits_event(tmp_path: Path) -> None:
    folder = _author_manifold(tmp_path)
    events = EventBus()
    seen = []
    events.subscribe(lambda e: seen.append(e)
                     if isinstance(e, ManifoldExtracted) else None)
    ManifoldExtractionPipeline(_Handle(), events).fit(folder)
    assert len(seen) == 1
    assert seen[0].name == "mood"


def test_fit_cache_hit_skips_forward_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    folder = _author_manifold(tmp_path)
    pipe = ManifoldExtractionPipeline(_Handle(), EventBus())
    pipe.fit(folder)

    calls = {"n": 0}
    real = _stub_encoder_batch

    def _counting(*args: Any, **kwargs: Any) -> dict[int, torch.Tensor]:
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(V, "_encode_and_capture_all_batch", _counting)
    manifold = pipe.fit(folder)  # second call — corpus unchanged
    assert calls["n"] == 0       # cache hit, no pooling
    assert manifold.name == "mood"


def test_fitted_cache_hit_ignores_capture_journal_scan_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    folder = _author_manifold(tmp_path)
    pipe = ManifoldExtractionPipeline(_Handle(), EventBus())
    first = pipe.fit(folder)
    real_glob = Path.glob

    def fail_pending_scan(path: Path, pattern: str):
        if path.name == "manifold_capture" and ".pending-" in pattern:
            raise OSError("transient capture directory read")
        return real_glob(path, pattern)

    monkeypatch.setattr(Path, "glob", fail_pending_scan)
    monkeypatch.setattr(
        V, "_encode_and_capture_all_batch",
        lambda *_args, **_kwargs: pytest.fail("fitted cache hit captured"),
    )

    second = pipe.fit(folder)

    assert torch.allclose(first.node_coords, second.node_coords)


def test_missing_whitener_fails_before_manifold_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from saklas.core.mahalanobis import WhitenerError

    folder = _author_manifold(tmp_path)
    handle = _Handle()
    setattr(handle, "whitener", None)
    monkeypatch.setattr(
        V, "_encode_and_capture_all_batch",
        lambda *_args, **_kwargs: pytest.fail(
            "mandatory-whitener failure ran manifold capture"
        ),
    )

    with pytest.raises(WhitenerError, match="must cover every fit layer"):
        ManifoldExtractionPipeline(handle, EventBus()).fit(folder)


def test_fitted_tensor_hit_does_not_resolve_whitener(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    folder = _author_manifold(tmp_path)
    handle = _Handle()
    pipe = ManifoldExtractionPipeline(handle, EventBus())
    first = pipe.fit(folder)
    setattr(handle, "whitener", None)
    monkeypatch.setattr(
        V, "_encode_and_capture_all_batch",
        lambda *_args, **_kwargs: pytest.fail("fitted tensor hit captured"),
    )

    second = pipe.fit(folder)

    assert torch.allclose(first.node_coords, second.node_coords)


def test_fit_rebuilds_tampered_requested_tensor_from_capture_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from saklas.io.paths import tensor_filename

    folder = _author_manifold(tmp_path)
    pipe = ManifoldExtractionPipeline(_Handle(), EventBus())
    pipe.fit(folder)
    tensor = folder / tensor_filename("stub-model")
    original = tensor.read_bytes()
    damaged = bytearray(original)
    damaged[-1] ^= 1
    tensor.write_bytes(damaged)

    def _must_not_capture(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("artifact repair should reuse activation capture")

    monkeypatch.setattr(V, "_encode_and_capture_all_batch", _must_not_capture)
    pipe.fit(folder)
    assert tensor.read_bytes() == original


def test_capture_cache_identity_includes_node_partition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels = ["calm", "uneasy", "afraid"]
    folder = _author_manifold(tmp_path, labels=labels)
    flat_rows = [f"row{i} statement" for i in range(6)]
    cursor = 0
    for idx, size in enumerate([1, 3, 2]):
        (folder / "nodes" / f"{idx:02d}_{labels[idx]}.json").write_text(
            json.dumps(flat_rows[cursor:cursor + size])
        )
        cursor += size
    pipe = ManifoldExtractionPipeline(_Handle(), EventBus())
    pipe.fit(folder)

    cursor = 0
    for idx, size in enumerate([2, 2, 2]):
        (folder / "nodes" / f"{idx:02d}_{labels[idx]}.json").write_text(
            json.dumps(flat_rows[cursor:cursor + size])
        )
        cursor += size
    calls = {"n": 0}

    def _counting(*args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        return _stub_encoder_batch(*args, **kwargs)

    monkeypatch.setattr(V, "_encode_and_capture_all_batch", _counting)
    pipe.fit(folder)
    assert calls["n"] > 0


def test_capture_cache_prunes_old_groups_but_keeps_current(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from saklas.core.extraction import _prune_manifold_capture_cache

    old_stem = "a" * 64
    keep_stem = "b" * 64
    old = tmp_path / f"{old_stem}.centroids.safetensors"
    keep = tmp_path / f"{keep_stem}.centroids.safetensors"
    old.write_bytes(b"old-cache")
    keep.write_bytes(b"current-cache")
    monkeypatch.setenv("SAKLAS_MANIFOLD_CAPTURE_CACHE_GB", "0.000000001")
    _prune_manifold_capture_cache(tmp_path, keep_stem=keep_stem)
    assert not old.exists()
    assert keep.exists()


def test_capture_cache_prune_waits_for_active_victim_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Eviction never deletes a foreign stem while its fit lock is held."""
    from saklas.core.extraction import _prune_manifold_capture_cache
    from saklas.io import atomic

    victim_stem = "a" * 64
    keep_stem = "b" * 64
    victim = tmp_path / f"{victim_stem}.rows.layer_0.safetensors"
    keep = tmp_path / f"{keep_stem}.centroids.layer_0.safetensors"
    victim.write_bytes(b"old-cache")
    keep.write_bytes(b"current-cache")
    monkeypatch.setenv("SAKLAS_MANIFOLD_CAPTURE_CACHE_GB", "0.000000001")

    real_lock = atomic.artifact_lock
    victim_waiting = threading.Event()

    @contextmanager
    def _tracked_lock(path: Path):
        if Path(path).name == victim_stem:
            victim_waiting.set()
        with real_lock(path):
            yield

    monkeypatch.setattr(atomic, "artifact_lock", _tracked_lock)
    errors: list[BaseException] = []

    def _run_prune() -> None:
        try:
            _prune_manifold_capture_cache(tmp_path, keep_stem=keep_stem)
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    with real_lock(tmp_path / victim_stem):
        thread = threading.Thread(target=_run_prune)
        thread.start()
        assert victim_waiting.wait(timeout=5)
        assert victim.exists()
        assert thread.is_alive()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert errors == []
    assert not victim.exists()
    assert keep.exists()


def test_capture_process_lease_survives_payload_publish_and_blocks_prune(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from saklas.core.extraction import (
        _capture_group_paths,
        _prune_manifold_capture_cache,
    )
    from saklas.io.atomic import (
        artifact_has_live_lease,
        artifact_process_lease,
    )

    victim_stem = "a" * 64
    keep_stem = "b" * 64
    victim = tmp_path / f"{victim_stem}.rows.layer_0.safetensors"
    keep = tmp_path / f"{keep_stem}.centroids.layer_0.safetensors"
    victim.write_bytes(b"leased")
    keep.write_bytes(b"current")
    monkeypatch.setenv("SAKLAS_MANIFOLD_CAPTURE_CACHE_GB", "0.000000001")

    with artifact_process_lease(tmp_path / victim_stem):
        assert artifact_has_live_lease(tmp_path / victim_stem)
        assert all(".lease." not in path.name for path in _capture_group_paths(
            tmp_path, victim_stem,
        ))
        _prune_manifold_capture_cache(tmp_path, keep_stem=keep_stem)
        assert victim.exists()
    assert not artifact_has_live_lease(tmp_path / victim_stem)
    _prune_manifold_capture_cache(tmp_path, keep_stem=keep_stem)
    assert not victim.exists()


def test_capture_lease_enter_failure_releases_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from saklas.io import atomic

    folder = _author_manifold(tmp_path)
    real_lock_class = atomic.ReleasableArtifactLock
    paths: list[Path] = []

    def _track(path: Path) -> atomic.ReleasableArtifactLock:
        paths.append(Path(path))
        return real_lock_class(path)

    class _BrokenLease:
        def __enter__(self) -> None:
            raise OSError("injected lease marker failure")

        def __exit__(self, *_args: object) -> None:
            raise AssertionError("unentered lease must not exit")

    monkeypatch.setattr(atomic, "ReleasableArtifactLock", _track)
    monkeypatch.setattr(atomic, "artifact_process_lease", lambda _path: _BrokenLease())
    with pytest.raises(OSError, match="lease marker"):
        ManifoldExtractionPipeline(_Handle(), EventBus()).fit(folder)
    acquired = threading.Event()

    def _acquire() -> None:
        with atomic.artifact_lock(paths[0]):
            acquired.set()

    thread = threading.Thread(target=_acquire)
    thread.start()
    assert acquired.wait(timeout=5)
    thread.join(timeout=5)


def test_capture_lease_marker_removed_while_transaction_held(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from saklas.io import atomic

    folder = _author_manifold(tmp_path)
    real_lock_class = atomic.ReleasableArtifactLock
    active: list[Any] = []
    exit_states: list[bool] = []

    class _TrackedLock:
        def __init__(self, path: Path) -> None:
            self.inner = real_lock_class(path)
            self.held = False
            active.append(self)

        def acquire(self) -> Any:
            self.inner.acquire()
            self.held = True
            return self

        def release(self) -> None:
            self.inner.release()
            self.held = False

    class _Lease:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *_args: object) -> None:
            exit_states.append(bool(active[-1].held))

    monkeypatch.setattr(atomic, "ReleasableArtifactLock", _TrackedLock)
    monkeypatch.setattr(atomic, "artifact_process_lease", lambda _path: _Lease())
    ManifoldExtractionPipeline(_Handle(), EventBus()).fit(folder)
    assert exit_states == [True]
    assert active[-1].held is False


def test_tampered_activation_cache_recaptures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from saklas.io.paths import model_dir

    folder = _author_manifold(tmp_path)
    pipe = ManifoldExtractionPipeline(_Handle(), EventBus())
    pipe.fit(folder)
    cache_files = list(
        (model_dir("stub-model") / "manifold_capture").glob(
            "*.centroids.layer_*.safetensors"
        )
    )
    assert len(cache_files) == _N_LAYERS
    payload = bytearray(cache_files[0].read_bytes())
    payload[-1] ^= 1
    cache_files[0].write_bytes(payload)
    manifest = json.loads((folder / "manifold.json").read_text())
    manifest["nodes"][0]["coords"] = [0.125]
    (folder / "manifold.json").write_text(json.dumps(manifest))
    calls = {"n": 0}

    def _counting(*args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        return _stub_encoder_batch(*args, **kwargs)

    monkeypatch.setattr(V, "_encode_and_capture_all_batch", _counting)
    pipe.fit(folder)
    assert calls["n"] > 0


def test_fit_force_bypasses_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``force=True`` re-pools/re-fits even when the cache is valid.

    The discover/multi-node analogue of ``manifold extract -f``: with the corpus
    unchanged, a plain refit cache-hits (previous test), so a code-level fit
    change (e.g. the topology-selection dim-match fix) can only be picked up via
    ``force``.
    """
    folder = _author_manifold(tmp_path)
    pipe = ManifoldExtractionPipeline(_Handle(), EventBus())
    pipe.fit(folder)

    calls = {"n": 0}
    real = _stub_encoder_batch

    def _counting(*args: Any, **kwargs: Any) -> dict[int, torch.Tensor]:
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(V, "_encode_and_capture_all_batch", _counting)
    manifold = pipe.fit(folder, force=True)  # corpus unchanged, but forced
    assert calls["n"] > 0        # cache bypassed, forward passes re-run
    assert manifold.name == "mood"


def test_forced_subset_preserves_unselected_capture_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from saklas.io.paths import model_dir

    folder = _author_manifold(tmp_path)
    pipe = ManifoldExtractionPipeline(_Handle(), EventBus())
    pipe.fit(folder)
    capture_dir = model_dir("stub-model") / "manifold_capture"
    before = {
        path.name: (path.stat().st_ino, path.stat().st_mtime_ns)
        for path in capture_dir.glob("*.layer_*.safetensors")
        if ".layer_1." not in path.name
    }
    scopes: list[list[int]] = []

    def _counting(*args: Any, **kwargs: Any) -> Any:
        scopes.append(list(kwargs["layer_indices"]))
        return _stub_encoder_batch(*args, **kwargs)

    monkeypatch.setattr(V, "_encode_and_capture_all_batch", _counting)
    pipe.fit(folder, layer_indices=[1], force=True)
    assert scopes == [[1]]
    assert before == {
        path.name: (path.stat().st_ino, path.stat().st_mtime_ns)
        for path in capture_dir.glob("*.layer_*.safetensors")
        if ".layer_1." not in path.name
    }

    scopes.clear()
    pipe.fit(folder)
    assert scopes == []


def test_curved_raw_fit_reuses_retained_rows_for_sigma_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Known-curved raw fits should not re-run capture for the sigma field."""
    folder = _author_manifold(tmp_path)
    calls = {"n": 0}
    real = _stub_encoder_batch

    def _counting(*args: Any, **kwargs: Any) -> dict[int, torch.Tensor]:
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(V, "_encode_and_capture_all_batch", _counting)
    manifold = ManifoldExtractionPipeline(_Handle(), EventBus()).fit(folder)

    # Five one-response nodes now share one fit-wide batch; sigma reuses those
    # retained rows rather than adding a second pass.
    assert calls["n"] == 1
    assert all(sub.has_sigma for sub in manifold.layers.values())
    assert "sigma_field_per_layer" in manifold.metadata


def test_fit_cache_miss_on_corpus_change(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    folder = _author_manifold(tmp_path)
    pipe = ManifoldExtractionPipeline(_Handle(), EventBus())
    pipe.fit(folder)

    (folder / "nodes" / "00_calm.json").write_text(
        json.dumps(["calm statement 0", "calm statement 1"])
    )

    calls = {"n": 0}
    real = _stub_encoder_batch

    def _counting(*args: Any, **kwargs: Any) -> dict[int, torch.Tensor]:
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(V, "_encode_and_capture_all_batch", _counting)
    pipe.fit(folder)
    assert calls["n"] > 0  # corpus changed -> forward passes re-run


def test_fit_cache_miss_on_domain_change(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    folder = _author_manifold(tmp_path, periodic=True)
    pipe = ManifoldExtractionPipeline(_Handle(), EventBus())
    pipe.fit(folder)

    # Move a node coordinate — the geometry changes, so the cached
    # tensor is stale even though the node corpus is byte-identical.
    meta = json.loads((folder / "manifold.json").read_text())
    meta["nodes"][1]["coords"] = [0.123]
    (folder / "manifold.json").write_text(json.dumps(meta))

    calls = {"n": 0}
    real = _stub_encoder_batch

    def _counting(*args: Any, **kwargs: Any) -> dict[int, torch.Tensor]:
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(V, "_encode_and_capture_all_batch", _counting)
    pipe.fit(folder)
    # Geometry changed, so the subspace is re-fit, but the token-exact corpus
    # capture cache reuses the same centroids/rows without another model pass.
    assert calls["n"] == 0


def test_fit_sae_variant(tmp_path: Path) -> None:
    folder = _author_manifold(tmp_path)
    sae = MockSaeBackend(
        layers=frozenset(range(_N_LAYERS)), d_model=_DIM,
        release="mock-rel",
    )
    manifold = ManifoldExtractionPipeline(_Handle(), EventBus()).fit(
        folder, sae=sae,
    )
    assert manifold.feature_space == "sae-mock-rel"
    assert sorted(manifold.layers) == list(range(_N_LAYERS))
    assert (folder / "stub-model_sae-mock-rel.safetensors").exists()


def test_multi_node_sae_fit_releases_raw_centroid_layers_during_reconstruction(
    tmp_path: Path,
) -> None:
    """SAE reconstruction does not retain a second raw K x D x L roster."""
    folder = _author_manifold(tmp_path)
    raw_refs: list[tuple[int, weakref.ReferenceType[torch.Tensor]]] = []

    def encode(idx: int, hidden: torch.Tensor) -> torch.Tensor:
        gc.collect()
        assert all(
            ref() is None for prior_idx, ref in raw_refs if prior_idx < idx
        ), "a reconstructed layer retained its raw centroid roster"
        raw_refs.append((idx, weakref.ref(hidden)))
        return hidden

    def decode(_idx: int, features: torch.Tensor) -> torch.Tensor:
        # Force a distinct reconstructed payload; an identity return would
        # intentionally make raw and reconstructed storage the same tensor.
        return features.clone()

    sae = MockSaeBackend(
        layers=frozenset(range(_N_LAYERS)), d_model=_DIM,
        release="memory-rel", fingerprint="memory-rel-fingerprint",
        encode_fn=encode, decode_fn=decode,
    )
    ManifoldExtractionPipeline(_Handle(), EventBus()).fit(folder, sae=sae)

    gc.collect()
    assert len(raw_refs) == _N_LAYERS
    assert all(ref() is None for _idx, ref in raw_refs)


def test_curved_fit_releases_centroid_rosters_before_covariance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The retained-row pass must not overlap the K x D centroid rosters."""
    import saklas.core.manifold as manifold_module

    folder = _author_manifold(tmp_path)
    real_fit = manifold_module.fit_layer_subspace
    real_covariances = manifold_module.compute_store_reduced_covariances
    stack_refs: list[weakref.ReferenceType[torch.Tensor]] = []
    whitened_refs: list[weakref.ReferenceType[torch.Tensor]] = []
    gram_refs: list[weakref.ReferenceType[torch.Tensor]] = []
    covariance_calls = 0

    def track_fit(
        centroids: torch.Tensor, *args: Any, **kwargs: Any,
    ) -> Any:
        stack_refs.append(weakref.ref(centroids))
        whitened_refs.append(weakref.ref(kwargs["whitened_rows"]))
        gram_refs.append(weakref.ref(kwargs["whitened_gram"]))
        return real_fit(centroids, *args, **kwargs)

    def check_lifetime(*args: Any, **kwargs: Any) -> Any:
        nonlocal covariance_calls
        covariance_calls += 1
        gc.collect()
        assert len(stack_refs) == _N_LAYERS
        assert all(ref() is None for ref in stack_refs)
        assert all(ref() is None for ref in whitened_refs)
        assert all(ref() is None for ref in gram_refs)
        return real_covariances(*args, **kwargs)

    monkeypatch.setattr(manifold_module, "fit_layer_subspace", track_fit)
    monkeypatch.setattr(
        manifold_module, "compute_store_reduced_covariances", check_lifetime,
    )

    ManifoldExtractionPipeline(_Handle(), EventBus()).fit(folder)
    assert covariance_calls == 1


def test_fit_sae_cache_hit_resolves_identity_without_loading_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fitted SAE tensor resolves transform identity but loads no weights."""
    folder = _author_manifold(tmp_path)
    pipe = ManifoldExtractionPipeline(_Handle(), EventBus())
    sae = MockSaeBackend(
        layers=frozenset(range(_N_LAYERS)), d_model=_DIM,
        release="mock-rel", revision="rev-1",
    )
    pipe.fit(folder, sae=sae)

    import saklas.core.sae as sae_module

    resolved = 0

    def _metadata_only(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal resolved
        resolved += 1
        return MockSaeBackend(
            layers=frozenset(range(_N_LAYERS)), d_model=_DIM,
            release="mock-rel", revision="rev-1",
            fingerprint=sae.fingerprint,
            encode_fn=lambda *_args: (_ for _ in ()).throw(
                AssertionError("cache hit loaded/used SAE weights")
            ),
            decode_fn=lambda *_args: (_ for _ in ()).throw(
                AssertionError("cache hit loaded/used SAE weights")
            ),
        )

    monkeypatch.setattr(sae_module, "load_sae_backend", _metadata_only)
    cached = pipe.fit(folder, sae="mock-rel", sae_revision="rev-1")
    assert resolved == 1
    assert cached.feature_space == "sae-mock-rel"


def test_default_sae_fit_does_not_accept_partial_layer_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    folder = _author_manifold(tmp_path)
    pipe = ManifoldExtractionPipeline(_Handle(), EventBus())
    partial_backend = MockSaeBackend(
        layers=frozenset(range(_N_LAYERS)), d_model=_DIM,
        release="mock-rel", revision="rev-1",
    )
    pipe.fit(folder, sae=partial_backend, layer_indices=[1, 3])

    import saklas.core.sae as sae_module

    monkeypatch.setattr(
        sae_module, "load_sae_backend",
        lambda *_args, **_kwargs: MockSaeBackend(
            layers=frozenset(range(_N_LAYERS)), d_model=_DIM,
            release="mock-rel", revision="rev-1",
        ),
    )
    captured_scopes: list[list[int]] = []

    def _counting(*args: Any, **kwargs: Any) -> Any:
        captured_scopes.append(list(kwargs["layer_indices"]))
        return _stub_encoder_batch(*args, **kwargs)

    monkeypatch.setattr(V, "_encode_and_capture_all_batch", _counting)
    full = pipe.fit(folder, sae="mock-rel", sae_revision="rev-1")
    assert sorted(full.layers) == list(range(_N_LAYERS))
    assert captured_scopes == [[0, 2]]


def test_fit_sae_variant_captures_only_covered_layers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    folder = _author_manifold(tmp_path)
    sae = MockSaeBackend(
        layers=frozenset({1, 3}), d_model=_DIM,
        release="partial-rel",
    )
    seen: list[tuple[int, ...]] = []

    def _filtered_batch(*args: Any, **kwargs: Any) -> dict[int, torch.Tensor]:
        layer_indices = kwargs.get("layer_indices")
        seen.append(tuple(layer_indices or range(_N_LAYERS)))
        out = _stub_encoder_batch(*args, **kwargs)
        if layer_indices is None:
            return out
        return {idx: out[idx] for idx in layer_indices}

    monkeypatch.setattr(V, "_encode_and_capture_all_batch", _filtered_batch)

    manifold = ManifoldExtractionPipeline(_Handle(), EventBus()).fit(
        folder, sae=sae,
    )

    assert sorted(manifold.layers) == [1, 3]
    assert seen
    assert all(layers == (1, 3) for layers in seen)


def test_fit_sae_no_coverage_raises_before_pooling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An SAE covering none of the model's layers raises ``SaeCoverageError``
    *before* the per-node centroid pooling loop — fail-fast on the
    ``ManifoldExtractionPipeline`` ordering contract."""
    from saklas.core.errors import SaeCoverageError

    folder = _author_manifold(tmp_path)
    # SAE layers disjoint from [0, _N_LAYERS) — zero coverage.
    sae = MockSaeBackend(
        layers=frozenset({_N_LAYERS + 1, _N_LAYERS + 2}), d_model=_DIM,
        release="empty-rel",
    )

    # Pooling must never run on the no-coverage path.
    from saklas.core import manifold as M

    def _explode(*_a: Any, **_k: Any) -> None:
        raise AssertionError(
            "compute_node_centroid called before SAE-coverage check"
        )

    monkeypatch.setattr(M, "compute_node_centroid", _explode)

    with pytest.raises(SaeCoverageError, match="covers no layers"):
        ManifoldExtractionPipeline(_Handle(), EventBus()).fit(folder, sae=sae)


def test_fit_natural_manifold(tmp_path: Path) -> None:
    from saklas.core.manifold import BoxDomain
    folder = _author_manifold(tmp_path, periodic=False)
    manifold = ManifoldExtractionPipeline(_Handle(), EventBus()).fit(folder)
    assert manifold.domain.intrinsic_dim == 1
    # The domain must be a BoxDomain (authored with box spec); narrowing
    # the type lets pyright resolve the .axes attribute.
    assert isinstance(manifold.domain, BoxDomain)
    assert manifold.domain.axes[0].periodic is False
    for sub in manifold.layers.values():
        _np, _, _ = sub.rbf_params()
        assert _np.shape[0] == len(_LABELS)


def test_fit_n2_box_manifold(tmp_path: Path) -> None:
    labels = [f"n{i}" for i in range(9)]
    domain = {
        "type": "box",
        "axes": [
            {"name": "u", "periodic": False, "lo": 0.0, "hi": 1.0},
            {"name": "v", "periodic": False, "lo": 0.0, "hi": 1.0},
        ],
    }
    coords = [[x, y] for x in (0.0, 0.5, 1.0) for y in (0.0, 0.5, 1.0)]
    folder = _author_manifold(
        tmp_path, labels=labels, domain=domain, coords=coords,
    )
    manifold = ManifoldExtractionPipeline(_Handle(), EventBus()).fit(folder)
    assert manifold.domain.intrinsic_dim == 2
    pt = manifold.manifold_point(0, (0.3, 0.7))
    assert pt.shape == (_DIM,)


def test_fit_rejects_poisedness_failure(tmp_path: Path) -> None:
    # Five nodes on a 2-D domain, all collinear -> the affine term is
    # underdetermined and the RBF fit raises.
    labels = [f"n{i}" for i in range(5)]
    domain = {
        "type": "box",
        "axes": [
            {"name": "u", "periodic": False, "lo": 0.0, "hi": 1.0},
            {"name": "v", "periodic": False, "lo": 0.0, "hi": 1.0},
        ],
    }
    coords = [[t, t] for t in (0.0, 0.25, 0.5, 0.75, 1.0)]
    with pytest.warns(UserWarning, match="poised"):
        folder = _author_manifold(
            tmp_path, labels=labels, domain=domain, coords=coords,
        )
        ManifoldFolder.load(folder)
    with pytest.warns(UserWarning, match="poised"), \
            pytest.raises(ValueError, match="poisedness"):
        ManifoldExtractionPipeline(_Handle(), EventBus()).fit(folder)


# ============================================================ discover mode ===

# The stub encoder is deterministic per label, so discover-mode fits are
# also deterministic — every node's centroid is fixed by its label, the
# derived coords come straight out of the PCA/spectral analysis of those
# centroids, and the per-layer RBF fits become repeatable.  No model
# required.


def _discover_folder(
    root: Path,
    *,
    name: str = "personas",
    fit_mode: str = "pca",
    labels: list[str] | None = None,
    hyperparams: dict[str, Any] | None = None,
) -> Path:
    """Hand-author a discover-mode manifold folder without going through
    create_discover_manifold_folder (which writes to ~/.saklas/)."""
    if labels is None:
        labels = ["pirate", "caveman", "scholar", "assistant", "robot"]
    folder = root / name
    (folder / "nodes").mkdir(parents=True)
    for idx, label in enumerate(labels):
        (folder / "nodes" / f"{idx:02d}_{label}.json").write_text(
            json.dumps([f"{label} statement {i}" for i in range(3)])
        )
    (folder / "manifold.json").write_text(json.dumps({
        "format_version": MANIFOLD_FORMAT_VERSION,
        "name": name,
        "description": f"discover-{fit_mode}",
        "fit_mode": fit_mode,
        "hyperparams": hyperparams or {},
        "nodes": [{"label": label} for label in labels],
        "files": {},
    }))
    return folder


def test_discover_pca_produces_custom_domain(tmp_path: Path) -> None:
    """PCA discover fit produces a CustomDomain of the picked intrinsic dim."""
    folder = _discover_folder(
        tmp_path, fit_mode="pca",
        hyperparams={"max_dim": 4, "var_threshold": 0.70},
    )
    manifold = ManifoldExtractionPipeline(_Handle(), EventBus()).fit(folder)

    # CustomDomain with identity embedding — intrinsic_dim == embed_dim.
    from saklas.core.manifold import CustomDomain
    assert isinstance(manifold.domain, CustomDomain)
    assert manifold.domain.intrinsic_dim == manifold.domain.embed_dim
    assert 1 <= manifold.domain.intrinsic_dim <= 4

    # Coords were derived from the activations; node_coords shape lines up.
    assert manifold.node_coords.shape[0] == 5
    assert manifold.node_coords.shape[1] == manifold.domain.intrinsic_dim

    # Per-layer fits cover every layer in the model.
    assert sorted(manifold.layers) == list(range(_N_LAYERS))


def test_discover_pca_produces_affine_subspaces(tmp_path: Path) -> None:
    """PCA discover fits a **flat affine** subspace per layer (no RBF surface)
    — the 4.0 reclassification (ARCHITECTURE §1/§5): a personas-shaped artifact
    is a rank-k flat subspace, not a curved manifold.  Each layer carries the
    real per-layer node coords ``(K, R)``; the basis is whitened/Fisher (the
    stub handle carries a covering synthetic whitener — activation-space fits
    require one post-4.0, there is no Euclidean fit path)."""
    folder = _discover_folder(
        tmp_path, fit_mode="pca",
        hyperparams={"max_dim": 4, "var_threshold": 0.70},
    )
    manifold = ManifoldExtractionPipeline(_Handle(), EventBus()).fit(folder)
    assert manifold.layers  # at least one layer survived DLS
    for sub in manifold.layers.values():
        assert sub.is_affine                          # flat — no RBF surface
        with pytest.raises(ValueError, match="affine"):
            sub.rbf_params()                          # no RBF triple
        assert sub.node_coords is not None            # real per-layer coords
        assert sub.node_coords.shape[0] == 5          # one row per node
        assert sub.node_coords.shape[1] == sub.rank   # (K, R)
    # subspace_metric is always "mahalanobis": the stub carries a covering
    # whitener and there is no Euclidean fit path.
    sidecar = json.loads((folder / "stub-model.json").read_text())
    assert sidecar["subspace_metric"] == "mahalanobis"


def test_discover_pca_layout_is_neutral_centered(tmp_path: Path) -> None:
    """The flat (pca) display layout is re-anchored on neutral instead of the
    node centroid.  ``derive_pca_coords`` returns coords with a zero node mean
    (PCA-centered), so pre-re-anchoring the origin *was* the centroid; after
    re-anchoring the origin is neutral's projection into the layout span (the
    closest layout point to neutral, like the geometry plot's ``neutral_white``),
    so the centroid sits off-origin and a coord-form ``%`` at (0,…,0) pushes less
    than one toward the centroid.  Re-anchoring is a pure translation, so node-
    exact steering is unchanged: a coord-form push at a node's layout coords
    still reproduces that node's label-form push, layer for layer."""
    from saklas.core.session import _affine_manifold_push

    folder = _discover_folder(
        tmp_path, fit_mode="pca",
        hyperparams={"max_dim": 4, "var_threshold": 0.99},
    )
    m = ManifoldExtractionPipeline(_Handle(), EventBus()).fit(folder)
    k = m.domain.intrinsic_dim
    centroid = m.node_coords.mean(dim=0)                    # (k,)

    # (a) origin moved off the node centroid (PCA mean was exactly 0; the
    #     re-anchor shifted it to −neutral_layout_coord).
    assert float(centroid.norm()) > 1e-2

    # (b) origin == neutral's projection ⇒ steering toward (0,…,0) displaces less
    #     than steering toward the node centroid, and less than a full node push.
    push_norm = lambda d: max(float(v.norm()) for v in d.values())  # noqa: E731
    _, at_origin = _affine_manifold_push(m, tuple([0.0] * k))
    _, at_centroid = _affine_manifold_push(m, tuple(centroid.tolist()))
    _, at_node = _affine_manifold_push(m, m.node_labels[0])
    assert push_norm(at_origin) < push_norm(at_centroid)
    assert push_norm(at_origin) < push_norm(at_node)

    # (c) node-exactness preserved by the shift: coord-form at a node's layout
    #     coords reproduces its label-form push, layer for layer.
    for i in range(len(m.node_labels)):
        _, by_label = _affine_manifold_push(m, m.node_labels[i])
        _, by_coord = _affine_manifold_push(m, tuple(m.node_coords[i].tolist()))
        assert set(by_label) == set(by_coord)
        for L in by_label:
            assert torch.allclose(by_label[L], by_coord[L], atol=1e-4)


def test_discover_records_fit_mode_and_diagnostics(tmp_path: Path) -> None:
    """The sidecar carries fit_mode + diagnostics so the inspector can render."""
    folder = _discover_folder(
        tmp_path, fit_mode="pca",
        hyperparams={"max_dim": 4, "var_threshold": 0.70},
    )
    ManifoldExtractionPipeline(_Handle(), EventBus()).fit(folder)

    sidecar = json.loads((folder / "stub-model.json").read_text())
    assert sidecar["fit_mode"] == "pca"
    assert "diagnostics" in sidecar
    diag = sidecar["diagnostics"]
    assert "per_component_variance" in diag
    assert "cumulative_variance" in diag
    assert "picked_k" in diag
    assert diag["picked_k"] >= 1
    assert sidecar["hyperparams"]["max_dim"] == 4
    assert sidecar["hyperparams"]["var_threshold"] == 0.70


def test_discover_cache_hit_skips_forward_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second fit with unchanged inputs short-circuits to the cached tensor."""
    folder = _discover_folder(tmp_path, fit_mode="pca")
    ManifoldExtractionPipeline(_Handle(), EventBus()).fit(folder)

    # Patch the centroid pooler to crash if called — cache hit must skip it.
    def _explode(*_a: Any, **_k: Any) -> None:
        raise AssertionError("compute_node_centroid called on cache hit")
    from saklas.core import manifold as M
    monkeypatch.setattr(M, "compute_node_centroid", _explode)

    manifold = ManifoldExtractionPipeline(_Handle(), EventBus()).fit(folder)
    assert manifold.name == "personas"


def test_discover_cache_invalidates_on_hyperparam_change(tmp_path: Path) -> None:
    """Changing max_dim refits rather than serving the cached tensor."""
    folder = _discover_folder(
        tmp_path, fit_mode="pca", hyperparams={"max_dim": 4},
    )
    m1 = ManifoldExtractionPipeline(_Handle(), EventBus()).fit(folder)

    # Rewrite manifest with a different max_dim.
    data = json.loads((folder / "manifold.json").read_text())
    data["hyperparams"] = {"max_dim": 2}
    (folder / "manifold.json").write_text(json.dumps(data))

    m2 = ManifoldExtractionPipeline(_Handle(), EventBus()).fit(folder)
    # The two fits agree on at most max_dim=2, so the second's intrinsic
    # dim is bounded by 2; if cache had hit we'd still see m1's dim.
    assert m1.domain.intrinsic_dim >= m2.domain.intrinsic_dim
    assert m2.domain.intrinsic_dim <= 2


def test_discover_cache_invalidates_on_fit_mode_change(tmp_path: Path) -> None:
    """Switching pca ↔ spectral forces a refit.

    Uses 9 labels so both fit modes pick a ``k`` satisfying
    ``min_nodes(k) <= 9`` regardless of which one the eigengap
    heuristic picks (worst case k=4 ⇒ min_nodes(4)=9).
    """
    labels = [f"p{i}" for i in range(9)]
    folder = _discover_folder(
        tmp_path, fit_mode="pca", labels=labels,
        hyperparams={"max_dim": 4},
    )
    ManifoldExtractionPipeline(_Handle(), EventBus()).fit(folder)
    sidecar_pca = json.loads((folder / "stub-model.json").read_text())
    assert sidecar_pca["fit_mode"] == "pca"

    data = json.loads((folder / "manifold.json").read_text())
    data["fit_mode"] = "spectral"
    (folder / "manifold.json").write_text(json.dumps(data))

    ManifoldExtractionPipeline(_Handle(), EventBus()).fit(folder)
    sidecar_spec = json.loads((folder / "stub-model.json").read_text())
    assert sidecar_spec["fit_mode"] == "spectral"


def test_discover_round_trip_through_load_manifold(tmp_path: Path) -> None:
    """A fitted discover manifold loads back with the same domain + coords."""
    from saklas.core.manifold import CustomDomain, load_manifold
    folder = _discover_folder(
        tmp_path, fit_mode="pca", hyperparams={"max_dim": 4},
    )
    m1 = ManifoldExtractionPipeline(_Handle(), EventBus()).fit(folder)
    m2 = load_manifold(folder / "stub-model.safetensors")
    assert isinstance(m2.domain, CustomDomain)
    assert m2.domain.intrinsic_dim == m1.domain.intrinsic_dim
    assert m2.node_labels == m1.node_labels
    assert m2.node_coords.shape == m1.node_coords.shape
    assert torch.allclose(m2.node_coords, m1.node_coords, atol=1e-5)
    # Per-layer subspaces round-trip — every layer present, same shapes.
    assert sorted(m2.layers) == sorted(m1.layers)
    for L in m1.layers:
        assert m2.layers[L].mean.shape == m1.layers[L].mean.shape
        assert m2.layers[L].basis.shape == m1.layers[L].basis.shape


def test_discover_subspace_inject_translates_by_target(tmp_path: Path) -> None:
    """End-to-end behavior check: ``subspace_inject`` *translates* the in-subspace
    component by the fixed offset ``along·target``.

    A direct push call leaves ``kappa`` at its scalar-0 default (pure translate),
    so ``along=1`` shifts the projected foot by the full ``target`` offset
    (preserving the per-token spread) rather than snapping it onto ``target``.
    The fit is flat (``fit_mode=pca``) so ``H_n ≡ 0`` and ``onto`` is vacuous; the
    reduced coords land at exactly ``h_in + target`` (the soft norm cap does not
    fire — the offset is small against the far-out hidden).
    """
    from saklas.core.manifold import subspace_inject
    # Seed the global RNG before the fit: the stub encoder perturbs each
    # layer's centroid with a generator-less ``torch.randn``, so without
    # this the fitted subspace jitters with test order.
    torch.manual_seed(0)
    folder = _discover_folder(
        tmp_path, fit_mode="pca", hyperparams={"max_dim": 4},
    )
    manifold = ManifoldExtractionPipeline(_Handle(), EventBus()).fit(folder)

    layer = 0
    sub = manifold.layers[layer]
    domain = manifold.domain
    n = domain.intrinsic_dim
    position = manifold.node_coords[0].to(torch.float32)  # a coord on M (the target)

    # Hidden states far from any natural manifold point so the translate is
    # well-resolved against the soft norm cap.
    g = torch.Generator().manual_seed(0)
    hidden = 3.0 * torch.randn(1, 3, _DIM, generator=g)
    seed = position.reshape((1,) * 2 + (n,)).expand(1, 3, n)
    out, _foot = subspace_inject(
        hidden, sub, domain, position, seed,
        along=1.0, onto=1.0, gn_steps=4,
    )

    for pos in range(hidden.shape[1]):
        h_in = (hidden[0, pos] - sub.mean) @ sub.basis.T
        h_out = (out[0, pos] - sub.mean) @ sub.basis.T
        # along=1, κ=0 ⇒ the in-subspace coord is translated by the full target
        # offset: h_out == h_in + target (the per-token spread h_in is kept).
        assert torch.allclose(h_out, h_in + position, atol=1e-3), (
            f"position {pos}: expected translate h_in + target, "
            f"got Δ={h_out - h_in} vs target={position}"
        )


def test_discover_pca_two_node_is_steering_vector(tmp_path: Path) -> None:
    """A 2-node ``fit_mode=pca`` folder *is* a difference-of-means steering
    vector — the 4.0 unification (ARCHITECTURE §1/§5, "a vector = a 2-node
    folder").  K=2 ⇒ a rank-1 affine subspace; the RBF poisedness floor
    ``min_nodes(1)=3`` does **not** gate the flat path (only ``k+1=2``
    centroids are needed to span a 1-D subspace).  Each layer's two node
    coords straddle the origin — the μ-centered pos/neg contrast is the DiM
    axis itself.
    """
    folder = _discover_folder(
        tmp_path, name="anger", fit_mode="pca",
        labels=["angry", "calm"],
        hyperparams={"max_dim": 4, "var_threshold": 0.70},
    )
    manifold = ManifoldExtractionPipeline(_Handle(), EventBus()).fit(folder)

    assert manifold.domain.intrinsic_dim == 1          # rank-1 == a vector
    assert manifold.node_labels == ["angry", "calm"]
    assert manifold.node_coords.shape == (2, 1)
    assert manifold.layers                             # survived DLS
    for sub in manifold.layers.values():
        assert sub.is_affine                           # flat — no RBF surface
        assert sub.rank == 1
        assert sub.node_coords is not None
        assert sub.node_coords.shape == (2, 1)         # (K, R) = (2, 1)
        # μ-centered ⇒ the angry/calm coords sit on opposite sides of 0:
        # the difference-of-means axis, with neutral implicitly between.
        assert sub.node_coords[0, 0] * sub.node_coords[1, 0] < 0


def test_two_node_pca_reads_as_affine_pole_push(tmp_path: Path) -> None:
    """A fitted 2-node pca manifold reads through the session's
    ``_affine_manifold_push`` — the ``name%pole`` steer path — as a rank-1
    affine push: per-layer basis + the pole node's real coord.  This closes
    the author→fit→steer loop for "a vector = a 2-node folder": steering
    toward the ``angry`` pole and toward the ``calm`` pole read
    opposite-signed coords (the difference-of-means contrast).
    """
    from saklas.core.session import _affine_manifold_push

    folder = _discover_folder(
        tmp_path, name="anger", fit_mode="pca",
        labels=["angry", "calm"],
        hyperparams={"max_dim": 4, "var_threshold": 0.70},
    )
    manifold = ManifoldExtractionPipeline(_Handle(), EventBus()).fit(folder)

    basis_angry, coord_angry = _affine_manifold_push(manifold, "angry")
    assert set(basis_angry) == set(manifold.layers)
    for L, sub in manifold.layers.items():
        assert sub.node_coords is not None
        assert torch.equal(basis_angry[L], sub.basis)        # per-layer basis
        assert torch.equal(coord_angry[L], sub.node_coords[0])  # angry == node 0

    _, coord_calm = _affine_manifold_push(manifold, "calm")
    for L in manifold.layers:
        # Opposite poles slide to opposite sides of the neutral-anchored origin.
        assert coord_angry[L][0] * coord_calm[L][0] < 0


def test_affine_push_coord_form_equals_label_at_node(tmp_path: Path) -> None:
    """Coord form and label form are equivalent at the nodes: a free coord-form
    position placed at a node's authoring coords reproduces the label-form push
    (the cardinal RBF weights are ``e_idx`` at node ``idx``).  This is the
    equivalence that makes ``personas%<pirate's coords>`` ≡ ``personas%pirate``.
    """
    from saklas.core.session import _affine_manifold_push

    folder = _discover_folder(
        tmp_path, name="trio", fit_mode="pca", labels=["a", "b", "c"],
        hyperparams={"max_dim": 2, "var_threshold": 0.999},
    )
    manifold = ManifoldExtractionPipeline(_Handle(), EventBus()).fit(folder)
    assert all(sub.is_affine for sub in manifold.layers.values())

    for idx, label in enumerate(manifold.node_labels):
        _, coord_label = _affine_manifold_push(manifold, label)
        node_coords = tuple(float(c) for c in manifold.node_coords[idx].tolist())
        _, coord_free = _affine_manifold_push(manifold, node_coords)
        for L in manifold.layers:
            assert torch.allclose(coord_free[L], coord_label[L], atol=1e-4), (
                f"layer {L} node {label}: coord form != label form"
            )


def test_affine_push_coord_form_interpolates_between_nodes(tmp_path: Path) -> None:
    """A free coord-form position between two nodes blends their per-layer
    targets — distinct from either endpoint, and the layout's two nearest nodes
    carry the dominant cardinal weight.
    """
    from saklas.core.manifold import rbf_cardinal_weights
    from saklas.core.session import _affine_manifold_push

    folder = _discover_folder(
        tmp_path, name="trio2", fit_mode="pca", labels=["a", "b", "c"],
        hyperparams={"max_dim": 2, "var_threshold": 0.999},
    )
    manifold = ManifoldExtractionPipeline(_Handle(), EventBus()).fit(folder)

    mid = (manifold.node_coords[0] + manifold.node_coords[1]) / 2
    w = rbf_cardinal_weights(manifold.node_coords, mid)
    # nodes 0 and 1 dominate the blend at their midpoint
    assert int(torch.argsort(w, descending=True)[0]) in (0, 1)
    assert int(torch.argsort(w, descending=True)[1]) in (0, 1)

    _, coord_mid = _affine_manifold_push(manifold, tuple(float(c) for c in mid))
    _, coord_a = _affine_manifold_push(manifold, "a")
    _, coord_b = _affine_manifold_push(manifold, "b")
    for L in manifold.layers:
        assert not torch.allclose(coord_mid[L], coord_a[L], atol=1e-3)
        assert not torch.allclose(coord_mid[L], coord_b[L], atol=1e-3)


def test_affine_push_coord_form_arity_mismatch_raises(tmp_path: Path) -> None:
    """A coord-form position with the wrong number of coordinates raises
    ``ManifoldArityError`` (a ``SteeringExprError``), matching the curved path.
    """
    from saklas.core.errors import ManifoldArityError
    from saklas.core.steering_expr import SteeringExprError
    from saklas.core.session import _affine_manifold_push

    folder = _discover_folder(
        tmp_path, name="trio3", fit_mode="pca", labels=["a", "b", "c"],
        hyperparams={"max_dim": 2, "var_threshold": 0.999},
    )
    manifold = ManifoldExtractionPipeline(_Handle(), EventBus()).fit(folder)
    n = manifold.domain.intrinsic_dim
    with pytest.raises(ManifoldArityError) as exc:
        _affine_manifold_push(manifold, tuple([0.0] * (n + 1)))
    assert isinstance(exc.value, SteeringExprError)


def test_discover_pca_flat_fit_skips_rbf_floor(tmp_path: Path) -> None:
    """The flat (``pca``) path is gated by the affine-span floor ``k+1``, not
    the RBF poisedness floor ``2k+1``: a flat subspace has no interpolant to
    keep poised.  Three nodes picking ``k=2`` fit fine (``3 == k+1``) where the
    old ``min_nodes(2)=5`` floor would have raised.  The ``2k+1`` floor stays in
    force on the curved (``spectral``) path.
    """
    folder = _discover_folder(
        tmp_path, fit_mode="pca", labels=["a", "b", "c"],
        hyperparams={"max_dim": 2, "var_threshold": 0.999},
    )
    manifold = ManifoldExtractionPipeline(_Handle(), EventBus()).fit(folder)
    assert manifold.domain.intrinsic_dim <= 2
    assert manifold.layers
    for sub in manifold.layers.values():
        assert sub.is_affine


# ----------------------------------------------------- fit_mode="auto" -------

def _circle_encoder_batch(
    model: Any, tokenizer: Any, prompts: Any, responses: Any, layers: Any,
    device: Any, **kwargs: Any,
) -> dict[int, torch.Tensor]:
    """Place node ``node<NN>`` at angle ``2π·NN/12`` on a circle in dims 0–1.

    Exercises the ``auto`` → persistent-homology → periodic ``BoxDomain`` path
    end to end: the per-node centroids trace a loop, so ``select_topology``
    must resolve a ``spectral`` curved fit over a 1-periodic box (a circle).
    """
    import math as _math
    rows = []
    for r in responses:
        i = int(r.split()[0][4:])           # "node07 statement 2" -> 7
        ang = 2.0 * _math.pi * i / 12.0
        base = torch.zeros(_DIM)
        base[0] = _math.cos(ang)
        base[1] = _math.sin(ang)
        out: dict[int, torch.Tensor] = {}
        for layer in range(len(layers)):
            v = base.clone()
            v[2] = 0.3 * layer
            out[layer] = v + 0.02 * torch.randn(_DIM)
        rows.append(out)
    return {
        idx: torch.stack([row[idx] for row in rows]) for idx in range(len(layers))
    }


def test_auto_records_resolution_metadata(tmp_path: Path) -> None:
    """An ``auto`` fit resolves a concrete geometry and records the ranking."""
    folder = _discover_folder(
        tmp_path, name="autoflat", fit_mode="auto",
        labels=[f"n{i:02d}" for i in range(8)], hyperparams={"max_dim": 4},
    )
    manifold = ManifoldExtractionPipeline(_Handle(), EventBus()).fit(folder)

    assert manifold.layers
    meta = manifold.metadata
    assert meta["fit_mode"] == "auto"
    assert meta["resolved_fit_mode"] in {"pca", "spectral"}
    assert meta["method"] == "manifold_discover_auto"
    candidates = cast(list[dict[str, Any]], meta["topology_candidates"])
    names = {c["name"] for c in candidates}
    assert "flat-pca" in names
    assert meta["topology_winner"] in names


def test_auto_detects_circle_as_periodic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``auto`` on loop-structured centroids resolves a periodic BoxDomain."""
    from saklas.core.manifold import BoxDomain
    monkeypatch.setattr(V, "_encode_and_capture_all_batch", _circle_encoder_batch)
    folder = _discover_folder(
        tmp_path, name="autocircle", fit_mode="auto",
        labels=[f"node{i:02d}" for i in range(12)], hyperparams={"max_dim": 4},
    )
    manifold = ManifoldExtractionPipeline(_Handle(), EventBus()).fit(folder)

    assert isinstance(manifold.domain, BoxDomain)
    assert manifold.domain.axes[0].periodic
    assert manifold.metadata["resolved_fit_mode"] == "spectral"
    assert manifold.metadata["topology_winner"] == "torus-T1"
    # The curved fit landed an RBF surface per layer (not affine).
    for sub in manifold.layers.values():
        assert not sub.is_affine
    # Fuzzy-manifold σ-field: the curved fit ran the within-node spread second
    # pass (same monkeypatched encoder), attaching a log-σ RBF to each layer and
    # stamping the sidecar summary.  End-to-end coverage of
    # compute_node_reduced_covariance → fit_sigma_field → save/load.
    for sub in manifold.layers.values():
        assert sub.has_sigma
        z = manifold.domain.embed(manifold.node_coords[0])
        assert float(sub.sigma_at(z)) > 0.0
    assert "sigma_field_per_layer" in manifold.metadata


def test_auto_curved_partial_topup_reuses_durable_rows_without_recapture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from saklas.io.paths import model_dir

    folder = _discover_folder(
        tmp_path, name="auto-topup", fit_mode="auto",
        labels=[f"node{i:02d}" for i in range(12)],
        hyperparams={"max_dim": 4},
    )
    scopes: list[list[int]] = []

    def _counting(*args: Any, **kwargs: Any) -> Any:
        scopes.append(list(kwargs["layer_indices"]))
        return _circle_encoder_batch(*args, **kwargs)

    monkeypatch.setattr(V, "_encode_and_capture_all_batch", _counting)
    pipe = ManifoldExtractionPipeline(_Handle(), EventBus())
    pipe.fit(folder, layer_indices=[0])
    row_zero, = (
        model_dir("stub-model") / "manifold_capture"
    ).glob("*.rows.layer_0.safetensors")
    before = row_zero.stat()
    scopes.clear()

    pipe.fit(folder, layer_indices=[0, 1])

    assert scopes == [[1]]
    after = row_zero.stat()
    assert (after.st_ino, after.st_mtime_ns, after.st_size) == (
        before.st_ino, before.st_mtime_ns, before.st_size,
    )


def test_adopt_fitted_manifold_rebinds_loaded_probe_profile_and_prefix(
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    from saklas.core.session import SaklasSession
    from saklas.core.vectors import fold_directions_to_subspace

    old = fold_directions_to_subspace(
        "mood", {0: torch.tensor([1.0, 0.0])}, None, label="mood",
    )
    other = fold_directions_to_subspace(
        "mood", {0: torch.tensor([0.0, 1.0])}, None, label="mood",
    )
    new = fold_directions_to_subspace(
        "mood", {0: torch.tensor([0.0, 2.0])}, None, label="mood",
    )

    class _Monitor:
        def __init__(self) -> None:
            self.probes = {
                "mood-probe": SimpleNamespace(manifold=old, top_n=4),
                "other-mood-probe": SimpleNamespace(manifold=other, top_n=4),
            }

        def attached_probes(self):
            return dict(self.probes)

        def remove_probe(self, name: str) -> None:
            self.probes.pop(name)

        def add_probe(self, name: str, manifold: Any, *, top_n: int) -> None:
            self.probes[name] = SimpleNamespace(
                manifold=manifold, top_n=top_n,
            )

    session: Any = object.__new__(SaklasSession)
    session._device = torch.device("cpu")
    session._dtype = torch.float32
    session._manifolds = {"local/mood": old, "other/mood": other}
    session._profiles = {
        "local/mood": {0: torch.tensor([1.0, 0.0])},
        "other/mood": {0: torch.tensor([0.0, 1.0])},
    }
    session.events = EventBus()
    session._monitor = _Monitor()
    session._probe_hash_cache = {
        "mood-probe": "old", "other-mood-probe": "other",
    }
    session._analytics_cpu_cache = {"local/mood": object()}
    session._prefix_cache = object()

    session._adopt_fitted_manifold(tmp_path / "local" / "mood", new)

    live = session._manifolds["local/mood"]
    assert torch.equal(live.layers[0].basis, new.layers[0].basis)
    attached = session._monitor.probes["mood-probe"]
    assert attached.manifold is live
    assert attached.top_n == 4
    assert torch.allclose(
        session._profiles["local/mood"][0], torch.tensor([0.0, 2.0]),
    )
    assert session._prefix_cache is None
    assert session._analytics_cpu_cache == {}
    assert "mood-probe" not in session._probe_hash_cache


def test_failed_fit_override_evicts_stale_manifold_consumers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from saklas.core.session import SaklasSession
    from saklas.core.vectors import fold_directions_to_subspace

    old = fold_directions_to_subspace(
        "mood", {0: torch.tensor([1.0, 0.0])}, None, label="mood",
    )
    other = fold_directions_to_subspace(
        "mood", {0: torch.tensor([0.0, 1.0])}, None, label="mood",
    )

    class _Monitor:
        def __init__(self) -> None:
            self.probes = {
                "mood-probe": SimpleNamespace(manifold=old, top_n=4),
                "other-mood-probe": SimpleNamespace(manifold=other, top_n=4),
            }

        def attached_probes(self) -> dict[str, Any]:
            return dict(self.probes)

        def remove_probe(self, name: str) -> None:
            self.probes.pop(name)

    session: Any = object.__new__(SaklasSession)
    session._manifolds = {"local/mood": old, "other/mood": other}
    session._profiles = {
        "local/mood": {0: torch.tensor([1.0, 0.0])},
        "other/mood": {0: torch.tensor([0.0, 1.0])},
    }
    session.events = EventBus()
    session._monitor = _Monitor()
    session._probe_hash_cache = {
        "mood-probe": "old", "other-mood-probe": "other",
    }
    session._analytics_cpu_cache = {"local/mood": object()}
    session._prefix_cache = object()

    @contextmanager
    def _exclusive(*_args: Any, **_kwargs: Any):
        yield

    folder = tmp_path / "local" / "mood"
    folder.mkdir(parents=True)
    (folder / "manifold.json").write_text('{"fit_mode":"auto"}')

    def _fail_fit(
        _pipe: Any, target: Any, *_args: Any, **_kwargs: Any,
    ) -> Any:
        (Path(target) / "manifold.json").write_text('{"fit_mode":"pca"}')
        raise RuntimeError("simulated override fit failure")

    monkeypatch.setattr(SaklasSession, "_model_exclusive", _exclusive)
    monkeypatch.setattr(
        SaklasSession, "_assert_unsteered_artifact_operation",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(ManifoldExtractionPipeline, "fit", _fail_fit)

    with pytest.raises(RuntimeError, match="override fit failure"):
        session.fit(
            folder,
            fit_mode="pca", hyperparams={"max_dim": 1},
        )

    assert session._manifolds == {"other/mood": other}
    assert set(session._profiles) == {"other/mood"}
    assert set(session._monitor.probes) == {"other-mood-probe"}
    assert session._probe_hash_cache == {"other-mood-probe": "other"}
    assert session._prefix_cache is None
    assert session._analytics_cpu_cache == {}


def test_override_validation_failure_preserves_unchanged_consumers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from saklas.core.session import SaklasSession
    from saklas.core.vectors import fold_directions_to_subspace

    old = fold_directions_to_subspace(
        "mood", {0: torch.tensor([1.0, 0.0])}, None, label="mood",
    )

    class _Monitor:
        def attached_probes(self) -> dict[str, Any]:
            return {}

    session: Any = object.__new__(SaklasSession)
    session._manifolds = {"local/mood": old}
    session._profiles = {"local/mood": {0: torch.tensor([1.0, 0.0])}}
    session.events = EventBus()
    session._monitor = _Monitor()
    session._probe_hash_cache = {}
    session._analytics_cpu_cache = {}
    session._prefix_cache = object()
    folder = tmp_path / "local" / "mood"
    folder.mkdir(parents=True)
    (folder / "manifold.json").write_text('{"fit_mode":"authored"}')

    @contextmanager
    def _exclusive(*_args: Any, **_kwargs: Any):
        yield

    def _fail_without_write(*_args: Any, **_kwargs: Any) -> Any:
        raise ValueError("overrides are discover-mode only")

    monkeypatch.setattr(SaklasSession, "_model_exclusive", _exclusive)
    monkeypatch.setattr(
        SaklasSession, "_assert_unsteered_artifact_operation",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        ManifoldExtractionPipeline, "fit", _fail_without_write,
    )

    with pytest.raises(ValueError, match="discover-mode only"):
        session.fit(folder, fit_mode="pca")

    assert session._manifolds == {"local/mood": old}
    assert set(session._profiles) == {"local/mood"}
