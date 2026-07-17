"""StaticCache + CUDA-graph detection and session-cache reuse.

The actual graph-capture path needs CUDA hardware to exercise; these
tests pin the *gating* behavior — what makes
``is_cuda_graphs_supported`` return ``False`` on the eager / non-CUDA
side, and that the session correctly routes around the support flag.

CUDA-side equivalence (``cuda_graphs=True`` produces the same token IDs
as ``cuda_graphs=False`` at fixed seed) is owned by the GPU smoke tests
in ``tests/test_smoke.py`` — gated on ``torch.cuda.is_available()`` and
not run in the CPU-only matrix.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest
import torch

from saklas.core import cuda_graphs as cg


@pytest.fixture(autouse=True)
def _isolate_cg_module_caches():
    """Clear the module-level support/warn caches around every test.

    ``is_static_cache_supported`` memoizes by ``id(model)``, which Python can
    reuse across the throwaway ``_FakeModel`` instances different tests build —
    so without isolation a cached ``(id, device) -> True`` can leak into a later
    test that expects a construction failure.  Clear before and after.
    """
    cg._support_cache.clear()
    cg._warned_models.clear()
    yield
    cg._support_cache.clear()
    cg._warned_models.clear()


# ---------------------------------------------------------------------------
# Device gating — CUDA graphs are CUDA-only.
# ---------------------------------------------------------------------------


class _FakeModel:
    """Minimal stand-in so ``is_cuda_graphs_supported`` can probe.

    Only the attribute paths the function touches matter:
    ``model.config`` (forwarded to StaticCache constructor) and
    ``next(model.parameters()).dtype`` for the cache dtype.
    """

    def __init__(self):
        self.config = SimpleNamespace(model_type="qwen3", num_hidden_layers=2)

    def parameters(self):
        return iter([torch.zeros(1, dtype=torch.bfloat16)])


def test_cpu_device_returns_unsupported():
    """CPU is the safest signal: don't try to build StaticCache, don't
    even import — just say no with a reason mentioning the device."""
    m: Any = _FakeModel()
    supported, reason = cg.is_cuda_graphs_supported(m, "cpu")
    assert supported is False
    assert reason is not None and "CUDA-only" in reason


def test_mps_device_returns_unsupported():
    """``is_cuda_graphs_supported`` still gates MPS out — CUDA-graph capture
    (``reduce-overhead``) is a CUDA-only torch feature.  StaticCache + compile
    *fusion* on MPS is a separate, device-agnostic probe
    (:func:`is_static_cache_supported`, covered below).
    """
    m: Any = _FakeModel()
    supported, reason = cg.is_cuda_graphs_supported(m, "mps")
    assert supported is False
    assert reason is not None and "CUDA-only" in reason


def test_static_cache_supported_is_device_agnostic():
    """``is_static_cache_supported`` does NOT gate on CUDA — StaticCache (the
    enabler for ``torch.compile`` fusion) pays off on MPS/CPU too, so it
    proceeds to the construction probe on any device instead of short-
    circuiting with a 'CUDA-only' reason."""
    import transformers

    class _OKCache:
        def __init__(self, *a: Any, **k: Any) -> None:
            self.layers = [object(), object()]

    m: Any = _FakeModel()
    # Patch the binding ``is_static_cache_supported`` reads (``from transformers
    # import StaticCache``) — patching ``cache_utils.StaticCache`` is fragile
    # once the lazy top-level attribute has resolved.
    with patch.object(transformers, "StaticCache", _OKCache):
        for dev in ("mps", "cpu", "cuda"):
            supported, reason = cg.is_static_cache_supported(m, dev)
            assert supported is True, (dev, reason)
            assert reason is None
    # The CUDA-graphs probe still rejects the non-CUDA devices (graph capture),
    # even though StaticCache itself is supported there.
    assert cg.is_cuda_graphs_supported(m, "mps")[0] is False
    assert cg.is_cuda_graphs_supported(m, "cpu")[0] is False


def test_static_cache_construction_failure_device_agnostic():
    """A StaticCache that fails to construct is reported as unsupported on MPS
    too (a real reason, not the CUDA-only gate)."""
    import transformers
    m: Any = _FakeModel()
    with patch.object(
        transformers, "StaticCache",
        side_effect=RuntimeError("synthetic: layer_types missing"),
    ):
        supported, reason = cg.is_static_cache_supported(m, "mps")
    assert supported is False
    assert reason is not None and "StaticCache construction failed" in reason


def test_torch_device_object_accepted():
    """The function accepts both string and torch.device — the session
    stores ``self._device`` as a ``torch.device`` after the parameter-
    iteration probe, so we must handle either shape."""
    dev = torch.device("cpu")
    m: Any = _FakeModel()
    supported, reason = cg.is_cuda_graphs_supported(m, dev)
    assert supported is False
    assert reason is not None


# ---------------------------------------------------------------------------
# StaticCache import / construction failure paths.
# ---------------------------------------------------------------------------


def test_static_cache_construction_failure_returns_unsupported():
    """Architectures that StaticCache doesn't know how to build for
    (custom modeling, MLA quirks, etc.) raise inside the constructor.
    The probe must catch broadly and surface a reason rather than
    propagating the raw exception — callers expect a ``(bool, str|None)``
    contract regardless of the underlying failure mode.
    """
    # Build a model that crashes StaticCache construction.  The function does
    # ``from transformers import StaticCache`` at call time, so patch that
    # binding (not ``cache_utils.StaticCache``, which the lazy top-level
    # attribute may already have resolved past).
    import transformers
    with patch.object(
        transformers, "StaticCache",
        side_effect=RuntimeError("synthetic: layer_types missing"),
    ):
        m: Any = _FakeModel()
        supported, reason = cg.is_cuda_graphs_supported(
            m, "cuda",
        )
    assert supported is False
    assert reason is not None
    assert "StaticCache construction failed" in reason
    assert "RuntimeError" in reason


# ---------------------------------------------------------------------------
# warn_once dedupe — fallback reason should fire once per model lifetime,
# not per generation step.
# ---------------------------------------------------------------------------


def test_warn_once_dedupes_per_model(caplog: pytest.LogCaptureFixture):
    """The session calls ``warn_once`` at construction; subsequent calls
    on the same model object should be silent."""
    import logging
    caplog.set_level(logging.INFO, logger=cg.log.name)
    cg._warned_models.clear()  # reset between tests

    m1: Any = _FakeModel()
    m2: Any = _FakeModel()

    cg.warn_once(m1, "test reason A")
    cg.warn_once(m1, "test reason A")  # dedupe
    cg.warn_once(m2, "test reason B")  # fresh model, fires

    fallback_records = [
        r for r in caplog.records
        if "CUDA graphs disabled" in r.getMessage()
    ]
    assert len(fallback_records) == 2, (
        f"expected 2 distinct warnings (one per model), got "
        f"{[r.getMessage() for r in fallback_records]}"
    )


def test_support_probe_cache_survives_compile_wrapper(monkeypatch: pytest.MonkeyPatch):
    """from_pretrained probes before torch.compile; __init__ sees wrapper.

    The second support check should hit the cached result through
    ``_orig_mod`` instead of constructing another StaticCache probe.
    """
    import transformers

    cg._support_cache.clear()
    calls: list[object] = []

    class _FakeStaticCache:
        def __init__(self, config: Any, *, max_cache_len: int, device: Any, dtype: Any) -> None:
            calls.append((config, max_cache_len, device, dtype))
            self.layers = [object()]

    monkeypatch.setattr(transformers, "StaticCache", _FakeStaticCache, raising=False)
    model: Any = _FakeModel()

    assert cg.is_cuda_graphs_supported(model, "cuda") == (True, None)
    wrapper: Any = SimpleNamespace(_orig_mod=model)
    assert cg.is_cuda_graphs_supported(wrapper, "cuda") == (True, None)
    assert len(calls) == 1


def test_make_static_cache_early_initializes_standard_gqa_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fresh caches must allocate and mark K/V tensors before Dynamo tracing.

    Transformers 5.x lazy initialization inside the compiled prefill leaves a
    CPU ``cumulative_length`` input and unmarked mutated K/V tensors, which
    disables CUDA graphs and recompiles for each new cache.
    """
    import transformers

    calls: list[dict[str, Any]] = []

    class _CacheLayer:
        is_sliding = False

    class _EarlyCache:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.layers = [_CacheLayer(), _CacheLayer()]

        def early_initialization(self, **kwargs: Any) -> None:
            calls.append(kwargs)

    text_config = SimpleNamespace(
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=64,
        hidden_size=512,
    )
    model: Any = SimpleNamespace(
        config=SimpleNamespace(get_text_config=lambda: text_config),
    )
    monkeypatch.setattr(transformers, "StaticCache", _EarlyCache, raising=False)

    cache = cg.make_static_cache(
        model,
        max_cache_len=256,
        device="cuda",
        dtype=torch.bfloat16,
    )

    assert isinstance(cache, _EarlyCache)
    assert calls == [{
        "batch_size": 1,
        "num_heads": 2,
        "head_dim": 64,
        "dtype": torch.bfloat16,
        "device": torch.device("cuda"),
    }]


def test_session_reuses_and_resets_generation_static_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from saklas.core.session import SaklasSession

    class _Cache:
        def __init__(self, capacity: int) -> None:
            self.capacity = capacity
            self.resets = 0

        def reset(self) -> None:
            self.resets += 1

    class _Model:
        def __init__(self) -> None:
            self.param = torch.zeros(1, dtype=torch.float16)

        def parameters(self):
            return iter([self.param])

    made: list[_Cache] = []

    def _make_static_cache(
        model: Any,
        *,
        max_cache_len: int,
        device: Any,
        dtype: Any,
    ) -> _Cache:
        del model, device, dtype
        cache = _Cache(max_cache_len)
        made.append(cache)
        return cache

    monkeypatch.setattr(
        "saklas.core.cuda_graphs.make_static_cache",
        _make_static_cache,
    )
    session = SaklasSession.__new__(SaklasSession)
    session._device = torch.device("cpu")
    session._generation_static_cache = None
    session._generation_static_cache_len = 0
    model = _Model()

    first = cast(
        _Cache,
        session._acquire_generation_static_cache(
            cast(Any, model), max_cache_len=512,
        ),
    )
    reused = cast(
        _Cache,
        session._acquire_generation_static_cache(
            cast(Any, model), max_cache_len=256,
        ),
    )
    grown = cast(
        _Cache,
        session._acquire_generation_static_cache(
            cast(Any, model), max_cache_len=768,
        ),
    )

    assert reused is first
    assert first.resets == 1
    assert grown is not first
    assert [cache.capacity for cache in made] == [512, 768]


# ---------------------------------------------------------------------------
# CLI + YAML opt-in plumbing.
# ---------------------------------------------------------------------------


def test_cuda_graphs_flag_parses():
    from saklas import cli
    args = cli.parse_args(["serve", "google/gemma-2-2b-it"])
    assert getattr(args, "cuda_graphs", False) is False
    args = cli.parse_args(["serve", "google/gemma-2-2b-it", "--cuda-graphs"])
    assert args.cuda_graphs is True


def test_yaml_cuda_graphs_true_folds_onto_args(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from saklas import cli
    from saklas.cli import runners as cli_runners
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    p = tmp_path / "on.yaml"
    p.write_text("model: google/gemma-2-2b-it\ncuda_graphs: true\n")
    args = cli.parse_args(["serve", "-c", str(p)])
    assert getattr(args, "cuda_graphs", False) is False
    cli_runners._load_effective_config(args)
    assert args.cuda_graphs is True


def test_yaml_cuda_graphs_invalid_type_errors(tmp_path: Path):
    """``cuda_graphs: "true"`` (a YAML string) must reject rather than
    coerce — coercion would silently turn the static-cache path on."""
    from saklas.cli.config_file import ConfigFile, ConfigFileError
    p = tmp_path / "bad.yaml"
    p.write_text('cuda_graphs: "false"\n')
    with pytest.raises(ConfigFileError, match="cuda_graphs must be a boolean"):
        ConfigFile.load(p)


def test_yaml_compile_and_cuda_graphs_compose(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Both opt-ins in one YAML — the runner sees both args set."""
    from saklas import cli
    from saklas.cli import runners as cli_runners
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    p = tmp_path / "on.yaml"
    p.write_text(
        "model: google/gemma-2-2b-it\ncompile: true\ncuda_graphs: true\n"
    )
    args = cli.parse_args(["serve", "-c", str(p)])
    cli_runners._load_effective_config(args)
    assert args.compile is True
    assert args.cuda_graphs is True
