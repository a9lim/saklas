"""StaticCache + CUDA-graphs detection (Phase B, v2.2).

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

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from saklas.core import cuda_graphs as cg


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
    supported, reason = cg.is_cuda_graphs_supported(_FakeModel(), "cpu")
    assert supported is False
    assert reason is not None and "CUDA-only" in reason


def test_mps_device_returns_unsupported():
    """MPS path mirrors CPU.  We could in principle build StaticCache on
    MPS but graph capture is a CUDA-only torch feature, so the static-
    shape benefit doesn't pay off — simpler to gate at device.
    """
    supported, reason = cg.is_cuda_graphs_supported(_FakeModel(), "mps")
    assert supported is False
    assert reason is not None and "CUDA-only" in reason


def test_torch_device_object_accepted():
    """The function accepts both string and torch.device — the session
    stores ``self._device`` as a ``torch.device`` after the parameter-
    iteration probe, so we must handle either shape."""
    dev = torch.device("cpu")
    supported, reason = cg.is_cuda_graphs_supported(_FakeModel(), dev)
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
    # Force the device check to think we're on CUDA so the StaticCache
    # branch fires; mock the import to raise.
    with patch.object(cg, "torch", torch):  # keep the real torch
        # Build a fake "cuda" device string and a model that crashes
        # StaticCache construction.  The function imports StaticCache
        # locally; we patch it via the transformers module.
        from transformers import cache_utils
        with patch.object(
            cache_utils, "StaticCache",
            side_effect=RuntimeError("synthetic: layer_types missing"),
        ):
            supported, reason = cg.is_cuda_graphs_supported(
                _FakeModel(), "cuda",
            )
    assert supported is False
    assert reason is not None
    assert "StaticCache construction failed" in reason
    assert "RuntimeError" in reason


# ---------------------------------------------------------------------------
# warn_once dedupe — fallback reason should fire once per model lifetime,
# not per generation step.
# ---------------------------------------------------------------------------


def test_warn_once_dedupes_per_model(caplog):
    """The session calls ``warn_once`` at construction; subsequent calls
    on the same model object should be silent."""
    import logging
    caplog.set_level(logging.INFO, logger=cg.log.name)
    cg._warned_models.clear()  # reset between tests

    m1 = _FakeModel()
    m2 = _FakeModel()

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


# ---------------------------------------------------------------------------
# CLI + YAML opt-out plumbing.
# ---------------------------------------------------------------------------


def test_no_cuda_graphs_flag_parses():
    from saklas import cli
    args = cli.parse_args(["tui", "google/gemma-2-2b-it"])
    assert getattr(args, "no_cuda_graphs", False) is False
    args = cli.parse_args(["tui", "google/gemma-2-2b-it", "--no-cuda-graphs"])
    assert args.no_cuda_graphs is True


def test_yaml_cuda_graphs_false_folds_onto_args(monkeypatch, tmp_path):
    from saklas import cli
    from saklas.cli import runners as cli_runners
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    p = tmp_path / "off.yaml"
    p.write_text("model: google/gemma-2-2b-it\ncuda_graphs: false\n")
    args = cli.parse_args(["tui", "-c", str(p)])
    assert getattr(args, "no_cuda_graphs", False) is False
    cli_runners._load_effective_config(args)
    assert args.no_cuda_graphs is True


def test_yaml_cuda_graphs_invalid_type_errors(tmp_path):
    """``cuda_graphs: "false"`` (a YAML string) must reject rather than
    coerce — coercion would leave the static-cache path silently active."""
    from saklas.cli.config_file import ConfigFile, ConfigFileError
    p = tmp_path / "bad.yaml"
    p.write_text('cuda_graphs: "false"\n')
    with pytest.raises(ConfigFileError, match="cuda_graphs must be a boolean"):
        ConfigFile.load(p)


def test_yaml_compile_and_cuda_graphs_compose(monkeypatch, tmp_path):
    """Both opt-outs in one YAML — the runner sees both args set."""
    from saklas import cli
    from saklas.cli import runners as cli_runners
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    p = tmp_path / "off.yaml"
    p.write_text(
        "model: google/gemma-2-2b-it\ncompile: false\ncuda_graphs: false\n"
    )
    args = cli.parse_args(["tui", "-c", str(p)])
    cli_runners._load_effective_config(args)
    assert args.no_compile is True
    assert args.no_cuda_graphs is True
