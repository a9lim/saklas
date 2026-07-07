"""GPU integration tests for the Jacobian lens on a real model.

Micro-fit only (a handful of prompts at a large dim_batch) — enough to
exercise the real backward path, the motor-regime sanity check, and a
jlens steering atom end-to-end. Real lens quality needs `saklas lens fit`
with ≥100 prompts; nothing here asserts readout *quality* beyond the
late-layer identity.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from saklas import SamplingConfig

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not (torch.cuda.is_available() or torch.backends.mps.is_available()),
        reason="No GPU backend available (neither CUDA nor MPS)",
    ),
]

MODEL_ID = "google/gemma-3-4b-it"

_FIT_PROMPTS = [
    "The history of the printing press begins in fifteenth-century Mainz, "
    "where Johannes Gutenberg combined movable metal type with a screw press "
    "adapted from wine-making, transforming the economics of written text "
    "across Europe within a single generation.",
    "Photosynthesis converts light energy into chemical energy in two stages: "
    "the light-dependent reactions split water and generate ATP and NADPH, "
    "while the Calvin cycle fixes atmospheric carbon dioxide into "
    "three-carbon sugars that feed the plant's metabolism.",
]


@pytest.fixture(scope="module")
def session() -> Any:
    from saklas import SaklasSession

    with SaklasSession.from_pretrained(MODEL_ID, device="auto", probes=[]) as s:
        yield s


@pytest.fixture(scope="module")
def micro_lens(session: Any, tmp_path_factory: pytest.TempPathFactory) -> Any:
    import os

    home = tmp_path_factory.mktemp("saklas-home")
    old = os.environ.get("SAKLAS_HOME")
    os.environ["SAKLAS_HOME"] = str(home)
    try:
        lens = session.fit_jlens(
            _FIT_PROMPTS, corpus_spec="gpu-micro-test", dim_batch=64, seq_len=64,
        )
        yield lens
    finally:
        session._jlens = None
        if old is None:
            os.environ.pop("SAKLAS_HOME", None)
        else:
            os.environ["SAKLAS_HOME"] = old


def test_micro_fit_shape(session: Any, micro_lens: Any) -> None:
    n_layers = len(session.layers)
    assert micro_lens.n_prompts == 2
    assert len(micro_lens.source_layers) == n_layers - 1
    d = session.model_info["hidden_dim"]
    assert micro_lens.d_model == d
    for J in micro_lens.jacobians.values():
        assert torch.isfinite(J).all()


def test_motor_regime_top1_matches_model(session: Any, micro_lens: Any) -> None:
    """At the last source layer J ≈ I, so the lens top-1 must equal the
    model's actual next-token prediction (the logit-lens identity)."""
    session._jlens = micro_lens
    last = micro_lens.source_layers[-1]
    prompt = "Fact: the currency used in the country shaped like a boot is the"
    out = session.jlens_readout(prompt, layers=[last], top_k=5)

    ids = session.tokenizer(prompt, return_tensors="pt")["input_ids"].to(session.device)
    with torch.inference_mode():
        logits = session.model(input_ids=ids).logits[0, -1]
    model_top1 = session.tokenizer.decode([int(logits.argmax())])
    lens_top_tokens = [tok for tok, _ in out[last][0]]
    assert model_top1 in lens_top_tokens[:2], (model_top1, lens_top_tokens)


def test_jlens_steering_atom_generates(session: Any, micro_lens: Any) -> None:
    session._jlens = micro_lens
    result = session.generate(
        "Say something brief.",
        steering="0.5 jlens/ocean",
        sampling=SamplingConfig(max_tokens=16, seed=7),
        stateless=True,
    )
    assert result.first.text
    assert "jlens/ocean" in (result.first.applied_steering or "")


def test_live_lens_does_not_break_static_steerable(session: Any, micro_lens: Any) -> None:
    session._jlens = micro_lens
    baseline = session._steering.static_steerable()
    layers = session.enable_live_lens(top_k=3)
    try:
        assert session._steering.static_steerable() == baseline
        assert layers  # resolved a mid-band subset
        result = session.generate(
            "Count to three.",
            sampling=SamplingConfig(max_tokens=8, seed=3),
            stateless=True,
        )
        assert result.first.text
    finally:
        session.disable_live_lens()
