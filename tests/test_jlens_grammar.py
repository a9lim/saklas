"""CPU tests for the jlens grammar tier: gates, lazy resolution, ns guard."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import pytest
import torch

from saklas.core.steering_composer import SteeringComposer
from saklas.core.steering_expr import format_expr, parse_expr
from saklas.cli.runners import (
    _lens_fit_source_preflight_matches,
    _try_lens_fit_noop_preflight,
)
from saklas.io.manifold_folder import ManifoldFormatError
from saklas.io.manifold_authoring import create_discover_manifold_folder
from tests.test_jlens_session import _PROMPTS, _StubSession


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))


# ------------------------------------------------------------- gate grammar --


def _sole_trigger(steering: Any) -> Any:
    (entry,) = steering.alphas.values()
    assert isinstance(entry, tuple)
    return entry[1]


def test_when_gate_accepts_namespaced_probe() -> None:
    trigger = _sole_trigger(parse_expr("0.3 formal.casual@when:jlens/fake>0.4"))
    assert trigger.gate is not None
    assert trigger.gate.probe == "jlens/fake"
    assert trigger.gate.op == ">"
    assert trigger.gate.threshold == 0.4


def test_when_gate_namespaced_probe_round_trips() -> None:
    text = "0.3 formal.casual@when:jlens/fake>0.4"
    assert format_expr(parse_expr(text)) == format_expr(parse_expr(format_expr(parse_expr(text))))
    assert "jlens/fake" in format_expr(parse_expr(text))


def test_when_gate_namespaced_manifold_channels() -> None:
    for text, probe in [
        ("0.5 warm.clinical@when:default/emotions@happy>-0.5", "default/emotions@happy"),
        ("0.5 warm.clinical@when:default/emotions:fraction>0.5", "default/emotions:fraction"),
        ("0.5 warm.clinical@when:default/personas~hacker>0.5", "default/personas~hacker"),
        ("0.5 warm.clinical@when:default/personas[3]>0.4", "default/personas[3]"),
    ]:
        trigger = _sole_trigger(parse_expr(text))
        assert trigger.gate is not None
        assert trigger.gate.probe == probe, text


def test_jlens_steering_atom_round_trips() -> None:
    steering = parse_expr("0.5 jlens/orange")
    assert "jlens/orange" in steering.alphas
    round_tripped = parse_expr(format_expr(steering))
    assert "jlens/orange" in round_tripped.alphas

    ablation = parse_expr("!jlens/fake")
    assert "jlens/fake" in format_expr(ablation)


# ---------------------------------------------------------- lazy resolution --


def test_ensure_profile_registered_resolves_jlens_atom() -> None:
    session = _StubSession()
    session.fit_jlens(_PROMPTS)
    composer = SteeringComposer(session)  # type: ignore[arg-type]

    dirs = composer.ensure_profile_registered("jlens/g")
    assert "jlens/g" in session._profiles
    # workspace-band restricted: the 3-layer toy's band is layer 1
    assert set(dirs) == {1}
    assert all(isinstance(v, torch.Tensor) for v in dirs.values())


def test_ensure_profile_registered_jlens_requires_fitted_lens() -> None:
    from saklas.core.jlens import LensNotFittedError

    session = _StubSession()
    composer = SteeringComposer(session)  # type: ignore[arg-type]
    with pytest.raises(LensNotFittedError, match="saklas lens fit"):
        composer.ensure_profile_registered("jlens/g")


def test_add_probe_routes_jlens_to_lens_registry() -> None:
    """A ``jlens/<word>`` probe is a READOUT-channel probe (probability
    under the lens softmax), not a linear probe — ``add_probe``
    routes it to the session lens-probe registry, never folding the
    ``W_U[v] @ J_l`` direction and never falling through to ``extract()``."""
    session = _StubSession()
    session.fit_jlens(_PROMPTS)
    from saklas.core.session import SaklasSession

    name = SaklasSession.add_probe(session, "jlens/g")  # type: ignore[arg-type]
    assert name == "jlens/g"
    spec = session._lens_probes["jlens/g"]
    assert spec["word"] == "g"
    assert spec["layers"] == [
        int(l) for l in session._jlens_workspace_band(session.jlens)
    ]
    # No direction fold, no profile registration — the readout channel is
    # computed from lens logits, not a whitened coordinate.
    assert "jlens/g" not in session._profiles


def test_lens_probe_scores_strength_channel() -> None:
    """Lens-probe readings carry ONE channel — ``coords = (strength,)``,
    the mean band probability (the gate channel + the workspace card's
    number; apples-to-apples across tokens and layers) — plus the
    per-layer ``(p_l,)`` trace, matching ``token_readout_stats`` on the
    same logits."""
    from saklas.core.jlens import token_readout_stats
    from saklas.core.session import SaklasSession

    session = _StubSession()
    session.fit_jlens(_PROMPTS)
    SaklasSession.add_probe(session, "jlens/g")  # type: ignore[arg-type]
    layers = [int(l) for l in session._jlens_workspace_band(session.jlens)]
    d_model = next(iter(session.jlens.jacobians.values())).shape[0]
    hidden = {
        l: torch.randn(d_model, generator=torch.Generator().manual_seed(l))
        for l in layers
    }
    readings = session._score_lens_probes(hidden)
    reading = readings["jlens/g"]
    (strength,) = reading.coords
    assert 0.0 <= strength <= 1.0
    assert sorted(reading.coords_per_layer) == layers
    # Cross-check against the pure math on the same logits.
    logits = session._jlens_logits_rows(
        session.jlens, [(l, hidden[l]) for l in layers],
    )
    depths = [l / (len(session._layers) - 1) for l in layers]
    token_id = session._lens_probes["jlens/g"]["token_id"]
    ((exp_str, exp_com, exp_spread, per_layer),) = token_readout_stats(
        logits.float(), depths, [token_id],
    )
    assert strength == pytest.approx(exp_str)
    # strength is the mean of the per-layer probabilities — the objective
    # apples-to-apples identity a within-layer normalization can't satisfy.
    assert strength == pytest.approx(sum(per_layer) / len(per_layer))
    assert reading.depth_com == pytest.approx((exp_com,))
    assert reading.depth_spread == pytest.approx((exp_spread,))
    for l, p_l in zip(layers, per_layer):
        assert reading.coords_per_layer[l] == pytest.approx((p_l,))
        assert 0.0 <= p_l <= 1.0
    # Geometry fields are defaulted — there is no subspace behind a readout
    # probe.
    assert reading.fraction == 0.0
    assert reading.residual == 0.0
    assert reading.nearest == []
    assert reading.membership == 1.0


def test_gated_lens_probe_keys_and_gate_scalars() -> None:
    """Composer detection + the gate scalar key space: ``jlens/<word>`` =
    strength (the one channel), emitted by ``Monitor.flat_scalars`` over
    the synthesized reading."""
    from saklas.core.session import SaklasSession

    session = _StubSession()
    session.fit_jlens(_PROMPTS)
    SaklasSession.add_probe(session, "jlens/g")  # type: ignore[arg-type]

    class _FlatCapture:
        def __init__(self, latest: dict[int, torch.Tensor]) -> None:
            self._latest = latest

        def latest_per_layer(self) -> dict[int, torch.Tensor]:
            return self._latest

    layers = [int(l) for l in session._jlens_workspace_band(session.jlens)]
    d_model = next(iter(session.jlens.jacobians.values())).shape[0]
    session._capture = _FlatCapture(
        {l: torch.randn(d_model, generator=torch.Generator().manual_seed(l))
         for l in layers}
    )
    scalars = session._score_lens_gate_scalars()
    assert "jlens/g" in scalars
    assert "jlens/g[0]" in scalars
    assert "jlens/g[1]" not in scalars  # one channel — strength only
    assert scalars["jlens/g"] == scalars["jlens/g[0]"]
    # The stash is armed for the display step to reuse this forward's logits.
    assert session._lens_step_stash is not None
    assert session._lens_step_stash["fresh"] is True
    assert session._lens_step_stash["layers"] == tuple(layers)

    # Composer-side detection: a gate on the pinned lens probe is recognized
    # from the steering stack without monitor attachment.
    composer = SteeringComposer(session)  # type: ignore[arg-type]
    steering = parse_expr("0.3 jlens/g@when:jlens/g>0.4")
    composer._stack.append(dict(steering.alphas))  # type: ignore[arg-type]
    session._monitor = type(
        "_M", (), {"probe_names": ()},
    )()  # no monitor probes attached
    assert composer.gated_lens_probe_keys() == {"jlens/g"}


def test_lens_gate_scalar_scores_only_referenced_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import saklas.core.jlens as jlens_module
    from saklas.core.session import SaklasSession

    session = _StubSession()
    session.fit_jlens(_PROMPTS)
    SaklasSession.add_probe(session, "jlens/g")  # type: ignore[arg-type]
    SaklasSession.add_probe(session, "jlens/a")  # type: ignore[arg-type]
    session.enable_live_lens(layers=[1], top_k=3)
    setattr(session, "_live_lens_active_for_generation", False)

    class _FlatCapture:
        def __init__(self, latest: dict[int, torch.Tensor]) -> None:
            self._latest = latest

        def latest_per_layer(self) -> dict[int, torch.Tensor]:
            return self._latest

    layers = [int(l) for l in session._jlens_workspace_band(session.jlens)]
    d_model = next(iter(session.jlens.jacobians.values())).shape[0]
    session._capture = _FlatCapture(
        {l: torch.randn(d_model, generator=torch.Generator().manual_seed(l + 10))
         for l in layers}
    )

    def _fail_full_probabilities(_logits: torch.Tensor) -> torch.Tensor:
        raise AssertionError("gate-only lens scalar scoring should be column-only")

    monkeypatch.setattr(
        jlens_module, "readout_probabilities", _fail_full_probabilities,
    )

    scalars = session._score_lens_gate_scalars({"jlens/g"})

    assert "jlens/g" in scalars
    assert "jlens/g[0]" in scalars
    assert "jlens/a" not in scalars
    assert session._lens_step_stash is not None
    assert "probabilities" not in session._lens_step_stash


# -------------------------------------------------------------- ns reservation


@pytest.mark.parametrize("namespace", ["jlens", "sae"])
def test_manifold_authoring_rejects_reserved_runtime_namespace(
    namespace: str,
) -> None:
    with pytest.raises(ManifoldFormatError, match="reserved"):
        create_discover_manifold_folder(
            namespace, "fake", "should not exist",
            node_corpora={"a": ["x"], "b": ["y"]},
            fit_mode="pca",
        )


def test_lens_preflight_rejects_contiguous_prefix_for_all_layers() -> None:
    sidecar: dict[str, object] = {
        "source_layers": list(range(5)),
        "model_layer_count": 12,
    }
    assert not _lens_fit_source_preflight_matches(sidecar, "all")


def test_lens_preflight_resolves_workspace_from_model_depth() -> None:
    sidecar: dict[str, object] = {
        "source_layers": [4, 5, 6, 7, 8, 9],
        "model_layer_count": 11,
    }
    assert _lens_fit_source_preflight_matches(sidecar, "workspace")
    sidecar["source_layers"] = [4, 5, 6]
    assert not _lens_fit_source_preflight_matches(sidecar, "workspace")


def test_lens_noop_preflight_requires_exact_model_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from saklas.io.lens import lens_paths

    expected_spec = "hf:repo/config@dataset-rev-a"
    tensor_path, _ = lens_paths("toy/model")
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    tensor_path.write_bytes(b"lens")
    sidecar: dict[str, object] = {
        "source_layers": [0],
        "model_layer_count": 2,
        "model_source_fingerprint": "source-a",
        "tensor_sha256": hashlib.sha256(b"lens").hexdigest(),
        "seq_len": 128,
        "corpus_spec": expected_spec,
        "raw_prompt_count": 100,
        "usable_prompt_count": 100,
        "n_prompts": 100,
        "d_model": 6,
    }
    monkeypatch.setattr(
        "saklas.io.lens.load_lens_sidecar", lambda _model: sidecar,
    )
    monkeypatch.setattr(
        "saklas.core.model.model_source_fingerprint",
        lambda *_args, **_kwargs: "source-a",
    )
    monkeypatch.setattr(
        "saklas.io.lens.resolved_default_lens_corpus_spec",
        lambda: ("dataset-rev-a", expected_spec),
    )
    args = argparse.Namespace(
        force=False, model="toy/model", quantize=None, device="cpu",
        seq_len=None, corpus=None, prompts=100,
    )
    assert _try_lens_fit_noop_preflight(args, [0])
    tensor_path.write_bytes(b"corrupt")
    assert not _try_lens_fit_noop_preflight(args, [0])
    tensor_path.write_bytes(b"lens")
    sidecar["model_source_fingerprint"] = "source-b"
    assert not _try_lens_fit_noop_preflight(args, [0])


def test_lens_noop_preflight_rejects_changed_default_dataset_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from saklas.io.lens import lens_paths

    tensor_path, _ = lens_paths("toy/model")
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    tensor_path.write_bytes(b"lens")
    sidecar = {
        "source_layers": [0], "model_layer_count": 2,
        "model_source_fingerprint": "source-a",
        "tensor_sha256": hashlib.sha256(b"lens").hexdigest(),
        "seq_len": 128, "corpus_spec": "hf:repo/config@old",
        "raw_prompt_count": 100, "usable_prompt_count": 100,
        "n_prompts": 100, "d_model": 6,
    }
    monkeypatch.setattr(
        "saklas.io.lens.load_lens_sidecar", lambda _model: sidecar,
    )
    monkeypatch.setattr(
        "saklas.core.model.model_source_fingerprint",
        lambda *_args, **_kwargs: "source-a",
    )
    monkeypatch.setattr(
        "saklas.io.lens.resolved_default_lens_corpus_spec",
        lambda: ("new", "hf:repo/config@new"),
    )
    args = argparse.Namespace(
        force=False, model="toy/model", quantize=None, device="cpu",
        seq_len=None, corpus=None, prompts=100,
    )
    assert not _try_lens_fit_noop_preflight(args, [0])
