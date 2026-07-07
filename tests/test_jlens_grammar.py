"""CPU tests for the jlens grammar tier: gates, lazy resolution, ns guard."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from saklas.core.steering_composer import SteeringComposer
from saklas.core.steering_expr import format_expr, parse_expr
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


def test_resolve_probe_manifold_routes_jlens_to_profile_fold() -> None:
    session = _StubSession()
    session.fit_jlens(_PROMPTS)
    folded: list[str] = []
    session._fold_profile_probe = lambda name, profile: folded.append(name) or name  # type: ignore[attr-defined]
    from saklas.core.session import SaklasSession

    result = SaklasSession._resolve_probe_manifold(session, "jlens/g")  # type: ignore[arg-type]
    assert result == "jlens/g"
    assert folded == ["jlens/g"]
    assert "jlens/g" in session._profiles  # never fell through to extract()


# -------------------------------------------------------------- ns reservation


def test_manifold_authoring_rejects_jlens_namespace() -> None:
    with pytest.raises(ManifoldFormatError, match="reserved"):
        create_discover_manifold_folder(
            "jlens", "fake", "should not exist",
            node_corpora={"a": ["x"], "b": ["y"]},
            fit_mode="pca",
        )
