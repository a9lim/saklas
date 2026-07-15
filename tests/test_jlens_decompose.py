"""CPU tests for the J-space sparse nonnegative decomposition."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from saklas.core.jlens import sparse_nonneg_decompose
from tests.test_jlens_session import _PROMPTS, _StubSession

_D = 32
_VOCAB = 60


def _dictionary(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    jacobian = torch.randn(_D, _D, generator=gen) / _D**0.5
    unembed = torch.randn(_VOCAB, _D, generator=gen)
    return jacobian, unembed


def test_recovers_planted_sparse_combination() -> None:
    jacobian, unembed = _dictionary()
    atoms = {7: 3.0, 21: 2.0, 40: 1.2}
    target = torch.zeros(_D)
    for v, c in atoms.items():
        target = target + c * (unembed[v].float() @ jacobian)

    dec = sparse_nonneg_decompose(target, jacobian, unembed, layer=0, k=8)

    got = dict(dec.tokens)
    assert set(atoms) <= set(got)
    for v, c in atoms.items():
        assert abs(got[v] - c) < 0.05, (v, got[v], c)
    assert dec.share > 0.99


def test_positive_tiny_solve_skips_projected_gradient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    d = 8
    jacobian = torch.eye(d)
    unembed = torch.eye(d)
    target = 2.0 * unembed[2] + 1.5 * unembed[5]

    def _explode(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise AssertionError("positive NNLS case should not need eigvalsh/PGD")

    monkeypatch.setattr(torch.linalg, "eigvalsh", _explode)
    dec = sparse_nonneg_decompose(target, jacobian, unembed, layer=0, k=2)

    got = dict(dec.tokens)
    assert got[2] == pytest.approx(2.0, abs=1e-5)
    assert got[5] == pytest.approx(1.5, abs=1e-5)
    assert dec.share == pytest.approx(1.0)


def test_share_low_for_offspace_direction() -> None:
    jacobian, unembed = _dictionary()
    # a direction orthogonal to every atom's positive span is unreachable
    # with nonneg coefficients from a *random* dictionary only approximately;
    # instead check that pure noise explains much less than a planted target.
    gen = torch.Generator().manual_seed(3)
    noise = torch.randn(_D, generator=gen)
    dec = sparse_nonneg_decompose(noise, jacobian, unembed, layer=0, k=4)
    assert dec.share < 0.9


def test_zero_target_yields_empty() -> None:
    jacobian, unembed = _dictionary()
    dec = sparse_nonneg_decompose(torch.zeros(_D), jacobian, unembed, layer=0, k=4)
    assert dec.share == 0.0 and dec.tokens == []


def test_respects_k_budget() -> None:
    jacobian, unembed = _dictionary()
    gen = torch.Generator().manual_seed(5)
    target = torch.randn(_D, generator=gen)
    dec = sparse_nonneg_decompose(target, jacobian, unembed, layer=0, k=3)
    assert len(dec.tokens) <= 3


@pytest.fixture()
def _isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))


def test_session_jspace_decompose_on_registered_profile(_isolated_home: None) -> None:
    session = _StubSession()
    lens = session.fit_jlens(_PROMPTS)
    # register the direction for toy-vocab token 'g', then decompose it —
    # the decomposition should attribute it (near-)fully to that atom.
    name = session.register_jlens_direction("g")
    session.ensure_profile_registered = lambda sel: session._profiles[sel]  # type: ignore[attr-defined]
    from saklas.core.session import SaklasSession

    out = SaklasSession.jspace_decompose(session, name, k=4)  # type: ignore[arg-type]
    del lens
    assert set(out) == {0, 1}
    for share, tokens in out.values():
        assert share > 0.99
        assert tokens and tokens[0][0] == "g"
