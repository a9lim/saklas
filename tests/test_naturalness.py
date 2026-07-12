"""Behavior-manifold naturalness eval — CPU only, synthetic + mock model."""
from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from tests.conftest import CharTokenizer, FakeLogitsModel
from saklas.core.manifold import BoxAxis, BoxDomain
from saklas.core.naturalness import (
    bhattacharyya_distance,
    compute_node_behavior_centroid,
    compute_trajectory_distributions,
    fit_behavior_manifold,
    to_hellinger,
    trajectory_naturalness,
)


def _curve_dists(k: int, v: int) -> torch.Tensor:
    """K probability distributions tracing a smooth circle in logit space."""
    dists = torch.zeros(k, v)
    for i in range(k):
        theta = 2.0 * math.pi * i / k
        logits = torch.zeros(v)
        logits[0] = 3.0 * math.cos(theta)
        logits[1] = 3.0 * math.sin(theta)
        dists[i] = torch.softmax(logits, dim=-1)
    return dists


def _loop_domain(k: int) -> tuple[BoxDomain, torch.Tensor, torch.Tensor]:
    """A 1-D periodic domain over K nodes; returns (domain, coords, params)."""
    domain = BoxDomain([BoxAxis("t", periodic=True, period=1.0)])
    coords = torch.tensor([[i / k] for i in range(k)])
    return domain, coords, domain.embed(coords)


# --------------------------------------------------------------- metrics ---

def test_to_hellinger_round_trip():
    p = torch.softmax(torch.randn(20), dim=-1)
    assert torch.allclose(to_hellinger(p) ** 2, p, atol=1e-6)


def test_bhattacharyya_identical_is_zero():
    p = torch.softmax(torch.randn(32), dim=-1)
    assert bhattacharyya_distance(p, p).item() == abs(
        bhattacharyya_distance(p, p).item()
    )
    assert bhattacharyya_distance(p, p).item() < 1e-5


def test_bhattacharyya_disjoint_is_large():
    v = 16
    p = torch.zeros(v)
    p[:4] = 0.25
    q = torch.zeros(v)
    q[4:8] = 0.25
    # Disjoint support -> BC = 0 -> distance saturates high.
    assert bhattacharyya_distance(p, q).item() > 10.0


def test_bhattacharyya_batched():
    p = torch.softmax(torch.randn(5, 12), dim=-1)
    q = torch.softmax(torch.randn(5, 12), dim=-1)
    d = bhattacharyya_distance(p, q)
    assert d.shape == (5,)
    assert torch.all(d >= 0)


# ----------------------------------------------------- behavior manifold ---

def test_fit_behavior_manifold_interpolates_nodes():
    dists = _curve_dists(8, 32)
    domain, coords, params = _loop_domain(8)
    behavior = fit_behavior_manifold(dists, params)
    # The fitted RBF passes through the Hellinger-mapped centroids.
    for i in range(8):
        got = behavior.eval_at(domain.embed(coords[i]))
        assert torch.allclose(got, to_hellinger(dists[i]), atol=1e-3)


def test_on_manifold_trajectory_scores_low():
    dists = _curve_dists(8, 32)
    domain, coords, params = _loop_domain(8)
    behavior = fit_behavior_manifold(dists, params)
    # A trajectory made of the node centroids sits exactly on the curve.
    on_manifold = trajectory_naturalness(dists, behavior, domain, coords)
    assert torch.all(on_manifold < 1e-2)


def test_off_manifold_trajectory_scores_higher():
    torch.manual_seed(0)
    dists = _curve_dists(8, 32)
    domain, coords, params = _loop_domain(8)
    behavior = fit_behavior_manifold(dists, params)
    on_manifold = trajectory_naturalness(dists, behavior, domain, coords).mean()
    # Random distributions are off the learned behavior manifold.
    random_traj = torch.softmax(torch.randn(8, 32), dim=-1)
    off_manifold = trajectory_naturalness(
        random_traj, behavior, domain, coords,
    ).mean()
    assert off_manifold > on_manifold
    assert off_manifold > 1e-2


# -------------------------------------------------------- mock-model path ---
#
# The behavior reads (centroid / trajectory) only care that the model returns a
# per-position ``.logits``; the shared ``FakeLogitsModel`` (conftest) supplies
# that wrapper, parametrized by a seeded-random ``logits_fn`` so the trajectory
# stays deterministic per token sequence.  ``CharTokenizer(mod=13)`` reproduces
# the old whitespace-free char-code tokenizer.


def _MockModel(vocab: int) -> FakeLogitsModel:
    def _logits(input_ids: torch.Tensor) -> torch.Tensor:
        seq = input_ids.shape[1]
        torch.manual_seed(int(input_ids.sum()))
        return torch.randn(1, seq, vocab)

    # No ``parameters()`` consumer on this path (the behavior reads pass an
    # explicit device), so keep the stub minimal.
    return FakeLogitsModel(_logits, with_parameters=False)


def _MockTok() -> CharTokenizer:
    return CharTokenizer(mod=13)


def test_compute_node_behavior_centroid_is_distribution():
    model = _MockModel(24)
    centroid = compute_node_behavior_centroid(
        model, _MockTok(), torch.device("cpu"),
        ["calm one", "calm two", "calm three"],
    )
    assert centroid.shape == (24,)
    assert centroid.sum().item() == __import__("pytest").approx(1.0, abs=1e-4)
    assert torch.all(centroid >= 0)


def test_compute_trajectory_distributions_shape():
    model = _MockModel(24)
    traj = compute_trajectory_distributions(
        model, _MockTok(), torch.device("cpu"), "a steered sentence",
    )
    assert traj.shape[1] == 24
    assert traj.shape[0] > 0
    # Every row is a valid distribution.
    assert torch.allclose(traj.sum(dim=-1), torch.ones(traj.shape[0]), atol=1e-4)


def test_naturalness_end_to_end_mock():
    # Fit a behavior manifold, score a trajectory drawn from the same
    # mock model — exercises the full metric path without a real LM.
    model = _MockModel(24)
    tok = _MockTok()
    groups = [["calm a", "calm b", "calm c"],
              ["warm a", "warm b", "warm c"],
              ["hot a", "hot b", "hot c"],
              ["cold a", "cold b", "cold c"]]
    centroids = torch.stack([
        compute_node_behavior_centroid(model, tok, torch.device("cpu"), g)
        for g in groups
    ])
    domain, coords, params = _loop_domain(len(groups))
    behavior = fit_behavior_manifold(centroids, params)
    traj = compute_trajectory_distributions(
        model, tok, torch.device("cpu"), "some generated text here",
    )
    scores = trajectory_naturalness(traj, behavior, domain, coords)
    assert scores.shape[0] == traj.shape[0]
    assert torch.all(scores >= 0)


def test_naturalness_preflight_does_not_hash_fitted_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json
    from argparse import Namespace

    import saklas.cli.runners as runners
    from saklas.io import packs
    from saklas.io.manifolds import (
        ManifoldFolder,
        create_manifold_folder,
    )
    from saklas.io.manifold_folder import canonical_manifold_sidecar_payload

    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0},
    ]}
    nodes = [
        {"label": label, "coords": [i / 2], "statements": [label]}
        for i, label in enumerate(("a", "b", "c"))
    ]
    folder, _ = create_manifold_folder("local", "behavior", "", domain, nodes)
    tensor = folder / "unrelated.safetensors"
    sidecar = folder / "unrelated.json"
    tensor.write_bytes(b"unrelated fitted payload")
    sidecar.write_text(json.dumps(canonical_manifold_sidecar_payload(
        name="behavior", method="manifold_pca", saklas_version="0",
        domain=domain, node_labels=["a", "b", "c"], feature_space="raw",
        fit_mode="authored",
    )))
    ManifoldFolder.load(folder, verify_manifest=False).update_file_hashes(
        tensor, sidecar,
    )
    packs._FINGERPRINT_CACHE.clear()
    real_hash = packs.hash_file
    hashed: list[Path] = []

    def track_hash(path: Path) -> str:
        hashed.append(Path(path))
        return real_hash(Path(path))

    class StopAfterPreflight(Exception):
        pass

    def stop(_args: Namespace) -> None:
        raise StopAfterPreflight

    monkeypatch.setattr(packs, "hash_file", track_hash)
    monkeypatch.setattr(runners, "_load_effective_config", lambda _args: None)
    monkeypatch.setattr(runners, "_print_startup", stop)
    with pytest.raises(StopAfterPreflight):
        runners._run_experiment_naturalness(Namespace(manifold=str(folder)))

    assert hashed == []
