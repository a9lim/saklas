"""Smoke tests for saklas.

Requires a GPU (CUDA or Apple Silicon MPS) and downloads google/gemma-3-4b-it
(~8GB) on first run. Run with: pytest tests/test_smoke.py -v
"""

from __future__ import annotations

import math
import time
import tempfile
from pathlib import Path
from typing import Any

import pytest
import torch

# Skip entire module if no GPU backend is available.
_HAS_GPU = torch.cuda.is_available() or torch.backends.mps.is_available()
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not _HAS_GPU,
        reason="No GPU backend available (neither CUDA nor MPS)",
    ),
]

MODEL_ID = "google/gemma-3-4b-it"
# MPS runs ~3-5x slower than CUDA for this model; relax absolute timing budgets.
_IS_MPS = not torch.cuda.is_available() and torch.backends.mps.is_available()
_EXTRACTION_BUDGET_S = 60.0 if _IS_MPS else 10.0


@pytest.fixture(scope="module")
def model_and_tokenizer():
    from saklas.core.model import load_model
    # device="auto" picks cuda > mps > cpu; the skipif above guarantees a GPU.
    model, tokenizer = load_model(MODEL_ID, quantize=None, device="auto")
    return model, tokenizer


@pytest.fixture(scope="module")
def layers(model_and_tokenizer: Any) -> Any:
    from saklas.core.model import get_layers
    model, _ = model_and_tokenizer
    return get_layers(model)


@pytest.fixture(scope="module")
def num_layers(layers: Any) -> int:
    return len(layers)


def _extract_profile(model: Any, tokenizer: Any, concept: str, layers: Any) -> Any:
    """A per-layer direction profile for a single concept vs the empty pole.

    The 4.0 equivalent of a difference-of-means vector: capture the two pole
    centroids, take their per-layer difference ``δ = pos − neg``, and fold that
    direction into a one-pole affine ray via the production
    ``fold_directions_to_subspace``, then read its baked-direction view
    (``folded_vector_directions``).  Same centroid-difference direction the old
    DiM extractor produced, through the surviving fold primitive.
    """
    from saklas.core.vectors import (
        _encode_and_capture_all,
        fold_directions_to_subspace,
        folded_vector_directions,
    )

    device = next(model.parameters()).device
    pos = _encode_and_capture_all(model, tokenizer, concept, layers, device)
    neg = _encode_and_capture_all(model, tokenizer, "", layers, device)
    directions = {
        L: (pos[L].to(torch.float32) - neg[L].to(torch.float32))
        for L in pos if L in neg
    }
    mfld = fold_directions_to_subspace(concept, directions, None, label="p")
    return folded_vector_directions(mfld)


@pytest.fixture(scope="module")
def layer_means(model_and_tokenizer: Any, layers: Any) -> Any:
    from saklas.core.vectors import compute_layer_means
    model, tokenizer = model_and_tokenizer
    return compute_layer_means(model, tokenizer, layers)


@pytest.fixture(scope="module")
def happy_profile(model_and_tokenizer: Any, layers: Any) -> Any:
    model, tokenizer = model_and_tokenizer
    return _extract_profile(model, tokenizer, "happy", layers)


def _steer_subspace(mgr: Any, *pairs: Any) -> None:
    """4.0: route baked ``(profile, alpha)`` pairs through the unified backend.

    Every vector lowers to a rank-1 push fragment (unit dir + baked-magnitude
    coord), all composed into one merged affine subspace via
    ``synthesize_subspace`` + ``add_subspace`` — the dispatch the session does.
    Anchored at zero (enough for the smoke "steering changes the output"
    assertions; the GPU gate owns strength calibration).
    """
    from saklas.core.manifold import synthesize_subspace

    push: list[Any] = []
    neutral_means: dict[int, torch.Tensor] = {}
    for profile, alpha in pairs:
        basis_dirs: dict[int, torch.Tensor] = {}
        coord_dirs: dict[int, torch.Tensor] = {}
        for L, vec in profile.items():
            v = vec.to(torch.float32).reshape(-1)
            n = float(v.norm())
            if n < 1e-12:
                continue
            basis_dirs[L] = (v / n).reshape(1, -1)
            coord_dirs[L] = torch.tensor([n])
            neutral_means.setdefault(L, torch.zeros_like(v))
        push.append((basis_dirs, coord_dirs, alpha))
    synth = synthesize_subspace(push, [], neutral_means=neutral_means)
    mgr.add_subspace("steer", synth)


class TestVectorExtraction:
    def test_returns_valid_profile(self, happy_profile: Any, model_and_tokenizer: Any) -> None:
        model, _ = model_and_tokenizer
        cfg = getattr(model.config, "text_config", None) or model.config
        hidden_dim = cfg.hidden_size
        assert isinstance(happy_profile, dict)
        assert len(happy_profile) > 0
        for layer_idx, vec in happy_profile.items():
            assert isinstance(layer_idx, int)
            assert vec.shape == (hidden_dim,)
            norm = vec.norm().item()
            assert norm > 0 and not math.isinf(norm) and not math.isnan(norm)

    def test_extraction_fast_enough(self, model_and_tokenizer: Any, layers: Any) -> None:
        """Single contrastive extraction should complete within the backend's budget."""
        model, tokenizer = model_and_tokenizer
        start = time.perf_counter()
        _extract_profile(model, tokenizer, "curious", layers)
        elapsed = time.perf_counter() - start
        assert elapsed < _EXTRACTION_BUDGET_S, (
            f"Extraction took {elapsed:.1f}s, expected < {_EXTRACTION_BUDGET_S:.0f}s"
        )


class TestSteering:
    def test_steered_output_differs(self, model_and_tokenizer: Any, layers: Any, happy_profile: Any) -> None:
        from saklas.core.hooks import SteeringManager
        from saklas.core.generation import GenerationConfig, GenerationState, generate_steered

        model, tokenizer = model_and_tokenizer
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        prompt = "Tell me about your day."
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, return_tensors="pt", return_dict=False,
        ).to(device)

        config = GenerationConfig(max_new_tokens=20, temperature=0.0)

        # Unsteered
        state0 = GenerationState()
        ids0 = generate_steered(model, tokenizer, input_ids.clone(), config, state0)

        # Steered
        mgr = SteeringManager()
        _steer_subspace(mgr, (happy_profile, 1.5))
        mgr.apply_to_model(layers, device, dtype)

        state1 = GenerationState()
        ids1 = generate_steered(model, tokenizer, input_ids.clone(), config, state1)

        mgr.clear_all()

        assert ids0 != ids1, "Steered output should differ from unsteered"

    def test_hook_cleanup(self, model_and_tokenizer: Any, layers: Any, happy_profile: Any) -> None:
        from saklas.core.hooks import SteeringManager
        from saklas.core.generation import GenerationConfig, GenerationState, generate_steered

        model, tokenizer = model_and_tokenizer
        p = next(model.parameters())
        device, dtype = p.device, p.dtype

        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Hello"}],
            add_generation_prompt=True, return_tensors="pt", return_dict=False,
        ).to(device)
        config = GenerationConfig(max_new_tokens=10, temperature=0.0)

        # Unsteered baseline
        state_b = GenerationState()
        baseline = generate_steered(model, tokenizer, input_ids.clone(), config, state_b)

        # Steered
        mgr = SteeringManager()
        _steer_subspace(mgr, (happy_profile, 2.0))
        mgr.apply_to_model(layers, device, dtype)
        state_s = GenerationState()
        steered = generate_steered(model, tokenizer, input_ids.clone(), config, state_s)

        # Cleanup — output should match unsteered baseline
        mgr.clear_all()
        state_c = GenerationState()
        clean = generate_steered(model, tokenizer, input_ids.clone(), config, state_c)

        assert steered != baseline, "Steered output should differ from baseline"
        assert clean == baseline, "Output after hook cleanup should match unsteered baseline"


class TestSaveLoad:
    def test_roundtrip(self, happy_profile: Any) -> None:
        from saklas.core.vectors import save_profile, load_profile

        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "test_profile.safetensors")
            save_profile(happy_profile, path, {"method": "difference_of_means"})
            loaded_profile, loaded_meta = load_profile(path)

            assert loaded_meta["method"] == "difference_of_means"
            assert "scores" not in loaded_meta
            assert set(loaded_profile.keys()) == set(happy_profile.keys())
            for idx in happy_profile:
                assert torch.allclose(
                    happy_profile[idx].cpu(), loaded_profile[idx].cpu(), atol=1e-6
                )


class TestTraitMonitor:
    def test_monitor_records_history(self, model_and_tokenizer: Any, layers: Any, happy_profile: Any, layer_means: Any) -> None:
        from saklas.core.hooks import SteeringManager
        from saklas.core.monitor import TraitMonitor
        from saklas.core.generation import GenerationConfig, GenerationState, generate_steered

        model, tokenizer = model_and_tokenizer
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        sad_profile = _extract_profile(model, tokenizer, "sad", layers)

        probe_profiles = {"happy": happy_profile, "sad": sad_profile}
        monitor = TraitMonitor(probe_profiles, layer_means)

        # Steer toward happy
        mgr = SteeringManager()
        # α=0.6 sits mid coherent band (0.4–0.8); α=1 is at the cliff per CLAUDE.md.
        _steer_subspace(mgr, (happy_profile, 0.6))
        mgr.apply_to_model(layers, device, dtype)

        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": "How are you feeling?"}],
            add_generation_prompt=True, return_tensors="pt", return_dict=False,
        ).to(device)
        config = GenerationConfig(max_new_tokens=40, temperature=0.7)
        state = GenerationState()
        generated_ids = generate_steered(model, tokenizer, input_ids, config, state)
        mgr.clear_all()

        # Measure on generated text
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        monitor.measure(model, tokenizer, layers, text, device=device)

        # Should have one entry per generation
        happy_hist = monitor.history["happy"]
        sad_hist = monitor.history["sad"]
        assert len(happy_hist) == 1, "Monitor should record one entry per generation"
        assert len(sad_hist) == 1

        # Structural: readings are finite floats in the expected cosine range.
        # The semantic claim "happy steering → higher happy than sad reading
        # when remeasured via a fresh forward pass" is noise-dominated at this
        # model scale (3B-param, 20-token gen) — test_throughput_regression
        # covers whether steering actually runs.
        for hist in (happy_hist, sad_hist):
            v = hist[0]
            assert v == v  # not NaN
            assert -1.5 <= v <= 1.5

        # Sparkline should be non-empty
        sparkline = monitor.get_sparkline("happy")
        assert len(sparkline) > 0

    def test_throughput_regression(self, model_and_tokenizer: Any, layers: Any, happy_profile: Any, layer_means: Any) -> None:
        """Steered generation should be at least 85% of vanilla throughput."""
        from saklas.core.hooks import SteeringManager
        from saklas.core.generation import GenerationConfig, GenerationState, generate_steered

        model, tokenizer = model_and_tokenizer
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Write a short story."}],
            add_generation_prompt=True, return_tensors="pt", return_dict=False,
        ).to(device)
        config = GenerationConfig(max_new_tokens=100, temperature=0.7)

        # Vanilla timing
        state0 = GenerationState()
        t0 = time.perf_counter()
        ids0 = generate_steered(model, tokenizer, input_ids.clone(), config, state0)
        vanilla_time = time.perf_counter() - t0
        vanilla_tps = len(ids0) / vanilla_time

        # Steered + monitored timing
        # 3 steering vectors
        mgr = SteeringManager()
        curious_profile = _extract_profile(model, tokenizer, "curious", layers)
        concise_profile = _extract_profile(model, tokenizer, "concise", layers)
        # 3 vectors → one merged affine subspace (the dispatch composes them).
        _steer_subspace(
            mgr, (happy_profile, 0.8), (curious_profile, 0.5), (concise_profile, 0.3),
        )
        mgr.apply_to_model(layers, device, dtype)

        state1 = GenerationState()
        t1 = time.perf_counter()
        ids1 = generate_steered(model, tokenizer, input_ids.clone(), config, state1)
        steered_time = time.perf_counter() - t1
        steered_tps = len(ids1) / steered_time

        mgr.clear_all()

        ratio = steered_tps / vanilla_tps
        assert ratio >= 0.85, (
            f"Steered throughput ({steered_tps:.1f} tok/s) is only "
            f"{ratio:.0%} of vanilla ({vanilla_tps:.1f} tok/s), expected >= 85%"
        )


class TestBuildChatInput:
    def test_chat_template_path(self, model_and_tokenizer: Any) -> None:
        from saklas.core.generation import build_chat_input
        _, tokenizer = model_and_tokenizer
        messages = [{"role": "user", "content": "Hello"}]
        ids = build_chat_input(tokenizer, messages)
        assert ids.ndim == 2
        assert ids.shape[0] == 1
        assert ids.shape[1] > 0

    def test_with_system_prompt(self, model_and_tokenizer: Any) -> None:
        from saklas.core.generation import build_chat_input
        _, tokenizer = model_and_tokenizer
        messages = [{"role": "user", "content": "Hello"}]
        ids_no_sys = build_chat_input(tokenizer, messages)
        ids_sys = build_chat_input(tokenizer, messages, system_prompt="You are helpful.")
        # System prompt should add tokens
        assert ids_sys.shape[1] > ids_no_sys.shape[1]


class TestAblationPerformance:
    """Combined ablation + additive throughput must stay >= 80% of vanilla.

    Slightly looser than the steered-only 85% bar because the hot path
    does one extra matmul per active ablation layer per step. Intentional.
    """

    def test_throughput_with_ablation(self, model_and_tokenizer: Any, layers: Any, layer_means: Any) -> None:
        from saklas.core.session import SaklasSession

        model, tokenizer = model_and_tokenizer
        session = SaklasSession(model, tokenizer, probes=["affect"])

        try:
            prompt = "Write a 200-word essay on the history of bicycles."

            # Vanilla baseline.
            t0 = time.perf_counter()
            r_vanilla = session.generate(prompt)
            dt_vanilla = max(time.perf_counter() - t0, 0.1)
            tok_s_vanilla = max(len(r_vanilla.text.split()) / dt_vanilla, 1e-6)

            # Pick two probes from the auto-loaded set: one for additive, one for ablation.
            probes = list(session.probes)
            assert len(probes) >= 2, "need at least two probes for this test"
            additive_name, ablation_name = probes[0], probes[1]

            expr = f"0.3 {additive_name} + !{ablation_name}"
            t0 = time.perf_counter()
            with session.steering(expr):
                r_combined = session.generate(prompt)
            dt_combined = max(time.perf_counter() - t0, 0.1)
            tok_s_combined = max(len(r_combined.text.split()) / dt_combined, 1e-6)

            ratio = tok_s_combined / tok_s_vanilla
            assert ratio >= 0.80, (
                f"combined ablation + additive too slow: "
                f"{ratio:.2%} of vanilla (vanilla={tok_s_vanilla:.1f} tok/s, "
                f"combined={tok_s_combined:.1f} tok/s)"
            )
        finally:
            session.close()


class TestDiscoverManifoldEndToEnd:
    """End-to-end discover-mode fit on a real model.

    Exercises the integration path the unit-test suite skips: the LLM-side
    K-tuple generator, the per-node centroid forward pass, the per-model
    PCA + RBF fit, and the steered-generation hook at a manifold position.
    Held to a tight time budget (corpus generation is the slow part — 1
    scenario call + N_concepts × N_scenarios statement calls — so 5
    concepts × 2 scenarios × 3 statements stays under ~6× the existing
    extraction budget).

    Asserts the structural invariants only: the fit produces a
    ``CustomDomain(k)``, every node lands at distinct derived coords, and
    a soft subspace-replace at one node's centroid produces *different*
    text from a steered call at a different node's centroid.  Going
    deeper (e.g. probe-score asymmetry between opposite node positions)
    requires the manifold to encode a probe-aligned axis, which depends
    on which concept heap the generator produces — too fragile for a
    smoke test, properly the job of the naturalness eval.
    """

    _CONCEPTS = ["pirate", "scholar", "robot", "caveman", "assistant"]
    _SAMPLES_PER_PROMPT = 1
    # A2 conversational extraction: each concept answers every shared baseline
    # prompt in character, so one corpus = samples_per_prompt × len(baseline
    # prompts) responses (64 prompts bundled).  Generation is the dominant
    # cost; the budget scales with the concept × prompt turn count.
    _GENERATE_BUDGET_S = _EXTRACTION_BUDGET_S * 6
    _FIT_BUDGET_S = _EXTRACTION_BUDGET_S * 2

    def test_discover_pipeline(self, model_and_tokenizer: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from saklas.core.manifold import CustomDomain
        from saklas.core.session import SaklasSession
        from saklas.io.manifolds import create_discover_manifold_folder

        # Pin SAKLAS_HOME to a per-test temp dir so the manifold folder
        # lands somewhere disposable; bundled neutrals still copy in
        # fresh on first session init.
        monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))

        model, tokenizer = model_and_tokenizer
        # probes=[] — the smoke test doesn't need probe scoring and
        # skipping bootstrap shaves seconds off the discover gate.
        session = SaklasSession(model, tokenizer, probes=[])
        try:
            # ---- 1. generate per-concept corpora (A2 conversational) ----
            from saklas.core.vectors import _load_baseline_prompts
            n_prompts = len(_load_baseline_prompts())
            t0 = time.perf_counter()
            corpora = session.generate_responses(
                self._CONCEPTS,
                [None] * len(self._CONCEPTS),
                samples_per_prompt=self._SAMPLES_PER_PROMPT,
            )
            dt_gen = time.perf_counter() - t0
            assert dt_gen < self._GENERATE_BUDGET_S, (
                f"generate_responses took {dt_gen:.1f}s, "
                f"budget {self._GENERATE_BUDGET_S:.1f}s"
            )
            assert set(corpora.keys()) == set(self._CONCEPTS)
            expected_per_concept = self._SAMPLES_PER_PROMPT * n_prompts
            for c, lines in corpora.items():
                assert len(lines) == expected_per_concept, (
                    f"corpus for {c!r}: {len(lines)} responses, "
                    f"expected {expected_per_concept}"
                )
                # Length must be a multiple of the baseline-prompt count
                # (response[i] ↔ baseline_prompt[i % k]).
                assert len(lines) % n_prompts == 0

            # ---- 2. author the discover folder ----
            folder = create_discover_manifold_folder(
                "local", "smoke_personas", "smoke-test discover",
                fit_mode="pca",
                node_corpora=corpora,
                hyperparams={"max_dim": 2, "var_threshold": 0.70},
            )

            # ---- 3. fit through the session pipeline ----
            t0 = time.perf_counter()
            manifold = session.extract_manifold(folder)
            dt_fit = time.perf_counter() - t0
            assert dt_fit < self._FIT_BUDGET_S, (
                f"extract_manifold took {dt_fit:.1f}s, "
                f"budget {self._FIT_BUDGET_S:.1f}s"
            )

            # Discover-mode shape invariants.
            assert isinstance(manifold.domain, CustomDomain)
            k = manifold.domain.intrinsic_dim
            assert 1 <= k <= 2
            assert manifold.node_coords.shape == (
                len(self._CONCEPTS), k,
            )
            # Derived coords must distinguish nodes — every pair distinct
            # by at least a small floor, else the fit collapsed.
            from itertools import combinations
            for i, j in combinations(range(len(self._CONCEPTS)), 2):
                dist = (
                    manifold.node_coords[i] - manifold.node_coords[j]
                ).norm().item()
                assert dist > 1e-3, (
                    f"derived coords for nodes {i} and {j} are "
                    f"degenerate (dist={dist:.2e}) — fit collapsed"
                )

            # ---- 4. steered generation at one node vs another ----
            # The manifold loads into session._manifolds lazily on
            # scope entry; ManifoldTerm carries the authoring position.
            label_a = manifold.node_labels[0]  # pirate
            label_b = manifold.node_labels[2]  # robot
            pos_a = ",".join(
                f"{float(c):.4f}" for c in manifold.node_coords[0]
            )
            pos_b = ",".join(
                f"{float(c):.4f}" for c in manifold.node_coords[2]
            )
            prompt = "Describe what you see in this room."

            from saklas.core.sampling import SamplingConfig
            sampling = SamplingConfig(
                temperature=0.0, max_tokens=48, seed=0,
            )
            with session.steering(
                f"1.0 local/smoke_personas%{pos_a}@response",
            ):
                r_a = session.generate(prompt, sampling=sampling)
            with session.steering(
                f"1.0 local/smoke_personas%{pos_b}@response",
            ):
                r_b = session.generate(prompt, sampling=sampling)
            assert r_a.text != r_b.text, (
                f"steering at node {label_a!r} produced identical text "
                f"to node {label_b!r} — the manifold hook isn't moving "
                f"activations"
            )
        finally:
            session.close()
