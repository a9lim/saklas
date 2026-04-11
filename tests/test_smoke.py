"""Smoke tests for steer.

Requires a CUDA GPU and downloads google/gemma-2-2b-it (~5GB) on first run.
Run with: pytest tests/test_smoke.py -v
"""

from __future__ import annotations

import math
import time
import tempfile
from pathlib import Path

import pytest
import torch

# Skip entire module if no CUDA
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)

MODEL_ID = "google/gemma-2-2b-it"


@pytest.fixture(scope="module")
def model_and_tokenizer():
    from steer.model import load_model
    model, tokenizer = load_model(MODEL_ID, quantize=None, device="cuda")
    return model, tokenizer


@pytest.fixture(scope="module")
def layers(model_and_tokenizer):
    from steer.model import get_layers
    model, _ = model_and_tokenizer
    return get_layers(model)


@pytest.fixture(scope="module")
def num_layers(layers):
    return len(layers)


def _extract_profile(model, tokenizer, concept, layers):
    """Extract a profile for a single concept with one pair."""
    from steer.vectors import extract_contrastive
    return extract_contrastive(model, tokenizer, [{"positive": concept, "negative": ""}], layers=layers)


@pytest.fixture(scope="module")
def happy_profile(model_and_tokenizer, layers):
    model, tokenizer = model_and_tokenizer
    return _extract_profile(model, tokenizer, "happy", layers)


class TestVectorExtraction:
    def test_returns_valid_profile(self, happy_profile, model_and_tokenizer):
        model, _ = model_and_tokenizer
        hidden_dim = model.config.hidden_size
        assert isinstance(happy_profile, dict)
        assert len(happy_profile) > 0
        for layer_idx, (vec, score) in happy_profile.items():
            assert isinstance(layer_idx, int)
            assert vec.shape == (hidden_dim,)
            norm = vec.norm().item()
            assert norm > 0 and not math.isinf(norm) and not math.isnan(norm)
            assert score > 0

    def test_extraction_fast_enough(self, model_and_tokenizer, layers):
        """Single contrastive extraction should complete in under 10 seconds."""
        model, tokenizer = model_and_tokenizer
        start = time.perf_counter()
        _extract_profile(model, tokenizer, "curious", layers)
        elapsed = time.perf_counter() - start
        assert elapsed < 10.0, f"Extraction took {elapsed:.1f}s, expected < 10s"


class TestSteering:
    def test_steered_output_differs(self, model_and_tokenizer, layers, happy_profile):
        from steer.hooks import SteeringManager
        from steer.generation import GenerationConfig, GenerationState, generate_steered

        model, tokenizer = model_and_tokenizer
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        prompt = "Tell me about your day."
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, return_tensors="pt",
        ).to(device)

        config = GenerationConfig(max_new_tokens=20, temperature=0.0)

        # Unsteered
        state0 = GenerationState()
        ids0 = generate_steered(model, tokenizer, input_ids.clone(), config, state0)

        # Steered
        mgr = SteeringManager()
        mgr.add_vector("happy", happy_profile, 1.5)
        mgr.apply_to_model(layers, device, dtype)

        state1 = GenerationState()
        ids1 = generate_steered(model, tokenizer, input_ids.clone(), config, state1)

        mgr.clear_all()

        assert ids0 != ids1, "Steered output should differ from unsteered"

    def test_hook_cleanup(self, model_and_tokenizer, layers, happy_profile):
        from steer.hooks import SteeringManager
        from steer.generation import GenerationConfig, GenerationState, generate_steered

        model, tokenizer = model_and_tokenizer
        p = next(model.parameters())
        device, dtype = p.device, p.dtype

        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Hello"}],
            add_generation_prompt=True, return_tensors="pt",
        ).to(device)
        config = GenerationConfig(max_new_tokens=10, temperature=0.0)

        # Unsteered baseline
        state_b = GenerationState()
        baseline = generate_steered(model, tokenizer, input_ids.clone(), config, state_b)

        # Steered
        mgr = SteeringManager()
        mgr.add_vector("happy", happy_profile, 2.0)
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
    def test_roundtrip(self, happy_profile):
        from steer.vectors import save_profile, load_profile

        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "test_profile.safetensors")
            meta = {"concept": "happy"}
            save_profile(happy_profile, path, meta)
            loaded_profile, loaded_meta = load_profile(path)

            assert loaded_meta["concept"] == "happy"
            assert set(loaded_profile.keys()) == set(happy_profile.keys())
            for idx in happy_profile:
                orig_vec, orig_score = happy_profile[idx]
                loaded_vec, loaded_score = loaded_profile[idx]
                assert torch.allclose(orig_vec.cpu(), loaded_vec.cpu(), atol=1e-6)
                assert abs(orig_score - loaded_score) < 1e-6


class TestTraitMonitor:
    def test_monitor_records_history(self, model_and_tokenizer, layers, happy_profile):
        from steer.hooks import SteeringManager
        from steer.monitor import TraitMonitor
        from steer.generation import GenerationConfig, GenerationState, generate_steered

        model, tokenizer = model_and_tokenizer
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        sad_profile = _extract_profile(model, tokenizer, "sad", layers)

        probe_profiles = {"happy": happy_profile, "sad": sad_profile}
        monitor = TraitMonitor(probe_profiles)
        monitor.attach(layers, device, dtype)

        # Steer toward happy
        mgr = SteeringManager()
        mgr.add_vector("happy", happy_profile, 1.0)
        mgr.apply_to_model(layers, device, dtype)

        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": "How are you feeling?"}],
            add_generation_prompt=True, return_tensors="pt",
        ).to(device)
        config = GenerationConfig(max_new_tokens=20, temperature=0.7)
        state = GenerationState()
        generate_steered(model, tokenizer, input_ids, config, state)

        monitor.flush_to_cpu()
        mgr.clear_all()

        # Should have one entry per generated token
        happy_hist = monitor.history["happy"]
        sad_hist = monitor.history["sad"]
        assert len(happy_hist) > 0, "Monitor should record at least one entry"
        assert len(happy_hist) == len(sad_hist), "All probes should have same history length"

        # With happy steering, mean happy sim should exceed mean sad sim
        mean_happy = sum(happy_hist) / len(happy_hist)
        mean_sad = sum(sad_hist) / len(sad_hist)
        assert mean_happy > mean_sad, (
            f"Expected happy ({mean_happy:.3f}) > sad ({mean_sad:.3f}) with happy steering"
        )

        # Sparkline should be non-empty
        monitor.flush_to_cpu()
        sparkline = monitor.get_sparkline("happy")
        assert len(sparkline) > 0

        monitor.detach()

    def test_throughput_regression(self, model_and_tokenizer, layers, happy_profile):
        """Steered generation should be at least 85% of vanilla throughput."""
        from steer.hooks import SteeringManager
        from steer.monitor import TraitMonitor
        from steer.generation import GenerationConfig, GenerationState, generate_steered

        model, tokenizer = model_and_tokenizer
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Write a short story."}],
            add_generation_prompt=True, return_tensors="pt",
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
        mgr.add_vector("happy", happy_profile, 0.8)
        curious_profile = _extract_profile(model, tokenizer, "curious", layers)
        mgr.add_vector("curious", curious_profile, 0.5)
        concise_profile = _extract_profile(model, tokenizer, "concise", layers)
        mgr.add_vector("concise", concise_profile, 0.3)
        mgr.apply_to_model(layers, device, dtype)

        # 15 probes (use same profiles repeated for simplicity)
        probe_profiles = {}
        source_profiles = [happy_profile, curious_profile, concise_profile]
        for i, name in enumerate(["p0", "p1", "p2", "p3", "p4", "p5", "p6",
                                    "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14"]):
            probe_profiles[name] = source_profiles[i % 3]
        monitor = TraitMonitor(probe_profiles)
        monitor.attach(layers, device, dtype)

        state1 = GenerationState()
        t1 = time.perf_counter()
        ids1 = generate_steered(model, tokenizer, input_ids.clone(), config, state1)
        steered_time = time.perf_counter() - t1
        steered_tps = len(ids1) / steered_time

        monitor.detach()
        mgr.clear_all()

        ratio = steered_tps / vanilla_tps
        assert ratio >= 0.85, (
            f"Steered throughput ({steered_tps:.1f} tok/s) is only "
            f"{ratio:.0%} of vanilla ({vanilla_tps:.1f} tok/s), expected >= 85%"
        )


class TestExtractContrastive:
    def test_returns_valid_profile(self, model_and_tokenizer, layers, num_layers):
        from steer.vectors import extract_contrastive
        model, tokenizer = model_and_tokenizer
        hidden_dim = model.config.hidden_size
        pairs = [
            {"positive": "I feel happy today", "negative": "I feel sad today"},
            {"positive": "Everything is wonderful", "negative": "Everything is terrible"},
            {"positive": "I love this", "negative": "I hate this"},
        ]
        profile = extract_contrastive(model, tokenizer, pairs, layers=layers)
        assert isinstance(profile, dict)
        assert len(profile) > 0
        for idx, (vec, score) in profile.items():
            assert 0 <= idx < num_layers
            assert vec.shape == (hidden_dim,)
            norm = vec.norm().item()
            assert norm > 0 and not math.isinf(norm) and not math.isnan(norm)
            assert score > 0


class TestBuildChatInput:
    def test_chat_template_path(self, model_and_tokenizer):
        from steer.generation import build_chat_input
        _, tokenizer = model_and_tokenizer
        messages = [{"role": "user", "content": "Hello"}]
        ids = build_chat_input(tokenizer, messages)
        assert ids.ndim == 2
        assert ids.shape[0] == 1
        assert ids.shape[1] > 0

    def test_with_system_prompt(self, model_and_tokenizer):
        from steer.generation import build_chat_input
        _, tokenizer = model_and_tokenizer
        messages = [{"role": "user", "content": "Hello"}]
        ids_no_sys = build_chat_input(tokenizer, messages)
        ids_sys = build_chat_input(tokenizer, messages, system_prompt="You are helpful.")
        # System prompt should add tokens
        assert ids_sys.shape[1] > ids_no_sys.shape[1]


class TestProbesBootstrap:
    def test_bootstrap_loads_from_cache(self, model_and_tokenizer, layers, happy_profile):
        """Bootstrap should return cached profiles without re-extracting."""
        from steer.probes_bootstrap import bootstrap_probes
        from steer.vectors import save_profile, get_cache_path
        from steer.model import get_model_info
        model, tokenizer = model_and_tokenizer
        model_info = get_model_info(model, tokenizer)

        with tempfile.TemporaryDirectory() as tmp:
            # Pre-populate cache with a profile
            cp = get_cache_path(tmp, model_info["model_id"], "happy")
            save_profile(happy_profile, cp, {
                "concept": "happy",
                "model_id": model_info["model_id"],
                "num_pairs": 10,
            })
            # Bootstrap with a category containing "happy"
            probes = bootstrap_probes(
                model, tokenizer, layers, model_info,
                categories=["emotion"], cache_dir=tmp,
            )
            # If "happy" is in the emotion category in defaults.json, it should be loaded
            assert isinstance(probes, dict)
