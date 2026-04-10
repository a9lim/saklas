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
    model, tokenizer = load_model(MODEL_ID, quantize=None, device="cuda", no_compile=True)
    return model, tokenizer


@pytest.fixture(scope="module")
def layers(model_and_tokenizer):
    from steer.model import get_layers
    model, _ = model_and_tokenizer
    return get_layers(model)


@pytest.fixture(scope="module")
def num_layers(layers):
    return len(layers)


@pytest.fixture(scope="module")
def middle_layer(num_layers):
    return num_layers // 2


@pytest.fixture(scope="module")
def happy_vector(model_and_tokenizer, middle_layer):
    from steer.vectors import extract_actadd
    model, tokenizer = model_and_tokenizer
    return extract_actadd(model, tokenizer, "happy", middle_layer)


class TestVectorExtraction:
    def test_actadd_returns_valid_vector(self, happy_vector, model_and_tokenizer):
        model, _ = model_and_tokenizer
        hidden_dim = model.config.hidden_size
        assert happy_vector.shape == (hidden_dim,)
        norm = happy_vector.norm().item()
        assert norm > 0 and not math.isinf(norm) and not math.isnan(norm)

    def test_actadd_fast_enough(self, model_and_tokenizer, middle_layer):
        """Single ActAdd extraction should complete in under 10 seconds."""
        from steer.vectors import extract_actadd
        model, tokenizer = model_and_tokenizer
        start = time.perf_counter()
        extract_actadd(model, tokenizer, "curious", middle_layer)
        elapsed = time.perf_counter() - start
        assert elapsed < 10.0, f"ActAdd took {elapsed:.1f}s, expected < 10s"


class TestSteering:
    def test_steered_output_differs(self, model_and_tokenizer, layers, middle_layer, happy_vector):
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
        mgr.add_vector("happy", happy_vector, 1.5, middle_layer)
        mgr.apply_to_model(layers, device, dtype)

        state1 = GenerationState()
        ids1 = generate_steered(model, tokenizer, input_ids.clone(), config, state1)

        mgr.clear_all()

        assert ids0 != ids1, "Steered output should differ from unsteered"

    def test_hook_cleanup(self, model_and_tokenizer, layers, middle_layer, happy_vector):
        from steer.hooks import SteeringManager
        from steer.generation import GenerationConfig, GenerationState, generate_steered

        model, tokenizer = model_and_tokenizer
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Hello"}],
            add_generation_prompt=True, return_tensors="pt",
        ).to(device)
        config = GenerationConfig(max_new_tokens=10, temperature=0.0)

        mgr = SteeringManager()
        mgr.add_vector("happy", happy_vector, 2.0, middle_layer)
        mgr.apply_to_model(layers, device, dtype)
        state_s = GenerationState()
        steered = generate_steered(model, tokenizer, input_ids.clone(), config, state_s)

        mgr.clear_all()
        state_c = GenerationState()
        clean = generate_steered(model, tokenizer, input_ids.clone(), config, state_c)

        assert steered != clean, "Output after hook cleanup should differ from steered"


class TestSaveLoad:
    def test_roundtrip(self, happy_vector):
        from steer.vectors import save_vector, load_vector

        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "test_vec.safetensors")
            meta = {"concept": "happy", "method": "actadd", "layer_idx": 10}
            save_vector(happy_vector, path, meta)
            loaded_vec, loaded_meta = load_vector(path)

            assert torch.allclose(happy_vector.cpu(), loaded_vec.cpu(), atol=1e-6)
            assert loaded_meta["concept"] == "happy"


class TestTraitMonitor:
    def test_monitor_records_history(self, model_and_tokenizer, layers, middle_layer, happy_vector):
        from steer.vectors import extract_actadd
        from steer.hooks import SteeringManager
        from steer.monitor import TraitMonitor
        from steer.generation import GenerationConfig, GenerationState, generate_steered

        model, tokenizer = model_and_tokenizer
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        num_layers = len(layers)

        sad_vector = extract_actadd(model, tokenizer, "sad", num_layers - 2)

        probes = {"happy": happy_vector, "sad": sad_vector}
        monitor = TraitMonitor(probes, num_layers - 2)
        monitor.attach(layers, device, dtype)

        # Steer toward happy
        mgr = SteeringManager()
        mgr.add_vector("happy", happy_vector, 1.0, middle_layer)
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

    def test_throughput_regression(self, model_and_tokenizer, layers, middle_layer, happy_vector):
        """Steered generation should be at least 85% of vanilla throughput."""
        from steer.vectors import extract_actadd
        from steer.hooks import SteeringManager
        from steer.monitor import TraitMonitor
        from steer.generation import GenerationConfig, GenerationState, generate_steered

        model, tokenizer = model_and_tokenizer
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        num_layers = len(layers)

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
        mgr.add_vector("happy", happy_vector, 0.8, middle_layer)
        curious_vec = extract_actadd(model, tokenizer, "curious", middle_layer)
        mgr.add_vector("curious", curious_vec, 0.5, middle_layer)
        concise_vec = extract_actadd(model, tokenizer, "concise", middle_layer)
        mgr.add_vector("concise", concise_vec, 0.3, middle_layer + 2)
        mgr.apply_to_model(layers, device, dtype)

        # 15 probes (use same vectors repeated for simplicity)
        probe_dict = {}
        for i, name in enumerate(["p0", "p1", "p2", "p3", "p4", "p5", "p6",
                                    "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14"]):
            probe_dict[name] = [happy_vector, curious_vec, concise_vec][i % 3]
        monitor = TraitMonitor(probe_dict, num_layers - 2)
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
