"""Smoke tests for saklas.

Requires a GPU (CUDA or Apple Silicon MPS) and downloads google/gemma-3-4b-it
(~8GB) on first run. Run with: pytest tests/test_smoke.py -v
"""

from __future__ import annotations

import math
import time
import tempfile
from pathlib import Path

import pytest
import torch

# Skip entire module if no GPU backend is available.
_HAS_GPU = torch.cuda.is_available() or torch.backends.mps.is_available()
pytestmark = pytest.mark.skipif(
    not _HAS_GPU,
    reason="No GPU backend available (neither CUDA nor MPS)",
)

MODEL_ID = "google/gemma-3-4b-it"
# MPS runs ~3-5x slower than CUDA for this model; relax absolute timing budgets.
_IS_MPS = not torch.cuda.is_available() and torch.backends.mps.is_available()
_EXTRACTION_BUDGET_S = 60.0 if _IS_MPS else 10.0


@pytest.fixture(scope="module")
def model_and_tokenizer():
    from saklas.model import load_model
    # device="auto" picks cuda > mps > cpu; the skipif above guarantees a GPU.
    model, tokenizer = load_model(MODEL_ID, quantize=None, device="auto")
    return model, tokenizer


@pytest.fixture(scope="module")
def layers(model_and_tokenizer):
    from saklas.model import get_layers
    model, _ = model_and_tokenizer
    return get_layers(model)


@pytest.fixture(scope="module")
def num_layers(layers):
    return len(layers)


def _extract_profile(model, tokenizer, concept, layers):
    """Extract a profile for a single concept with one pair."""
    from saklas.vectors import extract_contrastive
    return extract_contrastive(model, tokenizer, [{"positive": concept, "negative": ""}], layers=layers)


@pytest.fixture(scope="module")
def layer_means(model_and_tokenizer, layers):
    from saklas.vectors import compute_layer_means
    model, tokenizer = model_and_tokenizer
    return compute_layer_means(model, tokenizer, layers)


@pytest.fixture(scope="module")
def happy_profile(model_and_tokenizer, layers):
    model, tokenizer = model_and_tokenizer
    return _extract_profile(model, tokenizer, "happy", layers)


class TestVectorExtraction:
    def test_returns_valid_profile(self, happy_profile, model_and_tokenizer):
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

    def test_extraction_fast_enough(self, model_and_tokenizer, layers):
        """Single contrastive extraction should complete within the backend's budget."""
        model, tokenizer = model_and_tokenizer
        start = time.perf_counter()
        _extract_profile(model, tokenizer, "curious", layers)
        elapsed = time.perf_counter() - start
        assert elapsed < _EXTRACTION_BUDGET_S, (
            f"Extraction took {elapsed:.1f}s, expected < {_EXTRACTION_BUDGET_S:.0f}s"
        )


class TestSteering:
    def test_steered_output_differs(self, model_and_tokenizer, layers, happy_profile):
        from saklas.hooks import SteeringManager
        from saklas.generation import GenerationConfig, GenerationState, generate_steered

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
        mgr.add_vector("happy", happy_profile, 1.5)
        mgr.apply_to_model(layers, device, dtype)

        state1 = GenerationState()
        ids1 = generate_steered(model, tokenizer, input_ids.clone(), config, state1)

        mgr.clear_all()

        assert ids0 != ids1, "Steered output should differ from unsteered"

    def test_hook_cleanup(self, model_and_tokenizer, layers, happy_profile):
        from saklas.hooks import SteeringManager
        from saklas.generation import GenerationConfig, GenerationState, generate_steered

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
        from saklas.vectors import save_profile, load_profile

        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "test_profile.safetensors")
            save_profile(happy_profile, path, {"method": "contrastive_pca"})
            loaded_profile, loaded_meta = load_profile(path)

            assert loaded_meta["method"] == "contrastive_pca"
            assert "scores" not in loaded_meta
            assert set(loaded_profile.keys()) == set(happy_profile.keys())
            for idx in happy_profile:
                assert torch.allclose(
                    happy_profile[idx].cpu(), loaded_profile[idx].cpu(), atol=1e-6
                )


class TestTraitMonitor:
    def test_monitor_records_history(self, model_and_tokenizer, layers, happy_profile, layer_means):
        from saklas.hooks import SteeringManager
        from saklas.monitor import TraitMonitor
        from saklas.generation import GenerationConfig, GenerationState, generate_steered

        model, tokenizer = model_and_tokenizer
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        sad_profile = _extract_profile(model, tokenizer, "sad", layers)

        probe_profiles = {"happy": happy_profile, "sad": sad_profile}
        monitor = TraitMonitor(probe_profiles, layer_means)

        # Steer toward happy
        mgr = SteeringManager()
        mgr.add_vector("happy", happy_profile, 1.0)
        mgr.apply_to_model(layers, device, dtype)

        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": "How are you feeling?"}],
            add_generation_prompt=True, return_tensors="pt", return_dict=False,
        ).to(device)
        config = GenerationConfig(max_new_tokens=20, temperature=0.7)
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

        # With happy steering, happy sim should exceed sad sim
        assert happy_hist[0] > sad_hist[0], (
            f"Expected happy ({happy_hist[0]:.3f}) > sad ({sad_hist[0]:.3f}) with happy steering"
        )

        # Sparkline should be non-empty
        sparkline = monitor.get_sparkline("happy")
        assert len(sparkline) > 0

    def test_throughput_regression(self, model_and_tokenizer, layers, happy_profile, layer_means):
        """Steered generation should be at least 85% of vanilla throughput."""
        from saklas.hooks import SteeringManager
        from saklas.generation import GenerationConfig, GenerationState, generate_steered

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
        mgr.add_vector("happy", happy_profile, 0.8)
        curious_profile = _extract_profile(model, tokenizer, "curious", layers)
        mgr.add_vector("curious", curious_profile, 0.5)
        concise_profile = _extract_profile(model, tokenizer, "concise", layers)
        mgr.add_vector("concise", concise_profile, 0.3)
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


class TestExtractContrastive:
    def test_returns_valid_profile(self, model_and_tokenizer, layers, num_layers):
        from saklas.vectors import extract_contrastive
        model, tokenizer = model_and_tokenizer
        cfg = getattr(model.config, "text_config", None) or model.config
        hidden_dim = cfg.hidden_size
        pairs = [
            {"positive": "I feel happy today", "negative": "I feel sad today"},
            {"positive": "Everything is wonderful", "negative": "Everything is terrible"},
            {"positive": "I love this", "negative": "I hate this"},
        ]
        profile = extract_contrastive(model, tokenizer, pairs, layers=layers)
        assert isinstance(profile, dict)
        assert len(profile) > 0
        for idx, vec in profile.items():
            assert 0 <= idx < num_layers
            assert vec.shape == (hidden_dim,)
            norm = vec.norm().item()
            assert norm > 0 and not math.isinf(norm) and not math.isnan(norm)


class TestBuildChatInput:
    def test_chat_template_path(self, model_and_tokenizer):
        from saklas.generation import build_chat_input
        _, tokenizer = model_and_tokenizer
        messages = [{"role": "user", "content": "Hello"}]
        ids = build_chat_input(tokenizer, messages)
        assert ids.ndim == 2
        assert ids.shape[0] == 1
        assert ids.shape[1] > 0

    def test_with_system_prompt(self, model_and_tokenizer):
        from saklas.generation import build_chat_input
        _, tokenizer = model_and_tokenizer
        messages = [{"role": "user", "content": "Hello"}]
        ids_no_sys = build_chat_input(tokenizer, messages)
        ids_sys = build_chat_input(tokenizer, messages, system_prompt="You are helpful.")
        # System prompt should add tokens
        assert ids_sys.shape[1] > ids_no_sys.shape[1]


class TestProbesBootstrap:
    def test_bootstrap_loads_from_cache(self, monkeypatch, model_and_tokenizer, layers, happy_profile):
        """Bootstrap should return cached profiles without re-extracting."""
        from saklas.probes_bootstrap import bootstrap_probes
        from saklas.vectors import save_profile
        from saklas.paths import concept_dir, safe_model_id
        from saklas.packs import materialize_bundled, PackMetadata, hash_file
        from saklas.model import get_model_info
        model, tokenizer = model_and_tokenizer
        model_info = get_model_info(model, tokenizer)

        with tempfile.TemporaryDirectory() as tmp:
            monkeypatch.setenv("SAKLAS_HOME", tmp)
            materialize_bundled()
            # Pre-populate the `happy.sad` concept tensor for this model
            folder = concept_dir("default", "happy.sad")
            ts_path = folder / f"{safe_model_id(model_info['model_id'])}.safetensors"
            save_profile(happy_profile, str(ts_path), {
                "method": "contrastive_pca",
                "statements_sha256": hash_file(folder / "statements.json"),
            })
            # Refresh the pack.json files map to include the new tensor
            meta = PackMetadata.load(folder)
            meta.files[ts_path.name] = hash_file(ts_path)
            meta.files[ts_path.with_suffix(".json").name] = hash_file(ts_path.with_suffix(".json"))
            meta.write(folder)

            probes = bootstrap_probes(
                model, tokenizer, layers, model_info,
                categories=["affect"],
            )
            assert isinstance(probes, dict)
            assert "happy.sad" in probes
