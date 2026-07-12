"""Tests for SaklasSession programmatic API.
Requires a GPU (CUDA or Apple Silicon MPS) and downloads
google/gemma-3-4b-it (~8GB) on first run.
"""
from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING
import pytest
import torch
from saklas.core.profile import Profile
from saklas.core.results import GenerationResult, RunSet, TokenEvent

if TYPE_CHECKING:
    from saklas.core.session import SaklasSession

_HAS_GPU = torch.cuda.is_available() or torch.backends.mps.is_available()
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not _HAS_GPU,
        reason="No GPU backend available (neither CUDA nor MPS)",
    ),
]

MODEL_ID = "google/gemma-3-4b-it"


def _corpus(response: str) -> list[str]:
    from saklas.core.vectors import _load_baseline_prompts

    return [response] * len(_load_baseline_prompts())


@pytest.fixture(scope="module")
def session(tmp_path_factory: pytest.TempPathFactory):
    import os
    from saklas.core.session import SaklasSession
    # Isolate $SAKLAS_HOME so this module's extract/merge writes (e.g.
    # local/formal.casual, happy.sad, honest) land in a throwaway cache instead
    # of the user's real ~/.saklas — where they would shadow bundled manifolds of
    # the same name and break bare-name resolution (AmbiguousSelectorError)
    # across the whole suite on the next run.
    home = tmp_path_factory.mktemp("saklas_home")
    prev = os.environ.get("SAKLAS_HOME")
    os.environ["SAKLAS_HOME"] = str(home)
    # device="auto" picks cuda > mps > cpu; skipif above guarantees a GPU.
    try:
        s = SaklasSession.from_pretrained(MODEL_ID, device="auto", probes=["register"])
        yield s
        s.close()
    finally:
        if prev is None:
            os.environ.pop("SAKLAS_HOME", None)
        else:
            os.environ["SAKLAS_HOME"] = prev

class TestConstruction:
    def test_model_info(self, session: SaklasSession) -> None:
        info = session.model_info
        # gemma-3-4b-it loads as the text-only submodule of a multimodal checkpoint,
        # so model_type is "gemma3_text" (see model.py:_load_text_from_multimodal).
        assert info["model_type"].startswith("gemma3")
        assert info["hidden_dim"] > 0
        assert info["num_layers"] > 0

    def test_config_defaults(self, session: SaklasSession) -> None:
        assert session.config.temperature == 1.0
        assert session.config.top_p == 0.9
        assert session.config.max_new_tokens == 1024

    def test_probes_loaded(self, session: SaklasSession) -> None:
        assert len(session.probes) > 0

    def test_history_starts_empty(self, session: SaklasSession) -> None:
        assert session.history == []

    def test_vectors_starts_empty(self, session: SaklasSession) -> None:
        assert session.vectors == {}

    def test_last_result_starts_none(self, session: SaklasSession) -> None:
        assert session.last_result is None

class TestSteering:
    def test_extract_and_steer(self, session: SaklasSession) -> None:
        name, profile = session.extract_vector_from_corpora(
            "happy", _corpus("I am happy"), _corpus("I am sad"),
        )
        assert isinstance(profile, Profile)
        assert all(isinstance(k, int) for k in profile)
        session.steer("happy", profile)
        assert "happy" in session.vectors
        # vectors registry speaks Profile, not bare dicts (saklas 1.x → 3.x).
        assert isinstance(session.vectors["happy"], Profile)

    def test_unsteer(self, session: SaklasSession) -> None:
        session.unsteer("happy")
        assert "happy" not in session.vectors

    def test_extract_curated(self, session: SaklasSession) -> None:
        name, profile = session.extract("happy", baseline="sad")
        assert name == "happy.sad"
        assert isinstance(profile, Profile)
        assert len(profile) > 0

    def test_extract_datasource(self, session: SaklasSession) -> None:
        name, profile = session.extract_vector_from_corpora(
            "formal.casual", _corpus("formal"), _corpus("casual"),
        )
        assert isinstance(profile, Profile)

class TestMonitoring:
    def test_monitor_and_unmonitor(self, session: SaklasSession) -> None:
        # Extract registers the folded direction; ``add_probe`` resolves it
        # (the unified probe API — one attach for vector + manifold probes).
        name, _profile = session.extract_vector_from_corpora(
            "honest", _corpus("I am honest"), _corpus("I am deceptive"),
        )
        session.add_probe(name, as_name="test_probe")
        assert "test_probe" in session.probes
        session.remove_probe("test_probe")
        assert "test_probe" not in session.probes

class TestLifecycle:
    def test_context_manager(self):
        from saklas.core.session import SaklasSession
        with SaklasSession.from_pretrained(MODEL_ID, device="auto", probes=[]) as s:
            assert s.model_info["model_type"].startswith("gemma3")

class TestGeneration:
    def test_generate_unsteered(self, session: SaklasSession) -> None:
        result = session.generate("Say hello in one word.")
        assert isinstance(result, RunSet)
        assert isinstance(result.first, GenerationResult)
        single = result.first
        assert len(single.text) > 0
        assert single.token_count > 0
        assert single.tok_per_sec > 0
        assert single.elapsed > 0
        assert single.steering_alphas == {}

    def test_generate_blocking_messages(self, session: SaklasSession) -> None:
        result = session.generate([
            {"role": "user", "content": "Say hello in one word."},
        ])
        assert isinstance(result, RunSet)
        assert isinstance(result.first, GenerationResult)
        assert len(result.first.text) > 0

    def test_generate_appends_to_history(self, session: SaklasSession) -> None:
        session.clear_history()
        session.generate("Say hi.")
        assert len(session.history) == 2
        assert session.history[0]["role"] == "user"
        assert session.history[1]["role"] == "assistant"

    def test_generate_with_alphas(self, session: SaklasSession) -> None:
        name, profile = session.extract_vector_from_corpora(
            "formal.casual", _corpus("formal"), _corpus("casual"),
        )
        session.steer(name, profile)
        result = session.generate("Hello.", steering=f"0.1 {name}").first
        assert result.steering_alphas == {name: 0.1}
        session.unsteer(name)

    def test_generate_with_probes(self, session: SaklasSession) -> None:
        session.clear_history()
        result = session.generate("Tell me something exciting!").first
        if session.probes:
            assert isinstance(result.probe_readings, dict)

    def test_last_result(self, session: SaklasSession) -> None:
        session.clear_history()
        result = session.generate("Hello.")
        assert session.last_result is result.first

    def test_ab_comparison(self, session: SaklasSession) -> None:
        """A/B test: same prompt, with and without steering."""
        name, profile = session.extract_vector_from_corpora(
            "happy", _corpus("I am happy"), _corpus("I am sad"),
        )
        session.steer(name, profile)
        session.clear_history()
        steered = session.generate("Describe a sunset.", steering=f"0.2 {name}").first
        session.clear_history()
        unsteered = session.generate("Describe a sunset.").first
        assert steered.steering_alphas == {name: 0.2}
        assert unsteered.steering_alphas == {}
        # Both should produce text
        assert len(steered.text) > 0
        assert len(unsteered.text) > 0
        session.unsteer(name)

    def test_unknown_vector_raises(self, session: SaklasSession) -> None:
        with pytest.raises(KeyError, match="nonexistent"):
            session.generate("Hello.", steering="0.1 nonexistent")

class TestCliRoundTrip:
    def test_extract_cli_roundtrip(self, tmp_path: Path) -> None:
        import subprocess
        import sys
        from saklas.io.paths import manifold_dir, safe_model_id

        folder = manifold_dir("local", "happy.sad")
        sid = safe_model_id(MODEL_ID)
        tensor_path = folder / f"{sid}.safetensors"
        created_here = not tensor_path.exists()

        try:
            proc = subprocess.run(
                [
                    sys.executable, "-m", "saklas", "manifold", "extract",
                    "happy.sad", "-m", MODEL_ID, "--namespace", "local",
                ],
                capture_output=True, text=True, timeout=600,
            )
            assert proc.returncode == 0, (
                f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )
            assert tensor_path.exists(), f"expected {tensor_path} to exist"
        finally:
            # Only unlink the per-model tensor if this test created it;
            # leave the corpus and any pre-existing tensor alone.
            if created_here and tensor_path.exists():
                tensor_path.unlink()
                sidecar = folder / f"{sid}.json"
                if sidecar.exists():
                    sidecar.unlink()


class TestStreamingGeneration:
    def test_generate_stream(self, session: SaklasSession) -> None:
        session.clear_history()
        tokens = []
        for event in session.generate_stream("Say hello."):
            assert isinstance(event, TokenEvent)
            tokens.append(event)
        assert len(tokens) > 0
        assert all(isinstance(t.text, str) for t in tokens)
        assert session.last_result is not None
        assert session.last_result.token_count == len(tokens)

    def test_stream_with_alphas(self, session: SaklasSession) -> None:
        name, profile = session.extract_vector_from_corpora(
            "happy", _corpus("I am happy"), _corpus("I am sad"),
        )
        session.steer(name, profile)
        session.clear_history()
        tokens = list(session.generate_stream("Hello.", steering=f"0.15 {name}"))
        assert len(tokens) > 0
        assert session.last_result is not None  # generate_stream guarantees last_result is set
        assert session.last_result.steering_alphas == {name: 0.15}
        session.unsteer(name)


class TestAblation:
    @pytest.mark.skip(reason=(
        "Pinned the Euclidean TraitMonitor, removed in the monitor "
        "unification (Mahalanobis-only). The ablation operator is Euclidean "
        "(h' = h - alpha(h.d_hat - mu.d_hat)d_hat) but the unified Monitor "
        "reads the Mahalanobis coordinate, which by design does NOT collapse "
        "under Euclidean ablation, so a monitor-based assertion no longer "
        "matches the operator. Recommended rewrite: assert directly on the "
        "metric-agnostic Euclidean projection of the post-ablation hidden "
        "onto folded_vector_directions(session._monitor.manifolds[probe]) "
        "(the exact component the operator zeros) instead of a monitor read. "
        "Deferred to the monitor-shape pass (GPU, owner: a9); also revisit the "
        "fixture, which bootstraps the dropped 'affect' category."
    ))
    def test_ablation_suppresses_self_probe_score(self, session: SaklasSession) -> None:
        """Ablating a concept suppresses its own direction in activation space.

        See the skip reason: the original pinned the removed Euclidean monitor;
        the faithful Mahalanobis-era rewrite tests the operator via a direct
        Euclidean projection onto the probe's folded direction.
        """


def test_return_hidden_round_trip(session: SaklasSession) -> None:
    """return_hidden=True populates hidden_states; score_hidden round-trips."""
    from saklas import SamplingConfig

    num_layers = len(session._layers)
    hidden_dim = session.model_info["hidden_dim"]

    result = session.generate(
        "Count to three.",
        sampling=SamplingConfig(
            max_tokens=16, temperature=0.0, return_hidden=True,
        ),
    ).first
    assert result.hidden_states is not None
    # All layers captured.
    assert len(result.hidden_states) == num_layers
    # Shape: [T, D] per layer, T == len(generated tokens).
    T = len(result.tokens)
    for layer_idx, h in result.hidden_states.items():
        assert h.shape == (T, hidden_dim), (
            f"layer {layer_idx}: expected ({T}, {hidden_dim}), got {tuple(h.shape)}"
        )
        assert h.device.type == "cpu"

    # Round-trip: re-score the captured dict and compare against the
    # per-token scores the session computed inline. The fixture
    # bootstraps with probes=["affect"]; if that invariant ever
    # regresses, silent-skip would hide the real test.
    assert session._monitor.probe_names, (
        "fixture must have probes loaded for round-trip coverage"
    )
    _, per_token = session.score_hidden(
        result.hidden_states, per_token=True,
    )
    expected = session.last_per_token_scores or {}
    # Both sides route through the same _score_probes kernel; the only
    # noise is the GPU→CPU move on the result-stored tensors, which is
    # well below 1e-4 for bf16. Looser tolerances let a one-token
    # pooling shift slip past.
    tol = 1e-4
    for name, vals in per_token.items():
        if name not in expected:
            continue
        assert len(vals) == len(expected[name])
        for a, b in zip(vals, expected[name]):
            assert abs(a - b) < tol, f"probe {name}: {a} vs {b}"


def test_return_hidden_false_leaves_hidden_states_none(session: SaklasSession) -> None:
    from saklas import SamplingConfig

    result = session.generate(
        "Hello.",
        sampling=SamplingConfig(max_tokens=4, temperature=0.0),
    ).first
    assert result.hidden_states is None


class TestPrefixCache:
    """Prefix KV cache: opt-in optimization for batch workloads with a
    shared chat prefix.  See ``SaklasSession.cache_prefix``.
    """

    def _shared_prefix_messages(self, session: SaklasSession, prompt_body: str):
        """Build a (prefix_messages, full_messages) pair where the
        full chat-template encoding of full_messages begins with the
        prefix_messages encoding.

        Trick: encode the prefix-only messages with
        ``add_generation_prompt=False`` and prepend its decoded text
        to the prompt body in a single user message — that shape
        always satisfies the byte-prefix invariant on tokenizers
        whose user-turn tokens partition cleanly on whitespace
        (Gemma's chat template is one such).
        """
        # Prefix is its own user turn; full is the same prefix +
        # additional user turn.  The chat-template encoding of
        # [u1] is a prefix of [u1, u2] under add_generation_prompt
        # = False for prefix_only and True for full.  But that
        # ISN'T sufficient — the False-encoded prefix may differ
        # by trailing tokens.  Instead, encode the prefix as a token
        # tensor to bypass the issue.
        from saklas.core.generation import build_chat_input

        prefix_msg = [
            {"role": "user", "content": "Be concise. Always end with a period."},
        ]
        full_msg = prefix_msg + [
            {"role": "assistant", "content": "Understood."},
            {"role": "user", "content": prompt_body},
        ]
        full_ids = build_chat_input(
            session._tokenizer, full_msg, session.config.system_prompt,
        )
        # Find the longest token prefix that's a clean head of full_ids.
        # We just take everything up to (but not including) the second
        # user-turn opener.  Falling back to a deterministic split by
        # token-string match keeps this independent of model family.
        prefix_ids = build_chat_input(
            session._tokenizer, prefix_msg, session.config.system_prompt,
            add_generation_prompt=False,
        )
        # Trim prefix_ids to the longest matching head of full_ids.
        L = 0
        max_L = min(prefix_ids.shape[1], full_ids.shape[1])
        for i in range(max_L):
            if int(prefix_ids[0, i]) == int(full_ids[0, i]):
                L = i + 1
            else:
                break
        prefix_ids_trim = prefix_ids[:, :L]
        return prefix_ids_trim, full_msg

    def test_cache_hit_matches_no_cache_output(self, session: SaklasSession) -> None:
        from saklas import SamplingConfig

        session.clear_history()
        prefix_ids, full_msg = self._shared_prefix_messages(
            session, "Say the word 'banana' three times.",
        )

        # Baseline: no cache, deterministic.
        session.cache_prefix(None)
        baseline = session.generate(
            full_msg,
            sampling=SamplingConfig(max_tokens=12, temperature=0.0, seed=42),
            stateless=True,
        ).first

        # Warm the cache and rerun the same prompt; outputs must match.
        prefix_len = session.cache_prefix(prefix_ids)
        assert prefix_len > 0
        cached = session.generate(
            full_msg,
            sampling=SamplingConfig(max_tokens=12, temperature=0.0, seed=42),
            stateless=True,
        ).first
        # Same prompt + same seed + same model state → identical token
        # stream regardless of how prefill was sliced.
        assert cached.tokens == baseline.tokens, (
            f"prefix-cache hit produced different tokens than no-cache:\n"
            f"  no-cache: {baseline.tokens}\n"
            f"  cached:   {cached.tokens}"
        )
        # Cleanup so other tests aren't affected.
        session.cache_prefix(None)

    def test_steering_invalidates_cache(self, session: SaklasSession) -> None:
        # Warm the cache.
        prefix_ids, _ = self._shared_prefix_messages(session, "Hello.")
        session.cache_prefix(prefix_ids)
        assert session._prefix_cache is not None

        # Make sure there's a steering vector to push.
        if not session.has_vector("happy"):
            _, prof = session.extract_vector_from_corpora(
                "happy", _corpus("I am happy"), _corpus("I am sad"),
            )
            session.steer("happy", prof)
        # steer() itself invalidates; re-warm and verify scope-entry
        # is what we're really testing.
        session.cache_prefix(prefix_ids)
        assert session._prefix_cache is not None

        with session.steering("0.1 happy"):
            # Scope entry must drop the cache.
            assert session._prefix_cache is None

        # And the post-scope rebuild must NOT magically reinstate it.
        assert session._prefix_cache is None
        session.unsteer("happy")
