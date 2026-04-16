"""Tests for SaklasSession programmatic API.
Requires a GPU (CUDA or Apple Silicon MPS) and downloads
google/gemma-3-4b-it (~8GB) on first run.
"""
from __future__ import annotations
import pytest
import torch
from saklas.core.profile import Profile
from saklas.core.results import GenerationResult, TokenEvent

_HAS_GPU = torch.cuda.is_available() or torch.backends.mps.is_available()
pytestmark = pytest.mark.skipif(
    not _HAS_GPU,
    reason="No GPU backend available (neither CUDA nor MPS)",
)

MODEL_ID = "google/gemma-3-4b-it"

@pytest.fixture(scope="module")
def session():
    from saklas.core.session import SaklasSession
    # device="auto" picks cuda > mps > cpu; skipif above guarantees a GPU.
    s = SaklasSession.from_pretrained(MODEL_ID, device="auto", probes=["affect"])
    yield s
    s.close()

class TestConstruction:
    def test_model_info(self, session):
        info = session.model_info
        # gemma-3-4b-it loads as the text-only submodule of a multimodal checkpoint,
        # so model_type is "gemma3_text" (see model.py:_load_text_from_multimodal).
        assert info["model_type"].startswith("gemma3")
        assert info["hidden_dim"] > 0
        assert info["num_layers"] > 0

    def test_config_defaults(self, session):
        assert session.config.temperature == 1.0
        assert session.config.top_p == 0.9
        assert session.config.max_new_tokens == 1024

    def test_probes_loaded(self, session):
        assert len(session.probes) > 0

    def test_history_starts_empty(self, session):
        assert session.history == []

    def test_vectors_starts_empty(self, session):
        assert session.vectors == {}

    def test_last_result_starts_none(self, session):
        assert session.last_result is None

class TestSteering:
    def test_extract_and_steer(self, session):
        name, profile = session.extract([("I am happy", "I am sad")])
        assert isinstance(profile, Profile)
        assert all(isinstance(k, int) for k in profile)
        session.steer("happy", profile)
        assert "happy" in session.vectors
        # vectors registry returns the dict-shaped inner wire format.
        assert isinstance(session.vectors["happy"], dict)

    def test_unsteer(self, session):
        session.unsteer("happy")
        assert "happy" not in session.vectors

    def test_extract_curated(self, session):
        name, profile = session.extract("happy", baseline="sad")
        assert name == "happy.sad"
        assert isinstance(profile, Profile)
        assert len(profile) > 0

    def test_extract_datasource(self, session):
        from saklas.io.datasource import DataSource
        ds = DataSource(pairs=[("formal", "casual")])
        name, profile = session.extract(ds)
        assert isinstance(profile, Profile)

class TestMonitoring:
    def test_monitor_and_unmonitor(self, session):
        _, profile = session.extract([("I am honest", "I am deceptive")])
        session.probe("test_probe", profile)
        assert "test_probe" in session.probes
        session.unprobe("test_probe")
        assert "test_probe" not in session.probes

class TestLifecycle:
    def test_context_manager(self):
        from saklas.core.session import SaklasSession
        with SaklasSession.from_pretrained(MODEL_ID, device="auto", probes=[]) as s:
            assert s.model_info["model_type"].startswith("gemma3")

class TestGeneration:
    def test_generate_unsteered(self, session):
        result = session.generate("Say hello in one word.")
        assert isinstance(result, GenerationResult)
        assert len(result.text) > 0
        assert result.token_count > 0
        assert result.tok_per_sec > 0
        assert result.elapsed > 0
        assert result.vectors == {}  # no alphas = no steering snapshot

    def test_generate_blocking_messages(self, session):
        result = session.generate([
            {"role": "user", "content": "Say hello in one word."},
        ])
        assert isinstance(result, GenerationResult)
        assert len(result.text) > 0

    def test_generate_appends_to_history(self, session):
        session.clear_history()
        session.generate("Say hi.")
        assert len(session.history) == 2
        assert session.history[0]["role"] == "user"
        assert session.history[1]["role"] == "assistant"

    def test_generate_with_alphas(self, session):
        _, profile = session.extract([("formal", "casual")])
        session.steer("formal", profile)
        result = session.generate("Hello.", steering={"formal": 0.1})
        assert result.vectors == {"formal": 0.1}
        session.unsteer("formal")

    def test_generate_with_probes(self, session):
        session.clear_history()
        result = session.generate("Tell me something exciting!")
        if session.probes:
            assert isinstance(result.readings, dict)

    def test_last_result(self, session):
        session.clear_history()
        result = session.generate("Hello.")
        assert session.last_result is result

    def test_ab_comparison(self, session):
        """A/B test: same prompt, with and without steering."""
        _, profile = session.extract([("I am happy", "I am sad")])
        session.steer("happy", profile)
        session.clear_history()
        steered = session.generate("Describe a sunset.", steering={"happy": 0.2})
        session.clear_history()
        unsteered = session.generate("Describe a sunset.")
        assert steered.vectors == {"happy": 0.2}
        assert unsteered.vectors == {}
        # Both should produce text
        assert len(steered.text) > 0
        assert len(unsteered.text) > 0
        session.unsteer("happy")

    def test_unknown_vector_raises(self, session):
        with pytest.raises(KeyError, match="nonexistent"):
            session.generate("Hello.", steering={"nonexistent": 0.1})

class TestCloning:
    def test_clone_from_corpus_end_to_end(self, session, tmp_path):
        from saklas.io.paths import concept_dir, safe_model_id

        pirate_lines = [
            "Arr matey, the briny deep be calling me name once more tonight",
            "Yo ho ho, we be sailing for doubloons afore the sun comes up",
            "Shiver me timbers, that cursed kraken nearly swallowed the whole crew",
            "Avast ye scurvy dogs, bring that grog barrel over to the quarterdeck",
            "Blimey, the cap'n be fouler than rotten fish on a humid afternoon",
            "Hoist the colors high lads, we be running from no king's navy",
            "The black spot upon me palm means me days be numbered now",
            "Batten down the hatches boys, a squall be rolling in from starboard",
            "Dead men tell no tales, or so the old pirate proverb claims",
            "Splice the mainbrace tonight mates, we've earned a proper ration of rum",
            "Me parrot squawks louder than the bosun on a windy morning watch",
            "That treasure map be worth more than any galleon full of silver",
            "Heave ho ye landlubbers, put yer backs into haulin' that anchor chain",
            "The Jolly Roger flutters proud above our weathered mast this fine dawn",
            "Keelhaul the traitor at first light, let the barnacles teach him manners",
            "Arr, this grog tastes like bilge water but a pirate drinks regardless",
            "The spyglass shows merchant sails on the horizon, ripe for the takin'",
            "Load the cannons double-shot, we be givin' them no quarter today",
            "A pirate's life be hard but the rum and plunder make it worthwhile",
            "Walk the plank ye mutinous cur, the sharks be hungry this morning",
            "The compass points nowhere useful when the devil's fog rolls thick",
            "Pieces of eight clatter sweet as music on the captain's oaken table",
            "Me peg leg aches afore every storm like a cursed weather vane",
            "The sea be a cruel mistress, takin' good men and givin' naught back",
            "Hoist the black flag and prepare to board her starboard side smartly",
            "That Spanish galleon rides low, heavy laden with colonial gold no doubt",
            "A pirate without a ship be naught but a drunkard on the shore",
            "Bury the chest deep beneath the third palm tree on Skull Isle",
            "The crow's nest spotted a frigate bearin' down on us from windward",
            "Sing a shanty loud enough to drown the groanin' of the old hull",
        ]
        corpus = tmp_path / "pirate_corpus.txt"
        corpus.write_text("\n".join(pirate_lines), encoding="utf-8")
        assert len(set(pirate_lines)) == len(pirate_lines)

        folder = concept_dir("local", "pirate_test")
        try:
            canonical, profile = session.clone_from_corpus(
                str(corpus), name="pirate_test", n_pairs=10, seed=42, force=True,
            )
            assert canonical == "pirate_test"
            assert isinstance(profile, Profile) and len(profile) > 0
            assert all(isinstance(k, int) for k in profile)
            for layer_idx, tensor in profile.items():
                assert tensor.numel() > 0
                assert torch.linalg.vector_norm(tensor.float()).item() > 0, (
                    f"layer {layer_idx} baked tensor is all zeros"
                )

            sid = safe_model_id(session.model_id)
            assert folder.exists()
            assert (folder / "pack.json").exists()
            assert (folder / "statements.json").exists()
            assert (folder / f"{sid}.safetensors").exists()

            # Probe path: add as probe, generate, score. Asserts scoring runs clean.
            session.probe("pirate_test", profile)
            try:
                session.clear_history()
                result = session.generate(
                    "Describe your morning.", steering=None,
                )
                readings = result.readings or {}
                if "pirate_test" in readings:
                    val = readings["pirate_test"].mean
                    assert val == val  # finite, not NaN
                    assert -1.5 <= val <= 1.5
            finally:
                session.unprobe("pirate_test")

            # Steering path: register and generate with α=1, compare a bundled
            # probe reading against the unsteered baseline on the same prompt.
            session.steer("pirate_test", profile)
            try:
                prompt = "Tell me about your day."
                session.clear_history()
                unsteered = session.generate(prompt)
                session.clear_history()
                steered = session.generate(prompt, steering={"pirate_test": 1.0})
                assert len(steered.text) > 0
                # Baseline shift assertion — loose. Only run if we share a probe.
                u_read = unsteered.readings or {}
                s_read = steered.readings or {}
                shared = set(u_read) & set(s_read)
                if shared:
                    diffs = [abs(u_read[k].mean - s_read[k].mean) for k in shared]
                    assert max(diffs) > 1e-6, (
                        "α=1 steering produced identical probe readings "
                        f"to unsteered on {sorted(shared)}"
                    )
            finally:
                session.unsteer("pirate_test")
        finally:
            if folder.exists():
                import shutil
                shutil.rmtree(folder, ignore_errors=True)

    def test_extract_cli_roundtrip(self, tmp_path):
        import subprocess
        import sys
        from saklas.io.paths import concept_dir, safe_model_id

        folder = concept_dir("default", "happy.sad")
        sid = safe_model_id(MODEL_ID)
        tensor_path = folder / f"{sid}.safetensors"
        created_here = not tensor_path.exists()

        try:
            proc = subprocess.run(
                [sys.executable, "-m", "saklas", "pack", "extract",
                 "happy.sad", "-m", MODEL_ID],
                capture_output=True, text=True, timeout=600,
            )
            assert proc.returncode == 0, (
                f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )
            assert tensor_path.exists(), f"expected {tensor_path} to exist"
        finally:
            # Only unlink the per-model tensor if this test created it;
            # leave the bundled statements.json and any pre-existing tensor alone.
            if created_here and tensor_path.exists():
                tensor_path.unlink()
                sidecar = folder / f"{sid}.json"
                if sidecar.exists():
                    sidecar.unlink()


class TestStreamingGeneration:
    def test_generate_stream(self, session):
        session.clear_history()
        tokens = []
        for event in session.generate_stream("Say hello."):
            assert isinstance(event, TokenEvent)
            tokens.append(event)
        assert len(tokens) > 0
        assert all(isinstance(t.text, str) for t in tokens)
        assert session.last_result is not None
        assert session.last_result.token_count == len(tokens)

    def test_stream_with_alphas(self, session):
        _, profile = session.extract([("I am happy", "I am sad")])
        session.steer("happy", profile)
        session.clear_history()
        tokens = list(session.generate_stream("Hello.", steering={"happy": 0.15}))
        assert len(tokens) > 0
        assert session.last_result.vectors == {"happy": 0.15}
        session.unsteer("happy")
