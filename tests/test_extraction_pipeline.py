"""ExtractionPipeline focused tests — CPU only, duck-typed dependencies.

Mirrors the ``MockSaeBackend`` pattern in :mod:`saklas.core.sae`:
construct duck-typed stubs against the structural protocols (no model
load, no real forward pass) and exercise the pipeline's cache-hit
short-circuits, scenario reuse, and ``force_statements`` behavior.

GPU-end-to-end coverage stays in :mod:`tests.test_session`; this file
keeps the pipeline addressable without a model in the loop.
"""
from __future__ import annotations

import json
import pathlib

import pytest
import torch

from saklas.core.events import EventBus
from saklas.core.extraction import (
    ExtractionPipeline,
    ModelHandle,
    PackWriter,
    VectorRegistry,
)


# ----------------------------------------------------------------------
# Minimal duck-typed handle that satisfies all three Protocols at once.
# Tracks every model-side / generator-side call so tests can assert the
# pipeline took the expected path.
# ----------------------------------------------------------------------


class _StubHandle:
    """Single object satisfying ModelHandle + PackWriter + VectorRegistry.

    Mirrors the session's natural shape: the pipeline is constructed
    against ``handle, handle, handle, events`` so the structural
    protocols line up one-to-one with concrete attrs/methods.
    """

    def __init__(
        self,
        tmp_path: pathlib.Path,
        *,
        scenarios_response: list[str] | None = None,
        pairs_response: list[tuple[str, str]] | None = None,
    ):
        self.model_id = "stub-model"
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.model = object()
        self.tokenizer = object()
        self.layers = [object(), object(), object(), object()]

        self._tmp = pathlib.Path(tmp_path)
        self._profiles: dict = {}

        # Tracking — every test that cares asserts on these counters.
        self.run_generator_calls = 0
        self.scenarios_calls = 0
        self.pairs_calls = 0
        self.promote_calls = 0
        self.update_pack_calls = 0
        self.added: dict = {}

        self._scenarios_response = scenarios_response or [f"domain {i}" for i in range(9)]
        self._pairs_response = pairs_response or [
            (f"positive {i}", f"negative {i}") for i in range(4)
        ]

    # ModelHandle surface ------------------------------------------------

    def _run_generator(self, system_msg, prompt, max_new_tokens):  # pragma: no cover
        self.run_generator_calls += 1
        # Tests should not actually take this path; if they do the
        # response is bogus on purpose.
        return ""

    def generate_scenarios(self, concept, baseline=None, n=9, *, on_progress=None):
        self.scenarios_calls += 1
        return list(self._scenarios_response)

    def generate_pairs(self, concept, baseline=None, n=45, *, scenarios=None, on_progress=None):
        self.pairs_calls += 1
        return list(self._pairs_response)

    # PackWriter surface -------------------------------------------------

    def _local_concept_folder(self, canonical):
        from saklas.io.packs import PackMetadata
        folder = self._tmp / "vectors" / "local" / canonical
        folder.mkdir(parents=True, exist_ok=True)
        if not (folder / "pack.json").exists():
            PackMetadata(
                name=canonical, description="test", version="1.0.0",
                license="MIT", tags=[], recommended_alpha=0.5,
                source="local", files={},
            ).write(folder)
        return folder

    def _promote_profile(self, p):
        self.promote_calls += 1
        return p

    def _update_local_pack_files(self, folder):
        self.update_pack_calls += 1

    # VectorRegistry surface --------------------------------------------

    def __contains__(self, name):
        return name in self._profiles

    def add(self, name, profile):
        self.added[name] = profile


def _fake_extract(monkeypatch, *, response=None):
    """Replace ``extract_contrastive`` inside the extraction module."""
    from saklas.core import extraction as E

    captured: dict = {}

    def _fake(model, tokenizer, pairs, layers, device=None, *, sae=None):
        captured["pairs"] = pairs
        captured["sae"] = sae
        captured["call_count"] = captured.get("call_count", 0) + 1
        return response if response is not None else {0: torch.ones(4), 2: torch.ones(4)}

    monkeypatch.setattr(E, "extract_contrastive", _fake)
    return captured


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


class TestProtocolShape:
    """Runtime-checkable Protocols accept the implicit session implementation."""

    def test_session_satisfies_modelhandle(self, tmp_path, monkeypatch):
        # SaklasSession's natural shape passes isinstance against all three
        # protocols.  Validates the `runtime_checkable` decoration on each.
        handle = _StubHandle(tmp_path)
        assert isinstance(handle, ModelHandle)
        assert isinstance(handle, PackWriter)
        assert isinstance(handle, VectorRegistry)

    def test_pipeline_constructs_against_stub(self, tmp_path):
        handle = _StubHandle(tmp_path)
        pipeline = ExtractionPipeline(handle, handle, handle, EventBus())
        # Hold the references the plan promised.
        assert pipeline._handle is handle
        assert pipeline._packs is handle
        assert pipeline._registry is handle


class TestTensorCacheShortCircuit:
    """Cache-hit semantics: pre-populated tensor → no model forward fires."""

    def test_tensor_cache_hit_skips_extract_contrastive(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))

        captured = _fake_extract(monkeypatch)

        # Pre-populate a baked tensor under default/<concept>/.
        from saklas.core.vectors import save_profile
        from saklas.io.packs import PackMetadata, hash_folder_files
        from saklas.io.paths import tensor_filename
        from saklas.io.selectors import invalidate
        invalidate()

        folder = tmp_path / "vectors" / "default" / "honest.deceptive"
        folder.mkdir(parents=True)
        save_profile(
            {0: torch.full((4,), 0.5)},
            str(folder / tensor_filename("stub-model")),
            {"method": "contrastive_pca"},
        )
        PackMetadata(
            name="honest.deceptive", description="x", version="1.0.0",
            license="MIT", tags=[], recommended_alpha=0.5,
            source="local", files=hash_folder_files(folder),
        ).write(folder)

        handle = _StubHandle(tmp_path)
        pipeline = ExtractionPipeline(handle, handle, handle, EventBus())

        name, profile = pipeline.extract("honest.deceptive")

        assert name == "honest.deceptive"
        assert "call_count" not in captured  # extract_contrastive never fired
        assert handle.scenarios_calls == 0
        assert handle.pairs_calls == 0
        assert handle.promote_calls == 1  # cache load promoted to device


class TestForceStatementsRegenerates:
    """force_statements=True: cache exists, statements regenerated."""

    def test_force_statements_bypasses_cache_and_calls_generators(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))

        captured = _fake_extract(monkeypatch)

        # Pre-populate tensor + statements caches (both should be ignored).
        from saklas.core.vectors import save_profile
        from saklas.io.packs import PackMetadata, hash_folder_files
        from saklas.io.paths import tensor_filename
        from saklas.io.selectors import invalidate
        invalidate()

        folder = tmp_path / "vectors" / "local" / "honest.deceptive"
        folder.mkdir(parents=True)
        save_profile(
            {0: torch.full((4,), 0.5)},
            str(folder / tensor_filename("stub-model")),
            {"method": "contrastive_pca"},
        )
        (folder / "statements.json").write_text(json.dumps([
            {"positive": "stale-p", "negative": "stale-n"},
            {"positive": "stale-p2", "negative": "stale-n2"},
        ]))
        PackMetadata(
            name="honest.deceptive", description="x", version="1.0.0",
            license="MIT", tags=[], recommended_alpha=0.5,
            source="local", files=hash_folder_files(folder),
        ).write(folder)

        handle = _StubHandle(tmp_path)
        pipeline = ExtractionPipeline(handle, handle, handle, EventBus())

        name, profile = pipeline.extract("honest.deceptive", force_statements=True)

        assert name == "honest.deceptive"
        # Generators fire — the stale statements.json was bypassed.
        assert handle.scenarios_calls == 1
        assert handle.pairs_calls == 1
        # extract_contrastive fired once on the freshly-generated pairs.
        assert captured.get("call_count") == 1
        # Pairs must come from the stub responses, not the stale file.
        new_pairs = captured["pairs"]
        assert all(p["positive"].startswith("positive ") for p in new_pairs)


class TestExplicitScenariosBypass:
    """scenarios=[...]: pair gen runs, but scenario gen does NOT."""

    def test_explicit_scenarios_skips_scenario_generation(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))

        captured = _fake_extract(monkeypatch)

        from saklas.io.selectors import invalidate
        invalidate()

        handle = _StubHandle(tmp_path)
        pipeline = ExtractionPipeline(handle, handle, handle, EventBus())

        name, profile = pipeline.extract(
            "honest.deceptive",
            scenarios=["caller-supplied domain 1", "caller-supplied domain 2"],
        )

        assert name == "honest.deceptive"
        # Scenario generator skipped (caller provided them).
        assert handle.scenarios_calls == 0
        # Pair generator still fired against the supplied scenarios.
        assert handle.pairs_calls == 1
        # extract_contrastive fired exactly once on the new pairs.
        assert captured.get("call_count") == 1

        # scenarios.json on disk reflects the caller's input, not the stub default.
        scn_path = (
            tmp_path / "vectors" / "local" / "honest.deceptive" / "scenarios.json"
        )
        assert scn_path.exists()
        data = json.loads(scn_path.read_text())
        assert data["scenarios"] == [
            "caller-supplied domain 1", "caller-supplied domain 2",
        ]


class TestVectorExtractedEvent:
    """The pipeline still emits VectorExtracted on the supplied EventBus."""

    def test_vector_extracted_event_fires_on_cache_hit(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))

        _fake_extract(monkeypatch)

        # Build a cache hit so the event is emitted with the cache method.
        from saklas.core.vectors import save_profile
        from saklas.io.packs import PackMetadata, hash_folder_files
        from saklas.io.paths import tensor_filename
        from saklas.io.selectors import invalidate
        invalidate()

        folder = tmp_path / "vectors" / "default" / "honest.deceptive"
        folder.mkdir(parents=True)
        save_profile(
            {0: torch.full((4,), 0.5)},
            str(folder / tensor_filename("stub-model")),
            {"method": "contrastive_pca"},
        )
        PackMetadata(
            name="honest.deceptive", description="x", version="1.0.0",
            license="MIT", tags=[], recommended_alpha=0.5,
            source="local", files=hash_folder_files(folder),
        ).write(folder)

        handle = _StubHandle(tmp_path)
        bus = EventBus()
        seen = []
        from saklas.core.events import VectorExtracted
        bus.subscribe(lambda e: seen.append(e) if isinstance(e, VectorExtracted) else None)

        pipeline = ExtractionPipeline(handle, handle, handle, bus)
        pipeline.extract("honest.deceptive")

        assert len(seen) == 1
        evt = seen[0]
        assert evt.name == "honest.deceptive"
        # Cache-hit metadata flows through unchanged.
        assert evt.metadata.get("method") == "contrastive_pca"


class TestSessionGate:
    """SaklasSession.extract gates on GenState.IDLE before delegating."""

    def test_session_extract_raises_when_generation_active(self):
        # Bypass SaklasSession.__init__ — we only need _gen_phase + _extraction.
        from types import SimpleNamespace
        from saklas.core.session import (
            ConcurrentExtractionError, GenState, SaklasSession,
        )

        import threading
        session = SaklasSession.__new__(SaklasSession)
        session._gen_phase = GenState.RUNNING
        session._gen_lock = threading.Lock()
        # _extraction won't be reached — gate fires first.
        session._extraction = SimpleNamespace(extract=lambda *a, **kw: ("x", None))

        with pytest.raises(ConcurrentExtractionError):
            session.extract("honest.deceptive")

    def test_concurrent_extraction_error_subclasses_saklas_error(self):
        from saklas.core.errors import SaklasError
        from saklas.core.session import ConcurrentExtractionError
        assert issubclass(ConcurrentExtractionError, SaklasError)
        assert issubclass(ConcurrentExtractionError, RuntimeError)
        # 409 conflict — same shape as ConcurrentGenerationError.
        code, _msg = ConcurrentExtractionError("x").user_message()
        assert code == 409
