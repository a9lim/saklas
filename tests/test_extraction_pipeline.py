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
from pathlib import Path
from typing import Any, Callable

import pytest
import torch

from saklas.core.events import EventBus
from saklas.core.extraction import (
    ExtractionPipeline,
    ModelHandle,
    PackWriter,
)



# ----------------------------------------------------------------------
# Minimal duck-typed handle that satisfies both Protocols at once.
# Tracks every model-side / generator-side call so tests can assert the
# pipeline took the expected path.
# ----------------------------------------------------------------------


class _StubHandle:
    """Single object satisfying ModelHandle + PackWriter.

    Mirrors the session's natural shape: the pipeline is constructed
    against ``handle, handle, events`` so the structural
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
        self.model: torch.nn.Module = object()  # pyright: ignore[reportAttributeAccessIssue]  # stub object satisfies duck-typed usage only; real Module not needed for CPU tests
        self.tokenizer: Any = object()
        self.layers: Any = [object(), object(), object(), object()]

        self._tmp = pathlib.Path(tmp_path)
        self._profiles: dict[str, Any] = {}

        # Tracking — every test that cares asserts on these counters.
        self.run_generator_calls = 0
        self.scenarios_calls = 0
        self.pairs_calls = 0
        self.promote_calls = 0
        self.update_pack_calls = 0
        self.added: dict[str, Any] = {}

        self._scenarios_response = scenarios_response or [f"domain {i}" for i in range(9)]
        self._pairs_response = pairs_response or [
            (f"positive {i}", f"negative {i}") for i in range(4)
        ]

    # ModelHandle surface ------------------------------------------------

    def _run_generator(self, system_msg: str, prompt: str, max_new_tokens: int) -> str:  # pragma: no cover
        self.run_generator_calls += 1
        # Tests should not actually take this path; if they do the
        # response is bogus on purpose.
        return ""

    def generate_scenarios(
        self,
        concept: str,
        baseline: str | None = None,
        n: int = 9,
        *,
        on_progress: Callable[[str], None] | None = None,
        role: str | None = None,
    ) -> list[str]:
        self.scenarios_calls += 1
        self.last_scenarios_role = role
        return list(self._scenarios_response)

    def generate_statements(
        self,
        concepts: list[str],
        *,
        scenarios: list[str] | None = None,
        n_scenarios: int = 9,
        statements_per_cell: int = 5,
        share_moment: bool = False,
        on_progress: Callable[[str], None] | None = None,
        role: str | None = None,
    ) -> dict[str, list[str]]:
        self.pairs_calls += 1
        self.last_pairs_role = role
        # Reshape the canned ``(positive, negative)`` pairs into the
        # dict-of-corpora shape the new API returns.  The pipeline
        # always passes a 2-element concepts list under
        # ``share_moment=True``, so ``concepts[0]`` is positive and
        # ``concepts[1]`` is the negative slot (real baseline or the
        # synthesized ``the_opposite_of_<X>`` for monopolar concepts).
        pos_lines = [p for p, _ in self._pairs_response]
        neg_lines = [n for _, n in self._pairs_response]
        return {
            concepts[0]: pos_lines,
            concepts[1]: neg_lines,
        }

    # PackWriter surface -------------------------------------------------

    def _local_concept_folder(self, canonical: str, *, namespace: str = "local") -> pathlib.Path:
        from saklas.io.packs import PackMetadata
        folder = self._tmp / "vectors" / namespace / canonical
        folder.mkdir(parents=True, exist_ok=True)
        if not (folder / "pack.json").exists():
            PackMetadata(
                name=canonical, description="test", version="1.0.0",
                license="MIT", tags=[], recommended_alpha=0.5,
                source="local", files={},
            ).write(folder)
        return folder

    def _promote_profile(self, profile: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
        self.promote_calls += 1
        return profile

    def _update_local_pack_files(self, folder: pathlib.Path) -> None:
        self.update_pack_calls += 1


def _fake_extract(monkeypatch: Any, *, response: Any = None) -> dict[str, Any]:
    """Replace both ``extract_contrastive`` and ``extract_difference_of_means``
    inside the extraction module.

    The pipeline dispatches to one or the other based on ``method=``; tests
    that don't care which method ran (the dispatch architecture itself, not
    the per-method math) get one shared stub via this helper.  Tests that
    do care about per-method dispatch hit ``captured["method"]`` to read
    back which extractor fired.
    """
    from saklas.core import extraction as E

    captured: dict[str, Any] = {}

    def _make(label: str) -> Callable[..., Any]:
        def _fake(model: Any, tokenizer: Any, pairs: Any, layers: Any, device: Any = None, *,
                  sae: Any = None, concept_label: Any = None, **_kwargs: Any) -> Any:
            # ``**_kwargs`` swallows ``dls`` / ``layer_means`` /
            # ``whitener`` (added in v2.1) — the fake doesn't model
            # any of them, just records what the pipeline asked for.
            captured["pairs"] = pairs
            captured["sae"] = sae
            captured["concept_label"] = concept_label
            captured["method"] = label
            captured["call_count"] = captured.get("call_count", 0) + 1
            profile = response if response is not None else {0: torch.ones(4), 2: torch.ones(4)}
            return profile, {}
        return _fake

    monkeypatch.setattr(E, "extract_contrastive", _make("pca"))
    monkeypatch.setattr(E, "extract_difference_of_means", _make("dim"))
    return captured


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


class TestProtocolShape:
    """Runtime-checkable Protocols accept the implicit session implementation."""

    def test_session_satisfies_modelhandle(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # SaklasSession's natural shape passes isinstance against both
        # protocols.  Validates the `runtime_checkable` decoration on each.
        handle = _StubHandle(tmp_path)
        assert isinstance(handle, ModelHandle)
        assert isinstance(handle, PackWriter)

    def test_pipeline_constructs_against_stub(self, tmp_path: Path) -> None:
        handle = _StubHandle(tmp_path)
        pipeline = ExtractionPipeline(handle, handle, EventBus())  # pyright: ignore[reportArgumentType]
        # Hold the references the plan promised.
        assert pipeline._handle is handle
        assert pipeline._packs is handle


class TestTensorCacheShortCircuit:
    """Cache-hit semantics: pre-populated tensor → no model forward fires."""

    def test_tensor_cache_hit_skips_extract_contrastive(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
        pipeline = ExtractionPipeline(handle, handle, EventBus())

        name, profile = pipeline.extract("honest.deceptive")

        assert name == "honest.deceptive"
        assert "call_count" not in captured  # extract_contrastive never fired
        assert handle.scenarios_calls == 0
        assert handle.pairs_calls == 0
        assert handle.promote_calls == 1  # cache load promoted to device


class TestForceStatementsRegenerates:
    """force_statements=True: cache exists, statements regenerated."""

    def test_force_statements_bypasses_cache_and_calls_generators(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
        pipeline = ExtractionPipeline(handle, handle, EventBus())

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

    def test_explicit_scenarios_skips_scenario_generation(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))

        captured = _fake_extract(monkeypatch)

        from saklas.io.selectors import invalidate
        invalidate()

        handle = _StubHandle(tmp_path)
        pipeline = ExtractionPipeline(handle, handle, EventBus())

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

    def test_vector_extracted_event_fires_on_cache_hit(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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

        pipeline = ExtractionPipeline(handle, handle, bus)
        pipeline.extract("honest.deceptive")

        assert len(seen) == 1
        evt = seen[0]
        assert evt.name == "honest.deceptive"
        # Cache-hit metadata flows through unchanged.
        assert evt.metadata.get("method") == "contrastive_pca"


class TestSessionGate:
    """SaklasSession.extract gates on GenState.IDLE before delegating."""

    def test_session_extract_raises_when_generation_active(self) -> None:
        # Bypass SaklasSession.__init__ — we only need _gen_phase + _extraction.
        from types import SimpleNamespace
        from saklas.core.session import (
            ConcurrentExtractionError, GenState, SaklasSession,
        )

        import threading
        session = SaklasSession.__new__(SaklasSession)
        session._gen_phase = GenState.RUNNING
        session._gen_lock = threading.RLock()  # gate uses acquire(blocking=False), needs RLock
        # _extraction won't be reached — gate fires first.
        session._extraction = SimpleNamespace(extract=lambda *a, **kw: ("x", None))  # pyright: ignore[reportAttributeAccessIssue]  # stub bypasses ExtractionPipeline; gate fires before _extraction is reached

        with pytest.raises(ConcurrentExtractionError):
            session.extract("honest.deceptive")

    def test_concurrent_extraction_error_subclasses_saklas_error(self) -> None:
        from saklas.core.errors import SaklasError
        from saklas.core.session import ConcurrentExtractionError
        assert issubclass(ConcurrentExtractionError, SaklasError)
        assert issubclass(ConcurrentExtractionError, RuntimeError)
        # 409 conflict — same shape as ConcurrentGenerationError.
        code, _msg = ConcurrentExtractionError("x").user_message()
        assert code == 409


# ----------------------------------------------------------------------
# Role-augmented extraction (role-extraction Phase 7)
# ----------------------------------------------------------------------


class TestRoleVariant:
    """``ExtractionPipeline.extract(role=...)`` writes the per-role tensor
    file, threads the role through generator + extractor calls, and
    refuses to compose with ``sae=``.
    """

    def test_extract_with_role_writes_role_variant_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))

        captured = _fake_extract(monkeypatch)

        from saklas.io.selectors import invalidate
        invalidate()

        handle = _StubHandle(tmp_path)
        pipeline = ExtractionPipeline(handle, handle, EventBus())

        name, profile = pipeline.extract("honest.deceptive", role="pirate")

        # The variant-qualified output name surfaces the role.
        assert name == "honest.deceptive:role-pirate"
        # The extractor saw role= and model_type= via the kwargs dict.
        assert captured.get("call_count") == 1
        # The role-tagged tensor file landed on disk under the canonical
        # local-pack folder, matching the io.paths suffix scheme.
        from saklas.io.paths import tensor_filename
        expected_path = (
            tmp_path / "vectors" / "local" / "honest.deceptive"
            / tensor_filename("stub-model", role="pirate")
        )
        assert expected_path.exists()
        # The generators received the role kwarg too — same persona
        # drives scenario + pair generation as the activation pool.
        assert handle.last_scenarios_role == "pirate"
        assert handle.last_pairs_role == "pirate"

    def test_extract_role_rejects_with_sae(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))

        from saklas.io.selectors import invalidate
        invalidate()
        from saklas.core.sae import MockSaeBackend

        handle = _StubHandle(tmp_path)
        pipeline = ExtractionPipeline(handle, handle, EventBus())

        # Compose-time mutex — role and sae are both kind suffixes and
        # paths.py refuses to fuse them.  The pipeline surfaces the
        # error eagerly with a clear message before any forward-pass
        # work begins.
        sae_backend = MockSaeBackend(
            layers=frozenset({0, 1}), d_model=4, release="test-release",
        )
        with pytest.raises(ValueError, match="mutually exclusive"):
            pipeline.extract(
                "honest.deceptive", role="pirate", sae=sae_backend,
            )

    def test_extract_role_cache_hit_returns_role_qualified_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A pre-baked role tensor short-circuits the pipeline and the
        cached output name still carries the role suffix."""
        monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))

        captured = _fake_extract(monkeypatch)

        # Pre-populate a role-tagged tensor under local/<concept>/.
        from saklas.core.vectors import save_profile
        from saklas.io.packs import PackMetadata, hash_folder_files
        from saklas.io.paths import tensor_filename
        from saklas.io.selectors import invalidate
        invalidate()

        folder = tmp_path / "vectors" / "local" / "honest.deceptive"
        folder.mkdir(parents=True)
        save_profile(
            {0: torch.full((4,), 0.5)},
            str(folder / tensor_filename("stub-model", role="pirate")),
            {"method": "difference_of_means", "role": "pirate"},
        )
        PackMetadata(
            name="honest.deceptive", description="x", version="1.0.0",
            license="MIT", tags=[], recommended_alpha=0.5,
            source="local", files=hash_folder_files(folder),
        ).write(folder)

        handle = _StubHandle(tmp_path)
        pipeline = ExtractionPipeline(handle, handle, EventBus())

        name, _profile = pipeline.extract("honest.deceptive", role="pirate")

        assert name == "honest.deceptive:role-pirate"
        assert "call_count" not in captured  # cache hit — no extractor
