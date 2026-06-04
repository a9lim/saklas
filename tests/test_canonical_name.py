"""Unit tests for canonical_concept_name / _slug.

CPU-only: tests pure string normalization without loading a model.
Bipolar separator is `.` (see saklas.session.BIPOLAR_SEP).
"""
from __future__ import annotations

from typing import Any

from saklas.core.session import (
    SaklasSession,
    _humanize_concept,
    _role_for,
    _split_composite_source,
    _system_for,
    canonical_concept_name,
)


class TestSlug:
    def test_monopolar(self):
        assert canonical_concept_name("happy") == "happy"

    def test_bipolar(self):
        assert canonical_concept_name("happy", "sad") == "happy.sad"

    def test_hyphen_normalizes_to_underscore(self):
        assert canonical_concept_name("high-context") == "high_context"

    def test_bipolar_with_hyphens(self):
        assert canonical_concept_name("high-context", "low-context") == "high_context.low_context"

    def test_slug_symmetry_hyphen_dot(self):
        # /steer happy - sad  and  /steer happy.sad  resolve identically.
        via_bipolar = canonical_concept_name("happy", "sad")
        via_composite = canonical_concept_name("happy.sad")
        assert via_bipolar == via_composite == "happy.sad"

    def test_slug_symmetry_multi_word(self):
        via_bipolar = canonical_concept_name("high-context", "low-context")
        via_composite = canonical_concept_name("high-context.low-context")
        assert via_bipolar == via_composite == "high_context.low_context"

    def test_whitespace_collapsed(self):
        assert canonical_concept_name("  happy   go   lucky  ") == "happy_go_lucky"

    def test_mixed_punctuation(self):
        assert canonical_concept_name("high/low-context!") == "high_low_context"

    def test_case_normalized(self):
        assert canonical_concept_name("HAPPY", "SAD") == "happy.sad"

    def test_order_matters(self):
        # sign is meaningful — (A, B) and (B, A) are distinct vectors
        assert canonical_concept_name("happy", "sad") != canonical_concept_name("sad", "happy")


class TestHumanize:
    def test_humanize_pure_string(self):
        assert _humanize_concept("artificial_intelligence") == "artificial intelligence"
        assert _humanize_concept("happy") == "happy"
        assert _humanize_concept("high_context") == "high context"

    def test_humanize_leaves_canonical_untouched(self):
        # Slug path is the identifier; humanize is for LLM prompts only.
        assert canonical_concept_name("artificial_intelligence") == "artificial_intelligence"

    def test_system_for_humanizes_and_keys_on_kind(self):
        """The A2 system prompt humanizes the slug and switches on kind."""
        abstract = _system_for("artificial intelligence", "abstract")
        assert "someone artificial intelligence" in abstract
        assert "artificial_intelligence" not in abstract
        concrete = _system_for("alien", "concrete")
        assert "an alien" in concrete          # a/an article
        assert "someone" not in concrete

    def test_role_for_kind(self):
        """Abstract -> someone_<slug>; concrete -> bare slug."""
        assert _role_for("artificial_intelligence", "abstract") == (
            "someone_artificial_intelligence"
        )
        assert _role_for("pirate", "concrete") == "pirate"

    def test_split_composite_source_splits_on_dot(self):
        # Composite "pos.neg" with no baseline: split into distinct poles.
        assert _split_composite_source("human.artificial_intelligence", None) == (
            "human", "artificial_intelligence",
        )

    def test_split_composite_source_passes_through_monopolar(self):
        # No dot: leave alone.
        assert _split_composite_source("honest", None) == ("honest", None)

    def test_split_composite_source_respects_explicit_baseline(self):
        # Explicit baseline wins — don't second-guess the caller even if
        # ``concept`` also contains a dot.
        assert _split_composite_source(
            "human.ai", "override",
        ) == ("human.ai", "override")

    def test_split_composite_source_strips_whitespace(self):
        assert _split_composite_source("human . ai", None) == ("human", "ai")

    def test_generate_responses_humanizes_and_sets_role(self, monkeypatch: Any):
        """generate_responses humanizes the system prompt + sets the kind role."""
        from saklas.core import vectors as V
        monkeypatch.setattr(
            V, "_load_baseline_prompts", lambda: ["How are you today?"],
        )
        captured: dict[str, Any] = {}

        class _FakeSession(SaklasSession):
            def __init__(self) -> None:
                pass

            def _run_generator(self, system_msg: str, prompt: str, max_new_tokens: int, **kw: Any) -> str:
                captured["system"] = system_msg
                captured["role"] = kw.get("role")
                captured.setdefault("prompts", []).append(prompt)
                return "a generated in-character response"

        out = _FakeSession().generate_responses(
            ["artificial_intelligence"], ["abstract"], samples_per_prompt=2,
        )
        assert "someone artificial intelligence" in captured["system"]
        assert "artificial_intelligence" not in captured["system"]
        assert captured["role"] == "someone_artificial_intelligence"
        # 2 samples x 1 baseline prompt = 2 responses, each paired with the bare
        # prompt (the shared brevity directive rides the system prompt instead).
        from saklas.core.vectors import _LENGTH_DIRECTIVE
        assert len(out["artificial_intelligence"]) == 2
        assert captured["prompts"] == ["How are you today?", "How are you today?"]
        assert captured["system"].startswith(_LENGTH_DIRECTIVE)
