"""Unit tests for canonical_concept_name / _slug.

CPU-only: tests pure string normalization without loading a model.
Bipolar separator is `.` (see saklas.session.BIPOLAR_SEP).
"""
from __future__ import annotations

from saklas.core.session import (
    SaklasSession,
    _humanize_concept,
    _split_composite_source,
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

    def test_scenarios_prompt_uses_humanized_form(self):
        """Underscored slugs become spaces in the LLM-facing prompt."""
        captured = {}

        class _FakeSession(SaklasSession):
            def __init__(self):  # bypass real construction
                pass

            def _run_generator(self, system_msg, prompt, max_new_tokens):
                captured["prompt"] = prompt
                return "\n".join(f"{i}. domain {i}" for i in range(1, 10))

        _FakeSession().generate_scenarios(
            "artificial_intelligence", baseline=None, n=9,
        )
        prompt = captured["prompt"]
        assert "artificial intelligence" in prompt
        assert "artificial_intelligence" not in prompt

    def test_scenarios_prompt_humanizes_baseline(self):
        captured = {}

        class _FakeSession(SaklasSession):
            def __init__(self):
                pass

            def _run_generator(self, system_msg, prompt, max_new_tokens):
                captured["prompt"] = prompt
                return "\n".join(f"{i}. domain {i}" for i in range(1, 10))

        _FakeSession().generate_scenarios(
            "high_context", baseline="low_context", n=9,
        )
        prompt = captured["prompt"]
        assert "high context" in prompt
        assert "low context" in prompt
        assert "high_context" not in prompt
        assert "low_context" not in prompt

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

    def test_pairs_prompt_uses_humanized_form(self):
        captured = {}

        class _FakeSession(SaklasSession):
            def __init__(self):
                pass

            def _run_generator(self, system_msg, prompt, max_new_tokens):
                captured.setdefault("prompts", []).append(prompt)
                return (
                    "1a. Statement one.\n1b. Statement two.\n"
                    "2a. Statement three.\n2b. Statement four.\n"
                )

        _FakeSession().generate_pairs(
            "artificial_intelligence",
            baseline=None, n=2, scenarios=["a domain"],
        )
        assert captured["prompts"], "pair generator was not invoked"
        prompt = captured["prompts"][0]
        assert "artificial intelligence" in prompt
        assert "artificial_intelligence" not in prompt
