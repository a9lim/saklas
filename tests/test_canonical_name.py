"""Unit tests for canonical_concept_name / _slug.

CPU-only: tests pure string normalization without loading a model.
Bipolar separator is `.` (see saklas.session.BIPOLAR_SEP).
"""
from __future__ import annotations

from saklas.session import canonical_concept_name


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
