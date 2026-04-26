"""CPU-only unit tests for saklas.cloning helpers.

End-to-end extraction (which requires loading a model) is covered
by the GPU-gated test in test_session.py. These tests exercise the
pure helpers: corpus filtering, sampling, chunking, prompt building,
and numbered-list parsing.
"""
from __future__ import annotations

from pathlib import Path
import random

import pytest

from saklas.io.cloning import (
    _BATCH_SIZE,
    _build_neutralize_prompt,
    _chunk,
    _filter_corpus,
    _parse_numbered,
    _sample_lines,
)


# -- _filter_corpus ---------------------------------------------------------

def test_filter_corpus_length_filter(tmp_path: Path):
    p = tmp_path / "c.txt"
    p.write_text(
        "this line has more than five words\n"
        "short line here\n"
        "another line that exceeds the threshold\n",
        encoding="utf-8",
    )
    out = _filter_corpus(p)
    assert "this line has more than five words" in out
    assert "another line that exceeds the threshold" in out
    assert "short line here" not in out


def test_filter_corpus_dedupes(tmp_path: Path):
    p = tmp_path / "c.txt"
    p.write_text(
        "line one with plenty of words\n"
        "line two with plenty of words\n"
        "line one with plenty of words\n",
        encoding="utf-8",
    )
    out = _filter_corpus(p)
    assert out.count("line one with plenty of words") == 1
    assert len(out) == 2


def test_filter_corpus_strips_whitespace(tmp_path: Path):
    p = tmp_path / "c.txt"
    p.write_text("   padded line has plenty of words   \n", encoding="utf-8")
    out = _filter_corpus(p)
    assert out == ["padded line has plenty of words"]


def test_filter_corpus_handles_bom(tmp_path: Path):
    p = tmp_path / "c.txt"
    p.write_bytes(
        "\ufeffthis line has more than five words\n".encode("utf-8")
    )
    out = _filter_corpus(p)
    assert out == ["this line has more than five words"]
    assert not out[0].startswith("\ufeff")


def test_filter_corpus_bad_bytes(tmp_path: Path):
    p = tmp_path / "c.txt"
    p.write_bytes(
        b"good line with plenty of words here\n"
        b"bad line \xff\xfe with plenty of words\n"
    )
    # Should not raise
    out = _filter_corpus(p)
    assert any("good line" in line for line in out)
    assert len(out) == 2


# -- _sample_lines ----------------------------------------------------------

def test_sample_lines_seeded():
    lines = [f"line {i} with enough words here" for i in range(100)]
    a = _sample_lines(lines, 10, random.Random(42))
    b = _sample_lines(lines, 10, random.Random(42))
    c = _sample_lines(lines, 10, random.Random(7))
    assert a == b
    assert a != c


def test_sample_lines_empty_raises():
    with pytest.raises(ValueError):
        _sample_lines([], 5, random.Random(0))


def test_sample_lines_clamps_n():
    lines = ["a", "b", "c"]
    out = _sample_lines(lines, 10, random.Random(0))
    assert sorted(out) == sorted(lines)


def test_sample_lines_n_equals_len():
    lines = ["a", "b", "c", "d"]
    out = _sample_lines(lines, 4, random.Random(0))
    assert sorted(out) == sorted(lines)


# -- _chunk -----------------------------------------------------------------

def test_chunk_complete_batches():
    items = [str(i) for i in range(10)]
    batches = _chunk(items, 5)
    assert len(batches) == 2
    assert all(len(b) == 5 for b in batches)


def test_chunk_ragged_last_batch():
    items = [str(i) for i in range(7)]
    batches = _chunk(items, 3)
    assert [len(b) for b in batches] == [3, 3, 1]


def test_chunk_batch_size_larger_than_input():
    items = ["1", "2"]
    batches = _chunk(items, 10)
    assert batches == [["1", "2"]]


# -- _build_neutralize_prompt ----------------------------------------------

def test_build_neutralize_prompt_contains_all_lines():
    batch = ["alpha line", "bravo line", "charlie line"]
    prompt = _build_neutralize_prompt(batch)
    for line in batch:
        assert line in prompt
    # Ordering check
    assert prompt.index("alpha") < prompt.index("bravo") < prompt.index("charlie")
    assert "neutral" in prompt.lower()
    assert "preserve meaning" in prompt.lower()


def test_build_neutralize_prompt_numbering():
    batch = ["one", "two", "three"]
    prompt = _build_neutralize_prompt(batch)
    assert "1. one" in prompt
    assert "2. two" in prompt
    assert "3. three" in prompt
    # Exact count mentioned
    assert "3" in prompt


# -- _parse_numbered --------------------------------------------------------

def test_parse_numbered_happy():
    out = _parse_numbered("1. foo\n2. bar\n3. baz", 3)
    assert out == ["foo", "bar", "baz"]


def test_parse_numbered_with_preamble():
    out = _parse_numbered("Sure, here are the rewrites:\n1. foo\n2. bar\n3. baz", 3)
    assert out == ["foo", "bar", "baz"]


def test_parse_numbered_with_trailing_notes():
    out = _parse_numbered("1. foo\n2. bar\n3. baz\n\nLet me know if you want changes.", 3)
    assert out == ["foo", "bar", "baz"]


def test_parse_numbered_count_mismatch():
    assert _parse_numbered("1. foo\n2. bar", 3) is None


def test_parse_numbered_missing_number():
    assert _parse_numbered("1. foo\n3. baz", 3) is None


def test_parse_numbered_out_of_order():
    assert _parse_numbered("2. foo\n1. bar", 2) is None


def test_parse_numbered_preserves_empty_rewrite():
    # Empty rewrites are preserved here — pair-assembly drops them
    # individually so the rest of the batch still contributes.
    out = _parse_numbered("1. foo\n2.   \n3. baz", 3)
    assert out == ["foo", "", "baz"]


def test_parse_numbered_accepts_paren():
    out = _parse_numbered("1) foo\n2) bar", 2)
    assert out == ["foo", "bar"]


# -- module constants -------------------------------------------------------

def test_batch_size_is_five():
    assert _BATCH_SIZE == 5


