"""Mocked-LM tests for SaklasSession.generate_concept_statements.

The K-tuple generalization of generate_pairs for discover-mode
manifolds.  All tests bypass the real model — they subclass
:class:`SaklasSession` with a blank ``__init__`` and override
``_run_generator`` to return canned strings.  CPU-only, no IO.
"""
from __future__ import annotations

import pytest

from saklas.core.session import SaklasSession


_N_SCENARIOS = 3
_K = 4  # statements per (concept, scenario) cell


def _scenarios_text(n: int) -> str:
    """Canned scenario list — n entries, predictable labels."""
    return "\n".join(f"{i}. scenario_{i}" for i in range(1, n + 1))


def _statements_text(k: int, scenario_tag: str, concept_tag: str) -> str:
    """Canned per-cell statement list — k entries that string-encode the
    scenario index so the test can recover scenario sharing from the
    flat output corpora.
    """
    return "\n".join(
        f"{i}. statement for {concept_tag} under {scenario_tag} — "
        f"index {i}, more than twelve words filler text here yes."
        for i in range(1, k + 1)
    )


def _fake_session_class():
    """Build a fake SaklasSession that records every _run_generator call.

    Returns the class so each test can instantiate fresh.  Each instance
    accumulates calls on ``self._calls`` as
    ``(prompt, system_msg, max_new_tokens)`` tuples.
    """
    class _FakeSession(SaklasSession):
        def __init__(self):  # bypass real construction
            self._calls: list[tuple[str, str, int]] = []
            self._scenario_call_count = 0

        def _run_generator(
            self, system_msg, prompt, max_new_tokens, **_kwargs,
        ):
            self._calls.append((prompt, system_msg, max_new_tokens))
            if "situational domains" in prompt:
                # Scenario-generation call.
                self._scenario_call_count += 1
                return _scenarios_text(_N_SCENARIOS)
            # Statement-generation call.  Recover the scenario tag and
            # concept tag from the prompt — both appear verbatim in the
            # canned text the prompt builder emits.
            scenario_tag = "unknown_scenario"
            concept_tag = "unknown_concept"
            for line in prompt.split("\n"):
                line = line.strip()
                if line.startswith("Concept: "):
                    concept_tag = line[len("Concept: "):].strip("\".")
                if line.startswith("Domain: "):
                    scenario_tag = line[len("Domain: "):].strip(".")
            return _statements_text(_K, scenario_tag, concept_tag)

    return _FakeSession


# ----------------------------------------------------- shape + key invariants ---

def test_returns_dict_of_per_concept_corpora_with_right_shape():
    session = _fake_session_class()()
    out = session.generate_concept_statements(
        ["pirate", "caveman", "assistant", "scholar", "robot"],
        n_scenarios=_N_SCENARIOS,
        statements_per_concept_per_scenario=_K,
    )
    # One corpus per concept, in input order.
    assert list(out.keys()) == [
        "pirate", "caveman", "assistant", "scholar", "robot",
    ]
    # Each corpus has exactly n_scenarios * K statements.
    for c, corpus in out.items():
        assert len(corpus) == _N_SCENARIOS * _K, (
            f"corpus for {c!r} has {len(corpus)}, expected "
            f"{_N_SCENARIOS * _K}"
        )
        assert all(isinstance(s, str) and s for s in corpus)


def test_calls_lm_once_per_scenario_plus_per_cell():
    """One scenario call + one statement call per (scenario, concept) cell."""
    session = _fake_session_class()()
    concepts = ["pirate", "caveman", "robot"]
    session.generate_concept_statements(
        concepts,
        n_scenarios=_N_SCENARIOS,
        statements_per_concept_per_scenario=_K,
    )
    # 1 scenario call + N_scenarios * len(concepts) statement calls.
    expected = 1 + _N_SCENARIOS * len(concepts)
    assert len(session._calls) == expected
    assert session._scenario_call_count == 1


# ----------------------------------------------------- prompt-shape invariants ---

def test_scenario_prompt_contains_every_concept_name():
    """The scenario generator sees the whole concept list at once."""
    session = _fake_session_class()()
    session.generate_concept_statements(
        ["pirate", "caveman", "robot"],
        n_scenarios=_N_SCENARIOS,
        statements_per_concept_per_scenario=_K,
    )
    scenario_prompt = session._calls[0][0]
    assert "situational domains" in scenario_prompt
    for c in ("pirate", "caveman", "robot"):
        assert c in scenario_prompt, f"scenario prompt missing {c!r}"


def test_humanized_concept_names_in_prompts():
    """Underscored slugs read as spaces in LLM-facing prompts."""
    session = _fake_session_class()()
    session.generate_concept_statements(
        ["sea_captain", "cave_dweller"],
        n_scenarios=_N_SCENARIOS,
        statements_per_concept_per_scenario=_K,
    )
    scenario_prompt = session._calls[0][0]
    assert "sea captain" in scenario_prompt
    assert "cave dweller" in scenario_prompt
    assert "sea_captain" not in scenario_prompt
    assert "cave_dweller" not in scenario_prompt


def test_anti_allegory_clause_in_scenario_prompt():
    """The 'don't force human-social framing' clause is load-bearing."""
    session = _fake_session_class()()
    session.generate_concept_statements(
        ["pirate", "caveman", "robot"],
        n_scenarios=_N_SCENARIOS,
        statements_per_concept_per_scenario=_K,
    )
    scenario_prompt = session._calls[0][0]
    assert "human-social framing" in scenario_prompt, (
        "anti-allegory clause missing from scenario prompt — discover "
        "fits on heaps containing non-human concepts will collapse to "
        "a stereotyped-human centroid"
    )
    assert "literally" in scenario_prompt or "literal" in scenario_prompt


def test_anti_allegory_clause_in_every_statement_prompt():
    """Every per-cell statement prompt must carry the clause too."""
    session = _fake_session_class()()
    session.generate_concept_statements(
        ["pirate", "caveman", "robot"],
        n_scenarios=_N_SCENARIOS,
        statements_per_concept_per_scenario=_K,
    )
    # Skip the first (scenario) call; every subsequent prompt is a cell.
    for prompt, _system, _max in session._calls[1:]:
        assert "human-social framing" in prompt, (
            f"anti-allegory clause missing from a statement prompt: "
            f"{prompt[:200]!r}"
        )
        assert "literal" in prompt or "literally" in prompt


# ----------------------------------------- scenario-sharing structural invariant ---

def test_scenario_index_shared_across_concept_corpora():
    """Statement index ``j`` of every concept came from the same scenario.

    The K-tuple analogue of generate_pairs's paired-contrast structure:
    without shared scenarios, per-concept centroids would mix concept
    signal with scenario signal and the discover-mode layout would
    surface scenario as the dominant axis.

    The fake _run_generator embeds the scenario tag in each statement;
    this test recovers that tag from each concept's corpus at every
    flat index and asserts every concept sees the same scenario at
    each position within a scenario block.
    """
    session = _fake_session_class()()
    concepts = ["pirate", "caveman", "robot"]
    out = session.generate_concept_statements(
        concepts,
        n_scenarios=_N_SCENARIOS,
        statements_per_concept_per_scenario=_K,
    )
    # For each scenario block [j*K .. j*K+K), every concept's statements
    # in that block should reference the same scenario tag.
    for scenario_block in range(_N_SCENARIOS):
        per_concept_tags = []
        for c in concepts:
            block = out[c][scenario_block * _K:(scenario_block + 1) * _K]
            # Each statement embeds the scenario tag verbatim
            # (e.g. "under scenario_1").
            tags = {
                next(
                    (token for token in stmt.split() if token.startswith("scenario_")),
                    "",
                )
                for stmt in block
            }
            assert len(tags) == 1, (
                f"concept {c!r} block {scenario_block} spans multiple "
                f"scenarios: {tags}"
            )
            per_concept_tags.append(next(iter(tags)))
        assert len(set(per_concept_tags)) == 1, (
            f"scenario block {scenario_block} disagrees across "
            f"concepts: {dict(zip(concepts, per_concept_tags))}"
        )


# --------------------------------------------------------------- error paths ---

def test_rejects_fewer_than_two_concepts():
    session = _fake_session_class()()
    with pytest.raises(ValueError, match=">= 2 concepts"):
        session.generate_concept_statements(["lone"])


def test_rejects_duplicate_concepts():
    session = _fake_session_class()()
    with pytest.raises(ValueError, match="duplicate concept"):
        session.generate_concept_statements(["a", "b", "a"])


def test_rejects_zero_scenarios():
    session = _fake_session_class()()
    with pytest.raises(ValueError, match="must both be > 0"):
        session.generate_concept_statements(
            ["a", "b", "c"], n_scenarios=0,
        )


def test_rejects_zero_statements_per_cell():
    session = _fake_session_class()()
    with pytest.raises(ValueError, match="must both be > 0"):
        session.generate_concept_statements(
            ["a", "b", "c"], statements_per_concept_per_scenario=0,
        )


def test_raises_when_scenario_generation_fails():
    """If the LM never produces any scenarios, surface a clean error."""
    class _BadSession(SaklasSession):
        def __init__(self):
            pass

        def _run_generator(self, *a, **kw):
            return "no list here"

    with pytest.raises(ValueError, match="could not generate scenarios"):
        _BadSession().generate_concept_statements(
            ["a", "b", "c"], n_scenarios=2,
            statements_per_concept_per_scenario=2,
        )


def test_short_cell_pads_to_K_preserving_shape():
    """A short statement-generation result is padded so the scenario index
    invariant survives — a ragged corpus would silently miscount
    scenario positions downstream.
    """
    class _PartialSession(SaklasSession):
        def __init__(self):
            self.calls = 0

        def _run_generator(self, system_msg, prompt, max_new_tokens, **_kwargs):
            self.calls += 1
            if "situational domains" in prompt:
                return _scenarios_text(_N_SCENARIOS)
            # Always return only 1 statement when K > 1 was requested.
            return "1. only one statement, more than twelve words filler text yes."

    session = _PartialSession()
    out = session.generate_concept_statements(
        ["a", "b", "c"],
        n_scenarios=_N_SCENARIOS,
        statements_per_concept_per_scenario=_K,
    )
    # Every corpus is still exactly _N_SCENARIOS * _K long — padding
    # preserved the shape.
    for c, corpus in out.items():
        assert len(corpus) == _N_SCENARIOS * _K, (
            f"corpus for {c!r} ragged: {len(corpus)}"
        )
