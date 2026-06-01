"""Mocked-LM tests for SaklasSession.generate_statements.

The unified statement-corpus generator: one LM call per (scenario,
concept) cell, scenarios shared across the row so statement index ``j``
of every concept came from the same scenario.  A steering vector is the
2-concept case (the caller zips the two corpora).  All tests bypass the
real model — they subclass :class:`SaklasSession` with a blank
``__init__`` and override ``_run_generator`` to return canned strings.
CPU-only, no IO.
"""
from __future__ import annotations

from typing import Any

import pytest

from saklas.core.session import SaklasSession


_N_SCENARIOS = 3
_K = 4  # statements per (scenario, concept) cell


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
    ``(prompt, system_msg, max_new_tokens)`` tuples.  The fake dispatches
    on prompt content to emit a scenario list or numbered statements.
    """
    class _FakeSession(SaklasSession):
        def __init__(self):  # bypass real construction
            self._calls: list[tuple[str, str, int]] = []
            self._scenario_call_count = 0

        def _run_generator(
            self, system_msg: Any, prompt: Any, max_new_tokens: Any, **_kwargs: Any,
        ) -> Any:
            self._calls.append((prompt, system_msg, max_new_tokens))
            if "situational domains" in prompt:
                self._scenario_call_count += 1
                return _scenarios_text(_N_SCENARIOS)
            # Statement-generation call.  Recover the scenario tag and
            # concept from the prompt — both appear verbatim in the
            # canned text the prompt builder emits.
            scenario_tag = "unknown_scenario"
            for line in prompt.split("\n"):
                line = line.strip()
                if line.startswith("Domain: "):
                    scenario_tag = line[len("Domain: "):].strip(".")
            concept_tag = "unknown_concept"
            for line in prompt.split("\n"):
                line = line.strip()
                if line.startswith("Concept: "):
                    concept_tag = line[len("Concept: "):].strip("\".")
            return _statements_text(_K, scenario_tag, concept_tag)

    return _FakeSession


# ----------------------------------------------------- shape + key invariants ---

def test_returns_dict_of_per_concept_corpora_with_right_shape():
    session = _fake_session_class()()
    out = session.generate_statements(
        ["pirate", "caveman", "assistant", "scholar", "robot"],
        n_scenarios=_N_SCENARIOS,
        statements_per_cell=_K,
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
    """1 scenario call + 1 statement call per (scenario, concept) cell."""
    session = _fake_session_class()()
    concepts = ["pirate", "caveman", "robot"]
    session.generate_statements(
        concepts,
        n_scenarios=_N_SCENARIOS,
        statements_per_cell=_K,
    )
    expected = 1 + _N_SCENARIOS * len(concepts)
    assert len(session._calls) == expected
    assert session._scenario_call_count == 1


# ----------------------------------------------------- streaming sinks ---

def test_streaming_sinks_match_all_at_once():
    """on_corpus streams the same per-concept corpora the return dict
    would carry; with a sink set the method retains nothing (empty
    return) and emits each node in concept-completion order."""
    concepts = ["pirate", "caveman", "robot", "scholar"]
    ref = _fake_session_class()().generate_statements(
        concepts, n_scenarios=_N_SCENARIOS, statements_per_cell=_K,
    )
    streamed: dict[str, list[str]] = {}
    order: list[str] = []
    seen_scenarios: list[list[str]] = []

    def _corpus(label: str, stmts: list[str]) -> None:
        order.append(label)
        streamed[label] = list(stmts)

    out = _fake_session_class()().generate_statements(
        concepts, n_scenarios=_N_SCENARIOS, statements_per_cell=_K,
        on_scenarios=lambda s: seen_scenarios.append(list(s)),
        on_corpus=_corpus,
    )
    assert out == {}                       # streaming retains nothing
    assert streamed == ref                 # identical content
    assert order == concepts               # concept-outer completion order
    assert len(seen_scenarios) == 1        # fires exactly once
    assert len(seen_scenarios[0]) == _N_SCENARIOS


def test_on_scenarios_echoes_passed_scenarios_without_generating():
    """Explicit scenarios are echoed to on_scenarios and skip the
    scenario-generation LM call entirely."""
    session = _fake_session_class()()
    seen: list[list[str]] = []
    session.generate_statements(
        ["pirate", "robot"],
        scenarios=["alpha", "beta"],
        statements_per_cell=_K,
        on_scenarios=lambda s: seen.append(list(s)),
    )
    assert seen == [["alpha", "beta"]]
    assert session._scenario_call_count == 0


# ----------------------------------------------------- prompt-shape invariants ---

def test_scenario_prompt_contains_every_concept_name():
    """The scenario generator sees the whole concept list at once."""
    session = _fake_session_class()()
    session.generate_statements(
        ["pirate", "caveman", "robot"],
        n_scenarios=_N_SCENARIOS,
        statements_per_cell=_K,
    )
    scenario_prompt = session._calls[0][0]
    assert "situational domains" in scenario_prompt
    for c in ("pirate", "caveman", "robot"):
        assert c in scenario_prompt, f"scenario prompt missing {c!r}"


def test_humanized_concept_names_in_prompts():
    """Underscored slugs read as spaces in LLM-facing prompts."""
    session = _fake_session_class()()
    session.generate_statements(
        ["sea_captain", "cave_dweller"],
        n_scenarios=_N_SCENARIOS,
        statements_per_cell=_K,
    )
    scenario_prompt = session._calls[0][0]
    assert "sea captain" in scenario_prompt
    assert "cave dweller" in scenario_prompt
    assert "sea_captain" not in scenario_prompt
    assert "cave_dweller" not in scenario_prompt


def test_literal_directive_in_scenario_prompt():
    """The ``literal concept`` directive is load-bearing — without it
    the model defaults non-human concepts to human-social allegory and
    extracted vectors degrade.
    """
    session = _fake_session_class()()
    session.generate_statements(
        ["pirate", "caveman", "robot"],
        n_scenarios=_N_SCENARIOS,
        statements_per_cell=_K,
    )
    scenario_prompt = session._calls[0][0]
    assert "literal" in scenario_prompt, (
        "anti-allegory directive missing from scenario prompt — without "
        "the 'literal concept' anchor, fits on heaps containing "
        "non-human concepts collapse to stereotyped-human centroids"
    )


def test_literal_directive_in_every_statement_prompt():
    """Every per-cell statement prompt must carry the literal-reading
    directive too.
    """
    session = _fake_session_class()()
    session.generate_statements(
        ["pirate", "caveman", "robot"],
        n_scenarios=_N_SCENARIOS,
        statements_per_cell=_K,
    )
    for prompt, _system, _max in session._calls[1:]:
        assert "literal" in prompt, (
            f"literal-reading directive missing from a statement prompt: "
            f"{prompt[:200]!r}"
        )


# ----------------------------------------- scenario-sharing structural invariant ---

def test_scenario_index_shared_across_concept_corpora():
    """Statement index ``j`` of every concept came from the same scenario.

    The K-tuple structural invariant: without shared scenarios,
    per-concept centroids would mix concept signal with scenario signal
    and the discover-mode layout would surface scenario as the dominant
    axis.
    """
    session = _fake_session_class()()
    concepts = ["pirate", "caveman", "robot"]
    out = session.generate_statements(
        concepts,
        n_scenarios=_N_SCENARIOS,
        statements_per_cell=_K,
    )
    for scenario_block in range(_N_SCENARIOS):
        per_concept_tags = []
        for c in concepts:
            block = out[c][scenario_block * _K:(scenario_block + 1) * _K]
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

def test_rejects_empty_concept_list():
    session = _fake_session_class()()
    with pytest.raises(ValueError, match=">= 1 concept"):
        session.generate_statements([])


def test_rejects_duplicate_concepts():
    session = _fake_session_class()()
    with pytest.raises(ValueError, match="duplicate concept"):
        session.generate_statements(["a", "b", "a"])


def test_rejects_zero_scenarios():
    session = _fake_session_class()()
    with pytest.raises(ValueError, match="must both be > 0"):
        session.generate_statements(
            ["a", "b", "c"], n_scenarios=0,
        )


def test_rejects_zero_statements_per_cell():
    session = _fake_session_class()()
    with pytest.raises(ValueError, match="must both be > 0"):
        session.generate_statements(
            ["a", "b", "c"], statements_per_cell=0,
        )


def test_raises_when_scenario_generation_fails():
    """If the LM never produces any scenarios, surface a clean error."""
    class _BadSession(SaklasSession):
        def __init__(self):
            pass

        def _run_generator(self, *a: Any, **kw: Any) -> Any:
            return "no list here"

    with pytest.raises(ValueError, match="could not generate scenarios"):
        _BadSession().generate_statements(
            ["a", "b", "c"], n_scenarios=2,
            statements_per_cell=2,
        )


def test_short_cell_pads_to_K_preserving_shape():
    """A short statement-generation result is padded so the scenario index
    invariant survives — a ragged corpus would silently miscount
    scenario positions downstream.
    """
    class _PartialSession(SaklasSession):
        def __init__(self):
            self.calls = 0

        def _run_generator(self, system_msg: Any, prompt: Any, max_new_tokens: Any, **_kwargs: Any) -> Any:
            self.calls += 1
            if "situational domains" in prompt:
                return _scenarios_text(_N_SCENARIOS)
            return "1. only one statement, more than twelve words filler text yes."

    session = _PartialSession()
    out = session.generate_statements(
        ["a", "b", "c"],
        n_scenarios=_N_SCENARIOS,
        statements_per_cell=_K,
    )
    for c, corpus in out.items():
        assert len(corpus) == _N_SCENARIOS * _K, (
            f"corpus for {c!r} ragged: {len(corpus)}"
        )


# --------------------------------------- bipolar parity (zip → contrastive pairs) ---

def test_two_concepts_zip_recovers_scenario_aligned_corpus():
    """A steering vector is the 2-concept case: the caller zips the two
    scenario-aligned corpora to recover positive/negative pairs.  DiM is
    centroid-based, so scenario-alignment (not moment-pairing) is the
    load-bearing structure.
    """
    session = _fake_session_class()()
    out = session.generate_statements(
        ["happy", "sad"],
        n_scenarios=_N_SCENARIOS,
        statements_per_cell=_K,
    )
    pairs = list(zip(out["happy"], out["sad"]))
    assert len(pairs) == _N_SCENARIOS * _K
    # Each zipped position draws both poles from the same scenario block.
    for block in range(_N_SCENARIOS):
        for j in range(_K):
            flat = block * _K + j
            happy_stmt, sad_stmt = pairs[flat]
            h_scn = next(t for t in happy_stmt.split() if t.startswith("scenario_"))
            s_scn = next(t for t in sad_stmt.split() if t.startswith("scenario_"))
            assert h_scn == s_scn, (
                f"position {flat} disagrees on scenario: "
                f"happy={happy_stmt!r}, sad={sad_stmt!r}"
            )


# --------------------------------------------- single-concept (neutrals path) ---

def _neutrals_fake_session_class():
    """Fake session that emits canned scenarios + numbered statements
    without trying to parse a concept name from the prompt.

    Used for the N=1 (neutrals) tests, where the prompt deliberately
    omits any ``Concept:`` header.
    """
    class _NeutralsFake(SaklasSession):
        def __init__(self):
            self._calls: list[tuple[str, str, int]] = []

        def _run_generator(
            self, system_msg: Any, prompt: Any, max_new_tokens: Any, **_kwargs: Any,
        ) -> Any:
            self._calls.append((prompt, system_msg, max_new_tokens))
            if "everyday" in prompt and "domains" in prompt:
                return _scenarios_text(_N_SCENARIOS)
            return _statements_text(_K, "scn", "neutral")

    return _NeutralsFake


def test_single_concept_returns_dict_with_one_key():
    """Neutrals path: dict has one entry, sized n_scenarios * K."""
    session = _neutrals_fake_session_class()()
    out = session.generate_statements(
        ["neutral"],
        n_scenarios=_N_SCENARIOS,
        statements_per_cell=_K,
        neutrals=True,
    )
    assert list(out.keys()) == ["neutral"]
    assert len(out["neutral"]) == _N_SCENARIOS * _K


def test_single_concept_scenario_prompt_omits_concept_naming():
    """N=1 scenario prompt should not name the concept (anti-allegory
    by absence — model writes from its default voice across domains).
    """
    session = _neutrals_fake_session_class()()
    session.generate_statements(
        ["neutral"],
        n_scenarios=_N_SCENARIOS,
        statements_per_cell=_K,
        neutrals=True,
    )
    scenario_prompt = session._calls[0][0]
    # Bracketed concept name should be absent (the N≥2 prompt has
    # ``Concepts: "X", ...``; the N=1 form drops it entirely).
    assert 'Concepts:' not in scenario_prompt, (
        "N=1 scenario prompt leaked a concept-list header"
    )
    assert "neutral" not in scenario_prompt, (
        "N=1 scenario prompt leaked the concept slug — defeats the "
        "no-concept-pull property of the neutrals path"
    )
    # The "everyday" anchor steers the model toward affect-neutral
    # domains rather than open-ended (potentially affect-loaded) ones.
    assert "everyday" in scenario_prompt


def test_single_concept_statement_prompt_omits_concept_naming():
    """N=1 statement prompt should write from the default voice."""
    session = _neutrals_fake_session_class()()
    session.generate_statements(
        ["neutral"],
        n_scenarios=_N_SCENARIOS,
        statements_per_cell=_K,
        neutrals=True,
    )
    # First call is scenarios; the next n_scenarios calls are per-cell
    # statement prompts.  All of them should be concept-free.
    for prompt, _system, _max in session._calls[1:]:
        assert 'Concept:' not in prompt, (
            f"N=1 statement prompt leaked a Concept: header: "
            f"{prompt[:200]!r}"
        )
        assert "literal concept" not in prompt, (
            "N=1 statement prompt should drop the 'as the literal "
            "concept' anchor — neutrals have no concept axis"
        )
        # The first-person directive is the load-bearing piece for
        # register-match with downstream contrastive corpora.
        assert "first-person" in prompt


def test_single_concept_calls_lm_once_per_scenario():
    """N=1: one scenario call + one statement call per scenario."""
    session = _neutrals_fake_session_class()()
    session.generate_statements(
        ["neutral"],
        n_scenarios=_N_SCENARIOS,
        statements_per_cell=_K,
        neutrals=True,
    )
    # 1 scenario call + n_scenarios cell calls (single concept, so
    # no inner loop iterations).
    assert len(session._calls) == 1 + _N_SCENARIOS


def test_single_concept_names_by_default():
    """Without neutrals=True a single concept is *named* (the discover
    one-node resume case) — the statement prompt carries Concept:."""
    session = _fake_session_class()()
    out = session.generate_statements(
        ["pirate"],
        n_scenarios=_N_SCENARIOS,
        statements_per_cell=_K,
    )
    assert list(out.keys()) == ["pirate"]
    assert len(out["pirate"]) == _N_SCENARIOS * _K
    # Scenario prompt names the concept; statement prompts carry Concept:.
    assert "pirate" in session._calls[0][0]
    assert any("Concept:" in prompt for prompt, _s, _m in session._calls[1:])


def test_neutrals_flag_rejects_multiple_concepts():
    session = _fake_session_class()()
    with pytest.raises(ValueError, match="single-concept baseline"):
        session.generate_statements(["a", "b"], neutrals=True)


# -------------------------------------------------------- scenarios override ---

def test_caller_scenarios_skip_scenario_generation():
    """Passing ``scenarios=...`` bypasses the scenario LM call."""
    session = _fake_session_class()()
    out = session.generate_statements(
        ["a", "b", "c"],
        scenarios=["a quiet morning", "a sudden storm"],
        statements_per_cell=_K,
    )
    assert session._scenario_call_count == 0
    assert len(session._calls) == 2 * 3  # 2 scenarios × 3 concepts
    for c, corpus in out.items():
        assert len(corpus) == 2 * _K
