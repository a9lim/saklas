"""Mocked-LM tests for SaklasSession.generate_statements.

The unified statement-corpus generator. Two cell-fill modes are
covered: ``share_moment=False`` (the K-tuple discover-mode shape, one
LM call per (scenario, concept) cell) and ``share_moment=True`` (the
moment-shared bipolar contrastive shape, one LM call per scenario
writing N statements per moment).  All tests bypass the real model —
they subclass :class:`SaklasSession` with a blank ``__init__`` and
override ``_run_generator`` to return canned strings.  CPU-only, no IO.
"""
from __future__ import annotations

import re

import pytest

from saklas.core.session import SaklasSession


_SPEAKER_BINDING_RE = re.compile(r'([a-z])="([^"]+)"')


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


def _grouped_statements_text(
    k: int, n_concepts: int, scenario_tag: str, concept_tags: list[str],
) -> str:
    """Canned moment-shared groups: K groups × N concepts each.

    Each line is ``<group_idx><letter>. <body>``.  The body embeds the
    scenario tag and the moment index so the test can recover both.
    """
    lines = []
    for group_idx in range(1, k + 1):
        for letter_idx, concept_tag in enumerate(concept_tags[:n_concepts]):
            letter = chr(ord("a") + letter_idx)
            lines.append(
                f"{group_idx}{letter}. statement for {concept_tag} "
                f"under {scenario_tag} — moment {group_idx}, more than "
                f"twelve words filler text here yes."
            )
    return "\n".join(lines)


def _fake_session_class(*, share_moment: bool = False):
    """Build a fake SaklasSession that records every _run_generator call.

    Returns the class so each test can instantiate fresh.  Each instance
    accumulates calls on ``self._calls`` as
    ``(prompt, system_msg, max_new_tokens)`` tuples.  The fake dispatches
    on prompt content to emit scenario lists, numbered statements, or
    moment-shared groups.
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
                self._scenario_call_count += 1
                return _scenarios_text(_N_SCENARIOS)
            # Statement-generation call.  Recover the scenario tag and
            # concept list from the prompt — both appear verbatim in
            # the canned text the prompt builder emits.
            scenario_tag = "unknown_scenario"
            for line in prompt.split("\n"):
                line = line.strip()
                if line.startswith("Domain: "):
                    scenario_tag = line[len("Domain: "):].strip(".")
            if share_moment:
                # Pull the concept list from the ``Speakers: a="...",
                # b="..."`` inline binding line the share_moment prompt
                # emits.
                concept_tags: list[str] = []
                for line in prompt.split("\n"):
                    bindings = _SPEAKER_BINDING_RE.findall(line)
                    if bindings:
                        concept_tags = [name for _letter, name in bindings]
                        break
                if not concept_tags:
                    return ""
                return _grouped_statements_text(
                    _K, len(concept_tags), scenario_tag, concept_tags,
                )
            else:
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


def test_calls_lm_once_per_scenario_plus_per_cell_independent_mode():
    """independent mode: 1 scenario call + 1 statement call per (scenario, concept)."""
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


def test_calls_lm_once_per_scenario_in_share_moment_mode():
    """share_moment=True: 1 scenario call + 1 statement call per scenario.

    All concepts share the same per-scenario LM call so the moments are
    coordinated across the row — the bipolar contrastive shape.
    """
    session = _fake_session_class(share_moment=True)()
    concepts = ["pirate", "caveman", "robot"]
    session.generate_statements(
        concepts,
        n_scenarios=_N_SCENARIOS,
        statements_per_cell=_K,
        share_moment=True,
    )
    expected = 1 + _N_SCENARIOS
    assert len(session._calls) == expected
    assert session._scenario_call_count == 1


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


def test_literal_directive_in_every_statement_prompt_independent_mode():
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


def test_literal_directive_in_every_statement_prompt_share_moment_mode():
    """Moment-shared prompts must also carry the literal-reading directive."""
    session = _fake_session_class(share_moment=True)()
    session.generate_statements(
        ["pirate", "caveman", "robot"],
        n_scenarios=_N_SCENARIOS,
        statements_per_cell=_K,
        share_moment=True,
    )
    for prompt, _system, _max in session._calls[1:]:
        assert "literal" in prompt, (
            f"literal-reading directive missing from a moment-shared "
            f"prompt: {prompt[:200]!r}"
        )


def test_share_moment_prompt_binds_each_concept_to_a_letter():
    """``share_moment=True`` prompts must declare the speaker binding
    (``a="X", b="Y", ...``) so the format example's letter labels are
    unambiguous.
    """
    session = _fake_session_class(share_moment=True)()
    session.generate_statements(
        ["pirate", "caveman", "robot"],
        n_scenarios=_N_SCENARIOS,
        statements_per_cell=_K,
        share_moment=True,
    )
    statement_prompt = session._calls[1][0]
    for letter, concept in zip("abc", ("pirate", "caveman", "robot")):
        assert (
            f'{letter}="{concept}"' in statement_prompt
        ), (
            f"speaker binding for letter {letter} / concept {concept} "
            f"missing from share_moment prompt"
        )


# ----------------------------------------- scenario-sharing structural invariant ---

def test_scenario_index_shared_across_concept_corpora_independent_mode():
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


def test_moment_shared_across_concept_corpora_in_share_moment_mode():
    """share_moment=True: at index j within a scenario block, every
    concept's statement comes from the *same* moment.
    """
    session = _fake_session_class(share_moment=True)()
    concepts = ["pirate", "caveman", "robot"]
    out = session.generate_statements(
        concepts,
        n_scenarios=_N_SCENARIOS,
        statements_per_cell=_K,
        share_moment=True,
    )
    for scenario_block in range(_N_SCENARIOS):
        for j in range(_K):
            flat_idx = scenario_block * _K + j
            scenario_tags = set()
            moment_tags = set()
            for c in concepts:
                stmt = out[c][flat_idx]
                scenario_tags.add(
                    next(
                        (tok for tok in stmt.split() if tok.startswith("scenario_")),
                        "",
                    )
                )
                tokens = stmt.split()
                for ti, tok in enumerate(tokens):
                    if tok == "moment" and ti + 1 < len(tokens):
                        moment_tags.add(tokens[ti + 1].rstrip(","))
                        break
            assert len(scenario_tags) == 1, (
                f"position {flat_idx} disagrees on scenario across "
                f"concepts: {scenario_tags}"
            )
            assert len(moment_tags) == 1, (
                f"position {flat_idx} disagrees on moment across "
                f"concepts: {moment_tags}"
            )


# --------------------------------------------------------------- error paths ---

def test_rejects_empty_concept_list():
    session = _fake_session_class()()
    with pytest.raises(ValueError, match=">= 1 concept"):
        session.generate_statements([])


def test_rejects_share_moment_with_single_concept():
    """``share_moment=True`` is meaningless with only one speaker."""
    session = _fake_session_class()()
    with pytest.raises(ValueError, match="share_moment.*>= 2"):
        session.generate_statements(["lone"], share_moment=True)


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


def test_rejects_share_moment_with_too_many_concepts():
    """The grouped parser uses single-letter labels (a–z)."""
    session = _fake_session_class()()
    too_many = [f"c{i}" for i in range(27)]
    with pytest.raises(ValueError, match="up to 26"):
        session.generate_statements(too_many, share_moment=True)


def test_raises_when_scenario_generation_fails():
    """If the LM never produces any scenarios, surface a clean error."""
    class _BadSession(SaklasSession):
        def __init__(self):
            pass

        def _run_generator(self, *a, **kw):
            return "no list here"

    with pytest.raises(ValueError, match="could not generate scenarios"):
        _BadSession().generate_statements(
            ["a", "b", "c"], n_scenarios=2,
            statements_per_cell=2,
        )


def test_short_cell_pads_to_K_preserving_shape_independent_mode():
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


def test_short_cell_pads_to_K_preserving_shape_share_moment_mode():
    """Moment-shared shorts also pad up to K groups so the per-concept
    corpora stay the same length.
    """
    class _PartialSession(SaklasSession):
        def __init__(self):
            self.calls = 0

        def _run_generator(self, system_msg, prompt, max_new_tokens, **_kwargs):
            self.calls += 1
            if "situational domains" in prompt:
                return _scenarios_text(_N_SCENARIOS)
            return (
                "1a. only one moment for pole a, more than twelve words filler.\n"
                "1b. only one moment for pole b, more than twelve words filler.\n"
            )

    session = _PartialSession()
    out = session.generate_statements(
        ["a", "b"],
        n_scenarios=_N_SCENARIOS,
        statements_per_cell=_K,
        share_moment=True,
    )
    for c, corpus in out.items():
        assert len(corpus) == _N_SCENARIOS * _K, (
            f"corpus for {c!r} ragged in share_moment mode: {len(corpus)}"
        )


# --------------------------------------- bipolar parity (zip → legacy pairs) ---

def test_share_moment_two_concepts_zip_recovers_paired_contrastive_corpus():
    """The 2-concept ``share_moment=True`` shape is the legacy pair
    generator's contract: caller zips two corpora to recover positive/
    negative pairs aligned by moment.
    """
    session = _fake_session_class(share_moment=True)()
    out = session.generate_statements(
        ["happy", "sad"],
        n_scenarios=_N_SCENARIOS,
        statements_per_cell=_K,
        share_moment=True,
    )
    pairs = list(zip(out["happy"], out["sad"]))
    assert len(pairs) == _N_SCENARIOS * _K
    for happy_stmt, sad_stmt in pairs:
        happy_tokens = happy_stmt.split()
        sad_tokens = sad_stmt.split()
        for tokens in (happy_tokens, sad_tokens):
            assert "moment" in tokens
        h_moment = happy_tokens[happy_tokens.index("moment") + 1]
        s_moment = sad_tokens[sad_tokens.index("moment") + 1]
        assert h_moment == s_moment, (
            f"paired statements disagree on moment: "
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
            self, system_msg, prompt, max_new_tokens, **_kwargs,
        ):
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
    )
    # 1 scenario call + n_scenarios cell calls (single concept, so
    # no inner loop iterations).
    assert len(session._calls) == 1 + _N_SCENARIOS


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
