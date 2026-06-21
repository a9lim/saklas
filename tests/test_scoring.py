"""CPU tests for ``saklas.core.scoring`` — the restricted-choice logit read.

The headline question this module answers is *"did steering shift the
distribution, not just the argmax?"*  These tests pin the math that backs that
answer on a tiny deterministic stub so it runs in CI with no model download:

- the per-candidate ``sum_logprob`` (joint ``log P(candidate | context)``),
  computed by the ``logsumexp``-normalized gather, against a hand-checked value;
- the ``mean_logprob`` length-normalization (``sum / n_tokens``);
- the two restricted-choice softmax *views* (``prob_sum`` over joint logprobs vs
  ``prob_mean`` over the length-normalized ones) — neither silently chosen;
- the degenerate-candidate exclusion (an empty / fully-prefix-absorbed choice
  carries ``sum_logprob`` 0.0 and ~0 restricted probability, not the largest);
- the ``_shared_prefix_len`` boundary-token merge that recovers the scored span;
- ``steering=`` actually wrapping the forward (the before/after read) and the
  steering label round-trip;
- ``score_template`` fanning one ``ChoiceScores`` out per context.

The model is a :class:`FakeLogitsModel` from ``conftest`` whose ``logits_fn``
emits *fixed* per-token logits keyed on the last input id, so every scored span's
logprob is analytically known.  No randomness, no network, no GPU.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from saklas.core.scoring import (
    ChoiceScore,
    ChoiceScores,
    score_choices,
    score_template,
)

from tests.conftest import CharTokenizer, FakeLogitsModel


# --------------------------------------------------------------------------- #
# A deterministic scoring stub.
#
# ``score_choices`` renders the messages through ``build_chat_input`` (the
# CharTokenizer has no chat_template, so the base-model fallback fires:
# "User: <content>\nAssistant:"), then for each choice tokenizes
# ``assistant_prefix + choice`` and scores the distinct completion tokens.
#
# We want the scored token's logprob to be *exactly knowable*, so the stub
# returns logits that depend only on the *previous* token id (a deterministic
# bigram).  For a token id ``t`` predicted from previous id ``p`` we set the
# logit of ``t`` to a value we control and every other vocab logit to 0, so
# ``log P(t | p) = logit[t] - logsumexp(logits)``.
# --------------------------------------------------------------------------- #

_VOCAB = 300


class _ScriptedTokenizer(CharTokenizer):
    """CharTokenizer + the surface ``score_choices`` reads.

    Two adjustments over the base ``CharTokenizer``:

    - ``pad_token_id`` / ``eos_token_id`` — ``score_choices`` reads
      ``pad_token_id`` (falls back to ``eos_token_id``) to right-pad the batched
      forward; the base only carries ``bos_token_id``.
    - ``__call__`` returns a *flat* ``input_ids`` list when ``return_tensors`` is
      ``None`` (matching the real HF tokenizer surface ``scoring._tok_ids``
      relies on: ``tokenizer(text, add_special_tokens=False)["input_ids"]`` is a
      flat ``list[int]``), and a ``(1, T)`` tensor only when ``return_tensors=
      "pt"`` (what ``build_chat_input``'s base-model fallback wants).  The base
      ``CharTokenizer`` always returns a batched tensor, which ``_tok_ids`` can't
      iterate to scalars.
    """

    pad_token_id = 0
    eos_token_id = 0

    def __call__(
        self,
        text: str,
        return_tensors: str | None = None,
        add_special_tokens: bool | None = None,
    ) -> dict[str, Any]:
        del add_special_tokens
        ids = self._ids(text)
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}
        return {"input_ids": ids}


def _const_logits_fn(value_for_id: dict[int, float]) -> Any:
    """A ``logits_fn`` giving each vocab id its mapped logit, else 0.

    Position-independent: every row gets the same vocab logit vector.  That
    makes ``log P(token=t)`` constant across positions and analytically
    ``value_for_id.get(t, 0.0) - logsumexp(base)`` — the value we assert.
    """
    base = torch.zeros(_VOCAB, dtype=torch.float32)
    for tid, val in value_for_id.items():
        base[tid] = val

    def fn(input_ids: torch.Tensor) -> torch.Tensor:
        n, t = input_ids.shape
        return base.view(1, 1, _VOCAB).expand(n, t, _VOCAB).clone()

    return fn


def _make_session(logits_fn: Any, *, steering_cm: Any = None) -> Any:
    """A duck-typed session exposing exactly what ``score_choices`` touches.

    The scorer reads ``session._model`` / ``session._tokenizer`` and, when
    ``steering`` is passed, calls ``session.steering(expr)`` for a context
    manager.  Everything else is irrelevant.
    """
    model = FakeLogitsModel(logits_fn)
    tok = _ScriptedTokenizer(mod=_VOCAB - 2)

    calls: list[Any] = []

    def steering(expr: Any) -> Any:
        calls.append(expr)
        if steering_cm is not None:
            return steering_cm
        import contextlib
        return contextlib.nullcontext()

    return SimpleNamespace(
        _model=model, _tokenizer=tok, steering=steering, _steer_calls=calls,
    )


def _logsumexp_base(value_for_id: dict[int, float]) -> float:
    base = torch.zeros(_VOCAB, dtype=torch.float32)
    for tid, val in value_for_id.items():
        base[tid] = val
    return float(torch.logsumexp(base, dim=-1).item())


# --------------------------------------------------------------------------- #
# sum / mean logprob math
# --------------------------------------------------------------------------- #


def test_single_token_sum_equals_logprob():
    """A one-token choice's ``sum_logprob`` is the analytic ``log P(token)``."""
    tok = _ScriptedTokenizer(mod=_VOCAB - 2)
    # "x" -> a single token id under the char tokenizer.
    x_ids = tok.encode("x")
    assert len(x_ids) == 1
    x = x_ids[0]
    value_for_id = {x: 5.0}
    norm = _logsumexp_base(value_for_id)

    session = _make_session(_const_logits_fn(value_for_id))
    scores = score_choices(session, [{"role": "user", "content": "pick"}], ["x"])

    assert isinstance(scores, ChoiceScores)
    (c,) = scores.choices
    assert c.n_tokens == 1
    assert c.sum_logprob == pytest.approx(5.0 - norm, abs=1e-4)
    # Single token -> mean == sum.
    assert c.mean_logprob == pytest.approx(c.sum_logprob, abs=1e-6)


def test_multi_token_sum_and_mean_views_diverge():
    """sum is the joint logprob; mean is length-normalized — both reported."""
    tok = _ScriptedTokenizer(mod=_VOCAB - 2)
    short_ids = tok.encode("a")          # 1 token
    long_ids = tok.encode("bcd")         # 3 tokens
    assert len(short_ids) == 1 and len(long_ids) == 3

    # Each token gets logit +3 over the flat background.
    value_for_id = {tid: 3.0 for tid in (*short_ids, *long_ids)}
    norm = _logsumexp_base(value_for_id)
    per_tok = 3.0 - norm                  # identical per-token logprob

    session = _make_session(_const_logits_fn(value_for_id))
    scores = score_choices(
        session, [{"role": "user", "content": "q"}], ["a", "bcd"],
    )
    by_text = {c.text: c for c in scores.choices}

    assert by_text["a"].n_tokens == 1
    assert by_text["bcd"].n_tokens == 3
    # sum scales with length; mean does not.
    assert by_text["a"].sum_logprob == pytest.approx(per_tok, abs=1e-4)
    assert by_text["bcd"].sum_logprob == pytest.approx(3 * per_tok, abs=1e-4)
    assert by_text["bcd"].mean_logprob == pytest.approx(per_tok, abs=1e-4)

    # The two softmax views disagree: under the length-biased joint logprob the
    # short candidate dominates; under the mean view they tie.
    assert by_text["a"].prob_sum > by_text["bcd"].prob_sum
    assert by_text["a"].prob_mean == pytest.approx(by_text["bcd"].prob_mean, abs=1e-4)


def test_restricted_softmax_matches_manual_softmax():
    """``prob_sum`` is exactly ``softmax`` over the set's ``sum_logprob``s."""
    tok = _ScriptedTokenizer(mod=_VOCAB - 2)
    ids = {t: tok.encode(t)[0] for t in ("p", "q", "r")}
    assert all(len(tok.encode(t)) == 1 for t in ("p", "q", "r"))
    # Distinct logits so the distribution is non-degenerate.
    value_for_id = {ids["p"]: 6.0, ids["q"]: 2.0, ids["r"]: 0.0}

    session = _make_session(_const_logits_fn(value_for_id))
    scores = score_choices(session, [{"role": "user", "content": "z"}], ["p", "q", "r"])

    sums = [c.sum_logprob for c in scores.choices]
    expect = torch.softmax(torch.tensor(sums), dim=0).tolist()
    got = [c.prob_sum for c in scores.choices]
    assert got == pytest.approx(expect, abs=1e-5)
    assert sum(got) == pytest.approx(1.0, abs=1e-5)
    # Ranking follows the highest-logit token.
    assert scores.ranked(by="sum")[0].text == "p"


# --------------------------------------------------------------------------- #
# degenerate-candidate handling + the prefix-merge span recovery
# --------------------------------------------------------------------------- #


def test_empty_choice_is_degenerate_and_excluded():
    """An empty choice carries sum 0.0 and ~0 restricted probability.

    Without the exclusion, sum_logprob 0.0 would be the *largest* value in the
    softmax (real candidates are strictly negative) and would dominate it.
    """
    tok = _ScriptedTokenizer(mod=_VOCAB - 2)
    real = tok.encode("w")[0]
    value_for_id = {real: 4.0}

    session = _make_session(_const_logits_fn(value_for_id))
    scores = score_choices(session, [{"role": "user", "content": "q"}], ["", "w"])
    by_text = {c.text: c for c in scores.choices}

    assert by_text[""].n_tokens == 0
    assert by_text[""].sum_logprob == 0.0
    assert by_text[""].token_ids == ()
    # Excluded from the restricted softmax: ~0 probability, all mass on "w".
    assert by_text[""].prob_sum == pytest.approx(0.0, abs=1e-6)
    assert by_text["w"].prob_sum == pytest.approx(1.0, abs=1e-6)


def test_all_degenerate_guard_returns_zeros():
    """Softmax over an all-degenerate set is the zero vector, not NaN."""
    session = _make_session(_const_logits_fn({}))
    scores = score_choices(session, [{"role": "user", "content": "q"}], ["", ""])
    for c in scores.choices:
        assert c.n_tokens == 0
        assert c.prob_sum == 0.0
        assert c.prob_mean == 0.0
        assert not math.isnan(c.prob_sum)


def test_shared_prefix_merge_recovers_completion_span():
    """The scored span is the choice's distinct tokens, not the whole sequence.

    ``assistant_prefix`` is conditioned on (not scored); only the choice's own
    tokens enter the logprob.  Two choices sharing the prefix differ only in
    their completion span, so their ``token_ids`` are exactly the choice tokens.
    """
    tok = _ScriptedTokenizer(mod=_VOCAB - 2)
    a_id = tok.encode("a")[0]
    b_id = tok.encode("b")[0]
    value_for_id = {a_id: 1.0, b_id: 1.0}

    session = _make_session(_const_logits_fn(value_for_id))
    scores = score_choices(
        session,
        [{"role": "user", "content": "q"}],
        ["a", "b"],
        assistant_prefix="prefix ",
    )
    by_text = {c.text: c for c in scores.choices}
    # The scored span is the single choice token, not the prefix tokens.
    assert by_text["a"].token_ids == (a_id,)
    assert by_text["b"].token_ids == (b_id,)
    assert by_text["a"].n_tokens == 1


def test_empty_choice_set_raises():
    session = _make_session(_const_logits_fn({}))
    with pytest.raises(ValueError, match="empty choice set"):
        score_choices(session, [{"role": "user", "content": "q"}], [])


def test_labels_length_mismatch_raises():
    session = _make_session(_const_logits_fn({}))
    with pytest.raises(ValueError, match="length mismatch"):
        score_choices(
            session, [{"role": "user", "content": "q"}], ["a", "b"],
            labels=["only-one"],
        )


def test_labels_default_to_choices_and_override():
    tok = _ScriptedTokenizer(mod=_VOCAB - 2)
    value_for_id = {tok.encode("a")[0]: 1.0, tok.encode("b")[0]: 1.0}
    session = _make_session(_const_logits_fn(value_for_id))

    default = score_choices(session, [{"role": "user", "content": "q"}], ["a", "b"])
    assert [c.label for c in default.choices] == ["a", "b"]

    relabelled = score_choices(
        session, [{"role": "user", "content": "q"}], ["a", "b"],
        labels=["first", "second"],
    )
    assert [c.label for c in relabelled.choices] == ["first", "second"]
    # The label rides into the dict view too.
    assert relabelled.choices[0].to_dict()["label"] == "first"


# --------------------------------------------------------------------------- #
# steering: the before/after read
# --------------------------------------------------------------------------- #


def test_steering_wraps_forward_and_labels_result():
    """``steering=`` enters ``session.steering`` and stamps the expression."""
    import contextlib

    entered: list[bool] = []

    @contextlib.contextmanager
    def cm() -> Any:
        entered.append(True)
        yield

    tok = _ScriptedTokenizer(mod=_VOCAB - 2)
    value_for_id = {tok.encode("a")[0]: 1.0}
    session = _make_session(_const_logits_fn(value_for_id), steering_cm=cm())

    scores = score_choices(
        session, [{"role": "user", "content": "q"}], ["a"],
        steering="0.5 calm",
    )
    assert session._steer_calls == ["0.5 calm"]
    assert entered == [True]
    assert scores.steering == "0.5 calm"
    assert scores.to_dict()["steering"] == "0.5 calm"


def test_no_steering_leaves_label_none():
    tok = _ScriptedTokenizer(mod=_VOCAB - 2)
    value_for_id = {tok.encode("a")[0]: 1.0}
    session = _make_session(_const_logits_fn(value_for_id))
    scores = score_choices(session, [{"role": "user", "content": "q"}], ["a"])
    assert scores.steering is None
    assert session._steer_calls == []


def test_steering_object_label_via_format():
    """A non-str steering carrying ``.format()`` labels via that callable."""
    tok = _ScriptedTokenizer(mod=_VOCAB - 2)
    value_for_id = {tok.encode("a")[0]: 1.0}
    import contextlib
    session = _make_session(
        _const_logits_fn(value_for_id), steering_cm=contextlib.nullcontext(),
    )

    class _Steer:
        def format(self) -> str:
            return "formatted-expr"

    scores = score_choices(
        session, [{"role": "user", "content": "q"}], ["a"], steering=_Steer(),
    )
    assert scores.steering == "formatted-expr"


def test_steering_actually_shifts_distribution():
    """Steering can change the restricted distribution — the headline read.

    A steering context manager that *swaps the model's logits_fn* (a stand-in
    for what real injection does to the forward) moves probability between the
    two candidates: the before/after distributions differ.
    """
    import contextlib

    tok = _ScriptedTokenizer(mod=_VOCAB - 2)
    a_id = tok.encode("a")[0]
    b_id = tok.encode("b")[0]

    # Unsteered: "a" favored.
    base = FakeLogitsModel(_const_logits_fn({a_id: 5.0, b_id: 0.0}))
    # Steered: swap in a logits_fn favoring "b".
    steered_fn = _const_logits_fn({a_id: 0.0, b_id: 5.0})

    @contextlib.contextmanager
    def cm() -> Any:
        prev = base._logits_fn
        base._logits_fn = steered_fn
        try:
            yield
        finally:
            base._logits_fn = prev

    def steering(expr: Any) -> Any:
        return cm()

    session: Any = SimpleNamespace(_model=base, _tokenizer=tok, steering=steering)

    before = score_choices(session, [{"role": "user", "content": "q"}], ["a", "b"])
    after = score_choices(
        session, [{"role": "user", "content": "q"}], ["a", "b"], steering="x",
    )
    before_by = {c.text: c.prob_sum for c in before.choices}
    after_by = {c.text: c.prob_sum for c in after.choices}

    assert before_by["a"] > before_by["b"]      # unsteered: a wins
    assert after_by["b"] > after_by["a"]        # steered:   b wins
    # The distribution genuinely moved (not just the argmax).
    assert after_by["a"] < before_by["a"]


# --------------------------------------------------------------------------- #
# score_template: one ChoiceScores per context
# --------------------------------------------------------------------------- #


def _toy_template() -> Any:
    """A 2-value, 2-context template via the real TemplateFolder.from_payload."""
    from saklas.io.templates import TemplateFolder

    return TemplateFolder.from_payload({
        "format_version": 1,
        "name": "weekday",
        "slot": "<DAY>",
        "values": ["Monday", "Friday"],
        "contexts": [
            {
                "turns": [{"role": "user", "content": "what comes after sunday"}],
                "assistant": "It is <DAY>.",
            },
            {
                "turns": [{"role": "user", "content": "name the work day"}],
                "assistant": "Today is <DAY>!",
            },
        ],
    })


def test_score_template_fans_out_per_context():
    tmpl = _toy_template()
    tok = _ScriptedTokenizer(mod=_VOCAB - 2)
    # Give every token in either value a small positive logit so the forward is
    # well-defined; exact values don't matter for the fan-out assertion.
    ids: set[int] = set()
    for v in ("Monday", "Friday"):
        ids.update(tok.encode(v))
    session = _make_session(_const_logits_fn({i: 1.0 for i in ids}))

    out = score_template(session, tmpl, steering="0.3 calm")
    assert isinstance(out, list)
    assert len(out) == 2                            # one per context
    for cs in out:
        assert isinstance(cs, ChoiceScores)
        assert [c.text for c in cs.choices] == ["Monday", "Friday"]
        assert [c.label for c in cs.choices] == ["monday", "friday"]
        assert cs.steering == "0.3 calm"            # steering applied to every context
        # Each context's restricted softmax is a proper distribution.
        assert sum(c.prob_sum for c in cs.choices) == pytest.approx(1.0, abs=1e-5)


# --------------------------------------------------------------------------- #
# dataclass surface
# --------------------------------------------------------------------------- #


def test_choicescore_to_dict_shape():
    cs = ChoiceScore(
        text="Mon", label="mon", token_ids=(1, 2), n_tokens=2,
        sum_logprob=-3.0, mean_logprob=-1.5, prob_sum=0.7, prob_mean=0.6,
    )
    d = cs.to_dict()
    assert d == {
        "text": "Mon", "label": "mon", "n_tokens": 2,
        "sum_logprob": -3.0, "mean_logprob": -1.5,
        "prob_sum": 0.7, "prob_mean": 0.6,
    }
    # token_ids is internal provenance, not in the serialized dict.
    assert "token_ids" not in d


def test_choicescores_ranked_by_sum_vs_mean():
    short = ChoiceScore("a", "a", (1,), 1, -1.0, -1.0, 0.6, 0.4)
    long = ChoiceScore("bbb", "bbb", (2, 3, 4), 3, -2.0, -0.66, 0.4, 0.6)
    cs = ChoiceScores(choices=(short, long))
    assert [c.text for c in cs.ranked(by="sum")] == ["a", "bbb"]   # higher sum
    assert [c.text for c in cs.ranked(by="mean")] == ["bbb", "a"]  # higher mean
    assert cs.to_dict()["choices"][0]["text"] == "a"


# --------------------------------------------------------------------------- #
# GPU integration (documents intent; skipped in CI)
# --------------------------------------------------------------------------- #


@pytest.mark.gpu
def test_real_forward_score_choices_distribution():
    """Real-model forward: a sensible next-day prediction shifts the argmax.

    Not run in CI (no GPU / no model download) — documents that ``score_choices``
    produces a coherent restricted-choice distribution on a real causal LM.
    """
    from saklas import SaklasSession

    with SaklasSession.from_pretrained("google/gemma-3-4b-it", device="auto") as session:
        scores = session.score_choices(
            [{"role": "user", "content": "The day after Monday is"}],
            ["Tuesday", "Saturday", "December"],
            assistant_prefix=" ",
        )
        assert sum(c.prob_sum for c in scores.choices) == pytest.approx(1.0, abs=1e-4)
        # Tuesday should be the model's top restricted choice.
        assert scores.ranked(by="sum")[0].text == "Tuesday"
