"""Restricted-choice completion scoring — the logit read of a template.

Given a conversation context and a fixed set of candidate completions, score the
**conditional probability of each candidate** and report the distribution over the
set. This is forced-choice / multiple-choice scoring: the model's belief about
which slot-fill comes next, not a sampled token.

It is the logit counterpart to the manifold fit. The manifold pools each
candidate's slot-filled *activation* into a node; the scorer reads each
candidate's slot-filled *logprob*. Both run off the same
:class:`saklas.io.templates.TemplateFolder`.

Scoring is against the **raw** model distribution (temperature 1, no top-k/p
truncation), so the probabilities are the model's beliefs, not a reshaping by
sampler knobs. Multi-token candidates are handled two ways and both are reported:
``sum_logprob`` (the true joint ``log P(candidate | context)``) and
``mean_logprob`` (length-normalized), each with its own restricted-choice softmax
over the candidate set — length bias is real when candidates differ in token
count, so neither view is silently chosen for you.

An optional ``steering=`` runs the scoring forward under a steering expression, so
the project's core question — *did steering shift the distribution, not just the
argmax?* — answers directly: ``P("one second")`` before vs after a steer.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from saklas.core.generation import build_chat_input
from saklas.core.joint_logprobs import _call_model, _shared_prefix_len

if TYPE_CHECKING:  # avoid an import cycle at module load
    from saklas.core.session import SaklasSession
    from saklas.io.templates import TemplateFolder

# Choices are scored in chunks so a large value set (e.g. 107 personas) does not
# materialize one ``[n_choices, max_len, vocab]`` logits tensor — vocab is ~256k
# on Gemma, so an unbounded batch would blow memory.  Each chunk is one forward.
_SCORE_BATCH = 16


@dataclass(frozen=True)
class ChoiceScore:
    """One candidate's score within a :class:`ChoiceScores` set."""

    text: str
    label: str
    token_ids: tuple[int, ...]
    n_tokens: int
    sum_logprob: float
    mean_logprob: float
    prob_sum: float       # restricted-choice softmax over the set's sum_logprobs
    prob_mean: float      # restricted-choice softmax over the set's mean_logprobs

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "label": self.label,
            "n_tokens": self.n_tokens,
            "sum_logprob": self.sum_logprob,
            "mean_logprob": self.mean_logprob,
            "prob_sum": self.prob_sum,
            "prob_mean": self.prob_mean,
        }


@dataclass(frozen=True)
class ChoiceScores:
    """A restricted-choice distribution over one context's candidate set."""

    choices: tuple[ChoiceScore, ...]
    steering: str | None = None

    def ranked(self, by: str = "sum") -> list[ChoiceScore]:
        """Choices sorted most- to least-probable.

        ``by`` selects the statistic: ``"sum"`` (joint logprob, length-biased
        toward short completions) or ``"mean"`` (length-normalized).
        """
        key = (lambda c: c.sum_logprob) if by == "sum" else (lambda c: c.mean_logprob)
        return sorted(self.choices, key=key, reverse=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "steering": self.steering,
            "choices": [c.to_dict() for c in self.choices],
        }


def _tok_ids(tokenizer: Any, text: str) -> list[int]:
    if not text:
        return []
    enc = tokenizer(text, add_special_tokens=False)["input_ids"]
    return [int(t) for t in enc]


def score_choices(
    session: "SaklasSession",
    messages: list[dict[str, str]],
    choices: list[str],
    *,
    assistant_prefix: str = "",
    labels: list[str] | None = None,
    steering: "str | Any | None" = None,
    system_prompt: str | None = None,
) -> ChoiceScores:
    """Score each candidate completion and return the restricted-choice set.

    ``messages`` is the conversation history (ending on a user turn);
    ``assistant_prefix`` is any assistant text *before* the slot (the model
    conditions on it); ``choices`` are the candidate slot-fills. Each candidate's
    completion span is recovered with :func:`_shared_prefix_len`, which absorbs
    the boundary-token merge (``"Monday"`` vs ``"Tuesday"`` may retokenize the
    last prefix token). One batched teacher-forced forward per chunk, no sampling.
    """
    if not choices:
        raise ValueError("score_choices: empty choice set")
    model = session._model
    tokenizer = session._tokenizer
    device = next(model.parameters()).device
    labels = labels if labels is not None else list(choices)
    if len(labels) != len(choices):
        raise ValueError("score_choices: labels and choices length mismatch")

    render = build_chat_input(
        tokenizer, messages, system_prompt=system_prompt, add_generation_prompt=True,
    )
    render_ids = [int(t) for t in render[0].tolist()]
    prefix_ids = render_ids + _tok_ids(tokenizer, assistant_prefix)

    # Per-choice full sequence + the start index of its scored completion span.
    seqs: list[list[int]] = []
    cuts: list[int] = []
    for c in choices:
        full = render_ids + _tok_ids(tokenizer, assistant_prefix + c)
        cut = _shared_prefix_len(prefix_ids, full)
        if cut >= len(full):           # candidate adds no distinct token
            cut = len(full)
        seqs.append(full)
        cuts.append(cut)

    _pad: Any = tokenizer.pad_token_id
    if _pad is None:
        _pad = tokenizer.eos_token_id
    if isinstance(_pad, (list, tuple)):
        _pad = _pad[0] if _pad else 0
    pad_id = int(_pad) if _pad is not None else 0

    steering_cm: Any = (
        session.steering(steering) if steering is not None else contextlib.nullcontext()
    )

    sum_lps: list[float] = []
    n_toks: list[int] = []
    span_ids: list[tuple[int, ...]] = []

    with steering_cm, torch.inference_mode():
        for start in range(0, len(seqs), _SCORE_BATCH):
            chunk = list(range(start, min(start + _SCORE_BATCH, len(seqs))))
            max_len = max(len(seqs[i]) for i in chunk)
            n = len(chunk)
            input_ids = torch.full((n, max_len), pad_id, dtype=torch.long, device=device)
            attn = torch.zeros((n, max_len), dtype=torch.long, device=device)
            for r, i in enumerate(chunk):
                s = seqs[i]
                input_ids[r, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)
                attn[r, : len(s)] = 1
            out = _call_model(
                model, input_ids=input_ids, attention_mask=attn, use_cache=False,
            )
            # fp32 vocab reduction — the project-wide norm/softmax invariant.
            logits = out.logits.float()                  # [n, max_len, V]
            norm = torch.logsumexp(logits, dim=-1)        # [n, max_len]
            for r, i in enumerate(chunk):
                s = seqs[i]
                cut, L = cuts[i], len(seqs[i])
                if cut >= L:                              # empty completion
                    sum_lps.append(0.0)
                    n_toks.append(0)
                    span_ids.append(())
                    continue
                tok = torch.tensor(s[cut:L], dtype=torch.long, device=device)  # [m]
                pos = torch.arange(cut - 1, L - 1, device=device)              # [m]
                chosen = logits[r, pos, :].gather(1, tok[:, None]).squeeze(1)  # [m]
                lp = chosen - norm[r, pos]
                sum_lps.append(float(lp.sum().item()))
                n_toks.append(int(tok.numel()))
                span_ids.append(tuple(int(t) for t in s[cut:L]))
            del out, logits, norm

    mean_lps = [sl / nt if nt else sl for sl, nt in zip(sum_lps, n_toks)]
    # Degenerate candidates (no distinct completion token — an empty-string
    # choice, or one fully absorbed by the prefix) carry ``sum_logprob`` 0.0,
    # which is the *largest* value in the restricted-choice softmax (every real
    # candidate's joint logprob is strictly negative) and would wrongly dominate
    # it.  Exclude them: -inf in the softmax input gives them ~0 probability,
    # with an all-degenerate guard (softmax over all -inf is NaN).
    degenerate = [nt == 0 for nt in n_toks]

    def _restricted_softmax(values: list[float]) -> list[float]:
        masked = [
            float("-inf") if deg else v
            for v, deg in zip(values, degenerate)
        ]
        if all(m == float("-inf") for m in masked):
            return [0.0] * len(masked)
        return [float(p) for p in torch.softmax(torch.tensor(masked), dim=0).tolist()]

    prob_sum = _restricted_softmax(sum_lps)
    prob_mean = _restricted_softmax(mean_lps)

    scored = tuple(
        ChoiceScore(
            text=choices[i],
            label=labels[i],
            token_ids=span_ids[i],
            n_tokens=n_toks[i],
            sum_logprob=sum_lps[i],
            mean_logprob=mean_lps[i],
            prob_sum=float(prob_sum[i]),
            prob_mean=float(prob_mean[i]),
        )
        for i in range(len(choices))
    )
    steer_str = _steering_label(steering)
    return ChoiceScores(choices=scored, steering=steer_str)


def score_template(
    session: "SaklasSession",
    template: "TemplateFolder",
    *,
    steering: "str | Any | None" = None,
    system_prompt: str | None = None,
) -> list[ChoiceScores]:
    """Score a template's value set against each of its contexts.

    Returns one :class:`ChoiceScores` per context (the per-context shape the
    template artifact is designed around). The same ``steering=`` reshaping is
    applied to every context.
    """
    out: list[ChoiceScores] = []
    for spec in template.score_inputs():
        out.append(score_choices(
            session,
            spec["messages"],
            spec["choices"],
            assistant_prefix=spec["assistant_prefix"],
            labels=spec["labels"],
            steering=steering,
            system_prompt=system_prompt,
        ))
    return out


def _steering_label(steering: "str | Any | None") -> str | None:
    if steering is None:
        return None
    if isinstance(steering, str):
        return steering
    fmt = getattr(steering, "format", None)
    if callable(fmt):
        try:
            return str(fmt())
        except Exception:
            return None
    return str(steering)
