"""Tests for the cross-branch joint-logprob computation
(``saklas.core.joint_logprobs``).

Covers the pure-math helpers in isolation — building a real model would
load weights and a tokenizer, but the inner alignment + lookup is a
plain tensor + index walk that we can exercise with hand-built
log-softmax tensors.  The IO wrapper (``compute_joint_logprobs``) is
checked via a thin in-memory ``MockSession`` shim so we cover the full
forward-pass + cache + reorient path without HF weights.
"""

from __future__ import annotations

import torch

from saklas.core.joint_logprobs import (
    JointLogprobRow,
    JointLogprobs,
    _approx_kl_topk,
    _cache_key,
    _compute_rows,
    _shared_prefix_len,
    reorient_for_request,
)


# ---------------------------------------------------------------------------
# Shared-prefix walker
# ---------------------------------------------------------------------------


def test_shared_prefix_len_identical():
    assert _shared_prefix_len([1, 2, 3, 4], [1, 2, 3, 4]) == 4


def test_shared_prefix_len_divergent():
    assert _shared_prefix_len([1, 2, 3, 4], [1, 2, 99, 4]) == 2


def test_shared_prefix_len_empty():
    assert _shared_prefix_len([], [1, 2]) == 0
    assert _shared_prefix_len([1, 2], []) == 0


def test_shared_prefix_len_one_is_prefix_of_other():
    assert _shared_prefix_len([1, 2, 3], [1, 2, 3, 4]) == 3


# ---------------------------------------------------------------------------
# Cache key symmetry + reorient
# ---------------------------------------------------------------------------


def test_cache_key_is_symmetric():
    assert _cache_key("a", "b") == _cache_key("b", "a") == ("a", "b")


def test_reorient_for_request_noop():
    res = JointLogprobs(
        a_id="a", b_id="b", parent_id="p",
        rows=(JointLogprobRow(
            a_index=0, b_index=0, a_text="x", b_text="x", aligned=True,
            lp_a_in_a=-0.5, lp_b_in_b=-0.7,
            lp_a_in_b=-1.0, lp_b_in_a=-1.2,
            rank_changed=False, approx_kl=0.1,
        ),),
        n_rank1_changed=0,
    )
    out = reorient_for_request(res, "a", "b")
    assert out is res  # no copy when orientation matches


def test_reorient_for_request_flips_labels_and_columns():
    res = JointLogprobs(
        a_id="a", b_id="b", parent_id="p",
        rows=(JointLogprobRow(
            a_index=3, b_index=4, a_text="x", b_text="y", aligned=True,
            lp_a_in_a=-0.5, lp_b_in_b=-0.7,
            lp_a_in_b=-1.0, lp_b_in_a=-1.2,
            rank_changed=True, approx_kl=0.1,
        ),),
        n_rank1_changed=1,
    )
    out = reorient_for_request(res, "b", "a")
    assert out.a_id == "b" and out.b_id == "a"
    assert out.parent_id == "p"
    assert out.n_rank1_changed == 1
    row = out.rows[0]
    # Indices and texts swap; chosen-prob columns swap; cross columns swap.
    assert row.a_index == 4 and row.b_index == 3
    assert row.a_text == "y" and row.b_text == "x"
    assert row.lp_a_in_a == -0.7 and row.lp_b_in_b == -0.5
    assert row.lp_a_in_b == -1.2 and row.lp_b_in_a == -1.0
    assert row.rank_changed is True
    assert row.approx_kl == 0.1


# ---------------------------------------------------------------------------
# Approx-KL math
# ---------------------------------------------------------------------------


def _logits_to_logsoftmax(rows: list[list[float]]) -> torch.Tensor:
    """Build a ``[T, V]`` log-softmax tensor from raw logits rows."""
    return torch.log_softmax(torch.tensor(rows, dtype=torch.float32), dim=-1)


def test_approx_kl_identical_distributions_is_zero():
    lp = _logits_to_logsoftmax([[0.0, 1.0, 2.0, 0.5]])[0]
    assert abs(_approx_kl_topk(lp, lp, top_k=4)) < 1e-6


def test_approx_kl_topk_grows_with_divergence():
    # Two distributions with the same support: B is a sharper version
    # of A's argmax.  KL(A || B) is strictly positive and grows when we
    # widen the cross-entropy.
    lp_a = _logits_to_logsoftmax([[0.0, 1.0, 0.5, -0.5]])[0]
    lp_b = _logits_to_logsoftmax([[-3.0, 5.0, -1.0, -3.0]])[0]
    kl = _approx_kl_topk(lp_a, lp_b, top_k=4)
    assert kl > 0
    # KL should be larger when B is even sharper.
    lp_b2 = _logits_to_logsoftmax([[-9.0, 9.0, -3.0, -9.0]])[0]
    kl2 = _approx_kl_topk(lp_a, lp_b2, top_k=4)
    assert kl2 > kl


def test_approx_kl_topk_respects_truncation():
    # With top_k=1 we only see the argmax token; coarser estimate.
    lp_a = _logits_to_logsoftmax([[0.0, 1.0, 0.5, -0.5]])[0]
    lp_b = _logits_to_logsoftmax([[-3.0, 5.0, -1.0, -3.0]])[0]
    kl_full = _approx_kl_topk(lp_a, lp_b, top_k=4)
    kl_top1 = _approx_kl_topk(lp_a, lp_b, top_k=1)
    # Full top-K should be >= top-1 (more mass observed).  Equality
    # is allowed (both argmax-dominated distributions) but inequality
    # is the typical shape.
    assert kl_full >= kl_top1 - 1e-6


# ---------------------------------------------------------------------------
# Per-row computation
# ---------------------------------------------------------------------------


def _build_logp(rows: list[list[float]]) -> torch.Tensor:
    return _logits_to_logsoftmax(rows)


def test_compute_rows_self_evaluation_only_on_divergent_tail():
    # Setup: vocab = 4 tokens.  Prefix is 2 tokens long.
    # Branch A's assistant tokens: [1, 2].
    # Branch B's assistant tokens: [1, 3] — diverges at position 1.
    vocab = 4
    prefix_len = 2

    # logp_a: T=4 rows.  Position 1 (predicts assistant[0]==1) puts
    # mass on token 1.  Position 2 (predicts assistant[1]==2) puts
    # mass on token 2.
    logp_a = _build_logp([
        [0.0] * vocab,                # position 0 (unused as predictor here)
        [-3.0, 0.0, -3.0, -3.0],      # position 1 — A's first assistant token (id=1)
        [-3.0, -3.0, 0.0, -3.0],      # position 2 — A's second assistant token (id=2)
        [-3.0, -3.0, -3.0, -3.0],     # tail
    ])
    # logp_b: B's second assistant token is 3, so position 2 puts mass
    # on token 3 instead.  Position 1 matches A (still on token 1) so
    # the aligned row should show lp_a_in_b == lp_a_in_a.
    logp_b = _build_logp([
        [0.0] * vocab,
        [-3.0, 0.0, -3.0, -3.0],
        [-3.0, -3.0, -3.0, 0.0],
        [-3.0, -3.0, -3.0, -3.0],
    ])
    ids_a = [10, 20, 1, 2]   # prefix tokens are dummy; assistant tail is [1, 2]
    ids_b = [10, 20, 1, 3]
    strs_a = ["P0", "P1", "tok_one", "tok_two"]
    strs_b = ["P0", "P1", "tok_one", "tok_three"]

    rows = _compute_rows(
        logp_a, logp_b, ids_a, ids_b, strs_a, strs_b, prefix_len,
    )
    # Two assistant positions on each side → two rows for the common
    # prefix.  Position 0 is byte-aligned ("tok_one" matches), position
    # 1 is not ("tok_two" vs "tok_three").
    assert len(rows) >= 2
    aligned0 = rows[0]
    assert aligned0.aligned is True
    assert aligned0.a_index == 0 and aligned0.b_index == 0
    # Self-eval: A's chosen token (id=1) at position prefix_len+0 is
    # predicted by logits at position prefix_len+0-1 == 1.  logp_a[1][1]
    # ≈ 0 after softmax (heaviest mass on token 1).
    assert aligned0.lp_a_in_a is not None
    assert aligned0.lp_a_in_a > -0.2  # near zero
    # Cross: B's distribution at the same position has the same mass on
    # token 1, so lp_a_in_b ≈ lp_a_in_a.
    assert aligned0.lp_a_in_b is not None
    assert abs(aligned0.lp_a_in_b - aligned0.lp_a_in_a) < 1e-5

    # Position 1: not byte-aligned (token strings differ), so cross-
    # evaluation should be None and rank_changed is False (it's only
    # set on aligned rows).
    misaligned = rows[1]
    assert misaligned.aligned is False
    assert misaligned.lp_a_in_b is None
    assert misaligned.lp_b_in_a is None
    assert misaligned.rank_changed is False


def test_compute_rows_rank_changed_flag():
    # Build two distributions that disagree on the argmax at the
    # aligned position.  ``pa_pos = prefix_len + a_idx - 1`` so with
    # prefix_len=2 and a_idx=0 we read logp[1] — the row we shaped.
    # Token id 0 wins under A, token id 1 wins under B.  Both branches
    # actually emit id=0 at position 0 so the strings byte-align.
    vocab = 3
    prefix_len = 2
    logp_a = _build_logp([
        [0.0] * vocab,          # position 0 (unused as predictor)
        [0.0, -5.0, -5.0],      # position 1 → predicts assistant[0]; argmax=0
        [0.0] * vocab,          # tail
    ])
    logp_b = _build_logp([
        [0.0] * vocab,
        [-5.0, 0.0, -5.0],      # position 1 → predicts assistant[0]; argmax=1
        [0.0] * vocab,
    ])
    ids_a = [99, 100, 0]
    ids_b = [99, 100, 0]
    strs_a = ["P0", "P1", "same"]
    strs_b = ["P0", "P1", "same"]
    rows = _compute_rows(logp_a, logp_b, ids_a, ids_b, strs_a, strs_b, prefix_len)
    assert len(rows) == 1
    r = rows[0]
    assert r.aligned is True
    assert r.rank_changed is True
    # approx_kl should be strictly positive when the argmaxes diverge.
    assert r.approx_kl is not None and r.approx_kl > 0


def test_compute_rows_empty_when_no_assistant_tokens():
    logp = _build_logp([[0.0, 0.0]])
    rows = _compute_rows(
        logp, logp, [0], [0], ["P"], ["P"], prefix_len=1,
    )
    assert rows == []


# ---------------------------------------------------------------------------
# IO wrapper — verified against an in-memory mock session.
# ---------------------------------------------------------------------------


class _MockTokenizer:
    """Whitespace-split tokenizer with stable ids.

    Just enough surface area to feed ``build_chat_input``'s base-model
    fallback branch (no ``chat_template`` attribute) plus
    ``compute_joint_logprobs``'s ``decode([id])`` calls.
    """

    chat_template = None  # forces the base-model branch in build_chat_input

    def __init__(self):
        self._vocab: dict[str, int] = {}
        self._rev: dict[int, str] = {}

    def _intern(self, tok: str) -> int:
        if tok not in self._vocab:
            tid = len(self._vocab)
            self._vocab[tok] = tid
            self._rev[tid] = tok
        return self._vocab[tok]

    def __call__(self, text: str, return_tensors: str = "pt"):
        # Whitespace split — good enough for the assertion shape.  The
        # base-model branch of ``build_chat_input`` calls this; the
        # tensor shape it expects is ``{"input_ids": Tensor[1, T]}``.
        toks = text.split()
        ids = [self._intern(t) for t in toks]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}

    def decode(self, ids):
        if isinstance(ids, list):
            return " ".join(self._rev.get(int(i), "<unk>") for i in ids)
        return self._rev.get(int(ids), "<unk>")


class _MockModel:
    """Returns a constant log-uniform distribution at every position.

    All chosen-token logprobs equal ``-log(vocab)``; cross-evaluation
    matches self-evaluation exactly (KL is zero everywhere).  Cheap
    parameter list satisfies ``next(model.parameters()).device``.
    """

    def __init__(self, vocab: int):
        self._vocab = vocab
        # One throwaway parameter so ``next(model.parameters()).device``
        # works — the actual values are irrelevant.
        self._param = torch.zeros(1, requires_grad=False)

    def parameters(self):
        yield self._param

    def __call__(self, *, input_ids, use_cache=False):
        del use_cache  # ignored — mock is stateless
        T = input_ids.shape[-1]
        # Logits flat across vocab → log_softmax = -log(vocab).
        logits = torch.zeros((1, T, self._vocab), dtype=torch.float32)
        return type("Out", (), {"logits": logits})()


class _MockTree:
    """Minimal LoomTree surface the joint-logprob path consumes."""

    def __init__(self):
        from saklas.core.loom import LoomNode

        # parent (user node) and two assistant children with different
        # assistant text.  The base-model fallback in build_chat_input
        # serializes as ``"Role: content\nRole: content\nAssistant:"``.
        root = LoomNode(id="root", parent_id=None, role="system", text="")
        user = LoomNode(id="u1", parent_id="root", role="user", text="ask")
        a1 = LoomNode(id="a1", parent_id="u1", role="assistant", text="hello a")
        a2 = LoomNode(id="a2", parent_id="u1", role="assistant", text="hello b")
        self.nodes = {"root": root, "u1": user, "a1": a1, "a2": a2}

    def messages_for(self, leaf_id: str, *, include_system: bool = False):
        from saklas.core.loom import LoomNode

        # Walk parent chain; return [{role, content}], skip synthetic
        # root unless include_system is true.
        path: list[LoomNode] = []
        cur = leaf_id
        while cur is not None:
            n = self.nodes[cur]
            path.append(n)
            cur = n.parent_id
        path.reverse()
        out: list[dict[str, str]] = []
        for n in path:
            if n.id == "root" and not include_system:
                continue
            out.append({"role": n.role, "content": n.text})
        return out


class _MockConfig:
    system_prompt: str | None = None


class _MockSession:
    """Just enough surface for ``compute_joint_logprobs`` to run."""

    def __init__(self):
        self.tokenizer = _MockTokenizer()
        # Build a vocabulary that covers the strings we'll feed.  The
        # decode path needs every id to round-trip.
        for word in ("User:", "ask", "Assistant:", "hello", "a", "b"):
            self.tokenizer._intern(word)
        self._model = _MockModel(vocab=len(self.tokenizer._vocab) + 16)
        self.tree = _MockTree()
        self.config = _MockConfig()


def test_compute_joint_logprobs_runs_end_to_end_on_mock():
    from saklas.core.joint_logprobs import compute_joint_logprobs

    session = _MockSession()
    result = compute_joint_logprobs(session, "a1", "a2")
    assert result.a_id == "a1"
    assert result.b_id == "a2"
    assert result.parent_id == "u1"
    # Both branches: prefix is identical up to "hello ", then diverges
    # ("a" vs "b").  At least one row should be aligned ("hello"),
    # and there should be at least one row where the byte alignment
    # fails ("a" vs "b").
    assert any(r.aligned for r in result.rows)
    # Under the log-uniform mock, every chosen logprob equals
    # -log(vocab).  Aligned rows therefore have lp_a_in_a == lp_a_in_b.
    aligned_rows = [r for r in result.rows if r.aligned]
    assert aligned_rows  # at least the "hello" position aligns
    for r in aligned_rows:
        assert r.lp_a_in_a is not None and r.lp_a_in_b is not None
        assert abs(r.lp_a_in_a - r.lp_a_in_b) < 1e-5
        # KL between identical distributions is ~0.
        assert r.approx_kl is not None and abs(r.approx_kl) < 1e-5
        # Argmax is the same on both sides (any token is tied) — the
        # ``argmax`` resolves deterministically to id 0, so rank_changed
        # is False.
        assert r.rank_changed is False
    # n_rank1_changed mirrors the aligned-row count where rank flipped.
    assert result.n_rank1_changed == 0


def test_compute_joint_logprobs_to_dict_round_trip():
    from saklas.core.joint_logprobs import compute_joint_logprobs

    session = _MockSession()
    result = compute_joint_logprobs(session, "a1", "a2")
    payload = result.to_dict()
    assert payload["a_id"] == "a1" and payload["b_id"] == "a2"
    assert payload["parent_id"] == "u1"
    assert payload["n_rank1_changed"] == result.n_rank1_changed
    assert len(payload["rows"]) == len(result.rows)
    # Every row has every expected key.
    expected_keys = {
        "a_index", "b_index", "a_text", "b_text", "aligned",
        "lp_a_in_a", "lp_b_in_b", "lp_a_in_b", "lp_b_in_a",
        "rank_changed", "approx_kl",
    }
    for row in payload["rows"]:
        assert set(row.keys()) == expected_keys
