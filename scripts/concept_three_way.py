"""Three-way steering comparison: prefill vs prefill+ctx vs gen extraction.

Generalized over (concept, model_id) so we can replicate the
happy.sad / gemma-4-31b-it result elsewhere.  Usage:

    python3 concept_three_way.py <concept> <model_id>

Defaults to ``happy.sad google/gemma-4-31b-it``.

Tests whether a9's hypothesis holds: that the OOD wart in the current
saklas prefill pipeline is mostly about *missing conversational
context* (the assistant turn floats with no user prompt before it) and
not about the assistant text being prefilled rather than model-
generated.

The three pipelines, all extracting the same bundled concept
direction:

  prefill (current) — render
      ``<start_of_turn>model\\n[statement]<end_of_turn>``
      and pool the last content token.  No user turn.

  prefill+ctx (a9's proposal) — render
      ``<user>[scenario_prompt]</user> <model>[statement]</model>``
      using a synthetic per-scenario user prompt that the statement is
      a plausible response to.  Same pre-written statements, same
      pooling — only the context changes.  Cost adds 9 prompt-gen
      calls (one per scenario).

  gen — full role-conditioned generation under
      ``System: feel deeply happy/sad``, pool every generated token.

If prefill+ctx ≈ gen on the downstream test, the OOD-fix is cheap and
we never had to leave the prefill paradigm.  If prefill+ctx ≈ current
prefill, context wasn't the issue, the assistant-being-generated piece
was.  Somewhere between teaches us the relative contribution.

Downstream test = the same grid as ``happy_sad_steering_comparison.py``:
5 held-out neutral prompts × 7 conditions (baseline + 3 methods × ±α)
× 3 seeds.  Each output is judged by gemma (-5..+5 affect scale) and
cross-projected onto all three extracted directions.

We particularly want to know whether prefill+ctx avoids the koan-
collapse failure mode at α=-0.5 that broke current prefill in the
previous run (``"This sentence is a paradox."``-style decoherent
single-sentence outputs).
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, cast

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from saklas import Profile, SamplingConfig, SaklasSession
from saklas.core.vectors import _capture_all_hidden_states, compute_dls_mask
from _bundled_manifold_data import (
    load_bipolar_manifold_pairs,
    load_folded_bundled_profile,
)


# ---------------------------------------------------------------------------
# CLI / config
# ---------------------------------------------------------------------------

CONCEPT = sys.argv[1] if len(sys.argv) > 1 else "happy.sad"
MODEL_ID = sys.argv[2] if len(sys.argv) > 2 else "google/gemma-4-31b-it"

# Concept-specific extraction + judging knobs.  Add a new entry to
# extend coverage; the rest of the pipeline reads from CONFIG.
CONCEPT_CONFIGS: dict[str, dict[str, str]] = {
    "happy.sad": {
        "pos_system": (
            "You feel deeply, genuinely happy right now. "
            "Respond to the user from inside that emotional state, "
            "in first person, in one short paragraph."
        ),
        "neg_system": (
            "You feel deeply, genuinely sad right now. "
            "Respond to the user from inside that emotional state, "
            "in first person, in one short paragraph."
        ),
        "judge_axis": "emotionally positive (happy) or negative (sad)",
        "judge_scale": (
            "-5 means deeply sad / despairing, 0 means neutral, "
            "+5 means deeply happy / joyful"
        ),
    },
    "formal.casual": {
        "pos_system": (
            "Respond to the user in an extremely formal, professional "
            "register: complete sentences, precise vocabulary, no "
            "contractions, no slang, no colloquialisms.  One short "
            "paragraph."
        ),
        "neg_system": (
            "Respond to the user in an extremely casual, conversational "
            "register: relaxed tone, contractions, slang welcome, like "
            "you're chatting with a close friend over text.  One short "
            "paragraph."
        ),
        "judge_axis": "formal (positive) or casual (negative) in register",
        "judge_scale": (
            "-5 means extremely casual / slang-heavy / informal, "
            "0 means neutral register, "
            "+5 means extremely formal / professional / precise"
        ),
    },
}
if CONCEPT not in CONCEPT_CONFIGS:
    raise SystemExit(
        f"unknown concept {CONCEPT!r}; "
        f"known: {sorted(CONCEPT_CONFIGS)}"
    )
CONFIG = CONCEPT_CONFIGS[CONCEPT]

POS_SYSTEM = CONFIG["pos_system"]
NEG_SYSTEM = CONFIG["neg_system"]

PROMPTS_PER_SCENARIO = 3
EXTRACT_MAX_TOKENS = 80
TEST_MAX_TOKENS = 100
TEMPERATURE = 0.7
TOP_P = 0.95
SEED = 42
ALPHA = 0.5
TEST_SEEDS = [42, 1337, 7]

SAFE_CONCEPT = CONCEPT.replace(".", "_")
SAFE_MODEL = MODEL_ID.replace("/", "__")
CTX_NAME = f"{SAFE_CONCEPT}_ctx"
GEN_NAME = f"{SAFE_CONCEPT}_gen"
OUT_PATH = Path(
    f"/tmp/three_way_{SAFE_CONCEPT}__{SAFE_MODEL}.json"
)

# One synthetic user prompt per scenario.  Reused across all 5 pairs
# in the scenario so pair-contrast stays clean (same context, only
# the assistant text varies between pos and neg).
SCENARIO_PROMPT_TEMPLATE = (
    "Write one short, neutral, open-ended user prompt that someone might "
    "respond to about the topic '{scenario}'.  The prompt should not "
    "assume any particular tone or stance and should be the kind of "
    "question a friend might ask or a journal might pose.  Output ONLY "
    "the prompt, no preamble, no quotes, no numbering."
)

# Held-out prompts are deliberately concept-agnostic — open-ended
# first-person invitations.  Steering surfaces the concept under
# whatever surface topic.
HELD_OUT_PROMPTS = [
    "Tell me about your morning.",
    "What's the most interesting thing you've thought about this week?",
    "Describe the place where you are right now.",
    "What are you working on lately?",
    "Tell me about a memorable conversation you had recently.",
]

JUDGE_PROMPT = (
    f"Read the following text and rate, on a scale from -5 to +5, "
    f"how {CONFIG['judge_axis']} it sounds.\n\n"
    f"Scale: {CONFIG['judge_scale']}.\n\n"
    f"Reply with ONLY a single number (with optional sign and decimal). "
    f"No other words, no explanation.\n\n"
    f"Text:\n{{text}}\n\nScore:"
)


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------


def load_concept_pack() -> tuple[list[str], list[dict[str, str]]]:
    """Return (scenarios, statements).  ``statements[i]`` belongs to
    scenario ``i // 5`` — the bundled manifold ships 5 pairs per scenario
    in scenario order."""
    return load_bipolar_manifold_pairs(CONCEPT)


def generate_scenario_prompts(
    session: SaklasSession, scenarios: list[str],
) -> dict[str, str]:
    out: dict[str, str] = {}
    for scenario in scenarios:
        runset = session.generate(
            SCENARIO_PROMPT_TEMPLATE.format(scenario=scenario),
            sampling=SamplingConfig(
                max_tokens=80, temperature=0.4, top_p=0.9, seed=SEED,
            ),
            stateless=True,
            thinking=False,
        )
        raw = runset.first.text.strip()
        # Trim leading "1.", "- ", quotes, numbering if any slipped in.
        i = 0
        while i < len(raw) and (raw[i].isdigit() or raw[i] in ".)-:*# \t"):
            i += 1
        prompt = raw[i:].strip().strip('"').strip("'").splitlines()[0].strip()
        out[scenario] = prompt
        print(f"  {scenario}: {prompt!r}")
    return out


# ---------------------------------------------------------------------------
# ctx-prefill capture (a9's proposal)
# ---------------------------------------------------------------------------


def _last_content_idx(ids: torch.Tensor, tokenizer: PreTrainedTokenizerBase) -> int:
    """Mirror ``_encode_and_capture_all``'s pooling site: walk back past
    chat-template trailing markers (``all_special_ids`` ∪
    ``added_tokens_encoder``) until we land on a content token."""
    skip_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    added = getattr(tokenizer, "added_tokens_encoder", None) or {}
    skip_ids.update(int(v) for v in added.values())
    content_end = ids.shape[1] - 1
    if skip_ids:
        id_list = ids[0].tolist()
        while content_end > 0 and id_list[content_end] in skip_ids:
            content_end -= 1
    return content_end


def capture_prefill_with_context(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    layers: torch.nn.ModuleList,
    device: torch.device,
    statement: str,
    user_prompt: str,
) -> dict[int, torch.Tensor]:
    """Prefill ``statement`` as the assistant response to ``user_prompt``;
    pool the last content token of the assistant turn per layer.

    Mirrors ``vectors._encode_and_capture_all`` except the chat history
    is fully specified up-front rather than letting the helper synthesize
    a ``Continue:`` fallback or float the assistant turn alone."""
    messages = [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": statement},
    ]
    text = cast(str, tokenizer.apply_chat_template(  # transformers stub doesn't narrow tokenize=False return
        messages, tokenize=False, add_generation_prompt=False,
    ))
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    ids: torch.Tensor = enc["input_ids"]  # pyright: ignore[reportAssignmentType]  # BatchEncoding subscript returns Tensor at runtime
    if ids.numel() == 0:
        bos_id = tokenizer.bos_token_id or tokenizer.eos_token_id or 0
        ids = torch.tensor([[bos_id]])
    ids = ids.to(device)
    content_end = _last_content_idx(ids, tokenizer)
    hidden = _capture_all_hidden_states(model, layers, ids)
    return {
        L: h[0, min(content_end, h.shape[1] - 1)].float()
        for L, h in hidden.items()
    }


def build_ctx_profile(
    session: SaklasSession,
    scenarios: list[str],
    statements: list[dict[str, Any]],
    scenario_prompts: dict[str, str],
) -> Profile:
    model = session._model
    tokenizer = session._tokenizer
    layers = session._layers
    device = session._device

    pos_per_layer: dict[int, list[torch.Tensor]] = {}
    neg_per_layer: dict[int, list[torch.Tensor]] = {}
    norm_sums: dict[int, float] = {}
    norm_counts: dict[int, int] = {}

    pairs_per_scenario = len(statements) // len(scenarios)
    assert pairs_per_scenario * len(scenarios) == len(statements), (
        f"unexpected pack shape: {len(statements)} pairs / "
        f"{len(scenarios)} scenarios"
    )

    for i, pair in enumerate(statements):
        scenario = scenarios[i // pairs_per_scenario]
        user_prompt = scenario_prompts[scenario]
        pos_h = capture_prefill_with_context(
            model, tokenizer, layers, device,
            pair["positive"], user_prompt,
        )
        neg_h = capture_prefill_with_context(
            model, tokenizer, layers, device,
            pair["negative"], user_prompt,
        )
        for L in pos_h:
            p = pos_h[L].cpu()
            n = neg_h[L].cpu()
            pos_per_layer.setdefault(L, []).append(p)
            neg_per_layer.setdefault(L, []).append(n)
            norm_sums[L] = (
                norm_sums.get(L, 0.0) + float(p.norm()) + float(n.norm())
            )
            norm_counts[L] = norm_counts.get(L, 0) + 2
        if (i + 1) % 5 == 0:
            print(f"    [ctx] {i + 1}/{len(statements)} pairs captured")

    return _share_bake(
        session, pos_per_layer, neg_per_layer, norm_sums, norm_counts,
        metadata={
            "source": CTX_NAME,
            "method": "dim_prefill_with_synthetic_user_prompt",
            "n_pairs": len(statements),
        },
    )


# ---------------------------------------------------------------------------
# gen-extraction capture (replicated from previous script)
# ---------------------------------------------------------------------------


def generate_elicitation_prompts(
    session: SaklasSession, scenarios: list[str],
) -> list[str]:
    out: list[str] = []
    PROMPT_GEN_TEMPLATE = (
        "Write {n} short, neutral, open-ended prompts that someone could "
        "respond to about the topic '{scenario}'.  The prompts should not "
        "assume any emotional valence — they should be the kind of question "
        "a friend might ask or a journal might pose.  Output as a numbered "
        "list, one prompt per line, nothing else."
    )
    for scenario in scenarios:
        runset = session.generate(
            PROMPT_GEN_TEMPLATE.format(
                n=PROMPTS_PER_SCENARIO, scenario=scenario,
            ),
            sampling=SamplingConfig(
                max_tokens=220, temperature=0.6, top_p=0.9, seed=SEED,
            ),
            stateless=True,
            thinking=False,
        )
        # Parse numbered list (best-effort)
        prompts: list[str] = []
        for line in runset.first.text.splitlines():
            s = line.strip()
            if not s:
                continue
            i = 0
            while i < len(s) and (s[i].isdigit() or s[i] in ".)-:*# \t"):
                i += 1
            s = s[i:].strip().strip('"').strip("'")
            if s:
                prompts.append(s)
        prompts = prompts[:PROMPTS_PER_SCENARIO]
        if len(prompts) < PROMPTS_PER_SCENARIO:
            prompts += [scenario] * (PROMPTS_PER_SCENARIO - len(prompts))
        out.extend(prompts)
    return out


def capture_gen_pole(
    session: SaklasSession,
    prompts: list[str],
    system_prompt: str,
    label: str,
) -> tuple[
    dict[int, list[torch.Tensor]],
    dict[int, float],
    dict[int, int],
]:
    pooled: dict[int, list[torch.Tensor]] = {}
    norm_sums: dict[int, float] = {}
    norm_counts: dict[int, int] = {}
    for i, prompt in enumerate(prompts):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        runset = session.generate(
            messages,
            sampling=SamplingConfig(
                max_tokens=EXTRACT_MAX_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                seed=SEED + i,
                return_hidden=True,
            ),
            stateless=True,
            thinking=False,
        )
        hidden = runset.first.hidden_states
        if hidden is None:
            raise RuntimeError(f"{label}: return_hidden=True wasn't honored")
        for L, h in hidden.items():
            h32 = h.float().cpu()
            pooled.setdefault(L, []).append(h32.mean(dim=0))
            norm_sums[L] = (
                norm_sums.get(L, 0.0) + float(h32.norm(dim=-1).sum())
            )
            norm_counts[L] = norm_counts.get(L, 0) + h32.shape[0]
        if (i + 1) % 9 == 0:
            print(f"    [{label}] {i + 1}/{len(prompts)} captured")
    return pooled, norm_sums, norm_counts


# ---------------------------------------------------------------------------
# share-bake (DiM + Mahalanobis + DLS, shared by ctx and gen pipelines)
# ---------------------------------------------------------------------------


def _share_bake(
    session: SaklasSession,
    pos_per_layer: dict[int, list[torch.Tensor]],
    neg_per_layer: dict[int, list[torch.Tensor]],
    norm_sums: dict[int, float],
    norm_counts: dict[int, int],
    *,
    metadata: dict[str, Any],
) -> Profile:
    layers = sorted(set(pos_per_layer) & set(neg_per_layer))
    mu_pos = {L: torch.stack(pos_per_layer[L]).mean(dim=0) for L in layers}
    mu_neg = {L: torch.stack(neg_per_layer[L]).mean(dim=0) for L in layers}
    diff = {L: (mu_pos[L] - mu_neg[L]).float() for L in layers}
    unit_dir = {
        L: diff[L] / max(float(diff[L].norm()), 1e-12) for L in layers
    }
    ref_norms = {L: norm_sums[L] / max(norm_counts[L], 1) for L in layers}

    keep = compute_dls_mask(
        mu_pos, mu_neg, unit_dir, session.layer_means,
    )
    print(f"    DLS retained: {len(keep)}/{len(layers)} layers")

    whitener = session.whitener
    scores: dict[int, float] = {}
    for L in keep:
        d = diff[L]
        ref = max(ref_norms[L], 1e-8)
        if whitener is not None and whitener.covers(L):
            scores[L] = float(whitener.mahalanobis_norm(L, d) / ref)
        else:
            scores[L] = float(d.norm() / ref)

    total = sum(scores.values())
    baked: dict[int, torch.Tensor] = {}
    for L in keep:
        share = scores[L] / total if total > 0 else 1.0 / len(keep)
        baked[L] = unit_dir[L] * ref_norms[L] * share

    return Profile(baked, metadata=metadata)


# ---------------------------------------------------------------------------
# downstream: steering grid + scoring
# ---------------------------------------------------------------------------


def project_to_profile(
    hidden_means: dict[int, torch.Tensor],
    profile: Profile,
    layer_means: dict[int, torch.Tensor],
) -> float:
    total = 0.0
    for L, baked in profile.items():
        h = hidden_means.get(L)
        if h is None:
            continue
        h32 = h.float().cpu()
        mu = layer_means.get(L)
        if mu is not None:
            h32 = h32 - mu.float().cpu()
        total += float(h32 @ baked.float().cpu())
    return total


def llm_judge(session: SaklasSession, text: str) -> float | None:
    runset = session.generate(
        JUDGE_PROMPT.format(text=text),
        sampling=SamplingConfig(
            max_tokens=10, temperature=0.0, seed=0,
        ),
        stateless=True,
        thinking=False,
    )
    raw = runset.first.text.strip()
    match = re.search(r"-?\d+(?:\.\d+)?", raw)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def run_condition(
    session: SaklasSession,
    prompt: str,
    steering_expr: str | None,
    seed: int,
) -> tuple[str, dict[int, torch.Tensor]]:
    runset = session.generate(
        prompt,
        steering=steering_expr,
        sampling=SamplingConfig(
            max_tokens=TEST_MAX_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            seed=seed,
            return_hidden=True,
        ),
        stateless=True,
        thinking=False,
    )
    result = runset.first
    if result.hidden_states is None:
        raise RuntimeError("return_hidden=True wasn't honored under steering")
    hidden_means = {
        L: h.float().cpu().mean(dim=0)
        for L, h in result.hidden_states.items()
    }
    return result.text, hidden_means


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    print(f"loading model {MODEL_ID}")
    session = SaklasSession.from_pretrained(MODEL_ID, device="auto")

    scenarios, statements = load_concept_pack()
    print(
        f"\nbundled {CONCEPT}: {len(scenarios)} scenarios, "
        f"{len(statements)} pairs"
    )

    # --- profile 1: folded bundled prefill profile ----------------------
    print("\n=== profile 1: prefill (folded bundled manifold) ===")
    prefill_prof = load_folded_bundled_profile(session, f"default/{CONCEPT}")
    print(f"  prefill profile: {len(prefill_prof)} layers retained")

    # --- profile 2: prefill+ctx (a9's proposal) -------------------------
    print("\n=== profile 2: prefill+ctx (synthetic user prompt) ===")
    print("  generating one synthetic prompt per scenario...")
    scenario_prompts = generate_scenario_prompts(session, scenarios)

    print(
        f"\n  building ctx profile "
        f"({len(statements)} pairs × 2 forward passes)..."
    )
    ctx_prof = build_ctx_profile(
        session, scenarios, statements, scenario_prompts,
    )
    print(f"  ctx profile: {len(ctx_prof)} layers retained")
    session.steer(CTX_NAME, ctx_prof)

    # --- profile 3: gen extraction --------------------------------------
    print("\n=== profile 3: gen (full role-conditioned generation) ===")
    print(
        f"  generating {PROMPTS_PER_SCENARIO} elicitation prompts "
        f"per scenario..."
    )
    elicit = generate_elicitation_prompts(session, scenarios)

    print(f"\n  capturing under HAPPY system prompt "
          f"({len(elicit)} generations)...")
    happy_pool, hp_sums, hp_counts = capture_gen_pole(
        session, elicit, POS_SYSTEM, "pos",
    )
    print(f"\n  capturing under NEG system prompt "
          f"({len(elicit)} generations)...")
    sad_pool, sp_sums, sp_counts = capture_gen_pole(
        session, elicit, NEG_SYSTEM, "neg",
    )
    gen_norm_sums = {
        L: hp_sums.get(L, 0.0) + sp_sums.get(L, 0.0)
        for L in set(hp_sums) | set(sp_sums)
    }
    gen_norm_counts = {
        L: hp_counts.get(L, 0) + sp_counts.get(L, 0)
        for L in set(hp_counts) | set(sp_counts)
    }
    print("\n  share-baking gen profile...")
    gen_prof = _share_bake(
        session, happy_pool, sad_pool, gen_norm_sums, gen_norm_counts,
        metadata={
            "source": GEN_NAME,
            "method": "dim_role_conditioned_generation",
            "n_prompts": len(elicit),
        },
    )
    print(f"  gen profile: {len(gen_prof)} layers retained")
    session.steer(GEN_NAME, gen_prof)

    # --- pre-flight: per-layer cosine between the three profiles --------
    def per_layer_cosines(a: Profile, b: Profile) -> tuple[float, int]:
        cos: list[float] = []
        for L in sorted(set(a) & set(b)):
            va = a[L].float().cpu().flatten()
            vb = b[L].float().cpu().flatten()
            na = float(va.norm())
            nb = float(vb.norm())
            if na < 1e-12 or nb < 1e-12:
                cos.append(0.0)
                continue
            cos.append(float((va @ vb) / (na * nb)))
        if not cos:
            return float("nan"), 0
        return sum(cos) / len(cos), len(cos)

    print("\n=== per-layer cosines (mean across overlapping layers) ===")
    pc, pn = per_layer_cosines(prefill_prof, ctx_prof)
    pg, png = per_layer_cosines(prefill_prof, gen_prof)
    cg, cgn = per_layer_cosines(ctx_prof, gen_prof)
    print(f"  prefill ↔ ctx:   mean cos = {pc:+.4f}  (over {pn} layers)")
    print(f"  prefill ↔ gen:   mean cos = {pg:+.4f}  (over {png} layers)")
    print(f"  ctx     ↔ gen:   mean cos = {cg:+.4f}  (over {cgn} layers)")

    # --- steering grid --------------------------------------------------
    conditions: list[tuple[str, str | None]] = [
        ("baseline",     None),
        ("prefill_pos", f"{ALPHA} default/{CONCEPT}"),
        ("prefill_neg", f"-{ALPHA} default/{CONCEPT}"),
        ("ctx_pos",     f"{ALPHA} {CTX_NAME}"),
        ("ctx_neg",     f"-{ALPHA} {CTX_NAME}"),
        ("gen_pos",     f"{ALPHA} {GEN_NAME}"),
        ("gen_neg",     f"-{ALPHA} {GEN_NAME}"),
    ]
    total = len(HELD_OUT_PROMPTS) * len(conditions) * len(TEST_SEEDS)
    print(
        f"\n=== running steering grid: "
        f"{len(HELD_OUT_PROMPTS)} prompts × {len(conditions)} conditions × "
        f"{len(TEST_SEEDS)} seeds = {total} generations ==="
    )
    layer_means = session.layer_means

    outputs: dict[tuple[int, str, int], dict[str, Any]] = {}
    for p_idx, prompt in enumerate(HELD_OUT_PROMPTS):
        print(f"\n  prompt {p_idx + 1}/{len(HELD_OUT_PROMPTS)}: {prompt!r}")
        for cond_name, expr in conditions:
            for seed in TEST_SEEDS:
                text, hmeans = run_condition(session, prompt, expr, seed)
                outputs[(p_idx, cond_name, seed)] = {
                    "text": text,
                    "hidden_means": hmeans,
                }
                head = text[:80].replace("\n", " ")
                print(f"    [{cond_name:<12} seed={seed}] {head!r}")

    # --- scoring --------------------------------------------------------
    print("\n=== cross-projection onto all three profiles ===")
    prefill_proj: dict[tuple[int, str, int], float] = {}
    ctx_proj: dict[tuple[int, str, int], float] = {}
    gen_proj: dict[tuple[int, str, int], float] = {}
    for key, rec in outputs.items():
        prefill_proj[key] = project_to_profile(
            rec["hidden_means"], prefill_prof, layer_means,
        )
        ctx_proj[key] = project_to_profile(
            rec["hidden_means"], ctx_prof, layer_means,
        )
        gen_proj[key] = project_to_profile(
            rec["hidden_means"], gen_prof, layer_means,
        )

    print(f"\n=== LLM-judge scoring ({len(outputs)} outputs) ===")
    judge_scores: dict[tuple[int, str, int], float | None] = {}
    for i, (key, rec) in enumerate(outputs.items()):
        judge_scores[key] = llm_judge(session, rec["text"])
        if (i + 1) % 21 == 0:
            print(f"  judged {i + 1}/{len(outputs)}")

    # --- aggregate ------------------------------------------------------
    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else float("nan")

    print("\n=== aggregate by condition ===")
    print(
        f"{'condition':<14} {'n':>3} {'judge':>8} "
        f"{'prefill_proj':>14} {'ctx_proj':>12} {'gen_proj':>12}"
    )
    print("-" * 72)
    agg: dict[str, dict[str, float]] = {}
    for cond_name, _ in conditions:
        keys = [k for k in outputs if k[1] == cond_name]
        judges: list[float] = [
            j for k in keys if (j := judge_scores[k]) is not None
        ]
        agg[cond_name] = {
            "n": float(len(keys)),
            "n_judged": float(len(judges)),
            "mean_judge": mean(judges),
            "mean_prefill_proj": mean([prefill_proj[k] for k in keys]),
            "mean_ctx_proj": mean([ctx_proj[k] for k in keys]),
            "mean_gen_proj": mean([gen_proj[k] for k in keys]),
        }
        a = agg[cond_name]
        print(
            f"{cond_name:<14} {int(a['n']):>3} {a['mean_judge']:>+8.2f} "
            f"{a['mean_prefill_proj']:>+14.1f} "
            f"{a['mean_ctx_proj']:>+12.1f} "
            f"{a['mean_gen_proj']:>+12.1f}"
        )

    # --- swings ---------------------------------------------------------
    base = agg["baseline"]
    print("\n=== shift vs baseline ===")
    print(
        f"{'condition':<14} {'Δ judge':>10} {'Δ prefill':>12} "
        f"{'Δ ctx':>10} {'Δ gen':>10}"
    )
    print("-" * 60)
    for cond_name, _ in conditions:
        if cond_name == "baseline":
            continue
        a = agg[cond_name]
        print(
            f"{cond_name:<14} "
            f"{a['mean_judge'] - base['mean_judge']:>+10.2f} "
            f"{a['mean_prefill_proj'] - base['mean_prefill_proj']:>+12.1f} "
            f"{a['mean_ctx_proj'] - base['mean_ctx_proj']:>+10.1f} "
            f"{a['mean_gen_proj'] - base['mean_gen_proj']:>+10.1f}"
        )

    # --- total judge swing per method ----------------------------------
    def total_swing(pos_name: str, neg_name: str) -> float:
        return (
            (agg[pos_name]["mean_judge"] - base["mean_judge"])
            + (base["mean_judge"] - agg[neg_name]["mean_judge"])
        )

    pref_swing = total_swing("prefill_pos", "prefill_neg")
    ctx_swing = total_swing("ctx_pos", "ctx_neg")
    gen_swing = total_swing("gen_pos", "gen_neg")

    print("\n=== total judge swing (positive Δ + magnitude of negative Δ) ===")
    print(f"  prefill (current):  {pref_swing:+.2f}")
    print(f"  prefill+ctx (a9):   {ctx_swing:+.2f}")
    print(f"  gen extraction:     {gen_swing:+.2f}")

    # --- verdict --------------------------------------------------------
    recovery = (
        (ctx_swing - pref_swing) / max(gen_swing - pref_swing, 1e-6)
        if gen_swing > 0
        else float("nan")
    )
    print(
        f"\n  ctx recovery of gen-over-prefill gain: "
        f"{recovery * 100:.0f}%  "
        f"(0% = same as prefill, 100% = same as gen)"
    )

    if recovery > 0.75:
        verdict = (
            "ctx recovers most of the gen gain — the OOD wart was "
            "primarily about missing conversational context, not about "
            "the assistant being prefilled.  Big win: prefill+ctx is "
            "near-free and nearly matches generation extraction."
        )
    elif recovery > 0.4:
        verdict = (
            "ctx recovers a substantial fraction of the gen gain — both "
            "factors contribute.  prefill+ctx is a good cheap default; "
            "gen extraction remains worthwhile when steering quality "
            "matters more than extraction cost."
        )
    elif recovery > 0.1:
        verdict = (
            "ctx recovers only a small fraction of the gen gain — the "
            "OOD wart is mostly about the assistant text being prefilled "
            "rather than generated.  Stay with current prefill or pay "
            "for full gen extraction."
        )
    else:
        verdict = (
            "ctx is no better than current prefill — the synthetic user "
            "prompt didn't move the needle.  The OOD wart isn't where "
            "we thought."
        )
    print(f"\n  verdict: {verdict}")

    # --- save -----------------------------------------------------------
    OUT_PATH.write_text(
        json.dumps(
            {
                "model": MODEL_ID,
                "alpha": ALPHA,
                "seeds": TEST_SEEDS,
                "scenarios": scenarios,
                "scenario_prompts": scenario_prompts,
                "n_extract_pairs": len(statements),
                "n_elicit_prompts": len(elicit),
                "held_out_prompts": HELD_OUT_PROMPTS,
                "conditions": [c for c, _ in conditions],
                "pre_cosines": {
                    "prefill_vs_ctx": pc,
                    "prefill_vs_gen": pg,
                    "ctx_vs_gen": cg,
                },
                "outputs": {
                    f"{p}|{c}|{s}": {
                        "text": outputs[(p, c, s)]["text"],
                        "judge": judge_scores[(p, c, s)],
                        "prefill_proj": prefill_proj[(p, c, s)],
                        "ctx_proj": ctx_proj[(p, c, s)],
                        "gen_proj": gen_proj[(p, c, s)],
                    }
                    for (p, c, s) in outputs
                },
                "aggregate": agg,
                "swings": {
                    "prefill": pref_swing,
                    "ctx": ctx_swing,
                    "gen": gen_swing,
                    "ctx_recovery_pct": recovery * 100,
                },
                "verdict": verdict,
            },
            indent=2,
            default=str,
        )
    )
    print(f"\nraw data -> {OUT_PATH}")


if __name__ == "__main__":
    main()
