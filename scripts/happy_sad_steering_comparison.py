"""Steering quality: prefill vs generation-extracted happy.sad at matched α.

Follow-up to ``happy_sad_prefill_vs_generation.py``.  That script
established that the two extraction methods recover meaningfully
distinct directions (median cosine ~0.30 in 5120-dim space, 5× larger
magnitudes at deep layers under generation extraction).  Cosines alone
don't say which is *better*; this one runs the downstream test.

Pipeline:

  1. Extract the bundled prefill profile (``default/happy.sad``).
  2. Re-run the generation-extraction pass and share-bake it into a
     full ``Profile`` via the same DiM + Mahalanobis-score + DLS-mask +
     share-bake math ``extract_difference_of_means`` runs internally.
     Register it as ``happy_sad_gen`` in the session.
  3. For each of 5 held-out neutral prompts, generate at 3 seeds under
     5 conditions:  baseline, prefill ±0.5, gen ±0.5.
  4. Score every output two ways:
       LLM_JUDGE        — the same gemma rates each text on a -5..+5
                          happy/sad scale (thinking off, T=0).
       CROSS_PROJ       — mean hidden state per layer · baked direction,
                          centered by layer_means.  Saklas-native,
                          direction-aware, free (we already capture
                          hidden states with return_hidden=True).

The cross-projection answers two distinct questions:

  * Does each profile steer along its own direction?  prefill steering
    should shift prefill_proj; gen steering should shift gen_proj.
  * Does prefill steering also shift gen_proj (and vice versa)?  If
    yes, the two directions span the same axis even though they
    differ geometrically (cosine ~0.30 from the prior experiment).
    If no, they're orthogonal axes — switching profile would mean
    steering a different concept.

The LLM judge is the behavioral readout that mediates whichever of
those two stories cosines tell.  Both scores together let us decide
whether the cost of generation extraction buys anything users would
notice.
"""

from __future__ import annotations

import json
import logging
import re
from importlib import resources
from pathlib import Path

from typing import Any

import torch

from saklas import Profile, SamplingConfig, SaklasSession
from saklas.core.vectors import compute_dls_mask


MODEL_ID = "google/gemma-4-31b-it"
PROMPTS_PER_SCENARIO = 3
EXTRACT_MAX_TOKENS = 80
TEST_MAX_TOKENS = 100
TEMPERATURE = 0.7
TOP_P = 0.95
SEED = 42
ALPHA = 0.5                        # bundled happy.sad ``recommended_alpha``
TEST_SEEDS = [42, 1337, 7]
OUT_PATH = Path("/tmp/happy_sad_steering_comparison.json")

HAPPY_SYSTEM = (
    "You feel deeply, genuinely happy right now. "
    "Respond to the user from inside that emotional state, "
    "in first person, in one short paragraph."
)
SAD_SYSTEM = (
    "You feel deeply, genuinely sad right now. "
    "Respond to the user from inside that emotional state, "
    "in first person, in one short paragraph."
)

PROMPT_GEN_TEMPLATE = (
    "Write {n} short, neutral, open-ended prompts that someone could "
    "respond to about the topic '{scenario}'.  The prompts should not "
    "assume any emotional valence — they should be the kind of question "
    "a friend might ask or a journal might pose.  Output as a numbered "
    "list, one prompt per line, nothing else."
)

# Deliberately held out from the happy.sad extraction scenarios.  All
# open-ended first-person invitations — affect-neutral surface, the
# kind of prompt where steering surfaces affect cleanly without having
# to override topical signal.
HELD_OUT_PROMPTS = [
    "Tell me about your morning.",
    "What's the most interesting thing you've thought about this week?",
    "Describe the place where you are right now.",
    "What are you working on lately?",
    "Tell me about a memorable conversation you had recently.",
]

JUDGE_PROMPT = (
    "Read the following text and rate, on a scale from -5 to +5, how "
    "emotionally positive (happy) or negative (sad) it sounds.\n\n"
    "Scale: -5 means deeply sad / despairing, 0 means neutral, +5 means "
    "deeply happy / joyful.\n\n"
    "Reply with ONLY a single number (with optional sign and decimal). "
    "No other words, no explanation.\n\n"
    "Text:\n{text}\n\nScore:"
)


# ---------------------------------------------------------------------------
# generation-extraction (replicates the prior script + share-bake)
# ---------------------------------------------------------------------------


def load_happy_sad_scenarios() -> list[str]:
    pkg = resources.files("saklas.data.vectors").joinpath(
        "happy.sad/scenarios.json",
    )
    with pkg.open() as f:
        return json.load(f)["scenarios"]


def parse_numbered_list(text: str, expected: int) -> list[str]:
    out: list[str] = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        i = 0
        while i < len(s) and (s[i].isdigit() or s[i] in ".)-:*# \t"):
            i += 1
        s = s[i:].strip().strip('"').strip("'")
        if s:
            out.append(s)
    return out[:expected]


def generate_elicitation_prompts(
    session: SaklasSession, scenarios: list[str],
) -> list[str]:
    out: list[str] = []
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
        prompts = parse_numbered_list(runset.first.text, PROMPTS_PER_SCENARIO)
        if len(prompts) < PROMPTS_PER_SCENARIO:
            prompts += [scenario] * (PROMPTS_PER_SCENARIO - len(prompts))
        out.extend(prompts)
    return out


def capture_pole(
    session: SaklasSession,
    prompts: list[str],
    system_prompt: str,
    label: str,
) -> tuple[
    dict[int, list[torch.Tensor]],
    dict[int, float],
    dict[int, int],
]:
    """Pool hidden states per generation under ``system_prompt``.

    Returns:
        ``(per_layer_pooled, norm_sums, norm_counts)``.
        ``per_layer_pooled[L]`` is a list of fp32 CPU vectors, one per
        prompt, each = mean(generated-tokens hidden states) at layer L.
        ``norm_sums`` and ``norm_counts`` accumulate over every captured
        token (not the pooled vector!) for ``ref_norm`` computation.
    """
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
            raise RuntimeError(
                f"{label}: return_hidden=True wasn't honored"
            )
        for L, h in hidden.items():
            h32 = h.float().cpu()                          # (T, D)
            pooled.setdefault(L, []).append(h32.mean(dim=0))
            norm_sums[L] = (
                norm_sums.get(L, 0.0) + float(h32.norm(dim=-1).sum())
            )
            norm_counts[L] = norm_counts.get(L, 0) + h32.shape[0]
        head = runset.first.text[:60].replace("\n", " ")
        print(f"  [{label}] {i + 1}/{len(prompts)}  {head!r}")

    return pooled, norm_sums, norm_counts


def build_gen_profile(
    session: SaklasSession,
    happy_pool: dict[int, list[torch.Tensor]],
    sad_pool: dict[int, list[torch.Tensor]],
    norm_sums: dict[int, float],
    norm_counts: dict[int, int],
) -> Profile:
    """Replicate ``extract_difference_of_means``'s share-bake path.

    Inputs are per-prompt pooled activations under each pole; outputs
    are the same shape as ``session.extract`` would produce — a
    ``Profile`` of share-baked, ref-norm-scaled per-layer directions
    over a DLS-retained layer set.
    """
    layers = sorted(set(happy_pool) & set(sad_pool))

    mu_pos = {L: torch.stack(happy_pool[L]).mean(dim=0) for L in layers}
    mu_neg = {L: torch.stack(sad_pool[L]).mean(dim=0) for L in layers}
    diff = {L: (mu_pos[L] - mu_neg[L]).float() for L in layers}
    unit_dir = {
        L: diff[L] / max(float(diff[L].norm()), 1e-12) for L in layers
    }
    ref_norms = {
        L: norm_sums[L] / max(norm_counts[L], 1) for L in layers
    }

    keep = compute_dls_mask(
        mu_pos, mu_neg, unit_dir, session.layer_means,
    )
    print(f"  DLS retained: {len(keep)}/{len(layers)} layers")

    whitener = session.whitener
    scores: dict[int, float] = {}
    for L in keep:
        d = diff[L]
        ref = max(ref_norms[L], 1e-8)
        if whitener is not None and whitener.covers(L):
            m_norm = whitener.mahalanobis_norm(L, d)
            scores[L] = float(m_norm / ref)
        else:
            scores[L] = float(d.norm() / ref)

    total = sum(scores.values())
    baked: dict[int, torch.Tensor] = {}
    for L in keep:
        share = scores[L] / total if total > 0 else 1.0 / len(keep)
        baked[L] = unit_dir[L] * ref_norms[L] * share

    return Profile(
        baked,
        metadata={
            "source": "happy_sad_gen",
            "method": "dim_generation",
            "n_prompts": len(next(iter(happy_pool.values()))),
            "model_id": MODEL_ID,
        },
    )


# ---------------------------------------------------------------------------
# steering comparison
# ---------------------------------------------------------------------------


def project_to_profile(
    hidden_means: dict[int, torch.Tensor],
    profile: Profile,
    layer_means: dict[int, torch.Tensor],
) -> float:
    """Saklas-native readout: ∑_L (mean_hidden_L − μ_L) · baked_L (fp32).

    Centered by layer_means so the baseline-bias cancels.  Sign matches
    the profile's polarity convention (positive ↔ +happy by extraction
    orientation).  Magnitude is in raw activation·direction units —
    not normalised, only relative differences across conditions are
    interpretable.
    """
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
        raise RuntimeError(
            "return_hidden=True wasn't honored under steering"
        )
    # Pool over generated tokens for the per-layer projection score.
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

    # --- prefill profile (cache hit) ------------------------------------
    print("\n=== prefill extraction (bundled happy.sad) ===")
    _, prefill_prof = session.extract("default/happy.sad")
    print(f"  prefill profile: {len(prefill_prof)} layers")

    # --- generation profile (fresh capture pass) ------------------------
    print("\n=== generation extraction ===")
    scenarios = load_happy_sad_scenarios()
    print(f"  reusing {len(scenarios)} bundled scenarios")
    print("  writing elicitation prompts...")
    elicit_prompts = generate_elicitation_prompts(session, scenarios)
    print(f"  total: {len(elicit_prompts)} elicitation prompts")

    print("\n  capturing under HAPPY system prompt...")
    happy_pool, hp_sums, hp_counts = capture_pole(
        session, elicit_prompts, HAPPY_SYSTEM, "happy",
    )
    print("\n  capturing under SAD system prompt...")
    sad_pool, sp_sums, sp_counts = capture_pole(
        session, elicit_prompts, SAD_SYSTEM, "sad",
    )

    # Combine per-pole token-norm accumulators (DiM bake uses
    # pos+neg averaged refnorm exactly like the prefill side).
    norm_sums = {
        L: hp_sums.get(L, 0.0) + sp_sums.get(L, 0.0)
        for L in set(hp_sums) | set(sp_sums)
    }
    norm_counts = {
        L: hp_counts.get(L, 0) + sp_counts.get(L, 0)
        for L in set(hp_counts) | set(sp_counts)
    }

    print("\n  share-baking gen profile...")
    gen_prof = build_gen_profile(
        session, happy_pool, sad_pool, norm_sums, norm_counts,
    )
    print(f"  gen profile: {len(gen_prof)} layers")
    session.steer("happy_sad_gen", gen_prof)
    print("  registered as 'happy_sad_gen' in session")

    # --- steering comparison --------------------------------------------
    conditions: list[tuple[str, str | None]] = [
        ("baseline",     None),
        ("prefill_pos", f"{ALPHA} default/happy.sad"),
        ("prefill_neg", f"-{ALPHA} default/happy.sad"),
        ("gen_pos",     f"{ALPHA} happy_sad_gen"),
        ("gen_neg",     f"-{ALPHA} happy_sad_gen"),
    ]

    print(
        f"\n=== running steering grid: "
        f"{len(HELD_OUT_PROMPTS)} prompts × "
        f"{len(conditions)} conditions × {len(TEST_SEEDS)} seeds "
        f"= {len(HELD_OUT_PROMPTS) * len(conditions) * len(TEST_SEEDS)} "
        f"generations ==="
    )
    layer_means = session.layer_means

    # Keyed by (prompt_idx, condition_name, seed).
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
    print("\n=== cross-projection scoring (free, model-internal) ===")
    prefill_proj: dict[tuple[int, str, int], float] = {}
    gen_proj: dict[tuple[int, str, int], float] = {}
    for key, rec in outputs.items():
        prefill_proj[key] = project_to_profile(
            rec["hidden_means"], prefill_prof, layer_means,
        )
        gen_proj[key] = project_to_profile(
            rec["hidden_means"], gen_prof, layer_means,
        )

    print("\n=== LLM-judge scoring (gemma rates each output, -5..+5) ===")
    judge_scores: dict[tuple[int, str, int], float | None] = {}
    for i, (key, rec) in enumerate(outputs.items()):
        score = llm_judge(session, rec["text"])
        judge_scores[key] = score
        if (i + 1) % 15 == 0:
            print(f"  judged {i + 1}/{len(outputs)}")

    # --- aggregate ------------------------------------------------------
    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else float("nan")

    print("\n=== aggregate by condition ===")
    print(
        f"{'condition':<14} {'n':>3} {'n_judged':>9} "
        f"{'judge(-5..+5)':>14} {'prefill_proj':>14} {'gen_proj':>12}"
    )
    print("-" * 72)
    agg: dict[str, dict[str, float]] = {}
    for cond_name, _ in conditions:
        keys = [k for k in outputs if k[1] == cond_name]
        judges: list[float] = [
            j for k in keys
            if (j := judge_scores[k]) is not None
        ]
        prefs = [prefill_proj[k] for k in keys]
        gens = [gen_proj[k] for k in keys]
        agg[cond_name] = {
            "n": len(keys),
            "n_judged": len(judges),
            "mean_judge": mean(judges),
            "mean_prefill_proj": mean(prefs),
            "mean_gen_proj": mean(gens),
        }
        a = agg[cond_name]
        judge_str = (
            f"{a['mean_judge']:>+14.2f}" if judges else f"{'n/a':>14}"
        )
        print(
            f"{cond_name:<14} {a['n']:>3} {a['n_judged']:>9} {judge_str} "
            f"{a['mean_prefill_proj']:>+14.2f} {a['mean_gen_proj']:>+12.2f}"
        )

    # --- behavioral shifts (vs baseline) --------------------------------
    base = agg["baseline"]
    print("\n=== shift vs baseline (positive = more 'happy' direction) ===")
    print(
        f"{'condition':<14} {'Δ judge':>10} {'Δ prefill':>12} "
        f"{'Δ gen':>10}"
    )
    print("-" * 50)
    for cond_name, _ in conditions:
        if cond_name == "baseline":
            continue
        a = agg[cond_name]
        d_judge = a["mean_judge"] - base["mean_judge"]
        d_pref = a["mean_prefill_proj"] - base["mean_prefill_proj"]
        d_gen = a["mean_gen_proj"] - base["mean_gen_proj"]
        print(
            f"{cond_name:<14} {d_judge:>+10.2f} {d_pref:>+12.2f} "
            f"{d_gen:>+10.2f}"
        )

    # --- verdict --------------------------------------------------------
    pref_pos_shift = agg["prefill_pos"]["mean_judge"] - base["mean_judge"]
    pref_neg_shift = base["mean_judge"] - agg["prefill_neg"]["mean_judge"]
    gen_pos_shift = agg["gen_pos"]["mean_judge"] - base["mean_judge"]
    gen_neg_shift = base["mean_judge"] - agg["gen_neg"]["mean_judge"]

    pref_total = pref_pos_shift + pref_neg_shift
    gen_total = gen_pos_shift + gen_neg_shift

    print("\n=== verdict ===")
    print(
        f"  prefill total swing (judge):  "
        f"{pref_pos_shift:+.2f} (+) + {pref_neg_shift:+.2f} (-) "
        f"= {pref_total:+.2f}"
    )
    print(
        f"  gen total swing (judge):      "
        f"{gen_pos_shift:+.2f} (+) + {gen_neg_shift:+.2f} (-) "
        f"= {gen_total:+.2f}"
    )
    diff = gen_total - pref_total
    if abs(diff) < 0.3:
        verdict = (
            "matched steering quality — both methods produce comparable "
            "behavioral shifts at α=0.5.  Generation extraction's "
            "K× cost doesn't buy measurable steering quality."
        )
    elif diff > 0:
        verdict = (
            "generation steers MORE strongly than prefill at matched α — "
            "the larger deep-layer magnitudes do produce a larger "
            "behavioral shift.  Cost case for switching becomes "
            "decidable on the basis of how much shift is wanted."
        )
    else:
        verdict = (
            "prefill steers MORE strongly than generation at matched α — "
            "despite generation's larger raw magnitudes, the angular "
            "share allocation funnels budget into deep layers that "
            "don't move the behavioral output as much.  Prefill stays "
            "the better default."
        )
    print(f"  verdict: {verdict}")

    # --- save -----------------------------------------------------------
    OUT_PATH.write_text(
        json.dumps(
            {
                "model": MODEL_ID,
                "alpha": ALPHA,
                "seeds": TEST_SEEDS,
                "held_out_prompts": HELD_OUT_PROMPTS,
                "conditions": [c for c, _ in conditions],
                "n_extract_prompts": len(elicit_prompts),
                "outputs": {
                    f"{p}|{c}|{s}": {
                        "text": outputs[(p, c, s)]["text"],
                        "judge": judge_scores[(p, c, s)],
                        "prefill_proj": prefill_proj[(p, c, s)],
                        "gen_proj": gen_proj[(p, c, s)],
                    }
                    for (p, c, s) in outputs
                },
                "aggregate": agg,
                "shifts_vs_baseline": {
                    cond_name: {
                        "delta_judge": agg[cond_name]["mean_judge"]
                            - base["mean_judge"],
                        "delta_prefill_proj": agg[cond_name]["mean_prefill_proj"]
                            - base["mean_prefill_proj"],
                        "delta_gen_proj": agg[cond_name]["mean_gen_proj"]
                            - base["mean_gen_proj"],
                    }
                    for cond_name, _ in conditions
                    if cond_name != "baseline"
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
