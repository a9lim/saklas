"""Prefill vs generation extraction of happy.sad on gemma-4-31b-it.

The current saklas pipeline prefills hand-written contrastive statements
as if the assistant said them, then pools the last content token's
hidden state per layer.  This script re-extracts the same concept by
running real generations under a role-conditioning system prompt
("respond as someone who feels deeply happy" / "...deeply sad") and
pooling hidden states over the generated tokens.

Reports per-layer cosine between the two direction profiles plus a
summary.  Tight alignment (median |cos| > ~0.7) says the prefill
OOD-wart isn't actually costing the current pipeline much.  Material
divergence (median |cos| < ~0.4) says the generation-extracted vector
is meaningfully different and worth chasing.

Two generation-side pooling modes are reported side-by-side:

  * MEAN_ALL   — mean over every generated token of each response.
                 Whole-trajectory readout; the persona signal
                 dilutes as the model conditions on its own
                 continuation.
  * MEAN_EARLY — mean over the first ``EARLY_TOKENS`` generated
                 tokens.  Closer in spirit to prefill's
                 last-content-token pooling — captures the residual
                 right after the model has committed to the persona
                 and before its own continuation dominates.

If MEAN_EARLY tracks prefill notably better than MEAN_ALL, that's
evidence the dilution hypothesis is real and the middle-path option
(generate-but-cap-capture) is worth wiring up properly.

Reuses the bundled happy.sad scenarios (9) for topic parity; asks the
model to write ``PROMPTS_PER_SCENARIO`` neutral elicitation prompts per
scenario; each prompt is then run through both poles under matched
seeds.  ~9*3*2 = 54 generations at MAX_TOKENS=80.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

from saklas import SamplingConfig, SaklasSession
from saklas.core.vectors import compute_dls_axes
from _bundled_manifold_data import (
    load_bundled_manifold_scenarios,
    load_folded_bundled_profile,
)


MODEL_ID = "google/gemma-4-31b-it"
PROMPTS_PER_SCENARIO = 3
MAX_TOKENS = 80
EARLY_TOKENS = 5            # MEAN_EARLY pooling window
TEMPERATURE = 0.7
TOP_P = 0.95
SEED = 42
OUT_PATH = Path("/tmp/happy_sad_prefill_vs_generation.json")

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


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def load_happy_sad_scenarios() -> list[str]:
    return load_bundled_manifold_scenarios("happy.sad")


def parse_numbered_list(text: str, expected: int) -> list[str]:
    out: list[str] = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        # Trim leading "1.", "1)", "1:", "- ", "* ", etc.
        i = 0
        while i < len(s) and (s[i].isdigit() or s[i] in ".)-:*# \t"):
            i += 1
        s = s[i:].strip().strip('"').strip("'")
        if s:
            out.append(s)
    return out[:expected]


def generate_elicitation_prompts(
    session: SaklasSession, scenarios: list[str]
) -> dict[str, list[str]]:
    """One LLM call per scenario; ask for N neutral prompts each."""
    out: dict[str, list[str]] = {}
    for scenario in scenarios:
        runset = session.generate(
            PROMPT_GEN_TEMPLATE.format(
                n=PROMPTS_PER_SCENARIO, scenario=scenario,
            ),
            sampling=SamplingConfig(
                max_tokens=220,
                temperature=0.6,
                top_p=0.9,
                seed=SEED,
            ),
            stateless=True,
            # Gemma-4 has reasoning on by default; without this the
            # response is a chain-of-thought planning prelude instead of
            # the requested numbered list.
            thinking=False,
        )
        prompts = parse_numbered_list(
            runset.first.text, PROMPTS_PER_SCENARIO,
        )
        if len(prompts) < PROMPTS_PER_SCENARIO:
            # Pad with the bare scenario name — usable as a prompt and
            # leaves a clear signal in the JSON output that the parse
            # missed.
            print(
                f"  warn: parsed {len(prompts)}/{PROMPTS_PER_SCENARIO} "
                f"prompts for {scenario!r}; padding with bare scenario"
            )
            prompts += [scenario] * (PROMPTS_PER_SCENARIO - len(prompts))
        out[scenario] = prompts
        print(f"  {scenario}: {prompts[0][:80]!r}")
    return out


def capture_pole_means(
    session: SaklasSession,
    prompts: list[str],
    system_prompt: str,
    label: str,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], int]:
    """Generate one response per prompt under ``system_prompt``.

    Returns:
        ``(mean_all_per_layer, mean_early_per_layer, n_generations)``.
        Both means are per-layer averages of per-generation pooled
        hidden states (fp32 on CPU).
    """
    sums_all: dict[int, torch.Tensor] = {}
    sums_early: dict[int, torch.Tensor] = {}
    n = 0

    for i, prompt in enumerate(prompts):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        runset = session.generate(
            messages,
            sampling=SamplingConfig(
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                # Matched seed across poles for the same prompt index so
                # the only contrastive lever is the system prompt.
                seed=SEED + i,
                return_hidden=True,
            ),
            stateless=True,
            # Critical: without this, captured hidden states are
            # dominated by the model's reasoning-trace tokens
            # ("thought * Emotional state: deeply happy. * Format: ...")
            # rather than the actual persona-coded response.  Extracting
            # under thinking-on contrasts "planning to be happy" vs
            # "planning to be sad" — a meta-cognitive axis on the
            # system prompt, not the affect axis.
            thinking=False,
        )
        result = runset.first
        hidden = result.hidden_states
        if hidden is None:
            raise RuntimeError(
                f"{label}: return_hidden=True wasn't honored"
            )
        for layer_idx, h in hidden.items():
            h32 = h.float().cpu()                  # (T, D)
            mean_all = h32.mean(dim=0)             # (D,)
            mean_early = h32[:EARLY_TOKENS].mean(dim=0)
            sums_all[layer_idx] = (
                mean_all if layer_idx not in sums_all
                else sums_all[layer_idx] + mean_all
            )
            sums_early[layer_idx] = (
                mean_early if layer_idx not in sums_early
                else sums_early[layer_idx] + mean_early
            )
        n += 1
        head = result.text[:60].replace("\n", " ")
        print(f"  [{label}] {i + 1}/{len(prompts)}  {head!r}")

    means_all = {idx: s / n for idx, s in sums_all.items()}
    means_early = {idx: s / n for idx, s in sums_early.items()}
    return means_all, means_early, n


def per_layer_cosine(
    a: dict[int, torch.Tensor],
    b: dict[int, torch.Tensor],
) -> dict[int, float]:
    out: dict[int, float] = {}
    for L in sorted(set(a) & set(b)):
        va = a[L].float().flatten().cpu()
        vb = b[L].float().flatten().cpu()
        na = float(va.norm())
        nb = float(vb.norm())
        if na < 1e-12 or nb < 1e-12:
            out[L] = 0.0
            continue
        out[L] = float((va @ vb) / (na * nb))
    return out


def summarize(cosines: dict[int, float], label: str) -> dict[str, float]:
    vals = list(cosines.values())
    vals_sorted = sorted(vals)
    median_cos = vals_sorted[len(vals_sorted) // 2]
    mean_cos = sum(vals) / len(vals)
    mean_abs = sum(abs(v) for v in vals) / len(vals)
    n_aligned = sum(1 for v in vals if v > 0.5)
    n_opposed = sum(1 for v in vals if v < -0.5)
    n_orth = sum(1 for v in vals if abs(v) < 0.3)
    print(f"\n--- {label} ---")
    print(f"  overlapping layers:        {len(cosines)}")
    print(f"  median cosine:             {median_cos:+.4f}")
    print(f"  mean cosine:               {mean_cos:+.4f}")
    print(f"  mean |cosine|:             {mean_abs:+.4f}")
    print(f"  layers with cos > +0.5:    {n_aligned}/{len(vals)}")
    print(f"  layers with cos < -0.5:    {n_opposed}/{len(vals)}")
    print(f"  layers with |cos| < 0.3:   {n_orth}/{len(vals)}")
    return {
        "median_cosine": median_cos,
        "mean_cosine": mean_cos,
        "mean_abs_cosine": mean_abs,
        "n_layers": len(vals),
        "n_aligned_strong": n_aligned,
        "n_opposed_strong": n_opposed,
        "n_orthogonal": n_orth,
    }


def verdict_for(mean_abs: float) -> str:
    if mean_abs > 0.7:
        return (
            "tight alignment — prefill OOD wart is cheap, current pipeline "
            "captures essentially the same direction as generation"
        )
    if mean_abs > 0.4:
        return (
            "partial alignment — generation captures related but distinct "
            "signal; worth pursuing the middle path"
        )
    return (
        "directions disagree — generation extraction is materially different, "
        "strong case for switching to (or composing with) generation-based "
        "extraction"
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    print(f"loading model {MODEL_ID}")
    session = SaklasSession.from_pretrained(MODEL_ID, device="auto")

    # --- prefill side ------------------------------------------------------
    print("\n=== prefill profile (folded bundled happy.sad) ===")
    prefill_profile = load_folded_bundled_profile(session, "default/happy.sad")
    prefill_dirs: dict[int, torch.Tensor] = {
        L: t.float().cpu() for L, t in prefill_profile.items()
    }
    print(
        f"  prefill profile: {len(prefill_dirs)} layers retained "
        f"(out of full layer set after DLS)"
    )

    # --- generation side ---------------------------------------------------
    print("\n=== generation extraction ===")
    scenarios = load_happy_sad_scenarios()
    print(f"  reusing {len(scenarios)} bundled scenarios for topic parity")

    print("\n  generating elicitation prompts (1 LLM call per scenario)...")
    prompts_by_scenario = generate_elicitation_prompts(session, scenarios)
    all_prompts = [p for ps in prompts_by_scenario.values() for p in ps]
    print(f"  total: {len(all_prompts)} elicitation prompts")

    print("\n  capturing under HAPPY system prompt...")
    happy_all, happy_early, _ = capture_pole_means(
        session, all_prompts, HAPPY_SYSTEM, "happy",
    )

    print("\n  capturing under SAD system prompt...")
    sad_all, sad_early, _ = capture_pole_means(
        session, all_prompts, SAD_SYSTEM, "sad",
    )

    # Difference-of-means → generation direction profiles per pooling mode.
    gen_dirs_all: dict[int, torch.Tensor] = {
        L: (happy_all[L] - sad_all[L]).float().cpu()
        for L in sorted(set(happy_all) & set(sad_all))
    }
    gen_dirs_early: dict[int, torch.Tensor] = {
        L: (happy_early[L] - sad_early[L]).float().cpu()
        for L in sorted(set(happy_early) & set(sad_early))
    }
    print(
        f"\n  generation profiles: {len(gen_dirs_all)} layers each "
        f"(mean_all + mean_early)"
    )

    # --- DLS keep-set overlap ---------------------------------------------
    # Prefill side: layers present in `prefill_dirs` are already
    # DLS-retained.  Generation side: run the same check against the
    # session's cached layer_means.  Useful sanity context — does
    # generation extraction's discriminative-layer band line up with
    # prefill's?
    layer_means = session.layer_means
    # Build unit-norm directions for the DLS check.
    gen_unit_all = {
        L: v / max(float(v.norm()), 1e-12) for L, v in gen_dirs_all.items()
    }
    gen_keep_all = {
        L for L, ax in compute_dls_axes(
            {L: torch.stack([happy_all[L].reshape(-1), sad_all[L].reshape(-1)])
             for L in happy_all},
            {L: d.reshape(1, -1) for L, d in gen_unit_all.items()},
            layer_means,
        ).items()
        if ax
    }
    prefill_keep = set(prefill_dirs)
    print(
        f"  DLS keep-set: prefill={len(prefill_keep)}  "
        f"gen(mean_all)={len(gen_keep_all)}  "
        f"intersection={len(prefill_keep & gen_keep_all)}"
    )

    # --- per-layer cosine table -------------------------------------------
    cos_all = per_layer_cosine(prefill_dirs, gen_dirs_all)
    cos_early = per_layer_cosine(prefill_dirs, gen_dirs_early)
    cos_internal = per_layer_cosine(gen_dirs_all, gen_dirs_early)

    print("\n=== per-layer cosine vs prefill ===")
    print(
        f"{'layer':>6} {'cos(pre,mean_all)':>20} {'cos(pre,mean_early)':>22} "
        f"{'cos(all,early)':>16} {'||prefill||':>12} {'||gen_all||':>12}"
    )
    print("-" * 96)
    for L in sorted(cos_all):
        cp = float(prefill_dirs[L].norm())
        cg = float(gen_dirs_all[L].norm())
        ce = cos_early.get(L, float("nan"))
        ci = cos_internal.get(L, float("nan"))
        print(
            f"{L:>6} {cos_all[L]:>+20.4f} {ce:>+22.4f} {ci:>+16.4f} "
            f"{cp:>12.3f} {cg:>12.3f}"
        )

    # --- summary -----------------------------------------------------------
    s_all = summarize(cos_all, "prefill vs MEAN_ALL")
    s_early = summarize(cos_early, "prefill vs MEAN_EARLY")
    s_internal = summarize(cos_internal, "MEAN_ALL vs MEAN_EARLY")

    print("\n=== verdicts ===")
    print(f"  MEAN_ALL:    {verdict_for(s_all['mean_abs_cosine'])}")
    print(f"  MEAN_EARLY:  {verdict_for(s_early['mean_abs_cosine'])}")

    early_minus_all = s_early["mean_abs_cosine"] - s_all["mean_abs_cosine"]
    if early_minus_all > 0.1:
        dilution = (
            "MEAN_EARLY tracks prefill notably better than MEAN_ALL — "
            "dilution hypothesis SUPPORTED (persona signal weakens as the "
            "model conditions on its own continuation).  The capped-early-"
            "tokens middle path is the cheap-and-effective option."
        )
    elif early_minus_all < -0.05:
        dilution = (
            "MEAN_ALL tracks prefill better than MEAN_EARLY — dilution "
            "hypothesis CONTRADICTED.  Persona signal accumulates rather "
            "than dilutes, the whole trajectory is the right capture window."
        )
    else:
        dilution = (
            "MEAN_EARLY and MEAN_ALL track prefill comparably — dilution "
            "effect is small; cost-cap on capture window is mostly a "
            "compute optimization."
        )
    print(f"\n  dilution check: {dilution}")

    # --- save raw data -----------------------------------------------------
    with OUT_PATH.open("w") as f:
        json.dump(
            {
                "model": MODEL_ID,
                "concept": "happy.sad",
                "n_scenarios": len(scenarios),
                "prompts_per_scenario": PROMPTS_PER_SCENARIO,
                "max_tokens": MAX_TOKENS,
                "early_tokens": EARLY_TOKENS,
                "happy_system": HAPPY_SYSTEM,
                "sad_system": SAD_SYSTEM,
                "scenarios": scenarios,
                "prompts_by_scenario": prompts_by_scenario,
                "dls": {
                    "prefill_keep": sorted(prefill_keep),
                    "gen_keep_all": sorted(gen_keep_all),
                    "intersection": sorted(prefill_keep & gen_keep_all),
                },
                "per_layer_cosine_mean_all": {
                    str(L): float(v) for L, v in cos_all.items()
                },
                "per_layer_cosine_mean_early": {
                    str(L): float(v) for L, v in cos_early.items()
                },
                "per_layer_cosine_internal": {
                    str(L): float(v) for L, v in cos_internal.items()
                },
                "per_layer_prefill_norm": {
                    str(L): float(prefill_dirs[L].norm()) for L in cos_all
                },
                "per_layer_gen_all_norm": {
                    str(L): float(gen_dirs_all[L].norm()) for L in cos_all
                },
                "per_layer_gen_early_norm": {
                    str(L): float(gen_dirs_early[L].norm()) for L in cos_all
                },
                "summary": {
                    "prefill_vs_mean_all": s_all,
                    "prefill_vs_mean_early": s_early,
                    "mean_all_vs_mean_early": s_internal,
                    "verdict_mean_all": verdict_for(s_all["mean_abs_cosine"]),
                    "verdict_mean_early": verdict_for(s_early["mean_abs_cosine"]),
                    "dilution_check": dilution,
                },
            },
            f,
            indent=2,
        )
    print(f"\nraw data -> {OUT_PATH}")


if __name__ == "__main__":
    main()
