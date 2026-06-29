"""Synthetic ground-truth stress harness for saklas's geometry detector.

The concept_geometry experiments keep finding the same thing: gemma flattens
structured concept geometry (taxonomy -> weak flat tree, circumplex -> flat
dominant axis, even months auto-fits flat). Every one of those conclusions rests
on a premise that has never itself been tested: that ``select_topology`` is
*trustworthy* -- that when it says "flat" the geometry is flat, not that the
detector has a flat-bias and is mislabelling real curvature.

This harness establishes the detector's operating characteristics on synthetic
clouds of KNOWN topology, under the conditions the real fits actually run in
(anisotropic, rogue-channel-dominated activation space whitened by the
Fisher/Mahalanobis metric). It answers four questions:

  confusion   -- does each ground-truth topology get the right label, across
                 whitener anisotropy and seeds? (the headline robustness table)
  flatcurved  -- WHERE is the flat<->curved decision boundary, and does GCV have
                 a flat-bias margin vs the raw reconstruction gain? (the
                 research-critical de-confound of "the model is flat")
  periodic    -- recall/precision of ring & torus detection: arc-extent sweep,
                 faint-ring recall, T2/T3, and false-positive rate on
                 blob/grid/persona-fan/arc (the funky-geometry path)
  stability   -- persistence_frac sensitivity, determinism, K sensitivity

Pure CPU, no model load: ``select_topology`` operates on coordinate Grams, so the
whole envelope is measurable from synthetic centroids. Run from this directory.

    python3 geometry_stress.py confusion
    python3 geometry_stress.py flatcurved
    python3 geometry_stress.py periodic
    python3 geometry_stress.py stability
    python3 geometry_stress.py all --quick

Results print as tables and optionally dump to JSON (--json out.json).
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass

import torch

from saklas.core.manifold import (
    _count_persistent_loops,
    select_topology,
)
from saklas.core.mahalanobis import LayerWhitener

# --------------------------------------------------------------------------- #
# Defaults                                                                     #
# --------------------------------------------------------------------------- #

DEFAULT_LAYERS = list(range(6))
DEFAULT_DIM = 96
DEFAULT_MAX_DIM = 6


# --------------------------------------------------------------------------- #
# Whiteners -- the realistic-anisotropy fidelity piece                         #
# --------------------------------------------------------------------------- #
#
# The existing test fixtures cover isotropic (Sigma ~ I) and mild graded
# anisotropy (<=3x). Neither exercises the condition the whitened/Fisher metric
# was *built* for: a handful of massive-activation (rogue) channels at 50-500x
# the background variance. A robust detector must recover the same topology
# regardless of how big the rogue background is, because the Fisher metric is
# supposed to divide it out. ``make_whitener("rogue", ...)`` builds that, and
# returns the set of CLEAN dims the signal must live in (rogue dims are
# background-only -- placing signal there would be correctly down-weighted, which
# is a different test).


def make_whitener(
    kind: str,
    layers: list[int],
    dim: int,
    *,
    seed: int = 11,
    n: int = 256,
    rogue_count: int = 4,
    rogue_mag: float = 100.0,
) -> tuple[LayerWhitener, list[int]]:
    """Build a per-layer whitener of a given anisotropy ``kind``.

    Returns ``(whitener, clean_dims)`` -- ``clean_dims`` is the index list the
    signal should be lifted into (all dims except the rogue ones; the full range
    for non-rogue kinds).
    """
    means = {L: torch.zeros(dim, dtype=torch.float32) for L in layers}
    acts: dict[int, torch.Tensor] = {}
    clean = list(range(dim))
    rogue: list[int] = []
    if kind == "rogue":
        # Deterministic rogue channel indices spread across the dim range.
        rogue = [int(round(i * (dim - 1) / max(1, rogue_count - 1))) for i in range(rogue_count)]
        rogue = sorted(set(rogue))
        clean = [d for d in range(dim) if d not in rogue]
    for L in layers:
        g = torch.Generator().manual_seed(seed * 7 + L)
        if kind == "iso":
            scale = torch.ones(dim, dtype=torch.float32)
        elif kind == "aniso":
            # Graded anisotropy in [0.5, 1.5] (the existing synthetic_whitener
            # regime), a different draw per layer.
            scale = 0.5 + torch.rand(dim, generator=g, dtype=torch.float32)
        elif kind == "rogue":
            scale = torch.ones(dim, dtype=torch.float32)
            for d in rogue:
                scale[d] = float(rogue_mag)
        else:
            raise ValueError(f"unknown whitener kind {kind!r}")
        X = torch.randn(n, dim, generator=g, dtype=torch.float32) * scale
        acts[L] = X + means[L].reshape(1, dim)
    wh = LayerWhitener.from_neutral_activations(acts, means, ridge_scale=1.0)
    return wh, clean


# --------------------------------------------------------------------------- #
# Topology generators -- low-dim embedded "concept space" coords (K, p)        #
# --------------------------------------------------------------------------- #
#
# Each returns the embedded centroid cloud BEFORE the random per-layer lift into
# activation space; ``truth`` is the ground-truth regime the detector should
# report ("flat" | "curved" | "periodic-T{d}"). "sphere" has no auto target (S^n
# is authored-only) -- it's a documented don't-spuriously-ring case.


def _std(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean()) / x.std().clamp(min=1e-9)


def gen_line(k: int) -> torch.Tensor:
    t = torch.linspace(0, 1, k)
    return torch.stack([t, 2 * t, -0.5 * t], dim=1)


def gen_blob(k: int, seed: int = 0, dim: int = 3) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(k, dim, generator=g)


def gen_plane(k: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(k, 2, generator=g)


def gen_arc(k: int, deg: float) -> torch.Tensor:
    th = torch.linspace(0, math.radians(deg), k)
    return torch.stack([torch.cos(th), torch.sin(th)], dim=1)


def gen_curve_poly(k: int, c: float) -> torch.Tensor:
    """A 1-D curve whose nonlinear (curved) energy is set by ``c``.

    At ``c=0`` the embedding is affine in the intrinsic ``t`` (a straight line ->
    flat). As ``c`` grows the quadratic/cubic terms -- which an affine map cannot
    reproduce -- grow, so the curved RBF fit beats the flat affine fit by more.
    ``c`` is the ratio of nonlinear to linear embedded energy, the clean knob for
    the flat<->curved boundary.
    """
    t = torch.linspace(0, 1, k)
    lin = _std(t)
    quad = _std(t * t - t)
    cub = _std(t ** 3 - 1.5 * t ** 2 + 0.5 * t)
    return torch.stack([lin, c * quad, c * cub], dim=1)


def gen_circle(k: int) -> torch.Tensor:
    th = torch.linspace(0, 2 * math.pi, k + 1)[:-1]
    return torch.stack([torch.cos(th), torch.sin(th)], dim=1)


def gen_ellipse(k: int, a: float = 3.0, b: float = 1.0) -> torch.Tensor:
    th = torch.linspace(0, 2 * math.pi, k + 1)[:-1]
    rot = torch.tensor([[0.8, -0.6], [0.6, 0.8]])
    return torch.stack([a * torch.cos(th), b * torch.sin(th)], dim=1) @ rot


def gen_faint_ring(k: int, mod: float = 0.16, common: int = 8) -> torch.Tensor:
    th = torch.linspace(0, 2 * math.pi, k + 1)[:-1]
    ring = mod * torch.stack([torch.cos(th), torch.sin(th)], dim=1)
    return torch.cat([ring, torch.ones(k, common)], dim=1)


def gen_torus(d: int, side: int) -> torch.Tensor:
    grids = torch.meshgrid(
        *[torch.linspace(0, 2 * math.pi, side + 1)[:-1] for _ in range(d)],
        indexing="ij",
    )
    cols = []
    for g in grids:
        cols.append(torch.cos(g).flatten())
        cols.append(torch.sin(g).flatten())
    return torch.stack(cols, dim=1)


def gen_grid(side: int) -> torch.Tensor:
    g = torch.arange(side, dtype=torch.float32)
    a, b = torch.meshgrid(g, g, indexing="ij")
    return torch.stack([a.flatten(), b.flatten()], dim=1)


def gen_sphere(k: int, seed: int = 3) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(k, 3, generator=g)
    return v / v.norm(dim=1, keepdim=True)


def gen_persona_fan(k: int, rank: int = 8, seed: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(k, rank, generator=g).abs() @ torch.randn(rank, 12, generator=g)


# Shapes outside the detector's practical envelope -- reported, but not flagged as
# leaks. torus-T3: a 3-torus needs many points per loop to keep its holes fat
# enough for PH, but that pushes K past the practical periodic regime; coarse tori
# (side 4-5) fill inside the eps_max=2*eps_c window and read as flat. No bundled
# manifold is a 3-torus. sphere: S^n is authored-only (never auto-selected).
KNOWN_HARD = {"torus-T3", "sphere"}


# The labelled corpus for the confusion matrix. (name, factory, truth-regime.)
def topology_zoo(seed: int) -> list[tuple[str, torch.Tensor, str]]:
    return [
        ("line", gen_line(30), "flat"),
        ("blob", gen_blob(40, seed=seed), "flat"),
        ("plane", gen_plane(40, seed=seed), "flat"),
        ("arc-90", gen_arc(30, 90), "flat"),
        ("curve", gen_curve_poly(40, 1.2), "curved"),
        ("circle", gen_circle(36), "periodic-T1"),
        ("ellipse", gen_ellipse(36), "periodic-T1"),
        ("faint-ring", gen_faint_ring(12), "periodic-T1"),
        ("torus-T2", gen_torus(2, 7), "periodic-T2"),
        ("torus-T3", gen_torus(3, 4), "periodic-T3"),
        ("grid", gen_grid(6), "flat"),
        ("persona-fan", gen_persona_fan(40, seed=seed), "flat"),
        ("sphere", gen_sphere(60, seed=seed), "none"),  # authored-only; don't-ring
    ]


# --------------------------------------------------------------------------- #
# Lift + detect                                                                #
# --------------------------------------------------------------------------- #


def lift(
    low: torch.Tensor,
    layers: list[int],
    dim: int,
    *,
    noise: float,
    seed: int,
    clean_dims: list[int],
) -> dict[int, torch.Tensor]:
    """Random per-layer linear lift of ``low`` into activation space + noise.

    Each layer gets an independent Gaussian projection (different layers encode
    the concept differently), restricted to ``clean_dims`` so the signal is
    whitener-visible (the rogue dims are background-only). Signal columns are
    standardized so ``noise`` is a stable fraction of signal scale; noise is
    isotropic over the full ``dim``.
    """
    k, p = low.shape
    low = low - low.mean(0, keepdim=True)
    # Standardize the embedded cloud to unit average coordinate scale, so the
    # signal amplitude is comparable to the unit clean-background variance and
    # ``noise`` reads as signal-relative.
    low = low / low.std().clamp(min=1e-9)
    clean_t = torch.tensor(clean_dims, dtype=torch.long)
    out: dict[int, torch.Tensor] = {}
    for L in layers:
        g = torch.Generator().manual_seed(1000 * seed + L)
        proj = torch.zeros(p, dim, dtype=torch.float32)
        proj[:, clean_t] = torch.randn(p, len(clean_dims), generator=g) / math.sqrt(p)
        signal = low @ proj
        eps = noise * torch.randn(k, dim, generator=g)
        out[L] = (signal + eps).float()
    return out


@dataclass
class DetectResult:
    regime: str            # "flat" | "curved" | "periodic-T{d}"
    winner: str            # raw winner_name
    gcv_flat: float
    gcv_curved: float
    gcv_torus: float       # inf if no periodic candidate
    dim_flat: int
    dim_curved: int
    dim_torus: int
    sep: float             # median whitened pairwise centroid distance (SNR sanity)


def _cand_score(choice, predicate) -> tuple[float, int]:
    for c in choice.candidates:
        if predicate(c.name) and c.viable:
            return float(c.score), int(c.intrinsic_dim)
    return math.inf, 0


def regime_of(winner: str) -> str:
    if winner == "flat-pca":
        return "flat"
    if winner == "spectral":
        return "curved"
    if winner.startswith("torus-T"):
        return f"periodic-T{winner.split('T')[-1]}"
    return winner


def detect(
    low: torch.Tensor,
    wh: LayerWhitener,
    clean_dims: list[int],
    *,
    layers: list[int],
    dim: int,
    noise: float,
    seed: int,
    max_dim: int = DEFAULT_MAX_DIM,
    persistence_frac: float = 0.5,
) -> DetectResult:
    stacks = lift(low, layers, dim, noise=noise, seed=seed, clean_dims=clean_dims)
    grams = {
        L: wh.subspace_gram(L, stacks[L] - stacks[L].mean(0, keepdim=True))
        for L in layers
    }
    consensus = torch.stack([grams[L] for L in layers]).mean(0)
    choice = select_topology(
        stacks, grams, consensus, whitener=wh,
        max_dim=max_dim, persistence_frac=persistence_frac,
    )
    # whitened separation sanity: median off-diagonal distance from consensus Gram
    cg = 0.5 * (consensus + consensus.t())
    diag = cg.diagonal()
    d2 = (diag.unsqueeze(0) + diag.unsqueeze(1) - 2 * cg).clamp(min=0).sqrt()
    k = d2.shape[0]
    iu = torch.triu_indices(k, k, 1)
    sep = float(d2[iu[0], iu[1]].median())
    gcv_flat, dim_flat = _cand_score(choice, lambda n: n == "flat-pca")
    gcv_curved, dim_curved = _cand_score(choice, lambda n: n == "spectral")
    gcv_torus, dim_torus = _cand_score(choice, lambda n: n.startswith("torus-T"))
    return DetectResult(
        regime=regime_of(choice.winner_name),
        winner=choice.winner_name,
        gcv_flat=gcv_flat, gcv_curved=gcv_curved, gcv_torus=gcv_torus,
        dim_flat=dim_flat, dim_curved=dim_curved, dim_torus=dim_torus,
        sep=sep,
    )


# --------------------------------------------------------------------------- #
# Sweep 1 -- confusion matrix                                                  #
# --------------------------------------------------------------------------- #


def sweep_confusion(args: argparse.Namespace) -> dict[str, object]:
    layers, dim = args.layers, args.dim
    whiteners = ["iso", "aniso", "rogue"]
    seeds = list(range(args.seeds))
    noise = args.noise
    results: dict = {}
    print(f"\n{'=' * 78}\nCONFUSION MATRIX  (noise={noise}, seeds={len(seeds)}, "
          f"dim={dim}, layers={len(layers)})\n{'=' * 78}")
    for wkind in whiteners:
        per_truth: dict[str, Counter] = {}
        for seed in seeds:
            wh, clean = make_whitener(wkind, layers, dim, seed=100 + seed)
            for name, low, truth in topology_zoo(seed):
                r = detect(low, wh, clean, layers=layers, dim=dim,
                           noise=noise, seed=seed)
                per_truth.setdefault(f"{name}|{truth}", Counter())[r.regime] += 1
        results[wkind] = {k: dict(v) for k, v in per_truth.items()}
        print(f"\n--- whitener: {wkind} ---")
        print(f"{'shape':<14}{'truth':<14}{'detected (count)':<40}{'acc':>6}")
        for key, ctr in per_truth.items():
            name, truth = key.split("|")
            total = sum(ctr.values())
            hit = ctr.get(truth, 0)
            acc = hit / total if total else 0.0
            top = ", ".join(f"{k}:{v}" for k, v in ctr.most_common())
            if name in KNOWN_HARD or truth == "none":
                flag = "  (known limit)"
            elif acc >= 0.9:
                flag = ""
            else:
                flag = "  <-- LEAK"
            print(f"{name:<14}{truth:<14}{top:<40}{acc:>6.2f}{flag}")
    return results


# --------------------------------------------------------------------------- #
# Sweep 2 -- flat<->curved decision boundary (the flat-bias map)              #
# --------------------------------------------------------------------------- #


def sweep_flatcurved(args: argparse.Namespace) -> dict[str, object]:
    layers, dim = args.layers, args.dim
    cs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.4, 2.0]
    noises = [0.0, 0.05, 0.1, 0.2, 0.4]
    seeds = list(range(max(3, args.seeds // 4)))
    print(f"\n{'=' * 78}\nFLAT<->CURVED BOUNDARY  (poly-curvature c x noise, "
          f"seeds={len(seeds)})\n{'=' * 78}")
    print("Cell = majority regime; (gf-gc) = mean GCV margin flat-curved "
          "(>0 => curved fits better).\n")
    header = "noise\\c    " + "".join(f"{c:>7}" for c in cs)
    print(header)
    grid: dict = {}
    for noise in noises:
        row_regime = []
        row_margin = []
        for c in cs:
            regs = Counter()
            margins = []
            for seed in seeds:
                wh, clean = make_whitener("aniso", layers, dim, seed=200 + seed)
                low = gen_curve_poly(40, c)
                r = detect(low, wh, clean, layers=layers, dim=dim,
                           noise=noise, seed=seed, max_dim=args.max_dim)
                regs[r.regime] += 1
                if math.isfinite(r.gcv_flat) and math.isfinite(r.gcv_curved):
                    margins.append(r.gcv_flat - r.gcv_curved)
            maj = regs.most_common(1)[0][0]
            row_regime.append(maj)
            row_margin.append(sum(margins) / len(margins) if margins else float("nan"))
            grid[f"n{noise}_c{c}"] = {"regime": maj, "margin": row_margin[-1],
                                      "regimes": dict(regs)}
        sym = {"flat": " flat", "curved": "curv", }
        cells = "".join(f"{sym.get(rr, rr)[:6]:>7}" for rr in row_regime)
        print(f"{noise:<10}{cells}")
    print("\nGCV margin (flat - curved), aniso whitener, noise=0.0 (cleanest):")
    print("c       " + "".join(f"{c:>8}" for c in cs))
    line = "margin  "
    for c in cs:
        m = grid[f"n0.0_c{c}"]["margin"]
        line += f"{m:>8.2f}" if math.isfinite(m) else f"{'nan':>8}"
    print(line)
    print("\nReading: the c where the regime flips flat->curv at low noise is the\n"
          "detector's curvature threshold. A large flip-c (needs strong curvature\n"
          "before admitting it) is flat-bias; margin crossing 0 should align with\n"
          "the flip.")
    return grid


# --------------------------------------------------------------------------- #
# Sweep 3 -- periodic detection (recall + precision)                          #
# --------------------------------------------------------------------------- #


def sweep_periodic(args: argparse.Namespace) -> dict[str, object]:
    layers, dim = args.layers, args.dim
    seeds = list(range(args.seeds))
    out: dict = {}
    print(f"\n{'=' * 78}\nPERIODIC DETECTION  (seeds={len(seeds)})\n{'=' * 78}")

    # 3a. Arc-extent sweep: flat -> curved -> periodic as the arc closes.
    print("\n--- (a) arc-extent sweep (deg): flat -> curved -> periodic-T1 ---")
    degs = [30, 60, 90, 135, 180, 225, 270, 315, 350, 360]
    print(f"{'deg':<8}{'regime (majority over seeds)':<28}{'periodic-rate':>14}")
    arc_res = {}
    for deg in degs:
        regs = Counter()
        for seed in seeds:
            wh, clean = make_whitener("aniso", layers, dim, seed=300 + seed)
            low = gen_circle(36) if deg >= 360 else gen_arc(36, deg)
            r = detect(low, wh, clean, layers=layers, dim=dim, noise=0.08, seed=seed)
            regs[r.regime] += 1
        per = sum(v for k, v in regs.items() if k.startswith("periodic")) / len(seeds)
        arc_res[deg] = dict(regs)
        print(f"{deg:<8}{regs.most_common(1)[0][0]:<28}{per:>14.2f}")
    out["arc"] = arc_res

    # 3b. Faint-ring recall vs modulation amplitude x noise.
    print("\n--- (b) faint-ring recall: periodic-detection rate ---")
    mods = [0.08, 0.12, 0.16, 0.24, 0.35, 0.5]
    noises = [0.0, 0.02, 0.05, 0.1]
    print("mod\\noise " + "".join(f"{n:>8}" for n in noises))
    faint_res = {}
    for mod in mods:
        cells = []
        for noise in noises:
            hit = 0
            for seed in seeds:
                wh, clean = make_whitener("aniso", layers, dim, seed=400 + seed)
                low = gen_faint_ring(12, mod=mod)
                r = detect(low, wh, clean, layers=layers, dim=dim,
                           noise=noise, seed=seed)
                hit += r.regime.startswith("periodic")
            rate = hit / len(seeds)
            cells.append(rate)
            faint_res[f"m{mod}_n{noise}"] = rate
        print(f"{mod:<10}" + "".join(f"{c:>8.2f}" for c in cells))
    out["faint"] = faint_res

    # 3c. Torus dimension recovery + coarseness floor (points-per-loop).
    print("\n--- (c) torus dimension recovery + coarseness floor ---")
    print("    (side = points per loop; a thin hole at small side fills inside "
          "eps_max=2*eps_c and reads flat)")
    tori = [("circle-K36", gen_circle(36), "periodic-T1"),
            ("T2-side5", gen_torus(2, 5), "periodic-T2"),
            ("T2-side6", gen_torus(2, 6), "periodic-T2"),
            ("T2-side7", gen_torus(2, 7), "periodic-T2"),
            ("T3-side4", gen_torus(3, 4), "periodic-T3"),
            ("T3-side5", gen_torus(3, 5), "periodic-T3")]
    torus_res = {}
    for name, low, truth in tori:
        regs = Counter()
        for seed in seeds:
            wh, clean = make_whitener("aniso", layers, dim, seed=500 + seed)
            r = detect(low, wh, clean, layers=layers, dim=dim, noise=0.05, seed=seed)
            regs[r.regime] += 1
        acc = regs.get(truth, 0) / len(seeds)
        torus_res[name] = {"acc": acc, "regimes": dict(regs)}
        note = "  (out of practical envelope)" if "T3" in name else ""
        print(f"{name:<12}truth={truth:<14}acc={acc:>5.2f}  {dict(regs)}{note}")
    out["torus"] = torus_res

    # 3d. Periodic FALSE POSITIVE rate on non-cyclic shapes.
    print("\n--- (d) periodic FALSE-POSITIVE rate (must be ~0) ---")
    fp_shapes = [("blob", lambda s: gen_blob(40, seed=s)),
                 ("plane", lambda s: gen_plane(40, seed=s)),
                 ("grid", lambda s: gen_grid(6)),
                 ("arc-200", lambda s: gen_arc(30, 200)),
                 ("persona-fan", lambda s: gen_persona_fan(40, seed=s)),
                 ("line", lambda s: gen_line(30))]
    fp_res = {}
    for name, fac in fp_shapes:
        fp = 0
        nseed = max(args.seeds, 30)
        for seed in range(nseed):
            wh, clean = make_whitener("aniso", layers, dim, seed=600 + seed)
            low = fac(seed)
            r = detect(low, wh, clean, layers=layers, dim=dim, noise=0.05, seed=seed)
            fp += r.regime.startswith("periodic")
        rate = fp / nseed
        fp_res[name] = rate
        flag = "" if rate <= 0.05 else "  <-- HIGH FP"
        print(f"{name:<14}FP rate = {rate:>5.2f}  ({fp}/{nseed}){flag}")
    out["fp"] = fp_res
    return out


# --------------------------------------------------------------------------- #
# Sweep 4 -- stability (persistence_frac, determinism, K)                      #
# --------------------------------------------------------------------------- #


def sweep_stability(args: argparse.Namespace) -> dict[str, object]:
    layers, dim = args.layers, args.dim
    out: dict = {}
    print(f"\n{'=' * 78}\nSTABILITY\n{'=' * 78}")

    # 4a. persistence_frac sensitivity on borderline cases.
    print("\n--- (a) persistence_frac sensitivity (periodic-rate over seeds) ---")
    fracs = [0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9]
    cases = [("noisy-circle", lambda: gen_circle(36), 0.15),
             ("ellipse-6:1", lambda: gen_ellipse(36, 6.0, 1.0), 0.08),
             ("faint-ring", lambda: gen_faint_ring(12, 0.16), 0.03),
             ("blob(FP-check)", lambda: gen_blob(40, seed=0), 0.05)]
    seeds = list(range(max(5, args.seeds // 2)))
    print(f"{'case':<18}" + "".join(f"pf={f:<5}" for f in fracs))
    pf_res = {}
    for name, fac, noise in cases:
        rates = []
        for pf in fracs:
            hit = 0
            for seed in seeds:
                wh, clean = make_whitener("aniso", layers, dim, seed=700 + seed)
                r = detect(fac(), wh, clean, layers=layers, dim=dim,
                           noise=noise, seed=seed, persistence_frac=pf)
                hit += r.regime.startswith("periodic")
            rates.append(hit / len(seeds))
        pf_res[name] = dict(zip([str(f) for f in fracs], rates))
        print(f"{name:<18}" + "".join(f"{r:<8.2f}" for r in rates))
    out["persistence_frac"] = pf_res

    # 4b. Determinism: repeated identical calls must match bit-for-bit.
    print("\n--- (b) determinism (repeated select_topology identical?) ---")
    det_ok = True
    for name, low, _ in topology_zoo(0):
        wh, clean = make_whitener("aniso", layers, dim, seed=800)
        r1 = detect(low, wh, clean, layers=layers, dim=dim, noise=0.05, seed=0)
        r2 = detect(low, wh, clean, layers=layers, dim=dim, noise=0.05, seed=0)

        def _close(a: float, b: float) -> bool:
            # inf == inf is fine (an unviable candidate); guard the nan it'd make.
            return a == b or (math.isfinite(a) and math.isfinite(b)
                              and abs(a - b) < 1e-9)
        same = (r1.winner == r2.winner
                and _close(r1.gcv_flat, r2.gcv_flat)
                and _close(r1.gcv_curved, r2.gcv_curved))
        det_ok = det_ok and same
        if not same:
            print(f"  NON-DETERMINISTIC: {name}  {r1.winner} vs {r2.winner}")
    print(f"  all {len(topology_zoo(0))} shapes deterministic: {det_ok}")
    out["deterministic"] = det_ok

    # 4c. K sensitivity: smallest K at which a clean circle still rings.
    print("\n--- (c) K sensitivity: clean-circle periodic-rate by node count ---")
    ks = [6, 7, 8, 9, 10, 12, 16, 24, 36]
    k_res = {}
    for k in ks:
        hit = 0
        seeds_k = list(range(max(5, args.seeds // 2)))
        for seed in seeds_k:
            wh, clean = make_whitener("aniso", layers, dim, seed=900 + seed)
            r = detect(gen_circle(k), wh, clean, layers=layers, dim=dim,
                       noise=0.03, seed=seed)
            hit += r.regime.startswith("periodic")
        k_res[k] = hit / len(seeds_k)
        print(f"  K={k:<4} periodic-rate={k_res[k]:.2f}")
    out["k_sensitivity"] = k_res

    # 4d. Dimension-bias demonstration: raw loop count grows with spectral dim
    #     budget, but the decoupled selector doesn't crown the highest dim.
    print("\n--- (d) dimension-bias guard: PH loop count is dim-stable ---")
    pts = gen_circle(40)
    D = torch.cdist(pts, pts)
    counts = {md: _count_persistent_loops(D, max_dim=md) for md in [1, 2, 4, 8]}
    print(f"  circle loop count vs max_dim cap: {counts}  (want all == 1)")
    out["dim_bias_loopcount"] = counts
    return out


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

SWEEPS = {
    "confusion": sweep_confusion,
    "flatcurved": sweep_flatcurved,
    "periodic": sweep_periodic,
    "stability": sweep_stability,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep", choices=[*SWEEPS, "all"])
    ap.add_argument("--seeds", type=int, default=24)
    ap.add_argument("--dim", type=int, default=DEFAULT_DIM)
    ap.add_argument("--layers", type=int, default=len(DEFAULT_LAYERS))
    ap.add_argument("--max-dim", type=int, default=DEFAULT_MAX_DIM)
    ap.add_argument("--noise", type=float, default=0.1,
                    help="confusion-matrix noise level")
    ap.add_argument("--quick", action="store_true", help="fewer seeds")
    ap.add_argument("--json", type=str, default="")
    args = ap.parse_args()
    if args.quick:
        args.seeds = max(6, args.seeds // 4)
    args.layers = list(range(args.layers))

    torch.manual_seed(0)
    results: dict = {}
    todo = list(SWEEPS) if args.sweep == "all" else [args.sweep]
    for name in todo:
        results[name] = SWEEPS[name](args)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
