"""Where does default-mode behavior land on the persona manifold?

For each layer the bundled `personas` manifold covers, project the model's
default-mode activations (during a stock assistant response) onto the
manifold's per-layer PCA subspace and report:

  - ||c_default||                 distance from "centroid of personas"
                                  (== PCA origin by construction)
  - typical persona ||c_i||       median norm of the persona cloud
  - ratio                         how many persona-norms out is default-mode?
  - nearest persona               which persona is closest to default-mode
  - top-5 nearest                 sanity check on cluster membership

If default-mode lands near the origin: the current manifold's mean-of-personas
already coincides with the model's natural default — concern is theoretical.
If default-mode lands consistently off-origin: that's the empirical mandate
for pinning the origin (option 1 / 2 from the conversation).

Reads the pre-fit manifold tensor for google/gemma-4-31b-it; loads that
model; generates a stock assistant response with return_hidden=True and
seed=42; runs the projection in pure tensor math.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from saklas import SamplingConfig, SaklasSession
from saklas.core.manifold import eval_rbf, load_manifold


MODEL_ID = "google/gemma-4-31b-it"
MANIFOLD_PATH = Path(
    "/Users/a9lim/.saklas/manifolds/default/personas/"
    "google__gemma-4-31b-it.safetensors"
)
PROMPT = "Explain how photosynthesis works in simple terms."
MAX_TOKENS = 80
SEED = 42
OUT_PATH = Path("/tmp/persona_default_mode.json")


def project_to_coords(centroid: torch.Tensor, sub) -> torch.Tensor:
    """Activation -> PCA-subspace coordinates (R,)."""
    return (centroid - sub.mean) @ sub.basis.T


def persona_pca_coords(manifold, sub) -> torch.Tensor:
    """Each persona node's centroid mapped to (K, R) via the RBF.

    The RBF was fit to interpolate the per-layer PCA-coords of the
    persona centroids at the node authoring coords. Evaluating the RBF
    at each persona's own authoring coord recovers those PCA-coords
    (modulo RBF interpolation error, which is zero at the nodes
    themselves since the interpolant is exact at fit points).
    """
    K = manifold.node_coords.shape[0]
    coords_list = []
    for k in range(K):
        authoring = manifold.node_coords[k].to(sub.mean.device, sub.mean.dtype)
        embedded = manifold.domain.embed(manifold.domain.clamp_position(authoring))
        normalized = (embedded - sub.coord_offset) / sub.coord_scale
        c = eval_rbf(
            sub.node_params, sub.rbf_weights, sub.poly_coeffs,
            normalized.unsqueeze(0),
        ).squeeze(0)
        coords_list.append(c)
    return torch.stack(coords_list, dim=0)  # (K, R)


def main() -> None:
    print(f"loading manifold from {MANIFOLD_PATH}")
    manifold = load_manifold(MANIFOLD_PATH)
    print(
        f"  manifold: {manifold.name}  "
        f"K={len(manifold.node_labels)} nodes  "
        f"layers={len(manifold.layers)}  "
        f"R (PCA rank, sample layer)="
        f"{next(iter(manifold.layers.values())).rank}"
    )

    print(f"loading model {MODEL_ID}")
    session = SaklasSession.from_pretrained(MODEL_ID, device="auto")

    # Promote manifold to the session's device/dtype.
    manifold = manifold.to(device=session._device, dtype=torch.float32)

    print(f"generating: {PROMPT!r}")
    runset = session.generate(
        PROMPT,
        sampling=SamplingConfig(
            max_tokens=MAX_TOKENS,
            temperature=0.7,
            top_p=0.95,
            seed=SEED,
            return_hidden=True,
        ),
    )
    result = runset.first
    print(f"  response head: {result.text[:160]!r}")
    print(f"  generated {len(result.tokens)} tokens")

    hidden = result.hidden_states  # dict[int, (T, D)] CPU fp32 by convention
    if hidden is None:
        raise RuntimeError("hidden_states is None — return_hidden=True wasn't honored")
    print(f"  captured hidden layers: {len(hidden)}")

    # --- analysis ---------------------------------------------------------

    per_layer: dict[int, dict] = {}
    for layer_idx, sub in manifold.layers.items():
        if layer_idx not in hidden:
            continue
        h = hidden[layer_idx].to(device=sub.mean.device, dtype=torch.float32)
        # Mean over decode positions (the default-mode trajectory's center).
        h_mean = h.mean(dim=0)  # (D,)
        c_default = project_to_coords(h_mean, sub)  # (R,)
        c_personas = persona_pca_coords(manifold, sub)  # (K, R)

        dist_to_origin = float(torch.linalg.vector_norm(c_default))
        persona_norms = torch.linalg.vector_norm(c_personas, dim=-1)
        median_persona_norm = float(persona_norms.median())
        max_persona_norm = float(persona_norms.max())
        # How far past the persona cloud is default-mode?
        ratio_median = dist_to_origin / max(median_persona_norm, 1e-9)
        ratio_max = dist_to_origin / max(max_persona_norm, 1e-9)

        dists = torch.linalg.vector_norm(c_default - c_personas, dim=-1)  # (K,)
        order = torch.argsort(dists)
        nearest_idx = int(order[0])
        nearest_label = manifold.node_labels[nearest_idx]
        # Cosine to nearest persona direction (from origin).
        if persona_norms[nearest_idx] > 0:
            cos_to_nearest = float(
                (c_default @ c_personas[nearest_idx]) /
                (dist_to_origin * persona_norms[nearest_idx] + 1e-12)
            )
        else:
            cos_to_nearest = 0.0

        per_layer[layer_idx] = {
            "dist_to_origin": dist_to_origin,
            "median_persona_norm": median_persona_norm,
            "max_persona_norm": max_persona_norm,
            "ratio_to_median_persona": ratio_median,
            "ratio_to_max_persona": ratio_max,
            "nearest_persona": nearest_label,
            "nearest_persona_dist": float(dists[nearest_idx]),
            "cos_to_nearest_from_origin": cos_to_nearest,
            "top5_nearest": [
                (manifold.node_labels[int(i)], float(dists[int(i)]))
                for i in order[:5]
            ],
        }

    # --- pretty-print summary ---------------------------------------------

    print(
        "\n========== where does default-mode land "
        "on the persona manifold? ==========\n"
    )
    print(
        f"{'layer':>5}  {'||def||':>9}  {'median_||p||':>13}  "
        f"{'max_||p||':>11}  {'ratio_med':>9}  {'cos_near':>9}  "
        f"{'nearest':>18}  {'dist':>9}"
    )
    print("-" * 100)
    for L in sorted(per_layer):
        r = per_layer[L]
        print(
            f"{L:>5}  {r['dist_to_origin']:>9.3f}  "
            f"{r['median_persona_norm']:>13.3f}  "
            f"{r['max_persona_norm']:>11.3f}  "
            f"{r['ratio_to_median_persona']:>9.2f}  "
            f"{r['cos_to_nearest_from_origin']:>+9.3f}  "
            f"{r['nearest_persona']:>18}  "
            f"{r['nearest_persona_dist']:>9.3f}"
        )

    # Aggregate readout: what's the most-frequent nearest persona, and how
    # consistent is the offset direction across layers?
    print("\n=== nearest-persona consistency across layers ===")
    from collections import Counter
    nearest_counter = Counter(r["nearest_persona"] for r in per_layer.values())
    for label, count in nearest_counter.most_common(10):
        pct = 100 * count / len(per_layer)
        print(f"  {label:>20}  {count:>3} layers ({pct:>5.1f}%)")

    # Mean ratio across layers — is default-mode systematically off-origin?
    median_ratios = [r["ratio_to_median_persona"] for r in per_layer.values()]
    mean_ratio = sum(median_ratios) / len(median_ratios)
    print(
        f"\n  mean ratio (||default|| / median ||persona||) "
        f"across {len(per_layer)} layers: {mean_ratio:.3f}"
    )
    if mean_ratio < 0.3:
        verdict = "default-mode lives NEAR the manifold origin — current origin is principled"
    elif mean_ratio < 0.8:
        verdict = "default-mode is offset from origin but inside the persona cloud — origin somewhat principled"
    else:
        verdict = "default-mode is OFFSET from origin, on the order of a typical persona — empirical mandate to pin"
    print(f"  verdict: {verdict}")

    # Save raw data
    with OUT_PATH.open("w") as f:
        json.dump(
            {
                "model": MODEL_ID,
                "prompt": PROMPT,
                "max_tokens": MAX_TOKENS,
                "seed": SEED,
                "response": result.text,
                "n_tokens": len(result.tokens),
                "per_layer": {str(k): v for k, v in per_layer.items()},
                "summary": {
                    "mean_ratio": mean_ratio,
                    "nearest_counter": dict(nearest_counter.most_common()),
                    "verdict": verdict,
                },
            },
            f,
            indent=2,
        )
    print(f"\nraw data -> {OUT_PATH}")


if __name__ == "__main__":
    main()
