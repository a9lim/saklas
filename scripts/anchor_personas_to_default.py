"""Anchor the persona manifold's origin to gemma's default-mode behavior.

Pipeline:
  1. Generate a 45-statement `default` corpus by feeding gemma 9 bare
     conversational openers (thematically matched to the persona manifold's
     reverse-engineered scenarios) and sentence-splitting the responses.
  2. Pool the per-layer centroid of the `default` corpus using the same
     input-position pooling that produced the existing 100 centroids
     (`compute_node_centroid`).
  3. Approximate augmentation — project default's centroid onto the existing
     per-layer PCA bases, add it as the 101st node, translate all authoring
     coords so default sits at (0,...,0), refit RBF interpolants per layer.
     Skips re-running PCA on the augmented centroid set; adding 1 of 101 is a
     small basis shift, and the experimental question (does default-mode now
     land near origin?) is robust to it.
  4. Save the anchored manifold to ~/.saklas/manifolds/local/personas-anchored/
     so the bundled state isn't clobbered.
  5. Re-run the default-mode experiment against the anchored manifold to
     verify the origin is now coincident with default-mode behavior.
"""

from __future__ import annotations

import json
import re
import shutil
from collections import Counter
from pathlib import Path

import torch

from saklas import SamplingConfig, SaklasSession
from saklas.core.manifold import (
    LayerSubspace,
    Manifold,
    compute_node_centroid,
    eval_rbf,
    fit_rbf_interpolant,
    load_manifold,
    save_manifold,
)


MODEL_ID = "google/gemma-4-31b-it"
SAFE_MODEL = "google__gemma-4-31b-it"
SOURCE_DIR = Path.home() / ".saklas/manifolds/default/personas"
FORK_DIR = Path.home() / ".saklas/manifolds/local/personas-anchored"
PROMPT = "Explain how photosynthesis works in simple terms."
MAX_TOKENS_RESPONSE = 220
MAX_TOKENS_VERIFY = 80
SEED = 42
N_STATEMENTS_PER_OPENER = 5
TARGET_STATEMENTS = 45

# Bare conversational openers, one per scenario the existing 100 corpora were
# generated against. The scenarios were reverse-engineered from statement
# blocks in the bundled `personas` corpus (5 statements per block, 9 blocks
# per node, indices align across all 100 nodes).
DEFAULT_OPENERS = [
    ("disaster",   "i think there's an earthquake happening, what do i do"),
    ("celestial",  "i just saw something really weird in the sky tonight"),
    ("dispute",    "my neighbor and i are arguing about where the property line is"),
    ("portal",     "this is going to sound crazy but there's something glowing in my basement"),
    ("wilderness", "i'm pretty sure i'm lost on this hiking trail, what should i do"),
    ("silence",    "are you still there? you went quiet"),
    ("threat",     "i think someone has been following me on my walk home"),
    ("heirloom",   "how do i take care of an antique my grandmother left me"),
    ("scarcity",   "i'm running low on money before my next paycheck, any tips"),
]


# =============================================================== sentences ==

def split_sentences(text: str) -> list[str]:
    """Pull self-contained sentences out of a (possibly markdown-flecked) response."""
    # Strip code blocks (they're not sentences in the register we want)
    text = re.sub(r"```[\s\S]*?```", " ", text)
    # Strip leading list-item markers per line.
    text = re.sub(r"(?m)^\s*[-*+•]\s+", "", text)
    text = re.sub(r"(?m)^\s*\d+\.\s+", "", text)
    # Strip markdown emphasis markers.
    text = re.sub(r"[*_]{1,3}", "", text)
    # Flatten newlines (sentence boundaries are punctuation, not line breaks).
    text = re.sub(r"\s*\n+\s*", " ", text)
    # Split on '.', '?', '!' followed by whitespace.
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    out: list[str] = []
    for p in parts:
        p = p.strip()
        if len(p) < 15:
            continue  # too short — fragments, list-marker leftovers
        if not p[0].isalpha() and not p[0].isdigit():
            continue  # punctuation-only artifact
        out.append(p)
    return out


def generate_default_corpus(session: SaklasSession) -> list[str]:
    """9 bare openers -> 45 first-person default-voice sentences."""
    sentences: list[str] = []
    for tag, opener in DEFAULT_OPENERS:
        print(f"  [{tag:>10}] {opener!r}")
        runset = session.generate(
            opener,
            thinking=False,
            sampling=SamplingConfig(
                max_tokens=MAX_TOKENS_RESPONSE,
                temperature=0.7,
                top_p=0.95,
                seed=SEED,
            ),
        )
        response = runset.first.text.strip()
        sents = split_sentences(response)
        print(f"             -> {len(sents)} sentences extracted")
        if len(sents) >= N_STATEMENTS_PER_OPENER:
            sents = sents[:N_STATEMENTS_PER_OPENER]
        elif sents:
            # Pad with the last sentence so the row stays at 5
            # (matches the existing pipeline's short-cell padding).
            sents = sents + [sents[-1]] * (N_STATEMENTS_PER_OPENER - len(sents))
        else:
            # Fallback: emit the raw response as one statement repeated.
            placeholder = response[:200] or "I am here to help."
            sents = [placeholder] * N_STATEMENTS_PER_OPENER
        sentences.extend(sents)
    assert len(sentences) == TARGET_STATEMENTS, (
        f"expected {TARGET_STATEMENTS} statements, got {len(sentences)}"
    )
    return sentences


# ================================================================== forking ==

def fork_manifold_folder(src: Path, dst: Path) -> None:
    """Copy the source manifold folder (corpus + sidecars) to dst."""
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    # Copy manifold.json + nodes/, skip per-model safetensors (we'll write our own).
    shutil.copy2(src / "manifold.json", dst / "manifold.json")
    (dst / "nodes").mkdir()
    for f in (src / "nodes").iterdir():
        shutil.copy2(f, dst / "nodes" / f.name)


def append_default_node(fork_dir: Path, statements: list[str]) -> None:
    """Update manifold.json + write nodes/100_default.json in the fork."""
    mj_path = fork_dir / "manifold.json"
    manifest = json.loads(mj_path.read_text())
    # Rename the manifold so it doesn't collide in selector resolution.
    manifest["name"] = "personas_anchored"
    manifest["description"] = (
        manifest.get("description", "")
        + " ANCHORED: includes a `default` node generated from gemma's "
        "responses to 9 bare conversational openers; post-hoc translated "
        "so default sits at (0,...,0) in authoring-coord space."
    )
    # The bundled discover-mode nodes list is `[{label: ...}, ...]`.
    manifest["nodes"].append({"label": "default"})
    # Files manifest cleared — the fit will repopulate (or our save will).
    manifest["files"] = {}
    mj_path.write_text(json.dumps(manifest, indent=2))
    # Write the corpus.
    node_path = fork_dir / "nodes" / "100_default.json"
    node_path.write_text(json.dumps(statements, indent=2))


# ==================================================================== fit ==

def compute_default_centroid_per_layer(
    session: SaklasSession, statements: list[str],
) -> dict[int, torch.Tensor]:
    """Pool centroid_default at every layer using the manifold-pipeline pooling."""
    return compute_node_centroid(
        session._model, session._tokenizer, session._layers,
        session._device, statements,
        role=None, model_type=session._model_info.get("model_type"),
    )


def augment_manifold(
    manifold: Manifold,
    default_centroids: dict[int, torch.Tensor],
    reference_layer: int = 30,
) -> Manifold:
    """Add default centroid as a 101st node + translate origin to it.

    Approximate: keeps the existing per-layer PCA bases unchanged. The default
    centroid's authoring coord is its projection onto the reference layer's
    existing 8-D basis. Translation shifts all node coords so default sits at
    (0,...,0). Per-layer RBFs are refit on the translated coord system.

    Why this is approximate: a true refit would re-run PCA on 101 centroids,
    yielding bases that differ from the 100-centroid bases by O(1/100). For
    the experimental question ("does default-mode now land near origin?")
    this approximation is robust — the verdict shifts on the translation, not
    on basis-shift artifacts.
    """
    sub_ref = manifold.layers[reference_layer]
    device, dtype = sub_ref.mean.device, sub_ref.mean.dtype

    # Default's authoring coord = projection at reference layer onto the
    # first `intrinsic_dim` components of the existing PCA basis. The
    # discover-PCA at the reference layer IS the leading prefix of the
    # per-layer PCA basis at that layer (both are SVD on the same centered
    # matrix; one keeps R=64 components, the other picked k=8). So
    # `sub_ref.basis[:intrinsic_dim]` recovers the discover basis exactly.
    discover_dim = manifold.domain.intrinsic_dim
    centroid_ref = default_centroids[reference_layer].to(device=device, dtype=dtype)
    default_authoring = (centroid_ref - sub_ref.mean) @ sub_ref.basis[:discover_dim].T  # (k,)
    print(f"  default's authoring coord (before translate): "
          f"{default_authoring.cpu().tolist()}")
    print(f"  ||default_authoring|| = {torch.linalg.vector_norm(default_authoring):.4f}")

    # Translate all coords so default sits at origin.
    shift = default_authoring.clone()
    existing_coords = manifold.node_coords.to(device=device, dtype=dtype)
    new_existing = existing_coords - shift                          # (100, R)
    new_default = torch.zeros_like(default_authoring)               # (R,)
    new_node_coords = torch.cat(
        [new_existing, new_default.unsqueeze(0)], dim=0,
    )  # (101, R)

    # For each layer: refit RBF in the translated, augmented coord system.
    new_layers: dict[int, LayerSubspace] = {}
    K_existing = existing_coords.shape[0]
    for layer_idx, sub in manifold.layers.items():
        # Recover existing nodes' per-layer PCA values via the existing RBF.
        old_pca_values_list: list[torch.Tensor] = []
        for k in range(K_existing):
            embedded = manifold.domain.embed(
                manifold.domain.clamp_position(
                    existing_coords[k].to(device=sub.mean.device, dtype=sub.mean.dtype),
                ),
            )
            normalized = (embedded - sub.coord_offset) / sub.coord_scale
            val = eval_rbf(
                sub.node_params, sub.rbf_weights, sub.poly_coeffs,
                normalized.unsqueeze(0),
            ).squeeze(0)
            old_pca_values_list.append(val)
        old_pca_values = torch.stack(old_pca_values_list, dim=0)  # (100, R)

        # Default's per-layer PCA value.
        centroid_l = default_centroids[layer_idx].to(device=sub.mean.device, dtype=sub.mean.dtype)
        default_pca_l = (centroid_l - sub.mean) @ sub.basis.T  # (R,)
        all_pca_values = torch.cat(
            [old_pca_values, default_pca_l.unsqueeze(0)], dim=0,
        )  # (101, R)

        # New normalization in translated coord system.
        new_embedded = manifold.domain.embed(
            manifold.domain.clamp_position(new_node_coords.to(device=sub.mean.device, dtype=sub.mean.dtype)),
        )  # CustomDomain identity, so == new_node_coords
        new_offset = new_embedded.min(dim=0).values
        new_scale = (new_embedded.max(dim=0).values - new_offset).clamp(min=1e-9)
        new_normalized = (new_embedded - new_offset) / new_scale

        # RBF fit goes via CPU — fit_rbf_interpolant runs a matrix_rank
        # poisedness check that calls SVD, which is unimplemented on MPS.
        # The math is tiny (101x8 coord matrix, 101x64 values), CPU is fine.
        new_rbf_weights_cpu, new_poly_coeffs_cpu = fit_rbf_interpolant(
            new_normalized.cpu(), all_pca_values.cpu(),
        )
        target_device = sub.mean.device
        target_dtype = sub.mean.dtype
        new_layers[layer_idx] = LayerSubspace(
            mean=sub.mean,
            basis=sub.basis,
            node_params=new_normalized.to(device=target_device, dtype=target_dtype),
            rbf_weights=new_rbf_weights_cpu.to(device=target_device, dtype=target_dtype),
            poly_coeffs=new_poly_coeffs_cpu.to(device=target_device, dtype=target_dtype),
            coord_offset=new_offset.to(device=target_device, dtype=target_dtype),
            coord_scale=new_scale.to(device=target_device, dtype=target_dtype),
        )

    new_node_roles = (
        list(manifold.node_roles) + [None]
        if manifold.node_roles
        else []
    )

    return Manifold(
        name="personas_anchored",
        domain=manifold.domain,
        node_labels=list(manifold.node_labels) + ["default"],
        node_coords=new_node_coords.to(manifold.node_coords.dtype),
        layers=new_layers,
        feature_space=manifold.feature_space,
        metadata={
            **manifold.metadata,
            "anchored_to": "default",
            "augmentation": "approximate_posthoc",
            "reference_layer": reference_layer,
        },
        node_roles=new_node_roles,
    )


# =============================================================== experiment ==

def persona_pca_coords_per_layer(manifold: Manifold) -> dict[int, torch.Tensor]:
    """Each persona node's PCA-coords, per layer, via RBF eval at its own coord."""
    out: dict[int, torch.Tensor] = {}
    K = manifold.node_coords.shape[0]
    for layer_idx, sub in manifold.layers.items():
        rows = []
        for k in range(K):
            authoring = manifold.node_coords[k].to(device=sub.mean.device, dtype=sub.mean.dtype)
            embedded = manifold.domain.embed(manifold.domain.clamp_position(authoring))
            normalized = (embedded - sub.coord_offset) / sub.coord_scale
            c = eval_rbf(
                sub.node_params, sub.rbf_weights, sub.poly_coeffs,
                normalized.unsqueeze(0),
            ).squeeze(0)
            rows.append(c)
        out[layer_idx] = torch.stack(rows, dim=0)
    return out


def run_experiment(
    session: SaklasSession, manifold: Manifold, label: str,
) -> dict:
    """Where does default-mode land on the manifold?"""
    runset = session.generate(
        PROMPT,
        thinking=False,
        sampling=SamplingConfig(
            max_tokens=MAX_TOKENS_VERIFY,
            temperature=0.7,
            top_p=0.95,
            seed=SEED,
            return_hidden=True,
        ),
    )
    result = runset.first
    print(f"  [{label}] response head: {result.text[:120]!r}")
    hidden = result.hidden_states
    assert hidden is not None

    personas = persona_pca_coords_per_layer(manifold)
    per_layer: dict[int, dict] = {}
    for layer_idx, sub in manifold.layers.items():
        if layer_idx not in hidden:
            continue
        h = hidden[layer_idx].to(device=sub.mean.device, dtype=torch.float32)
        h_mean = h.mean(dim=0)
        c_default = (h_mean - sub.mean) @ sub.basis.T  # (R,)
        c_personas = personas[layer_idx]                # (K, R)

        dist_to_origin = float(torch.linalg.vector_norm(c_default))
        persona_norms = torch.linalg.vector_norm(c_personas, dim=-1)
        median_persona_norm = float(persona_norms.median())
        ratio_med = dist_to_origin / max(median_persona_norm, 1e-9)

        dists = torch.linalg.vector_norm(c_default - c_personas, dim=-1)
        nearest_idx = int(torch.argmin(dists))
        nearest_label = manifold.node_labels[nearest_idx]

        per_layer[layer_idx] = {
            "dist_to_origin": dist_to_origin,
            "median_persona_norm": median_persona_norm,
            "ratio_to_median_persona": ratio_med,
            "nearest_persona": nearest_label,
            "nearest_persona_dist": float(dists[nearest_idx]),
        }

    nearest_counter = Counter(r["nearest_persona"] for r in per_layer.values())
    ratios = [r["ratio_to_median_persona"] for r in per_layer.values()]
    mean_ratio = sum(ratios) / len(ratios)

    print(
        f"\n  [{label}] mean ratio (||default|| / median ||persona||): "
        f"{mean_ratio:.3f}"
    )
    print(f"  [{label}] most-frequent nearest personas:")
    for name, count in nearest_counter.most_common(5):
        pct = 100 * count / len(per_layer)
        print(f"      {name:>20}  {count:>3}/{len(per_layer)} ({pct:.1f}%)")

    return {
        "label": label,
        "mean_ratio": mean_ratio,
        "nearest_counter": dict(nearest_counter.most_common()),
        "per_layer": {str(k): v for k, v in per_layer.items()},
    }


# =================================================================== main ==

def main() -> None:
    print(f"==> loading model {MODEL_ID}")
    session = SaklasSession.from_pretrained(MODEL_ID, device="auto")

    # ----- phase 1: generate default corpus -----------------------------
    print("\n==> phase 1: generating default corpus")
    statements = generate_default_corpus(session)
    print(f"   collected {len(statements)} statements, sample:")
    for s in statements[:3]:
        print(f"     - {s[:100]}")

    # ----- phase 2: fork folder + write corpus --------------------------
    print(f"\n==> phase 2: forking manifold to {FORK_DIR}")
    fork_manifold_folder(SOURCE_DIR, FORK_DIR)
    append_default_node(FORK_DIR, statements)

    # ----- phase 3: pool default centroid + augment manifold -----------
    print("\n==> phase 3: pooling default centroid (45 forward passes)")
    default_centroids = compute_default_centroid_per_layer(session, statements)
    print(f"   pooled {len(default_centroids)} layers, "
          f"sample shape: {next(iter(default_centroids.values())).shape}")

    print("\n==> loading source manifold + augmenting")
    source_manifold = load_manifold(SOURCE_DIR / f"{SAFE_MODEL}.safetensors")
    source_manifold = source_manifold.to(
        device=session._device, dtype=torch.float32,
    )
    anchored = augment_manifold(
        source_manifold, default_centroids,
        reference_layer=source_manifold.metadata.get("reference_layer", 30) if isinstance(source_manifold.metadata, dict) else 30,
    )

    # ----- phase 4: save the anchored manifold --------------------------
    print(f"\n==> phase 4: saving anchored manifold to {FORK_DIR}")
    out_path = FORK_DIR / f"{SAFE_MODEL}.safetensors"
    save_manifold(
        anchored, out_path,
        metadata={
            "method": "manifold_discover_pca_anchored",
            "fit_mode": "pca",
            "anchored_to": "default",
            "augmentation": "approximate_posthoc",
            "reference_layer": 30,
        },
    )

    # ----- phase 5: re-run experiment -----------------------------------
    print("\n==> phase 5: comparing original vs anchored")
    print("  -- original manifold (default-mode lands at offset 3.41x):")
    before = run_experiment(session, source_manifold, "BEFORE")

    print("\n  -- anchored manifold:")
    after = run_experiment(session, anchored, "AFTER")

    delta = after["mean_ratio"] - before["mean_ratio"]
    print(
        f"\n==> verdict: mean ratio "
        f"{before['mean_ratio']:.3f} -> {after['mean_ratio']:.3f} "
        f"(Δ {delta:+.3f})"
    )
    if after["mean_ratio"] < 0.5:
        print("   anchoring succeeded — default-mode now lives NEAR origin")
    elif after["mean_ratio"] < before["mean_ratio"] * 0.5:
        print("   anchoring helped substantially but still off-origin")
    elif after["mean_ratio"] < before["mean_ratio"]:
        print("   anchoring helped, but not as much as hoped")
    else:
        print("   anchoring did NOT reduce the offset — investigate")

    # Save raw side-by-side data.
    out = Path("/tmp/persona_anchor_comparison.json")
    out.write_text(json.dumps({"before": before, "after": after}, indent=2))
    print(f"\n   raw data -> {out}")


if __name__ == "__main__":
    main()
