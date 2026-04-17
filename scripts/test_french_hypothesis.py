#!/usr/bin/env python3
"""Test whether ministral's elevated L0 perturbation is English-specific.

Loads ministral-3-8b, translates angry.calm contrastive pairs to French via
the same model, re-extracts contrastive PCA on the French pairs, and compares
L0/L1/L2 perturbation ratios against the existing English extraction.

If the French ratios drop to qwen/gemma levels (~0.04) → language-bias confirmed.
If they stay at ~0.10+ → architectural, language is not the culprit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from saklas.core.model import load_model, get_layers
from saklas.core.vectors import extract_contrastive
from saklas.io.paths import saklas_home, safe_model_id


TRANSLATE_SYSTEM = (
    "You are a precise English-to-French translator. Translate the user's "
    "sentence into natural, fluent French. Output only the translation, with "
    "no preamble, no quotes, no commentary."
)


def _apply_template(tokenizer, msgs):
    out = tokenizer.apply_chat_template(
        msgs, return_tensors="pt", add_generation_prompt=True,
    )
    # Some tokenizers return a BatchEncoding (dict), others a bare tensor.
    if isinstance(out, torch.Tensor):
        return out
    return out["input_ids"] if "input_ids" in out else out.input_ids


def translate_one(model, tokenizer, text: str, device) -> str:
    msgs = [
        {"role": "system", "content": TRANSLATE_SYSTEM},
        {"role": "user", "content": text},
    ]
    try:
        input_ids = _apply_template(tokenizer, msgs).to(device)
    except Exception:
        msgs = [{"role": "user", "content": f"{TRANSLATE_SYSTEM}\n\n{text}"}]
        input_ids = _apply_template(tokenizer, msgs).to(device)

    with torch.inference_mode():
        out = model.generate(
            input_ids,
            max_new_tokens=256,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    reply = tokenizer.decode(out[0][input_ids.shape[-1]:], skip_special_tokens=True).strip()
    # Strip occasional self-echoes
    for prefix in ("French:", "French :", "Français:", "Français :"):
        if reply.startswith(prefix):
            reply = reply[len(prefix):].strip()
    # Strip surrounding quotes
    if len(reply) >= 2 and reply[0] in "\"'«" and reply[-1] in "\"'»":
        reply = reply[1:-1].strip()
    return reply


def load_statements(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def layer_ratios(baked_path: Path, layer_means_path: Path, indices=(0, 1, 2)):
    baked = load_file(str(baked_path))
    lm = load_file(str(layer_means_path))
    out = {}
    for i in indices:
        k = f"layer_{i}"
        if k in baked and k in lm:
            b = baked[k].to(torch.float32).norm().item()
            h = lm[k].to(torch.float32).norm().item()
            out[i] = b / h if h > 0 else float("nan")
        else:
            out[i] = float("nan")
    # Also median + sum across all layers
    all_layers = sorted(int(k.split("_")[-1]) for k in baked)
    ratios = []
    for i in all_layers:
        k = f"layer_{i}"
        if k in lm:
            b = baked[k].to(torch.float32).norm().item()
            h = lm[k].to(torch.float32).norm().item()
            if h > 0:
                ratios.append(b / h)
    ratios.sort()
    out["median"] = ratios[len(ratios)//2] if ratios else float("nan")
    out["sum"] = sum(ratios)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mistralai/Ministral-3-8B-Instruct-2512")
    ap.add_argument("--concept", default="angry.calm",
                    help="Source concept (read from ~/.saklas/vectors/default/<c>/statements.json)")
    ap.add_argument("--dst-name", default=None,
                    help="Local concept name for translated pairs (default: <concept>_fr with dots→_)")
    ap.add_argument("--skip-translate", action="store_true",
                    help="Reuse existing translated statements.json if present")
    args = ap.parse_args()

    src_concept = args.concept
    dst_concept = args.dst_name or src_concept.replace(".", "_") + "_fr"

    home = saklas_home()
    src_dir = home / "vectors" / "default" / src_concept
    dst_dir = home / "vectors" / "local" / dst_concept
    dst_dir.mkdir(parents=True, exist_ok=True)

    src_statements = load_statements(src_dir / "statements.json")
    print(f"[load] {len(src_statements)} English pairs from {src_dir}", file=sys.stderr)

    print(f"[load] {args.model} …", file=sys.stderr)
    model, tokenizer = load_model(args.model, device="auto")
    device = next(model.parameters()).device
    print(f"[load] model on {device}", file=sys.stderr)

    dst_statements_path = dst_dir / "statements.json"
    if args.skip_translate and dst_statements_path.exists():
        print(f"[skip] reusing {dst_statements_path}", file=sys.stderr)
        translated = load_statements(dst_statements_path)
    else:
        translated = []
        for i, pair in enumerate(src_statements):
            pos_fr = translate_one(model, tokenizer, pair["positive"], device)
            neg_fr = translate_one(model, tokenizer, pair["negative"], device)
            translated.append({"positive": pos_fr, "negative": neg_fr})
            print(f"[{i+1:2d}/{len(src_statements)}]", file=sys.stderr)
            print(f"  EN+: {pair['positive'][:90]}", file=sys.stderr)
            print(f"  FR+: {pos_fr[:90]}", file=sys.stderr)
            print(f"  EN-: {pair['negative'][:90]}", file=sys.stderr)
            print(f"  FR-: {neg_fr[:90]}", file=sys.stderr)
        with open(dst_statements_path, "w") as f:
            json.dump(translated, f, indent=2, ensure_ascii=False)
        # Minimal pack.json + copy scenarios so saklas pack ls sees it
        with open(dst_dir / "pack.json", "w") as f:
            json.dump({
                "name": dst_concept,
                "description": f"French translation of {src_concept} (L0 bias test)",
                "version": "1.0.0",
                "license": "AGPL-3.0-or-later",
                "tags": ["test", "french"],
                "recommended_alpha": 0.5,
                "source": "local",
                "files": {},
            }, f, indent=2)
        (dst_dir / "scenarios.json").write_text(
            (src_dir / "scenarios.json").read_text()
        )
        print(f"[save] translations → {dst_statements_path}", file=sys.stderr)

    # Extract contrastive PCA on French pairs, using the same ministral model
    print(f"[extract] contrastive PCA on {len(translated)} French pairs …", file=sys.stderr)
    layers = get_layers(model)
    profile = extract_contrastive(model, tokenizer, translated, layers, device=device)

    safe_mid = safe_model_id(args.model)
    baked_out = dst_dir / f"{safe_mid}.safetensors"
    save_file({f"layer_{i}": t.cpu() for i, t in profile.items()}, str(baked_out))
    print(f"[save] baked → {baked_out}", file=sys.stderr)

    # Compare ratios
    lm_path = home / "models" / safe_mid / "layer_means.safetensors"
    en_baked = home / "vectors" / "default" / src_concept / f"{safe_mid}.safetensors"

    print()
    print(f"=== L0/L1/L2 perturbation ratios on {safe_mid} ===")
    print(f"{'variant':30s}  {'L0':>7s} {'L1':>7s} {'L2':>7s}  {'median':>8s} {'sum':>8s}")
    en = layer_ratios(en_baked, lm_path)
    fr = layer_ratios(baked_out, lm_path)
    print(f"{'English (angry.calm)':30s}  {en[0]:7.4f} {en[1]:7.4f} {en[2]:7.4f}  "
          f"{en['median']:8.4f} {en['sum']:8.4f}")
    print(f"{'French  (' + dst_concept + ')':30s}  {fr[0]:7.4f} {fr[1]:7.4f} {fr[2]:7.4f}  "
          f"{fr['median']:8.4f} {fr['sum']:8.4f}")
    print()
    drop = (en[0] - fr[0]) / en[0] * 100 if en[0] > 0 else 0
    print(f"L0 change: {drop:+.1f}%  (qwen/gemma baseline ≈ 0.03-0.04)")
    if fr[0] < 0.06:
        print("→ French L0 in normal range: language-bias hypothesis supported")
    elif fr[0] > en[0] * 0.8:
        print("→ French L0 unchanged: architectural, language is not the cause")
    else:
        print("→ partial drop: mixed signal, likely both architectural and language contributions")


if __name__ == "__main__":
    main()
