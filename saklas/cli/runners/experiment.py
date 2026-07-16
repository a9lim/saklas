"""``saklas experiment <verb>`` runners (fan / transcript / naturalness)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import saklas.cli.runners as _pkg
from saklas.cli.parsers import _EXPERIMENT_VERBS
from saklas.cli.runners.shared import _saklas_error_exit


@_saklas_error_exit
def _run_experiment(args: argparse.Namespace) -> None:
    """Dispatch ``saklas experiment <verb>``."""
    cmd = getattr(args, "experiment_cmd", None)
    if cmd is None:
        print("usage: saklas experiment <verb> [...]")
        print()
        width = max(len(v) for v, _ in _EXPERIMENT_VERBS)
        for v, desc in _EXPERIMENT_VERBS:
            print(f"  {v:<{width}}  {desc}")
        sys.exit(0)
    if cmd == "fan":
        _run_experiment_fan(args)
        return
    if cmd == "transcript":
        _run_experiment_transcript(args)
        return
    if cmd == "naturalness":
        _run_experiment_naturalness(args)
        return
    print(f"unknown experiment verb {cmd!r}", file=sys.stderr)
    sys.exit(2)


def _run_experiment_naturalness(args: argparse.Namespace) -> None:
    """Score a steered generation's behavior-manifold naturalness."""
    import json as _json

    import torch

    from saklas.core.manifold import domain_from_spec
    from saklas.core.naturalness import (
        compute_node_behavior_centroid,
        compute_trajectory_distributions,
        fit_behavior_manifold,
        trajectory_naturalness,
    )
    from saklas.core.profile import Profile
    from saklas.core.sampling import SamplingConfig
    from saklas.core.steering_expr import ManifoldTerm, parse_expr
    from saklas.io.manifolds import ManifoldFolder

    _pkg._load_effective_config(args)
    mfolder = Path(args.manifold)
    if not (mfolder / "manifold.json").exists():
        print(
            f"experiment naturalness: no manifold.json in {mfolder}",
            file=sys.stderr,
        )
        sys.exit(2)
    # Naturalness consumes only authoring geometry + node corpus; fitted
    # payload integrity is irrelevant to this independent behavior-space fit.
    mf = ManifoldFolder.load(mfolder, verify_manifest=False)
    node_groups = mf.node_groups()
    domain = domain_from_spec(mf.domain)
    node_coords = torch.tensor(mf.node_coords, dtype=torch.float32)
    node_params = domain.embed(node_coords)

    _pkg._print_startup(args)
    session = _pkg._make_session(args)
    _pkg._print_model_info(session)

    # 1. Fit the behavior manifold from the node corpus — each node's
    #    mean next-token distribution, in Hellinger space.
    print(f"fitting behavior manifold ({len(node_groups)} nodes)...")
    centroids = [
        compute_node_behavior_centroid(
            session.model, session.tokenizer, session.device, statements,
        )
        for _label, statements in node_groups
    ]
    behavior = fit_behavior_manifold(torch.stack(centroids), node_params)

    sampling = SamplingConfig(max_tokens=args.max_tokens, seed=0)

    def _score(steer: str | None) -> tuple[str, float, float]:
        result = session.generate(
            args.prompt, steering=steer, sampling=sampling,
        ).first
        text = result.text
        traj = compute_trajectory_distributions(
            session.model, session.tokenizer, session.device, text,
        )
        per_step = trajectory_naturalness(traj, behavior, domain, node_coords)
        return text, float(per_step.mean()), float(per_step.max())

    rows: list[dict[str, Any]] = []
    _text, mean_d, max_d = _score(args.steer)
    rows.append({
        "label": "manifold", "steering": args.steer,
        "mean_bhattacharyya": mean_d, "max_bhattacharyya": max_d,
    })

    if args.compare_linear:
        steering = parse_expr(args.steer)
        mterms = [
            v for v in steering.alphas.values()
            if isinstance(v, ManifoldTerm)
        ]
        if len(mterms) != 1 or len(steering.alphas) != 1:
            print(
                "experiment naturalness: --compare-linear requires the "
                "steer expression to be a single manifold term",
                file=sys.stderr,
            )
            sys.exit(2)
        mt = mterms[0]
        session.ensure_manifold_loaded(mt.manifold)
        act_manifold = session.manifolds[mt.manifold]
        # Resolve label-form positions to coords up front — every
        # downstream call here wants a coord tuple, and the chord
        # baseline is per-coord arithmetic that can't operate on a
        # string label.
        mt_position = act_manifold.resolve_position(mt.position)
        # Linear baseline: the straight chord through activation space
        # from the manifold point at node 0 to the term's position, per
        # layer — what plain additive steering would do instead of
        # following the manifold's curvature.
        origin = act_manifold.node_coords[0]
        chord = {
            L: (
                act_manifold.manifold_point(L, mt_position)
                - act_manifold.manifold_point(L, origin)
            )
            for L in act_manifold.layer_indices
        }
        session.steer("__manifold_linear_baseline__", Profile(chord))
        _ltext, lmean, lmax = _score(
            f"{mt.coeff:g} __manifold_linear_baseline__"
        )
        pos_label = (
            mt.position if isinstance(mt.position, str)
            else ",".join(f"{c:g}" for c in mt_position)
        )
        rows.append({
            "label": "linear-chord", "steering": f"chord@{pos_label}",
            "mean_bhattacharyya": lmean, "max_bhattacharyya": lmax,
        })

    if args.json_output:
        print(_json.dumps({"prompt": args.prompt, "results": rows}, indent=2))
        return
    print()
    print("behavior-manifold naturalness  (lower = more natural)")
    for r in rows:
        print(
            f"  {r['label']:<14} mean D_B={r['mean_bhattacharyya']:.4f}  "
            f"max D_B={r['max_bhattacharyya']:.4f}"
        )
    if len(rows) == 2:
        delta = rows[1]["mean_bhattacharyya"] - rows[0]["mean_bhattacharyya"]
        verdict = (
            "manifold steering stays closer to the behavior manifold"
            if delta > 0 else
            "linear steering stays closer to the behavior manifold"
        )
        print(f"  -> {verdict} (Δmean={delta:+.4f})")


def _run_experiment_transcript(args: argparse.Namespace) -> None:
    """Dispatch ``saklas experiment transcript <verb>``."""
    cmd = getattr(args, "transcript_cmd", None)
    if cmd is None:
        print("usage: saklas experiment transcript <verb> [...]")
        print()
        print("  run  Replay a transcript on the current session")
        sys.exit(0)
    if cmd == "run":
        _run_transcript_run(args)
        return
    print(f"unknown experiment transcript verb {cmd!r}", file=sys.stderr)
    sys.exit(2)


def _parse_grid_terms(raw_terms: list[str]) -> dict[str, list[float]]:
    from saklas.cli.alpha_grid import AlphaListError, parse_alpha_list

    grid: dict[str, list[float]] = {}
    for raw in raw_terms:
        if "=" not in raw:
            print(
                f"experiment fan: grid term must be CONCEPT=ALPHAS, got {raw!r}",
                file=sys.stderr,
            )
            sys.exit(2)
        name, alpha_text = raw.split("=", 1)
        name = name.strip()
        if not name:
            print("experiment fan: grid concept cannot be empty", file=sys.stderr)
            sys.exit(2)
        try:
            alphas = parse_alpha_list(alpha_text)
        except AlphaListError as e:
            print(f"experiment fan: {name}: {e}", file=sys.stderr)
            sys.exit(2)
        if not alphas:
            print(f"experiment fan: {name}: alpha list is empty", file=sys.stderr)
            sys.exit(2)
        grid[name] = [float(a) for a in alphas]
    return grid


def _run_experiment_fan(args: argparse.Namespace) -> None:
    import json as _json

    _pkg._load_effective_config(args)
    _pkg._print_startup(args)
    session = _pkg._make_session(args)
    _pkg._print_model_info(session)

    grid = _parse_grid_terms(args.grid)
    runset = session.generate_sweep(
        args.prompt,
        grid,
        base_steering=args.base_steering,
        stateless=False,
    )
    if args.json_output:
        print(_json.dumps(runset.to_dict(), indent=2))
        return
    print(f"experiment fan: {len(runset)} run(s)")
    for idx, result in enumerate(runset):
        node_id = runset.node_ids[idx] if idx < len(runset.node_ids) else None
        row = runset.grid[idx] if idx < len(runset.grid) else {}
        row_str = ", ".join(f"{k}={v:+.3f}" for k, v in row.items())
        node_str = f" node={node_id[:8]}" if node_id else ""
        print(
            f"{idx:>3}: {row_str}{node_str} "
            f"tokens={result.token_count} finish={result.finish_reason}"
        )


def _run_transcript_run(args: argparse.Namespace) -> None:
    from saklas.core.transcript import (
        Transcript, TranscriptError,
    )

    transcript_path = Path(args.path)
    if not transcript_path.is_file():
        print(f"transcript run: {transcript_path}: file not found", file=sys.stderr)
        sys.exit(2)
    try:
        transcript = Transcript.load(transcript_path)
    except TranscriptError as e:
        print(f"transcript run: {e}", file=sys.stderr)
        sys.exit(2)

    _pkg._load_effective_config(args)
    if not args.model:
        if transcript.model_id:
            args.model = transcript.model_id
        else:
            print(
                "transcript run: model required (pass <model> or include "
                "`model_id` in the transcript)",
                file=sys.stderr,
            )
            sys.exit(2)

    _pkg._print_startup(args)
    session = _pkg._make_session(args)
    _pkg._print_model_info(session)

    # Import via ``default`` so the transcript lands as a fresh branch
    # under the synthetic root; replay walks the imported branch and
    # reports drift inline.
    try:
        leaf_id = transcript.import_into(
            session, mode="default", strict=args.strict,
        )
    except TranscriptError as e:
        print(f"transcript run: {e}", file=sys.stderr)
        sys.exit(2)

    print(f"transcript: {len(transcript.turns)} turns loaded "
          f"(leaf: {leaf_id[:8]})")
    print()
    for idx, turn in enumerate(transcript.turns):
        if turn.role != "user":
            continue
        print(f"--- replay turn {idx} ---")
        print(f"user: {turn.text[:80]}")
        # Look ahead for the assistant turn this user prompt produced.
        expected = None
        if idx + 1 < len(transcript.turns) and transcript.turns[idx + 1].role == "assistant":
            expected = transcript.turns[idx + 1]
        try:
            recipe = expected.recipe if expected is not None else None
            steering = recipe.steering if recipe is not None else None
            sampling = recipe.sampling if recipe is not None else None
            result = session.generate(
                turn.text,
                steering=steering,
                sampling=sampling,
                stateless=True,
            ).first
        except Exception as e:
            print(f"  replay failed: {e}")
            continue
        print(f"assistant: {result.text[:120]}")
        if expected is not None and expected.readings:
            actual = {
                name: (reading.coords[0] if reading.coords else 0.0)
                for name, reading in result.probe_readings.items()
            }
            deltas = []
            for name, expected_v in expected.readings.items():
                actual_v = actual.get(name, 0.0)
                deltas.append((name, actual_v - expected_v, expected_v, actual_v))
            deltas.sort(key=lambda x: abs(x[1]), reverse=True)
            print("  readings drift (top 5):")
            for name, d, ev, av in deltas[:5]:
                print(f"    {name:<32}  Δ {d:+.4f}  (expected {ev:+.4f} → got {av:+.4f})")
        print()
