"""``saklas manifold <verb>`` compute runners (extract/generate/from-template/
fit/bake/merge/transfer/compare/why) and their dispatch table."""

from __future__ import annotations

import argparse
from operator import itemgetter
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import saklas.cli.runners as _pkg
from saklas.cli.parsers import _MANIFOLD_VERBS
from saklas.cli.runners.shared import (
    _resolve_manifold_folder, _resolve_manifold_ns_name, _saklas_error_exit,
    _split_manifold_ns_name,
)
from saklas.core.histogram import summarize_diagnostics
from saklas.core.stats import median_or_zero
from saklas.io.paths import VARIANT_SUFFIX_RE

if TYPE_CHECKING:
    from saklas.core.profile import Profile


def _run_manifold_bake(args: argparse.Namespace) -> None:
    from saklas.io import merge as merge_mod
    dst = merge_mod.merge_into_manifold(
        args.name, args.expression, model=args.model,
        force=args.force, strict=args.strict,
    )
    print(f"Merged manifold written to {dst}")


def _require_model(args: argparse.Namespace) -> None:
    if not args.model:
        # The compute leaves (extract / fit / generate / bake / transfer) carry
        # the leaf on ``manifold_cmd`` — surface it so the error reads
        # "manifold fit".
        manifold_cmd = getattr(args, "manifold_cmd", None)
        cmd = f"manifold {manifold_cmd}" if manifold_cmd else "?"
        print(f"{cmd}: -m/--model is required", file=sys.stderr)
        sys.exit(2)


def _run_manifold_extract(args: argparse.Namespace) -> None:
    _require_model(args)
    import pathlib
    from saklas.io.paths import manifold_dir, tensor_filename

    if len(args.concept) == 1:
        raw = args.concept[0]
        baseline = None
    elif len(args.concept) == 2:
        raw = args.concept[0]
        baseline = args.concept[1]
    else:
        print(
            "extract: expected 1 or 2 positional arguments "
            f"(got {len(args.concept)})",
            file=sys.stderr,
        )
        sys.exit(2)

    requested_release = getattr(args, "sae", None)
    requested_role = getattr(args, "role", None)
    if requested_release and requested_role:
        print(
            "extract: --sae and --role are mutually exclusive "
            "(role substitution bakes into the manifold's node corpora; SAE "
            "reconstructs the centroids — the two don't compose)",
            file=sys.stderr,
        )
        sys.exit(2)

    # A steering vector is a 2-node ``pca`` manifold (4.0); a role-augmented
    # fit bakes into its node corpora and writes the canonical tensor name (no
    # ``_role-`` suffix). Cache validation belongs to the loaded session/pipeline
    # because bare file existence cannot prove sidecar integrity, corpus/role/SAE
    # identity, or the loaded model fingerprint.
    ns = getattr(args, "namespace", None) or "local"
    tensor_name = tensor_filename(args.model, release=requested_release)
    _pkg._print_startup(args)
    session = _pkg._make_session(args, load_probes=False)
    _pkg._print_model_info(session)

    extract_kwargs: dict[str, Any] = {}
    if requested_release:
        extract_kwargs["sae"] = args.sae
    if requested_role:
        extract_kwargs["role"] = requested_role
    if getattr(args, "namespace", None) is not None:
        extract_kwargs["namespace"] = args.namespace
    if args.force:
        extract_kwargs["force"] = True

    try:
        if baseline is not None:
            name, _profile = session.extract(raw, baseline=baseline, **extract_kwargs)
        else:
            name, _profile = session.extract(raw, **extract_kwargs)
    except Exception as e:
        print(f"extract failed: {e}", file=sys.stderr)
        sys.exit(1)

    final_path = pathlib.Path(manifold_dir(ns, name)) / tensor_name
    print(f"extracted {name} -> {final_path}")


# Single source of truth lives in ``io.paths`` (owns the variant scheme).
_VARIANT_SUFFIX_RE = VARIANT_SUFFIX_RE


def _split_variant_suffix(raw: str) -> tuple[str, str | None]:
    """Peel a trailing ``:<variant>`` off a selector string.

    Returns ``(name_part, variant_or_None)``. ``variant`` is one of the
    tensor suffixes understood by :func:`saklas.io.packs.enumerate_variants`.
    Non-variant colon usage (``tag:``, ``namespace:``, ``model:``) passes
    through unchanged with ``variant=None`` — those prefixes are caught by
    ``sel.parse`` later.
    """
    if ":" not in raw:
        return raw, None
    head, _, tail = raw.rpartition(":")
    if _VARIANT_SUFFIX_RE.match(tail) and head:
        # ``_VARIANT_SUFFIX_RE`` is anchored over ``[a-z0-9._-]`` only, so a
        # matching ``tail`` can't contain ``/`` — ``model:<org>/<name>`` falls
        # through to ``return raw, None`` because its tail fails the regex.
        return head, tail
    return raw, None


def _fold_manifold_to_profile_with_identity(
    name: str, model_id: str, variant: str | None,
) -> "tuple[Profile, str, str] | None":
    """Fold a fitted 2-node ``pca`` manifold and return its namespace identity.

    The identity is needed by ``compare``'s rank-all mode: bundled and user
    concepts can share a bare name across namespaces, so callers must key
    scanned profiles by ``namespace/name`` rather than letting later matches
    overwrite earlier ones.
    """
    from saklas.core.profile import Profile
    from saklas.core.manifold import load_manifold
    from saklas.core.vectors import folded_vector_directions
    from saklas.io.paths import manifold_dir, manifolds_dir, tensor_filename
    from saklas.io.selectors import AmbiguousSelectorError

    if variant in (None, "raw"):
        release: str | None = None
    elif variant.startswith("sae-"):
        release = variant[len("sae-"):]
    else:
        return None

    if "/" in name:
        ns, bare = name.split("/", 1)
        search_ns = [ns]
    else:
        bare = name
        root = manifolds_dir()
        search_ns = (
            sorted(d.name for d in root.iterdir() if d.is_dir())
            if root.exists() else []
        )
    fname = tensor_filename(model_id, release=release)
    hits = [
        manifold_dir(ns, bare) / fname
        for ns in search_ns
        if (manifold_dir(ns, bare) / fname).exists()
    ]
    if not hits:
        return None
    if len(hits) > 1:
        qualified = ", ".join(f"{p.parent.parent.name}/{bare}" for p in hits)
        raise AmbiguousSelectorError(
            f"ambiguous manifold '{bare}': matches {qualified}. "
            f"Qualify it with a namespace."
        )
    try:
        manifold = load_manifold(hits[0])
        dirs = folded_vector_directions(manifold)
    except Exception:
        return None
    ns = hits[0].parent.parent.name
    return Profile(dirs), ns, bare


def _fold_all_fitted_manifolds(
    model_id: str, *, exclude_identity: tuple[str, str] | None = None,
) -> "dict[str, Profile]":
    """Fold every fitted 2-node ``pca`` manifold into ``{ns/name: Profile}``.

    Backs ``compare``'s 1-arg "rank against all installed" mode now that
    bundled concepts are manifolds.  Only manifolds with a fitted raw tensor
    for ``model_id`` that fold to a 2-node affine are included;
    ``exclude_identity`` drops the target by exact ``(namespace, name)``.
    """
    from saklas.core.profile import Profile
    from saklas.core.manifold import load_manifold
    from saklas.core.vectors import folded_vector_directions
    from saklas.io.paths import manifolds_dir, tensor_filename

    out: dict[str, Profile] = {}
    root = manifolds_dir()
    if not root.is_dir():
        return out
    fname = tensor_filename(model_id, release=None)
    for ns_dir in sorted(root.iterdir()):
        if not ns_dir.is_dir():
            continue
        for mdir in sorted(ns_dir.iterdir()):
            if not mdir.is_dir():
                continue
            identity = (ns_dir.name, mdir.name)
            if identity == exclude_identity:
                continue
            tensor = mdir / fname
            if not tensor.is_file():
                continue
            try:
                dirs = folded_vector_directions(load_manifold(tensor))
            except Exception:
                continue  # not a foldable 2-node affine (curved / rank > 1)
            out[f"{ns_dir.name}/{mdir.name}"] = Profile(dirs)
    return out


def _run_manifold_compare(args: argparse.Namespace) -> None:
    import json as _json
    from saklas.io.selectors import parse as sel_parse, resolve

    # Compare is Mahalanobis-only: load the per-model whitener once up
    # front and share it across every ``cosine_similarity`` call below.
    # Failure is fatal — there is no Euclidean path, so a missing neutral
    # cache surfaces directly instead of degrading silently.
    from saklas.core.mahalanobis import LayerWhitener, WhitenerError

    whitener: "Any | None"
    try:
        whitener = LayerWhitener.from_cache(
            args.model,
            ridge_scale=getattr(args, "ridge_scale", 1.0),
        )
    except WhitenerError as e:
        print(f"compare: {e}", file=sys.stderr)
        sys.exit(1)

    # Expand selectors into (name, variant) pairs. Variant travels with the
    # name through the load loop so ``foo:sae`` picks the SAE tensor.
    names: list[tuple[str, str | None]] = []
    for raw in args.concepts:
        name_part, variant = _split_variant_suffix(raw)
        try:
            sel = sel_parse(name_part)
        except Exception:
            names.append((name_part, variant))
            continue
        if sel.kind == "name":
            names.append((name_part, variant))
        else:
            # Bulk selectors (tag:/namespace:/all) expand to individual
            # names; inherit the variant suffix so `tag:emotion:sae`
            # resolves SAE tensors across the tag.
            resolved = resolve(sel)
            names.extend((f"{c.namespace}/{c.name}", variant) for c in resolved)

    # Load profiles by folding each concept's fitted 2-node ``pca`` manifold.
    profiles: dict[str, Profile] = {}
    profile_identities: dict[str, tuple[str, str]] = {}
    for name, variant in names:
        try:
            folded = _pkg._fold_manifold_to_profile_with_identity(
                name, args.model, variant,
            )
        except Exception as e:
            print(f"warning: {e}, skipping", file=sys.stderr)
            continue
        if folded is None:
            print(
                f"warning: no fitted manifold for '{name}' with model "
                f"{args.model}, skipping",
                file=sys.stderr,
            )
            continue
        loaded, ns, bare = folded
        display_base = f"{ns}/{bare}" if "/" in name else bare
        display = display_base if variant is None else f"{display_base}:{variant}"
        profiles[display] = loaded
        profile_identities[display] = (ns, bare)

    if len(profiles) < 1:
        print("compare: no loadable profiles found", file=sys.stderr)
        sys.exit(1)

    ordered = list(profiles.keys())

    # 1-arg mode: rank all installed against the target.
    if len(args.concepts) == 1 and len(ordered) == 1:
        target_name = ordered[0]
        target = profiles[target_name]
        target_identity = profile_identities.get(target_name)

        # Every concept ships as a fitted 2-node ``pca`` manifold — fold
        # every fitted one for this model into the ranking pool.  Keep the
        # exact namespace identity internally, then choose display names after
        # the full pool is known: bare for unique names, qualified for
        # collisions.
        other_entries: dict[tuple[str, str], Profile] = {}
        for mname, mprof in _pkg._fold_all_fitted_manifolds(
            args.model, exclude_identity=target_identity,
        ).items():
            ns, bare = mname.split("/", 1)
            other_entries[(ns, bare)] = mprof

        from collections import Counter

        target_bare = (
            target_identity[1] if target_identity is not None
            else target_name.rsplit("/", 1)[-1]
        )
        bare_counts = Counter(name for _ns, name in other_entries)
        others: dict[str, Profile] = {}
        for (ns, bare), profile in sorted(other_entries.items()):
            display = (
                f"{ns}/{bare}"
                if bare_counts[bare] > 1 or bare == target_bare
                else bare
            )
            others[display] = profile

        if not others:
            print(f"compare: no other profiles found for model {args.model}", file=sys.stderr)
            sys.exit(1)

        scores: dict[str, float] = {
            name: target.cosine_similarity(p, whitener=whitener)
            for name, p in others.items()
        }
        ranked = sorted(scores.items(), key=itemgetter(1), reverse=True)

        if args.json_output:
            result: dict[str, Any] = {"target": target_name, "model": args.model,
                            "similarities": [{"name": n, "similarity": round(s, 6)}
                                              for n, s in ranked]}
            if args.verbose:
                top3 = ranked[:3]
                result["per_layer_top3"] = {
                    n: {str(k): round(v, 6)
                        for k, v in target.cosine_similarity(others[n], per_layer=True, whitener=whitener).items()}
                    for n, _ in top3
                }
            print(_json.dumps(result, indent=2))
        else:
            width = max(len(n) for n, _ in ranked)
            print(f"{target_name} vs all installed ({args.model}):")
            for name, score in ranked:
                print(f"  {name:<{width}}  {score:+.4f}")
            if args.verbose and ranked:
                print()
                print("  per-layer (top 3):")
                for name, _ in ranked[:3]:
                    per_layer_top3: dict[int, float] = target.cosine_similarity(others[name], per_layer=True, whitener=whitener)
                    print(f"    {name}:")
                    for layer in sorted(per_layer_top3):
                        print(f"      layer {layer:>3}: {per_layer_top3[layer]:+.4f}")
        return

    if len(ordered) < 2:
        print("compare: need at least 2 profiles to compare", file=sys.stderr)
        sys.exit(1)

    # 2-arg mode: pairwise.
    if len(ordered) == 2:
        a_name, b_name = ordered
        a, b = profiles[a_name], profiles[b_name]
        sim: float = a.cosine_similarity(b, whitener=whitener)

        if args.json_output:
            result = {"a": a_name, "b": b_name, "model": args.model,
                      "similarity": round(sim, 6)}
            if args.verbose:
                result["per_layer"] = {str(k): round(v, 6)
                                       for k, v in a.cosine_similarity(b, per_layer=True, whitener=whitener).items()}
            print(_json.dumps(result, indent=2))
        else:
            print(f"{a_name} ~ {b_name}: {sim:+.4f}")
            if args.verbose:
                per_layer_2: dict[int, float] = a.cosine_similarity(b, per_layer=True, whitener=whitener)
                for layer in sorted(per_layer_2):
                    print(f"  layer {layer:>3}: {per_layer_2[layer]:+.4f}")
        return

    # 3+ mode: N×N matrix.
    matrix: dict[str, dict[str, float]] = {}
    for a_name in ordered:
        matrix[a_name] = {}
        for b_name in ordered:
            if a_name == b_name:
                matrix[a_name][b_name] = 1.0
            else:
                matrix[a_name][b_name] = profiles[a_name].cosine_similarity(profiles[b_name], whitener=whitener)

    if args.json_output:
        result = {"model": args.model, "concepts": ordered,
                  "matrix": {a: {b: round(v, 6) for b, v in row.items()}
                              for a, row in matrix.items()}}
        if args.verbose:
            per_layer: dict[str, dict[str, float]] = {}
            for i, a_name in enumerate(ordered):
                for b_name in ordered[i + 1:]:
                    key = f"{a_name}|{b_name}"
                    per_layer[key] = {
                        str(k): round(v, 6)
                        for k, v in profiles[a_name].cosine_similarity(
                            profiles[b_name], per_layer=True, whitener=whitener,
                        ).items()
                    }
            result["per_layer"] = per_layer
        print(_json.dumps(result, indent=2))
    else:
        width = max(len(n) for n in ordered)
        header = " " * (width + 2) + "  ".join(f"{n:>{width}}" for n in ordered)
        print(header)
        for a_name in ordered:
            row = "  ".join(f"{matrix[a_name][b]:>{width}.4f}" for b in ordered)
            print(f"{a_name:<{width}}  {row}")


def _run_manifold_why(args: argparse.Namespace) -> None:
    import json as _json
    from saklas.io.selectors import AmbiguousSelectorError

    # Peel off a ``:<variant>`` suffix before resolving the manifold.
    name_part, variant = _split_variant_suffix(args.concept)

    # Every concept is a fitted 2-node ``pca`` manifold — fold it to a vector.
    profile: "Profile | None" = None
    concept_name = name_part if variant is None else f"{name_part}:{variant}"
    try:
        folded = _pkg._fold_manifold_to_profile_with_identity(
            name_part, args.model, variant,
        )
    except AmbiguousSelectorError as e:
        print(f"why: {e}", file=sys.stderr)
        sys.exit(1)
    if folded is not None:
        profile, ns, bare = folded
        concept_base = f"{ns}/{bare}" if "/" in name_part else bare
        concept_name = concept_base if variant is None else f"{concept_base}:{variant}"
    if profile is None:
        print(
            f"why: no fitted manifold for '{args.concept}' with model "
            f"{args.model}. If it's authored but unfit, fit it first: "
            f"`saklas manifold fit {name_part} -m {args.model}`.",
            file=sys.stderr,
        )
        sys.exit(1)

    layer_mags: list[tuple[int, float]] = sorted(
        ((layer, float(tensor.norm().item())) for layer, tensor in profile.items()),
        key=itemgetter(0),
    )
    total_layers = len(profile)
    diagnostics = profile.diagnostics  # None when extracted before saklas 1.6

    if args.json_output:
        result: dict[str, Any] = {
            "concept": concept_name,
            "model": args.model,
            "total_layers": total_layers,
            "layers": [{"layer": l, "magnitude": round(m, 6)} for l, m in layer_mags],
        }
        if diagnostics is not None:
            result["diagnostics_by_layer"] = {
                str(layer): {k: round(float(v), 6) for k, v in metrics.items()}
                for layer, metrics in sorted(diagnostics.items())
            }
            result["diagnostics_summary"] = summarize_diagnostics(diagnostics)
        print(_json.dumps(result, indent=2))
    else:
        _print_why_histogram(concept_name, args.model, total_layers, layer_mags)
        if diagnostics is not None:
            _print_diagnostics(diagnostics)


def _print_diagnostics(diagnostics: dict[int, dict[str, float]]) -> None:
    """Render the diagnostics summary + per-layer table beneath the histogram."""
    summary = summarize_diagnostics(diagnostics)
    quality = summary["quality"]
    print()
    print(f"  DIAGNOSTICS (probe quality: {quality}):")
    print(
        f"    median EVR:                 {summary['median_evr']:.3f}\n"
        f"    median intra-pair variance: {summary['median_intra_pair_variance']:.4f}\n"
        f"    median inter-pair alignment:{summary['median_inter_pair_alignment']:>7.3f}\n"
        f"    median diff→PC projection:  {summary['median_diff_principal_projection']:.3f}"
    )


def _print_why_histogram(
    concept_name: str,
    model_id: str,
    total_layers: int,
    layer_mags: list[tuple[int, float]],
) -> None:
    import shutil
    from saklas.core.histogram import HIST_BUCKETS, bucketize

    print(f"{concept_name} ({total_layers} layers, {model_id}):")
    print("  LAYERS (mean ||baked|| per bucket):")
    if not layer_mags:
        return

    term_w = shutil.get_terminal_size((80, 24)).columns
    buckets = bucketize(layer_mags, HIST_BUCKETS)
    max_norm = max(v for _, _, v in buckets) or 1.0
    label_w = max(2, len(str(max(hi for _, hi, _ in buckets))))

    def _label(lo: int, hi: int) -> str:
        return f"L{lo:0{label_w}}" if lo == hi else f"L{lo:0{label_w}}-{hi:0{label_w}}"

    label_col = max(len(_label(lo, hi)) for lo, hi, _ in buckets)
    # "    <label>  <bar>  <value>" — 4 indent + label_col + 2 + bar + 2 + 8
    value_w = 8
    bar_w = max(12, term_w - 4 - label_col - 2 - 2 - value_w)
    for lo, hi, norm in buckets:
        filled = min(int(norm / max_norm * bar_w), bar_w)
        bar = "█" * filled + "░" * (bar_w - filled)
        print(f"    {_label(lo, hi):<{label_col}}  {bar}  {norm:>{value_w}.3f}")


def _run_manifold_fit(args: argparse.Namespace) -> None:
    """``saklas manifold fit`` — fit an authored OR discover-mode manifold.

    The positional ``target`` is a manifold name OR a folder path.  When it
    resolves to an authored folder the fit runs as-is (no hyperparam knobs
    allowed).  When it resolves to a discover-mode folder (``pca`` /
    ``spectral``) any supplied hyperparam override (``--method`` / ``--max-dim``
    / ``--var-threshold`` / ``--k-nn`` / ``--bandwidth`` / ``--max-subspace-dim``)
    is passed into the fit transaction, which updates ``manifold.json`` under
    the same lock used to derive the cache key. Supplying overrides against an
    authored folder is an error (mirrors the server's 400).
    """
    from saklas.io.manifolds import (
        ManifoldFolder, ManifoldFormatError, domain_label,
    )

    _require_model(args)

    # Resolve ``target`` to a folder.  A path that exists on disk (or that
    # carries a ``manifold.json``) is taken as an authored-folder path —
    # ``manifold fit /path/to/folder``; otherwise it is a manifold *name*
    # resolved cross-namespace the same way the lifecycle verbs resolve.
    target = args.target
    target_path = Path(target)
    if target_path.is_dir() or (target_path / "manifold.json").exists():
        folder = target_path
    else:
        folder = _resolve_manifold_folder(target)

    if not (folder / "manifold.json").exists():
        print(
            f"manifold fit: no manifold.json in {folder}", file=sys.stderr,
        )
        sys.exit(2)

    # Load the folder to learn its fit_mode before paying for a model load.
    try:
        mf = ManifoldFolder.load(folder, verify_manifest=False)
    except ManifoldFormatError as e:
        print(f"manifold fit: {e}", file=sys.stderr)
        sys.exit(2)

    # Did the user supply any discover-mode hyperparam override?
    override_supplied = any(
        getattr(args, attr, None) is not None
        for attr in (
            "method", "max_dim", "min_dim", "var_threshold", "k_nn",
            "bandwidth", "max_subspace_dim", "smoothing",
            "persistence_frac",
        )
    )

    requested_fit_mode: str | None = None
    requested_hyperparams: dict[str, object] | None = None
    if override_supplied:
        if not mf.is_discover:
            # Mirror the server's 400: hyperparam overrides are discover-only.
            print(
                f"manifold fit: fit_mode/hyperparams overrides are "
                f"discover-mode only; {folder} is authored",
                file=sys.stderr,
            )
            sys.exit(2)
        requested_fit_mode = args.method
        if requested_fit_mode is not None and requested_fit_mode not in {
            "pca", "spectral", "auto",
        }:
            print(
                f"manifold fit: invalid method {requested_fit_mode!r}",
                file=sys.stderr,
            )
            sys.exit(2)
        requested_hyperparams = {}
        # PCA knobs.
        if args.max_dim is not None:
            requested_hyperparams["max_dim"] = int(args.max_dim)
        if args.var_threshold is not None:
            requested_hyperparams["var_threshold"] = float(args.var_threshold)
        # Spectral knobs.
        if getattr(args, "min_dim", None) is not None:
            requested_hyperparams["min_dim"] = int(args.min_dim)
        if args.k_nn is not None:
            requested_hyperparams["k_nn"] = int(args.k_nn)
        if args.bandwidth is not None:
            requested_hyperparams["bandwidth"] = float(args.bandwidth)
        # Spectral-only knob. The exact-mode sanitizer rejects it for PCA (a
        # flat fit's per-layer subspace dim is its layout dim, capped by
        # ``--max-dim``, not a separate knob).
        if getattr(args, "max_subspace_dim", None) is not None:
            requested_hyperparams["max_subspace_dim"] = int(
                args.max_subspace_dim,
            )
        # Curved-fit smoothing (spectral/auto): "auto" → GCV, else a float λ.
        if getattr(args, "smoothing", None) is not None:
            s = args.smoothing
            requested_hyperparams["smoothing"] = (
                s if s == "auto" else float(s)
            )
        # Auto-only: periodic-detection persistence threshold.
        if getattr(args, "persistence_frac", None) is not None:
            requested_hyperparams["persistence_frac"] = float(
                args.persistence_frac,
            )

    _pkg._print_startup(args)
    session = _pkg._make_session(args, load_probes=False)
    _pkg._print_model_info(session)
    try:
        manifold = session.fit(
            folder,
            sae=getattr(args, "sae", None),
            layers=getattr(args, "layers", None),
            fit_mode=requested_fit_mode,
            hyperparams=requested_hyperparams,
            force=bool(getattr(args, "force", False)),
            on_progress=lambda m: print(f"  {m}"),
        )
    except (ValueError, ManifoldFormatError) as e:
        print(f"manifold fit failed: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"manifold fit failed: {e}", file=sys.stderr)
        sys.exit(1)
    fitted_mode = getattr(manifold, "metadata", {}).get("fit_mode", mf.fit_mode)
    tail = f", fit_mode={fitted_mode}" if mf.is_discover else ""
    print(
        f"fitted manifold '{manifold.name}' "
        f"({len(manifold.layers)} layers, {len(manifold.node_labels)} nodes, "
        f"{domain_label(manifold.domain.to_spec())}, "
        f"{manifold.feature_space}{tail})"
    )


def _run_manifold_generate(args: argparse.Namespace) -> None:
    """``saklas manifold generate`` -- author + LLM-fill a discover folder (A2).

    Two-step end-to-end: load the model, run :meth:`SaklasSession.generate_responses`
    to fill each node's conversational corpus (in-character responses to the
    shared baseline prompts), then leave the folder ready for ``manifold
    discover`` to fit.  The two-step split keeps failure modes legible (a flaky
    generation leaves inspectable corpora) and lets the user review before
    paying for the fit.
    """
    from saklas.io.manifolds import (
        ManifoldFormatError,
        append_discover_manifold_node,
        plan_discover_generation,
    )
    from saklas.io.paths import manifold_dir

    _require_model(args)
    if args.kind == "custom" and not getattr(args, "custom_system", None):
        print(
            "manifold generate: --kind custom requires --system "
            '(a template with a {c} placeholder, e.g. "You are the month of '
            '{c}; speak as that month.")',
            file=sys.stderr,
        )
        sys.exit(2)
    if len(args.concepts) < 2:
        print(
            "manifold generate: need >= 2 concepts (a discover manifold is "
            "meaningless with one node)",
            file=sys.stderr,
        )
        sys.exit(2)

    namespace = "local"
    name = args.name
    if "/" in name:
        namespace, name = name.split("/", 1)

    folder = manifold_dir(namespace, name)
    # ``--role-per-node``: the concept slug doubles as that node's assistant-role
    # substitution -- a persona manifold pooled in role-baselined space (the
    # explicit role overrides the kind-derived elicitation label at both
    # generation and capture).  Otherwise the node\'s ``kind`` drives a
    # generation-only elicitation role and capture stays standard (swap-back).
    node_roles: dict[str, str | None] | None = None
    if getattr(args, "role_per_node", False):
        node_roles = {concept: concept for concept in args.concepts}
    node_kinds: dict[str, str | None] = {c: args.kind for c in args.concepts}

    # Plan first -- no model load needed to learn there is nothing to do.
    try:
        plan = plan_discover_generation(
            folder, name, args.description,
            fit_mode="pca", labels=list(args.concepts),
            node_roles=node_roles, node_kinds=node_kinds,
            force=args.force,
        )
    except ManifoldFormatError as e:
        print(f"manifold generate failed: {e}", file=sys.stderr)
        sys.exit(1)

    n_total = len(plan.index_of)
    if not plan.pending:
        print(
            f"manifold {namespace}/{name} already complete ({n_total} nodes)"
            f" -- pass -f/--force to regenerate"
        )
        return
    if plan.resumed and len(plan.pending) < n_total:
        print(
            f"resuming {namespace}/{name}: "
            f"{n_total - len(plan.pending)}/{n_total} nodes present, "
            f"generating {len(plan.pending)}"
        )
    if plan.added:
        print(f"adding {len(plan.added)} new node(s): {', '.join(plan.added)}")

    _pkg._print_startup(args)
    session = _pkg._make_session(args)
    _pkg._print_model_info(session)

    print(
        f"generating {args.samples_per_prompt} response(s) per baseline prompt "
        f"for {len(plan.pending)} concept(s)..."
    )
    try:
        for concept in plan.pending:
            gen_roles: dict[str, str | None] | None = (
                {concept: concept} if node_roles else None
            )
            corpora = session.generate_responses(
                [concept], [args.kind],
                roles=gen_roles,
                custom_system=getattr(args, "custom_system", None),
                samples_per_prompt=args.samples_per_prompt,
                on_progress=lambda m: print(f"  {m}"),
            )
            append_discover_manifold_node(
                folder, plan.index_of[concept], concept, corpora[concept],
            )
    except (ValueError, RuntimeError) as e:
        print(f"manifold generate failed: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"wrote {namespace}/{name} ({n_total} nodes)")
    print(f"  -> run `saklas manifold fit {namespace}/{name}` to fit")


def _run_manifold_from_template(args: argparse.Namespace) -> None:
    """``saklas manifold from-template`` — author a discover manifold from a template.

    Pure-IO (no model): resolves the standalone template, expands its values x
    contexts into per-value node corpora, and writes a discover folder that
    stores the corpus + ``template_ref``. Fit it next with
    ``saklas manifold fit <name> -m MODEL``.
    """
    from saklas.io.manifolds import ManifoldFormatError, create_manifold_from_template
    from saklas.io.templates import (
        AmbiguousTemplateError, TemplateNotFoundError, resolve_template,
    )

    try:
        tmpl = resolve_template(args.template)
    except (TemplateNotFoundError, AmbiguousTemplateError) as e:
        print(f"manifold from-template: {e}", file=sys.stderr)
        sys.exit(1)
    ref = f"{tmpl.path.parent.name}/{tmpl.name}" if tmpl.path else tmpl.name
    namespace, name = _split_manifold_ns_name(args.name or tmpl.name)

    hyperparams: dict[str, object] = {}
    if args.max_dim is not None:
        hyperparams["max_dim"] = args.max_dim
    if args.var_threshold is not None:
        hyperparams["var_threshold"] = args.var_threshold

    try:
        create_manifold_from_template(
            namespace, name, args.description,
            template_ref=ref,
            fit_mode=args.fit_mode,
            hyperparams=hyperparams or None,
            force=args.force,
        )
    except FileExistsError:
        print(
            f"manifold from-template: {namespace}/{name} already exists "
            f"-- pass -f/--force to overwrite",
            file=sys.stderr,
        )
        sys.exit(1)
    except ManifoldFormatError as e:
        print(f"manifold from-template failed: {e}", file=sys.stderr)
        sys.exit(1)

    print(
        f"wrote {namespace}/{name} ({len(tmpl.values)} nodes x "
        f"{len(tmpl.contexts)} contexts, template_ref={ref}, "
        f"fit_mode={args.fit_mode})"
    )
    print(f"  -> run `saklas manifold fit {namespace}/{name} -m MODEL` to fit")


def _run_manifold_merge(args: argparse.Namespace) -> None:
    from saklas.io.manifolds import (
        ManifoldFormatError, merge_discover_manifolds,
    )

    target_ns, target_name = _split_manifold_ns_name(args.name)
    sources = [_split_manifold_ns_name(s) for s in args.sources]
    if len(sources) < 2:
        print(
            "manifold merge: need >= 2 sources (union of one manifold is a no-op)",
            file=sys.stderr,
        )
        sys.exit(2)
    try:
        dst = merge_discover_manifolds(
            target_ns, target_name, args.description,
            sources=sources,
            fit_mode=args.method,  # parser dest unified to ``method`` (matches discover)
            force=args.force,
        )
    except (FileNotFoundError, FileExistsError, ManifoldFormatError, ValueError) as e:
        print(f"manifold merge failed: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"Merged manifold written to {dst}")
    print(
        f"  -> run `saklas manifold fit {target_ns}/{target_name}` to fit"
    )


def _run_manifold_transfer(args: argparse.Namespace) -> None:
    """Cross-model manifold transfer via Procrustes.

    Resolve the manifold folder, fit (or load) the per-layer Procrustes
    alignment between the source and target models' cached neutral activations
    — the part that needs both models loaded — then hand the prebuilt alignment
    dict to :func:`saklas.io.manifolds.transfer_manifold`, which is pure-io and
    only *applies* the map.  Steering vectors transfer the same way (a vector is
    the K=2 case of a flat ``pca`` manifold).
    """
    import json as _json

    from saklas.io.manifolds import (
        ManifoldFormatError, preflight_transfer_manifold, transfer_manifold,
    )
    from saklas.io.paths import encode_release_id, manifold_dir

    ns, name = _resolve_manifold_ns_name(args.name)
    folder = manifold_dir(ns, name)
    if not (folder / "manifold.json").exists():
        print(
            f"manifold transfer: {ns}/{name} not found at {folder}",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        source_proof = preflight_transfer_manifold(
            folder, from_model=args.src_model, to_model=args.tgt_model,
            force=args.force,
        )
    except (FileNotFoundError, FileExistsError, ManifoldFormatError) as e:
        print(f"manifold transfer failed: {e}", file=sys.stderr)
        sys.exit(1)
    requested_alignment_layers = (
        list(source_proof.layers) if source_proof is not None else None
    )

    # Alignment construction also returns the identity-matched target
    # whitener. On a cold/stale fit it is built directly from the target
    # activations already resident for Procrustes; an exact model-free repeat
    # uses the offline cache loader. Either path reads the target neutral
    # artifact once for this operation.
    from saklas.core.mahalanobis import WhitenerError

    try:
        (
            M, quality_per_layer, _, source_identity, target_identity,
            target_whitener, target_layer_means,
        ) = _pkg._load_or_fit_transfer_alignment(
            args.src_model, args.tgt_model, force=args.force,
            label="manifold transfer",
            requested_layers=requested_alignment_layers,
        )
    except WhitenerError as e:
        print(f"manifold transfer failed: {e}", file=sys.stderr)
        sys.exit(1)

    median_quality = median_or_zero(list(quality_per_layer.values())) if quality_per_layer else None

    try:
        out_path = transfer_manifold(
            folder,
            from_model=args.src_model,
            to_model=args.tgt_model,
            alignment=M,
            transfer_quality_estimate=median_quality,
            source_model_fingerprint=source_identity["model_fingerprint"],
            target_model_fingerprint=target_identity["model_fingerprint"],
            whitener=target_whitener,
            target_layer_means=target_layer_means,
            force=args.force,
            expected_source_proof=source_proof,
        )
    except (FileNotFoundError, FileExistsError, ManifoldFormatError) as e:
        print(f"manifold transfer failed: {e}", file=sys.stderr)
        sys.exit(1)

    payload = {
        "namespace": ns,
        "name": name,
        "source_model": args.src_model,
        "target_model": args.tgt_model,
        "tensor": str(out_path),
        "transferred_layers": sorted(M.keys()),
        "median_transfer_quality": (
            round(median_quality, 4) if median_quality is not None else None
        ),
    }
    if args.json_output:
        print(_json.dumps(payload, indent=2))
        return

    quality_str = f"{median_quality:.3f}" if median_quality is not None else "n/a"
    print(
        f"Transferred manifold {ns}/{name} "
        f"from {args.src_model} -> {args.tgt_model}\n"
        f"  layers:           {len(M)} shared\n"
        f"  median quality:   {quality_str} (R^2 across shared layers)\n"
        f"  tensor:           {out_path}\n"
        f"  variant suffix:   :from-{encode_release_id(args.src_model)}"
    )


# The nine steering-vector / manifold *compute* verbs (model-loading +
# analysis).  ``manifold transfer`` keeps its own runner (it was already a
# manifold verb); the rest are the renamed former ``subspace`` verbs plus the
# discover→fit fold.
_MANIFOLD_RUNNERS = {
    "extract":  _run_manifold_extract,
    "generate": _run_manifold_generate,
    "from-template": _run_manifold_from_template,
    "fit":      _run_manifold_fit,
    "bake":     _run_manifold_bake,
    "merge":    _run_manifold_merge,
    "transfer": _run_manifold_transfer,
    "compare":  _run_manifold_compare,
    "why":      _run_manifold_why,
}


@_saklas_error_exit
def _run_manifold(args: argparse.Namespace) -> None:
    """Dispatch ``saklas manifold <verb>`` (the compute verbs)."""
    cmd = getattr(args, "manifold_cmd", None)
    if cmd is None:
        print("usage: saklas manifold <verb> [...]")
        print()
        width = max(len(v) for v, _ in _MANIFOLD_VERBS)
        for v, desc in _MANIFOLD_VERBS:
            print(f"  {v:<{width}}  {desc}")
        print()
        print("Run `saklas manifold <verb> -h` for verb-specific options.")
        sys.exit(0)
    runner = _MANIFOLD_RUNNERS.get(cmd)
    if runner is None:
        print(f"unknown manifold verb {cmd!r}", file=sys.stderr)
        sys.exit(2)
    runner(args)
