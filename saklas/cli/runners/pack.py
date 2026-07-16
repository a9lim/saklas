"""``saklas pack <verb>`` lifecycle/distribution runners and dispatch table."""

from __future__ import annotations

import argparse
import sys
from typing import Any

from saklas.cli.parsers import _PACK_VERBS
from saklas.cli.runners.shared import (
    _iter_manifold_folders, _resolve_manifold_ns_name, _saklas_error_exit,
    _split_manifold_ns_name,
)


def _run_pack_ls(args: argparse.Namespace) -> None:
    import json as _json
    from saklas.io.manifolds import domain_label

    rows = list(_iter_manifold_folders(getattr(args, "namespace", None)))
    if getattr(args, "json_output", False):
        print(_json.dumps([
            {
                "namespace": ns,
                "name": mf.name,
                "domain": mf.domain,
                "nodes": mf.node_labels,
                "fitted_models": mf.tensor_models(),
            }
            for ns, mf in rows
        ], indent=2))
        return
    if not rows:
        print("no manifolds installed under ~/.saklas/manifolds/")
        return
    verbose = getattr(args, "verbose", False)
    for ns, mf in rows:
        kind = domain_label(mf.domain)
        models = ", ".join(mf.tensor_models()) or "(unfitted)"
        print(
            f"  {ns}/{mf.name}  [{kind}, {len(mf.node_labels)} nodes]  {models}"
        )
        if verbose and mf.description:
            print(f"    {mf.description}")


def _run_pack_show(args: argparse.Namespace) -> None:
    import json as _json
    from saklas.io.manifolds import domain_label

    name = args.name
    target_ns = None
    if "/" in name:
        target_ns, name = name.split("/", 1)
    matches = [
        (ns, mf)
        for ns, mf in _iter_manifold_folders(target_ns)
        if mf.name == name
    ]
    if not matches:
        print(f"manifold '{args.name}' not found", file=sys.stderr)
        sys.exit(1)
    if len(matches) > 1:
        qualified = ", ".join(f"{ns}/{mf.name}" for ns, mf in matches)
        print(
            f"ambiguous manifold '{name}': matches {qualified}; "
            f"qualify with a namespace",
            file=sys.stderr,
        )
        sys.exit(2)
    ns, mf = matches[0]
    fitted = []
    for stem in mf.tensor_models():
        sc = mf.sidecar(stem)
        entry: dict[str, Any] = {
            "stem": stem,
            "method": sc.method,
            "feature_space": sc.feature_space,
            "node_count": sc.node_count,
            "fit_mode": sc.fit_mode,
        }
        if sc.hyperparams:
            entry["hyperparams"] = sc.hyperparams
        if sc.diagnostics:
            entry["diagnostics"] = sc.diagnostics
        if sc.node_spread_per_layer:
            entry["node_spread"] = sc.node_spread_per_layer
        fitted.append(entry)

    # In discover mode the folder itself has no per-node coords (they are
    # derived per-model at fit time) — try to surface the derived coords
    # from the per-model tensor when a fit exists.  Cheap: one safetensors
    # header read.  Soft-fails to an empty list if the tensor isn't there.
    derived_node_coords: list[list[float]] = []
    if mf.is_discover and fitted:
        from saklas.io.manifold_tensors import load_manifold
        stem_for_session = fitted[0]["stem"]
        try:
            m = load_manifold(mf.tensor_path(stem_for_session))
            derived_node_coords = [
                [float(x) for x in row]
                for row in m.node_coords.tolist()
            ]
        except (FileNotFoundError, KeyError, ValueError):
            derived_node_coords = []

    # Per-node roles ride alongside coords in the JSON shape; ``None``
    # for a given node means "pooled under the standard assistant
    # baseline" (the legacy / non-role default).  Padded to the label
    # count so consumers can index by node.
    node_roles_padded = list(mf.node_roles) + [None] * (
        len(mf.node_labels) - len(mf.node_roles)
    )
    if getattr(args, "json_output", False):
        # Share the session-independent summary serializer with the
        # server's ``GET /saklas/v1/manifolds/{ns}/{name}`` route so both
        # surfaces emit the same keys.  For a discover folder
        # ``manifold_summary`` reports the on-disk (empty) geometry —
        # ``node_coords == []`` — since the derived per-model layout lives
        # in the fitted safetensors; the text path below still surfaces
        # the derived coords for interactive use.
        from saklas.io.manifolds import manifold_summary
        print(_json.dumps(manifold_summary(mf.folder), indent=2))
        return

    print(f"{ns}/{mf.name}")
    if mf.description:
        print(f"  {mf.description}")
    if mf.is_discover:
        print(f"  fit_mode: {mf.fit_mode} (discover — coords derived per-model)")
        if mf.hyperparams:
            hp = ", ".join(f"{k}={v}" for k, v in sorted(mf.hyperparams.items()))
            print(f"  hyperparams: {hp}")
        print("  nodes:")
        for i, label in enumerate(mf.node_labels):
            role_tail = (
                f"  [role={node_roles_padded[i]}]"
                if node_roles_padded[i] else ""
            )
            if i < len(derived_node_coords):
                coord_str = ", ".join(f"{c:.3g}" for c in derived_node_coords[i])
                print(f"    {label}  ({coord_str}){role_tail}")
            else:
                print(f"    {label}  (coords pending fit){role_tail}")
    else:
        print(f"  domain: {domain_label(mf.domain)}")
        print("  nodes:")
        for label, coords, role in zip(
            mf.node_labels, mf.node_coords, node_roles_padded, strict=True,
        ):
            coord_str = ", ".join(f"{c:g}" for c in coords)
            role_tail = f"  [role={role}]" if role else ""
            print(f"    {label}  ({coord_str}){role_tail}")
    if fitted:
        print("  fitted models:")
        for f in fitted:
            tag = f"{f['method']}, {f['feature_space']}"
            if f.get("fit_mode") and f["fit_mode"] != "authored":
                tag += f", fit_mode={f['fit_mode']}"
            print(f"    {f['stem']}  ({tag})")
            diag = f.get("diagnostics") or {}
            # Per-method one-line summary.  PCA: picked_k + cumvar at k.
            # Spectral: picked_k + gap_magnitude + bandwidth + k_nn.
            if diag and f.get("fit_mode") == "pca":
                cumvar = diag.get("cumulative_variance") or []
                k = diag.get("picked_k")
                if k and 1 <= k <= len(cumvar):
                    print(
                        f"      pca: picked_k={k}, "
                        f"cumvar@k={cumvar[k - 1]:.3f} "
                        f"(threshold={diag.get('threshold', '?')})"
                    )
            elif diag and f.get("fit_mode") == "spectral":
                k = diag.get("picked_k")
                gap = diag.get("gap_magnitude")
                bw = diag.get("bandwidth")
                knn = diag.get("k_nn")
                bits = [f"picked_k={k}"]
                if gap is not None:
                    bits.append(f"gap={gap:.3g}")
                if bw is not None:
                    bits.append(f"bandwidth={bw:.3g}")
                if knn is not None:
                    bits.append(f"k_nn={knn}")
                print(f"      spectral: {', '.join(bits)}")
            if f.get("hyperparams"):
                hp = ", ".join(
                    f"{k}={v}" for k, v in sorted(f["hyperparams"].items())
                )
                print(f"      hyperparams: {hp}")
            # Concept layer profile: the whitened between-node spread per layer
            # (background-σ² units).  Report the peak layers — where this
            # concept's signal concentrates across the stack.
            spread = f.get("node_spread") or {}
            if spread:
                ranked = sorted(
                    ((int(layer), float(v)) for layer, v in spread.items()),
                    key=lambda kv: kv[1],
                    reverse=True,
                )
                peak = ", ".join(f"L{layer}={v:.2f}" for layer, v in ranked[:5])
                print(f"      signal by layer (peak): {peak}")
    else:
        print("  fitted models: (none — run `saklas manifold fit`)")


def _run_pack_install(args: argparse.Namespace) -> None:
    from saklas.io.hf_manifolds import install_manifold

    dst = install_manifold(args.target, args.as_target, force=args.force)
    print(f"Installed {args.target} -> {dst}")


def _run_pack_search(args: argparse.Namespace) -> None:
    import json as _json
    from saklas.io.hf_manifolds import search_manifolds

    try:
        rows = search_manifolds(args.query or None)
    except ImportError as e:
        print(f"saklas manifold search unavailable: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"hf search failed: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
    if getattr(args, "json_output", False):
        print(_json.dumps(rows, indent=2))
        return
    if not rows:
        print("(no matches)")
        return
    for r in rows:
        line = (
            f"{r['name']:<24} {r['namespace']:<12} [hf]          "
            f"{r['domain_label']:<14} {r['node_count']:>3} nodes  "
            f"{', '.join(r['tensor_models']) or '(unfitted)'}"
        )
        if getattr(args, "verbose", False):
            line = (
                f"{r['name']:<24} {r['namespace']:<12} [hf]          "
                f"{r['domain_label']:<14} {r['node_count']:>3} nodes  "
                f"{r.get('description', '')}"
            )
        print(line)


def _run_pack_push(args: argparse.Namespace) -> None:
    from saklas.io.hf import resolve_target_coord
    from saklas.io.hf_manifolds import push_manifold
    from saklas.io.paths import manifold_dir

    ns, name = _split_manifold_ns_name(args.selector)
    folder = manifold_dir(ns, name)
    if not (folder / "manifold.json").exists():
        print(
            f"manifold push: {ns}/{name} not found at {folder}", file=sys.stderr,
        )
        sys.exit(1)

    # The coord follows pack push's resolution: ``--as owner/name`` wins,
    # else ``<whoami>/<name>``.  ``push_manifold`` takes the resolved
    # coord directly (no internal selector machinery), so the runner owns
    # the resolution the way ``cache_ops.push`` does for packs.
    try:
        coord = resolve_target_coord(name, args.as_target)
    except Exception as e:
        print(f"manifold push: {e}", file=sys.stderr)
        sys.exit(2)

    try:
        repo_url, sha = push_manifold(
            folder, coord,
            private=args.private,
            model_scope=args.model,
            variant=args.variant,
            dry_run=args.dry_run,
        )
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        sys.exit(2)

    if sha:
        print(f"Pushed {coord} -> {repo_url} @ {sha[:12]}")
    elif args.dry_run:
        print(f"Dry-run: would push {coord} -> {repo_url}")
    else:
        print(f"Pushed {coord} -> {repo_url}")


def _run_pack_rm(args: argparse.Namespace) -> None:
    from saklas.io.manifolds import remove_manifold_folder

    ns, name = _resolve_manifold_ns_name(args.selector)
    # Bundled (``default/``) manifolds re-materialize on next session
    # init — mirror ``pack rm``'s confirmation guard for the broad/
    # destructive case (here: a bundled folder the user likely didn't
    # mean to nuke).  ``-y`` skips it.
    if ns == "default" and not args.yes:
        print(
            f"refusing to remove bundled manifold {ns}/{name} "
            f"(re-materializes on restart); pass --yes to confirm",
            file=sys.stderr,
        )
        sys.exit(2)
    try:
        result = remove_manifold_folder(ns, name)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
    msg = f"Removed {result['namespace']}/{result['name']}"
    if result.get("rematerializes_on_restart"):
        msg += " (re-materializes on next session start)"
    print(msg)


def _run_pack_clear(args: argparse.Namespace) -> None:
    from saklas.io.manifolds import clear_manifold_tensors

    ns, name = _resolve_manifold_ns_name(args.selector)
    # ``args.model`` is the raw model id; ``clear_manifold_tensors`` does
    # the safe-id conversion at the io boundary, exactly as the pack
    # ``cache_ops.delete_tensors`` path does (it passes ``args.model``
    # straight through to ``enumerate_variants``).
    try:
        n = clear_manifold_tensors(ns, name, args.model, variant=args.variant)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
    print(f"Deleted {n} files")


def _run_pack_refresh(args: argparse.Namespace) -> None:
    from saklas.io.manifolds import refresh_manifold

    ns, name = _resolve_manifold_ns_name(args.selector)
    # ``args.model`` is the raw model id; ``refresh_manifold`` converts to
    # a safe id at the io boundary (via ``clear_manifold_tensors``), the
    # same convention ``cache_ops.refresh``'s scoped path uses.
    try:
        tier = refresh_manifold(ns, name, model_scope=args.model)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
    if tier == "scoped":
        print(f"Refreshed {ns}/{name} ({args.model} tensors cleared — re-fits on next use)")
    elif tier == "skipped":
        print(f"{ns}/{name}: local source, nothing to refresh")
    elif tier == "bundled":
        print(f"Refreshed {ns}/{name} (bundled — re-materialized from package data)")
    else:  # "hf"
        print(f"Refreshed {ns}/{name} (re-pulled from HF)")


def _run_pack_export(args: argparse.Namespace) -> None:
    """Export a fitted 2-node ``pca`` manifold to an interchange format (gguf).

    Folds the manifold down to a single steering direction
    (:func:`~saklas.core.capture.folded_directions`) and writes a
    llama.cpp control-vector GGUF.
    """
    fmt = getattr(args, "format", None)
    if fmt != "gguf":
        print(f"Unknown export format: {fmt}", file=sys.stderr)
        sys.exit(2)
    from saklas.io.cache_ops import export_gguf_manifold

    ns, name = _resolve_manifold_ns_name(args.name)
    try:
        written = export_gguf_manifold(
            ns, name,
            model_scope=args.model,
            output=args.output,
            model_hint=args.model_hint,
        )
    except (FileNotFoundError, RuntimeError) as e:
        print(f"manifold export failed: {e}", file=sys.stderr)
        sys.exit(1)
    for p in written:
        print(f"Wrote {p}")


# The nine manifold *lifecycle* verbs (pure-IO over ~/.saklas/manifolds/,
# addressed by (namespace, name); install / share / inspect / remove).
_PACK_RUNNERS = {
    "ls":       _run_pack_ls,
    "show":     _run_pack_show,
    "install":  _run_pack_install,
    "search":   _run_pack_search,
    "push":     _run_pack_push,
    "rm":       _run_pack_rm,
    "clear":    _run_pack_clear,
    "refresh":  _run_pack_refresh,
    "export":   _run_pack_export,
}


@_saklas_error_exit
def _run_pack(args: argparse.Namespace) -> None:
    """Dispatch ``saklas pack <verb>`` (the manifold lifecycle verbs)."""
    cmd = getattr(args, "pack_cmd", None)
    if cmd is None:
        print("usage: saklas pack <verb> [...]")
        print()
        width = max(len(v) for v, _ in _PACK_VERBS)
        for v, desc in _PACK_VERBS:
            print(f"  {v:<{width}}  {desc}")
        print()
        print("Run `saklas pack <verb> -h` for verb-specific options.")
        sys.exit(0)
    runner = _PACK_RUNNERS.get(cmd)
    if runner is None:
        print(f"unknown pack verb {cmd!r}", file=sys.stderr)
        sys.exit(2)
    runner(args)
