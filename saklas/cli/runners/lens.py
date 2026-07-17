"""``saklas lens <verb>`` runners (per-model Jacobian lens) and dispatch table."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import saklas.cli.runners as _pkg
from saklas.cli.parsers import _LENS_VERBS
from saklas.cli.runners.shared import _saklas_error_exit


#: Documents are sliced to this many characters before tokenization — the fit
#: truncates to --seq-len tokens anyway, so tokenizing a full web page is waste.
#: (The default-corpus streamer lives in ``io.lens`` — shared with the server's
#: fit route — and applies the same slice.)
_LENS_DOC_CHARS = 4000


def _parse_layer_list(raw: "str | None") -> "list[int] | str | None":
    if raw is None:
        return None
    lowered = raw.strip().lower()
    if lowered in {"workspace", "band", "sample", "all"}:
        return lowered
    try:
        layers = [int(part) for part in raw.split(",") if part.strip() != ""]
    except ValueError:
        print(
            f"lens: bad --layers value {raw!r} "
            "(want e.g. 12,24,36 or workspace)",
            file=sys.stderr,
        )
        sys.exit(2)
    if not layers:
        print("lens: --layers must name at least one source layer", file=sys.stderr)
        sys.exit(2)
    return layers


def _load_lens_corpus(args: argparse.Namespace) -> tuple[list[str], str]:
    """Return ``(documents, corpus_spec)`` for ``lens fit``.

    ``--corpus FILE`` reads one document per line (a JSON object line with a
    ``text`` field also works). Unset, streams the default web-text sample
    via the optional ``datasets`` dependency.
    """
    import json as _json

    n = int(args.prompts)
    if args.corpus is not None:
        path = Path(args.corpus)
        if not path.exists():
            print(f"lens fit: corpus file not found: {path}", file=sys.stderr)
            sys.exit(2)
        docs: list[str] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("{"):
                    try:
                        obj = _json.loads(line)
                        line = str(obj.get("text", "")) or line
                    except _json.JSONDecodeError:
                        pass
                docs.append(line[:_LENS_DOC_CHARS])
                if len(docs) >= n:
                    break
        return docs, f"file:{path.name}"
    from saklas.core.jlens import JacobianLensError
    from saklas.io.lens import DEFAULT_LENS_CORPUS, stream_default_lens_corpus

    repo, config = DEFAULT_LENS_CORPUS
    print(f"Streaming {n} documents from {repo} ({config})...")
    try:
        return stream_default_lens_corpus(n)
    except JacobianLensError as e:
        print(f"lens fit: {e}", file=sys.stderr)
        sys.exit(2)


def _lens_fit_source_preflight_matches(
    sidecar: dict[str, object],
    requested_layers: "list[int] | str | None",
) -> bool:
    raw_layers = sidecar.get("source_layers", [])
    if not isinstance(raw_layers, list):
        return False
    source_layers = [int(layer) for layer in raw_layers]
    if not source_layers:
        return False
    if isinstance(requested_layers, list):
        return set(source_layers) >= set(requested_layers)
    raw_count = sidecar.get("model_layer_count")
    if not isinstance(raw_count, int) or raw_count < 2:
        return False
    final_idx = raw_count - 1
    if requested_layers is None or requested_layers == "all":
        return set(source_layers) >= set(range(final_idx))
    if requested_layers in {"workspace", "band"}:
        from saklas.core.model import workspace_layer_indices

        expected = set(workspace_layer_indices(range(final_idx), raw_count))
        return set(source_layers) >= expected
    return False


def _try_lens_fit_noop_preflight(
    args: argparse.Namespace,
    requested_layers: "list[int] | str | None",
    *,
    docs: list[str] | None = None,
) -> bool:
    """Serialize the complete model-free proof/reap/report transaction."""
    if args.force:
        return False
    from saklas.io.lens import lens_fit_lock

    with lens_fit_lock(args.model):
        return _try_lens_fit_noop_preflight_locked(
            args, requested_layers, docs=docs,
        )


def _try_lens_fit_noop_preflight_locked(
    args: argparse.Namespace,
    requested_layers: "list[int] | str | None",
    *,
    docs: list[str] | None = None,
) -> bool:
    """Prove an existing fit is exact without loading model weights."""
    import hashlib

    from saklas.core.jlens import DEFAULT_SEQ_LEN
    from saklas.core.model import model_source_fingerprint
    from saklas.io.lens import (
        lens_artifact_size,
        lens_paths,
        lens_payloads_match,
        load_lens_sidecar,
        remove_subsumed_lens_checkpoint,
        resolved_default_lens_corpus_spec,
    )

    sidecar = load_lens_sidecar(args.model)
    if sidecar is None:
        return False
    source_fp = model_source_fingerprint(
        args.model, quantize=args.quantize, device=args.device,
    )
    if (
        source_fp is None
        or sidecar.get("model_source_fingerprint") != source_fp
        or sidecar.get("seq_len") != (args.seq_len or DEFAULT_SEQ_LEN)
        or not _lens_fit_source_preflight_matches(sidecar, requested_layers)
    ):
        return False
    if docs is not None:
        raw_sha = hashlib.sha256(repr(docs).encode("utf-8")).hexdigest()
        if (
            sidecar.get("raw_corpus_sha256") != raw_sha
            or sidecar.get("raw_prompt_count") != len(docs)
        ):
            return False
    else:
        if args.corpus is not None:
            return False
        try:
            _revision, expected_spec = resolved_default_lens_corpus_spec()
        except Exception:
            return False
        if (
            sidecar.get("corpus_spec") != expected_spec
            or sidecar.get("raw_prompt_count") != int(args.prompts)
        ):
            return False
    usable_count = int(sidecar.get("usable_prompt_count", -1))
    if usable_count < 0 or int(sidecar.get("n_prompts", -1)) < usable_count:
        return False
    _ts_path, sidecar_path = lens_paths(args.model)
    if not lens_payloads_match(args.model, sidecar):
        return False
    remove_subsumed_lens_checkpoint(
        args.model, verified_final_sidecar=sidecar,
    )
    size_mb = lens_artifact_size(args.model, sidecar) / 1024**2
    source_layers = [int(layer) for layer in sidecar["source_layers"]]
    print(
        f"Fitted Jacobian lens: {len(source_layers)} layers, "
        f"{sidecar.get('n_prompts', usable_count)} prompts, "
        f"d_model={sidecar.get('d_model', '?')}"
    )
    print("Already fitted for this corpus and exact model source — nothing to do.")
    print(f"Artifact: {sidecar_path} ({size_mb:.0f} MB across layer shards)")
    return True


def _run_lens_fit(args: argparse.Namespace) -> None:
    from saklas.core.session import SaklasSession
    from saklas.io.lens import lens_artifact_size, lens_paths, load_lens_sidecar

    requested_layers = _parse_layer_list(getattr(args, "layers", None))
    if requested_layers == "sample":
        print(
            "lens fit: --layers sample is not a wall-time optimization; "
            "use --layers workspace or an explicit comma-separated band",
            file=sys.stderr,
        )
        sys.exit(2)
    if args.corpus is None and _try_lens_fit_noop_preflight(
        args, requested_layers,
    ):
        return
    docs, spec = _load_lens_corpus(args)
    if args.corpus is not None and _try_lens_fit_noop_preflight(
        args, requested_layers, docs=docs,
    ):
        return
    _pkg._print_startup(args)
    with SaklasSession.from_pretrained(
        args.model, device=args.device, quantize=args.quantize, probes=[],
    ) as session:
        _pkg._print_model_info(session)
        if args.force:
            print("Refitting from zero (-f).")
        lens = session.fit_jlens(
            docs,
            corpus_spec=spec,
            source_layers=requested_layers,
            dim_batch=args.dim_batch,
            prompt_batch=args.prompt_batch,
            seq_len=args.seq_len,
            force=args.force,
            checkpoint_every=args.checkpoint_every,
            on_progress=lambda m: print(f"  {m}"),
        )
    _ts_path, sidecar_path = lens_paths(args.model)
    sidecar = load_lens_sidecar(args.model)
    size_mb = (
        lens_artifact_size(args.model, sidecar) / 1024**2
        if sidecar is not None else 0.0
    )
    print(
        f"Fitted Jacobian lens: {len(lens.source_layers)} layers, "
        f"{lens.n_prompts} prompts, d_model={lens.d_model}"
    )
    print(f"Artifact: {sidecar_path} ({size_mb:.0f} MB across layer shards)")


def _run_lens_fetch(args: argparse.Namespace) -> None:
    import json as _json

    from saklas.io.lens_sources import fetch_neuronpedia_lens

    if args.source != "neuronpedia":
        print("lens fetch: source must be neuronpedia", file=sys.stderr)
        sys.exit(2)
    if not args.json_output:
        print(f"Fetching official Jacobian lens for {args.model} into Hugging Face cache...")
    binding = fetch_neuronpedia_lens(
        args.model,
        repo_id=args.repo,
        revision=args.revision,
        force=args.force,
    )
    payload = binding.to_json()
    if args.json_output:
        print(_json.dumps(payload, indent=2))
        return
    print(
        f"Fetched neuronpedia for {args.model}: {len(binding.source_layers)} layers, "
        f"{binding.n_prompts} prompts"
    )
    print(f"  provider: {binding.repo_id}@{binding.repo_revision}")
    print(f"  checkpoint: {binding.checkpoint}")
    print("Provider payload remains in the Hugging Face cache; binding is active.")


def _run_lens_ls(args: argparse.Namespace) -> None:
    import json as _json

    from saklas.io.lens_sources import list_lens_sources

    rows = list_lens_sources(args.model)
    if args.json_output:
        print(_json.dumps(rows, indent=2))
        return
    if not rows:
        print(f"No Jacobian-lens sources for {args.model}.")
        return
    for row in rows:
        marker = "*" if row["active"] else " "
        print(f"{marker} {row['source']}")


def _run_lens_use(args: argparse.Namespace) -> None:
    from saklas.io.lens_sources import use_lens_source

    use_lens_source(args.model, args.source)
    print(f"Active Jacobian lens for {args.model}: {args.source}")


def _run_lens_show(args: argparse.Namespace) -> None:
    import json as _json

    from saklas.io.lens import (
        lens_artifact_size,
        lens_paths,
        load_lens_sidecar,
        load_local_lens_sidecar,
    )
    from saklas.io.lens_sources import (
        load_active_lens_source,
        load_external_lens_binding,
        external_lens_sidecar,
        lens_binding_path,
    )

    source = args.source
    active = load_active_lens_source(args.model)
    if source is None:
        sidecar = load_lens_sidecar(args.model)
        selected = active
    elif source.startswith("local:"):
        if source != "local:default":
            print(f"no local lens source {source}", file=sys.stderr)
            sys.exit(1)
        sidecar = load_local_lens_sidecar(args.model)
        selected = {"kind": "local", "name": "default"}
    elif source == "neuronpedia":
        binding = load_external_lens_binding(args.model)
        sidecar = None if binding is None else external_lens_sidecar(binding)
        selected = {"kind": "huggingface", "name": "neuronpedia"}
    else:
        print("lens show: source must be local:default or neuronpedia", file=sys.stderr)
        sys.exit(2)
    if sidecar is None:
        print(f"no Jacobian lens for {args.model}", file=sys.stderr)
        sys.exit(1)
    external = isinstance(sidecar.get("_source"), dict)
    if external:
        sidecar_path = lens_binding_path(args.model, "neuronpedia")
        size_mb = None
    else:
        _ts_path, sidecar_path = lens_paths(args.model)
        size_mb = lens_artifact_size(args.model, sidecar) / 1024**2
    source_layers = [int(layer) for layer in sidecar["source_layers"]]
    if getattr(args, "json_output", False):
        print(_json.dumps({
            "model": args.model,
            "path": str(sidecar_path),
            "size_mb": None if size_mb is None else round(size_mb, 1),
            "active": bool(
                active is not None and selected is not None
                and active["kind"] == selected["kind"]
                and active["name"] == selected["name"]
            ),
            **sidecar,
        }, indent=2))
        return
    print(f"Jacobian lens for {args.model}")
    print(f"  layers:   {source_layers[0]}..{source_layers[-1]} "
          f"({len(source_layers)})")
    print(f"  d_model:  {sidecar['d_model']}")
    print(f"  prompts:  {sidecar.get('n_prompts', '?')}")
    print(f"  corpus:   {sidecar.get('corpus_spec', '?')} "
          f"(sha256 {str(sidecar.get('corpus_sha256', ''))[:12]}…)")
    print(f"  seq_len:  {sidecar.get('seq_len', '?')}, "
          f"skip_first: {sidecar.get('skip_first_positions', '?')}")
    if external:
        source_meta = sidecar["_source"]
        print(f"  source:   {source_meta['provider']} ({source_meta['repo_id']}@{source_meta['repo_revision']})")
        print(f"  binding:  {sidecar_path}")
        print("  payload:  Hugging Face cache (provider-owned)")
    else:
        assert size_mb is not None
        print(f"  artifact: {sidecar_path} ({size_mb:.0f} MB across layer shards)")


def _run_lens_top(args: argparse.Namespace) -> None:
    import json as _json

    from saklas.core.session import SaklasSession

    layers = _parse_layer_list(args.layers)
    _pkg._print_startup(args)
    with SaklasSession.from_pretrained(
        args.model, device=args.device, quantize=args.quantize, probes=[],
    ) as session:
        _pkg._print_model_info(session)
        lens = session.jlens
        if lens is None:
            print(
                f"no lens for {args.model} — run `saklas lens fetch "
                f"{args.model}` or `saklas lens fit {args.model}`",
                file=sys.stderr,
            )
            sys.exit(1)
        if layers is None:
            layers = list(lens.source_layers)
        out, agg = session.jlens_readout(
            args.prompt, layers=layers, positions=args.position,
            top_k=args.top_k, aggregate=True,
        )
    if getattr(args, "json_output", False):
        print(_json.dumps({
            "model": args.model,
            "prompt": args.prompt,
            "positions": args.position or [-1],
            # Layer-aggregated view across every displayed layer:
            # per-layer softmax → mean-probability strength +
            # probability-mass-weighted depth center of mass, strength-descending.
            "aggregate": [
                [
                    {
                        "token": t,
                        "strength": round(s, 6),
                        "com": round(c, 4),
                        "spread": round(sp, 4),
                    }
                    for t, s, c, sp in rows
                ]
                for rows in agg
            ],
            "layers": {
                str(layer): [
                    [{"token": t, "logprob": round(lp, 4)} for t, lp in row]
                    for row in rows
                ]
                for layer, rows in out.items()
            },
        }, indent=2))
        return
    positions = args.position or [-1]
    for pos_idx, pos in enumerate(positions):
        if len(positions) > 1:
            print(f"\nposition {pos}:")
        print("  aggregate (all displayed layers):")
        for t, s, c, sp in agg[pos_idx]:
            tok = t.strip() or repr(t)
            print(
                f"    {tok:<20} strength {s:.3f}   com {c:.2f} ±{sp:.2f}"
            )
        print("  per-layer:")
        for layer in sorted(out):
            row = out[layer][pos_idx]
            toks = "  ".join(f"{t.strip() or repr(t)}" for t, _ in row)
            print(f"  L{layer:>3}  {toks}")


def _run_lens_decompose(args: argparse.Namespace) -> None:
    import json as _json

    from saklas.core.session import SaklasSession

    layers = _parse_layer_list(args.layers)
    _pkg._print_startup(args)
    with SaklasSession.from_pretrained(
        args.model, device=args.device, quantize=args.quantize, probes=[],
    ) as session:
        _pkg._print_model_info(session)
        out = session.jspace_decompose(
            args.selector, k=args.top_k, layers=layers,
        )
    shares = [share for share, _ in out.values()]
    mean_share = sum(shares) / len(shares)
    if getattr(args, "json_output", False):
        print(_json.dumps({
            "selector": args.selector,
            "model": args.model,
            "k": args.top_k,
            "mean_share": round(mean_share, 4),
            "layers": {
                str(layer): {
                    "share": round(share, 4),
                    "tokens": [
                        {"token": t, "coeff": round(c, 4)} for t, c in tokens
                    ],
                }
                for layer, (share, tokens) in out.items()
            },
        }, indent=2))
        return
    print(f"J-space share of '{args.selector}' (k={args.top_k}):")
    print(f"  mean over {len(out)} layers: {mean_share:.1%}")
    for layer in sorted(out):
        share, tokens = out[layer]
        head = "  ".join(f"{t.strip() or repr(t)}" for t, _ in tokens[:6])
        print(f"  L{layer:>3}  {share:>6.1%}  {head}")


def _run_lens_rm(args: argparse.Namespace) -> None:
    from saklas.io.lens import lens_paths, remove_lens
    from saklas.io.lens_sources import (
        lens_binding_path,
        load_active_lens_source,
        remove_external_lens_binding,
    )

    source = args.source
    if source is None:
        active = load_active_lens_source(args.model)
        if active is None:
            print(f"no lens source for {args.model}", file=sys.stderr)
            sys.exit(1)
        source = (
            f"local:{active['name']}"
            if active["kind"] == "local" else active["name"]
        )
    local = source.startswith("local:")
    if local and source != "local:default":
        print(f"no local lens source {source}", file=sys.stderr)
        sys.exit(1)
    if not local and source != "neuronpedia":
        print("lens rm: source must be local:default or neuronpedia", file=sys.stderr)
        sys.exit(2)
    sidecar_path = (
        lens_paths(args.model)[1]
        if local else lens_binding_path(args.model, "neuronpedia")
    )
    if not sidecar_path.exists():
        print(f"no lens source {source} for {args.model}", file=sys.stderr)
        sys.exit(1)
    if not args.yes:
        action = "Remove local artifact" if local else "Forget external binding"
        answer = input(f"{action} {sidecar_path}? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("Aborted.")
            return
    removed = (
        remove_lens(args.model)
        if local else remove_external_lens_binding(args.model)
    )
    if not removed:
        print(f"no lens source {source} for {args.model}", file=sys.stderr)
        sys.exit(1)
    if local:
        print(f"Removed Saklas-owned lens {source} for {args.model}.")
    else:
        print("Forgot Neuronpedia binding; Hugging Face cache was not modified.")


_LENS_RUNNERS = {
    "fit":       _run_lens_fit,
    "fetch":     _run_lens_fetch,
    "ls":        _run_lens_ls,
    "show":      _run_lens_show,
    "use":       _run_lens_use,
    "top":       _run_lens_top,
    "decompose": _run_lens_decompose,
    "rm":        _run_lens_rm,
}


@_saklas_error_exit
def _run_lens(args: argparse.Namespace) -> None:
    """Dispatch ``saklas lens <verb>`` (the per-model Jacobian lens)."""
    cmd = getattr(args, "lens_cmd", None)
    if cmd is None:
        print("usage: saklas lens <verb> [...]")
        print()
        width = max(len(v) for v, _ in _LENS_VERBS)
        for v, desc in _LENS_VERBS:
            print(f"  {v:<{width}}  {desc}")
        print()
        print("Run `saklas lens <verb> -h` for verb-specific options.")
        sys.exit(0)
    runner = _LENS_RUNNERS.get(cmd)
    if runner is None:
        print(f"unknown lens verb {cmd!r}", file=sys.stderr)
        sys.exit(2)
    runner(args)
