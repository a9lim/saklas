"""``saklas sae <verb>`` runners and dispatch table."""

from __future__ import annotations

import argparse
import sys

import saklas.cli.runners as _pkg
from saklas.cli.parsers import _SAE_VERBS
from saklas.cli.runners.shared import _saklas_error_exit


def _load_sae_training_corpus(args: argparse.Namespace) -> tuple[list[str], str]:
    import json as _json
    import math
    from pathlib import Path

    target_docs = max(1, math.ceil(args.tokens / args.seq_len))
    if args.corpus is not None:
        path = Path(args.corpus)
        if not path.exists():
            print(f"sae train: corpus file not found: {path}", file=sys.stderr)
            sys.exit(2)
        docs: list[str] = []
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("{"):
                    try:
                        payload = _json.loads(line)
                        line = str(payload.get("text", "")) or line
                    except _json.JSONDecodeError:
                        pass
                docs.append(line)
                if len(docs) >= target_docs:
                    break
        if not docs:
            print("sae train: corpus has no non-empty documents", file=sys.stderr)
            sys.exit(2)
        return docs, f"file:{path.name}"
    from saklas.core.jlens import JacobianLensError
    from saklas.io.lens import DEFAULT_LENS_CORPUS, stream_default_lens_corpus

    repo, config = DEFAULT_LENS_CORPUS
    print(f"Streaming {target_docs} documents from {repo} ({config})...")
    try:
        return stream_default_lens_corpus(target_docs)
    except JacobianLensError as exc:
        print(f"sae train: {exc}", file=sys.stderr)
        sys.exit(2)


def _run_sae_train(args: argparse.Namespace) -> None:
    import json
    from saklas.core.session import SaklasSession

    if args.learning_rate <= 0 or args.l1 < 0 or args.dead_threshold < 0:
        print("sae train: learning-rate must be > 0 and sparsity values >= 0", file=sys.stderr)
        sys.exit(2)
    docs, spec = _load_sae_training_corpus(args)
    _pkg._print_startup(args)
    with SaklasSession.from_pretrained(
        args.model, device=args.device, quantize=args.quantize, probes=[],
    ) as session:
        _pkg._print_model_info(session)
        layer = args.layer
        if layer is None:
            layer = round(0.65 * max(len(session.layers) - 1, 0))
        result = session.train_sae(
            args.name, docs, layer=layer, corpus_spec=spec,
            tokens=args.tokens, seq_len=args.seq_len, batch_size=args.batch_size,
            d_sae=args.width, expansion=args.expansion,
            learning_rate=args.learning_rate, l1_coefficient=args.l1,
            dead_feature_threshold=args.dead_threshold, seed=args.seed,
            force=args.force, on_progress=lambda message: print(f"  {message}"),
        )
    if args.json_output:
        print(json.dumps(result, indent=2))
        return
    metrics = result["metrics"]
    print(
        f"Trained {result['source']} for {args.model}: L{layer}, "
        f"{metrics['d_sae']} features, {metrics['tokens_trained']:,} tokens"
    )
    print(f"Artifact: {result['artifact']}")


def _run_sae_fetch(args: argparse.Namespace) -> None:
    import json
    from saklas.core.session import SaklasSession

    if not args.source.startswith("saelens:"):
        print("sae fetch: source must be saelens:RELEASE", file=sys.stderr)
        sys.exit(2)
    release = args.source[len("saelens:"):]
    if not release:
        print("sae fetch: release must not be empty", file=sys.stderr)
        sys.exit(2)
    _pkg._print_startup(args)
    with SaklasSession.from_pretrained(
        args.model, device=args.device, quantize=args.quantize, probes=[],
    ) as session:
        _pkg._print_model_info(session)
        info = session.load_sae(release, layer=args.layer)
    if args.json_output:
        print(json.dumps({"model": args.model, "source": args.source, **info}, indent=2))
    else:
        print(
            f"Fetched {args.source} for {args.model}: "
            f"L{info['layer']}, {info['width']} features"
        )
        print("Provider weights remain in the Hugging Face cache; binding is active.")


def _run_sae_ls(args: argparse.Namespace) -> None:
    import json
    from saklas.io.sae import list_sae_sources

    rows = list_sae_sources(args.model)
    if args.json_output:
        print(json.dumps(rows, indent=2))
        return
    if not rows:
        print(f"No SAE sources for {args.model}.")
        return
    for row in rows:
        marker = "*" if row["active"] else " "
        print(f"{marker} {row['source']}  L{row['layer']}  {row['features']} features")


def _resolve_active_sae_source(model: str, source: str | None) -> str | None:
    if source is not None:
        return source
    from saklas.io.sae import load_active_sae_source

    active = load_active_sae_source(model)
    if active is None:
        return None
    return (
        f"local:{active['name']}"
        if active["kind"] == "local"
        else f"saelens:{active['name']}"
    )


def _run_sae_show(args: argparse.Namespace) -> None:
    import json
    from saklas.io.sae import load_active_sae_source, load_sae_metadata
    from saklas.io.sae_artifacts import load_local_sae_manifest

    source = _resolve_active_sae_source(args.model, args.source)
    if source is None:
        print(f"no SAE source for {args.model}", file=sys.stderr)
        sys.exit(1)
    if source.startswith("local:"):
        payload = load_local_sae_manifest(args.model, source[6:])
    elif source.startswith("saelens:"):
        payload = load_sae_metadata(args.model, source[len("saelens:"):])
    else:
        print("sae show: source must be local:NAME or saelens:RELEASE", file=sys.stderr)
        sys.exit(2)
    if payload is None:
        print(f"no SAE source {source} for {args.model}", file=sys.stderr)
        sys.exit(1)
    active = load_active_sae_source(args.model)
    out = {
        "model": args.model,
        "source": source,
        "active": bool(
            active is not None
            and (
                (source.startswith("local:") and active["kind"] == "local" and active["name"] == source[6:])
                or (source.startswith("saelens:") and active["kind"] == "saelens" and active["name"] == source[len("saelens:"):])
            )
        ),
        **payload,
    }
    if args.json_output:
        print(json.dumps(out, indent=2))
        return
    print(f"SAE {source} for {args.model}{' (active)' if out['active'] else ''}")
    print(f"  layer:    {payload['layer']}")
    print(f"  features: {payload.get('d_sae', payload.get('width'))}")
    if source.startswith("local:"):
        print(f"  tokens:   {payload['tokens_trained']}")
        print(f"  corpus:   {payload['corpus_spec']}")
    else:
        print(f"  revision: {payload['revision']}")
        print("  payload:  Hugging Face cache (provider-owned)")


def _run_sae_use(args: argparse.Namespace) -> None:
    from saklas.io.sae import use_sae_source

    use_sae_source(args.model, args.source)
    print(f"Active SAE for {args.model}: {args.source}")


def _run_sae_rm(args: argparse.Namespace) -> None:
    from saklas.io.sae import remove_sae_binding
    from saklas.io.sae_artifacts import remove_local_sae

    source = _resolve_active_sae_source(args.model, args.source)
    if source is None:
        print(f"no SAE source for {args.model}", file=sys.stderr)
        sys.exit(1)
    if not args.yes:
        action = "Remove local artifact" if source.startswith("local:") else "Forget external binding"
        answer = input(f"{action} {source}? [y/N] ").strip().lower()
        if answer not in {"y", "yes"}:
            print("Aborted.")
            return
    if source.startswith("local:"):
        removed = remove_local_sae(args.model, source[6:])
    elif source.startswith("saelens:"):
        removed = remove_sae_binding(args.model, source[len("saelens:"):])
    else:
        print("sae rm: source must be local:NAME or saelens:RELEASE", file=sys.stderr)
        sys.exit(2)
    if not removed:
        print(f"no SAE source {source} for {args.model}", file=sys.stderr)
        sys.exit(1)
    if source.startswith("local:"):
        print(f"Removed Saklas-owned SAE {source}.")
    else:
        print("Forgot SAELens binding; Hugging Face cache was not modified.")


_SAE_RUNNERS = {
    "train": _run_sae_train,
    "fetch": _run_sae_fetch,
    "ls": _run_sae_ls,
    "show": _run_sae_show,
    "use": _run_sae_use,
    "rm": _run_sae_rm,
}


@_saklas_error_exit
def _run_sae(args: argparse.Namespace) -> None:
    cmd = getattr(args, "sae_cmd", None)
    if cmd is None:
        print("usage: saklas sae <verb> [...]")
        print()
        width = max(len(verb) for verb, _ in _SAE_VERBS)
        for verb, desc in _SAE_VERBS:
            print(f"  {verb:<{width}}  {desc}")
        sys.exit(0)
    runner = _SAE_RUNNERS.get(cmd)
    if runner is None:
        print(f"unknown sae verb {cmd!r}", file=sys.stderr)
        sys.exit(2)
    runner(args)
