"""``saklas template <verb>`` runners (the templated-completion artifact)."""

from __future__ import annotations

import argparse
import sys

from saklas.cli.parsers import _TEMPLATE_VERBS
from saklas.cli.runners.shared import _saklas_error_exit, _split_manifold_ns_name


def _normalize_context_entry(entry: object) -> dict[str, object]:
    """Accept either a multi-turn ``{turns, assistant}`` context or the
    single-turn sugar ``{user, assistant}`` and return the canonical shape."""
    if not isinstance(entry, dict):
        raise ValueError("each context must be an object")
    if "turns" in entry:
        return {"turns": entry["turns"], "assistant": entry.get("assistant")}
    if "user" in entry:                      # single-turn sugar
        return {
            "turns": [{"role": "user", "content": entry["user"]}],
            "assistant": entry.get("assistant"),
        }
    raise ValueError(
        "each context needs either 'turns' (multi-turn) or 'user' "
        "(single-turn sugar), plus 'assistant'"
    )


def _run_template_create(args: argparse.Namespace) -> None:
    """``saklas template create`` — author a standalone template artifact.

    Pure-IO (no model): reads the ``--contexts`` JSON file (multi-turn contexts
    or the single-turn ``{user, assistant}`` sugar), validates the slot/value
    invariants, and writes ``~/.saklas/templates/<ns>/<name>/template.json``.
    """
    import json as _json

    from saklas.io.templates import TemplateFormatError, create_template_folder

    try:
        with open(args.contexts_file) as f:
            raw = _json.load(f)
    except (OSError, ValueError) as e:
        print(f"template create: cannot read --contexts: {e}", file=sys.stderr)
        sys.exit(1)
    if not isinstance(raw, list) or not raw:
        print(
            "template create: --contexts must be a non-empty JSON list of "
            'contexts ({"turns": [...], "assistant": ...} or {"user": ..., '
            '"assistant": ...})',
            file=sys.stderr,
        )
        sys.exit(2)
    try:
        contexts = [_normalize_context_entry(e) for e in raw]
    except ValueError as e:
        print(f"template create: {e}", file=sys.stderr)
        sys.exit(2)

    namespace, name = _split_manifold_ns_name(args.name)
    try:
        create_template_folder(
            namespace, name,
            slot=args.slot,
            values=list(args.values),
            contexts=contexts,
            description=args.description,
            force=args.force,
        )
    except FileExistsError:
        print(
            f"template create: {namespace}/{name} already exists "
            f"-- pass -f/--force to overwrite",
            file=sys.stderr,
        )
        sys.exit(1)
    except TemplateFormatError as e:
        print(f"template create failed: {e}", file=sys.stderr)
        sys.exit(1)

    print(
        f"wrote template {namespace}/{name} ({len(args.values)} values x "
        f"{len(contexts)} contexts)"
    )
    print(f"  -> `saklas template score {namespace}/{name} -m MODEL` to read the "
          f"value distribution")


def _run_template_ls(args: argparse.Namespace) -> None:
    import json as _json

    from saklas.io.templates import iter_template_folders

    rows = list(iter_template_folders())
    if getattr(args, "json_output", False):
        print(_json.dumps([t.summary() for t in rows], indent=2))
        return
    if not rows:
        print("no templates installed under ~/.saklas/templates/")
        return
    for t in rows:
        ns = t.path.parent.name if t.path else "?"
        print(
            f"  {ns}/{t.name}  [slot {t.slot!r}, {len(t.values)} values x "
            f"{len(t.contexts)} contexts]"
        )
        if t.description:
            print(f"    {t.description}")


def _run_template_show(args: argparse.Namespace) -> None:
    import json as _json

    from saklas.io.templates import (
        AmbiguousTemplateError, TemplateNotFoundError, resolve_template,
    )

    try:
        t = resolve_template(args.name)
    except (TemplateNotFoundError, AmbiguousTemplateError) as e:
        print(f"template show: {e}", file=sys.stderr)
        sys.exit(1)
    if getattr(args, "json_output", False):
        payload = t.summary()
        payload["contexts"] = [
            {"turns": list(c.turns), "assistant": c.assistant} for c in t.contexts
        ]
        print(_json.dumps(payload, indent=2))
        return
    print(f"{t.name}  (slot {t.slot!r})")
    if t.description:
        print(f"  {t.description}")
    print(f"  values ({len(t.values)}): {', '.join(t.values)}")
    print(f"  contexts ({len(t.contexts)}):")
    for i, c in enumerate(t.contexts):
        turns = " / ".join(f"{turn['role']}: {turn['content']}" for turn in c.turns)
        print(f"    [{i}] {turns}")
        print(f"        assistant: {c.assistant}")


def _run_template_rm(args: argparse.Namespace) -> None:
    from saklas.io.templates import (
        AmbiguousTemplateError, TemplateNotFoundError, remove_template_folder,
        resolve_template,
    )

    try:
        t = resolve_template(args.name)
    except (TemplateNotFoundError, AmbiguousTemplateError) as e:
        print(f"template rm: {e}", file=sys.stderr)
        sys.exit(1)
    namespace = t.path.parent.name if t.path else "local"
    name = t.name
    if not args.yes:
        print(
            f"remove template {namespace}/{name}? pass -y/--yes to confirm",
            file=sys.stderr,
        )
        sys.exit(2)
    if remove_template_folder(namespace, name):
        print(f"Removed template {namespace}/{name}")
    else:
        print(f"template rm: no template {namespace}/{name}", file=sys.stderr)
        sys.exit(1)


def _run_template_score(args: argparse.Namespace) -> None:
    """``saklas template score`` — the restricted-choice value distribution."""
    import json as _json

    from saklas.core.scoring import score_template
    from saklas.core.session import SaklasSession
    from saklas.io.templates import (
        AmbiguousTemplateError, TemplateNotFoundError, resolve_template,
    )

    try:
        tmpl = resolve_template(args.name)
    except (TemplateNotFoundError, AmbiguousTemplateError) as e:
        print(f"template score: {e}", file=sys.stderr)
        sys.exit(1)

    with SaklasSession.from_pretrained(
        args.model, device=args.device, quantize=args.quantize,
    ) as session:
        per_ctx = score_template(session, tmpl, steering=args.steer)

    if getattr(args, "json_output", False):
        print(_json.dumps({
            "template": tmpl.name,
            "model": args.model,
            "steering": args.steer,
            "contexts": [sc.to_dict() for sc in per_ctx],
        }, indent=2))
        return

    by = getattr(args, "by", "sum")
    steer_note = f"  (steering: {args.steer})" if args.steer else ""
    print(f"{tmpl.name} on {args.model}{steer_note}")
    for i, sc in enumerate(per_ctx):
        print(f"\n  context [{i}]:")
        for c in sc.ranked(by=by):
            prob = c.prob_sum if by == "sum" else c.prob_mean
            bar = "#" * int(round(prob * 30))
            print(
                f"    {c.label:14s} P={prob:6.3f}  "
                f"sum={c.sum_logprob:8.3f}  mean={c.mean_logprob:8.3f}  "
                f"n={c.n_tokens}  {bar}"
            )


_TEMPLATE_RUNNERS = {
    "create": _run_template_create,
    "ls":     _run_template_ls,
    "show":   _run_template_show,
    "score":  _run_template_score,
    "rm":     _run_template_rm,
}


@_saklas_error_exit
def _run_template(args: argparse.Namespace) -> None:
    """Dispatch ``saklas template <verb>`` (the templated-completion artifact)."""
    cmd = getattr(args, "template_cmd", None)
    if cmd is None:
        print("usage: saklas template <verb> [...]")
        print()
        width = max(len(v) for v, _ in _TEMPLATE_VERBS)
        for v, desc in _TEMPLATE_VERBS:
            print(f"  {v:<{width}}  {desc}")
        print()
        print("Run `saklas template <verb> -h` for verb-specific options.")
        sys.exit(0)
    runner = _TEMPLATE_RUNNERS.get(cmd)
    if runner is None:
        print(f"unknown template verb {cmd!r}", file=sys.stderr)
        sys.exit(2)
    runner(args)
