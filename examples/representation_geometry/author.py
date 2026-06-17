"""Author a probe's template + discover manifold (pure-IO, no model).

    python examples/representation_geometry/author.py <probe> [--force]
    python examples/representation_geometry/author.py all

then fit it against a model:

    saklas manifold fit <probe> -m google/gemma-3-4b-it

`<probe>` is one of: countries, years, years_now, years_now_future. Authoring is
deterministic — the node corpora are just the template's ``values × contexts`` —
so this step needs no GPU; the fit does.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from data import PROBES  # noqa: E402

from saklas.io.manifolds import create_manifold_from_template
from saklas.io.templates import create_template_folder

NS = "local"


def author(name: str, *, force: bool) -> None:
    probe = PROBES[name]
    create_template_folder(
        NS, name, slot=probe.slot, values=list(probe.values),
        contexts=probe.canonical_contexts(), description=probe.description,
        force=force,
    )
    create_manifold_from_template(
        NS, name, probe.description, template_ref=f"{NS}/{name}",
        fit_mode="auto", force=force,
    )
    print(f"authored {NS}/{name}: {len(probe.values)} values × "
          f"{len(probe.contexts)} contexts  ({probe.framing})")
    print(f"  → saklas manifold fit {name} -m google/gemma-3-4b-it")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("probe", choices=[*PROBES, "all"])
    ap.add_argument("-f", "--force", action="store_true",
                    help="overwrite an existing template/manifold of the same name")
    args = ap.parse_args()
    names = list(PROBES) if args.probe == "all" else [args.probe]
    for name in names:
        author(name, force=args.force)


if __name__ == "__main__":
    main()
