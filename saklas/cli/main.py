"""CLI entry point for saklas.

Eight top-level verbs. ``manifold`` is the steering-vector / manifold *compute*
surface (a steering vector is the K=2 case of a flat manifold); ``pack`` is the
manifold *lifecycle* surface (install / share / inspect / remove); ``template``
owns the standalone templated-completion artifact; ``lens`` owns the per-model
Jacobian-lens artifact (residual→output transport + vocabulary readout):

    saklas serve <model> [...]
    saklas manifold {extract,generate,from-template,fit,bake,merge,transfer,compare,why} ...
    saklas pack {ls,show,install,search,push,rm,clear,refresh,export} ...
    saklas experiment {fan,transcript,naturalness} ...
    saklas config {show,validate} ...
    saklas template {create,ls,show,score,rm} ...
    saklas lens {fit,fetch,ls,show,use,top,decompose,rm} ...
    saklas sae {train,fetch,ls,show,use,rm} ...

``saklas`` with no arguments prints help.
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("PYTHONWARNINGS", "ignore::UserWarning:multiprocessing.resource_tracker")

from saklas.cli.parsers import _build_root_parser
from saklas.cli.runners import _COMMAND_RUNNERS


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    if argv is None:
        argv = sys.argv[1:]
    parser = _build_root_parser()
    # Zero-arg: print help+hint and exit 0 (not argparse's exit 2).
    if not argv:
        parser.print_help()
        print()
        print("try 'saklas serve <model_id>' or 'saklas --help'")
        sys.exit(0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    cmd = getattr(args, "command", None)
    if cmd is None:
        _build_root_parser().print_help()
        sys.exit(0)
    _COMMAND_RUNNERS[cmd](args)
