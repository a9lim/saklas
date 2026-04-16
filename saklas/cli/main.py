"""CLI entry point for saklas.

Top-level shape (v2 hard break):

    saklas tui <model> [...]
    saklas serve <model> [...]
    saklas pack {install,refresh,clear,rm,ls,search,push,export} ...
    saklas vector {extract,merge,clone,compare} ...
    saklas config {show,validate} ...

There is no bare-TUI mode. ``saklas`` with no arguments prints help.
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
        print("try 'saklas tui <model_id>' or 'saklas --help'")
        sys.exit(0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    cmd = getattr(args, "command", None)
    if cmd is None:
        _build_root_parser().print_help()
        sys.exit(0)
    _COMMAND_RUNNERS[cmd](args)
