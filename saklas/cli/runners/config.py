"""``saklas config <verb>`` runners (show / validate)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from saklas.cli.runners.shared import _saklas_error_exit


@_saklas_error_exit
def _run_config(args: argparse.Namespace) -> None:
    cmd = getattr(args, "config_cmd", None)
    if cmd == "show":
        _run_config_show(args)
    elif cmd == "validate":
        _run_config_validate(args)
    else:
        print("usage: saklas config {show,validate}")
        print()
        print("  show      Print the effective merged config")
        print("  validate  Validate a config file (exit 0 valid, 2 invalid)")
        sys.exit(0)


def _run_config_show(args: argparse.Namespace) -> None:
    from saklas import __version__
    from saklas.cli.config_file import ConfigFile, apply_flag_overrides
    extras = [Path(p) for p in (args.config or [])]
    composed = ConfigFile.effective(extras, include_default=not args.no_default)
    if args.model is not None:
        composed = apply_flag_overrides(composed, model=args.model)
    header = f"# effective merged config for saklas {__version__}"
    sys.stdout.write(composed.to_yaml(header=header))


def _run_config_validate(args: argparse.Namespace) -> None:
    from saklas.cli.config_file import ConfigFile, ConfigFileError
    from saklas.core.steering_expr import referenced_selectors
    p = Path(args.file)
    if not p.exists():
        print(f"config validate: {p}: file not found", file=sys.stderr)
        sys.exit(2)
    try:
        cfg = ConfigFile.load(p)
        if cfg.vectors is None:
            print(f"{p}: ok")
            return
        # Dry-run: don't install, just check resolvability.
        from saklas.io.selectors import all_concepts
        installed = {(c.namespace, c.name) for c in all_concepts()}
        installed_names = {c.name for c in all_concepts()}
        missing: list[str] = []
        for ns, concept, _variant in referenced_selectors(cfg.vectors):
            if ns is None:
                if concept in installed_names:
                    continue
                # Bare pole of an installed bipolar resolves fine too.
                slug = concept.split(".")[0] if "." in concept else concept
                if any(
                    slug in c.name.split(".")
                    for c in all_concepts()
                    if "." in c.name
                ):
                    continue
                missing.append(concept)
                continue
            if ns == "default" or (ns, concept) in installed:
                continue
            if ns == "local":
                missing.append(f"{ns}/{concept}")
                continue
            # HF namespace — we assume install would succeed; don't probe.
        if missing:
            raise ConfigFileError(
                f"unresolvable vectors (not installed and no namespace to install from): {missing}"
            )
    except ConfigFileError as e:
        print(f"config validate: {p}: {e}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"config validate: {p}: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(2)
    print(f"{p}: ok")
