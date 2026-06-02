"""Crash-recoverable staging helpers for cache installs."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable


ErrorFactory = Callable[[str], Exception]
StageBuilder = Callable[[Path], None]


def stage_verify_swap(
    target_folder: Path,
    *,
    force: bool,
    label: str,
    build: StageBuilder,
    make_error: ErrorFactory,
) -> Path:
    """Build ``target_folder`` under ``.staging`` and atomically promote it.

    The caller owns artifact-specific staging and validation through
    ``build(staging)``. This helper owns the shared install choreography:
    recover a previous ``.bak`` when the destination is missing, wipe stale
    staging, build a fully validated staging tree, then promote via
    ``target -> .bak`` and ``.staging -> target`` with best-effort restore on
    failure.
    """
    if target_folder.exists() and not force:
        raise make_error(f"{target_folder} exists; pass force=True to overwrite")

    staging = target_folder.with_name(target_folder.name + ".staging")
    backup = target_folder.with_name(target_folder.name + ".bak")

    if not target_folder.exists() and backup.exists():
        try:
            backup.rename(target_folder)
        except OSError:
            pass

    if staging.exists():
        shutil.rmtree(staging)
    if backup.exists():
        shutil.rmtree(backup)

    staging.mkdir(parents=True, exist_ok=True)
    try:
        build(staging)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    had_existing = target_folder.exists()
    if had_existing:
        try:
            target_folder.rename(backup)
        except OSError as e:
            shutil.rmtree(staging, ignore_errors=True)
            raise make_error(
                f"{label}: could not move existing install aside ({e})"
            ) from e

    try:
        staging.rename(target_folder)
    except OSError as e:
        if had_existing and backup.exists() and not target_folder.exists():
            try:
                backup.rename(target_folder)
            except OSError:
                pass
        shutil.rmtree(staging, ignore_errors=True)
        raise make_error(
            f"{label}: could not promote staging into place ({e})"
        ) from e

    if had_existing:
        shutil.rmtree(backup, ignore_errors=True)

    return target_folder
