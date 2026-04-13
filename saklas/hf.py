"""Hugging Face Hub consumption wrappers for saklas pack distribution.

This module is a stub in Phase 5 — the public functions raise
NotImplementedError until Phase 6 lands their real implementations.
They exist here so cache_ops and tests can reference them by name.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional


class HFError(RuntimeError):
    pass


def pull_pack(coord: str, target_folder: Path, *, force: bool) -> Path:
    raise NotImplementedError("saklas.hf.pull_pack lands in Phase 6")


def search_packs(selector) -> list[dict]:
    raise NotImplementedError("saklas.hf.search_packs lands in Phase 6")


def fetch_info(coord: str) -> dict:
    raise NotImplementedError("saklas.hf.fetch_info lands in Phase 6")
