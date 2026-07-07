"""Experiment route schemas for native saklas routes."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from saklas.server.ws_models import WSSamplingParams


class ExperimentFanRequest(BaseModel):
    prompt: Any
    grid: dict[str, list[float]]
    base_steering: str | None = None
    sampling: WSSamplingParams | None = None
    thinking: bool | None = None
    raw: bool = False
