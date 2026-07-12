"""Experiment route schemas for native saklas routes."""

from __future__ import annotations

from saklas.server.native_common import NativeRequest
from saklas.server.ws_models import WSSamplingParams


class ExperimentFanRequest(NativeRequest):
    prompt: str | list[dict[str, str]]
    grid: dict[str, list[float]]
    base_steering: str | None = None
    sampling: WSSamplingParams | None = None
    thinking: bool | None = None
    raw: bool = False
