"""Tests for shared native SSE helpers."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from saklas.server.sse import ProgressCallback, progress_sse_response


def _client_for(
    job: Callable[[ProgressCallback], Awaitable[Any]],
    *,
    error_formatter: Callable[[Exception], dict[str, Any] | None] | None = None,
) -> TestClient:
    app = FastAPI()

    @app.get("/sse")
    async def sse_route():
        return progress_sse_response(
            asyncio.Lock(),
            job,
            error_message="job failed",
            log_message="test job failed",
            error_formatter=error_formatter,
        )

    return TestClient(app)


def _body(client: TestClient) -> str:
    with client.stream("GET", "/sse") as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        return b"".join(resp.iter_bytes()).decode("utf-8")


def test_progress_sse_streams_progress_then_done() -> None:
    async def job(on_progress: ProgressCallback) -> dict[str, Any]:
        on_progress("first")
        on_progress("second")
        return {"ok": True}

    text = _body(_client_for(job))
    assert text.count("event: progress") == 2
    assert "first" in text
    assert "second" in text
    assert "event: done" in text
    assert '"ok": true' in text


def test_progress_sse_generic_error_is_scrubbed() -> None:
    async def job(_on_progress: ProgressCallback) -> dict[str, Any]:
        raise RuntimeError("/secret/cache/path")

    text = _body(_client_for(job))
    assert "event: error" in text
    assert '"message": "job failed"' in text
    assert '"code": "RuntimeError"' in text
    assert "/secret/cache/path" not in text


def test_progress_sse_typed_error_formatter_can_expose_safe_message() -> None:
    async def job(_on_progress: ProgressCallback) -> dict[str, Any]:
        raise ValueError("safe author-facing message")

    def format_error(e: Exception) -> dict[str, Any] | None:
        if isinstance(e, ValueError):
            return {"message": str(e), "code": "ValueError"}
        return None

    text = _body(_client_for(job, error_formatter=format_error))
    assert "event: error" in text
    assert "safe author-facing message" in text
    assert '"code": "ValueError"' in text


def test_progress_sse_formatter_crash_falls_back_to_generic_error() -> None:
    async def job(_on_progress: ProgressCallback) -> dict[str, Any]:
        raise RuntimeError("/private/path")

    def bad_formatter(_e: Exception) -> dict[str, Any] | None:
        raise AssertionError("formatter bug")

    text = _body(_client_for(job, error_formatter=bad_formatter))
    assert "event: error" in text
    assert '"message": "job failed"' in text
    assert '"code": "RuntimeError"' in text
    assert "/private/path" not in text
    assert "formatter bug" not in text
