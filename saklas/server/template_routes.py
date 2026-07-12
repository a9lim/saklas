"""Native template route group — ``/saklas/v1/templates/*``.

The standalone templated-completion artifact (``saklas.io.templates``): a slot, a
set of candidate values, and one or more multi-turn contexts. Lifecycle routes
(list / get / create / delete) are pure-IO; ``score`` runs the loaded model under
the session lock to return the per-context restricted-choice value distribution
(``session.score_template``) — the logit read, steering-aware. A manifold is fit
from a template via the manifold routes (``template_ref``); these routes own the
template itself + its scoring.
"""
from __future__ import annotations

import asyncio
from typing import Any, Literal

from fastapi import FastAPI, HTTPException

from saklas.io.templates import (
    AmbiguousTemplateError,
    TemplateFolder,
    TemplateFormatError,
    TemplateNotFoundError,
    create_template_folder,
    iter_template_folders,
    remove_template_folder,
    resolve_template,
    template_dir,
)
from saklas.server.app import acquire_session_lock
from saklas.server.native_common import NativeRequest


class TemplateTurn(NativeRequest):
    role: Literal["system", "user", "assistant"]
    content: str


class TemplateContextSpec(NativeRequest):
    """A multi-turn context: history turns + the slotted final assistant turn."""

    turns: list[TemplateTurn]
    assistant: str


class CreateTemplateRequest(NativeRequest):
    namespace: str = "local"
    name: str
    slot: str
    values: list[str]
    contexts: list[TemplateContextSpec]
    description: str = ""
    tags: list[str] = []
    force: bool = False


class ScoreTemplateRequest(NativeRequest):
    """Score the value distribution against each context.

    ``steering`` runs the scoring forward under a steering expression (the
    distributional before/after read); ``None`` is the unsteered baseline.
    """

    steering: str | None = None


def _template_detail(t: TemplateFolder) -> dict[str, Any]:
    payload = t.summary()
    payload["namespace"] = t.path.parent.name if t.path else "local"
    payload["contexts"] = [
        {"turns": [dict(turn) for turn in c.turns], "assistant": c.assistant}
        for c in t.contexts
    ]
    return payload


def register_template_routes(app: FastAPI) -> None:
    """Mount the ``/saklas/v1/templates/*`` tree onto ``app``."""

    session = app.state.session

    @app.get("/saklas/v1/templates")
    def list_templates():
        rows = []
        for t in iter_template_folders():
            row = t.summary()
            row["namespace"] = t.path.parent.name if t.path else "local"
            rows.append(row)
        return {"templates": rows}

    @app.get("/saklas/v1/templates/{namespace}/{name}")
    def get_template(namespace: str, name: str):
        path = template_dir(namespace, name)
        if not (path / "template.json").exists():
            raise HTTPException(404, f"no template {namespace}/{name}")
        return _template_detail(TemplateFolder.load(path))

    @app.post("/saklas/v1/templates", status_code=201)
    def create_template(req: CreateTemplateRequest):
        contexts = [
            {"turns": [t.model_dump() for t in c.turns], "assistant": c.assistant}
            for c in req.contexts
        ]
        try:
            t = create_template_folder(
                req.namespace, req.name,
                slot=req.slot, values=list(req.values), contexts=contexts,
                description=req.description, tags=list(req.tags), force=req.force,
            )
        except FileExistsError as e:
            raise HTTPException(409, str(e)) from e
        except TemplateFormatError as e:
            raise HTTPException(400, str(e)) from e
        return _template_detail(t)

    @app.delete("/saklas/v1/templates/{namespace}/{name}", status_code=200)
    def delete_template(namespace: str, name: str):
        removed = remove_template_folder(namespace, name)
        if not removed:
            raise HTTPException(404, f"no template {namespace}/{name}")
        return {"namespace": namespace, "name": name, "removed": True}

    @app.post("/saklas/v1/templates/{namespace}/{name}/score")
    async def score_template_route(
        namespace: str, name: str, req: ScoreTemplateRequest,
    ):
        try:
            tmpl = resolve_template(f"{namespace}/{name}")
        except (TemplateNotFoundError, AmbiguousTemplateError) as e:
            raise HTTPException(404, str(e)) from e

        async with acquire_session_lock(session) as acquired:
            if not acquired:
                raise HTTPException(503, "session locked")
            try:
                per_ctx = await asyncio.to_thread(
                    session.score_template, tmpl, steering=req.steering,
                )
            except Exception as e:  # steering-expr / scoring failure → 400
                raise HTTPException(400, f"scoring failed: {type(e).__name__}") from e
        return {
            "template": tmpl.name,
            "namespace": namespace,
            "steering": req.steering,
            "contexts": [sc.to_dict() for sc in per_ctx],
        }
