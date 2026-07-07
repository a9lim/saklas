"""Loom-tree route schemas and serializers for native saklas routes."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from saklas.core.session import SaklasSession


class TreeNavigateRequest(BaseModel):
    node_id: str


class TreeEditRequest(BaseModel):
    node_id: str
    text: str


class TreeBranchRequest(BaseModel):
    node_id: str
    text: str = ""
    role: str | None = None


class TreeStarRequest(BaseModel):
    node_id: str
    on: bool = True


class TreeNoteRequest(BaseModel):
    node_id: str
    text: str


class TreeTranscriptRequest(BaseModel):
    node_id: str | None = None


class TreeTranscriptLoadRequest(BaseModel):
    yaml: str
    mode: str = "default"
    strict: bool = False


class TreeDiffRequest(BaseModel):
    a_id: str
    b_id: str


class JointLogprobsRequest(BaseModel):
    a_id: str
    b_id: str


def tree_to_json(session: SaklasSession) -> dict[str, Any]:
    """Serialize the session's loom tree to JSON with token payloads."""
    return session.tree.to_dict(include_tokens=True)


def active_path_json(session: SaklasSession) -> dict[str, Any]:
    tree = session.tree
    path = tree.active_path()
    messages: list[dict[str, str]] = []
    node_ids: list[str] = []
    for node in path:
        if node.id == tree.root_id:
            continue
        messages.append({"role": node.role, "content": node.text})
        node_ids.append(node.id)
    return {
        "active_node_id": tree.active_node_id,
        "rev": tree.rev,
        "messages": messages,
        "node_ids": node_ids,
    }


def node_json(session: SaklasSession, node_id: str) -> dict[str, Any]:
    node = session.tree.get(node_id)
    out = node.to_dict(include_tokens=True)
    out["children"] = list(session.tree.children_of.get(node_id, []))
    return out


# Backcompat aliases for the old ``saklas_api.py`` import surface.
_tree_to_json = tree_to_json
_active_path_json = active_path_json
_node_json = node_json
