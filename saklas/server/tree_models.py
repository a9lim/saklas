"""Loom-tree route schemas and serializers for native saklas routes."""

from __future__ import annotations

from typing import Any, Literal

from saklas.core.session import SaklasSession
from saklas.server.native_common import NativeRequest


class TreeNavigateRequest(NativeRequest):
    node_id: str


class TreeEditRequest(NativeRequest):
    node_id: str
    text: str


class TreeBranchRequest(NativeRequest):
    node_id: str
    text: str = ""
    role: Literal["user", "assistant", "system"] | None = None


class TreeStarRequest(NativeRequest):
    node_id: str
    on: bool = True


class TreeNoteRequest(NativeRequest):
    node_id: str
    text: str


class CastMemberRequest(NativeRequest):
    """Create/replace a cast member — a named label plus its standing
    recipe fragment.  All fields optional; a bare body authors a plain
    named label."""

    steering: str | None = None
    thinking: bool | None = None
    seed: int | None = None
    notes: str = ""


class TreeTranscriptRequest(NativeRequest):
    node_id: str | None = None


class TreeTranscriptLoadRequest(NativeRequest):
    yaml: str
    mode: Literal["default", "here", "merge"] = "default"
    strict: bool = False


class TreeRestoreRequest(NativeRequest):
    tree: dict[str, Any]


class TreeDiffRequest(NativeRequest):
    a_id: str
    b_id: str


class JointLogprobsRequest(NativeRequest):
    a_id: str
    b_id: str


def cast_json(session: SaklasSession) -> dict[str, Any]:
    """Serialize the effective auto-derived cast with configuration origin."""
    configured = session.tree.cast
    out: dict[str, Any] = {}
    for label, member in session.tree.cast_roster().items():
        row = member.to_dict()
        row["origin"] = (
            "configured" if label in configured
            else "structural" if label in ("user", "assistant")
            else "observed"
        )
        out[label] = row
    return out


def tree_to_json(session: SaklasSession) -> dict[str, Any]:
    """Serialize the session's loom tree to JSON with token payloads."""
    out = session.tree.to_dict(include_tokens=True)
    out["cast"] = cast_json(session)
    return out


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
