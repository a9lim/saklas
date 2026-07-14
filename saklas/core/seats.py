"""Native conversation-seat vocabulary.

Saklas presents the two structural seats as ``human`` and ``model``.  Chat
templates and compatibility protocols still spell those roles ``user`` and
``assistant``; conversion belongs at those adapters, not in UI semantics.
"""

from __future__ import annotations

from typing import Literal, cast


Seat = Literal["human", "model"]
ChatRole = Literal["user", "assistant"]


def seat_to_chat_role(seat: Seat) -> ChatRole:
    """Lower a native seat to the role name expected by chat protocols."""
    if seat == "human":
        return "user"
    if seat == "model":
        return "assistant"
    raise ValueError(f"seat must be 'human' or 'model', got {seat!r}")


def chat_role_to_seat(role: str) -> Seat:
    """Lift a chat-protocol role to the native Saklas seat vocabulary."""
    if role == "user":
        return "human"
    if role == "assistant":
        return "model"
    raise ValueError(f"chat role must be 'user' or 'assistant', got {role!r}")


def coerce_seat(value: str) -> Seat:
    """Validate a dynamically supplied native seat for typed call sites."""
    if value not in ("human", "model"):
        raise ValueError(f"seat must be 'human' or 'model', got {value!r}")
    return cast(Seat, value)


__all__ = ["ChatRole", "Seat", "chat_role_to_seat", "coerce_seat", "seat_to_chat_role"]
