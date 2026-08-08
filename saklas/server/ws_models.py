"""WebSocket request schemas and serializers for native saklas streams."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from saklas.core.results import GenerationResult
from saklas.core.sampling import SamplingConfig
from saklas.server import request_helpers
from saklas.server.native_common import NativeRequest


class WSSamplingParams(NativeRequest):
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    seed: int | None = None
    stop: list[str] | None = None
    logit_bias: dict[int, float] | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    return_top_k: int = 0
    return_probe_readings: bool = True
    persist_per_layer_scores: bool = False
    persist_subspace_coords: bool = False
    user_role: str | None = None
    assistant_role: str | None = None


class WSInputMessage(NativeRequest):
    """One current chat message accepted by the native generation stream."""

    role: Literal["system", "user", "assistant"]
    content: str
    #: Cast label for this turn — the same per-turn ``role_label`` the loom
    #: stamps on its nodes, rendered into the constructed header by the scene
    #: stitcher (``core.generation.build_chat_input`` reads the ``"label"``
    #: key).  The dashboard's auto-regen shadow replays a conversation as an
    #: explicit message list, so without this field a shadow of any
    #: custom-labelled turn was rejected outright by ``extra="forbid"`` — and
    #: dropping it would silently re-render those turns under the default
    #: labels, which is a different prompt from the one being shadowed.
    label: str | None = None


class WSGenerateMessage(NativeRequest):
    """The specialist generation frame: fork, prefill, shadow, seat.

    Authored turns are ``submit``'s job alone — this frame carries no
    ``commit_*`` vocabulary.  ``submit`` lowers onto this schema for its
    generating branch (:func:`saklas.server.ws_stream._normalize_submit`).
    """

    type: Literal["generate"]
    input: str | list[WSInputMessage] | None = None
    steering: str | None = None
    sampling: WSSamplingParams | None = None
    thinking: bool | None = None
    stateless: bool = True
    raw: bool = False
    parent_node_id: str | None = None
    n: int = Field(default=1, ge=1)
    recipe_override: str | None = None
    fork_node_id: str | None = None
    fork_raw_index: int | None = None
    fork_alt_token_id: int | None = None
    prefill_node_id: str | None = None
    prefill_text: str | None = None
    # Cast model: which seat the generated turn occupies.  ``"user"``
    # renders the generation prompt as a user-seat header and lands the
    # node with ``role="user"`` + a stamped recipe (generated is
    # provenance, not a seat).  ``None`` = assistant (the classic flow).
    # Pair with ``input: null`` for a continue — no committed turn, the
    # model speaks next from the current leaf (a/a and u/u sequences).
    generate_seat: Literal["user", "assistant"] | None = None

    @model_validator(mode="after")
    def _check_mode_fields(self) -> "WSGenerateMessage":
        """Fork and prefill are mutually exclusive, and each needs its whole
        field group.

        These live here rather than in a hand-rolled handler pass so the
        schema is the single description of a well-formed frame: one
        validation layer, one error shape (a ``ValidationError`` the WS
        reader renders through the shared error-frame builder).
        :class:`PydanticCustomError` keeps the message verbatim — a plain
        ``ValueError`` would reach the wire as ``"Value error, …"``.
        """
        is_fork = self.fork_node_id is not None
        if is_fork and (
            self.fork_raw_index is None or self.fork_alt_token_id is None
        ):
            raise PydanticCustomError(
                "fork_fields",
                "fork requires fork_node_id, fork_raw_index, and "
                "fork_alt_token_id together",
            )
        is_prefill = self.prefill_node_id is not None
        if is_prefill and not self.prefill_text:
            raise PydanticCustomError(
                "prefill_fields",
                "prefill requires prefill_node_id and prefill_text together",
            )
        if is_fork and is_prefill:
            raise PydanticCustomError(
                "mode_conflict",
                "a generate message cannot be both a fork and a prefill",
            )
        return self


class WSSubmitMessage(NativeRequest):
    """Native role-neutral chat submission.

    ``text`` plus ``authored_role`` appends a span.  ``generated_role`` then
    optionally asks the model to continue from it; omit it for append-only.
    With no text, ``generated_role`` is a bare continue from the selected leaf.
    """

    type: Literal["submit"]
    text: str | None = None
    authored_role: Literal["user", "assistant"] | None = None
    generated_role: Literal["user", "assistant"] | None = None
    steering: str | None = None
    sampling: WSSamplingParams | None = None
    thinking: bool | None = None
    authored_thinking: str | None = None
    raw: bool = False
    parent_node_id: str | None = None
    # Constrained here as well as on ``WSGenerateMessage``: the reader must
    # reject a bad fan width at parse time, because ``_normalize_submit``
    # forwards this value into a ``WSGenerateMessage`` construction that is
    # not inside the handler's error-frame guard.
    n: int = Field(default=1, ge=1)
    recipe_override: str | None = None


def build_sampling_config(body: WSSamplingParams | None) -> SamplingConfig | None:
    """Adapt the WS sampling block onto the shared sampling constructor.

    Same operation the OpenAI and Ollama routes perform under the same name
    (:func:`saklas.server.request_helpers.build_sampling_config`); this is just
    the native wire body's field mapping onto it.
    """
    if body is None:
        return None
    return request_helpers.build_sampling_config(
        temperature=body.temperature,
        top_p=body.top_p,
        top_k=body.top_k,
        max_tokens=body.max_tokens,
        seed=body.seed,
        stop=body.stop,
        logit_bias=body.logit_bias,
        presence_penalty=body.presence_penalty,
        frequency_penalty=body.frequency_penalty,
        return_top_k=body.return_top_k,
        return_probe_readings=body.return_probe_readings,
        user_role=body.user_role,
        assistant_role=body.assistant_role,
        persist_per_layer_scores=body.persist_per_layer_scores,
        persist_subspace_coords=body.persist_subspace_coords,
    )


def build_input(
    body: str | list[WSInputMessage] | None,
) -> str | list[dict[str, Any]] | None:
    """Lower the strict wire message objects to the engine's mapping shape.

    ``model_dump`` carries ``label`` through verbatim, which is exactly the
    per-turn key ``session._prepare_input`` hands to ``build_chat_input`` —
    the wire message and a loom-derived message dict are the same shape.
    """
    if not isinstance(body, list):
        return body
    return [message.model_dump() for message in body]


def result_to_json(result: GenerationResult) -> dict[str, Any]:
    prompt_tokens = result.prompt_tokens
    completion = result.token_count
    return {
        "text": result.text,
        "tokens": completion,
        "finish_reason": result.finish_reason,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion,
            "total_tokens": prompt_tokens + completion,
        },
    }
