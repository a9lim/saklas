"""Session route schemas and serializers for ``/saklas/v1/sessions``."""

from __future__ import annotations

from typing import Any

from saklas.core.generation import supports_thinking, thinking_is_optional
from saklas.core.session import SaklasSession
from saklas.core.steering import Steering
from saklas.server.native_common import NativeRequest


class CreateSessionRequest(NativeRequest):
    model: str | None = None
    device: str | None = None
    dtype: str | None = None


class PatchSessionRequest(NativeRequest):
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    system_prompt: str | None = None
    thinking: bool | None = None


def session_config_dict(session: SaklasSession) -> dict[str, Any]:
    cfg = session.config
    return {
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "top_k": cfg.top_k,
        "max_tokens": cfg.max_new_tokens,
        "system_prompt": cfg.system_prompt,
        "thinking": cfg.thinking,
    }


def session_model_type(session: SaklasSession) -> str | None:
    """Resolve the loaded model's ``model_type`` through multimodal wrappers."""
    model_cfg = session.model.config
    text_cfg = getattr(model_cfg, "text_config", None)
    mt = getattr(text_cfg, "model_type", None) if text_cfg is not None else None
    return mt or getattr(model_cfg, "model_type", None)


def role_support(session: SaklasSession) -> tuple[bool, bool]:
    """``(assistant_supported, user_supported)`` for the loaded model family."""
    from saklas.core.role_templates import ROLE_HEADERS, USER_ROLE_HEADERS

    mt = session_model_type(session)
    if mt is None:
        return (False, False)
    return (
        ROLE_HEADERS.get(mt) is not None,
        USER_ROLE_HEADERS.get(mt) is not None,
    )


def default_role_labels(
    session: SaklasSession,
) -> tuple[str | None, str | None]:
    """The family-standard assistant/user role labels, if substitutable."""
    from saklas.core.role_templates import ROLE_HEADERS, USER_ROLE_HEADERS

    mt = session_model_type(session)
    if mt is None:
        return (None, None)
    asst = ROLE_HEADERS.get(mt)
    usr = USER_ROLE_HEADERS.get(mt)
    return (
        asst.label if asst is not None else None,
        usr.label if usr is not None else None,
    )


def device_dtype(session: SaklasSession) -> tuple[str, str]:
    info = session.model_info
    return str(info["device"]), str(info["dtype"])


def _scene_capabilities(session: SaklasSession) -> dict[str, bool]:
    """Serialize the current session's validated scene grammar."""
    grammar = session.scene_grammar
    return {
        "scene_mode": grammar is not None,
        "thinking_input_supported": bool(
            grammar is not None and isinstance(grammar.think_open, str)
        ),
        "strips_history_thinking": bool(
            grammar is not None and grammar.strips_history_thinking is True
        ),
    }


def session_info(
    session: SaklasSession,
    default_steering: Steering | None,
    created_ts: int,
) -> dict[str, Any]:
    device, dtype = device_dtype(session)
    thinks = bool(supports_thinking(session.tokenizer))
    thinks_optional = bool(thinking_is_optional(session.tokenizer))
    is_base = bool(session.is_base_model)
    default_expr = str(default_steering) if default_steering is not None else None
    assistant_role_ok, user_role_ok = role_support(session)
    default_assistant_role, default_user_role = default_role_labels(session)
    jlens_fitted = session.has_compatible_jlens()
    sae_info = session.sae_info
    return {
        "id": "default",
        "model_id": session.model_id,
        "device": device,
        "dtype": dtype,
        "created": created_ts,
        "config": session_config_dict(session),
        "vectors": sorted(session.vectors.keys()),
        "probes": sorted(session.probes.keys()),
        "history_length": len(session.tree.messages_for()),
        "supports_thinking": thinks,
        "thinking_is_optional": thinks_optional,
        "is_base_model": is_base,
        "jlens_fitted": jlens_fitted,
        "live_lens_layers": session.live_lens_layers,
        "sae_loaded": sae_info is not None,
        "sae_info": sae_info,
        "live_sae": session.live_sae,
        # CAA live toggle state (POST .../probes/live): whether per-token
        # monitor scoring feeds live consumers.
        "live_probe_scores": session.live_probe_scores,
        "default_steering": default_expr,
        "role_substitution_supported": assistant_role_ok,
        "user_role_supported": user_role_ok,
        "default_assistant_role": default_assistant_role,
        "default_user_role": default_user_role,
        # Scene-grammar capabilities (the cast model): scene_mode gates
        # the seat toggle + free commit seating, thinking_input_supported
        # the committed-thinking box, strips_history_thinking its
        # "lasts one turn" pre-submit warning.
        **_scene_capabilities(session),
    }
