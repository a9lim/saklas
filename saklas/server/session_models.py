"""Session route schemas and serializers for ``/saklas/v1/sessions``."""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel

from saklas.core.generation import supports_thinking, thinking_is_optional
from saklas.core.session import SaklasSession
from saklas.core.steering import Steering
from saklas.server.native_common import resolve_session_id, session_aliases


class CreateSessionRequest(BaseModel):
    model: str | None = None
    device: str | None = None
    dtype: str | None = None


class PatchSessionRequest(BaseModel):
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    system_prompt: str | None = None
    thinking: bool | None = None


def session_config_dict(session: SaklasSession) -> dict[str, Any]:
    cfg = session.config
    return {
        "temperature": getattr(cfg, "temperature", None),
        "top_p": getattr(cfg, "top_p", None),
        "top_k": getattr(cfg, "top_k", None),
        "max_tokens": getattr(cfg, "max_new_tokens", None),
        "system_prompt": getattr(cfg, "system_prompt", None),
        "thinking": getattr(cfg, "thinking", None),
    }


def session_model_type(session: SaklasSession) -> str | None:
    """Resolve the loaded model's ``model_type`` through multimodal wrappers."""
    model_cfg = getattr(getattr(session, "model", None), "config", None)
    if model_cfg is None:
        return None
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
    info = session.model_info or {}
    device = str(info.get("device", getattr(session, "device", "")))
    dtype = str(info.get("dtype", getattr(session, "_dtype", "")))
    return device, dtype


def _live_probe_scores(session: SaklasSession) -> bool:
    """The session's CAA live-toggle state, defensively coerced.

    Only a real bool passes through; a stub session (MagicMock in tests,
    or a pre-toggle engine) reads as the default-on.
    """
    state = getattr(session, "live_probe_scores", True)
    return state if isinstance(state, bool) else True


def _live_lens_layers(session: SaklasSession) -> list[int] | None:
    """The session's live-lens layer list, or ``None`` when off.

    Coerces defensively: only a real int sequence passes through, so a
    stub session (MagicMock in tests, or a pre-lens engine) reads as
    "live lens off" rather than an unserializable payload.
    """
    layers = getattr(session, "live_lens_layers", None)
    if isinstance(layers, (list, tuple)):
        try:
            return [int(l) for l in layers]
        except (TypeError, ValueError):
            return None
    return None


def _sae_info(session: SaklasSession) -> dict[str, Any] | None:
    info = getattr(session, "sae_info", None)
    return dict(info) if isinstance(info, dict) else None


def _scene_capabilities(session: SaklasSession) -> dict[str, bool]:
    """Scene-grammar capability flags, defensively coerced.

    Reading ``session.scene_grammar`` runs the lazy autopsy once on a
    real session; a stub (MagicMock in tests) or a fallback family
    reads as scene mode off.  ``thinking_input_supported`` gates the
    composer's committed-thinking box; ``strips_history_thinking``
    drives its "lasts one turn" pre-submit warning.
    """
    from saklas.core.scene import TurnGrammar

    try:
        grammar = getattr(session, "scene_grammar", None)
    except Exception:
        grammar = None
    if not isinstance(grammar, TurnGrammar):
        # MagicMock stubs and legacy sessions both read as scene off.
        grammar = None
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
    session: SaklasSession, default_steering: Steering | None,
) -> dict[str, Any]:
    device, dtype = device_dtype(session)
    try:
        thinks = bool(supports_thinking(session.tokenizer))
        thinks_optional = bool(thinking_is_optional(session.tokenizer))
    except Exception:
        thinks = False
        thinks_optional = False
    try:
        is_base = bool(session.is_base_model)
    except Exception:
        is_base = False
    created = getattr(session, "_created_ts", None) or int(time.time())
    default_expr = str(default_steering) if default_steering is not None else None
    try:
        assistant_role_ok, user_role_ok = role_support(session)
        default_assistant_role, default_user_role = default_role_labels(session)
    except Exception:
        assistant_role_ok = user_role_ok = False
        default_assistant_role = default_user_role = None
    try:
        checker = getattr(type(session), "has_compatible_jlens", None)
        if callable(checker):
            jlens_fitted = bool(checker(session))
        else:
            from saklas.io.lens import load_lens_sidecar

            # Protocol/test doubles have no loaded model to fingerprint.
            jlens_fitted = load_lens_sidecar(session.model_id) is not None
    except Exception:
        jlens_fitted = False
    return {
        "id": "default",
        "aliases": session_aliases(session),
        "model_id": session.model_id,
        "device": device,
        "dtype": dtype,
        "created": created,
        "config": session_config_dict(session),
        "vectors": sorted(session.vectors.keys()),
        "probes": sorted(session.probes.keys()),
        "history_length": len(session.history) if hasattr(session, "history") else 0,
        "supports_thinking": thinks,
        "thinking_is_optional": thinks_optional,
        "is_base_model": is_base,
        "jlens_fitted": jlens_fitted,
        # Live workspace readout state (POST .../lens/live): the resolved
        # layer list while enabled, null while off — lets the dashboard
        # rehydrate its WORKSPACE panel toggle on reload.  Coerced so a
        # stub session (tests) reads as off rather than unserializable.
        "live_lens_layers": _live_lens_layers(session),
        "sae_loaded": _sae_info(session) is not None,
        "sae_info": _sae_info(session),
        "live_sae": bool(getattr(session, "live_sae", False))
        if isinstance(getattr(session, "live_sae", False), bool) else False,
        # CAA live toggle state (POST .../probes/live): whether per-token
        # monitor scoring feeds live consumers.  Coerced so a stub session
        # (tests) reads as the default-on.
        "live_probe_scores": _live_probe_scores(session),
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


# Backcompat aliases for the old ``saklas_api.py`` import surface.
_resolve_session_id = resolve_session_id
_session_config_dict = session_config_dict
_session_model_type = session_model_type
_role_support = role_support
_default_role_labels = default_role_labels
_device_dtype = device_dtype
_session_info = session_info
