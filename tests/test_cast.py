"""Cast-recipe composition at generation time (phase 3).

The roster tier is the weakest: a cast member's recipe fills only the
fields a call left unset, sampling merges field-wise with the call's
non-default fields winning, and regen overrides still compose on top
(covered by the existing ``_resolve_recipe_override`` tests — this file
pins the roster tier itself).  ``_apply_cast_defaults`` needs only
``self.tree``, so a bare stub stands in for the session.
"""
from __future__ import annotations

from typing import Any, cast

from saklas import CastMember, LoomTree, Recipe, SamplingConfig
from saklas.core.session import SaklasSession


class _Stub:
    def __init__(self) -> None:
        self.tree = LoomTree()


def _as_session(stub: _Stub) -> SaklasSession:
    return cast(SaklasSession, stub)


def _apply(
    stub: _Stub,
    steering: str | None = None,
    sampling: SamplingConfig | None = None,
    thinking: bool | None = None,
    *,
    raw: bool = False,
    gen_seat: str = "assistant",
) -> tuple[Any, SamplingConfig | None, bool | None]:
    return SaklasSession._apply_cast_defaults(
        _as_session(stub), steering, sampling, thinking,
        raw=raw, gen_seat=gen_seat,
    )


def test_no_label_no_change():
    stub = _Stub()
    stub.tree.set_cast_member(
        "deer", CastMember(recipe=Recipe(steering="0.5 skittish")),
    )
    # No sampling box at all → no label → roster untouched.
    assert _apply(stub) == (None, None, None)
    # A sampling box without the seat's label field → still no lookup.
    s = SamplingConfig(temperature=0.7)
    assert _apply(stub, sampling=s) == (None, s, None)


def test_structural_assistant_member_applies_without_custom_label():
    stub = _Stub()
    stub.tree.set_cast_member(
        "assistant", CastMember(recipe=Recipe(steering="0.2 concise")),
    )
    steering, _, _ = _apply(stub)
    assert steering == "0.2 concise"


def test_label_without_member_no_change():
    stub = _Stub()
    s = SamplingConfig(assistant_role="pirate")
    assert _apply(stub, sampling=s) == (None, s, None)


def test_cast_fills_unset_fields():
    stub = _Stub()
    stub.tree.set_cast_member(
        "deer",
        CastMember(recipe=Recipe(steering="0.5 skittish", thinking=False)),
    )
    s = SamplingConfig(assistant_role="deer")
    steering, sampling, thinking = _apply(stub, sampling=s)
    assert steering == "0.5 skittish"
    assert thinking is False
    assert sampling == s


def test_explicit_kwargs_win():
    stub = _Stub()
    stub.tree.set_cast_member(
        "deer",
        CastMember(recipe=Recipe(steering="0.5 skittish", thinking=False)),
    )
    s = SamplingConfig(assistant_role="deer")
    steering, _, thinking = _apply(
        stub, steering="0.2 formal", sampling=s, thinking=True,
    )
    assert steering == "0.2 formal"
    assert thinking is True
    # The empty string is the explicit unsteered override.
    steering, _, _ = _apply(stub, steering="", sampling=s)
    assert steering == ""


def test_sampling_merges_fieldwise():
    stub = _Stub()
    stub.tree.set_cast_member(
        "deer",
        CastMember(recipe=Recipe(
            sampling=SamplingConfig(temperature=1.1, top_p=0.9),
        )),
    )
    call = SamplingConfig(assistant_role="deer", temperature=0.3)
    _, merged, _ = _apply(stub, sampling=call)
    assert merged is not None
    assert merged.temperature == 0.3      # call's non-default field wins
    assert merged.top_p == 0.9            # cast fills the unset field
    assert merged.assistant_role == "deer"  # label survives the merge


def test_seed_applies_only_when_unpinned():
    stub = _Stub()
    stub.tree.set_cast_member(
        "deer", CastMember(recipe=Recipe(seed=1234)),
    )
    _, merged, _ = _apply(stub, sampling=SamplingConfig(assistant_role="deer"))
    assert merged is not None and merged.seed == 1234
    _, merged, _ = _apply(
        stub, sampling=SamplingConfig(assistant_role="deer", seed=7),
    )
    assert merged is not None and merged.seed == 7


def test_user_seat_reads_user_role():
    stub = _Stub()
    stub.tree.set_cast_member(
        "captain", CastMember(recipe=Recipe(steering="0.4 formal")),
    )
    s = SamplingConfig(user_role="captain", assistant_role="deer")
    steering, _, _ = _apply(stub, sampling=s, gen_seat="user")
    assert steering == "0.4 formal"
    # Assistant-seat gen with the same box reads assistant_role instead.
    steering, _, _ = _apply(stub, sampling=s, gen_seat="assistant")
    assert steering is None


def test_raw_mode_skips_roster():
    stub = _Stub()
    stub.tree.set_cast_member(
        "deer", CastMember(recipe=Recipe(steering="0.5 skittish")),
    )
    s = SamplingConfig(assistant_role="deer")
    assert _apply(stub, sampling=s, raw=True) == (None, s, None)


def test_bare_member_contributes_nothing():
    stub = _Stub()
    stub.tree.set_cast_member("deer", CastMember(notes="just a name"))
    s = SamplingConfig(assistant_role="deer")
    assert _apply(stub, sampling=s) == (None, s, None)


def test_session_set_cast_member_validates_steering():
    import pytest
    from saklas.core.steering_expr import SteeringExprError

    stub = _Stub()
    with pytest.raises(SteeringExprError):
        SaklasSession.set_cast_member(
            _as_session(stub), "deer", steering="0.5 !!bad!!",
        )
    member = SaklasSession.set_cast_member(
        _as_session(stub), "deer", steering="0.5 formal.casual",
    )
    assert member.recipe is not None
    assert member.recipe.steering == "0.5 formal.casual"
    assert stub.tree.cast["deer"] == member
    # Field-less call authors a bare named label.
    bare = SaklasSession.set_cast_member(_as_session(stub), "narrator")
    assert bare.recipe is None
    SaklasSession.remove_cast_member(_as_session(stub), "narrator")
    assert "narrator" not in stub.tree.cast


def test_ws_explicit_clear_survives_as_empty_steering():
    """The server's steering merge must hand the session an *empty*
    Steering on an explicit clear (``steering: ""``) — None means
    "unset" and would let the cast roster fill it back in."""
    from saklas.server.request_helpers import merge_steering, parse_request_steering

    req, clear = parse_request_steering("")
    assert req is None and clear is True
    merged = merge_steering(req, None, clear, None)
    assert merged is not None
    assert merged.alphas == {}

    # No expression at all stays None — the roster's fill case.
    req, clear = parse_request_steering(None)
    assert merge_steering(req, None, clear, None) is None
