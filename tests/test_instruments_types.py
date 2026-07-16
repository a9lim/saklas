"""Instrument-protocol vocabulary: GateRef parsing, channel validation,
and the ScalarReading -> ProbeReading phase-1 wire bridge."""

import pytest

from saklas.core.errors import SaklasError, UnsupportedProbeChannelError
from saklas.core.instruments import (
    Assignment,
    Axis,
    DepthSummary,
    Distance,
    Fraction,
    GateRef,
    Membership,
    ScalarReading,
    parse_gate_ref,
    validate_gate_channels,
)
from saklas.core.results import ProbeReading


# --------------------------------------------------------------------------
# parse_gate_ref / scalar_key
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "key,expected",
    [
        ("confident.uncertain", GateRef("confident.uncertain", Axis(0))),
        ("personas[3]", GateRef("personas", Axis(3))),
        ("emotions:fraction", GateRef("emotions", Fraction())),
        ("emotions:membership", GateRef("emotions", Membership())),
        ("emotions@happy", GateRef("emotions", Distance("happy"))),
        ("personas~hacker", GateRef("personas", Assignment("hacker"))),
        ("personas@neutral", GateRef("personas", Distance("neutral"))),
        # namespaced probes keep their full attached name
        ("default/emotions@happy", GateRef("default/emotions", Distance("happy"))),
        ("jlens/fake", GateRef("jlens/fake", Axis(0))),
        ("sae/123", GateRef("sae/123", Axis(0))),
        ("default/personas[3]", GateRef("default/personas", Axis(3))),
        # a variant-suffixed probe name is a name, not a channel
        ("pirate:role-x", GateRef("pirate:role-x", Axis(0))),
        ("alice/pirate:role-x@node", GateRef("alice/pirate:role-x", Distance("node"))),
    ],
)
def test_parse_gate_ref_shapes(key: str, expected: GateRef) -> None:
    assert parse_gate_ref(key) == expected


@pytest.mark.parametrize(
    "key",
    [
        "confident.uncertain",
        "personas[3]",
        "emotions:fraction",
        "emotions:membership",
        "default/emotions@happy",
        "personas~hacker",
        "jlens/fake",
        "sae/123",
        "pirate:role-x",
    ],
)
def test_key_round_trip(key: str) -> None:
    assert parse_gate_ref(key).scalar_key() == key


def test_explicit_axis_zero_normalizes_to_bare():
    ref = parse_gate_ref("personas[0]")
    assert ref == GateRef("personas", Axis(0))
    # format(parse) normalizes [0] to the bare spelling — both read the
    # same channel; parse(format) is exact.
    assert ref.scalar_key() == "personas"
    assert parse_gate_ref(ref.scalar_key()) == ref


def test_ref_round_trip_is_exact():
    for ref in [
        GateRef("a/b.c", Axis(0)),
        GateRef("a/b.c", Axis(7)),
        GateRef("x", Fraction()),
        GateRef("x", Membership()),
        GateRef("ns/x", Distance("lbl")),
        GateRef("ns/x", Assignment("lbl")),
    ]:
        assert parse_gate_ref(ref.scalar_key()) == ref


# --------------------------------------------------------------------------
# Channel validation
# --------------------------------------------------------------------------

def test_validate_passes_allowed_channel():
    validate_gate_channels(
        GateRef("sae/123", Axis(0)), (Axis,), family="sae",
    )


def test_validate_rejects_unsupported_channel():
    with pytest.raises(UnsupportedProbeChannelError) as exc_info:
        validate_gate_channels(
            GateRef("sae/123", Membership()), (Axis,), family="sae",
        )
    err = exc_info.value
    # SaklasError family + stdlib MRO + 400 user_message, per errors.py
    # conventions.
    assert isinstance(err, SaklasError)
    assert isinstance(err, ValueError)
    status, text = err.user_message()
    assert status == 400
    assert "sae/123:membership" in text
    assert "sae" in text


def test_validate_rejects_label_channels_for_lens():
    with pytest.raises(UnsupportedProbeChannelError):
        validate_gate_channels(
            GateRef("jlens/fake", Distance("x")), (Axis,), family="lens",
        )
    with pytest.raises(UnsupportedProbeChannelError):
        validate_gate_channels(
            GateRef("jlens/fake", Assignment("x")), (Axis,), family="lens",
        )


# --------------------------------------------------------------------------
# ScalarReading -> ProbeReading bridge (must be bit-identical to the
# historical synthesized shapes)
# --------------------------------------------------------------------------

def test_bridge_matches_lens_synthesis():
    """The exact shape ``_score_lens_probes`` emitted: coords=(strength,),
    coords_per_layer[l]=(p_l,), depth from the readout stats, geometry
    fields defaulted."""
    layers = [10, 11, 12]
    per_layer = {10: 0.2, 11: 0.5, 12: 0.8}
    strength, com, spread = 0.5, 0.61, 0.07
    reading = ScalarReading(
        value=strength,
        unit="mean_token_probability",
        per_layer=per_layer,
        depth=DepthSummary((com,), (spread,), "readout_probability_mass"),
        meta={"word": "fake", "token_id": 12345},
    )
    expected = ProbeReading(
        fraction=0.0,
        nearest=[],
        coords=(strength,),
        residual=0.0,
        coords_per_layer={l: (per_layer[l],) for l in layers},
        depth_com=(com,),
        depth_spread=(spread,),
    )
    assert reading.to_probe_reading().to_dict() == expected.to_dict()


def test_bridge_matches_sae_synthesis():
    """The exact shape ``_score_sae_probes`` emitted: coords=(value,),
    coords_per_layer={hook_layer: (value,)}, depth_com the single-layer
    constant, depth_spread (0.0,)."""
    layer, n_layers = 12, 34
    value = 0.73
    depth_center = layer / max(n_layers - 1, 1)
    reading = ScalarReading(
        value=value,
        unit="activation_over_max",
        per_layer={layer: value},
        depth=DepthSummary((depth_center,), (0.0,), "single_layer"),
        meta={"feature_id": 123, "layer": layer},
    )
    expected = ProbeReading(
        fraction=0.0,
        nearest=[],
        coords=(value,),
        residual=0.0,
        coords_per_layer={layer: (value,)},
        depth_com=(depth_center,),
        depth_spread=(0.0,),
    )
    assert reading.to_probe_reading().to_dict() == expected.to_dict()


def test_bridge_defaults_without_depth():
    reading = ScalarReading(value=0.1, unit="raw_activation")
    bridged = reading.to_probe_reading()
    assert bridged.depth_com == ()
    assert bridged.depth_spread == ()
    assert bridged.membership == 1.0
    assert bridged.assignment == []
