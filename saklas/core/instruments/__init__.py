"""The unified read-side instruments.

One contract over the three read families — geometry (Monitor subspace
probes), the Jacobian-lens readout channel, SAE feature reads.  See
``types.py`` for the shared vocabulary and ``protocol.py`` for the
instrument/run contract and the division of labor.
"""

from saklas.core.instruments.types import (
    AGG_TAIL_DEPTH,
    Assignment,
    Axis,
    DepthBasis,
    DepthSummary,
    Distance,
    Fraction,
    GateChannel,
    GateRef,
    InstrumentBinding,
    InstrumentPlan,
    InstrumentPrep,
    LensPrep,
    Membership,
    ReadRequest,
    ScalarReading,
    parse_gate_ref,
    validate_gate_channels,
)

__all__ = [
    "AGG_TAIL_DEPTH",
    "Assignment",
    "Axis",
    "DepthBasis",
    "DepthSummary",
    "Distance",
    "Fraction",
    "GateChannel",
    "GateRef",
    "InstrumentBinding",
    "InstrumentPlan",
    "InstrumentPrep",
    "LensPrep",
    "Membership",
    "ReadRequest",
    "ScalarReading",
    "parse_gate_ref",
    "validate_gate_channels",
]
