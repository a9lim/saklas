"""The unified read-side instrument protocol.

One contract over the three read families — geometry (Monitor subspace
probes), the Jacobian-lens readout channel, SAE feature reads.  See
``types.py`` for the shared vocabulary and ``protocol.py`` for the
Instrument / InstrumentRun contract and the division of labor.
"""

from saklas.core.instruments.protocol import Instrument, InstrumentRun, Reading
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
    GeometryLiveConfig,
    InstrumentBinding,
    InstrumentFamily,
    InstrumentPlan,
    InstrumentPrep,
    LensLiveConfig,
    LensPrep,
    LiveConfig,
    Membership,
    ReadRequest,
    SaeLiveConfig,
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
    "GeometryLiveConfig",
    "Instrument",
    "InstrumentBinding",
    "InstrumentFamily",
    "InstrumentPlan",
    "InstrumentPrep",
    "InstrumentRun",
    "LensLiveConfig",
    "LensPrep",
    "LiveConfig",
    "Membership",
    "ReadRequest",
    "Reading",
    "SaeLiveConfig",
    "ScalarReading",
    "parse_gate_ref",
    "validate_gate_channels",
]
