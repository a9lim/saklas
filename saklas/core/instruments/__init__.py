"""The read-side instrument protocol (phase 1 of the 5.x unification).

One contract over the three read families — geometry (Monitor subspace
probes), the Jacobian-lens readout channel, SAE feature reads.  See
``types.py`` for the shared vocabulary and ``protocol.py`` for the
Instrument / InstrumentRun contract and the division of labor.
"""

from saklas.core.instruments.protocol import Instrument, InstrumentRun, Reading
from saklas.core.instruments.types import (
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
    LensLiveConfig,
    LiveConfig,
    Membership,
    ReadRequest,
    SaeLiveConfig,
    ScalarReading,
    parse_gate_ref,
    validate_gate_channels,
)

__all__ = [
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
    "InstrumentRun",
    "LensLiveConfig",
    "LiveConfig",
    "Membership",
    "ReadRequest",
    "Reading",
    "SaeLiveConfig",
    "ScalarReading",
    "parse_gate_ref",
    "validate_gate_channels",
]
