"""Shared vocabulary for the read-side instrument protocol.

The three read families — geometry (Monitor subspace probes), the Jacobian
lens readout channel, and SAE feature reads — implement one protocol
(:mod:`saklas.core.instruments.protocol`) over the types defined here.

Design (2026-07-15 instrument unification):

* **GateRef** is the structured view of a probe-gate scalar key.  The
  steering grammar still stores the verbatim string in
  ``ProbeGate.probe`` and the per-step runtime lookup stays a plain
  ``dict.get`` on that string (hot-path discipline) — ``GateRef`` exists
  for *composition preflight*: each family validates that a referenced
  channel is one it can actually produce, so ``@when:sae/123:membership``
  is a 400 at parse/compose time instead of a silently-inactive gate.
  ``parse_gate_ref`` is the ONE place the key-shape discrimination lives
  (it inverts the key family ``Monitor.flat_scalars`` emits).
* **ScalarReading** is the honest reading shape for the single-channel
  families (lens strength, SAE feature activation).  The geometry family
  keeps the full :class:`~saklas.core.results.ProbeReading`.  Until the
  wire flips (phase 2), ``ScalarReading.to_probe_reading()`` reproduces
  the historical synthesized-``ProbeReading`` shape bit-for-bit, so the
  current wire is unchanged while the type system stops pretending a
  feature activation has a ``fraction``.
* **DepthSummary carries its basis** because ``depth_com`` means three
  mathematically unrelated things across families (share-weighted
  coordinate mass / readout probability mass / a single-layer constant);
  a bare float invites cross-family comparison that is meaningless.
* **InstrumentPlan declares capture demand, not mechanics** — the
  session-side capture planner unions plans and picks physical retention
  (incremental vs tail ring vs full stack); the
  ``INCREMENTAL -> set_tail_with_sink`` upgrade is cross-instrument
  resource sharing and deliberately stays out of the families.
* **InstrumentBinding is an immutable per-generation snapshot** of
  source identity + attached specs, so mid-generation mutations (e.g.
  the SAE metadata backfill, which runs without the generation lock)
  cannot change what a running generation is measuring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, TYPE_CHECKING, Union

from saklas.core.errors import UnsupportedProbeChannelError

if TYPE_CHECKING:
    from saklas.core.results import ProbeReading


InstrumentFamily = Literal["geometry", "lens", "sae"]

#: Depth of the bounded capture tail ring finalize aggregates pool from —
#: deep enough to walk back past trailing special tokens to the last
#: content token.  Declared here (the plan vocabulary) so instruments can
#: state the ring depth they demand; the session planner's own uses alias
#: this value.
AGG_TAIL_DEPTH = 8

DepthBasis = Literal[
    # geometry: mass = share_weight_L * |coord_L[axis]| (monitor _depth_stats)
    "share_weighted_coord_mass",
    # lens: mass = the token's per-layer readout probability p_l
    "readout_probability_mass",
    # sae: the resident hook layer's normalized depth, a constant
    "single_layer",
]


# --------------------------------------------------------------------------
# Gate channels
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Axis:
    """A coordinate axis of the probe's primary reading (axis 0 is the
    bare-name channel; lens/SAE readings have exactly this one axis)."""

    index: int = 0


@dataclass(frozen=True)
class Fraction:
    """The manifold subspace-fraction channel (``<probe>:fraction``)."""


@dataclass(frozen=True)
class Membership:
    """The fuzzy-manifold tube-density channel (``<probe>:membership``)."""


@dataclass(frozen=True)
class Distance:
    """Negated whitened distance to a named node (``<probe>@<label>``)."""

    label: str


@dataclass(frozen=True)
class Assignment:
    """Soft node-assignment probability (``<probe>~<label>``)."""

    label: str


GateChannel = Union[Axis, Fraction, Membership, Distance, Assignment]

_CHANNEL_WORDS = {
    Axis: "coordinate axis",
    Fraction: "subspace fraction",
    Membership: "tube membership",
    Distance: "label distance",
    Assignment: "soft assignment",
}


@dataclass(frozen=True)
class GateRef:
    """Structured view of one probe-gate scalar key.

    ``probe`` is the full attached probe name exactly as registered
    (namespace and variant segments included — ``jlens/fake``,
    ``default/emotions``, ``pirate:role-x``).  ``scalar_key()`` formats
    back to the canonical string ``Monitor.flat_scalars`` emits.

    Round-trip: ``parse_gate_ref(ref.scalar_key()) == ref`` always;
    ``parse_gate_ref(key).scalar_key() == key`` except that an explicit
    ``[0]`` axis normalizes to the bare-name form (both spellings read
    the same channel).
    """

    probe: str
    channel: GateChannel

    def scalar_key(self) -> str:
        ch = self.channel
        if isinstance(ch, Axis):
            return self.probe if ch.index == 0 else f"{self.probe}[{ch.index}]"
        if isinstance(ch, Fraction):
            return f"{self.probe}:fraction"
        if isinstance(ch, Membership):
            return f"{self.probe}:membership"
        if isinstance(ch, Distance):
            return f"{self.probe}@{ch.label}"
        return f"{self.probe}~{ch.label}"

    def describe_channel(self) -> str:
        return _CHANNEL_WORDS[type(self.channel)]


def parse_gate_ref(key: str) -> GateRef:
    """Parse a verbatim gate scalar key into its structured form.

    The inverse of the key family ``Monitor.flat_scalars`` emits.  Safe
    on namespaced (``ns/name``) and variant-suffixed (``name:role-x``)
    probe names: ``@`` and ``~`` cannot appear inside a probe name or a
    node label (``NAME_REGEX`` forbids both), and ``:`` is treated as a
    channel separator only for the two exact channel words.
    """
    if key.endswith("]"):
        head, bracket, idx = key[:-1].rpartition("[")
        if bracket and head and idx.isdigit():
            return GateRef(head, Axis(int(idx)))
    if key.endswith(":fraction"):
        return GateRef(key[: -len(":fraction")], Fraction())
    if key.endswith(":membership"):
        return GateRef(key[: -len(":membership")], Membership())
    for sep, ctor in (("@", Distance), ("~", Assignment)):
        if sep in key:
            head, _, label = key.partition(sep)
            if head and label:
                return GateRef(head, ctor(label))
    return GateRef(key, Axis(0))


def validate_gate_channels(
    ref: GateRef,
    allowed: tuple[type, ...],
    *,
    family: str,
) -> None:
    """Raise :class:`UnsupportedProbeChannelError` when ``ref.channel``
    is not one of ``allowed`` for this family.

    The runtime rule is unchanged for *supported* channels: a valid
    channel whose value is temporarily absent this step (prefill, capture
    not yet available) stays quietly inactive.  This guard rejects only
    configuration that can never fire.
    """
    if not isinstance(ref.channel, allowed):
        raise UnsupportedProbeChannelError(
            f"probe gate '{ref.scalar_key()}' references the "
            f"{ref.describe_channel()} channel, which a {family} probe "
            f"does not produce"
        )


# --------------------------------------------------------------------------
# Readings
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class DepthSummary:
    """Per-axis depth center of mass (+ spread) with its mass basis.

    ``basis`` names what the mass *is* — the three families weight depth
    by mathematically unrelated quantities, so a consumer must never
    compare centers across bases.
    """

    center: tuple[float, ...]
    spread: tuple[float, ...]
    basis: DepthBasis


@dataclass
class ScalarReading:
    """One-channel reading for the lens and SAE families.

    ``unit`` makes the scalar's meaning explicit:

    * ``"mean_token_probability"`` — lens strength, ``mean_l p_l(v)`` in
      ``[0, 1]``;
    * ``"activation_over_max"`` — SAE activation normalized by the
      Neuronpedia ``maxActApprox`` corpus max, ~``[0, 1]``;
    * ``"raw_activation"`` — SAE activation with no metadata available
      (offline / unlisted feature), comparable across tokens for one
      feature only.
    """

    value: float
    unit: str
    per_layer: dict[int, float] = field(default_factory=dict)
    depth: DepthSummary | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_probe_reading(self) -> "ProbeReading":
        """The phase-1 wire bridge: the exact synthesized-``ProbeReading``
        shape the lens/SAE families emitted historically (geometry fields
        defaulted — ``fraction``/``residual`` 0, ``nearest``/``assignment``
        empty, ``membership`` 1.0), so the current wire and gate surfaces
        are unchanged until the measurement envelope lands in phase 2."""
        from saklas.core.results import ProbeReading

        return ProbeReading(
            fraction=0.0,
            nearest=[],
            coords=(self.value,),
            residual=0.0,
            coords_per_layer={
                layer: (v,) for layer, v in self.per_layer.items()
            },
            depth_com=tuple(self.depth.center) if self.depth else (),
            depth_spread=tuple(self.depth.spread) if self.depth else (),
        )


# --------------------------------------------------------------------------
# Plans, preps, bindings, live configs
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ReadRequest:
    """What the session knows about a generation's read demand when it
    plans capture — the input to ``Instrument.prepare`` and
    ``Instrument.plan``."""

    gate_keys: frozenset[str] = frozenset()
    live: bool = False
    per_token_consumers: bool = False
    final_aggregate: bool = True
    batch: bool = False
    return_hidden: bool = False


@dataclass(frozen=True)
class InstrumentPrep:
    """A family's generation-boundary source snapshot — produced by
    ``Instrument.prepare`` (after the prior run is closed), consumed by
    the same family's ``plan`` and ``bind``.

    Family-opaque to the session planner: it threads the prep from
    prepare through plan into bind without reading it, so
    source-boundary work (a disk refresh, a pin decision, a spec
    snapshot) is protocol shape rather than session special-casing.
    ``request`` carries the generation's read demand forward — ``plan``
    derives solely from the prep, so a live-registry mutation landing
    between prepare and bind cannot desynchronize the plan from the
    binding (sol's round-3 P1).
    """

    family: str
    request: ReadRequest = field(default_factory=ReadRequest)


@dataclass(frozen=True)
class LensPrep(InstrumentPrep):
    """The lens family's prep: the refresh/pin decision AND the
    authoritative source/spec snapshot.

    ``prepare`` reads the disk-refreshing ``session.jlens`` getter under
    pin demand — the adoption path rewrites live probe layer lists when
    an external replacement lens landed, which is exactly why the read
    must precede planning and the spec freeze.  Because the pin itself
    only lands at ``bind``, an interleaved **unpinned** getter read (the
    un-locked ``has_compatible_jlens`` on the session-info route) can
    adopt a *newer* disk lens inside the prepare→bind window and rewrite
    the live registry again; the prep therefore carries everything
    downstream steps may consume:

    * ``lens`` — the refreshed resident lens (``None`` when demand was
      absent or the artifact is validated-missing); ``pinned`` records
      the demand itself, so per-token paths never reopen the sidecar
      even for a vanished lens.
    * ``specs`` — the probe registry snapshot with ``layers`` derived
      from ``lens`` (the prepared identity), never reread from the live
      registry by ``plan``/``bind``.
    * ``live_state`` — the live-readout runtime dict as of prepare
      (adoption rebuilds it against the new lens; the bound run must
      keep reading the one that matches its pin).
    """

    lens: Any = None
    pinned: bool = False
    specs: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    live_state: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class InstrumentPlan:
    """A family's declared capture demand for one generation.

    Demand, not mechanics: the session-side planner unions plans across
    instruments and chooses the physical retention mode
    (``set_incremental`` / ``set_tail_with_sink`` / ``set_aggregate_tail``
    / full stack).  Cross-instrument interactions (a lens aggregate
    upgrading geometry's incremental capture to a tail ring) are the
    planner's business, never a family's.
    """

    family: str
    latest_layers: frozenset[int] = frozenset()
    tail_layers: frozenset[int] = frozenset()
    tail_depth: int = 0
    prompt_layers: frozenset[int] = frozenset()
    per_step: bool = False
    gate_keys: frozenset[str] = frozenset()
    final_aggregate: bool = False
    batch_aggregate: bool = False


@dataclass(frozen=True)
class InstrumentBinding:
    """Immutable per-generation snapshot of what a run is measuring:
    source identity, its disk/revision fingerprint, and the attached
    probe specs frozen at bind time."""

    family: str
    source: str | None = None
    fingerprint: str | None = None
    specs: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class GeometryLiveConfig:
    """User intent for per-token monitor scoring (the CAA live toggle)."""

    enabled: bool = True


@dataclass(frozen=True)
class LensLiveConfig:
    """User intent for the live J-lens readout; ``layers`` empty means
    every fitted layer.  Device residency of ``J_l`` is runtime state,
    not intent — disabling does not evict transported stacks."""

    enabled: bool = False
    layers: tuple[int, ...] = ()


@dataclass(frozen=True)
class SaeLiveConfig:
    """User intent for live SAE feature discovery."""

    enabled: bool = False
    top_k: int = 12


LiveConfig = Union[GeometryLiveConfig, LensLiveConfig, SaeLiveConfig]


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
    "InstrumentBinding",
    "InstrumentFamily",
    "InstrumentPlan",
    "LensLiveConfig",
    "LiveConfig",
    "Membership",
    "ReadRequest",
    "SaeLiveConfig",
    "ScalarReading",
    "parse_gate_ref",
    "validate_gate_channels",
]
