"""Golden payloads for the one read-side wire record.

The measurement envelope (``core/measurements.py``) is carried by the native
WS ``token`` and ``done`` frames, persisted loom token rows, and all three
replay endpoints.  Its contract used to live only in hand-mirrored
TypeScript, which is how a never-read aggregate block and a unit no producer
held both survived unnoticed.

These tests assert **exact key sets**, so wire drift is a test failure rather
than a dashboard bug.  They also pin the 5.x/6.x reading split: geometry
carries the full whitened ``ProbeReading``, the single-axis families carry
their native ``ScalarReading`` (``value``/``unit``/``per_layer``/``depth``)
and NOT eight constant geometry fields.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from saklas.core.instruments.types import DepthSummary, ScalarReading
from saklas.core.measurements import MEASUREMENTS_VERSION, build_measurements
from saklas.core.results import ProbeReading

# The eight fields a synthesized ``ProbeReading`` used to ship for every
# lens/SAE probe on every token — constants masquerading as measurements.
_FAKE_GEOMETRY_FIELDS = frozenset({
    "fraction", "residual", "nearest", "assignment", "membership",
    "fraction_per_layer", "residual_per_layer", "subspace_coords_per_layer",
})


def _geometry_reading() -> ProbeReading:
    return ProbeReading(
        fraction=0.42,
        nearest=[("formal", 1.25)],
        coords=(0.61, -0.2),
        residual=0.03,
        coords_per_layer={12: (0.5, -0.1), 18: (0.7, -0.3)},
        depth_com=(0.55, 0.5),
        depth_spread=(0.1, 0.2),
    )


def _lens_reading() -> ScalarReading:
    return ScalarReading(
        value=0.0125,
        unit="mean_token_probability",
        per_layer={12: 0.01, 18: 0.015},
        depth=DepthSummary(
            center=(0.62,), spread=(0.08,), basis="readout_probability_mass",
        ),
    )


def _sae_reading() -> ScalarReading:
    return ScalarReading(
        value=0.84,
        unit="activation_over_max",
        per_layer={14: 0.84},
        depth=DepthSummary(
            center=(0.56,), spread=(0.0,), basis="single_layer",
        ),
    )


# ---------------------------------------------------------------------------
# The native reading shapes
# ---------------------------------------------------------------------------

class TestReadingShapes:
    def test_scalar_reading_wire_keys(self) -> None:
        wire = _lens_reading().to_dict()
        assert set(wire) == {"value", "unit", "per_layer", "depth"}
        assert wire["value"] == pytest.approx(0.0125)
        assert wire["unit"] == "mean_token_probability"
        assert wire["per_layer"] == {"12": 0.01, "18": 0.015}
        assert set(wire["depth"]) == {"center", "spread", "basis"}
        assert wire["depth"]["basis"] == "readout_probability_mass"

    def test_scalar_reading_carries_no_geometry_fields(self) -> None:
        """The 171 bytes of constants per probe per token are off the wire."""
        wire = _lens_reading().to_dict()
        assert not (_FAKE_GEOMETRY_FIELDS & set(wire))

    def test_depth_basis_is_family_specific(self) -> None:
        """``depth_com`` means three unrelated things, so the mass names
        itself — a consumer must never compare centers across bases."""
        assert (
            _lens_reading().to_dict()["depth"]["basis"]
            != _sae_reading().to_dict()["depth"]["basis"]
        )

    def test_compat_projection_reintroduces_them_deliberately(self) -> None:
        """``to_probe_reading`` survives ONLY at the two compatibility
        boundaries; there it must still produce the historical shape."""
        compat = _lens_reading().to_probe_reading().to_dict()
        assert _FAKE_GEOMETRY_FIELDS <= set(compat)
        assert compat["coords"] == [pytest.approx(0.0125)]
        assert compat["fraction"] == 0.0
        assert compat["membership"] == 1.0


# ---------------------------------------------------------------------------
# Golden envelopes
# ---------------------------------------------------------------------------

class TestTokenFrame:
    @staticmethod
    def _envelope() -> dict[str, Any]:
        env = build_measurements(
            scope="token",
            geometry_readings={"formal.casual": _geometry_reading()},
            lens_readings={"jlens/fake": _lens_reading()},
            sae_readings={"sae/548": _sae_reading()},
            per_layer_scores={"12": {"formal.casual": 0.5}},
            lens_readout={12: [(" a", 0.78)], 18: [(" b", 0.6)]},
            lens_aggregate=[(" a", 0.41, 0.31, 0.05)],
            lens_token_ids={12: [5], 18: [7]},
            lens_source="local:default",
            sae_features=[(42, 3.5, "fruit", 121.1)],
            sae_source="saelens:scope",
            sae_layer=14,
            steering="0.3 formal.casual",
        )
        assert env is not None
        return dict(env)

    def test_top_level_keys(self) -> None:
        env = self._envelope()
        assert set(env) == {
            "version", "scope", "provenance", "instruments",
            "scores", "per_layer_scores",
        }
        assert env["version"] == MEASUREMENTS_VERSION
        assert env["scope"] == "token"
        assert env["provenance"] == "captured"

    def test_families_and_channel_keys(self) -> None:
        inst = self._envelope()["instruments"]
        assert set(inst) == {"geometry", "lens", "sae"}
        assert set(inst["geometry"]) == {"readings"}
        assert set(inst["lens"]) == {"readings", "readout", "binding"}
        assert set(inst["sae"]) == {"readings", "readout", "binding"}

    def test_bindings(self) -> None:
        inst = self._envelope()["instruments"]
        assert inst["lens"]["binding"] == {
            "source": "local:default", "steering": "0.3 formal.casual",
        }
        assert inst["sae"]["binding"] == {
            "source": "saelens:scope",
            "steering": "0.3 formal.casual",
            "layer": 14,
        }

    def test_readings_are_family_native(self) -> None:
        inst = self._envelope()["instruments"]
        geo = inst["geometry"]["readings"]["formal.casual"]
        assert "coords" in geo and "fraction" in geo
        lens = inst["lens"]["readings"]["jlens/fake"]
        assert set(lens) == {"value", "unit", "per_layer", "depth"}
        sae = inst["sae"]["readings"]["sae/548"]
        assert sae["unit"] == "activation_over_max"

    def test_flat_scores_join_all_three_families(self) -> None:
        env = self._envelope()
        assert env["scores"] == {
            "formal.casual": pytest.approx(0.61),
            "jlens/fake": pytest.approx(0.0125),
            "sae/548": pytest.approx(0.84),
        }

    def test_lens_readout_rows(self) -> None:
        readout = self._envelope()["instruments"]["lens"]["readout"]
        assert set(readout) == {"layers", "aggregate"}
        assert [row["layer"] for row in readout["layers"]] == [12, 18]
        assert set(readout["layers"][0]["tokens"][0]) == {
            "token", "id", "logprob",
        }
        assert set(readout["aggregate"][0]) == {
            "token", "strength", "com", "spread",
        }

    def test_sae_readout_rows(self) -> None:
        features = self._envelope()["instruments"]["sae"]["readout"]["features"]
        assert set(features[0]) == {"id", "activation", "label", "max_act"}


class TestDoneAggregate:
    """The ``done`` frame's envelope — built once by the engine at finalize
    (``GenerationResult.measurements``), forwarded verbatim by the server."""

    @staticmethod
    def _envelope() -> dict[str, Any]:
        env = build_measurements(
            scope="aggregate",
            geometry_readings={"formal.casual": _geometry_reading()},
            lens_readings={"jlens/fake": _lens_reading()},
            sae_readings={"sae/548": _sae_reading()},
            lens_source="local:default",
            sae_source="saelens:scope",
            sae_layer=14,
            steering="0.3 formal.casual",
        )
        assert env is not None
        return dict(env)

    def test_top_level_keys(self) -> None:
        env = self._envelope()
        assert set(env) == {
            "version", "scope", "provenance", "instruments", "scores",
        }
        assert env["scope"] == "aggregate"

    def test_no_readout_channel_without_a_discovery_surface(self) -> None:
        """An aggregate has attached-probe ``readings`` only — the native
        top-k ``readout`` is a per-step discovery surface."""
        inst = self._envelope()["instruments"]
        assert set(inst["lens"]) == {"readings", "binding"}
        assert set(inst["sae"]) == {"readings", "binding"}

    def test_none_when_nothing_measured(self) -> None:
        assert build_measurements(scope="aggregate") is None

    def test_the_result_carries_it(self) -> None:
        """The finalize path — not the server — owns the aggregate
        envelope, so the lens/SAE channels keep their native shape and the
        ``done`` frame is a forward, not a re-split."""
        from saklas.core.results import GenerationResult

        result = GenerationResult(
            text="", tokens=[], token_count=0, tok_per_sec=0.0, elapsed=0.0,
            measurements=self._envelope(),
        )
        assert result.to_dict()["measurements"]["scope"] == "aggregate"


class TestReplayEnvelopes:
    """One ``{"measurements": …}`` body per family, built by the family's
    own ``token_readout`` — the route no longer reshapes anything."""

    def test_geometry(self) -> None:
        env = build_measurements(
            scope="replay",
            provenance="replayed",
            geometry_readings={"formal.casual": _geometry_reading()},
            geometry_binding={"source": None, "steering": "0.5 formal.casual"},
        )
        assert env is not None
        assert set(env) == {
            "version", "scope", "provenance", "instruments", "scores",
        }
        assert env["scope"] == "replay"
        assert env["provenance"] == "replayed"
        assert set(env["instruments"]) == {"geometry"}
        assert set(env["instruments"]["geometry"]) == {"readings", "binding"}
        assert env["instruments"]["geometry"]["binding"] == {
            "source": None, "steering": "0.5 formal.casual",
        }

    def test_lens(self) -> None:
        env = build_measurements(
            scope="replay",
            provenance="replayed",
            lens_readout={12: [(" a", 0.78)]},
            lens_aggregate=[(" a", 0.41, 0.31, 0.05)],
            lens_token_ids={12: [5]},
            lens_source="local:default",
            steering=None,
        )
        assert env is not None
        assert set(env) == {"version", "scope", "provenance", "instruments"}
        assert set(env["instruments"]) == {"lens"}
        assert set(env["instruments"]["lens"]) == {"readout", "binding"}
        assert env["instruments"]["lens"]["binding"] == {
            "source": "local:default", "steering": None,
        }

    def test_sae(self) -> None:
        env = build_measurements(
            scope="replay",
            provenance="replayed",
            sae_features=[(42, 3.5, "fruit", 121.1)],
            sae_source="saelens:scope",
            sae_layer=14,
            steering="0.2 sae/42",
        )
        assert env is not None
        assert set(env["instruments"]) == {"sae"}
        assert set(env["instruments"]["sae"]) == {"readout", "binding"}
        assert env["instruments"]["sae"]["binding"]["layer"] == 14

    def test_every_family_returns_the_same_envelope_wrapper(self) -> None:
        """``Instrument.token_readout`` is uniform: ``{"measurements": …}``
        from all three families, so the route dispatches without a branch."""
        from saklas.core.instruments.geometry import GeometryInstrument
        from saklas.core.instruments.lens import LensInstrument
        from saklas.core.instruments.sae import SaeInstrument

        native = {
            "steering": "0.3 formal.casual",
            "readings": {"formal.casual": _geometry_reading()},
            "readout": {},
            "aggregate": [],
            "features": [],
            "layer": 14,
        }
        session = SimpleNamespace(
            model_id="test/model",
            sae_info=None,
            geometry_token_readout=lambda *a, **k: native,
            jlens_token_readout=lambda *a, **k: native,
            sae_token_readout=lambda *a, **k: native,
        )
        for cls in (GeometryInstrument, LensInstrument, SaeInstrument):
            instrument = cls(session)  # type: ignore[arg-type]
            out = _token_readout(instrument, cls.family)
            assert set(out) == {"measurements"}


def _token_readout(instrument: Any, family: str) -> dict[str, Any]:
    """Call ``token_readout`` with only the knobs the family accepts."""
    if family == "lens":
        return instrument.token_readout("n1", 0, top_k=4, layers="all")
    if family == "sae":
        return instrument.token_readout("n1", 0, top_k=4)
    return instrument.token_readout("n1", 0)


class TestRejectedKnobs:
    """A family that cannot honor a knob rejects it (400 at the route)
    instead of dropping it silently — the route used to ignore ``top_k``
    and ``layers`` on geometry with only a comment to say so."""

    @staticmethod
    def _instrument(family: str) -> Any:
        from saklas.core.instruments.geometry import GeometryInstrument
        from saklas.core.instruments.sae import SaeInstrument

        session = SimpleNamespace(
            model_id="test/model",
            sae_info=None,
            geometry_token_readout=lambda *a, **k: {"readings": {}},
            sae_token_readout=lambda *a, **k: {"features": [], "layer": 1},
        )
        cls = GeometryInstrument if family == "geometry" else SaeInstrument
        return cls(session)  # type: ignore[arg-type]

    def test_geometry_rejects_top_k(self) -> None:
        with pytest.raises(ValueError, match="no top_k"):
            self._instrument("geometry").token_readout("n1", 0, top_k=8)

    def test_geometry_rejects_layers(self) -> None:
        with pytest.raises(ValueError, match="no layers"):
            self._instrument("geometry").token_readout("n1", 0, layers=[1, 2])

    def test_sae_rejects_layers(self) -> None:
        with pytest.raises(ValueError, match="no layers"):
            self._instrument("sae").token_readout("n1", 0, layers=[1])


# ---------------------------------------------------------------------------
# The family enumeration is one list
# ---------------------------------------------------------------------------

def test_payload_family_slots_match_the_registry() -> None:
    """``token_payloads``' per-family slots are pinned to the registry, so a
    fourth read family cannot land a slot-less payload."""
    from saklas.core.session import SaklasSession
    from saklas.core.token_payloads import _FAMILY_SLOTS

    session = SaklasSession.__new__(SaklasSession)
    assert set(_FAMILY_SLOTS) == set(session.instruments)


def test_capability_table_covers_every_family() -> None:
    from saklas.core.session import SaklasSession
    from saklas.server.instrument_routes import CAPABILITIES

    session = SaklasSession.__new__(SaklasSession)
    assert set(CAPABILITIES) == set(session.instruments)


if __name__ == "__main__":  # pragma: no cover - convenience
    raise SystemExit(pytest.main([__file__]))
