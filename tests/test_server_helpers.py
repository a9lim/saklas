"""Tests for the shared server request/response helpers.

These cover the normalizations the three protocols (OpenAI, Ollama, native
WebSocket) share through :mod:`saklas.server.request_helpers` — the single
``SamplingConfig`` construction site — plus the streaming aggregate helpers.
"""

from __future__ import annotations

from typing import Any

import pytest

from saklas.server.request_helpers import (
    build_sampling_config,
    logprob_count,
    normalize_stop,
)


class TestNormalizeStop:
    def test_none_passes_through(self) -> None:
        assert normalize_stop(None) is None

    def test_bare_string_becomes_one_tuple(self) -> None:
        assert normalize_stop("###") == ("###",)

    def test_list_becomes_tuple(self) -> None:
        assert normalize_stop(["a", "b"]) == ("a", "b")

    def test_non_string_members_are_stringified(self) -> None:
        assert normalize_stop([1, "b"]) == ("1", "b")

    def test_empty_sequence_is_none(self) -> None:
        assert normalize_stop([]) is None
        assert normalize_stop(()) is None


class TestLogprobCount:
    def test_none_disables(self) -> None:
        assert logprob_count(None) is None

    def test_chat_false_disables(self) -> None:
        """chat sends a bool; False means "no logprobs at all"."""
        assert logprob_count(False, 5) is None

    def test_chat_true_takes_top_logprobs(self) -> None:
        assert logprob_count(True, 5) == 5

    def test_chat_true_without_top_logprobs_is_chosen_only(self) -> None:
        assert logprob_count(True, None) == 0

    def test_completions_int_passes_through(self) -> None:
        """completions sends the alternative count directly."""
        assert logprob_count(3) == 3

    def test_completions_zero_is_chosen_only_not_disabled(self) -> None:
        assert logprob_count(0) == 0


class TestBuildSamplingConfig:
    def test_normalizes_stop_and_logprobs_in_one_place(self) -> None:
        sc = build_sampling_config(
            temperature=0.7,
            stop="###",
            logprobs=True,
            top_logprobs=4,
        )
        assert sc.temperature == 0.7
        assert sc.stop == ("###",)
        assert sc.logprobs == 4

    def test_penalty_none_collapses_to_zero(self) -> None:
        sc = build_sampling_config(
            presence_penalty=None,  # pyright: ignore[reportArgumentType]  # protocols hand through a possibly-None number
            frequency_penalty=None,  # pyright: ignore[reportArgumentType]
        )
        assert sc.presence_penalty == 0.0
        assert sc.frequency_penalty == 0.0

    def test_blank_role_labels_become_none(self) -> None:
        sc = build_sampling_config(user_role="", assistant_role="")
        assert sc.user_role is None
        assert sc.assistant_role is None

    def test_defaults_match_engine_neutral_sentinels(self) -> None:
        sc = build_sampling_config()
        assert sc.temperature is None
        assert sc.stop is None
        assert sc.logprobs is None
        assert sc.return_top_k == 0
        assert sc.return_probe_readings is True


class TestWsAdapterSharesTheConstructor:
    def test_ws_params_lower_onto_the_same_config(self) -> None:
        from saklas.server.ws_models import WSSamplingParams
        from saklas.server.ws_models import (
            build_sampling_config as ws_build_sampling_config,
        )

        body = WSSamplingParams(
            temperature=0.5,
            top_k=40,
            stop=["END"],
            return_top_k=3,
            persist_per_layer_scores=True,
        )
        sc = ws_build_sampling_config(body)
        assert sc is not None
        assert sc.temperature == 0.5
        assert sc.top_k == 40
        assert sc.stop == ("END",)
        assert sc.return_top_k == 3
        assert sc.persist_per_layer_scores is True

    def test_none_body_is_none(self) -> None:
        from saklas.server.ws_models import (
            build_sampling_config as ws_build_sampling_config,
        )

        assert ws_build_sampling_config(None) is None


class TestProbeMeasurementsAggregate:
    """The native WS ``done`` frame's aggregate-scope measurement envelope."""

    @staticmethod
    def _session(monitor_names: list[str], lens_names: list[str],
                 sae_names: list[str]) -> Any:
        from unittest.mock import MagicMock

        session = MagicMock()
        session.monitor.probe_names = monitor_names
        session.lens_probe_names = lens_names
        session.sae_probe_names = sae_names
        # The live source/layer binding is read through the PUBLIC instrument
        # surface, not the session's delegating ``_live_lens``/``_live_sae``
        # private aliases — a MagicMock would happily serve either, so the
        # aliases are pinned to a sentinel that would fail the assertions if
        # the helper ever reached for them again.
        session.lens.live = {"source": "local:default"}
        session.sae.live = {"source": "saelens:rel", "layer": 17}
        session._live_lens = {"source": "WRONG-private-alias"}
        session._live_sae = {"source": "WRONG-private-alias", "layer": -1}
        return session

    @staticmethod
    def _reading() -> Any:
        from saklas.core.monitor import ProbeReading

        return ProbeReading(coords=(0.5,), fraction=0.1, residual=0.0, nearest=())

    def test_none_without_result(self) -> None:
        from saklas.server.streaming import probe_measurements_aggregate

        assert probe_measurements_aggregate(self._session([], [], []), None) is None

    def test_none_without_readings(self) -> None:
        from unittest.mock import MagicMock

        from saklas.server.streaming import probe_measurements_aggregate

        result = MagicMock()
        result.probe_readings = {}
        assert probe_measurements_aggregate(
            self._session(["a"], [], []), result,
        ) is None

    def test_splits_readings_by_family_with_bindings(self) -> None:
        from unittest.mock import MagicMock

        from saklas.server.streaming import probe_measurements_aggregate

        reading = self._reading()
        result = MagicMock()
        result.probe_readings = {
            "warm.clinical": reading,
            "jlens/fake": reading,
            "sae/12": reading,
        }
        result.applied_steering = "0.5 warm.clinical"
        session = self._session(["warm.clinical"], ["jlens/fake"], ["sae/12"])

        env = probe_measurements_aggregate(session, result)
        assert env is not None
        assert env["scope"] == "aggregate"
        instruments = env["instruments"]
        assert set(instruments["geometry"]["readings"]) == {"warm.clinical"}
        assert set(instruments["lens"]["readings"]) == {"jlens/fake"}
        assert set(instruments["sae"]["readings"]) == {"sae/12"}
        assert instruments["lens"]["binding"]["source"] == "local:default"
        assert instruments["sae"]["binding"]["source"] == "saelens:rel"
        assert instruments["sae"]["binding"]["layer"] == 17

    def test_family_without_readings_carries_no_source(self) -> None:
        """A historical row stays interpretable after a source switch."""
        from unittest.mock import MagicMock

        from saklas.server.streaming import probe_measurements_aggregate

        result = MagicMock()
        result.probe_readings = {"warm.clinical": self._reading()}
        result.applied_steering = None
        session = self._session(["warm.clinical"], ["jlens/fake"], [])

        env = probe_measurements_aggregate(session, result)
        assert env is not None
        lens = env["instruments"].get("lens")
        assert lens is None or lens.get("readings") in (None, {})

    def test_bindings_read_the_public_instrument_surface(self) -> None:
        """``session.lens.live`` / ``session.sae.live``, not the private
        session aliases the helper used to reach through."""
        from unittest.mock import MagicMock

        from saklas.server.streaming import probe_measurements_aggregate

        reading = self._reading()
        result = MagicMock()
        result.probe_readings = {"jlens/fake": reading, "sae/12": reading}
        result.applied_steering = None
        session = self._session([], ["jlens/fake"], ["sae/12"])
        # Any private-alias read would surface these sentinels instead.
        del session._live_lens
        del session._live_sae

        env = probe_measurements_aggregate(session, result)
        assert env is not None
        instruments = env["instruments"]
        assert instruments["lens"]["binding"]["source"] == "local:default"
        assert instruments["sae"]["binding"]["source"] == "saelens:rel"
        assert instruments["sae"]["binding"]["layer"] == 17

    def test_readings_merge_by_name_across_families(self) -> None:
        """The dashboard's ``done`` handler merges the three families into one
        name-keyed rack update, exactly as the ``token`` path does — so the
        envelope must key every attached probe uniquely."""
        from unittest.mock import MagicMock

        from saklas.server.streaming import probe_measurements_aggregate

        reading = self._reading()
        result = MagicMock()
        result.probe_readings = {
            "warm.clinical": reading, "jlens/fake": reading, "sae/12": reading,
        }
        result.applied_steering = None
        session = self._session(["warm.clinical"], ["jlens/fake"], ["sae/12"])

        instruments = (probe_measurements_aggregate(session, result) or {})[
            "instruments"
        ]
        merged: dict[str, Any] = {}
        for family in ("geometry", "lens", "sae"):
            merged.update((instruments.get(family) or {}).get("readings") or {})
        assert set(merged) == {"warm.clinical", "jlens/fake", "sae/12"}

    def test_unattached_reading_is_dropped(self) -> None:
        """A reading whose probe was detached mid-generation belongs to no
        family and must not reach the rack."""
        from unittest.mock import MagicMock

        from saklas.server.streaming import probe_measurements_aggregate

        result = MagicMock()
        result.probe_readings = {
            "warm.clinical": self._reading(), "detached": self._reading(),
        }
        result.applied_steering = None
        session = self._session(["warm.clinical"], [], [])

        instruments = (probe_measurements_aggregate(session, result) or {})[
            "instruments"
        ]
        assert set(instruments["geometry"]["readings"]) == {"warm.clinical"}


if __name__ == "__main__":  # pragma: no cover - convenience
    raise SystemExit(pytest.main([__file__]))
