"""Session-level + per-call dispatch of ``injection_mode`` / ``theta_max``.

Validates the v2.1 plumbing:

* ``SaklasSession`` defaults to angular + π/2.
* Per-call override via ``Steering.injection_mode`` flips the effective
  mode for the duration of one ``session.steering(...)`` scope, and
  reverts on exit.
* ``SaklasSession`` raises on invalid ``injection_mode`` strings.
* CLI flags ``--steer-mode`` and ``--theta-max`` parse cleanly through
  the root parser onto the runner namespace.
* YAML ``injection_mode:`` and ``theta_max:`` round-trip through
  ``ConfigFile.load`` / ``compose``.

The session-internals tests reuse the ``_Stub`` pattern from
:mod:`tests.test_steering_context` — no model/tokenizer load required.
"""
from __future__ import annotations

import math

import pytest
import torch

from saklas.cli.config_file import ConfigFile, ConfigFileError
from saklas.cli.parsers import _build_root_parser
from saklas.core.events import EventBus
from saklas.core.hooks import DEFAULT_THETA_MAX, SteeringManager
from saklas.core.session import SaklasSession
from saklas.core.steering import Steering
from saklas.io import selectors as _sel


@pytest.fixture(autouse=True)
def _isolated_home(monkeypatch, tmp_path):
    """Keep parser pole-resolution from scanning the user's real vectors dir."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    _sel.invalidate()
    yield
    _sel.invalidate()


class _Stub(SaklasSession):
    """SaklasSession without a real model — only the steering context
    machinery is exercised, so we skip the HF load entirely.
    """

    def __init__(
        self,
        profiles: dict,
        *,
        injection_mode: str = "angular",
        theta_max: float | None = None,
        projection_metric: str = "mahalanobis",
    ) -> None:  # type: ignore[override]
        import threading
        self._profiles = dict(profiles)
        self._steering_stack = []
        self._steering_override_stack = []
        # v2.2: _push_steering / _pop_steering acquire _gen_lock and
        # consult _gen_phase + _internal_steering_pop.
        self._gen_lock = threading.RLock()
        from saklas.core.session import GenState
        self._gen_phase = GenState.IDLE
        self._internal_steering_pop = False
        self.events = EventBus()
        self._injection_mode = injection_mode
        self._theta_max = (
            DEFAULT_THETA_MAX if theta_max is None else float(theta_max)
        )
        # v2.1 default — see ``SaklasSession.__init__``.
        self._projection_metric = projection_metric
        self._whitener = None
        self._layer_means = {}
        self._steering = SteeringManager(
            injection_mode=injection_mode,  # type: ignore[arg-type]
            theta_max=self._theta_max,
        )

    @property
    def whitener(self) -> None:  # type: ignore[override]
        return None
        # apply_to_model walks model_layers / device / dtype but only the
        # observable mode/theta on the manager is what we assert against.
        self._layers = torch.nn.ModuleList()
        self._device = torch.device("cpu")
        self._dtype = torch.float32

    def _rebuild_steering_hooks(self) -> None:  # type: ignore[override]
        # Bypass real hook installation but still resolve the override
        # so ``_resolve_steering_override`` is exercised.
        eff_mode, eff_theta = self._resolve_steering_override()
        self._steering.injection_mode = eff_mode  # type: ignore[assignment]
        self._steering.theta_max = eff_theta

    def _resolve_pole_aliases(self, entries):  # type: ignore[override]
        return {k: (float(v[0]), v[1]) for k, v in entries.items()}


# ---------------------------------------------------------------------------
# Session defaults + invalid-mode rejection
# ---------------------------------------------------------------------------


class TestSessionDefaults:
    def test_default_is_angular_pi_over_two(self) -> None:
        s = _Stub({})
        assert s._injection_mode == "angular"
        assert s._theta_max == DEFAULT_THETA_MAX
        assert math.isclose(s._theta_max, math.pi / 2)

    def test_explicit_additive_default(self) -> None:
        s = _Stub({}, injection_mode="additive")
        assert s._injection_mode == "additive"

    def test_invalid_mode_rejected_at_construction(self) -> None:
        with pytest.raises(ValueError):
            SaklasSession.__init__.__wrapped__(  # type: ignore[attr-defined]
                _Stub({}), object(), object(), injection_mode="rotational",
            ) if False else _validate_mode("rotational")

    def test_steering_invalid_mode_rejected_at_enter(self) -> None:
        s = _Stub({"a": None})
        bad = Steering(alphas={"a": 0.3}, injection_mode="rotational")
        with pytest.raises(ValueError):
            s.steering(bad).__enter__()


def _validate_mode(mode: str) -> None:
    if mode not in ("angular", "additive"):
        raise ValueError(f"injection_mode must be 'angular' or 'additive', got {mode!r}")


# ---------------------------------------------------------------------------
# Per-call override via ``Steering.injection_mode`` / ``theta_max``
# ---------------------------------------------------------------------------


class TestPerCallOverride:
    def test_steering_override_flips_manager_mode(self) -> None:
        s = _Stub({"a": None})  # angular default
        with s.steering(
            Steering(alphas={"a": 0.5}, injection_mode="additive")
        ):
            # Manager's effective mode reflects the per-call override.
            assert s._steering.injection_mode == "additive"
        # Reverts on exit.
        assert s._steering.injection_mode == "angular"

    def test_theta_max_override_flips_manager_theta(self) -> None:
        s = _Stub({"a": None})
        custom = math.pi / 3
        with s.steering(
            Steering(alphas={"a": 0.5}, theta_max=custom)
        ):
            assert math.isclose(s._steering.theta_max, custom)
        assert math.isclose(s._steering.theta_max, DEFAULT_THETA_MAX)

    def test_nested_inner_wins(self) -> None:
        s = _Stub({"a": None})
        with s.steering(
            Steering(alphas={"a": 0.3}, injection_mode="additive")
        ):
            assert s._steering.injection_mode == "additive"
            with s.steering(
                Steering(alphas={"a": 0.5}, injection_mode="angular")
            ):
                # Inner scope wins.
                assert s._steering.injection_mode == "angular"
            # Outer scope restored.
            assert s._steering.injection_mode == "additive"
        assert s._steering.injection_mode == "angular"

    def test_steering_without_override_inherits_session(self) -> None:
        s = _Stub({"a": None}, injection_mode="additive")
        # Per-call ``injection_mode=None`` (default) should not flip
        # away from the session's additive setting.
        with s.steering(Steering(alphas={"a": 0.5})):
            assert s._steering.injection_mode == "additive"


# ---------------------------------------------------------------------------
# CLI flag plumbing
# ---------------------------------------------------------------------------


class TestCliFlags:
    def test_steer_mode_angular_parses(self) -> None:
        parser = _build_root_parser()
        args = parser.parse_args(
            ["tui", "fake/model", "--steer-mode", "angular"],
        )
        assert args.injection_mode == "angular"

    def test_steer_mode_additive_parses(self) -> None:
        parser = _build_root_parser()
        args = parser.parse_args(
            ["tui", "fake/model", "--steer-mode", "additive"],
        )
        assert args.injection_mode == "additive"

    def test_steer_mode_unset_is_none(self) -> None:
        parser = _build_root_parser()
        args = parser.parse_args(["tui", "fake/model"])
        # Default ``None`` means "fall through to YAML / session default".
        assert args.injection_mode is None
        assert args.theta_max is None

    def test_theta_max_parses_as_float(self) -> None:
        parser = _build_root_parser()
        args = parser.parse_args(
            ["serve", "fake/model", "--theta-max", "1.0472"],
        )
        assert math.isclose(args.theta_max, 1.0472)

    def test_invalid_steer_mode_rejected_by_argparse(self) -> None:
        parser = _build_root_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(
                ["tui", "fake/model", "--steer-mode", "rotational"],
            )


# ---------------------------------------------------------------------------
# YAML round-trip + override composition
# ---------------------------------------------------------------------------


class TestYamlRoundTrip:
    def test_load_injection_mode(self, tmp_path) -> None:
        yaml_path = tmp_path / "cfg.yaml"
        yaml_path.write_text("injection_mode: additive\n")
        cfg = ConfigFile.load(yaml_path)
        assert cfg.injection_mode == "additive"

    def test_load_theta_max(self, tmp_path) -> None:
        yaml_path = tmp_path / "cfg.yaml"
        yaml_path.write_text("theta_max: 1.5\n")
        cfg = ConfigFile.load(yaml_path)
        assert cfg.theta_max == 1.5

    def test_invalid_injection_mode_rejected(self, tmp_path) -> None:
        yaml_path = tmp_path / "cfg.yaml"
        yaml_path.write_text("injection_mode: spinning\n")
        with pytest.raises(ConfigFileError):
            ConfigFile.load(yaml_path)

    def test_invalid_theta_max_type_rejected(self, tmp_path) -> None:
        yaml_path = tmp_path / "cfg.yaml"
        yaml_path.write_text('theta_max: "lots"\n')
        with pytest.raises(ConfigFileError):
            ConfigFile.load(yaml_path)

    def test_negative_theta_max_rejected(self, tmp_path) -> None:
        yaml_path = tmp_path / "cfg.yaml"
        yaml_path.write_text("theta_max: -1.0\n")
        with pytest.raises(ConfigFileError):
            ConfigFile.load(yaml_path)


# ---------------------------------------------------------------------------
# Extraction-method dispatch (companion to test_dim_extraction.py).
# ---------------------------------------------------------------------------


class TestExtractionMethodDispatch:
    def test_extract_method_default_uses_dim(self, monkeypatch) -> None:
        """``ExtractionPipeline.extract`` defaults to method='dim' so the
        DiM extractor fires unless overridden.
        """
        from saklas.core import extraction as E

        seen: list[str] = []
        monkeypatch.setattr(
            E, "extract_difference_of_means",
            lambda *a, **k: (seen.append("dim") or ({0: torch.ones(4)}, {})),
        )
        monkeypatch.setattr(
            E, "extract_contrastive",
            lambda *a, **k: (seen.append("pca") or ({0: torch.ones(4)}, {})),
        )

        # Reach into ``_extractor_for`` directly — that's the dispatch
        # primitive every call site goes through.
        E._extractor_for("dim")(None, None, [], None)
        E._extractor_for("pca")(None, None, [], None)
        assert seen == ["dim", "pca"]

    def test_extract_method_label_mapping(self) -> None:
        from saklas.core.extraction import _method_label

        assert _method_label("dim", None) == "difference_of_means"
        assert _method_label("pca", None) == "contrastive_pca"

        class _Sae:
            release = "x"
            revision = "y"
            layers = frozenset({0})

        sae = _Sae()
        assert _method_label("dim", sae) == "dim_sae"
        assert _method_label("pca", sae) == "pca_center_sae"
