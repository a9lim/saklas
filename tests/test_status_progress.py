"""The status vertical: every long construction step narrates.

A first ``saklas serve <model>`` runs an HF load, a neutral-corpus forward
pass for the Mahalanobis whitener, and a fit for every bundled concept not yet
fitted for the loaded weights.  All of that used to be silent — the loader's
own status goes to ``log.info``, which nothing configures for a library, and
neither the whitener build nor the probe bootstrap had a narration surface at
all.  These tests pin the ``on_progress`` chain that closes it:
``load_model`` → ``SaklasSession.__init__`` → whitener build /
``_bootstrap_manifold_probes``, plus the CLI's printing sink.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pytest
import torch

from saklas.core.session import SaklasSession


class _Sink:
    """A stable-identity progress callback that records what it is handed.

    ``list.append`` allocates a fresh bound method per attribute access, so an
    identity assertion on the threaded callback needs a real object.
    """

    def __init__(self) -> None:
        self.lines: list[str] = []

    def __call__(self, message: str) -> None:
        self.lines.append(message)


# ------------------------------------------------- the neutral / whitener leg --

def test_neutral_loader_threads_on_progress_into_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``load_or_compute_neutral_activations`` narrates the cache-miss pass."""
    import saklas.core.capture as capture_mod
    from saklas.io.alignment import load_or_compute_neutral_activations

    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path / "home"))
    seen: dict[str, Any] = {}

    def _fake_compute(
        model: object, tokenizer: object, layers: object, **kwargs: object,
    ) -> dict[int, torch.Tensor]:
        del model, tokenizer, layers
        seen["on_progress"] = kwargs.get("on_progress")
        return {0: torch.zeros(4, 3)}

    monkeypatch.setattr(capture_mod, "compute_neutral_activations", _fake_compute)

    class _Tokenizer:
        chat_template = None
        all_special_ids: list[int] = []
        added_tokens_encoder: dict[str, int] = {}
        bos_token_id = 1
        eos_token_id = 2

        def __call__(
            self, text: str, *, return_tensors: str,
            add_special_tokens: bool = False,
        ) -> dict[str, torch.Tensor]:
            del return_tensors, add_special_tokens
            return {"input_ids": torch.tensor([[3 + ord(c) % 31 for c in text]])}

    sink = _Sink()
    load_or_compute_neutral_activations(
        model=torch.nn.Module(), tokenizer=_Tokenizer(), layers=[0],
        model_id="test/model", on_progress=sink,
    )

    assert seen["on_progress"] is sink
    assert any("neutral activations" in line for line in sink.lines)


def test_compute_neutral_activations_reports_each_capture_chunk() -> None:
    """The capture loop itself emits one line per forward chunk."""
    from saklas.core import capture as capture_mod

    lines: list[str] = []
    chunks: list[tuple[int, int]] = []

    class _Ctx:
        def __enter__(self) -> "_Ctx":
            return self

        def __exit__(self, *_exc: object) -> bool:
            return False

    def _fake_batch(
        _model: object, _tokenizer: object, prompts: list[str],
        _responses: list[str], _layers: object, _device: object,
        **_kwargs: object,
    ) -> dict[int, torch.Tensor]:
        chunks.append((len(prompts), len(chunks)))
        return {0: torch.zeros(len(prompts), 3)}

    class _Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(capture_mod, "_ReusablePooledCapture", lambda *_a, **_k: _Ctx())
        patch.setattr(capture_mod, "_encode_and_capture_all_batch", _fake_batch)
        patch.setattr(
            capture_mod, "_neutral_pairs", lambda: [("p", "r")] * 5,
        )
        patch.setattr(capture_mod, "_CAPTURE_BATCH_MAX", 2)
        capture_mod.compute_neutral_activations(
            _Model(), object(), [torch.nn.Identity()],
            on_progress=lines.append,
        )

    # 5 pairs at a batch of 2 -> three chunks, three lines, last one closing at 5.
    assert len(lines) == 3
    assert lines[0].startswith("Capturing neutral activations 1-2/5")
    assert lines[-1].startswith("Capturing neutral activations 5-5/5")


def test_whitener_build_forwards_on_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``__init__``'s eager whitener build hands the callback down."""
    from saklas.io import alignment as alignment_mod

    class _Session:
        _model = object()
        _tokenizer = object()
        _layers: list[Any] = []
        _model_info = {"model_id": "test/model"}
        _layer_means: dict[int, torch.Tensor] = {}

    seen: dict[str, Any] = {}

    def _load(*_args: Any, **kwargs: Any) -> dict[int, torch.Tensor]:
        seen["on_progress"] = kwargs.get("on_progress")
        return {0: torch.randn(6, 4)}

    monkeypatch.setattr(
        alignment_mod, "load_or_compute_neutral_activations", _load,
    )
    sink = _Sink()
    whitener = SaklasSession._build_whitener_from_cache_or_compute(
        _Session(), sink,  # pyright: ignore[reportArgumentType]
    )
    assert whitener is not None
    assert seen["on_progress"] is sink


# ------------------------------------------------------ the probe-roster leg --

class _BootstrapSession:
    """The exact ``_bootstrap_manifold_probes`` collaborators, nothing else."""

    def __init__(self, fitted: set[str]) -> None:
        self.model_id = "test/model"
        self._fitted = fitted
        self._manifolds: dict[str, object] = {}
        self.fit_calls: list[tuple[Any, Any]] = []

    def ensure_manifold_loaded(self, key: str) -> None:
        from saklas.core.session import ManifoldNotRegisteredError

        if key not in self._fitted:
            raise ManifoldNotRegisteredError(key)
        self._manifolds[key] = object()

    def fit(self, folder: Any, *, on_progress: Any = None, **_kw: Any) -> None:
        self.fit_calls.append((folder, on_progress))
        self._fitted.add(f"default/{Path(folder).name}")


def _patch_roster(
    monkeypatch: pytest.MonkeyPatch, roster: dict[str, list[str]],
) -> None:
    import saklas.io.probes_bootstrap as probes_mod

    monkeypatch.setattr(probes_mod, "load_default_manifolds", lambda: roster)


def test_bootstrap_probes_announces_roster_then_each_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unfitted roster leads with the count, then narrates per concept."""
    _patch_roster(monkeypatch, {"epistemic": ["confident.uncertain", "honest.deceptive"]})
    session = _BootstrapSession(fitted=set())
    sink = _Sink()

    SaklasSession._bootstrap_manifold_probes(
        session,  # pyright: ignore[reportArgumentType]
        ["epistemic"], on_progress=sink,
    )

    assert sink.lines[0] == "Fitting default probe roster (2 concepts)..."
    assert "Fitting probe 'confident.uncertain'..." in sink.lines
    assert "Fitting probe 'honest.deceptive'..." in sink.lines
    # Every fit gets the same sink, so per-batch fit progress lands too.
    assert len(session.fit_calls) == 2
    assert all(cb is sink for _folder, cb in session.fit_calls)


def test_bootstrap_probes_counts_the_roster_once_across_categories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A concept tagged in two requested categories is one fit, counted once."""
    _patch_roster(monkeypatch, {"a": ["formal.casual"], "b": ["formal.casual"]})
    session = _BootstrapSession(fitted=set())
    lines: list[str] = []

    SaklasSession._bootstrap_manifold_probes(
        session,  # pyright: ignore[reportArgumentType]
        ["a", "b"], on_progress=lines.append,
    )

    assert lines[0] == "Fitting default probe roster (1 concepts)..."
    assert len(session.fit_calls) == 1


def test_bootstrap_probes_is_silent_without_a_callback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """The library default stays quiet — narration is opt-in."""
    _patch_roster(monkeypatch, {"epistemic": ["confident.uncertain"]})
    session = _BootstrapSession(fitted=set())

    SaklasSession._bootstrap_manifold_probes(
        session,  # pyright: ignore[reportArgumentType]
        ["epistemic"],
    )
    assert capsys.readouterr().out == ""
    assert session.fit_calls[0][1] is None


def test_bootstrap_probes_already_fitted_roster_says_nothing_per_concept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A warm cache still announces the roster but runs (and reports) no fit."""
    _patch_roster(monkeypatch, {"epistemic": ["confident.uncertain"]})
    session = _BootstrapSession(fitted={"default/confident.uncertain"})
    lines: list[str] = []

    SaklasSession._bootstrap_manifold_probes(
        session,  # pyright: ignore[reportArgumentType]
        ["epistemic"], on_progress=lines.append,
    )

    assert lines == ["Fitting default probe roster (1 concepts)..."]
    assert session.fit_calls == []


# ------------------------------------------------------------ the loader leg --

def test_load_model_reports_device_and_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``log.info`` status lines reach a caller-supplied sink.

    A library must not configure global logging, so the loader's device /
    weights / memory lines are invisible by default; the sink is how the CLI
    sees them.
    """
    import saklas.core.model as model_mod

    plan = argparse.Namespace(
        model_id="test/model", device="cpu", dtype=torch.float32,
        tokenizer_kwargs={}, pin_kwargs={}, revision=None,
    )
    monkeypatch.setattr(model_mod, "detect_device", lambda d: "cpu")
    monkeypatch.setattr(
        model_mod, "_resolve_load_plan", lambda *_a, **_k: plan,
    )
    monkeypatch.setattr(
        model_mod.AutoTokenizer, "from_pretrained",
        staticmethod(lambda *_a, **_k: object()),
    )
    monkeypatch.setattr(model_mod, "_materialize_model", lambda _plan: object())
    monkeypatch.setattr(
        model_mod, "_finalize_model",
        lambda model, _tok, _plan, *, compile, compile_mode, on_progress=None: (
            on_progress("Memory used: 1.50 GB") or model
            if on_progress is not None else model
        ),
    )

    lines: list[str] = []
    model_mod.load_model("test/model", device="cpu", on_progress=lines.append)

    assert lines[0] == "Device: cpu"
    assert any("Loading weights" in line for line in lines)
    assert "Memory used: 1.50 GB" in lines


# --------------------------------------------------------------- the CLI leg --

def test_progress_printer_prints_indented(capsys: pytest.CaptureFixture[str]) -> None:
    from saklas.cli.runners import _progress_printer

    sink = _progress_printer(argparse.Namespace())
    assert sink is not None
    sink("Device: mps")
    assert capsys.readouterr().out == "  Device: mps\n"


def test_progress_printer_suppressed_under_json_output() -> None:
    """A ``-j`` verb keeps stdout parseable."""
    from saklas.cli.runners import _progress_printer

    assert _progress_printer(argparse.Namespace(json_output=True)) is None
    assert _progress_printer(argparse.Namespace(json_output=False)) is not None


def test_make_session_passes_the_printing_sink(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_make_session`` is where every model-loading verb picks up narration."""
    import saklas.core.session as session_mod
    from saklas.cli.runners import _make_session

    seen: dict[str, Any] = {}

    def _from_pretrained(model_id: str, **kwargs: Any) -> object:
        seen.update(kwargs, model_id=model_id)
        return object()

    monkeypatch.setattr(
        session_mod.SaklasSession, "from_pretrained",
        staticmethod(_from_pretrained),
    )
    args = argparse.Namespace(
        model="test/model", device="auto", quantize=None, probes=None,
    )
    _make_session(args)

    assert callable(seen["on_progress"])


def test_manifold_extract_runner_narrates(monkeypatch: pytest.MonkeyPatch) -> None:
    """``manifold extract`` passes ``on_progress`` like ``fit`` / ``generate``.

    Extract runs up to 96 in-character generations plus a full fit; without
    the callback it printed the model info and then nothing until the final
    "extracted" line.
    """
    import saklas.cli.runners as runners_pkg
    from saklas.cli.runners.manifold import _run_manifold_extract

    captured: dict[str, Any] = {}

    class _Session:
        def extract(self, *args: Any, **kwargs: Any) -> tuple[str, object]:
            captured.update(kwargs)
            captured["args"] = args
            return "happy.sad", object()

    monkeypatch.setattr(runners_pkg, "_print_startup", lambda _a: None)
    monkeypatch.setattr(
        runners_pkg, "_make_session", lambda _a, **_k: _Session(),
    )
    monkeypatch.setattr(runners_pkg, "_print_model_info", lambda _s: None)

    _run_manifold_extract(argparse.Namespace(
        concept=["happy", "sad"], model="test/model", force=False,
        sae=None, role=None, namespace=None, kind="abstract",
        custom_system=None, manifold_cmd="extract",
    ))

    assert callable(captured["on_progress"])
    assert captured["kind"] == "abstract"
    assert captured["custom_system"] is None


def test_manifold_extract_runner_rejects_custom_without_system(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Same rejection ``manifold generate`` applies, before the model loads."""
    from saklas.cli.runners.manifold import _run_manifold_extract

    with pytest.raises(SystemExit) as exc:
        _run_manifold_extract(argparse.Namespace(
            concept=["january", "july"], model="test/model", force=False,
            sae=None, role=None, namespace=None, kind="custom",
            custom_system=None, manifold_cmd="extract",
        ))
    assert exc.value.code == 2
    assert "--kind custom requires --system" in capsys.readouterr().err


def test_pack_install_runner_narrates(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """``pack install`` prints each stage the io layer reports."""
    import saklas.io.hf_manifolds as hfm
    from saklas.cli.runners.pack import _run_pack_install

    def _install(
        target: str, as_: Any = None, *, force: bool = False,
        on_progress: Any = None,
    ) -> Path:
        assert on_progress is not None
        on_progress("Downloading alice/mood from Hugging Face...")
        on_progress("Validating staged alice/mood...")
        return Path("/home/.saklas/manifolds/local/mood")

    monkeypatch.setattr(hfm, "install_manifold", _install)
    _run_pack_install(argparse.Namespace(
        target="alice/mood", as_target=None, force=False,
    ))

    out = capsys.readouterr().out
    assert "  Downloading alice/mood from Hugging Face..." in out
    assert "  Validating staged alice/mood..." in out
    assert "Installed alice/mood ->" in out


def test_install_manifold_reports_every_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The io entry point narrates resolve → download for an HF coord."""
    import saklas.io.hf_manifolds as hfm

    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path / "home"))

    def _fake_pull(
        coord: str, target_folder: Path, *, force: bool, revision: Any,
        on_progress: Any = None,
    ) -> Path:
        assert on_progress is not None
        on_progress(f"Downloading {coord} from Hugging Face...")
        target_folder.mkdir(parents=True, exist_ok=True)
        return target_folder

    monkeypatch.setattr(hfm, "pull_manifold", _fake_pull)
    lines: list[str] = []
    hfm.install_manifold("alice/mood", on_progress=lines.append)

    assert lines[0] == "Resolving alice/mood..."
    assert "Downloading alice/mood from Hugging Face..." in lines
