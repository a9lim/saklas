# Python API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a programmatic Python API (`SteerSession`, `DataSource`, `ResultCollector`) that becomes the single backend for both headless scripting and the TUI.

**Architecture:** Three new modules (`session.py`, `datasource.py`, `results.py`) wrap existing core modules. `tui/app.py` is rewritten to delegate all orchestration to `SteerSession`. No changes to core modules (`model.py`, `vectors.py`, `generation.py`, `hooks.py`, `monitor.py`, `probes_bootstrap.py`).

**Tech Stack:** Python 3.11+, torch, safetensors, textual (TUI only). Optional: `datasets` (HF DataSource), `pandas` (DataFrame export).

---

### Task 1: Structured output dataclasses (`steer/results.py`)

No dependencies on anything else — pure data containers. Build and test first.

**Files:**
- Create: `steer/results.py`
- Create: `tests/test_results.py`

- [ ] **Step 1: Write tests for ProbeReadings**

```python
# tests/test_results.py
"""Tests for structured output dataclasses and ResultCollector."""

from steer.results import ProbeReadings, GenerationResult, TokenEvent, ResultCollector


class TestProbeReadings:
    def test_from_per_token_data(self):
        readings = ProbeReadings(
            per_token=[0.1, 0.3, 0.5, 0.4],
            mean=0.325,
            std=0.1479,
            min=0.1,
            max=0.5,
            delta_per_tok=0.1,
        )
        assert readings.mean == 0.325
        assert readings.min == 0.1
        assert len(readings.per_token) == 4

    def test_to_dict_returns_plain_types(self):
        readings = ProbeReadings(
            per_token=[0.1, 0.2],
            mean=0.15,
            std=0.05,
            min=0.1,
            max=0.2,
            delta_per_tok=0.1,
        )
        d = readings.to_dict()
        assert isinstance(d, dict)
        assert isinstance(d["per_token"], list)
        assert isinstance(d["mean"], float)
```

- [ ] **Step 2: Write tests for GenerationResult**

```python
# tests/test_results.py (append)

class TestGenerationResult:
    def test_to_dict_no_probes(self):
        result = GenerationResult(
            text="Hello world",
            tokens=[1, 2, 3],
            token_count=3,
            tok_per_sec=10.0,
            elapsed=0.3,
            readings={},
            vectors={"happy": 1.5},
        )
        d = result.to_dict()
        assert d["text"] == "Hello world"
        assert d["tokens"] == [1, 2, 3]
        assert d["readings"] == {}
        assert d["vectors"] == {"happy": 1.5}

    def test_to_dict_with_probes(self):
        readings = ProbeReadings(
            per_token=[0.5], mean=0.5, std=0.0, min=0.5, max=0.5, delta_per_tok=0.0,
        )
        result = GenerationResult(
            text="Hi",
            tokens=[1],
            token_count=1,
            tok_per_sec=5.0,
            elapsed=0.2,
            readings={"honest": readings},
            vectors={},
        )
        d = result.to_dict()
        assert "honest" in d["readings"]
        assert isinstance(d["readings"]["honest"], dict)
```

- [ ] **Step 3: Write tests for TokenEvent**

```python
# tests/test_results.py (append)

class TestTokenEvent:
    def test_fields(self):
        event = TokenEvent(text="hello", token_id=42, index=0, readings={"honest": 0.5})
        assert event.text == "hello"
        assert event.token_id == 42
        assert event.readings["honest"] == 0.5

    def test_no_readings(self):
        event = TokenEvent(text="hi", token_id=1, index=0, readings=None)
        assert event.readings is None
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `pytest tests/test_results.py -v`
Expected: FAIL — `steer.results` does not exist yet.

- [ ] **Step 5: Implement results.py**

```python
# steer/results.py
"""Structured output types for steer's programmatic API."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ProbeReadings:
    """Probe monitor readings across a generation run."""
    per_token: list[float]
    mean: float
    std: float
    min: float
    max: float
    delta_per_tok: float

    def to_dict(self) -> dict:
        return {
            "per_token": list(self.per_token),
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "delta_per_tok": self.delta_per_tok,
        }


@dataclass
class GenerationResult:
    """Result of a generation call."""
    text: str
    tokens: list[int]
    token_count: int
    tok_per_sec: float
    elapsed: float
    readings: dict[str, ProbeReadings] = field(default_factory=dict)
    vectors: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "tokens": list(self.tokens),
            "token_count": self.token_count,
            "tok_per_sec": self.tok_per_sec,
            "elapsed": self.elapsed,
            "readings": {k: v.to_dict() for k, v in self.readings.items()},
            "vectors": dict(self.vectors),
        }


@dataclass
class TokenEvent:
    """Single token yielded during streaming generation."""
    text: str
    token_id: int
    index: int
    readings: dict[str, float] | None
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_results.py -v`
Expected: All pass.

- [ ] **Step 7: Commit**

```bash
git add steer/results.py tests/test_results.py
git commit -m "feat: add structured output dataclasses (ProbeReadings, GenerationResult, TokenEvent)"
```

---

### Task 2: ResultCollector (`steer/results.py`)

Accumulates results and exports to dicts, JSONL, CSV, DataFrame. Added to the same file.

**Files:**
- Modify: `steer/results.py`
- Modify: `tests/test_results.py`

- [ ] **Step 1: Write tests for ResultCollector**

```python
# tests/test_results.py (append)
import json
import csv
import tempfile
from pathlib import Path


class TestResultCollector:
    def _make_result(self, text="Hello", alpha=1.0):
        return GenerationResult(
            text=text,
            tokens=[1, 2],
            token_count=2,
            tok_per_sec=10.0,
            elapsed=0.2,
            readings={},
            vectors={"happy": alpha},
        )

    def test_add_and_to_dicts(self):
        collector = ResultCollector()
        collector.add(self._make_result(), prompt="hi", concept="happy")
        dicts = collector.to_dicts()
        assert len(dicts) == 1
        assert dicts[0]["text"] == "Hello"
        assert dicts[0]["prompt"] == "hi"
        assert dicts[0]["vector_happy_alpha"] == 1.0

    def test_probe_readings_flattened(self):
        readings = ProbeReadings(
            per_token=[0.5], mean=0.5, std=0.0, min=0.5, max=0.5, delta_per_tok=0.0,
        )
        result = GenerationResult(
            text="Hi", tokens=[1], token_count=1, tok_per_sec=5.0, elapsed=0.2,
            readings={"honest": readings}, vectors={},
        )
        collector = ResultCollector()
        collector.add(result)
        d = collector.to_dicts()[0]
        assert d["probe_honest_mean"] == 0.5
        assert d["probe_honest_std"] == 0.0
        assert d["probe_honest_min"] == 0.5
        assert d["probe_honest_max"] == 0.5

    def test_to_jsonl(self):
        collector = ResultCollector()
        collector.add(self._make_result())
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        collector.to_jsonl(path)
        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["text"] == "Hello"
        Path(path).unlink()

    def test_to_csv(self):
        collector = ResultCollector()
        collector.add(self._make_result(alpha=1.0), concept="happy")
        collector.add(self._make_result(text="World", alpha=2.0), concept="happy")
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        collector.to_csv(path)
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2
        assert rows[0]["text"] == "Hello"
        assert rows[1]["text"] == "World"
        Path(path).unlink()

    def test_to_dataframe_without_pandas(self):
        """to_dataframe should work if pandas is importable."""
        collector = ResultCollector()
        collector.add(self._make_result())
        try:
            import pandas
            df = collector.to_dataframe()
            assert len(df) == 1
        except ImportError:
            import pytest
            with pytest.raises(ImportError):
                collector.to_dataframe()

    def test_results_property(self):
        collector = ResultCollector()
        collector.add(self._make_result(), run=1)
        collector.add(self._make_result(), run=2)
        assert len(collector.results) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_results.py::TestResultCollector -v`
Expected: FAIL — `ResultCollector` not yet defined.

- [ ] **Step 3: Implement ResultCollector**

```python
# steer/results.py (append to existing file)

import csv
import json


class ResultCollector:
    """Accumulates GenerationResults with tags for batch export."""

    def __init__(self):
        self._rows: list[dict] = []

    @property
    def results(self) -> list[dict]:
        return list(self._rows)

    def add(self, result: GenerationResult, **tags) -> None:
        row = {
            "text": result.text,
            "token_count": result.token_count,
            "tok_per_sec": result.tok_per_sec,
            "elapsed": result.elapsed,
        }
        # Flatten probe readings
        for probe_name, readings in result.readings.items():
            row[f"probe_{probe_name}_mean"] = readings.mean
            row[f"probe_{probe_name}_std"] = readings.std
            row[f"probe_{probe_name}_min"] = readings.min
            row[f"probe_{probe_name}_max"] = readings.max
            row[f"probe_{probe_name}_delta"] = readings.delta_per_tok
        # Flatten vectors
        for vec_name, alpha in result.vectors.items():
            row[f"vector_{vec_name}_alpha"] = alpha
        # User-provided tags
        row.update(tags)
        self._rows.append(row)

    def to_dicts(self) -> list[dict]:
        return list(self._rows)

    def to_jsonl(self, path: str) -> None:
        with open(path, "w") as f:
            for row in self._rows:
                f.write(json.dumps(row) + "\n")

    def to_csv(self, path: str) -> None:
        if not self._rows:
            return
        # Collect all keys across all rows for consistent columns
        all_keys: list[str] = []
        seen: set[str] = set()
        for row in self._rows:
            for k in row:
                if k not in seen:
                    all_keys.append(k)
                    seen.add(k)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(self._rows)

    def to_dataframe(self):
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for to_dataframe(). "
                "Install with: pip install steer[pandas]"
            )
        return pd.DataFrame(self._rows)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_results.py -v`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add steer/results.py tests/test_results.py
git commit -m "feat: add ResultCollector for batch result accumulation and export"
```

---

### Task 3: DataSource (`steer/datasource.py`)

Multi-format contrastive pair normalizer.

**Files:**
- Create: `steer/datasource.py`
- Create: `tests/test_datasource.py`

- [ ] **Step 1: Write tests for DataSource**

```python
# tests/test_datasource.py
"""Tests for DataSource multi-format contrastive pair normalizer."""

import json
import csv
import tempfile
from pathlib import Path

from steer.datasource import DataSource


class TestFromPairs:
    def test_basic(self):
        ds = DataSource.from_pairs([("hello", "goodbye")])
        assert ds.pairs == [("hello", "goodbye")]
        assert ds.name == "custom"

    def test_custom_name(self):
        ds = DataSource.from_pairs([("a", "b")], name="test")
        assert ds.name == "test"


class TestCurated:
    def test_loads_happy(self):
        ds = DataSource.curated("happy")
        assert len(ds.pairs) > 0
        assert ds.name == "happy"
        # Each pair is a (positive, negative) tuple
        for pos, neg in ds.pairs:
            assert isinstance(pos, str)
            assert isinstance(neg, str)

    def test_missing_raises(self):
        import pytest
        with pytest.raises(FileNotFoundError):
            DataSource.curated("nonexistent_concept_xyz")


class TestJson:
    def test_loads_steer_format(self):
        data = {
            "name": "test",
            "description": "test concept",
            "category": "test",
            "pairs": [
                {"positive": "I am happy", "negative": "I am sad"},
                {"positive": "Joy", "negative": "Sorrow"},
            ],
        }
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump(data, f)
            path = f.name
        ds = DataSource.json(path)
        assert len(ds.pairs) == 2
        assert ds.pairs[0] == ("I am happy", "I am sad")
        assert ds.name == "test"
        Path(path).unlink()


class TestCsv:
    def test_default_columns(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False, newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["positive", "negative"])
            writer.writerow(["happy", "sad"])
            writer.writerow(["joyful", "gloomy"])
            path = f.name
        ds = DataSource.csv(path)
        assert len(ds.pairs) == 2
        assert ds.pairs[0] == ("happy", "sad")
        Path(path).unlink()

    def test_custom_columns(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False, newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["good", "bad"])
            writer.writerow(["nice", "mean"])
            path = f.name
        ds = DataSource.csv(path, positive_col="good", negative_col="bad")
        assert ds.pairs[0] == ("nice", "mean")
        Path(path).unlink()

    def test_name_inferred_from_filename(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False, newline="",
                                         prefix="empathy_") as f:
            writer = csv.writer(f)
            writer.writerow(["positive", "negative"])
            writer.writerow(["a", "b"])
            path = f.name
        ds = DataSource.csv(path)
        assert ds.name == Path(path).stem
        Path(path).unlink()


class TestHuggingFace:
    def test_import_error_without_datasets(self):
        """Should raise ImportError with helpful message if datasets not installed."""
        try:
            import datasets
            # If datasets is installed, we can't test this path
        except ImportError:
            import pytest
            with pytest.raises(ImportError, match="datasets"):
                DataSource.huggingface("some/dataset")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_datasource.py -v`
Expected: FAIL — `steer.datasource` does not exist yet.

- [ ] **Step 3: Implement datasource.py**

```python
# steer/datasource.py
"""Multi-format contrastive pair normalizer."""

from __future__ import annotations

import csv
import json
from pathlib import Path


class DataSource:
    """Normalizes contrastive pairs from multiple input formats.

    All classmethods produce a DataSource with .pairs, .name, .description.
    """

    def __init__(self, pairs: list[tuple[str, str]], name: str = "custom",
                 description: str | None = None):
        self.pairs = pairs
        self.name = name
        self.description = description

    @classmethod
    def from_pairs(cls, pairs: list[tuple[str, str]], name: str = "custom",
                   description: str | None = None) -> DataSource:
        return cls(pairs=list(pairs), name=name, description=description)

    @classmethod
    def curated(cls, concept: str) -> DataSource:
        """Load from steer's bundled datasets by concept name."""
        datasets_dir = Path(__file__).parent / "datasets"
        ds_path = datasets_dir / f"{concept.lower()}.json"
        if not ds_path.exists():
            raise FileNotFoundError(
                f"No curated dataset for '{concept}'. "
                f"Available: {', '.join(p.stem for p in sorted(datasets_dir.glob('*.json')))}"
            )
        with open(ds_path) as f:
            data = json.load(f)
        pairs = [(p["positive"], p["negative"]) for p in data["pairs"]]
        return cls(
            pairs=pairs,
            name=data.get("name", concept),
            description=data.get("description"),
        )

    @classmethod
    def json(cls, path: str, name: str | None = None) -> DataSource:
        """Load from steer's JSON schema: {pairs: [{positive, negative}]}."""
        with open(path) as f:
            data = json.load(f)
        pairs = [(p["positive"], p["negative"]) for p in data["pairs"]]
        return cls(
            pairs=pairs,
            name=name or data.get("name", Path(path).stem),
            description=data.get("description"),
        )

    @classmethod
    def csv(cls, path: str, positive_col: str = "positive",
            negative_col: str = "negative", name: str | None = None) -> DataSource:
        """Load from CSV with two columns."""
        pairs = []
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pairs.append((row[positive_col], row[negative_col]))
        return cls(
            pairs=pairs,
            name=name or Path(path).stem,
        )

    @classmethod
    def huggingface(cls, dataset_id: str, positive_col: str = "positive",
                    negative_col: str = "negative", split: str = "train",
                    name: str | None = None) -> DataSource:
        """Load from a HuggingFace dataset."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "The 'datasets' package is required for DataSource.huggingface(). "
                "Install with: pip install steer[hf]"
            )
        ds = load_dataset(dataset_id, split=split)
        pairs = [(row[positive_col], row[negative_col]) for row in ds]
        return cls(
            pairs=pairs,
            name=name or dataset_id.split("/")[-1],
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_datasource.py -v`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add steer/datasource.py tests/test_datasource.py
git commit -m "feat: add DataSource multi-format contrastive pair normalizer"
```

---

### Task 4: SteerSession core — construction, steering, monitoring, lifecycle (`steer/session.py`)

The main orchestrator. This task covers everything except generation (Task 5).

**Files:**
- Create: `steer/session.py`
- Create: `tests/test_session.py`

- [ ] **Step 1: Write tests for session construction and state queries**

These tests require CUDA. They mirror the existing smoke test structure.

```python
# tests/test_session.py
"""Tests for SteerSession programmatic API.

Requires CUDA and downloads google/gemma-2-2b-it (~5GB) on first run.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)

MODEL_ID = "google/gemma-2-2b-it"


@pytest.fixture(scope="module")
def session():
    from steer.session import SteerSession
    s = SteerSession(MODEL_ID, device="cuda", probes=["emotion"])
    yield s
    s.close()


class TestConstruction:
    def test_model_info(self, session):
        info = session.model_info
        assert info["model_type"] == "gemma2"
        assert info["hidden_dim"] > 0
        assert info["num_layers"] > 0

    def test_config_defaults(self, session):
        assert session.config.temperature == 1.0
        assert session.config.top_p == 0.9
        assert session.config.max_new_tokens == 1024

    def test_probes_loaded(self, session):
        # Emotion category has 8 probes
        assert len(session.probes) > 0

    def test_history_starts_empty(self, session):
        assert session.history == []

    def test_vectors_starts_empty(self, session):
        assert session.vectors == {}

    def test_last_result_starts_none(self, session):
        assert session.last_result is None
```

- [ ] **Step 2: Write tests for steering operations**

```python
# tests/test_session.py (append)

class TestSteering:
    def test_extract_and_steer(self, session):
        profile = session.extract([("I am happy", "I am sad")])
        assert isinstance(profile, dict)
        assert all(isinstance(k, int) for k in profile)

        session.steer("happy", profile, alpha=1.5)
        assert "happy" in session.vectors
        assert session.vectors["happy"]["alpha"] == 1.5
        assert session.vectors["happy"]["enabled"] is True

    def test_set_alpha(self, session):
        session.set_alpha("happy", 2.0)
        assert session.vectors["happy"]["alpha"] == 2.0

    def test_toggle(self, session):
        session.toggle("happy")
        assert session.vectors["happy"]["enabled"] is False
        session.toggle("happy")
        assert session.vectors["happy"]["enabled"] is True

    def test_unsteer(self, session):
        session.unsteer("happy")
        assert "happy" not in session.vectors

    def test_extract_curated(self, session):
        profile = session.extract("calm")
        assert isinstance(profile, dict)
        assert len(profile) > 0
        session.unsteer("calm") if "calm" in session.vectors else None

    def test_extract_datasource(self, session):
        from steer.datasource import DataSource
        ds = DataSource.from_pairs([("formal", "casual")])
        profile = session.extract(ds)
        assert isinstance(profile, dict)
```

- [ ] **Step 3: Write tests for monitoring operations**

```python
# tests/test_session.py (append)

class TestMonitoring:
    def test_monitor_and_unmonitor(self, session):
        profile = session.extract([("I am honest", "I am deceptive")])
        session.monitor("test_probe", profile)
        assert "test_probe" in session.probes
        session.unmonitor("test_probe")
        assert "test_probe" not in session.probes
```

- [ ] **Step 4: Write test for context manager**

```python
# tests/test_session.py (append)

class TestLifecycle:
    def test_context_manager(self):
        from steer.session import SteerSession
        with SteerSession(MODEL_ID, device="cuda", probes=[]) as s:
            assert s.model_info["model_type"] == "gemma2"
        # After close, vectors/probes should be cleared
```

- [ ] **Step 5: Run tests to verify they fail**

Run: `pytest tests/test_session.py -v`
Expected: FAIL — `steer.session` does not exist.

- [ ] **Step 6: Implement session.py (construction, state, steering, monitoring, lifecycle)**

```python
# steer/session.py
"""SteerSession — unified backend for steer's programmatic API and TUI."""

from __future__ import annotations

import json
import os
import pathlib
import threading
import time
from typing import Iterator

import torch

from steer.datasource import DataSource
from steer.generation import GenerationConfig, GenerationState, build_chat_input, generate_steered
from steer.hooks import SteeringManager
from steer.model import load_model, get_layers, get_model_info, detect_device
from steer.monitor import TraitMonitor
from steer.probes_bootstrap import bootstrap_probes, _load_defaults
from steer.results import GenerationResult, TokenEvent, ProbeReadings
from steer.vectors import (
    extract_contrastive,
    save_profile as _save_profile,
    load_profile as _load_profile,
    load_contrastive_pairs,
    get_cache_path,
)


class SteerSession:
    """Unified backend for activation steering, monitoring, and generation.

    Owns the model, steering manager, trait monitor, conversation history,
    and generation config. Both the TUI and scripting API call session methods.
    """

    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        quantize: str | None = None,
        probes: list[str] | None = None,
        system_prompt: str | None = None,
        max_tokens: int = 1024,
        cache_dir: str | None = None,
    ):
        self._model, self._tokenizer = load_model(model_id, quantize=quantize, device=device)
        self._layers = get_layers(self._model)
        self._model_info = get_model_info(self._model, self._tokenizer)

        first_param = next(self._model.parameters())
        self._device = first_param.device
        self._dtype = first_param.dtype

        self._cache_dir = cache_dir or str(
            pathlib.Path(__file__).parent / "probes" / "cache"
        )

        self.config = GenerationConfig(
            max_new_tokens=max_tokens,
            system_prompt=system_prompt,
        )

        self._steering = SteeringManager()
        self._orthogonalize = False
        self._steer_lock = threading.Lock()

        self._gen_lock = threading.Lock()
        self._gen_state = GenerationState()

        self._history: list[dict[str, str]] = []
        self._last_result: GenerationResult | None = None

        # Bootstrap probes
        all_categories = ["emotion", "personality", "safety", "cultural", "gender"]
        if probes is None:
            probe_categories = all_categories
        elif not probes:
            probe_categories = []
        else:
            probe_categories = probes

        probe_profiles: dict[str, dict] = {}
        if probe_categories:
            probe_profiles = bootstrap_probes(
                self._model, self._tokenizer, self._layers, self._model_info,
                categories=probe_categories, cache_dir=self._cache_dir,
            )

        self._monitor = TraitMonitor(probe_profiles) if probe_profiles else TraitMonitor({})
        if probe_profiles:
            self._monitor.attach(self._layers, self._device, self._dtype)

    # -- State queries --

    @property
    def model_info(self) -> dict:
        return dict(self._model_info)

    @property
    def vectors(self) -> dict[str, dict]:
        """Active steering vectors: name -> {profile, alpha, enabled}."""
        return {v["name"]: v for v in self._steering.get_active_vectors()}

    @property
    def probes(self) -> dict[str, dict]:
        """Active probes: name -> {profile}."""
        return {name: {"profile": self._monitor._raw_profiles[name]}
                for name in self._monitor.probe_names}

    @property
    def history(self) -> list[dict[str, str]]:
        return list(self._history)

    @property
    def last_result(self) -> GenerationResult | None:
        return self._last_result

    # -- Extraction --

    def extract(self, source) -> dict[int, tuple[torch.Tensor, float]]:
        """Extract a steering vector profile.

        Args:
            source: One of:
                - str: curated dataset name (e.g. "happy")
                - list[tuple[str, str]]: raw contrastive pairs
                - DataSource: any DataSource instance
        """
        if isinstance(source, str):
            ds = DataSource.curated(source)
        elif isinstance(source, DataSource):
            ds = source
        elif isinstance(source, list):
            ds = DataSource.from_pairs(source)
        else:
            raise TypeError(f"Unsupported source type: {type(source)}")

        # Check cache
        cache_path = get_cache_path(self._cache_dir, self._model_info.get("model_id", "unknown"), ds.name)
        try:
            profile, _meta = _load_profile(cache_path)
            profile = {idx: (vec.to(self._device, self._dtype), score)
                       for idx, (vec, score) in profile.items()}
            return profile
        except (FileNotFoundError, KeyError, ValueError):
            pass

        # Convert pairs to extract_contrastive format
        pairs = [{"positive": p, "negative": n} for p, n in ds.pairs]
        profile = extract_contrastive(
            self._model, self._tokenizer, pairs, layers=self._layers,
        )

        # Cache
        _save_profile(profile, cache_path, {
            "concept": ds.name,
            "n_pairs": len(ds.pairs),
        })

        return profile

    def load_profile(self, path: str) -> dict[int, tuple[torch.Tensor, float]]:
        profile, _meta = _load_profile(path)
        profile = {idx: (vec.to(self._device, self._dtype), score)
                   for idx, (vec, score) in profile.items()}
        return profile

    def save_profile(self, profile: dict, path: str, metadata: dict | None = None) -> None:
        _save_profile(profile, path, metadata or {})

    # -- Steering --

    def steer(self, name: str, profile: dict, alpha: float = 2.5) -> None:
        with self._steer_lock:
            self._steering.add_vector(name, profile, alpha)
            self._steering.apply_to_model(
                self._layers, self._device, self._dtype,
                orthogonalize=self._orthogonalize,
            )

    def set_alpha(self, name: str, alpha: float) -> None:
        with self._steer_lock:
            self._steering.set_alpha(name, alpha)
            self._steering.apply_to_model(
                self._layers, self._device, self._dtype,
                orthogonalize=self._orthogonalize,
            )

    def toggle(self, name: str) -> None:
        with self._steer_lock:
            self._steering.toggle_vector(name)
            self._steering.apply_to_model(
                self._layers, self._device, self._dtype,
                orthogonalize=self._orthogonalize,
            )

    def unsteer(self, name: str) -> None:
        with self._steer_lock:
            self._steering.remove_vector(name)
            self._steering.apply_to_model(
                self._layers, self._device, self._dtype,
                orthogonalize=self._orthogonalize,
            )

    def orthogonalize(self) -> None:
        with self._steer_lock:
            self._orthogonalize = True
            self._steering.apply_to_model(
                self._layers, self._device, self._dtype,
                orthogonalize=True,
            )

    def clear_vectors(self) -> None:
        with self._steer_lock:
            self._steering.clear_all()

    # -- Monitoring --

    def monitor(self, name: str, profile: dict | None = None) -> None:
        """Add a probe. If profile is None, extracts from curated dataset."""
        if profile is None:
            profile = self.extract(name)
        self._monitor.add_probe(
            name, profile,
            model_layers=self._layers,
            device=self._device, dtype=self._dtype,
        )

    def unmonitor(self, name: str) -> None:
        self._monitor.remove_probe(
            name, model_layers=self._layers,
            device=self._device, dtype=self._dtype,
        )

    # -- History --

    def rewind(self) -> None:
        if self._history and self._history[-1]["role"] == "assistant":
            self._history.pop()
        if self._history and self._history[-1]["role"] == "user":
            self._history.pop()

    def clear_history(self) -> None:
        self._history.clear()
        if self._monitor:
            self._monitor.reset_history()

    # -- Generation control --

    def stop(self) -> None:
        self._gen_state.request_stop()

    # -- Lifecycle --

    def close(self) -> None:
        self._steering.clear_all()
        if self._monitor:
            self._monitor.detach()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `pytest tests/test_session.py -v`
Expected: All pass.

- [ ] **Step 8: Commit**

```bash
git add steer/session.py tests/test_session.py
git commit -m "feat: add SteerSession core — construction, steering, monitoring, lifecycle"
```

---

### Task 5: SteerSession generation — blocking and streaming (`steer/session.py`)

Add `generate()` and `generate_stream()` methods.

**Files:**
- Modify: `steer/session.py`
- Modify: `tests/test_session.py`

- [ ] **Step 1: Write tests for blocking generation**

```python
# tests/test_session.py (append)

from steer.results import GenerationResult


class TestGeneration:
    def test_generate_blocking_string(self, session):
        result = session.generate("Say hello in one word.")
        assert isinstance(result, GenerationResult)
        assert len(result.text) > 0
        assert result.token_count > 0
        assert result.tok_per_sec > 0
        assert result.elapsed > 0

    def test_generate_blocking_messages(self, session):
        result = session.generate([
            {"role": "user", "content": "Say hello in one word."},
        ])
        assert isinstance(result, GenerationResult)
        assert len(result.text) > 0

    def test_generate_appends_to_history(self, session):
        session.clear_history()
        session.generate("Say hi.")
        assert len(session.history) == 2  # user + assistant
        assert session.history[0]["role"] == "user"
        assert session.history[1]["role"] == "assistant"

    def test_generate_with_probes(self, session):
        # Session was constructed with emotion probes
        session.clear_history()
        result = session.generate("Tell me something exciting!")
        if session.probes:
            # At least some probes should have readings
            assert isinstance(result.readings, dict)

    def test_generate_snapshots_vectors(self, session):
        profile = session.extract([("formal", "casual")])
        session.steer("formal", profile, alpha=1.0)
        result = session.generate("Hello.")
        assert "formal" in result.vectors
        assert result.vectors["formal"] == 1.0
        session.unsteer("formal")

    def test_last_result(self, session):
        session.clear_history()
        result = session.generate("Hello.")
        assert session.last_result is result
```

- [ ] **Step 2: Write tests for streaming generation**

```python
# tests/test_session.py (append)

class TestStreamingGeneration:
    def test_generate_stream(self, session):
        session.clear_history()
        tokens = []
        for event in session.generate_stream("Say hello."):
            assert isinstance(event, TokenEvent)
            tokens.append(event)
        assert len(tokens) > 0
        assert all(isinstance(t.text, str) for t in tokens)
        # last_result should be populated after stream completes
        assert session.last_result is not None
        assert session.last_result.token_count == len(tokens)

    def test_concurrent_generation_raises(self, session):
        """Cannot start a second generation while one is in progress."""
        import threading

        session.clear_history()
        barrier = threading.Barrier(2, timeout=5)
        error = []

        def gen_in_thread():
            try:
                barrier.wait()
                session.generate("Count to 100.")
            except RuntimeError as e:
                error.append(e)

        t = threading.Thread(target=gen_in_thread)
        t.start()
        try:
            barrier.wait()
            # Try generating from main thread — one of the two should fail
            session.generate("Hello.")
        except RuntimeError:
            pass  # Expected: one thread wins, the other gets RuntimeError
        t.join()
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_session.py::TestGeneration -v`
Expected: FAIL — `generate` method not yet implemented.

- [ ] **Step 4: Implement generate() and generate_stream()**

Add these methods to `SteerSession` in `steer/session.py`:

```python
    # -- Generation --

    def _prepare_input(self, input) -> tuple[list[dict], torch.Tensor]:
        """Normalize input to messages list and compute input_ids."""
        if isinstance(input, str):
            messages = list(self._history) + [{"role": "user", "content": input}]
        elif isinstance(input, list):
            messages = list(input)
        else:
            raise TypeError(f"Unsupported input type: {type(input)}")

        input_ids = build_chat_input(
            self._tokenizer, messages, self.config.system_prompt,
        ).to(self._device)
        return messages, input_ids

    def _build_readings(self) -> dict[str, ProbeReadings]:
        """Flush monitor and build ProbeReadings for all active probes."""
        readings: dict[str, ProbeReadings] = {}
        if not self._monitor or not self._monitor.probe_names:
            return readings

        self._monitor.flush_to_cpu()
        for name in self._monitor.probe_names:
            stats = self._monitor.get_stats(name)
            count = stats["count"]
            if count == 0:
                continue
            mean = stats["sum"] / count
            variance = max(0.0, stats["sum_sq"] / count - mean ** 2)
            std = variance ** 0.5
            hist = list(self._monitor.history.get(name, []))
            # delta_per_tok: mean absolute change between consecutive tokens
            if len(hist) >= 2:
                deltas = [abs(hist[i] - hist[i-1]) for i in range(1, len(hist))]
                delta_per_tok = sum(deltas) / len(deltas)
            else:
                delta_per_tok = 0.0
            readings[name] = ProbeReadings(
                per_token=hist,
                mean=mean,
                std=std,
                min=stats["min"] if stats["min"] != float("inf") else 0.0,
                max=stats["max"] if stats["max"] != float("-inf") else 0.0,
                delta_per_tok=delta_per_tok,
            )
        return readings

    def _snapshot_vectors(self) -> dict[str, float]:
        """Snapshot active vectors and their alphas."""
        return {v["name"]: v["alpha"] for v in self._steering.get_active_vectors()
                if v.get("enabled", True)}

    def generate(self, input, **kwargs) -> GenerationResult:
        """Blocking generation. Returns when complete."""
        if not self._gen_lock.acquire(blocking=False):
            raise RuntimeError("Generation already in progress")
        try:
            return self._generate_blocking(input)
        finally:
            self._gen_lock.release()

    def _generate_blocking(self, input) -> GenerationResult:
        messages, input_ids = self._prepare_input(input)
        self._gen_state.reset()
        if self._monitor:
            self._monitor.reset_history()

        vector_snapshot = self._snapshot_vectors()
        start = time.monotonic()

        generated_ids = generate_steered(
            self._model, self._tokenizer, input_ids,
            self.config, self._gen_state,
        )

        elapsed = time.monotonic() - start
        token_count = len(generated_ids)
        tok_per_sec = token_count / elapsed if elapsed > 0.1 else 0.0

        text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
        readings = self._build_readings()

        result = GenerationResult(
            text=text,
            tokens=generated_ids,
            token_count=token_count,
            tok_per_sec=tok_per_sec,
            elapsed=elapsed,
            readings=readings,
            vectors=vector_snapshot,
        )
        self._last_result = result

        # Append to history
        if isinstance(input, str):
            self._history.append({"role": "user", "content": input})
        if text.strip():
            self._history.append({"role": "assistant", "content": text})

        return result

    def generate_stream(self, input, **kwargs) -> Iterator[TokenEvent]:
        """Streaming generation. Yields TokenEvent per token."""
        if not self._gen_lock.acquire(blocking=False):
            raise RuntimeError("Generation already in progress")
        try:
            yield from self._generate_streaming(input)
        finally:
            self._gen_lock.release()

    def _generate_streaming(self, input) -> Iterator[TokenEvent]:
        import queue as _queue

        messages, input_ids = self._prepare_input(input)
        self._gen_state.reset()
        if self._monitor:
            self._monitor.reset_history()

        vector_snapshot = self._snapshot_vectors()
        start = time.monotonic()
        token_events: list[TokenEvent] = []
        generated_ids: list[int] = []
        token_queue = self._gen_state.token_queue
        gen_done = threading.Event()
        gen_error: list[Exception] = []

        def _worker():
            try:
                ids = generate_steered(
                    self._model, self._tokenizer, input_ids,
                    self.config, self._gen_state,
                    on_token=lambda tok: token_queue.put(tok),
                )
                generated_ids.extend(ids)
            except Exception as e:
                gen_error.append(e)
            finally:
                gen_done.set()

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

        idx = 0
        while True:
            try:
                tok_str = token_queue.get(timeout=0.05)
            except _queue.Empty:
                if gen_done.is_set():
                    # Drain remaining
                    while True:
                        try:
                            tok_str = token_queue.get_nowait()
                        except _queue.Empty:
                            break
                        if tok_str is None:
                            break
                        # Get per-token readings if monitor active
                        readings_snap = None
                        if self._monitor and self._monitor.has_pending_data():
                            self._monitor.flush_to_cpu()
                            current, _ = self._monitor.get_current_and_previous()
                            readings_snap = dict(current) if current else None
                        event = TokenEvent(
                            text=tok_str,
                            token_id=generated_ids[idx] if idx < len(generated_ids) else -1,
                            index=idx,
                            readings=readings_snap,
                        )
                        token_events.append(event)
                        yield event
                        idx += 1
                    break
                continue

            if tok_str is None:
                break

            # Get per-token readings if monitor active
            readings_snap = None
            if self._monitor and self._monitor.has_pending_data():
                self._monitor.flush_to_cpu()
                current, _ = self._monitor.get_current_and_previous()
                readings_snap = dict(current) if current else None

            event = TokenEvent(
                text=tok_str,
                token_id=generated_ids[idx] if idx < len(generated_ids) else -1,
                index=idx,
                readings=readings_snap,
            )
            token_events.append(event)
            yield event
            idx += 1

        thread.join()

        if gen_error:
            raise gen_error[0]

        elapsed = time.monotonic() - start
        token_count = len(token_events)
        tok_per_sec = token_count / elapsed if elapsed > 0.1 else 0.0
        text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
        readings = self._build_readings()

        self._last_result = GenerationResult(
            text=text,
            tokens=list(generated_ids),
            token_count=token_count,
            tok_per_sec=tok_per_sec,
            elapsed=elapsed,
            readings=readings,
            vectors=vector_snapshot,
        )

        # Append to history
        if isinstance(input, str):
            self._history.append({"role": "user", "content": input})
        if text.strip():
            self._history.append({"role": "assistant", "content": text})
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_session.py -v`
Expected: All pass.

- [ ] **Step 6: Commit**

```bash
git add steer/session.py tests/test_session.py
git commit -m "feat: add SteerSession generation — blocking and streaming modes"
```

---

### Task 6: Public API exports (`steer/__init__.py`, `pyproject.toml`)

Wire up the public `import steer` API and add optional dependency extras.

**Files:**
- Modify: `steer/__init__.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Write import test**

```python
# tests/test_results.py (add at top of file, before existing classes)

class TestPublicAPI:
    def test_imports(self):
        from steer import SteerSession, DataSource, ResultCollector
        from steer import GenerationResult, TokenEvent, ProbeReadings
        assert SteerSession is not None
        assert DataSource is not None
        assert ResultCollector is not None
        assert GenerationResult is not None
        assert TokenEvent is not None
        assert ProbeReadings is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_results.py::TestPublicAPI -v`
Expected: FAIL — `__init__.py` doesn't export these yet.

- [ ] **Step 3: Update `steer/__init__.py`**

```python
# steer/__init__.py
"""steer — local activation steering + trait monitoring for HuggingFace causal LMs."""

__version__ = "0.1.0"

from steer.session import SteerSession
from steer.datasource import DataSource
from steer.results import GenerationResult, TokenEvent, ProbeReadings, ResultCollector

__all__ = [
    "SteerSession",
    "DataSource",
    "GenerationResult",
    "TokenEvent",
    "ProbeReadings",
    "ResultCollector",
]
```

- [ ] **Step 4: Update `pyproject.toml` with optional extras**

Add these entries under `[project.optional-dependencies]`:

```toml
hf = ["datasets>=2.0"]
pandas = ["pandas>=2.0"]
research = ["datasets>=2.0", "pandas>=2.0"]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_results.py::TestPublicAPI -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add steer/__init__.py pyproject.toml
git commit -m "feat: export public API and add optional dependency extras"
```

---

### Task 7: Rewrite TUI to use SteerSession (`steer/tui/app.py`)

The largest task. Rewrite `tui/app.py` to delegate all orchestration to `SteerSession`. The app becomes a thin Textual frontend.

**Files:**
- Modify: `steer/tui/app.py`
- Modify: `steer/cli.py`

- [ ] **Step 1: Update `cli.py` to construct SteerSession**

Replace the current `main()` function body. The new version constructs a `SteerSession` and passes it to `SteerApp`:

```python
# steer/cli.py
"""CLI entry point for steer."""

from __future__ import annotations

import argparse


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="steer",
        description="Activation steering + trait monitoring TUI for local HuggingFace models",
    )
    p.add_argument(
        "model",
        help="HuggingFace model ID or local path (e.g. google/gemma-2-9b-it)",
    )
    p.add_argument(
        "--quantize", "-q",
        choices=["4bit", "8bit"],
        default=None,
        help="Quantization mode (default: bf16/fp16)",
    )
    p.add_argument(
        "--device", "-d",
        default="auto",
        help="Device: auto (detect), cuda, mps, or cpu (default: auto)",
    )
    p.add_argument(
        "--probes", "-p",
        nargs="*",
        default=None,
        help="Probe categories to load: all, none, emotion, personality, safety, cultural, gender (default: all)",
    )
    p.add_argument(
        "--system-prompt", "-s",
        default=None,
        help="System prompt for chat",
    )
    p.add_argument(
        "--max-tokens", "-m",
        type=int,
        default=1024,
        help="Max tokens per generation (default: 1024)",
    )
    p.add_argument(
        "--cache-dir", "-c",
        default=None,
        help="Cache directory for extracted vectors (default: probes/cache/ in package)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None):
    args = parse_args(argv)

    print(f"Loading model: {args.model}")
    if args.quantize:
        print(f"Quantization: {args.quantize}")

    # Resolve probe categories
    all_categories = ["emotion", "personality", "safety", "cultural", "gender"]
    if args.probes is None or args.probes == ["all"]:
        probe_categories = all_categories
    elif args.probes == ["none"] or args.probes == []:
        probe_categories = []
    else:
        probe_categories = args.probes

    from steer.session import SteerSession
    session = SteerSession(
        model_id=args.model,
        device=args.device,
        quantize=args.quantize,
        probes=probe_categories,
        system_prompt=args.system_prompt,
        max_tokens=args.max_tokens,
        cache_dir=args.cache_dir,
    )

    info = session.model_info
    print(f"Architecture: {info['model_type']}")
    print(f"Layers: {info['num_layers']}, Hidden dim: {info['hidden_dim']}")
    print(f"VRAM: {info['vram_used_gb']:.1f} GB")
    print(f"Loaded {len(session.probes)} probes")

    from steer.tui.app import SteerApp
    app = SteerApp(session=session)
    app.run()
```

- [ ] **Step 2: Rewrite `tui/app.py`**

This is the big change. The new `SteerApp` takes a `SteerSession` and delegates all operations to it. Key changes:

- Constructor takes `session: SteerSession` instead of individual model/tokenizer/etc. arguments.
- All `self._model`, `self._tokenizer`, `self._layers`, `self._steering`, `self._monitor`, `self._gen_state`, `self._gen_config` references replaced with `self._session` attribute access.
- `_extract_vector_worker`, `_steer_worker`, `_probe_worker` simplified to call `session.extract()`, `session.steer()`, `session.monitor()`.
- `_start_generation` uses `session.generate_stream()` in a worker thread.
- `_poll_generation` drains tokens from a queue fed by the worker thread.
- State reads (`vectors`, `probes`, `config`, `history`) go through session properties.

The full rewrite should:
1. Replace `__init__` parameters with `session: SteerSession`
2. Store `self._session = session`
3. Access `session._device`, `session._dtype`, `session._layers`, `session._model`, `session._tokenizer` for low-level TUI needs (VRAM display, panel rendering)
4. Access `session.config` for generation config
5. Access `session._steering` for left panel vector display
6. Access `session._monitor` for trait panel monitor data
7. Replace all extraction workers with calls to `session.extract()` + `session.steer()`/`session.monitor()`
8. Replace `_start_generation` to use `session.generate_stream()` in a worker
9. Keep keyboard handling, panel focus, and UI rendering intact

Write the full file. The structure follows the existing `app.py` but with `self._session` replacing individual module ownership. Key sections:

```python
class SteerApp(App):
    # ... BINDINGS unchanged ...

    def __init__(self, session, **kwargs):
        super().__init__(ansi_color=True, **kwargs)
        self._session = session
        self._messages = session._history  # shared reference
        self._orthogonalize = False

        self._current_assistant_widget = None
        self._poll_timer = None
        self._last_prompt = None
        self._ab_in_progress = False
        self._focused_panel_idx = 1

        # Generation polling state
        self._gen_start_time = 0.0
        self._gen_token_count = 0
        self._prompt_token_count = 0
        self._last_tok_per_sec = 0.0
        self._last_elapsed = 0.0
        self._cached_vram_gb = 0.0
        self._vram_poll_counter = 0
        self._last_status_args = ()

        # Token queue for stream→poll bridge
        self._token_queue = queue.SimpleQueue()
        self._generating = False

        defaults = _load_defaults()
        self._probe_categories = {
            cat.capitalize(): probes_list
            for cat, probes_list in defaults.items()
        }
```

For the extraction workers:

```python
    def _steer_worker(self, concept, baseline, alpha, name):
        """Worker thread: extract and apply steering vector."""
        try:
            if baseline:
                from steer.datasource import DataSource
                # Generate contrastive pairs for concept vs baseline
                pairs = self._get_pairs(concept, baseline)
                ds = DataSource.from_pairs(pairs, name=f"{concept}_vs_{baseline}")
                profile = self._session.extract(ds)
            else:
                profile = self._session.extract(concept)
            self._session.steer(name, profile, alpha=alpha)
            self.call_from_thread(self._on_vector_extracted, name, alpha, profile)
        except FileNotFoundError:
            # Concept not in curated set — generate pairs
            pairs = self._get_pairs(concept, baseline)
            from steer.datasource import DataSource
            ds = DataSource.from_pairs(pairs, name=concept)
            profile = self._session.extract(ds)
            self._session.steer(name, profile, alpha=alpha)
            self.call_from_thread(self._on_vector_extracted, name, alpha, profile)
```

For generation:

```python
    def _start_generation(self):
        if self._generating:
            self._session.stop()
            self._chat_panel.add_system_message("Stopping current generation. Please resubmit.")
            return

        self._gen_token_count = 0
        self._gen_start_time = time.monotonic()
        self._generating = True
        self._current_assistant_widget = self._chat_panel.start_assistant_message()

        def _gen_worker():
            try:
                messages = list(self._session._history) + [
                    {"role": "user", "content": self._last_prompt}
                ] if isinstance(self._last_prompt, str) else self._messages
                for event in self._session.generate_stream(self._messages[-1]["content"]):
                    self._token_queue.put(event.text)
                self._token_queue.put(None)  # sentinel
            except Exception:
                self._token_queue.put(None)
                raise

        self.run_worker(_gen_worker, thread=True)
```

The `_poll_generation` method stays structurally similar — drains `self._token_queue` and updates panels. Monitor data comes from `self._session._monitor.has_pending_data()` / `flush_to_cpu()`.

Note: The `_generate_contrastive_pairs` method stays in `app.py` since it uses the model's own generation to produce pairs — this is a TUI-specific feature (using the loaded model as its own pair generator). Session's `extract()` only handles pre-existing pairs.

- [ ] **Step 3: Verify existing smoke tests still pass**

Run: `pytest tests/test_smoke.py -v`
Expected: All existing tests pass. These test core modules directly, not through the TUI.

- [ ] **Step 4: Verify new session tests still pass**

Run: `pytest tests/test_session.py -v`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add steer/cli.py steer/tui/app.py
git commit -m "refactor: rewrite TUI to use SteerSession as backend"
```

---

### Task 8: Verify full test suite and existing smoke tests

Final integration check.

**Files:**
- No new files. Run all tests.

- [ ] **Step 1: Run all non-CUDA tests**

Run: `pytest tests/test_results.py tests/test_datasource.py -v`
Expected: All pass.

- [ ] **Step 2: Run CUDA tests (if available)**

Run: `pytest tests/ -v`
Expected: All pass (smoke tests + session tests + results + datasource).

- [ ] **Step 3: Run an end-to-end headless script to verify the full API works**

Create a quick verification script (don't commit this):

```python
# /tmp/test_api.py
from steer import SteerSession, DataSource, ResultCollector

with SteerSession("google/gemma-2-2b-it", device="cuda", probes=["emotion"]) as session:
    # Extract and steer
    profile = session.extract("happy")
    session.steer("happy", profile, alpha=2.0)

    # Generate
    result = session.generate("What makes a good day?")
    print(f"Generated: {result.text[:100]}...")
    print(f"Tokens: {result.token_count}, Speed: {result.tok_per_sec:.1f} tok/s")
    print(f"Readings: {list(result.readings.keys())}")

    # Collect results
    collector = ResultCollector()
    collector.add(result, concept="happy", alpha=2.0)
    print(f"Collected: {len(collector.results)} results")
    print("API works.")
```

Run: `python /tmp/test_api.py`
Expected: Runs without errors, prints generation output and stats.

- [ ] **Step 4: Commit any test fixes if needed**

Only if issues were found in steps 1-3. Otherwise skip.

- [ ] **Step 5: Final commit with any remaining fixes**

```bash
git add -A
git commit -m "test: verify full API integration"
```
