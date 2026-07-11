"""Tests for the SAE extraction pipeline."""
from __future__ import annotations

from typing import Any

import pytest


def test_errors_subclass_saklas_error():
    from saklas.core.errors import (
        SaklasError,
        SaeBackendImportError,
        SaeReleaseNotFoundError,
        SaeModelMismatchError,
        SaeCoverageError,
        AmbiguousVariantError,
        UnknownVariantError,
    )
    for cls in (
        SaeBackendImportError,
        SaeReleaseNotFoundError,
        SaeModelMismatchError,
        SaeCoverageError,
        AmbiguousVariantError,
        UnknownVariantError,
    ):
        assert issubclass(cls, SaklasError)


def test_errors_preserve_stdlib_mro():
    from saklas.core.errors import (
        SaeBackendImportError,
        SaeReleaseNotFoundError,
        SaeModelMismatchError,
        SaeCoverageError,
        AmbiguousVariantError,
        UnknownVariantError,
    )
    assert issubclass(SaeBackendImportError, ImportError)
    assert issubclass(SaeReleaseNotFoundError, ValueError)
    assert issubclass(SaeModelMismatchError, ValueError)
    assert issubclass(SaeCoverageError, ValueError)
    assert issubclass(AmbiguousVariantError, ValueError)
    assert issubclass(UnknownVariantError, KeyError)


def test_sae_backend_protocol_shape():
    from saklas.core.sae import SaeBackend, MockSaeBackend
    # Structural conformance — downstream code type-hints `SaeBackend | None`
    # and we want the mock to pass isinstance checks at runtime.
    assert hasattr(SaeBackend, "encode_layer")
    assert hasattr(SaeBackend, "decode_layer")
    mock = MockSaeBackend(layers=frozenset({0}), d_model=4)
    assert isinstance(mock, SaeBackend)


def test_mock_sae_backend_roundtrip():
    """Identity mock: encode and decode are both identity, d_feature == d_model.

    Used throughout extract/session tests to exercise the SAE branch without
    needing sae_lens or real SAE weights.
    """
    import torch
    from saklas.core.sae import MockSaeBackend

    backend = MockSaeBackend(
        layers=frozenset({4, 8, 12}),
        d_model=16,
        release="mock-release",
    )
    assert backend.layers == frozenset({4, 8, 12})
    assert backend.release == "mock-release"

    h = torch.randn(5, 16)
    f = backend.encode_layer(8, h)
    assert f.shape == (5, 16)
    assert torch.allclose(f, h)

    v_feat = torch.randn(16)
    v_model = backend.decode_layer(8, v_feat)
    assert v_model.shape == (16,)
    assert torch.allclose(v_model, v_feat)


def test_mock_sae_backend_custom_encode_decode():
    """MockSaeBackend lets tests inject per-layer transforms for non-identity cases."""
    import torch
    from saklas.core.sae import MockSaeBackend

    backend = MockSaeBackend(
        layers=frozenset({3}),
        d_model=4,
        d_feature=4,
        encode_fn=lambda idx, h: h * 2.0,
        decode_fn=lambda idx, f: f * 0.5,
    )
    h = torch.ones(2, 4)
    f = backend.encode_layer(3, h)
    assert torch.allclose(f, torch.full((2, 4), 2.0))
    v = backend.decode_layer(3, torch.full((4,), 4.0))
    assert torch.allclose(v, torch.full((4,), 2.0))


def test_mock_sae_backend_passes_layer_idx_to_overrides():
    """Per-layer fns receive the layer index so tests can verify dispatch."""
    import torch
    from saklas.core.sae import MockSaeBackend

    seen: list[int] = []
    backend = MockSaeBackend(
        layers=frozenset({2, 5}),
        d_model=4,
        encode_fn=lambda idx, h: (seen.append(idx) or h),
        decode_fn=lambda idx, f: (seen.append(-idx) or f),
    )
    backend.encode_layer(2, torch.zeros(1, 4))
    backend.decode_layer(5, torch.zeros(4))
    assert seen == [2, -5]


def test_sae_lens_backend_encodes_and_decodes(monkeypatch: pytest.MonkeyPatch):
    """SaeLensBackend wraps per-layer SAE modules and dispatches by layer index."""
    import torch
    import sys
    import types

    fake_sae_lens = types.ModuleType("sae_lens")
    seen_revisions: list[str | None] = []

    class FakeSAE:
        def __init__(self, d_in: Any, d_sae: Any, hook_layer: Any) -> None:
            self.cfg = types.SimpleNamespace(
                d_in=d_in, d_sae=d_sae, model_name="test-model", hook_layer=hook_layer,
            )
            self.W_enc = torch.eye(d_in)[:, :d_sae] if d_in >= d_sae else torch.zeros(d_in, d_sae)
            self.W_dec = self.W_enc.T
            self.b_enc = torch.zeros(d_sae)

        def encode(self, x: Any) -> Any:
            return x @ self.W_enc + self.b_enc

        def decode(self, f: Any) -> Any:
            return f @ self.W_dec

        @classmethod
        def from_pretrained(
            cls, release: Any, sae_id: Any, device: Any = None,
            revision: str | None = None,
        ) -> Any:
            seen_revisions.append(revision)
            hook_layer = int(sae_id.split("_")[1])
            return (
                cls(d_in=4, d_sae=4, hook_layer=hook_layer),
                {"d_in": 4, "d_sae": 4, "hook_layer": hook_layer},
                None,
            )

    fake_sae_lens.SAE = FakeSAE  # pyright: ignore[reportAttributeAccessIssue]  # types.ModuleType stub has no dynamic attrs
    fake_sae_lens.get_pretrained_saes_directory = lambda: {  # pyright: ignore[reportAttributeAccessIssue]  # types.ModuleType stub has no dynamic attrs
        "mock-canonical": {
            "saes_map": {f"layer_{i}": i for i in (2, 5, 8)},
            "model": "test-model",
        }
    }
    monkeypatch.setitem(sys.modules, "sae_lens", fake_sae_lens)

    from saklas.core.sae import load_sae_backend
    backend = load_sae_backend("mock-canonical", model_id="test-model", device="cpu")
    assert backend.layers == frozenset({2, 5, 8})
    assert backend.release == "mock-canonical"
    # Registry resolution is cheap; weights stay cold until a covered layer is
    # actually visited, and only the current layer remains resident.
    assert backend._active_sae is None

    h = torch.randn(3, 4)
    f = backend.encode_layer(5, h)
    assert f.shape == (3, 4)
    sae5 = backend._active_sae
    v = backend.decode_layer(5, torch.randn(4))
    assert v.shape == (4,)
    assert backend._active_sae is sae5
    backend.encode_layer(8, h)
    assert backend._active_layer == 8
    assert backend._active_sae is not sae5
    pinned = load_sae_backend(
        "mock-canonical", revision="commit-123",
        model_id="test-model", device="cpu",
    )
    pinned.encode_layer(2, h)
    assert seen_revisions[-1] == "commit-123"


def test_installed_sae_lens_registry_api_resolves_without_loading_weights() -> None:
    pytest.importorskip("sae_lens")
    from saklas.core.sae import load_sae_backend

    backend = load_sae_backend(
        "gemma-scope-2-4b-it-res",
        model_id="google/gemma-3-4b-it", device="cpu",
    )
    assert backend.layers == frozenset({9, 17, 22, 29})
    assert backend._active_sae is None


def test_release_discovery_omits_known_non_residual_families(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys
    import types

    fake = types.ModuleType("sae_lens")
    fake.get_pretrained_saes_directory = lambda: {  # pyright: ignore[reportAttributeAccessIssue]
        "scope-res": {"saes_map": {"layer_1": 1}, "model": "m"},
        "scope-att": {"saes_map": {"layer_1": 1}, "model": "m"},
        "scope-mlp": {"saes_map": {"layer_1": 1}, "model": "m"},
        "scope-transcoders": {"saes_map": {"layer_1": 1}, "model": "m"},
        "custom": {"saes_map": {"layer_1": 1}, "model": "m"},
    }
    monkeypatch.setitem(sys.modules, "sae_lens", fake)

    from saklas.core.sae import list_sae_releases

    assert [row["release"] for row in list_sae_releases("m")] == [
        "custom", "scope-res",
    ]


def test_loaded_attention_hook_is_rejected_before_use() -> None:
    import types
    from saklas.core.errors import SaeCoverageError
    from saklas.core.sae import _validate_residual_hook

    sae = types.SimpleNamespace(cfg=types.SimpleNamespace(
        metadata={"hook_name": "blocks.22.attn.hook_z"},
    ))
    with pytest.raises(SaeCoverageError, match="choose the corresponding '-res'"):
        _validate_residual_hook(sae, "scope-att", 22)


def test_residual_width_must_match_model_hidden_size() -> None:
    from saklas.core.errors import SaeCoverageError
    from saklas.core.sae import MockSaeBackend, validate_residual_width

    backend = MockSaeBackend(layers=frozenset({2}), d_model=3)
    with pytest.raises(SaeCoverageError, match="model residual stream has width 4"):
        validate_residual_width(backend, 2, 4)


def test_sae_lens_backend_missing_dep_raises(monkeypatch: pytest.MonkeyPatch):
    """When sae_lens isn't installed, load_sae_backend raises SaeBackendImportError."""
    import sys
    monkeypatch.setitem(sys.modules, "sae_lens", None)
    from saklas.core.sae import load_sae_backend
    from saklas.core.errors import SaeBackendImportError
    with pytest.raises(SaeBackendImportError):
        load_sae_backend("any", model_id="m", device="cpu")


def test_sae_lens_backend_release_not_found(monkeypatch: pytest.MonkeyPatch):
    import sys
    import types

    fake = types.ModuleType("sae_lens")
    fake.get_pretrained_saes_directory = lambda: {  # pyright: ignore[reportAttributeAccessIssue]  # types.ModuleType stub has no dynamic attrs
        "mock-a": {"saes_map": {}, "model": "m"},
        "mock-b": {"saes_map": {}, "model": "m"},
    }
    fake.SAE = object  # pyright: ignore[reportAttributeAccessIssue]  # types.ModuleType stub has no dynamic attrs
    monkeypatch.setitem(sys.modules, "sae_lens", fake)

    from saklas.core.sae import load_sae_backend
    from saklas.core.errors import SaeReleaseNotFoundError
    with pytest.raises(SaeReleaseNotFoundError) as exc:
        load_sae_backend("nonexistent", model_id="m", device="cpu")
    msg = str(exc.value)
    # Message should list near matches so user knows what's available.
    assert "mock-a" in msg or "mock-b" in msg


def test_sae_lens_backend_model_mismatch(monkeypatch: pytest.MonkeyPatch):
    import sys
    import types

    fake = types.ModuleType("sae_lens")

    class FakeSAE:
        def __init__(self) -> None:
            self.cfg = types.SimpleNamespace(model_name="other-model", hook_layer=0)

        @classmethod
        def from_pretrained(cls, release: Any, sae_id: Any, device: Any = None) -> Any:
            return cls(), {"hook_layer": 0}, None

    fake.SAE = FakeSAE  # pyright: ignore[reportAttributeAccessIssue]  # types.ModuleType stub has no dynamic attrs
    fake.get_pretrained_saes_directory = lambda: {  # pyright: ignore[reportAttributeAccessIssue]  # types.ModuleType stub has no dynamic attrs
        "mock": {"saes_map": {"layer_0": 0}, "model": "other-model"},
    }
    monkeypatch.setitem(sys.modules, "sae_lens", fake)

    from saklas.core.sae import load_sae_backend
    from saklas.core.errors import SaeModelMismatchError
    with pytest.raises(SaeModelMismatchError):
        load_sae_backend("mock", model_id="my-model", device="cpu")


def test_sae_lens_backend_canonical_layer_map_warns_on_multiple(monkeypatch: pytest.MonkeyPatch, recwarn: Any):
    """When a release has multiple SAEs per layer, pick narrowest + warn."""
    import sys
    import types

    fake = types.ModuleType("sae_lens")

    class FakeSAE:
        def __init__(self) -> None:
            self.cfg = types.SimpleNamespace(model_name="test-model", hook_layer=0)

        @classmethod
        def from_pretrained(cls, release: Any, sae_id: Any, device: Any = None) -> Any:
            # Parse `layer_N` prefix out of canonical sae_id strings like
            # `layer_0/width_16k/l0_100`.
            import re
            m = re.search(r"layer[_-]?(\d+)", sae_id)
            layer = int(m.group(1)) if m else 0
            sae = cls()
            sae.cfg.hook_layer = layer
            return sae, {"hook_layer": layer}, None

    fake.SAE = FakeSAE  # pyright: ignore[reportAttributeAccessIssue]  # types.ModuleType stub has no dynamic attrs
    fake.get_pretrained_saes_directory = lambda: {  # pyright: ignore[reportAttributeAccessIssue]  # types.ModuleType stub has no dynamic attrs
        "mock": {
            "saes_map": {
                "layer_0/width_16k/l0_100": 0,
                "layer_0/width_65k/l0_500": 0,
                "layer_2/width_16k/l0_100": 2,
            },
            "model": "test-model",
        },
    }
    monkeypatch.setitem(sys.modules, "sae_lens", fake)

    from saklas.core.sae import load_sae_backend
    backend = load_sae_backend("mock", model_id="test-model", device="cpu")
    # Warning emitted because layer 0 has two candidates.
    warnings_about_multiple = [w for w in recwarn.list if "multiple SAEs" in str(w.message)]
    assert len(warnings_about_multiple) >= 1
    # Layers 0 and 2 are both represented.
    assert backend.layers == frozenset({0, 2})


def test_canonical_layer_map_sorts_width_and_l0_numerically() -> None:
    from saklas.core.sae import _canonical_layer_map

    with pytest.warns(UserWarning, match="multiple SAEs"):
        chosen = _canonical_layer_map({
            "layer_0_width_131k_l0_small": 0,
            "layer_0_width_16k_l0_big": 0,
            "layer_0_width_16k_l0_medium": 0,
            "layer_0_width_16k_l0_small": 0,
            "layer_1/width_16k/average_l0_105": 1,
            "layer_1/width_16k/average_l0_13": 1,
        })

    assert chosen == {
        "layer_0_width_16k_l0_small": 0,
        "layer_1/width_16k/average_l0_13": 1,
    }


# --- DLS tests (raw PCA path; co-located with the SAE extract tests above
# because they share the ``_encode_and_capture_all`` mock infrastructure).
# Replaces the v2.0–v2.1 ``drop_edges`` test family — edge-drop is gone in
# v2.1, layer selection is now data-driven via :func:`compute_dls_axes`.
