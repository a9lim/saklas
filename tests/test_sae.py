"""Tests for the SAE extraction pipeline."""
from __future__ import annotations

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


def test_extract_contrastive_sae_subset_layers(monkeypatch):
    """With sae=MockSaeBackend(layers={1,3}), profile covers only those layers."""
    import torch
    from saklas.core import vectors as V
    from saklas.core.sae import MockSaeBackend

    def fake_encode_and_capture(model, tokenizer, text, layers, device):
        torch.manual_seed(hash(text) & 0xFFFF)
        out = {}
        for idx in range(len(layers)):
            base = torch.randn(8)
            sign = 1.0 if "pos" in text else -1.0
            out[idx] = base + sign * (idx + 1) * 0.3
        return out

    monkeypatch.setattr(V, "_encode_and_capture_all", fake_encode_and_capture)

    pairs = [{"positive": f"pos_{i}", "negative": f"neg_{i}"} for i in range(5)]

    class FakeModel:
        def parameters(self):
            yield torch.zeros(1)
    class FakeTok:
        pass

    layers_list = [object()] * 4
    sae = MockSaeBackend(layers=frozenset({1, 3}), d_model=8)

    profile = V.extract_contrastive(
        FakeModel(), FakeTok(), pairs, layers=layers_list,
        device=torch.device("cpu"),
        sae=sae,
    )
    assert set(profile.keys()) == {1, 3}
    mags = [profile[i].norm().item() for i in profile]
    assert all(m > 0 for m in mags)


def test_extract_contrastive_sae_pca_center_orients_correctly(monkeypatch):
    """pos > neg on the resulting direction, majority-vote orientation."""
    import torch
    from saklas.core import vectors as V
    from saklas.core.sae import MockSaeBackend

    torch.manual_seed(0)

    def fake_encode_and_capture(model, tokenizer, text, layers, device):
        out = {}
        for idx in range(len(layers)):
            base = torch.zeros(4)
            base[0] = 1.0 if "pos" in text else -1.0
            out[idx] = base + 0.01 * torch.randn(4)
        return out

    monkeypatch.setattr(V, "_encode_and_capture_all", fake_encode_and_capture)

    pairs = [{"positive": f"pos_{i}", "negative": f"neg_{i}"} for i in range(5)]
    class FakeModel:
        def parameters(self):
            yield torch.zeros(1)
    class FakeTok:
        pass

    layers_list = [object()] * 2
    sae = MockSaeBackend(layers=frozenset({0, 1}), d_model=4)

    profile = V.extract_contrastive(
        FakeModel(), FakeTok(), pairs, layers=layers_list,
        device=torch.device("cpu"),
        sae=sae,
    )
    for idx, vec in profile.items():
        assert vec[0].item() > 0


def test_extract_contrastive_sae_zero_coverage_raises(monkeypatch):
    """An SAE covering no model layers raises SaeCoverageError."""
    import torch
    from saklas.core import vectors as V
    from saklas.core.errors import SaeCoverageError
    from saklas.core.sae import MockSaeBackend

    monkeypatch.setattr(
        V, "_encode_and_capture_all",
        lambda *a, **k: {i: torch.zeros(4) for i in range(2)},
    )

    pairs = [{"positive": "p", "negative": "n"}, {"positive": "p2", "negative": "n2"}]
    class FakeModel:
        def parameters(self):
            yield torch.zeros(1)
    class FakeTok:
        pass

    layers_list = [object()] * 2
    sae = MockSaeBackend(layers=frozenset({5, 7}), d_model=4)

    with pytest.raises(SaeCoverageError):
        V.extract_contrastive(
            FakeModel(), FakeTok(), pairs, layers=layers_list,
            device=torch.device("cpu"),
            sae=sae,
        )


def test_extract_contrastive_sae_bakes_shares_proportional_to_evr(monkeypatch):
    """Per-layer baked magnitudes should scale with the layer's share (evr/sum)."""
    import torch
    from saklas.core import vectors as V
    from saklas.core.sae import MockSaeBackend

    # Seed so the noise doesn't flip signs or perturb magnitudes.
    torch.manual_seed(0)

    def fake_encode_and_capture(model, tokenizer, text, layers, device):
        # Layer 0: high-SNR separation (evr near 1.0)
        # Layer 1: low-SNR separation (evr lower — noise dominates)
        out = {}
        sign = 1.0 if "pos" in text else -1.0
        out[0] = torch.tensor([sign * 5.0, 0.0, 0.0, 0.0]) + 0.01 * torch.randn(4)
        out[1] = torch.tensor([sign * 0.2, 0.0, 0.0, 0.0]) + 0.5 * torch.randn(4)
        return out

    monkeypatch.setattr(V, "_encode_and_capture_all", fake_encode_and_capture)
    pairs = [{"positive": f"pos_{i}", "negative": f"neg_{i}"} for i in range(30)]

    class FakeModel:
        def parameters(self):
            yield torch.zeros(1)
    class FakeTok:
        pass

    layers_list = [object()] * 2
    sae = MockSaeBackend(layers=frozenset({0, 1}), d_model=4)

    profile = V.extract_contrastive(
        FakeModel(), FakeTok(), pairs, layers=layers_list,
        device=torch.device("cpu"),
        sae=sae,
    )
    # Both layers covered, both have signal, both non-zero
    assert set(profile.keys()) == {0, 1}
    mag_0 = profile[0].norm().item()
    mag_1 = profile[1].norm().item()
    # Shares sum to 1.0 in the sense that (mag_i / ref_norm_i) sums to 1.0.
    # Simpler check: layer 0 has much stronger signal → its share (and thus
    # its baked magnitude, at comparable ref_norms) should dominate.
    assert mag_0 > mag_1


def test_sae_lens_backend_encodes_and_decodes(monkeypatch):
    """SaeLensBackend wraps per-layer SAE modules and dispatches by layer index."""
    import torch
    import sys
    import types

    fake_sae_lens = types.ModuleType("sae_lens")

    class FakeSAE:
        def __init__(self, d_in, d_sae, hook_layer):
            self.cfg = types.SimpleNamespace(
                d_in=d_in, d_sae=d_sae, model_name="test-model", hook_layer=hook_layer,
            )
            self.W_enc = torch.eye(d_in)[:, :d_sae] if d_in >= d_sae else torch.zeros(d_in, d_sae)
            self.W_dec = self.W_enc.T
            self.b_enc = torch.zeros(d_sae)

        def encode(self, x):
            return x @ self.W_enc + self.b_enc

        def decode(self, f):
            return f @ self.W_dec

        @classmethod
        def from_pretrained(cls, release, sae_id, device=None):
            hook_layer = int(sae_id.split("_")[1])
            return (
                cls(d_in=4, d_sae=4, hook_layer=hook_layer),
                {"d_in": 4, "d_sae": 4, "hook_layer": hook_layer},
                None,
            )

    fake_sae_lens.SAE = FakeSAE
    fake_sae_lens.get_pretrained_saes_directory = lambda: {
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

    h = torch.randn(3, 4)
    f = backend.encode_layer(5, h)
    assert f.shape == (3, 4)
    v = backend.decode_layer(5, torch.randn(4))
    assert v.shape == (4,)


def test_sae_lens_backend_missing_dep_raises(monkeypatch):
    """When sae_lens isn't installed, load_sae_backend raises SaeBackendImportError."""
    import sys
    monkeypatch.setitem(sys.modules, "sae_lens", None)
    from saklas.core.sae import load_sae_backend
    from saklas.core.errors import SaeBackendImportError
    with pytest.raises(SaeBackendImportError):
        load_sae_backend("any", model_id="m", device="cpu")


def test_sae_lens_backend_release_not_found(monkeypatch):
    import sys
    import types

    fake = types.ModuleType("sae_lens")
    fake.get_pretrained_saes_directory = lambda: {
        "mock-a": {"saes_map": {}, "model": "m"},
        "mock-b": {"saes_map": {}, "model": "m"},
    }
    fake.SAE = object
    monkeypatch.setitem(sys.modules, "sae_lens", fake)

    from saklas.core.sae import load_sae_backend
    from saklas.core.errors import SaeReleaseNotFoundError
    with pytest.raises(SaeReleaseNotFoundError) as exc:
        load_sae_backend("nonexistent", model_id="m", device="cpu")
    msg = str(exc.value)
    # Message should list near matches so user knows what's available.
    assert "mock-a" in msg or "mock-b" in msg


def test_sae_lens_backend_model_mismatch(monkeypatch):
    import sys
    import types
    import torch

    fake = types.ModuleType("sae_lens")

    class FakeSAE:
        def __init__(self):
            self.cfg = types.SimpleNamespace(model_name="other-model", hook_layer=0)

        @classmethod
        def from_pretrained(cls, release, sae_id, device=None):
            return cls(), {"hook_layer": 0}, None

    fake.SAE = FakeSAE
    fake.get_pretrained_saes_directory = lambda: {
        "mock": {"saes_map": {"layer_0": 0}, "model": "other-model"},
    }
    monkeypatch.setitem(sys.modules, "sae_lens", fake)

    from saklas.core.sae import load_sae_backend
    from saklas.core.errors import SaeModelMismatchError
    with pytest.raises(SaeModelMismatchError):
        load_sae_backend("mock", model_id="my-model", device="cpu")


def test_sae_lens_backend_canonical_layer_map_warns_on_multiple(monkeypatch, recwarn):
    """When a release has multiple SAEs per layer, pick narrowest + warn."""
    import sys
    import types
    import torch

    fake = types.ModuleType("sae_lens")

    class FakeSAE:
        def __init__(self):
            self.cfg = types.SimpleNamespace(model_name="test-model", hook_layer=0)

        @classmethod
        def from_pretrained(cls, release, sae_id, device=None):
            # Parse `layer_N` prefix out of canonical sae_id strings like
            # `layer_0/width_16k/l0_100`.
            import re
            m = re.search(r"layer[_-]?(\d+)", sae_id)
            layer = int(m.group(1)) if m else 0
            sae = cls()
            sae.cfg.hook_layer = layer
            return sae, {"hook_layer": layer}, None

    fake.SAE = FakeSAE
    fake.get_pretrained_saes_directory = lambda: {
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
