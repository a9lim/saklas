"""SamplingConfig unit tests — frozen, merged_with, stop-as-tuple."""

import pytest

from saklas.core.sampling import SamplingConfig


def test_defaults():
    sc = SamplingConfig()
    assert sc.temperature is None
    assert sc.top_p is None
    assert sc.top_k is None
    assert sc.max_tokens is None
    assert sc.seed is None
    assert sc.stop is None
    assert sc.logit_bias is None
    assert sc.presence_penalty == 0.0
    assert sc.frequency_penalty == 0.0
    assert sc.logprobs is None
    assert sc.return_hidden is False


def test_frozen():
    sc = SamplingConfig(temperature=0.5)
    with pytest.raises((AttributeError, Exception)):
        sc.temperature = 1.0  # type: ignore[misc]


def test_stop_coerced_to_tuple():
    # __post_init__ coerces list→tuple at runtime; type annotation narrows
    # the public contract to tuple, so the list input here needs a pragma.
    sc = SamplingConfig(stop=["a", "b"])  # type: ignore[arg-type]
    assert sc.stop == ("a", "b")
    assert isinstance(sc.stop, tuple)

    sc2 = SamplingConfig(stop=("x",))
    assert sc2.stop == ("x",)


def test_merged_with_other_wins():
    base = SamplingConfig(temperature=0.7, top_p=0.9, max_tokens=128)
    override = SamplingConfig(temperature=0.5)
    merged = base.merged_with(override)
    assert merged.temperature == 0.5
    assert merged.top_p == 0.9
    assert merged.max_tokens == 128


def test_merged_with_none_returns_self():
    base = SamplingConfig(temperature=0.7)
    assert base.merged_with(None) is base


def test_merged_with_preserves_penalties():
    base = SamplingConfig(presence_penalty=0.1)
    override = SamplingConfig(frequency_penalty=0.3)
    merged = base.merged_with(override)
    assert merged.presence_penalty == 0.1
    assert merged.frequency_penalty == 0.3


def test_merged_with_all_defaults_is_noop():
    base = SamplingConfig(temperature=0.7)
    override = SamplingConfig()
    merged = base.merged_with(override)
    assert merged.temperature == 0.7


def test_sampling_config_return_hidden_defaults_false():
    cfg = SamplingConfig()
    assert cfg.return_hidden is False


def test_sampling_config_return_hidden_merged_with_override():
    base = SamplingConfig(temperature=0.7)
    override = SamplingConfig(return_hidden=True)
    merged = base.merged_with(override)
    assert merged.return_hidden is True
    assert merged.temperature == 0.7  # unrelated field preserved


def test_sampling_config_return_hidden_merged_with_default_does_not_override():
    base = SamplingConfig(return_hidden=True)
    merged = base.merged_with(SamplingConfig())  # other is all-defaults
    assert merged.return_hidden is True  # other.return_hidden=False is the default; must not override
