"""Steering dataclass unit tests — from_value coercion."""

from saklas.core.steering import Steering


def test_construct():
    s = Steering(alphas={"foo": 0.5})
    assert s.alphas == {"foo": 0.5}
    assert s.thinking is None


def test_thinking_field():
    s = Steering(alphas={"foo": 0.1}, thinking=True)
    assert s.thinking is True


def test_from_value_dict_to_steering():
    s = Steering.from_value({"foo": 0.5})
    assert isinstance(s, Steering)
    assert s.alphas == {"foo": 0.5}


def test_from_value_none_returns_none():
    assert Steering.from_value(None) is None


def test_from_value_passthrough():
    s = Steering(alphas={"x": 1.0})
    assert Steering.from_value(s) is s


def test_from_value_empty_dict():
    s = Steering.from_value({})
    assert isinstance(s, Steering)
    assert dict(s.alphas) == {}
