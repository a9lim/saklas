from __future__ import annotations

import pytest

from saklas.cli.alpha_grid import AlphaListError, parse_alpha_list


def test_parse_alpha_list_comma_list() -> None:
    assert parse_alpha_list("0.0, 0.3, 0.7") == [0.0, 0.3, 0.7]


def test_parse_alpha_list_strips_empty_entries() -> None:
    assert parse_alpha_list("0.1, ,0.2,") == [0.1, 0.2]


def test_parse_alpha_list_linspace() -> None:
    assert parse_alpha_list("linspace(-1, 1, 3)") == pytest.approx([-1.0, 0.0, 1.0])


def test_parse_alpha_list_linspace_single_point() -> None:
    assert parse_alpha_list("linspace(0.5, 1.0, 1)") == [0.5]


def test_parse_alpha_list_range_form() -> None:
    assert parse_alpha_list("0.0:1.0:0.25") == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])


@pytest.mark.parametrize(
    "value",
    ["", "  ", "linspace(0, 1, 0)", "0:1:0", "0:1:-0.1", "0.1, banana, 0.2"],
)
def test_parse_alpha_list_rejects_invalid_values(value: str) -> None:
    with pytest.raises(AlphaListError):
        parse_alpha_list(value)
