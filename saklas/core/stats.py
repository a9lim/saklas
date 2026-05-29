"""Small statistics helpers shared across CLI and core diagnostics."""

from __future__ import annotations

from collections.abc import Iterable


def median_or_zero(values: Iterable[float]) -> float:
    """Return the median of ``values``, or ``0.0`` for an empty iterable."""
    sorted_values = sorted(float(v) for v in values)
    n = len(sorted_values)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2 == 1:
        return sorted_values[mid]
    return 0.5 * (sorted_values[mid - 1] + sorted_values[mid])
