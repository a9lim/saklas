"""Alpha-grid parsing for ``saklas experiment fan``."""

from __future__ import annotations

import math
import re

from saklas.core.errors import SaklasError


class AlphaListError(ValueError, SaklasError):
    """Raised when :func:`parse_alpha_list` cannot parse its input."""

    def user_message(self) -> tuple[int, str]:
        return (400, str(self) or self.__class__.__name__)


_LINSPACE_RE = re.compile(
    r"^linspace\s*\(\s*([^,]+?)\s*,\s*([^,]+?)\s*,\s*([^,)]+?)\s*\)\s*$",
    re.IGNORECASE,
)


def parse_alpha_list(text: str) -> list[float]:
    """Parse comma lists, ``start:stop:step``, or ``linspace(a, b, n)``."""
    text = (text or "").strip()
    if not text:
        raise AlphaListError("alpha list is empty")

    match = _LINSPACE_RE.match(text)
    if match:
        try:
            start = float(match.group(1))
            stop = float(match.group(2))
        except ValueError:
            raise AlphaListError("linspace bounds must be numbers") from None
        try:
            count = int(match.group(3))
        except ValueError:
            raise AlphaListError("linspace count must be a positive integer") from None
        if count < 1:
            raise AlphaListError("linspace count must be a positive integer")
        if not math.isfinite(start) or not math.isfinite(stop):
            raise AlphaListError("linspace bounds must be numbers")
        if count == 1:
            return [start]
        step = (stop - start) / (count - 1)
        return [start + step * i for i in range(count)]

    if ":" in text and "," not in text:
        parts = [part.strip() for part in text.split(":")]
        if len(parts) != 3:
            raise AlphaListError("range form is start:stop:step (three values)")
        try:
            start, stop, step = (float(part) for part in parts)
        except ValueError:
            raise AlphaListError("range values must be numbers") from None
        if not all(math.isfinite(value) for value in (start, stop, step)):
            raise AlphaListError("range values must be numbers")
        if step == 0:
            raise AlphaListError("range step must be non-zero")
        if (stop - start) * step < 0:
            raise AlphaListError("range step direction disagrees with start→stop")

        epsilon = abs(step) * 1e-9
        values: list[float] = []
        current = start
        ascending = step > 0
        while len(values) < 10_000:
            if ascending and current > stop + epsilon:
                break
            if not ascending and current < stop - epsilon:
                break
            values.append(round(current, 12))
            current += step
        else:
            raise AlphaListError("range produced too many values")
        if not values:
            raise AlphaListError("range produced no values")
        return values

    values = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = float(token)
        except ValueError:
            raise AlphaListError(f"'{token}' is not a number") from None
        if not math.isfinite(value):
            raise AlphaListError(f"'{token}' is not a finite number")
        values.append(value)
    if not values:
        raise AlphaListError("no values parsed")
    return values
