"""Shared TUI helpers."""


def build_bar(value: float, max_value: float, width: int = 20) -> tuple[str, str]:
    """Return (filled, empty) bar halves sized by abs(value)/max_value."""
    filled = min(int(abs(value) / max_value * width), width)
    return "█" * filled, "░" * (width - filled)
