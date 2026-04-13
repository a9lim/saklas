"""Shared TUI helpers."""


def build_bar(value: float, max_value: float, width: int = 20) -> tuple[str, str]:
    filled = min(int(abs(value) / max_value * width), width)
    return "█" * filled, "░" * (width - filled)
