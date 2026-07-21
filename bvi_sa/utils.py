"""Provide shared filesystem, serialization, and plotting utility helpers.

This module is part of the BVI ACT-R navigation simulation workflow.
"""


def clamp(value, low=0.0, high=1.0):
    """Handle clamp behavior."""
    return max(low, min(high, value))


def mean_safe(values):
    """Handle mean safe behavior."""
    return sum(values) / len(values) if values else 0.0


def atom_to_float(value):
    """Handle atom to float behavior."""
    try:
        return float(value)
    except (TypeError, ValueError):
        text = str(value)
        try:
            return float(text)
        except (TypeError, ValueError):
            return None


def atom_to_name(value):
    """Handle atom to name behavior."""
    text = str(value).strip()
    if text.startswith("=") or text.startswith("~"):
        return None
    return text
