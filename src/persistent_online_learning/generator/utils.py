"""Small validation helpers shared by generator primitives."""

import math


def require_positive_int(name: str, value: int) -> None:
    """Reject non-integer or nonpositive dimension values."""

    if type(value) is not int:
        raise TypeError(f"{name} must use int")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def require_probability(name: str, value: float) -> float:
    """Validate a finite probability and return its float representation."""

    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise TypeError(f"{name} must be a finite number")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be between zero and one")
    return result
