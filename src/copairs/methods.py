"""The implementation choice shared by copairs' public functions."""

METHODS = ("fast", "legacy")


def check_method(method: str) -> None:
    """Raise ValueError unless ``method`` is ``"fast"`` or ``"legacy"``."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
