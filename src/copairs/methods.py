"""The implementation choices shared by copairs' public functions."""

METHODS = ("fast", "legacy")
BACKENDS = ("auto", "cuda", "numba", "numpy")


def check_method(method: str, backend: str = "auto") -> None:
    """Raise ValueError unless ``method`` and ``backend`` name known choices.

    The backend name is checked under ``method="legacy"`` too, which ignores
    it, so a misspelled backend fails whichever method is used.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; expected one of {BACKENDS}")
