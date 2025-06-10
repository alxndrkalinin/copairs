import pytest

from copairs.matching import _validate


def test_validate_string_inputs():
    """_validate should convert string inputs to tuples."""
    sameby, diffby = _validate("c", "p")
    assert sameby == ("c",)
    assert diffby == ("p",)
