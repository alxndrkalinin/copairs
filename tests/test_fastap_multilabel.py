"""Tests for inverted-index multilabel matching and per-label AP."""

import numpy as np
import pandas as pd
import pytest

from copairs import compute, matching
from copairs.map import multilabel

N_LABELS = 12


def label_frame(seed=0, n=120, empty=True):
    """Rows with 0-3 labels each and two monolabel columns."""
    rng = np.random.default_rng(seed)
    labels = [
        list(
            rng.choice(
                N_LABELS, size=rng.integers(0 if empty else 1, 4), replace=False
            ).astype(str)
        )
        for _ in range(n)
    ]
    return pd.DataFrame(
        {
            "labels": labels,
            "plate": rng.integers(3, size=n).astype(str),
            "well": rng.integers(5, size=n).astype(str),
        }
    )


def sql_pairs(*args):
    """find_pairs_multilabel through the DuckDB implementation."""
    return matching.find_pairs_multilabel(*args, method="legacy")


def as_set(pairs):
    """Unordered pair set."""
    return set(map(tuple, np.sort(np.asarray(pairs, dtype=np.int64), axis=1)))


@pytest.mark.parametrize(
    "sameby,diffby",
    [
        (["labels"], []),
        (["labels", "plate"], []),
        (["labels"], ["well"]),
        (["labels"], ["labels"]),  # share a label, but not the whole list
    ],
)
def test_shared_label_pairs_match_sql(sameby, diffby):
    """Pairs sharing a label, and per-label counts, equal the SQL result."""
    dframe = label_frame()
    args = (dframe, sameby, diffby, "labels")
    assert matching._find_pairs_multilabel_fast(*args) is not None
    pairs, keys, counts = matching.find_pairs_multilabel(*args)
    sql_pairs_, sql_keys, sql_counts = sql_pairs(*args)
    np.testing.assert_array_equal(keys, sql_keys)
    np.testing.assert_array_equal(counts, sql_counts)
    start = 0
    for count in counts:  # same pairs label by label
        got = as_set(pairs[start : start + count])
        expected = as_set(sql_pairs_[start : start + count])
        assert got == expected
        start += count
    assert pairs.dtype == np.uint32


@pytest.mark.parametrize(
    "sameby,diffby",
    [([], ["labels"]), (["plate"], ["labels"]), ([], ["labels", "well"])],
)
def test_disjoint_label_pairs_match_sql(sameby, diffby):
    """Pairs sharing no label equal the SQL result, sorted and unique."""
    dframe = label_frame(1)
    args = (dframe, sameby, diffby, "labels")
    assert matching._find_pairs_multilabel_fast(*args) is not None
    pairs = matching.find_pairs_multilabel(*args)
    assert as_set(pairs) == as_set(sql_pairs(*args))
    keys = pairs[:, 0].astype(np.int64) * len(dframe) + pairs[:, 1]
    assert (np.diff(keys) > 0).all() and (pairs[:, 0] < pairs[:, 1]).all()


@pytest.mark.parametrize(
    "labels,index",
    [
        ([["a", None], ["a"], ["b"]], None),  # missing value inside a list
        ([["a", "a"], ["a"], ["b"]], None),  # duplicate label in a row
        ([["a"], [1], ["b"]], None),  # mixed label types
        ([["a"], ["a"], ["b"]], [5, 6, 7]),  # non-default index
    ],
)
def test_unsupported_labels_fall_back(labels, index):
    """Inputs the inverted index cannot represent exactly use the SQL path."""
    dframe = pd.DataFrame({"labels": labels}, index=index)
    assert (
        matching._find_pairs_multilabel_fast(dframe, ["labels"], [], "labels") is None
    )


def consistency_input(seed=2):
    """Profiles with labels such that every profile has negatives."""
    dframe = label_frame(seed, n=150, empty=False)
    feats = np.random.default_rng(seed).normal(size=(len(dframe), 10))
    return dframe, feats


CONSISTENCY = dict(
    pos_sameby=["labels"], pos_diffby=[], neg_sameby=[], neg_diffby=["labels"]
)


@pytest.mark.parametrize("backend", ["numba"])
def test_multilabel_ap_matches_legacy(backend):
    """Per-label APs equal the legacy loop for the same similarities."""
    dframe, feats = consistency_input()

    def cosine(x, y):
        return compute.pairwise_cosine(x, y)

    kwargs = dict(
        CONSISTENCY, multilabel_col="labels", distance=cosine, progress_bar=False
    )
    fast = multilabel.average_precision(dframe, feats, backend=backend, **kwargs)
    legacy = multilabel.average_precision(dframe, feats, method="legacy", **kwargs)
    pd.testing.assert_frame_equal(
        fast.drop(columns=["average_precision", "normalized_average_precision"]),
        legacy.drop(columns=["average_precision", "normalized_average_precision"]),
    )
    for col in ["average_precision", "normalized_average_precision"]:
        np.testing.assert_allclose(fast[col], legacy[col], rtol=1e-12, atol=1e-15)


def test_numeric_label_keys():
    """Integer labels give integer keys, equal to the SQL path's."""
    dframe = pd.DataFrame({"labels": [[1, 2], [2], [1, 3], [3]]})
    _, keys, counts = matching.find_pairs_multilabel(dframe, ["labels"], [], "labels")
    _, sql_keys, sql_counts = sql_pairs(dframe, ["labels"], [], "labels")
    assert np.issubdtype(keys.dtype, np.integer)
    np.testing.assert_array_equal(keys, sql_keys)
    np.testing.assert_array_equal(counts, sql_counts)
