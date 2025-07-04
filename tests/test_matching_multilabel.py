"""Tests for the multilabel matching implementation."""

import pandas as pd
import pytest

from tests.helpers import simulate_random_plates
from copairs.matching import find_pairs_multilabel

SEED = 42


def get_naive_pairs(dframe: pd.DataFrame, sameby, diffby, multilabel_col: str):
    """Get pairs using a naive implementation."""
    dframe = dframe.copy()

    dframe[multilabel_col] = dframe[multilabel_col].apply(set)
    cross = dframe.reset_index().merge(
        dframe.reset_index(), how="cross", suffixes=("_x", "_y")
    )
    # remove rows that pair themselves
    cross = cross.query("index_x != index_y").copy()

    def all_diff(row):
        return len(row[f"{multilabel_col}_x"] & row[f"{multilabel_col}_y"]) == 0

    def any_equal(row):
        return len(row[f"{multilabel_col}_x"] & row[f"{multilabel_col}_y"]) > 0

    cross[f"{multilabel_col}_all_diff"] = cross.apply(all_diff, axis=1)
    cross[f"{multilabel_col}_any_equal"] = cross.apply(any_equal, axis=1)
    mask = True
    for col in sameby:
        if col == multilabel_col:
            mask = cross[f"{col}_any_equal"] & mask
        else:
            mask = (cross[f"{col}_x"] == cross[f"{col}_y"]) & mask
    for col in diffby:
        if col == multilabel_col:
            mask = cross[f"{col}_all_diff"] & mask
        else:
            mask = (cross[f"{col}_x"] != cross[f"{col}_y"]) & mask

    pairs = cross.loc[mask, ["index_x", "index_y"]]
    # Drop duplicates
    pairs = pairs.apply(frozenset, axis=1).drop_duplicates().values
    pairs = pd.DataFrame(zip(*pairs), index=["index_x", "index_y"]).T
    pairs = pairs.sort_values(["index_x", "index_y"]).reset_index(drop=True)
    return pairs


def check_naive(dframe, sameby, diffby, multilabel_col):
    """Check find_pairs_multilabel and naive generate same pairs."""
    gt_pairs = get_naive_pairs(dframe, sameby, diffby, multilabel_col)
    vals = find_pairs_multilabel(dframe, sameby, diffby, multilabel_col)
    if multilabel_col in sameby:  # output is different if based on multilabel_col
        vals = vals[0]
    vals = pd.DataFrame(vals, columns=["index_x", "index_y"])
    vals = vals.sort_values(["index_x", "index_y"]).reset_index(drop=True)
    vals = set(vals.apply(frozenset, axis=1))
    gt_pairs = set(gt_pairs.apply(frozenset, axis=1))
    assert gt_pairs == vals


def test_sameby():
    """Check the multilabel implementation with sameby."""
    multilabel_col = "c"
    sameby = ["c"]
    diffby = ["p", "w"]
    dframe = simulate_random_plates(
        n_compounds=4, n_replicates=5, plate_size=5, sameby=sameby, diffby=diffby
    )
    dframe = dframe.groupby(["p", "w"])["c"].unique().reset_index()
    check_naive(dframe, sameby, diffby, multilabel_col)


def test_sameby_other_cols():
    """Check the multilabel implementation with sameby and other cols."""
    multilabel_col = "c"
    sameby = ["c", "p"]
    diffby = ["w"]
    dframe = simulate_random_plates(
        n_compounds=4, n_replicates=5, plate_size=5, sameby=sameby, diffby=diffby
    )
    dframe = dframe.groupby(["p", "w"])["c"].unique().reset_index()
    check_naive(dframe, sameby, diffby, multilabel_col)


def test_diffby():
    """Check the multilabel implementation with sameby."""
    multilabel_col = "c"
    sameby = ["p"]
    diffby = ["c", "w"]
    dframe = simulate_random_plates(
        n_compounds=4, n_replicates=5, plate_size=5, sameby=sameby, diffby=diffby
    )
    dframe = dframe.groupby(["p", "w"])["c"].unique().reset_index()

    check_naive(dframe, sameby, diffby, multilabel_col)


def test_only_diffby():
    """Check the multilabel implementation with only diffby being equal to c."""
    multilabel_col = "c"
    sameby = []
    diffby = ["c"]
    dframe = simulate_random_plates(
        n_compounds=4, n_replicates=5, plate_size=5, sameby=sameby, diffby=diffby
    )
    dframe = dframe.groupby(["p", "w"])["c"].unique().reset_index()
    check_naive(dframe, sameby, diffby, multilabel_col)


def test_only_diffby_many_cols():
    """Check the multilabel implementation with only diffby being equal to c."""
    multilabel_col = "c"
    sameby = []
    diffby = ["c", "w"]
    dframe = simulate_random_plates(
        n_compounds=4, n_replicates=5, plate_size=5, sameby=sameby, diffby=diffby
    )
    dframe = dframe.groupby(["p", "w"])["c"].unique().reset_index()
    check_naive(dframe, sameby, diffby, multilabel_col)


def test_only_sameby_many_cols():
    """Check the multilabel implementation with only diffby being equal to c."""
    multilabel_col = "c"
    sameby = ["c", "w"]
    diffby = []
    dframe = simulate_random_plates(
        n_compounds=4, n_replicates=5, plate_size=5, sameby=sameby, diffby=diffby
    )
    dframe = dframe.groupby(["p", "w"])["c"].unique().reset_index()
    check_naive(dframe, sameby, diffby, multilabel_col)


def test_accepts_tuples_inputs():
    """find_pairs_multilabel should accept tuples for sameby and diffby."""
    multilabel_col = "c"
    sameby = ("c",)
    diffby = ("p", "w")
    dframe = simulate_random_plates(
        n_compounds=4, n_replicates=5, plate_size=5, sameby=sameby, diffby=diffby
    )
    dframe = dframe.groupby(["p", "w"])["c"].unique().reset_index()
    check_naive(dframe, sameby, diffby, multilabel_col)


def test_accepts_string_inputs():
    """find_pairs_multilabel should accept strings for sameby and diffby."""
    multilabel_col = "c"
    sameby = "c"
    diffby = "p"
    dframe = simulate_random_plates(
        n_compounds=4,
        n_replicates=5,
        plate_size=5,
        sameby=[sameby],
        diffby=[diffby],
    )
    dframe = dframe.groupby(["p", "w"])["c"].unique().reset_index()

    gt_pairs = get_naive_pairs(dframe, [sameby], [diffby], multilabel_col)
    with pytest.deprecated_call():
        vals = find_pairs_multilabel(dframe, sameby, diffby, multilabel_col)
    if multilabel_col == sameby:
        vals = vals[0]
    vals = pd.DataFrame(vals, columns=["index_x", "index_y"])
    vals = vals.sort_values(["index_x", "index_y"]).reset_index(drop=True)

    assert set(vals.apply(frozenset, axis=1)) == set(gt_pairs.apply(frozenset, axis=1))
