"""Class for getting Percent replicating metric"""

from typing import List, Literal

import numpy as np
import pandas as pd

from copairs.compute import get_distance_fn

from .matching import Matcher


def corr_from_null_pairs(X: np.ndarray, null_pairs, n_replicates):
    """Correlation from a given list of unnamed pairs."""
    null_pairs = np.asarray(null_pairs, int)
    corr_fn = get_distance_fn("correlation")
    corrs = corr_fn(X, null_pairs, batch_size=20000)
    corrs = corrs.reshape(-1, n_replicates)
    null_dist = np.nanmedian(corrs, axis=1)
    return pd.Series(null_dist)


def corr_between_non_replicates(
    X: np.ndarray,
    meta: pd.DataFrame,
    n_samples: int,
    n_replicates: int,
    diffby: List[str],
):
    """
    Null distribution between random "replicates".
    Parameters:
    ------------
    df: pandas.DataFrame
    n_samples: int
    n_replicates: int
    diffby: list of columns that should be different
    use_rep: which data to use from .obsm property. `None` (default) uses `adata.X`
    Returns:
    --------
    list-like of correlation values, with a  length of `n_samples`
    """
    matcher = Matcher(meta, diffby, seed=0)
    n_pairs = n_replicates * n_samples

    null_pairs = [matcher.sample_null_pair(diffby) for _ in range(n_pairs)]
    return corr_from_null_pairs(X, null_pairs, n_replicates)


def corr_from_pairs(X: np.ndarray, pairs: dict, sameby: List[str]):
    """
    Correlation from a list of named pairs. Generated by Matcher.get_all_pairs
    Parameters:
    -----------
    X: Matrix containing samples in rows
    pairs: dictionary with list of index pairs.
    Returns:
    --------
    list-like of correlation values and median of number of replicates
    """
    pair_ix = np.vstack(list(pairs.values()))
    corr_fn = get_distance_fn("correlation")
    corrs = corr_fn(X, pair_ix, batch_size=20000)
    counts = [len(v) for v in pairs.values()]

    if len(sameby) == 1:
        sameby_vals = np.repeat(list(pairs.keys()), counts)
    else:
        sameby_vals = np.repeat(list(map("_".join, pairs.keys())), counts)

    sameby_col = "_".join(sameby)

    corrs = pd.DataFrame(
        {
            sameby_col: sameby_vals,
            "corr": corrs,
            "row_x": pair_ix[:, 0],
            "row_y": pair_ix[:, 1],
        }
    )
    corrs = corrs.groupby(sameby_col).agg(
        {"corr": ["median", "count"], "row_x": "nunique"}
    )

    median_num_repl = int(corrs["row_x", "nunique"].median())
    corr_dist = corrs["corr"]

    return corr_dist, median_num_repl


def corr_between_replicates(
    X: np.ndarray, meta: pd.DataFrame, sameby: List[str], diffby: List[str]
):
    """
    Correlation between replicates
    Parameters:
    -----------
    adata: ad.AnnData
    sameby: Feature name to group the data frame by
    diffby: Feature name to force different values
    use_rep: which data to use from .obsm property. `None` (default) uses `adata.X`
    Returns:
    --------
    list-like of correlation values and median of number of replicates
    """
    matcher = Matcher(meta, sameby + diffby, seed=0)
    pairs = matcher.get_all_pairs(sameby, diffby)
    return corr_from_pairs(X, pairs, sameby)


class CorrelationTestResult:
    """Class representing the percent replicating score. It stores distributions"""

    def __init__(self, corr_df: pd.DataFrame, null_dist: pd.Series):
        """Initialize object"""
        self.corr_df = corr_df
        self.corr_dist = corr_df["median"]
        self.null_dist = null_dist

    def percent_score_left(self):
        """
        Calculates the percent score using the 5th percentile threshold.
        :return: proportion of correlation distribution beyond the threshold and the threshold
        """
        perc_5 = np.nanpercentile(self.null_dist, 5)
        below_threshold = self.corr_dist.dropna() < perc_5
        return np.nanmean(below_threshold.astype(float)), perc_5

    def percent_score_right(self):
        """
        Calculates the percent score using the 95th percentile threshold.
        :return: proportion of correlation distribution beyond the threshold and the threshold
        """
        perc_95 = np.nanpercentile(self.null_dist, 95)
        above_threshold = self.corr_dist.dropna() > perc_95
        return np.nanmean(above_threshold.astype(float)), perc_95

    def percent_score_both(self):
        """
        Calculates the percent score using the 5th and 95th percentile or thresholds.
        :return: proportion of correlation distribution beyond the thresholds and the thresholds
        """
        perc_95 = np.nanpercentile(self.null_dist, 95)
        above_threshold = self.corr_dist.dropna() > perc_95
        perc_5 = np.nanpercentile(self.null_dist, 5)
        below_threshold = self.corr_dist.dropna() < perc_5
        return (
            (
                np.nanmean(above_threshold.astype(float))
                + np.nanmean(below_threshold.astype(float))
            ),
            perc_5,
            perc_95,
        )

    def percent_score(self, how: Literal["left", "right", "both"]):
        left_th, right_th = None, None
        if how == "right":
            percent_score, right_th = self.percent_score_right()
        elif how == "left":
            percent_score, left_th = self.percent_score_left()
        elif how == "both":
            percent_score, left_th, right_th = self.percent_score_both()
        else:
            raise ValueError(f"Invalid value: {how} for how param")

        return percent_score, left_th, right_th

    def wasserstein_distance(self):
        """
        Compute the Wasserstein distance between null and corr distributions.
        """
        from scipy.stats import wasserstein_distance

        return wasserstein_distance(self.null_dist.values, self.corr_dist.values)


def correlation_test(
    X: np.ndarray,
    meta: pd.DataFrame,
    sameby: List[str],
    diffby: List[str],
    n_samples: int = 1000,
) -> CorrelationTestResult:
    """
    Generate Null and replicate distribution for replicate correlation analysis
    """
    corr_df, median_num_repl = corr_between_replicates(X, meta, sameby, diffby)

    n_replicates = min(median_num_repl, 50)
    null_dist = corr_between_non_replicates(
        X,
        meta,
        n_samples=n_samples,
        n_replicates=n_replicates,
        diffby=sameby + diffby,
    )

    return CorrelationTestResult(corr_df, null_dist)


def correlation_test_from_pairs(
    X: np.ndarray, pairs: dict, null_pairs: list, sameby: list
) -> CorrelationTestResult:
    """
    Generate Null and replicate distribution for replicate correlation analysis
    """
    corr_df, median_num_repl = corr_from_pairs(X, pairs, sameby)
    n_replicates = min(median_num_repl, 50)
    null_dist = corr_from_null_pairs(X, null_pairs, n_replicates)
    return CorrelationTestResult(corr_df, null_dist)
