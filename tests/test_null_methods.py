"""Tests for backend= of copairs' public AP and mAP p-value functions, and their nulls."""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytest

import copairs.map.map as map_module
from copairs import nulls, compute
from copairs.map import (
    get_map_pvalue,
    mean_average_precision,
    mean_average_precision_hierarchical,
)
from copairs.nulls import sampler
from tests.helpers import (
    ap_frame,
    exact_ap_null,
    brute_ap_pvalues,
    brute_map_pvalues,
)
from copairs.matching import UnpairedException
from copairs.map.normalization import expected_ap
from copairs.map.average_precision import p_values as frame_p_values

BACKENDS = nulls.available_backends()
CONFS = np.array([[1, 7], [3, 10], [5, 20], [9, 49], [99, 1099], [3, 1000]])


def random_ap_frame(rng, n=400, n_groups=60):
    """AP scores of ``n`` profiles in ``n_groups`` groups, near their expected AP."""
    frame = pd.DataFrame(
        {
            "g": rng.integers(n_groups, size=n).astype(str),
            "n_pos_pairs": rng.integers(3, 8, size=n),
        }
    )
    frame["n_total_pairs"] = frame["n_pos_pairs"] + rng.integers(40, 60, size=n)
    m, total = frame["n_pos_pairs"], frame["n_total_pairs"]
    mu = np.array([expected_ap(a, b - a) for a, b in zip(m, total)])
    frame["average_precision"] = np.clip(mu + rng.normal(0, 0.15, size=n), 0, 1)
    frame["normalized_average_precision"] = 0.0
    return frame


@pytest.mark.parametrize("backend", BACKENDS)
def test_nulls_are_copairs_nulls_and_not_cached(backend, tmp_path, monkeypatch):
    """get_null_dists samples with copairs.nulls; nothing is written to ~/.copairs."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    got = compute.get_null_dists(
        CONFS, 300, seed=4, progress_bar=False, backend=backend
    )
    np.testing.assert_array_equal(got, nulls.ap_nulls(CONFS, 300, seed=4))
    frame = random_ap_frame(np.random.default_rng(0), n=50, n_groups=8)
    mean_average_precision(
        frame, ["g"], 300, 0.05, 4, progress_bar=False, null="copied", backend=backend
    )
    compute.p_values([0.5], [[3, 10]], 300, 4, progress_bar=False, backend=backend)
    assert not (tmp_path / "home").exists()


@pytest.mark.parametrize("backend", BACKENDS)
def test_compute_p_values_match_brute_force(backend):
    """compute.p_values equals the brute-force count over the materialized nulls."""
    rng = np.random.default_rng(7)
    null_confs = CONFS[rng.integers(len(CONFS), size=200)]
    scores = rng.random(200)
    got = compute.p_values(
        scores, null_confs, 3000, seed=1, progress_bar=False, backend=backend
    )
    confs, rev_ix = np.unique(null_confs, axis=0, return_inverse=True)
    expected = brute_ap_pvalues(scores, rev_ix.ravel(), confs, 3000, 1)
    np.testing.assert_array_equal(got, expected.astype(np.float32))


@pytest.mark.parametrize("backend", BACKENDS)
def test_frame_p_values(backend):
    """average_precision.p_values passes the backend to compute.p_values."""
    rng = np.random.default_rng(8)
    frame = random_ap_frame(rng, n=40)
    frame.loc[:4, "n_pos_pairs"] = 0
    got = frame_p_values(frame, 1000, seed=2, progress_bar=False, backend=backend)
    assert np.isnan(got[:5]).all()
    valid = frame.iloc[5:]
    expected = compute.p_values(
        valid["average_precision"].to_numpy(),
        valid[["n_pos_pairs", "n_total_pairs"]].values,
        1000,
        2,
        progress_bar=False,
    )
    np.testing.assert_array_equal(got[5:], expected)


@pytest.mark.parametrize("backend", BACKENDS)
def test_copied_map_matches_brute_force(backend):
    """Copied mAP p-values count group means of the exact AP nulls, ties as ``>=``."""
    rng = np.random.default_rng(2)
    frame = random_ap_frame(rng, n=200, n_groups=30)
    # A group whose mAP equals one of its null samples exactly.
    frame.loc[:2, "g"] = "tied"
    frame.loc[:2, ["n_pos_pairs", "n_total_pairs"]] = [3, 10]
    tied = nulls.ap_nulls([[3, 10]], 1, seed=6, dtype=np.float64)[0, 0]
    frame.loc[:2, "average_precision"] = tied
    result = mean_average_precision(
        frame,
        ["g"],
        2000,
        0.05,
        seed=6,
        progress_bar=False,
        null="copied",
        backend=backend,
    )
    confs, rev_ix = np.unique(
        frame[["n_pos_pairs", "n_total_pairs"]].values, axis=0, return_inverse=True
    )
    # Members sharing a configuration share samples.
    members = [
        np.unique(rev_ix.ravel()[ix], return_counts=True) for ix in result["indices"]
    ]
    ptr = np.cumsum([0] + [len(c) for c, _ in members])
    conf_ix = np.concatenate([c for c, _ in members])
    conf_cnt = np.concatenate([n for _, n in members])
    expected = brute_map_pvalues(
        result["mean_average_precision"].to_numpy(),
        ptr,
        conf_ix,
        conf_cnt,
        confs,
        2000,
        6,
    )
    np.testing.assert_array_equal(result["p_value"].to_numpy(), expected)
    tied_p = result.set_index("g").loc["tied", "p_value"]
    null = nulls.ap_nulls([[3, 10]], 2000, seed=6, dtype=np.float64)[0]
    assert tied_p > ((null > tied).sum() + 1) / 2001


def exact_group_null(members):
    """Every value of a group's copied null, one per equally likely placement.

    ``members`` maps each configuration to its number of members. Members with
    equal configurations share one AP draw, and distinct configurations draw
    independently, so the null is the members' weighted mean over every
    combination of the configurations' placements of positives.
    """
    values, weights = np.zeros(1), 0
    for conf, count in members.items():
        values = (values[:, None] + count * exact_ap_null(*conf)[None, :]).ravel()
        weights += count
    return values / weights


def tail_scores(null):
    """Scores on four atoms of ``null`` and one between its two largest.

    The atoms are the median, 90th-percentile, second-largest and largest.
    """
    atoms = np.unique(np.round(null, 12))
    picks = atoms[[len(atoms) // 2, int(0.9 * len(atoms)), -2, -1]]
    return [*picks, (atoms[-2] + atoms[-1]) / 2]


def assert_estimates(got, exact, null_size):
    """Assert Monte Carlo p-values ``(hits + 1) / (draws + 1)`` estimate ``exact``."""
    got, exact = np.asarray(got, dtype=np.float64), np.asarray(exact)
    mean = (null_size * exact + 1) / (null_size + 1)
    se = np.sqrt(exact * (1 - exact) / null_size)
    assert (np.abs(got - mean) <= 5 * se + 1e-7).all(), np.c_[got, mean, se]


@pytest.mark.parametrize("backend", BACKENDS)
def test_ap_pvalues_match_exact_null(backend):
    """AP p-values estimate P(AP >= score) under the exact AP null.

    Small configurations are enumerated. Scores on an atom count its mass, and
    scores between atoms do not.
    """
    scores, confs, exact = [], [], []
    for conf in [(1, 5), (2, 6), (3, 8), (2, 10)]:
        null = exact_ap_null(*conf)
        for score in tail_scores(null):
            scores.append(score)
            confs.append(conf)
            exact.append(np.mean(null >= nulls.tie_thresholds(score)))
    got = compute.p_values(
        np.array(scores), np.array(confs), 50_000, 3, False, backend=backend
    )
    assert_estimates(got, exact, 50_000)


@pytest.mark.parametrize("backend", BACKENDS)
def test_copied_map_pvalues_match_exact_null(backend):
    """Copied mAP p-values estimate P(mAP >= score) under the exact copied null.

    Groups of members sharing one configuration keep that configuration's AP
    null; groups mixing configurations average independent draws, weighted by
    their member counts.
    """
    designs = {
        "shared": {(2, 6): 3},
        "pair": {(1, 5): 1, (2, 6): 1},
        "weighted": {(1, 5): 2, (3, 8): 1},
    }
    rows, exact = [], {}
    for name, members in designs.items():
        null = exact_group_null(members)
        for k, score in enumerate(tail_scores(null)):
            exact[f"{name}{k}"] = np.mean(null >= nulls.tie_thresholds(score))
            for (num_pos, total), count in members.items():
                rows += [(f"{name}{k}", score, num_pos, total)] * count
    result = mean_average_precision(
        ap_frame(rows),
        ["g"],
        50_000,
        0.05,
        seed=3,
        progress_bar=False,
        null="copied",
        backend=backend,
    ).set_index("g")
    assert_estimates(result.loc[list(exact), "p_value"], list(exact.values()), 50_000)


def test_backends_identical_through_public_api():
    """NumPy and Numba give identical AP and mAP p-values."""
    pytest.importorskip("numba")
    frame = random_ap_frame(np.random.default_rng(4))
    kwargs = dict(null_size=5000, threshold=0.05, seed=1, progress_bar=False)
    a = mean_average_precision(frame, ["g"], null="copied", backend="numpy", **kwargs)
    b = mean_average_precision(frame, ["g"], null="copied", backend="numba", **kwargs)
    pd.testing.assert_frame_equal(a, b)
    confs = frame[["n_pos_pairs", "n_total_pairs"]].values
    ap = frame["average_precision"].to_numpy()
    np.testing.assert_array_equal(
        compute.p_values(ap, confs, 5000, 1, progress_bar=False, backend="numpy"),
        compute.p_values(ap, confs, 5000, 1, progress_bar=False, backend="numba"),
    )


def perfect_retrieval_scores():
    """Return two groups of two replicates (one positive each) among 100 profiles."""
    return pd.DataFrame(
        {
            "g": ["a", "a", "b", "b"],
            "average_precision": [1.0, 1.0, 0.5, 0.5],
            "normalized_average_precision": 0.0,
            "n_pos_pairs": 1,
            "n_total_pairs": 100,
        }
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_float32_scores_count_ties(backend):
    """float32-rounded scores still tie with the null atoms they round from.

    With one positive among 3, AP is 1, 1/2 or 1/3, so every null value is
    ``>= 1/3`` and the p-value of an AP or mAP of 1/3 is exactly 1, although
    float32(1/3) is above 1/3. pandas 2 averages these float32 APs to float32
    and pandas 3 to float64; ties hold under both, and the reported mAP keeps
    the APs' dtype.
    """
    kwargs = dict(null_size=1000, seed=0, progress_bar=False, backend=backend)
    for dtype in (np.float64, np.float32):
        p = compute.p_values(np.array([1 / 3], dtype=dtype), [[1, 3]], **kwargs)
        assert p[0] == 1.0
    scores = pd.DataFrame(
        {
            "g": ["a", "a"],
            "average_precision": np.float32([1 / 3, 1 / 3]),
            "normalized_average_precision": 0.0,
            "n_pos_pairs": 1,
            "n_total_pairs": 3,
        }
    )
    fast = mean_average_precision(
        scores, ["g"], threshold=0.05, null="copied", **kwargs
    )
    assert fast["p_value"].iloc[0] == 1.0
    assert fast["mean_average_precision"].dtype == np.float32


@pytest.mark.parametrize("backend", BACKENDS)
def test_max_workers_sets_numpy_threads(backend, monkeypatch):
    """max_workers sets the NumPy sampler's threads; p-values do not depend on it.

    Numba runs on its own threads, so its calls open no NumPy thread pool.
    """
    monkeypatch.setattr(sampler, "_numpy_workers", lambda: 16)
    pools = []

    def pool(n):
        pools.append(n)
        return ThreadPoolExecutor(n)

    monkeypatch.setattr(sampler, "ThreadPoolExecutor", pool)
    frame = random_ap_frame(np.random.default_rng(5))
    scores = frame["average_precision"].to_numpy()
    confs = frame[["n_pos_pairs", "n_total_pairs"]].to_numpy()
    kwargs = dict(null_size=2000, seed=0, progress_bar=False, backend=backend)
    runs = {
        "ap": lambda w: compute.p_values(scores, confs, max_workers=w, **kwargs),
        "map": lambda w: mean_average_precision(
            frame, ["g"], threshold=0.05, max_workers=w, null="copied", **kwargs
        )["p_value"].to_numpy(),
    }
    for name, run in runs.items():
        got = {}
        for workers in (None, 1, 4):
            pools.clear()
            got[workers] = run(workers)
            if backend == "numba":
                assert pools == [], name
            else:  # one worker samples without a pool
                assert max(pools, default=1) == (workers or 16), (name, pools)
        for workers in (1, 4):
            assert got[workers].tobytes() == got[None].tobytes(), (name, workers)


@pytest.mark.parametrize("max_workers", [0, -1, 1.5])
def test_max_workers_validated(max_workers):
    """The fast method rejects worker counts ThreadPoolExecutor would."""
    with pytest.raises(ValueError, match="max_workers"):
        compute.p_values([0.5], [[2, 10]], 10, 0, max_workers=max_workers)
    with pytest.raises(ValueError, match="max_workers"):
        nulls.map_pvalues(
            [0.5], [0, 1], [0], [1], CONFS[:1], 10, 0, max_workers=max_workers
        )


def test_configurations_without_positives():
    """Configurations with no positives get NaN nulls and p = 1.

    Callers such as the paper's simulations pass every ranked profile's
    configuration, controls (0 positives) included.
    """
    confs = np.array([[0, 10], [2, 10], [0, 0], [3, 3]])
    kwargs = dict(seed=0, progress_bar=False)
    got = compute.get_null_dists(confs, 50, **kwargs)
    assert got.dtype == np.float32
    np.testing.assert_array_equal(np.isnan(got).all(axis=1), [True, False, True, False])
    np.testing.assert_array_equal(
        got[[1, 3]], compute.get_null_dists(confs[[1, 3]], 50, **kwargs)
    )
    scores = np.array([0.3, 0.5, 0.0, 1.0])
    p = compute.p_values(scores, confs, 50, **kwargs)
    assert p[0] == p[2] == 1.0 and p[3] == 1.0 and p[1] < 1
    only_controls = compute.p_values(scores[[0, 2]], confs[[0, 2]], 50, **kwargs)
    np.testing.assert_array_equal(only_controls, [1.0, 1.0])


def empty_ap_frame():
    """AP scores of no profiles, with the usual columns."""
    return pd.DataFrame(
        {
            "g": pd.Series([], dtype=str),
            "h": pd.Series([], dtype=str),
            "average_precision": pd.Series([], dtype=float),
            "normalized_average_precision": pd.Series([], dtype=float),
            "n_pos_pairs": pd.Series([], dtype=int),
            "n_total_pairs": pd.Series([], dtype=int),
        }
    )


@pytest.mark.parametrize(
    "confs", [[2, 10], [[1, 5, 1], [2, 7, 3]], [[1, 5, 1, 9], [2, 7, 3, 9]], [[[1, 5]]]]
)
def test_null_dists_need_conf_rows(confs):
    """Configurations that are not (n, 2) rows raise rather than being reshaped."""
    with pytest.raises(ValueError, match=r"shape \(n, 2\)"):
        compute.get_null_dists(confs, 10, 0)


def test_null_dists_of_no_confs():
    """An empty list of configurations gives no nulls."""
    assert compute.get_null_dists([], 10, 0, progress_bar=False).shape == (0, 10)


@pytest.mark.parametrize(
    "bad,match",
    [
        (dict(seed=-1), "seed must be"),
        (dict(seed=2**64), r"seed must be an integer in \[0, 2\*\*64\)"),
        (dict(null_size=-1), "null_size"),
    ],
)
def test_empty_input_still_validates(bad, match):
    """Seeds outside [0, 2**64) and bad null sizes raise, with nothing to rank too.

    The copied null samples Philox streams, so it shares the joint and
    permutation nulls' seed range.
    """
    kwargs = dict(null_size=100, seed=0, progress_bar=False) | bad
    for confs in (np.empty((0, 2), dtype=int), np.array([[0, 10]])):
        with pytest.raises(ValueError, match=match):
            compute.p_values(np.zeros(len(confs)), confs, **kwargs)
    for frame in (empty_ap_frame(), perfect_retrieval_scores()):
        with pytest.raises(ValueError, match=match):
            mean_average_precision(
                frame, ["g"], threshold=0.05, null="copied", **kwargs
            )


@pytest.mark.parametrize("null", ["copied", "joint", "permutation"])
def test_unknown_backend(null):
    """A misspelled backend raises in every entry point, under every mAP null."""
    scores = np.full(len(CONFS), 0.5)
    with pytest.raises(ValueError, match="unknown backend"):
        compute.p_values(scores, CONFS, 10, 0, backend="nmuba")
    with pytest.raises(ValueError, match="unknown backend"):
        compute.get_null_dists(CONFS, 10, 0, backend="nmuba")
    with pytest.raises(ValueError, match="unknown backend"):
        frame_p_values(perfect_retrieval_scores(), 10, 0, backend="nmuba")
    with pytest.raises(ValueError, match="unknown backend"):
        mean_average_precision(
            perfect_retrieval_scores(), ["g"], 10, 0.05, 0, null=null, backend="x"
        )


@pytest.mark.skipif("numba" in BACKENDS, reason="numba is installed")
def test_numba_backend_without_numba_names_the_extra():
    """backend="numba" without numba raises an ImportError naming the extra."""
    with pytest.raises(ImportError, match=r"copairs\[numba\]"):
        compute.p_values(np.array([0.5]), [[2, 10]], 10, 0, backend="numba")
    with pytest.raises(ImportError, match=r"copairs\[numba\]"):
        compute.p_values(np.array([]), np.empty((0, 2)), 10, 0, backend="numba")
    with pytest.raises(ImportError, match=r"copairs\[numba\]"):
        mean_average_precision(
            perfect_retrieval_scores(),
            ["g"],
            10,
            0.05,
            0,
            null="copied",
            backend="numba",
        )
    with pytest.raises(ImportError, match=r"copairs\[numba\]"):
        mean_average_precision(
            empty_ap_frame(), ["g"], 10, 0.05, 0, null="copied", backend="numba"
        )
