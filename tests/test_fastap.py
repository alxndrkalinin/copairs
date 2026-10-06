"""Tests for fast pair similarities and counting-based average precision."""

import numpy as np
import pytest

from copairs import nulls, fastap, compute
from copairs.map import average_precision
from tests.helpers import simulate_random_dframe
from copairs.map.average_precision import build_rank_lists

BACKENDS = [b for b in nulls.available_backends() if b != "numpy"]


def random_pairs(rng, n_profiles, n_pos, n_neg):
    """Random positive and negative pairs over ``n_profiles`` profiles."""
    pos = rng.integers(n_profiles, size=(n_pos, 2))
    neg = rng.integers(n_profiles, size=(n_neg, 2))
    return pos, neg


def legacy_ap(pos, neg, pos_sims, neg_sims):
    """AP from copairs' rank lists."""
    paired_ix, rel_k_list, counts = build_rank_lists(pos, neg, pos_sims, neg_sims)
    ap_scores, null_confs = compute.ap_contiguous(rel_k_list, counts)
    return paired_ix, ap_scores, null_confs


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("ties", [False, True])
def test_ap_from_pairs_matches_rank_lists(backend, ties):
    """Counting AP equals rank-list AP, including tied similarities."""
    rng = np.random.default_rng(0)
    pos, neg = random_pairs(rng, 300, 900, 20_000)
    pos_sims = rng.random(len(pos)).astype(np.float32)
    neg_sims = rng.random(len(neg)).astype(np.float32)
    if ties:
        pos_sims = np.round(pos_sims * 8) / 8
        neg_sims = np.round(neg_sims * 8) / 8
    expected = legacy_ap(pos, neg, pos_sims, neg_sims)
    got = fastap.ap_from_pairs(pos, neg, pos_sims, neg_sims, backend=backend)
    np.testing.assert_array_equal(got[0], expected[0])
    np.testing.assert_array_equal(got[2], expected[2])
    np.testing.assert_allclose(got[1], expected[1], rtol=1e-12, atol=0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_ap_from_pairs_edge_cases(backend):
    """Profiles without positives get NaN; NaN similarities rank last."""
    rng = np.random.default_rng(1)
    pos, neg = random_pairs(rng, 50, 40, 500)
    pos_sims = rng.random(len(pos)).astype(np.float32)
    neg_sims = rng.random(len(neg)).astype(np.float32)
    pos_sims[::7] = np.nan
    neg_sims[::5] = np.nan
    expected = legacy_ap(pos, neg, pos_sims, neg_sims)
    got = fastap.ap_from_pairs(pos, neg, pos_sims, neg_sims, backend=backend)
    np.testing.assert_array_equal(got[0], expected[0])
    np.testing.assert_array_equal(np.isnan(got[1]), np.isnan(expected[1]))
    assert np.isnan(got[1]).any()
    np.testing.assert_allclose(got[1], expected[1], rtol=1e-12, atol=0)


METRICS = list(fastap.FAST_METRICS)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_pair_similarity_matches_generic(backend, metric, dtype):
    """Kernel similarities match copairs' generic functions to float32 rounding."""
    rng = np.random.default_rng(2)
    feats = rng.normal(size=(200, 37)).astype(dtype)
    pairs = rng.integers(200, size=(3000, 2))
    generic = compute.get_similarity_fn(metric, progress_bar=False)(feats, pairs, 512)
    got = fastap.pair_similarity(feats, metric, backend)(pairs)
    assert got.dtype == np.float32
    np.testing.assert_allclose(got, generic, rtol=1e-5, atol=1e-6)


def test_pair_similarity_without_kernel():
    """Metrics without a kernel and callables fall back to the generic path."""
    feats = np.ones((3, 2))
    assert fastap.pair_similarity(feats, "jaccard", "numba") is None
    assert fastap.pair_similarity(feats, lambda x, y: x[:, 0], "numba") is None


def simulated_pipeline_input(seed=0):
    """Metadata and features with replicated labels."""
    rng = np.random.default_rng(seed)
    vocab_size = {"p": 6, "w": 8, "l": 30}
    meta = simulate_random_dframe(400, vocab_size, ["l"], ["p"], rng)
    feats = rng.normal(size=(len(meta), 16))
    return meta, feats


PIPELINE = dict(pos_sameby=["l"], pos_diffby=["p"], neg_sameby=[], neg_diffby=["l"])


@pytest.mark.parametrize("backend", BACKENDS)
def test_average_precision_fast_ap_stage(backend):
    """With the same similarities, fast and legacy pipelines agree to rounding."""
    meta, feats = simulated_pipeline_input()

    def cosine(x, y):
        return compute.pairwise_cosine(x, y)

    kwargs = dict(PIPELINE, distance=cosine, progress_bar=False)
    fast = average_precision(meta, feats, backend=backend, **kwargs)
    legacy = average_precision(meta, feats, method="legacy", **kwargs)
    for col in ["n_pos_pairs", "n_total_pairs"]:
        np.testing.assert_array_equal(fast[col], legacy[col])
    for col in ["average_precision", "normalized_average_precision"]:
        np.testing.assert_allclose(fast[col], legacy[col], rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("backend", BACKENDS + ["numpy"])
def test_average_precision_fast_similarity(backend):
    """Fast kernels give the legacy APs up to float32 near-ties."""
    meta, feats = simulated_pipeline_input(1)
    kwargs = dict(PIPELINE, progress_bar=False)
    fast = average_precision(meta, feats, backend=backend, **kwargs)
    legacy = average_precision(meta, feats, method="legacy", **kwargs)
    np.testing.assert_array_equal(fast["n_total_pairs"], legacy["n_total_pairs"])
    diff = np.abs(fast["average_precision"] - legacy["average_precision"])
    assert (diff.dropna() < 1e-12).mean() > 0.99
    np.testing.assert_allclose(
        fast["average_precision"], legacy["average_precision"], atol=0.05
    )
