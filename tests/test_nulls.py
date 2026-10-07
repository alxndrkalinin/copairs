"""Tests for the fast AP null sampler and streamed p-values."""

import itertools

import numpy as np
import pytest
from scipy.stats import chisquare

from copairs import nulls
from copairs.nulls import philox, pvalues, sampler
from tests.helpers import brute_ap_pvalues
from copairs.map.normalization import expected_ap

BACKENDS = nulls.available_backends()
CPU_BACKENDS = [b for b in BACKENDS if b != "cuda"]
CONFS = np.array(
    [[1, 1], [1, 7], [2, 2], [3, 10], [5, 20], [9, 49], [40, 45], [99, 1099], [3, 1000]]
)


def to_numpy(x):
    """Return a NumPy array for NumPy or CuPy input."""
    return x.get() if hasattr(x, "get") else np.asarray(x)


def exact_null(num_pos, total):
    """Every AP value of a (num_pos, total) rank list, one per placement of positives."""
    ranks = np.array(list(itertools.combinations(range(1, total + 1), num_pos)))
    return (np.arange(1, num_pos + 1) / ranks).sum(axis=1) / num_pos


@pytest.mark.parametrize(
    "ctr,key,expected",
    [
        ((0, 0, 0, 0), (0, 0), (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)),
        (
            (0xFFFFFFFF,) * 4,
            (0xFFFFFFFF,) * 2,
            (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD),
        ),
        (
            (0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344),
            (0xA4093822, 0x299F31D0),
            (0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1),
        ),
    ],
)
def test_philox_known_answers(ctr, key, expected):
    """Philox4x32-10 matches the Random123 known-answer vectors."""
    words = philox.philox4x32(*map(np.uint64, ctr), *map(np.uint64, key))
    assert [int(w) for w in words] == list(expected)


@pytest.mark.parametrize("backend", BACKENDS)
def test_backends_bitwise_identical(backend):
    """Every backend returns the scalar reference's float32 values bit for bit."""
    k0, k1 = sampler.config_keys(CONFS, 3)
    ref = np.array(
        [
            [
                np.float32(sampler._ap_sample(int(p), int(t), j, k0[c], k1[c]))
                for j in range(64)
            ]
            for c, (p, t) in enumerate(CONFS)
        ]
    )
    out = to_numpy(nulls.ap_nulls(CONFS, 64, seed=3, backend=backend))
    np.testing.assert_array_equal(out.view(np.uint32), ref.view(np.uint32))


@pytest.mark.parametrize("backend", BACKENDS)
def test_chunks_and_prefixes(backend):
    """Sample j depends only on (seed, num_pos, total, j)."""
    whole = to_numpy(nulls.ap_nulls(CONFS, 3000, seed=11, backend=backend))
    parts = [
        to_numpy(nulls.ap_nulls(CONFS, size, seed=11, start=start, backend=backend))
        for start, size in [(0, 1000), (1000, 1500), (2500, 500)]
    ]
    np.testing.assert_array_equal(np.concatenate(parts, axis=1), whole)
    alone = to_numpy(nulls.ap_nulls(CONFS[[4]], 100, seed=11, backend=backend))
    np.testing.assert_array_equal(alone[0], whole[4, :100])
    other_seed = to_numpy(nulls.ap_nulls(CONFS, 100, seed=12, backend=backend))
    assert not np.array_equal(other_seed, whole[:, :100])


def test_float64_rounds_to_float32():
    """float32 output is the float64 sample rounded once."""
    f64 = nulls.ap_nulls(CONFS, 500, seed=5, dtype=np.float64)
    f32 = nulls.ap_nulls(CONFS, 500, seed=5)
    np.testing.assert_array_equal(f64.astype(np.float32), f32)


@pytest.mark.parametrize("num_pos,total", [(1, 9), (3, 10), (5, 20), (2, 60)])
def test_matches_exact_distribution(num_pos, total):
    """Sample frequencies of each AP value match exact enumeration."""
    exact = exact_null(num_pos, total)
    atoms, weights = np.unique(np.round(exact, 9), return_counts=True)
    null = nulls.ap_nulls([[num_pos, total]], 200_000, seed=1, dtype=np.float64)[0]
    observed = np.bincount(
        np.searchsorted(atoms, np.round(null, 9)), minlength=len(atoms)
    )
    assert observed.sum() == len(null)
    expected = weights / weights.sum() * len(null)
    big = expected >= 5
    obs = np.append(observed[big], observed[~big].sum())
    exp = np.append(expected[big], expected[~big].sum())
    if exp[-1] == 0:
        obs, exp = obs[:-1], exp[:-1]
    assert chisquare(obs, exp).pvalue > 1e-3
    # The upper tail, which decides small p-values, in particular.
    q = np.quantile(exact, 0.99)
    tail = np.mean(exact >= q)
    se = np.sqrt(tail * (1 - tail) / len(null))
    assert abs(np.mean(null >= q - pvalues.TIE_TOL) - tail) < 5 * se


@pytest.mark.parametrize("num_pos,total", [(10, 100), (99, 1099), (3, 5000)])
def test_mean_matches_expected_ap(num_pos, total):
    """The null mean matches the closed-form E[AP] within Monte Carlo error."""
    null = nulls.ap_nulls([[num_pos, total]], 100_000, seed=2, dtype=np.float64)[0]
    se = null.std() / np.sqrt(len(null))
    assert abs(null.mean() - expected_ap(num_pos, total - num_pos)) < 5 * se


def test_degenerate_configurations():
    """All-positive lists have AP 1; a single positive has AP 1/rank, rank uniform."""
    null = nulls.ap_nulls([[4, 4], [1, 1]], 100, seed=0)
    assert (null == 1).all()
    ap = nulls.ap_nulls([[1, 5]], 50_000, seed=0, dtype=np.float64)[0]
    ranks = np.rint(1 / ap).astype(int)
    np.testing.assert_allclose(1 / ranks, ap)
    freq = np.bincount(ranks, minlength=6)[1:] / len(ap)
    np.testing.assert_allclose(freq, 0.2, atol=0.01)


def test_invalid_configurations():
    """Configurations without positives, or with more positives than ranks, raise."""
    for confs in ([[0, 5]], [[6, 5]], [1, 5]):
        with pytest.raises(ValueError):
            nulls.ap_nulls(confs, 10, seed=0)
    with pytest.raises(ValueError):
        nulls.ap_nulls(CONFS, 10, seed=-1)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("budget", [pvalues.DEFAULT_BUDGET, 8 * 700])
def test_ap_pvalues_match_brute_force(backend, budget):
    """Streamed AP p-values equal counting over the materialized null."""
    rng = np.random.default_rng(0)
    conf_ix = rng.integers(len(CONFS), size=300)
    scores = rng.random(300)
    # Include exact ties with null values.
    null = nulls.ap_nulls(CONFS, 50, seed=4, dtype=np.float64)
    scores[:50] = null[conf_ix[:50], np.arange(50)]
    got = nulls.ap_pvalues(
        scores, conf_ix, CONFS, 2000, seed=4, backend=backend, budget_bytes=budget
    )
    np.testing.assert_array_equal(
        got, brute_ap_pvalues(scores, conf_ix, CONFS, 2000, 4)
    )


def random_groups(rng, n_groups):
    """CSR (group -> configuration counts) with mixed and repeated configurations."""
    ptr, conf_ix, conf_cnt = [0], [], []
    for _ in range(n_groups):
        confs = rng.choice(len(CONFS), size=rng.integers(1, 4), replace=False)
        conf_ix.extend(sorted(confs))
        conf_cnt.extend(rng.integers(1, 5, size=len(confs)))
        ptr.append(len(conf_ix))
    return np.array(ptr), np.array(conf_ix), np.array(conf_cnt)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("budget", [pvalues.DEFAULT_BUDGET, 8 * len(CONFS) * 300])
def test_map_pvalues_match_brute_force(backend, budget):
    """Streamed mAP p-values equal counting over materialized group nulls."""
    rng = np.random.default_rng(1)
    ptr, conf_ix, conf_cnt = random_groups(rng, 40)
    null = nulls.ap_nulls(CONFS, 1500, seed=9, dtype=np.float64)
    group_null = np.empty((40, 1500))
    for g in range(40):
        acc = np.zeros(1500)
        for e in range(ptr[g], ptr[g + 1]):
            acc += conf_cnt[e] * null[conf_ix[e]]
        group_null[g] = acc / conf_cnt[ptr[g] : ptr[g + 1]].sum()
    map_scores = rng.random(40) * 0.6
    map_scores[:10] = group_null[:10, 7]  # exact ties
    num = (group_null >= map_scores[:, None] - pvalues.TIE_TOL).sum(axis=1)
    got = nulls.map_pvalues(
        map_scores,
        ptr,
        conf_ix,
        conf_cnt,
        CONFS,
        1500,
        seed=9,
        backend=backend,
        budget_bytes=budget,
    )
    np.testing.assert_array_equal(got, (num + 1) / 1501)


def test_backend_selection():
    """Unknown or unavailable backends raise; auto picks the first available."""
    assert nulls.resolve_backend("auto") == BACKENDS[0]
    with pytest.raises(ValueError):
        nulls.resolve_backend("tpu")


@pytest.mark.parametrize("backend", [b for b in BACKENDS if b != "numpy"])
@pytest.mark.parametrize(
    "num_pos,total", [(2, 5000), (3, 3000), (8, 20000), (4, 70000)]
)
def test_guided_gap_search_matches_loop(backend, num_pos, total):
    """The guided gap search reproduces Algorithm A's gaps bit for bit."""
    assert total - num_pos > sampler.GUIDED_RATIO * num_pos * (num_pos + 1)
    ref = nulls.ap_nulls([[num_pos, total]], 400, seed=8, backend="numpy")
    got = nulls.ap_nulls([[num_pos, total]], 400, seed=8, backend=backend)
    np.testing.assert_array_equal(got.view(np.uint32), ref.view(np.uint32))
    rng = np.random.default_rng(num_pos)
    for u in np.concatenate([rng.random(200), [1e-300, 0.5, 1 - 2**-53]]):
        k = int(rng.integers(2, num_pos + 1))
        assert sampler._gap_guided(total, k, u) == sampler.gap_loop(total, k, u)
