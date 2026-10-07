"""Average precision from pair lists without building rank lists.

In copairs' rank list of profile ``i``, pairs are ordered by the float32 key
``1 - similarity``, and a positive ties ahead of a negative with the same key.
The ``t``-th positive (1-based, by key) therefore sits at rank
``t + #{negatives with key < key_t}``, so AP needs only each profile's sorted
positive keys and, for every negative, a binary search into them: no global
sort of all pairs.
"""

import numba
import numpy as np


@numba.njit(inline="always")
def _upper_bound(vals, lo, hi, key):
    """First index in ``vals[lo:hi]`` ordered after ``key`` (NumPy order, NaN last)."""
    key_nan = np.isnan(key)
    while lo < hi:
        mid = (lo + hi) >> 1
        v = vals[mid]
        if key_nan or (not np.isnan(v) and v <= key):
            lo = mid + 1
        else:
            hi = mid
    return lo


def rank_keys(sims: np.ndarray) -> np.ndarray:
    """Ranking keys of similarities, as computed by ``build_rank_lists``."""
    return np.float32(1) - np.asarray(sims, dtype=np.float32)


@numba.njit(cache=True)
def _positive_csr(pairs, keys, n):
    """Positive keys of each profile, sorted, in CSR layout."""
    ptr = np.zeros(n + 1, dtype=np.int64)
    for p in range(len(pairs)):
        ptr[pairs[p, 0] + 1] += 1
        ptr[pairs[p, 1] + 1] += 1
    for i in range(n):
        ptr[i + 1] += ptr[i]
    fill = ptr[:-1].copy()
    vals = np.empty(2 * len(pairs), dtype=keys.dtype)
    for p in range(len(pairs)):
        for side in range(2):
            i = pairs[p, side]
            vals[fill[i]] = keys[p]
            fill[i] += 1
    for i in range(n):
        vals[ptr[i] : ptr[i + 1]] = np.sort(vals[ptr[i] : ptr[i + 1]])
    return ptr, vals


@numba.njit(parallel=True, cache=True)
def _negative_hist(neg_pairs, neg_keys, ptr, vals, n, n_chunks):
    """Per-profile histograms of negatives over the gaps between positive keys.

    Bin ``ptr[i] + i + q`` counts the negatives of profile ``i`` whose key lies
    between its ``q``-th and ``q + 1``-th smallest positive keys (ties go after
    the positive). Each chunk of pairs fills its own histogram row.
    """
    size = len(neg_pairs)
    hist = np.zeros((n_chunks, ptr[n] + n), dtype=np.int64)
    n_neg = np.zeros((n_chunks, n), dtype=np.int64)
    step = (size + n_chunks - 1) // n_chunks
    for c in numba.prange(n_chunks):
        for p in range(c * step, min(size, (c + 1) * step)):
            key = neg_keys[p]
            for side in range(2):
                i = neg_pairs[p, side]
                hist[c, _upper_bound(vals, ptr[i], ptr[i + 1], key) + i] += 1
                n_neg[c, i] += 1
    return hist.sum(axis=0), n_neg.sum(axis=0)


@numba.njit(parallel=True, cache=True)
def _ap_from_hist(ptr, hist, n):
    ap = np.empty(n, dtype=np.float64)
    for i in numba.prange(n):
        num_pos = ptr[i + 1] - ptr[i]
        if num_pos == 0:
            ap[i] = np.nan
            continue
        base = ptr[i] + i
        before = 0
        acc = 0.0
        for t in range(num_pos):
            before += hist[base + t]
            acc += (t + 1) / (t + 1 + before)
        ap[i] = acc / num_pos
    return ap


def ap_from_pairs(
    pos_pairs: np.ndarray,
    neg_pairs: np.ndarray,
    pos_sims: np.ndarray,
    neg_sims: np.ndarray,
    backend: str = "numba",
    budget_bytes: int = 2**30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average precision of every profile that appears in a pair.

    Parameters
    ----------
    pos_pairs, neg_pairs : np.ndarray
        ``(n, 2)`` profile indices of positive and negative pairs.
    pos_sims, neg_sims : np.ndarray
        Similarities of the pairs.
    backend : str
        ``"numba"`` or ``"cuda"``.
    budget_bytes : int
        Upper bound on the per-thread negative histograms (Numba).

    Returns
    -------
    paired_ix : np.ndarray
        Sorted indices of profiles in at least one pair.
    ap_scores : np.ndarray
        float64 AP of each profile in ``paired_ix`` (NaN without positives).
    null_confs : np.ndarray
        ``(n, 2)`` uint32 ``(num_pos, total)`` of each profile in ``paired_ix``.
    """
    pos_pairs = np.ascontiguousarray(pos_pairs, dtype=np.int64).reshape(-1, 2)
    neg_pairs = np.ascontiguousarray(neg_pairs, dtype=np.int64).reshape(-1, 2)
    n = int(max(pos_pairs.max(initial=-1), neg_pairs.max(initial=-1))) + 1
    if backend == "cuda" and n >= 2**32:
        backend = "numba"  # the GPU sort packs profile indices into 32 bits
    if backend == "cuda":
        from copairs.fastap import cuda

        ap, num_pos, n_neg = cuda.ap_from_pairs(
            pos_pairs, neg_pairs, rank_keys(pos_sims), rank_keys(neg_sims), n
        )
    elif backend == "numba":
        ptr, vals = _positive_csr(pos_pairs, rank_keys(pos_sims), n)
        bins = int(ptr[n]) + n
        n_chunks = budget_bytes // (8 * (bins + n))
        n_chunks = max(1, min(numba.get_num_threads(), n_chunks))
        hist, n_neg = _negative_hist(
            neg_pairs, rank_keys(neg_sims), ptr, vals, n, n_chunks
        )
        ap = _ap_from_hist(ptr, hist, n)
        num_pos = np.diff(ptr)
    else:
        raise ValueError(f"unknown backend {backend!r}; expected numba or cuda")
    total = num_pos + n_neg
    paired_ix = np.flatnonzero(total)
    null_confs = np.stack([num_pos[paired_ix], total[paired_ix]], axis=1)
    return paired_ix, ap[paired_ix], null_confs.astype(np.uint32)
