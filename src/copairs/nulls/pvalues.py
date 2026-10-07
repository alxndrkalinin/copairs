"""Permutation p-values against the fast AP null, streamed in sample chunks.

Nulls are regenerated chunk by chunk instead of being stored, so memory is
bounded by ``budget_bytes`` whatever ``null_size`` and the number of
configurations. Null samples are kept in float64, the precision copairs
computes observed APs in, and a null value counts against an observed score
when ``null >= score - TIE_TOL``. The tolerance absorbs summation-order
rounding between the two computations, so a null rank list identical to the
observed one is counted as a tie, as in the paper's ``>=`` definition.
"""

import numba
import numpy as np

from copairs.nulls.sampler import _ap_nulls, null_plan, resolve_backend

# Far above float64 summation-order differences of an AP (~k * 2**-52). Distinct
# AP values can lie closer than this in larger configurations (e.g. 9e-13 apart
# for 2 positives among 5000), so a null value just below a score may count as
# a tie; each such value has null probability ~1/C(total, num_pos).
TIE_TOL = 1e-10
DEFAULT_BUDGET = 2**30


def resolve_seed(seed: int | None) -> int:
    """Return ``seed``, or fresh OS entropy when it is ``None``."""
    if seed is None:
        return int(np.random.SeedSequence().entropy % 2**64)
    seed = int(seed)
    if not 0 <= seed < 2**64:
        raise ValueError(f"seed must be in [0, 2**64), got {seed}")
    return seed


def _check_null_size(null_size: int) -> int:
    if int(null_size) != null_size or null_size < 0:
        raise ValueError(f"null_size must be a non-negative integer, got {null_size}")
    return int(null_size)


def _chunk_size(null_size: int, rows: int, budget_bytes: int) -> int:
    """Return the samples per chunk that fit ``rows`` float64 nulls in the budget."""
    return max(1, min(null_size, budget_bytes // (8 * max(rows, 1))))


def _chunks(null_size: int, size: int):
    """``(start, size)`` sample chunks, generated lazily."""
    return ((s, min(size, null_size - s)) for s in range(0, null_size, size))


def _progress(items, total: int, enabled: bool, desc: str):
    if not enabled or total < 2:
        return items
    from tqdm.autonotebook import tqdm

    return tqdm(items, total=total, desc=desc, leave=False)


def _count_ge_host(null, thr, ptr, counts):
    """``counts[p] += #{x in null[c] : x >= thr[p]}`` for each segment ``c``."""
    for c in range(null.shape[0]):
        lo, hi = ptr[c], ptr[c + 1]
        if hi == lo:
            continue
        q = np.searchsorted(thr[lo:hi], null[c], side="right")
        hist = np.bincount(q, minlength=hi - lo + 1)
        counts[lo:hi] += np.cumsum(hist[::-1])[::-1][1:]


@numba.njit(inline="always")
def _upper_bound(vals, lo, hi, key):
    """``lo`` plus #{vals[lo:hi] <= key} for sorted ``vals`` (NaN sorts last)."""
    while lo < hi:
        mid = (lo + hi) >> 1
        v = vals[mid]
        if not np.isnan(v) and v <= key:
            lo = mid + 1
        else:
            hi = mid
    return lo


# Upper bound on the per-block histograms of _count_ge_numba.
_HIST_BYTES = 2**28


def _count_ge_numba(null, thr, ptr, counts):
    """:func:`_count_ge_host` in parallel over configurations and sample blocks."""
    bins = int(ptr[-1] - ptr[0]) + null.shape[0]
    max_blocks = min(4 * numba.get_num_threads(), max(1, _HIST_BYTES // (8 * bins)))
    _count_ge_kernel(np.ascontiguousarray(null), thr, ptr, counts, max_blocks)


@numba.njit(parallel=True, cache=True)
def _count_ge_kernel(null, thr, ptr, counts, max_blocks):
    n_conf, size = null.shape
    n_blocks = max(1, min((size + (1 << 16) - 1) >> 16, max_blocks))
    block = (size + n_blocks - 1) // n_blocks
    base = ptr[0]
    partial = np.zeros((n_blocks, ptr[n_conf] - base + n_conf), dtype=np.int64)
    for task in numba.prange(n_conf * n_blocks):
        c = task // n_blocks
        b = task - c * n_blocks
        lo, hi = ptr[c], ptr[c + 1]
        if hi == lo:
            continue
        off = c - base
        for t in range(b * block, min(size, (b + 1) * block)):
            partial[b, off + _upper_bound(thr, lo, hi, null[c, t])] += 1
    for c in numba.prange(n_conf):
        lo, hi = ptr[c], ptr[c + 1]
        off = lo - base + c
        run = 0
        for q in range(hi - lo - 1, -1, -1):
            for b in range(n_blocks):
                run += partial[b, off + q + 1]
            counts[lo + q] += run


def _count_ge_device(null, thr, ptr, counts):
    from copairs.nulls import cuda

    cuda.count_ge(null, thr, ptr, counts)


def ap_pvalues(
    scores: np.ndarray,
    conf_ix: np.ndarray,
    confs: np.ndarray,
    null_size: int,
    seed: int | None,
    backend: str = "auto",
    budget_bytes: int = DEFAULT_BUDGET,
    progress_bar: bool = False,
) -> np.ndarray:
    """P-value of each AP score against the null of its configuration.

    ``p_i = (1 + #{null >= scores[i] - TIE_TOL}) / (null_size + 1)``, where the
    null is that of ``confs[conf_ix[i]]``.

    Parameters
    ----------
    scores : np.ndarray
        Observed AP scores.
    conf_ix : np.ndarray
        Row of ``confs`` holding each score's ``(num_pos, total)`` configuration.
    confs : np.ndarray
        ``(n_conf, 2)`` configurations.
    null_size, seed, backend
        See :func:`copairs.nulls.sampler.ap_nulls`.
    budget_bytes : int
        Upper bound on the null samples held in memory at once.
    progress_bar : bool
        Show a progress bar over chunks.
    """
    scores = np.asarray(scores, dtype=np.float64)
    conf_ix = np.asarray(conf_ix, dtype=np.int64)
    confs = np.asarray(confs)
    if len(conf_ix) != len(scores):
        raise ValueError(f"{len(scores)} scores but {len(conf_ix)} conf_ix")
    if len(conf_ix) and (conf_ix.min() < 0 or conf_ix.max() >= len(confs)):
        raise ValueError(f"conf_ix must index the {len(confs)} rows of confs")
    null_size = _check_null_size(null_size)
    seed = resolve_seed(seed)
    backend = resolve_backend(backend)
    plan = null_plan(confs, seed)
    if backend == "cuda":  # upload the configurations once
        import cupy as cp

        plan = tuple(cp.asarray(a) for a in plan)
    order = np.lexsort((scores, conf_ix))
    thr = scores[order] - TIE_TOL
    ptr = np.searchsorted(conf_ix[order], np.arange(len(confs) + 1))
    # Process configurations in batches whose chunk of samples fits the budget.
    chunk = _chunk_size(null_size, 1, budget_bytes)
    batch = max(1, budget_bytes // (8 * chunk))
    batches = range(0, len(confs), batch)
    work = (
        (b, start, size) for b in batches for start, size in _chunks(null_size, chunk)
    )
    n_work = len(batches) * -(-null_size // chunk)
    if backend == "cuda":
        import cupy as cp

        xp, count = cp, _count_ge_device
    elif backend == "numba":
        xp, count = np, _count_ge_numba
    else:
        xp, count = np, _count_ge_host
    thr_x, counts = xp.asarray(thr), xp.zeros(len(scores), dtype=np.int64)
    for b, start, size in _progress(work, n_work, progress_bar, "AP null"):
        sl = slice(b, min(b + batch, len(confs)))
        part = tuple(a[sl] for a in plan)
        null = _ap_nulls(part, size, start, backend, np.float64)
        count(null, thr_x, ptr[sl.start : sl.stop + 1], counts)
    counts = counts.get() if backend == "cuda" else counts
    pvals = np.empty(len(scores), dtype=np.float64)
    pvals[order] = (counts + 1) / (null_size + 1)
    return pvals


def _group_ge_host(null, ptr, conf_ix, conf_cnt, n_group, thr, counts):
    for g in range(len(ptr) - 1):
        acc = np.zeros(null.shape[1])
        for e in range(ptr[g], ptr[g + 1]):
            acc += conf_cnt[e] * null[conf_ix[e]]
        counts[g] += np.count_nonzero(acc / n_group[g] >= thr[g])


@numba.njit(parallel=True, cache=True)
def _group_ge_numba(null, ptr, conf_ix, conf_cnt, n_group, thr, counts):
    n_groups = len(ptr) - 1
    size = null.shape[1]
    block = 1 << 14
    n_blocks = (size + block - 1) // block
    partial = np.zeros(n_groups * n_blocks, dtype=np.int64)
    for flat in numba.prange(n_groups * n_blocks):
        g = flat // n_blocks
        t0 = (flat - g * n_blocks) * block
        t1 = min(t0 + block, size)
        hits = 0
        for t in range(t0, t1):
            acc = 0.0
            for e in range(ptr[g], ptr[g + 1]):
                acc += conf_cnt[e] * null[conf_ix[e], t]
            if acc / n_group[g] >= thr[g]:
                hits += 1
        partial[flat] = hits
    for g in range(n_groups):
        counts[g] += partial[g * n_blocks : (g + 1) * n_blocks].sum()


def map_pvalues(
    map_scores: np.ndarray,
    ptr: np.ndarray,
    conf_ix: np.ndarray,
    conf_cnt: np.ndarray,
    confs: np.ndarray,
    null_size: int,
    seed: int | None,
    backend: str = "auto",
    budget_bytes: int = DEFAULT_BUDGET,
    progress_bar: bool = False,
) -> np.ndarray:
    """P-value of each group's mAP against the mean of its members' AP nulls.

    Group ``g`` has ``conf_cnt[e]`` members with configuration
    ``confs[conf_ix[e]]`` for ``e`` in ``ptr[g]:ptr[g + 1]``. Members sharing a
    configuration share null samples, as in copairs' original implementation,
    so the group null at sample ``j`` is
    ``sum_e conf_cnt[e] * AP_{conf_ix[e]}(j) / sum_e conf_cnt[e]`` and
    ``p_g = (1 + #{null_g >= map_scores[g] - TIE_TOL}) / (null_size + 1)``.
    """
    map_scores = np.asarray(map_scores, dtype=np.float64)
    ptr = np.asarray(ptr, dtype=np.int64)
    conf_ix = np.asarray(conf_ix, dtype=np.int64)
    conf_cnt = np.asarray(conf_cnt, dtype=np.int64)
    confs = np.asarray(confs)
    null_size = _check_null_size(null_size)
    seed = resolve_seed(seed)
    backend = resolve_backend(backend)
    plan = null_plan(confs, seed)
    if backend == "cuda":  # upload the configurations once
        import cupy as cp

        plan = tuple(cp.asarray(a) for a in plan)
    if (
        len(ptr) < 1
        or ptr[0] != 0
        or ptr[-1] != len(conf_ix)
        or (np.diff(ptr) < 1).any()
    ):
        raise ValueError(
            "ptr must start at 0, end at len(conf_ix) and give every group a member"
        )
    n_group = np.add.reduceat(conf_cnt, ptr[:-1]) if len(conf_cnt) else conf_cnt
    if len(map_scores) == 0:
        return np.zeros(0, dtype=np.float64)
    thr = map_scores - TIE_TOL
    if backend == "cuda":
        from copairs.nulls import cuda

        group = cuda.GroupCounter(ptr, conf_ix, conf_cnt, n_group, thr)
    elif backend == "numba":
        counts = np.zeros(len(map_scores), dtype=np.int64)
        args = (ptr, conf_ix, conf_cnt, n_group, thr, counts)

        def group(null):
            _group_ge_numba(null, *args)
    else:
        counts = np.zeros(len(map_scores), dtype=np.int64)

        def group(null):
            _group_ge_host(null, ptr, conf_ix, conf_cnt, n_group, thr, counts)

    chunk = _chunk_size(null_size, len(confs), budget_bytes)
    n_work = -(-null_size // chunk)
    for start, size in _progress(
        _chunks(null_size, chunk), n_work, progress_bar, "mAP null"
    ):
        group(_ap_nulls(plan, size, start, backend, np.float64))
    if backend == "cuda":
        counts = group.counts()
    return (counts + 1) / (null_size + 1)
