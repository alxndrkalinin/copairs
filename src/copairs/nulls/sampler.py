"""Exact Monte Carlo samples of the average-precision null distribution.

Under the null, the ``num_pos`` positives of a rank list of length ``total``
occupy a uniformly random subset of the ranks. Instead of shuffling a
``null_size x total`` matrix, each sample places the positives one at a time:
the number of negatives before the next positive is drawn by inverting its
survival function (Vitter 1984, "Faster methods for random sampling",
Algorithm A), which needs one uniform per positive and no memory beyond a few
scalars.

Sample ``j`` of configuration ``(num_pos, total)`` is a pure function of
``(seed, num_pos, total, j)`` through a Philox counter, and every backend
performs the same IEEE double operations in the same order, so the NumPy,
Numba and CUDA implementations return bitwise-identical nulls.
"""

import math

import numpy as np

from copairs.nulls.philox import M32, S32, uniform53, config_key, philox4x32

try:
    import numba
except ImportError:  # pragma: no cover - exercised only without numba
    numba = None


def _make_ap_sample(philox, uniform):
    """Build the scalar sampler around ``philox`` and ``uniform`` (Python or Numba)."""

    def ap_sample(num_pos, total, j, k0, k1):
        """Average precision of sample ``j`` of configuration ``(num_pos, total)``."""
        c0 = np.uint64(j) & M32
        c1 = np.uint64(j) >> S32
        remaining = total
        k = num_pos
        rank = 0
        acc = 0.0
        i = 0
        u_next = 0.0
        while k > 0:
            if k == remaining:
                # Every remaining rank holds a positive.
                while k > 0:
                    rank += 1
                    i += 1
                    acc += i / rank
                    k -= 1
                break
            if i % 2 == 0:
                w0, w1, w2, w3 = philox(c0, c1, np.uint64(i // 2), np.uint64(0), k0, k1)
                u = uniform(w0, w1)
                u_next = uniform(w2, w3)
            else:
                u = u_next
            if k == 1:
                gap = int(math.floor(remaining * u))
                if gap > remaining - 1:
                    gap = remaining - 1
            else:
                # Smallest gap g with P(G > g) <= u; P(G > g) = prod (top - j) / (remaining - j).
                top = remaining - k
                quot = top / remaining
                gap = 0
                while quot > u:
                    gap += 1
                    top -= 1
                    quot = quot * (top / (remaining - gap))
            rank += gap + 1
            i += 1
            acc += i / rank
            remaining -= gap + 1
            k -= 1
        return acc / num_pos

    return ap_sample


_ap_sample = _make_ap_sample(philox4x32, uniform53)


def _ap_null_numpy(num_pos, total, start, size, k0, k1):
    """Vectorized NumPy version of :func:`_ap_sample` over ``size`` samples (float64)."""
    j = np.arange(start, start + size, dtype=np.uint64)
    c0, c1 = j & M32, j >> S32
    k0, k1 = np.uint64(k0), np.uint64(k1)
    remaining = np.full(size, total, dtype=np.int64)
    rank = np.zeros(size, dtype=np.int64)
    acc = np.zeros(size, dtype=np.float64)
    live = np.arange(size)
    for i in range(num_pos):
        k = num_pos - i
        full = remaining[live] == k
        if full.any():
            ix = live[full]
            for step in range(k):
                rank[ix] += 1
                acc[ix] += (i + step + 1) / rank[ix]
            live = live[~full]
        if live.size == 0:
            break
        w = philox4x32(c0[live], c1[live], np.uint64(i // 2), np.uint64(0), k0, k1)
        u = uniform53(w[0], w[1]) if i % 2 == 0 else uniform53(w[2], w[3])
        rem = remaining[live]
        if k == 1:
            gap = np.minimum(np.floor(rem * u).astype(np.int64), rem - 1)
        else:
            top = rem - k
            quot = top / rem
            gap = np.zeros(live.size, dtype=np.int64)
            run = np.flatnonzero(quot > u)
            while run.size:
                gap[run] += 1
                top[run] -= 1
                quot[run] = quot[run] * (top[run] / (rem[run] - gap[run]))
                run = run[quot[run] > u[run]]
        rank[live] += gap + 1
        acc[live] += (i + 1) / rank[live]
        remaining[live] = rem - gap - 1
    return acc / num_pos


if numba is not None:
    _ap_sample_nb = numba.njit(cache=True)(
        _make_ap_sample(
            numba.njit(inline="always")(philox4x32),
            numba.njit(inline="always")(uniform53),
        )
    )

    @numba.njit(parallel=True, cache=True)
    def _ap_nulls_numba(num_pos, total, k0, k1, start, out):
        n_conf, size = out.shape
        for flat in numba.prange(n_conf * size):
            c = flat // size
            t = flat - c * size
            out[c, t] = _ap_sample_nb(num_pos[c], total[c], start + t, k0[c], k1[c])


def available_backends() -> list[str]:
    """Backends usable in this environment, fastest first."""
    backends = ["numpy"]
    if numba is not None:
        backends.insert(0, "numba")
    from copairs.nulls import cuda

    if cuda.is_available():
        backends.insert(0, "cuda")
    return backends


def resolve_backend(backend: str) -> str:
    """Return a concrete backend name for ``backend`` (``"auto"`` picks the fastest)."""
    available = available_backends()
    if backend == "auto":
        return available[0]
    if backend not in ("cuda", "numba", "numpy"):
        raise ValueError(
            f"unknown backend {backend!r}; expected auto, cuda, numba or numpy"
        )
    if backend not in available:
        raise ValueError(f"backend {backend!r} is not available here ({available})")
    return backend


def _validate_confs(confs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    confs = np.asarray(confs)
    if confs.ndim != 2 or confs.shape[1] != 2:
        raise ValueError(f"confs must have shape (n, 2), got {confs.shape}")
    num_pos = confs[:, 0].astype(np.int64)
    total = confs[:, 1].astype(np.int64)
    if (num_pos < 1).any() or (total < num_pos).any():
        raise ValueError("each configuration needs 1 <= num_pos <= total")
    return num_pos, total


def config_keys(confs: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Philox keys ``(k0, k1)`` of each ``(num_pos, total)`` row of ``confs``."""
    num_pos, total = _validate_confs(confs)
    keys = [config_key(seed, int(p), int(t)) for p, t in zip(num_pos, total)]
    keys = np.array(keys, dtype=np.uint64).reshape(-1, 2)
    return keys[:, 0].copy(), keys[:, 1].copy()


def ap_nulls(
    confs: np.ndarray,
    null_size: int,
    seed: int,
    start: int = 0,
    backend: str = "auto",
    dtype=np.float32,
):
    """Sample the AP null of every ``(num_pos, total)`` row of ``confs``.

    Parameters
    ----------
    confs : np.ndarray
        ``(n, 2)`` integer array of ``(num_pos, total)`` configurations.
    null_size : int
        Number of samples per configuration.
    seed : int
        Seed in ``[0, 2**64)``. Sample ``j`` of a configuration depends only on
        ``(seed, num_pos, total, j)``.
    start : int
        Index of the first sample, so a null can be generated in chunks.
    backend : str
        ``"auto"``, ``"cuda"``, ``"numba"`` or ``"numpy"``. All return identical
        values.
    dtype : np.float32 or np.float64
        Output precision. Samples are computed in float64 and rounded once.

    Returns
    -------
    np.ndarray
        ``(n, null_size)`` AP samples.
    """
    out = _ap_nulls(confs, null_size, seed, start, resolve_backend(backend), dtype)
    return out.get() if hasattr(out, "get") else out


def _ap_nulls(confs, null_size, seed, start, backend, dtype):
    """:func:`ap_nulls` for a resolved backend; CUDA results stay on the device."""
    dtype = np.dtype(dtype)
    if dtype not in (np.float32, np.float64):
        raise ValueError(f"dtype must be float32 or float64, got {dtype}")
    num_pos, total = _validate_confs(confs)
    k0, k1 = config_keys(confs, seed)
    if backend == "cuda":
        from copairs.nulls import cuda

        return cuda.ap_nulls(num_pos, total, k0, k1, start, null_size, dtype)
    out = np.empty((len(num_pos), null_size), dtype=dtype)
    if backend == "numba":
        _ap_nulls_numba(num_pos, total, k0, k1, start, out)
    else:
        for c in range(len(num_pos)):
            out[c] = _ap_null_numpy(
                int(num_pos[c]), int(total[c]), start, null_size, k0[c], k1[c]
            )
    return out
