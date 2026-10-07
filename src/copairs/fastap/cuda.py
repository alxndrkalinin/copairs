"""CUDA (CuPy) kernels for pair similarities and counting-based AP."""

import numpy as np

from copairs.nulls import cuda as _null_cuda

cp = _null_cuda.cp
_BLOCK = _null_cuda._BLOCK
_grid = _null_cuda.grid

_SOURCE = r"""
// numpy ordering: NaN sorts after every number and ties with NaN.
__device__ __forceinline__ bool le(float v, float key) {
    return isnan(key) || (!isnan(v) && v <= key);
}

template <typename T>
__device__ void dot_pairs(const T* x, long long d, const long long* pairs, long long n,
                          double* out) {
    int lane = threadIdx.x & 31;
    long long warp = (blockIdx.x * (long long)blockDim.x + threadIdx.x) >> 5;
    long long n_warps = ((long long)gridDim.x * blockDim.x) >> 5;
    for (long long p = warp; p < n; p += n_warps) {
        const T* a = x + pairs[2 * p] * d;
        const T* b = x + pairs[2 * p + 1] * d;
        double acc = 0.0;
        for (long long t = lane; t < d; t += 32) acc += (double)a[t] * (double)b[t];
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
        if (lane == 0) out[p] = acc;
    }
}

// 1 / (1 + distance); kind 0 = euclidean, 1 = manhattan, 2 = chebyshev.
template <typename T>
__device__ void minkowski_pairs(const T* x, long long d, const long long* pairs, long long n,
                                int kind, double* out) {
    int lane = threadIdx.x & 31;
    long long warp = (blockIdx.x * (long long)blockDim.x + threadIdx.x) >> 5;
    long long n_warps = ((long long)gridDim.x * blockDim.x) >> 5;
    for (long long p = warp; p < n; p += n_warps) {
        const T* a = x + pairs[2 * p] * d;
        const T* b = x + pairs[2 * p + 1] * d;
        double acc = 0.0;
        for (long long t = lane; t < d; t += 32) {
            double diff = fabs((double)a[t] - (double)b[t]);
            acc = kind == 0 ? acc + diff * diff : kind == 1 ? acc + diff : fmax(acc, diff);
        }
        for (int off = 16; off > 0; off >>= 1) {
            double other = __shfl_down_sync(0xffffffffu, acc, off);
            acc = kind == 2 ? fmax(acc, other) : acc + other;
        }
        if (lane == 0) out[p] = 1.0 / (1.0 + (kind == 0 ? sqrt(acc) : acc));
    }
}

extern "C" __global__ void dot_pairs_f32(const float* x, long long d, const long long* pairs,
                                         long long n, double* out) {
    dot_pairs<float>(x, d, pairs, n, out);
}
extern "C" __global__ void dot_pairs_f64(const double* x, long long d, const long long* pairs,
                                         long long n, double* out) {
    dot_pairs<double>(x, d, pairs, n, out);
}
extern "C" __global__ void minkowski_pairs_f32(const float* x, long long d,
                                               const long long* pairs, long long n, int kind,
                                               double* out) {
    minkowski_pairs<float>(x, d, pairs, n, kind, out);
}
extern "C" __global__ void minkowski_pairs_f64(const double* x, long long d,
                                               const long long* pairs, long long n, int kind,
                                               double* out) {
    minkowski_pairs<double>(x, d, pairs, n, kind, out);
}

// For both endpoints i of each negative pair, count it in bin ptr[i] + i + q, where
// q = #{positive keys of i <= key}.
extern "C" __global__ void negative_hist(const long long* neg_pairs, const float* keys,
                                         long long n_neg, const long long* ptr,
                                         const float* vals, unsigned long long* hist,
                                         unsigned long long* n_negative) {
    for (long long e = blockIdx.x * (long long)blockDim.x + threadIdx.x; e < 2 * n_neg;
         e += (long long)gridDim.x * blockDim.x) {
        long long i = neg_pairs[e];
        float key = keys[e >> 1];
        long long lo = ptr[i], hi = ptr[i + 1];
        while (lo < hi) {
            long long mid = (lo + hi) >> 1;
            if (le(vals[mid], key)) lo = mid + 1; else hi = mid;
        }
        atomicAdd(hist + lo + i, 1ull);
        atomicAdd(n_negative + i, 1ull);
    }
}

extern "C" __global__ void ap_from_hist(const long long* ptr, const unsigned long long* hist,
                                        long long n, double* ap) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n;
         i += (long long)gridDim.x * blockDim.x) {
        long long num_pos = ptr[i + 1] - ptr[i], base = ptr[i] + i;
        if (num_pos == 0) { ap[i] = nan(""); continue; }
        unsigned long long before = 0;
        double acc = 0.0;
        for (long long t = 0; t < num_pos; ++t) {
            before += hist[base + t];
            acc += (double)(t + 1) / (double)(t + 1 + before);
        }
        ap[i] = acc / (double)num_pos;
    }
}
"""


def _kernel(name: str):
    return _null_cuda.kernel(_SOURCE, name)


class PairSimilarity:
    """GPU counterpart of :class:`copairs.fastap.similarity.PairSimilarity`."""

    def __init__(self, x: np.ndarray, metric: str):
        self.metric = metric
        self.x = cp.asarray(x)
        self.suffix = "f64" if self.x.dtype == cp.float64 else "f32"

    def __call__(self, pairs, as_numpy: bool = True):
        """float32 similarity of each ``(i, j)`` row of ``pairs``."""
        pairs = cp.ascontiguousarray(cp.asarray(pairs, dtype=cp.int64).reshape(-1, 2))
        n, d = len(pairs), self.x.shape[1]
        out = cp.empty(n, dtype=cp.float64)
        if n:
            grid = (_grid(32 * n),)
            if self.metric in ("cosine", "abs_cosine", "correlation"):
                args = (self.x, np.int64(d), pairs, np.int64(n), out)
                _kernel(f"dot_pairs_{self.suffix}")(grid, (_BLOCK,), args)
                if self.metric == "abs_cosine":
                    cp.abs(out, out=out)
            else:
                kind = np.int32(
                    ("euclidean", "manhattan", "chebyshev").index(self.metric)
                )
                args = (self.x, np.int64(d), pairs, np.int64(n), kind, out)
                _kernel(f"minkowski_pairs_{self.suffix}")(grid, (_BLOCK,), args)
        out = out.astype(cp.float32)
        return out.get() if as_numpy else out


def _sortable(keys):
    """uint64 that orders like float32 ``keys`` in NumPy (NaN last)."""
    keys = cp.where(cp.isnan(keys), cp.float32(np.nan), keys).astype(cp.float32)
    bits = keys.view(cp.uint32)
    flipped = cp.where(bits >> 31 == 1, ~bits, bits | cp.uint32(0x80000000))
    return flipped.astype(cp.uint64)


def ap_from_pairs(pos_pairs, neg_pairs, pos_keys, neg_keys, n: int):
    """``(ap, num_pos, num_neg)`` of profiles ``0..n-1``, computed on the GPU."""
    pos_pairs = cp.asarray(pos_pairs, dtype=cp.int64).reshape(-1, 2)
    neg_pairs = cp.ascontiguousarray(
        cp.asarray(neg_pairs, dtype=cp.int64).reshape(-1, 2)
    )
    neg_keys = cp.asarray(neg_keys, dtype=cp.float32)
    profile = pos_pairs.ravel()
    keys = cp.repeat(cp.asarray(pos_keys, dtype=cp.float32), 2)
    order = cp.argsort((profile.astype(cp.uint64) << 32) | _sortable(keys))
    vals = cp.ascontiguousarray(keys[order])
    ptr = cp.searchsorted(profile[order], cp.arange(n + 1, dtype=cp.int64))
    hist = cp.zeros(int(ptr[-1]) + n, dtype=cp.uint64)
    n_neg = cp.zeros(n, dtype=cp.uint64)
    if len(neg_pairs):
        args = (neg_pairs, neg_keys, np.int64(len(neg_pairs)), ptr, vals, hist, n_neg)
        _kernel("negative_hist")((_grid(2 * len(neg_pairs)),), (_BLOCK,), args)
    ap = cp.empty(n, dtype=cp.float64)
    _kernel("ap_from_hist")((_grid(n),), (_BLOCK,), (ptr, hist, np.int64(n), ap))
    return ap.get(), cp.diff(ptr).get(), n_neg.get().astype(np.int64)


_DRAW_SOURCE = r"""
// AP of each (draw, query) row of sims (n_rows, k + m); the first k columns are the
// draw's queries (the row's own column is skipped), the rest its references.
extern "C" __global__ void draw_ap(const float* sims, int k, int m, long long n_rows,
                                   double* ap) {
    extern __shared__ float smem[];
    float* pos = smem;                                   // k - 1, descending
    unsigned int* hist = (unsigned int*)(smem + k);      // k bins
    for (long long row = blockIdx.x; row < n_rows; row += gridDim.x) {
        const float* s = sims + row * (long long)(k + m);
        int q = (int)(row % k);
        for (int j = threadIdx.x; j < k; j += blockDim.x) hist[j] = 0u;
        for (int j = threadIdx.x; j < k; j += blockDim.x) {
            if (j == q) continue;
            float v = s[j];
            int r = 0;  // rank in descending order, ties by column
            for (int i = 0; i < k; ++i) {
                if (i == q) continue;
                float w = s[i];
                r += (w > v) || (w == v && i < j);
            }
            pos[r] = v;
        }
        __syncthreads();
        for (int e = threadIdx.x; e < m; e += blockDim.x) {
            float v = s[k + e];
            int lo = 0, hi = k - 1;  // first t with pos[t] < v
            while (lo < hi) {
                int mid = (lo + hi) >> 1;
                if (pos[mid] >= v) lo = mid + 1; else hi = mid;
            }
            atomicAdd(hist + lo, 1u);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            unsigned long long before = 0;
            double acc = 0.0;
            for (int t = 0; t < k - 1; ++t) {
                before += hist[t];
                acc += (double)(t + 1) / (double)(t + 1 + before);
            }
            ap[row] = acc / (double)(k - 1);
        }
        __syncthreads();
    }
}
"""


def draw_average_precisions(feats, queries, references, normalized, budget_bytes):
    """CUDA backend of :func:`copairs.fastap.draws.draw_average_precisions`."""
    from copairs.fastap.draws import _chunk, unit_rows

    x = (
        cp.asarray(feats, dtype=cp.float32)
        if normalized
        else unit_rows(cp.asarray(feats))
    )
    idx = cp.asarray(np.concatenate([queries, references], axis=1))
    n_draws, k = queries.shape
    m = references.shape[1]
    out = cp.empty((n_draws, k), dtype=cp.float64)
    kernel = _null_cuda.kernel(_DRAW_SOURCE, "draw_ap")
    threads = 128 if m >= 128 else 32
    step = _chunk(n_draws, k, m, x.shape[1], budget_bytes)
    for start in range(0, n_draws, step):
        rows = x[idx[start : start + step]]  # (b, k + m, d)
        sims = cp.ascontiguousarray(cp.matmul(rows[:, :k], rows.transpose(0, 2, 1)))
        n_rows = sims.shape[0] * k
        grid = (int(min(n_rows, 65535 * 4)),)
        args = (
            sims,
            np.int32(k),
            np.int32(m),
            np.int64(n_rows),
            out[start : start + step],
        )
        kernel(grid, (threads,), args, shared_mem=8 * k)
    return out.get()
