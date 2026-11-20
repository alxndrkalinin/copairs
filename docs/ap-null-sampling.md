# AP null sampling

AP p-values and [copied mAP p-values](map-nulls.md#copied) use the v0.6.0 sampler by
default (`sampler="legacy"`). Seeded draws keep the v0.6.0 random stream,
tie handling, and disk cache. Use `cache_dir` on the functions that expose
it to select the cache location. Calls with `seed=None` draw fresh samples
without writing a cache.

## Stream AP nulls

Pass `sampler="streamed"` to opt into exact random sampling from the AP
null with Philox and Vitter's Algorithm A. The AP and copied mAP p-value
functions process draws in chunks without a disk cache. A sample depends on its
configuration, draw index, and seed. The new seed stream and dtype-aware
tie handling can change p-values obtained with a fixed seed. Streamed
seeds are integers in `[0, 2**64)` or `None`.

```python
from copairs.map import mean_average_precision
from copairs.map.average_precision import p_values

ap_pvalues = p_values(ap, null_size=10_000, seed=0, sampler="streamed")
copied_map = mean_average_precision(
    ap, ["compound"], null_size=10_000, threshold=0.05,
    null="copied", sampler="streamed",
)
```

## Choose a backend

For streamed sampling, `backend="auto"` selects Numba when it imports and
NumPy otherwise. Both engines return the same draws and p-values. Install
the optional accelerator with `pip install 'copairs[numba]'`, or choose an
engine explicitly with `backend="numba"` or `backend="numpy"`. The streamed
NumPy path can be slower than the legacy cache, especially for a warm cache
or a small number of draws with many positives. Where exposed,
`max_workers` controls the streamed NumPy sampler. Use `NUMBA_NUM_THREADS`
to set the Numba thread count.
