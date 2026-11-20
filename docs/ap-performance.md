# AP performance

NumPy is the default backend for AP. To use the Numba backend, install the
optional extra.

```bash
pip install 'copairs[numba]'
```

Pass `backend="numba"` when calling `map.average_precision`,
`map.multilabel.average_precision`, `matching.find_pairs_multilabel`, or
`map.normalization.normalize_ap`. Each function loads Numba on request.

```python
ap = copairs.map.average_precision(meta, feats, ..., backend="numba")
```

## AP computation

The Numba backend has pair-similarity kernels for `cosine`, `abs_cosine`,
`correlation`, `euclidean`, `manhattan`, and `chebyshev`. Other distances and
callables use NumPy. Euclidean, Manhattan, and Chebyshev also use NumPy for
integer features outside float64's exact integer range.

The backend computes AP by counting how many sorted positives precede each
negative. It normalizes AP using a table of harmonic numbers. For multilabel
AP, it finds pairs with an inverted label index when possible and falls back
to SQL for unsupported inputs. It then computes AP per label in one kernel.
Use `copairs.fastap.draw_average_precisions` to score batched
query-versus-reference draws.

With the same input similarities, AP agrees with the NumPy backend to float64
rounding and normalized AP agrees bit for bit. The similarity kernels compute
in float64 and round their output to float32. Near ties can rank differently.
Float32 feature rows whose norms would overflow in float32 retain their
similarity in the kernel.

The Numba AP stage attaches a function that recomputes pair similarities
with the same kernels for [permutation mAP](map-nulls.md#permutation). It uses
the serial kernel on the null's pool threads, with identical similarities, and
keeps the kernel's feature copy for the duration of the null calculation.
See [AP null sampling](ap-null-sampling.md) for acceleration of p-values.

## Thread settings

In a container or a job with a CPU quota, set `NUMBA_NUM_THREADS` to the number
of CPUs available to the process. With GNU OpenMP, a forked child cannot run
Numba parallel kernels once the parent has initialized them. Use the `spawn`
or `forkserver` start method, or set `NUMBA_THREADING_LAYER=workqueue` before the
first copairs call. Copairs serializes its parallel Numba kernel launches for
concurrent workqueue calls.
