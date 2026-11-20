# GPU acceleration

On Linux or Windows with an NVIDIA GPU and a compatible driver, install
the optional CUDA 12 backend.

```bash
pip install 'copairs[gpu]'
```

The extra installs Numba and the `cupy-cuda12x` wheel. CuPy also needs
CUDA libraries from a CUDA 12 toolkit or from
`pip install 'cupy-cuda12x[ctk]'`. copairs compiles its kernels at runtime
with NVRTC and uses cuBLAS for batched AP draws.

Call `copairs.nulls.available_backends()` to check for `"cuda"`. An explicit
CUDA operation reports missing dependencies or an unavailable device.
Use the multiprocessing `spawn` start method when workers need CUDA after
the parent has initialized it.

## Average precision

Pass `backend="cuda"` to `copairs.map.average_precision` or
`copairs.map.multilabel.average_precision`. The GPU computes supported pair
similarities and single-label AP, streaming negatives in chunks. Numba
handles matching by label, per-label AP and AP normalization. See
[AP performance](ap-performance.md) for supported distance kernels and
what to expect for near-tied similarities.

The batched `copairs.fastap.draw_average_precisions` helper accepts NumPy
features. It uses the GPU for draws whose sort fits in the kernel's shared
memory and uses Numba for larger sorts.

## Null sampling

For AP p-values and the copied mAP null, choose `sampler="streamed"` and
`backend="cuda"`. The GPU runs the Philox sampler and counts exceedances.
See [AP null sampling](ap-null-sampling.md) for reproducibility and cache
behavior. Automatic backend selection considers CPU engines only.

The joint and permutation mAP samplers use Numba when given
`backend="cuda"`. If AP was computed using CUDA similarity kernels, the
permutation null reuses those kernels on the original CUDA device when
recomputing similarities.
