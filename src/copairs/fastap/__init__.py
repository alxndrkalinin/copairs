"""Fast pair similarities and counting-based average precision.

Backends are ``"cuda"`` (CuPy) and ``"numba"``; ``"auto"`` picks CUDA when a
GPU is usable. AP values match copairs' rank-list implementation to float64
rounding given the same similarities. :func:`draw_average_precisions` scores
many query-vs-reference draws at once.
"""

import numpy as np

from copairs import compute
from copairs.nulls import resolve_backend
from copairs.fastap.draws import draw_average_precisions
from copairs.fastap.ranking import ap_from_pairs
from copairs.fastap.similarity import FAST_METRICS, PairSimilarity

__all__ = [
    "draw_average_precisions",
    "FAST_METRICS",
    "PairSimilarity",
    "ap_from_pairs",
    "pair_similarity",
    "resolve_backend",
    "setup",
]


def setup(
    method: str,
    backend: str,
    feats,
    distance,
    progress_bar: bool,
    on_device: bool = False,
):
    """Resolve the AP stage's method and backend and pick its similarity function.

    ``backend="numpy"`` selects the legacy NumPy implementation. The returned
    function has ``compute.get_similarity_fn``'s ``(feats, pairs, batch_size)``
    signature and uses a kernel when ``distance`` has one; with ``on_device``,
    a CUDA kernel's similarities stay on the GPU for :func:`ap_from_pairs`.

    Returns
    -------
    tuple
        ``(method, backend, similarity_fn)``.
    """
    compute._check_method(method)
    similarity_fn = compute.get_similarity_fn(distance, progress_bar=progress_bar)
    if method == "legacy":
        return method, backend, similarity_fn
    backend = resolve_backend(backend)
    if backend == "numpy":
        return "legacy", backend, similarity_fn
    kernel = pair_similarity(np.asarray(feats), distance, backend)
    if kernel is not None:
        keep = {"as_numpy": False} if on_device and backend == "cuda" else {}

        def similarity_fn(feats, pairs, batch_size):
            return kernel(pairs, **keep)

    return method, backend, similarity_fn


def pair_similarity(feats: np.ndarray, distance, backend: str):
    """Return a ``pairs -> float32 similarities`` function, or None without a kernel."""
    if not isinstance(distance, str) or distance not in FAST_METRICS:
        return None
    sim = PairSimilarity(feats, distance)
    if backend != "cuda":
        return sim
    from copairs.fastap import cuda

    return cuda.PairSimilarity(sim.x, distance)
