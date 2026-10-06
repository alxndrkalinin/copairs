"""Fast pair similarities and counting-based average precision.

Backends are ``"cuda"`` (CuPy) and ``"numba"``; ``"auto"`` picks CUDA when a
GPU is usable. AP values match copairs' rank-list implementation to float64
rounding given the same similarities.
"""

import numpy as np

from copairs.fastap.ranking import ap_from_pairs
from copairs.fastap.similarity import FAST_METRICS, PairSimilarity

__all__ = [
    "FAST_METRICS",
    "PairSimilarity",
    "ap_from_pairs",
    "pair_similarity",
    "resolve_backend",
]


def resolve_backend(backend: str) -> str:
    """Concrete backend for the AP stage: ``"cuda"``, ``"numba"`` or ``"numpy"``."""
    from copairs.nulls import resolve_backend as resolve

    return resolve(backend)


def pair_similarity(feats: np.ndarray, distance, backend: str):
    """Return a ``pairs -> float32 similarities`` function, or None without a kernel."""
    if not isinstance(distance, str) or distance not in FAST_METRICS:
        return None
    sim = PairSimilarity(feats, distance)
    if backend != "cuda":
        return sim
    from copairs.fastap import cuda

    return cuda.PairSimilarity(sim.x, distance)
