"""The pairs the Numba and CUDA AP stages attach for the permutation null."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("numba", reason="Numba optional extra is unavailable")

from copairs import compute
from copairs.map import multilabel, average_precision, mean_average_precision
from tests.helpers import shared_labels, random_targets
from tests.markers import AP_BACKENDS, needs_gpu
from copairs.fastap import similarity
from copairs._backend import backend_module
from copairs.map.average_precision import attached_pairs, _FeatureSimilarity

NULL = dict(null_size=2000, threshold=0.05, seed=3, progress_bar=False)
NULLS = ["permutation", "joint", "copied"]


def plate_meta(groups=8, replicates=4, plates=3):
    """Return ``groups`` treatments with ``replicates`` profiles on each of ``plates``."""
    names = [f"g{i}" for i in range(groups)]
    return pd.DataFrame(
        {
            "g": np.tile(np.repeat(names, replicates), plates),
            "plate": np.repeat(np.arange(plates), groups * replicates),
        }
    )


def label_meta(n, labels=5, seed=0):
    """Return ``n`` profiles with some of ``labels`` targets each, plus one of their own."""
    has = np.random.default_rng(seed).random((n, labels)) < 0.3
    return pd.DataFrame(
        {
            "target": random_targets(has),
            "plate": np.arange(n) % 3,
        }
    )


def plate_ap(feats, backend, **kwargs):
    """APs of :func:`plate_meta`'s treatments, ranked against the same plate."""
    return average_precision(
        plate_meta(), feats, ["g"], [], ["plate"], ["g"], progress_bar=False,
        backend=backend, **kwargs,
    )  # fmt: skip


def label_ap(feats, backend, **kwargs):
    """Multilabel APs of :func:`label_meta`'s targets, ranked against the same plate."""
    return multilabel.average_precision(
        label_meta(len(feats)), feats, ["target"], [], ["plate"], ["target"],
        multilabel_col="target", progress_bar=False, backend=backend, **kwargs,
    )  # fmt: skip


def near_ties(seed, n, step, d=64):
    """``n`` float32 profiles with near-tied similarities.

    In each triple ``i, j = i + step, k = i + 2 * step``, profile ``k``
    mirrors ``j`` through a hyperplane holding ``i``, so both have the same
    similarity to ``i`` in exact arithmetic. The float32 NumPy functions
    and the float64 kernels round those two differently, and some of them
    rank in another order.
    """
    rng = np.random.default_rng(seed)
    feats = rng.standard_normal((n, d))
    for i in [i for i in range(n - 2 * step) if i // step % 3 == 0]:
        j, k = i + step, i + 2 * step
        q = feats[i] / np.linalg.norm(feats[i])
        u = rng.standard_normal(d)
        u -= (u @ q) * q
        u /= np.linalg.norm(u)
        feats[k] = feats[j] - 2 * (feats[j] @ u) * u
    return feats.astype(np.float32)


@pytest.mark.parametrize("backend", AP_BACKENDS)
@pytest.mark.parametrize("distance", ["cosine", "euclidean"])
def test_pairs_recompute_the_ap_stage_similarities(backend, distance):
    """The attached pairs give the AP stage's kernel similarities, bit for bit.

    The permutation null re-ranks profiles by them. NumPy's similarities of
    float32 rows, which it computes in float32, differ in rounding.
    """
    feats = np.random.default_rng(1).standard_normal((96, 24)).astype(np.float32)
    kernel = backend_module(backend).pair_similarity(feats, distance)
    numpy = compute.pairwise_similarity_fn(distance)
    _, rank_pairs = plate_ap(feats, backend, distance=distance, return_pairs=True)
    _, label_pairs = label_ap(feats, backend, distance=distance, return_pairs=True)
    # A label table's positives are pairs of its rows, its negatives of profiles.
    for pairs in (rank_pairs["pos"], rank_pairs["neg"], label_pairs["neg"]):
        assert not np.array_equal(
            numpy(feats[pairs[:, 0]], feats[pairs[:, 1]]), kernel(pairs)
        )
        got = rank_pairs["similarity"](pairs)
        np.testing.assert_array_equal(
            got.view(np.uint32), kernel(pairs).view(np.uint32)
        )
        got = label_pairs["similarity"](pairs)
        np.testing.assert_array_equal(
            got.view(np.uint32), kernel(pairs).view(np.uint32)
        )


def test_distances_without_kernel_recompute_with_numpy():
    """A Numba AP stage's pairs use NumPy where the stage did: no kernel for the distance."""
    feats = np.random.default_rng(1).standard_normal((96, 24)).astype(np.float32)
    _, pairs = plate_ap(feats, "numba", distance="cityblock", return_pairs=True)
    assert pairs["similarity"].kernel_backend is None
    numpy = compute.get_similarity_fn("cityblock", progress_bar=False)
    np.testing.assert_array_equal(
        pairs["similarity"](pairs["neg"]), numpy(feats, pairs["neg"], 20000)
    )


@pytest.mark.parametrize("multi", [False, True], ids=["single", "multilabel"])
def test_permutation_null_reproduces_numba_aps_of_near_ties(multi):
    """The null re-ranks a Numba AP stage's near-tied pairs as the stage did.

    Recomputed by NumPy, the near ties of :func:`near_ties` rank in another
    order for some profiles, the null's mAP of the observed labels differs
    from the AP table's, and its check raises.
    """
    # Triples on one plate: label_meta's plates alternate, plate_meta's are blocks.
    ap_fn, sameby, rows, step = (label_ap, "target", shared_labels, 3) if multi else (
        plate_ap, "g", lambda ap: ap, 1
    )  # fmt: skip
    for seed in range(20):
        feats = near_ties(seed, 96, step)
        aps = {b: ap_fn(feats, b)["average_precision"] for b in ("numpy", "numba")}
        if not np.allclose(*aps.values(), rtol=0, atol=1e-9, equal_nan=True):
            break
    else:
        pytest.fail("no seed gave near ties that the backends rank differently")
    ap = rows(ap_fn(feats, "numba"))
    for null in NULLS:
        p = mean_average_precision(ap, [sameby], null=null, **NULL)["p_value"]
        assert ((p > 0) & (p <= 1)).all()


@pytest.mark.parametrize("backend", AP_BACKENDS)
def test_permutation_null_ranks_by_the_kernel_similarities(backend):
    """Rows whose norms overflow float32 keep the kernels' similarities in the null.

    NumPy computes their cosines in float32, where every norm overflows and
    every similarity comes out 0; re-ranked by those, the null would
    reproduce no AP and raise.
    """
    rng = np.random.default_rng(2)
    feats = (
        rng.standard_normal((96, 8))
        + np.repeat(rng.standard_normal((8, 8)), 4, 0)[np.tile(np.arange(32), 3)]
    )  # a shift per treatment of plate_meta
    feats = (feats * 1e20).astype(np.float32)
    ap = plate_ap(feats, backend)
    got = mean_average_precision(ap, ["g"], backend="numba", **NULL)
    assert 0 < got["below_p"].sum() < len(got)
    labels = shared_labels(label_ap(feats, backend))
    got = mean_average_precision(labels, ["target"], backend="numba", **NULL)
    assert ((got["p_value"] > 0) & (got["p_value"] <= 1)).all()


@pytest.mark.parametrize("backend", AP_BACKENDS)
@pytest.mark.parametrize("null", NULLS)
def test_kernel_ap_stages_give_numpy_pvalues(backend, null):
    """p-values after a kernel AP stage equal those after NumPy's, under every null.

    Integer features make every euclidean similarity exact in both, so the
    AP tables and the pairs (in the multilabel table, each AP row's label
    and profile; the label index orders positive pairs within a label
    otherwise than SQL) must agree for the p-values to.
    """
    base = np.random.default_rng(4).integers(-3, 4, size=(96, 6)).astype(np.float64)
    plate_signal = plate_meta()["g"].isin(["g0", "g1"]).to_numpy()
    label_signal = np.array(["L0" in t for t in label_meta(96)["target"]])
    for ap_fn, sameby, signal in (
        (plate_ap, "g", plate_signal),
        (label_ap, "target", label_signal),
    ):
        feats = base.copy()
        feats[signal, :2] += 4
        got = []
        for b in (backend, "numpy"):
            ap = ap_fn(feats, b, distance="euclidean")
            if sameby == "target":
                ap = shared_labels(ap)
            m = mean_average_precision(ap, [sameby], null=null, backend="numba", **NULL)
            got.append(m.sort_values(sameby).reset_index(drop=True))
        pd.testing.assert_frame_equal(*got)
        assert got[0]["below_p"].any() and got[0]["p_value"].nunique() > 1


@pytest.mark.parametrize("null_backend", ["numba", "numpy"])
def test_null_threads_compute_similarities_serially(null_backend, monkeypatch):
    """The permutation null's threads run a Numba AP stage's kernel serially.

    A parallel launch from a pool thread would run on that thread's Numba
    threads (the count is per thread, so not the caller's), under the NumPy
    null too, whose draws run no Numba, and many small launches from
    several threads cost more than they gain.
    """
    feats = np.random.default_rng(5).standard_normal((96, 8))
    frames = (
        (plate_ap(feats, "numba"), "g"),
        (shared_labels(label_ap(feats, "numba")), "target"),
    )
    sims, parallel = [], []
    call = _FeatureSimilarity.__call__

    def counted(self, pairs):
        sims.append(len(pairs))
        return call(self, pairs)

    monkeypatch.setattr(_FeatureSimilarity, "__call__", counted)
    for name in ("_dot_pairs", "_minkowski_pairs"):
        kernel = getattr(similarity, name)

        def recorded(*args, kernel=kernel, name=name):
            parallel.append(name)
            return kernel(*args)

        monkeypatch.setattr(similarity, name, recorded)
    for ap, sameby in frames:
        mean_average_precision(ap, [sameby], backend=null_backend, **NULL)
    assert sims and not parallel


@needs_gpu
def test_pairs_check_the_gpu_before_every_kernel_call(monkeypatch):
    """The CUDA kernel, once built, still runs the GPU check before each call.

    The check is the fork guard: a child forked after its parent's null
    built the kernel raises its RuntimeError instead of launching on CUDA
    it cannot use.
    """
    feats = np.random.default_rng(6).standard_normal((96, 8)).astype(np.float32)
    _, pairs = plate_ap(feats, "cuda", return_pairs=True)
    pairs["similarity"](pairs["neg"][:4])  # builds the kernel

    def forked():
        raise RuntimeError("forked")

    monkeypatch.setattr("copairs._cuda.require", forked)  # imports CuPy
    with pytest.raises(RuntimeError, match="forked"):
        pairs["similarity"](pairs["neg"][:4])


@pytest.mark.parametrize("backend", AP_BACKENDS)
def test_null_releases_the_kernel_it_built(backend, monkeypatch):
    """Each permutation null drops the AP stage's kernel when it returns.

    The kernel holds its own copy of the features (on the GPU under
    "cuda"), which the frame's pairs would otherwise keep; the next null
    builds it again.
    """
    feats = np.random.default_rng(7).standard_normal((96, 8)).astype(np.float32)
    frames = (
        (plate_ap(feats, backend), "g"),
        (shared_labels(label_ap(feats, backend)), "target"),
    )
    kernels = backend_module(backend)
    builds = []
    build = kernels.pair_similarity

    def counted(*args, **kwargs):
        builds.append(args[1])
        return build(*args, **kwargs)

    monkeypatch.setattr(kernels, "pair_similarity", counted)
    for ap, sameby in frames:
        similarity = attached_pairs(ap)["similarity"]
        for _ in range(2):
            mean_average_precision(ap, [sameby], backend="numba", **NULL)
            assert similarity._kernel is None
    assert builds == ["cosine"] * 4
