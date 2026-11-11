 # copairs

`copairs` is a Python package for finding groups of profiles based on metadata and calculate mean Average Precision to assess intra- vs inter-group similarities.

## Getting started

### System requirements
copairs supports Python 3.11 and newer (tested through Python 3.14) and should work with all modern operating systems (tested with MacOS 13.5, Ubuntu 18.04, Windows 10).

### Dependencies
copairs depends on widely used Python packages:
* numpy
* pandas
* tqdm
* statsmodels

### Installation

To install copairs and dependencies, run:
```bash
pip install copairs
```

To also install dependencies for running examples, run:
```bash
pip install copairs[demo]
```

#### Null distributions and the optional Numba backend

The AP and mAP p-value functions (`compute.p_values`,
`map.average_precision.p_values`, `mean_average_precision`,
`mean_average_precision_hierarchical` and `get_map_pvalue`) take
`method="fast"` (default) or `"legacy"`, and a `backend`:

- `"fast"` samples each AP null exactly (Philox counter-based RNG, Vitter's
  Algorithm A) in O(1) memory per sample, instead of permuting a
  `null_size x total` matrix, and streams p-values in bounded memory without a
  disk cache. Sample `j` of a `(num_pos, total)` null depends only on
  `(seed, num_pos, total, j)`, with `seed` in `[0, 2**64)`.
- `"legacy"` is copairs 0.5.5's null, seeded per configuration and cached
  under `cache_dir` (default `~/.copairs`). In the mAP functions it is the
  `"copied"` null (see [mAP null distributions](#map-null-distributions)):
  pass `null="copied"` with it; other nulls raise.
- The fast method's null samples come from a different random stream, so
  p-values agree with `"legacy"` within Monte Carlo error. Its mAP p-values
  count null values `>=` the observed mAP, as in the paper; legacy counts `>`,
  which gives too small p-values when the mAP equals an atom of the null
  (e.g. perfect retrieval).
- With the NumPy backend, `"fast"` can be slower than `"legacy"` for
  configurations with many positives at small null sizes (for 1000 positives
  among 100,000 pairs at `null_size=1000`, 3.6 s against 1.6 s; 0.02 s with
  Numba), so install the `numba` extra for such data.

`backend="numpy"` needs nothing extra. `backend="numba"` compiles the same
code with [Numba](https://numba.pydata.org), an optional dependency:
```bash
pip install copairs[numba]
```
In the mAP functions, `backend` applies to each null of the fast method: the
copied null's AP samples, the joint null's samples, and the permutation
null's relabellings (moves of a label, for multilabel APs) and their mAPs.
For each, the Numba backend compiles the scalar functions that define a draw
and the NumPy backend computes the same draws vectorized. The two backends
return bitwise-identical nulls and p-values (tested), so `backend="auto"`
(default) uses Numba whenever it imports; if an installed Numba fails to
import, `"auto"` warns once and uses NumPy. A Numba that imports but cannot
compile a kernel raises instead; the tests pass on Numba 0.60 (the extra's
minimum) and 0.68. The first call of each null under Numba compiles its
kernels (a few seconds per null) and caches the compiled code next to
copairs' sources, where later processes load it in under a second.

Numba starts one thread per host CPU. In a container or job with a CPU quota,
set `NUMBA_NUM_THREADS` to the quota, or kernels oversubscribe the CPUs. The
NumPy backend samples on one thread per CPU the process may run on (its CPU
affinity, which follows `taskset` and Slurm allocations but not a CPU quota),
or on `max_workers` threads where the function takes it. The permutation
null runs groups on `max_workers` threads (by default one per CPU, at most
32, and under Numba one per Numba thread), which share a large group's
draws. Under Numba these threads run the draws themselves when they run as
many groups at once as Numba has threads; otherwise the draws run on Numba's
threads, one group at a time, as many of them as the calling thread has
(`numba.set_num_threads` is per thread). If Numba uses its GNU OpenMP threading
layer (the default when it cannot load TBB), a process forked after copairs
ran a Numba kernel stops when it runs one itself ("Terminating: fork() called
from a process already using GNU OpenMP"). Use the `spawn` or `forkserver`
start method, set `NUMBA_THREADING_LAYER=workqueue` before the first copairs
call, or pass `backend="numpy"` in the children.

### Testing

To run tests, run:
```bash
pip install -e .[test]
pytest
```

## Usage

We provide examples demonstrating how to use copairs for:
- [grouping profiles based on their metadata](https://github.com/cytomining/copairs/blob/main/docs/examples/finding_pairs.ipynb)
- [calculating mAP to assess phenotypic activity of perturbations](https://github.com/cytomining/copairs/blob/main/docs/examples/phenotypic_activity.ipynb)
- [calculating mAP to assess phenotypic consistency of perturbations](https://github.com/cytomining/copairs/blob/main/docs/examples/phenotypic_consistency.ipynb)
- [estimating null size for mAP p-value calculation](https://github.com/cytomining/copairs/blob/main/docs/examples/null_size.ipynb)
- [comparing the mAP null distributions](https://github.com/cytomining/copairs/blob/main/docs/examples/null_comparison.ipynb)

## mAP null distributions

`mean_average_precision`, `mean_average_precision_hierarchical` and
`get_map_pvalue` take `null=`, the null of a group's mAP. Under
`method="fast"` (default) all three count null mAPs `>=` the observed mAP, so
an mAP equal to an atom of the null (e.g. perfect retrieval) counts the atom.
The models make different dependence assumptions; calibration depends on the
experimental design.

- `"permutation"` (default) tests exchangeability. For single-label APs it uses
  reference-defined blocks: members ranked against identical reference sets form a block (e.g. one plate's
  replicates and controls). Its pool contains those members and references;
  each draw preserves the number of group members in every block. Draws relabel profiles and recompute the mAP from the
  observed similarities, so whatever dependence the data has between rank
  lists stays in the null. A relabelled profile is ranked against the
  block's unlabelled profiles, as the members are ranked against their
  references. The original positive and negative rules define the graph and
  pools; neither is reapplied to fixed metadata during a draw. Profiles move
  into roles on the original positive graph, including non-complete graphs.
  Thus a layout rule such as `neg_diffby=["g", "plate"]` or
  `pos_diffby=["plate"]` need not hold on the relabelled profiles.
  All positive edges must stay inside the tested group; for groups of one profile whose
  positives are in other groups (e.g. one profile per guide, positives by
  gene), use `"joint"`, which there is that profile's AP null. It needs the
  pairs the APs came from, which `average_precision` attaches to the frame
  it returns, so the usual calls need nothing extra:

  ```python
  ap = copairs.map.average_precision(meta, feats, ...)
  result = copairs.map.mean_average_precision(
      ap, sameby, null_size=10_000, threshold=0.05, seed=0
  )
  ```

  The pairs (`ap.attrs["copairs_pairs"]`) follow the frame through filtering
  and copies, but not through saving, pickling or concatenating frames of
  different AP calls; frames without pairs require an explicit pair dictionary
  or AP recomputation. `average_precision(..., return_pairs=True)` returns the
  dictionary with `"pos"`, `"neg"` and `"similarity"` entries. Explicit `pairs=`
  takes precedence over attachments; use `pairs["pos"]` for raw joint pairs.
  Keep the original AP index, matching pairs and complete tested groups or labels.
  Pair/count/layout checks do not establish source identity or group completeness.
  Filter whole groups; recompute AP when removing members of a tested group.
  Copies share the dictionary, backing features and pair arrays; keep them
  unchanged between AP calculation and testing. The attached pairs keep the features and
  every positive and negative pair alive with the frame (8 bytes per pair,
  e.g. 800 MB for 10^8 negative pairs), whichever null is used;
  `ap.attrs.pop("copairs_pairs")` drops them. Multilabel APs carry pair dictionaries
  with the same lifecycle contract and use the component model described below.
  The null recomputes similarities from the features,
  so distances that estimate their scale from each batch of pairs
  (`seuclidean`, `mahalanobis`) raise under it; use `"joint"` or `"copied"`.
  Custom similarities must give the same value for a pair regardless of
  batch size or unrelated pairs in that batch.
  Overlapping-but-different reference sets within a single-label group remain
  unsupported; sharing references across separate groups is allowed.

  The null assumes the group's profiles are exchangeable with the
  references they are ranked against: replicates that share structure the
  references lack (the same well, clone or batch) make it too narrow, so match
  references to that structure (`neg_sameby`) or test whole units. It is the
  slowest of the three nulls, and its cost grows with group size.
- `"joint"` samples the members' APs together: a positive pair inside the
  group is one random number in both members' rank lists, so the null
  narrows as replicates are added. Other similarities are treated as
  independent, which real profiles are not (members share references, and
  some profiles are close to many others), so on real data it can give too
  small p-values for groups of many replicates. It takes each group's
  positive pairs from the attached pairs (or `pairs=`, `pos_pairs=`), so
  positives in other groups stay private to each member. Multi-member groups
  require actual pairs: counts alone cannot identify their graph. Singletons
  can use counts alone. Explicit `pos_pairs=` must agree with an available
  selected pair dictionary. Missing information raises rather than selecting
  another null.
  Equivalent signatures share simulation draws and correlated Monte Carlo errors.
- `"copied"` retains the earlier statistical model: each member's AP null is
  sampled per `(n_pos_pairs, n_total_pairs)` configuration and the group's
  null averages them, so members with equal pair counts share one draw and
  the null does not narrow with replicates. It is the only null of
  `method="legacy"`, which samples, caches and counts as 0.5.5 did (null mAPs
  `>` the observed mAP); under the fast method it is sampled exactly and
  counts `>=`. Both agree with 0.5.5 within Monte Carlo error, not bitwise:
  each configuration is now seeded by its pair counts.

### Multilabel component permutation

`map.multilabel.average_precision` attaches a pair dictionary with additional
profile/label mappings (`"profile"`, `"row_key"`, `"keys"`, `"label_col"`) and a
lazy `"_lone"` pair function. The `"label_col"` entry selects the multilabel
permutation path. Each AP row represents one (source profile, label) pair; group
by the label column. Positive pairs index expanded AP rows; negatives index
source profiles. Joint sampling uses the attached positive AP-row graph.

For multilabel permutation, pools are connected components of the original
negative-pair graph. A draw keeps the tested label's member count in each
component, assigns its members' positive-graph roles to sampled profiles, and
retains every other label. Negatives come from the original negatives plus
pairs sharing only the tested label, excluding the selected receivers. Their
counts can change between draws. The label column must appear in `neg_diffby`
and not `neg_sameby`.

This model accepts overlapping-but-different member reference
sets that the single-label block model rejects. For profiles labelled `A`,
`A+B`, `B`, `C`, the A members have references `{B, C}` and `{C}`. Their negative
graph connects all four profiles, allowing A to move to two of them per draw.
Other labels stay attached to their profiles, but A's overlap with them can
change, so this tests label placement rather than an effect unique to A.
Profiles must be exchangeable within each component; connectivity alone is
insufficient. Use `neg_sameby` to separate plate or batch structure where needed.

### Numerical policy and reproducibility

**0.6.0 changes the default null and statistical results.** See the
[migration notes](docs/migration-0.6.0.md) for API and numerical changes.

- Public mAP functions default to `seed=0`; explicit `None` resolves fresh entropy
  once per run. Joint/permutation seeds must be integers in `[0, 2**64)`;
  copied accepts any non-negative integer. Fresh copied runs are not cached.
- With fixed inputs, row identities, seed and software environment, worker counts
  and computation chunks do not change draws/results.
- Means accumulate in float64 and are reported in the AP dtype. A null value
  counts when `null >= score - tol`, with
  `tol = max(1e-10, 8 * eps(AP dtype) * abs(score))`; copied also accounts for
  its float32 AP draws. Closely spaced distinct values can fall within this tolerance.
- Positive draw counts use Monte Carlo `(hits + 1)/(draws + 1)`, with minimum
  sampled p-value `1/(draws + 1)`. There is no automatic exact enumeration or
  adaptive stopping; all requested draws run.
- `null_size=0` requests scores only, after argument/layout validation. P-values
  and corrected values are `NaN`; significance flags are nullable booleans with
  `pd.NA`, including hierarchical stage-1 outputs. No sampling or correction runs.
- Rows without usable AP/positive pairs or group keys are excluded with a log
  warning. No usable groups, invalid data or unsupported layouts raise.

### Per-call permutation resources

All three public mAP functions accept these keyword arguments:

```python
max_reference_profiles = 16_384
pool_memory_budget = 4 * 1024**3
key_table_memory_budget = 256 * 1024**2
```

The reference cap is a non-negative integer counting reference **profile rows
across all blocks of one tested group**, not compounds or the total dataset.
For multilabel APs, it counts the union of the group's components **minus that
label's members**, summed across all its components. Pool tables include members
and may therefore be larger than reference-only tables at the same cap.
Splitting the same references into additional blocks does not reduce that count.
It is checked before building quadratic pool tables. Raising it can require
substantially more RAM.

The positive pool byte budget schedules estimated concurrent key/rank-table
memory; **one oversized pool may run alone**. The non-negative key-table budget
limits an optional cross-pool reuse table; zero disables that optimization.
The budgets cover tables rather than total process RAM; features, layouts and
temporary arrays need additional memory. Per-call settings leave global limits
unchanged.

`max_workers` controls permutation group work and shared-key construction, plus
copied p-value evaluation. It does not control joint sampling, DuckDB, the original
AP similarity thread pool, or BLAS/internal library threads; Numba's kernels
run on Numba's threads (see the Numba backend above). Persistent
joint/permutation caches are separate work.

## Citation
If you find this work useful for your research, please cite our [paper](https://doi.org/10.1038/s41467-025-60306-2):

Kalinin, A.A., Arevalo, J., Serrano, E., Vulliard, L., Tsang, H., Bornholdt, M., Muñoz, A.F., Sivagurunathan, S., Rajwa, B., Carpenter, A.E., Way, G.P. and Singh, S., 2025. A versatile information retrieval framework for evaluating profile strength and similarity. _Nature Communications_ 16, 5181. doi:10.1038/s41467-025-60306-2

BibTeX:
```
@article{kalinin2025versatile,
  author       = {Kalinin, Alexandr A. and Arevalo, John and Serrano, Erik and Vulliard, Loan and Tsang, Hillary and Bornholdt, Michael and Muñoz, Alán F. and Sivagurunathan, Suganya and Rajwa, Bartek and Carpenter, Anne E. and Way, Gregory P. and Singh, Shantanu},
  title        = {A versatile information retrieval framework for evaluating profile strength and similarity},
  journal      = {Nature Communications},
  year         = {2025},
  volume       = {16},
  number       = {1},
  pages        = {5181},
  doi          = {10.1038/s41467-025-60306-2},
  url          = {https://doi.org/10.1038/s41467-025-60306-2},
  issn         = {2041-1723}
}
```
