# mAP null distributions

`mean_average_precision`, `mean_average_precision_hierarchical` and
`get_map_pvalue` take `null=`, the null of a group's mAP. All three count null
mAPs `>=` the observed mAP, so an mAP equal to an atom of the null (e.g.
perfect retrieval) counts the atom. The models make different dependence
assumptions. Calibration depends on the experimental design.

## Permutation

`"permutation"` (default) tests exchangeability. For single-label APs, members
ranked against identical reference sets form a block, such as a plate's
replicates and controls. The pool contains those members and references. On
each draw, the model preserves the number of group members in every block,
relabels profiles, and recomputes mAP from the observed similarities. This
retains the dependence between rank lists. The model ranks each relabelled
profile against the unlabelled profiles in its block.

The model uses the original positive and negative rules to construct the graph
and pools. During each draw, it assigns profiles to roles on the original
positive graph, including non-complete graphs, without reapplying those rules
to fixed metadata. Layout rules such as `neg_diffby=["g", "plate"]` or
`pos_diffby=["plate"]` need not hold after relabelling.

All positive edges must stay inside the tested group. For singleton groups
with positives in other groups, use `"joint"` to obtain the profile's AP null.
For example, that applies to one profile per guide with positives defined by
gene. When computing permutation mAP, pass the frame from `average_precision`,
which carries the pairs used to compute the APs.

```python
ap = copairs.map.average_precision(meta, feats, ...)
result = copairs.map.mean_average_precision(
    ap, sameby, null_size=10_000, threshold=0.05, seed=0
)
```

The frame carries its pairs in `ap.attrs["copairs_pairs"]` through filtering
and copying. Supply `pairs=` or recompute AP after saving, pickling, or
concatenating frames from different AP calls. Use
`average_precision(..., return_pairs=True)` to return the dictionary with
`"pos"`, `"neg"`, and `"similarity"` entries. Explicit `pairs=` takes precedence
over attached pairs. Use `pairs["pos"]` for raw joint pairs.
Keep the original AP index, matching pairs and complete tested groups or labels.
Pair/count/layout checks do not establish source identity or group completeness.
Filter whole groups. Recompute AP when removing members of a tested group.
Copies share the dictionary, backing features, and pair arrays. Keep them
unchanged between AP calculation and testing. The frame retains the features
and every positive and negative pair at 8 bytes per pair for any null. Use
`ap.attrs.pop("copairs_pairs")` to remove its reference to them. Multilabel APs carry pair dictionaries
with the same lifecycle contract and use the component model described below.
The permutation model recomputes similarities from the features. Use `"joint"`
or `"copied"` for distances that estimate their scale from each batch of pairs
(`seuclidean`, `mahalanobis`).
Custom similarities must give the same value for a pair regardless of
batch size or unrelated pairs in that batch.
The permutation model supports shared references across separate groups. Within
a single-label group, use identical or disjoint reference sets.

The model assumes exchangeability between the group's profiles and their
references. It can produce a null that is too narrow when replicates share
structure the references lack, such as a well, clone, or batch. Match references
to that structure with `neg_sameby` or test whole units. The cost of recomputing
mAP grows with group size.

## Joint

`"joint"` samples the members' APs together. The model uses one random number
in both members' rank lists for a positive pair inside the group, so the null
narrows as replicates are added. Other similarities are treated as
independent, which real profiles are not (members share references, and
some profiles are close to many others), so on real data it can give too
small p-values for groups of many replicates. It takes each group's
positive pairs from the attached pairs (or `pairs=`, `pos_pairs=`), so
positives in other groups stay private to each member. Multi-member groups
require actual pairs to identify their graph. Singletons
can use counts alone. Explicit `pos_pairs=` must agree with an available
selected pair dictionary. Copairs raises an error when it lacks required pair
information.
Equivalent signatures share simulation draws and correlated Monte Carlo errors.

## Copied

`"copied"` retains the earlier statistical model. The model draws APs per
`(n_pos_pairs, n_total_pairs)` configuration and averages the members' values
for each group, so members with equal pair counts share one draw and
the null does not narrow with replicates. The default AP sampler caches seeded
draws. See [AP null sampling](ap-null-sampling.md) for streaming options.

## Multilabel component permutation

`map.multilabel.average_precision` attaches a pair dictionary with additional
profile/label mappings (`"profile"`, `"row_key"`, `"keys"`, `"label_col"`) and a
lazy `"_lone"` pair function. The `"label_col"` entry selects the multilabel
permutation path. Each AP row represents one (source profile, label) pair.
Group by the label column. Positive pairs index expanded AP rows. Negatives index
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
change, so the model tests label placement. Profiles must be exchangeable
within each component. Use `neg_sameby` to separate plate or batch structure
where needed.

## Numerical policy and reproducibility

**0.6.0 changes the default null and statistical results.** See the
[migration notes](migration-0.6.0.md) for API and numerical changes.

- Public mAP functions default to `seed=0`. Use `None` to resolve fresh entropy
  once per run. Joint, permutation, and streamed copied sampling use integer
  seeds in `[0, 2**64)`. The legacy copied sampler accepts any non-negative
  integer. Fresh copied draws bypass the cache.
- With fixed inputs, row identities, seed and software environment, worker counts
  and computation chunks do not change draws/results.
- Means accumulate in float64 and are reported in the AP dtype. A null value
  counts when `null >= score - tol`, with
  `tol = max(1e-10, 8 * eps(AP dtype) * abs(score))`. The legacy copied null
  also accounts for float32 AP draws. Closely spaced distinct values can fall
  within this tolerance.
- Positive draw counts use Monte Carlo `(hits + 1)/(draws + 1)`, with minimum
  sampled p-value `1/(draws + 1)`. The sampler runs all requested draws.
- `null_size=0` requests scores only, after argument/layout validation. P-values
  and corrected values are `NaN`. Significance flags are nullable booleans with
  `pd.NA`, including hierarchical stage-1 outputs. No sampling or correction runs.
- Rows without usable AP/positive pairs or group keys are excluded with a log
  warning. No usable groups, invalid data or unsupported layouts raise.

## Per-call permutation resources

These are the defaults for the three public mAP functions.

```python
max_reference_profiles = 16_384
pool_memory_budget = 4 * 1024**3
key_table_memory_budget = 256 * 1024**2
```

The reference cap is a non-negative integer counting reference **profile rows
across all blocks of one tested group**.
For multilabel APs, it counts the union of the group's components **minus that
label's members**, summed across all its components. Pool tables include members
and may therefore be larger than reference-only tables at the same cap.
Splitting the same references into additional blocks does not reduce that count.
It is checked before building quadratic pool tables. Raising it can require
substantially more RAM.

Copairs uses the positive pool byte budget to schedule concurrent key and rank
tables by their estimated memory use. One oversized pool may run alone. The
non-negative key-table budget limits an optional cross-pool reuse table. Set it
to zero to disable that optimization. Features, layouts, and temporary arrays
need memory in addition to these budgets. These limits apply per call.

`max_workers` controls permutation group work and shared-key construction,
legacy copied p-value evaluation, and streamed NumPy sampling for the copied
null. Joint sampling, DuckDB, the AP similarity thread pool, Numba, and BLAS
use their own thread settings.
