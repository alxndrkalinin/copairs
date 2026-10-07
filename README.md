 # copairs

`copairs` is a Python package for finding groups of profiles based on metadata and calculate mean Average Precision to assess intra- vs inter-group similarities.

## Getting started

### System requirements
copairs supports Python 3.10 and newer (tested through Python 3.14) and should work with all modern operating systems (tested with MacOS 13.5, Ubuntu 18.04, Windows 10).

For Python 3.9, use the last compatible release: `pip install "copairs==0.5.5"`.

### Dependencies
copairs depends on widely used Python packages:
* numpy
* pandas
* tqdm
* statsmodels
* duckdb
* numba

### Installation

To install copairs and dependencies, run:
```bash
pip install copairs
```

To also install dependencies for running examples, run:
```bash
pip install copairs[demo]
```

#### GPU acceleration (optional)

With [CuPy](https://cupy.dev) installed and a CUDA GPU visible, copairs runs
null distributions, p-values, pair similarities and AP on the GPU
(`backend="auto"` picks it). Install the CuPy wheel matching your driver, one
of the two (the `ctk` extra brings the CUDA runtime and NVRTC, so no CUDA
Toolkit is needed):
```bash
pip install "cupy-cuda12x[ctk]>=14"  # CUDA 12 drivers, any GPU from Maxwell on
pip install "cupy-cuda13x[ctk]>=14"  # CUDA 13 drivers, compute capability >= 7.5
```

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

## Performance

`average_precision`, `mean_average_precision` and the p-value functions take
`method="fast"` (default) or `"legacy"`, and `backend="auto"`, `"cuda"`,
`"numba"` or `"numpy"`.

- Null distributions are sampled exactly (Philox counter-based RNG, Vitter's
  Algorithm A) in O(1) memory per sample instead of permuting a
  `null_size x total` matrix, and p-values are streamed in chunks. Sample `j`
  of a `(num_pos, total)` null depends only on `(seed, num_pos, total, j)`, and
  the NumPy, Numba and CUDA backends return identical samples.
- AP is computed by counting, without sorting rank lists, and similarities by
  per-pair kernels; multilabel pairs come from an inverted label index.
- `copairs.fastap.draw_average_precisions` scores many query-vs-reference draws
  in one batched call.

Differences from `method="legacy"` (copairs <= 0.5.5): null samples come from a
different random stream (p-values agree within Monte Carlo error); mAP p-values
count null values `>=` the observed mAP, as in the paper, where legacy counted
`>` and returned too small p-values when the mAP equals an atom of the null
(e.g. perfect retrieval); and similarities from the kernels match the generic
ones to float32 rounding, so near-tied pairs can rank differently.

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
