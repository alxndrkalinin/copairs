# copairs

copairs is a Python package for comparing profiles from experiments. It finds
pairs using metadata and measures how well profiles retrieve related profiles
using average precision (AP) and mean average precision (mAP).

## Install

copairs requires Python 3.11 or newer.

```bash
pip install copairs
```

## Documentation

- [Example notebooks](docs/examples/README.md) for finding pairs and measuring
  phenotypic activity and consistency.
- [mAP null distributions](docs/map-nulls.md) for choosing a statistical model.
- [AP null sampling](docs/ap-null-sampling.md) for reproducibility and faster
  sampling.
- [API reference](docs/api/index.md) for functions and parameters.
- [Contributing](CONTRIBUTING.md) for development and testing.

## Citation

If you use copairs in your research, cite [Kalinin et al., *Nature Communications*
(2025)](https://doi.org/10.1038/s41467-025-60306-2). [BibTeX](docs/index.md#citation)
