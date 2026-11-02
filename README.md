# sctransform-native

Independent NumPy/SciPy implementation of the documented SCTransform v2 model. The
project is an alpha (`0.1.0a1`): it implements a validated core, not the historical R API.

The runtime has no R dependency, reference binary, or project-owned compiled extension.
GPL implementations are external numerical oracles only; their source expression, tests,
comments, structure, and binaries are not copied or shipped.

## Supported core

- dense NumPy and SciPy CSR/CSC raw UMI counts;
- one explicit internal orientation: genes by cells;
- reproducible cell/gene selection manifests;
- v2 fixed-slope offset fitting, theta estimation, Poisson genes, regularization,
  Pearson residuals, clipping, residual variance, and HVG ranking;
- a non-mutating AnnData adapter for cells-by-genes CPU data;
- float64 reference arithmetic.

Corrected UMI counts, batches, additional covariates, v1 compatibility, float32 fitting,
and a package GPU backend are not implemented. Unsupported behavior is not approximated
silently.

## Quickstart

```bash
python -m pip install -e .
```

```python
from sctransform_native import SCTConfig, fit_transform

result = fit_transform(
    counts_genes_by_cells,
    gene_names=gene_names,
    cell_names=cell_names,
    matrix_orientation="genes_by_cells",
    config=SCTConfig.r_v2(seed=1448145),
)

residuals = result.residuals
hvg_2000 = result.highly_variable_genes(2000)
```

`SCTConfig.r_v2()` selects the validated compatibility path. Plain `SCTConfig()` retains
the native research defaults; mixed configurations are labelled `experimental`. Sparse
input remains sparse during validation, selection, and reductions. Residuals are dense
and computed in bounded gene blocks.

Install the optional AnnData adapter dependency with:

```bash
python -m pip install -e ".[anndata]"
```

## Development

```bash
python -m pip install -e ".[test,lint]"
ruff check .
ruff format --check .
python -m pytest
```

CI covers Windows and Linux on Python 3.10--3.13, package build, metadata checks, and an
isolated wheel smoke test. CUDA is not required by the package or test suite.

Implementation details are kept in five focused documents:

- [SCT v2 specification](docs/SCT_V2_SPEC.md)
- [validation and profiling](docs/VALIDATION.md)
- [implementation survey](docs/IMPLEMENTATION_SURVEY.md)
- [RAPIDS/GPU boundary](docs/RAPIDS_INTEGRATION.md)
- [licensing and provenance](docs/LICENSING.md)

## License

BSD-3-Clause. See [LICENSE](LICENSE) and [docs/LICENSING.md](docs/LICENSING.md).
