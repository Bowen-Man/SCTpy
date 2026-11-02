# sctransform-native

`sctransform-native` is a clean-room, native Python research implementation of the
documented SCTransform v2 statistical model. The first release line is deliberately
labelled alpha: it implements only behavior that has an explicit mathematical contract
and does not claim the complete historical R API.

The project has two strict boundaries:

- the numerical core is independent NumPy/SciPy code and contains no project-owned
  C, C++, Fortran, or R runtime component;
- GPL implementations may be executed outside this repository as black-box oracles,
  but their source, tests, comments, layout, and binaries are not inputs to this codebase.

Current version: `0.1.0a1`.

## Status

The independent specification and NumPy/SciPy CPU alpha are available. Differential
validation, profiling, and any GPU experiment are introduced in separate, reviewable pull
requests. Corrected UMI counts, batch models, and extra covariates are out of scope until
their contracts are independently verified.

See:

- [implementation survey](docs/IMPLEMENTATION_SURVEY.md)
- [SCT v2 contract](docs/SCT_V2_SPEC.md)
- [CPU alpha API](docs/CPU_API.md)
- [licensing rationale](docs/LICENSING.md)
- [RAPIDS integration boundary](docs/RAPIDS_INTEGRATION.md)

## CPU quickstart

```python
from sctransform_native import SCTConfig, fit_transform

result = fit_transform(
    counts_genes_by_cells,
    gene_names=gene_names,
    cell_names=cell_names,
    matrix_orientation="genes_by_cells",
    config=SCTConfig(seed=1448145),
)

residuals = result.residuals
hvg_2000 = result.highly_variable_genes(2000)
```

## Development

```bash
python -m pip install -e ".[test]"
python -m pytest
```

The package supports Python 3.10 through 3.13 on Windows and Linux. GPU packages are
never required for CPU installation.

## License

BSD-3-Clause. See [LICENSE](LICENSE) and [docs/LICENSING.md](docs/LICENSING.md).
