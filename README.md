# sctransform-native

`sctransform-native` is an independent native Python research implementation of the
documented SCTransform v2 statistical model. The first release line is deliberately
labelled alpha: it implements only behavior that has an explicit mathematical contract
and does not claim the complete historical R API.

The project has two strict boundaries:

- the numerical core is independent NumPy/SciPy code and contains no project-owned
  C, C++, Fortran, or R runtime component;
- external implementations may define observable algorithmic behavior and serve as
  differential oracles, but no source text, tests, comments, layout, binaries, or runtime
  component is copied or shipped here.

Current version: `0.1.0a1`.

## Status

The independent specification and NumPy/SciPy CPU alpha are available. Differential
validation, profiling, and any GPU experiment are introduced in separate, reviewable pull
requests. Corrected UMI counts, batch models, and extra covariates are out of scope until
their contracts are independently verified.

See:

- [implementation survey](docs/IMPLEMENTATION_SURVEY.md)
- [current state assessment](docs/CURRENT_STATE.md)
- [SCT v2 contract](docs/SCT_V2_SPEC.md)
- [CPU alpha API](docs/CPU_API.md)
- [correctness gate](docs/CORRECTNESS.md)
- [stage-report contract](docs/REPORT_SCHEMA.md)
- [real-data validation plan](docs/REAL_DATA_VALIDATION.md)
- [benchmark and profiling report](docs/BENCHMARKS.md)
- [GPU feasibility report](docs/GPU_FEASIBILITY.md)
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
    config=SCTConfig.r_v2(seed=1448145),
)

residuals = result.residuals
hvg_2000 = result.highly_variable_genes(2000)
```

`SCTConfig.r_v2()` selects the validated R-v2 compatibility path. Plain `SCTConfig()`
retains the original native research defaults for API compatibility; mixed, unvalidated
algorithm combinations are reported as `config.statistical_mode == "experimental"`. The
preset also fixes the compatibility-sensitive Cox--Reid gene bin at 500; it is not a
performance-only batch size. Independent score evaluation uses a separately configurable,
execution-only 64-gene block by default.

## Development

```bash
python -m pip install -e ".[test,lint]"
ruff check .
ruff format --check .
python -m pytest
```

The package supports Python 3.10 through 3.13. CI exercises every supported Python on
Windows and Linux, plus Python 3.11 and 3.13 on the pinned macOS 15 x64 image. GPU
packages are never required for CPU installation.

AnnData support is an optional, non-mutating adapter:

```bash
python -m pip install -e ".[anndata]"
```

CUDA experiments use explicit versioned extras (`gpu-cuda12` or `gpu-cuda13`) and are
never imported by the CPU path.

## License

BSD-3-Clause. See [LICENSE](LICENSE) and [docs/LICENSING.md](docs/LICENSING.md).
