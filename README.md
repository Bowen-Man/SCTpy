# SCTpy

SCTpy is an independent NumPy/SciPy implementation of the SCTransform v2 model. The
project is an alpha (`0.1.0a1`): it implements a validated core, not the historical R API.

Previously named `sctransform-native`; the package and import name are now `sctpy`.

The runtime has no R dependency, reference binary, or project-owned compiled extension.
GPL implementations are external numerical oracles only; their source expression, tests,
comments, structure, and binaries are not copied or shipped.

## Supported core

- dense NumPy and SciPy CSR/CSC raw UMI counts;
- one explicit internal orientation: genes by cells;
- reproducible cell/gene selection manifests;
- v2 fixed-slope offset fitting, theta estimation, Poisson genes, regularization,
  Pearson residuals, clipping, residual variance, and HVG ranking;
- block-streamed residual summaries and HVG ranking without retaining a dense matrix;
- corrected UMI counts at a fixed depth, returned as genes-by-cells CSR;
- a non-mutating AnnData adapter and explicit result write-back;
- float64 reference arithmetic with an opt-in float32 residual path;
- an opt-in CuPy backend for batched Cox--Reid dispersion roots.

Batches, additional covariates, v1 compatibility, float32 fitting, and a full GPU pipeline
are not implemented. Unsupported behavior is not approximated silently.

## Quickstart

```bash
python -m pip install -e .
```

```python
from sctpy import SCTConfig, fit_transform

result = fit_transform(
    counts_genes_by_cells,
    gene_names=gene_names,
    cell_names=cell_names,
    matrix_orientation="genes_by_cells",
    config=SCTConfig.r_v2(seed=1448145),
    return_corrected_umi=True,
)

residuals = result.residuals
corrected_umi = result.corrected_umi
hvg_2000 = result.highly_variable_genes(2000)
```

`SCTConfig.r_v2()` selects the validated compatibility path. Plain `SCTConfig()` retains
the native research defaults; mixed configurations are labelled `experimental`. Sparse
input remains sparse during validation, selection, and reductions. Residuals are computed
in bounded gene blocks. Set `residual_output="summary"` in the configuration to retain
only residual mean, variance, and HVG ranks; `result.residuals` is then `None`.
Corrected UMI output remains sparse and defaults to the median input library size;
`corrected_umi_depth=` selects another positive fixed depth.
`residual_dtype="float32"` reduces residual block and output storage; fitting,
regularization, summary reductions, and corrected UMI reconstruction remain float64.

Install the optional AnnData adapter dependency with:

```bash
python -m pip install -e ".[anndata]"
```

`fit_transform_anndata(adata, layer="counts", config=SCTConfig.r_v2())` returns an
`SCTResult` without mutation. `write_anndata(adata, result, n_top_genes=2000)` stores
HVG annotations and parameters in `.var`, configuration/selection in `.uns["sct"]`,
and available outputs in `.layers["sct_residuals"]` / `.layers["sct_corrected_umi"]`.
It preserves `.X` and other layers; `copy=True` returns a new AnnData. Summary results
write no dense layer and replace previous SCT outputs. Residuals for excluded genes
are NaN: subset to `.var["sct_eligible_gene"]` or HVGs before downstream PCA.

Install the experimental fitting backend with `.[gpu]` for the current CUDA 13/RAPIDS
stack or `.[gpu-cu12]` for CUDA 12, then select it with
`SCTConfig.r_v2(cox_reid_backend="cupy")`. Inputs and results remain on CPU; only bounded
Cox--Reid root blocks move to the current CuPy device. The backend respects the active
stream and allocator and never installs or reconfigures RMM.

## Development

```bash
python -m pip install -e ".[test,lint,anndata]"
ruff check .
ruff format --check .
mypy
python -m pytest
```

CI covers Windows and Linux on Python 3.10--3.14, AnnData H5AD round-trips, package build,
metadata checks, and an isolated wheel smoke test. Type checks run on Python 3.13;
each test has a 60-second timeout. CUDA is not required by the package or test suite.

Implementation details are kept in five focused documents:

- [SCT v2 specification](docs/SCT_V2_SPEC.md)
- [validation and profiling](docs/VALIDATION.md)
- [implementation survey](docs/IMPLEMENTATION_SURVEY.md)
- [RAPIDS/GPU boundary](docs/RAPIDS_INTEGRATION.md)
- [licensing and provenance](docs/LICENSING.md)

## License

BSD-3-Clause. See [LICENSE](LICENSE) and [docs/LICENSING.md](docs/LICENSING.md).
