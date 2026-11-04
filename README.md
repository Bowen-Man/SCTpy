# SCTpy

SCTransform v2 normalization in Python, using NumPy and SciPy. SCTpy is an alpha
(`0.1.0a1`) with a CPU core and an experimental CuPy fitting backend. The runtime
requires neither R nor reference binaries and includes no project-owned compiled extension.

Previously named `sctransform-native`; install and import as `sctpy`.

## Supported core

- Dense NumPy and SciPy CSR/CSC raw UMI counts, with explicit cell/gene selection manifests.
- Fixed-slope offset fitting, dispersion estimation, Poisson genes and regularization.
- Clipped Pearson residuals, residual statistics and highly variable gene (HVG) ranking.
- Block-streamed residual summaries and sparse corrected UMI counts at a fixed depth.
- AnnData input and explicit write-back, with optional float32 residual computation/storage.

Batch models, extra covariates, SCT v1, float32 fitting and a full GPU pipeline are not
implemented. This package does not reproduce the complete historical R API.

## Quickstart

```bash
git clone https://github.com/Bowen-Man/SCTpy.git
cd SCTpy
python -m pip install -e .
```

Use raw, nonnegative integer counts. The numerical core returns **genes × cells**
matrices; the AnnData adapter accepts **cells × genes** and handles orientation explicitly.

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

`SCTConfig.r_v2()` selects the R-v2 compatibility settings. Plain `SCTConfig()` uses
the native research defaults; these modes need not produce the same results. Other
combinations are labelled `experimental`. Reuse `result.selection` through the
`selection=` argument to replay the exact selected cell/gene indices on the same input.

Sparse input stays sparse during validation, selection and reductions. Residual blocks
are dense; the default also retains a dense residual matrix. For HVG-only workflows,
use `SCTConfig.r_v2(residual_output="summary")`: statistics and ranks remain available,
but `result.residuals` is `None`. `residual_dtype="float32"` reduces residual storage;
fitting, regularization, summary reductions and corrected-count reconstruction stay float64.
Corrected UMI counts are returned as CSR, using the median input library size by default;
`corrected_umi_depth=` selects another positive fixed depth.

## AnnData

Install the optional AnnData adapter dependency with:

```bash
python -m pip install -e ".[anndata]"
```

```python
from sctpy import SCTConfig, fit_transform_anndata, write_anndata

result = fit_transform_anndata(
    adata, layer="counts", config=SCTConfig.r_v2(), return_corrected_umi=True
)
write_anndata(adata, result, n_top_genes=2000)
```

Fitting does not mutate `adata`. Explicit write-back stores HVGs and parameters in `.var`,
configuration/selection in `.uns["sct"]`, and outputs in `.layers["sct_residuals"]` and
`.layers["sct_corrected_umi"]`. It preserves `.X` and unrelated layers; `copy=True`
returns a new AnnData. Write-back replaces previous SCT outputs, removing the residual
layer for summary results. Excluded genes have NaN residuals: subset to
`.var["sct_eligible_gene"]` or HVGs before PCA.

## Experimental GPU fitting

Install one matching extra: `python -m pip install -e ".[gpu]"` for CUDA 13, or
`.[gpu-cu12]` for CUDA 12. Select it with `SCTConfig.r_v2(cox_reid_backend="cupy")`.
Only batched Cox--Reid dispersion roots run on GPU; inputs and results remain on CPU,
with bounded fitting blocks transferred to/from the selected CuPy device. The backend
respects the active stream and allocator without reconfiguring RMM. See the
[backend boundary](docs/RAPIDS_INTEGRATION.md) for scope and profiling details.

## Development

```bash
python -m pip install -e ".[test,lint,anndata]"
ruff check .
ruff format --check .
mypy
python -m pytest
```

CI covers Windows/Linux on Python 3.10–3.14 and macOS on 3.11, 3.13 and 3.14, plus
package build, metadata checks, source-distribution tests and an isolated wheel install.
Tests include high-precision dispersion references, filtered-gene residual floors,
sparse/dense and block consistency, AnnData round-trips, and frozen R numerical outputs.
GPU tests explicitly skip when their dependencies or hardware are unavailable.

R comparisons apply to the tested inputs and selection manifests; they do not establish
bit identity or universal equivalence. Further details:

- [SCT v2 specification](docs/SCT_V2_SPEC.md)
- [validation and profiling](docs/VALIDATION.md)
- [implementation survey](docs/IMPLEMENTATION_SURVEY.md)
- [RAPIDS/GPU boundary](docs/RAPIDS_INTEGRATION.md)
- [licensing and provenance](docs/LICENSING.md)

## License

BSD-3-Clause. See [LICENSE](LICENSE) and [docs/LICENSING.md](docs/LICENSING.md).
