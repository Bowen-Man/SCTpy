# CPU alpha API

The public low-level entry point is:

```python
from sctransform_native import SCTConfig, fit_transform

result = fit_transform(
    counts_genes_by_cells,
    gene_names=gene_names,
    cell_names=cell_names,
    matrix_orientation="genes_by_cells",
    config=SCTConfig.r_v2(seed=1448145),
)
```

`matrix_orientation="cells_by_genes"` is also accepted and performs one explicit
transpose at validation. Internally, every matrix is genes x cells. Dense NumPy and SciPy
sparse matrix/array inputs are supported. Sparse formats (including CSR, CSC, and COO) are
validated and normalized once to sorted CSR; they stay sparse through statistics and
selection. Only the current gene row/block and the documented dense residual output are
materialized. Model fitting and residual output use float64.

The optional AnnData adapter reads `adata.X` or one explicitly named layer without
mutating the object:

```python
from sctransform_native import fit_transform_anndata

result = fit_transform_anndata(adata, layer="counts")
```

AnnData is cells x genes, so the adapter calls the core with that orientation explicitly.
A CuPy-backed `X` or layer raises `ImplicitDeviceTransferError`; callers must choose and
perform device movement themselves.

## Statistical modes

`SCTConfig.r_v2()` is the frozen compatibility preset validated against R 4.5.2,
`sctransform` 0.4.3, and `glmGamPoi` 1.22.0. It selects:

```text
cox_reid / gene_bin_500 / sct_v2 / r_stats / clamp / moment_ratio
```

The preset accepts selection controls (`min_cells`, `n_genes`, `n_cells`, and `seed`) and
the execution-only `residual_block_size` and `cox_reid_batch_size`. The validated batch
default is 64 genes; pass `None` to replay the scalar solver. Batch size does not change
`statistical_mode`. Plain `SCTConfig()` retains the existing native research path and
therefore remains backward compatible. `config.statistical_mode` returns `"r_v2"`,
`"native"`, or `"experimental"`; the last label covers mixed algorithm settings that have
not passed the complete compatibility gate as one configuration.

## Result contract

`SCTResult` exposes:

- input statistics and ordered names;
- the complete `SelectionManifest`;
- raw intercept/theta fits and convergence diagnostics;
- fixed slopes, Poisson flags, theta-postprocess flags, regularized parameters, and
  outlier flags;
- clipped Pearson residuals, residual mean/variance, and deterministic HVG order.

Genes below `min_cells` remain visible with `fit_status="insufficient_cells"`, NaN
parameters/residuals, and HVG rank `-1`. Failed NB fits are named in `fit_status` and are
excluded from the smoother training data; if a curve can be learned from other genes, the
failed gene receives a visibly labelled regularized prediction.

The default selection uses `Generator(PCG64(seed))` and samples cells without replacement.
It recomputes per-gene detection on the selected columns in sparse form, excludes
genes with full-data variance not exceeding their mean from step-one fitting, and uses
inverse abundance-density weights when `n_genes` truncates the remaining pool. Indices are
sorted before fitting and the manifest records `sct_v2_inverse_density_pcg64_v1`. Pass a
serialized manifest to replay exactly the same cells and genes. NumPy and R RNG streams
are intentionally not treated as identical.

Raw fitting is explicit. `fit_method="nb_mle"` remains the alpha default. The optional
`fit_method="cox_reid"` uses per-gene Cox--Reid adjusted dispersion estimates followed by
an abundance-local weighted-median dispersion trend for the intercept refit. It is native
NumPy/SciPy code and does not import `glmGamPoi` or R. The independent default evaluates
one trend across all fitted genes. `SCTConfig.r_v2()` instead fixes
`cox_reid_gene_bin_size=500`; the bin boundary affects the local trend and is therefore a
statistical compatibility setting. It is distinct from `cox_reid_batch_size`, which only
bounds the dense selected-cell working block used by the initial dispersion solve.

Dispersion regularization is also explicit. The alpha default remains `native`; use
`SCTConfig(dispersion_regularization="sct_v2")` to smooth
`log10(1 + geometric_mean / theta)` and apply its matching inverse transform. This switch
does not claim full R behavioral compatibility: the estimator and sampling remain native.

Regularization behavior is independently selectable. The default
`regularization_method="native"` retains the paper-derived alpha implementation. Use
`regularization_method="r_stats"` for the independently written R `stats` bandwidth,
normal-kernel smoothing, and abundance-local outlier semantics. A controlled v2 comparison
therefore uses:

```python
SCTConfig.r_v2()
```

This does not change device placement or add an R dependency.

Kernel predictions use `regularization_support="extrapolate"` by default. Set the policy
to `"clamp"` to evaluate genes outside the training abundance range at the nearest
training boundary. Device placement and matrix orientation are unaffected by this scalar
coordinate policy.

Theta postprocessing is disabled by default. The explicit
`theta_postprocess="moment_ratio"` policy compares the method-of-moments theta
with each available raw fit, using the recorded
`theta_moment_ratio_threshold` (default `1e-3`). Flagged raw theta values become
infinite, are exposed through `result.theta_postprocessed`, and are excluded
from smoother training while still receiving regularized predictions. Raw theta values
that are already infinite are also excluded and counted as parameter outliers, but are
not labelled as newly postprocessed.

## Deliberately unsupported

`return_corrected_umi=True`, `batch=...`, and `covariates=...` raise
`NotImplementedError`. The alpha never substitutes an approximation for those paths.

## Numerical implementation

The default per-gene NB2 intercept and theta are fitted by independently derived maximum
likelihood with analytic gradients and SciPy L-BFGS-B. The optional Cox--Reid path groups
independent score evaluations into bounded dense gene blocks and uses safeguarded per-gene
roots; preliminary and final intercept fits remain independent scalar solves. Both
regularization methods are NumPy/SciPy code and keep pair-distance, smoothing, and outlier
temporaries bounded.
The native derivative-functional sums are block bounded. The `r_stats` path uses at most
1000 distance bins for larger training sets and truncates the normal kernel at four
standard deviations. The package contains no R-math layer and does not claim bit identity.
