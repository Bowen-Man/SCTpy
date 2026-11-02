# CPU alpha API

The public low-level entry point is:

```python
from sctransform_native import SCTConfig, fit_transform

result = fit_transform(
    counts_genes_by_cells,
    gene_names=gene_names,
    cell_names=cell_names,
    matrix_orientation="genes_by_cells",
    config=SCTConfig(seed=1448145),
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

The default selection uses `Generator(PCG64(seed))`, samples without replacement, sorts
indices before fitting, and records the algorithm name. Pass a previously serialized
manifest to replay exactly the same cells and genes. The alpha's uniform selection is an
explicit known difference from the reference density-dependent gene selection.

Dispersion regularization is also explicit. The alpha default remains `native`; use
`SCTConfig(dispersion_regularization="sct_v2")` to smooth
`log10(1 + geometric_mean / theta)` and apply its matching inverse transform. This switch
does not claim full R behavioral compatibility: the estimator, sampling, outlier, and
smoother contracts remain native.

Kernel predictions use `regularization_support="extrapolate"` by default. Set the policy
to `"clamp"` to evaluate genes outside the training abundance range at the nearest
training boundary. Device placement and matrix orientation are unaffected by this scalar
coordinate policy.

Theta postprocessing is disabled by default. The experimental
`theta_postprocess="moment_ratio"` policy compares the method-of-moments theta
with each converged native fit, using the recorded
`theta_moment_ratio_threshold` (default `1e-3`). Flagged raw theta values become
infinite, are exposed through `result.theta_postprocessed`, and are excluded
from smoother training while still receiving regularized predictions. This is
a native research policy, not a claim of estimator-level R compatibility.

## Deliberately unsupported

`return_corrected_umi=True`, `batch=...`, and `covariates=...` raise
`NotImplementedError`. The alpha never substitutes an approximation for those paths.

## Numerical implementation

Per-gene NB2 intercept and theta are fitted by independently derived maximum likelihood
with analytic gradients and SciPy L-BFGS-B. Regularization uses a Gaussian
Nadaraya-Watson smoother, the published Sheather--Jones solve-the-equation bandwidth
selector (multiplied by the configured adjustment), and a recorded 10-MAD outlier rule.
The derivative-functional sums are evaluated in NumPy blocks so their temporary memory is
bounded. The smoother's Gaussian scale follows the public `ksmooth` convention: kernel
quartiles are at `+/- 0.25 * bandwidth`, so its standard deviation is
`bandwidth / 2.697959...`. These are algorithmic definitions from papers and public API
contracts; the package does not contain an R-math compatibility layer or claim bit
identity.
