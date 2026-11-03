# SCT v2 alpha specification

This is the implemented `0.1.0a1` contract. Evidence labels follow
[IMPLEMENTATION_SURVEY.md](IMPLEMENTATION_SURVEY.md).

## Input and orientation

The numerical core accepts a two-dimensional **genes x cells** matrix plus unique,
length-matched gene and cell names. Counts must be real, finite, nonnegative integers and
every cell must have positive library size. Dense NumPy and SciPy CSR/CSC are supported;
fitting uses float64. AnnData conversion from **cells x genes** occurs only in the adapter.

Genes detected in fewer than `min_cells` are reported as excluded. All-zero genes are
representable but are not fit. **project decision**

## Selection

A selection manifest records the schema version, seed, algorithm, source shape, ordered
gene/cell indices and names, `min_cells`, and requested sample sizes. Replaying a manifest
does not sample again; order is significant because it defines 500-gene fit bins.

Generated manifests use `Generator(PCG64(seed))`. Cells are sampled first; detection is
recomputed in those cells; non-overdispersed genes are removed from the fitted pool; and
remaining genes are sampled with probabilities inverse to a Gaussian density estimate of
log10 geometric mean. Selected indices are sorted. The named algorithm is
`sct_v2_inverse_density_pcg64_v1`; identical Python and R seeds are not expected to select
identical names. **confirmed from official documentation/source** for the v2 order and
inverse-density rule; **project decision** for PCG64 and numerical details.

## Model and fitting

For gene `g` and cell `c`:

```text
X_gc ~ NB2(mu_gc, theta_g)
log(mu_gc) = beta_g0 + log(n_c)
Var(X_gc) = mu_gc + mu_gc**2 / theta_g
```

Here `n_c` is the cell library size. Using `log10(n_c)` fixes the slope to `ln(10)`.
**confirmed from paper**

`SCTConfig.r_v2()` selects Cox--Reid adjusted dispersion estimation, a local dispersion
trend within ordered 500-gene bins, and a final offset-intercept refit. The reported raw
theta is the inverse raw dispersion, not the trend. Independent score evaluations are
grouped into 64-gene blocks; this changes only the bounded working set. The native
`nb_mle` path remains a separate research mode. **confirmed from official documentation
and measured**

A gene is Poisson-like when its full-data arithmetic mean is below `0.001` or its sample
variance is no greater than its mean. It receives `theta = infinity` and

```text
beta_g0 = log(sum_c X_gc / sum_c n_c).
```

An all-zero gene has intercept `-infinity` and exact zero mean. **confirmed from paper and
official documentation**; the explicit zero representation is a **project decision**.

## Regularization

Raw non-Poisson parameters are smoothed against
`x_g = log10(exp(mean(log(X_gc + 1))) - 1)`. The validated v2 path regularizes

```text
od_g = log10(1 + gmean_g / theta_g)
theta_fit_g = gmean_g / (10**od_fit_g - 1)
```

using independent public R-statistics semantics for the Sheather--Jones bandwidth,
Gaussian smoothing, abundance-bin outlier scores, and clamped prediction support.
`log1p`/`expm1` are used for stability. Failed fits are flagged and excluded from smoother
training; Poisson genes retain analytic parameters. **confirmed from paper and official
documentation; measured**

The v2 compatibility path also applies the documented moment-ratio postprocessing rule:
a finite raw theta becomes infinite when
`(mean**2 / (variance - mean)) / theta < 1e-3`. The branch and outlier state are recorded.

## Residuals and HVGs

Using regularized parameters:

```text
mu_gc = exp(beta_fit_g0 + log(n_c))
model_sd_gc = sqrt(mu_gc + mu_gc**2 / theta_fit_g)
floor = median(nonzero counts) / 5
Z_gc = (X_gc - mu_gc) / max(model_sd_gc, floor)
```

Residuals are clipped to `[-sqrt(C), sqrt(C)]` by the low-level API. Residual mean and
sample variance use all cells after clipping. HVGs are ordered by descending residual
variance, then original gene index. **confirmed from paper and official documentation**;
the deterministic tie-break is a **project decision**.

`residual_output="full"` retains the dense genes-by-cells residual matrix. The `summary`
mode computes each bounded gene block once, updates residual mean/variance and HVG ranks,
then releases it. Both modes use identical block arithmetic; summary mode never allocates
the full residual matrix. **project decision; measured**

The default `residual_dtype="float64"` is the reference path. The opt-in `"float32"`
path uses float32 residual-block arithmetic and storage, with float64 reductions for mean
and variance. Parameter fitting and regularization remain float64. **project decision;
measured**

## Corrected UMI

When requested, the same regularized model is inverted at one fixed library depth `n0`:

```text
Z_gc = (X_gc - mu_gc) / model_sd_gc
mu0_g = exp(beta_fit_g0) * n0
X_corrected_gc = round(max(0, mu0_g + Z_gc * sqrt(mu0_g + mu0_g**2/theta_fit_g)))
```

This path uses the unclipped Pearson residual and model standard deviation without the
residual floor. `n0` defaults to the median input library size. The result is an int64
genes-by-cells CSR matrix; excluded genes remain zero. It is produced in the residual
gene blocks, does not require retaining the dense residual matrix, and always uses the
float64 model even when residual output is float32. **confirmed from paper and official
documentation; measured**

## Explicit exclusions

Multiple batches, additional/categorical covariates, v1 aliases, float32 reference
fitting, and implicit CPU fallback presented as GPU execution are not supported. Public
boundaries raise instead of applying an undocumented approximation.

The pre-registered numerical gates and measured fixtures are in
[VALIDATION.md](VALIDATION.md); thresholds are implemented in
`benchmarks/correctness.py`.
