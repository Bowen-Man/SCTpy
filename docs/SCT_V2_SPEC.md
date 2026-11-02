# Independent SCT v2 alpha specification

Status: implemented alpha contract for `0.1.0a1`. The labels below use the definitions
in [IMPLEMENTATION_SURVEY.md](IMPLEMENTATION_SURVEY.md).

## Supported input contract

The numerical core receives a two-dimensional **genes x cells** matrix `X[g, c]`, gene
names of length `G`, and cell names of length `C`. This is the only internal orientation.
An AnnData adapter, when added, performs the single explicit conversion from AnnData's
**cells x genes** convention. **project decision**

Counts must be finite, nonnegative, integer-valued, and real. Names must be unique and
length-matched. Every cell must have positive library size. Dense NumPy and SciPy CSR/CSC
inputs are in scope; fitting arithmetic is float64. **project decision**

Genes detected in fewer than `min_cells` are reported as excluded rather than silently
discarded. All-zero genes are representable but not fit. **project decision**

## Reproducible selection manifest

Selection is a pure preprocessing step. A run stores a JSON-serializable manifest with:

```text
schema_version, seed, algorithm, source_shape,
selected_gene_indices, selected_gene_names,
selected_cell_indices, selected_cell_names,
min_cells, requested_genes, requested_cells
```

Callers may supply a complete manifest, in which case no random sampling occurs. Generated
manifests use NumPy's documented `Generator(PCG64(seed))`; sorted selected indices define
the execution order. Cells are sampled first. Gene detection is then recomputed within
those cells without densifying a sparse matrix, and genes below `min_cells` are excluded
from step-one fitting. Genes whose full-data variance does not exceed their mean are also
excluded from the fitted-gene pool. If a gene limit remains, sampling probabilities are
proportional to the inverse of a Gaussian density estimate over log10 geometric mean. The
estimator uses the documented `bw.nrd` scale and a 512-point linearly binned grid.
**confirmed from official source** for the order and inverse-density rule; **project
decision** for PCG64, the independent grid implementation, deterministic sorting, and
numerical fallbacks.

A fixed R 4.5.2 black-box golden for inverse-density probabilities is covered by unit
tests. On an additional 41-point clustered/outlier probe, native versus R probabilities
had Pearson correlation `0.999999984`, Spearman correlation `0.998258`, and maximum
absolute error `6.76e-05`. **measured on 2026-11-02**

The native RNG is deliberately not an R-RNG compatibility layer, so identical seeds do
not imply identical sampled names across languages. Oracle comparisons therefore persist
and replay one manifest. The algorithm field
`sct_v2_inverse_density_pcg64_v1` makes this boundary explicit. **project decision**

## Statistical model

For gene `g` and cell `c`:

```text
X_gc ~ NB2(mu_gc, theta_g)
log(mu_gc) = beta_g0 + log(n_c)
Var(X_gc) = mu_gc + mu_gc**2 / theta_g
```

where `n_c = sum_g X_gc`. Equivalently, using `log10(n_c)` as the covariate fixes the
slope to `ln(10)`. **confirmed from paper**

The default `fit_method="nb_mle"` independently maximizes the NB2 log likelihood in
float64. Optimizer bounds, stopping rules, and convergence flags are part of the output
and differential-testing contract. **project decision**

The optional `fit_method="cox_reid"` follows the v2 estimator stages: a moment dispersion
initializes an offset-intercept fit, each gene receives a Cox--Reid adjusted raw dispersion
estimate, and an abundance-local weighted-median dispersion trend is used to refit the
intercept. The reported theta is the inverse raw dispersion, not the trend value. The
implementation is original NumPy/SciPy code with one-dimensional per-gene working sets.
Fixed golden tests cover raw theta, the zero-dispersion branch, the local trend, and final
intercepts. **confirmed from official source and measured**

## Poisson-like genes

Let `m_g` be the arithmetic mean across all cells and `s_g^2` the sample variance
(`ddof=1`). A gene is Poisson-like when `m_g < 0.001` or `s_g^2 <= m_g`. Such a gene is
excluded from NB parameter regularization, receives `theta_g = infinity`, and uses the
analytic offset intercept:

```text
beta_g0 = log(sum_c X_gc / sum_c n_c).
```

For an all-zero gene the intercept is `-infinity` and `mu` is exactly zero. Selection may
subsample cells for NB fitting, but this classification, the Poisson intercept, and the
abundance statistics used by regularization remain full-data quantities. **confirmed from
paper and official source** for the rule, data scope, and analytic intercept; explicit zero
representation is a **project decision**.

The two upstream stages remain distinct: a low-mean gene with variance above its mean may
receive a raw step-one fit, but its final regularized theta is infinity and its final
intercept is the full-data analytic offset. **confirmed from official source**

## Regularization

Raw non-Poisson estimates are smoothed against log10 gene geometric mean, with a
pseudocount of 1. V2 documentation calls the theta target `od_factor`; under NB2 the
dimensionless extra-Poisson factor is `m_g / theta_g`. The alpha therefore defines:

```text
gmean_g = exp(mean_c(log(X_gc + 1))) - 1
x_g = log10(max(gmean_g, smallest_positive_float64))
od_g = log10(m_g / theta_g)
```

and independently smooths `beta_g0` and `od_g` as functions of `x_g`. Then
`theta_fit_g = m_g / 10**od_fit_g`. This remains the default `native` alpha contract.

An explicit `sct_v2` dispersion regularization option instead uses the dimensionless
NB2 variance factor and the geometric abundance measure described by Choudhary and
Satija (2022):

```text
od_g = log10(1 + gmean_g / theta_g)
theta_fit_g = gmean_g / (10**od_fit_g - 1)
```

The implementation evaluates these expressions with `log1p` and `expm1` for stability.
This option changes only the dispersion target and inverse transform. Smoothing behavior
is selected separately. The default `regularization_method="native"` uses a Gaussian
Nadaraya-Watson smoother and an independent implementation of the published
Sheather--Jones solve-the-equation selector, including diagonal terms, with the configured
default adjustment of 3. The selector is derived from equations 12--13 of Sheather and
Jones ([1991](https://doi.org/10.1111/j.2517-6161.1991.tb01857.x)).
**confirmed from paper and official documentation** for geometric-mean regularization and
the NB2 variance factor; **project decision** for the transform switch, stable arithmetic,
numerical blocking, and root solver.

`regularization_method="r_stats"` independently implements the relevant public R 4.5.2
`stats` semantics: `bw.SJ(method="ste")` pair-distance binning, normal `ksmooth` bandwidth
scaling and four-standard-deviation support, and the two shifted abundance-bin robust
scores used for parameter outliers. Golden tests were generated by a separately installed
R process and cover both exact and binned bandwidth paths, smoothing, and the outlier
mask. No R runtime or source is packaged. This option describes only regularization and
does not alter selection or fitting.
**confirmed from official source and measured**

The semantic references were frozen on 2026-11-02: `satijalab/sctransform` v0.4.3 at
[`49e35b5`](https://github.com/satijalab/sctransform/tree/49e35b5aeb76a602910207cbfde1561093340be3)
and the observed R `R-4-5-branch` at
[`b4322f3`](https://github.com/wch/r-source/tree/b4322f3207e8d99d9dc6e6fa6b80ad23c4dc2f5f).
The Cox--Reid semantic reference is `glmGamPoi` 1.22.0,
[`908cde9`](https://github.com/const-ae/glmGamPoi/tree/908cde9d595259b4a317343ddfbcf20036ef0895).

Prediction support is explicit: `extrapolate` passes every log-geometric-mean coordinate
to the smoother, while `clamp` restricts prediction coordinates to the minimum and
maximum retained training coordinate. The alpha default remains `extrapolate`.
**project decision**

Poisson-like genes retain `theta=infinity` and the analytic intercept. Failed NB fits are
never silently regularized; they are flagged and excluded from the smoothing training set.
**project decision**

The alpha default remains the documented global residual-MAD rule. The explicit `r_stats`
method instead evaluates each parameter within two shifted bins along log geometric mean
and flags a value only when both robust-score magnitudes exceed the threshold. A named
black-box checkpoint against `sctransform::vst()` remains part of the correctness report.
**confirmed from official source and measured**

The differential harness also reports finite versus infinite raw-theta masks. The default
`theta_postprocess="none"` leaves raw estimates unchanged. The explicit `moment_ratio`
policy computes

```text
theta_mm = mean**2 / (variance - mean), when variance > mean
flag = theta_mm / theta_native < threshold
```

with a recorded default threshold of `1e-3`. Flagged finite raw theta values become
infinity; raw nonfinite estimates and newly flagged estimates are excluded from smoother
training and counted as parameter outliers. They remain non-Poisson modeled genes and
receive regularized predictions. **confirmed from official source and measured**

## Pearson residuals and v2 floor

Using regularized parameters:

```text
mu_gc = exp(beta_fit_g0 + log(n_c))
model_sd_gc = sqrt(mu_gc + mu_gc**2 / theta_fit_g)
floor = median({X_gc | X_gc > 0, over the count matrix}) / 5
denom_gc = max(model_sd_gc, floor)
Z_gc = (X_gc - mu_gc) / denom_gc
```

The official public helper documentation defines `nzmedian` as the median of non-zero
entries in the UMI matrix (optionally over a named gene set), so the alpha uses one scalar
floor for the run. If the matrix has no nonzero counts, `floor = 0`; positive cell-library
validation normally makes that case unreachable. **confirmed from paper and official
documentation** except that explicit empty-set behavior is a **project decision**.

Residual clipping is explicit. The low-level native default is
`[-sqrt(C), +sqrt(C)]`, matching the public low-level documentation. An AnnData/Seurat-like
adapter may offer `[-sqrt(C/30), +sqrt(C/30)]`, but must record that choice. **confirmed
from official documentation; project decision**

Residual mean and sample variance use all cells after clipping; HVGs are ranked by
descending residual variance with original gene index as deterministic tie-breaker.
**confirmed from paper; project decision** tie-break.

## Deliberately unsupported in the alpha

- corrected UMI counts;
- batch interactions or multiple batches;
- additional/categorical covariates;
- v1 behavior or historical API aliases;
- float32 reference fitting;
- implicit CPU fallback presented as a GPU result.

Each raises `NotImplementedError` at a public boundary rather than applying an
approximation. **project decision**

## Correctness gate fixed before oracle experiments

Every oracle comparison uses identical counts, ordered names, and a persisted selection
manifest. Metrics are computed on finite paired entries; theta is compared in log10 space
for non-Poisson genes. Initial alpha acceptance thresholds are fixed as follows and may be
changed only prospectively in a new PR:

| Quantity | Required alpha gate |
|---|---|
| fixed slope | max absolute error `<= 1e-12` |
| Poisson-gene set | Jaccard `= 1.0` |
| raw and regularized intercept | MAE `<= 0.10`, Pearson `>= 0.995` |
| raw and regularized log10(theta) | MAE `<= 0.25`, Spearman `>= 0.90` |
| clipped Pearson residual | RMSE `<= 0.15`, Pearson `>= 0.99` |
| residual variance | Spearman `>= 0.98` |
| HVG top 1000/2000/3000 | Jaccard `>= 0.90` when the fixture contains at least K eligible genes |

MAE, RMSE, maximum absolute error, Pearson, and Spearman are always reported even when a
gate uses only a subset. A failing gate is investigated in the prescribed order: model
branch, selection, theta, regularization, optimizer, clipping, orientation, dtype, then
floating-point order. **project decision**
