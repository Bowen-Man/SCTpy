# Current state assessment

Audit date: **2026-11-03**. Pre-GPU audit baseline:
`91016da` on a clean `main` branch. Its GitHub Actions
run completed successfully on Windows and Linux with Python 3.10--3.13. The pre-GPU
validation work adds pinned macOS 15 x64 coverage on Python 3.11/3.13 and a frozen
small-fixture compatibility gate to that matrix.

## Package structure and dependencies

- `core.py` owns configuration, stage orchestration, result state, residuals, and HVG
  ranking.
- `_fit.py` contains the native NB MLE research estimator and the validated Cox--Reid
  offset estimator.
- `_regularize.py` contains the native paper-derived smoother and the independently
  implemented R-stats compatibility path.
- `selection.py` owns the explicit PCG64 cell/gene manifest and sparse-aware v2 selection.
- `anndata.py` is a non-mutating CPU adapter; `backend.py` defines only an explicit future
  device/transfer boundary.

Runtime dependencies are NumPy and SciPy. AnnData, psutil, and CuPy are optional extras.
The project ships no R runtime, reference binary, or project-owned compiled extension.

## Statistical paths

Plain `SCTConfig()` remains the original native research path:

```text
nb_mle / native / native / extrapolate / none
```

`SCTConfig.r_v2()` is the formal compatibility preset:

```text
cox_reid / gene_bin_500 / sct_v2 / r_stats / clamp / moment_ratio
```

Mixed paths are explicitly labelled `experimental`. Changing selection counts or the
residual block size does not change the statistical mode. The default was not silently
changed, preserving the alpha API.

## Independently reproduced correctness baseline

The final pre-GPU suite reports 85 passed tests and one explicit optional-CuPy skip.

| Metric vs R oracle | 800 x 240 | 3200 x 240 |
|---|---:|---:|
| raw intercept RMSE | 3.62e-4 | 4.24e-4 |
| raw log10(theta) RMSE | 6.20e-7 | 2.36e-6 |
| regularized intercept RMSE | 4.05e-5 | 4.16e-5 |
| regularized log10(theta) RMSE | 1.44e-8 | 1.55e-8 |
| residual RMSE | 8.52e-5 | 6.86e-5 |
| residual-variance RMSE | 5.97e-5 | 4.67e-5 |

Finite/Inf theta masks and parameter-outlier masks were exact on both fixtures. The larger
fixture had exact HVG Top 1000/2000/3000 sets. This confirms the existing residual and
regularization reinjection conclusions; those modules should not be rewritten without new
contrary evidence.

## Numerical gap classification

- **Resolved issue:** the R oracle applies the local dispersion trend inside gene bins of
  500. The compatibility preset now encodes this setting, reducing raw-intercept RMSE from
  `4.24e-4` to `3.88e-10` on the larger fixed fixture.
- **Falsified on the larger fixture:** preliminary offset iteration/stopping behavior is
  not the dominant gap. The A/B holds every preliminary and dispersion fit fixed, and the
  opt-in trace records score evaluations and solver termination directly.
- **Unverified concern:** synthetic Gamma--Poisson fixtures do not establish behavior on
  licensed real scRNA-seq data or very large cell counts.
- **Resolved on fixed fixtures:** final residual arithmetic, regularization from oracle raw
  parameters, theta finite/Inf branching, parameter outliers, and HVG sets.

## Performance and backend boundary

On the controlled larger fixture, 64-gene Cox--Reid batching reduces median compute time
from `4.502 s` to `2.961 s` (`1.52x`) and fit time from `4.301 s` to `2.767 s` (`1.55x`).
Median sampled peak RSS changes from `108.59 MiB` to `110.07 MiB`. The batch working-set
estimate is 0.94 MiB at 64 x 240, and every block is released before advancing.
Regularization and residual/HVG work are not current optimization targets.

Production execution is CPU NumPy/SciPy only. CUDA inputs are rejected at the AnnData
boundary rather than copied implicitly. CuPy currently appears only in an isolated,
optional microbenchmark; PyTorch is not a dependency or backend.

## Next correctness work

The differential report now exports a versioned stage record with input/selection
provenance, numeric masks, appropriate relative errors, convergence summaries, top-error
genes, and complete HVG rank agreement.

P0 repository audit, compatibility preset, stage diagnostics, offset-fit diagnosis, and
external fixed-fixture non-regression are complete. The pre-GPU audit also found and
repaired a stale fixture-generator call site after the sparse-aware selection API gained
explicit matrix and geometric-mean inputs. A CLI regression test now protects every
replay artifact.

The diagnostic JSON contract, numerical-extreme coverage, and licensed real-data source
with immutable provenance are now fixed. A daily isolated workflow reruns the larger
Level-A/Level-B/Level-C differential and both stage-reinjection checks without adding R or
the GPL oracle to package dependencies. The first sparse PBMC Level-A/B/C differential
also passed with exact named theta/outlier masks and exact HVG Top 1000/2000/3000 sets;
its full metrics and provenance are recorded in
[REAL_DATA_VALIDATION.md](REAL_DATA_VALIDATION.md).

The pre-GPU correctness, real-data, memory/workspace telemetry, and profiled CPU-batching
work are complete. Scalar/batched theta and outlier masks are exact on the fixed large
gate; residual RMSE between paths is `1.18e-13`, and Top 1000/2000/3000 HVG order is exact.
The batched sparse PBMC rerun also preserved exact named raw-infinite and outlier sets and
exact Top 1000/2000/3000 HVG sets against the R oracle.
The only justified next GPU work is an isolated, transfer-inclusive CuPy microbenchmark of
the initial 64-gene Cox--Reid score/root block. It is not yet a GPU backend, and custom
CUDA or Triton remains out of scope without that evidence.
