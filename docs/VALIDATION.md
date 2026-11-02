# Validation and profiling

The package uses three independent levels on identical counts, ordered names, and a
persisted selection manifest:

- **Level A:** external R 4.5.2 with `sctransform` 0.4.3, `glmGamPoi` 1.22.0, and
  `matrixStats` 1.5.0;
- **Level B:** `Bowen-Man/sctransform-cpu` v0.4.0 at
  `b79041796e3f7ca3b33029adde5d7132eedd2430`, called through its public API;
- **Level C:** this repository.

Levels A and B are black-box execution-time oracles. They are not package, build, or CI
dependencies. Generated reports and downloaded data remain under ignored
`benchmark-results/`.

## Pre-registered gates

| Quantity | Requirement |
|---|---|
| fixed slope | max absolute error `<= 1e-12` |
| Poisson-gene set | Jaccard `= 1.0` |
| raw/final intercept | MAE `<= 0.10`, Pearson `>= 0.995` |
| raw/final log10(theta) | MAE `<= 0.25`, Spearman `>= 0.90` |
| clipped residual | RMSE `<= 0.15`, Pearson `>= 0.99` |
| residual variance | Spearman `>= 0.98` |
| HVG Top 1000/2000/3000 | Jaccard `>= 0.90` when defined |

All reports also contain MAE, RMSE, maximum absolute error, Pearson, and Spearman where
defined. These thresholds live in `benchmarks/correctness.py` and may change only
prospectively.

The committed 49 KiB R golden contains only factual numerical outputs for the synthetic
small fixture, plus provenance. CI regenerates the counts, verifies their hash, runs the
CPU path on CSR input, and requires exact branch masks and HVG order. Its tighter
cross-platform regression limits are:

| Quantity | RMSE | Max absolute error |
|---|---:|---:|
| raw intercept | `1e-7` | `1e-6` |
| log10 raw theta | `1e-5` | `1e-3` |
| regularized intercept | `1e-4` | `1e-3` |
| log10 regularized theta | `1e-6` | `1e-5` |
| residual mean/variance | `1e-4` | `1e-3` |

## Measured results

Synthetic fixtures use seed `20261102`. On the 3,200 x 240 fixture, Level C matched
Level A's finite/Inf theta and parameter-outlier masks and had exact HVG Top
1000/2000/3000 sets. Representative RMSE values were `3.88e-10` for raw intercept,
`2.36e-6` for finite raw log10(theta), `4.94e-6` for regularized intercept, `1.55e-8`
for regularized log10(theta), `2.56e-6` for residuals, and `1.73e-6` for residual
variance. **measured on 2026-11-03**

The licensed real-data gate derives 3,000 cells from 10x Genomics' *10k Human PBMCs, 3'
v3.1, Chromium X* dataset (CC BY 4.0). The source file is 39,503,734 bytes with SHA-256
`42a789ff3f92f0e117d1ff4ffcb2fa8edee87dd2a03d83a81502709344a38bf0`; it is downloaded,
verified, and never committed. The derived sparse-count SHA-256 is
`d9c23eda8654b17fab1a3672df4961b53f2d9e63356dd631de567c76db570532`.

With the oracle's ordered 2,000-gene/2,000-cell manifest, Level C matched 142 raw-infinite
theta names, 159 outlier names, and all three HVG sets exactly. Residual RMSE was
`1.77e-5`, residual-variance RMSE `2.20e-5`, and residual-variance Spearman
`0.9999999952`. Selection order is part of the contract because reordering changes
500-gene bin membership. **measured on 2026-11-03**

## Report and benchmark contracts

`schemas/correctness-report-v1.schema.json` defines six required stages: `input`,
`selection`, `raw_fitting`, `regularization`, `residual`, and `hvg`. Readers reject
unknown versions and missing stages but allow additive fields. Breaking field changes
require a new schema version.

`run_baseline.py` performs one warm-up and five fresh-process runs, fixes common BLAS
thread variables, samples RSS, and enforces time/memory budgets. `profile_stages.py`
records stage time, shapes, dtype, bytes, `nnz`, lifetimes, RSS, cProfile output, and
supplementary Python-only `tracemalloc` data. The fixed 3,200 x 240 validated path measured
`2.961 s` median compute time, `3.000 s` process CPU time, and `110.07 MiB` peak RSS on
the recorded Windows machine. These are fixture/machine measurements, not general claims.

Minimal synthetic reproduction:

```bash
python -m benchmarks.generate_fixture --output benchmark-results/large/fixture.npz --genes 3200 --cells 240 --seed 20261102
Rscript benchmarks/export_r_oracle.R benchmark-results/large/fixture benchmark-results/large/r-oracle
python -m benchmarks.export_reference --fixture benchmark-results/large/fixture.npz --manifest benchmark-results/large/fixture.selection.json --output benchmark-results/large/reference-v0.4.0.npz
python -m benchmarks.compare_correctness --fixture benchmark-results/large/fixture.npz --manifest benchmark-results/large/fixture.selection.json --r-oracle benchmark-results/large/r-oracle --reference benchmark-results/large/reference-v0.4.0.npz --output benchmark-results/large/correctness.json --enforce
```

The scheduled oracle workflow and manually triggered real-data workflow preserve full
commands and upload only compact diagnostics. Unit tests separately cover dense/CSR
parity, all-zero genes, counts above 32-bit range, extreme library sizes, CLI replay, and
the versioned report schema.
