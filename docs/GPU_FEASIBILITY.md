# GPU feasibility

Status: narrow feasibility experiment, not an SCT GPU backend.

## Why this operation was measured

The 3200 genes x 240 cells CPU benchmark showed median stage time of 5.860 seconds for
regularization (69.6%), 2.481 seconds for per-gene NB fitting (29.5%), and 0.022 seconds for
residuals plus HVG ranking. Moving residual arithmetic first would therefore optimize a
non-hot stage on this fixture.

`benchmarks/gpu_sj_functional.py` isolates one profiler-selected operation: the
diagonal-in fourth Gaussian derivative functional used by the Sheather--Jones bandwidth
selector. Both paths use float64, identical blocking, and the same mathematical expression.
The GPU path is ordinary CuPy elementwise operations plus reductions; it contains no
custom CUDA, Triton, persistent kernel, asynchronous copy, or double buffering.

This selection was valid for the provisional `nb_mle/native` path. Reprofiling the
validated `cox_reid/gene_bin_500/sct_v2/r_stats/clamp/moment_ratio` path changes the
decision: on the same larger fixture, median fit time is 4.993 seconds, regularization is
0.134 seconds, and residuals plus HVG ranking are 0.026 seconds. The SJ primitive is
therefore no longer the primary migration candidate. **measured on 2026-11-02**

## Measurement contract

The report includes:

- CPU median wall time after warm-up;
- GPU kernel time measured with CUDA events;
- H2D, scalar D2H, and full H2D + compute + D2H wall time separately;
- transferred bytes, estimated peak-live block bytes, pool allocation, and device memory;
- absolute/relative error with a pre-set float64 relative tolerance of `1e-10`.

The default block is 256 x N training values. Five float64 intermediates are budgeted,
and execution stops if that estimate exceeds the explicit device-memory budget (12 GiB by
default). This benchmark deliberately transfers only the one-dimensional regularization
covariate. A full backend would need to account for counts, parameters, and output movement
separately.

## Current measurement

Measured on 2026-11-02 with an NVIDIA GeForce RTX 5070, 12,227 MiB reported VRAM, CuPy
14.2.0, CUDA runtime 12.9, and CUDA driver API 13.1. One warm-up and five measured runs used
the 3,113 eligible values from the fixed larger fixture, block size 256, float64, and all
recorded CPU thread environment variables set to 1.

| Measurement | Median / value |
|---|---:|
| CPU functional | 0.133215 s |
| GPU kernel (CUDA events) | 0.002245 s |
| GPU H2D + compute + D2H | 0.002262 s |
| one-time H2D, 24,904 bytes | 0.000664 s |
| one-time scalar D2H, 8 bytes | 0.000092 s |
| CPU / GPU-kernel ratio | 59.4x |
| CPU / GPU-end-to-end ratio | 58.9x |
| estimated live block working set | 30.4 MiB |
| CuPy memory-pool allocation | 48.7 MiB |
| relative numerical error | 2.20e-16 (pass) |

An independent repeat with the same settings produced 0.118693 seconds CPU, 0.002202
seconds GPU kernel, and 0.002227 seconds GPU end-to-end. Across the two controlled
invocations, the observed ratio is therefore 53.3--59.4x rather than a single universal
number.

This is evidence that the isolated regularization primitive maps efficiently to CuPy. It
is not evidence for a 58.9x SCT speedup and, after validated-path profiling, is not the
next end-to-end optimization target. Results are specific to this fixture and machine.

## Current candidate

The CPU batching prerequisite is complete. On the fixed 3200 x 240 fixture, a 64-gene
block reduced median compute time from 4.502 s to 2.961 s (`1.52x`) and fit time from
4.301 s to 2.767 s (`1.55x`). Median sampled peak RSS increased from 108.59 MiB to
110.07 MiB. Scalar and batched finite/Inf theta and parameter-outlier masks were exact;
finite raw log10(theta) RMSE was `8.21e-10`, residual RMSE was `1.18e-13`, and Top
1000/2000/3000 HVG order was exact. **measured on 2026-11-03**

The narrow GPU candidate is now the initial Cox--Reid dispersion score and safeguarded
root iteration for one dense genes-by-selected-cells block. It performs repeated log,
digamma, elementwise arithmetic, and cell-axis reductions while the block is resident.
Sparse traversal, preliminary/final intercept roots, the 500-gene trend, regularization,
and result construction remain CPU candidates. This is a hybrid boundary, not a whole
pipeline migration. **project decision based on profiling**

At 64 x 240, one float64 block is 0.117 MiB and the eight-array workspace estimate is
0.94 MiB. At the sparse PBMC gate's 3,000 selected cells, those values become 1.46 MiB and
11.72 MiB. Its 14,004 initial-fit candidates would require about 219 blocks and roughly
321 MiB of count H2D traffic if every block moves once and remains resident for its root
iterations. The full 36,601 x 3,000 dense residual result is about 838 MiB, so it fits a
12 GiB device but is not part of the first microbenchmark. **inferred from measured shapes**

The isolated experiment must report CPU, kernel, H2D, required scalar D2H, and complete
transfer-inclusive time. It must also record active-root counts per iteration and device
pool memory. Async copies or double buffering are not assumed useful until a synchronous
baseline shows transfer exposure. Triton or custom CUDA remains unjustified.

## Interpretation boundary

Even a fast result for this primitive would not establish an end-to-end GPU speedup. The
historical SJ selector invokes the functional repeatedly through a root solve, while the
current path performs repeated Cox--Reid block evaluations with a shrinking active-root
mask. An implementation candidate must show that saved compute exceeds transfers, launch
costs, divergence, and additional working memory at a realistic gene and cell count. For
the current path this applies to initial Cox--Reid fitting, not the already measured SJ
primitive.

RAPIDS-singlecell is not used as an SCT oracle: its documented Pearson-residual
normalization uses one analytic shared theta and does not implement the v2 fixed-slope,
per-gene theta, Poisson, regularization, and clipping contract.

## Reproduction

```bash
python -m pip install -e ".[benchmark,gpu-cuda12]"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -m benchmarks.gpu_sj_functional --fixture benchmark-results/large/fixture.npz --output benchmark-results/large/gpu-sj.json --warmups 1 --repeats 5 --block-size 256 --memory-budget-gib 12
```

Use `gpu-cuda13` instead when the installed CUDA major version is 13.
