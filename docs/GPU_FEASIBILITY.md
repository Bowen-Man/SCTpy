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

cProfile of the validated path recorded 3,091 per-gene fits, 9,262 Brent root solves, and
53,675 Cox--Reid score evaluations. The score function is the largest project-owned
cumulative call path. It operates on one gene's selected-cell vector per evaluation, so
an isolated experiment must first determine whether genes can be grouped without changing
independent convergence semantics. Arithmetic intensity, root divergence, launch count,
and the cost of moving sparse count rows remain **unknown**. No GPU implementation is
justified until a batched CPU baseline and one transfer-inclusive microbenchmark answer
those questions.

## Interpretation boundary

Even a fast result for this primitive would not establish an end-to-end GPU speedup. The
historical SJ selector invokes the functional repeatedly through a root solve, while the
current path performs thousands of small Cox--Reid root solves. An implementation candidate
must show that saved compute exceeds transfers, launch costs, and additional working memory
at a realistic gene count. For the current path this applies to Cox--Reid fitting, not the
already measured SJ primitive.

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
