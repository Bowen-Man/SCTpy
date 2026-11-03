# RAPIDS and GPU boundary

The package has one opt-in GPU operation: batched Cox--Reid dispersion roots. CPU
installation and import do not require or import CUDA packages, and the AnnData adapter
rejects CUDA-backed counts instead of performing a hidden device transfer.

## Ecosystem contract

- AnnData uses cells by genes; the numerical core uses genes by cells.
- rapids-singlecell exposes explicit AnnData CPU/GPU transfer helpers. Its analytic
  Pearson-residual normalization uses a shared supplied theta and is not complete SCT v2.
- CuPy/SciPy sparse conversion transfers data and is never treated as implicit.
- RMM may supply the active CuPy allocator, but allocator, stream, and device selection
  remain caller decisions.
- ScaleSC's relevant contribution here is explicit movement and chunked data flow, not an
  SCT algorithm.

These points are **confirmed from official documentation**; their use as design
constraints is a **project decision**.

## Profile-selected candidate

On the fixed 3,200 x 240 fixture, grouping 64 independent Cox--Reid score evaluations
reduced median CPU compute time from `4.502 s` to `2.961 s` and fit time from `4.301 s` to
`2.767 s`. Peak RSS changed from `108.59 MiB` to `110.07 MiB`. Scalar and batched paths
had exact finite/Inf theta and outlier masks, residual RMSE `1.18e-13`, and identical HVG
Top 1000/2000/3000 order. **measured on 2026-11-03**

The first justified GPU experiment is therefore the initial Cox--Reid score/root block,
not residual construction. On the 2,000-cell PBMC fit manifest, one 64 x 2,000 float64
array is about `0.98 MiB`; an eight-array block is about `7.81 MiB`. The complete sparse
count transfer is about `30.5 MiB`, while a full dense residual output would be about
`838 MiB`. **measured/estimated from recorded shapes**

An isolated experiment must report H2D, kernel, D2H, synchronization, active roots, and
memory-pool usage separately. It starts synchronously and compares end-to-end time with
the CPU block. Chunking, async copies, double buffering, Triton, persistent kernels, and
custom CUDA require new profile evidence. SciPy-controlled scalar roots may remain on
CPU.

## Isolated GPU result

The pre-registered gate requires exact Poisson/ceiling masks, log10-theta RMSE at most
`1e-6`, and maximum absolute error at most `1e-4`. On a local 10,319 x 36,601 sparse
raw-count H5AD, a fixed seed selected 10,000 cells and the most overdispersed genes.
One warm-up and five float64 runs used CuPy 14.2, RMM 26.10, a 512 MiB pool, an RTX
5070 12 GB, and OMP/MKL/OpenBLAS thread limits of one. **measured on 2026-11-03**

| Genes | CPU root | H2D | GPU kernel | D2H | GPU end-to-end | Speedup | RMM peak |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 1.1407 s | 0.0009 s | 0.0599 s | 0.0002 s | 0.0611 s | 18.66x | 43.96 MiB |
| 128 | 1.7742 s | 0.0016 s | 0.0950 s | 0.0002 s | 0.0971 s | 18.28x | 87.91 MiB |
| 256 | 3.6374 s | 0.0030 s | 0.1998 s | 0.0002 s | 0.2031 s | 17.91x | 175.82 MiB |

All branch masks and expansion counts matched; worst log10-theta RMSE was `1.08e-14`.
The score/root block is therefore justified as the first package GPU candidate. This is
not an end-to-end SCT speed claim. The kernel dominates, transfers do not yet justify
asynchrony, and temporary memory grows approximately with block size. Extrapolating the
measured peak to all 17,387 candidate genes gives about 11.7 GiB before runtime and pool
overhead, leaving no safe 12 GB headroom. A backend must therefore retain CPU preliminary
fitting and process bounded gene blocks. The benchmark owns its RMM pool; library code
must not replace a caller's allocator.

## Package backend result

`SCTConfig.r_v2(cox_reid_backend="cupy")` transfers only each dense fitting block and
returns ordinary CPU results. It does not accept device-resident AnnData, initialize RMM,
or move regularization and residual stages. CuPy is confined to the optional `gpu`
(CUDA 13) and `gpu-cu12` extras; RAPIDS and RMM are not runtime dependencies. Two CPU
score checks preserve exact floor/ceiling branch decisions, with CPU roots used only for
disagreements.

On the same local H5AD, a shared manifest selected 2,000 genes and 10,000 cells, with
128-gene blocks and streamed residual summaries. CPU and CuPy fit stages took `21.926 s`
and `3.632 s`; full calls took `24.931 s` and `6.161 s` (6.04x fit, 4.05x end-to-end).
Finite/Inf theta, theta-postprocess, and outlier masks and HVG order were exact; raw
log10-theta RMSE was `1.82e-8`, regularized log10-theta RMSE was `1.89e-13`, and
residual-variance RMSE was `2.61e-14`.
On the 3,200 x 240 fixture the GPU fit was slower (`0.272 s` versus `0.092 s`), so the
backend remains explicit rather than automatic. **measured on 2026-11-03**
