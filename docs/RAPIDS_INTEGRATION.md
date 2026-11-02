# RAPIDS and GPU boundary

There is currently no package GPU backend. CPU installation and import do not require or
import CUDA packages, and the AnnData adapter rejects CUDA-backed counts instead of
performing a hidden device transfer.

## Ecosystem contract

- AnnData uses cells by genes; the numerical core uses genes by cells.
- rapids-singlecell exposes explicit AnnData CPU/GPU transfer helpers. Its analytic
  Pearson-residual normalization uses a shared supplied theta and is not complete SCT v2.
- CuPy/SciPy sparse conversion transfers data and is never treated as implicit.
- RMM may later supply a CuPy allocator, but allocator and device selection remain caller
  decisions.
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

GPU dependencies will be published only when a GPU implementation enters the package.
Until then, the local CUDA development environment is intentionally not package metadata.
