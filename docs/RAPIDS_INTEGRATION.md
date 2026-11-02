# RAPIDS and GPU integration boundary

Status: design only. No GPU kernel is authorized until CPU correctness and stage profiling
identify a worthwhile target.

## Current ecosystem behavior

- AnnData stores observations (cells) by variables (genes) and supports NumPy/SciPy plus
  dense and CSR/CSC CuPy objects. AnnData 0.14 development documentation rejects other
  CuPy sparse formats for `X`. **confirmed from official documentation**
- rapids-singlecell uses AnnData directly and exposes explicit
  `get.anndata_to_GPU(...)` / `get.anndata_to_CPU(...)` transfers. Its analytic Pearson
  residual routine is not SCT v2. **confirmed from official documentation**
- CuPy 14.2 recommends sparse arrays for new code, supports CSR/CSC/COO/DIA with a limited
  data-dtype set, and documents that SciPy/CuPy sparse conversion is not implicit and
  incurs transfer cost. **confirmed from official documentation**
- RMM can provide CuPy's allocator via `rmm.allocators.cupy.rmm_cupy_allocator`; allocator
  selection is independent of CUDA device selection. **confirmed from official documentation**
- ScaleSC demonstrates cell-axis chunking and explicit batch movement to avoid device
  working-set and sparse-index limits. That is a data-flow pattern, not an SCT algorithm.
  **confirmed from official documentation; inferred**

## Narrow backend protocol

The numerical core owns genes-by-cells logical blocks. Adapters own external orientation.
The planned boundary is intentionally small:

```text
ArrayBackend
  name: "numpy" | "cupy"
  device(array) -> explicit device descriptor
  asarray(array, *, dtype, copy) -> same-device array
  to_device(array, device) -> explicit transfer result + transfer metadata
  synchronize() -> timing boundary
```

Sparse input remains sparse for validation, library statistics, and reductions. Any dense
gene-by-cell residual block is allocated only by an explicit `residual_block_size` policy.
No adapter may move data as a side effect of a numerical operation. **project decision**

## Feasibility questions and required measurements

Before a GPU microbenchmark, CPU profiling must report per stage:

- wall and process CPU time;
- input/output and peak-live shapes, dtype, bytes, and `nnz`;
- whether operations are sparse reductions, per-gene scalar optimization, or dense
  elementwise residual arithmetic;
- residual materialization policy and peak RSS;
- estimated arithmetic intensity and transfer bytes.

For a 12 GB device, feasibility is evaluated against the peak **live** working set, not the
final array alone. Host-to-device and device-to-host time are included in end-to-end GPU
results and reported separately. Chunking is introduced only when the measured working set
or sparse-index support requires it. Async copies, double buffering, Triton, persistent
kernels, and custom CUDA are not candidates without profile evidence. **project decision**

Likely candidates, not commitments, are dense residual arithmetic, blockwise residual
variance, sparse column/row reductions, and batched likelihood evaluation. Scalar optimizer
control flow and SciPy-only routines may remain on CPU. **inferred**

## Optional dependency rule

CPU imports must never import CUDA modules. A future extra may expose CuPy support, but a
missing driver or optional package must produce a clear unavailable-backend error. A skipped
GPU test is reported as skipped; it must not execute the NumPy backend and call that a GPU
pass. **project decision**
