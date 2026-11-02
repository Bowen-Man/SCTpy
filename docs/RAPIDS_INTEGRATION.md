# RAPIDS and GPU integration boundary

Status: CPU/AnnData boundary implemented; isolated CuPy feasibility benchmark only. There
is no package GPU backend and no implicit CPU fallback presented as GPU execution.

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
The implemented protocol in `sctransform_native.backend` is intentionally small:

```text
ArrayBackend
  name: backend identifier
  device(array) -> explicit device descriptor
  asarray(array, *, dtype, copy) -> same-device array
  to_device(array, device) -> explicit transfer result + transfer metadata
  synchronize() -> timing boundary
```

Sparse input remains sparse for validation, library statistics, and reductions. Any dense
gene-by-cell residual block is allocated only by an explicit `residual_block_size` policy.
No adapter may move data as a side effect of a numerical operation. **project decision**

`fit_transform_anndata` consumes `X` or one named layer in AnnData's cells-by-genes
orientation and does not mutate AnnData. It rejects CUDA-backed data with
`ImplicitDeviceTransferError`. `infer_device` detects CuPy/cupyx without importing either
package, so a CPU import remains CUDA-free. **implemented and tested**

## Profile evidence and feasibility boundary

The CPU profile reports stage wall/process time, shapes, dtypes, bytes, `nnz`, object
lifetimes, RSS, and Python-only `tracemalloc` as a supplement. On the measured 3200 x 240
fixture, regularization is 69.6% of uninstrumented stage time, NB fitting 29.5%, and
residual/HVG calculation about 0.3%. **measured**

The first isolated GPU target is therefore one diagonal-in fourth Gaussian derivative
functional used by the SJ selector, not the residual path. See
[GPU_FEASIBILITY.md](GPU_FEASIBILITY.md). Later candidates, not commitments, are batched
likelihood/gradient evaluation and, only on materially larger cell counts, dense residual
arithmetic or streamed residual variance. Scalar optimizer control flow and SciPy-only
routines may remain on CPU. **inferred**

For a 12 GB device, feasibility is evaluated against the peak **live** working set, not the
final array alone. H2D and D2H time are included in end-to-end GPU results and reported
separately. Chunking is introduced only when measured working set or sparse-index support
requires it. Async copies, double buffering, Triton, persistent kernels, and custom CUDA
are not candidates without further profile evidence. **project decision**

## Optional dependency rule

CPU imports never import CUDA modules. The project exposes versioned optional extras,
`gpu-cuda12` and `gpu-cuda13`, because CuPy wheel choice is tied to the CUDA major version.
A missing driver or optional package produces a clear unavailable-backend error. A skipped
GPU test is reported as skipped; it never executes NumPy and calls that a GPU pass. RMM
remains an allocator option for a future RAPIDS backend; the microbenchmark does not
require or silently configure it. **project decision**
