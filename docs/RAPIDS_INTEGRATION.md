# RAPIDS integration

One package provides a NumPy/SciPy core and an optional CuPy fitting backend. CPU
installation/import requires no CUDA. This is a hybrid implementation: CPU counts,
selection, regularization, residuals and outputs; GPU batched Cox--Reid roots.

## AnnData and device contract

- AnnData is cells by genes; the core is genes by cells.
- `fit_transform_anndata` returns a result without mutation. `write_anndata` writes
  standard HVG fields, model parameters, serializable metadata and optional result layers.
  Raw counts are preserved. Summary mode does not retain a dense residual matrix.
- CUDA-backed counts are rejected explicitly. Only selected internal fitting blocks
  transfer when `cox_reid_backend="cupy"`; there is no automatic backend selection.
- Device, stream and allocator belong to the caller. Package code never resets RMM.
  The benchmark owns its optional 512 MiB RMM pool.
- Residuals for excluded genes are NaN. Select eligible genes or HVGs for downstream PCA.
  Transfer a result layer explicitly with RAPIDS' `anndata_to_GPU` before GPU PCA.

These conventions follow the [AnnData API](https://anndata.readthedocs.io/en/stable/generated/anndata.AnnData.html)
and [RAPIDS usage principles](https://rapids-singlecell.scverse.org/en/latest/usage_principles.html).
**confirmed from official documentation; project decisions**

## Upstream alignment

The [contribution guide](https://rapids-singlecell.scverse.org/en/latest/contributing.html)
permits CuPy implementations and calls for style/type checks, tests with timeouts, and
explicit optional-dependency skips. We use Ruff, mypy, a 60-second pytest timeout, and
Windows/Linux Python 3.10--3.14 CI; CPU wheels require no GPU dependencies.
**confirmed from official documentation; implemented**

GPU regressions cover dense/CSR/CSC, full/summary and float32 residual output, corrected
counts, branch repair, caller allocators and non-default streams. The frozen R gate runs
on both backends when CUDA is available. On AnnData 0.13.4, Scanpy 1.12.5 and
rapids-singlecell 0.18.1, downstream CPU/GPU PCA explained-variance ratios differed by at
most `1.70e-7` on the 800 x 240 fixture. **measured on 2026-11-03**

An upstream API name, dependency arrangement and acceptance remain **unknown** until
maintainer discussion. This package does not claim the historical R API or a fully
device-resident RAPIDS implementation. CUDA 12 has an installation extra but was not
tested on this machine; CUDA 13 / CuPy 14.2 / RMM 26.10 was tested.

## Performance evidence

CPU profiling selected Cox--Reid score/root evaluation as the first GPU candidate.
The isolated 64/128/256-gene, 10,000-cell experiment measured 17.9--18.7x root-block
speedup and 44--176 MiB logical RMM peak usage, including transfers. Those figures are
not complete-pipeline speedups. No custom CUDA, Triton or asynchronous staging was added.

Public PBMC uses the licensed fixture and ordered 2,000-gene/2,000-cell manifest in
[VALIDATION.md](VALIDATION.md), 128-gene fit blocks, float64 and one BLAS/OpenMP thread.
On WSL, Ryzen 9 9950X3D, Python 3.13.17, NumPy 2.4.6, SciPy 1.18.1 and an RTX 5070 12 GB:

| Output | CPU full call | Hybrid full call | Ratio | CPU peak RSS | Hybrid peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Summary | 4.565 s | 2.448 s | 1.86x | 514 MiB | 934 MiB |
| Dense residuals | 4.867 s | 2.848 s | 1.71x | 1197 MiB | 1617 MiB |

These are medians of five measured calls after warmup, not cold-process latency.
Each worker warms the entire pipeline once; an additional complete worker is discarded.
Full-call timings include all numerical stages and internal transfers, but exclude file
loading, CUDA/RMM setup and warmup. Reports retain those costs separately, process CPU
time, all runs, RSS, thread counts, BLAS configuration and input hashes.
Timing configurations use `return_corrected_umi=False`; count reconstruction is
validated separately in the differential gate.
For summary mode with zero per-process warmups, median fresh-process latency was
4.811 s CPU / 3.424 s hybrid (1.41x), including interpreter startup, input loading and
GPU setup. CUDA kernels had an existing disk cache; this is not a first-install run.
RMM logical peak was 17.6 MiB within a 512 MiB pool; host RSS includes CUDA runtime
overhead. Full residual output remains a CPU dense matrix. **measured on 2026-11-03**

The local 10,319-cell x 36,601-gene CSR dataset used 2,000 selected genes and 10,000
selected cells (PCG64 seed 20261103). With the same repeat/thread protocol and summary
output, full-call medians were 25.618 s CPU and 6.254 s hybrid (4.10x). Median peak RSS
was 2036/2495 MiB; logical RMM peak was 87.9 MiB. CPU/GPU branch masks and full HVG order
were exact and parameter/residual-summary errors passed the pre-registered GPU limits.
Per-run budgets were 120 seconds and 6 GiB RSS. This supersedes the earlier single-call
4.05x observation. Local counts, identifiers and reports are not committed.

The GPU boundary therefore stays narrow and explicit. Bounded fitting blocks fit the
12 GB device; moving all dense intermediates at once is unnecessary. Async copies,
double buffering and new kernels require further profile evidence.
On public summary runs, fitting still accounts for about 85% of CPU call time and 73%
of hybrid call time. The corrected cProfile call tree identifies sparse row extraction,
offset-intercept refits and smoothing as remaining CPU work. These traces include
instrumentation overhead and do not justify migrating every stage to the GPU.
