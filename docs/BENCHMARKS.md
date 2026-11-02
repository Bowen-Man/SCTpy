# CPU benchmark and profiling report

Status: measured on 2026-11-02 on Windows 11, Python 3.13.7, AMD Ryzen 9 9950X3D
(16 cores/32 logical processors), approximately 93.6 GiB RAM, NumPy linked to
OpenBLAS 0.3.30. These are fixture-and-machine results, not general performance claims.

## Protocol

`benchmarks/run_baseline.py` launches a fresh process for every run, performs one warm-up
and five measured runs, fixes `OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`, `BLIS_NUM_THREADS`, and `NUMEXPR_NUM_THREADS` to 1, samples process
RSS every 5 ms, and enforces per-run time and memory budgets. The timed runs do not enable
`tracemalloc`; its overhead is measured separately by `profile_stages.py` and its value is
treated as Python-only supplementary evidence.

Despite the thread environment, `psutil` observed a maximum of 8 process threads. Those
may include runtime/service threads, so the report records both configured and observed
values and does not label the measurements as strictly single-threaded.

## Baseline

Input is CSR; the retained residual output is dense float64.

| Fixture | Median compute wall | Median process CPU | Median peak RSS | Median RSS growth | Budget |
|---|---:|---:|---:|---:|---|
| 800 x 240 | 1.001 s | 1.000 s | 87.66 MiB | 17.98 MiB | 60 s / 2 GiB |
| 3200 x 240 | 8.417 s | 8.500 s | 152.69 MiB | 83.07 MiB | 180 s / 4 GiB |

Median uninstrumented stage timers:

| Fixture | Fit | Regularization | Residuals + HVG | Total |
|---|---:|---:|---:|---:|
| 800 x 240 | 0.578 s | 0.417 s | 0.0024 s | 1.000 s |
| 3200 x 240 | 2.481 s | 5.860 s | 0.0222 s | 8.416 s |

For the larger fixture, exact Sheather--Jones bandwidth selection and smoothing account
for about 69.6% of stage time; independent per-gene NB fitting accounts for about 29.5%.
Residual construction is not the dominant hotspot on this fixture.

## Memory and object lifetimes

For the larger fixture, the CSR input is 6.64 MiB (`578,937` nonzeros) and the dense
residual output is 5.86 MiB. A 256 x 240 residual block is 0.469 MiB per float64 array;
roughly five block-sized arrays can coexist and are released after each gene block.

The SJ derivative calculation is deliberately blocked at 256 training rows. Its pairwise
temporary is about 256 x 3,000 float64 values (roughly 5.9 MiB per array at the upper end),
with several ufunc intermediates alive together; each block is released before advancing.
The Gaussian smoother similarly creates bounded prediction-by-training weight blocks.
Per-gene parameter vectors are only 25 KiB each at 3,200 genes and live through result
construction.

The instrumented larger run reported 155.3 MiB sampled peak RSS and 74.7 MiB
`tracemalloc` peak. The latter excludes many NumPy/SciPy native allocations and must not be
read as total memory. cProfile identified the 3,091 independent SciPy L-BFGS-B calls and
the NB likelihood/gradient as the main fitting call path; call tracing substantially
inflates optimizer time, so performance percentages above come from the uninstrumented
stage medians.

## GPU decision from evidence

The evidence still does not justify a custom CUDA or Triton kernel. The isolated CuPy
microbenchmark selected the SJ fourth-derivative functional because regularization is the
dominant stage. Across two controlled invocations it was 53.3--59.4x faster than the CPU
primitive including H2D and scalar D2H, with float64 relative error `2.20e-16` and a 30.4
MiB estimated live block working set. This supports a future CuPy prototype for that
primitive, not an end-to-end SCT claim. See [GPU_FEASIBILITY.md](GPU_FEASIBILITY.md).

The next evidence steps are Gaussian smoothing and repeated root-solver integration,
followed by batched NB likelihood/gradient evaluation without assuming a GPU optimizer.
Residual calculation should be reconsidered only on a fixture where profiling makes it
material. The package continues to execute all production paths on CPU; CUDA remains an
optional benchmark dependency.

## Reproduction

```bash
python -m benchmarks.run_baseline --fixture benchmark-results/large/fixture.npz --manifest benchmark-results/large/fixture.selection.json --output benchmark-results/large/baseline.json --warmups 1 --repeats 5 --threads 1 --time-budget-seconds 180 --memory-budget-mib 4096
python -m benchmarks.profile_stages --fixture benchmark-results/large/fixture.npz --manifest benchmark-results/large/fixture.selection.json --output benchmark-results/large/profile.json --profile benchmark-results/large/profile.prof
```
