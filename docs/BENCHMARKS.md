# CPU benchmark and profiling report

Status: measured on 2026-11-03 on Windows 11, Python 3.13.7, AMD Ryzen 9 9950X3D
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

## Baselines

Input is CSR; the retained residual output is dense float64.

### Validated v2 path

The current correctness path is explicit:

```text
cox_reid / gene_bin_500 / sct_v2 / r_stats / clamp / moment_ratio
```

| Fixture | Median compute wall | Median process CPU | Median peak RSS | Median RSS growth | Budget |
|---|---:|---:|---:|---:|---|
| 800 x 240 | 0.728 s | 0.734 s | 81.07 MiB | 10.01 MiB | 60 s / 2 GiB |
| 3200 x 240 | 2.961 s | 3.000 s | 110.07 MiB | 39.09 MiB | 180 s / 4 GiB |

Median uninstrumented stage timers:

| Fixture | Fit | Regularization | Residuals + HVG | Total |
|---|---:|---:|---:|---:|
| 800 x 240 | 0.686 s | 0.0320 s | 0.0025 s | 0.727 s |
| 3200 x 240 | 2.767 s | 0.1466 s | 0.0162 s | 2.960 s |

In a same-session A/B on the larger fixture, scalar initial fitting had median compute and
fit times of 4.502 s and 4.301 s. Grouping 64 independent score evaluations reduced them
to 2.961 s and 2.767 s: `1.52x` end-to-end and `1.55x` for the fit stage. Median sampled
peak RSS changed from 108.59 MiB to 110.07 MiB (`+1.4%`). **measured**

The optimized fit remains about 93.5% of stage time; regularization is about 5.0%, and
residual construction plus HVG ranking is about 0.5%. cProfile recorded 49 batched initial
fits and 2,450 batched score calls. Preliminary and final intercepts deliberately retain
their independent scalar roots. Call tracing raised total stage time to 5.72 seconds, so
these counts identify structure but the percentages above come from uninstrumented runs.
**measured**

The instrumented larger run reported 118.37 MiB sampled peak RSS, 29.94 MiB RSS growth,
and 32.90 MiB `tracemalloc` peak. The latter remains a Python-only supplement.

### Provisional alpha-default path

The earlier `nb_mle / native / native / extrapolate / none` measurements are retained for
comparison, not used to choose the next optimization target.

| Fixture | Median compute wall | Median process CPU | Median peak RSS | Median RSS growth | Budget |
|---|---:|---:|---:|---:|---|
| 800 x 240 | 1.001 s | 1.000 s | 87.66 MiB | 17.98 MiB | 60 s / 2 GiB |
| 3200 x 240 | 8.417 s | 8.500 s | 152.69 MiB | 83.07 MiB | 180 s / 4 GiB |

Median uninstrumented stage timers:

| Fixture | Fit | Regularization | Residuals + HVG | Total |
|---|---:|---:|---:|---:|
| 800 x 240 | 0.578 s | 0.417 s | 0.0024 s | 1.000 s |
| 3200 x 240 | 2.481 s | 5.860 s | 0.0222 s | 8.416 s |

For this earlier path, exact Sheather--Jones bandwidth selection and smoothing account
for about 69.6% of stage time; independent per-gene NB fitting accounts for about 29.5%.
Residual construction is not the dominant hotspot on this fixture.

## Memory and object lifetimes

For the larger fixture, the CSR input is 6.64 MiB (`578,937` nonzeros) and the dense
residual output is 5.86 MiB. A 256 x 240 residual block is 0.469 MiB per float64 array;
roughly five block-sized arrays can coexist and are released after each gene block.

In the validated path, the initial Cox--Reid solve densifies at most 64 selected gene rows.
A 64 x 240 float64 array is 0.117 MiB; the profiler budgets eight simultaneous arrays, or
0.94 MiB, and releases the block before advancing. Preliminary and final intercepts still
operate on one row. No dense genes-by-genes fitting workspace is retained. The measured
39.09 MiB median RSS growth includes these blocks, SciPy state, and the dense result.

In the provisional native path, the SJ derivative calculation is blocked at 256 training
rows. Its pairwise temporary is about 256 x 3,000 float64 values (roughly 5.9 MiB per array
at the upper end),
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

The evidence still does not justify a custom CUDA or Triton kernel. It does establish a
narrow candidate for an isolated CuPy experiment: the 64-gene Cox--Reid score block and
safeguarded dispersion iterations. The scalar/batched correctness A/B preserved masks and
HVG order, while the CPU experiment demonstrated useful grouping. Any GPU experiment must
still measure H2D, kernel, D2H, divergence, and the 12 GiB working-set limit explicitly.
Residual calculation remains out of scope until profiling makes it material. See
[GPU_FEASIBILITY.md](GPU_FEASIBILITY.md).

## Reproduction

```bash
python -m benchmarks.run_baseline --fixture benchmark-results/large/fixture.npz --manifest benchmark-results/large/fixture.selection.json --output benchmark-results/large/baseline-v2.json --warmups 1 --repeats 5 --threads 1 --time-budget-seconds 180 --memory-budget-mib 4096 --fit-method cox_reid --cox-reid-gene-bin-size 500 --cox-reid-batch-size 64 --dispersion-regularization sct_v2 --regularization-method r_stats --regularization-support clamp --theta-postprocess moment_ratio
python -m benchmarks.profile_stages --fixture benchmark-results/large/fixture.npz --manifest benchmark-results/large/fixture.selection.json --output benchmark-results/large/profile-v2.json --profile benchmark-results/large/profile-v2.prof --fit-method cox_reid --cox-reid-gene-bin-size 500 --cox-reid-batch-size 64 --dispersion-regularization sct_v2 --regularization-method r_stats --regularization-support clamp --theta-postprocess moment_ratio
```
