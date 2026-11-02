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

## Baselines

Input is CSR; the retained residual output is dense float64.

### Validated v2 path

The current correctness path is explicit:

```text
cox_reid / sct_v2 / r_stats / clamp / moment_ratio
```

| Fixture | Median compute wall | Median process CPU | Median peak RSS | Median RSS growth | Budget |
|---|---:|---:|---:|---:|---|
| 800 x 240 | 1.523 s | 1.547 s | 79.96 MiB | 8.76 MiB | 60 s / 2 GiB |
| 3200 x 240 | 5.855 s | 5.922 s | 105.98 MiB | 34.98 MiB | 180 s / 4 GiB |

Median uninstrumented stage timers:

| Fixture | Fit | Regularization | Residuals + HVG | Total |
|---|---:|---:|---:|---:|
| 800 x 240 | 1.459 s | 0.0336 s | 0.0042 s | 1.523 s |
| 3200 x 240 | 5.634 s | 0.1501 s | 0.0113 s | 5.854 s |

On the larger fixture, Cox--Reid fitting is about 96.2% of stage time;
regularization is about 2.6%, and residual construction plus HVG ranking is about 0.2%.
cProfile recorded 3,091 per-gene fits, 9,262 Brent root solves, and 53,675 evaluations of
the Cox--Reid score. The score function was the largest project-owned cumulative call
path. Call tracing raised wall time to 14.08 seconds, so these counts identify structure
but the percentages above come from uninstrumented runs. **measured**

The instrumented larger run reported 107.33 MiB sampled peak RSS and 32.33 MiB
`tracemalloc` peak. The latter remains a Python-only supplement.

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

In the validated path, a Cox--Reid score evaluation operates on one dense 240-cell gene
row at a time. The row and its temporary vectors are released between genes; no dense
genes-by-genes or genes-by-cells fitting workspace is retained. The measured 34.98 MiB
median RSS growth includes these working vectors, SciPy state, and the dense result.

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

The evidence still does not justify a custom CUDA or Triton kernel. The isolated SJ
functional result remains valid for the provisional native regularizer, but accelerating
that 2.6% stage cannot materially change the validated v2 path on this fixture. The next
candidate must address Cox--Reid score evaluation and root-solver orchestration. A batched
CPU experiment should establish how much work can be grouped before a GPU experiment is
added; divergent per-gene roots and transfer/launch costs remain open questions. Residual
calculation should be reconsidered only on a fixture where profiling makes it material.
See [GPU_FEASIBILITY.md](GPU_FEASIBILITY.md).

## Reproduction

```bash
python -m benchmarks.run_baseline --fixture benchmark-results/large/fixture.npz --manifest benchmark-results/large/fixture.selection.json --output benchmark-results/large/baseline-v2.json --warmups 1 --repeats 5 --threads 1 --time-budget-seconds 180 --memory-budget-mib 4096 --fit-method cox_reid --dispersion-regularization sct_v2 --regularization-method r_stats --regularization-support clamp --theta-postprocess moment_ratio
python -m benchmarks.profile_stages --fixture benchmark-results/large/fixture.npz --manifest benchmark-results/large/fixture.selection.json --output benchmark-results/large/profile-v2.json --profile benchmark-results/large/profile-v2.prof --fit-method cox_reid --dispersion-regularization sct_v2 --regularization-method r_stats --regularization-support clamp --theta-postprocess moment_ratio
```
