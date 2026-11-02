from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np

from sctransform_native._regularize import _normal_derivative_functional

from ._common import load_fixture, write_json

THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _cupy_functional(
    cp: Any,
    x: Any,
    *,
    bandwidth: float,
    block_size: int,
) -> Any:
    total = cp.zeros((), dtype=cp.float64)
    normalizer = 1.0 / np.sqrt(2.0 * np.pi)
    for start in range(0, int(x.size), block_size):
        stop = min(start + block_size, int(x.size))
        u = (x[start:stop, None] - x[None, :]) / bandwidth
        squared = u * u
        density = normalizer * cp.exp(-0.5 * squared)
        polynomial = squared * squared - 6.0 * squared + 3.0
        total += cp.sum(polynomial * density, dtype=cp.float64)
    denominator = x.size * (x.size - 1) * bandwidth**5
    return total / denominator


def _event_seconds(cp: Any, operation: Any) -> tuple[float, Any]:
    start = cp.cuda.Event()
    stop = cp.cuda.Event()
    start.record()
    value = operation()
    stop.record()
    stop.synchronize()
    return float(cp.cuda.get_elapsed_time(start, stop) / 1000.0), value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--min-cells", type=int, default=5)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--memory-budget-gib", type=float, default=12.0)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmups < 1 or args.block_size < 1:
        raise ValueError("repeats, warmups, and block size must be positive")

    try:
        import cupy as cp
    except ImportError as error:
        raise RuntimeError(
            "CuPy is optional; install a matching GPU extra such as .[gpu-cuda13]"
        ) from error

    counts, _, _ = load_fixture(args.fixture)
    detected = np.count_nonzero(counts, axis=1)
    geometric_mean = np.expm1(np.mean(np.log1p(counts), axis=1))
    x_cpu = np.log10(
        np.maximum(geometric_mean[detected >= args.min_cells], np.finfo(float).tiny)
    ).astype(np.float64, copy=False)
    q25, q75 = np.percentile(x_cpu, [25.0, 75.0])
    bandwidth = float(0.920 * (q75 - q25) * x_cpu.size ** (-1.0 / 7.0))
    block_rows = min(args.block_size, x_cpu.size)
    estimated_working_set = 5 * block_rows * x_cpu.size * 8
    memory_budget = int(args.memory_budget_gib * 1024**3)
    if estimated_working_set > memory_budget:
        raise RuntimeError("estimated live block working set exceeds the memory budget")

    with cp.cuda.Device(args.device):
        properties = cp.cuda.runtime.getDeviceProperties(args.device)
        free_before, total_memory = cp.cuda.runtime.memGetInfo()
        transfer_start = time.perf_counter()
        x_gpu = cp.asarray(x_cpu)
        cp.cuda.get_current_stream().synchronize()
        h2d_seconds = time.perf_counter() - transfer_start

        for _ in range(args.warmups):
            _cupy_functional(cp, x_gpu, bandwidth=bandwidth, block_size=args.block_size)
            cp.cuda.get_current_stream().synchronize()

        gpu_kernel_seconds: list[float] = []
        gpu_result = None
        for _ in range(args.repeats):
            elapsed, gpu_result = _event_seconds(
                cp,
                lambda: _cupy_functional(
                    cp, x_gpu, bandwidth=bandwidth, block_size=args.block_size
                ),
            )
            gpu_kernel_seconds.append(elapsed)

        d2h_start = time.perf_counter()
        gpu_value = float(cp.asnumpy(gpu_result))
        d2h_seconds = time.perf_counter() - d2h_start
        pool_total_bytes = int(cp.get_default_memory_pool().total_bytes())
        free_after, _ = cp.cuda.runtime.memGetInfo()

        end_to_end_seconds: list[float] = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            local_x = cp.asarray(x_cpu)
            local_result = _cupy_functional(
                cp, local_x, bandwidth=bandwidth, block_size=args.block_size
            )
            float(cp.asnumpy(local_result))
            cp.cuda.get_current_stream().synchronize()
            end_to_end_seconds.append(time.perf_counter() - started)

    for _ in range(args.warmups):
        _normal_derivative_functional(x_cpu, bandwidth=bandwidth, order=4)
    cpu_seconds: list[float] = []
    cpu_value = 0.0
    for _ in range(args.repeats):
        started = time.perf_counter()
        cpu_value = _normal_derivative_functional(
            x_cpu, bandwidth=bandwidth, order=4, block_size=args.block_size
        )
        cpu_seconds.append(time.perf_counter() - started)

    absolute_error = abs(cpu_value - gpu_value)
    relative_error = absolute_error / max(abs(cpu_value), np.finfo(float).tiny)
    report = {
        "schema_version": 1,
        "generated_on": "2026-11-02",
        "operation": "blocked diagonal-in fourth Gaussian derivative functional",
        "fixture": str(args.fixture),
        "training_values": int(x_cpu.size),
        "dtype": "float64",
        "block_size": args.block_size,
        "estimated_live_block_bytes": estimated_working_set,
        "memory_budget_bytes": memory_budget,
        "transfer_bytes": {"h2d": int(x_cpu.nbytes), "d2h": 8},
        "timing_seconds": {
            "h2d_once": h2d_seconds,
            "d2h_scalar_once": d2h_seconds,
            "cpu_median": statistics.median(cpu_seconds),
            "gpu_kernel_median": statistics.median(gpu_kernel_seconds),
            "gpu_end_to_end_median": statistics.median(end_to_end_seconds),
        },
        "correctness": {
            "cpu": cpu_value,
            "gpu": gpu_value,
            "absolute_error": absolute_error,
            "relative_error": relative_error,
            "rtol_gate": 1.0e-10,
            "gate": relative_error <= 1.0e-10,
        },
        "gpu": {
            "cupy_version": cp.__version__,
            "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
            "cuda_driver_version": int(cp.cuda.runtime.driverGetVersion()),
            "device": args.device,
            "name": properties["name"].decode()
            if isinstance(properties["name"], bytes)
            else str(properties["name"]),
            "total_memory_bytes": int(total_memory),
            "free_memory_before_bytes": int(free_before),
            "free_memory_after_bytes": int(free_after),
            "memory_pool_total_bytes": pool_total_bytes,
        },
        "notes": {
            "scope": "isolated feasibility benchmark; not a package GPU backend",
            "kernel": "CuPy elementwise/reduction operations only; no custom CUDA or Triton",
        },
        "cpu_thread_environment": {
            name: os.environ.get(name, "unset") for name in THREAD_VARIABLES
        },
    }
    write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["correctness"]["gate"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
