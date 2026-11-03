from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import psutil
from scipy import sparse

from sctpy import SCTConfig, SCTResult, fit_transform

from ._common import load_fixture, load_manifest, matrix_summary, write_json

THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _sample_process(
    process: psutil.Process,
    stop: threading.Event,
    samples: list[int],
    thread_counts: list[int],
) -> None:
    while not stop.wait(0.005):
        samples.append(process.memory_info().rss)
        thread_counts.append(process.num_threads())
    samples.append(process.memory_info().rss)
    thread_counts.append(process.num_threads())


def _worker(
    fixture: Path,
    manifest_path: Path,
    input_format: str,
    *,
    fit_method: str,
    cox_reid_gene_bin_size: int | None,
    cox_reid_batch_size: int | None,
    dispersion_regularization: str,
    regularization_method: str,
    regularization_support: str,
    theta_postprocess: str,
    residual_output: str,
    residual_dtype: str,
    return_corrected_umi: bool,
    trace_python_allocations: bool,
    cox_reid_backend: str,
    compute_warmups: int,
    allocator: str,
) -> dict[str, Any]:
    process = psutil.Process()
    baseline_rss = process.memory_info().rss
    rss_samples = [baseline_rss]
    thread_counts = [process.num_threads()]
    stop = threading.Event()
    sampler = threading.Thread(
        target=_sample_process,
        args=(process, stop, rss_samples, thread_counts),
        daemon=True,
    )
    if trace_python_allocations:
        tracemalloc.start()
    cpu_start = time.process_time()
    wall_start = time.perf_counter()
    sampler.start()
    gpu = None
    gpu_statistics = None
    gpu_setup_start = time.perf_counter()
    if cox_reid_backend == "cupy":
        import cupy as cp

        if allocator == "rmm":
            import rmm
            import rmm.statistics
            from rmm.allocators.cupy import rmm_cupy_allocator

            rmm.reinitialize(pool_allocator=True, initial_pool_size=512 * 1024**2)
            cp.cuda.set_allocator(rmm_cupy_allocator)
            gpu_statistics = rmm.statistics
        properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        gpu = {
            "device": properties["name"].decode(),
            "total_memory_bytes": properties["totalGlobalMem"],
            "cupy": cp.__version__,
            "allocator": allocator,
            "rmm": rmm.__version__ if allocator == "rmm" else None,
            "runtime_version": cp.cuda.runtime.runtimeGetVersion(),
            "driver_version": cp.cuda.runtime.driverGetVersion(),
        }
    gpu_setup_seconds = time.perf_counter() - gpu_setup_start
    counts, genes, cells = load_fixture(fixture)
    input_matrix: np.ndarray | sparse.csr_matrix
    input_matrix = (
        sparse.csr_matrix(counts)
        if input_format == "csr"
        else (counts.toarray() if sparse.issparse(counts) else counts)
    )
    manifest = load_manifest(manifest_path)
    config = SCTConfig(
        min_cells=5,
        n_genes=None,
        n_cells=None,
        fit_method=fit_method,
        cox_reid_gene_bin_size=cox_reid_gene_bin_size,
        cox_reid_batch_size=cox_reid_batch_size,
        dispersion_regularization=dispersion_regularization,
        regularization_method=regularization_method,
        regularization_support=regularization_support,
        theta_postprocess=theta_postprocess,
        residual_output=residual_output,
        residual_dtype=residual_dtype,
        cox_reid_backend=cox_reid_backend,
    )

    def compute() -> SCTResult:
        return fit_transform(
            input_matrix,
            gene_names=genes,
            cell_names=cells,
            config=config,
            selection=manifest,
            return_corrected_umi=return_corrected_umi,
        )

    warmup_start = time.perf_counter()
    for _ in range(compute_warmups):
        compute()
    warmup_seconds = time.perf_counter() - warmup_start
    memory_context = (
        gpu_statistics.statistics() if gpu_statistics else contextlib.nullcontext()
    )
    with memory_context:
        compute_start = time.perf_counter()
        result = compute()
        compute_wall = time.perf_counter() - compute_start
        if gpu_statistics:
            gpu["rmm_peak_bytes"] = gpu_statistics.get_statistics().peak_bytes
    if gpu is not None:
        gpu["cupy_pool_reserved_bytes"] = cp.get_default_memory_pool().total_bytes()
    end_to_end_wall = time.perf_counter() - wall_start
    process_cpu = time.process_time() - cpu_start
    stop.set()
    sampler.join()
    traced_peak: int | None = None
    if trace_python_allocations:
        _, traced_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    peak_rss = max(rss_samples)
    output = {
        "gpu": gpu,
        "gpu_setup_seconds": gpu_setup_seconds,
        "compute_warmup_seconds": warmup_seconds,
        "compute_wall_seconds": compute_wall,
        "end_to_end_wall_seconds": end_to_end_wall,
        "process_cpu_seconds": process_cpu,
        "baseline_rss_bytes": baseline_rss,
        "peak_rss_bytes": peak_rss,
        "peak_rss_growth_bytes": peak_rss - baseline_rss,
        "max_process_threads": max(thread_counts),
        "stage_seconds": result.timing,
        "input": matrix_summary(input_matrix),
        "residuals": (
            matrix_summary(result.residuals) if result.residuals is not None else None
        ),
        "corrected_umi": (
            matrix_summary(result.corrected_umi)
            if result.corrected_umi is not None
            else None
        ),
        "converged_gene_count": int(np.count_nonzero(result.converged)),
        "eligible_gene_count": int(np.count_nonzero(result.eligible_gene)),
    }
    if traced_peak is not None:
        output["python_tracemalloc_peak_bytes"] = traced_peak
    return output


def _run_budgeted(
    command: list[str],
    environment: dict[str, str],
    time_budget: float,
    memory_budget: float,
) -> str:
    started = time.perf_counter()
    with subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    ) as process:
        observed = psutil.Process(process.pid)
        while True:
            try:
                stdout, stderr = process.communicate(timeout=0.02)
                break
            except subprocess.TimeoutExpired:
                if time.perf_counter() - started > time_budget:
                    process.kill()
                    process.communicate()
                    raise RuntimeError("time budget exceeded") from None
                try:
                    rss = observed.memory_info().rss
                except psutil.NoSuchProcess:
                    continue
                if rss > memory_budget:
                    process.kill()
                    process.communicate()
                    raise RuntimeError("memory budget exceeded") from None
        if process.returncode:
            raise subprocess.CalledProcessError(
                process.returncode, command, stdout, stderr
            )
    return stdout


def _blas_report() -> str:
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        np.show_config()
    return stream.getvalue()


def _median_fields(runs: list[dict[str, Any]]) -> dict[str, float]:
    fields = (
        "process_wall_seconds",
        "compute_wall_seconds",
        "end_to_end_wall_seconds",
        "process_cpu_seconds",
        "peak_rss_bytes",
        "peak_rss_growth_bytes",
        "max_process_threads",
    )
    medians = {
        field: float(statistics.median(float(run[field]) for run in runs))
        for field in fields
    }
    if all("python_tracemalloc_peak_bytes" in run for run in runs):
        medians["python_tracemalloc_peak_bytes"] = float(
            statistics.median(
                float(run["python_tracemalloc_peak_bytes"]) for run in runs
            )
        )
    return medians


def _median_stage_fields(runs: list[dict[str, Any]]) -> dict[str, float]:
    stages = runs[0]["stage_seconds"]
    return {
        stage: float(
            statistics.median(float(run["stage_seconds"][stage]) for run in runs)
        )
        for stage in stages
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--input-format", choices=("csr", "dense"), default="csr")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--compute-warmups",
        type=int,
        default=0,
        help="Untimed fit calls inside each worker; end-to-end/process times still include them",
    )
    parser.add_argument(
        "--cox-reid-backend", choices=("numpy", "cupy"), default="numpy"
    )
    parser.add_argument("--allocator", choices=("cupy", "rmm"), default="cupy")
    parser.add_argument("--time-budget-seconds", type=float, default=300.0)
    parser.add_argument("--memory-budget-mib", type=float, default=4096.0)
    parser.add_argument("--trace-python-allocations", action="store_true")
    parser.add_argument(
        "--fit-method", choices=("nb_mle", "cox_reid"), default="nb_mle"
    )
    parser.add_argument("--cox-reid-gene-bin-size", type=int)
    parser.add_argument("--cox-reid-batch-size", type=int)
    parser.add_argument(
        "--dispersion-regularization",
        choices=("native", "sct_v2"),
        default="native",
    )
    parser.add_argument(
        "--regularization-method",
        choices=("native", "r_stats"),
        default="native",
    )
    parser.add_argument(
        "--regularization-support",
        choices=("extrapolate", "clamp"),
        default="extrapolate",
    )
    parser.add_argument(
        "--theta-postprocess",
        choices=("none", "moment_ratio"),
        default="none",
    )
    parser.add_argument(
        "--residual-output", choices=("full", "summary"), default="full"
    )
    parser.add_argument(
        "--residual-dtype", choices=("float64", "float32"), default="float64"
    )
    parser.add_argument("--corrected-umi", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        print(
            json.dumps(
                _worker(
                    args.fixture,
                    args.manifest,
                    args.input_format,
                    fit_method=args.fit_method,
                    cox_reid_gene_bin_size=args.cox_reid_gene_bin_size,
                    cox_reid_batch_size=args.cox_reid_batch_size,
                    dispersion_regularization=args.dispersion_regularization,
                    regularization_method=args.regularization_method,
                    regularization_support=args.regularization_support,
                    theta_postprocess=args.theta_postprocess,
                    residual_output=args.residual_output,
                    residual_dtype=args.residual_dtype,
                    return_corrected_umi=args.corrected_umi,
                    trace_python_allocations=args.trace_python_allocations,
                    cox_reid_backend=args.cox_reid_backend,
                    compute_warmups=args.compute_warmups,
                    allocator=args.allocator,
                )
            )
        )
        return
    if args.warmups < 1 or args.repeats < 1 or args.threads < 1:
        raise ValueError("warmups, repeats, and threads must be positive")
    if args.compute_warmups < 0 or any(
        not np.isfinite(value) or value <= 0
        for value in (args.time_budget_seconds, args.memory_budget_mib)
    ):
        raise ValueError(
            "compute-warmups must be nonnegative and budgets positive/finite"
        )

    environment = os.environ.copy()
    for name in THREAD_VARIABLES:
        environment[name] = str(args.threads)
    runs: list[dict[str, Any]] = []
    warmup_runs: list[dict[str, Any]] = []
    command = [
        sys.executable,
        "-m",
        "benchmarks.run_baseline",
        "--worker",
        "--cox-reid-backend",
        args.cox_reid_backend,
        "--compute-warmups",
        str(args.compute_warmups),
        "--allocator",
        args.allocator,
        "--fixture",
        str(args.fixture),
        "--manifest",
        str(args.manifest),
        "--input-format",
        args.input_format,
        "--fit-method",
        args.fit_method,
        "--dispersion-regularization",
        args.dispersion_regularization,
        "--regularization-method",
        args.regularization_method,
        "--regularization-support",
        args.regularization_support,
        "--theta-postprocess",
        args.theta_postprocess,
        "--residual-output",
        args.residual_output,
        "--residual-dtype",
        args.residual_dtype,
    ]
    if args.cox_reid_gene_bin_size is not None:
        command.extend(["--cox-reid-gene-bin-size", str(args.cox_reid_gene_bin_size)])
    if args.cox_reid_batch_size is not None:
        command.extend(["--cox-reid-batch-size", str(args.cox_reid_batch_size)])
    if args.trace_python_allocations:
        command.append("--trace-python-allocations")
    if args.corrected_umi:
        command.append("--corrected-umi")
    for index in range(args.warmups + args.repeats):
        process_start = time.perf_counter()
        stdout = _run_budgeted(
            command,
            environment,
            args.time_budget_seconds,
            args.memory_budget_mib * 1024**2,
        )
        process_wall = time.perf_counter() - process_start
        value = json.loads(stdout.strip().splitlines()[-1])
        value["process_wall_seconds"] = process_wall
        if value["peak_rss_bytes"] > args.memory_budget_mib * 1024**2:
            raise RuntimeError("memory budget exceeded")
        if process_wall > args.time_budget_seconds:
            raise RuntimeError("time budget exceeded")
        (warmup_runs if index < args.warmups else runs).append(value)

    report = {
        "schema_version": 1,
        "fixture_sha256": hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "generated_on": datetime.now(timezone.utc).date().isoformat(),
        "python": sys.version,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "sctpy")
        },
        "compute_warmups_per_process": args.compute_warmups,
        "platform": sys.platform,
        "processor": platform.processor(),
        "cpu_logical_count": psutil.cpu_count(logical=True),
        "configured_thread_environment": {
            name: environment[name] for name in THREAD_VARIABLES
        },
        "model": {
            "fit_method": args.fit_method,
            "cox_reid_gene_bin_size": args.cox_reid_gene_bin_size,
            "cox_reid_batch_size": args.cox_reid_batch_size,
            "cox_reid_backend": args.cox_reid_backend,
            "dispersion_regularization": args.dispersion_regularization,
            "regularization_method": args.regularization_method,
            "regularization_support": args.regularization_support,
            "theta_postprocess": args.theta_postprocess,
            "residual_output": args.residual_output,
            "residual_dtype": args.residual_dtype,
            "corrected_umi": args.corrected_umi,
        },
        "blas_configuration": _blas_report(),
        "warmups": warmup_runs,
        "measured_runs": runs,
        "median": _median_fields(runs),
        "median_stage_seconds": _median_stage_fields(runs),
        "budgets": {
            "time_seconds_per_run": args.time_budget_seconds,
            "peak_rss_mib": args.memory_budget_mib,
        },
        "notes": {
            "timing": "compute times exclude setup, I/O and per-process warmups; end-to-end, CPU time and peak RSS include them; each worker is a fresh process",
            "rss": "sampled process RSS includes Python and native allocations",
            "tracemalloc": (
                "enabled; timing includes its overhead"
                if args.trace_python_allocations
                else "disabled for timing; profile_stages.py reports traced allocations, not total native/GPU memory"
            ),
        },
    }
    if args.output is not None:
        write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
