from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import platform
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
import tracemalloc
from typing import Any

import numpy as np
import psutil
from scipy import sparse

from sctransform_native import SCTConfig, fit_transform

try:
    from ._common import load_fixture, load_manifest, matrix_summary, write_json
except ImportError:  # pragma: no cover - direct script execution
    from _common import load_fixture, load_manifest, matrix_summary, write_json


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
    trace_python_allocations: bool,
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
    counts, genes, cells = load_fixture(fixture)
    input_matrix: np.ndarray | sparse.csr_matrix
    input_matrix = sparse.csr_matrix(counts) if input_format == "csr" else counts
    manifest = load_manifest(manifest_path)
    compute_start = time.perf_counter()
    result = fit_transform(
        input_matrix,
        gene_names=genes,
        cell_names=cells,
        config=SCTConfig(min_cells=5, n_genes=None, n_cells=None),
        selection=manifest,
    )
    compute_wall = time.perf_counter() - compute_start
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
        "compute_wall_seconds": compute_wall,
        "end_to_end_wall_seconds": end_to_end_wall,
        "process_cpu_seconds": process_cpu,
        "baseline_rss_bytes": baseline_rss,
        "peak_rss_bytes": peak_rss,
        "peak_rss_growth_bytes": peak_rss - baseline_rss,
        "max_process_threads": max(thread_counts),
        "stage_seconds": result.timing,
        "input": matrix_summary(input_matrix),
        "residuals": matrix_summary(result.residuals),
        "converged_gene_count": int(np.count_nonzero(result.converged)),
        "eligible_gene_count": int(np.count_nonzero(result.eligible_gene)),
    }
    if traced_peak is not None:
        output["python_tracemalloc_peak_bytes"] = traced_peak
    return output


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
    parser.add_argument("--time-budget-seconds", type=float, default=300.0)
    parser.add_argument("--memory-budget-mib", type=float, default=4096.0)
    parser.add_argument("--trace-python-allocations", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        print(
            json.dumps(
                _worker(
                    args.fixture,
                    args.manifest,
                    args.input_format,
                    trace_python_allocations=args.trace_python_allocations,
                )
            )
        )
        return
    if args.warmups < 1 or args.repeats < 1 or args.threads < 1:
        raise ValueError("warmups, repeats, and threads must be positive")

    environment = os.environ.copy()
    for name in THREAD_VARIABLES:
        environment[name] = str(args.threads)
    runs: list[dict[str, Any]] = []
    warmup_runs: list[dict[str, Any]] = []
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--fixture",
        str(args.fixture),
        "--manifest",
        str(args.manifest),
        "--input-format",
        args.input_format,
    ]
    if args.trace_python_allocations:
        command.append("--trace-python-allocations")
    for index in range(args.warmups + args.repeats):
        process_start = time.perf_counter()
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        process_wall = time.perf_counter() - process_start
        value = json.loads(completed.stdout.strip().splitlines()[-1])
        value["process_wall_seconds"] = process_wall
        if value["peak_rss_bytes"] > args.memory_budget_mib * 1024**2:
            raise RuntimeError("memory budget exceeded")
        if process_wall > args.time_budget_seconds:
            raise RuntimeError("time budget exceeded")
        (warmup_runs if index < args.warmups else runs).append(value)

    report = {
        "schema_version": 1,
        "generated_on": "2026-11-02",
        "python": sys.version,
        "platform": sys.platform,
        "processor": platform.processor(),
        "cpu_logical_count": psutil.cpu_count(logical=True),
        "configured_thread_environment": {
            name: environment[name] for name in THREAD_VARIABLES
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
            "rss": "sampled process RSS includes Python and native allocations",
            "tracemalloc": (
                "enabled; timing includes its overhead"
                if args.trace_python_allocations
                else "disabled for timing; use profile_stages.py for the supplemental Python-only peak"
            ),
        },
    }
    if args.output is not None:
        write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
