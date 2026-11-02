from __future__ import annotations

import argparse
import cProfile
import json
from pathlib import Path
import pstats
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


def _rss_sampler(stop: threading.Event, samples: list[int]) -> None:
    process = psutil.Process()
    while not stop.wait(0.005):
        samples.append(process.memory_info().rss)
    samples.append(process.memory_info().rss)


def _top_profile(profile: cProfile.Profile, limit: int = 25) -> list[dict[str, Any]]:
    stats = pstats.Stats(profile)
    entries: list[dict[str, Any]] = []
    for (filename, line, function), values in sorted(
        stats.stats.items(), key=lambda item: item[1][3], reverse=True
    )[:limit]:
        primitive_calls, total_calls, total_time, cumulative_time, _ = values
        entries.append(
            {
                "file": filename,
                "line": line,
                "function": function,
                "primitive_calls": primitive_calls,
                "total_calls": total_calls,
                "total_seconds": total_time,
                "cumulative_seconds": cumulative_time,
            }
        )
    return entries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--block-size", type=int, default=256)
    args = parser.parse_args()

    counts, genes, cells = load_fixture(args.fixture)
    input_matrix = sparse.csr_matrix(counts)
    manifest = load_manifest(args.manifest)
    config = SCTConfig(
        min_cells=5,
        n_genes=None,
        n_cells=None,
        residual_block_size=args.block_size,
    )
    rss_samples = [psutil.Process().memory_info().rss]
    stop = threading.Event()
    sampler = threading.Thread(target=_rss_sampler, args=(stop, rss_samples), daemon=True)
    profile = cProfile.Profile()
    tracemalloc.start()
    sampler.start()
    wall_start = time.perf_counter()
    profile.enable()
    result = fit_transform(
        input_matrix,
        gene_names=genes,
        cell_names=cells,
        config=config,
        selection=manifest,
    )
    profile.disable()
    wall_seconds = time.perf_counter() - wall_start
    stop.set()
    sampler.join()
    _, traced_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    args.profile.parent.mkdir(parents=True, exist_ok=True)
    profile.dump_stats(args.profile)

    gene_block_shape = [min(args.block_size, len(genes)), len(cells)]
    block_bytes = int(np.prod(gene_block_shape) * np.dtype(np.float64).itemsize)
    sj_training_count = int(
        np.count_nonzero(
            result.selected_gene
            & result.converged
            & ~result.poisson_gene
            & ~result.parameter_outlier
        )
    )
    sj_block_shape = [min(256, sj_training_count), sj_training_count]
    sj_block_bytes = int(np.prod(sj_block_shape) * np.dtype(np.float64).itemsize)
    report = {
        "schema_version": 1,
        "generated_on": "2026-11-02",
        "wall_seconds": wall_seconds,
        "stage_seconds": result.timing,
        "peak_rss_bytes": max(rss_samples),
        "python_tracemalloc_peak_bytes": traced_peak,
        "matrices": {
            "input_counts": matrix_summary(input_matrix),
            "dense_residual_output": matrix_summary(result.residuals),
            "residual_gene_block": {
                "shape": gene_block_shape,
                "dtype": "float64",
                "nbytes_each": block_bytes,
                "estimated_simultaneous_dense_arrays": 5,
                "estimated_peak_block_bytes": block_bytes * 5,
                "lifetime": "allocated and released once per residual gene block",
            },
            "sj_pairwise_derivative_block": {
                "shape": sj_block_shape,
                "dtype": "float64",
                "nbytes_each": sj_block_bytes,
                "estimated_simultaneous_dense_arrays": 5,
                "estimated_peak_block_bytes": sj_block_bytes * 5,
                "lifetime": "allocated and released for each pilot/root functional evaluation",
            },
            "smoother_weight_block": {
                "shape": [min(512, len(genes)), sj_training_count],
                "dtype": "float64",
                "nbytes_each": min(512, len(genes)) * sj_training_count * 8,
                "estimated_simultaneous_dense_arrays": 3,
                "lifetime": "allocated and released once per prediction block and parameter curve",
            },
            "per_gene_parameter_vectors": {
                "shape": [len(genes)],
                "dtype": "float64",
                "nbytes_each": len(genes) * 8,
                "lifetime": "fit through result construction; selected arrays retained",
            },
        },
        "top_cprofile_cumulative": _top_profile(profile),
        "notes": {
            "rss": "sampled process RSS includes Python and native allocations",
            "tracemalloc": "supplement only; does not represent all NumPy/SciPy native memory",
            "cprofile": "call tracing changes absolute timing; use stage medians from run_baseline.py for performance claims",
        },
    }
    write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
