from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import threading
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import psutil
from scipy import sparse

from sctransform_native import SCTConfig, fit_transform

from ._common import load_fixture, load_manifest, matrix_summary, write_json


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
    parser.add_argument("--corrected-umi", action="store_true")
    args = parser.parse_args()

    counts, genes, cells = load_fixture(args.fixture)
    input_matrix = sparse.csr_matrix(counts)
    manifest = load_manifest(args.manifest)
    config = SCTConfig(
        min_cells=5,
        n_genes=None,
        n_cells=None,
        fit_method=args.fit_method,
        cox_reid_gene_bin_size=args.cox_reid_gene_bin_size,
        cox_reid_batch_size=args.cox_reid_batch_size,
        dispersion_regularization=args.dispersion_regularization,
        regularization_method=args.regularization_method,
        regularization_support=args.regularization_support,
        theta_postprocess=args.theta_postprocess,
        residual_block_size=args.block_size,
        residual_output=args.residual_output,
    )
    baseline_rss = psutil.Process().memory_info().rss
    rss_samples = [baseline_rss]
    stop = threading.Event()
    sampler = threading.Thread(
        target=_rss_sampler, args=(stop, rss_samples), daemon=True
    )
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
        return_corrected_umi=args.corrected_umi,
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
    if args.regularization_method == "native":
        bandwidth_shape = [min(256, sj_training_count), sj_training_count]
        bandwidth_dtype = np.dtype(np.float64)
        bandwidth_arrays = 5
        bandwidth_lifetime = "released after each derivative-functional block"
    elif sj_training_count <= 500:
        bandwidth_shape = [sj_training_count, sj_training_count]
        bandwidth_dtype = np.dtype(np.int64)
        bandwidth_arrays = 2
        bandwidth_lifetime = "released after the binned pair-count histogram"
    else:
        bandwidth_shape = [1000]
        bandwidth_dtype = np.dtype(np.float64)
        bandwidth_arrays = 4
        bandwidth_lifetime = "retained only during bandwidth selection"
    bandwidth_bytes = int(np.prod(bandwidth_shape) * bandwidth_dtype.itemsize)
    cox_reid_block_size = args.cox_reid_batch_size or 1
    cox_reid_block_shape = [
        min(cox_reid_block_size, len(manifest.selected_gene_indices)),
        len(manifest.selected_cell_indices),
    ]
    cox_reid_block_bytes = int(
        np.prod(cox_reid_block_shape) * np.dtype(np.float64).itemsize
    )
    report = {
        "schema_version": 1,
        "generated_on": datetime.now(timezone.utc).date().isoformat(),
        "model": {
            "fit_method": args.fit_method,
            "cox_reid_gene_bin_size": args.cox_reid_gene_bin_size,
            "cox_reid_batch_size": args.cox_reid_batch_size,
            "dispersion_regularization": args.dispersion_regularization,
            "regularization_method": args.regularization_method,
            "regularization_support": args.regularization_support,
            "theta_postprocess": args.theta_postprocess,
            "residual_output": args.residual_output,
            "corrected_umi": args.corrected_umi,
        },
        "wall_seconds": wall_seconds,
        "stage_seconds": result.timing,
        "baseline_rss_bytes": baseline_rss,
        "peak_rss_bytes": max(rss_samples),
        "peak_rss_growth_bytes": max(rss_samples) - baseline_rss,
        "python_tracemalloc_peak_bytes": traced_peak,
        "matrices": {
            "input_counts": matrix_summary(input_matrix),
            "dense_residual_output": (
                matrix_summary(result.residuals)
                if result.residuals is not None
                else None
            ),
            "sparse_corrected_umi_output": (
                matrix_summary(result.corrected_umi)
                if result.corrected_umi is not None
                else None
            ),
            "cox_reid_gene_block": {
                "shape": cox_reid_block_shape,
                "dtype": "float64",
                "nbytes_each": cox_reid_block_bytes,
                "estimated_simultaneous_dense_arrays": 8,
                "estimated_peak_block_bytes": cox_reid_block_bytes * 8,
                "lifetime": "allocated and released once per initial-fit gene block",
                "active": args.fit_method == "cox_reid",
            },
            "residual_gene_block": {
                "shape": gene_block_shape,
                "dtype": "float64",
                "nbytes_each": block_bytes,
                "estimated_simultaneous_dense_arrays": 5,
                "estimated_peak_block_bytes": block_bytes * 5,
                "lifetime": "allocated and released once per residual gene block",
            },
            "bandwidth_workspace": {
                "shape": bandwidth_shape,
                "dtype": str(bandwidth_dtype),
                "nbytes_each": bandwidth_bytes,
                "estimated_simultaneous_dense_arrays": bandwidth_arrays,
                "estimated_peak_block_bytes": bandwidth_bytes * bandwidth_arrays,
                "lifetime": bandwidth_lifetime,
            },
            "smoother_weight_block": {
                "shape": (
                    [min(512, len(genes)), sj_training_count]
                    if args.regularization_method == "native"
                    else [sj_training_count]
                ),
                "dtype": "float64",
                "nbytes_each": (
                    min(512, len(genes)) * sj_training_count * 8
                    if args.regularization_method == "native"
                    else sj_training_count * 8
                ),
                "estimated_simultaneous_dense_arrays": 3,
                "lifetime": "released after each prediction block or point",
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
