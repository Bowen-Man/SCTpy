from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import statistics
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from sctpy._fit import (
    _DISPERSION_CEILING,
    _DISPERSION_FLOOR,
    _cox_reid_score_batch,
    offset_intercept,
)
from sctpy._gpu import _dispersion_roots_cupy

from ._common import load_fixture, write_json
from .correctness import numeric_metrics

GPU_THRESHOLDS = {
    "log10_theta_rmse": 1.0e-6,
    "log10_theta_max_absolute_error": 1.0e-4,
    "branch_masks_exact": True,
}


@dataclass(frozen=True)
class RootInputs:
    counts: np.ndarray
    exposure: np.ndarray
    mu: np.ndarray
    initial_dispersion: np.ndarray


def prepare_root_inputs(counts: np.ndarray, exposure: np.ndarray) -> RootInputs:
    """Prepare the common float64 inputs outside the timed root solver."""
    counts = np.asarray(counts, dtype=np.float64)
    exposure = np.asarray(exposure, dtype=np.float64)
    if counts.ndim != 2 or counts.shape[1] != exposure.size:
        raise ValueError("counts must be genes by exposure-sized cells")
    if np.any(np.sum(counts, axis=1) == 0.0):
        raise ValueError("benchmark genes must contain at least one count")
    if np.any(~np.isfinite(exposure)) or np.any(exposure <= 0.0):
        raise ValueError("exposure must be finite and positive")

    means = np.mean(counts, axis=1, dtype=np.float64)
    variances = np.var(counts, axis=1, ddof=1, dtype=np.float64)
    initial_dispersion = np.maximum(
        (variances - means / float(np.mean(exposure))) / means**2,
        0.0,
    )
    intercept = np.asarray(
        [
            offset_intercept(row, exposure, float(dispersion))[0]
            for row, dispersion in zip(counts, initial_dispersion, strict=True)
        ],
        dtype=np.float64,
    )
    mu = np.exp(intercept[:, None]) * exposure[None, :]
    return RootInputs(counts, exposure, mu, initial_dispersion)


def dispersion_roots_numpy(
    inputs: RootInputs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Run the existing NumPy score through the safeguarded batched roots."""
    counts = inputs.counts
    mu = inputs.mu
    lower = np.full(counts.shape[0], np.log(_DISPERSION_FLOOR), dtype=np.float64)
    lower_score = _cox_reid_score_batch(lower, counts, mu)
    poisson = lower_score < 0.0
    upper_dispersion = np.minimum(
        np.maximum(inputs.initial_dispersion, _DISPERSION_FLOOR * 4.0),
        _DISPERSION_CEILING,
    )
    upper_score = _cox_reid_score_batch(np.log(upper_dispersion), counts, mu)
    expansion_rounds = 0
    expanding = (
        (~poisson) & (upper_score > 0.0) & (upper_dispersion < _DISPERSION_CEILING)
    )
    while np.any(expanding):
        expansion_rounds += 1
        upper_dispersion[expanding] = np.minimum(
            upper_dispersion[expanding] * 4.0, _DISPERSION_CEILING
        )
        indices = np.flatnonzero(expanding)
        upper_score[indices] = _cox_reid_score_batch(
            np.log(upper_dispersion[indices]), counts[indices], mu[indices]
        )
        expanding = (
            (~poisson) & (upper_score > 0.0) & (upper_dispersion < _DISPERSION_CEILING)
        )

    ceiling = (~poisson) & (upper_score > 0.0)
    bracketed = ~(poisson | ceiling)
    upper = np.log(upper_dispersion)
    for _ in range(48):
        indices = np.flatnonzero(bracketed)
        if indices.size == 0:
            break
        midpoint = 0.5 * (lower[indices] + upper[indices])
        score = _cox_reid_score_batch(midpoint, counts[indices], mu[indices])
        positive = score > 0.0
        lower[indices[positive]] = midpoint[positive]
        upper[indices[~positive]] = midpoint[~positive]

    dispersion = np.exp(0.5 * (lower + upper))
    dispersion[poisson] = 0.0
    dispersion[ceiling] = _DISPERSION_CEILING
    theta = np.divide(
        1.0,
        dispersion,
        out=np.full_like(dispersion, np.inf),
        where=dispersion != 0.0,
    )
    return theta, poisson, ceiling, expansion_rounds


def _select_block(
    counts_cxg: sparse.csr_matrix,
    *,
    requested_cells: int,
    requested_genes: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    if requested_cells < counts_cxg.shape[0]:
        cell_indices = np.sort(
            rng.choice(counts_cxg.shape[0], size=requested_cells, replace=False)
        )
        selected = counts_cxg[cell_indices]
    else:
        cell_indices = np.arange(counts_cxg.shape[0], dtype=np.int64)
        selected = counts_cxg
    exposure = np.asarray(selected.sum(axis=1), dtype=np.float64).ravel()
    positive_cells = exposure > 0.0
    selected = selected[positive_cells]
    cell_indices = cell_indices[positive_cells]
    exposure = exposure[positive_cells]
    if exposure.size < 2:
        raise ValueError("fewer than two positive-library cells remain")

    cells = selected.shape[0]
    sums = np.asarray(selected.sum(axis=0), dtype=np.float64).ravel()
    squared = selected.copy().astype(np.float64)
    squared.data **= 2
    sum_squares = np.asarray(squared.sum(axis=0), dtype=np.float64).ravel()
    means = sums / cells
    variances = (sum_squares - cells * means**2) / (cells - 1)
    detected = np.asarray(selected.getnnz(axis=0)).ravel()
    candidates = np.flatnonzero((detected >= 5) & (variances > means))
    if candidates.size < requested_genes:
        raise ValueError(
            f"only {candidates.size} overdispersed genes are available for "
            f"a {requested_genes}-gene block"
        )
    priority = variances[candidates] - means[candidates]
    chosen = np.sort(candidates[np.argsort(-priority, kind="stable")[:requested_genes]])
    block = selected[:, chosen].T.toarray().astype(np.float64, copy=False)
    if np.any(block != np.rint(block)) or np.any(block < 0.0):
        raise ValueError("selected input is not a raw nonnegative UMI count block")
    summary = {
        "source_shape_cells_by_genes": list(counts_cxg.shape),
        "source_nnz": int(counts_cxg.nnz),
        "selected_cells": int(exposure.size),
        "selected_genes": int(chosen.size),
        "selected_block_shape_genes_by_cells": list(block.shape),
        "selected_block_nbytes": int(block.nbytes),
        "selected_block_sha256": hashlib.sha256(block.tobytes()).hexdigest(),
        "selected_cell_indices_sha256": hashlib.sha256(
            cell_indices.astype("<i8", copy=False).tobytes()
        ).hexdigest(),
        "candidate_genes": int(candidates.size),
    }
    return block, exposure, chosen, summary


def _load_counts(args: argparse.Namespace) -> tuple[sparse.csr_matrix, dict[str, Any]]:
    if args.h5ad is not None:
        import anndata as ad

        adata = ad.read_h5ad(args.h5ad)
        counts = sparse.csr_matrix(adata.X)
        return counts, {
            "kind": "h5ad",
            "name": args.h5ad.name,
        }
    counts_gxc, _, _ = load_fixture(args.fixture)
    counts = sparse.csr_matrix(counts_gxc).T.tocsr()
    return counts, {
        "kind": "fixture",
        "name": args.fixture.name,
    }


def _median(values: list[dict[str, float]], field: str) -> float:
    return float(statistics.median(value[field] for value in values))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Isolated CuPy Cox-Reid dispersion-root microbenchmark."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--h5ad", type=Path)
    source.add_argument("--fixture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cells", type=int, default=10_000)
    parser.add_argument("--genes", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--allocator", choices=("cupy", "rmm"), default="rmm")
    parser.add_argument("--rmm-pool-mib", type=int, default=512)
    args = parser.parse_args()
    if min(args.cells, args.genes, args.warmups, args.repeats) < 1:
        raise ValueError("cells, genes, warmups, and repeats must be positive")
    if args.allocator == "rmm" and args.rmm_pool_mib < 1:
        raise ValueError("rmm-pool-mib must be positive")

    load_start = time.perf_counter()
    counts_cxg, source_info = _load_counts(args)
    block, exposure, gene_indices, data_summary = _select_block(
        counts_cxg,
        requested_cells=args.cells,
        requested_genes=args.genes,
        seed=args.seed,
    )
    load_seconds = time.perf_counter() - load_start
    del counts_cxg

    prepare_start = time.perf_counter()
    inputs = prepare_root_inputs(block, exposure)
    prepare_seconds = time.perf_counter() - prepare_start

    import cupy as cp
    from cupyx.scipy.special import digamma

    cp.cuda.Device(args.device).use()
    rmm_statistics = None
    rmm_version = None
    if args.allocator == "rmm":
        import rmm
        import rmm.statistics
        from rmm.allocators.cupy import rmm_cupy_allocator

        rmm.reinitialize(
            pool_allocator=True,
            initial_pool_size=args.rmm_pool_mib * 1024**2,
            devices=args.device,
        )
        cp.cuda.set_allocator(rmm_cupy_allocator)
        rmm_statistics = rmm.statistics
        rmm_version = rmm.__version__

    cpu_runs: list[dict[str, float]] = []
    cpu_result: tuple[np.ndarray, np.ndarray, np.ndarray, int] | None = None
    for index in range(args.warmups + args.repeats):
        started = time.perf_counter()
        result = dispersion_roots_numpy(inputs)
        elapsed = time.perf_counter() - started
        if index >= args.warmups:
            cpu_runs.append({"root_wall_seconds": elapsed})
        cpu_result = result
    assert cpu_result is not None

    gpu_runs: list[dict[str, float]] = []
    gpu_result: tuple[np.ndarray, np.ndarray, np.ndarray, int, int] | None = None
    memory_runs: list[dict[str, int]] = []
    for index in range(args.warmups + args.repeats):
        stream = cp.cuda.get_current_stream()
        stream.synchronize()
        free_before, total_memory = cp.cuda.runtime.memGetInfo()
        memory_context = (
            rmm_statistics.statistics() if rmm_statistics is not None else nullcontext()
        )
        wall_started = time.perf_counter()
        with memory_context:
            h2d_start = cp.cuda.Event()
            h2d_end = cp.cuda.Event()
            h2d_start.record()
            device_counts = cp.asarray(inputs.counts)
            device_mu = cp.asarray(inputs.mu)
            device_initial = cp.asarray(inputs.initial_dispersion)
            h2d_end.record()
            h2d_end.synchronize()

            kernel_start = cp.cuda.Event()
            kernel_end = cp.cuda.Event()
            kernel_start.record()
            device_result = _dispersion_roots_cupy(
                cp, digamma, device_counts, device_mu, device_initial
            )
            kernel_end.record()
            kernel_end.synchronize()

            d2h_start = cp.cuda.Event()
            d2h_end = cp.cuda.Event()
            d2h_start.record()
            theta = cp.asnumpy(device_result.theta)
            poisson = cp.asnumpy(device_result.poisson)
            ceiling = cp.asnumpy(device_result.ceiling)
            d2h_end.record()
            d2h_end.synchronize()
            elapsed = time.perf_counter() - wall_started
            stats = (
                rmm_statistics.get_statistics() if rmm_statistics is not None else None
            )
        free_after, _ = cp.cuda.runtime.memGetInfo()
        if index >= args.warmups:
            gpu_runs.append(
                {
                    "h2d_seconds": cp.cuda.get_elapsed_time(h2d_start, h2d_end)
                    / 1_000.0,
                    "kernel_seconds": cp.cuda.get_elapsed_time(kernel_start, kernel_end)
                    / 1_000.0,
                    "d2h_seconds": cp.cuda.get_elapsed_time(d2h_start, d2h_end)
                    / 1_000.0,
                    "end_to_end_wall_seconds": elapsed,
                }
            )
            memory_runs.append(
                {
                    "cuda_total_bytes": int(total_memory),
                    "cuda_free_before_bytes": int(free_before),
                    "cuda_free_after_bytes": int(free_after),
                    "rmm_peak_bytes": int(stats.peak_bytes) if stats else 0,
                    "rmm_total_bytes": int(stats.total_bytes) if stats else 0,
                    "cupy_pool_used_bytes": int(
                        cp.get_default_memory_pool().used_bytes()
                    ),
                    "cupy_pool_total_bytes": int(
                        cp.get_default_memory_pool().total_bytes()
                    ),
                }
            )
        gpu_result = (
            theta,
            poisson,
            ceiling,
            device_result.expansion_rounds,
            device_result.host_synchronizations,
        )
        del device_counts, device_mu, device_initial, device_result
    assert gpu_result is not None

    cpu_theta, cpu_poisson, cpu_ceiling, cpu_expansions = cpu_result
    gpu_theta, gpu_poisson, gpu_ceiling, gpu_expansions, host_syncs = gpu_result
    finite = np.isfinite(cpu_theta) & np.isfinite(gpu_theta)
    theta_metrics = numeric_metrics(
        np.log10(gpu_theta[finite]), np.log10(cpu_theta[finite])
    )
    masks_exact = bool(
        np.array_equal(cpu_poisson, gpu_poisson)
        and np.array_equal(cpu_ceiling, gpu_ceiling)
    )
    gate_passed = bool(
        masks_exact
        and float(theta_metrics["rmse"]) <= GPU_THRESHOLDS["log10_theta_rmse"]
        and float(theta_metrics["max_absolute_error"])
        <= GPU_THRESHOLDS["log10_theta_max_absolute_error"]
    )
    cpu_median = _median(cpu_runs, "root_wall_seconds")
    gpu_kernel_median = _median(gpu_runs, "kernel_seconds")
    gpu_end_to_end_median = _median(gpu_runs, "end_to_end_wall_seconds")
    properties = cp.cuda.runtime.getDeviceProperties(args.device)
    blas = np.__config__.CONFIG.get("Build Dependencies", {}).get("blas", {})
    report = {
        "schema_version": 1,
        "generated_on": datetime.now(timezone.utc).date().isoformat(),
        "experiment": "isolated_cox_reid_dispersion_roots",
        "thresholds_preregistered": GPU_THRESHOLDS,
        "gate_passed": gate_passed,
        "source": {**source_info, **data_summary},
        "selection": {
            "seed": args.seed,
            "gene_indices_sha256": hashlib.sha256(
                gene_indices.astype("<i8", copy=False).tobytes()
            ).hexdigest(),
            "rule": "largest positive variance-minus-mean among detected genes",
        },
        "configuration": {
            "warmups": args.warmups,
            "repeats": args.repeats,
            "allocator": args.allocator,
            "rmm_pool_mib": args.rmm_pool_mib if args.allocator == "rmm" else None,
            "dtype": "float64",
            "device": args.device,
            "root_iterations": 48,
        },
        "software": {
            "numpy": np.__version__,
            "scipy": importlib.metadata.version("scipy"),
            "cupy": cp.__version__,
            "rmm": rmm_version,
            "blas": {key: blas.get(key) for key in ("name", "version")},
            "cpu_thread_environment": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                )
            },
        },
        "device": {
            "name": properties["name"].decode(),
            "compute_capability": cp.cuda.Device(args.device).compute_capability,
            "driver_version": int(cp.cuda.runtime.driverGetVersion()),
            "runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
        },
        "timing": {
            "data_load_seconds": load_seconds,
            "shared_input_preparation_seconds": prepare_seconds,
            "cpu_runs": cpu_runs,
            "gpu_runs": gpu_runs,
            "median": {
                "cpu_root_seconds": cpu_median,
                "gpu_h2d_seconds": _median(gpu_runs, "h2d_seconds"),
                "gpu_kernel_seconds": gpu_kernel_median,
                "gpu_d2h_seconds": _median(gpu_runs, "d2h_seconds"),
                "gpu_end_to_end_seconds": gpu_end_to_end_median,
                "kernel_speedup_vs_cpu": cpu_median / gpu_kernel_median,
                "end_to_end_speedup_vs_cpu": cpu_median / gpu_end_to_end_median,
            },
        },
        "correctness": {
            "log10_theta": theta_metrics,
            "poisson_mask_exact": bool(np.array_equal(cpu_poisson, gpu_poisson)),
            "ceiling_mask_exact": bool(np.array_equal(cpu_ceiling, gpu_ceiling)),
            "cpu_expansion_rounds": cpu_expansions,
            "gpu_expansion_rounds": gpu_expansions,
        },
        "synchronization": {
            "explicit_phase_synchronizations_per_run": 3,
            "root_control_host_synchronizations_per_run": host_syncs,
        },
        "memory": memory_runs,
        "notes": {
            "scope": "dispersion score/root only; preliminary intercepts are shared CPU input",
            "device_transfer": "all H2D and D2H are explicit and included in end-to-end timing",
            "arithmetic": "special-function-heavy repeated dense scans; no FLOP-rate claim",
        },
    }
    write_json(args.output, report)
    print(json.dumps(report, allow_nan=False, indent=2, sort_keys=True))
    if not gate_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
