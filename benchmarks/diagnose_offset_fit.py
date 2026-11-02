from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from sctransform_native._fit import (
    fit_cox_reid_offset,
    local_median_dispersion_trend,
    offset_intercept,
)
from sctransform_native._matrix import dense_gene_row

from ._common import count_sha256, load_fixture, load_manifest, write_json
from .compare_correctness import load_r_oracle
from .correctness import numeric_metrics, top_absolute_errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trace offset fits and compare local-trend gene batch sizes."
    )
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--r-oracle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bin-sizes", nargs="+", default=("500", "all"))
    parser.add_argument("--top", type=int, default=3)
    return parser.parse_args()


def _resolve_bin_size(value: str, gene_count: int) -> int:
    if value == "all":
        return gene_count
    size = int(value)
    if size < 1:
        raise ValueError("bin sizes must be positive integers or 'all'")
    return size


def _batched_trend(
    abundance: np.ndarray, dispersion: np.ndarray, bin_size: int
) -> np.ndarray:
    return local_median_dispersion_trend(abundance, dispersion, gene_bin_size=bin_size)


def _refit_intercepts(
    counts: np.ndarray,
    exposure: np.ndarray,
    gene_indices: np.ndarray,
    trend: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        [
            offset_intercept(
                dense_gene_row(counts, int(gene_index)), exposure, float(dispersion)
            )[0]
            for gene_index, dispersion in zip(gene_indices, trend, strict=True)
        ],
        dtype=np.float64,
    )


def main() -> None:
    args = parse_args()
    if args.top < 1:
        raise ValueError("top must be positive")

    counts, gene_names, cell_names = load_fixture(args.fixture)
    manifest = load_manifest(args.manifest)
    oracle = load_r_oracle(args.r_oracle)
    selected_cells = np.asarray(manifest.selected_cell_indices, dtype=np.int64)
    fit_counts = counts[:, selected_cells]
    library_size = np.asarray(counts.sum(axis=0), dtype=np.float64).ravel()
    fit_library_size = library_size[selected_cells]

    gene_lookup = {name: index for index, name in enumerate(gene_names)}
    fit_gene_names = tuple(str(name) for name in oracle["fit_genes"])
    fit_gene_indices = np.asarray(
        [gene_lookup[name] for name in fit_gene_names], dtype=np.int64
    )
    fits = [
        fit_cox_reid_offset(
            dense_gene_row(fit_counts, int(gene_index)), fit_library_size
        )
        for gene_index in fit_gene_indices
    ]
    preliminary_intercept = np.asarray([fit.intercept for fit in fits])
    theta = np.asarray([fit.theta for fit in fits])
    dispersion = np.zeros(theta.shape, dtype=np.float64)
    finite_theta = np.isfinite(theta)
    dispersion[finite_theta] = 1.0 / theta[finite_theta]
    abundance = np.exp(preliminary_intercept) * float(np.mean(fit_library_size))

    candidate_results: dict[str, dict[str, Any]] = {}
    candidate_trends: dict[str, np.ndarray] = {}
    candidate_intercepts: dict[str, np.ndarray] = {}
    for value in args.bin_sizes:
        bin_size = _resolve_bin_size(value, len(fit_gene_names))
        label = "all" if bin_size == len(fit_gene_names) else str(bin_size)
        trend = _batched_trend(abundance, dispersion, bin_size)
        intercept = _refit_intercepts(
            fit_counts, fit_library_size, fit_gene_indices, trend
        )
        metrics = numeric_metrics(intercept, oracle["raw_intercept"])
        metrics["top_absolute_error_genes"] = top_absolute_errors(
            fit_gene_names,
            intercept,
            oracle["raw_intercept"],
            limit=args.top,
        )
        candidate_results[label] = {"bin_size": bin_size, "metrics": metrics}
        candidate_trends[label] = trend
        candidate_intercepts[label] = intercept

    baseline_label = (
        "all" if "all" in candidate_intercepts else next(iter(candidate_intercepts))
    )
    baseline_error = np.abs(
        candidate_intercepts[baseline_label] - oracle["raw_intercept"]
    )
    target_indices = np.argsort(-baseline_error, kind="stable")[: args.top]
    targets: list[dict[str, Any]] = []
    for index in target_indices:
        gene_index = int(fit_gene_indices[index])
        y = dense_gene_row(fit_counts, gene_index)
        fit_trace: list[dict[str, float | int | str | bool]] = []
        fit_cox_reid_offset(y, fit_library_size, trace=fit_trace.append)
        refit_traces: dict[str, list[dict[str, float | int | str | bool]]] = {}
        for label, trend in candidate_trends.items():
            events: list[dict[str, float | int | str | bool]] = []
            offset_intercept(
                y,
                fit_library_size,
                float(trend[index]),
                trace=events.append,
                trace_stage=f"local_trend_refit_{label}",
            )
            refit_traces[label] = events
        targets.append(
            {
                "gene": fit_gene_names[index],
                "counts": y.tolist(),
                "preliminary_intercept": float(preliminary_intercept[index]),
                "raw_theta": float(theta[index]),
                "oracle_final_intercept": float(oracle["raw_intercept"][index]),
                "candidate_dispersion_trend": {
                    label: float(trend[index])
                    for label, trend in candidate_trends.items()
                },
                "candidate_final_intercept": {
                    label: float(intercept[index])
                    for label, intercept in candidate_intercepts.items()
                },
                "fit_trace": fit_trace,
                "refit_traces": refit_traces,
            }
        )

    report = {
        "schema_version": 1,
        "generated_on": "2026-11-02",
        "hypothesis": (
            "R-v2 raw-intercept differences are dominated by the gene bin size "
            "used for the local dispersion trend, not offset-root tolerance."
        ),
        "input": {
            "shape_genes_by_cells": list(counts.shape),
            "counts_sha256": count_sha256(counts),
            "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
            "fit_gene_count": len(fit_gene_names),
            "fit_cell_count": len(selected_cells),
            "fit_cell_names": [cell_names[index] for index in selected_cells],
            "fit_library_size": fit_library_size.tolist(),
        },
        "candidates": candidate_results,
        "targets": targets,
    }
    write_json(args.output, report)
    print(args.output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
