from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from sctpy._matrix import CountMatrix, global_nonzero_median
from sctpy.core import _residuals_and_summary

from ._common import load_fixture, write_json
from .compare_correctness import load_r_oracle
from .correctness import jaccard, numeric_metrics

RESIDUAL_REINJECTION_THRESHOLDS = {
    "rmse": 1.0e-12,
    "max_absolute_error": 1.0e-10,
}


def _metric_gate(metrics: dict[str, float | int | None]) -> bool:
    return bool(
        metrics["rmse"] <= RESIDUAL_REINJECTION_THRESHOLDS["rmse"]
        and metrics["max_absolute_error"]
        <= RESIDUAL_REINJECTION_THRESHOLDS["max_absolute_error"]
    )


def residual_reinjection_report(
    counts: CountMatrix,
    gene_names: tuple[str, ...],
    cell_names: tuple[str, ...],
    oracle: dict[str, Any],
) -> dict[str, Any]:
    """Recompute residual outputs from an oracle's final parameters."""
    oracle_genes = tuple(str(value) for value in oracle["genes"])
    oracle_cells = tuple(str(value) for value in oracle["cells"])
    if oracle_cells != cell_names:
        raise ValueError("oracle and fixture cell names or order differ")
    if len(set(oracle_genes)) != len(oracle_genes):
        raise ValueError("oracle gene names must be unique")

    gene_lookup = {name: index for index, name in enumerate(gene_names)}
    missing = tuple(name for name in oracle_genes if name not in gene_lookup)
    if missing:
        raise ValueError(f"oracle genes are absent from the fixture: {missing[:3]}")
    oracle_indices = np.asarray(
        [gene_lookup[name] for name in oracle_genes], dtype=np.int64
    )

    genes, cells = counts.shape
    expected_shape = (len(oracle_genes), cells)
    oracle_residuals = np.asarray(oracle["residuals"], dtype=np.float64)
    if oracle_residuals.shape != expected_shape:
        raise ValueError(
            f"oracle residual shape {oracle_residuals.shape} != {expected_shape}"
        )

    eligible = np.zeros(genes, dtype=bool)
    eligible[oracle_indices] = True
    intercept = np.full(genes, np.nan, dtype=np.float64)
    theta = np.full(genes, np.nan, dtype=np.float64)
    intercept[oracle_indices] = oracle["regularized_intercept"]
    theta[oracle_indices] = oracle["regularized_theta"]

    library_size = np.asarray(counts.sum(axis=0), dtype=np.float64).ravel()
    nonzero_median = global_nonzero_median(counts)
    if nonzero_median == 0.0:
        raise ValueError("fixture must contain at least one nonzero count")
    clip_limit = float(np.sqrt(cells))
    residuals, _, residual_mean, residual_variance, hvg_order, _ = (
        _residuals_and_summary(
            sparse.csr_matrix(counts),
            library_size=library_size,
            eligible=eligible,
            intercept=intercept,
            theta=theta,
            clip_range=(-clip_limit, clip_limit),
            standard_deviation_floor=nonzero_median / 5.0,
            block_size=256,
            store_residuals=True,
            residual_dtype="float64",
            corrected_umi_depth=None,
        )
    )
    assert residuals is not None

    metrics: dict[str, dict[str, float | int | None | bool]] = {}
    for name, native_values, oracle_values in (
        ("residuals", residuals[oracle_indices], oracle_residuals),
        (
            "residual_mean",
            residual_mean[oracle_indices],
            oracle["residual_mean"],
        ),
        (
            "residual_variance",
            residual_variance[oracle_indices],
            oracle["residual_variance"],
        ),
    ):
        values = numeric_metrics(native_values, oracle_values)
        values["gate"] = _metric_gate(values)
        metrics[name] = values

    native_hvg = tuple(gene_names[index] for index in hvg_order)
    oracle_hvg = tuple(str(value) for value in oracle["hvg_genes"])
    hvg: dict[str, Any] = {
        "exact_order": native_hvg == oracle_hvg,
        "gene_count": len(native_hvg),
    }
    for top in (1000, 2000, 3000):
        if min(len(native_hvg), len(oracle_hvg)) >= top:
            hvg[f"top_{top}_jaccard"] = jaccard(native_hvg[:top], oracle_hvg[:top])

    gate_passed = all(bool(values["gate"]) for values in metrics.values()) and bool(
        hvg["exact_order"]
    )
    return {
        "schema_version": 1,
        "experiment": "oracle_final_parameter_residual_reinjection",
        "thresholds": RESIDUAL_REINJECTION_THRESHOLDS,
        "shape_genes_by_cells": [len(oracle_genes), cells],
        "nonzero_median": nonzero_median,
        "clip_range": [-clip_limit, clip_limit],
        "metrics": metrics,
        "hvg": hvg,
        "gate_passed": gate_passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--r-oracle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--enforce", action="store_true")
    args = parser.parse_args()

    counts, genes, cells = load_fixture(args.fixture)
    report = residual_reinjection_report(
        counts, genes, cells, load_r_oracle(args.r_oracle)
    )
    write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.enforce and not report["gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
