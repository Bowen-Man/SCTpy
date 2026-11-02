from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from sctransform_native._matrix import matrix_statistics
from sctransform_native._regularize import regularize_parameters

from ._common import load_fixture, write_json
from .compare_correctness import load_r_oracle
from .correctness import named_set_metrics, numeric_metrics

REGULARIZATION_REINJECTION_THRESHOLDS = {
    "rmse": 1.0e-10,
    "max_absolute_error": 1.0e-9,
}


def _metric_gate(metrics: dict[str, float | int | None]) -> bool:
    return bool(
        metrics["rmse"] <= REGULARIZATION_REINJECTION_THRESHOLDS["rmse"]
        and metrics["max_absolute_error"]
        <= REGULARIZATION_REINJECTION_THRESHOLDS["max_absolute_error"]
    )


def regularization_reinjection_report(
    counts: np.ndarray,
    gene_names: tuple[str, ...],
    oracle: dict[str, Any],
) -> dict[str, Any]:
    """Recompute regularized parameters from an oracle's raw parameters."""
    gene_lookup = {name: index for index, name in enumerate(gene_names)}
    oracle_genes = tuple(str(value) for value in oracle["genes"])
    fit_genes = tuple(str(value) for value in oracle["fit_genes"])
    required = set(oracle_genes) | set(fit_genes)
    missing = tuple(sorted(required - gene_lookup.keys()))
    if missing:
        raise ValueError(f"oracle genes are absent from the fixture: {missing[:3]}")

    genes = len(gene_names)
    eligible = np.zeros(genes, dtype=bool)
    eligible[[gene_lookup[name] for name in oracle_genes]] = True
    poisson = np.zeros(genes, dtype=bool)
    poisson[[gene_lookup[str(name)] for name in oracle["poisson_genes"]]] = True
    raw_intercept = np.full(genes, np.nan, dtype=np.float64)
    raw_theta = np.full(genes, np.nan, dtype=np.float64)
    fit_indices = np.asarray([gene_lookup[name] for name in fit_genes], dtype=np.int64)
    raw_intercept[fit_indices] = oracle["raw_intercept"]
    raw_theta[fit_indices] = oracle["raw_theta"]

    gene_mean, _, _, geometric_mean = matrix_statistics(counts)
    prediction_mask = eligible & ~poisson
    training_mask = prediction_mask & np.isfinite(raw_theta)
    intercept, theta, local_outlier, bandwidth = regularize_parameters(
        gene_mean=gene_mean,
        geometric_mean=geometric_mean,
        raw_intercept=raw_intercept,
        raw_theta=raw_theta,
        training_mask=training_mask,
        prediction_mask=prediction_mask,
        bandwidth_adjust=3.0,
        outlier_mad_threshold=10.0,
        theta_min=1.0e-4,
        theta_max=1.0e8,
        dispersion_regularization="sct_v2",
        regularization_method="r_stats",
        regularization_support="clamp",
    )

    oracle_indices = np.asarray(
        [gene_lookup[name] for name in oracle_genes], dtype=np.int64
    )
    oracle_theta = np.asarray(oracle["regularized_theta"], dtype=np.float64)
    non_poisson = np.isfinite(oracle_theta)
    intercept_metrics = numeric_metrics(
        intercept[oracle_indices][non_poisson],
        np.asarray(oracle["regularized_intercept"])[non_poisson],
    )
    theta_metrics = numeric_metrics(
        np.log10(theta[oracle_indices][non_poisson]),
        np.log10(oracle_theta[non_poisson]),
    )
    intercept_metrics["gate"] = _metric_gate(intercept_metrics)
    theta_metrics["gate"] = _metric_gate(theta_metrics)

    finite_oracle_outliers = set(oracle.get("outlier_genes") or ()) - set(
        np.asarray(fit_genes)[np.isinf(oracle["raw_theta"])]
    )
    local_outlier_names = np.asarray(gene_names)[np.flatnonzero(local_outlier)]
    outlier_metrics = named_set_metrics(local_outlier_names, finite_oracle_outliers)
    outlier_metrics["gate"] = outlier_metrics["jaccard"] == 1.0
    gate_passed = bool(
        intercept_metrics["gate"] and theta_metrics["gate"] and outlier_metrics["gate"]
    )
    return {
        "schema_version": 1,
        "experiment": "oracle_raw_parameter_regularization_reinjection",
        "thresholds": REGULARIZATION_REINJECTION_THRESHOLDS,
        "bandwidth": bandwidth,
        "regularized_intercept": intercept_metrics,
        "log10_regularized_theta": theta_metrics,
        "finite_parameter_outliers": outlier_metrics,
        "gate_passed": gate_passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--r-oracle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--enforce", action="store_true")
    args = parser.parse_args()

    counts, genes, _ = load_fixture(args.fixture)
    report = regularization_reinjection_report(
        counts, genes, load_r_oracle(args.r_oracle)
    )
    write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.enforce and not report["gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
