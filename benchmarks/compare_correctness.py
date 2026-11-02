from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from sctransform_native import SCTConfig, fit_transform

from ._common import load_fixture, load_manifest, write_json
from .correctness import (
    THRESHOLDS,
    gate_metrics,
    jaccard,
    named_set_metrics,
    numeric_metrics,
    theta_moment_ratio_mask,
)

THETA_MOMENT_RATIO_THRESHOLD = 1.0e-3


def _read_lines(path: Path) -> tuple[str, ...]:
    if not path.exists():
        return ()
    return tuple(line for line in path.read_text(encoding="utf-8").splitlines() if line)


def load_r_oracle(path: Path) -> dict[str, Any]:
    genes = _read_lines(path / "genes.txt")
    cells = _read_lines(path / "cells.txt")
    fit_genes = _read_lines(path / "fit_genes.txt")
    gene_count = len(genes)
    cell_count = len(cells)
    oracle = {
        "genes": genes,
        "cells": cells,
        "fit_genes": fit_genes,
        "regularized_intercept": np.fromfile(
            path / "regularized_intercept.f64", dtype="<f8"
        ),
        "regularized_slope": np.fromfile(path / "regularized_slope.f64", dtype="<f8"),
        "regularized_theta": np.fromfile(path / "regularized_theta.f64", dtype="<f8"),
        "raw_intercept": np.fromfile(path / "raw_intercept.f64", dtype="<f8"),
        "raw_theta": np.fromfile(path / "raw_theta.f64", dtype="<f8"),
        "residuals": np.fromfile(path / "residuals.f64", dtype="<f8").reshape(
            gene_count, cell_count
        ),
        "residual_mean": np.fromfile(path / "residual_mean.f64", dtype="<f8"),
        "residual_variance": np.fromfile(path / "residual_variance.f64", dtype="<f8"),
        "hvg_genes": _read_lines(path / "hvg_genes.txt"),
        "poisson_genes": _read_lines(path / "poisson_genes.txt"),
        "outlier_genes": (
            _read_lines(path / "outlier_genes.txt")
            if (path / "outlier_genes.txt").exists()
            else None
        ),
    }
    oracle["raw_infinite_theta_genes"] = np.asarray(fit_genes)[
        np.isinf(oracle["raw_theta"])
    ]
    return oracle


def load_reference(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as value:
        reference = {key: value[key] for key in value.files}
    reference["raw_infinite_theta_genes"] = np.asarray(reference["fit_genes"])[
        np.isinf(reference["raw_theta"])
    ]
    return reference


def _indices(names: tuple[str, ...] | np.ndarray) -> dict[str, int]:
    return {str(name): index for index, name in enumerate(names)}


def _aligned(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    field: str,
    names_field: str = "genes",
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    left_names = tuple(str(value) for value in left[names_field])
    right_names = tuple(str(value) for value in right[names_field])
    right_lookup = _indices(right_names)
    common = tuple(name for name in left_names if name in right_lookup)
    left_lookup = _indices(left_names)
    return (
        np.asarray(left[field])[[left_lookup[name] for name in common]],
        np.asarray(right[field])[[right_lookup[name] for name in common]],
        common,
    )


def compare_pair(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for field, threshold_name in (
        ("raw_intercept", "intercept"),
        ("regularized_intercept", "intercept"),
        ("regularized_slope", "slope"),
        ("residual_mean", "residual"),
        ("residual_variance", "residual_variance"),
    ):
        names_field = "fit_genes" if field == "raw_intercept" else "genes"
        left_values, right_values, _ = _aligned(
            left, right, field=field, names_field=names_field
        )
        metrics = numeric_metrics(left_values, right_values)
        metrics["gate"] = gate_metrics(metrics, THRESHOLDS[threshold_name])
        report[field] = metrics

    for field in ("raw_theta", "regularized_theta"):
        names_field = "fit_genes" if field == "raw_theta" else "genes"
        left_values, right_values, _ = _aligned(
            left, right, field=field, names_field=names_field
        )
        finite = (
            np.isfinite(left_values)
            & np.isfinite(right_values)
            & (left_values > 0)
            & (right_values > 0)
        )
        metrics = numeric_metrics(
            np.log10(left_values[finite]), np.log10(right_values[finite])
        )
        metrics["gate"] = gate_metrics(metrics, THRESHOLDS["log10_theta"])
        report[f"log10_{field}"] = metrics

    if "raw_infinite_theta_genes" in left and "raw_infinite_theta_genes" in right:
        report["raw_infinite_theta_genes"] = named_set_metrics(
            left["raw_infinite_theta_genes"], right["raw_infinite_theta_genes"]
        )

    left_residuals, right_residuals, _ = _aligned(
        left, right, field="residuals", names_field="genes"
    )
    residual_metrics = numeric_metrics(left_residuals, right_residuals)
    residual_metrics["gate"] = gate_metrics(residual_metrics, THRESHOLDS["residual"])
    report["residuals"] = residual_metrics
    poisson_metrics = {
        "jaccard": jaccard(left["poisson_genes"], right["poisson_genes"])
    }
    poisson_metrics["gate"] = gate_metrics(poisson_metrics, THRESHOLDS["poisson"])
    report["poisson_genes"] = poisson_metrics
    left_outliers = left.get("outlier_genes")
    right_outliers = right.get("outlier_genes")
    if left_outliers is not None and right_outliers is not None:
        report["parameter_outliers"] = named_set_metrics(left_outliers, right_outliers)

    eligible_count = min(len(left["hvg_genes"]), len(right["hvg_genes"]))
    hvg: dict[str, Any] = {}
    for top in (1000, 2000, 3000):
        if eligible_count >= top:
            value = jaccard(left["hvg_genes"][:top], right["hvg_genes"][:top])
            hvg[str(top)] = {
                "jaccard": value,
                "intersection": len(
                    set(left["hvg_genes"][:top]) & set(right["hvg_genes"][:top])
                ),
                "gate": value >= THRESHOLDS["hvg"]["jaccard"],
            }
    report["hvg"] = hvg
    return report


def native_result(
    fixture: Path,
    manifest_path: Path,
    *,
    dispersion_regularization: str = "native",
) -> tuple[dict[str, Any], dict[str, float]]:
    counts, genes, cells = load_fixture(fixture)
    manifest = load_manifest(manifest_path)
    result = fit_transform(
        sparse.csr_matrix(counts),
        gene_names=genes,
        cell_names=cells,
        config=SCTConfig(
            min_cells=5,
            n_genes=None,
            n_cells=None,
            dispersion_regularization=dispersion_regularization,
        ),
        selection=manifest,
    )
    eligible_indices = np.flatnonzero(result.eligible_gene)
    fit_indices = np.flatnonzero(result.selected_gene & ~result.poisson_gene)
    selected_cells = np.asarray(manifest.selected_cell_indices, dtype=np.int64)
    fit_counts = counts[:, selected_cells]
    fit_mean = np.mean(fit_counts, axis=1, dtype=np.float64)
    fit_variance = np.var(fit_counts, axis=1, ddof=1, dtype=np.float64)
    moment_ratio = theta_moment_ratio_mask(
        fit_mean[fit_indices],
        fit_variance[fit_indices],
        result.raw_theta[fit_indices],
        threshold=THETA_MOMENT_RATIO_THRESHOLD,
    )
    raw_infinite = fit_indices[np.isinf(result.raw_theta[fit_indices])]
    return (
        {
            "genes": np.asarray(result.gene_names)[eligible_indices],
            "cells": np.asarray(result.cell_names),
            "fit_genes": np.asarray(result.gene_names)[fit_indices],
            "regularized_intercept": result.regularized_intercept[eligible_indices],
            "regularized_slope": result.slope[eligible_indices],
            "regularized_theta": result.regularized_theta[eligible_indices],
            "raw_intercept": result.raw_intercept[fit_indices],
            "raw_theta": result.raw_theta[fit_indices],
            "raw_infinite_theta_genes": np.asarray(result.gene_names)[raw_infinite],
            "theta_moment_ratio_genes": np.asarray(result.gene_names)[
                fit_indices[moment_ratio]
            ],
            "residuals": result.residuals[eligible_indices],
            "residual_mean": result.residual_mean[eligible_indices],
            "residual_variance": result.residual_variance[eligible_indices],
            "hvg_genes": np.asarray(result.gene_names)[result.hvg_order],
            "poisson_genes": np.asarray(result.gene_names)[
                np.flatnonzero(result.poisson_gene & result.eligible_gene)
            ],
            "outlier_genes": np.asarray(result.gene_names)[
                np.flatnonzero(result.parameter_outlier)
            ],
        },
        result.timing,
    )


def _all_gates(value: Any) -> list[bool]:
    if isinstance(value, dict):
        gates = [bool(value["gate"])] if "gate" in value else []
        for child in value.values():
            gates.extend(_all_gates(child))
        return gates
    return []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--r-oracle", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dispersion-regularization",
        choices=("native", "sct_v2"),
        default="native",
    )
    parser.add_argument("--enforce", action="store_true")
    args = parser.parse_args()

    level_a = load_r_oracle(args.r_oracle)
    level_b = load_reference(args.reference)
    level_c, native_timing = native_result(
        args.fixture,
        args.manifest,
        dispersion_regularization=args.dispersion_regularization,
    )
    report = {
        "schema_version": 1,
        "generated_on": "2026-11-02",
        "dispersion_regularization": args.dispersion_regularization,
        "theta_moment_ratio_threshold": THETA_MOMENT_RATIO_THRESHOLD,
        "thresholds_preregistered": THRESHOLDS,
        "level_b_vs_level_a": compare_pair(level_b, level_a),
        "level_c_vs_level_a": compare_pair(level_c, level_a),
        "level_c_vs_level_b": compare_pair(level_c, level_b),
        "theta_moment_ratio_vs_level_a_raw_infinite": named_set_metrics(
            level_c["theta_moment_ratio_genes"],
            level_a["raw_infinite_theta_genes"],
        ),
        "level_c_timing_seconds": native_timing,
    }
    gates = _all_gates(report["level_c_vs_level_a"]) + _all_gates(
        report["level_c_vs_level_b"]
    )
    report["native_gate_passed"] = bool(gates and all(gates))
    write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.enforce and not report["native_gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
