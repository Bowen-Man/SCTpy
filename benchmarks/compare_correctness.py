from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from sctransform_native import SCTConfig, fit_transform
from sctransform_native._fit import theta_moment_ratio_mask

from ._common import count_sha256, load_fixture, load_manifest, write_json
from .correctness import (
    THRESHOLDS,
    gate_metrics,
    jaccard,
    named_set_metrics,
    numeric_mask_metrics,
    numeric_metrics,
    relative_error_metrics,
    top_absolute_errors,
    top_row_rmse,
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
        "fit_cells": (
            _read_lines(path / "fit_cells.txt")
            if (path / "fit_cells.txt").exists()
            else None
        ),
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
    reference["fit_cells"] = np.asarray(reference["cells"])[
        np.asarray(reference["fit_cell_indices"], dtype=np.int64)
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
        left_values, right_values, names = _aligned(
            left, right, field=field, names_field=names_field
        )
        metrics = numeric_metrics(left_values, right_values)
        metrics["gate"] = gate_metrics(metrics, THRESHOLDS[threshold_name])
        metrics["mask_agreement"] = numeric_mask_metrics(left_values, right_values)
        metrics["top_absolute_error_genes"] = top_absolute_errors(
            names, left_values, right_values
        )
        report[field] = metrics

    for field in ("raw_theta", "regularized_theta"):
        names_field = "fit_genes" if field == "raw_theta" else "genes"
        left_values, right_values, names = _aligned(
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
        metrics["mask_agreement"] = numeric_mask_metrics(left_values, right_values)
        metrics["relative_error"] = relative_error_metrics(
            left_values[finite], right_values[finite]
        )
        metrics["top_absolute_error_genes"] = top_absolute_errors(
            tuple(np.asarray(names)[finite]),
            np.log10(left_values[finite]),
            np.log10(right_values[finite]),
        )
        report[f"log10_{field}"] = metrics

    if "raw_infinite_theta_genes" in left and "raw_infinite_theta_genes" in right:
        report["raw_infinite_theta_genes"] = named_set_metrics(
            left["raw_infinite_theta_genes"], right["raw_infinite_theta_genes"]
        )

    left_residuals, right_residuals, residual_genes = _aligned(
        left, right, field="residuals", names_field="genes"
    )
    residual_metrics = numeric_metrics(left_residuals, right_residuals)
    residual_metrics["gate"] = gate_metrics(residual_metrics, THRESHOLDS["residual"])
    residual_metrics["mask_agreement"] = numeric_mask_metrics(
        left_residuals, right_residuals
    )
    residual_metrics["top_gene_rmse"] = top_row_rmse(
        residual_genes, left_residuals, right_residuals
    )
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
    left_hvg = tuple(str(value) for value in left["hvg_genes"])
    right_hvg = tuple(str(value) for value in right["hvg_genes"])
    right_hvg_set = set(right_hvg)
    common_hvg = tuple(name for name in left_hvg if name in right_hvg_set)
    left_rank = {name: index for index, name in enumerate(left_hvg)}
    right_rank = {name: index for index, name in enumerate(right_hvg)}
    rank_metrics = numeric_metrics(
        np.asarray([left_rank[name] for name in common_hvg]),
        np.asarray([right_rank[name] for name in common_hvg]),
    )
    hvg["complete_order_exact"] = left_hvg == right_hvg
    hvg["complete_gene_set"] = named_set_metrics(left_hvg, right_hvg)
    hvg["complete_rank_spearman"] = rank_metrics["spearman"]
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
    fit_method: str = "nb_mle",
    dispersion_regularization: str = "native",
    regularization_method: str = "native",
    regularization_support: str = "extrapolate",
    theta_postprocess: str = "none",
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
            fit_method=fit_method,
            dispersion_regularization=dispersion_regularization,
            regularization_method=regularization_method,
            regularization_support=regularization_support,
            theta_postprocess=theta_postprocess,
            theta_moment_ratio_threshold=THETA_MOMENT_RATIO_THRESHOLD,
        ),
        selection=manifest,
    )
    eligible_indices = np.flatnonzero(result.eligible_gene)
    fit_indices = np.flatnonzero(result.selected_gene & ~result.poisson_gene)
    selected_indices = np.flatnonzero(result.selected_gene)
    selected_cells = np.asarray(manifest.selected_cell_indices, dtype=np.int64)
    fit_counts = counts[:, selected_cells]
    fit_mean = np.mean(fit_counts, axis=1, dtype=np.float64)
    fit_variance = np.var(fit_counts, axis=1, ddof=1, dtype=np.float64)
    if theta_postprocess == "moment_ratio":
        moment_ratio = result.theta_postprocessed[fit_indices] | np.isinf(
            result.raw_theta[fit_indices]
        )
    else:
        moment_ratio = theta_moment_ratio_mask(
            fit_mean[fit_indices],
            fit_variance[fit_indices],
            result.raw_theta[fit_indices],
            threshold=THETA_MOMENT_RATIO_THRESHOLD,
        )
    raw_infinite = fit_indices[np.isinf(result.raw_theta[fit_indices])]
    iteration_summary: dict[str, float | int | None]
    if fit_indices.size:
        iteration_summary = {
            "median": float(np.median(result.fit_iterations[fit_indices])),
            "max": int(np.max(result.fit_iterations[fit_indices])),
        }
    else:
        iteration_summary = {"median": None, "max": 0}
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
            "fit_status_counts": dict(
                Counter(result.fit_status[index] for index in selected_indices)
            ),
            "converged_gene_count": int(
                np.count_nonzero(result.converged[fit_indices])
            ),
            "fit_gene_count": int(fit_indices.size),
            "fit_iterations": iteration_summary,
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
        "--fit-method",
        choices=("nb_mle", "cox_reid"),
        default="nb_mle",
    )
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
    parser.add_argument("--enforce", action="store_true")
    args = parser.parse_args()

    level_a = load_r_oracle(args.r_oracle)
    level_b = load_reference(args.reference)
    level_c, native_timing = native_result(
        args.fixture,
        args.manifest,
        fit_method=args.fit_method,
        dispersion_regularization=args.dispersion_regularization,
        regularization_method=args.regularization_method,
        regularization_support=args.regularization_support,
        theta_postprocess=args.theta_postprocess,
    )
    counts, fixture_genes, fixture_cells = load_fixture(args.fixture)
    manifest = load_manifest(args.manifest)
    level_c_vs_level_a = compare_pair(level_c, level_a)
    level_c_vs_level_b = compare_pair(level_c, level_b)
    level_a_fit_cells = level_a.get("fit_cells")
    selection_stage = {
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "algorithm": manifest.algorithm,
        "selected_gene_count": len(manifest.selected_gene_indices),
        "selected_cell_count": len(manifest.selected_cell_indices),
        "raw_fit_gene_set": named_set_metrics(
            level_c["fit_genes"], level_a["fit_genes"]
        ),
        "fit_cell_order_exact": (
            None
            if level_a_fit_cells is None
            else tuple(manifest.selected_cell_names)
            == tuple(str(value) for value in level_a_fit_cells)
        ),
        "fit_cell_oracle_status": (
            "not_exported" if level_a_fit_cells is None else "available"
        ),
        "cross_language_seed_comparison": "not_used; manifest names define selection",
    }
    report = {
        "schema_version": 1,
        "generated_on": "2026-11-02",
        "fit_method": args.fit_method,
        "dispersion_regularization": args.dispersion_regularization,
        "regularization_method": args.regularization_method,
        "regularization_support": args.regularization_support,
        "theta_postprocess": args.theta_postprocess,
        "theta_moment_ratio_threshold": THETA_MOMENT_RATIO_THRESHOLD,
        "thresholds_preregistered": THRESHOLDS,
        "level_b_vs_level_a": compare_pair(level_b, level_a),
        "level_c_vs_level_a": level_c_vs_level_a,
        "level_c_vs_level_b": level_c_vs_level_b,
        "theta_moment_ratio_vs_level_a_raw_infinite": named_set_metrics(
            level_c["theta_moment_ratio_genes"],
            level_a["raw_infinite_theta_genes"],
        ),
        "level_c_timing_seconds": native_timing,
        "stages": {
            "input": {
                "shape_genes_by_cells": list(counts.shape),
                "counts_sha256": count_sha256(counts),
                "fixture_gene_count": len(fixture_genes),
                "fixture_cell_count": len(fixture_cells),
                "oracle_cell_order_exact": tuple(level_a["cells"]) == fixture_cells,
            },
            "selection": selection_stage,
            "raw_fitting": {
                "intercept": level_c_vs_level_a["raw_intercept"],
                "log10_theta": level_c_vs_level_a["log10_raw_theta"],
                "infinite_theta_genes": level_c_vs_level_a["raw_infinite_theta_genes"],
                "level_c_optimizer": {
                    "fit_gene_count": level_c["fit_gene_count"],
                    "converged_gene_count": level_c["converged_gene_count"],
                    "status_counts": level_c["fit_status_counts"],
                    "iterations": level_c["fit_iterations"],
                },
                "level_a_optimizer_status": "not_exposed_by_oracle_artifact",
            },
            "regularization": {
                "intercept": level_c_vs_level_a["regularized_intercept"],
                "log10_theta": level_c_vs_level_a["log10_regularized_theta"],
                "parameter_outliers": level_c_vs_level_a.get("parameter_outliers"),
            },
            "residual": {
                "matrix": level_c_vs_level_a["residuals"],
                "mean": level_c_vs_level_a["residual_mean"],
                "variance": level_c_vs_level_a["residual_variance"],
            },
            "hvg": level_c_vs_level_a["hvg"],
        },
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
