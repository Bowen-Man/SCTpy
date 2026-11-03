import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from benchmarks.report_schema import (
    load_correctness_report,
    validate_correctness_report,
)

SCHEMA_PATH = (
    Path(__file__).parents[1] / "schemas" / "correctness-report-v1.schema.json"
)


def _named_set() -> dict[str, int | float]:
    return {
        "left_count": 2,
        "right_count": 2,
        "intersection": 2,
        "union": 2,
        "jaccard": 1.0,
    }


def _mask_agreement() -> dict[str, object]:
    counts = {
        "finite": 2,
        "positive_infinity": 0,
        "negative_infinity": 0,
        "nan": 0,
    }
    return {"left": counts, "right": counts, "mismatch_count": 0, "exact": True}


def _metric() -> dict[str, object]:
    return {
        "n": 2,
        "mae": 0.0,
        "rmse": 0.0,
        "max_absolute_error": 0.0,
        "pearson": 1.0,
        "spearman": 1.0,
        "gate": True,
        "mask_agreement": _mask_agreement(),
    }


def _report() -> dict[str, object]:
    metric = _metric()
    corrected_metric = {
        "n": 4,
        "mae": 0.0,
        "rmse": 0.0,
        "max_absolute_error": 0.0,
        "pearson": 1.0,
        "exact_fraction": 1.0,
        "gate": True,
    }
    return {
        "schema_version": 1,
        "native_gate_passed": True,
        "stages": {
            "input": {
                "shape_genes_by_cells": [2, 2],
                "counts_sha256": "0" * 64,
                "fixture_gene_count": 2,
                "fixture_cell_count": 2,
                "oracle_cell_order_exact": True,
            },
            "selection": {
                "manifest_sha256": "1" * 64,
                "algorithm": "fixed_manifest_v1",
                "selected_gene_count": 2,
                "selected_cell_count": 2,
                "raw_fit_gene_set": _named_set(),
                "fit_cell_order_exact": True,
                "fit_cell_oracle_status": "available",
                "cross_language_seed_comparison": "not_used",
            },
            "raw_fitting": {
                "intercept": metric,
                "log10_theta": metric,
                "infinite_theta_genes": _named_set(),
                "level_c_optimizer": {
                    "fit_gene_count": 2,
                    "converged_gene_count": 2,
                    "status_counts": {"fit": 2},
                    "iterations": {"median": 4.0, "max": 5},
                },
                "level_a_optimizer_status": "not_exposed_by_oracle_artifact",
            },
            "regularization": {
                "intercept": metric,
                "log10_theta": metric,
                "parameter_outliers": _named_set(),
            },
            "residual": {
                "matrix": metric,
                "mean": metric,
                "variance": metric,
                "corrected_umi": corrected_metric,
            },
            "hvg": {
                "complete_order_exact": True,
                "complete_gene_set": _named_set(),
                "complete_rank_spearman": 1.0,
            },
        },
    }


def test_correctness_report_v1_matches_published_json_schema() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(_report())


def test_report_reader_accepts_additive_v1_fields(tmp_path) -> None:
    report = _report()
    report["future_additive_field"] = {"value": 1}
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    assert load_correctness_report(path) == report


def test_report_reader_rejects_unknown_versions_and_missing_stages() -> None:
    report = _report()
    report["schema_version"] = 2
    with pytest.raises(ValueError, match="unsupported correctness report schema"):
        validate_correctness_report(report)

    report = _report()
    del report["stages"]["hvg"]  # type: ignore[index]
    with pytest.raises(ValueError, match="missing stages: hvg"):
        validate_correctness_report(report)
