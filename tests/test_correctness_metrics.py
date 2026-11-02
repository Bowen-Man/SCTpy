import numpy as np

from benchmarks.correctness import (
    gate_metrics,
    jaccard,
    named_set_metrics,
    numeric_mask_metrics,
    numeric_metrics,
    relative_error_metrics,
    top_absolute_errors,
    top_row_rmse,
)


def test_numeric_metrics_report_all_required_statistics() -> None:
    metrics = numeric_metrics(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.5, 2.5]))
    assert set(metrics) == {
        "n",
        "mae",
        "rmse",
        "max_absolute_error",
        "pearson",
        "spearman",
    }
    assert metrics["n"] == 3
    assert metrics["max_absolute_error"] == 0.5


def test_jaccard_and_gate_direction() -> None:
    assert jaccard(("a", "b"), ("b", "c")) == 1 / 3
    assert gate_metrics({"mae": 0.01, "pearson": 0.999}, {"mae": 0.1, "pearson": 0.99})
    assert not gate_metrics(
        {"mae": 0.2, "pearson": 0.999}, {"mae": 0.1, "pearson": 0.99}
    )


def test_named_set_metrics_reports_overlap_and_sizes() -> None:
    metrics = named_set_metrics(("a", "b"), ("b", "c", "d"))
    assert metrics == {
        "left_count": 2,
        "right_count": 3,
        "intersection": 1,
        "union": 4,
        "jaccard": 0.25,
    }


def test_effectively_constant_arrays_report_undefined_correlations() -> None:
    fixed = np.full(8, np.log(10.0))
    perturbed = fixed + np.linspace(-2.0e-15, 2.0e-15, fixed.size)
    metrics = numeric_metrics(perturbed, fixed)
    assert metrics["pearson"] is None
    assert metrics["spearman"] is None


def test_mask_relative_and_top_error_diagnostics() -> None:
    left = np.array([1.0, 4.0, np.inf, np.nan])
    right = np.array([1.0, 2.0, np.inf, np.nan])

    masks = numeric_mask_metrics(left, right)
    assert masks["exact"]
    assert masks["mismatch_count"] == 0
    assert masks["left"] == {
        "finite": 2,
        "positive_infinity": 1,
        "negative_infinity": 0,
        "nan": 1,
    }
    relative = relative_error_metrics(left, right)
    assert relative["n"] == 2
    assert relative["max"] == 1.0
    top = top_absolute_errors(("a", "b", "c", "d"), left, right, limit=1)
    assert top == [{"name": "b", "left": 4.0, "right": 2.0, "absolute_error": 2.0}]


def test_top_row_rmse_reports_named_rows() -> None:
    top = top_row_rmse(
        ("small", "large"),
        np.array([[0.0, 1.0], [0.0, 4.0]]),
        np.array([[0.0, 1.0], [0.0, 0.0]]),
        limit=1,
    )
    assert top == [{"name": "large", "rmse": np.sqrt(8.0)}]
