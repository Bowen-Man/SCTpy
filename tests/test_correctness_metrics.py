import numpy as np

from benchmarks.correctness import (
    gate_metrics,
    jaccard,
    named_set_metrics,
    numeric_metrics,
    theta_moment_ratio_mask,
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


def test_theta_moment_ratio_mask_handles_poisson_and_overdispersed_genes() -> None:
    mask = theta_moment_ratio_mask(
        np.array([2.0, 2.0, 2.0]),
        np.array([2.0, 6.0, 6.0]),
        np.array([1.0e8, 1.0e4, 1.0e2]),
        threshold=1.0e-3,
    )
    np.testing.assert_array_equal(mask, np.array([False, True, False]))


def test_effectively_constant_arrays_report_undefined_correlations() -> None:
    fixed = np.full(8, np.log(10.0))
    perturbed = fixed + np.linspace(-2.0e-15, 2.0e-15, fixed.size)
    metrics = numeric_metrics(perturbed, fixed)
    assert metrics["pearson"] is None
    assert metrics["spearman"] is None
