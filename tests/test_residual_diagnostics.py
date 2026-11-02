import numpy as np

from benchmarks.diagnose_residuals import residual_reinjection_report


def test_final_parameter_reinjection_matches_residual_oracle() -> None:
    counts = np.array(
        [
            [1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1],
        ],
        dtype=np.int64,
    )
    intercept = np.full(2, np.log(0.5), dtype=np.float64)
    theta = np.full(2, np.inf, dtype=np.float64)
    residuals = (counts - 0.5) / np.sqrt(0.5)
    residual_variance = np.var(residuals, axis=1, ddof=1)
    oracle = {
        "genes": ("g0", "g1"),
        "cells": tuple(f"c{index}" for index in range(6)),
        "regularized_intercept": intercept,
        "regularized_theta": theta,
        "residuals": residuals,
        "residual_mean": np.mean(residuals, axis=1),
        "residual_variance": residual_variance,
        "hvg_genes": ("g0", "g1"),
    }

    report = residual_reinjection_report(
        counts,
        ("g0", "g1"),
        oracle["cells"],
        oracle,
    )

    assert report["gate_passed"]
    assert report["metrics"]["residuals"]["max_absolute_error"] == 0.0
    assert report["hvg"]["exact_order"]
