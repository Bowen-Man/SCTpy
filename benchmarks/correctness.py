from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.stats import rankdata

THRESHOLDS: dict[str, dict[str, float]] = {
    "slope": {"max_absolute_error": 1e-12},
    "poisson": {"jaccard": 1.0},
    "intercept": {"mae": 0.10, "pearson": 0.995},
    "log10_theta": {"mae": 0.25, "spearman": 0.90},
    "residual": {"rmse": 0.15, "pearson": 0.99},
    "residual_variance": {"spearman": 0.98},
    "hvg": {"jaccard": 0.90},
}


def numeric_metrics(
    left: np.ndarray, right: np.ndarray
) -> dict[str, float | int | None]:
    left = np.asarray(left, dtype=np.float64).ravel()
    right = np.asarray(right, dtype=np.float64).ravel()
    finite = np.isfinite(left) & np.isfinite(right)
    left = left[finite]
    right = right[finite]
    if left.size == 0:
        return {
            "n": 0,
            "mae": float("nan"),
            "rmse": float("nan"),
            "max_absolute_error": float("nan"),
            "pearson": None,
            "spearman": None,
        }
    difference = left - right
    scale = max(float(np.max(np.abs(left))), float(np.max(np.abs(right))), 1.0)
    constant_tolerance = 64.0 * np.finfo(np.float64).eps * scale
    effectively_constant = (
        float(np.ptp(left)) <= constant_tolerance
        or float(np.ptp(right)) <= constant_tolerance
    )
    if left.size < 2 or effectively_constant:
        pearson = None
        spearman = None
    else:
        pearson = float(np.corrcoef(left, right)[0, 1])
        spearman = float(
            np.corrcoef(
                rankdata(left, method="average"), rankdata(right, method="average")
            )[0, 1]
        )
    return {
        "n": int(left.size),
        "mae": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "max_absolute_error": float(np.max(np.abs(difference))),
        "pearson": pearson,
        "spearman": spearman,
    }


def jaccard(left: Sequence[str], right: Sequence[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    return 1.0 if not union else len(left_set & right_set) / len(union)


def named_set_metrics(
    left: Sequence[str], right: Sequence[str]
) -> dict[str, int | float]:
    left_set = set(left)
    right_set = set(right)
    intersection = left_set & right_set
    union = left_set | right_set
    return {
        "left_count": len(left_set),
        "right_count": len(right_set),
        "intersection": len(intersection),
        "union": len(union),
        "jaccard": 1.0 if not union else len(intersection) / len(union),
    }


def gate_metrics(metrics: dict[str, Any], thresholds: dict[str, float]) -> bool:
    for name, threshold in thresholds.items():
        raw_value = metrics[name]
        if raw_value is None:
            return False
        value = float(raw_value)
        if not np.isfinite(value):
            return False
        if name in {"mae", "rmse", "max_absolute_error"}:
            if value > threshold:
                return False
        elif value < threshold:
            return False
    return True
