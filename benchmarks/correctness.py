from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy import sparse
from scipy.stats import rankdata

THRESHOLDS: dict[str, dict[str, float]] = {
    "slope": {"max_absolute_error": 1e-12},
    "poisson": {"jaccard": 1.0},
    "intercept": {"mae": 0.10, "pearson": 0.995},
    "log10_theta": {"mae": 0.25, "spearman": 0.90},
    "residual": {"rmse": 0.15, "pearson": 0.99},
    "residual_variance": {"spearman": 0.98},
    "corrected_umi": {
        "mae": 0.01,
        "max_absolute_error": 2.0,
        "exact_fraction": 0.99,
    },
    "hvg": {"jaccard": 0.90},
}


def numeric_metrics(
    left: np.ndarray, right: np.ndarray
) -> dict[str, float | int | None]:
    left = np.asarray(left, dtype=np.float64).ravel()
    right = np.asarray(right, dtype=np.float64).ravel()
    if left.shape != right.shape:
        raise ValueError("numeric arrays must have identical shapes")
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


def sparse_numeric_metrics(
    left: sparse.spmatrix, right: sparse.spmatrix
) -> dict[str, float | int | None]:
    """Compare sparse matrices without materializing structural zeros."""
    left = sparse.csr_matrix(left, dtype=np.float64)
    right = sparse.csr_matrix(right, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError("sparse matrices must have identical shapes")
    size = int(np.prod(left.shape))
    if size == 0:
        raise ValueError("sparse matrices must not be empty")
    difference = left - right
    difference.eliminate_zeros()
    absolute = np.abs(difference.data)
    squared_sum = float(np.dot(difference.data, difference.data))
    sum_left = float(left.sum())
    sum_right = float(right.sum())
    sum_left_squared = float(left.multiply(left).sum())
    sum_right_squared = float(right.multiply(right).sum())
    sum_product = float(left.multiply(right).sum())
    covariance = size * sum_product - sum_left * sum_right
    variance_left = size * sum_left_squared - sum_left**2
    variance_right = size * sum_right_squared - sum_right**2
    denominator = float(np.sqrt(max(variance_left * variance_right, 0.0)))
    pearson = covariance / denominator if denominator > 0.0 else None
    return {
        "n": size,
        "mae": float(np.sum(absolute) / size),
        "rmse": float(np.sqrt(squared_sum / size)),
        "max_absolute_error": float(np.max(absolute)) if absolute.size else 0.0,
        "pearson": pearson,
        "exact_fraction": float(1.0 - difference.nnz / size),
    }


def numeric_mask_metrics(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    """Compare finite, infinite, and NaN classifications without dropping values."""
    left = np.asarray(left, dtype=np.float64).ravel()
    right = np.asarray(right, dtype=np.float64).ravel()
    if left.shape != right.shape:
        raise ValueError("numeric arrays must have identical shapes")

    def labels(values: np.ndarray) -> np.ndarray:
        output = np.zeros(values.shape, dtype=np.int8)
        output[np.isposinf(values)] = 1
        output[np.isneginf(values)] = 2
        output[np.isnan(values)] = 3
        return output

    left_labels = labels(left)
    right_labels = labels(right)
    names = ("finite", "positive_infinity", "negative_infinity", "nan")
    return {
        "left": {
            name: int(np.count_nonzero(left_labels == index))
            for index, name in enumerate(names)
        },
        "right": {
            name: int(np.count_nonzero(right_labels == index))
            for index, name in enumerate(names)
        },
        "mismatch_count": int(np.count_nonzero(left_labels != right_labels)),
        "exact": bool(np.array_equal(left_labels, right_labels)),
    }


def relative_error_metrics(
    left: np.ndarray, right: np.ndarray
) -> dict[str, float | int]:
    """Report absolute relative error where the reference is finite and nonzero."""
    left = np.asarray(left, dtype=np.float64).ravel()
    right = np.asarray(right, dtype=np.float64).ravel()
    if left.shape != right.shape:
        raise ValueError("numeric arrays must have identical shapes")
    valid = np.isfinite(left) & np.isfinite(right) & (right != 0.0)
    error = np.abs(left[valid] - right[valid]) / np.abs(right[valid])
    if error.size == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "max": float("nan"),
        }
    return {
        "n": int(error.size),
        "mean": float(np.mean(error)),
        "median": float(np.median(error)),
        "max": float(np.max(error)),
    }


def top_absolute_errors(
    names: Sequence[str], left: np.ndarray, right: np.ndarray, *, limit: int = 10
) -> list[dict[str, float | str]]:
    """Return deterministic largest finite absolute errors for named values."""
    left = np.asarray(left, dtype=np.float64).ravel()
    right = np.asarray(right, dtype=np.float64).ravel()
    if left.shape != right.shape or left.size != len(names):
        raise ValueError("names and numeric arrays must have identical lengths")
    finite = np.isfinite(left) & np.isfinite(right)
    indices = np.flatnonzero(finite)
    difference = np.abs(left[indices] - right[indices])
    order = np.lexsort((np.asarray(names, dtype=str)[indices], -difference))[:limit]
    return [
        {
            "name": str(names[index]),
            "left": float(left[index]),
            "right": float(right[index]),
            "absolute_error": float(abs(left[index] - right[index])),
        }
        for index in indices[order]
    ]


def top_row_rmse(
    names: Sequence[str], left: np.ndarray, right: np.ndarray, *, limit: int = 10
) -> list[dict[str, float | str]]:
    """Rank named matrix rows by finite paired RMSE."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2 or left.shape[0] != len(names):
        raise ValueError("names and matrices must have compatible identical shapes")
    finite = np.isfinite(left) & np.isfinite(right)
    count = np.sum(finite, axis=1)
    difference = np.zeros(left.shape, dtype=np.float64)
    np.subtract(left, right, out=difference, where=finite)
    squared = np.sum(difference**2, axis=1)
    rmse = np.full(left.shape[0], np.nan, dtype=np.float64)
    np.divide(squared, count, out=rmse, where=count > 0)
    rmse = np.sqrt(rmse)
    indices = np.flatnonzero(np.isfinite(rmse))
    order = np.lexsort((np.asarray(names, dtype=str)[indices], -rmse[indices]))[:limit]
    return [
        {"name": str(names[index]), "rmse": float(rmse[index])}
        for index in indices[order]
    ]


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
