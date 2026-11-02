from __future__ import annotations

import numpy as np


def robust_bandwidth(x: np.ndarray, *, adjust: float) -> float:
    """Silverman-style robust bandwidth for a Gaussian kernel smoother."""
    if x.size < 2:
        return 1.0
    standard_deviation = float(np.std(x, ddof=1))
    q25, q75 = np.percentile(x, [25.0, 75.0])
    robust_scale = float((q75 - q25) / 1.349)
    candidates = [value for value in (standard_deviation, robust_scale) if value > 0]
    scale = min(candidates) if candidates else float(np.ptp(x))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = max(float(np.max(np.abs(x))), 1.0) * np.finfo(np.float64).eps
    bandwidth = 0.9 * scale * x.size ** (-0.2) * adjust
    floor = max(float(np.max(np.abs(x))), 1.0) * np.sqrt(
        np.finfo(np.float64).eps
    )
    return max(float(bandwidth), floor)


def gaussian_nadaraya_watson(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_predict: np.ndarray,
    *,
    bandwidth: float,
    block_size: int = 512,
) -> np.ndarray:
    output = np.empty_like(x_predict, dtype=np.float64)
    for start in range(0, x_predict.size, block_size):
        stop = min(start + block_size, x_predict.size)
        distance = (x_predict[start:stop, None] - x_train[None, :]) / bandwidth
        log_weights = -0.5 * distance**2
        log_weights -= np.max(log_weights, axis=1, keepdims=True)
        weights = np.exp(log_weights)
        output[start:stop] = (weights @ y_train) / weights.sum(axis=1)
    return output


def _mad_outliers(residual: np.ndarray, *, threshold: float) -> np.ndarray:
    center = float(np.median(residual))
    scale = 1.4826 * float(np.median(np.abs(residual - center)))
    if not np.isfinite(scale) or scale <= np.finfo(np.float64).eps:
        return np.zeros(residual.shape, dtype=bool)
    return np.abs(residual - center) > threshold * scale


def regularize_parameters(
    *,
    gene_mean: np.ndarray,
    geometric_mean: np.ndarray,
    raw_intercept: np.ndarray,
    raw_theta: np.ndarray,
    training_mask: np.ndarray,
    prediction_mask: np.ndarray,
    bandwidth_adjust: float,
    outlier_mad_threshold: float,
    theta_min: float,
    theta_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    train_indices = np.flatnonzero(training_mask)
    predict_indices = np.flatnonzero(prediction_mask)
    if train_indices.size == 0:
        raise RuntimeError("no converged non-Poisson genes are available to regularize")

    x_train = np.log10(np.maximum(geometric_mean[train_indices], np.finfo(float).tiny))
    intercept_train = raw_intercept[train_indices]
    od_train = np.log10(gene_mean[train_indices] / raw_theta[train_indices])
    bandwidth = robust_bandwidth(x_train, adjust=bandwidth_adjust)

    outlier = np.zeros(raw_intercept.shape, dtype=bool)
    if train_indices.size >= 4:
        initial_intercept = gaussian_nadaraya_watson(
            x_train, intercept_train, x_train, bandwidth=bandwidth
        )
        initial_od = gaussian_nadaraya_watson(
            x_train, od_train, x_train, bandwidth=bandwidth
        )
        local_outlier = _mad_outliers(
            intercept_train - initial_intercept, threshold=outlier_mad_threshold
        ) | _mad_outliers(od_train - initial_od, threshold=outlier_mad_threshold)
        if np.count_nonzero(~local_outlier) >= 3:
            outlier[train_indices[local_outlier]] = True
            x_train = x_train[~local_outlier]
            intercept_train = intercept_train[~local_outlier]
            od_train = od_train[~local_outlier]
            bandwidth = robust_bandwidth(x_train, adjust=bandwidth_adjust)

    x_predict = np.log10(
        np.maximum(geometric_mean[predict_indices], np.finfo(float).tiny)
    )
    regularized_intercept = np.full(raw_intercept.shape, np.nan, dtype=np.float64)
    regularized_theta = np.full(raw_theta.shape, np.nan, dtype=np.float64)
    regularized_intercept[predict_indices] = gaussian_nadaraya_watson(
        x_train, intercept_train, x_predict, bandwidth=bandwidth
    )
    od_predict = gaussian_nadaraya_watson(
        x_train, od_train, x_predict, bandwidth=bandwidth
    )
    theta = gene_mean[predict_indices] / np.power(10.0, od_predict)
    regularized_theta[predict_indices] = np.clip(theta, theta_min, theta_max)
    return regularized_intercept, regularized_theta, outlier, bandwidth
