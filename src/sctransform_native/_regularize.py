from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.optimize import brentq

# R's public ksmooth contract places Gaussian quartiles at +/- 0.25 * bandwidth.
# A standard normal has quartiles +/- 0.67448975..., hence sigma = bandwidth / this.
_NORMAL_BANDWIDTH_TO_SIGMA = 4.0 * 0.6744897501960817
_NORMAL_KERNEL_ROUGHNESS = 1.0 / (2.0 * np.sqrt(np.pi))
_LOG_10 = np.log(10.0)

DispersionRegularization = Literal["native", "sct_v2"]
RegularizationSupport = Literal["extrapolate", "clamp"]


def _normal_derivative_functional(
    x: np.ndarray,
    *,
    bandwidth: float,
    order: int,
    block_size: int = 256,
) -> float:
    """Evaluate the diagonal-in Gaussian derivative functional.

    This is the double-sum estimator defined by Sheather and Jones (1991),
    equations 12--13.  Blocking limits the temporary working set without
    changing the mathematical estimator.
    """
    if order not in (4, 6):
        raise ValueError("only fourth and sixth derivatives are supported")
    if not np.isfinite(bandwidth) or bandwidth <= 0.0:
        raise ValueError("bandwidth must be finite and positive")

    total = 0.0
    normalizer = 1.0 / np.sqrt(2.0 * np.pi)
    for start in range(0, x.size, block_size):
        stop = min(start + block_size, x.size)
        u = (x[start:stop, None] - x[None, :]) / bandwidth
        squared = u * u
        density = normalizer * np.exp(-0.5 * squared)
        if order == 4:
            polynomial = squared * squared - 6.0 * squared + 3.0
        else:
            polynomial = (
                squared * squared * squared
                - 15.0 * squared * squared
                + 45.0 * squared
                - 15.0
            )
        total += float(np.sum(polynomial * density, dtype=np.float64))

    denominator = x.size * (x.size - 1) * bandwidth ** (order + 1)
    value = total / denominator
    return value if order == 4 else -value


def sheather_jones_bandwidth(x: np.ndarray, *, adjust: float) -> float:
    """Solve the Sheather--Jones Gaussian-kernel bandwidth equation.

    The implementation follows the published solve-the-equation definition,
    including the diagonal terms.  It is intentionally expressed in NumPy
    rather than through an R-math compatibility layer.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("x must be one-dimensional")
    if x.size < 3:
        return robust_bandwidth(x, adjust=adjust)

    q25, q75 = np.percentile(x, [25.0, 75.0])
    interquartile_range = float(q75 - q25)
    if not np.isfinite(interquartile_range) or interquartile_range <= 0.0:
        return robust_bandwidth(x, adjust=adjust)

    n = float(x.size)
    pilot_a = 0.920 * interquartile_range * n ** (-1.0 / 7.0)
    pilot_b = 0.912 * interquartile_range * n ** (-1.0 / 9.0)
    second_derivative_roughness = _normal_derivative_functional(
        x, bandwidth=pilot_a, order=4
    )
    third_derivative_roughness = _normal_derivative_functional(
        x, bandwidth=pilot_b, order=6
    )
    if (
        not np.isfinite(second_derivative_roughness)
        or not np.isfinite(third_derivative_roughness)
        or second_derivative_roughness <= 0.0
        or third_derivative_roughness <= 0.0
    ):
        return robust_bandwidth(x, adjust=adjust)

    pilot_ratio = 1.357 * (
        second_derivative_roughness / third_derivative_roughness
    ) ** (1.0 / 7.0)

    def equation(bandwidth: float) -> float:
        alpha = pilot_ratio * bandwidth ** (5.0 / 7.0)
        roughness = _normal_derivative_functional(x, bandwidth=alpha, order=4)
        if not np.isfinite(roughness) or roughness <= 0.0:
            return np.inf
        estimate = (_NORMAL_KERNEL_ROUGHNESS / (n * roughness)) ** 0.2
        return float(estimate - bandwidth)

    initial = robust_bandwidth(x, adjust=1.0)
    lower = max(initial * 0.1, np.finfo(np.float64).eps)
    upper = initial * 2.0
    lower_value = equation(lower)
    upper_value = equation(upper)
    for _ in range(20):
        if np.signbit(lower_value) != np.signbit(upper_value):
            break
        if lower_value < 0.0:
            lower *= 0.5
            lower_value = equation(lower)
        else:
            upper *= 2.0
            upper_value = equation(upper)
    else:
        return initial * adjust

    root = brentq(equation, lower, upper, xtol=max(lower * 1.0e-6, 1.0e-12))
    return float(root * adjust)


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
    floor = max(float(np.max(np.abs(x))), 1.0) * np.sqrt(np.finfo(np.float64).eps)
    return max(float(bandwidth), floor)


def gaussian_nadaraya_watson(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_predict: np.ndarray,
    *,
    bandwidth: float,
    block_size: int = 512,
) -> np.ndarray:
    sigma = bandwidth / _NORMAL_BANDWIDTH_TO_SIGMA
    output = np.empty_like(x_predict, dtype=np.float64)
    for start in range(0, x_predict.size, block_size):
        stop = min(start + block_size, x_predict.size)
        distance = (x_predict[start:stop, None] - x_train[None, :]) / sigma
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


def _dispersion_target(
    gene_mean: np.ndarray,
    geometric_mean: np.ndarray,
    theta: np.ndarray,
    *,
    mode: DispersionRegularization,
) -> np.ndarray:
    if mode == "native":
        return np.log10(gene_mean / theta)
    if mode == "sct_v2":
        return np.log1p(geometric_mean / theta) / _LOG_10
    raise ValueError(f"unknown dispersion regularization mode: {mode}")


def _theta_from_dispersion(
    gene_mean: np.ndarray,
    geometric_mean: np.ndarray,
    dispersion: np.ndarray,
    *,
    mode: DispersionRegularization,
) -> np.ndarray:
    if mode == "native":
        return gene_mean / np.power(10.0, dispersion)
    if mode == "sct_v2":
        denominator = np.expm1(_LOG_10 * dispersion)
        theta = np.full(dispersion.shape, np.inf, dtype=np.float64)
        np.divide(geometric_mean, denominator, out=theta, where=denominator > 0.0)
        return theta
    raise ValueError(f"unknown dispersion regularization mode: {mode}")


def _regularization_prediction_points(
    x_train: np.ndarray,
    x_predict: np.ndarray,
    *,
    support: RegularizationSupport,
) -> np.ndarray:
    if support == "extrapolate":
        return x_predict
    if support == "clamp":
        return np.clip(x_predict, np.min(x_train), np.max(x_train))
    raise ValueError(f"unknown regularization support policy: {support}")


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
    dispersion_regularization: DispersionRegularization,
    regularization_support: RegularizationSupport,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    train_indices = np.flatnonzero(training_mask)
    predict_indices = np.flatnonzero(prediction_mask)
    if train_indices.size == 0:
        raise RuntimeError("no converged non-Poisson genes are available to regularize")

    x_train = np.log10(np.maximum(geometric_mean[train_indices], np.finfo(float).tiny))
    intercept_train = raw_intercept[train_indices]
    od_train = _dispersion_target(
        gene_mean[train_indices],
        geometric_mean[train_indices],
        raw_theta[train_indices],
        mode=dispersion_regularization,
    )
    bandwidth = sheather_jones_bandwidth(x_train, adjust=bandwidth_adjust)

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
            bandwidth = sheather_jones_bandwidth(x_train, adjust=bandwidth_adjust)

    x_predict = np.log10(
        np.maximum(geometric_mean[predict_indices], np.finfo(float).tiny)
    )
    x_predict = _regularization_prediction_points(
        x_train, x_predict, support=regularization_support
    )
    regularized_intercept = np.full(raw_intercept.shape, np.nan, dtype=np.float64)
    regularized_theta = np.full(raw_theta.shape, np.nan, dtype=np.float64)
    regularized_intercept[predict_indices] = gaussian_nadaraya_watson(
        x_train, intercept_train, x_predict, bandwidth=bandwidth
    )
    od_predict = gaussian_nadaraya_watson(
        x_train, od_train, x_predict, bandwidth=bandwidth
    )
    theta = _theta_from_dispersion(
        gene_mean[predict_indices],
        geometric_mean[predict_indices],
        od_predict,
        mode=dispersion_regularization,
    )
    regularized_theta[predict_indices] = np.clip(theta, theta_min, theta_max)
    return regularized_intercept, regularized_theta, outlier, bandwidth
