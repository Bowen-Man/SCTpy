from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import digamma, gammaln

FitMethod = Literal["nb_mle", "cox_reid"]
ThetaPostprocess = Literal["none", "moment_ratio"]
FitTrace = Callable[[dict[str, float | int | str | bool]], None]

_COX_REID_CORRECTION = 0.99
_DISPERSION_FLOOR = 1.0e-8
_DISPERSION_CEILING = 1.0e16


@dataclass(frozen=True)
class GeneFit:
    intercept: float
    theta: float
    converged: bool
    objective: float
    iterations: int
    message: str


def _emit_trace(
    trace: FitTrace | None,
    stage: str,
    event: str,
    **values: float | int | str | bool,
) -> None:
    if trace is not None:
        trace({"stage": stage, "event": event, **values})


def theta_moment_ratio_mask(
    gene_mean: np.ndarray,
    gene_variance: np.ndarray,
    theta: np.ndarray,
    *,
    threshold: float,
) -> np.ndarray:
    """Flag theta estimates far above their method-of-moments counterpart."""
    gene_mean = np.asarray(gene_mean, dtype=np.float64)
    gene_variance = np.asarray(gene_variance, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    if gene_mean.shape != gene_variance.shape or gene_mean.shape != theta.shape:
        raise ValueError("mean, variance, and theta must have identical shapes")
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("threshold must be finite and positive")

    excess_variance = gene_variance - gene_mean
    theta_moment = np.full(gene_mean.shape, np.inf, dtype=np.float64)
    np.divide(
        gene_mean**2,
        excess_variance,
        out=theta_moment,
        where=excess_variance > 0.0,
    )
    ratio = np.full(gene_mean.shape, np.inf, dtype=np.float64)
    valid_theta = np.isfinite(theta) & (theta > 0.0)
    np.divide(theta_moment, theta, out=ratio, where=valid_theta)
    return ratio < threshold


def poisson_offset_intercept(y: np.ndarray, exposure: np.ndarray) -> float:
    total = float(y.sum())
    if total == 0.0:
        return -np.inf
    return float(np.log(total / exposure.sum()))


def _negative_log_likelihood_and_gradient(
    parameters: np.ndarray,
    y: np.ndarray,
    log_exposure: np.ndarray,
) -> tuple[float, np.ndarray]:
    intercept, log_theta = parameters
    theta = float(np.exp(log_theta))
    log_mu = intercept + log_exposure
    mu = np.exp(log_mu)
    log_mu_plus_theta = np.logaddexp(log_mu, log_theta)

    log_likelihood = (
        gammaln(y + theta)
        - gammaln(theta)
        - gammaln(y + 1.0)
        + theta * log_theta
        + y * log_mu
        - (y + theta) * log_mu_plus_theta
    )
    score_intercept = np.sum(theta * (y - mu) / (mu + theta))
    score_theta = np.sum(
        digamma(y + theta)
        - digamma(theta)
        + log_theta
        + 1.0
        - log_mu_plus_theta
        - (y + theta) / (mu + theta)
    )
    objective = -float(np.sum(log_likelihood))
    gradient = -np.asarray([score_intercept, theta * score_theta], dtype=np.float64)
    return objective, gradient


def fit_nb_offset(
    y: np.ndarray,
    exposure: np.ndarray,
    *,
    theta_min: float,
    theta_max: float,
    maxiter: int,
    ftol: float,
    gtol: float,
) -> GeneFit:
    """Independently fit an NB2 intercept and gene-specific theta by MLE."""
    intercept = poisson_offset_intercept(y, exposure)
    mu = np.exp(intercept) * exposure
    denominator = float(np.dot(mu, mu))
    excess = float(np.sum((y - mu) ** 2 - y))
    if denominator > 0.0 and excess > 0.0:
        theta_start = denominator / excess
    else:
        theta_start = min(100.0, np.sqrt(theta_max))
    theta_start = float(np.clip(theta_start, theta_min, theta_max))
    initial = np.asarray([intercept, np.log(theta_start)], dtype=np.float64)

    result = minimize(
        _negative_log_likelihood_and_gradient,
        initial,
        args=(y, np.log(exposure)),
        method="L-BFGS-B",
        jac=True,
        bounds=((-50.0, 5.0), (np.log(theta_min), np.log(theta_max))),
        options={"maxiter": maxiter, "ftol": ftol, "gtol": gtol, "maxls": 50},
    )
    fitted_intercept = float(result.x[0])
    fitted_theta = float(np.exp(result.x[1]))
    finite = bool(
        np.isfinite(result.fun)
        and np.isfinite(fitted_intercept)
        and np.isfinite(fitted_theta)
    )
    return GeneFit(
        intercept=fitted_intercept,
        theta=fitted_theta,
        converged=bool(result.success and finite),
        objective=float(result.fun),
        iterations=int(result.nit),
        message=str(result.message),
    )


def offset_intercept(
    y: np.ndarray,
    exposure: np.ndarray,
    dispersion: float,
    *,
    trace: FitTrace | None = None,
    trace_stage: str = "offset_intercept",
) -> tuple[float, int]:
    """Fit one offset intercept for a fixed NB2 dispersion."""
    if float(np.sum(y)) == 0.0:
        _emit_trace(trace, trace_stage, "all_zero", intercept=-np.inf)
        return -np.inf, 0
    if dispersion == 0.0:
        intercept = poisson_offset_intercept(y, exposure)
        _emit_trace(trace, trace_stage, "poisson", intercept=intercept)
        return intercept, 0

    center = poisson_offset_intercept(y, exposure)
    lower = center - 30.0
    upper = center + 30.0
    _emit_trace(
        trace,
        trace_stage,
        "initialization",
        dispersion=dispersion,
        center=center,
        lower=lower,
        upper=upper,
    )

    def score(intercept: float) -> float:
        mu = np.exp(intercept) * exposure
        value = float(np.sum((y - mu) / (1.0 + dispersion * mu)))
        _emit_trace(
            trace,
            trace_stage,
            "score_evaluation",
            intercept=intercept,
            score=value,
        )
        return value

    root, result = brentq(
        score,
        lower,
        upper,
        xtol=1.0e-12,
        rtol=1.0e-14,
        full_output=True,
    )
    _emit_trace(
        trace,
        trace_stage,
        "result",
        intercept=float(root),
        iterations=int(result.iterations),
        function_calls=int(result.function_calls),
        converged=bool(result.converged),
        flag=str(result.flag),
    )
    return float(root), int(result.iterations)


def _cox_reid_score(log_dispersion: float, y: np.ndarray, mu: np.ndarray) -> float:
    dispersion = float(np.exp(log_dispersion))
    inverse_dispersion = 1.0 / dispersion
    scaled_mu = mu * dispersion

    likelihood_term = np.empty_like(scaled_mu)
    tiny = scaled_mu < 1.0e-10
    small = (scaled_mu >= 1.0e-10) & (scaled_mu < 1.0e-4)
    regular = ~(tiny | small)
    likelihood_term[tiny] = scaled_mu[tiny] ** 2 * (1.0 / (1.0 + scaled_mu[tiny]) - 0.5)
    suggestion = np.log(1.0 + scaled_mu[small]) - mu[small] / (
        mu[small] + inverse_dispersion
    )
    upper = scaled_mu[small] ** 2 / (1.0 + scaled_mu[small])
    lower = scaled_mu[small] ** 2 * (1.0 / (1.0 + scaled_mu[small]) - 0.5)
    likelihood_term[small] = np.clip(suggestion, lower, upper)
    likelihood_term[regular] = np.log(1.0 + scaled_mu[regular]) - mu[regular] / (
        mu[regular] + inverse_dispersion
    )
    likelihood_score = inverse_dispersion * np.sum(
        likelihood_term + y / (mu + inverse_dispersion), dtype=np.float64
    )

    count_sum = float(np.sum(y))
    correction = 0.0
    if inverse_dispersion > 1.0e5:
        correction = float(np.sum(y * (y - 1.0))) / (2.0 * inverse_dispersion)
    if float(np.max(y)) * 1.0e6 < inverse_dispersion:
        digamma_score = count_sum - correction
    else:
        digamma_score = float(
            np.sum(
                (digamma(y + inverse_dispersion) - digamma(inverse_dispersion))
                * inverse_dispersion,
                dtype=np.float64,
            )
        )
        digamma_score = min(digamma_score, count_sum - correction)

    weights = mu / (1.0 + dispersion * mu)
    adjustment = (
        0.5
        * dispersion
        * _COX_REID_CORRECTION
        * float(np.dot(weights, weights))
        / (float(np.sum(weights)) + 1.0e-6)
    )
    return float(likelihood_score - digamma_score + adjustment)


def fit_cox_reid_offset(
    y: np.ndarray, exposure: np.ndarray, *, trace: FitTrace | None = None
) -> GeneFit:
    """Estimate an offset NB2 model with Cox--Reid adjusted dispersion."""
    mean = float(np.mean(y))
    if mean == 0.0:
        return GeneFit(
            intercept=-np.inf,
            theta=np.inf,
            converged=True,
            objective=0.0,
            iterations=0,
            message="all-zero offset fit",
        )
    variance = float(np.var(y, ddof=1))
    initial_dispersion = max(
        (variance - mean / float(np.mean(exposure))) / mean**2,
        0.0,
    )
    _emit_trace(
        trace,
        "cox_reid",
        "initialization",
        mean=mean,
        variance=variance,
        initial_dispersion=initial_dispersion,
    )
    preliminary_intercept, beta_iterations = offset_intercept(
        y,
        exposure,
        initial_dispersion,
        trace=trace,
        trace_stage="preliminary_offset",
    )
    mu = np.exp(preliminary_intercept) * exposure

    if trace is None:
        score_function = _cox_reid_score
    else:

        def score_function(
            log_dispersion: float, values: np.ndarray, fitted_mean: np.ndarray
        ) -> float:
            value = _cox_reid_score(log_dispersion, values, fitted_mean)
            _emit_trace(
                trace,
                "cox_reid_dispersion",
                "score_evaluation",
                log_dispersion=log_dispersion,
                dispersion=float(np.exp(log_dispersion)),
                score=value,
            )
            return value

    lower = float(np.log(_DISPERSION_FLOOR))
    lower_score = score_function(lower, y, mu)
    dispersion_iterations = 0
    if lower_score < 0.0:
        dispersion = 0.0
    else:
        upper_dispersion = min(
            max(initial_dispersion, _DISPERSION_FLOOR * 4.0),
            _DISPERSION_CEILING,
        )
        upper_score = score_function(np.log(upper_dispersion), y, mu)
        while upper_score > 0.0 and upper_dispersion < _DISPERSION_CEILING:
            upper_dispersion = min(upper_dispersion * 4.0, _DISPERSION_CEILING)
            upper_score = score_function(np.log(upper_dispersion), y, mu)
            dispersion_iterations += 1
        if upper_score > 0.0:
            dispersion = _DISPERSION_CEILING
        else:
            root, result = brentq(
                score_function,
                lower,
                np.log(upper_dispersion),
                args=(y, mu),
                xtol=1.0e-10,
                rtol=1.0e-14,
                full_output=True,
            )
            dispersion = float(np.exp(root))
            dispersion_iterations += int(result.iterations)

    theta = np.inf if dispersion == 0.0 else 1.0 / dispersion
    if np.isfinite(theta):
        objective, _ = _negative_log_likelihood_and_gradient(
            np.asarray([preliminary_intercept, np.log(theta)]),
            y,
            np.log(exposure),
        )
    else:
        objective = float(
            np.sum(mu - y * np.log(mu) + gammaln(y + 1.0), dtype=np.float64)
        )
    _emit_trace(
        trace,
        "cox_reid",
        "result",
        preliminary_intercept=preliminary_intercept,
        theta=float(theta),
        dispersion=float(dispersion),
        beta_iterations=beta_iterations,
        dispersion_iterations=dispersion_iterations,
        objective=objective,
    )
    return GeneFit(
        intercept=preliminary_intercept,
        theta=float(theta),
        converged=True,
        objective=objective,
        iterations=beta_iterations + dispersion_iterations,
        message="Cox-Reid adjusted dispersion fit",
    )


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="stable")
    sorted_weights = weights[order]
    index = np.searchsorted(
        np.cumsum(sorted_weights), np.sum(sorted_weights) / 2.0, side="left"
    )
    return float(values[order[index]])


def local_median_dispersion_trend(
    abundance: np.ndarray,
    dispersion: np.ndarray,
    *,
    points: float | None = None,
    gene_bin_size: int | None = None,
) -> np.ndarray:
    """Fit the abundance-local weighted-median dispersion trend."""
    size = abundance.size
    if size == 0:
        return np.empty(0, dtype=np.float64)
    if gene_bin_size is not None:
        if gene_bin_size < 1:
            raise ValueError("gene_bin_size must be positive or None")
        if size > gene_bin_size:
            output = np.empty(size, dtype=np.float64)
            for start in range(0, size, gene_bin_size):
                stop = min(start + gene_bin_size, size)
                output[start:stop] = local_median_dispersion_trend(
                    abundance[start:stop],
                    dispersion[start:stop],
                    points=points,
                )
            return output
    points = max(100.0, 0.1 * size) if points is None else points
    points = min(size, max(1.0, points))
    half = int(np.floor(points / 2.0))
    order = np.argsort(abundance, kind="stable")
    sorted_dispersion = dispersion[order]
    trend = np.empty(size, dtype=np.float64)
    if size - half <= half:
        weights = np.exp(-0.5 * np.linspace(-3.0, 3.0, size) ** 2)
        trend.fill(_weighted_median(sorted_dispersion, weights))
    else:
        weights = np.exp(-0.5 * np.linspace(-3.0, 3.0, 2 * half + 1) ** 2)
        for center in range(half, size - half):
            trend[center] = _weighted_median(
                sorted_dispersion[center - half : center + half + 1], weights
            )
        trend[:half] = trend[half]
        trend[size - half :] = trend[size - half - 1]
    output = np.empty(size, dtype=np.float64)
    output[order] = trend
    return output
