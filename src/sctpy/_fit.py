from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import digamma, gammaln, xlogy

FitMethod = Literal["nb_mle", "cox_reid"]
ThetaPostprocess = Literal["none", "moment_ratio"]

_COX_REID_CORRECTION = 0.99
_DISPERSION_FLOOR = 1.0e-8
_DISPERSION_CEILING = 1.0e16
_DISPERSION_ROOT_STEPS = 48
_LOG_DISPERSION_TOLERANCE = 1.0e-10


@dataclass(frozen=True)
class GeneFit:
    intercept: float
    theta: float
    converged: bool
    objective: float
    iterations: int
    message: str


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


def offset_negative_log_likelihood(
    y: np.ndarray, exposure: np.ndarray, intercept: float, theta: float
) -> float:
    """Unadjusted NB2 (or Poisson-limit) NLL at the supplied parameters."""
    if np.isnan(theta) or theta <= 0.0:
        return np.nan
    if np.isinf(theta):
        mu = np.exp(intercept) * exposure
        return float(np.sum(mu - xlogy(y, mu) + gammaln(y + 1.0)))
    objective, _ = _negative_log_likelihood_and_gradient(
        np.asarray([intercept, np.log(theta)]), y, np.log(exposure)
    )
    return objective


def _cox_reid_fit_result(
    y: np.ndarray,
    exposure: np.ndarray,
    intercept: float,
    theta: float,
    *,
    converged: bool,
    iterations: int,
    message: str,
) -> GeneFit:
    objective = offset_negative_log_likelihood(y, exposure, intercept, theta)
    finite = np.isfinite(intercept) and np.isfinite(objective) and theta > 0.0
    return GeneFit(
        intercept=intercept,
        theta=theta,
        converged=bool(converged and finite),
        objective=objective,
        iterations=iterations,
        message=message if finite else "non-finite Cox-Reid fit",
    )


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
) -> tuple[float, int]:
    """Fit one offset intercept for a fixed NB2 dispersion."""
    if float(np.sum(y)) == 0.0:
        return -np.inf, 0
    if dispersion == 0.0:
        intercept = poisson_offset_intercept(y, exposure)
        return intercept, 0

    center = poisson_offset_intercept(y, exposure)
    lower = center - 30.0
    upper = center + 30.0

    def score(intercept: float) -> float:
        mu = np.exp(intercept) * exposure
        value = float(np.sum((y - mu) / (1.0 + dispersion * mu)))
        return value

    root, result = brentq(
        score,
        lower,
        upper,
        xtol=1.0e-12,
        rtol=1.0e-14,
        full_output=True,
    )
    if not result.converged or not np.isfinite(root):
        raise RuntimeError("offset intercept did not converge")
    return float(root), int(result.iterations)


def _cox_reid_score(log_dispersion: float, y: np.ndarray, mu: np.ndarray) -> float:
    return float(
        _cox_reid_score_batch(
            np.asarray([log_dispersion]),
            np.asarray(y, dtype=np.float64)[None, :],
            np.asarray(mu, dtype=np.float64)[None, :],
        )[0]
    )


def _scaled_digamma_difference(
    dispersion: Any, counts: Any, *, xp: Any = np, psi: Any = digamma
) -> Any:
    """Return theta*(psi(theta+y)-psi(theta)) without small-phi cancellation.

    DLMF 5.11.2 and 5.11(ii): after B12, the two positive-real remainder
    bounds sum to at most phi**13/6. The switch limits this to float64 epsilon
    per count, independently of observed oracle agreement.
    """
    cutoff = (6.0 * np.finfo(np.float64).eps) ** (1.0 / 13.0)
    phi = xp.minimum(dispersion, cutoff)[:, None]
    log_ratio = xp.log1p(phi * counts)
    series = log_ratio / phi + 0.5 * (phi * counts) / (1.0 + phi * counts)
    # B_(2k)/(2k), k=1..6, from the Bernoulli-number expansion.
    coefficients = (1 / 12, -1 / 120, 1 / 252, -1 / 240, 1 / 132, -691 / 32760)
    for k, coefficient in enumerate(coefficients, start=1):
        power = 2 * k
        series -= coefficient * phi ** (power - 1) * xp.expm1(-power * log_ratio)
    theta = (1.0 / dispersion)[:, None]
    # Bound the unused branch too, avoiding cancellation at huge theta.
    direct_theta = xp.minimum(theta, 1.0 / cutoff)
    direct = direct_theta * (psi(direct_theta + counts) - psi(direct_theta))
    return xp.where(dispersion[:, None] <= cutoff, series, direct)


def _cox_reid_score_batch(
    log_dispersion: np.ndarray, y: np.ndarray, mu: np.ndarray
) -> np.ndarray:
    """Evaluate independent Cox--Reid scores in one float64 gene block."""
    dispersion = np.exp(np.asarray(log_dispersion, dtype=np.float64))
    inverse_dispersion = 1.0 / dispersion
    scaled_mu = mu * dispersion[:, None]

    likelihood_term = np.empty_like(scaled_mu)
    tiny = scaled_mu < 1.0e-10
    small = (scaled_mu >= 1.0e-10) & (scaled_mu < 1.0e-4)
    regular = ~(tiny | small)
    likelihood_term[tiny] = scaled_mu[tiny] ** 2 * (1.0 / (1.0 + scaled_mu[tiny]) - 0.5)
    inverse = np.broadcast_to(inverse_dispersion[:, None], mu.shape)
    suggestion = np.log1p(scaled_mu[small]) - mu[small] / (mu[small] + inverse[small])
    upper = scaled_mu[small] ** 2 / (1.0 + scaled_mu[small])
    lower = scaled_mu[small] ** 2 * (1.0 / (1.0 + scaled_mu[small]) - 0.5)
    likelihood_term[small] = np.clip(suggestion, lower, upper)
    likelihood_term[regular] = np.log1p(scaled_mu[regular]) - mu[regular] / (
        mu[regular] + inverse[regular]
    )
    likelihood_score = inverse_dispersion * np.sum(
        likelihood_term + y / (mu + inverse_dispersion[:, None]), axis=1
    )

    digamma_score = np.sum(_scaled_digamma_difference(dispersion, y), axis=1)

    weights = mu / (1.0 + dispersion[:, None] * mu)
    adjustment = (
        0.5
        * dispersion
        * _COX_REID_CORRECTION
        * np.sum(weights * weights, axis=1, dtype=np.float64)
        / (np.sum(weights, axis=1, dtype=np.float64) + 1.0e-6)
    )
    return likelihood_score - digamma_score + adjustment


def fit_cox_reid_offset_batch(y: np.ndarray, exposure: np.ndarray) -> list[GeneFit]:
    """Fit a dense gene block with batched safeguarded dispersion roots."""
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 2 or y.shape[1] != exposure.size:
        raise ValueError("batched counts must be genes by exposure-sized cells")
    means = np.mean(y, axis=1, dtype=np.float64)
    if np.any(means == 0.0):
        raise ValueError("batched Cox-Reid rows must contain at least one count")
    variances = np.var(y, axis=1, ddof=1, dtype=np.float64)
    initial_dispersion = np.maximum(
        (variances - means / float(np.mean(exposure))) / means**2, 0.0
    )
    preliminary = [
        offset_intercept(row, exposure, float(dispersion))
        for row, dispersion in zip(y, initial_dispersion, strict=True)
    ]
    intercept = np.asarray([value[0] for value in preliminary])
    intercept_iterations = np.asarray([value[1] for value in preliminary])
    mu = np.exp(intercept[:, None]) * exposure[None, :]

    lower = np.full(y.shape[0], np.log(_DISPERSION_FLOOR), dtype=np.float64)
    lower_score = _cox_reid_score_batch(lower, y, mu)
    valid = np.isfinite(lower_score)
    poisson = lower_score < 0.0
    upper_dispersion = np.minimum(
        np.maximum(initial_dispersion, _DISPERSION_FLOOR * 4.0),
        _DISPERSION_CEILING,
    )
    upper_score = _cox_reid_score_batch(np.log(upper_dispersion), y, mu)
    valid &= poisson | np.isfinite(upper_score)
    expansion_iterations = np.zeros(y.shape[0], dtype=np.int64)
    expanding = (
        valid
        & (~poisson)
        & (upper_score > 0.0)
        & (upper_dispersion < _DISPERSION_CEILING)
    )
    while np.any(expanding):
        upper_dispersion[expanding] = np.minimum(
            upper_dispersion[expanding] * 4.0, _DISPERSION_CEILING
        )
        expanding_indices = np.flatnonzero(expanding)
        upper_score[expanding_indices] = _cox_reid_score_batch(
            np.log(upper_dispersion[expanding_indices]),
            y[expanding_indices],
            mu[expanding_indices],
        )
        expansion_iterations[expanding_indices] += 1
        valid &= poisson | np.isfinite(upper_score)
        expanding = (
            valid
            & (~poisson)
            & (upper_score > 0.0)
            & (upper_dispersion < _DISPERSION_CEILING)
        )

    ceiling = (~poisson) & (upper_score > 0.0)
    bracketed = valid & ~(poisson | ceiling)
    upper = np.log(upper_dispersion)
    root_iterations = np.zeros(y.shape[0], dtype=np.int64)
    for _ in range(_DISPERSION_ROOT_STEPS):
        active_indices = np.flatnonzero(bracketed)
        if active_indices.size == 0:
            break
        midpoint = 0.5 * (lower[active_indices] + upper[active_indices])
        score = _cox_reid_score_batch(midpoint, y[active_indices], mu[active_indices])
        valid[active_indices] &= np.isfinite(score)
        bracketed &= valid
        positive = score > 0.0
        lower[active_indices[positive]] = midpoint[positive]
        upper[active_indices[~positive]] = midpoint[~positive]
        root_iterations[active_indices] += 1

    dispersion = np.exp(0.5 * (lower + upper))
    dispersion[poisson] = 0.0
    dispersion[ceiling] = _DISPERSION_CEILING
    dispersion[~valid] = np.nan
    theta = np.divide(
        1.0,
        dispersion,
        out=np.full_like(dispersion, np.inf),
        where=dispersion != 0.0,
    )
    output: list[GeneFit] = []
    for index in range(y.shape[0]):
        success = bool(
            valid[index]
            and (
                poisson[index]
                or (
                    bracketed[index]
                    and upper[index] - lower[index] <= _LOG_DISPERSION_TOLERANCE
                )
            )
        )
        if ceiling[index]:
            message = "Cox-Reid dispersion ceiling reached without a root"
        elif poisson[index]:
            message = "Cox-Reid Poisson boundary"
        else:
            message = (
                "Cox-Reid dispersion root"
                if success
                else "Cox-Reid root did not converge"
            )
        output.append(
            _cox_reid_fit_result(
                y[index],
                exposure,
                float(intercept[index]),
                float(theta[index]),
                converged=success,
                iterations=int(
                    intercept_iterations[index]
                    + expansion_iterations[index]
                    + root_iterations[index]
                ),
                message=message,
            )
        )
    return output


def fit_cox_reid_offset(y: np.ndarray, exposure: np.ndarray) -> GeneFit:
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
    preliminary_intercept, beta_iterations = offset_intercept(
        y,
        exposure,
        initial_dispersion,
    )
    mu = np.exp(preliminary_intercept) * exposure

    lower = float(np.log(_DISPERSION_FLOOR))
    lower_score = _cox_reid_score(lower, y, mu)
    dispersion_iterations = 0
    converged = False
    message = "non-finite Cox-Reid score"
    if not np.isfinite(lower_score):
        dispersion = np.nan
    elif lower_score < 0.0:
        dispersion = 0.0
        converged = True
        message = "Cox-Reid Poisson boundary"
    else:
        upper_dispersion = min(
            max(initial_dispersion, _DISPERSION_FLOOR * 4.0),
            _DISPERSION_CEILING,
        )
        upper_score = _cox_reid_score(np.log(upper_dispersion), y, mu)
        while (
            np.isfinite(upper_score)
            and upper_score > 0.0
            and upper_dispersion < _DISPERSION_CEILING
        ):
            upper_dispersion = min(upper_dispersion * 4.0, _DISPERSION_CEILING)
            upper_score = _cox_reid_score(np.log(upper_dispersion), y, mu)
            dispersion_iterations += 1
        if not np.isfinite(upper_score):
            dispersion = np.nan
        elif upper_score > 0.0:
            dispersion = _DISPERSION_CEILING
            message = "Cox-Reid dispersion ceiling reached without a root"
        else:
            try:
                root, result = brentq(
                    _cox_reid_score,
                    lower,
                    np.log(upper_dispersion),
                    args=(y, mu),
                    xtol=_LOG_DISPERSION_TOLERANCE,
                    rtol=1.0e-14,
                    full_output=True,
                    disp=False,
                )
            except ValueError:
                # A non-finite score encountered inside the bracket is a failed fit.
                dispersion = np.nan
            else:
                dispersion = float(np.exp(root))
                converged = bool(result.converged and np.isfinite(root))
                # SciPy's endpoint roots need no iterations.
                if lower_score != 0.0 and upper_score != 0.0:
                    dispersion_iterations += int(result.iterations)
                message = (
                    "Cox-Reid dispersion root"
                    if converged
                    else "Cox-Reid root did not converge"
                )

    theta = np.inf if dispersion == 0.0 else 1.0 / dispersion
    return _cox_reid_fit_result(
        y,
        exposure,
        preliminary_intercept,
        float(theta),
        converged=converged,
        iterations=beta_iterations + dispersion_iterations,
        message=message,
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
