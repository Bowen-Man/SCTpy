from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import digamma, gammaln


@dataclass(frozen=True)
class GeneFit:
    intercept: float
    theta: float
    converged: bool
    objective: float
    iterations: int
    message: str


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
    gradient = -np.asarray(
        [score_intercept, theta * score_theta], dtype=np.float64
    )
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
